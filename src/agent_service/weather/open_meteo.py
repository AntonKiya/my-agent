import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from agent_service.config import AppSettings
from agent_service.observability.events import elapsed_ms, log_event, start_timer

logger = logging.getLogger(__name__)

WEATHER_FORECAST_SKILL_ID = "weather-forecast"
WEATHER_FORECAST_TOOL_NAME = "get_weather_forecast"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WeatherPeriod = Literal["now", "today", "tomorrow", "week"]
LocationLanguage = Literal["ru", "en"]

GEOCODING_RESULT_COUNT = 5
AMBIGUOUS_CANDIDATE_LIMIT = 3
CONFIDENT_POPULATION_MIN = 100_000
CONFIDENT_POPULATION_MULTIPLIER = 5
HTTP_ERROR_MESSAGE_LIMIT = 500

CURRENT_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
)
HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
)
DAILY_VARIABLES = (
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_probability_max",
    "precipitation_sum",
    "weather_code",
    "wind_speed_10m_max",
)

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

WEATHER_CODE_DESCRIPTIONS_EN = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

WEATHER_CODE_DESCRIPTIONS_RU = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозь и туман",
    51: "слабая морось",
    53: "умеренная морось",
    55: "сильная морось",
    56: "слабая ледяная морось",
    57: "сильная ледяная морось",
    61: "небольшой дождь",
    63: "умеренный дождь",
    65: "сильный дождь",
    66: "слабый ледяной дождь",
    67: "сильный ледяной дождь",
    71: "небольшой снег",
    73: "умеренный снег",
    75: "сильный снег",
    77: "снежные зерна",
    80: "небольшие ливни",
    81: "умеренные ливни",
    82: "сильные ливни",
    85: "небольшие снежные заряды",
    86: "сильные снежные заряды",
    95: "гроза",
    96: "гроза с небольшим градом",
    99: "гроза с сильным градом",
}


@dataclass(frozen=True, slots=True)
class GeocodingAttempt:
    query: str
    location_language: LocationLanguage


@dataclass(frozen=True, slots=True)
class LocationResolution:
    status: Literal["ok", "needs_location", "location_not_found", "ambiguous_location"]
    location: dict[str, Any] | None = None
    candidates: tuple[dict[str, Any], ...] = ()
    query: str | None = None
    location_language: LocationLanguage | None = None
    message: str | None = None


@dataclass(slots=True)
class OpenMeteoWeatherClient:
    http_client: httpx.AsyncClient
    geocoding_url: str = OPEN_METEO_GEOCODING_URL
    forecast_url: str = OPEN_METEO_FORECAST_URL

    async def forecast(
        self,
        *,
        location: str,
        period: WeatherPeriod = "week",
        location_language: LocationLanguage = "ru",
    ) -> dict[str, Any]:
        clean_location = location.strip()
        if not clean_location:
            return {
                "status": "needs_location",
                "message": "Ask the user which city or place they want the weather for.",
            }

        resolution = await self.resolve_location(
            clean_location,
            location_language=location_language,
        )
        if resolution.status != "ok" or resolution.location is None:
            return _resolution_payload(resolution)

        started_at = start_timer()
        try:
            payload = await self._fetch_forecast(resolution.location, period=period)
        except httpx.HTTPError as exc:
            log_event(
                logger,
                logging.WARNING,
                "Open-Meteo forecast request failed",
                event="open_meteo_forecast_failed",
                location_query=clean_location,
                period=period,
                duration_ms=elapsed_ms(started_at),
                error_type=type(exc).__name__,
                error_message=_safe_error_message(exc),
            )
            return {
                "status": "temporarily_unavailable",
                "message": "Open-Meteo forecast is temporarily unavailable.",
            }

        log_event(
            logger,
            logging.INFO,
            "Open-Meteo forecast request completed",
            event="open_meteo_forecast_completed",
            location_query=clean_location,
            resolved_location=resolution.location.get("name"),
            period=period,
            duration_ms=elapsed_ms(started_at),
        )
        return _compact_forecast_payload(
            payload,
            location=resolution.location,
            period=period,
            location_language=location_language,
        )

    async def resolve_location(
        self,
        location: str,
        *,
        location_language: LocationLanguage,
    ) -> LocationResolution:
        attempts = _geocoding_attempts(location, location_language)
        last_attempt: GeocodingAttempt | None = None
        for attempt in attempts:
            last_attempt = attempt
            try:
                candidates = await self._fetch_geocoding_candidates(attempt)
            except httpx.HTTPError as exc:
                log_event(
                    logger,
                    logging.WARNING,
                    "Open-Meteo geocoding request failed",
                    event="open_meteo_geocoding_failed",
                    location_query=location,
                    geocoding_query=attempt.query,
                    geocoding_location_language=attempt.location_language,
                    error_type=type(exc).__name__,
                    error_message=_safe_error_message(exc),
                )
                continue
            if not candidates:
                continue

            selected = _select_confident_location(candidates)
            if selected is not None:
                return LocationResolution(
                    status="ok",
                    location=selected,
                    candidates=tuple(candidates),
                    query=attempt.query,
                    location_language=attempt.location_language,
                )
            return LocationResolution(
                status="ambiguous_location",
                candidates=tuple(candidates[:AMBIGUOUS_CANDIDATE_LIMIT]),
                query=attempt.query,
                location_language=attempt.location_language,
                message="Ask the user which matching place they mean.",
            )

        return LocationResolution(
            status="location_not_found",
            query=last_attempt.query if last_attempt is not None else location,
            location_language=last_attempt.location_language
            if last_attempt is not None
            else location_language,
            message="Ask the user to clarify the city or place name.",
        )

    async def _fetch_geocoding_candidates(
        self,
        attempt: GeocodingAttempt,
    ) -> list[dict[str, Any]]:
        response = await self.http_client.get(
            self.geocoding_url,
            params={
                "name": attempt.query,
                "count": GEOCODING_RESULT_COUNT,
                "language": attempt.location_language,
                "format": "json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        mapping = _as_mapping(payload)
        if mapping is None:
            return []
        results = mapping.get("results")
        if not isinstance(results, list):
            return []
        candidates = []
        for item in results:
            candidate = _compact_location_candidate(item)
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    async def _fetch_forecast(
        self,
        location: Mapping[str, Any],
        *,
        period: WeatherPeriod,
    ) -> Mapping[str, Any]:
        response = await self.http_client.get(
            self.forecast_url,
            params={
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "current": ",".join(CURRENT_VARIABLES),
                "hourly": ",".join(HOURLY_VARIABLES),
                "daily": ",".join(DAILY_VARIABLES),
                "forecast_days": _forecast_days(period),
                "timezone": "auto",
            },
        )
        response.raise_for_status()
        payload = response.json()
        mapping = _as_mapping(payload)
        if mapping is None:
            return {}
        return mapping


def build_weather_forecast_toolsets(
    settings: AppSettings,
    *,
    http_client: httpx.AsyncClient,
) -> tuple[AgentToolset[Any], ...]:
    if not settings.weather_forecast_enabled:
        return ()

    client = OpenMeteoWeatherClient(http_client=http_client)

    async def get_weather_forecast(
        location: str,
        period: WeatherPeriod = "week",
        location_language: LocationLanguage = "ru",
    ) -> dict[str, Any]:
        """Get current weather or a forecast for a city/place.

        Args:
            location: City or place name from the user's message or clear conversation context.
            period: Forecast period. Use now, today, tomorrow, or week.
            location_language: Language used to geocode the location. Use ru or en.
        """
        return await client.forecast(
            location=location,
            period=period,
            location_language=location_language,
        )

    return (
        FunctionToolset(
            [get_weather_forecast],
            id=WEATHER_FORECAST_SKILL_ID,
            timeout=settings.weather_forecast_tool_timeout_seconds,
            require_parameter_descriptions=True,
        ),
    )


def _geocoding_attempts(
    location: str,
    location_language: LocationLanguage,
) -> tuple[GeocodingAttempt, ...]:
    attempts = [
        GeocodingAttempt(query=location, location_language=item)
        for item in _geocoding_languages(location, location_language)
    ]

    return tuple(_unique_attempts(attempts))


def _geocoding_languages(
    location: str,
    location_language: LocationLanguage,
) -> tuple[LocationLanguage, ...]:
    if CYRILLIC_RE.search(location):
        return _unique_location_languages((location_language, "ru", "en"))
    return _unique_location_languages((location_language, "en"))


def _unique_attempts(attempts: Sequence[GeocodingAttempt]) -> tuple[GeocodingAttempt, ...]:
    seen: set[tuple[str, str]] = set()
    unique = []
    for attempt in attempts:
        key = (attempt.query.casefold(), attempt.location_language)
        if key in seen:
            continue
        seen.add(key)
        unique.append(attempt)
    return tuple(unique)


def _unique_location_languages(
    values: Sequence[LocationLanguage],
) -> tuple[LocationLanguage, ...]:
    seen: set[LocationLanguage] = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return tuple(unique)


def _select_confident_location(candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    ranked = sorted(candidates, key=_location_score, reverse=True)
    best = ranked[0]
    runner_up = ranked[1]
    best_population = _population(best)
    runner_up_population = _population(runner_up)
    if best_population < CONFIDENT_POPULATION_MIN:
        return None
    if best_population >= runner_up_population * CONFIDENT_POPULATION_MULTIPLIER:
        return best
    if best_population - runner_up_population >= CONFIDENT_POPULATION_MIN:
        return best
    return None


def _location_score(candidate: Mapping[str, Any]) -> float:
    feature_score = {
        "PPLC": 1_000.0,
        "PPLA": 800.0,
        "PPLA2": 600.0,
        "PPLA3": 400.0,
        "PPL": 100.0,
    }.get(_optional_str(candidate.get("feature_code")) or "", 0.0)
    population = _population(candidate)
    return feature_score + math.log10(max(population, 1))


def _population(candidate: Mapping[str, Any]) -> int:
    value = candidate.get("population")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _compact_location_candidate(value: Any) -> dict[str, Any] | None:
    mapping = _as_mapping(value)
    if mapping is None:
        return None
    name = _optional_str(mapping.get("name"))
    latitude = _number(mapping.get("latitude"))
    longitude = _number(mapping.get("longitude"))
    if name is None or latitude is None or longitude is None:
        return None

    candidate: dict[str, Any] = {
        "name": name,
        "latitude": latitude,
        "longitude": longitude,
    }
    for key in (
        "country",
        "country_code",
        "admin1",
        "admin2",
        "timezone",
        "feature_code",
        "population",
    ):
        value = mapping.get(key)
        if value is not None:
            candidate[key] = value
    return candidate


def _resolution_payload(resolution: LocationResolution) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": resolution.status,
    }
    if resolution.message is not None:
        payload["message"] = resolution.message
    if resolution.query is not None:
        payload["query"] = resolution.query
    if resolution.location_language is not None:
        payload["location_language"] = resolution.location_language
    if resolution.candidates:
        payload["candidates"] = [_public_location(candidate) for candidate in resolution.candidates]
    return payload


def _compact_forecast_payload(
    payload: Mapping[str, Any],
    *,
    location: Mapping[str, Any],
    period: WeatherPeriod,
    location_language: LocationLanguage,
) -> dict[str, Any]:
    daily_items = _compact_daily_items(
        _as_mapping(payload.get("daily")),
        location_language=location_language,
    )
    hourly_items = _compact_hourly_items(
        _as_mapping(payload.get("hourly")),
        location_language=location_language,
    )
    selected_daily = _select_daily_items(daily_items, period=period)
    selected_hourly = _select_hourly_items(
        hourly_items,
        daily_items=daily_items,
        period=period,
    )
    return {
        "status": "ok",
        "period": period,
        "location": _public_location(location),
        "timezone": _optional_str(payload.get("timezone")),
        "current": _compact_current(
            _as_mapping(payload.get("current")),
            location_language=location_language,
        ),
        "daily": selected_daily,
        "hourly": selected_hourly,
        "units": _compact_units(payload),
    }


def _public_location(location: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in location.items()
        if key
        in {
            "name",
            "country",
            "country_code",
            "admin1",
            "admin2",
            "timezone",
            "feature_code",
            "population",
        }
    }


def _compact_current(
    current: Mapping[str, Any] | None,
    *,
    location_language: LocationLanguage,
) -> dict[str, Any] | None:
    if current is None:
        return None
    weather_code = _weather_code(current.get("weather_code"))
    return _drop_none(
        {
            "time": _optional_str(current.get("time")),
            "temperature_2m": _number(current.get("temperature_2m")),
            "relative_humidity_2m": _number(current.get("relative_humidity_2m")),
            "apparent_temperature": _number(current.get("apparent_temperature")),
            "precipitation": _number(current.get("precipitation")),
            "weather_code": weather_code,
            "weather_description": _weather_description(weather_code, location_language),
            "wind_speed_10m": _number(current.get("wind_speed_10m")),
        }
    )


def _compact_daily_items(
    daily: Mapping[str, Any] | None,
    *,
    location_language: LocationLanguage,
) -> list[dict[str, Any]]:
    if daily is None:
        return []
    dates = _sequence_value(daily.get("time"))
    items = []
    for index, date in enumerate(dates):
        if not isinstance(date, str):
            continue
        weather_code = _weather_code(_list_item(daily, "weather_code", index))
        items.append(
            _drop_none(
                {
                    "date": date,
                    "temperature_2m_max": _number(_list_item(daily, "temperature_2m_max", index)),
                    "temperature_2m_min": _number(_list_item(daily, "temperature_2m_min", index)),
                    "precipitation_probability_max": _number(
                        _list_item(daily, "precipitation_probability_max", index)
                    ),
                    "precipitation_sum": _number(_list_item(daily, "precipitation_sum", index)),
                    "weather_code": weather_code,
                    "weather_description": _weather_description(weather_code, location_language),
                    "wind_speed_10m_max": _number(_list_item(daily, "wind_speed_10m_max", index)),
                }
            )
        )
    return items


def _compact_hourly_items(
    hourly: Mapping[str, Any] | None,
    *,
    location_language: LocationLanguage,
) -> list[dict[str, Any]]:
    if hourly is None:
        return []
    times = _sequence_value(hourly.get("time"))
    items = []
    for index, time in enumerate(times):
        if not isinstance(time, str):
            continue
        weather_code = _weather_code(_list_item(hourly, "weather_code", index))
        items.append(
            _drop_none(
                {
                    "time": time,
                    "temperature_2m": _number(_list_item(hourly, "temperature_2m", index)),
                    "relative_humidity_2m": _number(
                        _list_item(hourly, "relative_humidity_2m", index)
                    ),
                    "apparent_temperature": _number(
                        _list_item(hourly, "apparent_temperature", index)
                    ),
                    "precipitation_probability": _number(
                        _list_item(hourly, "precipitation_probability", index)
                    ),
                    "precipitation": _number(_list_item(hourly, "precipitation", index)),
                    "weather_code": weather_code,
                    "weather_description": _weather_description(weather_code, location_language),
                    "wind_speed_10m": _number(_list_item(hourly, "wind_speed_10m", index)),
                }
            )
        )
    return items


def _select_daily_items(
    items: Sequence[dict[str, Any]],
    *,
    period: WeatherPeriod,
) -> list[dict[str, Any]]:
    if period in {"now", "today"}:
        return list(items[:1])
    if period == "tomorrow":
        return list(items[1:2])
    return list(items[:7])


def _select_hourly_items(
    items: Sequence[dict[str, Any]],
    *,
    daily_items: Sequence[dict[str, Any]],
    period: WeatherPeriod,
) -> list[dict[str, Any]]:
    if period == "week":
        target_dates = {
            date
            for item in daily_items[:7]
            if isinstance((date := item.get("date")), str)
        }
        return [item for item in items if _hourly_date(item) in target_dates]
    if period not in {"today", "tomorrow"}:
        return []
    index = 0 if period == "today" else 1
    if len(daily_items) <= index:
        return []
    target_date = daily_items[index].get("date")
    if not isinstance(target_date, str):
        return []
    return [item for item in items if _hourly_date(item) == target_date]


def _hourly_date(item: Mapping[str, Any]) -> str | None:
    time = item.get("time")
    if not isinstance(time, str):
        return None
    return time.split("T", 1)[0]


def _compact_units(payload: Mapping[str, Any]) -> dict[str, Any]:
    units: dict[str, Any] = {}
    for key in ("current_units", "hourly_units", "daily_units"):
        value = payload.get(key)
        if isinstance(value, dict):
            units[key] = {
                item_key: item_value
                for item_key, item_value in value.items()
                if isinstance(item_key, str) and isinstance(item_value, str)
            }
    return units


def _forecast_days(period: WeatherPeriod) -> int:
    if period == "tomorrow":
        return 2
    if period == "week":
        return 7
    return 1


def _weather_description(code: int | None, location_language: LocationLanguage) -> str | None:
    if code is None:
        return None
    if location_language == "ru":
        return WEATHER_CODE_DESCRIPTIONS_RU.get(code) or WEATHER_CODE_DESCRIPTIONS_EN.get(code)
    return WEATHER_CODE_DESCRIPTIONS_EN.get(code)


def _weather_code(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _list_item(mapping: Mapping[str, Any], key: str, index: int) -> Any:
    value = mapping.get(key)
    if not isinstance(value, list):
        return None
    if index >= len(value):
        return None
    return value[index]


def _sequence_value(value: Any) -> Sequence[Any]:
    if isinstance(value, list):
        return value
    return ()


def _drop_none(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return None


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _safe_error_message(exc: Exception) -> str | None:
    message = str(exc).strip()
    if not message:
        return None
    if len(message) <= HTTP_ERROR_MESSAGE_LIMIT:
        return message
    return f"{message[:HTTP_ERROR_MESSAGE_LIMIT]}..."
