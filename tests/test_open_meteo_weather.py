from typing import Any, cast

import httpx

from agent_service.config import AppSettings
from agent_service.weather import (
    WEATHER_FORECAST_SKILL_ID,
    build_weather_forecast_toolsets,
)
from agent_service.weather.open_meteo import OpenMeteoWeatherClient


def _geocoding_payload(*items: dict[str, object]) -> dict[str, object]:
    return {"results": list(items), "generationtime_ms": 1.0}


def _location(
    name: str,
    *,
    latitude: float,
    longitude: float,
    country: str,
    country_code: str,
    population: int,
    feature_code: str = "PPLA",
) -> dict[str, object]:
    return {
        "name": name,
        "latitude": latitude,
        "longitude": longitude,
        "country": country,
        "country_code": country_code,
        "admin1": name,
        "timezone": "Europe/Moscow",
        "population": population,
        "feature_code": feature_code,
    }


def _forecast_payload() -> dict[str, object]:
    dates = [
        "2026-06-04",
        "2026-06-05",
        "2026-06-06",
        "2026-06-07",
        "2026-06-08",
        "2026-06-09",
        "2026-06-10",
    ]
    return {
        "timezone": "Europe/Moscow",
        "current": {
            "time": "2026-06-04T12:00",
            "temperature_2m": 18.5,
            "relative_humidity_2m": 55,
            "apparent_temperature": 17.8,
            "precipitation": 0,
            "weather_code": 0,
            "wind_speed_10m": 12.2,
        },
        "current_units": {
            "temperature_2m": "°C",
            "wind_speed_10m": "km/h",
        },
        "daily": {
            "time": dates,
            "temperature_2m_max": [20, 21, 19, 18, 22, 24, 25],
            "temperature_2m_min": [10, 11, 12, 9, 13, 14, 15],
            "precipitation_probability_max": [10, 20, 40, 15, 5, 0, 25],
            "precipitation_sum": [0, 1.2, 3.0, 0, 0, 0, 2.1],
            "weather_code": [0, 2, 61, 3, 1, 0, 63],
            "wind_speed_10m_max": [16, 18, 20, 17, 12, 10, 15],
        },
        "daily_units": {
            "temperature_2m_max": "°C",
            "precipitation_probability_max": "%",
        },
        "hourly": {
            "time": [
                "2026-06-04T00:00",
                "2026-06-04T12:00",
                "2026-06-05T00:00",
                "2026-06-05T12:00",
            ],
            "temperature_2m": [10, 18, 11, 21],
            "relative_humidity_2m": [80, 55, 82, 60],
            "apparent_temperature": [9, 17, 10, 20],
            "precipitation_probability": [10, 15, 20, 25],
            "precipitation": [0, 0, 0.2, 1.0],
            "weather_code": [0, 0, 2, 61],
            "wind_speed_10m": [6, 12, 7, 14],
        },
        "hourly_units": {
            "temperature_2m": "°C",
            "precipitation_probability": "%",
        },
    }


async def test_weather_forecast_uses_model_location_language_before_fallback() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            assert request.url.params["name"] == "Москва"
            if request.url.params["language"] == "en":
                return httpx.Response(200, json={"generationtime_ms": 1.0})
            assert request.url.params["language"] == "ru"
            return httpx.Response(
                200,
                json=_geocoding_payload(
                    _location(
                        "Москва",
                        latitude=55.75204,
                        longitude=37.61781,
                        country="Россия",
                        country_code="RU",
                        population=10_381_222,
                        feature_code="PPLC",
                    )
                ),
            )
        assert request.url.host == "api.open-meteo.com"
        assert request.url.params["latitude"] == "55.75204"
        assert request.url.params["longitude"] == "37.61781"
        assert request.url.params["forecast_days"] == "7"
        return httpx.Response(200, json=_forecast_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        weather = OpenMeteoWeatherClient(http_client=http_client)

        result = await weather.forecast(
            location="Москва",
            period="week",
            location_language="en",
        )

    assert result["status"] == "ok"
    assert result["period"] == "week"
    assert result["location"]["name"] == "Москва"
    assert result["current"]["weather_description"] == "clear sky"
    assert len(result["daily"]) == 7
    assert [item["time"] for item in result["hourly"]] == [
        "2026-06-04T00:00",
        "2026-06-04T12:00",
        "2026-06-05T00:00",
        "2026-06-05T12:00",
    ]
    assert [request.url.params["language"] for request in requests[:2]] == ["en", "ru"]
    assert len(requests) == 3


async def test_weather_forecast_retries_english_for_latin_location() -> None:
    geocoding_languages: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "geocoding-api.open-meteo.com":
            geocoding_languages.append(request.url.params["language"])
            if request.url.params["language"] == "ru":
                return httpx.Response(200, json={"generationtime_ms": 1.0})
            return httpx.Response(
                200,
                json=_geocoding_payload(
                    _location(
                        "Moscow",
                        latitude=55.75204,
                        longitude=37.61781,
                        country="Russia",
                        country_code="RU",
                        population=10_381_222,
                        feature_code="PPLC",
                    )
                ),
            )
        return httpx.Response(200, json=_forecast_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        weather = OpenMeteoWeatherClient(http_client=http_client)

        result = await weather.forecast(
            location="Moscow",
            period="now",
            location_language="ru",
        )

    assert result["status"] == "ok"
    assert result["daily"][0]["date"] == "2026-06-04"
    assert result["hourly"] == []
    assert geocoding_languages == ["ru", "en"]


async def test_weather_forecast_does_not_canonicalize_petersburg() -> None:
    geocoding_names: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "geocoding-api.open-meteo.com":
            geocoding_names.append(request.url.params["name"])
            return httpx.Response(
                200,
                json=_geocoding_payload(
                    _location(
                        "Петербург",
                        latitude=37.22793,
                        longitude=-77.40193,
                        country="США",
                        country_code="US",
                        population=32_477,
                        feature_code="PPLA2",
                    ),
                    _location(
                        "Петербург",
                        latitude=38.99261,
                        longitude=-79.12392,
                        country="США",
                        country_code="US",
                        population=2_520,
                        feature_code="PPLA2",
                    ),
                ),
            )
        raise AssertionError("forecast should not be called for ambiguous Petersburg")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        weather = OpenMeteoWeatherClient(http_client=http_client)

        result = await weather.forecast(
            location="Петербург",
            period="tomorrow",
            location_language="ru",
        )

    assert result["status"] == "ambiguous_location"
    assert geocoding_names == ["Петербург"]


async def test_weather_forecast_returns_ambiguous_location_without_forecast_call() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == "geocoding-api.open-meteo.com"
        return httpx.Response(
            200,
            json=_geocoding_payload(
                _location(
                    "Springfield",
                    latitude=39.8017,
                    longitude=-89.6437,
                    country="United States",
                    country_code="US",
                    population=154_789,
                ),
                _location(
                    "Springfield",
                    latitude=42.1015,
                    longitude=-72.5898,
                    country="United States",
                    country_code="US",
                    population=116_250,
                ),
                _location(
                    "Springfield",
                    latitude=37.2153,
                    longitude=-93.2982,
                    country="United States",
                    country_code="US",
                    population=169_176,
                    feature_code="PPL",
                ),
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        weather = OpenMeteoWeatherClient(http_client=http_client)

        result = await weather.forecast(
            location="Springfield",
            period="week",
            location_language="en",
        )

    assert result["status"] == "ambiguous_location"
    assert len(result["candidates"]) == 3
    assert len(requests) == 1


async def test_build_weather_forecast_toolsets_can_be_disabled() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(500))
    ) as client:
        toolsets = build_weather_forecast_toolsets(
            AppSettings(environment="test", weather_forecast_enabled=False),
            http_client=client,
        )

    assert toolsets == ()


async def test_build_weather_forecast_toolsets_has_expected_id() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(500))
    ) as client:
        toolsets = build_weather_forecast_toolsets(
            AppSettings(environment="test"),
            http_client=client,
        )

    assert len(toolsets) == 1
    assert cast(Any, toolsets[0]).id == WEATHER_FORECAST_SKILL_ID
    assert set(cast(Any, toolsets[0]).tools) == {"get_weather_forecast"}
