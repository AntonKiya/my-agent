import copy
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4

from pydantic_ai import RunContext
from pydantic_ai.tools import ToolDefinition

from agent_service.mcp.toolsets import (
    ToolCallTransformer,
    ToolCallTransformResult,
    ToolDefinitionTransformer,
    ToolResultTransformContext,
)
from agent_service.tool_refs import ToolResultReference, ToolResultReferenceStore

TUTU_PROVIDER = "tutu"
TUTU_SELECTION_ID_PREFIX = "sel_"
TUTU_REFERENCE_TTL = timedelta(hours=6)
TUTU_CHECKOUT_SELECTION_EXTRA_FIELDS = frozenset(
    {
        "car_number",
        "fare_type",
        "gender_type",
        "kind",
        "offer_pack_hash",
        "seat_numbers",
    }
)
SelectionIdProvider = Callable[[], str]
DateTimeProvider = Callable[[], datetime]
TutuContextualResultTransformer = Callable[
    [ToolResultTransformContext, Any],
    Any | Awaitable[Any],
]


def build_tutu_search_result_transformer(
    reference_store: ToolResultReferenceStore,
    *,
    selection_id_provider: SelectionIdProvider | None = None,
    now_provider: DateTimeProvider | None = None,
    ttl: timedelta = TUTU_REFERENCE_TTL,
) -> TutuContextualResultTransformer:
    selection_ids = selection_id_provider or _new_selection_id
    current_time = now_provider or (lambda: datetime.now(UTC))

    async def transform(context: ToolResultTransformContext, value: Any) -> Any:
        return await _compact_json_like(
            value,
            lambda payload: _compact_tutu_search_payload(
                payload,
                context=context,
                reference_store=reference_store,
                selection_id_provider=selection_ids,
                now=current_time(),
                ttl=ttl,
            ),
        )

    return transform


def build_tutu_checkout_link_call_transformer(
    reference_store: ToolResultReferenceStore,
) -> ToolCallTransformer:
    async def transform(
        name: str,
        tool_args: Mapping[str, Any],
        ctx: RunContext[Any],
    ) -> ToolCallTransformResult | None:
        selection_id = tool_args.get("selection_id")
        if not isinstance(selection_id, str) or not selection_id.strip():
            return None

        owner = _owner_from_context(ctx)
        if owner is None:
            return ToolCallTransformResult(
                preflight_result=_selection_error_result(
                    name,
                    code="selection_context_missing",
                    message="selection_id cannot be resolved without user_id and conversation_id",
                )
            )

        user_id, conversation_id = owner
        reference = await reference_store.get(
            selection_id=selection_id,
            user_id=user_id,
            conversation_id=conversation_id,
            provider=TUTU_PROVIDER,
        )
        if reference is None:
            return ToolCallTransformResult(
                preflight_result=_selection_error_result(
                    name,
                    code="selection_not_found_or_expired",
                    message="selection_id was not found for this conversation or has expired",
                )
            )

        checkout_ref = reference.ref_payload.get("checkout_ref")
        if not isinstance(checkout_ref, dict):
            return ToolCallTransformResult(
                preflight_result=_selection_error_result(
                    name,
                    code="selection_missing_checkout_ref",
                    message="selection_id does not contain a checkout_ref",
                )
            )

        resolved_args = dict(checkout_ref)
        resolved_args.update(
            {
                key: value
                for key, value in tool_args.items()
                if key in TUTU_CHECKOUT_SELECTION_EXTRA_FIELDS and value is not None
            }
        )
        return ToolCallTransformResult(tool_args=resolved_args)

    return transform


def add_selection_id_to_checkout_link_tool_definition(
    tool_definition: ToolDefinition,
) -> ToolDefinition:
    schema = copy.deepcopy(tool_definition.parameters_json_schema)
    properties = schema.setdefault("properties", {})
    if isinstance(properties, dict):
        properties.setdefault(
            "selection_id",
            {
                "type": "string",
                "description": (
                    "Opaque id returned by Tutu search results. Prefer this over raw "
                    "checkout_ref fields when present."
                ),
            },
        )
    description = tool_definition.description or ""
    selection_note = (
        "\n\nWhen a Tutu search result contains selection_id, call this tool with that "
        "exact selection_id. Do not reconstruct checkout_ref fields."
    )
    if "selection_id" not in description:
        description = f"{description}{selection_note}".strip()
    return replace(
        tool_definition,
        parameters_json_schema=schema,
        description=description,
    )


def tutu_checkout_link_definition_transformers(
    tool_name: str,
) -> dict[str, ToolDefinitionTransformer]:
    return {tool_name: add_selection_id_to_checkout_link_tool_definition}


async def _compact_json_like(
    value: Any,
    payload_compactor: Callable[[Any], Awaitable[dict[str, Any] | None]],
) -> Any:
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            return value
        compact_payload = await payload_compactor(payload)
        if compact_payload is None:
            return value
        return json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))

    compact_payload = await payload_compactor(value)
    if compact_payload is None:
        return value
    return compact_payload


async def _compact_tutu_search_payload(
    payload: Any,
    *,
    context: ToolResultTransformContext,
    reference_store: ToolResultReferenceStore,
    selection_id_provider: SelectionIdProvider,
    now: datetime,
    ttl: timedelta,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if payload.get("ok") is False or "error" in payload:
        return None

    item_key = _search_result_item_key(payload)
    if item_key is None:
        return None
    items = payload.get(item_key)
    if not isinstance(items, list):
        return None

    owner = _owner_from_context(context.run_context)
    if owner is None:
        return None
    user_id, conversation_id = owner

    compact_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        item_kind = _item_kind(item_key, item)
        selection_id = selection_id_provider()
        compact_item = _compact_search_item(item_key, item, selection_id=selection_id)
        if compact_item is None:
            continue

        await reference_store.create(
            reference=ToolResultReference(
                selection_id=selection_id,
                provider=TUTU_PROVIDER,
                source_tool_name=context.tool_name,
                user_id=user_id,
                conversation_id=conversation_id,
                item_kind=item_kind,
                item_index=index,
                label=_item_label(item_key, item),
                display_snapshot=compact_item,
                ref_payload=_reference_payload(item),
                expires_at=now + ttl,
                created_at=now,
            )
        )
        compact_items.append(compact_item)

    if not compact_items:
        return None

    return cast(
        dict[str, Any],
        _prune_compacted_value(
            {
                "meta": _compact_meta(payload.get("meta")),
                item_key: compact_items,
            }
        ),
    )


def _search_result_item_key(payload: Mapping[str, Any]) -> str | None:
    for key in ("offers", "variants", "hotels"):
        if isinstance(payload.get(key), list):
            return key
    return None


def _item_kind(item_key: str, item: Mapping[str, Any]) -> str:
    if item_key == "hotels":
        return "hotel"
    transport = item.get("transport")
    return transport if isinstance(transport, str) and transport else "transport"


def _compact_search_item(
    item_key: str,
    item: Mapping[str, Any],
    *,
    selection_id: str,
) -> dict[str, Any] | None:
    if item_key == "hotels":
        return _compact_hotel_item(item, selection_id=selection_id)
    return _compact_transport_item(item, selection_id=selection_id)


def _compact_transport_item(
    item: Mapping[str, Any],
    *,
    selection_id: str,
) -> dict[str, Any] | None:
    price = _object_value(item.get("price"))
    legs = _compact_legs(item.get("legs"))
    compact_item = {
        "selection_id": selection_id,
        "transport": item.get("transport"),
        "price": _compact_price(price),
        "departure_at": item.get("departure_at"),
        "arrival_at": item.get("arrival_at"),
        "duration_min": item.get("duration_min"),
        "segments_count": item.get("segments_count"),
        "carriers": item.get("carriers"),
        "route": _route_summary(legs),
        "legs": legs,
        "fares": _compact_fares(item.get("fares")),
        "fare_options": _compact_fare_options(item.get("variants")),
        "review_summary": _compact_review_summary(item.get("review_summary")),
    }
    compact_item = cast(dict[str, Any], _prune_compacted_value(compact_item))
    if "selection_id" not in compact_item:
        return None
    return compact_item


def _compact_hotel_item(
    item: Mapping[str, Any],
    *,
    selection_id: str,
) -> dict[str, Any] | None:
    best_offer = _object_value(item.get("best_offer"))
    compact_item = {
        "selection_id": selection_id,
        "name": item.get("name"),
        "stars": item.get("stars"),
        "rating": item.get("rating"),
        "review_count": item.get("review_count"),
        "address": item.get("address"),
        "location": item.get("location"),
        "hotel_id": item.get("hotel_id"),
        "hotel_geo_id": item.get("hotel_geo_id"),
        "alias": item.get("alias"),
        "photos_total": item.get("photos_total"),
        "review_summary": _compact_review_summary(item.get("review_summary")),
        "best_offer": _compact_best_hotel_offer(best_offer),
    }
    compact_item = cast(dict[str, Any], _prune_compacted_value(compact_item))
    if "selection_id" not in compact_item or "name" not in compact_item:
        return None
    return compact_item


def _compact_meta(value: Any) -> dict[str, Any] | None:
    meta = _object_value(value)
    if not meta:
        return None
    resolved_geo = _object_value(meta.get("resolved_geo"))
    return cast(
        dict[str, Any],
        _prune_compacted_value(
            {
                "transport": meta.get("transport"),
                "from": _compact_geo(meta.get("from")),
                "to": _compact_geo(meta.get("to")),
                "round_trip": meta.get("round_trip"),
                "return_date": meta.get("return_date"),
                "page": meta.get("page"),
                "page_size": meta.get("page_size"),
                "has_more": meta.get("has_more"),
                "total_returned": meta.get("total_returned"),
                "sort": meta.get("sort"),
                "geo_id": meta.get("geo_id"),
                "search_id": meta.get("search_id"),
                "resolved_geo": {
                    "name": resolved_geo.get("name"),
                    "geo_id": resolved_geo.get("geo_id"),
                    "region": resolved_geo.get("region"),
                    "country": resolved_geo.get("country"),
                    "geo_type": resolved_geo.get("geo_type"),
                    "hotels_count": resolved_geo.get("hotels_count"),
                },
                "unavailable": meta.get("unavailable"),
            }
        ),
    )


def _compact_geo(value: Any) -> dict[str, Any] | None:
    geo = _object_value(value)
    if not geo:
        return None
    return cast(
        dict[str, Any],
        _prune_compacted_value(
            {
                "name": geo.get("name"),
                "iata": geo.get("iata"),
                "geo_id": geo.get("geo_id"),
                "region": geo.get("region"),
            }
        ),
    )


def _compact_legs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    legs: list[dict[str, Any]] = []
    for leg in value:
        if not isinstance(leg, dict):
            continue
        segments = _compact_segments(leg.get("segments"))
        compact_leg = {
            "label": leg.get("label"),
            "from": leg.get("from"),
            "to": leg.get("to"),
            "departure_at": leg.get("departure_at"),
            "arrival_at": leg.get("arrival_at"),
            "duration_min": leg.get("duration_min"),
            "segments": segments,
        }
        legs.append(cast(dict[str, Any], _prune_compacted_value(compact_leg)))
    return legs


def _compact_segments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    segments: list[dict[str, Any]] = []
    for segment in value:
        if not isinstance(segment, dict):
            continue
        compact_segment = {
            "from": segment.get("from"),
            "to": segment.get("to"),
            "carrier": segment.get("carrier"),
            "voyage_no": segment.get("voyage_no"),
            "departure_at": segment.get("departure_at"),
            "arrival_at": segment.get("arrival_at"),
            "duration_min": segment.get("duration_min"),
            "vehicle_meta": segment.get("vehicle_meta"),
            "review_summary": _compact_review_summary(segment.get("review_summary")),
        }
        segments.append(cast(dict[str, Any], _prune_compacted_value(compact_segment)))
    return segments


def _route_summary(legs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not legs:
        return None
    first_leg = legs[0]
    last_leg = legs[-1]
    segments = first_leg.get("segments")
    if isinstance(segments, list) and segments:
        first_segment = segments[0]
    else:
        first_segment = first_leg
    last_segments = last_leg.get("segments")
    if isinstance(last_segments, list) and last_segments:
        last_segment = last_segments[-1]
    else:
        last_segment = last_leg
    if not isinstance(first_segment, Mapping) or not isinstance(last_segment, Mapping):
        return None
    return cast(
        dict[str, Any],
        _prune_compacted_value(
            {
                "from": first_segment.get("from"),
                "to": last_segment.get("to"),
                "departure_at": first_segment.get("departure_at"),
                "arrival_at": last_segment.get("arrival_at"),
            }
        ),
    )


def _compact_fares(value: Any) -> dict[str, Any] | None:
    fares = _object_value(value)
    if not fares:
        return None
    return cast(
        dict[str, Any],
        _prune_compacted_value(
            {
                "count": fares.get("count"),
                "price_from": fares.get("price_from"),
                "price_to": fares.get("price_to"),
                "currency": fares.get("currency"),
                "refundable_count": fares.get("refundable_count"),
                "changeable_count": fares.get("changeable_count"),
            }
        ),
    )


def _compact_fare_options(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    options: list[dict[str, Any]] = []
    for option in value[:5]:
        if not isinstance(option, dict):
            continue
        conditions = _object_value(option.get("conditions"))
        option_price = _object_value(option.get("price"))
        compact_option = {
            "price": _compact_price(option_price),
            "service_class": option.get("service_class"),
            "fare_family": conditions.get("fare_family"),
            "refundable": conditions.get("refundable"),
            "changeable": conditions.get("changeable"),
            "baggage": conditions.get("baggage"),
            "cabin_baggage": conditions.get("cabin_baggage"),
        }
        options.append(cast(dict[str, Any], _prune_compacted_value(compact_option)))
    return options


def _compact_best_hotel_offer(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if not value:
        return None
    compact_offer = {
        "price": _compact_price(_object_value(value.get("price"))),
        "room_name": value.get("room_name"),
        "meal_name": value.get("meal_name"),
        "breakfast_included": value.get("breakfast_included"),
        "free_cancellation": value.get("free_cancellation"),
        "pay_at_hotel": value.get("pay_at_hotel"),
        "pay_online": value.get("pay_online"),
        "room_size_sqm": value.get("room_size_sqm"),
        "highlights": _compact_highlights(value.get("highlights")),
    }
    return cast(dict[str, Any], _prune_compacted_value(compact_offer))


def _compact_highlights(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    highlights: list[dict[str, Any]] = []
    for highlight in value[:5]:
        if not isinstance(highlight, dict):
            continue
        highlights.append(
            cast(
                dict[str, Any],
                _prune_compacted_value(
                    {
                        "text": highlight.get("text"),
                        "type": highlight.get("type"),
                    }
                ),
            )
        )
    return highlights


def _compact_price(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if not value:
        return None
    return cast(
        dict[str, Any],
        _prune_compacted_value(
            {
                "amount": value.get("amount"),
                "currency": value.get("currency"),
            }
        ),
    )


def _compact_review_summary(value: Any) -> dict[str, Any] | None:
    review = _object_value(value)
    if not review:
        return None
    return cast(
        dict[str, Any],
        _prune_compacted_value(
            {
                "label": review.get("label"),
                "rating": review.get("rating"),
                "scale": review.get("scale"),
                "review_count": review.get("review_count"),
                "subject": review.get("subject"),
                "scope": review.get("scope"),
            }
        ),
    )


def _reference_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkout_ref": item.get("checkout_ref"),
        "details_ref": item.get("details_ref"),
        "checkout_url": item.get("checkout_url"),
        "search_results_url": item.get("search_results_url"),
        "raw_item": dict(item),
    }


def _item_label(item_key: str, item: Mapping[str, Any]) -> str | None:
    if item_key == "hotels":
        name = item.get("name")
        return name if isinstance(name, str) else None
    transport = item.get("transport")
    departure_at = item.get("departure_at")
    price = _object_value(item.get("price"))
    amount = price.get("amount")
    route = _route_summary(_compact_legs(item.get("legs")))
    if route:
        return " | ".join(
            str(part)
            for part in (
                transport,
                route.get("from"),
                route.get("to"),
                departure_at,
                amount,
            )
            if part not in (None, "")
        )
    return None


def _owner_from_context(ctx: RunContext[Any]) -> tuple[UUID, UUID] | None:
    deps = getattr(ctx, "deps", None)
    user_id = _uuid_from_context_value(_context_value(deps, "user_id"))
    conversation_id = _uuid_from_context_value(_context_value(deps, "conversation_id"))
    if user_id is None or conversation_id is None:
        return None
    return user_id, conversation_id


def _context_value(deps: object, key: str) -> object:
    if isinstance(deps, Mapping):
        return deps.get(key)
    return getattr(deps, key, None)


def _uuid_from_context_value(value: object) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            return None
    return None


def _selection_error_result(tool_name: str, *, code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "tool_name": tool_name,
            "code": code,
            "message": message,
            "user_message": (
                "Не получилось создать ссылку: выбранный вариант устарел. "
                "Повторите поиск и выберите вариант снова."
            ),
            "hint": "Show error.user_message to the user. Do not retry this selection_id.",
            "retryable": False,
            "next_action": "rerun_search",
            "invalid_fields": ["selection_id"],
        },
    }


def _object_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _new_selection_id() -> str:
    return f"{TUTU_SELECTION_ID_PREFIX}{uuid4().hex[:16]}"


def _prune_compacted_value(value: Any) -> Any:
    if isinstance(value, dict):
        compacted_dict = {key: _prune_compacted_value(item) for key, item in value.items()}
        return {key: item for key, item in compacted_dict.items() if item not in (None, "", [], {})}
    if isinstance(value, list):
        compacted_list = [_prune_compacted_value(item) for item in value]
        return [item for item in compacted_list if item not in (None, "", [], {})]
    return value
