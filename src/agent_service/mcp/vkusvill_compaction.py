import json
from collections.abc import Callable
from typing import Any, cast

VKUSVILL_PRODUCT_COMPACT_PROPERTY_NAMES = frozenset(
    {
        "Состав",
        "Пищевая и энергетическая ценность в 100 г",
    }
)


def compact_vkusvill_products_search_result(value: Any) -> Any:
    return _compact_json_like(value, _compact_vkusvill_products_search_payload)


def compact_vkusvill_product_analogs_result(value: Any) -> Any:
    return _compact_json_like(value, _compact_vkusvill_product_analogs_payload)


def _compact_json_like(
    value: Any,
    payload_compactor: Callable[[Any], dict[str, Any] | None],
) -> Any:
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            return value
        compact_payload = payload_compactor(payload)
        if compact_payload is None:
            return value
        return json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))

    compact_payload = payload_compactor(value)
    if compact_payload is None:
        return value
    return compact_payload


def _compact_vkusvill_products_search_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    data = payload.get("data")
    if not isinstance(data, dict):
        return None

    meta = data.get("meta")
    items = data.get("items")
    if not isinstance(meta, dict) or not isinstance(items, list):
        return None

    compact_items = _compact_vkusvill_product_items(items)
    if not compact_items:
        return None

    return cast(
        dict[str, Any],
        _prune_compacted_value(
            {
                "ok": payload.get("ok"),
                "data": {
                    "meta": meta,
                    "items": compact_items,
                },
            }
        ),
    )


def _compact_vkusvill_product_analogs_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    data = payload.get("data")
    if not isinstance(data, dict):
        return None

    products = data.get("products")
    if not isinstance(products, list):
        return None

    compact_products = _compact_vkusvill_product_items(products)
    if not compact_products:
        return None

    return cast(
        dict[str, Any],
        _prune_compacted_value(
            {
                "ok": payload.get("ok"),
                "data": {
                    "product_id": data.get("product_id"),
                    "total": data.get("total"),
                    "products": compact_products,
                },
            }
        ),
    )


def _compact_vkusvill_product_items(items: list[Any]) -> list[dict[str, Any]]:
    compact_items: list[dict[str, Any]] = []
    for item in items:
        compact_item = _compact_vkusvill_product_item(item)
        if compact_item is not None:
            compact_items.append(compact_item)
    return compact_items


def _compact_vkusvill_product_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None

    raw_price = item.get("price")
    raw_weight = item.get("weight")
    raw_properties = item.get("properties")
    price: dict[str, Any] = raw_price if isinstance(raw_price, dict) else {}
    weight: dict[str, Any] = raw_weight if isinstance(raw_weight, dict) else {}
    properties: list[Any] = raw_properties if isinstance(raw_properties, list) else []

    compact_item = {
        "id": item.get("id"),
        "xml_id": item.get("xml_id"),
        "name": item.get("name"),
        "price": {
            "current": price.get("current"),
            "old": price.get("old"),
            "discount_percent": price.get("discount_percent"),
        },
        "unit": item.get("unit"),
        "weight": {
            "value": weight.get("value"),
            "unit": weight.get("unit"),
        },
        "properties": [
            {
                "name": prop.get("name"),
                "value": prop.get("value"),
            }
            for prop in properties
            if isinstance(prop, dict)
            and prop.get("name") in VKUSVILL_PRODUCT_COMPACT_PROPERTY_NAMES
            and prop.get("value") not in (None, "", [], {})
        ],
    }
    compact_item = cast(dict[str, Any], _prune_compacted_value(compact_item))
    if not compact_item or "xml_id" not in compact_item or "name" not in compact_item:
        return None
    return compact_item


def _prune_compacted_value(value: Any) -> Any:
    if isinstance(value, dict):
        compacted_dict = {
            key: _prune_compacted_value(item) for key, item in value.items()
        }
        return {
            key: item
            for key, item in compacted_dict.items()
            if item not in (None, "", [], {})
        }
    if isinstance(value, list):
        compacted_list = [_prune_compacted_value(item) for item in value]
        return [item for item in compacted_list if item not in (None, "", [], {})]
    return value
