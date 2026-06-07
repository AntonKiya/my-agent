import json
from typing import Any, cast

import httpx
from pydantic import SecretStr

from agent_service.config import AppSettings
from agent_service.web_research import WEB_RESEARCH_TOOLSET_ID, build_web_research_toolsets
from agent_service.web_research.tavily import TavilyWebResearchClient, WebResearchService


def _search_result(index: int) -> dict[str, object]:
    return {
        "url": f"https://example.com/{index}",
        "title": f"Source {index}",
        "content": f"search snippet {index}",
        "score": 1 - index / 100,
    }


def _extract_result(index: int, *, raw_content: str | None = None) -> dict[str, object]:
    content = raw_content if raw_content is not None else f"# Source {index}\nRead content"
    return {
        "url": f"https://example.com/{index}",
        "title": f"Extracted {index}",
        "raw_content": content,
    }


def _request_json(request: httpx.Request) -> dict[str, Any]:
    payload = json.loads(request.content)
    assert isinstance(payload, dict)
    return payload


async def test_web_research_returns_fetched_evidence_without_search_snippets() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer tvly-test"
        if request.url.path == "/search":
            payload = _request_json(request)
            assert payload["query"] == "renewable energy impacts"
            assert payload["search_depth"] == "advanced"
            assert payload["max_results"] == 10
            assert payload["include_raw_content"] is False
            return httpx.Response(
                200,
                json={
                    "results": [_search_result(index) for index in range(1, 11)],
                    "request_id": "search-1",
                },
            )

        assert request.url.path == "/extract"
        payload = _request_json(request)
        assert payload["urls"] == [f"https://example.com/{index}" for index in range(1, 6)]
        assert payload["extract_depth"] == "basic"
        assert payload["format"] == "markdown"
        return httpx.Response(
            200,
            json={
                "results": [_extract_result(index) for index in range(1, 6)],
                "failed_results": [],
                "request_id": "extract-1",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.tavily.com",
    ) as http_client:
        client = TavilyWebResearchClient(api_key="tvly-test", http_client=http_client)
        service = WebResearchService(client)

        result = await service.research(query="renewable energy impacts")

    assert result["status"] == "ok"
    assert len(requests) == 2
    assert len(result["sources"]) == 5
    assert result["sources"][0] == {
        "source_index": 1,
        "url": "https://example.com/1",
        "title": "Extracted 1",
        "content": "# Source 1\nRead content",
    }
    assert "search snippet" not in json.dumps(result)
    assert result["metadata"] == {
        "searched_results": 10,
        "attempted_extracts": 5,
        "successful_extracts": 5,
        "failed_extracts": 0,
        "fallback_extracts": 0,
        "search_request_id": "search-1",
        "extract_request_ids": ["extract-1"],
    }


async def test_web_research_uses_one_fallback_extract_batch_for_missing_sources() -> None:
    extract_batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [_search_result(index) for index in range(1, 11)],
                    "request_id": "search-1",
                },
            )

        payload = _request_json(request)
        urls = list(payload["urls"])
        extract_batches.append(urls)
        if urls == [f"https://example.com/{index}" for index in range(1, 6)]:
            return httpx.Response(
                200,
                json={
                    "results": [
                        _extract_result(1),
                        _extract_result(3),
                        _extract_result(5),
                    ],
                    "failed_results": [
                        {"url": "https://example.com/2"},
                        {"url": "https://example.com/4"},
                    ],
                    "request_id": "extract-1",
                },
            )

        assert urls == ["https://example.com/6", "https://example.com/7"]
        return httpx.Response(
            200,
            json={
                "results": [_extract_result(6), _extract_result(7)],
                "failed_results": [],
                "request_id": "extract-2",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.tavily.com",
    ) as http_client:
        client = TavilyWebResearchClient(api_key="tvly-test", http_client=http_client)
        service = WebResearchService(client)

        result = await service.research(query="query")

    assert result["status"] == "ok"
    assert extract_batches == [
        [f"https://example.com/{index}" for index in range(1, 6)],
        ["https://example.com/6", "https://example.com/7"],
    ]
    assert [source["url"] for source in result["sources"]] == [
        "https://example.com/1",
        "https://example.com/3",
        "https://example.com/5",
        "https://example.com/6",
        "https://example.com/7",
    ]
    assert result["metadata"]["attempted_extracts"] == 7
    assert result["metadata"]["successful_extracts"] == 5
    assert result["metadata"]["failed_extracts"] == 2
    assert result["metadata"]["fallback_extracts"] == 2
    assert result["metadata"]["extract_request_ids"] == ["extract-1", "extract-2"]


async def test_web_research_uses_configured_search_and_extract_depths() -> None:
    request_payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = _request_json(request)
        request_payloads.append(payload)
        if request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [_search_result(1)],
                    "request_id": "search-1",
                },
            )

        return httpx.Response(
            200,
            json={
                "results": [_extract_result(1)],
                "failed_results": [],
                "request_id": "extract-1",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.tavily.com",
    ) as http_client:
        client = TavilyWebResearchClient(api_key="tvly-test", http_client=http_client)
        service = WebResearchService(
            client,
            search_depth="fast",
            extract_depth="advanced",
        )

        result = await service.research(query="query")

    assert result["status"] == "ok"
    assert request_payloads[0]["search_depth"] == "fast"
    assert request_payloads[1]["extract_depth"] == "advanced"


async def test_web_research_stops_after_one_fallback_batch() -> None:
    extract_call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal extract_call_count
        if request.url.path == "/search":
            return httpx.Response(
                200,
                json={
                    "results": [_search_result(index) for index in range(1, 11)],
                    "request_id": "search-1",
                },
            )

        extract_call_count += 1
        if extract_call_count == 1:
            return httpx.Response(503)
        return httpx.Response(
            200,
            json={
                "results": [_extract_result(6), _extract_result(7)],
                "failed_results": [
                    {"url": "https://example.com/8"},
                    {"url": "https://example.com/9"},
                    {"url": "https://example.com/10"},
                ],
                "request_id": "extract-2",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.tavily.com",
    ) as http_client:
        client = TavilyWebResearchClient(api_key="tvly-test", http_client=http_client)
        service = WebResearchService(client)

        result = await service.research(query="query")

    assert extract_call_count == 2
    assert result["status"] == "ok"
    assert len(result["sources"]) == 2
    assert result["metadata"]["attempted_extracts"] == 10
    assert result["metadata"]["successful_extracts"] == 2
    assert result["metadata"]["failed_extracts"] == 8
    assert result["metadata"]["fallback_extracts"] == 5


async def test_web_research_handles_empty_query_without_http_call() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected HTTP request")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.tavily.com",
    ) as http_client:
        client = TavilyWebResearchClient(api_key="tvly-test", http_client=http_client)
        service = WebResearchService(client)

        result = await service.research(query="  ")

    assert result["status"] == "needs_query"
    assert result["sources"] == []
    assert result["metadata"]["attempted_extracts"] == 0


async def test_web_research_reports_search_unavailable() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.tavily.com",
    ) as http_client:
        client = TavilyWebResearchClient(api_key="tvly-test", http_client=http_client)
        service = WebResearchService(client)

        result = await service.research(query="query")

    assert result["status"] == "temporarily_unavailable"
    assert result["sources"] == []
    assert result.get("message") == "Tavily search is temporarily unavailable."


async def test_build_web_research_toolsets_can_be_disabled_or_unconfigured() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(500))
    ) as http_client:
        assert (
            build_web_research_toolsets(
                AppSettings(environment="test", web_research_enabled=False),
                http_client=http_client,
            )
            == ()
        )
        assert (
            build_web_research_toolsets(
                AppSettings(environment="test", tavily_api_key=None),
                http_client=http_client,
            )
            == ()
        )


async def test_build_web_research_toolsets_has_expected_id() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(500))
    ) as http_client:
        toolsets = build_web_research_toolsets(
            AppSettings(environment="test", tavily_api_key=SecretStr("tvly-test")),
            http_client=http_client,
        )

    assert len(toolsets) == 1
    toolset = cast(Any, toolsets[0])
    assert toolset.id == WEB_RESEARCH_TOOLSET_ID
    assert set(toolset.tools) == {"web_research"}
