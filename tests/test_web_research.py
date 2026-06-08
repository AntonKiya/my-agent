import json
from typing import Any, cast

import httpx
from pydantic import SecretStr

from agent_service.config import AppSettings
from agent_service.web_research import WEB_RESEARCH_TOOLSET_ID, build_web_research_toolsets
from agent_service.web_research.tavily import (
    ReadWebPagesService,
    TavilyWebResearchClient,
    WebResearchService,
)


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
        "content_length": 23,
        "content_truncated": False,
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


async def test_read_web_pages_extracts_specific_urls_without_search() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/extract"
        payload = _request_json(request)
        assert payload["urls"] == ["https://example.com/1", "https://example.com/2"]
        return httpx.Response(
            200,
            json={
                "results": [_extract_result(1), _extract_result(2)],
                "failed_results": [],
                "request_id": "extract-1",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.tavily.com",
    ) as http_client:
        client = TavilyWebResearchClient(api_key="tvly-test", http_client=http_client)
        service = ReadWebPagesService(client)

        result = await service.read_pages(
            urls=["https://example.com/1", "https://example.com/2"]
        )

    assert len(requests) == 1
    assert result["status"] == "ok"
    assert [source["url"] for source in result["sources"]] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert result["failed_urls"] == []
    assert result["metadata"] == {
        "requested_urls": 2,
        "valid_urls": 2,
        "successful_extracts": 2,
        "failed_extracts": 0,
        "extract_depth": "basic",
        "request_id": "extract-1",
    }


async def test_read_web_pages_returns_partial_success_with_failed_url_reason() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [_extract_result(1)],
                "failed_results": [
                    {
                        "url": "https://example.com/2",
                        "error": "Could not fetch page",
                    }
                ],
                "request_id": "extract-1",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.tavily.com",
    ) as http_client:
        client = TavilyWebResearchClient(api_key="tvly-test", http_client=http_client)
        service = ReadWebPagesService(client)

        result = await service.read_pages(
            urls=["https://example.com/1", "https://example.com/2"]
        )

    assert result["status"] == "partial_success"
    assert len(result["sources"]) == 1
    assert result["failed_urls"] == [
        {
            "url": "https://example.com/2",
            "status": "extract_failed",
            "reason": "Could not fetch page",
        }
    ]


async def test_read_web_pages_validates_urls_without_failing_valid_ones() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = _request_json(request)
        assert payload["urls"] == ["https://example.com/1"]
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
        service = ReadWebPagesService(client)

        result = await service.read_pages(urls=["notaurl", "https://example.com/1"])

    assert len(requests) == 1
    assert result["status"] == "partial_success"
    assert result["failed_urls"] == [
        {
            "url": "notaurl",
            "status": "invalid_url",
            "reason": "URL must be an absolute http:// or https:// URL.",
        }
    ]


async def test_read_web_pages_rejects_empty_or_too_many_urls_without_http_call() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("unexpected HTTP request")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.tavily.com",
    ) as http_client:
        client = TavilyWebResearchClient(api_key="tvly-test", http_client=http_client)
        service = ReadWebPagesService(client)

        empty_result = await service.read_pages(urls=[])
        too_many_result = await service.read_pages(
            urls=[f"https://example.com/{index}" for index in range(1, 7)]
        )

    assert empty_result["status"] == "validation_error"
    assert empty_result.get("message") == "read_web_pages requires at least one URL."
    assert too_many_result["status"] == "validation_error"
    assert len(too_many_result["failed_urls"]) == 6


async def test_read_web_pages_marks_all_valid_urls_failed_when_extract_request_fails() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.tavily.com",
    ) as http_client:
        client = TavilyWebResearchClient(api_key="tvly-test", http_client=http_client)
        service = ReadWebPagesService(client)

        result = await service.read_pages(
            urls=["https://example.com/1", "https://example.com/2"]
        )

    assert result["status"] == "temporarily_unavailable"
    assert result["sources"] == []
    assert [item["url"] for item in result["failed_urls"]] == [
        "https://example.com/1",
        "https://example.com/2",
    ]
    assert {item["status"] for item in result["failed_urls"]} == {"temporarily_unavailable"}


async def test_read_web_pages_truncates_large_content() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [_extract_result(1, raw_content="abcdef")],
                "failed_results": [],
                "request_id": "extract-1",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.tavily.com",
    ) as http_client:
        client = TavilyWebResearchClient(api_key="tvly-test", http_client=http_client)
        service = ReadWebPagesService(client, max_content_chars_per_source=3)

        result = await service.read_pages(urls=["https://example.com/1"])

    assert result["sources"][0]["content"] == "abc"
    assert result["sources"][0]["content_length"] == 6
    assert result["sources"][0]["content_truncated"] is True


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
    assert set(toolset.tools) == {"web_research", "read_web_pages"}
