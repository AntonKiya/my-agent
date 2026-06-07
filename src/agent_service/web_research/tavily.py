import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict, cast

import httpx
from pydantic_ai.toolsets import AgentToolset, FunctionToolset

from agent_service.config import AppSettings
from agent_service.observability.events import elapsed_ms, log_event, start_timer

logger = logging.getLogger(__name__)

WEB_RESEARCH_TOOLSET_ID = "web_research"
WEB_RESEARCH_TOOL_NAME = "web_research"
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
TAVILY_SEARCH_MAX_RESULTS = 10
WEB_RESEARCH_MAX_SOURCES = 5

SearchDepth = Literal["ultra-fast", "fast", "basic", "advanced"]
ExtractDepth = Literal["basic", "advanced"]
WebResearchStatus = Literal[
    "ok",
    "needs_query",
    "no_search_results",
    "no_sources",
    "temporarily_unavailable",
]


class WebResearchSource(TypedDict):
    source_index: int
    url: str
    title: str | None
    content: str


class WebResearchMetadata(TypedDict):
    searched_results: int
    attempted_extracts: int
    successful_extracts: int
    failed_extracts: int
    fallback_extracts: int
    search_request_id: str | None
    extract_request_ids: list[str]


class WebResearchResult(TypedDict):
    status: WebResearchStatus
    query: str
    sources: list[WebResearchSource]
    metadata: WebResearchMetadata
    message: NotRequired[str]


@dataclass(frozen=True, slots=True)
class TavilySearchResult:
    url: str
    title: str | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class TavilySearchResponse:
    results: tuple[TavilySearchResult, ...]
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class TavilyExtractResult:
    url: str
    raw_content: str
    title: str | None = None


@dataclass(frozen=True, slots=True)
class TavilyExtractResponse:
    results: tuple[TavilyExtractResult, ...]
    failed_urls: tuple[str, ...]
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractBatchOutcome:
    sources: tuple[WebResearchSource, ...]
    attempted_count: int
    failed_count: int
    request_id: str | None = None


class TavilyPayloadError(RuntimeError):
    """Raised when Tavily returns a successful response with an invalid shape."""


@dataclass(slots=True)
class TavilyWebResearchClient:
    api_key: str
    http_client: httpx.AsyncClient
    search_url: str = TAVILY_SEARCH_URL
    extract_url: str = TAVILY_EXTRACT_URL

    async def search(
        self,
        *,
        query: str,
        max_results: int = TAVILY_SEARCH_MAX_RESULTS,
        search_depth: SearchDepth = "advanced",
    ) -> TavilySearchResponse:
        started_at = start_timer()
        log_event(
            logger,
            logging.INFO,
            "Tavily search request started",
            event="tavily_search_started",
            query_length=len(query),
            max_results=max_results,
            search_depth=search_depth,
        )
        try:
            response = await self.http_client.post(
                self.search_url,
                headers=self._headers(),
                json={
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": max_results,
                    "include_answer": False,
                    "include_images": False,
                    "include_raw_content": False,
                },
            )
            response.raise_for_status()
            payload = _response_mapping(response)
            parsed = _parse_search_response(payload)
        except (httpx.HTTPError, TavilyPayloadError) as exc:
            log_event(
                logger,
                logging.WARNING,
                "Tavily search request failed",
                event="tavily_search_failed",
                query_length=len(query),
                duration_ms=elapsed_ms(started_at),
                error_type=type(exc).__name__,
                error_message=_safe_error_message(exc),
            )
            raise

        log_event(
            logger,
            logging.INFO,
            "Tavily search request completed",
            event="tavily_search_completed",
            query_length=len(query),
            duration_ms=elapsed_ms(started_at),
            result_count=len(parsed.results),
            request_id=parsed.request_id,
        )
        return parsed

    async def extract(
        self,
        *,
        urls: Sequence[str],
        extract_depth: ExtractDepth = "basic",
    ) -> TavilyExtractResponse:
        started_at = start_timer()
        url_count = len(urls)
        log_event(
            logger,
            logging.INFO,
            "Tavily extract request started",
            event="tavily_extract_started",
            url_count=url_count,
            extract_depth=extract_depth,
        )
        try:
            response = await self.http_client.post(
                self.extract_url,
                headers=self._headers(),
                json={
                    "urls": list(urls),
                    "extract_depth": extract_depth,
                    "format": "markdown",
                },
            )
            response.raise_for_status()
            payload = _response_mapping(response)
            parsed = _parse_extract_response(payload)
        except (httpx.HTTPError, TavilyPayloadError) as exc:
            log_event(
                logger,
                logging.WARNING,
                "Tavily extract request failed",
                event="tavily_extract_failed",
                url_count=url_count,
                duration_ms=elapsed_ms(started_at),
                error_type=type(exc).__name__,
                error_message=_safe_error_message(exc),
            )
            raise

        log_event(
            logger,
            logging.INFO,
            "Tavily extract request completed",
            event="tavily_extract_completed",
            url_count=url_count,
            duration_ms=elapsed_ms(started_at),
            result_count=len(parsed.results),
            failed_result_count=len(parsed.failed_urls),
            request_id=parsed.request_id,
        )
        return parsed

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


@dataclass(slots=True)
class WebResearchService:
    tavily_client: TavilyWebResearchClient
    search_max_results: int = TAVILY_SEARCH_MAX_RESULTS
    max_sources: int = WEB_RESEARCH_MAX_SOURCES
    search_depth: SearchDepth = "advanced"
    extract_depth: ExtractDepth = "basic"

    async def research(self, *, query: str) -> WebResearchResult:
        clean_query = query.strip()
        if not clean_query:
            return _web_research_payload(
                status="needs_query",
                query=clean_query,
                sources=(),
                searched_results=0,
                attempted_extracts=0,
                successful_extracts=0,
                failed_extracts=0,
                fallback_extracts=0,
                search_request_id=None,
                extract_request_ids=[],
                message="Ask the user what they want to research.",
            )

        try:
            search_response = await self.tavily_client.search(
                query=clean_query,
                max_results=self.search_max_results,
                search_depth=self.search_depth,
            )
        except (httpx.HTTPError, TavilyPayloadError):
            return _web_research_payload(
                status="temporarily_unavailable",
                query=clean_query,
                sources=(),
                searched_results=0,
                attempted_extracts=0,
                successful_extracts=0,
                failed_extracts=0,
                fallback_extracts=0,
                search_request_id=None,
                extract_request_ids=[],
                message="Tavily search is temporarily unavailable.",
            )

        candidates = search_response.results[: self.search_max_results]
        if not candidates:
            return _web_research_payload(
                status="no_search_results",
                query=clean_query,
                sources=(),
                searched_results=0,
                attempted_extracts=0,
                successful_extracts=0,
                failed_extracts=0,
                fallback_extracts=0,
                search_request_id=search_response.request_id,
                extract_request_ids=[],
                message="No search results were returned.",
            )

        first_batch = candidates[: self.max_sources]
        sources: list[WebResearchSource] = []
        attempted_extracts = 0
        failed_extracts = 0
        fallback_extracts = 0
        extract_request_ids: list[str] = []

        first_outcome = await self._extract_batch(first_batch, source_index_offset=0)
        sources.extend(first_outcome.sources)
        attempted_extracts += first_outcome.attempted_count
        failed_extracts += first_outcome.failed_count
        if first_outcome.request_id is not None:
            extract_request_ids.append(first_outcome.request_id)

        needed = self.max_sources - len(sources)
        if needed > 0:
            fallback_batch = candidates[self.max_sources : self.max_sources + needed]
            if fallback_batch:
                fallback_outcome = await self._extract_batch(
                    fallback_batch,
                    source_index_offset=len(sources),
                )
                sources.extend(fallback_outcome.sources)
                attempted_extracts += fallback_outcome.attempted_count
                failed_extracts += fallback_outcome.failed_count
                fallback_extracts += fallback_outcome.attempted_count
                if fallback_outcome.request_id is not None:
                    extract_request_ids.append(fallback_outcome.request_id)

        status: WebResearchStatus = "ok" if sources else "no_sources"
        message = None if sources else "No sources could be extracted from Tavily search results."
        return _web_research_payload(
            status=status,
            query=clean_query,
            sources=sources,
            searched_results=len(candidates),
            attempted_extracts=attempted_extracts,
            successful_extracts=len(sources),
            failed_extracts=failed_extracts,
            fallback_extracts=fallback_extracts,
            search_request_id=search_response.request_id,
            extract_request_ids=extract_request_ids,
            message=message,
        )

    async def _extract_batch(
        self,
        candidates: Sequence[TavilySearchResult],
        *,
        source_index_offset: int,
    ) -> ExtractBatchOutcome:
        urls = [candidate.url for candidate in candidates]
        if not urls:
            return ExtractBatchOutcome((), attempted_count=0, failed_count=0)

        titles_by_url = {candidate.url: candidate.title for candidate in candidates}
        try:
            response = await self.tavily_client.extract(
                urls=urls,
                extract_depth=self.extract_depth,
            )
        except (httpx.HTTPError, TavilyPayloadError):
            return ExtractBatchOutcome(
                (),
                attempted_count=len(urls),
                failed_count=len(urls),
            )

        sources: list[WebResearchSource] = []
        for item in response.results:
            source = _source_from_extract_result(
                item,
                title=titles_by_url.get(item.url),
                source_index=source_index_offset + len(sources) + 1,
            )
            if source is not None:
                sources.append(source)
        failed_count = len(urls) - len(sources)
        return ExtractBatchOutcome(
            tuple(sources),
            attempted_count=len(urls),
            failed_count=failed_count,
            request_id=response.request_id,
        )


def build_web_research_toolsets(
    settings: AppSettings,
    *,
    http_client: httpx.AsyncClient,
) -> tuple[AgentToolset[Any], ...]:
    if not settings.web_research_enabled:
        return ()
    if settings.tavily_api_key is None:
        return ()

    client = TavilyWebResearchClient(
        api_key=settings.tavily_api_key.get_secret_value(),
        http_client=http_client,
    )
    service = WebResearchService(
        client,
        search_depth=settings.web_research_search_depth,
        extract_depth=settings.web_research_extract_depth,
    )

    async def web_research(query: str) -> WebResearchResult:
        """Search the web and return evidence from fetched sources only.

        Args:
            query: The user's web research question as a plain search query.
        """
        return await service.research(query=query)

    return (
        FunctionToolset(
            [web_research],
            id=WEB_RESEARCH_TOOLSET_ID,
            timeout=settings.web_research_tool_timeout_seconds,
            require_parameter_descriptions=True,
        ),
    )


def _parse_search_response(payload: Mapping[str, object]) -> TavilySearchResponse:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise TavilyPayloadError("Tavily search response must include a results list")

    results: list[TavilySearchResult] = []
    for item in raw_results:
        mapping = _as_mapping(item)
        if mapping is None:
            continue
        url = _optional_str(mapping.get("url"))
        if url is None:
            continue
        results.append(
            TavilySearchResult(
                url=url,
                title=_optional_str(mapping.get("title")),
                score=_optional_float(mapping.get("score")),
            )
        )
    return TavilySearchResponse(
        results=tuple(results),
        request_id=_optional_str(payload.get("request_id")),
    )


def _parse_extract_response(payload: Mapping[str, object]) -> TavilyExtractResponse:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise TavilyPayloadError("Tavily extract response must include a results list")

    results: list[TavilyExtractResult] = []
    for item in raw_results:
        mapping = _as_mapping(item)
        if mapping is None:
            continue
        url = _optional_str(mapping.get("url"))
        raw_content = _optional_str(mapping.get("raw_content"))
        if url is None or raw_content is None:
            continue
        results.append(
            TavilyExtractResult(
                url=url,
                raw_content=raw_content,
                title=_optional_str(mapping.get("title")),
            )
        )

    failed_urls = []
    raw_failed_results = payload.get("failed_results")
    if isinstance(raw_failed_results, list):
        for item in raw_failed_results:
            mapping = _as_mapping(item)
            if mapping is None:
                continue
            url = _optional_str(mapping.get("url"))
            if url is not None:
                failed_urls.append(url)

    return TavilyExtractResponse(
        results=tuple(results),
        failed_urls=tuple(failed_urls),
        request_id=_optional_str(payload.get("request_id")),
    )


def _source_from_extract_result(
    result: TavilyExtractResult,
    *,
    title: str | None,
    source_index: int,
) -> WebResearchSource | None:
    content = result.raw_content.strip()
    if not content:
        return None
    return {
        "source_index": source_index,
        "url": result.url,
        "title": result.title or title,
        "content": content,
    }


def _web_research_payload(
    *,
    status: WebResearchStatus,
    query: str,
    sources: Sequence[WebResearchSource],
    searched_results: int,
    attempted_extracts: int,
    successful_extracts: int,
    failed_extracts: int,
    fallback_extracts: int,
    search_request_id: str | None,
    extract_request_ids: list[str],
    message: str | None = None,
) -> WebResearchResult:
    payload: WebResearchResult = {
        "status": status,
        "query": query,
        "sources": list(sources),
        "metadata": {
            "searched_results": searched_results,
            "attempted_extracts": attempted_extracts,
            "successful_extracts": successful_extracts,
            "failed_extracts": failed_extracts,
            "fallback_extracts": fallback_extracts,
            "search_request_id": search_request_id,
            "extract_request_ids": extract_request_ids,
        },
    }
    if message is not None:
        payload["message"] = message
    return payload


def _response_mapping(response: httpx.Response) -> Mapping[str, object]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise TavilyPayloadError("Tavily response is not valid JSON") from exc
    mapping = _as_mapping(payload)
    if mapping is None:
        raise TavilyPayloadError("Tavily response must be a JSON object")
    return mapping


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return None


def _optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _safe_error_message(exc: BaseException) -> str:
    message = str(exc)
    if not message:
        return type(exc).__name__
    return message[:500]
