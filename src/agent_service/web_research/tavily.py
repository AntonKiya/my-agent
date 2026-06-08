import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NotRequired, TypedDict, cast
from urllib.parse import urlparse

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
READ_WEB_PAGES_MAX_URLS = 5
TAVILY_RAW_RESPONSE_LOG_LIMIT = 12_000

SearchDepth = Literal["ultra-fast", "fast", "basic", "advanced"]
ExtractDepth = Literal["basic", "advanced"]
WebResearchStatus = Literal[
    "ok",
    "needs_query",
    "no_search_results",
    "no_sources",
    "temporarily_unavailable",
]
ReadWebPagesStatus = Literal[
    "ok",
    "partial_success",
    "no_sources",
    "validation_error",
    "temporarily_unavailable",
]
ReadWebPagesFailedStatus = Literal[
    "invalid_url",
    "extract_failed",
    "temporarily_unavailable",
]


class WebResearchSource(TypedDict):
    source_index: int
    url: str
    title: str | None
    content: str
    content_length: int
    content_truncated: bool


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


class ReadWebPagesFailedUrl(TypedDict):
    url: str
    status: ReadWebPagesFailedStatus
    reason: str


class ReadWebPagesMetadata(TypedDict):
    requested_urls: int
    valid_urls: int
    successful_extracts: int
    failed_extracts: int
    extract_depth: ExtractDepth
    request_id: str | None


class ReadWebPagesResult(TypedDict):
    status: ReadWebPagesStatus
    sources: list[WebResearchSource]
    failed_urls: list[ReadWebPagesFailedUrl]
    metadata: ReadWebPagesMetadata
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
class TavilyExtractFailure:
    url: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TavilyExtractResponse:
    results: tuple[TavilyExtractResult, ...]
    failed_results: tuple[TavilyExtractFailure, ...]
    request_id: str | None = None

    @property
    def failed_urls(self) -> tuple[str, ...]:
        return tuple(item.url for item in self.failed_results)


@dataclass(frozen=True, slots=True)
class ExtractBatchOutcome:
    sources: tuple[WebResearchSource, ...]
    attempted_count: int
    failed_count: int
    request_id: str | None = None


@dataclass(frozen=True, slots=True)
class LimitedContent:
    text: str
    original_length: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class UrlValidation:
    valid_urls: tuple[str, ...]
    failed_urls: tuple[ReadWebPagesFailedUrl, ...]
    message: str | None = None


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
            _log_raw_tavily_payload(
                event="tavily_extract_raw_response",
                request_id=parsed.request_id,
                payload=payload,
            )
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
    max_content_chars_per_source: int = 20_000

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
                max_content_chars=self.max_content_chars_per_source,
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


@dataclass(slots=True)
class ReadWebPagesService:
    tavily_client: TavilyWebResearchClient
    extract_depth: ExtractDepth = "basic"
    max_urls: int = READ_WEB_PAGES_MAX_URLS
    max_content_chars_per_source: int = 20_000

    async def read_pages(self, *, urls: list[str]) -> ReadWebPagesResult:
        log_event(
            logger,
            logging.INFO,
            "read_web_pages request started",
            event="read_web_pages_started",
            requested_urls=len(urls),
            extract_depth=self.extract_depth,
        )
        validation = _validate_read_web_pages_urls(urls, max_urls=self.max_urls)
        if validation.message is not None:
            result = _read_web_pages_payload(
                status="validation_error",
                sources=(),
                failed_urls=validation.failed_urls,
                requested_urls=len(urls),
                valid_urls=0,
                successful_extracts=0,
                failed_extracts=len(validation.failed_urls),
                extract_depth=self.extract_depth,
                request_id=None,
                message=validation.message,
            )
            _log_read_web_pages_completed(result, duration_ms=None)
            return result

        valid_urls = validation.valid_urls
        failed_urls = list(validation.failed_urls)
        started_at = start_timer()
        try:
            response = await self.tavily_client.extract(
                urls=valid_urls,
                extract_depth=self.extract_depth,
            )
        except (httpx.HTTPError, TavilyPayloadError) as exc:
            failed_urls.extend(
                _failed_url(
                    url,
                    status="temporarily_unavailable",
                    reason=f"Tavily extract request failed: {type(exc).__name__}",
                )
                for url in valid_urls
            )
            result = _read_web_pages_payload(
                status="temporarily_unavailable",
                sources=(),
                failed_urls=failed_urls,
                requested_urls=len(urls),
                valid_urls=len(valid_urls),
                successful_extracts=0,
                failed_extracts=len(failed_urls),
                extract_depth=self.extract_depth,
                request_id=None,
                message="Tavily extract is temporarily unavailable.",
            )
            _log_read_web_pages_completed(result, duration_ms=elapsed_ms(started_at))
            return result

        sources, extract_failures = _read_web_pages_sources_and_failures(
            requested_urls=valid_urls,
            response=response,
            max_content_chars=self.max_content_chars_per_source,
        )
        failed_urls.extend(extract_failures)
        status = _read_web_pages_status(sources, failed_urls)
        result = _read_web_pages_payload(
            status=status,
            sources=sources,
            failed_urls=failed_urls,
            requested_urls=len(urls),
            valid_urls=len(valid_urls),
            successful_extracts=len(sources),
            failed_extracts=len(failed_urls),
            extract_depth=self.extract_depth,
            request_id=response.request_id,
            message=None if sources else "No requested pages could be extracted.",
        )
        _log_read_web_pages_completed(result, duration_ms=elapsed_ms(started_at))
        return result


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
        max_content_chars_per_source=settings.web_research_max_content_chars_per_source,
    )
    read_pages_service = ReadWebPagesService(
        client,
        extract_depth=settings.web_research_extract_depth,
        max_content_chars_per_source=settings.web_research_max_content_chars_per_source,
    )

    async def web_research(query: str) -> WebResearchResult:
        """Search the web and return evidence from fetched sources only.

        Args:
            query: The user's web research question as a plain search query.
        """
        return await service.research(query=query)

    async def read_web_pages(urls: list[str]) -> ReadWebPagesResult:
        """Read specific web pages and return evidence from fetched pages only.

        Args:
            urls: One to five HTTP or HTTPS URLs to read directly without web search.
        """
        return await read_pages_service.read_pages(urls=urls)

    return (
        FunctionToolset(
            [web_research, read_web_pages],
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

    failed_results = []
    raw_failed_results = payload.get("failed_results")
    if isinstance(raw_failed_results, list):
        for item in raw_failed_results:
            mapping = _as_mapping(item)
            if mapping is None:
                continue
            url = _optional_str(mapping.get("url"))
            if url is not None:
                failed_results.append(
                    TavilyExtractFailure(
                        url=url,
                        reason=_extract_failure_reason(mapping),
                    )
                )

    return TavilyExtractResponse(
        results=tuple(results),
        failed_results=tuple(failed_results),
        request_id=_optional_str(payload.get("request_id")),
    )


def _source_from_extract_result(
    result: TavilyExtractResult,
    *,
    title: str | None,
    source_index: int,
    max_content_chars: int,
) -> WebResearchSource | None:
    content = _limited_content(result.raw_content, max_chars=max_content_chars)
    if not content.text:
        return None
    return {
        "source_index": source_index,
        "url": result.url,
        "title": result.title or title,
        "content": content.text,
        "content_length": content.original_length,
        "content_truncated": content.truncated,
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


def _read_web_pages_payload(
    *,
    status: ReadWebPagesStatus,
    sources: Sequence[WebResearchSource],
    failed_urls: Sequence[ReadWebPagesFailedUrl],
    requested_urls: int,
    valid_urls: int,
    successful_extracts: int,
    failed_extracts: int,
    extract_depth: ExtractDepth,
    request_id: str | None,
    message: str | None = None,
) -> ReadWebPagesResult:
    payload: ReadWebPagesResult = {
        "status": status,
        "sources": list(sources),
        "failed_urls": list(failed_urls),
        "metadata": {
            "requested_urls": requested_urls,
            "valid_urls": valid_urls,
            "successful_extracts": successful_extracts,
            "failed_extracts": failed_extracts,
            "extract_depth": extract_depth,
            "request_id": request_id,
        },
    }
    if message is not None:
        payload["message"] = message
    return payload


def _validate_read_web_pages_urls(urls: Sequence[str], *, max_urls: int) -> UrlValidation:
    if not urls:
        return UrlValidation(
            valid_urls=(),
            failed_urls=(),
            message="read_web_pages requires at least one URL.",
        )
    if len(urls) > max_urls:
        return UrlValidation(
            valid_urls=(),
            failed_urls=tuple(
                _failed_url(
                    url,
                    status="invalid_url",
                    reason=f"read_web_pages accepts at most {max_urls} URLs per call.",
                )
                for url in urls
            ),
            message=f"read_web_pages accepts at most {max_urls} URLs per call.",
        )

    valid_urls = []
    failed_urls = []
    for url in urls:
        clean_url = url.strip()
        if _is_supported_url(clean_url):
            valid_urls.append(clean_url)
            continue
        failed_urls.append(
            _failed_url(
                url,
                status="invalid_url",
                reason="URL must be an absolute http:// or https:// URL.",
            )
        )
    if not valid_urls:
        return UrlValidation(
            valid_urls=(),
            failed_urls=tuple(failed_urls),
            message="No valid URLs were provided.",
        )
    return UrlValidation(valid_urls=tuple(valid_urls), failed_urls=tuple(failed_urls))


def _is_supported_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _read_web_pages_sources_and_failures(
    *,
    requested_urls: Sequence[str],
    response: TavilyExtractResponse,
    max_content_chars: int,
) -> tuple[list[WebResearchSource], list[ReadWebPagesFailedUrl]]:
    results_by_url = {item.url: item for item in response.results}
    failures_by_url = {item.url: item for item in response.failed_results}
    sources: list[WebResearchSource] = []
    failed_urls: list[ReadWebPagesFailedUrl] = []
    for url in requested_urls:
        result = results_by_url.get(url)
        if result is not None:
            source = _source_from_extract_result(
                result,
                title=None,
                source_index=len(sources) + 1,
                max_content_chars=max_content_chars,
            )
            if source is not None:
                sources.append(source)
                continue

        failure = failures_by_url.get(url)
        failed_urls.append(
            _failed_url(
                url,
                status="extract_failed",
                reason=(
                    failure.reason
                    if failure is not None and failure.reason is not None
                    else "No extracted content returned."
                ),
            )
        )
    return sources, failed_urls


def _read_web_pages_status(
    sources: Sequence[WebResearchSource],
    failed_urls: Sequence[ReadWebPagesFailedUrl],
) -> ReadWebPagesStatus:
    if sources and failed_urls:
        return "partial_success"
    if sources:
        return "ok"
    if failed_urls:
        return "no_sources"
    return "no_sources"


def _failed_url(
    url: str,
    *,
    status: ReadWebPagesFailedStatus,
    reason: str,
) -> ReadWebPagesFailedUrl:
    return {
        "url": url,
        "status": status,
        "reason": reason,
    }


def _limited_content(value: str, *, max_chars: int) -> LimitedContent:
    text = value.strip()
    if len(text) <= max_chars:
        return LimitedContent(text=text, original_length=len(text), truncated=False)
    return LimitedContent(
        text=text[:max_chars],
        original_length=len(text),
        truncated=True,
    )


def _extract_failure_reason(mapping: Mapping[str, object]) -> str | None:
    for key in ("error", "reason", "message"):
        value = _optional_str(mapping.get(key))
        if value is not None:
            return value
    return None


def _log_read_web_pages_completed(
    result: ReadWebPagesResult,
    *,
    duration_ms: float | None,
) -> None:
    log_event(
        logger,
        logging.INFO,
        "read_web_pages request completed",
        event="read_web_pages_completed",
        status=result["status"],
        requested_urls=result["metadata"]["requested_urls"],
        valid_urls=result["metadata"]["valid_urls"],
        successful_extracts=result["metadata"]["successful_extracts"],
        failed_extracts=result["metadata"]["failed_extracts"],
        request_id=result["metadata"]["request_id"],
        duration_ms=duration_ms,
    )


def _log_raw_tavily_payload(
    *,
    event: str,
    request_id: str | None,
    payload: Mapping[str, object],
) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
        return
    text = json.dumps(payload, ensure_ascii=False, default=str)
    log_event(
        logger,
        logging.DEBUG,
        "Tavily raw response received",
        event=event,
        request_id=request_id,
        raw_response=_truncate_log_text(text, max_length=TAVILY_RAW_RESPONSE_LOG_LIMIT),
    )


def _truncate_log_text(value: str, *, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    omitted = len(value) - max_length
    return f"{value[:max_length]}...[truncated {omitted} chars]"


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
