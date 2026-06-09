import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

import httpx

from agent_service.image_analysis.interfaces import (
    EmptyImageAnalysisError,
    ImageAnalysisError,
    ImageAnalyzer,
)
from agent_service.image_analysis.models import ImageAnalysisRequest, ImageAnalysisResult
from agent_service.observability.events import elapsed_ms, log_event, start_timer

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_COMPLETIONS_PATH = "/api/v1/chat/completions"
OPENROUTER_IMAGE_ANALYSIS_PROVIDER = "openrouter"


class OpenRouterVisionAnalyzer(ImageAnalyzer):
    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.AsyncClient,
        model: str,
        api_base_url: str = "https://openrouter.ai",
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key must not be empty")
        if not model:
            raise ValueError("OpenRouter image analysis model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("OpenRouter image analysis timeout must be greater than zero")
        self._api_key = api_key
        self._client = client
        self._model = model
        self._api_base_url = api_base_url
        self._timeout_seconds = timeout_seconds

    async def analyze(self, request: ImageAnalysisRequest) -> ImageAnalysisResult:
        started_at = start_timer()
        log_event(
            logger,
            logging.INFO,
            "OpenRouter image analysis request started",
            event="openrouter_image_analysis_request_started",
            model=self._model,
            image_count=len(request.assets),
            total_image_size_bytes=sum(asset.size_bytes for asset in request.assets),
        )
        try:
            response = await asyncio.wait_for(
                self._post_analysis(request),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            raise ImageAnalysisError(
                "OpenRouter image analysis timed out",
                retryable=True,
                error_code="openrouter_image_analysis_timeout",
            ) from exc
        except httpx.TransportError as exc:
            raise ImageAnalysisError(
                "OpenRouter image analysis transport error",
                retryable=True,
                error_code=_transport_error_code(exc),
            ) from exc

        body = _response_json(response)
        log_event(
            logger,
            logging.INFO if response.status_code < 400 else logging.WARNING,
            "OpenRouter image analysis response received",
            event="openrouter_image_analysis_response_received",
            http_status_code=response.status_code,
            model=self._model,
            image_count=len(request.assets),
            duration_ms=elapsed_ms(started_at),
            openrouter_request_id=response.headers.get("x-request-id")
            or response.headers.get("x-openrouter-request-id"),
        )
        if response.status_code >= 400:
            raise ImageAnalysisError(
                _openrouter_error_message(response, body),
                retryable=_retryable_status(response.status_code),
                error_code=_openrouter_error_code(response, body),
            )
        analysis = _analysis_text(body)
        if analysis is None:
            raise EmptyImageAnalysisError(
                "OpenRouter image analysis returned empty text",
                retryable=False,
                error_code="openrouter_empty_image_analysis",
            )
        return ImageAnalysisResult(
            analysis=analysis,
            provider=OPENROUTER_IMAGE_ANALYSIS_PROVIDER,
            model=self._model,
            metadata={
                "request_id": response.headers.get("x-request-id")
                or response.headers.get("x-openrouter-request-id"),
            },
        )

    async def _post_analysis(self, request: ImageAnalysisRequest) -> httpx.Response:
        content: list[dict[str, object]] = [{"type": "text", "text": request.prompt}]
        for asset in request.assets:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": await _data_url(asset.storage_key, asset.content_type),
                    },
                }
            )
        return await self._client.post(
            self._url(),
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": content}],
            },
        )

    def _url(self) -> str:
        return f"{self._api_base_url.rstrip('/')}{OPENROUTER_CHAT_COMPLETIONS_PATH}"


async def _data_url(storage_key: str, content_type: str | None) -> str:
    path = Path(storage_key)
    content = await asyncio.to_thread(path.read_bytes)
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{content_type or 'application/octet-stream'};base64,{encoded}"


def _analysis_text(body: object) -> str | None:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


def _response_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _openrouter_error_code(response: httpx.Response, body: Any) -> str:
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict) and error.get("code") is not None:
        return f"openrouter_{error['code']}"
    return f"openrouter_http_{response.status_code}"


def _openrouter_error_message(response: httpx.Response, body: Any) -> str:
    error = body.get("error") if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    if isinstance(message, str):
        return message
    return response.reason_phrase or "OpenRouter image analysis request failed"


def _retryable_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _transport_error_code(exc: httpx.TransportError) -> str:
    if isinstance(exc, httpx.ConnectTimeout):
        return "openrouter_connect_timeout"
    if isinstance(exc, httpx.ReadTimeout):
        return "openrouter_read_timeout"
    if isinstance(exc, httpx.WriteTimeout):
        return "openrouter_write_timeout"
    if isinstance(exc, httpx.PoolTimeout):
        return "openrouter_pool_timeout"
    if isinstance(exc, httpx.TimeoutException):
        return "openrouter_timeout"
    if isinstance(exc, httpx.NetworkError):
        return "openrouter_network_error"
    if isinstance(exc, httpx.ProtocolError):
        return "openrouter_protocol_error"
    return "openrouter_transport_error"
