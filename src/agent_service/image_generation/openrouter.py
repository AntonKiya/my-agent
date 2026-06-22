import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

import httpx

from agent_service.image_generation.interfaces import (
    EmptyImageGenerationError,
    ImageGenerationError,
    ImageGenerator,
)
from agent_service.image_generation.models import (
    GeneratedImage,
    ImageGenerationRequest,
    ImageGenerationResult,
)
from agent_service.observability.events import elapsed_ms, log_event, start_timer

logger = logging.getLogger(__name__)

OPENROUTER_CHAT_COMPLETIONS_PATH = "/api/v1/chat/completions"
OPENROUTER_IMAGE_GENERATION_PROVIDER = "openrouter"
DEFAULT_OPENROUTER_IMAGE_GENERATION_MODEL = "google/gemini-2.5-flash-image"
SUPPORTED_GENERATED_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)


class OpenRouterImageGenerator(ImageGenerator):
    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.AsyncClient,
        model: str = DEFAULT_OPENROUTER_IMAGE_GENERATION_MODEL,
        api_base_url: str = "https://openrouter.ai",
        timeout_seconds: float = 120.0,
        max_output_size_bytes: int = 10_000_000,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key must not be empty")
        if not model:
            raise ValueError("OpenRouter image generation model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("OpenRouter image generation timeout must be greater than zero")
        if max_output_size_bytes <= 0:
            raise ValueError("OpenRouter generated image size limit must be greater than zero")
        self._api_key = api_key
        self._client = client
        self._model = model
        self._api_base_url = api_base_url
        self._timeout_seconds = timeout_seconds
        self._max_output_size_bytes = max_output_size_bytes

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        started_at = start_timer()
        log_event(
            logger,
            logging.INFO,
            "OpenRouter image generation request started",
            event="openrouter_image_generation_request_started",
            model=self._model,
            source_image_count=len(request.source_assets),
            total_source_image_size_bytes=sum(asset.size_bytes for asset in request.source_assets),
        )
        try:
            response = await asyncio.wait_for(
                self._post_generation(request),
                timeout=self._timeout_seconds,
            )
        except TimeoutError as exc:
            raise ImageGenerationError(
                "OpenRouter image generation timed out",
                retryable=True,
                error_code="openrouter_image_generation_timeout",
            ) from exc
        except httpx.TransportError as exc:
            raise ImageGenerationError(
                "OpenRouter image generation transport error",
                retryable=True,
                error_code=_transport_error_code(exc),
            ) from exc

        body = _response_json(response)
        request_id = response.headers.get("x-request-id") or response.headers.get(
            "x-openrouter-request-id"
        )
        log_event(
            logger,
            logging.INFO if response.status_code < 400 else logging.WARNING,
            "OpenRouter image generation response received",
            event="openrouter_image_generation_response_received",
            http_status_code=response.status_code,
            model=self._model,
            source_image_count=len(request.source_assets),
            duration_ms=elapsed_ms(started_at),
            openrouter_request_id=request_id,
        )
        if response.status_code >= 400:
            raise ImageGenerationError(
                _openrouter_error_message(response, body),
                retryable=_retryable_status(response.status_code),
                error_code=_openrouter_error_code(response, body),
            )

        images = _generated_images(body, max_size_bytes=self._max_output_size_bytes)
        if not images:
            raise EmptyImageGenerationError(
                "OpenRouter image generation returned no image",
                retryable=False,
                error_code="openrouter_empty_image_generation",
            )

        return ImageGenerationResult(
            images=tuple(images),
            provider=OPENROUTER_IMAGE_GENERATION_PROVIDER,
            model=self._model,
            text=_response_text(body),
            metadata={"request_id": request_id},
        )

    async def _post_generation(self, request: ImageGenerationRequest) -> httpx.Response:
        content: list[dict[str, object]] = [{"type": "text", "text": request.prompt}]
        for asset in request.source_assets:
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
                "modalities": ["image", "text"],
            },
        )

    def _url(self) -> str:
        return f"{self._api_base_url.rstrip('/')}{OPENROUTER_CHAT_COMPLETIONS_PATH}"


async def _data_url(storage_key: str, content_type: str | None) -> str:
    path = Path(storage_key)
    content = await asyncio.to_thread(path.read_bytes)
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{content_type or 'application/octet-stream'};base64,{encoded}"


def _generated_images(body: object, *, max_size_bytes: int) -> list[GeneratedImage]:
    message = _first_message(body)
    if message is None:
        return []
    images = message.get("images")
    if not isinstance(images, list):
        return []

    generated: list[GeneratedImage] = []
    for index, item in enumerate(images, start=1):
        data_url = _image_data_url(item)
        if data_url is None:
            continue
        image = _decode_generated_image(
            data_url,
            max_size_bytes=max_size_bytes,
            index=index,
        )
        generated.append(image)
    return generated


def _image_data_url(item: object) -> str | None:
    if not isinstance(item, dict):
        return None
    image_url = item.get("image_url")
    if isinstance(image_url, str):
        return image_url
    if isinstance(image_url, dict):
        url = image_url.get("url")
        if isinstance(url, str):
            return url
    return None


def _decode_generated_image(
    data_url: str,
    *,
    max_size_bytes: int,
    index: int,
) -> GeneratedImage:
    prefix, separator, encoded = data_url.partition(",")
    if separator != "," or not prefix.startswith("data:") or not prefix.endswith(";base64"):
        raise ImageGenerationError(
            "OpenRouter image generation returned an unsupported image URL",
            retryable=False,
            error_code="openrouter_unsupported_generated_image_url",
        )
    content_type = prefix.removeprefix("data:").removesuffix(";base64")
    if content_type not in SUPPORTED_GENERATED_IMAGE_CONTENT_TYPES:
        raise ImageGenerationError(
            "OpenRouter image generation returned an unsupported image content type",
            retryable=False,
            error_code="openrouter_unsupported_generated_image_content_type",
        )
    if _base64_decoded_size_upper_bound(encoded) > max_size_bytes:
        raise ImageGenerationError(
            "OpenRouter generated image exceeds configured size limit",
            retryable=False,
            error_code="openrouter_generated_image_too_large",
        )
    try:
        content = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ImageGenerationError(
            "OpenRouter generated image could not be decoded",
            retryable=False,
            error_code="openrouter_invalid_generated_image_base64",
        ) from exc
    if len(content) > max_size_bytes:
        raise ImageGenerationError(
            "OpenRouter generated image exceeds configured size limit",
            retryable=False,
            error_code="openrouter_generated_image_too_large",
        )
    return GeneratedImage(
        content=content,
        content_type=content_type,
        filename=f"generated-image-{index}{_extension_for_content_type(content_type)}",
    )


def _base64_decoded_size_upper_bound(encoded: str) -> int:
    return (len(encoded.strip()) * 3) // 4


def _extension_for_content_type(content_type: str) -> str:
    if content_type == "image/jpeg":
        return ".jpg"
    if content_type == "image/png":
        return ".png"
    if content_type == "image/webp":
        return ".webp"
    if content_type == "image/gif":
        return ".gif"
    return ".bin"


def _response_text(body: object) -> str | None:
    message = _first_message(body)
    if message is None:
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


def _first_message(body: object) -> dict[str, object] | None:
    if not isinstance(body, dict):
        return None
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if isinstance(message, dict):
        return message
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
    return response.reason_phrase or "OpenRouter image generation request failed"


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
