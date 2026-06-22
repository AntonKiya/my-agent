from typing import Protocol, runtime_checkable

from agent_service.image_generation.models import ImageGenerationRequest, ImageGenerationResult


class ImageGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.error_code = error_code


class EmptyImageGenerationError(ImageGenerationError):
    """Raised when an image generation provider returns no usable image."""


@runtime_checkable
class ImageGenerator(Protocol):
    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Generate or edit images from a text prompt and optional source images."""
        ...
