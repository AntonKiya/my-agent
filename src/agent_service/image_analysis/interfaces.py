from typing import Protocol, runtime_checkable

from agent_service.image_analysis.models import ImageAnalysisRequest, ImageAnalysisResult


class ImageAnalysisError(RuntimeError):
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


class EmptyImageAnalysisError(ImageAnalysisError):
    """Raised when the vision model returns no usable analysis."""


@runtime_checkable
class ImageAnalyzer(Protocol):
    async def analyze(self, request: ImageAnalysisRequest) -> ImageAnalysisResult:
        """Analyze one or more images using a vision-capable model."""
        ...
