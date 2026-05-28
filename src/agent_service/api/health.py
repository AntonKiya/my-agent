from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from agent_service.config import AppSettings, get_settings

router = APIRouter(tags=["health"])
SettingsDependency = Annotated[AppSettings, Depends(get_settings)]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    environment: str


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDependency) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        environment=settings.environment,
    )


@router.get("/ready", response_model=HealthResponse)
async def ready(settings: SettingsDependency) -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.service_name,
        environment=settings.environment,
    )
