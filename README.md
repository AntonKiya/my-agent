# Agent Service

Channel-agnostic async agent service. Telegram will be the first channel, but the service
foundation is intentionally transport-neutral: adapters, queues, workers, memory, and agent
boundaries should live behind explicit interfaces.

## Current Scope

This repository currently contains the step-0 service shell:

- Python package with `src` layout.
- FastAPI ASGI application factory.
- Typed settings loaded from environment variables.
- Health and readiness endpoints.
- Structured JSON logging.
- Trace id context helpers.
- Lightweight service container.
- Background task supervisor for graceful shutdown.
- Ruff, mypy, pytest, and pytest-asyncio quality gates.

No Telegram, Postgres, Redis, queue, or agent logic is implemented yet.

## Requirements

- Python 3.12+
- uv

## Setup

Install dependencies:

```bash
uv sync
```

Create local environment file when needed:

```bash
cp .env.example .env
```

Real secrets must stay in local environment variables or secret storage, not in git.

## Run

Start the service:

```bash
uv run agent-service
```

Or:

```bash
uv run python -m agent_service
```

Default local URL:

```text
http://127.0.0.1:8000
```

Health checks:

```bash
curl -L http://127.0.0.1:8000/health
curl -L http://127.0.0.1:8000/ready
```

## Configuration

Settings are loaded from environment variables with the `AGENT_SERVICE_` prefix.

Common local settings:

```text
AGENT_SERVICE_SERVICE_NAME=agent-service
AGENT_SERVICE_ENVIRONMENT=local
AGENT_SERVICE_DEBUG=false
AGENT_SERVICE_HOST=0.0.0.0
AGENT_SERVICE_PORT=8000
AGENT_SERVICE_LOG_LEVEL=INFO
AGENT_SERVICE_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS=10.0
```

Future integration settings are already reserved in `.env.example`:

```text
AGENT_SERVICE_POSTGRES_DSN=
AGENT_SERVICE_REDIS_DSN=
AGENT_SERVICE_TELEGRAM_BOT_TOKEN=
AGENT_SERVICE_LOGFIRE_TOKEN=
```

## Quality Gate

Format:

```bash
uv run ruff format .
```

Lint:

```bash
uv run ruff check .
```

Type check:

```bash
uv run mypy src tests
```

Tests:

```bash
uv run pytest
```

## Project Layout

```text
src/agent_service/
  api/                 HTTP routes such as health/readiness
  observability/       structured logs and trace context helpers
  runtime/             lifecycle helpers for background tasks
  app.py               FastAPI app factory and lifespan
  config.py            typed application settings
  container.py         infrastructure dependency assembly

tests/                 unit tests for the service shell
```
