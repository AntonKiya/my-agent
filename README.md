# Agent Service

Channel-agnostic async agent service. Telegram will be the first channel, but the service
foundation is intentionally transport-neutral: adapters, queues, workers, memory, and agent
boundaries should live behind explicit interfaces.

## Current Scope

This repository currently contains the service foundation:

- Python package with `src` layout.
- FastAPI ASGI application factory.
- Typed settings loaded from environment variables.
- Health and readiness endpoints.
- Structured JSON logging.
- Trace id context helpers.
- Lightweight service container.
- Channel-agnostic event models and queue interfaces.
- In-memory asyncio queue backend for inbound/outbound events.
- Inbound intake service that resolves users before queue publication.
- Telegram inbound webhook route and private text normalizer.
- Telegram send adapter for outbound text delivery through the Bot API.
- Background task supervisor for graceful shutdown.
- User identity domain models and bundled Postgres schema migrations.
- Runtime Postgres pool wiring for `PostgresUserStore`, `UserResolver`, and inbound intake
  when `AGENT_SERVICE_POSTGRES_DSN` is configured.
- Docker Compose Postgres service for local development and integration tests.
- Explicit database migration runner command.
- Ruff, mypy, pyright, pytest, and pytest-asyncio quality gates.

No Redis, worker, or agent logic is implemented yet.

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

For local Docker Postgres, keep these values aligned with the Postgres DSNs:

```text
POSTGRES_DB=agent_service
POSTGRES_USER=agent_service
POSTGRES_PASSWORD=agent_service_local_password
POSTGRES_TEST_DB=agent_service_test
POSTGRES_TEST_USER=agent_service_test
POSTGRES_TEST_PASSWORD=agent_service_test_local_password
AGENT_SERVICE_POSTGRES_DSN=postgresql://agent_service:agent_service_local_password@127.0.0.1:5432/agent_service
AGENT_SERVICE_TEST_POSTGRES_DSN=postgresql://agent_service_test:agent_service_test_local_password@127.0.0.1:5432/agent_service_test
```

Start local Postgres:

```bash
docker compose up -d postgres
```

Apply migrations:

```bash
uv run agent-service-db-migrate
```

Local Docker uses one application database and one isolated integration-test database:

- `agent_service` / `agent_service` for the running application.
- `agent_service_test` / `agent_service_test` for integration tests.

The application uses `AGENT_SERVICE_POSTGRES_DSN`. Integration tests use only
`AGENT_SERVICE_TEST_POSTGRES_DSN`, so test data and credentials are isolated from the app database.
The Docker init script creates only the test role and test database on first Postgres initialization.

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

Telegram webhook:

```text
POST /webhooks/telegram
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
AGENT_SERVICE_INBOUND_QUEUE_MAXSIZE=5000
AGENT_SERVICE_OUTBOUND_QUEUE_MAXSIZE=5000
AGENT_SERVICE_POSTGRES_POOL_MIN_SIZE=1
AGENT_SERVICE_POSTGRES_POOL_MAX_SIZE=10
AGENT_SERVICE_POSTGRES_COMMAND_TIMEOUT_SECONDS=30.0
```

Integration settings:

```text
POSTGRES_DB=agent_service
POSTGRES_USER=agent_service
POSTGRES_PASSWORD=agent_service_local_password
POSTGRES_TEST_DB=agent_service_test
POSTGRES_TEST_USER=agent_service_test
POSTGRES_TEST_PASSWORD=agent_service_test_local_password
AGENT_SERVICE_POSTGRES_DSN=postgresql://agent_service:agent_service_local_password@127.0.0.1:5432/agent_service
AGENT_SERVICE_TEST_POSTGRES_DSN=postgresql://agent_service_test:agent_service_test_local_password@127.0.0.1:5432/agent_service_test
AGENT_SERVICE_REDIS_DSN=
AGENT_SERVICE_TELEGRAM_BOT_TOKEN=
AGENT_SERVICE_LOGFIRE_TOKEN=
```

`AGENT_SERVICE_POSTGRES_DSN` enables runtime user resolution. Without it, supported inbound
webhooks return `503` instead of publishing unresolved events into the inbound queue.

## Inbound Flow

```text
Telegram webhook
→ Telegram normalizer
→ InboundIntakeService
→ UserResolver
→ PostgresUserStore
→ In-Memory Inbound Queue
```

The Telegram route never publishes directly to the inbound queue. Queue publication is allowed
only after user resolution succeeds and the event contains an internal `user_id`.

## Database

Bundled SQL migrations live in `src/agent_service/database/migrations`.

Current tables:

- `users`
- `channel_identities`

The schema enforces `UNIQUE(channel, external_user_id)` for stable external identity separation.
Migrations are applied by an explicit command:

```bash
uv run agent-service-db-migrate
```

The migration runner records applied migrations in `agent_service_schema_migrations`, uses a
Postgres advisory lock, validates migration checksums, and runs migrations in a transaction. The
application does not apply migrations automatically during startup; run migrations before starting
the service with a real `AGENT_SERVICE_POSTGRES_DSN`.

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
uv run pyright src tests
```

Tests:

```bash
uv run pytest
```

Integration tests that need Postgres are skipped unless `AGENT_SERVICE_TEST_POSTGRES_DSN` is set.
With the local Docker database running:

```bash
AGENT_SERVICE_TEST_POSTGRES_DSN=postgresql://agent_service_test:agent_service_test_local_password@127.0.0.1:5432/agent_service_test uv run pytest tests/test_postgres_integration.py
```

## User Identity Invariants

- Stable identity lookup is always `(channel, external_user_id)`.
- `username` is mutable metadata and must never identify or merge users.
- Inbound queue publication requires a resolved internal `user_id`.
- `blocked` and `pending` users do not reach the inbound queue.
- `UNIQUE(channel, external_user_id)` protects the Postgres boundary from duplicate identities.
- First-touch user creation must be race-safe; competing creates resolve to one identity.
- Channel-specific metadata may be updated on each successful resolve, but it must not change who
  the user is.

## Project Layout

```text
src/agent_service/
  api/                 HTTP routes such as health/readiness
  observability/       structured logs and trace context helpers
  channels/            channel-agnostic event models and adapter interfaces
  inbound/             user-resolution intake before inbound queue publication
  users/               user identity domain models and storage interfaces
  messaging/           queue interfaces and in-memory queue backend
  database/            bundled SQL migrations and database helpers
  runtime/             lifecycle helpers for background tasks
  app.py               FastAPI app factory and lifespan
  config.py            typed application settings
  container.py         infrastructure dependency assembly

tests/                 unit tests for the service shell
```
