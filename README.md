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
- User identity domain models and Postgres storage.
- Conversation domain models, Postgres storage, and resolver.
- Per-conversation async lock manager for ordered processing inside one conversation.
- Agent boundary contracts and Pydantic AI-oriented run context shape.
- Conversation memory service contracts for Postgres history and Redis working snapshots.
- Inbound worker orchestration from inbound queue to outbound queue.
- Runtime Postgres pool wiring for `PostgresUserStore`, `UserResolver`, `PostgresConversationStore`,
  `ConversationResolver`, and inbound intake when `AGENT_SERVICE_POSTGRES_DSN` is configured.
- Guarded inbound worker runtime wiring through the task supervisor.
- Docker Compose Postgres service for local development and integration tests.
- Explicit database migration runner command.
- Ruff, mypy, pyright, pytest, and pytest-asyncio quality gates.

The agent implementation, production conversation memory backend, Redis snapshot backend, and
delivery worker are not implemented yet. Their contracts and runtime boundaries are present.

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
AGENT_SERVICE_INBOUND_WORKER_COUNT=8
AGENT_SERVICE_AGENT_RETRY_MAX_ATTEMPTS=3
AGENT_SERVICE_AGENT_RETRY_BACKOFF_SECONDS='[1.0,5.0,15.0]'
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

`AGENT_SERVICE_POSTGRES_DSN` enables runtime user resolution and conversation resolution. Without
it, supported inbound webhooks return `503` instead of publishing unresolved events into the inbound
queue.

Inbound workers are started only when the container has both a `ConversationMemoryService` and an
`AgentBoundary`. Until production implementations are wired, the service can accept and enqueue
resolved inbound events but does not start agent processing tasks.

## Inbound Flow

```text
Telegram webhook
→ Telegram normalizer
→ InboundIntakeService
→ UserResolver
→ PostgresUserStore
→ In-Memory Inbound Queue
→ InboundWorker
→ ConversationResolver
→ PostgresConversationStore
→ ConversationLockManager
→ ConversationMemoryService.record_user_message()
→ ConversationMemoryService.prepare_agent_context()
→ AgentBoundary.run()
→ ConversationMemoryService.record_assistant_message()
→ In-Memory Outbound Queue
```

The Telegram route never publishes directly to the inbound queue. Queue publication is allowed
only after user resolution succeeds and the event contains an internal `user_id`.

The inbound worker does not call Telegram or any delivery adapter. It only creates an `OutboundEvent`
and publishes it to the outbound queue. Delivery is intentionally a separate worker boundary.

`InboundWorker` uses a lock keyed by internal `conversation.id`. Messages in one conversation are
processed sequentially; different conversations may be processed concurrently by separate worker
tasks. Agent failures are retried according to `AGENT_SERVICE_AGENT_RETRY_*`; after final failure,
the worker publishes a fallback outbound event instead of sending directly.

## Agent and Memory Boundaries

Channels never pass raw transport payloads to the agent. The agent layer receives an `AgentRequest`
with internal identifiers, text/attachments, prepared context, metadata, and trace id. It does not
receive Telegram updates or delivery-specific fields.

`ConversationMemoryService` is a contract with three operations:

```text
record_user_message()
prepare_agent_context()
record_assistant_message()
```

The prepared context includes both a channel-neutral `AgentContext` and a `PydanticAIRunContext`
with fields aligned to Pydantic AI `Agent.run(...)`: `user_prompt`, `message_history`,
`conversation_id`, and `instructions`. Real Postgres history storage, Redis working snapshots, and
compression are future implementations behind this interface.

## Database

Bundled SQL migrations live in `src/agent_service/database/migrations`.

Current tables:

- `users`
- `channel_identities`
- `conversations`

The schema enforces `UNIQUE(channel, external_user_id)` for stable external identity separation.
The conversations schema enforces `UNIQUE(conversation_key)` for idempotent conversation creation.
For Telegram private chats, the derived key is:

```text
telegram:private:{external_chat_id}
```

Internally, processing uses `conversation.id` as the stable UUID. The derived `conversation_key` is
only for lookup and idempotent creation.

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

## Conversation Invariants

- `conversation.id` is the internal identifier used by workers and the agent boundary.
- `conversation_key` is a derived lookup key, unique in Postgres.
- Existing `conversation_key` rows must belong to the same `user_id`; otherwise storage raises an
  ownership error instead of mixing user data.
- One conversation is processed under one lock at a time.
- Different conversations can be processed concurrently.
- Agent requests must not include raw Telegram updates or transport delivery fields.
- Fallback responses are emitted as outbound events; delivery remains outside the inbound worker.

## Project Layout

```text
src/agent_service/
  api/                 HTTP routes such as health/readiness
  observability/       structured logs and trace context helpers
  channels/            channel-agnostic event models and adapter interfaces
  inbound/             user-resolution intake and inbound worker orchestration
  users/               user identity domain models and storage interfaces
  conversations/       conversation models, resolver, storage, and locks
  agents/              agent boundary contracts and request/response models
  memory/              conversation memory service contracts and context models
  messaging/           queue interfaces and in-memory queue backend
  database/            bundled SQL migrations and database helpers
  runtime/             lifecycle helpers for background tasks
  app.py               FastAPI app factory and lifespan
  config.py            typed application settings
  container.py         infrastructure dependency assembly

tests/                 unit tests for the service shell
```
