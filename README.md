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
- Log-based observability events with trace ids, safe metadata, and operation durations.
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
- Pydantic AI agent boundary implementation with OpenRouter model factory, run timeout, safe text
  output normalization, and usage mapping.
- Explicit service-facing agent boundary safety contract: raw transport metadata is not passed into
  Pydantic AI run metadata, attachment inputs are rejected for the text-only MVP, and empty or
  non-text model outputs are rejected before an `AgentResponse` is emitted.
- Conversation memory service implementation for Postgres history and Redis working snapshots.
- Persistent `conversation_summaries` state model and Postgres compaction store contract.
- Explicit separation between memory state ownership and the external conversation compactor boundary.
- In-memory compaction queue and compaction worker that run under the same per-conversation lock.
- Pydantic AI conversation compactor contract with structured summary output and fixed rendered
  summary format.
- Inbound worker orchestration from inbound queue to outbound queue.
- Runtime Postgres pool wiring for `PostgresUserStore`, `UserResolver`, `PostgresConversationStore`,
  `ConversationResolver`, and inbound intake when `AGENT_SERVICE_POSTGRES_DSN` is configured.
- Runtime OpenRouter agent boundary wiring when `AGENT_SERVICE_AGENT_PROVIDER`,
  `AGENT_SERVICE_AGENT_MODEL`, and `AGENT_SERVICE_OPENROUTER_API_KEY` are configured.
- Guarded inbound worker runtime wiring through the task supervisor.
- Docker Compose Postgres and Redis services for local development and integration tests.
- Explicit database migration runner command.
- Ruff, mypy, pyright, pytest, and pytest-asyncio quality gates.

The delivery worker and real compaction/summarization implementation are not implemented yet. Their
contracts, runtime boundaries, and persistent summary state foundation are present.

## Implemented Guarantees

- Channel adapters do not call the agent directly.
- Inbound queue publication happens only after user resolution succeeds.
- Stable user identity is based on `(channel, external_user_id)`, not username.
- Conversations use internal UUIDs for processing and derived `conversation_key` values for lookup.
- One conversation is processed sequentially under a per-conversation lock.
- Different conversations can be processed concurrently by async worker tasks.
- User and assistant messages are persisted in Postgres with monotonic per-conversation sequence.
- Redis stores only hot working context snapshots; Postgres remains the source of truth.
- Redis snapshots are validated against Postgres `message_sequence` before use.
- Redis snapshot rebuilds restore the latest completed Postgres summary plus recent messages after
  `last_compacted_sequence`.
- Conversation summaries are modeled as per-conversation, per-user, sequence-bounded state.
- The Postgres compaction store enforces conversation/user ownership before summary persistence.
- Completed compaction is idempotent by `(conversation_id, user_id, to_sequence)`: duplicate jobs
  return the existing summary instead of creating a second completed summary.
- Compaction policy is token-budget based: trigger threshold and retained recent tail are computed
  from manually configured model context limits.
- Conversation memory prepares safe compaction requests and records compaction results; compactors do
  not own Postgres, Redis, or active context snapshots.
- Compaction prompts receive only user/assistant messages; tool calls and tool results are filtered
  out before model input.
- Compaction is scheduled only after a successful agent run and persisted assistant message, so
  multi-tool agent runs complete before memory compression is considered.
- Pydantic AI receives clean `user_prompt`, real `ModelMessage` history, and no raw transport update.
- Tool calls/results can be included in active context but are rejected from compaction input.
- Bounded inbound queues apply backpressure and return overload instead of silently dropping events.
- Worker loops survive single-event failures and continue processing the next queued event.

## Requirements

- Python 3.12+
- uv
- Docker Compose for local Postgres and Redis

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

For local Docker Postgres and Redis, keep these values aligned with the DSNs:

```text
POSTGRES_DB=agent_service
POSTGRES_USER=agent_service
POSTGRES_PASSWORD=agent_service_local_password
POSTGRES_TEST_DB=agent_service_test
POSTGRES_TEST_USER=agent_service_test
POSTGRES_TEST_PASSWORD=agent_service_test_local_password
AGENT_SERVICE_POSTGRES_DSN=postgresql://agent_service:agent_service_local_password@127.0.0.1:5432/agent_service
AGENT_SERVICE_TEST_POSTGRES_DSN=postgresql://agent_service_test:agent_service_test_local_password@127.0.0.1:5432/agent_service_test
AGENT_SERVICE_REDIS_DSN=redis://127.0.0.1:6379/0
```

Start local Postgres and Redis:

```bash
docker compose up -d postgres redis
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
AGENT_SERVICE_INBOUND_PUBLISH_TIMEOUT_SECONDS=1.0
AGENT_SERVICE_INBOUND_WORKER_COUNT=8
AGENT_SERVICE_INBOUND_WORKER_ERROR_BACKOFF_SECONDS=0.1
AGENT_SERVICE_AGENT_RETRY_MAX_ATTEMPTS=3
AGENT_SERVICE_AGENT_RETRY_BACKOFF_SECONDS='[1.0,5.0,15.0]'
AGENT_SERVICE_AGENT_PROVIDER=
AGENT_SERVICE_AGENT_MODEL=
AGENT_SERVICE_AGENT_TIMEOUT_SECONDS=60.0
AGENT_SERVICE_POSTGRES_POOL_MIN_SIZE=1
AGENT_SERVICE_POSTGRES_POOL_MAX_SIZE=10
AGENT_SERVICE_POSTGRES_COMMAND_TIMEOUT_SECONDS=30.0
AGENT_SERVICE_REDIS_CONTEXT_SNAPSHOT_TTL_SECONDS=86400
AGENT_SERVICE_MEMORY_COMPACTION_ENABLED=false
AGENT_SERVICE_MEMORY_MODEL_CONTEXT_WINDOW_TOKENS=196600
AGENT_SERVICE_MEMORY_RESERVED_OUTPUT_TOKENS=16384
AGENT_SERVICE_MEMORY_COMPACTION_TRIGGER_FRACTION=0.80
AGENT_SERVICE_MEMORY_RECENT_TAIL_FRACTION=0.30
AGENT_SERVICE_MEMORY_COMPACTION_QUEUE_MAXSIZE=1000
AGENT_SERVICE_MEMORY_COMPACTION_WORKER_COUNT=0
AGENT_SERVICE_MEMORY_COMPACTION_WORKER_ERROR_BACKOFF_SECONDS=0.1
AGENT_SERVICE_MEMORY_COMPACTION_PUBLISH_TIMEOUT_SECONDS=0.1
AGENT_SERVICE_MEMORY_COMPACTION_TARGET_SUMMARY_TOKENS=1000
AGENT_SERVICE_MEMORY_COMPACTION_MODEL=
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
AGENT_SERVICE_REDIS_DSN=redis://127.0.0.1:6379/0
AGENT_SERVICE_TELEGRAM_BOT_TOKEN=
AGENT_SERVICE_OPENROUTER_API_KEY=
AGENT_SERVICE_LOGFIRE_TOKEN=
```

`AGENT_SERVICE_POSTGRES_DSN` enables runtime user resolution and conversation resolution. Without
it, supported inbound webhooks return `503` instead of publishing unresolved events into the inbound
queue.

`AGENT_SERVICE_AGENT_PROVIDER=openrouter` selects OpenRouter-backed agent configuration. The
container creates a managed `PydanticAIAgentBoundary` only when `AGENT_SERVICE_AGENT_MODEL` and
`AGENT_SERVICE_OPENROUTER_API_KEY` are also configured. Without the full agent configuration, inbound
webhooks can still resolve users and publish events, but inbound workers will not start because
there is no agent boundary. `AGENT_SERVICE_MEMORY_COMPACTION_MODEL` is intentionally separate from
the main chat model so conversation summarization can use a cheaper or more specialized model later.

`AGENT_SERVICE_REDIS_DSN` enables Redis working context snapshots. When configured, the container
creates a Redis client on startup, pings it, and wires `RedisConversationContextSnapshotStore` into
`DefaultConversationMemoryService`. If Redis is not configured, context still works from Postgres
history, but without the hot snapshot cache.

Inbound workers are started only when the container has both a `ConversationMemoryService` and an
`AgentBoundary`. With Postgres configured and a complete OpenRouter agent configuration, the
container starts the configured inbound worker tasks automatically.

Compaction workers are started only when compaction is enabled, worker count is greater than zero,
and a real `ConversationCompactor` implementation is wired. Until then, the policy and queue
contracts are present but compaction processing is inert.

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
→ optional ConversationCompactionPolicy decision
→ optional In-Memory Compaction Queue
→ ConversationCompactionWorker
→ ConversationLockManager
→ ConversationMemoryService.prepare_compaction_request()
→ ConversationCompactor.compact()
→ ConversationMemoryService.record_compaction_result()
```

The Telegram route never publishes directly to the inbound queue. Queue publication is allowed
only after user resolution succeeds and the event contains an internal `user_id`.

The inbound worker does not call Telegram or any delivery adapter. It only creates an `OutboundEvent`
and publishes it to the outbound queue. Delivery is intentionally a separate worker boundary.

`InboundWorker` uses a lock keyed by internal `conversation.id`. Messages in one conversation are
processed sequentially; different conversations may be processed concurrently by separate worker
tasks. Agent failures are retried according to `AGENT_SERVICE_AGENT_RETRY_*`; after final failure,
the worker publishes a fallback outbound event instead of sending directly.

Inbound queues are bounded by `AGENT_SERVICE_INBOUND_QUEUE_MAXSIZE`. The intake layer publishes with
`AGENT_SERVICE_INBOUND_PUBLISH_TIMEOUT_SECONDS`; if the queue stays full past that timeout, the
event is not silently dropped. Telegram receives `503`, allowing the transport to retry instead of
holding the webhook request indefinitely.

Worker loops are resilient to single-event failures. `run_forever()` logs unexpected event
processing errors, waits `AGENT_SERVICE_INBOUND_WORKER_ERROR_BACKOFF_SECONDS`, and continues with
the next queued event. Cancellation still propagates normally during shutdown.

## Agent and Memory Boundaries

Channels never pass raw transport payloads to the agent. The agent layer receives an `AgentRequest`
with internal identifiers, text/attachments, prepared context, metadata, and trace id. It does not
receive Telegram updates or delivery-specific fields.

Pydantic AI owns the model-facing contract: typed `ModelMessage` history, `output_type`, provider
abstraction, usage reporting, and structured output support. The service boundary owns the
domain-facing contract around it: only allowlisted operational metadata is passed to
`Agent.run(...)`, attachments are rejected until non-text handling is implemented, provider errors
propagate to the inbound worker retry/fallback policy, and model output must normalize to non-empty
text before it becomes an `AgentResponse`.

The agent boundary applies `AGENT_SERVICE_AGENT_TIMEOUT_SECONDS` around the provider run. Provider
timeouts, provider errors, validation errors, and empty-output errors are not converted into
user-facing text in the boundary. They propagate to `InboundWorker`, which applies
`AGENT_SERVICE_AGENT_RETRY_*`; after the final failed attempt it publishes a fallback
`OutboundEvent` and leaves direct delivery to the delivery layer. Process-level parallelism is
controlled by `AGENT_SERVICE_INBOUND_WORKER_COUNT`; the boundary does not add a second global
concurrency gate on top of the worker model.

`ConversationMemoryService` is a contract with three operations:

```text
record_user_message()
prepare_agent_context()
record_assistant_message()
```

The prepared context includes both a channel-neutral `AgentContext` and a `PydanticAIRunContext`
with fields aligned to Pydantic AI `Agent.run(...)`: `user_prompt`, `message_history`,
`conversation_id`, and `instructions`. `message_history` contains real
`pydantic_ai.messages.ModelMessage` objects, not transport payloads or ad hoc dicts.

`DefaultConversationMemoryService` combines Postgres message history with an optional Redis working
snapshot. It writes the user message before the agent run, prepares context, then writes the
assistant message after a successful response. `AgentBoundary` does not write history directly.

Redis snapshots are never trusted only by key existence. Before a snapshot is used, the service reads
the current Postgres `message_sequence` for that conversation and compares it to
`snapshot.last_seen_sequence`. If the values differ, or if the snapshot belongs to another user,
conversation, version, or does not contain the latest user message, the snapshot is deleted and
rebuilt from recent Postgres history. This keeps the freshness check cheap while preserving Postgres
as the source of truth.

Tool calls and tool results are included in active context and Pydantic AI message history, but are
modeled as separate roles so future summarization can exclude them from compacted summaries.

Compaction currently exists as an interface and stub, not as a real summarization implementation.
`ConversationCompactionRequest` accepts only `user` and `assistant` messages; `tool_call` and
`tool_result` messages are intentionally rejected from compaction input. `NoopConversationCompactor`
preserves the previous summary and does not advance `last_compacted_sequence`, so wiring the
boundary cannot silently alter context behavior before a real summarizer is implemented.

## Database

Bundled SQL migrations live in `src/agent_service/database/migrations`.

Current tables:

- `users`
- `channel_identities`
- `conversations`
- `conversation_messages`

The schema enforces `UNIQUE(channel, external_user_id)` for stable external identity separation.
The conversations schema enforces `UNIQUE(conversation_key)` for idempotent conversation creation.
For Telegram private chats, the derived key is:

```text
telegram:private:{external_chat_id}
```

Internally, processing uses `conversation.id` as the stable UUID. The derived `conversation_key` is
only for lookup and idempotent creation.

Message history is stored in `conversation_messages`. Ordering is represented by
`conversation_messages.sequence`, which is unique per conversation. Tool calls and tool results are
stored as message roles so they can be passed into active context while future summarization can
exclude them from compacted summaries.

Redis working context snapshots use keys shaped as `conversation_context:{conversation_id}` with a
24-hour default TTL. Redis is a hot cache only; Postgres remains the source of truth.

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
uv run ruff check src tests
```

Type check:

```bash
uv run mypy src tests
uv run pyright
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

The test suite covers user identity separation, conversation ownership/order, Postgres message
sequence assignment, Redis snapshot validation, Pydantic AI context conversion, Pydantic AI
AgentBoundary safety and timeout behavior, OpenRouter container wiring, compaction invariants and
idempotency, safe structured observability events, queue backpressure, worker resilience, and
Telegram webhook overload behavior.

## Observability

The current observability layer intentionally uses standard Python logging, not Logfire. Logs are
JSON-formatted and include the active `trace_id` from context when one is available.

Workers emit structured, low-PII events around the important runtime boundaries:

- inbound intake publish, reject, and overload;
- inbound event processing success/failure;
- agent run success and retry scheduling;
- outbound/fallback event publication;
- compaction scheduling, skip, completion, and failure.

Observability fields should stay safe and operational: internal ids, channel, status, attempt,
queue size, token counts, sequence boundaries, and `duration_ms`. Raw Telegram updates, message
text, tool payloads, and full prompts should not be logged through these events.

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
