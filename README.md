# Agent Service

Channel-agnostic async agent service. Telegram will be the first channel, but the service
foundation is intentionally transport-neutral: adapters, queues, workers, memory, and agent
boundaries should live behind explicit interfaces.

## Current State

This repository currently contains the service shell and the first channel boundary layer.
The service can accept Telegram private text updates, normalize them into internal
channel-agnostic events, and publish them to an in-memory inbound queue. It can also
send channel-agnostic outbound text events through Telegram Bot API.

- Python package with `src` layout.
- FastAPI ASGI application factory.
- Typed settings loaded from environment variables.
- Health and readiness endpoints.
- Structured JSON logging.
- Trace id context helpers.
- Lightweight service container.
- Channel-agnostic event models.
- Channel adapter, normalizer, registry, and queue interfaces.
- In-memory `asyncio.Queue` backend for inbound/outbound events.
- Telegram inbound webhook route and private text normalizer.
- Telegram send adapter for outbound text delivery through the Bot API.
- Channel adapter registry wired through the container.
- Background task supervisor for graceful shutdown.
- Ruff, mypy, pytest, and pytest-asyncio quality gates.

No Postgres, Redis, worker, or agent logic is implemented yet.

## Implemented Architecture

Current inbound path:

```text
Telegram webhook
-> TelegramInboundNormalizer
-> InboundEvent
-> AsyncioInboundQueue
```

Current outbound path:

```text
OutboundEvent
-> TelegramAdapter.send()
-> Telegram Bot API sendMessage
```

The agent processing path is intentionally not implemented yet:

```text
Inbound Worker
-> User Resolver
-> Conversation Resolver
-> ConversationMemoryService
-> Agent Boundary
-> Outbound Queue
-> Delivery Worker
```

## Channel Contracts

The channel layer currently defines:

- `InboundEvent`
- `OutboundEvent`
- `Attachment`
- `MessageType`
- `InboundEventStatus`
- `OutboundEventStatus`
- `DeliveryStatus`
- `DeliveryResult`
- `ChannelInboundNormalizer`
- `ChannelAdapter`
- `ChannelAdapterRegistry`
- `InboundQueue`
- `OutboundQueue`

These contracts are transport-neutral. Telegram-specific code must convert to/from these
models instead of leaking raw Telegram updates into the service core.

## Queues

The current queue backend is in-memory and based on `asyncio.Queue`.

- `AsyncioInboundQueue`
- `AsyncioOutboundQueue`

Both queues are bounded by default:

```text
AGENT_SERVICE_INBOUND_QUEUE_MAXSIZE=5000
AGENT_SERVICE_OUTBOUND_QUEUE_MAXSIZE=5000
```

`maxsize=0` is still allowed by `asyncio.Queue` semantics and means unbounded, but the
service defaults to bounded queues to avoid unlimited memory growth.

The queue interfaces are intentionally narrow:

```python
async def publish(event): ...
async def consume(): ...
```

This keeps the future migration path open for RabbitMQ, Kafka/Redpanda, Redis Streams,
SQS, Pub/Sub, or another durable broker.

## Telegram Inbound

Telegram inbound support is exposed as:

```text
POST /webhooks/telegram
```

Supported now:

- private one-on-one text messages;
- `update_id`;
- numeric Telegram `user.id`;
- `chat_id`;
- `message_id`;
- `username` and `first_name` as metadata only;
- future-facing fields such as `thread_id` and `reply_to_message_id`.

Unsupported updates are accepted with HTTP 200 but are not published to the inbound queue.
This includes groups, media-only messages, edited messages, replies as context, topics,
voice/audio, and other Telegram features that are outside the current MVP.

The generated idempotency key for Telegram messages is:

```text
telegram:{chat_id}:{message_id}
```

The inbound route does not require `AGENT_SERVICE_TELEGRAM_BOT_TOKEN`, because receiving
and normalizing updates is separate from sending messages.

## Telegram Outbound

Telegram outbound support is implemented by `TelegramAdapter.send()`.

Supported now:

- `OutboundEvent` with `channel="telegram"`;
- text delivery through Telegram Bot API `sendMessage`;
- splitting long messages by Telegram's 4096 character limit;
- optional `parse_mode`;
- optional future-facing `thread_id` and `reply_to_message_id`;
- basic retry for retryable Telegram/API/network failures;
- `retry_after` handling for Telegram `429`;
- `DeliveryResult` with external Telegram message ids.

Unsupported outbound payloads become `dead_letter` results before any HTTP request is made:

- wrong channel;
- empty text;
- attachments/media.

`AGENT_SERVICE_TELEGRAM_BOT_TOKEN` is required only when the container should build and
register the Telegram send adapter.

## Container Wiring

`AppContainer` owns infrastructure assembly:

- `TaskSupervisor`
- `AsyncioInboundQueue`
- `AsyncioOutboundQueue`
- `InMemoryChannelAdapterRegistry`
- `TelegramAdapter`, only when `telegram_bot_token` is configured
- the underlying Telegram `httpx.AsyncClient`, closed during shutdown

Routes, adapters, and future workers should receive dependencies from the container instead
of creating queues, clients, or registries on their own.

## Boundary Guarantees Covered By Tests

The current test suite checks that:

- Telegram webhook route is registered in the app;
- supported Telegram private text updates are published to the inbound queue;
- unsupported Telegram updates are accepted but not published;
- Telegram inbound does not require a bot token or send adapter;
- username is metadata, not identity;
- raw Telegram updates are not stored inside `InboundEvent` metadata;
- media-only inbound updates are ignored for now;
- Telegram adapter satisfies the generic `ChannelAdapter` interface;
- queue implementations satisfy the generic queue interfaces;
- invalid outbound payloads do not trigger HTTP requests;
- partial delivery is reported if a later split message fails.

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
```

Future integration settings are already reserved in `.env.example`:

```text
AGENT_SERVICE_POSTGRES_DSN=
AGENT_SERVICE_REDIS_DSN=
AGENT_SERVICE_TELEGRAM_BOT_TOKEN=
AGENT_SERVICE_LOGFIRE_TOKEN=
```

Telegram sending requires:

```text
AGENT_SERVICE_TELEGRAM_BOT_TOKEN=...
```

Do not commit real tokens or secrets.

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
  messaging/           queue interfaces and in-memory queue backend
  channels/            channel contracts, registry, and Telegram channel code
    telegram/          Telegram inbound normalizer, webhook route, send adapter
  observability/       structured logs and trace context helpers
  runtime/             lifecycle helpers for background tasks
  app.py               FastAPI app factory and lifespan
  config.py            typed application settings
  container.py         infrastructure dependency assembly

tests/                 unit tests for the service shell
```
