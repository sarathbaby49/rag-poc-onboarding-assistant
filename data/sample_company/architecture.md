# Architecture Overview — Acme Payments Service

## High level

Requests enter through an **API gateway**, which routes to a set of **services**.
The `payments-service` is the core one and is what you'll work on first.

```
client -> API gateway -> payments-service -> PostgreSQL
                              |
                              +--> ledger-service (async, via message queue)
```

## Components

- **API gateway** — handles auth (validates JWTs), rate limiting, and routing.
- **payments-service** — Python (FastAPI). Owns payment intents and refunds.
- **ledger-service** — records immutable ledger entries. Called asynchronously
  over a **RabbitMQ** message queue so a slow ledger never blocks a payment.
- **PostgreSQL** — the primary datastore. Each service owns its own schema.

## Data flow for a payment

1. Gateway validates the JWT and forwards the request.
2. payments-service creates a `payment_intent` row (status `pending`).
3. On confirmation, status moves to `succeeded` and an event is published.
4. ledger-service consumes the event and writes a ledger entry.

## Conventions

- All money is stored in **integer minor units** (cents), never floats.
- Every table has `created_at` / `updated_at` timestamps.
- Services never share database tables — they talk over HTTP or the queue.
