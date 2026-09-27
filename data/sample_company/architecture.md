# Architecture Overview — Acme E-commerce Platform

Acme is an online store. The platform is a set of small **services** behind an
**API gateway**; the `payments-service` is the one most new joiners start on, but
it's one service among several.

## High level

```
        browser / mobile
              |
         API gateway  (auth, rate limiting, routing)
              |
  ┌───────────┼───────────────┬───────────────┬───────────────┐
  |           |               |               |               |
catalog-   cart-         checkout /       payments-       search-
service    service       orders-service   service         service
  |           |               |               |               |
Postgres   Redis          Postgres        Postgres      OpenSearch
                              |
                         RabbitMQ events ──> inventory-service, ledger-service,
                                             notifications-service
```

## Components

- **API gateway** — validates JWTs (see `code/auth.py`), rate limits, routes.
- **catalog-service** — products & categories (`code/catalog.py`). Read-heavy;
  hot products cached in Redis. Search is delegated to search-service.
- **search-service** — product search over **OpenSearch**, with faceted filters.
- **cart-service** — the shopping cart (`code/cart.py`). Carts live in **Redis**
  with a 7-day TTL.
- **checkout / orders-service** — the checkout saga (`code/checkout.py`) and the
  order lifecycle (`code/orders.py`). Owns `orders` in PostgreSQL.
- **inventory-service** — stock levels and reservations (`code/inventory.py`);
  prevents overselling.
- **payments-service** — Python (FastAPI). Owns payment intents and refunds.
- **ledger-service** — immutable ledger entries, written asynchronously off a
  **RabbitMQ** queue so a slow ledger never blocks a checkout.
- **notifications-service** — order emails / SMS, also event-driven.
- **PostgreSQL** — the primary datastore. Each service owns its own schema.

## Data flow for a purchase

1. Shopper builds a **cart** (cart-service, Redis).
2. **Checkout** reserves stock (inventory-service), then charges the shopper
   (payments-service), then creates an **order** in `pending` (orders-service).
3. On payment success the order moves `pending -> paid` and an event is published.
4. inventory-service commits the reservation; ledger-service writes an entry;
   notifications-service emails a receipt.
5. Fulfilment moves the order `paid -> fulfilled -> shipped -> delivered`.

## Conventions

- All money is stored in **integer minor units** (cents), never floats.
- Every table has `created_at` / `updated_at` timestamps.
- Services never share database tables — they talk over HTTP or the queue.
- Checkout is **idempotent** on an `Idempotency-Key` header, so a retried
  purchase never double-charges or double-orders.
