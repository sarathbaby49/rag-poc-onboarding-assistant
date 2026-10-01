# Architecture — Acme Shop

Acme Shop is a modular **FastAPI** monolith backed by **PostgreSQL** (orders,
users, catalog) and **Redis** (cart, session cache, rate-limiting).

## Request flow
Browser / mobile → **API gateway** (FastAPI) → service modules → DB / cache.

## Key modules
| Module | Responsibility |
| --- | --- |
| `catalog` | Product CRUD, search, categories |
| `cart` | Redis-backed cart with TTL |
| `checkout` | Order creation, payment orchestration |
| `auth` | JWT login, Google OAuth (see `authentication.md`) |
| `webhooks` | Stripe / PayPal / Razorpay callbacks |

## Database
- PostgreSQL 15, managed with **Alembic** migrations (`make migrate`).
- Every table has `created_at` / `updated_at` timestamps.
- Foreign keys enforce referential integrity; soft-deletes for orders.

## Caching
- Redis stores carts (hash per user, 24 h TTL) and session data.
- Cache invalidation is explicit — no automatic TTL on catalog data.
