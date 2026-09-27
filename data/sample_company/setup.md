# Local Development Setup — Acme Shop

Get the Acme Shop API running on your machine. Most people finish in ~15 minutes.

## Prerequisites
- Python 3.11+
- Docker Desktop (PostgreSQL + Redis run in containers)
- `make`

## Steps
1. **Clone the repo**
   ```
   git clone git@github.com:acme/shop.git
   cd shop
   ```
2. **Create a virtual environment and install dependencies**
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Copy the environment file** and fill in the required vars (table below)
   ```
   cp .env.example .env
   ```
4. **Start the datastores**
   ```
   docker compose up -d postgres redis
   ```
5. **Run migrations and seed data**
   ```
   make migrate
   make seed
   ```
6. **Start the app**
   ```
   make run
   ```
   The API is now at http://localhost:8000 — open http://localhost:8000/docs for
   the interactive API explorer.
7. **Verify it works**: `make test` (or `make smoke`) — you should see all tests pass.

## Environment variables
| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection (cart + session cache) |
| `JWT_SECRET` | signs auth tokens (see `authentication.md`) |
| `STRIPE_API_KEY` | Stripe secret key (see `payments.md`) |
| `PAYPAL_CLIENT_ID` / `PAYPAL_CLIENT_SECRET` | PayPal credentials |
| `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` | Razorpay credentials |
| `GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_SECRET` | Google social login |

Defaults in `.env.example` work for local dev except the payment-provider keys,
which you can leave blank unless you're testing checkout.

## Getting help
- Post in **#eng-help** on Slack — someone answers fast.
- If `make migrate` fails with "connection refused", the Postgres container isn't
  up yet: run `docker compose up -d postgres redis`, wait ~5 seconds, retry.
- Deployment / production access is handled by the **platform team** via
  **#platform-requests**. New joiners ship to staging first.
