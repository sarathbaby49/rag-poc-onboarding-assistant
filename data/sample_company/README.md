# Acme Shop

Acme Shop is an online store (e-commerce) backend. It's a Python **FastAPI**
service backed by **PostgreSQL** and **Redis**, run locally with Docker.

## What's here
- Product catalog, cart, checkout and orders
- **Payments** via multiple providers (Stripe, PayPal, Razorpay) — see `payments.md`
- **Authentication** — JWT with Google social login — see `authentication.md`

## Run it locally
Full steps are in `setup.md`. In short: clone → create a virtualenv →
`pip install -r requirements.txt` → `cp .env.example .env` →
`docker compose up -d postgres redis` → `make migrate` → `make run`
(the API serves at http://localhost:8000, interactive docs at /docs).

## Where to ask
`#eng-help` on Slack for engineering questions; `#platform-requests` for
deploy/prod access. New joiners ship to staging first.
