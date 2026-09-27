# Local Development Setup — Acme Payments Service

Welcome! Follow these steps to get the `payments-service` running locally.
Most people finish in about 30 minutes.

## Prerequisites

- Python 3.11+
- Docker Desktop (for the local PostgreSQL database)
- Access to the internal package registry (ask your onboarding buddy)

## Steps

1. **Clone the repo**
   ```
   git clone git@github.com:acme/payments-service.git
   cd payments-service
   ```

2. **Create a virtual environment and install dependencies**
   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Copy the environment file**
   ```
   cp .env.example .env
   ```
   Fill in `DATABASE_URL` and `JWT_SECRET`. Defaults in `.env.example` work for local dev.

4. **Start the database**
   ```
   docker compose up -d postgres
   ```

5. **Run migrations and start the app**
   ```
   make migrate
   make run
   ```
   The API is now at http://localhost:8000. Open http://localhost:8000/docs for the API explorer.

## Verifying it works

Run the smoke test: `make smoke`. You should see `OK: 12 passed`.

## Getting help

- Post in the **#eng-help** Slack channel — someone answers fast.
- For **deployment access or production credentials**, ask the **platform team** (they own the deploy pipeline).
- Your onboarding buddy is assigned on day one; they can pair with you on setup.
