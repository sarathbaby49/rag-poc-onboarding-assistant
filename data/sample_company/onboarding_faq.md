# New Joiner FAQ — Acme Engineering

**Where do I start?**
Get `payments-service` running locally (see setup.md), then read architecture.md
for how the services fit together. Your first ticket is usually a `good-first-issue`.

**Which service should I learn first?**
Most people start on `payments-service` or `orders-service`. The checkout saga
(`code/checkout.py`) is the best single file for seeing how services connect.

**Why is everything in cents?**
Floats lose pennies. All money is integer minor units (cents); we divide only for
display. This bit us historically — see ticket PAY-063.

**How do I run the tests?**
`make smoke` for the smoke test (expect `OK: 12 passed`); `make test` for the full
suite. If the login test is flaky locally, it's PAY-101, not your machine.

**My `make migrate` fails with "connection refused" — help?**
The Postgres container isn't up yet. Run `docker compose up -d postgres`, wait ~5s,
retry.

**How do I get deploy / production access?**
Ask the platform team in #platform-requests. New joiners ship to staging first.

**Where do I ask for help?**
`#eng-help` for engineering, `#platform-requests` for access/infra, `#incidents`
for anything customer-facing that's broken. Your onboarding buddy can pair with you.

**What's an Idempotency-Key and why do I keep seeing it?**
It makes a retried checkout safe (no double orders / double charges). See orders.md.
