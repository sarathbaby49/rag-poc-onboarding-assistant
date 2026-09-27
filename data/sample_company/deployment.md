# Deployment & Environments — Acme

## Environments

| Env | Purpose | URL |
| --- | --- | --- |
| local | your laptop (Docker) | http://localhost:8000 |
| staging | pre-prod, auto-deployed from `main` | https://staging.acme.internal |
| production | live store | https://acme.example.com |

## Pipeline

- CI runs on every pull request: lint, unit tests, and the `make smoke` test.
- Merging to `main` auto-deploys to **staging**.
- Production deploys are **gated**: they need a green staging run and a one-click
  approval from the **platform team**.

## Getting deploy access

- Deploy access and production credentials are managed by the **platform team**.
- Open a request in **#platform-requests** — never share prod credentials in DMs.
- New joiners do **not** get production access on day one; you ship to staging first.

## Rolling back

- Every deploy is tagged; roll back with `acme deploy rollback <service> <tag>`.
- If a checkout error spikes after a deploy, roll back first, investigate second.
- Post in **#incidents** and page the on-call for the affected service.
