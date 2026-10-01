# CI / CD Pipeline — Acme Shop

All merges to `main` trigger a **GitHub Actions** pipeline that builds, tests,
and deploys to staging automatically.

## Pipeline stages
1. **Lint** — `ruff check .` + `mypy` strict mode.
2. **Unit tests** — `pytest -x --cov` with a 90 % coverage gate.
3. **Build** — Docker image built and pushed to ECR.
4. **Deploy to staging** — Helm chart applied to the staging k8s cluster.
5. **Smoke tests** — a small Playwright suite hits staging endpoints.
6. **Manual gate** — a team lead approves promotion to production.
7. **Deploy to prod** — same Helm chart, production cluster.

## Running locally
```bash
make lint     # ruff + mypy
make test     # pytest with coverage
make build    # Docker build (optional, CI does this)
```

## Rollback
Every deploy tags the Git SHA. To roll back: re-run the deploy job for the
previous tag. Redis caches are flushed automatically on deploy.
