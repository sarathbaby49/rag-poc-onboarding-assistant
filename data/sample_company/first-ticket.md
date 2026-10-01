# Your First Ticket — Acme Shop

Your first ticket is intentionally small so you can learn the workflow without
pressure. Expect to finish it within your first week.

## Typical first tickets
- **Add a field** to an API response (e.g. `discount_pct` on `/products/:id`).
- **Write a missing test** for an existing endpoint.
- **Fix a typo** in user-facing copy or docs.

## Workflow
1. Pick a ticket labelled **good-first-issue** from the board.
2. Create a branch: `git checkout -b <your-name>/<ticket-id>`.
3. Make the change + add/update tests.
4. Open a PR — CI will run automatically (see `ci.md`).
5. Request review from your mentor.
6. After approval, merge via **squash-and-merge**.

## Code review norms
- Reviewers aim for < 24 h turnaround.
- "Nit" comments are optional; "blocking" comments must be resolved.
- If tests are green and one approval is given, you may merge.
