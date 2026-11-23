# Mandi price forecasting

Daily forecasts of wholesale tomato and onion prices for a small set of West Bengal Agmarknet market
series (data from data.gov.in), evaluated honestly against observed prices.

The plan is in `03-mandi-blueprint-v2.md`; milestone tasks are in `Milestone/`; what has actually been
built, with the reasons, is in one document per milestone in `docs/`.

**Status:** M01 (data experiment, go) and M02 (repo, CI, database schema) done. No ingestion or modelling yet.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync                       # create .venv and install dependencies
uv run pre-commit install     # run ruff automatically on every commit
cp .env.example .env          # then fill in the values (never commit .env)
```

## Common commands

```bash
uv run ruff check .                     # lint
uv run ruff format .                    # format
uv run pytest -rs                       # tests (-rs lists skipped tests and why)
uv run python -m mandi.store.migrate    # apply new SQL migrations to DATABASE_URL
```

Database tests run only when `TEST_DATABASE_URL` is set. Each test creates and drops its own temporary
schema, so it never touches existing tables. In CI they run against a throwaway PostgreSQL 18 container
(the same major version as Neon).

## Database

PostgreSQL on the **Neon Free plan**. Limits checked on 2026-11-23 ([Neon plans](https://neon.com/docs/introduction/plans),
[Free plan FAQ](https://neon.com/faqs/free-plan-limits-and-quotas)):

| Limit | Value | What it means here |
|---|---|---|
| Storage | 1 GB per project | Writes that grow the database fail beyond this. The M01 raw history is ~35 MB as JSON files. |
| Compute | 100 CU-hours per project per month | Ample for one daily job plus the dashboard; if used up, compute is suspended until next month. |
| Scale to zero | After 5 minutes idle, cannot be disabled | The first query after idle waits while the compute wakes up. |
| Branches | 10 per project | |
| Network transfer | 5 GB per month | |

Use Neon's **direct** connection string (host without `-pooler`) for `DATABASE_URL` and
`TEST_DATABASE_URL`. The pooled one doesn't keep session settings between transactions, which broke a
migration run during M02.

Schema changes are plain numbered SQL files in `sql/migrations/`, applied in order by
`src/mandi/store/migrate.py`. An applied file is never edited; changes go in a new numbered file.

## Secrets

`DATA_GOV_IN_API_KEY` and `DATABASE_URL` live in a local `.env` (git-ignored) and as GitHub Actions
secrets. They are never committed or printed.
