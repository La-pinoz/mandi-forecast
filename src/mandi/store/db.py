"""Open a PostgreSQL connection from an environment variable.

The URL is never hard-coded or logged. Locally it comes from the project `.env` file (loaded by the
command-line entry points, not here); in GitHub Actions it comes from a repository secret.
"""

import os

import psycopg


def get_connection(env_var: str = "DATABASE_URL") -> psycopg.Connection:
    """Return a new connection to the database whose URL is in `env_var`.

    psycopg starts a transaction on the first statement; nothing is saved until the caller commits.
    Raises RuntimeError if the variable is missing or empty.
    """
    database_url = os.environ.get(env_var, "").strip()
    if not database_url:
        raise RuntimeError(f"{env_var} is not set. Add it to .env locally or as a GitHub Actions secret.")
    return psycopg.connect(database_url)
