"""Apply the numbered SQL files in sql/migrations/ to a database, in order, each exactly once.

Run from the project root:  uv run python -m mandi.store.migrate
(uses DATABASE_URL from the environment or .env).

How it works:
- Files are named `NNN_description.sql` (e.g. `001_pipeline_runs.sql`) and applied in number order.
- The table `schema_migrations` records each applied file with a checksum of its contents.
- Each file runs in its own transaction together with its `schema_migrations` row. PostgreSQL DDL is
  transactional, so a failing file leaves no half-created tables behind and is not recorded.
- An applied file must never be edited (add a new numbered file instead). If its checksum no longer
  matches, the run stops, because the database would no longer match the files.
"""

import hashlib
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from mandi.store.db import get_connection

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MIGRATIONS_DIR = PROJECT_ROOT / "sql" / "migrations"
MIGRATION_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")

CREATE_TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text PRIMARY KEY,
    checksum   text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def list_migration_files(migrations_dir: Path) -> list[Path]:
    """Return the .sql files in `migrations_dir`, sorted by their 3-digit number.

    Raises ValueError for a .sql file that doesn't follow `NNN_description.sql` or for two files
    sharing a number, since either would make the order ambiguous.
    """
    files_by_number: dict[int, Path] = {}
    for path in migrations_dir.glob("*.sql"):
        match = MIGRATION_NAME.match(path.name)
        if match is None:
            raise ValueError(f"Migration file name must look like 001_description.sql: {path.name}")
        number = int(match.group(1))
        if number in files_by_number:
            raise ValueError(
                f"Two migrations share number {number}: {files_by_number[number].name}, {path.name}"
            )
        files_by_number[number] = path
    return [files_by_number[number] for number in sorted(files_by_number)]


def file_checksum(path: Path) -> str:
    """SHA-256 of the file text with Windows line endings normalised.

    Normalising means the same file gives the same checksum on Windows (CRLF) and in CI (LF).
    """
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_migrations(conn: psycopg.Connection, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every migration not yet recorded in schema_migrations. Return the file names applied.

    Stops at the first failing file: that file is rolled back and the error is raised.
    """
    conn.execute(CREATE_TRACKING_TABLE)
    rows = conn.execute("SELECT filename, checksum FROM schema_migrations").fetchall()
    conn.commit()
    applied_checksums = {filename: checksum for filename, checksum in rows}

    newly_applied = []
    for path in list_migration_files(migrations_dir):
        checksum = file_checksum(path)
        if path.name in applied_checksums:
            if applied_checksums[path.name] != checksum:
                raise RuntimeError(
                    f"{path.name} was changed after it was applied. "
                    "Revert it and put the change in a new migration file."
                )
            continue

        try:
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT INTO schema_migrations (filename, checksum) VALUES (%s, %s)",
                (path.name, checksum),
            )
            conn.commit()
        except psycopg.Error:
            conn.rollback()
            raise
        newly_applied.append(path.name)
    return newly_applied


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    with get_connection() as conn:
        newly_applied = apply_migrations(conn)
    if newly_applied:
        print(f"Applied {len(newly_applied)} migration(s): {', '.join(newly_applied)}")
    else:
        print("Database is up to date; nothing applied.")


if __name__ == "__main__":
    main()
