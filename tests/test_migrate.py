from pathlib import Path

import psycopg
import pytest

from mandi.store.migrate import MIGRATIONS_DIR, apply_migrations, file_checksum, list_migration_files

EXPECTED_TABLES = {
    "schema_migrations",
    "pipeline_runs",
    "raw_snapshots",
    "series",
    "price_vintages",
    "prices_current",
    "series_day_status",
    "quarantine",
    "reporting_gaps_by_date",  # a view (information_schema.tables lists views too)
}


def write_file(directory: Path, name: str, text: str) -> None:
    (directory / name).write_text(text, encoding="utf-8")


def tables_in_current_schema(conn: psycopg.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
    ).fetchall()
    return {row[0] for row in rows}


# --- File listing and checksums (no database needed) ---


def test_files_are_ordered_by_number_not_by_name(tmp_path):
    write_file(tmp_path, "010_zzz.sql", "")
    write_file(tmp_path, "002_bbb.sql", "")
    write_file(tmp_path, "001_ccc.sql", "")
    write_file(tmp_path, "notes.txt", "")  # non-.sql files are ignored

    names = [path.name for path in list_migration_files(tmp_path)]

    assert names == ["001_ccc.sql", "002_bbb.sql", "010_zzz.sql"]


@pytest.mark.parametrize("bad_name", ["1_short_number.sql", "001-dash.sql", "001_Upper.sql", "create.sql"])
def test_badly_named_file_is_rejected(tmp_path, bad_name):
    write_file(tmp_path, bad_name, "")

    with pytest.raises(ValueError, match="must look like"):
        list_migration_files(tmp_path)


def test_two_files_with_the_same_number_are_rejected(tmp_path):
    write_file(tmp_path, "003_first.sql", "")
    write_file(tmp_path, "003_second.sql", "")

    with pytest.raises(ValueError, match="share number 3"):
        list_migration_files(tmp_path)


def test_checksum_ignores_windows_line_endings(tmp_path):
    (tmp_path / "001_lf.sql").write_bytes(b"CREATE TABLE a (x int);\nSELECT 1;\n")
    (tmp_path / "002_crlf.sql").write_bytes(b"CREATE TABLE a (x int);\r\nSELECT 1;\r\n")

    assert file_checksum(tmp_path / "001_lf.sql") == file_checksum(tmp_path / "002_crlf.sql")


def test_project_migrations_are_validly_named():
    files = list_migration_files(MIGRATIONS_DIR)

    assert files[0].name.startswith("001_")
    assert len(files) == 8


# --- Applying migrations (needs TEST_DATABASE_URL) ---


def test_project_migrations_apply_cleanly_to_a_fresh_schema(db_conn):
    applied = apply_migrations(db_conn)

    assert applied == [path.name for path in list_migration_files(MIGRATIONS_DIR)]
    assert tables_in_current_schema(db_conn) == EXPECTED_TABLES


def test_second_run_applies_nothing(db_conn):
    apply_migrations(db_conn)

    assert apply_migrations(db_conn) == []


def test_failing_migration_is_rolled_back_and_not_recorded(db_conn, tmp_path):
    write_file(tmp_path, "001_good.sql", "CREATE TABLE good (x int);")
    write_file(tmp_path, "002_bad.sql", "CREATE TABLE half_done (x int);\nTHIS IS NOT SQL;")
    write_file(tmp_path, "003_never_reached.sql", "CREATE TABLE never_reached (x int);")

    with pytest.raises(psycopg.errors.SyntaxError):
        apply_migrations(db_conn, tmp_path)

    assert tables_in_current_schema(db_conn) == {"schema_migrations", "good"}
    recorded = db_conn.execute("SELECT filename FROM schema_migrations").fetchall()
    assert recorded == [("001_good.sql",)]


def test_fixed_migration_is_applied_on_the_next_run(db_conn, tmp_path):
    write_file(tmp_path, "001_table.sql", "CREATE TABLE t (x int) TYPO;")
    with pytest.raises(psycopg.errors.SyntaxError):
        apply_migrations(db_conn, tmp_path)

    write_file(tmp_path, "001_table.sql", "CREATE TABLE t (x int);")

    assert apply_migrations(db_conn, tmp_path) == ["001_table.sql"]


def test_editing_an_applied_migration_stops_the_run(db_conn, tmp_path):
    write_file(tmp_path, "001_table.sql", "CREATE TABLE t (x int);")
    apply_migrations(db_conn, tmp_path)

    write_file(tmp_path, "001_table.sql", "CREATE TABLE t (x int, y int);")

    with pytest.raises(RuntimeError, match="changed after it was applied"):
        apply_migrations(db_conn, tmp_path)
