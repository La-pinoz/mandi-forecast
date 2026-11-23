import pytest

from mandi.store.db import get_connection


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_database_url_gives_a_clear_error(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", value)

    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        get_connection()
