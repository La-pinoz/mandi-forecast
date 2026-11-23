"""Fetching from data.gov.in (ingest/ogd.py) without the network: `request_once` is replaced by a fake that
returns prepared (status, body) answers in order, so paging, retries and error handling can be checked."""

import json
from datetime import date
from urllib.error import URLError

import pytest

from mandi.ingest import ogd
from mandi.store.raw_snapshots import payload_hash

API_KEY = "secret-test-key"


def page(total: int, records: list[dict]) -> tuple[int, str]:
    return 200, json.dumps({"total": total, "updated_date": "2026-11-22T23:34:48Z", "records": records})


def row(arrival: str = "21/11/2026", commodity: str = "Tomato", market: str = "Barasat APMC") -> dict:
    return {"State": "West Bengal", "Commodity": commodity, "Arrival_Date": arrival, "Market": market}


class FakeApi:
    """Stands in for `request_once`: returns `answers` in order (an Exception is raised instead) and keeps
    the params of each call in `calls`."""

    def __init__(self) -> None:
        self.answers: list = []
        self.calls: list[dict] = []

    def request_once(self, params: dict) -> tuple[int, str]:
        self.calls.append(params)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> FakeApi:
    api = FakeApi()
    monkeypatch.setattr(ogd, "request_once", api.request_once)
    monkeypatch.setattr(ogd, "SECONDS_BETWEEN_CALLS", 0)
    monkeypatch.setattr(ogd, "RETRY_WAITS_SECONDS", [0, 0, 0])
    return api


def test_fetch_date_pages_until_total_and_keeps_key_out_of_stored_params(fake_api, monkeypatch) -> None:
    monkeypatch.setattr(ogd, "PAGE_SIZE", 2)
    fake_api.answers.extend(
        [page(3, [row(market="A"), row(market="B")]), page(3, [row(market="C")])],
    )

    snapshot = ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)

    calls = fake_api.calls
    assert [call["offset"] for call in calls] == [0, 2]
    assert all(call["api-key"] == API_KEY for call in calls)
    assert [record["Market"] for record in snapshot.payload["records"]] == ["A", "B", "C"]
    assert snapshot.payload["total"] == 3
    assert snapshot.payload["pages"] == 2
    assert snapshot.payload["updated_date"] == "2026-11-22T23:34:48Z"
    assert snapshot.request_params == {
        "filters[State]": "West Bengal",
        "filters[Commodity]": "Tomato",
        "filters[Arrival_Date]": "21/11/2026",
    }
    assert snapshot.fetched_at.utcoffset().total_seconds() == 5.5 * 3600


def test_zero_rows_is_a_valid_response(fake_api) -> None:
    fake_api.answers.append(page(0, []))
    snapshot = ogd.fetch_date("Onion", date(2026, 11, 23), API_KEY)
    assert snapshot.payload["records"] == []
    assert snapshot.payload["pages"] == 1


def test_server_error_is_retried(fake_api) -> None:
    fake_api.answers.extend([(502, "Bad Gateway"), page(1, [row()])])
    snapshot = ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)
    assert len(fake_api.calls) == 2
    assert snapshot.payload["total"] == 1


def test_non_json_body_and_body_without_total_are_retried(fake_api) -> None:
    fake_api.answers.extend(
        [(200, "<html>maintenance</html>"), (200, '{"message": "busy"}'), page(1, [row()])]
    )
    snapshot = ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)
    assert len(fake_api.calls) == 3
    assert snapshot.payload["total"] == 1


def test_gives_up_after_all_retries(fake_api) -> None:
    fake_api.answers.extend([(503, ""), (503, ""), (429, ""), (502, "")])
    with pytest.raises(ogd.FetchError, match="HTTP 502"):
        ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)
    assert len(fake_api.calls) == 4


def test_network_error_message_never_contains_the_key(fake_api) -> None:
    error = URLError(f"connection reset for ?api-key={API_KEY}")
    fake_api.answers.extend([error, error, error, error])
    with pytest.raises(ogd.FetchError) as raised:
        ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)
    assert API_KEY not in str(raised.value)
    assert "<API_KEY>" in str(raised.value)


def test_client_error_stops_at_once_without_leaking_the_key(fake_api) -> None:
    fake_api.answers.append((403, f'{{"error": "Key not authorised: {API_KEY}"}}'))
    with pytest.raises(RuntimeError, match="HTTP 403") as raised:
        ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)
    assert API_KEY not in str(raised.value)
    assert len(fake_api.calls) == 1


def test_row_for_another_date_is_an_error(fake_api) -> None:
    fake_api.answers.append(page(2, [row(), row(arrival="20/11/2026")]))
    with pytest.raises(ogd.FetchError, match="does not match the filters"):
        ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)


def test_row_for_another_commodity_is_an_error(fake_api) -> None:
    fake_api.answers.append(page(1, [row(commodity="Onion")]))
    with pytest.raises(ogd.FetchError, match="does not match the filters"):
        ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)


def test_total_changing_between_pages_is_an_error(fake_api, monkeypatch) -> None:
    monkeypatch.setattr(ogd, "PAGE_SIZE", 1)
    fake_api.answers.extend([page(2, [row(market="A")]), page(3, [row(market="B")])])
    with pytest.raises(ogd.FetchError, match="total changed"):
        ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)


def test_empty_page_before_total_is_reached_is_an_error(fake_api, monkeypatch) -> None:
    monkeypatch.setattr(ogd, "PAGE_SIZE", 1)
    fake_api.answers.extend([page(2, [row(market="A")]), page(2, [])])
    with pytest.raises(ogd.FetchError, match="empty page"):
        ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)


def test_ingest_window_is_yesterday_and_today() -> None:
    dates = ogd.dates_for_command("ingest", date(2026, 11, 23), None, None)
    assert dates == [date(2026, 11, 22), date(2026, 11, 23)]


def test_recheck_window_is_the_seven_dates_before_the_ingest_window() -> None:
    dates = ogd.dates_for_command("recheck", date(2026, 11, 23), None, None)
    assert dates == ogd.date_range(date(2026, 11, 15), date(2026, 11, 21))
    assert len(dates) == 7


def test_recheck_window_crosses_a_month_boundary_by_calendar_days() -> None:
    dates = ogd.dates_for_command("recheck", date(2026, 12, 3), None, None)
    assert dates[0] == date(2026, 11, 25)
    assert dates[-1] == date(2026, 12, 1)


def test_backfill_uses_the_given_range_and_rejects_a_reversed_one() -> None:
    dates = ogd.dates_for_command("backfill", date(2026, 11, 23), date(2026, 11, 1), date(2026, 11, 3))
    assert dates == [date(2026, 11, 1), date(2026, 11, 2), date(2026, 11, 3)]
    with pytest.raises(ValueError):
        ogd.dates_for_command("backfill", date(2026, 11, 23), date(2026, 11, 3), date(2026, 11, 1))
    with pytest.raises(ValueError):
        ogd.dates_for_command("backfill", date(2026, 11, 23), None, None)


def test_same_rows_in_another_order_give_the_same_payload_hash(fake_api) -> None:
    # Seen live on 2026-11-23: two fetches of Onion 22/11 returned the same rows in different orders.
    rows = [row(market="Barasat APMC"), row(market="Habra APMC"), row(market="Kalna APMC")]
    fake_api.answers.extend([page(3, rows), page(3, [rows[2], rows[0], rows[1]])])

    first = ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)
    second = ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)

    assert payload_hash(first.payload) == payload_hash(second.payload)


def test_connection_reset_while_reading_is_retried(fake_api) -> None:
    # Not wrapped in URLError by urllib, unlike a failure while connecting.
    fake_api.answers.extend([ConnectionResetError("reset by peer"), page(1, [row()])])
    snapshot = ogd.fetch_date("Tomato", date(2026, 11, 21), API_KEY)
    assert len(fake_api.calls) == 2
    assert snapshot.payload["total"] == 1
