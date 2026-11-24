# Incidents

Real outages and bad-data events only, newest first, each with what changed afterwards. Days with no report from
the source are not incidents by themselves; they are counted on the dashboard's **Data health** tab.

---

## 2026-11-24: no data or forecasts for 24/11; the API does not answer GitHub's runners

**What happened.** The first GitHub Actions run of the Daily workflow (a manual run at 04:37 UTC) failed. Every
request to data.gov.in got `Connection refused` or timed out. The same command worked from the owner's laptop in
India. The likely cause is that data.gov.in blocks traffic from outside India or from cloud data centres
(assumed, not confirmed). While the job waited through retries, Neon ended its idle database session
(`IdleInTransactionSessionTimeout`), so the run could not be marked `failed`.

**Impact.** No ingest, recheck, forecast or scores from GitHub on 24/11. The dashboard's Data health tab still showed
the 23/11 runs as the latest, all succeeded, so the failure was visible only on GitHub.

**What changed.** `ingest_dates` in `src/mandi/ingest/ogd.py` now commits right after reading the series, so no
transaction is open during API calls and a failed fetch is recorded as a `failed` run. Regression test:
`test_no_transaction_is_left_open_while_waiting_for_the_api`. **Still open:** where the daily job runs (the owner's
decision); see `docs/M05-daily-ingest.md`, §5 item 6.

---

## 2026-11-23: first live daily ingest failed (run 5)

**What happened.** The first live `ingest` run stopped with `UniqueViolation` on `raw_snapshots`. Tomato and onion
for "today" (23/11) both came back empty, with the same `updated_date`, so the two responses were byte-identical.
The table's key at that time, `(run_id, payload_hash)`, allowed only one of them per run.

**Impact.** Run 5 was marked `failed`. The 3 responses stored before the error stayed. Run 6, a few minutes later,
completed the ingest, and its refresh repaired `prices_current` (5 rows) and `series_day_status` (8 rows).

**What changed.** Migration `009_raw_snapshots_key_per_request.sql` puts the request into the key, so two requests
with identical answers are both kept. Regression tests: `test_two_requests_with_identical_responses_in_one_run_are_both_kept`
and `test_empty_responses_for_both_crops_in_one_run_are_both_stored`. Run 6 also showed that the API returns the same
rows in a varying order; records are now sorted before hashing, so an unchanged day is not stored twice.

---

## History (2023–2026, found 2026-11-23): conflicting duplicate rows

**What happened.** For 5 observation keys (series and date), the source sent two rows with the same key and
different prices.

**Impact.** All 10 rows were quarantined with reason `conflicting_duplicate`, because there is no way to tell which
row is right. Those 5 series-days count as `unreported` and are not used as prices or actuals.

**What changed.** The `conflicting_duplicate` contract in `src/mandi/validate/contracts.py` handles this on every
ingest from now on (M04). Test: `test_conflicting_duplicates_are_both_quarantined`.
