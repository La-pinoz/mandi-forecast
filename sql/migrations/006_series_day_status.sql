-- One row per tracked series per calendar day (blueprint 5.3). Keeps "no trade" (a fact) apart from
-- "no report" (missing data). M01 found no NIL marker in the source, so 'no_transaction' is
-- currently never assigned; it stays in case the source adds one.
CREATE TABLE series_day_status (
    series_id      integer NOT NULL REFERENCES series (series_id),
    arrival_date   date NOT NULL,
    status         text NOT NULL CHECK (status IN ('reported', 'no_transaction', 'unreported')),
    -- Statistical anomaly flag (M04). Flagged values are kept, never removed.
    is_anomaly     boolean NOT NULL DEFAULT false,
    anomaly_reason text,
    updated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (series_id, arrival_date)
);
