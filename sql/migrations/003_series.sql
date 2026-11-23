-- The price series we track: one (market, commodity, variety, grade) each, e.g.
-- Barasat / Tomato / Other / FAQ. Filled from the M01 shortlist in M03.
CREATE TABLE series (
    series_id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source    text NOT NULL,
    -- Canonical market name: source name with a trailing " APMC" removed (M01 rename decision).
    market    text NOT NULL,
    commodity text NOT NULL,
    variety   text NOT NULL,
    grade     text NOT NULL,
    district  text NOT NULL,
    is_active boolean NOT NULL DEFAULT true,
    added_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source, market, commodity, variety, grade)
);
