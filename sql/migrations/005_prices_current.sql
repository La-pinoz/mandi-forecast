-- Latest known version of each observation; a convenience copy of price_vintages for modelling
-- and the dashboard. Never the source of truth for as-of questions (use price_vintages).
CREATE TABLE prices_current (
    source       text NOT NULL,
    market       text NOT NULL,
    commodity    text NOT NULL,
    variety      text NOT NULL,
    grade        text NOT NULL,
    arrival_date date NOT NULL,
    district     text NOT NULL,
    min_price    numeric(12, 2) NOT NULL,
    max_price    numeric(12, 2) NOT NULL,
    modal_price  numeric(12, 2) NOT NULL,
    vintage_id   bigint NOT NULL REFERENCES price_vintages (vintage_id),
    CHECK (min_price > 0 AND min_price <= modal_price AND modal_price <= max_price),
    PRIMARY KEY (source, market, commodity, variety, grade, arrival_date)
);
