-- Per arrival date: how many tracked series have a status that day, and how many are unreported
-- (blueprint 5.3). is_widespread_gap marks dates where at least half of them are unreported.
-- A data-health signal only: never read it as "markets were closed". A state-wide gap may be a
-- publication problem, and the most recent dates are often incomplete because rows arrive late.
-- A view, not a table, so it can never disagree with series_day_status (owner decision, M04).
CREATE VIEW reporting_gaps_by_date AS
SELECT arrival_date,
       count(*) AS series_tracked,
       count(*) FILTER (WHERE status = 'unreported') AS series_unreported,
       count(*) FILTER (WHERE status = 'unreported') * 2 >= count(*) AS is_widespread_gap
FROM series_day_status
GROUP BY arrival_date;
