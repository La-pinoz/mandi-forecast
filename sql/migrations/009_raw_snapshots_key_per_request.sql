-- M05: one raw snapshot per REQUEST per run (owner decision, 2026-11-23).
-- The old key (run_id, payload_hash) assumed two responses in one run never have the same body. The first
-- live daily run broke it: Tomato and Onion for today both returned an empty answer with the same
-- updated_date, so the two bodies were identical. Adding source and request_params to the key keeps both;
-- the same request stored twice in one run is still rejected. payload_hash keeps its meaning (hash of the
-- response body). Existing rows already satisfy the new key, so no data changes.
ALTER TABLE raw_snapshots DROP CONSTRAINT raw_snapshots_pkey;
ALTER TABLE raw_snapshots ADD PRIMARY KEY (run_id, source, request_params, payload_hash);
