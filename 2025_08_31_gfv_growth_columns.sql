-- 2025_08_31_gfv_growth_columns.sql
-- Purpose: add GFV growth columns, indexes, and a helper view used by Home.py
-- Safe to run multiple times (uses IF NOT EXISTS)

BEGIN;

-- === greer_fair_value_daily: new columns used by calculator & UI ===
ALTER TABLE greer_fair_value_daily
    ADD COLUMN IF NOT EXISTS cagr_years           integer,
    ADD COLUMN IF NOT EXISTS growth_rate          double precision,         -- legacy overall growth (kept if you still write it)
    ADD COLUMN IF NOT EXISTS discount_rate        double precision,
    ADD COLUMN IF NOT EXISTS terminal_growth      double precision,
    ADD COLUMN IF NOT EXISTS graham_yield_y       double precision,
    ADD COLUMN IF NOT EXISTS growth_rate_fcf      double precision,
    ADD COLUMN IF NOT EXISTS growth_rate_eps      double precision,
    ADD COLUMN IF NOT EXISTS growth_method_fcf    text,
    ADD COLUMN IF NOT EXISTS growth_method_eps    text;

-- If you didn’t already have this column (used for the tooltip “today’s price”)
-- Comment out if already present.
-- ALTER TABLE greer_fair_value_daily ADD COLUMN IF NOT EXISTS close_price double precision;

-- Optional: ensure gfv_status exists (text)
-- ALTER TABLE greer_fair_value_daily ADD COLUMN IF NOT EXISTS gfv_status text;

-- === Performance indexes ===
-- Fast “latest per ticker” and time-ranged queries
CREATE INDEX IF NOT EXISTS idx_gfv_daily_ticker_date
    ON greer_fair_value_daily (ticker, date DESC);

-- === Helper view for Home.py badge ===
-- Picks the most recent row per ticker.
-- If you already created something similar, DROP/CREATE or skip.
CREATE OR REPLACE VIEW greer_fair_value_latest AS
SELECT DISTINCT ON (ticker)
       ticker,
       date,
       close_price,
       eps,
       fcf_per_share,
       gfv_price,
       gfv_status,
       dcf_value,
       graham_value,
       growth_rate_fcf,
       growth_rate_eps,
       discount_rate,
       terminal_growth,
       graham_yield_y
FROM greer_fair_value_daily
ORDER BY ticker, date DESC;

-- === (Optional) normalize companies.exchange labels later via app-level job; no schema change needed ===

COMMIT;
