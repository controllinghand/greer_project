--
-- PostgreSQL database dump
--

-- Dumped from database version 14.18 (Homebrew)
-- Dumped by pg_dump version 14.18 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

DROP INDEX IF EXISTS public.latest_company_snapshot_ticker_idx;
DROP INDEX IF EXISTS public.idx_unique_fvg;
DROP INDEX IF EXISTS public.idx_prices_ticker_date;
DROP INDEX IF EXISTS public.idx_latest_snapshot_filters;
DROP INDEX IF EXISTS public.idx_greer_yields_daily_ticker_date;
DROP INDEX IF EXISTS public.idx_greer_scores_ticker_report;
DROP INDEX IF EXISTS public.idx_greer_ops_null_exit;
DROP INDEX IF EXISTS public.idx_greer_ops_exit_date;
DROP INDEX IF EXISTS public.idx_greer_opps_ticker_unique;
DROP INDEX IF EXISTS public.idx_greer_opps_ticker_trgm;
DROP INDEX IF EXISTS public.idx_greer_opps_ticker;
DROP INDEX IF EXISTS public.idx_greer_buyzone_daily_ticker_date;
DROP INDEX IF EXISTS public.idx_financials_ticker_report_date;
DROP INDEX IF EXISTS public.idx_fair_value_gaps_ticker_date;
DROP INDEX IF EXISTS public.idx_company_snapshot_snapshot_date;
DROP INDEX IF EXISTS public.fair_value_gap_unique_idx;
ALTER TABLE IF EXISTS ONLY public.greer_buyzone_daily DROP CONSTRAINT IF EXISTS unique_ticker_date;
ALTER TABLE IF EXISTS ONLY public.prices DROP CONSTRAINT IF EXISTS prices_pkey;
ALTER TABLE IF EXISTS ONLY public.greer_yields_daily DROP CONSTRAINT IF EXISTS greer_yields_daily_pkey;
ALTER TABLE IF EXISTS ONLY public.greer_scores DROP CONSTRAINT IF EXISTS greer_scores_ticker_report_date_key;
ALTER TABLE IF EXISTS ONLY public.greer_scores DROP CONSTRAINT IF EXISTS greer_scores_pkey;
ALTER TABLE IF EXISTS ONLY public.greer_opportunity_periods DROP CONSTRAINT IF EXISTS greer_opportunity_periods_pkey;
ALTER TABLE IF EXISTS ONLY public.greer_buyzone_daily DROP CONSTRAINT IF EXISTS greer_buyzone_daily_pkey;
ALTER TABLE IF EXISTS ONLY public.financials DROP CONSTRAINT IF EXISTS financials_ticker_report_date_key;
ALTER TABLE IF EXISTS ONLY public.financials DROP CONSTRAINT IF EXISTS financials_pkey;
ALTER TABLE IF EXISTS ONLY public.fair_value_gaps DROP CONSTRAINT IF EXISTS fair_value_gaps_pkey;
ALTER TABLE IF EXISTS ONLY public.company_snapshot DROP CONSTRAINT IF EXISTS company_snapshot_pkey;
ALTER TABLE IF EXISTS ONLY public.companies DROP CONSTRAINT IF EXISTS companies_pkey;
ALTER TABLE IF EXISTS ONLY public.backtest_results DROP CONSTRAINT IF EXISTS backtest_results_unique_ticker_run_date;
ALTER TABLE IF EXISTS public.greer_scores ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.greer_buyzone_daily ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.financials ALTER COLUMN id DROP DEFAULT;
ALTER TABLE IF EXISTS public.fair_value_gaps ALTER COLUMN id DROP DEFAULT;
DROP TABLE IF EXISTS public.greer_yields_backup;
DROP TABLE IF EXISTS public.greer_yield_daily_backup;
DROP SEQUENCE IF EXISTS public.greer_scores_id_seq;
DROP MATERIALIZED VIEW IF EXISTS public.greer_opportunities_snapshot;
DROP MATERIALIZED VIEW IF EXISTS public.latest_company_snapshot;
DROP TABLE IF EXISTS public.prices;
DROP TABLE IF EXISTS public.greer_yields_daily;
DROP TABLE IF EXISTS public.greer_scores;
DROP TABLE IF EXISTS public.greer_opportunity_periods;
DROP SEQUENCE IF EXISTS public.greer_buyzone_daily_id_seq;
DROP TABLE IF EXISTS public.greer_buyzone_daily;
DROP SEQUENCE IF EXISTS public.financials_id_seq;
DROP TABLE IF EXISTS public.financials;
DROP SEQUENCE IF EXISTS public.fair_value_gaps_id_seq;
DROP TABLE IF EXISTS public.fair_value_gaps;
DROP TABLE IF EXISTS public.company_snapshot;
DROP TABLE IF EXISTS public.companies;
DROP TABLE IF EXISTS public.backtest_results;
DROP SCHEMA IF EXISTS public;
--
-- Name: public; Type: SCHEMA; Schema: -; Owner: -
--

CREATE SCHEMA public;


--
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON SCHEMA public IS 'standard public schema';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: backtest_results; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.backtest_results (
    ticker text NOT NULL,
    entry_date date NOT NULL,
    entry_close double precision NOT NULL,
    last_date date NOT NULL,
    last_close double precision NOT NULL,
    pct_return double precision NOT NULL,
    days_held integer NOT NULL,
    run_date date NOT NULL
);


--
-- Name: companies; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.companies (
    ticker text NOT NULL,
    name text,
    sector text,
    industry text,
    added_at timestamp without time zone DEFAULT now(),
    delisted_date date,
    delisted boolean DEFAULT false,
    exchange text
);


--
-- Name: company_snapshot; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.company_snapshot (
    snapshot_date timestamp without time zone NOT NULL,
    ticker text NOT NULL,
    greer_value_score numeric,
    greer_yield_score integer,
    buyzone_flag boolean,
    fvg_last_direction text
);


--
-- Name: fair_value_gaps; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.fair_value_gaps (
    id integer NOT NULL,
    ticker text NOT NULL,
    date date NOT NULL,
    direction text NOT NULL,
    gap_min numeric NOT NULL,
    gap_max numeric NOT NULL,
    mitigated boolean DEFAULT false NOT NULL,
    CONSTRAINT fair_value_gaps_direction_check CHECK ((direction = ANY (ARRAY['bullish'::text, 'bearish'::text])))
);


--
-- Name: fair_value_gaps_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.fair_value_gaps_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: fair_value_gaps_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.fair_value_gaps_id_seq OWNED BY public.fair_value_gaps.id;


--
-- Name: financials; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.financials (
    id integer NOT NULL,
    ticker text NOT NULL,
    report_date date NOT NULL,
    book_value_per_share double precision,
    free_cash_flow double precision,
    net_margin double precision,
    total_revenue double precision,
    net_income double precision,
    shares_outstanding double precision
);


--
-- Name: financials_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.financials_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: financials_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.financials_id_seq OWNED BY public.financials.id;


--
-- Name: greer_buyzone_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.greer_buyzone_daily (
    id integer NOT NULL,
    ticker text NOT NULL,
    date date NOT NULL,
    high double precision,
    low double precision,
    aroon_upper double precision,
    aroon_lower double precision,
    midpoint double precision,
    buyzone_start boolean,
    buyzone_end boolean,
    in_buyzone boolean,
    in_sellzone boolean,
    close_price numeric
);


--
-- Name: greer_buyzone_daily_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.greer_buyzone_daily_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: greer_buyzone_daily_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.greer_buyzone_daily_id_seq OWNED BY public.greer_buyzone_daily.id;


--
-- Name: greer_opportunity_periods; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.greer_opportunity_periods (
    ticker text NOT NULL,
    entry_date date NOT NULL,
    exit_date date NOT NULL
);


--
-- Name: greer_scores; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.greer_scores (
    id integer NOT NULL,
    ticker text NOT NULL,
    report_date date NOT NULL,
    greer_score numeric,
    above_50_count integer,
    book_pct numeric,
    fcf_pct numeric,
    margin_pct numeric,
    revenue_pct numeric,
    income_pct numeric,
    shares_pct numeric,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP
);


--
-- Name: greer_yields_daily; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.greer_yields_daily (
    ticker text NOT NULL,
    date date NOT NULL,
    eps_yield double precision,
    fcf_yield double precision,
    revenue_yield double precision,
    book_yield double precision,
    avg_eps_yield double precision,
    avg_fcf_yield double precision,
    avg_revenue_yield double precision,
    avg_book_yield double precision,
    tvpct double precision,
    tvavg double precision,
    tvavg_trend boolean,
    score integer
);


--
-- Name: prices; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.prices (
    ticker text NOT NULL,
    date date NOT NULL,
    close numeric NOT NULL,
    high_price numeric,
    low_price numeric
);


--
-- Name: latest_company_snapshot; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.latest_company_snapshot AS
 WITH universe AS (
         SELECT DISTINCT greer_scores.ticker
           FROM public.greer_scores
        UNION
         SELECT DISTINCT greer_yields_daily.ticker
           FROM public.greer_yields_daily
        UNION
         SELECT DISTINCT greer_buyzone_daily.ticker
           FROM public.greer_buyzone_daily
        UNION
         SELECT DISTINCT fair_value_gaps.ticker
           FROM public.fair_value_gaps
        )
 SELECT u.ticker,
    ( SELECT gs.greer_score
           FROM public.greer_scores gs
          WHERE (gs.ticker = u.ticker)
          ORDER BY gs.report_date DESC
         LIMIT 1) AS greer_value_score,
    ( SELECT gs.above_50_count
           FROM public.greer_scores gs
          WHERE (gs.ticker = u.ticker)
          ORDER BY gs.report_date DESC
         LIMIT 1) AS above_50_count,
    ( SELECT gyd.score
           FROM public.greer_yields_daily gyd
          WHERE (gyd.ticker = u.ticker)
          ORDER BY gyd.date DESC
         LIMIT 1) AS greer_yield_score,
    ( SELECT gbd.in_buyzone
           FROM public.greer_buyzone_daily gbd
          WHERE (gbd.ticker = u.ticker)
          ORDER BY gbd.date DESC
         LIMIT 1) AS buyzone_flag,
    ( SELECT gbd.date
           FROM public.greer_buyzone_daily gbd
          WHERE ((gbd.ticker = u.ticker) AND gbd.buyzone_start)
          ORDER BY gbd.date DESC
         LIMIT 1) AS bz_start_date,
    ( SELECT gbd.date
           FROM public.greer_buyzone_daily gbd
          WHERE ((gbd.ticker = u.ticker) AND gbd.buyzone_end)
          ORDER BY gbd.date DESC
         LIMIT 1) AS bz_end_date,
    ( SELECT fvg.date
           FROM public.fair_value_gaps fvg
          WHERE ((fvg.ticker = u.ticker) AND (fvg.mitigated = false))
          ORDER BY fvg.date DESC
         LIMIT 1) AS fvg_last_date,
    ( SELECT fvg.direction
           FROM public.fair_value_gaps fvg
          WHERE ((fvg.ticker = u.ticker) AND (fvg.mitigated = false))
          ORDER BY fvg.date DESC
         LIMIT 1) AS fvg_last_direction,
    ( SELECT min(p.date) AS min
           FROM public.prices p
          WHERE (p.ticker = u.ticker)) AS first_trade_date,
    (( SELECT min(p.date) AS min
           FROM public.prices p
          WHERE (p.ticker = u.ticker)) > ((date_trunc('year'::text, (CURRENT_DATE)::timestamp with time zone))::date - '1 day'::interval)) AS is_new_company
   FROM universe u
  ORDER BY u.ticker
  WITH NO DATA;


--
-- Name: greer_opportunities_snapshot; Type: MATERIALIZED VIEW; Schema: public; Owner: -
--

CREATE MATERIALIZED VIEW public.greer_opportunities_snapshot AS
 WITH current_ops AS (
         SELECT greer_opportunity_periods.ticker,
            max(greer_opportunity_periods.entry_date) AS last_entry_date
           FROM public.greer_opportunity_periods
          WHERE (greer_opportunity_periods.exit_date >= ((CURRENT_DATE AT TIME ZONE 'America/New_York'::text) - '1 day'::interval))
          GROUP BY greer_opportunity_periods.ticker
        )
 SELECT l.ticker,
    l.greer_value_score AS greer_value,
    l.greer_yield_score AS yield_score,
    l.buyzone_flag,
    l.fvg_last_direction,
    o.last_entry_date
   FROM (public.latest_company_snapshot l
     JOIN current_ops o ON ((l.ticker = o.ticker)))
  WHERE ((l.greer_value_score >= (50)::numeric) AND (l.greer_yield_score >= 3) AND (l.buyzone_flag IS TRUE) AND (l.fvg_last_direction = 'bullish'::text))
  ORDER BY l.greer_value_score DESC
  WITH NO DATA;


--
-- Name: greer_scores_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.greer_scores_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: greer_scores_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.greer_scores_id_seq OWNED BY public.greer_scores.id;


--
-- Name: greer_yield_daily_backup; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.greer_yield_daily_backup (
    ticker text,
    date date,
    fiscal_year integer,
    eps real,
    fcf real,
    revenue real,
    book_value real,
    close_price real,
    eps_yield real,
    fcf_yield real,
    revenue_yield real,
    book_yield real,
    avg_eps_yield real,
    avg_fcf_yield real,
    avg_revenue_yield real,
    avg_book_yield real,
    tvpct real,
    tvavg real,
    tvavg_trend boolean,
    score integer
);


--
-- Name: greer_yields_backup; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.greer_yields_backup (
    ticker text,
    fiscal_year integer,
    eps numeric,
    fcf numeric,
    revenue numeric,
    shares_outstanding numeric,
    book_value numeric,
    close_price numeric,
    eps_yield numeric,
    fcf_yield numeric,
    revenue_yield numeric,
    book_yield numeric,
    avg_eps_yield numeric,
    avg_fcf_yield numeric,
    avg_revenue_yield numeric,
    avg_book_yield numeric,
    tvpct numeric,
    tvavg numeric,
    tvavg_trend boolean,
    score integer
);


--
-- Name: fair_value_gaps id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fair_value_gaps ALTER COLUMN id SET DEFAULT nextval('public.fair_value_gaps_id_seq'::regclass);


--
-- Name: financials id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.financials ALTER COLUMN id SET DEFAULT nextval('public.financials_id_seq'::regclass);


--
-- Name: greer_buyzone_daily id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.greer_buyzone_daily ALTER COLUMN id SET DEFAULT nextval('public.greer_buyzone_daily_id_seq'::regclass);


--
-- Name: greer_scores id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.greer_scores ALTER COLUMN id SET DEFAULT nextval('public.greer_scores_id_seq'::regclass);


--
-- Name: backtest_results backtest_results_unique_ticker_run_date; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.backtest_results
    ADD CONSTRAINT backtest_results_unique_ticker_run_date UNIQUE (ticker, run_date);


--
-- Name: companies companies_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.companies
    ADD CONSTRAINT companies_pkey PRIMARY KEY (ticker);


--
-- Name: company_snapshot company_snapshot_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.company_snapshot
    ADD CONSTRAINT company_snapshot_pkey PRIMARY KEY (ticker, snapshot_date);


--
-- Name: fair_value_gaps fair_value_gaps_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.fair_value_gaps
    ADD CONSTRAINT fair_value_gaps_pkey PRIMARY KEY (id);


--
-- Name: financials financials_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.financials
    ADD CONSTRAINT financials_pkey PRIMARY KEY (id);


--
-- Name: financials financials_ticker_report_date_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.financials
    ADD CONSTRAINT financials_ticker_report_date_key UNIQUE (ticker, report_date);


--
-- Name: greer_buyzone_daily greer_buyzone_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.greer_buyzone_daily
    ADD CONSTRAINT greer_buyzone_daily_pkey PRIMARY KEY (id);


--
-- Name: greer_opportunity_periods greer_opportunity_periods_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.greer_opportunity_periods
    ADD CONSTRAINT greer_opportunity_periods_pkey PRIMARY KEY (ticker, entry_date);


--
-- Name: greer_scores greer_scores_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.greer_scores
    ADD CONSTRAINT greer_scores_pkey PRIMARY KEY (id);


--
-- Name: greer_scores greer_scores_ticker_report_date_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.greer_scores
    ADD CONSTRAINT greer_scores_ticker_report_date_key UNIQUE (ticker, report_date);


--
-- Name: greer_yields_daily greer_yields_daily_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.greer_yields_daily
    ADD CONSTRAINT greer_yields_daily_pkey PRIMARY KEY (ticker, date);


--
-- Name: prices prices_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.prices
    ADD CONSTRAINT prices_pkey PRIMARY KEY (ticker, date);


--
-- Name: greer_buyzone_daily unique_ticker_date; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.greer_buyzone_daily
    ADD CONSTRAINT unique_ticker_date UNIQUE (ticker, date);


--
-- Name: fair_value_gap_unique_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX fair_value_gap_unique_idx ON public.fair_value_gaps USING btree (ticker, date, direction, gap_min, gap_max);


--
-- Name: idx_company_snapshot_snapshot_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_company_snapshot_snapshot_date ON public.company_snapshot USING btree (snapshot_date DESC);


--
-- Name: idx_fair_value_gaps_ticker_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_fair_value_gaps_ticker_date ON public.fair_value_gaps USING btree (ticker, date DESC);


--
-- Name: idx_financials_ticker_report_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_financials_ticker_report_date ON public.financials USING btree (ticker, report_date);


--
-- Name: idx_greer_buyzone_daily_ticker_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_greer_buyzone_daily_ticker_date ON public.greer_buyzone_daily USING btree (ticker, date DESC);


--
-- Name: idx_greer_opps_ticker; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_greer_opps_ticker ON public.greer_opportunities_snapshot USING btree (ticker);


--
-- Name: idx_greer_opps_ticker_trgm; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_greer_opps_ticker_trgm ON public.greer_opportunities_snapshot USING gin (ticker public.gin_trgm_ops);


--
-- Name: idx_greer_opps_ticker_unique; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_greer_opps_ticker_unique ON public.greer_opportunities_snapshot USING btree (ticker);


--
-- Name: idx_greer_ops_exit_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_greer_ops_exit_date ON public.greer_opportunity_periods USING btree (exit_date DESC, ticker);


--
-- Name: idx_greer_ops_null_exit; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_greer_ops_null_exit ON public.greer_opportunity_periods USING btree (ticker, entry_date) WHERE (exit_date IS NULL);


--
-- Name: idx_greer_scores_ticker_report; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_greer_scores_ticker_report ON public.greer_scores USING btree (ticker, report_date DESC);


--
-- Name: idx_greer_yields_daily_ticker_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_greer_yields_daily_ticker_date ON public.greer_yields_daily USING btree (ticker, date DESC);


--
-- Name: idx_latest_snapshot_filters; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_latest_snapshot_filters ON public.latest_company_snapshot USING btree (greer_value_score DESC, greer_yield_score, buyzone_flag, fvg_last_direction, ticker);


--
-- Name: idx_prices_ticker_date; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX idx_prices_ticker_date ON public.prices USING btree (ticker, date);


--
-- Name: idx_unique_fvg; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX idx_unique_fvg ON public.fair_value_gaps USING btree (ticker, date, direction, gap_min, gap_max);


--
-- Name: latest_company_snapshot_ticker_idx; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX latest_company_snapshot_ticker_idx ON public.latest_company_snapshot USING btree (ticker);


--
-- PostgreSQL database dump complete
--

