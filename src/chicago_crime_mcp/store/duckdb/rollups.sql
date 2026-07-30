-- DuckDB OLAP rollup tables, materialized from the Hive-partitioned Parquet.
--
-- These back the aggregate arm of the query router. Every table shares the same
-- grain -- month x primary_type_canonical x stable_category x ONE geography --
-- and the same four measures. The caller creates the `incidents` view over
-- Parquet and the `iucr_reference` table over the committed CSV before running
-- this file (see rollups.py); no path is hardcoded here.
--
-- GRAIN: MONTH. Daily grain was measured and rejected: day x type x beat yields
-- 2.33M groups from 2.88M source rows (81% of the row count), i.e. no
-- compression and no benefit. Month buckets roll up cleanly to quarter and year
-- by summing. Arbitrary date ranges ("last 30 days") deliberately do NOT route
-- here -- they fall through to a live DuckDB scan over the same Parquet.
--
-- ONE GEOGRAPHY PER TABLE. `district` gets its own table rather than riding
-- along in rollup_beat. beat -> district is only *nearly* a functional
-- dependency: 248 source rows (0.009%) put a beat in the wrong district, which
-- would split a single (month, type, beat) bucket into two rows and make beat
-- totals silently wrong for anyone who filtered on both columns. A dedicated
-- district table is ~65k rows and removes that failure mode by construction.
-- Neither table infers geography from the other; each reports the source field.
--
-- MEASURES ARE COUNTS, NEVER RATES. arrest_rate is derived at read time as
-- arrests/incidents. Storing a rate would be wrong the moment two buckets are
-- summed -- averaging rates across buckets of different sizes does not give the
-- combined rate.
--
-- NULL GEOGRAPHY BUCKETS ARE KEPT. 221 rows have a null community_area and 56 a
-- null ward. Dropping them would silently lose incidents; keeping them preserves
-- the invariant SUM(incidents) == source row count, which the tests assert on
-- every table.
--
-- TWO TYPE DIMENSIONS, NEVER ONE. Every table carries BOTH
-- `primary_type_canonical` (what the city called it) and `stable_category` (what
-- it means for comparisons across time). They differ only for the handful of
-- curated codes in reference/iucr_codes.csv, so the row-count cost is under 1%
-- on every table (measured), and summing over the unwanted dimension recovers
-- either view from the same table. The tool layer picks one via
-- `taxonomy: "source" | "comparable"`, defaulting to source; storing only the
-- normalized value would bake analytic policy into the rollups and make the raw
-- counts unrecoverable.

-- Every rollup reads from here, not from `incidents` directly: the join tags each
-- row with its stable category, falling back to the canonical type for the ~99%
-- of codes with no curated override (and for rows whose IUCR is null or absent
-- from the reference). A LEFT JOIN, so an unknown code can never drop a row.
-- `iucr_reference` is created by rollups.py from the committed CSV.
CREATE OR REPLACE VIEW incidents_tagged AS
SELECT
    i.*,
    coalesce(r.stable_category, i.primary_type_canonical) AS stable_category
FROM incidents i
LEFT JOIN iucr_reference r ON i.iucr = r.iucr;

-- Citywide: no geography dimension. Tiny (~4k rows) and answers the most
-- common shape of question ("<type> per month in Chicago") without a scan.
CREATE OR REPLACE TABLE rollup_citywide AS
SELECT
    date_trunc('month', date)                    AS month,
    primary_type_canonical,
    stable_category,
    count(*)                                     AS incidents,
    count(*) FILTER (WHERE arrest)               AS arrests,
    count(*) FILTER (WHERE domestic)             AS domestic,
    count(*) FILTER (WHERE latitude IS NOT NULL) AS geocoded
FROM incidents_tagged
GROUP BY ALL
ORDER BY month, primary_type_canonical;

-- Beat: the finest geography (~275 beats). No `district` column -- see the
-- header note.
CREATE OR REPLACE TABLE rollup_beat AS
SELECT
    date_trunc('month', date)                    AS month,
    primary_type_canonical,
    stable_category,
    beat,
    count(*)                                     AS incidents,
    count(*) FILTER (WHERE arrest)               AS arrests,
    count(*) FILTER (WHERE domestic)             AS domestic,
    count(*) FILTER (WHERE latitude IS NOT NULL) AS geocoded
FROM incidents_tagged
GROUP BY ALL
ORDER BY month, primary_type_canonical, beat;

-- District: 24 police districts, taken from the source `district` field rather
-- than inferred from `beat`.
CREATE OR REPLACE TABLE rollup_district AS
SELECT
    date_trunc('month', date)                    AS month,
    primary_type_canonical,
    stable_category,
    district,
    count(*)                                     AS incidents,
    count(*) FILTER (WHERE arrest)               AS arrests,
    count(*) FILTER (WHERE domestic)             AS domestic,
    count(*) FILTER (WHERE latitude IS NOT NULL) AS geocoded
FROM incidents_tagged
GROUP BY ALL
ORDER BY month, primary_type_canonical, district;

-- Community area: the 77 official community areas -- the geography most people
-- mean by "neighborhood" (resolve_neighborhood maps colloquial names to these).
CREATE OR REPLACE TABLE rollup_community_area AS
SELECT
    date_trunc('month', date)                    AS month,
    primary_type_canonical,
    stable_category,
    community_area,
    count(*)                                     AS incidents,
    count(*) FILTER (WHERE arrest)               AS arrests,
    count(*) FILTER (WHERE domestic)             AS domestic,
    count(*) FILTER (WHERE latitude IS NOT NULL) AS geocoded
FROM incidents_tagged
GROUP BY ALL
ORDER BY month, primary_type_canonical, community_area;

-- Ward: the 50 aldermanic wards (political, not police, geography).
CREATE OR REPLACE TABLE rollup_ward AS
SELECT
    date_trunc('month', date)                    AS month,
    primary_type_canonical,
    stable_category,
    ward,
    count(*)                                     AS incidents,
    count(*) FILTER (WHERE arrest)               AS arrests,
    count(*) FILTER (WHERE domestic)             AS domestic,
    count(*) FILTER (WHERE latitude IS NOT NULL) AS geocoded
FROM incidents_tagged
GROUP BY ALL
ORDER BY month, primary_type_canonical, ward;

-- Provenance for the whole rollup set: one row, rebuilt with the tables so it
-- can never describe a different build. Feeds describe_schema (the available
-- date range and row count the model would otherwise guess at) and the freshness
-- field of the tool result envelope.
--
-- `built_at` is naive UTC: DuckDB's now() is TIMESTAMPTZ, which needs pytz to
-- cross into Python. Note this differs from every other timestamp in the
-- project -- source feed times are naive America/Chicago wall-clock -- because
-- this one is a system clock reading, not source data.
CREATE OR REPLACE TABLE rollup_meta AS
SELECT
    now() AT TIME ZONE 'UTC' AS built_at,
    count(*)                 AS source_rows,
    count(DISTINCT year)     AS partitions,
    min(date)                AS min_date,
    max(date)                AS max_date
FROM incidents;

-- Per-IUCR lifespan, derived from the data itself -- no curation, so it covers
-- codes nobody has audited yet, including ones CPD mints after this is written.
--
-- WHY: IUCR codes come and go mid-dataset, and a code that appears partway
-- through inflates its category's trend without looking suspicious. Measured
-- example: `0760` BURGLARY FROM MOTOR VEHICLE is 0.3% of all rows but 61% of
-- 2026 burglary rows. The tool layer compares a requested span against these
-- bounds (and against rollup_meta's min_date/max_date, which say what a "full
-- span" even is) and warns when the span crosses a code's introduction or
-- retirement, quantified as a share of rows. The warning fires in BOTH taxonomy
-- modes: `comparable` fixes only the drift someone curated, never all of it.
--
-- Grain is IUCR, not category, because that is the level churn happens at;
-- carrying both type columns lets the tool go from a category the user asked
-- about to the codes that constitute it. The null-IUCR bucket is kept, so
-- SUM(incidents) here still equals the source row count.
CREATE OR REPLACE TABLE code_coverage AS
SELECT
    iucr,
    min(primary_type_canonical)    AS primary_type_canonical,
    min(stable_category)           AS stable_category,
    min(description)               AS description,
    date_trunc('month', min(date)) AS first_month,
    date_trunc('month', max(date)) AS last_month,
    count(*)                       AS incidents
FROM incidents_tagged
GROUP BY iucr
ORDER BY iucr;
