-- DuckDB OLAP rollup tables, materialized from the Hive-partitioned Parquet.
--
-- These back the aggregate arm of the query router. Every table shares the same
-- grain -- month x primary_type_canonical x ONE geography -- and the same four
-- measures. The caller creates the `incidents` view over Parquet before running
-- this file (see rollups.py); no path is hardcoded here.
--
-- GRAIN: MONTH. Daily grain was measured and rejected: day x type x beat yields
-- 1.37M groups from 1.69M source rows (81% of the row count), i.e. no
-- compression and no benefit. Month buckets roll up cleanly to quarter and year
-- by summing. Arbitrary date ranges ("last 30 days") deliberately do NOT route
-- here -- they fall through to a live DuckDB scan over the same Parquet.
--
-- ONE GEOGRAPHY PER TABLE. `district` gets its own table rather than riding
-- along in rollup_beat. beat -> district is only *nearly* a functional
-- dependency: 207 source rows (0.012%) put a beat in the wrong district, which
-- would split a single (month, type, beat) bucket into two rows and make beat
-- totals silently wrong for anyone who filtered on both columns. A dedicated
-- district table is ~39k rows and removes that failure mode by construction.
-- Neither table infers geography from the other; each reports the source field.
--
-- MEASURES ARE COUNTS, NEVER RATES. arrest_rate is derived at read time as
-- arrests/incidents. Storing a rate would be wrong the moment two buckets are
-- summed -- averaging rates across buckets of different sizes does not give the
-- combined rate.
--
-- NULL GEOGRAPHY BUCKETS ARE KEPT. 150 rows have a null community_area and 36 a
-- null ward. Dropping them would silently lose incidents; keeping them preserves
-- the invariant SUM(incidents) == source row count, which the tests assert on
-- every table.

-- Citywide: no geography dimension. Tiny (~2.4k rows) and answers the most
-- common shape of question ("<type> per month in Chicago") without a scan.
CREATE OR REPLACE TABLE rollup_citywide AS
SELECT
    date_trunc('month', date)                    AS month,
    primary_type_canonical,
    count(*)                                     AS incidents,
    count(*) FILTER (WHERE arrest)               AS arrests,
    count(*) FILTER (WHERE domestic)             AS domestic,
    count(*) FILTER (WHERE latitude IS NOT NULL) AS geocoded
FROM incidents
GROUP BY ALL
ORDER BY month, primary_type_canonical;

-- Beat: the finest geography (~275 beats). No `district` column -- see the
-- header note.
CREATE OR REPLACE TABLE rollup_beat AS
SELECT
    date_trunc('month', date)                    AS month,
    primary_type_canonical,
    beat,
    count(*)                                     AS incidents,
    count(*) FILTER (WHERE arrest)               AS arrests,
    count(*) FILTER (WHERE domestic)             AS domestic,
    count(*) FILTER (WHERE latitude IS NOT NULL) AS geocoded
FROM incidents
GROUP BY ALL
ORDER BY month, primary_type_canonical, beat;

-- District: 24 police districts, taken from the source `district` field rather
-- than inferred from `beat`.
CREATE OR REPLACE TABLE rollup_district AS
SELECT
    date_trunc('month', date)                    AS month,
    primary_type_canonical,
    district,
    count(*)                                     AS incidents,
    count(*) FILTER (WHERE arrest)               AS arrests,
    count(*) FILTER (WHERE domestic)             AS domestic,
    count(*) FILTER (WHERE latitude IS NOT NULL) AS geocoded
FROM incidents
GROUP BY ALL
ORDER BY month, primary_type_canonical, district;

-- Community area: the 77 official community areas -- the geography most people
-- mean by "neighborhood" (resolve_neighborhood maps colloquial names to these).
CREATE OR REPLACE TABLE rollup_community_area AS
SELECT
    date_trunc('month', date)                    AS month,
    primary_type_canonical,
    community_area,
    count(*)                                     AS incidents,
    count(*) FILTER (WHERE arrest)               AS arrests,
    count(*) FILTER (WHERE domestic)             AS domestic,
    count(*) FILTER (WHERE latitude IS NOT NULL) AS geocoded
FROM incidents
GROUP BY ALL
ORDER BY month, primary_type_canonical, community_area;

-- Ward: the 50 aldermanic wards (political, not police, geography).
CREATE OR REPLACE TABLE rollup_ward AS
SELECT
    date_trunc('month', date)                    AS month,
    primary_type_canonical,
    ward,
    count(*)                                     AS incidents,
    count(*) FILTER (WHERE arrest)               AS arrests,
    count(*) FILTER (WHERE domestic)             AS domestic,
    count(*) FILTER (WHERE latitude IS NOT NULL) AS geocoded
FROM incidents
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
