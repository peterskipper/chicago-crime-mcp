-- DuckDB OLAP rollup tables, materialized from the Hive-partitioned Parquet.
--
-- These back the aggregate arm of the query router. Every table shares the same
-- grain -- month x primary_type_canonical x stable_category x ONE geography --
-- and the same four measures. The caller creates the `incidents` view over
-- Parquet before running this file (see rollups.py); no path is hardcoded here.
-- Both offense taxonomies arrive as columns in the Parquet, derived once at
-- ingest -- nothing here re-derives them.
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

-- Every rollup reads from here, not from `incidents` directly.
--
-- This view used to compute `stable_category` itself, by LEFT JOINing the
-- reference snapshot and coalescing onto `primary_type_canonical`. It no longer
-- does: the column is derived once at ingest and materialized into Parquet (see
-- ingest/schema.add_stable_category), which is what lets Postgres filter on it
-- too. Deriving it here as well would mean two expressions of one rule -- and
-- they differed in a real edge case, since this one fell back to the incident's
-- canonical type while a Postgres-side equivalent would have fallen back to the
-- reference's own label, disagreeing for any code absent from the snapshot.
--
-- The view is kept as the named relation the rollups and tier-3 scans read, so
-- the reader has one place to point at, and so a future row-level derivation
-- has somewhere to live. `iucr_reference` is still created by rollups.py from
-- the committed CSV -- code_coverage below needs it for offense descriptions.
CREATE OR REPLACE VIEW incidents_tagged AS
SELECT * FROM incidents;

-- Drop the reference table the old view joined against. A database file built
-- before this change still carries it, and nothing recreates it -- so without
-- this, a rebuilt database keeps a stale copy of the curation forever. Dropped
-- *after* the view is redefined, since the old view depends on it.
DROP TABLE IF EXISTS iucr_reference;

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
--
-- Bounds only. The *weight* behind a bound -- how many rows in the requested
-- span actually come from a code that enters or leaves mid-span -- comes from
-- rollup_code_month below; this table's `incidents` is a lifetime count and
-- must not be used as that share.
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

-- The weights behind code_coverage's bounds: incidents per IUCR per month, so
-- the tool layer can say what *share* of a requested span's rows come from
-- codes that do not cover the whole span.
--
-- WHY A TABLE AND NOT A SCAN. The coverage warning has to fire on every
-- aggregate call, and it has to be quantified -- naming a drifting code without
-- saying whether it moved 0.1% or 40% of the rows tells the model nothing it
-- can act on. Computing that share by scanning `incidents_tagged` per call
-- measures 37ms on a full-span query, which would swamp the 0.67ms rollup it is
-- meant to annotate and make the materialized tier pointless. Materialized, the
-- same warning costs 1.5ms and 33,510 rows.
--
-- It also works: run against the real dataset with no curation input, the top
-- codes it surfaces over the full span are exactly the ones the manual audit in
-- the README found -- 0760 burglary-from-MV, 3970 extortion, 1187 state
-- benefits fraud, 3400 looting.
--
-- CITYWIDE, NO GEOGRAPHY. Adding one would multiply this table by up to 275
-- (beats) to answer a question that is not geographic: taxonomy drift is a
-- property of how the city codes offenses, not of where they happen. The tool
-- layer reports the share as citywide and says so; a per-neighborhood share
-- would be a different, much more expensive claim.
--
-- Both type columns ride along so the share can be scoped to whatever the
-- caller filtered on, under either taxonomy mode. The null-IUCR bucket is kept
-- here too, so SUM(incidents) still equals the source row count -- the same
-- invariant every other table in this file holds.
CREATE OR REPLACE TABLE rollup_code_month AS
SELECT
    date_trunc('month', date) AS month,
    iucr,
    primary_type_canonical,
    stable_category,
    count(*)                  AS incidents
FROM incidents_tagged
GROUP BY ALL
ORDER BY month, iucr;
