-- Postgres/PostGIS schema for the Chicago crime incidents table.
--
-- System of record for point lookups, case-number retrieval, radius/spatial
-- queries, and boundary containment. Columns mirror the 22 coerced fields the
-- ingest layer lands in Parquet (see ingest/schema.py); `year` is deliberately
-- absent (it is derivable from `date` and was a redundant Parquet partition key).
--
-- Times carry no timezone in the source feed: every TIMESTAMP is
-- America/Chicago local wall-clock. `date` is the (often best-estimate)
-- occurrence time; `updated_on` is the record's last-modified time in the portal
-- and the watermark the incremental sync tracks.

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS incidents (
    -- Identity: `id` is the verified numeric primary key (unique). `case_number`
    -- is the CPD RD number and is NOT unique -- multi-offense incidents share it.
    id                      BIGINT PRIMARY KEY,
    case_number             TEXT,

    -- When (America/Chicago, no tz).
    date                    TIMESTAMP NOT NULL,
    updated_on              TIMESTAMP,

    -- What: raw `primary_type` kept for provenance; `primary_type_canonical`
    -- (derived from the IUCR reference, falling back to raw) drives filters and
    -- rollups. `stable_category` is the comparable taxonomy -- the curated
    -- override for codes the city moved between primary types, falling back to
    -- the canonical type. Both are derived once at ingest and landed in Parquet
    -- (see ingest/schema.py), never re-derived here, so this store and DuckDB
    -- cannot disagree about what a burglary is. Non-null in all data observed to
    -- date; asserted NOT NULL so a future null fails loudly rather than loading
    -- silently.
    iucr                    TEXT NOT NULL,
    primary_type            TEXT NOT NULL,
    primary_type_canonical  TEXT NOT NULL,
    stable_category         TEXT NOT NULL,
    description             TEXT,
    fbi_code                TEXT,

    -- Where (textual + codes): beat/district are zero-padded strings for joins.
    block                   TEXT,
    location_description    TEXT,
    beat                    TEXT,
    district                TEXT,
    ward                    SMALLINT,       -- domain ~1..50
    community_area          SMALLINT,       -- domain ~1..77

    -- The sharper geography, and the only one the city does not supply: a
    -- point-in-polygon tag against Chicago's 98 published neighborhood
    -- boundaries, derived once at ingest (see geo/boundaries.py) and landed in
    -- Parquet like the taxonomy columns, never re-derived here.
    --
    -- NULLABLE, unlike every other derived column: 1.84% of rows get no
    -- neighborhood -- 1.55% are ungeocoded and a further ~0.3% are geocoded but
    -- fall outside every polygon (the lake, the airport, the city edge). A NOT
    -- NULL here would reject rows that are legitimately unlocatable. So
    -- `community_area` remains the complete geography and this the precise one,
    -- and an answer has to say which it used.
    neighborhood            TEXT,

    -- Flags (real booleans in the feed; `domestic` = Illinois Domestic Violence
    -- Act qualifying, `arrest` = an arrest was made).
    arrest                  BOOLEAN,
    domestic                BOOLEAN,

    -- Where (coordinates): lat/long are WGS84 degrees (~0.6% null); x/y are the
    -- Illinois State Plane (EPSG:3435) coordinates kept for provenance.
    latitude                DOUBLE PRECISION,
    longitude               DOUBLE PRECISION,
    x_coordinate            DOUBLE PRECISION,
    y_coordinate            DOUBLE PRECISION,

    -- Derived spatial column, stored (computed once, indexable, cannot drift from
    -- lat/long). geography(Point,4326) => ST_DWithin measures in meters. NULL for
    -- ungeocoded rows, which GiST omits, so spatial tools exclude them for free.
    geom geography(Point, 4326) GENERATED ALWAYS AS (
        CASE
            WHEN longitude IS NOT NULL AND latitude IS NOT NULL
            THEN ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
        END
    ) STORED
);

-- Indexes backing the headline tools. Create these AFTER the bulk COPY in the
-- loader, not before -- maintaining them per-row during a 2.9M-row load is slow.
CREATE INDEX IF NOT EXISTS incidents_geom_gix ON incidents USING GIST (geom);              -- nearby_incidents (radius)
CREATE INDEX IF NOT EXISTS incidents_date_idx ON incidents (date);                         -- time filters / ranges
CREATE INDEX IF NOT EXISTS incidents_case_idx ON incidents (case_number);                  -- get_incident by case number

-- search_incidents: one composite per taxonomy. The tool's shape is a category
-- filter plus `ORDER BY date DESC, id DESC LIMIT n` (keyset pagination), so the
-- index has to satisfy the predicate AND deliver the sort order, letting the
-- planner stop after n rows instead of sorting the whole matching set. Two
-- indexes rather than one because `taxonomy` picks the column at query time;
-- neither serves the other. They replace a single-column index on
-- primary_type_canonical, which a btree on (a, b, c) makes redundant -- it
-- serves every predicate on (a) alone.
--
-- These do NOT speed up common categories, and that is expected. For THEFT or
-- BATTERY the planner still prefers `incidents_date_idx` + an Incremental Sort,
-- because walking `date` descending hits a full page of matches almost
-- immediately. What they fix is the SELECTIVE case, where that same walk has to
-- cross most of the table before it finds enough matches: a rare category with a
-- few hundred rows in millions degraded to a near-full scan, orders of magnitude
-- worse than everything else the tool does. With these it is an Index Only Scan.
-- `stable_category` was the worse of the two, having had no index at all.
--
-- Column order is load-bearing: equality column first, then the range/sort
-- columns. Leading with `date` instead is dramatically worse on that selective
-- case -- the index no longer narrows by category, so the scan walks time order
-- as before. `id` is not optional either: it is the keyset tiebreaker, and
-- hundreds of rows can share one `date` value, so a date-only cursor cannot
-- express "resume after this row". DESC is documentation rather than a
-- requirement, since Postgres reads an ASC index backwards just as cheaply.
-- Each index costs roughly a fifth of the heap.
CREATE INDEX IF NOT EXISTS incidents_ptc_date_idx
    ON incidents (primary_type_canonical, date DESC, id DESC);
CREATE INDEX IF NOT EXISTS incidents_stable_date_idx
    ON incidents (stable_category, date DESC, id DESC);

-- neighborhood gets the same treatment, and the reasoning above is why: it is
-- the same predicate-plus-keyset shape, so the index has to satisfy the filter
-- AND deliver the sort order for the planner to stop after n rows.
--
-- This was expected NOT to be worth building, by analogy to community_area
-- below. Measured, it is the single largest index win in the schema. The
-- analogy failed because what matters is not how many distinct values a column
-- has, but how many rows match INSIDE the requested date window -- and a small
-- neighborhood inside one year is a far thinner slice than any community area.
--
-- Museum Campus (1,756 rows of 2.9M), the tool's real query shape:
--
--                     no index              with this index
--   2025          19.24 ms / 68,659 buf     0.14 ms /   188 buf
--   last 90 days   8.38 ms / 34,044 buf     0.04 ms /    50 buf
--   all 11 years  11.90 ms / 34,825 buf     8.85 ms / 34,825 buf  (ignored)
--
-- Without it the plan is an Index Scan Backward on incidents_date_idx that
-- discards 77,338 rows by filter to find 51, touching 75% of the heap for one
-- page of results. With it, an Index Scan that stops at 51. Stable across
-- repeated ANALYZE, and every neighborhood -- rarest, median and most common --
-- lands under 2 ms. Costs ~139 MB, in line with the two indexes above.
--
-- The wide-span case is genuinely marginal: over all 11 years the planner flips
-- between using and ignoring this index across ANALYZE runs, because walking
-- date backwards finds 51 matches soon enough either way. That is fine. It is
-- also not the query anyone asks.
CREATE INDEX IF NOT EXISTS incidents_hood_date_idx
    ON incidents (neighborhood, date DESC, id DESC);

-- Deferred, and checked rather than assumed -- beat, district, community_area
-- and ward stay unindexed. A (community_area, date DESC, id DESC) composite was
-- built and measured: the planner ignored it entirely, reading the same pages by
-- the same plan, for another large index. The date index plus a sort already
-- wins for a filter that still matches a big fraction of the table.
--
-- Re-measured when neighborhood was added, because the original explanation --
-- "these columns are low-cardinality" -- turned out to be the wrong reason for
-- the right conclusion. A bare (community_area) index is still ignored on both
-- an 11-year and a one-year span, so the conclusion holds. But it does help the
-- narrow one: on a 90-day window, community area 9 goes 8.78 ms / 34,044 buf to
-- 1.36 ms / 122 buf. Not built, because that is one span of one shape and 20 MB
-- is not free -- but it is a real effect, so the note now says so rather than
-- implying cardinality settles it.
--
-- A bare (date DESC, id DESC) was similarly not worth its size --
-- `incidents_date_idx` plus an Incremental Sort covers that shape already.
-- Revisit any of these for a query shape that actually measures badly.
