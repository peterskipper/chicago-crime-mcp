-- Postgres/PostGIS schema for the Chicago crime incidents table.
--
-- System of record for point lookups, case-number retrieval, radius/spatial
-- queries, and boundary containment. Columns mirror the 21 coerced fields the
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
CREATE INDEX IF NOT EXISTS incidents_ptc_idx  ON incidents (primary_type_canonical);       -- filter/group by offense
CREATE INDEX IF NOT EXISTS incidents_case_idx ON incidents (case_number);                  -- get_incident by case number

-- Deferred (add EXPLAIN-driven, once the tools that filter on them exist):
--   beat, district, community_area, ward. These are low-cardinality (e.g.
--   community_area has ~77 values over 2.9M rows), so a bare single-column index
--   is often not selective enough to beat a seq scan; when added they likely
--   want to be composites with `date` (e.g. (community_area, date)).
