# chicago-crime-mcp

A constrained [MCP](https://modelcontextprotocol.io) server over the City of
Chicago's crime data, backed by a real **ingest → store → serve** pipeline. Ask
questions about Chicago crime in plain English (through an LLM client) and get
grounded, structured answers.

The design goal is a *system*, not a demo: a small surface of typed,
purpose-built tools with schema validation and deliberate query routing —
explicitly **not** a "hand the LLM a SQL prompt" text-to-SQL agent.

> **Status:** under construction. The ingest and storage layers are built and
> loaded (2.88M incidents, 2015 – present); the MCP tool surface is next. See the
> roadmap below.

## Architecture

```
Socrata (SODA API)
      │  paginated, checkpointed backfill + daily incremental (updated_on)
      ▼
   Parquet  ──►  Postgres/PostGIS   (point lookups, case-number, spatial joins)
            ──►  DuckDB + rollups   (OLAP aggregates over 2.88M rows)
            ──►  Redis              (cache-aside on hot results)
                       │  query router picks the right engine per tool
                       ▼
                   MCP server  ──►  LLM client (e.g. Anthropic API MCP connector)
```

### Query routing: three tiers
A question is served by the cheapest tier that can answer it correctly:

| tier | serves | engine | measured |
|---|---|---|---|
| 1 | point lookup, case number, radius/spatial | Postgres/PostGIS | GiST index scan |
| 2 | aggregate on an aligned grain (month/quarter/year × one geography) | DuckDB **materialized rollups** | ~0.7 ms |
| 3 | aggregate on an arbitrary range ("last 30 days", Jan 5 – Feb 20) | DuckDB **live scan** over the same Parquet | ~4 ms |

Tier 3 is the honest part: the rollups are built at **month** grain, so they
genuinely cannot answer an arbitrary date range. Rather than pretend otherwise,
those queries fall through to a live DuckDB scan of the same Parquet — same
engine, same view, same answers (verified identical), just not pre-aggregated.
The rollups are a latency and cacheability tier — they're what Redis will front
— not a rescue for a slow query.

**Rollup design** (`store/duckdb/rollups.sql`), five tables + provenance, 849k
rows / 10 MB total, full rebuild in ~0.5 s over 2.88M incidents:

- **Month grain, not day.** Measured: day × type × beat yields 2.33M groups from
  2.88M rows — 81% of the row count, so no compression and no benefit.
- **One geography per table** (`citywide`, `beat`, `district`, `community_area`,
  `ward`). `district` gets its own table rather than riding along in
  `rollup_beat`: beat → district is only *nearly* a functional dependency — 248
  source rows (0.009%) place a beat in the wrong district, splitting 235
  (month, type, beat) buckets in two and silently undercounting for anyone
  filtering on both. A dedicated district table removes that failure mode by
  construction, and neither table infers geography from the other.
- **Counts, never rates.** `arrest_rate` is derived at read time as
  `arrests/incidents`; a stored rate is wrong the moment two buckets are summed.
- **Null geography buckets are kept**, so `SUM(incidents)` equals the source row
  count on every table — an invariant the tests assert.
- **`geocoded` is a measure** because the geocode rate is *systematically*
  biased, not uniform: 91.5% for offenses involving children and 92.2% for sex
  offenses (location suppressed for victim privacy) versus 100% for homicide,
  against a 98.4% overall rate.
  Spatial tools only ever see geocoded rows, so an aggregate count and a radius
  count legitimately disagree — per-bucket `geocoded` lets the result envelope
  say by how much, for the exact slice asked about.
- **Refresh is always a full rebuild.** At ~0.5 s there is no case for incremental
  rollup maintenance — a deliberate contrast with the Postgres loader, where a
  multi-million-row `COPY` did justify a separate upsert path.

### Why the server doesn't call Socrata live
Live API calls mean unpredictable latency, rate limits, no cross-dataset joins,
and no query-plan control. The interesting engineering lives in the ingestion
and serving layer, so incidents are ingested locally and served with routing.

### Designed for an agent to use correctly and fast
The tool surface bakes in five affordances (expanded on as they're built):
1. **Schema discovery** — a `describe_schema` tool exposes valid enum values and
   the available date range, so the model doesn't guess.
2. **Teaching errors** — malformed calls return structured, retry-safe errors
   that name the bad field and its valid values, creating a self-correcting loop.
3. **Entity resolution** — `resolve_neighborhood` maps fuzzy names to official
   geography instead of letting the model invent IDs.
4. **Result envelopes** — every response echoes the applied filters, row count, a
   truncation flag, a page cursor, and the data-provenance caveat inline.
5. **Bounded results + cursors** — hard caps with pagination so broad queries
   degrade gracefully instead of dumping millions of rows.

### Observability
Every tool call is logged as structured JSON (trace id, args, resolved query
plan, row count, latency, compacted result). A batch job rolls those logs into
**outcome/failure telemetry** — empty-result rates, `resolve_neighborhood`
misses, malformed-arg rates — treating failures as the product signal. (The
server only ever sees structured tool arguments, never the user's raw prompt.)

## Data & provenance

Source: **Crimes — 2001 to Present** (`ijzp-q8t2`) from the
[Chicago Data Portal](https://data.cityofchicago.org/), reported incidents
extracted from CPD's CLEAR system, plus boundary shapefiles (community areas,
wards, police beats).

**Coverage: 2,884,106 incidents, 2015-01-01 through 2026-07-22**, a contiguous
12-year window ingested as one Parquet partition per year. The full dataset
reaches back to 2001; a decade-plus exercises every query pattern without
tripling the storage. One row is one *offense*, not one incident — `case_number`
(the CPD RD number) is not unique, though in practice it very nearly is: only
0.01% of cases carry more than one offense row, so `id` is the real key and
`case_number` lookups return a list.

The feed **excludes the most recent 7 days**, so "today" is never in the data —
the max date above trails the current date by design, not by staleness.

Please read the city's caveats — they shape the design and are honored in tool
output:
- The data is **preliminary**; classifications can change on further
  investigation.
- CPD states it **should not be used for comparison purposes over time**.
- Addresses are shown at **block level** to protect victim privacy, and deriving
  specific addresses from map visualizations is **prohibited**.

### On comparing crime over time

Investigators will ask "is this going up?" — refusing to answer is not a real
option, so this project takes the caveat seriously rather than literally.

Read in full, CPD's disclaimer makes the comparison clause a *conclusion* of the
preceding sentence, not an independent finding: the data is unverified,
classifications may change, "and there is always the possibility of mechanical or
human error. **Therefore** … the information should not be used for comparison
purposes over time." That is liability language. Taken literally it forbids every
analysis, including the city's own published crime-trend datasets.

There are, however, four *specific* reasons a naive time comparison breaks here —
and unlike the blanket warning, each one is detectable, so the tool layer flags it
instead of hedging everything equally:

1. **Trailing-window incompleteness.** Beyond the 7-day exclusion, records keep
   arriving and being reclassified for weeks. A series ending "now" always slopes
   down. This is the single most likely way an agent states a confident falsehood.
2. **Offense-taxonomy drift.** The IUCR code set is not fixed: codes are
   introduced, retired, and — the awkward part — *gradually adopted*, so a
   category's definition can shift without any announcement. Code `0760`
   ("burglary from motor vehicle") first appears in Nov 2021 and ramps to 775
   rows in 2024, 3,528 in 2025, and 4,612 in the first half of 2026. That is
   0.3% of the dataset overall but **61% of 2026 burglary rows** — negligible
   globally, dominant locally, which is exactly the shape that corrupts a
   long-range trend without looking suspicious. Code `3400` ("looting") is the
   mirror image: it exists only between March and August 2020.

   The commonly cited version of this caveat is the NIBRS transition — CPD began
   submitting NIBRS in July 2021, and Illinois flags 2016–2022 as needing
   caution. **We checked whether that break is present in this dataset, and it is
   not.** Offenses per `case_number` sit at 1.0001 every year from 2015 to 2026,
   with no movement at monthly resolution around the transition: this feed is
   CPD's CLEAR/IUCR extract, not their NIBRS submission, so the hierarchy-rule
   change that alters rows-per-incident elsewhere never applied here. The
   comparability problem is real, but it is code-level churn, not a migration
   date — which matters, because a fix pinned to July 2021 would correct nothing.
3. **Geography boundaries move.** Police districts were consolidated in 2012 and
   beats redrawn; wards are redistricted every decade. A per-ward or per-beat
   series can break with no change in crime. **Community areas are the exception**
   — the 77 have been essentially fixed for a century, making them the only
   geography safe for long series.
4. **Reported ≠ occurred.** Reporting propensity moves with trust in police,
   online-reporting availability, and insurance requirements.

**What must not be "corrected".** Between the pre- and post-2021 halves of the
data, motor vehicle theft rises 4.6 percentage points of all offenses and
narcotics falls 2.4. Those are real — a national vehicle-theft surge and a
genuine decline in drug enforcement. Taxonomy drift is an artifact worth
normalizing away; enforcement and behaviour change *is the answer to the user's
question*. Any normalization that smooths the second while fixing the first has
destroyed the signal it was built to protect.

The design response, rather than a disclaimer nobody reads: the result envelope
marks a series as `provisional` when its last bucket falls in the unsettled
window, flags spans that cross a code's introduction or retirement (with the
share of rows affected), steers multi-year questions toward `community_area`, and
labels the measure as *offenses recorded in CLEAR* rather than *crime*. Where a
defensible mapping exists, an explicit opt-in re-groups drifting codes into a
stable category — opt-in, and always named in the envelope, because silently
returning normalized counts is the same failure as silently truncating results:
the model cannot reason about a transformation it was not told about. An agent
that reports a 40% drop because last month has not finished loading is precisely
the failure this architecture exists to prevent.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,store]"    # dev tooling + storage-layer deps ([server] lands in Phase 3)
cp .env.example .env             # then paste your Socrata app token
```

The `store` extra pulls the storage-layer drivers: `psycopg` (Postgres/PostGIS),
`redis`, and `duckdb`.

### Local storage stack (PostGIS + Redis)

The storage layer routes across Postgres/PostGIS (point lookups + spatial
queries) and Redis (cache-aside). `docker-compose.yml` brings both up locally:

```bash
docker compose up -d             # start PostGIS + Redis
docker compose ps                # both should read "Up (healthy)"
```

- Ports bind to `127.0.0.1` only, so the default dev credentials (`crime:crime`,
  and Redis' no-auth default) are not reachable off-machine.
- On Apple Silicon the local Postgres uses `imresamu/postgis` — a multi-arch
  rebuild of the amd64-only official PostGIS image (same PG 17 / PostGIS 3.5).
- `StoreConfig.from_env()` defaults mirror the compose file, so a fresh checkout
  connects with no `.env` changes; production overrides `DATABASE_URL` /
  `REDIS_URL`.

Load the Parquet dataset into Postgres:

```bash
chicago-crime-load                       # full refresh (rebuild the incidents table)
chicago-crime-load --mode upsert --years 2025 2026   # merge only changed partitions
```

Build the DuckDB OLAP rollups from the same Parquet (no services needed — DuckDB
is embedded; writes to `DUCKDB_PATH`, default `data/duckdb/crime.duckdb`):

```bash
chicago-crime-rollup                     # full rebuild of all five rollup tables
```

Run it after each ingest; it always rebuilds from scratch, so it is safe to
re-run at any time.

### Running the tests

```bash
pytest                    # unit tests only (default; no services needed)
pytest -m integration     # integration tests (need the storage stack up)
```

Integration tests run against a **dedicated `<db>_test` database** that is created
on demand, so they never touch the data you've loaded into the dev database
(override the target with `TEST_DATABASE_URL`).

### Explore the source data
```bash
python scripts/download_data.py peek            # 200-row sample + field inventory
python scripts/download_data.py year 2025       # full year -> data/crimes_2025.parquet
python scripts/download_data.py inspect data/crimes_2025.parquet
```

## Project layout

```
src/chicago_crime_mcp/
  ingest/      backfill + incremental sync from Socrata
  store/       Postgres/PostGIS, DuckDB, Redis + query routing
  server/      MCP tools (typed, constrained, purpose-built)
  geo/         boundary shapefiles + neighborhood resolution
  telemetry/   structured logging + failure telemetry
scripts/       exploration-stage data puller
tests/
```

## Roadmap

- [x] **Phase 0** — project scaffolding
- [ ] **Phase 1** — ingestion (explore field shapes, then checkpointed backfill)
- [ ] **Phase 2** — storage & query routing (Postgres/PostGIS, DuckDB, Redis)
- [ ] **Phase 3** — MCP tool surface + the five agent affordances
- [ ] **Phase 4** — observability, failure telemetry & tests
- [ ] **Phase 5** — deploy to Railway + Anthropic API MCP connector demo
