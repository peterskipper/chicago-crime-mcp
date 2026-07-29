# chicago-crime-mcp

A constrained [MCP](https://modelcontextprotocol.io) server over the City of
Chicago's crime data, backed by a real **ingest → store → serve** pipeline. Ask
questions about Chicago crime in plain English (through an LLM client) and get
grounded, structured answers.

The design goal is a *system*, not a demo: a small surface of typed,
purpose-built tools with schema validation and deliberate query routing —
explicitly **not** a "hand the LLM a SQL prompt" text-to-SQL agent.

> **Status:** early construction. Project scaffolding is in place; ingestion is
> next. See the roadmap below.

## Architecture

```
Socrata (SODA API)
      │  paginated, checkpointed backfill + daily incremental (updated_on)
      ▼
   Parquet  ──►  Postgres/PostGIS   (point lookups, case-number, spatial joins)
            ──►  DuckDB + rollups   (OLAP aggregates over 8M+ rows)
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
| 2 | aggregate on an aligned grain (month/quarter/year × one geography) | DuckDB **materialized rollups** | ~1 ms |
| 3 | aggregate on an arbitrary range ("last 30 days", Jan 5 – Feb 20) | DuckDB **live scan** over the same Parquet | ~8 ms |

Tier 3 is the honest part: the rollups are built at **month** grain, so they
genuinely cannot answer an arbitrary date range. Rather than pretend otherwise,
those queries fall through to a live DuckDB scan of the same Parquet — same
engine, same view, same answers (verified identical), just not pre-aggregated.
The rollups are a latency and cacheability tier — they're what Redis will front
— not a rescue for a slow query.

**Rollup design** (`store/duckdb/rollups.sql`), five tables + provenance, 508k
rows / 3.9 MB total, full rebuild in ~0.5 s over 1.69M incidents:

- **Month grain, not day.** Measured: day × type × beat yields 1.37M groups from
  1.69M rows — 81% of the row count, so no compression and no benefit.
- **One geography per table** (`citywide`, `beat`, `district`, `community_area`,
  `ward`). `district` gets its own table rather than riding along in
  `rollup_beat`: beat → district is only *nearly* a functional dependency — 207
  source rows (0.012%) place a beat in the wrong district, which would split a
  single (month, type, beat) bucket in two and silently undercount for anyone
  filtering on both. A dedicated district table removes that failure mode by
  construction, and neither table infers geography from the other.
- **Counts, never rates.** `arrest_rate` is derived at read time as
  `arrests/incidents`; a stored rate is wrong the moment two buckets are summed.
- **Null geography buckets are kept**, so `SUM(incidents)` equals the source row
  count on every table — an invariant the tests assert.
- **`geocoded` is a measure** because the geocode rate is *systematically*
  biased, not uniform: 91.4% for offenses involving children and 92.5% for sex
  offenses (location suppressed for victim privacy) versus 100% for homicide.
  Spatial tools only ever see geocoded rows, so an aggregate count and a radius
  count legitimately disagree — per-bucket `geocoded` lets the result envelope
  say by how much, for the exact slice asked about.
- **Refresh is always a full rebuild.** At 0.5 s there is no case for incremental
  rollup maintenance — a deliberate contrast with the Postgres loader, where an
  8M-row `COPY` did justify a separate upsert path.

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
[Chicago Data Portal](https://data.cityofchicago.org/), ~8M+ reported incidents
extracted from CPD's CLEAR system, plus boundary shapefiles (community areas,
wards, police beats).

Please read the city's caveats — they shape the design and are honored in tool
output:
- The data is **preliminary**; classifications can change on further
  investigation.
- CPD states it **should not be used to compare crime over time**.
- Addresses are shown at **block level** to protect victim privacy, and deriving
  specific addresses from map visualizations is **prohibited**.

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
