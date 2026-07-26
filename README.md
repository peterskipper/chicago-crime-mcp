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
