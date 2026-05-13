# Changelog

All notable changes to OnionDB will be documented in this file.

## [0.6.0] — 2026-05-13

### Added
- **HNSW acceleration** — optional `hnswlib` backend for sub-10ms similarity search at 1M+ records per gap.
  - `enable_hnsw(dim=768, hnsw_threshold=1000, M=16, ef_construction=200, ef_search=50)` — activate per-gap HNSW indices.
  - **Two-phase query** — HNSW retrieves candidates → GRF v2 re-ranks with mass+decay.
  - **Auto-graduate** — brute-force below threshold, HNSW above. No config needed.
  - **Lazy loading** — cold gap indices are built on first query, not at startup.
  - **Zombie compaction** — deleted records are marked, auto-compacted at 10% ratio.
  - **Persistence** — `.hnsw` + `.hnsw.map` files saved alongside the `.db` file.
  - Fail-soft: if `hnswlib` is not installed, everything works without acceleration.
- **GRF v2 — Mass-weighted decay scoring** (Phase 5 schema).
  - `mass` column — reinforcement weight for records. Higher mass = stronger recall.
  - `last_review_clock` column — temporal marker for staleness-aware ranking.
  - `default_decay_rate` constructor parameter — enables time-decay scoring.
  - `current_clock` and `decay_rate` query parameters on `horizontal()` and `grf()`.
  - Scoring formula: `cosine × (1 + 0.2·log₁₊(mass)) × exp(-decay × Δclock)`.
  - Automatic schema migration — existing v0.5.0 databases gain `mass` and `last_review_clock` columns on first load.
- **`migrate_phase5.py`** — standalone migration script for Phase 5 schema.
- **71 new tests** (total: 164 across 3 suites).
  - `test_hnsw_grfv2.py` (24 tests) — HNSW backend, GRF v2 mass+decay, scaling benchmarks.
  - `test_query_coverage.py` (47 tests) — exhaustive validation of every query method with Phase 5 fields.

### Changed
- `insert()` now accepts `mass` and `last_review_clock` parameters.
- `horizontal()` and `grf()` accept `current_clock` and `decay_rate` for homeostatic scoring.
- README updated with Phase 5 documentation, HNSW architecture, scaling tables (10M+).
- Development Status upgraded from Alpha to Beta.
- `pyproject.toml` adds `[hnsw]` and `[all]` optional dependency groups.
- Install: `pip install oniondb[all]` for numpy + hnswlib acceleration.

## [0.5.0] — 2026-05-07

### Added
- **SimHash — 64-bit locality-sensitive hashing** for high-dimensional search beyond the 2D cell grid.
  - `fit_simhash(n_bits=64, seed=42)` — generate random hyperplanes and compute hashes for all records. Numpy-accelerated batch computation with pure-Python fallback.
  - `simhash_query(embedding, max_hamming, gap, k)` — search by Hamming distance pre-filter + cosine ranking. Finds cross-cell neighbors that PCA projection misses.
  - Auto-hash on `insert()` — new records are automatically hashed when planes are fitted.
  - `simhash` column added to `records` table with automatic schema migration.
  - Hyperplanes stored in SQLite `config` table for database portability.
- **`_hamming(a, b)`** — 64-bit masked Hamming distance computation.

### Fixed
- **`_hamming` infinite loop** on signed 64-bit integers — XOR of two negative Python ints produces infinite-precision values. Fixed with 64-bit mask (`& 0xFFFFFFFFFFFFFFFF`).

## [0.4.0] — 2026-05-07

### Added
- **`update(id, **kwargs)`** — partial record updates with automatic importance-cascade (gap/depth/cell recomputed when importance changes).
- **`bulk_delete(ids)`** — delete multiple records in a single transaction.
- **`iter_all(batch_size)`** — generator for memory-efficient bulk processing.
- **`export_jsonl(path)`** — export all records to portable JSONL format.
- **`import_jsonl(path, chunk_size)`** — streaming JSONL import in chunks (default 1000) to avoid OOM on large files.
- **`assign_temporal_gaps()`** — auto-bucket records into temporal shells by origin_date.
- **Pagination** — `offset` parameter on `shell_scan()` and `range_scan()`.
- **`category` filter on `temporal_grf()`** — now consistent with all other query methods.
- **Input validation** — `insert()` raises `ValueError` for empty id, None content, or importance outside [0.0, 1.0].
- **PCA stored in SQLite** — projection matrix saved in `config` table for total database portability.
- **`metadata` and `origin_date`** — now included in all query results (was missing from `_RECORD_COLS`).

### Fixed
- **`fit_projection()` crash on mixed embedding dimensions** — now filters to majority dimension and reports `skipped_dim_mismatch` count. Previously crashed with numpy `inhomogeneous shape` error when rogue records had different embedding sizes.
- **Schema migration** — automatic `memories` → `records` table rename on first load for backward compatibility with legacy databases.
- **`batch_insert()` silently dropped `origin_date`** — export→import roundtrips now preserve temporal metadata correctly.
- **CLI `info` misreported PCA status** — now checks the SQLite `config` table (where v0.4.0 stores PCA) instead of only looking for an external JSON file.

### Changed
- Timestamps use timezone-aware UTC (`datetime.now(timezone.utc)`) instead of naive `datetime.utcnow()`.
- `import_jsonl()` now streams in chunks instead of loading entire file into memory.
- `insert()` docstring now documents upsert behavior (existing IDs are silently replaced).

## [0.3.3] — 2026-05-05

### Fixed
- **README** — restored full documentation (hero image, use cases, benchmarks, technical details) lost in previous update.
- **Install section** — separated `pip install` commands into individual code blocks for clean copy-paste.
- **API Reference** — added missing `close()` method and context manager usage.
- **Scaling table** — fixed emoji rendering (❎ → ❌) for 1M+ row.

### Changed
- CHANGELOG updated with CI stability fixes (concurrent reads, coverage config).

## [0.3.2] — 2026-05-05

### Added
- **93 tests** — expanded from 60: parametrized boundary mapping, edge cases (importance 0.0/1.0), data integrity checks, CLI integration tests.
- **`N_TEMPORAL_GAPS` class constant** — replaces magic number in `temporal_grf()`.
- **`from __future__ import annotations`** — enables precise generic type hints (`tuple[int, int]`, `list[dict]`).
- **GitHub Actions CI** — full matrix (Python 3.9–3.13), numpy coexistence job, coverage gate (≥80%).

### Fixed
- **Example bug** — `basic_usage.py` and `quickstart.py` referenced non-existent `similarity` field (should be `score`).
- **Reindex N+1 query** — `reindex()` no longer runs per-record sub-queries for angle lookup; theta/phi included in initial SELECT.
- **Concurrent reads test** — `test_concurrent_reads` now uses per-thread database connections (fixes SQLite `InterfaceError` on Python 3.12+/Linux).
- **PEP 8 imports** — consolidated `import os` to module top in both examples.
- **Crash-safe examples** — both examples now clean up stale `.db` files on startup.

### Changed
- More specific type hints on private methods (`-> tuple[int, int]` instead of `-> tuple`).
- Misleading comments clarified ("importance level" → "gap (importance band)").

## [0.3.0] — 2026-05-05

### Added
- **Optional numpy acceleration** — cosine similarity ~10x faster, BLOB decoding ~5x faster when numpy is installed. Auto-detected at import time. Install via `pip install oniondb[fast]`.
- **Configurable grid resolution** — `OnionDB(theta_cells=24, phi_cells=12)` for custom grid sizing. Default 12×6 unchanged.
- **Beam search reverse ray** — `reverse_ray(embedding, beam_width=3)` explores multiple paths simultaneously instead of greedy descent.
- **CLI inspector** — `python -m oniondb stats|info|density|shell mydb.db` for terminal-based database inspection.
- **Dynamic shell boundaries** — `fit_boundaries(n_gaps=5)` suggests quantile-based boundaries from data distribution.
- **Reindex** — `reindex(boundaries=new_bounds)` recalculates all gap/depth/cell assignments in-place.
- **Numpy-accelerated PCA** — `fit_projection()` uses `np.linalg.eigh` when numpy is available (~50x faster than power iteration).
- **GitHub Actions CI** — automated tests on Python 3.9, 3.12, 3.14 with and without numpy.

### Changed
- Version bump to 0.3.0
- `pyproject.toml` now includes `[fast]` optional dependency group

## [0.2.0] — 2026-05-04

### Added
- Professional README with full API documentation
- Technical details (PCA, GRF, Reverse Ray, Curvature)
- Use case examples (AI agents, log analysis, knowledge management)
- Benchmark results vs flat vector search
- Comparison table vs FAISS, ChromaDB, Pinecone, pgvector
- Comprehensive test suite (38 tests)
- `pyproject.toml` for PyPI packaging
- GitHub topics and release

## [0.1.0] — 2026-05-03

### Added
- Initial release
- Core engine: insert, get, delete, count, batch_insert
- 6 query operations: horizontal, GRF, reverse_ray, temporal_grf, shell_scan, range_scan
- PCA self-calibrating projection (`fit_projection`)
- SQLite-backed storage with WAL mode
- Thread-safe operations with RLock
- Zero external dependencies
