# Changelog

All notable changes to OnionDB will be documented in this file.

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
