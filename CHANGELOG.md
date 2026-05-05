# Changelog

All notable changes to OnionDB will be documented in this file.

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
