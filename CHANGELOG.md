# Changelog

All notable changes to OnionDB will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-04

### Added
- Core `OnionDB` class with SQLite-backed geometric storage
- 4-part addressing: `(gap, θ, φ, depth)` — every data point has a physical location
- 6 query operations:
  - `horizontal()` — find nearby items within one shell
  - `grf()` — Geometric Ray Filter: drill through all importance shells at a direction
  - `reverse_ray()` — curved semantic gravity trace from outer to inner shells
  - `temporal_grf()` — drill through time-based shells
  - `shell_scan()` — return everything at one importance level
  - `range_scan()` — return everything between two importance levels
- PCA-based embedding projection to spherical coordinates (88% cell occupancy)
- Fallback v0 projection for use without PCA calibration
- `fit_projection()` — self-calibrating PCA from stored embeddings
- CRUD operations: `insert()`, `get()`, `delete()`, `batch_insert()`, `count()`
- Subshell clustering with soft/hard boost for topic-aware retrieval
- Cosine similarity ranking when embeddings provided
- SQLite WAL mode for concurrent read access
- Zero external dependencies — stdlib only
- Context manager support (`with OnionDB(...) as db:`)
