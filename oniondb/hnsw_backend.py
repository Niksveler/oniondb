# ╔══════════════════════════════════════════════════════════════╗
# ║  hnsw_backend.py — HNSW Acceleration Layer for OnionDB     ║
# ║                                                              ║
# ║  SYSTEM:   oniondb / acceleration                            ║
# ║  PURPOSE:  Per-gap HNSW indices for sub-10ms similarity      ║
# ║            search at 1M+ records, with auto-graduate from    ║
# ║            brute-force at small scale and lazy per-gap       ║
# ║            loading for fast cold start.                      ║
# ║  READS:    embeddings from OnionDB SQLite                    ║
# ║  WRITES:   .hnsw index files alongside .db file              ║
# ║  CLI:      n/a (library module)                              ║
# ║  NOT:      Not a standalone database. Not a replacement for  ║
# ║            SQLite. HNSW is pure acceleration — SQLite is     ║
# ║            source of truth. Rebuild from SQLite on crash.    ║
# ║                                                              ║
# ║  MODIFIED: Venus IDE 2026-05-12                              ║
# ╚══════════════════════════════════════════════════════════════╝
"""
OnionDB HNSW Backend — Optional acceleration layer.

Provides per-gap HNSW indices for fast approximate nearest neighbor
search. Designed to be used as Phase 1 of OnionDB's two-phase search:

  Phase 1 (HNSW): Fast candidate retrieval (~3ms at 1M records)
  Phase 2 (GRF):  Physics-based re-ranking (shell_weight, mass, decay)

Zero-dependency: if hnswlib is not installed, OnionDB falls back
to the v1 cell-based SQL scan automatically.

Decision Log:
  - Per-gap indices over global: GRF guarantees k-per-shell
  - M=16 uniform: recall@50 >0.99 at 1M/768D with k×5 over-fetch
  - ef = max(k*5, 100): prevents silent recall degradation
  - hnsw_threshold=1000: brute-force faster below this
  - auto_compact=True: silent recall degradation harder to debug
  - Lazy per-gap loading: inner shells load instantly
"""
from __future__ import annotations

import os
import struct
import logging
from typing import Optional

try:
    import hnswlib
    _HAS_HNSW = True
except ImportError:
    _HAS_HNSW = False

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

_log = logging.getLogger("oniondb.hnsw")


class HNSWBackend:
    """
    Per-gap HNSW acceleration for OnionDB.

    Each gap gets its own HNSW index. Indices are lazy-loaded on first
    query and persisted as .hnsw files alongside the database.

    Usage:
        backend = HNSWBackend("onion.db", n_gaps=5, dim=768)
        backend.add(gap=0, record_id="rec_001", embedding=[...])
        ids, dists = backend.query(gap=0, embedding=[...], k=10)
    """

    def __init__(
        self,
        db_path: str,
        n_gaps: int,
        dim: int = 768,
        M: int = 16,
        ef_construction: int = 200,
        hnsw_threshold: int = 1000,
        auto_compact: bool = True,
        zombie_threshold: float = 0.10,
    ):
        """
        Initialize the HNSW backend.

        Args:
            db_path: Path to the OnionDB .db file (index files stored alongside).
            n_gaps: Number of shell gaps in OnionDB.
            dim: Embedding dimensionality (default: 768).
            M: HNSW graph connectivity parameter (default: 16).
                Higher M = better recall but more memory.
                M=16 gives recall@50 >0.99 at 1M/768D.
            ef_construction: Build-time search depth (default: 200).
            hnsw_threshold: Min records per gap before HNSW activates.
                Below this, brute-force cosine is faster (default: 1000).
            auto_compact: Auto-rebuild indices when zombie ratio exceeds
                zombie_threshold (default: True). Set False for predictable
                latency — call maintenance() explicitly instead.
            zombie_threshold: Fraction of deleted entries that triggers
                compaction (default: 0.10 = 10%).
        """
        if not _HAS_HNSW:
            raise ImportError(
                "hnswlib is required for HNSW acceleration. "
                "Install with: pip install hnswlib"
            )

        self.db_path = db_path
        self.n_gaps = n_gaps
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.hnsw_threshold = hnsw_threshold
        self.auto_compact = auto_compact
        self.zombie_threshold = zombie_threshold

        # Per-gap state
        self._indices: dict[int, hnswlib.Index] = {}
        self._id_to_int: dict[int, dict[str, int]] = {}   # gap → {record_id: int_label}
        self._int_to_id: dict[int, dict[int, str]] = {}   # gap → {int_label: record_id}
        self._next_label: dict[int, int] = {}              # gap → next available int label
        self._zombie_counts: dict[int, int] = {}           # gap → count of mark_deleted items
        self._loaded_gaps: set[int] = set()                # gaps whose index is in memory
        self._gap_sizes: dict[int, int] = {}               # gap → logical record count

    # ─── Index file paths ───────────────────────────────────

    def _index_path(self, gap: int) -> str:
        """Path to the .hnsw file for a specific gap."""
        base = os.path.splitext(self.db_path)[0]
        return f"{base}_gap{gap}.hnsw"

    def _map_path(self, gap: int) -> str:
        """Path to the ID mapping file for a specific gap."""
        base = os.path.splitext(self.db_path)[0]
        return f"{base}_gap{gap}.hnsw.map"

    # ─── Core operations ────────────────────────────────────

    def add(self, gap: int, record_id: str, embedding: list | bytes) -> None:
        """
        Add a record's embedding to the gap's HNSW index.

        Creates the index if it doesn't exist yet.

        Args:
            gap: Shell gap index (0 = innermost).
            record_id: The record's string ID from SQLite.
            embedding: 768D float vector (list or blob).
        """
        vec = self._to_array(embedding)
        if vec is None:
            return

        # Initialize index for this gap if needed
        if gap not in self._indices:
            self._init_gap(gap)

        idx = self._indices[gap]
        label = self._next_label.get(gap, 0)

        # Resize if needed
        current_max = idx.get_max_elements()
        if label >= current_max:
            new_size = max(current_max * 2, 1024)
            idx.resize_index(new_size)

        idx.add_items(vec.reshape(1, -1), [label])
        self._id_to_int.setdefault(gap, {})[record_id] = label
        self._int_to_id.setdefault(gap, {})[label] = record_id
        self._next_label[gap] = label + 1
        self._gap_sizes[gap] = self._gap_sizes.get(gap, 0) + 1

    def delete(self, gap: int, record_id: str) -> bool:
        """
        Mark a record as deleted in the gap's HNSW index.

        Does NOT reclaim space — call rebuild_gap() to compact.

        Args:
            gap: Shell gap index.
            record_id: The record's string ID.

        Returns:
            True if the record was found and deleted.
        """
        if gap not in self._id_to_int:
            return False
        label = self._id_to_int[gap].get(record_id)
        if label is None:
            return False

        self._indices[gap].mark_deleted(label)
        del self._id_to_int[gap][record_id]
        del self._int_to_id[gap][label]
        self._zombie_counts[gap] = self._zombie_counts.get(gap, 0) + 1
        self._gap_sizes[gap] = max(0, self._gap_sizes.get(gap, 0) - 1)
        return True

    def query(
        self,
        gap: int,
        embedding: list | bytes,
        k: int = 50,
        ef: int | None = None,
    ) -> tuple[list[str], list[float]]:
        """
        Find k nearest neighbors in a gap's HNSW index.

        Args:
            gap: Shell gap index.
            embedding: Query vector.
            k: Number of neighbors to return.
            ef: Query-time search depth. Higher = better recall, slower.
                Default: max(k * 5, 100). DO NOT set too low — causes
                silent recall degradation.

        Returns:
            (record_ids, distances) — IDs mapped back to string IDs.
            Empty lists if gap is below threshold or not loaded.
        """
        if gap not in self._indices:
            return [], []

        logical_size = self._gap_sizes.get(gap, 0)
        if logical_size == 0:
            return [], []

        # Clamp k to actual available records
        actual_k = min(k, logical_size)
        if actual_k <= 0:
            return [], []

        vec = self._to_array(embedding)
        if vec is None:
            return [], []

        idx = self._indices[gap]

        # Set ef — non-negotiable recall parameter
        query_ef = ef if ef is not None else max(k * 5, 100)
        idx.set_ef(query_ef)

        labels, distances = idx.knn_query(vec.reshape(1, -1), k=actual_k)

        # Map integer labels back to record IDs
        gap_map = self._int_to_id.get(gap, {})
        record_ids = []
        dist_list = []
        for lbl, dist in zip(labels[0], distances[0]):
            rid = gap_map.get(int(lbl))
            if rid is not None:  # skip zombies
                record_ids.append(rid)
                dist_list.append(float(dist))

        return record_ids, dist_list

    def index_size(self, gap: int) -> int:
        """Return the logical (non-deleted) record count for a gap."""
        return self._gap_sizes.get(gap, 0)

    def should_use_hnsw(self, gap: int) -> bool:
        """Check if HNSW should be used for this gap (auto-graduate)."""
        return self.index_size(gap) >= self.hnsw_threshold

    # ─── Lifecycle ──────────────────────────────────────────

    def _init_gap(self, gap: int, max_elements: int = 1024) -> None:
        """Create an empty HNSW index for a gap."""
        idx = hnswlib.Index(space="cosine", dim=self.dim)
        idx.init_index(
            max_elements=max_elements,
            M=self.M,
            ef_construction=self.ef_construction,
        )
        idx.set_ef(100)  # reasonable default
        self._indices[gap] = idx
        self._id_to_int[gap] = {}
        self._int_to_id[gap] = {}
        self._next_label[gap] = 0
        self._zombie_counts[gap] = 0
        self._gap_sizes[gap] = 0
        self._loaded_gaps.add(gap)

    def rebuild_gap(
        self,
        gap: int,
        records: list[tuple[str, list | bytes]],
    ) -> int:
        """
        Rebuild a gap's HNSW index from scratch.

        Compacts zombies and resets label counter.

        Args:
            gap: Shell gap index.
            records: List of (record_id, embedding) tuples from SQLite.

        Returns:
            Number of records indexed.
        """
        max_elements = max(len(records), 1024)
        self._init_gap(gap, max_elements=max_elements)

        count = 0
        for record_id, embedding in records:
            vec = self._to_array(embedding)
            if vec is not None:
                self.add(gap, record_id, vec)
                count += 1

        _log.info(f"Rebuilt gap {gap} index: {count} records")
        return count

    def needs_compaction(self, gap: int) -> bool:
        """Check if a gap's index has too many zombies."""
        zombies = self._zombie_counts.get(gap, 0)
        logical = self._gap_sizes.get(gap, 0)
        total = zombies + logical
        if total == 0:
            return False
        return (zombies / total) > self.zombie_threshold

    def maintenance(self, fetch_records_fn=None) -> dict[int, int]:
        """
        Compact all dirty gaps. Call after session end or on a cron.

        Args:
            fetch_records_fn: Callable(gap) -> list[(record_id, embedding)]
                Must be provided to rebuild from SQLite.

        Returns:
            Dict of {gap: records_rebuilt} for compacted gaps.
        """
        if fetch_records_fn is None:
            _log.warning("maintenance() called without fetch_records_fn — skipping")
            return {}

        results = {}
        for gap in range(self.n_gaps):
            if self.needs_compaction(gap):
                records = fetch_records_fn(gap)
                count = self.rebuild_gap(gap, records)
                results[gap] = count
                _log.info(f"Compacted gap {gap}: {count} records (was {self._zombie_counts.get(gap, 0)} zombies)")

        return results

    # ─── Persistence ────────────────────────────────────────

    def save(self) -> int:
        """
        Persist all loaded HNSW indices and ID maps to disk.

        Returns:
            Number of gaps saved.
        """
        saved = 0
        for gap in list(self._loaded_gaps):
            if gap in self._indices and self._gap_sizes.get(gap, 0) > 0:
                self._save_gap(gap)
                saved += 1
        return saved

    def _save_gap(self, gap: int) -> None:
        """Save a single gap's index and ID map."""
        idx_path = self._index_path(gap)
        map_path = self._map_path(gap)

        self._indices[gap].save_index(idx_path)

        # Save ID mapping as JSON-compatible format
        map_data = {
            "id_to_int": self._id_to_int.get(gap, {}),
            "next_label": self._next_label.get(gap, 0),
            "zombie_count": self._zombie_counts.get(gap, 0),
            "logical_size": self._gap_sizes.get(gap, 0),
        }
        import json
        with open(map_path, "w") as f:
            json.dump(map_data, f)

        _log.debug(f"Saved gap {gap}: {self._gap_sizes.get(gap, 0)} records → {idx_path}")

    def load_gap(self, gap: int) -> bool:
        """
        Load a gap's HNSW index from disk (lazy loading).

        Called on first query to a gap, not at startup.

        Args:
            gap: Shell gap index to load.

        Returns:
            True if loaded successfully, False if no index file exists.
        """
        if gap in self._loaded_gaps:
            return True

        idx_path = self._index_path(gap)
        map_path = self._map_path(gap)

        if not os.path.exists(idx_path) or not os.path.exists(map_path):
            return False

        try:
            import json
            with open(map_path, "r") as f:
                map_data = json.load(f)

            idx = hnswlib.Index(space="cosine", dim=self.dim)
            idx.load_index(idx_path)
            idx.set_ef(100)

            self._indices[gap] = idx
            self._id_to_int[gap] = map_data.get("id_to_int", {})
            # Rebuild reverse map
            self._int_to_id[gap] = {
                int(v): k for k, v in self._id_to_int[gap].items()
            }
            self._next_label[gap] = map_data.get("next_label", 0)
            self._zombie_counts[gap] = map_data.get("zombie_count", 0)
            self._gap_sizes[gap] = map_data.get("logical_size", 0)
            self._loaded_gaps.add(gap)

            _log.info(f"Loaded gap {gap} index: {self._gap_sizes[gap]} records")
            return True

        except Exception as e:
            _log.warning(f"Failed to load gap {gap} index: {e}")
            return False

    def ensure_loaded(self, gap: int) -> bool:
        """
        Ensure a gap's index is in memory (lazy load if needed).

        Returns True if index is available, False if no index exists.
        """
        if gap in self._loaded_gaps:
            return True
        return self.load_gap(gap)

    # ─── Utilities ──────────────────────────────────────────

    def _to_array(self, embedding) -> "np.ndarray | None":
        """Convert embedding (list, bytes, or ndarray) to float32 array."""
        if embedding is None:
            return None

        if _HAS_NUMPY:
            if isinstance(embedding, np.ndarray):
                return embedding.astype(np.float32)
            elif isinstance(embedding, (bytes, bytearray)):
                n = len(embedding) // 4
                values = struct.unpack(f"<{n}f", embedding)
                return np.array(values, dtype=np.float32)
            elif isinstance(embedding, (list, tuple)):
                return np.array(embedding, dtype=np.float32)
        else:
            # Fallback without numpy
            if isinstance(embedding, (bytes, bytearray)):
                n = len(embedding) // 4
                values = list(struct.unpack(f"<{n}f", embedding))
                return values
            elif isinstance(embedding, (list, tuple)):
                return list(embedding)

        return None

    def stats(self) -> dict:
        """Return statistics about all gap indices."""
        result = {}
        for gap in range(self.n_gaps):
            result[gap] = {
                "loaded": gap in self._loaded_gaps,
                "logical_size": self._gap_sizes.get(gap, 0),
                "zombies": self._zombie_counts.get(gap, 0),
                "needs_compaction": self.needs_compaction(gap),
                "uses_hnsw": self.should_use_hnsw(gap),
                "index_file_exists": os.path.exists(self._index_path(gap)),
            }
        return result

    def close(self) -> None:
        """Save all indices and clean up."""
        self.save()
        self._indices.clear()
        self._loaded_gaps.clear()
        _log.info("HNSW backend closed")
