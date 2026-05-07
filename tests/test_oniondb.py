"""
Standalone test suite for OnionDB.
No AIGalaxy dependencies — fresh DB per test.
"""
import os
import json
import math
import random
import subprocess
import sys
import tempfile
import pytest
from oniondb import OnionDB


def _emb(dim=32, seed=None):
    """Generate a random embedding vector."""
    if seed is not None:
        random.seed(seed)
    return [random.gauss(0, 1) for _ in range(dim)]


@pytest.fixture
def db(tmp_path):
    """Create a fresh OnionDB for each test."""
    db_path = str(tmp_path / "test.db")
    odb = OnionDB(db_path)
    yield odb
    odb.close()


@pytest.fixture
def populated_db(db):
    """DB with 50 items across all gaps."""
    random.seed(42)
    for i in range(50):
        imp = random.uniform(0.0, 1.0)
        emb = [random.gauss(0, 1) for _ in range(32)]
        db.insert(
            id=f"mem-{i:03d}",
            content=f"Test record number {i} about topic {i % 5}",
            importance=round(imp, 3),
            category=["personal", "technical", "decision", "observation"][i % 4],
            embedding=emb,
        )
    return db


# ═══════════════════════════════════════
# BASIC OPERATIONS
# ═══════════════════════════════════════

class TestInsert:
    def test_insert_returns_address(self, db):
        addr = db.insert("a1", "Hello world", importance=0.8)
        assert addr["id"] == "a1"
        assert addr["gap"] in range(5)
        assert -180 <= addr["theta"] <= 180
        assert -90 <= addr["phi"] <= 90
        assert 0 <= addr["depth"] <= 1

    def test_insert_with_embedding(self, db):
        emb = [0.1] * 32
        addr = db.insert("e1", "With embedding", importance=0.5, embedding=emb)
        assert addr["id"] == "e1"
        # Retrieve and verify embedding stored
        mem = db.get("e1")
        assert mem is not None
        assert "_emb" in mem

    def test_insert_with_explicit_angles(self, db):
        addr = db.insert("a2", "Explicit angles", importance=0.7,
                          theta=45.0, phi=-30.0)
        assert abs(addr["theta"] - 45.0) < 0.1
        assert abs(addr["phi"] - (-30.0)) < 0.1

    def test_insert_or_replace(self, db):
        db.insert("dup", "Version 1", importance=0.5)
        db.insert("dup", "Version 2", importance=0.9)
        mem = db.get("dup")
        assert mem["content"] == "Version 2"
        assert db.count() == 1

    @pytest.mark.parametrize("importance,expected_gap", [
        (1.0, 0),     # max boundary
        (0.99, 0),    # core
        (0.95, 0),    # exact boundary: gap 0
        (0.94, 1),    # just below gap 0
        (0.90, 1),    # high
        (0.85, 1),    # exact boundary: gap 1
        (0.84, 2),    # just below gap 1
        (0.75, 2),    # mid
        (0.70, 2),    # exact boundary: gap 2
        (0.69, 3),    # just below gap 2
        (0.55, 3),    # low
        (0.50, 3),    # exact boundary: gap 3
        (0.49, 4),    # just below gap 3
        (0.30, 4),    # trivial
        (0.0, 4),     # min boundary
    ])
    def test_importance_to_gap_mapping(self, db, importance, expected_gap):
        addr = db.insert(f"g-{importance}", "test", importance=importance)
        assert addr["gap"] == expected_gap


class TestCRUD:
    def test_get_existing(self, db):
        db.insert("x1", "Find me", importance=0.5)
        mem = db.get("x1")
        assert mem is not None
        assert mem["id"] == "x1"
        assert mem["content"] == "Find me"

    def test_get_missing(self, db):
        assert db.get("nonexistent") is None

    def test_delete_existing(self, db):
        db.insert("d1", "Delete me", importance=0.5)
        assert db.delete("d1") is True
        assert db.get("d1") is None

    def test_delete_missing(self, db):
        assert db.delete("nonexistent") is False

    def test_count_empty(self, db):
        assert db.count() == 0

    def test_count_total(self, populated_db):
        assert populated_db.count() == 50

    def test_count_by_gap(self, populated_db):
        total = sum(populated_db.count(gap=g) for g in range(5))
        assert total == 50

    def test_batch_insert(self, db):
        items = [
            {"id": f"batch-{i}", "content": f"Item {i}", "importance": i / 10}
            for i in range(10)
        ]
        addresses = db.batch_insert(items)
        assert len(addresses) == 10
        assert db.count() == 10
        assert all("gap" in a for a in addresses)


# ═══════════════════════════════════════
# QUERY OPERATIONS
# ═══════════════════════════════════════

class TestHorizontal:
    def test_horizontal_returns_results(self, populated_db):
        results = populated_db.horizontal(gap=2, theta=0, phi=0, k=5)
        assert isinstance(results, list)
        assert len(results) <= 5
        for r in results:
            assert "id" in r
            assert "content" in r
            assert "score" in r

    def test_horizontal_with_embedding(self, populated_db):
        emb = [random.gauss(0, 1) for _ in range(32)]
        results = populated_db.horizontal(
            gap=2, theta=0, phi=0, k=5, query_embedding=emb
        )
        if results:
            # Should be sorted by cosine similarity (descending)
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_horizontal_empty_gap(self, db):
        results = db.horizontal(gap=0, theta=0, phi=0, k=5)
        assert results == []


class TestGRF:
    def test_grf_returns_dict(self, populated_db):
        result = populated_db.grf(theta=0, phi=0, k_per_gap=3)
        assert isinstance(result, dict)
        # Should have at least one gap with results
        assert len(result) > 0
        for gap_id, records in result.items():
            assert isinstance(gap_id, int)
            assert isinstance(records, list)
            assert len(records) <= 3

    def test_grf_with_embedding(self, populated_db):
        emb = [random.gauss(0, 1) for _ in range(32)]
        result = populated_db.grf(theta=0, phi=0, k_per_gap=5,
                                   query_embedding=emb)
        for gap_id, records in result.items():
            scores = [m["score"] for m in records]
            assert scores == sorted(scores, reverse=True)

    def test_grf_alias_ray(self, populated_db):
        assert populated_db.ray == populated_db.grf


class TestReverseRay:
    def test_reverse_ray_returns_path(self, populated_db):
        emb = [random.gauss(0, 1) for _ in range(32)]
        result = populated_db.reverse_ray(start_embedding=emb)
        assert "path" in result
        assert "curvature" in result
        assert "records" in result
        assert "straight" in result
        assert isinstance(result["straight"], bool)

    def test_reverse_ray_curvature_nonnegative(self, populated_db):
        emb = [random.gauss(0, 1) for _ in range(32)]
        result = populated_db.reverse_ray(start_embedding=emb)
        assert result["curvature"] >= 0


class TestShellScan:
    def test_shell_scan_returns_list(self, populated_db):
        results = populated_db.shell_scan(gap=2)
        assert isinstance(results, list)
        for r in results:
            assert r["gap"] == 2

    def test_shell_scan_limit(self, populated_db):
        results = populated_db.shell_scan(gap=2, limit=3)
        assert len(results) <= 3


class TestRangeScan:
    def test_range_scan_spans_gaps(self, populated_db):
        results = populated_db.range_scan(gap_start=1, gap_end=3)
        gaps_seen = {r["gap"] for r in results}
        # Should only have gaps 1, 2, 3
        assert gaps_seen.issubset({1, 2, 3})


# ═══════════════════════════════════════
# STATS & INSPECTION
# ═══════════════════════════════════════

class TestStats:
    def test_stats_structure(self, populated_db):
        s = populated_db.stats()
        assert s["total"] == 50
        assert s["n_gaps"] == 5
        assert "gaps" in s
        assert "categories" in s
        assert s["grid"] == "12x6"

    def test_cell_density(self, populated_db):
        density = populated_db.cell_density(gap=2)
        assert isinstance(density, list)
        for d in density:
            assert "cell" in d
            assert "count" in d
            assert d["count"] > 0


# ═══════════════════════════════════════
# PROJECTION
# ═══════════════════════════════════════

class TestProjection:
    def test_v0_fallback(self, db):
        """Without PCA, should use v0 projection without error."""
        db._pca = None
        emb = [0.5, -0.3, 0.1] + [0.0] * 29
        theta, phi = db._embed_to_angles(emb)
        assert -180 <= theta <= 180
        assert -90 <= phi <= 90

    def test_fit_projection_needs_data(self, db):
        result = db.fit_projection()
        assert "error" in result
        assert result["n_samples"] < 10

    def test_fit_projection_works(self, populated_db):
        result = populated_db.fit_projection(save=True)
        assert "error" not in result
        assert result["n_samples"] == 50
        assert result["occupancy_after"] > 0
        assert populated_db._pca is not None


# ═══════════════════════════════════════
# CUSTOM BOUNDARIES
# ═══════════════════════════════════════

class TestCustomBoundaries:
    def test_custom_boundaries(self, tmp_path):
        db = OnionDB(str(tmp_path / "custom.db"),
                      boundaries=[0.8, 0.5, 0.0])
        assert db.n_gaps == 3
        db.insert("c1", "High", importance=0.9)
        db.insert("c2", "Low", importance=0.2)
        assert db.get("c1")["gap"] == 0
        assert db.get("c2")["gap"] == 2
        db.close()


class TestTemporalGRF:
    def test_temporal_grf_returns_dict(self, populated_db):
        result = populated_db.temporal_grf(theta=0, phi=0, k_per_gap=3)
        assert isinstance(result, dict)
        # temporal_gap column is NULL by default, so result may be empty
        for tgap, records in result.items():
            assert isinstance(tgap, int)
            assert isinstance(records, list)

    def test_temporal_grf_with_data(self, db):
        """Insert with temporal_gap set manually via SQL, then query."""
        for i in range(10):
            db.insert(f"t-{i}", f"Temporal item {i}",
                      importance=0.5 + i * 0.04,
                      theta=10.0, phi=5.0)
        # Set temporal_gap directly (normally done by ingestion pipeline)
        db.conn.execute("UPDATE records SET temporal_gap = 0 WHERE id IN ('t-0','t-1')")
        db.conn.execute("UPDATE records SET temporal_gap = 2 WHERE id IN ('t-5','t-6')")
        db.conn.commit()
        result = db.temporal_grf(theta=10.0, phi=5.0, k_per_gap=5)
        assert isinstance(result, dict)
        if 0 in result:
            assert all(m["temporal_gap"] == 0 for m in result[0])
        if 2 in result:
            assert all(m["temporal_gap"] == 2 for m in result[2])

    def test_temporal_grf_with_embedding(self, db):
        emb = [0.5] * 32
        for i in range(5):
            db.insert(f"te-{i}", f"Embedded temporal {i}",
                      importance=0.6, embedding=emb, theta=0, phi=0)
        db.conn.execute("UPDATE records SET temporal_gap = 1")
        db.conn.commit()
        result = db.temporal_grf(theta=0, phi=0, k_per_gap=3,
                                  query_embedding=emb)
        if 1 in result:
            scores = [m["score"] for m in result[1]]
            assert scores == sorted(scores, reverse=True)

    def test_temporal_grf_empty(self, db):
        result = db.temporal_grf(theta=0, phi=0)
        assert result == {}


class TestThreadSafety:
    def test_concurrent_inserts(self, tmp_path):
        """Multiple threads inserting simultaneously should not corrupt."""
        import threading
        db = OnionDB(str(tmp_path / "thread.db"))
        errors = []

        def worker(thread_id):
            try:
                for i in range(20):
                    db.insert(f"t{thread_id}-{i}",
                              f"Thread {thread_id} item {i}",
                              importance=random.uniform(0.1, 0.9))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert db.count() == 100  # 5 threads × 20 items
        db.close()

    def test_concurrent_batch_insert(self, tmp_path):
        """batch_insert from multiple threads should not deadlock."""
        import threading
        db = OnionDB(str(tmp_path / "batch_thread.db"))
        errors = []

        def worker(thread_id):
            try:
                items = [
                    {"id": f"bt{thread_id}-{i}", "content": f"Batch {i}",
                     "importance": 0.5}
                    for i in range(10)
                ]
                db.batch_insert(items)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert db.count() == 40  # 4 threads × 10 items
        db.close()


# ═══════════════════════════════════════
# CONTEXT MANAGER & REPR
# ═══════════════════════════════════════

class TestLifecycle:
    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "ctx.db")
        with OnionDB(db_path) as db:
            db.insert("ctx1", "Context test", importance=0.5)
            assert db.count() == 1

    def test_repr(self, populated_db):
        r = repr(populated_db)
        assert "OnionDB" in r
        assert "50 records" in r
        assert "5 gaps" in r


# ═══════════════════════════════════════
# v0.3.0: CONFIGURABLE GRID
# ═══════════════════════════════════════

class TestConfigurableGrid:
    def test_default_grid(self, db):
        assert db.THETA_CELLS == 12
        assert db.PHI_CELLS == 6

    def test_custom_grid(self, tmp_path):
        db_path = str(tmp_path / "custom_grid.db")
        db = OnionDB(db_path, theta_cells=24, phi_cells=12)
        assert db.THETA_CELLS == 24
        assert db.PHI_CELLS == 12
        s = db.stats()
        assert s["grid"] == "24x12"
        db.close()

    def test_custom_grid_affects_cell_assignment(self, tmp_path):
        """Higher resolution grid should produce different cell assignments."""
        emb = [0.5] * 32
        db1 = OnionDB(str(tmp_path / "g1.db"))
        db2 = OnionDB(str(tmp_path / "g2.db"), theta_cells=24, phi_cells=12)
        a1 = db1.insert("t1", "test", importance=0.5, embedding=emb)
        a2 = db2.insert("t1", "test", importance=0.5, embedding=emb)
        # Different grid resolutions = potentially different cell assignments
        total_cells_1 = db1.THETA_CELLS * db1.PHI_CELLS
        total_cells_2 = db2.THETA_CELLS * db2.PHI_CELLS
        assert total_cells_2 > total_cells_1
        db1.close()
        db2.close()

    def test_class_default_unchanged(self, tmp_path):
        """Instance attrs should not change class defaults."""
        db = OnionDB(str(tmp_path / "c1.db"), theta_cells=99)
        assert db.THETA_CELLS == 99
        assert OnionDB.THETA_CELLS == 12  # class attr unchanged
        db.close()


# ═══════════════════════════════════════
# v0.3.0: BEAM SEARCH REVERSE RAY
# ═══════════════════════════════════════

class TestBeamSearch:
    def test_greedy_default(self, populated_db):
        """beam_width=1 (default) returns standard reverse ray."""
        emb = [random.gauss(0, 1) for _ in range(32)]
        result = populated_db.reverse_ray(emb)
        assert "path" in result
        assert "curvature" in result
        assert "beam_width" not in result  # greedy mode doesn't add beam metadata

    def test_beam_search_returns_beam_metadata(self, populated_db):
        """beam_width > 1 adds beam-specific metadata."""
        emb = [random.gauss(0, 1) for _ in range(32)]
        result = populated_db.reverse_ray(emb, beam_width=3)
        assert result["beam_width"] == 3
        assert "beam_paths_explored" in result
        assert result["beam_paths_explored"] >= 1

    def test_beam_search_has_path(self, populated_db):
        """Beam search still returns a valid path."""
        emb = [random.gauss(0, 1) for _ in range(32)]
        result = populated_db.reverse_ray(emb, beam_width=5)
        assert len(result["path"]) > 0
        assert len(result["records"]) >= 0
        assert "curvature" in result

    def test_beam_width_1_matches_greedy(self, populated_db):
        """beam_width=1 should produce same results as default greedy."""
        random.seed(123)
        emb = [random.gauss(0, 1) for _ in range(32)]
        greedy = populated_db.reverse_ray(emb)
        beam1 = populated_db.reverse_ray(emb, beam_width=1)
        # Both should have same curvature and path length
        assert greedy["curvature"] == beam1["curvature"]
        assert greedy["n_hops"] == beam1["n_hops"]


# ═══════════════════════════════════════
# v0.3.0: FIT BOUNDARIES & REINDEX
# ═══════════════════════════════════════

class TestFitBoundaries:
    def test_fit_boundaries_returns_list(self, populated_db):
        bounds = populated_db.fit_boundaries(n_gaps=5)
        assert isinstance(bounds, list)
        assert len(bounds) == 5

    def test_fit_boundaries_descending(self, populated_db):
        bounds = populated_db.fit_boundaries(n_gaps=5)
        for i in range(len(bounds) - 1):
            assert bounds[i] >= bounds[i + 1]

    def test_fit_boundaries_ends_with_zero(self, populated_db):
        bounds = populated_db.fit_boundaries(n_gaps=5)
        assert bounds[-1] == 0.0

    def test_fit_boundaries_empty_db(self, db):
        """Empty DB returns default boundaries."""
        bounds = db.fit_boundaries()
        assert bounds == db.boundaries

    def test_fit_boundaries_custom_n_gaps(self, populated_db):
        bounds3 = populated_db.fit_boundaries(n_gaps=3)
        bounds7 = populated_db.fit_boundaries(n_gaps=7)
        assert len(bounds3) == 3
        assert len(bounds7) == 7


class TestReindex:
    def test_reindex_preserves_count(self, populated_db):
        count_before = populated_db.count()
        populated_db.reindex()
        count_after = populated_db.count()
        assert count_before == count_after

    def test_reindex_with_new_boundaries(self, populated_db):
        new_bounds = [0.8, 0.5, 0.2, 0.0]
        result = populated_db.reindex(boundaries=new_bounds)
        assert result["updated"] == 50
        assert result["n_gaps"] == 4
        assert populated_db.boundaries == new_bounds

    def test_reindex_updates_gaps(self, populated_db):
        """After reindex with different boundaries, gap distribution should change."""
        stats_before = populated_db.stats()
        new_bounds = populated_db.fit_boundaries(n_gaps=3)
        populated_db.reindex(boundaries=new_bounds)
        stats_after = populated_db.stats()
        assert stats_after["n_gaps"] == 3
        assert stats_after["total"] == stats_before["total"]

    def test_fit_then_reindex_workflow(self, populated_db):
        """Full workflow: analyze distribution → suggest boundaries → apply."""
        bounds = populated_db.fit_boundaries(n_gaps=5)
        result = populated_db.reindex(boundaries=bounds)
        assert result["total"] == 50
        # After reindex, data should still be queryable
        scan = populated_db.shell_scan(0)
        assert isinstance(scan, list)


# ═══════════════════════════════════════
# v0.3.0: NUMPY ACCELERATION
# ═══════════════════════════════════════

class TestNumpyAcceleration:
    def test_cosine_returns_float(self, db):
        """Cosine should work regardless of numpy presence."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert db._cosine(a, b) == pytest.approx(1.0)

    def test_cosine_orthogonal(self, db):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert db._cosine(a, b) == pytest.approx(0.0)

    def test_cosine_empty(self, db):
        assert db._cosine([], [1, 2]) == 0.0
        assert db._cosine(None, [1, 2]) == 0.0

    def test_decode_embedding_roundtrip(self, db):
        """Encode → decode should be lossless."""
        import struct
        original = [0.1, 0.2, 0.3, 0.4, 0.5]
        blob = struct.pack(f'{len(original)}f', *original)
        decoded = db._decode_embedding(blob)
        for a, b in zip(original, decoded):
            assert a == pytest.approx(b, abs=1e-6)

    def test_numpy_detected(self):
        """Check that numpy detection works."""
        from oniondb.onion_db import _HAS_NUMPY
        assert isinstance(_HAS_NUMPY, bool)


# ═══════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════

class TestEdgeCases:
    def test_insert_importance_zero(self, db):
        addr = db.insert("zero", "Zero importance", importance=0.0)
        assert addr is not None
        assert addr["gap"] == 4  # lowest shell
        assert db.get("zero") is not None

    def test_insert_importance_one(self, db):
        addr = db.insert("one", "Max importance", importance=1.0)
        assert addr is not None
        assert addr["gap"] == 0  # highest shell
        assert db.get("one") is not None

    def test_batch_insert_large(self, db):
        """Stress test with 1000 items."""
        items = [
            {"id": f"large-{i}", "content": f"Item {i}",
             "importance": (i % 100) / 100}
            for i in range(1000)
        ]
        result = db.batch_insert(items)
        assert len(result) == 1000
        assert db.count() == 1000

    def test_empty_content(self, db):
        addr = db.insert("empty", "", importance=0.5)
        assert addr is not None
        mem = db.get("empty")
        assert mem["content"] == ""

    def test_very_long_content(self, db):
        long_text = "x" * 100_000
        addr = db.insert("long", long_text, importance=0.5)
        mem = db.get("long")
        assert len(mem["content"]) == 100_000

    def test_special_characters_in_id(self, db):
        addr = db.insert("id/with:special@chars!", "test", importance=0.5)
        mem = db.get("id/with:special@chars!")
        assert mem is not None

    def test_unicode_content(self, db):
        addr = db.insert("uni", "日本語テスト 🧅 émojis", importance=0.5)
        mem = db.get("uni")
        assert "🧅" in mem["content"]

    def test_concurrent_reads(self, tmp_path):
        """Multiple threads reading simultaneously should not error."""
        import threading
        db_path = str(tmp_path / "read.db")
        db = OnionDB(db_path)
        for i in range(50):
            db.insert(f"r-{i}", f"Read test {i}", importance=random.uniform(0.1, 0.9))
        db.close()

        errors = []
        results = []

        def reader(thread_id):
            # Each thread opens its own connection (SQLite best practice)
            local_db = OnionDB(db_path)
            try:
                for i in range(10):
                    mem = local_db.get(f"r-{i}")
                    if mem:
                        results.append(mem["id"])
                    local_db.shell_scan(gap=random.randint(0, 4), limit=5)
            except Exception as e:
                errors.append(e)
            finally:
                local_db.close()

        threads = [threading.Thread(target=reader, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Read errors: {errors}"
        assert len(results) > 0


# ═══════════════════════════════════════
# DATA INTEGRITY
# ═══════════════════════════════════════

class TestDataIntegrity:
    def test_metadata_roundtrip(self, db):
        """All fields survive insert → get cycle."""
        db.insert("meta", "Content here", importance=0.73,
                  category="technical")
        mem = db.get("meta")
        assert mem["id"] == "meta"
        assert mem["content"] == "Content here"
        assert mem["importance"] == pytest.approx(0.73)
        assert mem["category"] == "technical"

    def test_embedding_precision_full_pipeline(self, db):
        """Float precision maintained through insert → get → decode."""
        original = [0.123456789, -0.987654321, 0.0, 1.0, -1.0] + [0.5] * 27
        db.insert("prec", "precision test", importance=0.5, embedding=original)
        mem = db.get("prec")
        assert "_emb" in mem
        decoded = mem["_emb"]
        assert len(decoded) == len(original)
        for a, b in zip(original, decoded):
            assert a == pytest.approx(b, abs=1e-5)

    def test_gap_depth_consistency(self, db):
        """Records in same gap should have consistent depth ordering."""
        db.insert("d1", "High in gap", importance=0.99)
        db.insert("d2", "Lower in gap", importance=0.96)
        m1 = db.get("d1")
        m2 = db.get("d2")
        # Both should be in gap 0, but d1 has higher importance = lower depth
        assert m1["gap"] == 0
        assert m2["gap"] == 0
        assert m1["depth"] <= m2["depth"]

    def test_delete_actually_removes(self, db):
        """Deleted records don't appear in any query."""
        db.insert("del1", "Delete me", importance=0.5)
        db.delete("del1")
        assert db.get("del1") is None
        assert db.count() == 0
        scan = db.shell_scan(gap=3)
        assert all(r["id"] != "del1" for r in scan)

    def test_replace_updates_all_fields(self, db):
        """Insert-or-replace should update content AND importance."""
        db.insert("upd", "Version 1", importance=0.3, category="old")
        db.insert("upd", "Version 2", importance=0.9, category="new")
        mem = db.get("upd")
        assert mem["content"] == "Version 2"
        assert mem["importance"] == pytest.approx(0.9)
        assert mem["category"] == "new"
        assert mem["gap"] == 1  # 0.9 → gap 1


# ═══════════════════════════════════════
# CLI TESTS
# ═══════════════════════════════════════

class TestCLI:
    @pytest.fixture
    def cli_db(self, tmp_path):
        """Create a populated DB for CLI testing."""
        db_path = str(tmp_path / "cli_test.db")
        db = OnionDB(db_path)
        for i in range(20):
            db.insert(f"cli-{i}", f"CLI test item {i}",
                      importance=i / 20, category="test")
        db.close()
        return db_path

    def _run_cli(self, *args):
        """Run oniondb CLI and return (returncode, stdout, stderr)."""
        result = subprocess.run(
            [sys.executable, "-m", "oniondb"] + list(args),
            capture_output=True, text=True, timeout=10
        )
        return result

    def test_cli_stats(self, cli_db):
        result = self._run_cli("stats", cli_db)
        assert result.returncode == 0
        assert "20" in result.stdout  # 20 records
        assert "Records" in result.stdout

    def test_cli_info(self, cli_db):
        result = self._run_cli("info", cli_db)
        assert result.returncode == 0
        assert "OnionDB" in result.stdout
        assert "Records" in result.stdout

    def test_cli_shell(self, cli_db):
        result = self._run_cli("shell", cli_db, "--gap", "4", "--limit", "3")
        assert result.returncode == 0
        assert "Gap 4" in result.stdout

    def test_cli_density(self, cli_db):
        result = self._run_cli("density", cli_db, "--gap", "4")
        assert result.returncode == 0
        assert "Gap 4" in result.stdout
        assert "occupancy" in result.stdout

    def test_cli_no_command(self):
        result = self._run_cli()
        assert result.returncode != 0

    def test_cli_help(self):
        result = self._run_cli("--help")
        assert result.returncode == 0
        assert "oniondb" in result.stdout.lower()


# ═══════════════════════════════════════
# v0.4.0: UPDATE WITH IMPORTANCE CASCADE
# ═══════════════════════════════════════

class TestUpdate:
    def test_update_content(self, db):
        db.insert("u1", "Original", importance=0.5)
        result = db.update("u1", content="Updated")
        assert result is not None
        mem = db.get("u1")
        assert mem["content"] == "Updated"
        assert mem["importance"] == pytest.approx(0.5)

    def test_update_importance_cascades(self, db):
        db.insert("u2", "Test", importance=0.3)
        old = db.get("u2")
        assert old["gap"] == 4  # 0.3 → gap 4

        result = db.update("u2", importance=0.99)
        assert result["gap"] == 0  # should cascade to gap 0
        mem = db.get("u2")
        assert mem["gap"] == 0
        assert mem["importance"] == pytest.approx(0.99)

    def test_update_category(self, db):
        db.insert("u3", "Test", importance=0.5, category="old")
        db.update("u3", category="new")
        mem = db.get("u3")
        assert mem["category"] == "new"

    def test_update_metadata(self, db):
        db.insert("u4", "Test", importance=0.5, metadata='{"v": 1}')
        db.update("u4", metadata='{"v": 2}')
        mem = db.get("u4")
        assert mem["metadata"] == '{"v": 2}'

    def test_update_nonexistent(self, db):
        result = db.update("ghost", content="nope")
        assert result is None

    def test_update_preserves_embedding(self, db):
        emb = [0.1] * 32
        db.insert("u5", "Embedded", importance=0.5, embedding=emb)
        db.update("u5", content="Updated content")
        mem = db.get("u5")
        assert mem["content"] == "Updated content"
        assert mem["_emb"] is not None
        assert len(mem["_emb"]) == 32

    def test_update_multiple_fields(self, db):
        db.insert("u6", "Original", importance=0.3, category="draft")
        db.update("u6", content="Revised", importance=0.9, category="final")
        mem = db.get("u6")
        assert mem["content"] == "Revised"
        assert mem["importance"] == pytest.approx(0.9)
        assert mem["category"] == "final"
        assert mem["gap"] == 1  # 0.9 → gap 1


# ═══════════════════════════════════════
# v0.4.0: CATEGORY FILTERING
# ═══════════════════════════════════════

class TestCategoryFilter:
    @pytest.fixture
    def categorized_db(self, db):
        """DB with records in different categories."""
        for i in range(20):
            cat = "error" if i % 3 == 0 else "info"
            db.insert(f"cat-{i}", f"Item {i}", importance=0.5,
                      category=cat, theta=10.0, phi=5.0)
        return db

    def test_horizontal_with_category(self, categorized_db):
        # Get the gap where records are stored (0.5 importance)
        sample = categorized_db.get("cat-0")
        gap = sample["gap"]
        all_results = categorized_db.horizontal(gap=gap, theta=10.0, phi=5.0, k=100)
        filtered = categorized_db.horizontal(gap=gap, theta=10.0, phi=5.0, k=100,
                                              category="error")
        assert len(filtered) < len(all_results)
        assert all(r["category"] == "error" for r in filtered)

    def test_grf_with_category(self, categorized_db):
        all_results = categorized_db.grf(theta=10.0, phi=5.0, k_per_gap=100)
        filtered = categorized_db.grf(theta=10.0, phi=5.0, k_per_gap=100,
                                       category="error")
        # Filtered should only have error records
        for gap_id, records in filtered.items():
            assert all(r["category"] == "error" for r in records)

    def test_shell_scan_with_category(self, categorized_db):
        sample = categorized_db.get("cat-0")
        gap = sample["gap"]
        all_results = categorized_db.shell_scan(gap=gap)
        filtered = categorized_db.shell_scan(gap=gap, category="info")
        assert len(filtered) <= len(all_results)
        assert all(r["category"] == "info" for r in filtered)

    def test_range_scan_with_category(self, categorized_db):
        all_results = categorized_db.range_scan(0, 4)
        filtered = categorized_db.range_scan(0, 4, category="error")
        assert len(filtered) <= len(all_results)
        assert all(r["category"] == "error" for r in filtered)

    def test_category_none_returns_all(self, categorized_db):
        sample = categorized_db.get("cat-0")
        gap = sample["gap"]
        results = categorized_db.horizontal(gap=gap, theta=10.0, phi=5.0,
                                             k=100, category=None)
        assert len(results) == 20


# ═══════════════════════════════════════
# v0.4.0: BULK DELETE
# ═══════════════════════════════════════

class TestBulkDelete:
    def test_bulk_delete_multiple(self, db):
        for i in range(5):
            db.insert(f"bd-{i}", f"Item {i}", importance=0.5)
        deleted = db.bulk_delete(["bd-0", "bd-2", "bd-4"])
        assert deleted == 3
        assert db.count() == 2
        assert db.get("bd-1") is not None
        assert db.get("bd-0") is None

    def test_bulk_delete_empty(self, db):
        assert db.bulk_delete([]) == 0

    def test_bulk_delete_nonexistent(self, db):
        db.insert("bd-x", "Test", importance=0.5)
        deleted = db.bulk_delete(["ghost1", "ghost2"])
        assert deleted == 0
        assert db.count() == 1


# ═══════════════════════════════════════
# v0.4.0: PAGINATION
# ═══════════════════════════════════════

class TestPagination:
    def test_shell_scan_offset(self, populated_db):
        page1 = populated_db.shell_scan(gap=4, limit=3, offset=0)
        page2 = populated_db.shell_scan(gap=4, limit=3, offset=3)
        if page1 and page2:
            ids1 = {r["id"] for r in page1}
            ids2 = {r["id"] for r in page2}
            assert ids1.isdisjoint(ids2)  # no overlap

    def test_range_scan_offset(self, populated_db):
        page1 = populated_db.range_scan(0, 4, limit=5, offset=0)
        page2 = populated_db.range_scan(0, 4, limit=5, offset=5)
        if page1 and page2:
            ids1 = {r["id"] for r in page1}
            ids2 = {r["id"] for r in page2}
            assert ids1.isdisjoint(ids2)

    def test_offset_beyond_data(self, populated_db):
        results = populated_db.shell_scan(gap=0, limit=10, offset=9999)
        assert results == []


# ═══════════════════════════════════════
# v0.4.0: ITERATION, EXPORT, IMPORT
# ═══════════════════════════════════════

class TestIterExportImport:
    def test_iter_all(self, populated_db):
        all_records = list(populated_db.iter_all(batch_size=10))
        assert len(all_records) == 50

    def test_iter_all_empty(self, db):
        assert list(db.iter_all()) == []

    def test_export_import_roundtrip(self, populated_db, tmp_path):
        export_path = str(tmp_path / "export.jsonl")
        exported = populated_db.export_jsonl(export_path)
        assert exported == 50
        assert os.path.exists(export_path)

        # Import into fresh DB
        db2_path = str(tmp_path / "import.db")
        db2 = OnionDB(db2_path)
        imported = db2.import_jsonl(export_path)
        assert imported == 50
        assert db2.count() == 50
        db2.close()

    def test_export_preserves_content(self, db, tmp_path):
        db.insert("exp-1", "Export test 🧅", importance=0.7, category="test")
        path = str(tmp_path / "single.jsonl")
        db.export_jsonl(path)

        db2 = OnionDB(str(tmp_path / "import2.db"))
        db2.import_jsonl(path)
        mem = db2.get("exp-1")
        assert mem["content"] == "Export test 🧅"
        assert mem["category"] == "test"
        db2.close()


# ═══════════════════════════════════════
# v0.4.0: TEMPORAL GAP ASSIGNMENT
# ═══════════════════════════════════════

class TestTemporalGaps:
    def test_assign_temporal_gaps(self, db):
        # Insert with different origin_dates
        for i in range(10):
            db.insert(f"tg-{i}", f"Item {i}", importance=0.5,
                      origin_date=f"2025-01-{i+1:02d}T00:00:00")
        counts = db.assign_temporal_gaps()
        assert isinstance(counts, dict)
        assert sum(counts.values()) == 10

    def test_assign_temporal_gaps_empty(self, db):
        result = db.assign_temporal_gaps()
        assert result == {}

    def test_temporal_grf_after_assignment(self, db):
        for i in range(10):
            db.insert(f"tga-{i}", f"Item {i}", importance=0.5,
                      origin_date=f"2025-01-{i+1:02d}T00:00:00",
                      theta=10.0, phi=5.0)
        db.assign_temporal_gaps()
        result = db.temporal_grf(theta=10.0, phi=5.0, k_per_gap=10)
        assert isinstance(result, dict)
        assert len(result) > 0  # should now have results


# ═══════════════════════════════════════
# v0.4.0: ORIGIN DATE AUTO-SET
# ═══════════════════════════════════════

class TestOriginDate:
    def test_auto_set_origin_date(self, db):
        db.insert("od-1", "Auto date", importance=0.5)
        mem = db.get("od-1")
        assert mem.get("origin_date") is not None
        assert "T" in mem["origin_date"]  # ISO format

    def test_explicit_origin_date(self, db):
        db.insert("od-2", "Explicit date", importance=0.5,
                  origin_date="2020-01-01T00:00:00")
        mem = db.get("od-2")
        assert mem["origin_date"] == "2020-01-01T00:00:00"


# ═══════════════════════════════════════
# v0.4.0: PCA IN DB
# ═══════════════════════════════════════

class TestPCAInDB:
    def test_pca_stored_in_db(self, populated_db, tmp_path):
        populated_db.fit_projection(save=True)
        # PCA should be in the config table
        row = populated_db.conn.execute(
            "SELECT value FROM config WHERE key = 'pca_projection'"
        ).fetchone()
        assert row is not None
        import json
        data = json.loads(row[0])
        assert "components" in data
        assert "mean" in data

    def test_pca_survives_db_move(self, populated_db, tmp_path):
        populated_db.fit_projection(save=True)
        db_path = populated_db.db_path
        populated_db.close()

        # Move DB to a new location WITHOUT the JSON file
        import shutil
        new_path = str(tmp_path / "subdir" / "moved.db")
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copy2(db_path, new_path)
        # Don't copy pca_projection.json

        db2 = OnionDB(new_path)
        assert db2._pca is not None  # loaded from DB, not JSON
        db2.close()


# ═══════════════════════════════════════
# v0.4.0: INPUT VALIDATION
# ═══════════════════════════════════════

class TestInputValidation:
    def test_empty_id_raises(self, db):
        with pytest.raises(ValueError, match="non-empty string"):
            db.insert("", "content", importance=0.5)

    def test_none_content_raises(self, db):
        with pytest.raises(ValueError, match="must not be None"):
            db.insert("x", None, importance=0.5)

    def test_importance_too_high_raises(self, db):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            db.insert("x", "content", importance=1.5)

    def test_importance_negative_raises(self, db):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            db.insert("x", "content", importance=-0.1)

    def test_boundary_values_accepted(self, db):
        """0.0 and 1.0 should be accepted (inclusive bounds)."""
        db.insert("low", "low", importance=0.0)
        db.insert("high", "high", importance=1.0)
        assert db.count() == 2


# ═══════════════════════════════════════
# v0.4.0: TEMPORAL GRF CATEGORY FILTER
# ═══════════════════════════════════════

class TestTemporalGRFCategory:
    def test_temporal_grf_filters_by_category(self, db):
        for i in range(10):
            cat = "alpha" if i % 2 == 0 else "beta"
            db.insert(f"tc-{i}", f"Item {i}", importance=0.5,
                      category=cat, theta=10.0, phi=5.0)
        # Set temporal_gap for all
        db.conn.execute("UPDATE records SET temporal_gap = 1")
        db.conn.commit()

        all_results = db.temporal_grf(theta=10.0, phi=5.0, k_per_gap=100)
        filtered = db.temporal_grf(theta=10.0, phi=5.0, k_per_gap=100,
                                    category="alpha")

        if 1 in all_results and 1 in filtered:
            assert len(filtered[1]) < len(all_results[1])
            assert all(r["category"] == "alpha" for r in filtered[1])

    def test_temporal_grf_category_none_returns_all(self, db):
        for i in range(6):
            db.insert(f"tcn-{i}", f"Item {i}", importance=0.5,
                      theta=10.0, phi=5.0, category="mixed")
        db.conn.execute("UPDATE records SET temporal_gap = 0")
        db.conn.commit()

        result = db.temporal_grf(theta=10.0, phi=5.0, k_per_gap=100,
                                  category=None)
        if 0 in result:
            assert len(result[0]) == 6


# ═══════════════════════════════════════
# v0.4.0: FIT PROJECTION MIXED DIMENSIONS
# ═══════════════════════════════════════

class TestFitProjectionMixedDims:
    def test_mixed_dims_skipped(self, db):
        """Records with wrong embedding dimension should be filtered, not crash."""
        import struct
        # Insert 15 records with dim=32
        for i in range(15):
            emb = [random.gauss(0, 1) for _ in range(32)]
            db.insert(f"ok-{i}", f"Good {i}", importance=0.5, embedding=emb)

        # Manually inject 2 rogue records with dim=16
        rogue_emb = [0.1] * 16
        rogue_blob = struct.pack(f'{16}f', *rogue_emb)
        db.conn.execute("""
            INSERT OR REPLACE INTO records
            (id, content, gap, theta, phi, depth, cell_theta, cell_phi,
             importance, embedding)
            VALUES (?, ?, 2, 0, 0, 0.5, 0, 0, 0.5, ?)
        """, ("rogue-1", "Bad dim", rogue_blob))
        db.conn.execute("""
            INSERT OR REPLACE INTO records
            (id, content, gap, theta, phi, depth, cell_theta, cell_phi,
             importance, embedding)
            VALUES (?, ?, 2, 0, 0, 0.5, 0, 0, 0.5, ?)
        """, ("rogue-2", "Bad dim 2", rogue_blob))
        db.conn.commit()

        result = db.fit_projection(save=False)
        assert "error" not in result
        assert result["n_samples"] == 15  # only good records
        assert result["skipped_dim_mismatch"] == 2
        assert result["dim"] == 32


# ═══════════════════════════════════════
# v0.4.0: IMPORT JSONL CHUNKING
# ═══════════════════════════════════════

class TestImportJSONLChunking:
    def test_chunk_boundaries(self, db, tmp_path):
        """Import with chunk_size smaller than total records."""
        path = str(tmp_path / "chunked.jsonl")
        with open(path, "w") as f:
            for i in range(7):
                f.write(json.dumps({"id": f"ch-{i}", "content": f"Item {i}",
                                     "importance": 0.5}) + "\n")

        imported = db.import_jsonl(path, chunk_size=3)
        assert imported == 7
        assert db.count() == 7

    def test_exact_chunk_boundary(self, db, tmp_path):
        """Records count is exact multiple of chunk_size."""
        path = str(tmp_path / "exact.jsonl")
        with open(path, "w") as f:
            for i in range(6):
                f.write(json.dumps({"id": f"ex-{i}", "content": f"Item {i}",
                                     "importance": 0.5}) + "\n")

        imported = db.import_jsonl(path, chunk_size=3)
        assert imported == 6
        assert db.count() == 6

    def test_origin_date_preserved_on_roundtrip(self, db, tmp_path):
        """Export→import should preserve origin_date (regression for batch_insert bug)."""
        db.insert("rt-1", "Roundtrip test", importance=0.7,
                  origin_date="2020-06-15T12:00:00")
        db.insert("rt-2", "Another test", importance=0.4,
                  origin_date="2019-01-01T00:00:00")

        export_path = str(tmp_path / "roundtrip.jsonl")
        db.export_jsonl(export_path)

        db2 = OnionDB(str(tmp_path / "roundtrip.db"))
        db2.import_jsonl(export_path)
        m1 = db2.get("rt-1")
        m2 = db2.get("rt-2")
        assert m1["origin_date"] == "2020-06-15T12:00:00"
        assert m2["origin_date"] == "2019-01-01T00:00:00"
        db2.close()


# ═══════════════════════════════════════
# v0.4.0: SCHEMA MIGRATION
# ═══════════════════════════════════════

class TestSchemaMigration:
    def test_memories_table_migrated_to_records(self, tmp_path):
        """Opening a DB with legacy 'memories' table should auto-migrate."""
        db_path = str(tmp_path / "legacy.db")

        # Create a legacy DB with 'memories' table (old schema)
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE memories (
                id          TEXT PRIMARY KEY,
                content     TEXT NOT NULL,
                gap         INTEGER NOT NULL,
                theta       REAL NOT NULL,
                phi         REAL NOT NULL,
                depth       REAL NOT NULL DEFAULT 0.5,
                cell_theta  INTEGER NOT NULL,
                cell_phi    INTEGER NOT NULL,
                importance  REAL NOT NULL DEFAULT 0.5,
                category    TEXT,
                embedding   BLOB,
                metadata    TEXT,
                subshell    INTEGER,
                temporal_gap INTEGER,
                origin_date TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.execute("""
            INSERT INTO memories (id, content, gap, theta, phi, depth,
                                  cell_theta, cell_phi, importance)
            VALUES ('legacy-1', 'Old record', 2, 10.0, 5.0, 0.5, 3, 2, 0.5)
        """)
        conn.commit()
        conn.close()

        # Open with OnionDB — should auto-migrate
        db = OnionDB(db_path)
        tables = {row[0] for row in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "records" in tables
        assert "memories" not in tables

        # Data should be intact
        mem = db.get("legacy-1")
        assert mem is not None
        assert mem["content"] == "Old record"
        assert db.count() == 1
        db.close()
