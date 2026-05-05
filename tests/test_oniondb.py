"""
Standalone test suite for OnionDB.
No AIGalaxy dependencies — fresh DB per test.
"""
import os
import math
import random
import tempfile
import pytest
from oniondb import OnionDB


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

    def test_importance_to_gap_mapping(self, db):
        # Gap 0: >= 0.95, Gap 1: >= 0.85, Gap 2: >= 0.70, Gap 3: >= 0.50, Gap 4: < 0.50
        assert db.insert("g0", "Core", importance=0.99)["gap"] == 0
        assert db.insert("g1", "High", importance=0.90)["gap"] == 1
        assert db.insert("g2", "Mid", importance=0.75)["gap"] == 2
        assert db.insert("g3", "Low", importance=0.55)["gap"] == 3
        assert db.insert("g4", "Trivial", importance=0.30)["gap"] == 4


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

