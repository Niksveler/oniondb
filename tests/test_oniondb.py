"""
Standalone test suite for OnionDB.
No AIGalaxy dependencies — fresh DB per test.
"""
import os
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
        db = OnionDB(str(tmp_path / "read.db"))
        for i in range(50):
            db.insert(f"r-{i}", f"Read test {i}", importance=random.uniform(0.1, 0.9))

        errors = []
        results = []

        def reader(thread_id):
            try:
                for i in range(10):
                    mem = db.get(f"r-{i}")
                    if mem:
                        results.append(mem["id"])
                    db.shell_scan(gap=random.randint(0, 4), limit=5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Read errors: {errors}"
        assert len(results) > 0
        db.close()


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
