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
            content=f"Test memory number {i} about topic {i % 5}",
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
        for gap_id, memories in result.items():
            assert isinstance(gap_id, int)
            assert isinstance(memories, list)
            assert len(memories) <= 3

    def test_grf_with_embedding(self, populated_db):
        emb = [random.gauss(0, 1) for _ in range(32)]
        result = populated_db.grf(theta=0, phi=0, k_per_gap=5,
                                   query_embedding=emb)
        for gap_id, memories in result.items():
            scores = [m["score"] for m in memories]
            assert scores == sorted(scores, reverse=True)

    def test_grf_alias_ray(self, populated_db):
        assert populated_db.ray == populated_db.grf


class TestReverseRay:
    def test_reverse_ray_returns_path(self, populated_db):
        emb = [random.gauss(0, 1) for _ in range(32)]
        result = populated_db.reverse_ray(start_embedding=emb)
        assert "path" in result
        assert "curvature" in result
        assert "memories" in result
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
        assert "50 memories" in r
        assert "5 gaps" in r
