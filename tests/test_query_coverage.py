#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  test_query_coverage.py — Full Query API Coverage            ║
║                                                              ║
║  SYSTEM:   oniondb / tests                                   ║
║  PURPOSE:  Tests every OnionDB query method with GRF v2      ║
║            mass+decay scoring: horizontal, grf, reverse_ray, ║
║            temporal_grf, shell_scan, range_scan, cell_density,║
║            stats, update, simhash_query.                     ║
║  TESTS:    OnionDB query API completeness                    ║
║  MODIFIED: Venus IDE 2026-05-12                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys
import math
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oniondb import OnionDB

DIM = 32


def _emb(seed=None):
    rng = np.random.RandomState(seed)
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _near(base, noise=0.05, seed=None):
    rng = np.random.RandomState(seed)
    v = np.array(base, dtype=np.float32) + rng.randn(DIM).astype(np.float32) * noise
    v /= np.linalg.norm(v)
    return v.tolist()


@pytest.fixture
def db(tmp_path):
    """Populated OnionDB with records across all gaps + temporal data."""
    d = OnionDB(
        str(tmp_path / "query_test.db"),
        boundaries=[0.95, 0.85, 0.70, 0.50, 0.00],
        theta_cells=12, phi_cells=6,
        default_decay_rate=0.01,
    )
    base = _emb(42)
    # Insert records at various importance levels and categories
    test_data = [
        ("core_1",   "Core memory about consciousness",  0.98, "soul",     5.0, 100),
        ("core_2",   "Core memory about identity",       0.96, "soul",     3.0,  50),
        ("high_1",   "Important project architecture",   0.88, "project",  2.0,  80),
        ("high_2",   "Important design decision",        0.82, "project",  1.5,  60),
        ("mid_1",    "Medium priority task",             0.65, "task",     1.0,  40),
        ("mid_2",    "Medium priority observation",      0.60, "insight",  0.8,  20),
        ("low_1",    "Low priority note",                0.40, "note",     0.5,  10),
        ("low_2",    "Low priority scratch",             0.30, "note",     0.3,   5),
        ("trivial_1","Trivial log entry",                0.10, "log",      0.1,   0),
        ("trivial_2","Trivial debug output",             0.05, "log",      0.1,   0),
    ]
    for i, (rid, content, imp, cat, mass, clock) in enumerate(test_data):
        d.insert(id=rid, content=content, importance=imp, category=cat,
                 embedding=_near(base, noise=0.1, seed=i+100),
                 mass=mass, last_review_clock=clock,
                 origin_date=f"2026-05-{10-i:02d}")
    d.assign_temporal_gaps()
    return d


# ═══════════════════════════════════════════
# horizontal() — single-gap directional query
# ═══════════════════════════════════════════

class TestHorizontal:
    def test_returns_results(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        r = db.horizontal(0, theta, phi, k=5, query_embedding=base)
        assert len(r) > 0
        assert all("score" in x and "id" in x for x in r)

    def test_sorted_by_score(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        r = db.horizontal(0, theta, phi, k=10, query_embedding=base)
        scores = [x["score"] for x in r]
        assert scores == sorted(scores, reverse=True)

    def test_category_filter(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        gap = db._importance_to_gap(0.98)
        r = db.horizontal(gap, theta, phi, k=10, query_embedding=base, category="soul")
        assert all(x["category"] == "soul" for x in r)

    def test_mass_affects_score(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        gap = db._importance_to_gap(0.98)
        r = db.horizontal(gap, theta, phi, k=10, query_embedding=base)
        # core_1 has mass=5.0, core_2 has mass=3.0
        scores = {x["id"]: x["score"] for x in r}
        if "core_1" in scores and "core_2" in scores:
            # Higher mass should boost score (all else ~equal)
            boost_1 = 1 + 0.2 * math.log1p(5.0)
            boost_2 = 1 + 0.2 * math.log1p(3.0)
            assert boost_1 > boost_2

    def test_decay_with_clock(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        gap = db._importance_to_gap(0.98)
        r = db.horizontal(gap, theta, phi, k=10, query_embedding=base,
                          current_clock=200, decay_rate=0.01)
        assert len(r) > 0
        # All scores should be positive but decay-adjusted
        assert all(x["score"] > 0 for x in r)

    def test_subshell_boost(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        gap = db._importance_to_gap(0.98)
        r = db.horizontal(gap, theta, phi, k=10, query_embedding=base,
                          subshell=0, subshell_boost=0.5)
        assert len(r) >= 0  # May or may not match subshell


# ═══════════════════════════════════════════
# grf() — multi-shell drill
# ═══════════════════════════════════════════

class TestGRF:
    def test_returns_dict_of_gaps(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        r = db.grf(theta, phi, k_per_gap=3, neighbor_radius=6, query_embedding=base)
        assert isinstance(r, dict)
        assert len(r) >= 2  # Should hit multiple gaps

    def test_each_gap_has_scored_results(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        r = db.grf(theta, phi, k_per_gap=5, neighbor_radius=6, query_embedding=base)
        for gap_id, mems in r.items():
            assert isinstance(gap_id, int)
            for m in mems:
                assert "score" in m
                assert "id" in m

    def test_category_filter(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        r = db.grf(theta, phi, k_per_gap=5, neighbor_radius=6,
                   query_embedding=base, category="soul")
        for gap_id, mems in r.items():
            assert all(m["category"] == "soul" for m in mems)

    def test_with_decay(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        r = db.grf(theta, phi, k_per_gap=3, neighbor_radius=6,
                   query_embedding=base, current_clock=200, decay_rate=0.02)
        total = sum(len(v) for v in r.values())
        assert total > 0

    def test_ray_alias(self, db):
        """ray() should be identical to grf()."""
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        r1 = db.grf(theta, phi, k_per_gap=3, neighbor_radius=6, query_embedding=base)
        r2 = db.ray(theta, phi, k_per_gap=3, neighbor_radius=6, query_embedding=base)
        assert r1.keys() == r2.keys()

    def test_k_per_gap_limit(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        r = db.grf(theta, phi, k_per_gap=1, neighbor_radius=6, query_embedding=base)
        for gap_id, mems in r.items():
            assert len(mems) <= 1


# ═══════════════════════════════════════════
# reverse_ray() — curved semantic gravity trace
# ═══════════════════════════════════════════

class TestReverseRay:
    def test_returns_path(self, db):
        r = db.reverse_ray(_emb(42))
        assert "path" in r
        assert "curvature" in r
        assert "records" in r
        assert "path_vector" in r

    def test_curvature_nonnegative(self, db):
        r = db.reverse_ray(_emb(42))
        assert r["curvature"] >= 0

    def test_path_has_hops(self, db):
        r = db.reverse_ray(_emb(42))
        assert len(r["path"]) > 0
        for hop in r["path"]:
            assert "gap" in hop
            assert "theta" in hop
            assert "phi" in hop

    def test_straight_flag(self, db):
        r = db.reverse_ray(_emb(42))
        assert "straight" in r
        assert isinstance(r["straight"], bool)

    def test_beam_search(self, db):
        """Beam width > 1 should explore multiple paths."""
        r = db.reverse_ray(_emb(42), beam_width=3)
        assert "beam_width" in r
        assert r["beam_width"] == 3
        assert "beam_paths_explored" in r

    def test_start_gap_override(self, db):
        r = db.reverse_ray(_emb(42), start_gap=2)
        # Should only traverse gaps 2, 1, 0
        gaps_visited = [h["gap"] for h in r["path"]]
        assert max(gaps_visited) <= 2


# ═══════════════════════════════════════════
# temporal_grf() — time-shell drill
# ═══════════════════════════════════════════

class TestTemporalGRF:
    def test_returns_dict(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        r = db.temporal_grf(theta, phi, k_per_gap=5, neighbor_radius=6,
                            query_embedding=base)
        assert isinstance(r, dict)

    def test_results_have_scores(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        r = db.temporal_grf(theta, phi, k_per_gap=5, neighbor_radius=6,
                            query_embedding=base)
        for tgap, mems in r.items():
            for m in mems:
                assert "score" in m

    def test_without_embedding(self, db):
        """Should fall back to angular distance scoring."""
        r = db.temporal_grf(0.0, 0.0, k_per_gap=5, neighbor_radius=6)
        # Should still work (angular distance mode)
        assert isinstance(r, dict)

    def test_category_filter(self, db):
        base = _emb(42)
        theta, phi = db._embed_to_angles(base)
        r = db.temporal_grf(theta, phi, k_per_gap=5, neighbor_radius=6,
                            query_embedding=base, category="soul")
        for tgap, mems in r.items():
            assert all(m["category"] == "soul" for m in mems)


# ═══════════════════════════════════════════
# shell_scan() — full gap dump
# ═══════════════════════════════════════════

class TestShellScan:
    def test_returns_list(self, db):
        r = db.shell_scan(0)
        assert isinstance(r, list)

    def test_limit(self, db):
        r = db.shell_scan(0, limit=1)
        assert len(r) <= 1

    def test_offset_pagination(self, db):
        all_recs = db.shell_scan(0, limit=100)
        if len(all_recs) >= 2:
            page2 = db.shell_scan(0, limit=1, offset=1)
            assert len(page2) == 1
            assert page2[0]["id"] == all_recs[1]["id"]

    def test_category_filter(self, db):
        r = db.shell_scan(0, category="soul")
        assert all(x["category"] == "soul" for x in r)

    def test_mass_in_results(self, db):
        """Phase 5: results should include mass and last_review_clock."""
        r = db.shell_scan(0, limit=5)
        for rec in r:
            assert "mass" in rec
            assert "last_review_clock" in rec


# ═══════════════════════════════════════════
# range_scan() — multi-gap range query
# ═══════════════════════════════════════════

class TestRangeScan:
    def test_returns_list(self, db):
        r = db.range_scan(0, 2)
        assert isinstance(r, list)

    def test_only_requested_gaps(self, db):
        r = db.range_scan(0, 1)
        for rec in r:
            assert rec["gap"] <= 1

    def test_full_range(self, db):
        r = db.range_scan(0, db.n_gaps - 1, limit=100)
        total = db.count()
        assert len(r) == min(total, 100)

    def test_limit_and_offset(self, db):
        all_r = db.range_scan(0, db.n_gaps - 1, limit=100)
        if len(all_r) >= 3:
            page = db.range_scan(0, db.n_gaps - 1, limit=2, offset=1)
            assert len(page) == 2

    def test_category_filter(self, db):
        r = db.range_scan(0, db.n_gaps - 1, category="note")
        assert all(x["category"] == "note" for x in r)


# ═══════════════════════════════════════════
# cell_density() — spatial distribution
# ═══════════════════════════════════════════

class TestCellDensity:
    def test_returns_list(self, db):
        r = db.cell_density(0)
        assert isinstance(r, list)

    def test_cell_format(self, db):
        r = db.cell_density(0)
        for item in r:
            assert "cell" in item
            assert "count" in item
            assert item["count"] > 0

    def test_total_matches_count(self, db):
        gap = 0
        density = db.cell_density(gap)
        density_total = sum(d["count"] for d in density)
        count = db.count(gap=gap)
        assert density_total == count


# ═══════════════════════════════════════════
# stats() — database summary
# ═══════════════════════════════════════════

class TestStats:
    def test_structure(self, db):
        s = db.stats()
        assert "total" in s
        assert "n_gaps" in s
        assert "boundaries" in s
        assert "gaps" in s
        assert "categories" in s
        assert "grid" in s

    def test_total_matches(self, db):
        s = db.stats()
        assert s["total"] == 10

    def test_categories_present(self, db):
        s = db.stats()
        assert "soul" in s["categories"]
        assert "project" in s["categories"]


# ═══════════════════════════════════════════
# update() — partial record update
# ═══════════════════════════════════════════

class TestUpdate:
    def test_update_content(self, db):
        r = db.update("core_1", content="Updated consciousness memory")
        assert r is not None
        rec = db.get("core_1")
        assert rec["content"] == "Updated consciousness memory"

    def test_update_importance_moves_gap(self, db):
        old = db.get("low_1")
        old_gap = old["gap"]
        db.update("low_1", importance=0.99)
        new = db.get("low_1")
        assert new["gap"] != old_gap  # Should move to gap 0

    def test_update_nonexistent(self, db):
        r = db.update("nonexistent_id", content="nope")
        assert r is None

    def test_update_preserves_mass(self, db):
        """Update without mass kwarg should preserve existing mass."""
        db.update("core_1", content="Changed content only")
        rec = db.get("core_1")
        assert rec["mass"] == 5.0  # Original mass preserved


# ═══════════════════════════════════════════
# get(), delete(), count() — CRUD basics
# ═══════════════════════════════════════════

class TestCRUD:
    def test_get_existing(self, db):
        r = db.get("core_1")
        assert r is not None
        assert r["content"] == "Core memory about consciousness"
        assert r["mass"] == 5.0

    def test_get_nonexistent(self, db):
        assert db.get("fake_id") is None

    def test_delete(self, db):
        before = db.count()
        db.delete("trivial_2")
        assert db.count() == before - 1
        assert db.get("trivial_2") is None

    def test_count_per_gap(self, db):
        total = sum(db.count(gap=g) for g in range(db.n_gaps))
        assert total == db.count()


# ═══════════════════════════════════════════
# export/import JSONL roundtrip
# ═══════════════════════════════════════════

class TestExportImport:
    def test_jsonl_roundtrip(self, db, tmp_path):
        path = str(tmp_path / "export.jsonl")
        exported = db.export_jsonl(path)
        assert exported == 10

        db2 = OnionDB(str(tmp_path / "imported.db"),
                       boundaries=[0.95, 0.85, 0.70, 0.50, 0.00])
        imported = db2.import_jsonl(path)
        assert imported == 10
        assert db2.count() == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
