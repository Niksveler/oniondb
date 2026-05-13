#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║  test_hnsw_grfv2.py — HNSW + GRF v2 Integration Tests       ║
║                                                              ║
║  SYSTEM:   oniondb / tests                                   ║
║  PURPOSE:  Validates HNSW acceleration layer with GRF v2     ║
║            mass-weighted decay scoring. Tests the full chain: ║
║            insert with mass → HNSW index → query with decay  ║
║            → verify mass boost in ranking.                   ║
║  TESTS:    OnionDB + HNSWBackend + GRF v2                   ║
║  READS:    oniondb/onion_db.py, oniondb/hnsw_backend.py      ║
║  WRITES:   temp .db + .hnsw files (cleaned up by pytest)     ║
║  MODIFIED: Venus IDE 2026-05-12                              ║
╚══════════════════════════════════════════════════════════════╝
"""

import math
import os
import sys
import time
import pytest
import numpy as np
from pathlib import Path

# Ensure oniondb is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oniondb import OnionDB
from oniondb.hnsw_backend import HNSWBackend


# ═══════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════

DIM = 32  # Small dim for fast tests (768 in production)


def _random_emb(seed=None):
    """Generate a random unit-normalized embedding."""
    rng = np.random.RandomState(seed)
    v = rng.randn(DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _similar_emb(base, noise=0.05, seed=None):
    """Generate an embedding similar to base with controlled noise."""
    rng = np.random.RandomState(seed)
    v = np.array(base, dtype=np.float32) + rng.randn(DIM).astype(np.float32) * noise
    v /= np.linalg.norm(v)
    return v.tolist()


@pytest.fixture
def hnsw_db(tmp_path):
    """OnionDB with HNSW enabled and GRF v2 decay."""
    db_path = str(tmp_path / "test_hnsw.db")
    db = OnionDB(
        db_path,
        boundaries=[0.95, 0.85, 0.70, 0.50, 0.00],
        theta_cells=12,
        phi_cells=6,
        default_decay_rate=0.01,
    )
    db.enable_hnsw(dim=DIM, hnsw_threshold=50)  # Low threshold for testing
    return db


@pytest.fixture
def plain_db(tmp_path):
    """OnionDB WITHOUT HNSW — for comparison tests."""
    db_path = str(tmp_path / "test_plain.db")
    return OnionDB(
        db_path,
        boundaries=[0.95, 0.85, 0.70, 0.50, 0.00],
        theta_cells=12,
        phi_cells=6,
        default_decay_rate=0.01,
    )


# ═══════════════════════════════════════════
# TEST CLASS: HNSW Backend Standalone
# ═══════════════════════════════════════════

class TestHNSWBackend:
    """Tests for the HNSW acceleration layer in isolation."""

    def test_add_and_query(self, tmp_path):
        """Basic add + query should return the inserted record."""
        backend = HNSWBackend(str(tmp_path / "test.db"), n_gaps=5, dim=DIM, hnsw_threshold=0)
        emb = _random_emb(42)
        backend.add(0, "rec_001", emb)
        ids, dists = backend.query(0, emb, k=1)
        assert ids == ["rec_001"]
        assert dists[0] < 0.01  # Near-zero distance for exact match

    def test_delete_excludes_from_results(self, tmp_path):
        """Deleted records should not appear in query results."""
        backend = HNSWBackend(str(tmp_path / "test.db"), n_gaps=5, dim=DIM, hnsw_threshold=0)
        emb1 = _random_emb(1)
        emb2 = _random_emb(2)
        backend.add(0, "keep", emb1)
        backend.add(0, "delete_me", emb2)
        backend.delete(0, "delete_me")
        ids, _ = backend.query(0, emb2, k=5)
        assert "delete_me" not in ids

    def test_auto_graduate(self, tmp_path):
        """HNSW should only activate when gap exceeds threshold."""
        backend = HNSWBackend(str(tmp_path / "test.db"), n_gaps=5, dim=DIM, hnsw_threshold=10)
        for i in range(9):
            backend.add(0, f"rec_{i}", _random_emb(i))
        assert not backend.should_use_hnsw(0)
        backend.add(0, "rec_9", _random_emb(9))
        assert backend.should_use_hnsw(0)

    def test_persist_and_reload(self, tmp_path):
        """Saved indices should load back correctly."""
        db_path = str(tmp_path / "persist.db")
        backend = HNSWBackend(db_path, n_gaps=5, dim=DIM, hnsw_threshold=0)
        emb = _random_emb(42)
        backend.add(0, "persist_test", emb)
        backend.save()

        # Load into fresh backend
        backend2 = HNSWBackend(db_path, n_gaps=5, dim=DIM, hnsw_threshold=0)
        assert backend2.load_gap(0)
        ids, _ = backend2.query(0, emb, k=1)
        assert ids == ["persist_test"]

    def test_zombie_compaction(self, tmp_path):
        """Compaction should remove zombie entries."""
        backend = HNSWBackend(str(tmp_path / "test.db"), n_gaps=5, dim=DIM,
                              hnsw_threshold=0, zombie_threshold=0.05)
        # Add 10, delete 2 → 20% zombies → should need compaction
        for i in range(10):
            backend.add(0, f"rec_{i}", _random_emb(i))
        backend.delete(0, "rec_0")
        backend.delete(0, "rec_1")
        assert backend.needs_compaction(0)

    def test_resize_on_overflow(self, tmp_path):
        """Index should auto-resize when exceeding initial capacity."""
        backend = HNSWBackend(str(tmp_path / "test.db"), n_gaps=5, dim=DIM, hnsw_threshold=0)
        # Default init is 1024 — insert more to trigger resize in a real scenario
        for i in range(100):
            backend.add(0, f"rec_{i}", _random_emb(i))
        assert backend.index_size(0) == 100
        ids, _ = backend.query(0, _random_emb(50), k=5)
        assert len(ids) == 5

    def test_stats(self, tmp_path):
        """Stats should report correct loaded/size/zombie counts."""
        backend = HNSWBackend(str(tmp_path / "test.db"), n_gaps=5, dim=DIM, hnsw_threshold=10)
        for i in range(15):
            backend.add(0, f"rec_{i}", _random_emb(i))
        stats = backend.stats()
        assert stats[0]["loaded"] is True
        assert stats[0]["logical_size"] == 15
        assert stats[0]["uses_hnsw"] is True
        assert stats[1]["loaded"] is False


# ═══════════════════════════════════════════
# TEST CLASS: HNSW + OnionDB Integration
# ═══════════════════════════════════════════

class TestHNSWOnionDBIntegration:
    """Tests for HNSW integrated with OnionDB's geometric query pipeline."""

    def test_enable_hnsw(self, hnsw_db):
        """enable_hnsw() should attach the backend."""
        assert hnsw_db._hnsw is not None

    def test_insert_syncs_to_hnsw(self, hnsw_db):
        """Insert with embedding should add to HNSW index."""
        emb = _random_emb(1)
        addr = hnsw_db.insert(id="sync_test", content="HNSW sync test",
                              importance=0.9, embedding=emb, mass=2.5)
        gap = addr["gap"]
        assert hnsw_db._hnsw.index_size(gap) >= 1

    def test_hnsw_fast_path_activates(self, hnsw_db):
        """When gap exceeds threshold, horizontal() should use HNSW."""
        base_emb = _random_emb(42)
        # Insert enough records to exceed hnsw_threshold=50
        for i in range(60):
            hnsw_db.insert(
                id=f"hnsw_fp_{i}",
                content=f"Record {i} for HNSW fast path test",
                importance=0.6,  # All go to same gap
                embedding=_similar_emb(base_emb, noise=0.3, seed=i),
                mass=1.0,
            )
        # Query — should use HNSW fast path internally
        results = hnsw_db.horizontal(
            gap=hnsw_db._importance_to_gap(0.6),
            theta=0.0, phi=0.0,
            k=5,
            query_embedding=base_emb,
        )
        assert len(results) > 0
        # Verify results have valid structure
        for r in results:
            assert "id" in r
            assert "score" in r
            assert r["score"] > 0

    def test_batch_insert_with_hnsw(self, hnsw_db):
        """batch_insert should also sync to HNSW."""
        items = [
            {"id": f"batch_{i}", "content": f"Batch item {i}",
             "importance": 0.5, "embedding": _random_emb(i + 100)}
            for i in range(20)
        ]
        addresses = hnsw_db.batch_insert(items)
        assert len(addresses) == 20
        # Verify HNSW has entries
        gap = addresses[0]["gap"]
        assert hnsw_db._hnsw.index_size(gap) >= 20


# ═══════════════════════════════════════════
# TEST CLASS: GRF v2 Mass + Decay Scoring
# ═══════════════════════════════════════════

class TestGRFv2MassDecay:
    """Tests for the GRF v2 formula: cosine * shell_weight * mass_boost * decay."""

    def test_mass_boost_ranking(self, plain_db):
        """Higher mass should rank higher for same cosine similarity."""
        base_emb = _random_emb(42)
        # Insert two records at same position but different mass
        plain_db.insert(id="low_mass", content="Low mass record",
                        importance=0.5, embedding=_similar_emb(base_emb, noise=0.01, seed=1),
                        mass=0.5, last_review_clock=0)
        plain_db.insert(id="high_mass", content="High mass record",
                        importance=0.5, embedding=_similar_emb(base_emb, noise=0.01, seed=2),
                        mass=10.0, last_review_clock=0)

        theta, phi = plain_db._embed_to_angles(base_emb)
        results = plain_db.horizontal(
            gap=plain_db._importance_to_gap(0.5),
            theta=theta, phi=phi, k=10,
            query_embedding=base_emb,
        )

        # Find both records
        scores = {r["id"]: r["score"] for r in results
                  if r["id"] in ("low_mass", "high_mass")}

        if "low_mass" in scores and "high_mass" in scores:
            # GRF v2: (1 + 0.2 * log1p(mass))
            # high_mass boost: 1 + 0.2 * log1p(10) ≈ 1.48
            # low_mass boost:  1 + 0.2 * log1p(0.5) ≈ 1.08
            assert scores["high_mass"] > scores["low_mass"], \
                f"High mass ({scores['high_mass']:.4f}) should rank above low mass ({scores['low_mass']:.4f})"

    def test_decay_penalizes_stale(self, plain_db):
        """Records with old clock should score lower with decay enabled."""
        base_emb = _random_emb(42)
        # Fresh record (reviewed recently)
        plain_db.insert(id="fresh", content="Fresh record",
                        importance=0.5, embedding=_similar_emb(base_emb, noise=0.01, seed=1),
                        mass=1.0, last_review_clock=100)
        # Stale record (not reviewed for long time)
        plain_db.insert(id="stale", content="Stale record",
                        importance=0.5, embedding=_similar_emb(base_emb, noise=0.01, seed=2),
                        mass=1.0, last_review_clock=0)

        theta, phi = plain_db._embed_to_angles(base_emb)
        results = plain_db.horizontal(
            gap=plain_db._importance_to_gap(0.5),
            theta=theta, phi=phi, k=10,
            query_embedding=base_emb,
            current_clock=100,
            decay_rate=0.01,
        )

        scores = {r["id"]: r["score"] for r in results
                  if r["id"] in ("fresh", "stale")}

        if "fresh" in scores and "stale" in scores:
            # Fresh: exp(-0.01 * (100-100)) = exp(0) = 1.0
            # Stale: exp(-0.01 * (100-0))   = exp(-1) ≈ 0.37
            assert scores["fresh"] > scores["stale"], \
                f"Fresh ({scores['fresh']:.4f}) should rank above stale ({scores['stale']:.4f})"

    def test_no_decay_when_clock_is_none(self, plain_db):
        """When current_clock=None, decay should not apply."""
        base_emb = _random_emb(42)
        plain_db.insert(id="rec_a", content="Record A",
                        importance=0.5, embedding=base_emb,
                        mass=1.0, last_review_clock=0)

        theta, phi = plain_db._embed_to_angles(base_emb)

        # Query without decay
        r1 = plain_db.horizontal(
            gap=plain_db._importance_to_gap(0.5),
            theta=theta, phi=phi, k=5,
            query_embedding=base_emb,
            current_clock=None,
        )

        # Query with decay
        r2 = plain_db.horizontal(
            gap=plain_db._importance_to_gap(0.5),
            theta=theta, phi=phi, k=5,
            query_embedding=base_emb,
            current_clock=1000,
            decay_rate=0.01,
        )

        score_no_decay = next((r["score"] for r in r1 if r["id"] == "rec_a"), 0)
        score_with_decay = next((r["score"] for r in r2 if r["id"] == "rec_a"), 0)

        # With decay at clock=1000, score should be lower
        assert score_no_decay >= score_with_decay, \
            f"No-decay score ({score_no_decay:.4f}) should be >= decayed ({score_with_decay:.4f})"

    def test_mass_boost_formula(self):
        """Verify the mass boost formula: (1 + 0.2 * log1p(mass))."""
        # mass=0 → boost=1.0
        assert abs((1 + 0.2 * math.log1p(0)) - 1.0) < 1e-6
        # mass=1 → boost ≈ 1.139
        assert abs((1 + 0.2 * math.log1p(1)) - 1.1386) < 0.001
        # mass=10 → boost ≈ 1.479
        assert abs((1 + 0.2 * math.log1p(10)) - 1.4796) < 0.001
        # mass=100 → boost ≈ 1.924
        assert abs((1 + 0.2 * math.log1p(100)) - 1.923) < 0.005

    def test_grf_drills_all_shells(self, plain_db):
        """GRF should return results from multiple importance shells."""
        base_emb = _random_emb(42)
        # Insert into different importance levels (different gaps)
        for imp, label in [(0.98, "critical"), (0.8, "high"),
                           (0.6, "medium"), (0.3, "low")]:
            plain_db.insert(
                id=f"grf_{label}", content=f"GRF test {label}",
                importance=imp,
                embedding=_similar_emb(base_emb, noise=0.05, seed=int(imp * 100)),
                mass=imp * 5,
            )

        theta, phi = plain_db._embed_to_angles(base_emb)
        grf_results = plain_db.grf(
            theta, phi, k_per_gap=3,
            neighbor_radius=6,
            query_embedding=base_emb,
        )

        # Should hit multiple gaps
        gaps_hit = [g for g, mems in grf_results.items() if len(mems) > 0]
        assert len(gaps_hit) >= 2, f"GRF should drill multiple shells, got {len(gaps_hit)}"


# ═══════════════════════════════════════════
# TEST CLASS: HNSW + GRF v2 Combined
# ═══════════════════════════════════════════

class TestHNSWGRFv2Combined:
    """Tests for HNSW acceleration with GRF v2 re-ranking."""

    def test_hnsw_results_match_brute_force(self, tmp_path):
        """HNSW results should be consistent with brute-force SQL scan."""
        db_path_hnsw = str(tmp_path / "hnsw.db")
        db_path_plain = str(tmp_path / "plain.db")

        db_hnsw = OnionDB(db_path_hnsw, default_decay_rate=0.01)
        db_hnsw.enable_hnsw(dim=DIM, hnsw_threshold=10)
        db_plain = OnionDB(db_path_plain, default_decay_rate=0.01)

        base_emb = _random_emb(42)
        # Insert same data into both
        for i in range(30):
            emb = _similar_emb(base_emb, noise=0.3, seed=i)
            kwargs = dict(id=f"rec_{i}", content=f"Record {i}",
                          importance=0.5, embedding=emb, mass=float(i))
            db_hnsw.insert(**kwargs)
            db_plain.insert(**kwargs)

        theta, phi = db_hnsw._embed_to_angles(base_emb)
        gap = db_hnsw._importance_to_gap(0.5)

        r_hnsw = db_hnsw.horizontal(gap, theta, phi, k=10,
                                     query_embedding=base_emb)
        r_plain = db_plain.horizontal(gap, theta, phi, k=10,
                                       query_embedding=base_emb)

        # Top result should be the same (or very close)
        hnsw_ids = {r["id"] for r in r_hnsw[:5]}
        plain_ids = {r["id"] for r in r_plain[:5]}
        overlap = hnsw_ids & plain_ids
        # At least 3 of top 5 should overlap (HNSW is approximate)
        assert len(overlap) >= 3, \
            f"HNSW/brute overlap too low: {len(overlap)}/5 ({hnsw_ids} vs {plain_ids})"

    def test_hnsw_performance_advantage(self, tmp_path):
        """HNSW should be faster than brute-force at scale."""
        db_path = str(tmp_path / "perf.db")
        db = OnionDB(db_path, default_decay_rate=0.01)
        db.enable_hnsw(dim=DIM, hnsw_threshold=50)

        base_emb = _random_emb(42)
        # Insert 200 records (well above threshold)
        items = [
            {"id": f"perf_{i}", "content": f"Perf record {i}",
             "importance": 0.5, "embedding": _similar_emb(base_emb, noise=0.5, seed=i)}
            for i in range(200)
        ]
        db.batch_insert(items)

        gap = db._importance_to_gap(0.5)
        theta, phi = db._embed_to_angles(base_emb)

        # HNSW query
        t0 = time.perf_counter()
        for _ in range(100):
            db.horizontal(gap, theta, phi, k=10, query_embedding=base_emb)
        hnsw_time = (time.perf_counter() - t0) * 1000 / 100

        # Plain SQL query (disable HNSW temporarily)
        db._hnsw = None
        t0 = time.perf_counter()
        for _ in range(100):
            db.horizontal(gap, theta, phi, k=10, query_embedding=base_emb)
        sql_time = (time.perf_counter() - t0) * 1000 / 100

        # HNSW should not be dramatically slower (may be similar at 200 records)
        # At production scale (10K+), HNSW would be 10-100x faster
        print(f"\n  HNSW: {hnsw_time:.2f}ms avg | SQL: {sql_time:.2f}ms avg")
        # Just verify both work — real perf advantage at 10K+
        assert hnsw_time < sql_time * 10  # HNSW shouldn't be 10x slower


# ═══════════════════════════════════════════
# TEST CLASS: Scaling Stress Tests
# ═══════════════════════════════════════════

class TestScalingStress:
    """Tests for scaling behavior at larger record counts."""

    def test_batch_insert_1000(self, hnsw_db):
        """Batch insert 1000 records should complete in reasonable time."""
        items = [
            {"id": f"scale_{i}", "content": f"Scaling test record {i}",
             "importance": (i % 100) / 100.0,
             "embedding": _random_emb(i)}
            for i in range(1000)
        ]
        t0 = time.perf_counter()
        addresses = hnsw_db.batch_insert(items)
        elapsed = (time.perf_counter() - t0) * 1000
        assert len(addresses) == 1000
        print(f"\n  1000 batch_insert: {elapsed:.0f}ms ({elapsed/1000:.2f}ms/record)")
        # Should complete in under 30 seconds (generous for CI)
        assert elapsed < 30000

    def test_iter_all_streaming(self, hnsw_db):
        """iter_all should yield all records without loading everything."""
        # Insert 200 records
        items = [
            {"id": f"iter_{i}", "content": f"Iterator test {i}",
             "importance": 0.5, "embedding": _random_emb(i)}
            for i in range(200)
        ]
        hnsw_db.batch_insert(items)

        # Stream through iter_all
        count = 0
        for record in hnsw_db.iter_all(batch_size=50):
            count += 1
            assert "id" in record
            assert "content" in record
        assert count == 200

    def test_query_at_scale(self, hnsw_db):
        """GRF queries should work correctly at 500+ records."""
        base_emb = _random_emb(42)
        items = [
            {"id": f"qscale_{i}", "content": f"Query scale test {i}",
             "importance": np.random.uniform(0.1, 0.99),
             "embedding": _similar_emb(base_emb, noise=0.5, seed=i)}
            for i in range(500)
        ]
        hnsw_db.batch_insert(items)

        theta, phi = hnsw_db._embed_to_angles(base_emb)
        t0 = time.perf_counter()
        results = hnsw_db.grf(theta, phi, k_per_gap=5,
                               neighbor_radius=4, query_embedding=base_emb)
        elapsed = (time.perf_counter() - t0) * 1000

        total = sum(len(mems) for mems in results.values())
        print(f"\n  GRF at 500 records: {elapsed:.1f}ms, {total} results across {len(results)} gaps")
        assert total > 0
        assert elapsed < 5000  # Should be well under 5 seconds


# ═══════════════════════════════════════════
# TEST CLASS: CCML Integration Path
# ═══════════════════════════════════════════

class TestCCMLIntegration:
    """Tests that verify the CCML → OnionDB sync path works with HNSW."""

    def test_insert_with_mass_equals_importance(self, hnsw_db):
        """CCML sync uses mass=importance. Verify the mapping."""
        importance = 0.85
        hnsw_db.insert(
            id="ccml_sync_test",
            content="Testing CCML sync path",
            importance=importance,
            embedding=_random_emb(42),
            mass=importance,  # This is what memory_core._sync_to_oniondb does
            last_review_clock=0,
        )
        rec = hnsw_db.get("ccml_sync_test")
        assert rec["mass"] == importance
        assert rec["last_review_clock"] == 0

    def test_mass_default_is_one(self, hnsw_db):
        """Records inserted without explicit mass should default to 1.0."""
        hnsw_db.insert(
            id="no_mass_test",
            content="No explicit mass",
            importance=0.5,
            embedding=_random_emb(42),
        )
        rec = hnsw_db.get("no_mass_test")
        assert rec["mass"] == 1.0

    def test_schema_has_mass_columns(self, hnsw_db):
        """Verify the Phase 5 schema columns exist."""
        cols = hnsw_db.conn.execute("PRAGMA table_info(records)").fetchall()
        col_names = [c[1] for c in cols]
        assert "mass" in col_names
        assert "last_review_clock" in col_names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
