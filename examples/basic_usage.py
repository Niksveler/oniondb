"""
OnionDB — Basic Usage Example

Demonstrates core operations: insert, query, fit, calibrate boundaries.
Uses random embeddings (replace with real embeddings from your model).
"""
import random
from oniondb import OnionDB

# ═══════════════════════════════════════
# 1. CREATE & INSERT
# ═══════════════════════════════════════

db = OnionDB("example.db")

# Simulate embeddings (use real ones from sentence-transformers, OpenAI, etc.)
random.seed(42)
def fake_embedding(dim=128):
    return [random.gauss(0, 1) for _ in range(dim)]

# Insert records with different importance levels
records = [
    ("core-1",    "E=mc² defines mass-energy equivalence",           0.99),
    ("core-2",    "The speed of light is 299,792,458 m/s",           0.95),
    ("important", "Quantum entanglement allows instant correlation",  0.80),
    ("mid-1",     "The ISS orbits at 408 km altitude",               0.60),
    ("mid-2",     "Neptune has 16 known moons",                      0.55),
    ("low-1",     "I had a good lunch today",                        0.20),
    ("low-2",     "The weather is nice",                             0.15),
    ("trivial",   "Random shower thought about socks",               0.05),
]

for rid, content, importance in records:
    addr = db.insert(rid, content, importance=importance, embedding=fake_embedding())
    print(f"  Inserted '{rid}' → gap={addr['gap']}, depth={addr['depth']:.3f}")

print(f"\nTotal records: {db.count()}")

# ═══════════════════════════════════════
# 2. QUERY OPERATIONS
# ═══════════════════════════════════════

# Shell scan: everything at importance level 0 (core)
print("\n--- Shell Scan (gap=0, core records) ---")
core = db.shell_scan(gap=0)
for r in core:
    print(f"  [{r['id']}] imp={r['importance']:.2f}: {r['content'][:60]}")

# GRF: drill through ALL shells at a direction
print("\n--- GRF (drill through all shells) ---")
query_emb = fake_embedding()
profile = db.grf(theta=0, phi=0, query_embedding=query_emb, k_per_gap=3)
for gap_id, results in profile.items():
    if results:
        print(f"  Gap {gap_id}: {len(results)} records")
        for r in results[:2]:
            print(f"    [{r['id']}] sim={r.get('similarity', 0):.3f}")

# Reverse ray: trace semantic gravity inward
print("\n--- Reverse Ray ---")
trace = db.reverse_ray(query_emb)
print(f"  Curvature: {trace['curvature']:.1f}°")
print(f"  Straight path: {trace['straight']}")
print(f"  Hops: {trace['n_hops']}")

# Beam search reverse ray
print("\n--- Beam Search Reverse Ray (width=3) ---")
beam_trace = db.reverse_ray(query_emb, beam_width=3)
print(f"  Curvature: {beam_trace['curvature']:.1f}°")
print(f"  Paths explored: {beam_trace['beam_paths_explored']}")

# ═══════════════════════════════════════
# 3. BOUNDARY CALIBRATION
# ═══════════════════════════════════════

print("\n--- Boundary Calibration ---")
print(f"  Current boundaries: {db.boundaries}")

# Suggest optimal boundaries from data distribution
suggested = db.fit_boundaries(n_gaps=4)
print(f"  Suggested (4 gaps): {suggested}")

# Apply new boundaries
result = db.reindex(boundaries=suggested)
print(f"  Reindexed: {result['updated']} records across {result['n_gaps']} gaps")

# ═══════════════════════════════════════
# 4. STATS & INFO
# ═══════════════════════════════════════

print("\n--- Database Stats ---")
stats = db.stats()
print(f"  Total: {stats['total']} records")
print(f"  Gaps: {stats['n_gaps']}")
print(f"  Grid: {stats['grid']}")
for gap_id, info in stats["gaps"].items():
    print(f"  Gap {gap_id}: {info['count']} records "
          f"(imp range: {info['min_importance']:.2f}–{info['max_importance']:.2f})")

db.close()

# Clean up example file
import os
os.remove("example.db")
print("\nDone! (example.db cleaned up)")
