"""
OnionDB Quickstart — 20-line working example.

Run: python examples/quickstart.py
"""
from oniondb import OnionDB

# Create a database (creates SQLite file)
db = OnionDB("quickstart.db")

# Insert with importance — determines which shell
db.insert("physics-1", "E=mc² defines mass-energy equivalence", importance=0.99)
db.insert("physics-2", "Gravity bends spacetime", importance=0.95)
db.insert("daily-1", "Had coffee this morning", importance=0.3)
db.insert("work-1", "Finished the quarterly report", importance=0.6)
db.insert("idea-1", "What if databases had geometry?", importance=0.85)

print(f"Database: {db}")
print(f"Total: {db.count()} items")
print(f"Core (gap 0): {db.count(gap=0)} items")
print()

# Shell scan — see everything at one importance level
core = db.shell_scan(gap=0)
print("=== Core memories (gap 0) ===")
for m in core:
    print(f"  [{m['importance']:.2f}] {m['content']}")

# GRF — drill through ALL shells
print("\n=== GRF: depth profile ===")
profile = db.grf(theta=0, phi=0, k_per_gap=2)
for gap_id, memories in profile.items():
    print(f"  Gap {gap_id}:")
    for m in memories:
        print(f"    {m['content'][:50]}")

# Get and delete
print(f"\nGet 'daily-1': {db.get('daily-1')['content']}")
db.delete("daily-1")
print(f"After delete: {db.count()} items")

# Batch insert
items = [
    {"id": f"batch-{i}", "content": f"Batch item {i}", "importance": i / 10}
    for i in range(5)
]
db.batch_insert(items)
print(f"After batch: {db.count()} items")

# Cleanup
db.close()
import os
os.remove("quickstart.db")
print("\nDone! Database cleaned up.")
