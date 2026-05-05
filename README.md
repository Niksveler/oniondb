# 🧅 OnionDB

**A geometric memory database. Zero dependencies. Importance-stratified.**

Your data has a *location*, not just a vector.

---

OnionDB organizes data in concentric shells -- like layers of an onion. Every record has a 4-part geometric address `(gap, theta, phi, depth)` based on its importance and semantic content. This enables queries that flat vector databases can't do:

- **"Show me everything at importance level 3"** --> shell scan
- **"Drill through ALL importance levels at this semantic direction"** --> GRF (Geometric Ray Filter)
- **"Trace how this topic connects across depth levels"** --> reverse ray

```
          +-----------------+
         /   gap 4 (trivial) \
        /  +-------------+    \
       /  /  gap 3 (low)  \    \
      /  /  +-----------+  \    \
     /  /  /  gap 2 (mid)\  \    \
    /  /  /  +--------+   \  \    \
   |  |  |  | gap 1  |    |  |    |
   |  |  |  | +----+ |    |  |    |
   |  |  |  | | g0 | |    |  |    |   <-- GRF drills through ALL layers
   |  |  |  | |core| |    |  |    |       at angle (theta, phi)
   |  |  |  | +----+ |    |  |    |
   |  |  |  +--------+    |  |    |
    \  \  \______________/  /    /
     \  \__________________/    /
      \________________________/
```

## Install

```bash
pip install oniondb
```

## Quick Start

```python
from oniondb import OnionDB

# Create a database (SQLite file, zero config)
db = OnionDB("my_memories.db")

# Insert with importance (determines which shell)
db.insert("idea-1", "The Earth orbits the Sun", importance=0.9)
db.insert("idea-2", "I had coffee this morning", importance=0.3)
db.insert("idea-3", "E=mc2 defines mass-energy equivalence", importance=0.99)

# Shell scan -- everything at importance level 0 (core memories)
core = db.shell_scan(gap=0)

# GRF -- drill through ALL shells at a direction
# (requires embeddings for semantic direction)
profile = db.grf(theta=45.0, phi=10.0, query_embedding=my_embedding)

# Reverse ray -- follow semantic gravity inward
trace = db.reverse_ray(start_embedding=my_embedding)
print(f"Path curvature: {trace['curvature']} degrees")  # 0=straight, high=fragmented

# Count, get, delete
print(db.count())        # 3
print(db.get("idea-1"))  # full memory dict
db.delete("idea-2")      # True
```

## Features

| Feature | Description |
|---------|------------|
| **Zero dependencies** | stdlib only -- `sqlite3`, `math`, `struct`, `json`, `os` |
| **Geometric addressing** | Every record has a location: `(gap, theta, phi, depth)` |
| **Importance shells** | Data stratified by significance -- core vs trivial |
| **6 query operations** | horizontal, GRF, reverse_ray, temporal_grf, shell_scan, range_scan |
| **Embedding-agnostic** | Works with any embedding model (OpenAI, Ollama, sentence-transformers...) |
| **Single-file storage** | SQLite-backed, portable, copy-paste deployable |
| **Self-calibrating** | `fit_projection()` builds PCA from your data automatically |
| **Thread-safe** | RLock + WAL mode for concurrent access |

## The Signature Query: GRF (Geometric Ray Filter)

The GRF is what makes OnionDB unique. It "drills a core sample" through every importance shell at a given semantic direction, returning a **depth profile** of how a topic exists at every level of significance.

```python
# With embeddings: semantic direction from the embedding
profile = db.grf(theta=0, phi=0, query_embedding=embedding, k_per_gap=5)
# Returns: {0: [core memories], 1: [important], 2: [mid], 3: [low], 4: [trivial]}

# The reverse ray follows semantic gravity inward, bending as it goes
trace = db.reverse_ray(start_embedding=embedding)
# trace["curvature"] -- total angular deviation
# trace["straight"]  -- True if topic is well-organized across all depths
# trace["path"]      -- list of hops from outer to inner shells
```

## Using with Embeddings

OnionDB works with or without embeddings. Without them, queries use angular distance. With them, queries use cosine similarity for precise semantic ranking.

```python
# Any embedding model works -- just pass a list of floats
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

embedding = model.encode("quantum physics").tolist()
db.insert("q1", "Quantum entanglement is spooky", importance=0.8, embedding=embedding)

# After inserting enough data, calibrate the projection
stats = db.fit_projection()
print(f"Cell occupancy: {stats['occupancy_after']:.0%}")  # target: >80%
```

## API Reference

### Core Operations

| Method | Description |
|--------|-------------|
| `insert(id, content, importance, ...)` | Insert a record with auto-computed geometric address |
| `get(id)` | Retrieve a record by ID |
| `delete(id)` | Delete a record by ID |
| `count(gap=None)` | Count records (optionally per gap) |
| `batch_insert(items)` | Insert multiple records in a single transaction |

### Query Operations

| Method | Description |
|--------|-------------|
| `horizontal(gap, theta, phi, ...)` | Find nearby items within one shell |
| `grf(theta, phi, ...)` | **Geometric Ray Filter** -- drill through all shells |
| `reverse_ray(start_embedding, ...)` | Curved semantic trace from outer to inner |
| `temporal_grf(theta, phi, ...)` | Drill through time-based shells |
| `shell_scan(gap, limit)` | Return everything at one importance level |
| `range_scan(gap_start, gap_end, limit)` | Return everything between two levels |

### Configuration

| Method | Description |
|--------|-------------|
| `fit_projection(save=True)` | Self-calibrate PCA from stored embeddings |
| `stats()` | Database statistics (gaps, categories, grid) |
| `cell_density(gap)` | Cell occupancy map for a gap |

### Custom Boundaries

```python
# Default: 5 shells at [0.95, 0.85, 0.70, 0.50, 0.00]
db = OnionDB("custom.db", boundaries=[0.90, 0.70, 0.40, 0.00])  # 4 shells
```

## How It Works

1. **Importance to Gap**: Each record's importance score determines which shell (gap) it lives in. Gap 0 is the innermost core (most important).

2. **Embedding to Angles**: If an embedding is provided, PCA projects it onto spherical coordinates (theta, phi). This gives semantically similar items nearby angular positions.

3. **Address**: Every record gets a 4-part address: `(gap, theta, phi, depth)` where depth is the position within the gap based on exact importance.

4. **Cells**: The sphere is divided into a 12x6 grid. Queries search the target cell plus neighbors for efficiency.

## Comparison

| | OnionDB | FAISS | ChromaDB | Pinecone | pgvector |
|---|---------|-------|----------|----------|----------|
| Dependencies | **0** | numpy | many | cloud SDK | PostgreSQL |
| Importance hierarchy | **native** | no | no | metadata only | no |
| Geometric queries | **GRF, ray** | no | no | no | no |
| Storage | SQLite file | memory/file | SQLite | cloud | server |
| Setup | `pip install` | `pip install` | `pip install` | API key | DB server |

## License

MIT -- do whatever you want with it.
