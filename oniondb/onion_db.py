"""
OnionDB — A geometric database.

Data lives BETWEEN concentric shells (like air between balloons).
Every data point has a 4-part address: (gap, θ, φ, depth).
Six native query operations: horizontal, GRF, reverse_ray,
temporal_grf, shell_scan, range_scan.

Zero external dependencies. SQLite-backed. Embedding-agnostic.

https://github.com/Niksveler/oniondb
"""
import sqlite3
import math
import struct
import json
import os
import threading
from typing import Optional

# Optional numpy acceleration — zero-dependency fallback if not installed
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class OnionDB:
    """
    A multi-shell geometric database.

    Data lives BETWEEN shells (like air between balloons).
    Every data point has a 4-part address: (gap, θ, φ, d).
    Six native query operations: horizontal, GRF, reverse_ray,
    temporal_grf, shell_scan, range_scan.
    """

    # ─── Default shell boundaries (importance thresholds) ───
    DEFAULT_BOUNDARIES = [0.95, 0.85, 0.70, 0.50, 0.00]

    # ─── Angular cell grid resolution ───
    THETA_CELLS = 12   # latitude divisions (-180° to 180°)
    PHI_CELLS = 6      # longitude divisions (-90° to 90°)

    def __init__(self, db_path: str = "onion.db", boundaries: list = None,
                 theta_cells: int = None, phi_cells: int = None):
        """
        Initialize or open an OnionDB.

        Args:
            db_path: Path to SQLite database file.
            boundaries: Importance thresholds for shell gaps (descending).
                        Default: [0.95, 0.85, 0.70, 0.50, 0.00]
                        Creates N gaps where N = len(boundaries).
            theta_cells: Latitude grid divisions (default: 12).
                         Higher values = smaller cells = faster queries
                         but sparser per-cell populations.
            phi_cells: Longitude grid divisions (default: 6).
        """
        self.db_path = db_path
        self.boundaries = boundaries or self.DEFAULT_BOUNDARIES
        self.n_gaps = len(self.boundaries)
        # Instance attrs shadow class attrs — all self.THETA_CELLS refs auto-resolve
        if theta_cells is not None:
            self.THETA_CELLS = theta_cells
        if phi_cells is not None:
            self.PHI_CELLS = phi_cells
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self._load_pca()

    def _create_schema(self):
        """Create tables if they don't exist."""
        # Backward compat: migrate 'memories' table to 'records'
        existing_tables = {row[0] for row in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if 'memories' in existing_tables and 'records' not in existing_tables:
            self.conn.execute("ALTER TABLE memories RENAME TO records")
            self.conn.commit()

        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS shells (
                shell_id    INTEGER PRIMARY KEY,
                boundary    REAL NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS records (
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

            CREATE INDEX IF NOT EXISTS idx_gap ON records(gap);
            CREATE INDEX IF NOT EXISTS idx_cell ON records(gap, cell_theta, cell_phi);
            CREATE INDEX IF NOT EXISTS idx_gap_range ON records(gap, importance);
        """)
        # Insert shell boundaries if empty
        existing = self.conn.execute("SELECT COUNT(*) FROM shells").fetchone()[0]
        if existing == 0:
            for i, b in enumerate(self.boundaries):
                self.conn.execute(
                    "INSERT INTO shells (shell_id, boundary) VALUES (?, ?)", (i, b)
                )
            self.conn.commit()

    # ═══════════════════════════════════════════
    # ADDRESSING — compute 4-part address
    # ═══════════════════════════════════════════

    def _importance_to_gap(self, importance: float) -> int:
        """Assign gap based on importance. Gap 0 = most important."""
        for i, threshold in enumerate(self.boundaries):
            if importance >= threshold:
                return i
        return self.n_gaps - 1

    def _compute_depth(self, importance: float, gap: int) -> float:
        """Compute depth within gap (0.0=inner, 1.0=outer)."""
        upper = self.boundaries[gap]
        lower = self.boundaries[gap + 1] if gap + 1 < len(self.boundaries) else 0.0
        if upper == lower:
            return 0.5
        return max(0.0, min(1.0, 1.0 - (importance - lower) / (upper - lower)))

    def _angle_to_cell(self, theta: float, phi: float) -> tuple:
        """Convert continuous angles to cell indices."""
        # theta: -180 to 180 → cell 0..THETA_CELLS-1
        ct = int(((theta + 180.0) / 360.0) * self.THETA_CELLS) % self.THETA_CELLS
        # phi: -90 to 90 → cell 0..PHI_CELLS-1
        cp = int(((phi + 90.0) / 180.0) * self.PHI_CELLS) % self.PHI_CELLS
        return ct, cp

    def _poincare_to_angles(self, x: float, y: float) -> tuple:
        """Convert Poincaré (x,y) to (theta, phi) angles in degrees."""
        theta = math.degrees(math.atan2(y, x))  # -180 to 180
        r = math.sqrt(x * x + y * y)
        phi = math.degrees(math.asin(min(r, 1.0))) - 90  # map r to -90..0
        return theta, phi

    @staticmethod
    def _embed_to_angles_v0(embedding: list) -> tuple:
        """
        DEPRECATED v0 projection. Uses raw emb[0:2].
        Only 29% of cells occupied. Kept for reference.
        """
        if not embedding or len(embedding) < 2:
            return 0.0, 0.0
        norm = math.sqrt(embedding[0] ** 2 + embedding[1] ** 2) or 1.0
        theta = math.degrees(math.atan2(embedding[1], embedding[0]))
        phi = math.degrees(math.atan2(embedding[2] if len(embedding) > 2 else 0,
                                       norm)) * 0.5
        return theta, max(-90, min(90, phi))

    def _load_pca(self):
        """Load PCA projection matrix from JSON. Falls back to v0 if missing."""
        pca_path = os.path.join(os.path.dirname(self.db_path), "pca_projection.json")
        self._pca = None
        if os.path.exists(pca_path):
            try:
                with open(pca_path) as f:
                    data = json.load(f)
                self._pca = {
                    "mean": data["mean"],
                    "pc1": data["components"][0],
                    "pc2": data["components"][1],
                    "pc1_range": data.get("pc1_range"),
                    "pc2_range": data.get("pc2_range"),
                }
            except Exception:
                self._pca = None

    def _embed_to_angles(self, embedding: list) -> tuple:
        """
        Project embedding to (theta, phi) using PCA Linear projection.

        PCA finds the 2 directions of maximum variance in embedding space.
        PC1 → theta (longitude, -180° to 180°)
        PC2 → phi (latitude, -90° to 90°)

        Linear rescale ensures items spread across the entire sphere surface.
        88% cell occupancy vs 29% with raw emb[0:2].
        93% recall@10 at r=2 vs 68% with old projection.
        """
        if not embedding or len(embedding) < 2:
            return 0.0, 0.0

        # Fallback to v0 if no PCA loaded
        if self._pca is None:
            return self._embed_to_angles_v0(embedding)

        dim = len(embedding)
        mean = self._pca["mean"]
        pc1 = self._pca["pc1"]
        pc2 = self._pca["pc2"]

        # Project: dot product of centered embedding with each PC
        x = sum((embedding[d] - mean[d]) * pc1[d] for d in range(min(dim, len(mean))))
        y = sum((embedding[d] - mean[d]) * pc2[d] for d in range(min(dim, len(mean))))

        # Linear rescale to angle range using stored bounds
        if self._pca.get("pc1_range") and self._pca.get("pc2_range"):
            x_min, x_max = self._pca["pc1_range"]
            y_min, y_max = self._pca["pc2_range"]
        else:
            # Fallback: use atan2 for theta, clamp for phi
            theta = math.degrees(math.atan2(y, x))
            phi = max(-90, min(90, y * 500))  # rough linear scale
            return theta, phi

        # Map to angles with clamp for safety
        x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else 0.5
        y_norm = (y - y_min) / (y_max - y_min) if y_max != y_min else 0.5
        x_norm = max(0, min(1, x_norm))  # clamp to [0,1]
        y_norm = max(0, min(1, y_norm))

        theta = x_norm * 360.0 - 180.0  # -180 to 180
        phi = y_norm * 180.0 - 90.0     # -90 to 90

        return theta, phi

    # ═══════════════════════════════════════════
    # INSERT — place data in the onion
    # ═══════════════════════════════════════════

    def insert(self, id: str, content: str, importance: float = 0.5,
               category: str = None, embedding: list = None,
               theta: float = None, phi: float = None,
               poincare_x: float = None, poincare_y: float = None,
               metadata: str = None, _commit: bool = True) -> dict:
        """
        Insert a data point into the onion.

        Address is computed automatically:
        - gap: from importance
        - theta, phi: from Poincaré coords, or embedding projection, or explicit
        - depth: from importance within gap range

        Returns: dict with the computed address.
        """
        gap = self._importance_to_gap(importance)
        depth = self._compute_depth(importance, gap)

        # Determine angular coordinates
        if theta is not None and phi is not None:
            pass  # Use explicit angles
        elif poincare_x is not None and poincare_y is not None:
            theta, phi = self._poincare_to_angles(poincare_x, poincare_y)
        elif embedding:
            theta, phi = self._embed_to_angles(embedding)
        else:
            theta, phi = 0.0, 0.0

        cell_t, cell_p = self._angle_to_cell(theta, phi)

        # Encode embedding as BLOB
        emb_blob = None
        if embedding:
            emb_blob = struct.pack(f'{len(embedding)}f', *embedding)

        with self._lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO records
                (id, content, gap, theta, phi, depth, cell_theta, cell_phi,
                 importance, category, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (id, content, gap, theta, phi, depth, cell_t, cell_p,
                  importance, category, emb_blob, metadata))
            if _commit:
                self.conn.commit()

        return {"id": id, "gap": gap, "theta": theta, "phi": phi,
                "depth": depth, "cell": (cell_t, cell_p)}

    # ═══════════════════════════════════════════
    # QUERY OPERATION 1: HORIZONTAL
    # Find nearby items within one gap at (θ, φ)
    # ═══════════════════════════════════════════

    # Standard column list for all queries
    _RECORD_COLS = """id, content, gap, theta, phi, depth, importance, category,
                      embedding, cell_theta, cell_phi, subshell, temporal_gap,
                      origin_date"""

    def horizontal(self, gap: int, theta: float, phi: float,
                   k: int = 10, neighbor_radius: int = 4,
                   query_embedding: list = None,
                   subshell: int = None,
                   subshell_boost: float = 0.0) -> list:
        """
        Horizontal search: find items near (θ, φ) within one gap.

        Args:
            gap: Which gap to search (0=innermost)
            theta, phi: Angular position to search near
            k: Max results to return
            neighbor_radius: How many adjacent cells to include (1=8 neighbors)
            query_embedding: Optional embedding for cosine ranking
            subshell: Hard filter (only this subshell) OR soft boost target
            subshell_boost: If > 0, BOOST items in this subshell by this
                            amount instead of hard filtering. E.g. 0.1 adds
                            10% to matching items' scores. (Fix #3: soft filter)

        Returns: List of record dicts, sorted by relevance.
        """
        ct, cp = self._angle_to_cell(theta, phi)

        # Build cell range including neighbors
        cells = []
        for dt in range(-neighbor_radius, neighbor_radius + 1):
            for dp in range(-neighbor_radius, neighbor_radius + 1):
                t = (ct + dt) % self.THETA_CELLS
                p = (cp + dp) % self.PHI_CELLS
                cells.append((t, p))

        # Query all matching cells
        placeholders = ",".join(["(?,?)" for _ in cells])
        params = [gap]
        for t, p in cells:
            params.extend([t, p])

        # Hard filter only when boost is 0 (strict mode)
        subshell_clause = ""
        if subshell is not None and subshell_boost == 0.0:
            subshell_clause = "AND subshell = ?"
            params.append(subshell)

        sql = f"""
            SELECT {self._RECORD_COLS}
            FROM records
            WHERE gap = ?
            AND (cell_theta, cell_phi) IN ({placeholders})
            {subshell_clause}
        """
        rows = self.conn.execute(sql, params).fetchall()

        results = self._rows_to_dicts(rows)

        # Rank by cosine similarity if embedding provided
        if query_embedding and results:
            for r in results:
                r["score"] = self._cosine(query_embedding, r.get("_emb"))
                # Soft subshell boost: same cluster gets a score bump
                if subshell is not None and subshell_boost > 0.0:
                    if r.get("subshell") == subshell:
                        r["score"] *= (1.0 + subshell_boost)
            results.sort(key=lambda x: -x.get("score", 0))
        else:
            # Rank by angular distance
            for r in results:
                r["score"] = -self._angular_dist(theta, phi,
                                                  r["theta"], r["phi"])
            results.sort(key=lambda x: -x["score"])

        return results[:k]

    # ═══════════════════════════════════════════
    # QUERY OPERATION 2: GRF (Geometric Ray Filter)
    # Pierce all gaps at direction (θ, φ)
    # ═══════════════════════════════════════════

    def grf(self, theta: float, phi: float,
            k_per_gap: int = 5, neighbor_radius: int = 4,
            query_embedding: list = None,
            subshell: int = None,
            subshell_boost: float = 0.0) -> dict:
        """
        GRF — Geometric Ray Filter.

        The signature operation of OnionDB. Pierces ALL depth shells at
        direction (θ, φ), returning a topic-depth profile. Like drilling
        a core sample through the onion.

        The embedding acts as a semantic filter: same direction, different
        embedding → different results. Geometry selects candidates,
        cosine ranks by topic relevance.

        Args:
            theta, phi: Direction to drill (angular coordinates)
            k_per_gap: Max results per gap
            neighbor_radius: Cell neighbor expansion
            query_embedding: Semantic filter — if provided, results within
                             each gap are ranked by cosine similarity
                             (topic precision). Without it, ranked by
                             angular distance (geometric proximity).
            subshell: Optional topic cluster filter (0-14).
                      When set with boost=0: hard filter (only this subshell).
                      When set with boost>0: soft boost (prefer this subshell).
            subshell_boost: Score multiplier for matching subshell items.
                            0.0 = hard filter, >0.0 = soft boost.

        Returns: Dict mapping gap_id → list of records.
        """
        result = {}
        for gap_id in range(self.n_gaps):
            items = self.horizontal(gap_id, theta, phi,
                                    k=k_per_gap,
                                    neighbor_radius=neighbor_radius,
                                    query_embedding=query_embedding,
                                    subshell=subshell,
                                    subshell_boost=subshell_boost)
            if items:
                result[gap_id] = items
        return result

    # Backward-compatible alias
    ray = grf

    # ═══════════════════════════════════════════
    # QUERY OPERATION 2b: REVERSE RAY (Curved)
    # Follow semantic gravity from outer → center
    # ═══════════════════════════════════════════

    def reverse_ray(self, start_embedding: list,
                    start_gap: int = None, start_theta: float = None,
                    start_phi: float = None,
                    neighbor_radius: int = 2,
                    beam_width: int = 1) -> dict:
        """
        Reverse Ray — curved semantic gravity trace.

        Unlike GRF (straight vector, fixed direction), the reverse ray
        starts at a point and follows semantic similarity inward. At each
        gap, it finds the best cosine match, then uses THAT item's
        angular position for the next hop. The ray bends.

        The path itself is a vector that carries information:
        - Straight path → well-organized knowledge (same topic all depths)
        - Curved path → fragmented knowledge (topic connects to different
          areas at different depths)

        Args:
            start_embedding: Embedding to trace inward from
            start_gap: Starting gap (default: outermost gap with data)
            start_theta: Starting angle (if None, auto-detect from best
                         match in start_gap)
            start_phi: Starting angle
            neighbor_radius: Cell search radius at each hop
            beam_width: Number of candidate paths to maintain per hop.
                        1 = greedy (default, original behavior).
                        >1 = beam search — keeps top-k candidates at
                        each hop, traces all paths, returns the one with
                        the highest average score.

        Returns: dict with:
            - path: list of {gap, theta, phi, record, score} at each hop
            - curvature: total angular deviation (0=straight, high=curved)
            - records: list of records found along the path
            - path_vector: list of (theta, phi) coordinates tracing the ray
            - beam_paths: (only if beam_width > 1) all explored paths
        """
        # Determine starting position
        if start_gap is None:
            start_gap = self.n_gaps - 1  # outermost

        if start_theta is None or start_phi is None:
            best = self._find_best_in_gap(start_gap, start_embedding,
                                          neighbor_radius=neighbor_radius)
            if best:
                start_theta = best["theta"]
                start_phi = best["phi"]
            else:
                start_theta, start_phi = 0.0, 0.0

        if beam_width <= 1:
            # Greedy mode — original behavior, no overhead
            return self._reverse_ray_greedy(
                start_embedding, start_gap, start_theta, start_phi,
                neighbor_radius
            )

        # ─── Beam search mode ───
        # Each beam is (path_so_far, current_theta, current_phi, current_emb, total_score, total_curvature)
        beams = [(
            [],           # path
            start_theta,  # current theta
            start_phi,    # current phi
            start_embedding,  # current embedding
            0.0,          # total score
            0.0,          # total curvature
        )]

        for gap_id in range(start_gap, -1, -1):
            next_beams = []
            for path, ct, cp, ce, tscore, tcurv in beams:
                candidates = self.horizontal(
                    gap_id, ct, cp,
                    k=max(5, beam_width * 2),
                    neighbor_radius=neighbor_radius,
                    query_embedding=ce
                )

                if not candidates:
                    # Dead end — keep beam with empty hop
                    new_path = path + [{
                        "gap": gap_id, "theta": ct, "phi": cp,
                        "record": None, "score": 0.0
                    }]
                    next_beams.append((new_path, ct, cp, ce, tscore, tcurv))
                    continue

                # Branch: each of top beam_width candidates creates a new beam
                for cand in candidates[:beam_width]:
                    bend = self._angular_dist(ct, cp, cand["theta"], cand["phi"])
                    score = cand.get("score", 0.0)
                    new_path = path + [{
                        "gap": gap_id,
                        "theta": cand["theta"],
                        "phi": cand["phi"],
                        "record": cand,
                        "score": score,
                        "bend": round(bend, 2)
                    }]
                    new_emb = cand.get("_emb") or ce
                    next_beams.append((
                        new_path, cand["theta"], cand["phi"],
                        new_emb, tscore + score, tcurv + bend
                    ))

            # Prune to top beam_width beams by total score
            next_beams.sort(key=lambda b: -b[4])  # sort by total score desc
            beams = next_beams[:beam_width]

        # Select best beam (highest avg score)
        best_beam = max(beams, key=lambda b: b[4] / max(len(b[0]), 1))
        path, _, _, _, _, total_curvature = best_beam

        records = [p["record"] for p in path if p["record"]]
        path_vector = [(p["theta"], p["phi"]) for p in path]
        n_hops = max(len(path) - 1, 1)
        avg_curvature = total_curvature / n_hops

        result = {
            "path": path,
            "curvature": round(total_curvature, 2),
            "avg_curvature": round(avg_curvature, 2),
            "records": records,
            "path_vector": path_vector,
            "straight": avg_curvature < 15.0,
            "n_hops": len(path),
            "beam_width": beam_width,
            "beam_paths_explored": len(beams),
        }
        return result

    def _reverse_ray_greedy(self, start_embedding, start_gap, start_theta,
                            start_phi, neighbor_radius):
        """Original greedy reverse ray — kept as fast path for beam_width=1."""
        path = []
        current_theta = start_theta
        current_phi = start_phi
        current_emb = start_embedding
        total_curvature = 0.0

        for gap_id in range(start_gap, -1, -1):  # outer → inner
            candidates = self.horizontal(
                gap_id, current_theta, current_phi,
                k=5, neighbor_radius=neighbor_radius,
                query_embedding=current_emb
            )

            if not candidates:
                path.append({
                    "gap": gap_id, "theta": current_theta,
                    "phi": current_phi, "record": None, "score": 0.0
                })
                continue

            best = candidates[0]
            bend = self._angular_dist(
                current_theta, current_phi,
                best["theta"], best["phi"]
            )
            total_curvature += bend

            path.append({
                "gap": gap_id,
                "theta": best["theta"],
                "phi": best["phi"],
                "record": best,
                "score": best.get("score", 0.0),
                "bend": round(bend, 2)
            })

            current_theta = best["theta"]
            current_phi = best["phi"]
            if best.get("_emb"):
                current_emb = best["_emb"]

        records = [p["record"] for p in path if p["record"]]
        path_vector = [(p["theta"], p["phi"]) for p in path]
        n_hops = max(len(path) - 1, 1)
        avg_curvature = total_curvature / n_hops

        return {
            "path": path,
            "curvature": round(total_curvature, 2),
            "avg_curvature": round(avg_curvature, 2),
            "records": records,
            "path_vector": path_vector,
            "straight": avg_curvature < 15.0,
            "n_hops": len(path)
        }

    def _find_best_in_gap(self, gap: int, embedding: list,
                          neighbor_radius: int = 3) -> Optional[dict]:
        """Find the single best cosine match in an entire gap."""
        # Scan wider to find best starting point
        rows = self.conn.execute(f"""
            SELECT {self._RECORD_COLS}
            FROM records WHERE gap = ?
        """, (gap,)).fetchall()

        results = self._rows_to_dicts(rows)
        if not results:
            return None

        for r in results:
            r["score"] = self._cosine(embedding, r.get("_emb"))
        results.sort(key=lambda x: -x["score"])
        return results[0]

    # ═══════════════════════════════════════════
    # QUERY OPERATION 3: SHELL SCAN
    # Return everything in one gap
    # ═══════════════════════════════════════════

    def shell_scan(self, gap: int, limit: int = 1000) -> list:
        """
        Shell scan: return all data in gap E.

        Like peeling the onion to one layer and seeing everything there.

        Args:
            gap: Which gap to scan (0=innermost/core)
            limit: Maximum results

        Returns: List of record dicts.
        """
        rows = self.conn.execute(f"""
            SELECT {self._RECORD_COLS}
            FROM records
            WHERE gap = ?
            ORDER BY importance DESC
            LIMIT ?
        """, (gap, limit)).fetchall()

        return self._rows_to_dicts(rows)

    # ═══════════════════════════════════════════
    # QUERY OPERATION 5: TEMPORAL GRF
    # Pierce time shells instead of importance shells
    # ═══════════════════════════════════════════

    def temporal_grf(self, theta: float, phi: float,
                     k_per_gap: int = 5, neighbor_radius: int = 1,
                     query_embedding: list = None,
                     subshell: int = None) -> dict:
        """
        Temporal GRF — drill through TIME shells at direction (θ, φ).

        Like the importance GRF but organized by when records were created.
        Shows how a topic evolved over time at different importance levels.

        Temporal gaps:
            T0 = most recent (~last 10 days)
            T4 = oldest (~first month)

        Args:
            Same as grf(), but drills through temporal_gap instead of gap.

        Returns: Dict mapping temporal_gap_id → list of records.
        """
        ct, cp = self._angle_to_cell(theta, phi)

        cells = []
        for dt in range(-neighbor_radius, neighbor_radius + 1):
            for dp in range(-neighbor_radius, neighbor_radius + 1):
                t = (ct + dt) % self.THETA_CELLS
                p = (cp + dp) % self.PHI_CELLS
                cells.append((t, p))

        placeholders = ",".join(["(?,?)" for _ in cells])

        result = {}
        for tgap in range(5):  # 5 temporal gaps
            params = [tgap]
            for t, p in cells:
                params.extend([t, p])

            subshell_clause = ""
            if subshell is not None:
                subshell_clause = "AND subshell = ?"
                params.append(subshell)

            sql = f"""
                SELECT {self._RECORD_COLS}
                FROM records
                WHERE temporal_gap = ?
                AND (cell_theta, cell_phi) IN ({placeholders})
                {subshell_clause}
            """
            rows = self.conn.execute(sql, params).fetchall()
            items = self._rows_to_dicts(rows)

            if query_embedding and items:
                for m in items:
                    m["score"] = self._cosine(query_embedding, m.get("_emb"))
                items.sort(key=lambda x: -x.get("score", 0))
                items = items[:k_per_gap]
            elif items:
                for m in items:
                    m["score"] = -self._angular_dist(
                        theta, phi, m["theta"], m["phi"])
                items.sort(key=lambda x: -x["score"])
                items = items[:k_per_gap]

            if items:
                result[tgap] = items
        return result

    # ═══════════════════════════════════════════
    # QUERY OPERATION 6: RANGE SCAN
    # Return everything between shell E and shell B
    # ═══════════════════════════════════════════

    def range_scan(self, gap_start: int, gap_end: int,
                   limit: int = 1000) -> list:
        """
        Range scan: return all data between gap E and gap B (inclusive).

        Like cutting a thick ring out of the onion.

        Args:
            gap_start: Inner boundary (lower gap number = more important)
            gap_end: Outer boundary (higher gap number = less important)
            limit: Maximum results

        Returns: List of memory dicts.
        """
        rows = self.conn.execute(f"""
            SELECT {self._RECORD_COLS}
            FROM records
            WHERE gap >= ? AND gap <= ?
            ORDER BY gap ASC, importance DESC
            LIMIT ?
        """, (gap_start, gap_end, limit)).fetchall()

        return self._rows_to_dicts(rows)

    # ═══════════════════════════════════════════
    # STATS & INSPECTION
    # ═══════════════════════════════════════════

    def stats(self) -> dict:
        """Return database statistics."""
        total = self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        gaps = {}
        for row in self.conn.execute(
            "SELECT gap, COUNT(*), AVG(importance), MIN(importance), MAX(importance) "
            "FROM records GROUP BY gap ORDER BY gap"
        ).fetchall():
            gaps[row[0]] = {
                "count": row[1], "avg_importance": round(row[2], 3),
                "min_importance": round(row[3], 3),
                "max_importance": round(row[4], 3)
            }

        categories = {}
        for row in self.conn.execute(
            "SELECT category, COUNT(*) FROM records GROUP BY category ORDER BY COUNT(*) DESC"
        ).fetchall():
            categories[row[0]] = row[1]

        return {
            "total": total, "n_gaps": self.n_gaps,
            "boundaries": self.boundaries,
            "gaps": gaps, "categories": categories,
            "grid": f"{self.THETA_CELLS}x{self.PHI_CELLS}"
        }

    def cell_density(self, gap: int) -> list:
        """Show how many records are in each cell of a gap."""
        rows = self.conn.execute("""
            SELECT cell_theta, cell_phi, COUNT(*)
            FROM records WHERE gap = ?
            GROUP BY cell_theta, cell_phi
            ORDER BY COUNT(*) DESC
        """, (gap,)).fetchall()
        return [{"cell": (r[0], r[1]), "count": r[2]} for r in rows]

    # ═══════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════

    def _rows_to_dicts(self, rows: list) -> list:
        """Convert query rows to record dicts. Expects 14-column rows."""
        results = []
        for r in rows:
            mid, content, gap, theta, phi, depth, imp, cat, emb, ct, cp, \
                sub, tgap, odate = r
            d = {
                "id": mid, "content": content, "gap": gap,
                "theta": round(theta, 2), "phi": round(phi, 2),
                "depth": round(depth, 3), "importance": imp,
                "category": cat, "cell": (ct, cp),
                "subshell": sub, "temporal_gap": tgap,
                "origin_date": odate,
                "address": f"({gap}, {theta:.1f}°, {phi:.1f}°, d={depth:.2f})"
            }
            if emb:
                d["_emb"] = self._decode_embedding(emb)
            results.append(d)
        return results

    @staticmethod
    def _decode_embedding(blob) -> Optional[list]:
        """Decode float32 embedding from BLOB.

        Uses numpy if available for ~5x faster decoding on large embeddings.
        Falls back to struct.unpack (zero dependencies).
        """
        if blob is None:
            return None
        if isinstance(blob, str):
            blob = blob.encode('latin-1')
        if _HAS_NUMPY:
            return np.frombuffer(blob, dtype=np.float32).tolist()
        n = len(blob) // 4
        return list(struct.unpack(f'{n}f', blob))

    @staticmethod
    def _cosine(a, b) -> float:
        """Cosine similarity between two vectors.

        Uses numpy if available for ~10x faster computation.
        Falls back to pure Python (zero dependencies).
        """
        if not a or not b:
            return 0.0
        if _HAS_NUMPY:
            a_arr = np.asarray(a, dtype=np.float32)
            b_arr = np.asarray(b, dtype=np.float32)
            na = np.linalg.norm(a_arr)
            nb = np.linalg.norm(b_arr)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a_arr, b_arr) / (na * nb))
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _angular_dist(t1, p1, t2, p2) -> float:
        """Angular distance in degrees between two positions."""
        dt = abs(t1 - t2)
        if dt > 180:
            dt = 360 - dt
        dp = abs(p1 - p2)
        return math.sqrt(dt * dt + dp * dp)

    # ═══════════════════════════════════════════
    # CRUD — get, delete, count, batch_insert
    # ═══════════════════════════════════════════

    def get(self, id: str) -> Optional[dict]:
        """
        Get a single record by ID.

        Args:
            id: The record ID.

        Returns: Record dict or None if not found.
        """
        row = self.conn.execute(f"""
            SELECT {self._RECORD_COLS}
            FROM records WHERE id = ?
        """, (id,)).fetchone()
        if row is None:
            return None
        results = self._rows_to_dicts([row])
        return results[0] if results else None

    def delete(self, id: str) -> bool:
        """
        Delete a record by ID.

        Args:
            id: The record ID to delete.

        Returns: True if a row was deleted, False if ID not found.
        """
        with self._lock:
            cursor = self.conn.execute("DELETE FROM records WHERE id = ?", (id,))
            self.conn.commit()
        return cursor.rowcount > 0

    def count(self, gap: int = None) -> int:
        """
        Count records, optionally filtered by gap.

        Args:
            gap: If provided, count only records in this gap.

        Returns: Number of records.
        """
        if gap is not None:
            return self.conn.execute(
                "SELECT COUNT(*) FROM records WHERE gap = ?", (gap,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    def batch_insert(self, items: list) -> list:
        """
        Insert multiple items efficiently in a single transaction.

        Uses a single commit for all items instead of one per item.
        For 1000 items this is ~100x faster than individual inserts.

        Args:
            items: List of dicts, each with keys matching insert() params:
                   id, content, importance, category, embedding, metadata, etc.

        Returns: List of address dicts for each inserted item.
        """
        addresses = []
        with self._lock:
            for item in items:
                addr = self.insert(
                    id=item["id"],
                    content=item["content"],
                    importance=item.get("importance", 0.5),
                    category=item.get("category"),
                    embedding=item.get("embedding"),
                    theta=item.get("theta"),
                    phi=item.get("phi"),
                    poincare_x=item.get("poincare_x"),
                    poincare_y=item.get("poincare_y"),
                    metadata=item.get("metadata"),
                    _commit=False,
                )
                addresses.append(addr)
            self.conn.commit()
        return addresses

    # ═══════════════════════════════════════════
    # PROJECTION — self-calibrating PCA
    # ═══════════════════════════════════════════

    def fit_projection(self, save: bool = True) -> dict:
        """
        Fit PCA projection from stored embeddings.

        Computes the 2 principal components of maximum variance from all
        stored embeddings, then saves the projection matrix for future use.
        Call this after inserting a representative dataset to improve
        angular distribution across the sphere.

        Uses numpy when available for ~50x faster computation.
        Falls back to pure Python power iteration.

        Args:
            save: If True, write pca_projection.json next to the database.

        Returns: Dict with stats (n_samples, occupancy_before, occupancy_after).
        """
        # Collect all embeddings
        rows = self.conn.execute(
            "SELECT embedding FROM records WHERE embedding IS NOT NULL"
        ).fetchall()
        if len(rows) < 10:
            return {"error": "Need at least 10 embeddings to fit PCA",
                    "n_samples": len(rows)}

        embeddings = []
        for (blob,) in rows:
            emb = self._decode_embedding(blob)
            if emb:
                embeddings.append(emb)

        n = len(embeddings)
        dim = len(embeddings[0])

        # Measure occupancy before
        cells_before = set()
        for emb in embeddings:
            t, p = self._embed_to_angles(emb)
            ct, cp = self._angle_to_cell(t, p)
            cells_before.add((ct, cp))
        occ_before = len(cells_before) / (self.THETA_CELLS * self.PHI_CELLS)

        if _HAS_NUMPY:
            # Fast path: numpy vectorized PCA
            data = np.array(embeddings, dtype=np.float32)
            mean = np.mean(data, axis=0)
            centered = data - mean
            cov = centered.T @ centered / n
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # eigh returns ascending order — take last 2
            pc1 = eigenvectors[:, -1].tolist()
            pc2 = eigenvectors[:, -2].tolist()
            mean = mean.tolist()

            # Project for ranges
            pc1_vals = (centered @ np.array(pc1, dtype=np.float32)).tolist()
            pc2_vals = (centered @ np.array(pc2, dtype=np.float32)).tolist()
        else:
            # Pure Python fallback: power iteration
            mean = [0.0] * dim
            for emb in embeddings:
                for d in range(dim):
                    mean[d] += emb[d]
            mean = [m / n for m in mean]

            centered = []
            for emb in embeddings:
                centered.append([emb[d] - mean[d] for d in range(dim)])

            def power_iteration(data, n_iter=50):
                """Find top eigenvector via power iteration."""
                vec = [1.0 / math.sqrt(dim)] * dim
                for _ in range(n_iter):
                    new_vec = [0.0] * dim
                    for row in data:
                        dot = sum(row[d] * vec[d] for d in range(dim))
                        for d in range(dim):
                            new_vec[d] += dot * row[d]
                    norm = math.sqrt(sum(v * v for v in new_vec)) or 1.0
                    vec = [v / norm for v in new_vec]
                return vec

            pc1 = power_iteration(centered)
            deflated = []
            for row in centered:
                dot = sum(row[d] * pc1[d] for d in range(dim))
                deflated.append([row[d] - dot * pc1[d] for d in range(dim)])
            pc2 = power_iteration(deflated)

            pc1_vals = [sum(c[d] * pc1[d] for d in range(dim)) for c in centered]
            pc2_vals = [sum(c[d] * pc2[d] for d in range(dim)) for c in centered]

        pc1_range = [min(pc1_vals), max(pc1_vals)]
        pc2_range = [min(pc2_vals), max(pc2_vals)]

        # Save projection
        pca_data = {
            "components": [pc1, pc2],
            "mean": mean,
            "pc1_range": pc1_range,
            "pc2_range": pc2_range,
            "n_samples": n,
            "dim": dim,
        }

        if save:
            pca_path = os.path.join(
                os.path.dirname(self.db_path) or ".", "pca_projection.json"
            )
            with open(pca_path, "w") as f:
                json.dump(pca_data, f)

        # Reload and measure after
        self._pca = {
            "mean": mean, "pc1": pc1, "pc2": pc2,
            "pc1_range": pc1_range, "pc2_range": pc2_range,
        }

        cells_after = set()
        for emb in embeddings:
            t, p = self._embed_to_angles(emb)
            ct, cp = self._angle_to_cell(t, p)
            cells_after.add((ct, cp))
        occ_after = len(cells_after) / (self.THETA_CELLS * self.PHI_CELLS)

        return {
            "n_samples": n,
            "dim": dim,
            "occupancy_before": round(occ_before, 3),
            "occupancy_after": round(occ_after, 3),
            "pc1_range": pc1_range,
            "pc2_range": pc2_range,
        }

    # ═══════════════════════════════════════════
    # BOUNDARY CALIBRATION & REINDEX
    # ═══════════════════════════════════════════

    def fit_boundaries(self, n_gaps: int = 5) -> list:
        """
        Suggest optimal shell boundaries from the data distribution.

        Computes quantile-based boundaries so each gap contains roughly
        the same number of records. Use with reindex() to apply.

        Args:
            n_gaps: Number of gaps to create (default: 5).

        Returns: List of boundary thresholds (descending), suitable for
                 passing to the constructor or reindex().

        Example:
            >>> new_bounds = db.fit_boundaries(n_gaps=5)
            >>> db.reindex(boundaries=new_bounds)
        """
        rows = self.conn.execute(
            "SELECT importance FROM records"
        ).fetchall()
        if not rows:
            return self.boundaries

        values = sorted([r[0] for r in rows], reverse=True)
        n = len(values)

        # Compute quantile boundaries (evenly spaced percentiles)
        boundaries = []
        for i in range(n_gaps):
            idx = int((i / n_gaps) * n)
            idx = min(idx, n - 1)
            boundaries.append(round(values[idx], 4))

        # Ensure descending and last is 0.0
        boundaries = sorted(set(boundaries), reverse=True)
        if boundaries[-1] != 0.0:
            boundaries.append(0.0)

        # Ensure we have exactly n_gaps boundaries
        while len(boundaries) < n_gaps:
            boundaries.insert(-1, boundaries[-2] / 2)
        if len(boundaries) > n_gaps:
            boundaries = boundaries[:n_gaps - 1] + [0.0]

        return boundaries

    def reindex(self, boundaries: list = None) -> dict:
        """
        Recalculate all gap, depth, and cell assignments.

        Use after changing boundaries, grid resolution, or PCA projection.
        Updates all records in-place without deleting data.

        Args:
            boundaries: New boundaries to apply. If None, uses current.

        Returns: Dict with reindex stats (total, updated, boundaries).
        """
        if boundaries is not None:
            self.boundaries = boundaries
            self.n_gaps = len(boundaries)
            # Update shells table
            with self._lock:
                self.conn.execute("DELETE FROM shells")
                for i, b in enumerate(self.boundaries):
                    self.conn.execute(
                        "INSERT INTO shells (shell_id, boundary) VALUES (?, ?)",
                        (i, b)
                    )

        # Read all records
        rows = self.conn.execute(
            "SELECT id, importance, embedding FROM records"
        ).fetchall()

        updated = 0
        with self._lock:
            for rid, importance, emb_blob in rows:
                gap = self._importance_to_gap(importance)
                depth = self._compute_depth(importance, gap)

                # Recompute angles if embedding exists
                if emb_blob:
                    emb = self._decode_embedding(emb_blob)
                    theta, phi = self._embed_to_angles(emb)
                else:
                    # Keep existing angles
                    row = self.conn.execute(
                        "SELECT theta, phi FROM records WHERE id = ?",
                        (rid,)
                    ).fetchone()
                    theta, phi = row[0], row[1]

                cell_t, cell_p = self._angle_to_cell(theta, phi)

                self.conn.execute("""
                    UPDATE records
                    SET gap = ?, depth = ?, theta = ?, phi = ?,
                        cell_theta = ?, cell_phi = ?
                    WHERE id = ?
                """, (gap, depth, theta, phi, cell_t, cell_p, rid))
                updated += 1
            self.conn.commit()

        return {
            "total": len(rows),
            "updated": updated,
            "boundaries": self.boundaries,
            "n_gaps": self.n_gaps,
        }

    # ═══════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        total = self.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        return f"OnionDB('{self.db_path}', {total} records, {self.n_gaps} gaps)"
