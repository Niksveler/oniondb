"""
OnionDB CLI — inspect and query databases from the command line.

Usage:
    python -m oniondb stats mydb.db
    python -m oniondb density mydb.db --gap 0
    python -m oniondb shell mydb.db --gap 0 --limit 5
    python -m oniondb info mydb.db
"""
import argparse
import json
import sys
import os

from .onion_db import OnionDB


def cmd_stats(args):
    """Show database statistics."""
    db = OnionDB(args.db)
    s = db.stats()
    print(f"OnionDB: {args.db}")
    print(f"  Records: {s['total']}")
    print(f"  Gaps:    {s['n_gaps']} (boundaries: {s['boundaries']})")
    print(f"  Grid:    {s['grid']}")
    print()
    if s["gaps"]:
        print("  Gap | Count | Avg Imp | Min Imp | Max Imp")
        print("  ----|-------|---------|---------|--------")
        for gap_id, info in sorted(s["gaps"].items()):
            print(f"  {gap_id:>3} | {info['count']:>5} | {info['avg_importance']:>7.3f} "
                  f"| {info['min_importance']:>7.3f} | {info['max_importance']:>7.3f}")
    print()
    if s["categories"]:
        print("  Categories:")
        for cat, count in s["categories"].items():
            print(f"    {cat or '(none)'}: {count}")
    db.close()


def cmd_density(args):
    """Show cell density map for a gap."""
    db = OnionDB(args.db)
    cells = db.cell_density(args.gap)
    total = sum(c["count"] for c in cells)
    occupied = len(cells)
    total_cells = db.THETA_CELLS * db.PHI_CELLS
    print(f"Gap {args.gap}: {total} records across {occupied}/{total_cells} cells "
          f"({occupied / total_cells * 100:.0f}% occupancy)")
    print()
    if cells:
        print("  Cell (θ,φ) | Count")
        print("  -----------|------")
        for c in cells[:args.limit]:
            print(f"  ({c['cell'][0]:>3},{c['cell'][1]:>2})   | {c['count']}")
        if len(cells) > args.limit:
            print(f"  ... and {len(cells) - args.limit} more cells")
    db.close()


def cmd_shell(args):
    """Scan a shell and show records."""
    db = OnionDB(args.db)
    records = db.shell_scan(args.gap, limit=args.limit)
    print(f"Gap {args.gap}: {len(records)} records")
    print()
    for r in records:
        addr = r.get("address", "")
        snippet = r["content"][:80].replace("\n", " ")
        print(f"  [{r['id'][:12]}] imp={r['importance']:.3f} {addr}")
        print(f"    {snippet}")
        print()
    db.close()


def cmd_info(args):
    """Show database info and PCA status."""
    db = OnionDB(args.db)
    pca_path = os.path.join(os.path.dirname(args.db) or ".", "pca_projection.json")
    has_pca_db = db._pca is not None
    has_pca_json = os.path.exists(pca_path)
    has_pca = has_pca_db or has_pca_json

    print(f"OnionDB: {args.db}")
    print(f"  File size:  {os.path.getsize(args.db):,} bytes")
    print(f"  Records:    {db.count()}")
    print(f"  Gaps:       {db.n_gaps}")
    print(f"  Grid:       {db.THETA_CELLS}×{db.PHI_CELLS} = {db.THETA_CELLS * db.PHI_CELLS} cells")
    if has_pca_db:
        print(f"  PCA fitted: yes (stored in database)")
    elif has_pca_json:
        print(f"  PCA fitted: yes ({pca_path})")
    else:
        print(f"  PCA fitted: no — using v0 projection")

    if has_pca_json:
        with open(pca_path) as f:
            pca = json.load(f)
        print(f"  PCA dim:    {pca.get('dim', '?')}")
        print(f"  PCA samples:{pca.get('n_samples', '?')}")

    # Check for numpy
    try:
        import numpy as np
        print(f"  Numpy:      yes (v{np.__version__}) — accelerated cosine")
    except ImportError:
        print(f"  Numpy:      no — using pure Python fallback")

    db.close()


def main():
    parser = argparse.ArgumentParser(
        prog="oniondb",
        description="OnionDB CLI — inspect and query geometric databases"
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # stats
    p_stats = sub.add_parser("stats", help="Show database statistics")
    p_stats.add_argument("db", help="Path to OnionDB .db file")

    # density
    p_density = sub.add_parser("density", help="Show cell density map")
    p_density.add_argument("db", help="Path to OnionDB .db file")
    p_density.add_argument("--gap", type=int, default=0, help="Gap to inspect (default: 0)")
    p_density.add_argument("--limit", type=int, default=20, help="Max cells to show")

    # shell
    p_shell = sub.add_parser("shell", help="Scan a shell and show records")
    p_shell.add_argument("db", help="Path to OnionDB .db file")
    p_shell.add_argument("--gap", type=int, default=0, help="Gap to scan (default: 0)")
    p_shell.add_argument("--limit", type=int, default=10, help="Max records to show")

    # info
    p_info = sub.add_parser("info", help="Show database info and PCA status")
    p_info.add_argument("db", help="Path to OnionDB .db file")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "stats": cmd_stats,
        "density": cmd_density,
        "shell": cmd_shell,
        "info": cmd_info,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
