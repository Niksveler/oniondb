import sqlite3
import argparse
import random
from pathlib import Path

def migrate(db_path: str, current_clock: int = None, stagger: bool = False):
    """
    Migrate OnionDB to Phase 5.
    If current_clock is provided, existing records are set to a value near current_clock
    to prevent an immediate flood of reviews in the scheduler.
    """
    path = Path(db_path)
    if not path.exists():
        print(f"Error: Database {db_path} not found.")
        return

    print(f"Migrating {db_path} to Phase 5 schema...")
    conn = sqlite3.connect(db_path)
    
    # Check if columns exist
    try:
        conn.execute("SELECT mass, last_review_clock FROM records LIMIT 1")
        print("Columns already exist. Schema is up to date.")
    except Exception:
        print("Adding 'mass' and 'last_review_clock' columns...")
        # Since SQLite ALTER TABLE ADD COLUMN cannot be rolled back easily if we just
        # ran last_review_at_step previously, let's gracefully try to add them.
        try:
            conn.execute("ALTER TABLE records ADD COLUMN mass REAL NOT NULL DEFAULT 1.0")
        except Exception:
            pass # might exist
        try:
            conn.execute("ALTER TABLE records ADD COLUMN last_review_clock INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        conn.commit()

    if current_clock is not None:
        print(f"Applying review clock offsets based on current_clock={current_clock}...")
        
        # Get all record IDs
        rows = conn.execute("SELECT id FROM records").fetchall()
        
        if stagger:
            print("Staggering review clock to prevent review queue flooding...")
            # Update records with a slight jitter around current_clock so they don't all pop at once
            for (rid,) in rows:
                jitter = random.randint(0, min(100, current_clock))
                clock_val = max(0, current_clock - jitter)
                conn.execute("UPDATE records SET last_review_clock = ? WHERE id = ?", (clock_val, rid))
        else:
            print(f"Setting all records to last_review_clock={current_clock}...")
            conn.execute("UPDATE records SET last_review_clock = ?", (current_clock,))
            
        conn.commit()
        
    print(f"Migration complete for {db_path}.")
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 Schema Migration for OnionDB")
    parser.add_argument("--db", type=str, default="onion.db", help="Path to SQLite database")
    parser.add_argument("--clock", type=int, help="Current clock to use for existing records")
    parser.add_argument("--stagger", action="store_true", help="Stagger existing records slightly behind current clock")
    
    args = parser.parse_args()
    migrate(args.db, current_clock=args.clock, stagger=args.stagger)
