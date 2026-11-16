"""levin-phase-floor prompt 06 section 6: a read-only look at A3 v2's producer tables, to size the
in-process re-run of the producers (new in 06; not part of the run). Every shard is opened
sqlite3 mode=ro. Prints, per shard: the GkWKBIntegration, GkWKBValue, TkWKBIntegration,
TkWKBValue, GkSource and GkSourceValue row counts, the value tables' columns, and per
GkWKBIntegration the number of value rows (min/median/max)."""
import sqlite3
import statistics
import sys
from pathlib import Path

STORE_DIR = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores")
for shard in range(4):
    p = STORE_DIR / f"handover-A3-v2-lambdacdm-shard{shard:04d}.sqlite"
    conn = sqlite3.connect(f"{p.as_uri()}?mode=ro", uri=True)
    print(f"== shard {shard}")
    for t in ["GkWKBIntegration", "GkWKBValue", "TkWKBIntegration", "TkWKBValue", "GkSource", "GkSourceValue", "GkNumericIntegration", "QuadSourceIntegral", "GkSourcePolicyData"]:
        try:
            n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        except sqlite3.OperationalError as e:
            n = f"({e})"
        print(f"  {t}: {n}")
    if shard == 0:
        for t in ["GkWKBValue", "TkWKBValue", "GkSourceValue", "GkWKBIntegration", "TkWKBIntegration"]:
            cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
            print(f"  columns {t}: {cols}")
    rows = conn.execute('SELECT wkb_serial, COUNT(*) FROM "GkWKBValue" GROUP BY wkb_serial').fetchall()
    if rows:
        c = [r[1] for r in rows]
        print(f"  GkWKBValue per GkWKBIntegration: n={len(c)} min={min(c)} median={statistics.median(c)} max={max(c)} total={sum(c)}")
    rows = conn.execute('SELECT wavenumber_exit_serial, COUNT(*), SUM(numeric_serial IS NULL) FROM "GkWKBIntegration" GROUP BY wavenumber_exit_serial').fetchall()
    print(f"  GkWKBIntegration per k (k serial, n, n pure): {rows}")
    conn.close()
