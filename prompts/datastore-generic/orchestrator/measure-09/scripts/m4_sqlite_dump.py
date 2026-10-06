"""M4(a): dump every SQLite file under <dir> (opened mode=ro): sqlite_master (type, name, tbl_name,
sql; not rootpage) sorted, and every table's rows ordered by all columns. Writes JSON to <out>.
Usage: python m4_sqlite_dump.py <dir> <out.json>"""
import json, sqlite3, sys
from pathlib import Path
root, out = Path(sys.argv[1]), Path(sys.argv[2])
dump = {}
for f in sorted(root.rglob("*.sqlite")):
    con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
    master = con.execute("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name").fetchall()
    tables = {}
    for (name,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        cols = [r[1] for r in con.execute(f'PRAGMA table_info("{name}")')]
        order = ", ".join(f'"{c}"' for c in cols)
        rows = con.execute(f'SELECT * FROM "{name}" ORDER BY {order}').fetchall()
        tables[name] = {"columns": cols, "rows": [[repr(v) for v in r] for r in rows]}
    con.close()
    dump[str(f.relative_to(root))] = {"master": [list(m) for m in master], "tables": tables}
out.write_text(json.dumps(dump, indent=1, sort_keys=True))
print(out, "files", len(dump), "tables", sum(len(v["tables"]) for v in dump.values()),
      "rows", sum(len(t["rows"]) for v in dump.values() for t in v["tables"].values()))
