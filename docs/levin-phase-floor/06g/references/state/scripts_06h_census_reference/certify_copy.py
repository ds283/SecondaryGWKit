"""levin-phase-floor prompt 06 section 6, route (B): certify the scratch copy (make_copy.py) against
A3 v2, both opened sqlite3 mode=ro. For the primary and each shard:
- the schema: sqlite_master (type, name, tbl_name, sql) of every object identical, except that
  the CREATE TABLE sql of GkWKBValue, TkWKBValue and GkSourceValue on a shard, which with the one
  ", theta_residual FLOAT" that ALTER TABLE wrote taken out must be A3 v2's exactly; PRAGMA table_info of every table identical except that one trailing column;
- the rows: every row of every table identical, as A3 v2's columns read in rowid order (or in
  the table's own order where it has no rowid), each value compared by type and, for a float, by
  bit pattern (struct pack), never by ``==`` alone;
- the added column is NULL in every row.
Prints a line per table and a verdict."""
import sqlite3
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import STORE_DIR, WIDER, header  # noqa: E402

COPY_DIR = WIDER / "state" / "a3v2_copy"
NAMES = ["handover-A3-v2-lambdacdm.sqlite"] + [
    f"handover-A3-v2-lambdacdm-shard{i:04d}.sqlite" for i in range(4)
]
ADDED = {"GkWKBValue", "TkWKBValue", "GkSourceValue"}


def ro(p):
    return sqlite3.connect(f"{p.as_uri()}?mode=ro", uri=True)


def key(v):
    if isinstance(v, float):
        return ("f", struct.pack("<d", v))
    return (type(v).__name__, v)


def main():
    for l in header(Path(__file__).resolve(), "route (B): the scratch copy of A3 v2, certified"):
        print(l)
    ok = True
    total_rows = 0
    for name in NAMES:
        a, b = ro(STORE_DIR / name), ro(COPY_DIR / name)
        ma = sorted(a.execute("select type, name, tbl_name, sql from sqlite_master").fetchall(), key=lambda r: (r[0], r[1]))
        mb = sorted(b.execute("select type, name, tbl_name, sql from sqlite_master").fetchall(), key=lambda r: (r[0], r[1]))
        if [r[:3] for r in ma] != [r[:3] for r in mb]:
            ok = False
            print(f"!! {name}: sqlite_master objects differ")
        for ra, rb in zip(ma, mb):
            if ra[3] != rb[3]:
                # SQLite's ALTER TABLE ADD COLUMN writes ", theta_residual FLOAT" after the last
                # column definition, before the table constraints; with that one occurrence taken
                # out, the copy's sql must be A3 v2's exactly
                if (
                    ra[0] == "table"
                    and ra[1] in ADDED
                    and rb[3].count(", theta_residual FLOAT") == 1
                    and rb[3].replace(", theta_residual FLOAT", "", 1) == ra[3]
                ):
                    print(f"   {name}: {ra[1]} CREATE sql differs only by the added column ', theta_residual FLOAT'")
                else:
                    ok = False
                    print(f"!! {name}: sql of {ra[1]} differs:\n   {ra[3]!r}\n   {rb[3]!r}")
        tables = [r[1] for r in ma if r[0] == "table" and not r[1].startswith("sqlite_")]
        for t in tables:
            ia = a.execute(f'PRAGMA table_info("{t}")').fetchall()
            ib = b.execute(f'PRAGMA table_info("{t}")').fetchall()
            if t in ADDED:
                extra = ib[len(ia):]
                good = ib[: len(ia)] == ia and len(extra) == 1 and extra[0][1] == "theta_residual" and extra[0][2] == "FLOAT" and extra[0][3] == 0
            else:
                good = ia == ib
            if not good:
                ok = False
                print(f"!! {name}: {t} table_info differs: {ia} vs {ib}")
            cols = ", ".join(f'"{c[1]}"' for c in ia)
            wr = a.execute("select sql from sqlite_master where name=?", (t,)).fetchone()[0]
            order = "" if "WITHOUT ROWID" in (wr or "").upper() else " ORDER BY rowid"
            ca = a.execute(f'SELECT {cols} FROM "{t}"{order}')
            cb = b.execute(f'SELECT {cols} FROM "{t}"{order}')
            n = 0
            diff = 0
            while True:
                xa = ca.fetchmany(20000)
                xb = cb.fetchmany(20000)
                if len(xa) != len(xb):
                    ok = False
                    print(f"!! {name}: {t} row counts differ")
                    break
                if not xa:
                    break
                for p, q in zip(xa, xb):
                    if tuple(map(key, p)) != tuple(map(key, q)):
                        diff += 1
                n += len(xa)
            nulls = ""
            if t in ADDED:
                nn = b.execute(f'SELECT COUNT(*) FROM "{t}" WHERE theta_residual IS NOT NULL').fetchone()[0]
                nulls = f"; theta_residual non-NULL rows {nn}"
                if nn != 0:
                    ok = False
            total_rows += n
            if diff:
                ok = False
            print(f"   {name}: {t}: {n} rows, {diff} differing{nulls}")
        a.close()
        b.close()
    print(f"== {total_rows} rows compared")
    print(f"== the copy is A3 v2 but for the three added, all-NULL columns: {'CERTIFIED' if ok else 'NOT CERTIFIED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
