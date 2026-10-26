"""Why GkSource.parents_digest differs between R9 and R10: the serials it digests.

`parents_digest` is the SHA-256 (hex) of the compact JSON of the GkSource's parent set. The set is
the sorted list of (z_source serial, GkNumericIntegration serial or None, GkWKBIntegration serial
or None) triples (`ObjectFactories/GkSource.py` `GkSource_parents_digest`;
`ComputeTargets/GkSource.py` `GkSource_parent_set`). This script compares, between the two stores,
the serials of the rows those triples name:
- redshift serials, paired by `z`;
- GkNumericIntegration and GkWKBIntegration serials, paired by `label` with its trailing
  `datetime.now()` stamp removed.

It then recomputes each store's digests from its own `GkSource_parents` rows, to confirm the stored
column is that function of serials. Every file is opened sqlite `mode=ro`.

    ./venv/bin/python parent_serials.py <R9 primary> <R10 primary>   (PYTHONPATH=. for the import)
"""

import re
import sqlite3
import sys
from pathlib import Path

from ObjectFactories.GkSource import GkSource_parents_digest

STAMP = re.compile(r"-\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")


def shards(primary):
    primary = Path(primary)
    con = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    names = [r[0] for r in con.execute("SELECT filename FROM shards ORDER BY serial")]
    con.close()
    return [primary.parent / Path(n).name for n in names]


def read(primary):
    z, num, wkb, recomputed, total = {}, {}, {}, 0, 0
    for f in shards(primary):
        con = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
        for s, zz in con.execute("SELECT serial, z FROM redshift"):
            z.setdefault(zz, s)
        for table, out in (("GkNumericIntegration", num), ("GkWKBIntegration", wkb)):
            for s, label in con.execute(f"SELECT serial, label FROM {table}"):
                out[STAMP.sub("", label)] = s
        for serial, digest in con.execute("SELECT serial, parents_digest FROM GkSource"):
            members = con.execute(
                "SELECT z_source_serial, numeric_serial, wkb_serial FROM GkSource_parents "
                "WHERE gk_source_serial = ?", (serial,)).fetchall()
            total += 1
            recomputed += GkSource_parents_digest(members) == digest
        con.close()
    return z, num, wkb, recomputed, total


def main():
    a, b = read(sys.argv[1]), read(sys.argv[2])
    for name, i in (("redshift (by z)", 0), ("GkNumericIntegration (by label)", 1),
                    ("GkWKBIntegration (by label)", 2)):
        da, db = a[i], b[i]
        same_keys = set(da) == set(db)
        differ = sum(1 for k in da if k in db and da[k] != db[k])
        print(f"{name}: A {len(da)}, B {len(db)}, same keys {same_keys}; serial differs in "
              f"{differ} of {len(da)}")
    for side, r in (("A", a), ("B", b)):
        print(f"{side}: stored parents_digest equals GkSource_parents_digest(its GkSource_parents "
              f"rows) for {r[3]} of {r[4]} GkSource rows")


if __name__ == "__main__":
    main()
