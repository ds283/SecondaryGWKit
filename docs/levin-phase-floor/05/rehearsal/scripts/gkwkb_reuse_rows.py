"""The GkWKBIntegration rows whose metadata differs between R9 and R10, read from the stores.

store_value_diff's JSON names the metadata keys that differ (rho_evals, rho_reused) but not their
values. This reads them, every store file opened sqlite `mode=ro`. Rows are paired by `label` with
its trailing timestamp removed (`<job>-GkWKB-k<k>-sourcez<z>-zinit<z>`): the pairing used here is
cross-checked against store_value_diff's identity-matched list, by count and by label, and is not
a substitute for it.

    ./venv/bin/python var/levin-phase-floor-05/rehearsal/scripts/gkwkb_reuse_rows.py \
        <R9 primary> <R10 primary> <store_value_diff.json>
"""

import json
import re
import sqlite3
import sys
from pathlib import Path

STAMP = re.compile(r"-\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$")


def rows(primary):
    primary = Path(primary)
    con = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    shards = [r[0] for r in con.execute("SELECT filename FROM shards ORDER BY serial")]
    con.close()
    out = {}
    for name in shards:
        con = sqlite3.connect(f"file:{primary.parent / Path(name).name}?mode=ro", uri=True)
        for label, meta, rhs, steps in con.execute(
            "SELECT label, metadata, stage_1_RHS_evaluations, stage_1_compute_steps "
            "FROM GkWKBIntegration"
        ):
            key = STAMP.sub("", label)
            assert key not in out, key
            out[key] = (json.loads(meta) if meta else None, rhs, steps)
        con.close()
    return out


def main():
    a, b, report = sys.argv[1:4]
    ra, rb = rows(a), rows(b)
    print(f"GkWKBIntegration rows: A {len(ra)}, B {len(rb)}; same label set: {set(ra) == set(rb)}")
    diff = json.load(open(report))["classes"]["GkWKBIntegration"]["rows"]["rows"]
    flagged = sorted(
        STAMP.sub("", d["columns"]["label"]["a"])
        for d in diff
        if "metadata" in d["columns"]
    )
    mine = sorted(k for k in ra if ra[k][0] != rb[k][0])
    print(f"metadata differs: store_value_diff {len(flagged)}, this script {len(mine)}; "
          f"same rows: {flagged == mine}")
    print()
    print("label (timestamp removed) | A rho_reused rho_evals RHS_evals | B rho_reused rho_evals "
          "RHS_evals | other metadata keys differing")
    by_k = {}
    for k in mine:
        ma, rhs_a, _ = ra[k]
        mb, rhs_b, _ = rb[k]
        other = sorted(x for x in set(ma) | set(mb) if x not in ("rho_reused", "rho_evals")
                       and ma.get(x) != mb.get(x))
        print(f"{k} | {ma.get('rho_reused')} {ma.get('rho_evals')} {rhs_a} | "
              f"{mb.get('rho_reused')} {mb.get('rho_evals')} {rhs_b} | {other or '-'}")
        kk = k.split("-GkWKB-")[1].split("-")[0]
        by_k.setdefault(kk, [0, 0, 0])
        by_k[kk][0] += 1
        by_k[kk][1] += int(bool(ma.get("rho_reused")) and not mb.get("rho_reused"))
        by_k[kk][2] += int(bool(mb.get("rho_reused")) and not ma.get("rho_reused"))
    print()
    print("by k: rows differing, reused in A only, reused in B only")
    for kk, v in sorted(by_k.items()):
        print(f"  {kk}: {v}")
    for side, rr in (("A", ra), ("B", rb)):
        n = sum(1 for v in rr.values() if v[0] and not v[0].get("rho_reused"))
        print(f"{side}: rows with rho_reused false: {n} of {len(rr)}")


if __name__ == "__main__":
    main()
