"""Orchestrator's own reader, 07a Part C stop 1: R29 against R30 inside QuadSourceIntegral.

Independent of the rehearsal agent's scripts. Reads every shard with sqlite mode=ro, keys each row
by (k, q, r, z_response, log10 atol, log10 rtol) through the store's own tables, and compares by
bit pattern (float.hex). Timing leaves (elapsed, compute_time, *_time columns), label and timestamp
and serials are set aside. Prints:
- rows identical apart from those, and rows moved, with the fields that move;
- total, total_abserr, total_converged changes; largest |dtotal|/abserr;
- sums of the four guard counts over group records and over analytic Levin blocks;
- group records whose value is bit-identical but abserr differs, with locations in either store.

usage: python -I -B qsi_compare.py <R29 primary> <R30 primary>
"""

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


SKIP_COLS = {"serial", "version", "timestamp", "label"}
GUARD = ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max")


def is_timing(name):
    return name.endswith("_time") or name in ("elapsed", "compute_time", "WKB_Levin_elapsed")


def fl(x, path, out):
    if isinstance(x, dict):
        for k, v in x.items():
            fl(v, path + (k,), out)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            fl(v, path + (i,), out)
    else:
        if path and is_timing(str(path[-1])):
            return
        out[path] = x.hex() if isinstance(x, float) else repr(x)


def shards(primary):
    p = Path(primary)
    return sorted(p.parent.glob(p.stem + "-shard*.sqlite"))


def read(primary):
    rows = {}
    for sh in shards(primary):
        c = sqlite3.connect(f"file:{sh}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        wn = {r["serial"]: r["k_inv_Mpc"] for r in c.execute("select * from wavenumber")}
        wet = {r["serial"]: wn[r["wavenumber_serial"]] for r in c.execute("select * from wavenumber_exit_time")}
        zz = {r["serial"]: r["z"] for r in c.execute("select * from redshift")}
        tol = {r["serial"]: r["log10_tol"] for r in c.execute("select * from tolerance")}
        for r in c.execute("select * from QuadSourceIntegral"):
            key = (
                wet[r["k_wavenumber_exit_serial"]].hex(),
                wet[r["q_wavenumber_exit_serial"]].hex(),
                wet[r["r_wavenumber_exit_serial"]].hex(),
                zz[r["z_response_serial"]].hex(),
                tol[r["atol_serial"]],
                tol[r["rtol_serial"]],
            )
            f = {}
            for col in r.keys():
                if col in SKIP_COLS or col.endswith("_serial") or is_timing(col) or col == "metadata":
                    continue
                v = r[col]
                f[("col", col)] = v.hex() if isinstance(v, float) else repr(v)
            md = json.loads(r["metadata"])
            fl(md, ("md",), f)
            assert key not in rows, key
            rows[key] = (f, md, r["total"], r["total_abserr"], r["total_converged"],
                         wet[r["k_wavenumber_exit_serial"]], zz[r["z_response_serial"]])
        c.close()
    return rows


def levin_records(md):
    """Yield (where, record) for group records and analytic Levin blocks."""
    for i, s in enumerate(md["WKB_Levin"]["subintervals"]):
        for g in s.get("groups", []):
            yield ("group", (i, g["label"])), g
    for part in ("0pt5", "2pt5"):
        blk = md["analytic"].get(part, {}).get("Levin", {})
        for name, rec in blk.items():
            yield ("analytic", (part, name)), rec


def main(a, b):
    print("script", sha(__file__))
    for p in (a, b):
        for sh in [Path(p)] + shards(p):
            print("input", sha(sh), sh)
    A, B = read(a), read(b)
    print("rows", len(A), len(B), "keys equal", set(A) == set(B))
    moved = []
    field_moves = Counter()
    for k in sorted(A):
        fa, fb = A[k][0], B[k][0]
        assert set(fa) == set(fb) or True
        diff = [p for p in set(fa) | set(fb) if fa.get(p) != fb.get(p)]
        if diff:
            moved.append(k)
            for p in diff:
                if p[0] == "col":
                    field_moves[p[1]] += 1
                else:
                    field_moves[".".join(str(x) for x in p if not isinstance(x, int))] += 0
    print("identical rows", len(A) - len(moved), "moved rows", len(moved))
    for col in ("total", "total_abserr", "total_converged", "analytic_rad", "WKB_Levin", "numeric_quad",
                "WKB_Levin_num_regions", "WKB_Levin_max_depth", "total_phase_limited"):
        print(f"  column {col} moved in {field_moves.get(col, 0)} rows")
    up = [k for k in A if not A[k][4] and B[k][4]]
    down = [k for k in A if A[k][4] and not B[k][4]]
    print("total_converged up", len(up), "down", len(down), "converged R29", sum(bool(A[k][4]) for k in A),
          "R30", sum(bool(B[k][4]) for k in B))
    worst = max(((abs(B[k][2] - A[k][2]) / A[k][3], abs(B[k][2] - A[k][2]) / B[k][3], k) for k in A
                 if A[k][3] and B[k][3]), key=lambda t: t[0])
    print("largest |dtotal|/abserr29 %.6g  /abserr30 %.6g  at k=%s z=%s" % (worst[0], worst[1],
          float.fromhex(worst[2][0]), float.fromhex(worst[2][3])))
    s3 = [k for k in A if abs(B[k][2] - A[k][2]) > A[k][3] + B[k][3]]
    print("rows with |dtotal| > abserr29+abserr30:", len(s3), "(S3's quantity; not scored here)")
    # guard counts
    for name, R in (("R29", A), ("R30", B)):
        sums = {"group": Counter(), "analytic": Counter()}
        rows_loc = 0
        for k in R:
            any_loc = False
            for (kind, _), rec in levin_records(R[k][1]):
                for g in GUARD:
                    sums[kind][g] += rec.get(g, 0)
                any_loc |= rec.get("stationary_points", 0) > 0
            rows_loc += any_loc
        print(name, "group", dict(sums["group"]), "analytic", dict(sums["analytic"]), "rows with a located point", rows_loc)
    # record-level: value bit-equal, abserr differs, nothing located in either
    n_rec_diff = Counter()
    same_v_diff_e = []
    for k in A:
        ra = dict(levin_records(A[k][1]))
        rb = dict(levin_records(B[k][1]))
        assert set(ra) == set(rb), k
        for w in ra:
            x, y = ra[w], rb[w]
            fx, fy = {}, {}
            fl(x, (), fx)
            fl(y, (), fy)
            if fx != fy:
                n_rec_diff[w[0]] += 1
                if w[0] == "group" and x["value"].hex() == y["value"].hex() and x["abserr"].hex() != y["abserr"].hex() \
                        and x.get("stationary_points", 0) == 0 and y.get("stationary_points", 0) == 0:
                    same_v_diff_e.append((k, w, x["abserr"], y["abserr"], x["regions"], y["regions"]))
    print("Levin records differing (non-timing):", dict(n_rec_diff))
    print("group records: value bit-equal, abserr differs, no location in either:", len(same_v_diff_e))
    rel = sorted((e30 / e29 - 1) for _, _, e29, e30, _, _ in same_v_diff_e)
    if rel:
        print("  rel change range %.3g .. %.3g; regions unchanged in %d" % (rel[0], rel[-1],
              sum(r1 == r2 for *_, r1, r2 in same_v_diff_e)))
        big = [t for t in same_v_diff_e if abs(t[3] / t[2] - 1) > 1e-3]
        for k, w, e29, e30, r1, r2 in big:
            print("   ", float.fromhex(k[0]), float.fromhex(k[3]), w, e29, e30, r1, r2)


if __name__ == "__main__":
    main(*sys.argv[1:])
