"""depth20-tiling step 1: the cheap pass over every depth-20 call's deepest lineage.

For each of the 262 entries of census/out/deep_lines_06c.json:
1. **Reconstruct the line** from the call's x_span (census checkpoint, by hex; state/inventory.json)
   and the recorded leaf's start: at each depth the parent [a, b] is split at the driver's own
   midpoint c = (b + a) / 2.0 (AdaptiveLevin/levin_quadrature.py at e5764a6: _make_leaf's
   ``c = (b + a) / 2.0`` and _levin_leaf.break_point ``(self.end + self.start) / 2.0``), and the
   line descends into the half holding the leaf (``leaf.start >= c``: right; as worker.py's
   _deep_line). The last level must equal (leaf_start, leaf_end) bit for bit. Checked also against
   every level's start/end in the census pickle's full deep line (regions_*.pkl, record SHA-256
   checked), which also gives the halves' branch (Levin or direct).
2. **Computed end nodes** of the parent and of its halves, from the tree's own
   chebyshev_matrices(span, N): x[-1] (bottom) and x[0] (top), at the order each solve uses (the
   call's chebyshev_order 24 for a Levin solve; 2*24 - 1 = 47 for a direct (Clenshaw-Curtis)
   solve, whose grid is chebyshev_matrices(span, 2N - 1)). Also at order 24 for every solve
   ("tile24"), as the brief words it. **Tile**: left's bottom == parent's bottom, right's top ==
   parent's top, left's top == right's bottom. **Parent exact**: parent's bottom == a and top == b.
   The offsets are reported in units of ulp at the end point.
3. **residual / floor** of the parent, from the line's record (P's resid and floor, hex).

Writes state/cheap_pass.json (per level, every number in hex) and prints the totals.

Usage: PYTHONPATH=<tree>:<scripts> python -B cheap_pass.py
"""

import collections
import hashlib
import json
import math
import pickle
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dprov import CHECKPOINT, DEEP_LINES, RUN, STATE, header, hx, sha256  # noqa: E402

import AdaptiveLevin.levin_quadrature as LQ  # noqa: E402

ORDER = 24


def ends(a, b, n):
    g, _ = LQ.chebyshev_matrices((a, b), n)
    return float(g[-1]), float(g[0])


def ulps(x, ref):
    return (x - ref) / math.ulp(ref) if x != ref else 0.0


def main():
    inv_p = STATE / "inventory.json"
    for l in header(Path(__file__).resolve(), "depth20-tiling step 1: the cheap pass", [DEEP_LINES, inv_p, CHECKPOINT, Path(LQ.__file__)]):
        print(l)
    print(f"# levin_quadrature {LQ.__file__}")
    deep = json.loads(DEEP_LINES.read_text())
    inv = {(e["unit"], e["index"]): e for e in json.loads(inv_p.read_text())}
    units = {e["unit"] for e in deep}
    ck = {}
    with open(CHECKPOINT) as f:
        for line in f:
            u = line[10:60].split('"', 1)[0]
            if u in units:
                r = json.loads(line)
                ck[r["unit"]] = r["data"]
    full = {}
    for u in sorted(units):
        d = ck[u]
        with open(RUN / d["pickle"], "rb") as f:
            f.seek(d["pickle_offset"])
            blob = f.read(d["pickle_length"])
        if hashlib.sha256(blob).hexdigest() != d["pickle_record_sha256"]:
            raise SystemExit(f"{u}: pickle record sha256 differs")
        rec = pickle.loads(blob)
        for ln in rec["deep_lines"]:
            full[(u, ln["index"])] = ln
    out = []
    n_repro = n_pickle_match = 0
    for e in deep:
        key = (e["unit"], e["index"])
        iv = inv[key]
        fl = full[key]
        a, b = (float.fromhex(x) for x in iv["x_span"])
        ls, le = float.fromhex(e["leaf_start"]), float.fromhex(e["leaf_end"])
        levels = []
        pickle_match = True
        for depth, lv in enumerate(e["levels"]):
            c = (b + a) / 2.0
            fv = fl["levels"][depth]
            pickle_match &= (fv["depth"] == depth and fv["start"] == hx(a) and fv["end"] == hx(b))
            ptype, ltype, rtype = fv["P"]["type"], fv["L"]["type"], fv["R"]["type"]
            assert ptype == lv[0], (key, depth)
            order_of = {"Levin": ORDER, "direct": 2 * ORDER - 1}
            lo_P, hi_P = ends(a, b, order_of[ptype])
            lo_L, top_L = ends(a, c, order_of[ltype])
            bot_R, hi_R = ends(c, b, order_of[rtype])
            tile = (lo_L == lo_P) and (hi_R == hi_P) and (top_L == bot_R)
            p24 = ends(a, b, ORDER)
            l24 = ends(a, c, ORDER)
            r24 = ends(c, b, ORDER)
            tile24 = (l24[0] == p24[0]) and (r24[1] == p24[1]) and (l24[1] == r24[0])
            resid, floor = float.fromhex(lv[1]), float.fromhex(lv[2])
            levels.append({
                "depth": depth, "a": hx(a), "b": hx(b), "c": hx(c), "types": [ptype, ltype, rtype],
                "ends_P": [hx(lo_P), hx(hi_P)], "ends_L": [hx(lo_L), hx(top_L)], "ends_R": [hx(bot_R), hx(hi_R)],
                "ulps": {"P_lo": ulps(lo_P, a), "P_hi": ulps(hi_P, b), "L_lo": ulps(lo_L, a), "L_top": ulps(top_L, c),
                         "R_bot": ulps(bot_R, c), "R_hi": ulps(hi_R, b)},
                "tile": tile, "tile24": tile24, "parent_exact": (lo_P == a and hi_P == b),
                "resid": lv[1], "floor": lv[2], "trunc": lv[3], "e": lv[4],
                "resid_over_floor": (resid / floor) if floor > 0 else (math.inf if resid > 0 else None),
            })
            if depth == len(e["levels"]) - 1:
                break
            if ls >= c:
                a = c
            else:
                b = c
        repro = (a == ls and b == le)
        n_repro += repro
        n_pickle_match += pickle_match
        out.append({"unit": e["unit"], "index": e["index"], "notify_label": e["notify_label"], "block": iv["block"], "rtol": iv["rtol"],
                    "x_span": iv["x_span"], "leaf": [e["leaf_start"], e["leaf_end"]], "reproduces_leaf": repro,
                    "levels_match_census_pickle": pickle_match, "census_line_reproduced_its_leaf": e["line_end_equals_leaf_and_reproduces_its_residual_and_floor"],
                    "levels": levels})
    print(f"lines {len(out)}; reconstruction reproduces the recorded leaf bit for bit: {n_repro}; "
          f"every level's (start, end) equals the census pickle's full line: {n_pickle_match}")
    if n_repro != len(out):
        bad = [(o["unit"], o["index"]) for o in out if not o["reproduces_leaf"]]
        print(f"STOP: reconstruction fails on {bad}")
    p = STATE / "cheap_pass.json"
    p.write_text(json.dumps(out))
    print(f"wrote {p} sha256 {sha256(p)}")

    # ---------------------------------------------------------------- totals
    def tab(name, sel):
        c = collections.Counter()
        ratios = []
        for o in out:
            for lv in sel(o):
                big = lv["resid_over_floor"] is not None and lv["resid_over_floor"] > 10
                c[(lv["types"][0], "res>10floor" if big else "res<=10floor", "tile" if lv["tile"] else "no tile")] += 1
                c[("parent_exact", lv["parent_exact"])] += 1
                c[("tile24==tile", lv["tile24"] == lv["tile"])] += 1
                if lv["resid_over_floor"] is not None and math.isfinite(lv["resid_over_floor"]):
                    ratios.append(lv["resid_over_floor"])
        print(f"\n== {name}: {sum(v for k, v in c.items() if len(k) == 3)} levels")
        for k in sorted((k for k in c if len(k) == 3), key=str):
            print(f"   {k}: {c[k]}")
        print(f"   parent's computed ends equal a and b: {c[('parent_exact', True)]}; not: {c[('parent_exact', False)]}")
        print(f"   tile at each solve's own order equals tile at order 24 on every level: {c[('tile24==tile', False)] == 0}")
        if ratios:
            rs = sorted(ratios)
            print(f"   residual/floor: median {statistics.median(rs):.3g}, p10 {rs[len(rs)//10]:.3g}, p90 {rs[9*len(rs)//10]:.3g}, max {rs[-1]:.3g}, min {rs[0]:.3g}")
        lev = [lv for o in out for lv in sel(o) if lv["types"][0] == "Levin"]
        big = [lv for lv in lev if lv["resid_over_floor"] is not None and lv["resid_over_floor"] > 10]
        small = [lv for lv in lev if not (lv["resid_over_floor"] is not None and lv["resid_over_floor"] > 10)]
        if big:
            print(f"   Levin parents with residual > 10 floor: {len(big)}; of these not tiling {sum(not lv['tile'] for lv in big)} "
                  f"({100.0 * sum(not lv['tile'] for lv in big) / len(big):.1f} %)")
        if small:
            print(f"   Levin parents with residual <= 10 floor: {len(small)}; of these tiling {sum(lv['tile'] for lv in small)} "
                  f"({100.0 * sum(lv['tile'] for lv in small) / len(small):.1f} %)")

    tab("all levels", lambda o: o["levels"])
    tab("last five levels (depths 16-20)", lambda o: o["levels"][-5:])
    tab("depths 0-15", lambda o: o["levels"][:-5])
    for rt in ("1e-7", "1e-8"):
        tab(f"last five levels, rtol {rt}", lambda o, rt=rt: o["levels"][-5:] if o["rtol"] == rt else [])
    # per call: on the last five levels, how many Levin levels have residual > 10 floor and do not tile
    print("\n== per call, last five levels: Levin levels with residual > 10 floor that do not tile / Levin levels with residual > 10 floor")
    per = collections.Counter()
    for o in out:
        big = [lv for lv in o["levels"][-5:] if lv["types"][0] == "Levin" and lv["resid_over_floor"] is not None and lv["resid_over_floor"] > 10]
        nt = sum(not lv["tile"] for lv in big)
        per["no Levin level > 10 floor" if not big else ("all of them fail to tile" if nt == len(big) else ("none fails to tile" if nt == 0 else "some fail to tile"))] += 1
    for k, v in sorted(per.items()):
        print(f"   {k}: {v} calls")
    # by block
    print("\n== by rtol and block, last five levels: Levin residual>10floor levels (not tiling) | direct levels | median residual/floor")
    for rt in ("1e-7", "1e-8"):
        for blk in ["G+q+r", "G+q-r", "G-q+r", "G-q-r", "J1", "J2", "J3", "J4", "Y1", "Y2", "Y3", "Y4"]:
            lv5 = [lv for o in out if o["rtol"] == rt and o["block"] == blk for lv in o["levels"][-5:]]
            big = [lv for lv in lv5 if lv["types"][0] == "Levin" and lv["resid_over_floor"] is not None and lv["resid_over_floor"] > 10]
            rr = [lv["resid_over_floor"] for lv in lv5 if lv["resid_over_floor"] is not None and math.isfinite(lv["resid_over_floor"])]
            print(f"   {rt} {blk:<6} calls {sum(1 for o in out if o['rtol']==rt and o['block']==blk):>3}: {len(big):>3} ({sum(not lv['tile'] for lv in big):>3} not tiling) | "
                  f"direct {sum(lv['types'][0]=='direct' for lv in lv5):>3} | median r/f {statistics.median(rr) if rr else float('nan'):.3g}")


if __name__ == "__main__":
    main()
