"""levin-phase-floor prompt 06k, Part C ("Moves"): the leaves that carry each S1 move. New in 06k (after 06j's
w_prep.py constant_move). Read only.

For every call outside the 720 whose value moves by more than e_route + e_g (out/partC/<route>.json), both codes'
returned leaves are read from their region pickles (each record's SHA-256 checked): start, end, depth, value
(the value each leaf returns), V_P, value kind, trusted flag, abserr and e'. The cells of the coarsest common
refinement are formed (a cell is a leaf of one code that holds whole leaves of the other); per cell the two codes'
partial sums are differenced exactly, and the cell carrying the largest share of the move is reported with its
leaves on both sides. For route 3 the left leaves (the route record) are marked.
Writes out/partC/movers_<route>.json and prints one line per mover.
Usage: python -B movers_c.py ROUTE
"""

import hashlib
import json
import pickle
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

EX = RC.ROOT / "state" / "extract"


def blob(run, ptr):
    name, off, ln, sha = ptr
    with open(Path(run) / name, "rb") as f:
        f.seek(off)
        b = f.read(ln)
    if hashlib.sha256(b).hexdigest() != sha:
        raise SystemExit(f"{name}@{off}: sha256 differs")
    import ray.cloudpickle as cp

    return cp.loads(b)


def entry(rec, kind, index):
    key = "regions" if kind == "group" else "other_regions"
    hits = [e for e in rec[key] if e["index"] == index]
    return hits[0]


def leaves(e):
    n = len(e["start"])
    out = []
    for i in range(n):
        out.append({"start": float(e["start"][i]), "end": float(e["end"][i]), "depth": int(e["depth"][i]),
                    "type": int(e["type"][i]), "value": float(e["value"][i]), "value_own": float(e["value_own"][i]),
                    "value_kind": int(e["value_kind"][i]), "abserr": float(e["abserr"][i]),
                    "budget_err": float(e["budget_err"][i]),
                    "trusted": int(e["trusted"][i]), "r": float(e["convergence_ratio"][i]), "delta1": float(e["delta1"][i]),
                    "declared_resolution": float(e["declared_resolution"][i]), "phase_err": float(e["phase_err"][i])})
    return out


def cells(A, B):
    """Coarsest common refinement: walk both ascending partitions; a cell closes where both have an end."""
    i = j = 0
    out = []
    while i < len(A) and j < len(B):
        ca, cb = [A[i]], [B[j]]
        ea, eb = A[i]["end"], B[j]["end"]
        while ea != eb:
            if ea < eb:
                i += 1
                ca.append(A[i])
                ea = A[i]["end"]
            else:
                j += 1
                cb.append(B[j])
                eb = B[j]["end"]
        out.append((ca, cb))
        i += 1
        j += 1
    return out


def main():
    route = sys.argv[1]
    res = json.loads((RC.ROOT / "out" / "partC" / f"{route}.json").read_text())
    movers = [m for m in res["all_moves"] if not m["in_720"]]
    with open(EX / "g.pkl", "rb") as f:
        G = pickle.load(f)
    with open(EX / f"{route}.pkl", "rb") as f:
        R = pickle.load(f)
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 06k: the leaves carrying {route}'s S1 moves",
                 [RC.ROOT / "out" / "partC" / f"{route}.json", EX / "g.pkl", EX / f"{route}.pkl"])
    for l in hdr:
        print(l)
    out = []
    for m in sorted(movers, key=lambda m: -m["ratio"]):
        ug, ur = G["units"][m["unit"]], R["units"][m["unit"]]
        gr = entry(blob(G["run"], ug["pickle"]), m["kind"], m["index"])
        rr = entry(blob(R["run"], ur["pickle"]), m["kind"], m["index"])
        A, B = leaves(gr), leaves(rr)
        call_r = [c for c in ur["calls"] if c["kind"] == m["kind"] and c["index"] == m["index"]][0]
        cs = cells(A, B)
        total = sum((Fraction(x["value"]) for x in B), Fraction(0)) - sum((Fraction(x["value"]) for x in A), Fraction(0))
        rows = []
        for ca, cb in cs:
            d = sum((Fraction(x["value"]) for x in cb), Fraction(0)) - sum((Fraction(x["value"]) for x in ca), Fraction(0))
            rows.append((abs(d), d, ca, cb))
        rows.sort(key=lambda t: -t[0])
        big = rows[0]
        rec = {k: m[k] for k in ("unit", "rtol", "row", "kind", "index", "group", "ratio", "dv", "abserr", "regions", "depth", "converged", "n_firings")}
        rec.update({"n_cells": len(cs), "move_exact": float(total),
                    "carrying_cell": {"start": big[2][0]["start"], "end": big[2][-1]["end"], "diff": float(big[1]),
                                      "share_of_move": float(big[1] / total) if total else None,
                                      "g_leaves": big[2], "route_leaves": big[3]},
                    "second_cell_share": float(rows[1][1] / total) if len(rows) > 1 and total else None})
        out.append(rec)
        cc = rec["carrying_cell"]
        print(f"{m['row']} {m['rtol']} {m['kind']} #{m['index']} {m['group']}: ratio {m['ratio']:.3f}; regions {m['regions']}; "
              f"cells {len(cs)}; carrying cell [{cc['start']:.6g}, {cc['end']:.6g}] share {cc['share_of_move']:.4f}; "
              f"g leaves {len(cc['g_leaves'])} (depths {sorted({x['depth'] for x in cc['g_leaves']})}), route leaves {len(cc['route_leaves'])} "
              f"(depths {sorted({x['depth'] for x in cc['route_leaves']})}, kinds {[x['value_kind'] for x in cc['route_leaves']]})")
    p = RC.ROOT / "out" / "partC" / f"movers_{route}.json"
    p.write_text(json.dumps({"header": hdr, "route": route, "n": len(out), "movers": out}, indent=1, default=str))
    print(f"wrote {p} sha256 {RC.sha256(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
