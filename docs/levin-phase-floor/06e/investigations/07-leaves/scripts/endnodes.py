"""Q3 follow-up: does a Levin leaf's step-(4) residual above its eq. (151) floor coincide with a computed
end node off the bisection point? For every history row of out/q3_*.json (a Levin leaf whose halves
were solved): whether the computed end nodes of the parent and its halves tile -- the left half's bottom node
equals the parent's, the right half's top node equals the parent's, and the left half's top node
equals the right half's bottom node -- so that any end-node offset is common to both sides of the
step-(4) comparison; and whether the parent's own end nodes equal a and b. Rows are
split into residual > 10 floor and the rest, and by whether the halves are Levin or CC solves, after the leaf's own resolution has converged
(|V_P err| < 1e-15), and counted. Writes out/endnodes.json."""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lc  # noqa: E402

lq = lc.lq


def ends(a, b):
    g, _ = lq.chebyshev_matrices((a, b), lc.ORDER)
    return float(g[-1]), float(g[0])


def main():
    out = {"header": lc.header(__file__), "files": {}}
    for p in sorted(glob.glob(str(lc.OUT / "q3_*.json"))):
        d = json.load(open(p))
        tally = {}
        seen = set()
        for cfg, c in d["configs"].items():
            for h in c["history"]:
                a, b = h["a"], h["b"]
                if (a, b) in seen or abs(h["P_error"]) >= 1e-15:
                    continue
                seen.add((a, b))
                m = (b + a) / 2.0
                lo_L, top_L = ends(a, m)
                bot_R, hi_R = ends(m, b)
                lo_P, hi_P = ends(a, b)
                # the computed ranges tile: the halves' computed end nodes reproduce the parent's
                # and meet each other, so an end-node offset is common to both sides of step (4)
                tile = (lo_L == lo_P) and (hi_R == hi_P) and (top_L == bot_R)
                key = ("residual>10floor" if h["residual"] > 10 * h["floor"] else "residual<=10floor") + (
                    " | computed end nodes tile" if tile else " | computed end nodes do not tile"
                ) + (" | halves CC" if any(h["halves_direct"]) else " | halves Levin")
                t = tally.setdefault(key, {"n": 0, "max_residual": 0.0, "parent_ends_off": 0})
                t["n"] += 1
                t["max_residual"] = max(t["max_residual"], h["residual"])
                t["parent_ends_off"] += int(lo_P != a or hi_P != b)
        out["files"][Path(p).name] = tally
        print(Path(p).name)
        for k, v in sorted(tally.items()):
            print(f"   {k}: {v}")
    (lc.OUT / "endnodes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
