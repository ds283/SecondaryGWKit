"""levin-phase-floor prompt 07b extension, section 2.7.1 (X3): (f)'s chase. New in the extension.

For each of (f)'s unconverged calls, from its replay under the instrument ``fdiag`` (out/ext/replays/fdiag/, checked
equal to Part A's (f) replay: regions, value, abserr and every leaf) and its instrument records:
- **the leaves at depth 20** (the call's depth_max): route -- Levin, the C2 gate's fallback (a direct leaf, not a window
  piece), a window piece (or its descendant), a guarded leaf; trusted, untrusted or floor-limited;
- **descent:** whether each lies inside a Levin leaf (f)'s rule untrusted (an ``f`` record with ``trusted`` false);
- **the window stop:** whether 07a's iroff1 rule could count it (only a window leaf's bisections count);
- **honesty with converged set aside:** |v - ref|/abserr from Part A's scores (out/tables/scores_final.json, ``f``);
- **the cause, from the bisections:** every bisection of a direct (fallback) leaf inside an (f)-untrusted span, by why it
  was bisected and not stopped: untrusted with its residual above its floor (the round-off exit not considered);
  untrusted with the exit considered and refused, by the halving test (halves' roundoff floors below half its own)
  or, the halving test holding, by the budget clause; or trusted and bisected for the budget.
Writes OUT_JSON and OUT_MD. Usage: python -B x3_chase.py OUT_JSON OUT_MD
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEPTH_MAX = 20


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(h):
    return None if h is None else (float.fromhex(h) if isinstance(h, str) else float(h))


def main():
    out_j, out_m = Path(sys.argv[1]), Path(sys.argv[2])
    for p in (out_j, out_m):
        if p.exists():
            raise SystemExit(f"{p} exists")
    scores = ROOT / "out" / "tables" / "scores_final.json"
    table = json.loads(scores.read_text())["table"]
    rows, tot = [], Counter()
    for p in sorted((ROOT / "out" / "ext" / "replays" / "fdiag").glob("*.json")):
        if p.name.endswith(".instrument.json"):
            continue
        d = json.loads(p.read_text())
        sid = d["sid"]
        fa = json.loads((ROOT / "out" / "replays" / "f" / f"{sid}.json").read_text())
        same = fa["regions_sha256"] == d["regions_sha256"] and fa["replay"] == d["replay"] and fa["leaves"] == d["leaves"]
        recs = json.loads((p.parent / f"{sid}.instrument.json").read_text())["records"]
        recs = [{k: (fx(v) if isinstance(v, str) and v.startswith(("0x", "-0x", "inf", "-inf", "nan")) else v)
                 for k, v in r.items()} for r in recs]
        untr = [(min(r["start"], r["end"]), max(r["start"], r["end"])) for r in recs if r["kind"] == "f" and not r["trusted"]]

        def inside(a, b):
            lo, hi = min(a, b), max(a, b)
            return any(u0 <= lo and hi <= u1 for u0, u1 in untr)

        deep = []
        for L in d["leaves"]:
            if L["depth"] < DEPTH_MAX:
                continue
            a, b = fx(L["start"]), fx(L["end"])
            route = ("window" if L["window"] else "guarded" if L["guarded"] else "levin" if L["type"] == "Levin" else "fallback")
            state = "floor_limited" if L["phase_limited"] else ("trusted" if L["trusted"] else ("untrusted" if L["trusted"] is False else "unchecked"))
            deep.append({"route": route, "state": state, "from_f_untrusted": inside(a, b), "window_counts": bool(L["window"])})
        # why the chase's bisections were made: direct leaves inside an (f)-untrusted span
        why = Counter()
        exits = {(r["start"], r["end"], r["depth"]): r for r in recs if r["kind"] == "exit"}
        for r in recs:
            if r["kind"] != "bisect" or not r["direct"] or r["window"] or not inside(r["start"], r["end"]):
                continue
            ex = exits.get((r["start"], r["end"], r["depth"]))
            if r["trusted"] is False:
                if ex is None:
                    why["untrusted, residual above floor"] += 1
                elif ex["halves_floor_roundoff"] is not None and ex["halves_floor_roundoff"] < 0.5 * ex["floor_roundoff"]:
                    why["untrusted, exit refused by the halving test"] += 1
                else:
                    why["untrusted, exit refused by the budget clause"] += 1
            elif r["trusted"]:
                why["trusted, bisected for the budget"] += 1
            else:
                why["unchecked, bisected for the budget"] += 1
        n_f_untr = len(untr)
        sc = table.get(sid, {}).get("f", {})
        sc_a = table.get(sid, {}).get("tree", {})
        # where the (f)-untrusted leaves are: touching a window edge or not
        wins = d.get("windows") or []
        edges = {fx(w["lo"]) for w in wins} | {fx(w["hi"]) for w in wins}
        untr_touch = sum(1 for u0, u1 in untr if u0 in edges or u1 in edges)
        dc = Counter((x["route"], x["state"], x["from_f_untrusted"]) for x in deep)
        row = {"sid": sid, "set": d["set"], "same_as_f": same, "regions": d["replay"]["num_regions"],
               "ratio_f": sc.get("ratio"), "scored_f": (sc.get("spread_over_abserr") or 1) <= 0.01 if sc else None,
               "ratio_a": sc_a.get("ratio"), "converged_a": sc_a.get("converged"),
               "n_f_untrusted": n_f_untr, "n_f_untrusted_touching_window": untr_touch,
               "deep": {f"{k[0]}|{k[1]}|{'from_f' if k[2] else 'not_from_f'}": v for k, v in dc.items()},
               "n_deep": len(deep), "n_deep_window_countable": sum(1 for x in deep if x["window_counts"]),
               "chase_bisections": dict(why)}
        rows.append(row)
        tot.update({f"deep {k}": v for k, v in row["deep"].items()})
        tot.update({f"bisect {k}": v for k, v in why.items()})
        tot["calls"] += 1
        tot["same_as_f"] += int(same)
    doc = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "scores_sha256": sha256(scores)},
           "rows": rows, "totals": dict(tot)}
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps(doc, indent=1))
    md = ["## (X3) (f)'s unconverged calls: the leaves at depth 20 and the chase", "",
          "Totals: " + "; ".join(f"{k}: {v}" for k, v in sorted(tot.items())), "",
          "| call | set | (f) ratio (scored) | (a) ratio | f-untrusted leaves (touching a window) | depth-20 leaves: route, state, descent | window stop could count | chase bisections |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['sid']} | {r['set']} | {r['ratio_f']:.3g}{'' if r['scored_f'] else ' n'} | "
                  f"{(r['ratio_a'] or 0):.3g}{'' if r['converged_a'] else 'u'} | {r['n_f_untrusted']} ({r['n_f_untrusted_touching_window']}) | "
                  + "; ".join(f"{k} {v}" for k, v in sorted(r["deep"].items())) + f" | {r['n_deep_window_countable']} of {r['n_deep']} | "
                  + "; ".join(f"{k} {v}" for k, v in sorted(r["chase_bisections"].items())) + " |")
    out_m.write_text("\n".join(md) + "\n")
    print("\n".join(md[:3]))
    print("wrote", out_j, sha256(out_j), out_m, sha256(out_m))


if __name__ == "__main__":
    main()
