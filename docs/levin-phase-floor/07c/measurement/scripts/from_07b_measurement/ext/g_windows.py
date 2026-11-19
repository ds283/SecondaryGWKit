"""levin-phase-floor prompt 07b extension, section 2.7.2 (g): the windows (g) reached on the census calls, and the no-window
check for every new variant. New in the extension.

From the instrument records of the extension's replays (out/ext/replays/<v>/<sid>.instrument.json, ``kind`` "g"): for
each window of each call under g1, g2 and g2h, the half-width reached (2, 4 or 8 pi) and, at each final edge with a Levin
seed leaf beside it, the rate ratio across the first spacing, spacing/l and whether the edge test still fails (at the
cap). And, for every new variant, whether the five no-window calls n0-n4 are bit-identical to (a) (regions_sha256,
value and abserr).
Writes OUT_JSON and OUT_MD. Usage: python -B g_windows.py OUT_JSON OUT_MD
"""

import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
NEW = ("g1", "g2", "h", "g2h", "i", "i2")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fx(v):
    return float.fromhex(v) if isinstance(v, str) else v


def main():
    out_j, out_m = Path(sys.argv[1]), Path(sys.argv[2])
    for p in (out_j, out_m):
        if p.exists():
            raise SystemExit(f"{p} exists")
    res = {"windows": {}, "nowindow": {}}
    md = ["## (g) The windows reached on the census calls", ""]
    for v in ("g1", "g2", "g2h"):
        halves, edges, calls = Counter(), [], 0
        per_set = Counter()
        for p in sorted((ROOT / "out" / "ext" / "replays" / v).glob("*.instrument.json")):
            recs = [r for r in json.loads(p.read_text())["records"] if r.get("kind") == "g"]
            sid = p.name[: -len(".instrument.json")]
            for r in recs:
                calls += 1
                widened = False
                for w in r["windows"]:
                    h = round(fx(w["half_max"]) / math.pi)
                    halves[h] += 1
                    widened |= h > 2
                if widened:
                    per_set[json.loads((p.parent / f"{sid}.json").read_text())["set"]] += 1
                for e in r["edges"]:
                    if e["levin"]:
                        edges.append({"sid": sid, "ratio": fx(e["rate_ratio"]), "s_over_l": fx(e["spacing_over_l"]),
                                      "fails": e["fails"]})
        fails = [e for e in edges if e["fails"]]
        res["windows"][v] = {"calls_with_windows": calls, "windows_by_half_width_pi": dict(halves),
                             "calls_widened_by_set": dict(per_set), "levin_edges": len(edges),
                             "edges_failing_at_cap": len(fails), "calls_with_failing_edge": sorted({e["sid"] for e in fails}),
                             "spacing_over_l_passing_max": max((e["s_over_l"] for e in edges if not e["fails"]), default=None),
                             "spacing_over_l_failing_range": [min((e["s_over_l"] for e in fails), default=None),
                                                              max((e["s_over_l"] for e in fails), default=None)],
                             "rate_ratio_failing_range": [min((e["ratio"] for e in fails), default=None),
                                                          max((e["ratio"] for e in fails), default=None)]}
        x = res["windows"][v]
        md += ([f"- **{v}:** {calls} calls with windows; windows by half-width reached (pi): {dict(sorted(halves.items()))}; "
               f"calls widened by set: {dict(per_set)}; {len(edges)} final edges with a Levin seed leaf beside them, "
               f"{len(fails)} still failing at the cap (in {len(x['calls_with_failing_edge'])} calls: "
               f"{', '.join(x['calls_with_failing_edge'])}); spacing/l at a passing edge at most {x['spacing_over_l_passing_max']:.3g}; "
               f"at a failing edge {x['spacing_over_l_failing_range'][0]:.3g}-{x['spacing_over_l_failing_range'][1]:.3g} "
               f"(rate ratio {x['rate_ratio_failing_range'][0]:.3g}-{x['rate_ratio_failing_range'][1]:.3g})."]
               if fails else [f"- **{v}:** {calls} calls with windows; windows by half-width (pi): {dict(sorted(halves.items()))}; none failing."])
    md += ["", "## The no-window calls n0-n4 against (a) (regions_sha256, value, abserr)", "",
           "| variant | " + " | ".join(f"n{k}" for k in range(5)) + " |", "|---|" + "---|" * 5]
    for v in NEW:
        cells = []
        for k in range(5):
            a = json.loads((ROOT / "out" / "replays" / "tree" / f"n{k}.json").read_text())
            b = json.loads((ROOT / "out" / "ext" / "replays" / v / f"n{k}.json").read_text())
            same = a["regions_sha256"] == b["regions_sha256"] and a["replay"] == b["replay"]
            cells.append("identical" if same else "differs")
        res["nowindow"][v] = cells
        md.append(f"| {v} | " + " | ".join(cells) + " |")
    out_j.parent.mkdir(parents=True, exist_ok=True)
    out_j.write_text(json.dumps({"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__)}} | res, indent=1))
    out_m.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", out_j, sha256(out_j), out_m, sha256(out_m))


if __name__ == "__main__":
    main()
