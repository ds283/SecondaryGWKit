"""levin-phase-floor prompt 07a section 6 (new in 07a; read only): W7's violating calls (reading A, as ruled), in detail:
each call's label, located points by kind, its windows at h and at 07 (iroff1 count, stopped, leaves falling at the
stop, pieces), regions, depth, value and abserr at both, and |v_h - v_07| against abserr_07 and abserr_h; and the same
counts over every W7-covered call that moved. Usage: python -B w7_detail_07a.py OUT_JSON
"""
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

UNITS = WIDER / "state" / "score_07a_units.json"
SUMM = WIDER / "out" / "scores" / "summary.json"
CALLS = WIDER / "out" / "stop3" / "compare_census_calls.calls.jsonl"


def F(h):
    return Fraction(float.fromhex(h))


def main():
    out_p = Path(sys.argv[1])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: W7's violating calls in detail", [UNITS, SUMM, CALLS])
    U = {u["unit"]: u for u in json.loads(UNITS.read_text())["units"]}
    S = json.loads(SUMM.read_text())
    moved = []
    for line in open(CALLS):
        c = json.loads(line)
        r = c["route"] or {}
        if c["class"] != "window_both" or r.get("n_route_differs") or r.get("n_lu_differs") or (c["skipped_07"] and not c["skipped_h"]):
            continue
        if c["value"][0] == c["value"][1]:
            continue
        sc = next(x for x in U[c["unit"]]["calls"] if x["kind"] == c["kind"] and x["index"] == c["index"])
        vh, v7, eh, e7 = F(c["value"][0]), F(c["value"][1]), F(c["abserr"][0]), F(c["abserr"][1])
        moved.append({"unit": c["unit"], "rtol": c["rtol"], "kind": c["kind"], "index": c["index"], "label": c["label"],
                      "band": U[c["unit"]]["band"], "k": U[c["unit"]]["k"],
                      "move_over_abserr07": float(abs(vh - v7) / e7), "move_over_abserr_h": float(abs(vh - v7) / eh),
                      "abserr_h_over_07": float(eh / e7), "converged": c["converged"], "regions": c["num_regions"],
                      "max_depth": c["max_depth"], "points": dict(Counter(p["kind"] for p in sc.get("points") or [])),
                      "windows_h": [[w["iroff1_count"], w["stopped"], w["leaves_falling_at_stop"], w["n_pieces"]] for w in sc.get("windows") or []],
                      "windows_07": [[w["iroff1_count"], w["stopped"], w["leaves_falling_at_stop"], w["n_pieces"]] for w in sc.get("windows_07") or []],
                      "n_windows_stopped_h_07": sc.get("n_windows_stopped")})
    viol = [m for m in moved if m["move_over_abserr07"] > 1e-5]
    doc = {"header": hdr, "n_moved": len(moved), "n_violating": len(viol),
           "violating_by": dict(Counter(f"{m['rtol']}|{m['kind']}|{m['label'].split('group ')[-1] if 'group' in m['label'] else m['label']}|points={m['points']}" for m in viol)),
           "violating_by_band_k": dict(Counter(f"{m['band']}|{m['k']}" for m in viol)),
           "moved_by": dict(Counter(f"{m['rtol']}|{m['label'].split('group ')[-1] if 'group' in m['label'] else m['label']}|{m['move_over_abserr07'] > 1e-5}" for m in moved)),
           "violating": viol, "moved": moved}
    out_p.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("header", "moved", "violating")}, indent=1))
    for m in viol:
        print(m["unit"], m["index"], m["label"][-40:], f"{m['move_over_abserr07']:.3g} {m['move_over_abserr_h']:.3g} e_h/e_07 {m['abserr_h_over_07']:.4g}",
              m["regions"], m["max_depth"], m["points"], "h", m["windows_h"][:3], "07", m["windows_07"][:3], m["n_windows_stopped_h_07"])
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
