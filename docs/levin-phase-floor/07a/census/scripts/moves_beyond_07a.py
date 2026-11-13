"""levin-phase-floor prompt 07a section 6 (descriptive; new in 07a; read only): every call whose value moved from 07's by
more than an abserr (|v_h - v_07| > abserr_07, and whether also > abserr_h, and > max of the two), by class, with its
route changes; the classes of V4's 33 calls; and the aux_only calls whose abserr shrank by more than 2x, listed.
Usage: python -B moves_beyond_07a.py OUT_JSON
"""
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

CALLS = WIDER / "out" / "stop3" / "compare_census_calls.calls.jsonl"
W4 = WIDER / "out" / "scores" / "w4_scores.json"


def F(h):
    return Fraction(float.fromhex(h))


def main():
    out_p = Path(sys.argv[1])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: calls moved beyond an abserr; V4's classes", [CALLS, W4])
    w4 = json.loads(W4.read_text())
    want = {("h-" + x["unit"].split("-", 1)[1], x.get("kind", "group"), x["index"]): x["sid"] for x in w4["calls"] + w4["z"]}
    beyond, v4cls, shrink = [], {}, []
    for line in open(CALLS):
        c = json.loads(line)
        k = (c["unit"], c["kind"], c["index"])
        if k in want:
            v4cls[want[k]] = c["class"]
        vh, v7, eh, e7 = F(c["value"][0]), F(c["value"][1]), F(c["abserr"][0]), F(c["abserr"][1])
        d = abs(vh - v7)
        if d > e7:
            beyond.append({k_: c[k_] for k_ in ("unit", "kind", "index", "label", "class", "converged", "num_regions", "max_depth")}
                          | {"over_e07": float(d / e7), "over_eh": float(d / eh) if eh else None, "beyond_both": d > max(eh, e7),
                             "abserr_h_over_07": float(eh / e7), "route": c["route"] and {kk: c["route"][kk] for kk in ("n_route_differs", "n_lu_differs", "differs_by_class")}})
        if c["class"] == "aux_only" and e7 and eh / e7 < Fraction(1, 2):
            shrink.append({k_: c[k_] for k_ in ("unit", "kind", "index", "label", "converged", "num_regions")}
                          | {"abserr_h_over_07": float(eh / e7), "value_move_over_e07": float(d / e7), "value_move_over_eh": float(d / eh)})
    doc = {"header": hdr, "n_beyond_e07": len(beyond), "beyond_by_class": dict(Counter(b["class"] for b in beyond)),
           "beyond_both_by_class": dict(Counter(b["class"] for b in beyond if b["beyond_both"])), "beyond": beyond,
           "v4_classes": v4cls, "v4_by_class": dict(Counter(v4cls.values())), "aux_shrink_gt_2x": shrink,
           "aux_shrink_gt_2x_move_over_eh": {"max": max((s["value_move_over_eh"] for s in shrink), default=None),
                                            "n_gt_1": sum(1 for s in shrink if s["value_move_over_eh"] > 1)}}
    out_p.write_text(json.dumps(doc, indent=1, default=str))
    print(json.dumps({k: v for k, v in doc.items() if k not in ("header", "beyond", "aux_shrink_gt_2x")}, indent=1))
    for b in beyond:
        print(b["unit"], b["kind"], b["index"], b["label"][-38:], b["class"], f"{b['over_e07']:.3g} {b['over_eh']:.3g} both={b['beyond_both']} eh/e7={b['abserr_h_over_07']:.3g}", b["converged"], b["num_regions"], (b["route"] or {}).get("differs_by_class") and {k: v for k, v in b["route"]["differs_by_class"].items() if v["route"] or v["lu"]})
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
