"""levin-phase-floor prompt 07b Part A: the calls captured afresh at h (x1_capture_07b.py's selection). New in 07b.

- section 2.1 item 4: each of 06b's W2 calls dishonest at 07a (out/w2/w2_rescore.json, ``dishonest_h``);
- section 2.4: every R2 and R3 call of state/selection_07b.json except the pairs (07a's Part A captured them) and R1
  (07's cross-check captures X00-X09 hold them, re-checked by replay).
Each entry carries 07a's census record (``census_*``) and 07's (``census07_*``) from 07a's stop-3 call comparison.
Writes OUT_JSON (a list, x1_capture's format). Usage: PYTHONPATH=<tree>:<scripts> python -B capture_list_07b.py OUT_JSON
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import REC07A, WIDER, header, sha256  # noqa: E402

SEL = WIDER / "state" / "selection_07b.json"
W2 = WIDER / "out" / "w2" / "w2_rescore.json"
CALLS = REC07A / "out" / "stop3" / "compare_census_calls.calls.jsonl"


def main():
    out_p = Path(sys.argv[1])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    sel = json.loads(SEL.read_text())["calls"]
    w2 = json.loads(W2.read_text())["rows"]
    want = {(r["unit_h"], "group", r["index"]): f"W2-{'-'.join(r['unit_h'].split('-')[3:])}-{r['index']}" for r in w2 if r["dishonest_h"]}
    calls = {}
    with open(CALLS) as f:
        for line in f:
            c = json.loads(line)
            k = (c["unit"], c["kind"], c["index"])
            if k in want:
                calls[k] = c
    out = []
    for k, sid in sorted(want.items(), key=lambda kv: kv[1]):
        c = calls[k]
        out.append({"sid": sid, "set": "W2", "unit": c["unit"], "g_unit": "g-" + c["unit"].split("-", 1)[1], "index": c["index"],
                    "kind": c["kind"], "notify_label": c["label"], "class": c["class"], "rtol": c["rtol"],
                    "census_value": c["value"][0], "census_abserr": c["abserr"][0], "census_num_regions": c["num_regions"][0],
                    "census_max_depth": c["max_depth"][0], "census_converged": c["converged"][0],
                    "census07_value": c["value"][1], "census07_abserr": c["abserr"][1], "census07_num_regions": c["num_regions"][1],
                    "census07_max_depth": c["max_depth"][1], "census07_converged": c["converged"][1]})
    for e in sel:
        if e["set"] in ("R2", "R3-W7", "R3-aux", "R3-own"):
            out.append({k: v for k, v in e.items() if k != "stratum"})
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07b Part A: the calls captured afresh at h", [SEL, W2, CALLS])
    out_p.write_text(json.dumps(out, indent=1))
    for l in hdr:
        print(l)
    print(f"{len(out)} calls on {len({e['unit'] for e in out})} units; wrote {out_p} sha256 {sha256(out_p)}")
    (out_p.parent / (out_p.stem + "_header.txt")).write_text("\n".join(hdr) + "\n")


if __name__ == "__main__":
    main()
