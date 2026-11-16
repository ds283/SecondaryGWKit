"""levin-phase-floor prompt 07b section 2.1 item 4: 06b's 24 W2 calls re-scored from 07a's census checkpoint against their
committed references S_10 (docs/levin-phase-floor/06b/census/out/w2_reference.json, in the tree), by 06b's rule,
|v - S_10| / abserr, beside their ratio at 06b's code (``err_b_over_abserr_b``). Unit ``b-1e-7-X-NNNNN`` is 07a's
``h-1e-7-X-NNNNN``, the same group call index. A call is dishonest if the ratio exceeds 1. New in 07b.

The reference's stated error is |S_9 - S_10| (06b's agreement measure); a call whose |S_9 - S_10| exceeds abserr_h/100 is
reported, not scored (06b's drop rule, at 07a's abserr). Writes OUT_JSON with provenance. Read only otherwise.
Usage: PYTHONPATH=<tree>:<scripts> python -B w2_rescore_07b.py OUT_JSON
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07A_SHA256, RUN07A, TREES, header, sha256  # noqa: E402

W2 = TREES["h"][0] / "docs" / "levin-phase-floor" / "06b" / "census" / "out" / "w2_reference.json"


def main():
    out_p = Path(sys.argv[1])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    ck = RUN07A / "checkpoint.jsonl"
    if sha256(ck) != CHECKPOINT07A_SHA256:
        raise SystemExit("07a's checkpoint has another SHA-256")
    w2 = json.loads(W2.read_text())
    want = {}
    for r in w2["results"]:
        want.setdefault("h" + r["unit"][1:], []).append(r)
    recs = {}
    with open(ck) as f:
        for line in f:
            uid = line[10:60].split('"', 1)[0]
            if uid in want:
                d = json.loads(line)["data"]
                if d.get("status") == "done":
                    if uid in recs:
                        raise SystemExit(f"{uid}: two done records")
                    recs[uid] = d
    if set(recs) != set(want):
        raise SystemExit(f"missing units {sorted(set(want) - set(recs))}")
    rows = []
    for uid, rs in sorted(want.items()):
        d = recs[uid]
        for r in rs:
            c = [c for c in d["calls"] if c["index"] == r["index"]]
            if len(c) != 1:
                raise SystemExit(f"{uid} {r['index']}: {len(c)} call records")
            c = c[0]
            if c["notify_label"] != r["notify_label"]:
                raise SystemExit(f"{uid} {r['index']}: label differs")
            v = float.fromhex(c["value"]["hex"])
            e = float.fromhex(c["abserr"]["hex"])
            s10 = float.fromhex(r["S10"])
            s9 = float.fromhex(r["S9"])
            spread = abs(s9 - s10)
            ratio = abs(v - s10) / e
            rows.append({"unit_b": r["unit"], "unit_h": uid, "index": r["index"], "label": r["notify_label"], "band": r["band"],
                         "k_key": r["k_key"], "value_h": c["value"]["hex"], "abserr_h": c["abserr"]["hex"],
                         "converged_h": c["converged"], "num_regions_h": c["num_regions"], "max_depth_h": c["max_depth"],
                         "S10": r["S10"], "S9": r["S9"], "spread": spread, "spread_over_abserr_h": spread / e,
                         "scored": spread <= e / 100, "ratio_h": ratio, "dishonest_h": ratio > 1,
                         "ratio_b": r["err_b_over_abserr_b"], "value_b": r["value_b"], "abserr_b": r["abserr_b"],
                         "value_h_equals_b": c["value"]["hex"] == r["value_b"],
                         "abserr_h_over_b": e / float.fromhex(r["abserr_b"])})
    if sha256(ck) != CHECKPOINT07A_SHA256:
        raise SystemExit("07a's checkpoint changed")
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07b Part A: 06b's W2 calls re-scored at 07a's census", [ck, W2])
    n_dis = sum(r["dishonest_h"] and r["scored"] for r in rows)
    doc = {"header": hdr, "n_calls": len(rows), "n_scored": sum(r["scored"] for r in rows),
           "n_dishonest_h": n_dis, "n_dishonest_converged_h": sum(r["dishonest_h"] and r["scored"] and r["converged_h"] for r in rows),
           "n_dishonest_b": sum(r["ratio_b"] > 1 for r in rows), "rows": rows}
    out_p.write_text(json.dumps(doc, indent=1))
    for l in hdr:
        print(l)
    print("| unit (h) | idx | band | k | ratio at 06b | ratio at 07a | converged | regions | depth | spread/abserr_h | abserr_h/abserr_b |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['unit_h']} | {r['index']} | {r['band']} | {r['k_key']} | {r['ratio_b']:.3g} | {r['ratio_h']:.3g}"
              f"{' *' if r['dishonest_h'] else ''} | {r['converged_h']} | {r['num_regions_h']} | {r['max_depth_h']} | "
              f"{r['spread_over_abserr_h']:.2g} | {r['abserr_h_over_b']:.3g} |")
    print(f"{len(rows)} calls; scored {doc['n_scored']}; dishonest at 07a {n_dis} (converged {doc['n_dishonest_converged_h']}); "
          f"dishonest at 06b {doc['n_dishonest_b']}")
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
