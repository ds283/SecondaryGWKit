"""levin-phase-floor prompt 07a section 6, the user's references (state/ruling_w5.md item 2; new in 07a; read only but for
its output): the calls of rows 3-678 and 1-34599 as out/scores/moves_beyond.json and the aux_only table identify them --
the group calls index 2 (G+q-r) and 3 (G-q+r), at 1e-7 and at 1e-8, the rtols each was flagged at -- with their census
records at h and at 07 (value, abserr hex; num_regions, max_depth, converged; label). sid r<row>-<rtol>-<index>.
Usage: python -B refs_select_07a.py OUT_JSON
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

CALLS = WIDER / "out" / "stop3" / "compare_census_calls.calls.jsonl"
MB = WIDER / "out" / "scores" / "moves_beyond.json"
ROWS = ("3-678", "1-34599")


def main():
    out_p = Path(sys.argv[1])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: the user's two reference calls, selected", [CALLS, MB])
    mb = json.loads(MB.read_text())
    flagged = {(b["unit"], b["kind"], b["index"]) for b in mb["beyond"] if b["beyond_both"]} | {
        (s["unit"], s["kind"], s["index"]) for s in mb["aux_shrink_gt_2x"] if s["value_move_over_eh"] > 1}
    sel = []
    for line in open(CALLS):
        c = json.loads(line)
        k = (c["unit"], c["kind"], c["index"])
        if k not in flagged:
            continue
        row = "-".join(c["unit"].split("-")[3:])
        if row not in ROWS:
            raise SystemExit(f"a flagged call outside the two rows: {k}")
        sel.append({"sid": f"r{row}-{c['rtol']}-{c['index']}", "unit": c["unit"], "g_unit": "g-" + c["unit"].split("-", 1)[1],
                    "index": c["index"], "kind": c["kind"], "notify_label": c["label"], "class": c["class"],
                    "census_value": c["value"][0], "census_abserr": c["abserr"][0], "census_num_regions": c["num_regions"][0],
                    "census_max_depth": c["max_depth"][0], "census_converged": c["converged"][0],
                    "census07_value": c["value"][1], "census07_abserr": c["abserr"][1], "census07_num_regions": c["num_regions"][1],
                    "census07_max_depth": c["max_depth"][1], "census07_converged": c["converged"][1]})
    sel.sort(key=lambda e: e["sid"])
    out_p.write_text(json.dumps(sel, indent=1))
    for l in hdr:
        print(l)
    for e in sel:
        print(e["sid"], e["unit"], e["index"], e["class"], e["notify_label"])
    print(f"{len(sel)} calls; wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if len(sel) == 8 else 1


if __name__ == "__main__":
    sys.exit(main())
