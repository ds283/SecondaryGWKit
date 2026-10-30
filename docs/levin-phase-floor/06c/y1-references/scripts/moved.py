"""Y1 references, step 3: which Levin calls moved between R17 and R18.

Pairs the two trees' captures (capture.py) call by call (same row, same index; the label, the
sub-interval, the span and the keyword arguments must agree) and calls a call *moved* if its value,
abserr, num_regions or max_depth is not bit-identical. Writes the selection (a list of {"row",
"index", "label", ...}) for the pickling capture.

Usage: python -B moved.py CAPTURE_R17 CAPTURE_R18 OUT_SELECTION_JSON
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yprov import header  # noqa: E402


def main():
    p17, p18, outp = sys.argv[1:4]
    for l in header(Path(__file__).resolve(), "Y1 references: moved calls", [Path(p17), Path(p18)]):
        print(l)
    A = {r["row"]: r for r in json.loads(Path(p17).read_text())["rows"]}
    B = {r["row"]: r for r in json.loads(Path(p18).read_text())["rows"]}
    assert set(A) == set(B)
    sel = []
    for rid in sorted(A):
        a, b = A[rid], B[rid]
        assert len(a["calls"]) == len(b["calls"]), rid
        print(f"\n{rid}: z_response {float.fromhex(a['z_response']):.6g}; numeric_quad equal "
              f"{a['numeric_quad'] == b['numeric_quad']}")
        for ca, cb in zip(a["calls"], b["calls"]):
            same_call = (ca["label"], ca["sub"], ca["x_span"], ca["notify_label"]) == (cb["label"], cb["sub"], cb["x_span"], cb["notify_label"])
            kwa = {k: v for k, v in ca["kw"].items()}
            kwb = {k: v for k, v in cb["kw"].items()}
            if not same_call or kwa != kwb:
                raise SystemExit(f"{rid} call {ca['index']}: the two trees' calls differ in their inputs: {ca['label']} {cb['label']} {kwa} {kwb}")
            diffs = [f for f in ("value", "abserr", "num_regions", "max_depth") if ca[f] != cb[f]]
            moved = bool(diffs)
            span = [float.fromhex(x) for x in ca["x_span"]]
            print(f"  {ca['index']:2d} sub {ca['sub']} {ca['label']:<6} span [{span[0]:.6g}, {span[1]:.6g}] "
                  f"{'MOVED (' + ','.join(diffs) + ')' if moved else 'identical'}; "
                  f"R17 value {ca['value']} abserr {ca['abserr']} reg {ca['num_regions']} d {ca['max_depth']} | "
                  f"R18 value {cb['value']} abserr {cb['abserr']} reg {cb['num_regions']} d {cb['max_depth']}")
            if moved:
                sel.append({"row": rid, "index": ca["index"], "label": ca["label"], "sub": ca["sub"],
                            "x_span": ca["x_span"], "differs_in": diffs})
    Path(outp).write_text(json.dumps(sel, indent=1))
    by_row = {}
    for e in sel:
        by_row.setdefault(e["row"], []).append(f"{e['index']}:{e['label']}")
    print(f"\nmoved calls: {len(sel)} in {len(by_row)} rows")
    for rid in sorted(by_row):
        print(f"  {rid}: {', '.join(by_row[rid])}")
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
