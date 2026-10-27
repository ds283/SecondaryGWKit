"""levin-phase-floor 06, finding P1, step 2: rebuild both rows under one tree, from that tree's own
store (read-only), and compare every recomputed value with the stored one by float.hex.

Usage: PYTHONPATH=<tree> OMP_NUM_THREADS=1 python -B validate.py r13|r14
Writes out/validate_<tree>.json.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import OUT, header, sha256  # noqa: E402
import p1lib  # noqa: E402


def main():
    tree = sys.argv[1]
    out = {"header": header(Path(__file__), f"p1 validate, tree {tree}")}
    with p1lib.rebuilt_rows(tree, log=lambda s: print(s, flush=True)) as (ev, ctx, rows):
        out["evidence"] = ev
        out["context_checks"] = ctx["checks"]
        out["rows"] = {}
        for name, row in rows.items():
            gc, groups, levin = p1lib.low_subinterval_groups(row)
            out["rows"][name] = {
                "shard": row["shard"],
                "serial": row["serial"],
                "rebuild_checks_all_hold": row["rebuild_checks_all_hold"],
                "rebuild_checks": row["rb"]["checks"],
                "qs_check_summary": row["qs_check_summary"],
                "compute_wall_s": row["compute_wall_s"],
                "validation": row["validation"],
                "GkPolicy": row["rb"]["GkPolicy_summary"],
                "low_subinterval_levin_calls": {
                    lab: {
                        "x_span": [p1lib.fx(v) for v in c["args"][0]],
                        "atol": p1lib.fx(c["kwargs"]["atol"]),
                        "rtol": c["kwargs"]["rtol"],
                        "chebyshev_order": c["kwargs"]["chebyshev_order"],
                        "theta_keys": sorted(c["kwargs"]["theta"].keys()),
                        "value": p1lib.fx(c["result"]["value"]),
                        "abserr": p1lib.fx(c["result"]["abserr"]),
                        "abserr_resolution": p1lib.fx(c["result"]["abserr_resolution"]),
                        "abserr_roundoff": p1lib.fx(c["result"]["abserr_roundoff"]),
                        "abserr_truncation": p1lib.fx(c["result"]["abserr_truncation"]),
                        "converged": bool(c["result"]["converged"]),
                        "num_regions": int(c["result"]["num_regions"]),
                        "evaluations": int(c["result"]["evaluations"]),
                        "regions": [
                            [r.start, r.end, r.depth, r.type, r.abserr]
                            for r in c["result"]["regions"]
                        ],
                    }
                    for lab, c in levin.items()
                },
            }
            v = row["validation"]
            bad = [x for x in v["items"] if not x["identical"]]
            print(f"== {tree} {name}: all identical {v['all_identical']}; {len(v['items'])} "
                  f"items; differing {bad}", flush=True)
    path = OUT / f"validate_{tree}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {path} sha256 {sha256(path)}", flush=True)


if __name__ == "__main__":
    main()
