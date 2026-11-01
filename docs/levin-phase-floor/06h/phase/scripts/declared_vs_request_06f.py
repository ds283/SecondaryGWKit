"""levin-phase-floor prompt 06f, descriptive: across 06e's census, the declared-phase term summed over each
call's floor-attributed Clenshaw-Curtis leaves (checkpoint cc_floor) against the call's request
max(atol, rtol|v|), for the declared-dominated calls and for all calls with a non-zero declared term.
New in 06f. Usage: python -B declared_vs_request_06f.py"""

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT06E_SHA256, RUN06E, header, sha256  # noqa: E402

TERMS = ("t_value_rounding", "t_argument", "node_term", "declared")


def H(x):
    return float.fromhex(x["hex"])


def main():
    ck = RUN06E / "checkpoint.jsonl"
    for l in header(
        Path(__file__).resolve(),
        "levin-phase-floor 06f: the declared term against the request",
        [ck],
    ):
        print(l)
    assert sha256(ck) == CHECKPOINT06E_SHA256
    st = collections.defaultdict(collections.Counter)
    for line in open(ck):
        d = json.loads(line)["data"]
        for c in (d.get("calls") or []) + (d.get("other_calls") or []):
            fl = c.get("cc_floor")
            if not fl or fl.get("n_floor_attributed", 0) <= 0:
                continue
            s = {t: H(fl["sums_floor_attributed_cc_leaves"][t]) for t in TERMS}
            big = max(TERMS, key=lambda t: s[t])
            req = max(c["atol"], c["rtol"] * abs(H(c["value"])))
            for grp in (
                [f"{d['rtol']} declared-dominated"] if big == "declared" else []
            ) + ([f"{d['rtol']} any declared > 0"] if s["declared"] > 0 else []):
                k = st[grp]
                k["calls"] += 1
                r = s["declared"] / req
                k["D_attr >= request"] += r >= 1
                k["D_attr >= 0.1 request"] += r >= 0.1
                k["D_attr >= 0.01 request"] += r >= 0.01
                k["unconverged"] += not c["converged"]
                k["unconverged and D_attr >= 0.1 request"] += (
                    not c["converged"]
                ) and r >= 0.1
    for g in sorted(st):
        print(g, dict(st[g]))


if __name__ == "__main__":
    main()
