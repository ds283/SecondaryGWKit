"""levin-phase-floor prompt 06c section 5, development check (not part of the census): worker.py's
new per-call functions on two of 06c's own test problems at g, since Gate X0's g unit has no call
at depth_max. New in 06c.

- 06c's test 5 (``test5_problem``: f = (|x - 1/3|, 0), theta = 100 x on [0, 1], rtol 1e-12,
  depth_max 6): the call reaches depth_max, so ``_deep_line`` runs on it; its line's last level
  must reproduce the returned leaf's residual and floor bit for bit, and its line end the leaf.
- 06c's test 12d (``test12d_problem``, depth_max 0): one untrusted leaf at the cap, so the
  per-call summary's trust counts and the region arrays' ``trusted`` are exercised; ``_deep_line``
  runs on its depth-0 line too.

Each call is made with ``adaptive_levin_sincos`` (as QuadSourceIntegral calls it), and handed to
``_summarise_call_06c``, ``_region_arrays_06c`` and ``_deep_line`` in the dict form worker.py's
wrapper builds. Prints what they return. Read-only; writes nothing.

Usage: PYTHONPATH=<tree g> python -B dev_worker_functions.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import AdaptiveLevin.levin_quadrature as LQ  # noqa: E402
import AdaptiveLevin.tests.test_levin_global_budget as T  # noqa: E402

import worker as W  # noqa: E402
from prov import TREES, header  # noqa: E402


def call_dict(span, f, theta, kwargs, result):
    return {
        "index": 0,
        "notify_label": None,
        "parsed": None,
        "x_span": tuple(span),
        "atol": kwargs.get("atol"),
        "rtol": kwargs.get("rtol"),
        "result": result,
        "theta": theta,
        "f": f,
        "kw": dict(kwargs),
        "premise_log": [],
        "budget": None,
        "solves": None,
        "default_depth_max": LQ.DEFAULT_LEVIN_MAX_DEPTH,
    }


def main():
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06c: development check of worker.py's new functions"):
        print(l)
    print(f"# AdaptiveLevin.levin_quadrature from {LQ.__file__}")
    if not LQ.__file__.startswith(str(TREES["g"][0])):
        raise SystemExit("PYTHONPATH must be the tree g")
    ok = True
    for name, prob, span in (("test 5", T.test5_problem, T.TEST5_SPAN), ("test 12d", T.test12d_problem, T.TEST12D_SPAN)):
        f, theta, kwargs, _expr = prob()
        res = LQ.adaptive_levin_sincos(span, f, theta=theta, **kwargs)
        c = call_dict(span, f, theta, kwargs, res)
        s = W._summarise_call_06c(c)
        arrs = W._region_arrays_06c(c)
        line = W._deep_line(LQ, c)
        print(f"\n== {name}: kwargs {kwargs}; converged {res['converged']}, regions {res['num_regions']}, untrusted {res['num_untrusted_regions']}")
        print("   summary: " + json.dumps({k: s[k] for k in ("max_depth", "depth_max", "n_at_depth_max", "abserr", "abserr_finite", "num_untrusted_regions",
                                                            "E_share", "E_largest_share", "E_unwidened", "leaves", "r_max_finite", "n_r_not_finite")}))
        print(f"   region arrays: {sorted(arrs)}; trusted {arrs['trusted'].tolist()}; r {arrs['convergence_ratio'].tolist()[:8]}")
        rep = line["line_end_equals_leaf_and_reproduces_its_residual_and_floor"]
        ok &= bool(rep)
        print(f"   deep line: max_depth {line['max_depth']}, leaves there {line['n_leaves_at_max_depth']}, leaf [{line['leaf_start']}, {line['leaf_end']}], "
              f"reproduces {rep}, solves {line['n_solves']}")
        for L in line["levels"]:
            print(f"      depth {L['depth']}: P {L['P']['type']} resid {L['P']['resid']} floor {L['P']['floor']} trunc {L['P']['trunc']} e {L['P']['e']}; "
                  f"(eL+eR)/eP {L['eLR_over_eP']}; delta1 {L['delta1']} delta2 {L['delta2']} r {L['r']}")
        # the leaf's own trust numbers, where the call took the check, against the line's
        leaf = min((r for r in res["regions"] if r.depth == line["max_depth"]), key=lambda r: r.start)
        if leaf.delta1 is not None:
            last = line["levels"][-1]
            same = (float(leaf.delta1).hex(), float(leaf.delta2).hex()) == (last["delta1"], last["delta2"])
            ok &= same
            print(f"   the leaf's own delta1/delta2 equal the line's last level: {same}")
    print(f"\n== {'ALL HOLD' if ok else 'SOMETHING DIFFERS'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
