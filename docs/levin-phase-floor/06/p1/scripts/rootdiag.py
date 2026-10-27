"""levin-phase-floor 06, finding P1: why the root region of G+q+r is accepted under r14 and bisected
under r13. Under one tree, after the validated rebuild (p1lib).

adaptive_levin_sincos(..., depth_max=0) on a region accepts it whatever the test says, and its one
``used_interval`` records the quantities the acceptance test used: ``abserr`` = |I(a,b) -
I(a,c) - I(c,b)| (the step-(4) residual), ``relerr`` = abserr / max(min(|est|,|refined|),
local_atol), ``phase_err`` (the Chen et al. eq. (151) round-off floor), ``phase_limited``. The
returned value is the region's own estimate. Done for the root, then its two halves (each with its
share of atol, as the driver gives it), then the halves of the half that contains x_s, down to
depth DEPTH. Every other argument is the captured call's.

Usage: PYTHONPATH=<tree> OMP_NUM_THREADS=1 python -B rootdiag.py r13|r14
Writes out/rootdiag_<tree>.json.
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import OUT, header, sha256  # noqa: E402
import p1lib  # noqa: E402
from p1lib import fx  # noqa: E402

DEPTH = 6


def main():
    tree = sys.argv[1]
    log = lambda s: print(s, flush=True)  # noqa: E731
    ana = json.loads((OUT / f"analyse_{tree}.json").read_text())
    out = {"header": header(Path(__file__), f"p1 rootdiag, tree {tree}"), "rows": {}}
    with p1lib.rebuilt_rows(tree, log=log) as (ev, ctx, rows):
        from harness import QSI_MOD

        levin = QSI_MOD.adaptive_levin_sincos
        out["evidence"] = ev
        for name, row in rows.items():
            if not row["validation"]["all_identical"]:
                raise SystemExit(f"p1: {tree} {name} does not reproduce its store; stopping")
            gc, groups, calls = p1lib.low_subinterval_groups(row)
            call = calls["G+q+r"]
            f = call["args"][1]
            theta = call["kwargs"]["theta"]
            atol = call["kwargs"]["atol"]
            rtol = call["kwargs"]["rtol"]
            order = call["kwargs"]["chebyshev_order"]
            x0, x1 = call["args"][0]
            W = x1 - x0
            xs = float.fromhex(ana["rows"][name]["groups"]["G+q+r"]["phase"]["stationary_points"][0]["x"]["hex"])
            recs = []

            def probe(a, b, depth):
                local_atol = atol * (b - a) / W
                res = levin((a, b), f, theta=theta, atol=local_atol, rtol=rtol, chebyshev_order=order, depth_max=0)
                (reg,) = res["regions"]
                resolved = (reg.abserr < local_atol) or (reg.relerr < rtol)
                rec = {
                    "depth": depth,
                    "span": [fx(a), fx(b)],
                    "z_span": [fx(math.expm1(a)), fx(math.expm1(b))],
                    "contains_x_s": a <= xs <= b,
                    "type": reg.type,
                    "estimate": fx(res["value"]),
                    "step4_residual": fx(reg.abserr),
                    "relerr": fx(reg.relerr),
                    "rtol": rtol,
                    "local_atol": fx(local_atol),
                    "roundoff_floor": fx(reg.phase_err),
                    "phase_limited": bool(reg.phase_limited),
                    "resolved": bool(resolved),
                    "accepted_by_driver": bool(resolved or reg.phase_limited),
                }
                recs.append(rec)
                log(f"   d{depth} [{a:.6f},{b:.6f}] xs-in {rec['contains_x_s']} {reg.type}: est {res['value']:.12e} "
                    f"resid {reg.abserr:.3e} relerr {reg.relerr:.3e} floor {reg.phase_err:.3e} "
                    f"resolved {resolved} phase_limited {reg.phase_limited}")
                return res["value"]

            log(f"== {tree} {name} G+q+r (x_s {xs!r})")
            a, b = x0, x1
            probe(a, b, 0)
            for depth in range(1, DEPTH + 1):
                c = (a + b) / 2.0
                probe(a, c, depth)
                probe(c, b, depth)
                a, b = (a, c) if xs <= c else (c, b)
            out["rows"][name] = {"x_s": fx(xs), "probes": recs}
    path = OUT / f"rootdiag_{tree}.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    log(f"wrote {path} sha256 {sha256(path)}")


if __name__ == "__main__":
    main()
