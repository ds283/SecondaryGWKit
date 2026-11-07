"""levin-phase-floor 07 section 2.5, mechanism (B): the untrusted window leaves of z12. New in 07.

z12 (var/levin-phase-floor-06g-fcc-laptop/inputs/z12.pkl, read as replay.py reads it) under section 1's code at
Phi_w = 8 pi and depth_max 9, every other argument the capture's. Lists the call's abserr against its request,
the number of returned leaves the trust check left untrusted, and, for the first ``N_LISTED`` untrusted window
leaves in ascending x, each one's depth, width, own value, nested-pair estimate, floor, floor less the node term,
signed node error E, delta1, delta2 and r: the numbers behind "the trust check reads the integrand's noise as
non-convergence". Writes out/z12_untrusted_leaves.txt. No Ray (ray.cloudpickle only), no store.
"""

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prov  # noqa: E402

sys.path.insert(0, str(prov.REPO))
sys.path.insert(
    1, str(prov.REPO / "var" / "levin-phase-floor-06g-fcc-laptop" / "harness")
)

PHI_W_PI = 8.0
DEPTH_MAX = 9
N_LISTED = 12


def main():
    import ray
    import ray.cloudpickle as cp

    import AdaptiveLevin.levin_quadrature as lq

    pickle = (
        prov.REPO / "var" / "levin-phase-floor-06g-fcc-laptop" / "inputs" / "z12.pkl"
    )
    call = cp.loads(pickle.read_bytes())
    lq._LEVIN_WINDOW_HALF_PHASE = PHI_W_PI * math.pi
    kw = dict(call["kw"])
    kw["depth_max"] = DEPTH_MAX
    r = lq.adaptive_levin_sincos(call["x_span"], call["f"], call["theta"], **kw)
    request = max(kw["atol"], kw["rtol"] * abs(r["value"]))
    lines = prov.header(
        __file__, "section 2.5 (B), z12's untrusted window leaves", (pickle,)
    )
    lines.append(f"# ray initialised: {ray.is_initialized()}")
    lines.append(
        f"Phi_w {PHI_W_PI:g} pi, depth_max {DEPTH_MAX}: abserr {r['abserr']:.4g}, request {request:.4g}, "
        f"converged {r['converged']}, regions {r['num_regions']}, window pieces {r['num_window_pieces']}, "
        f"untrusted {r['num_untrusted_regions']}"
    )
    win = [g for g in r["regions"] if g.window]
    untrusted = [g for g in win if g.trusted is False]
    lines.append(
        f"window leaves {len(win)}, untrusted {len(untrusted)}, their summed e' "
        f"{sum(g.budget_err for g in untrusted):.4g}"
    )
    lines.append(
        "depth | width | value_own | nested pair | floor | floor less node | E | delta1 | delta2 | r"
    )
    for g in untrusted[:N_LISTED]:
        lines.append(
            f"{g.depth} | {g.width:.3e} | {g.value_own:.4e} | {g.abserr:.3e} | {g.phase_err:.3e} | "
            f"{g.phase_err_less_node:.3e} | {g.node_error_signed:.3e} | {g.delta1:.3e} | {g.delta2:.3e} | "
            f"{g.convergence_ratio:.3g}"
        )
    (prov.OUT / "z12_untrusted_leaves.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-(N_LISTED + 4) :]))


if __name__ == "__main__":
    main()
