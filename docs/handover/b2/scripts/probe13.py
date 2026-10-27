"""
Probe for [13-the-tlg-term-depends-on-the-levin-subdivision]: is the TLG interaction the Levin
driver being handed T's phase derivative from `omega` (Liouville-Green closed form) while the
phase it oscillates with is the stored one?

Three arms, each at three places, each forming N for E', E'+TLG, R-TLG and R (B2's objects, built
by B2's recorded b2_cells, imported, not edited):
  base     -- as B2 ran it: _OscillatoryT.theta_deriv = omega (1+z)
  phase    -- _OscillatoryT.theta_deriv = phase.theta_deriv(log derivative): the derivative now
              matches the phase the driver oscillates with; omega still enters the integrand's
              b = (1+z) M omega coefficient
  noderiv  -- LEVIN_USE_THETA_DERIV = False: the driver differentiates theta itself
If the mechanism holds, under `phase` the add-one A_TLG equals the remove-one B_TLG.
"""

import sys
import warnings
from math import exp
from pathlib import Path

REPO = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
sys.path.insert(0, str(REPO / "docs" / "handover" / "b2" / "scripts"))
import b2_cells as C  # noqa: E402
import ComputeTargets.phase_groups as PG  # noqa: E402
import importlib  # noqa: E402
QSI = importlib.import_module("ComputeTargets.QuadSourceIntegral")

warnings.simplefilter("ignore")

ORIGINAL_DERIV = PG._OscillatoryT.theta_deriv


def phase_deriv(self, log_z):
    return self.phase.theta_deriv(log_z, x_is_log=True, log_derivative=True)


def N(shape, x, obj):
    case, _ = C.build_case(shape, x, obj)
    out = case.run(atol=C.REF_ATOL, rtol=C.REF_RTOL)
    ref = C.reference_of(case, case.shape)
    total = float(out["total"])
    halves = abs(float(out["numeric_quad"])) + abs(float(out["WKB_Levin"]))
    declared = max(float(out["total_abserr"]), C.REF_RTOL * halves) / abs(total)
    n = C.KT.KT_NORM * total / ref["predicted"]
    meta = out.get("metadata") or {}
    return n, abs(n) * declared


def sign_check():
    """On E' (exact accessors), omega (1+z) and the phase's log derivative must agree."""
    case, _ = C.build_case("q-smooth", 980.0, "Ep")
    T = case.Tk_builder(case.model, case.r.k, *case.Tr_inputs)
    lo = T.WKB_region[1]
    hi = T.WKB_region[0]
    from math import log

    worst = 0.0
    for j in range(1, 20):
        u = log(1.0 + lo) + j / 20 * (log(1.0 + hi) - log(1.0 + lo))
        a = T.omega(u, z_is_log=True) * exp(u)
        b = T.phase.theta_deriv(u, x_is_log=True, log_derivative=True)
        worst = max(worst, abs(a - b) / abs(a))
    return worst


def arm(name):
    if name == "phase":
        PG._OscillatoryT.theta_deriv = phase_deriv
    else:
        PG._OscillatoryT.theta_deriv = ORIGINAL_DERIV
    QSI.LEVIN_USE_THETA_DERIV = name != "noderiv"


lines = [f"probe13, b2_cells sha256 {C.SCRIPT_SHA256}"]
lines.append(f"sign check on E' (exact T_r): max |omega(1+z) - phase'| / |omega(1+z)| = {sign_check():.2e}")
for shape, x in (("q-smooth", 980.0), ("q-smooth", 1e7), ("together", 1e7), ("together", 980.0)):
    for name in ("noderiv",):
        arm(name)
        v = {o: N(shape, x, o) for o in ("Ep", "Ep+TLG", "R-TLG", "R")}
        A = v["Ep+TLG"][0] - v["Ep"][0]
        B = v["R"][0] - v["R-TLG"][0]
        err = sum(e for _, e in v.values())
        lines.append(
            f"{shape:9s} {x:7g} {name:8s} N_E'+9/8 {v['Ep'][0] + 9 / 8:+.4e}  N_R+9/8 {v['R'][0] + 9 / 8:+.4e}  "
            f"A_TLG {A:+.4e}  B_TLG {B:+.4e}  A-B {A - B:+.3e}  (err {err:.1e})"
        )
        print(lines[-1], flush=True)
arm("base")
Path(__file__).with_name("probe13_noderiv_out.txt").write_text("\n".join(lines) + "\n")
