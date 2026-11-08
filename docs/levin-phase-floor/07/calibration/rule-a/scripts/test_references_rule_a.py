"""levin-phase-floor 07 section 10 (Rule A): the references of section 3 test 13 (a)-(d), which
AdaptiveLevin/tests/test_levin_stationary.py asserts against. New in 07's amendment.

Every reference is a closed form, computed at 40 working digits and again at 60; each value's two readings must
agree far below the tests' tolerances, and each closed form is also checked against mpmath.quad on [0, 1] split at
x_s (the phases here vary by at most 20 rad, so mpmath.quad converges directly). The tests hold the 40-digit
values as float hex, citing this file's output (rule-a/out/test_references_rule_a.json). No Ray, no store, no
driver code.

The integrands (I = int_0^1 [f_sin sin theta + f_cos cos theta] dx, f_sin = e^{-x}, f_cos = cos 3x, the
amplitudes of section 2's families):
- **(a) a zero rate:** theta = 0, theta' = 0: I = int_0^1 cos 3x dx = sin(3) / 3.
- **(b), (c) a small phase with a root:** theta = (c/2)(x - x_s)^2, x_s = 0.3819660112501051 (non-dyadic), at
  c = 70 (b: total variation V = (c/2)(x_s^2 + (1 - x_s)^2) = 18.475 < 6 pi) and c = 72 (c: V = 19.003, just
  above 6 pi = 18.850): reference.closed_R(c, x_s), the closed form in erf of section 2.1's family (R), with
  reference.contour_R beside it. V is also recorded exactly, from the doubles the test hands the driver.
- **(d) several phases:** test 9's two-phase basis (sin theta_1, cos theta_1, sin theta_2, cos theta_2) with
  theta_1 = lambda_1 x, lambda_1 = 200, and theta_2 = 0 identically; f = (e^{-x}, 0, 0, cos 3x):
  I = int_0^1 e^{-x} sin(lambda_1 x) dx + int_0^1 cos 3x dx
    = (lambda_1 - e^{-1} (sin lambda_1 + lambda_1 cos lambda_1)) / (1 + lambda_1^2) + sin(3) / 3.

Writes rule-a/out/test_references_rule_a.json.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mpmath as mp  # noqa: E402

import prov  # noqa: E402
import reference as R  # noqa: E402

X_S = 0.3819660112501051
C_BELOW = 70.0
C_ABOVE = 72.0
LAMBDA_1 = 200.0


def zero_rate(dps):
    with mp.workdps(dps):
        return mp.sin(3) / 3


def zero_rate_quad(dps):
    with mp.workdps(dps):
        return mp.quad(lambda x: mp.cos(3 * x), [0, 1])


def small_phase_quad(c, dps):
    with mp.workdps(dps):
        c = mp.mpf(c)
        x_s = mp.mpf(X_S)

        def f(x):
            t = c / 2 * (x - x_s) ** 2
            return mp.exp(-x) * mp.sin(t) + mp.cos(3 * x) * mp.cos(t)

        return mp.quad(f, [0, x_s, 1])


def variation(c, dps=40):
    with mp.workdps(dps):
        c = mp.mpf(c)
        x_s = mp.mpf(X_S)
        return c / 2 * (x_s**2 + (1 - x_s) ** 2)


def two_phase_zero(dps):
    with mp.workdps(dps):
        lam = mp.mpf(LAMBDA_1)
        e1 = mp.exp(-1)
        return (lam - e1 * (mp.sin(lam) + lam * mp.cos(lam))) / (1 + lam**2) + mp.sin(
            3
        ) / 3


def two_phase_zero_quad(dps):
    with mp.workdps(dps):
        lam = mp.mpf(LAMBDA_1)
        pieces = [mp.mpf(k) / 64 for k in range(65)]
        return mp.quad(lambda x: mp.exp(-x) * mp.sin(lam * x) + mp.cos(3 * x), pieces)


def main():
    out = {
        "header": prov.json_header(
            __file__,
            "section 10 (Rule A), section 3 test 13's references",
            (HERE / "reference.py",),
        ),
        "x_s": X_S,
        "six_pi": float(6 * mp.pi),
    }

    def record(name, v40, v60, check=None, extra=None):
        with mp.workdps(60):
            rec = {
                "value_40": mp.nstr(v40, 30),
                "value_hex": float(v40).hex(),
                "relative_40_vs_60": float(abs((v40 - v60) / v60)),
            }
            if check is not None:
                label, value = check
                rec[f"{label}_30"] = mp.nstr(value, 30)
                rec[f"relative_vs_{label}"] = float(abs((v40 - value) / v40))
        rec.update(extra or {})
        out[name] = rec
        print(name, json.dumps(rec), flush=True)

    record(
        "a_zero_rate",
        zero_rate(40),
        zero_rate(60),
        ("mpmath_quad", zero_rate_quad(40)),
    )

    for name, c in (("b_below", C_BELOW), ("c_above", C_ABOVE)):
        R.WORKING_DPS = 40
        v40 = R.closed_R(c, X_S, 0.0)
        contour = R.contour_R(c, X_S, 0.0)
        R.WORKING_DPS = 60
        v60 = R.closed_R(c, X_S, 0.0)
        R.WORKING_DPS = 40
        with mp.workdps(40):
            rel_contour = float(abs((v40 - contour) / v40))
            V = variation(c)
        record(
            name,
            v40,
            v60,
            ("mpmath_quad", small_phase_quad(c, 40)),
            {
                "c": c,
                "closed_minus_contour_relative": rel_contour,
                "variation_exact": float(V),
                "variation_exact_30": mp.nstr(V, 30),
                "variation_over_six_pi": float(V / (6 * mp.pi)),
            },
        )

    record(
        "d_two_phases_zero",
        two_phase_zero(40),
        two_phase_zero(60),
        ("mpmath_quad", two_phase_zero_quad(40)),
        {"lambda_1": LAMBDA_1},
    )

    path = prov.OUT / "test_references_rule_a.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
