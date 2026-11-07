"""levin-phase-floor 07 section 3: the references AdaptiveLevin/tests/test_levin_stationary.py asserts against.
New in 07.

Every reference is computed at 40 working digits and again at 60, and each value's two readings must agree to
far below the tests' tolerances; the closed forms are also checked against mpmath.quad subdivided to
convergence where that is practical. The tests hold the 40-digit values as float hex, citing this file's output
(out/run2/test_references.json). No Ray, no store, no driver code.

The integrands (x is the integration variable; I = int [f_sin sin theta + f_cos cos theta] dx):
- **P1** (tests 1, 4, 6, 7, 10 (c)): on [0, L], theta = A e^x - a x with a = A e^{x_s} (so theta' = A e^x - a
  has one root, x_s), f_sin = e^{g (x - x_s)}, f_cos = cos(h x); L = 16, A = 169, x_s = 9.381966011250105,
  g = 0.33, h = 0.78. Total variation of theta about 1.5e9 rad.
- **P5** (test 5): the same family with x_s = 10.0918.
- Closed form (both): with u = e^x, int_0^L e^{alpha x} e^{i theta} dx = int_1^{e^L} u^{s-1} e^{i A u} du,
  s = alpha - i a, and with t = -i A u (arg t = arg(-iA) on the whole path, so the principal branches agree),
  = (-iA)^{-s} [Gamma(s, -iA) - Gamma(s, -iA e^L)] = (-iA)^{-s} * mpmath.gammainc(s, -iA, -iA e^L). Then
  I = e^{-g x_s} Im J(g) + Re[J(ih) + J(-ih)] / 2.
  Checked against mpmath.quad on pieces of at most pi of phase, split at x_s, on a small member of the family
  (L = 5, A = 1, x_s = 2.1) where that takes seconds.
- **Two roots** (test 2): on [0, 1], theta = (b/3) u^3 + a u, u = x - x_0, x_0 = 0.3819660112501051, b = 1e4,
  a = -b (d/2)^2 with d = 0.01, so theta' = b u^2 + a has roots at x_0 -+ d/2; f = (e^{-x}, cos 3x). By
  mpmath.quad on pieces of at most pi/2 of phase, split at both roots and at x_0 (reference.literal), at 30
  and at 45 digits.
- **Family (C)** (test 3): reference.contour_C at lambda = 1, b = 1e4 and 1e8 (lambda = Lambda_c / 4).
- **Two phases** (test 9): on [0, 1], f = (1, 0, 0, 1) on (sin theta_1, cos theta_1, sin theta_2, cos theta_2),
  theta_1 = lambda_1 x, theta_2 = (c/2)(x - x_s)^2, lambda_1 = 200, c = 1e5, x_s = 0.3819660112501051:
  I = (1 - cos lambda_1) / lambda_1 + Re int_0^1 e^{i theta_2} dx, the second by the error function.
- **Family (R)** (test 10 (a), (b)): reference.closed_R at c = 1e6, omega = 0, x_s = 0.7236067977499790 (a) and
  0.3819660112501051 (b), with reference.contour_R beside it.

Writes out/run2/test_references.json.
"""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mpmath as mp  # noqa: E402

import prov  # noqa: E402
import reference as R  # noqa: E402

P1 = dict(L=16.0, A=169.0, x_s=9.381966011250105, g=0.33, h=0.78)
P5 = dict(L=16.0, A=169.0, x_s=10.0918, g=0.33, h=0.78)
SMALL = dict(L=5.0, A=1.0, x_s=2.1, g=0.33, h=0.78)
TWO_ROOTS = dict(b=1.0e4, d=0.01, x_0=0.3819660112501051)
TWO_PHASES = dict(lambda_1=200.0, c=1.0e5, x_s=0.3819660112501051)


def exp_family(p, dps):
    with mp.workdps(dps):
        L, A, x_s, g, h = (mp.mpf(p[k]) for k in ("L", "A", "x_s", "g", "h"))
        a = mp.mpf(p["A"] * math.exp(p["x_s"]))  # the double the test hands the driver
        c = mp.mpc(0, -1) * A

        def J(alpha):
            s = alpha - mp.mpc(0, 1) * a
            return c ** (-s) * mp.gammainc(s, c, c * mp.exp(L))

        return (
            mp.im(J(g)) * mp.exp(-g * x_s)
            + mp.re(J(mp.mpc(0, h)) + J(mp.mpc(0, -h))) / 2
        )


def exp_family_literal(p, dps=30):
    with mp.workdps(dps):
        L, A, x_s, g, h = (mp.mpf(p[k]) for k in ("L", "A", "x_s", "g", "h"))
        a = mp.mpf(p["A"] * math.exp(p["x_s"]))

        def theta(x):
            return A * mp.exp(x) - a * x

        def f(x):
            t = theta(x)
            return mp.exp(g * (x - x_s)) * mp.sin(t) + mp.cos(h * x) * mp.cos(t)

        total = mp.mpf(0)
        pieces = 0
        for lo, hi in ((mp.mpf(0), x_s), (x_s, L)):
            x = lo
            while x < hi:
                rate = abs(A * mp.exp(x) - a)
                step = mp.pi / max(rate, mp.pi)
                nxt = min(x + step, hi)
                while abs(theta(nxt) - theta(x)) > mp.pi:
                    nxt = x + (nxt - x) / 2
                total += mp.quad(f, [x, nxt])
                pieces += 1
                x = nxt
        return total, pieces


def two_roots(dps, piece):
    b = mp.mpf(TWO_ROOTS["b"])
    d = TWO_ROOTS["d"]
    a_float = -TWO_ROOTS["b"] * (d / 2) ** 2
    x0 = TWO_ROOTS["x_0"]
    with mp.workdps(dps):
        a = mp.mpf(a_float)
        x0m = mp.mpf(x0)
        s = mp.sqrt(-a / b)
        theta = lambda x: a * (x - x0m) + b / 3 * (x - x0m) ** 3  # noqa: E731
        rate = lambda x: a + b * (x - x0m) ** 2  # noqa: E731
        saved = R.REPORT_DPS
        R.REPORT_DPS = dps
        value, pieces = R.literal(theta, rate, [0, x0m - s, x0m, x0m + s, 1], piece)
        R.REPORT_DPS = saved
        return value, pieces, a_float


def two_phases(dps):
    with mp.workdps(dps):
        lam = mp.mpf(TWO_PHASES["lambda_1"])
        c = mp.mpf(TWO_PHASES["c"])
        x_s = mp.mpf(TWO_PHASES["x_s"])
        beta = mp.mpc(0, -1) * c / 2
        sb = mp.sqrt(beta)
        fresnel_part = (
            mp.sqrt(mp.pi) / (2 * sb) * (mp.erf(sb * (1 - x_s)) - mp.erf(sb * (-x_s)))
        )
        return (1 - mp.cos(lam)) / lam + mp.re(fresnel_part)


def main():
    out = {
        "header": prov.json_header(
            __file__, "section 3, the tests' references", (HERE / "reference.py",)
        )
    }

    def record(name, v40, v60, extra=None):
        with mp.workdps(60):
            rec = {
                "value_40": mp.nstr(v40, 30),
                "value_hex": float(v40).hex(),
                "relative_40_vs_60": float(abs((v40 - v60) / v60)),
            }
        rec.update(extra or {})
        out[name] = rec
        print(name, json.dumps(rec), flush=True)

    record("P1", exp_family(P1, 40), exp_family(P1, 60), {"params": P1})
    record("P5", exp_family(P5, 40), exp_family(P5, 60), {"params": P5})
    closed_small = exp_family(SMALL, 40)
    literal_small, pieces = exp_family_literal(SMALL)
    with mp.workdps(40):
        out["closed_form_check"] = {
            "params": SMALL,
            "closed_40": mp.nstr(closed_small, 30),
            "literal_30": mp.nstr(literal_small, 30),
            "pieces": pieces,
            "relative": float(abs((closed_small - literal_small) / literal_small)),
        }
    print("closed_form_check", json.dumps(out["closed_form_check"]), flush=True)

    v30, n30, a_float = two_roots(30, mp.pi / 2)
    v45, n45, _ = two_roots(45, mp.pi / 4)
    record(
        "two_roots",
        v30,
        v45,
        {"params": dict(TWO_ROOTS, a=a_float), "pieces_30": n30, "pieces_45": n45},
    )

    for b in (1.0e4, 1.0e8):
        a = R.a_float(b, 1.0)
        R.WORKING_DPS = 40
        v40, _ = R.contour_C(b, a)
        R.WORKING_DPS = 60
        v60, _ = R.contour_C(b, a)
        R.WORKING_DPS = 40
        record(f"C_b{b:g}_lambda1", v40, v60, {"b": b, "a": a, "x_0": R.X0_FLOAT})

    record("two_phases", two_phases(40), two_phases(60), {"params": TWO_PHASES})

    for name, x_s in (("R_a", R.XS_FLOATS[1]), ("R_b", R.XS_FLOATS[0])):
        R.WORKING_DPS = 40
        v40 = R.closed_R(1.0e6, x_s, 0.0)
        contour = R.contour_R(1.0e6, x_s, 0.0)
        R.WORKING_DPS = 60
        v60 = R.closed_R(1.0e6, x_s, 0.0)
        R.WORKING_DPS = 40
        with mp.workdps(40):
            rel = float(abs((v40 - contour) / v40))
        record(
            name,
            v40,
            v60,
            {
                "c": 1.0e6,
                "x_s": x_s,
                "omega": 0.0,
                "closed_minus_contour_relative": rel,
            },
        )

    (prov.OUT / "test_references.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {prov.OUT / 'test_references.json'}")


if __name__ == "__main__":
    main()
