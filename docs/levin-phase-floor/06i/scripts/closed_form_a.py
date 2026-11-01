"""levin-phase-floor prompt 06i, addition 2 (the user's ruling, 2026-11-01): the closed form for the
analytic blocks whose three orders are all 1/2 (samples 8, 10, 22, 23, 25). Descriptive. New in 06i.

For nu = 1/2 the Bessel phase residual vanishes and a_nu = 1 (J_1/2 = sqrt(2/(pi x)) sin x), so
the Levin amplitude of _three_bessel_Levin is
    Levin_f(u) = eta^(3/2 - b) prod_i sqrt(2/(pi m_i eta)) = A eta^(-b),  A = (2/pi)^(3/2) / sqrt(m_1 m_2 m_3),
with m_i the group's coefficients (k, q cs, r cs). On these five calls b = 0 (read from Levin_f's
closure), the amplitude occupies the cos slot with sign s_f (f_sin == 0, f_cos == s_f Levin_f, s_f
read off the captured callable and checked to be +-1 to rounding at 9 points), and
C = sum_i e_i (pi/4 - pi nu_i/2) = 0. The phase is K eta with K = sum_i e_i m_i exactly. So

    I = int_{u_a}^{u_b} s_f A cos(K e^u) du = s_f A int_{eta_a}^{eta_b} cos(K eta)/eta d eta
      = s_f A [Ci(|K| eta_b) - Ci(|K| eta_a)],   eta = e^u exactly,

evaluated in mpmath at 50 digits (mp.ci). Nothing of Route A's machinery is used: not filon.py,
not samples.RouteA, not the group's residual or its theta.
Check first, on sample 8: the residual is identically 0 at 9 points; and an mpmath.quad spot check
of s_f A cos(K eta)/eta over a short sub-span (the first ~100 wavelengths above eta_a, and 100
wavelengths about the geometric middle) against the Ci difference over the same sub-span.
Usage (census tree as cwd): python -B closed_form_a.py
"""

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prov  # noqa: E402

MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
INPUTS = MAIN / "var/levin-phase-floor-06i/inputs"
OUT = Path(__file__).resolve().parent.parent / "out"
SAMPLES = (8, 10, 22, 23, 25)


def load(n):
    import ray.cloudpickle as cp

    return cp.loads((INPUTS / f"s{n:02d}.pkl").read_bytes())


def one(n, spot):
    rec = load(n)
    g = rec["theta"]["theta"].__self__
    f_sin, f_cos = rec["f"]
    u_a, u_b = (float(v) for v in rec["x_span"])
    assert all(p.nu == 0.5 for p in g.phases)
    # b from Levin_f's closure
    lf = dict(
        zip(f_cos.__code__.co_freevars, [c.cell_contents for c in f_cos.__closure__])
    )["Levin_f"]
    b = dict(zip(lf.__code__.co_freevars, [c.cell_contents for c in lf.__closure__]))[
        "b"
    ]
    assert b == 0.0
    m = [mp.mpf(c) for c in g.coefficients]
    A = (2 / mp.pi) ** mp.mpf(1.5) / mp.sqrt(m[0] * m[1] * m[2])
    Kf = sum(
        (Fraction(s) * Fraction(c) for s, c in zip(g.signs, g.coefficients)),
        Fraction(0),
    )
    K = mp.mpf(Kf.numerator) / Kf.denominator
    C = mp.fsum(
        s * (mp.pi / 4 - mp.pi * mp.mpf(p.nu) / 2) for s, p in zip(g.signs, g.phases)
    )
    assert C == 0
    pts = [u_a + (u_b - u_a) * j / 8 for j in range(9)]
    ratios = [f_cos(u) / float(A) for u in pts]
    s_f = round(ratios[0])
    assert s_f in (1, -1)
    max_dev = max(abs(r - s_f) for r in ratios)
    sin_slot = max(abs(f_sin(u)) for u in pts)
    resid = max(abs(g.residual(math.exp(u))) for u in pts)
    eta_a, eta_b = mp.exp(mp.mpf(u_a)), mp.exp(mp.mpf(u_b))
    aK = abs(K)

    def I_ci(l, r):
        return s_f * A * (mp.ci(aK * r) - mp.ci(aK * l))

    I = I_ci(eta_a, eta_b)
    out = dict(
        sample=n,
        b=b,
        slot="cos",
        s_f=s_f,
        amplitude_A=mp.nstr(A, 30),
        captured_amplitude_over_A_max_dev=max_dev,
        f_sin_max=sin_slot,
        residual_max_at_9_points=resid,
        K=str(Kf),
        C=mp.nstr(C, 5),
        eta_a=mp.nstr(eta_a, 30),
        eta_b=mp.nstr(eta_b, 30),
        value_30=mp.nstr(I, 35),
        value_hex=float(I).hex(),
    )
    if spot:
        lam = 2 * mp.pi / aK
        checks = []
        for l in (eta_a, mp.sqrt(eta_a * eta_b)):
            r = l + 100 * lam
            q = mp.quad(
                lambda e: s_f * A * mp.cos(aK * e) / e,
                mp.linspace(l, r, 401),
                method="gauss-legendre",
            )
            c = I_ci(l, r)
            checks.append(
                dict(
                    sub_span=[mp.nstr(l, 20), mp.nstr(r, 20)],
                    quad=mp.nstr(q, 30),
                    ci=mp.nstr(c, 30),
                    rel_diff=float(abs(q - c) / abs(c)),
                )
            )
        out["spot_checks"] = checks
    return out


def main():
    mp.mp.dps = 50
    res = [one(n, spot=(n == 8)) for n in SAMPLES]
    out = {
        "header": prov.header([INPUTS / f"s{n:02d}.pkl" for n in SAMPLES]),
        "closed_form": "I = s_f A [Ci(|K| e^{u_b}) - Ci(|K| e^{u_a})], A = (2/pi)^(3/2)/sqrt(m1 m2 m3)",
        "samples": res,
    }
    (OUT / "closed_form_a.json").write_text(json.dumps(out, indent=1))
    for r in res:
        print(
            r["sample"],
            r["value_30"],
            r["value_hex"],
            "s_f",
            r["s_f"],
            "amp dev",
            r["captured_amplitude_over_A_max_dev"],
            "R max",
            r["residual_max_at_9_points"],
            r.get("spot_checks"),
        )


if __name__ == "__main__":
    main()
