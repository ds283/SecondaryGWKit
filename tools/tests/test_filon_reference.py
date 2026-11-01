"""
The Filon reference tool, tools/filon_reference.py (prompts/levin-phase-floor/
06g-leaf-values-and-node-sum.md, section 2 test 8).

1. The closed forms: 06i's Gate R0 cells at omega = 1e6 and 1e9 (docs/levin-phase-floor/06i/
   scripts/r0_closed_forms.py), both implementations at both resolutions, within the tolerance 06i
   used, 1e-13 max|g| / omega.
2. The Ci closed form: Route A on a stand-in Bessel group whose three orders are 1/2, so that its
   phase is exactly K eta (06i's addition 2, docs/levin-phase-floor/06i/scripts/closed_form_a.py),
   against s_f A [Ci(|K| eta_b) - Ci(|K| eta_a)] in mpmath, with K's own rounding nonzero.
3. Route B's fit: a stand-in cosmological-shaped phase, Omega (e^x - 1), with a known smooth part
   and white jitter of 06i's size added to theta_mod_2pi, recovered within the declared fit-noise
   bound.
4. The count rule: four readings agreeing count; one perturbed past the tolerance does not.
5. Determinism: one reading bit-identical in a fresh process, and with two worker processes.

No Ray and no datastore.
"""

import math
import random
import subprocess
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import mpmath as mp

import tools.filon_reference as F

REPO = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------------------------------------------
# 1. 06i's Gate R0 cells: on [L, 2L], L = W 2 pi / omega,
#   (1) g(y) = y^3 exp(-y/L), against its antiderivative at 50 digits;
#   (2) g(y) = exp(i beta / y), beta = 4 L, against mpmath Gauss-Legendre at 50 digits.
R0_OMEGAS = (1e6, 1e9)
R0_W = {1: (1, 10, 1000, 100000), 2: (1, 10, 1000, 10000)}


def _r0_reference(test, omega, W):
    with mp.workdps(50):
        om = mp.mpf(omega)
        L = W * 2 * mp.pi / om
        if test == 1:
            b = mp.mpc(-1 / L, om)

            def A(y):
                return mp.exp(b * y) * mp.fsum(
                    (-1) ** k
                    * mp.factorial(3)
                    / mp.factorial(3 - k)
                    * y ** (3 - k)
                    / b ** (k + 1)
                    for k in range(4)
                )

            return +(A(2 * L) - A(L))
        beta = 4 * L
        pts = mp.linspace(L, 2 * L, max(2, int(W // 2) + 2))
        return mp.quad(
            lambda y: mp.expj(om * y + beta / y), pts, method="gauss-legendre"
        )


def _r0_cell(test, omega, W):
    """(g in double, L, R as mpf at the tool's precision, the tolerance)."""
    Ld = W * 2 * math.pi / omega
    if test == 1:
        gmax = (2 * Ld) ** 3 * math.exp(-2.0)

        def g(y):
            return (y**3) * math.exp(-y / Ld) + 0j

    else:
        beta = 4 * Ld
        gmax = 1.0

        def g(y):
            return complex(math.cos(beta / y), math.sin(beta / y))

    with mp.workdps(F.DPS):
        L = mp.mpf(W) * 2 * mp.pi / mp.mpf(omega)
        R = 2 * L
    return g, L, R, 1e-13 * gmax / omega


# ------------------------------------------------------------------------------------------------
# 2. A stand-in Bessel group whose three orders are 1/2: each residual vanishes and a_nu = 1, so the
# phase is exactly K eta (C = sum_i e_i (pi/4 - pi/4) = 0), and an amplitude constant in u = log eta
# gives I = s_f A int cos(K eta) / eta d eta = s_f A [Ci(|K| eta_b) - Ci(|K| eta_a)]. The coefficients
# are chosen so that K = m_1 + m_2 - m_3 is not a double: K's rounding, K_lo, is nonzero.
CI_COEFFICIENTS = (200000.123456789, 0.37, 0.61)
CI_SIGNS = (1.0, 1.0, -1.0)
CI_SPAN = (0.0, 1.0)
CI_SIGN_F = -1.0


class _HalfOrder:
    """A constituent of order 1/2: its residual vanishes."""

    nu = 0.5

    def residual(self, x, x_is_log=False):
        return 0.0


class _HalfOrderGroup:
    """A stand-in BesselPhaseGroup of three order-1/2 constituents."""

    def __init__(self, coefficients, signs):
        self.phases = (_HalfOrder(), _HalfOrder(), _HalfOrder())
        self.coefficients = tuple(coefficients)
        self.signs = tuple(signs)
        # K rounded once, as the group forms it
        self.K = math.fsum(s * c for s, c in zip(signs, coefficients))

    def residual(self, x):
        return 0.0

    def theta(self, log_x):
        return self.K * math.exp(log_x)


def _ci_amplitude():
    m = CI_COEFFICIENTS
    return (2 / math.pi) ** 1.5 / math.sqrt(m[0] * m[1] * m[2])


def ci_call():
    """The captured call of the Ci case (module level, so a spawned worker can rebuild it)."""
    group = _HalfOrderGroup(CI_COEFFICIENTS, CI_SIGNS)
    amplitude = CI_SIGN_F * _ci_amplitude()
    return {
        "x_span": CI_SPAN,
        "f": [lambda u: 0.0, lambda u: amplitude],
        "theta": {"theta": group.theta},
        "kw": {},
    }


def _ci_reference():
    K = sum(
        (Fraction(s) * Fraction(c) for s, c in zip(CI_SIGNS, CI_COEFFICIENTS)),
        Fraction(0),
    )
    with mp.workdps(50):
        Km = abs(mp.mpf(K.numerator) / K.denominator)
        # the amplitude the integrand carries, a double
        amplitude = mp.mpf(CI_SIGN_F * _ci_amplitude())
        eta_a, eta_b = mp.exp(mp.mpf(CI_SPAN[0])), mp.exp(mp.mpf(CI_SPAN[1]))
        value = amplitude * (mp.ci(Km * eta_b) - mp.ci(Km * eta_a))
    tolerance = 1e-13 * _ci_amplitude() / float(Km)
    return value, tolerance, K


# ------------------------------------------------------------------------------------------------
# 3. Route B on a stand-in cosmological-shaped phase, theta_s(x) = Omega (e^x - 1) on [0, 1.5] (a
# monotone phase whose rate grows exponentially, as a conformal time does in x = log(1+z)), with
# f = (e^-x, 0). theta_mod_2pi is theta_s plus white jitter of rms JITTER rad, reduced mod 2 pi in
# mpmath: 06i measured the cosmological groups' jitter at up to 2.2e-8 rad per segment (its log,
# "Route B's t(x)"). The raw theta only chooses the 2 pi branch. With t = e^x,
#   I = int_1^{e^1.5} sin(Omega (t - 1)) / t^2 dt = [G(t)],
#   G(t) = -sin(Omega t - Omega) / t + Omega (cos(Omega) Ci(Omega t) + sin(Omega) Si(Omega t)).
B_OMEGA = 2000.0
B_JITTER = 1.0e-8
B_SPAN = (0.0, 1.5)
B_SEGMENTS = 12


def _jitter(x):
    """A deterministic standard normal draw for each abscissa, seeded by its bits."""
    return random.Random(float(x).hex()).gauss(0.0, 1.0)


def _b_theta_mod_2pi(x):
    with mp.workdps(30):
        v = mp.mpf(B_OMEGA) * (mp.exp(mp.mpf(x)) - 1) + B_JITTER * _jitter(x)
        return float(v - 2 * mp.pi * mp.floor(v / (2 * mp.pi)))


def route_b_call():
    return {
        "x_span": B_SPAN,
        "f": [lambda x: math.exp(-x), lambda x: 0.0],
        "theta": {
            "theta": lambda x: B_OMEGA * (math.exp(x) - 1.0),
            "theta_mod_2pi": _b_theta_mod_2pi,
            "theta_deriv": lambda x: B_OMEGA * math.exp(x),
        },
        "kw": {},
    }


def _b_edges():
    a, b = B_SPAN
    return [a + (b - a) * j / B_SEGMENTS for j in range(B_SEGMENTS + 1)]


def _b_reference():
    with mp.workdps(50):
        W = mp.mpf(B_OMEGA)

        def G(t):
            return -mp.sin(W * t - W) / t + W * (
                mp.cos(W) * mp.ci(W * t) + mp.sin(W) * mp.si(W * t)
            )

        return G(mp.exp(mp.mpf(B_SPAN[1]))) - G(mp.exp(mp.mpf(B_SPAN[0])))


def _err(reading, reference):
    with mp.workdps(50):
        return float(abs(mp.mpf(reading["value_30"]) - reference))


class TestFilonReference(unittest.TestCase):

    def test_closed_forms(self):
        """
        06i's R0 cells at omega = 1e6 and 1e9, 16 in all, by both implementations at both
        resolutions: each reading within 1e-13 max|g| / omega of the closed form (06i's cell
        criterion). 06i's worst over its 32 cells was 0.088 of the tolerance (its verification.md,
        R0.1).
        """
        worst = 0.0
        for test in (1, 2):
            for omega in R0_OMEGAS:
                for W in R0_W[test]:
                    ref = _r0_reference(test, omega, W)
                    g, L, R, tol = _r0_cell(test, omega, W)
                    for impl in F.IMPLEMENTATIONS:
                        for res in F.RESOLUTIONS:
                            reading = F.linear_phase_integral(g, L, R, omega, impl, res)
                            with mp.workdps(50):
                                err = float(abs(reading["value_mp"] - ref))
                            worst = max(worst, err / tol)
                            self.assertLessEqual(
                                err, tol, f"cell ({test}, {omega:g}, {W}) {impl} {res}"
                            )
        self.assertLess(worst, 0.1)

    def test_ci_closed_form(self):
        """
        Route A on the all-orders-1/2 stand-in, against s_f A [Ci(|K| eta_b) - Ci(|K| eta_a)] with
        K exact: every reading within 1e-13 A / |K| (the same criterion, max|g| = A at eta = 1).
        K's rounding is nonzero here, so a Route A that took K rounded once (the group's own K)
        would be in error by about |K_lo| A / |K|, about 90 times the tolerance.
        """
        call = ci_call()
        self.assertEqual(F.route_of(call), "A")
        reference, tolerance, K_exact = _ci_reference()
        group = call["theta"]["theta"].__self__
        K_lo = float(K_exact - Fraction(group.K))
        self.assertNotEqual(K_lo, 0.0)
        for impl in F.IMPLEMENTATIONS:
            for res in F.RESOLUTIONS:
                reading = F.reading(call, impl, res)
                self.assertEqual(reading["route"], "A")
                self.assertLessEqual(
                    _err(reading, reference), tolerance, f"{impl} {res}"
                )
                self.assertEqual(reading["K_lo"], K_lo)

    def test_route_b_fit(self):
        """
        Route B on the stand-in cosmological phase with jitter of rms 1e-8 rad: every segment
        monotone; the fit's residual rms, sigma, within a factor 2 of the jitter injected; each
        reading within three times the declared 1-sigma fit-noise bound of the smooth phase's
        integral (the FCC readings carry the bound; a QAWO reading uses the same fits, and is held
        to its resolution's FCC bound).
        """
        call = route_b_call()
        self.assertEqual(F.route_of(call), "B")
        reference = _b_reference()
        bounds = {}
        readings = {}
        for impl in ("fcc", "qawo"):
            for res in F.RESOLUTIONS:
                reading = F.reading(call, impl, res, edges=_b_edges())
                readings[(impl, res)] = reading
                self.assertTrue(reading["tmap"]["all_monotone"])
                self.assertEqual(reading["n_segments"], B_SEGMENTS)
                for segment in reading["segments"]:
                    self.assertGreater(segment["sigma"], 0.5 * B_JITTER)
                    self.assertLess(segment["sigma"], 2.0 * B_JITTER)
                if impl == "fcc":
                    bounds[res] = reading["tmap"]["fit_noise_1sigma"]
                    self.assertGreater(bounds[res], 0.0)
        for (impl, res), reading in readings.items():
            self.assertLessEqual(
                _err(reading, reference), 3.0 * bounds[res], f"{impl} {res}"
            )

    def test_count_rule(self):
        """Four readings within the tolerance count; one moved past it does not."""
        reference, tolerance, _ = _ci_reference()
        call = ci_call()
        readings = [
            F.reading(call, impl, res)
            for impl in F.IMPLEMENTATIONS
            for res in F.RESOLUTIONS
        ]
        self.assertTrue(F.counts(readings, tolerance))
        self.assertLessEqual(F.scatter(readings), tolerance)
        moved = [dict(r) for r in readings]
        with mp.workdps(F.DPS):
            moved[1]["value_30"] = mp.nstr(
                mp.mpf(moved[1]["value_30"]) + 2 * tolerance, 30
            )
        self.assertFalse(F.counts(moved, tolerance))
        # the whole reference applies the same rule
        result = F.reference(call, tolerance)
        self.assertTrue(result["counts"])
        self.assertEqual(len(result["readings"]), 4)
        fcc2 = [r for r in result["readings"] if r["impl"] == "fcc" and r["res"] == 2][
            0
        ]
        self.assertEqual(result["value_hex"], fcc2["value_hex"])

    def test_determinism(self):
        """One QAWO reading of the Ci case: bit-identical in a fresh process, and with two
        spawned workers."""
        call = ci_call()
        here = F.reading(call, "qawo", 1)
        code = (
            "import tools.filon_reference as F, tools.tests.test_filon_reference as T;"
            "r = F.reading(T.ci_call(), 'qawo', 1);"
            "print(r['value_hex'], r['value_30'], r['imag_30'])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO,
            env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        self.assertEqual(out, [here["value_hex"], here["value_30"], here["imag_30"]])
        pooled = F.reading(
            call,
            "qawo",
            1,
            workers=2,
            call_source=("tools.tests.test_filon_reference", "ci_call", ()),
        )
        self.assertEqual(
            (pooled["value_hex"], pooled["value_30"], pooled["imag_30"]),
            (here["value_hex"], here["value_30"], here["imag_30"]),
        )


if __name__ == "__main__":
    unittest.main()
