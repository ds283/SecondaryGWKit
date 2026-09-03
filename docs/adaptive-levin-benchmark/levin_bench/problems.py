"""
Synthetic oscillatory test problems for the AdaptiveLevin campaign.

Every problem has a closed-form oracle, evaluated with mpmath at 60 decimal
digits.  This matters at the top of the frequency ladder: at omega = 1e12 the
double-precision value of cos(omega) carries no significant digits, so a
float-arithmetic "exact" formula would itself be the error floor.

Each problem exposes

  x_span        (a, b)
  f             [f_sin, f_cos] amplitudes, so that the integral is
                \\int f_sin(x) sin(theta(x)) + f_cos(x) cos(theta(x)) dx
  theta         raw (unwrapped) phase
  theta_mod_2pi phase reduced mod 2pi via the repository's prime-factor
                range reduction, so that the phase stays accurate at large
                argument
  theta_deriv   analytic d(theta)/dx
  integrand     the raw oscillatory integrand, for the brute-force runners
  qawo          (amplitude, 'sin'|'cos', wvar) when the phase is linear in x
                and QUADPACK's oscillatory rule applies, else None
  reference     the closed-form value as a Python float
"""

import sys
from math import atan, cos, exp, pi, sin

import numpy as np

REPO = "/Users/ds283/Documents/Code/SecondaryGWKit"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import mpmath as mp
from LiouvilleGreen.range_reduce_mod_2pi import range_reduce_mod_2pi

mp.mp.dps = 60

NEAR_SINGULAR_C = 1.0e-3


class Problem:
    tier = "A"

    def __init__(self, omega):
        self.omega = float(omega)

    @property
    def name(self):
        return self._name

    @property
    def qawo(self):
        return None

    @property
    def n_oscillations(self):
        raise NotImplementedError


class DampedSine(Problem):
    """\\int_0^1 e^{-x} sin(omega x) dx -- smooth amplitude, linear phase."""

    _name = "damped_sine"
    x_span = (0.0, 1.0)

    def __init__(self, omega):
        super().__init__(omega)
        w = self.omega
        self.f = [lambda x: exp(-x), lambda x: 0.0]
        self.theta = lambda x: w * x
        self.theta_mod_2pi = lambda x: range_reduce_mod_2pi(w, x)[1]
        self.theta_deriv = lambda x: w
        self.integrand = lambda x: exp(-x) * sin(w * x)

        W = mp.mpf(w)
        self.reference = float(
            (W - mp.e ** (-1) * (mp.sin(W) + W * mp.cos(W))) / (1 + W * W)
        )

    @property
    def qawo(self):
        return (lambda x: exp(-x), "sin", self.omega)

    @property
    def n_oscillations(self):
        return self.omega / (2.0 * pi)


class Sinc(Problem):
    """\\int_1^100 sin(omega x)/x dx -- oracle is a difference of Si."""

    _name = "sinc"
    x_span = (1.0, 100.0)

    def __init__(self, omega):
        super().__init__(omega)
        w = self.omega
        self.f = [lambda x: 1.0 / x, lambda x: 0.0]
        self.theta = lambda x: w * x
        self.theta_mod_2pi = lambda x: range_reduce_mod_2pi(w, x)[1]
        self.theta_deriv = lambda x: w
        self.integrand = lambda x: sin(w * x) / x

        W = mp.mpf(w)
        self.reference = float(mp.si(100 * W) - mp.si(W))

    @property
    def qawo(self):
        return (lambda x: 1.0 / x, "sin", self.omega)

    @property
    def n_oscillations(self):
        return self.omega * 99.0 / (2.0 * pi)


class GRZ(Problem):
    """
    \\int_{-1}^{+1} cos(lambda arctan x)/(1+x^2) dx = (2/lambda) sin(pi lambda/4).

    Gradshteyn & Ryzhik; this is the repository's own fourth unit test, here
    driven far beyond lambda = 1000.  Nonlinear phase, so QAWO does not apply.
    """

    _name = "grz"
    x_span = (-1.0, 1.0)

    def __init__(self, omega):
        super().__init__(omega)
        w = self.omega
        self.f = [lambda x: 0.0, lambda x: 1.0 / (1.0 + x * x)]
        self.theta = lambda x: w * atan(x)
        self.theta_mod_2pi = lambda x: range_reduce_mod_2pi(w, atan(x))[1]
        self.theta_deriv = lambda x: w / (1.0 + x * x)
        self.integrand = lambda x: cos(w * atan(x)) / (1.0 + x * x)

        W = mp.mpf(w)
        self.reference = float((2 / W) * mp.sin(mp.pi * W / 4))

    @property
    def n_oscillations(self):
        return self.omega / 4.0


class Chirp(Problem):
    """
    \\int_0^1 x sin(omega x^2) dx = (1 - cos omega)/(2 omega).

    Quadratic phase: the local frequency sweeps from zero at x = 0 up to
    2 omega at x = 1, so a single interval never has a uniform oscillation
    rate and the leftmost region is genuinely non-oscillatory.
    """

    _name = "chirp"
    x_span = (0.0, 1.0)

    def __init__(self, omega):
        super().__init__(omega)
        w = self.omega
        self.f = [lambda x: x, lambda x: 0.0]
        self.theta = lambda x: w * x * x
        self.theta_mod_2pi = lambda x: range_reduce_mod_2pi(w, x * x)[1]
        self.theta_deriv = lambda x: 2.0 * w * x
        self.integrand = lambda x: x * sin(w * x * x)

        W = mp.mpf(w)
        self.reference = float((1 - mp.cos(W)) / (2 * W))

    @property
    def n_oscillations(self):
        return self.omega / (2.0 * pi)


class NearSingular(Problem):
    """
    \\int_0^1 sin(omega x)/(x + c) dx with c = 1e-3.

    The amplitude has a pole just outside the interval and varies over three
    decades across it, which is the direct test of the claim that Levin cost is
    set by the smoothness of the amplitude rather than by the oscillation
    count.  Oracle by substitution u = x + c:

      cos(omega c)[Si(omega(1+c)) - Si(omega c)]
        - sin(omega c)[Ci(omega(1+c)) - Ci(omega c)]
    """

    _name = "near_singular"
    x_span = (0.0, 1.0)

    def __init__(self, omega):
        super().__init__(omega)
        w = self.omega
        c = NEAR_SINGULAR_C
        self.c = c
        self.f = [lambda x: 1.0 / (x + c), lambda x: 0.0]
        self.theta = lambda x: w * x
        self.theta_mod_2pi = lambda x: range_reduce_mod_2pi(w, x)[1]
        self.theta_deriv = lambda x: w
        self.integrand = lambda x: sin(w * x) / (x + c)

        W, C = mp.mpf(w), mp.mpf(c)
        self.reference = float(
            mp.cos(W * C) * (mp.si(W * (1 + C)) - mp.si(W * C))
            - mp.sin(W * C) * (mp.ci(W * (1 + C)) - mp.ci(W * C))
        )

    @property
    def qawo(self):
        return (lambda x, c=NEAR_SINGULAR_C: 1.0 / (x + c), "sin", self.omega)

    @property
    def n_oscillations(self):
        return self.omega / (2.0 * pi)


PROBLEM_CLASSES = [DampedSine, Sinc, GRZ, Chirp, NearSingular]
PROBLEMS_BY_NAME = {cls._name: cls for cls in PROBLEM_CLASSES}


# ---------------------------------------------------------------------------
# Frequency ladder
# ---------------------------------------------------------------------------
#
# All five oracles decay like 1/omega, and each has zeros: GRZ vanishes
# identically whenever lambda is a multiple of 4, Chirp whenever cos(omega) = 1,
# Sinc wherever the two Si terms happen to cancel.  Landing on one of those
# makes a *relative* error meaningless -- the repository's own lambda = 1000
# test case is exactly such a point, with an oracle of 0 and a computed value of
# 1e-13.  So within each decade we scan candidate frequencies and take the first
# whose scaled oracle |I| * omega exceeds a threshold, which keeps the oracle
# O(1/omega) and the relative error interpretable.  The choice is deterministic
# and independent of any integrator.

_SCALED_ORACLE_FLOOR = 0.3


def select_omega(cls, decade, floor=_SCALED_ORACLE_FLOOR, n_candidates=400):
    """
    Return a frequency in [10**decade, 10**(decade+1)) whose oracle is
    well scaled, i.e. |reference| * omega >= floor.
    """
    base = 10.0**decade
    for i in range(n_candidates):
        w = base * (1.0 + 9.0 * i / n_candidates)
        if cls is GRZ:
            # sin(pi*lambda/4) = 1 exactly when lambda = 2 (mod 8)
            w = 8.0 * round((w - 2.0) / 8.0) + 2.0
            if w < base:
                continue
        try:
            p = cls(w)
        except Exception:
            continue
        if abs(p.reference) * w >= floor:
            return w
    raise RuntimeError(f"no well-scaled frequency found for {cls._name} in decade {decade}")


def omega_ladder(cls, decades):
    seen, out = set(), []
    for d in decades:
        w = select_omega(cls, d)
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def validate_oracles(omega=13.0):
    """
    Cross-check every closed-form oracle against direct mpmath quadrature of
    the integrand at 60 digits.  Returns a list of (name, closed_form,
    mpmath_quad, rel_diff).
    """
    out = []
    for cls in PROBLEM_CLASSES:
        p = cls(omega)
        W = mp.mpf(omega)
        a, b = mp.mpf(p.x_span[0]), mp.mpf(p.x_span[1])

        if cls is DampedSine:
            g = lambda x: mp.e ** (-x) * mp.sin(W * x)
        elif cls is Sinc:
            g = lambda x: mp.sin(W * x) / x
        elif cls is GRZ:
            g = lambda x: mp.cos(W * mp.atan(x)) / (1 + x * x)
        elif cls is Chirp:
            g = lambda x: x * mp.sin(W * x * x)
        else:
            C = mp.mpf(p.c)
            g = lambda x: mp.sin(W * x) / (x + C)

        # integrate over half-period panels so that mpmath is never asked to
        # resolve many oscillations inside a single panel
        n_panels = max(8, int(4 * p.n_oscillations))
        edges = [a + (b - a) * mp.mpf(i) / n_panels for i in range(n_panels + 1)]
        ref = mp.fsum(
            mp.quad(g, [edges[i], edges[i + 1]]) for i in range(n_panels)
        )
        rel = abs(mp.mpf(p.reference) - ref) / abs(ref)
        out.append((p.name, p.reference, float(ref), float(rel)))
    return out


if __name__ == "__main__":
    print(f"{'problem':>14s} {'closed form':>24s} {'mpmath quad':>24s} {'rel diff':>10s}")
    for name, cf, num, rel in validate_oracles():
        print(f"{name:>14s} {cf:>24.16e} {num:>24.16e} {rel:>10.2e}")
