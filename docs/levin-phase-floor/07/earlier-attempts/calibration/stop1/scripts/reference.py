"""levin-phase-floor 07 section 2: references for the two families, at 30 digits.

On x in [0, 1], with f_sin = exp(-x) and f_cos = cos(3x), the integral is

    I = int_0^1 [ exp(-x) sin theta(x) + cos(3x) cos theta(x) ] dx
      = Im J(-1) + Re [ J(3i) + J(-3i) ] / 2,      J(alpha) = int_0^1 exp(alpha x) exp(i theta(x)) dx,

since exp(-x) and cos(3x) are real on the real axis.

**The prompt's reference** is ``mpmath.quad`` at 30 digits, split at every zero of theta' and at
x_0. As written it does not converge on these cases (``time_reference.py``: mpmath's own error
estimates 0.11 and 0.2 on the largest); made to converge by subdividing into pieces of at most pi
of phase it is ``literal()`` here, which projects to 8 h and 3 h on the largest case of each family.

**Proposed in its place, not yet approved by the user** (the orchestrator's item 3):
- (R), theta = (c/2)(x - x_s)^2: ``closed_R()``, J(alpha) in closed form by completing the square,
      J = exp(alpha x_s) exp(i alpha^2 / 2c) sqrt(pi) / (2 sqrt(beta))
          [ erf(sqrt(beta)(u_1 - i alpha/c)) - erf(sqrt(beta)(u_0 - i alpha/c)) ],
  beta = -i c/2, u_0 = -x_s, u_1 = 1 - x_s, with mpmath's erf of a complex argument.
- (C), theta = a u + (b/3) u^3, u = x - x_0: ``contour_C()``, Cauchy's theorem. The integrand
  exp(alpha u) exp(i theta(u)) is entire, so the segment [u_0, u_1] may be replaced by the
  rectangle u_0 -> u_0 + i s -> u_1 + i s -> u_1, with s = sqrt(a/b), the height of the upper
  saddle point u = i s of theta. On it Im theta >= 0 everywhere, so |exp(i theta)| <= 1:
  - on the verticals u = u_j + i y, 0 <= y <= s, Im theta = y (a + b u_j^2 - b y^2/3), increasing
    in y, so the integrand decays smoothly from the real axis;
  - on the horizontal u = t + i s, Im theta = (2/3) lambda + b s t^2 and Re theta = b t^3 / 3, a
    Gaussian in t of width 1/sqrt(2 b s) with at most about 300/sqrt(lambda) rad of phase across
    it. Where b s t^2 > ``HORIZONTAL_CUTOFF`` the integrand is below exp(-HORIZONTAL_CUTOFF), and
    that part of the horizontal is not integrated; its bound is returned and is below 1e-60.
  Each segment is integrated by ``mpmath.quad`` at 40 digits, the verticals split geometrically
  from the decay length 1/(a + b u_j^2), the horizontal at t^3 = 3 k pi / b.

**Parameters are the exact doubles** the integrator is handed: x_s, x_0, c, b and a are passed
as Python floats and converted to mpf exactly (``mp.mpf(float)``), never re-read from decimal
text, so the reference integrates the very integrand the Levin call does. ``cases()`` defines
them once for every script; a is the double nearest the cube root of lambda^2 b, and the
lambda actually integrated, a^(3/2)/b^(1/2) of those doubles, is reported beside the nominal one.

Every function returns an ``mpmath.mpf`` at the working precision; callers report 30 digits.
"""

import mpmath as mp

WORKING_DPS = 40
REPORT_DPS = 30
X0_FLOAT = 0.3819660112501051
X0 = mp.mpf(X0_FLOAT)
XS_FLOATS = (0.3819660112501051, 0.7236067977499790)
C_VALUES = (1.0e4, 1.0e6, 1.0e8)
B_VALUES = (1.0e4, 1.0e8)
LAMBDAS = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0)


def a_float(b, lam):
    """The double nearest (lam^2 b)^(1/3)."""
    with mp.workdps(WORKING_DPS):
        return float(mp.cbrt(mp.mpf(lam) ** 2 * mp.mpf(b)))


def lambda_actual(b, a):
    with mp.workdps(WORKING_DPS):
        return float(mp.mpf(a) ** 1.5 / mp.sqrt(mp.mpf(b)))


def cases():
    """Every case of section 2, as dicts of exact doubles."""
    out = []
    for c in C_VALUES:
        for x_s in XS_FLOATS:
            out.append({"family": "R", "c": c, "x_s": x_s})
    for b in B_VALUES:
        for lam in LAMBDAS:
            a = a_float(b, lam)
            out.append(
                {
                    "family": "C",
                    "b": b,
                    "lambda": lam,
                    "a": a,
                    "x_0": X0_FLOAT,
                    "lambda_actual": lambda_actual(b, a),
                }
            )
    return out


HORIZONTAL_CUTOFF = 160


def _combine(J):
    """I = Im J(-1) + Re (J(3i) + J(-3i)) / 2, for J a function of alpha."""
    return mp.im(J(mp.mpf(-1))) + mp.re(J(mp.mpc(0, 3)) + J(mp.mpc(0, -3))) / 2


def closed_R(c, x_s):
    with mp.workdps(WORKING_DPS):
        c = mp.mpf(float(c))
        x_s = mp.mpf(float(x_s))
        beta = mp.mpc(0, -1) * c / 2
        sb = mp.sqrt(beta)
        u0, u1 = -x_s, 1 - x_s

        def J(alpha):
            shift = mp.mpc(0, 1) * alpha / c
            pref = (
                mp.exp(alpha * x_s)
                * mp.exp(mp.mpc(0, 1) * alpha**2 / (2 * c))
                * mp.sqrt(mp.pi)
                / (2 * sb)
            )
            return pref * (mp.erf(sb * (u1 - shift)) - mp.erf(sb * (u0 - shift)))

        return +_combine(J)


def contour_C(b, a):
    """Returns (value, bound on the omitted part of the horizontal). b and a are doubles."""
    with mp.workdps(WORKING_DPS):
        b = mp.mpf(float(b))
        a = mp.mpf(float(a))
        lam = a**1.5 / mp.sqrt(b)
        s = mp.sqrt(a / b)
        u0, u1 = -X0, 1 - X0
        I = mp.mpc(0, 1)

        def theta(u):
            return a * u + b / 3 * u**3

        def vertical(alpha, uj):
            kappa = a + b * uj**2
            pts = [mp.mpf(0)]
            y = 1 / kappa
            while y < s:
                pts.append(y)
                y *= 4
            pts.append(s)
            return mp.quad(
                lambda y: mp.exp(alpha * (uj + I * y)) * mp.exp(I * theta(uj + I * y)),
                pts,
            )

        T = mp.sqrt(HORIZONTAL_CUTOFF / (b * s))
        lo, hi = max(u0, -T), min(u1, T)
        pts = [lo]
        k = 1
        while True:
            t = mp.cbrt(3 * k * mp.pi / b)
            if t >= max(-lo, hi):
                break
            if -t > lo:
                pts.append(-t)
            if t < hi:
                pts.append(t)
            k += 1
        if lo < 0 < hi:
            pts.append(mp.mpf(0))
        pts = sorted(set(pts) | {hi})
        omitted = (
            mp.exp(-HORIZONTAL_CUTOFF - mp.mpf(2) / 3 * lam)
            * mp.exp(3 * s + 1)
            * ((lo - u0) + (u1 - hi))
        )

        def horizontal(alpha):
            return mp.quad(
                lambda t: mp.exp(alpha * (t + I * s)) * mp.exp(I * theta(t + I * s)),
                pts,
            )

        def J(alpha):
            return mp.exp(alpha * X0) * (
                I * vertical(alpha, u0) + horizontal(alpha) - I * vertical(alpha, u1)
            )

        return +_combine(J), omitted


def literal(theta, rate, split, piece_phase=mp.pi):
    """
    The prompt's reference made to converge: mpmath.quad at 30 digits on consecutive pieces of at
    most piece_phase of phase, between consecutive points of ``split`` (every zero of theta', and
    x_0). theta and rate take and return mpf. Practical only where the total variation is small.
    """
    with mp.workdps(REPORT_DPS):

        def g(x):
            t = theta(x)
            return mp.exp(-x) * mp.sin(t) + mp.cos(3 * x) * mp.cos(t)

        total = mp.mpf(0)
        pieces = 0
        for lo, hi in zip(split[:-1], split[1:]):
            lo, hi = mp.mpf(lo), mp.mpf(hi)
            # march from the end where |theta'| is smallest
            x = lo
            while x < hi:
                # a first guess from the local rate (at least one piece per unit of x), then
                # halved until the piece holds no more than piece_phase
                step = piece_phase / max(abs(rate(x)), piece_phase)
                nxt = min(x + step, hi)
                # never let one piece hold more than piece_phase: shrink until it does not
                while abs(theta(nxt) - theta(x)) > piece_phase and nxt - x > 0:
                    nxt = x + (nxt - x) / 2
                total += mp.quad(g, [x, nxt])
                pieces += 1
                x = nxt
        return total, pieces


def family_R(c, x_s):
    c, x_s = mp.mpf(float(c)), mp.mpf(float(x_s))
    return (
        lambda x: c / 2 * (x - x_s) ** 2,
        lambda x: c * (x - x_s),
        [0, x_s, 1],
    )


def family_C(b, a):
    b, a = mp.mpf(float(b)), mp.mpf(float(a))
    return (
        lambda x: a * (x - X0) + b / 3 * (x - X0) ** 3,
        lambda x: a + b * (x - X0) ** 2,
        [0, X0, 1],
    )
