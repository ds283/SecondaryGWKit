"""
levin-phase-floor prompt 04, step 4: the three arms of prompt section 2, as one dispatcher patched
at the class attribute ``ComputeTargets.primitive_phase.PrimitivePhase.theta_mod_2pi``; the exact
interval accessor Delta^ext; the magnitudes Phi^arm_i; and the predicted floor F^arm_g.

Nothing here changes production code. ``install(arm)`` replaces the class attribute for the
duration of a ``with`` block and restores the original in a ``finally``. ``raw_theta``,
``theta_deriv``, the amplitudes, the partition, the driver and every tolerance are production's.
``QuadSourceIntegral._ClampedPhase`` calls through to the method, so its clamp applies unchanged.

Definitions (prompt section 2)
------------------------------
For a factor i (a ``PrimitivePhase``): sign s_i, wavenumber k_i, leading primitive's
``CumulativeTable`` (``leading.table``), anchor a_i, residual spline phi_i(u), u = log(1+z).
At an abscissa x (log(1+z'), already clamped by ``_ClampedPhase`` where production clamps), z is
``expm1(x)`` and u is ``PrimitivePhase._clamped_u(z, x)``, exactly as ``raw_theta`` forms them.

- **Delta^ext(z_a, z_b)**: ``CumulativeTable.delta`` re-formed in mpmath (``mp.prec`` = 512 bits,
  about 154 decimal digits): the node limbs summed exactly, (hi_b - hi_a) + (lo_b - lo_a), plus
  the two end partials p_b - p_a, which are production's own floats from ``_integral`` at
  production's node choice (``_locate``). The result is exact (the inputs are a handful of doubles
  and 512 bits covers their exponent span; ``check_exact`` verifies it against ``fractions``).
- **Reduction**: against the double constant ``TWO_PI`` taken exactly, into production's
  (-TWO_PI, 0] convention, then rounded to a double once (round-to-nearest).
- **P**: the original method, called through the patch in pass-through mode.
- **X**: rho^X_i = (s_i k_i Delta^ext(z, a_i) + phi_i(u)) mod TWO_PI, every factor.
- **A**: for G only (sign -1, leading ``tau``):
  rho^A_G = WKB_mod_2pi(s k Delta_tau(z, z_top) + phi_G(u))[1] + c_G, with Delta_tau production's
  ``delta`` (double), z_top the node of the tau table at or nearest above ``max_x`` (the top of
  G's phi samples), and c_G the exact remainder of s k Delta^ext(z_top, z_response), computed once
  per G object. Not re-reduced: the sum lies in (-4 pi, 0]. T's remainders are production's.
- **Classification**: G is a ``PrimitivePhase`` with sign -1 whose leading primitive is labelled
  ``tau``; T one with sign +1 and leading ``cs_tau``. Anything else (none is expected at 98e68d3)
  passes through unchanged and is counted as ``other``. Every call is counted by kind.
- **Phi^P_i(z)** = |theta_i(z)| (production's double ``raw_theta``);
  **Phi^A_G(z)** = |s k Delta_tau(z, z_top) + phi_G(u)|, Phi^A_T = Phi^P_T;
  **Phi^X_i(z)** = k_i |p_i(z)| + 2 pi, p_i = p_b - p_a, the end partials of Delta^ext(z, a_i).
- **F^arm_g** = EPS x max over the group's sub-interval of sum_{i in g} Phi^arm_i, sampled at
  1000 points equispaced in log(1+z') (ends included), at the abscissae production would pass to
  each factor (after the clamp). EPS = numpy float64 eps, the one 01 and 03 used for F.
"""

import contextlib
import math
import time
from fractions import Fraction

import mpmath
import numpy as np
from mpmath.libmp import round_nearest, to_float

from ComputeTargets.primitive_phase import PrimitivePhase
from LiouvilleGreen.WKBtools import WKB_mod_2pi
from LiouvilleGreen.constants import TWO_PI

MP = mpmath.MPContext()
MP.prec = 512
TWO_PI_MP = MP.mpf(TWO_PI)  # the double constant, exactly
EPS = float(np.finfo(np.float64).eps)
F_SAMPLES = 1000

ORIG_THETA_MOD_2PI = PrimitivePhase.theta_mod_2pi


def mp_to_float(x) -> float:
    """Round an mpf to the nearest double, once."""
    return to_float(x._mpf_, rnd=round_nearest)


def mp_fraction(x) -> Fraction:
    sign, man, exp, _bc = x._mpf_
    if man == 0:
        return Fraction(0)
    v = Fraction(man) * (Fraction(2) ** exp)
    return -v if sign else v


# ------------------------------------------------------------------------------------------------
# classification and the factor's own pieces, formed exactly as raw_theta forms them


def kind_of(phase) -> str:
    if type(phase) is not PrimitivePhase:
        return "other"
    label = getattr(phase._leading, "label", None)
    if phase._sign == -1 and label == "tau":
        return "G"
    if phase._sign == +1 and label == "cs_tau":
        return "T"
    return "other"


def pieces(phase, x, x_is_log=True):
    """(raw_z, u, phi) exactly as PrimitivePhase.raw_theta forms them."""
    raw_x, log_x = phase._get_x(x, x_is_log)
    u = phase._clamped_u(raw_x, log_x)
    phi = float(phase._spline(u))
    return raw_x, u, phi


def delta_ext(table, z_a: float, z_b: float):
    """(Delta^ext as an exact mpf, the ends p_b - p_a as an exact mpf, p_a, p_b): production's
    CumulativeTable.delta re-formed exactly, with production's node choice and partials.
    """
    z_a = float(z_a)
    z_b = float(z_b)
    j_a, on_a = table._locate(z_a)
    j_b, on_b = table._locate(z_b)
    hi = table._hi_list
    lo = table._lo_list
    mpf = MP.mpf
    tab = (mpf(hi[j_b]) - mpf(hi[j_a])) + (mpf(lo[j_b]) - mpf(lo[j_a]))
    p_b = 0.0 if on_b else table._integral(z_b, table._z_list[j_b])
    p_a = 0.0 if on_a else table._integral(z_a, table._z_list[j_a])
    ends = mpf(p_b) - mpf(p_a)
    return tab + ends, ends, p_a, p_b


def delta_fraction(table, z_a: float, z_b: float, p_a: float, p_b: float) -> Fraction:
    """The same quantity in exact rational arithmetic, for check_exact (given the partials)."""
    j_a, _ = table._locate(float(z_a))
    j_b, _ = table._locate(float(z_b))
    hi = table._hi_list
    lo = table._lo_list
    F = Fraction
    return (F(hi[j_b]) - F(hi[j_a])) + (F(lo[j_b]) - F(lo[j_a])) + (F(p_b) - F(p_a))


def reduce_exact(theta_mp):
    """theta mod TWO_PI (the double, exactly) into (-TWO_PI, 0], as an exact mpf."""
    n = MP.ceil(theta_mp / TWO_PI_MP)
    r = theta_mp - n * TWO_PI_MP
    while r > 0:
        r -= TWO_PI_MP
    while r <= -TWO_PI_MP:
        r += TWO_PI_MP
    return r


def theta_ext(phase, x, x_is_log=True):
    """(exact theta = s k Delta^ext(z, a) + phi, ends, raw_z, phi)."""
    raw_x, u, phi = pieces(phase, x, x_is_log)
    d, ends, _pa, _pb = delta_ext(phase._leading.table, raw_x, phase._z_anchor)
    th = MP.mpf(phase._sign) * MP.mpf(phase._k) * d + MP.mpf(phi)
    return th, ends, raw_x, phi


def rho_X(phase, x, x_is_log=True) -> float:
    th, _ends, _z, _phi = theta_ext(phase, x, x_is_log)
    return mp_to_float(reduce_exact(th))


# --- arm A ---------------------------------------------------------------------------------------


class _AState:
    """Per-G constants for arm A: z_top and c_G. Holds a strong reference to the phase, so that
    an id() is never reused while the entry exists; the cache is cleared at the end of a unit.
    """

    def __init__(self):
        self.cache = {}

    def get(self, phase):
        entry = self.cache.get(id(phase))
        if entry is not None and entry[0] is phase:
            return entry[1], entry[2]
        z_top = z_top_of(phase)
        d, _ends, _pa, _pb = delta_ext(phase._leading.table, z_top, phase._z_anchor)
        c_G = mp_to_float(reduce_exact(MP.mpf(phase._sign) * MP.mpf(phase._k) * d))
        self.cache[id(phase)] = (phase, z_top, c_G)
        return z_top, c_G


def z_top_of(phase) -> float:
    """The node of the leading table at or nearest above max_x (the top of phi's samples)."""
    nodes = phase._leading.table._z_list  # descending
    above = [z for z in nodes if z >= phase.max_x]
    if not above:
        raise RuntimeError(
            f"phases: no node of the {phase._leading.label} table at or above max_x={phase.max_x!r}"
        )
    return min(above)


def inner_A(phase, x, z_top, x_is_log=True):
    """s k Delta_tau(z, z_top) + phi(u) in double, formed as raw_theta forms its own sum."""
    raw_x, u, phi = pieces(phase, x, x_is_log)
    leading = phase._leading.delta(raw_x, z_top)
    return phase._sign * phase._k * leading + phi


def rho_A_G(phase, x, astate, x_is_log=True) -> float:
    z_top, c_G = astate.get(phase)
    return WKB_mod_2pi(inner_A(phase, x, z_top, x_is_log))[1] + c_G


# ------------------------------------------------------------------------------------------------
# the dispatcher


class Dispatcher:
    ARMS = ("P", "A", "X")

    def __init__(self, arm: str, deadline=None, stop_exc=None):
        if arm not in self.ARMS:
            raise ValueError(arm)
        self.arm = arm
        # the run's budget (prompt section 3): past `deadline` (time.perf_counter()), the next
        # call raises `stop_exc`, from here, before any computation. Raised at this one known
        # place rather than from a signal handler, which can fire inside a bare `except:`
        # (mpmath's from_float) and be swallowed there.
        self.deadline = deadline
        self.stop_exc = stop_exc
        self.counts = {"G": 0, "T": 0, "other": 0}
        self.astate = _AState()

    def __call__(self, phase, x, x_is_log=False):
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise self.stop_exc()
        kind = kind_of(phase)
        self.counts[kind] += 1
        arm = self.arm
        if arm == "P" or kind == "other":
            return ORIG_THETA_MOD_2PI(phase, x, x_is_log)
        if arm == "X":
            return rho_X(phase, x, x_is_log)
        # arm A
        if kind == "G":
            return rho_A_G(phase, x, self.astate, x_is_log)
        return ORIG_THETA_MOD_2PI(phase, x, x_is_log)


@contextlib.contextmanager
def install(arm: str, deadline=None, stop_exc=None):
    """Patch PrimitivePhase.theta_mod_2pi at the class attribute with the arm's dispatcher, and
    restore the original in a finally. Yields the Dispatcher (its .counts)."""
    if PrimitivePhase.__dict__["theta_mod_2pi"] is not ORIG_THETA_MOD_2PI:
        raise RuntimeError("phases: PrimitivePhase.theta_mod_2pi is already patched")
    dispatcher = Dispatcher(arm, deadline=deadline, stop_exc=stop_exc)

    def theta_mod_2pi(self, x, x_is_log=False):
        return dispatcher(self, x, x_is_log)

    theta_mod_2pi.__name__ = "theta_mod_2pi"
    theta_mod_2pi.__qualname__ = f"PrimitivePhase.theta_mod_2pi[arm {arm}]"
    PrimitivePhase.theta_mod_2pi = theta_mod_2pi
    try:
        yield dispatcher
    finally:
        PrimitivePhase.theta_mod_2pi = ORIG_THETA_MOD_2PI
        dispatcher.astate.cache.clear()
    if PrimitivePhase.__dict__["theta_mod_2pi"] is not ORIG_THETA_MOD_2PI:
        raise RuntimeError("phases: PrimitivePhase.theta_mod_2pi was not restored")


# ------------------------------------------------------------------------------------------------
# Phi and F


def Phi(arm: str, phase, x, astate=None, x_is_log=True) -> float:
    """Phi^arm_i at the abscissa x (already clamped where production clamps)."""
    kind = kind_of(phase)
    if arm == "P" or (arm == "A" and kind != "G"):
        return abs(phase.raw_theta(x, x_is_log=x_is_log))
    if arm == "A":
        z_top, _c = (astate or _AState()).get(phase)
        return abs(inner_A(phase, x, z_top, x_is_log))
    if arm == "X":
        raw_x, _u, _phi = pieces(phase, x, x_is_log)
        _d, ends, _pa, _pb = delta_ext(phase._leading.table, raw_x, phase._z_anchor)
        return phase._k * abs(mp_to_float(ends)) + 2.0 * math.pi
    raise ValueError(arm)


def factor_list(regime, Gk_phase, Tq_wrapped, Tr_wrapped):
    """The oscillatory factors of a sub-interval as (name, PrimitivePhase, clamp) with clamp the
    abscissa map production applies before calling the factor (identity for G; _ClampedTk's
    _clamp for T, through _ClampedPhase)."""
    out = []
    if regime[0]:
        out.append(("G", Gk_phase, lambda x: x))
    for name, flag, wrapped in (
        ("q", regime[1], Tq_wrapped),
        ("r", regime[2], Tr_wrapped),
    ):
        if flag:
            cp = wrapped.phase  # _ClampedPhase
            out.append((name, cp._phase, cp._clamp))
    return out


def F_group(arm, factors, x_span, astate=None, n=F_SAMPLES):
    """EPS x max over n equispaced abscissae of x_span of sum over the group's factors of Phi^arm
    (each at its clamped abscissa). ``factors`` are (name, phase, clamp) of the group's factors.
    """
    astate = astate or _AState()
    xs = np.linspace(x_span[0], x_span[1], n)
    best = 0.0
    for x in xs:
        s = 0.0
        for _name, ph, clamp in factors:
            s += Phi(arm, ph, clamp(float(x)), astate)
        best = max(best, s)
    return EPS * best


def sum_Phi(arm, factors, x, astate=None):
    astate = astate or _AState()
    return sum(Phi(arm, ph, clamp(float(x)), astate) for _n, ph, clamp in factors)


def wrap_abs_exact(a: float, b: float) -> float:
    """|wrap(a - b)|, with the difference of the two doubles and its wrap onto (-pi, pi] (period
    TWO_PI, the double, exactly; pi taken as TWO_PI/2) formed exactly, then rounded once.
    """
    T = Fraction(TWO_PI)
    d = Fraction(a) - Fraction(b)
    r = d - T * round(d / T)
    if r > T / 2:
        r -= T
    elif r <= -T / 2:
        r += T
    return float(abs(r))


def wrap(d: float) -> float:
    """Map onto (-pi, pi] with period TWO_PI."""
    r = math.fmod(d, TWO_PI)
    if r > math.pi:
        r -= TWO_PI
    elif r <= -math.pi:
        r += TWO_PI
    return r
