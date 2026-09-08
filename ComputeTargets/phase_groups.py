"""
Phase-group decomposition of the source integrand for the source time integral.

The integrand of the source time integral (spec 03 R28, audit `docs/spec-code-audit-2026-09.md`
section 3.1) is

    G_k(z, z') f(z' | q, r) / H(z')^2

up to the (1 + z_response) prefactor, which is NOT applied here (`QuadSourceIntegral` applies it
to the integral, as the existing code does at `QuadSourceIntegral.py:975/:1045/:1167`). Each of
the three factors G, T_q, T_r is either *smooth* (super-horizon, or in its numeric
representation) or *oscillatory* (in its Liouville-Green representation M sin theta). This module
turns "which factors are oscillatory here" plus the representations of the three factors into a
list of **phase groups**, each being exactly the

    (f_sin, f_cos, theta, theta_mod_2pi, theta_deriv)

bundle that one `adaptive_levin_sincos` call needs (`AdaptiveLevin/levin_quadrature.py:2707`),
such that, pointwise in z',

    G f / H^2 = sum over groups [ f_sin(z') sin Psi(z') + f_cos(z') cos Psi(z') ] ,

with every f_sin, f_cos smooth and every composed phase Psi = +-theta_G +- theta_q +- theta_r.
No integration happens here, and nothing here imports Ray or the datastore. This is the
README section 6 phase-group table of `prompts/source-remediation/`, realised:

    oscillatory factors      groups   phases
    none                     (raises: the all-smooth case is ordinary quadrature of source_function)
    G only                   1        theta_G
    T_q only / T_r only      1        theta_q / theta_r
    G and one T              2        theta_G +- theta_q   (or +- theta_r)
    T_q and T_r, G smooth    2        theta_q +- theta_r
    all three                4        theta_G +- theta_q +- theta_r

Definitions
-----------

D denotes (1 + z') d/dz', the derivative with respect to log(1 + z'), which is the Levin
integration variable in `QuadSourceIntegral`. The source kernel (`QuadSource.source_function`,
verified identical to spec 03 R22 by audit QS-1) is, with w_0 = wBackground(z'),

    alpha = (5 + 3 w_0) / (3 (1 + w_0)),     beta = 2 / (3 (1 + w_0)),
    f = alpha T_q T_r + beta [ - DT_q T_r - DT_r T_q + DT_q DT_r ] .

`ComputeTargets/tests/sympy_phase_groups.py` checks symbolically that this is the same function
as `source_function`, and that every coefficient below is what sympy's own product-to-sum
expansion gives.

A *smooth* transfer-function factor is given by callables for its value T_i(z') and its plain
z-derivative dT_i/dz, from which DT_i = (1 + z') dT_i/dz. A smooth Green's function is given by
its value G(z') only.

An *oscillatory* transfer-function factor T_i = M_i sin theta_i is given by the
`TkSourceFunctions` fields M_i, d ln M_i/dz, `phase` (a `phase_spline`) and
omega_i = d theta_i/dz. Then

    DT_i = a_i sin theta_i + b_i cos theta_i,
    a_i = (1 + z') M_i d ln M_i/dz,     b_i = (1 + z') M_i omega_i .

The oscillatory Green's function is G = A_G sin theta_G with A_G = `GkSourceFunctions.sin_amplitude`
and theta_G = `GkSourceFunctions.phase`; its cosine amplitude is identically zero
(`GkSourcePolicyData.py:676`, `GkWKBIntegration.py:409`).

Every callable produced here takes log(1 + z') (the `z_is_log=True` / `x_is_log=True` convention
of `ZSplineWrapper` and `phase_spline`). The measure factor 1/H(z')^2 is folded into every
amplitude.

The algebra
-----------

Write S_i = sin theta_i, C_i = cos theta_i.

*Both T oscillatory.* Expanding f in the basis {S_q, C_q} x {S_r, C_r}:

    c_SS = alpha M_q M_r + beta (- a_q M_r - a_r M_q + a_q a_r)
    c_SC = beta (- b_r M_q + a_q b_r)            [coefficient of S_q C_r]
    c_CS = beta (- b_q M_r + b_q a_r)            [coefficient of C_q S_r]
    c_CC = beta b_q b_r

and product-to-sum with Theta_+- = theta_q +- theta_r gives

    f = P_+ cos Theta_+ + Q_+ sin Theta_+ + P_- cos Theta_- + Q_- sin Theta_-
    P_+ = (c_CC - c_SS)/2,  P_- = (c_CC + c_SS)/2,  Q_+ = (c_SC + c_CS)/2,  Q_- = (c_SC - c_CS)/2 .

*One T oscillatory* (T_q, with T_r smooth):  f = c_S S_q + c_C C_q,

    c_S = alpha M_q T_r + beta (- a_q T_r - DT_r M_q + a_q DT_r),     c_C = beta b_q (DT_r - T_r),

i.e. a single group P cos theta_q + Q sin theta_q with P = c_C, Q = c_S; symmetrically for T_r.

*Neither T oscillatory.*  A single "group" with constant phase, P = f, Q = 0.

*Multiplying by G.*  If G is smooth every group keeps its phase and its amplitudes are
multiplied by G/H^2: f_cos = G P / H^2, f_sin = G Q / H^2. If G = A_G sin theta_G, then using
sin theta_G cos Theta = [sin(theta_G + Theta) + sin(theta_G - Theta)]/2 and
sin theta_G sin Theta = [cos(theta_G - Theta) - cos(theta_G + Theta)]/2 each group splits into

    Psi = theta_G + Theta:  f_sin = + A_G P / (2 H^2),   f_cos = - A_G Q / (2 H^2)
    Psi = theta_G - Theta:  f_sin = + A_G P / (2 H^2),   f_cos = + A_G Q / (2 H^2)

except the constant-phase group (both T smooth), whose two halves coincide and re-combine into
the single group Psi = theta_G, f_sin = A_G f / H^2, f_cos = 0 -- the integrand the existing
`WKB_Levin_integral` uses (`QuadSourceIntegral.py:1099-1105`).

Phase composition
-----------------

Each composed phase Psi = s_G theta_G + s_q theta_q + s_r theta_r (s_i in {-1, 0, +1}) supplies
the three callables the Levin driver uses, each as the signed sum of the constituents':

  * `theta`          -- signed sum of `phase.raw_theta(x, x_is_log=True)`. Used by the driver for
                        its total-variation gate and derivative estimates; absolute precision
                        loss at |theta| ~ 1e7 is acceptable there.
  * `theta_mod_2pi`  -- signed sum of `phase.theta_mod_2pi(x, x_is_log=True)`. The sum lies in
                        (-6 pi, 6 pi); sin/cos are periodic so no re-reduction is needed. This is
                        deliberately NOT "sum raw_theta and reduce", which would throw away the
                        precision the stored (div 2pi, mod 2pi) split exists to protect
                        (`docs/resonance-scaffolding/sigw-resonance-reconciliation.md` section
                        3.1, last paragraph). It is what `_three_bessel_Levin` does
                        (`QuadSourceIntegral.py:514-528`).
  * `theta_deriv`    -- signed sum of d theta_i / d log(1 + z'). For a transfer function this is
                        the closed-form omega_i (1 + z') from `TkSourceFunctions.omega`; for the
                        Green's function it is `phase.theta_deriv(x, x_is_log=True,
                        log_derivative=True)`, as `WKB_Levin_integral.Levin_deriv` defines it
                        (`QuadSourceIntegral.py:1113-1120`). Provided always; whether it is passed
                        to the driver is the caller's decision (see the comment at
                        `QuadSourceIntegral.py:1128-1145`).

Sign conventions are inherited, not imposed: d theta/dz = +omega_eff integrated towards smaller z
means every stored transfer-function phase is negative and increasing with z, while the
Green's-function phase at fixed response redshift decreases with the source redshift
(`prompts/source-remediation/IMPLEMENTATION_STATE.md` section 5 note 5, `logs/05` deviation 2).
Nothing here depends on either sign: the identity G f / H^2 = sum of groups holds for whatever
theta_i the factor objects supply, as long as T_i = M_i sin theta_i and G = A_G sin theta_G.

Input protocol (duck-typed)
---------------------------

`build_phase_groups(regime, *, Gk, Tq, Tr, model_functions, w_background)` with
`regime = (G_osc, q_osc, r_osc)`:

  * oscillatory T (flag True): an object exposing `M(x, z_is_log=True)`, `dlnM_dz(x, z_is_log=True)`,
    `omega(x, z_is_log=True)` and `phase` (a `phase_spline` with `raw_theta`, `theta_mod_2pi`,
    `theta_deriv`, all taking `x_is_log=True`) -- a `TkSourceFunctions`;
  * smooth T (flag False): an object exposing `T(x, z_is_log=True)` and `dT_dz(x, z_is_log=True)`
    -- the numeric-region accessors of the same `TkSourceFunctions`;
  * oscillatory G: an object exposing `sin_amplitude(x, z_is_log=True)` and `phase` -- a
    `GkSourceFunctions`;
  * smooth G: either a callable `G(x, z_is_log=True)` (e.g. `GkSourceFunctions.numeric_Gk`, a
    `ZSplineWrapper`) or an object exposing `numeric_Gk`;
  * `model_functions`: exposes `Hubble(z)` (a `ModelFunctions`);
  * `w_background`: callable, w_0(z) at the source redshift (`model_functions.wBackground`).

Every accessor is called with log(1 + z'). Range checking is the accessors' own: `TkSourceFunctions`
raises outside `numeric_region`/`WKB_region`, `phase_spline` cannot be extrapolated. The caller
partitions its integral so that each region's regime matches the representations valid there.
"""

from dataclasses import dataclass
from functools import lru_cache
from math import exp, sin, cos, sqrt
from typing import Callable, Tuple, List, Sequence, Any

Regime = Tuple[bool, bool, bool]
Signs = Tuple[int, int, int]

# number of distinct log(1+z') abscissae for which the shared ingredient evaluation is memoised.
# The Levin driver samples f_sin, f_cos (and every group) on the same Chebyshev nodes of each
# subregion, so without this every node would cost 2 x (number of groups) evaluations of the
# amplitude ingredients instead of one.
INGREDIENT_CACHE_SIZE = 16384

_FACTOR_NAMES = ("G", "q", "r")


@dataclass(frozen=True)
class PhaseGroup:
    """
    One term f_sin sin Psi + f_cos cos Psi of the decomposition; exactly the bundle one
    `adaptive_levin_sincos(x_span, f=[f_sin, f_cos], theta=levin_theta())` call needs.

    `signs` records (s_G, s_q, s_r) with 0 for a factor that is smooth, so that
    Psi = s_G theta_G + s_q theta_q + s_r theta_r. `label` reads like "G+q-r".
    """

    label: str
    f_sin: Callable[[float], float]
    f_cos: Callable[[float], float]
    theta: Callable[[float], float]
    theta_mod_2pi: Callable[[float], float]
    theta_deriv: Callable[[float], float]
    signs: Signs

    def levin_theta(self, include_deriv: bool = True) -> dict:
        """
        The `theta` dict for `adaptive_levin_sincos`. `include_deriv=False` omits "theta_deriv",
        reproducing what `WKB_Levin_integral` currently passes (`QuadSourceIntegral.py:1122-1145`).
        """
        payload = {"theta": self.theta, "theta_mod_2pi": self.theta_mod_2pi}
        if include_deriv:
            payload["theta_deriv"] = self.theta_deriv
        return payload

    def value(self, log_z: float) -> float:
        """f_sin sin(Psi mod 2pi) + f_cos cos(Psi mod 2pi) at log(1+z')."""
        psi = self.theta_mod_2pi(log_z)
        return self.f_sin(log_z) * sin(psi) + self.f_cos(log_z) * cos(psi)

    def envelope(self, log_z: float) -> float:
        """sqrt(f_sin^2 + f_cos^2) at log(1+z'): the local amplitude of this group."""
        s = self.f_sin(log_z)
        c = self.f_cos(log_z)
        return sqrt(s * s + c * c)


# ------------------------------------------------------------------------------------------
# pure algebra
#
# Everything in this section is plain arithmetic on its arguments, with no math-module calls,
# so that it can be evaluated on sympy symbols by tests/sympy_phase_groups.py as well as on
# floats at run time. The run-time code below calls exactly these functions; nothing about
# the algebra lives anywhere else.


def source_coefficients(w):
    """
    alpha, beta of f = alpha T_q T_r + beta [ - DT_q T_r - DT_r T_q + DT_q DT_r ], for w_0 = w.
    """
    denominator = 3 * (1 + w)
    return (5 + 3 * w) / denominator, 2 / denominator


def both_oscillatory_coefficients(alpha, beta, Mq, aq, bq, Mr, ar, br):
    """
    f = P_+ cos Theta_+ + Q_+ sin Theta_+ + P_- cos Theta_- + Q_- sin Theta_-, Theta_+- = theta_q +- theta_r.
    Returns (P_+, Q_+, P_-, Q_-).
    """
    c_SS = alpha * Mq * Mr + beta * (-aq * Mr - ar * Mq + aq * ar)
    c_SC = beta * (-br * Mq + aq * br)
    c_CS = beta * (-bq * Mr + bq * ar)
    c_CC = beta * bq * br

    P_plus = (c_CC - c_SS) / 2
    P_minus = (c_CC + c_SS) / 2
    Q_plus = (c_SC + c_CS) / 2
    Q_minus = (c_SC - c_CS) / 2
    return P_plus, Q_plus, P_minus, Q_minus


def one_oscillatory_coefficients(alpha, beta, M, a, b, T_smooth, DT_smooth):
    """
    f = P cos theta + Q sin theta for one oscillatory factor (M, a, b) against one smooth factor
    (T, DT). Returns (P, Q) = (c_C, c_S).
    """
    c_S = alpha * M * T_smooth + beta * (-a * T_smooth - DT_smooth * M + a * DT_smooth)
    c_C = beta * b * (DT_smooth - T_smooth)
    return c_C, c_S


def smooth_source(alpha, beta, Tq, DTq, Tr, DTr):
    """
    f for two smooth factors; algebraically `QuadSource.source_function(...)["source"]` with
    DT = (1+z) dT/dz (checked symbolically in tests/sympy_phase_groups.py).
    """
    return alpha * Tq * Tr + beta * (-DTq * Tr - DTr * Tq + DTq * DTr)


def phase_group_terms(
    regime: Regime, alpha, beta, G, Tq, Tr
) -> List[Tuple[Signs, Any, Any]]:
    """
    The decomposition of G * f (the 1/H^2 is NOT included) at a single point.

    :param regime: (G_osc, q_osc, r_osc)
    :param G: A_G if G_osc, else the value G
    :param Tq: (M, a, b) if q_osc, else (T, DT); likewise Tr
    :return: list of (signs, f_sin, f_cos) with signs = (s_G, s_q, s_r), such that
        G f = sum f_sin sin Psi + f_cos cos Psi, Psi = s_G theta_G + s_q theta_q + s_r theta_r.
        Ordered by signs, descending, i.e. G+q+r, G+q-r, G-q+r, G-q-r.
    """
    G_osc, q_osc, r_osc = regime

    # step 1: f as a list of T-groups (s_q, s_r, P, Q) meaning P cos Theta + Q sin Theta
    if q_osc and r_osc:
        P_plus, Q_plus, P_minus, Q_minus = both_oscillatory_coefficients(
            alpha, beta, *Tq, *Tr
        )
        T_groups = [(1, 1, P_plus, Q_plus), (1, -1, P_minus, Q_minus)]
    elif q_osc:
        P, Q = one_oscillatory_coefficients(alpha, beta, *Tq, *Tr)
        T_groups = [(1, 0, P, Q)]
    elif r_osc:
        P, Q = one_oscillatory_coefficients(alpha, beta, *Tr, *Tq)
        T_groups = [(0, 1, P, Q)]
    else:
        T_groups = [(0, 0, smooth_source(alpha, beta, *Tq, *Tr), 0)]

    # step 2: multiply by G
    terms = []
    if not G_osc:
        for sq, sr, P, Q in T_groups:
            terms.append(((0, sq, sr), G * Q, G * P))
    else:
        for sq, sr, P, Q in T_groups:
            if sq == 0 and sr == 0:
                # constant-phase group: theta_G + 0 and theta_G - 0 coincide
                terms.append(((1, 0, 0), G * P, 0 * P))
            else:
                terms.append(((1, sq, sr), G * P / 2, -G * Q / 2))
                terms.append(((1, -sq, -sr), G * P / 2, G * Q / 2))

    terms.sort(key=lambda item: item[0], reverse=True)
    return terms


def signs_label(signs: Signs) -> str:
    """(1, -1, 0) -> "G-q", (0, 1, 1) -> "q+r", (1, 0, 0) -> "G"."""
    parts = []
    for name, s in zip(_FACTOR_NAMES, signs):
        if s == 0:
            continue
        if s > 0:
            parts.append(("+" if parts else "") + name)
        else:
            parts.append("-" + name)
    return "".join(parts)


def group_signs(regime: Regime) -> List[Signs]:
    """The sign tuples of the groups `build_phase_groups(regime, ...)` returns, in order."""
    _check_regime(regime)
    # any non-zero placeholders will do: only the structure is read
    G = 1
    Tq = (1, 1, 1) if regime[1] else (1, 1)
    Tr = (1, 1, 1) if regime[2] else (1, 1)
    return [signs for signs, _, _ in phase_group_terms(regime, 1, 1, G, Tq, Tr)]


def _check_regime(regime) -> Regime:
    if len(regime) != 3:
        raise ValueError(
            f"phase_groups: regime must be a 3-tuple (G_osc, q_osc, r_osc), got {regime!r}"
        )
    regime = tuple(bool(flag) for flag in regime)
    if not any(regime):
        raise ValueError(
            "phase_groups: regime (False, False, False) has no oscillatory factor; the all-smooth "
            "region is ordinary quadrature of QuadSource.source_function and is not handled here"
        )
    return regime


# ------------------------------------------------------------------------------------------
# factor adapters


class _SmoothT:
    def __init__(self, functions, name: str):
        for attr in ("T", "dT_dz"):
            if not hasattr(functions, attr):
                raise TypeError(
                    f"phase_groups: smooth factor T_{name} must expose .{attr}(x, z_is_log=True) "
                    f"(the numeric-region accessors of a TkSourceFunctions), got {type(functions).__name__}"
                )
        self._f = functions

    def values(self, log_z: float, one_plus_z: float):
        """(T, DT) with DT = (1+z) dT/dz."""
        return (
            self._f.T(log_z, z_is_log=True),
            one_plus_z * self._f.dT_dz(log_z, z_is_log=True),
        )


class _OscillatoryT:
    def __init__(self, functions, name: str):
        for attr in ("M", "dlnM_dz", "omega", "phase"):
            if not hasattr(functions, attr):
                raise TypeError(
                    f"phase_groups: oscillatory factor T_{name} must expose .{attr} "
                    f"(a TkSourceFunctions), got {type(functions).__name__}"
                )
        self._f = functions
        self.phase = functions.phase

    def values(self, log_z: float, one_plus_z: float):
        """(M, a, b) with a = (1+z) M d ln M/dz, b = (1+z) M omega."""
        M = self._f.M(log_z, z_is_log=True)
        return (
            M,
            one_plus_z * M * self._f.dlnM_dz(log_z, z_is_log=True),
            one_plus_z * M * self._f.omega(log_z, z_is_log=True),
        )

    def theta_deriv(self, log_z: float) -> float:
        """d theta / d log(1+z) = omega (1+z), closed form."""
        return self._f.omega(log_z, z_is_log=True) * exp(log_z)


class _SmoothG:
    def __init__(self, Gk):
        if callable(Gk):
            self._G = Gk
        elif hasattr(Gk, "numeric_Gk") and callable(Gk.numeric_Gk):
            self._G = Gk.numeric_Gk
        else:
            raise TypeError(
                "phase_groups: smooth Green's function must be a callable G(x, z_is_log=True) "
                f"(e.g. GkSourceFunctions.numeric_Gk) or expose .numeric_Gk, got {type(Gk).__name__}"
            )

    def value(self, log_z: float) -> float:
        return self._G(log_z, z_is_log=True)


class _OscillatoryG:
    def __init__(self, Gk):
        for attr in ("sin_amplitude", "phase"):
            if not hasattr(Gk, attr):
                raise TypeError(
                    f"phase_groups: oscillatory Green's function must expose .{attr} "
                    f"(a GkSourceFunctions), got {type(Gk).__name__}"
                )
        if Gk.sin_amplitude is None or Gk.phase is None:
            raise ValueError(
                "phase_groups: oscillatory Green's function has no WKB representation "
                "(sin_amplitude or phase is None)"
            )
        self._Gk = Gk
        self.phase = Gk.phase

    def value(self, log_z: float) -> float:
        """A_G, the sine amplitude."""
        return self._Gk.sin_amplitude(log_z, z_is_log=True)

    def theta_deriv(self, log_z: float) -> float:
        return self.phase.theta_deriv(log_z, x_is_log=True, log_derivative=True)


# ------------------------------------------------------------------------------------------
# public API


def build_phase_groups(
    regime: Regime,
    *,
    Gk,
    Tq,
    Tr,
    model_functions,
    w_background: Callable[[float], float],
) -> List[PhaseGroup]:
    """
    Decompose G f / H^2 into phase groups for the given regime. See the module docstring for the
    input protocol. Raises ValueError for the all-smooth regime (False, False, False).

    :param regime: (G_osc, q_osc, r_osc)
    :param Gk: GkSourceFunctions (oscillatory) or its numeric_Gk callable (smooth)
    :param Tq: TkSourceFunctions for q; its WKB accessors are used if q_osc, its numeric ones if not
    :param Tr: likewise for r
    :param model_functions: exposes Hubble(z)
    :param w_background: callable w_0(z), wBackground at the source redshift
    :return: list of PhaseGroup, 1, 2 or 4 long, ordered as `group_signs(regime)`
    """
    regime = _check_regime(regime)
    G_osc, q_osc, r_osc = regime

    G_factor = _OscillatoryG(Gk) if G_osc else _SmoothG(Gk)
    q_factor = _OscillatoryT(Tq, "q") if q_osc else _SmoothT(Tq, "q")
    r_factor = _OscillatoryT(Tr, "r") if r_osc else _SmoothT(Tr, "r")

    Hubble = model_functions.Hubble

    @lru_cache(maxsize=INGREDIENT_CACHE_SIZE)
    def terms_at(log_z: float):
        """
        The (signs, f_sin, f_cos) list at one abscissa, with 1/H^2 applied. Cached because the
        Levin driver asks every group's f_sin and f_cos at the same nodes.
        """
        one_plus_z = exp(log_z)
        z = one_plus_z - 1.0
        H = Hubble(z)
        inv_H_sq = 1.0 / (H * H)

        alpha, beta = source_coefficients(w_background(z))
        G = G_factor.value(log_z)
        Tq_values = q_factor.values(log_z, one_plus_z)
        Tr_values = r_factor.values(log_z, one_plus_z)

        return tuple(
            (signs, f_sin * inv_H_sq, f_cos * inv_H_sq)
            for signs, f_sin, f_cos in phase_group_terms(
                regime, alpha, beta, G, Tq_values, Tr_values
            )
        )

    # the oscillatory constituents, in (G, q, r) order, for phase composition
    constituents = (
        G_factor if G_osc else None,
        q_factor if q_osc else None,
        r_factor if r_osc else None,
    )

    groups = []
    for index, signs in enumerate(group_signs(regime)):
        active = [
            (s, factor)
            for s, factor in zip(signs, constituents)
            if s != 0 and factor is not None
        ]
        groups.append(
            PhaseGroup(
                label=signs_label(signs),
                f_sin=_component(terms_at, index, 1),
                f_cos=_component(terms_at, index, 2),
                theta=_composed_raw_theta(active),
                theta_mod_2pi=_composed_theta_mod_2pi(active),
                theta_deriv=_composed_theta_deriv(active),
                signs=signs,
            )
        )

    return groups


def _component(terms_at, index: int, slot: int):
    def component(log_z: float) -> float:
        return terms_at(log_z)[index][slot]

    return component


def _composed_raw_theta(active: Sequence):
    def theta(log_z: float) -> float:
        return sum(
            s * factor.phase.raw_theta(log_z, x_is_log=True) for s, factor in active
        )

    return theta


def _composed_theta_mod_2pi(active: Sequence):
    def theta_mod_2pi(log_z: float) -> float:
        # the signed sum of remainders lies in (-6 pi, 6 pi); sin/cos are periodic, so it is
        # not re-reduced (see the module docstring)
        return sum(
            s * factor.phase.theta_mod_2pi(log_z, x_is_log=True) for s, factor in active
        )

    return theta_mod_2pi


def _composed_theta_deriv(active: Sequence):
    def theta_deriv(log_z: float) -> float:
        return sum(s * factor.theta_deriv(log_z) for s, factor in active)

    return theta_deriv


def evaluate_sum(groups: Sequence[PhaseGroup], log_z: float) -> float:
    """
    sum over groups of [ f_sin sin(Psi mod 2pi) + f_cos cos(Psi mod 2pi) ] at log(1+z').

    For tests and for region-boundary consistency checks -- never for integration, which is
    the Levin driver's job.
    """
    return sum(group.value(log_z) for group in groups)


def evaluate_envelope(groups: Sequence[PhaseGroup], log_z: float) -> float:
    """
    sum over groups of sqrt(f_sin^2 + f_cos^2) at log(1+z'): an upper bound on |evaluate_sum|
    and the natural scale against which to measure a residual of the decomposition.
    """
    return sum(group.envelope(log_z) for group in groups)
