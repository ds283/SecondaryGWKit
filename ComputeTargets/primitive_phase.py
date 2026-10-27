"""
A Liouville-Green (WKB) phase evaluated as a closed-form leading term plus a spline of a small
residual, rather than as a spline of the phase itself.

**Why this exists.** The consumers of a stored WKB phase -- ``GkSourcePolicyData`` for the
Green's function, ``TkSourceFunctions`` for the transfer function -- used to build a
``LiouvilleGreen.phase_spline`` through the stored ``(theta_div_2pi, theta_mod_2pi)`` samples and
evaluate it between them. A cubic spline of a phase that grows like ``x`` has interpolation error
``h^4 theta'''' / 384 ~ h^4 x / 384`` (review ``docs/gk-wkb-review-fable-2026-09-09.md`` Sec 5,
measured at 8.26e-5 rad at ``x = 1e6`` and 8.26e-3 rad at ``x = 1e8`` on 100 samples per decade),
which is O(1)-O(10) rad at the production ``x_source`` of 1e9-1e12. No sample density fixes that,
and no rebasing scheme touches it: the *representation* has to change.

**The decomposition** (campaign README Sec 2 (a), (g)). With ``omega^2 = (k/H)^2 + C`` and the
conformal-time primitive ``tau = int dz/H``,

    theta(z) = sign * k * leading.delta(z, z_anchor) + phi(z)

where ``leading`` is a ``BackgroundModel`` ``TablePrimitive`` (``functions.tau`` for the
Green's function, ``functions.cs_tau`` for the transfer function), evaluated through its
double-double interval accessor, and ``phi`` is what remains: the phase residual
``rho = int C/(omega + k/H) dz`` (<= 1.5e-3 rad for ``G_k``, ~ -0.09 rad for ``T_k``) plus the
initial-data offset ``delta`` of a numeric hand-over and any constant the producer's cycle
bookkeeping carries. ``phi`` is smooth and O(1) rad, so a cubic spline of *it* has interpolation
error ``h^4 |phi''''| / 384``, six or more orders below the same spline of ``theta``.

``phi`` is supplied by the caller, which recovers it from the stored samples as

    phi(z_i) = theta_stored(z_i) - sign * k * leading.delta(z_i, z_anchor)

so that ``raw_theta`` reproduces ``theta_stored`` at every sample and interpolates through the
leading term between them.

**Sign.** ``sign = -1`` for the Green's function at fixed response redshift, where the phase
``theta(z_r; z_s) = -k [tau(z_s) - tau(z_r)] = -k * tau.delta(z_s, z_r)`` decreases as the source
redshift rises; ``sign = +1`` for the transfer function at fixed ``z_init``, where
``theta_T(z) = -k * cs_tau.delta(z_init, z) = +k * cs_tau.delta(z, z_init)``. Both are the same
formula because ``delta`` is antisymmetric (README Sec 2 (c):
``delta(z_a, z_b) = T(z_b) - T(z_a)``).

**Derivative in closed form.** ``d/dz [leading.delta(z, z_anchor)] = -dT/dz = +f(z)``, since
``CumulativeTable`` accumulates ``T(z) = int_z^{z_top} f``. The magnitude of that rate is
supplied by the ``rate`` constructor parameter, so

    d theta / dz = sign * k * rate(z) + phi'(z)

with no spectral differentiation of a sampled phase anywhere. For ``tau``, ``rate = 1/H``,
which is what ``rate`` defaults to when omitted (built from ``model_functions.Hubble``); for a
leading primitive whose rate is not ``1/H`` -- the transfer function's sound horizon ``cs_tau``,
``d(cs_tau)/dz = -c_s/H`` so ``rate = c_s/H`` -- the caller passes ``rate`` explicitly rather
than making ``model_functions`` lie about ``Hubble`` (prompt 15 of the campaign; before it,
``TkSourceFunctions`` passed a ``_SoundHorizonRate`` adapter whose ``.Hubble`` actually returned
``H/c_s``). (Exact-radiation check: ``theta = -k(1/s_r - 1/s_s)``, ``d theta/d z_s = -k/s_s^2 =
-k/H`` with ``H = H0 s^2``, ``H0 = 1``, i.e. ``rate = 1/H``.)

**Anchoring, and the floor it implies** (README Sec 7 D6; review Sec 13.4;
``prompts/levin-phase-floor`` prompt 05). There are two ways to form the remainder, chosen per
object by the ``exact_remainder`` constructor keyword.

*The default* (``exact_remainder=False``) reduces ``raw_theta`` with ``WKB_mod_2pi`` -- one
reduction of one unreduced value, per the house rules in ``LiouvilleGreen/range_reduce_mod_2pi.py``
-- against the *global* anchor ``z_anchor``. The reduced value is a double as large as the whole
range's phase, ``~ k tau(z_anchor)``, so the remainder inherits its ``eps k tau`` rounding: ~1e-7
rad at ``k = 1e5/Mpc`` and ~9e-4 rad at ``k = 3e8/Mpc`` (review Sec 1, Sec 13.4). That rounding is
not smooth in ``z``. The reduction is also against the double ``TWO_PI``, not against 2pi, which
adds a smooth ``0.18 eps |theta|``. ``levin-phase-floor`` found the jitter at 0.22-0.34 of
``F = eps k tau_R`` in every expensive phase group of A3 v2, and showed that it, and not the
integrand, sends the Levin bisection to its depth cap (its 03 and 04; VERIFICATION Sec 3-4).

*The node-anchored remainder* (``exact_remainder=True``) forms no double of size ``k tau_R`` on
the way to the remainder. ``CumulativeTable.delta`` is a node term plus two end partials,
``delta(z, z_R) = [T(n_R) - T(n_z)] + p_R - p_z``, with ``n_z`` the node ``delta`` assigns to
``z`` and ``p_z`` the local Gauss partial to it (zero on a node). So

    theta(z) = c_{n_z} + v(z),
    c_n  = (sign k [T(n_R) - T(n) + p_R]) mod 2pi,       v(z) = -sign k p_z + phi(u).

``c_n`` is computed once per node, lazily, and memoised: the node term exactly from the four
limbs (``CumulativeTable.node_limbs``), ``k`` multiplied in exactly, and the reduction against
2pi itself in extended precision (``mpmath``, 256 bits), then rounded once to a double in
``[-TWO_PI, 0]``. ``v`` is at most ``k`` times half a table interval of the primitive plus O(1)
rad; it is formed in double and reduced with ``WKB_mod_2pi``, whose modulus error,
``0.18 eps |v|``, is negligible. The remainder is ``c_{n_z} + rem(v)``, wrapped once by an exact
add of ``TWO_PI`` (Sterbenz). What is left is ``~ eps k |p_z|`` of jitter and ~1e-15 rad where
``n_z`` switches between nodes. ``raw_theta`` and ``theta_deriv`` are the same in both modes,
bit for bit: ``raw_theta`` is still the double of size ``k tau_R``, used for the driver's
total-variation gate and for diagnostics, where that size is harmless.

**Who uses which.** Both production factors use the node-anchored remainder.
- The Green's function (``GkSourcePolicyData._build_phase``), since prompt 05 of
  ``prompts/levin-phase-floor``, on the user's answer at its gate G1 (2026-10-26): its anchor term
  ``k tau_R`` was the floor (04's arm A matched the exact limit region for region). Its anchor,
  the response redshift, is a table node in production, so its ``p_R`` is zero. This is what
  answers per-*region* anchoring, ``[00-consumer-anchoring-floor]``, for ``G``.
- The transfer function (``TkSourceFunctions``, whose ``exact_remainder`` keyword defaults to
  on), since prompt 05a, on the user's decision of 2026-10-26 to treat ``T`` "in exactly the same
  way": ``T`` enters the integrand on the same footing as ``G`` and carried both errors, the
  ``eps k tau_s`` rounding and the modulus. Its anchor ``z_init`` is a root, not a table node, so
  its ``p_R`` is a non-zero double from the table's own quadrature, and enters ``c_n`` exactly as
  that double.

The default remains for objects whose leading primitive has no table nodes: test fixtures built
on a closed form, which pass ``exact_remainder=False`` explicitly. Neither mode touches the
rounding of the abscissa itself, ``z = expm1(u)``, about ``eps z/(1+z) |d theta/du|`` (04's I1),
which for ``T`` is the same size as the floor removed wherever ``z >> 1``, nor the noise in the
stored samples of ``phi`` (``[02-consumer-phi-below-the-storage-granularity]``).

**Protocol.** This class implements exactly the three methods every ``phase_spline`` consumer
calls -- ``raw_theta(x, x_is_log)``, ``theta_mod_2pi(x, x_is_log)``,
``theta_deriv(x, x_is_log, log_derivative)`` -- plus the ``num_chunks`` property that
``QuadSourceIntegral`` reads through ``getattr`` into its stored metadata. So
``AdaptiveLevin.levin_quadrature`` (through the phase dict), ``ComputeTargets.phase_groups``
(``_OscillatoryG.theta_deriv``, ``_composed_*``) and
``ComputeTargets.spline_wrappers.GkWKBSplineWrapper`` consume one of these unchanged.
``QuadSourceIntegral`` reaches it only through ``phase_groups``: its clamp adapter was removed by
``prompts/handover`` prompt 12.

``x`` is a redshift throughout (``x_is_log=True`` means ``x = log(1+z)``), matching the
``x_is_redshift=True, x_is_log=True`` configuration the two production consumers used.

No Ray, no datastore: this is a plain numerical object built from a ``TablePrimitive``, a
wavenumber and a sample of ``phi``.
"""

from fractions import Fraction
from math import expm1, log1p
from typing import Callable, Dict, Optional, Sequence, Tuple

import mpmath
import numpy as np
from mpmath.libmp import round_nearest, to_float
from scipy.interpolate import make_interp_spline

from LiouvilleGreen.WKBtools import WKB_mod_2pi
from LiouvilleGreen.constants import TWO_PI

# The out-of-range policy is phase_spline's, and is shared rather than re-stated: inside the
# sampled range evaluate normally; within SPLINE_TOP_BOTTOM_CUSHION of a fractional overshoot,
# clamp to the boundary; beyond that, raise.
from LiouvilleGreen.phase_spline import SPLINE_TOP_BOTTOM_CUSHION

# The node constants c_n of the node-anchored remainder are reduced against 2pi itself in this
# private mpmath context, never in the global one (which a caller may have set). 256 bits is
# ~77 decimal digits: |sign k [T(n_R) - T(n) + p_R]| is at most ~1e13 rad (~2^43), so the
# reduction's error, ~2^43 * 2^-256 of 2pi, is ~1e-63 rad, far below the one rounding to a
# double that follows it.
_REMAINDER_CONTEXT = mpmath.MPContext()
_REMAINDER_CONTEXT.prec = 256


def _remainder_against_two_pi(theta: Fraction) -> float:
    """
    ``theta mod 2pi`` into ``(-2pi, 0]``, for an exact rational ``theta``, reduced against 2pi
    itself (not against the double ``TWO_PI``) in ``_REMAINDER_CONTEXT``, then rounded once to
    the nearest double. The result lies in ``[-TWO_PI, 0]``: ``TWO_PI < 2pi``, so a remainder in
    ``(-2pi, -TWO_PI]`` rounds to ``-TWO_PI``, which is still in ``(-2pi, 0]``.
    """
    ctx = _REMAINDER_CONTEXT
    # every double is a dyadic rational, so the denominator is a power of two and the division
    # rounds only the numerator, relative 2^-256
    x = ctx.mpf(theta.numerator) / ctx.mpf(theta.denominator)
    two_pi = 2 * ctx.pi
    r = x - ctx.ceil(x / two_pi) * two_pi
    # ceil puts r in (-2pi, 0] up to the context's rounding; settle an edge case either way
    while r > 0:
        r -= two_pi
    while r <= -two_pi:
        r += two_pi
    return to_float(r._mpf_, rnd=round_nearest)


class PrimitivePhase:
    """
    ``theta(x) = sign * k * leading.delta(x, x_anchor) + phi(x)``, evaluated from a per-model
    ``CumulativeTable`` accessor and a spline of the small residual ``phi``. Implements the
    ``phase_spline`` protocol used by ``AdaptiveLevin`` (through the phase dict),
    ``phase_groups._OscillatoryG`` and ``spline_wrappers.GkWKBSplineWrapper``.

    See the module docstring for the decomposition, the sign convention, the closed-form
    derivative and the anchoring floor.

    :param k: the wavenumber, in the same units as ``leading`` (``wavenumber.k``, not
        ``k_inv_Mpc``, although in ``Mpc_units`` these coincide)
    :param leading: a ``BackgroundModel.TablePrimitive`` (or anything with a
        ``delta(z_a, z_b)`` interval accessor): ``functions.tau`` for the Green's function,
        ``functions.cs_tau`` for the transfer function
    :param z_anchor: the redshift the phase is measured from. ``raw_theta(z_anchor)`` is
        ``phi(z_anchor)``, not zero, because ``phi`` carries the producer's offsets
    :param z_samples: the redshifts at which ``phi`` is supplied; need not be sorted
    :param phi_samples: ``phi`` at those redshifts, in radians
    :param sign: ``-1`` (Green's function, fixed ``z_response``) or ``+1`` (transfer function,
        fixed ``z_init``)
    :param model_functions: the background model's ``ModelFunctions``; only ``.Hubble`` is
        used, to build the default ``rate`` when none is supplied. Kept as a required
        parameter (and still validated) even when ``rate`` is passed explicitly, so that
        ``model_functions.Hubble`` always returns the genuine Hubble rate off this object,
        never a repurposed one
    :param label: a name for error messages
    :param spline_order: the order of the ``phi`` spline (3, the default, or 5)
    :param rate: ``d/dz [leading.delta(z, z_anchor)]``'s magnitude, e.g. ``1/H(z)`` for ``tau``
        or ``c_s(z)/H(z)`` for ``cs_tau``. Defaults to ``None``, which builds
        ``lambda z: 1.0 / model_functions.Hubble(z)`` -- the value every caller that predates
        this parameter relied on, so omitting it changes nothing
    :param exact_remainder: ``False`` (the default) reduces ``raw_theta`` with ``WKB_mod_2pi``,
        exactly as before this parameter existed. ``True`` forms ``theta_mod_2pi`` node by node,
        with the large constant reduced exactly against 2pi and never rounded at the size of
        ``k tau_R`` (module docstring, "Anchoring"). It needs ``leading`` to supply the node
        accessors ``end_partial`` and ``node_limbs`` (``BackgroundModel.TablePrimitive`` passes
        them through from ``CumulativeTable``); without them the constructor raises
        ``ValueError``, and there is no fallback
    """

    def __init__(
        self,
        k: float,
        leading,
        z_anchor: float,
        z_samples: Sequence[float],
        phi_samples: Sequence[float],
        *,
        sign: int,
        model_functions,
        label: str = "",
        spline_order: int = 3,
        rate: Optional[Callable[[float], float]] = None,
        exact_remainder: bool = False,
    ):
        self._label = label

        sign = int(sign)
        if sign not in (-1, +1):
            raise ValueError(
                f"PrimitivePhase[{label}]: sign must be -1 or +1 (got {sign})"
            )
        self._sign = sign

        k = float(k)
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError(
                f"PrimitivePhase[{label}]: k must be finite and positive (got {k})"
            )
        self._k = k

        if leading is None or not hasattr(leading, "delta"):
            raise ValueError(
                f"PrimitivePhase[{label}]: leading must supply an interval accessor delta(z_a, z_b)"
            )
        self._leading = leading

        self._z_anchor = float(z_anchor)

        if model_functions is None or not hasattr(model_functions, "Hubble"):
            raise ValueError(
                f"PrimitivePhase[{label}]: model_functions must supply Hubble(z)"
            )
        self._Hubble = model_functions.Hubble

        # rate(z) is d/dz[leading.delta(z, z_anchor)]'s magnitude. When not supplied, build the
        # default every caller before this parameter existed relied on: 1/H(z), for `tau`.
        self._rate = rate if rate is not None else (lambda z: 1.0 / self._Hubble(z))

        z = np.asarray([float(v) for v in z_samples], dtype=float)
        phi = np.asarray([float(v) for v in phi_samples], dtype=float)
        if z.ndim != 1 or z.shape != phi.shape:
            raise ValueError(
                f"PrimitivePhase[{label}]: z_samples and phi_samples must be 1-d and the same "
                f"length (got {z.shape} and {phi.shape})"
            )

        spline_order = int(spline_order)
        if spline_order not in (3, 5):
            raise ValueError(
                f"PrimitivePhase[{label}]: spline_order must be 3 or 5 (got {spline_order})"
            )
        if z.size < spline_order + 1:
            raise RuntimeError(
                f"PrimitivePhase[{label}]: an order-{spline_order} spline of phi needs at least "
                f"{spline_order + 1} samples (got {z.size})"
            )

        # sort ascending in z = ascending in u; log1p is accurate to ~1 ulp (CLAUDE.md,
        # "Redshift arithmetic"), and the reverse map is only ever used to place a quadrature
        # endpoint, never in an equality test
        order = np.argsort(z)
        self._z_points = z[order]
        self._phi_points = phi[order]
        self._u_points = np.log1p(self._z_points)

        if np.any(np.diff(self._u_points) <= 0.0):
            raise ValueError(
                f"PrimitivePhase[{label}]: z_samples must be distinct in log(1+z)"
            )

        self._spline = make_interp_spline(
            self._u_points, self._phi_points, k=spline_order
        )
        self._spline_deriv = self._spline.derivative()
        self._spline_order = spline_order

        self.min_log_x = float(self._u_points[0])
        self.max_log_x = float(self._u_points[-1])
        self.min_x = float(self._z_points[0])
        self.max_x = float(self._z_points[-1])

        # the node-anchored remainder (module docstring, "Anchoring"): opt in, and only with the
        # leading primitive's node accessors. No fallback to the default: a caller that asks for
        # it and cannot have it is told so here, at construction
        self._exact_remainder = bool(exact_remainder)
        self._node_constants: Dict[int, float] = {}
        self._anchor_node = None
        self._anchor_exact = None
        if self._exact_remainder:
            missing = [
                name
                for name in ("end_partial", "node_limbs")
                if not callable(getattr(leading, name, None))
            ]
            if len(missing) > 0:
                raise ValueError(
                    f"PrimitivePhase[{label}]: exact_remainder needs the leading primitive's node "
                    f"accessors end_partial and node_limbs (CumulativeTable, through "
                    f"BackgroundModel.TablePrimitive); {type(leading).__name__} lacks "
                    f"{', '.join(missing)}"
                )
            # T(n_R) + p_R, exactly: the anchor's node limbs and its own end partial, which is
            # a constant of the object (zero when the anchor is a node, as G's z_R is in
            # production; non-zero for T, whose z_init is a root)
            j_R, p_R = leading.end_partial(self._z_anchor)
            hi_R, lo_R = leading.node_limbs(j_R)
            self._anchor_node = int(j_R)
            self._anchor_exact = Fraction(hi_R) + Fraction(lo_R) + Fraction(p_R)

    # ---------------------------------------------------------------------------------------
    # range discipline (phase_spline's, so that a consumer sees the same behaviour)
    # ---------------------------------------------------------------------------------------

    def _clamped_u(self, raw_x: float, log_x: float) -> float:
        """
        ``log_x`` moved onto the sampled range of ``phi``, or a ``RuntimeError`` if it is further
        outside than ``SPLINE_TOP_BOTTOM_CUSHION`` of its own magnitude (``phase_spline``'s rule,
        with its sign handling for a negative ``log(1+z)``).

        Only the ``phi`` spline is clamped. The leading term is evaluated at the requested
        redshift whatever happens, because the ``CumulativeTable`` behind it is exact over the
        whole background grid and has its own (wider) range check; clamping it as well would put
        a kink in ``theta`` just inside the boundary, where clamping ``phi`` alone leaves ``theta``
        continuous and still correct to the size of ``phi``'s own variation.
        """
        cushion = float(np.sign(log_x)) * SPLINE_TOP_BOTTOM_CUSHION

        if log_x < self.min_log_x * (1.0 - cushion):
            raise RuntimeError(
                f"PrimitivePhase[{self._label}]: evaluated out-of-bounds of lower limit at "
                f"log_x={log_x:.5g} (raw x={raw_x:.5g}) | Minimum allowed value is "
                f"log_x={self.min_log_x:.5g} (raw x={self.min_x:.5g})"
            )
        if log_x < self.min_log_x:
            return self.min_log_x

        if log_x > self.max_log_x * (1.0 + cushion):
            raise RuntimeError(
                f"PrimitivePhase[{self._label}]: evaluated out-of-bounds of upper limit at "
                f"log_x={log_x:.5g} (raw x={raw_x:.5g}) | Maximum allowed value is "
                f"log_x={self.max_log_x:.5g} (raw x={self.max_x:.5g})"
            )
        if log_x > self.max_log_x:
            return self.max_log_x

        return log_x

    @staticmethod
    def _get_x(x: float, x_is_log: bool):
        """``(z, log(1+z))`` from the supplied abscissa."""
        if x_is_log:
            log_x = float(x)
            # the recovered z is a quadrature endpoint only -- it is handed to the interval
            # accessor and to H(z), never compared for equality (README Sec 5 rule 9)
            return expm1(log_x), log_x
        raw_x = float(x)
        return raw_x, log1p(raw_x)

    # ---------------------------------------------------------------------------------------
    # the phase_spline protocol
    # ---------------------------------------------------------------------------------------

    def raw_theta(self, x: float, x_is_log: bool = False) -> float:
        """The unreduced phase in radians. Never pre-reduced: pass this to ``sin``/``cos``
        directly if you want the oscillation, or call ``theta_mod_2pi`` for the
        (cycle, remainder) reduction (README Sec 2 (e))."""
        raw_x, log_x = self._get_x(x, x_is_log)
        u = self._clamped_u(raw_x, log_x)
        leading = self._leading.delta(raw_x, self._z_anchor)
        return self._sign * self._k * leading + float(self._spline(u))

    def theta_mod_2pi(self, x: float, x_is_log: bool = False) -> float:
        """
        The phase reduced into ``(-2pi, 0]`` (the producers' negative-remainder convention).

        By default, ``raw_theta`` reduced with ``WKB_mod_2pi`` against the global anchor
        ``z_anchor``, carrying ``eps k tau_R`` of rounding. With ``exact_remainder``,
        ``c_{n_z} + rem(v(z))``, wrapped once into ``[-TWO_PI, 0]``, with no double of size
        ``k tau_R`` formed on the way. The module docstring, "Anchoring", has both.
        """
        if not self._exact_remainder:
            return WKB_mod_2pi(self.raw_theta(x, x_is_log=x_is_log))[1]

        # the abscissa exactly as raw_theta takes it
        raw_x, log_x = self._get_x(x, x_is_log)
        u = self._clamped_u(raw_x, log_x)

        # delta(z, z_R) = [T(n_R) - T(n_z)] + p_R - p_z: the node and the partial delta uses
        j, p_z = self._leading.end_partial(raw_x)
        c = self._node_constant(j)

        # the small variable part, -sign k p_z + phi(u), in double, as raw_theta forms its sum
        v = -self._sign * self._k * p_z + float(self._spline(u))

        # c is in [-TWO_PI, 0] and rem(v) in (-TWO_PI, 0], so the sum is in [-2 TWO_PI, 0]; one
        # add of TWO_PI to a value in [-2 TWO_PI, -TWO_PI] is exact (Sterbenz)
        r = c + WKB_mod_2pi(v)[1]
        if r <= -TWO_PI:
            r += TWO_PI
        return r

    def _node_constant(self, j: int) -> float:
        """
        ``c_j = (sign k [T(n_R) - T(n_j) + p_R]) mod 2pi`` in ``[-TWO_PI, 0]``: the node term from
        the four limbs and ``k`` multiplied in, both exactly (``Fraction``), reduced against 2pi
        itself in extended precision and rounded once. Computed on first use and memoised.
        """
        c = self._node_constants.get(j)
        if c is None:
            hi, lo = self._leading.node_limbs(j)
            node_term = self._anchor_exact - (Fraction(hi) + Fraction(lo))
            c = _remainder_against_two_pi(self._sign * Fraction(self._k) * node_term)
            self._node_constants[j] = c
        return c

    def theta_deriv(
        self, x: float, x_is_log: bool = False, log_derivative: bool = False
    ) -> float:
        """
        ``d theta / dz`` (or ``d theta / d log(1+z) = (1+z) d theta/dz`` when
        ``log_derivative``), in closed form for the leading term:
        ``sign * k * rate(z) + phi'(z)``. The ``phi`` spline is built in ``u = log(1+z)``, so
        its own derivative is already the logarithmic one.
        """
        raw_x, log_x = self._get_x(x, x_is_log)
        u = self._clamped_u(raw_x, log_x)

        dphi_du = float(self._spline_deriv(u))
        # d/dz leading.delta(z, z_anchor) = -d(leading)/dz = +f(z) = rate(z) (= 1/H(z) for tau)
        dtheta_dz_leading = self._sign * self._k * self._rate(raw_x)

        if log_derivative:
            return dtheta_dz_leading * (1.0 + raw_x) + dphi_du

        return dtheta_dz_leading + dphi_du / (1.0 + raw_x)

    @property
    def num_chunks(self) -> int:
        """Always 1. ``QuadSourceIntegral`` reads this through ``getattr`` into its stored
        ``WKB_phase_spline_chunks`` metadata; there are no chunks in this representation, just as
        there are none left in ``phase_spline`` since prompt 08."""
        return 1

    # ---------------------------------------------------------------------------------------
    # inspection
    # ---------------------------------------------------------------------------------------

    def phi(self, x: float, x_is_log: bool = False) -> float:
        """The residual alone, for diagnostics and tests."""
        raw_x, log_x = self._get_x(x, x_is_log)
        return float(self._spline(self._clamped_u(raw_x, log_x)))

    @property
    def k(self) -> float:
        return self._k

    @property
    def exact_remainder(self) -> bool:
        """Whether ``theta_mod_2pi`` is the node-anchored remainder (module docstring)."""
        return self._exact_remainder

    @property
    def sign(self) -> int:
        return self._sign

    @property
    def z_anchor(self) -> float:
        return self._z_anchor

    @property
    def z_samples(self) -> np.ndarray:
        """The sample redshifts, ascending."""
        return self._z_points

    @property
    def phi_samples(self) -> np.ndarray:
        """``phi`` at the sample redshifts, in the same order."""
        return self._phi_points

    @property
    def spline_order(self) -> int:
        return self._spline_order

    @property
    def label(self) -> str:
        return self._label


def build_phi_samples(
    k: float,
    leading,
    z_anchor: float,
    z_samples: Sequence[float],
    theta_samples: Sequence[float],
    *,
    sign: int,
) -> np.ndarray:
    """
    ``phi(z_i) = theta_stored(z_i) - sign * k * leading.delta(z_i, z_anchor)``: the residual a
    ``PrimitivePhase`` splines, recovered *the old way*, as the difference of two numbers of size
    ``k tau``.

    **No production consumer calls this since prompt 06 of ``prompts/levin-phase-floor``.** The
    difference above is quantised to ``ulp(theta)``, and the stored ``theta`` had already lost its
    small part to the same granularity (``[02-consumer-phi-below-the-storage-granularity]``). The
    producers now store that small part, ``theta_residual``, formed from its own small terms, and
    the two consumers build ``phi`` from it: ``GkSourcePolicyData._build_phase`` as
    ``r + k tau.delta(z_s, z_init)``, ``TkSourceFunctions`` as ``r`` itself. This function is kept
    for the tests that pin the old construction (``test_primitive_phase``,
    ``test_gk_source_primitive_phase``) and for measuring what it cost
    (``test_stored_phase_residual``, prompt 06 test 3). Do not reintroduce it into a consumer.

    What follows describes it as it was used until prompt 06.

    ``theta_stored`` must be the *unwrapped* phase ``theta_div_2pi * 2pi + theta_mod_2pi`` built
    from the **rectified** cycle counts (``GkSourceValue.WKB.theta_div_2pi``, not
    ``raw_theta_div_2pi``): the rectifier repairs the 2pi wraps of the initial-data offset
    ``delta`` that occur where the numeric stop point moves to the next extremum between
    neighbouring source redshifts (``RECONCILIATION.md`` Sec 2 item 6; campaign decision D5), and
    those wraps would otherwise appear in ``phi`` as jumps of a full cycle and be splined through.

    ``phi`` is defined only up to the additive constant the rectifier's rebasing carries (it puts
    the first sample in the fundamental block), which is an exact multiple of 2pi and changes
    nothing: only ``phi``'s smoothness matters to the interpolation, and ``theta_mod_2pi`` is
    unchanged by a whole number of cycles.
    """
    sign = int(sign)
    k = float(k)
    z_anchor = float(z_anchor)
    return np.array(
        [
            float(theta) - sign * k * leading.delta(float(z), z_anchor)
            for z, theta in zip(z_samples, theta_samples)
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------------------------
# The producers' exact phase (prompts/levin-phase-floor prompt 06)
# ---------------------------------------------------------------------------------------------
#
# ``WKB_phase_function`` and the two WKB producers' ``store()`` form every stored sample's phase
#
#     theta = sign * k * leading.delta(z_a, z_b) + (small terms)
#
# *exactly*, as a rational, and reduce it against 2pi itself. The large term comes from the
# table's four node limbs and two end partials (the same pieces ``CumulativeTable.delta`` rounds),
# with ``k`` multiplied in exactly; the small terms (the residual table's interval, the initial-data
# offset) are added exactly. Nothing of size ``k tau`` is rounded before the remainder is.
#
# The arithmetic is Python integers on a common power-of-two scale, not ``Fraction`` and
# ``mpmath`` per sample: the per-object wall-time tests of the producers leave no room for the
# ~20 us a sample that ``_remainder_against_two_pi``'s route costs, where this costs a few us.
# ``PrimitivePhase``'s own reducer, ``_remainder_against_two_pi``, is untouched; the two agree to
# the last bit wherever both are applied to the same rational (prompt 06 test 6).

# floor(2pi * 2^_TWO_PI_BITS), extended on demand; the reduction shifts it down to its own scale
_TWO_PI_BITS = 0
_TWO_PI_SCALED = 0

# guard bits beyond the cycle count when 2pi is truncated to the working scale: the truncation
# costs at most |cycles| * 2^-scale rad, so the scale is chosen to put that below 2^-_GUARD_BITS
_GUARD_BITS = 120


def _two_pi_scaled(bits: int) -> int:
    """``floor(2pi * 2^bits)``, exact, from one constant held at the largest precision yet asked
    for (computed in a private ``mpmath`` context, never the global one)."""
    global _TWO_PI_BITS, _TWO_PI_SCALED
    if bits > _TWO_PI_BITS:
        precision = max(bits, 2048) + 64
        ctx = mpmath.MPContext()
        ctx.prec = precision + 64
        _TWO_PI_SCALED = int(ctx.floor(ctx.ldexp(2 * ctx.pi, precision)))
        _TWO_PI_BITS = precision
    return _TWO_PI_SCALED >> (_TWO_PI_BITS - bits)


def _dyadic(x: float) -> Tuple[int, int]:
    """``(n, s)`` with ``x == n / 2^s`` exactly; every finite double is such a dyadic rational."""
    n, d = float(x).as_integer_ratio()
    return n, d.bit_length() - 1


def exact_leading_phase(
    k: float,
    leading,
    z_a: float,
    z_b: float,
    *,
    sign: int,
    small_terms: Sequence[float] = (),
) -> Fraction:
    """
    ``sign * k * leading.delta(z_a, z_b) + sum(small_terms)``, **exactly**, as a ``Fraction``.

    ``leading.delta(z_a, z_b)`` is taken as the table defines it before any rounding: the node
    term ``(hi_b + lo_b) - (hi_a + lo_a)`` from the four limbs (``node_limbs``), plus the two end
    partials ``p_b - p_a`` exactly as ``delta`` forms them (``delta_parts``, which spends the same
    integrand evaluations as one call of ``delta``). ``k``, the partials and every small term are
    doubles, so each is a dyadic rational and the sum is exact.

    ``leading`` must supply ``delta_parts`` and ``node_limbs`` (``BackgroundModel.TablePrimitive``
    passes them through from ``CumulativeTable``); there is no fallback to ``delta``.
    """
    j_a, j_b, p_a, p_b = leading.delta_parts(z_a, z_b)
    hi_a, lo_a = leading.node_limbs(j_a)
    hi_b, lo_b = leading.node_limbs(j_b)

    # the interval (hi_b + lo_b) - (hi_a + lo_a) + p_b - p_a, each double as n / 2^s, brought to
    # their finest common scale and summed as integers (written out: this runs once per sample
    # in the producers' wall-time budget)
    n1, d1 = hi_b.as_integer_ratio()
    n2, d2 = lo_b.as_integer_ratio()
    n3, d3 = hi_a.as_integer_ratio()
    n4, d4 = lo_a.as_integer_ratio()
    n5, d5 = float(p_b).as_integer_ratio()
    n6, d6 = float(p_a).as_integer_ratio()
    s1 = d1.bit_length() - 1
    s2 = d2.bit_length() - 1
    s3 = d3.bit_length() - 1
    s4 = d4.bit_length() - 1
    s5 = d5.bit_length() - 1
    s6 = d6.bit_length() - 1
    scale = max(s1, s2, s3, s4, s5, s6)
    interval = (
        (n1 << (scale - s1))
        + (n2 << (scale - s2))
        - (n3 << (scale - s3))
        - (n4 << (scale - s4))
        + (n5 << (scale - s5))
        - (n6 << (scale - s6))
    )

    k_n, k_s = _dyadic(k)
    numerator = int(sign) * k_n * interval
    scale += k_s

    for term in small_terms:
        n, s = _dyadic(term)
        if s > scale:
            numerator <<= s - scale
            scale = s
        numerator += n << (scale - s)

    if scale < 0:
        return Fraction(numerator << (-scale))
    return Fraction(numerator, 1 << scale)


def reduce_exactly_mod_2pi(theta: Fraction) -> Tuple[int, float]:
    """
    ``(theta_div_2pi, theta_mod_2pi)`` of an exact dyadic rational ``theta``, reduced against 2pi
    itself in the producers' negative-remainder convention: ``theta = div * 2pi + mod`` with ``div``
    the exact cycle count, ``ceil(theta / 2pi)``, and ``mod`` the exact remainder in ``(-2pi, 0]``
    rounded once to the nearest double.

    A remainder in ``(-2pi, -TWO_PI]``, i.e. within ``2pi - TWO_PI ~ 2.4e-16`` of ``-2pi``, would
    round to ``-TWO_PI``. It is stored as ``0.0``, with the cycle count moved one further from zero
    in the direction of the phase (``div - 1``: ``div * 2pi - 2pi + d = (div - 1) * 2pi + d`` with
    ``0 <= d < 2.4e-16``), so that the remainder stays in ``(-TWO_PI, 0]`` as ``WKB_mod_2pi``'s does
    (prompt 06 §9 point 6). For the producers' negative phases that raises the count's magnitude
    by one.

    The arithmetic is integer: ``theta`` is brought to a power-of-two scale fine enough that the
    truncation of 2pi to that scale costs at most ``2^-120`` rad, and the remainder is converted to
    a double by one correctly rounded integer division.
    """
    numerator = theta.numerator
    denominator = theta.denominator
    shift = denominator.bit_length() - 1
    if denominator != 1 << shift:
        raise ValueError(
            "reduce_exactly_mod_2pi: theta must be a dyadic rational (a sum of doubles), got "
            f"denominator {denominator}"
        )

    work = max(shift, numerator.bit_length() - shift + _GUARD_BITS, _GUARD_BITS)
    scaled = numerator << (work - shift)
    two_pi = _two_pi_scaled(work)

    cycles = -((-scaled) // two_pi)
    remainder = (scaled - cycles * two_pi) / (1 << work)

    if remainder <= -TWO_PI:
        remainder = 0.0
        cycles -= 1
    return int(cycles), remainder + 0.0
