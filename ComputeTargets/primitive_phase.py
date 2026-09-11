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

**Anchoring, and the floor it implies** (README Sec 7 D6; review Sec 13.4). ``theta_mod_2pi``
reduces ``raw_theta`` with ``WKB_mod_2pi`` -- one reduction of one unreduced value, per the house
rules in ``LiouvilleGreen/range_reduce_mod_2pi.py`` -- against the *global* anchor ``z_anchor``.
The reduced product is therefore as large as the whole range's phase, and inherits the
``eps k tau`` rounding floor: ~1e-7 rad at ``k = 1e5/Mpc`` and ~9e-4 rad at ``k = 3e8/Mpc``
(review Sec 1, Sec 13.4). That is accepted here: it is three to four orders below the consumer
error this class replaces and at or below the QCD Liouville-Green truncation floor of ~1e-3 rad
(review Sec 6), so nothing physical is visible under it. Per-*region* anchoring -- re-anchoring
at each Levin subinterval, so the floor scales with that region's own phase instead of the whole
range's -- is the recorded follow-up ``[00-consumer-anchoring-floor]`` and is deliberately not
done here (README Sec 0.4).

**Protocol.** This class implements exactly the three methods every ``phase_spline`` consumer
calls -- ``raw_theta(x, x_is_log)``, ``theta_mod_2pi(x, x_is_log)``,
``theta_deriv(x, x_is_log, log_derivative)`` -- plus the ``num_chunks`` property that
``QuadSourceIntegral`` reads through ``getattr`` into its stored metadata. So
``AdaptiveLevin.levin_quadrature`` (through the phase dict), ``ComputeTargets.phase_groups``
(``_OscillatoryG.theta_deriv``, ``_composed_*``), ``QuadSourceIntegral._ClampedPhase`` and
``ComputeTargets.spline_wrappers.GkWKBSplineWrapper`` consume one of these unchanged.

``x`` is a redshift throughout (``x_is_log=True`` means ``x = log(1+z)``), matching the
``x_is_redshift=True, x_is_log=True`` configuration the two production consumers used.

No Ray, no datastore: this is a plain numerical object built from a ``TablePrimitive``, a
wavenumber and a sample of ``phi``.
"""

from math import expm1, log1p
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.interpolate import make_interp_spline

from LiouvilleGreen.WKBtools import WKB_mod_2pi

# The out-of-range policy is phase_spline's, and is shared rather than re-stated: inside the
# sampled range evaluate normally; within SPLINE_TOP_BOTTOM_CUSHION of a fractional overshoot,
# clamp to the boundary; beyond that, raise.
from LiouvilleGreen.phase_spline import SPLINE_TOP_BOTTOM_CUSHION


class PrimitivePhase:
    """
    ``theta(x) = sign * k * leading.delta(x, x_anchor) + phi(x)``, evaluated from a per-model
    ``CumulativeTable`` accessor and a spline of the small residual ``phi``. Implements the
    ``phase_spline`` protocol used by ``AdaptiveLevin`` (through the phase dict),
    ``phase_groups._OscillatoryG``, ``QuadSourceIntegral._ClampedPhase`` and
    ``spline_wrappers.GkWKBSplineWrapper``.

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
        """``raw_theta`` reduced into ``(-2pi, 0]`` (the producers' negative-remainder
        convention), against the global anchor ``z_anchor`` -- see the module docstring for the
        floor this implies and why it is accepted."""
        return WKB_mod_2pi(self.raw_theta(x, x_is_log=x_is_log))[1]

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
    ``phi(z_i) = theta_stored(z_i) - sign * k * leading.delta(z_i, z_anchor)``, the residual a
    ``PrimitivePhase`` splines.

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
