"""
A cubic spline of a Liouville-Green (WKB) phase, evaluated safely mod 2*pi.

The stored phase is split into an integer cycle count (``theta_div_2pi``, growing without bound
over the WKB range) and a bounded remainder (``theta_mod_2pi``). ``phase_spline`` builds one
cubic spline of ``(theta_div_2pi - base) * 2*pi + theta_mod_2pi`` over ``log(x)`` (or
``log(1+z)`` when ``x_is_redshift``), rebased around a single integer ``base`` chosen from the
sample so that the spline's internal ordinates stay of order the data span rather than growing
with the absolute cycle count.

What this object is **not**: a cure for the interpolation error of splining a rapidly growing
phase. A cubic spline's error scales as ``h^4 * theta'''' / 384 ~ h^4 * x / 384`` in the WKB
regime (review ``docs/gk-wkb-review-fable-2026-09-09.md`` Sec 5), and no rebasing scheme changes
that -- it is O(1)-O(10) rad at production x. The cure is a different representation that splines
only a small, smooth residual against a closed-form leading term; see
``ComputeTargets/primitive_phase.py`` (``PrimitivePhase``, prompt 09 of the GkTk-remedial
campaign), which implements this class's protocol (``raw_theta``, ``theta_mod_2pi``,
``theta_deriv``) for production Green's-function consumers.

Chunking -- splitting the sample into several independent splines by cycle-count range, selected
at evaluation time by a hard switch -- was removed by GkTk-remedial prompt 08. The review measured
it as *harmful*, not merely useless: on the production consumer geometry it left the interpolation
error bit-for-bit identical to a single spline, while inflating spline ordinates by 64x, worsening
knot residuals 30-50x, and introducing a switch discontinuity of 1.4e-4 rad in theta and 3.3e-8
relative in theta' at the chunk boundary (review Sec 5). A single global rebase already bounds
ordinates by the data span, which is all the chunking scheme was trying to buy.

The ``chunk_step``/``chunk_logstep`` constructor arguments are therefore **deprecated and have no
effect**: they are accepted and ignored, retained only so that existing callers do not have to
change (``LiouvilleGreen/bessel_phase.py`` on ``main`` still constructs with
``chunk_logstep=125``, as do three test fixtures). ``num_chunks`` always returns ``1``.
"""

from math import log, exp, fmod
from typing import Iterable, Tuple, Optional

from numpy import sign
from scipy.interpolate import make_interp_spline

from .constants import TWO_PI

# Deprecated: chunk_step's historical default. No longer affects behaviour (see module
# docstring); retained only so the constructor's default argument value is unchanged.
DEFAULT_CHUNK_SIZE = 200

SPLINE_TOP_BOTTOM_CUSHION = 0.001


class _rebased_spline:
    """
    A single cubic spline of the phase over all supplied data, rebased so that its internal
    ordinates are close to zero rather than growing with the absolute cycle count.

    (Formerly ``_chunk_spline``, when several of these were built per ``phase_spline`` and
    selected between at evaluation time. GkTk-remedial prompt 08 removed chunking, so exactly one
    of these now backs every ``phase_spline``; the name and implementation are otherwise
    unchanged.)
    """

    def __init__(
        self,
        div_2pi_base: int,
        data: Iterable[dict[str, float]],
        x_is_log: bool = False,
        x_is_redshift: bool = False,
    ):
        self._x_is_redshift: bool = x_is_redshift
        self._div_2pi_base: int = div_2pi_base

        # cache supplied values
        self._data = data
        if self._data is None:
            raise RuntimeError(
                f"phase_spline: supplied data vector is empty (div_2pi_base={div_2pi_base})"
            )
        self.data_points: int = len(self._data)

        self._data.sort(key=lambda d: d["x"])
        self._x_sample = [p["x"] for p in self._data]

        if len(data) == 0:
            raise RuntimeError(
                f"phase_spline: no data points supplied (div_2pi_base={div_2pi_base})"
            )

        # build set of y values, but rebased mod 2pi so that all values are close to zero
        # in this way, we hope to avoid loss of precision when the result of the spline is evaluated mod 2pi, as it typically
        # will be
        self._y_points = [
            (p["div_2pi"] - div_2pi_base) * TWO_PI + p["mod_2pi"] for p in self._data
        ]

        if x_is_log:
            self._log_x_points = self._x_sample
        else:
            if x_is_redshift:
                self._log_x_points = [log(1.0 + x) for x in self._x_sample]
            else:
                self._log_x_points = [log(x) for x in self._x_sample]

        # find min and max values of x, so that we avoid extrapolation if we are asked to evaluate at an x position that is out-of-bounds
        self.min_log_x = min(self._log_x_points)
        self.max_log_x = max(self._log_x_points)

        self.min_x = exp(self.min_log_x)
        self.max_x = exp(self.max_log_x)

        # prefer to be 5% away from the spline boundaries so that we are free of edge-effects
        self._min_safe_log_x = 1.05 * self.min_log_x
        self._max_safe_log_x = 0.95 * self.max_log_x

        self._min_safe_x = exp(self._min_safe_log_x)
        self._max_safe_x = exp(self._max_safe_log_x)

        self._spline = make_interp_spline(self._log_x_points, self._y_points)
        self._derivative = self._spline.derivative()

    def _theta(self, raw_x: float, log_x: float, warn_unsafe: bool = True) -> float:
        # raise an error if requested value is too far outside our range
        # we allow a 1% cushion at the top and the bottom, in which we return just the top or bottom value
        if log_x < self.min_log_x * (1.0 - sign(log_x) * SPLINE_TOP_BOTTOM_CUSHION):
            raise RuntimeError(
                f"phase_spline: spline evaluated out-of-bounds of lower limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Minimum allowed value is log_x={self.min_log_x:.5g} (raw x={self.min_x:.5g}) | Recommended safe minimum is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g})"
            )
        elif log_x < self.min_log_x:
            log_x = self.min_log_x

        if log_x > self.max_log_x * (1.0 + sign(log_x) * SPLINE_TOP_BOTTOM_CUSHION):
            raise RuntimeError(
                f"phase_spline: spline evaluated out-of-bounds of upper limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Maximum allowed value is log_x={self.max_log_x:.5g} (raw x={self.max_x:.5g}) | Recommended safe maximum is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g})"
            )
        elif log_x > self.max_log_x:
            log_x = self.max_log_x

        if warn_unsafe:
            if log_x < self._min_safe_log_x:
                print(
                    f"## WARNING (phase_spline): spline evaluated within 5% of lower limit at log_x={log_x:.5g} (raw={raw_x:.5g}) | Recommended safe minimum is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g})"
                )

            if log_x > self._max_safe_log_x:
                print(
                    f"## WARNING (phase_spline): spline evaluated within 5% of upper limit at log_x={log_x:.5g} (raw={raw_x:.5g}) | Recommended safe maximum is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g})"
                )

        return self._spline(log_x)

    def _theta_deriv(
        self, raw_x: float, log_x: float, warn_unsafe: bool = True
    ) -> float:
        # raise an error if requested value is too far outside our range
        # we allow a 1% cushion at the top and the bottom, in which we return just the top or bottom value
        if log_x < self.min_log_x * (1.0 - sign(log_x) * SPLINE_TOP_BOTTOM_CUSHION):
            raise RuntimeError(
                f"phase_spline: spline evaluated out-of-bounds of lower limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Minimum allowed value is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g}) | Recommended safe minimum is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g})"
            )
        elif log_x < self.min_log_x:
            log_x = self.min_log_x

        if log_x > self.max_log_x * (1.0 + sign(log_x) * SPLINE_TOP_BOTTOM_CUSHION):
            raise RuntimeError(
                f"phase_spline: spline evaluated out-of-bounds of upper limit at log_x={log_x:.5g} (raw x={raw_x:.5g}) | Maximum allowed value is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g}) | Recommended safe maximum is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g})"
            )
        elif log_x > self.max_log_x:
            log_x = self.max_log_x

        if warn_unsafe:
            if log_x < self._min_safe_log_x:
                print(
                    f"## WARNING (phase_spline): spline evaluated within 5% of lower limit at log_x={log_x:.5g} (raw={raw_x:.5g}) | Recommended safe minimum is log_x={self._min_safe_log_x:.5g} (raw x={self._min_safe_x:.5g})"
                )

            if log_x > self._max_safe_log_x:
                print(
                    f"## WARNING (phase_spline): spline evaluated within 5% of upper limit at log_x={log_x:.5g} (raw={raw_x:.5g}) | Recommended safe maximum is log_x={self._max_safe_log_x:.5g} (raw x={self._max_safe_x:.5g})"
                )

        return self._derivative(log_x)

    def _get_x(
        self, x: float, raw_x: float, log_x: float, x_is_log: bool
    ) -> Tuple[float, float]:
        if raw_x is not None and log_x is not None:
            return raw_x, log_x

        if raw_x is not None and log_x is None:
            raise RuntimeError(
                "phase_spline: both log_x and raw_x are required, but only raw_x was supplied"
            )

        if raw_x is None and log_x is not None:
            raise RuntimeError(
                "phase_spline: both log_x and raw_x are required, but only log_x was supplied"
            )

        if x is None:
            raise RuntimeError(
                "phase_spline: a valid combination of x or (raw_x, log_x) must be supplied"
            )

        if x_is_log:
            if self._x_is_redshift:
                return exp(x) - 1.0, x
            return exp(x), x

        if self._x_is_redshift:
            return x, log(1.0 + x)
        return x, log(x)

    def raw_theta(
        self,
        x: Optional[float] = None,
        raw_x: Optional[float] = None,
        log_x: Optional[float] = None,
        x_is_log: bool = False,
        warn_unsafe: bool = True,
    ) -> float:
        raw_x, log_x = self._get_x(x, raw_x, log_x, x_is_log)
        theta = self._theta(raw_x=raw_x, log_x=log_x, warn_unsafe=warn_unsafe)
        return theta + self._div_2pi_base * TWO_PI

    def theta_mod_2pi(
        self,
        x: Optional[float] = None,
        raw_x: Optional[float] = None,
        log_x: Optional[float] = None,
        x_is_log: bool = False,
        warn_unsafe: bool = True,
    ) -> float:
        raw_x, log_x = self._get_x(x, raw_x, log_x, x_is_log)
        theta = self._theta(raw_x=raw_x, log_x=log_x, warn_unsafe=warn_unsafe)
        return fmod(theta, TWO_PI)

    def theta_deriv(
        self,
        x: Optional[float] = None,
        raw_x: Optional[float] = None,
        log_x: Optional[float] = None,
        x_is_log: bool = False,
        log_derivative: bool = False,
        warn_unsafe: bool = True,
    ) -> float:
        raw_x, log_x = self._get_x(x, raw_x, log_x, x_is_log)

        deriv = self._theta_deriv(raw_x=raw_x, log_x=log_x, warn_unsafe=warn_unsafe)

        if log_derivative:
            # spline is computed as a function of log(x) or log(1+z) if we are working in redshift, so derivative
            # will naturally by with respect to this
            return deriv

        # otherwise, this is not a log derivative, so we need to divide by 1/x or 1/(1+z)
        if self._x_is_redshift:
            return deriv / (1.0 + raw_x)

        return deriv / raw_x


class phase_spline:
    """
    Spline the phase function for a Liouville-Green representation.
    We try to do this intelligently so that we do not lose precision when the phase is evaluated mod 2pi,
    which is eventually what will usually be done (except perhaps during Levin integration when we need
    the derivative of the phase instead).

    Since GkTk-remedial prompt 08, this is a single cubic spline of the whole sample, rebased
    around one integer cycle count so that its internal ordinates stay of order the data span (see
    the module docstring for why the previous chunked design was removed rather than fixed).
    """

    def __init__(
        self,
        x_sample: Iterable[float],
        theta_div_2pi_sample: Iterable[int],
        theta_mod_2pi_sample: Iterable[float],
        chunk_step: Optional[int] = DEFAULT_CHUNK_SIZE,
        chunk_logstep: Optional[float] = None,
        x_is_log: bool = False,
        x_is_redshift: bool = False,
        increasing: bool = True,
    ):
        # chunk_step and chunk_logstep are deprecated: they have had no effect since GkTk-remedial
        # prompt 08 (module docstring). Accepted only so that existing callers -- notably
        # LiouvilleGreen/bessel_phase.py on `main`, which still passes chunk_logstep=125 -- do not
        # need to change.
        del chunk_step, chunk_logstep

        if len(x_sample) == 0:
            raise RuntimeError("phase_spline: empty x_sample")

        assert len(x_sample) == len(theta_div_2pi_sample)
        assert len(x_sample) == len(theta_mod_2pi_sample)

        self._x_is_redshift: bool = x_is_redshift
        # `increasing` no longer orders anything: with a single spline there are no chunks to put
        # in ascending/descending order, and the underlying spline already sorts its data by x
        # regardless of this flag. Accepted and ignored, for the same signature-compatibility
        # reason as chunk_step/chunk_logstep.
        self._increasing: bool = increasing

        # Rebase every ordinate around one integer cycle count chosen from the sample, so that the
        # spline's internal ordinates are bounded by the data span rather than by the absolute
        # cycle count. We pick the *median* sample's div_2pi rather than the one with the smallest
        # |div_2pi|: for data spanning, say, div_2pi in [-1000, 0], rebasing at the median (~-500)
        # keeps ordinates within about half the data span in each direction, whereas rebasing at
        # the extreme with the smallest |div_2pi| (here 0) keeps the full span in one direction.
        # Both choices satisfy the boundedness property this replaces (max|y| <= data span + 2*pi);
        # the median roughly halves the typical ordinate magnitude.
        sorted_divs = sorted(theta_div_2pi_sample)
        div_2pi_base = sorted_divs[len(sorted_divs) // 2]

        data = [
            {
                "id": i,
                "x": x_sample[i],
                "div_2pi": theta_div_2pi_sample[i],
                "mod_2pi": theta_mod_2pi_sample[i],
            }
            for i in range(len(x_sample))
        ]

        self._spline = _rebased_spline(
            div_2pi_base=div_2pi_base,
            data=data,
            x_is_log=x_is_log,
            x_is_redshift=x_is_redshift,
        )

    @property
    def num_chunks(self) -> int:
        # Always 1 since GkTk-remedial prompt 08. Kept as a property because
        # QuadSourceIntegral.py reads it via getattr() into stored metadata
        # (WKB_phase_spline_chunks).
        return 1

    def raw_theta(self, x: float, x_is_log: bool = False) -> float:
        return self._spline.raw_theta(x=x, x_is_log=x_is_log, warn_unsafe=False)

    def theta_mod_2pi(self, x: float, x_is_log: bool = False) -> float:
        return self._spline.theta_mod_2pi(x=x, x_is_log=x_is_log, warn_unsafe=False)

    def theta_deriv(
        self, x: float, x_is_log: bool = False, log_derivative: bool = False
    ) -> float:
        return self._spline.theta_deriv(
            x=x,
            x_is_log=x_is_log,
            log_derivative=log_derivative,
            warn_unsafe=False,
        )
