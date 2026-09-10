from math import log, sin, cos, exp

import numpy as np
from scipy.interpolate import UnivariateSpline

from LiouvilleGreen.phase_spline import phase_spline


# Fractional slack allowed outside a spline's declared range before an evaluation is rejected
# rather than softly clamped.
SPLINE_BOUND_SLACK = 0.01


def _outward(bound: float, direction: int) -> float:
    """
    Move `bound` by SPLINE_BOUND_SLACK of its own magnitude: upward for direction=+1, downward
    for direction=-1.

    The multiplier has to follow the sign of the bound. The obvious `1.01 * bound` /
    `0.99 * bound` is only "outward" while `bound > 0`, and log(1+z) is negative for z < 0: there,
    scaling a lower bound by 0.99 moves the threshold *inward*, so the wrapper rejected a band
    lying inside its own declared range. That is why a GenericEOS model, whose T(z) spline has a
    negative min_z, could not be evaluated at its own declared floor. For a positive bound this
    returns exactly `1.01 * bound` or `0.99 * bound` as before, so every other spline in this
    repository -- all of which have positive bounds -- is unaffected.
    """
    factor = 1.0 + direction * SPLINE_BOUND_SLACK
    if bound < 0.0:
        factor = 1.0 - direction * SPLINE_BOUND_SLACK
    return factor * bound


class ZSplineWrapper:
    def __init__(
        self,
        spline,
        label: str,
        max_z: float,
        min_z: float,
        log_z: bool = True,
        deriv=False,
    ):
        self._spline = spline
        self._label = label

        self._min_z = min_z
        self._max_z = max_z

        self._min_log_z = log(1.0 + min_z)
        self._max_log_z = log(1.0 + max_z)

        self._uses_log_z = log_z
        self._is_deriv = deriv

    def __call__(self, z: float, z_is_log: bool = False) -> float:
        if z_is_log:
            log_z = z
            raw_z = exp(z) - 1.0
        else:
            log_z = log(1.0 + z)
            raw_z = z

        # if some way out of bounds, reject
        if log_z > _outward(self._max_log_z, +1):
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={raw_z:.5g} (max allowed z={self._max_z:.5g}, recommended limit is z <= {_outward(self._max_z, -1):.5g})"
            )

        # otherwise, softly cushion the spline at the top end
        if log_z > self._max_log_z:
            log_z = self._max_log_z

        # same at lower limit
        if log_z < _outward(self._min_log_z, -1):
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={raw_z:.5g} (min allowed z={self._min_z:.5g}, recommended limit is z >= {_outward(self._min_z, +1):.5g})"
            )

        if log_z < self._min_log_z:
            log_z = self._min_log_z

        if self._uses_log_z:
            if self._is_deriv:
                # the spline will compute d/d(log (1+z)), so to get the raw derivative we need to divide by 1+z
                return np.float64(self._spline(log_z)) / (1.0 + raw_z)

            return np.float64(self._spline(log_z))

        return np.float64(self._spline(raw_z))


class GkWKBSplineWrapper:
    def __init__(
        self,
        theta_spline: phase_spline,
        sin_amplitude_spline,
        cos_amplitude_spline,
        label: str,
        max_z: float,
        min_z: float,
    ):
        self._theta_spline: phase_spline = theta_spline
        self._sin_amplitude_spline: UnivariateSpline = sin_amplitude_spline
        self._cos_amplitude_spline: UnivariateSpline = cos_amplitude_spline

        self._label = label

        self._min_z = min_z
        self._max_z = max_z

        self._min_log_z = log(1.0 + min_z)
        self._max_log_z = log(1.0 + max_z)

    def __call__(self, z: float, z_is_log: bool = False) -> float:
        if z_is_log:
            log_z = z
            raw_z = exp(z) - 1.0
        else:
            log_z = log(1.0 + z)
            raw_z = z

        # if some way out of bounds, reject
        if log_z > _outward(self._max_log_z, +1):
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={raw_z:.5g} (max allowed z={self._max_z:.5g}, recommended limit is z <= {_outward(self._max_z, -1):.5g})"
            )

        # otherwise, softly cushion the spline at the top end
        if log_z > self._max_log_z:
            log_z = self._max_log_z

        # same at lower limit
        if log_z < _outward(self._min_log_z, -1):
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={raw_z:.5g} (min allowed z={self._min_z:.5g}, recommended limit is z >= {_outward(self._min_z, +1):.5g})"
            )

        if log_z < self._min_log_z:
            log_z = self._min_log_z

        try:
            if self._sin_amplitude_spline is not None:
                sin_amplitude = self._sin_amplitude_spline(log_z)
                sin_factor = sin(self._theta_spline.theta_mod_2pi(log_z, x_is_log=True))

                sin_part = sin_amplitude * sin_factor
            else:
                sin_part = 0.0

            if self._cos_amplitude_spline is not None:
                cos_amplitude = self._cos_amplitude_spline(log_z)
                cos_factor = cos(self._theta_spline.theta_mod_2pi(log_z, x_is_log=True))

                cos_part = cos_amplitude * cos_factor
            else:
                cos_part = 0.0
        except ValueError as e:
            print(
                f"-- recorded z_interval (max, min) = ({self._max_z:.5g}, {self._min_z:.5g})"
            )
            print(f"-- requested z value = {raw_z:.5g}")
            raise e

        return sin_part + cos_part
