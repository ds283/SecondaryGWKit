import numpy as np
from math import log, sin, cos, exp
from scipy.interpolate import UnivariateSpline


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
        if log_z > 1.01 * self._max_log_z:
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={raw_z:.5g} (max allowed z={self._max_z:.5g}, recommended limit is z <= {0.95 * self._max_z:.5g})"
            )

        # otherwise, softly cushion the spline at the top end
        if log_z > self._max_log_z:
            log_z = self._max_log_z

        # same at lower limit
        if log_z < 0.99 * self._min_log_z:
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={raw_z:.5g} (min allowed z={self._min_z:.5g}, recommended limit is z >= {1.05 * self._min_z:.5g})"
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
        theta_spline,
        sin_amplitude_spline,
        cos_amplitude_spline,
        label: str,
        max_z: float,
        min_z: float,
    ):
        self._theta_spline: UnivariateSpline = theta_spline
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
        if log_z > 1.01 * self._max_log_z:
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={raw_z:.5g} (max allowed z={self._max_z:.5g}, recommended limit is z <= {0.95 * self._max_z:.5g})"
            )

        # otherwise, softly cushion the spline at the top end
        if log_z > self._max_log_z:
            log_z = self._max_log_z

        # same at lower limit
        if log_z < 0.99 * self._min_log_z:
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={raw_z:.5g} (min allowed z={self._min_z:.5g}, recommended limit is z >= {1.05 * self._min_z:.5g})"
            )

        if log_z < self._min_log_z:
            log_z = self._min_log_z

        try:
            if self._sin_amplitude_spline is not None:
                sin_amplitude = self._sin_amplitude_spline(log_z)
                sin_factor = sin(self._theta_spline(log_z))

                sin_part = sin_amplitude * sin_factor
            else:
                sin_part = 0.0

            if self._cos_amplitude_spline is not None:
                cos_amplitude = self._cos_amplitude_spline(log_z)
                cos_factor = cos(self._theta_spline(log_z))

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
