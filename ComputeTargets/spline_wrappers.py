from math import log, sin, cos


class ZSplineWrapper:
    def __init__(
        self, spline, label: str, max_z: float, min_z: float, log_z: bool = True
    ):
        self._spline = spline
        self._label = label

        self._min_z = min_z
        self._max_z = max_z

        self._log_z = log_z

    def __call__(self, z: float) -> float:
        if z > self._max_z:
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={z:.5g} (max allowed z={self._max_z:.5g}, recommended limit is z <= {0.95 * self._max_z:.5g})"
            )

        if z < self._min_z:
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={z:.5g} (min allowed z={self._min_z:.5g}, recommended limit is z >= {1.05 * self._min_z:.5g})"
            )

        if self._log_z:
            z = log(z)

        return self._spline(z)


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
        self._theta_spline = theta_spline
        self._sin_amplitude_spline = sin_amplitude_spline
        self._cos_amplitude_spline = cos_amplitude_spline

        self._label = label

        self._min_z = min_z
        self._max_z = max_z

    def __call__(self, z: float) -> float:
        if z > self._max_z:
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={z:.5g} (max allowed z={self._max_z:.5g}, recommended limit is z <= {0.95 * self._max_z:.5g})"
            )

        if z < self._min_z:
            raise RuntimeError(
                f"GkSource.function: evaluated {self._label} out of bounds @ z={z:.5g} (min allowed z={self._min_z:.5g}, recommended limit is z >= {1.05 * self._min_z:.5g})"
            )

        log_z = log(z)

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

        return sin_part + cos_part
