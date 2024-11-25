from collections import namedtuple
from math import log, fabs, sqrt
from typing import Optional

import ray
from scipy.interpolate import InterpolatedUnivariateSpline

from ComputeTargets.GkSource import GkSourceProxy, GkSource
from ComputeTargets.spline_wrappers import ZSplineWrapper, GkWKBSplineWrapper
from CosmologyConcepts import wavenumber_exit_time, redshift, wavenumber
from Datastore import DatastoreObject
from LiouvilleGreen.phase_spline import phase_spline
from MetadataConcepts import GkSourcePolicy
from defaults import DEFAULT_FLOAT_PRECISION

MIN_SPLINE_DATA_POINTS = 5


GkSourceFunctions = namedtuple(
    "GkSourceFunctions",
    [
        "numerical_region",
        "WKB_region",
        "numerical_Gk",
        "WKB_Gk",
        "phase",
        "sin_amplitude",
        "type",
        "quality",
        "crossover_z",
    ],
)


@ray.remote
def apply_GkSource_policy(source_proxy: GkSourceProxy, policy: GkSourcePolicy) -> dict:
    source: GkSource = source_proxy.get()

    if not source.available:
        raise

    payload = {"metadata": {}}

    crossover_data = _classify_crossover(source, policy)
    payload["type"] = crossover_data["type"]
    payload["quality"] = crossover_data["quality"]
    payload["crossover_z"] = crossover_data.get("crossover_z", None)
    payload["metadata"] = crossover_data["metadata"]

    Levin_data = _classify_Levin(source, policy, crossover_data)
    payload["Levin_z"] = Levin_data["Levin_z"]
    payload["metadata"].update(Levin_data["metadata"])

    return payload


def _build_WKB_values(source: GkSource, max_z, min_z):
    def max_predicate(z):
        if isinstance(max_z, redshift):
            return (
                z.z <= max_z.z + DEFAULT_FLOAT_PRECISION or z.store_id == max_z.store_id
            )

        return z.z <= max_z + DEFAULT_FLOAT_PRECISION

    def min_predicate(z):
        if isinstance(min_z, redshift):
            return (
                z.z >= min_z.z - DEFAULT_FLOAT_PRECISION or z.store_id == min_z.store_id
            )

        return z.z >= min_z - DEFAULT_FLOAT_PRECISION

    # must be sorted into ascending order of redshift for a smoothing spline
    WKB_data = [
        v
        for v in source.values
        if v.has_WKB and max_predicate(v.z_source) and min_predicate(v.z_source)
    ]

    WKB_data.sort(key=lambda x: x.z_source.z)

    return WKB_data


def _classify_Levin(source: GkSource, policy: GkSourcePolicy, data) -> dict:
    # find the Levin point, if we are using WKB data

    payload = {}
    metadata = {}

    source_type = data["type"]
    source_quality = data["quality"]
    crossover_z = data.get("crossover_z", None)

    if (
        source_type == "WKB" or source_type == "mixed"
    ) and source_quality != "incomplete":
        max_z = (
            crossover_z if crossover_z is not None else source.primary_WKB_largest_z.z
        )
        min_z = source.z_sample.min.z
        WKB_data = _build_WKB_values(source, max_z=max_z, min_z=min_z)
        theta_data = [
            (
                log(1.0 + v.z_source.z),
                v.WKB.theta,
            )
            for v in WKB_data
        ]
        theta_x, theta_y = zip(*theta_data)
        theta_spline = InterpolatedUnivariateSpline(theta_x, theta_y, ext="raise")

        # mark as deriv=False so that we get the raw spline deritive d theta / d log(1+z)
        # We are eventually going to do the Levin quadrature in log(1+z), so it is the
        # frequency of oscillation with respect to log(1+z) that matters
        theta_deriv = ZSplineWrapper(
            theta_spline.derivative(),
            "theta derivative",
            max_z,
            min_z,
            log_z=True,
            deriv=False,
        )

        # _z_sample is guaranteed to be in descending order of redshift
        for z_source in source.z_sample:
            if max_z >= z_source.z >= min_z:
                if fabs(theta_deriv(z_source.z)) > policy.Levin_threshold:
                    payload["Levin_z"] = z_source
                    metadata["Levin_z_dtheta_dz"] = float(theta_deriv(z_source.z))
                    break

    if "Levin_z" not in payload:
        payload["Levin_z"] = None
    payload["metadata"] = metadata
    return payload


def _classify_crossover(source: GkSource, policy: GkSourcePolicy) -> dict:
    has_WKB_spline_points = False

    payload = {}
    metadata = {}

    if source.primary_WKB_largest_z is not None:
        WKB_data = _build_WKB_values(
            source, max_z=source.primary_WKB_largest_z, min_z=source.z_sample.min
        )

        if len(WKB_data) >= MIN_SPLINE_DATA_POINTS:
            has_WKB_spline_points = True
        else:
            metadata["WKB_spline_points"] = len(WKB_data)

    # if primary WKB region is missing, or there are not enough points to construct an adequate spline,
    # we need numerical data to go all the way down to lowest redshift
    if source.primary_WKB_largest_z is None or not has_WKB_spline_points:
        payload["type"] = "numeric"
        payload["crossover_z"] = None

        if source.numerical_smallest_z.store_id == source.z_sample.min.store_id:
            payload["quality"] = "complete"
        else:
            payload["quality"] = "incomplete"
            metadata["comment"] = (
                "No WKB region, and numeric region does not extend to z_sample.min"
            )

    # if numerical data is missing, we need WKB data to go all the way up to the highest redshift
    elif source.numerical_smallest_z is None:
        payload["type"] = "WKB"
        payload["crossover_z"] = None

        if (
            source.primary_WKB_largest_z.store_id == source.z_sample.max.store_id
            and has_WKB_spline_points
        ):
            payload["quality"] = "complete"
        else:
            payload["quality"] = "incomplete"
            metadata["comment"] = (
                "No numeric region, and WKB region does not extend to z_sample.max"
            )

    # otherwise, there should be an overlap region
    else:
        # we're allowed to assume that the primary WKB region begins at z_sample.min, and the numerical region
        # (if it is present) begins at z_sample.max

        # we're also allowed to assume that we have at least a minimum number of sample points to produce a spline
        # between _primary_WKB_largest_z and z_min
        payload["type"] = "mixed"

        if (
            source.numerical_smallest_z.z
            >= source.primary_WKB_largest_z.z - DEFAULT_FLOAT_PRECISION
        ):
            payload["quality"] = "incomplete"
            payload["crossover_z"] = source.numerical_smallest_z.z
            metadata["comment"] = (
                "Both WKB and numeric regions are present, but numeric region does not extend below largest redshift in primary WKB region"
            )

        else:
            # can assume _numerical_smallest_z is strictly smaller than _primary_WKB_largest_z

            # find minimum of search window
            # we want this not to be too close the lower bound at _numerical_smallest_z, but it must fall below the upper bound at
            # _primary_WKB_largest_z
            crossover_trial_min_z_options = [
                1.06 * source.numerical_smallest_z.z,
                1.03 * source.numerical_smallest_z.z,
                1.015 * source.numerical_smallest_z.z,
                source.numerical_smallest_z.z,
            ]
            crossover_trial_min_z_options = [
                z
                for z in crossover_trial_min_z_options
                if z <= source.primary_WKB_largest_z.z + DEFAULT_FLOAT_PRECISION
            ]
            crossover_trial_min_z = max(crossover_trial_min_z_options)

            # find maximum of search window
            # we want this not to be too close to the upper bound at _primary_WKB_largest_z, but it must fall above the lower bound at
            # _numerical_smallest_z
            crossover_trial_max_z_options = [
                0.94 * source.primary_WKB_largest_z.z,
                0.97 * source.primary_WKB_largest_z.z,
                0.985 * source.primary_WKB_largest_z.z,
                source.primary_WKB_largest_z.z,
            ]
            crossover_trial_max_z_options = [
                z
                for z in crossover_trial_max_z_options
                if z >= source.numerical_smallest_z.z - DEFAULT_FLOAT_PRECISION
            ]
            crossover_trial_max_z = min(crossover_trial_max_z_options)

            # work through the search window, trying to find a crossover point at which we have enough data points to build a sensible spline
            # after a limited number of iterations we give up
            crossover_z = None
            step_width = (crossover_trial_max_z - crossover_trial_min_z) / 10.0

            iterations = 0
            # start near the lower end, so that we include as many numerical results as possible.
            # We probably trust the direct numerical results more than the WKB approximation.
            trial_z = crossover_trial_min_z
            while (
                iterations < 11
                and crossover_z is None
                and trial_z <= crossover_trial_max_z
            ):
                WKB_data = _build_WKB_values(
                    source, max_z=trial_z, min_z=source.z_sample.min
                )

                # are there enough data points to get a sensible spline?
                if len(WKB_data) >= MIN_SPLINE_DATA_POINTS:
                    crossover_z = trial_z

                    WKB_clearance = trial_z / source.primary_WKB_largest_z.z
                    numerical_clearance = trial_z / source.numerical_smallest_z.z

                    # try to classify the quality of this crossover choice
                    if WKB_clearance < 0.95 and numerical_clearance > 1.05:
                        payload["quality"] = "complete"
                    elif WKB_clearance < 0.985 and numerical_clearance > 1.025:
                        payload["quality"] = "acceptable"
                    elif WKB_clearance < 0.99 and numerical_clearance > 1.01:
                        payload["quality"] = "marginal"
                    else:
                        payload["quality"] = "minimal"

                    metadata["WKB_clearance"] = WKB_clearance
                    metadata["numerical_clearance"] = numerical_clearance
                    metadata["WKB_crossover_spline_points"] = len(WKB_data)
                    break

                iterations = iterations + 1
                trial_z = trial_z + step_width

            if crossover_z is None:
                # should get enough points for a good spline if we set the crossover scale to _primary_WKB_largest_z, since
                # we checked above that enough points are available.
                # The downside of this choice is only that it throws away numerical information that is potentially more accurate.
                crossover_z = source.primary_WKB_largest_z.z
                payload["quality"] = "minimal"

                WKB_data = _build_WKB_values(
                    source, max_z=crossover_z, min_z=source.z_sample.min
                )

                metadata["comment"] = (
                    f"Calculation of crossover_z did not converge. WKB spline contains {len(WKB_data)} points."
                )

            payload["crossover_z"] = crossover_z

    payload["metadata"] = metadata
    return payload


class GkSourcePolicyData(DatastoreObject):
    def __init__(
        self,
        payload,
        source: GkSourceProxy,
        policy: GkSourcePolicy,
        k: wavenumber_exit_time,
        label: Optional[str] = None,
    ):
        self._label = label
        self._k_exit = k
        self._source_proxy: GkSourceProxy = source
        self._policy: GkSourcePolicy = policy

        if payload is not None:
            DatastoreObject.__init__(self, payload["store_id"])

            self._type = payload["type"]
            self._quality = payload["quality"]
            self._crossover_z = payload["crossover_z"]
            self._Levin_z = payload["Levin_z"]

            self._metadata = payload["metadata"]
        else:
            DatastoreObject.__init__(self, None)

            self._type = None
            self._quality = None
            self._crossover_z = None
            self._Levin_z = None

            self._metadata = {}

        self._functions = None
        self._compute_ref = None

    def compute(self):
        if self._type is not None:
            raise RuntimeError("values have already been computed")

        if self._source_proxy.k.store_id != self._k_exit.k.store_id:
            raise RuntimeError(
                f"GkSourcePolicyData.compute(): supplied source has incompatible k-mode (store_id={self._source_proxy.k.store_id}, expected store_id={self._k_exit.k.store_id})"
            )

        self._compute_ref = apply_GkSource_policy.remote(
            self._source_proxy,
            self._policy,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "GkSourcePolicyData: store() called, but no compute() is in progress"
            )

        # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        # if not, return None
        if len(resolved) == 0:
            return None

        payload = ray.get(self._compute_ref)
        self._compute_ref = None

        self._type = payload["type"]
        self._quality = payload["quality"]
        self._crossover_z = payload["crossover_z"]
        self._Levin_z = payload["Levin_z"]
        self._metadata = payload["metadata"]

    @property
    def k(self) -> wavenumber:
        return self._k_exit.k

    @property
    def policy(self) -> GkSourcePolicy:
        return self._policy

    @property
    def type(self) -> Optional[str]:
        if self._type is None:
            raise RuntimeError("values have not yet been populated")

        return self._type

    @property
    def quality(self) -> Optional[str]:
        if self._type is None:
            raise RuntimeError("values have not yet been populated")

        return self._quality

    @property
    def crossover_z(self) -> Optional[float]:
        if self._type is None:
            raise RuntimeError("values have not yet been populated")

        return self._crossover_z

    @property
    def Levin_z(self) -> Optional[float]:
        if self._type is None:
            raise RuntimeError("values have not yet been populated")

        return self._Levin_z

    @property
    def metadata(self) -> dict:
        if self._type is None:
            raise RuntimeError("values have not yet been populated")

        return self._metadata

    @property
    def functions(self) -> GkSourceFunctions:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "GkSource: attempt to construct functions, but _do_not_populate is set"
            )

        if self._type is None:
            raise RuntimeError("values have not yet been populated")

        if self._functions is None:
            self._create_functions()

        return self._functions

    def _create_functions(self):
        source = self._source_proxy.get()

        numerical_region = None
        numerical_Gk = None

        if source.numerical_smallest_z is not None:
            max_z = source.z_sample.max
            min_z = source.numerical_smallest_z

            if self._crossover_z is not None and self._crossover_z < min_z.z:
                raise RuntimeError(
                    f"GkSourcePolicyData: inconsistent values of crossover_z={self._crossover_z:.5g} and smallest numerical z={min_z.z:.5g}"
                )

            # must be sorted into ascending order of redshift for a smoothing spline
            numerical_data = [
                v
                for v in source.values
                if v.has_numeric
                and (
                    v.z_source.z <= max_z.z + DEFAULT_FLOAT_PRECISION
                    or v.z_source.store_id == max_z.store_id
                )
                and (
                    v.z_source.z >= min_z.z - DEFAULT_FLOAT_PRECISION
                    or v.z_source.store_id == min_z.store_id
                )
            ]

            if len(numerical_data) >= MIN_SPLINE_DATA_POINTS:
                numerical_region = (max_z.z, min_z.z)

                numerical_data.sort(key=lambda x: x.z_source.z)

                numerical_Gk_data = [
                    (log(1.0 + v.z_source.z), v.numeric.G) for v in numerical_data
                ]
                numerical_Gk_x, numerical_Gk_y = zip(*numerical_Gk_data)

                _numerical_Gk_spline = InterpolatedUnivariateSpline(
                    numerical_Gk_x,
                    numerical_Gk_y,
                    ext="raise",
                )

                numerical_Gk = ZSplineWrapper(
                    _numerical_Gk_spline, "numerical Gk", max_z.z, min_z.z, log_z=True
                )

        WKB_region = None
        WKB_Gk = None
        WKB_theta = None
        WKB_theta_deriv = None
        WKB_sin_amplitude = None
        if source.primary_WKB_largest_z is not None:
            max_z = source.primary_WKB_largest_z
            min_z = source.z_sample.min

            if self._crossover_z is not None and self._crossover_z > max_z.z:
                raise RuntimeError(
                    f"GkSourcePolicyData: inconsistent values of crossover_z={self._crossover_z:.5g} and largest WKB z={max_z.z:.5g}"
                )

            WKB_data = _build_WKB_values(source, max_z, min_z)

            if len(WKB_data) >= MIN_SPLINE_DATA_POINTS:
                WKB_region = (max_z.z, min_z.z)

                if (
                    WKB_data[0].z_source.z / WKB_region[1]
                    > 1.0 + DEFAULT_FLOAT_PRECISION
                    or WKB_data[-1].z_source.z / WKB_region[0]
                    < 1.0 - DEFAULT_FLOAT_PRECISION
                ):
                    print("!! ERROR (GkSource.create_functions)")
                    print(
                        f"     ** WKB data missing: intended region = ({WKB_region[0]:.5g}, {WKB_region[1]:.5g}), but data available only between ({WKB_data[-1].z_source.z:.5g}, {WKB_data[0].z_source.z:.5g})"
                    )
                    print(
                        f"        GkSource (store_id={self.store_id}): z_response = {source.z_response.z:.5g} (store_id={source.z_response.store_id}), type = {self._type}, quality label = {self._quality}"
                    )
                    z_source_limit = sqrt(
                        source._k_exit.z_exit_subh_e3 * source._k_exit.z_exit_subh_e4
                    )
                    print(
                        f"        k = {source._k_exit.k.k_inv_Mpc:.5g}/Mpc, z_exit = {source._k_exit.z_exit:.5g}, z_source_limit = {z_source_limit:.5g}"
                    )
                    df = source.values_as_DataFrame()
                    df.to_csv("ERROR_VALUES.csv", header=True, index=False)
                    raise RuntimeError(
                        f"GkSource.create_functions: WKB data missing = intended region = ({WKB_region[0]:.5g}, {WKB_region[1]:.5g}), but data available only between ({WKB_data[-1].z_source.z:.5g}, {WKB_data[0].z_source.z:.5g})."
                    )

                sin_amplitude_data = [
                    (
                        log(1.0 + v.z_source.z),
                        v.WKB.sin_coeff * sqrt(v.WKB.H_ratio / sqrt(v.omega_WKB_sq)),
                    )
                    for v in WKB_data
                ]

                sin_amplitude_x, sin_amplitude_y = zip(*sin_amplitude_data)
                _sin_amplitude_spline = InterpolatedUnivariateSpline(
                    sin_amplitude_x,
                    sin_amplitude_y,
                    ext="raise",
                )

                theta_x_points = [log(1.0 + v.z_source.z) for v in WKB_data]
                theta_div_2pi_points = [v.WKB.theta_div_2pi for v in WKB_data]
                theta_mod_2pi_points = [v.WKB.theta_mod_2pi for v in WKB_data]
                _theta_spline = phase_spline(
                    theta_x_points,
                    theta_div_2pi_points,
                    theta_mod_2pi_points,
                    x_is_log=True,
                    chunk_step=None,
                    chunk_logstep=50,
                )

                WKB_Gk = GkWKBSplineWrapper(
                    _theta_spline,
                    _sin_amplitude_spline,
                    None,
                    "Gk WKB",
                    max_z.z,
                    min_z.z,
                )
                WKB_sin_amplitude = ZSplineWrapper(
                    _sin_amplitude_spline, "sin amplitude", max_z.z, min_z.z, log_z=True
                )

        self._functions = GkSourceFunctions(
            numerical_region=numerical_region,
            numerical_Gk=numerical_Gk,
            WKB_region=WKB_region,
            WKB_Gk=WKB_Gk,
            phase=_theta_spline,
            sin_amplitude=WKB_sin_amplitude,
            type=self._type,
            quality=self._quality,
            crossover_z=self._crossover_z,
        )
