from collections import namedtuple
from math import log, fabs, sqrt
from typing import Optional, Union, List

import ray
from scipy.interpolate import make_interp_spline

from ComputeTargets.GkSource import GkSourceProxy, GkSource
from ComputeTargets.primitive_phase import PrimitivePhase, build_phi_samples
from ComputeTargets.spline_wrappers import ZSplineWrapper, GkWKBSplineWrapper
from CosmologyConcepts import wavenumber_exit_time, redshift, wavenumber
from Datastore import DatastoreObject
from LiouvilleGreen.constants import TWO_PI
from MetadataConcepts import GkSourcePolicy
from config.defaults import DEFAULT_FLOAT_PRECISION

MIN_SPLINE_DATA_POINTS = 5


GkSourceFunctions = namedtuple(
    "GkSourceFunctions",
    [
        "numeric_region",
        "WKB_region",
        "numeric_Gk",
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

    # Levin_z is DIAGNOSTIC ONLY since prompt 10 of prompts/source-remediation. It is still
    # computed and persisted (dropping it would be a schema change for no gain, and the extract
    # scripts still plot it), but no consumer routes on it: prompt 08 replaced the
    # Green's-function-only Levin gate of QuadSourceIntegral with a partition at every factor's
    # hand-over plus one adaptive_levin_sincos call per phase group (audit 2026-09 A4/QI-6), and
    # QuadSourceIntegral now reads Levin_z only into metadata["partition"]["Levin_z_unused"].
    # GkSourcePolicy.Levin_threshold, which sets it, is likewise diagnostic from here on.
    Levin_data = _classify_Levin(source, policy, crossover_data)
    payload["Levin_z"] = Levin_data["Levin_z"]
    payload["metadata"].update(Levin_data["metadata"])

    return payload


RedshiftLike = Union[float, redshift]
StringListLike = Union[str, List[str]]


class RedshiftRangePredicate:

    def __init__(
        self,
        min_z: RedshiftLike,
        max_z: RedshiftLike,
        property_list: Optional[StringListLike] = None,
    ):
        self.min_z: RedshiftLike = min_z
        self.max_z: RedshiftLike = max_z

        self.property_list = None
        if property_list is not None:
            if isinstance(property_list, str):
                self.property_list = [property_list]
            else:
                self.property_list = property_list

    def _less_than_max(self, z: RedshiftLike):
        if isinstance(self.max_z, redshift):
            return (
                z.z <= self.max_z.z + DEFAULT_FLOAT_PRECISION
                or z.store_id == self.max_z.store_id
            )

        return float(z) <= self.max_z + DEFAULT_FLOAT_PRECISION

    def _greater_than_min(self, z: RedshiftLike):
        if isinstance(self.min_z, redshift):
            return (
                z.z >= self.min_z.z - DEFAULT_FLOAT_PRECISION
                or z.store_id == self.min_z.store_id
            )

        return float(z) >= self.min_z - DEFAULT_FLOAT_PRECISION

    def _has_properties(self, v):
        if self.property_list is None:
            return True

        return all(getattr(v, p) for p in self.property_list)

    def __call__(self, z):
        return (
            self._less_than_max(z.z_source)
            and self._greater_than_min(z.z_source)
            and self._has_properties(z)
        )


def _filter_values_from_range(
    source: GkSource,
    max_z: RedshiftLike,
    min_z: RedshiftLike,
    property_list: Optional[StringListLike] = None,
):
    predicate: RedshiftRangePredicate = RedshiftRangePredicate(
        min_z, max_z, property_list=property_list
    )

    # must be sorted into ascending order of redshift for a smoothing spline
    data = [v for v in source.values if predicate(v)]
    data.sort(key=lambda x: x.z_source.z)

    return data


# The Green's-function phase theta(z_response; z_source) is measured at a fixed response redshift
# and becomes more negative as the source redshift rises, so the conformal-time primitive enters
# with sign = -1 (campaign README Sec 2 (c), whose convention is
# tau.delta(z_a, z_b) = tau(z_b) - tau(z_a)):
#
#     theta(z_s) = -k * tau.delta(z_s, z_r) + phi(z_s)
GK_PHASE_SIGN = -1


def _build_phase(source: GkSource, WKB_data, label: str) -> PrimitivePhase:
    """
    The phase of this Green's function over its WKB region, as a ``PrimitivePhase``: the leading
    term ``-k tau.delta(z_s, z_response)`` evaluated from the background model's double-double
    conformal-time table, plus a cubic spline of the small residual ``phi``.

    This replaces the cubic spline of the stored phase that both consumers built until prompt 09
    of ``prompts/GkTk-remedial`` (``ComputeTargets/primitive_phase.py``'s module docstring has the
    full comparison). A cubic spline of ``theta`` itself carries
    ``h^4 x_source / 384`` (review Sec 5), which is 8.3e-3 rad at ``x_source = 1e7`` and
    O(1)-O(10) rad at the production ``x_source`` of 1e9-1e12: the phase it was interpolating is
    not resolvable at any affordable sample density. ``phi`` is O(1) rad and smooth, so the same
    cubic spline of *it* is six or more orders better, and the growing part of the phase is no
    longer interpolated at all.

    ``phi`` is built from the **rectified** cycle counts ``v.WKB.theta_div_2pi``, not from
    ``v.WKB.raw_theta_div_2pi``. ``GkSource.assemble_GkSource_values`` (:166-233) repairs the 2pi
    wraps of the initial-data offset that occur where the numeric stop point of a
    ``GkWKBIntegration`` moves to the next extremum between neighbouring source redshifts; those
    are genuine discontinuities of the *stored* phase, not of the physical one
    (``RECONCILIATION.md`` Sec 2 item 6, campaign decision D5), and building ``phi`` from the raw
    counts would spline straight through them.
    """
    model = source.model_proxy.get()

    k_float = float(source.k.k)
    z_anchor = source.z_response.z
    leading = model.functions.tau

    z_points = [v.z_source.z for v in WKB_data]
    theta_points = [
        v.WKB.theta_div_2pi * TWO_PI + v.WKB.theta_mod_2pi for v in WKB_data
    ]
    phi_points = build_phi_samples(
        k_float,
        leading,
        z_anchor,
        z_points,
        theta_points,
        sign=GK_PHASE_SIGN,
    )

    return PrimitivePhase(
        k_float,
        leading,
        z_anchor,
        z_points,
        phi_points,
        sign=GK_PHASE_SIGN,
        model_functions=model.functions,
        label=label,
    )


def _classify_Levin(source: GkSource, policy: GkSourcePolicy, data) -> dict:
    """
    Determine the point where we should enable Levin integration for this Green's function,
    assuming it has some WKB phase data that can be used for a Levin quadrature.
    (GkSource instances that are "numeric" only cannot be used with Levin quadrature.)
    :param source:
    :param policy:
    :param data:
    :return:
    """

    # Levin_z stays None unless the loop below finds a crossing. A Green's function whose
    # |d theta_G / d log(1+z)| never exceeds policy.Levin_threshold anywhere in its WKB range is
    # a legitimate outcome -- it means no Levin quadrature is indicated -- and the early return
    # below already reports that case as None. Without this default the loop's non-crossing path
    # left the key absent and apply_GkSource_policy:58 raised KeyError('Levin_z'), aborting the
    # --gk-source-policy-queue stage (board issue [10-classify-levin-keyerror], opened by prompt 10
    # of prompts/source-remediation, which was not allowed to edit this file).
    payload = {"Levin_z": None}
    metadata = {}

    source_type = data["type"]
    source_quality = data["quality"]
    crossover_z = data.get("crossover_z", None)

    if source_quality in ["incomplete"] or source_type in ["numeric", "fail"]:
        return {"Levin_z": None, "metadata": metadata}

    max_z = crossover_z if crossover_z is not None else source.primary_WKB_largest_z.z
    min_z = source.z_sample.min.z
    WKB_data = _filter_values_from_range(
        source, max_z=max_z, min_z=min_z, property_list="has_WKB"
    )

    # The threshold test below needs a smooth derivative. It is now available in closed form:
    # d theta / d log(1+z_s) = -k (1 + z_s)/H(z_s) + d phi / d log(1+z_s), with only the second,
    # small term coming from a spline (PrimitivePhase.theta_deriv). Previously the whole
    # derivative was obtained by differentiating a cubic spline of the growing phase.
    theta_phase: PrimitivePhase = _build_phase(source, WKB_data, "Levin classification")

    # note that we compute the logarithmic derivative of theta with respect to log(1+z)

    # We are eventually going to do the Levin quadrature in log(1+z), so it is the
    # frequency of oscillation with respect to log(1+z) that matters

    # finally, note _z_sample is guaranteed to be in descending order of redshift

    for z_source in source.z_sample:
        if max_z >= z_source.z >= min_z:
            if (
                fabs(theta_phase.theta_deriv(z_source.z, log_derivative=True))
                > policy.Levin_threshold
            ):
                payload["Levin_z"] = z_source
                metadata["Levin_z_dtheta_dlogz"] = float(
                    theta_phase.theta_deriv(z_source.z, log_derivative=True)
                )
                break

    payload["metadata"] = metadata

    return payload


# a point is good if we have more than 5% clearance;
# acceptable if we have less than 5% but more than 2.5% clearance
# marginal if we have less than 2.5% but more than 1% clearance
# otherwise minimal
CLEARANCE_GOOD = 0.05
CLEARANCE_ACCEPTABLE = 0.025
CLEARANCE_MARGINAL = 0.01


# select candidate crossover points in a given order of preference
CLASSIFICATION_BANDS = [
    ("complete", ["numeric_good", "WKB_good"]),
    ("acceptable", ["numeric_acceptable", "WKB_good"]),
    ("acceptable", ["numeric_good", "WKB_acceptable"]),
    ("acceptable", ["numeric_acceptable", "WKB_acceptable"]),
    ("marginal", ["numeric_marginal", "WKB_good"]),
    ("marginal", ["numeric_marginal", "WKB_acceptable"]),
    ("marginal", ["numeric_good", "WKB_marginal"]),
    ("marginal", ["numeric_acceptable", "WKB_marginal"]),
    ("marginal", ["numeric_marginal", "WKB_marginal"]),
    ("minimal", ["numeric_minimal", "WKB_minimal"]),
]


def _classify_crossover(source: GkSource, policy: GkSourcePolicy) -> dict:
    """
    Classify this Green's function. We classify whether the combination of numeric and WKB
    data form a complete or incomplete representation of the Green's function over the range of interest.
    We also classify whether the *quality* of the representation is good. This means that we can spline
    the data in both regions without getting too close to one of the end-points. For that we need
    some overlap between the numeric and WKB regions.
    Finally, we apply a numerical policy to decide where we should switch from using the
    numeric data to using the WKB data.
    :param source:
    :param policy:
    :return:
    """
    has_WKB_region: bool = source.primary_WKB_largest_z is not None
    has_WKB_spline_points: bool = False

    has_numeric_region: bool = source.numeric_smallest_z is not None
    has_numeric_spline_points: bool = False

    payload = {}
    metadata = {}

    # do we have WKB data?
    if has_WKB_region:
        WKB_data = _filter_values_from_range(
            source,
            max_z=source.primary_WKB_largest_z,
            min_z=source.z_sample.min,
            property_list="has_WKB",
        )

        # if we have enough data points, we can build an adequate spline. Otherwise, we can't.
        if len(WKB_data) >= MIN_SPLINE_DATA_POINTS:
            has_WKB_spline_points = True
        else:
            metadata["WKB_spline_points"] = len(WKB_data)

    # do we have numeric data?
    if has_numeric_region:
        numeric_data = _filter_values_from_range(
            source,
            max_z=source.z_sample.max,
            min_z=source.numeric_smallest_z,
            property_list="has_numeric",
        )

        if len(numeric_data) >= MIN_SPLINE_DATA_POINTS:
            has_numeric_spline_points = True
        else:
            metadata["numeric_spline_points"] = len(numeric_data)

    # if both the WKB and numeric regions are missing, we have a fail
    if (not has_numeric_region or not has_numeric_spline_points) and (
        not has_WKB_region or not has_WKB_spline_points
    ):
        payload["type"] = "fail"
        payload["crossover_z"] = None
        payload["quality"] = "incomplete"
        metadata["comment"] = "No WKB region and no numeric region"

    # if the primary WKB region is missing, or there are not enough points to construct an adequate spline,
    # we need the numeric data to go all the way *down* to the smallest relevant redshift. Otherwise we have
    # an incomplete representation
    elif not has_WKB_region or not has_WKB_spline_points:
        # can assume has_numeric_region and has_numeric_spline_points
        payload["type"] = "numeric"
        payload["crossover_z"] = None

        if source.numeric_smallest_z.store_id == source.z_sample.min.store_id:
            payload["quality"] = "complete"
        else:
            payload["quality"] = "incomplete"
            metadata["comment"] = (
                f"No WKB region. Numeric region does not cover full range. numeric_smallest_z={source.numeric_smallest_z.z:.5g}, z_sample.min={source.z_sample.min.z:.5g}"
            )

    # if the numeric region is missing, or there are not enough points to construct an adequate spline,
    # we need the WKB data to go all the way *up* to the highest relevant redshift. Otherwise we have
    # an incomplete representation
    elif not has_numeric_region or not has_numeric_spline_points:
        # can assume has_WKB_region and has_WKB_spline_points
        payload["type"] = "WKB"
        payload["crossover_z"] = None

        if source.primary_WKB_largest_z.store_id == source.z_sample.max.store_id:
            payload["quality"] = "complete"
        else:
            payload["quality"] = "incomplete"
            metadata["comment"] = (
                f"No numeric region. WKB region does not cover full range. primary_WKB_largest_z={source.primary_WKB_largest_z.z:.5g}, z_sample.max={source.z_sample.max.z:.5g}"
            )

    # otherwise, we have both a WKB region and a numeric region, and we need to decide where we are going
    # to switch between them. We're also allowed to assume that we have at least a minimum number of
    # sample points to produce both numeric and WKB splines
    else:
        payload["type"] = "mixed"

        # if there is no overlap, we have a gap, and this Green's function is incomplete
        if (
            source.numeric_smallest_z.z
            >= source.primary_WKB_largest_z.z - DEFAULT_FLOAT_PRECISION
        ):
            payload["quality"] = "incomplete"
            payload["crossover_z"] = source.numeric_smallest_z.z
            metadata["comment"] = (
                "WKB and numeric regions are both present, but the numeric region does not overlap with the primary WKB region"
            )

        else:
            # get numeric policy from the policy object
            policy_name = policy.numeric_policy

            # Can assume there is an overlap. Filter out all values in the overlapping region.
            overlap_values = _filter_values_from_range(
                source,
                max_z=source.primary_WKB_largest_z,
                min_z=source.numeric_smallest_z,
                property_list=["has_WKB", "has_numeric"],
            )

            # For each value, figure out how close we would be to the corresponding end of the spline, bearing in mind
            # that we spline in log(1+z), not z itself
            numeric_logz_low = log(1.0 + source.numeric_smallest_z.z)
            WKB_logz_high = log(1.0 + source.primary_WKB_largest_z.z)

            def classify_point(v) -> dict:
                logz = log(1.0 + v.z_source.z)
                numeric_clearance = (logz - numeric_logz_low) / numeric_logz_low
                WKB_clearance = (WKB_logz_high - logz) / WKB_logz_high

                return {
                    "value": v,
                    "numeric_clearance": numeric_clearance,
                    "WKB_clearance": WKB_clearance,
                    "numeric_good": numeric_clearance > CLEARANCE_GOOD,
                    "WKB_good": WKB_clearance > CLEARANCE_GOOD,
                    "numeric_acceptable": numeric_clearance > CLEARANCE_ACCEPTABLE,
                    "WKB_acceptable": WKB_clearance > CLEARANCE_ACCEPTABLE,
                    "numeric_marginal": numeric_clearance > CLEARANCE_MARGINAL,
                    "WKB_marginal": WKB_clearance > CLEARANCE_MARGINAL,
                    "numeric_minimal": numeric_clearance > 0.0,
                    "WKB_minimal": WKB_clearance > 0.0,
                }

            # classify each value in the overlapping region
            classified_values = [classify_point(v) for v in overlap_values]

            crossover_z = None
            quality = None
            i: int = 0
            while crossover_z is None and i < len(CLASSIFICATION_BANDS):
                # find properties that are required to classify as target_quality
                target_quality, property_list = CLASSIFICATION_BANDS[i]
                i += 1

                # find points that satisfy these properties (if any)
                filtered_values = [
                    item
                    for item in classified_values
                    if all(item[p] for p in property_list)
                ]

                # if there are no points, move on
                if len(filtered_values) == 0:
                    continue

                quality = target_quality

                # on the maximize-WKB policy we should switch to WKB as soon as possible
                if policy_name == "maximize-WKB":
                    sorted_values = sorted(
                        filtered_values, key=lambda v: v["value"].z_source, reverse=True
                    )
                    crossover_z = sorted_values[0]["value"].z_source
                    break

                # on the maximize-numeric policy we should switch to WKB as late as possible
                elif policy_name == "minimize-WKB":
                    sorted_values = sorted(
                        filtered_values, key=lambda v: v["value"].z_source
                    )
                    crossover_z = sorted_values[0]["value"].z_source
                    break

                else:
                    raise RuntimeError(
                        f'Unknown GkSourcePolicy numeric policy "{policy_name}"'
                    )

            if crossover_z is None:
                # should get enough points for a good spline if we set the crossover scale to _primary_WKB_largest_z, since
                # we checked above that enough points are available.
                crossover_z = source.primary_WKB_largest_z
                quality = "minimal"

                WKB_data = _filter_values_from_range(
                    source,
                    max_z=crossover_z,
                    min_z=source.z_sample.min,
                    property_list="has_WKB",
                )
                numeric_data = _filter_values_from_range(
                    source,
                    max_z=source.z_sample.max,
                    min_z=source.numeric_smallest_z,
                    property_list="has_numeric",
                )

                metadata["comment"] = (
                    f"Calculation of crossover_z did not yield a result. Crossover set to largest z in primary WKB region: z={source.primary_WKB_largest_z.z:.5g}. Numeric spline contains {len(numeric_data)} points. WKB spline contains {len(WKB_data)} points."
                )

            payload["crossover_z"] = crossover_z.z
            payload["quality"] = quality

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

        numeric_region = None
        numeric_Gk = None

        if source.numeric_smallest_z is not None:
            max_z = source.z_sample.max
            min_z = source.numeric_smallest_z

            if self._crossover_z is not None and self._crossover_z < min_z.z:
                raise RuntimeError(
                    f"GkSourcePolicyData: inconsistent values of crossover_z={self._crossover_z:.5g} and smallest numeric z={min_z.z:.5g}"
                )

            # must be sorted into ascending order of redshift for a smoothing spline
            numeric_data = _filter_values_from_range(
                source, max_z, min_z, property_list="has_numeric"
            )
            if len(numeric_data) >= MIN_SPLINE_DATA_POINTS:
                if (
                    numeric_data[0].z_source.z / min_z.z > 1.0 + DEFAULT_FLOAT_PRECISION
                    or numeric_data[-1].z_source.z / max_z.z
                    < 1.0 - DEFAULT_FLOAT_PRECISION
                ):
                    self._report_missing_data(
                        "numeric data", source, numeric_data, max_z, min_z
                    )

                numeric_Gk_data = [
                    (log(1.0 + v.z_source.z), v.numeric.G) for v in numeric_data
                ]
                numeric_Gk_x, numeric_Gk_y = zip(*numeric_Gk_data)

                _numeric_Gk_spline = make_interp_spline(
                    numeric_Gk_x,
                    numeric_Gk_y,
                )

                numeric_Gk = ZSplineWrapper(
                    _numeric_Gk_spline, "numeric Gk", max_z.z, min_z.z, log_z=True
                )

                numeric_region = (max_z.z, min_z.z)

        WKB_region = None
        WKB_Gk = None
        WKB_theta_phase = None
        WKB_sin_amplitude = None

        if source.primary_WKB_largest_z is not None:
            max_z = source.primary_WKB_largest_z
            min_z = source.z_sample.min

            if self._crossover_z is not None and self._crossover_z > max_z.z:
                raise RuntimeError(
                    f"GkSourcePolicyData: inconsistent values of crossover_z={self._crossover_z:.5g} and largest WKB z={max_z.z:.5g}"
                )

            WKB_data = _filter_values_from_range(
                source, max_z, min_z, property_list="has_WKB"
            )
            if len(WKB_data) >= MIN_SPLINE_DATA_POINTS:
                if (
                    WKB_data[0].z_source.z / min_z.z > 1.0 + DEFAULT_FLOAT_PRECISION
                    or WKB_data[-1].z_source.z / max_z.z < 1.0 - DEFAULT_FLOAT_PRECISION
                ):
                    self._report_missing_data(
                        "WKB data", source, WKB_data, max_z, min_z
                    )

                sin_amplitude_data = [
                    (
                        log(1.0 + v.z_source.z),
                        v.WKB.sin_coeff * sqrt(v.WKB.H_ratio / sqrt(v.omega_WKB_sq)),
                    )
                    for v in WKB_data
                ]

                sin_amplitude_x, sin_amplitude_y = zip(*sin_amplitude_data)
                _sin_amplitude_spline = make_interp_spline(
                    sin_amplitude_x,
                    sin_amplitude_y,
                )

                # The phase is evaluated from the conformal-time table plus a spline of the small
                # residual, not splined itself (see _build_phase). GkWKBSplineWrapper consumes it
                # duck-typed, through theta_mod_2pi() alone.
                WKB_theta_phase = _build_phase(source, WKB_data, "Gk WKB phase")

                WKB_Gk = GkWKBSplineWrapper(
                    WKB_theta_phase,
                    _sin_amplitude_spline,
                    None,
                    "Gk WKB",
                    max_z.z,
                    min_z.z,
                )
                WKB_sin_amplitude = ZSplineWrapper(
                    _sin_amplitude_spline, "sin amplitude", max_z.z, min_z.z, log_z=True
                )

                WKB_region = (max_z.z, min_z.z)

        self._functions = GkSourceFunctions(
            numeric_region=numeric_region,
            numeric_Gk=numeric_Gk,
            WKB_region=WKB_region,
            WKB_Gk=WKB_Gk,
            phase=WKB_theta_phase,
            sin_amplitude=WKB_sin_amplitude,
            type=self._type,
            quality=self._quality,
            crossover_z=self._crossover_z,
        )

    def _report_missing_data(self, label, source, data, max_z, min_z):
        print("!! ERROR (GkSource.create_functions)")
        print(
            f"     ** {label} missing: intended region = ({max_z.z:.5g}, {min_z.z:.5g}), but available only between ({data[-1].z_source.z:.5g}, {data[0].z_source.z:.5g})"
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
        df.to_csv(
            f"GkSourcePolicyData-create_functions-ERRORDATA-GkSource-{source.store_id}.csv",
            header=True,
            index=False,
        )
        raise RuntimeError(
            f"GkSource.create_functions: {label} missing = intended region = ({max_z.z:.5g}, {min_z.z:.5g}), but available only between ({data[-1].z_source.z:.5g}, {data[0].z_source.z:.5g})."
        )
