from collections import namedtuple
from typing import Optional, List

import ray
from math import fabs, pi, floor

from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.GkNumericalIntegration import GkNumericalValue
from ComputeTargets.GkWKBIntegration import GkWKBValue
from CosmologyConcepts import wavenumber_exit_time, redshift, wavenumber, redshift_array
from Datastore import DatastoreObject
from MetadataConcepts import store_tag, tolerance
from defaults import DEFAULT_ABS_TOLERANCE
from utilities import check_units

NumericData = namedtuple("NumericData", ["G", "Gprime"])
WKBData = namedtuple("WKBData", ["theta", "H_ratio", "sin_coeff", "cos_coeff", "G_WKB"])


@ray.remote
def marshal_values(z_sample: redshift_array, numeric_data, WKB_data):
    values = []

    # work through the z_sample array, and check whether there is any numeric or WKB data for each sample point.
    # We do this in reverse order (from low to high redshift). This is done so that we can try to smooth the phase function.
    # It will have step-like discontinuities due to the way we calculate it.
    last_theta = None
    for z_source in reversed(list(z_sample)):
        numeric: GkNumericalValue = numeric_data.get(z_source.store_id, None)
        WKB: GkWKBValue = WKB_data.get(z_source.store_id, None)

        if numeric is None and WKB is None:
            raise RuntimeError(
                f"marshal_values: no data supplied for source redshift z={z_source.z:.5g}"
            )

        if numeric is not None and WKB is not None:
            if fabs(numeric.analytic_G - WKB.analytic_G) > DEFAULT_ABS_TOLERANCE:
                raise RuntimeError(
                    "marshal_values: analytic G values unexpectedly differ by a large amount"
                )

            if (
                fabs(numeric.analytic_Gprime - WKB.analytic_Gprime)
                > DEFAULT_ABS_TOLERANCE
            ):
                raise RuntimeError(
                    "marshal_values: analytic G values unexpectedly differ by a large amount"
                )

        theta = WKB.theta if WKB is not None else None
        if theta is not None and last_theta is not None:
            abs_diff = fabs(theta - last_theta)
            if abs_diff > 2.0 * pi:
                n = int(floor(abs_diff / (2.0 * pi)))

                if theta > last_theta:
                    theta -= n * 2.0 * pi
                else:
                    theta += n * 2.0 * pi

        values.append(
            GkSourceValue(
                None,
                z_source=z_source,
                G=numeric.G if numeric is not None else None,
                Gprime=numeric.Gprime if numeric is not None else None,
                theta=theta,
                H_ratio=WKB.H_ratio if WKB is not None else None,
                sin_coeff=WKB.sin_coeff if WKB is not None else None,
                cos_coeff=WKB.cos_coeff if WKB is not None else None,
                G_WKB=WKB.G_WKB if WKB is not None else None,
                omega_WKB_sq=(
                    numeric.omega_WKB_sq
                    if numeric is not None
                    else WKB.omega_WKB_sq if WKB is not None else None
                ),
                WKB_criterion=(
                    numeric.WKB_criterion
                    if numeric is not None
                    else WKB.WKB_criterion if WKB is not None else None
                ),
                analytic_G=(
                    numeric.analytic_G
                    if numeric is not None
                    else WKB.analytic_G if WKB is not None else None
                ),
                analytic_Gprime=(
                    numeric.analytic_Gprime
                    if numeric is not None
                    else WKB.analytic_Gprime if WKB is not None else None
                ),
            )
        )

        last_theta = theta

    return {"values": values}


class GkSource(DatastoreObject):
    def __init__(
        self,
        payload,
        model: BackgroundModel,
        k: wavenumber_exit_time,
        z_response: redshift,
        atol: tolerance,
        rtol: tolerance,
        z_sample: Optional[redshift_array] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        k_wavenumber: wavenumber = k.k
        check_units(k_wavenumber, model.cosmology)

        self._z_sample = z_sample

        self._model = model
        self._k_exit = k
        self._z_response = z_response

        self._label = label
        self._tags = tags if tags is not None else []

        self._atol = atol
        self._rtol = rtol

        self._compute_ref = None

        if payload is not None:
            DatastoreObject.__init__(self, payload["store_id"])
            self._values = payload["values"]

        else:
            DatastoreObject.__init__(self, None)
            self._values = None

        if self._z_sample is not None:
            z_response_float = float(z_response)
            # check that each source redshift is earlier than the specified response redshift
            for z in self._z_sample:
                z_float = float(z)

                if z_float < z_response_float:
                    raise ValueError(
                        f"GkSource: source redshift sample point z={z_float:.5g} exceeds response redshift z={z_response_float:.5g}"
                    )

    @property
    def k(self) -> wavenumber:
        return self._k_exit.k

    @property
    def model(self) -> BackgroundModel:
        return self._model

    @property
    def z_response(self) -> redshift:
        return self._z_response

    @property
    def z_sample(self) -> redshift_array:
        return self._z_sample

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    @property
    def values(self) -> List:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError("GkSource: values read but _do_not_populate is set")

        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self, payload, label: Optional[str] = None):
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError("GkSource: compute() called but _do_not_populate is set")

        if self._values is not None:
            raise RuntimeError("values have already been computed")

        if self._z_response is None or self._z_sample is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_response or z_sample is missing). It can only represent a query."
            )

        # replace label if specified
        if label is not None:
            self._label = label

        # currently we have nothing to do here
        self._compute_ref = marshal_values.remote(
            self._z_sample, payload["numeric"], payload["WKB"]
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "GkSource: store() called, but no compute() is in progress"
            )

        # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        # if not, return None
        if len(resolved) == 0:
            return None

        payload = ray.get(self._compute_ref)
        self._compute_ref = None

        self._values = payload["values"]


class GkSourceValue(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z_source: redshift,
        G: Optional[float] = None,
        Gprime: Optional[float] = None,
        theta: Optional[float] = None,
        H_ratio: Optional[float] = None,
        sin_coeff: Optional[float] = None,
        cos_coeff: Optional[float] = None,
        G_WKB: Optional[float] = None,
        omega_WKB_sq: Optional[float] = None,
        WKB_criterion: Optional[float] = None,
        analytic_G: Optional[float] = None,
        analytic_Gprime: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z_source = z_source

        has_numeric = any([G is not None, Gprime is not None])
        all_numeric = all([G is not None, Gprime is not None])

        has_WKB = any(
            [
                theta is not None,
                H_ratio is not None,
                sin_coeff is not None,
                cos_coeff is not None,
                G_WKB is not None,
            ]
        )
        all_WKB = all(
            [
                theta is not None,
                H_ratio is not None,
                sin_coeff is not None,
                cos_coeff is not None,
                G_WKB is not None,
            ]
        )

        if has_numeric and not all_numeric:
            raise ValueError(
                "GkSourceValue: only partial numeric data were supplied. Please supply all of G and Gprime."
            )

        if has_WKB and not all_WKB:
            raise ValueError(
                "GkSourceValue: only partial WKB data were supplied. Please supply all of theta, H_ratio, sin_coeff, cos_coeff and G_WKB."
            )

        self._has_numeric = has_numeric
        self._has_WKB = has_WKB

        self._numeric_data = NumericData(
            G=G,
            Gprime=Gprime,
        )

        self._WKB_data = WKBData(
            theta=theta,
            H_ratio=H_ratio,
            sin_coeff=sin_coeff,
            cos_coeff=cos_coeff,
            G_WKB=G_WKB,
        )

        self._omega_WKB_sq = omega_WKB_sq
        self._WKB_criterion = WKB_criterion

        self._analytic_G = analytic_G
        self._analytic_Gprime = analytic_Gprime

    @property
    def z_source(self) -> redshift:
        return self._z_source

    @property
    def has_numeric(self) -> bool:
        return self._has_numeric

    @property
    def numeric(self) -> NumericData:
        return self._numeric_data

    @property
    def has_WKB(self) -> bool:
        return self._has_WKB

    @property
    def WKB(self) -> WKBData:
        return self._WKB_data

    @property
    def omega_WKB_sq(self) -> Optional[float]:
        return self._omega_WKB_sq

    @property
    def WKB_criterion(self) -> Optional[float]:
        return self._WKB_criterion

    @property
    def analytic_G(self) -> Optional[float]:
        return self._analytic_G

    @property
    def analytic_Gprime(self) -> Optional[float]:
        return self._analytic_Gprime
