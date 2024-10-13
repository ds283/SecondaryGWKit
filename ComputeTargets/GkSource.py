from collections import namedtuple
from typing import Optional, List

import ray
from math import fabs, pi, fmod, sqrt, cos, sin

from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.GkNumericalIntegration import GkNumericalValue
from ComputeTargets.GkWKBIntegration import GkWKBValue
from CosmologyConcepts import wavenumber_exit_time, redshift, wavenumber, redshift_array
from Datastore import DatastoreObject
from MetadataConcepts import store_tag, tolerance
from defaults import DEFAULT_ABS_TOLERANCE
from utilities import check_units

_NumericData = namedtuple("NumericData", ["G", "Gprime"])
_WKBData = namedtuple(
    "WKBData", ["theta", "raw_theta", "H_ratio", "sin_coeff", "cos_coeff", "G_WKB"]
)

_two_pi = 2.0 * pi

DEFAULT_G_WKB_DIFF_TOLERANCE = 1e-5
DEFAULT_PHASE_JITTER_TOLERANCE = 1e-2


@ray.remote
def marshal_values(
    k_exit: wavenumber_exit_time,
    z_response: redshift,
    z_sample: redshift_array,
    numeric_data,
    WKB_data,
):
    values = []

    # work through the z_sample array, and check whether there is any numeric or WKB data for each sample point.
    # We do this in reverse order (from low to high redshift). This is done so that we can try to smooth the phase function.
    # It will have step-like discontinuities due to the way we calculate it.
    theta_last_step = None
    theta_last_z = None
    current_2pi_block = None

    last_theta_mod_2pi = None

    # track whether we have seen a WKB data point
    # generally, we expect that the WKB region should be continuous; we use this as a validation step to test for
    # possible trouble in the numerical solution
    seen_WKB = False

    for z_source in reversed(list(z_sample)):
        numeric: GkNumericalValue = numeric_data.get(z_source.store_id, None)
        WKB: GkWKBValue = WKB_data.get(z_source.store_id, None)

        if numeric is None and WKB is None:
            raise RuntimeError(
                f"marshal_values: no data supplied at redshift z_source={z_source.z:.5g} for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
            )

        if numeric is not None and WKB is not None:
            if fabs(numeric.analytic_G - WKB.analytic_G) > DEFAULT_ABS_TOLERANCE:
                raise RuntimeError(
                    f"marshal_values: analytic G values unexpectedly differ between numeric and WKB data values by a large amount at z_source={z_source.z:.5g} for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )

            if (
                fabs(numeric.analytic_Gprime - WKB.analytic_Gprime)
                > DEFAULT_ABS_TOLERANCE
            ):
                raise RuntimeError(
                    f"marshal_values: analytic G values unexpectedly differ between numeric and WKB data values by a large amount at z_source={z_source.z:.5g} for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )

        if WKB is not None and z_response.z >= k_exit.z_exit_subh_e3:
            print(
                f"!! WARNING (marshal_values): WKB value detected for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}) for z_response less than 3 e-folds inside the horizon"
            )
            print(
                f"|  -- z_e3={k_exit.z_exit_subh_e3:.5g}, z_source={z_source.z:.5g} (store_id={z_source.store_id})), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
            )

        raw_theta = WKB.theta if WKB is not None else None

        # Adjust theta to produce a smooth, monotonic function of redshift
        # With our current way of calculating theta, we expect it to be an increasing function of redshift.
        # It is increasingly *negative* at large z.

        # The main idea is to keep track of the curent value of theta div 2pi and theta mod 2pi.
        # If the next value of theta mod 2pi is larger than the current one (bearing in mind we take
        # theta mod 2pi to be negative) then we must have moved into the next block.
        # This way we only ever deal with the value of theta mod 2pi from the computed solution.
        # We lay out the values of theta div 2pi in such a way that the phase is smooth and monotone.
        if raw_theta is not None:
            theta_mod_2pi = fmod(raw_theta, _two_pi)
            if theta_mod_2pi > 0:
                theta_mod_2pi = theta_mod_2pi - _two_pi

            assert theta_mod_2pi <= 0.0
            assert theta_mod_2pi > -_two_pi

            # warn if the region in which we have WKB data is (seemingly) not coniguous
            # note we do not need to be too concerned if this happens in the region between 3 and 5 e-folds inside the horizon where the analytic and
            # numerical solutions overlap.
            # In this region, the place where we cut the numerical solution to provide initial data for the WKB calculation can vary depending on
            # z_source. So for some z_source we will have data at a given z_response, and for others we won't. This can cause the WKB region
            # to appear non-contiguous.
            if (
                z_response.z < k_exit.z_exit_subh_e3
                and theta_last_step is not True
                and seen_WKB
            ):
                print(
                    f"!! WARNING (marshal_values): WKB region apparently non-contiguous at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )
                if theta_last_z is not None:
                    print(
                        f"|  -- last theta value seen at redshift z_source={theta_last_z.z:.5g} (store_id={theta_last_z.store_id})"
                    )
                print(
                    f"|  -- last_theta_mod_2pi={last_theta_mod_2pi:.5g}, current_2pi_block={current_2pi_block}"
                )

            if last_theta_mod_2pi is None:
                # presumably this is the first time we have seen a theta value. Start in the fundamental block (-2pi, 0]
                theta = theta_mod_2pi
                current_2pi_block = 0

            else:
                # decide whether this phase is more negative than the last one. If so, we can stay within the same 2pi-block.
                # Otherwise, we have to move one block to the left.

                if current_2pi_block is None:
                    raise RuntimeError(
                        f"marshal_values: current_2pi_block should not be None at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                    )

                # allow a small tolerance for jitter in the phase function, without stepping into a new block
                # otherwise, we may generate a misleadingly large d(theta)/dz that will produce a misleading Levin quadrature
                if theta_mod_2pi > last_theta_mod_2pi + DEFAULT_PHASE_JITTER_TOLERANCE:
                    current_2pi_block -= 1

                theta = current_2pi_block * _two_pi + theta_mod_2pi

            last_theta_mod_2pi = theta_mod_2pi
            seen_WKB = True
            theta_last_step = True
            theta_last_z = z_source

            if WKB.omega_WKB_sq is None or WKB.omega_WKB_sq < 0.0:
                raise RuntimeError(
                    f"marshal_values: cannot process WKB phase because omega_WKB_sq is negative or missing at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )

            if WKB.H_ratio is None or WKB.H_ratio < 0.0:
                raise RuntimeError(
                    f"marshal_values: cannot process WKB phase because H_ratio is negative or missing at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )

            if WKB.sin_coeff is None or WKB.cos_coeff is None:
                raise RuntimeError(
                    f"marshal_values: cannot process WKB phase because one of the sin or cos coefficients are missing at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )

            # defensively, test whether this has significantly changed the result
            # it should not, but ...
            omega_WKB = sqrt(WKB.omega_WKB_sq)
            norm_factor = sqrt(WKB.H_ratio / omega_WKB)

            new_G_WKB = norm_factor * (
                WKB.cos_coeff * cos(theta) + WKB.sin_coeff * sin(theta)
            )

            # if the fractional error is too large, treat as an exception, so that we cannot silently
            # write meaningless results into the datastore
            if fabs(WKB.G_WKB) < DEFAULT_ABS_TOLERANCE:
                # test abs difference
                if fabs(new_G_WKB - WKB.G_WKB) > DEFAULT_G_WKB_DIFF_TOLERANCE:
                    raise RuntimeError(
                        f"marshal_values: rectified G_WKB differs from original G_WKB by an unexpectedly large amount at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id}) | old G_WKB={WKB.G_WKB:.7g}, new G_WKB={new_G_WKB:.7g}"
                    )
            else:
                # test rel difference
                if (
                    fabs((new_G_WKB - WKB.G_WKB) / WKB.G_WKB)
                ) > DEFAULT_G_WKB_DIFF_TOLERANCE:
                    raise RuntimeError(
                        f"marshal_values: rectified G_WKB differs from original G_WKB by an unexpectedly large amount at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id}) | old G_WKB={WKB.G_WKB:.7g}, new G_WKB={new_G_WKB:.7g}"
                    )

        else:
            # leave last_theta_mod_2pi and current_2pi_block alone.
            # We can re-use their values later if needed.
            theta_last_step = False
            theta = None

        values.append(
            GkSourceValue(
                None,
                z_source=z_source,
                G=numeric.G if numeric is not None else None,
                Gprime=numeric.Gprime if numeric is not None else None,
                theta=theta,
                raw_theta=raw_theta,
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
            self._k_exit,
            self._z_response,
            self._z_sample,
            payload["numeric"],
            payload["WKB"],
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
        raw_theta: Optional[float] = None,
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

        self._numeric_data = _NumericData(
            G=G,
            Gprime=Gprime,
        )

        self._WKB_data = _WKBData(
            theta=theta,
            raw_theta=raw_theta,
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
    def numeric(self) -> _NumericData:
        return self._numeric_data

    @property
    def has_WKB(self) -> bool:
        return self._has_WKB

    @property
    def WKB(self) -> _WKBData:
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
