from collections import namedtuple
from typing import Optional, List

import ray
from math import fabs, pi, sqrt, cos, sin

from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.GkNumericalIntegration import GkNumericalValue
from ComputeTargets.GkWKBIntegration import GkWKBValue
from CosmologyConcepts import wavenumber_exit_time, redshift, wavenumber, redshift_array
from Datastore import DatastoreObject
from MetadataConcepts import store_tag, tolerance
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_FLOAT_PRECISION
from utilities import check_units

_NumericData = namedtuple("NumericData", ["G", "Gprime"])
_WKBData = namedtuple(
    "WKBData",
    [
        "theta_mod_2pi",
        "theta_div_2pi",
        "raw_theta_div_2pi",
        "H_ratio",
        "sin_coeff",
        "cos_coeff",
        "G_WKB",
        "new_G_WKB",
        "abs_G_WKB_err",
        "rel_G_WKB_err",
    ],
)

_two_pi = 2.0 * pi

DEFAULT_G_WKB_DIFF_REL_TOLERANCE = 1e-2
DEFAULT_G_WKB_DIFF_ABS_TOLERANCE = 1e-3


@ray.remote
def assemble_GkSource_values(
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
    had_theta_last_step: Optional[float] = None
    last_z_for_theta: Optional[redshift] = None
    rectified_theta_div_2pi: Optional[int] = None

    last_theta_mod_2pi: Optional[float] = None
    last_theta_div_2pi: Optional[float] = None

    # track whether we have seen a WKB data point
    # generally, we expect that the WKB region should be continuous; we use this as a validation step to test for
    # possible trouble in the numerical solution
    seen_WKB: bool = False

    # track whether we are in the "primary" contiguous WKB region (the one that extends to lowest redshift)
    in_primary_WKB_region: Optional[bool] = None
    primary_WKB_largest_z: Optional[redshift] = None

    # track the latest redshift for which we have numerical information
    numerical_smallest_z: Optional[redshift] = None

    for z_source in reversed(list(z_sample)):
        numeric: GkNumericalValue = numeric_data.get(z_source.store_id, None)
        WKB: GkWKBValue = WKB_data.get(z_source.store_id, None)

        # If there is neither a WKB or a numerical data point, then this is an error. We cannot proceed.
        if numeric is None and WKB is None:
            raise RuntimeError(
                f"assemble_GkSource_values: no data supplied at redshift z_source={z_source.z:.5g} for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
            )

        # If we have both data points, ensure that their analytic estimates match.
        if numeric is not None and WKB is not None:
            if fabs(numeric.analytic_G - WKB.analytic_G) > DEFAULT_ABS_TOLERANCE:
                raise RuntimeError(
                    f"assemble_GkSource_values: analytic G values unexpectedly differ between numeric and WKB data values by a large amount at z_source={z_source.z:.5g} for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )

            if (
                fabs(numeric.analytic_Gprime - WKB.analytic_Gprime)
                > DEFAULT_ABS_TOLERANCE
            ):
                raise RuntimeError(
                    f"assemble_GkSource_values: analytic G values unexpectedly differ between numeric and WKB data values by a large amount at z_source={z_source.z:.5g} for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )

        # If we have a WKB data point, check that the response redshift is in the expected region where z_response < z_e3 for this k-mode
        if WKB is not None and z_response.z >= k_exit.z_exit_subh_e3:
            print(
                f"!! WARNING (assemble_GkSource_values): WKB value detected for k={k_exit.k.k_inv_Mpc}/Mpc (store_id={k_exit.store_id}) for z_response less than 3 e-folds inside the horizon"
            )
            print(
                f"|  -- z_e3={k_exit.z_exit_subh_e3:.5g}, z_source={z_source.z:.5g} (store_id={z_source.store_id})), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
            )

        theta_div_2pi: Optional[int] = WKB.theta_div_2pi if WKB is not None else None
        theta_mod_2pi: Optional[float] = WKB.theta_mod_2pi if WKB is not None else None
        abs_err: Optional[float] = None
        rel_err: Optional[float] = None
        new_G_WKB: Optional[float] = None

        # We need to adjust theta to produce a smooth, monotonic function of redshift.
        # The Levin quadrature method relies on representing the oscillations of G_k(z, z') using a smooth function.
        # With our current way of calculating theta, we expect it to be an increasing function of redshift.
        # It is increasingly *negative* at large z.

        # The main idea is to keep track of the current value of theta div 2pi and theta mod 2pi.
        # If the next value of theta mod 2pi is less negative than the current one (bearing in mind we take
        # theta mod 2pi to be negative) then there is a possibility that we have moved into the next block.
        # Mathematically this is what should happen, because theta_WKB should be a monotone increasing
        # function. But in practice, if the current theta is quite close to the last one, then doing so would
        # entail nearly a 2pi discontinuity. In this case, it is more likely that we are seeing a small "jitter".
        # We need to set a tolerance to determine where we think there is more likely jitter, and where there is
        # a genuine progression.

        if theta_mod_2pi is not None:
            # warn if the region in which we have WKB data is (seemingly) not contiguous
            # note we do not need to be too concerned if this happens in the region between 3 and 5 e-folds inside the horizon where the analytic and
            # numerical solutions overlap.
            # In this region, the place where we cut the numerical solution to provide initial data for the WKB calculation can vary depending on
            # z_source. So for some z_source we will have data at a given z_response, and for others we won't. This can cause the WKB region
            # to appear non-contiguous.
            if (
                z_response.z
                < 0.65
                * k_exit.z_exit_subh_e3  # not sure quite what the multiplier should be here, but the cut should not be too much later than z_e3
                and had_theta_last_step is not True
                and seen_WKB
            ):
                print(
                    f"!! WARNING (assemble_GkSource_values): WKB region apparently non-contiguous at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )
                if last_z_for_theta is not None:
                    print(
                        f"|  -- last theta value seen at redshift z_source={last_z_for_theta.z:.5g} (store_id={last_z_for_theta.store_id}) | z_e3 = {k_exit.z_exit_subh_e3:.5g}"
                    )
                print(
                    f"|  -- last_theta_mod_2pi={last_theta_mod_2pi:.5g}, rectified_theta_div_2pi={rectified_theta_div_2pi}"
                )

            if last_theta_mod_2pi is None:
                # presumably this is the first time we have seen a theta value. Start in the fundamental block (-2pi, 0]

                current_2pi_block_subtraction = theta_div_2pi
                rectified_theta_div_2pi = 0

                in_primary_WKB_region = True

            else:
                if rectified_theta_div_2pi is None:
                    raise RuntimeError(
                        f"assemble_GkSource_values: rectified_theta_div_2pi should not be None at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                    )

                # if the 2pi block read from the data is more positive than our current one, we interpret this as a discontinuity.
                # we adjust the current subtraction to bring 2pi block into alignment with our current one.

                # note: in principle, discontinuities could occur in either sense, either with theta_div_2pi jumping suddenly up or down.
                # But in practice, because of the way we calculate theta, the most significant discontinuities seem to occur with theta_div_2pi
                # jumping *up*, which helpfully makes them easier to spot
                if (
                    theta_div_2pi > last_theta_div_2pi
                    and theta_mod_2pi > last_theta_mod_2pi
                ):
                    # now we have to decide whether this is likely just a jitter or numerical fluctuation,
                    # or whether it is more likely that the solution really descends very rapidly.

                    # We pick the new solution point to be the one closest to our current point.
                    if theta_mod_2pi - last_theta_mod_2pi > pi:
                        # closer if we jump to the next block
                        rectified_theta_div_2pi -= 1

                    current_2pi_block_subtraction = (
                        theta_div_2pi - rectified_theta_div_2pi
                    )

                # otherwise, we assume the current subtraction remains valid, and instead update our current 2pi block to match
                else:
                    rectified_theta_div_2pi = (
                        theta_div_2pi - current_2pi_block_subtraction
                    )

            # remember our current values of theta div 2pi and theta mod 2pi for the next step in the algorithm
            last_theta_mod_2pi = theta_mod_2pi
            last_theta_div_2pi = theta_div_2pi

            seen_WKB = True
            had_theta_last_step = True
            last_z_for_theta = z_source

            if in_primary_WKB_region and (
                primary_WKB_largest_z is None or z_source.z > primary_WKB_largest_z.z
            ):
                primary_WKB_largest_z = z_source

            # check that we are able to build the WKB solution from the data available
            if WKB.omega_WKB_sq is None or WKB.omega_WKB_sq < 0.0:
                raise RuntimeError(
                    f"assemble_GkSource_values: cannot process WKB phase because omega_WKB_sq is negative or missing at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )

            if WKB.H_ratio is None or WKB.H_ratio < 0.0:
                raise RuntimeError(
                    f"assemble_GkSource_values: cannot process WKB phase because H_ratio is negative or missing at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )

            if WKB.sin_coeff is None or WKB.cos_coeff is None:
                raise RuntimeError(
                    f"assemble_GkSource_values: cannot process WKB phase because at least of the sin/cos coefficients are missing at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )

            # defensively, test whether this has significantly changed the result
            # it should not, but ...
            omega_WKB = sqrt(WKB.omega_WKB_sq)
            norm_factor = sqrt(WKB.H_ratio / omega_WKB)

            new_G_WKB = norm_factor * (
                WKB.cos_coeff * cos(theta_mod_2pi) + WKB.sin_coeff * sin(theta_div_2pi)
            )

            # if the fractional error is too large, treat as an exception, so that we cannot silently
            # write meaningless results into the datastore
            if fabs(WKB.G_WKB) < DEFAULT_ABS_TOLERANCE:
                abs_err = abs(new_G_WKB - WKB.G_WKB)
                # test abs difference
                if abs_err > DEFAULT_G_WKB_DIFF_ABS_TOLERANCE:
                    print(
                        f"!! WARNING (assemble_GkSource_values): rectified G_WKB differs from original G_WKB by an unexpectedly large amount at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id}) | old G_WKB={WKB.G_WKB:.7g}, new G_WKB={new_G_WKB:.7g}, abserr={abs_err:.3g}"
                    )
            else:
                rel_err = fabs((new_G_WKB - WKB.G_WKB) / WKB.G_WKB)
                # test rel difference
                if rel_err > DEFAULT_G_WKB_DIFF_REL_TOLERANCE:
                    print(
                        f"!! WARNING (assemble_GkSource_values): rectified G_WKB differs from original G_WKB by an unexpectedly large amount at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id}) | old G_WKB={WKB.G_WKB:.7g}, new G_WKB={new_G_WKB:.7g}, relerr={rel_err:.3g}"
                    )

        else:
            # leave last_theta_mod_2pi and rectified_theta_div_2pi alone.
            # We can re-use their values later if needed.
            had_theta_last_step = False

            if in_primary_WKB_region:
                in_primary_WKB_region = False

        if numeric is not None:
            if numerical_smallest_z is None or z_source.z < numerical_smallest_z.z:
                numerical_smallest_z = z_source

        values.append(
            GkSourceValue(
                None,
                z_source=z_source,
                G=numeric.G if numeric is not None else None,
                Gprime=numeric.Gprime if numeric is not None else None,
                theta_mod_2pi=theta_mod_2pi,
                theta_div_2pi=rectified_theta_div_2pi,
                raw_theta_div_2pi=theta_div_2pi,
                H_ratio=WKB.H_ratio if WKB is not None else None,
                sin_coeff=WKB.sin_coeff if WKB is not None else None,
                cos_coeff=WKB.cos_coeff if WKB is not None else None,
                G_WKB=WKB.G_WKB if WKB is not None else None,
                new_G_WKB=new_G_WKB,
                abs_G_WKB_err=abs_err,
                rel_G_WKB_err=rel_err,
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

    if numerical_smallest_z is None:
        if primary_WKB_largest_z is None:
            print(
                f"!! WARNING (assemble_GkSource_values): for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
            )
            print(f"|    -- no numerical region or primary WKB region was detected")
        elif primary_WKB_largest_z.z < z_sample.max.z - DEFAULT_FLOAT_PRECISION:
            print(
                f"!! WARNING (assemble_GkSource_values): for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
            )
            print(
                f"|    -- no numerical values for G_k(z,z') were detected, but largest source redshift in primary WKB region appears to be z_max={primary_WKB_largest_z.z:.5g} (store_id={primary_WKB_largest_z.store_id})"
            )

    else:
        if primary_WKB_largest_z is None:
            if numerical_smallest_z.z > z_sample.min.z + DEFAULT_FLOAT_PRECISION:
                print(
                    f"!! WARNING (assemble_GkSource_values): for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )
                print(
                    f"|    -- no WKB values for G_k(z,z') were detected, but smallest numerical source redshift appears to be z_max={numerical_smallest_z.z:.5g} (store_id={numerical_smallest_z.store_id})"
                )
        else:
            if (
                numerical_smallest_z.z
                > primary_WKB_largest_z.z + DEFAULT_FLOAT_PRECISION
            ):
                print(
                    f"!! WARNING (assemble_GkSource_values): for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                )
                print(
                    f"|    smallest detected numerical source redshift z_min={numerical_smallest_z.z:.5g} (store_id={numerical_smallest_z.store_id}) is larger than largest detected source redshift in primary WKB region z_max={primary_WKB_largest_z.z:.5g} (store_id={primary_WKB_largest_z.store_id})"
                )

    return {
        "values": values,
        "numerical_smallest_z": numerical_smallest_z,
        "primary_WKB_largest_z": primary_WKB_largest_z,
    }


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

            self._numerical_smallest_z = payload["numerical_smallest_z"]
            self._primary_WKB_largest_z = payload["primary_WKB_largest_z"]

        else:
            DatastoreObject.__init__(self, None)
            self._values = None

            self._numerical_smallest_z = None
            self._primary_WKB_largest_z = None

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
    def numerical_smallest_z(self) -> Optional[redshift]:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "GkSource: numerical_smallest_z read but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError("values have not yet been populated")

        return self._numerical_smallest_z

    @property
    def primary_WKB_largest_z(self) -> Optional[redshift]:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "GkSource: primary_WKB_largest_z read but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError("values have not yet been populated")

        return self._primary_WKB_largest_z

    @property
    def values(self) -> List:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError("GkSource: values read but _do_not_populate is set")

        if self._values is None:
            raise RuntimeError("values have not yet been populated")
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
        self._compute_ref = assemble_GkSource_values.remote(
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
        self._numerical_smallest_z = payload["numerical_smallest_z"]
        self._primary_WKB_largest_z = payload["primary_WKB_largest_z"]


class GkSourceValue(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z_source: redshift,
        G: Optional[float] = None,
        Gprime: Optional[float] = None,
        theta_mod_2pi: Optional[float] = None,
        theta_div_2pi: Optional[int] = None,
        raw_theta_div_2pi: Optional[int] = None,
        H_ratio: Optional[float] = None,
        sin_coeff: Optional[float] = None,
        cos_coeff: Optional[float] = None,
        G_WKB: Optional[float] = None,
        new_G_WKB: Optional[float] = None,
        abs_G_WKB_err: Optional[float] = None,
        rel_G_WKB_err: Optional[float] = None,
        omega_WKB_sq: Optional[float] = None,
        WKB_criterion: Optional[float] = None,
        analytic_G: Optional[float] = None,
        analytic_Gprime: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z_source = z_source

        numeric_flags = [G is not None, Gprime is not None]
        has_numeric = any(numeric_flags)
        all_numeric = all(numeric_flags)

        WKB_flags = [
            theta_mod_2pi is not None,
            theta_div_2pi is not None,
            H_ratio is not None,
            sin_coeff is not None,
            cos_coeff is not None,
            G_WKB is not None,
        ]
        has_WKB = any(WKB_flags)
        all_WKB = all(WKB_flags)

        if has_numeric and not all_numeric:
            raise ValueError(
                "GkSourceValue: only partial numeric data were supplied. Please supply all of G and Gprime."
            )

        if has_WKB and not all_WKB:
            raise ValueError(
                "GkSourceValue: only partial WKB data were supplied. Please supply all of theta_mod_2pi, theta_div_2pi, H_ratio, sin_coeff, cos_coeff and G_WKB."
            )

        self._has_numeric = has_numeric
        self._has_WKB = has_WKB

        self._numeric_data = _NumericData(
            G=G,
            Gprime=Gprime,
        )

        self._WKB_data = _WKBData(
            theta_mod_2pi=theta_mod_2pi,
            theta_div_2pi=theta_div_2pi,
            raw_theta_div_2pi=raw_theta_div_2pi,
            H_ratio=H_ratio,
            sin_coeff=sin_coeff,
            cos_coeff=cos_coeff,
            G_WKB=G_WKB,
            new_G_WKB=new_G_WKB,
            abs_G_WKB_err=abs_G_WKB_err,
            rel_G_WKB_err=rel_G_WKB_err,
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
