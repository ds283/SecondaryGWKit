from collections import namedtuple
from typing import Optional, List

import pandas as pd
import ray
from math import fabs, pi, sqrt, cos, sin, log
from scipy.interpolate import InterpolatedUnivariateSpline

from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.GkNumericalIntegration import GkNumericalValue
from ComputeTargets.GkWKBIntegration import GkWKBValue
from ComputeTargets.spline_wrappers import GkWKBSplineWrapper, ZSplineWrapper
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
        "theta",
        "raw_theta_div_2pi",
        "raw_theta",
        "H_ratio",
        "sin_coeff",
        "cos_coeff",
        "z_init",  # if not None, shows that this WKB data point was obtained from a GkNumericalIntegration initial condition
        "G_WKB",
        "new_G_WKB",
        "abs_G_WKB_err",
        "rel_G_WKB_err",
    ],
)

GkSourceFunctions = namedtuple(
    "GkSourceFunctions",
    [
        "numerical_region",
        "WKB_region",
        "numerical_Gk",
        "WKB_Gk",
        "theta",
        "theta_deriv",
        "type",
        "quality",
        "crossover_z",
    ],
)

_two_pi = 2.0 * pi

DEFAULT_G_WKB_DIFF_REL_TOLERANCE = 1e-2
DEFAULT_G_WKB_DIFF_ABS_TOLERANCE = 1e-3

MIN_SPLINE_DATA_POINTS = 5

# value of dtheta/dz where we will switch to a Levin integration strategy
LEVIN_THRESHOLD = 5.0


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
    current_2pi_block_subtraction: Optional[int] = None
    current_2pi_block: Optional[int] = None

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

    lowest_z = z_sample.min

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
        rectified_theta_div_2pi: Optional[int] = None
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
                z_response.z < k_exit.z_exit_subh_e4
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
                # this is the first time we have seen a theta value. Start in the fundamental block (-2pi, 0].

                current_2pi_block_subtraction = theta_div_2pi
                current_2pi_block = 0
                rectified_theta_div_2pi = 0

                # if this is also the lowest source redshift, we can begin counting up the primary WKB region.
                # Otherwise, if the lowest source redshift has no WKB data, there is no WKB primary region.
                # Instead, we have to rely on the numerical region going all the way down the z_min.
                # (There will probably always be *some* k-modes/z_source combinations for which WKB data doee not
                # go all the way down to the lowest z_source. WKB data are missing only for z_response values that lie
                # between z_source and z_init, the time of the initial condition computed from a GkNumericalIntegration instance.
                # Since there will typically be *some* response redshifts in this missing window, there wll also be some
                # GkSource instances that also miss it.
                if z_source.store_id == lowest_z.store_id:
                    in_primary_WKB_region = True

            else:
                if current_2pi_block_subtraction is None:
                    raise RuntimeError(
                        f"assemble_GkSource_values: current_2pi_block_subtraction should not be None at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                    )
                if current_2pi_block is None:
                    raise RuntimeError(
                        f"assemble_GkSource_values: current_2pi_block should not be None at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id})"
                    )

                # if the 2pi block read from the data is more positive than our current one, we interpret this as a discontinuity.
                # we adjust the current subtraction to bring 2pi block into alignment with our current one.

                # note: in principle, discontinuities could occur in either sense, either with theta_div_2pi jumping suddenly up or down.
                # But in practice, because of the way we calculate theta, the most significant discontinuities seem to occur with theta_div_2pi
                # jumping *up*, which helpfully makes them easier to spot
                last_theta = _two_pi * last_theta_div_2pi + last_theta_mod_2pi
                theta = _two_pi * theta_div_2pi + theta_mod_2pi
                if theta > last_theta:
                    # we interpret this condition as a possible discontinuity
                    # attempt to align the phase blocks to produce as smooth a curve as possible
                    delta_theta_mod_2pi = theta_mod_2pi - last_theta_mod_2pi

                    # do we produce a closer result by staying in the current 2pi block, or by going up one, or by going down one?
                    delta_theta_mod_2pi_0 = fabs(delta_theta_mod_2pi)
                    delta_theta_mod_2pi_p1 = fabs(delta_theta_mod_2pi + _two_pi)
                    delta_theta_mod_2pi_m1 = fabs(delta_theta_mod_2pi - _two_pi)

                    jump_options = [
                        (delta_theta_mod_2pi_0, 0),
                        (delta_theta_mod_2pi_p1, +1),
                        (delta_theta_mod_2pi_m1, -1),
                    ]
                    best_option = min(jump_options, key=lambda x: x[0])
                    best_jump = best_option[1]

                    rectified_theta_div_2pi = current_2pi_block + best_jump

                    # adjust current subtraction to match value our chosen value of theta_div_2pi
                    current_2pi_block_subtraction = (
                        theta_div_2pi - rectified_theta_div_2pi
                    )

                else:
                    # baseline is that the rectified value of theta div 2pi is the current value minus the current subtraction:
                    rectified_theta_div_2pi = (
                        theta_div_2pi - current_2pi_block_subtraction
                    )

                current_2pi_block = rectified_theta_div_2pi

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
            # it should not, but this way we should catch any inadvertent errors
            omega_WKB = sqrt(WKB.omega_WKB_sq)
            norm_factor = sqrt(WKB.H_ratio / omega_WKB)

            new_G_WKB = norm_factor * (
                WKB.cos_coeff * cos(theta_mod_2pi) + WKB.sin_coeff * sin(theta_mod_2pi)
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
                    print(
                        f"|    -- WKB.theta_mod_2pi={WKB._theta_mod_2pi:.5g}, WKB.theta_div_2pi={WKB.theta_div_2pi}"
                    )
                    print(
                        f"|    -- WKB.H_ratio={WKB.H_ratio:.5g}, WKB.omega_WKB_sq={WKB.omega_WKB_sq:.5g}"
                    )
                    print(
                        f"|    -- WKB.sin_coeff={WKB.sin_coeff:.5g}, WKB.cos_coeff={WKB.cos_coeff:.5g}"
                    )
                    print(f"|    -- norm_factor={norm_factor:.5g}")
            else:
                rel_err = fabs((new_G_WKB - WKB.G_WKB) / WKB.G_WKB)
                # test rel difference
                if rel_err > DEFAULT_G_WKB_DIFF_REL_TOLERANCE:
                    print(
                        f"!! WARNING (assemble_GkSource_values): rectified G_WKB differs from original G_WKB by an unexpectedly large amount at z_source={z_source.z:.5g} (store_id={z_source.store_id}) for k={k_exit.k.k_inv_Mpc:.5g}/Mpc (store_id={k_exit.store_id}), z_response={z_response.z:.5g} (store_id={z_response.store_id}) | old G_WKB={WKB.G_WKB:.7g}, new G_WKB={new_G_WKB:.7g}, relerr={rel_err:.3g}"
                    )
                    print(
                        f"|    -- WKB.theta_mod_2pi={WKB._theta_mod_2pi:.5g}, WKB.theta_div_2pi={WKB.theta_div_2pi}"
                    )
                    print(
                        f"|    -- WKB.H_ratio={WKB.H_ratio:.5g}, WKB.omega_WKB_sq={WKB.omega_WKB_sq:.5g}"
                    )
                    print(
                        f"|    -- WKB.sin_coeff={WKB.sin_coeff:.5g}, WKB.cos_coeff={WKB.cos_coeff:.5g}"
                    )
                    print(f"|    -- norm_factor={norm_factor:.5g}")

        else:
            # leave last_theta_mod_2pi and current_2pi_block_subtraction alone.
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
                z_init=WKB.z_init if WKB is not None else None,
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
                    f"|    -- smallest detected numerical source redshift z_min={numerical_smallest_z.z:.5g} (store_id={numerical_smallest_z.store_id}) is larger than largest detected source redshift in primary WKB region z_max={primary_WKB_largest_z.z:.5g} (store_id={primary_WKB_largest_z.store_id})"
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

        if payload is not None:
            DatastoreObject.__init__(self, payload["store_id"])
            self._values = payload["values"]

            self._numerical_smallest_z = payload["numerical_smallest_z"]
            self._primary_WKB_largest_z = payload["primary_WKB_largest_z"]

            self._type = payload["type"]
            self._quality = payload["quality"]
            self._crossover_z = payload["crossover_z"]
            self._Levin_z = payload["Levin_z"]

            self._metadata = payload["metadata"]

        else:
            DatastoreObject.__init__(self, None)
            self._values = None

            self._numerical_smallest_z = None
            self._primary_WKB_largest_z = None

            self._type = None
            self._quality = None
            self._crossover_z = None
            self._Levin_z = None

            self._metadata = {}

        self._values_df = None
        self._functions = None
        self._compute_ref = None

        if self._z_sample is not None:
            z_response_float = float(z_response)
            # check that each source redshift is earlier than the specified response redshift
            for z in self._z_sample:
                z_float = float(z)

                if z_float < z_response_float:
                    raise ValueError(
                        f"GkSource: source redshift sample point z={z_float:.5g} exceeds response redshift z={z_response_float:.5g}"
                    )

    def _classify(self):
        self._classify_crossover()
        self._classify_Levin()

    def _classify_Levin(self):
        # find the Levin point, if we are using WKB data
        if (
            self._type == "WKB" or self._type == "mixed"
        ) and self._quality != "incomplete":
            self._Levin_z = None
            max_z = (
                self._crossover_z
                if self._crossover_z is not None
                else self._primary_WKB_largest_z.z
            )
            min_z = self._z_sample.min.z
            WKB_data = self._build_WKB_values(max_z=max_z, min_z=min_z)
            theta_data = [
                (
                    log(v.z_source.z),
                    v.WKB.theta,
                )
                for v in WKB_data
            ]
            theta_x, theta_y = zip(*theta_data)
            theta_spline = InterpolatedUnivariateSpline(theta_x, theta_y, ext="raise")
            theta_deriv = ZSplineWrapper(
                theta_spline.derivative(),
                "theta derivative",
                max_z,
                min_z,
                log_z=True,
                deriv=True,
            )

            # _z_sample is guaranteed to be in descending order of redshift
            for z_source in self._z_sample:
                if z_source.z <= max_z and z_source.z >= min_z:
                    if fabs(theta_deriv(z_source.z)) > LEVIN_THRESHOLD:
                        self._Levin_z = z_source
                        self._metadata["Levin_z_dtheta_dz"] = theta_deriv(z_source.z)
                        break

    def _classify_crossover(self):
        has_WKB_spline_points = False
        if self._primary_WKB_largest_z is not None:
            WKB_data = self._build_WKB_values(
                max_z=self._primary_WKB_largest_z, min_z=self._z_sample.min
            )

            if len(WKB_data) >= MIN_SPLINE_DATA_POINTS:
                has_WKB_spline_points = True
            else:
                self._metadata["WKB_spline_points"] = len(WKB_data)

        # if primary WKB region is missing, or there are not enough points to construct an adequate spline,
        # we need numerical data to go all the way down to lowest redshift
        if self._primary_WKB_largest_z is None or not has_WKB_spline_points:
            self._type = "numeric"
            self._crossover_z = None

            if self._numerical_smallest_z.store_id == self._z_sample.min.store_id:
                self._quality = "complete"
            else:
                self._quality = "incomplete"
                self._metadata["comment"] = (
                    "No WKB region, and numeric region does not extend to z_sample.min"
                )

        # if numerical data is missing, we need WKB data to go all the way up to the highest redshift
        elif self._numerical_smallest_z is None:
            self._type = "WKB"
            self._crossover_z = None

            if (
                self._primary_WKB_largest_z.store_id == self._z_sample.max.store_id
                and has_WKB_spline_points
            ):
                self._quality = "complete"
            else:
                self._quality = "incomplete"
                self._metadata["comment"] = (
                    "No numeric region, and WKB region does not extend to z_sample.max"
                )

        # otherwise, there should be an overlap region
        else:
            # we're allowed to assume that the primary WKB region begins at z_sample.min, and the numerical region
            # (if it is present) begins at z_sample.max

            # we're also allowed to assume that we have at least a minimum number of sample points to produce a spline
            # between _primary_WKB_largest_z and z_min
            self._type = "mixed"

            if (
                self._numerical_smallest_z.z
                >= self._primary_WKB_largest_z.z - DEFAULT_FLOAT_PRECISION
            ):
                self._quality = "incomplete"
                self._crossover_z = self._numerical_smallest_z.z
                self._metadata["comment"] = (
                    "Both WKB and numeric regions are present, but numeric region does not extend below largest redshift in primary WKB region"
                )

            else:
                # can assume _numerical_smallest_z is strictly smaller than _primary_WKB_largest_z

                # find minimum of search window
                # we want this not to be too close the lower bound at _numerical_smallest_z, but it must fall below the upper bound at
                # _primary_WKB_largest_z
                crossover_trial_min_z_options = [
                    1.06 * self._numerical_smallest_z.z,
                    1.03 * self._numerical_smallest_z.z,
                    1.015 * self._numerical_smallest_z.z,
                    self._numerical_smallest_z.z,
                ]
                crossover_trial_min_z_options = [
                    z
                    for z in crossover_trial_min_z_options
                    if z <= self._primary_WKB_largest_z.z + DEFAULT_FLOAT_PRECISION
                ]
                crossover_trial_min_z = max(crossover_trial_min_z_options)

                # find maximum of search window
                # we want this not to be too close to the upper bound at _primary_WKB_largest_z, but it must fall above the lower bound at
                # _numerical_smallest_z
                crossover_trial_max_z_options = [
                    0.94 * self._primary_WKB_largest_z.z,
                    0.97 * self._primary_WKB_largest_z.z,
                    0.985 * self._primary_WKB_largest_z.z,
                    self._primary_WKB_largest_z.z,
                ]
                crossover_trial_max_z_options = [
                    z
                    for z in crossover_trial_max_z_options
                    if z >= self._numerical_smallest_z.z - DEFAULT_FLOAT_PRECISION
                ]
                crossover_trial_max_z = min(crossover_trial_max_z_options)

                # work through the search window, trying to find a crossover point at which we have enough data points to build a sensible spline
                # after a limited number of iterations we give up
                self._crossover_z = None
                step_width = (crossover_trial_max_z - crossover_trial_min_z) / 10.0

                iterations = 0
                trial_z = crossover_trial_max_z
                while (
                    iterations < 10
                    and self._crossover_z is None
                    and trial_z >= crossover_trial_min_z
                ):
                    WKB_data = self._build_WKB_values(
                        max_z=trial_z, min_z=self._z_sample.min
                    )

                    # are there enough data points to get a sensible spline?
                    if len(WKB_data) >= MIN_SPLINE_DATA_POINTS:
                        self._crossover_z = trial_z

                        WKB_clearance = trial_z / self._primary_WKB_largest_z.z
                        numerical_clearance = trial_z / self._numerical_smallest_z.z

                        # try to classify the quality of this crossover choice
                        if WKB_clearance < 0.95 and numerical_clearance > 1.05:
                            self._quality = "complete"
                        elif WKB_clearance < 0.985 and numerical_clearance > 1.025:
                            self._quality = "acceptable"
                        elif WKB_clearance < 0.99 and numerical_clearance > 1.01:
                            self._quality = "marginal"
                        else:
                            self._quality = "minimal"

                        self._metadata["WKB_clearance"] = WKB_clearance
                        self._metadata["numerical_clearance"] = numerical_clearance
                        self._metadata["WKB_crossover_spline_points"] = len(WKB_data)
                        break

                    iterations = iterations + 1
                    trial_z = trial_z - step_width

                if self._crossover_z is None:
                    # should get enough points for a good spline if we set the crossover scale to _primary_WKB_largest_z, since
                    # we checked above that enough points are available.
                    # The downside of this choice is only that it throws away numerical information that is potentially more accurate.
                    self._crossover_z = self._primary_WKB_largest_z.z
                    self._quality = "minimal"

                    WKB_data = self._build_WKB_values(
                        max_z=self._crossover_z, min_z=self._z_sample.min
                    )

                    self._metadata["comment"] = (
                        f"Calculation of crossover_z did not converge. WKB spline contains {len(WKB_data)} points."
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
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._numerical_smallest_z

    @property
    def primary_WKB_largest_z(self) -> Optional[redshift]:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._primary_WKB_largest_z

    @property
    def type(self) -> Optional[str]:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._type

    @property
    def quality(self) -> Optional[str]:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._quality

    @property
    def crossover_z(self) -> Optional[float]:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._crossover_z

    @property
    def Levin_z(self) -> Optional[float]:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._Levin_z

    @property
    def metadata(self) -> dict:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._metadata

    @property
    def functions(self) -> GkSourceFunctions:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "GkSource: attempt to construct functions, but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError("values have not yet been populated")

        if self._functions is None:
            self._create_functions()

        return self._functions

    def _create_functions(self):
        numerical_region = None
        numerical_Gk = None
        if self._numerical_smallest_z is not None:
            max_z = self._z_sample.max
            min_z = self._numerical_smallest_z

            # must be sorted into ascending order of redshift for a smoothing spline
            numerical_data = [
                v
                for v in self._values
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

            if len(numerical_data) > 5:
                numerical_region = (max_z.z, min_z.z)

                numerical_data.sort(key=lambda x: x.z_source.z)

                numerical_Gk_data = [
                    (log(v.z_source.z), v.numeric.G) for v in numerical_data
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
        if self._primary_WKB_largest_z is not None:
            max_z = self._primary_WKB_largest_z
            min_z = self._z_sample.min

            WKB_data = self._build_WKB_values(max_z, min_z)

            if len(WKB_data) > MIN_SPLINE_DATA_POINTS:
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
                        f"        GkSource (store_id={self.store_id}): z_response = {self._z_response.z:.5g} (store_id={self._z_response.store_id}), type = {self._type}, quality label = {self._quality}"
                    )
                    z_source_limit = sqrt(
                        self._k_exit.z_exit_subh_e3 * self._k_exit.z_exit_subh_e4
                    )
                    print(
                        f"        k = {self._k_exit.k.k_inv_Mpc:.5g}/Mpc, z_exit = {self._k_exit.z_exit:.5g}, z_source_limit = {z_source_limit:.5g}"
                    )
                    df = self.values_as_DataFrame()
                    df.to_csv("ERROR_VALUES.csv", header=True, index=False)
                    raise RuntimeError(
                        f"GkSource.create_functions: WKB data missing = intended region = ({WKB_region[0]:.5g}, {WKB_region[1]:.5g}), but data available only between ({WKB_data[-1].z_source.z:.5g}, {WKB_data[0].z_source.z:.5g})."
                    )

                sin_amplitude_data = [
                    (
                        log(v.z_source.z),
                        v.WKB.sin_coeff * sqrt(v.WKB.H_ratio / sqrt(v.omega_WKB_sq)),
                    )
                    for v in WKB_data
                ]

                theta_data = [
                    (
                        log(v.z_source.z),
                        v.WKB.theta,
                    )
                    for v in WKB_data
                ]

                sin_amplitude_x, sin_amplitude_y = zip(*sin_amplitude_data)
                theta_x, theta_y = zip(*theta_data)

                _sin_amplitude_spline = InterpolatedUnivariateSpline(
                    sin_amplitude_x,
                    sin_amplitude_y,
                    ext="raise",
                )
                _theta_spline = InterpolatedUnivariateSpline(
                    theta_x,
                    theta_y,
                    ext="raise",
                )

                WKB_Gk = GkWKBSplineWrapper(
                    _theta_spline,
                    _sin_amplitude_spline,
                    None,
                    "Gk WKB",
                    max_z.z,
                    min_z.z,
                )
                WKB_theta = ZSplineWrapper(
                    _theta_spline, "theta", max_z.z, min_z.z, log_z=True
                )
                WKB_theta_deriv = ZSplineWrapper(
                    _theta_spline.derivative(),
                    "theta derivative",
                    max_z.z,
                    min_z.z,
                    log_z=True,
                    deriv=True,
                )

        self._functions = GkSourceFunctions(
            numerical_region=numerical_region,
            numerical_Gk=numerical_Gk,
            WKB_region=WKB_region,
            WKB_Gk=WKB_Gk,
            theta=WKB_theta,
            theta_deriv=WKB_theta_deriv,
            type=self._type,
            quality=self._quality,
            crossover=self._crossover_z,
        )

    def _build_WKB_values(self, max_z, min_z):
        def max_predicate(z: redshift):
            if isinstance(max_z, redshift):
                return (
                    z.z <= max_z.z + DEFAULT_FLOAT_PRECISION
                    or z.store_id == max_z.store_id
                )

            return z.z <= max_z + DEFAULT_FLOAT_PRECISION

        def min_predicate(z: redshift):
            if isinstance(min_z, redshift):
                return (
                    z.z >= min_z.z - DEFAULT_FLOAT_PRECISION
                    or z.store_id == min_z.store_id
                )

            return z.z >= min_z - DEFAULT_FLOAT_PRECISION

        # must be sorted into ascending order of redshift for a smoothing spline
        WKB_data = [
            v
            for v in self._values
            if v.has_WKB and max_predicate(v.z_source) and min_predicate(v.z_source)
        ]

        WKB_data.sort(key=lambda x: x.z_source.z)

        return WKB_data

    @property
    def values(self) -> List:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError("GkSource: values read but _do_not_populate is set")

        if self._values is None:
            raise RuntimeError("values have not yet been populated")
        return self._values

    def values_as_DataFrame(self) -> pd.DataFrame:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError("GkSource: values read but _do_not_populate is set")

        if self._values is None:
            raise RuntimeError("values have not yet been populated")

        if self._values_df is None:
            self._create_values_df()

        return self._values_df

    def _create_values_df(self):
        z_source_store_id_col = [value.z_source.store_id for value in self._values]
        z_source_col = [value.z_source.z for value in self._values]
        z_source_efolds_subh_col = [
            self._model.efolds_subh(self._k_exit.k, value.z_source)
            for value in self._values
        ]
        has_numeric_col = [value.has_numeric for value in self._values]
        G_col = [value.numeric.G for value in self._values]
        Gprime_col = [value.numeric.Gprime for value in self._values]
        has_WKB_col = [value.has_WKB for value in self._values]
        theta_mod_2pi_col = [value.WKB.theta_mod_2pi for value in self._values]
        theta_div_2pi_col = [value.WKB.theta_div_2pi for value in self._values]
        theta_col = [value.WKB.theta for value in self._values]
        raw_theta_div_2pi_col = [value.WKB.raw_theta_div_2pi for value in self._values]
        raw_theta_col = [value.WKB.raw_theta for value in self._values]
        H_ratio_col = [value.WKB.H_ratio for value in self._values]
        sin_coeff_col = [value.WKB.sin_coeff for value in self._values]
        cos_coeff_col = [value.WKB.cos_coeff for value in self._values]
        z_init_col = [value.WKB.z_init for value in self._values]
        G_WKB_col = [value.WKB.G_WKB for value in self._values]
        new_G_WKB_col = [value.WKB.new_G_WKB for value in self._values]
        abs_G_WKB_err_col = [value.WKB.abs_G_WKB_err for value in self._values]
        rel_G_WKB_err_col = [value.WKB.rel_G_WKB_err for value in self._values]
        omega_WKB_sq_col = [value.omega_WKB_sq for value in self._values]
        WKB_criterion_col = [value.WKB_criterion for value in self._values]
        analytic_G_col = [value.analytic_G for value in self._values]
        analytic_Gprime_col = [value.analytic_Gprime for value in self._values]

        df = pd.DataFrame.from_dict(
            {
                "z_source_store_id": z_source_store_id_col,
                "z_source": z_source_col,
                "z_source_efolds_subh": z_source_efolds_subh_col,
                "has_numeric": has_numeric_col,
                "G": G_col,
                "Gprime": Gprime_col,
                "has_WKB": has_WKB_col,
                "theta_mod_2pi": theta_mod_2pi_col,
                "theta_div_2pi": theta_div_2pi_col,
                "theta": theta_col,
                "raw_theta_div_2pi": raw_theta_div_2pi_col,
                "raw_theta": raw_theta_col,
                "H_ratio": H_ratio_col,
                "sin_coeff": sin_coeff_col,
                "cos_coeff": cos_coeff_col,
                "z_init": z_init_col,
                "G_WKB": G_WKB_col,
                "new_G_WKB": new_G_WKB_col,
                "abs_G_WKB_err": abs_G_WKB_err_col,
                "rel_G_WKB_err": rel_G_WKB_err_col,
                "omega_WKB_sq": omega_WKB_sq_col,
                "WKB_criterion": WKB_criterion_col,
                "analytic_G": analytic_G_col,
                "analytic_Gprime": analytic_Gprime_col,
            }
        )

        df.sort_values(by="z_source", ascending=False, inplace=True, ignore_index=True)
        self._values_df = df

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

        self._classify()


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
        z_init: Optional[float] = None,
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

        def emit_value(values, attr):
            value = values[attr]
            if value is None:
                print(f"|    -- {attr} is missing")
                return

            if isinstance(value, float):
                print(f"|    -- {attr} = {value:.5g}")
                return

            print(f"|    -- {attr} = {value}")

        if has_numeric and not all_numeric:
            print(
                f"!! ERROR (GkSourceValue): only partial numeric data were supplied. Please supply all of G and Gprime."
            )
            attrs = {"G": G, "Gprime": Gprime}
            for attr in attrs:
                emit_value(attrs, attr)

            raise ValueError(
                "GkSourceValue: only partial numeric data were supplied. Please supply all of G and Gprime."
            )

        if has_WKB and not all_WKB:
            print(
                f"!! ERROR (GkSourceValue): only partial WKB data were supplied. Please supply all of theta_mod_2pi, theta_div_2pi, H_ratio, sin_coeff, cos_coeff and G_WKB."
            )
            attrs = {
                "theta_mod_2pi": theta_mod_2pi,
                "theta_div_2pi": theta_div_2pi,
                "H_ratio": H_ratio,
                "sin_coeff": sin_coeff,
                "cos_coeff": cos_coeff,
                "G_WKB": G_WKB,
            }
            for attr in attrs:
                emit_value(attrs, attr)

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
            theta=(
                _two_pi * theta_div_2pi + theta_mod_2pi
                if theta_div_2pi is not None and theta_mod_2pi is not None
                else None
            ),
            raw_theta_div_2pi=raw_theta_div_2pi,
            raw_theta=(
                _two_pi * raw_theta_div_2pi + theta_mod_2pi
                if raw_theta_div_2pi is not None and theta_mod_2pi is not None
                else None
            ),
            H_ratio=H_ratio,
            sin_coeff=sin_coeff,
            cos_coeff=cos_coeff,
            G_WKB=G_WKB,
            new_G_WKB=new_G_WKB,
            abs_G_WKB_err=abs_G_WKB_err,
            rel_G_WKB_err=rel_G_WKB_err,
            z_init=z_init,
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
