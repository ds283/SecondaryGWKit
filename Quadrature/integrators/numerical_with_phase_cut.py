from math import fabs
from typing import Optional

import ray
from scipy.integrate import solve_ivp

from ComputeTargets import ModelProxy, BackgroundModel
from CosmologyConcepts import wavenumber_exit_time, redshift, redshift_array, wavenumber
from LiouvilleGreen.integration_tools import find_phase_minimum
from Quadrature.integration_metadata import IntegrationData
from Quadrature.supervisors.numerical import NumericalIntegrationSupervisor
from Units import check_units
from defaults import (
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
    DEFAULT_FLOAT_PRECISION,
)

# State layout:
#    state[0] = value (T_k or G_k)
#    state[1] = derivative (T_k' or G_k')
VALUE_INDEX = 0
DERIV_INDEX = 1
EXPECTED_SOL_LENGTH = 2


@ray.remote
def numerical_with_phase_cut(
    model_proxy: ModelProxy,
    k: wavenumber_exit_time,
    z_init: redshift,
    z_sample: redshift_array,
    initial_value: float,
    initial_deriv: float,
    RHS,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
    delta_logz: Optional[float] = None,
    mode: str = None,
    stop_search_window_z_begin: Optional[float] = None,
    stop_search_window_z_end: Optional[float] = None,
    task_label: str = "numerical_with_phase_cut",
    object_label: str = "(object)",
) -> dict:
    k_wavenumber: wavenumber = k.k
    check_units(k_wavenumber, model_proxy)

    model: BackgroundModel = model_proxy.get()

    mode = mode.lower()
    if mode is not None and mode not in ["stop"]:
        raise ValueError(f'{task_label}: unknown compute mode "{mode}"')

    if mode in ["stop"]:
        if stop_search_window_z_begin is None:
            raise ValueError(
                "{label}: in 'stop' mode, stop_search_window_z_begin must be specified"
            )
        if stop_search_window_z_end is None:
            raise ValueError(
                "{label}: in 'stop' mode, stop_search_window_z_end must be specified"
            )

        if stop_search_window_z_begin < stop_search_window_z_end:
            stop_search_window_z_begin, stop_search_window_z_end = (
                stop_search_window_z_end,
                stop_search_window_z_begin,
            )
            print(
                f"## {task_label}: search window start/end arguments in the wrong order (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc). Now searching in interval: z in ({stop_search_window_z_begin}, {stop_search_window_z_end})"
            )

        max_z = z_init.z
        min_z = z_sample.min.z
        if stop_search_window_z_begin > max_z:
            raise ValueError(
                f"{task_label}: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) specified 'stop' window starting redshift z={stop_search_window_z_begin:.5g} exceeds source redshift z_source={max_z:.5g}"
            )
        if stop_search_window_z_end < min_z:
            print(
                f"## {task_label}: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) specified 'stop' window ending redshift z={stop_search_window_z_end:.5g} is smaller than lowest z-response sample point z={min_z:.5g}. Search will terminate at z={min_z:.5g}."
            )
            stop_search_window_z_end = min_z

        if (
            fabs(stop_search_window_z_begin - stop_search_window_z_end)
            < DEFAULT_FLOAT_PRECISION
        ):
            raise ValueError(
                f"## {task_label}: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) specified search window has effectively zero extent"
            )

        if (
            fabs(z_init.z - z_sample.min.z) < DEFAULT_ABS_TOLERANCE
            or z_init.store_id == z_sample.min.store_id
        ):
            raise ValueError(
                f"## {task_label}: (for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc) in 'stop' mode, the source redshift and the lowest response redshift cannot be equal"
            )

    # obtain dimensionful value of wavenumber; this should be measured in the same units used by the cosmology
    k_float = k_wavenumber.k
    z_min = float(z_sample.min)

    with NumericalIntegrationSupervisor(
        k_wavenumber, z_init, z_sample.min, object_label, delta_logz=delta_logz
    ) as supervisor:
        initial_state = [initial_value, initial_deriv]

        if mode == "stop":
            # set up an event to terminate the integration after the end of the search window
            def stop_event(z, state, model, k_float, supervisor):
                return z - stop_search_window_z_end + DEFAULT_FLOAT_PRECISION

            # mark stop_event as terminal
            stop_event.terminal = True

            events = [stop_event]

            # need dense output for the root-finding algorithm, used to cut at a point of fixed phase
            dense_output = True
        else:
            events = None
            dense_output = False

        sol = solve_ivp(
            RHS,
            method="DOP853",
            t_span=(z_init.z, z_min),
            y0=initial_state,
            t_eval=z_sample.as_float_list(),
            events=events,
            dense_output=dense_output,
            atol=atol,
            rtol=rtol,
            args=(
                model,
                k_float,
                supervisor,
            ),
        )

    # test whether the integration concluded successfully
    if not sol.success:
        raise RuntimeError(
            f'{task_label}: integration did not terminate successfully (k={k_wavenumber.k_inv_Mpc}/Mpc, z_source={z_init.z}, error at z={sol.t[-1]}, "{sol.message}")'
        )

    if mode == "stop" and sol.status != 1:
        # in "stop" mode, we expect the integration to finish at an event; if this doesn't happen, it implies
        # we somehow missed the termination criterion
        raise RuntimeError(
            f'{task_label}: mode is "{mode}", but integration did not finish at a termination event'
        )

    sampled_z = sol.t
    sampled_data = sol.y
    if len(sampled_z) > 0 and len(sampled_data) != EXPECTED_SOL_LENGTH:
        raise RuntimeError(
            f"{task_label}: solution does not have expected number of members (expected {EXPECTED_SOL_LENGTH}, found {len(sampled_values)}; k={k_wavenumber.k_inv_Mpc}/Mpc, length of sol.t={len(sampled_z)})"
        )
    if len(sampled_data) > 0:
        sampled_values = sampled_data[VALUE_INDEX]
        sampled_derivs = sampled_data[DERIV_INDEX]
    else:
        sampled_values = []
        sampled_derivs = []

    # if no data points returned, check if this is because the target z (ie., lowest z_response)
    # and the source z agree.
    # If so, then we know the correct value from the initial data.
    if (
        len(sampled_z) == 0
        and len(z_sample) == 0
        and (
            fabs(z_init.z - z_sample.min.z) < DEFAULT_ABS_TOLERANCE
            or z_init.store_id == z_sample.min.store_id
        )
    ):
        sampled_z.append(z_init.z)
        sampled_values.append(0.0)
        sampled_derivs.append(1.0)

    returned_values = len(sampled_z)
    if mode != "stop":
        expected_values = len(z_sample)

        if returned_values != expected_values:
            raise RuntimeError(
                f"{task_label}: solve_ivp returned {returned_values} samples, but expected {expected_values}"
            )

    stop_deltaz_subh = None
    stop_value = None
    stop_deriv = None

    if mode == "stop":
        # find value of solution and derivative at a point of fixed phase (taken to be a minimum of the function)
        # we want to cut at a fixed phase to make the subsequent calculation of a WKB phase as stable
        # as possible (don't want jitter in the final value of the phase, just from starting at a
        # different point in the cycle)
        payload = find_phase_minimum(
            sol.sol,
            start_z=stop_search_window_z_begin,
            stop_z=stop_search_window_z_end,
            value_index=VALUE_INDEX,
            deriv_index=DERIV_INDEX,
        )
        stop_deltaz_subh = k.z_exit - payload["z"]
        stop_value = payload["value"]
        stop_deriv = payload["derivative"]

    # validate that the samples of the solution correspond to the z-sample points that we specified.
    # This really should be true, but there is no harm in being defensive.
    for i in range(returned_values):
        diff = sampled_z[i] - z_sample[i].z
        if fabs(diff) > DEFAULT_ABS_TOLERANCE:
            raise RuntimeError(
                f"{task_label}: solve_ivp returned sample points that differ from those requested (difference={diff} at i={i})"
            )

    return {
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=int(sol.nfev),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "value_sample": sampled_values,
        "deriv_sample": sampled_derivs,
        "solver_label": "solve_ivp+DOP853-stepping0",
        "has_unresolved_osc": supervisor.has_unresolved_osc,
        "unresolved_z": supervisor.unresolved_z,
        "unresolved_efolds_subh": supervisor.unresolved_efolds_subh,
        "stop_deltaz_subh": stop_deltaz_subh,
        "stop_value": stop_value,
        "stop_deriv": stop_deriv,
    }
