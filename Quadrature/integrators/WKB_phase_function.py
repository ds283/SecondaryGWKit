from math import fabs, sqrt, log
from typing import List

import ray
from scipy.integrate import solve_ivp

from ComputeTargets import BackgroundModel, ModelProxy
from CosmologyConcepts import wavenumber_exit_time, wavenumber, redshift_array
from LiouvilleGreen.WKBtools import WKB_mod_2pi, WKB_product_mod_2pi
from Quadrature.integration_metadata import IntegrationData
from Quadrature.supervisors.WKB import ThetaSupervisor, QSupervisor
from Quadrature.supervisors.base import RHS_timer
from Quadrature.supervisors.numeric import NumericIntegrationSupervisor
from Units import check_units
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE

THETA_INDEX = 0
Q_INDEX = 0
EXPECTED_PHASE_SOL_LENGTH = 1

FRICTION_INDEX = 0
EXPECTED_FRICTION_SOL_LENGTH = 1

# how large do we allow the WKB phase theta to become, before we terminate the integration and
# reset to a small value?
# we need to resolve the phase on the scale of (0, 2pi), otherwise we will compute cos(theta),
# sin(theta) and hence G_WKB incorrectly
DEFAULT_PHASE_RUN_LENGTH = 1e4

# how large do we allow omega_WKB_sq to get before switching to a "stage #2" integration?
DEFAULT_OMEGA_WKB_SQ_MAX = 1e6


def stage_1_evolution(
    model: BackgroundModel,
    k: wavenumber_exit_time,
    z_init: float,
    z_target: float,
    z_sample_list: List[float],
    z_break_list: List[float],
    omega_sq,
    d_ln_omega_dz,
    sampled_z: List[float],
    sampled_theta_mod_2pi: List[float],
    sampled_theta_div_2pi: List[float],
    current_div_2pi_offset: int,
    atol: float,
    rtol: float,
    task_label: str,
    object_label: str,
) -> dict:
    k_wavenumber: wavenumber = k.k
    k_float = k_wavenumber.k

    def phase_cycle_trigger(z, state, _) -> float:
        # terminate when the phase has accumulated to smaller than -1E3
        theta = state[THETA_INDEX]
        return theta + DEFAULT_PHASE_RUN_LENGTH

    phase_cycle_trigger.terminal = True

    def terminate_trigger(z, state, _) -> float:
        # terminate when omega_WKB_sq exceeds 1E6 (at that point we want to switch to a different numerical scheme)
        omega_WKB_sq = omega_sq(model, k_float, z)

        return omega_WKB_sq - DEFAULT_OMEGA_WKB_SQ_MAX

    terminate_trigger.terminal = True

    def RHS(z, state, supervisor) -> List[float]:
        with RHS_timer(supervisor) as timer:
            theta = state[THETA_INDEX]

            if supervisor.notify_available:
                supervisor.message(
                    z,
                    f"current state: theta_WKB = {theta:.5g}, 2pi offset = {current_div_2pi_offset}, {len(sampled_z)} stored samples, {len(z_break_list)} phase resets",
                )
                supervisor.reset_notify_time()

            omega_sq_value = omega_sq(model, k_float, z)

            if omega_sq_value < 0.0:
                raise ValueError(
                    f"{task_label}: omega_WKB^2 cannot be negative during WKB integration (omega_WKB^2={omega_sq_value:.5g})"
                )

            d_ln_omega_dz_value = d_ln_omega_dz(model, k_float, z)
            WKB_criterion = fabs(d_ln_omega_dz_value) / sqrt(fabs(omega_sq_value))
            if WKB_criterion > 1.0:
                H = model.functions.Hubble(z)
                k_over_H = k_float / H
                supervisor.report_WKB_violation(z, log((1.0 + z) * k_over_H))

            omega_value = sqrt(omega_sq_value)
            dtheta_dz = omega_value

            return [dtheta_dz]

    with ThetaSupervisor(k_wavenumber, z_init, z_target, object_label) as supervisor:
        state = [0.0]
        t_span = (z_init, z_target)

        z_terminate = None
        theta_terminate_mod_2pi = None

        while len(z_sample_list) > 0:
            sol = solve_ivp(
                RHS,
                method="DOP853",
                t_span=t_span,
                y0=state,
                t_eval=z_sample_list,
                events=[phase_cycle_trigger, terminate_trigger],
                atol=atol,
                rtol=rtol,
                args=(supervisor,),
            )

            if not sol.success:
                raise RuntimeError(
                    f'{task_label}: phase function stage #1 integration did not terminate successfully (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g}, error at z={sol.t[-1]:.5g}, "{sol.message}")'
                )

            supervisor.notify_new_nfev(int(sol.nfev))
            batch_z = sol.t
            batch_values = sol.y

            # check if integration stopped due to a termination event.
            # If not, we should have reached the end of the sample list
            if sol.status != 1:
                if len(batch_z) != len(z_sample_list):
                    raise RuntimeError(
                        f"{task_label}: phase function stage #1 integration reached end of domain, but an incorrect number of z-samples were recorded (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g}, expected={len(z_sample_list)}, found={len(batch_z)})"
                    )
            else:
                if len(batch_z) >= len(z_sample_list):
                    raise RuntimeError(
                        f"{task_label}: phase function stage #1 integration terminated at a phase cutoff, but no z-sample points are left (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g}, max expected={len(z_sample_list)}, found={len(batch_z)})"
                    )

            # check we have expected number of components (just one!) in the solution
            if len(batch_z) > 0:
                if len(batch_values) != EXPECTED_PHASE_SOL_LENGTH:
                    print(
                        f"!! {task_label}: phase function stage #1 solution does not have expected number of members for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc"
                    )
                    print(
                        f"   -- expected {EXPECTED_PHASE_SOL_LENGTH} members, found {len(batch_values)}"
                    )
                    print(
                        f"      z_init={supervisor.z_init:.5g}, z_target={supervisor.z_target:.5g}"
                    )
                    print(f"      sol.success={sol.success}, sol.message={sol.message}")
                    raise RuntimeError(
                        f"{task_label}: phase function stage #1 solution does not have expected number of members (expected {EXPECTED_PHASE_SOL_LENGTH}, found {len(batch_values)}; k={k_wavenumber.k_inv_Mpc}/Mpc, length of sol.t={len(batch_z)})"
                    )

                batch_theta = batch_values[THETA_INDEX]

                # walk through sampled values, evaluating theta div 2pi and theta mod 2pi
                for z, theta in zip(batch_z, batch_theta):
                    theta_div_2pi, theta_mod_2pi = WKB_mod_2pi(theta)

                    sampled_z.append(z)
                    sampled_theta_mod_2pi.append(theta_mod_2pi)
                    sampled_theta_div_2pi.append(theta_div_2pi + current_div_2pi_offset)

                    assert fabs(z - z_sample_list[0]) < DEFAULT_ABS_TOLERANCE
                    z_sample_list.pop(
                        0
                    )  # could be expensive, any cheaper way to do this?

            # if more work remains, prime 'state' and 'z_init' to restart the integration
            if len(z_sample_list) > 0:
                recycle_times = sol.t_events[0]
                terminate_times = sol.t_events[1]

                if len(recycle_times) == 1:
                    new_z_init = recycle_times[0]
                    t_span = (new_z_init, z_target)
                    z_break_list.append(new_z_init)

                    values = sol.y_events[0]
                    if len(values) != 1:
                        raise RuntimeError(
                            f"{task_label}: could not find event record to restart phase integration (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g})"
                        )
                    new_theta_init = values[0]
                    new_theta_init_div_2pi, new_theta_init_mod_2pi = WKB_mod_2pi(
                        new_theta_init
                    )

                    state = [new_theta_init_mod_2pi]
                    current_div_2pi_offset += new_theta_init_div_2pi

                    continue

                elif len(terminate_times) == 1:
                    z_terminate = terminate_times[0]
                    values = sol.y_events[1]
                    if len(values) != 1:
                        raise RuntimeError(
                            f"{task_label}: could not find event record to terminate stage 1 integration (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g})"
                        )
                    theta_terminate = values[0]
                    theta_terminate_div_2pi, theta_terminate_mod_2pi = WKB_mod_2pi(
                        theta_terminate
                    )

                    current_div_2pi_offset += theta_terminate_div_2pi

                    break

                else:
                    raise RuntimeError(
                        f"{task_label}: could not find event record to restart phase integration (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g})"
                    )

    return {
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=supervisor.nfev,
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "has_WKB_violation": supervisor.has_WKB_violation,
        "WKB_violation_z": supervisor.WKB_violation_z,
        "WKB_violation_efolds_subh": supervisor.WKB_violation_efolds_subh,
        "current_div_2pi_offset": current_div_2pi_offset,
        "z_terminate": z_terminate,
        "theta_terminate_mod_2pi": theta_terminate_mod_2pi,
    }


def stage_2_evolution(
    model: BackgroundModel,
    k: wavenumber_exit_time,
    z_init: float,
    z_target: float,
    z_sample_list: List[float],
    omega_sq,
    d_ln_omega_dz,
    sampled_z: List[float],
    sampled_theta_mod_2pi: List[float],
    sampled_theta_div_2pi: List[float],
    theta_mod_2pi_init: float,
    current_div_2pi_offset: int,
    atol: float,
    rtol: float,
    task_label: str,
    object_label: str,
) -> dict:
    k_wavenumber: wavenumber = k.k
    k_float = k_wavenumber.k

    omega_sq_init = omega_sq(model, k_float, z_init)
    omega_init = sqrt(omega_sq_init)

    # the idea here is to change variables so that u = z_init - z, theta = theta_init + omega_WKB_init Q (1+u).
    # Then Q will typically be fairly close to unity for typical evolutions when u >> 1, because omega_WKB is
    # fairly constant, and theta will be growing nearly like u

    def RHS(u, state, supervisor) -> List[float]:
        with RHS_timer(supervisor) as timer:
            Q = state[Q_INDEX]
            z = z_init - u

            supervisor.update_Q(Q)

            if supervisor.notify_available:
                supervisor.message(u, f"current state: Q = {Q:.8g}")
                supervisor.reset_notify_time()

            omega_sq_value = omega_sq(model, k_float, z)

            if omega_sq_value < 0.0:
                raise ValueError(
                    f"{task_label}: omega_WKB^2 cannot be negative during WKB integration (omega_WKB^2={omega_sq_value:.5g})"
                )

            d_ln_omega_dz_value = d_ln_omega_dz(model, k_float, z)
            WKB_criterion = fabs(d_ln_omega_dz_value) / sqrt(fabs(omega_sq_value))
            if WKB_criterion > 1.0:
                H = model.functions.Hubble(z)
                k_over_H = k_float / H
                supervisor.report_WKB_violation(z, log((1.0 + z) * k_over_H))

            omega__value = sqrt(omega_sq_value)
            one_plus_u = 1.0 + u
            dQ_du = -omega__value / omega_init / one_plus_u - Q / one_plus_u

            return [dQ_du]

    u_init = 0
    u_target = z_init - z_target
    with QSupervisor(k_wavenumber, u_init, u_target, object_label) as supervisor:
        # initial condition is Q = 0 at u = 0
        # then we expect Q close to unity for u >> 1
        state = [0.0]
        u_sample_list = [z_init - z for z in z_sample_list]

        sol = solve_ivp(
            RHS,
            method="DOP853",
            t_span=(u_init, u_target),
            y0=state,
            t_eval=u_sample_list,
            atol=atol,
            rtol=rtol,
            args=(supervisor,),
        )

        if not sol.success:
            raise RuntimeError(
                f'{task_label}: phase function stage #2 integration did not terminate successfully (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g}, error at u={sol.t[-1]:.5g}, "{sol.message}")'
            )

        batch_u = sol.t
        batch_values = sol.y

        if sol.status != 1:
            if len(batch_u) != len(u_sample_list):
                raise RuntimeError(
                    f"{task_label}: phase function stage #2 integration reached end of domain, but an incorrect number of z-samples were recorded (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g}, expected={len(z_sample_list)}, found={len(batch_u)})"
                )

        if len(batch_values) != EXPECTED_PHASE_SOL_LENGTH:
            print(
                f"!! {task_label}: phase function stage #2 solution does not have expected number of members for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc"
            )
            print(
                f"   -- expected {EXPECTED_PHASE_SOL_LENGTH} members, found {len(batch_values)}"
            )
            print(f"      z_init={z_init:.5g}, z_target={z_target:.5g}")
            print(f"      sol.success={sol.success}, sol.message={sol.message}")
            raise RuntimeError(
                f"{task_label}: phase function stage #2 solution does not have expected number of members (expected {EXPECTED_PHASE_SOL_LENGTH}, found {len(batch_values)}; k={k_wavenumber.k_inv_Mpc}/Mpc, length of sol.t={len(batch_u)})"
            )

        batch_Q = batch_values[Q_INDEX]

        # walk through the sampled values returned from solve_ivp, computing theta div 2pi and theta mod 2pi at each step
        for u, Q in zip(batch_u, batch_Q):
            # WKB_product_mod_2pi uses our custom range reduction method in an attempt to maintain precision mod 2pi when the product
            # omega_WKB_init * (1+u) * Q becomes very large
            theta_div_2pi, theta_mod_2pi = WKB_product_mod_2pi(
                omega_init * (1.0 + u), Q, theta_mod_2pi_init
            )

            z = z_init - u

            assert fabs(z - z_sample_list[0]) < DEFAULT_ABS_TOLERANCE
            z_sample_list.pop(0)

            sampled_z.append(z)
            sampled_theta_mod_2pi.append(theta_mod_2pi)
            sampled_theta_div_2pi.append(theta_div_2pi + current_div_2pi_offset)

    return {
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=int(sol.nfev),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "has_WKB_violation": supervisor.has_WKB_violation,
        "WKB_violation_z": supervisor.WKB_violation_z,
        "WKB_violation_efolds_subh": supervisor.WKB_violation_efolds_subh,
        "largest_Q": supervisor.largest_Q,
        "smallest_Q": supervisor.smallest_Q,
    }


def integrate_phase_function(
    model: BackgroundModel,
    k: wavenumber_exit_time,
    z_init: float,
    z_sample: redshift_array,
    omega_sq,
    d_ln_omega_dz,
    omega_sq_init,
    atol: float,
    rtol: float,
    metadata,
    task_label: str,
    object_label: str,
) -> dict:
    # We need to keep theta integration in bounds, otherwise we lose accuracy in the WKB solution.
    # Unfortunately there is no simple way to implement this directly in solve_ivp, which would require
    # inflight modification of the ODE state. See https://github.com/scipy/scipy/issues/19645.
    #
    # The solution used here is to compute the phase mod 2pi, i.e., we try to track theta mod 2pi
    # and theta div 2pi separately. We do that by postprocessing the output of the integration.
    # To keep accuracy, we keep cutting the integral after we've stepped about 1E3 in redshift.
    z_sample_list = z_sample.as_float_list()
    z_break_list = []
    z_target = z_sample.min.z

    # set up empty list of sampled z points
    sampled_z = []

    # empty list of sampled theta values (mod 2pi and div 2pi)
    sampled_theta_mod_2pi = []
    sampled_theta_div_2pi = []

    # keep track of the value of theta div 2pi that we need to add - because we break the integration
    # into stages, we need to add an offset to subsequent stages so that theta lines up properly
    current_div_2pi_offset = 0

    # if omega_WKB < 1E3, run a stage #1 integration.
    # This could exhaust the possible sample points, or it might terminate so that we can start a "stage 2" integration
    stage_1_data = None
    if len(z_sample_list) > 0 and omega_sq_init < DEFAULT_OMEGA_WKB_SQ_MAX:
        stage_1_data = stage_1_evolution(
            model,
            k,
            z_init,
            z_target,
            z_sample_list,
            z_break_list,
            omega_sq,
            d_ln_omega_dz,
            sampled_z,
            sampled_theta_mod_2pi,
            sampled_theta_div_2pi,
            current_div_2pi_offset,
            atol,
            rtol,
            task_label,
            object_label,
        )

        metadata["phase_cycle_events"] = len(z_break_list)

    # if there is any work left to do (so stage #1 didn't run, or didn't get to the end of the z sample points),
    # do a stage #2 integration
    stage_2_data = None
    if len(z_sample_list) > 0:
        current_div_2pi_offset: int = (
            stage_1_data["current_div_2pi_offset"]
            if stage_1_data is not None
            else current_div_2pi_offset
        )

        z_init: float = (
            stage_1_data["z_terminate"] if stage_1_data is not None else z_init
        )

        theta_stage2_init_mod_2pi: float = (
            stage_1_data["theta_terminate_mod_2pi"] if stage_1_data is not None else 0.0
        )

        metadata["stage1_z_terminate"] = z_init
        metadata["stage1_div_2pi_offset"] = current_div_2pi_offset
        metadata["stage1_theta_mod_2pi"] = theta_stage2_init_mod_2pi

        stage_2_data = stage_2_evolution(
            model,
            k,
            z_init,
            z_target,
            z_sample_list,
            omega_sq,
            d_ln_omega_dz,
            sampled_z,
            sampled_theta_mod_2pi,
            sampled_theta_div_2pi,
            theta_stage2_init_mod_2pi,
            current_div_2pi_offset,
            atol,
            rtol,
            task_label,
            object_label,
        )

        metadata["stage_2_largest_Q"] = stage_2_data["largest_Q"]
        metadata["stage_2_smallest_Q"] = stage_2_data["smallest_Q"]

    # check that at least one stage returned some usable data
    if (
        stage_1_data is None
        and stage_2_data is None
        and not metadata.get("initial_data_only", False)
    ):
        raise RuntimeError("{task_label}: both stage #1 and stage #2 data are empty")

    # check that combination of stage #1 + stage #2 covered all the z sample points
    returned_values = len(sampled_z)
    expected_values = len(z_sample)
    if returned_values != expected_values:
        raise RuntimeError(
            f"{task_label}: stage #1 and stage #2 evolution produced {returned_values} samples, but expected {expected_values}"
        )

    # validate that the samples of the solution correspond to the z-sample points that we specified.
    # This really should be true (and has also been checked in the stage #1 and stage #2 integrations
    # separately), but there is no harm in being defensive.
    for i in range(returned_values):
        diff = sampled_z[i] - z_sample[i].z
        if fabs(diff) > DEFAULT_ABS_TOLERANCE:
            raise RuntimeError(
                f"{task_label}: stage #1 and stage #2 evolution returned sample points that differ from those requested (difference={diff} at i={i})"
            )

    def merge_stage_data(field, merge_operation, allow_none=False):
        stage_1_field = stage_1_data[field] if stage_1_data is not None else None
        stage_2_field = stage_2_data[field] if stage_2_data is not None else None

        if stage_1_field is None and stage_2_field is not None:
            return stage_2_field

        if stage_1_field is not None and stage_2_field is None:
            return stage_1_field

        if stage_1_field is not None and stage_2_field is not None:
            return merge_operation(stage_1_field, stage_2_field)

        if allow_none:
            return None

        raise RuntimeError(
            f'{task_label}: both stage #1 and stage #2 fields are empty for payload "{field}"'
        )

    return {
        "stage_1_data": stage_1_data["data"] if stage_1_data is not None else None,
        "stage_2_data": stage_2_data["data"] if stage_2_data is not None else None,
        "theta_div_2pi_sample": sampled_theta_div_2pi,
        "theta_mod_2pi_sample": sampled_theta_mod_2pi,
        "phase_solver_label": "solve_ivp+DOP853-stepping0",
        "has_WKB_violation": merge_stage_data(
            "has_WKB_violation", lambda a, b: a or b, allow_none=True
        ),
        "WKB_violation_z": merge_stage_data(
            "WKB_violation_z", lambda a, b: a, allow_none=True
        ),
        "WKB_violation_efolds_subh": merge_stage_data(
            "WKB_violation_efolds_subh", lambda a, b: a, allow_none=True
        ),
    }


def integrate_friction_function(
    model, k, z_init, z_sample, friction, atol, rtol, metadata, task_label, object_label
) -> dict:
    k_wavenumber: wavenumber = k.k

    k_float = float(k_wavenumber.k)
    z_min = float(z_sample.min)

    with NumericIntegrationSupervisor(
        k_wavenumber,
        z_init,
        z_sample.min,
        task_label,
        delta_logz=None,
    ) as supervisor:
        initial_state = [0.0]

        sol = solve_ivp(
            friction,
            method="DOP853",
            t_span=(z_init, z_min),
            y0=initial_state,
            t_eval=z_sample.as_float_list(),
            dense_output=False,
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
            f'{task_label}: friction integration did not terminate successfully (k={k_wavenumber.k_inv_Mpc}/Mpc, z_source={z_init.z}, error at z={sol.t[-1]}, "{sol.message}")'
        )

    sampled_z = sol.t
    sampled_data = sol.y
    if len(sampled_z) > 0 and len(sampled_data) != EXPECTED_FRICTION_SOL_LENGTH:
        raise RuntimeError(
            f"{task_label}: solution does not have expected number of members (expected {EXPECTED_FRICTION_SOL_LENGTH}, found {len(sampled_values)}; k={k_wavenumber.k_inv_Mpc}/Mpc, length of sol.t={len(sampled_z)})"
        )
    if len(sampled_data) > 0:
        sampled_friction = sampled_data[FRICTION_INDEX]
    else:
        sampled_friction = []

    returned_values = len(sampled_z)
    expected_values = len(z_sample)

    if returned_values != expected_values:
        raise RuntimeError(
            f"{task_label}: solve_ivp returned {returned_values} samples, but expected {expected_values}"
        )

    # validate that the samples of the solution correspond to the z-sample points that we specified.
    # This really should be true, but there is no harm in being defensive.
    for i in range(returned_values):
        diff = sampled_z[i] - z_sample[i].z
        if fabs(diff) > DEFAULT_ABS_TOLERANCE:
            raise RuntimeError(
                f"{task_label}: solve_ivp returned sample points that differ from those requested (difference={diff} at i={i})"
            )

    return {
        "friction_data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=int(sol.nfev),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "friction_sample": sampled_friction,
        "friction_solver_label": "solve_ivp+DOP853-stepping0",
    }


@ray.remote
def WKB_phase_function(
    model_proxy: ModelProxy,
    k: wavenumber_exit_time,
    z_init: float,
    z_sample: redshift_array,
    omega_sq,
    d_ln_omega_dz,
    friction=None,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
    task_label: str = "WKB_phase_function",
    object_label: str = "(object)",
) -> dict:
    k_wavenumber: wavenumber = k.k
    k_float = float(k_wavenumber.k)

    check_units(k_wavenumber, model_proxy)

    model: BackgroundModel = model_proxy.get()

    omega_sq_init = omega_sq(model, k_float, z_init)
    d_ln_omega_dz_init = d_ln_omega_dz(model, k_float, z_init)

    if omega_sq_init < 0.0:
        raise RuntimeError(
            f"{task_label}: omega_WKB^2 is negative at the initial time for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc (z_init={z_init:.5g}, omega_WKB^2={omega_sq_init:.5g})"
        )

    WKB_criterion_init = fabs(d_ln_omega_dz_init) / sqrt(fabs(omega_sq_init))
    if WKB_criterion_init > 1.0:
        raise RuntimeError(
            f"{task_label}: WKB criterion |d(omega_WKB)/omega_WKB^2| > 1 at the initial time for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc (z_init={z_init:.5g}, WKB criterion={WKB_criterion_init:.5g})"
        )

    metadata = {}

    # check whether initial and target redshifts are equal; if so, the solution is just 0
    if fabs(z_init - z_sample.min.z) < atol:
        metadata["initial_data_only"] = True

        payload = {
            "theta_div_2pi_sample": [0.0],
            "theta_mod_2pi_sample": [0.0],
            "phase_solver_label": "solve_ivp+DOP853-stepping0",
            "has_WKB_violation": False,
            "WKB_violation_z": None,
            "WKB_violation_efolds_subh": None,
            "stage_1_data": None,
            "stage_2_data": None,
        }
        if friction is not None:
            payload["friction_sample"] = [0.0]
            payload["friction_solver_label"] = "solve_ivp+DOP853-stepping0"

    else:
        payload = integrate_phase_function(
            model,
            k,
            z_init,
            z_sample,
            omega_sq,
            d_ln_omega_dz,
            omega_sq_init,
            atol,
            rtol,
            metadata,
            task_label,
            object_label,
        )

        if friction is not None:
            payload = payload | integrate_friction_function(
                model,
                k,
                z_init,
                z_sample,
                friction,
                atol,
                rtol,
                metadata,
                task_label,
                object_label,
            )

    return payload | {"metadata": metadata}
