from math import fabs, log, pi, sqrt
from typing import Callable, Optional, Sequence

import ray
from scipy.integrate import solve_ivp

from ComputeTargets import ModelProxy, BackgroundModel
from CosmologyConcepts import wavenumber_exit_time, redshift, redshift_array, wavenumber
from LiouvilleGreen.integration_tools import find_phase_extremum
from Quadrature.integration_metadata import IntegrationData
from Quadrature.supervisors.numeric import NumericIntegrationSupervisor
from Units import check_units
from config.defaults import (
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


def scan_sample_grid_for_unresolved_osc(
    model: BackgroundModel,
    k: wavenumber,
    k_float: float,
    omega_sq: Callable[[BackgroundModel, float, float], float],
    sampled_z: Sequence[float],
    object_label: str,
) -> dict:
    """
    Test whether the *returned sample grid* is fine enough to resolve the oscillations the
    integrator stepped through, and print a warning if it is not.

    This is the oscillation-resolution diagnostic that used to be evaluated inside the ODE
    right-hand side, through ``NumericIntegrationSupervisor.report_wavelength``. Two things
    changed (prompt 11 of ``prompts/GkTk-remedial``):

    **What is tested.** The flag's documented meaning is "the sample grid *you supplied* is too
    coarse to resolve the oscillations I stepped through" (review §13.1) -- it is not a statement
    about the integrator's own accuracy, which review §10.1 measures independently at 2e-7 of the
    envelope. The test therefore now compares the local wavelength against the actual spacing of
    consecutive returned samples, rather than against a nominal spacing reconstructed from a
    ``delta_logz`` argument. That reconstruction carried two errors: ``delta_logz`` is supplied as
    a spacing in log10(1+z) and was consumed as a spacing in ln(1+z), understating the spacing by
    ln 10; and for ``GkNumericIntegration`` the value supplied describes the *source* grid while
    the samples are taken on the *response* grid, which is 12x sparser.

    **What is sampled.** On the right-hand side the test saw every internal step of the solver;
    evaluated on the sample grid it sees only the output points. A wavelength minimum that falls
    strictly between two output samples is therefore no longer caught. Since the subject of the
    test *is* the output grid, this is the more faithful form -- but the difference is real and is
    recorded here deliberately (review §13.1 asks that whoever implements this state which
    behaviour was chosen).

    The scan reports the **first** (largest-z) consecutive pair that fails, matching the old
    behaviour of latching on the first failure and reporting nothing afterwards.

    :param model: background model, supplying ``functions.Hubble``
    :param k: the wavenumber object, for the warning message
    :param k_float: dimensionful wavenumber, in the cosmology's units
    :param omega_sq: the sector's effective frequency, called as ``omega_sq(model, k_float, z)``
    :param sampled_z: the redshifts actually returned by the solver, in descending order
    :param object_label: label for the warning message
    :return: dict with ``has_unresolved_osc``, ``unresolved_z`` and ``unresolved_efolds_subh``
    """
    for i in range(len(sampled_z) - 1):
        z = float(sampled_z[i])

        omega_WKB_sq = omega_sq(model, k_float, z)
        if omega_WKB_sq <= 0.0:
            # not oscillatory here; there is nothing to resolve
            continue

        wavelength = 2.0 * pi / sqrt(omega_WKB_sq)
        grid_spacing = fabs(z - float(sampled_z[i + 1]))

        if wavelength < grid_spacing:
            efolds_subh = log((1.0 + z) * k_float / model.functions.Hubble(z))
            print(
                f"!! WARNING: {object_label} integration for k = {k.k_inv_Mpc:.5g}/Mpc (store_id={k.store_id}) may have developed unresolved oscillations"
            )
            print(
                f"|    current z={z:.5g}, e-folds inside horizon={efolds_subh:.3g} | approximate wavelength Delta z={wavelength:.5g}, approximate grid spacing at this z: {grid_spacing:.5g}"
            )
            return {
                "has_unresolved_osc": True,
                "unresolved_z": z,
                "unresolved_efolds_subh": efolds_subh,
            }

    return {
        "has_unresolved_osc": False,
        "unresolved_z": None,
        "unresolved_efolds_subh": None,
    }


@ray.remote
def numeric_with_phase_cut(
    model_proxy: ModelProxy,
    k: wavenumber_exit_time,
    z_init: redshift,
    z_sample: redshift_array,
    initial_value: float,
    initial_deriv: float,
    RHS,
    omega_sq: Optional[Callable[[BackgroundModel, float, float], float]] = None,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
    delta_logz: Optional[float] = None,
    mode: str = None,
    stop_search_window_z_begin: Optional[float] = None,
    stop_search_window_z_end: Optional[float] = None,
    task_label: str = "numeric_with_phase_cut",
    object_label: str = "(object)",
) -> dict:
    k_wavenumber: wavenumber = k.k
    check_units(k_wavenumber, model_proxy)

    model: BackgroundModel = model_proxy.get()

    # note the None test must come first: mode=None means "integrate the whole grid", and
    # calling .lower() on it raises AttributeError (review §10.2)
    if mode is not None:
        mode = mode.lower()

        if mode not in ["stop"]:
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

    # delta_logz is still accepted and still handed to the supervisor, so that callers (main.py)
    # need not change; but the oscillation-resolution diagnostic no longer uses it. It now runs
    # after the solve, against the actual spacing of the returned samples -- see
    # scan_sample_grid_for_unresolved_osc and NumericIntegrationSupervisor.report_wavelength.
    with NumericIntegrationSupervisor(
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

            # need dense output for the root-finding algorithm, used to cut at a point of fixed
            # phase (an extremum of the solution; see find_phase_extremum)
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
        # find value of solution and derivative at a point of fixed phase. The extremum found is a
        # *maximum* of the solution -- the derivative passes from negative to positive as z
        # decreases -- and review §10.1 measures value/envelope = +1.000000 there in every run.
        # Cutting at a reproducible point of fixed phase is what matters; which extremum it is does
        # not, because the WKB objects' store() rotates arbitrary (value, derivative) initial data
        # into a pure sine. The older motivation, "cut at a minimum to avoid jitter in the phase",
        # is obsolete for that reason.
        #
        # When the sector's effective frequency is available the search steps in phase rather than
        # in relative z, so that its resolution does not depend on how wide the search window is
        # (review §10.2). The window is unchanged.
        payload = find_phase_extremum(
            sol.sol,
            start_z=stop_search_window_z_begin,
            stop_z=stop_search_window_z_end,
            value_index=VALUE_INDEX,
            deriv_index=DERIV_INDEX,
            omega_sq=(
                (lambda z: omega_sq(model, k_float, z))
                if omega_sq is not None
                else None
            ),
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

    # test whether the sample grid the caller supplied is fine enough to resolve the oscillations
    # we stepped through. This used to be done inside the ODE right-hand side, at a cost of 45 % of
    # the run (review §10.2); it is now done once, on the returned samples, which is the grid the
    # flag is actually about (review §13.1). See scan_sample_grid_for_unresolved_osc for the change
    # in what is sampled.
    if omega_sq is not None:
        osc_diagnostic = scan_sample_grid_for_unresolved_osc(
            model,
            k_wavenumber,
            k_float,
            omega_sq,
            sampled_z,
            object_label,
        )
    else:
        # no effective frequency supplied: the diagnostic was not requested, and the flag fields
        # are reported as unknown, exactly as they were when delta_logz was omitted
        osc_diagnostic = {
            "has_unresolved_osc": None,
            "unresolved_z": None,
            "unresolved_efolds_subh": None,
        }

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
        "has_unresolved_osc": osc_diagnostic["has_unresolved_osc"],
        "unresolved_z": osc_diagnostic["unresolved_z"],
        "unresolved_efolds_subh": osc_diagnostic["unresolved_efolds_subh"],
        "stop_deltaz_subh": stop_deltaz_subh,
        "stop_value": stop_value,
        "stop_deriv": stop_deriv,
    }
