import time
from typing import Optional, List, Tuple

import ray
from math import log, sqrt, fabs, cos, sin, atan2, fmod, floor, pi
from scipy.integrate import solve_ivp

from ComputeTargets.BackgroundModel import BackgroundModel, ModelProxy
from ComputeTargets.WKB_tensor_Green import WKB_omegaEff_sq, WKB_d_ln_omegaEffPrime_dz
from ComputeTargets.analytic_Gk import (
    compute_analytic_G,
    compute_analytic_Gprime,
)
from ComputeTargets.integration_metadata import IntegrationSolver, IntegrationData
from ComputeTargets.integration_supervisor import (
    RHS_timer,
    IntegrationSupervisor,
    DEFAULT_UPDATE_INTERVAL,
)
from CosmologyConcepts import wavenumber_exit_time, redshift, redshift_array, wavenumber
from Datastore import DatastoreObject
from MetadataConcepts import tolerance, store_tag
from Units import check_units
from defaults import (
    DEFAULT_FLOAT_PRECISION,
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
)
from utilities import format_time

THETA_INDEX = 0
Q_INDEX = 0
EXPECTED_SOL_LENGTH = 1

# how large to we allow the WKB phase theta to become, before we terminate the integration and reset to a small value?
# we need to resolve the phase on the scale of (0, 2pi), otherwise we will compute cos(theta), sin(theta) and hence G_WKB incorrectly
DEFAULT_PHASE_RUN_LENGTH = 1e4

# how large do we allow omega_WKB_sq to get before switching to a "stage #2" integration?
DEFAULT_OMEGA_WKB_SQ_MAX = 1e6

_two_pi = 2.0 * pi


class GkWKBThetaSupervisor(IntegrationSupervisor):
    def __init__(
        self,
        k: wavenumber,
        z_init: float,
        z_target: float,
        notify_interval: int = DEFAULT_UPDATE_INTERVAL,
    ):
        super().__init__(notify_interval)

        self._k: wavenumber = k
        self._z_init: float = z_init
        self._z_target: float = z_target

        self._z_range: float = self._z_init - self._z_target

        self._last_z: float = self._z_init

        self._WKB_violation: bool = False
        self._WKB_violation_z: Optional[float] = None
        self._WKB_violation_efolds_subh: Optional[float] = None

        self._nfev: int = 0

    def __enter__(self):
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def message(self, current_z, msg):
        current_time = time.time()
        since_last_notify = current_time - self._last_notify
        since_start = current_time - self._start_time

        update_number = self.report_notify()

        z_complete = self._z_init - current_z
        z_remain = self._z_range - z_complete
        percent_remain = z_remain / self._z_range
        print(
            f"** STATUS UPDATE #{update_number}: Stage #1 integration for WKB theta_k(z) for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(
            f"|    current z={current_z:.5g} (initial z={self._z_init:.5g}, target z={self._z_target:.5g}, z complete={z_complete:.5g}, z remain={z_remain:.5g}, {percent_remain:.3%} remains)"
        )
        if self._last_z is not None:
            z_delta = self._last_z - current_z
            print(f"|    redshift advance since last update: Delta z = {z_delta:.5g}")
        print(
            f"|    {self.RHS_evaluations} RHS evaluations, mean {self.mean_RHS_time:.5g}s per evaluation, min RHS time = {self.min_RHS_time:.5g}s, max RHS time = {self.max_RHS_time:.5g}s"
        )
        print(f"|    {msg}")

        self._last_z = current_z

    def report_WKB_violation(self, z: float, efolds_subh: float):
        if self._WKB_violation:
            return

        print(
            f"!! WARNING: WKB integration for Gr_k(z, z') for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) may have violated the validity criterion for the WKB approximation"
        )
        print(f"|    current z={z:.5g}, e-folds inside horizon={efolds_subh:.3g}")
        self._WKB_violation = True
        self._WKB_violation_z = z
        self._WKB_violation_efolds_subh = efolds_subh

    @property
    def has_WKB_violation(self) -> bool:
        return self._WKB_violation

    @property
    def WKB_violation_z(self) -> float:
        return self._WKB_violation_z

    @property
    def WKB_violation_efolds_subh(self) -> float:
        return self._WKB_violation_efolds_subh

    def notify_new_nfev(self, nfev: int):
        self._nfev += nfev

    @property
    def nfev(self) -> int:
        return self._nfev

    @property
    def z_init(self) -> float:
        return self._z_init

    @property
    def z_target(self) -> float:
        return self._z_target


class GkWKBQSupervisor(IntegrationSupervisor):
    def __init__(
        self,
        k: wavenumber,
        u_init: float,
        u_target: float,
        notify_interval: int = DEFAULT_UPDATE_INTERVAL,
    ):
        super().__init__(notify_interval)

        self._k: wavenumber = k
        self._u_init: float = u_init
        self._u_target: float = u_target

        self._u_range: float = self._u_target - self._u_init

        self._last_u: float = self._u_init

        self._largest_Q: Optional[float] = 0
        self._smallest_Q: Optional[float] = 0

        self._WKB_violation: bool = False
        self._WKB_violation_z: Optional[float] = None
        self._WKB_violation_efolds_subh: Optional[float] = None

    def __enter__(self):
        super().__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)

    def message(self, current_u, msg):
        current_time = time.time()
        since_last_notify = current_time - self._last_notify
        since_start = current_time - self._start_time

        update_number = self.report_notify()

        u_complete = self._u_target - current_u
        u_remain = self._u_range - u_complete
        percent_remain = u_remain / self._u_range
        print(
            f"** STATUS UPDATE #{update_number}: Stage #2 integration for WKB Q_k(z) for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(
            f"|    current u={current_u:.5g} (initial u={self._u_init:.5g}, target z={self._u_target:.5g}, z complete={u_complete:.5g}, z remain={u_remain:.5g}, {percent_remain:.3%} remains)"
        )
        if self._last_u is not None:
            u_delta = current_u - self._last_u
            print(f"|    u advance since last update: Delta u = {u_delta:.5g}")
        print(
            f"|    {self.RHS_evaluations} RHS evaluations, mean {self.mean_RHS_time:.5g}s per evaluation, min RHS time = {self.min_RHS_time:.5g}s, max RHS time = {self.max_RHS_time:.5g}s"
        )
        print(f"|    {msg}")

        self._last_u = current_u

    def update_Q(self, Q: float):
        if self._largest_Q is None or Q > self._largest_Q:
            self._largest_Q = Q

        if self._smallest_Q is None or Q < self._smallest_Q:
            self._smallest_Q = Q

    def report_WKB_violation(self, z: float, efolds_subh: float):
        if self._WKB_violation:
            return

        print(
            f"!! WARNING: WKB integration for Gr_k(z, z') for k = {self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}) may have violated the validity criterion for the WKB approximation"
        )
        print(f"|    current z={z:.5g}, e-folds inside horizon={efolds_subh:.3g}")
        self._WKB_violation = True
        self._WKB_violation_z = z
        self._WKB_violation_efolds_subh = efolds_subh

    @property
    def has_WKB_violation(self) -> bool:
        return self._WKB_violation

    @property
    def WKB_violation_z(self) -> Optional[float]:
        return self._WKB_violation_z

    @property
    def WKB_violation_efolds_subh(self) -> Optional[float]:
        return self._WKB_violation_efolds_subh

    @property
    def largest_Q(self) -> Optional[float]:
        return self._largest_Q

    @property
    def smallest_Q(self) -> Optional[float]:
        return self._smallest_Q


def _mod_2pi(theta):
    theta_mod_2pi = fmod(theta, _two_pi)
    theta_div_2pi = int(floor(fabs(theta) / _two_pi))

    if theta < 0.0:
        theta_div_2pi = -theta_div_2pi

    if theta_mod_2pi > 0:
        theta_mod_2pi = theta_mod_2pi - _two_pi

    return theta_div_2pi, theta_mod_2pi


def stage_1_evolution(
    model: BackgroundModel,
    k: wavenumber_exit_time,
    z_init: float,
    z_target: float,
    z_sample_list: List[float],
    z_break_list: List[float],
    sampled_z: List[float],
    sampled_theta_mod_2pi: List[float],
    sampled_theta_div_2pi: List[float],
    current_div_2pi_offset: int,
    atol: float,
    rtol: float,
) -> dict:
    k_wavenumber: wavenumber = k.k
    k_float = k_wavenumber.k

    def phase_cycle_trigger(z, state, _) -> float:
        # terminate when the phase has accumulated to smaller than -1E3
        theta = state[THETA_INDEX]
        return theta + DEFAULT_PHASE_RUN_LENGTH

    phase_cycle_trigger.terminal = True

    def terminate_trigger(z, float, _) -> float:
        # terminate when omega_WKB_sq exceeds 1E6 (at that point we want to switch to a different numerical scheme)
        omega_WKB_sq = WKB_omegaEff_sq(model, k_float, z)

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

            omega_WKB_sq = WKB_omegaEff_sq(model, k_float, z)

            if omega_WKB_sq < 0.0:
                raise ValueError(
                    f"compute_Gk_WKB: omega_WKB^2 cannot be negative during WKB integration (omega_WKB^2={omega_WKB_sq:.5g})"
                )

            d_ln_omega_WKB_dz = WKB_d_ln_omegaEffPrime_dz(model, k_float, z)
            WKB_criterion = fabs(d_ln_omega_WKB_dz) / sqrt(fabs(omega_WKB_sq))
            if WKB_criterion > 1.0:
                H = model.functions.Hubble(z)
                k_over_H = k_float / H
                supervisor.report_WKB_violation(z, log((1.0 + z) * k_over_H))

            omega_WKB = sqrt(omega_WKB_sq)
            dtheta_dz = omega_WKB

            return [dtheta_dz]

    with GkWKBThetaSupervisor(k_wavenumber, z_init, z_target) as supervisor:
        state = [0.0]
        t_span = (z_init, z_target)

        z_terminate = None
        theta_terminate_mod_2pi = None

        while len(z_sample_list) > 0:
            sol = solve_ivp(
                RHS,
                method="RK45",
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
                    f'compute_Gk_WKB: stage #1 integration did not terminate successfully (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g}, error at z={sol.t[-1]:.5g}, "{sol.message}")'
                )

            supervisor.notify_new_nfev(int(sol.nfev))
            batch_z = sol.t
            batch_values = sol.y

            # check if integration stopped due to a termination event.
            # If not, we should have reached the end of the sample list
            if sol.status != 1:
                if len(batch_z) != len(z_sample_list):
                    raise RuntimeError(
                        f"compute_Gk_WKB: stage #1 integration reached end of domain, but an incorrect number of z-samples were recorded (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g}, expected={len(z_sample_list)}, found={len(batch_z)})"
                    )
            else:
                if len(batch_z) >= len(z_sample_list):
                    raise RuntimeError(
                        f"compute_Gk_WKB: stage #1 integration terminated at a phase cutoff, but no z-sample points are left (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g}, max expected={len(z_sample_list)}, found={len(batch_z)})"
                    )

            # check we have expected number of components (just one!) in the solution
            if len(batch_z) > 0:
                if len(batch_values) != EXPECTED_SOL_LENGTH:
                    print(
                        f"!! compute_Gk_WKB: stage #1 solution does not have expected number of members for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc"
                    )
                    print(
                        f"   -- expected {EXPECTED_SOL_LENGTH} members, found {len(batch_values)}"
                    )
                    print(
                        f"      z_init={supervisor.z_init:.5g}, z_target={supervisor.z_target:.5g}"
                    )
                    print(f"      sol.success={sol.success}, sol.message={sol.message}")
                    raise RuntimeError(
                        f"compute_Gk_WKB: stage #1 solution does not have expected number of members (expected {EXPECTED_SOL_LENGTH}, found {len(batch_values)}; k={k_wavenumber.k_inv_Mpc}/Mpc, length of sol.t={len(batch_z)})"
                    )

                batch_theta = batch_values[THETA_INDEX]

                # walk through sampled values, computing theta div 2pi and theta mod 2pi
                for z, theta in zip(batch_z, batch_theta):
                    theta_div_2pi, theta_mod_2pi = _mod_2pi(theta)

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
                            f"compute_Gk_WKB: could not find event record to restart phase integration (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g})"
                        )
                    new_theta_init = values[0]
                    new_theta_init_div_2pi, new_theta_init_mod_2pi = _mod_2pi(
                        new_theta_init
                    )

                    state = [new_theta_init_mod_2pi]
                    current_div_2pi_offset += new_theta_init_div_2pi

                elif len(terminate_times) == 1:
                    z_terminate = terminate_times[0]
                    values = sol.y_events[1]
                    if len(values) != 1:
                        raise RuntimeError(
                            f"compute_Gk_WKB: could not find event record to terminate stage 1 integration (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g})"
                        )
                    theta_terminate = values[0]
                    theta_terminate_div_2pi, theta_terminate_mod_2pi = _mod_2pi(
                        theta_terminate
                    )

                    current_div_2pi_offset += theta_terminate_div_2pi

                    break

                else:
                    raise RuntimeError(
                        f"compute_Gk_WKB: could not find event record to restart phase integration (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g})"
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
    sampled_z: List[float],
    sampled_theta_mod_2pi: List[float],
    sampled_theta_div_2pi: List[float],
    theta_mod_2pi_init: float,
    current_div_2pi_offset: int,
    atol: float,
    rtol: float,
) -> dict:
    k_wavenumber: wavenumber = k.k
    k_float = k_wavenumber.k

    omega_WKB_sq_init = WKB_omegaEff_sq(model, k_float, z_init)
    omega_WKB_init = sqrt(omega_WKB_sq_init)

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

            omega_WKB_sq = WKB_omegaEff_sq(model, k_float, z)

            if omega_WKB_sq < 0.0:
                raise ValueError(
                    f"compute_Gk_WKB: omega_WKB^2 cannot be negative during WKB integration (omega_WKB^2={omega_WKB_sq:.5g})"
                )

            d_ln_omega_WKB_dz = WKB_d_ln_omegaEffPrime_dz(model, k_float, z)
            WKB_criterion = fabs(d_ln_omega_WKB_dz) / sqrt(fabs(omega_WKB_sq))
            if WKB_criterion > 1.0:
                H = model.functions.Hubble(z)
                k_over_H = k_float / H
                supervisor.report_WKB_violation(z, log((1.0 + z) * k_over_H))

            omega_WKB = sqrt(omega_WKB_sq)
            one_plus_u = 1.0 + u
            dQ_du = -omega_WKB / omega_WKB_init / one_plus_u - Q / one_plus_u

            return [dQ_du]

    u_init = 0
    u_target = z_init - z_target
    with GkWKBQSupervisor(k_wavenumber, u_init, u_target) as supervisor:
        # initial condition is Q = 0 at u = 0
        # then we expect Q close to unity for u >> 1
        state = [0.0]
        u_sample_list = [z_init - z for z in z_sample_list]

        sol = solve_ivp(
            RHS,
            method="RK45",
            t_span=(u_init, u_target),
            y0=state,
            t_eval=u_sample_list,
            atol=atol,
            rtol=rtol,
            args=(supervisor,),
        )

        if not sol.success:
            raise RuntimeError(
                f'compute_Gk_WKB: stage #2 integration did not terminate successfully (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g}, error at u={sol.t[-1]:.5g}, "{sol.message}")'
            )

        batch_u = sol.t
        batch_values = sol.y

        if sol.status != 1:
            if len(batch_u) != len(u_sample_list):
                raise RuntimeError(
                    f"compute_Gk_WKB: stage #2 integration reached end of domain, but an incorrect number of z-samples were recorded (k={k_wavenumber.k_inv_Mpc:.5g}/Mpc, z_init={z_init:.5g}, expected={len(z_sample_list)}, found={len(batch_u)})"
                )

        if len(batch_values) != EXPECTED_SOL_LENGTH:
            print(
                f"!! compute_Gk_WKB: stage #2 solution does not have expected number of members for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc"
            )
            print(
                f"   -- expected {EXPECTED_SOL_LENGTH} members, found {len(batch_values)}"
            )
            print(f"      z_init={z_init:.5g}, z_target={z_target:.5g}")
            print(f"      sol.success={sol.success}, sol.message={sol.message}")
            raise RuntimeError(
                f"compute_Gk_WKB: stage #2 solution does not have expected number of members (expected {EXPECTED_SOL_LENGTH}, found {len(batch_values)}; k={k_wavenumber.k_inv_Mpc}/Mpc, length of sol.t={len(batch_u)})"
            )

        batch_Q = batch_values[Q_INDEX]

        # walk through the sampled values returned from solve_ivp, computing theta div 2pi and theta mod 2pi at each step
        for u, Q in zip(batch_u, batch_Q):
            theta = omega_WKB_init * (1.0 + u) * Q + theta_mod_2pi_init
            theta_div_2pi, theta_mod_2pi = _mod_2pi(theta)

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


@ray.remote
def compute_Gk_WKB(
    model_proxy: ModelProxy,
    k: wavenumber_exit_time,
    z_init: float,
    z_sample: redshift_array,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
) -> dict:
    k_wavenumber: wavenumber = k.k
    k_float = float(k_wavenumber.k)

    check_units(k_wavenumber, model_proxy)

    model: BackgroundModel = model_proxy.get()

    # need to keep theta integration in bounds, otherwise we lose accuracy in the WKB solution.
    # no simple way to do this. See https://github.com/scipy/scipy/issues/19645.
    # the solution used here is to compute the phase mod 2pi, i.e., we try to track theta mod 2pi
    # and theta div 2pi separately. We do that by postprocessing the output of the integration.
    # To keep accuracy, we keep cutting the integral after we've gone about 1E3 in redshift.
    # This should generate something accurate enough for our purposes.
    z_sample_list = z_sample.as_float_list()
    z_break_list = []
    z_target = z_sample.min.z

    sampled_z = []
    sampled_theta_mod_2pi = []
    sampled_theta_div_2pi = []

    current_div_2pi_offset = 0

    d_ln_omega_WKB_dz_init = WKB_d_ln_omegaEffPrime_dz(model, k_float, z_init)
    omega_WKB_sq_init = WKB_omegaEff_sq(model, k_float, z_init)

    if omega_WKB_sq_init < 0.0:
        raise RuntimeError(
            f"compute_Gk_WKB: omega_WKB^2 is negative at the initial time for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc (z_init={z_init:.5g}, omega_WKB^2={omega_WKB_sq_init:.5g})"
        )

    WKB_criterion_init = fabs(d_ln_omega_WKB_dz_init) / sqrt(fabs(omega_WKB_sq_init))
    if WKB_criterion_init > 1.0:
        raise RuntimeError(
            f"compute_Gk_WKB: WKB criterion |d(omega_WKB)/omega_WKB^2| > 1 at the initial time  for k={k_wavenumber.k_inv_Mpc:.5g}/Mpc (z_init={z_init:.5g}, WKB criterion={WKB_criterion_init:.5g})"
        )

    metadata = {}

    if fabs(z_init - z_target) < atol:
        z_sample_list.pop(0)
        assert len(z_sample_list) == 0

        sampled_z.append(z_init)
        sampled_theta_mod_2pi.append(0)
        sampled_theta_div_2pi.append(0)

        metadata["initial_data_only"] = True

    # if omega_WKB < 1E3, run a "stage 1" integration.
    # This could exhaust the possible sample points, or it might terminate so that we can start a "stage 2" integration
    stage_1_data = None
    if len(z_sample_list) > 0 and omega_WKB_sq_init < DEFAULT_OMEGA_WKB_SQ_MAX:
        stage_1_data = stage_1_evolution(
            model,
            k,
            z_init,
            z_target,
            z_sample_list,
            z_break_list,
            sampled_z,
            sampled_theta_mod_2pi,
            sampled_theta_div_2pi,
            current_div_2pi_offset,
            atol,
            rtol,
        )

        metadata["phase_cycle_events"] = len(z_break_list)

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
            sampled_z,
            sampled_theta_mod_2pi,
            sampled_theta_div_2pi,
            theta_stage2_init_mod_2pi,
            current_div_2pi_offset,
            atol,
            rtol,
        )

        metadata["stage_2_largest_Q"] = stage_2_data["largest_Q"]
        metadata["stage_2_smallest_Q"] = stage_2_data["smallest_Q"]

    if (
        stage_1_data is None
        and stage_2_data is None
        and not metadata.get("initial_data_only", False)
    ):
        raise RuntimeError("compute_Gk_WKB: both stage #1 and stage #2 data are empty")

    returned_values = len(sampled_z)
    expected_values = len(z_sample)

    if returned_values != expected_values:
        raise RuntimeError(
            f"compute_Gk_WKB: stage #1 and stage #2 evolution produced {returned_values} samples, but expected {expected_values}"
        )

    # validate that the samples of the solution correspond to the z-sample points that we specified.
    # This really should be true (and has also been checked in the stage #1 and stage #2 integrations
    # separately), but there is no harm in being defensive.
    for i in range(returned_values):
        diff = sampled_z[i] - z_sample[i].z
        if fabs(diff) > DEFAULT_ABS_TOLERANCE:
            raise RuntimeError(
                f"compute_Gk_WKB: stage #1 and stage #2 evolution returned sample points that differ from those requested (difference={diff} at i={i})"
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
            f'compute_Gk_WKB: both stage #1 and stage #2 fields are empty for payload "{field}"'
        )

    return {
        "stage_1_data": stage_1_data["data"] if stage_1_data is not None else None,
        "stage_2_data": stage_2_data["data"] if stage_2_data is not None else None,
        "theta_div_2pi_sample": sampled_theta_div_2pi,
        "theta_mod_2pi_sample": sampled_theta_mod_2pi,
        "solver_label": "solve_ivp+RK45-stepping0",
        "has_WKB_violation": merge_stage_data(
            "has_WKB_violation", lambda a, b: a or b, allow_none=True
        ),
        "WKB_violation_z": merge_stage_data(
            "WKB_violation_z", lambda a, b: a, allow_none=True
        ),
        "WKB_violation_efolds_subh": merge_stage_data(
            "WKB_violation_efolds_subh", lambda a, b: a, allow_none=True
        ),
        "metadata": metadata,
    }


class GkWKBIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced for a calculation of the WKB
    phase associated with the tensor Green's function
    """

    def __init__(
        self,
        payload,
        solver_labels: dict,
        model: ModelProxy,
        k: wavenumber_exit_time,
        atol: tolerance,
        rtol: tolerance,
        z_source: Optional[redshift] = None,
        z_init: Optional[float] = None,
        G_init: Optional[float] = 0.0,
        Gprime_init: Optional[float] = 1.0,
        z_sample: Optional[redshift_array] = None,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        k_wavenumber: wavenumber = k.k
        check_units(k_wavenumber, model)

        self._solver_labels = solver_labels
        self._z_sample = z_sample

        self._z_init = z_init
        self._G_init = G_init
        self._Gprime_init = Gprime_init

        if payload is None:
            DatastoreObject.__init__(self, None)

            self._stage_1 = IntegrationData(
                compute_time=None,
                compute_steps=None,
                RHS_evaluations=None,
                mean_RHS_time=None,
                max_RHS_time=None,
                min_RHS_time=None,
            )
            self._stage_2 = IntegrationData(
                compute_time=None,
                compute_steps=None,
                RHS_evaluations=None,
                mean_RHS_time=None,
                max_RHS_time=None,
                min_RHS_time=None,
            )

            self._has_WKB_violation = None
            self._WKB_violation_z = None
            self._WKB_violation_efolds_subh = None

            self._init_efolds_suph = None
            self._metadata = None

            self._solver = None

            self._sin_coeff = None
            self._cos_coeff = None
            self._values = None

        else:
            DatastoreObject.__init__(self, payload["store_id"])

            self._stage_1 = payload["stage_1_data"]
            self._stage_2 = payload["stage_2_data"]

            self._has_WKB_violation = payload["has_WKB_violation"]
            self._WKB_violation_z = payload["WKB_violation_z"]
            self._WKB_violation_efolds_subh = payload["WKB_violation_efolds_subh"]

            self._init_efolds_subh = payload["init_efolds_subh"]
            self._metadata = payload["metadata"]

            self._solver = payload["solver"]

            self._sin_coeff = payload["sin_coeff"]
            self._cos_coeff = payload["cos_coeff"]
            self._values = payload["values"]

        if z_sample is not None:
            z_limit = k.z_exit_subh_e3
            z_initial_float = z_init if z_init is not None else z_source.z
            for z in z_sample:
                # check that each response redshift is not too close to the horizon, or outside it, for this k-mode
                z_float = float(z)
                if z_float > z_limit - DEFAULT_FLOAT_PRECISION:
                    raise ValueError(
                        f"Specified response redshift z={z_float:.5g} is closer than 3-folds to horizon re-entry for wavenumber k={k_wavenumber:.5g}/Mpc"
                    )

                # also, check that each response redshift is later than the specified source redshift
                if z_float > z_initial_float:
                    raise ValueError(
                        f"Redshift sample point z={z_float:.5g} exceeds initial redshift z={z_initial_float:.5g}"
                    )

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._model_proxy = model

        self._k_exit = k
        self._z_source = z_source

        self._compute_ref = None

        self._atol = atol
        self._rtol = rtol

    @property
    def model_proxy(self) -> ModelProxy:
        return self._model_proxy

    @property
    def k(self) -> wavenumber:
        return self._k_exit.k

    @property
    def z_exit(self) -> float:
        return self._k_exit.z_exit

    @property
    def z_init(self) -> float:
        if self._z_init is not None:
            return self._z_init

        return self._z_source.z

    @property
    def G_init(self) -> Optional[float]:
        return self._G_init

    @property
    def Gprime_init(self) -> Optional[float]:
        return self._Gprime_init

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    @property
    def z_source(self):
        return self._z_source

    @property
    def z_sample(self):
        return self._z_sample

    @property
    def stage_1_data(self) -> IntegrationData:
        if self._values is None:
            raise RuntimeError("values have not yet been populated")

        return self._stage_1

    @property
    def stage_2_data(self) -> IntegrationData:
        if self._values is None:
            raise RuntimeError("values have not yet been populated")

        return self._stage_2

    @property
    def metadata(self) -> dict:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._metadata

    @property
    def has_WKB_violation(self) -> bool:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._has_WKB_violation

    @property
    def WKB_violation_z(self) -> Optional[float]:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        if not self._has_WKB_violation:
            return None

        return self._WKB_violation_z

    @property
    def WKB_violation_efolds_subh(self) -> Optional[float]:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        if not self._has_WKB_violation:
            return None

        return self._WKB_violation_efolds_subh

    @property
    def init_efolds_subh(self) -> float:
        if self._init_efolds_subh is None:
            raise RuntimeError("init_efolds_subh has not yet been populated")

        return self._init_efolds_subh

    @property
    def sin_coeff(self) -> float:
        if self._sin_coeff is None:
            raise RuntimeError("sin_coeff has not yet been populated")

        return self._sin_coeff

    @property
    def cos_coeff(self) -> float:
        if self._cos_coeff is None:
            raise RuntimeError("cos_coeff has not yet been populated")

        return self._cos_coeff

    @property
    def solver(self) -> IntegrationSolver:
        if self._solver is None:
            raise RuntimeError("solver has not yet been populated")
        return self._solver

    @property
    def values(self) -> List:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "GkWKBIntegration: values read but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self, label: Optional[str] = None):
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "GkWKBIntegration: compute() called but _do_not_populate is set"
            )

        if self._values is not None:
            raise RuntimeError("values have already been computed")

        if self._z_source is None or self._z_sample is None:
            raise RuntimeError(
                "Object has not been configured correctly for a concrete calculation (z_source or z_sample is missing). It can only represent a query."
            )

        # replace label if specified
        if label is not None:
            self._label = label

        model: BackgroundModel = self._model_proxy.get()

        initial_z = self._z_init if self._z_init is not None else self._z_source.z

        omega_WKB_sq_init = WKB_omegaEff_sq(model, self._k_exit.k.k, initial_z)
        d_ln_omega_WKB_init = WKB_d_ln_omegaEffPrime_dz(
            model, self._k_exit.k.k, initial_z
        )

        if omega_WKB_sq_init < 0.0:
            raise ValueError(
                f"omega_WKB^2 must be non-negative at the initial time (k={self._k_exit.k.k_inv_Mpc}/Mpc, z_init={initial_z:.5g}, omega_WKB^2={omega_WKB_sq_init:.5g})"
            )

        WKB_criterion_init = d_ln_omega_WKB_init / sqrt(omega_WKB_sq_init)
        if WKB_criterion_init > 1.0:
            print(f"!! Warning (GkWKBIntegration) k={self._k_exit.k.k_inv_Mpc:.5g}/Mpc")
            print(
                f"|    WKB diagnostic |d(omega) / omega^2| exceeds unity at initial time (value={WKB_criterion_init:.5g}, z_init={initial_z:.5g})"
            )
            print(f"     This may lead to meaningless results.")

        self._compute_ref = compute_Gk_WKB.remote(
            self._model_proxy,
            self._k_exit,
            initial_z,
            self._z_sample,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "GkWKBIntegration: store() called, but no compute() is in progress"
            )

        # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        # if not, return None
        if len(resolved) == 0:
            return None

        # retrieve result and populate ourselves
        data = ray.get(self._compute_ref)
        self._compute_ref = None

        self._stage_1 = data["stage_1_data"]
        self._stage_2 = data["stage_2_data"]

        self._has_WKB_violation = data["has_WKB_violation"]
        self._WKB_violation_z = data["WKB_violation_z"]
        self._WKB_violation_efolds_subh = data["WKB_violation_efolds_subh"]

        self._metadata = data["metadata"]

        model: BackgroundModel = self._model_proxy.get()

        initial_z = self._z_init if self._z_init is not None else self._z_source.z

        H_init = model.functions.Hubble(initial_z)
        eps_init = model.functions.epsilon(initial_z)

        one_plus_z_init = 1.0 + initial_z
        k_over_aH = one_plus_z_init * self.k.k / H_init
        self._init_efolds_subh = log(k_over_aH)

        # can assume here that omega_WKB_sq_init > 0 and the WKB criterion < 1.
        # This has been checked in compute()
        omega_WKB_sq_init = WKB_omegaEff_sq(model, self._k_exit.k.k, initial_z)
        d_ln_omega_WKB_init = WKB_d_ln_omegaEffPrime_dz(
            model, self._k_exit.k.k, initial_z
        )

        # we try to adjust the phase so that the solution is of the form G = amplitude * sin (theta + DeltaTheta)
        # the point is that WKB solutions defined inside the horizon always have the coefficient of cos equal to zero,
        # because the initial conditions are G = 0, G' = 0, and this forces the cos to be absent.
        # When we try to smoothly match the outside-the-horizon source times to these inside-the-horizon ones,
        # we do not want a discontinuity in the coefficients. So we have to keep the coefficient of the cos always zero.
        # This makes a corresponding adjustment to the phase, which we must calculate.
        omega_WKB_init = sqrt(omega_WKB_sq_init)
        sqrt_omega_WKB_init = sqrt(omega_WKB_init)

        num = sqrt_omega_WKB_init * self._G_init
        den = (
            self._Gprime_init
            + (self._G_init / 2.0) * (d_ln_omega_WKB_init + eps_init / one_plus_z_init)
        ) / sqrt_omega_WKB_init

        deltaTheta = atan2(num, den)
        alpha = sqrt(num * num + den * den)

        sin_deltaTheta = sin(deltaTheta)
        sgn_sin_deltaTheta = +1 if sin_deltaTheta >= 0.0 else -1
        sgn_G = +1 if self._G_init >= 0.0 else -1

        self._cos_coeff = 0.0
        self._sin_coeff = sgn_sin_deltaTheta * sgn_G * alpha

        # estimate tau at the source redshift
        H_source = model.functions.Hubble(self._z_source.z)
        tau_source = model.functions.tau(self._z_source.z)

        theta_div_2pi_sample = data["theta_div_2pi_sample"]
        theta_mod_2pi_sample = data["theta_mod_2pi_sample"]

        def wrap_theta(theta: float) -> Tuple[int, float]:
            if theta > 0.0:
                return +1, theta - _two_pi
            if theta <= -_two_pi:
                return -1, theta + _two_pi

            return 0, theta

        theta_sample_shifts = [
            wrap_theta(theta_mod_2pi_sample + deltaTheta)
            for theta_mod_2pi_sample in theta_mod_2pi_sample
        ]
        theta_div_2pi_shift, theta_mod_2pi_sample = zip(*theta_sample_shifts)

        theta_div_2pi_shift_base = theta_div_2pi_shift[0]
        theta_div_2pi_sample = [
            d + shift - theta_div_2pi_shift_base
            for (d, shift) in zip(theta_div_2pi_sample, theta_div_2pi_shift)
        ]

        self._values = []
        for i in range(len(theta_mod_2pi_sample)):
            current_z = self._z_sample[i]
            current_z_float = current_z.z
            H = model.functions.Hubble(current_z_float)
            tau = model.functions.tau(current_z_float)

            analytic_G = compute_analytic_G(
                self.k.k, 1.0 / 3.0, tau_source, tau, H_source
            )
            analytic_Gprime = compute_analytic_Gprime(
                self.k.k, 1.0 / 3.0, tau_source, tau, H_source, H
            )

            # should be safe to assume omega_WKB_sq > 0, since otherwise this would have been picked up during the integration
            omega_WKB_sq = WKB_omegaEff_sq(model, self._k_exit.k.k, current_z_float)
            omega_WKB = sqrt(omega_WKB_sq)

            d_ln_omega_WKB = WKB_d_ln_omegaEffPrime_dz(
                model, self._k_exit.k.k, current_z_float
            )
            WKB_criterion = fabs(d_ln_omega_WKB) / omega_WKB

            H_ratio = H_init / H
            norm_factor = sqrt(H_ratio / omega_WKB)

            # no need to include theta div 2pi in the calculation of G_WKB
            # (and indeed it may be more accurate if we don't)
            G_WKB = norm_factor * (
                self._cos_coeff * cos(theta_mod_2pi_sample[i])
                + self._sin_coeff * sin(theta_mod_2pi_sample[i])
            )

            # create new GkWKBValue object
            self._values.append(
                GkWKBValue(
                    None,
                    current_z,
                    H_ratio,
                    theta_mod_2pi_sample[i],
                    theta_div_2pi_sample[i],
                    omega_WKB_sq=omega_WKB_sq,
                    WKB_criterion=WKB_criterion,
                    G_WKB=G_WKB,
                    analytic_G=analytic_G,
                    analytic_Gprime=analytic_Gprime,
                    sin_coeff=self._sin_coeff,
                    cos_coeff=self._cos_coeff,
                    z_init=self._z_init,
                )
            )

        self._solver = self._solver_labels[data["solver_label"]]

        return True


class GkWKBValue(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        z: redshift,
        H_ratio: float,
        theta_mod_2pi: float,
        theta_div_2pi: int,
        omega_WKB_sq: Optional[float] = None,
        WKB_criterion: Optional[float] = None,
        G_WKB: Optional[float] = None,
        sin_coeff: Optional[float] = None,
        cos_coeff: Optional[float] = None,
        z_init: Optional[float] = None,
        analytic_G: Optional[float] = None,
        analytic_Gprime: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z
        self._H_ratio = H_ratio

        self._theta_mod_2pi = theta_mod_2pi
        self._theta_div_2pi = theta_div_2pi
        self._omega_WKB_sq = omega_WKB_sq
        self._WKB_criterion = WKB_criterion
        self._G_WKB = G_WKB

        self._analytic_G = analytic_G
        self._analytic_Gprime = analytic_Gprime

        self._sin_coeff = sin_coeff
        self._cos_coeff = cos_coeff

        self._z_init = z_init

    def __float__(self):
        """
        Cast to float. Returns value of G_k estimated using the WKB approximation
        :return:
        """
        return self._G_WKB

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def H_ratio(self) -> float:
        return self._H_ratio

    @property
    def theta_mod_2pi(self) -> float:
        return self._theta_mod_2pi

    @property
    def theta_div_2pi(self) -> int:
        return self._theta_div_2pi

    @property
    def theta(self) -> int:
        return self._theta_div_2pi * _two_pi + self._theta_mod_2pi

    @property
    def omega_WKB_sq(self) -> Optional[float]:
        return self._omega_WKB_sq

    @property
    def WKB_criterion(self) -> Optional[float]:
        return self._WKB_criterion

    @property
    def G_WKB(self) -> Optional[float]:
        return self._G_WKB

    @property
    def analytic_G(self) -> Optional[float]:
        return self._analytic_G

    @property
    def analytic_Gprime(self) -> Optional[float]:
        return self._analytic_Gprime

    @property
    def sin_coeff(self) -> Optional[float]:
        return self._sin_coeff

    @property
    def cos_coeff(self) -> Optional[float]:
        return self._cos_coeff

    @property
    def z_init(self) -> Optional[float]:
        return self._z_init
