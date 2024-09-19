import time
from math import log
from traceback import print_tb
from typing import Mapping

import ray
from scipy.integrate import solve_ivp

from CosmologyConcepts import wavenumber
from CosmologyModels import BaseCosmology
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE


class WallclockTimer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time

        if exc_type is not None:
            print(f"type={exc_type}, value={exc_val}")
            print_tb(exc_tb)


def check_units(A, B):
    """
    Check that objects A and B are defined with the same units.
    Assumes they both provide a .units property that returns a UnitsLike object
    :param A:
    :param B:
    :return:
    """
    if A.units != B.units:
        raise RuntimeError("Units used for wavenumber k and cosmology are not equal")


@ray.remote
def find_horizon_exit_time(
    cosmology: BaseCosmology,
    k: wavenumber,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
) -> Mapping[str, float]:
    """
    Compute the redshift of horizon exit for a mode of wavenumber k in the specified cosmology, by solving
    the equation (k/H) * (1+z) = k/(aH) = 1.
    To do this we fix an initial condition for q = ln[ (k/H)*(1+z) ] today via q = ln[ k/H0 ],
    and then integrate dq/dz up to a point where it crosses zero.
    :param cosmology:
    :param k:
    :return:
    """
    check_units(k, cosmology)

    q0 = log(float(k) / cosmology.H0)

    # RHS of ODE system for dq/dz = f(z)
    def RHS(z, state):
        # q = state[0]
        return [1.0 / (1.0 + z) - cosmology.d_lnH_dz(z)]

    # build event function to terminate when q crosses zero
    def q_zero_event(z, state):
        q = state[0]
        return q

    q_zero_event.terminal = True

    with WallclockTimer() as timer:
        # solve to find the zero crossing point; we set the upper limit of integration to be 1E12, which should be comfortably above
        # the redshift of any horizon crossing in which we are interested.
        sol = solve_ivp(
            RHS,
            t_span=(0.0, 1e20),
            y0=[q0],
            events=q_zero_event,
            atol=atol,
            rtol=rtol,
        )

    # test whether termination occurred due to the q_zero_event() firing
    if not sol.success:
        raise RuntimeError(
            f'find_horizon_exit_time: integration to find horizon-crossing time did not terminate successfully ("{sol.message}")'
        )

    if sol.status != 1:
        raise RuntimeError(
            f"find_horizon_exit_time: integration to find horizon-crossing time did not detect k/aH = 0 within the integration range"
        )

    if len(sol.t_events) != 1:
        raise RuntimeError(
            f"find_horizon_exit_time: unexpected number of event types returned from horizon-crossing integration (num={len(sol.t_events)})"
        )

    event_times = sol.t_events[0]
    if len(event_times) != 1:
        raise RuntimeError(
            f"find_horizon_exit_time: more than one horizon-crossing time returned from integration (num={len(event_times)})"
        )

    return {"compute_time": timer.elapsed, "z_exit": event_times[0]}


SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 60 * SECONDS_PER_MINUTE
SECONDS_PER_DAY = 24 * SECONDS_PER_HOUR


def format_time(interval: float) -> str:
    int_interval = int(interval)
    str = ""

    if int_interval > SECONDS_PER_DAY:
        days = int_interval // SECONDS_PER_DAY
        int_interval = int_interval - days * SECONDS_PER_DAY
        interval = interval - days * SECONDS_PER_DAY
        if len(str) > 0:
            str = str + f" {days}d"
        else:
            str = f"{days}d"

    if int_interval > SECONDS_PER_HOUR:
        hours = int_interval // SECONDS_PER_HOUR
        int_interval = int_interval - hours * SECONDS_PER_HOUR
        interval = interval - hours * SECONDS_PER_HOUR
        if len(str) > 0:
            str = str + f" {hours}h"
        else:
            str = f"{hours}h"

    if int_interval > SECONDS_PER_MINUTE:
        minutes = int_interval // SECONDS_PER_MINUTE
        int_interval = int_interval - minutes * SECONDS_PER_MINUTE
        interval = interval - minutes * SECONDS_PER_MINUTE
        if len(str) > 0:
            str = str + f" {minutes}m"
        else:
            str = f"{minutes}m"

    if len(str) > 0:
        str = str + f" {interval:.3g}s"
    else:
        str = f"{interval:.3g}s"

    return str
