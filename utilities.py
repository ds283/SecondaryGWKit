import time
from math import log, fabs

from scipy.integrate import solve_ivp

from CosmologyConcepts import wavenumber
from CosmologyModels.base import CosmologyBase


class WallclockTimer:
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed


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


def find_horizon_exit_time(
    cosmology: CosmologyBase,
    k: wavenumber,
    atol: float = 1e-5,
    rtol: float = 1e-7,
) -> float:
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
    def dq_dz(z, state):
        # q = state[0]
        return [1.0 / (1.0 + z) - cosmology.d_lnH_dz(z)]

    # build event function to terminate when q crosses zero
    def q_zero_event(z, state):
        q = state[0]
        return q

    q_zero_event.terminal = True

    # solve to find the zero crossing point; we set the upper limit of integration to be 1E12, which should be comfortably above
    # the redshift of any horizon crossing in which we are interested.
    sol = solve_ivp(
        dq_dz, t_span=(0.0, 1e20), y0=[q0], events=q_zero_event, atol=atol, rtol=rtol
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

    return event_times[0]
