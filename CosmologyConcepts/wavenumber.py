from functools import partial
from typing import Iterable, Optional, Mapping

import ray
from math import log10, log, fabs
from numpy import logspace
from scipy.integrate import solve_ivp

from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE
from utilities import check_units, WallclockTimer


class wavenumber(DatastoreObject):
    def __init__(self, store_id: int, k_inv_Mpc: float, units):
        """
        Represents a wavenumber, e.g.,
        used to sample a transfer function or power spectrum
        :param store_id: unique Datastore id. Should not be None
        :param k_inv_Mpc: wavenumber, measured in 1/Mpc
        :param units: units block (e.g. Mpc-based units)
        """
        if store_id is None:
            raise ValueError("Store ID cannot be None")
        DatastoreObject.__init__(self, store_id)

        # units are available for inspection
        self.units = units

        self.k_inv_Mpc = k_inv_Mpc
        self.k = k_inv_Mpc / units.Mpc

    def __float__(self):
        """
        Cast to float. Returns dimensionful wavenumber.
        :return:
        """
        return float(self.k)


class wavenumber_array:
    def __init__(self, k_array: Iterable[wavenumber]):
        """
        Construct a datastore-backed object representing an array of wavenumber values
        :param store_id: unique Datastore id. Should not be None
        :param k_inv_Mpc_array: array of wavenumbers, measured in 1/Mpc
        :param units: units block
        """
        # store array
        self._k_array = k_array

    def __iter__(self):
        for k in self._k_array:
            yield k

    def __getitem__(self, key):
        return self._k_array[key]

    def __len__(self):
        return len(self._k_array)

    def as_list(self) -> list[float]:
        return [float(k) for k in self._k_array]


class wavenumber_exit_time(DatastoreObject):
    def __init__(
        self,
        payload,
        k: wavenumber,
        cosmology: BaseCosmology,
        atol: tolerance,
        rtol: tolerance,
    ):
        """
        Represents the horizon exit time for a mode of wavenumber k
        :param store_id: unique Datastore id. May be None if the object has not yet been fully serialized
        :param k: wavenumber object
        :param cosmology: cosmology object satisfying the CosmologyBase concept
        """
        check_units(k, cosmology)

        # store the provided z_exit value and compute_time value
        # these may be None if store_id is also None. This represents the case that the computation has not yet been done.
        # In this case, the client code needs to call compute() in order to populate the z_exit value
        if payload is None:
            DatastoreObject.__init__(self, None)
            self._z_exit = None
            self._z_exit_suph_e3 = None
            self._z_exit_suph_e5 = None
            self._z_exit_subh_e3 = None
            self._z_exit_minus_5 = None

            self._compute_time = None
            self._stepping = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._z_exit = payload["z_exit"]
            self._z_exit_suph_e3 = payload["z_exit_suph_e3"]
            self._z_exit_suph_e5 = payload["z_exit_suph_e5"]
            self._z_exit_subh_e3 = payload["z_exit_subh_e3"]
            self._z_exit_subh_e5 = payload["z_exit_subh_e5"]

            self._compute_time = payload["compute_time"]
            self._stepping = payload["stepping"]

        # store parameters
        self.k = k
        self.cosmology = cosmology

        self._compute_ref = None

        self._atol = atol
        self._rtol = rtol

    def compute(self):
        if self._z_exit is not None:
            raise RuntimeError("z_exit has already been computed")
        self._compute_ref = find_horizon_exit_time.remote(
            self.cosmology,
            self.k,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "wavenumber_exit_time: store() called, but no compute() is in progress"
            )

        # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        # if not, return None
        if len(resolved) == 0:
            return None

        # retrieve result and populate ourselves
        data = ray.get(self._compute_ref)
        self._compute_ref = None

        self._z_exit = data["z_exit"]
        self._z_exit_suph_e3 = data["z_exit_suph_e3"]
        self._z_exit_suph_e5 = data["z_exit_suph_e5"]
        self._z_exit_subh_e3 = data["z_exit_subh_e3"]
        self._z_exit_subh_e5 = data["z_exit_subh_e5"]

        self._compute_time = data["compute_time"]
        self._stepping = 0

        return True

    @property
    def z_exit(self) -> float:
        if self._z_exit is None:
            raise RuntimeError("z_exit has not yet been populated")
        return self._z_exit

    @property
    def z_exit_suph_e3(self) -> float:
        if self._z_exit_suph_e3 is None:
            raise RuntimeError("z_exit+3efolds has not yet been populated")
        return self._z_exit_suph_e3

    @property
    def z_exit_suph_e5(self) -> float:
        if self._z_exit_suph_e5 is None:
            raise RuntimeError("z_exit+5efolds has not yet been populated")
        return self._z_exit_suph_e5

    @property
    def z_exit_subh_e3(self) -> float:
        if self._z_exit_subh_e3 is None:
            raise RuntimeError("z_exit-3efolds has not yet been populated")
        return self._z_exit_subh_e3

    @property
    def z_exit_subh_e5(self) -> float:
        if self._z_exit_subh_e5 is None:
            raise RuntimeError("z_exit-5efolds has not yet been populated")
        return self._z_exit_subh_e5

    @property
    def compute_time(self) -> float:
        if self._compute_time is None:
            raise RuntimeError("compute_time has not yet been populated")
        return self._compute_time

    @property
    def stepping(self) -> int:
        if self._stepping is None:
            raise RuntimeError("stepping has not yet been populated")
        return self._stepping

    @property
    def atol(self) -> float:
        return self._atol.tol

    @property
    def rtol(self) -> float:
        return self._rtol.tol

    def populate_z_sample(
        self,
        samples_per_log10z: int = 50,
        z_end: float = 0.1,
        outside_horizon_efolds: str = None,
    ):
        if outside_horizon_efolds is None:
            outside_horizon_efolds = "zero"

        if outside_horizon_efolds == "zero" or outside_horizon_efolds == "crossing":
            z_init = self._z_exit
        elif outside_horizon_efolds == "suph_e5":
            z_init = self._z_exit_suph_e5
        elif outside_horizon_efolds == "suph_e3":
            z_init = self._z_exit_suph_e3
        elif outside_horizon_efolds == "subh_e3":
            z_init = self._z_exit_subh_e3
        elif outside_horizon_efolds == "subh_e5":
            z_init = self._z_exit_subh_e5
        else:
            raise RuntimeError("Unknown outside_horizon_efolds value")

        # now we want to built a set of sample points for redshifts between z_init and
        # the final point z = z_final, using the specified number of redshift sample points
        num_z_sample = int(
            round(samples_per_log10z * (log10(z_init) - log10(z_end)) + 0.5, 0)
        )

        return logspace(log10(z_init), log10(z_end), num=num_z_sample)


DEFAULT_HEXIT_TOLERANCE = 1e-2


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

    # set initial conditions for q = ln(k/aH)
    # the normalization a0 is absorbed into k/a0 = k_phys.
    q0 = log(float(k) / cosmology.H0)

    # RHS of ODE system for dq/dz = f(z)
    def RHS(z, state):
        # q = state[0]
        return [1.0 / (1.0 + z) - cosmology.d_lnH_dz(z)]

    # build event function to terminate when q crosses zero
    def q_event(target_efolds_suph, z, state):
        # target e-folds determines how many e-folds in/outside the horizon we should be to trigger the event
        # q = ln(k/aH)
        # if k/aH = exp(N), with N +ve for e-folds inside the horizon, -ve for e-folds outside, then we should trigger then
        # ln(k/aH0 = q = N, or q - N = 0.
        # hence, we want to trigger when q + target_efolds_suph = 0
        q = state[0] + target_efolds_suph
        return q

    q_subh_e5_event = partial(q_event, -5.0)
    q_subh_e3_event = partial(q_event, -3.0)
    q_zero_event = partial(q_event, 0.0)
    q_suph_e3_event = partial(q_event, +3.0)
    q_suph_e5_event = partial(q_event, +5.0)
    q_suph_e5_event.terminal = True

    with WallclockTimer() as timer:
        # solve to find the zero crossing point; we set the upper limit of integration to be 1E12, which should be comfortably above
        # the redshift of any horizon crossing in which we are interested.
        sol = solve_ivp(
            RHS,
            t_span=(0.0, 1e20),
            y0=[q0],
            events=[
                q_subh_e5_event,
                q_subh_e3_event,
                q_zero_event,
                q_suph_e3_event,
                q_suph_e5_event,
            ],
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

    if len(sol.t_events) != 5:
        raise RuntimeError(
            f"find_horizon_exit_time: unexpected number of event types returned from horizon-crossing integration (num={len(sol.t_events)})"
        )

    subh_e5_times, subh_e3_times, zero_times, suph_e3_times, suph_e5_times = (
        sol.t_events
    )
    if len(subh_e5_times) != 1:
        raise RuntimeError(
            f"find_horizon_exit_time: more than one exp(-5) horizon-crossing time returned from integration (num={len(subh_e5_times)})"
        )
    if len(subh_e3_times) != 1:
        raise RuntimeError(
            f"find_horizon_exit_time: more than one exp(-3) horizon-crossing time returned from integration (num={len(subh_e3_times)})"
        )
    if len(zero_times) != 1:
        raise RuntimeError(
            f"find_horizon_exit_time: more than one horizon-crossing time returned from integration (num={len(zero_times)})"
        )
    if len(suph_e3_times) != 1:
        raise RuntimeError(
            f"find_horizon_exit_time: more than one exp(+3) horizon-crossing time returned from integration (num={len(suph_e3_times)})"
        )
    if len(suph_e5_times) != 1:
        raise RuntimeError(
            f"find_horizon_exit_time: more than one exp(+5) horizon-crossing time returned from integration (num={len(suph_e5_times)})"
        )

    z_exit = zero_times[0]
    z_exit_suph_e3 = suph_e3_times[0]
    z_exit_suph_e5 = suph_e5_times[0]
    z_exit_subh_e3 = subh_e3_times[0]
    z_exit_subh_e5 = subh_e5_times[0]

    def check_log_k_aH(z_test, expected_log_N):
        H_test = cosmology.Hubble(z_test)
        k_over_aH = (1.0 + z_test) * float(k) / H_test
        q_test = log(k_over_aH)

        if fabs(q_test + expected_log_N) > DEFAULT_HEXIT_TOLERANCE:
            print("!! INCONSISTENT DETERMINATION OF HORIZON-CROSSING TIME")
            print(
                f"|    k = {k.k_inv_Mpc:.5g}/Mpc, estimated z_exit+{expected_log_N}= {z_test:.5g}, k/(aH)_test = {k_over_aH:.5g}, q_test = {q_test:.5g}"
            )
            raise RuntimeError(
                f"Inconsistent determination of horizon-crossing time z_exit+{expected_log_N}"
            )

    check_log_k_aH(z_exit, 0.0)
    check_log_k_aH(z_exit_suph_e3, 3.0)
    check_log_k_aH(z_exit_suph_e5, 5.0)
    check_log_k_aH(z_exit_subh_e3, -3.0)
    check_log_k_aH(z_exit_subh_e5, -5.0)

    return {
        "compute_time": timer.elapsed,
        "z_exit": z_exit,
        "z_exit_suph_e3": z_exit_suph_e3,
        "z_exit_suph_e5": z_exit_suph_e5,
        "z_exit_subh_e3": z_exit_subh_e3,
        "z_exit_subh_e5": z_exit_subh_e5,
    }
