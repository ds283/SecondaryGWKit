from functools import partial, total_ordering
from math import log10, log, fabs, exp
from typing import Iterable, Optional, Mapping, List

import ray
from numpy import logspace
from scipy.integrate import solve_ivp

from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance
from Quadrature.supervisors.base import RHS_timer
from Quadrature.supervisors.wavenumber_exit import WavenumberExitSupervisor
from Units import check_units
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE


@total_ordering
class wavenumber(DatastoreObject):
    def __init__(
        self,
        store_id: int,
        k_inv_Mpc: float,
        units,
        is_source: bool = False,
        is_response: bool = False,
    ):
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

        self.is_source = is_source
        self.is_response = is_response

    def __float__(self):
        """
        Cast to float. Returns dimensionful wavenumber.
        :return:
        """
        return float(self.k)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.store_id == other.store_id

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.k < other.k

    def __hash__(self):
        return ("wavenumber", self.store_id).__hash__()


class wavenumber_array:
    def __init__(self, k_array: Iterable[wavenumber]):
        """
        Construct a datastore-backed object representing an array of wavenumber values
        """
        # store array in ascending order of k; the conversion to set ensure that we remove any duplicates
        self._k_array = sorted(set(k_array), key=lambda x: x.k)

    def __iter__(self):
        for k in self._k_array:
            yield k

    def __getitem__(self, key):
        return self._k_array[key]

    def __len__(self):
        return len(self._k_array)

    def __add__(self, other):
        full_set = set(self._k_array)
        full_set.update(set(other._k_array))
        return wavenumber_array(full_set)

    def as_list(self) -> list[float]:
        return [float(k) for k in self._k_array]

    def extend(self, k_array: Iterable[wavenumber]):
        full_set = set(self._k_array)
        full_set.update(set(k_array))
        self._k_array = sorted(full_set, key=lambda x: x.k)


WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS = [1, 2, 3, 4, 5]
WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS = [1, 2, 3, 4, 5, 6]


@total_ordering
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

            for z_offset in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
                setattr(self, f"_z_exit_suph_e{z_offset}", None)
            for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
                setattr(self, f"_z_exit_subh_e{z_offset}", None)

            self._compute_time = None
            self._stepping = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._z_exit = payload["z_exit"]

            for z_offset in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
                setattr(
                    self,
                    f"_z_exit_suph_e{z_offset}",
                    payload[f"z_exit_suph_e{z_offset}"],
                )
            for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
                setattr(
                    self,
                    f"_z_exit_subh_e{z_offset}",
                    payload[f"z_exit_subh_e{z_offset}"],
                )

            self._compute_time = payload["compute_time"]
            self._stepping = payload["stepping"]

        # store parameters
        self.k = k
        self.cosmology = cosmology

        self._compute_ref = None

        self._atol = atol
        self._rtol = rtol

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.store_id == other.store_id

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.k < other.k

    def __hash__(self):
        return ("wavenumber_exit_time", self.store_id).__hash__()

    def compute(self, label: Optional[str] = None):
        if self._z_exit is not None:
            raise RuntimeError("z_exit has already been computed")
        self._compute_ref = find_horizon_exit_time.remote(
            self.cosmology,
            self.k,
            suph_efolds=WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS,
            subh_efolds=WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS,
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

        for z_offset in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
            setattr(self, f"_z_exit_suph_e{z_offset}", data[f"z_exit_suph_e{z_offset}"])
        for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
            setattr(self, f"_z_exit_subh_e{z_offset}", data[f"z_exit_subh_e{z_offset}"])

        self._compute_time = data["compute_time"]
        self._stepping = 0

        return True

    @property
    def z_exit(self) -> float:
        if self._z_exit is None:
            raise RuntimeError("z_exit has not yet been populated")
        return self._z_exit

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
        outside_horizon_efolds: int = 3,
    ):
        """
        Build a set of z sample points, with specified density per log_10(z), and ending at the specified z_end.
        The initial time is taken to be the horizon re-entry time for this k-mode, or possibly offset by a specified
        number of e-folds in/outside the horizon, specified in 'outside_horizon_efolds'
        :param samples_per_log10z:
        :param z_end:
        :param outside_horizon_efolds:
        :return:
        """
        if outside_horizon_efolds == 0:
            z_init = self._z_exit
        elif outside_horizon_efolds > 0:
            if outside_horizon_efolds not in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
                raise RuntimeError(
                    f"wavenumber_exit_time: z_exit + superhorizon({outside_horizon_efolds}) is not computed"
                )

            z_init = getattr(self, f"z_exit_suph_e{outside_horizon_efolds}")
        elif outside_horizon_efolds < 0:
            if outside_horizon_efolds not in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
                raise RuntimeError(
                    f"wavenumber_exit_time: z_exit + subhorizon({outside_horizon_efolds}) is not computed"
                )

            z_init = getattr(self, f"z_exit_subh_e{outside_horizon_efolds}")
        else:
            raise RuntimeError(
                f"Unknown outside_horizon_efolds value {outside_horizon_efolds}"
            )

        # now we want to build a set of sample points for redshifts between z_init and
        # the final point z = z_final, using the specified number of redshift sample points
        num_z_sample = int(
            round(samples_per_log10z * (log10(z_init) - log10(z_end)) + 0.5, 0)
        )

        return logspace(log10(z_init), log10(z_end), num=num_z_sample)


# create accessors
def _create_accessor(attr_label):
    def accessor_template(self):
        if not hasattr(self, attr_label):
            raise RuntimeError(
                f'wavenumber_exit_time: object does not have attribute "{attr_label}"'
            )
        value = getattr(self, attr_label)
        if value is None:
            raise RuntimeError(
                f'wavenumber_exit_time: attribute "{attr_label}" has not yet been populated'
            )
        return value

    return accessor_template


for z_offset in WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS:
    setattr(
        wavenumber_exit_time,
        f"z_exit_suph_e{z_offset}",
        property(_create_accessor(f"_z_exit_suph_e{z_offset}")),
    )

for z_offset in WAVENUMBER_EXIT_TIMES_SUBHORIZON_EFOLDS:
    setattr(
        wavenumber_exit_time,
        f"z_exit_subh_e{z_offset}",
        property(_create_accessor(f"_z_exit_subh_e{z_offset}")),
    )


class wavenumber_exit_time_array:
    def __init__(self, k_exit_array: Iterable[wavenumber_exit_time]):
        """
        Construct a datastore-backed object representing an array of wavenumber values
        """
        # store array in ascending order of k; the conversion to set ensure that we remove any duplicates
        self._k_exit_array = sorted(set(k_exit_array), key=lambda x: x.k)

    def __iter__(self):
        for k in self._k_exit_array:
            yield k

    def __getitem__(self, key):
        return self._k_exit_array[key]

    def __len__(self):
        return len(self._k_exit_array)

    def __add__(self, other):
        full_set = set(self._k_exit_array)
        full_set.update(set(other._k_exit_array))
        return wavenumber_exit_time_array(full_set)

    def as_list(self) -> list[wavenumber_exit_time]:
        return [k for k in self._k_exit_array]

    def extend(self, k_array: Iterable[wavenumber]):
        full_set = set(self._k_exit_array)
        full_set.update(set(k_array))
        self._k_exit_array = sorted(full_set, key=lambda x: x.k)

    @property
    def max(self) -> wavenumber_exit_time:
        return self._k_exit_array[-1]

    @property
    def min(self) -> wavenumber_exit_time:
        return self._k_exit_array[0]


DEFAULT_HEXIT_TOLERANCE = 1e-2


@ray.remote
def find_horizon_exit_time(
    cosmology: BaseCosmology,
    k: wavenumber,
    suph_efolds: List[int],
    subh_efolds: List[int],
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
) -> Mapping[str, float]:
    """
    Compute the redshift of horizon exit for a mode of wavenumber k in the specified cosmology, by solving
    the equation (k/H) * (1+z) = k/(aH) = 1.
    To do this, we fix an initial condition for q = ln[ (k/H)*(1+z) ] today via q = ln[ k/H0 ],
    and then integrate dq/dz up to a point where it crosses zero.
    :param cosmology:
    :param k:
    :return:
    """
    check_units(k, cosmology)

    # GUESS A SENSIBLE INITIAL REDSHIFT FOR THE CALCULATION

    # in radiation domination H(z) = H0 (1+z)^2 because H^2 ~ rho ~ T^4 and T ~ (1+z).
    # Therefore a(z)H(z) ~ H0(1+z).
    # The normalization a0 of a(z) is absorbed into k/a0 = k_phys.
    # Hence we can guess a possible redshift of horizon exit for this mode as
    #   1 + z_exit(k) = k/H0

    # in practice we want to get redshifts for horizon crossing - subh_efolds and horizon_crossing + suph_efolds
    largest_subh_efold = max(subh_efolds)
    largest_suph_efold = max(suph_efolds)

    # find estimate for horizon crossing time of largest subhorizon e-fold
    z_lo_guess = exp(-largest_subh_efold) * (float(k) / cosmology.H0) - 1.0
    q_lo_guess = log(float(k) * (1.0 + z_lo_guess) / cosmology.Hubble(z_lo_guess))

    # if q_lo_guess is +ve then z_lo_guess is a good lower bound; its redshift is late enough.
    # Otherwise, we should gently decrease z_lo_guess until we get a lower bound
    COUNT_MAX = 25
    count = 0
    while (
        q_lo_guess - largest_subh_efold < DEFAULT_HEXIT_TOLERANCE and count < COUNT_MAX
    ):
        z_lo_guess = z_lo_guess / 2.0
        q_lo_guess = log(float(k) * (1.0 + z_lo_guess) / cosmology.Hubble(z_lo_guess))

        count += 1

    if count >= COUNT_MAX:
        raise RuntimeError(
            f"find_horizon_exit_time: failed to find lower bound for z_lo_guess"
        )

    # find estimate for horizon crossing time of largest superhorizon e-fold
    z_hi_guess = exp(+largest_suph_efold) * (float(k) / cosmology.H0) - 1.0
    q_hi_guess = log(float(k) * (1.0 + z_hi_guess) / cosmology.Hubble(z_hi_guess))

    # if q_hi_guess is -ve then z_hi_guess is a good upper bound; its redshift is early enough.
    # Otherwise, we should gently increase z_hi_guess until we get an upper bound
    count = 0
    while (
        q_hi_guess + largest_suph_efold > -DEFAULT_HEXIT_TOLERANCE and count < COUNT_MAX
    ):
        z_hi_guess = z_hi_guess * 2.0
        q_hi_guess = log(float(k) * (1.0 + z_hi_guess) / cosmology.Hubble(z_hi_guess))

        count += 1

    if count >= COUNT_MAX:
        raise RuntimeError(
            f"find_horizon_exit_time: failed to find upper bound for z_hi_guess"
        )

    # RHS of ODE system for dq/dz = f(z)
    def RHS(log_z, state, supervisor):
        with RHS_timer(supervisor) as timer:
            one_plus_z = exp(log_z)
            z = one_plus_z - 1.0

            if supervisor.notify_available:
                k_over_aH = one_plus_z * float(k) / cosmology.Hubble(z)
                q = log(k_over_aH)
                supervisor.message(
                    log_z, f"current k/aH = {k_over_aH:.5g}, log(k/aH) = {q:.5g}"
                )
                supervisor.reset_notify_time()

            # q = state[0]
            # evaluate (d/d ln(1+z)) (k/aH)
            return [1.0 - one_plus_z * cosmology.d_lnH_dz(z)]

    # build event function to terminate when q crosses zero
    def q_event(target_efolds_suph, log_z, state, supervisor):
        # target e-folds determines how many e-folds in/outside the horizon we should be to trigger the event
        # q = ln(k/aH)
        # if k/aH = exp(N), with N +ve for e-folds inside the horizon, -ve for e-folds outside, then we should trigger then
        # ln(k/aH0 = q = N, or q - N = 0.
        # hence, we want to trigger when q + target_efolds_suph = 0

        q = state[0] + target_efolds_suph
        return q

    suph_triggers = [partial(q_event, z_offset) for z_offset in suph_efolds]
    subh_triggers = [partial(q_event, -z_offset) for z_offset in subh_efolds]

    q_zero_event = partial(q_event, 0.0)

    suph_index_max = max(range(len(suph_efolds)), key=suph_efolds.__getitem__)
    terminal_event = suph_triggers[suph_index_max]
    terminal_event.terminal = True

    triggers = subh_triggers + [q_zero_event] + suph_triggers
    index_subh_trigger_start = 0
    index_zero_event = len(subh_triggers)
    index_suph_trigger_start = len(subh_triggers) + 1

    num_triggers = len(triggers)

    log_z_lo = log(1.0 + z_lo_guess)
    log_z_hi = log(1.0 + z_hi_guess)

    with WavenumberExitSupervisor(
        k=k, log_z_init=log_z_lo, log_z_final=log_z_hi
    ) as supervisor:
        # solve to find the zero crossing point; we set the upper limit of integration to be 1E12, which should be comfortably above
        # the redshift of any horizon crossing in which we are interested.
        sol = solve_ivp(
            RHS,
            t_span=(log_z_lo, log_z_hi),
            method="DOP853",
            y0=[q_lo_guess],
            events=triggers,
            atol=atol,
            rtol=rtol,
            args=(supervisor,),
        )

    # test whether termination occurred due to the q_zero_event() firing
    if not sol.success:
        raise RuntimeError(
            f'find_horizon_exit_time: integration to find horizon-crossing time did not terminate successfully ("{sol.message}")'
        )

    if sol.status != 1:
        print(sol)
        raise RuntimeError(
            f"find_horizon_exit_time: integration to find horizon-crossing time did not detect all required horizon exit times (and/or their offsets) within the integration range"
        )

    if len(sol.t_events) != num_triggers:
        raise RuntimeError(
            f"find_horizon_exit_time: unexpected number of event types returned from horizon-crossing integration (expected={num_triggers}, found={len(sol.t_events)})"
        )

    payload = {"compute_time": supervisor.integration_time}

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

    zero_times = sol.t_events[index_zero_event]
    if len(zero_times) != 1:
        raise RuntimeError(
            f"find_horizon_exit_time: more than one horizon-crossing time returned from integration (num={len(zero_times)})"
        )
    event = exp(zero_times[0]) - 1.0
    payload["z_exit"] = event
    check_log_k_aH(event, 0.0)

    for i, z_offset in enumerate(subh_efolds):
        times = sol.t_events[index_subh_trigger_start + i]
        if len(times) == 0:
            raise RuntimeError(
                f"find_horizon_exit_time: no horizon-crossing time returned from integration (subhorizon efolds={z_offset})"
            )
        if len(times) != 1:
            raise RuntimeError(
                f"find_horizon_exit_time: more than one horizon-crossing time returned from integration (subhorizon efolds={z_offset}, num={len(times)})"
            )
        event = exp(times[0]) - 1.0
        check_log_k_aH(event, -z_offset)
        payload[f"z_exit_subh_e{z_offset}"] = event

    for i, z_offset in enumerate(suph_efolds):
        times = sol.t_events[index_suph_trigger_start + i]
        if len(times) == 0:
            raise RuntimeError(
                f"find_horizon_exit_time: no horizon-crossing time returned from integration (superhorizon efolds={z_offset})"
            )
        if len(times) != 1:
            raise RuntimeError(
                f"find_horizon_exit_time: more than one horizon-crossing time returned from integration (superhorizon efolds={z_offset}, num={len(times)})"
            )
        event = exp(times[0]) - 1.0
        check_log_k_aH(event, z_offset)
        payload[f"z_exit_suph_e{z_offset}"] = event

    return payload
