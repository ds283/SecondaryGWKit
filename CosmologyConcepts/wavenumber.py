from functools import total_ordering
from math import log10, log, fabs, exp
from typing import Iterable, Optional, Mapping, List

import ray
from numpy import logspace
from scipy.optimize import root_scalar

from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import tolerance
from Units import check_units
from defaults import DEFAULT_ABS_TOLERANCE, DEFAULT_REL_TOLERANCE
from utilities import WallclockTimer


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


def _solve_horizon_exit(
    cosmology: BaseCosmology,
    k: wavenumber,
    offset_subh: int,
    atol: float = DEFAULT_ABS_TOLERANCE,
    rtol: float = DEFAULT_REL_TOLERANCE,
):
    """
    Solve the implicit equation log(k/aH) - offset_subh = 0 to find the horizon exit time (plus offset) associated with wavenumber k, i.e.
    (k/aH) = exp(offset).
    offset_subh should be a number representing the number of e-folds *inside* the horizon that we wish to locate.
    If it is a positive number, it represents the number of e-folds *inside* the horizon.
    If it is a negative number, it represents the number of e-folds *outside* the horizon.
    :param cosmology:
    :param k:
    :param offset:
    :param atol:
    :param rtol:
    :return:
    """

    # GUESS A SENSIBLE INITIAL REDSHIFT FOR THE CALCULATION

    # in radiation domination H(z) = H0 (1+z)^2 because H^2 ~ rho ~ T^4 and T ~ (1+z).
    # Therefore a(z)H(z) ~ H0(1+z).
    # The normalization a0 of a(z) is absorbed into k/a0 = k_phys.
    # Hence we can guess a possible redshift of horizon exit for this mode as
    #   1 + z_exit(k) = k/H0

    def q(log_z: float) -> float:
        z: float = exp(log_z) - 1.0
        return log(float(k) * (1.0 + z) / cosmology.Hubble(z)) - offset_subh

    log_z_guess = log(float(k) / cosmology.H0) - offset_subh
    q_guess = q(log_z_guess)

    # If q_guess is a positive number, then z_guess is a LOWER BOUND for the desired crossing time.
    # On the other hand, if q_quess is a negative number, then z_guess is an UPPER BOUND for the desired crossing time.

    COUNT_MAX = 25
    SEARCH_MULTIPLIER = 1.5
    LOG_SEARCH_OFFSET = log(SEARCH_MULTIPLIER)

    if q_guess > DEFAULT_HEXIT_TOLERANCE:
        log_z_lo = log_z_guess
        q_lo = q_guess

        log_z_hi = log_z_lo + LOG_SEARCH_OFFSET
        q_hi = q(log_z_hi)
        count = 0
        while q_hi > -DEFAULT_HEXIT_TOLERANCE and count < COUNT_MAX:
            log_z_hi = log_z_hi + LOG_SEARCH_OFFSET
            q_hi = q(log_z_hi)

            count += 1

        if count >= COUNT_MAX:
            raise RuntimeError(
                f"_solve_horizon_exit: failed to find upper bound z_hi for k={k.k_inv_Mpc:.5}/Mpc (z_lo={log_z_lo:.5g}, q_lo={q_lo:.5g}, last z_hi={log_z_hi:.5g}, last q_hi={q_hi:.5g})"
            )

    elif q_guess < -DEFAULT_HEXIT_TOLERANCE:
        log_z_hi = log_z_guess
        q_hi = q_guess

        log_z_lo = log_z_hi - LOG_SEARCH_OFFSET
        q_lo = q(log_z_lo)
        count = 0
        while q_lo < DEFAULT_HEXIT_TOLERANCE and count < COUNT_MAX:
            log_z_lo = log_z_lo - LOG_SEARCH_OFFSET
            q_lo = q(log_z_lo)

            count += 1

        if count >= COUNT_MAX:
            raise RuntimeError(
                f"_solve_horizon_exit: failed to find lower bound z_lo for k={k.k_inv_Mpc:.5}/Mpc (z_hi={log_z_hi:.5g}, q_hi={q_hi:.5g}, last z_lo={log_z_lo:.5g}, last q_lo={q_lo:.5g})"
            )

    else:
        # q_guess is very close to zero
        log_z_lo = log_z_guess - LOG_SEARCH_OFFSET
        log_z_hi = log_z_guess + LOG_SEARCH_OFFSET

        q_lo = q(log_z_lo)
        q_hi = q(log_z_hi)

    if q_hi * q_lo > 0.0:
        raise RuntimeError(
            f"_solve_horizon_exit: failed to bracket horizon crossing time for k={k.k_inv_Mpc:.5}/Mpc (z_lo={log_z_lo:.5g}, q_lo={q_lo:.5g}, z_hi={log_z_hi:.5g}, q_hi={q_hi:.5g})"
        )

    root = root_scalar(
        q,
        bracket=(log_z_lo, log_z_hi),
        xtol=atol,
        rtol=rtol,
    )

    if not root.converged:
        raise RuntimeError(
            f'_solve_horizon_exit: root_scalar() did not converge to a solution for k={k.k_inv_Mpc:.5}/Mpc: x_bracket=({log_z_lo:.5g}, {log_z_hi:.5g}), iterations={root.iterations}, method={root.method}: "{root.flag}"'
        )

    log_z_root = root.root
    q_root = q(log_z_root)
    if fabs(q_root) > DEFAULT_HEXIT_TOLERANCE:
        raise RuntimeError(
            f"_solve_horizon_exit: root_scalar() converged, but root is out of tolerance for k={k.k_inv_Mpc:.5}/Mpc: log_z_root={log_z_root:.5g}, |q_root|={fabs(q_root):.5g}, x_bracket=({log_z_lo:.5g}, {log_z_hi:.5g}), q_bracket-({q_lo:.5g}, {q_hi:.5g}), iterations={root.iterations}, method={root.method}"
        )

    return exp(log_z_root) - 1.0


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
    Compute the redshift of horizon exit for a mode of wavenumber k in the specified cosmology
    :param cosmology:
    :param k:
    :return:
    """
    check_units(k, cosmology)

    payload = {}
    with WallclockTimer() as timer:
        z_exit = _solve_horizon_exit(
            cosmology, k, offset_subh=0.0, atol=atol, rtol=rtol
        )
        payload["z_exit"] = z_exit

        for z_subh in subh_efolds:
            z_exit = _solve_horizon_exit(
                cosmology, k, offset_subh=z_subh, atol=atol, rtol=rtol
            )
            payload[f"z_exit_subh_e{z_subh}"] = z_exit

        for z_suph in suph_efolds:
            z_exit = _solve_horizon_exit(
                cosmology, k, offset_subh=-z_suph, atol=atol, rtol=rtol
            )
            payload[f"z_exit_suph_e{z_suph}"] = z_exit

    payload["compute_time"] = timer.elapsed

    return payload
