from math import log10, exp
from typing import Iterable, Optional

import ray
from numpy import logspace

from CosmologyConcepts.tolerance import tolerance
from CosmologyModels.base import CosmologyBase
from Datastore import DatastoreObject
from utilities import find_horizon_exit_time, check_units


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
        return self.k


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

    def as_list(self) -> list[float]:
        return [float(k) for k in self._k_array]


class wavenumber_exit_time(DatastoreObject):
    def __init__(
        self,
        payload,
        k: wavenumber,
        cosmology: CosmologyBase,
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
            self._compute_time = None
            self._stepping = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._z_exit = payload["z_exit"]
            self._compute_time = payload["compute_time"]
            self._stepping = payload["stepping"]

        # store parameters
        self.k = k
        self.cosmology = cosmology

        self._future = None

        self._atol = atol
        self._rtol = rtol

    def compute(self):
        if self._z_exit is not None:
            raise RuntimeError("z_exit has already been computed")
        self._future = find_horizon_exit_time.remote(
            self.cosmology,
            self.k,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
        )
        return self._future

    async def store(self) -> Optional[bool]:
        if self._future is None:
            return None

        data = await self._future
        self._future = None

        self._z_exit = data["z_exit"]
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

    def populate_z_samples(
        self,
        outside_horizon_efolds=10.0,
        samples_per_log10z: int = 100,
        z_end: float = 0.1,
    ):
        if outside_horizon_efolds is not None and outside_horizon_efolds > 0.0:
            z_init = (1.0 + self._z_exit) * exp(outside_horizon_efolds) - 1.0
        else:
            z_init = self._z_exit

        print(
            f"horizon-crossing time for wavenumber k = {self.k.k_inv_Mpc}/Mpc is z_exit = {self.z_exit}"
        )
        print(
            f"-- using N = {outside_horizon_efolds} e-folds of superhorizon evolution, initial time is z_init = {z_init}"
        )

        # now we want to built a set of sample points for redshifts between z_init and
        # the final point z = z_final, using the specified number of redshift sample points
        num_z_samples = int(round(samples_per_log10z * log10(z_init) + 0.5, 0))
        print(
            f"-- using {samples_per_log10z} z-points per log10(z) requires {num_z_samples} samples"
        )

        return logspace(log10(z_init), log10(z_end), num=num_z_samples)
