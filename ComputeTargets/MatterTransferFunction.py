from math import log10, exp

import numpy as np
from ray.actor import ActorHandle

from CosmologyConcepts import redshift_array, wavenumber, redshift, tolerance
from CosmologyConcepts.wavenumber import wavenumber_exit_time
from CosmologyModels.base import CosmologyBase
from Datastore import DatastoreObject
from defaults import (
    DEFAULT_ABS_TOLERANCE,
    DEFAULT_REL_TOLERANCE,
)
from utilities import check_units


class MatterTransferFunctionIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced during a single integration of the
    matter transfer function, labelled by a wavenumber k, and sampled over
    a range of redshifts
    """

    def __init__(
        self,
        payload,
        cosmology: CosmologyBase,
        label: str,
        k: wavenumber,
        z_samples: redshift_array,
        z_init: redshift,
        atol: tolerance,
        rtol: tolerance,
    ):
        check_units(k, cosmology)

        if payload is None:
            DatastoreObject.__init__(self, None)
            self._compute_time = None
            self._compute_steps = None
            self._solver = None

            self._z_samples = z_samples
            self._values = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._compute_time = payload["compute_time"]
            self._compute_steps = payload["compute_steps"]
            self._solver = payload["solver"]

            self._z_samples = payload["z_samples"]
            self._values = payload["values"]

        # check that all sample points are *later* than the specified initial redshift
        z_init_float = float(z_init)
        for z in self._z_samples:
            z_float = float(z)
            if z_float > z_init_float:
                raise ValueError(
                    f"Sample redshift z={z_float} exceeds initial redshift z={z_init_float}"
                )

        # store parameters
        self._k = k
        self._cosmology = cosmology

        self.label = label
        self._z_init = z_init

        self._future = None

        self._atol = atol
        self._rtol = rtol

    @property
    def cosmology(self):
        return self._cosmology

    @property
    def k(self):
        return self._k

    @property
    def label(self):
        return self._label

    @property
    def z_init(self):
        return self._z_init

    @property
    def z_samples(self):
        return self._z_samples

    @property
    def compute_time(self) -> float:
        if self._compute_time is None:
            raise RuntimeError("compute_time has not yet been populated")
        return self._compute_time

    @property
    def compute_steps(self) -> float:
        if self._compute_time is None:
            raise RuntimeError("compute_steps has not yet been populated")
        return self._compute_time

    @property
    def solver(self) -> float:
        if self._solver is None:
            raise RuntimeError("compute_steps has not yet been populated")
        return self._solver

    @property
    def values(self) -> float:
        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values


class MatterTransferFunctionValue(DatastoreObject):
    """
    Encapsulates a single sampled value of the matter transfer functions.
    Parameters such as wavenumber k, intiial redshift z_init, etc., are held by the
    owning MatterTransferFunctionIntegration object
    """

    def __init__(self, store_id: int, z: redshift, value: float):
        if store_id is None:
            raise ValueError("Store ID cannot be None")
        DatastoreObject.__init__(self, store_id)

        self.z = z
        self.value = value

    def __float__(self):
        """
        Cast to float. Returns value of the transfer function
        :return:
        """
        return self.value


class MatterTransferFunction:
    """
    Encapsulates the time-evolution of the matter transfer function, labelled by a wavenumber k,
    and sampled over a specified range of redshifts.
    Notice this is a broker object, not an object that is itself persisted in the datastore
    """

    def __init__(
        self,
        store: ActorHandle,
        cosmology: CosmologyBase,
        k: wavenumber,
        z_init: redshift,
        z_samples: redshift_array,
        target_atol: tolerance = None,
        target_rtol: tolerance = None,
    ):
        """
        :param store_id: unique Datastore id. May be None if the object has not yet been fully serialized
        :param cosmology: cosmology instance
        :param k: wavenumber object
        :param z_initial: initial redshift of the matter transfer function
        :param z_samples: redshift values at which to sample the matter transfer function
        """
        DatastoreObject.__init__(self, store)
        self._cosmology: CosmologyBase = cosmology

        # cache wavenumber and z-sample array
        self._k = k
        self._z_samples = z_samples
        self._z_initial = z_init

        if target_atol is None:
            target_atol = tolerance(store, tol=DEFAULT_ABS_TOLERANCE)
        if target_rtol is None:
            target_rtol = tolerance(store, tol=DEFAULT_REL_TOLERANCE)

        self._target_atol: tolerance = target_atol
        self._target_rtol: tolerance = target_rtol

        # query datastore to find out whether the necessary sample values are available,
        # and schedule asynchronous tasks to compute any that are missing.
        self._sample_values = [
            MatterTransferFunctionValue(
                store, cosmology, k, z_init, z, target_atol, target_rtol
            )
            for z in z_samples
        ]

        self._missing_zs = [val.z for val in self._sample_values if not val.available]
        print(
            f"Matter transfer function T(z) for '{cosmology.name}' k={k.k_inv_Mpc}/Mpc has {len(self._missing_zs)} missing z-sample values"
        )

        # schedule an integration to populate any missing values
        if len(self._missing_zs) > 0:
            print(f"Scheduling an integration ...")
