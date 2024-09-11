from typing import Optional, List

import ray

from CosmologyConcepts import wavenumber, redshift, redshift_array, tolerance
from CosmologyModels.base import BaseCosmology
from Datastore import DatastoreObject
from utilities import check_units
from .integration_metadata import IntegrationSolver


class TensorGreenFunctionIntegration(DatastoreObject):
    """
    Encapsulates all sample points produced during a single integration of the
    tensor Green's function, labelled by a wavenumber k, and two redshifts:
    the source redshift, and the response redshift.
    A single integration fixes the source redshift and determines the Green function
    as a function of the response redshift.
    However, once these have been computed and cached, we can obtain the result as
    a function of the source redshift if we wish
    """

    def __init__(
        self,
        payload,
        cosmology: BaseCosmology,
        label: str,
        k: wavenumber,
        z_source: redshift,
        z_sample: redshift_array,
        atol: tolerance,
        rtol: tolerance,
    ):
        check_units(k, cosmology)

        if payload is None:
            DatastoreObject.__init__(self, None)
            self._compute_time = None
            self._compute_steps = None
            self._solver = None

            self._z_sample = z_sample
            self._values = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._compute_time = payload["compute_time"]
            self._compute_steps = payload["compute_steps"]
            self._solver = payload["solver"]

            self._z_sample = z_sample
            self._values = payload["values"]

        # check that all sample points are *later* than the specified source redshift
        z_init_float = float(z_source)
        for z in self._z_sample:
            z_float = float(z)
            if z_float > z_init_float:
                raise ValueError(
                    f"Redshift sample point z={z_float} exceeds source redshift z={z_init_float}"
                )

        # store parameters
        self._k = k
        self._cosmology = cosmology

        self._label = label
        self._z_source = z_source

        self._compute_ref = None

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
    def z_source(self):
        return self._z_source

    @property
    def z_sample(self):
        return self._z_sample

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
    def values(self) -> List:
        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self):
        if self._values is not None:
            raise RuntimeError("values have already been computed")
        self._compute_ref = compute_tensor_Green.remote(
            self.cosmology,
            self.k,
            self.z_source,
            self.z_sample,
            atol=self._atol.tol,
            rtol=self._rtol.tol,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "MatterTransferFunctionIntegration: store() called, but no compute() is in progress"
            )

        # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        # if not, return None
        if len(resolved) == 0:
            return None

        # retrieve result and populate ourselves
        data = ray.get(self._compute_ref)
        self._compute_ref = None

        self._compute_time = data["compute_time"]
        self._compute_steps = data["compute_steps"]

        values = data["values"]
        self._values = []

        for i in range(len(values)):
            # create new TensorGreenFunctionValue object
            self._values.append(
                TensorGreenFunctionValue(None, self._z_sample[i], values[i])
            )

        self._solver = IntegrationSolver(
            store_id=None, label=data["solver_label"], stepping=data["solver_stepping"]
        )

        return True


class TensorGreenFunctionValue(DatastoreObject):
    """
    Encapsulates a single sampled value of the tensor Green's transfer functions.
    Parameters such as wavenumber k, source redshift z_source, etc., are held by the
    owning TensorGreenFunctionIntegration object
    """

    def __init__(self, store_id: int, z: redshift, value: float):
        DatastoreObject.__init__(self, store_id)

        self._z = z
        self._value = value

    def __float__(self):
        """
        Cast to float. Returns value of the transfer function
        :return:
        """
        return self.value

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def z_serial(self) -> int:
        return self._z.store_id

    @property
    def value(self) -> float:
        return self._value


class TensorGreenFunctionContainer:
    """
    Encapsulates the time-evolution of the tensor Green's function with *response redshift*,
    labelled by a wavenumber k, sampled over a specified range of redshifts.
    Notice this is a broker object, not an object that is itself persisted in the datastore
    """

    def __init__(
        self,
        payload,
        cosmology: BaseCosmology,
        k: wavenumber,
        z_source: redshift,
        z_sample: redshift_array,
        target_atol: tolerance,
        target_rtol: tolerance,
    ):
        """
        :param cosmology: cosmology instance
        :param k: wavenumber object
        :param z_source: initial redshift of the Green's function
        :param z_sample: redshift values at which to sample the matter transfer function
        """
        self._cosmology: BaseCosmology = cosmology

        # cache wavenumber and z-sample array
        self._k = k
        self._z_sample = z_sample
        self._z_source = z_source

        self._target_atol = target_atol
        self._target_rtol = target_rtol

        if payload is None:
            self._values = {}
        else:
            self._values = payload["values"]

        # determine if any values are missing from the sample
        self._missing_z_serials = set(z.store_id for z in z_sample).difference(
            self._values.keys()
        )
        self._missing_zs = redshift_array(
            z_array=[z for z in self._z_sample if z.store_id in self._missing_z_serials]
        )

        num_missing = len(self._missing_zs)
        if num_missing > 0:
            print(
                f"Tensor Green's function G^\chi_k(z, z_i) for '{cosmology.name}' k={k.k_inv_Mpc}/Mpc has {num_missing} missing z-sample values"
            )

    @property
    def k(self) -> wavenumber:
        return self._k

    @property
    def z_source(self) -> redshift:
        return self._z_source

    @property
    def available(self) -> bool:
        return len(self._missing_zs) == 0

    @property
    def missing_z_sample(self) -> redshift_array:
        return self._missing_zs
