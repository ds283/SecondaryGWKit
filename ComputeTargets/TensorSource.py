from typing import Optional, List

import ray

from ComputeTargets import (
    MatterTransferFunctionIntegration,
    MatterTransferFunctionValue,
)
from CosmologyConcepts import wavenumber, redshift_array, redshift
from CosmologyModels import BaseCosmology
from Datastore import DatastoreObject
from MetadataConcepts import store_tag
from utilities import check_cosmology, WallclockTimer


@ray.remote
def compute_tensor_source(
    cosmology: BaseCosmology,
    z_sample: redshift_array,
    Tq: MatterTransferFunctionIntegration,
    Tr: MatterTransferFunctionIntegration,
):
    Tq_zsample = Tq.z_sample
    Tr_zsample = Tr.z_sample

    source_terms = []
    undiff_parts = []
    diff_parts = []

    analytic_source_terms = []
    analytic_undiff_parts = []
    analytic_diff_parts = []

    q_idx = 0
    r_idx = 0

    with WallclockTimer() as timer:
        for i in range(len(z_sample)):
            missing_q = False
            missing_r = False

            z: redshift = z_sample[i]
            Tq_z: redshift = Tq_zsample[q_idx]
            Tr_z: redshift = Tr_zsample[r_idx]

            if Tq_z.store_id != z.store_id:
                if q_idx > 0:
                    raise RuntimeError(
                        f"z_sample[{i}].store_id = {z.store_id}, but this redshift is missing from Tq.z_sample. Current Tq.z_sample[{q_idx}].store_id = {Tq_z.store_id}"
                    )
                missing_q = True

            if Tr_z.store_id != z.store_id:
                if r_idx > 0:
                    raise RuntimeError(
                        f"z_sample[{i}].store_id = {z.store_id}, but this redshift is missing from Tr.z_sample. Current Tr.z_sample[{r_idx}].store_id = {Tr_z.store_id}"
                    )
                missing_r = True

            if not missing_q:
                Tq_: MatterTransferFunctionValue = Tq[q_idx]

                Tq_value = Tq_.T
                Tq_prime = Tq_.Tprime
                analytic_Tq_value = Tq_.analytic_T
                analytic_Tq_prime = Tq_.analytic_Tprime

                q_idx += 1
            else:
                Tq_value = 1.0
                Tq_prime = 0.0
                analytic_Tq_value = 1.0
                analytic_Tq_prime = 0.0

            if not missing_r:
                Tr_: MatterTransferFunctionValue = Tr[r_idx]

                Tr_value = Tr_.T
                Tr_prime = Tr_.Tprime
                analytic_Tr_value = Tr_.analytic_T
                analytic_Tr_prime = Tr_.analytic_Tprime

                r_idx += 1
            else:
                Tr_value = 1.0
                Tr_prime = 0.0
                analytic_Tr_value = 1.0
                analytic_Tr_prime = 0.0

            one_plus_z = 1.0 + z.z
            one_plus_z_2 = one_plus_z * one_plus_z

            wBackgrond = cosmology.wBackground(z.z)

            undiff_part = (
                (5.0 + 3.0 * wBackgrond)
                / (3.0 * (1.0 + wBackgrond))
                * Tq_value
                * Tr_value
            )
            diff_part = (
                2.0
                / (3.0 * (1.0 + wBackgrond))
                * (
                    -one_plus_z * Tq_value * Tr_prime
                    - one_plus_z * Tr_value * Tq_prime
                    + one_plus_z_2 * Tq_prime * Tr_prime
                )
            )
            source_term = undiff_part + diff_part

            if (
                analytic_Tq_value is not None
                and analytic_Tq_prime is not None
                and analytic_Tr_value is not None
                and analytic_Tr_prime is not None
            ):
                analytic_undiff_part = (
                    (5.0 + 3.0 * wBackgrond)
                    / (3.0 * (1.0 + wBackgrond))
                    * analytic_Tq_value
                    * analytic_Tr_value
                )
                analytic_diff_part = (
                    2.0
                    / (3.0 * (1.0 + wBackgrond))
                    * (
                        -one_plus_z * analytic_Tq_value * analytic_Tr_prime
                        - one_plus_z * analytic_Tr_value * analytic_Tq_prime
                        + one_plus_z_2 * analytic_Tq_prime * analytic_Tr_prime
                    )
                )
                analytic_source_term = analytic_undiff_part + analytic_diff_part
            else:
                analytic_undiff_part = None
                analytic_diff_part = None
                analytic_source_term = None

            source_terms.append(source_term)
            undiff_parts.append(undiff_part)
            diff_parts.append(diff_part)

            analytic_source_terms.append(analytic_source_term)
            analytic_undiff_parts.append(analytic_undiff_part)
            analytic_diff_parts.append(analytic_diff_part)

    return {
        "compute_time": timer.elapsed,
        "source_term": source_terms,
        "undiff_part": undiff_parts,
        "diff_part": diff_parts,
        "analytic_source_term": analytic_source_terms,
        "analytic_undiff_part": analytic_undiff_parts,
        "analytic_diff_part": analytic_diff_parts,
    }


class TensorSource(DatastoreObject):
    """
    Encapsulates the tensor source term produced in a particular cosmology,,
    with a particular set of z-sample points, labelled by two wavenumbers q and r.
    """

    def __init__(
        self,
        payload,
        z_sample: redshift_array,
        Tq: MatterTransferFunctionIntegration,
        Tr: MatterTransferFunctionIntegration,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
        q: Optional[wavenumber] = None,
    ):
        # q is not used, but needs to be accepted because it functions as the shard key;
        # there is a .q attribute, but this is derived from the Tq MatterTransferFunctionIntegration object

        # check compatibility of the ingredients we have been offered
        check_cosmology(Tq, Tr)

        if not Tq.available:
            raise RuntimeError("Supplied Tq is not available")
        if not Tr.available:
            raise RuntimeError("Supplied Tr is not available")

        self._z_sample = z_sample
        if payload is None:
            DatastoreObject.__init__(self, None)
            self._compute_time = None

            self._values = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._compute_time = payload["compute_time"]

            self._values = payload["values"]

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._Tq = Tq
        self._Tr = Tr

        self._cosmology = Tq.cosmology
        self._q_exit = Tq._k_exit
        self._r_exit = Tr._k_exit

        self._compute_ref = None

    @property
    def cosmology(self) -> BaseCosmology:
        return self._cosmology

    @property
    def Tq(self) -> MatterTransferFunctionIntegration:
        return self._Tq

    @property
    def Tr(self) -> MatterTransferFunctionIntegration:
        return self._Tr

    @property
    def q(self) -> wavenumber:
        return self._q_exit.k

    @property
    def r(self) -> wavenumber:
        return self._r_exit.k

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    @property
    def z_sample(self) -> redshift_array:
        return self._z_sample

    @property
    def compute_time(self) -> float:
        if self._compute_time is None:
            raise RuntimeError("compute_time has not yet been populated")
        return self._compute_time

    @property
    def values(self) -> List:
        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self, label: Optional[str] = None):
        if self._values is not None:
            raise RuntimeError("values have already been computed")

        # replace label if specified
        if label is not None:
            self._label = label

        self._compute_ref = compute_tensor_source.remote(
            self.cosmology,
            self.z_sample,
            self._Tq,
            self._Tr,
        )
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "TensorSource: store() called, but no compute() is in progress"
            )

            # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        if len(resolved) == 0:
            return None

        # retrieve result and populate ourselves
        data = ray.get(self._compute_ref)
        self._compute_ref = None

        self._compute_time = data["compute_time"]
        source_term = data["source_term"]
        undiff_part = data["undiff_part"]
        diff_part = data["diff_part"]
        analytic_source_term = data["analytic_source_term"]
        analytic_undiff_part = data["analytic_undiff_part"]
        analytic_diff_part = data["analytic_diff_part"]

        self._values = []
        for i in range(len(source_term)):
            self._values.append(
                TensorSourceValue(
                    None,
                    self._z_sample[i],
                    source_term[i],
                    undiff_part[i],
                    diff_part[i],
                    analytic_source_term=analytic_source_term[i],
                    analytic_undiff_part=analytic_diff_part[i],
                    analytic_diff_part=analytic_undiff_part[i],
                )
            )

        return True


class TensorSourceValue(DatastoreObject):
    """
    Encapsulates a single sampled value of the tensor source term.
    """

    def __init__(
        self,
        store_id: None,
        z: redshift,
        source_term: float,
        undiff_part: float,
        diff_part: float,
        analytic_source_term: Optional[float] = None,
        analytic_undiff_part: Optional[float] = None,
        analytic_diff_part: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z

        self._source_term = source_term
        self._undiff_part = undiff_part
        self._diff_part = diff_part

        self._analytic_source_term = analytic_source_term
        self._analytic_undiff_part = analytic_undiff_part
        self._analytic_diff_part = analytic_diff_part

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def source_term(self) -> float:
        return self._source_term

    @property
    def undiff_part(self) -> float:
        return self._undiff_part

    @property
    def diff_part(self) -> float:
        return self._diff_part

    @property
    def analytic_source_term(self) -> Optional[float]:
        return self._analytic_source_term

    @property
    def analytic_undiff_part(self) -> Optional[float]:
        return self._analytic_undiff_part

    @property
    def analytic_diff_part(self) -> Optional[float]:
        return self._analytic_diff_part
