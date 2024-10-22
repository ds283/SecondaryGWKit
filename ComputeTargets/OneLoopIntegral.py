from typing import Optional, List

import ray

from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.QuadSourceIntegral import QuadSourceIntegral
from CosmologyConcepts import wavenumber, redshift_array, wavenumber_exit_time, redshift
from CosmologyConcepts.redshift import check_zsample
from Datastore import DatastoreObject
from MetadataConcepts import store_tag, tolerance


class OneLoopIntegral(DatastoreObject):
    def __init__(
        self,
        payload,
        model,
        z_response_sample: redshift_array,
        k: wavenumber_exit_time,
        tol: tolerance,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        self._model = model

        self._k_exit = k
        self._q_exit = q
        self._r_exit = r

        self._z_response_sample = z_response_sample
        if payload is None:
            DatastoreObject.__init__(self, None)

            self._compute_time = None
            self._source_serial = None
            self._metadata = None

            self._values = None

        else:
            DatastoreObject.__init__(self, payload["store_id"])

            self._compute_time = payload["integration_data"]
            self._metadata = payload["metadata"]

            self._source_serial = payload["source_serial"]
            self._values = payload["values"]

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._tol = tol

        self._compute_ref = None

    @property
    def model(self) -> BackgroundModel:
        return self._model

    @property
    def k(self) -> wavenumber:
        return self._k_exit.k

    @property
    def q(self) -> wavenumber:
        return self._q_exit.k

    @property
    def r(self) -> wavenumber:
        return self._r_exit.k

    @property
    def compute_time(self) -> Optional[float]:
        if self._values is None:
            raise RuntimeError("values have not yet been populated")

        return self._compute_time

    @property
    def metadata(self) -> dict:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._values is None and not hasattr(self, "_do_not_populate"):
            raise RuntimeError("values have not yet been populated")

        return self._metadata

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    @property
    def z_response_sample(self) -> redshift_array:
        return self._z_response_sample

    @property
    def values(self) -> List:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "OneLoopIntegral: values read but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self, payload, label: Optional[str] = None):
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "OneLoopIntegral: compute() called, but _do_not_populate is set"
            )

        if self._values is not None:
            raise RuntimeError(
                "OneLoopIntegral: compute() called, but values have already been computed"
            )

        # replace label if specified
        if label is not None:
            self._label = label

        source: QuadSourceIntegral = payload["source"]

        # TODO: check compatibility condition between source and Gk
        check_zsample(source.z_response_sample, self._z_response_sample)

        self._source_serial = source.store_id

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "OneLoopIntegral: store() called, but no compute() is in progress"
            )

            # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        if len(resolved) == 0:
            return None


class OneLoopIntegralValue(DatastoreObject):
    def __init__(
        self,
        store_id: None,
        z_response: redshift,
        total: float,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z_response = z_response

        self._total = total

    @property
    def z_response(self) -> redshift:
        return self._z_response

    @property
    def total(self) -> float:
        return self._total
