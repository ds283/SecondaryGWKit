from typing import Optional, List

import ray

from ComputeTargets.BackgroundModel import ModelProxy
from CosmologyConcepts import wavenumber, wavenumber_exit_time, redshift
from Datastore import DatastoreObject
from MetadataConcepts import store_tag, tolerance


class OneLoopIntegral(DatastoreObject):
    def __init__(
        self,
        payload,
        model: ModelProxy,
        z_response: redshift,
        k: wavenumber_exit_time,
        tol: tolerance,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        self._model_proxy = model

        self._k_exit = k

        self._z_response = z_response
        if payload is None:
            DatastoreObject.__init__(self, None)

            self._value = None

            self._compute_time = None
            self._source_serial = None
            self._metadata = None

        else:
            DatastoreObject.__init__(self, payload["store_id"])

            self._value = payload["value"]

            self._compute_time = payload["compute_time"]
            self._source_serial = payload["source_serial"]
            self._metadata = payload["metadata"]

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._tol = tol

        self._compute_ref = None

    @property
    def model_proxy(self) -> ModelProxy:
        return self._model_proxy

    @property
    def k(self) -> wavenumber:
        return self._k_exit.k

    @property
    def z_response(self) -> redshift:
        return self._z_response

    @property
    def compute_time(self) -> Optional[float]:
        if self._value is None:
            raise RuntimeError("value has not yet been populated")

        return self._compute_time

    @property
    def metadata(self) -> dict:
        # allow this field to be read if we have been deserialized with _do_not_populate
        # otherwise, absence of _values implies that we have not yet computed our contents
        if self._value is None:
            raise RuntimeError("value has not yet been populated")

        return self._metadata

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    @property
    def value(self) -> float:
        if self._value is None:
            raise RuntimeError("value has not yet been populated")

        return self._value

    def compute(self, payload, label: Optional[str] = None):
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "OneLoopIntegral: compute() called, but _do_not_populate is set"
            )

        if self._value is None:
            raise RuntimeError(
                "OneLoopIntegral: compute() called, but value haa already been computed"
            )

        # replace label if specified
        if label is not None:
            self._label = label

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "OneLoopIntegral: store() called, but no compute() is in progress"
            )

            # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        if len(resolved) == 0:
            return None
