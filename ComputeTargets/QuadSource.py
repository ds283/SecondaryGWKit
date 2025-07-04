from collections import namedtuple
from math import log
from typing import Optional, List

import ray
from scipy.interpolate import make_interp_spline

from ComputeTargets.BackgroundModel import BackgroundModel, ModelProxy
from ComputeTargets.TkNumericIntegration import (
    TkNumericIntegration,
    TkNumericValue,
)
from ComputeTargets.spline_wrappers import ZSplineWrapper
from CosmologyConcepts import wavenumber, redshift_array, redshift, wavenumber_exit_time
from CosmologyModels.base import check_cosmology
from Datastore import DatastoreObject
from MetadataConcepts import store_tag
from utilities import WallclockTimer

QuadSourceFunctions = namedtuple("QuadSourceFunctions", ["source"])


def source_function(
    Tq: float, Tr: float, Tq_prime: float, Tr_prime: float, z: float, w: float
):
    if Tq is None or Tr is None or Tq_prime is None or Tr_prime is None:
        return None

    if z is None or w is None:
        raise ValueError("z and w cannot be None")

    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z

    undiff_part = (5.0 + 3.0 * w) / (3.0 * (1.0 + w)) * Tq * Tr
    diff_part = (
        2.0
        / (3.0 * (1.0 + w))
        * (
            -one_plus_z * Tq * Tr_prime
            - one_plus_z * Tr * Tq_prime
            + one_plus_z_2 * Tq_prime * Tr_prime
        )
    )
    source_term = undiff_part + diff_part

    return {
        "undiff": undiff_part,
        "diff": diff_part,
        "source": source_term,
    }


@ray.remote
def compute_quad_source(
    model_proxy: ModelProxy,
    z_sample: redshift_array,
    Tq: TkNumericIntegration,
    Tr: TkNumericIntegration,
):
    model: BackgroundModel = model_proxy.get()

    Tq_zsample = Tq.z_sample
    Tr_zsample = Tr.z_sample

    source = []
    undiff = []
    diff = []

    analytic_source_rad = []
    analytic_undiff_rad = []
    analytic_diff_rad = []

    analytic_source_w = []
    analytic_undiff_w = []
    analytic_diff_w = []

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
                Tq_: TkNumericValue = Tq[q_idx]

                Tq_value = Tq_.T
                Tq_prime = Tq_.Tprime
                analytic_Tq_rad = Tq_.analytic_T_rad
                analytic_Tq_prime_rad = Tq_.analytic_Tprime_rad
                analytic_Tq_w = Tq_.analytic_T_w
                analytic_Tq_prime_w = Tq_.analytic_Tprime_w

                q_idx += 1
            else:
                Tq_value = 1.0
                Tq_prime = 0.0
                analytic_Tq_rad = 1.0
                analytic_Tq_prime_rad = 0.0
                analytic_Tq_w = 1.0
                analytic_Tq_prime_w = 0.0

            if not missing_r:
                Tr_: TkNumericValue = Tr[r_idx]

                Tr_value = Tr_.T
                Tr_prime = Tr_.Tprime
                analytic_Tr_rad = Tr_.analytic_T_rad
                analytic_Tr_prime_rad = Tr_.analytic_Tprime_rad
                analytic_Tr_w = Tr_.analytic_T_w
                analytic_Tr_prime_w = Tr_.analytic_Tprime_w

                r_idx += 1
            else:
                Tr_value = 1.0
                Tr_prime = 0.0
                analytic_Tr_rad = 1.0
                analytic_Tr_prime_rad = 0.0
                analytic_Tr_w = 1.0
                analytic_Tr_prime_w = 0.0

            wBackground = model.functions.wBackground(z.z)

            numeric = source_function(
                Tq_value, Tr_value, Tq_prime, Tr_prime, z.z, wBackground
            )
            analytic_rad = source_function(
                analytic_Tq_rad,
                analytic_Tr_rad,
                analytic_Tq_prime_rad,
                analytic_Tr_prime_rad,
                z.z,
                wBackground,
            )
            analytic_w = source_function(
                analytic_Tq_w,
                analytic_Tr_w,
                analytic_Tq_prime_w,
                analytic_Tr_prime_w,
                z.z,
                wBackground,
            )

            source.append(numeric["source"])
            undiff.append(numeric["undiff"])
            diff.append(numeric["diff"])

            analytic_source_rad.append(analytic_rad["source"])
            analytic_undiff_rad.append(analytic_rad["undiff"])
            analytic_diff_rad.append(analytic_rad["diff"])

            analytic_source_w.append(analytic_w["source"])
            analytic_undiff_w.append(analytic_w["undiff"])
            analytic_diff_w.append(analytic_w["diff"])

    return {
        "compute_time": timer.elapsed,
        "source": source,
        "undiff": undiff,
        "diff": diff,
        "analytic_source_rad": analytic_source_rad,
        "analytic_undiff_rad": analytic_undiff_rad,
        "analytic_diff_rad": analytic_diff_rad,
        "analytic_source_w": analytic_source_w,
        "analytic_undiff_w": analytic_undiff_w,
        "analytic_diff_w": analytic_diff_w,
    }


class QuadSource(DatastoreObject):
    """
    Encapsulates the tensor source term produced in a particular cosmology,,
    with a particular set of z-sample points, labelled by two wavenumbers q and r.
    """

    def __init__(
        self,
        payload,
        model: ModelProxy,
        z_sample: redshift_array,
        q: wavenumber_exit_time,
        r: wavenumber_exit_time,
        label: Optional[str] = None,
        tags: Optional[List[store_tag]] = None,
    ):
        # q is not used, but needs to be accepted because it functions as the shard key;
        # there is a .q attribute, but this is derived from the Tq TkNumericIntegration object

        self._z_sample = z_sample
        if payload is None:
            DatastoreObject.__init__(self, None)
            self._compute_time = None

            self._values = None
            self._Tq_serial = None
            self._Tr_serial = None
        else:
            DatastoreObject.__init__(self, payload["store_id"])
            self._compute_time = payload["compute_time"]

            self._values = payload["values"]
            self._Tq_serial = payload["Tq_serial"]
            self._Tr_serial = payload["Tr_serial"]

        # store parameters
        self._label = label
        self._tags = tags if tags is not None else []

        self._q_exit = q
        self._r_exit = r

        self._model_proxy = model

        self._functions = None

        self._compute_ref = None

    @property
    def model_proxy(self) -> ModelProxy:
        return self._model_proxy

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
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError("QuadSource: values read but _do_not_populate is set")

        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    @property
    def functions(self) -> QuadSourceFunctions:
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "QuadSource: attempt to construct dense output functions, but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError(
                "QuadSource: attempt to construct dense output functions, but values have not yet been populated"
            )

        if self._functions is None:
            self._create_functions()

        return self._functions

    def _create_functions(self):
        source_data = [(log(1.0 + v.z.z), v.source) for v in self.values]
        source_data.sort(key=lambda pair: pair[0])

        source_x_data, source_y_data = zip(*source_data)
        source_spline = make_interp_spline(source_x_data, source_y_data)

        self._functions = QuadSourceFunctions(
            source=ZSplineWrapper(
                source_spline,
                "T_k",
                self._z_sample.max.z,
                self._z_sample.min.z,
                log_z=True,
            )
        )

    def compute(self, payload, label: Optional[str] = None):
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "QuadSource: compute() called, but _do_not_populate is set"
            )

        if self._values is not None:
            raise RuntimeError(
                "QuadSource: compute() called, but values have already been computed"
            )

        # replace label if specified
        if label is not None:
            self._label = label

        Tq: TkNumericIntegration = payload["Tq"]
        Tr: TkNumericIntegration = payload["Tr"]

        # check compatibility of the ingredients we have been offered
        if not Tq.available:
            raise RuntimeError("QuadSource: Supplied Tq is not available")
        if not Tr.available:
            raise RuntimeError("QuadSource: Supplied Tr is not available")

        # ensure that data ingredients are compatible with our specification
        check_cosmology(Tq.model_proxy, self._model_proxy)
        check_cosmology(Tr.model_proxy, self._model_proxy)

        # don't check for equality between z-sample grids, because Tq and Tr will probably have different initial times if q is not equal to r

        self._compute_ref = compute_quad_source.remote(
            self._model_proxy,
            self.z_sample,
            Tq,
            Tr,
        )

        self._Tq_serial = Tq.store_id
        self._Tr_serial = Tr.store_id

        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "QuadSource: store() called, but no compute() is in progress"
            )

            # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        if len(resolved) == 0:
            return None

        # retrieve result and populate ourselves
        data = ray.get(self._compute_ref)
        self._compute_ref = None

        self._compute_time = data["compute_time"]

        source = data["source"]
        undiff = data["undiff"]
        diff = data["diff"]

        analytic_source_rad = data["analytic_source_rad"]
        analytic_undiff_rad = data["analytic_undiff_rad"]
        analytic_diff_rad = data["analytic_diff_rad"]

        analytic_source_w = data["analytic_source_w"]
        analytic_undiff_w = data["analytic_undiff_w"]
        analytic_diff_w = data["analytic_diff_w"]

        self._values = []
        for i in range(len(source)):
            self._values.append(
                QuadSourceValue(
                    None,
                    self._z_sample[i],
                    source[i],
                    undiff[i],
                    diff[i],
                    analytic_source_rad=analytic_source_rad[i],
                    analytic_undiff_rad=analytic_undiff_rad[i],
                    analytic_diff_rad=analytic_diff_rad[i],
                    analytic_source_w=analytic_source_w[i],
                    analytic_undiff_w=analytic_undiff_w[i],
                    analytic_diff_w=analytic_diff_w[i],
                )
            )

        return True


class QuadSourceValue(DatastoreObject):
    """
    Encapsulates a single sampled value of the tensor source term.
    """

    def __init__(
        self,
        store_id: None,
        z: redshift,
        source: float,
        undiff: float,
        diff: float,
        analytic_source_rad: Optional[float] = None,
        analytic_undiff_rad: Optional[float] = None,
        analytic_diff_rad: Optional[float] = None,
        analytic_source_w: Optional[float] = None,
        analytic_undiff_w: Optional[float] = None,
        analytic_diff_w: Optional[float] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z = z

        self._source = source
        self._undiff = undiff
        self._diff = diff

        self._analytic_source_rad = analytic_source_rad
        self._analytic_undiff_rad = analytic_undiff_rad
        self._analytic_diff_rad = analytic_diff_rad

        self._analytic_source_w = analytic_source_w
        self._analytic_undiff_w = analytic_undiff_w
        self._analytic_diff_w = analytic_diff_w

    @property
    def z(self) -> redshift:
        return self._z

    @property
    def source(self) -> float:
        return self._source

    @property
    def undiff(self) -> float:
        return self._undiff

    @property
    def diff(self) -> float:
        return self._diff

    @property
    def analytic_source_rad(self) -> Optional[float]:
        return self._analytic_source_rad

    @property
    def analytic_undiff_rad(self) -> Optional[float]:
        return self._analytic_undiff_rad

    @property
    def analytic_diff_rad(self) -> Optional[float]:
        return self._analytic_diff_rad

    @property
    def analytic_source_w(self) -> Optional[float]:
        return self._analytic_source_w

    @property
    def analytic_undiff_w(self) -> Optional[float]:
        return self._analytic_undiff_w

    @property
    def analytic_diff_w(self) -> Optional[float]:
        return self._analytic_diff_w
