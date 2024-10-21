import time
from typing import Optional, List

import ray
from math import fabs, log, exp
from scipy.integrate import solve_ivp

from AdaptiveLevin import adaptive_levin_sincos
from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.GkSource import GkSource
from ComputeTargets.QuadSource import QuadSource
from ComputeTargets.integration_supervisor import (
    IntegrationSupervisor,
    DEFAULT_UPDATE_INTERVAL,
    RHS_timer,
)
from CosmologyConcepts import wavenumber, redshift_array, wavenumber_exit_time, redshift
from Datastore import DatastoreObject
from MetadataConcepts import store_tag, tolerance
from defaults import DEFAULT_QUADRATURE_TOLERANCE, DEFAULT_ABS_TOLERANCE
from utilities import WallclockTimer, format_time


class QuadSourceSupervisor(IntegrationSupervisor):
    def __init__(
        self,
        k: wavenumber,
        q: wavenumber,
        r: wavenumber,
        label: str,
        z_response: redshift,
        log_z_init: float,
        log_z_final: float,
        notify_interval: int = DEFAULT_UPDATE_INTERVAL,
    ):
        super().__init__(notify_interval)

        self._k = k
        self._q = q
        self._r = r

        self._label = label

        self._z_response: float = z_response.z
        self._log_z_init: float = log_z_init
        self._log_z_final: float = log_z_final

        self._log_z_range: float = self._log_z_init - self._log_z_final

        self._last_z: float = self._log_z_init

        def __enter__(self):
            super().__enter__()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            super().__exit__(exc_type, exc_val, exc_tb)

        def message(self, current_log_z, msg):
            current_time = time.time()
            since_last_notify = current_time - self._last_notify
            since_start = current_time - self._start_time

            update_number = self.report_notify()

            z_complete = self._z_int - current_log_z
            z_remain = self._log_z_range - z_complete
            percent_remain = z_remain / self._log_z_range
            print(
                f"** STATUS UPDATE #{update_number}: QuadSourceIntegral quadrature (type={self._label}) for k={self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}), q={self._q.k_inv_Mpc:.5g}/Mpc (store_id={self._q.store_id}), r={self._r.k_inv_Mpc:.5g}/Mpc (store_id={self._r.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
            )
            print(
                f"|    current log(1+z)={current_log_z:.5g} (init log(1+z)={self._log_z_init:.5g}, target log(1+z)={self._log_z_final:.5g}, log(1+z) complete={z_complete:.5g}, log(1+z) remain={z_remain:.5g}, {percent_remain:.3%} remains)"
            )
            if self._last_log_z is not None:
                log_z_delta = self._last_log_z - current_log_z
                print(
                    f"|    redshift advance since last update: Delta log(1+z) = {log_z_delta:.5g}"
                )
            print(
                f"|    {self.RHS_evaluations} RHS evaluations, mean {self.mean_RHS_time:.5g}s per evaluation, min RHS time = {self.min_RHS_time:.5g}s, max RHS time = {self.max_RHS_time:.5g}s"
            )
            print(f"|    {msg}")

            self._last_log_z = current_log_z


@ray.remote
def compute_QuadSource_integral(
    model: BackgroundModel,
    k: wavenumber_exit_time,
    q: wavenumber_exit_time,
    r: wavenumber_exit_time,
    source: QuadSource,
    Gks: List[GkSource],
    z_response_sample: redshift_array,
    tol: float = DEFAULT_QUADRATURE_TOLERANCE,
) -> dict:
    values = []

    if len(z_response_sample) != len(Gks):
        raise RuntimeError(
            "compute_QuadSource_integral: Gks and z_response_sample must have the same length"
        )

    with WallclockTimer() as timer:
        # work through the z_response_sample array, building the source redshift integral as we go
        for z_response, Gk in zip(z_response_sample, Gks):
            z_response: redshift
            Gk: GkSource

            numeric_quad: float = 0.0
            WKB_quad: float = 0.0
            WKB_Levin: float = 0.0

            if Gk.type == "numeric":
                regions = [
                    (Gk.z_sample.max, z_response),
                    (None, None),
                    (None, None),
                ]

            elif Gk.type == "WKB":
                Levin_z: Optional[float] = Gk.Levin_z

                regions = [
                    (None, None),
                    (Gk.z_sample.max, Levin_z if Levin_z is not None else z_response),
                    (Levin_z, z_response),
                ]

            elif Gk.type == "mixed":
                crossover_z: Optional[float] = Gk.crossover_z
                Levin_z: Optional[float] = Gk.Levin_z

                if Levin_z > crossover_z:
                    Levin_z = crossover_z

                regions = [
                    (Gk.z_sample.max, crossover_z),
                    (
                        crossover_z if crossover_z is not None else Gk.z_sample.max,
                        Levin_z if Levin_z is not None else z_response,
                    ),
                    (Levin_z, z_response),
                ]

            else:
                raise NotImplementedError(f"Gk {Gk.type} not implemented")

            max_z, min_z = regions.pop(0)
            if (
                max_z is not None
                and min_z is not None
                and max_z - min_z > DEFAULT_QUADRATURE_TOLERANCE
            ):
                numeric_quad = numeric_quad_integral(
                    model,
                    k.k,
                    q.k,
                    r.k,
                    source,
                    Gk,
                    z_response,
                    max_z=max_z,
                    min_z=min_z,
                    tol=tol,
                )

            max_z, min_z = regions.pop(0)
            if (
                max_z is not None
                and min_z is not None
                and max_z - min_z > DEFAULT_QUADRATURE_TOLERANCE
            ):
                WKB_quad = WKB_quad_integral(
                    model,
                    k.k,
                    q.k,
                    r.k,
                    source,
                    Gk,
                    z_response,
                    max_z=max_z,
                    min_z=min_z,
                    tol=tol,
                )

            max_z, min_z = regions.pop(0)
            if (
                max_z is not None
                and min_z is not None
                and max_z - min_z > DEFAULT_QUADRATURE_TOLERANCE
            ):
                WKB_Levin = WKB_Levin_integral(
                    model,
                    k.k,
                    q.k,
                    r.k,
                    source,
                    Gk,
                    z_response,
                    max_z=max_z,
                    min_z=min_z,
                    tol=tol,
                )

            values.append(
                QuadSourceIntegralValue(
                    None,
                    z_response=z_response,
                    total=numeric_quad + WKB_quad + WKB_Levin,
                    numeric_quad_part=numeric_quad,
                    WKB_quad_part=WKB_quad,
                    WKB_Levin_part=WKB_Levin,
                    Gk_serial=Gk.store_id,
                )
            )

    return {"values": values, "compute_time": timer.elapsed, "metadata": {}}


def numeric_quad_integral(
    model: BackgroundModel,
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    source: QuadSource,
    Gk: GkSource,
    z_response: redshift,
    max_z: float,
    min_z: float,
    tol: float,
) -> float:
    source_f = source.functions
    Gk_f = Gk.functions
    model_f = model.functions

    def RHS(log_z_source, state, supervisor) -> List[float]:
        with RHS_timer(supervisor) as timer:
            current_value = state[0]

            if supervisor.notify_available:
                supervisor.message(
                    log_z_source, f"current state: current value = {current_value:.5g}"
                )

            Green = Gk_f.numerical_Gk(log_z_source, z_is_log=True)
            H = model_f.Hubble(exp(log_z_source) - 1.0)
            H_sq = H * H
            f = source_f.source(log_z_source, z_is_log=True)

        return [Green * f / H_sq]

    with QuadSourceSupervisor(
        k, q, r, "numeric quad", z_response, max_z, min_z
    ) as supervisor:
        state = [0.0]

        sol = solve_ivp(
            RHS,
            method="RK45",
            t_span=(log(1.0 + min_z), log(1.0 + max_z)),
            y0=state,
            atol=tol,
            rtol=tol,
            args=(supervisor,),
        )

    if not sol.success:
        raise RuntimeError(
            f'compute_QuadSource_integral: quadrature did not terminate successfully | error at z={sol.t[-1]}, "{sol.message}"'
        )

    if fabs(sol.t[-1] - min_z) > DEFAULT_ABS_TOLERANCE:
        raise RuntimeError(
            f"compute_QuadSource_integral: quadrature did not terminate at expected redshift z={min_z:.5g} (final z={sol.t[-1]:.3g})"
        )

    if len(sol.y) != 1:
        raise RuntimeError(
            f"compute_QuadSource_integral: solution does not have expected number of members (expected 1, found {len(sol.y)})"
        )

    return (1.0 + z_response.z) * sol.y[0][-1]


def WKB_quad_integral(
    model: BackgroundModel,
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    source: QuadSource,
    Gk: GkSource,
    z_response: redshift,
    max_z: float,
    min_z: float,
    tol: float,
) -> float:
    source_f = source.functions
    Gk_f = Gk.functions
    model_f = model.functions

    def RHS(log_z_source, state, supervisor) -> List[float]:
        with RHS_timer(supervisor) as timer:
            current_value = state[0]

            if supervisor.notify_available:
                supervisor.message(
                    log_z_source, f"current state: current value = {current_value:.5g}"
                )

            Green = Gk_f.WKB_Gk(log_z_source, z_is_log=True)
            H = model_f.Hubble(exp(log_z_source) - 1.0)
            H_sq = H * H
            f = source_f.source(log_z_source, z_is_log=True)

        return [Green * f / H_sq]

    with QuadSourceSupervisor(
        k, q, r, "WKB quad", z_response, max_z, min_z
    ) as supervisor:
        state = [0.0]

        sol = solve_ivp(
            RHS,
            method="RK45",
            t_span=(log(1.0 + min_z), log(1.0 + max_z)),
            y0=state,
            atol=tol,
            rtol=tol,
            args=(supervisor,),
        )

    if not sol.success:
        raise RuntimeError(
            f'compute_QuadSource_integral: quadrature did not terminate successfully | error at z={sol.t[-1]}, "{sol.message}"'
        )

    if fabs(sol.t[-1] - min_z) > DEFAULT_ABS_TOLERANCE:
        raise RuntimeError(
            f"compute_QuadSource_integral: quadrature did not terminate at expected redshift z={min_z:.5g} (final z={sol.t[-1]:.3g})"
        )

    if len(sol.y) != 1:
        raise RuntimeError(
            f"compute_QuadSource_integral: solution does not have expected number of members (expected 1, found {len(sol.y)})"
        )

    return (1.0 + z_response.z) * sol.y[0][-1]


def WKB_Levin_integral(
    model: BackgroundModel,
    k: wavenumber,
    q: wavenumber,
    r: wavenumber,
    source: QuadSource,
    Gk: GkSource,
    z_response: redshift,
    max_z: float,
    min_z: float,
    tol: float,
) -> float:
    source_f = source.functions
    Gk_f = Gk.functions
    model_f = model.functions

    x_span = (log(1.0 + min_z), log(1.0 + max_z))

    # sinc integral is sin(x)/x, so sin weight is 1/x and cos weight is 0
    def Levin_f(log_z_source: float) -> float:
        H = model_f.Hubble(exp(log_z_source) - 1.0)
        H_sq = H * H
        f = source_f.source(log_z_source, z_is_log=True)

        return [f / H_sq, 0.0]

    def Levin_theta(log_z_source: float) -> float:
        return Gk_f.theta(log_z_source, z_is_log=True)

    value, p_sample, regions, evaluations = adaptive_levin_sincos(
        x_span,
        Levin_f,
        Levin_theta,
        tol=tol,
        chebyshev_order=12,
    )

    return value


class QuadSourceIntegral(DatastoreObject):
    def __init__(
        self,
        payload,
        model,
        z_response_sample: redshift_array,
        k: wavenumber_exit_time,
        q: wavenumber_exit_time,
        r: wavenumber_exit_time,
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

        # store parametesr
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
                "QuadSourceIntegral: values read but _do_not_populate is set"
            )

        if self._values is None:
            raise RuntimeError("values has not yet been populated")
        return self._values

    def compute(self, payload, label: Optional[str] = None):
        if hasattr(self, "_do_not_populate"):
            raise RuntimeError(
                "QuadSourceIntegral: compute() called, but _do_not_populate is set"
            )

        if self._values is not None:
            raise RuntimeError(
                "QuadSourceIntegral: compute() called, but values have already been computed"
            )

        # replace label if specified
        if label is not None:
            self._label = label

        source: QuadSource = payload["source"]
        Gks: GkSource = payload["Gk"]

        # TODO: improve compatibility check between source and Gk
        count = 0
        for z_response, Gk in zip(self._z_response_sample, Gks):
            if Gk.z_response.store_id != z_response.store_id:
                raise RuntimeError(
                    f"QuadSourceIntegral: supplied vector of Green's functions does not match z_response sample for z_response={z_response.z:.5g} (element {count} of z_response sample)"
                )
            count = count + 1

        self._compute_ref = compute_QuadSource_integral.remote(
            self._model,
            source,
            Gks,
            self._z_response_sample,
        )

        self._source_serial = source.store_id
        return self._compute_ref

    def store(self) -> Optional[bool]:
        if self._compute_ref is None:
            raise RuntimeError(
                "QuadSourceIntegral: store() called, but no compute() is in progress"
            )

            # check whether the computation has actually resolved
        resolved, unresolved = ray.wait([self._compute_ref], timeout=0)

        if len(resolved) == 0:
            return None

        payload = ray.get(self._compute_ref)
        self._compute_ref = None

        self._values = payload["values"]
        self._compute_time = payload["compute_time"]
        self._metadata = payload["metadata"]


class QuadSourceIntegralValue(DatastoreObject):
    def __init__(
        self,
        store_id: None,
        z_response: redshift,
        total: float,
        numeric_quad_part: float,
        WKB_quad_part: float,
        WKB_Levin_part: float,
        Gk_serial: Optional[int] = None,
    ):
        DatastoreObject.__init__(self, store_id)

        self._z_response = z_response

        self._total = total
        self._numeric_quad_part = numeric_quad_part
        self._WKB_quad_part = WKB_quad_part
        self._WKB_Levin_part = WKB_Levin_part

        self._Gk_serial = Gk_serial

    @property
    def z_response(self) -> redshift:
        return self._z_response

    @property
    def total(self) -> float:
        return self._total

    @property
    def numeric_quad_part(self) -> float:
        return self._numeric_quad_part

    @property
    def WKB_quad_part(self) -> float:
        return self._WKB_quad_part

    @property
    def WKB_Levin_part(self) -> float:
        return self._WKB_Levin_part

    @property
    def Gk_serial(self) -> Optional[int]:
        return self._Gk_serial
