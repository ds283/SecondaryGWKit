import time
from typing import Optional, List, Union

import ray
from math import log, exp
from scipy.integrate import solve_ivp

from AdaptiveLevin import adaptive_levin_sincos
from ComputeTargets.BackgroundModel import BackgroundModel
from ComputeTargets.GkSource import GkSource
from ComputeTargets.QuadSource import QuadSource
from ComputeTargets.integration_metadata import IntegrationData, LevinData
from ComputeTargets.integration_supervisor import (
    IntegrationSupervisor,
    DEFAULT_UPDATE_INTERVAL,
    RHS_timer,
)
from CosmologyConcepts import wavenumber, wavenumber_exit_time, redshift
from Datastore import DatastoreObject
from MetadataConcepts import store_tag, tolerance
from defaults import (
    DEFAULT_QUADRATURE_TOLERANCE,
    DEFAULT_FLOAT_PRECISION,
)
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

        self._log_z_range: float = self._log_z_final - self._log_z_init

        self._last_log_z: float = self._log_z_init

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

        z_complete = current_log_z - self._log_z_init
        z_remain = self._log_z_range - z_complete
        percent_remain = z_remain / self._log_z_range
        print(
            f"** STATUS UPDATE #{update_number}: QuadSourceIntegral quadrature (type={self._label}) for k={self._k.k_inv_Mpc:.5g}/Mpc (store_id={self._k.store_id}), q={self._q.k_inv_Mpc:.5g}/Mpc (store_id={self._q.store_id}), r={self._r.k_inv_Mpc:.5g}/Mpc (store_id={self._r.store_id}) has been running for {format_time(since_start)} ({format_time(since_last_notify)} since last notification)"
        )
        print(
            f"|    current log(1+z)={current_log_z:.5g} (init log(1+z)={self._log_z_init:.5g}, target log(1+z)={self._log_z_final:.5g}, log(1+z) complete={z_complete:.5g}, log(1+z) remain={z_remain:.5g}, {percent_remain:.3%} remains)"
        )
        if self._last_log_z is not None:
            log_z_delta = current_log_z - self._last_log_z
            print(
                f"|    redshift advance since last update: Delta log(1+z) = {log_z_delta:.5g}"
            )
        print(
            f"|    {self.RHS_evaluations} RHS evaluations, mean {self.mean_RHS_time:.5g}s per evaluation, min RHS time = {self.min_RHS_time:.5g}s, max RHS time = {self.max_RHS_time:.5g}s"
        )
        print(f"|    {msg}")

        self._last_log_z = current_log_z


def get_z(z):
    if isinstance(z, redshift):
        return z.z

    elif isinstance(z, float):
        return z

    return float(z)


@ray.remote
def compute_QuadSource_integral(
    model: BackgroundModel,
    k: wavenumber_exit_time,
    q: wavenumber_exit_time,
    r: wavenumber_exit_time,
    source: QuadSource,
    Gk: GkSource,
    z_response: redshift,
    z_source_max: redshift,
    tol: float = DEFAULT_QUADRATURE_TOLERANCE,
) -> dict:
    # we are entitled to assume that Gk and source are evaluated at the same z_response
    # also that source and Gk have z_samples at least as far back as z_source_max

    print(
        f"** QUADRATIC SOURCE INTEGRAL: k={k.k.k_inv_Mpc:.3g}/Mpc, q={q.k.k_inv_Mpc:.3g}/Mpc, r={r.k.k_inv_Mpc:.3g}/Mpc (source store_id={source.store_id}, k store_id={k.store_id}) starting calculation for z_response={z_response.z:.5g}"
    )
    start_time = time.time()

    with WallclockTimer() as timer:
        numeric_quad: float = 0.0
        WKB_quad: float = 0.0
        WKB_Levin: float = 0.0

        numeric_quad_data = None
        WKB_quad_data = None
        WKB_Levin_data = None

        if Gk.type == "numeric":
            regions = [
                (z_source_max, z_response),
                (None, None),
                (None, None),
            ]

        elif Gk.type == "WKB":
            Levin_z: Optional[float] = Gk.Levin_z

            regions = [
                (None, None),
                (z_source_max, Levin_z if Levin_z is not None else z_response),
                (Levin_z, z_response),
            ]

        elif Gk.type == "mixed":
            crossover_z: Optional[float] = Gk.crossover_z
            Levin_z: Optional[float] = Gk.Levin_z

            # if Levin threshold occurs before the crossover point, move it up to the crossover point
            if (
                Levin_z is not None
                and crossover_z is not None
                and Levin_z > crossover_z
            ):
                Levin_z = crossover_z

            regions = [
                (z_source_max, crossover_z),
                (
                    crossover_z if crossover_z is not None else z_source_max,
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
            and get_z(max_z) - get_z(min_z) > DEFAULT_QUADRATURE_TOLERANCE
        ):
            now = time.time()
            print(
                f"|  --  (source store_id={source.store_id}, k store_id={k.store_id}) running time={format_time(now - start_time)}, starting numerical quadrature part"
            )
            payload = numeric_quad_integral(
                model,
                k.k,
                q.k,
                r.k,
                source,
                Gk,
                z_response,
                max_z=get_z(max_z),
                min_z=get_z(min_z),
                tol=tol,
            )
            numeric_quad = payload["value"]
            numeric_quad_data = payload["data"]

        max_z, min_z = regions.pop(0)
        if (
            max_z is not None
            and min_z is not None
            and get_z(max_z) - get_z(min_z) > DEFAULT_QUADRATURE_TOLERANCE
        ):
            now = time.time()
            print(
                f"|  --  (source store_id={source.store_id}, k store_id={k.store_id}) running time={format_time(now - start_time)}, starting WKB quadrature part"
            )
            payload = WKB_quad_integral(
                model,
                k.k,
                q.k,
                r.k,
                source,
                Gk,
                z_response,
                max_z=get_z(max_z),
                min_z=get_z(min_z),
                tol=tol,
            )
            WKB_quad = payload["value"]
            WKB_quad_data = payload["data"]

        max_z, min_z = regions.pop(0)
        if (
            max_z is not None
            and min_z is not None
            and get_z(max_z) - get_z(min_z) > DEFAULT_QUADRATURE_TOLERANCE
        ):
            now = time.time()
            print(
                f"|  --  (source store_id={source.store_id}, k store_id={k.store_id}) running time={format_time(now - start_time)}, starting WKB Levin part"
            )
            payload = WKB_Levin_integral(
                model,
                k.k,
                q.k,
                r.k,
                source,
                Gk,
                z_response,
                max_z=get_z(max_z),
                min_z=get_z(min_z),
                tol=tol,
            )
            WKB_Levin = payload["value"]
            WKB_Levin_data = payload["data"]

    return {
        "total": numeric_quad + WKB_quad + WKB_Levin,
        "numeric_quad": numeric_quad,
        "WKB_quad": WKB_quad,
        "WKB_Levin": WKB_Levin,
        "Gk_serial": Gk.store_id,
        "source_serial": source.store_id,
        "numeric_quad_data": numeric_quad_data,
        "WKB_quad_data": WKB_quad_data,
        "WKB_Levin_data": WKB_Levin_data,
        "compute_time": timer.elapsed,
        "metadata": {},
    }


def _extract_z(z: Union[type(None), redshift, float]) -> str:
    if z is None:
        return "(not set)"

    if isinstance(z, redshift):
        return f"{z.z:.5g}"

    if isinstance(z, float):
        return f"{z:.5g}"

    return f"{float(z):..5g}"


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
) -> dict:
    source_f = source.functions
    Gk_f = Gk.functions
    model_f = model.functions

    if Gk.type not in ["numeric", "mixed"]:
        raise RuntimeError(
            f'compute_QuadSource_integral: attempting to evaluate numerical quadrature, but Gk object is not of "numeric" or "mixed" type [domain={max_z:.5g}, {min_z:.5g}]'
        )

    if Gk_f.numerical_Gk is None:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but Gk_f.numerical_Gk is absent (type={Gk.type}, quality={Gk.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    if Gk_f.numerical_region is None:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but Gk_f.numerical_region is absent (type={Gk.type}, quality={Gk.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    region_max_z, region_min_z = Gk_f.numerical_region
    if max_z > region_max_z + DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but max_z={max_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a numerical solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )
    if min_z < region_min_z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate numerical quadrature, but min_z={min_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a numerical solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )

    def RHS(log_z_source, state, supervisor: QuadSourceSupervisor) -> List[float]:
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

    log_min_z = log(1.0 + min_z)
    log_max_z = log(1.0 + max_z)

    with QuadSourceSupervisor(
        k, q, r, "numeric quad", z_response, min_z, max_z
    ) as supervisor:
        state = [0.0]

        sol = solve_ivp(
            RHS,
            method="RK45",
            t_span=(log_min_z, log_max_z),
            t_eval=[log_max_z],
            y0=state,
            dense_output=True,
            atol=tol,
            rtol=tol,
            args=(supervisor,),
        )

    if not sol.success:
        raise RuntimeError(
            f'compute_QuadSource_integral: quadrature did not terminate successfully | error at z={sol.t[0]}, "{sol.message}"'
        )

    if sol.t[0] < log_max_z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: quadrature did not terminate at expected redshift log(1+z)={log_max_z:.5g} (final log(1+z)={sol.t[0]:.3g})"
        )

    if len(sol.sol(log_max_z)) != 1:
        raise RuntimeError(
            f"compute_QuadSource_integral: solution does not have expected number of members (expected 1, found {len(sol.sol(log_max_z))})"
        )

    return {
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=int(sol.nfev),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "value": (1.0 + z_response.z) * sol.sol(log_max_z)[0],
    }


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
) -> dict:
    source_f = source.functions
    Gk_f = Gk.functions
    model_f = model.functions

    if Gk.type not in ["WKB", "mixed"]:
        raise RuntimeError(
            f'compute_QuadSource_integral: attempting to evaluate WKB quadrature, but Gk object is not of "WKB" or "mixed" type [domain={max_z:.5g}, {min_z:.5g}]'
        )

    if Gk_f.WKB_Gk is None:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB quadrature, but Gk_f.WKB_Gk is absent (type={Gk.type}, quality={Gk.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}, z_crossover={_extract_z(Gk.crossover_z)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    if Gk_f.WKB_region is None:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB quadrature, but Gk_f.WKB_region is absent (type={Gk.type}, quality={Gk.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}), z_crossover={_extract_z(Gk.crossover_z)} [domain={max_z:.5g}, {min_z:.5g}]"
        )

    region_max_z, region_min_z = Gk_f.WKB_region
    if max_z > region_max_z + DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB quadrature, but max_z={max_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a WKB solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )
    if min_z < region_min_z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB quadrature, but min_z={min_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a WKB solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )

    def RHS(log_z_source, state, supervisor: QuadSourceSupervisor) -> List[float]:
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

    log_min_z = log(1.0 + min_z)
    log_max_z = log(1.0 + max_z)

    with QuadSourceSupervisor(
        k, q, r, "WKB quad", z_response, log_min_z, log_max_z
    ) as supervisor:
        state = [0.0]

        sol = solve_ivp(
            RHS,
            method="RK45",
            t_span=(log_min_z, log_max_z),
            t_eval=[log_max_z],
            y0=state,
            dense_output=True,
            atol=tol,
            rtol=tol,
            args=(supervisor,),
        )

    if not sol.success:
        raise RuntimeError(
            f'compute_QuadSource_integral: quadrature did not terminate successfully | error at log(1+z)={sol.t[0]}, "{sol.message}"'
        )

    if sol.t[0] < log_max_z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: quadrature did not terminate at expected redshift log(1+z)={log_max_z:.5g} (final log(1+z)={sol.t[0]:.3g})"
        )

    if len(sol.sol(log_max_z)) != 1:
        raise RuntimeError(
            f"compute_QuadSource_integral: solution does not have expected number of members (expected 1, found {len(sol.sol(log_max_z))})"
        )

    return {
        "data": IntegrationData(
            compute_time=supervisor.integration_time,
            compute_steps=int(sol.nfev),
            RHS_evaluations=supervisor.RHS_evaluations,
            mean_RHS_time=supervisor.mean_RHS_time,
            max_RHS_time=supervisor.max_RHS_time,
            min_RHS_time=supervisor.min_RHS_time,
        ),
        "value": (1.0 + z_response.z) * sol.sol(log_max_z)[0],
    }


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
) -> dict:
    source_f = source.functions
    Gk_f = Gk.functions
    model_f = model.functions

    if Gk.type not in ["WKB", "mixed"]:
        raise RuntimeError(
            f'compute_QuadSource_integral: attempting to evaluate WKB Levin quadrature, but Gk object is not of "WKB" or "mixed" type [domain={max_z:.5g}, {min_z:.5g}]'
        )

    if Gk_f.theta is None:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB Levin quadrature, but Gk_f.theta is absent (type={Gk.type}, quality={Gk.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}, z_crossover={_extract_z(Gk.crossover_z)}, z_Levin={_extract_z(Gk.Levin_z)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    if Gk_f.WKB_region is None:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB Levin quadrature, but Gk_f.WKB_region is absent (type={Gk.type}, quality={Gk.quality}, lowest numeric z={_extract_z(Gk._numerical_smallest_z)}, primary WKB largest z={_extract_z(Gk._primary_WKB_largest_z)}, z_crossover={_extract_z(Gk.crossover_z)}, z_Levin={_extract_z(Gk.Levin_z)}) [domain={max_z:.5g}, {min_z:.5g}]"
        )

    region_max_z, region_min_z = Gk_f.WKB_region
    if max_z > region_max_z + DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB Levin quadrature, but max_z={max_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a WKB solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )
    if min_z < region_min_z - DEFAULT_FLOAT_PRECISION:
        raise RuntimeError(
            f"compute_QuadSource_integral: attempting to evaluate WKB Levin quadrature, but min_z={min_z:.5g} is out-of-bounds for the region ({region_max_z:.5g}, {region_min_z}:.5g) where a WKB solution is available [domain={max_z:.5g}, {min_z:.5g}]"
        )

    log_min_z = log(1.0 + min_z)
    log_max_z = log(1.0 + max_z)

    x_span = (log_min_z, log_max_z)

    def Levin_f(log_z_source: float) -> float:
        H = model_f.Hubble(exp(log_z_source) - 1.0)
        H_sq = H * H
        f = source_f.source(log_z_source, z_is_log=True)

        return [f / H_sq, 0.0]

    def Levin_theta(log_z_source: float) -> float:
        return Gk_f.theta(log_z_source, z_is_log=True)

    data = adaptive_levin_sincos(
        x_span,
        Levin_f,
        Levin_theta,
        tol=tol,
        chebyshev_order=12,
        notify_label=f"k={k.k_inv_Mpc:.3g}/Mpc, q={q.k_inv_Mpc:.3g}/Mpc, r={r.k_inv_Mpc:.3g}/Mpc @ z_response={z_response.z:.5g}",
    )

    return {
        "data": LevinData(
            num_regions=len(data["regions"]),
            evaluations=data["evaluations"],
            elapsed=data["elapsed"],
        ),
        "value": (1.0 + z_response.z) * data["value"],
    }


class QuadSourceIntegral(DatastoreObject):
    def __init__(
        self,
        payload,
        model,
        z_response: redshift,
        z_source_max: redshift,
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

        self._z_response = z_response
        self._z_source_max = z_source_max

        if payload is None:
            DatastoreObject.__init__(self, None)

            self._total = None
            self._numeric_quad = None
            self._WKB_quad = None
            self._WKB_Levin = None

            self._numeric_quad_data = None
            self._WKB_quad_data = None
            self._WKB_Levin_data = None

            self._compute_time = None
            self._Gk_serial = None
            self._source_serial = None

            self._metadata = {}

        else:
            DatastoreObject.__init__(self, payload["store_id"])

            self._total = payload["total"]
            self._numeric_quad = payload["numeric_quad"]
            self._WKB_quad = payload["WKB_quad"]
            self._WKB_Levin = payload["WKB_Levin"]

            self._Gk_serial = payload["Gk_serial"]

            self._numeric_quad_data = payload["numeric_quad_data"]
            self._WKB_quad_data = payload["WKB_quad_data"]
            self._WKB_Levin_data = payload["WKB_Levin_data"]

            self._compute_time = payload["compute_time"]
            self._Gk_serial = payload["Gk_serial"]
            self._source_serial = payload["source_serial"]

            self._metadata = payload["metadata"]

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
    def z_response(self) -> redshift:
        return self._z_response

    @property
    def total(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._total

    @property
    def numeric_quad(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._numeric_quad

    @property
    def WKB_quad(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._WKB_quad

    @property
    def WKB_Levin(self) -> float:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._WKB_Levin

    @property
    def Gk_serial(self) -> Optional[int]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._Gk_serial

    @property
    def source_serial(self) -> Optional[int]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._source_serial

    @property
    def numeric_quad_data(self) -> Optional[IntegrationData]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._numeric_quad_data

    @property
    def WKB_quad_data(self) -> Optional[IntegrationData]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._WKB_quad_data

    @property
    def WKB_Levin_data(self) -> Optional[LevinData]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._WKB_Levin_data

    @property
    def compute_time(self) -> Optional[float]:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._compute_time

    @property
    def metadata(self) -> dict:
        if self._total is None:
            raise RuntimeError("value has not yet been populated")

        return self._metadata

    @property
    def label(self) -> str:
        return self._label

    @property
    def tags(self) -> List[store_tag]:
        return self._tags

    def compute(self, payload, label: Optional[str] = None):
        if self._total is not None:
            raise RuntimeError(
                "QuadSourceIntegral: compute() called, but value has already been computed"
            )

        # replace label if specified
        if label is not None:
            self._label = label

        source: QuadSource = payload["source"]
        Gk: GkSource = payload["Gk"]

        # TODO: improve compatibility check between source and Gk
        if self._z_response.store_id != Gk.z_response.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied GkSource object does not match specified z_response (z_response={self._z_response.z:.5g}, GkSource object evaluated at z_response={Gk.z_response.z:.5g})"
            )

        if source.z_sample.max.z < self._z_source_max.z - DEFAULT_FLOAT_PRECISION:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied quadratic source term has maximum z_source={source.z_sample.max.z:.5g}, but required value is at least z_source={self._z_source_max.z:.5g}"
            )

        if Gk.z_sample.max.z < self._z_source_max.z - DEFAULT_FLOAT_PRECISION:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied Gk has maximum z_source={Gk.z_sample.max.z:.5g}, but required value is at least z_source={self._z_source_max.z:.5g}"
            )

        if Gk.k.store_id != self._k_exit.k.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied Gk is evaluated for a k-mode that does not match the required value (supplied Gk is for k={Gk.k.k_inv_Mpc:.3g}/Mpc [store_id={Gk.k.store_id}], required value is k={self._k_exit.k.k_inv_Mpc:.3g}/Mpc [store_id]{self._k_exit.k.store_id})"
            )

        if source.q.store_id != self._q_exit.k.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied QuadSource is evaluated for a q-mode that does not match the required value (supplied source is for q={source.q.k_inv_Mpc:.3g}/Mpc [store_id={source.q.store_id}], required value is k={self._q_exit.k.k_inv_Mpc:.3g}/Mpc [store_id]{self._q_exit.k.store_id})"
            )

        if source.r.store_id != self._r_exit.k.store_id:
            raise RuntimeError(
                f"QuadSourceIntegral: supplied QuadSource is evaluated for an r-mode that does not match the required value (supplied source is for r={source.r.k_inv_Mpc:.3g}/Mpc [store_id={source.r.store_id}], required value is k={self._r_exit.k.k_inv_Mpc:.3g}/Mpc [store_id]{self._r_exit.k.store_id})"
            )

        self._compute_ref = compute_QuadSource_integral.remote(
            self._model,
            self._k_exit,
            self._q_exit,
            self._r_exit,
            source,
            Gk,
            self._z_response,
            self._z_source_max,
        )

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

        self._total = payload["total"]
        self._numeric_quad = payload["numeric_quad"]
        self._WKB_quad = payload["WKB_quad"]
        self._WKB_Levin = payload["WKB_Levin"]

        self._Gk_serial = payload["Gk_serial"]
        self._source_serial = payload["source_serial"]

        self._numeric_quad_data = payload["numeric_quad_data"]
        self._WKB_quad_data = payload["WKB_quad_data"]
        self._WKB_Levin_data = payload["WKB_Levin_data"]

        self._compute_time = payload["compute_time"]
        self._metadata = payload["metadata"]
