"""
Uniform runners for the AdaptiveLevin performance campaign.

Every runner returns a flat record dict with a common schema so that Levin,
plain QUADPACK QAGS (`scipy.integrate.quad`) and QUADPACK QAWO
(`quad(..., weight='sin'|'cos')`) can be tabulated side by side.

Two measurement passes are used deliberately:

  * a *timing* pass with the raw, unwrapped callables, so that the
    instrumentation does not inflate the wall time of trivial integrands;
  * a *counting* pass with wrapped callables, which reports how many times the
    amplitude and the phase were evaluated.

Wall-clock budgets are enforced from inside the integrand (QUADPACK cannot be
interrupted from outside), so a diverging brute-force run is recorded as a
timeout rather than hanging the campaign.
"""

import time
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad

REPO = "/Users/ds283/Documents/Code/SecondaryGWKit"


class BudgetExceeded(Exception):
    """Raised from inside an integrand when the wall-clock budget is spent."""


class _Budget:
    """Wall-clock budget plus evaluation counters, shared across callables."""

    def __init__(self, seconds=None):
        self.seconds = seconds
        self.n_amp = 0
        self.n_phase = 0
        self._t0 = time.perf_counter()

    def _check(self):
        if self.seconds is not None and time.perf_counter() - self._t0 > self.seconds:
            raise BudgetExceeded

    def amp(self, fn):
        def wrapper(x):
            self.n_amp += 1
            if self.n_amp % 256 == 0:
                self._check()
            return fn(x)

        return wrapper

    def phase(self, fn):
        def wrapper(x):
            self.n_phase += 1
            if self.n_phase % 256 == 0:
                self._check()
            return fn(x)

        return wrapper


def _time_call(thunk, budget_seconds, max_repeats=3, repeat_if_under=0.5):
    """
    Run `thunk` and return (result, best_wall_time, status).

    A cheap call is repeated (up to `max_repeats`) and the minimum time kept; an
    expensive one is timed once. `thunk` must accept a single `_Budget`.
    """
    b = _Budget(seconds=budget_seconds)
    t0 = time.perf_counter()
    try:
        result = thunk(b)
    except BudgetExceeded:
        return None, time.perf_counter() - t0, "timeout", b
    best = time.perf_counter() - t0

    if best < repeat_if_under:
        for _ in range(max_repeats - 1):
            b2 = _Budget(seconds=budget_seconds)
            t0 = time.perf_counter()
            try:
                thunk(b2)
            except BudgetExceeded:
                break
            best = min(best, time.perf_counter() - t0)

    return result, best, "ok", b


def _errors(value, reference):
    if value is None or not np.isfinite(value):
        return np.nan, np.nan
    abs_err = abs(float(value) - float(reference))
    rel_err = abs_err / abs(float(reference)) if reference != 0.0 else np.nan
    return abs_err, rel_err


def _base_record(problem, method, **kw):
    rec = {
        "problem": problem.name,
        "tier": problem.tier,
        "method": method,
        "omega": problem.omega,
        "reference": float(problem.reference),
        "n_osc": problem.n_oscillations,
    }
    rec.update(kw)
    return rec


# ---------------------------------------------------------------------------
# Levin
# ---------------------------------------------------------------------------


def run_levin(
    problem,
    atol=1e-15,
    rtol=1e-10,
    chebyshev_order=12,
    phase_mode="mod2pi",
    depth_max=20,
    budget_seconds=120.0,
    keep_regions=False,
):
    """
    Run `adaptive_levin_sincos` on `problem`.

    phase_mode:
      'mod2pi'  -- supply theta, theta_mod_2pi and theta_deriv (production path)
      'naive'   -- supply theta only (tests loss of significance at large phase)
      'no_deriv'-- supply theta and theta_mod_2pi, let the driver differentiate
    """
    import sys

    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from AdaptiveLevin import adaptive_levin_sincos

    def build_theta(b, count=False):
        wrap = b.phase if count else (lambda f: f)
        d = {"theta": wrap(problem.theta)}
        if phase_mode in ("mod2pi", "no_deriv"):
            d["theta_mod_2pi"] = wrap(problem.theta_mod_2pi)
        if phase_mode == "mod2pi":
            d["theta_deriv"] = wrap(problem.theta_deriv)
        return d

    def make_thunk(count):
        def thunk(b):
            wrap = b.amp if count else (lambda f: f)
            f = [wrap(g) for g in problem.f]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return adaptive_levin_sincos(
                    problem.x_span,
                    f,
                    theta=build_theta(b, count=count),
                    atol=atol,
                    rtol=rtol,
                    chebyshev_order=chebyshev_order,
                    depth_max=depth_max,
                    notify_interval=10**9,
                )

        return thunk

    data, wall, status, _ = _time_call(make_thunk(False), budget_seconds)

    n_amp = n_phase = np.nan
    if status == "ok":
        _, _, _, b = _time_call(
            make_thunk(True), budget_seconds, max_repeats=1, repeat_if_under=0.0
        )
        n_amp, n_phase = b.n_amp, b.n_phase

    value = data["value"] if status == "ok" else None
    abs_err, rel_err = _errors(value, problem.reference)

    rec = _base_record(
        problem,
        "levin",
        atol=atol,
        rtol=rtol,
        chebyshev_order=chebyshev_order,
        phase_mode=phase_mode,
        value=value,
        abs_err=abs_err,
        rel_err=rel_err,
        time_s=wall,
        n_amp_eval=n_amp,
        n_phase_eval=n_phase,
        status=status,
    )
    if status == "ok":
        rec.update(
            num_regions=data["num_regions"],
            num_simple_regions=data["num_simple_regions"],
            levin_solves=data["evaluations"],
            num_direct_solves=data.get("num_direct_solves", np.nan),
            max_depth=data["max_depth"],
            num_SVD_errors=data["num_SVD_errors"],
            num_order_changes=data["num_order_changes"],
            chebyshev_min_order=data["chebyshev_min_order"],
            depth_saturated=int(data["max_depth"] >= depth_max),
            direct_fraction=(
                data["num_simple_regions"] / data["num_regions"]
                if data["num_regions"]
                else np.nan
            ),
        )
        if keep_regions:
            rec["_regions"] = data["regions"]
    return rec


# ---------------------------------------------------------------------------
# QUADPACK QAGS -- the brute-force baseline
# ---------------------------------------------------------------------------


def run_quad(problem, epsabs=1e-13, epsrel=1e-10, limit=200, budget_seconds=120.0):
    def make_thunk(count):
        def thunk(b):
            integrand = b.amp(problem.integrand) if count else problem.integrand
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IntegrationWarning)
                val, err = quad(
                    integrand,
                    problem.x_span[0],
                    problem.x_span[1],
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=limit,
                )
            return val, err, [str(w.message)[:60] for w in caught]

        return thunk

    out, wall, status, _ = _time_call(make_thunk(False), budget_seconds)

    n_amp = np.nan
    if status == "ok":
        _, _, _, b = _time_call(
            make_thunk(True), budget_seconds, max_repeats=1, repeat_if_under=0.0
        )
        n_amp = b.n_amp

    value = out[0] if status == "ok" else None
    quad_err = out[1] if status == "ok" else np.nan
    warned = int(bool(out[2])) if status == "ok" else 0
    abs_err, rel_err = _errors(value, problem.reference)

    return _base_record(
        problem,
        "quad",
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
        value=value,
        abs_err=abs_err,
        rel_err=rel_err,
        time_s=wall,
        n_amp_eval=n_amp,
        n_phase_eval=0,
        status=status,
        quad_reported_err=quad_err,
        quad_warned=warned,
    )


# ---------------------------------------------------------------------------
# QUADPACK QAWO -- the oscillatory rule, where the phase is linear
# ---------------------------------------------------------------------------


def run_qawo(
    problem, epsabs=1e-13, epsrel=1e-10, limit=200, maxp1=100, budget_seconds=120.0
):
    if problem.qawo is None:
        return None
    amp, weight, wvar = problem.qawo

    def make_thunk(count):
        def thunk(b):
            integrand = b.amp(amp) if count else amp
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IntegrationWarning)
                val, err = quad(
                    integrand,
                    problem.x_span[0],
                    problem.x_span[1],
                    weight=weight,
                    wvar=wvar,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=limit,
                    maxp1=maxp1,
                )
            return val, err, [str(w.message)[:60] for w in caught]

        return thunk

    out, wall, status, _ = _time_call(make_thunk(False), budget_seconds)

    n_amp = np.nan
    if status == "ok":
        _, _, _, b = _time_call(
            make_thunk(True), budget_seconds, max_repeats=1, repeat_if_under=0.0
        )
        n_amp = b.n_amp

    value = out[0] if status == "ok" else None
    quad_err = out[1] if status == "ok" else np.nan
    warned = int(bool(out[2])) if status == "ok" else 0
    abs_err, rel_err = _errors(value, problem.reference)

    return _base_record(
        problem,
        "qawo",
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
        maxp1=maxp1,
        value=value,
        abs_err=abs_err,
        rel_err=rel_err,
        time_s=wall,
        n_amp_eval=n_amp,
        n_phase_eval=0,
        status=status,
        quad_reported_err=quad_err,
        quad_warned=warned,
    )
