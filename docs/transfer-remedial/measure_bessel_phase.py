"""
Measure the *existing* ``LiouvilleGreen.bessel_phase`` construction, and record the pre-change
baseline of the transfer-function remedial campaign.

This is a diagnostic script, not a test. It changes nothing and asserts nothing: it prints a
Markdown report, which is committed alongside it as ``baseline-2026-09.md``. Prompt 09 of
``prompts/transfer-remedial`` compares against that file, and a baseline that lives only in a
transcript is not a baseline.

Run from the repository root:

    PYTHONPATH=. ./venv/bin/python docs/transfer-remedial/measure_bessel_phase.py \\
        > docs/transfer-remedial/baseline-2026-09.md

The report has five sections: accuracy, cost, the cost-and-cliff sweep, the standing derivative
regression gate, and the test suite's wall-clock baseline. Measured runtime on the machine that
produced the committed output: about 4 minutes for sections 1-4 (dominated by the four cliff cases
that spend the full 60 s timeout and never terminate), plus about 24 minutes for section 5, whose
last module does not finish -- pass ``--skip-suite`` to omit it, or ``--suite-timeout`` to change
its per-module cap.

Every reference here comes from ``LiouvilleGreen/tests/bessel_reference.py`` -- tier 1 (exact
half-integer closed forms) where the order allows and tier 2 (SciPy) otherwise, with the tier
recorded per case. Nothing is scored against a quantity derived from ``bessel_phase`` itself.
"""

import argparse
import contextlib
import io
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from typing import Optional, Sequence

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from LiouvilleGreen.tests import bessel_reference as br

# ----------------------------------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------------------------------

#: Orders production builds are 1/2 and 5/2 (``main.py:510, 520-528`` with ``b_value = 0.0``); the
#: rest are the orders the plan's acceptance table and the existing tests exercise.
ACCURACY_ORDERS = (0.5, 1.5, 1.75, 2.5, 20.5, 100.5)
ACCURACY_MAX_X = (1.0e3, 1.0e7)

#: ``config/defaults.py``: ``DEFAULT_REL_TOLERANCE = 1e-8``, ``DEFAULT_ABS_TOLERANCE = 1e-10``.
FIXTURE_TOLERANCES = (1.0e-8, 1.0e-10)
#: ``main.py:520-528``.
PRODUCTION_TOLERANCES = (5.0e-14, 1.0e-25)

TOLERANCE_CASES = (
    ("fixture", FIXTURE_TOLERANCES),
    ("production", PRODUCTION_TOLERANCES),
)

#: The cliff sweep of the prompt's section 4.
CLIFF_MAX_X = (1.0e11, 1.0e13, 1.0e14, 1.0e15, 2.0e15, 3.0e15, 8.6e15)
CLIFF_ORDERS = (0.5, 2.5)
CLIFF_TIMEOUT_SECONDS = 60.0

#: Cap on how many sample nodes are scored per case, so the report's cost stays bounded. Nodes are
#: subsampled uniformly in ``log x`` when a case has more; the count actually used is reported.
MAX_SCORED_NODES = 2500

#: Points placed inside the first and the last log-interval of the sample grid.
ENDPOINT_INTERVAL_POINTS = 64

EVAL_TIMING_CALLS = 4000


# ----------------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------------


def construction_min_x(nu: float) -> float:
    """``bessel_phase``'s lower bound, reproduced (``bessel_phase.py:85-88``)."""
    return math.sqrt(nu * nu - 0.25) if nu > 0.5 else 1e-5


def default_sample_points(nu: float, max_x: float) -> int:
    """``bessel_phase.py:113-116``: 250 samples per e-fold, rounded."""
    log_min = math.log(construction_min_x(nu))
    log_max = math.log(max_x)
    return int(round(250 * (log_max - log_min) + 0.5, 0))


class _Counter:
    """Wrap a SciPy special function and count calls to it."""

    def __init__(self, target):
        self._target = target
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self._target(*args, **kwargs)


def build_and_count(nu: float, max_x: float, rtol: float, atol: float):
    """
    Build a ``bessel_phase`` object, timing it and counting the ``jv``/``yv`` calls it makes.

    The counts are obtained by rebinding ``LiouvilleGreen.bessel_phase``'s module-level ``jv`` and
    ``yv`` for the duration of the call, and restoring them afterwards. Nothing in the module is
    edited: this prompt changes no production code.

    ``m(x) = J^2 + Y^2`` is evaluated once per ODE right-hand-side call, once per modulus sample,
    once for the initial condition and once at the match point, so the ODE right-hand-side count is
    ``jv_calls - sample_points - 2`` less the handful the ``root_scalar`` bracket search makes; the
    independent ``solve_ivp`` replication of :func:`replicate_ode` is the clean figure and is
    reported beside it.
    """
    from LiouvilleGreen import bessel_phase as module

    original_jv, original_yv = module.jv, module.yv
    counted_jv, counted_yv = _Counter(original_jv), _Counter(original_yv)
    module.jv, module.yv = counted_jv, counted_yv
    try:
        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            data = module.bessel_phase(nu, max_x, atol=atol, rtol=rtol)
        elapsed = time.perf_counter() - start
    finally:
        module.jv, module.yv = original_jv, original_yv

    return data, elapsed, counted_jv.calls, counted_yv.calls


def replicate_ode(nu: float, max_x: float, rtol: float, atol: float):
    """
    Re-run exactly the ODE ``bessel_phase`` integrates, to read ``nfev`` and the accepted step
    count off the solver -- neither of which the production function returns.

    This is a transcription of ``bessel_phase.py:90-147``, not an approximation of it: same
    right-hand side ``dQ/dlog x = (2/pi)/(x m(x)) - Q``, same DOP853 stepper, same initial
    condition ``Q(log min_x) = asin(J/sqrt(m))/min_x``, same tolerances.
    """
    from scipy.special import jv, yv

    min_x = construction_min_x(nu)

    def m(x):
        return jv(nu, x) ** 2 + yv(nu, x) ** 2

    def rhs(log_x, state):
        x = math.exp(log_x)
        return [(2.0 / math.pi) / x / m(x) - state[0]]

    init = math.asin(jv(nu, min_x) / math.sqrt(m(min_x))) / min_x

    start = time.perf_counter()
    solution = solve_ivp(
        rhs,
        method="DOP853",
        t_span=(math.log(min_x), math.log(max_x)),
        y0=[init],
        atol=atol,
        rtol=rtol,
    )
    return solution.nfev, len(solution.t) - 1, time.perf_counter() - start


# ----------------------------------------------------------------------------------------------
# Section 2 -- accuracy and cost of the existing construction
# ----------------------------------------------------------------------------------------------


def score_point_set(data, nu: float, x: np.ndarray, tier: str) -> dict:
    """
    Score the existing construction over one set of ``x``, returning ``E_theta``, ``E_A`` and the
    phase-derivative relative error, each with the ``x`` at which it occurred.

    The metrics and the reference are ``bessel_reference``'s; the reconstruction is exactly what
    ``bessel_phase``'s own ``bessel_j``/``bessel_y`` do -- ``mod(x) * sin(theta_mod_2pi(x))`` and
    ``-mod(x) * cos(theta_mod_2pi(x))``.
    """
    phase = data["phase"]
    modulus = data["mod"]

    theta = np.array([phase.theta_mod_2pi(value) for value in x])
    amplitude_ours = np.array([modulus(value) for value in x])
    deriv_ours = np.array([phase.theta_deriv(value) for value in x])

    reference = br.reference_bundle(nu, x, tier)

    E_theta, theta_index = br.phase_pair_error(
        np.sin(theta),
        -np.cos(theta),
        reference.J,
        reference.Y,
        reference.amplitude,
    )
    E_A, amplitude_index = br.amplitude_error(amplitude_ours, reference.amplitude)
    E_deriv, deriv_index = br.derivative_error(deriv_ours, reference.theta_deriv)

    return {
        "count": len(x),
        "E_theta": E_theta,
        "E_theta_at": float(x[theta_index]),
        "E_A": E_A,
        "E_A_at": float(x[amplitude_index]),
        "E_deriv": E_deriv,
        "E_deriv_at": float(x[deriv_index]),
    }


def time_evaluation(data, x_probe: np.ndarray) -> dict:
    """Per-call wall-clock of the three phase accessors and of the modulus."""
    phase = data["phase"]
    modulus = data["mod"]

    timings = {}
    for label, function in (
        ("theta_mod_2pi", phase.theta_mod_2pi),
        ("raw_theta", phase.raw_theta),
        ("theta_deriv", phase.theta_deriv),
        ("mod", modulus),
    ):
        start = time.perf_counter()
        for value in x_probe:
            function(value)
        timings[label] = (time.perf_counter() - start) / len(x_probe)

    return timings


def measure_accuracy_case(
    nu: float, max_x: float, label: str, rtol: float, atol: float
) -> dict:
    """One row of section 2: build, cost, chunking, evaluation cost and the three error metrics."""
    tier = br.best_available_tier(nu, max_x)

    data, build_seconds, jv_calls, yv_calls = build_and_count(nu, max_x, rtol, atol)
    nfev, accepted, ode_seconds = replicate_ode(nu, max_x, rtol, atol)

    min_x = data["min_x"]
    sample_points = default_sample_points(nu, max_x)
    log_nodes = np.linspace(math.log(min_x), math.log(max_x), sample_points)

    scored_nodes = log_nodes
    if sample_points > MAX_SCORED_NODES:
        stride = int(math.ceil(sample_points / MAX_SCORED_NODES))
        scored_nodes = log_nodes[::stride]

    nodes = np.exp(scored_nodes)
    midpoints = np.exp(0.5 * (scored_nodes[:-1] + scored_nodes[1:]))
    endpoints = np.exp(
        np.concatenate(
            [
                np.linspace(log_nodes[0], log_nodes[1], ENDPOINT_INTERVAL_POINTS),
                np.linspace(log_nodes[-2], log_nodes[-1], ENDPOINT_INTERVAL_POINTS),
            ]
        )
    )

    probe = np.exp(np.linspace(math.log(min_x), math.log(max_x), EVAL_TIMING_CALLS))

    return {
        "nu": nu,
        "max_x": max_x,
        "tolerance_label": label,
        "rtol": rtol,
        "atol": atol,
        "tier": tier,
        "min_x": min_x,
        "sample_points": sample_points,
        "num_chunks": data["phase"].num_chunks,
        "build_seconds": build_seconds,
        "jv_calls": jv_calls,
        "yv_calls": yv_calls,
        "ode_nfev": nfev,
        "ode_accepted_steps": accepted,
        "ode_seconds": ode_seconds,
        "phi": data["phi"],
        "evaluation": time_evaluation(data, probe),
        "nodes": score_point_set(data, nu, nodes, tier),
        "midpoints": score_point_set(data, nu, midpoints, tier),
        "endpoints": score_point_set(data, nu, endpoints, tier),
    }


# ----------------------------------------------------------------------------------------------
# Section 3 -- the cost and cliff sweep
# ----------------------------------------------------------------------------------------------


def _cliff_worker(nu, max_x, rtol, atol, queue):
    """Subprocess entry point for one cliff-sweep case (module level, so ``spawn`` can pickle it)."""
    try:
        from LiouvilleGreen.bessel_phase import bessel_phase

        start = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            data = bessel_phase(nu, max_x, atol=atol, rtol=rtol)
        queue.put(
            {
                "seconds": time.perf_counter() - start,
                "num_chunks": data["phase"].num_chunks,
                "sample_points": default_sample_points(nu, max_x),
            }
        )
    except (
        BaseException
    ) as error:  # noqa: BLE001 -- a diagnostic must report, not propagate
        queue.put({"error": f"{type(error).__name__}: {error}"})


def cliff_case(nu: float, max_x: float, timeout: float = CLIFF_TIMEOUT_SECONDS) -> dict:
    """
    Build at ``(nu, max_x)`` under a hard wall-clock timeout, in a subprocess.

    A timeout, not a cost measurement: ``RECONCILIATION.md`` C1 shows the build is flat at ~0.1 s
    across five decades and then does not terminate at all above ``x ~ 2.5e15``, because Amos noise
    in ``m(x)`` stops DOP853 from passing its error test and drives the step size to zero. There is
    no number to measure past the cliff, only a ``TIMEOUT``.
    """
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_cliff_worker,
        args=(nu, max_x, PRODUCTION_TOLERANCES[0], PRODUCTION_TOLERANCES[1], queue),
    )

    start = time.perf_counter()
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "nu": nu,
            "max_x": max_x,
            "status": "TIMEOUT",
            "seconds": time.perf_counter() - start,
        }

    result = queue.get() if not queue.empty() else {"error": "no result returned"}
    if "error" in result:
        return {"nu": nu, "max_x": max_x, "status": "ERROR", "detail": result["error"]}

    result.update({"nu": nu, "max_x": max_x, "status": "ok"})
    return result


def rhs_noise_samples(nu: float, centres: Sequence[float], count: int = 5) -> dict:
    """
    Sample ``(2/pi)/(x m(x))`` -- the ODE right-hand side's leading term, identically
    ``1 + O(nu^2/x^2)`` -- at ``count`` *adjacent doubles* from each centre.

    This is the mechanism behind the cliff, recorded directly so the diagnosis lives in the
    repository and not only in a planning document. It is not an integration-cost measurement: the
    stepper stalls because its right-hand side has become noise, and the noise is visible without
    integrating anything.
    """
    from scipy.special import jv, yv

    result = {}
    for centre in centres:
        x = centre
        values = []
        for _ in range(count):
            m = jv(nu, x) ** 2 + yv(nu, x) ** 2
            values.append((2.0 / math.pi) / (x * m) if m > 0.0 else math.inf)
            x = math.nextafter(x, math.inf)
        result[centre] = values

    return result


def scipy_high_order_boundary() -> dict:
    """
    Measure the order-dependent SciPy/Amos boundary for ``jv``/``yv``.

    ``RECONCILIATION.md`` 1 records the 7.13e8 boundary for ``hankel1e`` only. ``jv``/``yv`` are
    what ``bessel_reference``'s tier 2 is built from and what ``test_bessel_phase.py`` scores
    against, so the same measurement is needed for them; it turns out to be the same boundary,
    which is why ``bessel_reference.scipy_reference_max_x`` is order dependent.

    The probe is ``a = sqrt(pi x/2) hypot(J, Y)``, which is ``1 + O(nu^2/x^2)``: at ``x >= 1e8`` it
    must be 1.000000000000 for every order considered here, so any departure is library error.
    """
    from scipy.special import jv, yv

    def a_of(nu, x):
        return math.sqrt(0.5 * math.pi * x) * math.hypot(jv(nu, x), yv(nu, x))

    orders = (2.5, 20.5, 50.5, 70.5, 80.5, 85.5, 88.5, 89.5, 90.5, 100.5, 1000.5)
    sweep = np.geomspace(1.0e8, 2.0e15, 800)

    worst = {}
    for nu in orders:
        errors = np.array([abs(a_of(nu, float(x)) - 1.0) for x in sweep])
        index = int(np.argmax(errors))
        worst[nu] = (float(errors[index]), float(sweep[index]))

    boundary = np.geomspace(6.5e8, 8.5e8, 25)
    detail = [(float(x), a_of(100.5, float(x))) for x in boundary]

    return {"worst": worst, "boundary_detail": detail}


# ----------------------------------------------------------------------------------------------
# Section 6 -- the standing derivative regression gate
# ----------------------------------------------------------------------------------------------


def phase_derivative_gate_margin() -> Sequence[dict]:
    """
    Reproduce ``test_bessel_phase.test_phase_derivative`` (``:121-142``) and record its margin
    against the 1e-6 contract it asserts.

    ``RECONCILIATION.md`` C4: this is the campaign's standing derivative regression gate, it must
    pass from prompt 05 onward, and nobody loosens it. Its sampling starts at ``2 x_0 + 1``, so it
    *excludes* the turning-point interval where ``DRAFT-PLAN.md`` 4.7 locates every maximum -- which
    is why section 2 of this report scores endpoint intervals separately.
    """
    from LiouvilleGreen.bessel_phase import bessel_phase

    rows = []
    for nu in (2.5, 20.5, 100.5):
        max_x = max(10.0 * nu, 1000.0)
        tier = br.best_available_tier(nu, max_x)
        with contextlib.redirect_stdout(io.StringIO()):
            data = bessel_phase(nu, max_x)

        phase = data["phase"]
        min_x = data["min_x"]
        x = np.linspace(2.0 * min_x + 1.0, 0.95 * max_x, 25)

        reference = br.reference_bundle(nu, x, tier)
        ours = np.array([phase.theta_deriv(value) for value in x])
        error, index = br.derivative_error(ours, reference.theta_deriv)

        # the test itself uses jv/yv directly; quote that too so the margin is comparable
        scipy_reference = br.reference_bundle(nu, x, br.TIER_SCIPY)
        as_tested, _ = br.derivative_error(ours, scipy_reference.theta_deriv)

        rows.append(
            {
                "nu": nu,
                "max_x": max_x,
                "tier": tier,
                "x_lo": float(x[0]),
                "x_hi": float(x[-1]),
                "relerr": error,
                "relerr_at": float(x[index]),
                "relerr_as_tested": as_tested,
                "margin": 1.0e-6 / error if error > 0.0 else math.inf,
            }
        )

    return rows


# ----------------------------------------------------------------------------------------------
# Section 7 -- the existing test suite's wall-clock baseline
# ----------------------------------------------------------------------------------------------

TEST_MODULES = (
    "LiouvilleGreen.tests.test_bessel_phase",
    "LiouvilleGreen.tests.test_bessel_reference",
    "LiouvilleGreen.tests.test_range_reduce",
    "LiouvilleGreen.tests.test_three_bessel",
    "LiouvilleGreen.tests.test_3bessel_analytic",
)


def time_test_module(module: str, timeout: float) -> dict:
    """Run one test module in a subprocess under a timeout, and report its wall-clock."""
    environment = dict(os.environ, PYTHONPATH=".")
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", module],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return {
            "module": module,
            "status": "TIMEOUT",
            "seconds": time.perf_counter() - start,
        }

    tail = completed.stderr.strip().splitlines()
    return {
        "module": module,
        "status": (
            "ok" if completed.returncode == 0 else f"FAILED (rc={completed.returncode})"
        ),
        "seconds": time.perf_counter() - start,
        "summary": tail[-1] if tail else "",
    }


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------


def emit(text: str = "") -> None:
    print(text)


def report_header() -> None:
    import datetime

    emit("# Baseline: the existing `bessel_phase` construction, before any change")
    emit()
    emit(
        "Generated by `docs/transfer-remedial/measure_bessel_phase.py`, which is committed"
    )
    emit(
        "alongside this file. **Nothing in production code was modified to produce it.** Prompt 01"
    )
    emit(
        "of `prompts/transfer-remedial`; the tree is the campaign baseline `95cc326` plus this"
    )
    emit("commit's new test-side files.")
    emit()
    emit("| | |")
    emit("|---|---|")
    emit(f"| date | {datetime.date.today().isoformat()} |")
    emit(f"| python | {platform.python_version()} |")
    emit(f"| numpy | {np.__version__} |")
    emit(f"| scipy | {scipy.__version__} |")
    emit(f"| platform | {platform.platform()} |")
    emit()
    emit(
        "`DRAFT-PLAN.md` §9 Stage 1 requires the environment recorded, because §4.4's failure"
    )
    emit(
        "boundaries are properties of the bundled Amos library rather than guarantees."
    )
    emit()
    emit(
        "References are `LiouvilleGreen/tests/bessel_reference.py`'s: tier 1 (`exact` — closed-form"
    )
    emit(
        "half-integer, sharing no library with the object under test) where the order allows, tier 2"
    )
    emit(
        "(`scipy`) otherwise, with the tier recorded per case. Errors are `DRAFT-PLAN.md` §6.1's"
    )
    emit(
        "phase-pair `E_theta`, amplitude `E_A`, and the relative error of `theta_deriv` against the"
    )
    emit("exact oracle `(2/pi)/(x (J^2 + Y^2))`.")
    emit()


def report_accuracy(rows: Sequence[dict]) -> None:
    emit("## 1. Accuracy of the existing construction")
    emit()
    emit(
        "Three point sets are kept separate, as `DRAFT-PLAN.md` §9 Stage 1 requires: **nodes** (the"
    )
    emit(
        "ODE sample grid itself), **interior** (log-interval midpoints, which is where an"
    )
    emit(
        "interpolation error shows), and **endpoints** (64 points inside each of the first and last"
    )
    emit(
        "log-intervals — the turning-point interval that `test_phase_derivative` excludes)."
    )
    emit()
    emit(
        "`E_theta` and `E_A` are dimensionless; `E_deriv` is relative. Each maximum is quoted with"
    )
    emit("the `x` at which it occurred.")
    emit()

    for label, _ in TOLERANCE_CASES:
        emit(f"### 1.{1 if label == 'fixture' else 2} {label} tolerances")
        emit()
        if label == "fixture":
            emit(
                "`config/defaults.py`: `rtol=1e-8`, `atol=1e-10` — what every fixture that calls"
            )
            emit("`bessel_phase()` without tolerance arguments gets.")
        else:
            emit(
                "`main.py:520-528`: `rtol=5e-14`, `atol=1e-25` — what production builds with."
            )
        emit()
        emit(
            "| nu | x_max | tier | set | points | E_theta | at x | E_A | at x | E_deriv | at x |"
        )
        emit("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            if row["tolerance_label"] != label:
                continue
            for set_name in ("nodes", "midpoints", "endpoints"):
                scores = row[set_name]
                emit(
                    f"| {row['nu']} | {row['max_x']:.0e} | `{row['tier']}` | {set_name} | "
                    f"{scores['count']} | {scores['E_theta']:.3e} | {scores['E_theta_at']:.4g} | "
                    f"{scores['E_A']:.3e} | {scores['E_A_at']:.4g} | "
                    f"{scores['E_deriv']:.3e} | {scores['E_deriv_at']:.4g} |"
                )
        emit()


def report_cost(rows: Sequence[dict]) -> None:
    emit("## 2. Construction cost, sample count, chunking, `phi`, and evaluation cost")
    emit()
    emit(
        "`jv`/`yv` calls are counted by rebinding those names in `LiouvilleGreen.bessel_phase` for"
    )
    emit(
        "the duration of the build; `m(x) = J^2 + Y^2` is evaluated once per ODE right-hand side,"
    )
    emit(
        "once per modulus sample, once for the initial condition and a few times in the"
    )
    emit(
        "`root_scalar` bracket search, so `nfev` from an independent `solve_ivp` replication of the"
    )
    emit("same ODE is quoted beside it as the clean figure.")
    emit()
    emit(
        "`phi` is the spurious phase offset of `DRAFT-PLAN.md` §4.3 — the root solve's answer at a"
    )
    emit(
        "match point where the phase is already exact. At production tolerances it *is* the whole"
    )
    emit("phase error; compare the `E_theta` column of §1.2 against `|phi|` here.")
    emit()
    emit(
        "| nu | x_max | tol | build s | jv calls | yv calls | ODE nfev | steps | samples | "
        "chunks | phi | theta_mod_2pi us | raw_theta us | theta_deriv us | mod us |"
    )
    emit("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        timings = row["evaluation"]
        emit(
            f"| {row['nu']} | {row['max_x']:.0e} | {row['tolerance_label']} | "
            f"{row['build_seconds']:.3f} | {row['jv_calls']} | {row['yv_calls']} | "
            f"{row['ode_nfev']} | {row['ode_accepted_steps']} | {row['sample_points']} | "
            f"{row['num_chunks']} | {row['phi']:.6e} | "
            f"{1e6 * timings['theta_mod_2pi']:.2f} | {1e6 * timings['raw_theta']:.2f} | "
            f"{1e6 * timings['theta_deriv']:.2f} | {1e6 * timings['mod']:.2f} |"
        )
    emit()


def report_cliff(cases: Sequence[dict], noise: dict, boundary: dict) -> None:
    emit("## 3. The cost and cliff sweep")
    emit()
    emit(
        "`DRAFT-PLAN.md` §4.7 reports the ODE build taking >600 s at `x_max = 1e13`. **That does"
    )
    emit(
        "not reproduce** (`RECONCILIATION.md` C1): it is well under a second there, flat across"
    )
    emit(
        "five decades, and then does not terminate at all above `x ~ 2.5e15`. So the honest claim"
    )
    emit(
        "prompt 09 can make is that a *cliff* was removed, not that a cost curve was removed."
    )
    emit()
    emit(
        f"Production tolerances, {CLIFF_TIMEOUT_SECONDS:.0f} s hard timeout per case, each case in"
    )
    emit("its own subprocess so a stalled stepper can be killed.")
    emit()
    emit("| nu | x_max | build | samples | chunks |")
    emit("|---|---:|---:|---:|---:|")
    for case in cases:
        if case["status"] == "ok":
            emit(
                f"| {case['nu']} | {case['max_x']:.2g} | {case['seconds']:.3f} s | "
                f"{case['sample_points']} | {case['num_chunks']} |"
            )
        elif case["status"] == "TIMEOUT":
            emit(
                f"| {case['nu']} | {case['max_x']:.2g} | **TIMEOUT** (>{case['seconds']:.0f} s) | "
                f"{default_sample_points(case['nu'], case['max_x'])} | — |"
            )
        else:
            emit(
                f"| {case['nu']} | {case['max_x']:.2g} | {case['status']}: {case.get('detail', '')} | — | — |"
            )
    emit()

    for nu in CLIFF_ORDERS:
        completed = [c["max_x"] for c in cases if c["nu"] == nu and c["status"] == "ok"]
        last = max(completed) if completed else None
        emit(
            f"Last `x_max` at which the existing construction completes, nu={nu}: "
            f"**{last:.2g}**."
            if last is not None
            else f"nu={nu}: no case completed."
        )
    emit()

    emit("### 3.1 The mechanism, measured directly")
    emit()
    emit(
        "The ODE right-hand side is `dQ/dlog x = (2/pi)/(x m(x)) - Q`, and `(2/pi)/(x m(x))` is"
    )
    emit("identically `1 + O(nu^2/x^2)`. Sampled at five *adjacent doubles*:")
    emit()
    emit("| nu | x | five adjacent values of (2/pi)/(x m(x)) |")
    emit("|---|---|---|")
    for nu, samples in noise.items():
        for centre, values in samples.items():
            formatted = ", ".join(f"{value:.6f}" for value in values)
            emit(f"| {nu} | {centre:.3g} | {formatted} |")
    emit()
    emit("So the failure is upstream of the integrator: SciPy/Amos `jv`/`yv` lose")
    emit(
        "argument-reduction accuracy, `m` becomes O(1)-relatively noisy, and DOP853 at"
    )
    emit(
        "`rtol=5e-14` cannot pass its error test on noise, so it drives the step size to zero."
    )
    emit(
        "`np.sin`/`math.sin` are **not** implicated — `RECONCILIATION.md` §1 confirms them"
    )
    emit("correctly rounded to 1e16 — the degradation is confined to Amos.")
    emit()

    emit("### 3.2 The same boundary is order dependent, and it applies to `jv`/`yv`")
    emit()
    emit(
        "`RECONCILIATION.md` §1 records the 7.13e8 failure boundary for `hankel1e` at nu in"
    )
    emit(
        "{100.5, 1000.5}, and 2.247e15 for nu <= 20.5. It did not measure `jv`/`yv`, which are"
    )
    emit(
        "what every reference and every existing test in this area is built from. They fail at the"
    )
    emit(
        "same place. Probe: `a = sqrt(pi x/2) hypot(J, Y)`, identically `1 + O(nu^2/x^2)`, so over"
    )
    emit("`1e8 <= x <= 2e15` it must be 1.000000000000 for every order below.")
    emit()
    emit("| nu | max abs(a - 1) over 1e8..2e15 | at x |")
    emit("|---|---:|---:|")
    for nu, (error, where) in boundary["worst"].items():
        emit(f"| {nu} | {error:.3g} | {where:.4g} |")
    emit()
    emit(
        "The order threshold lies between **85.5** (1.8e-13, i.e. clean) and **88.5** (1.0, i.e."
    )
    emit("destroyed). In `x`, at nu=100.5, the transition is abrupt:")
    emit()
    emit("| x | a |")
    emit("|---:|---:|")
    for x, a in boundary["boundary_detail"]:
        emit(f"| {x:.6g} | {a:.10f} |")
    emit()
    emit(
        "This is a *silent* failure of exactly the kind `DRAFT-PLAN.md` §4.4 describes for"
    )
    emit(
        "`hankel1e`: the values are finite and non-zero, so `isfinite` passes, and they are wrong"
    )
    emit(
        "by up to a factor 100. `bessel_reference.scipy_reference` therefore refuses above"
    )
    emit(
        "`scipy_reference_max_x(nu)` = 7.13e8 for nu > 85.5 and 2e15 otherwise. Pinning both"
    )
    emit("boundaries as tests is prompt 02's.")
    emit()


def report_gate(rows: Sequence[dict]) -> None:
    emit("## 4. The standing derivative regression gate")
    emit()
    emit("`test_bessel_phase.test_phase_derivative` (`:121-142`) already contracts")
    emit(
        "`|theta_deriv/[(2/pi)/(x(J^2+Y^2))] - 1| < 1e-6` at nu in {2.5, 20.5, 100.5} over"
    )
    emit(
        "`x in [2 x_0 + 1, 0.95 x_max]`. `RECONCILIATION.md` C4: it is the campaign's standing"
    )
    emit(
        "regression gate, must pass from prompt 05 onward, and must never be loosened. Its margin"
    )
    emit(
        "today, measured over exactly its own 25 sample points and default tolerances:"
    )
    emit()
    emit(
        "| nu | x_max | x range | tier | relerr | at x | relerr vs `jv`/`yv` (as the test does) | margin to 1e-6 |"
    )
    emit("|---|---:|---|---|---:|---:|---:|---:|")
    for row in rows:
        emit(
            f"| {row['nu']} | {row['max_x']:.0f} | [{row['x_lo']:.4g}, {row['x_hi']:.4g}] | "
            f"`{row['tier']}` | {row['relerr']:.3e} | {row['relerr_at']:.4g} | "
            f"{row['relerr_as_tested']:.3e} | {row['margin']:.1f}x |"
        )
    emit()
    emit(
        "Note the sampling starts at `2 x_0 + 1`, so it **excludes** the turning-point interval"
    )
    emit(
        "where `DRAFT-PLAN.md` §4.7 locates every derivative maximum. §1's `endpoints` rows above"
    )
    emit("are the figures for that interval, and they are worse.")
    emit()


def report_suite(rows: Sequence[dict]) -> None:
    emit("## 5. Wall-clock baseline of the existing `LiouvilleGreen` test suite")
    emit()
    emit("`RECONCILIATION.md` §3.4: the full discovery run")
    emit(
        "`PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` did"
    )
    emit(
        "**not complete within 50 minutes** on the planning machine and was abandoned, dominated by"
    )
    emit(
        "`test_3bessel_analytic.py` rebuilding `bessel_phase` objects per case. So per-module times"
    )
    emit(
        "are recorded instead, as the prompt directs. Later prompts must not read a long run as a"
    )
    emit(
        "regression without comparing against this table, and should prefer per-module runs while"
    )
    emit("iterating.")
    emit()
    emit("| module | status | wall clock | summary |")
    emit("|---|---|---:|---|")
    for row in rows:
        seconds = row["seconds"]
        shown = f"{seconds:.1f} s" if seconds < 120 else f"{seconds / 60.0:.1f} min"
        emit(
            f"| `{row['module'].rsplit('.', 1)[1]}` | {row['status']} | {shown} | "
            f"{row.get('summary', '')} |"
        )
    emit()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-suite",
        action="store_true",
        help="skip section 5 (the test-suite wall-clock baseline), which is the slow part",
    )
    parser.add_argument(
        "--suite-timeout",
        type=float,
        default=1500.0,
        help="per-module timeout in seconds for section 5 (default 1500)",
    )
    arguments = parser.parse_args(argv)

    report_header()

    accuracy_rows = []
    for label, (rtol, atol) in TOLERANCE_CASES:
        for nu in ACCURACY_ORDERS:
            for max_x in ACCURACY_MAX_X:
                print(f"... case nu={nu} x_max={max_x:.0e} {label}", file=sys.stderr)
                accuracy_rows.append(
                    measure_accuracy_case(nu, max_x, label, rtol, atol)
                )

    report_accuracy(accuracy_rows)
    report_cost(accuracy_rows)

    cliff = []
    for nu in CLIFF_ORDERS:
        for max_x in CLIFF_MAX_X:
            print(f"... cliff nu={nu} x_max={max_x:.2g}", file=sys.stderr)
            cliff.append(cliff_case(nu, max_x))

    noise = {nu: rhs_noise_samples(nu, (1.0e15, 3.0e15, 1.0e16)) for nu in CLIFF_ORDERS}
    print("... scipy high-order boundary", file=sys.stderr)
    boundary = scipy_high_order_boundary()
    report_cliff(cliff, noise, boundary)

    print("... derivative gate", file=sys.stderr)
    report_gate(phase_derivative_gate_margin())

    if arguments.skip_suite:
        emit("## 5. Wall-clock baseline of the existing `LiouvilleGreen` test suite")
        emit()
        emit("*Skipped in this run (`--skip-suite`).*")
        emit()
    else:
        suite = []
        for module in TEST_MODULES:
            print(f"... timing {module}", file=sys.stderr)
            suite.append(time_test_module(module, arguments.suite_timeout))
        report_suite(suite)

    return 0


if __name__ == "__main__":
    sys.exit(main())
