"""Phase-noise injection on A2's realistic fixture: `levin-phase-floor` prompt 02.

Is a non-smooth error in a phase group's phase enough to drive the Levin driver to its depth cap at
production's tolerance, at an amplitude near `rtol`, and in which groups? This script adds
deterministic noise of amplitude delta to the composed `theta_mod_2pi` of the phase groups that
`phase_group_Levin_integral` integrates, on the eight fixtures of the prompt, and records what the
driver does. The prompt is `prompts/levin-phase-floor/02-phase-noise-injection.md`; its section 4
fixes the predictions before anything here runs, and `--report` scores them as written.

Runs from the repository root with no Ray and no datastore. The prompt's three entry points, each
with a report from its own ledger:

    PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/noise_injection.py --probe
    PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/noise_injection.py --probe-report \\
        --checkpoint <the probe run's checkpoint.jsonl>
    PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/noise_injection.py --run
    PYTHONPATH=. ./venv/bin/python docs/levin-phase-floor/noise_injection.py --report \\
        --checkpoint <the sweep run's checkpoint.jsonl>

* `--probe` runs the prompt's step 2 (the wrapper changes `total` at delta = 1e-4, and is
  bit-identical to the unwrapped integral at delta = 0, checked on all eight cases) and step 3
  (the timing probe). A noised cell can take tens of minutes, so the probe is a registered run
  (slug `noise-probe`), launched as `--run` is. `--probe-report` writes its tables and the
  projection of the sweep's wall time from its ledger alone.
* `--run` registers the sweep with `runregistrykit` (campaign `levin-phase-floor`, prompt `02`,
  slug `noise`) with its unit ledger in the run directory, then launches the worker detached
  under `nohup` in a new session, its stdout and stderr in the run directory, records the worker's
  pid with `Run.heartbeat(pid=...)`, prints the run id and pid, and returns.
* `--report` writes every table, and the scoring of N1-N4, from the checkpoint alone. A record
  whose script SHA-256 is not this script's is discarded with a notice, never re-stamped.

THE INJECTION. `ComputeTargets/QuadSourceIntegral.py` imports `build_phase_groups` into its own
namespace and `phase_group_Levin_integral` calls it through that name, so replacing the module
attribute `ComputeTargets.QuadSourceIntegral.build_phase_groups` reaches every Levin sub-interval.
The wrapper returns the real groups with `theta_mod_2pi` replaced, by `dataclasses.replace`, for
every group whose label is in the arm's set; `theta` and `theta_deriv` are left alone. A second,
pass-through wrapper on `ComputeTargets.QuadSourceIntegral.adaptive_levin_sincos` records each
Levin call's `max_depth` and region count by group label, which the stored metadata does not
(it keeps only the maximum depth over all groups). Both are restored in a `finally`, after every
cell. No repository file outside `docs/levin-phase-floor/` is modified;
`docs/handover/realistic_large_x.py` and `ComputeTargets/tests/test_quadsource_integral.py` are
imported, never edited.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import platform
import signal
import struct
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402
import scipy  # noqa: E402

# `ComputeTargets/__init__.py` re-exports a class named `QuadSourceIntegral`, which shadows the
# submodule as an attribute of the package, so `import ... as` would bind the class: take the
# module object itself
QSI = importlib.import_module("ComputeTargets.QuadSourceIntegral")
from AdaptiveLevin.levin_quadrature import DEFAULT_LEVIN_MAX_DEPTH  # noqa: E402
from config.defaults import DEFAULT_QUADRATURE_RTOL  # noqa: E402


def _load_realistic_large_x():
    """`docs/handover/` is not a package: load the module from its file, under its own name. On
    import it puts `docs/radiation-oracle/` on `sys.path` and hashes itself; it writes nothing
    unless run as `__main__`."""
    path = os.path.join(REPO_ROOT, "docs", "handover", "realistic_large_x.py")
    spec = importlib.util.spec_from_file_location("realistic_large_x", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["realistic_large_x"] = module
    spec.loader.exec_module(module)
    return module


RLX = _load_realistic_large_x()
from ComputeTargets.tests.test_quadsource_integral import (  # noqa: E402
    Case,
    SHAPES,
    w_of_b,
)

# ================================================================================================
# the construction (prompt section 1)

# A2's REF_ATOL: keeps the length-share `local_atol` branch dead, as it is in production
ATOL = 1e-45
RTOL = DEFAULT_QUADRATURE_RTOL
CAP = DEFAULT_LEVIN_MAX_DEPTH

FLAVOURS = ("exact", "realistic")
SHAPE_NAMES = ("together", "q-smooth")
X_RESPS = (980.0, 1.0e4)
CASES = tuple(
    (flavour, shape, x_resp)
    for flavour in FLAVOURS
    for shape in SHAPE_NAMES
    for x_resp in X_RESPS
)
LADDER = (0.0, 1e-10, 1e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 1e-5, 1e-4)
ARM_B_CASE = ("realistic", "q-smooth", 1.0e4)
ARM_B_DELTAS = (1e-9, 1e-7, 1e-5)
ARM_C_DELTAS = (1e-9, 1e-7, 1e-5)
NOISES = ("white", "staircase")
ALL = "all"

# abscissae inside each Levin sub-interval at which a group's theta_deriv is read for arm B's
# slow/fast labelling, as fractions of the sub-interval in log(1+z)
CLASSIFY_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
# a group is "slow" where |Psi'| is below this fraction of the fastest group's |Psi'| in the same
# sub-interval. For two groups G+-T the fastest is |theta_G'| + |theta_T'|, and for four it is
# the sum of all three, so this is |Psi'| / (sum of its constituents' |theta'|): a group whose
# constituents cancel by more than half.
SLOW_FRACTION = 0.5

SHAPE_BY_NAME = {shape.name: shape for shape in SHAPES}

INPUT_FILES = (
    "docs/handover/realistic_large_x.py",
    "ComputeTargets/tests/test_quadsource_integral.py",
    "ComputeTargets/tests/test_tk_source_functions.py",
    "ComputeTargets/tests/test_phase_groups.py",
    "ComputeTargets/phase_groups.py",
    "ComputeTargets/QuadSourceIntegral.py",
    "AdaptiveLevin/levin_quadrature.py",
    "config/defaults.py",
)


def sha256_of(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


SCRIPT_SHA256 = sha256_of(os.path.abspath(__file__))


def git_provenance() -> dict:
    def run(args):
        try:
            return subprocess.run(
                args,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "git_head": run(["git", "rev-parse", "HEAD"]) or "unknown",
        "git_dirty": bool(run(["git", "status", "--porcelain"])),
    }


def provenance() -> dict:
    return {
        "script": "docs/levin-phase-floor/noise_injection.py",
        "script_sha256": SCRIPT_SHA256,
        **git_provenance(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "inputs_sha256": {
            path: sha256_of(os.path.join(REPO_ROOT, path)) for path in INPUT_FILES
        },
        "host": platform.node(),
        "atol": ATOL,
        "rtol": RTOL,
        "depth_cap": CAP,
    }


def provenance_lines(prov: dict, prefix: str = "") -> list:
    lines = [
        f"{prefix}script {prov['script']} sha256 {prov['script_sha256']}",
        f"{prefix}git HEAD {prov['git_head']}"
        f"{' (tree not clean)' if prov['git_dirty'] else ' (clean)'}",
        f"{prefix}Python {prov['python']}, numpy {prov['numpy']}, scipy {prov['scipy']}, "
        f"host {prov['host']}",
        f"{prefix}atol {prov['atol']:g}, rtol {prov['rtol']:g} (DEFAULT_QUADRATURE_RTOL), "
        f"depth cap {prov['depth_cap']} (DEFAULT_LEVIN_MAX_DEPTH)",
    ]
    for path, digest in prov["inputs_sha256"].items():
        lines.append(f"{prefix}input {path} sha256 {digest}")
    return lines


def loadavg() -> list:
    try:
        return list(os.getloadavg())
    except OSError:
        return [float("nan")] * 3


# ================================================================================================
# the noise


def white_xi(label: str, x: float) -> float:
    """xi = 2u - 1, u = the first eight bytes of blake2b(label + pack('<d', x)) read as an
    unsigned big-endian integer, divided by 2^64."""
    digest = hashlib.blake2b(label.encode() + struct.pack("<d", float(x))).digest()
    return 2.0 * (int.from_bytes(digest[:8], "big") / 2.0**64) - 1.0


def noised_group(group, delta: float, noise: str):
    """The group with `theta_mod_2pi` replaced by theta_mod_2pi(x) + delta xi(x)."""
    base = group.theta_mod_2pi
    if noise == "white":
        label = group.label

        def theta_mod_2pi(x):
            return base(x) + delta * white_xi(label, x)

    elif noise == "staircase":
        if not delta > 0.0:
            raise ValueError(
                "staircase noise needs delta > 0: its grid step is 2 delta"
            )
        raw = group.theta
        step = 2.0 * delta

        def theta_mod_2pi(x):
            # delta xi = step round(theta_raw / step) - theta_raw: quantisation of the group's
            # own raw phase to a grid of step 2 delta, a sawtooth of amplitude delta
            t = raw(x)
            return base(x) + (step * round(t / step) - t)

    else:
        raise ValueError(f"unknown noise model {noise!r}")
    return dataclasses.replace(group, theta_mod_2pi=theta_mod_2pi)


_REAL_BUILD_PHASE_GROUPS = QSI.build_phase_groups
_REAL_ADAPTIVE_LEVIN = QSI.adaptive_levin_sincos


@contextmanager
def instrumented(delta: float, noise: str, labels, classify: bool = False):
    """
    Replace `QSI.build_phase_groups` by the noising wrapper and `QSI.adaptive_levin_sincos` by a
    pass-through observer, for the duration of one cell; both are restored in `finally`.
    `labels` is None (every group) or a set of group labels. Yields the list the observer fills,
    one entry per Levin call.
    """
    calls = []

    def build_phase_groups(*args, **kwargs):
        groups = _REAL_BUILD_PHASE_GROUPS(*args, **kwargs)
        return [
            (
                noised_group(group, delta, noise)
                if labels is None or group.label in labels
                else group
            )
            for group in groups
        ]

    def adaptive_levin_sincos(x_span, f, *args, **kwargs):
        data = _REAL_ADAPTIVE_LEVIN(x_span, f, *args, **kwargs)
        # phase_group_Levin_integral's notify_label ends ", group <label>"; the analytic
        # oracle's (`_three_bessel_Levin`, "analytic J1" ...) reaches this same module attribute
        # and is recorded as kind "analytic", never as a phase group
        notify = str(kwargs.get("notify_label", ""))
        is_group = ", group " in notify
        entry = {
            "kind": "group" if is_group else "analytic",
            "label": notify.rsplit(", group ", 1)[-1] if is_group else notify,
            "x_span": [float(x_span[0]), float(x_span[1])],
            "max_depth": int(data["max_depth"]),
            "num_regions": int(data["num_regions"]),
            "num_simple_regions": int(data["num_simple_regions"]),
            "evaluations": int(data["evaluations"]),
            "elapsed": float(data["elapsed"]),
            "converged": bool(data["converged"]),
            "phase_limited": bool(data["phase_limited"]),
        }
        if classify and is_group:
            # read only after the driver has returned, so that nothing the driver sees changes
            theta = kwargs["theta"]
            a, b = x_span
            entry["theta_deriv"] = [
                [a + frac * (b - a), float(theta["theta_deriv"](a + frac * (b - a)))]
                for frac in CLASSIFY_FRACTIONS
            ]
        calls.append(entry)
        return data

    QSI.build_phase_groups = build_phase_groups
    QSI.adaptive_levin_sincos = adaptive_levin_sincos
    try:
        yield calls
    finally:
        QSI.build_phase_groups = _REAL_BUILD_PHASE_GROUPS
        QSI.adaptive_levin_sincos = _REAL_ADAPTIVE_LEVIN


def assert_restored():
    if (
        QSI.build_phase_groups is not _REAL_BUILD_PHASE_GROUPS
        or QSI.adaptive_levin_sincos is not _REAL_ADAPTIVE_LEVIN
    ):
        raise RuntimeError("a wrapper was left installed after a cell")


# ================================================================================================
# one cell


def build_case(flavour: str, shape_name: str, x_resp: float):
    """
    The build of `realistic_large_x.run_cell` before its `case.run` call, which runs at
    REF_RTOL = 1e-12 and so cannot be called as it stands: b = 0, `scaled_shape`, `seed_fixtures`
    (which writes the class-level cache `Case._fixtures`), and the `Case`. The seam is closed by
    not calling `open_the_seam`.
    """
    w = w_of_b(0.0)
    shape = RLX.scaled_shape(SHAPE_BY_NAME[shape_name], x_resp)
    RLX.seed_fixtures(w, shape, x_resp)
    return Case(b=0.0, shape=shape, x_resp=x_resp, exact=(flavour == "exact"))


def run_cell(
    flavour: str,
    shape_name: str,
    x_resp: float,
    delta: float,
    noise: str,
    labels=None,
    instrument: bool = True,
    classify: bool = False,
) -> dict:
    """One integral, built from scratch and timed. Returns a flat, JSON-able record."""
    assert_restored()
    load_before = loadavg()
    t0 = time.perf_counter()
    case = build_case(flavour, shape_name, x_resp)
    t1 = time.perf_counter()
    c1 = time.process_time()
    calls = None
    if instrument:
        with instrumented(
            delta, noise, None if labels is None else set(labels), classify
        ) as calls:
            out = case.run(atol=ATOL, rtol=RTOL)
    else:
        out = case.run(atol=ATOL, rtol=RTOL)
    c2 = time.process_time()
    t2 = time.perf_counter()
    assert_restored()

    levin = out["WKB_Levin_data"]
    subintervals = out["metadata"]["WKB_Levin"]["subintervals"]
    group_elapsed = {}
    for item in subintervals:
        for group in item["groups"]:
            group_elapsed[group["label"]] = (
                group_elapsed.get(group["label"], 0.0) + group["elapsed"]
            )

    record = {
        "flavour": flavour,
        "shape": shape_name,
        "x_resp": x_resp,
        "delta": delta,
        "noise": noise,
        "labels": ALL if labels is None else sorted(labels),
        "instrumented": instrument,
        "k": case.shape.k,
        "q": case.shape.q,
        "r": case.shape.r,
        "z_resp": case.z_resp,
        "total": float(out["total"]),
        "total_abserr": float(out["total_abserr"]),
        "total_converged": bool(out["total_converged"]),
        "total_phase_limited": bool(out["total_phase_limited"]),
        "numeric_quad": float(out["numeric_quad"]),
        "WKB_Levin": float(out["WKB_Levin"]),
        "analytic_rad": float(out["analytic_rad"]),
        "max_depth": None if levin is None else int(levin.max_depth),
        "num_regions": None if levin is None else int(levin.num_regions),
        "num_simple_regions": None if levin is None else int(levin.num_simple_regions),
        "evaluations": None if levin is None else int(levin.evaluations),
        "Levin_elapsed": None if levin is None else float(levin.elapsed),
        "group_elapsed": group_elapsed,
        "subintervals": subintervals,
        "levin_calls": calls,
        "setup_time": t1 - t0,
        "wall_time": t2 - t1,
        "cpu_time": c2 - c1,
        "loadavg_before": load_before,
        "loadavg_after": loadavg(),
        "status": "ok",
    }
    return record


TIMING_KEYS = {
    "elapsed",
    "Levin_elapsed",
    "group_elapsed",
    "setup_time",
    "wall_time",
    "cpu_time",
    "loadavg_before",
    "loadavg_after",
    "levin_calls",
    "instrumented",
    "compute_time",
    # labels of the record, not results of the integral
    "tag",
    "arm",
    "provenance",
    "rel_change_vs_delta0",
}


def deterministic_view(record: dict):
    """The record with every timing and instrumentation field removed: what bit-identity is
    checked on. Floats survive `json.dumps` exactly (it writes `repr`)."""

    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in TIMING_KEYS}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return json.dumps(strip(record), sort_keys=True)


def first_difference(a: dict, b: dict) -> str:
    va, vb = json.loads(deterministic_view(a)), json.loads(deterministic_view(b))

    def walk(x, y, path):
        if type(x) is not type(y):
            return f"{path}: {x!r} vs {y!r}"
        if isinstance(x, dict):
            for key in sorted(set(x) | set(y)):
                if key not in x or key not in y:
                    return f"{path}.{key}: present in one only"
                found = walk(x[key], y[key], f"{path}.{key}")
                if found:
                    return found
            return ""
        if isinstance(x, list):
            if len(x) != len(y):
                return f"{path}: length {len(x)} vs {len(y)}"
            for i, (p, q) in enumerate(zip(x, y)):
                found = walk(p, q, f"{path}[{i}]")
                if found:
                    return found
            return ""
        return "" if x == y else f"{path}: {x!r} vs {y!r}"

    return walk(va, vb, "record") or "none"


# ================================================================================================
# the sweep's units


def unit_key(arm, noise, labels, flavour, shape, x_resp, delta) -> str:
    label = labels if isinstance(labels, str) else "+".join(sorted(labels))
    return f"{arm}|{noise}|{label}|{flavour}|{shape}|{x_resp:g}|{delta:g}"


BASELINE_UNITS = [
    dict(
        arm="A",
        noise="white",
        labels=ALL,
        flavour=f,
        shape=s,
        x_resp=x,
        delta=0.0,
    )
    for f, s, x in CASES
]
CLASSIFY_UNIT = dict(
    arm="classify",
    noise="white",
    labels=ALL,
    flavour=ARM_B_CASE[0],
    shape=ARM_B_CASE[1],
    x_resp=ARM_B_CASE[2],
    delta=0.0,
)


def noised_units(arm_b_labels) -> list:
    units = []
    for f, s, x in CASES:
        for delta in LADDER[1:]:
            units.append(
                dict(
                    arm="A",
                    noise="white",
                    labels=ALL,
                    flavour=f,
                    shape=s,
                    x_resp=x,
                    delta=delta,
                )
            )
        for delta in ARM_C_DELTAS:
            units.append(
                dict(
                    arm="C",
                    noise="staircase",
                    labels=ALL,
                    flavour=f,
                    shape=s,
                    x_resp=x,
                    delta=delta,
                )
            )
    f, s, x = ARM_B_CASE
    for label in arm_b_labels:
        for delta in ARM_B_DELTAS:
            for noise in NOISES:
                units.append(
                    dict(
                        arm="B",
                        noise=noise,
                        labels=[label],
                        flavour=f,
                        shape=s,
                        x_resp=x,
                        delta=delta,
                    )
                )
    flavour_order = {name: i for i, name in enumerate(FLAVOURS)}
    shape_order = {name: i for i, name in enumerate(SHAPE_NAMES)}
    arm_order = {"A": 0, "C": 1, "B": 2}
    # cheapest first: ascending delta, exact before realistic, small x before large
    units.sort(
        key=lambda u: (
            u["delta"],
            flavour_order[u["flavour"]],
            u["x_resp"],
            shape_order[u["shape"]],
            arm_order[u["arm"]],
            u["noise"],
            str(u["labels"]),
        )
    )
    return units


def key_of(unit: dict) -> str:
    return unit_key(
        unit["arm"],
        unit["noise"],
        unit["labels"],
        unit["flavour"],
        unit["shape"],
        unit["x_resp"],
        unit["delta"],
    )


def labels_of(record: dict) -> list:
    """Every group label the case produced, in first-seen order."""
    seen = []
    for item in record["subintervals"]:
        for group in item["groups"]:
            if group["label"] not in seen:
                seen.append(group["label"])
    return seen


def run_unit(unit: dict) -> dict:
    try:
        record = run_cell(
            unit["flavour"],
            unit["shape"],
            unit["x_resp"],
            unit["delta"],
            unit["noise"],
            labels=None if unit["labels"] == ALL else unit["labels"],
            classify=(unit["arm"] == "classify"),
        )
    except Exception as exc:  # noqa: BLE001
        # a cell that raises is a result, recorded durably, and the sweep continues
        assert_restored()
        record = {
            "flavour": unit["flavour"],
            "shape": unit["shape"],
            "x_resp": unit["x_resp"],
            "delta": unit["delta"],
            "noise": unit["noise"],
            "labels": unit["labels"],
            "status": "raised",
            "error": f"{type(exc).__name__}: {exc}",
        }
    record["arm"] = unit["arm"]
    return record


class BaselineAtCap(RuntimeError):
    pass


def sweep(record, known: dict, heartbeat, log=print) -> None:
    """
    Every unit not already in `known`, in order: the eight delta = 0 cells, the classification
    of arm B's labels, then every noised cell cheapest-first. `record(key, data)` appends one unit
    to the ledger; `heartbeat(stage=..., units_total=...)` refreshes the registry.

    A delta = 0 cell at the cap is the prompt's section 5 stop ("the baseline is not clean"): the
    sweep stops there rather than spend hours on a measurement whose baseline is void.
    """
    started = time.perf_counter()
    done = dict(known)

    def do(unit):
        key = key_of(unit)
        if key in done:
            return done[key]["data"]
        heartbeat(stage=f"running {key}")
        data = run_unit(unit)
        baseline = done.get(
            unit_key(
                "A",
                "white",
                ALL,
                unit["flavour"],
                unit["shape"],
                unit["x_resp"],
                0.0,
            )
        )
        if data["status"] == "ok" and baseline is not None and unit["delta"] > 0.0:
            t0 = baseline["data"]["total"]
            data["rel_change_vs_delta0"] = abs(data["total"] - t0) / abs(t0)
        data["provenance"] = PROVENANCE
        entry = record(key, data)
        done[key] = entry
        note = (
            f"max_depth {data['max_depth']} regions {data['num_regions']} "
            f"wall {data['wall_time']:.1f} s cpu {data['cpu_time']:.1f} s "
            f"load {data['loadavg_after'][0]:.1f}"
            if data["status"] == "ok"
            else f"RAISED {data['error']}"
        )
        log(
            f"[{time.perf_counter() - started:8.0f} s] {key}: {note}",
            flush=True,
        )
        return data

    for unit in BASELINE_UNITS:
        data = do(unit)
        if (
            data["status"] == "ok"
            and data["max_depth"] is not None
            and data["max_depth"] >= CAP
        ):
            raise BaselineAtCap(
                f"the delta = 0 cell {key_of(unit)} reached the depth cap "
                f"(max_depth {data['max_depth']} >= {CAP}): prompt section 5 stop, the "
                f"baseline is not clean"
            )

    classified = do(CLASSIFY_UNIT)
    if classified["status"] != "ok":
        raise RuntimeError(
            f"the arm-B classification cell raised: {classified['error']}"
        )
    arm_b_labels = labels_of(classified)
    log(f"arm B labels: {arm_b_labels}", flush=True)

    units = noised_units(arm_b_labels)
    total_units = len(BASELINE_UNITS) + 1 + len(units)
    heartbeat(stage="noised cells", units_total=total_units)
    for unit in units:
        do(unit)
    log(
        f"sweep complete: {total_units} units in {time.perf_counter() - started:.0f} s",
        flush=True,
    )


PROVENANCE = None


# ================================================================================================
# --probe: the prompt's steps 2 and 3, as a registered run of its own
#
# A fully noised cell can take tens of minutes or more (a scratch timing of exact/together/980 at
# delta = 1e-5 had not finished after 540 s), so the probe is itself a long job: it is registered,
# launched detached, and checkpoints each cell, exactly as the sweep does (CLAUDE.md, "Long-running
# jobs"). `--probe-report` writes its tables and the projection from that ledger alone.

PROBE_TIMING = (("exact", "together", 980.0), ARM_B_CASE)
PROBE_CHANGES = ("exact", "together", 980.0)


def probe_units() -> list:
    """Step 2's bit-identity pairs on all eight cases first (cheap, and they carry the section 5
    stop on a delta = 0 cell at the cap), then step 3's two noised timing cells, then step 2's
    delta = 1e-4 cell."""
    units = []
    for f, s, x in CASES:
        for tag in ("unwrapped", "wrapped"):
            units.append(dict(tag=tag, flavour=f, shape=s, x_resp=x, delta=0.0))
    for f, s, x in PROBE_TIMING:
        units.append(dict(tag="timing", flavour=f, shape=s, x_resp=x, delta=1e-5))
    f, s, x = PROBE_CHANGES
    units.append(dict(tag="changes", flavour=f, shape=s, x_resp=x, delta=1e-4))
    return units


def probe_key(unit: dict) -> str:
    return (
        f"probe|{unit['tag']}|{unit['flavour']}|{unit['shape']}|{unit['x_resp']:g}"
        f"|{unit['delta']:g}"
    )


def probe_sweep(record, known: dict, heartbeat, log=print) -> None:
    started = time.perf_counter()
    for unit in probe_units():
        key = probe_key(unit)
        if key in known:
            data = known[key]["data"]
        else:
            heartbeat(stage=f"running {key}")
            try:
                data = run_cell(
                    unit["flavour"],
                    unit["shape"],
                    unit["x_resp"],
                    unit["delta"],
                    "white",
                    instrument=(unit["tag"] != "unwrapped"),
                )
            except Exception as exc:  # noqa: BLE001
                assert_restored()
                data = {
                    "flavour": unit["flavour"],
                    "shape": unit["shape"],
                    "x_resp": unit["x_resp"],
                    "delta": unit["delta"],
                    "noise": "white",
                    "status": "raised",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            data["tag"] = unit["tag"]
            data["provenance"] = PROVENANCE
            record(key, data)
            note = (
                f"max_depth {data['max_depth']} regions {data['num_regions']} "
                f"wall {data['wall_time']:.1f} s cpu {data['cpu_time']:.1f} s "
                f"load {data['loadavg_after'][0]:.1f}"
                if data["status"] == "ok"
                else f"RAISED {data['error']}"
            )
            log(f"[{time.perf_counter() - started:8.0f} s] {key}: {note}", flush=True)
        if unit["tag"] == "wrapped" and data["status"] == "ok" and at_cap(data):
            raise BaselineAtCap(
                f"the delta = 0 cell {key} reached the depth cap (max_depth "
                f"{data['max_depth']} >= {CAP}): prompt section 5 stop, the baseline is not "
                f"clean"
            )


def probe_report(path: str) -> None:
    ledger = load_ledger(path)
    cells = {key: entry["data"] for key, entry in ledger.items()}

    def get(tag, f, s, x, d):
        rec = cells.get(probe_key(dict(tag=tag, flavour=f, shape=s, x_resp=x, delta=d)))
        return rec if rec is not None and rec.get("status") == "ok" else None

    print(
        "<!-- generated by docs/levin-phase-floor/noise_injection.py --probe-report -->"
    )
    print()
    print("# levin-phase-floor 02: wrapper checks and timing probe")
    print()
    print(
        f"Checkpoint: `{os.path.relpath(path, REPO_ROOT)}`, sha256 {sha256_of(path)}."
    )
    print()
    for prov in sorted(
        {
            json.dumps(rec["provenance"], sort_keys=True)
            for rec in cells.values()
            if "provenance" in rec
        }
    ):
        for line in provenance_lines(json.loads(prov)):
            print(f"- {line}")
    print()
    expected = [probe_key(u) for u in probe_units()]
    missing = [key for key in expected if key not in cells]
    raised = {k: r["error"] for k, r in cells.items() if r.get("status") != "ok"}
    print(
        f"**Cells:** {len(expected)} expected, {len(cells) - len(raised)} ok, "
        f"{len(raised)} raised, {len(missing)} missing."
    )
    for key, error in raised.items():
        print(f"- raised: `{key}`: {error}")
    for key in missing:
        print(f"- missing: `{key}`")
    print()

    print("## Step 2: the wrapper at delta = 0 is bit-identical to no wrapper")
    print()
    print(
        "Compared: every field of the record but the timing and instrumentation ones, floats "
        "by `repr`, including every group record of every Levin sub-interval."
    )
    print()
    print(
        "| flavour | shape | x_resp | total, unwrapped | total, wrapped | identical | first "
        "difference | max_depth | regions | wall, unwrapped | wall, wrapped | load after |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    verdicts, at_cap_cases = [], []
    for f, s, x in CASES:
        plain = get("unwrapped", f, s, x, 0.0)
        wrapped = get("wrapped", f, s, x, 0.0)
        if plain is None or wrapped is None:
            verdicts.append(None)
            print(f"| {f} | {s} | {x:g} | " + " | ".join(["--"] * 9) + " |")
            continue
        same = deterministic_view(plain) == deterministic_view(wrapped)
        verdicts.append(same)
        if at_cap(wrapped):
            at_cap_cases.append(f"{f}/{s}/{x:g}")
        print(
            f"| {f} | {s} | {x:g} | {plain['total']!r} | {wrapped['total']!r} "
            f"| {'yes' if same else '**NO**'} | {first_difference(plain, wrapped)} "
            f"| {wrapped['max_depth']} | {wrapped['num_regions']} "
            f"| {plain['wall_time']:.2f} s | {wrapped['wall_time']:.2f} s "
            f"| {wrapped['loadavg_after'][0]:.1f} |"
        )
    print()
    identical = "incomplete" if None in verdicts else ("yes" if all(verdicts) else "NO")
    print(
        f"Bit-identical in all eight cases: **{identical}**. delta = 0 cells at the cap: "
        f"**{', '.join(at_cap_cases) if at_cap_cases else 'none'}**."
    )
    print()

    print("## Step 2: an injected delta = 1e-4 changes `total`")
    print()
    zero = get("wrapped", *PROBE_CHANGES, 0.0)
    big = get("changes", *PROBE_CHANGES, 1e-4)
    if zero is None or big is None:
        print("Not available: the cell is missing or raised.")
    else:
        print(
            f"{'/'.join(str(v) for v in PROBE_CHANGES)}: total(0) = {zero['total']!r}, "
            f"total(1e-4) = {big['total']!r}, relative change "
            f"{abs(big['total'] - zero['total']) / abs(zero['total']):.3e}; max_depth "
            f"{zero['max_depth']} -> {big['max_depth']}, regions {zero['num_regions']} -> "
            f"{big['num_regions']}, wall {big['wall_time']:.1f} s. Changed: "
            f"**{'yes' if big['total'] != zero['total'] else 'NO'}**."
        )
    print()

    print("## Step 3: the timing probe")
    print()
    print(
        "| flavour | shape | x_resp | delta | max_depth | num_regions | wall | cpu | load before "
        "| load after |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    timing = {}
    for f, s, x in PROBE_TIMING:
        for tag, d in (("wrapped", 0.0), ("timing", 1e-5)):
            rec = get(tag, f, s, x, d)
            timing[(f, s, x, d)] = rec
            if rec is None:
                print(f"| {f} | {s} | {x:g} | {d:g} | " + " | ".join(["--"] * 6) + " |")
                continue
            print(
                f"| {f} | {s} | {x:g} | {d:g} | {rec['max_depth']} | {rec['num_regions']} "
                f"| {rec['wall_time']:.1f} s | {rec['cpu_time']:.1f} s "
                f"| {rec['loadavg_before'][0]:.1f} | {rec['loadavg_after'][0]:.1f} |"
            )
    print()
    baselines = {(f, s, x): get("wrapped", f, s, x, 0.0) for f, s, x in CASES}
    if any(rec is None for rec in timing.values()) or any(
        rec is None for rec in baselines.values()
    ):
        print("No projection: a timing or baseline cell is missing.")
        return
    arm_b_labels = labels_of(baselines[ARM_B_CASE])
    (ef, es, ex), (rf, rs, rx) = PROBE_TIMING
    factor = {
        "exact": timing[(ef, es, ex, 1e-5)]["wall_time"]
        / timing[(ef, es, ex, 0.0)]["wall_time"],
        "realistic": timing[(rf, rs, rx, 1e-5)]["wall_time"]
        / timing[(rf, rs, rx, 0.0)]["wall_time"],
    }
    slowest = {
        "exact": timing[(ef, es, ex, 1e-5)]["wall_time"],
        "realistic": timing[(rf, rs, rx, 1e-5)]["wall_time"],
    }
    worst = max(slowest.values())
    units = noised_units(arm_b_labels)
    fixed = sum(rec["wall_time"] + rec["setup_time"] for rec in baselines.values())
    fixed += baselines[ARM_B_CASE]["wall_time"] + baselines[ARM_B_CASE]["setup_time"]
    central, flavour_floor, flat = fixed, fixed, fixed
    by_group = {}
    for unit in units:
        rec = baselines[(unit["flavour"], unit["shape"], unit["x_resp"])]
        estimate = rec["wall_time"] * factor[unit["flavour"]]
        central += estimate + rec["setup_time"]
        flavour_floor += max(estimate, slowest[unit["flavour"]]) + rec["setup_time"]
        flat += max(estimate, worst) + rec["setup_time"]
        group = (unit["arm"], unit["flavour"])
        by_group[group] = by_group.get(group, 0) + 1
    n_units = len(BASELINE_UNITS) + 1 + len(units)
    print(
        f"Arm B labels in {'/'.join(str(v) for v in ARM_B_CASE)} (from its delta = 0 cell): "
        f"{arm_b_labels}. Units: {len(BASELINE_UNITS)} delta = 0 + 1 classification + "
        f"{len(units)} noised = {n_units}. Noised units by (arm, flavour): "
        + ", ".join(f"{k[0]}/{k[1]} {v}" for k, v in sorted(by_group.items()))
        + "."
    )
    print()
    print(
        f"Cost factor of white noise at delta = 1e-5 over delta = 0: exact "
        f"{factor['exact']:.1f}x ({ef}/{es}/{ex:g}), realistic {factor['realistic']:.1f}x "
        f"({rf}/{rs}/{rx:g})."
    )
    print()
    print(
        f"- **Central projection** (each noised cell costs its own case's delta = 0 wall time "
        f"times its flavour's factor, plus set-up): **{central / 3600.0:.2f} h**."
    )
    print(
        f"- **Flavour-floor projection** (as central, but no noised cell cheaper than its "
        f"flavour's probed noised cell): **{flavour_floor / 3600.0:.2f} h**."
    )
    print(
        f"- **Flat projection** (no noised cell cheaper than the slowest probed noised cell, "
        f"{worst:.0f} s): **{flat / 3600.0:.2f} h**."
    )
    print(
        f"- Over 24 h: central **{'yes' if central > 86400 else 'no'}**, flavour-floor "
        f"**{'yes' if flavour_floor > 86400 else 'no'}**, flat "
        f"**{'yes' if flat > 86400 else 'no'}**."
    )


# ================================================================================================
# --run / --probe: register and launch detached; --worker / --probe-worker: the detached job

MODES = {
    "sweep": {
        "slug": "noise",
        "worker": "--worker",
        "purpose": (
            "levin-phase-floor prompt 02, the sweep: deterministic phase noise of amplitude "
            "delta added to the composed theta_mod_2pi of QuadSourceIntegral's phase groups on "
            "A2's fixtures (b = 0, seam closed), at atol 1e-45 and rtol DEFAULT_QUADRATURE_RTOL; "
            "depth and regions against delta, scored against the prompt's N1-N4. No Ray, no "
            "store."
        ),
    },
    "probe": {
        "slug": "noise-probe",
        "worker": "--probe-worker",
        "purpose": (
            "levin-phase-floor prompt 02, steps 2-3: the wrapper's bit-identity at delta = 0 on "
            "all eight cases and its effect at delta = 1e-4, and the timing probe (exact/"
            "together/980 and realistic/q-smooth/1e4 at delta = 0 and 1e-5) the sweep's "
            "projection is taken from. No Ray, no store."
        ),
    },
}


def launch(mode: str, checkpoint=None) -> None:
    import runregistrykit

    if mode == "sweep":
        n_noised = 8 * (len(LADDER) - 1) + 8 * len(ARM_C_DELTAS)
        expected = len(BASELINE_UNITS) + 1 + n_noised
        scope = (
            f"arm A: white noise on every group, 8 cases x {len(LADDER)} rungs "
            f"({LADDER[0]:g} to {LADDER[-1]:g}); arm B: one label at a time in "
            f"{'/'.join(str(v) for v in ARM_B_CASE)}, delta {ARM_B_DELTAS}, white and "
            f"staircase; arm C: staircase on every group, 8 cases x {ARM_C_DELTAS}; plus one "
            f"classification cell. Arm B's size is known once its labels are read."
        )
    else:
        expected = len(probe_units())
        scope = (
            f"{len(probe_units())} cells: 8 cases x (unwrapped, wrapped) at delta = 0; "
            f"{len(PROBE_TIMING)} timing cells at delta = 1e-5; 1 cell at delta = 1e-4"
        )
    run = runregistrykit.begin(
        campaign="levin-phase-floor",
        prompt="02",
        slug=MODES[mode]["slug"],
        purpose=MODES[mode]["purpose"],
        script=os.path.abspath(__file__),
        unit="cell",
        expected_units=expected,
        checkpoint=checkpoint if checkpoint else True,
        scope=scope,
        heartbeat_means=(
            "Refreshed when a cell starts (the stage names it) and when it is recorded. One "
            "noised cell at the depth cap can take a long time, so a stale heartbeat means look "
            "at stdout.txt in this directory, not that the worker is dead."
        ),
    )
    run_dir = os.path.dirname(run.manifest_path)
    with open(run.stdout_path, "ab") as out, open(run.stderr_path, "ab") as err:
        child = subprocess.Popen(
            [
                "nohup",
                sys.executable,
                "-u",
                os.path.abspath(__file__),
                MODES[mode]["worker"],
                run_dir,
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONPATH": REPO_ROOT},
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
    run.heartbeat(pid=child.pid, stage="worker launched")
    print(f"run id     {run.id}")
    print(f"run dir    {run_dir}")
    print(f"worker pid {child.pid}")
    print(f"launcher   {os.getpid()}")
    print(f"stdout     {run.stdout_path}")
    print(f"ledger     {run.checkpoint_path}")


def worker(run_dir: str, mode: str) -> None:
    global PROVENANCE
    import runregistrykit

    warnings.simplefilter("ignore")
    run = runregistrykit.Run(run_dir)
    manifest_sha = run.manifest.get("script_sha256")
    if manifest_sha != SCRIPT_SHA256:
        run.finish("failed", exit_code=2)
        raise SystemExit(
            f"noise_injection: this script is {SCRIPT_SHA256[:12]} but the run's manifest "
            f"records {str(manifest_sha)[:12]}: the script changed after launch"
        )
    PROVENANCE = provenance()

    def terminal(signum, _frame):
        run.finish("killed", exit_code=128 + signum)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, terminal)

    print(
        f"** noise_injection {mode} worker: run {run.id}, pid {os.getpid()}", flush=True
    )
    for line in provenance_lines(PROVENANCE, "** "):
        print(line, flush=True)
    up = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    print(f"** uptime at start: {up}", flush=True)
    known = run.known()
    print(f"** {len(known)} unit(s) already in the ledger", flush=True)
    try:
        if mode == "sweep":
            sweep(run.record, known, run.heartbeat)
        else:
            probe_sweep(run.record, known, run.heartbeat)
    except BaselineAtCap as exc:
        print(f"** STOP: {exc}", flush=True)
        run.finish("failed", exit_code=3)
        raise SystemExit(3)
    except BaseException:
        run.finish("failed", exit_code=1)
        raise
    up = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    print(f"** uptime at end: {up}", flush=True)
    run.finish("done", exit_code=0)


# ================================================================================================
# --report: every table and the scoring, from the checkpoint alone


def load_ledger(path: str) -> dict:
    """Every unit in the ledger whose script hash is this script's; others are discarded with a
    notice and never re-stamped. A torn trailing line is ignored."""
    units, stale = {}, {}
    with open(path) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    for number, line in enumerate(lines, start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                continue
            raise
        if entry.get("script_sha256") != SCRIPT_SHA256:
            stale[entry.get("unit")] = entry.get("script_sha256")
            units.pop(entry.get("unit"), None)
            continue
        units[entry["unit"]] = entry
    if stale:
        print(
            f"<!-- discarded {len(stale)} record(s) written by another script version: "
            f"{sorted({str(h)[:12] for h in stale.values()})} -->"
        )
    return units


def fmt(value, spec):
    return "--" if value is None else format(value, spec)


def depth_by_label(rec: dict) -> dict:
    """The deepest bisection reached by each group label over all its Levin calls."""
    out = {}
    for call in rec.get("levin_calls") or []:
        if call["kind"] != "group":
            continue
        out[call["label"]] = max(out.get(call["label"], 0), call["max_depth"])
    return out


def regions_by_label(rec: dict) -> dict:
    out = {}
    for item in rec["subintervals"]:
        for group in item["groups"]:
            out[group["label"]] = out.get(group["label"], 0) + group["regions"]
    return out


def classify_labels(rec: dict) -> dict:
    """
    Per label: the ratio |Psi'| / (fastest |Psi'| among the groups of the same sub-interval) at
    each classification abscissa, and the verdict slow (every ratio < SLOW_FRACTION), fast (every
    ratio >= SLOW_FRACTION) or mixed.
    """
    by_span = {}
    for call in rec["levin_calls"]:
        if call["kind"] != "group":
            continue
        by_span.setdefault(tuple(call["x_span"]), []).append(call)
    ratios, magnitudes = {}, {}
    for calls in by_span.values():
        for i in range(len(CLASSIFY_FRACTIONS)):
            fastest = max(abs(c["theta_deriv"][i][1]) for c in calls)
            for c in calls:
                value = abs(c["theta_deriv"][i][1])
                ratios.setdefault(c["label"], []).append(
                    value / fastest if fastest > 0.0 else float("nan")
                )
                magnitudes.setdefault(c["label"], []).append(value)
    out = {}
    for label, values in ratios.items():
        if all(v < SLOW_FRACTION for v in values):
            verdict = "slow"
        elif all(v >= SLOW_FRACTION for v in values):
            verdict = "fast"
        else:
            verdict = "mixed"
        out[label] = {
            "min_ratio": min(values),
            "max_ratio": max(values),
            "min_abs": min(magnitudes[label]),
            "max_abs": max(magnitudes[label]),
            "subintervals": sum(
                1
                for calls in by_span.values()
                if any(c["label"] == label for c in calls)
            ),
            "verdict": verdict,
        }
    return out


def at_cap(rec) -> bool:
    return rec is not None and rec["max_depth"] is not None and rec["max_depth"] >= CAP


def verdict_word(value) -> str:
    return {True: "true", False: "**false**", None: "n/a"}[value]


def report(path: str) -> None:
    ledger = load_ledger(path)
    cells = {key: entry["data"] for key, entry in ledger.items()}
    ok = {key: rec for key, rec in cells.items() if rec.get("status") == "ok"}

    def get(arm, noise, labels, f, s, x, d):
        return ok.get(unit_key(arm, noise, labels, f, s, x, d))

    print("<!-- generated by docs/levin-phase-floor/noise_injection.py --report -->")
    print()
    print("# levin-phase-floor 02: phase-noise injection, the report")
    print()
    print(
        f"Checkpoint: `{os.path.relpath(path, REPO_ROOT)}`, sha256 {sha256_of(path)}."
    )
    print()
    print(
        "**Provenance of the records** (every record carries its own; distinct values shown):"
    )
    print()
    heads = sorted(
        {(e["git_head"], e["git_dirty"], e["script_sha256"]) for e in ledger.values()}
    )
    for head, dirty, sha in heads:
        print(f"- script sha256 `{sha}`, git HEAD `{head}`, dirty {dirty}")
    provs = sorted(
        {
            json.dumps(rec["provenance"], sort_keys=True)
            for rec in cells.values()
            if "provenance" in rec
        }
    )
    for prov in provs:
        for line in provenance_lines(json.loads(prov)):
            print(f"- {line}")
    loads = [rec["loadavg_after"][0] for rec in ok.values()]
    if loads:
        print(
            f"- one-minute load average after each cell: min {min(loads):.1f}, median "
            f"{sorted(loads)[len(loads) // 2]:.1f}, max {max(loads):.1f} (10 cores)"
        )
    print()

    # --- what is missing ------------------------------------------------------------------------
    classified = ok.get(key_of(CLASSIFY_UNIT))
    arm_b_labels = labels_of(classified) if classified else []
    expected = (
        [key_of(u) for u in BASELINE_UNITS]
        + [key_of(CLASSIFY_UNIT)]
        + [key_of(u) for u in noised_units(arm_b_labels)]
    )
    missing = [key for key in expected if key not in cells]
    raised = {key: rec["error"] for key, rec in cells.items() if rec["status"] != "ok"}
    print(
        f"**Cells:** {len(expected)} expected, {len(ok)} ok, {len(raised)} raised, "
        f"{len(missing)} missing."
        + (
            ""
            if classified
            else " Arm B's labels are unknown: its classification cell is missing."
        )
    )
    for key, error in raised.items():
        print(f"- raised: `{key}`: {error}")
    for key in missing:
        print(f"- missing: `{key}`")
    print()

    # --- arm A -----------------------------------------------------------------------------------
    print(
        "## Table A: white noise on every group, by case and delta\n\n"
        "`dTotal/abserr` is |total(delta) - total(0)| / total_abserr(delta): N4 holds in a cell "
        "where it is at most 1. `rel. change` is |total(delta) - total(0)| / |total(0)|."
    )
    print()
    print(
        "| flavour | shape | x_resp | delta | max_depth | num_regions | total | total_abserr "
        "| abserr/abs(total) | rel. change | dTotal/abserr | converged | phase_limited | wall "
        "| cpu |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f, s, x in CASES:
        zero = get("A", "white", ALL, f, s, x, 0.0)
        for d in LADDER:
            rec = get("A", "white", ALL, f, s, x, d)
            if rec is None:
                print(
                    f"| {f} | {s} | {x:g} | {d:g} | " + " | ".join(["--"] * 11) + " |"
                )
                continue
            change = None if zero is None else abs(rec["total"] - zero["total"])
            print(
                f"| {f} | {s} | {x:g} | {d:g} | {rec['max_depth']} | {rec['num_regions']} "
                f"| {rec['total']:+.10e} | {rec['total_abserr']:.2e} "
                f"| {rec['total_abserr'] / abs(rec['total']):.2e} "
                f"| {fmt(None if change is None else change / abs(zero['total']), '.2e')} "
                f"| {fmt(None if change is None or d == 0.0 else change / rec['total_abserr'], '.2f')} "
                f"| {'yes' if rec['total_converged'] else 'no'} "
                f"| {'yes' if rec['total_phase_limited'] else 'no'} "
                f"| {rec['wall_time']:.1f} s | {rec['cpu_time']:.1f} s |"
            )
    print()

    # --- N1 ----------------------------------------------------------------------------------------
    print("## N1: noise alone reaches the cap near rtol")
    print()
    print(
        "Conditions, per case, in arm A: (i) the delta = 0 cell is below the cap; (ii) "
        "delta* (the smallest rung whose max_depth reaches the cap) exists and is at most "
        "1e-6; (iii) from delta* up, num_regions is non-decreasing in delta within 10 %, read "
        "as: at every pair of consecutive rungs at or above delta*, "
        "num_regions(upper) >= 0.9 num_regions(lower)."
    )
    print()
    print(
        "| case | (i) delta = 0 below cap | delta* | (ii) | num_regions from delta* up | (iii) |"
    )
    print("|---|---|---|---|---|---|")
    n1 = {}
    for f, s, x in CASES:
        recs = [get("A", "white", ALL, f, s, x, d) for d in LADDER]
        if any(rec is None for rec in recs):
            n1[(f, s, x)] = None
            print(f"| {f}/{s}/{x:g} | incomplete | -- | -- | -- | -- |")
            continue
        c1 = not at_cap(recs[0])
        star = next((d for d, rec in zip(LADDER, recs) if at_cap(rec)), None)
        c2 = star is not None and star <= 1e-6
        if star is None:
            c3, regions, breaks = None, [], []
        else:
            tail = [
                (d, rec["num_regions"]) for d, rec in zip(LADDER, recs) if d >= star
            ]
            regions = [n for _, n in tail]
            breaks = [
                f"{tail[i][0]:g}->{tail[i + 1][0]:g}"
                for i in range(len(tail) - 1)
                if tail[i + 1][1] < 0.9 * tail[i][1]
            ]
            c3 = not breaks
        n1[(f, s, x)] = (c1, c2, c3)
        print(
            f"| {f}/{s}/{x:g} | {verdict_word(c1)} | {fmt(star, 'g')} | {verdict_word(c2)} "
            f"| {regions if regions else '--'} "
            f"| {verdict_word(c3)}{' (falls at ' + ', '.join(breaks) + ')' if breaks else ''} |"
        )
    print()
    if any(v is None for v in n1.values()):
        n1_verdict = "undecided (arm A incomplete)"
    else:
        all_true = all(all(c is True for c in v) for v in n1.values())
        realistic_false = [
            (case, i + 1)
            for case, v in n1.items()
            if case[0] == "realistic"
            for i, c in enumerate(v)
            if c is not True
        ]
        exact_false = [
            (case, i + 1)
            for case, v in n1.items()
            if case[0] == "exact"
            for i, c in enumerate(v)
            if c is not True
        ]
        if all_true:
            n1_verdict = "holds"
        elif realistic_false:
            n1_verdict = "fails"
        else:
            n1_verdict = "undecided"
        print(
            f"False in a realistic case: {realistic_false or 'none'}. False in an exact case: "
            f"{exact_false or 'none'} (as (case, condition))."
        )
        print()
    print(f"**N1: {n1_verdict}.**")
    print()

    # --- N2 ----------------------------------------------------------------------------------------
    print("## N2: the exact flavour's flatness is a property of its phases")
    print()
    print(
        "Ratio = exact num_regions at x_resp = 1e4 over x_resp = 980, arm A. Part 1: at "
        "delta = 0 within a factor of 2 (0.5 <= ratio <= 2) for each shape. Part 2: at "
        "delta = 1e-6 at least 3 for each shape."
    )
    print()
    print("| shape | delta | regions at 980 | regions at 1e4 | ratio | condition |")
    print("|---|---|---|---|---|---|")
    parts = {}
    for part, d, test, words in (
        (1, 0.0, lambda r: 0.5 <= r <= 2.0, "within 2x"),
        (2, 1e-6, lambda r: r >= 3.0, ">= 3"),
    ):
        results = []
        for s in SHAPE_NAMES:
            lo = get("A", "white", ALL, "exact", s, 980.0, d)
            hi = get("A", "white", ALL, "exact", s, 1e4, d)
            if lo is None or hi is None:
                results.append(None)
                print(f"| {s} | {d:g} | -- | -- | -- | incomplete |")
                continue
            ratio = hi["num_regions"] / lo["num_regions"]
            results.append(test(ratio))
            print(
                f"| {s} | {d:g} | {lo['num_regions']} | {hi['num_regions']} | {ratio:.2f} "
                f"| {words}: {verdict_word(test(ratio))} |"
            )
        parts[part] = (
            "undecided (incomplete)"
            if None in results
            else ("holds" if all(results) else "fails")
        )
    print()
    print(
        f"**N2 part 1 (flat at delta = 0): {parts[1]}. N2 part 2 (growth at 1e-6): {parts[2]}.**"
    )
    print()

    # --- arm B and N3 ------------------------------------------------------------------------------
    print("## Arm B: one label at a time in " + "/".join(str(v) for v in ARM_B_CASE))
    print()
    if classified is None:
        print("The classification cell is missing; arm B cannot be read.")
        print()
        n3_verdict = "undecided (no classification)"
        h_anchor = "undecided"
    else:
        classes = classify_labels(classified)
        print(
            f"Slow or fast, from `theta_deriv` at fractions {CLASSIFY_FRACTIONS} of each "
            f"sub-interval: ratio = abs(Psi') / (the fastest group's abs(Psi') in the same "
            f"sub-interval, which is the sum of the constituents' abs(theta')); slow if every "
            f"ratio < {SLOW_FRACTION}."
        )
        print()
        print(
            "| label | sub-intervals | ratio min | ratio max | abs(Psi') min | abs(Psi') max | class |"
        )
        print("|---|---|---|---|---|---|---|")
        for label in arm_b_labels:
            c = classes[label]
            print(
                f"| `{label}` | {c['subintervals']} | {c['min_ratio']:.3f} | {c['max_ratio']:.3f} "
                f"| {c['min_abs']:.4e} | {c['max_abs']:.4e} | {c['verdict']} |"
            )
        print()
        zero = get("A", "white", ALL, *ARM_B_CASE, 0.0)
        zero_depths = depth_by_label(zero) if zero else {}
        print(
            "`depth (label)` is the deepest bisection the noised label's own Levin calls reached "
            "(the pass-through observer); `max_depth (cell)` is the stored maximum over every "
            "group. At delta = 0 the labels' depths are "
            + ", ".join(f"`{k}` {v}" for k, v in zero_depths.items())
            + "."
        )
        print()
        print(
            "| label | class | noise | delta | depth (label) | max_depth (cell) | regions (label) "
            "| num_regions (cell) | rel. change | dTotal/abserr | wall |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        n3_rows = []
        for label in arm_b_labels:
            for noise in NOISES:
                for d in ARM_B_DELTAS:
                    rec = get("B", noise, [label], *ARM_B_CASE, d)
                    if rec is None:
                        print(
                            f"| `{label}` | {classes[label]['verdict']} | {noise} | {d:g} | "
                            + " | ".join(["--"] * 7)
                            + " |"
                        )
                        if noise == "white" and d == 1e-5:
                            n3_rows.append((label, None))
                        continue
                    depth = depth_by_label(rec).get(label)
                    change = None if zero is None else abs(rec["total"] - zero["total"])
                    print(
                        f"| `{label}` | {classes[label]['verdict']} | {noise} | {d:g} | {depth} "
                        f"| {rec['max_depth']} | {regions_by_label(rec).get(label)} "
                        f"| {rec['num_regions']} "
                        f"| {fmt(None if change is None else change / abs(zero['total']), '.2e')} "
                        f"| {fmt(None if change is None else change / rec['total_abserr'], '.2f')} "
                        f"| {rec['wall_time']:.1f} s |"
                    )
                    if noise == "white" and d == 1e-5:
                        n3_rows.append((label, depth))
        print()
        if any(depth is None for _, depth in n3_rows) or not n3_rows:
            n3_verdict = "undecided (arm B incomplete)"
            h_anchor = "undecided"
        else:
            below = [(label, depth) for label, depth in n3_rows if depth < CAP]
            n3_verdict = "holds" if not below else "fails"
            fast_below = [
                (label, depth)
                for label, depth in below
                if classes[label]["verdict"] != "slow"
            ]
            h_anchor = (
                f"fails H-anchor: fast or mixed label(s) below the cap at 1e-5: {fast_below}"
                if fast_below
                else "does not fail H-anchor (every fast group reaches the cap)"
            )
            print(f"Labels below the cap at white 1e-5: {below or 'none'}.")
            print()
    print(f"**N3: {n3_verdict}.** Consequence for H-anchor: {h_anchor}.")
    print()

    # --- N4 ----------------------------------------------------------------------------------------
    print("## N4: the declared error stays honest under noise")
    print()
    print(
        "Every cell of arms A and C with delta > 0: |total(delta) - total(0)| <= "
        "total_abserr(delta)."
    )
    print()
    n4_cells, n4_fail, n4_missing = 0, [], 0
    for f, s, x in CASES:
        zero = get("A", "white", ALL, f, s, x, 0.0)
        for arm, noise, deltas in (
            ("A", "white", LADDER[1:]),
            ("C", "staircase", ARM_C_DELTAS),
        ):
            for d in deltas:
                rec = get(arm, noise, ALL, f, s, x, d)
                if rec is None or zero is None:
                    n4_missing += 1
                    continue
                n4_cells += 1
                change = abs(rec["total"] - zero["total"])
                if not change <= rec["total_abserr"]:
                    n4_fail.append(
                        (
                            arm,
                            f,
                            s,
                            x,
                            d,
                            change / rec["total_abserr"],
                            change / abs(zero["total"]),
                        )
                    )
    if n4_fail:
        print("| arm | case | delta | dTotal/abserr | rel. change |")
        print("|---|---|---|---|---|")
        for arm, f, s, x, d, ratio, rel in n4_fail:
            print(f"| {arm} | {f}/{s}/{x:g} | {d:g} | {ratio:.2f} | {rel:.2e} |")
        print()
    n4_verdict = (
        f"undecided ({n4_missing} cell(s) missing)"
        if n4_missing and not n4_fail
        else ("holds" if not n4_fail else "fails")
    )
    print(
        f"{n4_cells} cells scored, {len(n4_fail)} where the declared error understates the "
        f"change, {n4_missing} missing. **N4: {n4_verdict}.**"
    )
    print()

    # --- descriptive: white against staircase ---------------------------------------------------
    print("## Descriptive: white against staircase (arms A and C)")
    print()
    print(
        "| case | delta | depth white | depth staircase | regions white | regions staircase "
        "| rel. change white | rel. change staircase | abserr/abs(total) white "
        "| abserr/abs(total) staircase |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for f, s, x in CASES:
        zero = get("A", "white", ALL, f, s, x, 0.0)
        for d in ARM_C_DELTAS:
            a = get("A", "white", ALL, f, s, x, d)
            c = get("C", "staircase", ALL, f, s, x, d)
            if a is None or c is None or zero is None:
                print(f"| {f}/{s}/{x:g} | {d:g} | " + " | ".join(["--"] * 8) + " |")
                continue
            print(
                f"| {f}/{s}/{x:g} | {d:g} | {a['max_depth']} | {c['max_depth']} "
                f"| {a['num_regions']} | {c['num_regions']} "
                f"| {abs(a['total'] - zero['total']) / abs(zero['total']):.2e} "
                f"| {abs(c['total'] - zero['total']) / abs(zero['total']):.2e} "
                f"| {a['total_abserr'] / abs(a['total']):.2e} "
                f"| {c['total_abserr'] / abs(c['total']):.2e} |"
            )
    print()

    # --- descriptive: elapsed by group --------------------------------------------------------
    print(
        "## Descriptive: share of each cell's Levin `elapsed` by group (arms A and C)"
    )
    print()
    print("| arm | case | delta | Levin elapsed | shares | deepest label(s) |")
    print("|---|---|---|---|---|---|")
    for f, s, x in CASES:
        for arm, noise, deltas in (
            ("A", "white", LADDER),
            ("C", "staircase", ARM_C_DELTAS),
        ):
            for d in deltas:
                rec = get(arm, noise, ALL, f, s, x, d)
                if rec is None:
                    continue
                total_elapsed = sum(rec["group_elapsed"].values())
                shares = (
                    ", ".join(
                        f"`{k}` {100.0 * v / total_elapsed:.0f} %"
                        for k, v in sorted(
                            rec["group_elapsed"].items(), key=lambda kv: -kv[1]
                        )
                    )
                    if total_elapsed > 0
                    else "--"
                )
                depths = depth_by_label(rec)
                top = max(depths.values()) if depths else None
                deepest = ", ".join(f"`{k}`" for k, v in depths.items() if v == top)
                print(
                    f"| {arm} | {f}/{s}/{x:g} | {d:g} | {total_elapsed:.1f} s | {shares} "
                    f"| {deepest} ({top}) |"
                )
    print()

    # --- the verdicts ------------------------------------------------------------------------------
    print("## Verdicts")
    print()
    print(f"- **N1:** {n1_verdict}")
    print(f"- **N2:** part 1 {parts[1]}; part 2 {parts[2]}")
    print(f"- **N3:** {n3_verdict}; {h_anchor}")
    print(f"- **N4:** {n4_verdict}")


# ================================================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--probe-report", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--worker", metavar="RUN_DIR")
    mode.add_argument("--probe-worker", metavar="RUN_DIR")
    parser.add_argument(
        "--checkpoint",
        help="--report / --probe-report: the ledger to read. --run / --probe: a ledger to resume "
        "from (default: a new one in the run directory)",
    )
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    if args.probe:
        launch("probe", args.checkpoint)
    elif args.run:
        launch("sweep", args.checkpoint)
    elif args.report or args.probe_report:
        if not args.checkpoint:
            parser.error("--report and --probe-report need --checkpoint")
        (report if args.report else probe_report)(os.path.abspath(args.checkpoint))
    elif args.worker:
        worker(args.worker, "sweep")
    else:
        worker(args.probe_worker, "probe")


if __name__ == "__main__":
    main()
