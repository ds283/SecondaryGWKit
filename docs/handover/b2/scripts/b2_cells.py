"""
handover prompt 13 (B2), Part B: one cell of the residue attribution -- A2's ladder at `HEAD`, in
thirteen objects that differ in which accessors are realistic.

The successor at `HEAD` of `docs/handover/realistic_large_x.py` (A2's harness), not an edit of
it. It **imports** from that module what still holds (`RUNS`, `scaled_shape`, `fixture_x_max`,
`seed_fixtures`, `reference_of`, `REF_ATOL`, `REF_RTOL`) and builds every case through
`ComputeTargets.tests.test_quadsource_integral.Case`, which since Part A (`28ade78`) is
production's construction in its realistic flavour. It reads `partition["handover"]` (`z_init`,
`z_switch`, `handover_continuity` per factor) where A2 read the retired clamp records
(`[12-realistic-large-x-reads-the-retired-clamp-records]`). Its cell/driver/launcher/tables shape
is A4's (`docs/handover/a4/scripts/`), whose scripts are records of their own SHA and cannot be
imported at `HEAD` (they import `OffsetBesselPhaseGk`, which Part A removed).

**The objects** (prompt 13 section 2.2). Every cell at a rung shares one shape, one lambda
scaling, one set of fixtures (`seed_fixtures`), one 50-digit reference (`reference_of`) and the
tolerances `REF_ATOL = 1e-45`, `REF_RTOL = 1e-12`.

    E        Case(exact=True): A2's exact flavour, its own geometry (T break at z_init). Control.
    R        Case(exact=False): production's construction and seam, unwrapped.
    Ep       E': R's objects with every accessor of the five terms delegated to the exact one.
    Ep+i     E' with term i's accessors realistic, the rest exact.
    R-i      R with term i's accessors exact, the rest realistic.

A hybrid is R's `Case` with a set D of terms delegated (`apply_hybrid`). Delegation is by
accessor name and applies to T_q and T_r together; R's geometry -- regions, switch nodes, the
partition, z_init, the hand-over record -- is kept, because only accessors are swapped:

    term   accessors delegated                      exact counterpart
    f      QuadSource.functions.source              Case.f_exact (FakeQuadSource(exact=True)'s)
    Tnum   T.T, T.dT_dz                             test_phase_groups.ExactTk
    Trep   T.M, T.phase                             ExactTk (phase: an _ExactPhase)
    TLG    T.omega, T.dlnM_dz                       ExactTk
    G      Gk.phase, Gk.sin_amplitude               test_phase_groups.ExactGk (GkSourceFunctions._replace)

Which regime reads which accessor is fixed by the code at `HEAD` (state/terms.md has the full
mapping): `numeric_quad_integral` reads only `source` and `numeric_Gk`, on the all-smooth
sub-interval; `phase_groups._SmoothT` reads `T`, `dT_dz` (a smooth factor in a Levin
sub-interval); `_OscillatoryT` reads `M`, `dlnM_dz`, `omega` and `phase.raw_theta`,
`phase.theta_mod_2pi`, and takes its phase derivative from `omega`; `_OscillatoryG` reads
`sin_amplitude` and `phase` (including `phase.theta_deriv`); `_SmoothG` reads `numeric_Gk`, which
is `compute_analytic_G` in R and E alike and so belongs to no term. `TkSourceFunctions.T` never
dispatches to the WKB form (it returns 1 above the numeric grid and the numeric spline within),
so name-delegation is region-delegation.

**What each cell records** (prompt 13 section 2.4): A4's fields -- per phase group and Levin
sub-interval `num_regions`, `max_depth`, `num_phase_limited_regions`, `converged`, value and
abserr, through the transparent wrapper of `phase_group_Levin_integral`; per cell
`Levin_regions`, the maximum depth and its group, `total`, `total_abserr`, the declared relative
error as A2 forms it, the reference, `setup_time`, `integral_time` and the load; the
`partition["handover"]` record; and which accessors were delegated, with each factor's phase
class and `exact_remainder`.

Usage:

    python -B b2_cells.py --cell SHAPE X_RESP OBJECT --out PATH.json

writes one cell's record and exits 0, or writes a `raised` record and exits 3.

No Ray runtime, no datastore, no lock. No repository file is modified. **After landing, this file
is a record of its own SHA-256**: the run's checkpoint records carry that hash, and it is not to
be edited.
"""

import argparse
import inspect
import json
import os
import socket
import sys
import time
import traceback
import warnings
from collections import Counter
from math import exp, expm1, log
from pathlib import Path

import numpy as np


def _find_repo() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ComputeTargets").is_dir() and (
            parent / "docs" / "handover" / "realistic_large_x.py"
        ).is_file():
            return parent
    raise RuntimeError("b2_cells: cannot find the repository root above this file")


REPO = _find_repo()
for _p in (str(REPO / "docs" / "handover"), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import realistic_large_x as A2  # noqa: E402
from realistic_large_x import (  # noqa: E402
    REF_ATOL,
    REF_RTOL,
    RUNS,
    fixture_x_max,
    reference_of,
    scaled_shape,
    seed_fixtures,
)
from ComputeTargets.QuadSource import QuadSourceFunctions  # noqa: E402
from ComputeTargets.tests import kohri_terada as KT  # noqa: E402
from ComputeTargets.tests.test_phase_groups import ExactGk, ExactTk  # noqa: E402
from ComputeTargets.tests.test_quadsource_integral import (  # noqa: E402
    SHAPES,
    Case,
    exact_with_the_geometry_of,
    qsi_module,
    w_of_b,
)

TERMS = ("f", "Tnum", "Trep", "TLG", "G")
T_ACCESSORS = {
    "Tnum": ("T", "dT_dz"),
    "Trep": ("M", "phase"),
    "TLG": ("omega", "dlnM_dz"),
}
G_ACCESSORS = ("phase", "sin_amplitude")
ACCESSORS = {
    "f": ("QuadSource.functions.source",),
    "Tnum": tuple(f"T.{a}" for a in T_ACCESSORS["Tnum"]),
    "Trep": tuple(f"T.{a}" for a in T_ACCESSORS["Trep"]),
    "TLG": tuple(f"T.{a}" for a in T_ACCESSORS["TLG"]),
    "G": tuple(f"Gk.{a}" for a in G_ACCESSORS),
}

OBJECTS = (
    ("E", "R", "Ep")
    + tuple(f"Ep+{t}" for t in TERMS)
    + tuple(f"R-{t}" for t in TERMS)
)

# A2's ladder, all of it (prompt 13 section 2.4): 16 rungs
LADDER = {name: tuple(rungs) for name, rungs in RUNS.items()}
assert LADDER == {
    "together": (980.0, 1e4, 1e5, 1e6, 1e7, 1e8),
    "T-first": (980.0, 1e4, 1e5, 1e6, 1e7),
    "q-smooth": (980.0, 1e4, 1e5, 1e6, 1e7),
}, LADDER


def sha256_of(path) -> str:
    import hashlib

    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


SCRIPT_SHA256 = sha256_of(__file__)


def delegated_terms(obj: str):
    """The set D of terms whose accessors are exact, or None for E and R (no hybrid)."""
    if obj in ("E", "R"):
        return None
    if obj == "Ep":
        return set(TERMS)
    if obj.startswith("Ep+"):
        return set(TERMS) - {obj[3:]}
    if obj.startswith("R-"):
        return {obj[2:]}
    raise ValueError(obj)


# ================================================================================================
# the hybrids


class HybridT:
    """
    A TkSourceFunctions-protocol object: R's object for everything (regions, switch node, z_init,
    hand-over record), with the accessors named in `names` taken from the exact stand-in instead.
    """

    ACCESSORS = ("T", "dT_dz", "M", "phase", "omega", "dlnM_dz")

    def __init__(self, real, exact, names):
        self._real = real
        self._exact = exact
        self.delegated = tuple(sorted(names))
        for name in self.ACCESSORS:
            setattr(self, name, getattr(exact if name in names else real, name))

    def __getattr__(self, name):
        return getattr(self._real, name)


def apply_hybrid(case, delegated):
    """Delegate the terms in `delegated` of the realistic `case`, in place. Returns a record."""
    if case.exact:
        raise RuntimeError("b2_cells: a hybrid starts from the realistic Case")
    shape, w, model = case.shape, case.w, case.model
    names = set()
    for term in ("Tnum", "Trep", "TLG"):
        if term in delegated:
            names |= set(T_ACCESSORS[term])
    exact_T = {
        float(shape.q): ExactTk(shape.q, w, model),
        float(shape.r): ExactTk(shape.r, w, model),
    }
    real_builder = case.Tk_builder

    def builder(model_, k, Tn, Tw):
        return HybridT(real_builder(model_, k, Tn, Tw), exact_T[float(k)], names)

    case.Tk_builder = builder

    if "G" in delegated:
        gk = ExactGk(shape.k, w, case.z_resp, model)
        case.GkPolicy.functions = case.GkPolicy.functions._replace(
            phase=gk.phase, sin_amplitude=gk.sin_amplitude
        )

    if "f" in delegated:

        def source(x, z_is_log=False):
            z = exp(x) - 1.0 if z_is_log else x
            return case.f_exact(z)

        case.source.functions = QuadSourceFunctions(
            source=source, numeric_region=case.source.numeric_region
        )

    return {
        "terms": sorted(delegated),
        "accessors": sorted(a for t in delegated for a in ACCESSORS[t]),
        "T_names": sorted(names),
    }


def build_case(shape_name: str, x_resp: float, obj: str):
    """The `Case` of one cell, ready to `run`. Returns (case, delegation record or None)."""
    if obj not in OBJECTS:
        raise ValueError(obj)
    base = [s for s in SHAPES if s.name == shape_name][0]
    w = w_of_b(0.0)
    shape = scaled_shape(base, x_resp)
    seed_fixtures(w, shape, x_resp)
    if obj == "E":
        return Case(b=0.0, shape=shape, x_resp=x_resp, exact=True), None
    case = Case(b=0.0, shape=shape, x_resp=x_resp, exact=False)
    delegated = delegated_terms(obj)
    if delegated is None:
        return case, None
    return case, apply_hybrid(case, delegated)


# ================================================================================================
# records of the objects


def _phase_record(phase) -> dict:
    if phase is None:
        return {"class": None}
    leading = getattr(phase, "_leading", None)
    return {
        "class": type(phase).__name__,
        "module": type(phase).__module__,
        "exact_remainder": getattr(phase, "exact_remainder", None),
        "leading_class": None if leading is None else type(leading).__name__,
        "sign": getattr(phase, "sign", None),
    }


def _owner(fn) -> str:
    return type(getattr(fn, "__self__", fn)).__name__


def phase_objects(case) -> dict:
    """Each factor's phase class and `exact_remainder`, and the owners of every accessor."""
    G = case.GkPolicy.functions
    out = {
        "G": _phase_record(G.phase),
        "G_sin_amplitude_owner": _owner(G.sin_amplitude),
        "G_numeric_Gk_owner": _owner(G.numeric_Gk),
        "source_owner": type(case.source.functions.source).__name__,
    }
    for label, inputs, k in (
        ("Tq", case.Tq_inputs, case.q.k),
        ("Tr", case.Tr_inputs, case.r.k),
    ):
        T = case.Tk_builder(case.model, k, *inputs)
        rec = _phase_record(getattr(T, "phase", None))
        rec["object_class"] = type(T).__name__
        rec["z_init"] = float(T.z_init)
        rec["crossover_z"] = float(T.crossover_z)
        rec["WKB_region_top"] = float(T.WKB_region[0])
        rec["accessor_owners"] = {
            name: _owner(getattr(T, name)) if name != "phase" else type(T.phase).__name__
            for name in HybridT.ACCESSORS
        }
        out[label] = rec
    return out


def construction_problems(obj: str, objects: dict, delegation) -> list:
    """R's objects are Part A's test (a); a hybrid's accessors come from where D says."""
    problems = []
    if obj == "E":
        return problems
    D = set() if delegation is None else set(delegation["terms"])
    G = objects["G"]
    if "G" in D:
        if G["class"] != "_ExactPhase" or objects["G_sin_amplitude_owner"] != "ExactGk":
            problems.append(f"{obj}: G not delegated: {G}")
    else:
        if (
            G["class"] != "PrimitivePhase"
            or G["exact_remainder"] is not True
            or G["leading_class"] != "TablePrimitive"
        ):
            problems.append(f"{obj}: G phase is {G}, not production's")
        if objects["G_sin_amplitude_owner"] != "BesselPhaseGk":
            problems.append(f"{obj}: G amplitude belongs to {objects['G_sin_amplitude_owner']}")
    want_source = "function" if "f" in D else "ZSplineWrapper"
    if objects["source_owner"] != want_source:
        problems.append(f"{obj}: source is {objects['source_owner']}, want {want_source}")
    names = set()
    for term in ("Tnum", "Trep", "TLG"):
        if term in D:
            names |= set(T_ACCESSORS[term])
    for label in ("Tq", "Tr"):
        T = objects[label]
        if not T["WKB_region_top"] < T["z_init"]:
            problems.append(f"{obj}: {label} WKB region starts at {T['WKB_region_top']!r}, not below z_init")
        if T["crossover_z"] != T["WKB_region_top"]:
            problems.append(f"{obj}: {label} switch node is not its WKB region's top")
        for name in HybridT.ACCESSORS:
            owner = T["accessor_owners"][name]
            if name == "phase":
                want = "_ExactPhase" if name in names else "PrimitivePhase"
            else:
                want = "ExactTk" if name in names else "TkSourceFunctions"
            if owner != want:
                problems.append(f"{obj}: {label}.{name} belongs to {owner}, want {want}")
        if "phase" not in names and (
            T["class"] != "PrimitivePhase"
            or T["leading_class"] != "TablePrimitive"
            or T["exact_remainder"] is not True
        ):
            problems.append(f"{obj}: {label} phase is {T}, not production's")
    return problems


# ================================================================================================
# the wrapper (A4's, unchanged in substance)


class GroupRecorder:
    """
    Wraps `ComputeTargets.QuadSourceIntegral.phase_group_Levin_integral` at its module attribute.
    The wrapper calls the original with the arguments it was given, unchanged, returns its payload
    unchanged, and keeps the payload's `groups` and `Levin_results` side by side.
    """

    def __init__(self):
        self.calls = []
        self._original = None

    def __enter__(self):
        original = qsi_module.phase_group_Levin_integral
        self._original = original
        signature = inspect.signature(original)
        calls = self.calls

        def wrapped(*args, **kwargs):
            payload = original(*args, **kwargs)
            bound = signature.bind(*args, **kwargs).arguments
            calls.append(
                {
                    "regime": tuple(bound["regime"]),
                    "max_z": bound["max_z"],
                    "min_z": bound["min_z"],
                    "groups": payload["groups"],
                    "Levin_results": payload["Levin_results"],
                }
            )
            return payload

        qsi_module.phase_group_Levin_integral = wrapped
        return self

    def __exit__(self, *exc):
        qsi_module.phase_group_Levin_integral = self._original
        return False


def _group_summary(call_index: int, call: dict, group: dict, data: dict) -> dict:
    regions = data["regions"]
    depths = [region.depth for region in regions]
    region_max = max(depths) if depths else None
    deepest = [region for region in regions if region.depth == region_max]
    lo = min(min(r.start, r.end) for r in deepest) if deepest else None
    hi = max(max(r.start, r.end) for r in deepest) if deepest else None
    regime = "".join(n for n, f in zip(("G", "q", "r"), call["regime"]) if f)
    return {
        "subinterval": call_index,
        "regime": regime,
        "z_max": call["max_z"],
        "z_min": call["min_z"],
        "label": group["label"],
        "num_regions": data["num_regions"],
        "num_simple_regions": data["num_simple_regions"],
        "max_depth": data["max_depth"],
        "max_region_depth": region_max,
        "num_phase_limited_regions": data["num_phase_limited_regions"],
        "converged": bool(data["converged"]),
        "phase_limited": bool(data["phase_limited"]),
        "value": group["value"],
        "abserr": group["abserr"],
        "raw_value": data["value"],
        "raw_abserr": data["abserr"],
        "evaluations": data["evaluations"],
        "elapsed": data["elapsed"],
        "depth_histogram": {
            str(d): n for d, n in sorted(Counter(depths).items(), key=lambda t: t[0])
        },
        "deepest_count": len(deepest),
        "deepest_z": None if lo is None else [expm1(lo), expm1(hi)],
    }


# ================================================================================================
# one cell


def run_cell(shape_name: str, x_resp: float, obj: str, instrument: bool = True) -> dict:
    """One cell, built from scratch and timed. Returns a flat JSON-able record."""
    warnings.simplefilter("ignore")
    load_before = os.getloadavg()

    t0 = time.perf_counter()
    case, delegation = build_case(shape_name, x_resp, obj)
    t1 = time.perf_counter()

    objects = phase_objects(case)
    problems = construction_problems(obj, objects, delegation)
    if problems:
        raise RuntimeError("b2_cells: construction check failed: " + "; ".join(problems))

    t2 = time.perf_counter()
    if instrument:
        with GroupRecorder() as recorder:
            out = case.run(atol=REF_ATOL, rtol=REF_RTOL)
    else:
        recorder = None
        out = case.run(atol=REF_ATOL, rtol=REF_RTOL)
    t3 = time.perf_counter()
    load_after = os.getloadavg()

    total = float(out["total"])
    halves = abs(float(out["numeric_quad"])) + abs(float(out["WKB_Levin"]))
    declared = max(float(out["total_abserr"]), REF_RTOL * halves) / abs(total)

    partition = out["metadata"]["partition"]
    handover = {
        label: {
            "z_init": rec["z_init"],
            "z_switch": rec["z_switch"],
            "handover_continuity": rec["handover_continuity"],
        }
        for label, rec in partition["handover"].items()
    }
    subintervals = [
        {
            "z_max": item["z_max"],
            "z_min": item["z_min"],
            "regime": list(item["regime"]),
            "method": item["method"],
        }
        for item in partition["subintervals"]
    ]
    regions_from_metadata = sum(
        group["regions"]
        for item in out["metadata"]["WKB_Levin"]["subintervals"]
        for group in item["groups"]
    )

    record = {
        "object": obj,
        "shape": shape_name,
        "x_resp": x_resp,
        "status": "ok",
        "delegated": delegation,
        "total": total,
        "total_abserr": float(out["total_abserr"]),
        "numeric_quad": float(out["numeric_quad"]),
        "WKB_Levin": float(out["WKB_Levin"]),
        "analytic_rad": float(out["analytic_rad"]),
        "declared": declared,
        "converged": bool(out["total_converged"]),
        "phase_limited": bool(out["total_phase_limited"]),
        "Levin_fraction": abs(float(out["WKB_Levin"])) / abs(total),
        "Levin_regions_from_metadata": regions_from_metadata,
        "handover": handover,
        "breakpoints": partition["breakpoints"],
        "partition_subintervals": subintervals,
        "phase_objects": objects,
        "construction_ok": True,
        "setup_time": t1 - t0,
        "check_time": t2 - t1,
        "integral_time": t3 - t2,
        "loadavg_before": list(load_before),
        "loadavg_after": list(load_after),
        "instrumented": instrument,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "b2_cells_sha256": SCRIPT_SHA256,
        "realistic_large_x_sha256": A2.SCRIPT_SHA256,
        "REF_ATOL": REF_ATOL,
        "REF_RTOL": REF_RTOL,
    }

    if recorder is not None:
        groups = []
        for index, call in enumerate(recorder.calls):
            if len(call["groups"]) != len(call["Levin_results"]):
                raise RuntimeError("b2_cells: groups and Levin_results differ in length")
            for group, data in zip(call["groups"], call["Levin_results"]):
                groups.append(_group_summary(index, call, group, data))
        record["groups"] = groups
        record["Levin_regions"] = sum(g["num_regions"] for g in groups)
        record["phase_limited_regions"] = sum(
            g["num_phase_limited_regions"] for g in groups
        )
        if groups:
            deepest = max(groups, key=lambda g: (g["max_depth"], g["num_regions"]))
            record["max_depth"] = deepest["max_depth"]
            record["max_depth_group"] = f"{deepest['subinterval']}:{deepest['regime']}:{deepest['label']}"
        else:
            record["max_depth"] = None
            record["max_depth_group"] = None
        if record["Levin_regions"] != regions_from_metadata:
            raise RuntimeError(
                f"b2_cells: wrapper counts {record['Levin_regions']} regions, the metadata "
                f"{regions_from_metadata}"
            )

    t4 = time.perf_counter()
    record["ref"] = reference_of(case, case.shape)
    record["reference_time"] = time.perf_counter() - t4
    record["N_plus_9_8"] = KT.KT_NORM * total / record["ref"]["predicted"] - KT.KT_NORM
    return record


def _write_json(path: Path, record: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(record, handle, indent=1, default=repr)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell", nargs=3, metavar=("SHAPE", "X_RESP", "OBJECT"), required=True
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    shape_name, x_text, obj = args.cell
    x_resp = float(x_text)
    out = Path(args.out)
    started = time.time()
    try:
        record = run_cell(shape_name, x_resp, obj)
        code = 0
    except Exception as exc:  # noqa: BLE001
        record = {
            "object": obj,
            "shape": shape_name,
            "x_resp": x_resp,
            "status": "raised",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "b2_cells_sha256": SCRIPT_SHA256,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        code = 3
    record["cell_wall"] = time.time() - started
    _write_json(out, record)
    sys.exit(code)


if __name__ == "__main__":
    main()
