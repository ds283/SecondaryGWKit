"""
handover prompt 04 (A4): one cell of the cost-growth measurement -- A2's ladder at `HEAD`, in four
arms that differ in the Green's function's phase alone.

A sibling of `docs/handover/realistic_large_x.py` (A2's harness), not an edit of it. It **imports**
from that module (`RUNS`, `scaled_shape`, `fixture_x_max`, `seed_fixtures`, `reference_of`,
`REF_ATOL`, `REF_RTOL`) and from the test modules, and replaces only what it must:

* **`run_cell`'s clamp reads.** B1 (handover prompt 12) removed `clamp_gaps_log1pz` and
  `max_clamp_gap_log1pz`; this copy reads `partition["handover"]` (`z_init`, `z_switch`,
  `handover_continuity` per factor) and the sub-interval records (`method`, `regime`, `z_max`,
  `z_min`) instead. That is `[12-realistic-large-x-reads-the-retired-clamp-records]`'s fix, in
  this copy only.
* **The case construction, per arm** (prompt 04 section 2.1):

      arm  G's phase                                      T_q, T_r                       seam
      E    ExactGk, closed form                           ExactTkFunctions, closed form  the exact flavour's
      S    OffsetBesselPhaseGk's phase_spline (A2's)      production's (below)           production's
      P0   PrimitivePhase on a CumulativeTable of 1/H,    production's                   production's
           exact_remainder False (production before 05)
      P    the same PrimitivePhase built by _build_phase,  production's                   production's
           exact_remainder True (production since 05)

  - E is A2's exact flavour with the closed seam, unchanged: `Case(exact=True)`.
  - S, P0 and P start from A2's realistic `Case` at `HEAD` (fixtures seeded by `seed_fixtures`).
  - **T in S, P0 and P** is built as `test_t_phase_reanchor._table_fixture_functions` builds it:
    a table-`cs_tau` fixture seeded as `seed_fixtures` seeds a `Fixture` (same w, k, x_max), with
    `drop_first_WKB_sample=True`; its samples captured with `exact_Tk_inputs`; the object built
    by `TkSourceFunctions(fixture.exact_envelope_model(), k, Tn, Tw)` with no keyword. **The
    fixture is `_UnionTableCsTauFixture`, not `_TableCsTauFixture`** (the user's two answers at
    A4's stops, 2026-10-26): `_TableCsTauFixture` builds its table on the dropped `z_WKB`, which
    starts one grid step below the phase anchor z_init, and `CumulativeTable` refuses z_init at
    rungs 1e4-1e6 by one ulp of `log1p`. This subclass builds the `cs_tau` `CumulativeTable` on
    the union of the fixture's numeric (`z_numeric`, `z_numeric_below`) and WKB nodes **without
    the one numeric node equal to z_init** (`z_numeric[-1] == crossover_z` bit for bit), so that
    the table covers z_init and z_init is off a node, as in production; the interval around it
    is two grid steps wide. The shift is `_TableCsTauFixture`'s: the closed form at the table's
    top node. The anchor's end partial `p_R` is recorded per fixture. The
    objects are built once per cell, during set-up, with the `k` `evaluate_QuadSource_integral`
    passes (`case.q.k`, `case.r.k`), and the builder hook returns them after checking that it
    was handed exactly those inputs.
  - **The seam** in S, P0 and P is production's since B1: the Liouville-Green samples begin at
    the largest grid node below z_init (`drop_first_WKB_sample=True`, as
    `test_quadsource_integral.production_seam_inputs`).
  - **G in P0 and P** is `_ProductionPhaseGk`'s `.old_phase` and `.phase`, built on the
    realistic `Case`'s own G grid (`log_grid(1 + G_crossover_z, 1 + z_resp, 100/decade)`, the
    grid `Case` hands `OffsetBesselPhaseGk`). If `_ProductionPhaseGk`'s own `phase_spline` (built
    by its `super().__init__`, then discarded) raises on that grid, the object is built with
    that one spline suppressed (`test_phase_groups.phase_spline` patched to return None for the
    duration of the constructor); `.phase` and `.old_phase` do not read it. Which path was taken
    is recorded per cell.
  - **Only `.phase` differs between S, P0 and P:** the policy's `GkSourceFunctions` namedtuple is
    `_replace`d in its `phase` field alone, so the amplitude and every other accessor are the
    same objects (`OffsetBesselPhaseGk`'s bound methods).

**Run 2** (the user's choice at A4's third stop, 2026-10-26). The first run's P0 and P cells raised
in `_ProductionPhaseGk`'s `tau` table: `Case`'s G grid ends at `exp(log(1 + z_resp)) - 1`, one ulp
below z_resp at most rungs, and `_ProductionPhaseGk` adds z_resp to the table's nodes, so the table
had two nodes with the same log(1+z) and refused every abscissa below them. This text passes
`_ProductionPhaseGk` `Case`'s G grid with its last node set exactly to z_resp (the amount it moved,
in ulp, is recorded per cell). And every cell now also runs `below_anchor_check` before its
integral: each phase is evaluated at a = log(1 + z_resp), at a minus 1-8 ulp and at the lowest
abscissa the Levin driver can form there (`lowest_levin_abscissa`); a P0 or P cell whose G phase
fails there is a construction failure.

**What every cell records** (prompt 04 section 2.2): `phase_group_Levin_integral` is wrapped at
its module attribute (how `evaluate_QuadSource_integral` calls it); the wrapper calls the original
unchanged and keeps its `groups` and `Levin_results` side by side. Nothing in `AdaptiveLevin/` is
instrumented. Per group and sub-interval: label, `num_regions`, `max_depth` (the driver's, and the
largest depth among its accepted regions), `num_phase_limited_regions`, `converged`, value and
abserr, a histogram of region depths, and the z-range of its deepest regions (a group's
`subinterval` is its wrapper call's index, i.e. its place among the Levin sub-intervals, not among
all of the partition's sub-intervals; `z_max` and `z_min` name it). Per cell:
`Levin_regions`, the maximum depth and the group carrying it, `total`, `total_abserr`, the declared
relative error as A2 forms it, the 50-digit reference, `setup_time`, `integral_time`, the load
average, and each phase object's class and `exact_remainder`.

Usage (from anywhere; the repository root is found by walking up from this file):

    python -B a4_cells.py --cell SHAPE X_RESP ARM --out PATH.json

writes one cell's record to PATH.json and exits 0, or writes a `raised` record and exits 3.

No Ray runtime, no datastore, no lock. No repository file is modified; `realistic_large_x.py` and
the test modules are imported, never edited. **After landing, this file is a record of its own
SHA-256**: the run's checkpoint records carry that hash, and it is not to be edited.
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
from math import expm1, sqrt, ulp
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _find_repo() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "ComputeTargets").is_dir() and (
            parent / "docs" / "handover" / "realistic_large_x.py"
        ).is_file():
            return parent
    raise RuntimeError("a4_cells: cannot find the repository root above this file")


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
from ComputeTargets.BackgroundModel import TAU_GAUSS_ORDER, TablePrimitive  # noqa: E402
from ComputeTargets.cumulative_table import CumulativeTable  # noqa: E402
from ComputeTargets.TkSourceFunctions import TkSourceFunctions  # noqa: E402
from ComputeTargets.primitive_phase import PrimitivePhase  # noqa: E402
from ComputeTargets.tests import kohri_terada as KT  # noqa: E402
from ComputeTargets.tests import test_phase_groups  # noqa: E402
from ComputeTargets.tests.test_g_phase_reanchor import (  # noqa: E402
    _ProductionPhaseGk,
    _TableCsTauFixture,
)
from ComputeTargets.tests.test_quadsource_integral import (  # noqa: E402
    SHAPES,
    Case,
    OffsetBesselPhaseGk,
    exact_Tk_inputs,
    qsi_module,
    w_of_b,
)
from ComputeTargets.tests.test_tk_source_functions import (  # noqa: E402
    PRODUCTION_SAMPLES_PER_LOG10Z,
    Fixture,
    log_grid,
)
from LiouvilleGreen.phase_spline import phase_spline  # noqa: E402
from AdaptiveLevin.levin_quadrature import (  # noqa: E402
    DEFAULT_LEVIN_MAX_DEPTH,
    _LEVIN_MINIMUM_ALLOWED_ORDER,
    chebyshev_matrices,
)

ARMS = ("E", "S", "P0", "P")

# A2's ladder (RUNS), with q-smooth to 1e6: the rung A2 could not reach, and no further
LADDER = {
    "together": tuple(RUNS["together"]),
    "T-first": tuple(RUNS["T-first"]),
    "q-smooth": tuple(x for x in RUNS["q-smooth"] if x <= 1.0e6),
}
assert LADDER == {
    "together": (980.0, 1e4, 1e5, 1e6, 1e7, 1e8),
    "T-first": (980.0, 1e4, 1e5, 1e6, 1e7),
    "q-smooth": (980.0, 1e4, 1e5, 1e6),
}, LADDER


def sha256_of(path) -> str:
    import hashlib

    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


SCRIPT_SHA256 = sha256_of(__file__)


# ================================================================================================
# the arms


class _UnionTableCsTauFixture(_TableCsTauFixture):
    """
    `_TableCsTauFixture` whose `cs_tau` table is built on the union of the fixture's numeric and
    WKB nodes, without the numeric node equal to z_init, descending, and shifted to the closed
    form at its top node as `_TableCsTauFixture` shifts it. Everything else is inherited.
    """

    def exact_envelope_model(self):
        model = Fixture.exact_envelope_model(self)
        nodes = {float(z) for z in self.z_numeric}
        nodes |= {float(z) for z in self.z_numeric_below}
        nodes |= {float(z) for z in self.z_WKB}
        nodes.discard(float(self.crossover_z))
        nodes = sorted(nodes, reverse=True)
        if not nodes[0] > self.crossover_z > nodes[-1]:
            raise RuntimeError("a4_cells: the union table does not cover z_init")
        w = self.w
        table = CumulativeTable(
            nodes, lambda z: sqrt(w) / model.Hubble(z), TAU_GAUSS_ORDER, label="cs_tau"
        ).shifted(model.cs_tau(nodes[0]))
        model.functions = model.functions._replace(
            cs_tau=TablePrimitive(table, "cs_tau")
        )
        return model


def _production_phase_gk(k, w, z_resp, model, z_grid):
    """`_ProductionPhaseGk`, or, if its inherited `phase_spline` raises, the same with that spline
    suppressed. Returns (object, path)."""
    try:
        return _ProductionPhaseGk(k, w, z_resp, model, z_grid), "_ProductionPhaseGk"
    except Exception as exc:  # noqa: BLE001
        first = f"{type(exc).__name__}: {exc}"
    with patch.object(test_phase_groups, "phase_spline", lambda *a, **kw: None):
        obj = _ProductionPhaseGk(k, w, z_resp, model, z_grid)
    return (
        obj,
        f"_ProductionPhaseGk without its inherited phase_spline, which raised {first}",
    )


def build_case(base, x_resp: float, arm: str):
    """
    The `Case` for one cell of one arm, ready to `run`. Returns (case, built) where `built` holds
    the T objects and the G phase builder's path, for the construction check.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    w = w_of_b(0.0)
    shape = scaled_shape(base, x_resp)
    seed_fixtures(w, shape, x_resp)
    exact = arm == "E"
    case = Case(b=0.0, shape=shape, x_resp=x_resp, exact=exact)
    built = {"T": None, "G_path": None, "G_grid": None}
    if exact:
        return case, built

    # --- T: production's, on a table cs_tau, production's seam ---------------------------------
    objects = {}
    for label, kk, k_obj in (("Tq", shape.q, case.q.k), ("Tr", shape.r, case.r.k)):
        fixture = _UnionTableCsTauFixture(
            w,
            k=kk,
            x_max=fixture_x_max(x_resp, kk, shape),
            drop_first_WKB_sample=True,
        )
        Tn, Tw = exact_Tk_inputs(fixture)
        T = TkSourceFunctions(fixture.exact_envelope_model(), k_obj, Tn, Tw)
        objects[float(kk)] = {
            "label": label,
            "fixture": fixture,
            "inputs": (Tn, Tw),
            "T": T,
        }
    case.Tq_inputs = objects[float(shape.q)]["inputs"]
    case.Tr_inputs = objects[float(shape.r)]["inputs"]

    def builder(model, k, Tn, Tw):
        entry = objects[float(k)]
        if Tn is not entry["inputs"][0] or Tw is not entry["inputs"][1]:
            raise RuntimeError(
                f"a4_cells: the T builder for k={float(k)!r} was handed inputs it did not build"
            )
        return entry["T"]

    case.Tk_builder = builder
    built["T"] = objects

    # --- G -------------------------------------------------------------------------------------
    if arm in ("P0", "P"):
        if case.G_type != "mixed":
            raise RuntimeError(f"a4_cells: G_type {case.G_type!r}, expected 'mixed'")
        # the grid Case hands OffsetBesselPhaseGk for a mixed G
        grid = log_grid(
            1.0 + case.G_crossover_z, 1.0 + case.z_resp, PRODUCTION_SAMPLES_PER_LOG10Z
        )
        if (max(grid), min(grid)) != tuple(case.gk.WKB_region):
            raise RuntimeError(
                f"a4_cells: the G grid {max(grid)!r}..{min(grid)!r} is not the Case's "
                f"{case.gk.WKB_region!r}"
            )
        # run 2 (the user's choice at A4's third stop): the grid's last node is
        # exp(log(1 + z_resp)) - 1, one ulp below z_resp at most rungs, and _ProductionPhaseGk
        # adds z_resp itself to its tau table's nodes, so the table had two nodes with the same
        # log(1+z) -- a zero-width bottom interval that refused every abscissa below it. Set the
        # last node exactly to z_resp, so that the table holds z_resp once.
        grid = [float(z) for z in grid]
        last = grid[-1]
        if abs(last - case.z_resp) > 4.0 * ulp(case.z_resp):
            raise RuntimeError(
                f"a4_cells: the G grid ends at {last!r}, not within 4 ulp of z_resp "
                f"{case.z_resp!r}"
            )
        grid[-1] = float(case.z_resp)
        if not grid[-2] > grid[-1]:
            raise RuntimeError(
                "a4_cells: the G grid is not strictly descending at its end"
            )
        production, path = _production_phase_gk(
            shape.k, w, case.z_resp, case.model, grid
        )
        phase = production.phase if arm == "P" else production.old_phase
        case.GkPolicy.functions = case.GkPolicy.functions._replace(phase=phase)
        built["G_path"] = path
        built["G_grid"] = {
            "nodes": len(grid),
            "z_max": max(grid),
            "z_min": min(grid),
            "case_grid_last_node": last,
            "last_node_moved_by_ulp": (case.z_resp - last) / ulp(case.z_resp),
            "tau_table_nodes": len(phase._leading._table._z),
        }
    return case, built


# ================================================================================================
# below the anchor: the abscissae the Levin driver can form at the bottom of the range


def lowest_levin_abscissa(a: float, b: float) -> dict:
    """
    The lowest abscissa `adaptive_levin_sincos` can form on (a, b) at its left end: every
    bisection child (a, (a + b_d) / 2) down to depth `DEFAULT_LEVIN_MAX_DEPTH + 1` (the comparison
    children of a capped region), at every Chebyshev order the driver can use (CHEBYSHEV_ORDER
    down to the minimum in steps of 2, each also at the Clenshaw-Curtis fallback's 2N - 1),
    through `chebyshev_matrices` itself.
    """
    orders = []
    n = max(qsi_module.CHEBYSHEV_ORDER, _LEVIN_MINIMUM_ALLOWED_ORDER)
    while n >= _LEVIN_MINIMUM_ALLOWED_ORDER:
        orders += [n, 2 * n - 1]
        n -= 2
    lowest = a
    where = None
    end = b
    for depth in range(DEFAULT_LEVIN_MAX_DEPTH + 2):
        for order in orders:
            grid, _ = chebyshev_matrices((a, end), order)
            m = float(min(grid))
            if m < lowest:
                lowest, where = m, (depth, order)
        end = (end + a) / 2.0
    return {"lowest": lowest, "below_a_ulp": (a - lowest) / ulp(a), "where": where}


def below_anchor_check(case) -> dict:
    """
    Evaluates the phase of every factor oscillatory in the bottom sub-interval's regime (`theta_mod_2pi`, `raw_theta`, `theta_deriv`, in log(1+z),
    as `phase_groups` and the Levin driver call them) at the bottom of the integration range: at
    a = log(1 + z_resp) as `phase_group_Levin_integral` forms it, at a minus 1 to 8 ulp, and at
    the lowest abscissa the driver can form there. Returns a record; `G_ok` is whether G's phase
    evaluated at every point. The first run's construction check evaluated nothing below z_resp,
    which is why it could not see the zero-width bottom interval of its tau table.
    """
    partition = case.partition()
    items = [
        item
        for item in partition["subintervals"]
        if item["method"] == "Levin"
        and item["z_min"] == min(i["z_min"] for i in partition["subintervals"])
    ]
    out = {"subintervals": [], "G_ok": True, "T_ok": True}
    phases = {"G": case.GkPolicy.functions.phase}
    for label, inputs, k in (
        ("Tq", case.Tq_inputs, case.q.k),
        ("Tr", case.Tr_inputs, case.r.k),
    ):
        phases[label] = case.Tk_builder(case.model, k, *inputs).phase
    for item in items:
        a = log1p_like(item["z_min"])
        b = log1p_like(item["z_max"])
        low = lowest_levin_abscissa(a, b)
        points = [("a", a)]
        u = a
        for k in range(1, 9):
            u = float(np.nextafter(u, -np.inf))
            points.append((f"a-{k}ulp", u))
        points.append(("lowest Levin abscissa", low["lowest"]))
        regime = "".join(n for n, f in zip(("G", "q", "r"), item["regime"]) if f)
        results = {}
        for index, (label, phase) in enumerate(phases.items()):
            # only a factor oscillatory in this sub-interval's regime has its phase evaluated
            # there (a smooth factor's numeric representation is read instead)
            if not item["regime"][index]:
                results[label] = "inactive"
                continue
            failures = []
            for name, x in points:
                try:
                    phase.theta_mod_2pi(x, x_is_log=True)
                    phase.raw_theta(x, x_is_log=True)
                    phase.theta_deriv(x, x_is_log=True, log_derivative=True)
                except Exception as exc:  # noqa: BLE001
                    failures.append(f"{name}: {type(exc).__name__}: {exc}")
            results[label] = failures
            if failures and label == "G":
                out["G_ok"] = False
            if failures and label != "G":
                out["T_ok"] = False
        out["subintervals"].append(
            {
                "regime": regime,
                "z_max": item["z_max"],
                "z_min": item["z_min"],
                "a": a,
                "lowest": low,
                "failures": results,
            }
        )
    return out


def log1p_like(z: float) -> float:
    """`log(1.0 + z)`, exactly as `phase_group_Levin_integral` forms its x_span."""
    from math import log

    return log(1.0 + z)


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


def phase_objects(case, built) -> dict:
    """Each phase object's class and `exact_remainder`, as the cell runs them."""
    G = case.GkPolicy.functions
    out = {"G": _phase_record(G.phase)}
    if built["T"] is None:
        for label, inputs, k in (
            ("Tq", case.Tq_inputs, case.q.k),
            ("Tr", case.Tr_inputs, case.r.k),
        ):
            T = case.Tk_builder(case.model, k, *inputs)
            out[label] = _phase_record(getattr(T, "phase", None))
            out[label]["object_class"] = type(T).__name__
    else:
        for entry in built["T"].values():
            out[entry["label"]] = _phase_record(entry["T"].phase)
            out[entry["label"]]["object_class"] = type(entry["T"]).__name__
            out[entry["label"]]["fixture_class"] = type(entry["fixture"]).__name__
            out[entry["label"]]["drop_first_WKB_sample"] = bool(
                entry["fixture"].z_switch != entry["fixture"].crossover_z
            )
            phase = entry["T"].phase
            j_R, p_R = phase._leading.end_partial(phase.z_anchor)
            out[entry["label"]]["z_anchor"] = float(phase.z_anchor)
            out[entry["label"]]["z_init"] = float(entry["fixture"].crossover_z)
            out[entry["label"]]["anchor_node_index"] = phase._leading._table.node_index(
                phase.z_anchor
            )
            out[entry["label"]]["anchor_nearest_node"] = int(j_R)
            out[entry["label"]]["p_R"] = float(p_R)
    # the amplitude and every other accessor of G: the same object in S, P0 and P
    out["G_sin_amplitude_owner"] = type(
        getattr(G.sin_amplitude, "__self__", G.sin_amplitude)
    ).__name__
    out["G_path"] = built["G_path"]
    out["G_grid"] = built["G_grid"]
    return out


def construction_problems(arm: str, objects: dict) -> list:
    """Prompt 04 section 2.1's check. An empty list means the arm is what the prompt says."""
    problems = []
    G = objects["G"]
    if arm == "E":
        return problems
    if arm == "S":
        if (
            G["class"] != phase_spline.__name__
            or G["module"] != phase_spline.__module__
        ):
            problems.append(f"S: G phase is {G}, not a phase_spline")
    else:
        want = arm == "P"
        if G["class"] != PrimitivePhase.__name__ or G["exact_remainder"] is not want:
            problems.append(
                f"{arm}: G phase is {G}, not a PrimitivePhase with exact_remainder {want}"
            )
        if G["leading_class"] != TablePrimitive.__name__:
            problems.append(f"{arm}: G leading primitive is {G['leading_class']}")
    for label in ("Tq", "Tr"):
        T = objects[label]
        if (
            T["class"] != PrimitivePhase.__name__
            or T["leading_class"] != TablePrimitive.__name__
            or T["exact_remainder"] is not True
        ):
            problems.append(
                f"{arm}: {label} phase is {T}, not a PrimitivePhase on a TablePrimitive with "
                f"exact_remainder True"
            )
        if T.get("fixture_class") != _UnionTableCsTauFixture.__name__ or not T.get(
            "drop_first_WKB_sample"
        ):
            problems.append(f"{arm}: {label} fixture is {T}")
        if (
            T.get("z_anchor") != T.get("z_init")
            or T.get("anchor_node_index") is not None
            or not T.get("p_R")
        ):
            problems.append(
                f"{arm}: {label} anchor is not z_init off a node with p_R != 0: {T}"
            )
    if objects["G_sin_amplitude_owner"] != OffsetBesselPhaseGk.__name__:
        problems.append(
            f"{arm}: G's amplitude belongs to {objects['G_sin_amplitude_owner']}"
        )
    return problems


# ================================================================================================
# the wrapper


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
                    "atol": bound["atol"],
                    "rtol": bound["rtol"],
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
        # the deepest regions' extent, in log(1+z') and in z'
        "deepest_log1pz": None if lo is None else [lo, hi],
        "deepest_z": None if lo is None else [expm1(lo), expm1(hi)],
    }


# ================================================================================================
# one cell


def run_cell(shape_name: str, x_resp: float, arm: str, instrument: bool = True) -> dict:
    """
    One cell, built from scratch and timed. Returns a flat JSON-able record. With
    `instrument=False` the wrapper is not installed and the per-group fields are absent (the
    transparency check's other side).
    """
    warnings.simplefilter("ignore")
    base = [s for s in SHAPES if s.name == shape_name][0]
    load_before = os.getloadavg()

    t0 = time.perf_counter()
    case, built = build_case(base, x_resp, arm)
    t1 = time.perf_counter()

    objects = phase_objects(case, built)
    problems = construction_problems(arm, objects)
    below = below_anchor_check(case)
    if arm in ("P0", "P") and not below["G_ok"]:
        problems.append(f"{arm}: G's phase does not evaluate below z_resp: {below}")
    if problems:
        raise RuntimeError(
            "a4_cells: construction check failed: " + "; ".join(problems)
        )

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

    # A2's own count, from the stored metadata
    regions_from_metadata = sum(
        group["regions"]
        for item in out["metadata"]["WKB_Levin"]["subintervals"]
        for group in item["groups"]
    )

    record = {
        "arm": arm,
        "shape": shape_name,
        "x_resp": x_resp,
        "status": "ok",
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
        "WKB_phase_spline_chunks": out.get("WKB_phase_spline_chunks"),
        "handover": handover,
        "breakpoints": partition["breakpoints"],
        "partition_subintervals": subintervals,
        "phase_objects": objects,
        "construction_ok": True,
        "below_anchor": below,
        "setup_time": t1 - t0,
        "check_time": t2 - t1,
        "integral_time": t3 - t2,
        "loadavg_before": list(load_before),
        "loadavg_after": list(load_after),
        "instrumented": instrument,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "a4_cells_sha256": SCRIPT_SHA256,
        "realistic_large_x_sha256": A2.SCRIPT_SHA256,
        "REF_ATOL": REF_ATOL,
        "REF_RTOL": REF_RTOL,
    }

    if recorder is not None:
        groups = []
        for index, call in enumerate(recorder.calls):
            if len(call["groups"]) != len(call["Levin_results"]):
                raise RuntimeError(
                    "a4_cells: groups and Levin_results differ in length"
                )
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
            record["max_depth_group"] = {
                key: deepest[key]
                for key in (
                    "subinterval",
                    "regime",
                    "label",
                    "z_max",
                    "z_min",
                    "num_regions",
                    "deepest_count",
                    "deepest_z",
                    "deepest_log1pz",
                )
            }
            record["groups_at_max_depth"] = [
                f"{g['subinterval']}:{g['regime']}:{g['label']}"
                for g in groups
                if g["max_depth"] == deepest["max_depth"]
            ]
        else:
            record["max_depth"] = None
            record["max_depth_group"] = None
            record["groups_at_max_depth"] = []
        if record["Levin_regions"] != regions_from_metadata:
            raise RuntimeError(
                f"a4_cells: wrapper counts {record['Levin_regions']} regions, the metadata "
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
        "--cell", nargs=3, metavar=("SHAPE", "X_RESP", "ARM"), required=True
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    shape_name, x_text, arm = args.cell
    x_resp = float(x_text)
    out = Path(args.out)
    started = time.time()
    try:
        record = run_cell(shape_name, x_resp, arm)
        code = 0
    except Exception as exc:  # noqa: BLE001
        record = {
            "arm": arm,
            "shape": shape_name,
            "x_resp": x_resp,
            "status": "raised",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "a4_cells_sha256": SCRIPT_SHA256,
            "host": socket.gethostname(),
            "pid": os.getpid(),
        }
        code = 3
    record["cell_wall"] = time.time() - started
    _write_json(out, record)
    sys.exit(code)


if __name__ == "__main__":
    main()
