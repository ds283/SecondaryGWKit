"""The realistic flavour of `evaluate_QuadSource_integral` at large x, with and without the seam gap.

Runs from the repository root with no arguments, no Ray and no datastore:

    PYTHONPATH=. ./venv/bin/python docs/handover/realistic_large_x.py

Its stdout is the tables of `REALISTIC-LARGE-X.md`. It takes about three and a half hours on an
Apple M1 Pro; see "REALISTIC CEILING" and "CHECKPOINTING" below, and Table 8, for why.

This is a sibling of `docs/radiation-oracle/large_x.py`, not an edit of it. That script ran the
`exact` flavour only -- `ExactTk`/`ExactGk` closed-form stand-ins injected through
`evaluate_QuadSource_integral`'s `Tk_functions_builder` hook -- and section 8 of
`KOHRI-TERADA-ORACLE.md` closes by saying so: "the realistic flavour's representation floors
(splined phases and amplitudes) are not exercised at large x". This script adds that flavour, and
adds the seam gap that production has and neither flavour has by default, as a 2x2 factorial:

    flavour   exact                          realistic
              -------------------------      ---------------------------------------------
              ExactTkFunctions, closed       TkSourceFunctions built from prompt 05's exact
              form on both sides of a        fixture: amplitude re-splined and the phase
              nominal crossover_z            carried as a PrimitivePhase on the production
                                             100-per-log10z grid, Liouville-Green closed
                                             forms for omega and d ln M / dz; the Green's
                                             function phase a real phase_spline

    seam      closed                         open
              -------------------------      ---------------------------------------------
              WKB_region[0] == crossover_z   WKB_region[0] one grid step below crossover_z,
              -- no gap, nothing clamped     which QuadSourceIntegral bridges by clamping the
                                             Liouville-Green accessors across it

crossed with the three fixture shapes and the x_resp ladder of section 8.

Two mechanisms are used to open the seam, because one mechanism cannot serve both flavours:

  * realistic -- `Fixture(drop_first_WKB_sample=True)`, which is production's own shape
    (`main.py` truncates the WKB grid to the largest source-grid point at or below z_init) and is
    what `TestHandOverClamp` uses. The dropped fixture is swapped into `Case.Tq_inputs` /
    `Case.Tr_inputs` after the Case is built, so that nothing else about the case moves.
  * exact -- `drop_first_WKB_sample` is INERT in the exact flavour, because `ExactTkFunctions`
    ignores the sampled inputs entirely and declares `WKB_region = (crossover_z, 0.0)` from a
    closed form valid at every z. Table 6 measures that inertness. To put the same clamp on exact
    accessors, `_GappedExactTkFunctions` moves `WKB_region[0]` down by exactly the gap the dropped
    fixture opens -- the fixture's own first WKB grid step in log(1+z). The geometry
    `build_partition` and `_ClampedTk` then see is identical; only the ingredients differ.

Everything else follows `large_x.py`: k, q and r are multiplied by lam = x_resp / 980, so u = q/k
and v = r/k and hence the Kohri-Terada integral I(v, u, x) are unchanged and the response time
stays at the same redshift; `Case._fixtures` is seeded with `Fixture` objects whose
Liouville-Green region reaches the response redshift; and every case is scored against eq. (22)
and the head both at 50 digits (`eq22_rounding.I_RD_mp`, `head_mp`), so section 7.2's small-u
rounding never enters. The reference is identical across the four cells of a factorial block, so
it cancels identically in every difference below and each difference carries only the two
pipelines' own declared errors.

REALISTIC CEILING. The realistic ladder stops at a different rung per shape, and that is a
measurement rather than a convenience. The realistic *integral* cost -- not the set-up, which is
flat at a few hundredths of a second everywhere -- saturates at about 2x per decade of x on
`together` and `T-first`, so their full ladders are affordable. On `q-smooth` it grows about 7.3x
per decade instead, so the realistic ladder stops at x_resp = 1e5 there and the 1e6 and 1e7 cells
are reported as NOT REACHED, with the cost curve that makes them so (Table 8). The exact cells are
cheap at every rung and are run at all of them.

CHECKPOINTING. Each cell is appended to a JSON-Lines file as soon as it finishes, and a re-run
reads that file back and recomputes only the cells it does not already hold, so an interruption
costs one cell rather than the run. Every table is built from those records alone -- no table reads
a live `Case` -- so the whole document is reproducible from what is on disk.

The file is `var/runs/realistic_large_x_cells.jsonl` under the repository root, created if absent
and gitignored, overridable with REALISTIC_LARGE_X_CHECKPOINT. It is deliberately *not* in the
system temporary directory: a file whose whole purpose is to survive an interruption should not
live somewhere the operating system periodically reaps, or somewhere nobody who has not read this
script would think to look.

Each record is stamped with the SHA-256 of this script's own source and with `git rev-parse HEAD`
(and a flag for an unclean tree) as they stood when the cell ran. **On load, a record whose script
hash differs from the running script's is discarded and the cell recomputed**, with a notice on
stderr naming both hashes and the number of records dropped. Discarding rather than refusing is
chosen so that an edited script is self-healing instead of stranding the operator on a file they
must know to delete; the notice is what makes the discard visible. Records are never blended
across script versions.

A cell that *raises* is recorded too, with its reason, so that a permanent failure is durable and
is not retried on every run. A cell that was killed mid-flight writes nothing and is simply
recomputed -- which is the distinction between a failure the code produced and an interruption
imposed on it. Delete a line, or the file, to force a recompute.

No repository file is modified. `ComputeTargets/tests/test_quadsource_integral.py` and
`docs/radiation-oracle/large_x.py` are imported, never edited.
"""

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import warnings
from math import exp, log

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "radiation-oracle"
    ),
)

from eq22_rounding import I_RD_mp, head_mp  # noqa: E402
from large_x import eq25_without_cos  # noqa: E402
from ComputeTargets.tests import kohri_terada as KT  # noqa: E402
from ComputeTargets.tests.test_quadsource_integral import (  # noqa: E402
    Case,
    ExactTkFunctions,
    SHAPES,
    Shape,
    exact_Tk_inputs,
    w_of_b,
)
from ComputeTargets.tests.test_tk_source_functions import Fixture  # noqa: E402

REF_ATOL, REF_RTOL = 1e-45, 1e-12

# the section 8 ladder: the top of each list is the largest x the fixture's Liouville-Green
# region reaches
RUNS = {
    "together": (980.0, 1e4, 1e5, 1e6, 1e7, 1e8),
    "T-first": (980.0, 1e4, 1e5, 1e6, 1e7),
    "q-smooth": (980.0, 1e4, 1e5, 1e6, 1e7),
}

# the top of the *realistic* ladder, per shape. See "REALISTIC CEILING" in the module docstring
# and Table 8: q-smooth's realistic cells grow ~7.3x per decade where the other two saturate at
# ~2x, so its 1e6 pair is hours each and its 1e7 pair ~12 h each. Both are reported as not
# reached rather than run. The 1e6 lower bound quoted in the document (>5600 s for the closed
# cell, without completing) comes from an earlier run that was killed, and is labelled as such.
REALISTIC_CEILING = {"together": 1.0e8, "T-first": 1.0e7, "q-smooth": 1.0e5}

# prompt 12 of prompts/source-remediation, [12-handover-clamp-error-in-production]: the gaps
# actually recorded on production rows, in log(1+z)
PRODUCTION_GAP_MEDIAN = 1.2e-2
PRODUCTION_GAP_MAX = 2.2e-2

CELLS = (
    ("exact", "closed"),
    ("exact", "open"),
    ("realistic", "closed"),
    ("realistic", "open"),
)
MISSING = "--"
NOT_REACHED = "not reached"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.environ.get(
    "REALISTIC_LARGE_X_CHECKPOINT",
    os.path.join(REPO_ROOT, "var", "runs", "realistic_large_x_cells.jsonl"),
)


def script_sha256() -> str:
    """SHA-256 of this file's own source: a record made by a different script is not reused."""
    with open(os.path.abspath(__file__), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def git_provenance() -> dict:
    """`git rev-parse HEAD` and whether the tree was clean, as they stood when the cell ran."""

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


SCRIPT_SHA256 = script_sha256()


class _GappedExactTkFunctions(ExactTkFunctions):
    """
    `ExactTkFunctions` whose Liouville-Green region starts `gap_log1pz` below the hand-over, so
    that `build_partition` records a gap and `_ClampedTk` holds the accessors across it exactly
    as it does for a `drop_first_WKB_sample` fixture -- but with closed-form accessors, so the
    only thing being measured is the clamp.
    """

    def __init__(self, k, w, model, crossover_z, gap_log1pz):
        super().__init__(k, w, model, crossover_z)
        self.WKB_region = (exp(log(1.0 + crossover_z) - gap_log1pz) - 1.0, 0.0)


def scaled_shape(base: Shape, x_resp: float) -> Shape:
    """`large_x.py`'s scaling: k, q, r all times lam, so u, v and I(v, u, x) are unchanged."""
    lam = x_resp / 980.0
    return Shape(
        base.name,
        k=lam * base.k,
        q=lam * base.q,
        r=lam * base.r,
        G_cross_x=base.G_cross_x,
    )


def fixture_x_max(x_resp: float, kk: float, shape: Shape) -> float:
    """`large_x.py`'s rule: extend the Liouville-Green region only where the response needs it."""
    return max(1.0e3, 1.05 * x_resp * kk / shape.r)


def seed_fixtures(w: float, shape: Shape, x_resp: float) -> None:
    for kk in (shape.q, shape.r):
        Case._fixtures[(w, kk)] = Fixture(
            w, k=kk, x_max=fixture_x_max(x_resp, kk, shape)
        )


def first_WKB_step(fixture: Fixture) -> float:
    """
    The gap `drop_first_WKB_sample=True` opens, in log(1+z): the fixture's own first WKB grid
    step. `Fixture.crossover_z` is `z_WKB[0]` whether or not the sample is dropped, so after the
    drop `TkSourceFunctions.WKB_region[0]` is `z_WKB[1]` and the shortfall at the breakpoint is
    exactly this.
    """
    return log(1.0 + fixture.z_WKB[0]) - log(1.0 + fixture.z_WKB[1])


def dropped_inputs(w: float, shape: Shape, x_resp: float, kk: float):
    return exact_Tk_inputs(
        Fixture(
            w,
            k=kk,
            x_max=fixture_x_max(x_resp, kk, shape),
            drop_first_WKB_sample=True,
        )
    )


def open_the_seam(
    case: Case, shape: Shape, w: float, x_resp: float, exact: bool
) -> dict:
    """
    Move both transfer functions' Liouville-Green regions one grid step below their hand-overs,
    by the mechanism appropriate to the flavour. Returns the gap offered per factor, in log(1+z).
    """
    gaps = {
        "Tq": first_WKB_step(Case._fixtures[(w, shape.q)]),
        "Tr": first_WKB_step(Case._fixtures[(w, shape.r)]),
    }
    if exact:
        functions = {
            float(shape.q): _GappedExactTkFunctions(
                shape.q, w, case.model, case.Fq.crossover_z, gaps["Tq"]
            ),
            float(shape.r): _GappedExactTkFunctions(
                shape.r, w, case.model, case.Fr.crossover_z, gaps["Tr"]
            ),
        }
        case.Tk_builder = lambda model, k, Tn, Tw: functions[float(k)]
    else:
        case.Tq_inputs = dropped_inputs(w, shape, x_resp, shape.q)
        case.Tr_inputs = dropped_inputs(w, shape, x_resp, shape.r)
    return gaps


def reference_of(case: Case, shape: Shape) -> dict:
    """Eq. (22) and the head at 50 digits, between the code's own limits."""
    k = shape.k
    u, v = shape.q / k, shape.r / k
    tau = case.model.functions.tau
    x = k * tau(case.z_resp)
    x_min = k * tau(case.z_source_max)
    I_exact = I_RD_mp(v, u, x)
    head = head_mp(v, u, x, x_min)
    asym = (
        eq25_without_cos(v, u, x)
        if abs(v - u) > KT.SQRT3
        else KT.I_RD_asymptotic(v, u, x)
    )
    return {
        "u": u,
        "v": v,
        "x": x,
        "x_min": x_min,
        "z_resp": case.z_resp,
        "predicted": KT.KT_NORM / (k * k) * float(I_exact - head),
        "predicted_no_head": KT.KT_NORM / (k * k) * float(I_exact),
        "vs_eq25": abs(float(I_exact) - asym) / abs(float(I_exact)),
        "eq25_note": " (no cos term)" if abs(v - u) > KT.SQRT3 else "",
    }


def run_cell(base: Shape, x_resp: float, flavour: str, seam: str) -> dict:
    """
    One cell of the factorial, built from scratch and timed. Returns a flat, JSON-able record:
    no table below reads a live `Case`, so the whole document is reproducible from the
    checkpoint file.
    """
    w = w_of_b(0.0)
    shape = scaled_shape(base, x_resp)
    exact = flavour == "exact"

    t0 = time.perf_counter()
    seed_fixtures(w, shape, x_resp)
    case = Case(b=0.0, shape=shape, x_resp=x_resp, exact=exact)
    offered = (
        open_the_seam(case, shape, w, x_resp, exact)
        if seam == "open"
        else {"Tq": 0.0, "Tr": 0.0}
    )
    t1 = time.perf_counter()
    out = case.run(atol=REF_ATOL, rtol=REF_RTOL)
    t2 = time.perf_counter()

    total = float(out["total"])
    halves = abs(float(out["numeric_quad"])) + abs(float(out["WKB_Levin"]))
    declared = max(float(out["total_abserr"]), REF_RTOL * halves) / abs(total)

    partition = out["metadata"]["partition"]
    recorded = {"Tq": 0.0, "Tr": 0.0, "source": 0.0}
    for item in partition["subintervals"]:
        for label, value in item["clamp_gaps_log1pz"].items():
            recorded[label] = max(recorded.get(label, 0.0), value)

    regions = 0
    levin_elapsed = 0.0
    for item in out["metadata"]["WKB_Levin"]["subintervals"]:
        for group in item["groups"]:
            regions += group["regions"]
            levin_elapsed += group["elapsed"]

    record = {
        "shape": base.name,
        "x_resp": x_resp,
        "flavour": flavour,
        "seam": seam,
        "total": total,
        "declared": declared,
        "converged": bool(out["total_converged"]),
        "phase_limited": bool(out["total_phase_limited"]),
        "Levin_fraction": abs(float(out["WKB_Levin"])) / abs(total),
        "Levin_regions": regions,
        "Levin_elapsed": levin_elapsed,
        "offered_gaps": offered,
        "recorded_gaps": recorded,
        "max_clamp_gap": partition["max_clamp_gap_log1pz"],
        "setup_time": t1 - t0,
        "integral_time": t2 - t1,
        "fixture_step": first_WKB_step(Case._fixtures[(w, shape.r)]),
        "status": "ok",
    }
    record["ref"] = reference_of(case, shape)
    return record


# ================================================================================================
# the checkpoint


def cell_key(shape_name: str, x_resp: float, flavour: str, seam: str) -> str:
    return f"{shape_name}|{x_resp:g}|{flavour}|{seam}"


def load_checkpoint() -> dict:
    """Every cell already on disk, keyed as `cell_key`. A malformed trailing line is ignored:
    the file is appended to as each cell lands, so a kill can truncate the last record.
    """
    cells = {}
    stale = {}
    if not os.path.exists(CHECKPOINT):
        return cells
    with open(CHECKPOINT, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = cell_key(
                record["shape"], record["x_resp"], record["flavour"], record["seam"]
            )
            if record.get("script_sha256") != SCRIPT_SHA256:
                stale[key] = record.get("script_sha256", "unstamped")
                cells.pop(key, None)
                continue
            cells[key] = record
    if stale:
        print(
            f"checkpoint: discarding {len(stale)} record(s) written by a different version of "
            f"this script ({', '.join(sorted({h[:12] for h in stale.values()}))}); this script "
            f"is {SCRIPT_SHA256[:12]}. Those cells will be recomputed.",
            file=sys.stderr,
            flush=True,
        )
    return cells


def append_checkpoint(record: dict) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT) or ".", exist_ok=True)
    with open(CHECKPOINT, "a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


# ================================================================================================
# derived quantities


def N_of(cell: dict, ref: dict) -> float:
    return KT.KT_NORM * cell["total"] / ref["predicted"]


def term(block: dict, hi, lo):
    """One difference of two cells' N, and the sum of their declared errors in N."""
    a, b = block["cells"].get(hi), block["cells"].get(lo)
    if a is None or b is None:
        return None, None
    Na, Nb = N_of(a, block["ref"]), N_of(b, block["ref"])
    return Na - Nb, abs(Na) * a["declared"] + abs(Nb) * b["declared"]


def fmt(value, spec="+.3e"):
    return MISSING if value is None else format(value, spec)


def slope(xs, ys) -> float:
    """Least-squares slope of log|y| against log x."""
    pairs = [(log(a), log(abs(b))) for a, b in zip(xs, ys) if b not in (None, 0.0)]
    if len(pairs) < 2:
        return float("nan")
    mx = statistics.fmean(p[0] for p in pairs)
    my = statistics.fmean(p[1] for p in pairs)
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    den = sum((p[0] - mx) ** 2 for p in pairs)
    return num / den if den != 0.0 else float("nan")


TERMS = (
    ("clamp term, exact", ("exact", "open"), ("exact", "closed")),
    ("clamp term, realistic", ("realistic", "open"), ("realistic", "closed")),
    ("representation term, gap closed", ("realistic", "closed"), ("exact", "closed")),
    ("representation term, gap open", ("realistic", "open"), ("exact", "open")),
)


# ================================================================================================
# the tables


def control_table(blocks):
    """Table 1: the exact x gap-closed cells, in the columns of KOHRI-TERADA-ORACLE.md Table 8.1."""
    print(
        "**Table 1 -- the control cell against Kohri & Terada section 8, Table 8.1.** Same "
        "columns, same order, same script inputs; the only difference is that this harness "
        "builds the case up to four times per row."
    )
    print()
    print(
        "| shape | x_resp | lam | x = k tau | z_resp | N + 9/8 | pipeline's declared error "
        "| N + 9/8, head omitted | Levin / abs(total) | eq. (22) vs eq. (25) "
        "| integral | fixture set-up |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for block in blocks:
        ref = block["ref"]
        cell = block["cells"][("exact", "closed")]
        N = N_of(cell, ref)
        N_no_head = KT.KT_NORM * cell["total"] / ref["predicted_no_head"]
        print(
            f"| {block['shape']} | {block['x_resp']:g} | {block['x_resp'] / 980.0:.3g} "
            f"| {ref['x']:.4e} | {ref['z_resp']:.3g} | {N - KT.KT_NORM:+.2e} "
            f"| {cell['declared']:.1e} | {N_no_head - KT.KT_NORM:+.2e} "
            f"| {cell['Levin_fraction']:.2f} | {ref['vs_eq25']:.1e}{ref['eq25_note']} "
            f"| {cell['integral_time']:.2f} s | {cell['setup_time']:.1f} s |",
            flush=True,
        )


def factorial_table(blocks):
    """Table 2: every cell of every block."""
    print(
        "**Table 2 -- the factorial.** `gap Tq` and `gap Tr` are the shortfalls "
        "`build_partition` actually recorded in `clamp_gaps_log1pz`, and `1.5 grid steps` is "
        "`max_clamp_gap_log1pz`, the most the clamp will bridge."
    )
    print()
    print(
        "| shape | x_resp | x = k tau | flavour | seam | N + 9/8 | pipeline's declared error "
        "| converged | Levin / abs(total) | Levin regions | gap Tq | gap Tr | 1.5 grid steps "
        "| integral | set-up |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for block in blocks:
        ref = block["ref"]
        for key in CELLS:
            cell = block["cells"].get(key)
            if cell is None:
                why = block["failures"].get(key, NOT_REACHED)
                print(
                    f"| {block['shape']} | {block['x_resp']:g} | {ref['x']:.4e} | {key[0]} "
                    f"| {key[1]} | {MISSING} | {MISSING} | {MISSING} | {MISSING} | {MISSING} "
                    f"| {MISSING} | {MISSING} | {MISSING} | {why} | {why} |",
                    flush=True,
                )
                continue
            N = N_of(cell, ref)
            print(
                f"| {block['shape']} | {block['x_resp']:g} | {ref['x']:.4e} | {cell['flavour']} "
                f"| {cell['seam']} | {N - KT.KT_NORM:+.3e} | {cell['declared']:.1e} "
                f"| {'yes' if cell['converged'] else 'no'} | {cell['Levin_fraction']:.2f} "
                f"| {cell['Levin_regions']} "
                f"| {cell['recorded_gaps']['Tq']:.3e} | {cell['recorded_gaps']['Tr']:.3e} "
                f"| {cell['max_clamp_gap']:.3e} | {cell['integral_time']:.2f} s "
                f"| {cell['setup_time']:.2f} s |",
                flush=True,
            )


def attribution_table(blocks):
    """Table 3: the deliverable -- the clamp term and the representation term."""
    print(
        "**Table 3 -- the attribution.** Each term is a difference of two N scored against the "
        "*same* 50-digit reference, so eq. (22) and the head cancel identically and the error "
        "beside each term is the sum of the two cells' own declared errors, converted to N. "
        "`interaction` is (clamp, realistic) - (clamp, exact), equivalently "
        "(representation, open) - (representation, closed): zero if the 2x2 is additive."
    )
    print()
    print(
        "| shape | x_resp | x = k tau | clamp term, exact | +/- | clamp term, realistic | +/- "
        "| representation term, gap closed | +/- | representation term, gap open | +/- "
        "| interaction | h^4 x_resp / 384 | h^4 (k tau) / 384 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for block in blocks:
        values = {label: term(block, hi, lo) for label, hi, lo in TERMS}
        ce, ce_err = values["clamp term, exact"]
        cr, cr_err = values["clamp term, realistic"]
        rc, rc_err = values["representation term, gap closed"]
        ro, ro_err = values["representation term, gap open"]
        interaction = None if (ce is None or cr is None) else cr - ce
        h = block["fixture_step"]
        print(
            f"| {block['shape']} | {block['x_resp']:g} | {block['ref']['x']:.4e} "
            f"| {fmt(ce)} | {fmt(ce_err, '.1e')} | {fmt(cr)} | {fmt(cr_err, '.1e')} "
            f"| {fmt(rc)} | {fmt(rc_err, '.1e')} | {fmt(ro)} | {fmt(ro_err, '.1e')} "
            f"| {fmt(interaction)} | {h**4 * block['x_resp'] / 384.0:.1e} "
            f"| {h**4 * block['ref']['x'] / 384.0:.1e} |",
            flush=True,
        )


def scaling_table(blocks):
    """Table 4: the x-scaling of each term, per shape."""
    print(
        "**Table 4 -- the x-scaling of each term:** least-squares slope of log|term| against "
        "log(x = k tau) over the rungs each term has. 0 is x-independent; 1 is linear in x. "
        "`rows` is how many rungs entered each fit. A slope is only as meaningful as the term "
        "is monotone -- read it beside Table 3's individual values, not instead of them."
    )
    print()
    print("| shape | term | rows | slope | smallest | largest |")
    print("|---|---|---|---|---|---|")
    for shape_name in RUNS:
        rows = [b for b in blocks if b["shape"] == shape_name]
        for label, hi, lo in TERMS:
            pairs = [
                (b["ref"]["x"], term(b, hi, lo)[0], b["x_resp"])
                for b in rows
                if term(b, hi, lo)[0] not in (None, 0.0)
            ]
            if not pairs:
                print(
                    f"| {shape_name} | {label} | 0 | {MISSING} | {MISSING} | {MISSING} |",
                    flush=True,
                )
                continue
            lo_p = min(pairs, key=lambda p: abs(p[1]))
            hi_p = max(pairs, key=lambda p: abs(p[1]))
            print(
                f"| {shape_name} | {label} | {len(pairs)} "
                f"| {slope([p[0] for p in pairs], [p[1] for p in pairs]):+.2f} "
                f"| {lo_p[1]:+.2e} (x_resp {lo_p[2]:g}) "
                f"| {hi_p[1]:+.2e} (x_resp {hi_p[2]:g}) |",
                flush=True,
            )


def gap_table(blocks):
    """Table 5: the gap in units of the fixture's own grid step, against production's."""
    print(
        "**Table 5 -- the gap this harness opens, against production's.** "
        f"`[12-handover-clamp-error-in-production]` records median **{PRODUCTION_GAP_MEDIAN:.1e}** "
        f"and max **{PRODUCTION_GAP_MAX:.1e}** in log(1+z), about one mean source-grid step. "
        "`mean source-grid step` is `max_clamp_gap_log1pz / HANDOVER_CLAMP_MAX_GRID_STEPS`, i.e. "
        "the step the clamp tolerance is measured in."
    )
    print()
    print(
        "| shape | x_resp | fixture WKB grid step | gap recorded, Tr | in fixture grid steps "
        "| mean source-grid step | in source-grid steps | / production median "
        "| / production max |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for block in blocks:
        cell = block["cells"].get(("realistic", "open")) or block["cells"].get(
            ("exact", "open")
        )
        if cell is None:
            continue
        h = cell["fixture_step"]
        gap = cell["recorded_gaps"]["Tr"]
        source_step = cell["max_clamp_gap"] / 1.5
        print(
            f"| {block['shape']} | {block['x_resp']:g} | {h:.4e} | {gap:.4e} "
            f"| {gap / h:.2f} | {source_step:.4e} | {gap / source_step:.2f} "
            f"| {gap / PRODUCTION_GAP_MEDIAN:.2f} | {gap / PRODUCTION_GAP_MAX:.2f} |",
            flush=True,
        )


def inertness_table(x_resp=980.0):
    """
    Table 6: `drop_first_WKB_sample` in the exact flavour. `ExactTkFunctions` ignores the sampled
    inputs, so the dropped fixture changes nothing -- which is why the exact half of the factorial
    uses `_GappedExactTkFunctions` instead. Measured rather than asserted.
    """
    print(
        "**Table 6 -- `drop_first_WKB_sample=True` is inert in the exact flavour.** "
        "`ExactTkFunctions` ignores the sampled inputs entirely, so production's own gap "
        "mechanism opens no gap there. This is why the exact half of Table 2 truncates "
        "`WKB_region` instead, and it is the reason the factorial's fourth cell is not simply "
        "the first two mechanisms applied together."
    )
    print()
    print(
        "| shape | x_resp | total, exact + Fixture as built | total, exact + dropped sample "
        "| relative difference | largest gap recorded |"
    )
    print("|---|---|---|---|---|---|")
    w = w_of_b(0.0)
    for base in SHAPES:
        shape = scaled_shape(base, x_resp)
        seed_fixtures(w, shape, x_resp)
        plain = Case(b=0.0, shape=shape, x_resp=x_resp, exact=True)
        dropped = Case(b=0.0, shape=shape, x_resp=x_resp, exact=True)
        dropped.Tq_inputs = dropped_inputs(w, shape, x_resp, shape.q)
        dropped.Tr_inputs = dropped_inputs(w, shape, x_resp, shape.r)
        a = float(plain.run(atol=REF_ATOL, rtol=REF_RTOL)["total"])
        out = dropped.run(atol=REF_ATOL, rtol=REF_RTOL)
        b = float(out["total"])
        gaps = [
            max(item["clamp_gaps_log1pz"].values(), default=0.0)
            for item in out["metadata"]["partition"]["subintervals"]
        ]
        print(
            f"| {base.name} | {x_resp:g} | {a:+.12e} | {b:+.12e} | {abs(a - b) / abs(a):.1e} "
            f"| {max(gaps, default=0.0):.1e} |",
            flush=True,
        )


def cost_table(blocks):
    """Table 7: the realistic fixtures' set-up and integral cost against the exact ones'."""
    print(
        "**Table 7 -- cost.** Set-up is everything before `evaluate_QuadSource_integral`: both "
        "`Fixture` objects, the `Case`, and (realistic only) the `OffsetBesselPhaseGk` phase "
        "spline. The integral column includes the two `bessel_phase` builds `Case.run` makes and "
        "the `analytic_rad` oracle the integral computes alongside `total`."
    )
    print()
    print(
        "| shape | x_resp | x = k tau | set-up, exact | set-up, realistic | ratio "
        "| integral, exact | integral, realistic | ratio |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for block in blocks:
        e = block["cells"][("exact", "closed")]
        r = block["cells"].get(("realistic", "closed"))
        if r is None:
            print(
                f"| {block['shape']} | {block['x_resp']:g} | {block['ref']['x']:.4e} "
                f"| {e['setup_time']:.3f} s | {NOT_REACHED} | {MISSING} "
                f"| {e['integral_time']:.2f} s | {NOT_REACHED} | {MISSING} |",
                flush=True,
            )
            continue
        print(
            f"| {block['shape']} | {block['x_resp']:g} | {block['ref']['x']:.4e} "
            f"| {e['setup_time']:.3f} s | {r['setup_time']:.3f} s "
            f"| {r['setup_time'] / e['setup_time']:.1f} | {e['integral_time']:.2f} s "
            f"| {r['integral_time']:.2f} s | {r['integral_time'] / e['integral_time']:.0f} |",
            flush=True,
        )


def cost_wall_table(blocks):
    """Table 8: how the realistic cost grows with x, and where that stops the ladder."""
    print(
        "**Table 8 -- the cost wall.** The Levin driver accepts each exact sub-interval in one "
        "or two regions at every x; against the realistic flavour's splined ingredients it "
        "bisects into hundreds or thousands. The growth rate of that cost with x is what sets "
        "how far this instrument reaches, and it is **not the same on every shape**. "
        "`decade factor` is the realistic integral time at this rung divided by the one below."
    )
    print()
    print(
        "| shape | x_resp | x = k tau | Levin regions, exact | Levin regions, realistic "
        "| integral, realistic | decade factor | status |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for shape_name in RUNS:
        previous = None
        for block in [b for b in blocks if b["shape"] == shape_name]:
            e = block["cells"][("exact", "closed")]
            r = block["cells"].get(("realistic", "closed"))
            if r is None:
                print(
                    f"| {shape_name} | {block['x_resp']:g} | {block['ref']['x']:.4e} "
                    f"| {e['Levin_regions']} | {MISSING} | {NOT_REACHED} | {MISSING} "
                    f"| **not reached** |",
                    flush=True,
                )
                continue
            factor = (
                MISSING if previous is None else f"{r['integral_time'] / previous:.1f}x"
            )
            print(
                f"| {shape_name} | {block['x_resp']:g} | {block['ref']['x']:.4e} "
                f"| {e['Levin_regions']} | {r['Levin_regions']} "
                f"| {r['integral_time']:.1f} s | {factor} | run |",
                flush=True,
            )
            previous = r["integral_time"]
    print()
    print(
        "| shape | realistic rungs | slope of log(realistic integral seconds) against log x "
        "| mean factor per decade of x_resp |"
    )
    print("|---|---|---|---|")
    for shape_name in RUNS:
        rows = [
            b
            for b in blocks
            if b["shape"] == shape_name and ("realistic", "closed") in b["cells"]
        ]
        times = [b["cells"][("realistic", "closed")]["integral_time"] for b in rows]
        xs = [b["ref"]["x"] for b in rows]
        decades = log(rows[-1]["x_resp"] / rows[0]["x_resp"], 10.0)
        mean_factor = (
            (times[-1] / times[0]) ** (1.0 / decades) if decades else float("nan")
        )
        print(
            f"| {shape_name} | {len(rows)} | {slope(xs, times):+.2f} | {mean_factor:.1f}x |",
            flush=True,
        )


# ================================================================================================


def ordered_cells():
    """
    Every (shape, x_resp, flavour, seam) to compute, cheapest-first: all the exact cells (about a
    second each) before any realistic one, so that the control table against Kohri & Terada is on
    disk within two minutes of the start, then the realistic cells ascending in x_resp so that the
    expensive rungs are last.
    """
    plan = []
    for shape_name, x_resps in RUNS.items():
        for x_resp in x_resps:
            for seam in ("closed", "open"):
                plan.append((shape_name, x_resp, "exact", seam))
    realistic = []
    for shape_name, x_resps in RUNS.items():
        for x_resp in x_resps:
            if x_resp > REALISTIC_CEILING[shape_name]:
                continue
            for seam in ("closed", "open"):
                realistic.append((shape_name, x_resp, "realistic", seam))
    realistic.sort(key=lambda item: (item[1], item[0], item[3]))
    return plan + realistic


def main():
    warnings.simplefilter("ignore")
    started = time.perf_counter()

    stored = load_checkpoint()
    provenance = git_provenance()
    known = {k: v for k, v in stored.items() if v.get("status") == "ok"}
    failures = {k: v["error"] for k, v in stored.items() if v.get("status") == "raised"}
    print(
        f"checkpoint {CHECKPOINT}: {len(known)} cells and {len(failures)} recorded failures "
        f"already on disk; script {SCRIPT_SHA256[:12]}, HEAD {provenance['git_head'][:12]}"
        f"{' (tree not clean)' if provenance['git_dirty'] else ''}",
        file=sys.stderr,
        flush=True,
    )

    for shape_name, x_resp, flavour, seam in ordered_cells():
        key = cell_key(shape_name, x_resp, flavour, seam)
        if key in known or key in failures:
            continue
        base = [s for s in SHAPES if s.name == shape_name][0]
        # a cell that raises is a result -- a representation that cannot be driven at this x is
        # exactly what this harness is for -- so it is recorded, durably, and the rest continues
        try:
            record = run_cell(base, x_resp, flavour, seam)
            note = f"{record['integral_time']:.1f} s"
        except Exception as exc:  # noqa: BLE001
            record = {
                "shape": shape_name,
                "x_resp": x_resp,
                "flavour": flavour,
                "seam": seam,
                "status": "raised",
                "error": f"{type(exc).__name__}: {exc}",
            }
            note = f"RAISED {record['error']}"
        record["script_sha256"] = SCRIPT_SHA256
        record.update(provenance)
        append_checkpoint(record)
        if record["status"] == "ok":
            known[key] = record
        else:
            failures[key] = record["error"]
        print(
            f"[{time.perf_counter() - started:7.0f} s] {shape_name} x_resp={x_resp:g} "
            f"{flavour}/{seam}: {note}",
            file=sys.stderr,
            flush=True,
        )

    blocks = []
    for shape_name, x_resps in RUNS.items():
        for x_resp in x_resps:
            cells = {}
            block_failures = {}
            for flavour, seam in CELLS:
                key = cell_key(shape_name, x_resp, flavour, seam)
                if key in known:
                    cells[(flavour, seam)] = known[key]
                elif key in failures:
                    block_failures[(flavour, seam)] = failures[key]
            if ("exact", "closed") not in cells:
                raise RuntimeError(
                    f"the control cell is missing at {shape_name} x_resp={x_resp:g}"
                )
            ref = cells[("exact", "closed")]["ref"]
            for cell in cells.values():
                assert abs(cell["ref"]["x"] - ref["x"]) <= 1e-14 * ref["x"], (
                    cell["flavour"],
                    cell["seam"],
                    cell["ref"]["x"],
                    ref["x"],
                )
            blocks.append(
                {
                    "shape": shape_name,
                    "x_resp": x_resp,
                    "ref": ref,
                    "cells": cells,
                    "failures": block_failures,
                    "fixture_step": cells[("exact", "closed")]["fixture_step"],
                }
            )

    print("<!-- generated by docs/handover/realistic_large_x.py -->")
    print()
    for table in (
        control_table,
        factorial_table,
        attribution_table,
        scaling_table,
        gap_table,
    ):
        table(blocks)
        print()
    inertness_table()
    print()
    cost_table(blocks)
    print()
    cost_wall_table(blocks)
    print()
    print(
        f"Cells: {len(known)} computed or read from the checkpoint, "
        f"{sum(len(b['failures']) for b in blocks)} raised, "
        f"{sum(1 for b in blocks for c in CELLS if c not in b['cells'] and c not in b['failures'])} "
        f"not reached. This invocation: {time.perf_counter() - started:.0f} s."
    )


if __name__ == "__main__":
    main()
