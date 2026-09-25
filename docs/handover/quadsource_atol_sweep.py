r"""Does ``DEFAULT_QUADRATURE_ATOL = 1e-32`` cost two hundred times what it buys?

Run from the repository root::

    PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py --prepare
    PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py \
        --cpus 6 --register quadsource-atol-sweep \
        --purpose "..." --campaign handover --prompt -
    PYTHONPATH=. ./venv/bin/python docs/handover/quadsource_atol_sweep.py --report

**The question, and why the existing sweep does not answer it.**
``prompts/tolerance-convergence`` prompt 06 already swept this constant --
[`docs/tolerance-convergence/QUADSOURCE-READONLY.md`](../tolerance-convergence/QUADSOURCE-READONLY.md)
§4 -- and found ``atol`` **inert over twenty-eight decades**, with ``rtol`` the binding parameter.
That measurement is not wrong and is not superseded. It measured **accuracy**, on the eighteen
offline cases of ``ComputeTargets/tests/test_quadsource_integral.py``, and on those cases the
verdict holds.

It could not see the thing this script measures, for two reasons. Its instrument has **no
``(k, q, r)`` triangle**: the fixture is a single constant-$w$ configuration, so the squeezed
geometry $r \approx k \gg q$ that the production work list is a quarter full of does not occur in
it. And its statistic is the residual, not the **cost**: ``atol`` that changes no digit of
``total`` can still change the region count by four orders, because ``_adaptive_levin`` accepts a
region on ``abserr < local_atol or relerr < rtol`` and ``local_atol`` is ``atol`` divided by the
region's share of the interval. On production's squeezed triples that denominator reaches 1.5e6,
so ``local_atol`` is around 1e-38, the first branch is dead, and every region must clear
``rtol = 1e-8`` against a phase the representation cannot deliver to eight digits. The bisection
then runs to ``DEFAULT_LEVIN_MAX_DEPTH = 20`` -- which
``QUADSOURCE-READONLY.md`` §0 finding 4 records as *never chosen* -- and
``levin_quadrature.py:2329`` says in its own words what that costs: *"the cost can be two orders
of magnitude higher than necessary"*.

Measured on the A3 baseline store on 2026-09-23, over its 7600 stored rows: rows below depth 20
cost a mean of **0.78 s**, rows at depth 20 cost a mean of **171 s**, and **74%** of the latter
are stored with ``total_converged = 0``. See
``prompts/handover/IMPLEMENTATION_STATE.md`` ``[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]``.

**What this script does.** It runs main.py's own pipeline, once per tolerance pair, over **nine
fixed production work items** chosen from that measurement -- four severe (depth 20, unconverged,
1.5e3 s each), three moderate, two cheap controls -- and reads the stored instrumentation back.
Nothing is reimplemented: the integrand, the partition, the Levin driver and the storage are
main.py's, reached through the same
``docs/gktk-remedial/scoped_pipeline_run.py`` machinery the A3 baseline itself ran under, whose
``RegisteredStream``/``StageTracker``/``terminal_state`` this file imports rather than copies.

**Four textual substitutions in main.py**, each exact and each required to match exactly once:

1. + 2. the two wavenumber-grid literals -> ``SWEEP_K_SAMPLE``, the **production** eight-point
   grid, so that the triangle filter keeps the same 64 triples and every case below is the
   production work item and not a lookalike;
3. + 4. ``tol=DEFAULT_QUADRATURE_ATOL`` and ``tol=DEFAULT_QUADRATURE_RTOL`` at ``main.py:3578-3579``
   -> the sweep's values. These are the **only** two sites that read those constants, which is
   what makes a one-line substitution safe here: every downstream lookup takes its tolerance from
   the objects those two calls return, so the work item and its lookup cannot disagree. (``CLAUDE.md``
   -- a site left on another constant does not raise, it silently recomputes.)
5. the work-list batching line -> the same line with ``SWEEP_SELECT_CASES`` in front of it, which
   filters the 9280 triangle-closing items down to the nine and **raises unless exactly nine
   survive**.

**The tolerance is part of the datastore key**, so each pair writes its own rows and no pair can
read another's. The production pair ``(1e-32, 1e-8)`` is therefore a free lookup of the rows the
baseline run already computed, and it is included precisely so that the sweep's own zero point is
the measured baseline and not a re-run of it.

**It runs against a copy.** ``--prepare`` copies the four A3 baseline shards to
``var/datastores/handover-atol-sweep-*``; the baseline store is opened by nothing here. That store
is the comparator for the hand-over change (campaign README §7 **E1**) and is left exactly as the
baseline run left it.
"""

import argparse
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = REPO_ROOT / "main.py"
SCOPED_DRIVER = REPO_ROOT / "docs" / "gktk-remedial" / "scoped_pipeline_run.py"

BASELINE_STORE = (
    REPO_ROOT / "var" / "datastores" / "handover-A3-baseline-lambdacdm.sqlite"
)
SWEEP_STORE = REPO_ROOT / "var" / "datastores" / "handover-atol-sweep.sqlite"
SHARDS = 4

# main.py's production wavenumber grid, verbatim from `main.py`'s own literals. The A3 baseline
# ran on exactly this, so the 64 triangle-closing triples below are its triples.
K_MIN, K_MAX, K_COUNT = 1e5, 3e8, 8

# The nine work items, as (k, q, r, z_response) in /Mpc and z. Chosen on 2026-09-23 from the
# baseline store's own `compute_time` column; the comment on each line is what that row cost at
# the production pair, its `WKB_Levin_max_depth`, and whether it converged. Four severe, three
# moderate, two controls. Matching is by value against the regenerated grid, not by store serial:
# `wavenumber_exit_time` carries separate source and response rows for the same k, so a serial is
# ambiguous here and a value is not.
CASES = (
    # --- severe: depth 20, unconverged, and the geometry is r == k >> q -----------------------
    (
        30454960.4,
        313856.8472,
        30454960.4,
        75.99254166,
    ),  # 1629.7 s, depth 20, unconverged
    (
        3091682.043,
        313856.8472,
        3091682.043,
        693.4704509,
    ),  # 1612.0 s, depth 20, unconverged
    (
        3091682.043,
        985061.2054,
        3091682.043,
        693.4704509,
    ),  # 1592.9 s, depth 20, unconverged
    (
        9703455.785,
        9703455.785,
        9703455.785,
        43.7226748,
    ),  # 1527.0 s, depth 20, unconverged
    # --- moderate: depth 20, two of the three converged ---------------------------------------
    (3091682.043, 100000.0, 3091682.043, 1589.001589),  # 294.8 s, depth 20, unconverged
    (30454960.4, 985061.2054, 30454960.4, 914.2397166),  # 291.6 s, depth 20, converged
    (
        9703455.785,
        3091682.043,
        9703455.785,
        1589.001589,
    ),  # 288.2 s, depth 20, converged
    # --- controls: the regime the sweep must not damage ----------------------------------------
    (30454960.4, 300000000.0, 300000000.0, 4800.128091),  # 0.3 s, depth 1, converged
    (30454960.4, 30454960.4, 30454960.4, 1205.291816),  # 0.8 s, depth 13, converged
)

# (atol, rtol). Ordered **loosest atol first**, so that the cheap end is measured before any
# wall-clock is committed to the expensive end and a collapse is visible in the first two rows.
# 1e-25 is the pre-`5255ac0` value; 1e-32 is production and is a lookup of the baseline's own rows.
# The last two hold atol at production and move rtol instead, which is the other lever and the one
# QUADSOURCE-READONLY.md §5 found binding on its fixture.
TOLERANCE_GRID = (
    (1e-22, 1e-8),
    (1e-25, 1e-8),
    (1e-28, 1e-8),
    (1e-30, 1e-8),
    (1e-32, 1e-6),
    (1e-32, 1e-7),
    (1e-32, 1e-8),
)

# --- phase 2, added 2026-09-23 on the measurement above -------------------------------------------
# The nine cases of phase 1 all sit at z_response >= 43.7, and its speed-ups must not be
# extrapolated into the 1024 items at z_response <= 6.32 that no run has ever attempted: eta_R
# reaches 13,728 at z = 0.1 against a largest *measured* 4936 at z = 8.33, and that block is where
# most of the remaining cost is. These seven re-use triples phase 1 already showed to be severe --
# so they are known to close a triangle and known to have every ingredient in the store -- at seven
# low z instead.
LOW_Z_CASES = (
    # z literals are the grid's own `repr`, not rounded: the selector matches at rel_tol=1e-9 and
    # a seven-figure 0.9125507 is 2.9e-08 away from the grid's 0.9125506737478801, which the guard
    # in make_case_selector() caught on the first attempt at this phase.
    (9703455.784651041, 985061.2054411147, 9703455.784651041, 0.1),
    (9703455.784651041, 3091682.0425413917, 9703455.784651041, 0.9125506737478801),
    (30454960.396664713, 313856.84721559234, 30454960.396664713, 2.0909967671302003),
    (3091682.0425413917, 100000.0, 3091682.0425413917, 6.316577898526916),
    (9703455.784651041, 9703455.784651041, 9703455.784651041, 0.3020845368018508),
    (30454960.396664713, 300000000.0000001, 300000000.0000001, 0.1),
    (30454960.396664713, 30454960.396664713, 30454960.396664713, 0.1),
)

# **No `rtol = 1e-8` reference here, deliberately.** At these eta_R a production-pair cell is a
# 1e+04 s item -- that is the whole finding -- so the reference is unaffordable and the test has to
# be self-convergence of the three loose rungs against each other, which is what
# `QUADSOURCE-READONLY.md` 8.1 does on its own axis. Loosest first, so that if 1e-7 proves
# unaffordable at z = 0.1 the two cheaper rungs are already in hand.
LOW_Z_TOLERANCE_GRID = (
    (1e-32, 1e-5),
    (1e-32, 1e-6),
    (1e-32, 1e-7),
)

# --- phase 3, added 2026-09-23 ---------------------------------------------------------------------
# Phase 2's three rungs agree to ~1e-5 at low z and I read that as a representation floor. c5 of
# phase 1 is the standing reason that reading is not safe: there, rtol 1e-5 and 1e-6 agreed with
# *each other* and were both 1.1e-02 wrong, and only 1e-7 found the missing contribution. Adjacent
# loose rungs agreeing is not convergence. There is also no oracle available to settle it --
# `domenech.py` and `kohri_terada.py` are both constant-w, and production is LambdaCDM with the
# Saikawa-Shirai QCD equation of state -- so self-convergence is the only instrument there is, and
# the only way to use it is to go one rung tighter (user, 2026-09-23).
#
# So: the three cheapest phase-2 cases at 1e-8, plus the free control. Cheapest, not worst, because
# phase 1's 1e-7 -> 1e-8 ratio was x6.8 and the worst phase-2 cell already costs 7751 s at 1e-7; if
# the ratio is worse down here, a run built from the severe cases would not finish. Their 1e-5,
# 1e-6 and 1e-7 values are already stored, so the report shows all four rungs.
LOW_Z_CHECK_CASES = (
    (30454960.396664713, 300000000.0000001, 300000000.0000001, 0.1),  # 0.3 s at 1e-7
    (30454960.396664713, 30454960.396664713, 30454960.396664713, 0.1),  # 397.7 s
    (
        9703455.784651041,
        9703455.784651041,
        9703455.784651041,
        0.3020845368018508,
    ),  # 878.3 s
    (3091682.0425413917, 100000.0, 3091682.0425413917, 6.316577898526916),  # 1321.6 s
)

LOW_Z_CHECK_TOLERANCE_GRID = ((1e-32, 1e-8),)

# The band where a numeric branch of G_k exists at all. Every GkNumericIntegration object for a
# given k stops at the same z_min, about 6.3 e-folds inside horizon entry (measured: z_exit/z_min
# is 479 to 612 across the eight modes), so below that response redshift there is no numeric data
# and GkSourcePolicyData classifies the row `WKB`. Only `mixed` rows carry a crossover_z, i.e. a
# switch of representation along the z_source axis, and only `numeric` and `mixed` rows have a
# numeric G for `_create_functions` to spline -- which is where `[03-numeric-g-consumer-spline-...]`
# (D1) lives. Five mixed and four numeric, severe geometry (r == k >> q), all sub-5 s rows.
SEAM_CASES = (
    (985061.2054411147, 313856.84721559234, 985061.2054411147, 4817913679.213864),
    (3091682.0425413917, 985061.2054411147, 3091682.0425413917, 13426973761.876026),
    (3091682.0425413917, 313856.84721559234, 3091682.0425413917, 4817913679.213864),
    (9703455.784651041, 3091682.0425413917, 9703455.784651041, 13426973761.876026),
    (313856.84721559234, 100000.0, 313856.84721559234, 527961220.9836534),
    (985061.2054411147, 313856.84721559234, 985061.2054411147, 13426973761.876026),
    (313856.84721559234, 100000.0, 313856.84721559234, 4817913679.213864),
    (9703455.784651041, 3091682.0425413917, 9703455.784651041, 145631461410.94043),
    (3091682.0425413917, 985061.2054411147, 3091682.0425413917, 48208843442.545044),
)

SEAM_TOLERANCE_GRID = (
    (1e-32, 1e-6),
    (1e-32, 1e-7),
    (1e-32, 1e-8),
)

PHASES = {
    "main": (CASES, TOLERANCE_GRID),
    "low-z": (LOW_Z_CASES, LOW_Z_TOLERANCE_GRID),
    "low-z-check": (LOW_Z_CHECK_CASES, LOW_Z_CHECK_TOLERANCE_GRID),
    "seam": (SEAM_CASES, SEAM_TOLERANCE_GRID),
}
ALL_CASES = CASES + LOW_Z_CASES + SEAM_CASES

K_GRID_LITERALS = (
    "np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_SOURCE_K_VALUES)",
    "np.logspace(np.log10(1e5), np.log10(3e8), NUMBER_RESPONSE_K_VALUES)",
)
K_GRID_NAME = "SWEEP_K_SAMPLE"

ATOL_SITE = ("tol=DEFAULT_QUADRATURE_ATOL", "tol=SWEEP_QUAD_ATOL")
RTOL_SITE = ("tol=DEFAULT_QUADRATURE_RTOL", "tol=SWEEP_QUAD_RTOL")

BATCH_LINE = (
    '        qsi_work_batches = list(grouper(qsi_work_items, n=750, incomplete="fill"))'
)
BATCH_REPLACEMENT = (
    "        qsi_work_items = SWEEP_SELECT_CASES(qsi_work_items)\n"
    '        qsi_work_batches = list(grouper(qsi_work_items, n=len(qsi_work_items), incomplete="fill"))'
)


def _load_scoped_driver():
    """Import `docs/gktk-remedial/scoped_pipeline_run.py` by path.

    Its directory name carries a hyphen, so it is not importable as a package. Its
    `RegisteredStream`, `StageTracker` and `terminal_state` are the run-registry plumbing the A3
    baseline ran under and are imported rather than copied: one copy of that machinery, and a
    sweep that goes stale the same way a pipeline run does.
    """
    spec = importlib.util.spec_from_file_location(
        "_scoped_pipeline_run", str(SCOPED_DRIVER)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def k_sample():
    return np.logspace(np.log10(K_MIN), np.log10(K_MAX), K_COUNT).tolist()


def make_case_selector(cases):
    """Return the `SWEEP_SELECT_CASES` main.py's work list is filtered through.

    The work list is a list of `(z_response, k, q, r)` where the wavenumbers are
    `wavenumber_exit_time` and `z_response` is a `redshift`. Matching is on value with
    `rel_tol=1e-9`: both sides are regenerated by the same deterministic construction from the
    same literals, so the floats are identical and the tolerance is there to say so rather than to
    absorb a difference. Comparing `z` directly is safe here -- these are two copies of one grid
    value, not a `log(1+z)` round trip (`CLAUDE.md`, "Redshift arithmetic").
    """

    def close(a, b):
        return math.isclose(a, b, rel_tol=1e-9)

    def select(work_items):
        kept = []
        for target in cases:
            tk, tq, tr, tz = target
            matches = [
                item
                for item in work_items
                if close(item[1].k.k_inv_Mpc, tk)
                and close(item[2].k.k_inv_Mpc, tq)
                and close(item[3].k.k_inv_Mpc, tr)
                and close(item[0].z, tz)
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"quadsource_atol_sweep: case (k={tk:.10g}, q={tq:.10g}, r={tr:.10g}, "
                    f"z_response={tz:.10g}) matched {len(matches)} work items, expected exactly "
                    f"1. The work list holds {len(work_items)} items. A case that does not match "
                    "is a case whose geometry this tree no longer builds, and the sweep must not "
                    "quietly measure a different one."
                )
            kept.extend(matches)
        print(
            f"** quadsource_atol_sweep: selected {len(kept)} of {len(work_items)} work items"
        )
        for z, k, q, r in kept:
            print(
                f"     k={k.k.k_inv_Mpc:11.6g} q={q.k.k_inv_Mpc:11.6g} "
                f"r={r.k.k_inv_Mpc:11.6g} z_response={z.z:.8g}"
            )
        return kept

    return select


def substitute_main_py(filter_cases=True):
    """main.py's source with the substitutions applied, each verified to match exactly once.

    `filter_cases=False` omits the work-list filter and applies only the four a full build needs:
    the two grid literals and the two tolerance sites. A build must run main.py's own work list,
    unfiltered, or it is not a build.
    """
    source = MAIN_PY.read_text()
    edits = [(literal, K_GRID_NAME) for literal in K_GRID_LITERALS] + [
        ATOL_SITE,
        RTOL_SITE,
    ]
    if filter_cases:
        edits.append((BATCH_LINE, BATCH_REPLACEMENT))
    for literal, replacement in edits:
        count = source.count(literal)
        if count != 1:
            raise RuntimeError(
                f"quadsource_atol_sweep: expected exactly 1 occurrence of {literal!r} in "
                f"main.py, found {count}. main.py has moved under this script; fix the literal "
                "rather than loosening the match."
            )
        source = source.replace(literal, replacement)
        print(
            f"** quadsource_atol_sweep: replaced {count} occurrence of {literal!r} in main.py"
        )
    return source


def run_child(args):
    """One tolerance pair: exec main.py with the five substitutions and let it store the rows."""
    # before Ray, before main.py: this child is about to hand `--database SWEEP_STORE` to a
    # ShardedPool, and a primary whose `shards` table names another store's files would send
    # every write there instead. Checked here as well as in prepare() because this is the process
    # that does the writing.
    assert_store_is_self_consistent(SWEEP_STORE)

    import ray

    _real_ray_init = ray.init

    def _local_ray_init(*_a, **_kw):
        print(
            f"** quadsource_atol_sweep: bootstrapping a local Ray instance "
            f"(num_cpus={args.cpus}, include_dashboard=False)"
        )
        return _real_ray_init(
            num_cpus=args.cpus, include_dashboard=False, ignore_reinit_error=True
        )

    ray.init = _local_ray_init

    import config.model_list as model_list_module

    _real_build = model_list_module.build_model_list

    def _filtered_build(pool, units):
        models = _real_build(pool, units)
        kept = [m for m in models if m["label"] == "LambdaCDM"]
        if len(kept) == 0:
            raise RuntimeError(
                "quadsource_atol_sweep: no LambdaCDM model in config.model_list"
            )
        return kept

    model_list_module.build_model_list = _filtered_build

    source = substitute_main_py()

    main_args = [
        "--database",
        str(SWEEP_STORE),
        "--job-name",
        f"handover-atol-sweep-a{args.atol:g}-r{args.rtol:g}",
        "--shards",
        str(SHARDS),
        "--zend",
        "0.1",
        "--source-samples-log10z",
        "100",
        "--no-prune-unvalidated",
    ]
    sys.argv = ["main.py"] + main_args
    print(
        f"** quadsource_atol_sweep: phase={args.phase} atol={args.atol:g} rtol={args.rtol:g}"
    )
    print(f"** quadsource_atol_sweep: main.py argv = {sys.argv[1:]}")

    namespace = {
        "__name__": "__main__",
        "__file__": str(MAIN_PY),
        K_GRID_NAME: k_sample(),
        "SWEEP_QUAD_ATOL": args.atol,
        "SWEEP_QUAD_RTOL": args.rtol,
        "SWEEP_SELECT_CASES": make_case_selector(PHASES[args.phase][0]),
    }
    exec(compile(source, str(MAIN_PY), "exec"), namespace)


def run_build(args):
    """A full 9280-item build at one tolerance pair, into a store that does not yet exist.

    The sweep's counterpart, not a sweep: the same four substitutions for the grid and the two
    tolerance sites, **no** work-list filter, and a `--database` of its own. It lives here rather
    than in `docs/gktk-remedial/scoped_pipeline_run.py` for one reason -- that driver cannot move
    the tolerance, and the tolerance is the whole point of rebuilding.

    The target must not already exist. A build that resumed into an existing store would silently
    inherit whatever was in it, which for a comparator is the one thing that must not happen. To
    *continue* an interrupted build, pass `--resume`, which is a separate and deliberate act and
    which re-checks that the store names its own shards first.
    """
    import ray

    database = Path(args.database).resolve()
    present = [pth for pth in [database] + shard_paths(database) if pth.exists()]
    if present and not args.resume:
        raise RuntimeError(
            f"quadsource_atol_sweep --build: {', '.join(pth.name for pth in present)} already "
            "exist(s). Refusing to build into an existing store: a comparator that silently "
            "inherited earlier rows is one nobody can interpret. Pass --resume to continue an "
            "interrupted build at the SAME tolerance pair, or choose a path that does not exist."
        )
    if present:
        assert_store_is_self_consistent(database)
        print(f"** quadsource_atol_sweep --build: resuming into {database.name}")
    if not database.parent.is_dir():
        raise RuntimeError(
            f"quadsource_atol_sweep --build: {database.parent} is not a directory"
        )

    scoped = _load_scoped_driver()
    run = None
    if args.register:
        import RunRegistry

        run = RunRegistry.begin(
            campaign=args.campaign,
            prompt=args.prompt,
            slug=args.register,
            purpose=args.purpose,
            script=__file__,
            results=str(database),
            scope=(
                f"full build: main.py's own 9280-item QuadSourceIntegral work list at "
                f"atol={args.atol:g} rtol={args.rtol:g}, {args.cpus} cpus, LambdaCDM, 8 "
                f"log-spaced k over {K_MIN:g}-{K_MAX:g} /Mpc, into {database.name}"
            ),
            heartbeat_means=scoped.HEARTBEAT_MEANS.format(
                interval=scoped.BEAT_MIN_INTERVAL
            ),
        )

        def terminal(signum, _frame):
            run.finish("killed", exit_code=128 + signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, terminal)

        sys.stdout = scoped.RegisteredStream(
            sys.stdout,
            run,
            scoped.StageTracker(),
            copy_to=run.stdout_path,
            interval=scoped.BEAT_MIN_INTERVAL,
        )
        sys.stderr = scoped.RegisteredStream(
            sys.stderr, run, scoped.StageTracker(), copy_to=run.stderr_path, beats=False
        )
        print(
            f"** quadsource_atol_sweep --build: registered as {run.id}\n"
            f"**   manifest {run.manifest_path}\n"
            f"**   list it with: PYTHONPATH=. ./venv/bin/python -m RunRegistry list"
        )

    _real_ray_init = ray.init

    def _local_ray_init(*_a, **_kw):
        print(
            f"** quadsource_atol_sweep --build: bootstrapping a local Ray instance "
            f"(num_cpus={args.cpus}, include_dashboard=False)"
        )
        return _real_ray_init(
            num_cpus=args.cpus, include_dashboard=False, ignore_reinit_error=True
        )

    ray.init = _local_ray_init

    import config.model_list as model_list_module

    _real_build = model_list_module.build_model_list

    def _filtered_build(pool, units):
        kept = [m for m in _real_build(pool, units) if m["label"] == "LambdaCDM"]
        if len(kept) == 0:
            raise RuntimeError("quadsource_atol_sweep --build: no LambdaCDM model")
        return kept

    model_list_module.build_model_list = _filtered_build

    source = substitute_main_py(filter_cases=False)
    sys.argv = [
        "main.py",
        "--database",
        str(database),
        "--job-name",
        args.job_name,
        "--shards",
        str(SHARDS),
        "--zend",
        "0.1",
        "--source-samples-log10z",
        "100",
        "--no-prune-unvalidated",
    ]
    print(f"** quadsource_atol_sweep --build: atol={args.atol:g} rtol={args.rtol:g}")
    print(f"** quadsource_atol_sweep --build: main.py argv = {sys.argv[1:]}")

    namespace = {
        "__name__": "__main__",
        "__file__": str(MAIN_PY),
        K_GRID_NAME: k_sample(),
        "SWEEP_QUAD_ATOL": args.atol,
        "SWEEP_QUAD_RTOL": args.rtol,
    }
    if run is None:
        exec(compile(source, str(MAIN_PY), "exec"), namespace)
        return
    try:
        exec(compile(source, str(MAIN_PY), "exec"), namespace)
    except SystemExit as exc:
        run.finish(
            scoped.terminal_state(exc.code), exit_code=exc.code, fingerprint=True
        )
        raise
    except BaseException:
        run.finish("failed", exit_code=1, fingerprint=True)
        raise
    run.finish("done", exit_code=0, fingerprint=True)
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, scoped.RegisteredStream):
            stream.close_copy()


def shard_paths(primary):
    return [
        primary.with_name(f"{primary.stem}-shard{i:04d}{primary.suffix}")
        for i in range(SHARDS)
    ]


def assert_store_is_self_consistent(primary):
    """Refuse to go on unless the primary's `shards` table names *its own* shard files.

    This is not a formality. `ShardedPool` does not derive the shard filenames from the
    `--database` path: it reads them, as **absolute paths**, out of a `shards` table inside the
    primary file (`Datastore/SQL/ShardedPool.py:_read_shard_data`, written by `_write_shard_data`
    at `:280`). The sidecar `.manifest.json` is a human note and is read by nothing. So copying a
    primary file copies the *previous* store's shard paths with it, and a pool opened on the copy
    writes to the original -- which is exactly what happened on 2026-09-23, when the first run of
    this sweep put 54 rows into the A3 baseline store it was written to leave alone. The rows were
    at six non-production tolerance pairs and the production population was untouched, so nothing
    was lost; but the guarantee in this module's docstring was false for one run, and a guarantee
    that is only true when `prepare()` is correct is not a guarantee. Hence this check, called
    from both `prepare()` and `run_child()`.
    """
    import sqlite3

    expected = {str(p.resolve()) for p in shard_paths(primary)}
    conn = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
    try:
        found = {row[1] for row in conn.execute("select serial, filename from shards")}
    finally:
        conn.close()
    if found != expected:
        raise RuntimeError(
            f"quadsource_atol_sweep: the `shards` table inside {primary.name} does not name its "
            f"own shard files, so a pool opened on it would read and write somewhere else.\n"
            f"  expected: {sorted(expected)}\n"
            f"  found:    {sorted(found)}\n"
            "Re-run --prepare --force. Do NOT run the sweep against this store."
        )


def prepare(force=False):
    """Copy the baseline store to the sweep store, and re-point the copy at its own shards."""
    import sqlite3

    src_shards = shard_paths(BASELINE_STORE)
    missing = [p for p in src_shards if not p.exists()]
    if missing:
        raise RuntimeError(
            f"quadsource_atol_sweep: baseline shard(s) not found: "
            f"{', '.join(str(p) for p in missing)}"
        )
    targets = shard_paths(SWEEP_STORE)
    existing = [p for p in targets + [SWEEP_STORE] if p.exists()]
    if existing and not force:
        raise RuntimeError(
            f"quadsource_atol_sweep: sweep store already exists "
            f"({', '.join(p.name for p in existing)}); pass --force to replace it, or --report "
            "to read what is already there. Refusing to overwrite a store that may hold a "
            "sweep someone is still reading."
        )
    for src, dst in zip(src_shards, targets):
        print(f"** copying {src.name} -> {dst.name} ({src.stat().st_size/1e6:.0f} MB)")
        shutil.copy2(src, dst)
    print(f"** copying {BASELINE_STORE.name} -> {SWEEP_STORE.name} (primary)")
    shutil.copy2(BASELINE_STORE, SWEEP_STORE)

    # and the reason this function exists at all: the copied primary still names the *baseline's*
    # shard files, because that is where ShardedPool keeps them. Re-point it at its own.
    conn = sqlite3.connect(SWEEP_STORE)
    try:
        with conn:
            for serial, path in enumerate(targets):
                conn.execute(
                    "update shards set filename = ? where serial = ?",
                    (str(path.resolve()), serial),
                )
    finally:
        conn.close()
    assert_store_is_self_consistent(SWEEP_STORE)
    print("** re-pointed the copied primary at its own shards:")
    for path in targets:
        print(f"     {path}")

    # the sidecar manifest is a human note that ShardedPool never reads. Copying the baseline's
    # verbatim would leave a file in var/datastores/ whose `datastore` field names another store,
    # so write our own rather than copy theirs.
    manifest = SWEEP_STORE.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "name": SWEEP_STORE.stem,
                "purpose": (
                    "Working copy of the A3 baseline store for "
                    "docs/handover/quadsource_atol_sweep.py. Disposable: every row that is not "
                    "at the production tolerance pair belongs to a sweep and nothing else reads "
                    "it. The comparator is handover-A3-baseline-lambdacdm, not this."
                ),
                "datastore": str(SWEEP_STORE.relative_to(REPO_ROOT)),
                "copied_from": str(BASELINE_STORE.relative_to(REPO_ROOT)),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"** quadsource_atol_sweep: sweep store ready at {SWEEP_STORE}")


def report():
    """Read the sweep store back and print the table. Read-only; every shard opened mode=ro."""
    import sqlite3

    rows = []
    shard0 = SWEEP_STORE.with_name(f"{SWEEP_STORE.stem}-shard0000{SWEEP_STORE.suffix}")
    c0 = sqlite3.connect(f"file:{shard0}?mode=ro", uri=True)
    zmap = dict(c0.execute("select serial, z from redshift"))
    kmap = {
        s: k
        for s, k in c0.execute(
            "select e.serial, w.k_inv_Mpc from wavenumber_exit_time e "
            "join wavenumber w on w.serial = e.wavenumber_serial"
        )
    }
    # `tolerance` stores log10 of the value, not the value (schema: `log10_tol`)
    tolmap = {
        serial: 10.0**log10_tol
        for serial, log10_tol in c0.execute("select serial, log10_tol from tolerance")
    }
    for i in range(SHARDS):
        shard = SWEEP_STORE.with_name(
            f"{SWEEP_STORE.stem}-shard{i:04d}{SWEEP_STORE.suffix}"
        )
        c = sqlite3.connect(f"file:{shard}?mode=ro", uri=True)
        for row in c.execute(
            "select k_wavenumber_exit_serial, q_wavenumber_exit_serial, "
            "r_wavenumber_exit_serial, z_response_serial, atol_serial, rtol_serial, "
            'total, total_abserr, total_converged, compute_time, "WKB_Levin_max_depth", '
            '"WKB_Levin_num_regions", "WKB_Levin_evaluations" from QuadSourceIntegral'
        ):
            rows.append(row)

    def is_case(k, q, r, z):
        return any(
            math.isclose(kmap[k], c[0], rel_tol=1e-9)
            and math.isclose(kmap[q], c[1], rel_tol=1e-9)
            and math.isclose(kmap[r], c[2], rel_tol=1e-9)
            and math.isclose(zmap[z], c[3], rel_tol=1e-9)
            for c in ALL_CASES
        )

    kept = [r for r in rows if is_case(r[0], r[1], r[2], r[3])]
    print(f"\n{len(kept)} sweep rows over {len(ALL_CASES)} cases\n")
    hdr = (
        f"{'atol':>8} {'rtol':>7} | {'k':>10} {'q':>10} {'r':>10} {'z_resp':>10} | "
        f"{'total':>13} {'conv':>4} {'depth':>5} {'regions':>9} {'time (s)':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for row in sorted(
        kept, key=lambda t: (kmap[t[0]], kmap[t[1]], zmap[t[3]], -tolmap[t[4]])
    ):
        k, q, r, z, at, rt, tot, abserr, conv, ct, depth, nreg, nev = row
        print(
            f"{tolmap[at]:8.0e} {tolmap[rt]:7.0e} | {kmap[k]:10.4g} {kmap[q]:10.4g} "
            f"{kmap[r]:10.4g} {zmap[z]:10.5g} | {tot:13.6e} {int(bool(conv)):4d} "
            f"{depth if depth is not None else -1:5d} "
            f"{nreg if nreg is not None else -1:9d} {ct:10.2f}"
        )
    # per-tolerance totals, which is the statistic the question is about
    print("\nper-tolerance totals over the nine cases:")
    print(
        f"{'atol':>8} {'rtol':>7} {'n':>4} {'sum time (s)':>14} {'max time (s)':>14} "
        f"{'at depth 20':>12} {'unconverged':>12}"
    )
    by = {}
    for row in kept:
        by.setdefault((tolmap[row[4]], tolmap[row[5]]), []).append(row)
    for key in sorted(by, key=lambda t: (-t[0], -t[1])):
        g = by[key]
        print(
            f"{key[0]:8.0e} {key[1]:7.0e} {len(g):4d} {sum(x[9] for x in g):14.1f} "
            f"{max(x[9] for x in g):14.1f} {sum(1 for x in g if x[10] == 20):12d} "
            f"{sum(1 for x in g if not x[8]):12d}"
        )


def sweep(args):
    """Parent: register, then run one child subprocess per tolerance pair."""
    scoped = _load_scoped_driver()
    run = None
    if args.register:
        import RunRegistry

        run = RunRegistry.begin(
            campaign=args.campaign,
            prompt=args.prompt,
            slug=args.register,
            purpose=args.purpose,
            script=__file__,
            results=str(SWEEP_STORE),
            scope=(
                f"phase {args.phase}: {len(PHASES[args.phase][0])} production "
                f"QuadSourceIntegral work items x {len(PHASES[args.phase][1])} tolerance pairs, "
                f"{args.cpus} cpus, on a copy of the A3 baseline store; rtol "
                f"{min(t[1] for t in PHASES[args.phase][1]):g} to "
                f"{max(t[1] for t in PHASES[args.phase][1]):g} at atol "
                f"{min(t[0] for t in PHASES[args.phase][1]):g}"
            ),
            heartbeat_means=(
                "Refreshed by the parent after each tolerance pair's child process exits, and "
                "the `stage` names the pair just finished. A pair can legitimately take an hour "
                "at the tight end of the atol axis -- that is the measurement -- so this run "
                "goes stale while it is working and stale means go and look at the child's "
                "output in this directory. It does not mean dead."
            ),
        )

        def terminal(signum, _frame):
            run.finish("killed", exit_code=128 + signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, terminal)

        tracker = scoped.StageTracker()
        sys.stdout = scoped.RegisteredStream(
            sys.stdout, run, tracker, copy_to=run.stdout_path, beats=False
        )
        sys.stderr = scoped.RegisteredStream(
            sys.stderr, run, scoped.StageTracker(), copy_to=run.stderr_path, beats=False
        )
        print(
            f"** quadsource_atol_sweep: registered as {run.id}\n"
            f"**   manifest {run.manifest_path}\n"
            f"**   list it with: PYTHONPATH=. ./venv/bin/python -m RunRegistry list"
        )

    try:
        cases, grid = PHASES[args.phase]
        for index, (atol, rtol) in enumerate(grid, start=1):
            stage = (
                f"{args.phase} pair {index}/{len(grid)}: atol={atol:g} rtol={rtol:g}"
            )
            print(f"\n{'='*90}\n** {stage}\n{'='*90}", flush=True)
            if run is not None:
                run.heartbeat(stage=stage)
            started = time.time()
            child = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    "--atol",
                    repr(atol),
                    "--rtol",
                    repr(rtol),
                    "--cpus",
                    str(args.cpus),
                    "--phase",
                    args.phase,
                ],
                cwd=str(REPO_ROOT),
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            )
            elapsed = time.time() - started
            print(
                f"** {stage} finished in {elapsed:.0f} s with exit code "
                f"{child.returncode}",
                flush=True,
            )
            if child.returncode != 0:
                raise RuntimeError(
                    f"quadsource_atol_sweep: child for atol={atol:g} rtol={rtol:g} exited "
                    f"{child.returncode}; stopping rather than reporting a partial matrix as "
                    "a complete one"
                )
            if run is not None:
                run.heartbeat(stage=f"{stage} done in {elapsed:.0f} s")
    except SystemExit as exc:
        if run is not None:
            run.finish(
                scoped.terminal_state(exc.code), exit_code=exc.code, fingerprint=True
            )
        raise
    except BaseException:
        if run is not None:
            run.finish("failed", exit_code=1, fingerprint=True)
        raise
    if run is not None:
        run.finish("done", exit_code=0, fingerprint=True)
    report()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prepare", action="store_true", help="copy the baseline store"
    )
    parser.add_argument("--force", action="store_true", help="with --prepare: replace")
    parser.add_argument(
        "--report", action="store_true", help="read the sweep store back"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="full 9280-item build at one tolerance pair into --database (not a sweep)",
    )
    parser.add_argument("--database", type=str, default=None, help="with --build")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="with --build: continue into an existing store at the SAME tolerance pair",
    )
    parser.add_argument("--job-name", type=str, default="handover-A3-baseline-v2")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--atol", type=float, default=None, help="with --build: absolute tolerance"
    )
    parser.add_argument(
        "--rtol", type=float, default=None, help="with --build: relative tolerance"
    )
    parser.add_argument("--cpus", type=int, default=6)
    parser.add_argument(
        "--phase", choices=sorted(PHASES), default="main", help="which case set to run"
    )
    parser.add_argument("--register", type=str, default=None, metavar="SLUG")
    parser.add_argument("--purpose", type=str, default=None)
    parser.add_argument("--campaign", type=str, default="handover")
    parser.add_argument("--prompt", type=str, default="-")
    args = parser.parse_args()

    if args.prepare:
        prepare(force=args.force)
        return
    if args.report:
        report()
        return
    if args.build:
        if args.database is None:
            parser.error("--build needs --database")
        if args.atol is None or args.rtol is None:
            parser.error(
                "--build needs --atol and --rtol explicitly: the pair is the point"
            )
        if args.register and not args.purpose:
            parser.error("--register needs --purpose")
        run_build(args)
        return
    if args.child:
        if args.atol is None or args.rtol is None:
            parser.error("--child needs --atol and --rtol")
        run_child(args)
        return
    if args.register and not args.purpose:
        parser.error(
            "--register needs --purpose: a run nobody can identify from disk is the thing "
            "the registry exists to prevent"
        )
    sweep(args)


if __name__ == "__main__":
    main()
