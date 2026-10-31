"""levin-phase-floor prompt 06h, section 2 test 8: LiouvilleGreen/tests/test_3bessel_analytic.py over
20 seeded draws, at the parent (the dispatch commit's three_bessel_integrals.py, 06e's code) and at
06h's.

06e's harness of the same name (docs/levin-phase-floor/06e/calibration/scripts/lg_seeded_draws.py),
with what is swapped between the two modes changed: 06h changes LiouvilleGreen/three_bessel_integrals.py
and not the driver, so in mode "parent" the parent's module (prov.load_parent(), read with `git show`
and imported under another name) is put in sys.modules in place of
LiouvilleGreen.three_bessel_integrals *before* the test module imports quad_JJJ and quad_YJJ from it,
and in both modes the driver is the working tree's AdaptiveLevin, which 06h does not change. 06e's
DEBUG capture of the driver's summary line is dropped (the driver is the same in both modes), and the
evaluators are recorded, for the per-oracle table. Each unit is one (seed, mode, test). In a fresh
worker process it

- wraps AdaptiveLevin.levin_quadrature._adaptive_levin, the module attribute adaptive_levin_sincos()
  calls it through, so that every Levin call is recorded;
- wraps the test module's quad_JJJ and quad_YJJ, so that every three-Bessel result (value, abserr,
  converged) is recorded in order;
- seeds the global ``random`` instance with the seed, immediately before the test -- the test
  module draws its wavenumbers from ``random.uniform``, unseeded, and is not edited;
- runs the one test method through unittest, capturing what it prints.

The tests are every test of Test3BesselAnalytic that runs by default: test_JJJ, test_YJJ,
test_YJJ_log_singularity and test_abserr_bounds_truth (a fixed triple, run at every seed all the
same). test_YJJ_log_scaling is skipped by default and asserts nothing; it is not run.

  run     execute every unit (a process pool), writing one JSON line per unit to
          ../out/lg_seeded_draws.jsonl, with a provenance header line first
  report  read that file and write ../out/lg_seeded_draws_report.txt

The report lists every test outcome (a test that fails under 06h and passes at the parent is the
prompt's stop, section 7); the honesty table (dishonest, dishonest-and-converged, worst ratio) under
both, with test_YJJ_log_singularity's near-singular cases (eps <= 1e-4) reported separately;
converged counts and calls at depth_max under both; per oracle, value, abserr and the true error
under both; and every Levin call's regions, solves, depth and convergence.

No Ray and no datastore; nothing is written outside ../out/.
"""

import argparse
import contextlib
import hashlib
import io
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prov  # noqa: E402

REPO = prov.REPO
TEST_MODULE = REPO / "LiouvilleGreen" / "tests" / "test_3bessel_analytic.py"
INPUTS = (TEST_MODULE, prov.THREE_BESSEL, REPO / "LiouvilleGreen" / "bessel_phase.py")

SEEDS = list(range(20))
MODES = ("parent", "new")
# most expensive first, so the pool's tail is short
TESTS = ("test_YJJ_log_singularity", "test_JJJ", "test_YJJ", "test_abserr_bounds_truth")

RUN_OUT = prov.OUT / "lg_seeded_draws.jsonl"
REPORT_OUT = prov.OUT / "lg_seeded_draws_report.txt"


def _region_record(region):
    return [
        region.start,
        region.end,
        region.depth,
        region.type,
        region.abserr,
        region.relerr,
        region.phase_err,
        region.phase_limited,
        list(region.p_ratios) if region.p_ratios is not None else None,
        region.abserr_truncation,
    ]


def _call_record(index, args, kwargs, data):
    regions = [_region_record(r) for r in data["regions"]]
    digest = hashlib.sha256(
        json.dumps(regions, sort_keys=True, default=repr).encode()
    ).hexdigest()
    return {
        "index": index,
        "x_span": [float(args[0][0]), float(args[0][1])],
        "atol": kwargs.get("atol"),
        "rtol": kwargs.get("rtol"),
        "notify_label": kwargs.get("notify_label"),
        "value": data["value"],
        "abserr": data["abserr"],
        "converged": data["converged"],
        "evaluations": data["evaluations"],
        "num_regions": data["num_regions"],
        "num_phase_limited_regions": data["num_phase_limited_regions"],
        "max_depth": data["max_depth"],
        "regions_sha256": digest,
        "num_untrusted_regions": data.get("num_untrusted_regions"),
    }


def run_unit(unit):
    """One (seed, mode, test), in a fresh worker process (maxtasksperchild=1)."""
    seed, mode, test = unit
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    import random
    import unittest

    import AdaptiveLevin.levin_quadrature as lq
    import LiouvilleGreen

    if mode == "parent":
        parent, parent_sha = prov.load_parent()
        sys.modules["LiouvilleGreen.three_bessel_integrals"] = parent
        LiouvilleGreen.three_bessel_integrals = parent
    else:
        import LiouvilleGreen.three_bessel_integrals as parent  # noqa: F401

        parent_sha = None

    driver = lq._adaptive_levin
    calls = []

    def recording_adaptive_levin(*args, **kwargs):
        data = driver(*args, **kwargs)
        record = _call_record(len(calls), args, kwargs, data)
        record["depth_max"] = kwargs.get("depth_max", lq.DEFAULT_LEVIN_MAX_DEPTH)
        calls.append(record)
        return data

    lq._adaptive_levin = recording_adaptive_levin

    import LiouvilleGreen.tests.test_3bessel_analytic as analytic

    module_used = sys.modules["LiouvilleGreen.three_bessel_integrals"]
    assert analytic.quad_JJJ is module_used.quad_JJJ
    results = []

    def recording(evaluator, kind):
        def wrapped(*args, **kwargs):
            result = evaluator(*args, **kwargs)
            results.append(
                {
                    "kind": kind,
                    "orders": [float(args[3]), float(args[4]), float(args[5])],
                    "kqs": [float(args[6]), float(args[7]), float(args[8])],
                    "value": float(result.value),
                    "abserr": float(result.abserr),
                    "converged": bool(result.converged),
                }
            )
            return result

        return wrapped

    analytic.quad_JJJ = recording(analytic.quad_JJJ, "JJJ")
    analytic.quad_YJJ = recording(analytic.quad_YJJ, "YJJ")

    case = analytic.Test3BesselAnalytic(test)
    result = unittest.TestResult()
    stdout = io.StringIO()
    start = time.perf_counter()
    random.seed(seed)
    with contextlib.redirect_stdout(stdout):
        case.run(result)
    elapsed = time.perf_counter() - start

    if result.errors:
        outcome = "error"
        detail = result.errors[0][1]
    elif result.failures:
        outcome = "fail"
        detail = result.failures[0][1]
    elif result.skipped:
        outcome = "skipped"
        detail = result.skipped[0][1]
    else:
        outcome = "pass"
        detail = ""

    return {
        "seed": seed,
        "mode": mode,
        "test": test,
        "outcome": outcome,
        "detail": detail,
        "elapsed_s": elapsed,
        "three_bessel_module": module_used.__file__,
        "parent_module_sha256": parent_sha,
        "num_calls": len(calls),
        "calls": calls,
        "results": results,
        "stdout": stdout.getvalue(),
    }


def do_run(processes):
    units = [(seed, mode, test) for test in TESTS for seed in SEEDS for mode in MODES]
    header = prov.header(
        __file__, "section 2 test 8, the LiouvilleGreen seeded draws (run)", INPUTS
    )
    header.append(
        f"# units {len(units)}: seeds {SEEDS[0]}-{SEEDS[-1]} x modes {MODES} x tests {TESTS};"
        f" processes {processes}; parent {prov.PARENT_SHA}"
    )
    start = time.perf_counter()
    records = []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=processes, maxtasksperchild=1) as pool:
        for record in pool.imap_unordered(run_unit, units):
            records.append(record)
            print(
                f"[{len(records)}/{len(units)}] seed {record['seed']} {record['mode']} "
                f"{record['test']}: {record['outcome']} in {record['elapsed_s']:.1f} s",
                flush=True,
            )
    wall = time.perf_counter() - start
    records.sort(key=lambda r: (TESTS.index(r["test"]), r["seed"], r["mode"]))
    header.append(f"# wall time {wall:.1f} s; uptime at end: {prov.uptime()}")
    with open(RUN_OUT, "w") as handle:
        handle.write(json.dumps({"header": header}) + "\n")
        for record in records:
            handle.write(json.dumps(record, default=repr) + "\n")
    print(f"wrote {RUN_OUT}")


def _printed_cases(stdout):
    """The test's printed cases, in order: label, true error, reported abserr, converged."""
    cases = []
    label = None
    true = None
    for line in stdout.split("\n"):
        line = line.strip()
        if line.startswith("@@"):
            label = line.split(":")[0]
            true = None
            if "true abserr=" in line:
                # test_abserr_bounds_truth prints one line per oracle
                cases.append(
                    {
                        "label": label,
                        "true": float(line.split("true abserr=")[1].split(",")[0]),
                        "reported": float(
                            line.split("reported abserr=")[1].split(",")[0]
                        ),
                        "converged": None,
                    }
                )
        elif line.startswith("relerr=") or line.startswith("abserr="):
            true = float(line.split("abserr=")[1].split(",")[0])
        elif line.startswith("reported abserr="):
            reported = float(line.split("reported abserr=")[1].split(",")[0])
            converged = line.split("converged=")[1].split(",")[0] == "True"
            cases.append(
                {
                    "label": label,
                    "true": true,
                    "reported": reported,
                    "converged": converged,
                }
            )
    return cases


def _near_singular(label):
    """test_YJJ_log_singularity's near-singular cases: eps <= 1e-4, read from the case label
    ('... eps=1e-06 lo')."""
    if label is None or "eps=" not in label:
        return False
    return float(label.split("eps=")[1].split()[0]) <= 1e-4


SHORT = {
    "test_YJJ_log_singularity": "logsing",
    "test_JJJ": "JJJ",
    "test_YJJ": "YJJ",
    "test_abserr_bounds_truth": "bounds",
}


def _oracle(result):
    kind = result["kind"]
    mu, nu, sigma = (int(round(o)) for o in result["orders"])
    return f"{kind[0]}{mu}{nu}{sigma}"


def do_report():
    with open(RUN_OUT) as handle:
        run_header = json.loads(handle.readline())["header"]
        records = [json.loads(line) for line in handle]

    lines = prov.header(
        __file__, "section 2 test 8, the LiouvilleGreen seeded draws (report)", INPUTS
    )
    lines.append(f"# input {RUN_OUT.relative_to(REPO)} sha256 {prov.sha256(RUN_OUT)}")
    lines.append("# the run's own header:")
    lines += [f"#   {line[2:]}" for line in run_header]
    lines.append("")

    by_key = {(r["seed"], r["mode"], r["test"]): r for r in records}
    expected = {(s, m, t) for s in SEEDS for m in MODES for t in TESTS}
    missing = sorted(expected - set(by_key))
    lines.append(f"units {len(records)} of {len(expected)}; missing {missing}")
    modules = sorted(
        {(r["mode"], Path(r["three_bessel_module"]).name) for r in records}
    )
    lines.append(f"three_bessel_integrals used, by mode: {modules}")
    lines.append(
        "parent module sha256: "
        f"{sorted({r['parent_module_sha256'] for r in records if r['mode'] == 'parent'})}"
    )

    stops = []
    outcome_rows = []
    pairs = []
    result_pairs = []
    for test in TESTS:
        for seed in SEEDS:
            parent = by_key.get((seed, "parent", test))
            new = by_key.get((seed, "new", test))
            if parent is None or new is None:
                continue
            outcome_rows.append((test, seed, parent, new))
            if new["outcome"] != "pass" and parent["outcome"] == "pass":
                stops.append((test, seed))
            if parent["num_calls"] != new["num_calls"]:
                lines.append(
                    f"!! {test} seed {seed}: different numbers of Levin calls"
                    f" ({parent['num_calls']} / {new['num_calls']})"
                )
            for pc, nc in zip(parent["calls"], new["calls"]):
                assert pc["x_span"] == nc["x_span"], (test, seed, pc["index"])
                pairs.append((test, seed, pc, nc))
            assert len(parent["results"]) == len(new["results"]), (test, seed)
            for pr, nr in zip(parent["results"], new["results"]):
                assert pr["kqs"] == nr["kqs"] and pr["orders"] == nr["orders"]
                result_pairs.append((test, seed, pr, nr))

    lines.append("")
    lines.append("== Test outcomes, parent / new")
    for test, seed, parent, new in outcome_rows:
        lines.append(
            f"   {test:26s} seed {seed:2d}  {parent['outcome']:5s} {new['outcome']:5s}"
            f"  calls {new['num_calls']:4d}  {parent['elapsed_s']:7.1f} s {new['elapsed_s']:7.1f} s"
        )
    for test in TESTS:
        tally = {}
        for t, seed, parent, new in outcome_rows:
            if t == test:
                key = (parent["outcome"], new["outcome"])
                tally[key] = tally.get(key, 0) + 1
        lines.append(f"   {test}: (parent, new) -> count {tally}")

    lines.append("")
    lines.append(
        "== Stop (prompt section 7): a seeded draw that fails under 06h and passes at the parent"
    )
    lines.append(f"   {stops if stops else 'none'}")

    conv = {
        (a, b): sum(
            1 for _, _, pc, nc in pairs if pc["converged"] == a and nc["converged"] == b
        )
        for a in (True, False)
        for b in (True, False)
    }
    same = sum(
        1
        for _, _, pc, nc in pairs
        if pc["value"] == nc["value"]
        and pc["abserr"] == nc["abserr"]
        and pc["regions_sha256"] == nc["regions_sha256"]
    )
    lines.append("")
    lines.append("== Totals over every unit")
    lines.append(
        f"   Levin calls {len(pairs)}; bit-identical in value, abserr and regions {same};"
        f" converged (parent, new): T,T {conv[(True, True)]}  T,F {conv[(True, False)]}"
        f"  F,T {conv[(False, True)]}  F,F {conv[(False, False)]}"
    )
    lines.append(
        f"   regions {sum(pc['num_regions'] for *_, pc, _ in pairs)} ->"
        f" {sum(nc['num_regions'] for *_, nc in pairs)}; solves"
        f" {sum(pc['evaluations'] for *_, pc, _ in pairs)} ->"
        f" {sum(nc['evaluations'] for *_, nc in pairs)}; phase-limited calls"
        f" {sum(1 for *_, pc, _ in pairs if pc['num_phase_limited_regions'])} ->"
        f" {sum(1 for *_, nc in pairs if nc['num_phase_limited_regions'])}"
    )
    lines.append(
        f"   converged calls: parent {sum(1 for *_, pc, _ in pairs if pc['converged'])}, new"
        f" {sum(1 for *_, nc in pairs if nc['converged'])}; calls at depth_max: parent"
        f" {sum(1 for *_, pc, _ in pairs if pc['max_depth'] >= pc['depth_max'])}, new"
        f" {sum(1 for *_, nc in pairs if nc['max_depth'] >= nc['depth_max'])}"
    )
    rconv = {
        (a, b): sum(
            1
            for *_, pr, nr in result_pairs
            if pr["converged"] == a and nr["converged"] == b
        )
        for a in (True, False)
        for b in (True, False)
    }
    lines.append(
        f"   three-Bessel results {len(result_pairs)}; converged (parent, new): T,T"
        f" {rconv[(True, True)]}  T,F {rconv[(True, False)]}  F,T {rconv[(False, True)]}"
        f"  F,F {rconv[(False, False)]}"
    )
    moved = [
        abs(nr["value"] - pr["value"]) / (pr["abserr"] + nr["abserr"])
        for *_, pr, nr in result_pairs
        if pr["abserr"] + nr["abserr"] > 0
    ]
    lines.append(
        "   |value_06h - value_parent| / (abserr_parent + abserr_06h): max"
        f" {max(moved):.4g}; results with ratio > 1: {sum(1 for m in moved if m > 1)}"
    )

    lines.append("")
    lines.append(
        "== Honesty at the integral level -- every printed case (a three-Bessel integral against"
        " its closed form) dishonest (true error > reported abserr) under either code"
    )
    lines.append(
        "   test seed case# label | parent: true, reported, converged | new: true, reported,"
        " converged | HONEST->DISHONEST marks a case honest at the parent and dishonest under 06h"
    )
    counts = {}
    flips = []
    for t, seed, parent, new in outcome_rows:
        pcases, ncases = _printed_cases(parent["stdout"]), _printed_cases(new["stdout"])
        if len(pcases) != len(ncases):
            lines.append(f"!! {t} seed {seed}: printed cases differ in number")
            continue
        for i, (pc, nc) in enumerate(zip(pcases, ncases)):
            pd, nd = pc["true"] > pc["reported"], nc["true"] > nc["reported"]
            key = (t, pd, nd)
            counts[key] = counts.get(key, 0) + 1
            if pd or nd:
                mark = " HONEST->DISHONEST" if (nd and not pd) else ""
                if nd and not pd:
                    flips.append((t, seed, i, nc["converged"]))
                if (
                    t == "test_YJJ_log_singularity"
                    and _near_singular(pc["label"])
                    and not mark
                ):
                    continue
                lines.append(
                    f"   {SHORT[t]} {seed:2d} #{i:3d} {pc['label']} | {pc['true']:.4e}"
                    f" {pc['reported']:.4e} {pc['converged']} | {nc['true']:.4e}"
                    f" {nc['reported']:.4e} {nc['converged']}{mark}"
                )
    lines.append(
        "   (test_YJJ_log_singularity's near-singular cases are counted below, and listed only"
        " where they change from honest to dishonest)"
    )
    lines.append("   summary (test, dishonest at parent, dishonest under 06h) -> cases:")
    for key in sorted(counts):
        lines.append(f"      {key}: {counts[key]}")
    lines.append(
        f"   honest at the parent and dishonest under 06h: {len(flips)} cases, of which flagged"
        f" converged under 06h: {sum(1 for f_ in flips if f_[3])}"
    )

    lines.append("")
    lines.append(
        "== The honesty table: per test and code, printed cases; dishonest; dishonest and"
        " flagged converged; worst true/reported (and among converged). test_YJJ_log_singularity"
        " is split: its near-singular cases (eps <= 1e-4, from the case label) and the rest"
    )
    for test, part in [
        ("test_YJJ_log_singularity", "near-singular"),
        ("test_YJJ_log_singularity", "rest"),
    ] + [(t, "all") for t in TESTS if t != "test_YJJ_log_singularity"]:
        for mode in MODES:
            cases = []
            for t, seed, parent, new in outcome_rows:
                if t == test:
                    record = parent if mode == "parent" else new
                    cases += _printed_cases(record["stdout"])
            if part != "all":
                near = [_near_singular(c["label"]) for c in cases]
                cases = [
                    c for c, ns in zip(cases, near) if ns == (part == "near-singular")
                ]
            if not cases:
                continue
            dishonest = [c for c in cases if c["true"] > c["reported"]]
            ratios = [c["true"] / c["reported"] for c in cases if c["reported"] > 0]
            ratios_conv = [
                c["true"] / c["reported"]
                for c in cases
                if c["reported"] > 0 and c["converged"]
            ]
            lines.append(
                f"   {test:26s} {part:13s} {mode:6s} cases {len(cases):4d}; dishonest"
                f" {len(dishonest):4d}; dishonest and converged"
                f" {sum(1 for c in dishonest if c['converged']):4d}; worst ratio"
                f" {max(ratios, default=float('nan')):.4g}; worst among converged"
                f" {max(ratios_conv, default=float('nan')):.4g}"
            )

    lines.append("")
    lines.append(
        "== Per oracle: three-Bessel results over every unit, under the parent and 06h -- count;"
        " the largest |value|; the largest abserr; and max |value_06h - value_parent| /"
        " (abserr_parent + abserr_06h). The true errors are in the honesty table above and, for"
        " the fixed triple, below"
    )
    oracles = sorted({_oracle(pr) for *_, pr, _ in result_pairs})
    for oracle in oracles:
        rows = [(pr, nr) for *_, pr, nr in result_pairs if _oracle(pr) == oracle]
        lines.append(
            f"   {oracle}: {len(rows)} results; |value| max parent"
            f" {max(abs(p['value']) for p, _ in rows):.6e} new"
            f" {max(abs(n['value']) for _, n in rows):.6e}; abserr max parent"
            f" {max(p['abserr'] for p, _ in rows):.4e} new"
            f" {max(n['abserr'] for _, n in rows):.4e}; moved/combined-abserr max"
            f" {max(abs(n['value'] - p['value']) / (p['abserr'] + n['abserr']) for p, n in rows):.4g}"
        )

    lines.append("")
    lines.append(
        "== test_abserr_bounds_truth, the fixed triple (1.3, 1.7, 2.1), seed 0: per oracle,"
        " value, abserr and true error under the parent and 06h"
    )
    for t, seed, parent, new in outcome_rows:
        if t != "test_abserr_bounds_truth" or seed != 0:
            continue
        pcases, ncases = _printed_cases(parent["stdout"]), _printed_cases(new["stdout"])
        for pr, nr, pc, nc in zip(parent["results"], new["results"], pcases, ncases):
            lines.append(
                f"   {_oracle(pr)}: parent value {pr['value']!r}, abserr {pr['abserr']:.4e},"
                f" true {pc['true']:.4e} | 06h value {nr['value']!r}, abserr"
                f" {nr['abserr']:.4e}, true {nc['true']:.4e}"
            )
        same_every_seed = all(
            json.dumps(r[2]["results"]) == json.dumps(parent["results"])
            and json.dumps(r[3]["results"]) == json.dumps(new["results"])
            for r in outcome_rows
            if r[0] == "test_abserr_bounds_truth"
        )
        lines.append(f"   identical at every seed, under each code: {same_every_seed}")

    lines.append("")
    lines.append("== Calls reaching depth_max, under either")
    capped = [
        (t, s, pc, nc)
        for t, s, pc, nc in pairs
        if pc["max_depth"] >= pc["depth_max"] or nc["max_depth"] >= nc["depth_max"]
    ]
    for t, s, pc, nc in capped:
        lines.append(
            f"   {t} seed {s} #{pc['index']} {pc['notify_label']} x_span"
            f" [{pc['x_span'][0]:.6g}, {pc['x_span'][1]:.6g}] rtol {pc['rtol']:g}:"
            f" regions {pc['num_regions']} -> {nc['num_regions']}, depth {pc['max_depth']} ->"
            f" {nc['max_depth']}, solves {pc['evaluations']} -> {nc['evaluations']}, converged"
            f" {pc['converged']} -> {nc['converged']}, abserr {pc['abserr']:.4e} ->"
            f" {nc['abserr']:.4e}"
        )
    if not capped:
        lines.append("   none")

    lines.append("")
    lines.append(
        "== Levin calls whose convergence changed: test, seed, call, phase group, rtol |"
        " regions, depth, abserr, parent -> 06h"
    )
    changed = [(t, s, pc, nc) for t, s, pc, nc in pairs if pc["converged"] != nc["converged"]]
    for t, s, pc, nc in changed:
        lines.append(
            f"   {SHORT[t]} {s:2d} #{pc['index']:3d} {pc['notify_label']} {pc['rtol']:g} |"
            f" {pc['num_regions']} -> {nc['num_regions']}, {pc['max_depth']} ->"
            f" {nc['max_depth']}, {pc['abserr']:.4e} -> {nc['abserr']:.4e},"
            f" {pc['converged']} -> {nc['converged']}"
        )
    if not changed:
        lines.append("   none")

    lines.append("")
    lines.append(
        "== Per seed: calls; regions, solves, largest depth, converged count, parent -> 06h"
    )
    for seed in SEEDS:
        sp = [(pc, nc) for t, s, pc, nc in pairs if s == seed]
        lines.append(
            f"   seed {seed:2d}: {len(sp)} calls; regions {sum(p['num_regions'] for p, _ in sp)}"
            f" -> {sum(n['num_regions'] for _, n in sp)}; solves"
            f" {sum(p['evaluations'] for p, _ in sp)} -> {sum(n['evaluations'] for _, n in sp)};"
            f" depth {max((p['max_depth'] for p, _ in sp), default=None)} ->"
            f" {max((n['max_depth'] for _, n in sp), default=None)}; converged"
            f" {sum(p['converged'] for p, _ in sp)} -> {sum(n['converged'] for _, n in sp)}"
        )

    lines.append("")
    lines.append(
        "== Every Levin call: test, seed, call index, phase group, rtol | parent -> 06h: regions,"
        " solves, max depth, converged"
    )
    for t, s, pc, nc in pairs:
        lines.append(
            f"   {SHORT[t]} {s:2d} #{pc['index']:3d} {pc['notify_label']} {pc['rtol']:g} |"
            f" {pc['num_regions']} -> {nc['num_regions']}, {pc['evaluations']} ->"
            f" {nc['evaluations']}, {pc['max_depth']} -> {nc['max_depth']},"
            f" {pc['converged']} -> {nc['converged']}"
        )

    with open(REPORT_OUT, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"wrote {REPORT_OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "report"))
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    if args.action == "run":
        do_run(args.processes)
    else:
        do_report()


if __name__ == "__main__":
    main()
