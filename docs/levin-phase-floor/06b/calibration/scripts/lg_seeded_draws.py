"""levin-phase-floor prompt 06b, section 2 test 5: LiouvilleGreen/tests/test_3bessel_analytic.py over
20 seeded draws, at the parent (by the bypass) and at the new code.

Each unit is one (seed, mode, test). In a fresh worker process it

- wraps AdaptiveLevin.levin_quadrature._adaptive_levin, the module attribute
  adaptive_levin_sincos() calls it through, so that every call is recorded and, in mode "parent",
  is made with _test_only_bypass_roundoff_premise=True (the round-off-limited exit as it was
  before 06b);
- seeds the global ``random`` instance with the seed, immediately before the test -- the test
  module draws its wavenumbers from ``random.uniform``, unseeded, and is not edited;
- runs the one test method through unittest, capturing what it prints.

The tests are every test of Test3BesselAnalytic that runs by default: test_JJJ, test_YJJ and
test_YJJ_log_singularity, which draw from ``random``, and test_abserr_bounds_truth, which does not
(a fixed triple), run at every seed all the same. test_YJJ_log_scaling is skipped by default
(THREE_BESSEL_DIAGNOSTIC_PLOTS unset) and asserts nothing; it is not run.

  run     execute every unit (a process pool), writing one JSON line per unit to
          ../out/lg_seeded_draws.jsonl, with a provenance header line first
  report  read that file and write ../out/lg_seeded_draws_report.txt

The report tabulates, per seed, every Levin call whose phase-limited regions changed between the
parent and the new code, checks that every call with no phase-limited region at the parent is
bit-identical in value, abserr, region list and evaluation count, and lists every test outcome; a
test that fails under the new code and passes at the parent is the prompt's stop (section 7).

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
THREE_BESSEL = REPO / "LiouvilleGreen" / "three_bessel_integrals.py"
INPUTS = (TEST_MODULE, THREE_BESSEL, REPO / "LiouvilleGreen" / "bessel_phase.py")

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
        "phase_limited_regions": [r for r in regions if r[7]],
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

    original = lq._adaptive_levin
    calls = []

    def recording_adaptive_levin(*args, **kwargs):
        if mode == "parent":
            kwargs["_test_only_bypass_roundoff_premise"] = True
        data = original(*args, **kwargs)
        calls.append(_call_record(len(calls), args, kwargs, data))
        return data

    lq._adaptive_levin = recording_adaptive_levin

    from LiouvilleGreen.tests.test_3bessel_analytic import Test3BesselAnalytic

    case = Test3BesselAnalytic(test)
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
        "num_calls": len(calls),
        "calls": calls,
        "stdout": stdout.getvalue(),
    }


def do_run(processes):
    units = [(seed, mode, test) for test in TESTS for seed in SEEDS for mode in MODES]
    header = prov.header(
        __file__, "section 2 test 5, the LiouvilleGreen seeded draws (run)", INPUTS
    )
    header.append(
        f"# units {len(units)}: seeds {SEEDS[0]}-{SEEDS[-1]} x modes {MODES} x tests {TESTS};"
        f" processes {processes}"
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


def _draw_lines(stdout):
    return [line.strip() for line in stdout.split("\n") if "k=" in line]


def do_report():
    with open(RUN_OUT) as handle:
        run_header = json.loads(handle.readline())["header"]
        records = [json.loads(line) for line in handle]

    lines = prov.header(
        __file__, "section 2 test 5, the LiouvilleGreen seeded draws (report)", INPUTS
    )
    lines.append(f"# input {RUN_OUT.relative_to(REPO)} sha256 {prov.sha256(RUN_OUT)}")
    lines.append("# the run's own header:")
    lines += [f"#   {line[2:]}" for line in run_header]
    lines.append("")

    by_key = {(r["seed"], r["mode"], r["test"]): r for r in records}
    expected = {(s, m, t) for s in SEEDS for m in MODES for t in TESTS}
    missing = sorted(expected - set(by_key))
    lines.append(f"units {len(records)} of {len(expected)}; missing {missing}")

    stops = []
    w1_violations = []
    changed_by_seed = {seed: [] for seed in SEEDS}
    outcome_rows = []
    totals = {
        "calls": 0,
        "calls_pl_parent": 0,
        "calls_pl_new": 0,
        "calls_changed": 0,
        "calls_pl_changed": 0,
    }

    for test in TESTS:
        for seed in SEEDS:
            parent = by_key.get((seed, "parent", test))
            new = by_key.get((seed, "new", test))
            if parent is None or new is None:
                continue
            outcome_rows.append(
                (test, seed, parent["outcome"], new["outcome"], parent, new)
            )
            if new["outcome"] != "pass" and parent["outcome"] == "pass":
                stops.append((test, seed))
            if _draw_lines(parent["stdout"]) != _draw_lines(new["stdout"]) and (
                parent["num_calls"] != new["num_calls"]
            ):
                lines.append(
                    f"!! {test} seed {seed}: the parent and new runs made different numbers"
                    f" of calls ({parent['num_calls']} / {new['num_calls']})"
                )
            for pc, nc in zip(parent["calls"], new["calls"]):
                totals["calls"] += 1
                assert pc["x_span"] == nc["x_span"], (test, seed, pc["index"])
                pl_p = pc["num_phase_limited_regions"]
                pl_n = nc["num_phase_limited_regions"]
                totals["calls_pl_parent"] += pl_p > 0
                totals["calls_pl_new"] += pl_n > 0
                same = (
                    pc["value"] == nc["value"]
                    and pc["abserr"] == nc["abserr"]
                    and pc["evaluations"] == nc["evaluations"]
                    and pc["regions_sha256"] == nc["regions_sha256"]
                )
                totals["calls_changed"] += not same
                if pl_p == 0 and not same:
                    w1_violations.append((test, seed, pc["index"]))
                if pc["phase_limited_regions"] != nc["phase_limited_regions"]:
                    totals["calls_pl_changed"] += 1
                    changed_by_seed[seed].append((test, pc, nc))

    lines.append("")
    lines.append("== Test outcomes, parent (bypass) / new")
    lines.append(
        f"   {'test':26s} {'seed':>4s}  {'parent':7s} {'new':7s}  {'calls':>5s}"
        f"  {'s parent':>8s} {'s new':>8s}"
    )
    for test, seed, po, no, parent, new in outcome_rows:
        lines.append(
            f"   {test:26s} {seed:4d}  {po:7s} {no:7s}  {new['num_calls']:5d}"
            f"  {parent['elapsed_s']:8.1f} {new['elapsed_s']:8.1f}"
        )
    for test in TESTS:
        tally = {}
        for t, seed, po, no, _, _ in outcome_rows:
            if t == test:
                tally[(po, no)] = tally.get((po, no), 0) + 1
        lines.append(f"   {test}: (parent, new) -> count {tally}")

    lines.append("")
    lines.append(
        "== Stop (prompt section 7): a seeded draw that fails under the new code and passes"
        " at the parent"
    )
    lines.append(f"   {stops if stops else 'none'}")

    lines.append("")
    lines.append(
        "== Calls with no phase-limited region at the parent that are not bit-identical"
        " (value, abserr, evaluations, region list)"
    )
    lines.append(f"   {w1_violations if w1_violations else 'none'}")

    lines.append("")
    lines.append("== Totals over every unit")
    lines.append(
        f"   Levin calls {totals['calls']}; with a phase-limited region at the parent"
        f" {totals['calls_pl_parent']}, under the new code {totals['calls_pl_new']};"
        f" calls whose result moved {totals['calls_changed']}; calls whose phase-limited"
        f" regions changed {totals['calls_pl_changed']}"
    )

    lines.append("")
    lines.append(
        "== Per seed, summary of the calls whose phase-limited regions changed"
    )
    lines.append(
        "   changed calls by test (log_singularity / JJJ / YJJ / abserr_bounds_truth);"
        " calls phase-limited parent -> new; converged F->T, T->F; largest max depth"
        " parent -> new; new calls at depth_max 20; abserr new/parent median and range"
    )
    short = {
        "test_YJJ_log_singularity": 0,
        "test_JJJ": 1,
        "test_YJJ": 2,
        "test_abserr_bounds_truth": 3,
    }
    for seed in SEEDS:
        changed = changed_by_seed[seed]
        counts = [0, 0, 0, 0]
        for test, _, _ in changed:
            counts[short[test]] += 1
        pl_p = sum(pc["num_phase_limited_regions"] > 0 for _, pc, _ in changed)
        pl_n = sum(nc["num_phase_limited_regions"] > 0 for _, _, nc in changed)
        f_to_t = sum((not pc["converged"]) and nc["converged"] for _, pc, nc in changed)
        t_to_f = sum(pc["converged"] and (not nc["converged"]) for _, pc, nc in changed)
        depth_p = max((pc["max_depth"] for _, pc, _ in changed), default=None)
        depth_n = max((nc["max_depth"] for _, _, nc in changed), default=None)
        at_cap = sum(nc["max_depth"] >= 20 for _, _, nc in changed)
        ratios = sorted(
            nc["abserr"] / pc["abserr"] for _, pc, nc in changed if pc["abserr"] > 0
        )
        median = ratios[len(ratios) // 2] if ratios else float("nan")
        lines.append(
            f"   seed {seed:2d}: {len(changed):3d} calls"
            f" ({counts[0]}/{counts[1]}/{counts[2]}/{counts[3]});"
            f" pl {pl_p} -> {pl_n}; converged F->T {f_to_t}, T->F {t_to_f};"
            f" depth {depth_p} -> {depth_n}; at cap {at_cap};"
            f" abserr ratio median {median:.3g}"
            f" [{ratios[0] if ratios else float('nan'):.3g},"
            f" {ratios[-1] if ratios else float('nan'):.3g}]"
        )

    lines.append("")
    lines.append(
        "== Largest printed |relerr| against the closed form, per test, over all seeds"
        " (triangle and non-triangle cases alike; parent / new)"
    )
    for test in TESTS:
        worst = {}
        for t, seed, po, no, parent, new in outcome_rows:
            if t != test:
                continue
            for mode, record in (("parent", parent), ("new", new)):
                for line in record["stdout"].split("\n"):
                    if (
                        "relerr=" in line
                        and "abserr=" in line
                        and "reported" not in line
                    ):
                        value = float(line.split("relerr=")[1].split(",")[0])
                        if abs(value) > worst.get(mode, (0.0, None))[0]:
                            worst[mode] = (abs(value), seed)
        lines.append(
            f"   {test}: parent {worst.get('parent')}, new {worst.get('new')}"
            " (value, seed)"
        )

    lines.append("")
    lines.append("== Per seed: the calls whose phase-limited regions changed")
    lines.append(
        "   test / call index / phase group / x_span: parent -> new phase-limited regions,"
        " regions, max depth; value; abserr; converged"
    )
    for seed in SEEDS:
        changed = changed_by_seed[seed]
        lines.append(f"-- seed {seed}: {len(changed)} call(s)")
        for test, pc, nc in changed:
            lines.append(
                f"   {test} #{pc['index']} {pc['notify_label']} x_span"
                f" [{pc['x_span'][0]:.6g}, {pc['x_span'][1]:.6g}] rtol {pc['rtol']:g}:"
                f" pl {pc['num_phase_limited_regions']} -> {nc['num_phase_limited_regions']},"
                f" regions {pc['num_regions']} -> {nc['num_regions']},"
                f" depth {pc['max_depth']} -> {nc['max_depth']}"
            )
            lines.append(
                f"      value {pc['value']!r} -> {nc['value']!r};"
                f" abserr {pc['abserr']:.4e} -> {nc['abserr']:.4e};"
                f" converged {pc['converged']} -> {nc['converged']}"
            )
            for region in pc["phase_limited_regions"]:
                lines.append(
                    f"      parent pl region [{region[0]:.6g}, {region[1]:.6g}] depth"
                    f" {region[2]} {region[3]} abserr {region[4]:.3e} floor {region[6]:.3e}"
                )
            for region in nc["phase_limited_regions"]:
                lines.append(
                    f"      new    pl region [{region[0]:.6g}, {region[1]:.6g}] depth"
                    f" {region[2]} {region[3]} abserr {region[4]:.3e} floor {region[6]:.3e}"
                )

    lines.append("")
    lines.append("== The draws, per test and seed (from the new run's output)")
    for test, seed, po, no, parent, new in outcome_rows:
        if test == "test_abserr_bounds_truth":
            continue
        lines.append(f"-- {test} seed {seed}")
        lines += [f"   {line}" for line in _draw_lines(new["stdout"])]

    lines.append("")
    lines.append(
        "== Every failing or erroring test's message (either mode), and every test's printed"
        " relerr/abserr lines where they differ between the modes"
    )
    for test, seed, po, no, parent, new in outcome_rows:
        for mode, record in (("parent", parent), ("new", new)):
            if record["outcome"] != "pass":
                lines.append(f"-- {test} seed {seed} {mode}: {record['outcome']}")
                lines += [
                    f"   {line}" for line in record["detail"].rstrip().split("\n")[-6:]
                ]
        if parent["stdout"] != new["stdout"]:
            p_lines = parent["stdout"].split("\n")
            n_lines = new["stdout"].split("\n")
            diffs = [
                (p, n) for p, n in zip(p_lines, n_lines) if p != n and "relerr" in p
            ]
            if diffs:
                lines.append(f"-- {test} seed {seed}: printed lines that moved")
                for p, n in diffs:
                    lines.append(f"   parent {p.strip()}")
                    lines.append(f"   new    {n.strip()}")

    REPORT_OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {REPORT_OUT}")
    print(
        f"stops {stops}; W1-style violations {len(w1_violations)}; totals {totals}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--processes", type=int, default=8)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "run":
        do_run(args.processes)
    else:
        do_report()


if __name__ == "__main__":
    main()
