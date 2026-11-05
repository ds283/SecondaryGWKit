"""levin-phase-floor prompt 06k, Part L ("The LiouvilleGreen draws"): 06g's test 9 harness
(docs/levin-phase-floor/06g/calibration/scripts/lg_seeded_draws.py, a record, not edited; its copy is in
state/scripts_06g_calibration_reference/), changed only where marked "prompt 06k":
- the modes are ``tree`` (the tree's module, 06g's code, the baseline) and the variants ``identity``,
  ``route1``, ``route2`` and ``route3``, each loaded by ``variant_loader.install`` in the unit's fresh
  process before anything imports AdaptiveLevin (scripts/variants/levin_quadrature_<mode>.py, its SHA-256
  recorded); every mode records its calls through the loaded module's own ``_adaptive_levin``; the
  parent's driver is not used;
- each call's record also keeps the route's firing count, and for route 3 the leaves left;
- each unit runs with its working directory in var/levin-phase-floor-06k/state/lg_cwd/ (06g: the tree), so
  that nothing a solve might write lands in the tree; the tree stays first on sys.path;
- ``report`` is not used (it compares 06g's two modes); lg_compare_06k.py reads the run.
06g's text follows.

levin-phase-floor prompt 06g, section 2 test 9: LiouvilleGreen/tests/test_3bessel_analytic.py over
20 seeded draws, at the parent (the dispatch commit's AdaptiveLevin/levin_quadrature.py, 06e's driver)
and at 06g's.

06h's harness of the same name (docs/levin-phase-floor/06h/calibration/scripts/lg_seeded_draws.py),
with what is swapped between the two modes changed back to 06e's: 06g changes the driver and not
LiouvilleGreen, so in mode "parent" every Levin call is made by the parent module's _adaptive_levin
(prov.load_parent(), read with `git show` and imported under another name), handed the working
tree's _Basis_SinCos object, which 06g does not change; in both modes the integrands are the working
tree's LiouvilleGreen. Each call also records 06g's new keys (node_limited, sum E, sum |E|) where the
driver returns them, and, from a recorder on the driver module's used_interval, every leaf the trust
check is taken on (those later bisected included), by test 3's classes: r < 1/16, [1/16, 1/4),
[1/4, 1) and >= 1, each split by delta1 above or below the leaf's floor. Each unit is one
(seed, mode, test). In a fresh worker process it

- wraps AdaptiveLevin.levin_quadrature._adaptive_levin, the module attribute adaptive_levin_sincos()
  calls it through, so that every Levin call is recorded, and in mode "parent" makes it by the
  parent's driver;
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

The report lists every test outcome (a test that fails under 06g and passes at the parent is the
prompt's stop, section 7); the honesty table (dishonest, dishonest-and-converged, worst ratio) under
both, with test_YJJ_log_singularity's near-singular cases (eps <= 1e-4) reported separately;
converged counts, calls at depth_max, node-limited calls and the exit's firings under both, the
near-singular cases' calls separately; test 3's table of checked leaves; per oracle, value, abserr
and the true error under both; and every Levin call's regions, solves, depth and convergence.

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
INPUTS = (
    TEST_MODULE,
    REPO / "LiouvilleGreen" / "three_bessel_integrals.py",
    REPO / "LiouvilleGreen" / "bessel_phase.py",
)
BANDS = ("r<1/16", "1/16<=r<1/4", "1/4<=r<1", "r>=1")
SIDES = ("d1>phi", "d1<=phi")


def _r_class(region):
    r = region.convergence_ratio
    if r < 1.0 / 16.0:
        band = "r<1/16"
    elif r < 0.25:
        band = "1/16<=r<1/4"
    elif r < 1.0:
        band = "1/4<=r<1"
    else:
        band = "r>=1"
    return f"{band}|{'d1<=phi' if region.delta1 <= region.phase_err else 'd1>phi'}"

SEEDS = list(range(20))
MODES = ("tree", "identity", "route1", "route2", "route3", "route2lit")  # prompt 06k (06g: ("parent", "new")); route2lit after stop 1
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
        "node_limited": data.get("node_limited"),
        "node_error_signed_sum": data.get("node_error_signed_sum"),
        "node_error_abs_sum": data.get("node_error_abs_sum"),
        "num_quarters_regions": sum(
            1 for r in data["regions"] if getattr(r, "value_kind", "own") == "quarters"
        ),
        # prompt 06k: the route's firings (None under the tree and the identity variant)
        "route_06k": _route_summary_06k(data.get("route_06k")),
    }


def _route_summary_06k(r):
    """prompt 06k: the route record without its per-leaf lists, plus the firings' depths and r."""
    if r is None:
        return None
    out = {k: v for k, v in r.items() if k not in ("firings", "left_returned")}
    out["firings_depth_r_d1_phi"] = [[f["depth"], f["r"], f["delta1"], f["phi_P"]] for f in r.get("firings", [])]
    return out


def run_unit(unit):
    """One (seed, mode, test), in a fresh worker process (maxtasksperchild=1)."""
    seed, mode, test = unit
    cwd_06k = HERE.parents[1] / "state" / "lg_cwd"  # prompt 06k: not the tree
    cwd_06k.mkdir(parents=True, exist_ok=True)
    os.chdir(cwd_06k)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    variant_06k = None
    if mode != "tree":  # prompt 06k: the variant, before anything imports AdaptiveLevin
        sys.path.insert(1, str(HERE.parent))
        import variant_loader

        vpath = HERE.parent / "variants" / f"levin_quadrature_{mode}.py"
        variant_loader.install(vpath, prov.sha256(vpath), mode)
        variant_06k = variant_loader.evidence()

    import random
    import unittest

    import AdaptiveLevin.levin_quadrature as lq

    driver_module, parent_sha = lq, None  # prompt 06k: every mode's own loaded module
    driver = driver_module._adaptive_levin

    # every leaf the driver makes, so that every checked leaf is seen
    made = []
    original_interval = driver_module.used_interval

    class RecordedInterval(original_interval):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    driver_module.used_interval = RecordedInterval
    calls = []

    def recording_adaptive_levin(*args, **kwargs):
        made.clear()
        data = driver(*args, **kwargs)
        record = _call_record(len(calls), args, kwargs, data)
        record["depth_max"] = kwargs.get("depth_max", lq.DEFAULT_LEVIN_MAX_DEPTH)
        classes = {}
        for region in made:
            if region.delta1 is not None:
                key = _r_class(region)
                classes[key] = classes.get(key, 0) + 1
        record["checked_classes"] = classes
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
        "driver_module": getattr(driver_module, "__file__", None),
        "variant_06k": variant_06k,  # prompt 06k
        "parent_module_sha256": parent_sha,
        "num_calls": len(calls),
        "calls": calls,
        "results": results,
        "stdout": stdout.getvalue(),
    }


def do_run(processes, modes=MODES, run_out=None):  # prompt 06k: modes and the output, after stop 1
    global RUN_OUT
    if run_out is not None:
        RUN_OUT = Path(run_out)
    units = [(seed, mode, test) for test in TESTS for seed in SEEDS for mode in modes]
    header = prov.header(
        __file__, "section 2 test 9, the LiouvilleGreen seeded draws (run)", INPUTS
    )
    header.append(
        f"# units {len(units)}: seeds {SEEDS[0]}-{SEEDS[-1]} x modes {tuple(modes)} x tests {TESTS};"
        f" processes {processes}; no parent mode (prompt 06k)"
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
    RUN_OUT.parent.mkdir(parents=True, exist_ok=True)  # prompt 06k
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
        __file__, "section 2 test 9, the LiouvilleGreen seeded draws (report)", INPUTS
    )
    lines.append(f"# input {RUN_OUT.relative_to(REPO)} sha256 {prov.sha256(RUN_OUT)}")
    lines.append("# the run's own header:")
    lines += [f"#   {line[2:]}" for line in run_header]
    lines.append("")

    by_key = {(r["seed"], r["mode"], r["test"]): r for r in records}
    expected = {(s, m, t) for s in SEEDS for m in MODES for t in TESTS}
    missing = sorted(expected - set(by_key))
    lines.append(f"units {len(records)} of {len(expected)}; missing {missing}")
    modules = sorted({(r["mode"], Path(r["driver_module"]).name) for r in records})
    lines.append(f"driver module used, by mode: {modules}")
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
        "== Stop (prompt section 7): a seeded draw that fails under 06g and passes at the parent"
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
        "   |value_06g - value_parent| / (abserr_parent + abserr_06g): max"
        f" {max(moved):.4g}; results with ratio > 1: {sum(1 for m in moved if m > 1)}"
    )

    lines.append("")
    lines.append(
        "== 06g's additions, per test: Levin calls; converged, at depth_max, node-limited, with an"
        " exit firing (calls with a floor-limited leaf) and floor-limited leaves, parent -> 06g;"
        " regions returning their quarters sum under 06g"
    )
    for test in TESTS:
        tp = [(pc, nc) for t, _, pc, nc in pairs if t == test]
        if not tp:
            continue
        lines.append(
            f"   {test:26s} calls {len(tp):5d}; converged"
            f" {sum(p['converged'] for p, _ in tp)} -> {sum(n['converged'] for _, n in tp)};"
            f" at depth_max {sum(p['max_depth'] >= p['depth_max'] for p, _ in tp)} ->"
            f" {sum(n['max_depth'] >= n['depth_max'] for _, n in tp)}; node-limited"
            f" {sum(bool(p['node_limited']) for p, _ in tp)} ->"
            f" {sum(bool(n['node_limited']) for _, n in tp)}; exit firing (calls)"
            f" {sum(p['num_phase_limited_regions'] > 0 for p, _ in tp)} ->"
            f" {sum(n['num_phase_limited_regions'] > 0 for _, n in tp)}; floor-limited leaves"
            f" {sum(p['num_phase_limited_regions'] for p, _ in tp)} ->"
            f" {sum(n['num_phase_limited_regions'] for _, n in tp)}; quarters regions (06g)"
            f" {sum(n['num_quarters_regions'] for _, n in tp)}"
        )
    for mode, index in (("parent", 2), ("06g", 3)):
        table = {}
        for row in pairs:
            for key, count in row[index]["checked_classes"].items():
                table[key] = table.get(key, 0) + count
        lines.append(
            f"   test 3's table, every checked leaf of every call ({mode}): r band x delta1 against"
            " the leaf's floor"
        )
        lines.append("      " + " ".join(f"{b:>12s}" for b in BANDS))
        for side in SIDES:
            lines.append(
                f"   {side:>7s}"
                + " ".join(f"{table.get(f'{b}|{side}', 0):12d}" for b in BANDS)
            )
        lines.append(f"      total {sum(table.values())}")
    nl = [(t, s_, nc) for t, s_, _, nc in pairs if nc["node_limited"]]
    lines.append(f"   node-limited calls under 06g: {len(nl)}")
    for t, s_, nc in nl[:40]:
        lines.append(
            f"      {SHORT[t]} {s_:2d} #{nc['index']:3d} {nc['notify_label']} rtol {nc['rtol']:g}:"
            f" regions {nc['num_regions']}, depth {nc['max_depth']}, abserr {nc['abserr']:.4e},"
            f" sum E {nc['node_error_signed_sum']:.4e}, sum |E| {nc['node_error_abs_sum']:.4e}"
        )

    lines.append("")
    lines.append(
        "== Honesty at the integral level -- every printed case (a three-Bessel integral against"
        " its closed form) dishonest (true error > reported abserr) under either code"
    )
    lines.append(
        "   test seed case# label | parent: true, reported, converged | new: true, reported,"
        " converged | HONEST->DISHONEST marks a case honest at the parent and dishonest under 06g"
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
    lines.append("   summary (test, dishonest at parent, dishonest under 06g) -> cases:")
    for key in sorted(counts):
        lines.append(f"      {key}: {counts[key]}")
    lines.append(
        f"   honest at the parent and dishonest under 06g: {len(flips)} cases, of which flagged"
        f" converged under 06g: {sum(1 for f_ in flips if f_[3])}"
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
        "== Per oracle: three-Bessel results over every unit, under the parent and 06g -- count;"
        " the largest |value|; the largest abserr; and max |value_06g - value_parent| /"
        " (abserr_parent + abserr_06g). The true errors are in the honesty table above and, for"
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
        " value, abserr and true error under the parent and 06g"
    )
    for t, seed, parent, new in outcome_rows:
        if t != "test_abserr_bounds_truth" or seed != 0:
            continue
        pcases, ncases = _printed_cases(parent["stdout"]), _printed_cases(new["stdout"])
        for pr, nr, pc, nc in zip(parent["results"], new["results"], pcases, ncases):
            lines.append(
                f"   {_oracle(pr)}: parent value {pr['value']!r}, abserr {pr['abserr']:.4e},"
                f" true {pc['true']:.4e} | 06g value {nr['value']!r}, abserr"
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
        " regions, depth, abserr, parent -> 06g"
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
        "== Per seed: calls; regions, solves, largest depth, converged count, parent -> 06g"
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
        "== Every Levin call: test, seed, call index, phase group, rtol | parent -> 06g: regions,"
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
    parser.add_argument("--modes", default=None)  # prompt 06k: a comma-separated subset of MODES (after stop 1)
    parser.add_argument("--out", default=None)  # prompt 06k: the run file (default RUN_OUT)
    args = parser.parse_args()
    if args.action == "run":
        modes = MODES if args.modes is None else tuple(args.modes.split(","))
        if any(m not in MODES for m in modes):
            raise SystemExit(f"modes must be among {MODES}")
        do_run(args.processes, modes, args.out)
    else:
        do_report()


if __name__ == "__main__":
    main()
