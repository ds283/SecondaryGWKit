"""
levin-phase-floor prompt 07d, Part B, section 3 item 7 ("The LiouvilleGreen draws"): 07c's harness,
docs/levin-phase-floor/07c/code/lg_draws_07c.py (itself 07b's and 07a's, changed where marked "prompt 07b" and
"prompt 07c"), copied, changed only where marked "prompt 07d". LiouvilleGreen/tests/test_3bessel_analytic.py over seeds
0-19, under three trees:
- the parent: the dispatch commit ca1d8cfa (07c's code), in a temporary detached worktree outside the repository;
- the commit: the main checkout's tree with Part B's change;
- V93T10: a second temporary detached worktree at ca1d8cfa with Part A''s variant V93T10,
  docs/levin-phase-floor/07d/measurement-a2/scripts/variants/levin_quadrature_a2_V93T10.py (SHA-256 f85976fc...), copied
  over its AdaptiveLevin/levin_quadrature.py, so that the commit is compared with V93T10 call by call (section 9.4.5,
  Z8, on the laptop's problems).
What it keeps of 07c's (and 07b's, 07a's and 07's before it):
- each unit is one (seed, test) in a process of its own, with a wall-clock limit (UNIT_TIMEOUT, 1200 s);
- the global ``random`` instance is seeded with the seed immediately before the test (the test module draws its
  wavenumbers from ``random.uniform`` unseeded, and is not edited);
- the tests are every test of Test3BesselAnalytic that runs by default: test_JJJ, test_YJJ,
  test_YJJ_log_singularity and test_abserr_bounds_truth; test_YJJ_log_scaling is skipped by default;
- every Levin call is recorded through a wrapper on AdaptiveLevin.levin_quadrature._adaptive_levin, the module
  attribute adaptive_levin_sincos() calls it through; and every three-Bessel result through wrappers on the test
  module's quad_JJJ and quad_YJJ, with its analytic value from the test's own oracle classes;
- 07a's route instrument on _Basis_SinCos.build_Levin_data is kept as it was, at 07a's 6 pi (its counts compare 07's
  estimate with 07a's V at the number 6 pi, not the gate);
- 07b's two refusal counts and 07c's three chase counts, recorded and left out of the result digest.
What it changes (prompt 07d):
- each call records 07d's counts, num_cc_solves, num_cc_gate_route_changes, num_cc_noise_set and num_cc_resampled
  (absent at the parent and read as 0 there). In V93T10's tree the variant keeps its counts in the module dict _A2_C,
  cleared at each adaptive_levin_sincos() call and copied into the result after _adaptive_levin() returns, so the
  wrapper clears _A2_C before the driver runs and reads it after: num_a2_cc93 is recorded as num_cc_solves,
  num_a2_route_changed as num_cc_gate_route_changes and num_a2_resampled as num_cc_resampled; the variant has no
  num_cc_noise_set, recorded as None there. The digest of the whole result leaves them out, so that the three trees
  compare;
- the report classes a call by whether the fix acts in it at the commit (num_cc_solves positive), checks that every
  call in which it does not is bit-identical to the parent's, tabulates the calls in which it does with every
  converged change listed (07a's three draw calls and 07c's 26 newly unconverged calls named), and compares every
  call of the commit with V93T10's, counts included.

  run     --tree PATH --label LABEL [--jobs N]: every unit under that tree, one JSON file per unit in
          out/LABEL/, and out/LABEL/run.json with the provenance
  unit    (internal) one unit
  report  read out/parent/, out/commit/ and out/V93T10/ and write out/lg_draws_report.txt and
          out/lg_draws_report.json

No Ray and no datastore; nothing is written outside out/.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
SEEDS = list(range(20))
TESTS = ("test_YJJ_log_singularity", "test_JJJ", "test_YJJ", "test_abserr_bounds_truth")
UNIT_TIMEOUT = 1200.0
SIX_PI = 6.0 * math.pi
TWENTY_PI = 20.0 * math.pi
REGION_FIELDS = (
    "start",
    "end",
    "depth",
    "type",
    "abserr",
    "relerr",
    "phase_err",
    "phase_limited",
    "value",
    "value_kind",
    "value_own",
    "trusted",
    "delta1",
    "delta2",
    "convergence_ratio",
    "declared_resolution",
    "total_err",
    "budget_err",
    "node_error_signed",
    "phase_err_less_node",
    "phase_err_roundoff",
    "phase_err_declared",
    "abserr_truncation",
    "window",
    "guarded",
    "guard_at_depth_max",
)


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canon(x):
    import numpy as np

    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x).hex()
    if isinstance(x, np.ndarray):
        return [canon(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [canon(v) for v in x]
    if isinstance(x, dict):
        return {
            str(k): canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
        }
    if x is None or isinstance(x, str):
        return x
    return repr(x)


def digest(obj):
    return hashlib.sha256(json.dumps(canon(obj), sort_keys=True).encode()).hexdigest()


def run_unit(tree, seed, test, out_file):
    tree = Path(tree).resolve()
    os.chdir(tree)
    sys.path.insert(0, str(tree))
    import numpy as np
    import unittest

    import AdaptiveLevin.levin_quadrature as lq
    import LiouvilleGreen.tests.test_3bessel_analytic as t3

    levin_calls = []
    bessel_results = []
    route = {"forced": False, "current": None}

    original_impl = lq._adaptive_levin_subregion_impl

    def impl(*args, **kwargs):
        previous = route["forced"]
        route["forced"] = bool(kwargs.get("force_direct", False))
        try:
            return original_impl(*args, **kwargs)
        finally:
            route["forced"] = previous

    original_build = lq._Basis_SinCos.build_Levin_data

    def build(self, grid, Dmat, *args, **kwargs):
        out = original_build(self, grid, Dmat, *args, **kwargs)
        current = route["current"]
        if current is not None and not route["forced"]:
            theta_prime, V = out[1], out[4]
            estimate = float(
                np.mean(np.fabs(theta_prime)) * np.fabs(grid[0] - grid[-1])
            )
            current["regions_gated"] += 1
            if (estimate < SIX_PI) != (V < SIX_PI):
                current["route_changed"] += 1
                current["route_changes"].append(
                    {
                        "span": [float(grid[-1]), float(grid[0])],
                        "V": V,
                        "estimate": estimate,
                    }
                )
            elif not (V < SIX_PI) and ((estimate > TWENTY_PI) != (V > TWENTY_PI)):
                current["lu_changed"] += 1
            if lq._changes_sign(theta_prime):
                current["node_partition"] += 1
        return out

    original_driver = lq._adaptive_levin
    # prompt 07d: V93T10 keeps its counts in the module dict _A2_C (absent elsewhere)
    a2_counts = getattr(lq, "_A2_C", None)

    def driver(*args, **kwargs):
        record = {
            "regions_gated": 0,
            "route_changed": 0,
            "lu_changed": 0,
            "node_partition": 0,
            "route_changes": [],
        }
        outer = route["current"]
        route["current"] = record
        # prompt 07d: the variant's counts, cleared before the driver runs and read after it
        if a2_counts is not None:
            a2_counts.clear()
        try:
            data = original_driver(*args, **kwargs)
        finally:
            route["current"] = outer
        a2 = dict(a2_counts) if a2_counts is not None else None
        x_span = args[0] if args else kwargs["x_span"]
        windows = data.get("stationary_windows", [])
        record.update(
            {
                "index": len(levin_calls),
                "x_span": [float(x_span[0]), float(x_span[1])],
                "atol": kwargs.get("atol"),
                "rtol": kwargs.get("rtol"),
                "value": float(data["value"]),
                "value_hex": float(data["value"]).hex(),
                "abserr": float(data["abserr"]),
                "abserr_hex": float(data["abserr"]).hex(),
                "converged": bool(data["converged"]),
                "num_regions": int(data["num_regions"]),
                "max_depth": int(data["max_depth"]),
                "evaluations": int(data["evaluations"]),
                "num_untrusted_regions": data.get("num_untrusted_regions"),
                # prompt 07b: the rate rules' two counts (absent before 07b's code)
                "num_edge_rule_refusals": int(data.get("num_edge_rule_refusals", 0)),
                "num_range_rule_refusals": int(data.get("num_range_rule_refusals", 0)),
                # prompt 07c: the stop's three counts ((S)'s names in its tree; absent at the parent)
                "num_chase_stops": int(
                    data.get("num_chase_stops", data.get("num_s07c_stops", 0))
                ),
                "num_chase_stopped_leaves": int(
                    data.get(
                        "num_chase_stopped_leaves",
                        data.get("num_s07c_stopped_leaves", 0),
                    )
                ),
                "num_chase_noise_bisections": int(
                    data.get(
                        "num_chase_noise_bisections", data.get("num_s07c_counted", 0)
                    )
                ),
                # prompt 07d: 07d's counts (V93T10's names read from _A2_C in its tree; absent at the parent)
                "num_cc_solves": int(
                    a2.get("num_a2_cc93", 0)
                    if a2 is not None
                    else data.get("num_cc_solves", 0)
                ),
                "num_cc_gate_route_changes": int(
                    a2.get("num_a2_route_changed", 0)
                    if a2 is not None
                    else data.get("num_cc_gate_route_changes", 0)
                ),
                "num_cc_noise_set": (
                    None if a2 is not None else int(data.get("num_cc_noise_set", 0))
                ),
                "num_cc_resampled": int(
                    a2.get("num_a2_resampled", 0)
                    if a2 is not None
                    else data.get("num_cc_resampled", 0)
                ),
                "num_points": len(data.get("stationary_points", [])),
                "point_kinds": sorted(
                    p["kind"] for p in data.get("stationary_points", [])
                ),
                "num_windows": len(windows),
                "windows": [
                    {
                        "iroff1_count": w["iroff1_count"],
                        "stopped": w["stopped"],
                        "stopped_after_split": w.get("stopped_after_split"),
                        "leaves_falling_at_stop": w.get("leaves_falling_at_stop"),
                    }
                    for w in windows
                ],
                "num_window_pieces": data.get("num_window_pieces"),
                "num_guarded_regions": data.get("num_guarded_regions"),
                "num_guard_at_depth_max": data.get("num_guard_at_depth_max"),
                "location_skipped_phases": canon(data.get("location_skipped_phases")),
                "regions_sha256": digest(
                    [
                        [getattr(r, name) for name in REGION_FIELDS]
                        for r in data["regions"]
                    ]
                ),
                "result_sha256": digest(
                    {
                        k: v
                        for k, v in data.items()
                        if k
                        not in (
                            "elapsed",
                            "regions",
                            "location_skipped_phases",
                            # prompt 07b
                            "num_edge_rule_refusals",
                            "num_range_rule_refusals",
                            # prompt 07c
                            "num_chase_stops",
                            "num_chase_stopped_leaves",
                            "num_chase_noise_bisections",
                            "num_s07c_stops",
                            "num_s07c_stopped_leaves",
                            "num_s07c_counted",
                            # prompt 07d
                            "num_cc_solves",
                            "num_cc_gate_route_changes",
                            "num_cc_noise_set",
                            "num_cc_resampled",
                        )
                    }
                ),
            }
        )
        levin_calls.append(record)
        return data

    oracles = {(o.mu, o.nu, o.sigma, "J"): o for o in t3.Jintegrals}
    oracles.update({(o.mu, o.nu, o.sigma, "Y"): o for o in t3.Yintegrals})

    def wrap(name, kind):
        evaluator = getattr(t3, name)

        def wrapped(*args, **kwargs):
            first = len(levin_calls)
            result = evaluator(*args, **kwargs)
            mu, nu, sigma, k, q, s = args[3:9]
            analytic = float(oracles[(mu, nu, sigma, kind)].analytic(k, q, s))
            bessel_results.append(
                {
                    "evaluator": name,
                    "mu_nu_sigma": [mu, nu, sigma],
                    "k_q_s": [k, q, s],
                    "rtol": kwargs.get("rtol"),
                    "value": float(result.value),
                    "abserr": float(result.abserr),
                    "converged": bool(result.converged),
                    "analytic": analytic,
                    "levin_calls": [first, len(levin_calls)],
                }
            )
            return result

        return wrapped

    lq._adaptive_levin_subregion_impl = impl
    lq._Basis_SinCos.build_Levin_data = build
    lq._adaptive_levin = driver
    t3.quad_JJJ = wrap("quad_JJJ", "J")
    t3.quad_YJJ = wrap("quad_YJJ", "Y")

    random.seed(seed)
    suite = unittest.TestSuite([t3.Test3BesselAnalytic(test)])
    started = time.perf_counter()
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
    wall = time.perf_counter() - started
    outcome = "pass" if result.wasSuccessful() else "fail"
    failures = [
        str(t) + ": " + tb.splitlines()[-1] for t, tb in result.failures + result.errors
    ]
    Path(out_file).write_text(
        json.dumps(
            {
                "seed": seed,
                "test": test,
                "tree": str(tree),
                "module_sha256": sha256_of(lq.__file__),
                "outcome": outcome,
                "failures": failures,
                "wall": wall,
                "levin_calls": levin_calls,
                "bessel_results": bessel_results,
            },
            indent=1,
        )
    )


def git_sha(tree):
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True
    ).stdout.strip()


def git_dirty(tree, path):
    return subprocess.run(
        ["git", "status", "--porcelain", "--", path],
        cwd=tree,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_all(tree, label, jobs):
    tree = Path(tree).resolve()
    out_dir = OUT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    python = str(tree / "venv" / "bin" / "python")
    provenance = {
        "label": label,
        "tree": str(tree),
        "tree_git_sha": git_sha(tree),
        "levin_quadrature_sha256": sha256_of(
            tree / "AdaptiveLevin" / "levin_quadrature.py"
        ),
        "levin_quadrature_status": git_dirty(tree, "AdaptiveLevin/levin_quadrature.py"),
        "test_module_sha256": sha256_of(
            tree / "LiouvilleGreen" / "tests" / "test_3bessel_analytic.py"
        ),
        "script_sha256": sha256_of(__file__),
        "seeds": SEEDS,
        "tests": list(TESTS),
        "unit_timeout": UNIT_TIMEOUT,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    def one(seed, test):
        out_file = out_dir / f"seed{seed:02d}_{test}.json"
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                [
                    python,
                    str(Path(__file__).resolve()),
                    "unit",
                    "--tree",
                    str(tree),
                    "--seed",
                    str(seed),
                    "--test",
                    test,
                    "--out-file",
                    str(out_file),
                ],
                cwd=tree,
                env={**os.environ, "PYTHONPATH": str(tree)},
                capture_output=True,
                text=True,
                timeout=UNIT_TIMEOUT,
            )
            if proc.returncode != 0 or not out_file.exists():
                out_file.write_text(
                    json.dumps(
                        {
                            "seed": seed,
                            "test": test,
                            "outcome": "harness_error",
                            "stderr": proc.stderr[-4000:],
                        }
                    )
                )
        except subprocess.TimeoutExpired:
            out_file.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "test": test,
                        "outcome": "did_not_finish",
                        "wall": UNIT_TIMEOUT,
                    }
                )
            )
        return seed, test, time.perf_counter() - started

    units = [(seed, test) for test in TESTS for seed in SEEDS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for seed, test, wall in pool.map(lambda u: one(*u), units):
            print(f"{label} seed {seed} {test}: {wall:.1f} s", flush=True)
    provenance["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out_dir / "run.json").write_text(json.dumps(provenance, indent=1))


def load(label):
    units = {}
    for path in sorted((OUT / label).glob("seed*.json")):
        unit = json.loads(path.read_text())
        units[(unit["seed"], unit["test"])] = unit
    return units


def honest(result):
    return abs(result["value"] - result["analytic"]) <= result["abserr"]


# prompt 07d: the calls the report names. 07a's three LiouvilleGreen draw calls (07a's
# docs/levin-phase-floor/07a/code/out/lg_draws_changes.txt; 07d section 2.1 item 5) and 07c's 26 calls converged at
# 07c's parent and unconverged at 07c's commit (07c's out/lg_draws_report.json, "converged_changes").
CALLS_07A = [(3, "test_YJJ", 4), (8, "test_JJJ", 4), (17, "test_JJJ", 11)]
REPORT_07C = HERE.parents[1] / "07c" / "code" / "out" / "lg_draws_report.json"


def report():
    # prompt 07d: three trees, parent, commit and V93T10; the classes are by whether the fix acts at the commit
    parent, commit, var = load("parent"), load("commit"), load("V93T10")
    lines = []
    out = {"units": len(commit)}
    lines.append(
        "levin-phase-floor 07d Part B: test_3bessel_analytic.py, seeds 0-19, parent, commit and V93T10"
    )
    lines.append(f"  report script sha256 {sha256_of(__file__)}")
    for label in ("parent", "commit", "V93T10"):
        run = json.loads((OUT / label / "run.json").read_text())
        lines.append(
            f"  {run['label']}: tree {run['tree']} at {run['tree_git_sha']}, levin_quadrature.py sha256 "
            f"{run['levin_quadrature_sha256']} ({'modified' if run['levin_quadrature_status'] else 'as committed'}), "
            f"script {run['script_sha256']}"
        )
    lines.append("")

    # 1. test outcomes
    def outcomes(a, b, name):
        tally = {}
        newly = []
        for key in sorted(b):
            ao, bo = a[key]["outcome"], b[key]["outcome"]
            tally[(ao, bo)] = tally.get((ao, bo), 0) + 1
            if bo != "pass" and ao == "pass":
                newly.append((key, bo, b[key].get("failures")))
        lines.append(f"1{name}. Test outcomes, units:")
        for (ao, bo), n in sorted(tally.items()):
            lines.append(f"   {ao} -> {bo}: {n}")
        lines.append(f"   newly failing: {len(newly)}")
        for key, bo, failures in newly:
            lines.append(f"     seed {key[0]} {key[1]}: {bo} {failures}")
        return {f"{ao}->{bo}": n for (ao, bo), n in tally.items()}, [
            [k[0], k[1], bo] for k, bo, _ in newly
        ]

    out["outcomes_parent_commit"], out["new_failures_commit"] = outcomes(
        parent, commit, "a (parent -> commit)"
    )
    out["outcomes_parent_V93T10"], out["new_failures_V93T10"] = outcomes(
        parent, var, "b (parent -> V93T10)"
    )
    lines.append("")

    def pairs_of(a, b):
        pairs, unpaired = [], 0
        for key in sorted(b):
            ac = a[key].get("levin_calls")
            bc = b[key].get("levin_calls")
            if ac is None or bc is None or len(ac) != len(bc):
                unpaired += 1
                continue
            for x, y in zip(ac, bc):
                if x["x_span"] != y["x_span"] or x["rtol"] != y["rtol"]:
                    unpaired += 1
                    continue
                pairs.append((key, x, y))
        return pairs, unpaired

    def identical(p, c):
        return (
            p["value_hex"] == c["value_hex"]
            and p["abserr_hex"] == c["abserr_hex"]
            and p["regions_sha256"] == c["regions_sha256"]
            and p["result_sha256"] == c["result_sha256"]
        )

    def same_fields(p, c):
        # value, abserr, converged, num_regions and the maximum depth (section 9.4.5, Z1 and Z8)
        return all(
            p[k] == c[k]
            for k in (
                "value_hex",
                "abserr_hex",
                "converged",
                "num_regions",
                "max_depth",
            )
        )

    def counts(c):
        return (
            c["num_cc_solves"],
            c["num_cc_gate_route_changes"],
            c["num_cc_noise_set"],
            c["num_cc_resampled"],
        )

    def acts(c):
        return c["num_cc_solves"] > 0

    # 2. parent against commit
    pairs, unpaired = pairs_of(parent, commit)
    untouched = [x for x in pairs if not acts(x[2])]
    acted = [x for x in pairs if acts(x[2])]
    untouched_moved = [x for x in untouched if not identical(x[1], x[2])]
    lines.append(
        f"2. Levin calls paired by unit and index, parent against commit: {len(pairs)} (unpaired units or calls: {unpaired})"
    )
    lines.append(
        f"   the fix does not act (num_cc_solves 0): {len(untouched)}, of which not bit-identical to the parent "
        f"(value, abserr, every region, every result key but elapsed, location_skipped_phases and the counts): "
        f"{len(untouched_moved)}"
    )
    for key, p, c in untouched_moved[:20]:
        lines.append(
            f"     seed {key[0]} {key[1]} call {c['index']}: {p['value_hex']} -> {c['value_hex']}"
        )
    conv = {}
    worst = 0.0
    for _, p, c in acted:
        conv[(p["converged"], c["converged"])] = (
            conv.get((p["converged"], c["converged"]), 0) + 1
        )
        if p["abserr"] > 0:
            worst = max(worst, abs(c["value"] - p["value"]) / p["abserr"])
    past_sum = [
        x
        for x in acted
        if abs(x[2]["value"] - x[1]["value"]) > x[1]["abserr"] + x[2]["abserr"]
    ]
    lines.append(
        f"   the fix acts: {len(acted)} calls (Clenshaw-Curtis solves {sum(c['num_cc_solves'] for _, _, c in acted)}, "
        f"gate route changes {sum(c['num_cc_gate_route_changes'] for _, _, c in acted)} in "
        f"{sum(1 for _, _, c in acted if c['num_cc_gate_route_changes'] > 0)} calls, solves the noise term sets "
        f"{sum(c['num_cc_noise_set'] for _, _, c in acted)} in "
        f"{sum(1 for _, _, c in acted if c['num_cc_noise_set'] > 0)} calls, resampled "
        f"{sum(c['num_cc_resampled'] for _, _, c in acted)}); not bit-identical "
        f"{sum(1 for x in acted if not identical(x[1], x[2]))}; converged (parent, commit): "
        + ", ".join(f"{k}: {v}" for k, v in sorted(conv.items()))
        + f"; largest |v_commit - v_parent| / abserr_parent {worst:.3g}; moved past the sum of the two abserrs: "
        f"{len(past_sum)}"
    )
    for key, p, c in past_sum[:20]:
        lines.append(
            f"     past the sum: seed {key[0]} {key[1]} call {c['index']}: |dv| {abs(c['value'] - p['value']):.3g}, "
            f"abserr {p['abserr']:.3g} -> {c['abserr']:.3g}"
        )
    ratios = sorted(c["abserr"] / p["abserr"] for _, p, c in acted if p["abserr"] > 0)
    if ratios:
        lines.append(
            f"   abserr_commit / abserr_parent where the fix acts: min {ratios[0]:.3g}, median "
            f"{ratios[len(ratios) // 2]:.3g}, max {ratios[-1]:.3g}"
        )
    lines.append(
        f"   regions, all paired calls: parent {sum(p['num_regions'] for _, p, _ in pairs)}, commit "
        f"{sum(c['num_regions'] for _, _, c in pairs)}; where the fix acts: parent "
        f"{sum(p['num_regions'] for _, p, _ in acted)}, commit {sum(c['num_regions'] for _, _, c in acted)}; "
        f"max depth 20 in: parent {sum(1 for _, p, _ in pairs if p['max_depth'] >= 20)}, commit "
        f"{sum(1 for _, _, c in pairs if c['max_depth'] >= 20)}"
    )
    lines.append(
        "   chase stops (07c), all paired calls: parent "
        f"{sum(p['num_chase_stops'] for _, p, _ in pairs)}, commit {sum(c['num_chase_stops'] for _, _, c in pairs)}; "
        f"stopped leaves: parent {sum(p['num_chase_stopped_leaves'] for _, p, _ in pairs)}, commit "
        f"{sum(c['num_chase_stopped_leaves'] for _, _, c in pairs)}"
    )
    changed = [(key, p, c) for key, p, c in acted if p["converged"] != c["converged"]]
    lines.append(f"   converged changes ({len(changed)}):")
    for key, p, c in changed:
        lines.append(
            f"     converged {p['converged']} -> {c['converged']}: seed {key[0]} {key[1]} call {c['index']}: regions "
            f"{p['num_regions']} -> {c['num_regions']}, max depth {p['max_depth']} -> {c['max_depth']}, abserr "
            f"{p['abserr']:.3g} -> {c['abserr']:.3g}, request max(atol, rtol|v|) "
            f"{max(c['atol'] or 0.0, (c['rtol'] or 0.0) * abs(c['value'])):.3g}, counts {counts(c)}"
        )
    out.update(
        {
            "paired_calls": len(pairs),
            "unpaired_units_or_calls": unpaired,
            "untouched": len(untouched),
            "untouched_not_identical": len(untouched_moved),
            "acted": len(acted),
            "acted_counts": [
                sum(c[k] for _, _, c in acted)
                for k in (
                    "num_cc_solves",
                    "num_cc_gate_route_changes",
                    "num_cc_noise_set",
                    "num_cc_resampled",
                )
            ],
            "acted_converged": {f"{k[0]}->{k[1]}": v for k, v in conv.items()},
            "acted_largest_move_over_parent_abserr": worst,
            "acted_moved_past_sum": len(past_sum),
            "regions_parent": sum(p["num_regions"] for _, p, _ in pairs),
            "regions_commit": sum(c["num_regions"] for _, _, c in pairs),
            "converged_changes": [
                [k[0], k[1], c["index"], p["converged"], c["converged"]]
                for k, p, c in changed
            ],
        }
    )
    lines.append("")

    # 3. the named calls
    by_key = {(key[0], key[1], c["index"]): (p, c) for key, p, c in pairs}
    lines.append(
        "3. Named calls, (converged, regions, max depth, abserr) at the parent -> the commit, and the counts:"
    )

    def named(label, keys):
        rows = []
        lines.append(f"   {label}:")
        for k in keys:
            k = (int(k[0]), str(k[1]), int(k[2]))
            if k not in by_key:
                lines.append(f"     seed {k[0]} {k[1]} call {k[2]}: not paired")
                continue
            p, c = by_key[k]
            lines.append(
                f"     seed {k[0]} {k[1]} call {k[2]}: ({p['converged']}, {p['num_regions']}, {p['max_depth']}, "
                f"{p['abserr']:.3g}) -> ({c['converged']}, {c['num_regions']}, {c['max_depth']}, {c['abserr']:.3g}), "
                f"counts {counts(c)}"
            )
            rows.append(
                [k[0], k[1], k[2], p["converged"], c["converged"], counts(c)[0]]
            )
        return rows

    out["named_07a"] = named("07a's three draw calls", CALLS_07A)
    report_07c = json.loads(REPORT_07C.read_text())
    calls_07c = [x[:3] for x in report_07c["converged_changes"]]
    out["named_07c"] = named(
        f"07c's {len(calls_07c)} calls newly unconverged at 07c's commit", calls_07c
    )
    lines.append(
        f"   07c's {len(calls_07c)}: converged at the parent {sum(1 for r in out['named_07c'] if r[3])}, at the commit "
        f"{sum(1 for r in out['named_07c'] if r[4])}"
    )
    lines.append("")

    # 4. V93T10 against commit
    vpairs, vunpaired = pairs_of(var, commit)
    differ = [x for x in vpairs if not identical(x[1], x[2])]
    fdiffer = [x for x in vpairs if not same_fields(x[1], x[2])]
    cdiffer = [
        x
        for x in vpairs
        if (
            x[1]["num_cc_solves"],
            x[1]["num_cc_gate_route_changes"],
            x[1]["num_cc_resampled"],
        )
        != (
            x[2]["num_cc_solves"],
            x[2]["num_cc_gate_route_changes"],
            x[2]["num_cc_resampled"],
        )
    ]
    lines.append(
        f"4. Levin calls paired, V93T10 against commit: {len(vpairs)} (unpaired units or calls: {vunpaired}); "
        f"not bit-identical (value, abserr, every region, every result key but elapsed, location_skipped_phases and "
        f"the counts): {len(differ)}; differing in value, abserr, converged, num_regions or max depth: {len(fdiffer)}; "
        f"counts differing (num_cc_solves against num_a2_cc93, num_cc_gate_route_changes against "
        f"num_a2_route_changed, num_cc_resampled against num_a2_resampled): {len(cdiffer)}"
    )
    for key, p, c in (differ + cdiffer)[:20]:
        lines.append(
            f"     seed {key[0]} {key[1]} call {c['index']}: {p['value_hex']} -> {c['value_hex']}, counts {counts(p)} -> {counts(c)}"
        )
    lines.append(
        f"   the variant's counts over all paired calls: cc93 {sum(p['num_cc_solves'] for _, p, _ in vpairs)}, "
        f"route changed {sum(p['num_cc_gate_route_changes'] for _, p, _ in vpairs)}, resampled "
        f"{sum(p['num_cc_resampled'] for _, p, _ in vpairs)}"
    )
    out["V93T10_paired_calls"] = len(vpairs)
    out["V93T10_unpaired"] = vunpaired
    out["V93T10_not_identical"] = len(differ)
    out["V93T10_fields_differing"] = len(fdiffer)
    out["V93T10_counts_differing"] = len(cdiffer)
    lines.append("")

    # 5. honesty of the three-Bessel results, by whether the fix acts in any of their calls
    lines.append(
        "5. Three-Bessel results (|value - analytic| <= abserr), by whether the fix acts in any of their Levin calls:"
    )
    hon = {}
    for key in sorted(commit):
        pb, cb = parent[key].get("bessel_results"), commit[key].get("bessel_results")
        if not pb or not cb or len(pb) != len(cb):
            continue
        cc = commit[key]["levin_calls"]
        for p, c in zip(pb, cb):
            lo, hi = c["levin_calls"]
            kind = "acted" if any(acts(cl) for cl in cc[lo:hi]) else "untouched"
            row = hon.setdefault(
                kind,
                {
                    "results": 0,
                    "honest_parent": 0,
                    "honest_commit": 0,
                    "dishonest_converged_parent": 0,
                    "dishonest_converged_commit": 0,
                    "converged_parent": 0,
                    "converged_commit": 0,
                    "worst_commit": 0.0,
                    "worst_parent": 0.0,
                },
            )
            row["results"] += 1
            row["honest_parent"] += honest(p)
            row["honest_commit"] += honest(c)
            row["converged_parent"] += p["converged"]
            row["converged_commit"] += c["converged"]
            row["dishonest_converged_parent"] += (not honest(p)) and p["converged"]
            row["dishonest_converged_commit"] += (not honest(c)) and c["converged"]
            if c["abserr"] > 0:
                row["worst_commit"] = max(
                    row["worst_commit"], abs(c["value"] - c["analytic"]) / c["abserr"]
                )
            if p["abserr"] > 0:
                row["worst_parent"] = max(
                    row["worst_parent"], abs(p["value"] - p["analytic"]) / p["abserr"]
                )
    for kind, row in sorted(hon.items()):
        lines.append(
            f"   {kind}: {row['results']} results; honest parent {row['honest_parent']}, commit {row['honest_commit']}; "
            f"converged parent {row['converged_parent']}, commit {row['converged_commit']}; "
            f"dishonest and converged: parent {row['dishonest_converged_parent']}, commit "
            f"{row['dishonest_converged_commit']}; worst |err|/abserr parent {row['worst_parent']:.3g}, commit "
            f"{row['worst_commit']:.3g}"
        )
    out["honesty"] = hon
    lines.append("")
    lines.append(
        "6. Wall (s), summed over units: parent {:.1f}, commit {:.1f}, V93T10 {:.1f}".format(
            sum(u.get("wall", 0.0) for u in parent.values()),
            sum(u.get("wall", 0.0) for u in commit.values()),
            sum(u.get("wall", 0.0) for u in var.values()),
        )
    )
    (OUT / "lg_draws_report.txt").write_text("\n".join(lines) + "\n")
    (OUT / "lg_draws_report.json").write_text(
        json.dumps(out, indent=1, default=str) + "\n"
    )
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tree", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--jobs", type=int, default=6)
    u = sub.add_parser("unit")
    u.add_argument("--tree", required=True)
    u.add_argument("--seed", type=int, required=True)
    u.add_argument("--test", required=True)
    u.add_argument("--out-file", required=True)
    sub.add_parser("report")
    args = parser.parse_args()
    if args.action == "run":
        run_all(args.tree, args.label, args.jobs)
    elif args.action == "unit":
        run_unit(args.tree, args.seed, args.test, args.out_file)
    else:
        report()


if __name__ == "__main__":
    main()
