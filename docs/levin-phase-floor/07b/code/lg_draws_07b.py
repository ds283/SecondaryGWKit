"""
levin-phase-floor prompt 07b, Part B, section 3 item 5 ("The LiouvilleGreen draws"): 07a's harness,
docs/levin-phase-floor/07a/code/lg_draws_07a.py, copied, changed only where marked "prompt 07b".
LiouvilleGreen/tests/test_3bessel_analytic.py over seeds 0-19, under three trees:
- the parent: the dispatch commit c768c4ad (07a's code), in a temporary detached worktree outside the repository;
- the commit: the main checkout's tree with Part B's change;
- h: a second temporary detached worktree at c768c4ad with Part A's variant (h),
  docs/levin-phase-floor/07b/measurement/scripts/ext/variants/levin_quadrature_07b_h.py (SHA-256 47512022...), copied
  over its AdaptiveLevin/levin_quadrature.py, so that the commit is compared with (h) call by call (section 9.5, U8).
What it keeps of 07a's (and 07's before it):
- each unit is one (seed, test) in a process of its own, with a wall-clock limit (UNIT_TIMEOUT, 1200 s);
- the global ``random`` instance is seeded with the seed immediately before the test (the test module draws its
  wavenumbers from ``random.uniform`` unseeded, and is not edited);
- the tests are every test of Test3BesselAnalytic that runs by default: test_JJJ, test_YJJ,
  test_YJJ_log_singularity and test_abserr_bounds_truth; test_YJJ_log_scaling is skipped by default;
- every Levin call is recorded through a wrapper on AdaptiveLevin.levin_quadrature._adaptive_levin, the module
  attribute adaptive_levin_sincos() calls it through; and every three-Bessel result through wrappers on the test
  module's quad_JJJ and quad_YJJ, with its analytic value from the test's own oracle classes;
- 07a's route instrument on _Basis_SinCos.build_Levin_data is kept as it was: it changes no value, and on these trees
  (all 07a's routing) every count it records is zero.
What it changes (prompt 07b):
- each call records the two refusal counts, num_edge_rule_refusals and num_range_rule_refusals (absent at the parent
  and in (h), read as 0 there), and the digest of the whole result leaves them out, so that the three trees compare;
- the report classes a call by whether the rate rules act in it at the commit (the sum of its counts positive), checks
  that every call in which they do not is bit-identical to the parent's, tabulates the calls in which they do, and
  compares every call of the commit with (h)'s.

  run     --tree PATH --label LABEL [--jobs N]: every unit under that tree, one JSON file per unit in
          out/LABEL/, and out/LABEL/run.json with the provenance
  unit    (internal) one unit
  report  read out/parent/, out/commit/ and out/h/ and write out/lg_draws_report.txt and out/lg_draws_report.json

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
        try:
            data = original_driver(*args, **kwargs)
        finally:
            route["current"] = outer
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


def report():
    # prompt 07b: three trees; the classes are by whether the rate rules act at the commit
    parent, commit, hvar = load("parent"), load("commit"), load("h")
    lines = []
    out = {"units": len(commit)}
    lines.append(
        "levin-phase-floor 07b Part B: test_3bessel_analytic.py, seeds 0-19, parent, commit and (h)"
    )
    for label in ("parent", "commit", "h"):
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
    out["outcomes_parent_h"], out["new_failures_h"] = outcomes(
        parent, hvar, "b (parent -> h)"
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

    def acts(c):
        return c["num_edge_rule_refusals"] + c["num_range_rule_refusals"] > 0

    # 2. parent against commit
    pairs, unpaired = pairs_of(parent, commit)
    untouched = [x for x in pairs if not acts(x[2])]
    acted = [x for x in pairs if acts(x[2])]
    untouched_moved = [x for x in untouched if not identical(x[1], x[2])]
    lines.append(
        f"2. Levin calls paired by unit and index, parent against commit: {len(pairs)} (unpaired units or calls: {unpaired})"
    )
    lines.append(
        f"   the rate rules do not act (both counts 0): {len(untouched)}, of which not bit-identical to the parent "
        f"(value, abserr, every region, every result key but elapsed, location_skipped_phases and the counts): "
        f"{len(untouched_moved)}"
    )
    for key, p, c in untouched_moved[:20]:
        lines.append(
            f"     seed {key[0]} {key[1]} call {c['index']}: {p['value_hex']} -> {c['value_hex']}"
        )
    conv = {}
    worst = 0.0
    edge_calls = sum(1 for _, _, c in acted if c["num_edge_rule_refusals"] > 0)
    range_calls = sum(1 for _, _, c in acted if c["num_range_rule_refusals"] > 0)
    for _, p, c in acted:
        conv[(p["converged"], c["converged"])] = (
            conv.get((p["converged"], c["converged"]), 0) + 1
        )
        if p["abserr"] > 0:
            worst = max(worst, abs(c["value"] - p["value"]) / p["abserr"])
    lines.append(
        f"   the rate rules act: {len(acted)} calls ({edge_calls} with an edge-rule refusal, {range_calls} with a "
        f"range-rule refusal; refusals {sum(c['num_edge_rule_refusals'] for _, _, c in acted)} edge, "
        f"{sum(c['num_range_rule_refusals'] for _, _, c in acted)} range); not bit-identical "
        f"{sum(1 for x in acted if not identical(x[1], x[2]))}; converged (parent, commit): "
        + ", ".join(f"{k}: {v}" for k, v in sorted(conv.items()))
        + f"; largest |v_commit - v_parent| / abserr_parent {worst:.3g}"
    )
    lines.append(
        f"   regions: parent {sum(p['num_regions'] for _, p, _ in pairs)}, commit {sum(c['num_regions'] for _, _, c in pairs)}; "
        f"calls with a window: {sum(1 for _, _, c in pairs if c['num_windows'] > 0)}"
    )
    newly_unconverged = [
        (key, p, c) for key, p, c in acted if p["converged"] and not c["converged"]
    ]
    for key, p, c in newly_unconverged:
        lines.append(
            f"     converged -> unconverged: seed {key[0]} {key[1]} call {c['index']}: regions {p['num_regions']} -> "
            f"{c['num_regions']}, max depth {p['max_depth']} -> {c['max_depth']}, untrusted returned "
            f"{c['num_untrusted_regions']}, refusals edge {c['num_edge_rule_refusals']} range {c['num_range_rule_refusals']}"
        )
    out.update(
        {
            "paired_calls": len(pairs),
            "unpaired_units_or_calls": unpaired,
            "untouched": len(untouched),
            "untouched_not_identical": len(untouched_moved),
            "acted": len(acted),
            "acted_edge_calls": edge_calls,
            "acted_range_calls": range_calls,
            "acted_converged": {f"{k[0]}->{k[1]}": v for k, v in conv.items()},
            "acted_largest_move_over_parent_abserr": worst,
            "regions_parent": sum(p["num_regions"] for _, p, _ in pairs),
            "regions_commit": sum(c["num_regions"] for _, _, c in pairs),
            "newly_unconverged": [
                [k[0], k[1], c["index"]] for k, _, c in newly_unconverged
            ],
        }
    )
    lines.append("")

    # 3. (h) against commit
    hpairs, hunpaired = pairs_of(hvar, commit)
    differ = [x for x in hpairs if not identical(x[1], x[2])]
    lines.append(
        f"3. Levin calls paired, (h) against commit: {len(hpairs)} (unpaired units or calls: {hunpaired}); "
        f"not bit-identical: {len(differ)}"
    )
    for key, p, c in differ[:20]:
        lines.append(
            f"     seed {key[0]} {key[1]} call {c['index']}: {p['value_hex']} -> {c['value_hex']}"
        )
    out["h_paired_calls"] = len(hpairs)
    out["h_unpaired"] = hunpaired
    out["h_not_identical"] = len(differ)
    lines.append("")

    # 4. honesty of the three-Bessel results, by whether the rules act in any of their calls
    lines.append(
        "4. Three-Bessel results (|value - analytic| <= abserr), by whether the rate rules act in any of their Levin calls:"
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
                    "worst_commit": 0.0,
                    "worst_parent": 0.0,
                },
            )
            row["results"] += 1
            row["honest_parent"] += honest(p)
            row["honest_commit"] += honest(c)
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
            f"dishonest and converged: parent {row['dishonest_converged_parent']}, commit "
            f"{row['dishonest_converged_commit']}; worst |err|/abserr parent {row['worst_parent']:.3g}, commit "
            f"{row['worst_commit']:.3g}"
        )
    out["honesty"] = hon
    lines.append("")
    lines.append(
        "5. Wall (s), summed over units: parent {:.1f}, commit {:.1f}, h {:.1f}".format(
            sum(u.get("wall", 0.0) for u in parent.values()),
            sum(u.get("wall", 0.0) for u in commit.values()),
            sum(u.get("wall", 0.0) for u in hvar.values()),
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
