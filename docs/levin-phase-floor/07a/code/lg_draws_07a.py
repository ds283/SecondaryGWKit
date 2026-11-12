"""
levin-phase-floor prompt 07a, Part B, section 3 ("The LiouvilleGreen draws"): LiouvilleGreen/tests/
test_3bessel_analytic.py over seeds 0-19, under the parent (the dispatch commit a9f6abc3, 07's code, in a
temporary detached worktree outside the repository) and under the commit (the main checkout's tree with Part B's
change). After 07's harness (docs/levin-phase-floor/07/calibration/scripts/lg_seeded_draws.py, a record, not
edited), cut down to what section 3 asks for: no 06x variant modes, no trust-check recorder, no section 2.5
account. What it keeps of 07's:
- each unit is one (seed, test) in a process of its own, with a wall-clock limit (UNIT_TIMEOUT, 07's 1200 s);
- the global ``random`` instance is seeded with the seed immediately before the test (the test module draws its
  wavenumbers from ``random.uniform`` unseeded, and is not edited);
- the tests are every test of Test3BesselAnalytic that runs by default: test_JJJ, test_YJJ,
  test_YJJ_log_singularity and test_abserr_bounds_truth (a fixed triple, run at every seed all the same);
  test_YJJ_log_scaling is skipped by default and asserts nothing, and is not run;
- every Levin call is recorded through a wrapper on AdaptiveLevin.levin_quadrature._adaptive_levin, the module
  attribute adaptive_levin_sincos() calls it through; and every three-Bessel result through wrappers on the test
  module's quad_JJJ and quad_YJJ, with its analytic value from the test's own oracle classes.

What it adds, for section 3's table:
- **the route instrument.** A wrapper on _Basis_SinCos.build_Levin_data computes, beside the V it returns, the
  parent's estimate mean|theta'| * width from the same theta' on the same nodes, and records for every region
  solved by the gate (not force_direct; a wrapper on _adaptive_levin_subregion_impl marks those) whether the C2
  gate (< SIX_PI) or the LU choice (> 20 pi, for a region the Levin solve takes) would differ under the estimate,
  and whether V took the node-partition sum (theta' changing sign on the nodes). It returns exactly what the
  method returned and changes no value: under the parent, V is the estimate, and every count is zero;
- per call, the window and location results (points, windows with their iroff1 counts and stops, skipped phases,
  guarded leaves) and digests of the regions and of the whole result ("elapsed" set aside), so that the two codes
  compare bit for bit.

  run     --tree PATH --label LABEL [--jobs N]: every unit under that tree, one JSON file per unit in
          out/LABEL/, and out/LABEL/run.json with the provenance
  unit    (internal) one unit
  report  read out/parent/ and out/commit/ and write out/lg_draws_report.txt and out/lg_draws_report.json

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
                        if k not in ("elapsed", "regions", "location_skipped_phases")
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
    parent, commit = load("parent"), load("commit")
    lines = []
    out = {"units": len(commit)}
    p_run = json.loads((OUT / "parent" / "run.json").read_text())
    c_run = json.loads((OUT / "commit" / "run.json").read_text())
    lines.append(
        "levin-phase-floor 07a Part B: test_3bessel_analytic.py, seeds 0-19, parent against commit"
    )
    for run in (p_run, c_run):
        lines.append(
            f"  {run['label']}: tree {run['tree']} at {run['tree_git_sha']}, levin_quadrature.py sha256 "
            f"{run['levin_quadrature_sha256']} ({'modified' if run['levin_quadrature_status'] else 'as committed'}), "
            f"script {run['script_sha256']}"
        )
    lines.append("")

    # 1. test outcomes
    outcome_rows = []
    new_failures = []
    for key in sorted(commit):
        po, co = parent[key]["outcome"], commit[key]["outcome"]
        outcome_rows.append((key, po, co))
        if co != "pass" and po == "pass":
            new_failures.append((key, co, commit[key].get("failures")))
    tally = {}
    for _, po, co in outcome_rows:
        tally[(po, co)] = tally.get((po, co), 0) + 1
    lines.append("1. Test outcomes (parent -> commit), units:")
    for (po, co), n in sorted(tally.items()):
        lines.append(f"   {po} -> {co}: {n}")
    lines.append(
        f"   tests failing under the commit that pass under the parent: {len(new_failures)}"
    )
    for key, co, failures in new_failures:
        lines.append(f"     seed {key[0]} {key[1]}: {co} {failures}")
    out["outcomes"] = {f"{po}->{co}": n for (po, co), n in tally.items()}
    out["new_failures"] = [[k[0], k[1], co] for k, co, _ in new_failures]
    lines.append("")

    # 2. the Levin calls, paired
    pairs = []
    unpaired = 0
    for key in sorted(commit):
        pc = parent[key].get("levin_calls")
        cc = commit[key].get("levin_calls")
        if pc is None or cc is None or len(pc) != len(cc):
            unpaired += 1
            continue
        for p, c in zip(pc, cc):
            if p["x_span"] != c["x_span"] or p["rtol"] != c["rtol"]:
                unpaired += 1
                continue
            pairs.append((key, p, c))
    out["paired_calls"] = len(pairs)
    out["unpaired_units_or_calls"] = unpaired

    def identical(p, c):
        return (
            p["value_hex"] == c["value_hex"]
            and p["abserr_hex"] == c["abserr_hex"]
            and p["regions_sha256"] == c["regions_sha256"]
            and p["result_sha256"] == c["result_sha256"]
        )

    classes = {
        "route_or_lu_changed": [],
        "window": [],
        "newly_located": [],
        "neither": [],
    }
    for key, p, c in pairs:
        changed = c["route_changed"] + c["lu_changed"] > 0
        window = p["num_windows"] > 0 or c["num_windows"] > 0
        newly = bool(p["location_skipped_phases"]) and c["num_points"] > 0
        if changed:
            classes["route_or_lu_changed"].append((key, p, c))
        if window:
            classes["window"].append((key, p, c))
        if newly:
            classes["newly_located"].append((key, p, c))
        if not changed and not window:
            classes["neither"].append((key, p, c))
    neither_moved = [x for x in classes["neither"] if not identical(x[1], x[2])]
    lines.append(
        f"2. Levin calls paired by unit and index: {len(pairs)} (unpaired units or calls: {unpaired})"
    )
    lines.append(
        f"   with no window at either code and no region whose route or LU choice differs: {len(classes['neither'])}, "
        f"of which not bit-identical (value, abserr, every region, every result key but elapsed and "
        f"location_skipped_phases): {len(neither_moved)}"
    )
    for key, p, c in neither_moved[:20]:
        lines.append(
            f"     seed {key[0]} {key[1]} call {c['index']}: {p['value_hex']} -> {c['value_hex']}"
        )
    out["neither"] = len(classes["neither"])
    out["neither_not_identical"] = len(neither_moved)

    def moved_table(name, rows):
        moved = [r for r in rows if not identical(r[1], r[2])]
        conv = {}
        worst = 0.0
        for _, p, c in rows:
            conv[(p["converged"], c["converged"])] = (
                conv.get((p["converged"], c["converged"]), 0) + 1
            )
            if p["abserr"] > 0:
                worst = max(worst, abs(c["value"] - p["value"]) / p["abserr"])
        lines.append(
            f"   {name}: {len(rows)} calls, {len(moved)} not bit-identical; converged (parent, commit): "
            + ", ".join(f"{k}: {v}" for k, v in sorted(conv.items()))
            + f"; largest |v_commit - v_parent| / abserr_parent {worst:.3g}"
        )
        out[name] = {
            "calls": len(rows),
            "not_identical": len(moved),
            "converged": {f"{k[0]}->{k[1]}": v for k, v in conv.items()},
            "largest_move_over_parent_abserr": worst,
        }

    moved_table("route_or_lu_changed", classes["route_or_lu_changed"])
    moved_table("window", classes["window"])
    moved_table("newly_located", classes["newly_located"])
    total_changes = sum(c["route_changed"] for _, _, c in pairs)
    total_lu = sum(c["lu_changed"] for _, _, c in pairs)
    total_np = sum(c["node_partition"] for _, _, c in pairs)
    total_gated = sum(c["regions_gated"] for _, _, c in pairs)
    guarded = [
        (p["num_guarded_regions"], c["num_guarded_regions"]) for _, p, c in pairs
    ]
    lines.append(
        f"   regions gated under the commit: {total_gated}; route changed {total_changes}, LU choice changed "
        f"{total_lu}, V by the node-partition sum {total_np}"
    )
    lines.append(
        f"   calls with a guarded leaf: parent {sum(1 for g, _ in guarded if g)}, commit {sum(1 for _, g in guarded if g)}"
    )
    stopped = [
        (
            sum(w["stopped"] for w in p["windows"]),
            sum(w["stopped"] for w in c["windows"]),
        )
        for _, p, c in pairs
    ]
    lines.append(
        f"   windows stopped: parent {sum(s for s, _ in stopped)}, commit {sum(s for _, s in stopped)}"
    )
    out.update(
        {
            "regions_gated": total_gated,
            "route_changed_regions": total_changes,
            "lu_changed_regions": total_lu,
            "node_partition_regions": total_np,
            "calls_guarded_parent": sum(1 for g, _ in guarded if g),
            "calls_guarded_commit": sum(1 for _, g in guarded if g),
            "windows_stopped_parent": sum(s for s, _ in stopped),
            "windows_stopped_commit": sum(s for _, s in stopped),
        }
    )
    lines.append("")

    # 3. honesty of the three-Bessel results holding such calls
    lines.append(
        "3. Three-Bessel results (|value - analytic| <= abserr), by the Levin calls they hold:"
    )
    hon = {}
    for key in sorted(commit):
        pb, cb = parent[key].get("bessel_results"), commit[key].get("bessel_results")
        if not pb or not cb or len(pb) != len(cb):
            continue
        cc = commit[key]["levin_calls"]
        pc = parent[key]["levin_calls"]
        for p, c in zip(pb, cb):
            lo, hi = c["levin_calls"]
            calls = list(zip(pc[lo:hi], cc[lo:hi]))
            changed = any(cl["route_changed"] + cl["lu_changed"] > 0 for _, cl in calls)
            window = any(
                pl["num_windows"] > 0 or cl["num_windows"] > 0 for pl, cl in calls
            )
            kind = (
                "route_or_lu_changed"
                if changed
                else ("window" if window else "neither")
            )
            row = hon.setdefault(
                kind,
                {
                    "results": 0,
                    "honest_parent": 0,
                    "honest_commit": 0,
                    "dishonest_converged_commit": 0,
                    "worst_commit": 0.0,
                    "worst_parent": 0.0,
                },
            )
            row["results"] += 1
            row["honest_parent"] += honest(p)
            row["honest_commit"] += honest(c)
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
            f"dishonest and converged under the commit {row['dishonest_converged_commit']}; worst |err|/abserr "
            f"parent {row['worst_parent']:.3g}, commit {row['worst_commit']:.3g}"
        )
    out["honesty"] = hon
    lines.append("")
    lines.append(
        "4. Wall (s), summed over units: parent {:.1f}, commit {:.1f}".format(
            sum(u.get("wall", 0.0) for u in parent.values()),
            sum(u.get("wall", 0.0) for u in commit.values()),
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
