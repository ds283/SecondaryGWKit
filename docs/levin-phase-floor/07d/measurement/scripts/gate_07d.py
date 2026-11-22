"""levin-phase-floor prompt 07d Part A: the gate's and the brief's facts, re-checked read-only at the start (and
re-run at the end, as the scope check). New in 07d.

Prints: the worktree's HEAD and ``status --short --ignored``; the main checkout's HEAD and status; the code diff
``67b5af74..HEAD -- AdaptiveLevin ComputeTargets LiouvilleGreen tools``; the module's SHA-256 and line count and the
line of each name the orchestrator note's section A.0 lists; ``CHEBYSHEV_ORDER``; every code (non-comment, non-docstring)
reader of ``SIX_PI`` in the module (by ``ast``); the registry; the lock and Ray processes of /tmp/ray-sgk; load and disk;
SHA-256 of 07c's census checkpoint and ``compare_census_calls.calls.jsonl`` against 07c's census KEPT_ON_MAC_STUDIO.txt;
the scratch copy's five files against ``copy_files_certified.txt``.

Usage: PYTHONPATH=<tree>:<scripts> python -B gate_07d.py [--quick]   (--quick skips the 3.5 GB checkpoint)
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import COPY_DIR, H_TREE, MAIN, REC07C, RUN07C, sha256  # noqa: E402

PY = str(MAIN / "venv" / "bin" / "python")
NAMES = ["SIX_PI", "_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO", "_LEVIN_DIRECT_SOLVE_PHASE_SPAN", "_LEVIN_CHASE_NOISE_FACTOR",
         "_LEVIN_WINDOW_IROFF1_LIMIT", "_DQK21_SATURATION_FACTOR", "_levin_G0_G1", "_roundoff_floor", "_cc_node_term_parts",
         "_cc_roundoff_floor", "_declared_endpoint_phase_err", "_cc_leaf_unresolved", "_window_bisection_counts",
         "theta_prime_on_fine_grid", "_adaptive_levin_subregion", "_adaptive_levin_subregion_impl",
         "_adaptive_levin_subregion_cc", "_oriented_cc", "_locate_stationary_points", "_adaptive_levin", "_add_leaf",
         "_check_leaf", "_floor_limit_stopped_window_leaf", "_stop_window", "adaptive_levin_sincos"]


def run(args, cwd=None):
    r = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    return (r.stdout + r.stderr).rstrip()


def main():
    quick = "--quick" in sys.argv
    print("## trees")
    print("worktree HEAD", run(["git", "--no-optional-locks", "-C", str(H_TREE), "rev-parse", "HEAD"]))
    st = run(["git", "--no-optional-locks", "-C", str(H_TREE), "status", "--short", "--ignored"])
    print("worktree status --short --ignored:", "empty" if not st else repr(st))
    print("main HEAD", run(["git", "--no-optional-locks", "-C", str(MAIN), "rev-parse", "HEAD"]))
    st = run(["git", "--no-optional-locks", "-C", str(MAIN), "status", "--short"])
    print("main status --short:", "empty" if not st else repr(st))
    d = run(["git", "--no-optional-locks", "-C", str(H_TREE), "diff", "--stat", "67b5af74", "HEAD", "--",
             "AdaptiveLevin", "ComputeTargets", "LiouvilleGreen", "tools"])
    print("code diff 67b5af74..HEAD:", "empty" if not d else repr(d))
    print("## the module")
    lq = H_TREE / "AdaptiveLevin" / "levin_quadrature.py"
    src = lq.read_text()
    print("levin_quadrature.py sha256", sha256(lq), "lines", len(src.splitlines()))
    tree = ast.parse(src)
    where = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            where.setdefault(node.name, node.lineno)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    where.setdefault(t.id, node.lineno)
    for n in NAMES:
        print(f"  {n}: line {where.get(n)}")
    readers = []
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child._parent = node
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "SIX_PI" and isinstance(node.ctx, ast.Load):
            p = node
            fn = None
            while hasattr(p, "_parent"):
                p = p._parent
                if isinstance(p, ast.FunctionDef):
                    fn = p.name
                    break
            readers.append((node.lineno, fn))
    print("code readers of SIX_PI (ast Load):", readers)
    qsi = (H_TREE / "ComputeTargets" / "QuadSourceIntegral.py").read_text().splitlines()
    print("QuadSourceIntegral CHEBYSHEV_ORDER:", [(i + 1, l.strip()) for i, l in enumerate(qsi) if l.startswith("CHEBYSHEV_ORDER")])
    print("## the machine")
    print(run([PY, "-B", "-m", "runregistrykit", "list"], cwd=str(MAIN)).splitlines()[-1])
    print(run([PY, "-B", "-m", "raypoolkit.rehearsal_lock", "status"]))
    print(run([PY, "-B", "-m", "raypoolkit.rehearsal_lock", "ray-processes", "--ray-temp-dir", "/tmp/ray-sgk"]))
    print(run(["uptime"]))
    print(run(["df", "-h", str(MAIN)]).splitlines()[-1])
    print("## inputs")
    kept = {}
    for line in (REC07C / "KEPT_ON_MAC_STUDIO.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and len(parts[0]) == 64:
            kept[parts[2]] = (parts[0], int(parts[1]))
    checks = [(RUN07C / "checkpoint.jsonl", "levin-phase-floor-07c-census-20261119T201419/checkpoint.jsonl"),
              (REC07C / "out" / "stop3" / "compare_census_calls.calls.jsonl", "out/stop3/compare_census_calls.calls.jsonl")]
    for p, key in checks:
        if quick and p.name == "checkpoint.jsonl":
            print(f"{p}: size {p.stat().st_size} (hash skipped, --quick)")
            continue
        want = kept[key]
        got = sha256(p)
        print(f"{p}: sha256 {got} size {p.stat().st_size}; listed {want[0]} {want[1]}: {'MATCH' if got == want[0] and p.stat().st_size == want[1] else 'DIFFER'}")
    cert = (COPY_DIR.parent / "copy_files_certified.txt").read_text().splitlines()
    for line in cert:
        if line.startswith("#"):
            continue
        parts = line.split()
        p = COPY_DIR / parts[-1]
        got = sha256(p)
        print(f"scratch copy {parts[-1]}: {'MATCH' if got == parts[0] and p.stat().st_size == int(parts[1]) else 'DIFFER'} {got}")


if __name__ == "__main__":
    main()
