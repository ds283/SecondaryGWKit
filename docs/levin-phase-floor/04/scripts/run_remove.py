"""
levin-phase-floor prompt 04, step 7: the registered run. Re-run prompt 03's seven depth-20 rows of
A3 v2 in-process (route (a): scripts/harness.py, scripts/rebuild.py; no Ray, no store write) under
the arms of prompt section 2, in prompt section 3's units and order, one ledger record each:

    1. P at rtol 1e-7, R1..R7   (M0: bit-identical to the stored row; a failure stops the run)
    2. X at rtol 1e-7, R1..R7
    3. A at rtol 1e-7, R1..R7
    4. X at rtol 1e-8, R1..R7

all at atol 1e-32 (the stored row's tolerance objects; the 1e-8 rung uses the store's own
``tolerance`` row of 1e-8, read through the read-only pool).

**How an arm is installed.** ``ComputeTargets.QuadSourceIntegral.evaluate_QuadSource_integral`` is
replaced at its module attribute by a wrapper that enters ``phases.install(arm)`` (the dispatcher
patched at ``PrimitivePhase.theta_mod_2pi``), calls the original, and leaves in a ``finally``. So
the patch is in place around each unit's ``evaluate_QuadSource_integral`` call only. Prompt 03's
transparent wrapper on ``adaptive_levin_sincos`` keeps each Levin call's result. Every patched
attribute is restored in a ``finally``.

**The budget.** An X or A unit may run for 3 x its row's P unit compute wall time. It is enforced
by the dispatcher itself (``phases.Dispatcher``): every ``theta_mod_2pi`` call past the deadline
raises ``OverBudget`` (a ``BaseException``, so that no ``except Exception`` in production swallows
it) before computing anything. This is the prompt's "or equivalent" of ``setitimer``. A first
version raised from a SIGALRM handler; the smoke test showed that the handler can fire inside
mpmath's ``from_float``, whose bare ``except:`` swallows the exception and returns NaN. An over-budget unit is recorded
``over budget`` with its elapsed time, and the Levin calls it had completed; the run continues.

**What a ledger record holds** (``data``): the arm, rtol, row; wall time, budget and the over-budget
flag; the dispatcher's call counts by kind; the row's result fields (floats exactly, as JSON reprs
and as hex); the stored metadata's sub-interval and group records; for each Levin group call the
group label parsed from notify_label, value, abserr, converged, phase_limited, the region count, the
depth histogram split by type, the count at depth >= 18, evaluations and elapsed; for P, M0's verdict
and lines; and the provenance prompt section 8 asks for: the worktree's HEAD and its
``git status --short``, and the SHA-256 of every script this run imports.

**What is pickled** (in the run directory, X and A units only): ``unit_<id>.pkl`` with the result
dict and, per Levin call, the region lists as numpy arrays (start, end, depth, type code 0 = Levin
/ 1 = Clenshaw-Curtis, abserr). P's region lists are not pickled again (03's pickles hold them).

A heartbeat thread refreshes the registry heartbeat every 60 s while the process lives, because a
unit can run for longer than the registry's 15-minute staleness window.
"""

import argparse
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import ray  # noqa: E402

from harness import (  # noqa: E402
    QSI_MOD,
    close_pool,
    dump_record,
    open_readonly_pool,
    standin_ray,
)
from prov import WORKTREE, sha256  # noqa: E402
from run_instrument import ROWS, m0, parse_label  # noqa: E402

import phases as PH  # noqa: E402

ROW_NAMES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
UNIT_PLAN = [("P", "1e-7"), ("X", "1e-7"), ("A", "1e-7"), ("X", "1e-8")]
BUDGET_FACTOR = 3.0
SCRIPTS = [
    "run_remove.py",
    "phases.py",
    "harness.py",
    "rebuild.py",
    "run_instrument.py",
    "prov.py",
]


class OverBudget(BaseException):
    pass


def worktree_state():
    head = subprocess.run(
        ["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(WORKTREE), "status", "--short"], capture_output=True, text=True
    ).stdout
    return head, status


def fx(x):
    """A float, exactly: repr (round-trips) and hex."""
    if x is None:
        return None
    x = float(x)
    return {"repr": repr(x), "hex": x.hex()}


def summarise_call(call):
    res = call["result"]
    regs = res["regions"]
    hist = {}
    n18 = 0
    maxd = 0
    for r in regs:
        h = hist.setdefault(str(r.depth), [0, 0])
        h[0 if r.type == "Levin" else 1] += 1
        if r.depth >= 18:
            n18 += 1
        maxd = max(maxd, r.depth)
    p = call["parsed"] or {}
    return {
        "index": call["index"],
        "notify_label": call["notify_label"],
        "regime": p.get("regime"),
        "group": p.get("group"),
        "x_span": [fx(call["x_span"][0]), fx(call["x_span"][1])],
        "rtol": call["rtol"],
        "atol": call["atol"],
        "value": fx(res["value"]),
        "abserr": fx(res["abserr"]),
        "converged": bool(res["converged"]),
        "phase_limited": bool(res["phase_limited"]),
        "num_regions": int(res["num_regions"]),
        "num_simple_regions": int(res["num_simple_regions"]),
        "evaluations": int(res["evaluations"]),
        "elapsed": float(res["elapsed"]),
        "depth_hist_levin_cc": hist,
        "n_depth_ge_18": n18,
        "max_depth": maxd,
    }


def region_arrays(call):
    regs = call["result"]["regions"]
    return {
        "start": np.array([r.start for r in regs], dtype=float),
        "end": np.array([r.end for r in regs], dtype=float),
        "depth": np.array([r.depth for r in regs], dtype=np.int16),
        "type": np.array([0 if r.type == "Levin" else 1 for r in regs], dtype=np.int8),
        "abserr": np.array([r.abserr for r in regs], dtype=float),
    }


def summarise_result(result):
    d = result["WKB_Levin_data"]
    md = result["metadata"]
    return {
        "total": fx(result["total"]),
        "total_abserr": fx(result["total_abserr"]),
        "total_converged": bool(result["total_converged"]),
        "total_phase_limited": bool(result["total_phase_limited"]),
        "numeric_quad": fx(result["numeric_quad"]),
        "WKB_Levin": fx(result["WKB_Levin"]),
        "analytic_rad": fx(result["analytic_rad"]),
        "b": fx(result["b"]),
        "compute_time": float(result["compute_time"]),
        "WKB_Levin_data": (
            None
            if d is None
            else {
                "max_depth": int(d.max_depth),
                "num_regions": int(d.num_regions),
                "num_simple_regions": int(d.num_simple_regions),
                "evaluations": int(d.evaluations),
            }
        ),
        "WKB_Levin_subintervals": json.loads(
            json.dumps(md["WKB_Levin"]["subintervals"], default=float)
        ),
        "numeric_quad_subintervals": json.loads(
            json.dumps(md["numeric_quad"]["subintervals"], default=float)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--out-dir")
    g.add_argument("--run-dir")
    parser.add_argument(
        "--units",
        default=None,
        help="smoke test only: comma-separated unit ids (e.g. P-1e-7-R6,X-1e-7-R6)",
    )
    parser.add_argument(
        "--budget-override",
        type=float,
        default=None,
        help="smoke test only: a fixed budget in seconds for X/A units",
    )
    args = parser.parse_args()

    from rebuild import MAIN_PY_SHA256, PRIMARY, build_context, rebuild_row

    units = [(arm, rt, row) for arm, rt in UNIT_PLAN for row in ROW_NAMES]
    if args.units:
        want = args.units.split(",")
        units = [u for u in units if f"{u[0]}-{u[1]}-{u[2]}" in want]

    run = None
    if args.run_dir is not None:
        if args.units or args.budget_override is not None:
            raise SystemExit("--units/--budget-override are for the smoke test only")
        import runregistrykit

        run = runregistrykit.Run(args.run_dir)
        out_dir = Path(run.path)

        def terminal(signum, _frame):
            run.finish("killed", exit_code=128 + signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, terminal)
        run.heartbeat(pid=os.getpid(), stage="started: building the context")

        stop_beat = threading.Event()

        def beat():
            while not stop_beat.wait(60.0):
                try:
                    run.heartbeat()
                except Exception:
                    pass

        threading.Thread(target=beat, daemon=True).start()
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    wt_head, wt_status = worktree_state()
    script_hashes = {name: sha256(HERE / name) for name in SCRIPTS}
    header = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": script_hashes["run_remove.py"],
        "scripts_sha256": script_hashes,
        "main_py_sha256": MAIN_PY_SHA256,
        "worktree": str(WORKTREE),
        "worktree_head": wt_head,
        "worktree_status_short": wt_status,
        "host": socket.gethostname(),
        "python": sys.version,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "units": [f"{a}-{r}-{w}" for a, r, w in units],
        "budget_factor": BUDGET_FACTOR,
        "budget_override": args.budget_override,
    }
    print("# run_remove header " + json.dumps(header, default=str), flush=True)
    if wt_head != "98e68d3e1b60e6de4378b7d9aded3a120d6088a9" or wt_status.strip():
        raise SystemExit(f"worktree not at 98e68d3 or not clean: {wt_head} {wt_status!r}")

    orig_levin = QSI_MOD.adaptive_levin_sincos
    orig_eval = QSI_MOD.evaluate_QuadSource_integral
    holder = {"calls": None, "arm": None, "disp": None, "deadline": None}

    def w_levin(*a, **kw):
        result = orig_levin(*a, **kw)
        calls = holder["calls"]
        if calls is not None:
            x_span = a[0] if len(a) > 0 else kw.get("x_span")
            calls.append(
                {
                    "index": len(calls),
                    "notify_label": kw.get("notify_label"),
                    "parsed": parse_label(kw.get("notify_label")),
                    "x_span": tuple(x_span),
                    "atol": kw.get("atol"),
                    "rtol": kw.get("rtol"),
                    "result": result,
                }
            )
        return result

    def w_eval(*a, **kw):
        with PH.install(
            holder["arm"], deadline=holder["deadline"], stop_exc=OverBudget
        ) as disp:
            holder["disp"] = disp
            return orig_eval(*a, **kw)

    p_wall = {}
    all_ok = True
    try:
        with standin_ray() as cluster:
            pool, printed = open_readonly_pool(PRIMARY, cluster)
            print(printed.strip(), flush=True)
            try:
                ctx = build_context(pool, log=lambda s: print(s, flush=True))
                rtol8 = ray.get(pool.object_get("tolerance", tol=1e-8))
                if not (rtol8.available and rtol8.tol == 1e-8):
                    raise RuntimeError(f"tolerance 1e-8 not available: {rtol8}")
                rtol_obj = {"1e-7": None, "1e-8": rtol8}
                print(
                    f"** context built; checks {ctx['checks']}; rtol 1e-7 store_id {ctx['quad_rtol'].store_id} "
                    f"tol {ctx['quad_rtol'].tol!r}; rtol 1e-8 store_id {rtol8.store_id} tol {rtol8.tol!r}",
                    flush=True,
                )
                for arm, rt, name in units:
                    uid = f"{arm}-{rt}-{name}"
                    shard, serial = ROWS[name]
                    if run is not None:
                        run.heartbeat(stage=f"unit {uid} starting")
                    print(f"\n== unit {uid} (shard {shard}, serial {serial})", flush=True)
                    rb = rebuild_row(
                        ctx,
                        pool,
                        shard,
                        serial,
                        log=lambda s: print(s, flush=True),
                        rtol=rtol_obj[rt],
                    )
                    budget = None
                    if arm != "P":
                        budget = (
                            args.budget_override
                            if args.budget_override is not None
                            else BUDGET_FACTOR * p_wall[name]
                        )
                    calls = []
                    holder.update(calls=calls, arm=arm, disp=None)
                    QSI_MOD.adaptive_levin_sincos = w_levin
                    QSI_MOD.evaluate_QuadSource_integral = w_eval
                    over = False
                    result = None
                    t0 = time.perf_counter()
                    holder["deadline"] = None if budget is None else t0 + budget
                    try:
                        ref = rb["work"].compute(payload=rb["compute_payload"])
                        result = ray.get(ref)
                    except OverBudget:
                        over = True
                    finally:
                        wall = time.perf_counter() - t0
                        holder.update(calls=None, arm=None, deadline=None)
                        QSI_MOD.adaptive_levin_sincos = orig_levin
                        QSI_MOD.evaluate_QuadSource_integral = orig_eval
                    assert QSI_MOD.adaptive_levin_sincos is orig_levin
                    assert QSI_MOD.evaluate_QuadSource_integral is orig_eval
                    assert PH.PrimitivePhase.__dict__["theta_mod_2pi"] is PH.ORIG_THETA_MOD_2PI
                    disp = holder["disp"]
                    counts = dict(disp.counts) if disp is not None else None
                    if over:
                        result = None
                    group_calls = [c for c in calls if (c["parsed"] or {}).get("kind") == "group"]
                    wt_head_now, wt_status_now = worktree_state()
                    data = {
                        "unit": uid,
                        "arm": arm,
                        "rtol": rt,
                        "row": name,
                        "shard": shard,
                        "serial": serial,
                        "status": "over budget" if over else "done",
                        "wall_s": wall,
                        "budget_s": budget,
                        "dispatcher_counts": counts,
                        "worktree_head": wt_head_now,
                        "worktree_status_short": wt_status_now,
                        "scripts_sha256": script_hashes,
                        "stored_compute_time": rb["stored"].compute_time,
                        "calls": [summarise_call(c) for c in group_calls],
                        "n_calls_total": len(calls),
                    }
                    if result is not None:
                        data["result"] = summarise_result(result)
                    lines = []
                    if arm == "P":
                        ok = m0(result, rb["stored"], lines)
                        data["m0_holds"] = bool(ok)
                        data["m0_lines"] = lines
                        p_wall[name] = wall
                    print(
                        f"unit {uid}: {data['status']}; wall {wall:.1f} s; budget {budget}; counts {counts}; "
                        + (
                            f"total {result['total']!r} abserr {result['total_abserr']!r} regions {result['WKB_Levin_data'].num_regions} max depth {result['WKB_Levin_data'].max_depth}"
                            if result is not None
                            else f"{len(group_calls)} group calls completed"
                        )
                        + (f"; M0 {'HOLDS' if data['m0_holds'] else 'FAILS'}" if arm == "P" else ""),
                        flush=True,
                    )
                    for line in lines:
                        if "DIFFER" in line:
                            print(line, flush=True)
                    if arm != "P":
                        pkl = out_dir / f"unit_{uid}.pkl"
                        dump_record(
                            {
                                "unit": uid,
                                "header": header,
                                "data": data,
                                "result": result,
                                "regions": [
                                    {
                                        "index": c["index"],
                                        "regime": c["parsed"]["regime"],
                                        "group": c["parsed"]["group"],
                                        "x_span": c["x_span"],
                                        **region_arrays(c),
                                    }
                                    for c in group_calls
                                ],
                            },
                            pkl,
                        )
                        data["pickle"] = pkl.name
                        data["pickle_sha256"] = sha256(pkl)
                    if run is not None:
                        run.record(uid, data)
                        run.heartbeat(stage=f"unit {uid} {data['status']}")
                    else:
                        with open(out_dir / "ledger.jsonl", "a") as f:
                            f.write(json.dumps({"unit": uid, "data": data}) + "\n")
                    if arm == "P" and not data["m0_holds"]:
                        all_ok = False
                        print(f"!! STOP: M0 fails on {name}; no further units are run", flush=True)
                        break
            finally:
                print(close_pool(pool, cluster).strip(), flush=True)
    except BaseException:
        traceback.print_exc()
        if run is not None:
            run.finish("failed", exit_code=1)
        raise
    finally:
        QSI_MOD.adaptive_levin_sincos = orig_levin
        QSI_MOD.evaluate_QuadSource_integral = orig_eval
    if ray.is_initialized():
        all_ok = False
        print("!! Ray was initialised", flush=True)
    print(f"\n== run ends; all M0 hold and no stop: {all_ok}", flush=True)
    if run is not None:
        run.finish("done" if all_ok else "failed", exit_code=0 if all_ok else 2)
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
