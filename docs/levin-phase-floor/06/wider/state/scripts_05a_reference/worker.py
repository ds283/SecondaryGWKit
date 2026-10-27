"""
levin-phase-floor prompt 05a section 6 (copied from prompt 05's scripts/worker.py,
docs/levin-phase-floor/05/wider/scripts/worker.py, sha256 e2ef68f0..., a record that is not
edited). Changed here, each change marked "prompt 05a": the trees are ``c`` (the worktree at
``c4eaefd``, 05's c) and ``t`` (the main checkout, the code commit ``1fc349c``'s code) in place of
05's ``p`` and ``c``; the module-provenance evidence also names ``ComputeTargets.TkSourceFunctions``
and whether its constructor has ``exact_remainder``; and, for the phase probe (prompt 05a section 6
with section 9; probe.py), ``build_partition`` is wrapped at its module attribute
(``ComputeTargets.QuadSourceIntegral.build_partition``, through which
``evaluate_QuadSource_integral`` calls it) to keep the sub-intervals and the two transfer-function
objects the unit built, every unit records each factor's ``phase.exact_remainder``, and a unit
whose spec has ``"probe": true`` (the rtol 1e-7 units) runs probe.py on them after its timed
section and records the windows (summaries in the checkpoint, samples in the pickle). The probe
reads phase objects only; it runs after ``wall_s`` is measured, outside the budget, with the
dispatcher uninstalled, so it changes neither the unit's result nor its counts.

(Prompt 05's text follows, unchanged.)

levin-phase-floor prompt 05 section 6: the worker. Runs a list of units, all in one tree, in this
process, by route (a) (harness.py, rebuild.py: a read-only stand-in ShardedPool, no Ray, no store
write), and emits one JSON line per unit.

Adapted from prompt 04's ``scripts/run_remove.py`` (docs/levin-phase-floor/04/scripts/
run_remove.py, sha256 3c43db3f..., a record that is not edited): its unit loop, its wrapper on
``adaptive_levin_sincos``, its per-call summary and region arrays, its result summary and its
budget. What differs:
- **One tree per process.** The tree is the one on ``PYTHONPATH``: the parent ``e648951`` (the
  worktree) for ``p``, the code commit (the main checkout) for ``c``. The worker checks at start
  that ``ComputeTargets``, ``ComputeTargets.primitive_phase``, ``ComputeTargets.QuadSource``,
  ``StoreFixtures`` and ``main.py`` (through ``test_main_plumbing.MAIN_PY``) all come from that
  tree, and that the tree's ``HEAD`` is the expected SHA with an empty ``git status --short``, and
  it emits that evidence first. The registered driver (run_wider.py) starts one worker per tree.
- **No arms.** The phase is production's in each tree; dispatch.py's dispatcher only counts calls
  and enforces the budget, raising ``OverBudget`` at the entry of the first ``theta_mod_2pi`` call
  past the deadline (04's deviation 10 and I9). An over-budget unit is a result: it is recorded
  ``over budget`` with its wall time and the group calls it completed.
- **No M0.** Under ``p`` the row's ``QuadSource`` is recomputed (rebuild.py), so ``p`` is not
  expected to equal the stored row. The stored row's own fields are recorded beside the result.
- An exception in a unit other than ``OverBudget`` is recorded as ``error`` with its traceback, and
  the next unit runs.
- **Every unit record** carries the tree's ``HEAD`` and ``git status --short`` taken at the end of
  the unit, and the SHA-256 of every script in this directory that the run uses.

Usage: python -B worker.py SPEC_JSON  (``PYTHONPATH`` the tree; see run_wider.py and the L0 runner)
SPEC: {"tree": "p"|"c", "units": [{"uid", "tree", "rtol", "row", "shard", "serial", "role",
       "budget"}...], "out_dir": DIR, "emit_fd": N or null, "emit_file": PATH or null,
       "pickle_regions": true}
"""

import inspect
import json
import os
import re
import socket
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import ray  # noqa: E402

from prov import TREES, sha256, tree_state  # noqa: E402

SCRIPTS = [
    "worker.py",
    "dispatch.py",
    "harness.py",
    "rebuild.py",
    "prov.py",
    "run_wider.py",
    "launch.py",
    "probe.py",  # prompt 05a
]


class OverBudget(BaseException):
    pass


# 03's run_instrument.py LABEL_RE and parse_label (docs/levin-phase-floor/04/scripts/
# run_instrument.py:81-114), unchanged
LABEL_RE = re.compile(
    r"^k=(?P<k>[^/]+)/Mpc, q=(?P<q>[^/]+)/Mpc, r=(?P<r>[^/]+)/Mpc @ z_response=(?P<zr>[^,]+), "
    r"z in \((?P<zmax>[^,]+), (?P<zmin>[^)]+)\), regime (?P<regime>\S+), group (?P<group>\S+)$"
)


def parse_label(notify_label):
    if notify_label is None:
        return None
    m = LABEL_RE.match(notify_label)
    if m is None:
        return {"kind": "other", "notify_label": notify_label}
    d = m.groupdict()
    d["kind"] = "group"
    return d


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


def _int(x):
    return None if x is None else int(x)


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
                "max_depth": _int(d.max_depth),
                "num_regions": _int(d.num_regions),
                "num_simple_regions": _int(d.num_simple_regions),
                "evaluations": _int(d.evaluations),
            }
        ),
        "WKB_Levin_subintervals": json.loads(
            json.dumps(md.get("WKB_Levin", {}).get("subintervals"), default=float)
        ),
        "numeric_quad_subintervals": json.loads(
            json.dumps(md.get("numeric_quad", {}).get("subintervals"), default=float)
        ),
        "partition": json.loads(json.dumps(md.get("partition"), default=float)),
    }


def summarise_stored(stored):
    d = stored.WKB_Levin_data
    return {
        "total": fx(stored.total),
        "total_abserr": fx(stored.total_abserr),
        "total_converged": bool(stored.total_converged),
        "WKB_Levin": fx(stored.WKB_Levin),
        "numeric_quad": fx(stored.numeric_quad),
        "compute_time": float(stored.compute_time),
        "WKB_Levin_data": (
            None
            if d is None
            else {"max_depth": _int(d.max_depth), "num_regions": _int(d.num_regions)}
        ),
    }


def tree_evidence(tree_name):
    """Where this process's packages come from, and the tree's state."""
    import importlib

    import ComputeTargets
    import StoreFixtures

    # the modules, not the classes ComputeTargets/__init__.py re-exports under the same names
    pp = importlib.import_module("ComputeTargets.primitive_phase")
    qs = importlib.import_module("ComputeTargets.QuadSource")
    qsi = importlib.import_module("ComputeTargets.QuadSourceIntegral")
    gpd = importlib.import_module("ComputeTargets.GkSourcePolicyData")
    tsf = importlib.import_module("ComputeTargets.TkSourceFunctions")  # prompt 05a
    from ComputeTargets.tests.test_main_plumbing import MAIN_PY

    import dispatch

    tree, sha = TREES[tree_name]
    files = {
        "ComputeTargets": ComputeTargets.__file__,
        "ComputeTargets.primitive_phase": pp.__file__,
        "ComputeTargets.QuadSource": qs.__file__,
        "ComputeTargets.QuadSourceIntegral": qsi.__file__,
        "ComputeTargets.GkSourcePolicyData": gpd.__file__,
        "ComputeTargets.TkSourceFunctions": tsf.__file__,  # prompt 05a
        "StoreFixtures": StoreFixtures.__file__,
        "main.py": str(MAIN_PY),
        "theta_mod_2pi (dispatch.ORIG_THETA_MOD_2PI)": dispatch.ORIG_FILE,
    }
    in_tree = {
        k: str(Path(v).resolve()).startswith(str(tree) + os.sep)
        for k, v in files.items()
    }
    head, status = tree_state(tree)
    params = list(inspect.signature(pp.PrimitivePhase.__init__).parameters)
    return {
        "tree": tree_name,
        "tree_path": str(tree),
        "expected_sha": sha,
        "head": head,
        "status_short": status,
        "files": files,
        "all_in_tree": all(in_tree.values()),
        "in_tree": in_tree,
        "PrimitivePhase_has_exact_remainder": "exact_remainder" in params,
        "build_phase_source_mentions_exact_remainder": "exact_remainder=True"
        in inspect.getsource(gpd._build_phase),
        # prompt 05a: the keyword the code commit adds (absent at c, present and True at t)
        "TkSourceFunctions_exact_remainder_default": (
            inspect.signature(tsf.TkSourceFunctions.__init__)
            .parameters["exact_remainder"]
            .default
            if "exact_remainder"
            in inspect.signature(tsf.TkSourceFunctions.__init__).parameters
            else "absent"
        ),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "dont_write_bytecode": sys.dont_write_bytecode,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "python": sys.version,
    }


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    tree_name = spec["tree"]
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if spec.get("emit_fd") is not None:
        emit_handle = os.fdopen(int(spec["emit_fd"]), "w", buffering=1)
    else:
        emit_handle = open(spec["emit_file"], "a", buffering=1)

    def emit(kind, payload):
        emit_handle.write(json.dumps({"kind": kind, **payload}) + "\n")
        emit_handle.flush()

    ev = tree_evidence(tree_name)
    script_hashes = {name: sha256(HERE / name) for name in SCRIPTS if (HERE / name).exists()}
    ev["scripts_sha256"] = script_hashes
    emit("hello", {"evidence": ev})
    print(f"# worker {tree_name} evidence " + json.dumps(ev), flush=True)
    if not (
        ev["all_in_tree"]
        and ev["head"] == ev["expected_sha"]
        and ev["status_short"].strip() == ""
        and sys.dont_write_bytecode
    ):
        emit("refused", {"why": "tree evidence does not hold", "evidence": ev})
        raise SystemExit(f"worker {tree_name}: tree evidence does not hold: {ev}")

    from harness import QSI_MOD, close_pool, dump_record, open_readonly_pool, standin_ray
    from rebuild import MAIN_PY_SHA256, PRIMARY, build_context, rebuild_row

    import dispatch as DP
    import probe as PROBE  # prompt 05a

    orig_levin = QSI_MOD.adaptive_levin_sincos
    orig_eval = QSI_MOD.evaluate_QuadSource_integral
    orig_partition = QSI_MOD.build_partition  # prompt 05a
    holder = {"calls": None, "disp": None, "deadline": None, "partition": None}

    def w_partition(*a, **kw):  # prompt 05a: keep what the unit's own call built
        result = orig_partition(*a, **kw)
        holder["partition"] = result
        return result

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
        with DP.install(deadline=holder["deadline"], stop_exc=OverBudget) as disp:
            holder["disp"] = disp
            return orig_eval(*a, **kw)

    try:
        with standin_ray() as cluster:
            pool, printed = open_readonly_pool(PRIMARY, cluster)
            print(printed.strip(), flush=True)
            try:
                t_ctx = time.perf_counter()
                ctx = build_context(pool, log=lambda s: print(s, flush=True))
                rtol8 = ray.get(pool.object_get("tolerance", tol=1e-8))
                if not (rtol8.available and rtol8.tol == 1e-8):
                    raise RuntimeError(f"tolerance 1e-8 not available: {rtol8}")
                rtol_obj = {"1e-7": None, "1e-8": rtol8}
                emit(
                    "context",
                    {
                        "tree": tree_name,
                        "seconds": time.perf_counter() - t_ctx,
                        "checks": ctx["checks"],
                        "rtol_1e-7": [ctx["quad_rtol"].store_id, repr(ctx["quad_rtol"].tol)],
                        "rtol_1e-8": [rtol8.store_id, repr(rtol8.tol)],
                        "main_py_sha256": MAIN_PY_SHA256,
                    },
                )
                for u in spec["units"]:
                    uid = u["uid"]
                    if u["tree"] != tree_name:
                        raise RuntimeError(f"unit {uid} is not for tree {tree_name}")
                    print(f"\n== unit {uid} ({u['row']}: shard {u['shard']}, serial {u['serial']})", flush=True)
                    status = "done"
                    err = None
                    result = None
                    calls = []
                    wall = None
                    rb = None
                    holder.update(disp=None, partition=None)  # prompt 05a: partition
                    try:
                        t_rb = time.perf_counter()
                        rb = rebuild_row(
                            ctx,
                            pool,
                            u["shard"],
                            u["serial"],
                            log=lambda s: print(s, flush=True),
                            rtol=rtol_obj[u["rtol"]],
                        )
                        rebuild_s = time.perf_counter() - t_rb
                        holder.update(calls=calls)
                        QSI_MOD.adaptive_levin_sincos = w_levin
                        QSI_MOD.evaluate_QuadSource_integral = w_eval
                        QSI_MOD.build_partition = w_partition  # prompt 05a
                        t0 = time.perf_counter()
                        budget = u.get("budget")
                        holder["deadline"] = None if budget is None else t0 + budget
                        try:
                            ref = rb["work"].compute(payload=rb["compute_payload"])
                            result = ray.get(ref)
                        except OverBudget:
                            status = "over budget"
                            result = None
                        finally:
                            wall = time.perf_counter() - t0
                            holder.update(calls=None, deadline=None)
                            QSI_MOD.adaptive_levin_sincos = orig_levin
                            QSI_MOD.evaluate_QuadSource_integral = orig_eval
                            QSI_MOD.build_partition = orig_partition  # prompt 05a
                    except Exception:
                        status = "error"
                        err = traceback.format_exc()
                        print(err, flush=True)
                        rebuild_s = None
                    finally:
                        QSI_MOD.adaptive_levin_sincos = orig_levin
                        QSI_MOD.evaluate_QuadSource_integral = orig_eval
                        QSI_MOD.build_partition = orig_partition  # prompt 05a
                    assert QSI_MOD.adaptive_levin_sincos is orig_levin
                    assert QSI_MOD.evaluate_QuadSource_integral is orig_eval
                    assert QSI_MOD.build_partition is orig_partition  # prompt 05a
                    assert DP.PrimitivePhase.__dict__["theta_mod_2pi"] is DP.ORIG_THETA_MOD_2PI
                    disp = holder["disp"]
                    group_calls = [
                        c for c in calls if (c["parsed"] or {}).get("kind") == "group"
                    ]
                    head_now, status_now = tree_state(TREES[tree_name][0])
                    data = {
                        "unit": uid,
                        "tree": tree_name,
                        "rtol": u["rtol"],
                        "row": u["row"],
                        "role": u.get("role"),
                        "shard": u["shard"],
                        "serial": u["serial"],
                        "status": status,
                        "error": err,
                        "wall_s": wall,
                        "rebuild_s": rebuild_s,
                        "budget_s": u.get("budget"),
                        "dispatcher_counts": None if disp is None else dict(disp.counts),
                        "dispatcher_flags": None if disp is None else dict(disp.flags),
                        "tree_head": head_now,
                        "tree_status_short": status_now,
                        "scripts_sha256": script_hashes,
                        "worker_pid": os.getpid(),
                        "calls": [summarise_call(c) for c in group_calls],
                        "n_calls_total": len(calls),
                    }
                    if rb is not None:
                        data["qs_check"] = rb["qs_check"]
                        data["stored"] = summarise_stored(rb["stored"])
                        data["rebuild_checks_all_hold"] = all(c[1] for c in rb["checks"])
                    if result is not None:
                        data["result"] = summarise_result(result)
                    # prompt 05a: T's phase flags, and the phase probe (probe.py)
                    part = holder["partition"]
                    holder["partition"] = None
                    probe_samples = None
                    data["T_phase_exact_remainder"] = (
                        None
                        if part is None or len(part["subintervals"]) == 0
                        else {
                            "Tq": getattr(part["subintervals"][0]["Tq"].phase, "exact_remainder", None),
                            "Tr": getattr(part["subintervals"][0]["Tr"].phase, "exact_remainder", None),
                        }
                    )
                    data["probe"] = None
                    data["probe_s"] = None
                    data["probe_error"] = None
                    if u.get("probe") and status != "error":
                        if part is None:
                            data["probe_error"] = "no partition was built in this unit"
                        else:
                            t_pr = time.perf_counter()
                            try:
                                data["probe"], probe_samples = PROBE.probe_partition(
                                    part["subintervals"]
                                )
                            except Exception:
                                data["probe_error"] = traceback.format_exc()
                                print(data["probe_error"], flush=True)
                            data["probe_s"] = time.perf_counter() - t_pr
                    part = None
                    if spec.get("pickle_regions", True) and status != "error":
                        pkl = out_dir / f"unit_{uid}.pkl"
                        dump_record(
                            {
                                "unit": uid,
                                "data": data,
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
                                "probe_samples": probe_samples,  # prompt 05a
                            },
                            pkl,
                        )
                        data["pickle"] = pkl.name
                        data["pickle_sha256"] = sha256(pkl)
                    r = data.get("result")
                    print(
                        f"unit {uid}: {status}; wall {wall if wall is None else round(wall, 3)} s; "
                        f"budget {u.get('budget')}; counts {data['dispatcher_counts']}; flags "
                        f"{data['dispatcher_flags']}; "
                        + (
                            f"total {r['total']['repr']} abserr {r['total_abserr']['repr']} "
                            f"converged {r['total_converged']} WKB_Levin_data {r['WKB_Levin_data']}"
                            if r is not None
                            else f"{len(group_calls)} group calls completed"
                        )
                        + (
                            f"; QuadSource L0a {rb['qs_check']['L0a_holds']}"
                            if rb is not None
                            else ""
                        )
                        + f"; T exact_remainder {data['T_phase_exact_remainder']}"  # prompt 05a
                        + (
                            f"; probe {len(data['probe'])} windows in {data['probe_s']:.1f} s"
                            if data["probe"] is not None
                            else (f"; probe error" if data["probe_error"] else "")
                        ),
                        flush=True,
                    )
                    emit("unit", {"data": data})
            finally:
                print(close_pool(pool, cluster).strip(), flush=True)
    finally:
        QSI_MOD.adaptive_levin_sincos = orig_levin
        QSI_MOD.evaluate_QuadSource_integral = orig_eval
        QSI_MOD.build_partition = orig_partition  # prompt 05a
    emit("bye", {"ray_initialized": ray.is_initialized()})
    emit_handle.close()
    if ray.is_initialized():
        raise SystemExit("worker: Ray was initialised")


if __name__ == "__main__":
    main()
