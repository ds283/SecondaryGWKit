"""
levin-phase-floor prompt 06a (the depth census; copied from prompt 06's scripts/worker.py,
docs/levin-phase-floor/06/wider/scripts/worker.py, a record that is not edited). Changed here, each
change marked "prompt 06a" or confined to ``main()``, which is rewritten from 06's:
- one tree, ``h`` (prov.TREES), reading 06's certified scratch copy (route (B), as in 06);
- **the producer cache** (prompt 06a section 0.3 change 1): one ``rebuild.CensusCache`` per worker
  process, passed to ``rebuild_row(..., cache=...)``; every producer is re-run once in the worker
  (each GkWKBIntegration of a k once, each TkWKBIntegration once), each ``GkSource`` and
  ``GkSourcePolicyData`` assembled once per (k, z_response), each ``QuadSource`` once per (q, r).
  What the cache builds while a unit runs is emitted as ``event`` lines (the driver writes them to
  ``events_<worker>.jsonl``) and named in the unit's ``cache_built``;
- **no phase probe and no per-row deltaTheta** (prompt 06a section 0.3 change 4): phase_probe06.py
  is not imported, ``build_partition`` is not wrapped;
- **the stationary-point scan** (spscan.py; prompt 06a section 5 item 7) in a unit whose spec has
  ``"scan": true`` (every rtol 1e-7 unit), after the timed section with the dispatcher uninstalled,
  on each Levin call's own ``theta["theta_deriv"]``, kept by the ``adaptive_levin_sincos`` wrapper;
- **what a unit keeps** (section 2 item 7): in its checkpoint, 06's per-call summary, the row's
  totals and wall time (``_slim_result``), the scan's counts per call; in the worker's pickle
  ``regions_<worker>.pkl``, a stream of one record per unit, appended and flushed as the unit ends
  (offset, length and SHA-256 in the checkpoint), each call's region arrays (06's, plus relerr,
  phase_err and phase_limited), the scan in full and 06's full result summary;
- the ``!! WARNING (GkSourcePolicyData`` counter, per unit as in 06, and per (k, z_response) around
  each assembly (in the ``Gk`` event).

(06's text follows, unchanged.)

levin-phase-floor prompt 06 section 6 (copied from prompt 05a's scripts/worker.py,
docs/levin-phase-floor/05a/wider/scripts/worker.py, a record that is not edited). Changed here,
each change marked "prompt 06":
- the trees are ``t5`` (the worktree at ``9f66622``, 05a's t) and ``s`` (the worktree at
  ``f5cee4c``, the code commit ``dc9d28c``'s code) in place of 05a's ``c`` and ``t``;
- **the re-run of the producers**: a unit whose spec has ``"rerun": true`` (every ``s`` unit, and
  U0's faithfulness units at ``t5``) calls ``rebuild_row(..., rerun=True)`` (rebuild.py, rerun.py);
  its record carries the re-run's summary (timings, the comparison of every re-run value with the
  stored one, the recomputed ``GkSourcePolicyData``) and its pickle every re-run object's
  ``deltaTheta``;
- every unit records its ``GkSourcePolicyData``'s classification (``Levin_z`` among it), and counts
  the ``!! WARNING (GkSourcePolicyData`` lines (the disagreement of the two repairs, prompt 06
  section 0.2) its process printed while the unit ran, by a pass-through counter on ``sys.stdout``;
- the module-provenance evidence also names the producers' modules, ``WKB_phase_function``'s,
  ``GkSource``'s and rerun.py's imports, and whether the producers' ``store()`` reduce exactly
  (``reduce_exactly_mod_2pi`` in its source: absent at t5, present at s);
- **route (B)** (the user's decision): ``s`` reads, read-only, the certified scratch copy of A3 v2
  with the three nullable all-NULL ``theta_residual`` columns (prov.STORE_DIR_BY_TREE,
  rebuild.use_store_dir); ``t5`` reads A3 v2;
- **the phase probe** is 06's (phase_probe06.py): ``build_partition`` is wrapped at its module
  attribute as in 05a, now keeping its arguments (the ``GkSourcePolicyData`` and the two
  ``TkSourceFunctions`` the unit's own call was handed) as well as its result, and a unit whose
  spec has ``"probe": true`` (the rtol 1e-7 units of R1-R7) runs phase_probe06.probe_row on them
  after its timed section, with the dispatcher uninstalled. 05a's probe.py is not called.

(05a's text follows, unchanged; read its c as t5, its t as s, and its probe as 06's.)

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

SCRIPTS = [  # prompt 06a: run_census.py, spscan.py in place of run_wider.py, phase_probe06.py
    "worker.py",
    "dispatch.py",
    "harness.py",
    "rebuild.py",
    "prov.py",
    "run_census.py",
    "launch.py",
    "rerun.py",
    "spscan.py",
]


class _WarnCounter:
    """prompt 06: a pass-through on sys.stdout that counts the lines starting with
    ``!! WARNING (GkSourcePolicyData`` (the two repairs' disagreement, prompt 06 section 0.2)."""

    PREFIX = "!! WARNING (GkSourcePolicyData"

    def __init__(self, stream):
        self._stream = stream
        self.count = 0
        self.lines = []
        self._buf = ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.startswith(self.PREFIX):
                self.count += 1
                if len(self.lines) < 5:
                    self.lines.append(line[:400])
        return self._stream.write(text)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


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
    gkw = importlib.import_module("ComputeTargets.GkWKBIntegration")  # prompt 06
    tkw = importlib.import_module("ComputeTargets.TkWKBIntegration")  # prompt 06
    gks = importlib.import_module("ComputeTargets.GkSource")  # prompt 06
    wpf = importlib.import_module("Quadrature.integrators.WKB_phase_function")  # prompt 06
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
        "ComputeTargets.GkWKBIntegration": gkw.__file__,  # prompt 06
        "ComputeTargets.TkWKBIntegration": tkw.__file__,  # prompt 06
        "ComputeTargets.GkSource": gks.__file__,  # prompt 06
        "Quadrature.integrators.WKB_phase_function": wpf.__file__,  # prompt 06
        "WKB_phase_function (the producers' remote's function)": (
            gkw.WKB_phase_function._function.__code__.co_filename
        ),  # prompt 06
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
        # prompt 06: the producers' exact reduction (absent at t5, present at s)
        "producers_store_reduce_exactly": {
            "GkWKBIntegration": "reduce_exactly_mod_2pi"
            in inspect.getsource(gkw.GkWKBIntegration.store),
            "TkWKBIntegration": "reduce_exactly_mod_2pi"
            in inspect.getsource(tkw.TkWKBIntegration.store),
        },
        "GkWKBValue_has_theta_residual": "theta_residual"
        in inspect.signature(gkw.GkWKBValue.__init__).parameters,
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "dont_write_bytecode": sys.dont_write_bytecode,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "python": sys.version,
    }


def _region_arrays_06a(call):
    """prompt 06a: 06's region arrays, plus each region's relerr, phase_err and phase_limited."""
    regs = call["result"]["regions"]
    out = region_arrays(call)
    out["relerr"] = np.array(
        [np.nan if r.relerr is None else r.relerr for r in regs], dtype=float
    )
    out["phase_err"] = np.array(
        [np.nan if r.phase_err is None else r.phase_err for r in regs], dtype=float
    )
    out["phase_limited"] = np.array([bool(r.phase_limited) for r in regs], dtype=bool)
    return out


def _slim_result(result):
    """prompt 06a section 2 item 7: the row's totals for the checkpoint (the full summary goes to
    the pickle)."""
    d = result["WKB_Levin_data"]
    return {
        "total": fx(result["total"]),
        "total_abserr": fx(result["total_abserr"]),
        "total_converged": bool(result["total_converged"]),
        "total_phase_limited": bool(result["total_phase_limited"]),
        "numeric_quad": fx(result["numeric_quad"]),
        "WKB_Levin": fx(result["WKB_Levin"]),
        "analytic_rad": fx(result["analytic_rad"]),
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
    }


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    tree_name = spec["tree"]
    worker_name = spec.get("name", tree_name)  # prompt 06a
    out_dir = Path(spec["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if spec.get("emit_fd") is not None:
        emit_handle = os.fdopen(int(spec["emit_fd"]), "w", buffering=1)
    else:
        emit_handle = open(spec["emit_file"], "a", buffering=1)

    def emit(kind, payload):
        emit_handle.write(json.dumps({"kind": kind, "worker": worker_name, **payload}) + "\n")
        emit_handle.flush()

    ev = tree_evidence(tree_name)
    script_hashes = {name: sha256(HERE / name) for name in SCRIPTS if (HERE / name).exists()}
    ev["scripts_sha256"] = script_hashes
    emit("hello", {"evidence": ev})
    print(f"# worker {worker_name} evidence " + json.dumps(ev), flush=True)
    if not (
        ev["all_in_tree"]
        and ev["head"] == ev["expected_sha"]
        and ev["status_short"].strip() == ""
        and sys.dont_write_bytecode
    ):
        emit("refused", {"why": "tree evidence does not hold", "evidence": ev})
        raise SystemExit(f"worker {worker_name}: tree evidence does not hold: {ev}")

    from harness import QSI_MOD, close_pool, dump_record, open_readonly_pool, standin_ray
    import rebuild as RB
    from rebuild import MAIN_PY_SHA256, CensusCache, build_context, rebuild_row  # prompt 06a
    from prov import STORE_DIR_BY_TREE

    # route (B), as in 06: h reads the certified scratch copy
    PRIMARY = RB.use_store_dir(STORE_DIR_BY_TREE[tree_name])
    emit("store", {"tree": tree_name, "primary": str(PRIMARY)})
    print(f"# worker {worker_name} reads {PRIMARY}", flush=True)

    import dispatch as DP
    import spscan as SP  # prompt 06a: the stationary-point scan

    # prompt 06a: one pickle per worker, a stream of one record per unit, appended and flushed as
    # each unit ends (its offset, length and sha256 go in the unit's checkpoint record)
    pkl_path = out_dir / f"regions_{worker_name}.pkl"
    pkl_handle = open(pkl_path, "ab")

    orig_levin = QSI_MOD.adaptive_levin_sincos
    orig_eval = QSI_MOD.evaluate_QuadSource_integral
    holder = {"calls": None, "disp": None, "deadline": None}

    def w_levin(*a, **kw):
        result = orig_levin(*a, **kw)
        calls = holder["calls"]
        if calls is not None:
            x_span = a[0] if len(a) > 0 else kw.get("x_span")
            theta = a[2] if len(a) > 2 else kw.get("theta")  # prompt 06a: kept for the scan
            calls.append(
                {
                    "index": len(calls),
                    "notify_label": kw.get("notify_label"),
                    "parsed": parse_label(kw.get("notify_label")),
                    "x_span": tuple(x_span),
                    "atol": kw.get("atol"),
                    "rtol": kw.get("rtol"),
                    "result": result,
                    "theta_deriv": None if theta is None else theta.get("theta_deriv"),
                }
            )
        return result

    def w_eval(*a, **kw):
        with DP.install(deadline=holder["deadline"], stop_exc=OverBudget) as disp:
            holder["disp"] = disp
            return orig_eval(*a, **kw)

    cache = CensusCache(STORE_DIR_BY_TREE[tree_name], warn_counter_factory=_WarnCounter)

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
                    holder.update(disp=None)
                    warn = _WarnCounter(sys.stdout)
                    sys.stdout = warn
                    n_events_before = len(cache.events)
                    try:
                        t_rb = time.perf_counter()
                        rb = rebuild_row(
                            ctx,
                            pool,
                            u["shard"],
                            u["serial"],
                            log=lambda s: print(s, flush=True),
                            rtol=rtol_obj[u["rtol"]],
                            cache=cache,  # prompt 06a
                        )
                        rebuild_s = time.perf_counter() - t_rb
                        holder.update(calls=calls)
                        QSI_MOD.adaptive_levin_sincos = w_levin
                        QSI_MOD.evaluate_QuadSource_integral = w_eval
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
                    except Exception:
                        status = "error"
                        err = traceback.format_exc()
                        print(err, flush=True)
                        rebuild_s = None
                    finally:
                        QSI_MOD.adaptive_levin_sincos = orig_levin
                        QSI_MOD.evaluate_QuadSource_integral = orig_eval
                        sys.stdout = warn._stream
                    assert QSI_MOD.adaptive_levin_sincos is orig_levin
                    assert QSI_MOD.evaluate_QuadSource_integral is orig_eval
                    assert DP.PrimitivePhase.__dict__["theta_mod_2pi"] is DP.ORIG_THETA_MOD_2PI
                    disp = holder["disp"]
                    group_calls = [
                        c for c in calls if (c["parsed"] or {}).get("kind") == "group"
                    ]
                    # prompt 06a: what the cache built while this unit ran, emitted once each
                    events = cache.events[n_events_before:]
                    for e in events:
                        emit("event", {"unit": uid, "event": e})
                    head_now, status_now = tree_state(TREES[tree_name][0])
                    data = {
                        "unit": uid,
                        "tree": tree_name,
                        "rtol": u["rtol"],
                        "row": u["row"],
                        "names05": u.get("names05"),
                        "shard": u["shard"],
                        "serial": u["serial"],
                        "k_key": u.get("k_key"),
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
                        "rows_sha256": spec.get("rows_sha256"),
                        "worker": worker_name,
                        "worker_pid": os.getpid(),
                        "calls": [summarise_call(c) for c in group_calls],
                        "n_calls_total": len(calls),
                        "cache_built": [
                            {k_: v for k_, v in e.items() if k_ in ("kind", "k_serial", "z_response_serial", "mode_serial", "q_serial", "r_serial")}
                            for e in events
                        ],
                    }
                    if rb is not None:
                        data["qs_check"] = {
                            k_: v for k_, v in rb["qs_check"].items()
                            if k_ in ("stored_serial", "keyed_as_stored", "n_stored", "n_recomputed",
                                      "n_common_differing_columns", "added_is_B1s", "L0a_holds")
                        }
                        data["stored"] = summarise_stored(rb["stored"])
                        data["rebuild_checks_all_hold"] = all(c[1] for c in rb["checks"])
                        data["GkPolicy"] = rb["GkPolicy_summary"]
                    data["policy_warnings"] = {"count": warn.count, "first": warn.lines}
                    if result is not None:
                        data["result"] = _slim_result(result)
                    # prompt 06a section 5 item 7: the stationary-point scan, after the timed
                    # section, with the dispatcher uninstalled (it reads phase objects only)
                    scans = None
                    data["scan"] = None
                    data["scan_s"] = None
                    data["scan_error"] = None
                    if u.get("scan") and status != "error":
                        t_sc = time.perf_counter()
                        scans = []
                        for c in group_calls:
                            try:
                                if c["theta_deriv"] is None:
                                    raise RuntimeError("the call was handed no theta_deriv")
                                scans.append(
                                    SP.scan_call(c["theta_deriv"], c["x_span"], c["result"]["regions"])
                                )
                            except Exception:
                                scans.append({"error": traceback.format_exc()})
                                data["scan_error"] = (data["scan_error"] or 0) + 1
                        data["scan_s"] = time.perf_counter() - t_sc
                        data["scan"] = [
                            None
                            if "error" in s
                            else {
                                "index": c["index"],
                                "h": s["h"],
                                "n_sign_changes": s["n_sign_changes"],
                                "min_over_max_abs_theta_deriv": s["min_over_max_abs_theta_deriv"],
                                "sign_change_x": s["sign_change_x"][:8],
                                "n_regions_listed": s["n_regions_listed"],
                                "n_regions_listed_grid": s["n_regions_listed_grid"],
                                "n_regions_listed_ends": s["n_regions_listed_ends"],
                                "n_nonfinite": s["n_nonfinite"],
                            }
                            for c, s in zip(group_calls, scans)
                        ]
                    for c in calls:
                        c["theta_deriv"] = None
                    if status != "error":
                        rec = {
                            "unit": uid,
                            "regions": [
                                {
                                    "index": c["index"],
                                    "regime": c["parsed"]["regime"],
                                    "group": c["parsed"]["group"],
                                    "x_span": c["x_span"],
                                    **_region_arrays_06a(c),
                                }
                                for c in group_calls
                            ],
                            "scan": scans,
                            "result_full": None if result is None else summarise_result(result),
                        }
                        import io as _io

                        buf = _io.BytesIO()
                        from harness import _GroupPickler

                        _GroupPickler(buf, protocol=5).dump(rec)
                        blob = buf.getvalue()
                        offset = pkl_handle.tell()
                        pkl_handle.write(blob)
                        pkl_handle.flush()
                        os.fsync(pkl_handle.fileno())
                        import hashlib as _hl

                        data["pickle"] = pkl_path.name
                        data["pickle_offset"] = offset
                        data["pickle_length"] = len(blob)
                        data["pickle_record_sha256"] = _hl.sha256(blob).hexdigest()
                    r = data.get("result")
                    print(
                        f"unit {uid}: {status}; wall {wall if wall is None else round(wall, 3)} s; "
                        f"rebuild {rebuild_s if rebuild_s is None else round(rebuild_s, 3)} s; "
                        + (
                            f"total {r['total']['repr']} abserr {r['total_abserr']['repr']} "
                            f"converged {r['total_converged']} WKB_Levin_data {r['WKB_Levin_data']}"
                            if r is not None
                            else f"{len(group_calls)} group calls completed"
                        )
                        + f"; policy warnings {warn.count}; cache built {len(events)}"
                        + (f"; scan {data['scan_s']:.2f} s" if data["scan_s"] is not None else ""),
                        flush=True,
                    )
                    emit("unit", {"data": data})
                    del calls, group_calls, result, rb, scans
            finally:
                print(close_pool(pool, cluster).strip(), flush=True)
    finally:
        QSI_MOD.adaptive_levin_sincos = orig_levin
        QSI_MOD.evaluate_QuadSource_integral = orig_eval
        pkl_handle.close()
    emit("bye", {"ray_initialized": ray.is_initialized()})
    emit_handle.close()
    if ray.is_initialized():
        raise SystemExit("worker: Ray was initialised")


if __name__ == "__main__":
    main()
