"""
levin-phase-floor prompt 06c section 5 (06b's census re-run under the global error budget; copied
from prompt 06b's census scripts/worker.py, docs/levin-phase-floor/06b/census/scripts/worker.py, a
record that is not edited; its copy is kept unchanged in state/scripts_06b_reference/). Changed
here, each change marked "prompt 06c" (prompt 06c section 5: "Change only: the tree and its paths;
what X1-X5 need recorded per call; the reference for X1"):
- **the trees** ``p`` (06c's code commit's parent fef401c, Gate X0's reproduction of 06b) and
  ``g`` (the census, e5764a6: 06c's code plus 06d's), prov.TREES; the tree evidence also names
  whether ``AdaptiveLevin.levin_quadrature`` has the trust check, the Clenshaw-Curtis floor of 06d
  and ``_local_atol``, and its default depth_max;
- **per call** (``calls`` and ``other_calls``, as 06b): besides 06b's summary, the call's
  ``converged``, ``num_untrusted_regions`` and the error breakdown its result returns, its
  depth_max and the leaves at it, and per-call sums over the returned leaves: the share of its
  error from residual, floor and truncation (the resolution term being the trust check's
  declared one where set), the error it would declare without the trust check's widening, and
  the leaves by branch and by trust (``_summarise_call_06c``);
- **the budget loop's own counters**, from the one DEBUG record ``_adaptive_levin`` emits per call
  ("global budget | ..."): bisections, floor-limited and depth_max leaves, trust checks taken and
  their passes, leaves found untrusted during the call and returned untrusted, unchecked leaves,
  the running test and the returned flag, and resum rejections. A logging filter on the module's
  logger captures that record; it lets through only records at the level that reached the
  module's handlers before (WARNING and above), so nothing else the module logs changes. None at
  ``p``, whose code has no such record;
- **the solves by purpose**: a pass-through at the module attribute ``_adaptive_levin_subregion``
  (installed for the timed section only, like 06b's premise counter) counts each solve by the
  ``description`` its caller ``_solve`` was given, and for a fallback leaf's halves by who asked
  for them (the trust check, or the premise check / bisection), so the solves the trust check
  cost are known;
- **every returned leaf** (the pickle's region arrays, ``_region_arrays_06c``): 06b's arrays plus
  its truncation, total_err, and the trust check's delta1, delta2, convergence_ratio, trusted
  (1, 0, or -1 for not taken) and declared_resolution;
- **the deepest line of every call that reaches depth_max** (``_deep_line``, after the timed
  section, with every wrapper uninstalled): from the root to the leftmost leaf at the call's
  greatest depth, each ancestor P and its halves L, R recomputed by the module's own
  ``_adaptive_levin_subregion`` with the call's arguments, with each one's residual, floor,
  truncation and error max(residual, floor) + truncation, so that (e_L + e_R)/e_P and the
  residual's flatness can be read level by level, and the trust check's delta1, delta2 and r of
  each ancestor from its quarters. The line's last level is checked against the returned leaf
  (residual and floor bit for bit);
- **X1's inputs** (a unit whose spec has ``"capture"``): as 06b's W2 capture, under the key
  ``x1_inputs`` and into ``inputs_<worker>.pkl``, each record now also carrying the call's
  num_regions, max_depth and converged. The census run captures nothing; the 24 calls are
  captured after the run by re-running their units at ``g`` (x1_capture.py).

(06b's text follows, unchanged.)

levin-phase-floor prompt 06b section 5 (06a's census re-run at the code commit; copied from prompt
06a's scripts/worker.py, docs/levin-phase-floor/06a/scripts/worker.py, a record that is not edited;
its copy is kept unchanged in state/scripts_06a_reference/). Changed here, each change marked
"prompt 06b" (prompt 06b section 5: "Change only: the tree and its paths; what section 4's and W2's
comparisons need recorded per call; the reference for W2"):
- **the trees** ``p`` (the code commit's parent, Gate W0's reproduction of 06a) and ``b`` (the
  census), prov.TREES; the tree evidence also names ``AdaptiveLevin.levin_quadrature`` and whether
  it has the premise check;
- **every unit records where its modules were loaded from** (``modules``): every module in
  ``sys.modules`` at the unit's end classified by root (its own tree, the other tree, the main
  checkout outside the venv, the venv, the scripts, the standard library), with any module from the
  other tree or the main checkout listed; the ``bye`` message lists every module loaded from the
  tree (Gate W0's first clause);
- **per call** (group calls in ``calls`` as 06a, and now the other Levin calls, the analytic
  blocks, in ``other_calls``): the number of phase-limited regions, and the number of times the
  code's premise check ``_roundoff_exit_premise_holds`` was asked and failed in the call (a
  pass-through counter at the module attribute, installed with the other wrappers for the timed
  section only; absent at ``p``). The other calls' region arrays go to the pickle
  (``other_regions``);
- **W3's evidence** (a unit whose spec has ``"w3": true``, every 1e-7 unit): after the timed section,
  with the dispatcher and every wrapper uninstalled, for every region a call accepted as
  phase-limited, the region's own floor and its two halves' floors are recomputed by the module's
  own ``_adaptive_levin_subregion`` on the region and on its halves (the halves' own samples), with
  the call's arguments; the summary goes to the checkpoint (``w3``), the per-region numbers (hex)
  to the pickle (``w3``);
- **W2's inputs** (a unit whose spec has ``"capture"``, a list of [call index, notify_label] from
  06a's records): after the timed section, each listed call's span, integrand ``f``, phase dict
  ``theta`` and keyword arguments as the call saw them, with its value and abserr, are pickled
  (harness._GroupPickler) to ``inputs_<worker>.pkl``, one record per call (offset, length and
  SHA-256 in the checkpoint's ``w2_inputs``), for w2_reference.py to reload.

(06a's text follows, unchanged.)

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
    lq = importlib.import_module("AdaptiveLevin.levin_quadrature")  # prompt 06b
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
        "AdaptiveLevin.levin_quadrature": lq.__file__,  # prompt 06b
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
        # prompt 06b: the premise check (absent at p, present at b) and its ratio
        "levin_has_roundoff_exit_premise_holds": hasattr(lq, "_roundoff_exit_premise_holds"),
        "levin_roundoff_exit_min_floor_ratio": getattr(
            lq, "_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO", None
        ),
        "levin_adaptive_levin_has_bypass": "_test_only_bypass_roundoff_premise"
        in inspect.signature(lq._adaptive_levin).parameters,
        # prompt 06c: the global budget's trust check (present at g), 06d's Clenshaw-Curtis floor
        # (present at g), the length share of atol (absent at g), the default depth_max
        "levin_has_trust_check": hasattr(lq.used_interval, "convergence_ratio"),
        "levin_has_cc_roundoff_floor": hasattr(lq, "_cc_roundoff_floor"),
        "levin_has_local_atol": hasattr(lq, "_local_atol"),
        "levin_default_max_depth": getattr(lq, "DEFAULT_LEVIN_MAX_DEPTH", None),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "dont_write_bytecode": sys.dont_write_bytecode,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "python": sys.version,
    }


def _summarise_call_06b(call):
    """prompt 06b: 06a's per-call summary, plus the number of phase-limited regions and the code's
    premise checks in the call (None where the tree has no premise check)."""
    out = summarise_call(call)
    out["n_phase_limited_regions"] = int(
        sum(1 for r in call["result"]["regions"] if r.phase_limited)
    )
    log_ = call["premise_log"]  # None where the tree has no premise check (p)
    out["premise_checks"] = None if log_ is None else len(log_)
    out["premise_failed"] = None if log_ is None else sum(1 for x in log_ if not x)
    return out


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


def _module_roots(tree_name):
    """prompt 06b: where every module in sys.modules was loaded from, by root. Returns
    (counts by root, the modules from a foreign root, the sorted (name, file) of the tree's own)."""
    own_tree = str(TREES[tree_name][0]) + os.sep
    others = {
        n: str(t) + os.sep for n, (t, _sha) in TREES.items() if n != tree_name
    }
    main_dir = str(Path("/Users/ds283/Documents/Code/SecondaryGWKit")) + os.sep
    venv_dir = main_dir + "venv" + os.sep
    scripts_dir = str(HERE) + os.sep
    std_dirs = tuple(
        sorted({str(Path(p).resolve()) + os.sep for p in (sys.base_prefix, sys.prefix)})
    )
    counts = {}
    foreign = []
    own = []
    for name, mod in sorted(list(sys.modules.items())):
        f = getattr(mod, "__file__", None)
        if not f:
            root = "no file"
        else:
            rp = str(Path(f).resolve())
            if rp.startswith(own_tree):
                root = "own tree"
                own.append((name, rp))
            elif any(rp.startswith(o) for o in others.values()):
                root = "OTHER TREE"
                foreign.append((name, rp))
            elif rp.startswith(venv_dir):
                root = "venv"
            elif rp.startswith(scripts_dir):
                root = "scripts"
            elif rp.startswith(main_dir):
                root = "MAIN CHECKOUT"
                foreign.append((name, rp))
            elif rp.startswith(std_dirs):
                root = "python"
            else:
                root = "other"
                foreign.append((name, rp))
        counts[root] = counts.get(root, 0) + 1
    return counts, foreign, own


def _w3_region_floors(LQ, call):
    """prompt 06b, W3: for every region the call accepted as phase-limited, the region's own floor
    and its two halves' floors, each recomputed by the module's own _adaptive_levin_subregion on
    its own samples, with the call's arguments (as _adaptive_levin calls it), and the premise
    (halves' sum >= 1/2 of the region's own) evaluated here, not read from the code's flag. The
    break point is _levin_interval.break_point's, (end + start) / 2.0."""
    kw = call["kw"]
    rtol = kw.get("rtol", LQ.DEFAULT_LEVIN_RELTOL)
    order = kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
    bps = kw.get("build_p_sample", False)
    dpath = kw.get("diagnostics_path")
    dpath = Path(dpath) if dpath is not None else LQ.DEFAULT_LEVIN_DIAGNOSTICS_PATH
    basis = LQ._Basis_SinCos(call["theta"])
    vcache = {}

    def solve(span):
        d = LQ._adaptive_levin_subregion(
            span,
            call["f"],
            basis,
            id_label=LQ._LazyUUID(),
            chebyshev_order=order,
            rtol=rtol,
            notify_label=call["notify_label"],
            build_p_sample=bps,
            diagnostics_path=dpath,
            vectorize_cache=vcache,
        )
        return d.get("phase_err", 0.0) or 0.0, bool(d.get("is_direct", False))

    out = []
    for r in call["result"]["regions"]:
        if not r.phase_limited:
            continue
        a, b = r.start, r.end
        c = (b + a) / 2.0
        own, own_direct = solve((a, b))
        pl, l_direct = solve((a, c))
        pr, r_direct = solve((c, b))
        stored = r.phase_err if r.phase_err is not None else float("nan")
        s_halves = pl + pr
        out.append(
            {
                "start": float(a).hex(),
                "end": float(b).hex(),
                "depth": int(r.depth),
                "type": r.type,
                "phase_err_stored": float(stored).hex(),
                "phase_err_own": float(own).hex(),
                "own_equals_stored": float(own).hex() == float(stored).hex(),
                "own_is_direct": own_direct,
                "phase_err_L": float(pl).hex(),
                "phase_err_R": float(pr).hex(),
                "L_is_direct": l_direct,
                "R_is_direct": r_direct,
                "halves_sum": float(s_halves).hex(),
                "ratio": (s_halves / own) if own != 0 else None,
                "premise_holds": bool(s_halves >= 0.5 * own),
            }
        )
    return out


# prompt 06c ------------------------------------------------------------------------------------

# the arguments of _adaptive_levin's one DEBUG record per call ("global budget | ..."), in order,
# after its label (AdaptiveLevin/levin_quadrature.py at g, the _logger.debug after the loop)
_BUDGET_RECORD_KEYS = (
    "num_leaves",
    "num_splits",
    "num_floor_limited",
    "num_at_depth_max",
    "num_checks",
    "num_check_passes",
    "num_untrusted_found",
    "num_untrusted_returned",
    "num_unchecked_returned",
    "running_stop",
    "budget_met",
    "disagree",
    "num_resum_rejections",
)


def _budget_record(args):
    """prompt 06c: the budget record's arguments as a dict (None if none was captured)."""
    if args is None:
        return None
    args = list(args)
    if len(args) != 1 + len(_BUDGET_RECORD_KEYS):
        return {"raw": [repr(a) for a in args]}
    out = {}
    for k, v in zip(_BUDGET_RECORD_KEYS, args[1:]):
        if isinstance(v, str):
            out[k] = v
        elif k in ("running_stop", "budget_met"):
            out[k] = bool(v)
        else:
            out[k] = int(v)
    return out


def _summarise_call_06c(call):
    """prompt 06c: 06b's per-call summary, plus what X1-X5 and the descriptive items need from the
    call's result and its returned leaves (see the module docstring)."""
    import math

    out = _summarise_call_06b(call)
    res = call["result"]
    regs = res["regions"]
    dmax = call["kw"].get("depth_max", call["default_depth_max"])
    out["depth_max"] = _int(dmax)
    out["n_at_depth_max"] = int(sum(1 for r in regs if r.depth >= dmax))
    out["abserr_finite"] = bool(math.isfinite(float(res["abserr"])))
    out["num_untrusted_regions"] = _int(res.get("num_untrusted_regions"))
    out["num_phase_limited_regions_result"] = _int(res.get("num_phase_limited_regions"))
    for k in ("relerr", "abserr_resolution", "abserr_roundoff", "abserr_fallback", "abserr_truncation"):
        out[k] = fx(res.get(k))
    for k in ("num_subregion_solves", "num_solves_direct", "num_solves_lstsq", "num_solves_pinv"):
        out[k] = _int(res.get(k))
    e_res, e_floor, e_trunc, e_raw = [], [], [], []
    counts = {"levin": 0, "cc": 0, "trusted": 0, "untrusted": 0, "unchecked": 0, "floor_limited": 0, "widened": 0}
    r_max = None
    n_r_inf = 0
    for r in regs:
        counts["levin" if r.type == "Levin" else "cc"] += 1
        dec = getattr(r, "declared_resolution", None)
        resid = r.abserr
        resol = dec if dec is not None else resid
        floor = r.phase_err
        trunc = r.abserr_truncation if r.abserr_truncation is not None else 0.0
        # attribution of used_interval.total_err's max(resolution, floor): to the larger (ties to
        # the resolution term, as max() keeps its first argument), plus the truncation term
        if resol is None:
            e_floor.append(floor if floor is not None else 0.0)
        elif floor is None or resol >= floor:
            e_res.append(resol)
        else:
            e_floor.append(floor)
        e_trunc.append(trunc)
        # what the leaf would declare without the trust check's widening
        if resid is None:
            base = floor if floor is not None else 0.0
        elif floor is None:
            base = resid
        else:
            base = max(resid, floor)
        e_raw.append(base + trunc)
        trusted = getattr(r, "trusted", None)
        if r.phase_limited:
            counts["floor_limited"] += 1
        elif trusted is None:
            counts["unchecked"] += 1
        elif trusted:
            counts["trusted"] += 1
        else:
            counts["untrusted"] += 1
        if trusted and dec is not None:
            counts["widened"] += 1
        cr = getattr(r, "convergence_ratio", None)
        if cr is not None:
            if math.isfinite(cr):
                r_max = cr if r_max is None else max(r_max, cr)
            else:
                n_r_inf += 1
    shares = {"resolution": math.fsum(e_res), "floor": math.fsum(e_floor), "truncation": math.fsum(e_trunc)}
    out["E_share"] = {k: fx(v) for k, v in shares.items()}
    out["E_largest_share"] = max(shares, key=lambda k: shares[k]) if regs else None
    out["E_unwidened"] = fx(math.fsum(e_raw))
    out["leaves"] = counts
    out["r_max_finite"] = r_max
    out["n_r_not_finite"] = n_r_inf
    out["budget"] = call.get("budget")
    out["solves_by_purpose"] = call.get("solves")
    return out


def _region_arrays_06c(call):
    """prompt 06c: 06b's region arrays (06a's plus relerr, phase_err, phase_limited), plus each
    leaf's truncation, total_err and the trust check's numbers (NaN, or -1 for trusted, where not
    taken or absent)."""
    regs = call["result"]["regions"]
    out = _region_arrays_06a(call)

    def arr(name):
        vals = []
        for r in regs:
            v = getattr(r, name, None)
            vals.append(np.nan if v is None else float(v))
        return np.array(vals, dtype=float)

    for name in ("abserr_truncation", "total_err", "delta1", "delta2", "convergence_ratio", "declared_resolution"):
        out[name] = arr(name)
    out["trusted"] = np.array(
        [-1 if getattr(r, "trusted", None) is None else int(bool(r.trusted)) for r in regs], dtype=np.int8
    )
    return out


def _deep_line(LQ, call):
    """prompt 06c: the flat-residual item. For a call whose greatest leaf depth is at least its
    depth_max: the line from the root to the leftmost leaf at that depth, each ancestor P (and the
    leaf) with its halves L, R and quarters LL, LR, RL, RR solved once each by the module's own
    _adaptive_levin_subregion, with the call's arguments, as _adaptive_levin calls it (break point
    (end + start) / 2.0). Each node's residual is formed as _make_leaf forms it (Levin: |V - (V_L +
    V_R)| on its halves; fallback: its nested-pair estimate), its floor is its phase_err, its
    truncation its abserr_truncation, and its error max(residual, floor) + truncation (the trust
    check's widening not applied). delta1, delta2 and r of each P are formed as _check_leaf forms
    them. Returns a dict with every number as hex."""
    import math

    kw = call["kw"]
    rtol = kw.get("rtol", LQ.DEFAULT_LEVIN_RELTOL)
    order = kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
    bps = kw.get("build_p_sample", False)
    dpath = kw.get("diagnostics_path")
    dpath = Path(dpath) if dpath is not None else LQ.DEFAULT_LEVIN_DIAGNOSTICS_PATH
    basis = LQ._Basis_SinCos(call["theta"])
    vcache = {}
    solved = {}

    def solve(a, b):
        key = (a, b)
        if key not in solved:
            solved[key] = LQ._adaptive_levin_subregion(
                (a, b),
                call["f"],
                basis,
                id_label=LQ._LazyUUID(),
                chebyshev_order=order,
                rtol=rtol,
                notify_label=call["notify_label"],
                build_p_sample=bps,
                diagnostics_path=dpath,
                vectorize_cache=vcache,
            )
        return solved[key]

    def hx_(x):
        return None if x is None else float(x).hex()

    def node(a, b):
        d = solve(a, b)
        is_direct = bool(d.get("is_direct", False))
        floor = d.get("phase_err", 0.0) or 0.0
        trunc = d.get("abserr_truncation")
        if is_direct:
            resid = d["abserr_direct"]
        else:
            c = (b + a) / 2.0
            resid = np.fabs(d["value"] - (solve(a, c)["value"] + solve(c, b)["value"]))
        e = max(resid, floor) + (trunc if trunc is not None else 0.0)
        return {"type": "direct" if is_direct else "Levin", "value": hx_(d["value"]), "resid": hx_(resid),
                "floor": hx_(floor), "trunc": hx_(trunc), "e": hx_(e), "_e": float(e), "_resid": float(resid), "_floor": float(floor)}

    regs = call["result"]["regions"]
    D = max(r.depth for r in regs)
    leaf = min((r for r in regs if r.depth == D), key=lambda r: r.start)
    a, b = call["x_span"]
    levels = []
    for depth in range(D + 1):
        c = (b + a) / 2.0
        cL = (c + a) / 2.0
        cR = (b + c) / 2.0
        P, L, R = node(a, b), node(a, c), node(c, b)
        vP = solve(a, b)["value"]
        halves_sum = solve(a, c)["value"] + solve(c, b)["value"]
        quarters_sum = (solve(a, cL)["value"] + solve(cL, c)["value"]) + (solve(c, cR)["value"] + solve(cR, b)["value"])
        delta1 = float(np.fabs(vP - halves_sum))
        delta2 = float(np.fabs(halves_sum - quarters_sum))
        if not (math.isfinite(delta1) and math.isfinite(delta2)):
            ratio = float("inf")
        elif delta1 == 0.0:
            ratio = 0.0 if delta2 == 0.0 else float("inf")
        else:
            ratio = delta2 / delta1
        eLR = L["_e"] + R["_e"]
        levels.append(
            {
                "depth": depth,
                "start": hx_(a),
                "end": hx_(b),
                "P": {k_: v_ for k_, v_ in P.items() if not k_.startswith("_")},
                "L": {k_: v_ for k_, v_ in L.items() if not k_.startswith("_")},
                "R": {k_: v_ for k_, v_ in R.items() if not k_.startswith("_")},
                "eLR_over_eP": (eLR / P["_e"]) if P["_e"] != 0 else None,
                "delta1": hx_(delta1),
                "delta2": hx_(delta2),
                "r": hx_(ratio),
            }
        )
        if depth == D:
            last = P
            break
        if leaf.start >= c:
            a = c
        else:
            b = c
    reproduces = (
        a == leaf.start
        and b == leaf.end
        and float(last["_resid"]).hex() == float(leaf.abserr).hex()
        and float(last["_floor"]).hex() == float(leaf.phase_err if leaf.phase_err is not None else 0.0).hex()
    )
    return {
        "index": call["index"],
        "notify_label": call["notify_label"],
        "max_depth": D,
        "n_leaves_at_max_depth": int(sum(1 for r in regs if r.depth == D)),
        "leaf_start": hx_(leaf.start),
        "leaf_end": hx_(leaf.end),
        "line_end_equals_leaf_and_reproduces_its_residual_and_floor": bool(reproduces),
        "n_solves": len(solved),
        "levels": levels,
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
    import importlib as _il  # prompt 06b

    LQ = _il.import_module("AdaptiveLevin.levin_quadrature")  # prompt 06b
    inputs_path = out_dir / f"inputs_{worker_name}.pkl"  # prompt 06b: W2's inputs; prompt 06c: X1's
    inputs_handle = None

    # prompt 06c: the budget loop's one DEBUG record per call, captured by a filter on the module's
    # logger. The logger is opened to DEBUG so that the record is made; the filter lets through
    # only records at or above the level that reached the module's handlers before (its effective
    # level, WARNING with nothing configured), so what the module emits is unchanged.
    import logging as _logging

    class _BudgetRecordFilter(_logging.Filter):
        PREFIX = "adaptive_levin (%s): global budget |"

        def __init__(self, pass_level):
            super().__init__()
            self.pass_level = pass_level
            self.last = None
            self.n_seen = 0

        def filter(self, record):
            if record.levelno == _logging.DEBUG and isinstance(record.msg, str) and record.msg.startswith(self.PREFIX):
                self.last = tuple(record.args)
                self.n_seen += 1
            return record.levelno >= self.pass_level

    lq_logger = LQ._logger
    lq_pass_level = lq_logger.getEffectiveLevel()
    budget_filter = _BudgetRecordFilter(lq_pass_level)
    lq_logger.addFilter(budget_filter)
    lq_logger.setLevel(_logging.DEBUG)
    emit("levin_logging", {"logger": lq_logger.name, "pass_level": lq_pass_level,
                           "handlers": [type(h).__name__ for h in lq_logger.handlers],
                           "root_handlers": [type(h).__name__ for h in _logging.getLogger().handlers]})

    # prompt 06c: the solves by purpose, by a pass-through at the module attribute
    # _adaptive_levin_subregion, read from its caller's frame (installed for the timed section only)
    orig_subregion = LQ._adaptive_levin_subregion

    def w_subregion(*a_, **kw_):
        acc = holder.get("solve_acc")
        if acc is not None:
            try:
                fr = sys._getframe(1)
                if fr.f_code.co_name == "_solve":
                    key = str(fr.f_locals.get("description"))
                    if key in ("left half of fallback region", "right half of fallback region"):
                        key = f"fallback half, for {sys._getframe(3).f_code.co_name}"
                    elif key.startswith("trust check:"):
                        key = "trust check: a quarter"
                else:
                    key = f"<called from {fr.f_code.co_name}>"
            except Exception as e_:  # never let the count change the computation
                key = f"<unclassified: {type(e_).__name__}>"
            acc[key] = acc.get(key, 0) + 1
        return orig_subregion(*a_, **kw_)

    # prompt 06a: one pickle per worker, a stream of one record per unit, appended and flushed as
    # each unit ends (its offset, length and sha256 go in the unit's checkpoint record)
    pkl_path = out_dir / f"regions_{worker_name}.pkl"
    pkl_handle = open(pkl_path, "ab")

    orig_levin = QSI_MOD.adaptive_levin_sincos
    orig_eval = QSI_MOD.evaluate_QuadSource_integral
    holder = {"calls": None, "disp": None, "deadline": None, "premise_log": None, "solve_acc": None}  # prompt 06c
    # prompt 06b: the code's premise check, counted per call by a pass-through (b only)
    orig_premise = getattr(LQ, "_roundoff_exit_premise_holds", None)

    def w_premise(phase_err, phase_err_L, phase_err_R):
        r_ = orig_premise(phase_err, phase_err_L, phase_err_R)
        log_ = holder["premise_log"]
        if log_ is not None:
            log_.append(bool(r_))
        return r_

    def install_premise():
        if orig_premise is not None:
            LQ._roundoff_exit_premise_holds = w_premise
        LQ._adaptive_levin_subregion = w_subregion  # prompt 06c

    def restore_premise():
        if orig_premise is not None:
            LQ._roundoff_exit_premise_holds = orig_premise
        LQ._adaptive_levin_subregion = orig_subregion  # prompt 06c

    def w_levin(*a, **kw):
        plog = []  # prompt 06b
        holder["premise_log"] = plog
        sacc = {}  # prompt 06c
        holder["solve_acc"] = sacc
        budget_filter.last = None
        try:
            result = orig_levin(*a, **kw)
        finally:
            holder["premise_log"] = None
            holder["solve_acc"] = None  # prompt 06c
        budget_args = budget_filter.last  # prompt 06c
        budget_filter.last = None
        calls = holder["calls"]
        if calls is not None:
            x_span = a[0] if len(a) > 0 else kw.get("x_span")
            theta = a[2] if len(a) > 2 else kw.get("theta")  # prompt 06a: kept for the scan
            f_ = a[1] if len(a) > 1 else kw.get("f")  # prompt 06b: kept for W3 and W2's inputs
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
                    "theta": theta,  # prompt 06b
                    "f": f_,  # prompt 06b
                    "kw": {k_: v_ for k_, v_ in kw.items() if k_ not in ("x_span", "f", "theta")},
                    "premise_log": plog if orig_premise is not None else None,  # prompt 06b
                    "budget": _budget_record(budget_args),  # prompt 06c
                    "solves": dict(sacc),  # prompt 06c
                    "default_depth_max": getattr(LQ, "DEFAULT_LEVIN_MAX_DEPTH", None),  # prompt 06c
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
                        install_premise()  # prompt 06b
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
                            restore_premise()  # prompt 06b
                    except Exception:
                        status = "error"
                        err = traceback.format_exc()
                        print(err, flush=True)
                        rebuild_s = None
                    finally:
                        QSI_MOD.adaptive_levin_sincos = orig_levin
                        QSI_MOD.evaluate_QuadSource_integral = orig_eval
                        restore_premise()  # prompt 06b
                        sys.stdout = warn._stream
                    assert QSI_MOD.adaptive_levin_sincos is orig_levin
                    assert QSI_MOD.evaluate_QuadSource_integral is orig_eval
                    assert orig_premise is None or LQ._roundoff_exit_premise_holds is orig_premise
                    assert LQ._adaptive_levin_subregion is orig_subregion  # prompt 06c
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
                        "w2_capture_sha256": spec.get("w2_capture_sha256"),  # prompt 06b
                        "worker": worker_name,
                        "worker_pid": os.getpid(),
                        "calls": [_summarise_call_06c(c) for c in group_calls],  # prompt 06c
                        # prompt 06b: the other Levin calls (the analytic blocks), as the group calls
                        "other_calls": [
                            _summarise_call_06c(c) for c in calls if id(c) not in {id(g) for g in group_calls}  # prompt 06c
                        ],
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
                    # prompt 06b, W3: the halves' floors of every phase-limited region, recomputed
                    w3_detail = None
                    data["w3"] = None
                    if u.get("w3") and status != "error":
                        t_w3 = time.perf_counter()
                        w3_detail = []
                        n_reg = n_viol = n_mis = n_err = 0
                        min_ratio = None
                        for c in calls:
                            if not any(r.phase_limited for r in c["result"]["regions"]):
                                continue
                            try:
                                regs = _w3_region_floors(LQ, c)
                            except Exception:
                                w3_detail.append({"index": c["index"], "error": traceback.format_exc()})
                                n_err += 1
                                continue
                            w3_detail.append({"index": c["index"], "notify_label": c["notify_label"], "regions": regs})
                            for g in regs:
                                n_reg += 1
                                n_viol += not g["premise_holds"]
                                n_mis += not g["own_equals_stored"]
                                if g["ratio"] is not None:
                                    min_ratio = g["ratio"] if min_ratio is None else min(min_ratio, g["ratio"])
                        data["w3"] = {
                            "n_regions": n_reg,
                            "n_premise_fails": n_viol,
                            "n_own_floor_differs_from_stored": n_mis,
                            "n_errors": n_err,
                            "min_ratio": min_ratio,
                            "seconds": time.perf_counter() - t_w3,
                        }
                    # prompt 06c: the deepest line of every call that reaches its depth_max, recomputed
                    deep_detail = None
                    data["deep_lines"] = None
                    if status != "error":
                        t_dl = time.perf_counter()
                        deep_detail = []
                        for c in calls:
                            regs_ = c["result"]["regions"]
                            dmax_ = c["kw"].get("depth_max", c["default_depth_max"])
                            if not regs_ or dmax_ is None or max(r_.depth for r_ in regs_) < dmax_:
                                continue
                            try:
                                deep_detail.append(_deep_line(LQ, c))
                            except Exception:
                                deep_detail.append({"index": c["index"], "notify_label": c["notify_label"], "error": traceback.format_exc()})
                        data["deep_lines"] = {
                            "n_calls": len(deep_detail),
                            "n_errors": sum(1 for d_ in deep_detail if "error" in d_),
                            "n_not_reproducing": sum(
                                1 for d_ in deep_detail
                                if "error" not in d_ and not d_["line_end_equals_leaf_and_reproduces_its_residual_and_floor"]
                            ),
                            "seconds": time.perf_counter() - t_dl,
                            # compact, per line: per level [P's residual, floor, truncation, e,
                            # (e_L + e_R)/e_P, r] (hex; the ratio as a float); the full line is in
                            # the pickle's deep_lines
                            "lines": [
                                d_ if "error" in d_ else {
                                    k_: d_[k_] for k_ in ("index", "notify_label", "max_depth", "n_leaves_at_max_depth",
                                                          "leaf_start", "leaf_end", "n_solves",
                                                          "line_end_equals_leaf_and_reproduces_its_residual_and_floor")
                                } | {"levels": [[L_["P"]["type"], L_["P"]["resid"], L_["P"]["floor"], L_["P"]["trunc"],
                                                 L_["P"]["e"], L_["eLR_over_eP"], L_["r"]] for L_ in d_["levels"]]}
                                for d_ in deep_detail
                            ],
                        }
                    # prompt 06b, W2's inputs: each listed call's span, f, theta and arguments
                    # prompt 06c: X1's, under the key x1_inputs
                    data["x1_inputs"] = None
                    if u.get("capture") and status != "error":
                        import hashlib as _hl2
                        import io as _io2
                        from harness import _GroupPickler as _GP2

                        data["x1_inputs"] = []  # prompt 06c
                        by_index = {c["index"]: c for c in calls}
                        if inputs_handle is None:
                            inputs_handle = open(inputs_path, "ab")
                        for idx, label in u["capture"]:
                            c = by_index.get(idx)
                            if c is None or c["notify_label"] != label:
                                data["x1_inputs"].append({"index": idx, "error": "call not found or its label differs",
                                                          "label_found": None if c is None else c["notify_label"]})
                                continue
                            try:
                                buf2 = _io2.BytesIO()
                                _GP2(buf2, protocol=5).dump(
                                    {
                                        "unit": uid,
                                        "tree": tree_name,
                                        "index": idx,
                                        "notify_label": c["notify_label"],
                                        "x_span": c["x_span"],
                                        "f": c["f"],
                                        "theta": c["theta"],
                                        "kw": c["kw"],
                                        "value": float(c["result"]["value"]).hex(),
                                        "abserr": float(c["result"]["abserr"]).hex(),
                                        "num_regions": int(c["result"]["num_regions"]),
                                        # prompt 06c: the rest of what "moved" and X1 read
                                        "max_depth": int(max(r_.depth for r_ in c["result"]["regions"])),
                                        "converged": bool(c["result"]["converged"]),
                                        "rtol": c["rtol"],
                                        "atol": c["atol"],
                                    }
                                )
                                blob2 = buf2.getvalue()
                                off2 = inputs_handle.tell()
                                inputs_handle.write(blob2)
                                inputs_handle.flush()
                                os.fsync(inputs_handle.fileno())
                                data["x1_inputs"].append({"index": idx, "pickle": inputs_path.name, "offset": off2,
                                                          "length": len(blob2), "sha256": _hl2.sha256(blob2).hexdigest()})
                            except Exception:
                                data["x1_inputs"].append({"index": idx, "error": traceback.format_exc()})
                    # prompt 06b: where this unit's modules were loaded from
                    m_counts, m_foreign, m_own = _module_roots(tree_name)
                    data["modules"] = {
                        "counts_by_root": m_counts,
                        "foreign": m_foreign,
                        "n_own_tree": len(m_own),
                        "files": ev["files"],
                        "all_in_tree": ev["all_in_tree"],
                    }
                    for c in calls:
                        c["theta_deriv"] = None
                        c["theta"] = None  # prompt 06b
                        c["f"] = None  # prompt 06b
                    if status != "error":
                        rec = {
                            "unit": uid,
                            "regions": [
                                {
                                    "index": c["index"],
                                    "regime": c["parsed"]["regime"],
                                    "group": c["parsed"]["group"],
                                    "x_span": c["x_span"],
                                    **_region_arrays_06c(c),  # prompt 06c
                                }
                                for c in group_calls
                            ],
                            "scan": scans,
                            "result_full": None if result is None else summarise_result(result),
                            # prompt 06b: the other Levin calls' region arrays, and W3's numbers
                            "other_regions": [
                                {
                                    "index": c["index"],
                                    "notify_label": c["notify_label"],
                                    "x_span": c["x_span"],
                                    **_region_arrays_06c(c),  # prompt 06c
                                }
                                for c in calls
                                if id(c) not in {id(g) for g in group_calls}
                            ],
                            "w3": w3_detail,
                            "deep_lines": deep_detail,  # prompt 06c
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
                    del calls, group_calls, result, rb, scans, deep_detail  # prompt 06c
            finally:
                print(close_pool(pool, cluster).strip(), flush=True)
    finally:
        QSI_MOD.adaptive_levin_sincos = orig_levin
        QSI_MOD.evaluate_QuadSource_integral = orig_eval
        pkl_handle.close()
        if inputs_handle is not None:  # prompt 06b
            inputs_handle.close()
    _c, _f, _own = _module_roots(tree_name)  # prompt 06b: Gate W0's first clause
    emit("bye", {"ray_initialized": ray.is_initialized(),
                 "modules_by_root": _c, "modules_foreign": _f, "modules_own_tree": _own})
    emit_handle.close()
    if ray.is_initialized():
        raise SystemExit("worker: Ray was initialised")


if __name__ == "__main__":
    main()
