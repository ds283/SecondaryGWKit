"""
levin-phase-floor prompt 04, step 3 (route (a) proved again on one row): prompt 03's
scripts/run_instrument.py copied at the dispatch SHA, changed only in its provenance (both trees'
HEADs, from prov.py). Prompt 03 steps 5-6: re-run QuadSourceIntegral rows of A3 v2 in-process,
with two transparent wrappers in place, check each against the stored row bit for bit (M0), and
pickle what the wrappers kept, row by row, into the output directory.

Usage (from the repository root):

    PYTHONPATH=. ./venv/bin/python var/levin-phase-floor-03/scripts/run_instrument.py \
        (--out-dir DIR | --run-dir RUN_DIR) [--rows R1,R2,...] [--extra SHARD:SERIAL ...]

``--run-dir`` is the registered run's directory, created (manifest first) by scripts/launch.py,
which starts this script detached: a unit is recorded in the run's ledger per row, the heartbeat
is refreshed at each row's start and end, and the run's end is recorded.  ``--out-dir`` names a
directory the caller owns, and nothing is registered (the smoke test).

The wrappers (prompt step 6)
----------------------------
- ``ComputeTargets.QuadSourceIntegral.adaptive_levin_sincos``: calls the original with exactly
  its arguments, returns its return value (the same object), and records that object, the
  arguments it can read without changing them (x_span, atol, rtol, chebyshev_order and the
  theta dict's functions), and the group label **parsed from notify_label**.
- ``ComputeTargets.QuadSourceIntegral.build_phase_groups``: likewise, keeping the returned list
  of ``PhaseGroup`` objects and its arguments.

Both module attributes are restored in a ``finally``.  A Levin call is matched to its
``PhaseGroup`` by (regime, label) parsed from notify_label, and the match is confirmed by object
identity: the call's ``theta["theta_mod_2pi"]`` must be that group's ``theta_mod_2pi``.

M0 (prompt step 5): exact equality (``==``; ``is`` for None) of total, total_abserr,
total_converged, WKB_Levin, numeric_quad, WKB_Levin_data.max_depth and num_regions, and every
field of every group record of metadata["WKB_Levin"]["subintervals"] except ``elapsed`` (a
wall-clock time); and analytic_rad (reported, not gated).  Floats are also compared by their
IEEE-754 bit pattern.  The sub-interval records' own fields are compared too.

The registered run stops at the first row that fails M0 (prompt §5).
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import re
import socket
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

import ray  # noqa: E402
import ray.cloudpickle as cloudpickle  # noqa: E402

from harness import (  # noqa: E402
    QSI_MOD,
    close_pool,
    dump_record,
    open_readonly_pool,
    standin_ray,
)

# the seven rows of prompt §1: (shard, serial)
ROWS = {
    "R1": (2, 37527),
    "R2": (0, 1797),
    "R3": (1, 37670),
    "R4": (2, 37720),
    "R5": (0, 37966),
    "R6": (2, 34728),
    "R7": (2, 34820),
}

LABEL_RE = re.compile(
    r"^k=(?P<k>[^/]+)/Mpc, q=(?P<q>[^/]+)/Mpc, r=(?P<r>[^/]+)/Mpc @ z_response=(?P<zr>[^,]+), "
    r"z in \((?P<zmax>[^,]+), (?P<zmin>[^)]+)\), regime (?P<regime>\S+), group (?P<group>\S+)$"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head() -> str:
    from prov import git_heads

    wt, wts, mh, mhs = git_heads()
    return f"worktree {wt} ({wts}); main checkout {mh} ({mhs})"


def git_status() -> str:
    from prov import WORKTREE

    return subprocess.run(
        ["git", "-C", str(WORKTREE), "status", "--short"], capture_output=True, text=True
    ).stdout


def parse_label(notify_label):
    if notify_label is None:
        return None
    m = LABEL_RE.match(notify_label)
    if m is None:
        return {"kind": "other", "notify_label": notify_label}
    d = m.groupdict()
    d["kind"] = "group"
    return d


class Recorder:
    """The two wrappers' records for one row."""

    def __init__(self):
        self.levin_calls = []
        self.build_calls = []


def make_wrappers(recorder_holder, orig_levin, orig_build):
    def adaptive_levin_sincos(*args, **kwargs):
        result = orig_levin(*args, **kwargs)
        rec = recorder_holder[0]
        if rec is not None:
            x_span = args[0] if len(args) > 0 else kwargs.get("x_span")
            theta = args[2] if len(args) > 2 else kwargs.get("theta")
            rec.levin_calls.append(
                {
                    "index": len(rec.levin_calls),
                    "notify_label": kwargs.get("notify_label"),
                    "parsed": parse_label(kwargs.get("notify_label")),
                    "x_span": tuple(x_span),
                    "atol": kwargs.get("atol"),
                    "rtol": kwargs.get("rtol"),
                    "chebyshev_order": kwargs.get("chebyshev_order"),
                    "theta_keys": (
                        sorted(theta.keys()) if isinstance(theta, dict) else None
                    ),
                    "theta_mod_2pi_id": (
                        id(theta.get("theta_mod_2pi"))
                        if isinstance(theta, dict)
                        else None
                    ),
                    "positional_args": len(args),
                    "kwarg_names": sorted(kwargs.keys()),
                    "result": result,
                }
            )
        return result

    def build_phase_groups(*args, **kwargs):
        groups = orig_build(*args, **kwargs)
        rec = recorder_holder[0]
        if rec is not None:
            regime = args[0] if len(args) > 0 else kwargs.get("regime")
            rec.build_calls.append(
                {
                    "index": len(rec.build_calls),
                    "regime": tuple(regime),
                    "regime_label": "".join(
                        name for name, flag in zip(("G", "q", "r"), regime) if flag
                    ),
                    "kwargs": dict(kwargs),
                    "groups": groups,
                    "group_ids": {g.label: id(g.theta_mod_2pi) for g in groups},
                }
            )
        return groups

    return adaptive_levin_sincos, build_phase_groups


# ---------------------------------------------------------------------------------------------
# M0
# ---------------------------------------------------------------------------------------------


def _bits(x):
    return struct.pack("<d", x) if isinstance(x, float) else None


def compare(name, computed, stored, lines, gate=True):
    if computed is None or stored is None:
        eq = computed is stored
        how = "is"
    else:
        eq = computed == stored
        how = "=="
        if isinstance(computed, float) and isinstance(stored, float):
            beq = _bits(computed) == _bits(stored)
            how = "== and bits"
            eq = eq and beq
        elif type(computed) is not type(stored) and not (
            isinstance(computed, (bool, int)) and isinstance(stored, (bool, int))
        ):
            how = f"== ({type(computed).__name__} vs {type(stored).__name__})"
    verdict = ("EQUAL" if eq else "DIFFER") + ("" if gate else " (not gated)")
    lines.append(
        f"    {name:<58} {verdict:<20} [{how}] computed={computed!r} stored={stored!r}"
    )
    return eq or not gate


GROUP_FIELDS = (
    "label",
    "value",
    "abserr",
    "converged",
    "phase_limited",
    "regions",
    "simple_regions",
    "evaluations",
)
SUB_FIELDS = (
    "z_max",
    "z_min",
    "regime",
    "value",
    "abserr",
    "converged",
    "phase_limited",
)


def m0(result: dict, stored, lines) -> bool:
    ok = True
    ok &= compare("total", result["total"], stored.total, lines)
    ok &= compare("total_abserr", result["total_abserr"], stored.total_abserr, lines)
    ok &= compare(
        "total_converged", result["total_converged"], stored.total_converged, lines
    )
    ok &= compare("WKB_Levin", result["WKB_Levin"], stored.WKB_Levin, lines)
    ok &= compare("numeric_quad", result["numeric_quad"], stored.numeric_quad, lines)
    rd, sd = result["WKB_Levin_data"], stored.WKB_Levin_data
    if rd is None or sd is None:
        ok &= compare("WKB_Levin_data", rd, sd, lines)
    else:
        ok &= compare("WKB_Levin_data.max_depth", rd.max_depth, sd.max_depth, lines)
        ok &= compare(
            "WKB_Levin_data.num_regions", rd.num_regions, sd.num_regions, lines
        )
        for f in (
            "evaluations",
            "num_simple_regions",
            "num_SVD_errors",
            "num_order_changes",
            "chebyshev_min_order",
        ):
            compare(
                f"WKB_Levin_data.{f}", getattr(rd, f), getattr(sd, f), lines, gate=False
            )
    compare(
        "analytic_rad",
        float(result["analytic_rad"]),
        stored.analytic_rad,
        lines,
        gate=False,
    )
    compare(
        "total_phase_limited",
        result["total_phase_limited"],
        stored.total_phase_limited,
        lines,
        gate=False,
    )
    compare("b", result["b"], stored.b, lines, gate=False)
    rsubs = result["metadata"]["WKB_Levin"]["subintervals"]
    ssubs = stored.metadata["WKB_Levin"]["subintervals"]
    ok &= compare("len(subintervals)", len(rsubs), len(ssubs), lines)
    # the stored metadata went through json.dumps/json.loads; compare like with like
    rsubs_json = json.loads(json.dumps(rsubs))
    for i, (rs, rsj, ss) in enumerate(zip(rsubs, rsubs_json, ssubs)):
        for f in SUB_FIELDS:
            ok &= compare(
                f"sub[{i}].{f}", rs[f] if f != "regime" else rsj[f], ss[f], lines
            )
        ok &= compare(
            f"sub[{i}] len(groups)", len(rs["groups"]), len(ss["groups"]), lines
        )
        for j, (rg, sg) in enumerate(zip(rs["groups"], ss["groups"])):
            for f in GROUP_FIELDS:
                ok &= compare(
                    f"sub[{i}].groups[{j}]({sg['label']}).{f}", rg[f], sg[f], lines
                )
            compare(
                f"sub[{i}].groups[{j}]({sg['label']}).elapsed",
                rg["elapsed"],
                sg["elapsed"],
                lines,
                gate=False,
            )
    return ok


# ---------------------------------------------------------------------------------------------


def match_calls(rec: Recorder, lines):
    """Match each group Levin call to its PhaseGroup by (regime, label) and confirm by identity."""
    by_regime = {}
    for b in rec.build_calls:
        by_regime.setdefault(b["regime_label"], []).append(b)
    ok = True
    for call in rec.levin_calls:
        p = call["parsed"]
        if p is None or p["kind"] != "group":
            call["build_index"] = None
            call["group_label"] = None
            continue
        builds = by_regime.get(p["regime"], [])
        matches = [
            (b["index"], g)
            for b in builds
            for g in b["groups"]
            if g.label == p["group"] and id(g.theta_mod_2pi) == call["theta_mod_2pi_id"]
        ]
        if len(matches) != 1:
            ok = False
            lines.append(
                f"    !! Levin call {call['index']} ({call['notify_label']}) matched {len(matches)} groups"
            )
            call["build_index"] = None
        else:
            call["build_index"] = matches[0][0]
        call["group_label"] = p["group"]
    return ok


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out-dir")
    group.add_argument("--run-dir")
    parser.add_argument("--rows", default=",".join(ROWS))
    parser.add_argument("--extra", nargs="*", default=[])
    args = parser.parse_args()

    from rebuild import MAIN_PY_SHA256, PRIMARY, build_context, rebuild_row

    rows = [(name, ROWS[name]) for name in args.rows.split(",") if name]
    for spec in args.extra:
        shard, serial = spec.split(":")
        rows.append((f"X{shard}:{serial}", (int(shard), int(serial))))

    run = None
    if args.run_dir is not None:
        import signal

        import runregistrykit

        run = runregistrykit.Run(args.run_dir)
        out_dir = Path(run.path)

        def terminal(signum, _frame):
            """Record the death, then die exactly as the default handler would have."""
            run.finish("killed", exit_code=128 + signum)
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, terminal)
        run.heartbeat(pid=os.getpid(), stage="started: building the context")
    else:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    header = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "harness_sha256": sha256(HERE / "harness.py"),
        "rebuild_sha256": sha256(HERE / "rebuild.py"),
        "main_py_sha256": MAIN_PY_SHA256,
        "git_head": git_head(),
        "git_status_short": git_status(),
        "host": socket.gethostname(),
        "python": sys.version,
        "pid": os.getpid(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rows": rows,
    }
    print("# run_instrument header " + json.dumps(header, default=str), flush=True)

    orig_levin = QSI_MOD.adaptive_levin_sincos
    orig_build = QSI_MOD.build_phase_groups
    holder = [None]
    w_levin, w_build = make_wrappers(holder, orig_levin, orig_build)

    summary = []
    all_ok = True
    try:
        with standin_ray() as cluster:
            pool, printed = open_readonly_pool(PRIMARY, cluster)
            print(printed.strip(), flush=True)
            try:
                t0 = time.perf_counter()
                ctx = build_context(pool, log=lambda s: print(s, flush=True))
                print(
                    f"** context built in {time.perf_counter() - t0:.1f} s; checks: {ctx['checks']}",
                    flush=True,
                )
                for name, (shard, serial) in rows:
                    if run is not None:
                        run.heartbeat(stage=f"row {name} ({shard}, {serial}) starting")
                    print(f"\n== {name} (shard {shard}, serial {serial})", flush=True)
                    t1 = time.perf_counter()
                    rb = rebuild_row(
                        ctx, pool, shard, serial, log=lambda s: print(s, flush=True)
                    )
                    print(
                        f"   rebuilt in {time.perf_counter() - t1:.1f} s; checks: {rb['checks']}",
                        flush=True,
                    )

                    rec = Recorder()
                    QSI_MOD.adaptive_levin_sincos = w_levin
                    QSI_MOD.build_phase_groups = w_build
                    holder[0] = rec
                    try:
                        t2 = time.perf_counter()
                        ref = rb["work"].compute(payload=rb["compute_payload"])
                        result = ray.get(ref)
                        wall = time.perf_counter() - t2
                    finally:
                        holder[0] = None
                        QSI_MOD.adaptive_levin_sincos = orig_levin
                        QSI_MOD.build_phase_groups = orig_build
                    assert QSI_MOD.adaptive_levin_sincos is orig_levin
                    assert QSI_MOD.build_phase_groups is orig_build

                    lines = []
                    ok = m0(result, rb["stored"], lines)
                    match_ok = match_calls(rec, lines)
                    n_group_calls = sum(
                        1 for c in rec.levin_calls if c["group_label"] is not None
                    )
                    n_records = sum(
                        len(s["groups"])
                        for s in result["metadata"]["WKB_Levin"]["subintervals"]
                    )
                    lines.append(
                        f"    wrapper records: {len(rec.build_calls)} build_phase_groups calls, "
                        f"{len(rec.levin_calls)} adaptive_levin_sincos calls ({n_group_calls} group calls "
                        f"for {n_records} stored group records); every group call matched by "
                        f"(regime, label) and identity: {match_ok}"
                    )
                    m0_text = f"M0 {name} (shard {shard}, serial {serial}): {'HOLDS' if ok else 'FAILS'} ; compute wall {wall:.1f} s, stored compute_time {rb['stored'].compute_time:.1f} s"
                    print(m0_text, flush=True)
                    for line in lines:
                        print(line, flush=True)

                    stored = rb["stored"]
                    record = {
                        "name": name,
                        "shard": shard,
                        "serial": serial,
                        "header": header,
                        "m0_holds": ok,
                        "m0_lines": lines,
                        "match_ok": match_ok,
                        "k_inv_Mpc": rb["k"].k.k_inv_Mpc,
                        "q_inv_Mpc": rb["q"].k.k_inv_Mpc,
                        "r_inv_Mpc": rb["r"].k.k_inv_Mpc,
                        "k_store_id": rb["k"].store_id,
                        "q_store_id": rb["q"].store_id,
                        "r_store_id": rb["r"].store_id,
                        "z_response": rb["z_response"].z,
                        "eta_response": stored.eta_response,
                        "stored_total": stored.total,
                        "stored_total_abserr": stored.total_abserr,
                        "stored_metadata": stored.metadata,
                        "result": result,
                        "levin_calls": rec.levin_calls,
                        "build_calls": rec.build_calls,
                        "model_functions": ctx["bg_model"].functions,
                        "compute_wall_s": wall,
                    }
                    pkl = out_dir / f"row_{name.replace(':', '_')}.pkl"
                    try:
                        dump_record(record, pkl)
                        size = pkl.stat().st_size
                        print(
                            f"   pickled {pkl.name}: {size} bytes, sha256 {sha256(pkl)}",
                            flush=True,
                        )
                    except Exception:
                        print(
                            "   !! PICKLE FAILED:\n" + traceback.format_exc(),
                            flush=True,
                        )
                        raise
                    summary.append((name, shard, serial, ok, match_ok, wall))
                    if run is not None:
                        run.heartbeat(
                            stage=f"row {name} done: M0 {'holds' if ok else 'FAILS'}"
                        )
                        run.record(
                            name,
                            {
                                "shard": shard,
                                "serial": serial,
                                "m0": ok,
                                "match": match_ok,
                                "compute_wall_s": round(wall, 1),
                                "pickle": pkl.name,
                                "pickle_sha256": sha256(pkl),
                            },
                        )
                    if not (ok and match_ok):
                        all_ok = False
                        print(
                            f"!! STOP: M0 fails (or a call did not match) on {name}; no further rows are run",
                            flush=True,
                        )
                        break
            finally:
                print(close_pool(pool, cluster).strip(), flush=True)
    except BaseException:
        if run is not None:
            run.finish("failed", exit_code=1)
        raise
    finally:
        QSI_MOD.adaptive_levin_sincos = orig_levin
        QSI_MOD.build_phase_groups = orig_build

    print("\n== summary", flush=True)
    for name, shard, serial, ok, match_ok, wall in summary:
        print(
            f"   {name} ({shard}, {serial}): M0 {'holds' if ok else 'FAILS'}; calls matched {match_ok}; {wall:.1f} s",
            flush=True,
        )
    print(f"   all rows M0 holds: {all_ok and len(summary) == len(rows)}", flush=True)
    if run is not None:
        run.finish("done" if all_ok else "failed", exit_code=0 if all_ok else 2)
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
