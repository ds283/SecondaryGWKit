"""levin-phase-floor 06, finding P1: the shared rebuild of the two rows under one tree.

New for p1 (no counterpart in 06's scripts; its tree evidence is cut down from 06's
``wider/scripts/worker.py`` ``tree_evidence``, sha256 31e4399b..., a record, not edited).

``rebuilt_rows(tree_name)`` runs, in this process and without Ray (harness.py's stand-ins, a
read-only ``ShardedPool`` whose actors open each file ``mode=ro``):
  * the tree evidence: every imported production module comes from the tree on ``PYTHONPATH``,
    whose HEAD is the expected SHA with an empty ``git status --short``, and ``-B`` is in force;
  * rebuild.py's ``build_context`` and ``rebuild_row`` for each row on that tree's own store;
  * the row's ``QuadSourceIntegral.compute()``, as main.py runs it, with two transparent
    pass-through wrappers at the ``ComputeTargets.QuadSourceIntegral`` module attributes through
    which ``phase_group_Levin_integral`` calls them: ``build_phase_groups`` (keeps its arguments
    and the groups it returns) and ``adaptive_levin_sincos`` (keeps each call's arguments and
    result). The wrappers return exactly what they are given; the originals are restored before
    anything else runs;
  * the validation against the stored row: ``total``, ``total_abserr`` and every sub-interval's
    and group's value and abserr, by ``float.hex``.

It yields, per row, the result, the validation and the captured groups, for the analysis scripts
to use in the same process.
"""

import contextlib
import importlib
import inspect
import json
import os
import socket
import sqlite3
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ray  # noqa: E402

from prov import ROWS, STORE_STEM_BY_TREE, TREES, sha256, tree_state  # noqa: E402

SCRIPTS = ["p1lib.py", "harness.py", "rebuild.py", "prov.py"]


def fx(x):
    if x is None:
        return None
    x = float(x)
    return {"repr": repr(x), "hex": x.hex()}


def tree_evidence(tree_name):
    import ComputeTargets
    import StoreFixtures

    mods = {
        name: importlib.import_module(name)
        for name in [
            "ComputeTargets.primitive_phase",
            "ComputeTargets.QuadSource",
            "ComputeTargets.QuadSourceIntegral",
            "ComputeTargets.GkSourcePolicyData",
            "ComputeTargets.TkSourceFunctions",
            "ComputeTargets.phase_groups",
            "ComputeTargets.GkWKBIntegration",
            "ComputeTargets.TkWKBIntegration",
            "AdaptiveLevin.levin_quadrature",
        ]
    }
    from ComputeTargets.tests.test_main_plumbing import MAIN_PY

    tree, sha = TREES[tree_name]
    files = {"ComputeTargets": ComputeTargets.__file__, "StoreFixtures": StoreFixtures.__file__}
    files.update({k: v.__file__ for k, v in mods.items()})
    files["main.py"] = str(MAIN_PY)
    in_tree = {
        k: str(Path(v).resolve()).startswith(str(tree) + os.sep) for k, v in files.items()
    }
    head, status = tree_state(tree)
    tsf = mods["ComputeTargets.TkSourceFunctions"]
    gkw = mods["ComputeTargets.GkWKBIntegration"]
    return {
        "tree": tree_name,
        "tree_path": str(tree),
        "expected_sha": sha,
        "head": head,
        "status_short": status,
        "files": files,
        "all_in_tree": all(in_tree.values()),
        "in_tree": in_tree,
        "GkWKBValue_has_theta_residual": "theta_residual"
        in inspect.signature(gkw.GkWKBValue.__init__).parameters,
        "TkSourceFunctions_reads_theta_residual": "theta_residual"
        in inspect.getsource(tsf.TkSourceFunctions),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "dont_write_bytecode": sys.dont_write_bytecode,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "python": sys.version,
        "ray_initialized": ray.is_initialized(),
    }


def check_evidence(ev):
    ok = (
        ev["all_in_tree"]
        and ev["head"] == ev["expected_sha"]
        and ev["status_short"].strip() == ""
        and ev["dont_write_bytecode"]
        and not ev["ray_initialized"]
        and ev["PYTHONPATH"] == str(TREES[ev["tree"]][0])
    )
    if not ok:
        raise SystemExit(f"p1: tree evidence does not hold: {json.dumps(ev, indent=1)}")


def stored_metadata(shard_path, serial):
    conn = sqlite3.connect(f"{Path(shard_path).as_uri()}?mode=ro", uri=True)
    try:
        (md,) = conn.execute(
            'SELECT metadata FROM "QuadSourceIntegral" WHERE serial = ?', (serial,)
        ).fetchone()
    finally:
        conn.close()
    return json.loads(md)


def _same(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, (list, tuple, str)) or isinstance(b, (list, tuple, str)):
        return a == b
    return float(a).hex() == float(b).hex()


def validate(result, raw, md_stored):
    """Every compared quantity, recomputed vs stored, by float.hex (ints and bools by ==)."""
    items = []

    def add(name, mine, stored):
        items.append(
            {
                "name": name,
                "recomputed": fx(mine) if isinstance(mine, float) else mine,
                "stored": fx(stored) if isinstance(stored, float) else stored,
                "identical": _same(mine, stored),
            }
        )

    add("total", result["total"], raw["total"])
    add("total_abserr", result["total_abserr"], raw["total_abserr"])
    add("total_converged", bool(result["total_converged"]), bool(raw["total_converged"]))
    add("numeric_quad", result["numeric_quad"], raw["numeric_quad"])
    add("WKB_Levin", result["WKB_Levin"], raw["WKB_Levin"])
    md = result["metadata"]
    mine_si = md["WKB_Levin"]["subintervals"]
    st_si = md_stored["WKB_Levin"]["subintervals"]
    add("n_WKB_subintervals", len(mine_si), len(st_si))
    for i, (a, b) in enumerate(zip(mine_si, st_si)):
        for key in ("z_max", "z_min", "value", "abserr"):
            add(f"sub[{i}].{key}", float(a[key]), float(b[key]))
        add(f"sub[{i}].converged", bool(a["converged"]), bool(b["converged"]))
        add(f"sub[{i}].regime", list(a["regime"]), list(b["regime"]))
        for j, (g, h) in enumerate(zip(a["groups"], b["groups"])):
            add(f"sub[{i}].group[{j}].label", g["label"], h["label"])
            for key in ("value", "abserr"):
                add(f"sub[{i}].{g['label']}.{key}", float(g[key]), float(h[key]))
            for key in ("converged", "regions", "simple_regions", "evaluations"):
                add(f"sub[{i}].{g['label']}.{key}", g[key], h[key])
    add("n_quad_subintervals", len(md["numeric_quad"]["subintervals"]), len(md_stored["numeric_quad"]["subintervals"]))
    return {"all_identical": all(x["identical"] for x in items), "items": items}


@contextlib.contextmanager
def rebuilt_rows(tree_name, row_names=("row1", "row2"), log=print):
    """Yields ``(ev, ctx, rows)``; ``rows[name]`` holds the rebuild, the result, the validation,
    the captured groups and Levin calls. Everything runs inside ``standin_ray()``."""
    ev = tree_evidence(tree_name)
    check_evidence(ev)
    ev["scripts_sha256"] = {n: sha256(HERE / n) for n in SCRIPTS}

    from harness import QSI_MOD, close_pool, open_readonly_pool, standin_ray
    import rebuild as RB

    PRIMARY = RB.use_store(STORE_STEM_BY_TREE[tree_name])
    ev["primary"] = str(PRIMARY)
    log(f"# p1 {tree_name}: reads {PRIMARY} (read-only)")

    orig_levin = QSI_MOD.adaptive_levin_sincos
    orig_groups = QSI_MOD.build_phase_groups
    holder = {"levin": None, "groups": None}

    def w_levin(*a, **kw):
        result = orig_levin(*a, **kw)
        if holder["levin"] is not None:
            holder["levin"].append({"args": a, "kwargs": kw, "result": result})
        return result

    def w_groups(*a, **kw):
        result = orig_groups(*a, **kw)
        if holder["groups"] is not None:
            holder["groups"].append({"args": a, "kwargs": kw, "groups": result})
        return result

    with standin_ray() as cluster:
        pool, printed = open_readonly_pool(PRIMARY, cluster)
        log(printed.strip())
        try:
            t0 = time.perf_counter()
            ctx = build_context = RB.build_context(pool, log=log)
            log(f"# p1 {tree_name}: context in {time.perf_counter() - t0:.1f} s")
            rows = {}
            for name in row_names:
                shard, serial = ROWS[name][tree_name]
                t1 = time.perf_counter()
                rb = RB.rebuild_row(ctx, pool, shard, serial, log=log)
                levin_calls, group_calls = [], []
                holder.update(levin=levin_calls, groups=group_calls)
                QSI_MOD.adaptive_levin_sincos = w_levin
                QSI_MOD.build_phase_groups = w_groups
                try:
                    t2 = time.perf_counter()
                    result = ray.get(rb["work"].compute(payload=rb["compute_payload"]))
                    wall = time.perf_counter() - t2
                finally:
                    QSI_MOD.adaptive_levin_sincos = orig_levin
                    QSI_MOD.build_phase_groups = orig_groups
                    holder.update(levin=None, groups=None)
                assert QSI_MOD.adaptive_levin_sincos is orig_levin
                assert QSI_MOD.build_phase_groups is orig_groups
                md_stored = stored_metadata(RB.shard_file(shard), serial)
                val = validate(result, rb["raw"], md_stored)
                qs = rb["qs_check"]
                rows[name] = {
                    "shard": shard,
                    "serial": serial,
                    "rb": rb,
                    "result": result,
                    "md_stored": md_stored,
                    "validation": val,
                    "levin_calls": levin_calls,
                    "group_calls": group_calls,
                    "compute_wall_s": wall,
                    "rebuild_s": t2 - t1,
                    "rebuild_checks_all_hold": all(c[1] for c in rb["checks"]),
                    "qs_check_summary": {
                        k: qs[k]
                        for k in (
                            "stored_serial",
                            "keyed_as_stored",
                            "n_stored",
                            "n_recomputed",
                            "n_common",
                            "n_common_differing_columns",
                            "removed",
                            "added",
                        )
                    },
                }
                log(
                    f"# p1 {tree_name} {name}: total {result['total']!r} (stored "
                    f"{rb['raw']['total']!r}); all identical {val['all_identical']}; compute "
                    f"{wall:.1f} s; {len(levin_calls)} Levin calls; QuadSource recomputed vs "
                    f"stored differing columns {qs['n_common_differing_columns']}"
                )
            yield ev, ctx, rows
        finally:
            log(close_pool(pool, cluster).strip())
    if ray.is_initialized():
        raise SystemExit("p1: Ray was initialised")


def low_subinterval_groups(row):
    """The captured build_phase_groups call for the regime (True, True, True) sub-interval, its
    groups by label, and the Levin call record for each group (in call order)."""
    calls = [c for c in row["group_calls"] if tuple(c["args"][0]) == (True, True, True)]
    if len(calls) != 1:
        raise RuntimeError(f"p1: expected one all-oscillatory build_phase_groups call, got {len(calls)}")
    gc = calls[0]
    groups = {g.label: g for g in gc["groups"]}
    levin = {}
    for c in row["levin_calls"]:
        label = c["kwargs"].get("notify_label") or ""
        if "regime Gqr," in label:
            glabel = label.rsplit("group ", 1)[1]
            levin[glabel] = c
    if set(levin) != set(groups):
        raise RuntimeError(f"p1: groups {sorted(groups)} vs Levin calls {sorted(levin)}")
    return gc, groups, levin
