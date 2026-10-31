"""
levin-phase-floor prompt 05 section 6: gate L0, before the run, on one depth-20 row and one
control (state/sample.json's "L0" names them; state/sample.md fixes which).

    python -I -B l0.py spec OUT_DIR      writes OUT_DIR/spec.json for worker.py: tree p, rtol 1e-7,
                                         the depth-20 row twice, then the control twice, in one
                                         process (two in-process runs of the same unit)
    python -I -B l0.py check OUT_DIR     reads OUT_DIR/ledger.jsonl and the pickles; prints the gate

Gate L0 (prompt 05 section 6), both clauses on both rows:
- (a) QuadSource recomputed at p equals A3 v2's stored QuadSource bit for bit at every node the
  stored row holds (all nine value columns, by == and by float.hex), and adds exactly B1's overlap
  nodes (the switch node of the earlier hand-over and HANDOVER_OVERLAP_NODES nodes below it, with
  the stored row ending at the node just above the switch node), and removes none. From rebuild.py's
  qs_check, made in each of the two runs.
- (b) the two in-process runs of the same unit at p are bit-identical: every float of the result
  (total, total_abserr, numeric_quad, WKB_Levin, analytic_rad, b) by float.hex and ==; every field
  of every sub-interval and group record of metadata["WKB_Levin"] and of every sub-interval record of
  metadata["numeric_quad"], except the wall-clock fields; the partition record; every group Levin
  call's summary except elapsed; and every pickled region array (start, end, depth, type, abserr)
  by its bytes. Only wall-clock fields are excluded: "elapsed", "compute_time", "wall_s",
  "rebuild_s".
"""

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import WIDER, header  # noqa: E402

WALL = {"elapsed", "compute_time", "wall_s", "rebuild_s"}


def spec(out_dir: Path):
    sample = json.loads((WIDER / "state" / "sample.json").read_text())
    by_name = {r["name"]: r for r in sample["rows"]}
    units = []
    for which in ("d20", "control"):
        row = by_name[sample["L0"][which]]
        for rep in (1, 2):
            units.append(
                {
                    "uid": f"L0-p-1e-7-{row['name']}-run{rep}",
                    "tree": "p",
                    "rtol": "1e-7",
                    "row": row["name"],
                    "shard": row["shard"],
                    "serial": row["serial"],
                    "role": row["role"],
                    "budget": None,
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    s = {
        "tree": "p",
        "units": units,
        "out_dir": str(out_dir),
        "emit_fd": None,
        "emit_file": str(out_dir / "ledger.jsonl"),
        "pickle_regions": True,
    }
    (out_dir / "spec.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


def strip(x):
    """Remove wall-clock fields, recursively."""
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in WALL}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def float_hex_walk(x):
    """Every float in a structure as float.hex (JSON floats are repr, which round-trips)."""
    if isinstance(x, dict):
        return {k: float_hex_walk(v) for k, v in x.items()}
    if isinstance(x, list):
        return [float_hex_walk(v) for v in x]
    if isinstance(x, float):
        return ("f", x.hex())
    return x


def diff_paths(a, b, path="", out=None):
    out = [] if out is None else out
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} vs {type(b).__name__}")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: present in one only")
            else:
                diff_paths(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{path}[{i}]", out)
    elif isinstance(a, float):
        if not (a.hex() == b.hex() and (a == b or (math.isnan(a) and math.isnan(b)))):
            out.append(f"{path}: {a!r} ({a.hex()}) vs {b!r} ({b.hex()})")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def check(out_dir: Path):
    import ray.cloudpickle as cloudpickle

    lines = header(Path(__file__).resolve(), "levin-phase-floor 05 section 6: gate L0")
    recs = [json.loads(l) for l in (out_dir / "ledger.jsonl").read_text().splitlines()]
    hello = [r for r in recs if r["kind"] == "hello"]
    units = [r["data"] for r in recs if r["kind"] == "unit"]
    lines.append(f"# ledger {out_dir / 'ledger.jsonl'}: {len(recs)} records")
    ok_all = True
    ev = hello[-1]["evidence"]
    lines.append("")
    lines.append("== the tree (worker evidence)")
    for k in ("tree", "tree_path", "expected_sha", "head", "all_in_tree",
              "PrimitivePhase_has_exact_remainder",
              "build_phase_source_mentions_exact_remainder", "PYTHONPATH",
              "dont_write_bytecode"):
        lines.append(f"   {k}: {ev[k]!r}")
    lines.append(f"   status --short: {'empty' if ev['status_short'].strip() == '' else repr(ev['status_short'])}")
    for k, v in ev["files"].items():
        lines.append(f"   {k}: {v}")
    tree_ok = (
        ev["tree"] == "p"
        and ev["head"] == ev["expected_sha"]
        and ev["all_in_tree"]
        and ev["status_short"].strip() == ""
        and not ev["PrimitivePhase_has_exact_remainder"]
    )
    lines.append(f"   the units ran at p, from the worktree, clean: {tree_ok}")
    ok_all &= tree_ok

    rows = {}
    for d in units:
        rows.setdefault(d["row"], []).append(d)
    for row, ds in rows.items():
        lines.append("")
        lines.append(f"== {row} (shard {ds[0]['shard']}, serial {ds[0]['serial']}, {ds[0]['role']})")
        for d in ds:
            lines.append(
                f"   {d['unit']}: status {d['status']}; wall {d['wall_s']}; "
                f"rebuild checks all hold {d.get('rebuild_checks_all_hold')}; dispatcher "
                f"{d['dispatcher_counts']} {d['dispatcher_flags']}; tree {d['tree_head']} "
                f"status {'empty' if d['tree_status_short'].strip() == '' else repr(d['tree_status_short'])}"
            )
            if d["status"] != "done":
                ok_all = False
                lines.append(f"   !! not done: {d.get('error')}")
                continue
            r = d["result"]
            lines.append(
                f"      total {r['total']['repr']} ({r['total']['hex']}), abserr "
                f"{r['total_abserr']['repr']}, converged {r['total_converged']}, "
                f"WKB_Levin_data {r['WKB_Levin_data']}"
            )
            s = d["stored"]
            lines.append(
                f"      stored: total {s['total']['repr']}, abserr {s['total_abserr']['repr']}, "
                f"WKB_Levin_data {s['WKB_Levin_data']} (descriptive: p is not expected to equal "
                f"the stored row, since QuadSource differs)"
            )
        # (a)
        lines.append("   -- (a) QuadSource recomputed at p against the stored row")
        for d in ds:
            q = d.get("qs_check")
            if q is None:
                ok_all = False
                lines.append(f"   !! {d['unit']}: no qs_check")
                continue
            lines.append(
                f"   {d['unit']}: stored serial {q['stored_serial']}; keyed as stored "
                f"{q['keyed_as_stored']} {q['Tq_Tr_serials']}; nodes stored {q['n_stored']}, "
                f"recomputed {q['n_recomputed']}, common {q['n_common']}; common (node, column) "
                f"pairs differing in bits: {q['n_common_differing_columns']} {q['differing_first']}; "
                f"removed {q['removed']}; added {q['added']} at z {q['added_z']}; B1's expected "
                f"{q['expected_added']} (HANDOVER_OVERLAP_NODES = {q['HANDOVER_OVERLAP_NODES']}); "
                f"added is B1's {q['added_is_B1s']}; the stored bottom is the node above the "
                f"switch node {q['old_bottom_is_node_above_switch']}; z_init q {q['z_init_q']!r}, "
                f"r {q['z_init_r']!r}; L0 (a) {'HOLDS' if q['L0a_holds'] else 'FAILS'}"
            )
            ok_all &= bool(q["L0a_holds"])
        # (b)
        lines.append("   -- (b) the two in-process runs, bit for bit")
        if len(ds) != 2 or any(d["status"] != "done" for d in ds):
            ok_all = False
            lines.append("   !! not two completed runs")
            continue
        a, b = ds
        comp = {}
        for key in ("result", "calls", "qs_check", "dispatcher_counts", "dispatcher_flags",
                    "n_calls_total", "stored"):
            comp[key] = diff_paths(
                float_hex_walk(strip(a[key])), float_hex_walk(strip(b[key])), key
            )
        # notes on what (b) compared
        n_floats = sum(1 for _ in _iter_floats(strip(a["result"])))
        lines.append(f"   floats compared in the result summary: {n_floats}")
        pa = cloudpickle.loads((out_dir / a["pickle"]).read_bytes())
        pb = cloudpickle.loads((out_dir / b["pickle"]).read_bytes())
        rdiff = []
        if len(pa["regions"]) != len(pb["regions"]):
            rdiff.append(f"region lists: {len(pa['regions'])} vs {len(pb['regions'])} calls")
        n_regions = 0
        for i, (x, y) in enumerate(zip(pa["regions"], pb["regions"])):
            for f in ("start", "end", "depth", "type", "abserr"):
                n_regions += len(x[f]) if f == "start" else 0
                if x[f].dtype != y[f].dtype or x[f].tobytes() != y[f].tobytes():
                    rdiff.append(f"call {i} ({x['regime']}, {x['group']}) {f} differs")
            if (x["regime"], x["group"]) != (y["regime"], y["group"]) or tuple(
                float(v).hex() for v in x["x_span"]
            ) != tuple(float(v).hex() for v in y["x_span"]):
                rdiff.append(f"call {i}: label or x_span differs")
        comp["regions"] = rdiff
        lines.append(f"   group Levin calls {len(pa['regions'])}; regions compared {n_regions}")
        same = all(len(v) == 0 for v in comp.values())
        for k, v in comp.items():
            lines.append(f"   {k}: {'identical' if not v else f'{len(v)} differences, first {v[:5]}'}")
        lines.append(f"   L0 (b) {'HOLDS' if same else 'FAILS'}")
        ok_all &= same
    lines.append("")
    lines.append(f"== GATE L0: {'HOLDS' if ok_all else 'FAILS'}")
    print("\n".join(lines))
    return 0 if ok_all else 1


def _iter_floats(x):
    if isinstance(x, dict):
        if set(x) == {"repr", "hex"}:  # worker.fx: a float as repr and hex
            yield x["hex"]
            return
        for v in x.values():
            yield from _iter_floats(v)
    elif isinstance(x, list):
        for v in x:
            yield from _iter_floats(v)
    elif isinstance(x, float):
        yield x


if __name__ == "__main__":
    cmd, out = sys.argv[1], Path(sys.argv[2])
    if cmd == "spec":
        spec(out)
    elif cmd == "check":
        sys.exit(check(out))
    else:
        raise SystemExit(f"unknown command {cmd}")
