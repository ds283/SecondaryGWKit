"""
levin-phase-floor prompt 05a section 6: gate T0's second and third clauses, before the run. New in
05a, derived from prompt 05's ``scripts/l0.py`` (docs/levin-phase-floor/05/wider/scripts/l0.py,
sha256 b2b48660..., a record that is not edited; its copy here is unused and unchanged): its
helpers ``strip``, ``float_hex_walk``, ``diff_paths`` and ``_iter_floats`` are copied unchanged, and
its ``spec`` and ``check`` are rewritten for T0's clauses. T0's first clause (the ``git diff
--stat``) is a command of its own (out/cmd/05_T0_clause1_git_diff.txt).

On one depth-20 row and one control: ``state/sample.json``'s "L0" names them (05's gate rows,
``low-G+r-1`` and ``ctrl-low-1``).

    python -I -B t0.py spec OUT_DIR    writes OUT_DIR/c/spec.json (tree c, rtol 1e-7, each row once)
                                       and OUT_DIR/t/spec.json (tree t, rtol 1e-7, each row twice, in
                                       one process: two in-process runs of the same unit), for
                                       worker.py; every unit carries the probe, as the run's 1e-7
                                       units do
    python -I -B t0.py check OUT_DIR   reads both ledgers and pickles, and 05's run; prints the gate

- **Clause (b), c reproduces 05's c bit for bit.** Each c unit against 05's ``c-1e-7-<row>`` record
  of ``levin-phase-floor-05-wider-20261026T020005`` (prov.RUN05): ``total``, ``total_abserr``, and
  every group call's ``value`` and ``abserr``, each by ``==`` and ``float.hex``, never ``isclose``;
  and, beyond what the clause names, the whole result summary, every call summary, ``qs_check``,
  the dispatcher's counts and flags, the call count, the stored row's summary (all but wall-clock
  fields), and every region array of 05's pickle by its bytes.
- **Clause (c), two in-process runs of the same t unit are bit-identical:** as 05's L0 (b) (every
  float of the result summary, every call summary, ``qs_check``, the dispatcher's counts and
  flags, the region arrays by bytes), and also the probe's window summaries and every probe sample
  array by its bytes.
- The tree evidence of both workers: each loaded its packages from its own tree, at its expected
  SHA, with an empty ``git status --short``; and T's phases report ``exact_remainder`` false under
  c and true under t.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import RUN05, WIDER, header, sha256  # noqa: E402

WALL = {"elapsed", "compute_time", "wall_s", "rebuild_s", "probe_s"}
COMPARED = ("result", "calls", "qs_check", "dispatcher_counts", "dispatcher_flags",
            "n_calls_total", "stored")


def spec(out_dir: Path):
    sample = json.loads((WIDER / "state" / "sample.json").read_text())
    by_name = {r["name"]: r for r in sample["rows"]}
    for tree, reps in (("c", (1,)), ("t", (1, 2))):
        units = []
        for which in ("d20", "control"):
            row = by_name[sample["L0"][which]]
            for rep in reps:
                units.append(
                    {
                        "uid": f"T0-{tree}-1e-7-{row['name']}" + (f"-run{rep}" if tree == "t" else ""),
                        "tree": tree,
                        "rtol": "1e-7",
                        "row": row["name"],
                        "shard": row["shard"],
                        "serial": row["serial"],
                        "role": row["role"],
                        "budget": None,
                        "probe": True,
                    }
                )
        d = out_dir / tree
        d.mkdir(parents=True, exist_ok=True)
        s = {
            "tree": tree,
            "units": units,
            "out_dir": str(d),
            "emit_fd": None,
            "emit_file": str(d / "ledger.jsonl"),
            "pickle_regions": True,
        }
        (d / "spec.json").write_text(json.dumps(s, indent=1))
        print(json.dumps(s, indent=1))


# ---- 05's l0.py helpers, unchanged ------------------------------------------------------------


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


# ---- T0 ----------------------------------------------------------------------------------------


def _fx_same(a, b):
    """Two worker.fx floats: equal by == and by float.hex (NaN equal to NaN only by hex)."""
    if a is None or b is None:
        return a is None and b is None, f"{a} vs {b}"
    x, y = float.fromhex(a["hex"]), float.fromhex(b["hex"])
    same = x.hex() == y.hex() and (x == y or (math.isnan(x) and math.isnan(y)))
    return same, f"{a['repr']} ({a['hex']}) vs {b['repr']} ({b['hex']})"


def _regions_diff(pa, pb):
    rdiff = []
    if len(pa["regions"]) != len(pb["regions"]):
        rdiff.append(f"region lists: {len(pa['regions'])} vs {len(pb['regions'])} calls")
    n = 0
    for i, (x, y) in enumerate(zip(pa["regions"], pb["regions"])):
        n += len(x["start"])
        for f in ("start", "end", "depth", "type", "abserr"):
            if x[f].dtype != y[f].dtype or x[f].tobytes() != y[f].tobytes():
                rdiff.append(f"call {i} ({x['regime']}, {x['group']}) {f} differs")
        if (x["regime"], x["group"]) != (y["regime"], y["group"]) or tuple(
            float(v).hex() for v in x["x_span"]
        ) != tuple(float(v).hex() for v in y["x_span"]):
            rdiff.append(f"call {i}: label or x_span differs")
    return rdiff, n


def _probe_samples_diff(sa, sb):
    out = []
    n = 0
    if (sa is None) != (sb is None):
        return ["probe samples present in one only"], 0
    if sa is None:
        return [], 0
    if len(sa) != len(sb):
        return [f"windows {len(sa)} vs {len(sb)}"], 0
    for w, (la, lb) in enumerate(zip(sa, sb)):
        if len(la) != len(lb):
            out.append(f"window {w}: levels {len(la)} vs {len(lb)}")
            continue
        for lev, (xa, xb) in enumerate(zip(la, lb)):
            if (xa is None) != (xb is None):
                out.append(f"window {w} level {lev}: present in one only")
                continue
            if xa is None:
                continue
            for f in ("z", "theta_z", "p_z", "node", "u", "theta_u"):
                if (xa[f] is None) != (xb[f] is None):
                    out.append(f"window {w} level {lev} {f}: present in one only")
                elif xa[f] is not None:
                    n += len(xa[f])
                    if xa[f].dtype != xb[f].dtype or xa[f].tobytes() != xb[f].tobytes():
                        out.append(f"window {w} level {lev} {f} differs")
    return out, n


def _evidence_lines(ev, tree, lines):
    lines.append(f"== the {tree} worker's evidence")
    for k in ("tree", "tree_path", "expected_sha", "head", "all_in_tree",
              "PrimitivePhase_has_exact_remainder", "build_phase_source_mentions_exact_remainder",
              "TkSourceFunctions_exact_remainder_default", "PYTHONPATH", "dont_write_bytecode"):
        lines.append(f"   {k}: {ev.get(k)!r}")
    lines.append(f"   status --short: {'empty' if ev['status_short'].strip() == '' else repr(ev['status_short'])}")
    for k, v in ev["files"].items():
        lines.append(f"   {k}: {v}")
    ok = (
        ev["tree"] == tree
        and ev["head"] == ev["expected_sha"]
        and ev["all_in_tree"]
        and ev["status_short"].strip() == ""
        and ev["dont_write_bytecode"]
    )
    lines.append(f"   the units ran at {tree}, from its own tree, clean: {ok}")
    return ok


def check(out_dir: Path):
    import ray.cloudpickle as cloudpickle

    lines = header(
        Path(__file__).resolve(),
        "levin-phase-floor 05a section 6: gate T0, clauses (b) and (c)",
        [RUN05 / "checkpoint.jsonl", WIDER / "state" / "sample.json"],
    )
    ok_all = True
    ledgers = {}
    for tree in ("c", "t"):
        recs = [json.loads(l) for l in (out_dir / tree / "ledger.jsonl").read_text().splitlines()]
        ledgers[tree] = recs
        lines.append(f"# ledger {out_dir / tree / 'ledger.jsonl'}: {len(recs)} records")
    lines.append("")
    for tree in ("c", "t"):
        hello = [r for r in ledgers[tree] if r["kind"] == "hello"]
        ok_all &= _evidence_lines(hello[-1]["evidence"], tree, lines)
        lines.append("")

    units = {tree: [r["data"] for r in ledgers[tree] if r["kind"] == "unit"] for tree in ("c", "t")}
    for tree in ("c", "t"):
        for d in units[tree]:
            lines.append(
                f"   {d['unit']}: status {d['status']}; wall {d['wall_s']}; rebuild checks all hold "
                f"{d.get('rebuild_checks_all_hold')}; QuadSource L0a {d.get('qs_check', {}).get('L0a_holds')}; "
                f"dispatcher {d['dispatcher_counts']} {d['dispatcher_flags']}; T phase exact_remainder "
                f"{d.get('T_phase_exact_remainder')}; probe windows "
                f"{None if d.get('probe') is None else len(d['probe'])} in {d.get('probe_s')} s, error "
                f"{d.get('probe_error')!r}; tree {d['tree_head']} status "
                f"{'empty' if d['tree_status_short'].strip() == '' else repr(d['tree_status_short'])}"
            )
            if d["status"] != "done" or d.get("probe_error"):
                ok_all = False
                lines.append(f"   !! not done or probe error: {d.get('error')} {d.get('probe_error')}")
            else:
                r = d["result"]
                lines.append(
                    f"      total {r['total']['repr']} ({r['total']['hex']}), abserr "
                    f"{r['total_abserr']['repr']} ({r['total_abserr']['hex']}), converged "
                    f"{r['total_converged']}, WKB_Levin_data {r['WKB_Levin_data']}"
                )
            want = {"c": False, "t": True}[tree]
            ter = d.get("T_phase_exact_remainder")
            flag_ok = ter is not None and ter["Tq"] is want and ter["Tr"] is want
            flags = d["dispatcher_flags"] or {}
            flag_keys_ok = all(
                (k.startswith("T ") and k == f"T exact_remainder={want}")
                or (k.startswith("G ") and k == "G exact_remainder=True")
                for k in flags
            )
            probe_flags = {w["exact_remainder"] for w in (d.get("probe") or [])}
            lines.append(
                f"      T's phases report exact_remainder {want} under {tree}: {flag_ok}; every "
                f"theta_mod_2pi call of T has exact_remainder={want} and of G True: "
                f"{flag_keys_ok}; probe windows' flags {sorted(probe_flags)}"
            )
            ok_all &= flag_ok and flag_keys_ok and probe_flags <= {want}

    # ---------------------------------------------------------------- clause (b)
    lines.append("")
    lines.append("== T0 (b): c at 1e-7 against 05's c unit, bit for bit "
                 f"({RUN05.name}, checkpoint sha256 {sha256(RUN05 / 'checkpoint.jsonl')})")
    ref = {}
    for l in (RUN05 / "checkpoint.jsonl").read_text().splitlines():
        r = json.loads(l)
        ref[r["unit"]] = r
    clause_b = True
    for d in units["c"]:
        ru = ref.get(f"c-1e-7-{d['row']}")
        lines.append(f"-- {d['unit']} against 05's {ru['unit'] if ru else None}")
        if ru is None or d["status"] != "done":
            clause_b = False
            lines.append("   !! no reference, or the unit did not finish")
            continue
        rd = ru["data"]
        lines.append(f"   05's record: script_sha256 {ru['script_sha256']}, tree {rd['tree_head']}, "
                     f"pickle {rd['pickle']} sha256 {rd['pickle_sha256']}")
        named = []
        for f in ("total", "total_abserr"):
            same, txt = _fx_same(d["result"][f], rd["result"][f])
            named.append(same)
            lines.append(f"   {f}: {'==' if same else 'DIFFERS'}  {txt}")
        if len(d["calls"]) != len(rd["calls"]):
            named.append(False)
            lines.append(f"   calls: {len(d['calls'])} vs {len(rd['calls'])}")
        for ca, cb in zip(d["calls"], rd["calls"]):
            for f in ("value", "abserr"):
                same, txt = _fx_same(ca[f], cb[f])
                named.append(same)
                lines.append(f"   call {ca['index']} {ca['regime']}/{ca['group']} {f}: "
                             f"{'==' if same else 'DIFFERS'}  {txt}")
        named_ok = all(named)
        lines.append(f"   the clause's own fields ({len(named)} doubles): "
                     f"{'all bit-identical' if named_ok else 'NOT all bit-identical'}")
        comp = {}
        for key in COMPARED:
            comp[key] = diff_paths(float_hex_walk(strip(d[key])), float_hex_walk(strip(rd[key])), key)
        pa = cloudpickle.loads((out_dir / "c" / d["pickle"]).read_bytes())
        pb = cloudpickle.loads((RUN05 / rd["pickle"]).read_bytes())
        comp["regions"], n_reg = _regions_diff(pa, pb)
        lines.append(f"   beyond the clause: floats in the result summary "
                     f"{sum(1 for _ in _iter_floats(strip(d['result'])))}; regions compared {n_reg}")
        for k, v in comp.items():
            lines.append(f"   {k}: {'identical' if not v else f'{len(v)} differences, first {v[:5]}'}")
        same_all = named_ok and all(len(v) == 0 for v in comp.values())
        lines.append(f"   T0 (b) on {d['row']}: {'HOLDS' if same_all else 'FAILS'}")
        clause_b &= same_all
    lines.append(f"== T0 (b): {'HOLDS' if clause_b else 'FAILS'}")
    ok_all &= clause_b

    # ---------------------------------------------------------------- clause (c)
    lines.append("")
    lines.append("== T0 (c): two in-process runs of the same t unit, bit for bit")
    rows = {}
    for d in units["t"]:
        rows.setdefault(d["row"], []).append(d)
    clause_c = True
    for row, ds in rows.items():
        lines.append(f"-- {row}")
        if len(ds) != 2 or any(d["status"] != "done" for d in ds):
            clause_c = False
            lines.append("   !! not two completed runs")
            continue
        a, b = ds
        comp = {}
        for key in COMPARED + ("probe", "T_phase_exact_remainder"):
            comp[key] = diff_paths(float_hex_walk(strip(a[key])), float_hex_walk(strip(b[key])), key)
        pa = cloudpickle.loads((out_dir / "t" / a["pickle"]).read_bytes())
        pb = cloudpickle.loads((out_dir / "t" / b["pickle"]).read_bytes())
        comp["regions"], n_reg = _regions_diff(pa, pb)
        comp["probe_samples"], n_ps = _probe_samples_diff(pa.get("probe_samples"), pb.get("probe_samples"))
        lines.append(f"   floats in the result summary {sum(1 for _ in _iter_floats(strip(a['result'])))}; "
                     f"group Levin calls {len(pa['regions'])}; regions compared {n_reg}; probe windows "
                     f"{len(a['probe'] or [])}; probe sample values compared {n_ps}")
        for k, v in comp.items():
            lines.append(f"   {k}: {'identical' if not v else f'{len(v)} differences, first {v[:5]}'}")
        same = all(len(v) == 0 for v in comp.values())
        lines.append(f"   T0 (c) on {row}: {'HOLDS' if same else 'FAILS'}")
        clause_c &= same
    lines.append(f"== T0 (c): {'HOLDS' if clause_c else 'FAILS'}")
    ok_all &= clause_c

    lines.append("")
    lines.append(f"== GATE T0, clauses (b) and (c), with both trees' evidence: {'HOLDS' if ok_all else 'FAILS'}")
    print("\n".join(lines))
    return 0 if ok_all else 1


if __name__ == "__main__":
    cmd, out = sys.argv[1], Path(sys.argv[2])
    if cmd == "spec":
        spec(out)
    elif cmd == "check":
        sys.exit(check(out))
    else:
        raise SystemExit(f"unknown command {cmd}")
