"""
levin-phase-floor prompt 06 section 6: gate U0's clauses (b), (c) and (d), before the run. New in
06, derived from 05a's ``scripts/t0.py`` (docs/levin-phase-floor/05a/wider/scripts/t0.py, a record
that is not edited; its copy here is unused and unchanged): 05's helpers ``strip``,
``float_hex_walk``, ``diff_paths``, ``_iter_floats`` and 05a's ``_fx_same``, ``_regions_diff`` and
``_evidence_lines`` are copied unchanged; ``spec`` and ``check`` are rewritten for U0. Clause (a)
is u0_clause_a.py.

On one depth-20 row and one control, 05's and 05a's gate rows (``state/sample.json``'s "L0":
``low-G+r-1`` and ``ctrl-low-1``). Every unit carries the probe, so that it is exercised before the
launch (the run probes only R1-R7).

    python -I -B u0.py spec OUT_DIR    writes OUT_DIR/t5/spec.json (t5 at 1e-7: each row read from
                                       A3 v2, then each row with the producers re-run on A3 v2, in
                                       one process) and OUT_DIR/s/spec.json (s at 1e-7, producers
                                       re-run on the certified copy, each row twice in one process)
    python -I -B u0.py check OUT_DIR   reads both ledgers and pickles, and 05a's run; prints U0

- **(b) t5 at 1e-7 reproduces 05a's t unit bit for bit** (``t-1e-7-<row>`` of
  ``levin-phase-floor-05a-wider-20261026T123047``, prov.RUN05A): ``total``, ``total_abserr`` and
  every call's ``value`` and ``abserr``, by ``==`` and ``float.hex``; beyond the clause, the whole
  result summary, the call summaries, ``qs_check``, the dispatcher's counts and flags, the call
  count, the stored row's summary, and every region array of 05a's pickle by its bytes. 05a's
  probe was not 06's, so the probe is not compared.
- **(c) the producer re-run is faithful at the parent.** In the t5 units with ``"rerun": true``,
  on A3 v2: every re-run ``GkWKBIntegration`` value's ``theta_div_2pi``, ``theta_mod_2pi`` and
  ``G_WKB``, and every re-run ``TkWKBIntegration`` value's ``theta_div_2pi``, ``theta_mod_2pi``
  and ``T_WKB``, are bit-identical to A3 v2's (rerun.py compares by bit pattern), and every
  re-run object's coefficients are. Beyond the clause, and reported: the other value columns;
  the assembled GkSource against A3 v2's, column by column; the recomputed GkSourcePolicyData
  against A3 v2's; and the re-run unit's result, calls, ``qs_check``, dispatcher and regions
  against the t5 unit that read every producer value from A3 v2.
- **(d) two in-process runs of the same s unit are bit-identical**: as 05a's T0 (c), plus the
  re-run's record (timings aside), the GkSourcePolicyData, the probe's windows and every probe
  sample, and every re-run object's ``deltaTheta``.
- Both workers' tree evidence.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import RUN05A, WIDER, header, sha256  # noqa: E402

WALL = {"elapsed", "compute_time", "wall_s", "rebuild_s", "probe_s", "timing"}
COMPARED = ("result", "calls", "qs_check", "dispatcher_counts", "dispatcher_flags",
            "n_calls_total", "stored")


def spec(out_dir: Path):
    sample = json.loads((WIDER / "state" / "sample.json").read_text())
    by_name = {r["name"]: r for r in sample["rows"]}
    plans = {
        "t5": [("", False, None), ("rerun-", True, None)],
        "s": [("", True, 1), ("", True, 2)],
    }
    for tree, plan in plans.items():
        units = []
        for prefix, rerun, rep in plan:
            for which in ("d20", "control"):
                row = by_name[sample["L0"][which]]
                units.append(
                    {
                        "uid": f"U0-{tree}-{prefix}1e-7-{row['name']}" + (f"-run{rep}" if rep else ""),
                        "tree": tree,
                        "rtol": "1e-7",
                        "row": row["name"],
                        "shard": row["shard"],
                        "serial": row["serial"],
                        "role": row["role"],
                        "budget": None,
                        "probe": True,
                        "rerun": rerun,
                    }
                )
        if tree == "s":  # each row's two runs adjacent, as 05a's T0 ran them
            units.sort(key=lambda u: (u["row"] != sample["L0"]["d20"], u["uid"]))
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


# ---- 05a's t0.py helpers, unchanged ---------------------------------------------------------


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
    """06's probe samples: per factor, z and phi arrays by bytes, and k, sign, z_anchor."""
    if (sa is None) != (sb is None):
        return ["probe samples present in one only"], 0
    if sa is None:
        return [], 0
    out = []
    n = 0
    if sorted(sa) != sorted(sb):
        return [f"factors {sorted(sa)} vs {sorted(sb)}"], 0
    for f in sa:
        for key in ("z", "phi"):
            a, b = sa[f][key], sb[f][key]
            n += len(a)
            if a.dtype != b.dtype or a.tobytes() != b.tobytes():
                out.append(f"{f} {key} differs")
        for key in ("k", "sign", "z_anchor"):
            if repr(sa[f][key]) != repr(sb[f][key]):
                out.append(f"{f} {key} differs")
    return out, n


def _dtheta_diff(da, db):
    if (da is None) != (db is None):
        return ["deltaTheta present in one only"]
    if da is None:
        return []
    if sorted(da) != sorted(db):
        return ["deltaTheta keys differ"]
    out = []
    for k in da:
        x, y = da[k], db[k]
        if float(x["deltaTheta"]).hex() != float(y["deltaTheta"]).hex() or x["z_init"] != y["z_init"] or x["pure_WKB"] != y["pure_WKB"] or x["wkb_serial"] != y["wkb_serial"]:
            out.append(f"z_source {k}: {x} vs {y}")
    return out


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
        "levin-phase-floor 06 section 6: gate U0, clauses (b), (c) and (d)",
        [RUN05A / "checkpoint.jsonl", WIDER / "state" / "sample.json"],
    )
    ok_all = True
    ledgers = {}
    for tree in ("t5", "s"):
        recs = [json.loads(l) for l in (out_dir / tree / "ledger.jsonl").read_text().splitlines()]
        ledgers[tree] = recs
        lines.append(f"# ledger {out_dir / tree / 'ledger.jsonl'}: {len(recs)} records")
    lines.append("")
    for tree in ("t5", "s"):
        hello = [r for r in ledgers[tree] if r["kind"] == "hello"]
        ev = hello[-1]["evidence"]
        ok_all &= _evidence_lines(ev, tree, lines)
        want = {"t5": False, "s": True}[tree]
        rex = ev.get("producers_store_reduce_exactly")
        lines.append(f"   producers' store() reduce exactly: {rex} (expected {want} under {tree}); "
                     f"GkWKBValue has theta_residual: {ev.get('GkWKBValue_has_theta_residual')}")
        ok_all &= rex == {"GkWKBIntegration": want, "TkWKBIntegration": want}
        store = [r for r in ledgers[tree] if r["kind"] == "store"]
        lines.append(f"   store read: {store[-1]['primary'] if store else None}")
        lines.append("")

    units = {tree: {r["data"]["unit"]: r["data"] for r in ledgers[tree] if r["kind"] == "unit"} for tree in ("t5", "s")}
    for tree in ("t5", "s"):
        for uid, d in units[tree].items():
            lines.append(
                f"   {uid}: status {d['status']}; wall {d['wall_s']}; rebuild checks all hold "
                f"{d.get('rebuild_checks_all_hold')}; QuadSource L0a {d.get('qs_check', {}).get('L0a_holds')}; "
                f"dispatcher {d['dispatcher_counts']} {d['dispatcher_flags']}; T exact_remainder "
                f"{d.get('T_phase_exact_remainder')}; rerun {d.get('rerun_requested')}; policy warnings "
                f"{d.get('policy_warnings')}; probe windows {None if d.get('probe') is None else len(d['probe'])}, "
                f"error {d.get('probe_error')!r}; tree {d['tree_head']} status "
                f"{'empty' if d['tree_status_short'].strip() == '' else repr(d['tree_status_short'])}"
            )
            if d["status"] != "done" or d.get("probe_error") or (d.get("policy_warnings") or {}).get("count"):
                ok_all = False
                lines.append(f"   !! not done, probe error or policy warning: {d.get('error')} {d.get('probe_error')}")
            else:
                r = d["result"]
                lines.append(
                    f"      total {r['total']['repr']} ({r['total']['hex']}), abserr "
                    f"{r['total_abserr']['repr']} ({r['total_abserr']['hex']}), converged "
                    f"{r['total_converged']}, WKB_Levin_data {r['WKB_Levin_data']}; GkPolicy {d.get('GkPolicy')}"
                )
            flags = d["dispatcher_flags"] or {}
            fl_ok = all(k in ("T exact_remainder=True", "G exact_remainder=True") for k in flags)
            ter = d.get("T_phase_exact_remainder")
            fl_ok &= ter is not None and ter["Tq"] is True and ter["Tr"] is True
            lines.append(f"      every theta_mod_2pi call node-anchored (G and T): {fl_ok}")
            ok_all &= fl_ok

    def comp_units(a, b, pdir_a, pdir_b, keys):
        comp = {}
        for key in keys:
            comp[key] = diff_paths(float_hex_walk(strip(a.get(key))), float_hex_walk(strip(b.get(key))), key)
        pa = cloudpickle.loads((pdir_a / a["pickle"]).read_bytes())
        pb = cloudpickle.loads((pdir_b / b["pickle"]).read_bytes())
        comp["regions"], n_reg = _regions_diff(pa, pb)
        return comp, n_reg, pa, pb

    # ---------------------------------------------------------------- clause (b)
    lines.append("")
    lines.append("== U0 (b): t5 at 1e-7 against 05a's t unit, bit for bit "
                 f"({RUN05A.name}, checkpoint sha256 {sha256(RUN05A / 'checkpoint.jsonl')})")
    ref = {}
    for l in (RUN05A / "checkpoint.jsonl").read_text().splitlines():
        r = json.loads(l)
        ref[r["unit"]] = r
    clause_b = True
    for uid, d in units["t5"].items():
        if d.get("rerun_requested"):
            continue
        ru = ref.get(f"t-1e-7-{d['row']}")
        lines.append(f"-- {uid} against 05a's {ru['unit'] if ru else None}")
        if ru is None or d["status"] != "done":
            clause_b = False
            lines.append("   !! no reference, or the unit did not finish")
            continue
        rd = ru["data"]
        lines.append(f"   05a's record: script_sha256 {ru['script_sha256']}, tree {rd['tree_head']}, "
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
        comp, n_reg, _pa, _pb = comp_units(d, rd, out_dir / "t5", RUN05A, COMPARED)
        lines.append(f"   beyond the clause: floats in the result summary "
                     f"{sum(1 for _ in _iter_floats(strip(d['result'])))}; regions compared {n_reg}")
        for k, v in comp.items():
            lines.append(f"   {k}: {'identical' if not v else f'{len(v)} differences, first {v[:5]}'}")
        same_all = named_ok and all(len(v) == 0 for v in comp.values())
        lines.append(f"   U0 (b) on {d['row']}: {'HOLDS' if same_all else 'FAILS'}")
        clause_b &= same_all
    lines.append(f"== U0 (b): {'HOLDS' if clause_b else 'FAILS'}")
    ok_all &= clause_b

    # ---------------------------------------------------------------- clause (c)
    lines.append("")
    lines.append("== U0 (c): the producer re-run at the parent (t5) on A3 v2, against A3 v2's values")
    clause_c = True
    for uid, d in units["t5"].items():
        if not d.get("rerun_requested"):
            continue
        lines.append(f"-- {uid}")
        rr = d.get("rerun")
        if d["status"] != "done" or rr is None:
            clause_c = False
            lines.append("   !! not done")
            continue
        G, T = rr["G_values_vs_stored"], rr["T_values_vs_stored"]
        gnamed = {c: G["n_differing_by_column"][c] for c in ("theta_div_2pi", "theta_mod_2pi", "G_WKB")}
        tnamed = {c: T["n_differing_by_column"][c] for c in ("theta_div_2pi", "theta_mod_2pi", "T_WKB")}
        tcoeff = {k: v["coeffs_bit_identical_to_stored"] for k, v in rr["T_objects"].items()}
        clause = (
            G["n_values"] > 0 and T["n_values"] > 0
            and all(v == 0 for v in gnamed.values()) and all(v == 0 for v in tnamed.values())
            and rr["G_coeff_mismatch_count"] == 0 and all(tcoeff.values())
        )
        lines.append(f"   GkWKBIntegration: {rr['G_objects_rerun']} objects, {G['n_values']} values; "
                     f"values differing in the clause's columns {gnamed}; coefficient mismatches "
                     f"{rr['G_coeff_mismatch_count']}")
        lines.append(f"   TkWKBIntegration: {len(rr['T_objects'])} objects, {T['n_values']} values; "
                     f"differing {tnamed}; coefficients bit-identical {tcoeff}")
        lines.append(f"   the clause on {d['row']}: {'HOLDS' if clause else 'FAILS'}")
        lines.append(f"   beyond the clause (reported): G values bit-identical in every column "
                     f"{G['n_bit_identical_in_every_column']}/{G['n_values']}, differing by column "
                     f"{ {k: v for k, v in G['n_differing_by_column'].items() if v} }, first "
                     f"{G['first_differences'][:3]}; T {T['n_bit_identical_in_every_column']}/{T['n_values']}")
        GS = rr["GkSource_vs_stored"]
        lines.append(f"   GkSource assembled: {GS['n_bit_identical_in_every_column']}/{GS['n_values']} "
                     f"values bit-identical in every stored column; differing "
                     f"{ {k: v for k, v in GS['n_differing_by_column'].items() if v} }, first {GS['first_differences'][:3]}")
        P = rr["GkSourcePolicyData"]
        pol_same = all(P["rerun"][k] == P["stored"][k] for k in ("type", "quality", "crossover_z", "Levin_z", "metadata"))
        lines.append(f"   GkSourcePolicyData recomputed equal to A3 v2's (type, quality, crossover_z, "
                     f"Levin_z, metadata): {pol_same}  {P['rerun']} vs {P['stored']}")
        ref_uid = uid.replace("rerun-", "")
        rd = units["t5"].get(ref_uid)
        comp, n_reg, _pa, _pb = comp_units(d, rd, out_dir / "t5", out_dir / "t5", COMPARED + ("GkPolicy",))
        lines.append(f"   against {ref_uid} (every producer value read from A3 v2): regions compared {n_reg}")
        for k, v in comp.items():
            lines.append(f"   {k}: {'identical' if not v else f'{len(v)} differences, first {v[:5]}'}")
        chain_same = all(len(v) == 0 for v in comp.values())
        lines.append(f"   beyond the clause, the whole re-run chain reproduces the stored-input unit "
                     f"bit for bit: {chain_same}")
        clause_c &= clause
    lines.append(f"== U0 (c): {'HOLDS' if clause_c else 'FAILS'}")
    ok_all &= clause_c

    # ---------------------------------------------------------------- clause (d)
    lines.append("")
    lines.append("== U0 (d): two in-process runs of the same s unit, bit for bit")
    rows = {}
    for uid, d in units["s"].items():
        rows.setdefault(d["row"], []).append(d)
    clause_d = True
    for row, ds in rows.items():
        lines.append(f"-- {row}")
        if len(ds) != 2 or any(d["status"] != "done" for d in ds):
            clause_d = False
            lines.append("   !! not two completed runs")
            continue
        a, b = ds
        comp, n_reg, pa, pb = comp_units(a, b, out_dir / "s", out_dir / "s",
                                         COMPARED + ("probe", "T_phase_exact_remainder", "GkPolicy", "rerun"))
        comp["probe_samples"], n_ps = _probe_samples_diff(pa.get("probe_samples"), pb.get("probe_samples"))
        comp["deltaTheta"] = _dtheta_diff(pa.get("deltaTheta_by_z_source"), pb.get("deltaTheta_by_z_source"))
        lines.append(f"   floats in the result summary {sum(1 for _ in _iter_floats(strip(a['result'])))}; "
                     f"group Levin calls {len(pa['regions'])}; regions compared {n_reg}; probe windows "
                     f"{len(a['probe'] or [])}; probe sample values compared {n_ps}; re-run deltaTheta "
                     f"{len(pa.get('deltaTheta_by_z_source') or {})}")
        for k, v in comp.items():
            lines.append(f"   {k}: {'identical' if not v else f'{len(v)} differences, first {v[:5]}'}")
        same = all(len(v) == 0 for v in comp.values())
        lines.append(f"   U0 (d) on {row}: {'HOLDS' if same else 'FAILS'}")
        clause_d &= same
    lines.append(f"== U0 (d): {'HOLDS' if clause_d else 'FAILS'}")
    ok_all &= clause_d

    lines.append("")
    lines.append(f"== GATE U0, clauses (b), (c) and (d), with both trees' evidence: {'HOLDS' if ok_all else 'FAILS'}")
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
