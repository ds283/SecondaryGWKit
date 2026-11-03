"""
levin-phase-floor prompt 05a section 6 (T1 as amended by section 9): score the run from its
checkpoints and pickles. Copied from prompt 05's ``scripts/analyse.py``
(docs/levin-phase-floor/05/wider/scripts/analyse.py, sha256 675d6bc7..., a record that is not
edited), and rewritten for 05a's rules (prompt 05a: "Change only those [05's paths, trees, unit
sets], and whatever the probe below needs"; 05's L1-L3 are not 05a's rules, so its scoring code
is replaced). Kept from 05: ``F``, ``le_exact``, ``g``, the per-row table, the group-sequence check
and the band/cost summary, with 05's trees p/c read as c/t.

Reads the run's checkpoint.jsonl and unit_*.pkl, state/sample.json, and 05's run's checkpoint
(prov.RUN05: which rows were capped under B1's code, i.e. 05's p at depth 20). Runs nothing;
imports nothing from either tree.

Rules (state/predictions.md, fixed before the run, with its "Readings fixed before the run"):
- **T1 (section 9).** Windows are paired by (row, sub-interval, factor) between c and t, and
  checked to start at the same double. The level used is the first n at which both trees' level
  is distinct and both trees' largest unwrapped z-sampled step is below pi/2; none: unresolved.
  In every resolved window with eps|theta_T| >= 1e-11 rad: sigma_c >= 0.1 eps|theta_T| and
  sigma_t <= max(1e-3 eps|theta_T|, eps (k max|p_z|_t + 2 pi), 1e-13). Holds if true in every such
  window; fails if false in any; undecided if none false but some such window unresolved.
  **Reported apart, unscored:** the same rule at the level chosen by the true step (the first n
  at which also |d theta/dz| (b - a)/64 < pi/2, which is the same double in both trees since
  theta_deriv is unchanged).
- **T2.** Every row (39), t at 1e-7: no group has a region at depth >= 18. Over budget: false.
- **T3.** Every row (39): |total_t - total_c| <= total_abserr_c at 1e-7. A row whose t@1e-7 unit is
  over budget is undecided.
- **T4.** Every depth-20 row (31 stored at depth 20) with both t units in budget:
  |total_t(1e-7) - total_t(1e-8)| <= total_abserr_t(1e-7).
The inequalities of T3/T4 are evaluated exactly on the doubles (fractions.Fraction); an infinite
abserr makes a rule true, a NaN false. The numbers printed are doubles.

Usage: python -I -B analyse.py RUN_DIR OUT_JSON PROBE_JSON   (text to stdout)
"""

import collections
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import ray.cloudpickle as cloudpickle  # noqa: E402

from prov import RUN05, TREES, WIDER, header  # noqa: E402

R03 = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
EPS = float(np.finfo(np.float64).eps)
HALF_PI = 0.5 * math.pi
T1_MIN_SCALE = 1.0e-11
TWO_PI = 2.0 * math.pi


def F(d):
    return None if d is None else float.fromhex(d["hex"])


def le_exact(lhs_a, lhs_b, rhs):
    """|a - b| <= rhs, exactly on the doubles. Returns (bool, |a-b| as float, ratio)."""
    if any(x is None for x in (lhs_a, lhs_b, rhs)):
        return None, None, None
    if any(math.isnan(x) for x in (lhs_a, lhs_b, rhs)):
        return False, float("nan"), float("nan")
    if math.isinf(lhs_a) or math.isinf(lhs_b):
        return False, float("inf"), float("inf")
    d = abs(Fraction(lhs_a) - Fraction(lhs_b))
    df = float(d)
    if math.isinf(rhs):
        return True, df, 0.0
    ok = d <= Fraction(rhs)
    ratio = (df / rhs) if rhs != 0 else (0.0 if df == 0 else float("inf"))
    return ok, df, ratio


def g(x, fmt=".3g"):
    if x is None:
        return "-"
    if isinstance(x, float):
        return format(x, fmt)
    return str(x)


def fit_sigma(x, vals, a, b):
    """test_t_phase_reanchor._sigma_theta's fit (probe.py's _fit), for re-computation."""
    th = np.unwrap(vals)
    t = (2.0 * x - (a + b)) / (b - a)
    coef = np.polynomial.polynomial.polyfit(t, th, 4)
    res = th - np.polynomial.polynomial.polyval(t, coef)
    return float(np.sqrt(np.mean(res * res))), float(np.max(np.abs(np.diff(th))))


def hexlist(a):
    return None if a is None else [float(v).hex() for v in a]


def main():
    run_dir = Path(sys.argv[1])
    out_json = Path(sys.argv[2])
    probe_json = Path(sys.argv[3])
    lines = header(
        Path(__file__).resolve(),
        "levin-phase-floor 05a section 6: the wider re-measurement, scored (T1 as amended by section 9)",
        [run_dir / "checkpoint.jsonl", WIDER / "state" / "sample.json", RUN05 / "checkpoint.jsonl"],
    )
    out = lines.append
    sample = json.loads((WIDER / "state" / "sample.json").read_text())
    rows = sample["rows"]

    recs = [json.loads(l) for l in (run_dir / "checkpoint.jsonl").read_text().splitlines()]
    units = {r["unit"]: r for r in recs}
    out(f"# run {run_dir.name}: {len(recs)} checkpoint records")

    # ------------------------------------------------------------------ provenance
    prov_bad = []
    for r in recs:
        d = r["data"]
        if d["tree_head"] != TREES[d["tree"]][1] or d["tree_status_short"].strip() != "":
            prov_bad.append(r["unit"])
    shas = collections.Counter(r["script_sha256"] for r in recs)
    out(f"# every record's script_sha256: {dict(shas)}; tree HEAD and status as expected on "
        f"every record: {not prov_bad} {prov_bad}")
    flags = collections.Counter()
    for r in recs:
        d = r["data"]
        for k, v in (d.get("dispatcher_flags") or {}).items():
            flags[(d["tree"], k)] += v
    out(f"# theta_mod_2pi calls by tree and phase flag: {dict(flags)}")
    ter = collections.Counter(
        (r["data"]["tree"], json.dumps(r["data"].get("T_phase_exact_remainder"))) for r in recs
    )
    out(f"# T's phase exact_remainder per unit, by tree: {dict(ter)}")
    qs_bad = [r["unit"] for r in recs if not r["data"]["qs_check"]["L0a_holds"]]
    out(f"# QuadSource recomputed equals the stored row at its nodes and adds B1's 3 nodes in every "
        f"unit: {not qs_bad} {qs_bad}")
    pe = [r["unit"] for r in recs if r["data"].get("probe_error")]
    out(f"# probe errors: {pe if pe else 'none'}")

    def unit(tree, rtol, name):
        r = units.get(f"{tree}-{rtol}-{name}")
        return None if r is None else r["data"]

    def regions(d):
        if d is None or d["status"] != "done":
            return None
        w = d["result"]["WKB_Levin_data"]
        return None if w is None else w["num_regions"]

    def maxdepth(d):
        if d is None or d["status"] != "done":
            return None
        w = d["result"]["WKB_Levin_data"]
        return None if w is None else w["max_depth"]

    def load_pkl(d):
        return cloudpickle.loads((run_dir / d["pickle"]).read_bytes())

    over = [r["unit"] for r in recs if r["data"]["status"] == "over budget"]
    errs = [r["unit"] for r in recs if r["data"]["status"] == "error"]
    out(f"# over-budget units: {over if over else 'none'}; error units: {errs if errs else 'none'}")
    worst = max(
        (r["data"]["wall_s"] / r["data"]["budget_s"], r["unit"])
        for r in recs
        if r["data"]["tree"] == "t" and r["data"]["budget_s"]
    )
    out(f"# largest t wall / budget: {worst[0]:.3f} ({worst[1]})")

    # the rows capped under B1's code: 05's p at depth 20
    r05 = {}
    for l in (RUN05 / "checkpoint.jsonl").read_text().splitlines():
        rr = json.loads(l)
        r05[rr["unit"]] = rr["data"]
    d20 = [r["name"] for r in rows if r["role"] == "d20"]
    capped = [
        n for n in d20
        if r05[f"p-1e-7-{n}"]["status"] == "done"
        and (r05[f"p-1e-7-{n}"]["result"]["WKB_Levin_data"] or {}).get("max_depth") == 20
    ]
    out(f"# rows capped under B1's code (05's p at depth 20, from {RUN05.name}): {len(capped)}: {capped}")

    # ------------------------------------------------------------------ per-row table
    res = {}
    out("")
    out("== per row: c and t at rtol 1e-7, t at 1e-8 (regions = WKB_Levin_data.num_regions; "
        "d = max depth; wall in s)")
    out("   name           cls  band  stored(d,reg)  c7(d,reg,wall)       t7(d,reg,wall,budget)        "
        "t8(d,reg,wall,status)   conv c7/t7/t8")
    for r in rows:
        name = r["name"]
        c7, t7, t8 = unit("c", "1e-7", name), unit("t", "1e-7", name), unit("t", "1e-8", name)
        sw = c7["stored"]["WKB_Levin_data"] or {}
        e = {
            "name": name, "role": r["role"], "cls": r["cls"], "band": r["band"], "k": float(r["k"]),
            "z_response": r["z_response"], "capped_B1": name in capped,
            "stored_depth": sw.get("max_depth"), "stored_regions": sw.get("num_regions"),
            "c7_depth": maxdepth(c7), "c7_regions": regions(c7), "c7_wall": c7["wall_s"],
            "c7_status": c7["status"],
            "t7_depth": maxdepth(t7), "t7_regions": regions(t7), "t7_wall": t7["wall_s"],
            "t7_budget": t7["budget_s"], "t7_status": t7["status"],
            "c7_conv": c7["result"]["total_converged"] if c7["status"] == "done" else None,
            "t7_conv": t7["result"]["total_converged"] if t7["status"] == "done" else None,
        }
        if t8 is not None:
            e.update(t8_depth=maxdepth(t8), t8_regions=regions(t8), t8_wall=t8["wall_s"],
                     t8_budget=t8["budget_s"], t8_status=t8["status"],
                     t8_conv=t8["result"]["total_converged"] if t8["status"] == "done" else None)
        if c7["status"] == "done":
            e.update(total_c7=F(c7["result"]["total"]), abserr_c7=F(c7["result"]["total_abserr"]))
        if t7["status"] == "done":
            e.update(total_t7=F(t7["result"]["total"]), abserr_t7=F(t7["result"]["total_abserr"]))
        if t8 is not None and t8["status"] == "done":
            e.update(total_t8=F(t8["result"]["total"]), abserr_t8=F(t8["result"]["total_abserr"]))
        res[name] = e
        out(
            f"   {name:<14s} {str(r['cls']):4s} {r['band']:4s}  "
            f"({g(e['stored_depth'])},{g(e['stored_regions'])})".ljust(44)
            + f"({g(e['c7_depth'])},{g(e['c7_regions'])},{e['c7_wall']:.2f})".ljust(21)
            + f"({g(e['t7_depth'])},{g(e['t7_regions'])},{e['t7_wall']:.2f},{g(e['t7_budget'])} {e['t7_status']})".ljust(29)
            + (f"({g(e.get('t8_depth'))},{g(e.get('t8_regions'))},{e['t8_wall']:.2f},{e['t8_status']})".ljust(24) if t8 is not None else "-".ljust(24))
            + f"{e['c7_conv']}/{e['t7_conv']}/{e.get('t8_conv', '-')}"
        )

    # ------------------------------------------------------------------ groups, c vs t
    group_tables = {}
    seq_ok = {}
    for r in rows:
        name = r["name"]
        c7, t7 = unit("c", "1e-7", name), unit("t", "1e-7", name)
        gc = [((c["regime"], c["group"]), c) for c in c7["calls"]]
        gt = [((c["regime"], c["group"]), c) for c in t7["calls"]] if t7["status"] == "done" else []
        seq_ok[name] = [(k, a["x_span"][0]["hex"], a["x_span"][1]["hex"]) for k, a in gc] == [
            (k, a["x_span"][0]["hex"], a["x_span"][1]["hex"]) for k, a in gt]
        pc = load_pkl(c7)
        pt = load_pkl(t7) if t7["status"] == "done" else {"regions": []}
        tab = []
        for i, (k, a) in enumerate(gc):
            b = gt[i][1] if i < len(gt) else None
            low = {}
            for tree, pk in (("c", pc), ("t", pt)):
                if i >= len(pk["regions"]):
                    low[tree] = (None, None)
                    continue
                reg = pk["regions"][i]
                x0, x1 = reg["x_span"]
                cut = x0 + 0.1 * (x1 - x0)
                m = reg["start"] < cut
                low[tree] = (int(m.sum()), int(reg["depth"][m].max()) if m.any() else None)
            tab.append({
                "index": a["index"], "regime": k[0], "group": k[1],
                "x_span": [a["x_span"][0]["repr"], a["x_span"][1]["repr"]],
                "c_regions": a["num_regions"], "c_maxd": a["max_depth"], "c_n18": a["n_depth_ge_18"],
                "t_regions": None if b is None else b["num_regions"],
                "t_maxd": None if b is None else b["max_depth"],
                "t_n18": None if b is None else b["n_depth_ge_18"],
                "c_conv": a["converged"], "t_conv": None if b is None else b["converged"],
                "c_low10_regions": low["c"][0], "c_low10_maxd": low["c"][1],
                "t_low10_regions": low["t"][0], "t_low10_maxd": low["t"][1],
            })
        group_tables[name] = tab
    out("")
    out(f"# the group calls of c and t@1e-7 are the same sequence of (regime, label, x_span) on "
        f"every row: {all(seq_ok.values())} {[n for n, v in seq_ok.items() if not v]}")

    scores = {}
    # ------------------------------------------------------------------ T2
    out("")
    out("== T2: every row (all 39), under t at 1e-7, no group has a region at depth >= 18 "
        "(an over-budget unit counts as false)")
    t2 = []
    for r in rows:
        n = r["name"]
        e = res[n]
        if e["t7_status"] != "done":
            ok, why = False, e["t7_status"]
            worst18, md = None, None
        else:
            worst18 = max((t["t_n18"] for t in group_tables[n]), default=0)
            md = max((t["t_maxd"] for t in group_tables[n]), default=None)
            ok, why = worst18 == 0, None
        t2.append((n, ok))
        out(f"   {n:<14s} {e['role']:<7s} t7 max depth {g(md)}; regions at depth >= 18 in its "
            f"worst group {g(worst18)} -> {'holds' if ok else 'FAILS' + (' (' + why + ')' if why else '')}")
    scores["T2"] = {"rows": len(t2), "hold": sum(1 for _n, ok in t2 if ok),
                    "fail": [n for n, ok in t2 if not ok],
                    "max_depth_t7": max((res[n]["t7_depth"] or 0) for n, _ in t2),
                    "verdict": "holds" if all(ok for _n, ok in t2) else "fails"}
    out(f"   T2: true on {scores['T2']['hold']} of {len(t2)} -> {scores['T2']['verdict'].upper()}; "
        f"largest max depth under t@1e-7 {scores['T2']['max_depth_t7']}")
    out(f"   (the 11 capped under B1: true on {sum(1 for n, ok in t2 if ok and n in capped)} of "
        f"{len(capped)}; max depth there {max(res[n]['t7_depth'] or 0 for n in capped)})")

    # ------------------------------------------------------------------ T3
    out("")
    out("== T3: every row (all 39), |total_t - total_c| <= total_abserr_c at 1e-7")
    t3 = []
    for r in rows:
        n = r["name"]
        e = res[n]
        if e["t7_status"] != "done" or e["c7_status"] != "done":
            t3.append((n, None, None, None))
            out(f"   {n:<14s} undecided: c7 {e['c7_status']}, t7 {e['t7_status']}")
            continue
        ok, df, ratio = le_exact(e["total_t7"], e["total_c7"], e["abserr_c7"])
        rel = df / abs(e["total_c7"]) if e["total_c7"] else (0.0 if df == 0 else float("inf"))
        bit = e["total_t7"].hex() == e["total_c7"].hex()
        e.update(T3_ok=ok, dt_c=df, dt_ratio_abserr_c=ratio, dt_rel_c=rel, total_bit_identical=bit)
        out(f"   {n:<14s} {e['role']:<7s} {e['band']:4s} {str(e['cls']):4s} capped-B1 {e['capped_B1']!s:5s} "
            f"c {e['total_c7']!r} t {e['total_t7']!r} |dt| {g(df)} abserr_c {g(e['abserr_c7'])} "
            f"ratio {g(ratio)} |dt|/|c| {g(rel)} bit-identical {bit} -> {'holds' if ok else 'FAILS'}")
        t3.append((n, ok, ratio, rel))
    fails = [x[0] for x in t3 if x[1] is False]
    und = [x[0] for x in t3 if x[1] is None]
    finite = [x[2] for x in t3 if x[2] is not None and not math.isnan(x[2])]
    scores["T3"] = {"rows": len(t3), "hold": sum(1 for x in t3 if x[1]), "fail": fails,
                    "undecided_rows": und, "max_ratio": max(finite) if finite else None,
                    "bit_identical_rows": [n for n in res if res[n].get("total_bit_identical")],
                    "verdict": "fails" if fails else ("undecided" if und else "holds")}
    out(f"   T3: true on {scores['T3']['hold']} of {len(t3)} -> {scores['T3']['verdict'].upper()}; "
        f"largest ratio {g(scores['T3']['max_ratio'])}; false on {fails}; undecided on {und}; "
        f"totals bit-identical on {len(scores['T3']['bit_identical_rows'])} rows {scores['T3']['bit_identical_rows']}")
    t3c = [x for x in t3 if x[0] in capped]
    out(f"   (on the {len(t3c)} rows capped under B1: true on {sum(1 for x in t3c if x[1])}; ratios "
        f"{[g(x[2]) for x in t3c]})")

    # ------------------------------------------------------------------ T4
    out("")
    out("== T4: every depth-20 row with both t units in budget, |total_t(1e-7) - total_t(1e-8)| "
        "<= total_abserr_t(1e-7)")
    t4 = []
    for n in d20:
        e = res[n]
        if e["t7_status"] != "done" or e.get("t8_status") != "done":
            out(f"   {n:<14s} excluded: t7 {e['t7_status']}, t8 {e.get('t8_status')}")
            continue
        ok, df, ratio = le_exact(e["total_t7"], e["total_t8"], e["abserr_t7"])
        q5 = df / abs(e["total_t8"]) if e["total_t8"] else None
        e.update(T4_ok=ok, T4_ratio=ratio, T4_rel=q5)
        # 05's L3 on the same row, for comparison (c in 05's run is this run's c)
        l3 = le_exact(F(r05[f"c-1e-7-{n}"]["result"]["total"]), F(r05[f"c-1e-8-{n}"]["result"]["total"]),
                      F(r05[f"c-1e-7-{n}"]["result"]["total_abserr"]))
        out(f"   {n:<14s} {e['band']:4s} {str(e['cls']):4s} capped-B1 {e['capped_B1']!s:5s} t7 {e['total_t7']!r} "
            f"t8 {e['total_t8']!r} |dt| {g(df)} abserr_t7 {g(e['abserr_t7'])} ratio {g(ratio)} "
            f"|dt|/|t8| {g(q5)} -> {'holds' if ok else 'FAILS'}   (05's L3 under c: ratio {g(l3[2])}, "
            f"{'holds' if l3[0] else 'fails'})")
        t4.append((n, ok, ratio, q5))
    finite = [x[2] for x in t4 if x[2] is not None and not math.isnan(x[2])]
    scores["T4"] = {"rows": len(t4), "hold": sum(1 for x in t4 if x[1]),
                    "fail": [x[0] for x in t4 if not x[1]], "excluded": [n for n in d20 if n not in [x[0] for x in t4]],
                    "max_ratio": max(finite) if finite else None,
                    "verdict": "holds" if all(x[1] for x in t4) else "fails"}
    out(f"   T4: true on {scores['T4']['hold']} of {len(t4)} -> {scores['T4']['verdict'].upper()}; "
        f"largest ratio {g(scores['T4']['max_ratio'])}; false on {scores['T4']['fail']}; excluded "
        f"{scores['T4']['excluded']}")
    t4c = [x for x in t4 if x[0] in capped]
    out(f"   (on the {len(t4c)} rows capped under B1: true on {sum(1 for x in t4c if x[1])}; ratios "
        f"{[g(x[2]) for x in t4c]})")

    # ------------------------------------------------------------------ T1, the probe
    out("")
    out("== T1 (as amended by section 9): the probe windows, c against t")
    windows = []
    mismatch = []
    for r in rows:
        n = r["name"]
        c7, t7 = unit("c", "1e-7", n), unit("t", "1e-7", n)
        wc, wt = c7.get("probe") or [], t7.get("probe") or []
        if len(wc) != len(wt):
            mismatch.append(f"{n}: {len(wc)} vs {len(wt)} windows")
            continue
        sc = load_pkl(c7).get("probe_samples") or []
        st = load_pkl(t7).get("probe_samples") or []
        for i, (a, b) in enumerate(zip(wc, wt)):
            key = (a["subinterval"], a["factor"])
            if key != (b["subinterval"], b["factor"]) or a["z_hex"] != b["z_hex"] or a["k"] != b["k"]:
                mismatch.append(f"{n} window {i}: {key} {a['z_hex']} vs {(b['subinterval'], b['factor'])} {b['z_hex']}")
                continue
            if a["theta"] != b["theta"] or a["abs_dtheta_dz"] != b["abs_dtheta_dz"]:
                mismatch.append(f"{n} window {i}: raw_theta or theta_deriv differ between trees")
            windows.append((n, a, b, sc[i], st[i]))
    out(f"   windows: {len(windows)} paired; mismatches: {mismatch if mismatch else 'none'}")

    def choose(a, b, true_step):
        for la, lb in zip(a["levels"], b["levels"]):
            if not (la["distinct"] and lb["distinct"]):
                return None
            if la["step_z"] < HALF_PI and lb["step_z"] < HALF_PI and (
                not true_step or (la["true_step_z_est"] < HALF_PI and lb["true_step_z_est"] < HALF_PI)
            ):
                return la["n"]
        return None

    def score(a, b, lev):
        la, lb = a["levels"][lev], b["levels"][lev]
        scale = EPS * a["abs_theta"]
        terms = {"1e-3 eps|theta|": 1e-3 * scale,
                 "eps(k max|p_z| + 2pi)": EPS * (a["k"] * lb["max_abs_p_z"] + TWO_PI),
                 "1e-13": 1e-13}
        bound = max(terms.values())
        largest = max(terms, key=terms.get)
        c_ok = la["sigma_z"] >= 0.1 * scale
        t_ok = lb["sigma_z"] <= bound
        return {"n": lev, "w": la["w"], "step_c": la["step_z"], "step_t": lb["step_z"],
                "true_step": lb["true_step_z_est"], "sigma_c": la["sigma_z"], "sigma_t": lb["sigma_z"],
                "sigma_u_c": la["sigma_u"], "sigma_u_t": lb["sigma_u"],
                "max_abs_p_z_t": lb["max_abs_p_z"], "max_abs_p_z_c": la["max_abs_p_z"],
                "bound_t": bound, "largest_term": largest, "terms": terms,
                "c_ratio": la["sigma_z"] / scale, "t_ratio": lb["sigma_z"] / scale,
                "u_t_ratio": None if lb["sigma_u"] is None else lb["sigma_u"] / scale,
                "c_ok": c_ok, "t_ok": t_ok, "ok": c_ok and t_ok, "nodes_t": lb["nodes"]}

    t1 = {"literal": [], "true_step": []}
    probe_out = []
    for (n, a, b, sa, sb) in windows:
        scale = EPS * a["abs_theta"]
        rec = {"row": n, "subinterval": a["subinterval"], "factor": a["factor"], "regime": a["regime"],
               "z": a["z"], "z_hex": a["z_hex"], "k": a["k"], "z_init": a["z_init"],
               "abs_theta": a["abs_theta"], "eps_abs_theta": scale, "abs_dtheta_du": a["abs_dtheta_du"],
               "abs_dtheta_dz": a["abs_dtheta_dz"], "abscissa_term": a["abscissa_term"],
               "exact_remainder": {"c": a["exact_remainder"], "t": b["exact_remainder"]},
               "scored_scale": scale >= T1_MIN_SCALE,
               "ladder": {"c": a["levels"], "t": b["levels"]}}
        for reading in ("literal", "true_step"):
            lev = choose(a, b, reading == "true_step")
            if lev is None:
                rec[reading] = None
            else:
                s = score(a, b, lev)
                s["samples"] = {
                    "z": hexlist(sb[lev]["z"]), "u": hexlist(sb[lev]["u"]),
                    "theta_z_c": hexlist(sa[lev]["theta_z"]), "theta_z_t": hexlist(sb[lev]["theta_z"]),
                    "theta_u_c": hexlist(sa[lev]["theta_u"]), "theta_u_t": hexlist(sb[lev]["theta_u"]),
                    "p_z_t": hexlist(sb[lev]["p_z"]), "node_t": [int(v) for v in sb[lev]["node"]],
                }
                # re-computed from the samples, as a check of the recorded sigma
                x = np.array([float.fromhex(v) for v in s["samples"]["z"]])
                for tree in ("c", "t"):
                    th = np.array([float.fromhex(v) for v in s["samples"][f"theta_z_{tree}"]])
                    sig, stp = fit_sigma(x, th, x[0], x[-1])
                    assert sig == s[f"sigma_{tree}"] and stp == s[f"step_{tree}"], (n, tree, sig, s[f"sigma_{tree}"])
                rec[reading] = s
            t1[reading].append(rec)
        probe_out.append(rec)

    for reading in ("literal", "true_step"):
        out("")
        out(f"   -- reading: {'LITERAL (scored): largest unwrapped z-step < pi/2 in both trees' if reading == 'literal' else 'TRUE STEP (unscored): also |dtheta/dz|(b-a)/64 < pi/2'}")
        out("   row            sub fac  z            k          |theta|     eps|th|    n  w         step c/t        true step   "
            "sig_c/eps|th| sig_t/eps|th| bound_t/eps|th| largest term          maxpz_t    u_t/eps|th| absc/eps|th| verdict")
        nf, nu, nh, nsc = [], [], 0, 0
        for rec in t1[reading]:
            s = rec[reading]
            sc = rec["scored_scale"]
            if s is None:
                tag = "UNRESOLVED" + ("" if sc else " (below 1e-11, not scored)")
                if sc:
                    nu.append(rec)
                out(f"   {rec['row']:<14s} {rec['subinterval']:>3d} {rec['factor']:3s} {rec['z']:<12.6g} {rec['k']:<10.4g} "
                    f"{rec['abs_theta']:<11.4g} {rec['eps_abs_theta']:<10.3g} {tag}")
                continue
            if sc:
                nsc += 1
                if s["ok"]:
                    nh += 1
                else:
                    nf.append(rec)
            verdict = ("holds" if s["ok"] else f"FAILS (c {s['c_ok']}, t {s['t_ok']})") if sc else "not scored (eps|theta| < 1e-11)"
            scale = rec["eps_abs_theta"]
            out(f"   {rec['row']:<14s} {rec['subinterval']:>3d} {rec['factor']:3s} {rec['z']:<12.6g} {rec['k']:<10.4g} "
                f"{rec['abs_theta']:<11.4g} {scale:<10.3g} {s['n']:<2d} {s['w']:<9.3g} {s['step_c']:.3f}/{s['step_t']:.3f}".ljust(150)
                + f"{s['true_step']:<11.3g} {s['c_ratio']:<13.4g} {s['t_ratio']:<13.4g} {s['bound_t'] / scale:<15.4g} "
                f"{s['largest_term']:<21s} {s['max_abs_p_z_t']:<10.3g} {g(s['u_t_ratio'], '.4g'):<11s} "
                f"{rec['abscissa_term'] / scale:<12.4g} {verdict}")
        verdict = "fails" if nf else ("undecided" if nu else "holds")
        scored = [rec for rec in t1[reading] if rec["scored_scale"] and rec[reading] is not None]
        pz_largest = [rec for rec in scored if rec[reading]["largest_term"] == "eps(k max|p_z| + 2pi)"]
        u_over = [rec for rec in scored if rec[reading]["u_t_ratio"] is not None and rec[reading]["u_t_ratio"] > 0.1]
        aliased = [rec for rec in scored if rec[reading]["true_step"] >= HALF_PI]
        summ = {
            "windows": len(t1[reading]),
            "with_scale_ge_1e-11": sum(1 for rec in t1[reading] if rec["scored_scale"]),
            "resolved_scored": nsc, "hold": nh,
            "fail": [(rec["row"], rec["subinterval"], rec["factor"], rec[reading]["c_ok"], rec[reading]["t_ok"]) for rec in nf],
            "fail_c_clause": sum(1 for rec in nf if not rec[reading]["c_ok"]),
            "fail_t_clause": sum(1 for rec in nf if not rec[reading]["t_ok"]),
            "unresolved_scored": [(rec["row"], rec["subinterval"], rec["factor"]) for rec in nu],
            "unresolved_below_scale": sum(1 for rec in t1[reading] if rec[reading] is None and not rec["scored_scale"]),
            "aliased_scored": len(aliased),
            "c_ratio_range": [min((r[reading]["c_ratio"] for r in scored), default=None), max((r[reading]["c_ratio"] for r in scored), default=None)],
            "t_ratio_range": [min((r[reading]["t_ratio"] for r in scored), default=None), max((r[reading]["t_ratio"] for r in scored), default=None)],
            "t_over_bound_max": max((r[reading]["sigma_t"] / r[reading]["bound_t"] for r in scored), default=None),
            "pz_term_largest_windows": len(pz_largest),
            "pz_term_largest_t_ratio_range": [min((r[reading]["t_ratio"] for r in pz_largest), default=None), max((r[reading]["t_ratio"] for r in pz_largest), default=None)],
            "largest_term_counts": dict(collections.Counter(r[reading]["largest_term"] for r in scored)),
            "u_t_over_0.1": [(r["row"], r["subinterval"], r["factor"], r[reading]["u_t_ratio"], r["abscissa_term"] / r["eps_abs_theta"]) for r in u_over],
            "levels_used": dict(collections.Counter(r[reading]["n"] for r in scored)),
            "verdict": verdict,
        }
        scores[f"T1_{reading}"] = summ
        out(f"   T1 [{reading}]: windows {summ['windows']}; with eps|theta| >= 1e-11: {summ['with_scale_ge_1e-11']}; "
            f"resolved and scored {nsc}; true in {nh}; false in {len(nf)} (c clause false {summ['fail_c_clause']}, t clause false "
            f"{summ['fail_t_clause']}); unresolved and >= 1e-11: {len(nu)}; scored windows whose true step >= pi/2 (aliased): "
            f"{summ['aliased_scored']} -> {verdict.upper()}")
        out(f"      sigma_c/eps|theta| in [{g(summ['c_ratio_range'][0], '.4g')}, {g(summ['c_ratio_range'][1], '.4g')}]; "
            f"sigma_t/eps|theta| in [{g(summ['t_ratio_range'][0], '.4g')}, {g(summ['t_ratio_range'][1], '.4g')}]; largest "
            f"sigma_t/bound_t {g(summ['t_over_bound_max'], '.4g')}; levels used {summ['levels_used']}")
        out(f"      (unscored) the term eps(k max|p_z| + 2pi) is the largest of the three in {summ['pz_term_largest_windows']} "
            f"windows; sigma_t/eps|theta| there in [{g(summ['pz_term_largest_t_ratio_range'][0], '.4g')}, "
            f"{g(summ['pz_term_largest_t_ratio_range'][1], '.4g')}]; largest term counts {summ['largest_term_counts']}")
        out(f"      (unscored) u-sampled sigma_t > 0.1 eps|theta| in {len(u_over)} windows: "
            + "; ".join(f"{r[0]} {r[1]} {r[2]}: {r[3]:.4g} against abscissa {r[4]:.4g}" for r in summ["u_t_over_0.1"]))

    # ------------------------------------------------------------------ descriptive
    out("")
    out("== descriptive: |total_t - total_c| / |total_c| and / total_abserr_c on every row: see T3's table")
    out("")
    out("== descriptive: regions and max depth per group, c -> t at 1e-7 (low10 = regions starting in the "
        "lowest 10% in log(1+z) of the group's sub-interval, and their max depth)")
    for r in rows:
        n = r["name"]
        tab = group_tables[n]
        if not tab:
            continue
        out(f"   {n} ({res[n]['cls']}, {res[n]['band']}{', capped under B1' if res[n]['capped_B1'] else ''}):")
        for gq in (False, True):
            for t in tab:
                if (t["regime"] == "Gqr") != gq:
                    continue
                out(f"      {'[Gqr] ' if gq else '      '}{t['regime']:>4s}/{t['group']:<6s} x in ({float(t['x_span'][0]):.5g}, "
                    f"{float(t['x_span'][1]):.5g}): regions {t['c_regions']:>4d} -> {g(t['t_regions']):>4s}; max depth "
                    f"{t['c_maxd']:>2d} -> {g(t['t_maxd']):>2s}; n>=18 {t['c_n18']} -> {g(t['t_n18'])}; low10 regions "
                    f"{g(t['c_low10_regions'])} -> {g(t['t_low10_regions'])}, max depth {g(t['c_low10_maxd'])} -> "
                    f"{g(t['t_low10_maxd'])}; converged {t['c_conv']} -> {t['t_conv']}")
    allg = [t for tab in group_tables.values() for t in tab]
    same_r = sum(1 for t in allg if t["c_regions"] == t["t_regions"])
    out(f"   groups: {len(allg)}; same region count under c and t: {same_r}; t more: "
        f"{sum(1 for t in allg if (t['t_regions'] or 0) > t['c_regions'])}; t fewer: "
        f"{sum(1 for t in allg if t['t_regions'] is not None and t['t_regions'] < t['c_regions'])}")
    gq = [t for t in allg if t["regime"] == "Gqr"]
    out(f"   Gqr groups: {len(gq)}; same region count {sum(1 for t in gq if t['c_regions'] == t['t_regions'])}; "
        f"regions c {sum(t['c_regions'] for t in gq)} -> t {sum(t['t_regions'] or 0 for t in gq)}; max depth c "
        f"{max(t['c_maxd'] for t in gq)} -> t {max(t['t_maxd'] or 0 for t in gq)}")

    out("")
    out("== descriptive: cost and regions by band (sums; wall in s)")
    for lab, sel in (("low band (drawn)", lambda n: res[n]["band"] == "low" and res[n]["role"] == "d20"),
                     ("mid band (drawn)", lambda n: res[n]["band"] == "mid" and n not in R03 and res[n]["role"] == "d20"),
                     ("03's R1-R7 (mid)", lambda n: n in R03),
                     ("capped under B1", lambda n: n in capped),
                     ("controls", lambda n: res[n]["role"] == "control")):
        es = [res[n] for n in res if sel(n)]
        out(f"   {lab:<18s}: rows {len(es)}, regions c7 {sum(e['c7_regions'] or 0 for e in es)} -> t7 "
            f"{sum(e['t7_regions'] or 0 for e in es)} -> t8 {sum(e.get('t8_regions') or 0 for e in es)}, wall c7 "
            f"{sum(e['c7_wall'] for e in es):.2f} -> t7 {sum(e['t7_wall'] for e in es):.2f} -> t8 "
            f"{sum(e.get('t8_wall') or 0 for e in es):.2f}, max depth c7 {max(e['c7_depth'] or 0 for e in es)} -> t7 "
            f"{max(e['t7_depth'] or 0 for e in es)} -> t8 {max(e.get('t8_depth') or 0 for e in es)}")

    out("")
    out("== descriptive: total_converged, c@1e-7 -> t@1e-7 -> t@1e-8")
    tr = collections.Counter()
    for r in rows:
        e = res[r["name"]]
        tr[(r["role"], e["c7_conv"], e["t7_conv"], e.get("t8_conv"))] += 1
    for k, v in sorted(tr.items(), key=str):
        out(f"   {k[0]:<8s} c7 {k[1]!s:5s} -> t7 {k[2]!s:5s} -> t8 {k[3]!s:5s}: {v} rows")
    out(f"   changed between c7 and t7: {[n for n in res if res[n]['c7_conv'] != res[n]['t7_conv']]}")
    out(f"   capped under B1: " + "; ".join(
        f"{n} {res[n]['c7_conv']}->{res[n]['t7_conv']}->{res[n].get('t8_conv')}" for n in capped))
    out(f"   rows with infinite total_abserr: c7 {[n for n in res if math.isinf(res[n].get('abserr_c7', 0))]}, "
        f"t7 {[n for n in res if math.isinf(res[n].get('abserr_t7', 0))]}")

    out("")
    out("== VERDICTS")
    out(f"   T1 (section 9, literal reading, scored): {scores['T1_literal']['verdict'].upper()}")
    out(f"   (T1 by the true step, unscored): {scores['T1_true_step']['verdict'].upper()}")
    for k in ("T2", "T3", "T4"):
        s = scores[k]
        out(f"   {k}: {s['verdict'].upper()} ({s['hold']} of {s['rows']} rows true; false on {s['fail']})")
    out(f"   over-budget units: {over if over else 'none'}")

    out_json.write_text(json.dumps({"rows": res, "groups": group_tables, "scores": scores,
                                    "capped_under_B1": capped, "over_budget": over}, indent=1, default=str))
    probe_json.write_text(json.dumps({
        "about": "levin-phase-floor 05a section 6, the phase probe: every window, both trees' ladders, and "
                 "the samples (as float.hex) at the level each reading uses. rescore_t1.py re-scores T1 from this file alone.",
        "eps": EPS, "windows": probe_out}, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
