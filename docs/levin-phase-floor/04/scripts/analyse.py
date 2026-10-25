"""
levin-phase-floor prompt 04, step 8: the analysis, from the run's checkpoints alone, into
out/remove_04.txt.

Usage (cwd var/levin-phase-floor-04):
    <venv python> -B scripts/analyse.py RUN_DIR > out/remove_04.txt

Nothing is re-run and no production code is imported. The inputs are:
- the run's ``checkpoint.jsonl`` (28 unit records; floats read back exactly from their hex);
- step 5's ``state/phases_04.json``, for each group's scored status (prompt 03's rule, from 03's
  pickles) and F^P_g, F^A_g, F^X_g, which prompt section 5 sets beside the depths;
- prompt 03's F per row (``out/phases_04.txt`` computes it as 03 did, recorded in the json).

Definitions (prompt section 5, as written; gates as amended by section 8, commit d170e6d):
- a group's "regions at depth 18 or more" is its Levin call's ``n_depth_ge_18``;
- "the row's region count" is ``WKB_Levin_data.num_regions`` of the row's result;
- every difference of two totals is formed exactly (``fractions``) from the two doubles, then
  rounded once;
- an over-budget unit counts as prompt section 5 says for each score.
Groups are matched across units by their position in the call order, and the (regime, group)
labels are checked equal.
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import MAIN, header, sha256  # noqa: E402

ROWS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
UNITS = [("P", "1e-7"), ("X", "1e-7"), ("A", "1e-7"), ("X", "1e-8")]
SWEEP = {"mid band": 1.9e-6, "low": 2.84e-6, "seam": 8.3e-7}


def out(s=""):
    print(s, flush=True)


def f(d):
    """A float back from the checkpoint's {repr, hex}."""
    if d is None:
        return None
    v = float.fromhex(d["hex"])
    assert repr(v) == d["repr"], d
    return v


def absdiff(a, b):
    return float(abs(Fraction(a) - Fraction(b)))


def g3(x, p=3):
    if x is None:
        return "n/a"
    return f"{x:.{p}g}"


def main():
    run_dir = Path(sys.argv[1]).resolve()
    ckpt = run_dir / "checkpoint.jsonl"
    phases_json = HERE.parent / "state" / "phases_04.json"
    for line in header(
        Path(__file__).resolve(),
        "levin-phase-floor prompt 04 step 8: remove_04",
        inputs=[ckpt, run_dir / "manifest.json", run_dir / "status.json", phases_json],
    ):
        out(line)
    sidecar = json.loads(
        (MAIN / "var/datastores/handover-A3-v2-lambdacdm.manifest.json").read_text()
    )
    out(f"# store fingerprint (sidecar, read only) {sidecar['fingerprint']['digest']}")
    out(f"# run directory {run_dir}")
    out(
        "# definitions: module docstring of analyse.py; prompt section 5 (scores), section 8 (gates)"
    )
    out()

    recs = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
    U = {}
    for r in recs:
        d = r["data"]
        U[(d["arm"], d["rtol"], d["row"])] = (r, d)
    status = json.loads((run_dir / "status.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    ph = json.loads(phases_json.read_text())

    out("== the run")
    out(
        f"   run {manifest['run_id']}; state {status['state']}, exit {status['exit_code']}, units {status['units_done']}/{status['units_total']}"
    )
    out(
        f"   manifest git_head {manifest['git_head']} (main checkout; section 8), script_sha256 {manifest['script_sha256']}"
    )
    wts = {(d["worktree_head"], d["worktree_status_short"]) for _r, d in U.values()}
    shas = {r["script_sha256"] for r, _d in U.values()}
    out(
        f"   every record: worktree (HEAD, status --short) in {sorted(wts)}; registry script_sha256 in {sorted(shas)}"
    )
    out(
        f"   {len(recs)} records; units present: {sum(1 for a, rt in UNITS for n in ROWS if (a, rt, n) in U)}/28; "
        f"order: {[r['data']['unit'] for r in recs] == [f'{a}-{rt}-{n}' for a, rt in UNITS for n in ROWS]}"
    )
    over = [d["unit"] for _r, d in U.values() if d["status"] != "done"]
    out(f"   units over budget: {over if over else 'none'}")
    out()

    # ---------------------------------------------------------------------------------------
    out("== M0: P at rtol 1e-7 equals the stored row (the run's own comparison, per unit)")
    m0_all = True
    for n in ROWS:
        _r, d = U[("P", "1e-7", n)]
        differ_gated = [
            l for l in d["m0_lines"] if " DIFFER " in l and "(not gated)" not in l
        ]
        differ_ungated = [
            l.split()[0] for l in d["m0_lines"] if "DIFFER (not gated)" in l
        ]
        ar = [l for l in d["m0_lines"] if l.strip().startswith("analytic_rad")]
        m0_all &= d["m0_holds"] and not differ_gated
        out(
            f"   {n}: M0 {'holds' if d['m0_holds'] else 'FAILS'}; gated fields differing {len(differ_gated)}; "
            f"ungated differing: {sorted(set(x.split('.')[-1] for x in differ_ungated))} ({len(differ_ungated)}); "
            f"analytic_rad {'EQUAL' if ar and 'EQUAL' in ar[0] else 'DIFFER'}"
        )
    out(f"   M0 {'HOLDS' if m0_all else 'FAILS'} on all seven rows")
    out()

    # ---------------------------------------------------------------------------------------
    out(
        "== per row, arm and rung: total, total_abserr, total_converged, total_phase_limited, regions, max depth, evaluations, elapsed"
    )
    for n in ROWS:
        out(f"-- {n} (F = {ph['rows'][n]['F']:.4g})")
        for a, rt in UNITS:
            _r, d = U[(a, rt, n)]
            if d["status"] != "done":
                out(f"   {a} {rt}: OVER BUDGET after {d['wall_s']:.1f} s (budget {d['budget_s']:.1f} s)")
                continue
            res = d["result"]
            wl = res["WKB_Levin_data"]
            out(
                f"   {a} {rt}: total {f(res['total'])!r:>24s}  abserr {f(res['total_abserr']):.6e}  converged {str(res['total_converged']):5s} "
                f"phase_limited {str(res['total_phase_limited']):5s} regions {wl['num_regions']:6d} max depth {wl['max_depth']:2d} "
                f"evaluations {wl['evaluations']:6d} wall {d['wall_s']:7.1f} s (compute_time {res['compute_time']:.1f} s; budget {g3(d['budget_s'], 4)})"
            )
    out()

    # ---------------------------------------------------------------------------------------
    out(
        "== per group: regions, depth histogram by type (depth:Levin/CC), count at depth >= 18, value and abserr, beside F^arm_g"
    )
    G = {}  # (row, idx) -> {arm-rt: call}
    for n in ROWS:
        meta = ph["rows"][n]["groups"]
        calls = {(a, rt): U[(a, rt, n)][1]["calls"] for a, rt in UNITS}
        for i, gm in enumerate(meta):
            for (a, rt), cl in calls.items():
                if U[(a, rt, n)][1]["status"] != "done":
                    continue
                c = cl[i]
                assert (c["regime"], c["group"]) == (gm["regime"], gm["label"]), (n, i, a, rt)
            G[(n, i)] = dict(meta=gm, calls=calls)
    for n in ROWS:
        out(f"-- {n}")
        for i, gm in enumerate(ph["rows"][n]["groups"]):
            out(
                f"   {gm['regime']:>4s} {gm['label']:<6s} scored {str(gm['scored']):5s} F^P {gm['F_P']:.3e} F^A {gm['F_A']:.3e} F^X {gm['F_X']:.3e}"
            )
            for a, rt in UNITS:
                d = U[(a, rt, n)][1]
                if d["status"] != "done":
                    out(f"      {a} {rt}: over budget")
                    continue
                c = d["calls"][i]
                hist = " ".join(
                    f"{k}:{v[0]}/{v[1]}"
                    for k, v in sorted(c["depth_hist_levin_cc"].items(), key=lambda kv: int(kv[0]))
                )
                out(
                    f"      {a} {rt}: regions {c['num_regions']:6d} depth>=18 {c['n_depth_ge_18']:6d} max {c['max_depth']:2d} value {f(c['value']):+.6e} abserr {f(c['abserr']):.3e} "
                    f"conv {str(c['converged']):5s} plim {str(c['phase_limited']):5s} | {hist}"
                )
    out()

    # ---------------------------------------------------------------------------------------
    out("== R1: under X at 1e-7, every scored group has < 10 regions at depth >= 18, and the row's region count <= P's / 10; holds on >= 6 of 7 rows")
    r1_rows = 0
    for n in ROWS:
        dX = U[("X", "1e-7", n)][1]
        dP = U[("P", "1e-7", n)][1]
        if dX["status"] != "done":
            out(f"   {n}: X over budget -> false")
            continue
        sc = [(i, gm) for i, gm in enumerate(ph["rows"][n]["groups"]) if gm["scored"]]
        worst = max(dX["calls"][i]["n_depth_ge_18"] for i, _ in sc)
        part1 = all(dX["calls"][i]["n_depth_ge_18"] < 10 for i, _ in sc)
        nX = dX["result"]["WKB_Levin_data"]["num_regions"]
        nP = dP["result"]["WKB_Levin_data"]["num_regions"]
        part2 = nX <= nP / 10
        ok = part1 and part2
        r1_rows += ok
        out(
            f"   {n}: scored groups {len(sc)}, largest count at depth>=18 under X {worst} ({'<10 in all' if part1 else 'NOT'}); "
            f"regions X {nX} / P {nP} = {nX / nP:.4g} ({'<=0.1' if part2 else '>0.1'}) -> {'true' if ok else 'false'}"
        )
    r1 = r1_rows >= 6
    out(f"   R1 true on {r1_rows} of 7 rows: {'HOLDS' if r1 else 'FAILS'}")
    out()

    out("== A1: under A at 1e-7, every scored group with F^A_g <= 1e-8 has < 10 regions at depth >= 18; holds if in every such group")
    covered = {(n, rg, lb) for n, rg, lb, _fa in ph["a1_groups"]}
    a1_ok = True
    a1_n = 0
    others = []
    for n in ROWS:
        dA = U[("A", "1e-7", n)][1]
        dX = U[("X", "1e-7", n)][1]
        for i, gm in enumerate(ph["rows"][n]["groups"]):
            if not gm["scored"]:
                continue
            key = (n, gm["regime"], gm["label"])
            cov = gm["F_A"] <= 0.1 * 1e-7
            assert cov == (key in covered)
            if dA["status"] != "done":
                cA = None
            else:
                cA = dA["calls"][i]
            cX = dX["calls"][i] if dX["status"] == "done" else None
            if cov:
                a1_n += 1
                ok = cA is not None and cA["n_depth_ge_18"] < 10
                a1_ok &= ok
                out(
                    f"   {n} {gm['regime']:>4s} {gm['label']:<6s} F^A {gm['F_A']:.3e}: depth>=18 under A {cA['n_depth_ge_18'] if cA else 'over budget'} "
                    f"(regions {cA['num_regions'] if cA else '-'}, max depth {cA['max_depth'] if cA else '-'}) -> {'true' if ok else 'false'}"
                )
            else:
                others.append((n, gm, cA, cX))
    a1 = "undecided" if a1_n == 0 else ("HOLDS" if a1_ok else "FAILS")
    out(f"   A1 over {a1_n} covered groups: {a1}")
    out("   the other scored groups (unscored here): depth under A beside X's, and F^A_g")
    for n, gm, cA, cX in others:
        out(
            f"   {n} {gm['regime']:>4s} {gm['label']:<6s} F^A {gm['F_A']:.3e} (F^A/F {gm['F_A'] / ph['rows'][n]['F']:.3g}): "
            f"A depth>=18 {cA['n_depth_ge_18'] if cA else 'over'} regions {cA['num_regions'] if cA else '-'} max {cA['max_depth'] if cA else '-'}; "
            f"X depth>=18 {cX['n_depth_ge_18'] if cX else 'over'} regions {cX['num_regions'] if cX else '-'} max {cX['max_depth'] if cX else '-'}"
        )
    out()

    out("== N4: |total_P - total_ref| <= total_abserr_P, total_ref from X at 1e-8 (X at 1e-7 if that was over budget); holds if on all seven")
    n4_ok = True
    n4_undec = False
    n4_fail = []
    for n in ROWS:
        dP = U[("P", "1e-7", n)][1]["result"]
        d8 = U[("X", "1e-8", n)][1]
        d7 = U[("X", "1e-7", n)][1]
        if d8["status"] == "done":
            ref, which = f(d8["result"]["total"]), "X 1e-8"
        elif d7["status"] == "done":
            ref, which = f(d7["result"]["total"]), "X 1e-7 (X 1e-8 over budget)"
        else:
            out(f"   {n}: both X units over budget -> undecided")
            n4_undec = True
            continue
        tP = f(dP["total"])
        eP = f(dP["total_abserr"])
        diff = absdiff(tP, ref)
        ok = diff <= eP
        n4_ok &= ok
        if not ok:
            n4_fail.append((n, diff / eP))
        out(
            f"   {n}: |P - ref| = {diff:.4e}  total_abserr_P {eP:.4e}  ratio {diff / eP:.4g}  |P - ref|/|ref| {diff / abs(ref):.3e}  (ref {which}) -> {'true' if ok else 'FALSE'}"
        )
    n4 = "undecided" if (n4_undec and n4_ok) else ("HOLDS" if n4_ok else "FAILS")
    out(f"   N4: {n4}" + (f"; failing rows (ratio |P - ref| / total_abserr_P): {[(n, round(r, 4)) for n, r in n4_fail]}" if n4_fail else ""))
    out()

    out("== R2: |total_X(1e-7) - total_X(1e-8)| <= total_abserr_X(1e-7); holds if on all seven; Q5's number on every row")
    r2_ok = True
    r2_undec = False
    for n in ROWS:
        d7 = U[("X", "1e-7", n)][1]
        d8 = U[("X", "1e-8", n)][1]
        if d7["status"] != "done" or d8["status"] != "done":
            out(f"   {n}: an X unit over budget -> undecided")
            r2_undec = True
            continue
        t7, e7 = f(d7["result"]["total"]), f(d7["result"]["total_abserr"])
        t8 = f(d8["result"]["total"])
        e8 = f(d8["result"]["total_abserr"])
        diff = absdiff(t7, t8)
        ok = diff <= e7
        r2_ok &= ok
        q5 = diff / abs(t8)
        out(
            f"   {n}: |X7 - X8| = {diff:.4e}  total_abserr_X7 {e7:.4e}  ratio {diff / e7:.4g} -> {'true' if ok else 'FALSE'};  "
            f"Q5 |X7 - X8|/|X8| = {q5:.3e}  (abserr_X7/|X8| {e7 / abs(t8):.3e}; abserr_X8/|X8| {e8 / abs(t8):.3e})"
        )
    r2 = "undecided" if (r2_undec and r2_ok) else ("HOLDS" if r2_ok else "FAILS")
    out(f"   R2: {r2}")
    out(f"   the 2026-09 sweep, for comparison: {SWEEP}")
    out()

    out("== descriptive: how far each arm's total moved from P's, relative to P's declared error")
    for n in ROWS:
        dP = U[("P", "1e-7", n)][1]["result"]
        tP, eP = f(dP["total"]), f(dP["total_abserr"])
        parts = []
        for a, rt in UNITS[1:]:
            d = U[(a, rt, n)][1]
            if d["status"] != "done":
                parts.append(f"{a}{rt}: over")
                continue
            t = f(d["result"]["total"])
            parts.append(f"{a} {rt}: |d|/abserr_P {absdiff(t, tP) / eP:.3g}, |d|/|P| {absdiff(t, tP) / abs(tP):.3e}")
        xa = absdiff(f(U[("X", "1e-7", n)][1]["result"]["total"]), f(U[("A", "1e-7", n)][1]["result"]["total"]))
        parts.append(f"|X7 - A7|/|X7| {xa / abs(f(U[('X', '1e-7', n)][1]['result']['total'])):.3e}")
        out(f"   {n}: " + "; ".join(parts) + f"; abserr_P/|P| {eP / abs(tP):.3e}")
    out()

    out(
        f"== summary: M0 {'holds' if m0_all else 'FAILS'}; R1 {'holds' if r1 else 'fails'} ({r1_rows}/7); A1 {a1.lower()} ({a1_n} groups); "
        f"N4 {n4.lower()}; R2 {r2.lower()}; over budget: {over if over else 'none'}"
    )


if __name__ == "__main__":
    main()
