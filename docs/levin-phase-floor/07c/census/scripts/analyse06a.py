"""levin-phase-floor prompt 06a: the analysis of the census run. New in 06a. Read-only; imports
nothing from the tree.

Reads the run's checkpoint.jsonl, its per-worker pickle streams (each record checked against the
sha256 its checkpoint line names) and its events_*.jsonl, and state/rows.json (A3 v2's stored
values per row). Scores C1-C3 against section 4's rules as written (C4 is c0.py's ``c4``) and
tabulates section 5's items 1-9.

Definitions (prompt section 4): a Levin call's depth is the largest region depth over its
regions; a row's depth is the largest over its calls. A row with no Levin call has no depth
("none"). Tabulations are by band (01's bands), by regime and by the class of the row's top group
(01's census class, stored; "none" where the row has no stored group record). For a row, its
regime is the regime of its deepest Levin call at h (the first such call where tied; "none" with
no call); for a call or a region, the call's own regime.

Writes: out/census_summary.txt (every table), out/census_rows.csv.gz (one line per (rtol, row)),
out/scan_listed_regions.csv (every accepted region the scan lists), state/analysis.json.
Usage: python -I -B analyse06a.py RUN_DIR
"""

import collections
import csv
import gzip
import hashlib
import io
import json
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, header, sha256  # noqa: E402

ROWS_JSON = WIDER / "state" / "rows.json"
OUT = WIDER / "out"
BANDS = ("low", "mid", "seam")
CLASSES = ("G+r", "G-r", "G", "Gqr", "none")
RTOLS = ("1e-7", "1e-8")


def fv(d):
    return None if d is None else float.fromhex(d["hex"])


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main(run_dir):
    run_dir = Path(run_dir)
    lines = []
    out = lines.append
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06a: the census analysis",
                    [ROWS_JSON, run_dir / "checkpoint.jsonl"]):
        out(l)
    rows = {r["name"]: r for r in json.loads(ROWS_JSON.read_text())["rows"]}

    # ------------------------------------------------------------------ read
    units = {}
    with open(run_dir / "checkpoint.jsonl") as f:
        for line in f:
            r = json.loads(line)
            units[r["unit"]] = r["data"]
    blobs = {}
    recs = {}
    for d in units.values():
        p = d.get("pickle")
        if p is None:
            continue
        if p not in blobs:
            blobs[p] = (run_dir / p).read_bytes()
        b = blobs[p][d["pickle_offset"]: d["pickle_offset"] + d["pickle_length"]]
        if hashlib.sha256(b).hexdigest() != d["pickle_record_sha256"]:
            raise SystemExit(f"{d['unit']}: pickle record sha256 differs")
        recs[d["unit"]] = pickle.loads(b)
    blobs = None
    out(f"# units {len(units)}; pickle records {len(recs)}, each checked against its sha256")
    events = []
    for p in sorted(run_dir.glob("events_*.jsonl")):
        with open(p) as f:
            for line in f:
                events.append(json.loads(line))

    # ------------------------------------------------------------------ per (rtol, row)
    R = {}
    for uid, d in units.items():
        row = rows[d["row"]]
        calls = d["calls"]
        res = d.get("result")
        deepest = None
        for c in calls:
            if deepest is None or c["max_depth"] > deepest["max_depth"]:
                deepest = c
        lev = sum(v[0] for c in calls for v in c["depth_hist_levin_cc"].values())
        cc = sum(v[1] for c in calls for v in c["depth_hist_levin_cc"].values())
        scan = d.get("scan") or []
        R[(d["rtol"], d["row"])] = dict(
            uid=uid, d=d, row=row, rtol=d["rtol"], band=row["band"],
            cls=row["cls"] or "none",
            regime="none" if deepest is None else deepest["regime"],
            depth=None if deepest is None else deepest["max_depth"],
            n_calls=len(calls), regions=sum(c["num_regions"] for c in calls),
            simple=sum(c["num_simple_regions"] for c in calls), lev=lev, cc=cc,
            n_ge18=sum(c["n_depth_ge_18"] for c in calls),
            calls_unconv=sum(1 for c in calls if not c["converged"]),
            calls_plim=sum(1 for c in calls if c["phase_limited"]),
            total=fv(res["total"]) if res else None,
            abserr=fv(res["total_abserr"]) if res else None,
            conv=res["total_converged"] if res else None,
            plim=res["total_phase_limited"] if res else None,
            status=d["status"],
            wall=d["wall_s"], rebuild=d["rebuild_s"], scan_s=d.get("scan_s"),
            sp_calls=sum(1 for s in scan if s and s["n_sign_changes"] > 0),
            sp_changes=sum(s["n_sign_changes"] for s in scan if s),
            sp_listed=sum(s["n_regions_listed"] for s in scan if s),
            sp_listed_grid=sum(s["n_regions_listed_grid"] for s in scan if s),
            sp_listed_ends=sum(s["n_regions_listed_ends"] for s in scan if s),
        )

    def tab(title, keyf, sel, valf=lambda x: 1, fmt="{:d}"):
        pass

    def by3(title, sel, cell):
        """A table of cell(subset) by band, by regime and by class."""
        out(f"   {title}")
        for name, keyf, keys in (
            ("band", lambda x: x["band"], BANDS),
            ("regime", lambda x: x["regime"], sorted({x["regime"] for x in sel})),
            ("class", lambda x: x["cls"], CLASSES),
        ):
            parts = []
            for k in keys:
                s = [x for x in sel if keyf(x) == k]
                parts.append(f"{k}: {cell(s)}")
            out(f"     by {name:6s}: " + "; ".join(parts))

    A = {"rtol": {}}
    out("")
    out("== status and budget")
    for rtol in RTOLS:
        sel = [x for (rt, _), x in R.items() if rt == rtol]
        out(f"   {rtol}: units {len(sel)}; status {dict(collections.Counter(x['status'] for x in sel))}; "
            f"rows with no Levin call {sum(1 for x in sel if x['depth'] is None)}")

    # ------------------------------------------------------------------ C1-C3
    out("")
    out("== Predictions (section 4), scored as written")
    verdict = {}
    for rtol, name in (("1e-7", "C1"), ("1e-8", "C2")):
        sel = [x for (rt, _), x in R.items() if rt == rtol]
        over = [x for x in sel if x["status"] == "over budget"]
        err = [x for x in sel if x["status"] == "error"]
        bad = [x for x in sel if x["n_ge18"] > 0]
        if bad or over:
            v = "fails"
        elif err:
            v = "undecided"
        else:
            v = "holds"
        verdict[name] = v
        md = max((x["depth"] for x in sel if x["depth"] is not None), default=None)
        out(f"   {name} (rtol {rtol}): rows with a region at depth >= 18: {len(bad)} of {len(sel)}; "
            f"over budget after the pass {len(over)}; errored {len(err)}; largest row depth {md} -> {v}")
    c3 = {}
    for rtol in RTOLS:
        sel = [x for (rt, _), x in R.items() if rt == rtol]
        c3[rtol] = [x for x in sel if x["depth"] is not None and x["depth"] > 11]
    verdict["C3"] = "holds" if not any(c3.values()) else "fails"
    out(f"   C3: rows whose depth exceeds 11: at 1e-7 {len(c3['1e-7'])}, at 1e-8 {len(c3['1e-8'])} "
        f"-> {verdict['C3']}")
    for rtol in RTOLS:
        if c3[rtol]:
            hist = collections.Counter(x["depth"] for x in c3[rtol])
            out(f"     {rtol}: their depths {dict(sorted(hist.items()))}")
    A["verdicts"] = verdict

    # ------------------------------------------------------------------ item 1
    out("")
    out("== 1. The distribution of row depth")
    for rtol in RTOLS:
        sel = [x for (rt, _), x in R.items() if rt == rtol]
        hist = collections.Counter("none" if x["depth"] is None else x["depth"] for x in sel)
        keys = sorted([k for k in hist if k != "none"]) + (["none"] if "none" in hist else [])
        out(f"   rtol {rtol}: " + ", ".join(f"{k}: {hist[k]}" for k in keys))
        A["rtol"].setdefault(rtol, {})["row_depth_hist"] = {str(k): hist[k] for k in keys}
        by3("largest row depth", sel,
            lambda s: max((x["depth"] for x in s if x["depth"] is not None), default="-"))
        by3("median row depth (rows with a Levin call)", sel,
            lambda s: (lambda v: f"{np.median(v):g}" if v else "-")([x["depth"] for x in s if x["depth"] is not None]))
        by3("rows", sel, lambda s: len(s))

        def sbin(x):
            d = x["row"]["stored_depth"]
            return "NULL" if d is None else ("<=10" if d <= 10 else ("11-19" if d < 20 else "20"))

        out(f"   cross-tabulation, A3 v2's stored depth (rows) x h's row depth at {rtol}:")
        hk = keys
        out("     stored \\ h   " + " ".join(f"{str(k):>5s}" for k in hk))
        for b in ("<=10", "11-19", "20", "NULL"):
            s = [x for x in sel if sbin(x) == b]
            c = collections.Counter("none" if x["depth"] is None else x["depth"] for x in s)
            out(f"     {b:12s} " + " ".join(f"{c.get(k, 0):5d}" for k in hk) + f"   (rows {len(s)})")

    # ------------------------------------------------------------------ item 2
    out("")
    out("== 2. Where depth falls by branch: regions at each depth >= 12, Levin / Clenshaw-Curtis")
    for rtol in RTOLS:
        sel = [x for (rt, _), x in R.items() if rt == rtol]
        tot = collections.defaultdict(lambda: [0, 0])
        cut = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        for x in sel:
            for c in x["d"]["calls"]:
                for dep, (nl, nc) in c["depth_hist_levin_cc"].items():
                    dep = int(dep)
                    if dep >= 12:
                        tot[dep][0] += nl
                        tot[dep][1] += nc
                        for key in (("band", x["band"]), ("regime", c["regime"]), ("class", x["cls"])):
                            cut[key][dep][0] += nl
                            cut[key][dep][1] += nc
        out(f"   rtol {rtol}: " + (", ".join(f"depth {k}: {v[0]} / {v[1]}" for k, v in sorted(tot.items())) or "no region at depth >= 12"))
        for key in sorted(cut):
            out(f"     {key[0]} {key[1]}: " + ", ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(cut[key].items())))
        allh = collections.defaultdict(lambda: [0, 0])
        for x in sel:
            for c in x["d"]["calls"]:
                for dep, (nl, nc) in c["depth_hist_levin_cc"].items():
                    allh[int(dep)][0] += nl
                    allh[int(dep)][1] += nc
        out(f"   rtol {rtol}, every region by depth (Levin/CC): "
            + ", ".join(f"{k}: {v[0]}/{v[1]}" for k, v in sorted(allh.items())))
        A["rtol"][rtol]["regions_by_depth_levin_cc"] = {str(k): v for k, v in sorted(allh.items())}

    # ------------------------------------------------------------------ item 3
    out("")
    out("== 3. The deepest rows (20 at each rtol; by row depth, then regions)")
    for rtol in RTOLS:
        sel = [x for (rt, _), x in R.items() if rt == rtol and x["depth"] is not None]
        sel.sort(key=lambda x: (-x["depth"], -x["regions"], x["row"]["name"]))
        out(f"   rtol {rtol}:")
        for x in sel[:20]:
            r = x["row"]
            out(f"   * {r['name']} {('(' + ','.join(r['names05']) + ')') if r['names05'] else ''} "
                f"k {float(r['k']):.4g} q {float(r['q']):.4g} r {float(r['r']):.4g} z_resp {float(r['z_response']):.5g} "
                f"[{x['band']}, class {x['cls']}, stored depth {r['stored_depth']}]: depth {x['depth']}, "
                f"regions {x['regions']} (Levin {x['lev']}, CC {x['cc']}), total_converged {x['conv']}, "
                f"|total|/abserr {abs(x['total']) / x['abserr'] if x['abserr'] else float('inf'):.3g}")
            for c in x["d"]["calls"]:
                v, e = fv(c["value"]), fv(c["abserr"])
                lev = sum(h[0] for h in c["depth_hist_levin_cc"].values())
                ccn = sum(h[1] for h in c["depth_hist_levin_cc"].values())
                out(f"       {c['regime']:4s} {c['group']:7s} depth {c['max_depth']:2d} regions {c['num_regions']:4d} "
                    f"(L {lev}, CC {ccn}) conv {str(c['converged']):5s} plim {str(c['phase_limited']):5s} "
                    f"|value|/abserr {abs(v) / e if e else float('inf'):.3g}")

    # ------------------------------------------------------------------ item 4
    out("")
    out("== 4. Regions, against A3 v2's stored WKB_Levin_num_regions")
    for rtol in RTOLS:
        sel = [x for (rt, _), x in R.items() if rt == rtol]
        h_tot = sum(x["regions"] for x in sel)
        s_tot = sum(x["row"]["stored_num_regions"] or 0 for x in sel)
        with_l = [x for x in sel if x["n_calls"]]
        out(f"   rtol {rtol}: store total h {h_tot:,} (simple {sum(x['simple'] for x in sel):,}; Levin {sum(x['lev'] for x in sel):,}, CC {sum(x['cc'] for x in sel):,}) "
            f"against stored {s_tot:,} (ratio {h_tot / s_tot:.4f})")
        v = [x["regions"] for x in with_l]
        out(f"     per row with a Levin call ({len(v)}): median {np.median(v):g}, 90th {pct(v, 90):g}, max {max(v)}")
        ratio = [x["regions"] / x["row"]["stored_num_regions"] for x in with_l if x["row"]["stored_num_regions"]]
        out(f"     h / stored per row: median {np.median(ratio):.3g}, 10th {pct(ratio, 10):.3g}, 90th {pct(ratio, 90):.3g}, "
            f"min {min(ratio):.3g}, max {max(ratio):.3g}; rows with more regions than stored "
            f"{sum(1 for x in with_l if x['row']['stored_num_regions'] and x['regions'] > x['row']['stored_num_regions'])}")
        nomatch = sum(1 for x in sel if (x["n_calls"] == 0) != (x["row"]["stored_num_regions"] is None))
        out(f"     rows where 'has a Levin call' differs from 'stored regions not NULL': {nomatch}")
        by3("regions (sum)", sel, lambda s: sum(x["regions"] for x in s))
        by3("stored regions (sum)", sel, lambda s: sum(x["row"]["stored_num_regions"] or 0 for x in s))
        A["rtol"][rtol]["regions_total"] = h_tot
        A["rtol"][rtol]["stored_regions_total"] = s_tot

    # ------------------------------------------------------------------ item 5
    out("")
    out("== 5. Convergence")
    for rtol in RTOLS:
        sel = [x for (rt, _), x in R.items() if rt == rtol]
        nconv = [x for x in sel if x["conv"] is False]
        inf = [x for x in sel if x["abserr"] is not None and math.isinf(x["abserr"])]
        out(f"   rtol {rtol}: total_converged False on {len(nconv)} of {len(sel)} rows "
            f"(A3 v2 stored False on {sum(1 for x in sel if x['row']['stored_total_converged'] is False)}); "
            f"unconverged Levin calls {sum(x['calls_unconv'] for x in sel)} of {sum(x['n_calls'] for x in sel)}; "
            f"phase-limited calls {sum(x['calls_plim'] for x in sel)}; total_phase_limited rows {sum(1 for x in sel if x['plim'])}")
        out(f"     rows with total_abserr = +inf: {len(inf)} (A3 v2 stored +inf on "
            f"{sum(1 for x in sel if x['row']['stored_total_abserr'] and math.isinf(fv(x['row']['stored_total_abserr'])))})")
        for x in inf[:40]:
            r = x["row"]
            out(f"       {r['name']} k {float(r['k']):.4g} q {float(r['q']):.4g} r {float(r['r']):.4g} z {float(r['z_response']):.5g} "
                f"[{x['band']}, {x['cls']}] stored abserr {r['stored_total_abserr']['repr']}")
        by3("total_converged False", sel, lambda s: sum(1 for x in s if x["conv"] is False))
        A["rtol"][rtol]["unconverged_rows"] = len(nconv)
        A["rtol"][rtol]["inf_abserr_rows"] = [x["row"]["name"] for x in inf]

    # ------------------------------------------------------------------ item 6
    out("")
    out("== 6. Totals at 1e-7 against A3 v2's stored totals")
    sel = [x for (rt, _), x in R.items() if rt == "1e-7"]
    rel, ag, nz = [], [], 0
    beyond = collections.Counter()
    ident = 0
    for x in sel:
        st = fv(x["row"]["stored_total"])
        se = fv(x["row"]["stored_total_abserr"])
        if x["total"] is None or st is None:
            continue
        dlt = abs(x["total"] - st)
        if x["total"].hex() == st.hex():
            ident += 1
        if st != 0:
            rel.append(dlt / abs(st))
        if se and math.isfinite(se) and se > 0:
            q = dlt / se
            ag.append(q)
            for m in (1, 10, 100):
                if q > m:
                    beyond[m] += 1
        else:
            nz += 1
    out(f"   rows compared {len(sel)}; bit-identical {ident}; stored total_abserr zero or non-finite on {nz} (not ratioed)")
    out(f"   relative |dtotal|/|stored|: median {np.median(rel):.3g}, 90th {pct(rel, 90):.3g}, 99th {pct(rel, 99):.3g}, max {max(rel):.3g}")
    out(f"   |dtotal| / stored total_abserr ({len(ag)} rows): median {np.median(ag):.3g}, 90th {pct(ag, 90):.3g}, "
        f"99th {pct(ag, 99):.3g}, max {max(ag):.3g}; beyond 1x {beyond[1]}, 10x {beyond[10]}, 100x {beyond[100]}")

    def cell6(s):
        q = []
        for x in s:
            st, se = fv(x["row"]["stored_total"]), fv(x["row"]["stored_total_abserr"])
            if x["total"] is not None and st is not None and se and math.isfinite(se) and se > 0:
                q.append(abs(x["total"] - st) / se)
        return f">1x {sum(1 for v in q if v > 1)}/{len(q)}"

    by3("rows beyond 1x the stored error", sel, cell6)
    A["totals_vs_A3v2"] = dict(beyond_1=beyond[1], beyond_10=beyond[10], beyond_100=beyond[100], n=len(ag),
                               median=float(np.median(ag)), max=float(max(ag)))
    worst = sorted(((abs(x["total"] - fv(x["row"]["stored_total"])) / fv(x["row"]["stored_total_abserr"]), x)
                    for x in sel if x["total"] is not None and fv(x["row"]["stored_total_abserr"])
                    and math.isfinite(fv(x["row"]["stored_total_abserr"])) and fv(x["row"]["stored_total_abserr"]) > 0),
                   key=lambda t: -t[0])[:10]
    out("   the ten largest against the stored error:")
    for q, x in worst:
        r = x["row"]
        out(f"     {r['name']} {r['names05'] or ''} [{x['band']}, {x['cls']}, stored depth {r['stored_depth']}]: "
            f"{q:.3g}x, relative {abs(x['total'] - fv(r['stored_total'])) / abs(fv(r['stored_total'])):.3g}")
    # 1e-7 against 1e-8 at h (descriptive, for 08)
    s5 = []
    for (rt, name), x in R.items():
        if rt != "1e-7":
            continue
        y = R.get(("1e-8", name))
        if x["total"] is None or y is None or y["total"] is None or not x["abserr"] or not math.isfinite(x["abserr"]):
            continue
        s5.append(abs(x["total"] - y["total"]) / x["abserr"])
    out(f"   descriptive: |total(1e-7) - total(1e-8)| / total_abserr(1e-7) at h, {len(s5)} rows: median {np.median(s5):.3g}, "
        f"90th {pct(s5, 90):.3g}, max {max(s5):.3g}; beyond 1x {sum(1 for v in s5 if v > 1)}")
    gk = [e for e in events if e["event"]["kind"] == "Gk"]
    for rtol in RTOLS:
        g = [e for e in gk if e["worker"].startswith(rtol)]
        diff = collections.Counter()
        for e in g:
            a = e["event"]["summary"]["GkSourcePolicyData"]
            for key in ("Levin_z", "type", "quality"):
                if a["rerun"][key] != a["stored"][key]:
                    diff[key] += 1
            if any(a["rerun"][k] != a["stored"][k] for k in ("Levin_z", "type", "quality")):
                diff["any"] += 1
        out(f"   GkSourcePolicyData assembled in the {rtol} workers: {len(g)}; differing from A3 v2's in "
            f"Levin_z {diff['Levin_z']}, type {diff['type']}, quality {diff['quality']} (any {diff['any']})")
        A["rtol"][rtol]["policy_diff"] = dict(diff, n=len(g))

    # ------------------------------------------------------------------ item 7
    out("")
    out("== 7. The stationary-point scan (1e-7; 4,097 points per call)")
    sel = [x for (rt, _), x in R.items() if rt == "1e-7"]
    calls = 0
    calls_sp = 0
    rows_sp = 0
    rows_listed = 0
    calls_listed = 0
    hs = []
    ratios_sp = []
    listed_regions = []
    for x in sel:
        rec = recs.get(x["uid"])
        if rec is None or rec["scan"] is None:
            continue
        rsp = False
        rl = False
        for reg, s in zip(rec["regions"], rec["scan"]):
            if "error" in s:
                continue
            calls += 1
            hs.append(s["h"])
            if s["n_sign_changes"] > 0:
                calls_sp += 1
                rsp = True
                ratios_sp.append(s["min_over_max_abs_theta_deriv"])
            if s["n_regions_listed"] > 0:
                calls_listed += 1
                rl = True
                for L in s["listed"]:
                    listed_regions.append(dict(row=x["row"]["name"], names05=",".join(x["row"]["names05"] or []),
                                               band=x["band"], cls=x["cls"], regime=reg["regime"], group=reg["group"],
                                               n_sign_changes=s["n_sign_changes"], h=s["h"], **L))
        rows_sp += rsp
        rows_listed += rl
    out(f"   calls scanned {calls}; grid spacing h (log(1+z')): median {np.median(hs):.3g}, min {min(hs):.3g}, max {max(hs):.3g}")
    out(f"   calls with an interior stationary point (a sign change of theta' on the grid): {calls_sp}; rows: {rows_sp} of {len(sel)}")
    out(f"   calls that accept a region across one: {calls_listed}; rows: {rows_listed}; regions listed {len(listed_regions)} "
        f"(grid {sum(1 for L in listed_regions if L['how'] == 'grid')}, ends {sum(1 for L in listed_regions if L['how'] == 'ends')})")
    out("   LIMIT: a pair of stationary points closer together than the grid spacing h is not seen.")
    if listed_regions:
        dep = collections.Counter(L["depth"] for L in listed_regions)
        br = collections.Counter(L["branch"] for L in listed_regions)
        voa = [L["value_over_abserr"] for L in listed_regions if L["value_over_abserr"] is not None]
        out(f"   listed regions by depth {dict(sorted(dep.items()))}; by branch {dict(br)}; phase-limited "
            f"{sum(1 for L in listed_regions if L['phase_limited'])}")
        out(f"   listed regions' 1/relerr (|value|/abserr): median {np.median(voa):.3g}, min {min(voa):.3g}, max {max(voa):.3g}")
        lr = [L["depth"] for L in listed_regions]
        out(f"   listed regions by band {dict(collections.Counter(L['band'] for L in listed_regions))}; "
            f"by regime/group {dict(collections.Counter(L['regime'] + '/' + L['group'] for L in listed_regions).most_common(12))}; "
            f"by class {dict(collections.Counter(L['cls'] for L in listed_regions))}")
    out(f"   min|theta'|/max|theta'| on calls with a sign change: median {np.median(ratios_sp) if ratios_sp else float('nan'):.3g}")
    by3("rows with an interior stationary point", sel, lambda s: sum(1 for x in s if x["sp_calls"] > 0))
    by3("rows accepting a region across one", sel, lambda s: sum(1 for x in s if x["sp_listed"] > 0))
    if C1_rows := [x for x in sel if x["n_ge18"] > 0]:
        out(f"   rows failing C1 that hold a listed region: {sum(1 for x in C1_rows if x['sp_listed'] > 0)} of {len(C1_rows)}")
    deep = [x for x in sel if x["depth"] is not None and x["depth"] > 11]
    out(f"   rows deeper than 11 at 1e-7 holding an interior stationary point: {sum(1 for x in deep if x['sp_calls'] > 0)} of {len(deep)}")
    with open(OUT / "scan_listed_regions.csv", "w", newline="") as f:
        if listed_regions:
            w = csv.DictWriter(f, fieldnames=list(listed_regions[0].keys()))
            w.writeheader()
            for L in listed_regions:
                w.writerow({k: (v.hex() if isinstance(v, float) and k in ("start", "end") else v) for k, v in L.items()})
    A["scan"] = dict(calls=calls, calls_sp=calls_sp, rows_sp=rows_sp, calls_listed=calls_listed,
                     rows_listed=rows_listed, regions_listed=len(listed_regions),
                     h_median=float(np.median(hs)), h_min=float(min(hs)), h_max=float(max(hs)))

    # ------------------------------------------------------------------ item 8
    out("")
    out("== 8. Cost")
    for rtol in RTOLS:
        sel = [x for (rt, _), x in R.items() if rt == rtol]
        w = [x["wall"] for x in sel]
        rb = [x["rebuild"] for x in sel if x["rebuild"] is not None]
        sc = [x["scan_s"] for x in sel if x["scan_s"] is not None]
        out(f"   rtol {rtol}: Levin wall per unit median {np.median(w):.3g} s, 90th {pct(w, 90):.3g}, max {max(w):.3g}, "
            f"sum {sum(w) / 3600:.3f} h; rebuild sum {sum(rb) / 3600:.3f} h (max {max(rb):.3g} s, the first unit of each worker "
            f"carries its producer re-run); scan sum {sum(sc) / 3600:.3f} h")
        by3("Levin wall sum (s)", sel, lambda s: f"{sum(x['wall'] for x in s):.0f}")
        mx = max(sel, key=lambda x: x["wall"])
        out(f"     slowest unit {mx['uid']} {mx['wall']:.3g} s (depth {mx['depth']}, regions {mx['regions']})")
    st = [x["row"]["stored_compute_time"] or 0 for (rt, _), x in R.items() if rt == "1e-7"]
    out(f"   A3 v2's stored compute_time, summed over the 9,280 rows: {sum(st) / 3600:.2f} h")
    pg = [e for e in events if e["event"]["kind"] == "producers_G"]
    for e in sorted(pg, key=lambda e: e["worker"]):
        s = e["event"]["summary"]
        out(f"   producer re-run {e['worker']}: {s['timing']['total']:.1f} s ({s['G_objects_rerun']} GkWKBIntegration, "
            f"{s['G_samples_rerun']} samples; coefficient mismatches {s['G_coeff_mismatch_count']}; values bit-identical "
            f"to stored in every column {s['G_values_vs_stored']['n_bit_identical_in_every_column']}/{s['G_values_vs_stored']['n_values']}; "
            f"theta_div_2pi changes {s['G_values_vs_stored']['theta_div_2pi_changes']}, not one cycle near a boundary "
            f"{len(s['G_values_vs_stored']['theta_div_2pi_changes_not_one_cycle_near_boundary'])})")
    pt = [e for e in events if e["event"]["kind"] == "producers_T"]
    out(f"   T re-runs {len(pt)}: coefficients bit-identical to stored in "
        f"{sum(1 for e in pt if e['event']['info']['coeffs_bit_identical_to_stored'])}; theta_div_2pi changes "
        f"{sum(e['event']['T_values_vs_stored']['theta_div_2pi_changes'] for e in pt)}")
    qs = [e for e in events if e["event"]["kind"] == "QuadSource"]
    out(f"   QuadSources recomputed {len(qs)}; L0a holds on {sum(1 for e in qs if e['event']['L0a_holds'])}")

    # ------------------------------------------------------------------ item 9
    out("")
    out("== 9. Disagreement warnings (!! WARNING (GkSourcePolicyData)")
    wu = sum(x["d"]["policy_warnings"]["count"] for x in R.values())
    wg = [e for e in gk if (e["event"]["summary"]["policy_warnings"] or {}).get("count")]
    out(f"   per-unit counters, summed over {len(R)} units: {wu}; (k, z_response) assemblies that printed one: {len(wg)} of {len(gk)}")
    for e in wg[:50]:
        out(f"     {e['worker']} k {e['event']['k_serial']} z_response {e['event']['z_response_serial']}: "
            f"{e['event']['summary']['policy_warnings']}")
    stdout_n = sum(open(p, errors="replace").read().count("!! WARNING (GkSourcePolicyData") for p in run_dir.glob("worker_*.log"))
    out(f"   grep of the workers' logs: {stdout_n}")
    A["policy_warnings"] = dict(units=wu, assemblies=len(wg), grep=stdout_n)

    # ------------------------------------------------------------------ the CSV
    fields = ["rtol", "name", "names05", "shard", "serial", "k", "q", "r", "z_response", "band", "regime", "class",
              "stored_depth", "h_depth", "n_calls", "regions", "simple_regions", "levin_regions", "cc_regions",
              "regions_depth_ge_18", "total_converged", "calls_unconverged", "calls_phase_limited", "total_hex",
              "total_abserr_hex", "stored_total_hex", "stored_total_abserr_hex", "stored_total_converged",
              "stored_num_regions", "stored_compute_time", "status", "wall_s", "rebuild_s", "scan_s",
              "scan_calls_with_sign_change", "scan_sign_changes", "scan_regions_listed", "scan_regions_listed_grid",
              "scan_regions_listed_ends"]
    buf = io.StringIO()
    w = csv.writer(buf)
    for l in lines[:8]:
        buf.write(l + "\n")
    w.writerow(fields)
    for (rtol, name), x in sorted(R.items(), key=lambda t: (t[0][0], t[1]["row"]["shard"], t[1]["row"]["serial"])):
        r = x["row"]
        w.writerow([rtol, name, ",".join(r["names05"] or []), r["shard"], r["serial"], r["k"], r["q"], r["r"],
                    r["z_response"], x["band"], x["regime"], x["cls"], r["stored_depth"], x["depth"], x["n_calls"],
                    x["regions"], x["simple"], x["lev"], x["cc"], x["n_ge18"], x["conv"], x["calls_unconv"],
                    x["calls_plim"], None if x["total"] is None else x["total"].hex(),
                    None if x["abserr"] is None else x["abserr"].hex(),
                    (r["stored_total"] or {}).get("hex"), (r["stored_total_abserr"] or {}).get("hex"),
                    r["stored_total_converged"], r["stored_num_regions"], r["stored_compute_time"], x["status"],
                    repr(x["wall"]), repr(x["rebuild"]), repr(x["scan_s"]), x["sp_calls"], x["sp_changes"],
                    x["sp_listed"], x["sp_listed_grid"], x["sp_listed_ends"]])
    with gzip.open(OUT / "census_rows.csv.gz", "wt") as f:
        f.write(buf.getvalue())
    (OUT / "census_summary.txt").write_text("\n".join(lines) + "\n")
    with open(WIDER / "state" / "analysis.json", "w") as f:
        json.dump(A, f, indent=1, default=str)
    print("\n".join(lines))
    print(f"\n# wrote out/census_summary.txt, out/census_rows.csv.gz ({(OUT / 'census_rows.csv.gz').stat().st_size} bytes), "
          f"out/scan_listed_regions.csv, state/analysis.json")


if __name__ == "__main__":
    main(sys.argv[1])
