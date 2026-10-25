"""The phase-group census of A3 v2 (levin-phase-floor prompt 01, step 3).

Read-only and Ray-free, following docs/handover/a3-v2/after/scripts/depth_census.py: the primary and
every shard are opened with SQLite URI `mode=ro`; the `redshift`, `wavenumber_exit_time`/`wavenumber`
and `tolerance` maps come from shard 0; `tolerance.log10_tol` is log10 of the value.

eta(z) and EPS are IMPORTED (not copied) from docs/handover/a3-v2/findings/depth20_floor.py, by file
path, so the floor F = EPS * k * eta(z_response) is exactly the quantity of findings §2.1.

    ./venv/bin/python -I var/levin-phase-floor-01/scripts/group_census.py \
        var/datastores/handover-A3-v2-lambdacdm.sqlite \
        [--hashes var/levin-phase-floor-01/state/store_files_before.txt]

Writes, under var/levin-phase-floor-01/out/:
    groups_01.csv      one flat record per group per Levin sub-interval (step 3(b))
    census_01.txt      T1-T5 and the section 2.5 confirmations (steps 3(c), 3(d))
    candidates_01.txt  candidate rows for prompt 03 (step 3(e))

Exit status 0; 2 on a stop condition of prompt section 4 (population != 9280, unparseable metadata
or no WKB_Levin key, a depth-20 row with no group records).
"""

import argparse
import collections
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys

import numpy as np

REPO = os.getcwd()
OUTDIR = os.path.join(REPO, "var/levin-phase-floor-01/out")
PAIR = (1e-32, 1e-7)
EXPECT = 9280
RTOL = 1e-7
DEPTH = 20
STORE_FILES = [
    "handover-A3-v2-lambdacdm.sqlite",
    "handover-A3-v2-lambdacdm-shard0000.sqlite",
    "handover-A3-v2-lambdacdm-shard0001.sqlite",
    "handover-A3-v2-lambdacdm-shard0002.sqlite",
    "handover-A3-v2-lambdacdm-shard0003.sqlite",
    "handover-A3-v2-lambdacdm.manifest.json",
]
FLOOR_SCRIPT = "docs/handover/a3-v2/findings/depth20_floor.py"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ro(path):
    import sqlite3

    return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)


def load_floor_module():
    spec = importlib.util.spec_from_file_location(
        "depth20_floor", os.path.join(REPO, FLOOR_SCRIPT)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def band(z):
    # as depth20_floor.py / depth_census.py: low z < 9, mid 9 <= z <= 1e5, seam z > 1e5
    return "low" if z < 9.0 else ("mid" if z <= 1.0e5 else "seam")


def regime_string(regime):
    return "".join(n for n, f in zip("Gqr", regime) if f)


def parse_signs(label):
    """'G+q-r' -> {'G': +1, 'q': +1, 'r': -1}; the inverse of phase_groups.signs_label."""
    signs = {}
    i = 0
    sign = +1
    while i < len(label):
        c = label[i]
        if c == "+":
            sign = +1
        elif c == "-":
            sign = -1
        else:
            if c not in "Gqr":
                raise ValueError(f"unexpected label {label!r}")
            signs[c] = sign
            sign = +1
        i += 1
    return signs


def cancelling(label, q_is_k, r_is_k):
    """Prompt section 1: a record containing G is cancelling if G and a transfer factor carry the
    same sign and that factor's wavenumber equals k. Returns None for a record without G."""
    s = parse_signs(label)
    if "G" not in s:
        return None
    for f, eq in (("q", q_is_k), ("r", r_is_k)):
        if f in s and s[f] == s["G"] and eq:
            return True
    return False


def geometry(k_s, q_s, r_s, q, r):
    if q_s == k_s and r_s == k_s:
        return "q==r==k"
    if r_s == k_s and q < r:
        return "r==k,q<r"
    if q_s == k_s:
        return "q==k"
    return "other"


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def store_show_fingerprint(primary):
    out = subprocess.run(
        [sys.executable, "-m", "runregistrykit", "store", "show", primary],
        capture_output=True,
        text=True,
        cwd=REPO,
    ).stdout
    start = out.index("fields:\n") + len("fields:\n")
    end = out.index("runs naming")
    fields = json.loads(out[start:end])
    problems = [l for l in out.splitlines() if l.startswith("problems:")]
    return fields["fingerprint"]["digest"], problems[0] if problems else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primary")
    ap.add_argument("--hashes", help="state/store_files_before.txt, to compare against")
    a = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    lines = []
    out = lines.append

    # ---------------------------------------------------------------- provenance header
    me = os.path.abspath(__file__)
    store_dir = os.path.dirname(os.path.abspath(a.primary))
    file_hashes = {f: sha256(os.path.join(store_dir, f)) for f in STORE_FILES}
    step2 = {}
    if a.hashes:
        for l in open(a.hashes):
            if l.startswith("#") or not l.strip() or l.startswith("-"):
                continue
            parts = l.split()
            if len(parts) == 4:
                step2[os.path.basename(parts[3])] = parts[0]
    fp, problems = store_show_fingerprint(a.primary)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO
    ).stdout.strip()
    floor_mod = load_floor_module()
    EPS = float(floor_mod.EPS)
    eta = floor_mod.eta

    header = [
        f"# group_census.py sha256 {sha256(me)}",
        f"# {FLOOR_SCRIPT} sha256 {sha256(os.path.join(REPO, FLOOR_SCRIPT))} (eta(z), EPS imported from it)",
        f"# git HEAD {head}",
        f"# store fingerprint (runregistrykit store show) {fp}; {problems}",
    ]
    for f in STORE_FILES:
        tag = ""
        if step2:
            tag = (
                " (= step 2)" if step2.get(f) == file_hashes[f] else " (DIFFERS FROM STEP 2)"
            )
        header.append(f"# store file sha256 {file_hashes[f]}  {f}{tag}")
    header.append(f"# EPS = {EPS!r}; RTOL = {RTOL}; population pair {PAIR}")
    for h in header:
        out(h)

    # ---------------------------------------------------------------- (a) the population
    p = ro(a.primary)
    shards = [
        os.path.join(store_dir, os.path.basename(f))
        for (f,) in p.execute("SELECT filename FROM shards ORDER BY serial")
    ]
    p.close()
    c0 = ro(shards[0])
    zmap = dict(c0.execute("select serial, z from redshift"))
    kmap = dict(
        c0.execute(
            "select e.serial, w.k_inv_Mpc from wavenumber_exit_time e "
            "join wavenumber w on w.serial = e.wavenumber_serial"
        )
    )
    tolmap = {s: 10.0**l for s, l in c0.execute("select serial, log10_tol from tolerance")}
    c0.close()

    cols = (
        "serial, k_wavenumber_exit_serial, q_wavenumber_exit_serial, r_wavenumber_exit_serial, "
        "z_response_serial, atol_serial, rtol_serial, total, total_abserr, total_converged, "
        'total_phase_limited, compute_time, "WKB_Levin_max_depth", eta_response, metadata, '
        "typeof(total_abserr), cast(total_abserr as text), typeof(total)"
    )
    rows, offpair, stops = [], 0, []
    for i, sh in enumerate(shards):
        c = ro(sh)
        for t in c.execute(f"select {cols} from QuadSourceIntegral"):
            (serial, ks, qs, rs, zs, at, rt, tot, abserr, conv, plim, ct, depth, eta_resp,
             md, ty_abserr, txt_abserr, ty_total) = t
            if not (
                math.isclose(tolmap[at], PAIR[0], rel_tol=1e-9)
                and math.isclose(tolmap[rt], PAIR[1], rel_tol=1e-9)
            ):
                offpair += 1
                continue
            try:
                m = json.loads(md)
            except Exception as e:  # noqa: BLE001
                stops.append(f"shard {i} serial {serial}: metadata does not parse: {e}")
                continue
            if "WKB_Levin" not in m:
                stops.append(f"shard {i} serial {serial}: metadata has no WKB_Levin key")
                continue
            rows.append(
                dict(
                    shard=i, serial=serial, ks=ks, qs=qs, rs=rs,
                    k=kmap[ks], q=kmap[qs], r=kmap[rs], z=zmap[zs],
                    total=tot, total_abserr=abserr, total_converged=conv,
                    total_phase_limited=plim, compute_time=ct, depth=depth,
                    eta_response=eta_resp, m=m, ty_abserr=ty_abserr,
                    txt_abserr=txt_abserr, ty_total=ty_total,
                    raw_md_has_inf=("Infinity" in md), raw_md_has_nan=("NaN" in md),
                )
            )
        c.close()

    out("")
    out(f"== (a) population: QuadSourceIntegral rows at (atol, rtol) = {PAIR}: {len(rows)} "
        f"(expected {EXPECT}: {'yes' if len(rows) == EXPECT else 'NO -- STOP'})")
    out(f"   rows at any other pair: {offpair}")
    out(f"   by shard: {[sum(1 for r in rows if r['shard'] == i) for i in range(len(shards))]}")
    out(f"   metadata parse failures / missing WKB_Levin: {len(stops)}")
    for s in stops:
        out(f"   STOP: {s}")
    if len(rows) != EXPECT:
        stops.append(f"population {len(rows)} != {EXPECT}")

    # floor per distinct z_response
    etas = {z: eta(z) for z in sorted({r["z"] for r in rows})}

    # eta check against the stored eta_response column (descriptive)
    ratios = collections.defaultdict(list)
    for r in rows:
        if r["eta_response"]:
            ratios[band(r["z"])].append(etas[r["z"]] / r["eta_response"])
    out("   check: depth20_floor eta(z_response) / stored eta_response, by band (min, median, max):")
    for b in ("low", "mid", "seam"):
        v = ratios[b]
        if v:
            out(f"     {b:4s} n={len(v):5d}  {min(v):.6f}  {float(np.median(v)):.6f}  {max(v):.6f}")

    # ---------------------------------------------------------------- (b) flat records
    recs = []
    d20_without_groups = []
    for r in rows:
        r["band"] = band(r["z"])
        q_is_k = r["qs"] == r["ks"]
        r_is_k = r["rs"] == r["ks"]
        assert q_is_k == (r["q"] == r["k"]) and r_is_k == (r["r"] == r["k"])
        r["geometry"] = geometry(r["ks"], r["qs"], r["rs"], r["q"], r["r"])
        r["F"] = float(EPS * r["k"] * etas[r["z"]])
        subs = r["m"]["WKB_Levin"]["subintervals"]
        n = 0
        for si, sub in enumerate(subs):
            reg = regime_string(sub["regime"])
            for g in sub["groups"]:
                n += 1
                v = g["value"]
                e = g["abserr"]
                eg = (e / abs(v)) if v != 0 else (math.inf if e != 0 else math.nan)
                recs.append(
                    dict(
                        row=r, sub_index=si, z_min=sub["z_min"], z_max=sub["z_max"],
                        regime=reg, label=g["label"], value=v, abserr=e,
                        converged=g["converged"], phase_limited=g["phase_limited"],
                        regions=g["regions"], simple_regions=g["simple_regions"],
                        evaluations=g["evaluations"], elapsed=g["elapsed"],
                        cancelling=cancelling(g["label"], q_is_k, r_is_k),
                        e_g=eg,
                    )
                )
        r["n_groups"] = n
        if r["depth"] == DEPTH and n == 0:
            d20_without_groups.append(r)

    out(f"\n== (b) group records written to groups_01.csv: {len(recs)} records from "
        f"{sum(1 for r in rows if r['n_groups'])} rows with a Levin sub-interval "
        f"({sum(1 for r in rows if not r['n_groups'])} rows have none)")
    out(f"   depth-20 rows: {sum(1 for r in rows if r['depth'] == DEPTH)}; "
        f"depth-20 rows with no group records: {len(d20_without_groups)}")
    for r in d20_without_groups:
        stops.append(f"depth-20 row shard {r['shard']} serial {r['serial']} has no group records")
        out(f"   STOP: depth-20 row shard {r['shard']} serial {r['serial']} has no group records")
    geo = collections.Counter(r["geometry"] for r in rows)
    out(f"   geometry of the {len(rows)} rows: {dict(geo)}")
    rk_qgt = sum(1 for r in rows if r["rs"] == r["ks"] and r["qs"] != r["ks"] and r["q"] > r["r"])
    out(f"   rows with r == k and q > r (classed 'other' by the prompt's wording): {rk_qgt}")
    out(f"   labels seen: {sorted(collections.Counter(x['label'] for x in recs).items())}")
    out(f"   records without G (neither class): {sum(1 for x in recs if x['cancelling'] is None)}")

    fields = [
        "shard", "serial", "k", "q", "r", "z_response", "band", "geometry",
        "WKB_Levin_max_depth", "compute_time", "total", "total_abserr", "total_converged",
        "sub_index", "z_min", "z_max", "regime", "label", "value", "abserr", "converged",
        "phase_limited", "regions", "simple_regions", "evaluations", "elapsed",
        "cancelling", "F", "e_g",
    ]
    with open(os.path.join(OUTDIR, "groups_01.csv"), "w", newline="") as fh:
        for h in header:
            fh.write(h + "\n")
        w = csv.writer(fh)
        w.writerow(fields)
        for x in recs:
            r = x["row"]
            w.writerow([
                r["shard"], r["serial"], repr(r["k"]), repr(r["q"]), repr(r["r"]), repr(r["z"]),
                r["band"], r["geometry"],
                "" if r["depth"] is None else r["depth"], repr(r["compute_time"]),
                repr(r["total"]), repr(r["total_abserr"]), r["total_converged"],
                x["sub_index"], repr(x["z_min"]), repr(x["z_max"]), x["regime"], x["label"],
                repr(x["value"]), repr(x["abserr"]), int(bool(x["converged"])),
                int(bool(x["phase_limited"])), x["regions"], x["simple_regions"],
                x["evaluations"], repr(x["elapsed"]),
                "" if x["cancelling"] is None else int(x["cancelling"]),
                repr(r["F"]), repr(x["e_g"]),
            ])

    d20 = [x for x in recs if x["row"]["depth"] == DEPTH]
    d20rows = [r for r in rows if r["depth"] == DEPTH]

    # ---------------------------------------------------------------- (c) T1
    def t1_block(title, sel):
        el = collections.Counter()
        rg = collections.Counter()
        nr = collections.Counter()
        for x in sel:
            key = (x["regime"], x["label"])
            el[key] += x["elapsed"]
            rg[key] += x["regions"]
            nr[key] += 1
        te, tr, tn = sum(el.values()), sum(rg.values()), sum(nr.values())
        out(f"\n   -- {title}: {tn} records; {te / 3600:.3f} CPU-hr; {tr:,} regions")
        out(f"   {'regime':6s} {'label':7s} {'CPU-hr':>10s} {'share':>7s} {'regions':>13s} {'share':>7s} {'records':>8s} {'share':>7s}")
        for key in sorted(el, key=lambda k_: -el[k_]):
            out(f"   {key[0]:6s} {key[1]:7s} {el[key] / 3600:10.3f} {el[key] / te if te else 0:7.1%} "
                f"{rg[key]:13,d} {rg[key] / tr if tr else 0:7.1%} {nr[key]:8d} {nr[key] / tn if tn else 0:7.1%}")
        return el

    out("\n== T1. per-group Levin elapsed (CPU-hr), regions and record count by (regime, label)")
    t1_block("all rows", recs)
    t1_block("depth-20 rows", d20)

    # ---------------------------------------------------------------- T2
    out("\n== T2. the group with the largest elapsed in each depth-20 row")
    top = {}
    for x in d20:
        r = x["row"]
        key = (r["shard"], r["serial"])
        if key not in top or x["elapsed"] > top[key]["elapsed"]:
            top[key] = x
    row_levin = collections.Counter()
    for x in d20:
        row_levin[(x["row"]["shard"], x["row"]["serial"])] += x["elapsed"]
    cnt = collections.Counter(
        (x["row"]["geometry"], x["regime"], x["label"]) for x in top.values()
    )
    out(f"   rows: {len(top)}")
    out(f"   {'geometry':10s} {'regime':6s} {'label':7s} {'rows':>5s} {'cancelling':>10s}")
    for key, v in cnt.most_common():
        ex = next(x for x in top.values() if (x["row"]["geometry"], x["regime"], x["label"]) == key)
        out(f"   {key[0]:10s} {key[1]:6s} {key[2]:7s} {v:5d} {str(ex['cancelling']):>10s}")
    shares = [x["elapsed"] / row_levin[k_] for k_, x in top.items() if row_levin[k_] > 0]
    out(f"   median share of the row's Levin elapsed taken by its top group: {float(np.median(shares)):.3f} "
        f"(10th {pct(shares, 10):.3f}, 90th {pct(shares, 90):.3f}; {len(shares)} rows)")
    for lim in (60.0, 100.0):
        sel = [x["elapsed"] / row_levin[k_] for k_, x in top.items()
               if row_levin[k_] > 0 and (x["row"]["compute_time"] or 0) > lim]
        out(f"   the same over depth-20 rows with compute_time > {lim:g} s: median {float(np.median(sel)):.3f} ({len(sel)} rows)")
    cls = collections.Counter(
        ("none" if x["cancelling"] is None else ("cancelling" if x["cancelling"] else "non-cancelling"))
        for x in top.values()
    )
    out(f"   top group's class: {dict(cls)}")

    # ---------------------------------------------------------------- T3
    out("\n== T3. T1's depth-20 column split by band and by k (CPU-hr; share of the slice; regions; records)")
    for title, keyf, keys in (
        ("band", lambda x: x["row"]["band"], ("low", "mid", "seam")),
        ("k /Mpc", lambda x: x["row"]["k"], sorted({x["row"]["k"] for x in d20})),
    ):
        for kv in keys:
            sel = [x for x in d20 if keyf(x) == kv]
            if not sel:
                continue
            nrows = len({(x["row"]["shard"], x["row"]["serial"]) for x in sel})
            t1_block(f"{title} = {kv if isinstance(kv, str) else f'{kv:.4e}'} ({nrows} depth-20 rows)", sel)

    # ---------------------------------------------------------------- T4
    out("\n== T4. the P1-P3 statistics, as prompt section 1 defines them")
    tot_el = sum(x["elapsed"] for x in d20)
    nc_el = sum(x["elapsed"] for x in d20 if x["cancelling"] is False)
    c_el = sum(x["elapsed"] for x in d20 if x["cancelling"] is True)
    ng_el = sum(x["elapsed"] for x in d20 if x["cancelling"] is None)
    S = nc_el / tot_el
    out(f"   P1: summed per-group elapsed over all depth-20 rows: {tot_el / 3600:.4f} CPU-hr")
    out(f"       on non-cancelling records {nc_el / 3600:.4f}; cancelling {c_el / 3600:.4f}; "
        f"records without G {ng_el / 3600:.6f}")
    out(f"       S = {S:.4f}   (S over G-containing records only: {nc_el / (nc_el + c_el):.4f})")
    for b in ("low", "mid", "seam"):
        te = sum(x["elapsed"] for x in d20 if x["row"]["band"] == b)
        ne = sum(x["elapsed"] for x in d20 if x["row"]["band"] == b and x["cancelling"] is False)
        if te:
            out(f"       descriptive: S within the {b} band = {ne / te:.4f} ({te / 3600:.3f} CPU-hr)")

    out("   P2: depth-20 records, mid and low bands, e_g finite and non-zero; fit log10 e_g on log10 F")
    p2 = {}
    for name, flag in (("cancelling", True), ("non-cancelling", False)):
        sel = [x for x in d20 if x["cancelling"] is flag and x["row"]["band"] in ("low", "mid")
               and math.isfinite(x["e_g"]) and x["e_g"] > 0]
        excl = [x for x in d20 if x["cancelling"] is flag and x["row"]["band"] in ("low", "mid")
                and not (math.isfinite(x["e_g"]) and x["e_g"] > 0)]
        F = np.array([x["row"]["F"] for x in sel])
        e = np.array([x["e_g"] for x in sel])
        slope, icpt = np.polyfit(np.log10(F), np.log10(e), 1) if len(sel) >= 2 else (float("nan"),) * 2
        ratio = e / F
        p2[name] = dict(n=len(sel), slope=slope, median=pct(ratio, 50), p10=pct(ratio, 10), p90=pct(ratio, 90))
        out(f"       {name:15s} n = {len(sel):6d} (excluded, e_g zero or non-finite: {len(excl)}); "
            f"slope {slope:.4f}, intercept {icpt:.4f}; e_g/F median {pct(ratio, 50):.4g}, "
            f"10th {pct(ratio, 10):.4g}, 90th {pct(ratio, 90):.4g}; log10 F spans "
            f"[{np.log10(F).min():.2f}, {np.log10(F).max():.2f}]")
        for b in ("low", "mid"):
            sb = [x for x in sel if x["row"]["band"] == b]
            if len(sb) >= 2:
                Fb = np.array([x["row"]["F"] for x in sb]); eb = np.array([x["e_g"] for x in sb])
                sl, _ = np.polyfit(np.log10(Fb), np.log10(eb), 1)
                out(f"         descriptive, {b} band only: n = {len(sb)}, slope {sl:.4f}, "
                    f"e_g/F median {pct(eb / Fb, 50):.4g} (10th {pct(eb / Fb, 10):.4g}, 90th {pct(eb / Fb, 90):.4g})")
        labs = sorted({x["label"] for x in sel})
        for lab in labs:
            sb = [x for x in sel if x["label"] == lab]
            if len(sb) >= 2:
                Fb = np.array([x["row"]["F"] for x in sb]); eb = np.array([x["e_g"] for x in sb])
                sl, _ = np.polyfit(np.log10(Fb), np.log10(eb), 1)
                out(f"         descriptive, label {lab:6s}: n = {len(sb):5d}, slope {sl:.4f}, "
                    f"e_g/F median {pct(eb / Fb, 50):.4g}")
    mc, mn = p2["cancelling"]["median"], p2["non-cancelling"]["median"]
    out(f"       ratio of the two class medians (cancelling / non-cancelling): {mc / mn:.4g}")

    out("   P3: share of elapsed on depth-20 records with F > rtol = 1e-7")
    p3 = {}
    for name, flag in (("non-cancelling", False), ("cancelling", True)):
        sel = [x for x in d20 if x["cancelling"] is flag]
        te = sum(x["elapsed"] for x in sel)
        ae = sum(x["elapsed"] for x in sel if x["row"]["F"] > RTOL)
        p3[name] = ae / te
        out(f"       {name:15s}: {ae / 3600:.4f} of {te / 3600:.4f} CPU-hr = {ae / te:.4f}")

    # ---------------------------------------------------------------- T5
    out("\n== T5. the top group's sub-interval in source time, depth-20 rows: quartiles (25/50/75) of log(1+z)")
    groups5 = collections.defaultdict(list)
    for x in top.values():
        groups5[(x["row"]["band"], x["regime"], x["label"])].append(x)
    out(f"   {'band':4s} {'regime':6s} {'label':7s} {'rows':>5s}   {'log(1+z_max) q1 / med / q3':>30s}   {'log(1+z_min) q1 / med / q3':>30s}")
    for key in sorted(groups5):
        xs = groups5[key]
        hi = [math.log1p(x["z_max"]) for x in xs]
        lo = [math.log1p(x["z_min"]) for x in xs]
        out(f"   {key[0]:4s} {key[1]:6s} {key[2]:7s} {len(xs):5d}   "
            f"{pct(hi, 25):9.3f} {pct(hi, 50):9.3f} {pct(hi, 75):9.3f}   "
            f"{pct(lo, 25):9.3f} {pct(lo, 50):9.3f} {pct(lo, 75):9.3f}")
    zr = collections.defaultdict(list)
    for x in top.values():
        zr[x["row"]["band"]].append(math.log1p(x["row"]["z"]))
    out("   for reference, log(1+z_response) of the same rows by band: " + "; ".join(
        f"{b}: {pct(v, 25):.3f} / {pct(v, 50):.3f} / {pct(v, 75):.3f}" for b, v in sorted(zr.items())))

    # ---------------------------------------------------------------- (d) findings 2.5
    out("\n== (d) findings 2.5, confirmed on the store")
    nonfin = [r for r in rows if r["total_abserr"] is None or not math.isfinite(r["total_abserr"])]
    out(f"   rows whose total_abserr is not finite: {len(nonfin)}")
    out(f"   SQLite typeof(total_abserr) over the population: "
        f"{dict(collections.Counter(r['ty_abserr'] for r in rows))}")
    out(f"   SQLite cast(total_abserr as text) on those rows: "
        f"{dict(collections.Counter(r['txt_abserr'] for r in nonfin))}; Python values read: "
        f"{dict(collections.Counter(repr(r['total_abserr']) for r in nonfin))}")
    nf_recs = [x for x in recs if not math.isfinite(x["abserr"])]
    nf_keys = {(x["row"]["shard"], x["row"]["serial"]) for x in nf_recs}

    def dsum(title, keyf, sel):
        out(f"     by {title}: " + ", ".join(f"{k_}: {v}" for k_, v in sorted(collections.Counter(map(keyf, sel)).items(), key=lambda t: str(t[0]))))

    out("   summary of the non-finite rows:")
    dsum("WKB_Levin_max_depth bin", lambda r: "NULL" if r["depth"] is None else ("20" if r["depth"] == DEPTH else "1-19"), nonfin)
    dsum("band", lambda r: r["band"], nonfin)
    dsum("(k, q, r) /Mpc", lambda r: f"({r['k']:.3e}, {r['q']:.3e}, {r['r']:.3e})", nonfin)
    dsum("q == r", lambda r: r["qs"] == r["rs"], nonfin)
    dsum("total_converged", lambda r: r["total_converged"], nonfin)
    dsum("total_phase_limited", lambda r: r["total_phase_limited"], nonfin)
    out(f"     every non-finite row has a non-finite group abserr: {all((r['shard'], r['serial']) in nf_keys for r in nonfin)}; "
        f"numeric_quad.abserr finite on all: {all(math.isfinite(r['m']['numeric_quad']['abserr']) for r in nonfin)}")
    dsum("non-finite group record (regime, label)", lambda x: (x["regime"], x["label"]), nf_recs)
    dsum("non-finite group record value", lambda x: "zero" if x["value"] == 0 else ("finite non-zero" if math.isfinite(x["value"]) else "non-finite"), nf_recs)
    dsum("non-finite group record abserr", lambda x: repr(x["abserr"]), nf_recs)
    dsum("non-finite group record (converged, phase_limited)", lambda x: (bool(x["converged"]), bool(x["phase_limited"])), nf_recs)
    dsum("non-finite group record regions", lambda x: x["regions"], nf_recs)
    dsum("non-finite group record sub-interval index", lambda x: x["sub_index"], nf_recs)
    out(f"     non-finite group records with q == r: {sum(1 for x in nf_recs if x['row']['qs'] == x['row']['rs'])} of {len(nf_recs)}")
    qr_eq = [x for x in recs if x["row"]["qs"] == x["row"]["rs"] and x["label"] in ("G+q-r", "G-q+r")]
    out(f"     for comparison, all G+q-r / G-q+r records on q == r rows: {len(qr_eq)}; of them abserr non-finite: "
        f"{sum(1 for x in qr_eq if not math.isfinite(x['abserr']))}")
    out("   the non-finite rows, one by one:")
    for r in nonfin:
        out(f"   shard {r['shard']} serial {r['serial']}: k {r['k']:.4e} q {r['q']:.4e} r {r['r']:.4e} "
            f"z_response {r['z']:.6g} ({r['band']}), depth {r['depth']}, compute_time {r['compute_time']:.3f} s, "
            f"total_converged {r['total_converged']}, total {r['total']!r}")
        out(f"       as read: Python {r['total_abserr']!r}; SQLite typeof '{r['ty_abserr']}', "
            f"cast(total_abserr as text) '{r['txt_abserr']}'; metadata JSON text contains 'Infinity': "
            f"{r['raw_md_has_inf']}, 'NaN': {r['raw_md_has_nan']}; WKB_Levin.abserr {r['m']['WKB_Levin']['abserr']!r}; "
            f"numeric_quad.abserr {r['m']['numeric_quad']['abserr']!r}")
        for x in recs:
            if x["row"] is r and not math.isfinite(x["abserr"]):
                out(f"       non-finite group abserr: sub-interval {x['sub_index']} "
                    f"(z {x['z_max']:.6g} -> {x['z_min']:.6g}, regime {x['regime']}), group {x['label']}: "
                    f"abserr {x['abserr']!r}, value {x['value']!r}, converged {x['converged']}, "
                    f"phase_limited {x['phase_limited']}, regions {x['regions']}, elapsed {x['elapsed']:.3f} s")
        for si, sub in enumerate(r["m"]["numeric_quad"]["subintervals"]):
            if not math.isfinite(sub["abserr"]):
                out(f"       non-finite numeric_quad sub-interval {si}: {sub}")
    nf_groups = [x for x in recs if not math.isfinite(x["abserr"])]
    out(f"   group records with non-finite abserr, store-wide: {len(nf_groups)} "
        f"(on {len({(x['row']['shard'], x['row']['serial']) for x in nf_groups})} rows)")

    out("   findings 2.5 item 1, recomputed (depth-20 rows; rel = total_abserr/|total|):")
    for label, sel in (
        ("total_converged = 1 and rel finite (depth20_floor.py's 'flagged')",
         [r for r in d20rows if r["total_converged"] and r["total"] and math.isfinite(r["total_abserr"] / abs(r["total"]))]),
        ("total_converged = 1, any rel", [r for r in d20rows if r["total_converged"]]),
    ):
        rel = [r["total_abserr"] / abs(r["total"]) if r["total"] else math.nan for r in sel]
        frel = [v for v in rel if math.isfinite(v)]
        out(f"     {label}: {len(sel)} rows; rel > 1e-6: {sum(1 for v in frel if v > 1e-6)}; "
            f"rel > 1e-5: {sum(1 for v in frel if v > 1e-5)}; rel > 1e-4: {sum(1 for v in frel if v > 1e-4)}; "
            f"max rel {max(frel):.4e}; non-finite rel: {len(rel) - len(frel)}")
    out(f"     depth-20 rows {len(d20rows)}; total_converged = 0: {sum(1 for r in d20rows if not r['total_converged'])}; "
        f"zero total: {sum(1 for r in d20rows if not r['total'])}")

    out("   group level: records flagged converged with e_g > 1e-6 and > 1e-4")
    for title, sel in (("depth-20 rows", d20), ("all rows", recs)):
        cv = [x for x in sel if x["converged"]]
        fin = [x for x in cv if math.isfinite(x["e_g"])]
        out(f"     {title}: {len(cv)} converged records of {len(sel)}; e_g > 1e-6: "
            f"{sum(1 for x in fin if x['e_g'] > 1e-6)}; e_g > 1e-4: {sum(1 for x in fin if x['e_g'] > 1e-4)}; "
            f"non-finite e_g (value 0 or abserr non-finite): {len(cv) - len(fin)}; "
            f"max finite e_g {max(x['e_g'] for x in fin):.4e}")
        for name, flag in (("cancelling", True), ("non-cancelling", False), ("no G", None)):
            sc = [x for x in fin if x["cancelling"] is flag]
            if sc:
                out(f"       {name:15s}: {len(sc)} finite; e_g > 1e-6: {sum(1 for x in sc if x['e_g'] > 1e-6)}; "
                    f"e_g > 1e-4: {sum(1 for x in sc if x['e_g'] > 1e-4)}")

    # ---------------------------------------------------------------- (e) candidates for 03
    cand_lines = list(header)
    cand_lines.append("# candidate rows for prompt 03 (README 4.3): top group by per-group elapsed, depth-20 rows")
    cand_lines.append("# preference: compute_time in [100, 800] s, mid band; low band [100, 800] s where the mid band has none")
    classes = [
        ("G+r (slow, cancelling) top on r == k", lambda x: x["label"] == "G+r" and x["row"]["rs"] == x["row"]["ks"]),
        ("G-r (fast) top on r == k", lambda x: x["label"] == "G-r" and x["row"]["rs"] == x["row"]["ks"]),
        ("lone G top", lambda x: x["label"] == "G" and x["regime"] == "G"),
        ("a Gqr group top", lambda x: x["regime"] == "Gqr"),
    ]
    out("\n== (e) candidates for 03 (detail in candidates_01.txt)")
    for title, pred in classes:
        allc = [x for x in top.values() if pred(x)]
        mid = [x for x in allc if x["row"]["band"] == "mid" and 100 <= (x["row"]["compute_time"] or 0) <= 800]
        low = [x for x in allc if x["row"]["band"] == "low" and 100 <= (x["row"]["compute_time"] or 0) <= 800]
        pick = sorted(mid, key=lambda x: (x["row"]["k"], x["row"]["compute_time"]))
        note = f"mid band 100-800 s: {len(mid)}; low band 100-800 s: {len(low)}; all depth-20 rows with this top group: {len(allc)}"
        if len(pick) < 2:
            pick = pick + sorted(low, key=lambda x: (x["row"]["k"], x["row"]["compute_time"]))
            note += "; mid band has fewer than two, low band added"
        if len(pick) < 2:
            note += "; FEWER THAN TWO in 100-800 s in mid or low band"
        # thin to at most 8, spread over k
        if len(pick) > 8:
            byk = collections.defaultdict(list)
            for x in pick:
                byk[x["row"]["k"]].append(x)
            thinned = []
            while len(thinned) < 8 and any(byk.values()):
                for kk in sorted(byk):
                    if byk[kk] and len(thinned) < 8:
                        thinned.append(byk[kk].pop(len(byk[kk]) // 2))
            pick = sorted(thinned, key=lambda x: (x["row"]["k"], x["row"]["compute_time"]))
            note += f"; thinned to {len(pick)} spread over k"
        out(f"   {title}: {note}")
        cand_lines.append("")
        cand_lines.append(f"== {title}")
        cand_lines.append(f"   {note}")
        for x in pick:
            r = x["row"]
            rel = r["total_abserr"] / abs(r["total"]) if r["total"] else math.nan
            cand_lines.append("")
            cand_lines.append(
                f"   (shard, serial) = ({r['shard']}, {r['serial']}); k {r['k']:.6e} q {r['q']:.6e} r {r['r']:.6e} /Mpc; "
                f"z_response {r['z']!r} ({r['band']}); geometry {r['geometry']}; compute_time {r['compute_time']:.2f} s; "
                f"depth {r['depth']}; total {r['total']:.6e}; total_abserr/|total| {rel:.3e}; "
                f"total_converged {r['total_converged']}; F {r['F']:.3e}"
            )
            cand_lines.append(f"     top group {x['regime']} {x['label']} (sub-interval {x['sub_index']}), "
                              f"{x['elapsed']:.2f} s of the row's {row_levin[(r['shard'], r['serial'])]:.2f} s Levin elapsed")
            cand_lines.append(f"     {'sub':>3s} {'z_max':>12s} {'z_min':>12s} {'regime':6s} {'label':7s} {'value':>13s} {'abserr':>11s} "
                              f"{'e_g':>10s} {'conv':>4s} {'plim':>4s} {'regions':>9s} {'simple':>9s} {'evals':>10s} {'elapsed s':>10s} {'canc':>5s}")
            for y in recs:
                if y["row"] is r:
                    cand_lines.append(
                        f"     {y['sub_index']:3d} {y['z_max']:12.5e} {y['z_min']:12.5e} {y['regime']:6s} {y['label']:7s} "
                        f"{y['value']:13.5e} {y['abserr']:11.3e} {y['e_g']:10.3e} {int(bool(y['converged'])):4d} "
                        f"{int(bool(y['phase_limited'])):4d} {y['regions']:9d} {y['simple_regions']:9d} "
                        f"{y['evaluations']:10d} {y['elapsed']:10.2f} {str(y['cancelling']):>5s}"
                    )
    with open(os.path.join(OUTDIR, "candidates_01.txt"), "w") as fh:
        fh.write("\n".join(cand_lines) + "\n")

    out("\n== stop conditions met: " + (str(len(stops)) if stops else "none"))
    for s in stops:
        out(f"   {s}")

    with open(os.path.join(OUTDIR, "census_01.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 2 if stops else 0


if __name__ == "__main__":
    sys.exit(main())
