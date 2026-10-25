"""
levin-phase-floor prompt 03, step 7: the analysis, from the run's pickles alone.

Usage (repository root):
    OMP_NUM_THREADS=1 PYTHONPATH=. ./venv/bin/python var/levin-phase-floor-03/scripts/analyse.py \
        RUN_DIR > var/levin-phase-floor-03/out/instrument_03.txt

Nothing is re-run: every quantity comes from the pickled PhaseGroups, Levin returns and model
functions. Each "driver discrepancy" below is computed by the driver's own single-region routine,
``AdaptiveLevin.levin_quadrature._adaptive_levin_subregion``, called on the span with the group's
own f and theta dict, at the call's chebyshev_order and rtol:
  - if the routine routes the span to Clenshaw-Curtis (``is_direct``), the discrepancy is its
    nested-pair ``abserr_direct`` (orders CHEBYSHEV_ORDER and 2N-1);
  - otherwise it is |v0 - vL - vR|, the three solves on (a, b), (a, c), (c, b), c = (a + b)/2.
These are the driver's own formulae (levin_quadrature.py:1984, 2129-2131); the analysis
checks, at every sampled depth-20 region, that the recomputed discrepancy equals the recorded
region.abserr exactly.

Definitions, fixed before anything is computed (prompt section 2 step 7, and the choices the
prompt leaves open, stated here once):
  E        group.envelope(midpoint of the region)  = sqrt(f_sin^2 + f_cos^2)
  h        region width in log(1+z')
  D        abserr / (E h)
  sigma    theta_mod_2pi at 65 equispaced points across the region (ends included), numpy.unwrap,
           degree-4 least-squares polynomial in the local coordinate t in [-1, 1], RMS residual.
           (Descriptive only, not scored: the same with degree 8, and a flag when
           max|theta'| h / 64 > pi/2, where the 65-point unwrap could be ambiguous.)
  F        EPS k eta(z_response), EPS = numpy float64 eps, eta from
           docs/handover/a3-v2/findings/depth20_floor.py, as prompt 01 computed it.
  scored   a group with at least 10 regions at depth >= 18.
  ancestry for up to 20 depth-20 regions per group, chosen evenly through the group's depth-20
           regions in order of start (indices round(linspace(0, n-1, 20)), unique): replay the
           driver's bisection from the call's x_span (c = (a+b)/2.0, the child containing the
           region's midpoint), which reproduces the region's own endpoints exactly at depth 20
           (checked); the spans at depths 10..20 are the aligned spans. D on each span uses that
           span's E (at its midpoint) and h.
  control  the depth-10 span's theta_mod_2pi replaced by a degree-12 Chebyshev-basis polynomial
           least-squares fit (numpy Chebyshev.fit, domain = the span) to it, sampled at N
           equispaced points with N = max(4097, ceil(TV / (pi/8)) + 1), TV = total variation of
           the phase over the span from theta_deriv at 1025 points, and unwrapped. theta (raw),
           theta_deriv and f are unchanged, so the driver's Levin/Clenshaw-Curtis gate (which
           reads theta_deriv) routes every span exactly as with the real phase.
  slope    least-squares slope of ln D on ln h over the group's ancestry spans at depths 12..20,
           all sampled regions pooled (spans with D = 0 excluded and counted). Per row, M2/M3 use
           the median over the row's scored groups of this per-group slope.
  D/sigma  per group, the median over all its depth-20 regions of D / sigma; per row, the
           median of that over the row's scored groups.
  M3 ratio per group, median D at depth 20 over the ancestry sample, real / control; per row,
           the median over scored groups.
  M4       share of depth-20 regions of scored groups with phase_err >= abserr: per row, and
           pooled over the seven rows; the rule is scored on the pooled share, and the rows are
           reported beside it.
  M5       on rows whose group calls include a Gr sub-interval: x_j = a + j (b - a)/6, j = 1..5,
           over that sub-interval's x_span; ratio |theta'_{G+r}| / |theta'_{G-r}| from the
           groups' theta_deriv; c_s = sqrt(wPerturbations(z)), z = expm1(x_j); pass if
           |ratio / ((1 - c_s)/(1 + c_s)) - 1| <= 0.05 at all five.
  cancelling  prompt 01's rule: a group containing G is cancelling if G and a transfer factor
           carry the same sign and that factor's wavenumber equals k.
"""

import hashlib
import importlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial import Chebyshev

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "docs" / "handover" / "a3-v2" / "findings"))

from harness import load_record  # noqa: E402
import depth20_floor  # noqa: E402

LQ = importlib.import_module("AdaptiveLevin.levin_quadrature")
QSI = importlib.import_module("ComputeTargets.QuadSourceIntegral")

EPS = depth20_floor.EPS
ROW_NAMES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
F_TABLE = {
    "R1": 1.69e-6,
    "R2": 4.60e-6,
    "R3": 9.19e-5,
    "R4": 2.60e-6,
    "R5": 6.49e-6,
    "R6": 1.96e-6,
    "R7": 2.60e-6,
}
TOP_GROUP = {
    "R1": "G+r",
    "R2": "G+r",
    "R3": "G+r",
    "R4": "G-r",
    "R5": "G-r",
    "R6": "G",
    "R7": "G+q+r",
}
SAMPLE = 20
SIGMA_POINTS = 65


def out(s=""):
    print(s, flush=True)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_signs(label):
    return {
        m.group(2): (1 if m.group(1) in ("", "+") else -1)
        for m in re.finditer(r"([+-]?)([Gqr])", label)
    }


def cancelling(label, q_is_k, r_is_k):
    s = parse_signs(label)
    if "G" not in s:
        return None
    for f, eq in (("q", q_is_k), ("r", r_is_k)):
        if f in s and s[f] == s["G"] and eq:
            return True
    return False


# --- the driver's single-region routine ---------------------------------------------------------


def make_basis(group, theta_mod_2pi=None):
    theta = dict(group.levin_theta(include_deriv=QSI.LEVIN_USE_THETA_DERIV))
    if theta_mod_2pi is not None:
        theta["theta_mod_2pi"] = theta_mod_2pi
    return LQ._Basis_SinCos(theta)


def driver_discrepancy(group, basis, a, b, order, rtol):
    f = [group.f_sin, group.f_cos]
    lazy = LQ._LazyUUID()
    data = LQ._adaptive_levin_subregion(
        (a, b), f, basis, id_label=lazy, chebyshev_order=order, rtol=rtol
    )
    if data.get("is_direct", False):
        return "direct", float(data["abserr_direct"])
    c = (b + a) / 2.0
    dL = LQ._adaptive_levin_subregion(
        (a, c), f, basis, id_label=lazy, chebyshev_order=order, rtol=rtol
    )
    dR = LQ._adaptive_levin_subregion(
        (c, b), f, basis, id_label=lazy, chebyshev_order=order, rtol=rtol
    )
    return "Levin", float(np.fabs(data["value"] - (dL["value"] + dR["value"])))


# --- jitter ------------------------------------------------------------------------------------


def sigma_theta(group, a, b):
    x = np.linspace(a, b, SIGMA_POINTS)
    th = np.unwrap(np.array([group.theta_mod_2pi(float(xi)) for xi in x]))
    t = (2.0 * x - (a + b)) / (b - a)
    out_ = []
    for deg in (4, 8):
        coef = np.polynomial.polynomial.polyfit(t, th, deg)
        res = th - np.polynomial.polynomial.polyval(t, coef)
        out_.append(float(np.sqrt(np.mean(res * res))))
    return out_[0], out_[1]


def control_phase(group, a, b):
    xd = np.linspace(a, b, 1025)
    tv = float(np.mean(np.abs([group.theta_deriv(float(v)) for v in xd])) * (b - a))
    n = max(4097, int(math.ceil(tv / (np.pi / 8.0))) + 1)
    x = np.linspace(a, b, n)
    raw = np.array([group.theta_mod_2pi(float(v)) for v in x])
    th = np.unwrap(raw)
    fit = Chebyshev.fit(x, th, 12, domain=[a, b])
    res = th - fit(x)
    rms = float(np.sqrt(np.mean(res * res)))
    maxstep = float(np.max(np.abs(np.diff(th))))

    def theta_mod_2pi(v):
        return float(fit(v))

    return theta_mod_2pi, dict(n=n, tv=tv, fit_rms=rms, max_step=maxstep)


def replay_spans(x_span, region):
    a, b = x_span[0], x_span[1]
    m = 0.5 * (region.start + region.end)
    spans = {0: (a, b)}
    for d in range(1, region.depth + 1):
        c = (b + a) / 2.0
        if m < c:
            b = c
        else:
            a = c
        spans[d] = (a, b)
    return spans


def lsq_slope(h, D):
    h = np.asarray(h)
    D = np.asarray(D)
    keep = D > 0
    if keep.sum() < 2:
        return float("nan"), int((~keep).sum())
    s, _ = np.polyfit(np.log(h[keep]), np.log(D[keep]), 1)
    return float(s), int((~keep).sum())


def med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if len(v) else float("nan")


def fmt(x, p=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{p}g}"


def main():
    run_dir = Path(sys.argv[1])
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    sidecar = json.loads(
        (REPO / "var/datastores/handover-A3-v2-lambdacdm.manifest.json").read_text()
    )
    out("# levin-phase-floor prompt 03 step 7: instrument_03")
    out(f"# analyse.py sha256 {sha256(__file__)}")
    out(f"# harness.py sha256 {sha256(HERE / 'harness.py')} (load_record)")
    out(
        f"# depth20_floor.py sha256 {sha256(depth20_floor.__file__)} (eta(z), EPS imported from it)"
    )
    out(f"# git HEAD {head}")
    out(f"# store fingerprint (sidecar, read only) {sidecar['fingerprint']['digest']}")
    out(f"# run directory {run_dir}")
    for name in ROW_NAMES:
        out(f"# pickle row_{name}.pkl sha256 {sha256(run_dir / f'row_{name}.pkl')}")
    out("# definitions: see the module docstring of analyse.py; EPS = %r" % EPS)
    out()

    rows = {}
    for name in ROW_NAMES:
        rec = load_record(run_dir / f"row_{name}.pkl")
        assert rec["m0_holds"] and rec["match_ok"], name
        F = float(EPS * rec["k_inv_Mpc"] * depth20_floor.eta(rec["z_response"]))
        F_stored_eta = float(EPS * rec["k_inv_Mpc"] * rec["eta_response"])
        q_is_k = rec["q_store_id"] == rec["k_store_id"]
        r_is_k = rec["r_store_id"] == rec["k_store_id"]
        builds = {b["index"]: b for b in rec["build_calls"]}
        groups = []
        for call in rec["levin_calls"]:
            if call["group_label"] is None:
                continue
            group = [
                g
                for g in builds[call["build_index"]]["groups"]
                if g.label == call["group_label"]
            ][0]
            assert (
                call["theta_mod_2pi_id"]
                == builds[call["build_index"]]["group_ids"][group.label]
            )
            groups.append(
                dict(
                    call=call,
                    group=group,
                    regime=call["parsed"]["regime"],
                    label=group.label,
                    canc=cancelling(group.label, q_is_k, r_is_k),
                )
            )
        rows[name] = dict(
            rec=rec,
            F=F,
            F_stored_eta=F_stored_eta,
            groups=groups,
            q_is_k=q_is_k,
            r_is_k=r_is_k,
        )

    # ------------------------------------------------------------------------------------------
    out("== rows")
    out(
        f"   {'row':3s} {'shard':>5s} {'serial':>6s} {'k':>10s} {'q':>10s} {'r':>10s} {'z_resp':>9s} {'F (01 method)':>13s} {'F table':>8s} {'F (stored eta)':>14s}"
    )
    for name, R in rows.items():
        rec = R["rec"]
        out(
            f"   {name:3s} {rec['shard']:5d} {rec['serial']:6d} {rec['k_inv_Mpc']:10.4g} {rec['q_inv_Mpc']:10.4g} {rec['r_inv_Mpc']:10.4g} {rec['z_response']:9.4g} {R['F']:13.4g} {F_TABLE[name]:8.3g} {R['F_stored_eta']:14.4g}"
        )
    out()

    # ------------------------------------------------------------------------------------------
    # per-region quantities on every group
    for name, R in rows.items():
        for G in R["groups"]:
            regs = G["call"]["result"]["regions"]
            group = G["group"]
            G["n"] = len(regs)
            G["n18"] = sum(1 for r in regs if r.depth >= 18)
            G["scored"] = G["n18"] >= 10
            hist = {}
            for r in regs:
                hist.setdefault(r.depth, [0, 0])
                hist[r.depth][0 if r.type == "Levin" else 1] += 1
            G["hist"] = hist
            d20 = [r for r in regs if r.depth == 20]
            G["d20"] = d20
            sig, sig8, D, Dsig, flag, plim = [], [], [], [], 0, 0
            for r in d20:
                a, b = r.start, r.end
                mid = 0.5 * (a + b)
                h = b - a
                E = group.envelope(mid)
                s4, s8 = sigma_theta(group, a, b)
                dmax = max(abs(group.theta_deriv(float(v))) for v in (a, mid, b))
                if dmax * h / 64.0 > np.pi / 2.0:
                    flag += 1
                Dr = r.abserr / (E * h)
                sig.append(s4)
                sig8.append(s8)
                D.append(Dr)
                Dsig.append(Dr / s4 if s4 > 0 else float("inf"))
                if r.phase_err is not None and r.phase_err >= r.abserr:
                    plim += 1
            G["sigma"] = sig
            G["sigma8"] = sig8
            G["D20"] = D
            G["Dsig"] = Dsig
            G["unwrap_flag"] = flag
            G["n_plim_capable"] = plim
            G["n20_direct"] = sum(1 for r in d20 if r.type == "direct")
            G["median_sigma_over_F"] = med(sig) / R["F"] if sig else float("nan")
            G["median_Dsig"] = med(Dsig) if Dsig else float("nan")

    # ancestry on scored groups (and on any group with depth-20 regions, for the record)
    harness_mismatch = 0
    harness_checked = 0
    for name, R in rows.items():
        for G in R["groups"]:
            d20 = G["d20"]
            G["anc"] = []
            if len(d20) == 0:
                continue
            idx = sorted(
                set(
                    int(round(i))
                    for i in np.linspace(0, len(d20) - 1, min(SAMPLE, len(d20)))
                )
            )
            call = G["call"]
            group = G["group"]
            order, rtol = call["chebyshev_order"], call["rtol"]
            real_basis = make_basis(group)
            for i in idx:
                reg = d20[i]
                spans = replay_spans(call["x_span"], reg)
                assert spans[20] == (reg.start, reg.end), (
                    name,
                    G["label"],
                    spans[20],
                    reg.start,
                    reg.end,
                )
                a10, b10 = spans[10]
                ctl_theta, ctl_info = control_phase(group, a10, b10)
                ctl_basis = make_basis(group, theta_mod_2pi=ctl_theta)
                entry = dict(region=(reg.start, reg.end), control=ctl_info, spans=[])
                for d in range(10, 21):
                    a, b = spans[d]
                    mid = 0.5 * (a + b)
                    h = b - a
                    E = group.envelope(mid)
                    kr, dr = driver_discrepancy(group, real_basis, a, b, order, rtol)
                    kc, dc = driver_discrepancy(group, ctl_basis, a, b, order, rtol)
                    entry["spans"].append(
                        dict(
                            depth=d,
                            h=h,
                            E=E,
                            type=kr,
                            type_ctl=kc,
                            disc=dr,
                            disc_ctl=dc,
                            D=dr / (E * h),
                            Dc=dc / (E * h),
                        )
                    )
                    if d == 20:
                        harness_checked += 1
                        if not (dr == reg.abserr and kr == reg.type):
                            harness_mismatch += 1
                G["anc"].append(entry)
            hs = [
                s["h"] for e in G["anc"] for s in e["spans"] if 12 <= s["depth"] <= 20
            ]
            Ds = [
                s["D"] for e in G["anc"] for s in e["spans"] if 12 <= s["depth"] <= 20
            ]
            Dcs = [
                s["Dc"] for e in G["anc"] for s in e["spans"] if 12 <= s["depth"] <= 20
            ]
            G["slope_real"], G["zero_real"] = lsq_slope(hs, Ds)
            G["slope_ctl"], G["zero_ctl"] = lsq_slope(hs, Dcs)
            d20r = [e["spans"][-1]["D"] for e in G["anc"]]
            d20c = [e["spans"][-1]["Dc"] for e in G["anc"]]
            G["medD20_real"] = med(d20r)
            G["medD20_ctl"] = med(d20c)
            G["ratio"] = (
                (G["medD20_real"] / G["medD20_ctl"])
                if G["medD20_ctl"] > 0
                else float("inf")
            )
            G["types_changed"] = sum(
                1 for e in G["anc"] for s in e["spans"] if s["type"] != s["type_ctl"]
            )
            G["ctl_fit_rms_med"] = med([e["control"]["fit_rms"] for e in G["anc"]])
            G["ctl_n_med"] = med([e["control"]["n"] for e in G["anc"]])
            G["ctl_maxstep"] = max(e["control"]["max_step"] for e in G["anc"])

    # ------------------------------------------------------------------------------------------
    out(
        "== every group: regions, scored, depth histogram by type (depth: Levin/CC), jitter, D"
    )
    out(
        "   (sigma/F, D/sigma: medians over the group's depth-20 regions; 'unwrap?' counts depth-20 regions where max|theta'| h/64 > pi/2)"
    )
    for name, R in rows.items():
        out(f"-- {name} (top group {TOP_GROUP[name]}; F = {R['F']:.4g})")
        for G in R["groups"]:
            hist = " ".join(f"{d}:{v[0]}/{v[1]}" for d, v in sorted(G["hist"].items()))
            out(
                f"   {G['regime']:>4s} {G['label']:<6s} canc={str(G['canc']):5s} regions {G['n']:6d} depth>=18 {G['n18']:6d} scored {str(G['scored']):5s} depth20 {len(G['d20']):6d} (CC {G['n20_direct']})"
            )
            out(f"        hist {hist}")
            if G["d20"]:
                out(
                    f"        median sigma {fmt(med(G['sigma']))} rad (deg-8 {fmt(med(G['sigma8']))}); sigma/F {fmt(G['median_sigma_over_F'])}; "
                    f"sigma/F 10-90% {fmt(np.percentile(G['sigma'],10)/R['F'])}-{fmt(np.percentile(G['sigma'],90)/R['F'])}; unwrap? {G['unwrap_flag']}"
                )
                out(
                    f"        median D(depth 20) {fmt(med(G['D20']))}; median D/sigma {fmt(G['median_Dsig'])}; phase_err>=abserr {G['n_plim_capable']}/{len(G['d20'])}"
                )
                out(
                    f"        ancestry ({len(G['anc'])} regions): slope real {fmt(G['slope_real'])} (D=0 excluded: {G['zero_real']}), control {fmt(G['slope_ctl'])} (D=0 excluded: {G['zero_ctl']}); "
                    f"median D20 real {fmt(G['medD20_real'])}, control {fmt(G['medD20_ctl'])}, ratio {fmt(G['ratio'])}; span types changed by control {G['types_changed']}; "
                    f"control fit: median N {fmt(G['ctl_n_med'],6)}, median fit RMS {fmt(G['ctl_fit_rms_med'])} rad, max unwrapped step {fmt(G['ctl_maxstep'])} rad"
                )
    out()

    out(
        "== ancestry, median D over the sampled regions at each depth (real | control), scored groups"
    )
    for name, R in rows.items():
        for G in R["groups"]:
            if not G["scored"] or not G["anc"]:
                continue
            line = []
            for k in range(11):
                d = 10 + k
                Dr = med([e["spans"][k]["D"] for e in G["anc"]])
                Dc = med([e["spans"][k]["Dc"] for e in G["anc"]])
                ty = "".join(sorted({e["spans"][k]["type"][0] for e in G["anc"]}))
                line.append(f"{d}:{fmt(Dr,2)}|{fmt(Dc,2)}[{ty}]")
            out(f"   {name} {G['label']:<6s} " + " ".join(line))
    out(
        f"   harness check: recomputed real discrepancy == recorded region.abserr (and type) at {harness_checked - harness_mismatch} of {harness_checked} sampled depth-20 regions"
    )
    out()

    # ------------------------------------------------------------------------------------------
    out(
        "== M1: median sigma(depth 20)/F in [0.1, 10] in every scored group of every row"
    )
    m1_all = True
    for name, R in rows.items():
        for G in R["groups"]:
            if G["scored"]:
                ok = 0.1 <= G["median_sigma_over_F"] <= 10
                G["M1"] = ok
                m1_all &= ok
                out(
                    f"   {name} {G['label']:<6s} canc={str(G['canc']):5s} {fmt(G['median_sigma_over_F'],4):>8s} {'in' if ok else 'OUT'}"
                )
    out(f"   M1 {'HOLDS' if m1_all else 'FAILS'}")
    out()

    out(
        "== M2: per row, median over scored groups of the real-phase slope in [-0.5, 0.5] (part 1) and of median D/sigma at depth 20 in [0.01, 10] (part 2); holds if both on >= 6 of 7 rows"
    )
    m2_rows = 0
    m3_rows = 0
    rowres = {}
    for name, R in rows.items():
        sc = [G for G in R["groups"] if G["scored"]]
        s_med = med([G["slope_real"] for G in sc])
        ds_med = med([G["median_Dsig"] for G in sc])
        p1 = -0.5 <= s_med <= 0.5
        p2 = 0.01 <= ds_med <= 10
        sc_med = med([G["slope_ctl"] for G in sc])
        ratio_med = med([G["ratio"] for G in sc])
        m3 = sc_med >= 2 and ratio_med >= 100
        rowres[name] = (s_med, ds_med, p1, p2, sc_med, ratio_med, m3)
        m2_rows += p1 and p2
        m3_rows += m3
        for G in sc:
            G["M2p1"] = -0.5 <= G["slope_real"] <= 0.5
            G["M2p2"] = 0.01 <= G["median_Dsig"] <= 10
        out(
            f"   {name}: scored groups {[G['label'] for G in sc]}; median slope {fmt(s_med,4)} ({'in' if p1 else 'OUT'}); median D/sigma {fmt(ds_med,4)} ({'in' if p2 else 'OUT'}) -> {'both' if p1 and p2 else 'not both'}"
        )
    m2 = m2_rows >= 6
    out(f"   M2 holds on {m2_rows} of 7 rows: {'HOLDS' if m2 else 'FAILS'}")
    out()
    out(
        "== M3: control, per row median over scored groups of slope >= 2 and of median-D20 ratio real/control >= 100; holds if on >= 6 of 7 rows"
    )
    for name, (s_med, ds_med, p1, p2, sc_med, ratio_med, m3r) in rowres.items():
        out(
            f"   {name}: median control slope {fmt(sc_med,4)} ({'>=2' if sc_med >= 2 else '<2'}); median ratio {fmt(ratio_med,4)} ({'>=100' if ratio_med >= 100 else '<100'}) -> {'true' if m3r else 'false'}"
        )
    m3 = m3_rows >= 6
    out(f"   M3 true on {m3_rows} of 7 rows: {'HOLDS' if m3 else 'FAILS'}")
    out()

    out(
        "== M4: among depth-20 regions of scored groups, share with phase_err >= abserr below 0.1; also the share that is Clenshaw-Curtis"
    )
    tot_n = tot_p = tot_cc = 0
    for name, R in rows.items():
        sc = [G for G in R["groups"] if G["scored"]]
        n = sum(len(G["d20"]) for G in sc)
        p = sum(G["n_plim_capable"] for G in sc)
        cc = sum(G["n20_direct"] for G in sc)
        tot_n += n
        tot_p += p
        tot_cc += cc
        out(
            f"   {name}: {p}/{n} = {fmt(p/n if n else float('nan'),4)} phase_err>=abserr; CC {cc}/{n} = {fmt(cc/n if n else float('nan'),4)}"
        )
    share = tot_p / tot_n
    m4 = share < 0.1
    out(
        f"   pooled: {tot_p}/{tot_n} = {share:.4g}; CC share {tot_cc}/{tot_n} = {tot_cc/tot_n:.4g}"
    )
    out(f"   M4 {'HOLDS' if m4 else 'FAILS'}")
    out()

    out(
        "== M5: |theta'_{G+r}|/|theta'_{G-r}| against (1-c_s)/(1+c_s) at five abscissae of the Gr sub-interval, within 5% at all five; rows R1-R5 and R7"
    )
    m5_ok = True
    m5_applied = 0
    for name, R in rows.items():
        gp = [G for G in R["groups"] if G["regime"] == "Gr" and G["label"] == "G+r"]
        gm = [G for G in R["groups"] if G["regime"] == "Gr" and G["label"] == "G-r"]
        if not gp or not gm:
            out(
                f"   {name}: no Gr sub-interval (regimes {sorted({G['regime'] for G in R['groups']})}); the rule cannot be applied"
            )
            continue
        m5_applied += 1
        a, b = gp[0]["call"]["x_span"]
        mf = R["rec"]["model_functions"]
        row_ok = True
        parts = []
        for j in range(1, 6):
            x = a + j * (b - a) / 6.0
            ratio = abs(gp[0]["group"].theta_deriv(x)) / abs(
                gm[0]["group"].theta_deriv(x)
            )
            cs = math.sqrt(mf.wPerturbations(math.expm1(x)))
            pred = (1 - cs) / (1 + cs)
            dev = ratio / pred - 1
            row_ok &= abs(dev) <= 0.05
            parts.append(
                f"x={x:.4f} ratio={ratio:.6f} pred={pred:.6f} c_s^2={cs*cs:.6f} dev={dev:+.2e}"
            )
        m5_ok &= row_ok
        out(f"   {name}: {'within 5% at all five' if row_ok else 'NOT within 5%'}")
        for p in parts:
            out(f"      {p}")
    out(
        f"   M5 on the {m5_applied} rows it applies to: {'HOLDS' if m5_ok else 'FAILS'}"
    )
    out()

    out(
        "== H-anchor / H-cancel, group by group (scored groups): M1, M2 part 1 (own slope), M2 part 2 (own D/sigma)"
    )
    for name, R in rows.items():
        for G in R["groups"]:
            if G["scored"]:
                out(
                    f"   {name} {G['regime']:>4s} {G['label']:<6s} canc={str(G['canc']):5s} top={str(G['label']==TOP_GROUP[name]):5s} M1 {str(G['M1']):5s} slope {fmt(G['slope_real']):>8s} {str(G['M2p1']):5s} D/sigma {fmt(G['median_Dsig']):>8s} {str(G['M2p2']):5s}"
                )
    out()
    out(
        f"== summary: M1 {'holds' if m1_all else 'fails'}; M2 {'holds' if m2 else 'fails'} ({m2_rows}/7); M3 {'holds' if m3 else 'fails'} ({m3_rows}/7); M4 {'holds' if m4 else 'fails'} ({share:.4g}); M5 {'holds' if m5_ok else 'fails'} on {m5_applied} rows"
    )


if __name__ == "__main__":
    main()
