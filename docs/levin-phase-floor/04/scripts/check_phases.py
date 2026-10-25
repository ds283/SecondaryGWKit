"""
levin-phase-floor prompt 04, step 5: the phase checks, Ray-free and with no Levin call, into
out/phases_04.txt (and the numbers the predictions and the analysis reuse, state/phases_04.json).

Usage (cwd var/levin-phase-floor-04; PYTHONPATH the worktree; the main checkout's venv):
    OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree> <venv python> -B \
        scripts/check_phases.py > out/phases_04.txt

One worker process per row (spawned; each builds its own route-(a) pool, as prompt 03's harness
does). Every worker:

1. **Rebuilds the row's inputs** through route (a) (scripts/rebuild.py, scripts/harness.py), and
   forms the sub-intervals and factors exactly as ``evaluate_QuadSource_integral`` does, on
   Ray-style copies of the payload: ``TkSourceFunctions`` for q and r, ``build_partition``, and
   for each Levin sub-interval ``build_phase_groups`` with phase_group_Levin_integral's arguments.
   The sub-intervals' x_spans and group labels are checked against prompt 03's recorded Levin calls
   (its pickles), which hold the regions E2 reads.
2. **E1** (gate), for each oscillatory factor of each Levin sub-interval, at 2000 abscissae
   equispaced in log(1+z') over the sub-interval and clamped as production clamps:
   (a) |float(Delta^ext) - delta()| <= 2 ulp of delta();
   (b) |wrap(rho^X - rho^P)| <= 4 EPS Phi^P_i + 1e-15;
   (c) for G, |wrap(rho^A - rho^X)| <= 4 EPS Phi^A_G + 4 x 2^-49 (prompt section 8's amended
       bound, commit d170e6d; the first dispatch used + 1e-15);
   and, not a gate but reported, that Delta^ext and the exact reduction are exact (``fractions``).
3. **E2** (gate), in every scored group (prompt 03's rule: >= 10 regions at depth >= 18 in its
   pickled regions): on prompt 03's sampled depth-20 regions (03's selection, indices
   round(linspace(0, n-1, min(20, n))) of the group's depth-20 regions in the order the driver
   returned them), sigma_theta of the group's composed theta_mod_2pi under P, A and X with 03's
   estimator (65 equispaced points, numpy.unwrap, degree-4 fit in t in [-1, 1], RMS residual);
   the median is compared with EPS x the median over the same regions' midpoints of sum_i
   Phi^arm_i, and under X also with 1e-2 F.
   **Reproduction of 03's medians.** 03 printed its median sigma over *all* of a group's depth-20
   regions, not over the sample; so P's sigma_theta is also computed on all of them, and its
   median (and median/F) printed with 03's format beside 03's printed strings.
4. **Descriptive**: F^P_g, F^A_g, F^X_g of every group (1000 points); sigma_phi of every factor
   (production's phi samples, and phi rebuilt in mpmath from the stored theta_div_2pi,
   theta_mod_2pi with Delta^ext), as the RMS residual about a degree-4 fit in u over each window
   of 9 consecutive samples (median, 90th percentile and maximum over windows, over all samples
   and over the windows centred inside the factor's oscillatory Levin sub-intervals); and the
   modulus error max |theta_i| / TWO_PI x (2 pi - TWO_PI) over the factor's sub-intervals.

F is prompt 03's: EPS k eta(z_response), eta from docs/handover/a3-v2/findings/depth20_floor.py.
"""

import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from prov import MAIN, RUN03, header, sha256  # noqa: E402

FINDINGS = MAIN / "docs" / "handover" / "a3-v2" / "findings"
INSTRUMENT_03 = MAIN / "docs" / "levin-phase-floor" / "03" / "out" / "instrument_03.txt"
ROW_NAMES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
ROWS = {
    "R1": (2, 37527),
    "R2": (0, 1797),
    "R3": (1, 37670),
    "R4": (2, 37720),
    "R5": (0, 37966),
    "R6": (2, 34728),
    "R7": (2, 34820),
}
E1_POINTS = 2000
# prompt section 8 (commit d170e6d): the amended gates
E1C_ADD = 4.0 * 2.0**-49  # E1 (c)'s additive term
E2_ADD = 2.0**-48  # E2's additive term
SAMPLE = 20
SIGMA_POINTS = 65
PHI_WINDOW = 9
PHI_DEG = 4


def fmt(x, p=3):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "n/a"
    return f"{x:.{p}g}"


def med(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if len(v) else float("nan")


def sigma_theta(theta_mod_2pi, a, b):
    """Prompt 03's estimator (its analyse.py sigma_theta, degree 4)."""
    x = np.linspace(a, b, SIGMA_POINTS)
    th = np.unwrap(np.array([theta_mod_2pi(float(xi)) for xi in x]))
    t = (2.0 * x - (a + b)) / (b - a)
    coef = np.polynomial.polynomial.polyfit(t, th, 4)
    res = th - np.polynomial.polynomial.polyval(t, coef)
    return float(np.sqrt(np.mean(res * res)))


def window_rms(u, y):
    """RMS residual about a degree-4 fit over each window of 9 consecutive samples, in u."""
    out = []
    centres = []
    for i in range(0, len(u) - PHI_WINDOW + 1):
        uu = u[i : i + PHI_WINDOW]
        yy = y[i : i + PHI_WINDOW]
        t = (2.0 * uu - (uu[0] + uu[-1])) / (uu[-1] - uu[0])
        coef = np.polynomial.polynomial.polyfit(t, yy, PHI_DEG)
        res = yy - np.polynomial.polynomial.polyval(t, coef)
        out.append(float(np.sqrt(np.mean(res * res))))
        centres.append(float(uu[PHI_WINDOW // 2]))
    return np.array(out), np.array(centres)


def stats(v):
    v = np.asarray(v)
    if v.size == 0:
        return None
    return dict(
        n=int(v.size),
        median=float(np.median(v)),
        p90=float(np.percentile(v, 90)),
        max=float(np.max(v)),
    )


# ------------------------------------------------------------------------------------------------
# the worker


def work_row(name):
    t_start = time.perf_counter()
    from harness import close_pool, load_record, open_readonly_pool, standin_ray
    from rebuild import PRIMARY, build_context, rebuild_row

    sys.path.insert(0, str(FINDINGS))

    shard, serial = ROWS[name]
    res = dict(name=name)

    rec03 = load_record(RUN03 / f"row_{name}.pkl")
    assert rec03["m0_holds"] and rec03["match_ok"]
    calls03 = [c for c in rec03["levin_calls"] if c["group_label"] is not None]

    import ray

    # everything runs inside the route-(a) stand-in (ray.get/ray.put stood in; a proxy's .get()
    # needs it); the pool stays open, read-only, until the worker is done
    with standin_ray() as cluster:
        pool, _printed = open_readonly_pool(PRIMARY, cluster)
        try:
            ctx = build_context(pool, log=lambda s: None)
            rb = rebuild_row(ctx, pool, shard, serial, log=lambda s: None)
            _body(name, res, rec03, calls03, ctx, rb)
        finally:
            close_pool(pool, cluster)
    if ray.is_initialized():
        raise RuntimeError("Ray was initialised")
    res["elapsed"] = time.perf_counter() - t_start
    return res


def _body(name, res, rec03, calls03, ctx, rb):
    import importlib
    from math import log, ulp

    from harness import _ray_copy
    import phases as PH
    import depth20_floor

    QSI = importlib.import_module("ComputeTargets.QuadSourceIntegral")
    from ComputeTargets.TkSourceFunctions import TkSourceFunctions
    from ComputeTargets.phase_groups import build_phase_groups
    from LiouvilleGreen.constants import TWO_PI

    # the inputs as compute_QuadSource_integral receives them: Ray-style copies
    payload = _ray_copy(rb["compute_payload"])
    model = _ray_copy(ctx["bg_model"])
    k, q, r = rb["k"], rb["q"], rb["r"]
    GkPolicy = payload["GkPolicy"]
    source = payload["source"]
    Gk_f = GkPolicy.functions
    model_f = model.functions
    Tq_f = TkSourceFunctions(model, q.k, payload["Tq_numeric"], payload["Tq_WKB"])
    Tr_f = TkSourceFunctions(model, r.k, payload["Tr_numeric"], payload["Tr_WKB"])
    z_resp = rb["z_response"].z
    z_top_int = ctx["z_source_integral_max_z"].z
    partition = QSI.build_partition(GkPolicy, Tq_f, Tr_f, source, z_resp, z_top_int)

    F = float(
        depth20_floor.EPS * rec03["k_inv_Mpc"] * depth20_floor.eta(rec03["z_response"])
    )
    res["F"] = F
    res["k"] = rec03["k_inv_Mpc"]
    res["q"] = rec03["q_inv_Mpc"]
    res["r"] = rec03["r_inv_Mpc"]
    res["z_response"] = rec03["z_response"]
    q_is_k = rec03["q_store_id"] == rec03["k_store_id"]
    r_is_k = rec03["r_store_id"] == rec03["k_store_id"]

    # sub-intervals and groups, rebuilt; matched to 03's calls in order
    subs = []
    groups = []
    ci = 0
    for item in partition["subintervals"]:
        if item["method"] == "quad":
            continue
        regime = tuple(item["regime"])
        regime_label = "".join(n for n, f in zip(("G", "q", "r"), regime) if f)
        x_span = (log(1.0 + item["z_min"]), log(1.0 + item["z_max"]))
        built = build_phase_groups(
            regime,
            Gk=Gk_f if regime[0] else Gk_f.numeric_Gk,
            Tq=item["Tq"],
            Tr=item["Tr"],
            model_functions=model_f,
            w_background=model_f.wBackground,
        )
        factors = PH.factor_list(
            regime, Gk_f.phase if regime[0] else None, item["Tq"], item["Tr"]
        )
        subs.append(
            dict(
                regime=regime, regime_label=regime_label, x_span=x_span, factors=factors
            )
        )
        for g in built:
            c = calls03[ci]
            ci += 1
            ok = (
                c["parsed"]["regime"] == regime_label
                and c["group_label"] == g.label
                and tuple(c["x_span"]) == x_span
            )
            if not ok:
                raise RuntimeError(
                    f"{name}: rebuilt group {regime_label} {g.label} {x_span} does not match 03's call "
                    f"{c['parsed']['regime']} {c['group_label']} {c['x_span']}"
                )
            gfactors = [
                f
                for f in factors
                if {"G": g.signs[0], "q": g.signs[1], "r": g.signs[2]}[f[0]] != 0
            ]
            groups.append(
                dict(
                    regime=regime_label,
                    label=g.label,
                    group=g,
                    x_span=x_span,
                    factors=gfactors,
                    regions=c["result"]["regions"],
                    sub=len(subs) - 1,
                )
            )
    if ci != len(calls03):
        raise RuntimeError(
            f"{name}: {ci} rebuilt groups against {len(calls03)} 03 calls"
        )
    res["n_subintervals"] = len(subs)
    res["subintervals"] = [
        dict(
            regime=s["regime_label"],
            x_span=s["x_span"],
            factors=[f[0] for f in s["factors"]],
        )
        for s in subs
    ]

    # ---------------- E1 ----------------
    astate = PH._AState()
    e1 = []
    exact_fail_delta = 0
    exact_fail_rho = 0
    exact_checked = 0
    for si, s in enumerate(subs):
        xs = np.linspace(s["x_span"][0], s["x_span"][1], E1_POINTS)
        for fname, ph, clamp in s["factors"]:
            table = ph._leading.table
            kind = PH.kind_of(ph)
            worst = dict(a=0.0, b=0.0, c=0.0)
            worst_at = dict(a=None, b=None, c=None)
            fails = dict(a=0, b=0, c=0)
            c_fail_big = 0
            c_fail_lhs = []
            maxPhiP = 0.0
            for x0 in xs:
                x = clamp(float(x0))
                raw_x, u, phi = PH.pieces(ph, x)
                d_prod = ph._leading.delta(raw_x, ph._z_anchor)
                d_mp, ends, p_a, p_b = PH.delta_ext(table, raw_x, ph._z_anchor)
                # exactness (reported): Delta^ext and the reduction against fractions
                exact_checked += 1
                d_fr = PH.delta_fraction(table, raw_x, ph._z_anchor, p_a, p_b)
                th_fr = Fraction(ph._sign) * Fraction(ph._k) * d_fr + Fraction(phi)
                T_fr = Fraction(TWO_PI)
                r_fr = th_fr - T_fr * (-((-th_fr) // T_fr))  # exact, in (-TWO_PI, 0]
                if PH.mp_fraction(d_mp) != d_fr:
                    exact_fail_delta += 1
                if float(r_fr) != PH.rho_X(ph, x, True):
                    exact_fail_rho += 1
                # (a)
                lhs = abs(PH.mp_to_float(d_mp) - d_prod)
                rhs = 2.0 * ulp(d_prod)
                ra = lhs / rhs if rhs > 0 else (0.0 if lhs == 0 else math.inf)
                if lhs > rhs:
                    fails["a"] += 1
                if ra > worst["a"]:
                    worst["a"], worst_at["a"] = ra, (x, lhs, rhs)
                # (b)
                rho_P = PH.ORIG_THETA_MOD_2PI(ph, x, True)
                rho_X = PH.rho_X(ph, x, True)
                PhiP = abs(ph.raw_theta(x, x_is_log=True))
                maxPhiP = max(maxPhiP, PhiP)
                lhs = PH.wrap_abs_exact(rho_X, rho_P)
                rhs = 4.0 * PH.EPS * PhiP + 1e-15
                rb_ = lhs / rhs
                if lhs > rhs:
                    fails["b"] += 1
                if rb_ > worst["b"]:
                    worst["b"], worst_at["b"] = rb_, (x, lhs, rhs)
                # (c)
                if kind == "G":
                    rho_A = PH.rho_A_G(ph, x, astate, True)
                    PhiA = PH.Phi("A", ph, x, astate)
                    lhs = PH.wrap_abs_exact(rho_A, rho_X)
                    rhs = 4.0 * PH.EPS * PhiA + E1C_ADD
                    rc = lhs / rhs
                    if lhs > rhs:
                        fails["c"] += 1
                        if abs(rho_A) >= 8.0:
                            c_fail_big += 1
                        c_fail_lhs.append(lhs)
                    if rc > worst["c"]:
                        worst["c"], worst_at["c"] = rc, (x, lhs, rhs)
            entry = dict(
                sub=si,
                regime=s["regime_label"],
                factor=fname,
                kind=kind,
                n=E1_POINTS,
                fails=fails,
                worst=worst,
                worst_at=worst_at,
                maxPhiP=maxPhiP,
                c_fail_rhoA_ge_8=c_fail_big,
                c_fail_max_lhs=max(c_fail_lhs) if c_fail_lhs else None,
            )
            if kind == "G":
                z_top, c_G = astate.get(ph)
                entry["z_top"] = z_top
                entry["c_G"] = c_G
                entry["max_x"] = ph.max_x
                entry["z_anchor"] = ph._z_anchor
            else:
                entry["z_anchor"] = ph._z_anchor
            e1.append(entry)
    res["e1"] = e1
    res["exact_checked"] = exact_checked
    res["exact_fail_delta"] = exact_fail_delta
    res["exact_fail_rho"] = exact_fail_rho

    # ---------------- E2 and the groups ----------------
    gout = []
    for G in groups:
        regs = G["regions"]
        n18 = sum(1 for rg in regs if rg.depth >= 18)
        d20 = [rg for rg in regs if rg.depth == 20]
        scored = n18 >= 10
        hist = {}
        for rg in regs:
            hist.setdefault(rg.depth, [0, 0])
            hist[rg.depth][0 if rg.type == "Levin" else 1] += 1
        sig = G["group"].theta_mod_2pi
        entry = dict(
            regime=G["regime"],
            label=G["label"],
            sub=G["sub"],
            x_span=G["x_span"],
            factors=[f[0] for f in G["factors"]],
            n=len(regs),
            n18=n18,
            n20=len(d20),
            scored=scored,
            canc=_cancelling(G["label"], q_is_k, r_is_k),
        )
        # reproduction: P sigma over every depth-20 region (03's population)
        if d20:
            with PH.install("P") as disp:
                allsig = [sigma_theta(sig, rg.start, rg.end) for rg in d20]
            entry["P_all_median_sigma"] = med(allsig)
            entry["P_all_counts"] = dict(disp.counts)
        # F^arm (every group)
        for arm in ("P", "A", "X"):
            entry[f"F_{arm}"] = PH.F_group(arm, G["factors"], G["x_span"], astate)
        # E2 on 03's sample
        if d20:
            idx = sorted(
                set(
                    int(round(i))
                    for i in np.linspace(0, len(d20) - 1, min(SAMPLE, len(d20)))
                )
            )
            sample = [d20[i] for i in idx]
            entry["sample_n"] = len(sample)
            for arm in ("P", "A", "X"):
                with PH.install(arm) as disp:
                    s_arm = [sigma_theta(sig, rg.start, rg.end) for rg in sample]
                mids = [0.5 * (rg.start + rg.end) for rg in sample]
                phis = [PH.sum_Phi(arm, G["factors"], m, astate) for m in mids]
                entry[f"E2_{arm}_median_sigma"] = med(s_arm)
                entry[f"E2_{arm}_median_epsPhi"] = PH.EPS * med(phis)
                entry[f"E2_{arm}_counts"] = dict(disp.counts)
                with PH.install(arm):
                    psi = [abs(sig(m)) for m in mids]
                entry[f"E2_{arm}_median_abs_psi_mod"] = med(psi)
            # the abscissa's own rounding, z = expm1(x) to half an ulp, carried into the phase
            # as EPS |d theta / d u|: a term of the amended E2 bound (prompt section 8)
            entry["E2_eps_dthdu"] = PH.EPS * med(
                [
                    sum(
                        abs(
                            ph.theta_deriv(clamp(m), x_is_log=True, log_derivative=True)
                        )
                        for _n, ph, clamp in G["factors"]
                    )
                    for m in mids
                ]
            )
        gout.append(entry)
    res["groups"] = gout

    # ---------------- sigma_phi and the modulus, per factor ----------------
    from ComputeTargets.primitive_phase import build_phi_samples

    fac_out = []
    Gk_source = GkPolicy._source_proxy.get()
    factor_objs = []
    if Gk_f.phase is not None and any(s["regime"][0] for s in subs):
        factor_objs.append(("G", Gk_f.phase, k))
    for fname, Tf, mode, wkb in (
        ("q", Tq_f, q, payload["Tq_WKB"]),
        ("r", Tr_f, r, payload["Tr_WKB"]),
    ):
        if any(s["regime"][1 if fname == "q" else 2] for s in subs):
            factor_objs.append((fname, Tf.phase, wkb))
    for fname, ph, src in factor_objs:
        zs = ph.z_samples
        phi_prod = ph.phi_samples
        u = np.log1p(zs)
        # the stored samples, by exact z
        if fname == "G":
            vals = {v.z_source.z: v for v in Gk_source.values if v.has_WKB}
            div_mod = [
                (vals[float(z)].WKB.theta_div_2pi, vals[float(z)].WKB.theta_mod_2pi)
                for z in zs
            ]
        else:
            vals = {v.z.z: v for v in src.values}
            div_mod = [
                (vals[float(z)].theta_div_2pi, vals[float(z)].theta_mod_2pi) for z in zs
            ]
        theta_dbl = [d * TWO_PI + m for d, m in div_mod]
        phi_check = build_phi_samples(
            ph._k, ph._leading, ph._z_anchor, zs, theta_dbl, sign=ph._sign
        )
        same_as_production = bool(np.array_equal(phi_check, phi_prod))
        phi_mp = []
        for z, (d, m) in zip(zs, div_mod):
            th = PH.MP.mpf(int(d)) * PH.TWO_PI_MP + PH.MP.mpf(m)
            dd, _e, _pa, _pb = PH.delta_ext(ph._leading.table, float(z), ph._z_anchor)
            phi_mp.append(
                PH.mp_to_float(th - PH.MP.mpf(ph._sign) * PH.MP.mpf(ph._k) * dd)
            )
        phi_mp = np.array(phi_mp)
        w_prod, centres = window_rms(u, phi_prod)
        w_mp, _ = window_rms(u, phi_mp)
        # the factor's oscillatory sub-intervals (clamped onto the factor's range)
        spans = [s["x_span"] for s in subs if any(f[0] == fname for f in s["factors"])]
        inside = np.array(
            [any(a <= c <= b for a, b in spans) for c in centres], dtype=bool
        )
        # modulus: max |theta| over the factor's sub-intervals (1000 points each, clamped)
        maxtheta = 0.0
        for s in subs:
            for fn, ph2, clamp in s["factors"]:
                if fn != fname:
                    continue
                for x0 in np.linspace(s["x_span"][0], s["x_span"][1], PH.F_SAMPLES):
                    maxtheta = max(
                        maxtheta, abs(ph2.raw_theta(clamp(float(x0)), x_is_log=True))
                    )
        two_pi_err = float(2 * PH.MP.pi - PH.TWO_PI_MP)
        fac_out.append(
            dict(
                factor=fname,
                kind=PH.kind_of(ph),
                n_samples=int(len(zs)),
                phi_prod_equals_build_phi_samples=same_as_production,
                max_abs_phi=float(np.max(np.abs(phi_prod))),
                max_phi_prod_minus_mp=float(np.max(np.abs(phi_prod - phi_mp))),
                sigma_phi_prod_all=stats(w_prod),
                sigma_phi_mp_all=stats(w_mp),
                sigma_phi_prod_levin=stats(w_prod[inside]),
                sigma_phi_mp_levin=stats(w_mp[inside]),
                max_abs_theta=maxtheta,
                modulus=maxtheta / PH.TWO_PI * two_pi_err,
            )
        )
    res["factors"] = fac_out


def _cancelling(label, q_is_k, r_is_k):
    s = {
        m.group(2): (1 if m.group(1) in ("", "+") else -1)
        for m in re.finditer(r"([+-]?)([Gqr])", label)
    }
    if "G" not in s:
        return None
    for f, eq in (("q", q_is_k), ("r", r_is_k)):
        if f in s and s[f] == s["G"] and eq:
            return True
    return False


# ------------------------------------------------------------------------------------------------
# 03's printed numbers


def parse_03():
    """Per row, the group lines of instrument_03.txt in order: regime, label, and the printed
    'median sigma' and 'sigma/F' strings (3 digits), and M1's 4-digit sigma/F."""
    text = INSTRUMENT_03.read_text().splitlines()
    rows = {}
    cur = None
    last = None
    m1 = {}
    in_m1 = False
    for line in text:
        m = re.match(r"^-- (R\d) ", line)
        if m:
            cur = m.group(1)
            rows[cur] = []
            continue
        if line.startswith("== M1"):
            in_m1 = True
            cur = None
            continue
        if in_m1:
            mm = re.match(r"^\s+(R\d) (\S+)\s+canc=\S+\s+(\S+)\s+(in|OUT)$", line)
            if mm:
                m1.setdefault(mm.group(1), []).append((mm.group(2), mm.group(3)))
                continue
            if line.startswith("   M1 "):
                in_m1 = False
            continue
        if cur is None:
            continue
        mg = re.match(r"^\s+(\S+) (\S+)\s+canc=", line)
        if mg and " regions " in line:
            last = dict(regime=mg.group(1), label=mg.group(2), sigma=None, sigF=None)
            rows[cur].append(last)
            continue
        ms = re.match(r"^\s+median sigma (\S+) rad .*; sigma/F (\S+);", line)
        if ms and last is not None:
            last["sigma"] = ms.group(1)
            last["sigF"] = ms.group(2)
    return rows, m1


# ------------------------------------------------------------------------------------------------


def run():
    sys.path.insert(0, str(FINDINGS))
    import depth20_floor

    pickles = [RUN03 / f"row_{n}.pkl" for n in ROW_NAMES]
    for line in header(
        Path(__file__).resolve(),
        "levin-phase-floor prompt 04 step 5, resumed under prompt section 8 (d170e6d; amended E1 (c) and E2): phases_04 (E1, E2 gates; descriptive F^arm, sigma_phi, modulus)",
        inputs=[
            HERE / "phases.py",
            HERE / "harness.py",
            HERE / "rebuild.py",
            HERE / "prov.py",
            Path(depth20_floor.__file__),
            INSTRUMENT_03,
            *pickles,
        ],
    ):
        print(line)
    sidecar = json.loads(
        (MAIN / "var/datastores/handover-A3-v2-lambdacdm.manifest.json").read_text()
    )
    print(
        f"# store fingerprint (sidecar, read only) {sidecar['fingerprint']['digest']}"
    )
    import phases as PH

    print(f"# EPS = {PH.EPS!r}; mpmath prec {PH.MP.prec} bits; TWO_PI = {PH.TWO_PI!r}")
    print(
        "# definitions: module docstrings of check_phases.py and phases.py; Ray never initialised; no Levin call"
    )
    print(flush=True)

    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=len(ROW_NAMES), mp_context=ctx) as ex:
        futs = {n: ex.submit(work_row, n) for n in ROW_NAMES}
        results = {n: futs[n].result() for n in ROW_NAMES}
    wall = time.perf_counter() - t0

    printed03, m1_03 = parse_03()
    e1_ok = True
    e2_ok = True
    repro_ok = True
    out = {}

    print("== rows")
    for n in ROW_NAMES:
        R = results[n]
        print(
            f"   {n} k={R['k']:.4g} q={R['q']:.4g} r={R['r']:.4g} z_response={R['z_response']:.4g} F={R['F']:.4g}; "
            f"Levin sub-intervals {[(s['regime'], tuple(round(v, 6) for v in s['x_span'])) for s in R['subintervals']]}; worker {R['elapsed']:.0f} s"
        )
    print()

    print(
        "== E1: the arms compute the same phase (2000 abscissae per factor per Levin sub-interval)"
    )
    print(
        "   (a) |float(Delta^ext) - delta()| <= 2 ulp(delta());  (b) |wrap(rho^X - rho^P)| <= 4 EPS Phi^P + 1e-15;  (c) G: |wrap(rho^A - rho^X)| <= 4 EPS Phi^A + 4 x 2^-49 (amended, prompt section 8)"
    )
    print(
        "   worst = largest lhs/rhs over the abscissae (<= 1 passes); fails = abscissae over the bound; wrap is evaluated exactly (fractions) on the two doubles"
    )
    for n in ROW_NAMES:
        R = results[n]
        for e in R["e1"]:
            f = e["fails"]
            ok = f["a"] == 0 and f["b"] == 0 and f["c"] == 0
            e1_ok &= ok
            extra = (
                f" z_top={e['z_top']:.6g} (max_x={e['max_x']:.6g}) c_G={e['c_G']:.17g} anchor={e['z_anchor']:.6g}"
                if e["kind"] == "G"
                else f" anchor={e['z_anchor']:.6g}"
            )
            print(
                f"   {n} sub{e['sub']} {e['regime']:>3s} {e['factor']} ({e['kind']}): "
                f"(a) worst {e['worst']['a']:.3f} fails {f['a']}; (b) worst {e['worst']['b']:.3f} fails {f['b']}; "
                + (
                    f"(c) worst {e['worst']['c']:.3f} fails {f['c']}; "
                    if e["kind"] == "G"
                    else "(c) n/a; "
                )
                + f"max Phi^P {e['maxPhiP']:.4g};{extra} -> {'ok' if ok else 'FAILS'}"
            )
            if f["c"]:
                print(
                    f"      (c): {f['c']} abscissae over the bound, largest lhs {e['c_fail_max_lhs']:.4g}; of them {e['c_fail_rhoA_ge_8']} have |rho^A| >= 8 (a double there has an ulp of 1.78e-15)"
                )
            for key in ("a", "b", "c"):
                if f[key]:
                    print(
                        f"      worst ({key}) at x={e['worst_at'][key][0]!r}: lhs={e['worst_at'][key][1]:.4g} rhs={e['worst_at'][key][2]:.4g}"
                    )
        print(
            f"   {n} exactness (reported, not a gate), against fractions: Delta^ext exact at "
            f"{R['exact_checked'] - R['exact_fail_delta']} of {R['exact_checked']} abscissae; rho^X equal to the "
            f"correctly rounded exact remainder at {R['exact_checked'] - R['exact_fail_rho']} of {R['exact_checked']}"
        )
    print(f"   E1 {'HOLDS' if e1_ok else 'FAILS'}")
    print()

    print(
        "== reproduction of 03's printed medians under P (all depth-20 regions of each group; 03's format)"
    )
    for n in ROW_NAMES:
        R = results[n]
        p03 = printed03[n]
        if len(p03) != len(R["groups"]):
            repro_ok = False
            print(
                f"   {n}: {len(R['groups'])} groups here, {len(p03)} printed by 03 -- MISMATCH"
            )
            continue
        m1_rows = dict(m1_03.get(n, []))
        for G, P3 in zip(R["groups"], p03):
            if (G["regime"], G["label"]) != (P3["regime"], P3["label"]):
                repro_ok = False
                print(
                    f"   {n}: group order differs: {G['regime']} {G['label']} vs {P3['regime']} {P3['label']}"
                )
                continue
            if G["n20"] == 0:
                continue
            s = fmt(G["P_all_median_sigma"])
            sf = fmt(G["P_all_median_sigma"] / R["F"])
            ok = s == P3["sigma"] and sf == P3["sigF"]
            line = f"   {n} {G['regime']:>4s} {G['label']:<6s} median sigma {s} (03: {P3['sigma']}); sigma/F {sf} (03: {P3['sigF']})"
            if G["scored"]:
                s4 = fmt(G["P_all_median_sigma"] / R["F"], 4)
                ok4 = s4 == m1_rows.get(G["label"])
                ok &= ok4
                line += f"; M1's sigma/F {s4} (03: {m1_rows.get(G['label'])})"
            repro_ok &= ok
            print(line + f" -> {'same' if ok else 'DIFFERENT'}")
    print(f"   reproduction {'HOLDS' if repro_ok else 'FAILS'}")
    print()

    print(
        "== E2 (amended, prompt section 8, d170e6d): scored groups; median sigma_theta over 03's sampled depth-20 regions vs bound = EPS x median sum_i Phi^arm_i + EPS x median sum_i |d theta_i/du| + 2^-48 (medians at the regions' midpoints); under X also <= 1e-2 F"
    )
    for n in ROW_NAMES:
        R = results[n]
        for G in R["groups"]:
            if not G["scored"]:
                continue
            parts = []
            gok = True
            for arm in ("P", "A", "X"):
                s = G[f"E2_{arm}_median_sigma"]
                b = G[f"E2_{arm}_median_epsPhi"] + G["E2_eps_dthdu"] + E2_ADD
                ok = s <= b
                if arm == "X":
                    okF = s <= 1e-2 * R["F"]
                    ok &= okF
                gok &= ok
                parts.append(
                    f"{arm}: sigma {fmt(s)} vs bound {fmt(b)} (EPS.Phi {fmt(G[f'E2_{arm}_median_epsPhi'])}; ratio {fmt(s / b if b else float('nan'))})"
                    + (f", vs 1e-2 F {fmt(1e-2 * R['F'])}" if arm == "X" else "")
                )
            e2_ok &= gok
            print(
                f"   {n} {G['regime']:>4s} {G['label']:<6s} n={G['sample_n']:2d}  "
                + "; ".join(parts)
                + f" -> {'ok' if gok else 'FAILS'}"
            )
            print(
                f"      EPS x median sum|d theta_i/du| {fmt(G['E2_eps_dthdu'])}; 2^-48 {fmt(E2_ADD)}; descriptive: median |composed theta_mod_2pi| P {fmt(G['E2_P_median_abs_psi_mod'])} A {fmt(G['E2_A_median_abs_psi_mod'])} X {fmt(G['E2_X_median_abs_psi_mod'])}"
            )
    print(f"   E2 {'HOLDS' if e2_ok else 'FAILS'}")
    print()

    print(
        "== every group: scored status (03's rule), regions, and F^P_g, F^A_g, F^X_g beside 03's F (descriptive)"
    )
    a1 = []
    for n in ROW_NAMES:
        R = results[n]
        print(f"-- {n} (F = {R['F']:.4g})")
        for G in R["groups"]:
            print(
                f"   {G['regime']:>4s} {G['label']:<6s} canc={str(G['canc']):5s} factors {''.join(G['factors']):3s} regions {G['n']:6d} depth>=18 {G['n18']:6d} scored {str(G['scored']):5s} "
                f"F^P {G['F_P']:.3e} F^A {G['F_A']:.3e} F^X {G['F_X']:.3e}  (F^P/F {G['F_P'] / R['F']:.3g}; F^A/F {G['F_A'] / R['F']:.3g})"
            )
            if G["scored"] and G["F_A"] <= 0.1 * 1e-7:
                a1.append((n, G["regime"], G["label"], G["F_A"]))
    print()
    print("== A1's covered groups: scored groups with F^A_g <= 0.1 x 1e-7")
    for n, rg, lb, fa in a1:
        print(f"   {n} {rg} {lb}  F^A = {fa:.3e}")
    if not a1:
        print("   none (A1 would be undecided)")
    print(f"   {len(a1)} groups")
    print()

    print(
        "== sigma_phi: RMS residual of phi about a degree-4 fit in u over each window of 9 consecutive samples (descriptive)"
    )
    print(
        "   'prod' = production's phi samples (build_phi_samples, double); 'mp' = rebuilt in mpmath from the stored div/mod and Delta^ext"
    )
    print(
        "   'all' = every window; 'levin' = windows centred inside the factor's oscillatory Levin sub-intervals; each: median / 90% / max over windows"
    )
    for n in ROW_NAMES:
        R = results[n]
        for f in R["factors"]:

            def s3(st):
                if st is None:
                    return "n/a (no window)"
                return f"{fmt(st['median'])} / {fmt(st['p90'])} / {fmt(st['max'])} (n={st['n']})"

            print(
                f"   {n} {f['factor']} ({f['kind']}): {f['n_samples']} samples, max|phi| {f['max_abs_phi']:.3g}, phi_prod==build_phi_samples {f['phi_prod_equals_build_phi_samples']}, "
                f"max|phi_prod - phi_mp| {f['max_phi_prod_minus_mp']:.3g}"
            )
            print(
                f"      prod all {s3(f['sigma_phi_prod_all'])}; levin {s3(f['sigma_phi_prod_levin'])}"
            )
            print(
                f"      mp   all {s3(f['sigma_phi_mp_all'])}; levin {s3(f['sigma_phi_mp_levin'])}"
            )
            print(
                f"      F = {R['F']:.3g}; prod levin median/F {fmt((f['sigma_phi_prod_levin'] or {}).get('median', float('nan')) / R['F'])}"
            )
    print()

    print(
        "== the modulus: max |theta_i| / TWO_PI x (2 pi - TWO_PI) over the factor's sub-intervals, against F (descriptive)"
    )
    for n in ROW_NAMES:
        R = results[n]
        for f in R["factors"]:
            print(
                f"   {n} {f['factor']} ({f['kind']}): max|theta| {f['max_abs_theta']:.4g} rad; modulus error {f['modulus']:.3g} rad; /F {f['modulus'] / R['F']:.3g}"
            )
    print()
    print(
        f"== summary: E1 {'holds' if e1_ok else 'FAILS'}; E2 {'holds' if e2_ok else 'FAILS'}; "
        f"reproduction of 03's medians {'holds' if repro_ok else 'FAILS'}; A1 covers {len(a1)} groups; wall {wall:.0f} s"
    )

    # the numbers later steps reuse
    for n in ROW_NAMES:
        R = results[n]
        out[n] = dict(
            F=R["F"],
            groups=[
                {
                    kk: vv
                    for kk, vv in G.items()
                    if kk
                    in (
                        "regime",
                        "label",
                        "sub",
                        "x_span",
                        "factors",
                        "n",
                        "n18",
                        "n20",
                        "scored",
                        "canc",
                        "F_P",
                        "F_A",
                        "F_X",
                    )
                }
                for G in R["groups"]
            ],
        )
    state = HERE.parent / "state" / "phases_04.json"
    state.write_text(
        json.dumps(
            dict(
                e1=e1_ok,
                e2=e2_ok,
                reproduction=repro_ok,
                a1_groups=a1,
                rows=out,
                script_sha256=sha256(Path(__file__).resolve()),
            ),
            indent=1,
        )
    )
    print(f"# wrote {state} sha256 {sha256(state)}")


if __name__ == "__main__":
    run()
