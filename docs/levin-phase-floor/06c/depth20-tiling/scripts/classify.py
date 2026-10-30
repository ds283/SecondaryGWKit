"""depth20-tiling step 3: classify each sampled call's depth-20 chase.

For each sampled call whose capture reproduced the census (state/captured.json, and its replay in
state/runs/replay/), on every level of its deepest line (state/cheap_pass.json; the spans as the
cheap pass reconstructed them), with the call's captured f and theta and its chebyshev_order and
rtol:

1. **Re-solve P = [a, b], L = [a, c], R = [c, b]** four ways:
   - double, computed ends: the module's own _adaptive_levin_subregion as it stands. Its residual
     (Levin: |V_P - (V_L + V_R)|; direct: P's nested pair) and floor must equal the census line's
     bit for bit;
   - double, pinned ends: the same with chebyshev_matrices pinned (x[0] := b, x[-1] := a; every
     other node and D the original's; dcommon.pinned);
   - mp (50 digits), computed ends and pinned ends: dcommon.solve_mp -- the same scheme with exact
     linear algebra and exact D, on the double data (f, theta' and theta_mod_2pi evaluated in
     double by the call's own callables at the grid's double nodes).
2. **Offset terms**: dcommon.offset_term for P, L and R; the predicted change of the signed residual
   s = V_P - V_L - V_R from computed to pinned ends is off_P - off_L - off_R; the measured change is
   s_mp(computed) - s_mp(pinned).
3. **Stationary point** (b): theta' (the dict's theta_deriv) on 200,001 equally spaced points of
   the call's x_span (its maximum |theta'| there is the span's), and on 20,001 points of the line's
   depth-16 region plus every collocation node of the line's Levin solves at depths 16-20: a sign
   change, or min |theta'| <= 1e-6 max_span |theta'|. Sign changes are bisected to adjacent doubles;
   x, z = e^x - 1 (group calls), theta' at the two doubles and theta'' (central difference, h = 1e-6)
   are reported. The whole span's sign changes are reported too (descriptive).

The rules, fixed before any re-solve was run (log.md):
- deep levels: depths 16-20 (the line's last five);
- a deep level is *flagged* if its parent is a Levin solve with residual > 10 floor;
- a flagged level is *attributed to tiling* if (i) the computed end nodes of P, L, R do not tile;
  (ii) with the ends pinned (double), P's residual <= 10 x P's pinned floor; and (iii)
  |dS_measured - dS_predicted| <= 0.1 |dS_predicted| + floor;
- the call is (a) if it has a flagged deep level and every flagged deep level is attributed;
  (b) if the stationary-point test holds on the depth-16 region; if both, it is classified by the
  deepest flagged level: (a) if that level is attributed, else (b); (c) otherwise, naming, over the
  flagged deep levels not attributed, the largest of: the pinned residual in mp ("resolution of the
  scheme"), |s_double - s_mp| at the computed ends ("solve rounding"), |dS_measured| ("end-node
  offset"); or, with no flagged deep level, the leaf's branch and the line's trust ratio r.

Writes state/classify/NN.json per call.

Usage: PYTHONPATH=<tree>:<scripts> python -B classify.py CAPTURED_JSON [--workers 6] [--only 0,1]
"""

import argparse
import concurrent.futures as cf
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dprov import STATE, header, hx  # noqa: E402

DEEP = range(16, 21)


def fx(m):
    import mpmath as mp

    return float(m) if m is not None else None


def one(args):
    cap_p, no = args
    import mpmath as mp
    import numpy as np

    import AdaptiveLevin.levin_quadrature as LQ
    import dcommon as DC

    t0 = time.perf_counter()
    e = [x for x in json.loads(Path(cap_p).read_text()) if x["sample_no"] == no][0]
    rec = DC.load_capture(e)
    f, theta, kw = rec["f"], rec["theta"], dict(rec["kw"])
    order = kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
    rtol = kw.get("rtol", LQ.DEFAULT_LEVIN_RELTOL)
    basis = LQ._Basis_SinCos(theta)
    line = [ln for ln in json.loads((STATE / "cheap_pass.json").read_text()) if (ln["unit"], ln["index"]) == (e["unit"], e["index"])][0]
    out = {"sample_no": no, "unit": e["unit"], "index": e["index"], "notify_label": rec["notify_label"], "rtol": e["rtol"],
           "block": e["block"], "order": order, "levels": []}
    vc = {}
    for lv in line["levels"]:
        a, b, c = (float.fromhex(lv[k]) for k in ("a", "b", "c"))
        spans = {"P": (a, b), "L": (a, c), "R": (c, b)}
        dd = {k: DC.solve_double(s, f, basis, order, rtol, False, vc) for k, s in spans.items()}
        dp = {k: DC.solve_double(s, f, basis, order, rtol, True) for k, s in spans.items()}
        P_direct = bool(dd["P"].get("is_direct", False))
        if P_direct:
            res_c = dd["P"]["abserr_direct"]
            res_p = dp["P"]["abserr_direct"]
        else:
            res_c = abs(dd["P"]["value"] - (dd["L"]["value"] + dd["R"]["value"]))
            res_p = abs(dp["P"]["value"] - (dp["L"]["value"] + dp["R"]["value"]))
        floor_c = dd["P"].get("phase_err", 0.0) or 0.0
        floor_p = dp["P"].get("phase_err", 0.0) or 0.0
        repro = (hx(res_c) == lv["resid"]) and (hx(floor_c) == lv["floor"])
        L = {"depth": lv["depth"], "a": lv["a"], "b": lv["b"], "c": lv["c"], "types": lv["types"], "tile": lv["tile"],
             "parent_exact": lv["parent_exact"], "ulps": lv["ulps"],
             "census_resid": lv["resid"], "census_floor": lv["floor"], "reproduces_census_resid_and_floor": repro,
             "resid_computed_double": hx(res_c), "floor_computed": hx(floor_c),
             "resid_pinned_double": hx(res_p), "floor_pinned": hx(floor_p),
             "values_double_computed": {k: hx(d["value"]) for k, d in dd.items()},
             "values_double_pinned": {k: hx(d["value"]) for k, d in dp.items()},
             "branch_pinned": {k: ("direct" if d.get("is_direct", False) else "Levin") for k, d in dp.items()},
             "resid_over_floor": (res_c / floor_c) if floor_c > 0 else None,
             "resid_pinned_over_floor_pinned": (res_p / floor_p) if floor_p > 0 else None}
        if lv["depth"] in DEEP and not P_direct:
            with mp.workdps(DC.MP_DPS):
                vm_c = {k: DC.solve_mp(s, f, basis, order, rtol, False, dd[k])[0] for k, s in spans.items()}
                vm_p = {k: DC.solve_mp(s, f, basis, order, rtol, True, dp[k])[0] for k, s in spans.items()}
                off = {k: DC.offset_term(s, f, basis, order, dd[k]) for k, s in spans.items()}
                s_dbl_c = mp.mpf(dd["P"]["value"]) - mp.mpf(dd["L"]["value"]) - mp.mpf(dd["R"]["value"])
                s_dbl_p = mp.mpf(dp["P"]["value"]) - mp.mpf(dp["L"]["value"]) - mp.mpf(dp["R"]["value"])
                s_mp_c = vm_c["P"] - vm_c["L"] - vm_c["R"]
                s_mp_p = vm_p["P"] - vm_p["L"] - vm_p["R"]
                dS_meas = s_mp_c - s_mp_p
                dS_pred = off["P"] - off["L"] - off["R"]
                mismatch = abs(dS_meas - dS_pred)
                L.update({
                    "s_double_computed": fx(s_dbl_c), "s_double_pinned": fx(s_dbl_p), "s_mp_computed": fx(s_mp_c), "s_mp_pinned": fx(s_mp_p),
                    "dS_measured": fx(dS_meas), "dS_predicted": fx(dS_pred), "dS_mismatch": fx(mismatch),
                    "offsets": {k: fx(v) for k, v in off.items()},
                    "values_mp_computed": {k: fx(v) for k, v in vm_c.items()}, "values_mp_pinned": {k: fx(v) for k, v in vm_p.items()},
                    "dbl_minus_mp_computed": fx(s_dbl_c - s_mp_c), "dbl_minus_mp_pinned": fx(s_dbl_p - s_mp_p),
                })
                flagged = floor_c > 0 and res_c > 10 * floor_c
                L["flagged"] = bool(flagged)
                if flagged:
                    i_ = not lv["tile"]
                    ii = res_p <= 10 * floor_p
                    iii = float(mismatch) <= 0.1 * float(abs(dS_pred)) + floor_c
                    L["tests"] = {"i_no_tile": bool(i_), "ii_pinned_le_10floor": bool(ii), "iii_change_matches_offsets": bool(iii)}
                    L["attributed_to_tiling"] = bool(i_ and ii and iii)
                    comps = {"resolution of the scheme (mp, pinned ends)": abs(float(s_mp_p)),
                             "solve rounding (double - mp, computed ends)": abs(float(s_dbl_c - s_mp_c)),
                             "end-node offset (mp, computed - pinned)": abs(float(dS_meas))}
                    L["largest_component"] = max(comps, key=comps.get)
                    L["components"] = comps
        elif lv["depth"] in DEEP:
            L["flagged"] = False
        out["levels"].append(L)
    # ---------------------------------------------------------------- stationary point
    thd = theta["theta_deriv"]
    x0, x1 = tuple(rec["x_span"])
    xs = np.linspace(x0, x1, 200001)
    xs[0], xs[-1] = x0, x1
    tps = np.array([thd(float(x)) for x in xs])
    max_span = float(np.max(np.abs(tps)))
    lv16 = [lv for lv in line["levels"] if lv["depth"] == 16][0]
    ra, rb = float.fromhex(lv16["a"]), float.fromhex(lv16["b"])
    xr = list(np.linspace(ra, rb, 20001))
    for lv in line["levels"]:
        if lv["depth"] >= 16:
            a_, b_, c_ = (float.fromhex(lv[k]) for k in ("a", "b", "c"))
            for s in ((a_, b_), (a_, c_), (c_, b_)):
                xr.extend(float(v) for v in LQ.chebyshev_matrices(s, order)[0])
    xr = np.array(sorted(set(float(v) for v in xr)))
    tpr = np.array([thd(float(x)) for x in xr])

    def zeros(xa, ta):
        zs = []
        for i in np.nonzero(np.signbit(ta[1:]) != np.signbit(ta[:-1]))[0]:
            lo, hi = float(xa[i]), float(xa[i + 1])
            slo = math.copysign(1.0, thd(lo))
            while True:
                mid = 0.5 * (lo + hi)
                if mid <= lo or mid >= hi:
                    break
                if math.copysign(1.0, thd(mid)) == slo:
                    lo = mid
                else:
                    hi = mid
            xm = 0.5 * (lo + hi)
            h = 1e-6
            tpp = (thd(xm + h) - thd(xm - h)) / (2 * h)
            zs.append({"x": xm, "x_hex": hx(xm), "z": math.expm1(xm) if e["block"].startswith("G") else None,
                       "theta_prime_lo": thd(lo), "theta_prime_hi": thd(hi), "theta_second": tpp})
        return zs

    zr = zeros(xr, tpr)
    imin = int(np.argmin(np.abs(tpr)))
    near = float(np.abs(tpr[imin])) <= 1e-6 * max_span
    xm = float(xr[imin])
    h = 1e-6
    out["stationary"] = {
        "max_abs_theta_prime_span": max_span, "span_sign_changes": zeros(xs, tps),
        "region16": [hx(ra), hx(rb)], "region16_points": int(len(xr)),
        "region16_min_abs_theta_prime": float(np.abs(tpr[imin])), "region16_min_at_x": xm,
        "region16_min_at_z": math.expm1(xm) if e["block"].startswith("G") else None,
        "region16_theta_prime_at_min": float(tpr[imin]), "region16_theta_second_at_min": (thd(xm + h) - thd(xm - h)) / (2 * h),
        "region16_min_over_span_max": float(np.abs(tpr[imin])) / max_span if max_span > 0 else None,
        "region16_sign_changes": zr, "holds": bool(zr) or near,
    }
    # ---------------------------------------------------------------- the call's class
    deep = [L for L in out["levels"] if L["depth"] in DEEP]
    flagged = [L for L in deep if L.get("flagged")]
    att = [L for L in flagged if L.get("attributed_to_tiling")]
    is_a = bool(flagged) and len(att) == len(flagged)
    is_b = out["stationary"]["holds"]
    if is_a and is_b:
        cls = "a" if flagged[-1].get("attributed_to_tiling") else "b"
    elif is_a:
        cls = "a"
    elif is_b:
        cls = "b"
    else:
        cls = "c"
    comp = None
    if cls == "c":
        na = [L for L in flagged if not L.get("attributed_to_tiling")]
        if na:
            tot = {}
            for L in na:
                for k, v in L["components"].items():
                    tot[k] = tot.get(k, 0.0) + v
            comp = max(tot, key=tot.get)
        else:
            comp = "no flagged deep level: deep parents are " + ",".join(sorted({L["types"][0] for L in deep}))
    out["summary"] = {
        "n_deep_flagged": len(flagged), "n_deep_attributed": len(att), "is_a": is_a, "is_b": is_b, "class": cls,
        "class_c_component": comp, "all_levels_reproduce_census": all(L["reproduces_census_resid_and_floor"] for L in out["levels"]),
        "deep_flagged_not_attributed": [{"depth": L["depth"], "tests": L["tests"], "largest_component": L["largest_component"]} for L in flagged if not L["attributed_to_tiling"]],
        "seconds": time.perf_counter() - t0,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captured")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.workers > 6:
        raise SystemExit("at most six workers")
    cap_p = Path(a.captured).resolve()
    hdr = header(Path(__file__).resolve(), "depth20-tiling step 3: classification re-solves",
                 [cap_p, STATE / "cheap_pass.json", HERE / "dcommon.py", Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6/AdaptiveLevin/levin_quadrature.py")])
    for l in hdr:
        print(l, flush=True)
    ents = json.loads(cap_p.read_text())
    nos = [e["sample_no"] for e in ents if e.get("capture_matches_census")]
    if a.only:
        keep = {int(x) for x in a.only.split(",")}
        nos = [n for n in nos if n in keep]
    od = STATE / "classify"
    od.mkdir(exist_ok=True)
    with cf.ProcessPoolExecutor(max_workers=a.workers, max_tasks_per_child=1) as ex:
        futs = {ex.submit(one, (str(cap_p), n)): n for n in nos}
        for fu in cf.as_completed(futs):
            n = futs[fu]
            try:
                r = fu.result()
            except Exception as exc:  # noqa: BLE001
                import traceback

                print(f"sample {n}: ERROR {exc!r}\n{traceback.format_exc()}", flush=True)
                continue
            r["header"] = hdr
            (od / f"{n:02d}.json").write_text(json.dumps(r, indent=1))
            s = r["summary"]
            print(f"sample {n:2d} {r['rtol']} {r['block']:<6} {r['unit']} {r['index']}: class {s['class']} (a {s['is_a']}, b {s['is_b']}); "
                  f"deep flagged {s['n_deep_flagged']}, attributed {s['n_deep_attributed']}; reproduces census line {s['all_levels_reproduce_census']}; "
                  f"{s['seconds']:.1f} s; c: {s['class_c_component']}; not attributed {s['deep_flagged_not_attributed']}", flush=True)


if __name__ == "__main__":
    main()
