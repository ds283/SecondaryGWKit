"""levin-phase-floor prompt 07d Part A' (section 9.2 items 0-5): one call's leaves, each measured for every item. New in
Part A'. Run at the tree's module (07c's code, imported unchanged); nothing is written but OUT.

Per leaf [a, b] of a captured call (N the call's order, 24):
- **V** as the driver computes it (``A.build_Levin_data`` on the leaf's order-N nodes; Part A's way);
- **item 0, v4** (state/refs_rule_leaf_v4.txt, leafref_v4.py): sigma_hat by v3's least squares (and v3's own least-squares
  ref, to check against Part A's record), then v4's ref, eps_ref, settled, the pieces' panels;
- **the Clenshaw-Curtis values:** CC_47 by the tree's gate solve (``_adaptive_levin_subregion(force_direct=True)``, as Part
  A: its value, pair |CC_47 - CC_24|, floor, floor less node, node error, resasc); CC_24 from its fine samples; CC_47, CC_93
  and CC_185 also by ``checkrepro_a2.CallContext.cc_value`` (np.dot, as the tree and V93 form them; CC_47's must equal the
  solve's), CC_93 by math.fsum too (|dot - fsum|); each rule's Chebyshev coefficients (Part A's ``dct1_coeffs``), its
  standardChop cutoff and plateau (Part A's ``plateau_sigma``), sum w^2, the tail of its last four coefficients, and
  s_own as Part A defined it (plateau, else the rms of the last 8 coefficients, x sqrt((n - 1)/2) x sqrt(sum w^2));
- **item 1:** ``checkrepro_a2.check`` at each T of {4, 6, 8, 10, 12, 16, 20} pi with V >= T, under the CC_47 and the CC_93
  fallback (each check's wall and solves);
- **item 4** (group leaves whose knots production_edges reads): the knots by kind (``typed_knots``: phase spline knots,
  the Green's function amplitude spline's knots, the cumulative tables' nodes, their nearest-node midpoints, their declared
  breaks; production_edges' own walk, kept apart by kind); per kind, the count inside the leaf and the composite CC_47,
  CC_93 and V_P (the Levin solve, gate disabled, on each piece) on the leaf split at that kind's knots alone, and at all
  kinds together; and the jump at the kind's knot nearest the leaf's midpoint in the value and first three derivatives of
  f_0, f_1 and the raw phase, by one-sided degree-6 least-squares fits on 24 Chebyshev points each side of width h, against
  the same measured at a knot-free control point of the leaf with the same h;
- **item 5:** walls of the CC_47 solve, CC_93, CC_185, and one Levin solve (gate disabled) of the leaf, with points sampled.

Spec JSON: Part A's leaf spec (out/leaves/specs/<round>/<sid>.json: capture, rtol, atol, call_value, x_span, leaves with id,
start, end, type, value_own), or this Part's (the same keys, and a leaf's ``fate`` for the new band).
Usage: PYTHONPATH=<tree>:<scripts> python -B leaf_a2.py SPEC OUT
"""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prov import TREE_LQ_SHA, sha256  # noqa: E402

T_GRID_PI = [4, 6, 8, 10, 12, 16, 20]
TAIL_K = 8
JUMP_M = 24
JUMP_DEG = 6


def H(x):
    return None if x is None else float(x).hex()


def typed_knots(call):
    """production_edges' walk (tools/filon_reference.py), each x kept with its kind; raises where it raises."""
    ta = call["f"][0].__closure__[2].cell_contents
    cl = dict(zip(ta.__wrapped__.__code__.co_freevars, [c.cell_contents for c in ta.__wrapped__.__closure__]))
    active = call["theta"]["theta"].__closure__[0].cell_contents
    kinds = {"phase": set(), "G": set(), "node": set(), "mid": set(), "break": set()}
    tables = []
    for _, fac in active:
        ph = fac.phase
        kinds["phase"].update(float(t) for t in ph._spline.t)
        tables.append(ph._leading._table)
    for name in ("q_factor", "r_factor"):
        f = cl[name]
        if hasattr(f, "_f") and hasattr(f._f, "_friction_F"):
            tables.append(f._f._friction_F._table)
    G = cl["G_factor"]
    if hasattr(G, "_Gk"):
        kinds["G"].update(float(t) for t in G._Gk.sin_amplitude._spline.t)
    seen = set()
    for t in tables:
        if id(t) in seen:
            continue
        seen.add(id(t))
        asc = np.asarray(t._u_asc, dtype=float)
        kinds["node"].update(asc.tolist())
        kinds["mid"].update((asc[:-1] + (asc[1:] - asc[:-1]) / 2).tolist())
        if t._breaks is not None:
            kinds["break"].update(float(v) for v in np.asarray(t._breaks).ravel())
    return {k: sorted(v) for k, v in kinds.items()}


def coeff_block(LQ, ck_fn, plateau_fn, span, order, g):
    ck = ck_fn(g)
    cut, sig_pl = plateau_fn(ck)
    n = len(ck)
    w = LQ._cc_weights(span, order)
    sw2 = float(np.sum(w * w))
    tail = ck[cut:] if cut < n else ck[-TAIL_K:]
    sig_own = float(np.sqrt(np.mean(tail * tail)) * math.sqrt((n - 1) / 2.0))
    width = span[1] - span[0]
    chop_tail = (0.5 * width * float(np.sum(np.abs(ck[max(cut - 4, 0):cut])))) if cut < n else \
        0.5 * width * float(np.sum(np.abs(ck[-4:])))
    return {"n": n, "chop": int(cut), "sigma_plateau": sig_pl, "sigma_own": sig_own, "sum_w2": sw2,
            "s_own": sig_own * math.sqrt(sw2), "tail_sum4": H(0.5 * width * float(np.sum(np.abs(ck[-4:])))),
            "chop_tail": H(chop_tail), "ck_last8": [H(x) for x in ck[-8:]]}


def one_sided(fun, x0, h, side):
    """Value and first three derivatives at x0 of a degree-JUMP_DEG least-squares fit on JUMP_M points on one side."""
    tt = np.cos(np.pi * (np.arange(JUMP_M) + 0.5) / JUMP_M)  # in (-1, 1)
    xs = x0 + side * h * 0.5 * (1.0 + tt)  # (x0, x0 + h) or (x0 - h, x0)
    ys = fun(xs)
    s = side * 0.5 * (1.0 + tt)  # (x - x0)/h
    co = np.polynomial.polynomial.polyfit(s, ys, JUMP_DEG)
    resid = ys - np.polynomial.polynomial.polyval(s, co)
    return [co[m] * math.factorial(m) / h ** m for m in range(4)], float(np.sqrt(np.mean(resid ** 2)))


def jump_at(fun, x0, h):
    L, rl = one_sided(fun, x0, h, -1)
    R, rr = one_sided(fun, x0, h, +1)
    return [abs(R[m] - L[m]) for m in range(4)], [abs(L[0]), abs(R[0])], max(rl, rr)


def main():
    spec_p, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    import AdaptiveLevin.levin_quadrature as LQ

    if sha256(LQ.__file__) != TREE_LQ_SHA:
        raise SystemExit("the module is not the tree's")
    import checkrepro_a2 as CR
    import leafref_v4 as R4
    from leafcc_07d import dct1_coeffs, plateau_sigma

    spec = json.loads(spec_p.read_text())
    rec = R4.load_capture(spec["capture"])
    kw = dict(rec["kw"])
    kw["rtol"] = float(spec["rtol"])
    C = CR.CallContext(LQ, rec, kw)
    N = C.N
    G = R4.Integrand(LQ, rec)
    xs0, xs1 = (float.fromhex(v) for v in spec["x_span"])
    rtol, atol = float(spec["rtol"]), float(spec["atol"])
    call_value = float.fromhex(spec["call_value"])
    request = max(atol, rtol * abs(call_value))
    edges_all, n_tables, edges_note = R4.call_knots(rec, xs0, xs1)
    try:
        tk = typed_knots(rec)
        tk_err = None
    except Exception as e:  # noqa: BLE001
        tk, tk_err = None, repr(e)[:200]
    raw = rec["theta"]["theta"]
    out = []
    t_all = time.perf_counter()
    for lf in spec["leaves"]:
        try:
            t_leaf = time.perf_counter()
            a, b = float.fromhex(lf["start"]), float.fromhex(lf["end"])
            if not a < b:
                out.append({"id": lf["id"], "error": "not an ascending span"})
                continue
            width = b - a
            grid_N, D_N = LQ.chebyshev_matrices((a, b), N)
            V = float(C.A.build_Levin_data(grid_N, D_N, need_AmatT=False, vectorize_cache=C.cache)[4])
            knots = R4.interior(edges_all, a, b)
            # --- item 0: v4
            t0 = time.perf_counter()
            n0 = G.n_eval
            s3 = R4.v3_sigma(G, a, b, knots, V)
            t1 = time.perf_counter()
            r4 = R4.v4_reference(G, a, b, knots, s3["sigma_hat"])
            t2 = time.perf_counter()
            v4 = {"ref": H(r4["ref"]), "eps_ref": H(r4["eps_ref"]), "settled": r4["settled"], "n_pieces": r4["n_pieces"],
                  "n_unsettled_pieces": r4["n_unsettled_pieces"], "max_panels": max(r4["panels"]),
                  "panels_hist": {str(n): r4["panels"].count(n) for n in sorted(set(r4["panels"]))},
                  "unsettled_pieces": [{k: p[k] for k in ("a", "b", "n", "D", "c", "history")}
                                       for p in r4["pieces"] if not p["settled"]][:20],
                  "evals": G.n_eval - n0, "wall_s": t2 - t1}
            v3s = {"sigma_hat": s3["sigma_hat"], "ref_v3_ls": H(s3["ref_v3_ls"]), "eps_LS": H(s3["eps_LS"]),
                   "wall_s": t1 - t0}
            # --- the Clenshaw-Curtis values
            t0 = time.perf_counter()
            d = LQ._adaptive_levin_subregion((a, b), rec["f"], C.A, id_label=C.id_label, chebyshev_order=N, rtol=C.rtol,
                                             vectorize_cache=C.cache, force_direct=True)
            w_cc47 = time.perf_counter() - t0
            cc47_dot, _, g47 = C.cc_value((a, b), 2 * N - 1)
            cc24 = float(np.dot(LQ._cc_weights((a, b), N), g47[::2]))
            t0 = time.perf_counter()
            cc93, grid93, g93 = C.cc_value((a, b), 4 * N - 3)
            w_cc93 = time.perf_counter() - t0
            cc93_fsum = math.fsum(LQ._cc_weights((a, b), 4 * N - 3) * g93)
            t0 = time.perf_counter()
            cc185, _, g185 = C.cc_value((a, b), 8 * N - 7)
            w_cc185 = time.perf_counter() - t0
            cc = {"cc47": H(d["value"]), "cc47_dot_equal": H(cc47_dot) == H(d["value"]), "cc24": H(cc24),
                  "pair": H(d["abserr_direct"]), "floor": H(d["phase_err"]), "floor_less_node": H(d.get("phase_err_less_node")),
                  "node_E": H(d.get("node_error_signed")), "resasc": H(d.get("resasc")),
                  "cc93": H(cc93), "cc93_fsum": H(cc93_fsum), "cc185": H(cc185),
                  "c47": coeff_block(LQ, dct1_coeffs, plateau_sigma, (a, b), 2 * N - 1, g47),
                  "c93": coeff_block(LQ, dct1_coeffs, plateau_sigma, (a, b), 4 * N - 3, g93),
                  "c185": coeff_block(LQ, dct1_coeffs, plateau_sigma, (a, b), 8 * N - 7, g185)}
            # --- item 5: one Levin solve, gate disabled
            t0 = time.perf_counter()
            try:
                lp = C.solve((a, b), -math.inf)
                lev_one = {"V_P": H(lp["value"]), "floor": H(lp.get("phase_err")), "trunc": H(lp.get("abserr_truncation")),
                           "guarded": bool(lp.get("guarded", False)), "lstsq": bool(V >= LQ._LEVIN_DIRECT_SOLVE_PHASE_SPAN)}
            except Exception as e:  # noqa: BLE001
                lev_one = {"error": repr(e)[:200]}
            w_lev = time.perf_counter() - t0
            cost = {"cc47_solve_s": w_cc47, "cc93_s": w_cc93, "cc185_s": w_cc185, "levin_solve_s": w_lev,
                    "points": {"cc47": 2 * N - 1, "cc93": 4 * N - 3, "cc185": 8 * N - 7, "levin_solve": N}}
            # --- item 1: the checks at each T
            item1 = []
            for Tp in T_GRID_PI:
                T = Tp * math.pi
                if V < T:
                    continue
                for fb in ("cc47", "cc93"):
                    s0, c0 = C.n_solves, C.n_cc93
                    t0 = time.perf_counter()
                    try:
                        ch = C.check(a, b, T, fb)
                    except Exception as e:  # noqa: BLE001
                        ch = {"T": T, "fallback": fb, "error": repr(e)[:200]}
                    ch.update({"T_pi": Tp, "wall_s": time.perf_counter() - t0, "solves": C.n_solves - s0,
                               "cc93_rules": C.n_cc93 - c0})
                    item1.append(ch)
            # --- item 4: the knots by kind (group leaves)
            item4 = None
            if tk is not None and knots is not None:
                item4 = {"counts": {}, "split": {}, "jumps": {}}
                allk = sorted(set(knots))
                kind_sets = {k: [x for x in v if a < x < b] for k, v in tk.items()}
                kind_sets["all"] = allk
                for k, ks in kind_sets.items():
                    item4["counts"][k] = len(ks)
                    if not ks:
                        continue
                    brk = [a] + ks + [b]
                    c47 = math.fsum(C.cc_value((pa, pb), 2 * N - 1)[0] for pa, pb in zip(brk[:-1], brk[1:]))
                    c93 = math.fsum(C.cc_value((pa, pb), 4 * N - 3)[0] for pa, pb in zip(brk[:-1], brk[1:]))
                    try:
                        vps = math.fsum(C.solve((pa, pb), -math.inf)["value"] for pa, pb in zip(brk[:-1], brk[1:]))
                    except Exception as e:  # noqa: BLE001
                        vps = None
                    item4["split"][k] = {"n_pieces": len(brk) - 1, "cc47": H(c47), "cc93": H(c93), "V_P": H(vps)}
                # jumps: per kind, the knot nearest the midpoint; a knot-free control point
                pts = [a] + allk + [b]
                gaps = sorted(((pts[i + 1] - pts[i], i) for i in range(len(pts) - 1)), reverse=True)
                ctrl = 0.5 * (pts[gaps[0][1]] + pts[gaps[0][1] + 1])
                ctrl_half = 0.5 * gaps[0][0]
                f0, f1 = rec["f"]

                def F(fn):
                    return lambda xs: LQ._sample_vectorized(fn, np.asarray(xs, dtype=float), C.cache, ("f", id(fn)))

                def TH(x0):
                    t0_ = float(raw(x0))
                    return lambda xs: np.array([float(raw(x)) for x in xs]) - t0_

                for k in ("phase", "G", "node", "mid", "break"):
                    ks = kind_sets[k]
                    if not ks:
                        continue
                    xk = min(ks, key=lambda x: abs(x - 0.5 * (a + b)))
                    i = pts.index(xk) if xk in pts else None
                    nb_lo = max(x for x in pts if x < xk)
                    nb_hi = min(x for x in pts if x > xk)
                    h = 0.45 * min(xk - nb_lo, nb_hi - xk)
                    hc = min(h, 0.45 * ctrl_half)
                    rowk = {"x": H(xk), "h": h, "h_control": hc}
                    for nm, fun_at in (("f0", lambda x0: F(f0)), ("f1", lambda x0: F(f1)), ("theta_raw", TH)):
                        try:
                            jk, vals, rk = jump_at(fun_at(xk), xk, h)
                            jc, _, rc = jump_at(fun_at(ctrl), ctrl, h) if h <= 0.45 * ctrl_half else (None, None, None)
                            jc2, _, rc2 = jump_at(fun_at(ctrl), ctrl, hc)
                            rowk[nm] = {"jump": jk, "value_abs": vals, "fit_rms": rk, "control_same_h": jc,
                                        "control_h_control": jc2, "control_fit_rms": rc2}
                        except Exception as e:  # noqa: BLE001
                            rowk[nm] = {"error": repr(e)[:200]}
                    item4["jumps"][k] = rowk
            out.append({"id": lf["id"], "start": lf["start"], "end": lf["end"], "type": lf.get("type"),
                        "fate": lf.get("fate"), "N": N, "V": V, "V_over_pi": V / math.pi, "width": width,
                        "n_knots": None if knots is None else len(knots), "request": request,
                        "request_share": request * width / abs(xs1 - xs0), "v4": v4, "v3_sigma": v3s, "cc": cc,
                        "levin_one": lev_one, "cost": cost, "item1": item1, "item4": item4,
                        "wall_s": time.perf_counter() - t_leaf})
        except Exception as e:  # noqa: BLE001
            import traceback

            out.append({"id": lf["id"], "error": "exception: " + repr(e)[:300], "traceback": traceback.format_exc()[-1500:]})
    doc = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
                      "leafref_v4_sha256": sha256(R4.__file__), "checkrepro_sha256": sha256(CR.__file__),
                      "leafcc_07d_sha256": sha256(Path(__file__).resolve().parent / "leafcc_07d.py"),
                      "module": LQ.__file__, "module_sha256": TREE_LQ_SHA, "spec": str(spec_p),
                      "spec_sha256": hashlib.sha256(spec_p.read_bytes()).hexdigest()},
           "sid": spec.get("sid"), "rtol": rtol, "N": N, "edges_note": edges_note, "n_tables": n_tables,
           "typed_knots_error": tk_err, "wall_s": time.perf_counter() - t_all, "leaves": out}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc))
    print(f"{spec_p.name}: {len(out)} leaves, {sum(1 for x in out if 'error' in x)} errors, {doc['wall_s']:.1f} s")


if __name__ == "__main__":
    main()
