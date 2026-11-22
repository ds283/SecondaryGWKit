"""levin-phase-floor prompt 07c Part A, sections 2.2 and 2.3 (D1, D2): one call's Clenshaw-Curtis leaves, each solved as
the driver solved it and measured against a reference by rules independent of the nested pair. New in 07c.

For each leaf [a, b] of a captured call (07b's code, the tree's module, imported unchanged; nothing is written but OUT):
- **the solve, as the driver made it:** ``_adaptive_levin_subregion((a, b), f, A, ..., force_direct=forced)`` at the call's
  order N, A = ``_Basis_SinCos(theta)``: its value CC_{2N-1}, ``abserr_direct`` = |CC_{2N-1} - CC_N|, ``phase_err`` (the
  floor, ``_cc_roundoff_floor`` with its node term and the declared term), ``phase_err_less_node``, ``node_error_signed``,
  ``resasc`` and ``phase_span`` V. Where the spec gives the leaf's recorded own value, the re-solve must equal it in hex
  (``reproduces``);
- **CC_N** from the same fine samples (w_N . g[::2], as the driver forms it), and the fine samples g_j themselves;
- **the reference rules,** on the integrand as the driver evaluates it (f by ``_sample_vectorized``, the basis by
  ``A.eval_basis_and_arguments``, so theta_mod_2pi where supplied): Clenshaw-Curtis on 4N-3 and 8N-7 nodes
  (``chebyshev_matrices``, ``_cc_weights``) and composite Gauss-Legendre (GL_PANELS panels of GL_POINTS points, numpy's
  ``leggauss``), each weighted sum also formed exactly (``math.fsum`` of the rounded products). The reference is CC_{8N-7}
  (fsum); its stated error is max(|CC_{8N-7} - CC_{4N-3}|, |CC_{8N-7} - GL|), every term from the fsum forms;
- **a second, noise-averaging reference** (added after the first leaves measured: at the chase's leaves every interpolatory
  rule disagreed with every other above the leaf's floor, the integrand as evaluated carrying noise the floor does not
  model): the integrand sampled at LS_M Chebyshev points of the first kind on the span, fitted by Chebyshev least squares
  at degrees LS_DEG[0] and LS_DEG[1], each fit integrated exactly; ``ls_ref`` the higher degree's integral, its stated
  error |I_hi - I_lo| + 2 sigma (b - a) / sqrt(LS_M), sigma the higher fit's RMS residual, which is also reported per sample
  (``noise_sigma``) and against the leaf's scale (sum_j |w_j| |g_j| / (b - a));
- **the reference used** is whichever of the two has the smaller stated error (``ref_used``);
- **CC_{2N-1}'s true error** |CC_{2N-1} - ref| (CC_{2N-1} as the driver computed it), and whether it is resolved: true error
  above the reference's stated error;
- **candidate estimates of CC_{2N-1}'s error**, each with its cost in extra evaluations of the integrand:
  * ``pair``: |CC_{2N-1} - CC_N| (the tree's; 0 extra);
  * ``nest3``: |CC_{4N-3} - CC_{2N-1}|, the third nested rule (2N - 2 extra: the 4N-3 grid nests the 2N-1 grid);
  * ``tail_sum4``: (b - a)/2 * sum of |c_k| over the last four of the 2N - 1 Chebyshev coefficients of the fine
    interpolant (DCT-I of g_j; 0 extra);
  * ``tail_max4x2``: (b - a) * max of the last four |c_k| (0 extra);
  * ``dqk21``: resasc * min(1, (200 |CC_{2N-1} - CC_N| / resasc)^1.5), QUADPACK's transform of the pair (0 extra);
  * ``pair_sq``: |CC_{2N-1} - CC_N|^2 / max(|resasc|, tiny), a squared-pair form (0 extra).

Spec JSON: {"capture": {path, offset, length, sha256, override}, "leaves": [{"id", "start" (hex), "end" (hex),
"forced" (bool), "value_own" (hex or null)}]}. Usage: PYTHONPATH=<tree>:<scripts> python -B leafcc_07c.py SPEC OUT
"""

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

TREE_LQ_SHA = "8956c7b6a8ec285f8898353eed65f4039a02b42edc6927565be3896cda52def1"
GL_PANELS = 4
GL_POINTS = 48
LS_M = 600
LS_DEG = (24, 32)


def H(x):
    return None if x is None else float(x).hex()


def load_capture(c):
    import ray.cloudpickle as cp

    with open(c["path"], "rb") as f:
        f.seek(c["offset"])
        blob = f.read(c["length"])
    if hashlib.sha256(blob).hexdigest() != c["sha256"]:
        raise SystemExit("capture blob has another SHA-256")
    rec = cp.loads(blob)
    kw = dict(rec["kw"])
    if c.get("override"):
        kw["rtol"] = c["override"]["rtol"]
        kw["atol"] = c["override"]["atol"]
    return rec, kw


def main():
    spec_p, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    import AdaptiveLevin.levin_quadrature as LQ

    if hashlib.sha256(Path(LQ.__file__).read_bytes()).hexdigest() != TREE_LQ_SHA:
        raise SystemExit("the module is not the tree's")
    spec = json.loads(spec_p.read_text())
    rec, kw = load_capture(spec["capture"])
    f = rec["f"]
    A = LQ._Basis_SinCos(rec["theta"])
    N = int(kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER))
    rtol = kw["rtol"]
    cache = {}
    idl = LQ._LazyUUID()

    def g_on(grid):
        fs = np.vstack([LQ._sample_vectorized(fn, grid, cache, ("f", id(fn))) for fn in f])
        bs = np.array([A.eval_basis_and_arguments(x)[0] for x in grid]).T
        return np.sum(fs * bs, axis=0)

    def cc(span, M):
        grid, _ = LQ.chebyshev_matrices(span, M)
        g = g_on(grid)
        w = LQ._cc_weights(span, M)
        return float(np.dot(w, g)), math.fsum(w * g), g

    xg, wg = np.polynomial.legendre.leggauss(GL_POINTS)
    out = []
    t0 = time.perf_counter()
    for lf in spec["leaves"]:
        a, b = float.fromhex(lf["start"]), float.fromhex(lf["end"])
        lo, hi = min(a, b), max(a, b)
        sgn = 1.0 if a <= b else -1.0
        d = LQ._adaptive_levin_subregion((a, b), f, A, id_label=idl, chebyshev_order=N, rtol=rtol, vectorize_cache=cache,
                                         force_direct=bool(lf.get("forced")))
        if not d.get("is_direct", False):
            out.append({"id": lf["id"], "error": "the re-solve took the Levin branch"})
            continue
        fine = 2 * N - 1
        grid_f, _ = LQ.chebyshev_matrices((lo, hi), fine)
        g_f = g_on(grid_f)
        w_f = LQ._cc_weights((lo, hi), fine)
        w_c = LQ._cc_weights((lo, hi), N)
        cc_fine = sgn * float(np.dot(w_f, g_f))
        cc_coarse = sgn * float(np.dot(w_c, g_f[::2]))
        cc4, cc4x, _ = cc((lo, hi), 4 * N - 3)
        cc8, cc8x, _ = cc((lo, hi), 8 * N - 7)
        gl_terms = []
        edges = np.linspace(lo, hi, GL_PANELS + 1)
        for pa, pb in zip(edges[:-1], edges[1:]):
            xs = 0.5 * (pa + pb) + 0.5 * (pb - pa) * xg
            gl_terms.append(0.5 * (pb - pa) * wg * g_on(xs))
        glx = math.fsum(np.concatenate(gl_terms))
        cc4x, cc8x, glx = sgn * cc4x, sgn * cc8x, sgn * glx
        ref_cc = cc8x
        ref_cc_err = max(abs(cc8x - cc4x), abs(cc8x - glx))
        tt = np.cos(np.pi * (np.arange(LS_M) + 0.5) / LS_M)
        xs = 0.5 * (lo + hi) + 0.5 * (hi - lo) * tt
        gs = g_on(xs)
        ints, resid = [], None
        for deg in LS_DEG:
            co = np.polynomial.chebyshev.chebfit(tt, gs, deg)
            kk = np.arange(deg + 1)
            wk = np.where(kk % 2 == 0, 2.0 / (1.0 - kk.astype(float) ** 2), 0.0)
            ints.append(sgn * 0.5 * (hi - lo) * math.fsum(co * wk))
            resid = gs - np.polynomial.chebyshev.chebval(tt, co)
        sigma = float(np.sqrt(np.mean(resid * resid)))
        ls_ref = ints[1]
        ls_err = abs(ints[1] - ints[0]) + 2.0 * sigma * (hi - lo) / math.sqrt(LS_M)
        if ls_err < ref_cc_err:
            ref, ref_err, ref_used = ls_ref, ls_err, "ls"
        else:
            ref, ref_err, ref_used = ref_cc, ref_cc_err, "cc8"
        v = d["value"]
        true_err = abs(v - ref)
        # Chebyshev coefficients of the fine interpolant (extremal nodes, DCT-I)
        n = fine
        j = np.arange(n)
        gj = g_f.copy()
        wts = np.ones(n)
        wts[0] = wts[-1] = 0.5
        ck = np.array([(2.0 / (n - 1)) * np.sum(wts * gj * np.cos(np.pi * j * k / (n - 1))) for k in range(n)])
        ck[0] *= 0.5
        ck[-1] *= 0.5
        width = hi - lo
        pair = abs(d["abserr_direct"])
        resasc = d.get("resasc")
        cand = {
            "pair": pair,
            "nest3": abs(cc4 * sgn - v),
            "tail_sum4": 0.5 * width * float(np.sum(np.abs(ck[-4:]))),
            "tail_max4x2": width * float(np.max(np.abs(ck[-4:]))),
            "dqk21": (resasc * min(1.0, (200.0 * pair / resasc) ** 1.5)) if resasc else None,
            "pair_sq": pair * pair / max(abs(resasc or 0.0), 1e-300),
        }
        own = lf.get("value_own")
        out.append({
            "id": lf["id"], "start": lf["start"], "end": lf["end"], "forced": bool(lf.get("forced")), "N": N,
            "reproduces": None if own is None else (H(v) == own), "value": H(v), "value_check_fine": H(cc_fine),
            "cc_coarse": H(cc_coarse), "abserr_direct": H(d["abserr_direct"]), "phase_err": H(d["phase_err"]),
            "phase_err_less_node": H(d.get("phase_err_less_node")), "node_error_signed": H(d.get("node_error_signed")),
            "resasc": H(resasc), "phase_span": H(d.get("phase_span")), "V_over_6pi": float(d.get("phase_span") or 0.0) / (6 * math.pi),
            "cc4": H(cc4 * sgn), "cc4_fsum": H(cc4x), "cc8": H(cc8 * sgn), "cc8_fsum": H(cc8x), "gl_fsum": H(glx),
            "ref": H(ref), "ref_err": H(ref_err), "ref_used": ref_used, "ref_cc": H(ref_cc), "ref_cc_err": H(ref_cc_err),
            "ls_ref": H(ls_ref), "ls_err": H(ls_err), "noise_sigma": H(sigma),
            "noise_over_mean_abs_g": sigma / max(float(np.mean(np.abs(gs))), 1e-300),
            "true_err": H(true_err), "resolved": bool(true_err > ref_err),
            "scale": H(float(np.dot(np.abs(w_f), np.abs(g_f)))), "cheb_tail": [H(x) for x in ck[-6:]],
            "candidates": {k: H(x) for k, x in cand.items()},
            "cost_extra_evals": {"pair": 0, "nest3": 2 * N - 2, "tail_sum4": 0, "tail_max4x2": 0, "dqk21": 0, "pair_sq": 0},
        })
    doc = {"header": {"script": str(Path(__file__).resolve()), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "module": LQ.__file__, "module_sha256": TREE_LQ_SHA, "spec": str(spec_p),
                      "spec_sha256": hashlib.sha256(spec_p.read_bytes()).hexdigest(), "gl": [GL_PANELS, GL_POINTS],
                      "ls": [LS_M, list(LS_DEG)]},
           "unit": rec.get("unit"), "index": rec.get("index"), "rtol": rtol, "N": N, "wall_s": time.perf_counter() - t0, "leaves": out}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc))
    print(f"{spec_p.name}: {len(out)} leaves, {sum(1 for x in out if x.get('reproduces'))} reproduce, "
          f"{sum(1 for x in out if x.get('resolved'))} resolved, {doc['wall_s']:.1f} s")


if __name__ == "__main__":
    main()
