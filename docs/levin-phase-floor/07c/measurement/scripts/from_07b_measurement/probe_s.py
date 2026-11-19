"""levin-phase-floor prompt 06j, Part S: S-a to S-e on one sampled call's sampled slow leaves, in this process, by
probes of the leaf rule in isolation. New in 06j. Every definition is
state/predictions_and_rules_20261103T0905Z.md section 3's (written before this script ran).

The capture (state/inputs/<sid>.pkl; loaded with ray.cloudpickle, this directory on PYTHONPATH after the tree)
gives the call's f, theta and keyword arguments. The **leaf rule** is ``_adaptive_levin``'s ``_check_leaf``
evaluated on a span: seven calls of the tree's own ``AdaptiveLevin.levin_quadrature._adaptive_levin_subregion``
with the call's own arguments (f; ``_Basis_SinCos(theta)``; ``chebyshev_order`` and ``rtol`` from the capture's
kw; a vectorise cache per probe set; diagnostics under this records directory), the midpoints formed as the
driver forms them ((b + a) / 2). Nothing in the tree is changed. S-c's perturbations are pass-throughs:
(i) at the module attribute ``chebyshev_matrices`` (installed for the perturbed solves only, restored in a
finally), (ii) wrappers around f_sin, f_cos and the phase dict's callables, handed to the solve in place of the
originals.

Per sampled leaf it records:
- **S-a:** the seams it holds (kind, source) or its distance to the nearest seam; per call the null at depth 20
  and the census's trusted returned leaves at depth >= 15 under the same test;
- **check:** the rule on the leaf's own span against the census's delta1, delta2, r and phase_err, by hex;
- **S-b:** the two branches to depth 24 (seam / off-seam, or max / min);
- **S-c:** the two perturbation sets, 16 draws each;
- **S-d:** the rule on the sub-spans either side of each held seam;
- **S-e:** the one-sided limits at each held seam, and the stencil at a control point.

Writes OUT_JSON (gzip if it ends in .gz). Usage:
PYTHONPATH=<tree>:<this directory> python -B probe_s.py SID OUT_JSON
"""

import gzip
import hashlib
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
import seams_06j as SM  # noqa: E402
from prov import header  # noqa: E402

N_DRAWS = 16
SEED_I = 2026110301
STEPS = (2, 4, 6, 8)  # h = w 2^-k
K_MAX = 4  # nodes moved by k ulp, k in {-K_MAX, ..., K_MAX}
TRUST_MAX_RATIO = 0.25


def fx(v):
    return None if v is None else float(v).hex()


def ratio_of(d1, d2):
    if not (math.isfinite(d1) and math.isfinite(d2)):
        return math.inf
    if d1 == 0.0:
        return 0.0 if d2 == 0.0 else math.inf
    return d2 / d1


def trusted_b(r, d1, phi):
    return bool(r < TRUST_MAX_RATIO or (r < 1.0 and d1 <= phi))


class Probe:
    def __init__(self, LQ, rec):
        self.LQ = LQ
        self.f = rec["f"]
        self.theta = rec["theta"]
        self.kw = rec["kw"]
        self.order = self.kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
        self.rtol = self.kw.get("rtol", LQ.DEFAULT_LEVIN_RELTOL)
        self.label = self.kw.get("notify_label")
        self.A = LQ._Basis_SinCos(self.theta)
        self.cache = {}
        self.diag = RC.ROOT / "state" / "levin_diagnostics"
        self.n_solves = 0

    def solve(self, a, b, f=None, A=None, cache=None):
        self.n_solves += 1
        return self.LQ._adaptive_levin_subregion(
            (a, b), self.f if f is None else f, self.A if A is None else A, id_label="06j-probe",
            chebyshev_order=self.order, rtol=self.rtol, notify_label=self.label, build_p_sample=False,
            diagnostics_path=self.diag, vectorize_cache=self.cache if cache is None else cache)

    def rule(self, a, b):
        """_check_leaf on [a, b]."""
        P = self.solve(a, b)
        c = (b + a) / 2.0
        L = self.solve(a, c)
        R = self.solve(c, b)
        cL = (c + a) / 2.0
        cR = (b + c) / 2.0
        LL, LR, RL, RR = self.solve(a, cL), self.solve(cL, c), self.solve(c, cR), self.solve(cR, b)
        vp = P["value"]
        halves = L["value"] + R["value"]
        quarters = (LL["value"] + LR["value"]) + (RL["value"] + RR["value"])
        d1 = float(np.fabs(vp - halves))
        d2 = float(np.fabs(halves - quarters))
        r = ratio_of(d1, d2)
        phi = P.get("phase_err", 0.0) or 0.0
        return {"a": a, "b": b, "a_hex": fx(a), "b_hex": fx(b), "is_direct": bool(P.get("is_direct", False)),
                "V_P": vp, "V_P_hex": fx(vp), "halves": halves, "quarters": quarters, "delta1": d1, "delta2": d2,
                "r": r, "phi": phi, "d1_over_phi": d1 / phi if phi > 0 else math.inf,
                "trusted_b": trusted_b(r, d1, phi), "hex": {"delta1": fx(d1), "delta2": fx(d2), "r": fx(r), "phi": fx(phi)},
                "node_error_signed": P.get("node_error_signed"), "abserr_direct": P.get("abserr_direct")}


def branch(probe, sm_x, node, depth_to, choose):
    """From ``node`` (a rule dict at depth d), evaluate both children at each level to depth_to, following
    ``choose(children)``; returns the list of followed nodes (each with its sibling's numbers) below ``node``."""
    path = []
    cur = node
    d = node["depth"]
    while d < depth_to:
        a, b = cur["a"], cur["b"]
        c = (b + a) / 2.0
        kids = []
        for (x0, x1) in ((a, c), (c, b)):
            k = probe.rule(x0, x1)
            k["depth"] = d + 1
            k["seams_held"] = len(SM.held(sm_x, x0, x1))
            kids.append(k)
        pick = choose(kids)
        if pick is None:
            path.append({"depth": d + 1, "ended": "no child qualifies", "children": kids})
            break
        nxt = dict(kids[pick])
        nxt["sibling"] = {k_: kids[1 - pick][k_] for k_ in ("delta1", "delta2", "r", "phi", "seams_held", "trusted_b")}
        path.append(nxt)
        cur = nxt
        d += 1
    return path


def choose_seam(kids):
    with_seam = [i for i, k in enumerate(kids) if k["seams_held"] > 0]
    if not with_seam:
        return None
    if len(with_seam) == 1:
        return with_seam[0]
    return max(with_seam, key=lambda i: kids[i]["delta1"])


def choose_max(kids):
    return 0 if kids[0]["delta1"] >= kids[1]["delta1"] else 1


def choose_min(kids):
    return 1 - choose_max(kids)


# ------------------------------------------------------------------ S-c
def _ulp_move(x, k):
    x = float(x)
    if k == 0:
        return x
    return x + k * math.ulp(x)


def set_i(probe, a, b, draw):
    """(i) the nodes moved: a pass-through at LQ.chebyshev_matrices for the solves of this draw."""
    LQ = probe.LQ
    orig = LQ.chebyshev_matrices
    rng = np.random.Generator(np.random.PCG64(SEED_I + 100 * draw))

    def w(x_span, N):
        x, D = orig(x_span, N)
        x = np.array(x, dtype=float, copy=True)
        ks = rng.integers(-K_MAX, K_MAX + 1, size=x.shape[0])
        for i in range(x.shape[0]):
            x[i] = _ulp_move(x[i], int(ks[i]))
        return x, D

    LQ.chebyshev_matrices = w
    try:
        c = (b + a) / 2.0
        cache = {}
        P = probe.solve(a, b, cache=cache)
        L = probe.solve(a, c, cache=cache)
        R = probe.solve(c, b, cache=cache)
    finally:
        LQ.chebyshev_matrices = orig
    return P["value"], L["value"] + R["value"]


def _k_of(draw, x):
    h = hashlib.sha256(f"06j-ii|{draw}|{float(x).hex()}".encode()).digest()
    return (int.from_bytes(h[:4], "big") % (2 * K_MAX + 1)) - K_MAX


def _wrap_inputs(fn, draw):
    def g(x):
        if np.ndim(x) == 0:
            return fn(_ulp_move(float(x), _k_of(draw, float(x))))
        arr = np.asarray(x, dtype=float)
        moved = np.array([_ulp_move(float(v), _k_of(draw, float(v))) for v in arr.ravel()]).reshape(arr.shape)
        return fn(moved)

    return g


def set_ii(probe, a, b, draw):
    """(ii) the integrand's inputs at the nodes moved: f_sin, f_cos and the phase callables wrapped."""
    LQ = probe.LQ
    f2 = [_wrap_inputs(fn, draw) for fn in probe.f]
    th2 = dict(probe.theta)
    for key in ("theta", "theta_mod_2pi", "theta_deriv"):
        if key in th2 and callable(th2[key]):
            th2[key] = _wrap_inputs(th2[key], draw)
    A2 = LQ._Basis_SinCos(th2)
    cache = {}
    c = (b + a) / 2.0
    P = probe.solve(a, b, f=f2, A=A2, cache=cache)
    L = probe.solve(a, c, f=f2, A=A2, cache=cache)
    R = probe.solve(c, b, f=f2, A=A2, cache=cache)
    return P["value"], L["value"] + R["value"]


def scatter_set(probe, a, b, base, which):
    fn = set_i if which == "i" else set_ii
    vp, hv, d1 = [], [], []
    for d in range(N_DRAWS):
        p, h = fn(probe, a, b, d)
        vp.append(p - base["V_P"])
        hv.append(h - base["halves"])
        d1.append(abs(p - h) - base["delta1"])
    rms = lambda v: float(math.sqrt(sum(x * x for x in v) / len(v)))
    return {"draws": N_DRAWS, "rms_V_P": rms(vp), "rms_halves": rms(hv), "rms_delta1": rms(d1),
            "max_abs_V_P": float(max(abs(x) for x in vp)), "n_V_P_changed": sum(1 for x in vp if x != 0.0),
            "V_P_deviations": [fx(x) for x in vp]}


# ------------------------------------------------------------------ S-e
def _ingredients(probe, x):
    f_sin, f_cos = probe.f
    th = probe.theta
    vals, args = probe.A.eval_basis_and_arguments(x)
    fs, fc = float(f_sin(x)), float(f_cos(x))
    out = {"F": fs * float(vals[0]) + fc * float(vals[1]), "f_sin": fs, "f_cos": fc, "theta": float(th["theta"](x))}
    if "theta_mod_2pi" in th:
        out["theta_mod_2pi"] = float(th["theta_mod_2pi"](x))
    if "theta_deriv" in th:
        out["theta_deriv"] = float(th["theta_deriv"](x))
    return out


def _wrap_pi(v):
    return (v + math.pi) % (2 * math.pi) - math.pi


def _one_sided(probe, xs, h, side):
    """Quadratic through x_s + side * j * h, j = 1, 2, 3: (limit, first, second derivative) at x_s, per ingredient."""
    pts = [xs + side * j * h for j in (1, 2, 3)]
    ev = [_ingredients(probe, p) for p in pts]
    t = np.array([p - xs for p in pts])
    V = np.vander(t, 3, increasing=True)
    res = {}
    for k in ev[0]:
        y = np.array([e[k] for e in ev])
        if k == "theta_mod_2pi":
            y = y[0] + np.array([_wrap_pi(v - y[0]) for v in y])
        c = np.linalg.solve(V, y)
        res[k] = (float(c[0]), float(c[1]), float(2 * c[2]))
    return res


def stencil(probe, xs, h):
    L = _one_sided(probe, xs, h, -1)
    R = _one_sided(probe, xs, h, +1)
    out = {}
    for k in L:
        jv = R[k][0] - L[k][0]
        if k == "theta_mod_2pi":
            jv = _wrap_pi(jv)
        out[k] = {"left": L[k], "right": R[k], "jump": [jv, R[k][1] - L[k][1], R[k][2] - L[k][2]]}
    return out


# ------------------------------------------------------------------ main
def main():
    sid, out_p = sys.argv[1], Path(sys.argv[2])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists; not overwritten")
    import ray
    import ray.cloudpickle as cp

    import AdaptiveLevin.levin_quadrature as LQ

    t0 = time.perf_counter()
    sel = {e["sid"]: e for e in json.loads((RC.ROOT / "out" / "sample" / "selection.json").read_text())}[sid]
    ssam = {s["sid"]: s for s in json.loads((RC.ROOT / "out" / "sample" / "s_sample.json").read_text())["calls"]}[sid]
    pkl = RC.ROOT / "state" / "inputs" / f"{sid}.pkl"
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 06j: Part S probes of {sid} ({sel['unit']} #{sel['index']})",
                 [pkl, RC.ROOT / "out" / "sample" / "selection.json", RC.ROOT / "out" / "sample" / "s_sample.json",
                  Path(LQ.__file__), HERE / "seams_06j.py"])
    rec = cp.loads(pkl.read_bytes())
    if rec["unit"] != sel["unit"] or rec["index"] != sel["index"]:
        raise SystemExit("capture is not the selection's call")
    probe = Probe(LQ, rec)
    a0, b0 = (float(v) for v in rec["x_span"])
    sm, n_tables = SM.seams(rec)
    tool_check = SM.check_against_tool(rec, sm, n_tables)
    sm_x = np.array(sorted({s["x"] for s in sm}), dtype=float)
    kinds_at = {}
    for s in sm:
        kinds_at.setdefault(s["x"], []).append((s["kind"], s["source"]))
    by_kind = {}
    for s in sm:
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1
    null20 = SM.null_fraction(sm_x, a0, b0, 20)
    # the census's trusted returned leaves at depth >= 15 (comparison)
    run06g = RC.census_run_06g()
    name, off, ln, sha = sel["g_pickle"]
    with open(run06g / name, "rb") as fh:
        fh.seek(off)
        blob = fh.read(ln)
    if hashlib.sha256(blob).hexdigest() != sha:
        raise SystemExit("census pickle record sha256 differs")
    prec = pickle.loads(blob)
    chk = [x for x in prec["checked_leaves_06g"] if x is not None and x["index"] == sel["index"]][0]
    comp = []
    for i in range(len(chk["depth"])):
        if bool(chk["returned"][i]) and int(chk["trusted_final"][i]) == 1 and int(chk["depth"][i]) >= 15:
            a, b = float(chk["start"][i]), float(chk["end"][i])
            hs = SM.held(sm_x, a, b)
            de, dc = (None, None) if hs else SM.distance(sm_x, a, b)
            comp.append({"start": a, "end": b, "depth": int(chk["depth"][i]), "r": float(chk["convergence_ratio"][i]),
                         "holds_seam": bool(hs), "kinds": sorted({k for j in hs for k, _ in kinds_at[sm_x[j]]}),
                         "null_at_depth": SM.null_fraction(sm_x, a0, b0, int(chk["depth"][i])),
                         "dist_end_widths": de, "dist_centre_widths": dc})
    leaves_out = []
    for lv in ssam["leaves"]:
        a, b = lv["start"], lv["end"]
        w = b - a
        t_leaf = time.perf_counter()
        hs = SM.held(sm_x, a, b)
        held_list = [{"x": float(sm_x[j]), "x_hex": fx(sm_x[j]), "u_in_leaf": (float(sm_x[j]) - a) / w,
                      "kinds": kinds_at[sm_x[j]]} for j in hs]
        de, dc = (None, None) if hs else SM.distance(sm_x, a, b)
        L = {"start": a, "end": b, "depth": lv["depth"], "cluster": lv["cluster"], "census": lv,
             "S_a": {"holds_seam": bool(hs), "n_seams": len(hs), "seams": held_list,
                     "kinds": sorted({k for h_ in held_list for k, _ in h_["kinds"]}),
                     "dist_end_widths": de, "dist_centre_widths": dc}}
        # the rule on the leaf's own span: reproduces the census?
        root = probe.rule(a, b)
        root["depth"] = lv["depth"]
        root["seams_held"] = len(hs)
        repro = (root["hex"]["delta1"] == lv["hex"]["delta1"] and root["hex"]["delta2"] == lv["hex"]["delta2"]
                 and root["hex"]["r"] == lv["hex"]["convergence_ratio"] and root["hex"]["phi"] == lv["hex"]["phase_err"])
        L["check"] = {"reproduces_census": repro, "probe": root["hex"],
                      "census": {k: lv["hex"][k] for k in ("delta1", "delta2", "convergence_ratio", "phase_err")}}
        # S-b
        if hs:
            seam_path = branch(probe, sm_x, root, 24, choose_seam)
            # off-seam: the depth-21 child with no seam, then the larger-delta1 child
            c = (b + a) / 2.0
            kids = []
            for (x0, x1) in ((a, c), (c, b)):
                k = probe.rule(x0, x1)
                k["depth"] = lv["depth"] + 1
                k["seams_held"] = len(SM.held(sm_x, x0, x1))
                kids.append(k)
            free = [i for i, k in enumerate(kids) if k["seams_held"] == 0]
            if free:
                start = dict(kids[free[0]])
                start["sibling"] = {k_: kids[1 - free[0]][k_] for k_ in ("delta1", "delta2", "r", "phi", "seams_held", "trusted_b")}
                off_path = [start] + branch(probe, sm_x, start, 24, choose_max)
            else:
                off_path = [{"depth": lv["depth"] + 1, "ended": "both children hold a seam", "children": kids}]
            L["S_b"] = {"root": root, "seam_branch": seam_path, "off_seam_branch": off_path}
        else:
            L["S_b"] = {"root": root, "max_branch": branch(probe, sm_x, root, 24, choose_max),
                        "min_branch": branch(probe, sm_x, root, 24, choose_min)}
        # S-c
        base = {"V_P": root["V_P"], "halves": root["halves"], "delta1": root["delta1"]}
        sc_i = scatter_set(probe, a, b, base, "i")
        sc_ii = scatter_set(probe, a, b, base, "ii")
        big = max(sc_i["rms_V_P"], sc_ii["rms_V_P"])
        L["S_c"] = {"i_nodes": sc_i, "ii_inputs": sc_ii, "delta1": lv["delta1"], "phi": lv["phase_err"],
                    "delta1_over_max_scatter": lv["delta1"] / big if big > 0 else math.inf,
                    "max_scatter_over_phi": big / lv["phase_err"] if lv["phase_err"] > 0 else math.inf}
        # S-d
        if hs:
            cuts = [a] + [float(sm_x[j]) for j in hs] + [b]
            subs = []
            for x0, x1 in zip(cuts[:-1], cuts[1:]):
                rr = probe.rule(x0, x1)
                subs.append({k_: rr[k_] for k_ in ("a", "b", "a_hex", "b_hex", "is_direct", "delta1", "delta2", "r", "phi",
                                                     "d1_over_phi", "trusted_b", "hex", "V_P")} | {"width": x1 - x0})
            L["S_d"] = {"sub_spans": subs, "all_trusted": all(s_["trusted_b"] for s_ in subs)}
        # S-e
        if hs:
            ses = []
            for h_ in held_list:
                xs = h_["x"]
                st = {f"h=w*2^-{k}": stencil(probe, xs, w * 2.0 ** -k) for k in STEPS}
                hmax = w * 2.0 ** -STEPS[0]
                far = b if (b - xs) >= (xs - a) else a
                xc = (xs + far) / 2.0
                ok = len(SM.held(sm_x, xc - 3 * hmax, xc + 3 * hmax)) == 0
                ctrl = {f"h=w*2^-{k}": stencil(probe, xc, w * 2.0 ** -k) for k in STEPS} if ok else None
                at = _ingredients(probe, xs)
                ses.append({"x": xs, "x_hex": fx(xs), "z": math.expm1(xs), "kinds": h_["kinds"], "at_seam": at,
                            "stencils": st, "control_x": xc if ok else None, "control": ctrl})
            L["S_e"] = ses
        L["solves_so_far"] = probe.n_solves
        L["seconds"] = time.perf_counter() - t_leaf
        leaves_out.append(L)
        print(f"{sid} leaf [{a!r}, {b!r}] seams {len(hs)} reproduces {repro} r {root['r']:.4g} d1/phi {root['d1_over_phi']:.3g}; "
              f"scatter i {sc_i['rms_V_P']:.3e} ii {sc_ii['rms_V_P']:.3e} d1 {root['delta1']:.3e}; {L['seconds']:.1f} s", flush=True)
    doc = {"header": hdr, "sid": sid, "unit": sel["unit"], "index": sel["index"], "group": ssam["group"],
           "rtol": ssam["rtol"], "x_span": [a0, b0], "seams": {"n": len(sm), "n_distinct_x": int(len(sm_x)),
                                                              "by_kind": by_kind, "n_tables": n_tables,
                                                              "merged_equals_production_edges": tool_check,
                                                              "list": sm},
           "null_depth20": null20, "comparison_trusted_depth_ge_15": comp, "leaves": leaves_out,
           "n_solves": probe.n_solves, "seconds": time.perf_counter() - t0, "ray_initialized": ray.is_initialized()}
    def _ser(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return repr(o)

    data = json.dumps(doc, default=_ser).encode()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if str(out_p).endswith(".gz"):
        with gzip.GzipFile(out_p, "wb", mtime=0) as g:
            g.write(data)
    else:
        out_p.write_bytes(data)
    print(f"{sid}: {len(leaves_out)} leaves, {probe.n_solves} solves, {doc['seconds']:.1f} s; seams {len(sm)} "
          f"({by_kind}); merged = production_edges: {tool_check['equal']}; null20 {null20:.3e}; wrote {out_p} "
          f"sha256 {hashlib.sha256(out_p.read_bytes()).hexdigest()}; ray initialised {ray.is_initialized()}")
    if ray.is_initialized():
        raise SystemExit("Ray was initialised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
