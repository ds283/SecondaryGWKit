"""levin-phase-floor prompt 06k, Part L: the three routes' trust decisions at leaf level, each evaluated in
isolation on a leaf's own span in 06g's code (the tree's ``AdaptiveLevin.levin_quadrature``), as 06j's
probe_s.py evaluated rule (b). New in 06k (its solve and rule are probe_s.py's).

For one captured call (state/inputs/<sid>.pkl, 06j's capture re-made and checked against 06g's census)
and each leaf of its populations:
- **L-a** (an S call): 06j's sampled slow leaves of the call (06j's out/sample/s_sample.json);
- **L-b** (a W call): the 06h leaf that carried the call's move (06j's out/w_prep_v2/<sid>.json, the
  ``constant_move`` cell with the largest |share_of_move|, whose coarse leaf is 06h's), evaluated at
  its own span in 06g's code;
- **L-c** (an S call): every returned leaf of the call that 06g trusted (06g's census region pickle,
  ``checked_leaves_06g``: returned, trusted_final 1);

the seven solves P, L, R, LL, LR, RL, RR are made by the tree's ``_adaptive_levin_subregion`` with the
call's own arguments, the midpoints formed as the driver forms them. From them, as ``_check_leaf``
forms them: delta1, delta2, r, phi_P, the residual (the step-(4) residual |V_P - (V_L + V_R)| on the
Levin branch, the nested-pair estimate on the fallback); and each route's numbers:
- route 1: nu1 = (phi_P + phi_L) + phi_R;
- route 2: the values corrected by -E (each solve's node_error_signed, 0 if none), delta1c, delta2c,
  r_c;
- route 3: D = max(residual, delta1) / (1 - r) if r < 1, else max(residual, delta1 + delta2), against
  eta * the call's request, eta = 1e-3, the request max(atol, rtol |v_g|) from 06g's census value.

Decisions: ``g`` is rule (b); ``route1`` rule (b) with nu1; ``route2_literal`` rule (b) on the corrected
numbers alone; ``route2`` the variant's (g, or else route2_literal); ``route3`` g's trust, or else left
if D <= eta * request. Each is "trusted", "left" or "untrusted". Where the census holds the leaf's
check (L-a, L-c), delta1, delta2, r and phi_P are compared with it by hex.

Writes OUT_JSON. Usage: PYTHONPATH=<tree>:<this directory> python -B probe_l.py SID OUT_JSON [CAPTURE_PKL]
"""

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
from prov import header  # noqa: E402

TRUST_MAX_RATIO = 0.25
ETA = 1e-3
J06 = RC.G_TREE / "docs" / "levin-phase-floor" / "06j" / "out"
SEL = RC.ROOT / "out" / "sample" / "selection.json"
SSAM = J06 / "sample" / "s_sample.json"
WPREP = J06 / "w_prep_v2"


def fx(v):
    return None if v is None else float(v).hex()


def ratio_of(d1, d2):
    if not (math.isfinite(d1) and math.isfinite(d2)):
        return math.inf
    if d1 == 0.0:
        return 0.0 if d2 == 0.0 else math.inf
    return d2 / d1


def rule_b(r, d1, noise):
    return bool(r < TRUST_MAX_RATIO or (r < 1.0 and d1 <= noise))


def node(d):
    e = d.get("node_error_signed")
    return 0.0 if e is None else e


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

    def solve(self, a, b):
        self.n_solves += 1
        return self.LQ._adaptive_levin_subregion(
            (a, b), self.f, self.A, id_label="06k-probe", chebyshev_order=self.order, rtol=self.rtol,
            notify_label=self.label, build_p_sample=False, diagnostics_path=self.diag, vectorize_cache=self.cache)

    def leaf(self, a, b, request):
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
        is_direct = bool(P.get("is_direct", False))
        residual = float(P["abserr_direct"]) if is_direct else d1
        phi = lambda d: d.get("phase_err", 0.0) or 0.0  # noqa: E731  (as _make_leaf reads it)
        phiP, phiL, phiR = phi(P), phi(L), phi(R)
        nu1 = (phiP + phiL) + phiR
        # route 2: the corrected values
        vp_c = vp - node(P)
        halves_c = (L["value"] - node(L)) + (R["value"] - node(R))
        quarters_c = ((LL["value"] - node(LL)) + (LR["value"] - node(LR))) + ((RL["value"] - node(RL)) + (RR["value"] - node(RR)))
        d1c = float(np.fabs(vp_c - halves_c))
        d2c = float(np.fabs(halves_c - quarters_c))
        rc = ratio_of(d1c, d2c)
        D = max(residual, d1) / (1.0 - r) if r < 1.0 else max(residual, d1 + d2)
        g = rule_b(r, d1, phiP)
        r1 = rule_b(r, d1, nu1)
        r2l = rule_b(rc, d1c, phiP)
        r3_left = (not g) and math.isfinite(D) and D <= ETA * request

        def dec(t, left=False):
            return "trusted" if t else ("left" if left else "untrusted")

        return {
            "start": a, "end": b, "start_hex": fx(a), "end_hex": fx(b), "is_direct": is_direct,
            "V_P": vp, "halves": halves, "quarters": quarters, "residual": residual,
            "delta1": d1, "delta2": d2, "r": r, "phi_P": phiP, "phi_L": phiL, "phi_R": phiR, "nu1": nu1,
            "d1_over_phiP": d1 / phiP if phiP > 0 else math.inf, "d1_over_nu1": d1 / nu1 if nu1 > 0 else math.inf,
            "E": {k: d.get("node_error_signed") for k, d in (("P", P), ("L", L), ("R", R), ("LL", LL), ("LR", LR), ("RL", RL), ("RR", RR))},
            "types": {k: ("direct" if d.get("is_direct") else "Levin") for k, d in (("P", P), ("L", L), ("R", R), ("LL", LL), ("LR", LR), ("RL", RL), ("RR", RR))},
            "delta1c": d1c, "delta2c": d2c, "r_c": rc, "d1c_over_phiP": d1c / phiP if phiP > 0 else math.inf,
            "D": D, "request": request, "D_over_request": D / request, "D_over_eta_request": D / (ETA * request),
            "decision": {"g": dec(g), "route1": dec(r1), "route2_literal": dec(r2l), "route2": dec(g or r2l),
                         "route3": dec(g, r3_left)},
            "declared_g_formula": max(residual, d1) / (1.0 - r) if r < 1.0 else None,
            "declared_route2_literal": max(residual, d1c) / (1.0 - rc) if (r2l and rc < 1.0) else None,
            "hex": {"delta1": fx(d1), "delta2": fx(d2), "r": fx(r), "phi": fx(phiP)},
        }


def census_pickle(sel):
    name, off, ln, sha = sel["g_pickle"]
    with open(RC.census_run_06g() / name, "rb") as fh:
        fh.seek(off)
        blob = fh.read(ln)
    if hashlib.sha256(blob).hexdigest() != sha:
        raise SystemExit("census pickle record sha256 differs")
    return pickle.loads(blob)


def main():
    sid, out_p = sys.argv[1], Path(sys.argv[2])
    # a development check may name another capture file (default state/inputs/<sid>.pkl)
    pkl_arg = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    if out_p.exists():
        raise SystemExit(f"{out_p} exists; not overwritten")
    import ray
    import ray.cloudpickle as cp

    import AdaptiveLevin.levin_quadrature as LQ

    t0 = time.perf_counter()
    sel = {e["sid"]: e for e in json.loads(SEL.read_text())}[sid]
    pkl = pkl_arg if pkl_arg is not None else RC.ROOT / "state" / "inputs" / f"{sid}.pkl"
    inputs = [pkl, SEL, Path(LQ.__file__)]
    if sel["part"] == "S":
        inputs.append(SSAM)
    else:
        inputs.append(WPREP / f"{sid}.json")
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 06k: Part L probes of {sid} ({sel['unit']} #{sel['index']})", inputs)
    rec = cp.loads(pkl.read_bytes())
    if rec["unit"] != sel["unit"] or rec["index"] != sel["index"]:
        raise SystemExit("capture is not the selection's call")
    if not str(Path(LQ.__file__).resolve()).startswith(str(RC.G_TREE) + "/"):
        raise SystemExit(f"levin_quadrature is not the tree's: {LQ.__file__}")
    probe = Probe(LQ, rec)
    v_g = float.fromhex(sel["census_value"])
    atol = float(rec["kw"]["atol"])
    rtol = float(rec["kw"]["rtol"])
    request = max(atol, rtol * abs(v_g))
    leaves = []
    if sel["part"] == "S":
        ssam = {s["sid"]: s for s in json.loads(SSAM.read_text())["calls"]}[sid]
        for lv in ssam["leaves"]:
            L = probe.leaf(lv["start"], lv["end"], request)
            L.update({"population": "L-a", "depth": lv["depth"], "cluster": lv.get("cluster")})
            L["reproduces_census"] = all(L["hex"][k] == lv["hex"][kk] for k, kk in (("delta1", "delta1"), ("delta2", "delta2"), ("r", "convergence_ratio"), ("phi", "phase_err")))
            leaves.append(L)
        prec = census_pickle(sel)
        chk = [x for x in prec["checked_leaves_06g"] if x is not None and x["index"] == sel["index"]][0]
        for i in range(len(chk["depth"])):
            if not (bool(chk["returned"][i]) and int(chk["trusted_final"][i]) == 1):
                continue
            a, b = float(chk["start"][i]), float(chk["end"][i])
            L = probe.leaf(a, b, request)
            L.update({"population": "L-c", "depth": int(chk["depth"][i])})
            census = {"delta1": fx(chk["delta1"][i]), "delta2": fx(chk["delta2"][i]), "r": fx(chk["convergence_ratio"][i]), "phi": fx(chk["phase_err"][i])}
            L["census_hex"] = census
            L["reproduces_census"] = all(L["hex"][k] == census[k] for k in census)
            leaves.append(L)
    else:
        wp = json.loads((WPREP / f"{sid}.json").read_text())
        cells = wp["constant_move"]["cells"]
        big = max(cells, key=lambda c_: abs(c_["share_of_move"]))
        if big["coarse"] != "06h":
            raise SystemExit(f"{sid}: the carrying cell's coarse leaf is {big['coarse']}, not 06h's")
        h_leaves = [h for h in wp["h06"]["leaves"] if h["start"] == big["start"] and h["end"] == big["end"]]
        if len(h_leaves) != 1:
            raise SystemExit(f"{sid}: {len(h_leaves)} 06h leaves at the carrying cell's span")
        hl = h_leaves[0]
        L = probe.leaf(big["start"], big["end"], request)
        L.update({"population": "L-b", "depth": hl["depth"], "share_of_move": big["share_of_move"],
                  "h06_record": {k: hl.get(k) for k in ("convergence_ratio", "delta1", "phase_err", "abserr", "type", "trusted")},
                  "h06_d1_over_phi": hl["delta1"] / hl["phase_err"] if hl.get("phase_err") else None})
        L["reproduces_census"] = None
        leaves.append(L)
    doc = {"header": hdr, "sid": sid, "part": sel["part"], "unit": sel["unit"], "index": sel["index"], "rtol": rtol,
           "atol": atol, "v_g": v_g, "request": request, "eta": ETA, "leaves": leaves, "n_solves": probe.n_solves,
           "seconds": time.perf_counter() - t0, "ray_initialized": ray.is_initialized(),
           "levin_quadrature": str(Path(LQ.__file__).resolve())}

    def ser(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        return repr(o)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, default=ser))
    n_rep = sum(1 for L in leaves if L["reproduces_census"])
    n_chk = sum(1 for L in leaves if L["reproduces_census"] is not None)
    print(f"{sid} ({sel['part']}): {len(leaves)} leaves ({dict((p, sum(1 for L in leaves if L['population'] == p)) for p in ('L-a', 'L-b', 'L-c'))}); "
          f"reproduce the census {n_rep} of {n_chk}; {probe.n_solves} solves; {doc['seconds']:.1f} s; wrote {out_p} "
          f"sha256 {hashlib.sha256(out_p.read_bytes()).hexdigest()}; ray initialised {ray.is_initialized()}")
    if ray.is_initialized():
        raise SystemExit("Ray was initialised")
    return 0


if __name__ == "__main__":
    sys.exit(main())
