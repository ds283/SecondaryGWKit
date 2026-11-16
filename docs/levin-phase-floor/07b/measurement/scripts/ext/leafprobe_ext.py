"""prompt 07b extension (section 2.7): Part A's scripts/leafprobe_07b.py copied, changed only where marked "prompt 07b
extension": it lives in scripts/ext/, and converts the capture's rtol/atol override from the replay header's hex strings
(Part A probed no call of the other rtol, so never met it). Its text follows.

levin-phase-floor prompt 07b Part A, section 2.2: the trust check of chosen leaves, re-derived solve by solve at the
module the replay ran, with the eighths and the input perturbations (M3). New in 07b.

For each chosen leaf of a replay (``out/replays/<variant>/<sid>.json``, its returned leaves), with the call's captured
f and theta and the replay's (rtol, atol, chebyshev_order):
- the leaf's own solve P on [a, b], its halves L = [a, c], R = [c, b] (c = (b + a)/2, ``_levin_leaf.break_point``), its
  quarters (cL = (c + a)/2, cR = (b + c)/2, as ``_check_leaf``) and, with --eighths, its eighths (each quarter's
  midpoint by the same expression), each by the module's ``_adaptive_levin_subregion`` on a ``_Basis_SinCos(theta)`` --
  the call the driver makes, with ``force_direct`` the leaf's window flag;
- **checked against the replay:** V_P equals the leaf's ``value_own``, and on a trusted leaf Q equals its returned
  ``value``; delta1, delta2 and r equal the leaf's; the declared resolution max(residual, delta1)/(1 - r) equals its
  ``declared_resolution``; each by hex. A leaf whose re-derivation does not reproduce is reported (``reproduces``);
- recorded: V_P, V_L + V_R, Q and the eighths' sum; delta1, delta2, r, nu_B (``_trust_noise_yardstick``) and which clause
  trusted it; residual, floor, truncation, e'; depth, span, the solves' phase variation V (``phase_span``) and route; theta'
  at each end; every piece's value, floor, V and route;
- with --perturb (M3), the leaf's P, halves and quarters re-solved with (i) theta' scaled by 1 + s 2^-10 (s = +1, -1;
  ``_CC_THETA_PRIME_REL_ERROR_SUPPLIED``), and (ii) theta_mod_2pi shifted by s d at the window-edge end only (d given:
  the stated proxy for an undeclared theta_abserr), each with the resulting V_P, V_L + V_R and Q.

Writes OUT_JSON. Usage: PYTHONPATH=<tree>:<scripts> python -B leafprobe_07b.py REPLAY_JSON OUT_JSON --leaves I [I ...]
[--eighths] [--perturb D_HEX]
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import datetime
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent  # prompt 07b extension: Part A's scripts/ (variant_loader)
ROOT = HERE.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-436ed5d")
TREE_LQ_SHA = "4ea1948b0a45b5e3e7028c360d8b6a3165680f747391adfdb4783b6646ca2577"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def H(x):
    return None if x is None else float(x).hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay_json")
    ap.add_argument("out_json")
    ap.add_argument("--leaves", type=int, nargs="+", required=True)
    ap.add_argument("--eighths", action="store_true")
    ap.add_argument("--perturb", default=None)
    a = ap.parse_args()
    out_p = Path(a.out_json)
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    if os.environ.get("PYTHONPATH", "").split(":")[0] != str(TREE):
        raise SystemExit("PYTHONPATH must start with the tree")
    sys.path.insert(1, str(HERE))
    rep = json.loads(Path(a.replay_json).read_text())
    if not rep["reproduces_census"] and rep["variant"] in ("tree", "07"):
        raise SystemExit("the replay does not reproduce its census")
    variant = rep["variant"]
    if variant == "tree":
        import AdaptiveLevin.levin_quadrature as LQ

        if sha256(LQ.__file__) != TREE_LQ_SHA:
            raise SystemExit("the tree's module has another SHA-256")
        msha = TREE_LQ_SHA
    else:
        import variant_loader as VL

        LQ = VL.install(rep["header"]["module"], rep["header"]["module_sha256"], f"07b_{variant}")
        msha = rep["header"]["module_sha256"]
    import ray.cloudpickle as cp

    c = rep["header"]["capture"]
    with open(c["path"], "rb") as f:
        f.seek(c["offset"])
        blob = f.read(c["length"])
    if hashlib.sha256(blob).hexdigest() != c["sha256"]:
        raise SystemExit("capture blob has another SHA-256")
    rec = cp.loads(blob)
    kw = dict(rec["kw"])
    if c["override"]:
        # prompt 07b extension: the replay header stores the override as hex strings (replay_07b.py's jsonable)
        ov = {k: (float.fromhex(c["override"][k]) if isinstance(c["override"][k], str) else c["override"][k]) for k in ("rtol", "atol")}
        kw["rtol"], kw["atol"] = ov["rtol"], ov["atol"]
    f_ = rec["f"]
    theta = rec["theta"]
    order = kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
    rtol = kw["rtol"]

    def solver(th):
        A = LQ._Basis_SinCos(th)
        cache = {}

        def solve(span, force):
            d = LQ._adaptive_levin_subregion(span, f_, A, id_label="leafprobe-07b", chebyshev_order=order, rtol=rtol,
                                             notify_label=rec["notify_label"], vectorize_cache=cache, force_direct=force)
            return d

        return solve

    def piece(d, span):
        return {"lo": H(span[0]), "hi": H(span[1]), "value": H(d["value"]), "phase_err": H(d.get("phase_err")),
                "phase_err_less_node": H(d.get("phase_err_less_node")), "node_error_signed": H(d.get("node_error_signed")),
                "abserr_truncation": H(d.get("abserr_truncation")), "abserr_direct": H(d.get("abserr_direct")),
                "phase_span": H(d.get("phase_span")), "is_direct": bool(d.get("is_direct", False)),
                "lu": int(d["metadata"].get("direct_solve", 0)), "lstsq": int(d["metadata"].get("lstsq_solve", 0)),
                "guarded": bool(d.get("guarded", False))}

    def check(solve, a_, b_, force, eighths):
        c_ = (b_ + a_) / 2.0
        cL, cR = (c_ + a_) / 2.0, (b_ + c_) / 2.0
        spans = {"P": (a_, b_), "L": (a_, c_), "R": (c_, b_), "LL": (a_, cL), "LR": (cL, c_), "RL": (c_, cR), "RR": (cR, b_)}
        if eighths:
            for q in ("LL", "LR", "RL", "RR"):
                lo, hi = spans[q]
                m = (hi + lo) / 2.0
                spans[q + "L"], spans[q + "R"] = (lo, m), (m, hi)
        d = {k: solve(s, force) for k, s in spans.items()}
        VP = d["P"]["value"]
        S1 = d["L"]["value"] + d["R"]["value"]
        Q = (d["LL"]["value"] + d["LR"]["value"]) + (d["RL"]["value"] + d["RR"]["value"])
        out = {"spans": {k: [H(s[0]), H(s[1])] for k, s in spans.items()}, "pieces": {k: piece(d[k], spans[k]) for k in spans},
               "V_P": H(VP), "S1": H(S1), "Q": H(Q)}
        if eighths:
            S3 = sum((d[q + "L"]["value"] + d[q + "R"]["value"] for q in ("LL", "LR", "RL", "RR")), 0.0)
            out["S3"] = H(S3)
        delta1 = float(np.fabs(VP - S1))
        delta2 = float(np.fabs(S1 - Q))
        ratio = (0.0 if delta2 == 0.0 else float("inf")) if delta1 == 0.0 else delta2 / delta1
        nu = LQ._trust_noise_yardstick(d["P"].get("phase_err", 0.0) or 0.0, bool(d["P"].get("is_direct", False)), d["L"], d["R"])
        out.update(delta1=H(delta1), delta2=H(delta2), r=H(ratio), nu_B=H(nu),
                   clause=("r<1/4" if ratio < LQ._LEVIN_TRUST_MAX_RATIO else ("noise" if ratio < 1.0 and delta1 <= nu else "untrusted")))
        return out, d

    rep_leaves = rep["leaves"]
    doc_leaves = []
    t0 = time.perf_counter()
    for i in a.leaves:
        L = rep_leaves[i]
        a_, b_ = float.fromhex(L["start"]), float.fromhex(L["end"])
        force = bool(L["window"])
        solve = solver(theta)
        rec_, d = check(solve, a_, b_, force, a.eighths)
        is_direct = bool(d["P"].get("is_direct", False))
        resid = float.fromhex(L["abserr"]) if L["abserr"] else None
        dl1 = float.fromhex(rec_["delta1"])
        r_ = float.fromhex(rec_["r"])
        declared = (max(resid, dl1) / (1.0 - r_)) if (resid is not None and r_ < 1.0) else None
        repro = {"V_P": rec_["V_P"] == L["value_own"],
                 "Q": (rec_["Q"] == L["value"]) if L["value_kind"] == "quarters" else None,
                 "delta1": rec_["delta1"] == L["delta1"], "delta2": rec_["delta2"] == L["delta2"],
                 "r": rec_["r"] == L["convergence_ratio"],
                 "declared_resolution": (H(declared) == L["declared_resolution"]) if L["trusted"] else None,
                 "residual_is_delta1": (L["abserr"] == rec_["delta1"]) if not is_direct else None}
        ends = {"theta_prime_a": H(theta["theta_deriv"](a_)), "theta_prime_b": H(theta["theta_deriv"](b_)),
                "theta_a": H(theta["theta"](a_)), "theta_b": H(theta["theta"](b_)),
                "theta_mod_2pi_a": H(theta["theta_mod_2pi"](a_)), "theta_mod_2pi_b": H(theta["theta_mod_2pi"](b_))}
        entry = {"i": i, "leaf": L, "probe": rec_, "reproduces": all(v for v in repro.values() if v is not None),
                 "reproduces_detail": repro, "ends": ends}
        if a.perturb is not None:
            dd = float.fromhex(a.perturb)
            edge_end = "a" if L["edge_lo_is"] else ("b" if L["edge_hi_is"] else None)
            pert = {}
            for s in (+1, -1):
                th = dict(theta)
                g = theta["theta_deriv"]
                fac = 1.0 + s * LQ._CC_THETA_PRIME_REL_ERROR_SUPPLIED
                th["theta_deriv"] = (lambda g_, fac_: (lambda x: np.asarray(g_(x)) * fac_))(g, fac)
                r1, _ = check(solver(th), a_, b_, force, False)
                pert[f"theta_prime_{'+' if s > 0 else '-'}"] = {k: r1[k] for k in ("V_P", "S1", "Q", "delta1", "delta2", "r", "clause")}
                if edge_end is not None:
                    x_e = a_ if edge_end == "a" else b_
                    th2 = dict(theta)
                    m2 = theta["theta_mod_2pi"]

                    def m2p(x, m2_=m2, x_e_=x_e, sh=s * dd):
                        v = m2_(x)
                        return v + sh if x == x_e_ else v

                    th2["theta_mod_2pi"] = m2p
                    r2, _ = check(solver(th2), a_, b_, force, False)
                    pert[f"theta_mod_2pi_edge_{'+' if s > 0 else '-'}"] = {k: r2[k] for k in ("V_P", "S1", "Q", "delta1", "delta2", "r", "clause")}
            entry["perturb"] = {"edge_end": edge_end, "d": a.perturb, "theta_prime_rel": H(LQ._CC_THETA_PRIME_REL_ERROR_SUPPLIED),
                                "results": pert}
        doc_leaves.append(entry)
        print(f"leaf {i} [{a_:.6g}, {b_:.6g}] depth {L['depth']} {L['type']}: reproduces {entry['reproduces']} {repro}; "
              f"clause {rec_['clause']} r {float.fromhex(rec_['r']):.3g}", flush=True)
    hdr = {"title": f"levin-phase-floor 07b Part A: leaf probe of {rep['sid']} at {variant}",
           "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "host": socket.gethostname(),
           "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "module_sha256": msha,
           "replay_json": str(Path(a.replay_json).resolve()), "replay_json_sha256": sha256(a.replay_json), "capture": c,
           "args": vars(a)}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({"header": hdr, "sid": rep["sid"], "variant": variant, "wall_s": time.perf_counter() - t0,
                                 "leaves": doc_leaves}, indent=1))
    print("wrote", out_p, sha256(out_p))
    return 0 if all(e["reproduces"] for e in doc_leaves) else 1


if __name__ == "__main__":
    sys.exit(main())
