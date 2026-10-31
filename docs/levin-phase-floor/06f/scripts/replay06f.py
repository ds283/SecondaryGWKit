"""levin-phase-floor prompt 06f sections 1-3: replay one captured call at the tree on PYTHONPATH, with
pass-throughs at four module attributes, and measure M1's and M2's quantities on it. New in 06f.

The capture is x1_capture.py's (06e's, run at g): the call's span, integrand f, phase dict theta and
keyword arguments as the census worker saw them, loaded with ray.cloudpickle (the census scripts
must be on PYTHONPATH after the tree: the group calls' pickles name harness functions). The replay
is ``AdaptiveLevin.levin_quadrature.adaptive_levin_sincos(x_span, f, theta=theta, **kw)``, as 06e's
replay_g.py replays a capture, in this one process, with these pass-throughs installed for the call
only and restored in a finally (each returns exactly what the module's own function returns):
- ``_adaptive_levin_subregion`` (the attribute _adaptive_levin's _solve calls): opens a record for
  the span being solved and files it, with the solve's value, phase_err, abserr_direct, branch and
  order, under (a, b) when the solve returns. Every solve is kept: the returned leaves' own solves
  and the trial solves (comparison halves, premise halves, trust-check quarters);
- ``_cc_node_term_parts``: its returned (E, R_g, R_2, R_c), E signed;
- ``_cc_roundoff_floor``: its node_term argument and its return, and its first two terms from the
  module's own function on its own inputs, as 06e's worker.py's _cc_floor_terms reads them (with
  the arguments zeroed it returns (n + 4) u S alone, with the basis values zeroed u sum|w||f||a|
  alone);
- ``_declared_endpoint_phase_err``: its return, and its two endpoint terms apart:
  _phase_error(theta_abserr_at(a), p_a) and _phase_error(theta_abserr_at(b), p_b), each from the
  module's own functions; the two are checked to add to the return in the code's order.
With --m2 the Clenshaw-Curtis solves also keep their weights, amplitudes and basis arguments.

Each returned leaf is matched to its own solve by its span; a leaf is reproduced if its solve's
phase_err (and on the fallback branch its abserr_direct) equals the leaf's bit for bit, and the
captured terms rebuild the floor in the code's order. Gate F0's first two clauses are checked here:
the replay against 06e's census record of the call (value, abserr, regions, maximum depth,
converged), and the summed terms over the returned Clenshaw-Curtis leaves against the checkpoint's
``cc_floor`` sums (both over all and over floor-attributed leaves, math.fsum, bit for bit).

Then M1 (prompt section 2) on every call, and with --m2 M2 (section 3); see m1() and m2(). Writes
OUT_DIR/<unit>_<index>/result.json (everything but the per-leaf and per-node tables), leaves.json.gz
(per returned leaf), solves_cc.json.gz (Z2's call: every Clenshaw-Curtis solve's E, for the depth
analysis) and, with --m2, nodes.json.gz (per node of item 1). A digest of the per-leaf table is
printed for the determinism clause of F0.

Usage: PYTHONPATH=<tree g>:<scripts> python -B replay06f.py CAPTURE_JSON UNIT INDEX OUT_DIR [--m2] [--tag T]
"""

import argparse
import collections
import gzip
import hashlib
import inspect
import json
import math
import os
import random
import socket
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT06E_SHA256, RUN06E, TREES, sha256, tree_state  # noqa: E402

CC_SUM_TERMS = (
    "t_value_rounding",
    "t_argument",
    "node_term",
    "R_g",
    "R_2",
    "R_c",
    "declared",
    "phase_err_resolve",
)
MP_DPS = 30
DENSE_N = 1000
DENSE_SEED = 20261031


def fx(x):
    return None if x is None else {"hex": float(x).hex(), "repr": repr(float(x))}


def H(x):
    return None if x is None else float.fromhex(x["hex"] if isinstance(x, dict) else x)


def jdump(path, obj):
    data = json.dumps(obj, indent=None, separators=(",", ":"), sort_keys=True).encode()
    if str(path).endswith(".gz"):
        with gzip.GzipFile(path, "wb", mtime=0) as f:
            f.write(data)
    else:
        Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def census_call(uid, index):
    with open(RUN06E / "checkpoint.jsonl") as f:
        for line in f:
            if f'"unit": "{uid}"' not in line[:200]:
                continue
            r = json.loads(line)
            if r["unit"] != uid:
                continue
            for c in (r["data"].get("calls") or []) + (
                r["data"].get("other_calls") or []
            ):
                if c["index"] == index:
                    return c
    return None


def load_capture(capture_json, uid, index):
    import ray.cloudpickle as cp

    ents = [
        e
        for e in json.loads(Path(capture_json).read_text())
        if e["unit"] == uid and e["index"] == index
    ]
    if len(ents) != 1:
        raise SystemExit(f"{uid} {index}: {len(ents)} capture entries")
    e = ents[0]
    if not e.get("capture_matches_census"):
        raise SystemExit(f"{uid} {index}: its capture does not match the census")
    with open(Path(e["dir"]) / e["pickle"], "rb") as f:
        f.seek(e["offset"])
        blob = f.read(e["length"])
    if hashlib.sha256(blob).hexdigest() != e["sha256"]:
        raise SystemExit(f"{uid} {index}: capture blob sha256 differs")
    return cp.loads(blob), e


# ---------------------------------------------------------------------------------------------
# the replay


def replay(LQ, rec, keep_arrays):
    solves = collections.defaultdict(list)
    stack = []
    orig = {
        n: getattr(LQ, n)
        for n in (
            "_adaptive_levin_subregion",
            "_cc_node_term_parts",
            "_cc_roundoff_floor",
            "_declared_endpoint_phase_err",
        )
    }
    sig_parts = inspect.signature(orig["_cc_node_term_parts"])
    sig_floor = inspect.signature(orig["_cc_roundoff_floor"])

    def w_sub(span, *a, **k):
        cur = {"n_parts": 0, "n_floor": 0, "n_decl": 0}
        stack.append(cur)
        try:
            d = orig["_adaptive_levin_subregion"](span, *a, **k)
        finally:
            stack.pop()
        cur.update(
            value=float(d["value"]),
            phase_err=float(d["phase_err"]),
            is_direct=bool(d.get("is_direct", False)),
            abserr_direct=(
                None if d.get("abserr_direct") is None else float(d["abserr_direct"])
            ),
            abserr_truncation=(
                None
                if d.get("abserr_truncation") is None
                else float(d["abserr_truncation"])
            ),
            order=d["metadata"].get("chebyshev_order"),
        )
        solves[(float(span[0]), float(span[1]))].append(cur)
        return d

    def w_parts(*a, **k):
        r = orig["_cc_node_term_parts"](*a, **k)
        if stack:
            cur = stack[-1]
            cur["n_parts"] += 1
            cur["E"], cur["R_g"], cur["R_2"], cur["R_c"] = (float(x) for x in r)
        return r

    def w_floor(*a, **k):
        r = orig["_cc_roundoff_floor"](*a, **k)
        if stack:
            cur = stack[-1]
            cur["n_floor"] += 1
            ba = sig_floor.bind(*a, **k)
            w_, f_, b_, g_ = (
                ba.arguments[n]
                for n in ("weights", "f_sample", "basis_sample", "argument_sample")
            )
            cur["t_value_rounding"] = float(
                orig["_cc_roundoff_floor"](w_, f_, b_, np.zeros_like(g_), 0.0)
            )
            cur["t_argument"] = float(
                orig["_cc_roundoff_floor"](w_, f_, np.zeros_like(b_), g_, 0.0)
            )
            cur["node_term"] = float(ba.arguments["node_term"])
            cur["floor_returned"] = float(r)
            if keep_arrays:
                cur["weights"] = np.array(w_, dtype=float)
                cur["f_sample"] = np.array(f_, dtype=float)
                cur["argument_sample"] = np.array(g_, dtype=float)
        return r

    def w_decl(BasisData, a, b, p_a, p_b):
        r = orig["_declared_endpoint_phase_err"](BasisData, a, b, p_a, p_b)
        if stack:
            cur = stack[-1]
            cur["n_decl"] += 1
            ta = BasisData.theta_abserr_at(a)
            tb = BasisData.theta_abserr_at(b)
            da = LQ._phase_error(ta, p_a) if ta is not None else 0.0
            db = LQ._phase_error(tb, p_b) if tb is not None else 0.0
            cur.update(
                declared=float(r),
                declared_a=float(da),
                declared_b=float(db),
                theta_abserr_a=ta,
                theta_abserr_b=tb,
                p_a=float(p_a),
                p_b=float(p_b),
                declared_adds=float((0.0 + da) + db).hex() == float(r).hex(),
            )
        return r

    LQ._adaptive_levin_subregion = w_sub
    LQ._cc_node_term_parts = w_parts
    LQ._cc_roundoff_floor = w_floor
    LQ._declared_endpoint_phase_err = w_decl
    try:
        t0 = time.perf_counter()
        rr = LQ.adaptive_levin_sincos(
            tuple(rec["x_span"]), rec["f"], theta=rec["theta"], **rec["kw"]
        )
        wall = time.perf_counter() - t0
    finally:
        for n, fn in orig.items():
            setattr(LQ, n, fn)
    for n, fn in orig.items():
        assert getattr(LQ, n) is fn
    return rr, solves, wall


def leaf_table(rr, solves):
    """One record per returned leaf, in ascending x, matched to its own solve."""
    leaves = []
    problems = []
    for i, R in enumerate(rr["regions"]):
        key = (float(R.start), float(R.end))
        recs = solves.get(key) or []
        if not recs:
            problems.append(f"leaf {i}: no solve of its span")
            continue
        s0 = recs[0]
        if any(
            (s["value"], s["phase_err"]) != (s0["value"], s0["phase_err"])
            for s in recs[1:]
        ):
            problems.append(
                f"leaf {i}: its span was solved {len(recs)} times with different results"
            )
        cc = R.type != "Levin"
        dec = R.declared_resolution
        L = {
            "i": i,
            "branch": "cc" if cc else "levin",
            "a": fx(R.start),
            "b": fx(R.end),
            "depth": int(R.depth),
            "value": fx(s0["value"]),
            "abserr": fx(R.abserr),
            "phase_err": fx(R.phase_err),
            "declared_resolution": fx(dec),
            "abserr_truncation": fx(R.abserr_truncation),
            "total_err": fx(R.total_err),
            "floor_limited": bool(R.phase_limited),
            "trusted": R.trusted,
            "order": s0["order"],
            "n_solves_of_span": len(recs),
            "declared": fx(s0.get("declared")),
            "declared_a": fx(s0.get("declared_a")),
            "declared_b": fx(s0.get("declared_b")),
            "theta_abserr_a": fx(s0.get("theta_abserr_a")),
            "theta_abserr_b": fx(s0.get("theta_abserr_b")),
            "p_a": fx(s0.get("p_a")),
            "p_b": fx(s0.get("p_b")),
        }
        ok = (
            float(s0["phase_err"]).hex() == float(R.phase_err).hex()
            and s0.get("n_decl", 0) >= 1
            and s0.get("declared_adds", False)
        )
        if cc:
            for n in (
                "E",
                "R_g",
                "R_2",
                "R_c",
                "t_value_rounding",
                "t_argument",
                "node_term",
                "floor_returned",
            ):
                L[n] = fx(s0.get(n))
            ok = ok and s0["n_parts"] == 1 and s0["n_floor"] == 1 and s0["n_decl"] == 1
            ok = ok and float(s0["abserr_direct"]).hex() == float(R.abserr).hex()
            ok = (
                ok
                and float(
                    s0["t_value_rounding"] + s0["t_argument"] + s0["node_term"]
                ).hex()
                == float(s0["floor_returned"]).hex()
            )
            ok = (
                ok
                and float(abs(s0["E"]) + s0["R_g"] + s0["R_2"] + s0["R_c"]).hex()
                == float(s0["node_term"]).hex()
            )
            ok = (
                ok
                and float(s0["floor_returned"] + s0["declared"]).hex()
                == float(s0["phase_err"]).hex()
            )
        # floor-attributed, as 06e's worker.py attributes it
        resol = dec if dec is not None else R.abserr
        L["floor_attributed"] = bool(
            resol is None or (R.phase_err is not None and resol < R.phase_err)
        )
        L["reproduced"] = bool(ok)
        if not ok:
            problems.append(f"leaf {i}: not reproduced")
        leaves.append(L)
    return leaves, problems


# ---------------------------------------------------------------------------------------------
# F0


def f0_checks(rr, leaves, census, cap_entry):
    out = {}
    got = {
        "value": float(rr["value"]).hex(),
        "abserr": float(rr["abserr"]).hex(),
        "num_regions": int(rr["num_regions"]),
        "max_depth": int(max(R.depth for R in rr["regions"])),
        "converged": bool(rr["converged"]),
    }
    want = {
        "value": census["value"]["hex"],
        "abserr": census["abserr"]["hex"],
        "num_regions": census["num_regions"],
        "max_depth": census["max_depth"],
        "converged": bool(census["converged"]),
    }
    out["replay"] = got
    out["census"] = want
    out["replay_equals_census"] = got == want
    out["capture_equals_census"] = bool(cap_entry.get("capture_matches_census"))
    # the leaves sum to the call's value and abserr, as _sum_leaves_ascending sums them
    v = 0.0
    e = 0.0
    for L in leaves:
        v = v + H(L["value"])
        if L["total_err"] is not None:
            e = e + H(L["total_err"])
    out["leaves_sum_to_value_and_abserr"] = (
        float(v).hex() == got["value"] and float(e).hex() == got["abserr"]
    )
    cc = [L for L in leaves if L["branch"] == "cc"]
    fl = census.get("cc_floor") or {}

    def sums(sel):
        s = {}
        for n in CC_SUM_TERMS:
            src = "phase_err" if n == "phase_err_resolve" else n
            s[n] = float(math.fsum(H(L[src]) for L in sel)).hex()
        s["abs_E"] = float(math.fsum(abs(H(L["E"])) for L in sel)).hex()
        return s

    mine = {
        "sums_all_cc_leaves": sums(cc),
        "sums_floor_attributed_cc_leaves": sums(
            [L for L in cc if L["floor_attributed"]]
        ),
    }
    terms_ok = True
    detail = {}
    for k in ("sums_all_cc_leaves", "sums_floor_attributed_cc_leaves"):
        cs = fl.get(k) or {}
        for n, hx in mine[k].items():
            want_hx = (cs.get(n) or {}).get("hex")
            same = want_hx == hx
            terms_ok &= same
            detail[f"{k}.{n}"] = {"replay": hx, "census": want_hx, "equal": same}
    out["n_cc_leaves"] = {"replay": len(cc), "census": fl.get("n_cc_leaves")}
    out["n_floor_attributed"] = {
        "replay": sum(1 for L in cc if L["floor_attributed"]),
        "census": fl.get("n_floor_attributed"),
    }
    terms_ok &= out["n_cc_leaves"]["replay"] == out["n_cc_leaves"]["census"]
    terms_ok &= (
        out["n_floor_attributed"]["replay"] == out["n_floor_attributed"]["census"]
    )
    out["terms_equal_census_sums"] = bool(terms_ok)
    out["terms_detail"] = detail
    out["every_leaf_reproduced"] = all(L["reproduced"] for L in leaves)
    out["holds"] = bool(
        out["replay_equals_census"]
        and terms_ok
        and out["every_leaf_reproduced"]
        and out["leaves_sum_to_value_and_abserr"]
    )
    return out


# ---------------------------------------------------------------------------------------------
# M1


def tree_of(x_span, leaf_spans):
    """The returned partition as a binary tree from the root, splitting at (a + b)/2.0 as
    _adaptive_levin's break_point does. Returns {span: (depth, parent span or None, children or None)}.
    """
    nodes = {}
    todo = [((float(x_span[0]), float(x_span[1])), 0, None)]
    while todo:
        span, d, parent = todo.pop()
        if span in leaf_spans:
            nodes[span] = (d, parent, None)
            continue
        if d > 64:
            raise RuntimeError("partition does not resolve into the returned leaves")
        a, b = span
        c = (b + a) / 2.0
        kids = ((a, c), (c, b))
        nodes[span] = (d, parent, kids)
        for k in kids:
            todo.append((k, d + 1, span))
    return nodes


def runs_test(signs):
    n = len(signs)
    if n < 2:
        return {"n": n}
    runs = 1 + sum(1 for i in range(1, n) if signs[i] != signs[i - 1])
    npos = sum(1 for s in signs if s > 0)
    nneg = n - npos
    longest = cur = 1
    for i in range(1, n):
        cur = cur + 1 if signs[i] == signs[i - 1] else 1
        longest = max(longest, cur)
    out = {
        "n": n,
        "n_positive": npos,
        "n_negative": nneg,
        "runs": runs,
        "longest_run": longest,
    }
    if npos and nneg:
        mu = 1 + 2.0 * npos * nneg / n
        var = (
            2.0 * npos * nneg * (2.0 * npos * nneg - n) / (n * n * (n - 1))
            if n > 1
            else 0.0
        )
        out["runs_expected_if_random"] = mu
        out["runs_z"] = (runs - mu) / math.sqrt(var) if var > 0 else None
    return out


def m1(rec, rr, leaves, solves, keep_solves):
    atol = rec["kw"]["atol"]
    rtol = rec["kw"]["rtol"]
    v = float(rr["value"])
    request = max(atol, rtol * abs(v))
    cc = [L for L in leaves if L["branch"] == "cc"]
    E = [H(L["E"]) for L in cc]
    N = len(cc)
    sum_absE = math.fsum(abs(x) for x in E)
    abs_sumE = abs(math.fsum(E))
    sumR = math.fsum(H(L["R_g"]) + H(L["R_2"]) + H(L["R_c"]) for L in cc)
    rms = math.sqrt(math.fsum(x * x for x in E) / N) if N else 0.0

    # the call's error with each Clenshaw-Curtis leaf's |E| taken out of its floor, every other term unchanged
    def e_without_absE(L):
        res = (
            H(L["declared_resolution"])
            if L["declared_resolution"] is not None
            else H(L["abserr"])
        )
        fl_noE = (
            H(L["t_value_rounding"])
            + H(L["t_argument"])
            + (H(L["R_g"]) + H(L["R_2"]) + H(L["R_c"]))
        ) + H(L["declared"])
        base = fl_noE if res is None else max(res, fl_noE)
        tr = H(L["abserr_truncation"])
        return base if tr is None else base + tr

    e_rest = []
    for L in leaves:
        if L["branch"] == "cc":
            e_rest.append(e_without_absE(L))
        else:
            e_rest.append(H(L["total_err"]) if L["total_err"] is not None else 0.0)
    rest = 0.0
    for x in e_rest:
        rest = rest + x
    signed_B = rest + abs_sumE
    signed_A = float(rr["abserr"]) - sum_absE + abs_sumE
    # siblings: both children of one parent returned as Clenshaw-Curtis leaves
    by_span = {(H(L["a"]), H(L["b"])): L for L in leaves}
    tree = tree_of(rec["x_span"], set(by_span))
    paired = set()
    pair_terms = []
    for span, (d, parent, kids) in tree.items():
        if kids is None:
            continue
        l, r = by_span.get(kids[0]), by_span.get(kids[1])
        if (
            l is not None
            and r is not None
            and l["branch"] == "cc"
            and r["branch"] == "cc"
        ):
            pair_terms.append(abs(H(l["E"]) + H(r["E"])))
            paired.add(kids[0])
            paired.add(kids[1])
    unpaired = [abs(H(L["E"])) for L in cc if (H(L["a"]), H(L["b"])) not in paired]
    local = math.fsum(pair_terms) + math.fsum(unpaired)
    signed_pairs = rest + local
    cc_sorted = sorted(cc, key=lambda L: H(L["a"]))
    signs = [1 if H(L["E"]) > 0 else (-1 if H(L["E"]) < 0 else 0) for L in cc_sorted]
    out = {
        "N_cc": N,
        "sum_absE": sum_absE,
        "abs_sumE": abs_sumE,
        "sumE": math.fsum(E),
        "ratio_sum_absE_over_abs_sumE": (sum_absE / abs_sumE) if abs_sumE > 0 else None,
        "sqrt_N_cc": math.sqrt(N),
        "sum_R": sumR,
        "rms_E": rms,
        "abs_sumE_over_sqrtN_rms": (
            abs_sumE / (math.sqrt(N) * rms) if N and rms > 0 else None
        ),
        "request": request,
        "declared_error": float(rr["abserr"]),
        "value": v,
        "atol": atol,
        "rtol": rtol,
        "converged": bool(rr["converged"]),
        "signed_combination_error": signed_B,
        "signed_combination_error_linear": signed_A,
        "rest_of_error_without_absE": rest,
        "n_sibling_pairs": len(pair_terms),
        "n_unpaired_cc": len(unpaired),
        "sum_local_pair_absE": local,
        "signed_combination_error_sibling_pairs": signed_pairs,
        "signs_along_call": runs_test([s for s in signs if s != 0]),
        "n_E_zero": sum(1 for s in signs if s == 0),
        "sign_sequence": (
            "".join("+" if s > 0 else ("-" if s < 0 else "0") for s in signs)
            if N <= 20000
            else None
        ),
        "n_floor_limited_cc": sum(1 for L in cc if L["floor_limited"]),
        "max_depth": int(max(L["depth"] for L in leaves)),
    }
    # by depth, and each Clenshaw-Curtis subtree against its root's own |E| (from the trial solves)
    by_depth = collections.defaultdict(list)
    for L in cc:
        by_depth[L["depth"]].append(H(L["E"]))
    out["by_depth"] = {
        str(d): {
            "n": len(es),
            "sum_absE": math.fsum(abs(x) for x in es),
            "abs_sumE": abs(math.fsum(es)),
        }
        for d, es in sorted(by_depth.items())
    }
    leaf_of = {}
    for span, (d, parent, kids) in tree.items():
        if kids is None:
            leaf_of[span] = by_span[span]
    sub = {}

    def walk(span):
        d, parent, kids = tree[span]
        if kids is None:
            L = leaf_of[span]
            if L["branch"] == "cc":
                return [H(L["E"])], True
            return [], False
        e1, c1 = walk(kids[0])
        e2, c2 = walk(kids[1])
        es = e1 + e2
        allcc = c1 and c2
        sub[span] = (d, es, allcc)
        return es, allcc

    walk((float(rec["x_span"][0]), float(rec["x_span"][1])))
    rows = []
    for span, (d, es, allcc) in sub.items():
        recs = solves.get(span) or []
        if not recs or not allcc or "E" not in recs[0]:
            continue
        E0 = abs(recs[0]["E"])
        if E0 == 0:
            continue
        rows.append(
            {
                "root_depth": d,
                "n_leaves": len(es),
                "root_absE": E0,
                "sub_sum_absE": math.fsum(abs(x) for x in es),
                "sub_abs_sumE": abs(math.fsum(es)),
            }
        )
    per_d = collections.defaultdict(list)
    for r in rows:
        per_d[r["root_depth"]].append(r)
    out["subtrees_vs_root"] = {
        str(d): {
            "n_subtrees": len(rs),
            "median_n_leaves": float(np.median([r["n_leaves"] for r in rs])),
            "median_sum_absE_over_root_absE": float(
                np.median([r["sub_sum_absE"] / r["root_absE"] for r in rs])
            ),
            "p10_sum_absE_over_root_absE": float(
                np.percentile([r["sub_sum_absE"] / r["root_absE"] for r in rs], 10)
            ),
            "p90_sum_absE_over_root_absE": float(
                np.percentile([r["sub_sum_absE"] / r["root_absE"] for r in rs], 90)
            ),
            "median_abs_sumE_over_root_absE": float(
                np.median([r["sub_abs_sumE"] / r["root_absE"] for r in rs])
            ),
        }
        for d, rs in sorted(per_d.items())
    }
    out["n_subtrees_compared"] = len(rows)
    solves_cc = None
    if keep_solves:
        solves_cc = [
            {"a": fx(a), "b": fx(b), "E": fx(s[0]["E"]), "n": len(s)}
            for (a, b), s in solves.items()
            if "E" in s[0]
        ]
        out["subtree_rows"] = rows
    return out, solves_cc


# ---------------------------------------------------------------------------------------------
# M2


def mp_group_phase(group, eta_hat, mp):
    """The group phase sum_i s_i theta_nu_i(c_i eta_hat), each theta = atan2(J, -Y) at 30 digits, at
    the double eta_hat (the argument production evaluates at), the products c_i eta_hat exact.
    """
    tot = mp.mpf(0)
    for phase, c, s in zip(group.phases, group.coefficients, group.signs):
        y = mp.mpf(c) * mp.mpf(float(eta_hat))
        J = mp.besselj(mp.mpf(phase.nu), y)
        Y = mp.bessely(mp.mpf(phase.nu), y)
        tot += mp.mpf(s) * mp.atan2(J, -Y)
    return tot


def wrap(mp, d):
    two_pi = 2 * mp.pi
    return d - two_pi * mp.nint(d / two_pi)


def delta_theta(group, u, mp):
    """production theta_mod_2pi(u) minus the reference, reduced into (-pi, pi]."""
    th = group.theta_mod_2pi(u)
    eta_hat = np.exp(u)
    ref = mp_group_phase(group, eta_hat, mp)
    return float(wrap(mp, mp.mpf(float(th)) - ref)), float(th)


def gl_integral(fun, a, b, n):
    xg, wg = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    us = mid + half * xg
    vals = np.array([fun(float(u)) for u in us])
    return math.fsum((half * wg * vals).tolist())


def m2(LQ, rec, rr, leaves, solves, mp):
    theta = rec["theta"]
    group = theta["theta_mod_2pi"].__self__
    f0, f1 = rec["f"]
    kw = rec["kw"]
    N = kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
    rtol = kw["rtol"]
    atol = kw["atol"]
    v = float(rr["value"])
    request = max(atol, rtol * abs(v))
    out = {
        "group": {
            "coefficients": list(group.coefficients),
            "signs": list(group.signs),
            "K": group.K,
            "C": group.C,
            "nu": [p.nu for p in group.phases],
            "x_star": [p.x_star for p in group.phases],
            "has_near": [p._corrections.near is not None for p in group.phases],
            "theta_abserr_scalar": [p.theta_abserr for p in group.phases],
        },
        "request": request,
        "declared_error": float(rr["abserr"]),
        "value": v,
        "atol": atol,
        "rtol": rtol,
    }
    a0, b0 = (float(x) for x in rec["x_span"])
    out["total_phase_abs"] = abs(float(group.theta(b0)) - float(group.theta(a0)))
    # kinks of theta_abserr in u = log eta: where c_i eta = x_star_i for a constituent with a near region
    kinks = sorted(
        math.log(p.x_star) - math.log(c)
        for p, c in zip(group.phases, group.coefficients)
        if p._corrections.near is not None
    )
    out["theta_abserr_kinks_u"] = kinks
    node_rows = []
    per_leaf = []
    bas = LQ._Basis_SinCos(theta)
    for L in leaves:
        a, b = H(L["a"]), H(L["b"])
        s0 = solves[(a, b)][0]
        if L["branch"] == "cc":
            nodes, _ = LQ.chebyshev_matrices((a, b), 2 * N - 1)
            w = s0["weights"]
            fs = s0["f_sample"]
            args = s0["argument_sample"]
        else:
            nodes, _ = LQ.chebyshev_matrices((a, b), s0["order"])
            w = fs = args = None
        # item 1: the measured phase error at every node of the leaf
        dth = []
        for k, u in enumerate(nodes):
            d, th = delta_theta(group, float(u), mp)
            ta = float(group.theta_abserr(float(u)))
            if args is not None and float(th).hex() != float(args[0][k]).hex():
                raise RuntimeError(
                    f"leaf {L['i']} node {k}: theta_mod_2pi differs from the solve's argument"
                )
            dth.append(d)
            node_rows.append(
                {
                    "leaf": L["i"],
                    "k": k,
                    "u": float(u).hex(),
                    "theta_mod_2pi": float(th).hex(),
                    "delta_theta": d,
                    "theta_abserr": ta,
                    "ratio": abs(d) / ta if ta > 0 else None,
                }
            )
        rowL = {
            "i": L["i"],
            "branch": L["branch"],
            "a": a,
            "b": b,
            "depth": L["depth"],
            "n_nodes": len(nodes),
            "declared": H(L["declared"]),
            "declared_a": H(L["declared_a"]),
            "declared_b": H(L["declared_b"]),
            "theta_abserr_a": H(L["theta_abserr_a"]),
            "theta_abserr_b": H(L["theta_abserr_b"]),
            "proxy_or_p_a": H(L["p_a"]),
            "proxy_or_p_b": H(L["p_b"]),
        }
        # item 4 (the rule's first-order response to the measured errors at the points it evaluates
        # theta) and the honest per-leaf integral of |f|
        if L["branch"] == "cc":
            g_prime = fs[0] * np.cos(args[0]) - fs[1] * np.sin(args[0])
            rowL["dI_measured"] = math.fsum((w * g_prime * np.array(dth)).tolist())
            ta_nodes = np.array([float(group.theta_abserr(float(u))) for u in nodes])
            rowL["dI_declared_rule_plus"] = math.fsum((w * g_prime * ta_nodes).tolist())
            fabs = np.sqrt(fs[0] ** 2 + fs[1] ** 2)
            rowL["int_abs_f_honest"] = math.fsum((np.fabs(w) * fabs).tolist())
            rowL["declared_honest"] = (
                0.5
                * rowL["int_abs_f_honest"]
                * (H(L["theta_abserr_a"]) + H(L["theta_abserr_b"]))
            )
            rowL["f_scale_width"] = float(np.max(fabs)) * abs(b - a)
        else:
            vc = {}
            d2 = LQ._adaptive_levin_subregion(
                (a, b),
                rec["f"],
                bas,
                id_label=LQ._LazyUUID(),
                chebyshev_order=N,
                rtol=rtol,
                notify_label=kw.get("notify_label"),
                build_p_sample=True,
                vectorize_cache=vc,
            )
            if float(d2["value"]).hex() != L["value"]["hex"]:
                raise RuntimeError(
                    f"leaf {L['i']}: the p re-solve's value differs from the leaf's"
                )
            P_b = np.array(d2["p_sample"][0][1])
            P_a = np.array(d2["p_sample"][-1][1])
            use = np.array([np.isfinite(r) and r > rtol for r in d2["p_ratios"]])
            tha, thb = float(theta["theta_mod_2pi"](a)), float(
                theta["theta_mod_2pi"](b)
            )
            bp_b = np.array([math.cos(thb), -math.sin(thb)])
            bp_a = np.array([math.cos(tha), -math.sin(tha)])
            # endpoint nodes: the grid is descending, node 0 is b and the last is a
            rowL["dI_measured"] = float(
                np.sum(np.where(use, P_b * bp_b, 0.0)) * dth[0]
                - np.sum(np.where(use, P_a * bp_a, 0.0)) * dth[-1]
            )
            rowL["dI_declared_rule_plus"] = float(
                np.sum(np.where(use, P_b * bp_b, 0.0)) * float(group.theta_abserr(b))
                - np.sum(np.where(use, P_a * bp_a, 0.0)) * float(group.theta_abserr(a))
            )
            rowL["declared_honest"] = H(L["declared"])
        rowL["max_abs_delta_theta"] = float(np.max(np.abs(dth)))
        per_leaf.append(rowL)

    # item 3: the response to delta theta = s * theta_abserr(x), per leaf by Gauss-Legendre, kinks split, doubled
    def g_of(s):
        def g(u):
            th = theta["theta_mod_2pi"](u)
            return (
                s
                * float(group.theta_abserr(u))
                * (f0(u) * math.cos(th) - f1(u) * math.sin(th))
            )

        return g

    conv = []
    for rowL in per_leaf:
        a, b = rowL["a"], rowL["b"]
        cuts = [a] + [k for k in kinks if a < k < b] + [b]
        tv = 0.0
        us = np.linspace(a, b, 257)
        tp = np.array([abs(float(theta["theta_deriv"](float(u)))) for u in us])
        tv = float(np.trapezoid(tp, us))
        n = max(32, 2 * int(math.ceil(tv / math.pi)) + 16)
        hist = []
        for s in (+1, -1):
            g = g_of(s)
            n_s = n
            prev = None
            while True:
                val = math.fsum(
                    gl_integral(g, cuts[j], cuts[j + 1], n_s)
                    for j in range(len(cuts) - 1)
                )
                if s == +1:
                    hist.append((n_s, val))
                if prev is not None and abs(val - prev) <= 1e-6 * abs(val):
                    break
                if n_s >= 1 << 15:
                    break
                prev = val
                n_s *= 2
            rowL[f"dI_{'plus' if s > 0 else 'minus'}"] = val
            rowL[f"dI_{'plus' if s > 0 else 'minus'}_previous"] = prev
            rowL[f"dI_{'plus' if s > 0 else 'minus'}_n"] = n_s
        rowL["phase_total_variation"] = tv
        rowL["n_kink_cuts"] = len(cuts) - 2
        rowL["dI_history_plus"] = hist
        conv.append(
            abs(rowL["dI_plus"] - rowL["dI_plus_previous"]) / abs(rowL["dI_plus"])
            if rowL["dI_plus"]
            else None
        )
    D = math.fsum(r["declared"] for r in per_leaf)
    dIp = math.fsum(r["dI_plus"] for r in per_leaf)
    dIm = math.fsum(r["dI_minus"] for r in per_leaf)
    dIp_prev = math.fsum(r["dI_plus_previous"] for r in per_leaf)
    dI_meas = math.fsum(r["dI_measured"] for r in per_leaf)
    sum_abs_leaf = math.fsum(abs(r["dI_plus"]) for r in per_leaf)
    D_honest = math.fsum(r["declared_honest"] for r in per_leaf)
    maxs = max(abs(dIp), abs(dIm))
    allnodes = np.array([r["delta_theta"] for r in node_rows])
    ratios = np.array([r["ratio"] for r in node_rows if r["ratio"] is not None])
    out.update(
        {
            "n_leaves": len(per_leaf),
            "n_cc": sum(1 for r in per_leaf if r["branch"] == "cc"),
            "D": D,
            "D_cc": math.fsum(r["declared"] for r in per_leaf if r["branch"] == "cc"),
            "D_levin": math.fsum(
                r["declared"] for r in per_leaf if r["branch"] == "levin"
            ),
            "dI_plus": dIp,
            "dI_minus": dIm,
            "dI_plus_at_half_resolution": dIp_prev,
            "max_s_abs_dI": maxs,
            "dI_converged_rel": abs(dIp - dIp_prev) / abs(dIp) if dIp else None,
            "sum_abs_dI_leaf": sum_abs_leaf,
            "D_over_max_abs_dI": D / maxs if maxs > 0 else None,
            "sum_abs_dI_leaf_over_max_abs_dI": (
                sum_abs_leaf / maxs if maxs > 0 else None
            ),
            "dI_declared_rule_plus": math.fsum(
                r["dI_declared_rule_plus"] for r in per_leaf
            ),
            "dI_measured": dI_meas,
            "item4_computed": True,
            "D_honest": D_honest,
            "D_honest_over_max_abs_dI": D_honest / maxs if maxs > 0 else None,
            "D_over_D_honest": D / D_honest if D_honest > 0 else None,
            "n_nodes": len(node_rows),
            "max_abs_delta_theta": float(np.max(np.abs(allnodes))),
            "median_abs_delta_theta": float(np.median(np.abs(allnodes))),
            "max_ratio_delta_theta_over_theta_abserr": float(np.max(ratios)),
            "median_ratio": float(np.median(ratios)),
            "n_nodes_violating": int(np.sum(ratios > 1.0)),
            "theta_abserr_range": [
                float(min(r["theta_abserr"] for r in node_rows)),
                float(max(r["theta_abserr"] for r in node_rows)),
            ],
            "per_leaf": per_leaf,
        }
    )
    # descriptive: a seeded uniform sample of DENSE_N points over the call's span
    rng = random.Random(DENSE_SEED)  # the same seed for every call
    pts = sorted(rng.uniform(a0, b0) for _ in range(DENSE_N))
    dd, rr_ = [], []
    for u in pts:
        d, _th = delta_theta(group, u, mp)
        ta = float(group.theta_abserr(u))
        dd.append(abs(d))
        rr_.append(abs(d) / ta)
    out["dense"] = {
        "n": DENSE_N,
        "seed": DENSE_SEED,
        "max_abs_delta_theta": max(dd),
        "median_abs_delta_theta": float(np.median(dd)),
        "max_ratio": max(rr_),
        "median_ratio": float(np.median(rr_)),
        "n_violating": sum(1 for x in rr_ if x > 1.0),
    }
    return out, node_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture_json")
    ap.add_argument("unit")
    ap.add_argument("index", type=int)
    ap.add_argument("out_dir")
    ap.add_argument("--m2", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import AdaptiveLevin.levin_quadrature as LQ

    tree = LQ.__file__.split("/AdaptiveLevin/")[0]
    head, status = tree_state(tree)
    hdr = {
        "title": "levin-phase-floor 06f: replay and M1/M2 of one captured call",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "script": "replay06f.py",
        "script_sha256": sha256(Path(__file__).resolve()),
        "prov_sha256": sha256(HERE / "prov.py"),
        "tree": tree,
        "tree_head": head,
        "tree_status_short": status,
        "tree_expected": list(map(str, TREES["g"])),
        "levin_quadrature_sha256": sha256(LQ.__file__),
        "checkpoint": str(RUN06E / "checkpoint.jsonl"),
        "checkpoint_sha256_expected": CHECKPOINT06E_SHA256,
        "capture_json": args.capture_json,
        "capture_json_sha256": sha256(args.capture_json),
        "unit": args.unit,
        "index": args.index,
        "m2": args.m2,
        "tag": args.tag,
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "dont_write_bytecode": sys.dont_write_bytecode,
    }
    if tree != str(TREES["g"][0]) or head != TREES["g"][1] or status.strip():
        raise SystemExit(f"tree evidence does not hold: {hdr}")
    rec, cap = load_capture(args.capture_json, args.unit, args.index)
    hdr["capture_blob_sha256"] = cap["sha256"]
    hdr["capture_pickle"] = str(Path(cap["dir"]) / cap["pickle"])
    census = census_call(args.unit, args.index)
    if census is None or census["notify_label"] != rec["notify_label"]:
        raise SystemExit("census call not found, or its label differs")
    odir = Path(args.out_dir) / f"{args.unit}_{args.index}{args.tag}"
    odir.mkdir(parents=True, exist_ok=False)
    rr, solves, wall = replay(LQ, rec, keep_arrays=args.m2)
    leaves, problems = leaf_table(rr, solves)
    f0 = f0_checks(rr, leaves, census, cap)
    f0["leaf_problems"] = problems[:50]
    leaves_sha = jdump(odir / "leaves.json.gz", leaves)
    result = {
        "header": hdr,
        "notify_label": rec["notify_label"],
        "x_span": [fx(x) for x in rec["x_span"]],
        "kw": rec["kw"],
        "wall_s": wall,
        "n_solves": sum(len(v) for v in solves.values()),
        "F0": f0,
        "leaves_sha256": leaves_sha,
    }
    result["digest"] = hashlib.sha256(
        (float(rr["value"]).hex() + float(rr["abserr"]).hex() + leaves_sha).encode()
    ).hexdigest()
    try:
        m1_out, solves_cc = m1(
            rec,
            rr,
            leaves,
            solves,
            keep_solves=(args.unit, args.index) == ("g-1e-8-0-33995", 5),
        )
        result["M1"] = m1_out
        if solves_cc is not None:
            result["solves_cc_sha256"] = jdump(odir / "solves_cc.json.gz", solves_cc)
    except Exception:
        result["M1_error"] = traceback.format_exc()
    if args.m2:
        try:
            import mpmath as mp

            mp.mp.dps = MP_DPS
            m2_out, node_rows = m2(LQ, rec, rr, leaves, solves, mp)
            result["M2"] = m2_out
            result["nodes_sha256"] = jdump(odir / "nodes.json.gz", node_rows)
        except Exception:
            result["M2_error"] = traceback.format_exc()
    (odir / "result.json").write_text(json.dumps(result, indent=1, default=float))
    print(
        json.dumps(
            {
                "unit": args.unit,
                "index": args.index,
                "tag": args.tag,
                "F0_holds": f0["holds"],
                "replay_equals_census": f0["replay_equals_census"],
                "terms_equal": f0["terms_equal_census_sums"],
                "wall_s": round(wall, 2),
                "digest": result["digest"],
                "M1_error": "M1_error" in result,
                "M2_error": "M2_error" in result,
            }
        ),
        flush=True,
    )
    return (
        0
        if f0["holds"] and "M1_error" not in result and "M2_error" not in result
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
