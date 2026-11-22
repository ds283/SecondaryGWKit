"""levin-phase-floor prompt 07b Part A, section 2.3: write the variants of the tree's AdaptiveLevin/levin_quadrature.py
(07a's code, 4ea1948b...), each by exact replacements (each old text must occur exactly once), with its diff against the
tree and its SHA-256 recorded in out/variants/variants.json. New in 07b (after 07a Part A's make_variants_07a.py).

Every variant adds the constant ``_VARIANT_07B = "<name>"`` after ``INTERVAL_TYPE_DIRECT = 1``. The variants and their
replacements are the module-level ``VARIANTS`` below; a variant already written with the same text is left, one with other
text is refused. Usage: python -B make_variants_07b.py NAME [NAME ...]
"""

import difflib
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE_LQ = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-436ed5d/AdaptiveLevin/levin_quadrature.py")
TREE_LQ_SHA = "4ea1948b0a45b5e3e7028c360d8b6a3165680f747391adfdb4783b6646ca2577"
VDIR = HERE / "variants"
ODIR = ROOT / "out" / "variants"
TAG = "INTERVAL_TYPE_DIRECT = 1\n"

VARIANTS = {
    # (b): a wider window, Phi_w = 4 pi and 8 pi; nothing else changes (the pieces stay at most 2 pi)
    "b4": [("_LEVIN_WINDOW_HALF_PHASE = 2.0 * np.pi\n",
            "_LEVIN_WINDOW_HALF_PHASE = 4.0 * np.pi  # 07b variant b4: Phi_w = 4 pi\n")],
    "b8": [("_LEVIN_WINDOW_HALF_PHASE = 2.0 * np.pi\n",
            "_LEVIN_WINDOW_HALF_PHASE = 8.0 * np.pi  # 07b variant b8: Phi_w = 8 pi\n")],
}


# The shared machinery of (c), (d) and (e): the window edges, an instrument, and a hook in _check_leaf. Each is a
# trust test added for a Levin leaf one of whose ends is a window's edge (lo or hi of a merged window of the location);
# it can only turn a trusted leaf untrusted (or, in (c), slow its declared ratio), so it changes nothing in a call with
# no window, and nothing on a leaf that does not touch one.
_EDGES = ("    num_evaluations = 0\n",
          "    num_evaluations = 0\n"
          "    # 07b variant: the window edges (lo and hi of each merged window), for the added trust test\n"
          "    _window_edges_07b = frozenset(\n"
          "        float(v) for w in (location[\"windows\"] if location is not None else []) for v in (w[\"lo\"], w[\"hi\"])\n"
          "    )\n")
_HOOK = ("        trusted = ratio < _LEVIN_TRUST_MAX_RATIO or (\n"
         "            ratio < 1.0 and delta1 <= noise_yardstick\n"
         "        )\n"
         "        _remove_error(leaf)\n",
         "        trusted = ratio < _LEVIN_TRUST_MAX_RATIO or (\n"
         "            ratio < 1.0 and delta1 <= noise_yardstick\n"
         "        )\n"
         "        # 07b variant: the added trust test for a Levin leaf that touches a window\n"
         "        if (\n"
         "            trusted\n"
         "            and not leaf.is_direct\n"
         "            and (leaf.start in _window_edges_07b or leaf.end in _window_edges_07b)\n"
         "        ):\n"
         "            trusted, ratio = _variant_test_07b(\n"
         "                leaf, ratio, delta1, delta2, residual, noise_yardstick, quarters_sum,\n"
         "                ((LL, LR), (RL, RR)),\n"
         "            )\n"
         "        _remove_error(leaf)\n")
_MEASURE = ("INTERVAL_TYPE_DIRECT = 1\n",
            "INTERVAL_TYPE_DIRECT = 1\n_MEASURE_07B = None  # 07b variant instrument: a list while set, one record per test\n")


def _defs(body):
    return ("    def _check_leaf(leaf: _levin_leaf) -> None:\n",
            body + "\n    def _check_leaf(leaf: _levin_leaf) -> None:\n")


# (c): trusted only on a check two levels down. The eighths are solved (each quarter's halves, by the expression of
# _levin_leaf.break_point), delta3 = |Q - S3| with S3 their sum, r3 = delta3 / delta2; the leaf stays trusted only if
# r3 passes the same test as r (r3 < 1/4, or r3 < 1 and delta2 <= nu_B), and then declares with max(r, r3). Basis: the
# declaration max(residual, delta1)/(1 - r) extrapolates a geometric fall from one observed ratio; (c) asks to see it at
# two consecutive levels before extrapolating, the cheapest check of section 0.2's (M2), and an error common to every
# level shows there as a delta3 that does not fall.
C_TEST = r"""    def _variant_test_07b(leaf, ratio, delta1, delta2, residual, nu, quarters_sum, gc):
        # 07b variant (c): a check two levels down for a Levin leaf touching a window
        forced = leaf.forced_direct
        a = leaf.start
        b = leaf.end
        c = leaf.break_point
        cL = (c + a) / 2.0
        cR = (b + c) / 2.0
        S3 = 0.0
        for lo, hi in ((a, cL), (cL, c), (c, cR), (cR, b)):
            m = (hi + lo) / 2.0
            S3 = S3 + (_solve((lo, m), "07b (c): eighth", force_direct=forced)["value"]
                       + _solve((m, hi), "07b (c): eighth", force_direct=forced)["value"])
        delta3 = float(np.fabs(quarters_sum - S3))
        if not (np.isfinite(delta3) and np.isfinite(delta2)):
            r3 = float("inf")
        elif delta2 == 0.0:
            r3 = 0.0 if delta3 == 0.0 else float("inf")
        else:
            r3 = delta3 / delta2
        ok = r3 < _LEVIN_TRUST_MAX_RATIO or (r3 < 1.0 and delta2 <= nu)
        if _MEASURE_07B is not None:
            _MEASURE_07B.append({"kind": "c", "start": a, "end": b, "depth": leaf.depth, "r": ratio, "r3": r3,
                                 "delta3": delta3, "S3": S3, "trusted": ok})
        return ok, (max(ratio, r3) if ok else ratio)
"""

# (d): an independent check. The leaf's own Levin antiderivative (its collocation solve on [a, b], re-solved with
# build_p_sample) is evaluated across the sub-span between the window's edge and the point s where the phase has moved
# _D_PHASE_07B beyond it (bisection on the raw phase), p(s) by barycentric interpolation on the leaf's Chebyshev-Lobatto
# nodes, so that V_sub = sum_j p_j(s) w_j(s) - p_j(edge) w_j(edge) (signed by orientation) with the p-modes the solve
# kept; the sub-span is also integrated by the nested Clenshaw-Curtis pair in pieces of at most 2 pi (the window pieces'
# phase). The leaf is untrusted if |V_sub - V_cc| exceeds the leaf's declared resolution plus floor plus the pieces'
# declared errors (abserr_direct + floor). Basis: prompt section 2.3 (d); 4 pi keeps each piece in the pair's regime
# (below 6 pi) and reaches past where section 2.2 finds the endpoint term's missing correction concentrated.
D_TEST = r"""    def _variant_test_07b(leaf, ratio, delta1, delta2, residual, nu, quarters_sum, gc):
        # 07b variant (d): an independent Clenshaw-Curtis check of the leaf's antiderivative beside the window
        a = leaf.start
        b = leaf.end
        lo_e, hi_e = min(a, b), max(a, b)
        at_lo = lo_e in _window_edges_07b
        edge = lo_e if at_lo else hi_e
        th = phase_fns[0]
        th_edge = float(th(edge))

        def moved(x):
            return abs(float(th(x)) - th_edge)

        far = hi_e if at_lo else lo_e
        if moved(far) <= _D_PHASE_07B:
            return True, ratio
        near_x, far_x = edge, far
        for _ in range(200):
            m = (near_x + far_x) / 2.0
            if m == near_x or m == far_x:
                break
            if moved(m) <= _D_PHASE_07B:
                near_x = m
            else:
                far_x = m
        s = near_x
        sub = (edge, s) if at_lo else (s, edge)
        full = _adaptive_levin_subregion(
            (lo_e, hi_e), f, BasisData, id_label=id_label,
            chebyshev_order=chebyshev_order, rtol=rtol, notify_label=notify_label,
            build_p_sample=True, diagnostics_path=resolved_diagnostics_path,
            vectorize_cache=vectorize_cache, force_direct=False,
            guard_rates=guard_rates, guard_sign_change_phases=guard_sign_change_phases,
        )
        xs = np.array([float(x) for x, _ in full["p_sample"]])
        P = np.array([[float(v) for v in ps] for _, ps in full["p_sample"]])  # nodes x components
        use = np.array([np.isfinite(r_) and r_ > rtol for r_ in full["p_ratios"]])
        n = len(xs)
        wb = np.array([(-1.0) ** j * (0.5 if j in (0, n - 1) else 1.0) for j in range(n)])

        def p_at(x):
            hit = np.nonzero(xs == x)[0]
            if hit.size:
                return P[hit[0]]
            t = wb / (x - xs)
            return (t @ P) / np.sum(t)

        def term(x):
            w = BasisData.eval_basis(x)
            px = p_at(x)
            return sum(px[j] * w[j] for j in range(len(w)) if use[j])

        V_sub = term(sub[1]) - term(sub[0])
        th_lo = float(th(sub[0]))
        n_pieces = int(np.ceil(_D_PHASE_07B / _LEVIN_WINDOW_PIECE_PHASE - 1e-12))
        cuts = [sub[0]]
        for k in range(1, n_pieces):
            target = k * _D_PHASE_07B / n_pieces
            x0, x1 = sub[0], sub[1]
            for _ in range(200):
                m = (x0 + x1) / 2.0
                if m == x0 or m == x1:
                    break
                if abs(float(th(m)) - th_lo) <= target:
                    x0 = m
                else:
                    x1 = m
            cuts.append(x0)
        cuts.append(sub[1])
        V_cc = 0.0
        e_cc = 0.0
        for p0, p1 in zip(cuts[:-1], cuts[1:]):
            d = _solve((p0, p1), "07b (d): Clenshaw-Curtis check piece", force_direct=True)
            V_cc = V_cc + d["value"]
            e_cc = e_cc + (d.get("abserr_direct") or 0.0) + (d.get("phase_err") or 0.0)
        e_leaf = max(residual, delta1) / (1.0 - ratio) + (leaf.region.phase_err or 0.0)
        ok = bool(abs(V_sub - V_cc) <= e_leaf + e_cc)
        if _MEASURE_07B is not None:
            _MEASURE_07B.append({"kind": "d", "start": a, "end": b, "depth": leaf.depth, "s": s, "V_sub": V_sub,
                                 "V_cc": V_cc, "e_cc": e_cc, "e_leaf": e_leaf, "trusted": ok,
                                 "V_full_check": term(hi_e) - term(lo_e), "V_full": full["value"]})
        return ok, ratio
"""

# (e): the window edge resolved. The leaf's Chebyshev-Lobatto grid (chebyshev_matrices, at the solve's working
# order) has its end node at the window's edge and one beside it; the leaf stays trusted only if every located phase's
# rate (the basis's closed-form rates) differs between those two nodes by at most a factor _E_RATIO_07B = 2. Basis
# (section 2.2): next to a stationary point the rate's own scale l = |theta'/theta''| at the window edge is about
# 2 Phi_w / |theta'|, and the endpoint term's correction, of relative size 1/(2 Phi_w), is missed while the grid's
# first spacing h is many l; a factor 2 is h about l. Bisection halves h, so an untrusted leaf is refined towards it.
E_TEST = r"""    def _variant_test_07b(leaf, ratio, delta1, delta2, residual, nu, quarters_sum, gc):
        # 07b variant (e): the window edge resolved by the leaf's own grid
        a = leaf.start
        b = leaf.end
        lo_e, hi_e = min(a, b), max(a, b)
        grid, _ = chebyshev_matrices((lo_e, hi_e), max(chebyshev_order, _LEVIN_MINIMUM_ALLOWED_ORDER))
        worst = 1.0
        for edge_x in (lo_e, hi_e):
            if edge_x not in _window_edges_07b:
                continue
            nxt = grid[-2] if edge_x == lo_e else grid[1]
            for rate in rate_fns:
                r0 = abs(float(rate(edge_x)))
                r1 = abs(float(rate(nxt)))
                q = max(r0, r1) / min(r0, r1) if min(r0, r1) > 0.0 else float("inf")
                worst = max(worst, q)
        ok = worst <= _E_RATIO_07B
        if _MEASURE_07B is not None:
            _MEASURE_07B.append({"kind": "e", "start": a, "end": b, "depth": leaf.depth, "rate_ratio": worst,
                                 "trusted": ok})
        return ok, ratio
"""

VARIANTS["c"] = [_MEASURE, _EDGES, _HOOK, _defs(C_TEST)]
VARIANTS["d"] = [_MEASURE, _EDGES, _HOOK, _defs(D_TEST),
                 ("_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n",
                  "_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n_D_PHASE_07B = 4.0 * np.pi  # 07b variant d: the check's sub-span phase\n")]
VARIANTS["e"] = [_MEASURE, _EDGES, _HOOK, _defs(E_TEST),
                 ("_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n",
                  "_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n_E_RATIO_07B = 2.0  # 07b variant e: largest rate ratio across the edge's first grid spacing\n")]


# (f): (e) made general, not tied to windows: every trusted Levin leaf, at both its ends, stays trusted only if the
# rate differs by at most _E_RATIO_07B = 2 across its grid's first spacing there. Basis (section 2.2): item 3's and item
# 4's carrying leaves touch no window; each ends at the call's own end, where the rate can vary on a scale its grid does
# not resolve, which is the window edge's condition without a window.
_HOOK_F = (_HOOK[0], _HOOK[1].replace(
    "            and (leaf.start in _window_edges_07b or leaf.end in _window_edges_07b)\n", "").replace(
    "        # 07b variant: the added trust test for a Levin leaf that touches a window\n",
    "        # 07b variant: the added trust test for every Levin leaf\n"))
F_TEST = E_TEST.replace("# 07b variant (e): the window edge resolved by the leaf's own grid",
                        "# 07b variant (f): both ends of every Levin leaf resolved by the leaf's own grid").replace(
    "            if edge_x not in _window_edges_07b:\n                continue\n", "")
VARIANTS["f"] = [_MEASURE, _EDGES, _HOOK_F, _defs(F_TEST),
                 ("_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n",
                  "_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n_E_RATIO_07B = 2.0  # 07b variant f: largest rate ratio across an end's first grid spacing\n")]


# (f), as written after section 2.2's tables (it replaces the first (f), (e)'s end rule made general, never run: on
# section 2.2's measured leaves the carriers without a window resolve their ends, rate ratio 1.02-1.06 across the first
# spacing, so that rule would not touch them). A trusted Levin leaf (every one, window or not) stays trusted only if its
# rate's range over its own nodes, max |theta'| / min |theta'| over chebyshev_matrices' grid at the solve's working order
# (the G1/G0 of _levin_G0_G1, unscaled), is at most _F_RANGE_07B = 10. Basis (section 2.2, out/mech/mech_all.md): of 88
# probed leaves with references, every one dishonest at 07a's code (11) spans a factor 49 or more, and none spanning 30
# or less is dishonest; across such a range the trust check's halves and quarters are pre-asymptotic at the slow end, or
# share the window edge's unresolved term. 10 is a decade, below the smallest dishonest range measured; bisection lowers a
# leaf's range, so an untrusted leaf is refined towards it.
F2_TEST = r"""    def _variant_test_07b(leaf, ratio, delta1, delta2, residual, nu, quarters_sum, gc):
        # 07b variant (f): the rate's range over the leaf's own nodes
        a = leaf.start
        b = leaf.end
        lo_e, hi_e = min(a, b), max(a, b)
        grid, _ = chebyshev_matrices((lo_e, hi_e), max(chebyshev_order, _LEVIN_MINIMUM_ALLOWED_ORDER))
        worst = 1.0
        for rate in (rate_fns or []):
            vals = np.abs(np.array([float(rate(x)) for x in grid]))
            q = float(np.max(vals) / np.min(vals)) if np.min(vals) > 0.0 else float("inf")
            worst = max(worst, q)
        ok = worst <= _F_RANGE_07B
        if _MEASURE_07B is not None:
            _MEASURE_07B.append({"kind": "f", "start": a, "end": b, "depth": leaf.depth, "rate_range": worst,
                                 "trusted": ok})
        return ok, ratio
"""
VARIANTS["f"] = [_MEASURE, _EDGES, _HOOK_F, _defs(F2_TEST),
                 ("_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n",
                  "_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n_F_RANGE_07B = 10.0  # 07b variant f: largest rate range over a trusted Levin leaf's nodes\n")]


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(name, src):
    out = src
    for old, new in VARIANTS[name]:
        if out.count(old) != 1:
            raise SystemExit(f"{name}: the text to replace occurs {out.count(old)} times: {old[:80]!r}")
        out = out.replace(old, new)
    if out.count(TAG) != 1:
        raise SystemExit("the tag line is not unique")
    return out.replace(TAG, TAG + f'_VARIANT_07B = "{name}"  # levin-phase-floor 07b Part A, section 2.3\n')


def main():
    if sha256(TREE_LQ) != TREE_LQ_SHA:
        raise SystemExit("the tree's module has another SHA-256")
    src = TREE_LQ.read_text()
    VDIR.mkdir(exist_ok=True)
    ODIR.mkdir(parents=True, exist_ok=True)
    vj_p = ODIR / "variants.json"
    vj = json.loads(vj_p.read_text()) if vj_p.exists() else {"tree": str(TREE_LQ), "tree_sha256": TREE_LQ_SHA, "variants": {}}
    for name in sys.argv[1:]:
        text = build(name, src)
        p = VDIR / f"levin_quadrature_07b_{name}.py"
        if p.exists() and p.read_text() != text:
            raise SystemExit(f"{p} exists with other text")
        p.write_text(text)
        diff = "".join(difflib.unified_diff(src.splitlines(True), text.splitlines(True), "tree/AdaptiveLevin/levin_quadrature.py",
                                            f"variants/{p.name}"))
        (ODIR / f"{name}.diff").write_text(diff)
        vj["variants"][name] = {"path": str(p), "sha256": sha256(p), "diff": str(ODIR / f"{name}.diff"),
                                "diff_sha256": sha256(ODIR / f"{name}.diff"), "maker_sha256": sha256(__file__)}
        print(name, sha256(p), f"diff {len(diff.splitlines())} lines")
    vj_p.write_text(json.dumps(vj, indent=1))
    print("wrote", vj_p, sha256(vj_p))


if __name__ == "__main__":
    main()
