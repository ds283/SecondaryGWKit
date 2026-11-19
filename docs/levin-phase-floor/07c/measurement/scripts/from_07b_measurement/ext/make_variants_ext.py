"""levin-phase-floor prompt 07b extension, section 2.7.2: write the extension's variants of the tree's
AdaptiveLevin/levin_quadrature.py (07a's code, 4ea1948b...), each by exact replacements (each old text must occur
exactly once), with its diff against the tree and its SHA-256 recorded in out/ext/variants/variants.json. New in the
extension, after Part A's make_variants_07b.py, whose shared pieces (the instrument, the window edges, the hook in
_check_leaf and (e)'s test) it imports unchanged.

Every variant adds ``_VARIANT_07B = "<name>"`` after ``INTERVAL_TYPE_DIRECT = 1``, as Part A's did. Usage:
python -B make_variants_ext.py NAME [NAME ...]

**(g) The adaptive window** (prompt section 2.7.2). Before the loop, after the location, each window's half-width is
doubled from 2 pi (_LEVIN_WINDOW_HALF_PHASE, unchanged) while the Levin seed leaf beside it fails (e)'s edge test, up to
_G_PHI_MAX_07B = 8 pi:
- **the leaf beside a window** is the seed leaf the driver makes between windows (or between a window and an end of
  x_span), the span ``_adaptive_levin``'s initial partition gives it. It is a Levin leaf if its phase variation V
  (``build_Levin_data`` on its grid at the working order, the C2 gate's quantity) is at least SIX_PI; one the gate sends
  to the Clenshaw-Curtis rule is not tested, as (e) does not test a direct leaf;
- **the edge test** is (e)'s: on that leaf's Chebyshev-Lobatto grid (``chebyshev_matrices`` at the working order),
  |theta'| of every located phase at the window-edge node and at its neighbour; the test fails if they differ by more
  than _G_RATIO_07B = 2 (``_E_RATIO_07B``'s value and basis);
- **doubling:** every located point of a failing window whose half-width is below the cap has it doubled (at most to
  the cap), and the location is re-run with those half-widths (``_locate_stationary_points`` with a per-point
  half-width; every other step of it unchanged, so windows that grow into one another merge and are cut into pieces of
  at most _LEVIN_WINDOW_PIECE_PHASE as the tree does). Repeated until no failing window can grow. Each point grows at
  most twice (2 pi, 4 pi, 8 pi), so the loop ends;
- **the cap** (prompt section 2.7.2): 8 pi is the widest half-width 07's calibration measured clean on family (R) at
  rtol 1e-9 (all ten cases honest and converged, abserr/request at most 0.881). 16 pi was never measured; at 32 pi the
  case c = 1e8, x_s = 0.7236, omega = 0 is not converged. Its basis reaches family (R) only: it says nothing measured of
  the census's windows beyond 8 pi, which is why the cap stops there;
- **at the cap:** (g1) trusts the leaf as the tree trusts it (the wider window alone, only where the edge test fails);
  (g2) applies (e)'s rule on the edges of the windows that reached the cap with the test still failing: a trusted Levin
  leaf touching such an edge whose rate differs by more than 2 across its grid's first spacing there is untrusted and
  bisected.
A call with no window, or whose every window passes at 2 pi, runs the tree's code: the location is the tree's, and only
the edge tests are added (their samples go to a fresh vectorize cache). The instrument records, per call, each window's
half-width reached and, at each final edge, the rate ratio, the first spacing over l = |theta'/theta''| (theta'' by a
central difference of the rate over 1e-3 of the first spacing) and whether the test still fails.
"""

import difflib
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import make_variants_07b as A  # noqa: E402  (Part A's maker; nothing of it runs on import)

ROOT = HERE.parent.parent
TREE_LQ, TREE_LQ_SHA, TAG = A.TREE_LQ, A.TREE_LQ_SHA, A.TAG
VDIR = HERE / "variants"
ODIR = ROOT / "out" / "ext" / "variants"

# --- (g): the location with a per-point half-width -------------------------------------------------------------------
_LOC = [
    ("    def window_edge(phase_index, x_p, t_p, direction):\n",
     "    def window_edge(phase_index, x_p, t_p, direction, half):\n"),
    ("            return abs(value - t_p) - window_half_phase\n",
     "            return abs(value - t_p) - half\n"),
    ("            if not (np.abs(theta_scan[j] - t_p) >= window_half_phase):\n",
     "            if not (np.abs(theta_scan[j] - t_p) >= half):\n"),
    ("        left = window_edge(p[\"phase\"], p[\"x\"], t_p, -1)\n        right = window_edge(p[\"phase\"], p[\"x\"], t_p, +1)\n",
     "        # 07b variant (g): a per-point half-width when window_half_phase is callable\n"
     "        half = window_half_phase(p) if callable(window_half_phase) else window_half_phase\n"
     "        left = window_edge(p[\"phase\"], p[\"x\"], t_p, -1, half)\n"
     "        right = window_edge(p[\"phase\"], p[\"x\"], t_p, +1, half)\n"),
]

_ADAPT_FN = r'''def _adapt_windows_07b(location, x_span, phase_fns, rate_fns, BasisData, chebyshev_order, vectorize_cache):
    """07b variant (g): each window's half-width doubled from _LEVIN_WINDOW_HALF_PHASE while the Levin seed leaf beside
    it fails (e)'s edge test, up to _G_PHI_MAX_07B (make_variants_ext.py's docstring)."""
    if location is None or not location["windows"]:
        return location
    base = _LEVIN_WINDOW_HALF_PHASE
    order = max(chebyshev_order, _LEVIN_MINIMUM_ALLOWED_ORDER)
    lo_span, hi_span = min(x_span[0], x_span[1]), max(x_span[0], x_span[1])
    halves = {}

    def half_of(p):
        return halves.get((p["phase"], p["x"]), base)

    def edge_tests(loc):
        gaps = []
        cursor, prev = lo_span, None
        for wi, w in enumerate(loc["windows"]):
            if w["lo"] > cursor:
                gaps.append((cursor, w["lo"], prev, wi))
            cursor = max(cursor, w["hi"])
            prev = wi
        if hi_span > cursor:
            gaps.append((cursor, hi_span, prev, None))
        tests = []
        for g_lo, g_hi, wl, wr in gaps:
            grid, Dmat = chebyshev_matrices((g_lo, g_hi), order)
            phase_span = float(BasisData.build_Levin_data(grid, Dmat, need_AmatT=False, vectorize_cache={})[4])
            levin = phase_span >= SIX_PI
            for wi, edge_x, nxt in ((wl, g_lo, float(grid[-2])), (wr, g_hi, float(grid[1]))):
                if wi is None:
                    continue
                worst, worst_k = 1.0, 0
                for k, rate in enumerate(rate_fns):
                    r0 = abs(float(rate(edge_x)))
                    r1 = abs(float(rate(nxt)))
                    q = max(r0, r1) / min(r0, r1) if min(r0, r1) > 0.0 else float("inf")
                    if q > worst:
                        worst, worst_k = q, k
                tests.append({"window": wi, "x": edge_x, "gap": (g_lo, g_hi), "levin": bool(levin),
                              "phase_span": phase_span, "rate_ratio": worst, "phase": worst_k,
                              "spacing": abs(nxt - edge_x), "fails": bool(levin and worst > _G_RATIO_07B)})
        return tests

    n_relocations = 0
    while True:
        tests = edge_tests(location)
        grow = set()
        for t in tests:
            if t["fails"]:
                for k in location["windows"][t["window"]]["points"]:
                    p = location["points"][k]
                    if half_of(p) < _G_PHI_MAX_07B:
                        grow.add((p["phase"], p["x"]))
        if not grow:
            break
        for key in grow:
            halves[key] = min(2.0 * halves.get(key, base), _G_PHI_MAX_07B)
        n_relocations += 1
        location = _locate_stationary_points(
            x_span,
            phase_fns,
            rate_fns,
            vectorize_cache,
            window_half_phase=half_of,
            coalescence_lambda=_LEVIN_COALESCENCE_LAMBDA,
            piece_phase=_LEVIN_WINDOW_PIECE_PHASE,
        )
    location["capped_failing_edges_07b"] = frozenset(t["x"] for t in tests if t["fails"])
    if _MEASURE_07B is not None:
        recs = []
        for t in tests:
            rate = _scalar_callable(rate_fns[t["phase"]])
            step = 1.0e-3 * t["spacing"]
            second = _first_difference(rate, t["x"], step, lo_span, hi_span) if step > 0.0 else float("nan")
            ell = abs(float(rate(t["x"]))) / abs(second) if second not in (0.0,) and np.isfinite(second) else float("inf")
            recs.append({"window": t["window"], "x": t["x"], "levin": t["levin"], "phase_span": t["phase_span"],
                         "rate_ratio": t["rate_ratio"], "spacing": t["spacing"], "ell": ell,
                         "spacing_over_l": t["spacing"] / ell if ell > 0.0 else float("inf"), "fails": t["fails"]})
        wins = []
        for wi, w in enumerate(location["windows"]):
            pts = [location["points"][k] for k in w["points"]]
            wins.append({"lo": w["lo"], "hi": w["hi"], "half_max": max(half_of(p) for p in pts),
                         "half_min": min(half_of(p) for p in pts), "n_pieces": len(w["cuts"]) - 1})
        _MEASURE_07B.append({"kind": "g", "n_relocations": n_relocations, "windows": wins, "edges": recs})
    return location


'''

_ADAPT_DEF = ("def _adaptive_levin(\n", _ADAPT_FN + "def _adaptive_levin(\n")
_ADAPT_CALL = (
    "            piece_phase=_LEVIN_WINDOW_PIECE_PHASE,\n        )\n\n    # For a basis of several phases, the nodal guard",
    "            piece_phase=_LEVIN_WINDOW_PIECE_PHASE,\n        )\n"
    "        # 07b variant (g): the adaptive window\n"
    "        location = _adapt_windows_07b(\n"
    "            location, x_span, phase_fns, rate_fns, BasisData, chebyshev_order, vectorize_cache\n"
    "        )\n\n    # For a basis of several phases, the nodal guard")
_G_CONST = ("_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n",
            "_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n"
            "_G_PHI_MAX_07B = 8.0 * np.pi  # 07b variant g: the adaptive window's cap\n"
            "_G_RATIO_07B = 2.0  # 07b variant g: (e)'s edge test, largest rate ratio across the edge's first grid spacing\n"
            "_E_RATIO_07B = _G_RATIO_07B  # 07b variant g2: (e)'s rule at the cap\n")
# (g2)'s edges: only those of windows that reached the cap with the edge test still failing
_EDGES_G2 = (A._EDGES[0],
             "    num_evaluations = 0\n"
             "    # 07b variant (g2): the edges of the windows that reached the cap with (e)'s edge test still failing\n"
             "    _window_edges_07b = frozenset(\n"
             "        location.get(\"capped_failing_edges_07b\", ()) if location is not None else ()\n"
             "    )\n")
_HOOK_G2 = (A._HOOK[0], A._HOOK[1].replace(
    "        # 07b variant: the added trust test for a Levin leaf that touches a window\n",
    "        # 07b variant (g2): (e)'s trust test for a Levin leaf that touches a capped window's failing edge\n"))
E_TEST_G2 = A.E_TEST.replace("# 07b variant (e): the window edge resolved by the leaf's own grid",
                             "# 07b variant (g2): (e)'s test at a window that reached the cap")

VARIANTS = {
    "g1": [A._MEASURE, _G_CONST] + _LOC + [_ADAPT_DEF, _ADAPT_CALL],
    "g2": [A._MEASURE, _G_CONST] + _LOC + [_ADAPT_DEF, _ADAPT_CALL, _EDGES_G2, _HOOK_G2, A._defs(E_TEST_G2)],
}

# **fdiag: an instrument, not a candidate** (X3). Part A's (f) with two records added to _MEASURE_07B and nothing else:
# at every leaf chosen for refinement, (i) where the round-off exit is considered (the leaf not a window or guarded leaf,
# its floor finite and its residual at most its floor), the premise's inputs and verdict; (ii) at every bisection, the
# leaf's span, depth, route, trust state, residual, floor and the trust check's delta1, delta2 and r. The records read
# state the driver already holds and compute nothing it does not; every value is checked equal to (f)'s by the replay.
_DIAG_EXIT = (
    "            if floor_limited:\n",
    "            if _MEASURE_07B is not None:  # 07b instrument fdiag: the round-off exit considered\n"
    "                _MEASURE_07B.append({\"kind\": \"exit\", \"start\": leaf.start, \"end\": leaf.end, \"depth\": leaf.depth,\n"
    "                                     \"direct\": bool(leaf.is_direct), \"trusted\": region._trusted,\n"
    "                                     \"abserr\": region.abserr, \"phase_err\": phase_err,\n"
    "                                     \"floor_roundoff\": region.phase_err_roundoff,\n"
    "                                     \"halves_floor_roundoff\": (_solve_floor_roundoff(leaf.dataL)\n"
    "                                                                + _solve_floor_roundoff(leaf.dataR))\n"
    "                                     if leaf.dataL is not None else None,\n"
    "                                     \"floor_limited\": bool(floor_limited)})\n"
    "            if floor_limited:\n")
_DIAG_BISECT = (
    "        # Bisect. Its halves become",
    "        if _MEASURE_07B is not None:  # 07b instrument fdiag: every bisection\n"
    "            _MEASURE_07B.append({\"kind\": \"bisect\", \"start\": leaf.start, \"end\": leaf.end, \"depth\": leaf.depth,\n"
    "                                 \"direct\": bool(leaf.is_direct), \"window\": bool(region.window),\n"
    "                                 \"guarded\": bool(leaf.guarded), \"trusted\": region._trusted,\n"
    "                                 \"abserr\": region.abserr, \"phase_err\": region.phase_err,\n"
    "                                 \"delta1\": region._delta1, \"delta2\": region._delta2,\n"
    "                                 \"r\": region._convergence_ratio, \"window_index\": leaf.window_index})\n"
    "        # Bisect. Its halves become")
VARIANTS["fdiag"] = list(A.VARIANTS["f"]) + [_DIAG_EXIT, _DIAG_BISECT]

# --- (h): (e) plus a rule for the leaves away from windows ------------------------------------------------------------
# (X2, out/ext/x2/x2_tables.md) found nothing the code computes that separates the six dishonest leaves away from windows
# from the honest trusted leaves of rate range above 30: the best, |Q(2N) - Q(N)|/e', has margin 0.76 (an honest leaf
# above the six's smallest), and a higher order does not remove the six's error. So, as the prompt directs, the rule is
# (f)'s rate range applied only to the trusted Levin leaves that touch no window: max|theta'|/min|theta'| over the
# leaf's own nodes at most _F_RANGE_07B = 10 (Part A's basis: every dishonest leaf of section 2.2's table spans 49.4 or
# more over its nodes, none at 30 or less is dishonest; 10 is a factor 4.9 below 49.4). A leaf touching a window gets
# (e)'s edge test instead, on the edges in _e_edges_07b (in (h) every window edge; in g2h only the capped failing
# edges). Its basis reaches only the leaves section 2.2 measured: 73 scored trusted leaves of 30 calls (V4, item 3, W2);
# it says nothing measured of the R1, R2 and R3 classes, which are the held-out set.
H_TEST = r"""    def _variant_test_07b(leaf, ratio, delta1, delta2, residual, nu, quarters_sum, gc):
        # 07b variant (h): (e)'s edge test on a Levin leaf touching a window; (f)'s rate range on one touching none
        a = leaf.start
        b = leaf.end
        lo_e, hi_e = min(a, b), max(a, b)
        grid, _ = chebyshev_matrices((lo_e, hi_e), max(chebyshev_order, _LEVIN_MINIMUM_ALLOWED_ORDER))
        worst = 1.0
        if lo_e in _window_edges_07b or hi_e in _window_edges_07b:
            kind = "h:e"
            for edge_x in (lo_e, hi_e):
                if edge_x not in _e_edges_07b:
                    continue
                nxt = grid[-2] if edge_x == lo_e else grid[1]
                for rate in rate_fns:
                    r0 = abs(float(rate(edge_x)))
                    r1 = abs(float(rate(nxt)))
                    q = max(r0, r1) / min(r0, r1) if min(r0, r1) > 0.0 else float("inf")
                    worst = max(worst, q)
            ok = worst <= _E_RATIO_07B
        else:
            kind = "h:f"
            for rate in (rate_fns or []):
                vals = np.abs(np.array([float(rate(x)) for x in grid]))
                q = float(np.max(vals) / np.min(vals)) if np.min(vals) > 0.0 else float("inf")
                worst = max(worst, q)
            ok = worst <= _F_RANGE_07B
        if _MEASURE_07B is not None:
            _MEASURE_07B.append({"kind": kind, "start": a, "end": b, "depth": leaf.depth, "ratio": worst, "trusted": ok})
        return ok, ratio
"""
_EDGES_H = (A._EDGES[0], A._EDGES[1] + "    _e_edges_07b = _window_edges_07b  # 07b variant (h): (e) on every window edge\n")
_EDGES_G2H = (A._EDGES[0], A._EDGES[1] + "    # 07b variant (g2h): (e) on the edges of the windows that reached the cap with the edge test still failing\n"
              "    _e_edges_07b = frozenset(\n"
              "        location.get(\"capped_failing_edges_07b\", ()) if location is not None else ()\n"
              "    )\n")
_HOOK_H = (A._HOOK[0], A._HOOK_F[1].replace("        # 07b variant: the added trust test for every Levin leaf\n",
                                            "        # 07b variant (h): the added trust test for every Levin leaf, by whether it touches a window\n"))
_H_CONST = ("_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n",
            "_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n"
            "_E_RATIO_07B = 2.0  # 07b variant h: (e)'s largest rate ratio across the edge's first grid spacing\n"
            "_F_RANGE_07B = 10.0  # 07b variant h: (f)'s largest rate range over a trusted Levin leaf's nodes, away from windows\n")
_G2H_CONST = ("_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n",
              "_LEVIN_WINDOW_PIECE_PHASE = 2.0 * np.pi\n"
              "_G_PHI_MAX_07B = 8.0 * np.pi  # 07b variant g2h: the adaptive window's cap\n"
              "_G_RATIO_07B = 2.0  # 07b variant g2h: (e)'s edge test, largest rate ratio across the edge's first grid spacing\n"
              "_E_RATIO_07B = _G_RATIO_07B  # 07b variant g2h: (e)'s rule at the cap\n"
              "_F_RANGE_07B = 10.0  # 07b variant g2h: (f)'s largest rate range over a trusted Levin leaf's nodes, away from windows\n")
VARIANTS["h"] = [A._MEASURE, _H_CONST, _EDGES_H, _HOOK_H, A._defs(H_TEST)]
VARIANTS["g2h"] = [A._MEASURE, _G2H_CONST] + _LOC + [_ADAPT_DEF, _ADAPT_CALL, _EDGES_G2H, _HOOK_H, A._defs(H_TEST)]

# --- (i): (f) with the chase addressed ---------------------------------------------------------------------------------
# (X3, out/ext/x3/x3_chase.md): every leaf at depth 20 in (f)'s 61 unconverged calls but two lies inside a Levin leaf
# (f)'s rule untrusted, almost all beside a window's edge; they are the C2 gate's fallback leaves (and some Levin leaves)
# whose trust check runs at the noise level (delta1 ~ 1e-43 against requests ~ 1e-33, r 0.3-8), untrusted, with residual
# above floor so the round-off exit is not considered, and not window leaves, so 07a's iroff1 stop never counts them.
# (i) gives each (f)-untrusted leaf's subtree a count of its own by 07a's rule unchanged (_window_bisection_counts:
# dqage's iroff1, _LEVIN_WINDOW_IROFF1_LIMIT = 6 counted bisections), and stops it as _stop_window stops a window
# (_floor_limit_stopped_window_leaf on every live leaf of the subtree). Basis: 07a's, for window leaves at the same noise
# level (07a section 1.2); it is extended here to the leaves (f) sends into the same regime. A leaf of a subtree is
# recognised by its seq (unique, never reused); its entry passes to its two children at its bisection.
_HOOK_I = (A._HOOK[0], A._HOOK_F[1].replace(
    "                ((LL, LR), (RL, RR)),\n            )\n",
    "                ((LL, LR), (RL, RR)),\n            )\n"
    "            if not trusted and leaf.seq not in _chase_group_07b:\n"
    "                # 07b variant (i): the untrusted leaf's subtree is counted by 07a's iroff1 rule, as a window is\n"
    "                _chase_group_07b[leaf.seq] = len(_chase_counts_07b)\n"
    "                _chase_counts_07b[_chase_group_07b[leaf.seq]] = 0\n"))
_CHASE_STATE = ("    window_counts = {}\n    windows_stopped = {}\n",
                "    window_counts = {}\n    windows_stopped = {}\n"
                "    # 07b variant (i): (f)-untrusted subtrees (leaf seq -> group), their iroff1 counts, the groups stopped\n"
                "    _chase_group_07b = {}\n    _chase_counts_07b = {}\n    _chase_stopped_07b = set()\n")
_CHASE_COUNT = ("        num_splits += 1\n",
                "        num_splits += 1\n\n"
                "        # 07b variant (i): count this bisection against its (f)-untrusted subtree by 07a's rule\n"
                "        stop_chase_07b = None\n"
                "        group_07b = _chase_group_07b.pop(leaf.seq, None)\n"
                "        if group_07b is not None:\n"
                "            _chase_group_07b[left.seq] = group_07b\n"
                "            _chase_group_07b[right.seq] = group_07b\n"
                "            if _window_bisection_counts(\n"
                "                region.value_own,\n"
                "                left.region.value_own + right.region.value_own,\n"
                "                _own_error(region),\n"
                "                _own_error(left.region),\n"
                "                _own_error(right.region),\n"
                "                _cc_leaf_unresolved(left.region.abserr, left.region.resasc),\n"
                "                _cc_leaf_unresolved(right.region.abserr, right.region.resasc),\n"
                "                rtol,\n"
                "            ):\n"
                "                _chase_counts_07b[group_07b] += 1\n"
                "                if (\n"
                "                    _chase_counts_07b[group_07b] >= _LEVIN_WINDOW_IROFF1_LIMIT\n"
                "                    and group_07b not in _chase_stopped_07b\n"
                "                ):\n"
                "                    stop_chase_07b = group_07b\n")
_CHASE_STOP = ("        if stop_window is not None:\n            _stop_window(stop_window)\n",
               "        if stop_window is not None:\n            _stop_window(stop_window)\n"
               "        if stop_chase_07b is not None:  # 07b variant (i): the subtree stopped as a window is\n"
               "            _chase_stopped_07b.add(stop_chase_07b)\n"
               "            members_07b = [m for m in leaves.values() if _chase_group_07b.get(m.seq) == stop_chase_07b]\n"
               "            if _MEASURE_07B is not None:\n"
               "                _MEASURE_07B.append({\"kind\": \"i_stop\", \"group\": stop_chase_07b, \"split\": num_splits,\n"
               "                                     \"members\": len(members_07b),\n"
               "                                     \"max_depth\": max(m.depth for m in members_07b)})\n"
               "            for member in members_07b:\n"
               "                _floor_limit_stopped_window_leaf(member)\n")
_fv = A.VARIANTS["f"]
VARIANTS["i"] = [_fv[0], _fv[1], _HOOK_I, _fv[3], _fv[4], _CHASE_STATE, _CHASE_COUNT, _CHASE_STOP]

# --- (i2): (i) restricted to the subtree's fallback leaves ---------------------------------------------------------------
# (i)'s replays (out/ext/tables/scores_ext_v1.md) fail on W2-2-14345-3 (1.55), R1-x241-7 (2.44) and W2-0-12977-5
# (unconverged, 1.32): in each, (f) untrusts the root seed leaf, the whole call is one subtree, 07a's count reaches 6
# after 9-11 bisections of Levin leaves that are not at the noise level, and the stop floor-limits them at depth 7 or
# less, returning V_P with the r-based declaration (f) exists to refuse. X3 finds the chase in the C2 gate's fallback
# leaves (3961 of the 4372 leaves at depth 20 descending from an (f)-untrusted leaf). (i2) counts only bisections of a
# fallback (direct) leaf of a subtree, and stops only the subtree's fallback leaves; its Levin leaves refine as under (f).
_CHASE_COUNT_I2 = (_CHASE_COUNT[0], _CHASE_COUNT[1].replace(
    "            if _window_bisection_counts(\n                region.value_own,\n",
    "            if leaf.is_direct and _window_bisection_counts(  # 07b variant (i2): fallback leaves only\n"
    "                region.value_own,\n"))
_CHASE_STOP_I2 = (_CHASE_STOP[0], _CHASE_STOP[1].replace(
    "if _chase_group_07b.get(m.seq) == stop_chase_07b]",
    "if _chase_group_07b.get(m.seq) == stop_chase_07b and m.is_direct]"))
VARIANTS["i2"] = [_fv[0], _fv[1], _HOOK_I, _fv[3], _fv[4], _CHASE_STATE, _CHASE_COUNT_I2, _CHASE_STOP_I2]


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
    return out.replace(TAG, TAG + f'_VARIANT_07B = "{name}"  # levin-phase-floor 07b extension, section 2.7.2\n')


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
        compile(text, name, "exec")
        p = VDIR / f"levin_quadrature_07b_{name}.py"
        if p.exists() and p.read_text() != text:
            raise SystemExit(f"{p} exists with other text")
        p.write_text(text)
        diff = "".join(difflib.unified_diff(src.splitlines(True), text.splitlines(True),
                                            "tree/AdaptiveLevin/levin_quadrature.py", f"ext/variants/{p.name}"))
        (ODIR / f"{name}.diff").write_text(diff)
        vj["variants"][name] = {"path": str(p), "sha256": sha256(p), "diff": str(ODIR / f"{name}.diff"),
                                "diff_sha256": sha256(ODIR / f"{name}.diff"), "maker_sha256": sha256(__file__)}
        print(name, sha256(p), f"diff {len(diff.splitlines())} lines")
    vj_p.write_text(json.dumps(vj, indent=1))
    print("wrote", vj_p, sha256(vj_p))


if __name__ == "__main__":
    main()
