"""levin-phase-floor prompt 06k section 1 ("The variants"): writes the identity variant and the three
route variants of the tree's ``AdaptiveLevin/levin_quadrature.py`` into ``variants/``. New in 06k.

Each route variant is the tree's file with exact text replacements, every one inside
``_adaptive_levin`` (checked afterwards by ``ast_check_variants.py``), each asserted to match its
original exactly once. The identity variant is the tree's file copied byte for byte. Nothing in the
tree is written. The routes are prompt 06k section 0.2's, as fixed in
state/predictions_and_rules_20261104T0915Z.md section 3:

- route 1: the noise clause compares delta1 with nu1 = (phi_P + phi_L) + phi_R;
- route 2: where 06g's rule (b) finds a leaf untrusted, rule (b) is evaluated on the values corrected
  by -E (each of the seven solves' node_error_signed, 0 where it has none); if that trusts, the leaf
  is trusted, declares max(residual, delta1c) / (1 - r_c) and returns Q (uncorrected);
- route 3: at each stop test the untrusted refinable leaves are ranked by D, and the longest prefix
  whose D sum is at most eta = 1e-3 of the request is left (declares D, not refined, does not block
  the stop or convergence).

Every route counts its firings (a leaf trusted or left that 06g's rule finds untrusted) and reports
them per call under the new result key ``route_06k``.

Usage: python -B make_variants.py TREE_FILE OUT_DIR
"""

import hashlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------------- common edits
COUNTERS_OLD = """    num_untrusted_found = 0

    def _solve(span, description):
"""


def counters_new(route):
    return f"""    num_untrusted_found = 0
    # prompt 06k, route {route}: every firing of the route in this call (a leaf trusted or left that
    # 06g's rule (b) finds untrusted), reported under the result key "route_06k"
    route_firings_06k = []

    def _solve(span, description):
"""


RETURN_OLD = """        "chebyshev_min_order": chebyshev_min_order,
        "max_depth": max_depth,
    }
"""


def return_new(extra):
    return (
        """        "chebyshev_min_order": chebyshev_min_order,
        "max_depth": max_depth,
        # prompt 06k: the route's firings in this call (a new key; the comparison with 06g ignores it)
        "route_06k": """
        + extra
        + """,
    }
"""
    )


# ---------------------------------------------------------------------------------- the trust test
TRUST_OLD = """        residual = region.abserr
        trusted = ratio < _LEVIN_TRUST_MAX_RATIO or (
            ratio < 1.0 and delta1 <= region.phase_err
        )
"""

TRUST_NEW_ROUTE1 = """        residual = region.abserr
        trusted_g_06k = ratio < _LEVIN_TRUST_MAX_RATIO or (
            ratio < 1.0 and delta1 <= region.phase_err
        )
        # prompt 06k, route 1: the noise yardstick covers the three solves delta1 is a difference
        # of, nu1 = phi_P + phi_L + phi_R, each the solve's own whole floor ("phase_err", |E| included)
        nu1_06k = (region.phase_err + (dataL.get("phase_err", 0.0) or 0.0)) + (
            dataR.get("phase_err", 0.0) or 0.0
        )
        trusted = ratio < _LEVIN_TRUST_MAX_RATIO or (ratio < 1.0 and delta1 <= nu1_06k)
        if trusted and not trusted_g_06k:
            route_firings_06k.append(
                {
                    "depth": int(leaf.depth),
                    "at_depth_max": bool(leaf.depth >= depth_max),
                    "start": float(a),
                    "end": float(b),
                    "is_direct": bool(leaf.is_direct),
                    "r": float(ratio),
                    "delta1": float(delta1),
                    "delta2": float(delta2),
                    "phi_P": float(region.phase_err),
                    "phi_L": float(dataL.get("phase_err", 0.0) or 0.0),
                    "phi_R": float(dataR.get("phase_err", 0.0) or 0.0),
                    "nu1": float(nu1_06k),
                    "residual": float(residual),
                }
            )
"""

TRUST_NEW_ROUTE2 = """        residual = region.abserr
        trusted = ratio < _LEVIN_TRUST_MAX_RATIO or (
            ratio < 1.0 and delta1 <= region.phase_err
        )
        # prompt 06k, route 2: where 06g's rule (b) finds the leaf untrusted, rule (b) as written on
        # the differences of the seven solves' values each corrected by -E (its node_error_signed; a
        # solve without one, by 0). If that trusts, the leaf is trusted and declares
        # max(residual, delta1c) / (1 - r_c); it returns Q, uncorrected, as any trusted leaf.
        declared_06k = None
        if not trusted:

            def _node_06k(solve_data):
                node = solve_data.get("node_error_signed")
                return 0.0 if node is None else node

            node_P_06k = region.node_error_signed
            own_c_06k = own_value - (0.0 if node_P_06k is None else node_P_06k)
            halves_c_06k = (dataL["value"] - _node_06k(dataL)) + (
                dataR["value"] - _node_06k(dataR)
            )
            quarters_c_06k = (
                (LL["value"] - _node_06k(LL)) + (LR["value"] - _node_06k(LR))
            ) + ((RL["value"] - _node_06k(RL)) + (RR["value"] - _node_06k(RR)))
            delta1c_06k = float(np.fabs(own_c_06k - halves_c_06k))
            delta2c_06k = float(np.fabs(halves_c_06k - quarters_c_06k))
            if not (np.isfinite(delta1c_06k) and np.isfinite(delta2c_06k)):
                ratio_c_06k = float("inf")
            elif delta1c_06k == 0.0:
                ratio_c_06k = 0.0 if delta2c_06k == 0.0 else float("inf")
            else:
                ratio_c_06k = delta2c_06k / delta1c_06k
            if ratio_c_06k < _LEVIN_TRUST_MAX_RATIO or (
                ratio_c_06k < 1.0 and delta1c_06k <= region.phase_err
            ):
                trusted = True
                declared_06k = max(residual, delta1c_06k) / (1.0 - ratio_c_06k)
                route_firings_06k.append(
                    {
                        "depth": int(leaf.depth),
                        "at_depth_max": bool(leaf.depth >= depth_max),
                        "start": float(a),
                        "end": float(b),
                        "is_direct": bool(leaf.is_direct),
                        "r": float(ratio),
                        "delta1": float(delta1),
                        "delta2": float(delta2),
                        "phi_P": float(region.phase_err),
                        "delta1c": float(delta1c_06k),
                        "delta2c": float(delta2c_06k),
                        "r_c": float(ratio_c_06k),
                        "residual": float(residual),
                        "declared": float(declared_06k),
                        "declared_g_formula": float(
                            max(residual, delta1) / (1.0 - ratio)
                            if ratio < 1.0
                            else max(residual, delta1 + delta2)
                        ),
                    }
                )
"""

DECLARE_OLD = """        if trusted:
            region._trusted = True
            region._declared_resolution = max(residual, delta1) / (1.0 - ratio)
"""

DECLARE_NEW_ROUTE2 = """        if trusted:
            region._trusted = True
            region._declared_resolution = (
                max(residual, delta1) / (1.0 - ratio)
                if declared_06k is None
                else declared_06k  # prompt 06k, route 2: a firing declares on the corrected numbers
            )
"""

ROUTE12_RETURN = """{
            "route": "%s",
            "n_firings": len(route_firings_06k),
            "firings": route_firings_06k,
        }"""

# ---------------------------------------------------------------------------------- route 3
R3_STATE_OLD = """    leaves = {}
    candidates = []
    untrusted_refinable = set()
"""

R3_STATE_NEW = """    leaves = {}
    candidates = []
    untrusted_refinable = set()
    # prompt 06k, route 3: the leaves left at the last stop test (by sequence number), and eta. A left
    # leaf is untrusted and refinable, declares D, is not a candidate and does not block the stop.
    left_06k = set()
    route3_eta_06k = 1e-3
    route3_unleft_06k = [0]
"""

R3_PUSH_OLD = """    def _push_candidate(leaf: _levin_leaf) -> None:
        leaf.version += 1
        if _refinable(leaf):
"""

R3_PUSH_NEW = """    def _push_candidate(leaf: _levin_leaf) -> None:
        leaf.version += 1
        if leaf.seq in left_06k:  # prompt 06k, route 3: a left leaf is not a candidate
            untrusted_refinable.discard(leaf.seq)
            return
        if _refinable(leaf):
"""

R3_HELPERS_OLD = """    _add_leaf(_make_leaf(x_span[0], x_span[1], 0, None, {}, {}, {}))

    running_stop = False
"""

R3_HELPERS_NEW = """    def _declaration_06k(leaf: _levin_leaf) -> float:
        # prompt 06k, route 3: D, what the code declares today for an untrusted leaf at depth_max
        region = leaf.region
        ratio = region._convergence_ratio
        residual = region.abserr
        delta1 = region._delta1
        delta2 = region._delta2
        return (
            max(residual, delta1) / (1.0 - ratio)
            if ratio < 1.0
            else max(residual, delta1 + delta2)
        )

    def _relevel_06k(request: float) -> bool:
        # prompt 06k, route 3: at a stop test, rank every untrusted refinable leaf (those left before
        # included) by D, smallest first (ties: start, then sequence number), and leave the longest
        # prefix whose D sum (correctly rounded) is at most eta * request. Returns True if the left
        # set changed (the caller then re-tests the budget, which now carries the new D).
        pool_06k = [leaves[seq] for seq in (untrusted_refinable | left_06k)]
        ranked_06k = sorted(
            pool_06k,
            key=lambda leaf: (_error_for_ordering(_declaration_06k(leaf)), leaf.start, leaf.seq),
        )
        cap_06k = route3_eta_06k * request
        chosen_06k = []
        terms_06k = []
        for leaf in ranked_06k:
            d_leaf_06k = _error_for_ordering(_declaration_06k(leaf))
            if not np.isfinite(d_leaf_06k) or math.fsum(terms_06k + [d_leaf_06k]) > cap_06k:
                break
            terms_06k.append(d_leaf_06k)
            chosen_06k.append(leaf)
        chosen_seqs_06k = {leaf.seq for leaf in chosen_06k}
        if chosen_seqs_06k == left_06k:
            return False
        for seq in sorted(left_06k - chosen_seqs_06k):
            leaf = leaves[seq]
            _remove_error(leaf)
            leaf.region._declared_resolution = None
            _restore_error(leaf)
            left_06k.discard(seq)
            _push_candidate(leaf)
            route3_unleft_06k[0] += 1
        for rank_06k, leaf in enumerate(chosen_06k):
            if leaf.seq in left_06k:
                continue
            d_leaf_06k = _declaration_06k(leaf)
            _remove_error(leaf)
            leaf.region._declared_resolution = d_leaf_06k
            _restore_error(leaf)
            left_06k.add(leaf.seq)
            _push_candidate(leaf)
            region = leaf.region
            route_firings_06k.append(
                {
                    "depth": int(leaf.depth),
                    "at_depth_max": bool(leaf.depth >= depth_max),
                    "start": float(leaf.start),
                    "end": float(leaf.end),
                    "is_direct": bool(leaf.is_direct),
                    "r": float(region._convergence_ratio),
                    "delta1": float(region._delta1),
                    "delta2": float(region._delta2),
                    "phi_P": float(region.phase_err),
                    "residual": float(region.abserr),
                    "D": float(d_leaf_06k),
                    "request": float(request),
                    "cap": float(cap_06k),
                    "rank": int(rank_06k),
                    "n_ranked": len(ranked_06k),
                }
            )
        return True

    _add_leaf(_make_leaf(x_span[0], x_span[1], 0, None, {}, {}, {}))

    running_stop = False
"""

R3_STOP_OLD = """                if not untrusted_refinable:
                    if not (check_eprime + np.fabs(check_node) <= check_request):
                        node_limited = True
                    break
"""

R3_STOP_NEW = """                if _relevel_06k(check_request):  # prompt 06k, route 3
                    continue
                if not untrusted_refinable:
                    if not (check_eprime + np.fabs(check_node) <= check_request):
                        node_limited = True
                    break
"""

R3_CONV_OLD = """    budget_met = abserr_total <= requested_total
    converged = budget_met and num_untrusted == 0
"""

R3_CONV_NEW = """    budget_met = abserr_total <= requested_total
    # prompt 06k, route 3: a left leaf is returned untrusted and does not hold the call unconverged
    left_returned_06k = [
        (i_06k, leaf) for i_06k, leaf in enumerate(leaves_ascending) if leaf.seq in left_06k
    ]
    converged = budget_met and num_untrusted - len(left_returned_06k) == 0
    untrusted_cap_06k = [
        leaf
        for leaf in leaves_ascending
        if leaf.region.trusted is False and leaf.seq not in left_06k
    ]
    left_D_sum_06k = math.fsum(
        leaf.region.declared_resolution for _, leaf in left_returned_06k
    )
    cap_D_sum_06k = math.fsum(
        _declaration_06k(leaf) for leaf in untrusted_cap_06k if leaf.depth >= depth_max
    )
"""

R3_RETURN = """{
            "route": "3",
            "eta": route3_eta_06k,
            "n_firings": len(route_firings_06k),
            "n_unleft": route3_unleft_06k[0],
            "firings": route_firings_06k,
            "request": float(requested_total),
            "n_left_returned": len(left_returned_06k),
            "left_returned": [
                {
                    "region_index": int(i_06k),
                    "depth": int(leaf.depth),
                    "start": float(leaf.start),
                    "end": float(leaf.end),
                    "D": float(leaf.region.declared_resolution),
                    "r": float(leaf.region.convergence_ratio),
                    "delta1": float(leaf.region.delta1),
                    "phi_P": float(leaf.region.phase_err),
                }
                for i_06k, leaf in left_returned_06k
            ],
            "left_D_sum": float(left_D_sum_06k),
            "eta_min": float(left_D_sum_06k / requested_total),
            "n_untrusted_not_left": len(untrusted_cap_06k),
            "n_untrusted_at_depth_max_not_left": sum(
                1 for leaf in untrusted_cap_06k if leaf.depth >= depth_max
            ),
            "untrusted_at_depth_max_D_sum": float(cap_D_sum_06k),
            "eta_min_with_depth_max": float(
                math.fsum([left_D_sum_06k, cap_D_sum_06k]) / requested_total
            ),
            "budget_met": bool(budget_met),
        }"""


def replace_once(text, old, new, what):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{what}: the original occurs {n} times, not once")
    return text.replace(old, new, 1)


def route1(src):
    s = replace_once(src, COUNTERS_OLD, counters_new(1), "counters")
    s = replace_once(s, TRUST_OLD, TRUST_NEW_ROUTE1, "trust test")
    s = replace_once(s, RETURN_OLD, return_new(ROUTE12_RETURN % "1"), "return")
    return s


def route2(src):
    s = replace_once(src, COUNTERS_OLD, counters_new(2), "counters")
    s = replace_once(s, TRUST_OLD, TRUST_NEW_ROUTE2, "trust test")
    s = replace_once(s, DECLARE_OLD, DECLARE_NEW_ROUTE2, "declaration")
    s = replace_once(s, RETURN_OLD, return_new(ROUTE12_RETURN % "2"), "return")
    return s


def route3(src):
    s = replace_once(src, COUNTERS_OLD, counters_new(3), "counters")
    s = replace_once(s, R3_STATE_OLD, R3_STATE_NEW, "state")
    s = replace_once(s, R3_PUSH_OLD, R3_PUSH_NEW, "push")
    s = replace_once(s, R3_HELPERS_OLD, R3_HELPERS_NEW, "helpers")
    s = replace_once(s, R3_STOP_OLD, R3_STOP_NEW, "stop test")
    s = replace_once(s, R3_CONV_OLD, R3_CONV_NEW, "converged")
    s = replace_once(s, RETURN_OLD, return_new(R3_RETURN), "return")
    return s


def main():
    tree_file, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    src_bytes = tree_file.read_bytes()
    src = src_bytes.decode()
    print(f"# levin-phase-floor 06k: the variants of {tree_file}")
    print(f"# script make_variants.py sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}")
    print(f"# input {tree_file} sha256 {hashlib.sha256(src_bytes).hexdigest()}")
    outs = {
        "identity": src_bytes,
        "route1": route1(src).encode(),
        "route2": route2(src).encode(),
        "route3": route3(src).encode(),
    }
    for name, data in outs.items():
        p = out_dir / f"levin_quadrature_{name}.py"
        p.write_bytes(data)
        print(f"{name}: {p} sha256 {hashlib.sha256(data).hexdigest()} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
