"""levin-phase-floor prompt 07a Part A, section 2.3: write the variants of the window's stop. New in 07a.

Each variant is a copy of the tree's ``AdaptiveLevin/levin_quadrature.py`` (SHA-256 checked: 07's g module,
``4e5da371...``) with the same exact text substitutions, differing only in the one constant ``_VARIANT_07A``:

- **the instrument** (every variant; records, changes no value): ``_MEASURE_07A``, a list the replay sets (None, the
  default, records nothing); at every bisection of a window leaf one record (section 2.2: V_P, V_L + V_R, their
  relative change, the raw estimates e_P, e_L, e_R, the declared e'_P, e'_L, e'_R, whether it counted at 07, under
  (c) and under (d), and the change against 1e-5 and against the call's request over |v|); at a window's stop each
  leaf before and after; at the end every returned leaf with its window. The raw estimate of a leaf is its own
  solve's: ``used_interval``'s budget error with no trust-check declaration (``_raw_err_07a``). For the
  "unresolved" test the Clenshaw-Curtis solve also returns ``resasc_07a`` (below), kept on the region;
- **(a) "a"**: 07 as is (the count, the stop and the outcome unchanged);
- **(b) "b"**: no stop: no bisection counts, so no window stops; window leaves are refined like any leaf, to
  ``depth_max``;
- **(c) "c"**: dqage's count: a bisection is not counted if either half's error estimate is unresolved, and the
  error test compares the raw estimates (e_L + e_R >= 0.99 e_P, all three raw), not e'. **The analogue of
  ``defab.eq.error``** (dqk21 sets its error to ``resasc`` when ``200 |K - G| >= resasc``, the Kronrod-Gauss
  difference not yet small against the integrand's mean absolute deviation, i.e. not yet in its asymptotic
  regime): a Clenshaw-Curtis leaf is unresolved if ``200 |CC_{2N-1} - CC_N| >= resasc`` with ``resasc =
  sum_j |w_j| |g_j - gbar|`` on its 2N-1 fine nodes, ``g_j`` the integrand sampled there, ``w_j`` the fine rule's
  weights and ``gbar = CC_{2N-1} / sum_j w_j`` (dqk21's ``resasc``, on the fine rule in place of Kronrod's);
  both positive (dqk21 rescales only then);
- **(d) "d"**: (c), with the value tolerance tied to the request: the value test is
  ``|ΔV| <= min(1e-5, rtol) |V_L + V_R|``, a change counts as "no change" only below the relative accuracy the
  call asks for (dqage's 1e-5 relative where the request is coarser);
- **(e) "e"**: 07's count, but dqage's outcome: a window stopped while the call's request is unmet (the call's
  running error, the halves of the bisection that reached the count added, above max(atol, rtol |V|): dqage's
  ``errsum > errbnd`` test, before which it sets ``ier = 2``) leaves the call unconverged; values and errors as (a);
- **(f) "f"**: dqage in full, section 0.1's default: (c)'s count and (e)'s outcome;
- **(g) "g"**: 07's count, with no count while another leaf of the window fell at its last bisection
  (``fell_at_last_bisection`` True): a window cannot stop with a leaf still falling, the failure 07's census found.

Every substitution outside the instrument is inside ``if leaf.window_index is not None`` (the bisection of a window
leaf) or reads ``windows_stopped`` / ``ier2_windows_07a`` (empty for a call with no window); the Clenshaw-Curtis
solve's extra key and the region's extra attribute are computed from values in hand and read only by the window code
and the instrument. So a call with no window runs the tree's arithmetic (section 2.3; checked on five calls).

Writes scripts/variants/levin_quadrature_07a_<v>.py, out/variants/<v>.diff (``diff -u`` against the tree) and
out/variants/variants.json (each file's SHA-256). Refuses to overwrite. Usage: python -B make_variants_07a.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-f93aafd")
SRC = TREE / "AdaptiveLevin" / "levin_quadrature.py"
SRC_SHA = "4e5da371f51e6e3dc20e73579eb45e0e985e2a3b18571ee29eb86b3d3e9c93fc"
VARIANTS = ("a", "b", "c", "d", "e", "f", "g")

SUBS = []

# 1. the constants, the instrument's list and helpers, after the iroff1 constants
SUBS.append((
    "_LEVIN_WINDOW_IROFF1_LIMIT = 6\n",
    '''_LEVIN_WINDOW_IROFF1_LIMIT = 6

# levin-phase-floor 07a Part A (a measurement, never production): which variant of the window's stop this copy is,
# and the instrument the replay installs (None records nothing). See
# /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/measurement/scripts/make_variants_07a.py.
_VARIANT_07A = "@@VARIANT@@"
_MEASURE_07A = None
# dqk21: its error is set to resasc when 200 |K - G| >= resasc (not yet in its asymptotic regime)
_DQK21_SATURATION_07A = 200.0


def _raw_err_07a(region) -> float:
    """07a instrument: the leaf's own error estimate, used_interval._combine with no trust-check declaration."""
    floor = region._phase_err_less_node if region._phase_err_less_node is not None else region._phase_err
    resolution = region._abserr
    if resolution is None:
        base = floor
    elif floor is None:
        base = resolution
    else:
        base = max(resolution, floor)
    if base is None:
        out = region._abserr_truncation
    elif region._abserr_truncation is None:
        out = base
    else:
        out = base + region._abserr_truncation
    return float("inf") if out is None else float(out)


def _unresolved_07a(region) -> bool:
    """07a: dqk21's defab.eq.error analogue on a Clenshaw-Curtis leaf: 200 |CC_{2N-1} - CC_N| >= resasc, both > 0."""
    resasc = getattr(region, "_resasc_07a", None)
    abserr = region._abserr
    if resasc is None or abserr is None or not (np.isfinite(resasc) and np.isfinite(abserr)):
        return False
    return bool(resasc > 0.0 and abserr > 0.0 and _DQK21_SATURATION_07A * abserr >= resasc)


def _leaf_info_07a(leaf) -> dict:
    """07a instrument: what a leaf carries now."""
    r = leaf.region
    return {
        "start": float(leaf.start), "end": float(leaf.end), "depth": int(leaf.depth), "seq": int(leaf.seq),
        "window_index": leaf.window_index, "fell": leaf.fell_at_last_bisection, "value": float(leaf.value),
        "value_own": float(r._value_own), "value_kind": r._value_kind, "raw": _raw_err_07a(r),
        "eprime": float(_error_for_ordering(leaf.error)), "abserr": r._abserr, "phase_err": r._phase_err,
        "phase_err_less_node": r._phase_err_less_node, "node_error_signed": r._node_error_signed,
        "abserr_truncation": r._abserr_truncation, "resasc": getattr(r, "_resasc_07a", None),
        "unresolved": _unresolved_07a(r), "trusted": r._trusted, "checked": bool(leaf.checked),
        "phase_limited": bool(r._phase_limited), "is_direct": bool(leaf.is_direct), "window": bool(r._window),
        "delta1": r._delta1, "delta2": r._delta2, "ratio": r._convergence_ratio, "declared": r._declared_resolution,
    }
''',
))

# 2. the Clenshaw-Curtis solve's resasc (dqk21's), from the samples in hand
SUBS.append((
    '''    return {
        "value": value_fine,
        "abserr_direct": abserr_direct,
''',
    '''    # 07a instrument: dqk21's resasc on the fine rule, sum_j |w_j| |g_j - gbar|, gbar = CC_{2N-1} / sum_j w_j
    _w_sum_07a = float(np.sum(w_fine))
    _gbar_07a = value_fine / _w_sum_07a if _w_sum_07a != 0.0 else 0.0
    _resasc_07a = float(np.sum(np.fabs(w_fine) * np.fabs(integrand_fine - _gbar_07a)))
    return {
        "resasc_07a": _resasc_07a,
        "value": value_fine,
        "abserr_direct": abserr_direct,
''',
))

# 3. the region keeps it (a fallback leaf only)
SUBS.append((
    '''                window=window,
                guarded=guarded,
                guard_at_depth_max=guard_at_depth_max,
            )
        else:
''',
    '''                window=window,
                guarded=guarded,
                guard_at_depth_max=guard_at_depth_max,
            )
            region._resasc_07a = data.get("resasc_07a")  # 07a instrument
        else:
''',
))

# 4. the windows stopped while the request was unmet (variants e, f), next to windows_stopped
SUBS.append((
    '''    window_counts = {}
    windows_stopped = {}
''',
    '''    window_counts = {}
    windows_stopped = {}
    ier2_windows_07a = set()  # 07a: windows stopped while the call's request was unmet (dqage's ier = 2)
    shadow_counts_07a = {}  # 07a instrument: 07's count, kept whatever the variant counts
''',
))

# 5. the stop: the instrument, before and after
SUBS.append((
    '''        members = [
            leaf for leaf in leaves.values() if leaf.window_index == window_index
        ]
        windows_stopped[window_index] = {''',
    '''        members = [
            leaf for leaf in leaves.values() if leaf.window_index == window_index
        ]
        _stop_rec_07a = None
        if _MEASURE_07A is not None:
            _stop_rec_07a = {"kind": "stop", "window_index": window_index, "split": num_splits,
                             "before": [_leaf_info_07a(m_) for m_ in sorted(members, key=lambda m_: m_.start)]}
        windows_stopped[window_index] = {''',
))
SUBS.append((
    '''        for member in members:
            _floor_limit_stopped_window_leaf(member)
''',
    '''        for member in members:
            _floor_limit_stopped_window_leaf(member)
        if _stop_rec_07a is not None:
            _stop_rec_07a["after"] = [_leaf_info_07a(m_) for m_ in sorted(members, key=lambda m_: m_.start)]
            _MEASURE_07A.append(_stop_rec_07a)
''',
))

# 6. the bisection of a window leaf: the count, by variant, and the instrument
SUBS.append((
    '''        stop_window = None
        if leaf.window_index is not None:
            value_halves = left.region.value_own + right.region.value_own
            error_halves = _error_for_ordering(left.error) + _error_for_ordering(
                right.error
            )
            error_parent = _error_for_ordering(leaf.error)
            error_flat = error_halves >= _LEVIN_WINDOW_IROFF1_ERROR_RATIO * error_parent
            left.fell_at_last_bisection = not error_flat
            right.fell_at_last_bisection = not error_flat
            if (
                np.fabs(value_halves - region.value_own)
                <= _LEVIN_WINDOW_IROFF1_VALUE_RTOL * np.fabs(value_halves)
                and error_flat
            ):
                window_counts[leaf.window_index] = (
                    window_counts.get(leaf.window_index, 0) + 1
                )
                if (
                    window_counts[leaf.window_index] >= _LEVIN_WINDOW_IROFF1_LIMIT
                    and leaf.window_index not in windows_stopped
                ):
                    stop_window = leaf.window_index

        _remove_leaf(leaf)
        leaf.dataL = None
        leaf.dataR = None
        leaf.grandchildren = None
        _add_leaf(left)
        _add_leaf(right)
        if stop_window is not None:
            _stop_window(stop_window)
''',
    '''        stop_window = None
        _bis_rec_07a = None
        if leaf.window_index is not None:
            value_halves = left.region.value_own + right.region.value_own
            error_halves = _error_for_ordering(left.error) + _error_for_ordering(
                right.error
            )
            error_parent = _error_for_ordering(leaf.error)
            error_flat = error_halves >= _LEVIN_WINDOW_IROFF1_ERROR_RATIO * error_parent
            left.fell_at_last_bisection = not error_flat
            right.fell_at_last_bisection = not error_flat
            # 07a: the three counts, and the one this variant uses
            value_change_07a = np.fabs(value_halves - region.value_own)
            counted_07 = bool(
                value_change_07a
                <= _LEVIN_WINDOW_IROFF1_VALUE_RTOL * np.fabs(value_halves)
                and error_flat
            )
            raw_parent_07a = _raw_err_07a(region)
            raw_left_07a = _raw_err_07a(left.region)
            raw_right_07a = _raw_err_07a(right.region)
            unres_left_07a = _unresolved_07a(left.region)
            unres_right_07a = _unresolved_07a(right.region)
            raw_flat_07a = bool(
                raw_left_07a + raw_right_07a >= _LEVIN_WINDOW_IROFF1_ERROR_RATIO * raw_parent_07a
            )
            skip_07a = unres_left_07a or unres_right_07a
            counted_c_07a = bool(
                not skip_07a
                and value_change_07a <= _LEVIN_WINDOW_IROFF1_VALUE_RTOL * np.fabs(value_halves)
                and raw_flat_07a
            )
            counted_d_07a = bool(
                not skip_07a
                and value_change_07a
                <= min(_LEVIN_WINDOW_IROFF1_VALUE_RTOL, rtol) * np.fabs(value_halves)
                and raw_flat_07a
            )
            others_falling_07a = [
                other_.seq
                for other_ in leaves.values()
                if other_.window_index == leaf.window_index
                and other_ is not leaf
                and other_.fell_at_last_bisection is True
            ]
            if _VARIANT_07A in ("a", "e"):
                counted_variant_07a = counted_07
            elif _VARIANT_07A == "b":
                counted_variant_07a = False
            elif _VARIANT_07A in ("c", "f"):
                counted_variant_07a = counted_c_07a
            elif _VARIANT_07A == "d":
                counted_variant_07a = counted_d_07a
            elif _VARIANT_07A == "g":
                counted_variant_07a = counted_07 and not others_falling_07a
            else:
                raise RuntimeError(f"07a: unknown variant {_VARIANT_07A!r}")
            if counted_07:
                shadow_counts_07a[leaf.window_index] = (
                    shadow_counts_07a.get(leaf.window_index, 0) + 1
                )
            if counted_variant_07a:
                window_counts[leaf.window_index] = (
                    window_counts.get(leaf.window_index, 0) + 1
                )
                if (
                    window_counts[leaf.window_index] >= _LEVIN_WINDOW_IROFF1_LIMIT
                    and leaf.window_index not in windows_stopped
                ):
                    stop_window = leaf.window_index
            if _MEASURE_07A is not None:
                rv_07a, re_07a, rn_07a = _running_sums()
                request_07a = max(atol, rtol * np.fabs(rv_07a))
                rel_change_07a = (
                    float(value_change_07a / np.fabs(value_halves))
                    if value_halves != 0.0
                    else float("inf")
                )
                request_over_v_07a = (
                    float(request_07a / np.fabs(rv_07a)) if rv_07a != 0.0 else float("inf")
                )
                _bis_rec_07a = {
                    "kind": "bisection", "split": num_splits, "window_index": leaf.window_index,
                    "parent": _leaf_info_07a(leaf), "left": _leaf_info_07a(left), "right": _leaf_info_07a(right),
                    "V_P": float(region.value_own), "V_LR": float(value_halves),
                    "abs_change": float(value_change_07a), "rel_change": rel_change_07a,
                    "e_P": raw_parent_07a, "e_L": raw_left_07a, "e_R": raw_right_07a,
                    "ep_P": float(error_parent), "ep_L": float(_error_for_ordering(left.error)),
                    "ep_R": float(_error_for_ordering(right.error)),
                    "error_flat_07": bool(error_flat), "raw_flat": raw_flat_07a,
                    "unres_L": unres_left_07a, "unres_R": unres_right_07a,
                    "counted_07": counted_07, "counted_c": counted_c_07a, "counted_d": counted_d_07a,
                    "others_falling": others_falling_07a, "counted_variant": bool(counted_variant_07a),
                    "count_variant_after": int(window_counts.get(leaf.window_index, 0)),
                    "count_07_after": int(shadow_counts_07a.get(leaf.window_index, 0)),
                    "rel_change_le_1e-5": bool(rel_change_07a <= _LEVIN_WINDOW_IROFF1_VALUE_RTOL),
                    "request_over_v": request_over_v_07a,
                    "rel_change_le_request_over_v": bool(rel_change_07a <= request_over_v_07a),
                    "running_value": float(rv_07a), "running_eprime": float(re_07a), "running_node": float(rn_07a),
                    "running_request": float(request_07a), "rtol": float(rtol),
                    "stop": stop_window is not None,
                }

        _remove_leaf(leaf)
        leaf.dataL = None
        leaf.dataR = None
        leaf.grandchildren = None
        _add_leaf(left)
        _add_leaf(right)
        if stop_window is not None:
            # 07a: dqage's outcome (variants e, f): the call's request unmet once the halves are in
            rv2_07a, re2_07a, rn2_07a = _running_sums()
            unmet_07a = bool(re2_07a + rn2_07a > max(atol, rtol * np.fabs(rv2_07a)))
            if unmet_07a and _VARIANT_07A in ("e", "f"):
                ier2_windows_07a.add(stop_window)
            if _bis_rec_07a is not None:
                _bis_rec_07a["request_unmet_after_halves"] = unmet_07a
            _stop_window(stop_window)
        if _bis_rec_07a is not None:
            _MEASURE_07A.append(_bis_rec_07a)
''',
))

# 7. the end: the instrument's record of every returned leaf
SUBS.append((
    '''    leaves_ascending = _leaves_ascending()
    used_regions = [leaf.region for leaf in leaves_ascending]
''',
    '''    leaves_ascending = _leaves_ascending()
    used_regions = [leaf.region for leaf in leaves_ascending]
    if _MEASURE_07A is not None:
        _MEASURE_07A.append({
            "kind": "final", "variant": _VARIANT_07A, "num_splits": num_splits,
            "window_counts": {int(k_): int(v_) for k_, v_ in window_counts.items()},
            "shadow_counts_07": {int(k_): int(v_) for k_, v_ in shadow_counts_07a.items()},
            "windows_stopped": {int(k_): dict(v_) for k_, v_ in windows_stopped.items()},
            "ier2_windows": sorted(int(k_) for k_ in ier2_windows_07a),
            "leaves": [_leaf_info_07a(leaf_) for leaf_ in leaves_ascending],
        })
''',
))

# 8. converged: dqage's outcome (variants e, f)
SUBS.append((
    '''    converged = (
        budget_met and num_untrusted == 0 and num_guard_at_depth_max_returned == 0
    )
''',
    '''    converged = (
        budget_met and num_untrusted == 0 and num_guard_at_depth_max_returned == 0
    )
    if ier2_windows_07a:
        # 07a, variants e and f: a window stopped while the call's request was unmet (dqage's ier = 2)
        converged = False
''',
))


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if sha256(SRC) != SRC_SHA:
        raise SystemExit("the tree's levin_quadrature.py has another SHA-256")
    src = SRC.read_text()
    vdir = HERE / "variants"
    odir = HERE.parent / "out" / "variants"
    vdir.mkdir(exist_ok=True)
    odir.mkdir(parents=True, exist_ok=True)
    body = src
    for old, new in SUBS:
        n = body.count(old)
        if n != 1:
            raise SystemExit(f"substitution anchor found {n} times: {old[:80]!r}")
        body = body.replace(old, new)
    index = {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__), "tree_module": str(SRC),
             "tree_module_sha256": SRC_SHA, "variants": {}}
    for v in VARIANTS:
        out = vdir / f"levin_quadrature_07a_{v}.py"
        if out.exists():
            raise SystemExit(f"{out} exists")
        assert body.count("@@VARIANT@@") == 1
        out.write_text(body.replace("@@VARIANT@@", v))
        diff = subprocess.run(["diff", "-u", str(SRC), str(out)], capture_output=True, text=True).stdout
        (odir / f"{v}.diff").write_text(diff)
        index["variants"][v] = {"path": str(out), "sha256": sha256(out), "diff": str(odir / f"{v}.diff"),
                                "diff_sha256": sha256(odir / f"{v}.diff")}
        print(v, sha256(out))
    (odir / "variants.json").write_text(json.dumps(index, indent=1))
    print("wrote", odir / "variants.json", sha256(odir / "variants.json"))


if __name__ == "__main__":
    sys.exit(main())
