"""levin-phase-floor prompt 07c Part A, section 2.2 (D1): write the instrument ``instr``, a copy of the tree's
AdaptiveLevin/levin_quadrature.py (07b's code, 8956c7b6...) that records and changes nothing. New in 07c.

Every patch is an exact replacement of one anchor that occurs once in the tree's module (asserted), marked
"07c instrument". The instrument only reads: every quantity it records is already computed by the tree's code at that
point, or is computed by a pure function of the driver's state (``_running_sums()``, ``_split_could_meet_budget()``,
both of which read the exact sums and change nothing). Two refactorings keep the tree's arithmetic exactly:
- the rate rules' third return value, which the tree discards as ``_``, is bound to a name;
- the round-off exit's budget clause, ``_split_could_meet_budget(leaf)``, is evaluated once into a local and that local
  passed to ``_roundoff_exit_premise_holds``, which receives the same bool as before.
Records go to the module list ``_MEASURE_07C`` when the caller sets it (None: nothing is recorded). Kinds:
- ``seed``: each leaf of the initial partition (seq, span, route, window index);
- ``check``: every trust check: seq, span, depth, route, delta1, delta2, r, nu_B, the residual (region.abserr), the
  floor (region.phase_err) and floor less node, trusted by the test, which clause failed, the rate rules' outcome
  (refused, rule, q), trusted in the end, the call's request and error then (running sums);
- ``pop``: every leaf chosen: seq, depth, route, why (``untrusted`` or ``budget``), trusted state, residual and floor,
  whether the round-off exit was considered (not window, not guarded, finite floor, residual <= floor), and if so its
  halving test and budget clause separately, and whether the leaf was found floor-limited;
- ``split``: every bisection: parent seq, children seqs, the halves' floors (phase_err, phase_err_roundoff), the halving
  test and the budget clause evaluated on them at that moment (pure), the children's own values, the parent's own
  value, the request then;
- ``stop_window``: a window stopped;
- ``final``: every returned leaf: seq, span, depth, route, trusted, value kind, phase_limited.

Usage: python -B make_instr_07c.py  (writes scripts/variants/levin_quadrature_07c_instr.py; refuses to overwrite)
"""

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREE_LQ = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f/AdaptiveLevin/levin_quadrature.py")
TREE_SHA = "8956c7b6a8ec285f8898353eed65f4039a02b42edc6927565be3896cda52def1"
OUT = HERE / "variants" / "levin_quadrature_07c_instr.py"

PATCHES = [
    # module constants
    ("_LEVIN_EDGE_RATE_RATIO_MAX = 2.0\n_LEVIN_LEAF_RATE_RANGE_MAX = 10.0\n",
     "_LEVIN_EDGE_RATE_RATIO_MAX = 2.0\n_LEVIN_LEAF_RATE_RANGE_MAX = 10.0\n"
     "_VARIANT_07C = \"instr\"  # 07c instrument: its name\n"
     "_MEASURE_07C = None  # 07c instrument: a list set by the caller, or None\n"),
    # helpers inside _adaptive_levin, before the first leaf is made
    ("    seq_counter = 0\n\n    def _make_leaf(\n",
     "    seq_counter = 0\n"
     "    _m07c = _MEASURE_07C  # 07c instrument\n\n"
     "    def _route07c(leaf):  # 07c instrument: a leaf's route\n"
     "        if leaf.region.guard_at_depth_max:\n"
     "            return \"guarded\"\n"
     "        if leaf.region.window:\n"
     "            return \"window\"\n"
     "        if leaf.is_direct:\n"
     "            return \"cc_gate\"\n"
     "        return \"levin\"\n\n"
     "    def _req07c():  # 07c instrument: the running request and error (pure)\n"
     "        v_, e_, n_ = _running_sums()\n"
     "        return float(max(atol, rtol * np.fabs(v_))), float(e_), float(n_)\n\n"
     "    def _make_leaf(\n"),
    # check: keep q
    ("            refused, rule, _ = _leaf_rate_rule(\n",
     "            refused, rule, _q07c = _leaf_rate_rule(  # 07c instrument: q kept (the tree discards it)\n"),
    # check: before the rate rules, remember the test's verdict
    ("        trusted = ratio < _LEVIN_TRUST_MAX_RATIO or (\n            ratio < 1.0 and delta1 <= noise_yardstick\n        )\n",
     "        trusted = ratio < _LEVIN_TRUST_MAX_RATIO or (\n            ratio < 1.0 and delta1 <= noise_yardstick\n        )\n"
     "        _test07c = bool(trusted)  # 07c instrument\n"
     "        _rr07c = None  # 07c instrument\n"),
    ("            if refused:\n                trusted = False\n                if rule == \"edge\":\n",
     "            _rr07c = {\"refused\": bool(refused), \"rule\": rule, \"q\": float(_q07c)}  # 07c instrument\n"
     "            if refused:\n                trusted = False\n                if rule == \"edge\":\n"),
    # check: the record, after the leaf's trust is set and before its value moves
    ("        _restore_error(leaf)\n        if trusted:\n            _set_value(leaf, quarters_sum, \"quarters\")\n",
     "        _restore_error(leaf)\n"
     "        if _m07c is not None:  # 07c instrument: the check\n"
     "            _rq, _ep, _nd = _req07c()\n"
     "            _m07c.append({\"kind\": \"check\", \"seq\": leaf.seq, \"start\": a, \"end\": b, \"depth\": leaf.depth,\n"
     "                          \"route\": _route07c(leaf), \"window_index\": leaf.window_index,\n"
     "                          \"delta1\": delta1, \"delta2\": delta2, \"r\": ratio, \"nu_B\": noise_yardstick,\n"
     "                          \"residual\": residual, \"floor\": region.phase_err,\n"
     "                          \"floor_less_node\": region.phase_err_less_node, \"resasc\": region.resasc,\n"
     "                          \"own_value\": own_value, \"halves_sum\": halves_sum, \"quarters_sum\": quarters_sum,\n"
     "                          \"trusted_by_test\": _test07c,\n"
     "                          \"clause_failed\": None if _test07c else (\"r>=1\" if not ratio < 1.0 else \"noise\"),\n"
     "                          \"rate_rule\": _rr07c, \"trusted\": bool(trusted),\n"
     "                          \"declared_resolution\": region._declared_resolution,\n"
     "                          \"request\": _rq, \"call_eprime\": _ep, \"call_node\": _nd})\n"
     "        if trusted:\n            _set_value(leaf, quarters_sum, \"quarters\")\n"),
    # seeds and splits: record in _add_leaf? No: seeds recorded where made. Initial partition, no windows
    ("    if location is None or not location[\"windows\"]:\n        _add_leaf(_make_leaf(x_span[0], x_span[1], 0, None, {}, {}, {}))\n",
     "    if location is None or not location[\"windows\"]:\n        _add_leaf(_make_leaf(x_span[0], x_span[1], 0, None, {}, {}, {}))\n"
     "        if _m07c is not None:  # 07c instrument: the seed\n"
     "            for _lf in leaves.values():\n"
     "                _m07c.append({\"kind\": \"seed\", \"seq\": _lf.seq, \"start\": _lf.start, \"end\": _lf.end,\n"
     "                              \"route\": _route07c(_lf), \"window_index\": _lf.window_index})\n"),
    # initial partition with windows
    ("                    window=seed_window_index is not None,\n                    window_index=seed_window_index,\n                )\n            )\n",
     "                    window=seed_window_index is not None,\n                    window_index=seed_window_index,\n                )\n            )\n"
     "        if _m07c is not None:  # 07c instrument: the seeds\n"
     "            for _lf in sorted(leaves.values(), key=lambda _x: _x.seq):\n"
     "                _m07c.append({\"kind\": \"seed\", \"seq\": _lf.seq, \"start\": _lf.start, \"end\": _lf.end,\n"
     "                              \"route\": _route07c(_lf), \"window_index\": _lf.window_index})\n"),
    # the round-off exit: budget clause into a local, record
    ("                if leaf.dataL is None:\n                    _solve_halves(leaf)\n"
     "                floor_limited = _roundoff_exit_premise_holds(\n"
     "                    region.phase_err_roundoff,\n"
     "                    _solve_floor_roundoff(leaf.dataL),\n"
     "                    _solve_floor_roundoff(leaf.dataR),\n"
     "                    split_could_meet_budget=_split_could_meet_budget(leaf),\n"
     "                )\n",
     "                if leaf.dataL is None:\n                    _solve_halves(leaf)\n"
     "                _scmb07c = _split_could_meet_budget(leaf)  # 07c instrument: evaluated once, passed as before\n"
     "                floor_limited = _roundoff_exit_premise_holds(\n"
     "                    region.phase_err_roundoff,\n"
     "                    _solve_floor_roundoff(leaf.dataL),\n"
     "                    _solve_floor_roundoff(leaf.dataR),\n"
     "                    split_could_meet_budget=_scmb07c,\n"
     "                )\n"
     "                if _m07c is not None:  # 07c instrument: the exit's two tests\n"
     "                    _exit07c = {\"halves_floor_roundoff\": _solve_floor_roundoff(leaf.dataL) + _solve_floor_roundoff(leaf.dataR),\n"
     "                                \"floor_roundoff\": region.phase_err_roundoff,\n"
     "                                \"halving_holds\": bool((_solve_floor_roundoff(leaf.dataL) + _solve_floor_roundoff(leaf.dataR))\n"
     "                                                       >= _LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO * region.phase_err_roundoff),\n"
     "                                \"split_could_meet_budget\": bool(_scmb07c), \"floor_limited\": bool(floor_limited)}\n"),
    # the pop record: right after the leaf is chosen and before the exit test
    ("        leaf = _pop_candidate()\n        if leaf is None:\n            break\n        region = leaf.region\n",
     "        leaf = _pop_candidate()\n        if leaf is None:\n            break\n        region = leaf.region\n"
     "        _exit07c = None  # 07c instrument\n"
     "        if _m07c is not None:  # 07c instrument: the leaf chosen\n"
     "            _rq, _ep, _nd = _req07c()\n"
     "            _pop07c = {\"kind\": \"pop\", \"seq\": leaf.seq, \"start\": leaf.start, \"end\": leaf.end, \"depth\": leaf.depth,\n"
     "                       \"route\": _route07c(leaf), \"why\": \"untrusted\" if region._trusted is False else \"budget\",\n"
     "                       \"trusted\": region._trusted, \"checked\": bool(leaf.checked), \"residual\": region.abserr,\n"
     "                       \"floor\": region.phase_err, \"floor_less_node\": region.phase_err_less_node,\n"
     "                       \"eprime\": _error_for_ordering(leaf.error), \"window\": bool(region.window),\n"
     "                       \"guarded\": bool(leaf.guarded),\n"
     "                       \"exit_considered\": bool(not region.window and not leaf.guarded\n"
     "                                                and np.isfinite(region.phase_err) and region.abserr <= region.phase_err),\n"
     "                       \"request\": _rq, \"call_eprime\": _ep, \"call_node\": _nd}\n"
     "            _m07c.append(_pop07c)\n"),
    # after the exit block: record its outcome on the pop record
    ("            if floor_limited:\n                region._phase_limited = True\n                _exempt_floor_limited(leaf)\n",
     "            if _m07c is not None:  # 07c instrument: the exit's outcome on the pop record\n"
     "                _pop07c[\"exit\"] = _exit07c\n"
     "            if floor_limited:\n                region._phase_limited = True\n                _exempt_floor_limited(leaf)\n"),
    # the split record, after the children are made and counted
    ("        num_splits += 1\n\n        # The window's stopping rule",
     "        num_splits += 1\n"
     "        if _m07c is not None:  # 07c instrument: the split\n"
     "            _rq, _ep, _nd = _req07c()\n"
     "            _m07c.append({\"kind\": \"split\", \"seq\": leaf.seq, \"left\": left.seq, \"right\": right.seq,\n"
     "                          \"depth\": leaf.depth, \"route\": _route07c(leaf),\n"
     "                          \"parent_own_value\": region.value_own, \"parent_value\": leaf.value,\n"
     "                          \"left_own_value\": left.region.value_own, \"right_own_value\": right.region.value_own,\n"
     "                          \"parent_floor_roundoff\": region.phase_err_roundoff,\n"
     "                          \"halves_floor_roundoff\": left.region.phase_err_roundoff + right.region.phase_err_roundoff,\n"
     "                          \"left_route\": _route07c(left), \"right_route\": _route07c(right),\n"
     "                          \"request\": _rq, \"call_eprime\": _ep, \"call_node\": _nd})\n\n"
     "        # The window's stopping rule"),
    # window stop
    ("        for member in members:\n            _floor_limit_stopped_window_leaf(member)\n",
     "        for member in members:\n            _floor_limit_stopped_window_leaf(member)\n"
     "        if _m07c is not None:  # 07c instrument: a window stopped\n"
     "            _m07c.append({\"kind\": \"stop_window\", \"window_index\": window_index, \"n_members\": len(members)})\n"),
    # final leaves
    ("    leaves_ascending = _leaves_ascending()\n    used_regions = [leaf.region for leaf in leaves_ascending]\n",
     "    leaves_ascending = _leaves_ascending()\n    used_regions = [leaf.region for leaf in leaves_ascending]\n"
     "    if _m07c is not None:  # 07c instrument: the returned leaves\n"
     "        for _lf in leaves_ascending:\n"
     "            _m07c.append({\"kind\": \"final\", \"seq\": _lf.seq, \"start\": _lf.start, \"end\": _lf.end,\n"
     "                          \"depth\": _lf.depth, \"route\": _route07c(_lf), \"trusted\": _lf.region._trusted,\n"
     "                          \"value_kind\": _lf.region._value_kind, \"phase_limited\": bool(_lf.region._phase_limited),\n"
     "                          \"residual\": _lf.region.abserr, \"floor\": _lf.region.phase_err,\n"
     "                          \"eprime\": _error_for_ordering(_lf.error), \"value\": _lf.value})\n"),
]


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    src = TREE_LQ.read_text()
    if hashlib.sha256(src.encode()).hexdigest() != TREE_SHA:
        raise SystemExit("the tree's module has another SHA-256")
    s = src
    for old, new in PATCHES:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"anchor occurs {n} times: {old[:80]!r}")
        s = s.replace(old, new)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(s)
    print(f"wrote {OUT} sha256 {hashlib.sha256(s.encode()).hexdigest()}")


if __name__ == "__main__":
    sys.exit(main())
