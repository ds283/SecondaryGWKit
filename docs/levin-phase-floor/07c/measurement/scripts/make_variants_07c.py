"""levin-phase-floor prompt 07c Part A, section 2.4: write a candidate variant of the tree's AdaptiveLevin/levin_quadrature.py
(07b's code, 8956c7b6...). New in 07c. Each patch replaces one anchor that occurs once (asserted), marked "07c variant".

Usage: python -B make_variants_07c.py NAME [--s LIMIT NOISE_FACTOR] [--e SCOPE ESTIMATE] [--n] [--instr] --basis TEXT
``--instr`` starts from the instrument (scripts/variants/levin_quadrature_07c_instr.py, its SHA-256 checked) instead of the
tree: the instrumented variant, run on the chase calls for table 4's descent column, and shown equal to the plain variant
bit for bit there. Its ``_VARIANT_07C`` is NAME (the variant's assignment follows the instrument's).
Writes scripts/variants/levin_quadrature_07c_<NAME>.py (refusing to overwrite) and state/variants/<NAME>.json (the
options and the basis text). Register it with register_variant_07c.py.

**(S) A stop that reaches the leaves made after it** (``--s LIMIT NOISE_FACTOR``).
- Every leaf the rate rules refuse roots a subtree: the leaf's root is the outermost refused leaf above it, or itself
  (``_s07c_root``, by sequence number; children inherit their parent's root at bisection).
- A checked leaf is *at the noise level* if delta1 <= NOISE_FACTOR nu_B and delta2 <= NOISE_FACTOR nu_B, with nu_B its trust
  check's own noise yardstick (``_trust_noise_yardstick``): the leaf, its halves and its quarters differ by no more than the
  rounding of the solves they come from. A leaf never checked is not at the noise level.
- A bisection of a leaf at the noise level in a subtree counts against that subtree's root. At LIMIT counts the subtree
  is stopped: every leaf of it then at the noise level is floor-limited by ``_floor_limit_stopped_window_leaf`` (not
  refinable, exempt from the trust check, declaring what it carries: an untrusted leaf as an untrusted leaf at depth_max
  declares; value its own), and every leaf made in it later is floor-limited the same way once its own trust check finds
  it at the noise level. A leaf not at the noise level is never stopped (07b's (i): its stop returned Levin leaves not at
  the noise level at V_P with the r-based declaration).
- Counts in the result: ``num_s07c_stops`` (subtrees stopped), ``num_s07c_stopped_leaves`` (leaves floor-limited by a
  stop), ``num_s07c_counted`` (bisections counted).

**(E) A sharper Clenshaw-Curtis estimate** (``--e SCOPE ESTIMATE``): the fallback's ``abserr_direct`` is ESTIMATE in place
of |CC_{2N-1} - CC_N| (kept as ``abserr_pair_07c`` in the solve's dict); the value stays CC_{2N-1}.
- ESTIMATE ``nest3``: |CC_{4N-3} - CC_{2N-1}|, CC_{4N-3} on its own grid (``chebyshev_matrices``, ``_cc_weights``), the
  integrand sampled as the fallback samples it, 4N - 3 more evaluations (the 2N - 2 a nesting implementation would need,
  plus the 2N - 1 shared nodes evaluated again);
- ESTIMATE ``tail_sum4``: (b - a)/2 times the sum of the last four |c_k| of the fine interpolant's Chebyshev coefficients
  (DCT-I of the 2N - 1 fine samples), no extra evaluation.
- SCOPE ``gate``: only the C2 gate's leaves (``_adaptive_levin_subregion_cc`` called from the gate, not from
  ``_oriented_cc``); ``all``: every Clenshaw-Curtis solve, window pieces and guarded leaves at the cap included.

**(N) The noise clause without r** (``--n``): a checked leaf with delta1 <= nu_B and delta2 <= nu_B that the tree's test
does not trust (r >= 1) is trusted, declaring max(residual, delta1 + delta2) with no geometric factor (a Levin leaf then
meets the rate rules as any trusted leaf); counted in ``num_n07c_trusted``.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE_LQ = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f/AdaptiveLevin/levin_quadrature.py")
TREE_SHA = "8956c7b6a8ec285f8898353eed65f4039a02b42edc6927565be3896cda52def1"


def rep(s, old, new, within=None):
    if within is not None:
        a, b = within
        i, j = s.index(a), None
        j = s.index(b, i)
        seg = s[i:j]
        if seg.count(old) != 1:
            raise SystemExit(f"anchor occurs {seg.count(old)} times in its section: {old[:70]!r}")
        return s[:i] + seg.replace(old, new) + s[j:]
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"anchor occurs {n} times: {old[:70]!r}")
    return s.replace(old, new)


def patch_common(s, name, opts):
    return rep(s, "_LEVIN_EDGE_RATE_RATIO_MAX = 2.0\n_LEVIN_LEAF_RATE_RANGE_MAX = 10.0\n",
               "_LEVIN_EDGE_RATE_RATIO_MAX = 2.0\n_LEVIN_LEAF_RATE_RANGE_MAX = 10.0\n"
               f"_VARIANT_07C = {name!r}  # 07c variant: its name\n"
               "_MEASURE_07C = None  # 07c variant: unused (replay_07c.py's instrument hook)\n"
               f"_S07C = {opts.get('s')!r}  # 07c variant (S): (LIMIT, NOISE_FACTOR) or None\n"
               f"_E07C = {opts.get('e')!r}  # 07c variant (E): (SCOPE, ESTIMATE) or None\n"
               f"_N07C = {bool(opts.get('n'))!r}  # 07c variant (N)\n")


def patch_E(s):
    # the estimate, a module function before the fallback
    s = rep(s, "def _adaptive_levin_subregion_cc(\n",
            "def _e07c_estimate(x_span, f, BasisData, chebyshev_order, value_fine, integrand_fine, vectorize_cache):\n"
            "    \"\"\"07c variant (E): the sharper estimate of CC_{2N-1}'s error (see make_variants_07c.py).\"\"\"\n"
            "    kind = _E07C[1]\n"
            "    if kind == \"nest3\":\n"
            "        order3 = 4 * chebyshev_order - 3\n"
            "        grid3, _ = chebyshev_matrices(x_span, order3)\n"
            "        f3 = np.vstack([_sample_vectorized(func, grid3, vectorize_cache, (\"f\", id(func))) for func in f])\n"
            "        basis3 = np.array([BasisData.eval_basis_and_arguments(x)[0] for x in grid3]).T\n"
            "        value3 = float(np.dot(_cc_weights(x_span, order3), np.sum(f3 * basis3, axis=0)))\n"
            "        return float(np.fabs(value3 - value_fine))\n"
            "    if kind == \"tail_sum4\":\n"
            "        n = integrand_fine.shape[0]\n"
            "        j = np.arange(n)\n"
            "        wts = np.ones(n)\n"
            "        wts[0] = wts[-1] = 0.5\n"
            "        ck = np.array([(2.0 / (n - 1)) * np.sum(wts * integrand_fine * np.cos(np.pi * j * k / (n - 1))) for k in range(n - 4, n)])\n"
            "        ck[-1] *= 0.5\n"
            "        return float(0.5 * np.fabs(x_span[1] - x_span[0]) * np.sum(np.abs(ck)))\n"
            "    raise RuntimeError(\"07c variant (E): unknown estimate\")\n\n\n"
            "def _adaptive_levin_subregion_cc(\n")
    s = rep(s, "    vectorize_cache: Optional[dict] = None,\n):\n    \"\"\"\n    Bounded-cost fallback",
            "    vectorize_cache: Optional[dict] = None,\n    forced: bool = False,  # 07c variant (E): called from _oriented_cc()\n"
            "):\n    \"\"\"\n    Bounded-cost fallback")
    s = rep(s, "    abserr_direct = float(np.fabs(value_fine - value_coarse))\n",
            "    abserr_direct = float(np.fabs(value_fine - value_coarse))\n"
            "    abserr_pair_07c = abserr_direct  # 07c variant (E)\n"
            "    if _E07C is not None and (_E07C[0] == \"all\" or not forced):  # 07c variant (E)\n"
            "        abserr_direct = _e07c_estimate(x_span, f, BasisData, chebyshev_order, value_fine, integrand_fine, vectorize_cache)\n")
    s = rep(s, "        \"abserr_direct\": abserr_direct,\n        \"resasc\": resasc,\n",
            "        \"abserr_direct\": abserr_direct,\n        \"abserr_pair_07c\": abserr_pair_07c,  # 07c variant (E)\n        \"resasc\": resasc,\n")
    sec = ("def _oriented_cc(", "def _scalar_callable(")
    s = rep(s, "            id_label,\n            vectorize_cache=vectorize_cache,\n        )\n\n    ascending = (b, a)\n",
            "            id_label,\n            vectorize_cache=vectorize_cache,\n            forced=True,  # 07c variant (E)\n        )\n\n    ascending = (b, a)\n", within=sec)
    s = rep(s, "        id_label,\n        vectorize_cache=vectorize_cache,\n    )\n    data[\"value\"] = -data[\"value\"]\n",
            "        id_label,\n        vectorize_cache=vectorize_cache,\n        forced=True,  # 07c variant (E)\n    )\n    data[\"value\"] = -data[\"value\"]\n", within=sec)
    return s


def patch_N(s):
    s = rep(s, "        trusted = ratio < _LEVIN_TRUST_MAX_RATIO or (\n            ratio < 1.0 and delta1 <= noise_yardstick\n        )\n",
            "        trusted = ratio < _LEVIN_TRUST_MAX_RATIO or (\n            ratio < 1.0 and delta1 <= noise_yardstick\n        )\n"
            "        _n07c = False  # 07c variant (N)\n"
            "        if _N07C and not trusted and delta1 <= noise_yardstick and delta2 <= noise_yardstick:\n"
            "            trusted = True\n            _n07c = True\n")
    s = rep(s, "        if trusted:\n            region._trusted = True\n            region._declared_resolution = max(residual, delta1) / (1.0 - ratio)\n",
            "        if trusted:\n            region._trusted = True\n"
            "            if _n07c:  # 07c variant (N): no geometric factor\n"
            "                num_n07c_trusted += 1\n"
            "                region._declared_resolution = max(residual, delta1 + delta2)\n"
            "            else:\n"
            "                region._declared_resolution = max(residual, delta1) / (1.0 - ratio)\n")
    s = rep(s, "        nonlocal num_checks, num_untrusted_found\n        nonlocal num_edge_rule_refusals, num_range_rule_refusals\n",
            "        nonlocal num_checks, num_untrusted_found\n        nonlocal num_edge_rule_refusals, num_range_rule_refusals\n"
            "        nonlocal num_n07c_trusted  # 07c variant (N)\n")
    s = rep(s, "    num_edge_rule_refusals = 0\n    num_range_rule_refusals = 0\n\n    def _solve(",
            "    num_edge_rule_refusals = 0\n    num_range_rule_refusals = 0\n    num_n07c_trusted = 0  # 07c variant (N)\n\n    def _solve(")
    s = rep(s, "        \"num_range_rule_refusals\": int(num_range_rule_refusals),\n",
            "        \"num_range_rule_refusals\": int(num_range_rule_refusals),\n        \"num_n07c_trusted\": int(num_n07c_trusted),  # 07c variant (N)\n")
    return s


def patch_S(s):
    # state, beside the window stop's
    s = rep(s, "    window_counts = {}\n    windows_stopped = {}\n",
            "    window_counts = {}\n    windows_stopped = {}\n"
            "    _s07c_root = {}  # 07c variant (S): leaf seq -> the outermost refused leaf above it (or itself)\n"
            "    _s07c_nu = {}  # 07c variant (S): leaf seq -> (delta1, delta2, nu_B) of its trust check\n"
            "    _s07c_count = {}  # 07c variant (S): root -> bisections counted\n"
            "    _s07c_stopped = set()  # 07c variant (S): roots stopped\n"
            "    _s07c_n = {\"stops\": 0, \"stopped_leaves\": 0, \"counted\": 0}  # 07c variant (S)\n\n"
            "    def _s07c_noise(seq):  # 07c variant (S): at the noise level, from the leaf's own trust check\n"
            "        if seq not in _s07c_nu:\n"
            "            return False\n"
            "        d1_, d2_, nu_ = _s07c_nu[seq]\n"
            "        return bool(d1_ <= _S07C[1] * nu_ and d2_ <= _S07C[1] * nu_)\n\n"
            "    def _s07c_stop_leaf(leaf):  # 07c variant (S): floor-limited as a stopped window's leaf\n"
            "        if leaf.region._phase_limited:\n"
            "            return\n"
            "        _floor_limit_stopped_window_leaf(leaf)\n"
            "        _s07c_n[\"stopped_leaves\"] += 1\n")
    # the check: record the differences and the yardstick; root a refused leaf; stop a noise-level leaf of a stopped subtree
    s = rep(s, "        _remove_error(leaf)\n        region._delta1 = delta1\n",
            "        _s07c_nu[leaf.seq] = (delta1, delta2, noise_yardstick)  # 07c variant (S)\n"
            "        if refused_07c:  # 07c variant (S): a refused leaf roots a subtree, unless one above it does\n"
            "            _s07c_root.setdefault(leaf.seq, leaf.seq)\n"
            "        _remove_error(leaf)\n        region._delta1 = delta1\n")
    s = rep(s, "            if refused:\n                trusted = False\n                if rule == \"edge\":\n",
            "            if refused:\n                refused_07c = True  # 07c variant (S)\n                trusted = False\n                if rule == \"edge\":\n")
    s = rep(s, "        # The rate rules (prompt 07b section 1.1): on a Levin leaf the test above trusts, the edge rule\n",
            "        refused_07c = False  # 07c variant (S)\n"
            "        # The rate rules (prompt 07b section 1.1): on a Levin leaf the test above trusts, the edge rule\n")
    s = rep(s, "        leaf.checked = True\n        num_checks += 1\n        _push_candidate(leaf)\n",
            "        leaf.checked = True\n        num_checks += 1\n        _push_candidate(leaf)\n"
            "        if _s07c_root.get(leaf.seq) in _s07c_stopped and _s07c_noise(leaf.seq):  # 07c variant (S): made after the stop\n"
            "            _s07c_stop_leaf(leaf)\n")
    # the bisection: children inherit the root; count; stop
    s = rep(s, "        _remove_leaf(leaf)\n        leaf.dataL = None\n        leaf.dataR = None\n        leaf.grandchildren = None\n"
            "        _add_leaf(left)\n        _add_leaf(right)\n        if stop_window is not None:\n            _stop_window(stop_window)\n",
            "        _s07c_r = _s07c_root.get(leaf.seq)  # 07c variant (S)\n"
            "        if _s07c_r is not None:\n"
            "            _s07c_root[left.seq] = _s07c_r\n"
            "            _s07c_root[right.seq] = _s07c_r\n"
            "        _remove_leaf(leaf)\n        leaf.dataL = None\n        leaf.dataR = None\n        leaf.grandchildren = None\n"
            "        _add_leaf(left)\n        _add_leaf(right)\n        if stop_window is not None:\n            _stop_window(stop_window)\n"
            "        if _s07c_r is not None and _s07c_noise(leaf.seq):  # 07c variant (S): a noise-level bisection below a refused leaf\n"
            "            _s07c_n[\"counted\"] += 1\n"
            "            _s07c_count[_s07c_r] = _s07c_count.get(_s07c_r, 0) + 1\n"
            "            if _s07c_count[_s07c_r] >= _S07C[0] and _s07c_r not in _s07c_stopped:\n"
            "                _s07c_stopped.add(_s07c_r)\n"
            "                _s07c_n[\"stops\"] += 1\n"
            "                for _m in [m for m in leaves.values() if _s07c_root.get(m.seq) == _s07c_r and _s07c_noise(m.seq)]:\n"
            "                    _s07c_stop_leaf(_m)\n")
    s = rep(s, "        \"num_range_rule_refusals\": int(num_range_rule_refusals),\n",
            "        \"num_range_rule_refusals\": int(num_range_rule_refusals),\n"
            "        \"num_s07c_stops\": int(_s07c_n[\"stops\"]),  # 07c variant (S)\n"
            "        \"num_s07c_stopped_leaves\": int(_s07c_n[\"stopped_leaves\"]),  # 07c variant (S)\n"
            "        \"num_s07c_counted\": int(_s07c_n[\"counted\"]),  # 07c variant (S)\n")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--s", nargs=2, type=float, default=None)
    ap.add_argument("--e", nargs=2, default=None)
    ap.add_argument("--n", action="store_true")
    ap.add_argument("--instr", action="store_true")
    ap.add_argument("--basis", required=True)
    a = ap.parse_args()
    out = HERE / "variants" / f"levin_quadrature_07c_{a.name}.py"
    meta = ROOT / "state" / "variants" / f"{a.name}.json"
    for p in (out, meta):
        if p.exists():
            raise SystemExit(f"{p} exists")
    src = TREE_LQ.read_text()
    if hashlib.sha256(src.encode()).hexdigest() != TREE_SHA:
        raise SystemExit("the tree's module has another SHA-256")
    if a.instr:
        ip = HERE / "variants" / "levin_quadrature_07c_instr.py"
        reg = json.loads((ROOT / "out" / "variants" / "variants.json").read_text())["variants"]["instr"]
        src = ip.read_text()
        if hashlib.sha256(src.encode()).hexdigest() != reg["sha256"]:
            raise SystemExit("the instrument has another SHA-256")
        src = src.replace('_VARIANT_07C = "instr"  # 07c instrument: its name\n', "", 1)
    opts = {"s": (int(a.s[0]), float(a.s[1])) if a.s else None, "e": tuple(a.e) if a.e else None, "n": a.n}
    if opts["e"] and (opts["e"][0] not in ("gate", "all") or opts["e"][1] not in ("nest3", "tail_sum4")):
        raise SystemExit("--e SCOPE (gate|all) ESTIMATE (nest3|tail_sum4)")
    s = patch_common(src, a.name, opts)
    if opts["e"]:
        s = patch_E(s)
    if opts["n"]:
        s = patch_N(s)
    if opts["s"]:
        s = patch_S(s)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(s)
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps({"name": a.name, "options": opts, "instrumented": a.instr, "basis": a.basis, "sha256": hashlib.sha256(s.encode()).hexdigest(),
                                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=1))
    print(f"wrote {out} sha256 {hashlib.sha256(s.encode()).hexdigest()}")


if __name__ == "__main__":
    sys.exit(main())
