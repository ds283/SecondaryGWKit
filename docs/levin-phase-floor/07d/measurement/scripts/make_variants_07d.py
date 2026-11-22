"""levin-phase-floor prompt 07d Part A, section 2.5: the variants, each a copy of the tree's AdaptiveLevin/levin_quadrature.py
(07c's code, d141a80b...) with textual patches asserted to match once, written to scripts/variants/ with its diff against
the tree (out/variants/<name>.diff), its SHA-256 (out/variants/variants.json) and its basis (state/variants/<name>.json).
New in 07d.

**The estimate (E)** (section 2.3's candidates, state/candidates_07d.json): in ``_adaptive_levin_subregion_cc``, on the solves
its scope reaches, the fallback's estimate ``abserr_direct`` becomes E + kappa s_own and kappa s_own is added to every floor
term (``phase_err``, ``phase_err_roundoff``, ``phase_err_less_node``), so that the leaf's declaration
max(abserr, floor) is exactly the candidate's max(E, floor) + kappa s_own:
- s_own = sigma_own sqrt(sum w_j^2) on the fine rule, sigma_own = rms of the fine interpolant's Chebyshev coefficients past
  Chebfun's standardChop cutoff (tol 2^-52) x sqrt((n - 1)/2), or of its last 8 where standardChop does not chop;
- ``nest3``: E = |CC_{4N-3} - CC_{2N-1}|, CC_{4N-3} from the 4N-3 grid, whose even nodes are the fine grid (the 2N - 2
  odd nodes sampled anew where the nodes nest bit for bit, all 4N - 3 otherwise, counted); ``pair``: E = |CC_{2N-1} -
  CC_N| (the tree's).
- scope ``gate``: the C2 gate's solves (``_adaptive_levin_subregion_impl``'s fallback); ``all``: every Clenshaw-Curtis solve,
  the forced ones too (window pieces, guarded leaves at depth_max, through ``_oriented_cc``). Stopped leaves are the leaves
  these solves make. A window piece's estimate feeds its window's count (``_cc_leaf_unresolved`` reads ``abserr``).
**The threshold (T):** the gate's ``phase_span < SIX_PI`` reads ``_T07D_THRESHOLD``. ``SIX_PI`` keeps its value; Rule A
reads no threshold at 07c's code (``_locate_stationary_points`` reads ``SIX_PI`` in its docstring only), so nothing else
moves.
**Diagnostics** (result keys ``num_07d_*``, counted per call by module counters reset at each ``adaptive_levin_sincos``
call): ``num_07d_route_changed`` (gate decisions with SIX_PI <= V < T: solves the tree sends to Levin, the variant to the
fallback), ``num_07d_cc_new_estimate`` (fallback solves whose declaration the variant set), ``num_07d_noise_term_larger``
(of those, kappa s_own > E), ``num_07d_nest_resampled`` (solves where the grids did not nest bit for bit).
Usage: PYTHONPATH=<tree>:<scripts> python -B make_variants_07d.py
"""

import difflib
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE_LQ = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-49ce964/AdaptiveLevin/levin_quadrature.py")
TREE_SHA = "d141a80bf0484058a0d7f625926b2e283a4bb64114bc84ccfd81339a6f249be6"

VARIANTS = {
    "Eg": {"form": "nest3", "scope": "gate", "T_pi": 6, "kappa": 4.0,
           "basis": "(E-gate): nest3N4, the one zero-understatement candidate on every D-E stratum (928 scored leaves) and "
                    "on 07c's 176 truncation-dominated chase leaves; kappa 4 from D-N (sigma_own/sigma p05 0.69); margin: 0 "
                    "understatements on n scored leaves bounds the rate at about 3/n, 3/40 a stratum (3/8 for the analytic "
                    "chase at 1e-8, exhausted at 8). Gate solves only."},
    "Ea": {"form": "nest3", "scope": "all", "T_pi": 6, "kappa": 4.0,
           "basis": "(E-all): as Eg on every Clenshaw-Curtis solve, window pieces, guarded leaves at depth_max and the chase's "
                    "stopped leaves included. D-E's window and chase strata are its basis there; guarded leaves at the cap: "
                    "none in 07c's census, so no basis and none to touch."},
    "ET20": {"form": "nest3", "scope": "all", "T_pi": 20, "kappa": 4.0,
             "basis": "(ET): Ea with the threshold at 20 pi, the one grid value at which D-T finds both routes bounded under "
                      "nest3N4 - vacuously on the Levin side, since D-T's bands end at 20 pi; the Levin solve understates "
                      "on leaves of every band below it. Clenshaw-Curtis under nest3N4 understates on no leaf with V < 20 pi "
                      "(2,005 decided). Flagged: its Levin side has no measured basis above 20 pi."},
    "ET12": {"form": "nest3", "scope": "all", "T_pi": 12, "kappa": 4.0,
             "basis": "Ea with the threshold at 12 pi. **Not bounded on both sides:** the Levin solve understates on 85 of "
                      "430 decided leaves with V >= 12 pi (34 with the trust check). Run, flagged, so that the tables show a "
                      "threshold between 6 pi and 20 pi."},
    "Ep": {"form": "pair", "scope": "all", "T_pi": 6, "kappa": 4.0,
           "basis": "pairN4 on every Clenshaw-Curtis solve: the tree's pair with the noise term, no extra evaluation. **Does "
                    "not qualify** (D-E: understates on 5 scored leaves, all with 59-82 knots inside; 4 of the 176): run, "
                    "flagged with no bound basis, to show the noise term alone."},
    "T20": {"form": None, "scope": None, "T_pi": 20, "kappa": None,
            "basis": "the threshold at 20 pi under the tree's estimate. **No bound basis:** the tree's declaration "
                     "understates below every T (62 of 2,005 decided leaves with V < 20 pi). Run, flagged, for the route "
                     "change alone."},
}

P_CONST = ('SIX_PI = 6.0 * np.pi\n', None)


def patch(src, name, v):
    s = src
    head = (f'SIX_PI = 6.0 * np.pi\n\n'
            f'# prompt 07d Part A variant {name!r} (scripts/make_variants_07d.py; not production code)\n'
            f'_VARIANT_07D = {name!r}\n'
            f'_E07D_FORM = {v["form"]!r}\n_E07D_SCOPE = {v["scope"]!r}\n_E07D_KAPPA = {v["kappa"]!r}\n'
            f'_T07D_THRESHOLD = {v["T_pi"]}.0 * np.pi\n'
            f'_C07D = {{"num_07d_route_changed": 0, "num_07d_cc_new_estimate": 0, "num_07d_noise_term_larger": 0, '
            f'"num_07d_nest_resampled": 0}}\n')
    assert s.count('SIX_PI = 6.0 * np.pi\n') == 1
    s = s.replace('SIX_PI = 6.0 * np.pi\n', head)
    # the gate
    old = '''    if phase_span < SIX_PI:
        return _adaptive_levin_subregion_cc('''
    new = '''    if SIX_PI <= phase_span < _T07D_THRESHOLD:  # prompt 07d variant: a route the tree would not take
        _C07D["num_07d_route_changed"] += 1
    if phase_span < _T07D_THRESHOLD:  # prompt 07d variant (the tree: SIX_PI)
        return _adaptive_levin_subregion_cc('''
    assert s.count(old) == 1
    s = s.replace(old, new)
    # the fallback's signature and the forced callers' scope
    old = '''    id_label,
    vectorize_cache: Optional[dict] = None,
):
    """
    Bounded-cost fallback for a region'''
    new = '''    id_label,
    vectorize_cache: Optional[dict] = None,
    scope_07d: str = "gate",  # prompt 07d variant: "gate" from the C2 gate, "forced" from _oriented_cc
):
    """
    Bounded-cost fallback for a region'''
    assert s.count(old) == 1
    s = s.replace(old, new)
    i0 = s.index("def _oriented_cc(")
    i1 = s.index("def _scalar_callable(")
    body = s[i0:i1]
    for ind in ("            ", "        "):  # the ascending call (in the if) and the reversed one
        old_oc = f"{ind}notify_label,\n{ind}id_label,\n{ind}vectorize_cache=vectorize_cache,\n{ind[4:]})"
        assert body.count(old_oc) == 1, ind
        body = body.replace(old_oc, f"{ind}notify_label,\n{ind}id_label,\n{ind}vectorize_cache=vectorize_cache,\n"
                                    f"{ind}scope_07d=\"forced\",  # prompt 07d variant\n{ind[4:]})")
    s = s[:i0] + body + s[i1:]
    # the estimate, just before the fallback's return
    old = '''    return {
        "value": value_fine,
        "abserr_direct": abserr_direct,'''
    new = '''    # prompt 07d variant: the estimate with its noise term (scripts/make_variants_07d.py's docstring)
    if _E07D_FORM is not None and (scope_07d == "gate" or _E07D_SCOPE == "all"):
        nf_ = integrand_fine.shape[0]
        jj_ = np.arange(nf_)
        wt_ = np.ones(nf_)
        wt_[0] = wt_[-1] = 0.5
        ck_ = np.array([(2.0 / (nf_ - 1)) * np.sum(wt_ * integrand_fine * np.cos(np.pi * jj_ * k_ / (nf_ - 1))) for k_ in range(nf_)])
        ck_[0] *= 0.5
        ck_[-1] *= 0.5
        cut_ = _standard_chop_07d(ck_, 2.0 ** -52)
        tail_ = ck_[cut_:] if cut_ < nf_ else ck_[-8:]
        s_own_ = float(np.sqrt(np.mean(tail_ * tail_)) * np.sqrt((nf_ - 1) / 2.0)) * float(np.sqrt(np.sum(w_fine * w_fine)))
        if _E07D_FORM == "nest3":
            grid3_, _ = chebyshev_matrices(x_span, 2 * fine_order - 1)
            if np.array_equal(grid3_[::2], fine_grid):
                odd_ = grid3_[1::2]
                f_odd_ = np.vstack([_sample_vectorized(func, odd_, vectorize_cache, ("f", id(func))) for func in f])
                b_odd_ = np.array([BasisData.eval_basis_and_arguments(x)[0] for x in odd_]).T
                g3_ = np.empty(grid3_.shape[0])
                g3_[::2] = integrand_fine
                g3_[1::2] = np.sum(f_odd_ * b_odd_, axis=0)
            else:
                _C07D["num_07d_nest_resampled"] += 1
                f3_ = np.vstack([_sample_vectorized(func, grid3_, vectorize_cache, ("f", id(func))) for func in f])
                b3_ = np.array([BasisData.eval_basis_and_arguments(x)[0] for x in grid3_]).T
                g3_ = np.sum(f3_ * b3_, axis=0)
            est_ = float(np.fabs(float(np.dot(_cc_weights(x_span, 2 * fine_order - 1), g3_)) - value_fine))
        else:
            est_ = abserr_direct
        noise_ = _E07D_KAPPA * s_own_
        _C07D["num_07d_cc_new_estimate"] += 1
        if noise_ > est_:
            _C07D["num_07d_noise_term_larger"] += 1
        abserr_direct = est_ + noise_
        round_off_err = round_off_err + noise_
        round_off_err_less_node = round_off_err_less_node + noise_

    return {
        "value": value_fine,
        "abserr_direct": abserr_direct,'''
    if v["form"] is not None:
        assert s.count(old) == 1
        s = s.replace(old, new)
        # standardChop, transcribed (state/standardChop_rule.md), before the fallback
        old = "def _adaptive_levin_subregion_cc("
        chop = '''def _standard_chop_07d(coeffs, tol):
    """prompt 07d variant: Chebfun's standardChop (Aurentz and Trefethen 2017), transcribed from
    state/standardChop_rule.md of 07d Part A's records; returns the 1-based cutoff, n if it does not chop."""
    n = len(coeffs)
    if tol >= 1:
        return 1
    if n < 17:
        return n
    b = np.abs(np.asarray(coeffs, dtype=float))
    m = np.maximum.accumulate(b[::-1])[::-1].copy()
    if m[0] == 0:
        return 1
    env = m / m[0]
    plateau_point = None
    j2 = None
    for j in range(2, n + 1):
        j2 = int(round(1.25 * j + 5))
        if j2 > n:
            return n
        e1, e2 = env[j - 1], env[j2 - 1]
        r = 3 * (1 - np.log(e1) / np.log(tol)) if e1 > 0 else 0.0
        if e1 == 0 or e2 / e1 > r:
            plateau_point = j - 1
            break
    if env[plateau_point - 1] == 0:
        return plateau_point
    j3 = int(np.sum(env >= tol ** (7 / 6)))
    if j3 < j2:
        j2 = j3 + 1
        env[j2 - 1] = tol ** (7 / 6)
    cc = np.log10(env[:j2]) + np.linspace(0, (-1 / 3) * np.log10(tol), j2)
    return max(int(np.argmin(cc)) + 1 - 1, 1)


def _adaptive_levin_subregion_cc('''
        assert s.count(old) == 1
        s = s.replace(old, chop)
    # the diagnostics: counters reset per call and returned
    old = "def adaptive_levin_sincos("
    assert s.count(old) == 1
    s = s.replace(old, "def _adaptive_levin_sincos_tree_07d(")
    s += '''

def adaptive_levin_sincos(*args, **kwargs):
    """prompt 07d variant: the tree's adaptive_levin_sincos (renamed _adaptive_levin_sincos_tree_07d), with the variant's
    counters reset before the call and added to its result as num_07d_* keys."""
    for k_ in _C07D:
        _C07D[k_] = 0
    result = _adaptive_levin_sincos_tree_07d(*args, **kwargs)
    result.update(dict(_C07D))
    return result
'''
    # the round-off floor variable names the patch adds noise to must exist in the fallback
    return s


def main():
    src = TREE_LQ.read_text()
    if hashlib.sha256(src.encode()).hexdigest() != TREE_SHA:
        raise SystemExit("the tree's module has another SHA-256")
    vd = HERE / "variants"
    od = ROOT / "out" / "variants"
    sd = ROOT / "state" / "variants"
    for d in (vd, od, sd):
        d.mkdir(parents=True, exist_ok=True)
    reg = {}
    for name, v in VARIANTS.items():
        p = vd / f"levin_quadrature_07d_{name}.py"
        if p.exists():
            raise SystemExit(f"{p} exists")
        s = patch(src, name, v)
        p.write_text(s)
        diff = "".join(difflib.unified_diff(src.splitlines(True), s.splitlines(True), "tree/levin_quadrature.py", p.name))
        (od / f"{name}.diff").write_text(diff)
        (sd / f"{name}.json").write_text(json.dumps(dict(v, name=name), indent=1))
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        reg[name] = {"path": str(p), "sha256": sha, "form": v["form"], "scope": v["scope"], "T_pi": v["T_pi"], "kappa": v["kappa"]}
        print(f"{name}: {sha} (+{sum(1 for l in diff.splitlines() if l.startswith('+') and not l.startswith('+++'))}/"
              f"-{sum(1 for l in diff.splitlines() if l.startswith('-') and not l.startswith('---'))})")
    (od / "variants.json").write_text(json.dumps({"tree_sha256": TREE_SHA, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                                  "variants": reg}, indent=1))


if __name__ == "__main__":
    main()
