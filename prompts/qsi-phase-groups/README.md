# QuadSourceIntegral phase groups — a one-prompt campaign

**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)

## 1. Why this exists

`ComputeTargets/QuadSourceIntegral.py`'s `_three_bessel_Levin` (`:1210-1442`) makes **eight**
`adaptive_levin_sincos` calls whose phase is a signed sum of three Bessel phases. All eight pass

```python
theta={"theta": phaseN, "theta_mod_2pi": phaseN_mod_2pi}
```

— two keys. No `theta_deriv`, so `need_theta_Cheb` is `True` there
(`AdaptiveLevin/levin_quadrature.py:1038`) and Levin obtains \(\theta'\) by spectral
differentiation of a raw phase whose absolute resolution is \(\varepsilon\theta\). No
`theta_abserr`, so the reported `abserr` cannot see the phase construction's own error at all.

Both `phaseN` and `phaseN_mod_2pi` are built by summing three independently reconstructed phases —
`raw_theta` values in one and `theta_mod_2pi` values in the other. That is exactly the construction
`prompts/transfer-remedial`'s prompt 07 replaced in the sibling module
`LiouvilleGreen/three_bessel_integrals.py`, where it measured the group phase error at
**2.86e-5 rad** against **1.42e-13** for the \(Kt+C+R(t)\) assembly at exact resonance — a factor
\(2.0\times10^{8}\) — and the group log-derivative at 3.05e-5 against 1.00e-12.

**The file is already internally inconsistent about this.** 170 lines earlier, `:1043` passes
`group.levin_theta(include_deriv=LEVIN_USE_THETA_DERIV)` with `LEVIN_USE_THETA_DERIV = True`
(`:93`), and the file's own comment at `:74` notes these eight calls are the odd ones out.

### Why it was not fixed by either campaign that touched the file

- **`prompts/source-remediation`** rewrote this file wholesale (its prompts 08–10) and is complete
  at 13/13. It never had an item for this: the only thing its board records against these call
  sites is **B6**, "`analytic_integral` ignores its `atol`/`rtol`", which is tolerance forwarding
  alone. The missing derivative and the missing declared error appear nowhere in its audit.
- **`prompts/transfer-remedial`** identified the gap in its planning pass
  (`RECONCILIATION.md` §3.2) but is forbidden `ComputeTargets/` production code by its own
  README §1.1 and §4.2. Its prompt 09 handed the finding to `source-remediation`'s §3 — which by
  then was already complete, so the entry landed on a board with no prompt left to discharge it.

So the issue is real, unfixed, and ownerless. This campaign owns it.

### Why it matters beyond tidiness

`_three_bessel_Levin` is not a diagnostic. It is reached from `analytic_integral` (`:865`), whose
result is stored as the **`analytic_rad` column** on `QuadSourceIntegral` (`:917`, property at
`:1945`) for every work item. `analytic_rad` is also **acceptance oracle 1** of the
`source-remediation` campaign — `ComputeTargets/tests/test_quadsource_integral.py`'s
`TestAnalyticOracle` scores `total` against it — so the accuracy of this phase input gates that
campaign's headline test.

## 2. Scope

**In scope**

- `ComputeTargets/QuadSourceIntegral.py` — `_three_bessel_Levin`'s phase assembly, and two stale
  comments in the same file that are about the quantity being changed.
- `LiouvilleGreen/three_bessel_integrals.py` — making the existing `_PhaseGroup` importable under
  a public name. **No change to its behaviour.**
- `ComputeTargets/tests/test_quadsource_integral.py` and
  `LiouvilleGreen/tests/test_three_bessel.py` — tests.

**Out of scope, and none of it is to be touched**

- `ComputeTargets/phase_groups.py`. It looks like the same thing and is not. It composes
  *cosmological* phases — `phase_spline` objects over \(\log(1+z)\) — which have no analytic
  leading term to split off, which is precisely why `prompts/transfer-remedial`'s README §1.1
  excludes "general cosmological transfer-function and Green's-function stored phases": the Bessel
  leading term \(x\) is special. Its `PhaseGroup` legitimately sums `raw_theta` values. Do not
  "fix" it and do not route the Bessel case through it.
- `AdaptiveLevin/`, `Datastore/`, `LiouvilleGreen/bessel_phase.py`, `LiouvilleGreen/phase_spline.py`,
  `thirdparty/`, `main.py`, any `extract_*.py`.
- `DEFAULT_3BESSEL_CHEBYSHEV_ORDER` and `CHEBYSHEV_ORDER`. Both are live open issues
  (`[08-3bessel-chebyshev-order-is-now-the-limit]`); neither is this campaign's.

## 3. The prompt

| # | Prompt | Covers | Model |
|---|---|---|---|
| 01 | [Three-Bessel Levin phase groups](01-three-bessel-levin-phase-groups.md) | `[transfer-remedial-qsi-phase-groups]`, `[05-quadsource-order-check-docstring-stale]` | Opus |

## 4. Rules

The invariants in the repository's [`CLAUDE.md`](../../CLAUDE.md) apply unchanged: one commit for
the prompt; a log at `logs/01-<name>.md` classifying every deviation as `STRUCTURALLY REQUIRED`,
`IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`; `IMPLEMENTATION_STATE.md` and
[`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) updated in that same commit; and no fixing of
things the prompt did not ask for — record those in the log's "Observations not acted on" and open
a §3 issue instead.

Two conventions this campaign inherits and must not "correct":

- The Bessel convention is \(J_\nu=A_\nu\sin\theta_\nu\), \(Y_\nu=-A_\nu\cos\theta_\nu\) with
  \(\theta\) increasing in \(x\), zero-point \(c_\nu=\pi/4-\pi\nu/2\).
- \(a_0\) is absorbed, never "set to 1".

### 4.1 Log format

As `prompts/transfer-remedial/README.md` §5.1: front matter with **Prompt**, **Commit**, **Model**,
**Date**, **Result**; then `## What shipped`, `## Deviations from the prompt`,
`## Verification performed` (quote the numbers, not pass/fail), `## Observations not acted on`,
`## State handed to the next prompt`.

## 5. Acceptance

The campaign is done when prompt 01's row is ✅ or ⚠️, every `adaptive_levin_sincos` call in
`QuadSourceIntegral.py` supplies all four phase keys, the improvement is **measured** rather than
argued, and `ComputeTargets/tests` and `LiouvilleGreen/tests` both pass.
