# Prompt 01 — Fix the `GenericEOS` perturbation sound speed (A1)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** `docs/spec-code-audit-2026-09.md` §0.2 A1, §2.1; `docs/spec-code-audit/TK-report.md` TK-1
**Depends on:** nothing
**Recommended model:** Opus (one-line physics fix, but the regression test has to be designed
so it would have caught this)
**Files you may touch:** `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`, new
`CosmologyModels/tests/__init__.py`, new `CosmologyModels/tests/test_wPerturbations.py`, plus the
log and the status board.

Read first: audit §0.2 row A1 and TK-report §2 TK-1 (the measurement table), spec 01's Tier 3
author note on $c_s^2$ (`docs/spec/01-transfer-function.md`, head block), spec 03 §0.5.

---

## Character of this commit

One line of physics and one new test module. The defect is unambiguous: the class's own comment
states the intent and the code contradicts it. Do not widen this into a `csSquared(z)` hook or any
other restructuring (README §1.1).

## The defect

`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:281-292`:

```python
    def wPerturbations(self, z: float) -> float:
        rho = self._rho_fluid(z)
        T = rho["T"]

        # perturbations w(z) includes contributions from radiation and matter, but not the cosmological constant,
        # which we take not to have perturbations. ...
        numerator = self._eos.w(T) * rho["radiation"]
        denominator = self.rho(z)          # <-- rho_m + rho_r + rho_Lambda

        return numerator / denominator
```

`self.rho(z)` (`:250-257`) is the total density including $\rho_\Lambda$. The author's convention
(spec 01 Tier 3; spec 03 §0.5) is $c_s^2 = \delta p/\delta\rho$ of the *perturbed* fluid with
$\Lambda$ unperturbed, i.e. the denominator is $\rho_m + \rho_r$. The sibling
`CosmologyModels/LambdaCDM/LambdaCDM.py:215-224` does this correctly. Measured (audit TK-1): the
code's $c_s^2$ is too small by $\times3.21$ at $z=0$, $\times1.28$ at $z=1$, $<1\%$ for $z\gtrsim5$.

## What to do

1. Change the denominator to the matter-plus-radiation density from `rho` (the dict returned by
   `_rho_fluid`), i.e. `rho["matter"] + rho["radiation"]`. Check `_rho_fluid`'s return keys before
   assuming these names. Leave `wBackground` alone — it correctly divides by the total.
2. Leave the comment, but remove the parenthetical speculation about letting $\Lambda$ cluster only
   if you judge it misleading; otherwise keep it. Record the choice.
3. **Add a regression test** `CosmologyModels/tests/test_wPerturbations.py` (unittest) with at
   least:
   - **Agreement with `LambdaCDM`.** Construct a `LambdaCDM_GenericEOS` whose EOS is pure radiation
     ($w(T)\equiv 1/3$, i.e. $g_S = g$) and a `LambdaCDM` with the same $\Omega$'s and units, and
     assert `wPerturbations(z)` agree to $10^{-10}$ relative at $z\in\{0, 0.5, 1, 2, 10, 10^3\}$.
     Look at how `LambdaCDM`, `LambdaCDM_GenericEOS` and `GenericEOS.py` are constructed
     (`CosmologyModels/GenericEOS/QCD_Test.py` and `CosmologyModels/LambdaCDM/Planck.py` show the
     parameters); if constructing a pure-radiation `GenericEOSBase` needs a small stub subclass,
     write it inside the test. Do not add a production-code helper for this.
   - **$\Lambda$-independence.** For the fixed EOS above, `wPerturbations(z)` must be unchanged
     when $\Omega_\Lambda$ is varied at fixed $\Omega_m,\Omega_r$ (that is the property the bug
     violated). `wBackground` must *change*.
   - **Limits.** `wPerturbations → 1/3` as $z\to\infty$ and `→ 0` as $z\to 0$ for a $\Lambda$CDM
     parameter set; the ratio to `wBackground` at $z\gtrsim 10^3$ within $10^{-3}$ of 1.
   Confirm the test **fails** on the unfixed code (revert locally, run, restore) and say so in the
   log with the printed values.
4. The stored `BackgroundModel` values for any `GenericEOS`/QCD model in an existing datastore are
   now stale (`wPerturbations`, `d_wPerturbations_dz`, `d2_wPerturbations_dz2`). Say so in the log
   and in `IMPLEMENTATION_STATE.md` §5; do not attempt a migration.

## Verification

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .` passes.
- Re-run `docs/spec-code-audit/scripts/TK_05_background_derivatives.py` (it prints the two
  `wPerturbations` variants side by side) and quote the $z=0$ ratio before and after.

## Log and commit

Log to `logs/01-genericeos-sound-speed.md` (template README §5.1). Update the board: row 01, item
A1, §5 note about stale datastores. One commit, message per README §5 item 2; the body should say
what the denominator was, what it is now, the measured factor at $z=0$, and which cosmology classes
are affected.
