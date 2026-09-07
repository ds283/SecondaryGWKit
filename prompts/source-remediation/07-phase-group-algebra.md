# Prompt 07 — Phase-group decomposition of the source integrand (A4, part 1)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** §0.2 A4; `QI-report.md` QI-5, QI-6 and its §1 rows R17, R20, R21;
`docs/resonance-scaffolding/sigw-resonance-reconciliation.md` §3.1 ("Tier 1")
**Depends on:** 05 (consumes `TkSourceFunctions`). Read `logs/05-tk-source-functions.md` §"State
handed to the next prompt" for the final field names.
**Recommended model:** Fable. This prompt realises the design in README §6; an algebra or sign
error here propagates into every `QuadSourceIntegral` value and would be invisible in a fixed-$w$
oracle comparison only if the oracle were built the same wrong way, which is why the tests below
are built two independent ways.
**Files you may touch:** new `ComputeTargets/phase_groups.py`, `ComputeTargets/__init__.py`
(export), new `ComputeTargets/tests/test_phase_groups.py`, new
`ComputeTargets/tests/sympy_phase_groups.py` (a re-runnable derivation script), plus the log and
the status board. **Do not** touch `QuadSourceIntegral.py` (prompt 08) or anything in
`AdaptiveLevin/`/`LiouvilleGreen/`.

Read first: README §6; `QuadSource.source_function` (`QuadSource.py:23-51`);
`_three_bessel_Levin` (`QuadSourceIntegral.py:461-730`) — the analytic branch already does the
four-group construction and is the template for phase composition and error handling;
`adaptive_levin_sincos`'s docstring (`AdaptiveLevin/levin_quadrature.py:2707-2800`) for the
`f=[f_sin, f_cos]`, `theta={"theta", "theta_mod_2pi", "theta_deriv"}` contract;
`GkSourceFunctions` (`GkSourcePolicyData.py:19-60`) for the Green's-function side.

---

## 1. Character of this commit

A pure-function module, fully unit-tested, that turns "which of $G$, $T_q$, $T_r$ is oscillatory
here" plus the representations of those three factors into a list of **phase groups**, each being
exactly the `(f_sin, f_cos, theta, theta_mod_2pi, theta_deriv)` bundle one `adaptive_levin_sincos`
call needs, such that

$$\frac{\bar G_k(z,z')\,f(z'\mid q,r)}{H(z')^2} \;=\; \sum_{\rm groups}\big[f_{\sin}(z')\sin\Psi(z') + f_{\cos}(z')\cos\Psi(z')\big]$$

pointwise, for every row of the phase-group table. No integration happens in this module; prompt
08 does that. The module must not import Ray or the datastore.

## 2. Definitions (fix these in the module docstring)

- $D \equiv (1+z)\,d/dz$. The source kernel (`source_function`, verified identical to spec 03 R22
  by audit QS-1) is, with $\alpha = (5+3w_0)/(3(1+w_0))$, $\beta = 2/(3(1+w_0))$,
  $w_0 = $ `wBackground(z')`:
  $$f = \alpha\,T_qT_r + \beta\big[-DT_q\,T_r - DT_r\,T_q + DT_q\,DT_r\big].$$
  (Check this against `source_function` symbolically in the sympy script — it must be the *same*
  function, since prompt 08 will use `source_function` for the all-smooth region and this module
  everywhere else, and the two must agree at region boundaries.)
- A **smooth factor** is given by callables for its value and its $D$-derivative:
  $T_i(z')$, $DT_i(z') = (1+z')\,\texttt{dT\_dz}$; for $G$: $G(z')$ only.
- An **oscillatory factor** $T_i = M_i\sin\theta_i$ is given by `TkSourceFunctions` fields:
  $M_i(z')$, $d\ln M_i/dz$, `phase` (a `phase_spline`), $\omega_i = d\theta_i/dz$. Then
  $$DT_i = a_i\sin\theta_i + b_i\cos\theta_i,\qquad a_i = (1+z')\,M_i\,\frac{d\ln M_i}{dz},\quad b_i = (1+z')\,M_i\,\omega_i .$$
- The oscillatory Green's function is $G = A_G\sin\theta_G$ with $A_G =$ `GkSourceFunctions.sin_amplitude`,
  $\theta_G =$ `GkSourceFunctions.phase` (cos amplitude is identically zero:
  `GkSourcePolicyData.py:676`, `GkWKBIntegration.py:409`).
- All callables take $\log(1+z')$ (`z_is_log=True` / `x_is_log=True`), because that is the Levin
  integration variable in `QuadSourceIntegral`. The measure factor $1/H(z')^2$ is folded into every
  amplitude; the $(1+z_{\rm resp})$ prefactor is **not** (prompt 08 applies it, as the current
  code does at `:975/:1045/:1167`).

## 3. The algebra to implement (and to verify, not trust)

Write $S_i = \sin\theta_i$, $C_i = \cos\theta_i$.

**Both $T$ oscillatory.** Expanding $f$ in the basis $\{S_q,C_q\}\times\{S_r,C_r\}$:

$$
\begin{aligned}
c_{SS} &= \alpha M_qM_r + \beta\,(-a_qM_r - a_rM_q + a_qa_r), &
c_{SC} &= \beta\,(-b_rM_q + a_qb_r),\\
c_{CS} &= \beta\,(-b_qM_r + b_qa_r), &
c_{CC} &= \beta\,b_qb_r .
\end{aligned}
$$

Product-to-sum with $\Theta_\pm = \theta_q\pm\theta_r$:

$$f = P_+\cos\Theta_+ + Q_+\sin\Theta_+ + P_-\cos\Theta_- + Q_-\sin\Theta_-,$$
$$P_+ = \tfrac12(c_{CC}-c_{SS}),\quad P_- = \tfrac12(c_{CC}+c_{SS}),\quad Q_+ = \tfrac12(c_{SC}+c_{CS}),\quad Q_- = \tfrac12(c_{SC}-c_{CS}).$$

**One $T$ oscillatory** ($T_r$ smooth): $f = c_S S_q + c_C C_q$ with
$c_S = \alpha M_qT_r + \beta(-a_qT_r - DT_r\,M_q + a_q\,DT_r)$, $c_C = \beta\,b_q\,(DT_r - T_r)$.
(Symmetric formulas for $T_q$ smooth.)

**Multiplying by $G$.**
- $G$ smooth: each group keeps its phase; amplitudes are multiplied by $G/H^2$:
  for a group $P\cos\Theta + Q\sin\Theta$, `f_cos = G P/H²`, `f_sin = G Q/H²`.
- $G = A_G\sin\theta_G$: each group splits in two, using
  $\sin\theta_G\cos\Theta = \tfrac12[\sin(\theta_G+\Theta)+\sin(\theta_G-\Theta)]$ and
  $\sin\theta_G\sin\Theta = \tfrac12[\cos(\theta_G-\Theta)-\cos(\theta_G+\Theta)]$:
  $$\Psi = \theta_G+\Theta:\; f_{\sin} = \tfrac12 A_GP/H^2,\ f_{\cos} = -\tfrac12 A_GQ/H^2;\qquad
    \Psi = \theta_G-\Theta:\; f_{\sin} = \tfrac12 A_GP/H^2,\ f_{\cos} = +\tfrac12 A_GQ/H^2 .$$
- $G$ oscillatory, both $T$ smooth: one group, $\Psi = \theta_G$, `f_sin = A_G f/H²`, `f_cos = 0`
  — this is the current `WKB_Levin_integral` integrand (`QuadSourceIntegral.py:1099-1105`) and
  must reproduce it exactly.

That gives 1, 2 or 4 groups as in README §6. **Derive all of this yourself in
`tests/sympy_phase_groups.py`** — start from `source_function` and the LG forms, let sympy do the
product-to-sum, and assert that the module's coefficient functions equal the sympy result
identically. If your derivation disagrees with the formulas above, the formulas above are wrong:
say so in the log as STRUCTURALLY REQUIRED and ship the derived ones.

## 4. Phase composition

Each composed phase $\Psi = \theta_G \pm \theta_q \pm \theta_r$ must supply the three callables the
Levin driver uses:

- `theta` (raw): sum of the constituents' `phase.raw_theta(x, x_is_log=True)` with signs. This is
  what the analytic branch does (`:505-512`) and is used by the driver for the total-variation gate
  and derivative estimates; absolute precision loss at $|\theta|\sim10^7$ is acceptable there.
- `theta_mod_2pi`: signed sum of the constituents' `theta_mod_2pi(...)`. The sum lies in
  $(-6\pi, 6\pi)$; `sin`/`cos` are periodic so no re-reduction is needed for evaluation — this is
  again exactly what `:514-528` does. Do **not** sum `raw_theta` and reduce; that throws away the
  precision the `(div, mod)` split exists to protect (reconciliation document §3.1, last paragraph).
- `theta_deriv`: signed sum of $d\theta_i/d\log(1+z')$. For the $T$ factors use the closed-form
  $\omega_i\cdot(1+z')$ from `TkSourceFunctions.omega`; for $G$ use
  `phase.theta_deriv(x, x_is_log=True, log_derivative=True)` as `WKB_Levin_integral` defines
  `Levin_deriv` (`:1113-1120`). Read the long comment at `:1128-1145` about why `theta_deriv` is
  currently *not* passed to the driver; this module must *provide* it, and prompt 08 decides
  whether to pass it. Provide it.

Represent a group as a small frozen dataclass `PhaseGroup(label, f_sin, f_cos, theta,
theta_mod_2pi, theta_deriv, signs)` where `signs` records $(\pm_G, \pm_q, \pm_r)$ with `0` for a
factor that is smooth. The `label` should read like `"G+q-r"` for diagnostics/metadata.

## 5. Public API

```python
def build_phase_groups(regime, *, Gk, Tq, Tr, model_functions, w_background) -> list[PhaseGroup]
```

with `regime` a tuple of three flags `(G_osc, q_osc, r_osc)` and `Gk`, `Tq`, `Tr` the
`GkSourceFunctions` / `TkSourceFunctions` (or, for a smooth `Tq`/`Tr`, the numeric-region
accessors of the same object; for a smooth `Gk`, its `numeric_Gk`). Also export

```python
def evaluate_sum(groups, log_z) -> float
```

which evaluates $\sum[f_{\sin}\sin\Psi_{\bmod} + f_{\cos}\cos\Psi_{\bmod}]$ at a point — used by the
tests and by prompt 08's region-boundary consistency check, never by the integrator.

The regime `(False, False, False)` must raise: prompt 08 handles the all-smooth case with
`source_function` and ordinary quadrature, and this module must not offer a second path for it.

## 6. Tests — two independent oracles

`ComputeTargets/tests/test_phase_groups.py`, with the constant-$w$ stand-in model and exact LG
fixtures of prompt 05's test (import the fixture builders from `test_tk_source_functions.py` or a
shared `tests/fixtures.py`; do not duplicate 100 lines).

**Oracle 1 — pointwise identity against the direct product.** For each of the seven oscillatory
regimes, at ~200 random $z'$ in the appropriate region, compare `evaluate_sum(groups, log_z)` with
$G\cdot f/H^2$ where $f$ is `source_function` evaluated on $T_i = M_i\sin\theta_i$,
$T_i' = dT_i/dz$ reconstructed from $(M_i, d\ln M_i/dz, \theta_i, \omega_i)$, and $G$ from
$A_G\sin\theta_G$ or the smooth value. Require agreement to $10^{-12}$ relative to
$\max(|G f/H^2|, \text{envelope})$. This checks the algebra and the sign bookkeeping, and is
independent of any Bessel function.

**Oracle 2 — the exact Bessel fixture.** With $T$ from `bessel_phase(1.5+b)` and $G$ from
`bessel_phase(0.5+b)` (README §2 (c): $G_{\rm code}(z') = H(z')\tfrac{\pi}{2}\sqrt{\eta\eta'}\,
m(k\eta)m(k\eta')\sin(\theta(k\eta')-\theta(k\eta))$ — derive and assert this form against
`analytic_Gk.compute_analytic_G` first, to $10^{-10}$), compare `evaluate_sum` in the all-oscillatory
regime against `compute_analytic_G · source_function(analytic T) / H²` directly, to $10^{-8}$
relative-to-envelope. This checks that the fixtures, the phase conventions (sign of $\theta$, the
constant offset in $G$'s phase) and the module agree end to end. Note `bessel_phase` returns
$J_\nu = m\sin\vartheta$, $Y_\nu = -m\cos\vartheta$ (`bessel_phase.py:274-282`; audit QI §3 note 6).

**Phase composition.** For a synthetic pair of `phase_spline`s with known analytic phases,
assert that the composed `theta_mod_2pi` agrees with $(\Psi \bmod 2\pi)$ of the exact composed
phase to $10^{-10}$ at $|\Psi| \sim 10^6$, where summing `raw_theta` and reducing would lose
~$10^{-10}\cdot10^6$; and that `theta_deriv` agrees with a finite difference of the exact $\Psi$.

**Boundary consistency.** At a $z'$ just below the hand-over of one factor, `evaluate_sum` of the
one-oscillatory regime and `source_function`·$G/H^2$ with that factor evaluated from its *numeric*
spline agree to the spline's accuracy (~$10^{-6}$ at 3 e-folds sub-horizon, prompt 06 §3.4 gives
the number). This is the seam prompt 08 will stitch across.

Run every test for $w=1/3$ and $w=0.2$.

## 7. Verification

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` passes.
- `PYTHONPATH=. ./venv/bin/python ComputeTargets/tests/sympy_phase_groups.py` prints zero
  residuals for every coefficient.
- In the log, quote the measured maxima for Oracles 1 and 2 per regime, in a table.

## 8. Log and commit

Log to `logs/07-phase-group-algebra.md`. **State handed to the next prompt:** the `PhaseGroup`
fields, the `build_phase_groups` signature, how a smooth factor is passed, and the measured
Oracle-2 floor (prompt 08 needs it as an acceptance threshold). Board: row 07, item A4 (1/3). One
commit; body states the number of groups per regime and that the module is pure and untested
against a live pipeline.
