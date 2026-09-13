# Prompt 17 — Does one absolute tolerance cover the production $k$-grid? (measurement only)

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Covers:** §3 `[12-tk-numeric-atol-largest-k-excursion]`
**Review sections:** §12.5 (the `atol` mis-scaling and its measured table); §10.1 for the $G_k$
comparison. No new findings.
**Design facts:** README §2 (d) — **the floors are not targets**; §6's error definitions.
**Depends on:** 12 (which set the constant and opened the issue).
**Recommended model:** Opus — no production code, but the reference construction decides whether
the whole measurement means anything.
**Files you may touch:** new `docs/gktk-remedial/tk_numeric_atol_sweep.py`, new
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`, plus the log, the status board and
`docs/OPEN_ISSUES.md`.
**Do not touch — this prompt changes no production code and no test:** `config/defaults.py` (the
constant stays at `1e-13` whatever you measure), `main.py`, anything under `ComputeTargets/` or
`Quadrature/` (including `ComputeTargets/tests/test_tk_numeric_atol.py`, whose thresholds stay as
prompt 12 shipped them even if your sweep shows a $k$ where they would fail), the Datastore
factories, and everything README §5 rule 8 lists.

Read first: README §2 (d), §6 (the error definitions, especially **envelope-relative**);
`IMPLEMENTATION_STATE.md`'s `[12-tk-numeric-atol-largest-k-excursion]` entry in §3 and row M22;
`logs/12-tk-numeric-atol.md` — its Verification section and "State handed to the next prompt";
review §12.5; `ComputeTargets/tests/wkb_reference.py` (prompt 01's fixtures and error helpers).

---

## 1. The question, and why it is worth a prompt

Prompt 12 gave the transfer function's numeric run its own `atol = 1e-13`, measured at
$k=10^6$ on `RadiationModel`: $\delta T/\mathrm{env}$ falls $9.928\times10^{-6}\to2.534\times10^{-6}$
for $+14.3\,\%$ evaluations, meeting README §6's $\le3\times10^{-6}$. It then found, on a $k$-sweep
of the same control, that **at $k=3\times10^8$ an isolated excursion of $2.56\times10^{-4}$ of the
envelope survives near $x\approx10.8$**, and that `atol = 1e-16` removes it at *fewer* evaluations.

So README §6's $T_k$ row is demonstrated at one $k$ and unknown at the rest, and the largest
production $k$ is ~85× over target on the one control where anyone has looked. Two things make
that inconclusive rather than damning: it is an *isolated* excursion at one $x$, not a raised
error level; and it is on the radiation control, whose exact $T$ makes an envelope-relative
measure sharp in a way it need not be on a real background.

**Why now, and not as part of prompt 13.** The tolerance is part of every `TkNumericIntegration`
row's datastore key. Prompt 13 builds a fresh datastore and runs a scoped pipeline against it. If
that happens at `1e-13` and the answer is later `1e-16`, the datastore *and* prompt 13's
verification run are both invalidated. This measurement is much cheaper than the run it protects.

**You are not deciding.** Produce the numbers and a recommendation; the orchestrator stops and the
user chooses. Changing the constant, if it comes to that, is a separate prompt.

## 2. Method

### 2.1 The reference, which is the part that can invalidate everything else

There is no closed-form $T$ on `LambdaCDMModel` or `QCDModel`, so the comparison is against a
**converged run of the same integrator**, not an oracle: for each (model, $k$), integrate the
`TkNumericIntegration` RHS through `numeric_with_phase_cut` on the same grid and stop window at a
reference tolerance far tighter than any candidate, and measure the candidates against it at the
returned samples.

**Demonstrate that the reference is converged before using it.** Pick the reference pair (start
from `atol = 1e-18`, `rtol = 1e-12`), then show that tightening it again — one further decade in
each — moves the reference by at least an order of magnitude less than the smallest difference you
intend to report, at the worst $k$ of each model. Put that check in the document as its own table.
If it does not hold, say so and stop: a sweep measured against an unconverged reference measures
the reference.

On `RadiationModel` you also have the exact $3(\sin x - x\cos x)/x^3$. Report the converged run
against it as a second column, as a check on the reference construction itself.

### 2.2 The error definition

README §6: **envelope-relative**, dividing by the local Liouville–Green envelope, not by the
value. Use `wkb_reference.envelope_relative_error`, with the envelope formed as
$\mathrm{hypot}(T, T'/\omega)$ and $\omega=\sqrt{\texttt{Tk\_omegaEff\_sq}}$ at that sample; skip
samples where $\omega^2\le0$. This is model-agnostic, which is the point.

### 2.3 The sweep

- **Models:** `RadiationModel`, `LambdaCDMModel`, `QCDModel` (prompt 01).
- **Wavenumbers:** the production source grid, `np.logspace(np.log10(1e5), np.log10(3e8), 50)`
  (`main.py:3091-3099`, `NUMBER_SOURCE_K_VALUES = 50`) — `TkNumericIntegration` is one object per
  $k$ on this grid.
- **Candidates:** `atol` $\in\{10^{-10},10^{-13},10^{-16}\}$ at fixed `rtol = 1e-8`. Include
  $10^{-10}$ so the pre-prompt-12 baseline is visible across $k$, not just at $k=10^6$.
- **Geometry:** production initial data $T=1$, $T'=0$ five e-folds outside the horizon, production
  source grid, production stop window — prompt 12's geometry, so the two are comparable.

Record, per (model, $k$, `atol`): the **maximum** $\delta T/\mathrm{env}$ and the $x$ at which it
occurs; the **median** $\delta T/\mathrm{env}$ over the returned samples; the **second-largest**
per-sample value; and the RHS-evaluation count.

The median and second-largest are not padding — they are what distinguishes an isolated spike
(median at the floor, one outlier) from a genuinely raised error level (median lifted too). That
distinction is the whole question, so report it explicitly for every $k$ where the maximum exceeds
$3\times10^{-6}$.

### 2.4 The control that has to reproduce first

Before the sweep, reproduce prompt 12's two figures on `RadiationModel`: $2.534\times10^{-6}$ at
$k=10^6$, `atol = 1e-13`; and the $2.56\times10^{-4}$ excursion near $x\approx10.8$ at
$k=3\times10^8$. If either does not reproduce to a couple of significant figures, **stop and say
so** — it means the harness differs from prompt 12's and nothing downstream is comparable.

## 3. What the document must answer

`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`, additive (campaign convention: a later re-run adds
a subsection, it does not rewrite this one):

1. **Is the excursion real off the radiation control?** For each real background, the maximum
   $\delta T/\mathrm{env}$ over the production $k$-grid at `1e-13`, the $k$ and $x$ where it
   occurs, and how many of the 50 $k$ exceed $3\times10^{-6}$.
2. **Is it a spike or a level?** Per offending $k$: maximum, second-largest, median.
3. **Does `1e-16` fix it, and at what cost?** The same three numbers, plus total RHS evaluations
   summed over the grid for each `atol`. Prompt 12 found `1e-16` *cheaper* at $k=3\times10^8$ on
   the control; say whether that holds across $k$ and across models, or was a step-size accident
   at one point. **Counts, not wall time** — §5 note 14: this machine's elapsed times overstate.
4. **Where is the initial-condition floor?** README §2 (d) puts it at $2.5\times10^{-6}$ of the
   envelope with $T=1,T'=0$. Say, per model, which $k$ are at that floor and which are above it —
   an error at the floor is not fixable by any `atol` and must not be reported as if it were.
5. **A recommendation, with the evidence for it**: keep `1e-13`; move to `1e-16`; or neither is
   uniform over the grid and the constant should be $k$-dependent (say what the split would be).
   State plainly what is *not* established — in particular, whether anything here bears on the
   production hand-over, which is §1.1's and not this campaign's.

## 4. Verification and acceptance

- The script runs from the repository root, needs no Ray and no datastore, and reads prompt 01's
  fixtures — the pattern of `docs/gktk-remedial/residual_convergence.py` (prompt 02). Record its
  runtime and the command line in the document.
- `discover -s ComputeTargets/tests -t .` passes, at **299 tests, unchanged** — this prompt adds
  no test and removes none.
- `git diff HEAD~1 --stat` shows **no** production module: nothing under `ComputeTargets/`,
  `Quadrature/`, `LiouvilleGreen/`, `config/`, and not `main.py`.
- `black --check` clean on the new script.
- The reference-convergence table of §2.1 is present, and the §2.4 control reproduced.

## 5. Log and commit

Do **not** close `[12-tk-numeric-atol-largest-k-excursion]` — narrow it with the measurement and
record the recommendation; it closes when a constant is settled, which is the user's call. Update
`docs/OPEN_ISSUES.md` in the same commit (the count does not change). Update board row 17 and M22.

"State handed to the next prompt": the recommendation and the three numbers behind it; whether
prompt 13 may build its datastore at `1e-13`; the document's path.

Commit subject, or something equally specific:
`Measure the transfer-function numeric tolerance across the k-grid`.
