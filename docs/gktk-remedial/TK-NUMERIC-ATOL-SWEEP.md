# Does one absolute tolerance cover the production $k$-grid for $T_k$?

**Campaign:** [`prompts/GkTk-remedial/README.md`](../../prompts/GkTk-remedial/README.md) ·
**Prompt:** [`17-tk-numeric-atol-k-sweep.md`](../../prompts/GkTk-remedial/17-tk-numeric-atol-k-sweep.md) ·
**Issue:** `[12-tk-numeric-atol-largest-k-excursion]`
**Generated:** 2026-09-12 by

```
PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py
```

in 276 s (Python 3.12.14, NumPy 2.2.4, SciPy 1.15.2). Every table below is that command's
standard output, verbatim; the prose applies the prompt's questions to it.

> **This document is additive.** A later re-run adds a new top-level section recording its own
> measurement; it does not rewrite this one, which was correct for the tree it was taken on
> (`CLAUDE.md`, campaign conventions). **Nothing here changes any code.**
> `config.defaults.DEFAULT_TK_NUMERIC_ABS_TOLERANCE` stands at `1e-13` and the decision is the
> user's.

---

## 1. The answer, in one page

Prompt 12 set the transfer function's numeric absolute tolerance to `1e-13` on a measurement at
one wavenumber, and then found — on the exact-radiation control alone — an isolated excursion of
$2.56\times10^{-4}$ of the envelope surviving at $k=3\times10^8$, which `atol = 1e-16` removed at
fewer evaluations. This sweep measures all 50 production wavenumbers on the control and on both
production backgrounds.

**1. Is the excursion real off the radiation control?** Yes, and it is not confined to the largest
$k$. At `atol = 1e-13` the maximum $\delta T/\mathrm{env}$ over the grid is
$2.5\times10^{-4}$ (RadiationModel, $k=3\times10^8$, $x=10.8$),
$8.6\times10^{-4}$ (LambdaCDMModel, $k=8.37\times10^5$, $x=17.4$) and
$2.8\times10^{-4}$ (QCDModel, $k=1.58\times10^7$, $x=10.3$).
**3 of 50, 13 of 50 and 8 of 50** wavenumbers exceed README §6's $3\times10^{-6}$ (one of the
QCD eight below its own reference noise, §4).

**2. Is it a spike or a level?** Both, in sequence. Almost every excursion above $10^{-5}$ falls
at a single $x$ between 8 and 18 — where the mode is entering the horizon, an e-fold or so above
the stop window's $z_{e3}$ edge (the one exception is QCD $k=5.86\times10^7$ at `1e-16`,
$1.8\times10^{-5}$ at $x=218$, which is also the wavenumber of largest reference drift) — but
the *median* of the same run is lifted with it, from $10^{-8}$–$10^{-7}$ in a quiet run to
$1.7$–$5.8\times10^{-5}$. Every one of the 13 LambdaCDM offenders and every one of the 3
radiation offenders is a *level* by the test in §5 (median itself above $3\times10^{-6}$). So it
is one bad step whose consequence is carried to the end of the integration, not one bad sample:
the last returned sample is still wrong by $8.6\times10^{-6}$–$2.2\times10^{-4}$ in those runs.
Four of the eight QCD offenders are spikes instead, and they are the small ones.

**3. Does `1e-16` fix it, and at what cost?** **No.** It clears the radiation control (0 of 50
above target) but on LambdaCDM it leaves **10 of 50** — two of them wavenumbers that `1e-13`
handles perfectly — with a worst of $4.8\times10^{-4}$; on QCD it leaves 4 of 50 with a worst of
$2.5\times10^{-5}$. Nor is it reliably cheaper: prompt 12's observation holds on the control (34
of 50 $k$ cheaper than `1e-13`) but not on either real background (15 of 50 each), where the grid
total is **+0.5 %** (LambdaCDM) and **+1.1 %** (QCD) *more* evaluations than `1e-13`.

**4. Where is the initial-condition floor?** At $2.52\times10^{-6}$ of the envelope, and it is
$k$-independent, because the production grid starts five e-folds outside the horizon at *every*
$k$: $x_i = c_s e^{-5} = 0.00389$ throughout. On the radiation control the measurement is exact —
the converged run with $T=1,T'=0$ differs from the closed-form $T$ by $2.52\times10^{-6}$ at every
$k$, and with exact initial data by $2.3$–$4.3\times10^{-11}$. So the floor is where README §2 (d)
puts it, no $k$ is *below* it, and no `atol` can reach past it. The sweep's own numbers are
measured against a reference carrying the same initial data, so they are solver error alone and
must be read against that floor rather than added to it.

**5. Recommendation: keep `1e-13`.** Not because it meets the target everywhere — it does not —
but because **the absolute tolerance is not the lever**, which §7 establishes directly:

* holding `atol = 1e-13` and tightening `rtol` from `1e-8` to `1e-9` removes the excursion at the
  worst wavenumber of every model ($8.6\times10^{-4}\to7.4\times10^{-8}$ on LambdaCDM) for +23 %
  evaluations;
* on both real backgrounds, changing $k$ by **one part in $10^6$** removes it as well
  ($8.6\times10^{-4}\to3.9\times10^{-7}$). An excursion that a $10^{-6}$ change in a parameter
  turns off is an accident of the step sequence, not a property of that wavenumber.

A $k$-dependent constant is therefore not supportable: the offending wavenumbers are not a
contiguous band, they differ between `1e-13` and `1e-16`, and they move when $k$ moves in the
sixth digit. What `1e-13` *does* buy is the thing prompt 12 chose it for — the error *level* — and
it buys it uniformly: the median over the 50 $k$ of the per-$k$ maximum falls
$1.25\times10^{-5}\to3.8\times10^{-7}$ (radiation), $1.19\times10^{-5}\to4.5\times10^{-7}$
(LambdaCDM) and $1.38\times10^{-5}\to1.0\times10^{-6}$ (QCD) for +25–30 % evaluations, and
`1e-16` improves none of those by as much as a factor of three.

**Prompt 13 may build its datastore at `1e-13`.** Nothing measured here would move the constant,
so the `TkNumericIntegration` row key is safe; what prompt 13's verification document should
*not* do is quote one wavenumber's $\delta T/\mathrm{env}$ as if it characterised the grid.

### 1.1 What is not established

* **Why `rtol = 1e-8` occasionally mis-steps.** §7 shows that it does and that `1e-9` does not.
  It does not identify the mechanism, and it does not measure what `rtol = 1e-9` would cost over a
  production run or what else shares that tolerance. `rtol` is one number for every integration in
  the pipeline (`main.py:2980-2998`), so moving it is not the local change `atol` was.
* **The QCD reference is not converged at every $k$** (§4). Four QCD wavenumbers carry a reference
  drift of $1.6$–$6.2\times10^{-6}$ of the envelope, so at those $k$ nothing below $10^{-5}$ is
  measurable here at all. The cause is `QCD_Cosmology`'s discontinuous $H(z)$, not the tolerance.
* **Nothing here bears on the numeric$\to$WKB hand-over.** Where the switch sits, how wide the
  overlap is and whether the response grid resolves the mode through the seam are
  `docs/OPEN_ISSUES.md` §1.1's, not this campaign's (README §0.3). The excursions happen to fall
  an e-fold above the stop window's $z_{e3}$ edge; that is where the mode enters the horizon and
  the solution first oscillates, and no more should be read into it.
* **The initial-condition floor is not addressed.** Replacing $T=1,T'=0$ by the series is
  `[00-tk-superhorizon-ic-series]` and out of scope (README §0.4). The series is used here only as
  a *measure* of the floor.

---

## 2. Method

`docs/gktk-remedial/tk_numeric_atol_sweep.py`. No Ray, no datastore, no production module touched:
`numeric_with_phase_cut` is called through its undecorated `_function` with prompt 01's stand-ins
(`ComputeTargets/tests/wkb_reference.py`), the pattern of prompt 02's
`docs/gktk-remedial/residual_convergence.py`.

* **Geometry**, identical to prompt 12's and to `main.py`'s `build_Tk_numeric_work`: production
  initial data $T=1$, $T'=0$ five e-folds outside the horizon; the production source grid, 100
  samples per decade of $z$, truncated below at $0.85z_{e6}$; the $(z_{e3}, z_{e6})$ stop window;
  `rtol = 1e-8`; `warn_unresolved_osc=False`.
* **Wavenumbers**: `np.logspace(log10(1e5), log10(3e8), 50)` — the production source grid
  (`main.py:3090-3099`). `TkNumericIntegration` is one object per $k$, so these are exactly the 50
  production objects per model.
* **Candidates**: `atol` $\in\{10^{-10}, 10^{-13}, 10^{-16}\}$, the middle one being what
  prompt 12 shipped and the first the pre-prompt-12 baseline.
* **Reference**: the *same integrator* at `(atol, rtol) = (1e-18, 1e-12)` on the same grid and
  window, per (model, $k$) — there is no closed-form $T$ on either production background.
* **Error**: README §6's envelope-relative error, $|T_{\rm cand} - T_{\rm ref}|$ divided by the
  local Liouville–Green envelope $\mathrm{hypot}(T, T'/\omega)$ of the reference run with
  $\omega = \sqrt{\texttt{Tk\_omegaEff\_sq}}$; samples with $\omega^2\le0$ (the super-horizon
  part of the grid) are skipped.
* **$x$** labels where an error falls: $x = k c_s(1+z)/H$, which is $k c_s\tau$ identically in
  exact radiation and the same quantity evaluated locally on a real background.

**What the sweep's $\delta T/\mathrm{env}$ is, and is not.** The reference carries the *same*
$T=1,T'=0$ initial data as the candidate, so the initial-condition error cancels and what is
measured is solver error alone — the quantity `atol` controls. That is deliberate, and §8 supplies
the floor the numbers have to be read against. On the radiation control the runs are *also* scored
against the exact $3(\sin x - x\cos x)/x^3$, which is the measure prompt 12 used and which
includes the floor; both columns are reported in §5.

---

## 3. The prompt 12 control reproduces

Prompt 17 §2.4: if these two do not reproduce, the harness differs from prompt 12's and nothing
downstream is comparable. Both are against the exact $T$, as prompt 12's were. The script raises
and stops if either misses by more than 3 %.

| control | prompt 12 | measured max dT/env | miss | at x | prompt 12 x | RHS evals |
|---|---|---|---|---|---|---|
| k=1e6 | 2.53e-06 | 2.534e-06 | 0.02% | 28.22 | -- | 7403 |
| k=3e8 | 0.000256 | 0.000256 | 0.01% | 10.78 | 10.8 | 8483 |

$2.534\times10^{-6}$ against prompt 12's 2.534e-6, and $2.56\times10^{-4}$ at $x=10.78$ against
its 2.56e-4 near $x\approx10.8$: 0.02 % and 0.01 %. The right-hand-side evaluation counts, 7403 and
8483, are prompt 12's as well.

---

## 4. Is the reference converged? (on two models of three)

Prompt 17 §2.1: a sweep measured against an unconverged reference measures the reference. The
check tightens both tolerances by one decade — `(1e-19, 1e-13)` — and asks whether the reference
moves by at least an order of magnitude less than the smallest difference the sweep reports.
(One decade, not two: SciPy clamps `rtol` at $2.22\times10^{-14}$, so a further decade would be
silently ignored.)

| model | worst reference drift | at k [1/Mpc] | its median | median drift over the grid | smallest candidate error reported | drift <= 1/10 of it? |
|---|---|---|---|---|---|---|
| RadiationModel | 4.21e-11 | 3e+08 | 3.54e-12 | 1.92e-11 | 1.32e-07 | yes |
| LambdaCDMModel | 5.7e-11 | 1.561e+08 | 4.66e-12 | 3.76e-11 | 3.73e-07 | yes |
| QCDModel | 6.17e-06 | 5.855e+07 | 5.63e-09 | 2.65e-08 | 3.45e-07 | **no** |

**It holds on `RadiationModel` and `LambdaCDMModel`, by three to four orders.** The reference
moves by at most $4.2\times10^{-11}$ and $5.7\times10^{-11}$ of the envelope against a smallest
reported difference of $1.3\times10^{-7}$ and $3.7\times10^{-7}$. The radiation control confirms
this against a true oracle rather than against itself: the reference with *exact* initial data
reproduces the closed-form $T$ to $2.3$–$4.3\times10^{-11}$ at every $k$ (§5, the oracle table).

**It fails on `QCDModel`, at four wavenumbers out of fifty**, where the drift reaches
$6.2\times10^{-6}$. The median drift over the QCD grid is $2.7\times10^{-8}$ and the median of the
worst-drifting run is $5.6\times10^{-9}$, so this is not a uniform failure: it is
$k=1.58\times10^7$ ($1.6\times10^{-6}$), $4.97\times10^7$ ($2.8\times10^{-6}$),
$5.86\times10^7$ ($6.2\times10^{-6}$) and $2.55\times10^8$ ($5.3\times10^{-6}$).

The cause is not the tolerance. Prompt 02 established that `QCD_EOS`'s `G(T)` and `Gs(T)` switch
between the Saikawa–Shirai fit and an asymptotic constant at fixed temperatures and the two pieces
do not match, so **`QCD_Cosmology`'s $H(z)$ jumps** — by $1.0\times10^{-4}$ relative at
$z=8.64\times10^{11}$ and $4.4\times10^{-4}$ at $z=4.25\times10^7$
(`RESIDUAL-CONVERGENCE.md` §2). A discontinuous right-hand side reduces an eighth-order
Runge–Kutta to first order across the jump, so tightening a decade shrinks the step by $10^{1/8}$
and the error by only about a quarter. Profiling the drift confirms it — a separate diagnostic
run, not part of the script's output: at $k=10^8$ the difference sits at $2$–$3\times10^{-9}$
of the envelope from the top of the grid down to $z\approx8.9\times10^{11}$, steps to
$4\times10^{-8}$ and then $6\times10^{-7}$ *immediately below* $z\approx8.6\times10^{11}$, and
persists to the end.

**What that costs this document.** At those four QCD wavenumbers nothing below about $10^{-5}$ of
the envelope is measurable, and the per-$k$ tables therefore carry the drift as its own column so
every QCD number can be read against its own noise. Three of the four are also offenders: at
$k=1.58\times10^7$ and $4.97\times10^7$ the `1e-13` excursions ($2.8\times10^{-4}$ and
$1.6\times10^{-4}$) stand 175× and 57× above their drift and are signal, while at
$k=2.55\times10^8$ the `1e-13` figure of $4.6\times10^{-6}$ is *below* its own
$5.3\times10^{-6}$ drift and is not counted as an excursion; the fourth, $k=5.86\times10^7$, is
an offender only at `1e-16` ($1.8\times10^{-5}$, 2.9× its drift).

---

## 5. The sweep

Per model: the summary over the grid; the shape of the offending runs; then every wavenumber, as
**max / 2nd-largest / median / last returned sample** of $\delta T/\mathrm{env}$, with `@x` the
position of the maximum. The last-sample column is not in the prompt's list — it is here because
it is what distinguishes one bad sample from one bad step whose consequence is carried to the
deepest point, which is the part of the run `TkWKBIntegration` reads as its initial condition.

### 5.1 RadiationModel

| atol | worst max dT/env | at k [1/Mpc] | at x | median over the 50 k of the per-k max | k with max > 3e-6 | total RHS evals over the grid |
|---|---|---|---|---|---|---|
| 1e-10 | 1.57e-05 | 1.922e+05 | 224.9 | 1.25e-05 | 50 | 322228 |
| 1e-13 | 0.00025 | 3e+08 | 10.78 | 3.8e-07 | 3 | 401677 |
| 1e-16 | 2.21e-07 | 3e+08 | 230.9 | 1.48e-07 | 0 | 381547 |
Shape of the offending runs — *level* when the median is itself above $3\times10^{-6}$, *spike* when only the maximum is:

| atol | k above 3e-6 | of those, level | of those, spike |
|---|---|---|---|
| 1e-10 | 50 | 0 | 50 |
| 1e-13 | 3 | 3 | 0 |
| 1e-16 | 0 | 0 | 0 |
Every wavenumber of the production grid, as **max / 2nd-largest / median / last sample** of $\delta T/\mathrm{env}$. `x@max` is where the maximum fell; the grid runs from $x_i\approx0.0039$ to $x\approx275$.

| k [1/Mpc] | atol 1e-10 (x@max) | atol 1e-13 (x@max) | atol 1e-16 (x@max) | ref drift | IC floor |
|---|---|---|---|---|---|
| 1e+05 | 1.5e-05 / 1.4e-05 / 3.1e-07 / 1.5e-05 @229 | 1.7e-07 / 1.6e-07 / 4.2e-09 / 1.6e-07 @218 | 1.4e-07 / 1.2e-07 / 8.2e-09 / 1.2e-07 @218 | 1.1e-11 | 2.52e-06 |
| 1.178e+05 | 1.6e-05 / 1.4e-05 / 1.8e-06 / 1.4e-05 @219 | 1.8e-07 / 1.6e-07 / 6.2e-09 / 1e-07 @219 | 1.4e-07 / 1.3e-07 / 3.7e-09 / 8.2e-08 @219 | 1.1e-11 | 2.52e-06 |
| 1.387e+05 | 1.4e-05 / 1.2e-05 / 9.6e-08 / 7.5e-06 @219 | 1.8e-07 / 1.7e-07 / 4.5e-09 / 3.7e-08 @209 | 1.3e-07 / 1.3e-07 / 3.2e-09 / 3e-08 @209 | 1.1e-11 | 2.52e-06 |
| 1.633e+05 | 1.3e-05 / 1.2e-05 / 2.7e-07 / 6e-06 @225 | 1.9e-07 / 1.7e-07 / 6e-09 / 4.2e-08 @225 | 1.4e-07 / 1.2e-07 / 5.6e-09 / 3.8e-08 @225 | 1.2e-11 | 2.52e-06 |
| 1.922e+05 | 1.6e-05 / 1.3e-05 / 2.3e-06 / 3.9e-06 @225 | 2e-07 / 1.6e-07 / 8.5e-09 / 1.1e-07 @225 | 1.5e-07 / 1.3e-07 / 1.1e-08 / 1e-07 @225 | 1.2e-11 | 2.52e-06 |
| 2.264e+05 | 1.4e-05 / 1e-05 / 6.3e-08 / 7.9e-06 @225 | 1.9e-07 / 1.9e-07 / 8e-09 / 1.7e-07 @225 | 1.4e-07 / 1.4e-07 / 8.3e-09 / 1.3e-07 @225 | 1.1e-11 | 2.52e-06 |
| 2.665e+05 | 1.4e-05 / 1.3e-05 / 8.2e-08 / 1.4e-05 @231 | 2.1e-07 / 2.1e-07 / 8.1e-09 / 2.1e-07 @231 | 1.5e-07 / 1.4e-07 / 3.9e-09 / 1.5e-07 @231 | 1.2e-11 | 2.52e-06 |
| 3.139e+05 | 1.4e-05 / 1.2e-05 / 2.2e-07 / 1.4e-05 @231 | 2.4e-07 / 2.2e-07 / 9.4e-09 / 2.4e-07 @231 | 1.5e-07 / 1.3e-07 / 3.6e-09 / 1.5e-07 @231 | 1.2e-11 | 2.52e-06 |
| 3.696e+05 | 1.4e-05 / 1.1e-05 / 5.5e-08 / 1.4e-05 @231 | 2.5e-07 / 2.2e-07 / 1.3e-08 / 2.5e-07 @231 | 1.5e-07 / 1.3e-07 / 7.3e-09 / 1.5e-07 @231 | 1.2e-11 | 2.52e-06 |
| 4.352e+05 | 1.2e-05 / 1.1e-05 / 4.4e-08 / 1.1e-05 @219 | 2.6e-07 / 2.3e-07 / 8.3e-09 / 1.7e-07 @219 | 1.4e-07 / 1.2e-07 / 7.4e-09 / 9.3e-08 @219 | 1.2e-11 | 2.52e-06 |
| 5.124e+05 | 1.1e-05 / 1e-05 / 3.6e-08 / 7.2e-06 @219 | 2.7e-07 / 2.5e-07 / 7e-09 / 1.3e-07 @219 | 1.4e-07 / 1.4e-07 / 9.5e-09 / 5.7e-08 @209 | 1.2e-11 | 2.52e-06 |
| 6.034e+05 | 1.3e-05 / 1.3e-05 / 4.4e-08 / 9.3e-06 @219 | 2.7e-07 / 2.6e-07 / 1.1e-08 / 3.8e-08 @209 | 1.3e-07 / 1.2e-07 / 4.3e-09 / 1.2e-08 @209 | 1.1e-11 | 2.52e-06 |
| 7.105e+05 | 1.3e-05 / 1.3e-05 / 2.7e-07 / 1.8e-08 @225 | 3e-07 / 2.9e-07 / 7.1e-09 / 4.2e-08 @225 | 1.4e-07 / 1.2e-07 / 9.2e-09 / 3.2e-08 @225 | 1.3e-11 | 2.52e-06 |
| 8.366e+05 | 1.2e-05 / 1.1e-05 / 2.1e-07 / 1.1e-06 @225 | 3.3e-07 / 2.6e-07 / 1.2e-08 / 1.5e-07 @225 | 1.5e-07 / 1.1e-07 / 8.2e-09 / 8e-08 @225 | 1.3e-11 | 2.52e-06 |
| 9.851e+05 | 1.2e-05 / 9e-06 / 5.4e-08 / 6.8e-06 @225 | 3.7e-07 / 3e-07 / 2.3e-08 / 2.2e-07 @225 | 1.4e-07 / 1.2e-07 / 4.1e-09 / 1.1e-07 @225 | 1.4e-11 | 2.52e-06 |
| 1.16e+06 | 1.3e-05 / 1.1e-05 / 7.4e-07 / 9.7e-06 @225 | 3.2e-07 / 3e-07 / 1.4e-08 / 2.9e-07 @225 | 1.3e-07 / 1.3e-07 / 6.2e-09 / 1.3e-07 @215 | 1.3e-11 | 2.52e-06 |
| 1.366e+06 | 1.2e-05 / 1.1e-05 / 6.7e-08 / 1.2e-05 @231 | 3.2e-07 / 3.2e-07 / 1e-08 / 3.2e-07 @231 | 1.5e-07 / 1.4e-07 / 8.9e-09 / 1.5e-07 @231 | 1.4e-11 | 2.52e-06 |
| 1.608e+06 | 1.4e-05 / 1.4e-05 / 3.3e-07 / 1.4e-05 @231 | 3.2e-07 / 3e-07 / 1.2e-08 / 3.2e-07 @231 | 1.5e-07 / 1.4e-07 / 7.6e-09 / 1.5e-07 @231 | 1.5e-11 | 2.52e-06 |
| 1.894e+06 | 1.5e-05 / 1.4e-05 / 2.2e-06 / 1.5e-05 @231 | 3.4e-07 / 3e-07 / 1.5e-08 / 3.4e-07 @231 | 1.5e-07 / 1.3e-07 / 8.2e-09 / 1.5e-07 @231 | 1.6e-11 | 2.52e-06 |
| 2.23e+06 | 1.4e-05 / 1.1e-05 / 4.5e-08 / 1.4e-05 @232 | 3.3e-07 / 3.1e-07 / 2e-08 / 3.3e-07 @232 | 1.4e-07 / 1.3e-07 / 3.8e-09 / 1.4e-07 @232 | 1.4e-11 | 2.52e-06 |
| 2.626e+06 | 1.2e-05 / 9.8e-06 / 5.6e-08 / 7.2e-06 @219 | 3.5e-07 / 3.3e-07 / 1.5e-08 / 1.6e-07 @219 | 1.4e-07 / 1.3e-07 / 3.4e-09 / 5.8e-08 @219 | 1.6e-11 | 2.52e-06 |
| 3.092e+06 | 1.3e-05 / 1.3e-05 / 2.1e-07 / 4.2e-06 @219 | 3.5e-07 / 3.3e-07 / 1.8e-08 / 4.6e-08 @209 | 1.3e-07 / 1.3e-07 / 4.2e-09 / 1.8e-08 @209 | 1.6e-11 | 2.52e-06 |
| 3.64e+06 | 1.2e-05 / 1.1e-05 / 6.4e-08 / 8.8e-07 @224 | 3.5e-07 / 3.4e-07 / 2.1e-08 / 2e-08 @210 | 1.3e-07 / 1.3e-07 / 5e-09 / 2.2e-08 @224 | 1.7e-11 | 2.52e-06 |
| 4.287e+06 | 1.1e-05 / 1e-05 / 5.2e-08 / 3.3e-07 @225 | 3.7e-07 / 3.1e-07 / 2.5e-08 / 1.3e-07 @225 | 1.5e-07 / 1.2e-07 / 7.2e-09 / 6.6e-08 @225 | 1.9e-11 | 2.52e-06 |
| 5.048e+06 | 1.2e-05 / 9.1e-06 / 3.5e-08 / 5.4e-06 @225 | 4e-07 / 3.4e-07 / 3.1e-08 / 2.3e-07 @225 | 1.5e-07 / 1.2e-07 / 9.9e-09 / 9.6e-08 @225 | 1.9e-11 | 2.52e-06 |
| 5.943e+06 | 1.3e-05 / 1.1e-05 / 5e-08 / 8.4e-06 @225 | 3.8e-07 / 3.3e-07 / 2.4e-08 / 2.8e-07 @225 | 1.5e-07 / 1.3e-07 / 4.7e-09 / 1.1e-07 @225 | 1.9e-11 | 2.52e-06 |
| 6.998e+06 | 1.2e-05 / 1.1e-05 / 2.7e-07 / 1.1e-05 @225 | 3.8e-07 / 3.6e-07 / 3e-08 / 3.6e-07 @215 | 1.4e-07 / 1.3e-07 / 8.1e-09 / 1.3e-07 @215 | 2.1e-11 | 2.52e-06 |
| 8.241e+06 | 1.2e-05 / 1.1e-05 / 2e-07 / 1.2e-05 @231 | 3.9e-07 / 3.9e-07 / 2.6e-08 / 3.9e-07 @231 | 1.5e-07 / 1.4e-07 / 1.1e-08 / 1.5e-07 @231 | 2.4e-11 | 2.52e-06 |
| 9.703e+06 | 1.4e-05 / 1.3e-05 / 2e-07 / 1.4e-05 @231 | 5.2e-07 / 5e-07 / 1.8e-07 / 5e-07 @216 | 1.5e-07 / 1.4e-07 / 4.8e-09 / 1.5e-07 @231 | 2.7e-11 | 2.52e-06 |
| 1.143e+07 | 1.5e-05 / 1.4e-05 / 1.2e-06 / 1.5e-05 @231 | 1.6e-06 / 1.6e-06 / 1e-06 / 1.1e-06 @226 | 1.5e-07 / 1.4e-07 / 3e-09 / 1.5e-07 @231 | 2.8e-11 | 2.52e-06 |
| 1.345e+07 | 1.2e-05 / 9.4e-06 / 2.1e-07 / 7.7e-06 @219 | 4.9e-07 / 4.6e-07 / 5.5e-08 / 2.6e-07 @219 | 1.4e-07 / 1.4e-07 / 7.6e-09 / 5.4e-08 @219 | 2.9e-11 | 2.52e-06 |
| 1.584e+07 | 1.2e-05 / 1e-05 / 3e-07 / 5.4e-06 @219 | 4.2e-07 / 4.1e-07 / 3.1e-08 / 1.3e-07 @209 | 1.4e-07 / 1.3e-07 / 5.3e-09 / 4.7e-08 @209 | 2.7e-11 | 2.52e-06 |
| 1.865e+07 | 1.2e-05 / 1.2e-05 / 2.2e-07 / 9.7e-07 @209 | 4.4e-07 / 4.2e-07 / 6.1e-08 / 5e-08 @209 | 1.4e-07 / 1.3e-07 / 3.1e-09 / 1.2e-09 @209 | 2.8e-11 | 2.52e-06 |
| 2.197e+07 | 1.1e-05 / 1e-05 / 5.1e-08 / 6.5e-07 @225 | 4.2e-07 / 4e-07 / 3e-08 / 8.1e-08 @225 | 1.5e-07 / 1.3e-07 / 4.6e-09 / 3.9e-08 @225 | 2.9e-11 | 2.52e-06 |
| 2.586e+07 | 1.1e-05 / 9.5e-06 / 4.1e-08 / 4.4e-07 @225 | 4.5e-07 / 3.8e-07 / 3.2e-08 / 1.8e-07 @225 | 1.5e-07 / 1.2e-07 / 9.1e-09 / 8.3e-08 @225 | 3.1e-11 | 2.52e-06 |
| 3.045e+07 | 1.4e-05 / 1e-05 / 2.3e-07 / 6.6e-06 @225 | 6.6e-07 / 5.8e-07 / 1.8e-07 / 3e-07 @225 | 1.6e-07 / 1.3e-07 / 4.4e-09 / 1e-07 @225 | 3.1e-11 | 2.52e-06 |
| 3.586e+07 | 1.3e-05 / 1.1e-05 / 5.2e-08 / 8.8e-06 @225 | 3.8e-07 / 3.7e-07 / 4.6e-08 / 3.4e-07 @225 | 1.5e-07 / 1.5e-07 / 8.8e-09 / 1.3e-07 @215 | 3.1e-11 | 2.52e-06 |
| 4.223e+07 | 1.1e-05 / 1.1e-05 / 4.6e-08 / 1.1e-05 @225 | 4.2e-07 / 4e-07 / 3.9e-08 / 4e-07 @215 | 1.5e-07 / 1.5e-07 / 9.4e-09 / 1.5e-07 @215 | 3.2e-11 | 2.52e-06 |
| 4.972e+07 | 1.2e-05 / 1.1e-05 / 6.5e-08 / 1.2e-05 @231 | 4.4e-07 / 4.4e-07 / 3.9e-08 / 4.4e-07 @231 | 1.6e-07 / 1.5e-07 / 4.6e-09 / 1.6e-07 @231 | 3.4e-11 | 2.52e-06 |
| 5.855e+07 | 1.3e-05 / 1.2e-05 / 5.4e-08 / 1.3e-05 @231 | 4.5e-07 / 4.1e-07 / 4.4e-08 / 4.5e-07 @231 | 1.7e-07 / 1.5e-07 / 3.7e-09 / 1.7e-07 @231 | 3.4e-11 | 2.52e-06 |
| 6.894e+07 | 1.4e-05 / 1.1e-05 / 2.8e-07 / 1.4e-05 @231 | 6.9e-07 / 6.8e-07 / 2.3e-07 / 6.9e-07 @231 | 1.6e-07 / 1.5e-07 / 6.7e-09 / 1.6e-07 @231 | 3.3e-11 | 2.52e-06 |
| 8.118e+07 | 1.2e-05 / 1e-05 / 6.2e-08 / 6.4e-06 @219 | 4.3e-07 / 4.2e-07 / 3.8e-08 / 1.8e-07 @219 | 1.6e-07 / 1.6e-07 / 3.6e-09 / 5e-08 @219 | 3.1e-11 | 2.52e-06 |
| 9.558e+07 | 1.1e-05 / 1.1e-05 / 3.6e-08 / 2.9e-06 @219 | 4.4e-07 / 4.2e-07 / 5e-08 / 8.1e-08 @209 | 1.6e-07 / 1.5e-07 / 3.8e-09 / 2.3e-08 @209 | 3.1e-11 | 2.52e-06 |
| 1.126e+08 | 1.1e-05 / 1.1e-05 / 4.7e-08 / 6.9e-07 @209 | 4.4e-07 / 4.2e-07 / 5.1e-08 / 2.9e-09 @209 | 1.7e-07 / 1.7e-07 / 1.2e-08 / 1.1e-08 @209 | 3.5e-11 | 2.52e-06 |
| 1.325e+08 | 1.1e-05 / 1e-05 / 6.6e-08 / 5.2e-07 @225 | 0.00024 / 0.00015 / 2e-05 / 4.1e-05 @10.7 | 1.8e-07 / 1.5e-07 / 4.2e-09 / 5.4e-08 @225 | 3.7e-11 | 2.52e-06 |
| 1.561e+08 | 1.2e-05 / 9.7e-06 / 5.9e-08 / 2.9e-06 @225 | 4.4e-07 / 3.7e-07 / 4.5e-08 / 1.8e-07 @225 | 1.9e-07 / 1.5e-07 / 4.2e-09 / 9.4e-08 @225 | 3.7e-11 | 2.52e-06 |
| 1.838e+08 | 1.3e-05 / 9.2e-06 / 5.6e-08 / 6.9e-06 @225 | 0.00023 / 0.00016 / 1.9e-05 / 2.3e-05 @10.8 | 1.9e-07 / 1.6e-07 / 8.5e-09 / 1.4e-07 @225 | 3.6e-11 | 2.52e-06 |
| 2.164e+08 | 1.2e-05 / 1e-05 / 4.9e-08 / 9.1e-06 @225 | 4.4e-07 / 4.1e-07 / 3.7e-08 / 3.4e-07 @225 | 1.9e-07 / 1.8e-07 / 7e-09 / 1.5e-07 @225 | 3.6e-11 | 2.52e-06 |
| 2.548e+08 | 1.1e-05 / 1.1e-05 / 1.9e-07 / 1.1e-05 @225 | 4.3e-07 / 4.1e-07 / 3.9e-08 / 4e-07 @215 | 2e-07 / 2e-07 / 7.7e-09 / 2e-07 @215 | 3.5e-11 | 2.52e-06 |
| 3e+08 | 1.2e-05 / 1.2e-05 / 3.9e-08 / 1.2e-05 @231 | 0.00025 / 0.00018 / 1.8e-05 / 8.9e-06 @10.8 | 2.2e-07 / 2.1e-07 / 8.9e-09 / 2.2e-07 @231 | 4.2e-11 | 2.52e-06 |

Initial-condition floor (production $T=1,T'=0$ against the series $1-x^2/10$, both at reference tolerance): min 2.52e-06, median 2.52e-06, max 2.52e-06 of the envelope, at $x_i$ = 0.00389--0.00389.

The oracle column: the same runs scored against the exact $T$ rather than against the reference,
at every eighth wavenumber. `reference vs exact T` is the initial-condition floor and nothing else;
`reference, exact IC, vs exact T` is the reference's own accuracy.

| k [1/Mpc] | reference vs exact T | reference, exact IC, vs exact T | IC floor (series measure) | atol 1e-13 vs exact T | atol 1e-13 vs reference |
|---|---|---|---|---|---|
| 1e+05 | 2.52e-06 | 2.3e-11 | 2.52e-06 | 2.52e-06 | 1.69e-07 |
| 2.665e+05 | 2.52e-06 | 2.26e-11 | 2.52e-06 | 2.53e-06 | 2.1e-07 |
| 7.105e+05 | 2.52e-06 | 1.74e-11 | 2.52e-06 | 2.52e-06 | 3.03e-07 |
| 1.894e+06 | 2.52e-06 | 2.35e-11 | 2.52e-06 | 2.54e-06 | 3.36e-07 |
| 5.048e+06 | 2.52e-06 | 2.42e-11 | 2.52e-06 | 2.58e-06 | 3.97e-07 |
| 1.345e+07 | 2.52e-06 | 3.49e-11 | 2.52e-06 | 2.62e-06 | 4.92e-07 |
| 3.586e+07 | 2.52e-06 | 3.81e-11 | 2.52e-06 | 2.61e-06 | 3.81e-07 |
| 9.558e+07 | 2.52e-06 | 4.05e-11 | 2.52e-06 | 2.54e-06 | 4.39e-07 |
| 2.548e+08 | 2.52e-06 | 4.28e-11 | 2.52e-06 | 2.66e-06 | 4.28e-07 |

Three wavenumbers exceed target at `1e-13` — $1.33\times10^8$, $1.84\times10^8$ and
$3\times10^8$ — all at $x\approx10.7$–$10.8$, all *levels*. `1e-16` clears all three and
`1e-10` exceeds at every $k$, uniformly, at $1.1$–$1.6\times10^{-5}$: that is the level prompt 12
removed. Note that the excursions here are *not* $k$-accidents (§7 (b)), which is a property of
the control and not of the production backgrounds: on $H=H_0(1+z)^2$ the problem in $x$ is
$k$-independent, so the only thing that breaks the scaling between one $k$ and another is `atol`
itself acting on $T'$, whose size falls like $1/k$. That is exactly why prompt 12's control made
the excursion look like a function of $k$, and why it is not one on a real background.

### 5.2 LambdaCDMModel

| atol | worst max dT/env | at k [1/Mpc] | at x | median over the 50 k of the per-k max | k with max > 3e-6 | total RHS evals over the grid |
|---|---|---|---|---|---|---|
| 1e-10 | 0.028 | 1.865e+07 | 40 | 1.19e-05 | 50 | 328897 |
| 1e-13 | 0.000864 | 8.366e+05 | 17.38 | 4.52e-07 | 13 | 429178 |
| 1e-16 | 0.000478 | 2.23e+06 | 13.85 | 4.2e-07 | 10 | 431260 |
Shape of the offending runs — *level* when the median is itself above $3\times10^{-6}$, *spike* when only the maximum is:

| atol | k above 3e-6 | of those, level | of those, spike |
|---|---|---|---|
| 1e-10 | 50 | 1 | 49 |
| 1e-13 | 13 | 13 | 0 |
| 1e-16 | 10 | 10 | 0 |
Every wavenumber of the production grid, as **max / 2nd-largest / median / last sample** of $\delta T/\mathrm{env}$. `x@max` is where the maximum fell; the grid runs from $x_i\approx0.0039$ to $x\approx275$.

| k [1/Mpc] | atol 1e-10 (x@max) | atol 1e-13 (x@max) | atol 1e-16 (x@max) | ref drift | IC floor |
|---|---|---|---|---|---|
| 1e+05 | 1.3e-05 / 1.1e-05 / 5.4e-08 / 8.1e-06 @225 | 4.6e-07 / 4.1e-07 / 4e-08 / 3.2e-07 @225 | 4.2e-07 / 3.8e-07 / 3.8e-08 / 2.8e-07 @225 | 3.8e-11 | 2.52e-06 |
| 1.178e+05 | 1.3e-05 / 1.1e-05 / 5e-08 / 1e-05 @225 | 4.1e-07 / 4e-07 / 4e-08 / 3.7e-07 @215 | 4e-07 / 3.8e-07 / 4.5e-08 / 3.7e-07 @215 | 3.7e-11 | 2.52e-06 |
| 1.387e+05 | 1.2e-05 / 1.1e-05 / 2e-07 / 1.2e-05 @231 | 5.9e-07 / 5.6e-07 / 1.6e-07 / 5.4e-07 @215 | 5.4e-07 / 5.1e-07 / 1.5e-07 / 5e-07 @215 | 3.8e-11 | 2.52e-06 |
| 1.633e+05 | 1.2e-05 / 9.6e-06 / 6.5e-08 / 7.9e-06 @219 | 4.6e-07 / 4.3e-07 / 4e-08 / 2.4e-07 @219 | 3.8e-07 / 3.7e-07 / 3.8e-08 / 1.7e-07 @219 | 4.4e-11 | 2.52e-06 |
| 1.922e+05 | 1.2e-05 / 9.6e-06 / 4.8e-08 / 6.1e-06 @219 | 4.1e-07 / 4.1e-07 / 3e-08 / 1.4e-07 @209 | 4.1e-07 / 4.1e-07 / 3.3e-08 / 1.5e-07 @209 | 3.5e-11 | 2.52e-06 |
| 2.264e+05 | 1.1e-05 / 1.1e-05 / 4.3e-08 / 3.3e-06 @209 | 4.2e-07 / 4e-07 / 3.9e-08 / 7.9e-08 @209 | 3.9e-07 / 3.8e-07 / 3.9e-08 / 5.6e-08 @209 | 4.5e-11 | 2.52e-06 |
| 2.665e+05 | 1.1e-05 / 1.1e-05 / 2e-07 / 1.1e-07 @209 | 0.00027 / 0.00014 / 2.2e-05 / 5.5e-05 @10.7 | 6.3e-07 / 6.2e-07 / 1.6e-07 / 1e-07 @209 | 3.8e-11 | 2.52e-06 |
| 3.139e+05 | 1.2e-05 / 1.1e-05 / 4.6e-08 / 7.7e-07 @225 | 4.3e-07 / 4.1e-07 / 4.1e-08 / 7.5e-08 @225 | 4.2e-07 / 4e-07 / 4.1e-08 / 5.4e-08 @225 | 3.8e-11 | 2.52e-06 |
| 3.696e+05 | 1.1e-05 / 1e-05 / 4.1e-08 / 4.4e-07 @225 | 4.3e-07 / 3.6e-07 / 4.4e-08 / 1.8e-07 @225 | 4.2e-07 / 3.5e-07 / 3.4e-08 / 1.2e-07 @225 | 4e-11 | 2.52e-06 |
| 4.352e+05 | 1.3e-05 / 9.8e-06 / 7.5e-08 / 3.7e-06 @225 | 0.00032 / 0.00019 / 2.5e-05 / 5e-05 @10.8 | 4.1e-07 / 3.5e-07 / 4e-08 / 2.1e-07 @225 | 4e-11 | 2.52e-06 |
| 5.124e+05 | 1.3e-05 / 9.5e-06 / 2.3e-07 / 6.6e-06 @225 | 0.0005 / 0.00049 / 5.8e-05 / 0.00012 @14.2 | 0.00025 / 0.00016 / 2.1e-05 / 2.6e-05 @10.8 | 3.6e-11 | 2.52e-06 |
| 6.034e+05 | 1.3e-05 / 1.1e-05 / 5.3e-08 / 8.4e-06 @225 | 4.4e-07 / 3.9e-07 / 4.8e-08 / 3.1e-07 @225 | 4.1e-07 / 3.7e-07 / 4.8e-08 / 3e-07 @225 | 3.8e-11 | 2.52e-06 |
| 7.105e+05 | 1.2e-05 / 1e-05 / 4.8e-08 / 1e-05 @225 | 0.00026 / 0.00017 / 2.2e-05 / 9.9e-06 @10.8 | 4e-07 / 4e-07 / 3.9e-08 / 3.6e-07 @225 | 3.6e-11 | 2.52e-06 |
| 8.366e+05 | 1.2e-05 / 9.8e-06 / 6.5e-08 / 8.7e-06 @219 | 0.00086 / 0.00076 / 1.7e-05 / 0.00022 @17.4 | 3.9e-07 / 3.7e-07 / 3.8e-08 / 1.9e-07 @219 | 3.8e-11 | 2.52e-06 |
| 9.851e+05 | 1.2e-05 / 1e-05 / 5.2e-08 / 6.9e-06 @219 | 4.2e-07 / 4.1e-07 / 3.8e-08 / 1.8e-07 @219 | 4e-07 / 3.9e-07 / 3.8e-08 / 1.7e-07 @219 | 3.6e-11 | 2.52e-06 |
| 1.16e+06 | 1.2e-05 / 1.1e-05 / 6.8e-08 / 4.8e-06 @219 | 4.3e-07 / 4.1e-07 / 5.8e-08 / 1.3e-07 @209 | 4.2e-07 / 4.1e-07 / 5.8e-08 / 1.3e-07 @209 | 4.9e-11 | 2.52e-06 |
| 1.366e+06 | 1.1e-05 / 1.1e-05 / 5.5e-08 / 1.6e-06 @209 | 4.1e-07 / 3.9e-07 / 3e-08 / 6.6e-08 @209 | 0.00024 / 0.00014 / 2e-05 / 5e-05 @10.7 | 3.6e-11 | 2.52e-06 |
| 1.608e+06 | 1.1e-05 / 1.1e-05 / 5.1e-08 / 7.5e-07 @209 | 4.3e-07 / 4.2e-07 / 3.4e-08 / 1.7e-08 @209 | 4.2e-07 / 4e-07 / 3.4e-08 / 2.8e-08 @209 | 3.6e-11 | 2.52e-06 |
| 1.894e+06 | 1.2e-05 / 1.1e-05 / 4.2e-08 / 1.9e-07 @225 | 4.2e-07 / 3.7e-07 / 3.5e-08 / 1.2e-07 @225 | 3.7e-07 / 3.4e-07 / 3.6e-08 / 9.6e-08 @225 | 3.7e-11 | 2.52e-06 |
| 2.23e+06 | 1.2e-05 / 1e-05 / 1.9e-07 / 8.4e-09 @225 | 5.8e-07 / 5.2e-07 / 1.4e-07 / 1.5e-07 @225 | 0.00048 / 0.00034 / 2.7e-05 / 9.2e-05 @13.8 | 3.8e-11 | 2.52e-06 |
| 2.626e+06 | 1.3e-05 / 9.8e-06 / 5.2e-08 / 5e-06 @225 | 4.5e-07 / 3.8e-07 / 3.1e-08 / 2.1e-07 @225 | 4.4e-07 / 3.8e-07 / 3.4e-08 / 2e-07 @225 | 3.6e-11 | 2.52e-06 |
| 3.092e+06 | 1.3e-05 / 1e-05 / 6.1e-08 / 7.3e-06 @225 | 4.5e-07 / 3.9e-07 / 3.5e-08 / 2.9e-07 @225 | 4.2e-07 / 3.7e-07 / 3.6e-08 / 2.6e-07 @225 | 4e-11 | 2.52e-06 |
| 3.64e+06 | 1.2e-05 / 1e-05 / 4e-08 / 8.6e-06 @225 | 4.5e-07 / 4.2e-07 / 4.1e-08 / 3.7e-07 @225 | 4.2e-07 / 3.9e-07 / 4.1e-08 / 3.3e-07 @225 | 3.7e-11 | 2.52e-06 |
| 4.287e+06 | 1.2e-05 / 1.1e-05 / 2.1e-07 / 1.1e-05 @225 | 0.00026 / 0.00017 / 2.1e-05 / 8.6e-06 @10.8 | 0.00025 / 0.00017 / 2e-05 / 8.3e-06 @10.8 | 3.7e-11 | 2.52e-06 |
| 5.048e+06 | 1.2e-05 / 9.4e-06 / 4.8e-08 / 7.6e-06 @219 | 5.1e-07 / 4.8e-07 / 3.9e-08 / 2.2e-07 @43.7 | 4.4e-07 / 4.3e-07 / 4.6e-08 / 2.2e-07 @219 | 3.4e-11 | 2.52e-06 |
| 5.943e+06 | 1.2e-05 / 1e-05 / 5.6e-08 / 5.9e-06 @219 | 4.3e-07 / 4.2e-07 / 3.2e-08 / 1.4e-07 @209 | 3.8e-07 / 3.8e-07 / 3e-08 / 1.3e-07 @209 | 3.8e-11 | 2.52e-06 |
| 6.998e+06 | 1.1e-05 / 1.1e-05 / 4.3e-08 / 3.6e-06 @219 | 0.00034 / 0.0003 / 3e-05 / 8.8e-05 @10.7 | 0.00034 / 0.0003 / 3e-05 / 8.8e-05 @10.7 | 3.7e-11 | 2.52e-06 |
| 8.241e+06 | 1.1e-05 / 1.1e-05 / 2.2e-07 / 2.5e-07 @209 | 0.00027 / 0.00014 / 2.3e-05 / 5.5e-05 @10.7 | 0.00026 / 0.00014 / 2.2e-05 / 5.4e-05 @10.7 | 3.4e-11 | 2.52e-06 |
| 9.703e+06 | 1.2e-05 / 1.1e-05 / 6.3e-08 / 9.7e-07 @225 | 4.4e-07 / 4.3e-07 / 4.7e-08 / 3.9e-08 @210 | 3.9e-07 / 3.9e-07 / 4.4e-08 / 2.9e-08 @210 | 3.6e-11 | 2.52e-06 |
| 1.143e+07 | 1.1e-05 / 1e-05 / 4.3e-08 / 5.9e-07 @225 | 4.5e-07 / 4e-07 / 3.4e-08 / 1.3e-07 @225 | 4.3e-07 / 3.7e-07 / 3.3e-08 / 1.2e-07 @225 | 3.9e-11 | 2.52e-06 |
| 1.345e+07 | 1.2e-05 / 9.7e-06 / 4.2e-08 / 1.1e-06 @225 | 0.0005 / 0.0005 / 3.5e-05 / 0.00011 @14.2 | 3.9e-07 / 3.4e-07 / 4.1e-08 / 1.7e-07 @225 | 3.9e-11 | 2.52e-06 |
| 1.584e+07 | 1.2e-05 / 9.3e-06 / 5.2e-08 / 5.6e-06 @225 | 4.8e-07 / 4.1e-07 / 4.6e-08 / 2.5e-07 @225 | 4.2e-07 / 3.6e-07 / 4.3e-08 / 2.2e-07 @225 | 4.1e-11 | 2.52e-06 |
| 1.865e+07 | 0.028 / 0.012 / 7.3e-08 / 0.0062 @40 | 4.5e-07 / 3.9e-07 / 4.8e-08 / 2.9e-07 @225 | 4.4e-07 / 3.9e-07 / 5.3e-08 / 2.7e-07 @225 | 4e-11 | 2.52e-06 |
| 2.197e+07 | 0.00024 / 0.00016 / 2e-05 / 5.8e-06 @10.8 | 4.5e-07 / 4.3e-07 / 4e-08 / 3.9e-07 @225 | 4e-07 / 3.9e-07 / 3.7e-08 / 3.4e-07 @225 | 3.6e-11 | 2.52e-06 |
| 2.586e+07 | 1.4e-05 / 1.1e-05 / 5.4e-08 / 9.6e-06 @219 | 4.5e-07 / 4.1e-07 / 3.6e-08 / 2.6e-07 @219 | 4.3e-07 / 3.9e-07 / 3.9e-08 / 2.3e-07 @219 | 3.8e-11 | 2.52e-06 |
| 3.045e+07 | 1.2e-05 / 1e-05 / 4.9e-08 / 6.7e-06 @219 | 4.1e-07 / 4.1e-07 / 3.5e-08 / 2e-07 @219 | 3.7e-07 / 3.7e-07 / 3.4e-08 / 1.8e-07 @219 | 3.4e-11 | 2.52e-06 |
| 3.586e+07 | 1.2e-05 / 1.1e-05 / 1.8e-07 / 4.9e-06 @219 | 6e-07 / 5.9e-07 / 1.5e-07 / 2.4e-07 @209 | 5.8e-07 / 5.7e-07 / 1.5e-07 / 1.9e-07 @209 | 3.6e-11 | 2.52e-06 |
| 4.223e+07 | 1.1e-05 / 1.1e-05 / 5.4e-08 / 1.7e-06 @209 | 3.7e-07 / 3.5e-07 / 2.7e-08 / 6.4e-08 @209 | 3.9e-07 / 3.8e-07 / 2.6e-08 / 8.4e-08 @209 | 4e-11 | 2.52e-06 |
| 4.972e+07 | 1.1e-05 / 1.1e-05 / 5.2e-08 / 6.5e-07 @209 | 4.1e-07 / 4e-07 / 2.9e-08 / 9e-09 @209 | 4e-07 / 3.9e-07 / 3.4e-08 / 2.2e-09 @209 | 3.7e-11 | 2.52e-06 |
| 5.855e+07 | 1.2e-05 / 1.1e-05 / 4.6e-08 / 6.3e-07 @225 | 4.6e-07 / 4.3e-07 / 3.8e-08 / 6e-08 @225 | 4.2e-07 / 4e-07 / 4e-08 / 8.7e-08 @225 | 3.8e-11 | 2.52e-06 |
| 6.894e+07 | 1.1e-05 / 1e-05 / 2e-07 / 6.5e-07 @225 | 0.00026 / 0.00015 / 2.2e-05 / 4.3e-05 @10.7 | 0.00026 / 0.00015 / 2.2e-05 / 4.2e-05 @10.7 | 3.9e-11 | 2.52e-06 |
| 8.118e+07 | 1.1e-05 / 9.4e-06 / 4.3e-08 / 2.1e-06 @225 | 4.7e-07 / 4e-07 / 4.3e-08 / 2e-07 @225 | 4.4e-07 / 3.8e-07 / 4.2e-08 / 2e-07 @225 | 3.6e-11 | 2.52e-06 |
| 9.558e+07 | 1.3e-05 / 9.5e-06 / 4.5e-08 / 6e-06 @225 | 0.00025 / 0.00016 / 2.1e-05 / 2.9e-05 @10.8 | 0.00026 / 0.00016 / 2.2e-05 / 3e-05 @10.8 | 4.2e-11 | 2.52e-06 |
| 1.126e+08 | 1.3e-05 / 9.8e-06 / 7.6e-08 / 7.7e-06 @225 | 4.2e-07 / 3.7e-07 / 4.1e-08 / 3.1e-07 @225 | 4.3e-07 / 3.9e-07 / 4.6e-08 / 3.1e-07 @225 | 3.9e-11 | 2.52e-06 |
| 1.325e+08 | 1.3e-05 / 1e-05 / 2.4e-07 / 9.2e-06 @225 | 0.00026 / 0.00017 / 2.1e-05 / 1.5e-05 @10.8 | 0.00025 / 0.00017 / 2.1e-05 / 1.5e-05 @10.8 | 3.6e-11 | 2.52e-06 |
| 1.561e+08 | 1.2e-05 / 9.8e-06 / 6.9e-08 / 7.4e-06 @219 | 4.5e-07 / 4.2e-07 / 4.7e-08 / 2.3e-07 @219 | 4.3e-07 / 4.1e-07 / 4.7e-08 / 2.2e-07 @219 | 5.7e-11 | 2.52e-06 |
| 1.838e+08 | 1.2e-05 / 9.6e-06 / 5.4e-08 / 5.9e-06 @219 | 4.1e-07 / 4e-07 / 3.1e-08 / 1.3e-07 @209 | 4.4e-07 / 4.3e-07 / 3.2e-08 / 1.5e-07 @209 | 3.5e-11 | 2.52e-06 |
| 2.164e+08 | 1.1e-05 / 1.1e-05 / 4.4e-08 / 3.4e-06 @219 | 3.9e-07 / 3.7e-07 / 4e-08 / 7.7e-08 @209 | 4e-07 / 3.9e-07 / 4e-08 / 5.7e-08 @209 | 3.4e-11 | 2.52e-06 |
| 2.548e+08 | 1.1e-05 / 1.1e-05 / 4.9e-08 / 5.1e-07 @209 | 0.00027 / 0.00014 / 2.3e-05 / 5.7e-05 @10.7 | 0.00027 / 0.00014 / 2.2e-05 / 5.7e-05 @10.7 | 5.3e-11 | 2.52e-06 |
| 3e+08 | 1.1e-05 / 1.1e-05 / 5.5e-08 / 7.9e-07 @225 | 4.5e-07 / 4.3e-07 / 5.8e-08 / 6.2e-08 @225 | 4e-07 / 3.9e-07 / 5.7e-08 / 6e-08 @225 | 3.7e-11 | 2.52e-06 |

Initial-condition floor (production $T=1,T'=0$ against the series $1-x^2/10$, both at reference tolerance): min 2.52e-06, median 2.52e-06, max 2.52e-06 of the envelope, at $x_i$ = 0.00389--0.00389.

Thirteen wavenumbers exceed target at `1e-13`, ten at `1e-16`, and the two sets are not nested:
five fail only at `1e-13` ($2.67\times10^5$, $4.35\times10^5$, $7.11\times10^5$,
$8.37\times10^5$, $1.35\times10^7$), two only at `1e-16` ($1.37\times10^6$, $2.23\times10^6$),
and eight at both. Every offender is a
*level*, with medians of $1.7$–$5.8\times10^{-5}$ and last samples of $8.6\times10^{-6}$ to
$2.2\times10^{-4}$. The worst single figure in this document is here: $8.6\times10^{-4}$ at
$k=8.37\times10^5$, $x=17.4$, 290× README §6's row.

At `atol = 1e-10` — the pre-prompt-12 configuration — one wavenumber, $k=1.87\times10^7$, reaches
**$2.8\times10^{-2}$** of the envelope at $x=40$, a 2.8 % error in the transfer function. That is
the largest excursion anywhere in this sweep and it is on the setting prompt 12 replaced.

### 5.3 QCDModel

| atol | worst max dT/env | at k [1/Mpc] | at x | median over the 50 k of the per-k max | k with max > 3e-6 | total RHS evals over the grid |
|---|---|---|---|---|---|---|
| 1e-10 | 0.000372 | 2.548e+08 | 10.52 | 1.38e-05 | 50 | 388357 |
| 1e-13 | 0.000283 | 1.584e+07 | 10.33 | 9.99e-07 | 8 | 485467 |
| 1e-16 | 2.49e-05 | 3.045e+07 | 6.757 | 9.49e-07 | 4 | 490729 |
Shape of the offending runs — *level* when the median is itself above $3\times10^{-6}$, *spike* when only the maximum is:

| atol | k above 3e-6 | of those, level | of those, spike |
|---|---|---|---|
| 1e-10 | 50 | 7 | 43 |
| 1e-13 | 8 | 4 | 4 |
| 1e-16 | 4 | 0 | 4 |
| k [1/Mpc] | atol 1e-10 (x@max) | atol 1e-13 (x@max) | atol 1e-16 (x@max) | ref drift | IC floor |
|---|---|---|---|---|---|
| 1e+05 | 1.4e-05 / 1.1e-05 / 4.7e-07 / 5.4e-06 @225 | 8e-07 / 7.8e-07 / 3.8e-07 / 7.8e-07 @215 | 2.3e-06 / 2.1e-06 / 1.4e-06 / 1.8e-06 @15.2 | 5.9e-09 | 2.39e-06 |
| 1.178e+05 | 1.4e-05 / 1.1e-05 / 1.2e-06 / 1.4e-05 @228 | 7.3e-07 / 6.7e-07 / 2.9e-07 / 5.4e-07 @218 | 1.1e-06 / 1e-06 / 5.4e-07 / 6.4e-07 @218 | 1.4e-07 | 2.54e-06 |
| 1.387e+05 | 1.3e-05 / 1.2e-05 / 1.6e-06 / 1.2e-05 @227 | 1.1e-06 / 1.1e-06 / 5.9e-07 / 1.1e-06 @233 | 1.3e-06 / 1.3e-06 / 8.4e-07 / 5.8e-07 @2.32 | 6.6e-09 | 2.41e-06 |
| 1.633e+05 | 1.3e-05 / 1.2e-05 / 1e-07 / 1.3e-05 @232 | 4.9e-07 / 4.5e-07 / 1.8e-07 / 4.9e-07 @232 | 1.5e-06 / 1.5e-06 / 9.4e-07 / 1.4e-07 @15.5 | 1.4e-07 | 2.56e-06 |
| 1.922e+05 | 1.7e-05 / 1.5e-05 / 3.8e-07 / 1.7e-05 @232 | 4.6e-07 / 4.6e-07 / 1.3e-07 / 4.6e-07 @216 | 7.7e-07 / 7.6e-07 / 4.8e-07 / 3.4e-07 @221 | 4.3e-08 | 2.49e-06 |
| 2.264e+05 | 1.4e-05 / 1.3e-05 / 4.3e-06 / 9.3e-06 @225 | 3e-06 / 3e-06 / 2.1e-06 / 2.4e-06 @21.7 | 5.2e-07 / 5.2e-07 / 2.4e-07 / 5.1e-07 @215 | 1.9e-08 | 2.45e-06 |
| 2.665e+05 | 1.5e-05 / 1.3e-05 / 2.3e-06 / 1.5e-05 @230 | 5.6e-07 / 5.2e-07 / 2.1e-07 / 8.8e-08 @15.4 | 8.7e-07 / 8.2e-07 / 5.7e-07 / 3.3e-07 @9.33 | 3.4e-08 | 2.42e-06 |
| 3.139e+05 | 1.4e-05 / 1.2e-05 / 4.1e-07 / 1.2e-05 @220 | 7.4e-07 / 7.2e-07 / 1.9e-07 / 3e-08 @1.48 | 9.7e-07 / 9.6e-07 / 5e-07 / 3.4e-09 @210 | 2.4e-08 | 2.41e-06 |
| 3.696e+05 | 1.4e-05 / 1.1e-05 / 3.5e-07 / 4.6e-06 @226 | 3.4e-07 / 3e-07 / 6.8e-08 / 2.2e-07 @226 | 6.6e-07 / 6.2e-07 / 3.1e-07 / 1.9e-07 @15.8 | 3.1e-09 | 2.45e-06 |
| 4.352e+05 | 1.4e-05 / 1.2e-05 / 1.5e-06 / 1.4e-05 @232 | 5.6e-07 / 5e-07 / 1.7e-07 / 5e-07 @15.8 | 4.2e-07 / 4e-07 / 4.8e-08 / 4.2e-07 @232 | 4.8e-08 | 2.54e-06 |
| 5.124e+05 | 1.4e-05 / 1.3e-05 / 5.8e-07 / 1.4e-05 @232 | 1.1e-06 / 1.1e-06 / 6.7e-07 / 9.9e-07 @175 | 9.5e-07 / 9.4e-07 / 3.7e-07 / 9.5e-07 @232 | 1.3e-07 | 2.62e-06 |
| 6.034e+05 | 1.4e-05 / 1.1e-05 / 6.9e-07 / 2.9e-06 @224 | 5.9e-07 / 5.4e-07 / 1.5e-07 / 3.4e-07 @224 | 2.5e-06 / 2.4e-06 / 1.4e-06 / 2.4e-06 @205 | 5.7e-09 | 2.56e-06 |
| 7.105e+05 | 1.3e-05 / 1.1e-05 / 1e-06 / 1.3e-05 @231 | 2.2e-06 / 2.2e-06 / 1.4e-06 / 2.5e-07 @18.9 | 6.9e-07 / 6e-07 / 3.5e-07 / 5.2e-08 @3.94 | 8.1e-09 | 2.51e-06 |
| 8.366e+05 | 2.3e-05 / 1.8e-05 / 4.2e-06 / 1.4e-06 @6.25 | 3.4e-06 / 3.4e-06 / 1.6e-06 / 3.1e-06 @5.57 | 2.2e-06 / 2.2e-06 / 9.8e-07 / 1.7e-06 @5.32 | 1.5e-08 | 2.5e-06 |
| 9.851e+05 | 1.4e-05 / 1.3e-05 / 5.3e-06 / 1.4e-05 @229 | 1.5e-06 / 1.4e-06 / 7.5e-07 / 1.2e-08 @12.5 | 4.8e-07 / 4.7e-07 / 2.4e-07 / 9e-08 @8.83 | 7.5e-09 | 2.49e-06 |
| 1.16e+06 | 1.2e-05 / 1.1e-05 / 1.3e-06 / 1.1e-05 @221 | 7.4e-06 / 7.4e-06 / 4.9e-06 / 6.9e-06 @177 | 9.6e-07 / 9.2e-07 / 5.2e-07 / 9.2e-07 @15.7 | 4.4e-08 | 2.52e-06 |
| 1.366e+06 | 1.1e-05 / 1.1e-05 / 1.1e-06 / 5.3e-07 @223 | 8.4e-07 / 8.3e-07 / 4.1e-07 / 8e-07 @218 | 1.2e-06 / 1e-06 / 4e-07 / 6.5e-07 @2.62 | 4.8e-08 | 2.54e-06 |
| 1.608e+06 | 1.3e-05 / 1.1e-05 / 1.8e-06 / 1.3e-05 @229 | 6.8e-07 / 4.8e-07 / 8.8e-08 / 4.2e-07 @3.47 | 1.1e-06 / 1.1e-06 / 5e-07 / 1.1e-06 @229 | 3.1e-08 | 2.3e-06 |
| 1.894e+06 | 1.2e-05 / 1.1e-05 / 4e-07 / 1.1e-05 @220 | 6.1e-07 / 5.9e-07 / 2.5e-07 / 1.5e-07 @210 | 7e-07 / 7e-07 / 2.6e-07 / 2.6e-07 @220 | 5.3e-08 | 2.57e-06 |
| 2.23e+06 | 1.3e-05 / 1.2e-05 / 1.2e-06 / 1.1e-05 @224 | 5.2e-07 / 5e-07 / 1.5e-07 / 4.3e-07 @214 | 5.4e-07 / 4.6e-07 / 2.6e-07 / 8e-08 @4.55 | 7.6e-09 | 2.49e-06 |
| 2.626e+06 | 1.2e-05 / 1.2e-05 / 2.8e-07 / 1.2e-05 @230 | 1.1e-06 / 1e-06 / 5.6e-07 / 3.3e-07 @225 | 5.9e-07 / 5.5e-07 / 2.1e-07 / 5.9e-07 @230 | 4.1e-09 | 2.42e-06 |
| 3.092e+06 | 1.3e-05 / 1.3e-05 / 1.3e-06 / 1.3e-05 @225 | 6.3e-07 / 5.6e-07 / 2.2e-07 / 1.7e-07 @6.42 | 8.8e-07 / 8.7e-07 / 5.2e-07 / 6.5e-08 @113 | 1.1e-07 | 2.56e-06 |
| 3.64e+06 | 1.2e-05 / 1.2e-05 / 8.3e-08 / 1.2e-05 @225 | 8.7e-07 / 8.6e-07 / 5.5e-07 / 1.1e-07 @210 | 4.5e-07 / 4.1e-07 / 1.3e-07 / 4.5e-07 @230 | 7.7e-09 | 2.35e-06 |
| 4.287e+06 | 1.5e-05 / 1.2e-05 / 9.1e-07 / 8.9e-06 @225 | 6.6e-07 / 6.6e-07 / 3.8e-07 / 3.9e-07 @4.53 | 8.6e-07 / 8.2e-07 / 4.9e-07 / 8.6e-07 @230 | 1e-08 | 2.43e-06 |
| 5.048e+06 | 1.5e-05 / 1.1e-05 / 1.8e-06 / 1.5e-05 @228 | 1.4e-06 / 1.4e-06 / 7.5e-07 / 6.1e-07 @199 | 4e-07 / 3.7e-07 / 2.2e-07 / 2.2e-07 @6.31 | 8.8e-07 | 2.36e-06 |
| 5.943e+06 | 1.2e-05 / 1.2e-05 / 6.8e-07 / 1e-07 @222 | 1.3e-06 / 1.3e-06 / 7.2e-07 / 1.1e-06 @227 | 9.5e-07 / 8.7e-07 / 4.1e-07 / 4.9e-07 @227 | 8.1e-07 | 2.44e-06 |
| 6.998e+06 | 1.3e-05 / 9.6e-06 / 3.8e-07 / 8.5e-06 @222 | 2.6e-06 / 2.5e-06 / 1.6e-06 / 1.6e-06 @101 | 4.9e-07 / 4.8e-07 / 2.2e-07 / 3.7e-07 @4.18 | 3.3e-07 | 2.45e-06 |
| 8.241e+06 | 1.3e-05 / 9.8e-06 / 6.6e-07 / 2.6e-06 @225 | 1e-06 / 9.8e-07 / 4.8e-07 / 8.6e-07 @215 | 7.8e-07 / 7.7e-07 / 5.1e-07 / 7.7e-07 @9.03 | 2.1e-07 | 2.45e-06 |
| 9.703e+06 | 1.3e-05 / 1.3e-05 / 3.6e-07 / 1.3e-06 @210 | 9.8e-07 / 8.9e-07 / 3.9e-07 / 3.7e-07 @225 | 1.5e-06 / 1.5e-06 / 5.8e-07 / 1.8e-07 @210 | 6e-09 | 2.45e-06 |
| 1.143e+07 | 1.3e-05 / 1.3e-05 / 4e-07 / 3.5e-06 @209 | 4.1e-07 / 4.1e-07 / 2.4e-07 / 3.8e-07 @166 | 7.8e-07 / 7.6e-07 / 3e-07 / 3.9e-07 @5.11 | 5.2e-07 | 2.46e-06 |
| 1.345e+07 | 1.4e-05 / 1.2e-05 / 2.3e-07 / 6.6e-06 @219 | 1.6e-06 / 1.4e-06 / 6.9e-07 / 1.2e-06 @14.9 | 5.4e-07 / 5.3e-07 / 1.5e-07 / 1.5e-08 @209 | 2.8e-08 | 2.44e-06 |
| 1.584e+07 | 1.2e-05 / 9.5e-06 / 1.2e-06 / 8.4e-06 @219 | 0.00028 / 0.00024 / 2.1e-05 / 7e-05 @10.3 | 2e-06 / 2e-06 / 8e-07 / 6.1e-07 @4.79 | 1.6e-06 | 3.41e-06 |
| 1.865e+07 | 1.5e-05 / 1.3e-05 / 2.4e-06 / 1.1e-05 @223 | 3.6e-05 / 3.5e-05 / 1.2e-05 / 2.4e-05 @8.04 | 1.3e-06 / 1.3e-06 / 4.4e-07 / 1.2e-06 @213 | 2.5e-08 | 2.5e-06 |
| 2.197e+07 | 1.4e-05 / 1.4e-05 / 4.3e-07 / 1.4e-05 @213 | 1.6e-06 / 9.1e-07 / 3.2e-07 / 8.8e-08 @9.44 | 2.2e-06 / 1.6e-06 / 7.6e-07 / 8e-07 @9.44 | 1.2e-06 | 2.51e-06 |
| 2.586e+07 | 1e-05 / 9.1e-06 / 1e-06 / 1e-05 @228 | 2.1e-06 / 1.2e-06 / 5.7e-07 / 7.7e-07 @6.43 | 8e-07 / 7.6e-07 / 2.6e-07 / 3.2e-07 @9.85 | 5.2e-09 | 2.5e-06 |
| 3.045e+07 | 1.7e-05 / 1.3e-05 / 3e-06 / 1.7e-05 @228 | 2.2e-06 / 1.1e-06 / 1.1e-07 / 4.1e-07 @6.61 | 2.5e-05 / 1.9e-05 / 1.5e-06 / 3.9e-06 @6.76 | 3.8e-09 | 2.52e-06 |
| 3.586e+07 | 1.3e-05 / 1.1e-05 / 6.9e-07 / 1.3e-05 @228 | 5.5e-07 / 5.4e-07 / 1.1e-07 / 1.9e-07 @4.84 | 2.6e-06 / 2.5e-06 / 9.8e-07 / 1.4e-06 @14.3 | 1.6e-06 | 2.79e-06 |
| 4.223e+07 | 1.4e-05 / 1.4e-05 / 1.3e-06 / 1.4e-05 @228 | 5.5e-07 / 5.4e-07 / 2.4e-07 / 1.6e-07 @10.2 | 8.8e-07 / 8.3e-07 / 3.6e-07 / 6.2e-07 @218 | 3e-08 | 2.69e-06 |
| 4.972e+07 | 1.4e-05 / 1.4e-05 / 4.6e-07 / 6.7e-06 @46.1 | 0.00016 / 8.9e-05 / 1.2e-05 / 2.2e-05 @9.24 | 1.5e-06 / 1.4e-06 / 4.6e-07 / 9.6e-07 @19.2 | 2.8e-06 | 3.42e-06 |
| 5.855e+07 | 0.00032 / 0.00029 / 1.7e-05 / 6.9e-05 @12.1 | 1.3e-06 / 1.3e-06 / 7.3e-07 / 1.3e-06 @228 | 1.8e-05 / 1.8e-05 / 9.2e-08 / 1.1e-05 @218 | 6.2e-06 | 2.52e-06 |
| 6.894e+07 | 1.4e-05 / 1.4e-05 / 4.7e-08 / 1.4e-05 @228 | 6.9e-06 / 6.9e-06 / 9.7e-07 / 5.5e-06 @217 | 4.2e-06 / 4.2e-06 / 2.6e-06 / 5.1e-07 @184 | 4.2e-09 | 2.52e-06 |
| 8.118e+07 | 1.4e-05 / 1.3e-05 / 8.2e-08 / 9.8e-07 @210 | 7.9e-07 / 7.7e-07 / 3.6e-07 / 5.2e-10 @210 | 6.8e-07 / 5.8e-07 / 3.4e-07 / 3.9e-07 @11.2 | 2e-09 | 2.51e-06 |
| 9.558e+07 | 1.7e-05 / 1.3e-05 / 2.6e-07 / 1.7e-05 @224 | 9.6e-07 / 8.8e-07 / 1.5e-07 / 9.6e-07 @224 | 7.3e-07 / 6.5e-07 / 2.8e-07 / 1.2e-07 @24.9 | 7.8e-09 | 2.52e-06 |
| 1.126e+08 | 1.3e-05 / 1.3e-05 / 1.8e-07 / 8.7e-06 @213 | 4.1e-07 / 3.9e-07 / 8.8e-08 / 2.4e-07 @53 | 7.5e-07 / 6.5e-07 / 3.2e-07 / 8.5e-09 @2.17 | 4.1e-09 | 2.52e-06 |
| 1.325e+08 | 1.8e-05 / 1.6e-05 / 3.3e-07 / 1.8e-05 @223 | 1.5e-06 / 1.5e-06 / 4.5e-07 / 1.4e-06 @213 | 1e-06 / 9.7e-07 / 1.5e-07 / 1e-06 @223 | 2.9e-09 | 2.51e-06 |
| 1.561e+08 | 1.6e-05 / 1.5e-05 / 1.8e-07 / 1.5e-05 @213 | 8.6e-07 / 8.2e-07 / 4.8e-07 / 4.8e-07 @12 | 1.6e-06 / 1.5e-06 / 9.8e-07 / 3.3e-07 @166 | 1.2e-07 | 2.58e-06 |
| 1.838e+08 | 2.1e-05 / 1.9e-05 / 3.3e-06 / 1.7e-05 @209 | 6.7e-07 / 6.5e-07 / 3.5e-07 / 3e-07 @20.7 | 1.8e-06 / 1.8e-06 / 8.7e-07 / 2.4e-07 @214 | 7.6e-09 | 2.52e-06 |
| 2.164e+08 | 1.4e-05 / 1.3e-05 / 3.6e-07 / 1.2e-05 @206 | 1.4e-06 / 1.4e-06 / 7e-07 / 1.3e-06 @210 | 4.8e-07 / 4.7e-07 / 1.2e-07 / 4.7e-07 @184 | 4.7e-09 | 2.52e-06 |
| 2.548e+08 | 0.00037 / 0.00023 / 3.2e-05 / 9.4e-05 @10.5 | 4.6e-06 / 4.6e-06 / 1.2e-07 / 2.6e-06 @109 | 5.4e-06 / 5.4e-06 / 4.5e-07 / 3.5e-06 @128 | 5.3e-06 | 7.49e-06 |
| 3e+08 | 1.3e-05 / 1.3e-05 / 2.5e-07 / 1.1e-05 @209 | 1.4e-06 / 1.4e-06 / 9.4e-07 / 1.1e-06 @8.71 | 9.7e-07 / 9.7e-07 / 6.2e-07 / 7e-07 @2.61 | 2.2e-08 | 2.52e-06 |

Initial-condition floor (production $T=1,T'=0$ against the series $1-x^2/10$, both at reference tolerance): min 2.3e-06, median 2.51e-06, max 7.49e-06 of the envelope, at $x_i$ = 0.003774--0.00389.

Eight wavenumbers exceed target at `1e-13` — $2.26\times10^5$, $8.37\times10^5$,
$1.16\times10^6$, $1.58\times10^7$, $1.87\times10^7$, $4.97\times10^7$, $6.89\times10^7$,
$2.55\times10^8$ — of which four are *spikes* rather than levels and one ($2.55\times10^8$, at
$4.6\times10^{-6}$) is below its own reference drift and should not be counted. Only three exceed
$10^{-5}$: $2.8\times10^{-4}$ at $x=10.3$, $1.6\times10^{-4}$ at $x=9.2$ and $3.6\times10^{-5}$
at $x=8.0$. `1e-16` leaves four, none of them levels, worst $2.5\times10^{-5}$ at
$k=3.05\times10^7$ — a wavenumber `1e-13` handles at $2.2\times10^{-6}$. The QCD initial-condition floor is the only
one that varies with $k$ ($2.30$ to $7.49\times10^{-6}$, median $2.51\times10^{-6}$), because
$c_s^2$ departs from $1/3$ across the QCD transition and $x_i$ is not quite the same at every $k$
($0.003774$ to $0.00389$).

---

## 6. Cost

Counts, not wall time (README §6's note 14: this machine's elapsed times overstate). Totals are
over all 50 wavenumbers, i.e. one production model's whole `TkNumericIntegration` stage.

| model | RHS evals, atol 1e-10 | 1e-13 | 1e-16 | 1e-13 vs 1e-10 | 1e-16 vs 1e-10 | k where 1e-16 is cheaper than 1e-13 |
|---|---|---|---|---|---|---|
| RadiationModel | 322228 | 401677 | 381547 | +24.7% | +18.4% | 34/50 |
| LambdaCDMModel | 328897 | 429178 | 431260 | +30.5% | +31.1% | 15/50 |
| QCDModel | 388357 | 485467 | 490729 | +25.0% | +26.4% | 15/50 |

`1e-13` costs +25 % to +30 % over `1e-10`, reproducing prompt 12's +14.3 % at $k=10^6$ as the low
end of a spread. **`1e-16` is not the free improvement prompt 12's single point suggested**: it is
cheaper than `1e-13` at 34 of the 50 radiation wavenumbers but at only 15 of 50 on each real
background, where the grid total is 0.5 % (LambdaCDM) and 1.1 % (QCD) *higher*. The "cheaper at
fewer evaluations" observation was a property of the control at one $k$.

---

## 7. Is it the absolute tolerance at all?

Neither diagnostic below is in the prompt's method. Both are here because the sweep's answer to
"does `1e-16` fix it" is *no*, and a recommendation to leave the constant alone has to say what
the excursions are if they are not an absolute-tolerance phenomenon. Each runs at the wavenumber
where `atol = 1e-13` is worst on that model.

**(a) Hold `atol = 1e-13` and tighten `rtol`**, at the wavenumber where the shipped tolerance is worst on each model:

| model | k [1/Mpc] | rtol 1e-08: max (evals) | rtol 1e-09: max (evals) | rtol 1e-10: max (evals) |
|---|---|---|---|---|
| RadiationModel | 3e+08 | 0.00025 @x=10.78 (8507) | 1.01e-07 @x=225.7 (10601) | 3.31e-08 @x=225.7 (11552) |
| LambdaCDMModel | 8.3657e+05 | 0.000864 @x=17.38 (8633) | 7.37e-08 @x=218.8 (10625) | 2.48e-08 @x=218.8 (11492) |
| QCDModel | 1.5842e+07 | 0.000283 @x=10.33 (9701) | 9.8e-07 @x=28.98 (12074) | 1.46e-06 @x=89.19 (13847) |**(b) Perturb $k$ at the shipped tolerance** (`atol = 1e-13`, `rtol = 1e-8`), same wavenumber. A feature of the solution survives a $10^{-6}$ change in $k$; an accident of step selection does not:

| model | k [1/Mpc] | baseline | k(1+1e-06) | k(1+1e-05) | k(1+1e-04) | k(1+1e-03) |
|---|---|---|---|---|---|---|
| RadiationModel | 3e+08 | 0.00025 @x=10.78 | 0.00025 @x=10.78 | 0.00025 @x=10.78 | 0.00025 @x=10.78 | 0.000249 @x=10.78 |
| LambdaCDMModel | 8.3657e+05 | 0.000864 @x=17.38 | 3.92e-07 @x=218.8 | 4.04e-07 @x=218.8 | 4.29e-07 @x=218.8 | 4.03e-07 @x=218.8 |
| QCDModel | 1.5842e+07 | 0.000283 @x=10.33 | 2.16e-06 @x=8.845 | 1.43e-06 @x=224.1 | 2.08e-06 @x=224.1 | 7.1e-07 @x=8.085 |

**(a)** says the excursion is a relative-tolerance phenomenon: at fixed `atol = 1e-13`, one decade
of `rtol` removes it on all three models — $2.5\times10^{-4}\to1.0\times10^{-7}$,
$8.6\times10^{-4}\to7.4\times10^{-8}$, $2.8\times10^{-4}\to9.8\times10^{-7}$ — for +25 %, +23 %
and +24 % evaluations, and the position of the maximum moves from $x\approx10$–$17$ to the far end
of the grid, i.e. the excursion is gone rather than reduced. A second decade buys little more, and
on QCD nothing (its reference floor, §4).

**(b)** says that on both *production* backgrounds the excursion is not a property of the
wavenumber: perturbing $k$ by one part in $10^6$ drops LambdaCDM's $8.6\times10^{-4}$ to
$3.9\times10^{-7}$ and QCD's $2.8\times10^{-4}$ to $2.2\times10^{-6}$, and it stays down through
$10^{-3}$. On the exact-radiation control it does *not* move — $2.50\times10^{-4}$ at every
perturbation — for the scaling reason given in §5.1. So the control is the one model on which the
excursion looks like a function of $k$, and it is the model prompt 12 measured.

Taken together: `rtol = 1e-8` occasionally selects a step sequence that mis-resolves the first few
oscillations as the mode enters the horizon, and the resulting error is carried to the end of the
run. `atol` changes which step sequence is selected, which is why `1e-13` and `1e-16` fail at
different wavenumbers, but it does not change how often this happens.

---

## 8. The initial-condition floor

README §2 (d) puts the floor at $2.5\times10^{-6}$ of the envelope with $T=1$, $T'=0$. Measured
two ways, and they agree to three figures:

* against the exact $T$ on the radiation control, the converged run with production initial data
  is wrong by $2.52\times10^{-6}$ at **every** one of the 50 wavenumbers (§5.1's oracle table);
* against a reference with the super-horizon series $T = 1 - x^2/10$ substituted for $T=1$, on all
  three models, the same $2.52\times10^{-6}$ — $2.30$ to $7.49\times10^{-6}$ on QCD.

It is $k$-independent because the production grid starts five e-folds outside the horizon at every
$k$, so $x_i = c_s e^{-5} = 0.00389$ everywhere and $x_i^2/10 = 1.5\times10^{-6}$; the QCD spread
is the $c_s^2$ departure across the QCD transition moving $x_i$ in its fourth digit.

**No wavenumber on any model is at the floor in the sense of being limited by it.** The sweep
measures solver error with the floor removed as common mode, so the two add: at a quiet
wavenumber the total is $2.5\times10^{-6}$ (floor) against $4\times10^{-7}$ (solver), and the
floor dominates — which is what README §6's row already says. At the 3, 13 and 8 offending
wavenumbers the solver error is 100× the floor and the floor is irrelevant. There is no
wavenumber where the answer is "this cannot be improved by any `atol`, it is the initial
condition": either the run is quiet, and the floor is what remains, or it has an excursion, and
the excursion is what remains.

---

*Runtime 276 s (QCD stand-in build 0.6 s of it); 50 wavenumbers x 3 models x 6-7 solves each, plus 14 per model for the diagnostics.*

---

## 9. After the split: prompt 18's measurement

<!-- generated 2026-09-13 by
     PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py --break-points
     in 740 s; Python 3.12.14, NumPy 2.2.4, SciPy 1.15.2 -->

`numeric_with_phase_cut` now asks the cosmology where its background quantities *jump* and
integrates the pieces between those points in sequence, instead of stepping across them in one
`solve_ivp` call (`prompts/GkTk-remedial` prompt 18). §§1–8 above were taken before that change
and are not rewritten; this section is the same §2.1 convergence test after it, extended to the
Green's-function sector, which had never been measured against a converged reference on any model.

**The one-paragraph answer.** The jumps were most of the cause but not all of it. On `QCDModel`
the transfer function's reference-convergence drift falls from 23 wavenumbers above the criterion
to **3**, and from a worst of $6.17\times10^{-6}$ to $1.97\times10^{-7}$; all four wavenumbers §4
names are fixed, by 347× to 5764×. $G_k$ **never had the failure** — its drift is
$8.4\times10^{-9}$ on QCD with or without the split, and $2\times10^{-11}$ on both smooth models —
so the failure is a $T_k$ phenomenon, consistent with review §12.5's observation that `atol` binds
for $T_k$ and never for $G_k$. The two smooth models declare nothing, take the unsplit code path
and reproduce §§3–4 exactly. The split costs **+1.38 %** ($T_k$) and **+0.39 %** ($G_k$) of the
production right-hand-side evaluations on QCD. It **moves** the production answer on QCD by up to
$2.8\times10^{-4}$ of the envelope, so a QCD datastore built before it is not reusable. What
remains is the $T(z)$ spline's $C^2$ knots: splitting at all 407 declared break points rather than
the 3 jumps closes the residue ($1.97\times10^{-7}\to4.65\times10^{-9}$) at **+219 % / +155 %** of
the production evaluations — §9.7, and the user's decision.

### 9.0 The control and the regression (prompt 18 §3 item 3)

`RadiationModel` and `LambdaCDMModel` declare no discontinuities, so they take the single-call
path and every figure of §§3–4 must reproduce *exactly*, not nearly. Prompt 12's two control
figures, against the exact $T$:

| control | prompt 12 | measured max dT/env | miss | at x | prompt 12 x | RHS evals |
|---|---|---|---|---|---|---|
| k=1e6 | 2.53e-06 | 2.534e-06 | 0.02% | 28.22 | -- | 7403 |
| k=3e8 | 0.000256 | 0.000256 | 0.01% | 10.78 | 10.8 | 8483 |

$2.534\times10^{-6}$ at $x = 28.22$ in **7403** right-hand-side evaluations and
$2.56\times10^{-4}$ at $x = 10.78$ in **8483** — the same two numbers and the same two counts as
§3, to the digit. §9.1's smooth rows likewise reproduce §4's: worst drift
$4.21\times10^{-11}$ at $k = 3\times10^8$ on Radiation and $5.7\times10^{-11}$ at
$k = 1.561\times10^8$ on LambdaCDM, with **1 segment** at every wavenumber. Separately,
`ComputeTargets/tests/test_numeric_break_points.py` asserts that a LambdaCDM $G_k$ run reproduces
an independently issued `solve_ivp` call **bit for bit** at every sample.

### 9.1 Is the reference converged now?

The criterion is prompt 17 §2.1's: the drift between the reference `(1e-18, 1e-12)` and a run one
decade tighter must be at most a tenth of the smallest candidate difference the sweep reports on
that model, i.e. $\le3.4\times10^{-8}$ of the envelope on QCD. "Max segments" is the number of
integrations one object is cut into — 1 means the historic single call.

| sector | model | max segments | worst drift, split | worst drift, unsplit | at k [1/Mpc] | median drift | k above 3e-08 | acceptance met? |
|---|---|---|---|---|---|---|---|---|
| Tk | RadiationModel | 1 | 4.21e-11 | -- | 3e+08 | 1.92e-11 | 0 | **yes** |
| Tk | LambdaCDMModel | 1 | 5.7e-11 | -- | 1.561e+08 | 3.76e-11 | 0 | **yes** |
| Tk | QCDModel | 2 | 1.97e-07 | 6.17e-06 | 4.287e+06 | 5.12e-09 | 3 | **no** |
| Gk | RadiationModel | 1 | 1.94e-11 | -- | 1.561e+08 | 1.35e-11 | 0 | **yes** |
| Gk | LambdaCDMModel | 1 | 2.1e-11 | -- | 2.197e+07 | 1.39e-11 | 0 | **yes** |
| Gk | QCDModel | 2 | 8.41e-09 | 8.89e-09 | 6.034e+05 | 2.05e-09 | 0 | **yes** |

### 9.2 The four wavenumbers §4 names

The four §4 could not measure through. Every one is now below the criterion, and the improvement
is 347× to 5764×. The $G_k$ rows are there to show that the same wavenumbers were never a problem
in that sector: the "before" and "after" columns agree, because the split changes the step
sequence but there was no error to remove.

| sector | k [1/Mpc] | drift before | drift after | improvement | <= 3e-08? |
|---|---|---|---|---|---|
| Tk | 1.584e+07 | 1.6e-06 | 4.6e-09 | 347x | yes |
| Tk | 4.972e+07 | 2.76e-06 | 6.48e-09 | 426x | yes |
| Tk | 5.855e+07 | 6.17e-06 | 7.94e-09 | 777x | yes |
| Tk | 2.548e+08 | 5.29e-06 | 9.18e-10 | 5764x | yes |
| Gk | 1.584e+07 | 2.87e-09 | 2.25e-09 | 1x | yes |
| Gk | 4.972e+07 | 2.32e-09 | 2.31e-09 | 1x | yes |
| Gk | 5.855e+07 | 6.87e-09 | 6.86e-09 | 1x | yes |
| Gk | 2.548e+08 | 1.54e-09 | 1.54e-09 | 1x | yes |

### 9.3 What the split costs, in right-hand-side evaluations

`T_120_MEV`'s crossing at $z = 8.64\times10^{11}$ is the only declared discontinuity inside any
production numeric range — `T_LO`'s at $z = 4.25\times10^{7}$ is below every grid and `T_HI` is
above the model's range — so every QCD object is cut into exactly two segments and pays one extra
startup transient: about 130 right-hand-side evaluations in the $T_k$ sector and about 50 in the
$G_k$ sector, against ~10k and ~13k per object.

| sector | model | per object, unsplit | per object, split | grid total, unsplit | grid total, split | change |
|---|---|---|---|---|---|---|
| Tk | QCDModel | 9709 | 9843 | 485467 | 492158 | +1.38% |
| Gk | QCDModel | 13268 | 13320 | 663403 | 666011 | +0.39% |

### 9.4 How far the split moves the production answer on QCD

The split changes what production computes on QCD, and this is by how much: the shipped-tolerance
run before the split, scored against the converged reference after it. The worst $T_k$ shift,
$2.8\times10^{-4}$ of the envelope, is two orders above README §6's $3\times10^{-6}$ row and 300×
the median solver error at the same tolerance. **A `QCDModel` datastore built before this commit
holds different numbers from one built after it, and `solver_serial` is not part of either numeric
lookup key** (`Datastore/SQL/ObjectFactories/GkNumericIntegration.py:221-227` and
`TkNumericIntegration.py:225-231` both filter on `wavenumber_exit_serial`, `model_serial`,
`atol_serial` and `rtol_serial` only), so the two are indistinguishable by key —
`[18-numeric-solver-not-in-lookup-key]`. `LambdaCDMModel` and `RadiationModel` rows are
bit-identical and need no regeneration.

| sector | model | worst shift | at k [1/Mpc] | median shift | smallest shift | median solver error at production tolerance |
|---|---|---|---|---|---|---|
| Tk | QCDModel | 0.000282 | 1.584e+07 | 8.53e-07 | 4.52e-08 | 8.35e-07 |
| Gk | QCDModel | 1.77e-06 | 2.197e+07 | 1.92e-07 | 1.96e-09 | 9.37e-07 |

### 9.5 Every wavenumber

Per wavenumber, per sector, per model. `segments` is 1 wherever the cosmology declares nothing.

#### Tk, RadiationModel

| k [1/Mpc] | segments | drift after |
|---|---|---|
| 1e+05 | 1 | 1.1e-11 |
| 1.178e+05 | 1 | 1.1e-11 |
| 1.387e+05 | 1 | 1.1e-11 |
| 1.633e+05 | 1 | 1.2e-11 |
| 1.922e+05 | 1 | 1.2e-11 |
| 2.264e+05 | 1 | 1.1e-11 |
| 2.665e+05 | 1 | 1.2e-11 |
| 3.139e+05 | 1 | 1.2e-11 |
| 3.696e+05 | 1 | 1.2e-11 |
| 4.352e+05 | 1 | 1.2e-11 |
| 5.124e+05 | 1 | 1.2e-11 |
| 6.034e+05 | 1 | 1.1e-11 |
| 7.105e+05 | 1 | 1.3e-11 |
| 8.366e+05 | 1 | 1.3e-11 |
| 9.851e+05 | 1 | 1.4e-11 |
| 1.16e+06 | 1 | 1.3e-11 |
| 1.366e+06 | 1 | 1.4e-11 |
| 1.608e+06 | 1 | 1.5e-11 |
| 1.894e+06 | 1 | 1.6e-11 |
| 2.23e+06 | 1 | 1.4e-11 |
| 2.626e+06 | 1 | 1.6e-11 |
| 3.092e+06 | 1 | 1.6e-11 |
| 3.64e+06 | 1 | 1.7e-11 |
| 4.287e+06 | 1 | 1.9e-11 |
| 5.048e+06 | 1 | 1.9e-11 |
| 5.943e+06 | 1 | 1.9e-11 |
| 6.998e+06 | 1 | 2.1e-11 |
| 8.241e+06 | 1 | 2.4e-11 |
| 9.703e+06 | 1 | 2.7e-11 |
| 1.143e+07 | 1 | 2.8e-11 |
| 1.345e+07 | 1 | 2.9e-11 |
| 1.584e+07 | 1 | 2.7e-11 |
| 1.865e+07 | 1 | 2.8e-11 |
| 2.197e+07 | 1 | 2.9e-11 |
| 2.586e+07 | 1 | 3.1e-11 |
| 3.045e+07 | 1 | 3.1e-11 |
| 3.586e+07 | 1 | 3.1e-11 |
| 4.223e+07 | 1 | 3.2e-11 |
| 4.972e+07 | 1 | 3.4e-11 |
| 5.855e+07 | 1 | 3.4e-11 |
| 6.894e+07 | 1 | 3.3e-11 |
| 8.118e+07 | 1 | 3.1e-11 |
| 9.558e+07 | 1 | 3.1e-11 |
| 1.126e+08 | 1 | 3.5e-11 |
| 1.325e+08 | 1 | 3.7e-11 |
| 1.561e+08 | 1 | 3.7e-11 |
| 1.838e+08 | 1 | 3.6e-11 |
| 2.164e+08 | 1 | 3.6e-11 |
| 2.548e+08 | 1 | 3.5e-11 |
| 3e+08 | 1 | 4.2e-11 |

#### Tk, LambdaCDMModel

| k [1/Mpc] | segments | drift after |
|---|---|---|
| 1e+05 | 1 | 3.8e-11 |
| 1.178e+05 | 1 | 3.7e-11 |
| 1.387e+05 | 1 | 3.8e-11 |
| 1.633e+05 | 1 | 4.4e-11 |
| 1.922e+05 | 1 | 3.5e-11 |
| 2.264e+05 | 1 | 4.5e-11 |
| 2.665e+05 | 1 | 3.8e-11 |
| 3.139e+05 | 1 | 3.8e-11 |
| 3.696e+05 | 1 | 4e-11 |
| 4.352e+05 | 1 | 4e-11 |
| 5.124e+05 | 1 | 3.6e-11 |
| 6.034e+05 | 1 | 3.8e-11 |
| 7.105e+05 | 1 | 3.6e-11 |
| 8.366e+05 | 1 | 3.8e-11 |
| 9.851e+05 | 1 | 3.6e-11 |
| 1.16e+06 | 1 | 4.9e-11 |
| 1.366e+06 | 1 | 3.6e-11 |
| 1.608e+06 | 1 | 3.6e-11 |
| 1.894e+06 | 1 | 3.7e-11 |
| 2.23e+06 | 1 | 3.8e-11 |
| 2.626e+06 | 1 | 3.6e-11 |
| 3.092e+06 | 1 | 4e-11 |
| 3.64e+06 | 1 | 3.7e-11 |
| 4.287e+06 | 1 | 3.7e-11 |
| 5.048e+06 | 1 | 3.4e-11 |
| 5.943e+06 | 1 | 3.8e-11 |
| 6.998e+06 | 1 | 3.7e-11 |
| 8.241e+06 | 1 | 3.4e-11 |
| 9.703e+06 | 1 | 3.6e-11 |
| 1.143e+07 | 1 | 3.9e-11 |
| 1.345e+07 | 1 | 3.9e-11 |
| 1.584e+07 | 1 | 4.1e-11 |
| 1.865e+07 | 1 | 4e-11 |
| 2.197e+07 | 1 | 3.6e-11 |
| 2.586e+07 | 1 | 3.8e-11 |
| 3.045e+07 | 1 | 3.4e-11 |
| 3.586e+07 | 1 | 3.6e-11 |
| 4.223e+07 | 1 | 4e-11 |
| 4.972e+07 | 1 | 3.7e-11 |
| 5.855e+07 | 1 | 3.8e-11 |
| 6.894e+07 | 1 | 3.9e-11 |
| 8.118e+07 | 1 | 3.6e-11 |
| 9.558e+07 | 1 | 4.2e-11 |
| 1.126e+08 | 1 | 3.9e-11 |
| 1.325e+08 | 1 | 3.6e-11 |
| 1.561e+08 | 1 | 5.7e-11 |
| 1.838e+08 | 1 | 3.5e-11 |
| 2.164e+08 | 1 | 3.4e-11 |
| 2.548e+08 | 1 | 5.3e-11 |
| 3e+08 | 1 | 3.7e-11 |

#### Tk, QCDModel

| k [1/Mpc] | segments | drift after | drift before | production shift | RHS evals before -> after |
|---|---|---|---|---|---|
| 1e+05 | 2 | 5.1e-09 | 5.9e-09 | 4.6e-07 | 9956 -> 10042 |
| 1.178e+05 | 2 | 5.6e-09 | 1.4e-07 | 8.4e-07 | 9974 -> 9970 |
| 1.387e+05 | 2 | 1.3e-08 | 6.6e-09 | 7.3e-07 | 9782 -> 10153 |
| 1.633e+05 | 2 | 2.6e-09 | 1.4e-07 | 5.1e-07 | 9911 -> 10006 |
| 1.922e+05 | 2 | 1.6e-09 | 4.3e-08 | 6.5e-07 | 9893 -> 9811 |
| 2.264e+05 | 2 | 5.2e-09 | 1.9e-08 | 3.2e-06 | 9830 -> 9928 |
| 2.665e+05 | 2 | 5.3e-09 | 3.4e-08 | 9.7e-07 | 9842 -> 10219 |
| 3.139e+05 | 2 | 4.7e-09 | 2.4e-08 | 8.7e-07 | 9854 -> 10180 |
| 3.696e+05 | 2 | 6.1e-09 | 3.1e-09 | 9e-07 | 10079 -> 10177 |
| 4.352e+05 | 2 | 4.4e-09 | 4.8e-08 | 1.1e-06 | 9923 -> 10159 |
| 5.124e+05 | 2 | 9.7e-09 | 1.3e-07 | 9.5e-07 | 9842 -> 10105 |
| 6.034e+05 | 2 | 5.6e-09 | 5.7e-09 | 6.2e-07 | 9977 -> 10216 |
| 7.105e+05 | 2 | 1.6e-08 | 8.1e-09 | 2.8e-06 | 10049 -> 10057 |
| 8.366e+05 | 2 | 6.1e-08 | 1.5e-08 | 4.9e-06 | 9581 -> 9970 |
| 9.851e+05 | 2 | 7.3e-09 | 7.5e-09 | 2.4e-06 | 9974 -> 10270 |
| 1.16e+06 | 2 | 8.9e-09 | 4.4e-08 | 7.6e-06 | 9728 -> 9991 |
| 1.366e+06 | 2 | 8.7e-09 | 4.8e-08 | 5.2e-07 | 9935 -> 9997 |
| 1.608e+06 | 2 | 1.1e-08 | 3.1e-08 | 9.9e-07 | 9902 -> 10231 |
| 1.894e+06 | 2 | 9.8e-09 | 5.3e-08 | 2.8e-07 | 9686 -> 10102 |
| 2.23e+06 | 2 | 3.5e-09 | 7.6e-09 | 9.6e-07 | 9788 -> 9886 |
| 2.626e+06 | 2 | 2e-08 | 4.1e-09 | 8.3e-07 | 9941 -> 10093 |
| 3.092e+06 | 2 | 1.8e-08 | 1.1e-07 | 2.8e-07 | 9935 -> 10090 |
| 3.64e+06 | 2 | 8.1e-09 | 7.7e-09 | 5.7e-07 | 9875 -> 9943 |
| 4.287e+06 | 2 | 2e-07 | 1e-08 | 6.6e-07 | 9713 -> 9931 |
| 5.048e+06 | 2 | 5.1e-09 | 8.8e-07 | 9.6e-07 | 9737 -> 9847 |
| 5.943e+06 | 2 | 6.2e-09 | 8.1e-07 | 1.2e-06 | 9644 -> 9874 |
| 6.998e+06 | 2 | 4.4e-09 | 3.3e-07 | 6.2e-07 | 9767 -> 9832 |
| 8.241e+06 | 2 | 1.2e-09 | 2.1e-07 | 9.8e-07 | 9647 -> 9649 |
| 9.703e+06 | 2 | 4.9e-09 | 6e-09 | 9.9e-07 | 9626 -> 9844 |
| 1.143e+07 | 2 | 2.8e-09 | 5.2e-07 | 4.7e-07 | 9818 -> 9793 |
| 1.345e+07 | 2 | 1.4e-08 | 2.8e-08 | 5.6e-07 | 9704 -> 9829 |
| 1.584e+07 | 2 | 4.6e-09 | 1.6e-06 | 0.00028 | 9701 -> 9751 |
| 1.865e+07 | 2 | 3.6e-09 | 2.5e-08 | 2.7e-06 | 9392 -> 9622 |
| 2.197e+07 | 2 | 3.7e-09 | 1.2e-06 | 3.4e-07 | 9665 -> 9853 |
| 2.586e+07 | 2 | 3.1e-09 | 5.2e-09 | 8.3e-07 | 9473 -> 9634 |
| 3.045e+07 | 2 | 3.8e-09 | 3.8e-09 | 2.9e-06 | 9659 -> 9526 |
| 3.586e+07 | 2 | 2.8e-09 | 1.6e-06 | 1.9e-06 | 9584 -> 9664 |
| 4.223e+07 | 2 | 3.5e-08 | 3e-08 | 9.1e-07 | 9644 -> 9550 |
| 4.972e+07 | 2 | 6.5e-09 | 2.8e-06 | 9.3e-07 | 9215 -> 9313 |
| 5.855e+07 | 2 | 7.9e-09 | 6.2e-06 | 5.7e-07 | 9596 -> 9595 |
| 6.894e+07 | 2 | 2.5e-09 | 4.2e-09 | 6.8e-06 | 9335 -> 9457 |
| 8.118e+07 | 2 | 2.1e-09 | 2e-09 | 6.4e-07 | 9242 -> 9328 |
| 9.558e+07 | 2 | 1.2e-09 | 7.8e-09 | 4.7e-07 | 9242 -> 9391 |
| 1.126e+08 | 2 | 3e-09 | 4.1e-09 | 3.9e-07 | 9407 -> 9577 |
| 1.325e+08 | 2 | 2.9e-09 | 2.9e-09 | 2e-06 | 9539 -> 9511 |
| 1.561e+08 | 2 | 8.9e-10 | 1.2e-07 | 1.3e-06 | 9542 -> 9544 |
| 1.838e+08 | 2 | 7.3e-09 | 7.6e-09 | 2.6e-07 | 9461 -> 9571 |
| 2.164e+08 | 2 | 4.7e-09 | 4.7e-09 | 3.6e-07 | 9656 -> 9667 |
| 2.548e+08 | 2 | 9.2e-10 | 5.3e-06 | 4.4e-07 | 9740 -> 9784 |
| 3e+08 | 2 | 3.8e-09 | 2.2e-08 | 4.5e-08 | 9461 -> 9625 |

#### Gk, RadiationModel

| k [1/Mpc] | segments | drift after |
|---|---|---|
| 1e+05 | 1 | 1.4e-11 |
| 1.178e+05 | 1 | 1.1e-11 |
| 1.387e+05 | 1 | 1.7e-11 |
| 1.633e+05 | 1 | 9.3e-12 |
| 1.922e+05 | 1 | 9.2e-12 |
| 2.264e+05 | 1 | 1.2e-11 |
| 2.665e+05 | 1 | 1.4e-11 |
| 3.139e+05 | 1 | 1.1e-11 |
| 3.696e+05 | 1 | 9.3e-12 |
| 4.352e+05 | 1 | 1.9e-11 |
| 5.124e+05 | 1 | 1e-11 |
| 6.034e+05 | 1 | 1.4e-11 |
| 7.105e+05 | 1 | 1.2e-11 |
| 8.366e+05 | 1 | 1.5e-11 |
| 9.851e+05 | 1 | 1.7e-11 |
| 1.16e+06 | 1 | 1.5e-11 |
| 1.366e+06 | 1 | 1.4e-11 |
| 1.608e+06 | 1 | 1.6e-11 |
| 1.894e+06 | 1 | 5e-12 |
| 2.23e+06 | 1 | 1.7e-11 |
| 2.626e+06 | 1 | 1.2e-11 |
| 3.092e+06 | 1 | 1.9e-11 |
| 3.64e+06 | 1 | 1.2e-11 |
| 4.287e+06 | 1 | 1.7e-11 |
| 5.048e+06 | 1 | 9.1e-12 |
| 5.943e+06 | 1 | 5.7e-12 |
| 6.998e+06 | 1 | 1.8e-11 |
| 8.241e+06 | 1 | 6.8e-12 |
| 9.703e+06 | 1 | 1.2e-11 |
| 1.143e+07 | 1 | 9.1e-12 |
| 1.345e+07 | 1 | 1.5e-11 |
| 1.584e+07 | 1 | 1.3e-11 |
| 1.865e+07 | 1 | 1.3e-11 |
| 2.197e+07 | 1 | 1.9e-11 |
| 2.586e+07 | 1 | 1.6e-11 |
| 3.045e+07 | 1 | 1.9e-11 |
| 3.586e+07 | 1 | 1.7e-11 |
| 4.223e+07 | 1 | 1.6e-11 |
| 4.972e+07 | 1 | 1.4e-11 |
| 5.855e+07 | 1 | 1.6e-11 |
| 6.894e+07 | 1 | 5.5e-12 |
| 8.118e+07 | 1 | 7e-12 |
| 9.558e+07 | 1 | 1.1e-11 |
| 1.126e+08 | 1 | 1.9e-11 |
| 1.325e+08 | 1 | 1.1e-11 |
| 1.561e+08 | 1 | 1.9e-11 |
| 1.838e+08 | 1 | 9.8e-12 |
| 2.164e+08 | 1 | 6.5e-12 |
| 2.548e+08 | 1 | 1.8e-11 |
| 3e+08 | 1 | 6.1e-12 |

#### Gk, LambdaCDMModel

| k [1/Mpc] | segments | drift after |
|---|---|---|
| 1e+05 | 1 | 1e-11 |
| 1.178e+05 | 1 | 7.9e-12 |
| 1.387e+05 | 1 | 1.9e-11 |
| 1.633e+05 | 1 | 1.7e-11 |
| 1.922e+05 | 1 | 1e-11 |
| 2.264e+05 | 1 | 1.8e-11 |
| 2.665e+05 | 1 | 1.3e-11 |
| 3.139e+05 | 1 | 4.5e-12 |
| 3.696e+05 | 1 | 1.6e-11 |
| 4.352e+05 | 1 | 1.9e-11 |
| 5.124e+05 | 1 | 1.7e-11 |
| 6.034e+05 | 1 | 2.1e-11 |
| 7.105e+05 | 1 | 1.4e-11 |
| 8.366e+05 | 1 | 8.5e-12 |
| 9.851e+05 | 1 | 6.6e-12 |
| 1.16e+06 | 1 | 7.3e-12 |
| 1.366e+06 | 1 | 1.4e-11 |
| 1.608e+06 | 1 | 1.4e-11 |
| 1.894e+06 | 1 | 7.9e-12 |
| 2.23e+06 | 1 | 1.4e-11 |
| 2.626e+06 | 1 | 1.2e-11 |
| 3.092e+06 | 1 | 2e-11 |
| 3.64e+06 | 1 | 1.1e-11 |
| 4.287e+06 | 1 | 7.3e-12 |
| 5.048e+06 | 1 | 1.6e-11 |
| 5.943e+06 | 1 | 1.7e-11 |
| 6.998e+06 | 1 | 1e-11 |
| 8.241e+06 | 1 | 1.5e-11 |
| 9.703e+06 | 1 | 1.1e-11 |
| 1.143e+07 | 1 | 4.2e-12 |
| 1.345e+07 | 1 | 1.6e-11 |
| 1.584e+07 | 1 | 1.8e-11 |
| 1.865e+07 | 1 | 1.8e-11 |
| 2.197e+07 | 1 | 2.1e-11 |
| 2.586e+07 | 1 | 1.8e-11 |
| 3.045e+07 | 1 | 9.9e-12 |
| 3.586e+07 | 1 | 1.1e-11 |
| 4.223e+07 | 1 | 7.7e-12 |
| 4.972e+07 | 1 | 1.5e-11 |
| 5.855e+07 | 1 | 1.6e-11 |
| 6.894e+07 | 1 | 8.4e-12 |
| 8.118e+07 | 1 | 1.1e-11 |
| 9.558e+07 | 1 | 1.2e-11 |
| 1.126e+08 | 1 | 2.1e-11 |
| 1.325e+08 | 1 | 1.1e-11 |
| 1.561e+08 | 1 | 1.1e-11 |
| 1.838e+08 | 1 | 1.9e-11 |
| 2.164e+08 | 1 | 1.5e-11 |
| 2.548e+08 | 1 | 1.1e-11 |
| 3e+08 | 1 | 1.4e-11 |

#### Gk, QCDModel

| k [1/Mpc] | segments | drift after | drift before | production shift | RHS evals before -> after |
|---|---|---|---|---|---|
| 1e+05 | 2 | 7.6e-10 | 1.8e-09 | 6.6e-07 | 12992 -> 13210 |
| 1.178e+05 | 2 | 6.3e-09 | 8.9e-09 | 1.4e-06 | 12998 -> 13180 |
| 1.387e+05 | 2 | 1.7e-09 | 3.8e-09 | 8.9e-07 | 13052 -> 13270 |
| 1.633e+05 | 2 | 1.3e-09 | 1.3e-09 | 6e-07 | 13166 -> 13258 |
| 1.922e+05 | 2 | 5.2e-09 | 5.3e-09 | 9.7e-07 | 13049 -> 13282 |
| 2.264e+05 | 2 | 1e-09 | 2.5e-09 | 1.2e-06 | 13127 -> 13270 |
| 2.665e+05 | 2 | 6.5e-09 | 7.3e-09 | 3.4e-07 | 13172 -> 13303 |
| 3.139e+05 | 2 | 4.4e-09 | 4.2e-09 | 5.1e-07 | 13229 -> 13321 |
| 3.696e+05 | 2 | 4.3e-09 | 3.6e-09 | 7.4e-07 | 13241 -> 13333 |
| 4.352e+05 | 2 | 4.4e-09 | 4.6e-09 | 1.7e-06 | 13244 -> 13483 |
| 5.124e+05 | 2 | 7.6e-09 | 7.8e-09 | 6.7e-07 | 13256 -> 13384 |
| 6.034e+05 | 2 | 8.4e-09 | 8.9e-09 | 4.5e-07 | 13418 -> 13423 |
| 7.105e+05 | 2 | 2.8e-09 | 2.7e-09 | 3.7e-07 | 13238 -> 13456 |
| 8.366e+05 | 2 | 6.4e-10 | 1.9e-09 | 4.4e-07 | 13193 -> 13336 |
| 9.851e+05 | 2 | 1.4e-09 | 3.2e-09 | 2.8e-07 | 13202 -> 13291 |
| 1.16e+06 | 2 | 7.6e-09 | 4.2e-09 | 2.6e-07 | 13172 -> 13315 |
| 1.366e+06 | 2 | 1e-09 | 1.4e-09 | 3.7e-07 | 13136 -> 13264 |
| 1.608e+06 | 2 | 4.5e-09 | 4e-09 | 4.8e-07 | 13037 -> 13267 |
| 1.894e+06 | 2 | 1.1e-09 | 1.8e-09 | 3.9e-07 | 13163 -> 13126 |
| 2.23e+06 | 2 | 1.4e-09 | 1.3e-09 | 1.8e-07 | 13328 -> 13381 |
| 2.626e+06 | 2 | 3e-09 | 2.9e-09 | 1.3e-07 | 13160 -> 13225 |
| 3.092e+06 | 2 | 4.6e-09 | 3.5e-09 | 4.5e-07 | 13106 -> 13285 |
| 3.64e+06 | 2 | 4.3e-09 | 4.6e-09 | 1.5e-07 | 13187 -> 13210 |
| 4.287e+06 | 2 | 1.4e-09 | 1.2e-09 | 2.7e-08 | 13277 -> 13390 |
| 5.048e+06 | 2 | 1.2e-09 | 7.8e-10 | 1.5e-07 | 13301 -> 13219 |
| 5.943e+06 | 2 | 1.6e-09 | 1.5e-09 | 1.1e-07 | 13154 -> 13186 |
| 6.998e+06 | 2 | 2e-09 | 2e-09 | 3.7e-07 | 13160 -> 13189 |
| 8.241e+06 | 2 | 7.8e-10 | 7.4e-10 | 2.3e-07 | 13073 -> 13210 |
| 9.703e+06 | 2 | 2e-09 | 2e-09 | 2.7e-07 | 13181 -> 13195 |
| 1.143e+07 | 2 | 5.2e-09 | 5.2e-09 | 1.3e-07 | 13238 -> 13129 |
| 1.345e+07 | 2 | 2.4e-09 | 2.4e-09 | 2.1e-07 | 13031 -> 13171 |
| 1.584e+07 | 2 | 2.2e-09 | 2.9e-09 | 7.7e-08 | 13214 -> 13120 |
| 1.865e+07 | 2 | 3.6e-09 | 3.5e-09 | 7.1e-08 | 13172 -> 13168 |
| 2.197e+07 | 2 | 2.3e-09 | 2.4e-09 | 1.8e-06 | 12953 -> 13156 |
| 2.586e+07 | 2 | 1.7e-09 | 1.7e-09 | 1.4e-07 | 13244 -> 13210 |
| 3.045e+07 | 2 | 2e-09 | 2e-09 | 4.3e-08 | 13076 -> 13111 |
| 3.586e+07 | 2 | 9.6e-10 | 9.6e-10 | 7.8e-09 | 13124 -> 13138 |
| 4.223e+07 | 2 | 8.1e-10 | 7.4e-10 | 1.2e-08 | 13367 -> 13189 |
| 4.972e+07 | 2 | 2.3e-09 | 2.3e-09 | 1.8e-07 | 13262 -> 13258 |
| 5.855e+07 | 2 | 6.9e-09 | 6.9e-09 | 2e-08 | 13259 -> 13255 |
| 6.894e+07 | 2 | 3.8e-09 | 3.8e-09 | 2.7e-09 | 13553 -> 13375 |
| 8.118e+07 | 2 | 7.2e-09 | 7.2e-09 | 4.9e-09 | 13529 -> 13486 |
| 9.558e+07 | 2 | 1.2e-09 | 1.2e-09 | 8.8e-09 | 13529 -> 13480 |
| 1.126e+08 | 2 | 2.1e-09 | 2.1e-09 | 8.2e-08 | 13532 -> 13501 |
| 1.325e+08 | 2 | 8.7e-10 | 8.7e-10 | 1.7e-08 | 13679 -> 13672 |
| 1.561e+08 | 2 | 1.3e-09 | 1.3e-09 | 3.4e-08 | 13643 -> 13627 |
| 1.838e+08 | 2 | 2.8e-09 | 2.8e-09 | 3.8e-08 | 13691 -> 13657 |
| 2.164e+08 | 2 | 1.6e-09 | 1.6e-09 | 4.3e-08 | 13625 -> 13576 |
| 2.548e+08 | 2 | 1.5e-09 | 1.5e-09 | 9.5e-08 | 13748 -> 13702 |
| 3e+08 | 2 | 1.7e-09 | 1.7e-09 | 2e-09 | 13922 -> 13768 |

### 9.6 Where the segment boundary is placed

Splitting at the declared crossing is not by itself enough, and this is the measurement that says
why. An explicit Runge-Kutta evaluates a stage at the far end of every step, so a segment ending
exactly on the crossing evaluates its last stage exactly there — and which branch of the equation
of state answers at that point is decided by the rounding of the cosmology's own $T(z)$ lookup, a
coin flip. When it lands on the far branch, the departing segment's final step is a straddling
step again, at the controller's full step size.

The same `QCDModel` $T_k$ reference-convergence drift, differing only in where the *declared crossing* is reported, as a relative displacement in $z$. The shipped `BREAK_POINT_STANDOFF` of $+10^{-12}$ is applied on top of whatever is declared, so the columns read: **as shipped** = one standoff on the near (higher-$z$) side, the side the departing segment lives on; $-10^{-12}$ = the two cancel and the boundary sits *on* the crossing; $-10^{-9}$ = the boundary is on the *far* side; the two positive columns move further onto the near side.

| k [1/Mpc] | as shipped | declared crossing +1e-12 | declared crossing -1e-12 | declared crossing +1e-09 | declared crossing -1e-09 |
|---|---|---|---|---|---|
| 1.5842e+07 | 4.6e-09 | 4.49e-09 | 4.8e-09 | 5.69e-09 | 5.84e-07 |
| 4.9721e+07 | 6.48e-09 | 6.48e-09 | 2.99e-06 | 6.3e-09 | 2.99e-06 |
| 5.8547e+07 | 7.94e-09 | 7.94e-09 | 7.94e-09 | 8.23e-09 | 8.5e-08 |

### 9.7 Would splitting at the C2 spline knots as well close the gap?

Prompt 18 §4 permits splitting at the 404 $C^2$ spline knots as well only if the cost is stated
and the user is asked. It is stated here and **not done**: no code change was made either way, and
the machinery is one argument. Note that an earlier run of this comparison, taken before the
standoff of §9.6 was added, showed the all-breaks split making $k = 4.972\times10^7$ *worse*
(7.06e-06); that was the boundary-placement effect, not the knots.

Accuracy, at the wavenumbers §9.1 leaves above the criterion -- the reference-convergence drift with the ODE split at the 3 declared jumps, and with it split at all 407 declared break points:

| k [1/Mpc] | drift, jumps only | drift, jumps + knots | reference evals, jumps only | reference evals, jumps + knots |
|---|---|---|---|---|
| 8.3657e+05 | 6.08e-08 | 6.89e-10 | 28420 | 47797 |
| 4.2867e+06 | 1.97e-07 | 4.65e-09 | 28162 | 48100 |
| 4.2226e+07 | 3.46e-08 | 2.3e-09 | 26977 | 48613 |

Cost, at the **production** tolerances, over every fifth wavenumber of the grid — which is the figure that matters, because `GkNumericIntegration` is one object per $(k, z_{\rm source})$ and there are ~65,000 of them per model:

| sector | k sampled | RHS evals, jumps only | jumps + knots | change |
|---|---|---|---|---|
| Tk | 10 | 98389 | 313698 | +218.8% |
| Gk | 10 | 132874 | 338526 | +154.8% |

*Runtime 740 s (QCD stand-in build 0.7 s of it); 50 wavenumbers x 3 models x 2 sectors, 3 solves each on a model that declares nothing and 6 on QCDModel.*

---

## 10. The per-sector policy: prompt 19's measurement

<!-- generated 2026-09-13 by
     PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py --per-sector
     in 574 s; Python 3.12.14, NumPy 2.2.4, SciPy 1.15.2 -->

Which kind of declared break point the numeric ODE splits at is no longer a property of the
integrator: `numeric_with_phase_cut` takes a `break_point_kind`, and the two sectors that share it
ask for different things (`prompts/GkTk-remedial` prompt 19, enacting the user's decision of
2026-09-13). §§1–8 were taken before prompt 18 and §9 before this change; neither is rewritten.
This section is §9's measurement once more, under the policy as it now stands.

**The one-paragraph answer.** The acceptance test passes. On `QCDModel` the transfer function's
reference-convergence drift is below the $3.4\times10^{-8}$ criterion at **all 50** production
wavenumbers — worst $8.72\times10^{-9}$, median $3.96\times10^{-9}$, against a worst of
$1.97\times10^{-7}$ and three offenders with the jumps alone — so §9.7's three-wavenumber
measurement generalises and the other 47 are not disturbed by splitting at the knots. $G_k$, whose
policy did not change, reproduces §9 **exactly**: 1.94e-11, 2.1e-11 and 8.41e-09 worst drift on
Radiation, LambdaCDM and QCD, at the same wavenumbers, and 13320 right-hand-side evaluations per
QCD object; at all 50 wavenumbers the run issued with the argument omitted is bit-identical to the
run issued with it named, which is what establishes that the default *is* the old behaviour. The
two smooth models take the single-`solve_ivp` path under **either** policy and are bit-identical
between them at all 50 wavenumbers in both sectors, prompt 17's two control figures included
(2.53e-06 in 7403 evaluations; 2.56e-04 in 8483). The cost is **+220 %** of the production
evaluations for $T_k$ on QCD and **+0.00 %** for $G_k$ — the same integer, not merely the same to
rounding. In seconds that is 0.98 s against 0.32 s per $T_k$ object, i.e. **49 s for the whole
sector** at 50 objects per model, against the 3.8 core-hours per model the same proportional
increase would have cost in $G_k$. The policy moves the $T_k$ answer on QCD a second time, by up
to $1.6\times10^{-4}$ of the envelope, so `[18-numeric-solver-not-in-lookup-key]` bites again and a
QCD datastore built before this commit is still not reusable.

### 10.1 The policy, and the acceptance test

`numeric_with_phase_cut` now takes a `break_point_kind`, defaulting to `BREAK_POINT_DISCONTINUITY` -- what it asked for unconditionally when §9 was measured. Both production integrators pass it explicitly: `TkNumericIntegration` asks for `BREAK_POINT_ALL` and `GkNumericIntegration` for `BREAK_POINT_DISCONTINUITY`. "Segments" below is the number of integrations one object is cut into under the sector's own policy, against the number the jumps alone would give.

| sector | model | policy | segments, shipped / jumps only | worst drift, jumps only | worst drift, shipped | at k [1/Mpc] | median drift, shipped | k above 3e-08 | acceptance met? |
|---|---|---|---|---|---|---|---|---|---|
| Tk | RadiationModel | all | 1 / 1 | 4.21e-11 | 4.21e-11 | 3e+08 | 1.92e-11 | 0 | **yes** |
| Tk | LambdaCDMModel | all | 1 / 1 | 5.7e-11 | 5.7e-11 | 1.561e+08 | 3.76e-11 | 0 | **yes** |
| Tk | QCDModel | all | 128 / 2 | 1.97e-07 | 8.72e-09 | 1e+05 | 3.96e-09 | 0 | **yes** |
| Gk | RadiationModel | discontinuity | 1 / 1 | 1.94e-11 | 1.94e-11 | 1.561e+08 | 1.35e-11 | 0 | **yes** |
| Gk | LambdaCDMModel | discontinuity | 1 / 1 | 2.1e-11 | 2.1e-11 | 2.197e+07 | 1.39e-11 | 0 | **yes** |
| Gk | QCDModel | discontinuity | 2 / 2 | 8.41e-09 | 8.41e-09 | 6.034e+05 | 2.05e-09 | 0 | **yes** |

### 10.2 The $G_k$ regression: §9 reproduced, bit for bit

$G_k$'s policy is unchanged by this prompt, so every §9 figure for that sector must come back unchanged -- and the driver itself was touched (a guard on the separation of segment boundaries), so this is the check that the sector which did not ask for a change did not get one. "default matches" is the same production run issued with `break_point_kind` omitted, compared sample by sample with `==`.

| model | §9.1 worst drift | measured now | reproduces? | k above 3e-08 | RHS evals per object | default matches explicit policy |
|---|---|---|---|---|---|---|
| RadiationModel | 1.94e-11 | 1.94e-11 | **yes** | 0 | 12743 | all 50 |
| LambdaCDMModel | 2.1e-11 | 2.1e-11 | **yes** | 0 | 12804 | all 50 |
| QCDModel | 8.41e-09 | 8.41e-09 | **yes** | 0 | 13320 | all 50 |

§9.3 gives 13320 right-hand-side evaluations per QCD $G_k$ object at the production tolerances; the table's QCD row is the same quantity.

### 10.3 The smooth-model regression

`RadiationModel` and `LambdaCDMModel` declare nothing, so both policies reach the same single-`solve_ivp` call. That is asserted twice over: the segment count is 1 under either policy at every wavenumber, and the production run issued under one policy is compared with the run issued under the other sample by sample with `==`.

| sector | model | max segments, shipped policy | max segments, jumps only | worst drift | policies bit-identical |
|---|---|---|---|---|---|
| Tk | RadiationModel | 1 | 1 | 4.21e-11 | all 50 |
| Tk | LambdaCDMModel | 1 | 1 | 5.7e-11 | all 50 |
| Gk | RadiationModel | 1 | 1 | 1.94e-11 | n/a (same request) |
| Gk | LambdaCDMModel | 1 | 1 | 2.1e-11 | n/a (same request) |

And prompt 17's two control figures, against the exact $T$, under the jumps-only policy (§9's column) and under the transfer function's shipped policy:

| control | prompt 12 | jumps only | RHS evals | shipped policy | RHS evals | identical? |
|---|---|---|---|---|---|---|
| k=1e6 | 2.53e-06 | 2.53e-06 | 7403 | 2.53e-06 | 7403 | yes |
| k=3e8 | 0.000256 | 0.000256 | 8483 | 0.000256 | 8483 | yes |

### 10.4 What the policy costs, in evaluations and in seconds

Right-hand-side evaluation counts are the reproducible measure (§5 note 14): they do not depend on the machine. The seconds below them are for scoping only -- they are what make the decision legible, because it turned on the sector's *object count* rather than its per-object cost.

| sector | model | policy | per object, jumps only | per object, shipped | grid total, jumps only | grid total, shipped | change |
|---|---|---|---|---|---|---|---|
| Tk | QCDModel | all | 9843 | 31521 | 492158 | 1576030 | +220.23% |
| Gk | QCDModel | discontinuity | 13320 | 13320 | 666011 | 666011 | +0.00% |

Wall time per object, best of 5 single-core runs at k = 4.972e+07/Mpc and the production tolerances. **The counts above are the measure; these seconds are for scoping.**

| sector | model | policy | s per object, shipped | s per object, jumps only | objects per model | sector total, shipped |
|---|---|---|---|---|---|---|
| Tk | RadiationModel | all | 0.0481 | 0.0464 | 50 | 2.4 s |
| Gk | RadiationModel | discontinuity | 0.0662 | 0.0658 | ~65,000 | 1.2 core-hours |
| Tk | LambdaCDMModel | all | 0.0584 | 0.0594 | 50 | 2.9 s |
| Gk | LambdaCDMModel | discontinuity | 0.0775 | 0.0803 | ~65,000 | 1.4 core-hours |
| Tk | QCDModel | all | 0.9805 | 0.3188 | 50 | 49.0 s |
| Gk | QCDModel | discontinuity | 0.2084 | 0.2093 | ~65,000 | 3.8 core-hours |

### 10.5 How far the transfer function's answer moves on QCD, again

The production-tolerance run under the jumps-only policy, scored against the converged reference under the shipped policy -- the same envelope-relative measure as §9.4. Prompt 18 already moved this sector's QCD values by up to 2.82e-04; this is the second move, on top of it.

| sector | model | worst shift | at k [1/Mpc] | median shift | smallest shift |
|---|---|---|---|---|---|
| Tk | QCDModel | 0.000161 | 4.972e+07 | 8.27e-07 | 1.71e-07 |

### 10.6 Every wavenumber

#### Tk, RadiationModel

| k [1/Mpc] | segments | drift, jumps only | drift, shipped | <= 3e-08? | production shift | RHS evals, jumps only -> shipped |
|---|---|---|---|---|---|---|
| 1e+05 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 7262 -> 7262 |
| 1.178e+05 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 7184 -> 7184 |
| 1.387e+05 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 7169 -> 7169 |
| 1.633e+05 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7094 -> 7094 |
| 1.922e+05 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7151 -> 7151 |
| 2.264e+05 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 7136 -> 7136 |
| 2.665e+05 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7106 -> 7106 |
| 3.139e+05 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7175 -> 7175 |
| 3.696e+05 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7058 -> 7058 |
| 4.352e+05 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7031 -> 7031 |
| 5.124e+05 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 7049 -> 7049 |
| 6.034e+05 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 7103 -> 7103 |
| 7.105e+05 | 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 7136 -> 7136 |
| 8.366e+05 | 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 7175 -> 7175 |
| 9.851e+05 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 7352 -> 7352 |
| 1.16e+06 | 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 7670 -> 7670 |
| 1.366e+06 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 7976 -> 7976 |
| 1.608e+06 | 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 8138 -> 8138 |
| 1.894e+06 | 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 8210 -> 8210 |
| 2.23e+06 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 8234 -> 8234 |
| 2.626e+06 | 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 8246 -> 8246 |
| 3.092e+06 | 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 8321 -> 8321 |
| 3.64e+06 | 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 8357 -> 8357 |
| 4.287e+06 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 8396 -> 8396 |
| 5.048e+06 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 8432 -> 8432 |
| 5.943e+06 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 8444 -> 8444 |
| 6.998e+06 | 1 | 2.1e-11 | 2.1e-11 | yes | 0 | 8369 -> 8369 |
| 8.241e+06 | 1 | 2.4e-11 | 2.4e-11 | yes | 0 | 8390 -> 8390 |
| 9.703e+06 | 1 | 2.7e-11 | 2.7e-11 | yes | 0 | 8387 -> 8387 |
| 1.143e+07 | 1 | 2.8e-11 | 2.8e-11 | yes | 0 | 8363 -> 8363 |
| 1.345e+07 | 1 | 2.9e-11 | 2.9e-11 | yes | 0 | 8366 -> 8366 |
| 1.584e+07 | 1 | 2.7e-11 | 2.7e-11 | yes | 0 | 8405 -> 8405 |
| 1.865e+07 | 1 | 2.8e-11 | 2.8e-11 | yes | 0 | 8519 -> 8519 |
| 2.197e+07 | 1 | 2.9e-11 | 2.9e-11 | yes | 0 | 8444 -> 8444 |
| 2.586e+07 | 1 | 3.1e-11 | 3.1e-11 | yes | 0 | 8456 -> 8456 |
| 3.045e+07 | 1 | 3.1e-11 | 3.1e-11 | yes | 0 | 8459 -> 8459 |
| 3.586e+07 | 1 | 3.1e-11 | 3.1e-11 | yes | 0 | 8645 -> 8645 |
| 4.223e+07 | 1 | 3.2e-11 | 3.2e-11 | yes | 0 | 8534 -> 8534 |
| 4.972e+07 | 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8573 -> 8573 |
| 5.855e+07 | 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8561 -> 8561 |
| 6.894e+07 | 1 | 3.3e-11 | 3.3e-11 | yes | 0 | 8570 -> 8570 |
| 8.118e+07 | 1 | 3.1e-11 | 3.1e-11 | yes | 0 | 8630 -> 8630 |
| 9.558e+07 | 1 | 3.1e-11 | 3.1e-11 | yes | 0 | 8657 -> 8657 |
| 1.126e+08 | 1 | 3.5e-11 | 3.5e-11 | yes | 0 | 8516 -> 8516 |
| 1.325e+08 | 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8528 -> 8528 |
| 1.561e+08 | 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8495 -> 8495 |
| 1.838e+08 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8552 -> 8552 |
| 2.164e+08 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8660 -> 8660 |
| 2.548e+08 | 1 | 3.5e-11 | 3.5e-11 | yes | 0 | 8486 -> 8486 |
| 3e+08 | 1 | 4.2e-11 | 4.2e-11 | yes | 0 | 8507 -> 8507 |

#### Tk, LambdaCDMModel

| k [1/Mpc] | segments | drift, jumps only | drift, shipped | <= 3e-08? | production shift | RHS evals, jumps only -> shipped |
|---|---|---|---|---|---|---|
| 1e+05 | 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8567 -> 8567 |
| 1.178e+05 | 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8528 -> 8528 |
| 1.387e+05 | 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8462 -> 8462 |
| 1.633e+05 | 1 | 4.4e-11 | 4.4e-11 | yes | 0 | 8639 -> 8639 |
| 1.922e+05 | 1 | 3.5e-11 | 3.5e-11 | yes | 0 | 8738 -> 8738 |
| 2.264e+05 | 1 | 4.5e-11 | 4.5e-11 | yes | 0 | 8714 -> 8714 |
| 2.665e+05 | 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8435 -> 8435 |
| 3.139e+05 | 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8648 -> 8648 |
| 3.696e+05 | 1 | 4e-11 | 4e-11 | yes | 0 | 8654 -> 8654 |
| 4.352e+05 | 1 | 4e-11 | 4e-11 | yes | 0 | 8468 -> 8468 |
| 5.124e+05 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8429 -> 8429 |
| 6.034e+05 | 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8564 -> 8564 |
| 7.105e+05 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8585 -> 8585 |
| 8.366e+05 | 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8633 -> 8633 |
| 9.851e+05 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8690 -> 8690 |
| 1.16e+06 | 1 | 4.9e-11 | 4.9e-11 | yes | 0 | 8597 -> 8597 |
| 1.366e+06 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8501 -> 8501 |
| 1.608e+06 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8702 -> 8702 |
| 1.894e+06 | 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8480 -> 8480 |
| 2.23e+06 | 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8549 -> 8549 |
| 2.626e+06 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8651 -> 8651 |
| 3.092e+06 | 1 | 4e-11 | 4e-11 | yes | 0 | 8648 -> 8648 |
| 3.64e+06 | 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8624 -> 8624 |
| 4.287e+06 | 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8531 -> 8531 |
| 5.048e+06 | 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8612 -> 8612 |
| 5.943e+06 | 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8624 -> 8624 |
| 6.998e+06 | 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8507 -> 8507 |
| 8.241e+06 | 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8621 -> 8621 |
| 9.703e+06 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8549 -> 8549 |
| 1.143e+07 | 1 | 3.9e-11 | 3.9e-11 | yes | 0 | 8588 -> 8588 |
| 1.345e+07 | 1 | 3.9e-11 | 3.9e-11 | yes | 0 | 8408 -> 8408 |
| 1.584e+07 | 1 | 4.1e-11 | 4.1e-11 | yes | 0 | 8651 -> 8651 |
| 1.865e+07 | 1 | 4e-11 | 4e-11 | yes | 0 | 8588 -> 8588 |
| 2.197e+07 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8612 -> 8612 |
| 2.586e+07 | 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8627 -> 8627 |
| 3.045e+07 | 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8609 -> 8609 |
| 3.586e+07 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8504 -> 8504 |
| 4.223e+07 | 1 | 4e-11 | 4e-11 | yes | 0 | 8612 -> 8612 |
| 4.972e+07 | 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8687 -> 8687 |
| 5.855e+07 | 1 | 3.8e-11 | 3.8e-11 | yes | 0 | 8732 -> 8732 |
| 6.894e+07 | 1 | 3.9e-11 | 3.9e-11 | yes | 0 | 8432 -> 8432 |
| 8.118e+07 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8561 -> 8561 |
| 9.558e+07 | 1 | 4.2e-11 | 4.2e-11 | yes | 0 | 8558 -> 8558 |
| 1.126e+08 | 1 | 3.9e-11 | 3.9e-11 | yes | 0 | 8588 -> 8588 |
| 1.325e+08 | 1 | 3.6e-11 | 3.6e-11 | yes | 0 | 8444 -> 8444 |
| 1.561e+08 | 1 | 5.7e-11 | 5.7e-11 | yes | 0 | 8564 -> 8564 |
| 1.838e+08 | 1 | 3.5e-11 | 3.5e-11 | yes | 0 | 8603 -> 8603 |
| 2.164e+08 | 1 | 3.4e-11 | 3.4e-11 | yes | 0 | 8501 -> 8501 |
| 2.548e+08 | 1 | 5.3e-11 | 5.3e-11 | yes | 0 | 8711 -> 8711 |
| 3e+08 | 1 | 3.7e-11 | 3.7e-11 | yes | 0 | 8648 -> 8648 |

#### Tk, QCDModel

| k [1/Mpc] | segments | drift, jumps only | drift, shipped | <= 3e-08? | production shift | RHS evals, jumps only -> shipped |
|---|---|---|---|---|---|---|
| 1e+05 | 128 | 5.1e-09 | 8.7e-09 | yes | 4.8e-07 | 10042 -> 29565 |
| 1.178e+05 | 128 | 5.6e-09 | 8.1e-09 | yes | 1.4e-06 | 9970 -> 29859 |
| 1.387e+05 | 127 | 1.3e-08 | 3.1e-09 | yes | 8.8e-07 | 10153 -> 29641 |
| 1.633e+05 | 128 | 2.6e-09 | 8.6e-09 | yes | 7.4e-07 | 10006 -> 29859 |
| 1.922e+05 | 128 | 1.6e-09 | 2.4e-09 | yes | 8.6e-07 | 9811 -> 30255 |
| 2.264e+05 | 128 | 5.2e-09 | 3.2e-09 | yes | 3.1e-07 | 9928 -> 30303 |
| 2.665e+05 | 128 | 5.3e-09 | 9.4e-10 | yes | 7.9e-07 | 10219 -> 30558 |
| 3.139e+05 | 127 | 4.7e-09 | 2.7e-09 | yes | 3.7e-07 | 10180 -> 30663 |
| 3.696e+05 | 128 | 6.1e-09 | 5e-09 | yes | 7.4e-07 | 10177 -> 30639 |
| 4.352e+05 | 128 | 4.4e-09 | 5.1e-09 | yes | 7.5e-07 | 10159 -> 30705 |
| 5.124e+05 | 128 | 9.7e-09 | 7.1e-09 | yes | 4.4e-07 | 10105 -> 30768 |
| 6.034e+05 | 127 | 5.6e-09 | 1.3e-09 | yes | 7.8e-07 | 10216 -> 30727 |
| 7.105e+05 | 127 | 1.6e-08 | 3e-09 | yes | 8.8e-07 | 10057 -> 30784 |
| 8.366e+05 | 127 | 6.1e-08 | 6.9e-10 | yes | 1.6e-06 | 9970 -> 30799 |
| 9.851e+05 | 127 | 7.3e-09 | 5.9e-10 | yes | 2.2e-06 | 10270 -> 31206 |
| 1.16e+06 | 127 | 8.9e-09 | 7.8e-10 | yes | 2.1e-07 | 9991 -> 31039 |
| 1.366e+06 | 127 | 8.7e-09 | 5.3e-09 | yes | 1.2e-06 | 9997 -> 30913 |
| 1.608e+06 | 127 | 1.1e-08 | 2e-09 | yes | 5.5e-07 | 10231 -> 31228 |
| 1.894e+06 | 127 | 9.8e-09 | 1.4e-09 | yes | 7.3e-07 | 10102 -> 31420 |
| 2.23e+06 | 127 | 3.5e-09 | 1.9e-09 | yes | 1.2e-06 | 9886 -> 31497 |
| 2.626e+06 | 127 | 2e-08 | 6.5e-09 | yes | 3.9e-07 | 10093 -> 31360 |
| 3.092e+06 | 127 | 1.8e-08 | 6.3e-10 | yes | 5.7e-07 | 10090 -> 31318 |
| 3.64e+06 | 127 | 8.1e-09 | 4.9e-09 | yes | 3.3e-06 | 9943 -> 31552 |
| 4.287e+06 | 127 | 2e-07 | 4.7e-09 | yes | 4.6e-07 | 9931 -> 31618 |
| 5.048e+06 | 127 | 5.1e-09 | 4.5e-09 | yes | 6.7e-07 | 9847 -> 31417 |
| 5.943e+06 | 127 | 6.2e-09 | 6.1e-09 | yes | 1.9e-07 | 9874 -> 31723 |
| 6.998e+06 | 127 | 4.4e-09 | 5e-09 | yes | 2.1e-06 | 9832 -> 31579 |
| 8.241e+06 | 127 | 1.2e-09 | 4.9e-09 | yes | 1.8e-06 | 9649 -> 32014 |
| 9.703e+06 | 127 | 4.9e-09 | 3.9e-09 | yes | 1e-06 | 9844 -> 31744 |
| 1.143e+07 | 126 | 2.8e-09 | 3.6e-09 | yes | 7.6e-07 | 9793 -> 31751 |
| 1.345e+07 | 126 | 1.4e-08 | 4e-09 | yes | 1.4e-06 | 9829 -> 31601 |
| 1.584e+07 | 126 | 4.6e-09 | 5e-09 | yes | 2.1e-06 | 9751 -> 31838 |
| 1.865e+07 | 127 | 3.6e-09 | 4.5e-09 | yes | 3.7e-05 | 9622 -> 31951 |
| 2.197e+07 | 127 | 3.7e-09 | 4.5e-09 | yes | 5.1e-07 | 9853 -> 32131 |
| 2.586e+07 | 126 | 3.1e-09 | 4.1e-09 | yes | 2.1e-06 | 9634 -> 32045 |
| 3.045e+07 | 126 | 3.8e-09 | 4.7e-09 | yes | 2.6e-06 | 9526 -> 32150 |
| 3.586e+07 | 127 | 2.8e-09 | 2.8e-09 | yes | 5.3e-07 | 9664 -> 32398 |
| 4.223e+07 | 127 | 3.5e-08 | 2.3e-09 | yes | 6.2e-07 | 9550 -> 32200 |
| 4.972e+07 | 126 | 6.5e-09 | 4e-09 | yes | 0.00016 | 9313 -> 32351 |
| 5.855e+07 | 126 | 7.9e-09 | 3.1e-09 | yes | 1e-05 | 9595 -> 32483 |
| 6.894e+07 | 126 | 2.5e-09 | 5.1e-09 | yes | 1.3e-05 | 9457 -> 32431 |
| 8.118e+07 | 127 | 2.1e-09 | 4.3e-09 | yes | 7.9e-06 | 9328 -> 32506 |
| 9.558e+07 | 126 | 1.2e-09 | 2.8e-09 | yes | 3.5e-07 | 9391 -> 32405 |
| 1.126e+08 | 126 | 3e-09 | 2.3e-09 | yes | 3.1e-07 | 9577 -> 32483 |
| 1.325e+08 | 126 | 2.9e-09 | 1.9e-09 | yes | 6.6e-07 | 9511 -> 32573 |
| 1.561e+08 | 126 | 8.9e-10 | 4.2e-09 | yes | 9e-07 | 9544 -> 32758 |
| 1.838e+08 | 126 | 7.3e-09 | 4.3e-09 | yes | 2.8e-06 | 9571 -> 32648 |
| 2.164e+08 | 126 | 4.7e-09 | 3e-09 | yes | 9.8e-07 | 9667 -> 32801 |
| 2.548e+08 | 126 | 9.2e-10 | 2.3e-09 | yes | 1.7e-07 | 9784 -> 32816 |
| 3e+08 | 126 | 3.8e-09 | 2e-09 | yes | 1.4e-06 | 9625 -> 32957 |

#### Gk, RadiationModel

| k [1/Mpc] | segments | drift, jumps only | drift, shipped | <= 3e-08? | production shift | RHS evals, jumps only -> shipped |
|---|---|---|---|---|---|---|
| 1e+05 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12764 -> 12764 |
| 1.178e+05 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12725 -> 12725 |
| 1.387e+05 | 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12764 -> 12764 |
| 1.633e+05 | 1 | 9.3e-12 | 9.3e-12 | yes | 0 | 12737 -> 12737 |
| 1.922e+05 | 1 | 9.2e-12 | 9.2e-12 | yes | 0 | 12677 -> 12677 |
| 2.264e+05 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12740 -> 12740 |
| 2.665e+05 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12701 -> 12701 |
| 3.139e+05 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12740 -> 12740 |
| 3.696e+05 | 1 | 9.3e-12 | 9.3e-12 | yes | 0 | 12713 -> 12713 |
| 4.352e+05 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12752 -> 12752 |
| 5.124e+05 | 1 | 1e-11 | 1e-11 | yes | 0 | 12692 -> 12692 |
| 6.034e+05 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12752 -> 12752 |
| 7.105e+05 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12704 -> 12704 |
| 8.366e+05 | 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12752 -> 12752 |
| 9.851e+05 | 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12704 -> 12704 |
| 1.16e+06 | 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12665 -> 12665 |
| 1.366e+06 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12728 -> 12728 |
| 1.608e+06 | 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12755 -> 12755 |
| 1.894e+06 | 1 | 5e-12 | 5e-12 | yes | 0 | 12740 -> 12740 |
| 2.23e+06 | 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12755 -> 12755 |
| 2.626e+06 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12740 -> 12740 |
| 3.092e+06 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12680 -> 12680 |
| 3.64e+06 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12755 -> 12755 |
| 4.287e+06 | 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12728 -> 12728 |
| 5.048e+06 | 1 | 9.1e-12 | 9.1e-12 | yes | 0 | 12755 -> 12755 |
| 5.943e+06 | 1 | 5.7e-12 | 5.7e-12 | yes | 0 | 12728 -> 12728 |
| 6.998e+06 | 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12755 -> 12755 |
| 8.241e+06 | 1 | 6.8e-12 | 6.8e-12 | yes | 0 | 12782 -> 12782 |
| 9.703e+06 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12743 -> 12743 |
| 1.143e+07 | 1 | 9.1e-12 | 9.1e-12 | yes | 0 | 12782 -> 12782 |
| 1.345e+07 | 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12767 -> 12767 |
| 1.584e+07 | 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 12719 -> 12719 |
| 1.865e+07 | 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 12770 -> 12770 |
| 2.197e+07 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12707 -> 12707 |
| 2.586e+07 | 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12770 -> 12770 |
| 3.045e+07 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12743 -> 12743 |
| 3.586e+07 | 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12770 -> 12770 |
| 4.223e+07 | 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12743 -> 12743 |
| 4.972e+07 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12695 -> 12695 |
| 5.855e+07 | 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12809 -> 12809 |
| 6.894e+07 | 1 | 5.5e-12 | 5.5e-12 | yes | 0 | 12782 -> 12782 |
| 8.118e+07 | 1 | 7e-12 | 7e-12 | yes | 0 | 12722 -> 12722 |
| 9.558e+07 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12695 -> 12695 |
| 1.126e+08 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12758 -> 12758 |
| 1.325e+08 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12785 -> 12785 |
| 1.561e+08 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12758 -> 12758 |
| 1.838e+08 | 1 | 9.8e-12 | 9.8e-12 | yes | 0 | 12797 -> 12797 |
| 2.164e+08 | 1 | 6.5e-12 | 6.5e-12 | yes | 0 | 12770 -> 12770 |
| 2.548e+08 | 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12797 -> 12797 |
| 3e+08 | 1 | 6.1e-12 | 6.1e-12 | yes | 0 | 12773 -> 12773 |

#### Gk, LambdaCDMModel

| k [1/Mpc] | segments | drift, jumps only | drift, shipped | <= 3e-08? | production shift | RHS evals, jumps only -> shipped |
|---|---|---|---|---|---|---|
| 1e+05 | 1 | 1e-11 | 1e-11 | yes | 0 | 12815 -> 12815 |
| 1.178e+05 | 1 | 7.9e-12 | 7.9e-12 | yes | 0 | 12740 -> 12740 |
| 1.387e+05 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12803 -> 12803 |
| 1.633e+05 | 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12767 -> 12767 |
| 1.922e+05 | 1 | 1e-11 | 1e-11 | yes | 0 | 12815 -> 12815 |
| 2.264e+05 | 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12815 -> 12815 |
| 2.665e+05 | 1 | 1.3e-11 | 1.3e-11 | yes | 0 | 12764 -> 12764 |
| 3.139e+05 | 1 | 4.5e-12 | 4.5e-12 | yes | 0 | 12815 -> 12815 |
| 3.696e+05 | 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12842 -> 12842 |
| 4.352e+05 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12815 -> 12815 |
| 5.124e+05 | 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12854 -> 12854 |
| 6.034e+05 | 1 | 2.1e-11 | 2.1e-11 | yes | 0 | 12827 -> 12827 |
| 7.105e+05 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12815 -> 12815 |
| 8.366e+05 | 1 | 8.5e-12 | 8.5e-12 | yes | 0 | 12854 -> 12854 |
| 9.851e+05 | 1 | 6.6e-12 | 6.6e-12 | yes | 0 | 12827 -> 12827 |
| 1.16e+06 | 1 | 7.3e-12 | 7.3e-12 | yes | 0 | 12815 -> 12815 |
| 1.366e+06 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12788 -> 12788 |
| 1.608e+06 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12803 -> 12803 |
| 1.894e+06 | 1 | 7.9e-12 | 7.9e-12 | yes | 0 | 12788 -> 12788 |
| 2.23e+06 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12815 -> 12815 |
| 2.626e+06 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12791 -> 12791 |
| 3.092e+06 | 1 | 2e-11 | 2e-11 | yes | 0 | 12752 -> 12752 |
| 3.64e+06 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12779 -> 12779 |
| 4.287e+06 | 1 | 7.3e-12 | 7.3e-12 | yes | 0 | 12827 -> 12827 |
| 5.048e+06 | 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12827 -> 12827 |
| 5.943e+06 | 1 | 1.7e-11 | 1.7e-11 | yes | 0 | 12767 -> 12767 |
| 6.998e+06 | 1 | 1e-11 | 1e-11 | yes | 0 | 12740 -> 12740 |
| 8.241e+06 | 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12854 -> 12854 |
| 9.703e+06 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12815 -> 12815 |
| 1.143e+07 | 1 | 4.2e-12 | 4.2e-12 | yes | 0 | 12815 -> 12815 |
| 1.345e+07 | 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12842 -> 12842 |
| 1.584e+07 | 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12815 -> 12815 |
| 1.865e+07 | 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12854 -> 12854 |
| 2.197e+07 | 1 | 2.1e-11 | 2.1e-11 | yes | 0 | 12788 -> 12788 |
| 2.586e+07 | 1 | 1.8e-11 | 1.8e-11 | yes | 0 | 12815 -> 12815 |
| 3.045e+07 | 1 | 9.9e-12 | 9.9e-12 | yes | 0 | 12842 -> 12842 |
| 3.586e+07 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12803 -> 12803 |
| 4.223e+07 | 1 | 7.7e-12 | 7.7e-12 | yes | 0 | 12791 -> 12791 |
| 4.972e+07 | 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12740 -> 12740 |
| 5.855e+07 | 1 | 1.6e-11 | 1.6e-11 | yes | 0 | 12815 -> 12815 |
| 6.894e+07 | 1 | 8.4e-12 | 8.4e-12 | yes | 0 | 12788 -> 12788 |
| 8.118e+07 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12815 -> 12815 |
| 9.558e+07 | 1 | 1.2e-11 | 1.2e-11 | yes | 0 | 12779 -> 12779 |
| 1.126e+08 | 1 | 2.1e-11 | 2.1e-11 | yes | 0 | 12827 -> 12827 |
| 1.325e+08 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12767 -> 12767 |
| 1.561e+08 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12752 -> 12752 |
| 1.838e+08 | 1 | 1.9e-11 | 1.9e-11 | yes | 0 | 12815 -> 12815 |
| 2.164e+08 | 1 | 1.5e-11 | 1.5e-11 | yes | 0 | 12779 -> 12779 |
| 2.548e+08 | 1 | 1.1e-11 | 1.1e-11 | yes | 0 | 12827 -> 12827 |
| 3e+08 | 1 | 1.4e-11 | 1.4e-11 | yes | 0 | 12815 -> 12815 |

#### Gk, QCDModel

| k [1/Mpc] | segments | drift, jumps only | drift, shipped | <= 3e-08? | production shift | RHS evals, jumps only -> shipped |
|---|---|---|---|---|---|---|
| 1e+05 | 2 | 7.6e-10 | 7.6e-10 | yes | 0 | 13210 -> 13210 |
| 1.178e+05 | 2 | 6.3e-09 | 6.3e-09 | yes | 0 | 13180 -> 13180 |
| 1.387e+05 | 2 | 1.7e-09 | 1.7e-09 | yes | 0 | 13270 -> 13270 |
| 1.633e+05 | 2 | 1.3e-09 | 1.3e-09 | yes | 0 | 13258 -> 13258 |
| 1.922e+05 | 2 | 5.2e-09 | 5.2e-09 | yes | 0 | 13282 -> 13282 |
| 2.264e+05 | 2 | 1e-09 | 1e-09 | yes | 0 | 13270 -> 13270 |
| 2.665e+05 | 2 | 6.5e-09 | 6.5e-09 | yes | 0 | 13303 -> 13303 |
| 3.139e+05 | 2 | 4.4e-09 | 4.4e-09 | yes | 0 | 13321 -> 13321 |
| 3.696e+05 | 2 | 4.3e-09 | 4.3e-09 | yes | 0 | 13333 -> 13333 |
| 4.352e+05 | 2 | 4.4e-09 | 4.4e-09 | yes | 0 | 13483 -> 13483 |
| 5.124e+05 | 2 | 7.6e-09 | 7.6e-09 | yes | 0 | 13384 -> 13384 |
| 6.034e+05 | 2 | 8.4e-09 | 8.4e-09 | yes | 0 | 13423 -> 13423 |
| 7.105e+05 | 2 | 2.8e-09 | 2.8e-09 | yes | 0 | 13456 -> 13456 |
| 8.366e+05 | 2 | 6.4e-10 | 6.4e-10 | yes | 0 | 13336 -> 13336 |
| 9.851e+05 | 2 | 1.4e-09 | 1.4e-09 | yes | 0 | 13291 -> 13291 |
| 1.16e+06 | 2 | 7.6e-09 | 7.6e-09 | yes | 0 | 13315 -> 13315 |
| 1.366e+06 | 2 | 1e-09 | 1e-09 | yes | 0 | 13264 -> 13264 |
| 1.608e+06 | 2 | 4.5e-09 | 4.5e-09 | yes | 0 | 13267 -> 13267 |
| 1.894e+06 | 2 | 1.1e-09 | 1.1e-09 | yes | 0 | 13126 -> 13126 |
| 2.23e+06 | 2 | 1.4e-09 | 1.4e-09 | yes | 0 | 13381 -> 13381 |
| 2.626e+06 | 2 | 3e-09 | 3e-09 | yes | 0 | 13225 -> 13225 |
| 3.092e+06 | 2 | 4.6e-09 | 4.6e-09 | yes | 0 | 13285 -> 13285 |
| 3.64e+06 | 2 | 4.3e-09 | 4.3e-09 | yes | 0 | 13210 -> 13210 |
| 4.287e+06 | 2 | 1.4e-09 | 1.4e-09 | yes | 0 | 13390 -> 13390 |
| 5.048e+06 | 2 | 1.2e-09 | 1.2e-09 | yes | 0 | 13219 -> 13219 |
| 5.943e+06 | 2 | 1.6e-09 | 1.6e-09 | yes | 0 | 13186 -> 13186 |
| 6.998e+06 | 2 | 2e-09 | 2e-09 | yes | 0 | 13189 -> 13189 |
| 8.241e+06 | 2 | 7.8e-10 | 7.8e-10 | yes | 0 | 13210 -> 13210 |
| 9.703e+06 | 2 | 2e-09 | 2e-09 | yes | 0 | 13195 -> 13195 |
| 1.143e+07 | 2 | 5.2e-09 | 5.2e-09 | yes | 0 | 13129 -> 13129 |
| 1.345e+07 | 2 | 2.4e-09 | 2.4e-09 | yes | 0 | 13171 -> 13171 |
| 1.584e+07 | 2 | 2.2e-09 | 2.2e-09 | yes | 0 | 13120 -> 13120 |
| 1.865e+07 | 2 | 3.6e-09 | 3.6e-09 | yes | 0 | 13168 -> 13168 |
| 2.197e+07 | 2 | 2.3e-09 | 2.3e-09 | yes | 0 | 13156 -> 13156 |
| 2.586e+07 | 2 | 1.7e-09 | 1.7e-09 | yes | 0 | 13210 -> 13210 |
| 3.045e+07 | 2 | 2e-09 | 2e-09 | yes | 0 | 13111 -> 13111 |
| 3.586e+07 | 2 | 9.6e-10 | 9.6e-10 | yes | 0 | 13138 -> 13138 |
| 4.223e+07 | 2 | 8.1e-10 | 8.1e-10 | yes | 0 | 13189 -> 13189 |
| 4.972e+07 | 2 | 2.3e-09 | 2.3e-09 | yes | 0 | 13258 -> 13258 |
| 5.855e+07 | 2 | 6.9e-09 | 6.9e-09 | yes | 0 | 13255 -> 13255 |
| 6.894e+07 | 2 | 3.8e-09 | 3.8e-09 | yes | 0 | 13375 -> 13375 |
| 8.118e+07 | 2 | 7.2e-09 | 7.2e-09 | yes | 0 | 13486 -> 13486 |
| 9.558e+07 | 2 | 1.2e-09 | 1.2e-09 | yes | 0 | 13480 -> 13480 |
| 1.126e+08 | 2 | 2.1e-09 | 2.1e-09 | yes | 0 | 13501 -> 13501 |
| 1.325e+08 | 2 | 8.7e-10 | 8.7e-10 | yes | 0 | 13672 -> 13672 |
| 1.561e+08 | 2 | 1.3e-09 | 1.3e-09 | yes | 0 | 13627 -> 13627 |
| 1.838e+08 | 2 | 2.8e-09 | 2.8e-09 | yes | 0 | 13657 -> 13657 |
| 2.164e+08 | 2 | 1.6e-09 | 1.6e-09 | yes | 0 | 13576 -> 13576 |
| 2.548e+08 | 2 | 1.5e-09 | 1.5e-09 | yes | 0 | 13702 -> 13702 |
| 3e+08 | 2 | 1.7e-09 | 1.7e-09 | yes | 0 | 13768 -> 13768 |

*Runtime 574 s (QCD stand-in build 0.6 s of it); 50 wavenumbers x 3 models x 2 sectors.*
