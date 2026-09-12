# Log 17 — Does one absolute tolerance cover the production $k$-grid? (measurement only)

**Prompt:** prompts/GkTk-remedial/17-tk-numeric-atol-k-sweep.md
**Commit:** *(this commit)* — Measure the transfer-function numeric tolerance across the k-grid
**Model:** Claude Opus 5
**Date:** 2026-09-12
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

No production code, no test, no `config/`, no `main.py`. Two new files under `docs/gktk-remedial/`,
plus this log, the board and `docs/OPEN_ISSUES.md`.

### `docs/gktk-remedial/tk_numeric_atol_sweep.py` (new, 946 lines)

Runs from the repository root, needs no Ray and no datastore, and calls
`numeric_with_phase_cut._function` with prompt 01's stand-ins — the pattern of prompt 02's
`residual_convergence.py`. Runtime **276 s**; the command line is in the document's header.

Public surface (all module level):

- `geometry(cosmology, k_inv_Mpc) -> dict` — `main.py`'s `build_Tk_numeric_work` geometry:
  production source grid from five e-folds outside the horizon, truncated at `0.85 z_e6`, stop
  window `(z_e3, z_e6)`.
- `run(model, k_inv_Mpc, geo, atol, rtol, ic=None) -> dict` — one `TkNumericIntegration` solve;
  `ic=None` is the production `T = 1, T' = 0`.
- `x_local(model, k_inv_Mpc, z) -> float` — `x = k c_s (1+z)/H`, identically `k c_s tau` in exact
  radiation.
- `sample_errors(model, k, geo, candidate, reference)` / `exact_errors(...)` — README §6's
  envelope-relative error against the reference run, and (radiation only) against the exact `T`.
- `summarise(errors) -> dict` — `max`, `max_z`, `max_x`, `second`, `median`, `terminal`,
  `terminal_x`, `samples`.
- `exact_initial_data(...)`, `series_initial_data(...)` — exact radiation `(T, dT/dz)`, and the
  super-horizon series `T = 1 - x^2/10` used to *measure* the initial-condition floor.
- `reproduce_control(radiation)` — prompt 12's two figures; raises and stops if either misses by
  more than 3 %.
- `sweep_model(name, model, cosmology, is_radiation)`, `rtol_ladder(...)`, `k_sensitivity(...)`,
  `worst_k_for(result, atol)`.

Constants: `PRODUCTION_K_GRID = np.logspace(log10(1e5), log10(3e8), 50)`,
`CANDIDATE_ATOL = (1e-10, 1e-13, 1e-16)`, `PRODUCTION_RTOL = 1e-8`,
`(REFERENCE_ATOL, REFERENCE_RTOL) = (1e-18, 1e-12)`,
`(TIGHTENED_ATOL, TIGHTENED_RTOL) = (1e-19, 1e-13)`, `RTOL_LADDER = (1e-8, 1e-9, 1e-10)`,
`K_PERTURBATIONS = (0, 1e-6, 1e-5, 1e-4, 1e-3)`.

Output is a markdown fragment on stdout and a progress log on stderr.

### `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` (new, 563 lines)

§1 the answers to the prompt's five questions and the recommendation, §1.1 what is not
established, §2 method, §3 the prompt 12 control, §4 reference convergence, §5 the sweep per model
(summary, shape, and every one of the 50 wavenumbers), §6 cost, §7 the two follow-up diagnostics,
§8 the initial-condition floor. Every table is the script's stdout verbatim; the prose is written
around it. The document carries the additive note at the top.

## Deviations from the prompt

### 1. The QCD reference does not meet §2.1's convergence test at four wavenumbers, and the sweep was reported anyway — STRUCTURALLY REQUIRED

The prompt says: "If it does not hold, say so and stop: a sweep measured against an unconverged
reference measures the reference."

It holds on `RadiationModel` (worst drift 4.21e-11) and `LambdaCDMModel` (5.70e-11) by three to
four orders. On `QCDModel` the median drift over the grid is 2.65e-08 but **four wavenumbers carry
1.6e-06 to 6.2e-06** — comparable with the candidate errors there. Tightening further does not
help and is not monotone (measured: at $k=10^8$, `(1e-19,1e-13)` moves the reference by 1.03e-06
while `(1e-20,1e-14)` moves it by 7.6e-09, and SciPy clamps `rtol` at 2.22e-14 anyway), so no
reference of this construction converges at those $k$.

The cause is not the tolerance: prompt 02 established that `QCD_Cosmology`'s $H(z)$ **jumps** at
the `QCD_EOS` branch boundaries, and a profile of the drift (a diagnostic run, not in the script)
shows it flat at 2–3e-09 down to $z\approx8.9\times10^{11}$ and stepping to 6e-07 immediately
below $z\approx8.6\times10^{11}$ — the `T_120_MEV` boundary, where `RESIDUAL-CONVERGENCE.md` §2
measures a 1.04e-04 relative jump in $H$. A discontinuous right-hand side reduces DOP853 to first
order across the jump, so a decade of tolerance buys about a quarter.

What was done instead of stopping: the convergence table is reported as its own section with the
failure stated in bold, the per-$k$ reference drift is carried as a **column of every per-$k$
table** so no QCD number can be read without its own noise floor beside it, and conclusions are
drawn only where the signal exceeds that floor (three of the four drifting wavenumbers carry
excursions 57×–175× their drift and are counted; $k=2.55\times10^8$'s 4.6e-06 is below its
5.3e-06 drift and is explicitly not counted). Stopping outright would have discarded the
LambdaCDM measurement — which is the decisive one, and which passes the test — and would have
withheld the answer to the prompt's own question for the sake of 4 of 150 (model, $k$) pairs.

### 2. Two diagnostics beyond the prompt's method: an `rtol` ladder and a $k$-perturbation — IMPLEMENTATION CHOICE

The prompt fixes `rtol = 1e-8` and sweeps `atol`. The sweep's answer to its own question 3 is that
`1e-16` does *not* fix the excursions, so a recommendation to leave the constant alone has to say
what they are if they are not an absolute-tolerance phenomenon. Two measurements at the worst $k$
of each model settle it (§7 of the document), at a cost of 14 extra solves per model:

- holding `atol = 1e-13` and stepping `rtol` to 1e-9 removes the excursion on all three models
  (8.64e-04 → 7.37e-08 on LambdaCDM) for +23 % evaluations;
- perturbing $k$ by one part in $10^6$ removes it on both *production* backgrounds (8.64e-04 →
  3.92e-07) but not on the radiation control, where it is unchanged at 2.50e-04 out to $10^{-3}$.

Alternatives considered: report "1e-16 does not fix it" and stop, leaving the user to guess
whether a $k$-dependent constant would; or sweep `rtol` across the grid as a fourth candidate. The
first leaves the recommendation unsupported; the second is a different prompt (and `rtol` is
shared by every integration in `main.py`, so it is not a local change). Both diagnostics are
measurement-only and change nothing.

### 3. `summarise` also records the error at the last returned sample — IMPLEMENTATION CHOICE

The prompt asks for maximum, second-largest, median and evaluation count. A fourth number, the
error at the deepest returned sample, is recorded and reported because it is what distinguishes
"one bad sample" from "one bad step whose consequence is carried to the end" — and the end of the
run is what `TkWKBIntegration` reads as its initial condition. It turned out to be the number that
answers the prompt's question 2: in every LambdaCDM offender the last sample is still wrong by
8.6e-06 to 2.2e-04.

### 4. The per-$k$ tables list all 50 wavenumbers, not only the offenders — IMPLEMENTATION CHOICE

The prompt asks for max/2nd/median "for every $k$ where the maximum exceeds 3e-6". At
`atol = 1e-10` that is all 50 $k$ on all three models, so the filter saves nothing on two thirds
of the table and hides the quiet baseline the offenders have to be read against. All 50 are listed,
with a preceding count-and-shape table so the offenders can be found without reading 50 rows.

### 5. The document is assembled from the script's stdout rather than written by the script — IMPLEMENTATION CHOICE

Prompt 02's `residual_convergence.py` regenerates its document in full. This one prints the tables
and the document is assembled around them, because prompt 17 §3 says
`TK-NUMERIC-ATOL-SWEEP.md` is **additive** — a re-run must add a section, not rewrite the file —
and a script that writes the whole file cannot honour that. No number in the document is typed by
hand: every table is a verbatim slice of the script's output.

## Verification performed

All figures measured, from `PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/tk_numeric_atol_sweep.py`
(276 s) unless stated. The document carries the full tables; the acceptance items are here.

### Prompt §2.4 — the control reproduces

| control | prompt 12 | measured | miss | at $x$ | RHS evals |
|---|---|---|---|---|---|
| $k=10^6$, `atol=1e-13`, vs exact $T$ | 2.534e-06 | **2.534e-06** | 0.02 % | 28.22 | **7403** |
| $k=3\times10^8$, `atol=1e-13`, vs exact $T$ | 2.56e-04 near $x\approx10.8$ | **2.56e-04** | 0.01 % | **10.78** | 8483 |

Both evaluation counts are prompt 12's as well. The script raises and stops if either misses by
more than 3 %, so this is a precondition of everything below, not a comparison made afterwards.

### Prompt §2.1 — the reference-convergence table

| model | worst drift | at $k$ | median drift over the grid | smallest candidate error reported | passes? |
|---|---|---|---|---|---|
| RadiationModel | 4.21e-11 | 3e+08 | 1.92e-11 | 1.32e-07 | yes |
| LambdaCDMModel | 5.70e-11 | 1.561e+08 | 3.76e-11 | 3.73e-07 | yes |
| QCDModel | **6.17e-06** | 5.855e+07 | 2.65e-08 | 3.45e-07 | **no** — deviation 1 |

The radiation control's second column checks the construction against a true oracle: the reference
run **with exact initial data** reproduces $3(\sin x - x\cos x)/x^3$ to **2.3e-11 – 4.3e-11** of
the envelope at every $k$, and with production initial data to 2.52e-06 at every $k$ — which is
the initial-condition floor and agrees with the series measure to three figures.

### The prompt's five questions

1. **Real off the control?** Yes. Worst $\delta T/\mathrm{env}$ at `atol = 1e-13`: **2.50e-04**
   (Radiation, $k=3\times10^8$, $x=10.78$), **8.64e-04** (LambdaCDM, $k=8.366\times10^5$,
   $x=17.38$), **2.83e-04** (QCD, $k=1.584\times10^7$, $x=10.33$). Above 3e-06: **3 / 50**,
   **13 / 50**, **8 / 50**.
2. **Spike or level?** Level. Of the offenders at `1e-13`: 3 of 3 (Radiation) and 13 of 13
   (LambdaCDM) have a median above 3e-06 (1.7e-05 – 5.8e-05 against 3e-08 – 7e-08 in a quiet run);
   QCD is 4 level, 4 spike. Last-sample errors in the LambdaCDM offenders: 8.6e-06 – 2.2e-04.
3. **Does `1e-16` fix it?** No. Radiation 3 → 0, but LambdaCDM 13 → **10** (worst 4.78e-04; five
   $k$ fail only at `1e-13`, **two only at `1e-16`**, eight at both) and QCD 8 → 4 (worst
   2.49e-05 at $k=3.045\times10^7$, which `1e-13` handles at 2.2e-06). Cost, summed over the grid
   (counts, not wall time): Radiation 322228 / 401677 / 381547, LambdaCDM 328897 / 429178 /
   431260, QCD 388357 / 485467 / 490729 at 1e-10 / 1e-13 / 1e-16. `1e-16` is cheaper than `1e-13`
   at **34 of 50** radiation wavenumbers but only **15 of 50** on each real background, where the
   grid total is +0.5 % and +1.1 % *higher*. Prompt 12's "cheaper" was a single point on the
   control.
4. **The initial-condition floor.** 2.52e-06 of the envelope, $k$-independent, measured twice —
   against the oracle on radiation (2.52e-06 at all 50 $k$) and against a super-horizon-series
   reference on all three models (2.52e-06; 2.30e-06 – 7.49e-06 on QCD, median 2.51e-06). It is
   $k$-independent because the grid starts five e-folds outside the horizon at every $k$, so
   $x_i = c_s e^{-5} = 0.00389$ throughout. No $k$ is *limited* by it: either the run is quiet and
   the floor is the residue, or it has an excursion 100× larger.
5. **Recommendation: keep `1e-13`,** because `atol` is not the lever — deviation 2's two
   diagnostics. The level `1e-13` was chosen for does improve uniformly: median over the 50 $k$ of
   the per-$k$ maximum, 1.25e-05 → 3.80e-07 (Radiation), 1.19e-05 → 4.52e-07 (LambdaCDM),
   1.38e-05 → 9.99e-07 (QCD), for +24.7 %, +30.5 %, +25.0 % evaluations; `1e-16` improves none of
   the three by a factor of three. A $k$-dependent constant is not supportable: the offending
   wavenumbers are not contiguous, they differ between `1e-13` and `1e-16`, and on the real
   backgrounds they move when $k$ moves in the sixth digit.

### Prompt §4 — acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` —
  **299 tests, OK**, unchanged; this prompt adds no test and removes none.
- `git diff HEAD~1 --stat` touches only `docs/gktk-remedial/tk_numeric_atol_sweep.py`,
  `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`, `docs/OPEN_ISSUES.md`,
  `prompts/GkTk-remedial/IMPLEMENTATION_STATE.md` and this log. Nothing under `ComputeTargets/`,
  `Quadrature/`, `LiouvilleGreen/`, `config/`; not `main.py`.
- `./venv/bin/python -m black --check docs/gktk-remedial/tk_numeric_atol_sweep.py` — clean.
- The §2.1 reference-convergence table is present as its own section (§4 of the document) and the
  §2.4 control is reproduced (§3).

### Not run

No pipeline execution and no datastore. Nothing here needs one: every solve is the undecorated
`numeric_with_phase_cut` on a stand-in model.

## Observations not acted on

1. **`rtol = 1e-8` is where the remaining $T_k$ numeric error lives, not `atol`.** One decade of
   `rtol` removes every excursion measured, at +23 %–25 % evaluations, and would take the whole
   grid to ~1e-07 of the envelope — below the 2.5e-06 initial-condition floor, so it would be
   spent on an error that is no longer the largest one. But `rtol` is a single shared constant
   (`main.py:2980-2998`) that keys every integration object in the datastore, so changing it is a
   pipeline-wide decision, not a $T_k$ one. Folded into the narrowed
   `[12-tk-numeric-atol-largest-k-excursion]` rather than opened as a new issue: prompt 17 §5 says
   the index count does not change, and it is the same excursion seen from its cause.
2. **`QCD_Cosmology`'s discontinuous $H(z)$ puts a floor under any ODE solution on that model.**
   Prompt 02 measured the jumps and split its *quadrature* panels at them; nothing splits the ODE
   there, and nothing can without changing the integrator. The consequence measured here is that
   two runs at very tight and slightly tighter tolerances differ by up to 6.2e-06 of the envelope
   at four wavenumbers. That is a bound on what any future QCD numeric measurement can resolve,
   and it is recorded in the narrowed issue for that reason. Not acted on: it is a property of the
   cosmology model, and no prompt in this campaign owns `CosmologyModels/`.
3. **At `atol = 1e-10` — the pre-prompt-12 setting — one LambdaCDM wavenumber reaches 2.8e-02 of
   the envelope** ($k=1.865\times10^7$, $x=40$), a 2.8 % error in $T$. It is the largest excursion
   anywhere in this sweep, it is on the configuration prompt 12 replaced, and it is a reason to
   keep prompt 12's change independently of anything argued above. Recorded, not acted on.
4. **The excursions sit an e-fold above the stop window's $z_{e3}$ edge**, at $x\approx8$–18,
   where the mode enters the horizon and the solution first oscillates. Whether the hand-over
   window should be anywhere near there is `docs/OPEN_ISSUES.md` §1.1's and explicitly out of
   scope (README §0.3); the document says so rather than drawing the inference.

## State handed to the next prompt

- **The recommendation: keep `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13`.** The three numbers
  behind it: (i) `1e-16` leaves **10 of 50** LambdaCDM wavenumbers above README §6's 3e-06 against
  `1e-13`'s 13, and two of those ten are wavenumbers `1e-13` handles — it is not an improvement,
  it is a different lottery; (ii) at the worst $k$ of each model, holding `atol = 1e-13` and
  tightening `rtol` 1e-8 → 1e-9 takes 8.64e-04 → 7.37e-08 (LambdaCDM), 2.50e-04 → 1.01e-07
  (Radiation), 2.83e-04 → 9.80e-07 (QCD) for +23 %–25 % evaluations, so the lever is `rtol`;
  (iii) perturbing $k$ by $10^{-6}$ at the shipped tolerance takes LambdaCDM's 8.64e-04 to
  3.92e-07 and QCD's 2.83e-04 to 2.16e-06, so on the production backgrounds the excursion is a
  step-sequence accident and not a function of $k$ — which is why no $k$-dependent constant is
  proposed.
- **Prompt 13 may build its datastore at `atol = 1e-13`.** Nothing measured here would move the
  constant, so no `TkNumericIntegration` row key changes. What prompt 13's verification document
  should *not* do is quote one wavenumber's $\delta T/\mathrm{env}$ as characteristic: the grid
  distribution at `1e-13` is median-of-per-$k$-maxima 3.8e-07 / 4.5e-07 / 1.0e-06 with 3 / 13 / 8
  wavenumbers above 3e-06 on Radiation / LambdaCDM / QCD, worst 8.64e-04.
- **The document:** `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`; the script,
  `docs/gktk-remedial/tk_numeric_atol_sweep.py` (276 s, no Ray, no datastore).
- **The measurement floor on QCD.** Any future ODE measurement on `QCDModel` is limited to about
  6e-06 of the envelope at $k\in\{1.58\times10^7, 4.97\times10^7, 5.86\times10^7,
  2.55\times10^8\}$ by the discontinuity in `QCD_Cosmology`'s $H(z)$, whatever tolerances are
  used. Prompt 13 should not score a QCD $T_k$ numeric quantity below that without splitting the
  integration at the break points.
- **The initial-condition floor is 2.52e-06 of the envelope at every $k$ on every model**
  (2.30e-06–7.49e-06 on QCD), because the grid starts five e-folds outside the horizon at every
  $k$ and $x_i = 0.00389$ throughout. README §2 (d)'s 2.5e-06 is confirmed against the exact $T$,
  not just asserted.
- `[12-tk-numeric-atol-largest-k-excursion]` is **narrowed, not closed** — the constant is the
  user's call, and the issue now carries the `rtol` finding as its next step.
