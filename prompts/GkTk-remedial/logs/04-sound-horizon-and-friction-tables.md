# Log 04 — The sound-horizon and friction tables

**Prompt:** prompts/GkTk-remedial/04-sound-horizon-and-friction-tables.md
**Commit:** *(this commit)* — Tabulate the sound horizon and the LG friction integral per model
**Model:** Claude Opus 5
**Date:** 2026-09-11
**Result:** COMPLETE WITH DEVIATIONS

`compute_background` now builds **three** Gauss–Legendre cumulative tables on the sample grid
instead of one: prompt 03's conformal time $\tau$, the sound horizon
$\tau_s=\int c_s\,dz/H$ with $c_s^2=$ `wPerturbations` (double-double, review §12.2), and the
Liouville–Green friction integral $F$ with $dF/dz=\tfrac32(1+c_s^2)/(1+z)$ (a single double,
review §12.7). `functions.cs_tau` and `functions.friction_F` are `TablePrimitive`s of exactly
prompt 03's kind, with the same `delta` sign convention. Measured on the production grid (1,732
nodes): **LambdaCDM $\tau_s$ 2.52e-16 and $F$ 3.31e-16 relative at the JSON checkpoints**
(targets $2\times10^{-14}$ and $10^{-13}$); **QCD $\tau_s$ 2.11e-14** against a reference whose
own floor is 1.89e-14, $F$ 3.34e-16; short baselines — including all three equation-of-state
break intervals and the steepest $c_s^2$ transition — $\le2.14\times10^{-15}$ relative for
$\tau_s$ and $\le8.23\times10^{-15}$ absolute for $F$. The friction ODE the tables replace is
measured, at production tolerances, at **2.261e-07 absolute in $F$** — review §12.3's 2.3e-7 —
i.e. seven orders worse than the table. Two deviations are `STRUCTURALLY REQUIRED` (the sign of
the integrand handed to `CumulativeTable`, and where the new integrand-evaluation counts are
reported); the rest are `IMPLEMENTATION CHOICE`; none is `UNINTENDED DRIFT`. `main.py` was not
touched, and did not need to be.

---

## What shipped

### `ComputeTargets/BackgroundModel.py`

- **Orders.** New module constants `CS_TAU_GAUSS_ORDER = 4` and `FRICTION_F_GAUSS_ORDER = 4`
  beside prompt 03's `TAU_GAUSS_ORDER = 4` (log 02: $N_{\tau_s}=N_F=4$). Separate names so a
  later re-measurement can move one alone; while all three agree, `TAU_SOLVER_LABEL`
  (`"cumulative-GL-stepping4"`) still describes the payload and **no `main.py` registration
  changes** (log 03's hand-over said this explicitly).
- **`_sound_speed_sq(cosmology, z)`** — `wPerturbations(z)`, raising `ValueError` naming the
  cosmology class, its `store_id` and $z$ when it is negative ("c_s^2 < 0 has no sound horizon").
- **`_cs_over_Hubble(cosmology)`** → `f(z) = sqrt(c_s^2)/H`; **`_friction_integrand(cosmology)`**
  → `f(z) = -1.5 (1 + c_s^2)/(1+z)`. The minus sign is deviation 1 below.
- **`compute_background`** builds both tables inside the existing supervisor block, on the same
  `break_points` as $\tau$. `cs_tau` is shifted by `cs_tau_init = sqrt(c_s^2(z_init)) * tau_init`,
  the radiation-era sibling of the author's `tau_init` asymptote (deviation 3); `friction_F` is
  anchored at zero at the top node. New payload keys: `"cs_tau_hi_sample"`, `"cs_tau_lo_sample"`,
  `"cs_tau_order"`, `"cs_tau_evaluations"`, `"friction_F_sample"`, `"friction_F_order"`,
  `"friction_F_evaluations"`. `IntegrationData.compute_time` now covers all three tables;
  `RHS_evaluations` still counts $\tau$'s alone (deviation 2).
- **`ModelFunctions`** gains `cs_tau` and `friction_F` as fields 14 and 15 with namedtuple
  `defaults=(None, None)`, so every thirteen-argument stand-in keeps constructing.
- **`BackgroundModel._create_functions`** builds them through two new methods,
  `_build_cs_tau_primitive()` and `_build_friction_F_primitive()`, alongside prompt 03's
  `_build_tau_primitive()`; a shared `_persisted_limbs(values, attr, label)` raises a
  `RuntimeError` naming the regeneration if any value carries `None`. The friction table is
  reconstructed with `lo = [0.0] * n`, as the prompt specifies.
- **`BackgroundModelValue`** gains keyword fields `cs_tau`, `cs_tau_lo`, `friction_F`
  (`Optional[float] = None`, after `tau_lo`) and the three matching properties;
  `values_from_payload` fills them.
- Docstrings: `TablePrimitive` and the `BackgroundModel` class docstring now name all three
  primitives; `_build_friction_F_primitive` carries the prompt's "a single double suffices"
  statement with its arithmetic.

### `Datastore/SQL/ObjectFactories/BackgroundModel.py`

- `sqla_BackgroundModelValue_factory.register()` gains `cs_tau_Mpc`, `cs_tau_lo_Mpc` and
  `friction_F`, all `Float(64)`, `nullable=False`, after `tau_lo_Mpc`. `friction_F` is
  dimensionless and carries no unit suffix.
- Written in `sqla_BackgroundModelFactory.store()` as `value.cs_tau / Mpc`,
  `value.cs_tau_lo / Mpc`, `value.friction_F` (no scaling); selected and read back in
  `sqla_BackgroundModelFactory.build()` as `row.cs_tau_Mpc * Mpc`, `row.cs_tau_lo_Mpc * Mpc`,
  `row.friction_F`; the same three added to both branches of
  `sqla_BackgroundModelValue_factory.build()` (payload keys `"cs_tau"`, `"cs_tau_lo"`,
  `"friction_F"`, matching prompt 03's treatment of `"tau_lo"`).
- The module's SCHEMA NOTE now lists all four new columns and what each holds; `build()`'s
  missing-column guard tests for any of `tau_lo_Mpc`, `cs_tau_Mpc`, `cs_tau_lo_Mpc`,
  `friction_F` and names the one it found in the error.

### `ComputeTargets/tests/test_background_cs_tau_friction.py` (new, 17 tests)

Four classes over one shared build (a radiation control plus both production models on the
production grid): `TestRadiationControl`, `TestProductionModels`,
`TestPayloadSchemaAndPersistence`, `TestFrictionODEComparison`. The prompt's six tests are all
there; the module adds the payload-shape, reconstruction, missing-limb and negative-$c_s^2$
tests, and a `_RadiationCosmology` stand-in (`H = H_0(1+z)^2`, $\rho = 3M_P^2H^2$, so
`tau_init` returns the closed form exactly) that drives the real `compute_background`.

---

## Deviations from the prompt

### 1. The integrand handed to `CumulativeTable` for $F$ carries a minus sign — `STRUCTURALLY REQUIRED`

**What the prompt assumed.** §2 item 4: "`friction_F(z)`, `friction_F.delta(a, b)`, the latter
reconstructed from a single-limb table (`lo` all zeros) with `f = 1.5*(1+wPerturbations(z))/(1+z)`
for partials."

**What is actually there.** `CumulativeTable` accumulates $T(z)=\int_z^{z_{\rm top}}f\,dz$, i.e.
$T'=-f$, and `delta(z_a, z_b) = T(z_b) - T(z_a)`. The primitive to be held is $F$ with
$dF/dz=+\tfrac32(1+c_s^2)/(1+z)$, so the integrand that makes the table hold $F$ is
$f=-\tfrac32(1+c_s^2)/(1+z)$. With the prompt's positive $f$ the table would hold $-F$ and
`friction_F.delta(z_a, z_b)` would be $F(z_a)-F(z_b)$ — the negative of the campaign's convention
(c), and the negative of what prompt 07 needs.

**What was done.** `_friction_integrand` returns the negative form. The shipped behaviour
satisfies, unchanged, every other statement the prompt and the campaign make about this quantity:

- README §2 (c): `X.delta(z_a, z_b) = X(z_b) - X(z_a)`. **Preserved, not altered** — this
  deviation exists in order to preserve it.
- Prompt §1: "`F` as `TkWKBIntegration` stores it today is $F(z)-F(z_{\rm init})$ with
  $F(z_{\rm init})=0$; prompt 07 will produce exactly that from `friction_F.delta(z_init, z)`."
  Reproduced: `friction_F.delta(z_init, z)` is negative for $z<z_{\rm init}$ and agrees with the
  ODE to 2.261e-07 (test 6).
- Prompt §3 test 1's formula, $2\ln\frac{1+z_b}{1+z_a}$. Reproduced to 7.1e-15 absolute. Note
  that the same sentence *labels* that expression "$F(z_a)-F(z_b)$", which is its negative; the
  formula, not the label, is what shipped, and the formula is the one consistent with convention
  (c), with prompt §1, and with `wkb_reference.py`'s own docstring ("`F` *decreases* towards
  lower redshift, where `tau` and `cs_tau` increase") and with the sign of the JSON's
  `friction_F_minus_top` records, which are negative.

The same reading applies to $\tau$: `dtau/dz = -1/H` and prompt 03 hands the table $+1/H$. The
rule is that the integrand is always *minus* the derivative of the primitive; the prompt's §2
item 4 omitted that minus, and it is stated in the new `_friction_integrand` docstring so the
next reader does not have to re-derive it.

### 2. The new integrand-evaluation counts are payload keys, not `IntegrationData` — `STRUCTURALLY REQUIRED`

**What the prompt assumed.** §2 item 1: "Add the integrand evaluations to the returned
`IntegrationData`."

**What is actually there.** `IntegrationData` (`Quadrature/integration_metadata.py`) is a fixed
six-field namedtuple with a single counter, `RHS_evaluations`, and prompt 03's
`test_background_tau.test_payload_shape` asserts

```python
self.assertEqual(payload["data"].RHS_evaluations, TAU_GAUSS_ORDER * (len(grid) - 1))
```

— an **exact** equality, on LambdaCDM, which any aggregate count breaks (the true total is 3×).
`ComputeTargets/tests/test_background_tau.py` is not in this prompt's "Files you may touch", and
prompt 04 §4 requires `discover -s ComputeTargets/tests` to pass; adding a field to
`IntegrationData` would touch every factory that constructs one.

**What was done.** `RHS_evaluations` keeps its prompt-03 meaning (the $\tau$ table's Hubble
evaluations) and the other two counts are reported as `payload["cs_tau_evaluations"]` and
`payload["friction_F_evaluations"]`; `compute_time` does cover all three tables. Both the
payload comment and the campaign board carry the reason. Opened as
`[04-background-rhs-evaluations-count]`; the fix, if the aggregate is wanted, is two lines in
`test_background_tau.py` plus the sum, in its own commit.

*Side effect worth knowing:* `test_background_tau.test_build_cost` prints `data.compute_time` as
"table … s"; that figure is now 0.145 s (LambdaCDM) / 0.409 s (QCD) because it covers three
tables, where log 03 recorded 0.064 s / 0.141 s for one. Nothing asserts on it except the
$\le0.5$ s LambdaCDM bound, which still holds.

### 3. `cs_tau`'s absolute anchor is the radiation-era asymptote — `IMPLEMENTATION CHOICE`

The prompt fixes no absolute normalisation for $\tau_s$, but its §3 test 1 requires
$\tau_s=\tau/\sqrt3$ **at the nodes** on the radiation control, which is a statement about the
absolute value. Alternatives: (a) anchor at zero at the top node, so `cs_tau(z)` is
"$\tau_s$ minus top" and only `delta` is meaningful — matches the JSON's `cs_tau_minus_top`
records but fails test 1 as written; (b) anchor at
$\tau_s(z_{\rm init})=c_s(z_{\rm init})\,\tau_{\rm init}$, the exact radiation-era companion of
the author's `tau_init` closed form, since $c_s$ is constant in the radiation era.

(b) was chosen: it is the same kind of object as `tau_init` (a convention fixed by the
radiation asymptote, README §5 rule 6), it makes the pointwise accessor physically meaningful
rather than grid-dependent, and it costs nothing — the shift is applied in double-double through
`CumulativeTable.shifted`, and every consumer in the campaign uses `delta`, which is unaffected
by the choice. It is asserted exactly (`cs_tau(z_top) == sqrt(wPerturbations(z_top)) * tau_init`)
so that a later change is caught.

### 4. Short-baseline $\tau_s$ references are built at test time — `IMPLEMENTATION CHOICE`

Prompt §3 test 2 asks for "`cs_tau.delta` over the short baselines to $\le10^{-13}$", but
`wkb_reference_data.json`'s `short_baseline` records carry **only** `delta_tau_full` and
`delta_tau_fraction` — there is no $\tau_s$ short-baseline reference in the JSON (prompt 01 built
them for $\tau$ alone). Alternatives: score against the JSON's cumulative `cs_tau_minus_top`
differenced across a baseline (which would measure the reference's cancellation, not the table);
regenerate the JSON (prompt 01's file, not this prompt's); or build a reference in the test.

The test builds one: `scipy.integrate.quad` at `epsrel=1e-13` in the **exact-width**
parametrisation $1+z=(1+z_{\rm lo})e^t$, $t\in[0,W]$ with
$W=\log1p((z_{\rm hi}-z_{\rm lo})/(1+z_{\rm lo}))$, with the cosmology's declared break points
passed as `points=`. That is the parametrisation `[03-qcd-short-baseline-reference-endpoint-rounding]`
identifies as the one that does not inherit $\mathrm{ulp}(u)/W$ from rounded endpoints — i.e. the
form log 03 recommends for exactly this measurement. Cost is a few hundred evaluations per
baseline.

### 5. Friction short baselines are scored absolutely, not relatively — `IMPLEMENTATION CHOICE`

A single-limb table's `delta` inherits up to one ulp of $\max|F|$ — 1.42e-14 on the production
grid, where $\max|F| = 70.585$ — **independently of the baseline**, because the error is in the
two stored node values, not in their subtraction. On a one-interval increment of 0.046 that is
1.3e-13 *relative* but 6.1e-15 absolute. Scoring such a baseline relatively would be asserting
against the storage decision the prompt itself took (§1: "a single double suffices … $10^{-14}$
absolute error in $F$ is $10^{-14}$ relative in the amplitude"), so the short-baseline friction
assertions are absolute, at `FRICTION_SHORT_BASELINE_ABS_TOL = 2e-14` (≈1.4 ulp of $\max|F|$),
with the relative figure printed. The **checkpoint** assertions, which the prompt states as
relative and which run over increments of order 1–70, are relative as asked.

### 6. Test 6's "2–4e-7 relative" is read as relative *in the amplitude* — `IMPLEMENTATION CHOICE`

Prompt §3 test 6 asks to show the ODE "differs from `friction_F.delta` by $2$–$4\times10^{-7}$
relative (review §12.3)". Review §12.3's table reports that figure in a column headed $\delta F$
and describes it in the text as "a relative amplitude error of 2.3e-7 ($k=10^5$) to 4.1e-7
($k=3\times10^8$)" — i.e. it is an **absolute** error in $F$, which is a relative error in
$e^F$. Relative *in $F$* the same discrepancy is 5.8e-09 ($F\approx-38.9$). The test asserts the
absolute-in-$F$ figure lies in $[5\times10^{-8},5\times10^{-6}]$ and prints both, with the
reading stated.

### 7. Two order constants rather than one — `IMPLEMENTATION CHOICE`

`CS_TAU_GAUSS_ORDER` and `FRICTION_F_GAUSS_ORDER` are separate names, both 4, rather than reuses
of `TAU_GAUSS_ORDER`. Log 02 decided the three orders independently and a later re-measurement
could move one; keeping them separate makes that a one-line change and keeps the payload's
`cs_tau_order` / `friction_F_order` honest. The cost is that the `IntegrationSolver` label is
derived from `TAU_GAUSS_ORDER` alone, which is correct only while the three agree — a comment on
the constants says so and points at `[03-integrationsolver-stepping-minimum-lookup]`.

### Not deviations, recorded to save the reader a check

- **No `main.py` hunk.** The prompt forbids touching `main.py`. None was needed: all three tables
  are order 4, so log 03's registration of `"cumulative-GL"` with `stepping=4` covers the
  payload, whose `solver_label` is unchanged.
- **Eleven extra tests** beyond the prompt's six (payload shape and orders, the reconstruction
  from persisted limbs, the missing-limb refusal, the negative-$c_s^2$ refusal, the sign and
  anchor contract, build cost). None changes production behaviour.

---

## Verification performed

Everything below was **run**, on this tree, with `PYTHONPATH=. ./venv/bin/python`.

### Prompt §3 test 1 — the exact-radiation control

`_RadiationCosmology` on a 1,301-node grid, $z=10^{12}\to0.1$ ($\max|F| = 55.07$), through the
real `compute_background` and `_create_functions`.

| quantity | measured | at | threshold |
|---|---|---|---|
| $\tau_s$ at nodes vs $1/(\sqrt3H_0(1+z))$ | **2.782e-16** | $z=7.586\times10^5$ | 2e-15 |
| $\tau_s$ at nodes vs $\tau/\sqrt3$ | **2.443e-16** | — | 2e-15 |
| `cs_tau.delta`, adjacent nodes | **4.797e-16** | $z=0.1318$ | 2e-15 |
| `cs_tau.delta`, 37 % fraction (off-grid endpoint) | **0.000e+00** | mid-grid | 2e-15 |
| `cs_tau.delta`, whole range | **2.115e-16** | — | 2e-15 |
| `friction_F.delta` vs $2\ln\frac{1+z_b}{1+z_a}$, adjacent | **7.105e-15** abs | $z=3.548\times10^9$ | 1e-14 abs |
| … 37 % fraction | **1.117e-15** abs | mid-grid | 1e-14 abs |
| … whole range ($F=-55.0714$) | **7.105e-15** abs | — | 1e-14 abs |

The adjacent-node and whole-range friction errors are the same number because both are the
single-limb floor, $\mathrm{ulp}(55.07) = 7.105\times10^{-15}$, and not quadrature error: in
exact radiation the friction integrand in $u$ is the constant 2, which order-4 Gauss integrates
exactly.

### Prompt §3 tests 2 and 3 — the production models on the production grid (1,732 nodes)

| model | quantity | measured | at | threshold |
|---|---|---|---|---|
| LambdaCDM | $\tau_s$ at the 13 JSON checkpoints | **2.521e-16** | $z=10.01$ | 2e-14 |
| LambdaCDM | $F$ at the checkpoints | **3.314e-16** rel (1.421e-14 abs) | $z=1.005\times10^7$ | 1e-13 |
| LambdaCDM | $\tau_s$ short baselines, one interval | 2.804e-16 / 1.275e-16 / **2.793e-16** | $z=1.004\times10^6$, 100.2, 1.001 | 1e-13 |
| LambdaCDM | … 37 % fraction (off-grid) | 0.0 / 1.724e-16 / **3.770e-16** | same | 1e-13 |
| LambdaCDM | $F$ short baselines | $\le$**2.377e-15** abs | $z=1.001$ | 2e-14 abs |
| QCD | $\tau_s$ at the checkpoints | **2.108e-14** | $z=1.005\times10^7$ | 5.660e-14 (3× the JSON's own 1.887e-14 floor) |
| QCD | $F$ at the checkpoints | **3.340e-16** rel (1.421e-14 abs) | $z=1.005\times10^7$ | 1e-13 |

QCD short baselines, the three JSON intervals **and** the transition intervals read from prompt
02's `convergence.geometry.QCDModel` block (`branch_boundaries[*].interval_index` and
`cs2_transition.interval_index`):

| interval | $z$ | what it carries | $\tau_s$ full / 37 % | $F$ abs full / 37 % |
|---|---|---|---|---|
| 862 | 4.9241e+07 | `T_LO` ($H$ jumps 4.4e-4) | **2.143e-15** / 1.459e-16 | **6.072e-15** / 3.469e-18 |
| 1031 | 1.0043e+06 | JSON baseline | 0.000e+00 / 9.533e-16 | 9.159e-16 / 3.469e-18 |
| 1107 | 1.7445e+05 | `EOS_T_LO` ($c_s^2$ slope jumps 99.5 %) | 3.956e-16 / 4.038e-16 | 1.124e-15 / 0.0 |
| 1293 | 2405.3 | `T_120_MEV` **and** the steepest $c_s^2$ transition | 2.084e-16 / 4.236e-16 | 9.645e-16 / 3.469e-18 |
| 1431 | 100.19 | JSON baseline | 5.101e-16 / 0.000e+00 | 1.353e-15 / 5.204e-18 |
| 1631 | 1.0006 | JSON baseline | 1.397e-16 / 3.771e-16 | **8.226e-15** / 8.674e-19 |

Worst over every QCD baseline: $\tau_s$ 2.143e-15 (threshold 1e-13), $F$ 8.226e-15 absolute
(threshold 2e-14). The break-point machinery is doing the work: these are the intervals log 02
showed unsplit order-4 Gauss failing on.

### Prompt §3 test 4 — the stand-in contract

`ModelFunctions(*thirteen_callables)` constructs; `cs_tau is None` and `friction_F is None`;
`_fields` has 15 entries ending `("cs_tau", "friction_F")`. `BackgroundModelValue` still
constructs from prompt 03's nine-argument call, with all three new fields `None`.
**`test_tk_source_functions.py` and `test_phase_groups.py` are unchanged and pass** — the proof
the prompt asks for.

### Prompt §3 test 5 — persistence

For every one of the 1,732 nodes of both models: `(cs_tau / Mpc) * Mpc == cs_tau` and
`(cs_tau_lo / Mpc) * Mpc == cs_tau_lo` exactly (`Mpc == 1.0` asserted), `|cs_tau_lo| <=
spacing(|cs_tau|)` (the low limb is a genuine second limb), and `friction_F` is a plain float
written with no unit scaling.

### Prompt §3 test 6 — the friction ODE is the inaccurate one

LambdaCDM, $k=10^5/{\rm Mpc}$, from $z_{e3}=2.3076\times10^9$ down to $z=0.1$ on the production
source grid, `solve_ivp(friction_RHS, DOP853, atol=1e-10, rtol=1e-8)` — the production call —
against `friction_F.delta(z_init, z)`:

> **2.261e-07 absolute in $F$** (i.e. relative in the amplitude $e^F$) at $z=0.1$, 5.810e-09
> relative in $F$, 1,814 RHS evaluations.

Review §12.3 measures 2.3e-7 for this $k$. The table's own error at the same nodes is the
1e-14-class figure tabulated above, so the difference is the ODE's.

### Build cost (recorded, not thresholded)

| model | $\tau$ / $\tau_s$ / $F$ integrand evaluations | all three tables | whole `compute_background` |
|---|---|---|---|
| LambdaCDM | 6,924 / 6,924 / 6,924 (20,772) | 0.145 s | 0.171 s |
| QCD | 8,552 / 8,552 / 8,552 (25,656) | 0.409 s | 0.553 s |

Log 02 predicted 20,772 / 25,656 evaluations for the three tables together, and 0.020 s / 0.305 s;
the totals match exactly and the times are the same order (log 02 timed the tables alone in a
script, this is inside `compute_background`).

### Suites

- `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` →
  **Ran 160 tests … OK** (143 before this commit; 17 new).
- `… discover -s CosmologyModels/tests -t .` → **Ran 11 tests … OK**.
- `ComputeTargets.tests.test_background_tau` alone → **Ran 14 tests … OK**, unchanged.
- `./venv/bin/python -m black --check` on all three touched Python files → clean.

Not run, and not runnable here: any live datastore exercise. The schema change is verified by
construction and by the round-trip test on the in-memory values, not against SQLite;
`sqla_BackgroundModelValue_factory.build()` remains unexercised
(`[03-backgroundmodelvalue-build-path]`).

---

## Observations not acted on

1. **`RHS_evaluations` now understates `compute_background`'s cost 3×.** Deviation 2; board
   issue `[04-background-rhs-evaluations-count]`; indexed in `docs/OPEN_ISSUES.md` §3.
2. **`[03-backgroundmodelvalue-build-path]` is confirmed again and again unrepaired.** This
   commit adds three keys to the same `build()` insert dict that still says `"wkb_serial"` where
   the column is `model_serial`, and reads `row_data.Hubble` where the select provides
   `Hubble_GeV`. Not this prompt's, and production does not take the path.
3. **`wkb_reference.QCDModel` now pays for two extra tables** on construction (it calls the
   undecorated `compute_background`), about +0.15 s per instance. Every test module that builds
   one absorbs it; the full suite went from ~110 s to 123 s. No action needed, but prompt 13's
   timings should not be compared with prompt 01's without knowing this.
4. **The JSON has no $\tau_s$ or $F$ short-baseline references** (deviation 4). If prompt 13
   wants those measurements against a cached reference rather than a test-time `quad`, prompt
   01's generator has to grow three more records per model.
5. **$\max|F| = 70.585$ on QCD puts the single-limb floor at 1.42e-14 absolute**, slightly above
   the $10^{-14}$ the prompt quotes for the design. It is still four orders below the friction
   ODE and seven below anything $T_k$ can resolve, so nothing is affected; but a later reader
   comparing $F$ increments at the $10^{-15}$ level should know the floor is one ulp of the
   *absolute* $F$, not of the increment. Recorded in the test module's docstring, not opened as
   an issue.

---

## State handed to the next prompt

**The two new accessors.** `model.functions.cs_tau` and `model.functions.friction_F` are
`ComputeTargets.BackgroundModel.TablePrimitive` objects, identical in interface to
`functions.tau` (log 03): `__call__(z) -> float`, `delta(z_a, z_b) -> float`, `.table`, `.label`
(`"cs_tau"`, `"friction_F"`). Both carry the cosmology's break points, both are order 4, both are
reconstructed from persisted limbs with **zero** quadrature on-grid.

**The sign convention prompt 07 must reproduce, stated three ways:**

```
friction_F.delta(z_a, z_b) = F(z_b) - F(z_a),          dF/dz = +(3/2)(1 + c_s^2)/(1+z)
friction_F.delta(z_init, z) < 0   for z < z_init
friction_F.delta(z_init, z) == what integrate_friction_function accumulates from F(z_init) = 0
```

So prompt 07 replaces `friction_sample[i]` with `friction_F.delta(z_init, z_sample[i])`
**directly, with no sign flip**, and `T = ... * exp(friction_sample[i])` keeps working unchanged.
Measured agreement with the ODE it retires: 2.261e-07 absolute in $F$ (LambdaCDM, $k=10^5$, at
$z=0.1$), which is the ODE's error, not the table's. `cs_tau.delta(z_a, z_b) = cs_tau(z_b) -
cs_tau(z_a) > 0` for $z_b<z_a$, so the transfer-function phase is
$\theta_T(z;z_i) = -k\,$`cs_tau.delta(z_i, z)`$\,-\,\Delta\rho_T$, exactly as $\tau$ gives
$\theta_G$ (README §2 (a), (c)).

**Pointwise anchors.** `cs_tau(z_top) = sqrt(wPerturbations(z_top)) * tau_init` (the radiation
asymptote, deviation 3); `friction_F(z_top) = 0.0` exactly. Only `delta` should be used
downstream; both anchors are conventions.

**Payload keys added to `compute_background`** (prompt 03's unchanged):
`"cs_tau_hi_sample"`, `"cs_tau_lo_sample"` (lists of floats in `z_sample` order),
`"cs_tau_order"` (= 4), `"cs_tau_evaluations"`, `"friction_F_sample"` (the high limb only),
`"friction_F_order"` (= 4), `"friction_F_evaluations"`. `"solver_label"` is still
`"cumulative-GL-stepping4"` and **`main.py` needs no change**.

**Module-level and class names added to `ComputeTargets/BackgroundModel.py`:**
`CS_TAU_GAUSS_ORDER = 4`, `FRICTION_F_GAUSS_ORDER = 4`, `_sound_speed_sq(cosmology, z)`,
`_cs_over_Hubble(cosmology)`, `_friction_integrand(cosmology)`,
`BackgroundModel._build_cs_tau_primitive()`, `BackgroundModel._build_friction_F_primitive()`,
`BackgroundModel._persisted_limbs(values, attr, label)`. `ModelFunctions` is now a 15-field
namedtuple with `defaults=(None, None)`; `BackgroundModelValue(..., tau_lo=0.0, cs_tau=None,
cs_tau_lo=None, friction_F=None)` with properties `.cs_tau`, `.cs_tau_lo`, `.friction_F`.

**Columns.** `BackgroundModelValue.cs_tau_Mpc`, `cs_tau_lo_Mpc` (`Float(64)`, `nullable=False`,
written as `value.cs_tau / Mpc`, read as `row.cs_tau_Mpc * Mpc`, exact in `Mpc_units`) and
`friction_F` (`Float(64)`, `nullable=False`, **dimensionless — written and read with no unit
conversion**), all after `tau_lo_Mpc`.

**Datastore.** Unchanged from prompt 03 in effect: every existing datastore must be regenerated.
The factory now refuses one missing any of `tau_lo_Mpc`, `cs_tau_Mpc`, `cs_tau_lo_Mpc`,
`friction_F`, naming the column it found missing.

**Accuracies prompt 07 and prompt 13 may rely on** (production grid, order 4):

- $\tau_s$ at the nodes: LambdaCDM **2.52e-16**, QCD **2.11e-14** (reference floor 1.89e-14 —
  do not assert below 3× it, `[02-qcd-reference-floor]`).
- $\tau_s$ over one production interval, on- or off-grid endpoint: $\le$**2.14e-15** on both
  models, including all three equation-of-state break intervals.
- $F$ at the nodes: **3.3e-16** relative on both models; **absolute** error of any $F$ increment
  $\le$ 1 ulp of $\max|F|$ = **1.42e-14** (QCD; 55.07 and 7.1e-15 on a radiation grid to
  $z=10^{12}$). In amplitude that is 1.4e-14 relative — seven orders below the $T_k$ LG
  truncation floor `[00-tk-lg-truncation-floor]`.
- $k\,\Delta\tau_s$ at $k=3\times10^8$ reaches 1.85e11 rad (review §12.2), so the double-double
  limbs matter for $\tau_s$ exactly as for $\tau$; $F$ does not need them.

**Build cost** (all three tables): LambdaCDM 20,772 integrand evaluations, 0.145 s of the 0.171 s
`compute_background`; QCD 25,656, 0.409 s of 0.553 s (plus ~0.02–0.3 s to construct
`QCD_Cosmology` itself).

**A guard prompt 07 may trip:** `compute_background` raises `ValueError` naming the cosmology and
$z$ if `wPerturbations(z) < 0` anywhere on the grid or in an off-grid partial. Neither production
model does ($c_s^2\in[1.08\times10^{-4},\,0.3336]$ over the production range on both), but a
stand-in that returns a negative $c_s^2$ will now fail loudly at table-build time rather than
returning NaN.
