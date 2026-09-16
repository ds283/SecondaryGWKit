# Log 01 — The equality-solve characterisation test

**Prompt:** prompts/background-solver-robustness/01-equality-solve-characterisation.md
**Commit:** *(this commit)* — Characterise the equality solve before bracketing it
**Model:** Claude Opus 5
**Date:** 2026-09-16
**Result:** COMPLETE WITH DEVIATIONS

The deviations are two acceptance thresholds, both widened from the prompt's stated figure and
both to a value the measurement chose. **Neither is a loosening to accommodate a failure**: the
prompt's "≤ 2 ulp" is arithmetically incompatible with `AUDIT.md` §2.2's own headline −4.00e-16,
which *is* 3.0 ulp at $z = 3406.67$, and the audit's measurement reproduces at this tree to the
digit printed. D1 below states the arithmetic.

---

## What shipped

**One new file, no production file in the diff.**

- `CosmologyModels/tests/test_rho_equality.py` (**new**, 368 lines) — four tests in one
  `TestRhoEquality` class, both models built once in `setUpClass`:
  - `test_equality_redshifts_match_a_bracketed_reference` — all four (model, pair) combinations
    against an independent bracketed `brentq`, at `SOLVE_VS_REFERENCE_ULP = 4`.
  - `test_the_density_ratios_are_monotone_where_their_roots_live` — audit §4.1's two probe sets,
    strict, on **both** models.
  - `test_the_lambda_equality_is_closed_form_on_any_equation_of_state` — at
    `LAMBDA_CLOSED_FORM_ULP = 2`, as the prompt specifies.
  - `test_the_matter_radiation_closed_form_is_exact_only_because_g_star_is_flat` — $g_*(T)$
    float-equal at $z_{\rm eq}/10$, $z_{\rm eq}$, $10z_{\rm eq}$ on `QCD_Cosmology`, plus the
    closed form against the reference at `MATTER_RADIATION_CLOSED_FORM_ULP = 8`.

  New module-level symbols: `species_of(pair: str)`, `closed_form_guess(model, pair: str) -> float`,
  `match_rho(model, pair: str)` (returns the residual closure), `bracketed_reference(model, pair:
  str) -> float`, `ulps_between(got: float, reference: float) -> float`; constants
  `BRENT_RTOL_FLOOR = 8.9e-16`, `BRENT_XTOL_DISABLED = 1.0e-300`, `QCD_MAX_Z = 1.0e12`,
  `RADIATION_MAX_Z = 1.0e6`, `SOLVE_VS_REFERENCE_ULP = 4`, `LAMBDA_CLOSED_FORM_ULP = 2`,
  `MATTER_RADIATION_CLOSED_FORM_ULP = 8`, `MATTER_RADIATION_PROBES`, `MATTER_LAMBDA_PROBES`.

- `prompts/background-solver-robustness/IMPLEMENTATION_STATE.md` — board row 01, item rows (a) and
  (d), and the §3 entry `[01-agreement-threshold-comment-predates-the-representation]` narrowed
  with the measurement §4.3 below.
- `docs/OPEN_ISSUES.md` — that issue's §1.8 row narrowed to the measured figures. **No issue opened
  or closed; the count stays at 72.**

`T_Z_REPRESENTATION_VERSION` is **6** before and **6** after (`LambdaCDM_GenericEOS.py:419`,
untouched).

## The two equality redshifts

**No bit moved. This prompt changes no production code at all**, so "before" and "after" are the
same tree; the table is the measurement the campaign is scored against from here on. All values at
17 significant figures, on `QCD_Cosmology(max_z=1e12)` and the `PureRadiationEOS` stand-in
(`max_z=1e6`), both at `Planck2018()` and `Mpc_units()`, taken at `3e820eb`.

| Model | Pair | `_find_rho_equality` (before = after) | Bracketed `brentq` reference | Separation |
|---|---|---|---|---|
| `QCD_Cosmology` | matter = radiation | `3406.6689742499498` | `3406.6689742499511` | **−3.0 ulp** |
| `QCD_Cosmology` | matter = $\Lambda$ | `0.30342303299640738` | `0.30342303299640749` | **−2.0 ulp** |
| pure-radiation stand-in | matter = radiation | `3403.1059638279453` | `3403.1059638279457` | **−1.0 ulp** |
| pure-radiation stand-in | matter = $\Lambda$ | `0.30342303299640738` | `0.30342303299640749` | **−2.0 ulp** |

As `float.hex()`, because the decimals differ in their last digit only:

| Quantity | `float.hex()` |
|---|---|
| QCD m=r, solve | `0x1.a9d5683cafacdp+11` |
| QCD m=r, reference | `0x1.a9d5683cafad0p+11` |
| QCD m=r, closed form | `0x1.a9d5683cafac9p+11` |
| m=$\Lambda$ solve and closed form (**both models**) | `0x1.36b4870e4a718p-2` |
| m=$\Lambda$ reference (**both models**) | `0x1.36b4870e4a71ap-2` |
| radiation m=r, solve and closed form | `0x1.a963640e40f2bp+11` |
| radiation m=r, reference | `0x1.a963640e40f2cp+11` |

The −3.0 ulp entry is `AUDIT.md` §2.2's **−4.00e-16** expressed in ulp: one ulp at
$z = 3406.67$ is $2^{-41}/3406.67 = 1.335\times10^{-16}$ relative, so $-4.00\times10^{-16}$ is
$-3.00$ ulp exactly. **The audit reproduces; it is the prompt's threshold that does not fit it.**

## Deviations from the prompt

### D1 — the four equality redshifts are asserted at 4 ulp, not 2 — **STRUCTURALLY REQUIRED**

**What the prompt assumed.** §3 item 3 and the §5 acceptance table both give "**Threshold: 2 ulp**,
expressed as `abs(got - ref) <= 2 * np.spacing(abs(ref))`".

**What is actually there.** The worst of the four is **3.0 ulp** (`QCD_Cosmology`, matter =
radiation). That is not a disagreement with the audit — it *is* the audit's own figure. §6's stop
condition is written for the case where "audit §2 or §4.1 does not hold on this tree"; §2 holds
exactly, and `RECONCILIATION.md` §2 is confirmed. The incompatibility is internal to the prompt:
−4.00e-16 relative at $z\sim3.4\times10^3$ cannot be ≤ 2 ulp, and the same slip is visible in
`RECONCILIATION.md` §6, whose −9.34e-16 for the closed form is 7 ulp.

**What was done instead.** `SOLVE_VS_REFERENCE_ULP = 4`, with the measured 3 / 2 / 1 / 2 and the
reason for the fourth ulp written into the constant's comment. The extra ulp is **not** slack for
the solve: it is the reference's own arbitrariness, measured. Re-taking the `QCD_Cosmology`
matter-radiation reference on brackets of ±1 %, ±5 %, ±10 %, $[0.5\times, 2\times]$ and
$[0.8\times, 1.5\times]$ of the guess gives `3406.668974249951` (±1 %, ±5 %),
`3406.6689742499516` (±10 %, $[0.5,2]$) and `3406.6689742499507` ($[0.8,1.5]$) — a spread of
**2 ulp**, i.e. ±1 about the value the shipped bracket returns. A 2-ulp budget would therefore be
measuring which bracket the reference happened to use as much as anything about the solve.

**Why this is not a loosening.** 4 ulp on a quantity whose relative ulp is $1.3\times10^{-16}$ is
$5\times10^{-16}$ — still eleven orders inside the shipped `rtol=1e-4`, and inside every displaced-
guess error in audit §3.1 except the +0 % row. Prompt 02's own acceptance is **bit-identity**
against the values tabulated above, which is strictly stronger than anything asserted here; this
threshold is the standing guard for later readers, not prompt 02's gate. **It must not be widened
again**: the campaign's claim is that these roots do not move at all, and the values are in this
log at 17 digits and in hex so that a later disagreement is a diff, not a judgement.

### D2 — the matter–radiation closed form is asserted at 8 ulp — **IMPLEMENTATION CHOICE**

§3 item 6 asks that "the closed form matches the reference there" without naming a threshold; the
§5 acceptance table names ≤ 2 ulp only for the **$\Lambda$** closed form (item 5), which is met
exactly. Measured, $\Omega_m/\Omega_r - 1$ against the bracketed reference is **7 ulp** on
`QCD_Cosmology` (`RECONCILIATION.md` §6's −9.34e-16, reproduced) and **1 ulp** on the stand-in.

Alternatives considered. (i) Assert only the $g_*$ flatness and drop the closed-form comparison —
rejected, because the flatness is the *premise* and the comparison is the *conclusion*, and a test
that keeps only the premise does not say that the guess is the root. (ii) Assert at 7 ulp exactly —
rejected for the same reason D1 gives one ulp of reference arbitrariness. (iii) **8 ulp, chosen**,
as 7 measured plus that ulp, in its own named constant so it can never be confused with the
solve's budget. The wider figure is not slop: $\Omega_m/\Omega_r$ is a quotient of two separately
rounded quantities, so evaluating an expression that is exact in exact arithmetic still costs a few
ulp. That the stand-in manages 1 ulp and `QCD_Cosmology` 7 is itself the shape of the finding.

### D3 — monotonicity is asserted on both models, not on `QCD_Cosmology` alone — **IMPLEMENTATION CHOICE**

`measure_rho_equality.py` §4.1 and audit §4.1 probe `QCD_Cosmology` only. The test probes both
models. Reason: prompt 02's bracket-expansion policy will run on every model the campaign builds,
and the stand-in is one of README §3.1's four anchors; the cost is 28 extra `_rho_fluid` calls on a
model that is already constructed. Both are strict at every probe (§4.4 below). The alternative —
match the script exactly — was rejected because item (d) is a claim about the *policy*, not about
`QCD_Cosmology`.

### D4 — `PureRadiationEOS` and `lambdaCDM_gstar` are imported from `test_wPerturbations` — **IMPLEMENTATION CHOICE**

README §5 rule 6 says to "use the stand-in model pattern of
`CosmologyModels/tests/test_wPerturbations.py`", which admits either importing or reproducing.
Imported, as `measure_rho_equality.py:22` does, so there is exactly one definition of the stand-in
and a later change to it cannot silently give the two files different cosmologies. The prompt's
"do not touch `test_wPerturbations.py`" is respected: it is read, never written. The countervailing
argument — that a test module importing another test module couples two suites — is real but
cheaper than two divergent stand-ins, and the campaign's §4.3 finding below is about *that file's*
comment, which makes the coupling explicit rather than hidden.

### D5 — the reference is seeded from the closed form, not from the shipped root — **IMPLEMENTATION CHOICE**

`measure_rho_equality.bracketed_reference` takes a `z_root` and §2.2 passes it `shipped.root`. The
test passes the closed form instead, so that no part of the reference's construction goes through
the thing being scored. Verified equivalent: seeding from either gives the same reference float for
all four cases at this tree. This is the point the prompt's §1 makes about the trap, taken one step
further than the script does.

## Verification performed

Everything below was **run**; nothing in this section is reasoning about what would happen.

### 1. The new module

```
PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_rho_equality -v
```
→ `Ran 4 tests in 0.083s / OK`. **0.083 s**, well inside the prompt's ~5 s; `setUpClass` runs once
(a `QCD_Cosmology(max_z=1e12)` construction is 0.051 s measured on its own).

Printed by `test_equality_redshifts_match_a_bracketed_reference`:

```
  equality redshifts against a bracketed brentq reference (17 digits):
             QCD_Cosmology  matter_radiation: solve = 3406.6689742499498     reference = 3406.668974249951      (-3.0 ulp)
             QCD_Cosmology     matter_lambda: solve = 0.3034230329964074     reference = 0.3034230329964075     (-2.0 ulp)
   pure-radiation stand-in  matter_radiation: solve = 3403.1059638279453     reference = 3403.1059638279457     (-1.0 ulp)
   pure-radiation stand-in     matter_lambda: solve = 0.3034230329964074     reference = 0.3034230329964075     (-2.0 ulp)
```

| Acceptance check | Threshold | Measured |
|---|---|---|
| Four equality redshifts vs. the reference | ≤ 4 ulp (D1; prompt said 2) | **3, 2, 1, 2** |
| Both monotonicity statements | strict at every probe | **strict, 6 intervals × 2 pairs × 2 models** |
| $\Lambda$ closed form, both models | ≤ 2 ulp | **2, 2** |
| matter–radiation closed form, both models | ≤ 8 ulp (D2) | **7** (QCD), **1** (stand-in) |
| Module runtime | quote; explain if > ~5 s | **0.083 s** |
| Production files in the diff | zero | **zero** |

### 2. Suites

| Suite | Before (`3e820eb`) | After |
|---|---|---|
| `CosmologyModels/tests` | **30**, OK, 0.610 s | **34**, OK, 0.691 s (*n* = **4**) |
| `ComputeTargets/tests` | **447**, OK, 177.1 s | **447**, OK, 181.4 s |

Both re-run at `3e820eb` before the change rather than quoted from `RECONCILIATION.md` §10.

### 3. Formatting

`./venv/bin/python -m black CosmologyModels/tests/test_rho_equality.py` reformatted once; the tree
is clean under `--check`.

## §4 — measured, recorded, not asserted

Reproduction for all three blocks: the models are
`QCD_Cosmology(store_id=10, units=Mpc_units(), params=Planck2018(), max_z=1e12)` and
`LambdaCDM_GenericEOS(store_id=11, eos=PureRadiationEOS(units, lambdaCDM_gstar(params.Neff)),
units, params, max_z=1e6)`, exactly as `measure_rho_equality.py:187-198` builds them. §4.1, §4.2
and §4.4 are also produced by
`PYTHONPATH=. ./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py`
(9 s, no Ray, no datastore), which reproduces at `3e820eb` to the digit printed.

### 4.1 The failure boundary, at `3e820eb`, `QCD_Cosmology`, matter = radiation

Both columns: the bare `root_scalar(f, x0=guess, xtol=1e-6, rtol=1e-4)` and
`_find_rho_equality("matter", "radiation", init_z=guess)` itself. **They are identical at every
offset** — which is the finding: the guard adds nothing on any path.

| Offset | $z_0$ | `root_scalar` | `_find_rho_equality` |
|---|---|---|---|
| −5 % | 3236.335526 | `converged=True, root=3406.6689742503595, iterations=4` | returned `3406.6689742503595` |
| −10 % | 3066.002077 | `converged=True, root=3406.668974249972, iterations=5` | returned `3406.668974249972` |
| −20 % | 2725.335179 | `converged=True, root=3406.6689831134686, iterations=7` | returned `3406.6689831134686` |
| −30 % | 2384.668282 | `ValueError: math domain error` | `ValueError: math domain error` |
| −50 % | 1703.334487 | `RuntimeError` (text below) | `RuntimeError` (same text) |
| −80 % | 681.333795 | `RuntimeError` (text below) | `RuntimeError` (same text) |

Exact text of the two `RuntimeError`s, verbatim:

```
TemperatureRepresentation: evaluated T(z) out of bounds @ z=-0.25719 (min allowed z=-0.24, recommended limit is z >= -0.2376)
TemperatureRepresentation: evaluated T(z) out of bounds @ z=-0.38344 (min allowed z=-0.24, recommended limit is z >= -0.2376)
```

Confirmations asked for by the prompt:

- `RECONCILIATION.md` §4 is right: the prefix is `TemperatureRepresentation:`, not `GkSource.`
  (`f023eb8` closed that).
- **`_find_rho_equality`'s guard at `:1015` never fires.** `root.converged` is `True` on every
  offset that returns at all, and the three that fail raise from inside `_rho_fluid`, two frames
  below the guard. The message names neither `_find_rho_equality`, nor the species pair, nor the
  range searched.
- The −20 % row returns `converged=True` with a root **8.9e-6 in $z$** away from the reference
  (2.7e-9 relative) — a silently wrong answer inside the requested `rtol=1e-4`, audit §3.2's
  third row.

### 4.2 `_rho_fluid` evaluation counts at the production call sites

| Model | Pair | evals | `iterations` | `function_calls` | `converged` |
|---|---|---|---|---|---|
| `QCD_Cosmology` | matter = radiation | **3** | 1 | 2 | True |
| `QCD_Cosmology` | matter = $\Lambda$ | **1** | 0 | 1 | True |
| pure-radiation stand-in | matter = radiation | **1** | 0 | 1 | True |
| pure-radiation stand-in | matter = $\Lambda$ | **1** | 0 | 1 | True |

(`evals` counts calls to the residual, `function_calls` is `scipy`'s own count; they differ on the
first row because secant evaluates $x_0$ and its internally perturbed $x_1$ before iterating.
Audit §2.2's "3, 1, 1, 1" is the `evals` column and it reproduces.)

### 4.3 `RECONCILIATION.md` §9.3 — `test_wPerturbations.py:34-41`'s two figures, re-measured

The comment says the $T(z)$ inversion is "tabulated on a 500-point spline in log(1+z)" and that the
agreement with `LambdaCDM`'s closed forms is "~1.3e-9 relative with max_z = 1e4 and ~4e-7 with the
default max_z = 1e20", which is what `AGREEMENT_RTOL = 1.0e-8` was set against. Re-measured at
`3e820eb` on exactly what `test_agrees_with_LambdaCDM` compares — `generic.wPerturbations(z)`
against `lcdm.wPerturbations(z)` over its own probe set $z \in \{0, 0.5, 1, 2, 10, 10^3\}$, with
the `PureRadiationEOS` stand-in built at each `max_z`:

| `max_z` | Comment claims | **Measured, worst over the probe set** | At |
|---|---|---|---|
| `1e4` (`AGREEMENT_MAX_Z`) | ~1.3e-9 | **8.8818e-16** | $z = 1$ |
| `1e20` (the default) | ~4e-7 | **6.6613e-16** | $z = 0.5$ |

Per-probe at `max_z=1e4`: 2.2204e-16, 3.3307e-16, 8.8818e-16, 5.5511e-16, 4.4409e-16, **0.0** (at
$z = 10^3$ the two agree bit-for-bit). At `max_z=1e20`: 2.2204e-16, 6.6613e-16, 6.6613e-16,
4.4409e-16, 5.5511e-16, 6.6613e-16.

**Seven to nine orders tighter than the comment, and the `max_z` dependence has gone entirely.**
The reason is the replacement `qcd-background-audit` prompts 05 and 06 made: what is tabulated is
no longer $T$ but a segmented entropy factor, and on an equation of state with constant
$g_* = g_{S,*}$ that factor is exactly constant, so $T(z) = T_{\rm CMB}(1+z)$ is recovered to
rounding at any `max_z`. The old 500-point spline in $T$ interpolated a quantity that varies over
twenty decades and paid for it; the comment is a description of a representation that no longer
exists. `AGREEMENT_RTOL = 1.0e-8` is now a ceiling eight orders above what the code delivers.
**Not edited** (the prompt forbids it); the §3 issue
`[01-agreement-threshold-comment-predates-the-representation]`, opened by the planning commit, is
narrowed with these figures and remains assigned to prompt 08 behind README §7 D3.

### 4.4 Monotonicity, for the record

`QCD_Cosmology`, $\rho_m/\rho_r$ over $z \in \{33, 339, 1702, 3406, 6814, 34075, 340766\}$:
1.002256e+02, 1.002256e+01, 2.000980e+00, 1.000196e+00, 5.000248e-01, 1.000020e-01, 9.999997e-03 —
**strictly decreasing**. $\rho_m/\rho_\Lambda$ over $z \in \{0, 0.1, 0.303, 0.5, 1, 3, 10\}$:
4.515895e-01, 6.010656e-01, 9.990266e-01, 1.524115e+00, 3.612716e+00, 2.890173e+01, 6.010656e+02 —
**strictly increasing**. The stand-in is strict on both as well (asserted, not tabulated here).

## Observations not acted on

1. **`AUDIT.md` §2.2 and `RECONCILIATION.md` §6 quote their agreement figures as relative errors,
   and every threshold in this campaign is in ulp.** The conversion is where the prompt's 2-ulp
   figure came from being wrong (D1). Not actionable as an issue — the documents are correct, they
   are simply in different units — but a later prompt that reads "−4.00e-16" and writes "2 ulp"
   will repeat it. One ulp is 1.335e-16 relative at $z = 3406.67$ and 3.66e-16 at $z = 0.3034$;
   both conversions are now in this log.

2. **`bracketed_reference`'s bracket is a hard-coded ±5 % / [0.5×, 1.5×] and is not itself
   defended.** It works because both roots are where the closed form says they are; on a model
   where the guess were poor it could fail to straddle, and `brentq` would raise rather than return
   a wrong answer, so the failure is loud. Prompt 02 is about to write a bracket-expansion policy
   for the production solve and will have to decide this question properly for that case; the test
   helper deliberately does **not** anticipate the answer, because a reference that shared prompt
   02's policy would stop being independent of it. No issue opened.

3. **The −20 % row of §4.1 is the interesting one, not the exceptions.** `converged=True` with
   2.7e-9 relative error is the only measured case in this campaign where the shipped solve returns
   a *wrong* number rather than raising, and it is well inside the `rtol=1e-4` it asks for. Already
   covered by `[00-equality-solve-is-unbracketed-and-loose]`; noted here because a reader skimming
   §4.1 for exception names will skip it.

## State handed to the next prompt

**Prompt 02's bit-identity target.** These are the four values `_find_rho_equality` returns at
`3e820eb`, at 17 significant figures and in hex. Prompt 02's acceptance is bit-identity against
this table.

| Model | Pair | `_find_rho_equality` | `float.hex()` |
|---|---|---|---|
| `QCD_Cosmology(max_z=1e12)` | matter = radiation | `3406.6689742499498` | `0x1.a9d5683cafacdp+11` |
| `QCD_Cosmology(max_z=1e12)` | matter = $\Lambda$ | `0.30342303299640738` | `0x1.36b4870e4a718p-2` |
| `PureRadiationEOS(max_z=1e6)` | matter = radiation | `3403.1059638279453` | `0x1.a963640e40f2bp+11` |
| `PureRadiationEOS(max_z=1e6)` | matter = $\Lambda$ | `0.30342303299640738` | `0x1.36b4870e4a718p-2` |

**The independent reference, and its bracket.** `brentq(match_rho, lo, hi, xtol=1e-300,
rtol=8.9e-16)` with `lo, hi = 0.95 z_g, 1.05 z_g` for $z_g > 1$ and
`lo, hi = max(0.5 z_g, -0.9), 1.5 z_g` otherwise, $z_g$ the **closed form**, never the shipped
root:

| Model | Pair | reference | `float.hex()` |
|---|---|---|---|
| `QCD_Cosmology` | matter = radiation | `3406.6689742499511` | `0x1.a9d5683cafad0p+11` |
| `QCD_Cosmology` | matter = $\Lambda$ | `0.30342303299640749` | `0x1.36b4870e4a71ap-2` |
| `PureRadiationEOS` | matter = radiation | `3403.1059638279457` | `0x1.a963640e40f2cp+11` |
| `PureRadiationEOS` | matter = $\Lambda$ | `0.30342303299640749` | `0x1.36b4870e4a71ap-2` |

**The reference is itself only good to ±1 ulp.** Brackets of ±1 %, ±5 %, ±10 %, $[0.5\times,
2\times]$ and $[0.8\times, 1.5\times]$ give `3406.668974249951`, `3406.668974249951`,
`3406.6689742499516`, `3406.6689742499516`, `3406.6689742499507` for the `QCD_Cosmology`
matter-radiation root. **Prompt 02 must not treat the reference as exact**; the solve's bit-identity
against the table above is the sharper statement and is the one to use.

**The failure boundary, to be shown failing on `HEAD~1`** (this commit) — `QCD_Cosmology`,
matter = radiation, guess displaced by a fraction of $z_0 = \Omega_m/\Omega_r - 1$. Exception text
verbatim, so an `assertRaisesRegex` written against prompt 02's new error can be shown *not* to
match these:

| Offset | $z_0$ | What `_find_rho_equality` does today |
|---|---|---|
| −5 % | 3236.335526 | returns `3406.6689742503595` |
| −10 % | 3066.002077 | returns `3406.668974249972` |
| −20 % | 2725.335179 | returns `3406.6689831134686` — **wrong by 2.7e-9 relative, `converged=True`** |
| −30 % | 2384.668282 | raises `ValueError: math domain error` |
| −50 % | 1703.334487 | raises `RuntimeError: TemperatureRepresentation: evaluated T(z) out of bounds @ z=-0.25719 (min allowed z=-0.24, recommended limit is z >= -0.2376)` |
| −80 % | 681.333795 | raises `RuntimeError: TemperatureRepresentation: evaluated T(z) out of bounds @ z=-0.38344 (min allowed z=-0.24, recommended limit is z >= -0.2376)` |

**The guard at `:1015` never fires at any offset.** Every returning call has `root.converged =
True`; every failing call raises from inside `_rho_fluid`. A new assertion that
`_find_rho_equality` raises *its own* `RuntimeError`, naming the species pair and the range
searched, fails on this tree in three distinguishable ways — `ValueError` at −30 %, and a
`RuntimeError` whose text names `TemperatureRepresentation` rather than the species pair at −50 %
and −80 %. **An `assertRaises(RuntimeError)` alone would pass on `HEAD~1` at −50 %**; the assertion
has to be on the message.

**Evaluation counts today**, to be quoted against prompt 02's (audit §3.1 predicts +6 to +9): **3,
1, 1, 1** residual evaluations for QCD m=r, QCD m=$\Lambda$, stand-in m=r, stand-in m=$\Lambda$.

**Names prompt 02 can use.** `CosmologyModels/tests/test_rho_equality.py` exports
`species_of`, `closed_form_guess`, `match_rho`, `bracketed_reference`, `ulps_between`,
`SOLVE_VS_REFERENCE_ULP`, `LAMBDA_CLOSED_FORM_ULP`, `MATTER_RADIATION_CLOSED_FORM_ULP`,
`MATTER_RADIATION_PROBES`, `MATTER_LAMBDA_PROBES`, `QCD_MAX_Z`, `RADIATION_MAX_Z`,
`BRENT_RTOL_FLOOR`, `BRENT_XTOL_DISABLED`. **Prompt 02 should add its failure-mode assertions to
this module rather than rewriting any of the four tests**: the four that exist must pass unchanged
on both trees, which is the evidence that the roots did not move.

**Suite counts to carry forward:** `CosmologyModels` **34**, `ComputeTargets` **447**.
`T_Z_REPRESENTATION_VERSION` = **6**.

**Reproduction commands.**

```bash
PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_rho_equality -v
PYTHONPATH=. ./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
```
