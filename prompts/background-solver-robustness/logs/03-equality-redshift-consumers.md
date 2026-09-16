# Log 03 — what the equality redshifts actually feed

**Prompt:** prompts/background-solver-robustness/03-equality-redshift-consumers.md
**Commit:** *(this commit)* — Write down what the equality redshifts actually feed
**Model:** Claude Opus 5
**Date:** 2026-09-16
**Result:** COMPLETE WITH DEVIATIONS

---

## What shipped

**No production behaviour changed.** Two files carry a diff, one of them a docstring.

| File | Before → after |
|---|---|
| `main.py:525-528` → `:525-529` | The one sentence of `cosmology_feature_redshifts`'s docstring that claimed the closed form *"agrees with `LambdaCDM_GenericEOS`'s own root solve to 4e-13 relative in z on `QCD_Cosmology` at production parameters"* is replaced by the measured figures — **−9.3e-16 relative (7 ulp)** at matter–radiation and **the same float** at matter–$\Lambda$ — taken at `921f41c`, with the statement that 4e-13 was a safe over-estimate, unverified rather than wrong. **Zero executable lines of `main.py` are in the diff** (`git diff main.py` is one hunk, entirely inside the triple-quoted docstring). |
| `CosmologyModels/tests/test_rho_equality.py` | Three new module-level helpers `main_py_closed_form(cosmology, pair) -> float`, `generic_eos_closed_form(model, pair) -> float`, `lambdaCDM_closed_form(model, pair) -> float`, and the tuple `CLOSED_FORM_SITES = (("main.py:553/:555", …), ("LambdaCDM_GenericEOS.py:502/:507", …), ("LambdaCDM.py:73/:74", …))`; a `LambdaCDM(Planck2018)` instance on the test class as `cls.lambdaCDM`; and one new test, `test_the_three_closed_form_sites_agree`. Imports gain `builtins`, `math` and `LambdaCDM`. |

**Line numbers move.** The corrected sentence is four lines longer than the one it replaces, so
everything below it in `main.py` shifts by **+4**. Re-anchored at this commit, for every document
that cites them: the two closed-form expressions are **`:553`** and **`:555`** (were `:549`, `:551`);
the early return is `:545-546` (was `:541-542`); the `getattr` block is `:549-551` (was `:545-547`);
`break_z, feature_z = cosmology_feature_redshifts(...)` is **`:907`** (was `:903`) and
`feature_z=feature_z` is **`:932`** (was `:928`). The corrected docstring sentence itself is
`:525-531`. Nothing else in `main.py` changed.

`T_Z_REPRESENTATION_VERSION` = **6** before and **6** after. Nothing in either file reads or
writes it; it is quoted because README §5 and board note 2 require it at every commit.

### The three helpers, and why they are written the way they are

Each transcribes its site *including which `pow` is in scope in that file*, which is the only way
the three expressions could ever produce different floats:

| Site | `pow` in scope | Why |
|---|---|---|
| `main.py:553`, `:555` | **builtin** | `main.py` does `from math import sqrt` only |
| `LambdaCDM_GenericEOS.py:502`, `:507` | **builtin** | that module imports `exp, sqrt, log, log1p, expm1` from `math` — not `pow` |
| `LambdaCDM.py:73`, `:74` | **`math.pow`** | `from math import sqrt, pow` |

`main.py` is not imported — `CLAUDE.md` says it cannot be — so its expression is mirrored, with
the comment block above `main_py_closed_form` naming `main.py:553` and `:555` as the site it
mirrors and saying that an edit to either announces itself here.

---

## The two equality redshifts

**No bit moved.** The four values are log 02's table, unchanged, re-taken on this tree
(`PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_rho_equality -v`):

| Model | Pair | `_find_rho_equality`, before **and** after | `float.hex()` |
|---|---|---|---|
| `QCD_Cosmology(max_z=1e12)` | matter = radiation | `3406.6689742499511` | `0x1.a9d5683cafad0p+11` |
| `QCD_Cosmology(max_z=1e12)` | matter = $\Lambda$ | `0.30342303299640738` | `0x1.36b4870e4a718p-2` |
| `PureRadiationEOS(max_z=1e6)` | matter = radiation | `3403.1059638279457` | `0x1.a963640e40f2cp+11` |
| `PureRadiationEOS(max_z=1e6)` | matter = $\Lambda$ | `0.30342303299640738` | `0x1.36b4870e4a718p-2` |

Relative move against log 02: **0.0 on all four, bit for bit.** The two printed banner lines are
character-identical. `QCD_Cosmology` at the *production* `max_z = 1e20` gives the same
matter–radiation root, `3406.668974249951`, and the same matter–$\Lambda$ root.

---

## §2.1 — the three copies, scored against each other

`PYTHONPATH=. ./venv/bin/python <scratch>/measure_sites.py`, at this commit. Every column is the
expression **as written in its own file**, not a normalised form. `main.py` closed form,
`LambdaCDM_GenericEOS` guess and `LambdaCDM.py:73-74` are three separate evaluations; that they
collapse to one float is the measurement, not an assumption.

### `QCD_Cosmology(max_z=1e12)` — `omega_m = 0.31110000000000004`, `omega_r = 9.12940788412335e-05`, `omega_cc = 0.6889`

| Column | matter = radiation | hex | matter = $\Lambda$ | hex |
|---|---|---|---|---|
| `main.py:553`/`:555` | `3406.668974249948` | `0x1.a9d5683cafac9p+11` | `0.3034230329964074` | `0x1.36b4870e4a718p-2` |
| `LambdaCDM_GenericEOS:502`/`:507` | `3406.668974249948` | `0x1.a9d5683cafac9p+11` | `0.3034230329964074` | `0x1.36b4870e4a718p-2` |
| `LambdaCDM.py:73`/`:74` | `3406.668974249948` | `0x1.a9d5683cafac9p+11` | `0.3034230329964074` | `0x1.36b4870e4a718p-2` |
| the corrected solve | `3406.668974249951` | `0x1.a9d5683cafad0p+11` | `0.3034230329964074` | `0x1.36b4870e4a718p-2` |
| independent `brentq` | `3406.668974249951` | `0x1.a9d5683cafad0p+11` | `0.3034230329964075` | `0x1.36b4870e4a71ap-2` |
| **ulp spread, all columns** | **7.000** (2 distinct floats of 5) | | **2.000** (2 of 5) | |
| **ulp spread, three sites only** | **0.000 — identical** | | **0.000 — identical** | |

### pure-radiation stand-in, `max_z = 1e6` — `omega_m = 0.31110000000000004`, `omega_r = 9.138963454890973e-05`, `omega_cc = 0.6889`

| Column | matter = radiation | hex | matter = $\Lambda$ | hex |
|---|---|---|---|---|
| `main.py:553`/`:555` | `3403.1059638279453` | `0x1.a963640e40f2bp+11` | `0.3034230329964074` | `0x1.36b4870e4a718p-2` |
| `LambdaCDM_GenericEOS:502`/`:507` | `3403.1059638279453` | `0x1.a963640e40f2bp+11` | `0.3034230329964074` | `0x1.36b4870e4a718p-2` |
| `LambdaCDM.py:73`/`:74` | `3403.1059638279453` | `0x1.a963640e40f2bp+11` | `0.3034230329964074` | `0x1.36b4870e4a718p-2` |
| the corrected solve | `3403.1059638279457` | `0x1.a963640e40f2cp+11` | `0.3034230329964074` | `0x1.36b4870e4a718p-2` |
| independent `brentq` | `3403.1059638279457` | `0x1.a963640e40f2cp+11` | `0.3034230329964075` | `0x1.36b4870e4a71ap-2` |
| **ulp spread, all columns** | **1.000** (2 of 5) | | **2.000** (2 of 5) | |
| **ulp spread, three sites only** | **0.000 — identical** | | **0.000 — identical** | |

### `LambdaCDM(Planck2018)` — `omega_m = 0.31110000000000004`, `omega_r = 9.138963454890973e-05`, `omega_cc = 0.6889`

| Column | matter = radiation | matter = $\Lambda$ |
|---|---|---|
| `main.py:553`/`:555` | `3403.1059638279453` | `0.3034230329964074` |
| `LambdaCDM_GenericEOS:502`/`:507` | `3403.1059638279453` | `0.3034230329964074` |
| `LambdaCDM.py:73`/`:74` | `3403.1059638279453` | `0.3034230329964074` |
| the corrected solve | **n/a** — `LambdaCDM` has no equation of state, no `_rho_fluid` and no `_find_rho_equality` | **n/a** |
| independent `brentq` | **n/a** — there is no residual to bracket | **n/a** |
| **ulp spread, three sites only** | **0.000 — identical** | **0.000 — identical** |

### The third model, and why it is not `RadiationModel`

The prompt asks for `QCD_Cosmology`, `LambdaCDM(Planck2018)` and `RadiationModel`, "or the nearest
thing each has". **`RadiationModel` cannot appear in this table at all.** It is
`ComputeTargets/tests/wkb_reference.py:176`, a WKB *reference* stand-in whose whole surface is
`H0`, `Hubble`, `tau`, `tau_delta` and a `ModelFunctions` block; it exposes **no `omega_m`,
`omega_r` or `omega_cc`**, so none of the three expressions can be evaluated on it, and it is not
a production cosmology. `cosmology_feature_redshifts` is safe on it for two independent reasons:
it declares no `integration_break_points`, so the function returns `([], [])` at `main.py:545-546`
before reaching any $\Omega$; and the three `getattr(..., None)` guards at `:549-551` would return
`None` even if it did not. The campaign's own third anchor (README §3.1) is the **pure-radiation
stand-in**, a `LambdaCDM_GenericEOS` over `PureRadiationEOS` where $g_* = g_{s,*}$ is constant and
both closed forms are exact, and it is used here instead.

### What this table says

1. **The three sites agree bit for bit — on every model, on both pairs.** `math.pow` and the
   builtin `pow` return the same double for these arguments, and the `float(...)` wrapper
   `main.py` applies is a no-op on a float. *Nothing in the language guarantees that*; both route
   to the platform `pow()`, and it is measured here rather than assumed.
2. **The spread in the table is entirely between the closed form and the solve**, and only for
   matter = radiation: 7 ulp on `QCD_Cosmology`, 1 ulp on the stand-in. The matter–$\Lambda$
   closed form and the solve are **the same float** on both models; only the `brentq` reference
   differs from them, by 2 ulp, which is
   `[02-bracketed-reference-is-not-the-exact-root]` — the reference is the **less accurate** of
   the two on that pair, and the sign of the finding is that the shipped answers are better than
   what scores them.
3. **The 7 ulp on `QCD_Cosmology` is the interesting number**, because it is the only place where
   "unify on the solve" and "unify on the closed form" give different grids. It is 7 ulp and not 1
   because `omega_m/omega_r` is a ratio of two separately-rounded quantities: the *expression* is
   exact in exact arithmetic wherever $g_*$ is flat, but its evaluation is not. The stand-in's
   `omega_r` is the LambdaCDM one (`9.138963454890973e-05`) while `QCD_Cosmology`'s is
   `9.12940788412335e-05`, which is why the two matter–radiation roots differ in the third
   significant figure and why one model does not predict the other.

---

## §2.2 — `main.py:525-527`'s figure (`:525-529` after this commit), re-taken against the corrected solve

Against **prompt 02's** solve (`921f41c`), not the secant `RECONCILIATION.md` §6 measured:

| Model | Pair | closed form | corrected solve | relative | ulp |
|---|---|---|---|---|---|
| `QCD_Cosmology` | matter = radiation | `3406.668974249948` | `3406.668974249951` | **−9.344e-16** | −7.0 |
| `QCD_Cosmology` | matter = $\Lambda$ | `0.3034230329964074` | `0.3034230329964074` | **+0.000e+00** | 0.0 |
| pure-radiation stand-in | matter = radiation | `3403.1059638279453` | `3403.1059638279457` | **−1.336e-16** | −1.0 |
| pure-radiation stand-in | matter = $\Lambda$ | `0.3034230329964074` | `0.3034230329964074` | **+0.000e+00** | 0.0 |

The docstring now says **−9.3e-16 (7 ulp)** and **the same float**, at `921f41c`. `RECONCILIATION.md`
§6's figures were −9.34e-16 and −3.66e-16 *against the `brentq` reference*; the matter–radiation
figure is unchanged to three digits because prompt 02 moved the solve **onto** that reference, and
the matter–$\Lambda$ figure has gone to exactly zero for the same reason — the solve and the closed
form are now the same float, and it is the reference that sits 2 ulp away.

---

## §2.3 — the production source grid is byte-identical

`PYTHONPATH=. ./venv/bin/python <scratch>/measure_grid.py`, which builds the grid through
`main.py`'s **own** `cosmology_feature_redshifts` and `source_grid_spacing_profile`, extracted with
`ComputeTargets.tests.test_main_plumbing.load_main_py_functions` via
`ComputeTargets/tests/test_source_grid.py`'s `_production_grid` — production geometry,
`z_init = 2.0636395964161516e16`, `z_end = 0.1`, 100 samples per decade, 50 wavenumbers.
`QCD_Cosmology` is built at the production `max_z = 1e20`.

| Tree | `QCD_Cosmology` samples / digest | `LambdaCDM(Planck2018)` samples / digest |
|---|---|---|
| `7fdc49b` (prompt 01 — the **unbracketed secant**) | 1,996 / **`a2c32f67`** | 1,778 / **`60a3205a`** |
| `921f41c` (`HEAD~1`, prompt 02) | 1,996 / **`a2c32f67`** | 1,778 / **`60a3205a`** |
| this commit | 1,996 / **`a2c32f67`** | 1,778 / **`60a3205a`** |

**Identical at all three.** The pre-prompt-02 row is measured as well as the two the acceptance
asks for, because `HEAD~1` of *this* commit is prompt 02's own commit and comparing against it
alone would not test what the prompt's stop condition is about ("a difference means prompt 02
changed something it should not have"). The worktree was `git worktree add --detach` with the main
checkout's `venv` symlinked in.

`QCD_Cosmology`'s `feature_z` is `['3406.668974249948', '0.3034230329964074']` — the closed forms,
to the bit — and both appear in `grid.features` and in the 8 protected samples.
`LambdaCDM(Planck2018)`'s `break_z` and `feature_z` are both `[]`: the whole cosmology-aware path
is gated on the model declaring break points (`main.py:545-546`), which is prompt 11's deliberate
choice and is documented in the paragraph below the sentence corrected here.

---

## §3 — what a decision-maker needs

### 1. The map

| # | Site | Expression | What it feeds | Load-bearing? |
|---|---|---|---|---|
| 1 | `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:502`, `:507` | `self.omega_m / self.omega_r - 1.0`; `pow(self.omega_cc / self.omega_m, 1.0/3.0) - 1.0` | the `init_z` of the two `_find_rho_equality` calls; the results are printed at `:516-517` with `:.4g` and discarded | **no** |
| 2 | `CosmologyModels/LambdaCDM/LambdaCDM.py:73`, `:74` | the same two, with `math.pow` | two `print`s at `:81-82` | **no** |
| 3 | **`main.py:553`, `:555`** | `float(omega_m / omega_r - 1.0)`; `float(pow(omega_cc / omega_m, 1.0/3.0) - 1.0)` | `feature_z` | **yes** |

The chain from site 3, hop by hop:

```
main.py:553, :555          feature_z.append(...)                      the two closed forms
main.py:907                break_z, feature_z = cosmology_feature_redshifts(...)
main.py:932                feature_z=feature_z   ->  populate_source_grid
CosmologyConcepts/wavenumber.py:349    features = sorted(z for z in feature_z if in range)
CosmologyConcepts/wavenumber.py:408-411  each admitted feature appended to `others`
CosmologyConcepts/wavenumber.py:413      protected = sorted(pairs + others)  -> forced into the grid
CosmologyConcepts/redshift.py:290        redshift_array.digest()
CosmologyConcepts/redshift.py:36-39      blake2b over struct.pack("<d", z) of EVERY sample
Datastore/SQL/ObjectFactories/BackgroundModel.py:225   source_grid_digest = z_sample.digest()
Datastore/SQL/ObjectFactories/BackgroundModel.py:182   the indexed column
Datastore/SQL/ObjectFactories/BackgroundModel.py:273-276  q.filter(source_grid_digest == ...)
```

The digest is *"computed over the **exact bits** of its values"* (`redshift.py:19`), by design and
for a documented reason: *"it distinguishes two grids that differ in **any** sample, including in
the protected set alone"* (`redshift.py:31`). So one ulp at site 3 is one different `<d` packing is
a different digest is a `BackgroundModel` miss.

Today the path is reached by `QCD_Cosmology` alone, because it is gated on the cosmology declaring
break points (`main.py:545-546`) and `QCD_Cosmology` is the only production model that does.
**`AUDIT.md` §2.1's grep found none of this** because site 3 is a *duplicated closed form*, not a
call to `_find_rho_equality`; the audit's *"the blast radius of this solve is two banner lines"* is
exactly true of the solve and exactly false of the quantity.

### 2. The measured agreement

§2.1's tables. In one line: **the three sites are the same float on every model and both pairs;**
the closed form and the corrected solve differ by **7 ulp** at matter–radiation on
`QCD_Cosmology`, by **1 ulp** on the stand-in, and **not at all** at matter–$\Lambda$ on either.

### 3. The three options, priced

**(i) Leave all three, document the duplication as deliberate, keep the new test as the guard.**
- **Cost: zero.** Nothing moves; the grid digests above are already the record.
- **What it buys:** `main.py:522-528` already explains *why* the duplication exists (prompt 11 of
  `qcd-background-audit` could not modify the model), and `test_the_three_closed_form_sites_agree`
  now makes the agreement a standing assertion rather than a paragraph — a one-ulp edit at any of
  the three sites fails it on all three models (demonstrated below).
- **Risk:** a fourth copy appears one day. The test does not see a site it does not know about.

**(ii) Unify on the closed form** — one helper, imported by all three sites.
- **Safe only if the resulting float is bit-identical to what `main.py:553`/`:555` produce today.**
  §2.1 measures that it currently would be: all three spellings, including the `math.pow` /
  builtin-`pow` split, give the same double on all three models.
- **Cost:** a refactor across three packages, and **`main.py` acquires an import from a cosmology
  model** — which is the coupling prompt 11 deliberately avoided, and the helper would have to
  live somewhere that `main.py`, `LambdaCDM.py` and `LambdaCDM_GenericEOS.py` can all reach.
- **Risk:** bit-identity here is a property of *this* CPython on *this* platform, measured once. A
  unified helper that anyone later rewrites in a "cleaner" but arithmetically different form —
  `(omega_m - omega_r)/omega_r`, `expm1(log(...)/3)`, `**` instead of `pow` — moves the grid. The
  duplication at least makes `main.py`'s copy visibly load-bearing where it sits.

**(iii) Unify on the solve** — `main.py` imports the result of `_find_rho_equality`.
- **Cost: measured, and it is not zero.** Substituting the solve's answers for the closed forms in
  `feature_z` takes the `QCD_Cosmology` production grid from digest **`a2c32f67`** to
  **`4849552b`**, sample count unchanged at 1,996 — a 7-ulp move in *one* sample out of 1,996 is a
  different `BackgroundModel` identity.
- **Consequence:** every stored object of the **eight** types `qcd-background-audit` log 11 §5
  prices is unfindable and must be recomputed —
  `TkNumericIntegration`, `TkWKBIntegration`, `QuadSource`, `GkNumericIntegration`,
  `GkWKBIntegration`, `GkSource`, `GkSourcePolicyData`, `QuadSourceIntegral` — for the QCD half.
  That log's measured production-shaped run was 15,020 objects in 6 m 35 s **with the
  `QuadSourceIntegral` stage stopped incomplete after ~3 h**, at one tenth of production in each
  wavenumber sample.
- It also makes the grid depend on the model's tabulated `max_z` and on the solve's tolerances,
  which is exactly the coupling `main.py:522-528` says prompt 11 avoided on purpose.

### 4. The recommendation: **(i)**

Leave all three, keep the duplication documented, and let
`test_the_three_closed_form_sites_agree` be the standing guard.

Why: the duplication is already deliberate and already explained at the load-bearing site; the
three copies agree to the bit today, measured, so there is no defect to repair; the *only* benefit
of unifying is aesthetic; and the downside of getting it wrong is a datastore whose
`QuadSourceIntegral` stage has never been measured to completion. Option (ii) is safe *today* but
converts a visible duplication into an invisible one-ulp dependency in a shared helper, and option
(iii) has a measured price of `a2c32f67 → 4849552b`.

**This is a report, not a decision.** README §7 **D2** is the user's, and
`[00-equality-redshift-closed-form-is-duplicated-three-times]` stays open on the board.

---

## Deviations from the prompt

### D1 — `IMPLEMENTATION CHOICE` — the third model is the pure-radiation stand-in, not `RadiationModel`

The prompt's §2.1 names `QCD_Cosmology`, `LambdaCDM(Planck2018)` and `RadiationModel`, "or the
nearest thing each has; say which and why if one of the three does not expose all three $\Omega$s".
`RadiationModel` (`ComputeTargets/tests/wkb_reference.py:176`) exposes **none** of them: it is an
exact-background WKB reference stand-in with `H0`, `Hubble`, `tau`, `tau_delta` and a
`ModelFunctions` block, and it is not a cosmology at all. The alternatives were to report it as an
empty row, or to substitute the campaign's own third anchor. The pure-radiation stand-in was
chosen because README §3.1 already names it as an anchor, because $g_*$ is constant on it so
*both* closed forms are exact and any departure is the representation's, and because it is a real
`LambdaCDM_GenericEOS` and therefore has the solve and the `brentq` columns that `RadiationModel`
could not fill. Both are reported: §2.1's "The third model" subsection states what `RadiationModel`
is and the two independent reasons `cosmology_feature_redshifts` is safe on it.

### D2 — `STRUCTURALLY REQUIRED` — the new test's budget is the module's measured constants, not the prompt's "2 ulp"

Prompt §2.4 says the test should assert that both expressions "are within 2 ulp of the bracketed
reference". **That is arithmetically unreachable for the matter–radiation pair.** The closed form
sits **7 ulp** from the reference on `QCD_Cosmology` — which is the −9.34e-16 `RECONCILIATION.md`
§6 itself reports, since one ulp at $z = 3406.67$ is $1.335\times10^{-16}$ relative — and prompt 01
already shipped `MATTER_RADIATION_CLOSED_FORM_ULP = 8` for exactly that reason. Board standing note
7 anticipates this deviation in terms: *"A later prompt converting a relative figure in `AUDIT.md`
or `RECONCILIATION.md` into ulp must do the arithmetic rather than copy the '2'."*

The test therefore uses the module's own two constants — `LAMBDA_CLOSED_FORM_ULP = 2` for the
$\Lambda$ pair (where the prompt's 2 is exactly right, and is kept) and
`MATTER_RADIATION_CLOSED_FORM_ULP = 8` for the other. **Neither constant was widened**, and no new
one was introduced: there is one budget per pair in this module and this test reuses it, so a later
prompt that retightens either retightens both tests at once. The test's docstring records the
substitution and why.

### D3 — `IMPLEMENTATION CHOICE` — the test covers three sites on three models, not two sites on two

The prompt's §2.4 prose asks for "the `LambdaCDM_GenericEOS` guess expression and the `main.py`
expression"; its *name* for the test is `test_the_three_closed_form_sites_agree` and its §2.1 is a
three-site measurement. The third site was included because it is the only one that uses a
**different `pow`** — `LambdaCDM.py` imports `math.pow` where the other two get the builtin — so a
two-site test would pin the pair that cannot differ and leave unpinned the pair that could.
`LambdaCDM(Planck2018)` was added to the models for the same reason: site 2 only ever executes on a
`LambdaCDM` object, and testing its expression solely on a `LambdaCDM_GenericEOS` would not
exercise the model the site runs on. `LambdaCDM` has no `_rho_fluid`, so it is scored for
site-agreement only and the test says so at the `hasattr` guard.

### D4 — `IMPLEMENTATION CHOICE` — the grid digest is taken on three trees, not two

§2.3 asks for `HEAD~1` and this commit. `HEAD~1` here **is prompt 02's commit**, so that pair
cannot detect what the prompt's stop condition is aimed at (*"a difference means prompt 02 changed
something it should not have"*). `7fdc49b` — the last tree carrying the unbracketed secant — was
measured as well. All three agree; the acceptance row is satisfied by the middle two and the third
is the one that actually clears prompt 02.

---

## Verification performed

All figures at this commit unless a tree is named. `black` reports both changed files unchanged
under `--check`.

### 1. The table and the corrected figure — **ran**

`PYTHONPATH=. ./venv/bin/python <scratch>/measure_sites.py`. Output transcribed into §2.1 and
§2.2 above, including the hex of every float. Three sites identical on 3 models × 2 pairs = 6
comparisons, 6/6 identical.

### 2. The grid digest — **ran**, on three trees

§2.3's table. `QCD_Cosmology` 1,996 samples / `a2c32f67`; `LambdaCDM(Planck2018)` 1,778 /
`60a3205a`; identical at `7fdc49b`, `921f41c` and here. **Acceptance row "identical": met.**

### 3. Option (iii)'s price — **ran**

`PYTHONPATH=. ./venv/bin/python <scratch>/measure_option_iii.py`:

```
  closed form (main.py today) : ['3406.668974249948', '0.3034230329964074']
  solve       (option iii)    : ['3406.668974249951', '0.3034230329964074']
  separation                  : [7.0, 0.0] ulp
   closed form: samples = 1996, digest = a2c32f67
         solve: samples = 1996, digest = 4849552b
```

### 4. The new test has teeth — **ran**

Prompt 03 claims no behaviour change, so there is no `HEAD~1` failure to show; but the campaign's
other stop condition is "nothing moved", and a test that tests nothing reports that too
(README §2 (e)). The guard was therefore mutated rather than the tree: multiplying `omega_m` by
`(1.0 + 2.0e-16)` inside `main_py_closed_form` alone — a **one-ulp** perturbation of the single
load-bearing site — fails the new test on **all three** models:

```
FAIL: test_the_three_closed_form_sites_agree (model='QCD_Cosmology', pair='matter_radiation')
AssertionError: 2 != 1 : QCD_Cosmology, matter_radiation: the three closed-form sites no longer
produce the same float -- main.py:553/:555 = 3406.6689742499484,
LambdaCDM_GenericEOS.py:502/:507 = 3406.668974249948, LambdaCDM.py:73/:74 = 3406.668974249948.
main.py:553/:555 is a production sample location inside the BackgroundModel lookup key, so a site
that has drifted from the others either has moved the source grid or is about to.
```

and identically for `pure-radiation stand-in` and `LambdaCDM(Planck2018)`. The mutation was
reverted; `git diff --stat` after restoring shows the two intended files only.

### 5. `main.py`'s diff — **ran**

`git diff main.py` is a single hunk, lines 525-528 → 525-529, entirely inside
`cosmology_feature_redshifts`'s triple-quoted docstring. **Executable lines changed: 0.**
**Acceptance row met.**

### 6. Suites

| Suite | Before (log 02, `921f41c`) | After | Result |
|---|---|---|---|
| `CosmologyModels/tests` | **37** | **38** | OK, 0.713 s — **+1, as the acceptance requires** |
| `ComputeTargets/tests` | **447** | **447** | OK, 171.4 s |

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
```

`CosmologyModels.tests.test_rho_equality` alone: 8 tests, OK, 0.126 s.

### 7. Machine state

Load average 54 at the time of the `ComputeTargets` run, but **not** board note 10's cause: the
`while :; do :; done` shells are gone (`ps` shows none), and the load is Adobe Creative Cloud
helper processes. No figure in this log is a timing measurement. **Prompt 05 must still check for
itself.**

### 8. Not run

No datastore, no Ray, no production pipeline. Nothing in this prompt can reach either.

---

## Observations not acted on

1. **`main.py:524-525` cites `LambdaCDM_GenericEOS.py:483` for the constructor's equality solve;
   the solve is at `:501-508`.** Line 483 is now inside the break-point-crossing cache assignment.
   The sentence carrying that citation is *not* the sentence this prompt was permitted to touch
   (`:525-527`), and the prompt is explicit that `main.py` is one docstring sentence only, so it is
   left. **Opened as a §3 issue.**
2. **`CosmologyModels/tests/test_rho_equality.closed_form_guess` reaches `math.pow` through the
   module's `from math import pow`, while `LambdaCDM_GenericEOS.__init__:507` uses the builtin.**
   Measured identical on every model and both pairs here, and the new test is now the standing
   statement of that, so `closed_form_guess` remains a faithful stand-in and nothing is wrong. It
   is worth knowing that it is faithful *by measurement* rather than by construction. Not opened as
   a separate issue: the new test is the guard, and the test's docstring records it.
3. **The grid is 1,996 samples for `QCD_Cosmology` and 1,778 for `LambdaCDM`**, where
   `qcd-background-audit` log 11 §5 records 1,773 and 1,732 and its tag labels are built from those
   counts. Prompts 12 and 15 of that campaign changed the density after log 11 was written, so the
   labels in that log's closing table are archival. Not an issue — log 11 is correct for the tree it
   was taken on (`CLAUDE.md`: verification documents are additive) — but a reader arriving at that
   table from here should not expect 1,732.
4. **`QCD_Cosmology`'s `omega_r` (`9.12940788412335e-05`) differs from `LambdaCDM`'s
   (`9.138963454890973e-05`) in the third significant figure**, which is why the two matter–radiation
   equality redshifts are 3406.67 and 3403.11. That is the equation of state's extra relativistic
   content and is expected; recorded only because §2.1's three models otherwise look as though two
   of them should agree.

---

## State handed to the next prompt

**Nothing this prompt did moves a number.** The four equality redshifts are log 02's table, bit for
bit; the two banner lines are character-identical; the production source-grid digests are
`a2c32f67` (QCD, 1,996 samples) and `60a3205a` (LambdaCDM, 1,778), identical at `7fdc49b`,
`921f41c` and here. `T_Z_REPRESENTATION_VERSION` = **6**.

**The three closed-form sites are one float.** `main.py:553`/`:555`,
`LambdaCDM_GenericEOS.py:502`/`:507` and `LambdaCDM.py:73`/`:74` agree bit for bit on
`QCD_Cosmology`, the pure-radiation stand-in and `LambdaCDM(Planck2018)`, on both pairs, despite
`LambdaCDM.py` using `math.pow` where the other two use the builtin. This is now asserted by
`CosmologyModels/tests/test_rho_equality.test_the_three_closed_form_sites_agree`, which a one-ulp
edit at any site fails on all three models.

**New public names in `CosmologyModels/tests/test_rho_equality.py`**, usable by any later prompt:

- `main_py_closed_form(cosmology, pair) -> float` — `main.py:553`/`:555`, builtin `pow`
- `generic_eos_closed_form(model, pair) -> float` — `LambdaCDM_GenericEOS.py:502`/`:507`, builtin `pow`
- `lambdaCDM_closed_form(model, pair) -> float` — `LambdaCDM.py:73`/`:74`, `math.pow`
- `CLOSED_FORM_SITES` — the three as `(label, callable)` pairs
- `TestRhoEquality.lambdaCDM` — a `LambdaCDM(Planck2018)` built once per class

`pair` is `"matter_radiation"` or `"matter_lambda"`, the same spelling log 01 and log 02 use.

**For prompt 06's provenance entry.** The closed form vs. the corrected solve, at `921f41c`:
**−9.344e-16 (−7 ulp)** matter–radiation on `QCD_Cosmology`, **−1.336e-16 (−1 ulp)** on the
stand-in, **exactly zero (same float)** for matter–$\Lambda$ on both. `main.py:525-529`'s docstring now
carries the first and the last, and names the commit. The competing figure it replaced, 4e-13, was
an unverified safe over-estimate.

**For whoever takes README §7 D2.** The price of option (iii) is measured and is a single number:
the `QCD_Cosmology` production grid digest goes **`a2c32f67` → `4849552b`** on a 7-ulp move in one
sample of 1,996, invalidating eight stored object types. Options (i) and (ii) both leave it at
`a2c32f67` today. The recommendation is **(i)**; the decision is not taken here and
`[00-equality-redshift-closed-form-is-duplicated-three-times]` stays open.

**Still open and still relevant to anything scored in ulp:**
`[02-bracketed-reference-is-not-the-exact-root]`. On the matter–$\Lambda$ pair the closed form and
the solve are the same float and it is the **reference** that is 2 ulp away, which is why this
log's §2.2 table quotes the solve rather than the reference as the thing the closed form is scored
against. A later prompt tabulating that pair against `bracketed_reference` will see "−2 ulp" and
should read that issue before treating it as error in the solve.

**Reproduction commands.**

```bash
PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_rho_equality -v
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
```

The three measurement scripts (`measure_sites.py`, `measure_grid.py`, `measure_option_iii.py`) were
written to the session scratchpad rather than the tree: each is under 80 lines, each reads only
`CosmologyModels/tests/test_rho_equality.py` and `ComputeTargets/tests/test_source_grid.py`
helpers that are already in the tree, and every figure they produce is transcribed above. The grid
digests are reproducible from `ComputeTargets/tests/test_source_grid._production_grid(cosmology)`
plus `_to_redshift_array(grid.z_values).digest()` in four lines.
