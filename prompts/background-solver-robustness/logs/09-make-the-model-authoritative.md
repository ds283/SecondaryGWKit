# Log 09 — Make the model authoritative for its own equality redshifts

**Prompt:** prompts/background-solver-robustness/09-make-the-model-authoritative.md
**Commit:** *(this commit)* — Make the cosmology authoritative for its equality redshifts
**Model:** Claude Opus 5
**Date:** 2026-09-16
**Result:** COMPLETE WITH DEVIATIONS

---

## What shipped

`T_Z_REPRESENTATION_VERSION` = **6** before and **6** after (`LambdaCDM_GenericEOS.py:419`).
Nothing in this prompt reads or writes it; it is quoted because README §5 and board note 2 require
it at every commit.

| File | Before → after |
|---|---|
| `CosmologyModels/base.py` | Two new abstract properties on `BaseCosmology`, alongside `H0`, `T_photon`, `wBackground` and `wPerturbations`: **`z_matter_radiation_equality -> float`** and **`z_matter_lambda_equality -> float`**. Each docstring states the contract — the model is authoritative, a consumer may not compute the value instead, and *a subclass for which the radiation-domination closed form is not exact must not return it* — with the reason ($\Omega_r$ is a present-day density parameter, so $1+z_{\rm eq}=\Omega_m/\Omega_r$ is exact only while $\rho_r\propto(1+z)^4$ holds from today back to equality) and the user's sentence that `BaseCosmology` declares the obligation and does not police it. `check_cosmology` and everything else in the file untouched. |
| `CosmologyModels/LambdaCDM/LambdaCDM.py` | The two properties, implemented with the closed form, placed after `H0` and before `rho`. The docstrings say **exact**, not "a good approximation": this model has no equation of state, `rho` is `rho_m0 (1+z)^3 + rho_r0 (1+z)^4 + rho_cc` identically, so `1 + z = Omega_m/Omega_r` solves `rho_m = rho_r` with nothing left over, and there is no root solve because there is no root to solve for. `:73-74`'s banner diagnostic is now `matter_radiation_equality = self.z_matter_radiation_equality` / `matter_cc_equality = self.z_matter_lambda_equality` — a **use** of the properties, not a fourth transcription. The two `print` lines are unchanged and their output is character-identical. |
| `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` | `__init__`'s two `_find_rho_equality` results land on `self._z_matter_radiation_equality` / `self._z_matter_lambda_equality` instead of two locals that were printed and discarded; the two properties return them. **The two `init_z=` expressions are character-identical** (`git diff` shows both context-unchanged), and `_find_rho_equality`'s body, bracket policy and tolerances are not in the diff at all. The two banner `print`s now read the attributes; `black` split the first over three lines, and the printed text is character-identical. |
| `main.py` (`cosmology_feature_redshifts`, `:507-586`) | The body reads the two properties and **computes nothing**. The three `getattr(cosmology, "omega_*", None)` lookups and both arithmetic expressions are gone. A cosmology that declares break points but does not supply a redshift raises `RuntimeError` naming the cosmology, its class, which of the two was missing and the attribute it was looked for under. The docstring paragraph is rewritten: the model is authoritative, base `LambdaCDM`'s closed form is exact and `GenericEOS`'s may not be, prompt 03's measured agreement is re-pointed at the solve's **initial guess** rather than at what reaches the grid, and the absence of a fallback is stated as deliberate. The early return at `:563-564` and the paragraph explaining it are unchanged. |
| `ComputeTargets/tests/test_source_grid.py` | New `_BrokenCosmology` stand-in and new class `TestTheModelIsAuthoritativeForItsEqualityRedshifts` with two tests (§4.2 items 1 and 2). `_SmoothCosmology`'s docstring re-pointed. `TestTheProtectedSetOnQCD`'s `feature_z[0]` literal `3406.668974249948` → **`3406.6689742499511`** with its comment rewritten (§4.3). Two digest literals moved: the break-point-only QCD grid `303f9ce7` → **`81c6e682`** at 1,773 samples, and the production QCD grid `a2c32f67` → **`4849552b`** at 1,996. `LambdaCDM`'s `60a3205a` / 1,778 and the bare lattice's `0960e169` / 1,732 are untouched. |
| `CosmologyModels/tests/test_rho_equality.py` | `main_py_closed_form` and its `CLOSED_FORM_SITES` entry deleted; the comment block above them rewritten to say that `main.py` no longer computes the quantity and where it gets it instead. `test_the_three_closed_form_sites_agree` → **`test_the_remaining_closed_form_sites_agree`**, reworked around the two sites that remain and extended to assert that `LambdaCDM`'s **property** returns exactly the expression the module transcribes. New helper `model_property(model, pair) -> float` and new test **`test_each_model_answers_for_its_own_equality_redshifts`** (§4.2 item 3 and its `GenericEOS` counterpart). `LAMBDA_CLOSED_FORM_ULP` (2), `MATTER_RADIATION_CLOSED_FORM_ULP` (8) and `SOLVE_VS_REFERENCE_ULP` (4) are **unchanged** (board note 7). |

New public symbols:

- `BaseCosmology.z_matter_radiation_equality` — abstract property, `-> float`
- `BaseCosmology.z_matter_lambda_equality` — abstract property, `-> float`
- concrete implementations of both on `LambdaCDM` and `LambdaCDM_GenericEOS`
- `CosmologyModels/tests/test_rho_equality.model_property(model, pair) -> float`
- `ComputeTargets/tests/test_source_grid._BrokenCosmology`

Removed: `CosmologyModels/tests/test_rho_equality.main_py_closed_form`, and the
`"main.py:553/:555"` entry of `CLOSED_FORM_SITES` (now two entries, relabelled
`"LambdaCDM_GenericEOS.py guess"` and `"LambdaCDM.py property"`).

---

## The two equality redshifts

**No bit moved.** Every one of the four values `_find_rho_equality` produces is log 02's and
log 03's, re-taken at this commit through the new properties
(`PYTHONPATH=. ./venv/bin/python <scratch>/measure_props.py`):

| Model | Pair | Before **and** after | `float.hex()` |
|---|---|---|---|
| `QCD_Cosmology(max_z=1e12)` | matter = radiation | `3406.668974249951` | `0x1.a9d5683cafad0p+11` |
| `QCD_Cosmology(max_z=1e12)` | matter = $\Lambda$ | `0.3034230329964074` | `0x1.36b4870e4a718p-2` |
| `PureRadiationEOS(max_z=1e6)` | matter = radiation | `3403.1059638279457` | `0x1.a963640e40f2cp+11` |
| `PureRadiationEOS(max_z=1e6)` | matter = $\Lambda$ | `0.3034230329964074` | `0x1.36b4870e4a718p-2` |
| `QCD_Cosmology(max_z=1e20)` | matter = radiation | `3406.668974249951` | `0x1.a9d5683cafad0p+11` |
| `QCD_Cosmology(max_z=1e20)` | matter = $\Lambda$ | `0.3034230329964074` | `0x1.36b4870e4a718p-2` |

Relative move against log 03: **0.0 on all six, bit for bit.** Nothing about the solve changed —
same two calls, same closed-form guesses, same bracket, same `xtol=1e-300, rtol=8.9e-16` — and the
diff of `_find_rho_equality` is empty. The two printed banner lines are character-identical
(board note 5), on both models; `black` reflowed one of the two `print` calls in
`LambdaCDM_GenericEOS` across three source lines, which does not touch the text.

`LambdaCDM(Planck2018)`, for completeness, answers `3403.1059638279453`
(`0x1.a963640e40f2bp+11`) and `0.3034230329964074` (`0x1.36b4870e4a718p-2`) — its closed form,
which for that model is the exact root.

**What did move is which of these reaches the grid**, which is §3 below and is the point of the
prompt.

---

## What moved, on purpose

Prompt 03 measured the landing site and this commit lands on it.

| Quantity | `HEAD~1` (`6f3cd8e`) | this commit | Prompt §3's target |
|---|---|---|---|
| `QCD_Cosmology` production grid digest | `a2c32f67` | **`4849552b`** | `4849552b` ✅ |
| `QCD_Cosmology` production sample count | 1,996 | **1,996** | unchanged ✅ |
| `QCD_Cosmology` `feature_z[0]` | `3406.668974249948` | **`3406.668974249951`** | log 02's solve ✅ |
| `QCD_Cosmology` `feature_z[1]` | `0.3034230329964074` | **`0.3034230329964074`** | the same double ✅ |
| `LambdaCDM(Planck2018)` digest | `60a3205a` | **`60a3205a`** | unchanged ✅ |
| `LambdaCDM(Planck2018)` sample count | 1,778 | **1,778** | unchanged ✅ |
| samples that differ | — | **exactly 1** (index **1540**, **+7.0 ulp**) | exactly one ✅ |

`LambdaCDM`'s `feature_z` is `[]` on both trees: the whole cosmology-aware path is gated on the
cosmology declaring break points, and it declares none, so it never reaches the property at all.

One further digest moves and the prompt does not mention it: the **break-point-only** QCD grid
(`build_z_sample` with `break_z` and `feature_z` but no `spacing`), `303f9ce7` → **`81c6e682`**,
1,773 samples unchanged. It carries the same single moved sample. That is deviation D1.

**There is no regeneration to do** (the user, 2026-09-16; board note 14). This is the build phase
of a science code; no stored data is curated here, no migration is added, and nothing about the
moved digest is left outstanding. The digest moving *is* the acceptance.

---

## Deviations from the prompt

### D1 — `STRUCTURALLY REQUIRED` — a second digest literal in `test_source_grid.py` had to move

The prompt's §4.3 names one thing in that file to re-point, `:315-319`'s `feature_z` literals, and
§5's acceptance names one digest, the production grid's. The file in fact pins **two** QCD digests
in `test_the_production_grids_are_the_ones_the_campaign_recorded`: the production grid
(`a2c32f67`, 1,996 samples) *and* prompt 11/14's break-point-only grid (`303f9ce7`, 1,773), built
from the same `break_z` and `feature_z` with no spacing profile. The moved sample is in both, so
the second literal is `81c6e682` at this commit and the test cannot pass otherwise.

Alternatives considered: none is available — the prompt permits the file, requires the production
digest to land on `4849552b`, and a suite that fails is a stop. What was done instead of silently
retyping it: both literals carry a comment naming the old value, prompt 09 and the single moved
sample, so the change is legible in the diff and in `git blame`. The break-point grid's count
(1,773), the bare lattice (`0960e169`, 1,732) and both `LambdaCDM` rows are untouched, which is
the check that nothing beyond the intended sample moved.

### D2 — `IMPLEMENTATION CHOICE` — the reworked guard is renamed

`test_the_three_closed_form_sites_agree` is now
**`test_the_remaining_closed_form_sites_agree`**. The prompt says to rework it and not to delete
it, and is silent on the name. There are now two sites, not three, so the old name is false;
leaving it would mean a reader grepping for "the three closed-form sites" finds a test of two.
The alternative — keeping the name and explaining the arithmetic in the docstring — was rejected
because the name is the first thing a failure prints. Board standing note 13 cites the old name
and is updated in this commit. Nothing else in the tree references it (`grep` over `*.py` and
`*.md`: the board, `docs/OPEN_ISSUES.md` and log 03, all amended or additive).

### D3 — `IMPLEMENTATION CHOICE` — `LambdaCDM`'s site stays a transcription, with the property checked against it

Prompt §4.1 says the remaining sites are "`LambdaCDM`'s property and `LambdaCDM_GenericEOS`'s two
initial guesses". `CLOSED_FORM_SITES` is applied to **all three models**, including two
`LambdaCDM_GenericEOS` instances whose properties return the *solve* and not the closed form, so a
site callable that read `model.z_matter_radiation_equality` would compare the solve against a
closed form and fail by 7 ulp on `QCD_Cosmology` — it would no longer be a test of the `math.pow` /
builtin-`pow` split, which is the only way the two sites can differ and the only reason the test
exists. `lambdaCDM_closed_form` therefore remains a transcription, relabelled
`"LambdaCDM.py property"`, and the tie to the original is asserted separately: on the one model
where that expression *is* the answer, the test asserts
`model_property(model, pair).hex() == values["LambdaCDM.py property"].hex()` in the branch that
used to `continue` past `LambdaCDM` with nothing checked. §4.2 item 3 is satisfied there and again
in `test_each_model_answers_for_its_own_equality_redshifts`.

### D4 — `IMPLEMENTATION CHOICE` — the two §4.2 behaviour tests live in `test_source_grid.py`

The prompt does not say which of its two permitted test files takes them. Tests 1 and 2 are about
`cosmology_feature_redshifts`, which cannot be imported (`CLAUDE.md`: `main.py` parses `sys.argv`
and opens a Ray connection at module scope) and is reached only through
`load_main_py_functions`. `test_source_grid.py` already loads it that way and already builds
`QCD_Cosmology` at the production `max_z = 1e20`; `test_rho_equality.py` does neither and would
have to acquire both. Test 3, which is about the models rather than about the consumer, is in
`test_rho_equality.py`.

### D5 — `IMPLEMENTATION CHOICE` — the acceptance grep is clean of code but not of prose

§5's row reads: `grep -n "omega_r\|omega_cc" main.py` inside `cosmology_feature_redshifts` →
**nothing**. The executable body matches nothing. One docstring line does:

```
    This function used to recompute both from the public ``omega_m`` / ``omega_r`` / ``omega_cc``
```

It is the sentence that records what was removed and why, and the paragraph it opens carries the
physical argument the user's decision rests on ($\Omega_r$ is a present-day density parameter).
The two ways to make the grep literally empty were to drop the sentence — which loses the reason
the change exists, at the one site a later reader will look for it — or to spell the names
`Omega_r`/`Omega_cc`, which is gaming a check rather than satisfying it. Neither was taken. The
row's intent, that nothing in the function computes a redshift from the $\Omega$s, is met: see
Verification §4.

### D6 — `IMPLEMENTATION CHOICE` — `test_source_grid.py:318`'s `places=6` was left as it is

§4.3 says to re-point the literal and the comment, which is done. It does **not** say to tighten
the comparison, and `places=6` cannot resolve a 3.2e-12 move, so that assertion still passes for
either quantity. Tightening it was not done (README §5 rule 5), and it is not needed: the guard
with teeth is §4.2 test 1, which compares `float.hex()` and asserts the closed form is **not**
what reaches `feature_z`. Recorded here so that a later reader does not mistake the loose
assertion for the new one.

---

## Verification performed

Everything below was **run** at this commit unless a tree is named. The machine was loaded
(load average 16–18, Adobe Creative Cloud helpers; board note 10's `while :; do :; done` shells
are gone). No figure in this log is a timing measurement.

### 1. The digests and the single moved sample — ran

`PYTHONPATH=. ./venv/bin/python <scratch>/measure_grid.py`, which drives `main.py`'s own
`cosmology_feature_redshifts` and `source_grid_spacing_profile` through
`ComputeTargets/tests/test_source_grid._production_grid`, with `QCD_Cosmology` at the production
`max_z = 1e20`.

At `6f3cd8e` (`HEAD~1`):

```
QCD_Cosmology
  feature_z  = ['3406.668974249948', '0.3034230329964074']
  hex        = ['0x1.a9d5683cafac9p+11', '0x1.36b4870e4a718p-2']
  break-point grid : samples = 1773, digest = 303f9ce7
  production grid  : samples = 1996, digest = a2c32f67
LambdaCDM(Planck2018)
  feature_z  = []
  break-point grid : samples = 1732, digest = 0960e169
  production grid  : samples = 1778, digest = 60a3205a
```

At this commit:

```
QCD_Cosmology
  feature_z  = ['3406.668974249951', '0.3034230329964074']
  hex        = ['0x1.a9d5683cafad0p+11', '0x1.36b4870e4a718p-2']
  break-point grid : samples = 1773, digest = 81c6e682
  production grid  : samples = 1996, digest = 4849552b
LambdaCDM(Planck2018)
  feature_z  = []
  break-point grid : samples = 1732, digest = 0960e169
  production grid  : samples = 1778, digest = 60a3205a
```

**`4849552b` at 1,996, which is the value prompt 03 and the orchestrator measured independently.
`60a3205a` at 1,778, unmoved.**

`<scratch>/measure_moved.py` builds both production grids side by side from the same
`break_z` and the same spacing profile, differing only in `feature_z`, and diffs them
element-wise:

```
  3406.668974249948 -> 3406.668974249951   ulp = +7.0
  0.3034230329964074 -> 0.3034230329964074   ulp = +0.0
old: samples = 1996, digest = a2c32f67
new: samples = 1996, digest = 4849552b
lengths: 1996 vs 1996
samples that differ: 1 at indices [1540]
  [1540] 3406.668974249948 -> 3406.668974249951  ulp = +7.0
```

**Exactly one sample of 1,996.** Acceptance rows "exactly one" and "1,996 samples": met.

### 2. §4.2 test 1 fails on `HEAD~1` — ran

A `git worktree add --detach` at `6f3cd8e` with the main checkout's `venv` symlinked in and the
two new test files copied over it. Both new `test_source_grid.py` tests fail, and the
`assertNotEqual` is the one that matters:

```
ERROR: test_feature_z_is_the_models_answer_and_not_the_closed_form (attribute='z_matter_radiation_equality')
AttributeError: 'QCD_Cosmology' object has no attribute 'z_matter_radiation_equality'

ERROR: test_feature_z_is_the_models_answer_and_not_the_closed_form (attribute='z_matter_lambda_equality')
AttributeError: 'QCD_Cosmology' object has no attribute 'z_matter_lambda_equality'

FAIL: test_feature_z_is_the_models_answer_and_not_the_closed_form
AssertionError: '0x1.a9d5683cafac9p+11' == '0x1.a9d5683cafac9p+11' : feature_z[0] =
3406.668974249948 is the radiation-domination closed form Omega_m/Omega_r - 1 =
3406.668974249948, not the model's own solve. The two differ by 7 ulp on this cosmology and the
closed form is the one that is only accidentally right.

FAIL: test_a_cosmology_that_cannot_answer_raises_instead_of_falling_back
AssertionError: RuntimeError not raised

Ran 2 tests in 0.097s
FAILED (failures=2, errors=2)
```

and in `CosmologyModels/tests/test_rho_equality.py`:

```
ERROR: test_the_remaining_closed_form_sites_agree (model='LambdaCDM(Planck2018)', pair='matter_lambda')
  File ".../CosmologyModels/tests/test_rho_equality.py", line 188, in model_property
    return float(model.z_matter_lambda_equality)
AttributeError: 'LambdaCDM' object has no attribute 'z_matter_lambda_equality'

Ran 2 tests in 0.084s
FAILED (errors=9)
```

**README §2 (e): met.** The second `test_source_grid` failure — `RuntimeError not raised` — is the
"no fallback" assertion, and its failure on `HEAD~1` is the statement that the old code *did*
silently supply the closed form for a cosmology that never vouched for it.

### 3. The no-fallback error, at this commit — ran

```
cosmology_feature_redshifts: the cosmology 'a cosmology that cannot answer' (_BrokenCosmology)
declares break points, so the source grid is built around the features it declares, but it does
not supply matter-radiation equality as 'z_matter_radiation_equality'. BaseCosmology declares
that obligation and every production cosmology answers it. There is no fallback to
1 + z_eq = Omega_m/Omega_r here on purpose -- see this function's docstring -- so a cosmology that
declares non-smoothness and cannot say where its own equality redshifts are is a broken cosmology,
not a grid to build anyway.
```

`_BrokenCosmology` carries `omega_m`, `omega_r` and `omega_cc`, so the removed code path would
have answered it. That is the point.

### 4. No fallback, and nothing computed — ran

```
$ sed -n '/^def cosmology_feature_redshifts/,/^def pre_grid_background_proxy/p' main.py \
    | grep -n "omega_m\|omega_r\|omega_cc"
23:    This function used to recompute both from the public ``omega_m`` / ``omega_r`` / ``omega_cc``
```

One hit, in the docstring, in the sentence that says the recomputation was removed (deviation D5).
**Executable lines of `cosmology_feature_redshifts` mentioning any `omega_*`: 0.** There is no
`getattr(..., None)` in the function, no `if ... is None`, and the only `except` clause re-raises
as `RuntimeError`.

`grep -n "omega_m / omega_r\|omega_cc / omega_m" main.py` → no match anywhere in the file.

### 5. `_find_rho_equality` and the two initial guesses — ran

`git diff CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` is three hunks: the constructor's
two assignments and their comment, the two banner `print`s, and the two new properties.
`_find_rho_equality` does not appear in the diff at all — **body, bracket-expansion policy,
`BRACKET_EXPANSION_FACTOR`, `BRACKET_EXPANSION_MAX_STEPS`, the clamp, both `RuntimeError`s and
`xtol=1e-300, rtol=8.9e-16` are character-identical.** Both `init_z=` lines are context lines in
the diff, so the guesses are character-identical too. Acceptance rows: met.

### 6. `T_Z_REPRESENTATION_VERSION` — ran

`LambdaCDM_GenericEOS.py:419: T_Z_REPRESENTATION_VERSION: int = 6`, unchanged. No `CosmologyModels`
change here touches the $T(z)$ representation.

### 7. Suites

| Suite | Before (`6f3cd8e`) | After | Result |
|---|---|---|---|
| `CosmologyModels/tests` | **38**, OK, 0.715 s | **39**, OK, 0.708 s | +1 (`test_each_model_answers_for_its_own_equality_redshifts`) |
| `ComputeTargets/tests` | **447**, OK, 176.4 s | **449**, OK, 177.1 s | +2 (the two tests of §4.2) |

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
```

`CosmologyModels.tests.test_rho_equality` alone: 9 tests, OK, 0.117 s.
`ComputeTargets.tests.test_source_grid` alone: 38 tests, OK, 8.4 s.
No count falls.

### 8. `black`

`./venv/bin/python -m black main.py CosmologyModels/base.py
CosmologyModels/LambdaCDM/LambdaCDM.py CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py
CosmologyModels/tests/test_rho_equality.py ComputeTargets/tests/test_source_grid.py` — one file
reformatted at the time of writing (`test_source_grid.py`), all six clean under `--check`
afterwards.

### 9. Not run

No datastore, no Ray, no production pipeline; nothing in this prompt can reach any of them. No
regeneration was attempted and none is outstanding (board note 14).

---

## Observations not acted on

1. **`ComputeTargets/tests/test_retire_samples_per_decade_tag.py:29-32`'s docstring now cites a
   superseded digest.** It says the production grids are pinned "to 1,996 samples / digest
   ``a2c32f67`` (QCD) and 1,778 / ``60a3205a`` (LambdaCDM)". The QCD half is `4849552b` from this
   commit. Nothing there *asserts* the digest — the file pins `SOURCE_GRID_CONSTRUCTION_VERSION`
   and `T_Z_REPRESENTATION_VERSION` only, and the whole suite passes — so this is prose pointing
   at a test in another file. The file is not in this prompt's permitted list (README §5 rule 5).
   **Opened as a §3 issue.**
2. **`docs/qcd-background-verification.md:898` and several `prompts/qcd-background-audit/` logs
   record `SourceRedshiftGrid_1773_303f9ce7` and `a2c32f67`.** Those are verification documents
   and campaign logs, correct for the trees they were taken on, and `CLAUDE.md` says they are
   additive. Nothing to do; recorded so a later reader arriving from one of them knows why the
   digests differ.
3. **`docs/qcd-background-audit/grid_density_criterion.py`,
   `source_grid_consumer_check.py` and `equidistributed_grid_check.py` build grids through
   `cosmology_feature_redshifts`** and will therefore produce the moved grid if re-run. They are
   measurement scripts, not tests; their recorded outputs live in verification documents that are
   additive. `[09-audit-script-section-5-prose-counts-the-wrong-set]` already assigns
   `docs/qcd-background-audit/` to prompt 05, which re-runs one of them.
4. **`CosmologyModels/tests/test_rho_equality.py`'s module docstring still says the campaign
   replaces the secant "at ``xtol=1e-300, rtol=1e-14``".** The shipped tolerance is `rtol=8.9e-16`
   (the user's amended README §7 D1; board item (b)). That sentence predates prompt 02 and was not
   this prompt's subject; it is a stale figure in the same file this prompt edits, which is why it
   is recorded rather than silently corrected. Not opened as a separate issue: it is one clause of
   the same class as `[01-agreement-threshold-comment-predates-the-representation]` and prompt 06
   has the provenance of all three solves in scope.

---

## State handed to the next prompt

**The digest moved, and that was the acceptance.** The `QCD_Cosmology` production source-grid
digest is **`4849552b`** at **1,996** samples; `LambdaCDM(Planck2018)` is **`60a3205a`** at
**1,778**, unmoved. The QCD break-point-only grid (no spacing profile) is **`81c6e682`** at
**1,773**, was `303f9ce7`. Board note 12's pair is superseded for QCD by note 15's exception and
this prompt is that exception; **note 12 still binds for `LambdaCDM`, and for QCD the value to
compare against from here on is `4849552b`.**

Reproduce with `ComputeTargets/tests/test_source_grid._production_grid(cosmology)` plus
`_to_redshift_array(grid.z_values).digest()`, `QCD_Cosmology` built at the production
`max_z = 1e20` and not the test modules' `1e12`.

**The four equality redshifts did not move**, bit for bit, and neither did the banner lines. The
values are the table above; `LambdaCDM(Planck2018)` answers `3403.1059638279453` /
`0.3034230329964074`, its closed form.

**New surface any later prompt may use.** `BaseCosmology.z_matter_radiation_equality` and
`BaseCosmology.z_matter_lambda_equality`, abstract properties returning `float`. Both concrete
cosmologies implement them; there are only two (`LambdaCDM` and `LambdaCDM_GenericEOS`), so making
them abstract costs nothing today, but **a new `BaseCosmology` subclass must now implement both or
it cannot be instantiated.** Duck-typed stand-ins that are not subclasses are unaffected unless
they declare `integration_break_points`, in which case `cosmology_feature_redshifts` raises.

**`main.py` no longer holds a copy of either closed form.** Board note 13's "three closed-form
sites" is now two: `LambdaCDM_GenericEOS.__init__`'s two initial guesses and `LambdaCDM`'s two
properties. The guard is
`CosmologyModels/tests/test_rho_equality.test_the_remaining_closed_form_sites_agree` — **renamed**
from `test_the_three_closed_form_sites_agree` (deviation D2).

**For prompt 05**, which edits `LambdaCDM_GenericEOS.py` next: this prompt added two properties
after `H0` and changed two lines of `__init__` and two `print`s. It did not touch
`TemperatureRepresentation`, `_solve_T_z`, `_rho_fluid`, `_build_T_z_spline` or any range logic,
so prompt 05's bit-identity acceptance is unaffected by it. `T_Z_REPRESENTATION_VERSION` is **6**.

**For prompt 06**, the close-out: `[00-equality-redshift-closed-form-is-duplicated-three-times]`
and `[03-main-py-cites-a-stale-line-for-the-equality-solve]` are both closed here, and one issue
is opened (observation 1). `main.py`'s `cosmology_feature_redshifts` docstring is the place the
D2 decision and its reason are written down in production code.

**Line numbers.** `main.py`'s `cosmology_feature_redshifts` now runs `:507-586` (was `:507-557`);
below it everything shifts by **+29**. Re-anchored: `pre_grid_background_proxy` is **`:589`**;
`break_z, feature_z = cosmology_feature_redshifts(...)` is **`:936`** (was `:907`);
`feature_z=feature_z` is **`:961`** (was `:932`). Board note 11's +4 is superseded by this for
anything below `:558`. In `LambdaCDM_GenericEOS.py` the two `_find_rho_equality` calls start at
**`:512`** and **`:515`** (their `init_z=` guesses at **`:513`** and **`:518`**, were `:502`
and `:507`), the two banner `print`s are **`:527-530`**, and the two new properties are at
**`:549`** and **`:566`**. In `LambdaCDM.py` the closed-form expressions now live inside the two
properties, at **`:121`** and **`:132`** (were `:73` and `:74`), and the banner reads the
properties at **`:75-76`**.

**Reproduction commands.**

```bash
PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_rho_equality -v
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_source_grid -v
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
```

The three measurement scripts (`measure_grid.py`, `measure_moved.py`, `measure_props.py`) were
written to the session scratchpad rather than the tree: each is under 50 lines, each reads only
helpers already in `ComputeTargets/tests/test_source_grid.py`, and every figure they produce is
transcribed above. The digest comparison is four lines from `_production_grid` and
`_to_redshift_array`.
