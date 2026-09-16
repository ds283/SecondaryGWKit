# Prompt 09 — Make the model authoritative for its own equality redshifts

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** README §2 **(l)**
**Implements:** README §7 **D2**, decided by the user 2026-09-16 — **option (iii)**, by the
mechanism §2 below sets out, which is not the mechanism D2's option (iii) describes. Read §1.
**Closes:** `[00-equality-redshift-closed-form-is-duplicated-three-times]`
**Measurements:** [`logs/03-equality-redshift-consumers.md`](logs/03-equality-redshift-consumers.md)
§2.1, §2.3 and §3 — prompt 03 measured everything this prompt needs, including **the digest this
prompt must land on**.
**Depends on:** **02** (the bracketed solve is what makes an inaccurate guess harmless) and **03**
(the measurement and the decision).
**Recommended model:** **Opus** — it changes an inheritance contract, moves a datastore identity on
purpose, and the one thing it must not do is the thing that looks most careful.

**Files you may touch:** `CosmologyModels/base.py`, `CosmologyModels/LambdaCDM/LambdaCDM.py`,
`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`, `main.py`
(`cosmology_feature_redshifts` — **body and docstring**, and this is the first prompt in the
campaign permitted its body), `ComputeTargets/tests/test_source_grid.py`,
`CosmologyModels/tests/test_rho_equality.py`, plus this campaign's log, board and
`docs/OPEN_ISSUES.md`.

**Do not touch:** `_find_rho_equality`'s **body, bracket policy or tolerances** — prompt 02 settled
them and README §7 D1 is the user's; the two **initial guesses** at `LambdaCDM_GenericEOS.py:502`
and `:507`, which **stay closed forms** (§2.4); `CosmologyConcepts/wavenumber.py`; `Datastore/`;
`_solve_T_z`; `_bisect_temperature_crossing_log1pz`; `ComputeTargets/BackgroundModel.py`.

**Read first:** `logs/03-equality-redshift-consumers.md` in full — it is the measurement this
prompt acts on; `main.py:506-556` (`cosmology_feature_redshifts`) and `:900-935`;
`CosmologyModels/base.py` in full (it is short, and it is the contract you are extending);
`CosmologyModels/LambdaCDM/LambdaCDM.py:60-85`;
`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:495-520` (the constructor's two solves and the
banner) and `:999-1100` (`_find_rho_equality`, for what it now guarantees);
`ComputeTargets/tests/test_source_grid.py:305-325`; and
`CosmologyModels/tests/test_rho_equality.py:465-540` (`test_the_three_closed_form_sites_agree`,
which prompt 03 shipped to guard the option this decision did not take).

---

## 1. Why this prompt exists, and why it is not D2's option (iii) as written

`main.py:553`/`:555` computes $1+z_{\rm eq} = \Omega_m/\Omega_r$ and
$1+z_\Lambda = (\Omega_\Lambda/\Omega_m)^{1/3}$ and puts both into `feature_z`, which becomes a
protected sample in the production source grid, whose digest is a `BackgroundModel` lookup key
(log 03 §3 item 1 has the chain, hop by hop).

**$\Omega_r$ is a present-day density parameter**, so the closed form is exact only if
$\rho_r \propto (1+z)^4$ all the way from today back to equality. On `QCD_Cosmology` that holds to
7 ulp for one reason: all of the $g_*(T)$ structure sits at $z\sim10^{12}$, twelve orders above
$z_{\rm eq}$. A cosmology with entropy injection below $z_{\rm eq}$, a decaying species, or extra
relativistic degrees of freedom appearing late breaks it — **and not by ulps**. The closed form's
accuracy at that site is a property of where the QCD transition happens to sit, not of the code.

The campaign already made this argument one level down. Prompt 02 refused to short-circuit
`_find_rho_equality` when the guess is already the root, and wrote the reason into the solve
(`LambdaCDM_GenericEOS.py`, at the `f_guess` evaluation):

> That the guess is already the root to rounding is a property of this equation of state at this
> redshift ... not of the code, and writing it in would make permanent the very accident this
> change exists to remove.

This prompt applies the same principle at `main.py`. **The user's decision (2026-09-16) is that the
closed form must not be the thing that reaches `feature_z`.**

**But not by `main.py` importing `_find_rho_equality`**, which is how README §7 D2 words option
(iii). The distinction that matters is not *has a solver* — it is *is the closed form valid for
this model*, and only the model knows. In base `LambdaCDM` the closed form **is** the answer, not
an approximation to it: that model has no equation of state and $\rho_r\propto(1+z)^4$ by
construction. In `LambdaCDM_GenericEOS` it may not be. So the model answers, and `main.py` asks.

**The inheritance contract is the user's, stated 2026-09-16:** it is the cosmology's business to
supply the correct equality redshifts. `BaseCosmology` declares that obligation; it does not police
it. A subclass that chooses to answer incoherently has made its own problem.

## 2. The change

### 2.1 `BaseCosmology` declares the obligation

Two properties, alongside `H0`, `T_photon`, `wBackground` and `wPerturbations`, in the same style
as the surface already there:

```python
z_matter_radiation_equality
z_matter_lambda_equality
```

The docstring on each says what the contract is — *the redshift at which this cosmology's matter
and radiation (or matter and $\Lambda$) energy densities are equal, as that cosmology computes it*
— and that a subclass for which the radiation-domination closed form is not exact **must not**
return it.

### 2.2 `LambdaCDM` implements them with the closed form

Because for that model the closed form is exact, and the docstring must say so in those terms —
not "a good approximation", not "agrees to 7 ulp", but: this model has no equation of state,
$\rho_r\propto(1+z)^4$ holds by construction, and therefore $\Omega_m/\Omega_r - 1$ *is* the root.
`:73-74`'s existing banner diagnostic becomes a **use** of the property rather than a fourth
transcription of the expression.

### 2.3 `LambdaCDM_GenericEOS` implements them from the solve it already runs

`__init__:501-508` already calls `_find_rho_equality` twice and drops both results into locals that
are printed at `:516-517` with `:.4g` and discarded. **Keep them.** The properties return them.
No new solve, no extra cost, nothing about the solve changes.

### 2.4 The two initial guesses stay closed forms

`:502` and `:507` keep `self.omega_m / self.omega_r - 1.0` and
`pow(self.omega_cc / self.omega_m, 1.0/3.0) - 1.0`. This is the user's decision and it is correct:
$z_{\rm eq}$ cannot move by a large factor without wrecking the CMB, so radiation domination is a
sound *guess*, and prompt 02's bracketing makes a wrong guess harmless anyway — it expands by
$\sqrt2$ per step up to $2^{70}$, clamps to the tabulated range, and raises its own named
`RuntimeError` rather than wandering. **Do not "finish the job" by removing them.**

### 2.5 `main.py` computes nothing

`cosmology_feature_redshifts` reads the two properties. The three `getattr(..., None)` guards on
`omega_m`/`omega_r`/`omega_cc` and both arithmetic expressions go.

**There is to be no fallback to the closed form. This is the whole point of the prompt and it is
the one mistake that will look like care.** Something of this shape —

```python
z_eq = getattr(cosmology, "z_matter_radiation_equality", None)
if z_eq is None:
    z_eq = float(omega_m / omega_r - 1.0)     # <- reinstates the defect
```

— is a **stop condition**, not a defensive measure: a nonstandard cosmology that for any reason
does not supply the value would silently receive the answer this prompt exists to remove, with
nothing to say so.

What is permitted instead:

- **No break points → no features**, exactly as today. The early return at `:545-546` is unchanged
  and is still what keeps every `LambdaCDM`, `RadiationModel` and stand-in on the untouched code
  path. Nothing in §2.5 reaches a cosmology that declares no non-smoothness.
- **Break points declared but the cosmology cannot answer → raise**, naming the cosmology and
  which of the two it could not supply. It is a broken cosmology, not a grid to build anyway.

The docstring paragraph at `:521-531` is rewritten to describe what the function now does and why:
that the model is authoritative, that base `LambdaCDM`'s closed form is exact and `GenericEOS`'s
may not be, and that there is deliberately no fallback. Prompt 03's corrected agreement figure is
now a statement about the *guess*, not about what reaches the grid — re-point it, do not delete the
measurement.

## 3. What moves, and what must not

**This is the first prompt in the campaign that moves a stored identity on purpose.** README §0.2's
*"no prompt moves a stored number or implies a regeneration"* is scoped to workstreams A–C; this
prompt is workstream E and is outside it. Say so in the log rather than treating the move as a
deviation.

Prompt 03 measured the landing site, so **you have a digest to hit, not merely one to report**:

| Quantity | Before | After |
|---|---|---|
| `QCD_Cosmology` source-grid digest | `a2c32f67` | **`4849552b`** |
| `QCD_Cosmology` sample count | 1,996 | **1,996 — unchanged** |
| `QCD_Cosmology` `feature_z[0]` | `3406.668974249948` | **`3406.6689742499511`** (log 02's solve) |
| `QCD_Cosmology` `feature_z[1]` | `0.3034230329964074` | **`0.3034230329964074` — the same double** |
| `LambdaCDM(Planck2018)` digest | `60a3205a` | **`60a3205a` — unchanged** |
| `LambdaCDM(Planck2018)` sample count | 1,778 | **1,778 — unchanged** |

Two of those rows are the real tests. **Only one sample moves**: matter–$\Lambda$ is the same double
from either route (verified — both are `0x1.36b4870e4a718p-2`), so a second moved sample means
something else changed. And **`LambdaCDM`'s digest must not move at all**: its closed form is exact,
its property returns the same float, and it never reaches the feature path anyway.

Build the grids with `ComputeTargets/tests/test_source_grid._production_grid`, which drives
`main.py`'s own functions through `load_main_py_functions` (`CLAUDE.md`: `main.py` cannot be
imported). Record `HEAD~1` and `HEAD` for both models.

**There is no regeneration to do** (the user, 2026-09-16; board standing note 14). This is the
build phase of a science code and there is no stored data to curate, so "invalidates eight object
types" is a statement about lookup keys and not a cost anyone pays. Do not attempt a regeneration,
do not add a migration, and **do not hedge the change in the log as though a cost were outstanding**
— the digest moves, that is the acceptance, and nothing further follows from it.

## 4. The tests

### 4.1 `test_the_three_closed_form_sites_agree` must be reworked, not deleted

Prompt 03 shipped it to make D2 option (i) safe. That option was not taken, and after §2.5
`main.py` has no closed-form site — so the test's premise is gone. **Deleting it would lose a real
guard**: that base `LambdaCDM`'s closed form still equals the root, which is now a *correctness*
claim about `LambdaCDM.z_matter_radiation_equality` and not merely a consistency one.

Rework it into that: the remaining closed-form sites — `LambdaCDM`'s property and
`LambdaCDM_GenericEOS`'s two initial guesses — still agree with each other and still sit within the
module's measured budgets of the bracketed reference. `main_py_closed_form` and its entry in
`CLOSED_FORM_SITES` go, and the comment block above them, which names `main.py:553`/`:555`, is
rewritten to say that `main.py` no longer computes the quantity and where it gets it instead.
**Do not widen `LAMBDA_CLOSED_FORM_ULP` or `MATTER_RADIATION_CLOSED_FORM_ULP`** (board standing
note 7).

### 4.2 New assertions

1. **The property is what reaches `feature_z`.** On `QCD_Cosmology`, `feature_z[0]` is
   `cosmology.z_matter_radiation_equality` **as the same float**, and is *not* the closed form.
   This is the test that fails on `HEAD~1` (README §2 (e)) — show it failing and quote the output.
2. **No silent fallback.** A cosmology that declares break points and cannot supply an equality
   redshift **raises**, and the message names it. Use a stand-in in the test tree; do not add a
   broken cosmology to the production tree.
3. **`LambdaCDM`'s property is the closed form and is exact**, per §4.1's rework.

### 4.3 `test_source_grid.py:315-319` is now wrong in its provenance

It pins `feature_z[0]` to `3406.668974249948` at `places=6` — which still *passes*, since the move
is 3.2e-12 — and comments that *"the closed form agrees with its own root solve far inside a grid
interval"*. Re-point both: the literal becomes the solve's value and the comment says the model
supplies it. **A test that still passes for the wrong reason is exactly what README §2 (e) is
about**; fix it deliberately rather than leaving it green.

## 5. Acceptance

| Check | Threshold |
|---|---|
| `QCD_Cosmology` grid digest | `a2c32f67` → **`4849552b`**, 1,996 samples |
| `LambdaCDM(Planck2018)` grid digest | **`60a3205a`**, 1,778 — **unchanged** |
| Samples that moved | **exactly one** |
| `grep -n "omega_r\|omega_cc" main.py` inside `cosmology_feature_redshifts` | **nothing** |
| A fallback to the closed form anywhere in `main.py` | **none** |
| `_find_rho_equality` body, bracket, tolerances | **character-identical** |
| `:502`/`:507` initial guesses | **character-identical** |
| §4.2 test 1 | shown **failing on `HEAD~1`**, output in the log |
| `CosmologyModels` suite | rises, OK |
| `ComputeTargets` suite | **447**, OK |
| `T_Z_REPRESENTATION_VERSION` | **6** |
| `black --check` | clean |

## 6. Stop conditions

- **The QCD digest is anything other than `4849552b`.** Prompt 03 and the orchestrator both
  measured that value independently. A different one means more than the intended sample moved.
- **`LambdaCDM`'s digest moves**, or its sample count changes. Its closed form is exact and nothing
  here should reach it.
- **More than one sample moved.**
- **You are writing a fallback to the closed form**, in any shape, anywhere.
- **You find you must change `_find_rho_equality`, its tolerances, or the initial guesses** to make
  something pass.
- **A suite count falls.**

## 7. Deliverables

1. The two properties on `BaseCosmology`, implemented in `LambdaCDM` and `LambdaCDM_GenericEOS`.
2. `cosmology_feature_redshifts` reading them, computing nothing, with no fallback, and its
   docstring rewritten.
3. §4's reworked guard, three new assertions, and the re-pointed `test_source_grid.py` literal.
4. `logs/09-make-the-model-authoritative.md` per README §5.1, with the before/after digest table
   for both models, the single moved sample identified, and §4.2 test 1's `HEAD~1` failure quoted.
5. Board row 09, item row (l), and
   `[00-equality-redshift-closed-form-is-duplicated-three-times]` moved to §4 —
   `docs/OPEN_ISSUES.md` updated in the same commit, count corrected.
6. One commit, README §5 rule 2.
