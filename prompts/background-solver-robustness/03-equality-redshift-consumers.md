# Prompt 03 — What the equality redshifts actually feed

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** README §2 **(f)** · **Reports on:** README §7 **D2**
**Closes:** `[00-main-py-equality-agreement-figure-is-stale]`
**Narrows:** `[00-equality-redshift-closed-form-is-duplicated-three-times]` — it does **not** close
it; the decision is the user's.
**Measurements:** [`RECONCILIATION.md`](RECONCILIATION.md) §5, §6, §7
**Depends on:** **02** — the figure you take is the closed form against the *corrected* solve.
**Recommended model:** **Opus** — this is cross-package reconciliation whose consequence is a
datastore identity. An agent that gets the blast radius wrong here either invalidates every stored
`BackgroundModel` or leaves the next reader believing a claim that is about to cost them one.

**Files you may touch:** `main.py` (**one docstring sentence only**, `:525-527`),
`CosmologyModels/tests/test_rho_equality.py` (one new test), plus this campaign's log, board and
`docs/OPEN_ISSUES.md`.

**Do not touch:** any executable line of `main.py` — **not one**; `cosmology_feature_redshifts`'s
body at `:544-553` is out of bounds even though it is the subject of your measurement.
`CosmologyConcepts/wavenumber.py`, `Datastore/`, `LambdaCDM_GenericEOS.py`,
`CosmologyModels/LambdaCDM/LambdaCDM.py`. **No production behaviour changes in this prompt.**

**Read first:** `RECONCILIATION.md` §5, §6 and §7 in full; README §0.3 and §7 D2;
`main.py:506-553` (`cosmology_feature_redshifts`, including its docstring, which is the thing you
are correcting) and `:900-935` (where `feature_z` is consumed);
`CosmologyConcepts/wavenumber.py:280-360` (`build_z_sample`, and `:350` where `feature_z` enters);
`Datastore/SQL/ObjectFactories/BackgroundModel.py:180-280` (the digest columns and the filter);
`CosmologyModels/LambdaCDM/LambdaCDM.py:70-85`; and `prompts/qcd-background-audit/logs/11-*.md` §5,
which prices a digest change.

---

## 1. Why this prompt exists

`AUDIT.md` §2.1 concludes, from a correct repository-wide grep, that

> **The blast radius of this solve is two banner lines.**

That is true of **the solve**. It is false of **the quantity**, and the difference is not academic.
`main.py` recomputes both equality redshifts from the same closed forms and forces them into the
production source grid:

```
main.py:549   feature_z.append(float(omega_m / omega_r - 1.0))
main.py:551   feature_z.append(float(pow(omega_cc / omega_m, 1.0 / 3.0) - 1.0))
main.py:903   break_z, feature_z = cosmology_feature_redshifts(model_cosmology, zend, z_init_grid)
main.py:928       feature_z=feature_z,
     -> CosmologyConcepts/wavenumber.py:350            forced into the grid
     -> Datastore/.../BackgroundModel.py:225           source_grid_digest = z_sample.digest()
     -> Datastore/.../BackgroundModel.py:182,251,273   a lookup-key column, filtered on
```

On a cosmology that declares break points — today, `QCD_Cosmology`, the whole path being gated on
exactly that at `main.py:530-537` — **the two equality redshifts are production sample locations
inside a datastore identity.**

The consequence is that `AUDIT.md` §6 observation 2's suggestion (have the method derive its own
guess and expose the result, so `main.py` can import it) is **not the tidy it looks like**. One ulp
of movement in either redshift moves a grid sample, moves the digest, and invalidates every stored
object of the eight types `qcd-background-audit` log 11 §5 priced. **Nobody has written this down.**
This prompt writes it down and hands the decision to the user.

## 2. What to measure

### 2.1 The three copies, scored against each other

For each of `QCD_Cosmology`, `LambdaCDM(Planck2018)` and `RadiationModel` — or the nearest thing
each has; say which and why if one of the three does not expose all three $\Omega$s — produce a
table with, for both pairs (matter = radiation, matter = $\Lambda$):

| column | what |
|---|---|
| `main.py` closed form | `cosmology_feature_redshifts`'s expression, evaluated as written, at 17 digits |
| `LambdaCDM_GenericEOS` guess | `__init__:501-506`'s expression, at 17 digits |
| `LambdaCDM.py:73-74` | where the model has it, at 17 digits |
| the corrected solve | `_find_rho_equality` at prompt 02's tree |
| independent `brentq` | at `xtol=1e-300, rtol=8.9e-16` |
| ulp spread | across every column present |

**Evaluate the expressions as they are written in each file**, not a normalised form. Whether
`omega_m / omega_r - 1.0` and `self.omega_m / self.omega_r - 1.0` produce the same float is exactly
the question — if the three sites agree bit for bit today, that is a finding worth stating, and if
they do not, that is a bigger one.

### 2.2 The stale figure

`main.py:525-527` says the closed form agrees with the model's own root solve *"to 4e-13 relative
in z on `QCD_Cosmology` at production parameters"*. Measured at `f023eb8`
(`RECONCILIATION.md` §6) it is **−9.34e-16** and **−3.66e-16** — three orders tighter. Re-take it
against **prompt 02's corrected solve** and correct the sentence to what you measure.

**The correction is one sentence in one docstring.** It must quote the two figures, name the commit
they were taken at, and say that the original 4e-13 was a safe over-estimate used only to argue
"far below a grid interval" — so nothing was wrong, only unverified. Do not rewrite the paragraph
around it; prompt 11's reasoning for recomputing rather than importing is still correct and is now
*more* correct, not less.

### 2.3 The grid is byte-identical

Demonstrate that this prompt moved nothing. Build the production source grid for `QCD_Cosmology`
and for `LambdaCDM(Planck2018)` at `HEAD~1` and at your commit and compare the **digest** and the
element count. Use whatever `main.py` uses — `load_main_py_functions`
(`ComputeTargets/tests/test_main_plumbing.py`) is how a test extracts a `main.py` function without
importing the module, and `CLAUDE.md` says `main.py` cannot be imported.

**If the digest differs, stop.** Nothing in this prompt or prompt 02 should be able to move it, and
a difference means prompt 02 changed something it should not have.

### 2.4 One test

Add `test_the_three_closed_form_sites_agree` to `CosmologyModels/tests/test_rho_equality.py`:
assert that the `LambdaCDM_GenericEOS` guess expression and the `main.py` expression, evaluated on
the same model, produce the **same float**, and that both are within 2 ulp of the bracketed
reference. The test does not import `main.py`; it reproduces the expression with a comment naming
`main.py:549` and `:551` as the site it mirrors, and a docstring saying that the point of the test
is that a later edit to either site announces itself. **This is the guard that makes D2 option (i)
— "leave all three and document" — safe to choose.**

## 3. What to write

A section in the log, and a §3 board entry, stating in prose a decision-maker can act on:

1. **The map.** Three sites, what each feeds, which one is load-bearing, and the chain from
   `feature_z` to the lookup key with file and line at each hop.
2. **The measured agreement**, §2.1's table.
3. **The three options**, as README §7 D2 states them, each with its cost:
   - **(i) leave all three, document the duplication as deliberate, and keep §2.4's test as the
     guard.** Cost: zero. Risk: a fourth copy appears one day.
   - **(ii) unify on the closed form** — one helper, imported by all three sites. Safe **only** if
     the resulting float is bit-identical to what `main.py:549`/`:551` produce today, which §2.1
     measures. Cost: a small refactor across three packages, and `main.py` acquires an import from
     a cosmology model.
   - **(iii) unify on the solve** — `main.py` imports the result of `_find_rho_equality`. Moves the
     grid by whatever §2.1's last column says, which is not zero, and therefore costs a full
     regeneration of eight object types.
4. **The campaign's recommendation: (i)**, with §2.4's test as the standing guard. Say why: the
   duplication is documented and deliberate (`main.py:522-528` explains it), the sites agree to the
   floor, and the only benefit of unifying is aesthetic while the downside of getting it wrong is a
   datastore.

**Report; do not decide.** The orchestrator stops and puts this to the user.

## 4. Acceptance

| Check | Threshold |
|---|---|
| §2.1's table | complete, all three models, both pairs, 17 digits |
| `main.py:525-527` | corrected to the measured figures, with the commit named |
| Production source-grid digest, both models, `HEAD~1` vs `HEAD` | **identical** |
| Executable lines of `main.py` in the diff | **zero** |
| `CosmologyModels` suite | +1, OK |
| `ComputeTargets` suite | **447 → 447**, OK |
| README §7 D2 | reported to the user with the three options priced |

## 5. Stop conditions

- **The grid digest moves.** Stop; something upstream is wrong.
- **The three closed-form sites disagree by more than 1 ulp.** That is a bigger finding than this
  prompt is scoped for — report it, do not unify them, and stop.
- **You are tempted to take option (ii) or (iii).** You may not. D2 is the user's.

## 6. Deliverables

1. The corrected sentence in `main.py`'s `cosmology_feature_redshifts` docstring.
2. `test_the_three_closed_form_sites_agree`.
3. `logs/03-equality-redshift-consumers.md` per README §5.1, with §3's four written items and §2's
   tables.
4. Board row 03, item row (f), `[00-main-py-equality-agreement-figure-is-stale]` moved to §4,
   `[00-equality-redshift-closed-form-is-duplicated-three-times]` narrowed with the measurement and
   the recommendation — `docs/OPEN_ISSUES.md` updated in the same commit.
5. One commit, README §5 rule 2.
