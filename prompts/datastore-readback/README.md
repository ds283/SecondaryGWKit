# Campaign — datastore read-back

## 0. Why this campaign exists

On 2026-09-21 a scoped pipeline run was resumed against an existing datastore
(`var/datastores/handover-A3-baseline-lambdacdm.sqlite`, 10 h 33 m of work, written by the same
tree). Every earlier stage replayed correctly from stored results in about ninety seconds. The run
then crashed at `CALCULATE QUADRATIC SOURCE INTEGRALS`, reading back a row it had written itself:

```
sqlalchemy.exc.NoSuchColumnError: Could not locate column in row for column 'numeric_quad'
```

`Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`'s `SELECT` at `:234` omits
`table.c.numeric_quad`; `build()` at `:324` reads `row_data.numeric_quad`. The column exists in the
table and the data is there. The sibling `SELECT` at `:530` includes it, which is why nothing else
trips over this.

**It has never fired because the path has never been taken.** A fresh run *computes* these rows and
never reads them back; only a resume, or a consumer doing an `object_get` on a stored row, takes
this path. So:

- no pipeline run can currently be resumed past the quadratic-source-integral stage;
- the ~7,552 `QuadSourceIntegral` rows already written cannot be read back at all.

**0.1 What this campaign is really about.** The one-line fix is not the point. The point is that a
whole class of defect — `build()` reading a column its own `SELECT` does not request — is invisible
to every test in the tree, and that 12 of the 22 object factories read `row_data` attributes and so
could carry it. A fix without a guard leaves the next one to be found the same way: by losing a
day of compute.

**0.2 Correctness is the only objective.** Sequence by epistemic dependency, never by urgency or by
what is cheap. That a fix looks like one line is not a reason to skip its test.

## 1. Scope

**In scope:** `Datastore/SQL/ObjectFactories/`, and a new `Datastore/tests/`.

**Out of scope:** the physics. No `ComputeTargets/` production file, no `main.py`, no `config/`, no
schema change, no migration, and nothing under `docs/spec/`. The stored data is correct; only the
reading of it is wrong.

## 2. Prompts

| # | Prompt | Covers |
|---|---|---|
| 01 | [`01-quadsourceintegral-readback.md`](01-quadsourceintegral-readback.md) | Fix the omitted column; add the guard that would have caught it; audit the other 11 factories and **open** what it finds without fixing |

## 3. Datastore

The failing datastore is `var/datastores/handover-A3-baseline-lambdacdm.sqlite` (4 shards, 385 MB,
`var/` is gitignored). **A backup is retained** at
`var/datastores/backup-pre-resume-20260921T091011` and must not be deleted by any prompt in this
campaign. Row counts are identical between the two; the failed resume damaged nothing.

Its manifest, `handover-A3-baseline-lambdacdm.manifest.json`, records the run history, the defect
and the restart command. **Read it before touching the store.**

## 4. Baselines

Measured at `ab7079c` on `handover-remedial`: `ComputeTargets` **552**, `CosmologyModels` **39**,
`LiouvilleGreen` **148** (`skipped=1`), all OK. `AdaptiveLevin/tests` also exists and was **not**
baselined during the hand-over campaign — baseline it before dispatching.

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . 2>&1 | tail -40
```

The suites print model banners on stdout, so `| tail -5` will not show the verdict. **Do not set
`THREE_BESSEL_DIAGNOSTIC_PLOTS`.** `ComputeTargets.tests.test_tk_wkb_phase.TestCost.test_wall_time_per_object`
is a known wall-clock flake; re-run that module alone before attributing anything to a change.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, unchanged, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit**, plus `docs/OPEN_ISSUES.md`.
4. **Do not fix things the prompt did not ask for.** Record them and open a §3 issue. This campaign
   is *specifically* structured so that the audit opens issues rather than fixing them: a sweep that
   silently repairs eleven factories is unreviewable.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**

### 5.1 The log template

Subject, commit, result; **What shipped**; **Deviations from the prompt** (each classified);
**Verification performed**; **Observations not acted on**; **State handed to the next prompt**.
