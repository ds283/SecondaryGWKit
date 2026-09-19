# Workstream A — the equality solve (prompts 01–02)

**Read [`README.md`](README.md) in this directory first.** The orchestrator rules, the three checks
and the dispatch template are there and are not repeated here.

**Prompts:** 01 (Opus), 02 (Opus). **Both are Opus.** Do not downgrade either: 01's whole content
is a judgement about what to assert, and 02 is the campaign's only production numerics change.

---

## 0. What this workstream is

`_find_rho_equality` runs an unbracketed secant at `xtol=1e-6, rtol=1e-4` and returns the right
answer to 4.0e-16 **because its caller hands it the closed-form root**, not because its tolerances
are adequate. Prompt 01 builds the test that can tell the difference; prompt 02 brackets the solve
so it is right for the right reason.

**The campaign's claim is that the two equality redshifts do not move.** Your job is to be the
person who would notice if they did.

## 1. Baselines — take these before dispatching prompt 01

They cannot be reconstructed afterwards.

```bash
git rev-parse HEAD                                    # record it; every diff below is against it
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .    # expect 30, OK
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .     # expect 447, OK
PYTHONPATH=. ./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py
```

**Keep the whole `measure_rho_equality.py` output.** Its §2.2 table (four roots, four evaluation
counts, four relative errors) and its §3.2 failure boundary (six rows, with exact exception text)
are what you score both prompts against. Note the `PYTHONPATH=.`; without it the script does not
run (`RECONCILIATION.md` §1).

Also record, from `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`, the two banner lines that
`QCD_Cosmology` construction prints:

```
|  matter-radiation equality at z = 3407
|  matter-Lambda equality at z = 0.3034
```

These are the user-visible surface of the whole campaign and they must be character-identical at
every commit.

## 2. Prompt 01 — the characterisation test

Dispatch with the template. Model: **Opus**.

### Review criteria

| # | Check | How |
|---|---|---|
| 1 | **Zero production files in the diff** | `git diff --name-only HEAD~1 HEAD` — only `CosmologyModels/tests/test_rho_equality.py`, the log, the board and `docs/OPEN_ISSUES.md` |
| 2 | The new tests pass **when you run them** | `PYTHONPATH=. ./venv/bin/python -m unittest CosmologyModels.tests.test_rho_equality -v` |
| 3 | `CosmologyModels` rose; `ComputeTargets` is 447 | the three checks |
| 4 | **The assertions are against something independent** | Read the test. If it compares `_find_rho_equality`'s output to a hard-coded literal, or to `init_z`, **that is the trap the prompt names** and it is a stop. It must compare against a `brentq` reference computed in the test |
| 5 | The threshold is ulp-based, not relative | `2 * np.spacing(...)`, per the prompt |
| 6 | The log's **State handed to the next prompt** carries the failure boundary with **exact exception text** | Prompt 02 needs it to prove its own change; a paraphrase is useless |
| 7 | §4 item 3's two figures were measured and the issue opened | `[01-agreement-threshold-comment-predates-the-representation]` in the board §3 and in `docs/OPEN_ISSUES.md` |
| 8 | No production file was "improved" in passing | criterion 1 covers it; read the diff anyway |

### Stop and report if

- Any of the three assertion families fails. `RECONCILIATION.md` §2 says every audit figure
  reproduces at `f023eb8`; a failure means something is different and the campaign's premise needs
  re-examining before prompt 02 changes anything.
- The monotonicity probe is not strict somewhere. Prompt 02's whole design rests on it.
- The agent touched `LambdaCDM_GenericEOS.py`. Even a comment. Prompt 01 has zero production files.
- The new module's runtime is more than a few seconds. It means a `QCD_Cosmology` is being built
  per test rather than per class, and the suite is 0.62 s today.

## 3. Prompt 02 — bracket the solve

Dispatch with the template. Model: **Opus**. **Do not dispatch until prompt 01's review is clean** —
its reference values are 02's acceptance.

### Review criteria

| # | Check | How |
|---|---|---|
| 1 | **The four equality redshifts are bit-identical** | Compare the log's 17-digit values against your §1 baseline and against prompt 01's log. ≤ 1 ulp with both floats quoted is acceptable **with an explanation**; anything larger is a stop |
| 2 | **The banner lines are character-identical** | Construct a `QCD_Cosmology` yourself and diff the two lines against §1 |
| 3 | **The new assertions fail on `HEAD~1`** | **Run this yourself.** `git stash` is not enough — check out the test file from `HEAD` onto `HEAD~1` in a scratch worktree, or `git checkout HEAD~1 -- CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` and run, then restore. This is the most important check in the workstream |
| 4 | Prompt 01's four tests pass **unchanged** | `git diff HEAD~1 HEAD -- CosmologyModels/tests/test_rho_equality.py` must show additions, not edits to existing methods |
| 5 | The diff is confined to `_find_rho_equality` | `git diff HEAD~1 HEAD -- CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` touches nothing outside `:999-1021` plus whatever the bracket helper needs. **`__init__`, `_rho_fluid`, `_solve_T_z` and the two crossing methods must not appear** |
| 6 | The comment meets the `:569-583` standard | Read it. It must say what the tolerances are *for*, what the cost is, and **the audit §2.3 finding** — that the solve used to be accurate because its caller handed it the answer. If that sentence is missing, the most valuable line in the diff is missing |
| 7 | No short-circuit on a small residual | Search the new code for an early return when `match_rho(init_z)` is near zero. That reintroduces the defect explicitly and the prompt forbids it |
| 8 | No `try/except` around `_rho_fluid` | Same reason: it hides the coupling instead of removing it |
| 9 | All four of §2.1's design points are classified as `IMPLEMENTATION CHOICE` with alternatives | The prompt leaves them open deliberately; a log that does not justify them is not reviewable |
| 10 | The error message names the species pair **and** the range searched | Trigger it yourself: call `_find_rho_equality("matter", "radiation", init_z=…)` with a guess displaced −80 % and read what comes out |
| 11 | `ComputeTargets` 447; `T_Z_REPRESENTATION_VERSION` 6 | the three checks |

### Stop and report if

- **Either root moved by more than 1 ulp.** Report both floats at 17 digits. This is the campaign's
  primary stop condition and there is no threshold to adjust.
- **Check 3 does not fail on `HEAD~1`.** Then the new assertions do not distinguish the trees and
  prompt 02 has proved nothing, whatever else it did.
- **`ComputeTargets` moved at all.** Find out why before anything else happens.
- The agent asked to change `__init__` or the signature. Relay it verbatim — that is README §7 D2
  and workstream B's, and the answer is no.

## 4. Completion criterion for workstream A

All of:

- 01 and 02 are ✅ on the board, each with its commit SHA and log.
- The four equality redshifts are bit-identical to the §1 baseline, quoted at 17 digits in log 02.
- Prompt 02's §3 items 1 and 2 were demonstrated failing on `HEAD~1`, **by you**, with the output
  in your report.
- `CosmologyModels` has risen by 7 or so and is OK; `ComputeTargets` is 447.
- `[00-equality-solve-is-unbracketed-and-loose]` is in the board's §4 and out of
  `docs/OPEN_ISSUES.md`, count corrected.

Report to the user: the four redshifts before and after, the evaluation counts before and after,
the `HEAD~1` failure output, and the one sentence prompt 02 put in the comment about audit §2.3.
Then start workstream B.
