# Orchestrator prompt — workstream A, the representation (prompts 01–06)

You are orchestrating workstream A of the QCD background campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `qcd-background-audit`, at or after `e8f746d`
— record the base you find). **You do not write code.** You dispatch one fresh-context subagent per
prompt, review what it produced against fixed criteria, and either continue or stop and report.

Six prompts. Three of them (01, 02, 03) move **no number at all**; three of them (04, 05, 06) move
every QCD number in the tree. The review is therefore in two modes, and confusing them is the way
this workstream goes wrong:

- **For 01, 02, 03 the review is "nothing moved".** Bit-identity is the criterion, everywhere.
- **For 04, 05, 06 the review is "only what was supposed to moved, and it moved by the predicted
  amount".** The audit §4 table is the prediction; it is quoted in campaign README §6.1, and each
  prompt owns exactly one of its three columns — **04 the p90, 05 the median, 06 the max**. A
  prompt that improves a column it does not own has done something other than what it was asked.

## What to read

`../README.md` — all of it, and **§0.2, §2 (all nine design facts), §6 and §7** twice.
`../IMPLEMENTATION_STATE.md` — the board and §5. `docs/qcd-background-audit-2026-09.md` §§0–6.
`orchestrator/README.md` — the rules and the dispatch template. `CLAUDE.md`.

Read `01-background-reference-harness.md` only when about to dispatch it; likewise each of the
others.

## Preconditions

`git status` clean; board rows ⬜; `HEAD` recorded. **Take all three baselines before dispatching
anything:**

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -3

PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py \
  > /tmp/qcdbg-audit-baseline.txt 2>&1                      # ~1 s

PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py \
  > /tmp/qcdbg-verify-baseline.txt 2>&1                     # ~55 s
```

Note that `LiouvilleGreen/tests` contains `test_3bessel_analytic`, which is slow
(`[08-3bessel-plot-cost-dominates-the-suite]`, ~21 min for one test). Record what you ran; if you
skip that module, **say so to the user and use the same exclusion at every later check**, never a
different one.

Nothing in this workstream should move `verify_production_path.py`'s **LambdaCDM** rows at any
point. Prompt 09 is where the QCD rows are formally re-measured, but you should diff against this
baseline after **every** prompt anyway — an unexplained LambdaCDM movement caught at prompt 04 is a
different problem from one discovered at prompt 09.

## Reviewing prompt 01 — the harness

1. One new commit; message per README §5 rule 2; log present and following §5.1.
2. **No production file in the diff.** `git diff HEAD~1 --stat` must show only
   `CosmologyModels/tests/`, the campaign's log and board, and `docs/OPEN_ISSUES.md`.
3. **The reference is independent.** Read `CosmologyModels/tests/T_z_reference.py` yourself and
   confirm it never calls `_solve_T_z`, `_build_T_z_spline`, `T_photon`, `_T_z_spline` or
   `integration_break_points` in the reference path (campaign README §2 (a)). This is the one thing
   the prompt exists to get right, and a reference that touches the thing it measures makes every
   later prompt's acceptance meaningless. **Grep for it.**
4. **Seven tests, and they pass.** Counts: `CosmologyModels` rises by seven; the other two
   unchanged.
5. **The numbers match the audit.** Every threshold's measured value is quoted in the log next to
   the audit's figure. Both were taken on trees where no production file differs, so they should
   agree to the digit. A disagreement is worth stopping for.
6. **Test 7 did what it claims.** The bisected and root-found segment edges must **disagree**, or
   the root-finder must fail. If the log says they agreed, **stop**: audit §2's central claim would
   not hold on this tree and prompt 06's whole design rests on it.
7. `/tmp/qcdbg-audit-baseline.txt` still reproduces exactly (nothing changed).

## Reviewing prompt 02 — the regenerable fixture

1–2 as above; allowed files are `docs/qcd-background-audit/` plus log and board.
3. **`--dry-run` reports no change, and the JSON is untouched.** Verify by hand:
   ```bash
   PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py --dry-run
   git status --porcelain ComputeTargets/tests/wkb_reference_data.json   # must be empty
   git diff HEAD~1 --stat -- ComputeTargets/tests/wkb_reference_data.json  # must be empty
   ```
   If the generator does not reproduce the shipped file, **stop** — that means the JSON was not
   produced by the script it names, which is a finding and not something to work around.
4. **`REFERENCE-FIXTURE.md` is exhaustive.** Spot-check it: pick three QCD assertions yourself with
   `grep` and confirm each appears in the map with its tolerance and its "what would move it"
   classification. The next four prompts consume this document; a map that misses a module means a
   prompt will discover a failing test it was not expecting and will be tempted to loosen it.
5. `docs/gktk-remedial/generate_references.py` is **not** in the diff.

## Reviewing prompt 03 — the key

1–2 as above.
3. **Nothing moved.** The log must quote a dense bit-identity comparison of `T_photon`, `Hubble`
   and `rho` against `HEAD~1` on all three model kinds. Re-run the audit script and diff against
   your baseline: **it must be identical but for timings**.
4. **The filter reads the constant.** Open `Datastore/SQL/ObjectFactories/QCD_Cosmology.py` and
   confirm `build()`'s filter uses `T_Z_REPRESENTATION_VERSION`, not a literal. A literal would
   stop tracking the constant silently at prompt 04, which is precisely the failure prompt 03
   exists to prevent, and the prompt requires an `ast` test for it.
5. **The stale-datastore path is stated, not invented.** The log must say what a pre-campaign
   datastore actually does — which error, from which call, with which message. If the agent
   invented a migration, a version table or a schema upgrade, **stop**.
6. **`LambdaCDM.py`'s factory is not in the diff.**
7. If the agent reports that it preferred README §7 D1 option (ii), **stop and relay to the user**;
   that decision is reserved.

## Reviewing prompts 04, 05, 06 — the three that move numbers

Apply all of the following to each.

1. One commit, log, board, `docs/OPEN_ISSUES.md` — all in the same commit, count and date correct.
2. **`T_Z_REPRESENTATION_VERSION` was bumped**, and the campaign board's version table has the row.
   A numeric change without a bump is a **stop**: it is exactly the silent staleness prompt 03
   landed to prevent.
3. **The prompt moved its own column and not the others.** Against README §6.1:
   - **04** — p90 to ≤2.0e-07; max **essentially unchanged** at ~7.26e-04; node error to ≤1e-14.
   - **05** — median to ≤3.0e-10; max **essentially unchanged**; cost per call not worse.
   - **06** — max to ≤1e-10, p90 ≤1e-14, median ≤1e-15, **and the $\int\mathrm{d}z/H$ guard to
     ≤1e-15**.
   A prompt whose max improves before prompt 06 has probably segmented early; ask how, and if it
   did, **stop** — the audit's separability is the campaign's only defence against a wrong
   representation.
4. **The failure-on-`HEAD~1` check, run by you.** Each prompt names the files to check out. Do it
   yourself:
   ```bash
   git show HEAD --stat
   git checkout HEAD~1 -- CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -5
   git checkout HEAD -- CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py
   git status   # must be clean
   ```
   The middle command **must fail**, naming the test the prompt says it should. If it passes, the
   threshold was not tightened and **the prompt has measured nothing** — stop.
5. **The fixture was regenerated by the tool, not by hand.** `git diff HEAD~1 --
   ComputeTargets/tests/wkb_reference_data.json` should show the QCD block's numbers and its
   provenance strings, **and nothing in the `RadiationModel` or `LambdaCDMModel` blocks**. A
   changed number in either of those is a **stop**.
6. **Tolerances tightened or held.** For every test whose tolerance moved, the log gives old, new
   and achieved. **At most one loosened tolerance**, argued; two is a stop (prompt 04 §2 item 5).
7. **Cost.** `T_photon` per call ≤ 2.5 µs (README §2 (c) — the audit measures the new
   representation *cheaper*); `_build_T_z_spline` ≤ 100 ms. A reported runtime regression means
   something other than the measured design was built: **stop**.
8. **Bit-identity where it is owed.** LambdaCDM, `RadiationModel`, every stand-in, and — from
   prompt 05 — a constant-$g_s$ equation of state, which should become **exact**. Diff
   `verify_production_path.py` against your baseline and confirm every LambdaCDM row is unchanged.
9. `git diff HEAD~1 --stat`: allowed files only. **No `ComputeTargets/` compute target, no
   `Quadrature/`, no `Datastore/` beyond what prompt 03 left, no `main.py`, no `QCD_EOS.py`.**
   `QCD_EOS.py` in a diff is always a stop (README §0.5).
10. **Prompt 06 only:** the log must give the segment edges to 17 digits, say how they were located,
    and report the edge-misplacement test. If the log does not make clear that no bracketing solver
    was applied to $T(z)-T_{\rm break}$ anywhere, **stop** — that is the audit's single named trap
    and the one the campaign is most likely to ship.

## Continue or stop

Continue when the log is `COMPLETE`-class with only `IMPLEMENTATION CHOICE` deviations and every
check above passes. Stop on any campaign README §4 condition, on a moved number that should have
been bit-identical, on a missing version bump, on a "failure on `HEAD~1`" check that does not fail,
on a runtime regression, or on prompt 06 not demonstrating the edge-misplacement trap.

**Relay every subagent question verbatim. Do not answer it yourself.**

## Completion criterion

Rows 01–06 ✅/⚠️; six logs; three suites passing with no count fallen; `T_Z_REPRESENTATION_VERSION`
at 4; the audit script's §0, §3 and §4 tables reading README §6.1's and §6.2's target columns.

Report to the user, in this order:

1. **T1, before and after**: the relative error in $\int\mathrm{d}z/H$ (3.461e-08 → ?) and the
   equivalent phase at $k=10^5/10^7/3\times10^8$ against the 3.05e-7 / 3.05e-5 / 9.15e-4 rad floors.
2. **The three columns**, each with the prompt that moved it.
3. **Cost**, per call and per build.
4. **What a pre-campaign datastore now does**, from prompt 03's log — the user must be *told* this,
   not asked.
5. **The new `BREAK_POINT_ALL` count**, which is workstream B's starting point.
6. Anything that missed, with its issue.
