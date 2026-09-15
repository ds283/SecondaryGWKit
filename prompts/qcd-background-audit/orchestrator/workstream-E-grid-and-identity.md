# Orchestrator prompt — workstream E, the last smooth interpolant and the grid's identity (prompts 13–15)

You are orchestrating workstream E of the QCD background campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `qcd-background-audit`). **You do not write
code.** You dispatch one fresh-context subagent per prompt, review what it produced against fixed
criteria, and either continue or stop and report.

Workstreams A–D are complete: prompts 01–12 landed, the campaign was closed at 12 / 12, and **this
workstream reopens it**. Three prompts, added after the close on the user's decision:

- **13** repairs the last place in the tree where a smooth interpolant still runs across a point the
  cosmology declares non-smooth. It is the reason prompt 11's fix does not currently reach
  production.
- **14** gives a run a **name** at write time and a **derived check** at read time, and keys
  `BackgroundModel` on the grid that built it.
- **15** takes prompt 12's measured density recommendation at the cap the user chose.

They are **strictly sequential** and every arrow is real: 13 before 15 so the density change is
measured against a background that no longer rings; **14 before 15 for the reason prompt 03 came
before prompt 04 — the key must exist before the thing it keys moves.**

## What to read

`../README.md` §0.2, §2 (g), §4, §5. `../IMPLEMENTATION_STATE.md` — the board, its §3 and §5.
`docs/qcd-background-verification.md` **§9.2, §10.0 and §10.4** — §10.4 is prompt 13's whole
justification and §10.0 carries the ceiling every claim in 15 inherits.
`orchestrator/README.md` — the rules and the dispatch template. `CLAUDE.md`.

Read `13-…`, `14-…`, `15-…` each only when about to dispatch it.

## Preconditions

`git status` clean; workstream D complete; `HEAD` recorded. Two decisions the user has already
taken, which you carry and do not re-open:

- **README §7 D5 is settled: keep both `BREAK_POINT_KIND` values as they are.** Prompt 13 §5 records
  it on the board. No prompt here may change either value.
- **Regeneration cost is not a constraint.** The user has stated that everything is development,
  there is no production data to curate, and a superseded datastore retains archival value. **Do not
  accept a cheaper grid, a coarser cap or a skipped regeneration as a virtue**, and stop any agent
  that argues from regeneration cost.

**Baselines, at workstream D's final commit:**

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -3

PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py \
  > /tmp/qcdbg-audit-preE.txt 2>&1
PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py \
  > /tmp/qcdbg-verify-preE.txt 2>&1
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/grid_density_criterion.py \
  > /tmp/qcdbg-density-preE.txt 2>&1
```

`test_3bessel_analytic` is ~20 minutes of the LiouvilleGreen suite and touches no cosmology; if you
exclude it, **say so and use the same exclusion at every later check**, never a different one.

**A standing warning this campaign learned the hard way.** Wall-clock numbers on this machine have
been worthless for hours at a time under external load — it has peaked above load average 140.
**Prefer integrand or RHS evaluation counts to seconds in every review**, require any wall-time row
to be paired with a same-process control and labelled with the load, and re-measure a cost miss
yourself in a quiet window before reporting it as real.

## Reviewing prompt 13 — the derivative splines

1. One commit; log per §5.1; board; `docs/OPEN_ISSUES.md` — same commit, count and date correct.
2. **`T_Z_REPRESENTATION_VERSION` bumped to 6**, with its table row.
3. **Both spline sites, or a measurement saying why not.** `ComputeTargets/BackgroundModel.py`'s
   `_build_derivative` (`:367`) and `_create_functions._build_func` (`:592`). A log that segments one
   and asserts the other is fine without a number is **not** acceptable — the prompt says so.
4. **The ringing came down, and the control did not move.** `T_LO` 2.04e-02 → ≤1e-08 and
   `T_120_MEV` 1.03e-03 → ≤1e-08, with **`EOS_T_LO` still ~1.6e-09**. That control is the whole
   evidence that this is a spline ringing at a step rather than an error of the cosmology; a log
   that omits it has not established its own claim. **If the control got worse, stop.**
5. **The four-way table, in the production configuration.** This is the trap prompt 11 fell into and
   the single most likely way this prompt ships a non-fix. The row that counts is **background built
   on the shipped grid × samples from the shipped grid**, and it must come from **2.4859e-05 rad**
   to at or below **3.9744e-07**. A table scored only in prompt 11's harness configuration —
   background on the *base* grid — **looks like success and means nothing: stop.**
6. **The failure-on-`HEAD~1` check, run by you.** Check out the previous
   `ComputeTargets/BackgroundModel.py`, run the `ComputeTargets` suite, and confirm it **fails**,
   naming the ringing test. Restore, confirm `git status` clean. A test that passes both before and
   after has measured nothing (README §0.2).
7. **Bit-identity where it is owed.** LambdaCDM, `RadiationModel` and every stand-in declare no break
   points, build one segment, and must be **byte-identical**. Diff `verify_production_path.py`
   against `/tmp/qcdbg-verify-preE.txt` and confirm every LambdaCDM row is unchanged.
8. **Fixture by the tool, not by hand.** `git diff HEAD~1 -- ComputeTargets/tests/wkb_reference_data.json`
   shows the QCD block only; a changed number in `RadiationModel` or `LambdaCDMModel` is a stop.
9. `git diff HEAD~1 --stat`: allowed files only. **No `CosmologyModels/`, no `CosmologyConcepts/`,
   no `main.py`, no `Quadrature/`, no `extract_*.py`.**

## Reviewing prompt 14 — the run label and the grid key

1. One commit; log; board; `docs/OPEN_ISSUES.md`; `[11-background-model-not-keyed-on-the-source-grid]`
   moved to §4; the archival-library issue **opened, not built**.
2. **The grid did not move.** QCD 1,773 and LambdaCDM 1,732, element for element, same digests.
   This prompt records *which* algorithm built a grid; changing one is prompt 15. **A moved grid
   here is a stop.**
3. **`T_Z_REPRESENTATION_VERSION` unchanged at 6.** This prompt moves no background number.
4. **The filter reads the constants, not literals** — open
   `Datastore/SQL/ObjectFactories/BackgroundModel.py` and check `build()` yourself, and confirm the
   `ast` test exists. A literal is the exact failure prompt 03 existed to prevent.
5. **The check is real, not decorative.** The acceptance that matters is *a label spanning two
   configurations is refused, naming both*. Verify that test exists and passes. **A label with no
   verification is the design README §7 D1 rejected** — if the agent shipped selection without
   consistency checking, stop.
6. **The archival requirement holds.** A store with no label still reads, labelled unknown; a store
   with exactly one run needs no label and says which it used. **A design that makes old stores
   unreadable fails**, however clean, and is a stop — it is the user's stated requirement.
7. **The two authorised scope breaks are recorded as authorised**, not presented as drift: the
   `extract_*.py` exclusion lifted for this prompt only, and `main.py`'s new argument. Both are in
   the prompt; the log must name them.
8. **What `source_samples_per_log10z` feeds** is answered in the log as a *finding*, not assumed.
   Prompt 15 retires its tag and must not be surprised.
9. **What was run versus asserted.** The extract scripts need a datastore. A log claiming a script
   works without running it is a stop; a log saying plainly which were executed and which were
   `ast`-asserted is fine. **Do not let "asserted by `ast`" be reported as "verified".**
10. `git diff HEAD~1 --stat`: allowed files only. **No `CosmologyModels/`, no compute target, no
    `Quadrature/`.** `main.py` hunks must be argument parsing, grid construction and tagging — **not
    the Bessel, WKB or numeric stages.**

## Reviewing prompt 15 — the density

1. One commit; log; board; `docs/OPEN_ISSUES.md`;
   `[12-source-grid-density-is-uniform-over-a-curvature-that-spans-eight-orders]` moved to §4.
2. **The construction version bumped to 2**, with its row. A grid change without a bump is the
   silent staleness prompt 14 landed to prevent.
3. **The cap held: nothing is coarser than today, anywhere.** There is a test asserting it directly
   and it is the user's decision expressed as code. **If it was weakened, relaxed, or the 1.75×
   saving was taken, stop** — the saving was measured, offered and *declined*, and the log must show
   it declined rather than missed.
4. **`[03-derivative-pad-clamp-on-coarse-grids]` still does not bind** — the lowest grid interval
   below $-\log(0.9)/12 = 8.7800\times10^{-3}$ in $u$. This is the measured consequence of the cap.
5. **The rows that missed now pass**, scored in the **production configuration** — background
   rebuilt on the new grid, samples from the new grid. Same trap as prompt 13 item 5. And **show it
   fails on `HEAD~1`**, run by you.
6. **LambdaCDM's grid changes here, deliberately, and it is the first time in this campaign.** Every
   prior prompt held LambdaCDM bit-identical and prompt 11's test asserts it. That test must be
   **updated, not deleted**, and the change tagged `STRUCTURALLY REQUIRED` with an argument. **If it
   was deleted, or quietly relaxed, stop.** What must still hold: the construction consults no
   equation-of-state module, and a cosmology with no break points gets no protected points.
7. **The response grid is still a subset**, and its size is quoted. Prompt 12 measured 147 at this
   cap against today's 148; the ladder's coarse end reaches 44, which it called "almost certainly not
   usable". You are at the safe end — confirm the number says so.
8. **`SourceSamplesPerLog10ZTag` retired**, in this commit, because this is the commit that makes it
   false.
9. **The ceiling restated where the result is**, not in a footnote: no pipeline has ever been run on
   any grid in this line of work, **including the one that ships**, and no verification run reached
   production $x$ (`docs/OPEN_ISSUES.md` §5). This prompt changes the grid on a Gauss-quadrature
   oracle. **A log that omits this is a stop** — it is the single most over-claimable result in the
   workstream.
10. `git diff HEAD~1 --stat`: allowed files only. **No `extract_*.py`** — prompt 14's exception was
    prompt 14's. No `CosmologyModels/`, no `ComputeTargets/BackgroundModel.py`.

## Continue or stop

Continue when the log is `COMPLETE`-class with only `IMPLEMENTATION CHOICE` deviations and every
check above passes. **Stop** on any campaign README §4 condition; on a four-way table scored in the
wrong configuration; on a "failure on `HEAD~1`" check that does not fail; on a moved LambdaCDM number
in 13 or 14; on a weakened cap in 15; on a label mechanism without its verification; on an old store
made unreadable; or on any argument from regeneration cost.

**Relay every subagent question verbatim. Do not answer it yourself.**

## Completion criterion

Rows 13–15 ✅/⚠️; three logs; three suites passing with no count fallen;
`T_Z_REPRESENTATION_VERSION` at 6; the source-grid construction version at 2; the campaign's board
closed again, at 15 / 15.

Report to the user, in this order:

1. **The ringing, before and after**, all three crossings with `EOS_T_LO` as the control — and the
   four-way table's production row, which is what says prompt 11's fix now reaches production.
2. **What a run is now called**, what a pre-prompt-14 store does on the compute path and on the read
   path, and which extract scripts were actually executed.
3. **The grid**: samples per model, the cap, the response-grid size, and the confirmation that
   nothing anywhere got coarser.
4. **What it cost**, in evaluation counts.
5. **What must be regenerated**, from log 11 §5's table — and that the user has said this is not a
   constraint, so it is reported as scope, not as a caution.
6. **The ceiling**, restated: no pipeline run on any grid, production $x$ never reached.
7. Anything that missed, with its issue.
