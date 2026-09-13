# Orchestrator prompt — workstream D, the source grid and the consumer spline (prompts 10–12)

You are orchestrating workstream D of the QCD background campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `qcd-background-audit`). **You do not write
code.**

> **This workstream is gated.** Campaign README §7 **D7**: the audit itself calls these items "a
> second campaign that depends on 3". **Do not start it unless the user has said to**, and confirm
> which of the three prompts they want. Prompt 10 is small and closes an inherited issue; prompts
> 11 and 12 redesign the production sample grid, which invalidates every stored object keyed on it.
> If you were started without an explicit go-ahead, **stop and ask which prompts to run.**

Three prompts of quite different character:

- **10** closes `[13-consumer-spline-crosses-eos-break-points]`, the issue
  `prompts/phase-representation` prompt 02 stopped on. Its most likely correct outcome is **a
  measurement and no code change**, and the review must not treat that as a failure.
- **11** changes the production grid and a datastore **tag**.
- **12** changes nothing and **stops for the user by design**.

## What to read

`../README.md` §0.3, §0.5, §7 D7. `../IMPLEMENTATION_STATE.md`.
`prompts/phase-representation/02-primitive-phase-break-point-knots.md` and **its log in full** —
the record of what a knot vector does and does not buy.
`prompts/phase-representation/README.md` §2 (d) — **splitting at declared break points is not
chunking**. `docs/qcd-background-audit-2026-09.md` §7 and §9. `docs/qcd-background-verification.md`
as prompt 09 wrote it. `orchestrator/README.md`. `CLAUDE.md`.

## Preconditions

Workstreams A, B and C complete; the board reading CLOSED at 9 / 12; `git status` clean; the user's
go-ahead recorded. Baselines at workstream C's final commit:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py \
  > /tmp/qcdbg-verify-preD.txt 2>&1
```

## Reviewing prompt 10 — `PrimitivePhase`

1. One commit; log per §5.1; board; `[13-consumer-spline-crosses-eos-break-points]` moved to the
   `GkTk-remedial` board's §4; `docs/OPEN_ISSUES.md` in the same commit.
2. **The re-measurement came first.** The log must open with prompt 02's §4 measurements re-taken on
   the corrected background: how much of the 1.907e-06 / 3.186e-06 rad survives, and how much of
   that is the break points. **If the error is already at the floor and the prompt changed no code,
   that is `COMPLETE` and the issue closes on a measurement** — a perfectly good outcome, and the
   one the campaign's own evidence makes likely. Do not send it back for a code change.
3. **Scored at all three wavenumbers**, both sectors. A table scored at $k=10^5$ alone is a stop:
   `[02-consumer-phi-below-the-storage-granularity]` means a scheme that looks good there can be a
   3× and 10× regression at $10^7$ and $3\times10^8$, and prompt 02's log warns about exactly this.
4. **The `theta_deriv` residue is split and attributed**, not reported as one number — the break
   points' share against `[02-consumer-phi-below-the-storage-granularity]`'s. A residue attributed
   to the latter is a **narrowing, reported**; a missed row with the Result still `COMPLETE` is a
   stop.
5. **If code changed: which construction, and is it chunking?** The repeated-knot vector is now the
   default because prompt 07 removed the blocker. If the agent took per-segment splines instead, the
   log must give `phase-representation` README §2 (d)'s argument — declared boundaries not an
   arbitrary `logstep`, a bounded residual not the growing phase, no rebased ordinates, no switch
   discontinuity — with a **test** for the last, not an assertion. **If the log does not distinguish
   the two, stop**: that is the one confusion this line of work is most likely to ship.
6. **`num_chunks` still returns 1:**
   ```bash
   grep -n "num_chunks" ComputeTargets/primitive_phase.py
   ```
   Anything else is a changed stored value (`WKB_phase_spline_chunks`) and a stop.
7. **Cost** ≤ 2× the 0.0010 s / 468 evaluations baseline, both models.
8. **Smooth cosmologies bit-identical**; LambdaCDM rows in the verify diff unchanged.
9. `git diff HEAD~1 --stat`: **no `CosmologyModels/`** — the break points are correct and prompt 07
   owns them. No `Quadrature/`, no `LiouvilleGreen/`, no producer, no `Datastore/`, no `main.py`.

## Reviewing prompt 11 — the grid

1. One commit; log; board; `docs/OPEN_ISSUES.md`.
2. **A cosmology declaring nothing gives the grid it gives today**, element for element, for
   LambdaCDM at production parameters. This is what keeps the change inert where it should be;
   check the test exists and passes.
3. **No equation-of-state import in `CosmologyConcepts/`.** The duck-typed pattern of
   `_cosmology_break_points` is the precedent; grep the diff.
4. **The standoff was argued, not borrowed.** `BREAK_POINT_STANDOFF = 1e-12` exists for a different
   purpose; a log that reuses the number without arguing it for grid placement has not made the
   choice the prompt asked for.
5. **The response grid is still a subset of the source grid**, and protected points survive the
   winnow. Both asserted by test.
6. **The tag change is stated with its blast radius.** This is the review's substantive item:
   changing `SourceRedshiftGrid_{len}` makes every object carrying the old tag unfindable. The log
   must quantify what that invalidates. If it does not, **stop** — the user is about to lose a
   datastore and must be told in numbers.
7. `git diff HEAD~1 --stat`: `CosmologyConcepts/`, the named `main.py` hunks, tests, log, board
   only. **No `CosmologyModels/`, no compute target, no `Datastore/` factory.** Confirm the
   `main.py` hunks are the grid and tag construction and not the Bessel, WKB or numeric stages.

## Reviewing prompt 12 — the criterion

1. One commit; log; board §3 carrying the recommendation so it survives the campaign's close;
   `docs/OPEN_ISSUES.md`.
2. **No production file in the diff.** A production hunk is a stop; this prompt measures.
3. **All three sub-questions answered, or the gap declared** (prompt 12 §1).
4. **The criterion is computable before the grid is built** — from $H$, $c_s^2$ and $k$. One that
   needs $\varphi$ to exist already cannot be used by `populate_z_sample` and is not an answer.
5. **The large-$k$ floor is respected.** A criterion derived at $k=3\times10^8$, where $\varphi$
   spans 2.0 ulp, is measuring rounding. The log must say where the criterion stops meaning
   anything.
6. **The production-$x$ ceiling is restated** where the recommendation is (`docs/OPEN_ISSUES.md`
   §5), not in a footnote.
7. **"The current grid is adequate" is an acceptable answer**, if that is what was measured, and
   must be stated as plainly as any other.
8. The prompt **stops for the user**. Do not dispatch an implementation follow-up; the remaining
   scope is the user's to set.

## Continue or stop

Continue between prompts when the log is `COMPLETE`-class and every check passes. Stop on any
campaign README §4 condition, on a grid tag change without a quantified blast radius, on a
single-wavenumber score in prompt 10, or on a production hunk in 12.

**Relay every subagent question verbatim.**

## Completion criterion and report

Rows 10–12 at their final states; three logs; suites passing; `docs/qcd-background-verification.md`
carrying a dated section per prompt.

Report: whether `[13-consumer-spline-crosses-eos-break-points]` closed on a code change or on a
measurement, and what survives of its 1.907e-06 / 3.186e-06 rad; what the grid now protects and
what the tag change invalidates; and prompt 12's recommendation with its cost — which is a decision
for the user and not a task for another agent.
