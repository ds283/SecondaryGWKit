# Orchestrator prompt — the phase-representation campaign (prompts 01, 02)

You are orchestrating the phase-representation campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`, at or after `9daa2cb` —
record the base you find). **You do not write code.** You dispatch one fresh-context subagent per
prompt, review what it produced against fixed criteria, and either continue or stop and report to
the user.

Two prompts, each closing a defect that `GkTk-remedial` prompt 13 measured and was forbidden to
fix. Both change production code that every stored WKB phase passes through, and both must leave
LambdaCDM bit-identical. So the review here is about **blast radius**: what moved that should not
have.

## What to read

`../README.md` — all of it, and **§2 (all seven design facts)** and **§6** twice.
`../IMPLEMENTATION_STATE.md` — the board and §5.
`docs/gktk-remedial-verification.md` §3.5, §3.6, §3.7 — the measurements both prompts are scored
against. `prompts/GkTk-remedial/IMPLEMENTATION_STATE.md` §3, the two `[13-...]` entries.
`CLAUDE.md`.

Read `01-wkb-mod-2pi-cycle-count.md` only when about to dispatch it; likewise 02.

## Preconditions

`git status` clean; board rows ⬜. Both suites pass and you have recorded the counts:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -3
```

Expect 339 and 141 at `9daa2cb`. **Take a baseline of the production measurement before
dispatching anything** — you will diff both prompts' runs against it:

```bash
PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py > /tmp/verify-baseline.txt 2>&1
```

It takes ~55 s and needs no Ray and no datastore.

## Dispatching a prompt

For prompt NN, launch a subagent with **exactly** this context:

> You are the implementation agent for one prompt in a campaign. Read, in this order:
> `prompts/phase-representation/README.md`,
> `prompts/phase-representation/IMPLEMENTATION_STATE.md`, then your prompt
> `prompts/phase-representation/NN-<name>.md` and the sections of
> `docs/gktk-remedial-verification.md` it cites. Execute the prompt exactly. Do not read the other
> prompt in this campaign. Follow README §5 for the commit, the log, this campaign's board, the
> `GkTk-remedial` board entry your prompt closes, and `docs/OPEN_ISSUES.md`. Other commits may land
> on this branch while you work: make exactly one commit, and do not amend, reset or rebase
> anything you did not create — if you need to change a commit you already made and it is no longer
> `HEAD`, stop and say so rather than rewriting. When you finish, reply with: the commit SHA, the
> **Result** line from your log, the "State handed to the next prompt" section verbatim, and every
> deviation with its classification tag.

Model: **01 → Opus**, **02 → Opus**. Run 01 → 02.

## Reviewing prompt 01

1. One new commit, message per README §5 rule 2.
2. `logs/01-wkb-mod-2pi-cycle-count.md` exists, follows §5.1, classifies every deviation.
3. This campaign's board row and P1 updated; the `GkTk-remedial` §3 entry moved to that board's §4;
   `[10-wrap-theta-loop-at-large-phase]`'s clause corrected; `docs/OPEN_ISSUES.md` in the same
   commit with its count and date correct.
4. **The tests fail on the old code.** This is the check that matters most, because a
   self-consistency test that passes both before and after proves nothing. Verify by hand:
   ```bash
   git stash list; git show HEAD --stat
   git checkout HEAD~1 -- LiouvilleGreen/WKBtools.py LiouvilleGreen/range_reduce_mod_2pi.py
   PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t . 2>&1 | tail -5
   git checkout HEAD -- LiouvilleGreen/WKBtools.py LiouvilleGreen/range_reduce_mod_2pi.py
   ```
   The middle command **must fail**, naming the adversarial case. Restore the tree and confirm
   `git status` is clean afterwards.
5. Both suites pass when **you** run them; neither count falls.
6. **The production measurement.** Re-run the script and diff against your baseline:
   ```bash
   PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py > /tmp/verify-01.txt 2>&1
   diff /tmp/verify-baseline.txt /tmp/verify-01.txt
   ```
   Section 6's LambdaCDM $k=3\times10^8$ row must read **0 of 77,975**, the uniform control at
   $4\times10^{12}$ must go **25 → 0**, and **the only other change in the whole diff** should be
   the LambdaCDM $k=3\times10^8$ consumer row of section 3 falling from 6.17 rad to the floor, plus
   timings. **Any other moved number is a stop** — the remainder was supposed to be bit-identical.
7. `git diff HEAD~1 --stat`: allowed files only. No producer, no `Quadrature/`, no `Datastore/`, no
   `main.py`, no consumer. `test_tk_source_functions.py` may appear with a **comment-only** hunk —
   check that with `git diff HEAD~1 -- ComputeTargets/tests/test_tk_source_functions.py`.
8. **The datastore consequence is stated, not papered over** (prompt 01 §5): the log says
   `theta_div_2pi` is in no lookup key, that a pre-01 datastore is served silently, and which
   quantities are and are not affected. If the agent invented a migration, a column or a key,
   **stop**.

## Reviewing prompt 02

1–3 as above (P2; the other `[13-...]` entry).
4. **Which construction was taken**, and if per-segment, the README §2 (d) argument in the log that
   this is not the chunking prompt 08 deleted — physically declared boundaries, a bounded residual,
   no rebased ordinates, no switch discontinuity — with a **test** demonstrating the last, not an
   assertion. If the log does not distinguish them, **stop**: that is the one confusion this
   campaign is most likely to ship.
5. **`num_chunks` still returns 1**, and `WKB_phase_spline_chunks` would still be 1:
   ```bash
   grep -n "num_chunks" ComputeTargets/primitive_phase.py
   ```
   Anything else is a changed stored value and a stop.
6. **Smooth cosmologies are bit-identical.** The prompt's test asserts it; verify the production
   half yourself in the section 3 and section 5 tables of the diff — every LambdaCDM row unchanged.
7. **The numbers.** Section 3.5's two QCD $k=10^5$ rows meet $\le10^{-6}$ rad **and** $\le2$ ulp;
   the ten rows at 1.00 ulp stay there. Section 3.6 reports `theta_deriv` as **two** contributions
   — the knots' and `[02-qcd-T-z-spline-node-tolerance]`'s — not one number. A residue attributed
   to the latter is a narrowing, reported, not a miss hidden; a missed row with the Result still
   `COMPLETE` is a stop.
8. **Cost** recorded, both models; stop if $>2\times$ the 0.0010 s / 468 evaluations baseline.
9. `git diff HEAD~1 --stat`: allowed files only. **No `CosmologyModels/`** — the break points are
   already declared. `BackgroundModel.py` only if §2 item 1 forced it, and then only as a
   deviation.

## Continue or stop

Continue when both logs are `COMPLETE`-class with only `IMPLEMENTATION CHOICE` deviations and every
check passes. Stop on any README §4 condition, on a moved number that should have been
bit-identical, on prompt 02's chunking confusion, or on a silent miss.

**Relay every subagent question verbatim. Do not answer it yourself.**

## Completion criterion

Both rows ✅/⚠️; two logs; both suites passing; both `GkTk-remedial` §3 entries in that board's §4.

Report: "The phase-representation campaign is complete; the tree is at `<SHA>`. The consumer path
is at 1 ulp of the span on both models at every measured wavenumber" — or the rows that are not,
with their issues. Include the before/after for the LambdaCDM $k=3\times10^8$ consumer row
(6.17 rad → ?), the two QCD $k=10^5$ rows (1.907e-6 / 3.186e-6 rad → ?), the `theta_deriv` split,
and the per-object cost. **Tell the user that a datastore written before prompt 01 carries the old
`theta_div_2pi` at the affected samples and is not refused on read.**

Then write the campaign's close-out: a dated **§8** appended to
`docs/gktk-remedial-verification.md` recording what moved, with **nothing above §7 edited**
(`CLAUDE.md`: verification documents are additive). That is the only edit this campaign makes to
that document.
