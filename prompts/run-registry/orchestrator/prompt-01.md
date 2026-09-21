# Orchestrator — prompt 01, the run registry

Read [`../README.md`](../README.md) first, especially §0. **You do not write code.**

**The prompt:** [`01-the-run-registry.md`](../01-the-run-registry.md) · **Board item:** G1

## 0. What makes this prompt unusual

Infrastructure prompts fail by succeeding too much. The suite will be green, the code will look
tidy, and the failure mode is a framework: config layers, a plugin point, a scheduler nobody asked
for, a `cleanup()` that deletes evidence. README §1 and prompt §3 forbid all of it, and §5 rule 7
says every manifest field must trace to one of the five failures in README §0.

So the review is mostly subtractive. For each thing built, ask which of the five it would have
caught. If the answer is "none", it should not be there.

The one positive check that matters: **the self-match regression test** of prompt §1.5. That single
mistake — `pgrep -f <name>` matching the polling shell's own command line — produced nineteen
immortal shells and burned a large part of a day's quota on turns that learned nothing. A registry
whose liveness check can repeat it is worthless. Verify that test exists **and that it fails
against a naive pattern-matching implementation**.

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`**, record the SHA, and tell the agent it is not its own.
2. **Baseline every suite**: `ComputeTargets` 552, `CosmologyModels` 39, `LiouvilleGreen` 148
   (skipped=1) at `ab7079c`, plus `AdaptiveLevin`, plus `Datastore` if
   `prompts/datastore-readback` prompt 01 has landed. Record them.
3. **Record the live state prompt §5 forbids touching**, so you can check it afterwards:
   ```bash
   wc -l < var/runs/realistic_large_x_cells.jsonl        # expect 60
   ls var/runs/ var/datastores/
   du -sh var/datastores/backup-pre-resume-20260921T091011
   ```
4. `git status` clean.

## 2. Dispatch

One fresh-context subagent: the prompt, the campaign README, the SHA, the baselines, and the files
its "Read first" names. Nothing else.

Tell it plainly: **one commit**; log at `logs/01-the-run-registry.md`; board created per §8;
`docs/OPEN_ISSUES.md` in the same commit; `black`; **nothing under `var/` may be deleted, tidied or
modified**; and §7's stop conditions mean *stop and ask*.

## 3. The review — eight checks

1. **Size.** `git diff --stat`. If `RunRegistry/` is more than a few hundred lines excluding tests,
   ask what justifies it. Prompt §7's first stop condition exists for this; if the agent sailed
   past it, that is itself the finding.
2. **The self-match regression exists and bites.** Read the test. Then verify by hand that a naive
   `pgrep -f`-style check fails it — the agent should have shown this; if not, construct the naive
   case yourself and confirm the test would catch it. **This is the one check worth doing
   manually.**
3. **Every manifest field traces to a README §0 failure.** Go field by field. A field that traces
   to nothing is scope creep with a plausible face.
4. **Nothing forbidden was built.** Grep the package for anything resembling cleanup, deletion,
   retry, restart, locking, or a daemon. Prompt §3 forbids each by name.
5. **The lister degrades on the pre-registry directories.** Run
   `python -m RunRegistry list` yourself. `var/runs/a3-pilot/` and `a3-pilot-resume/` have no
   manifest; the lister must handle them without crashing and without inventing state.
6. **`var/` is untouched.** Checkpoint still 60 cells; both pilot directories intact; the datastore
   and its backup present at their recorded sizes. A registry campaign that deleted evidence on its
   first outing would be a poor start.
7. **The `CLAUDE.md` section states all six rules of prompt §2**, including — explicitly — *do not
   babysit: launch, verify once, end the turn*. That rule is the one that saves the user's quota,
   and it is the easiest to drop because it is about behaviour rather than code.
8. **Suites unchanged; `RunRegistry/tests` up by exactly the tests added.** `black --check` clean;
   board created; index updated.

## 4. What a good outcome looks like

- A package small enough to read in one sitting, and a `CLAUDE.md` section short enough to be
  followed.
- A limits section that concedes the real ones: per-unit resolution, the need for a serialisable
  unit, a heartbeat that cannot distinguish "progressing" from "alive" in every case, and that a
  registry nobody reads is worse than none.
- The self-match test, demonstrated to fail against the naive implementation.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate.

- The helper has grown into a framework, or wants a dependency.
- The self-match regression cannot be written honestly.
- The agent proposes changing a production compute path, or deleting anything under `var/`.
- The agent proposes a manifest field it cannot trace to a README §0 failure — it may have found a
  sixth failure worth recording, which is the user's to judge.
- Any check in §3 fails. **Report it; do not repair it.**

## 6. After it lands

Report: the commit; the package size; the six rules as they went into `CLAUDE.md`; the self-match
test and whether you saw it bite; the lister's output on the two pre-registry directories; the
manifest fields with the failure each traces to; the `var/` state before and after; and the suite
counts.

Then stop. Prompt 02 is adoption and has its own hazard — do not dispatch it without being asked.
