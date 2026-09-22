# Orchestrator — prompt 03, the overloaded `checkpoint` field

Read [`../README.md`](../README.md) first, especially §0 items 4 and 5 and §0.2. **You do not write
code.**

**The prompt:** [`03-the-checkpoint-field-means-two-things.md`](../03-the-checkpoint-field-means-two-things.md)
· **Board item:** G3 · **Prerequisite:** prompts 01 and 02 have landed.

## 0. What makes this prompt unusual

It is the smallest prompt in the campaign and the easiest to overbuild. The defect is one field
carrying two meanings; the fix is two fields and a refusal. Everything that could be added around
it — a schema version, a migration path, a validator, a manifest type system — is forbidden by
prompt §4, and each of them will look reasonable while the agent is writing it.

There is also a window. Prompt 01 §1.2 makes the manifest immutable, so a field cannot be
re-meaned under a run that already exists. Right now no registry-written manifest survives, which
is why this is a rename rather than a migration. **Confirm that yourself before dispatching** — if
a real run has been registered since, the prompt's §6 first stop condition has already fired and
the agent must not start.

The positive check that matters: **the refusal test with a byte-identical results file**
(prompt §5 item 2). Everything else is bookkeeping; that one is the bug.

## 1. Before you dispatch

1. **Confirm `HEAD`** and that prompts 01 and 02 have landed. Record the SHA; tell the agent it is
   not its own.
2. **Confirm the window is still open** — this is the one that decides whether the prompt can run
   at all:
   ```bash
   find var/runs -name manifest.json
   ```
   Expect **nothing**. The three prompt-02 demonstration directories were removed by the user on
   2026-09-22. If this prints a manifest, read it: if it belongs to a real run rather than a
   demonstration, **do not dispatch** — relay it and ask.
3. **Baseline the suites**, including `RunRegistry` as prompt 02 left it (30).
4. **Record the protected state** — the A3 datastore and backup, both pilot directories,
   `realistic_large_x_cells.jsonl` at 60 cells with the single hash `c1cd3598c23a`. Prompt 03 has
   no business near any of it, which is exactly why it is worth being able to prove afterwards.
5. `git status` clean.

## 2. Dispatch

One fresh-context subagent: the prompt, the campaign README, prompt 02's log, the SHA, the
baselines, the `find` result, and the files its "Read first" names.

Tell it plainly: **one commit**; log at `logs/03-the-checkpoint-field-means-two-things.md`; board
G3 row and the issue moved to §4; `docs/OPEN_ISSUES.md` row **deleted** and the count corrected, in
the same commit; `black`; nothing under `var/` touched; and §6's stop conditions mean *stop and
ask*.

## 3. The review — seven checks

1. **The refusal test exists and is about bytes.** Read it. A run whose results are a store and
   which declares no ledger must have its results file **byte-identical** after a refused
   `record()`. A test that asserts only that an exception was raised has tested the exception, not
   the file.
2. **`known()` is no longer silent** on a file that is not a ledger, and is *still* silent on a
   ledger that does not exist yet. Both halves — the second is how a first run works, and breaking
   it would be worse than the bug.
3. **Prompt 01's torn-final-line test passes unchanged.** `git diff` must not touch it. If any test
   prompt 01 or 02 wrote has changed, that is a finding: read why, and do not accept "it needed
   updating" without a reason that is about the code rather than the test.
4. **Nothing forbidden was built.** Grep the diff for a schema version, a migration, a validator, a
   compatibility shim, or an import from `Datastore/`. Prompt §4 forbids each by name.
5. **Exactly one new manifest field**, traced in the log to README §0 item 4 or 5. A second new
   field is a §5 relay, not something to wave through.
6. **`scoped_pipeline_run.py` is still confined** — the registration block only, and `main.py`
   still runs byte-identically with and without `--register`. Prompt 02's constraint did not lapse
   because a later prompt touched the file.
7. **Suites at baseline; `black --check` clean; the index row deleted, not left behind.** The
   closed issue must be in the board's §4 and *gone* from `docs/OPEN_ISSUES.md`, with the count
   corrected — `CLAUDE.md` is explicit that the index keeps no resolved section.

## 4. What a good outcome looks like

- A diff small enough that the whole change is legible in one screen, plus tests that are larger
  than it.
- An error message a person who copied the pipeline pattern can act on without reading the source.
- A log that says what `find var/runs -name manifest.json` printed before the work started, which
  is the evidence that no migration was owed.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate.

- A manifest belonging to a real run exists — the window has closed.
- The agent proposes a second new field, a schema version, a migration or a validator.
- The agent proposes changing a test prompt 01 or 02 wrote.
- Any check in §3 fails. **Report it; do not repair it.**

## 6. After it lands

Report: the commit; the two field names and what each means; the refusal message as a caller sees
it; what `known()` now does on an unreadable ledger; the `find` output before and after; the
diff to `scoped_pipeline_run.py`; and the suite counts.

Then stop. With G3 closed the campaign's own issues are clear; whether anything further adopts the
registry is the user's call.
