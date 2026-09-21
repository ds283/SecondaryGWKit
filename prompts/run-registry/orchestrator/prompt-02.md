# Orchestrator — prompt 02, adopting the registry

Read [`../README.md`](../README.md) first, especially §0 and §4. **You do not write code**, and in
particular you do not perform the deliberate kill that prompt 02 §4 item 2 requires — the agent
does that and records it; you check the record.

**The prompt:** [`02-adopt-the-registry.md`](../02-adopt-the-registry.md) · **Board item:** G2
**Prerequisite:** prompt 01 has landed and `RunRegistry/` exists.

## 0. What makes this prompt unusual

It has one hazard and one temptation, and they are the same thing seen from two sides.

`realistic_large_x.py` gates checkpoint reuse on a hash of its own source. Editing it discards the
sixty cells behind every published table in `docs/handover/REALISTIC-LARGE-X.md`. The prompt says
leave it alone. The temptation, for an agent that has been told to make scripts adopt the registry,
is either to retrofit it anyway, or — far worse — to retrofit it *and* re-stamp the existing
records with the new hash so that reuse still works.

**Re-stamping is falsification.** The hash exists to assert that a specific script produced
specific numbers; rewriting it to match a changed script makes that assertion false while leaving
it looking sound. If the log shows any migration, regeneration or re-stamping of existing
checkpoint records, that is a stop, and it is the most serious thing that could come out of this
campaign.

The second thing to watch is honesty about the heartbeat. A Ray-parallel pipeline's parent process
can be alive and blocked forever. If the log claims the heartbeat means "progressing" without
saying how it knows, it probably means "alive".

## 1. Before you dispatch

1. **Confirm `HEAD`** and that prompt 01 landed. Record the SHA; tell the agent it is not its own.
2. **Baseline the suites**, including `RunRegistry/tests` as prompt 01 left it.
3. **Record the protected state, exactly**, because these are the numbers the review turns on:
   ```bash
   wc -l < var/runs/realistic_large_x_cells.jsonl
   ./venv/bin/python -c "import json;print({json.loads(l)['script_sha256'][:12] for l in open('var/runs/realistic_large_x_cells.jsonl') if l.strip()})"
   shasum -a 256 docs/handover/realistic_large_x.py
   ls var/runs/ var/datastores/
   ```
   Expect 60 cells, the single hash `c1cd3598c23a`, and both pilot directories present.
4. `git status` clean.

## 2. Dispatch

One fresh-context subagent: the prompt, the campaign README, prompt 01's log, the SHA, the
baselines, the protected state above, and the files its "Read first" names.

Tell it plainly: **one commit**; log at `logs/02-adopt-the-registry.md`; board and
`docs/OPEN_ISSUES.md` in the same commit; `black`; **`realistic_large_x.py` is not to be edited and
its checkpoint is not to be touched**; the A3 datastore, its backup and both existing pilot
directories are read-only; the demonstration run must use a **throwaway** datastore; and §6's stop
conditions mean *stop and ask*.

Tell it also to launch the demonstration detached and end its turn rather than polling — the rule
it is adopting applies to it while it adopts it.

## 3. The review — eight checks

1. **`realistic_large_x.py` is byte-identical.** `shasum -a 256` against your §1.3 value, and
   `git diff HEAD~1 HEAD -- docs/handover/realistic_large_x.py` empty.
2. **Its checkpoint is untouched**: 60 cells, one script hash `c1cd3598c23a`. If the hash set has
   more than one member, records were written under a changed script — stop.
3. **No record anywhere was re-stamped, migrated or regenerated.** Read the log for any language
   suggesting it. §0 says why this is the serious one.
4. **The demonstration is real.** The log must quote the lister's output at three points: after
   launch (`running`, manifest present), after the deliberate kill (**stale or `killed`, not
   alive**), and the manifest's existence from the first second. A demonstration asserted rather
   than quoted is not one.
5. **The driver still changes only three things about `main.py`.** Read the diff to
   `scoped_pipeline_run.py`. Its whole claim is that substitution is textual, exact, and limited;
   registry calls must not be scattered through the pipeline logic. A thin wrapper or an opt-in
   flag is fine; edits interleaved with the substitution logic are not.
6. **The heartbeat's meaning is stated**, and the manifest does not imply a guarantee it cannot
   give.
7. **Protected state intact**: A3 datastore and backup at their sizes, both pilot directories
   present and untidied, throwaway datastore confined to `var/runs/`.
8. **Suites unchanged**; `black --check` clean; board and index updated; the non-adoption of
   `realistic_large_x.py` recorded as a classified `IMPLEMENTATION CHOICE` with the hash reasoning.

## 4. What a good outcome looks like

- A pipeline run that shows up in the lister within a second of starting, and shows up as dead
  within a heartbeat window of being killed.
- `realistic_large_x.py` deliberately and visibly *not* adopted, with the reasoning written down,
  so the next person understands it is a decision rather than an oversight.
- A worked example short enough that the next measurement script can follow it without importing a
  framework.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate.

- **Any re-stamping, migration or regeneration of existing checkpoint records.**
- The agent concludes `realistic_large_x.py` should be retrofitted after all.
- Registering the driver cannot be done without changing what it computes.
- The demonstration cannot be kept short enough to be a demonstration.
- Any check in §3 fails. **Report it; do not repair it.**

## 6. After it lands

Report: the commit; `realistic_large_x.py`'s hash and its checkpoint's cell count and script hash
before and after; the lister's output at the three demonstration points; what the heartbeat means;
the diff to the driver and whether it is confined; the protected state before and after; and the
suite counts.

Then stop. Whether anything else adopts the registry is the user's call, and the interrupted A3
run is still theirs to finish once `prompts/datastore-readback` prompt 01 has landed.
