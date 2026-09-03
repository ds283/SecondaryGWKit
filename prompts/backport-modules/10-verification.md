# Prompt 10 — Verification pass and campaign close-out

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit section:** §8 (verification checklist for the implementer)
**Depends on:** prompts 01–09, whichever of them landed
**Files you may touch:** verification scripts and campaign documentation only. **No production code.**

---

## Purpose

The audit's §8 checklist spans all the code commits and cannot be discharged inside any one of
them. This prompt collects it, runs everything that can be run, and produces an honest statement of
what remains unverified.

**The deliverable is an accurate account, not a green tick.** A checklist item that could not be
exercised must be reported as not exercised. Do not infer a pass from reading the code and then
record it as though it were run — a later reader has no way to tell the difference, and the whole
point of this pass is to be the thing they can trust.

---

## Step 1 — Reconcile the campaign

Before running anything:

1. Read every log in `prompts/backport-modules/logs/`. Build a picture of what actually shipped, and
   in particular of every deviation classified as an **implementation choice** or **unintended
   drift**.
2. Read `IMPLEMENTATION_STATE.md` §3 and collect every open issue. Most will be "could not run this
   behavioural check" — those are this prompt's work list.
3. `git log --oneline` since the campaign baseline `79f0360`. Confirm one commit per prompt, in
   order, with no stray commits interleaved. If the history is not clean, say so; do not rewrite it.
4. Re-run the **standing-note counts** from `IMPLEMENTATION_STATE.md` §5 note 5 and confirm they moved
   as expected:
   - `read_wavenumber_table|read_redshift_table` → 0 (was 19), if prompt 04 landed
   - `store_handler=None` → 35, each paired with `persist_handler=None`, if prompt 05 landed
   - `RayWorkPool(` → 45, unchanged
   - `_generic_read_table` → 0, if prompt 04 landed
   - `def inventory` under `ObjectFactories/` → 28 (13 replicated + 15 sharded), if 07 and 08 landed
   - `key_id` in `ShardedPool.py` → 0
   - `_db_file` in `ShardedPool.py` → 0

## Step 2 — Static verification

Cheap, and all of it should pass:

- Every touched file parses and byte-compiles: `python -m py_compile` across `main.py`,
  `extract_*.py`, `Datastore/SQL/*.py`, `RayTools/*.py`, `config/sharding.py`.
- `black --check` across the touched files, if `black` is available.
- Import the modules (`Datastore.SQL.ShardedPool`, `Datastore.SQL.Datastore`,
  `RayTools.RayWorkPool`, `config.sharding`) and confirm no import-time errors and no reference to a
  name that a prompt removed.
- `config.sharding.read_table_config` is keyed by class name (`"wavenumber"`, `"redshift"`) with no
  `"class"` entries, and its `tables_arg` values match the factory signatures at
  `ObjectFactories/wavenumber.py:90` (no `tables` → `False`) and `ObjectFactories/redshift.py:76`
  (takes `tables` → `True`).
- `config.sharding.inventory_config` covers every sharded class and only sharded classes.
- No `def inventory(self,` anywhere under `ObjectFactories/` — that is the upstream-copy failure
  mode, since this tree registers factory classes with staticmethods rather than instances.
- `ObjectFactories/base.py` is unchanged: `inventory` was never made abstract.

## Step 3 — Behavioural verification

This is the substance. The audit's §8 checklist, with what each needs:

| # | Check | Needs |
|---|---|---|
| 1 | On a fresh datastore, every `shard_keys.key_serial` equals the `store_id` of the corresponding `wavenumber` row; no `!! _assign_shard_keys MISMATCH` lines | a fresh datastore + `tools/shard_key_audit.py` |
| 2 | A run stopped and resumed against the same datastore finds all previously-written records (no unexpected recomputation) | two pipeline runs |
| 3 | Reopening an existing sharded datastore completes `_read_shard_data` without `AttributeError`; a mismatched `shard_key_type` gives the intended `RuntimeError`, not a second `AttributeError` | a datastore in the current schema |
| 4 | Each `extract_*.py` constructs its `ShardedPool` and returns the expected wavenumber/redshift arrays | a populated datastore |
| 5 | `pool.read_table("GkSource", …)` (sharded) raises; `pool.read_table("LambdaCDM", …)` (replicated, unconfigured) raises | a constructed pool |
| 6 | A task builder returning `None` completes under `store_results=False` and raises under `store_results=True` | a `RayWorkPool` |
| 7 | A work pool using the default handlers still stores results as before | a pipeline run |
| 8 | Full smoke test: `main.py` through one compute→store→validate cycle on a fresh datastore, then a restart against the same datastore | a real run |
| 9 | *(F2)* `_merge_queue` handles every policy, the empty-shard case, and rejects unknown policies and missing fields | nothing — pure function |
| 10 | *(F2)* every sharded class has both an `inventory()` and a matching `inventory_config` entry, agreeing field by field in both directions | nothing — static |
| 11 | *(F2)* the inventory report runs against a populated datastore and against an empty one, with no `None` leaking into the output | a datastore |
| 12 | *(F2)* a value-table count matches `SELECT COUNT(*)` summed by hand across shard files — proving the `"sum"` policy, which a `"latest"`-style bug would make look plausible | a datastore |

**Sequencing that unlocks the most for the least.** Checks 1, 3, 4, 5 and 7 all need a datastore in
the current schema, and none exists (`IMPLEMENTATION_STATE.md` §5 note 1 — `test-qcd-db.sqlite` is
pre-`a2bd966` and unusable as a fixture). **So the highest-value single action is to create the
smallest fresh datastore the pipeline will produce**, then run the cheap checks against it. Look for
the smallest viable configuration `main.py` accepts — fewest wavenumbers, fewest redshifts, shortest
integration range, fewest shards — before assuming a full run is needed.

Checks 6, 9 and 10 are independent of all this and should be exercised regardless — none needs a
datastore. Check 6 needs a `RayWorkPool` but a task builder returning `None` never reaches the pool;
check 9 is a pure function of its arguments; check 10 is static analysis of the config against the
factories. If prompts 06 and 08 did their own verification properly these will already have been
run, in which case re-run them rather than taking the logs' word for it — they are cheap.

Check 2 is the one that actually tests B1's value — it is the restart case, which is the only way
the bug ever manifested. If you can afford exactly one behavioural test, make it a small
run-stop-resume cycle.

**If a check needs more compute than is reasonable to spend here, that is a legitimate outcome.**
Record it as not run, say what it would take, and hand it to the user.

## Step 4 — Write the close-out

Produce `docs/backport-modules-verification.md`:

- **What was verified, and how.** Per checklist item: run / not run / partially run, the exact
  command or script, and the actual output. Include the failures if any.
- **What was not verified, and what it would take.** Be specific: "check 2 needs a
  run-stop-resume cycle on a fresh datastore, roughly N minutes at the smallest configuration" is
  useful; "needs a pipeline run" is not.
- **Deviations across the campaign**, gathered from the logs into one table: item, prompt, what
  changed relative to the plan, classification (structurally required / implementation choice /
  unintended drift), and — for implementation choices — a one-line statement of the reasoning so a
  reader can decide whether to revisit it without opening five logs.
- **Anything found during verification that is still wrong.** If a check fails, do **not** fix it in
  this prompt. Record it, open a §3 issue, and recommend which prompt should be re-run or what a
  follow-up should do. This prompt does not touch production code.
- **The deferred items**, restated so the close-out is self-contained: F4 (licence headers,
  repo-wide or not at all), and the suggestion to report X1 upstream to the `StochasticInstantons`
  tree as a latent break. F2 is **no longer deferred** — the user asked for it and prompts 06–09
  deliver it — so report its state like any other landed work, including any class whose inventory
  was deliberately limited (a label list dropped as too expensive, say) so a later reader knows the
  omission was a choice.

Keep any verification script you wrote that has ongoing value under `tools/` and commit it. Delete
throwaways.

---

## Finish

1. Write `prompts/backport-modules/logs/10-verification.md` using the template in `README.md` §5.1.
   Here "what shipped" means the verification document and any retained scripts.
2. Update `IMPLEMENTATION_STATE.md`: the prompt 10 row, the progress count, "last updated". Close
   every §3 issue you discharged by moving it to §4 with the outcome; leave open the ones you could
   not, with an updated next step. If the campaign is complete, say so at the top of the board.
3. Commit in one commit. Suggested message:

```
Verify the Ray/Datastore backport campaign

Collects the verification checklist from docs/backport-modules-audit.md,
extended to cover the inventory() service, which the audit deferred and the
user subsequently asked for. The checklist spans all the code commits and
could not be discharged inside any one of them.

Records what was actually executed and what was not, with the reason and the
cost of doing it, so the distinction between a check that passed and a check
that was only reasoned about is visible to a later reader. Gathers the
per-prompt deviation logs into one table, separating changes that were
structurally required from implementation choices and from unintended drift.

Also restates what remains deferred: Apache-2.0 licence headers, which should
be applied repo-wide or not at all, and the latent break in
StochasticInstantons' object_read_batch guard, which is worth reporting
upstream.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Adjust to match what you actually did — in particular if any check failed.
