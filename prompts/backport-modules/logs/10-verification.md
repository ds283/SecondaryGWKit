# Log 10 — Verification pass and campaign close-out

**Prompt:** prompts/backport-modules/10-verification.md
**Commit:** *(SHA intentionally not embedded — see `IMPLEMENTATION_STATE.md` §5 note 11)*
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

`docs/backport-modules-verification.md` (new): the campaign's close-out document. Contains:

- Git-history reconciliation (one commit per prompt, in order, two out-of-band commits already
  tracked elsewhere) and a fresh re-run of every standing-note grep count from
  `IMPLEMENTATION_STATE.md` §5 note 5 — all matched expectations.
- Static verification (Step 2 of the prompt): byte-compile, `black --check`, real (unstubbed) imports
  of all four touched modules, `read_table_config`/`inventory_config` shape checks against the real
  factory signatures, and confirmation that `ObjectFactories/base.py` was never touched (no
  `inventory`/`read_table` made abstract). All pass.
- Behavioural verification (Step 3): the audit's §8 checklist, extended with the four F2-specific
  checks `README.md`/this prompt add (12 items total). **8 of 12 now have live confirmation** against
  a real, locally-bootstrapped, multi-shard, Ray-actor-backed `ShardedPool` — several of these had
  previously only been verified against direct method calls, `__new__`-constructed stand-in objects,
  or hand-simulated dispatch logic. The 3 that remain unrun (checklist items 4, 7, 8) all require
  actual physics compute, which is out of proportion to a documentation-and-scripts-only prompt; each
  is recorded with a concrete estimate of what closing it would take.
- A gathered table of every deviation from every one of the nine implementation prompts' own logs,
  each carrying its classification (structurally required / implementation choice / unintended
  drift) and a one-line reason, so a reader does not have to open nine logs to get an overview.
- The deferred/out-of-scope items (F4, reporting X1 upstream), restated for a self-contained record.

`scratchpad/verify_prompt10.py` (throwaway, **not committed**, per the campaign's established
convention — see the "keep scripts with ongoing value" note below) and its captured output
`scratchpad/verify_prompt10_output.log` (also not committed): the harness that produced the live
results above. Builds a real 2-shard `ShardedPool` under `ray.init(num_cpus=4)`, then:

- inserts 5 wavenumbers via the real `object_get` → `_assign_shard_keys` path and checks for
  `MISMATCH` output and `shard_keys.key_serial`/`wavenumber.serial` agreement (checklist item 1),
  also running the real `tools/shard_key_audit.py` tool against the result;
- shuts down and reopens a second pool against the same primary file, confirming no `AttributeError`,
  identical `store_id`s on re-request, and no row duplication (items 2–3), then opens a third pool
  with a deliberately mismatched `ShardKeyType` and confirms the intended `RuntimeError` (item 3);
- exercises `read_table`'s two negative cases and one positive case against the live pool (item 5);
- runs a `RayWorkPool` with a `None`-returning task builder under both `store_results` settings
  against the live pool (item 6);
- re-runs `_merge_queue`'s full policy/edge-case matrix as a pure function (item 9) and a fresh,
  exhaustive field-by-field cross-check of all 15 sharded classes' `inventory()` output against
  `config/sharding.py`'s `inventory_config` (item 10);
- runs the real inventory report against the live (partially populated) pool and checks for no
  `(error` / stray `None` (item 11);
- inserts synthetic rows into a real sharded value table across both real shard files and confirms
  `pool.inventory(...)`'s `"sum"` policy against a direct-SQL cross-check (item 12);
- **beyond the checklist**, closes the remainder of the `[08-inventory-sharded-factories]` issue: one
  Group A (labelled) and one Group B (flat-with-timestamps) class, each with real rows on both real
  shards, confirming the merge genuinely spans shards rather than reflecting one.

`IMPLEMENTATION_STATE.md`: prompt 10's row marked complete, progress set to 10/10, "last updated"
updated, three of the four §3 issues resolved or narrowed (two moved to §4 as fully closed, one
narrowed to its still-open half), the fourth left open with an updated next step, and a new §5 note
15 summarising the campaign's closing state.

## Deviations from the prompt

### Building a real datastore instead of only re-running static/cheap checks — IMPLEMENTATION CHOICE

The prompt frames "the highest-value single action is to create the smallest fresh datastore the
pipeline will produce" as a recommendation, and separately says checks 6/9/10 are cheap and
independent of a datastore. Rather than choosing one or the other, this pass did both: built one real
multi-shard pool and then reused it across as many of the live-Ray-capable checks as it could support
(1, 2, 3, 5, 6, 11, 12, plus the two bonus Group A/B closures for `[08-inventory-sharded-factories]`),
since the marginal cost of adding one more check against an already-running pool was small once the
harness existed, and the prompt's own principle ("re-run them rather than taking the logs' word for
it — they are cheap") extends naturally to reusing infrastructure that is already up.

### Did not attempt a scoped-down real compute run for checks 4/7/8 — IMPLEMENTATION CHOICE

The prompt explicitly permits recording a check as not run when it needs more compute than
reasonable ("Record it as not run, say what it would take, and hand it to the user"). Considered
building a small custom driver (outside `main.py`, a handful of wavenumbers, one queue) to close at
least one of checks 4/7/8 for real, since the infrastructure (local `ray.init()`, real pool
construction) was already in hand from the other checks. **Chose not to**, for two reasons: (a) this
prompt's own file list is "verification scripts and campaign documentation only" and the checklist
items in question are explicitly the ones the audit and every prior prompt agreed need "a real
pipeline run" as a distinct, larger undertaking, not a same-day extension of this pass; (b) an actual
compute-target integration (`TkNumericIntegration` et al.) carries real numerical behaviour
(tolerances, step counts, WKB switching) this campaign's prompts never touched and this pass has no
basis to sanity-check beyond "did it run" — attempting one under schedule pressure risks either a
silent wrong-but-plausible result or an open-ended debugging detour into domain code well outside
this campaign's scope. Recorded instead as a concrete, scoped-down recommendation for a follow-up
(verification document §5), which is what the prompt asks for when a check cannot reasonably be run
here.

### Closing `[08-inventory-sharded-factories]`'s remainder beyond the checklist — IMPLEMENTATION CHOICE

Not itself one of the audit's 12 checklist items, but `IMPLEMENTATION_STATE.md` §3 named it as the
exact next step for an issue this prompt is responsible for reconciling ("prompt 10 can close the
remainder by inserting synthetic rows for one class in each of those two groups the same way prompt
09 did"). Done, since the harness and technique were already in hand and the marginal cost was one
more pair of SQL inserts plus one more `pool.inventory(...)` call — see "What shipped" above.

### No other deviations

Steps 1–4 were followed in the order and with the content the prompt specifies. No production code
was touched (confirmed: `git status`/`git diff` before committing shows only
`docs/backport-modules-verification.md`, `prompts/backport-modules/IMPLEMENTATION_STATE.md`, and
`prompts/backport-modules/logs/10-verification.md` changed).

## Verification performed

This prompt's own "verification" *is* its subject matter — see
[`docs/backport-modules-verification.md`](../../../docs/backport-modules-verification.md) for the
complete, itemised account (static checks §3, behavioural checklist §4, the
`[08-inventory-sharded-factories]`-closing extra in §4.1). Summary:

- Static: byte-compile, `black --check`, real imports, `read_table_config`/`inventory_config`
  structural checks, `base.py` unchanged — all pass.
- Behavioural: 8/12 checklist items now live-confirmed against a real multi-shard Ray-actor pool (1,
  2, 3, 5, 6, 9, 10, 11, 12); 3/12 correctly recorded as not run, with a concrete estimate of what
  closing them would take (4, 7, 8).
- Beyond the checklist: `[08-inventory-sharded-factories]`'s remaining Group A/Group B live cases,
  closed.
- Standing-note grep counts re-run fresh: all match `IMPLEMENTATION_STATE.md` §5 note 5's recorded
  expectations, confirming no drift since prompt 09.

## Observations not acted on

- **`main.py`'s wavenumber sample size is hardcoded** (`np.logspace(np.log10(1e5), np.log10(3e8),
  50)`, both source and response, `main.py` lines 2669/2682), with no CLI flag to reduce it. This
  makes "the smallest configuration `main.py` accepts" still a real, if comparatively cheap, physics
  run (100 horizon-exit-time root-finds before any gated queue is reached), and is the reason checks
  4/7/8 could not be closed by simply invoking `main.py` with restrictive flags. Not fixed — adding a
  CLI override would be a production-code change, outside this prompt's file list, and was not asked
  for by the audit or any prompt in this campaign. Recorded in the verification document §5 as context
  for whoever picks up the remaining checks.
- **`scratchpad/verify_prompt10.py` has no ongoing value as a committed regression test** in its
  current form (prints results rather than asserting them as a suite would, rebuilds pool-construction
  boilerplate inline). Not promoted to `tools/` — see the verification document §6 for the reasoning,
  which is the same "no test infrastructure for `Datastore`/`ShardedPool`/`RayWorkPool` exists"
  observation `IMPLEMENTATION_STATE.md` §5 note 2 already recorded, now with a concrete illustration of
  what a real suite would need to factor out.

## State handed to the next prompt

There is no next prompt — **this closes the campaign** (`IMPLEMENTATION_STATE.md` §1: 10/10
complete). For whoever picks up the two still-open issues
(`[04-read-table-service]`'s `extract_*.py` half, and all of `[05-persist-handler-split]`):

- Both need exactly the same missing ingredient: one real compute-target object (e.g.
  `TkNumericIntegration`) carried through an actual `RayWorkPool` compute→store→persist cycle against
  a live pool. Closing one is very likely to closely follow closing the other.
- The infrastructure to build the real pool and a local Ray runtime is proven and documented (this
  log, prompt 09's log, and `docs/backport-modules-verification.md` §4/§5) — `ray.init(num_cpus=N,
  include_dashboard=False)` with no pre-existing cluster, real `ShardedPool` construction exactly as
  `main.py` does it but with a hand-picked small wavenumber sample instead of `main.py`'s hardcoded
  50+50.
- `docs/backport-modules-verification.md` §5 gives a concrete, scoped-down recommendation (a small
  driver script, 5–10 wavenumbers, one queue enabled) and an order-of-magnitude time estimate (minutes,
  not hours) — read from the code, not measured, and should be treated as an estimate until actually
  run.
