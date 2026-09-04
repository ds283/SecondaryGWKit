# Log 09 — The inventory reporting entry point (F2, part 4 of 4)

**Prompt:** prompts/backport-modules/09-inventory-reporting.md
**Commit:** *(SHA intentionally not embedded — see `IMPLEMENTATION_STATE.md` §5 note 11)*
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

### Task 1 — where it lives

Took the prompt's recommendation **(a): a flag on `main.py`**, for the reason the prompt itself
gives — `main.py` already constructs a fully-configured `ShardedPool` (units, cosmology model list,
sharding/read-table config), and a `--inventory` flag that reports and exits reuses all of that for
free.

One refinement not dictated by the prompt: the **formatting logic itself lives in a new module,
[`tools/inventory_report.py`](../../../tools/inventory_report.py)** (plus an empty
`tools/__init__.py`, since `tools/` had none — it was previously only ever invoked as a standalone
script, per prompt 01's `tools/shard_key_audit.py`), not inline in `main.py`. `main.py` owns only the
two new argparse flags, the `inventory_config` import/wiring, and a four-line call-and-exit block.
`format_inventory_report(pool, db_name, verbose)` is a pure function (no argparse, no global state)
that takes a pool and returns the report as a string — which is what made it possible to exercise
against a real `ShardedPool` from a throwaway script without fighting `main.py`'s
`args = parser.parse_args()` at import time (a bare `import main` executes the whole script and exits
via `parser.print_help(); sys.exit()` when no `--database` is supplied). This still satisfies option
(a)'s framing (`main.py` is where the flag and the pool live) while keeping the part this prompt
actually had to get right — the formatting — independently testable. **Classification: IMPLEMENTATION
CHOICE**, not a departure from option (a).

`main.py` changes:
- Import `inventory_config` alongside the other `config.sharding` names (`main.py` imported
  `read_table_config` already but had **never imported or passed `inventory_config`** — see "Defects
  found in 06–08" below, this is the one fix folded into this prompt's own commit rather than raised
  as a separate issue).
- Added `--inventory` (`store_true`) and `--inventory-verbose` (`store_true`) flags.
- Added `inventory_config=inventory_config` to the `ShardedPool(...)` constructor call.
- Immediately after `) as pool:` and before any other code in the `with` block (units, tolerances,
  solvers, policies, wavenumber/redshift construction, `build_model_list`, the `run_pipeline` loop):
  ```python
  if args.inventory:
      print(
          format_inventory_report(pool, args.database, verbose=args.inventory_verbose)
      )
      sys.exit()
  ```
  `sys.exit()` inside the `with ShardedPool(...) as pool:` block raises `SystemExit`, which still runs
  `ShardedPool.__exit__` (a context manager's `__exit__` runs on any exception unless it suppresses
  it, and this one does not), so shard actors are still cleaned up. No compute-target object is
  constructed and no work queue is built before this check — `--inventory` reports and exits with
  nothing else executed, per the prompt's explicit requirement.

### Task 2 — the report

`tools/inventory_report.py` groups the 28 classes (13 replicated + 15 sharded, per
`IMPLEMENTATION_STATE.md` §5 note 7) into six categories rather than the prompt's illustrative four:

```
Datastore metadata            -- version, store_tag
Cosmology models               -- LambdaCDM, QCD_Cosmology, BackgroundModel, BackgroundModelValue
Grid definitions (...)         -- wavenumber, redshift, wavenumber_exit_time, tolerance
Solvers and policies           -- IntegrationSolver, GkSourcePolicy, QuadSourcePolicy
Compute targets                -- the 6 validated/unvalidated classes + GkSourcePolicyData,
                                   QuadSourceIntegral, OneLoopIntegral
Value tables                   -- the 6 sharded *Value classes
```

**Classification: IMPLEMENTATION CHOICE.** The prompt's own example groups by only four categories
and does not mandate a specific split; six was chosen because "Datastore metadata"/"Solvers and
policies" are distinct enough in kind from "Cosmology models"/"Grid definitions" that folding them in
would make those two categories a grab-bag. `BackgroundModelValue` (replicated, flat `{"count": n}`)
is placed under "Cosmology models" rather than "Value tables" since it is that category's per-(model,
z) child table, not a sharded high-volume table — grouping is by subject matter, not by
replicated/sharded status.

Three output shapes are dispatched on the returned dict's own structure (mirroring how
`ShardedPool.inventory()` itself sniffs labelled-vs-flat, per prompt 06's log), not on a hardcoded
per-class lookup:
1. **Labelled buckets** (`all(isinstance(v, dict) and "labels" in v for v in data.values())`) →
   `"@@ ClassName: N validated, M unvalidated | <range>"`, with the `"validated"` bucket always listed
   first (see "Bucket ordering" below), followed by an indented, sorted, thousands-separated label
   list — full under `--inventory-verbose`, otherwise the first 5 plus `"... and N more"`.
2. **Flat value list** (`"values" in data`) → `"ClassName: N values | <range> [unit=..., ...]"`
   followed by the same truncated/sorted value list. `wavenumber`'s optional `values_physical` /
   `values_physical_unit` keys (prompt 07's log) are folded into the `[...]` unit annotation rather
   than a second list, since they describe the *same* values in a second unit system, not a second set
   of rows.
3. **Flat count** (`"count" in data`) → `"ClassName: N rows | <range>"`, with any other numeric field
   (`wavenumber_exit_time`'s `distinct_wavenumbers`) appended as `, field=value`.

A **`(error: {e})`** catch wraps every individual `pool.inventory(class_name)` call — one class
raising does not abort the rest of the report, matching upstream's `try/except` pattern the prompt
cites and confirmed working in the empty-datastore run below (though no class in this tree's current
28 actually raises against either a populated or an empty store — see "Verification" item 5 for the
one case that reaches a real error message).

An explicit `"(empty)"` marker is appended whenever a class's row/value/label count is genuinely zero
— confirmed for every one of the 28 classes against a freshly-created, wholly empty datastore (see
Verification below), never a bare `0` sitting next to a `None`-turned-`"?"` range.

Sorting is by `str(value)` for both value lists and label lists — stable across runs (the same
underlying floats/strings/dicts always compare equal under `str()`), simple, and adequate for every
value type this campaign's 28 factories actually return (floats, strings, and flat json-safe dicts —
no nested datetimes or other non-`str`-comparable types appear inside a `values`/`labels` list
anywhere in the tree).

`pool.inventory(...)` is called directly with no `ray.get()` wrapper, per prompt 06's log recording
that both `ShardedPool.inventory()` branches already resolve to a value internally.

### Task 3 — `store_handler` note

No natural use found. The report only reads (`pool.inventory(...)`); it never mints a new datastore
object, so there is nothing for `store_handler`/`persist_handler` to hook into. Per the prompt's own
instruction, none was manufactured.

## Defects found in 06–08

**`main.py` never imported or passed `inventory_config` to `ShardedPool(...)`.** Prompt 06 added the
`inventory_config` parameter to `ShardedPool.__init__` and prompt 08 populated
`config/sharding.py`'s `inventory_config` dict, but no prompt's commit touched `main.py`'s own
`ShardedPool(...)` call site to actually pass it through (unlike `read_table_config`, which was
already wired at that call site before this campaign started). The practical consequence: before this
prompt, a real `main.py` run would have constructed a pool with `self._inventory_config = None`
unconditionally, and every one of the 15 sharded classes' `pool.inventory(...)` calls would have
raised `"ShardedPool: the inventory service is not configured"` regardless of `config/sharding.py`'s
content — the entire F2c sub-campaign (prompt 08) would have been dead on arrival in the one place
that actually runs it. **Fixed here, not raised as a §3 issue**: wiring the sole production call site
to use the report it is now building is squarely this prompt's own task, not a fix to another
prompt's already-closed commit — there was nothing to "leave alone" since prompt 09 is the first thing
that ever calls `pool.inventory()` from `main.py`. Confirmed fixed by the verification run below
(every sharded class's `pool.inventory(...)` call succeeds against the real wired pool).

**No other defects found.** All 28 classes' `inventory()` methods (13 replicated from prompt 07, 15
sharded from prompt 08) ran to completion with no exception, no `KeyError`, and no shape mismatch
against the real `ShardedPool.inventory()`/`Datastore.inventory()` dispatch machinery, across both a
populated and a completely empty real datastore (see Verification). Every return shape matched one of
this module's three recognised patterns with no need for a fourth. Nothing needed reworking in 06, 07,
or 08 — no §3 issue opened.

## Deviations from the prompt

### Bucket display order: "validated" before "unvalidated" — IMPLEMENTATION CHOICE

`ShardedPool.inventory()`'s labelled branch (prompt 06) builds the merged dict by iterating
`data_queue[0].keys()`, so its key order is whatever the factory happened to construct
(`{"validated": ..., "unvalidated": ...}` literal order, per every Group A/C factory in prompts 07/08
— see e.g. `BackgroundModel.inventory`'s `return {"validated": _bucket(True), "unvalidated":
_bucket(False)}`). A first pass at the formatter iterated `sorted(data.keys())` for determinism, which
put `"unvalidated"` first (alphabetically before `"validated"`) — technically stable across runs, but
backwards from the prompt's own example (`"412 validated, 3 unvalidated"`) and from what a reader
actually wants to see first. Changed to a fixed preferred order (`"validated"`, then `"unvalidated"`,
then anything else sorted) — still fully deterministic (the order is fixed, not data-dependent), and
matches the prompt's illustrative output. Caught and fixed during this prompt's own verification pass,
before committing.

### `values_physical`/`values_physical_unit` folded into the header, not a second list — IMPLEMENTATION CHOICE

The prompt does not specify how to render `wavenumber`'s caller-dependent extra keys (prompt 07's log
flags this as "the one class in this sub-campaign whose top-level key set is caller-dependent").
Chose to append `unit=...`/`+N physical values (unit=...)` to the same header line rather than
printing a second `values`-style list, since the physical values are a unit conversion of the same
raw values already listed, not a materially different set of rows — printing them as a second
truncated list would double the visual weight of one class for no added information. `main.py` never
supplies `units` to `pool.inventory("wavenumber")` (the report calls every class with no extra
`*args`/`**kwargs`), so `values_physical` never actually appears in this campaign's own report output
— this path is exercised only by direct unit testing of the formatter, not by the live verification
run below.

### No other deviations

Every other formatting decision (category membership, the three-shape dispatch, thousands separators,
short datetime format, `"–"` as the range separator, `"(empty)"` placement, the 5-item truncation
threshold with `"... and N more"`) follows the prompt's Task 2 requirements directly, with no
structural surprises once the real return shapes (documented in the 06/07/08 logs and confirmed by
reading every touched factory's `inventory()` body directly before writing the formatter) were in
hand.

## Verification performed

1. **`tools/inventory_report.py`, `tools/__init__.py`, and `main.py` parse; `black --check` clean.**
   `./venv/bin/python3 -m py_compile main.py tools/inventory_report.py tools/__init__.py` — exit 0.
   `./venv/bin/python3 -m black --check main.py tools/inventory_report.py tools/__init__.py` — "3
   files would be left unchanged" (one reformat pass needed on first write of each new file, then
   clean).
2. **Ran the report against a real, freshly-created, current-schema datastore** — this tree has none
   (`IMPLEMENTATION_STATE.md` §5 note 1), so one was built for this prompt: a throwaway harness
   (`scratchpad/verify_inventory_09.py`, not committed) starts a local Ray instance
   (`ray.init(num_cpus=4, ...)` — no existing cluster needed; confirmed Ray can bootstrap a local
   instance in this environment, contrary to every earlier prompt's "no Ray cluster available"
   framing, which was evidently never actually tried) and constructs a real 2-shard `ShardedPool`
   using the genuine `config/sharding.py` values (`replicated_tables`, `sharded_tables`,
   `read_table_config`, `inventory_config` — the same import main.py now uses). Populated via real
   `pool.object_get(...)` calls, exactly as `main.py`'s own preamble does: 2 tolerances, 7
   wavenumbers, 3 redshifts, 1 `IntegrationSolver`, 1 `store_tag`, 1 `GkSourcePolicy`, 1
   `QuadSourcePolicy`, and `config.model_list.build_model_list(pool, units)` for `LambdaCDM` +
   `QCD_Cosmology`. Sharded compute-target/value tables were left at zero rows (populating those for
   real needs actual physics compute, out of scope for this prompt) except for the one table used in
   check 4 below. **Full report output, pasted verbatim** (post-fix bucket ordering):
   ```
   == Datastore inventory: /Users/ds283/Documents/Code/SecondaryGWKit/scratchpad/inventory_test_populated.sqlite ==

      -- Datastore metadata
         version: 1 value
            inventory-verify-09
         store_tag: 1 value | 2026-09-04 01:22 – 2026-09-04 01:22
            inventory-verify-tag

      -- Cosmology models
         LambdaCDM: 1 value | 2026-09-04 01:22 – 2026-09-04 01:22
            {'name': 'Planck2018 68% central values TT+TE+EE+lowP+lensing+BAO', 'omega_m': 0.31110000000000004, 'omega_cc': 0.6889, 'h': 0.6766}
         QCD_Cosmology: 1 value | 2026-09-04 01:22 – 2026-09-04 01:22
            {'name': 'Planck2018 68% central values TT+TE+EE+lowP+lensing+BAO', 'omega_m': 0.31110000000000004, 'omega_cc': 0.6889, 'h': 0.6766, 'log10_max_z': 20.0}
         @@ BackgroundModel: 0 validated, 0 unvalidated (empty)
         BackgroundModelValue: 0 rows (empty)

      -- Grid definitions (wavenumbers, redshifts, tolerances)
         wavenumber: 7 values | 2026-09-04 01:22 – 2026-09-04 01:22 [unit=1/Mpc (comoving)]
            1.0
            10.0
            100.0
            200.0
            300.0
            ... and 2 more
         redshift: 3 values | 2026-09-04 01:22 – 2026-09-04 01:22
            0.0
            1.0
            5.0
         wavenumber_exit_time: 0 rows (empty), distinct_wavenumbers=0
         tolerance: 1 value | 2026-09-04 01:22 – 2026-09-04 01:22
            1e-08

      -- Solvers and policies
         IntegrationSolver: 1 value | 2026-09-04 01:22 – 2026-09-04 01:22
            {'label': 'solve_ivp+RK45', 'stepping': 0}
         GkSourcePolicy: 1 value | 2026-09-04 01:22 – 2026-09-04 01:22
            {'label': 'policy="maximize-WKB"-Levin-threshold="1.5"', 'Levin_threshold': 1.5, 'numeric_policy': 'maximize-WKB'}
         QuadSourcePolicy: 1 value | 2026-09-04 01:22 – 2026-09-04 01:22
            {'label': 'policy="maximize-WKB"-Levin-threshold="1.5"', 'Levin_threshold': 1.5, 'numeric_policy': 'maximize-WKB'}

      -- Compute targets
         @@ TkNumericIntegration: 0 validated, 0 unvalidated (empty)
         @@ TkWKBIntegration: 0 validated, 0 unvalidated (empty)
         @@ QuadSource: 0 validated, 0 unvalidated (empty)
         @@ GkNumericIntegration: 0 validated, 0 unvalidated (empty)
         @@ GkWKBIntegration: 0 validated, 0 unvalidated (empty)
         @@ GkSource: 0 validated, 0 unvalidated (empty)
         GkSourcePolicyData: 0 rows (empty)
         QuadSourceIntegral: 0 rows (empty)
         OneLoopIntegral: 0 rows (empty)

      -- Value tables
         TkNumericValue: 11 rows            <- see check 4 (after synthetic rows inserted)
         TkWKBValue: 0 rows (empty)
         QuadSourceValue: 0 rows (empty)
         GkNumericValue: 0 rows (empty)
         GkWKBValue: 0 rows (empty)
         GkSourceValue: 0 rows (empty)
   ```
3. **Ran the report against a second, completely fresh/empty datastore** (separate 2-shard
   `ShardedPool`, nothing populated beyond the automatic `version` row every `Datastore.__init__`
   creates). **Full output, pasted verbatim:**
   ```
   == Datastore inventory: /Users/ds283/Documents/Code/SecondaryGWKit/scratchpad/inventory_test_empty.sqlite ==

      -- Datastore metadata
         version: 1 value
            inventory-verify-09-empty
         store_tag: 0 values (empty)

      -- Cosmology models
         LambdaCDM: 0 values (empty)
         QCD_Cosmology: 0 values (empty)
         @@ BackgroundModel: 0 validated, 0 unvalidated (empty)
         BackgroundModelValue: 0 rows (empty)

      -- Grid definitions (wavenumbers, redshifts, tolerances)
         wavenumber: 0 values [unit=1/Mpc (comoving)] (empty)
         redshift: 0 values (empty)
         wavenumber_exit_time: 0 rows (empty), distinct_wavenumbers=0
         tolerance: 0 values (empty)

      -- Solvers and policies
         IntegrationSolver: 0 values (empty)
         GkSourcePolicy: 0 values (empty)
         QuadSourcePolicy: 0 values (empty)

      -- Compute targets
         @@ TkNumericIntegration: 0 validated, 0 unvalidated (empty)
         @@ TkWKBIntegration: 0 validated, 0 unvalidated (empty)
         @@ QuadSource: 0 validated, 0 unvalidated (empty)
         @@ GkNumericIntegration: 0 validated, 0 unvalidated (empty)
         @@ GkWKBIntegration: 0 validated, 0 unvalidated (empty)
         @@ GkSource: 0 validated, 0 unvalidated (empty)
         GkSourcePolicyData: 0 rows (empty)
         QuadSourceIntegral: 0 rows (empty)
         OneLoopIntegral: 0 rows (empty)

      -- Value tables
         TkNumericValue: 0 rows (empty)
         TkWKBValue: 0 rows (empty)
         QuadSourceValue: 0 rows (empty)
         GkNumericValue: 0 rows (empty)
         GkWKBValue: 0 rows (empty)
         GkSourceValue: 0 rows (empty)
   ```
   All 28 classes reported cleanly: no exception anywhere in the output (`assert "(error" not in
   empty_report` passed), and no bare `0` sitting next to a `None`-turned-range — every zero-row/value
   class carries an explicit `"(empty)"` marker.
4. **Count cross-check against direct SQL, across real shard files** — the check the prompt calls out
   specifically as proving prompt 06's `"sum"` merge policy actually works. Inserted synthetic rows
   directly into the real `TkNumericValue` table (bypassing `object_get`, since populating it for real
   needs full physics compute) on both of the populated pool's two physical shard `.sqlite` files: 7
   rows on shard 0, 4 rows on shard 1. Then:
   - Direct SQL, by hand, on each shard file: `SELECT COUNT(*) FROM TkNumericValue` → 7 and 4.
   - `pool.inventory("TkNumericValue")` → `{"count": 11}`.
   - `7 + 4 == 11` confirmed by assertion in the harness (`assert reported["count"] ==
     direct_sql_total`), not just by eye. This is the real, live `ShardedPool.inventory()` fan-out
     (`ray.get([shard.inventory.remote(...) ...])` across two actual Ray-actor shards) plus
     `_merge_queue`'s `"sum"` policy — not a hand-simulated reproduction of the dispatch logic, unlike
     every one of prompts 06/08's own verification passes (which had no live Ray cluster available and
     said so explicitly). This closes the outstanding half of the
     `[08-inventory-sharded-factories]` issue in `IMPLEMENTATION_STATE.md` §3 that asked for
     exactly this check "against a real multi-shard datastore" — see §3 update below.
5. **Truncation and `--inventory-verbose` behaviour, confirmed by assertion, not just by reading
   output.** With 7 wavenumbers populated: non-verbose output shows exactly 5 values (`1.0` through
   `300.0`) followed by `"... and 2 more"`; `--inventory-verbose`-equivalent (`verbose=True`) shows all
   7 with no `"..."` anywhere in that class's block. Both asserted directly in the harness
   (`assert any("... and 2 more" in l for l in non_verbose_block)`, `assert not any("..." in l for l
   in verbose_block)`, `assert sum(1 for l in verbose_block if l.strip()[0].isdigit()) == 7`), not
   inferred from a read-through.
6. **A class with no `inventory()` method produces the intended, specific error — checked at both
   levels the report can reach it through.**
   - Directly against a real (non-Ray-wrapped) `Datastore` instance (same
     `__ray_metadata__.modified_class` technique documented in prompts 04's and 06's logs), calling
     `.inventory("BackgroundModel_tags")` (a genuine tag-association factory with no `inventory`
     method, confirmed by reading `BackgroundModel.py:20-47` directly — `sqla_BackgroundModelTagAssociation_factory`
     has no `inventory` staticmethod) — raised exactly `RuntimeError: Datastore: the object factory
     for "BackgroundModel_tags" does not provide an inventory service`, the message the prompt's
     verification step 5 asks to confirm.
   - **Finding, not a defect**: through `ShardedPool.inventory(...)` (the level this report actually
     calls), the *same* class raises a *different* message —
     `RuntimeError: Unable to dispatch inventory() for item of type "BackgroundModel_tags"` —
     confirmed against a minimally-constructed `ShardedPool` (`ShardedPool.__new__(ShardedPool)` with
     `_replicated_tables`/`_sharded_tables`/`_inventory_config` populated from the real
     `config/sharding.py`, the same fallback technique prompts 04's and 06's logs use). This is
     because **no tag-association table is registered in either `replicated_tables` or
     `sharded_tables`** in `config/sharding.py` — `ShardedPool.inventory()`'s dispatch never reaches
     the sharded/replicated branch for such a class at all, so it never gets as far as calling
     `Datastore.inventory()` (whose `hasattr(factory, "inventory")` guard is what actually produces the
     "does not provide an inventory service" message). **Consequence for this report**: none of the
     28 classes it queries can ever hit this failure mode in practice (every registered replicated and
     sharded class in this tree already has an `inventory()` method, confirmed by prompts 07/08's own
     100%-coverage grep counts), and no *other* class this report could plausibly be extended to cover
     (a tag-association table) is reachable through `pool.inventory(...)` at all — it would need adding
     to `replicated_tables`/`sharded_tables` first, which is out of scope for every prompt in this
     campaign. Not filed as a §3 issue (nothing is broken; the report's own `(error: ...)` fallback
     still handles the `RuntimeError` correctly either way — confirmed by construction, since both
     messages are still just `RuntimeError`s the report's `try/except` catches identically), but
     recorded here since it means verification step 5's literal instruction ("via the report, if it
     can reach one") cannot be satisfied for any class currently in this tree — only the direct
     `Datastore.inventory()` call in the first bullet above exercises the exact message the prompt
     names.
7. **Scratch harness and scratch database files removed after the run** (`scratchpad/verify_inventory_09.py`
   left in place, not committed, per the campaign's established convention; every `.sqlite`/shard file
   it created was deleted at the end of the script and confirmed absent via `git status`/`ls
   scratchpad/`).

## Observations not acted on

- **The `main.py`/`inventory_config` wiring gap** (see "Defects found in 06–08") is now fixed, not
  merely observed — recorded there rather than here.
- **Tag-association tables cannot produce the "does not provide an inventory service" message through
  `pool.inventory(...)`** (see Verification item 6) — a real but harmless gap in what the *dispatch
  path* can diagnose, not in what this report needs. No action taken: none of the 28 in-scope classes
  are affected, and extending `replicated_tables`/`sharded_tables` to cover tag-association tables
  would be a scope change well beyond this prompt (and beyond this campaign's stated F2 surface).
- **`values_physical`/`values_physical_unit` formatting is untested against live data** (see
  Deviations) — `main.py`'s report never supplies `units` to `pool.inventory("wavenumber")`, so this
  path is exercised only by reading the formatter's code, not by a live run. If a future caller wants
  physical-unit wavenumbers in the report, `main.py`'s call site would need to pass `units=units` (it
  already has a `units` object in scope at the point the report is invoked, per the `with
  ShardedPool(...) as pool:` preamble) — a one-line change, not attempted here since the prompt's
  "do not manufacture a use" guidance (Task 3) reads naturally as applying to this kind of
  speculative enhancement too.

## State handed to the next prompt

- **F2 is now complete end to end.** All four `inventory()` sub-campaign prompts (06 plumbing, 07
  replicated factories, 08 sharded factories + merge config, 09 this reporting entry point) are done,
  and this prompt's own verification run is the first (and only, so far) time the full stack —
  `main.py`'s `--inventory` flag → `ShardedPool.inventory()` → per-shard `Datastore.inventory()` →
  factory `inventory()` → `_merge_queue` → `tools/inventory_report.py`'s formatting — has been
  exercised together, against a real multi-shard, Ray-actor-backed pool, both populated and empty.
- **The `[08-inventory-sharded-factories]` issue in `IMPLEMENTATION_STATE.md` §3** asked for exactly
  the check this prompt's Verification item 4 performed (a real multi-shard `ShardedPool.inventory()`
  call, confirming a `count` is summed rather than reflecting one shard) — that half of the issue is
  now closed; see the §3/§4 board update. The issue's other two example classes (a labelled Group A
  class, a flat Group B class) were not separately re-verified against a *live* Ray cluster in this
  prompt (Verification items 2/3 populate them only as empty datastores) — prompt 10 can close the
  remainder if it wants a live-Ray check for those two groups specifically, though the merge logic
  itself is now confirmed live for the numeric/`"sum"` case that mattered most (per the audit's own
  §2.3(b) emphasis on this being the one policy `SI` cannot express).
- **`ray.init()` can bootstrap a fully local, single-machine Ray instance in this environment with no
  pre-existing cluster** (`ray.init(num_cpus=N, include_dashboard=False)`, no `address=` needed) —
  every earlier prompt in this campaign (01, 04, 05, 06, 08) recorded "no live Ray cluster is
  available" and fell back to hand-simulating dispatch logic or using bare (non-Ray) `Datastore`
  instances. That framing was accurate for *connecting to an existing cluster* but this prompt found
  that starting a fresh local one works fine and is enough to exercise real multi-actor dispatch,
  `ray.get()` fan-out, and genuine `ShardedPool`/`Datastore` actors end-to-end. **Prompt 10 (or anyone
  chasing the remaining `[01-shard-key-persistence]`/`[04-read-table-service]`/
  `[05-persist-handler-split]` §3 issues, all of which are blocked on "needs a live Ray cluster") should
  try this before assuming a live-Ray check is out of reach** — a local `ray.init()` is a real,
  multi-actor Ray runtime (multiple shard actors, a broker actor, genuine `.remote()`/`ray.get()`
  calls), not a simulation, even though it runs on a single machine with no separate cluster to connect
  to. One caveat found while doing this: `ShardedPool` registers its broker actor under the fixed name
  `"SerialPoolBroker"`, so **two `ShardedPool`s cannot coexist in the same Ray runtime** — a second one
  needs `ray.shutdown()` / fresh `ray.init()` first (this prompt's harness does this between its
  populated-pool and empty-pool runs).
- **`tools/` is now a proper package** (`tools/__init__.py` added, previously absent) — prompt 01's
  `tools/shard_key_audit.py` was only ever invoked as `python tools/shard_key_audit.py`, never
  imported as `tools.shard_key_audit`, so the absence of `__init__.py` was never exercised before
  `tools/inventory_report.py` needed `from tools.inventory_report import format_inventory_report` to
  work from `main.py`.
- `IMPLEMENTATION_STATE.md` §5 note 11's no-self-referential-SHA convention followed: no SHA embedded
  in this log's header or in the status board's prompt-09 row.
