# Log 02 — The sweep prepares through the registry, and checks shards through the resolver

**Prompt:** [`prompts/store-retirement/02-the-sweep-prepares-through-the-registry.md`](../02-the-sweep-prepares-through-the-registry.md)
**Commit:** *(this commit)* — "Make quadsource_atol_sweep prepare and check through the registry"
**Model:** Claude Sonnet 5
**Date:** 2026-09-25
**Result:** COMPLETE. No §5 stop condition was met. `prepare()` now makes the sweep store with
one `RunRegistry.stores.copy_store` call, so a tombstone at the sweep name is never overwritten.
`assert_store_is_self_consistent` now compares serial by serial through
`Datastore.shard_paths.resolve_shard_path`, and the three real primaries, all legacy-absolute,
still pass it. Closes `datastore-portability`'s `[03-atol-sweep-prepare-writes-its-store-sidecar-by-hand]`
and `[01-atol-sweep-check-expects-absolute-shard-records]`.

## What shipped

**`docs/handover/quadsource_atol_sweep.py`** — `prepare()`, `assert_store_is_self_consistent`, and
`--force`'s handling and help, only. Full diff below.

- **`prepare(force=False)`.** The hand copy of the four shards and the primary
  (`shutil.copy2` x5), the `UPDATE shards set filename = ?` re-pointing loop, and the
  hand-written `manifest.write_text(json.dumps(...))` sidecar are gone. In their place, one call:

      import RunRegistry.stores
      RunRegistry.stores.copy_store(BASELINE_STORE, SWEEP_STORE, purpose=<the old purpose text, verbatim>)

  `copy_store` copies the store's files with `ShardedPool.copy_store`, then writes the
  destination's sidecar as a new, problem-free registry sidecar (new `store_id`, `copied_from`
  naming the baseline's `store_id`, a `copy` history entry). It refuses, before anything is
  written, if the destination's sidecar name is already taken — by an existing store, or by a
  tombstone `RunRegistry store retire` (prompt 03) will one day leave behind. That refusal is
  exactly what a retired name needs: a plain `--prepare` at a retired name must not silently
  replace the tombstone. A `RuntimeError` from `copy_store` is re-raised with a
  `quadsource_atol_sweep:` prefix, text intact.
- **`--force`.** `copy_store` never overwrites, so `--force` can no longer mean "replace". The
  flag is kept (an old command line fails loudly rather than being silently reinterpreted), but
  `prepare()` now refuses immediately when `force=True`, before reading anything, with a message
  saying that a sweep store is no longer replaced and that the remedy is a name that has never
  held a store (or, once it exists, `store retire` first). `--force`'s `argparse` help says the
  same in one line.
- **`assert_store_is_self_consistent(primary)`.** Reads the `shards` table `mode=ro`, as before,
  but now resolves each stored record through `Datastore.shard_paths.resolve_shard_path(primary,
  stored)` and compares it, serial by serial, with the expected sibling
  `shard_paths(primary)[serial]`. It requires exactly the serials `0 .. SHARDS-1`: an extra or a
  missing serial is refused explicitly, separately from a mismatched one, and the refusal names
  each. A bare-name store and a legacy store whose absolute records name its own siblings are
  accepted alike, because the resolver treats them the same way (§ below explains the backup's
  shape specifically). The docstring drops the false claim that `ShardedPool` reads shard records
  "as absolute paths" — it has read every record through the resolver since
  `datastore-portability` prompt 01 — while keeping the account of the 2026-09-23 accident, which
  is still why the check exists. The refusal no longer says "Re-run --prepare --force", which is
  wrong for a build and, after this prompt, for `prepare()` too; it says neutrally that the store
  is not safe to use, and names the two remedies that apply across its three callers (a fresh
  `--prepare` for the sweep store; a different `--database` or an investigation for a `--build`
  target).

**`RunRegistry/tests/test_quadsource_atol_sweep_prepare.py`** (new, 11 tests). Imports the script
by path with `importlib.util`; `BASELINE_STORE` / `SWEEP_STORE` are monkeypatched onto the loaded
module, pointed at a temporary directory per test, and `RunRegistry.stores.DEFAULT_ROOT` is
monkeypatched at a temporary runs root in the same directory, so the running-run check never
reads `var/runs/`. No Ray; stores are built with `Datastore/tests/shard_store_fixtures.py`.

1. `test_a_fresh_sweep_name_is_made_through_the_registry` — §3 test 1.
2. `test_a_tombstone_at_the_sweep_name_is_never_overwritten` — §3 test 2, with and without
   `force=True`.
3. `test_an_existing_sweep_store_is_never_replaced` — §3 test 3, with and without `force=True`.
4. `test_the_check_accepts_bare_names`, `test_the_check_accepts_a_legacy_store_naming_its_own_siblings`,
   `test_the_check_accepts_the_backups_shape`, `test_the_check_refuses_a_renamed_record`,
   `test_the_check_refuses_a_missing_serial`, `test_the_check_refuses_an_extra_serial`,
   `test_the_check_refuses_two_swapped_serials` — §3 test 4, the last being the case mutation
   (iv) needs and that test 4 did not already have.
5. `test_the_build_resume_path_accepts_a_store_written_by_write_new_store` — §3 test 5. No build
   is run; this calls the check function directly, as `run_build`'s pre-resume branch would.

Test 6 (no Ray, `test_store_fingerprint.py` test 12 unmodified and passing) is not a test in this
module — nothing here imports Ray, and test 12 is verified by running the full `RunRegistry` suite
(below), since the script's `run.finish(` calls are untouched.

## Why the check accepts the backup's shape

`test_the_check_accepts_the_backups_shape` builds a primary at `.../Y/store.sqlite` whose `shards`
table holds, as its legacy absolute records, the paths of another, *actually existing* primary's
shards at `.../X/store-shard000N.sqlite` — the shape audit §2.7 found for real: the backup's
primary names the **live** A3 store's shards, byte for byte. `resolve_shard_path` reads a legacy
absolute record by its **final component only**, and resolves it beside the primary being
checked, never beside the directory the stored string names. So the check reads `Y`'s own
`store-shard000N.sqlite` files — which the test also creates — and passes, regardless of what
sits at `X`. This is the one property `datastore-portability` prompt 01 exists to guarantee: a
copied or backed-up primary's stale absolute records are never followed to the original they were
copied from. The check could not do its job at all without it, because the alternative (comparing
literal absolute paths, mutation ii) is exactly what would make it read `X`'s files as `Y`'s, and
is the failure mode audit §2.7 calls out as this campaign's most important deliberate-breakage
case for the eventual deletion code (prompt 01 of this campaign, not this one).

## Deviations from the prompt

1. **`import RunRegistry.stores`, not bare `import RunRegistry` — STRUCTURALLY REQUIRED.** The
   prompt says "Import RunRegistry inside `prepare()`, as the script already imports it where it
   registers runs, so that the module's own imports do not change." Elsewhere in the script,
   `import RunRegistry` works because the only thing called afterwards, `RunRegistry.begin(...)`,
   is defined directly in `RunRegistry/__init__.py`. `stores` is a **submodule**: `RunRegistry/__init__.py`
   does not import it at module scope (by design — its own docstring says `import RunRegistry`
   must load neither `ray` nor `sqlalchemy`, and `stores.copy_store`/`move_store` import
   `ShardedPool`, which does), so a bare `import RunRegistry` does not make `RunRegistry.stores`
   an attribute unless something else in the process already imported that submodule. Checked
   directly: a fresh interpreter with only `import RunRegistry` has `hasattr(RunRegistry,
   "stores") == False`. `prepare()` calls `RunRegistry.stores.copy_store(...)` with no `begin()`
   call beforehand to prime that attribute, so `import RunRegistry.stores` is what actually works,
   and is still exactly one import of the top-level `RunRegistry` package (the same style
   `RunRegistry/__main__.py` uses: `from . import stores`). This costs nothing the prompt cares
   about — the module's own top-level imports are unchanged either way — and keeps the one-line
   call the prompt asks for.
2. **The `--force` refusal message, and the check's refusal message, are original text, not
   verbatim from any source — IMPLEMENTATION CHOICE.** The prompt specifies what each message must
   say, not its exact wording. Both were written to satisfy the prompt's requirements (§2's "the
   message says that a sweep store is no longer replaced"; §2's "Its refusal says what to do for
   each of its three callers, or says it neutrally") as directly as possible.
3. **Test module structure and naming — IMPLEMENTATION CHOICE.** The prompt asks for "a new module
   in `RunRegistry/tests/`" without naming it; `test_quadsource_atol_sweep_prepare.py` was chosen
   to name what it tests. The class subclasses `RunRegistry.tests.store_fixtures.StoreTestCase`
   for its temporary-directory and runs-root machinery (`setUp`, `state()`), but does not reuse its
   `a_store`/`a_registry_store` helpers, which are hard-wired to that module's own `N_SHARDS = 3`;
   this script's `SHARDS = 4`, so this module's own fixture helpers build stores at
   `self.sweep.SHARDS` instead.

No other deviation was made. `git diff` of the script touches only `prepare()`,
`assert_store_is_self_consistent`, `--force`'s handling and help, and their docstrings and
messages, matching acceptance §4 item 4.

## Verification performed

`black --check` is clean on both changed/new files.

| Suite | Before (baseline) | After |
|---|---|---|
| `Datastore` | 177 OK (`42d4910`) | 177 OK |
| `RunRegistry` | 117 OK (`42d4910`) | 128 OK (117 + this prompt's 11) |
| `ComputeTargets` | 552 OK (`50a24ac`; `test_tk_wkb_phase.TestCost.test_wall_time_per_object` a known wall-clock flake) | 552 OK |

`AdaptiveLevin`, `CosmologyModels` and `LiouvilleGreen` were not re-run: nothing in their import
graph touches `docs/handover/quadsource_atol_sweep.py`, `RunRegistry` or `Datastore/shard_paths.py`,
and the prompt asks only for RunRegistry, Datastore and ComputeTargets "if you can" (done).

Commands:

    PYTHONPATH=. ./venv/bin/python -m unittest discover -s RunRegistry/tests -t . 2>&1 | tail -40
    PYTHONPATH=. ./venv/bin/python -m unittest discover -s Datastore/tests -t . 2>&1 | tail -40
    PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -40

`RunRegistry.tests.test_store_fingerprint.TestDrivers.test_the_finish_calls` (test 12) is included
in the `RunRegistry` run above and passed: the script's `run.finish(` calls were not touched.

### The read-only check on the three real primaries

Ran the new `assert_store_is_self_consistent` against the three real primaries in
`var/datastores/`, `mode=ro`, with a before/after table of size, mtime (ns) and SHA-256 for each
primary file, and a directory listing of `var/datastores/` and the backup directory, before and
after. The throwaway script lives at
`/private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/146aab00-cec5-40eb-91b1-89a122f1e971/scratchpad/impl02_real_check.py`
and was never committed. `python -m RunRegistry list` showed nothing `running` beforehand, and
`var/runs/` (11 entries: 7 registered runs, `a3-pilot`, `a3-pilot-resume`, and 2 loose files) was
unchanged throughout — nothing here begins a run.

| Primary | Result | Size (bytes) | mtime (ns) | SHA-256 |
|---|---|---|---|---|
| `handover-A3-baseline-lambdacdm.sqlite` (live A3) | **PASS**, unchanged | 32768 → 32768 | 1789934590990297815 → 1789934590990297815 | `fdbe93a5…` → `fdbe93a5…` |
| `handover-atol-sweep.sqlite` (sweep) | **PASS**, unchanged | 32768 → 32768 | 1790163664518981413 → 1790163664518981413 | `c3cda49d…` → `c3cda49d…` |
| `backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite` (backup) | **PASS**, unchanged | 32768 → 32768 | 1789934590990297815 → 1789934590990297815 | `fdbe93a5…` → `fdbe93a5…` |

All three pass, as expected: each holds legacy absolute records naming its own siblings by name
(audit §1, §2.7). The live A3 primary and the backup's primary are byte-for-byte identical
(same size, mtime and SHA-256), which matches the audit's finding that the backup's primary is a
straight, unmodified copy of the live primary. Every size, mtime and SHA-256 is unchanged, and the
directory listings of `var/datastores/` and the backup directory are unchanged. Nothing under
`var/` was written; `prepare()`, `copy_store`, `--build` and every other writer were never called
against a real store or name.

### What a fresh sweep sidecar would carry (audit §2.12, `[00-a-copy-carries-its-sources-present-tense-fields]`)

Read (read-only) the live A3 sidecar's unknown fields, which `copy_store` carries verbatim into
any copy: `backup`, `driver`, `git_dirty`, `git_head`, `grid_criterion`, `note`, `restart`,
`run_history`, `scope`, `status_files`. Three of these are present-tense claims that would be
false or misleading in a sweep-store copy, exactly as the audit found: `backup` (`"retained":
true`, describing a backup *of the source*), `restart.command` (whose `--database` names the
**live** store), and `run_history` (the live store's runs, not the sweep's). This prompt does not
change `copy_store`'s carrying behaviour (§6 of the prompt: "it does not change what a copy
carries from its source") and fixes nothing here; the existing issue,
`[00-a-copy-carries-its-sources-present-tense-fields]`, already covers it and is left as is.

## The deliberate-breakage record

Each mutation was applied to `docs/handover/quadsource_atol_sweep.py`, the affected tests run,
and the file restored byte-for-byte (`diff -q` against the pre-mutation file after every restore)
before the next mutation or the final commit. None was committed.

**(i) `ShardedPool.copy_store` called directly, sidecar written by hand as before.** Test 2 must
fail (the tombstone is overwritten).

    --- a/docs/handover/quadsource_atol_sweep.py
    +++ b/docs/handover/quadsource_atol_sweep.py
    @@ -674,16 +674,28 @@
             "retire` exists (store-retirement prompt 03), retire the old one first."
         )

    -    import RunRegistry.stores
    -
    -    try:
    -        RunRegistry.stores.copy_store(
    -            BASELINE_STORE,
    -            SWEEP_STORE,
    -            purpose=(
    -                "Working copy of the A3 baseline store for "
    -                "docs/handover/quadsource_atol_sweep.py. Disposable: every row that is not "
    -                "at the production tolerance pair belongs to a sweep and nothing else reads "
    -                "it. The comparator is handover-A3-baseline-lambdacdm, not this."
    -            ),
    +    # --- MUTATION (i), deliberate breakage: bypass the registry entirely ---
    +    import json
    +    import time
    +
    +    from Datastore.SQL.ShardedPool import ShardedPool
    +
    +    ShardedPool.copy_store(BASELINE_STORE, SWEEP_STORE)
    +    manifest = SWEEP_STORE.with_suffix(".manifest.json")
    +    manifest.write_text(
    +        json.dumps(
    +            {
    +                "name": SWEEP_STORE.stem,
    +                "purpose": "mutation (i): hand-written sidecar",
    +                "datastore": str(SWEEP_STORE),
    +                "copied_from": str(BASELINE_STORE),
    +                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    +            },
    +            indent=2,
             )
    -    except RuntimeError as e:
    -        raise RuntimeError(f"quadsource_atol_sweep: {e}") from e
    +        + "\n"
    +    )
    +    # --- end MUTATION (i) ---
         assert_store_is_self_consistent(SWEEP_STORE)
         print(f"** quadsource_atol_sweep: sweep store ready at {SWEEP_STORE}")

Tests that failed: `test_a_fresh_sweep_name_is_made_through_the_registry` (the resulting sidecar
reads as `legacy`, not a problem-free registry sidecar — `copy_store`'s registry-sidecar guarantee
is gone) and `test_a_tombstone_at_the_sweep_name_is_never_overwritten` (no `RuntimeError` is
raised; the tombstone is silently replaced). This is exactly the failure the prompt names.

**(ii) The check compares against literal absolute paths, as before.** Tests 4 and 5 must fail.

    --- a/docs/handover/quadsource_atol_sweep.py
    +++ b/docs/handover/quadsource_atol_sweep.py
    @@ -611,46 +611,23 @@
         no more and no fewer. A bare-name store and a legacy store whose absolute records name its own
         siblings are accepted alike, since the resolver treats them the same way.
         """
    +    # --- MUTATION (ii), deliberate breakage: literal absolute paths, no resolver, no serials ---
         import sqlite3

    -    from Datastore.shard_paths import resolve_shard_path
    -
    -    expected = {i: path.resolve() for i, path in enumerate(shard_paths(primary))}
    +    expected = {str(p.resolve()) for p in shard_paths(primary)}
         conn = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
         try:
    -        found = dict(conn.execute("select serial, filename from shards"))
    +        found = {row[1] for row in conn.execute("select serial, filename from shards")}
         finally:
             conn.close()
    -
    -    missing = sorted(set(expected) - set(found))
    -    extra = sorted(set(found) - set(expected))
    -    mismatched = []
    -    for serial in sorted(set(expected) & set(found)):
    -        resolved = resolve_shard_path(primary, found[serial])
    -        if resolved != expected[serial]:
    -            mismatched.append((serial, found[serial], resolved, expected[serial]))
    -
    -    if missing or extra or mismatched:
    -        lines = [
    +    if found != expected:
    +        raise RuntimeError(
                 f"quadsource_atol_sweep: the `shards` table inside {primary.name} does not name its "
    -            "own shard files, so a pool opened on it would read and write somewhere else."
    -        ]
    -        if missing:
    -            lines.append(f"  missing serial(s): {missing}")
    -        if extra:
    -            lines.append(f"  extra serial(s), not among 0..{SHARDS - 1}: {extra}")
    -        for serial, stored, resolved, exp in mismatched:
    -            lines.append(
    -                f"  serial {serial}: record {stored!r} resolves to {resolved}, expected {exp}"
    -            )
    -        lines.append(
    -            "Not safe to use until this is corrected. If this is the sweep store, make a fresh "
    -            "one with --prepare, at a name that has never held a store: this script never "
    -            "repairs a store in place. If this is a --build --database target, its shards table "
    -            "was not written the way this script expects; choose a different --database or "
    -            "investigate how this one was made."
    +            f"own shard files (mutation ii).\n"
    +            f"  expected: {sorted(expected)}\n"
    +            f"  found:    {sorted(found)}\n"
             )
    -        raise RuntimeError("\n".join(lines))
    +    # --- end MUTATION (ii) ---

Tests that failed: `test_the_check_accepts_the_backups_shape` (now refused — literal absolute-path
comparison follows a stored record into the *other* directory it names, which is precisely the
accident this check exists to prevent, audit §2.7), `test_the_check_refuses_a_renamed_record`,
`test_the_check_refuses_a_missing_serial` and `test_the_check_refuses_an_extra_serial` (all three
are still refused, but the message text the tests assert on — the per-serial diagnostic — is gone,
since the mutated check reports only two flat lists of strings), and
`test_the_check_refuses_two_swapped_serials` (no `RuntimeError` at all: swapping two records
between two serials leaves the *set* of stored absolute-path strings identical, so a plain set
comparison sees nothing wrong — a second demonstration of the blind spot mutation (iv) is written
to show, arrived at differently here, because absolute-path comparison never distinguishes serials
in the first place). Two further tests failed as collateral, for the reason
`[01-atol-sweep-check-expects-absolute-shard-records]` was opened for: a modern, bare-name store
(what `prepare()` now makes, and what `run_build` writes) is refused outright by
literal-absolute-path comparison, since its bare-name records never equal the absolute paths this
mutation expects — `test_the_check_accepts_bare_names`,
`test_the_build_resume_path_accepts_a_store_written_by_write_new_store` (`run_build`'s pre-resume
case, also a bare-name store), and `test_a_fresh_sweep_name_is_made_through_the_registry`
(`prepare()`'s own post-copy call to the check). Eight tests failed in total, matching the
`failures=4, errors=4` unittest reported.

**(iii) `force=True` falls through to the copy.** Test 2's `force` case or test 3 must fail.

    --- a/docs/handover/quadsource_atol_sweep.py
    +++ b/docs/handover/quadsource_atol_sweep.py
    @@ -664,16 +664,7 @@
         refusal is exactly what a retired name needs -- a plain `--prepare` at a retired name must not
         silently replace the tombstone that records what was there.
         """
    -    if force:
    -        raise RuntimeError(
    -            'quadsource_atol_sweep --prepare: --force no longer means "replace". '
    -            "RunRegistry.stores.copy_store never overwrites a destination whose sidecar name is "
    -            "already taken -- by an existing store or by a retired store's tombstone -- and "
    -            "prepare() no longer either. A sweep store, once made, is not replaced: point "
    -            "SWEEP_STORE at a name that has never held a store, or, once `RunRegistry store "
    -            "retire` exists (store-retirement prompt 03), retire the old one first."
    -        )
    -
    +    # --- MUTATION (iii), deliberate breakage: force=True falls through to the copy ---
         import RunRegistry.stores

         try:

Tests that failed: `test_a_tombstone_at_the_sweep_name_is_never_overwritten` (its `force=True`
case: `copy_store` itself still refuses, since its destination sidecar name is still taken, so no
overwrite happens — but the expected message `"no longer means"` is gone, replaced by
`copy_store`'s own "already exist" text) and `test_an_existing_sweep_store_is_never_replaced`
(same reason, on its `force=True` case). This is the failure the prompt names: `force=True` no
longer gets `prepare()`'s own explanation, and instead falls through into the same call path as
`force=False`.

**(iv) The check compares sets of resolved paths, ignoring serials.** A test with two serials'
records swapped must fail; test 4 already has `test_the_check_refuses_two_swapped_serials` for
this, added as part of §3's test 4 rather than as an afterthought.

    --- a/docs/handover/quadsource_atol_sweep.py
    +++ b/docs/handover/quadsource_atol_sweep.py
    @@ -611,46 +611,27 @@
         no more and no fewer. A bare-name store and a legacy store whose absolute records name its own
         siblings are accepted alike, since the resolver treats them the same way.
         """
    +    # --- MUTATION (iv), deliberate breakage: sets of resolved paths, serials ignored ---
         import sqlite3

         from Datastore.shard_paths import resolve_shard_path

    -    expected = {i: path.resolve() for i, path in enumerate(shard_paths(primary))}
    +    expected = {path.resolve() for path in shard_paths(primary)}
         conn = sqlite3.connect(f"file:{primary}?mode=ro", uri=True)
         try:
    -        found = dict(conn.execute("select serial, filename from shards"))
    +        found_raw = dict(conn.execute("select serial, filename from shards"))
         finally:
             conn.close()
    -
    -    missing = sorted(set(expected) - set(found))
    -    extra = sorted(set(found) - set(expected))
    -    mismatched = []
    -    for serial in sorted(set(expected) & set(found)):
    -        resolved = resolve_shard_path(primary, found[serial])
    -        if resolved != expected[serial]:
    -            mismatched.append((serial, found[serial], resolved, expected[serial]))
    -
    -    if missing or extra or mismatched:
    -        lines = [
    +    found = {resolve_shard_path(primary, v) for v in found_raw.values()}
    +
    +    if found != expected:
    +        raise RuntimeError(
                 f"quadsource_atol_sweep: the `shards` table inside {primary.name} does not name its "
    -            "own shard files, so a pool opened on it would read and write somewhere else."
    -        ]
    -        if missing:
    -            lines.append(f"  missing serial(s): {missing}")
    -        if extra:
    -            lines.append(f"  extra serial(s), not among 0..{SHARDS - 1}: {extra}")
    -        for serial, stored, resolved, exp in mismatched:
    -            lines.append(
    -                f"  serial {serial}: record {stored!r} resolves to {resolved}, expected {exp}"
    -            )
    -        lines.append(
    -            "Not safe to use until this is corrected. If this is the sweep store, make a fresh "
    -            "one with --prepare, at a name that has never held a store: this script never "
    -            "repairs a store in place. If this is a --build --database target, its shards table "
    -            "was not written the way this script expects; choose a different --database or "
    -            "investigate how this one was made."
    +            f"own shard files (mutation iv).\n"
    +            f"  expected: {sorted(str(p) for p in expected)}\n"
    +            f"  found:    {sorted(str(p) for p in found)}\n"
             )
    -        raise RuntimeError("\n".join(lines))
    +    # --- end MUTATION (iv) ---

Tests that failed: `test_the_check_refuses_two_swapped_serials` (no `RuntimeError` — the set of
resolved paths is unchanged by the swap, exactly the case the prompt describes), plus
`test_the_check_refuses_a_renamed_record`, `test_the_check_refuses_a_missing_serial` and
`test_the_check_refuses_an_extra_serial` (still refused, but with the mutation's own message text,
not the per-serial diagnostic the real tests assert on).

After each mutation the file was restored from a saved copy of the committed version and
`diff -q` confirmed byte-identity before continuing; the working tree carries none of them.

## Observations not acted on

- **`shutil` and `json` are now unused imports** at module scope in
  `docs/handover/quadsource_atol_sweep.py` (both were used only by the old `prepare()`). Not
  removed: acceptance §4 item 4 restricts the diff to `prepare()`, `assert_store_is_self_consistent`,
  `--force`'s handling and help, and their docstrings and messages, and the top-level import block
  is none of those. No test or tool in this repository flags an unused import (`black` does not),
  so this costs nothing today. Not worth a §3 issue on its own; whoever next touches this script's
  imports for another reason can drop them then.
- **The sweep sidecar a future `--prepare` makes will carry the live store's `backup`, `restart`
  and `run_history` fields verbatim** (§ above). This is exactly
  `[00-a-copy-carries-its-sources-present-tense-fields]`, already open on this board's §3 from the
  audit; nothing new to add.
- **`copy_store`'s destination-taken refusal text does not name which of the several possible
  destination artefacts collided** (it lists `dst_sidecar`, its `.tmp`, and — for a move — the
  `.incomplete-move` names, but for a *copy* only the sidecar and its `.tmp` are checked, and the
  message already names them individually). Read while tracing the refusal path in
  `RunRegistry/stores.py` `_prepare` (`:644-679`); it is already precise enough for this prompt's
  purposes and is not `quadsource_atol_sweep.py`'s code to change.

## State handed to the next prompt

- `docs/handover/quadsource_atol_sweep.py --prepare` now refuses at any name whose sidecar exists
  — an existing store or a future tombstone — which is exactly the precondition prompt 03's
  `store retire` needs: once a store is retired, nothing in this tree will silently recreate it at
  the same name through this script.
- `assert_store_is_self_consistent` now accepts every store shape prompt 01's deletion code and
  prompt 03's retirement will encounter (bare, legacy-own-siblings, legacy-backup-shape), so
  nothing about this prompt constrains their design.
- The three real primaries were read `mode=ro` only and are unchanged; prompt 05 (remedial
  retirement) still finds them exactly as prompt 01's audit measured them.
- `RunRegistry` suite count is now 128 (117 + 11); update any future baseline reference to this
  prompt's commit accordingly.
