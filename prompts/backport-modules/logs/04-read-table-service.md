# Log 04 — Replace generated read-table methods with a `read_table()` service (B3)

**Prompt:** prompts/backport-modules/04-read-table-service.md
**Commit:** *(SHA intentionally not embedded — see `IMPLEMENTATION_STATE.md` §5 note 11 for why,
and §4's resolved `[commit-sha-links-stale]` entry for the problem this avoids)*
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

### Task 1 — `Datastore.read_table` (`Datastore/SQL/Datastore.py`)

- `Datastore.py:226-237` (constructor `read_table_config` loop) → single line:
  `self._read_table_config: Optional[ReadTableConfigType] = read_table_config`. The three-fault
  `setattr`/closure/dict-iteration block is gone entirely.
- `Datastore.py:739-762` (`_generic_read_table(self, cls, method_name, **kwargs)`) → `read_table(self,
  cls, *args, **kwargs)`. Before: unconditionally injected `tables=self._tables` into every call
  (fault 4) and had no guards at all — a call against an unconfigured or factory-less class fell
  through to a bare `KeyError`/`AttributeError` deep in the method body. After: validates, in order,
  that `self._read_table_config is not None`, that `class_name in self._read_table_config`, and that
  `hasattr(factory, "read_table")`, each with a distinct `RuntimeError` message; injects `tables`
  into `kwargs` only when `config.get("tables_arg", False)`, then calls
  `factory.read_table(conn, tab, *args, **kwargs)`.
- `Datastore.py:160-162` — updated the comment above `ReadTableConfigType` from the old
  `"method_name" -> {"class": ..., "tables_arg": ...}` shape to the new `class_name -> {"tables_arg":
  bool}` shape. `ReadTableConfigType = Dict[str, Any]` itself was already present in this tree
  (unrelated prior work) and needed no change — only the doc comment above it did.

### Task 2 — `ShardedPool.read_table` (`Datastore/SQL/ShardedPool.py`)

- `ShardedPool.py:187-200` (constructor `read_table_config` loop) → kept the existing
  `if read_table_config is not None:` guard (per **X3**, see below), replaced the broken
  `for method_name, method_config in read_table_config:` / `class_specifier = method_config["class"]`
  body with `for class_name, config in read_table_config.items():`, and fixed the pre-existing
  unformatted f-string bug in the error message (`'...(class id="{class_specifier}")'` with no `f`
  prefix → `f'...(class name="{class_name}")'`). The `setattr`/wrapper-closure block that built
  per-method dispatch is gone; nothing is attached to `self` any more.
- `ShardedPool.py:834-851` (`_generic_read_table(self, method_name, **kwargs)`) → `read_table(self,
  cls, *args, **kwargs)`. Before: no validation at all — `getattr(shard, method_name)` on a
  dynamically-`setattr`'d method name that (per faults 1-3) never actually existed. After: validates
  `self._read_table_config is not None`, rejects `class_name in self._sharded_tables` with a
  `RuntimeError` naming the sharded-vs-replicated mismatch, rejects `class_name not in
  self._read_table_config`, then keeps the existing random-shard-selection block unchanged and calls
  `shard.read_table.remote(class_name, *args, **kwargs)` instead of `getattr(shard,
  method_name).remote(**kwargs)`. This is possible only because Task 1 gives every `Datastore` shard
  actor a real `read_table` method to dispatch onto.

### Task 3 — re-key `config/sharding.py`

- `config/sharding.py:37-40`: `read_table_config` re-keyed from method name to class name, `"class"`
  entries dropped:
  ```python
  read_table_config = {
      "wavenumber": {"tables_arg": False},
      "redshift": {"tables_arg": True},
  }
  ```
  `tables_arg` values unchanged (confirmed against the factory signatures below), as required.

### Task 4 — 17 call sites across 6 `extract_*.py` scripts

All 17 sites, verified by grep both before (19 hits: 17 calls + 2 `config/sharding.py` keys) and
after (0 hits for the old names), moved from the generated-method call to the service call. Purely
mechanical — every site fit one of two shapes with no adaptation needed:

```
pool.read_wavenumber_table(units=…, is_source=True)   → pool.read_table("wavenumber", units=…, is_source=True)
pool.read_wavenumber_table(units=…, is_response=True) → pool.read_table("wavenumber", units=…, is_response=True)
pool.read_redshift_table(is_source=True, model_proxy=model_proxy)   → pool.read_table("redshift", is_source=True, model_proxy=model_proxy)
pool.read_redshift_table(is_response=True, model_proxy=model_proxy) → pool.read_table("redshift", is_response=True, model_proxy=model_proxy)
```

Applied with two `sed` substitutions (`pool\.read_wavenumber_table(` → `pool.read_table("wavenumber",
`, and the `redshift` equivalent) across all six files at once, rather than editing line-by-line —
safe here because the call is always `pool.read_<name>_table(`, never a differently-named variable,
and grep confirmed 17 hits before and 17 replacements after with 0 stragglers. Sites, by file:
`extract_Gk_data.py` (318, 320, 368, 373), `extract_GkSource_data.py` (775, 777, 821, 826),
`extract_GkWKB_data.py` (362, 364, 411), `extract_QuadSourceIntegral_data.py` (993, 995, 1045, 1050),
`extract_TkWKB_data.py` (386), `extract_tensor_source_data.py` (317). `extract_common.py` checked and
confirmed to have no occurrences, as the prompt warned not to assume.

## Deviations from the prompt

### `ProfileBatchManager`/return placement — IMPLEMENTATION CHOICE

The prompt explicitly asks for a deliberate choice here, since upstream returns *inside* the `with
ProfileBatchManager(...)` block while `SGWK`'s existing `_generic_read_table` returns *outside* it.
**Chosen: keep `SGWK`'s existing placement (return outside the `with` block).** Checked the two other
methods in this file that follow the same `with ProfileBatchManager(...) as mgr: ... return` shape —
`object_get` (`Datastore.py:480-546`) and `object_validate` (`Datastore.py:701-737`) — both close the
profile block before returning. Matching that existing, consistent local convention was preferred
over matching upstream, since this file's own pattern is the more relevant standard for a value added
to this file, and switching only `read_table` to close-after-return would make it the odd one out
among three structurally similar methods.

### Commit-SHA convention — IMPLEMENTATION CHOICE (see also `IMPLEMENTATION_STATE.md` §4/§5)

While updating the status board for this prompt, found that the recorded commit SHAs for prompts 01
(`fbc3a90`), 02 (`34380ba`) *and* 03 (`3e8a984`) are all unreachable from the branch tip — the same
`[commit-sha-links-stale]` defect prompt 03 diagnosed for 01/02 recurred in prompt 03's own commit.
Rather than repeat the guess-then-amend cycle a fourth time, this prompt (a) corrected all three
stale references (state board + the three log headers) to the branch-reachable SHAs (`2610abe`,
`9206704`, `e81b145`, confirmed with `git merge-base --is-ancestor`), and (b) stopped trying to embed
this prompt's own commit SHA at all, recording instead why it is omitted. This is a deviation from
the literal README.md §5.1 template (`**Commit:** <sha> — <subject>`), made because the template's
literal requirement is self-referentially impossible to satisfy without an amend, and three
consecutive prompts hitting the same bug is a process defect worth fixing rather than repeating.
Recorded as a new standing note (§5 note 11) for prompts 05–10.

### No other deviations

Both `Datastore.read_table` and `ShardedPool.read_table` were implemented exactly as specified in
the prompt's code blocks; `config/sharding.py`'s re-keying matches the prompt's example verbatim; all
17 call sites needed only the mechanical substitution — none required additional adaptation.

## Verification performed

1. **Grep counts (actually run):**
   - `grep -rn "read_wavenumber_table\|read_redshift_table" --include="*.py" .` — 19 hits before this
     prompt's edits, **0 after**.
   - `grep -rn "_generic_read_table" --include="*.py" .` — **0** hits after (both `Datastore.py` and
     `ShardedPool.py` occurrences removed).
2. **Formatting:** `./venv/bin/black --check` on all 9 touched files (2 pool/datastore modules, 1
   config module, 6 extract scripts) — "All done! 9 files would be left unchanged."
3. **Compile:** `./venv/bin/python3 -m py_compile` on all 9 touched files plus `extract_common.py` —
   clean, exit 0.
4. **Import, for real (not stubbed):** imported `Datastore.SQL.Datastore` and
   `Datastore.SQL.ShardedPool` under `./venv/bin/python3`, working around the still-open
   `[02-shard-config-reader]` `defaults`/`config.defaults` import issue with the same `sys.modules`
   stub technique prompts 02/03 used (no source touched). Confirmed via
   `DS.Datastore.__ray_metadata__.modified_class` (the `@ray.remote` actor's underlying class,
   necessary because `Datastore/SQL/__init__.py` does `from .Datastore import Datastore`, which
   causes `import Datastore.SQL.Datastore as DS` to bind `DS` to the *actor class*, not the module —
   a pre-existing package quirk, unrelated to this prompt, worth knowing if a later prompt tries the
   same import pattern): `read_table` present, `_generic_read_table` absent, on both classes.
5. **Behavioural — real datastore construction and a real SQL comparison (actually run, not
   reasoned about):** Built a throwaway harness (`/private/tmp/.../scratchpad/verify_read_table.py`,
   not committed) that:
   - Instantiates the actual `Datastore` class (bypassing Ray's actor wrapper by calling
     `__ray_metadata__.modified_class(...)` directly — a plain Python object, no `ray.init()`
     needed) against a fresh SQLite file, with the real `read_table_config` from
     `config/sharding.py`.
   - Inserts wavenumber and redshift rows **directly via the datastore's own SQLAlchemy engine and
     table objects** (`ds._engine`, `ds._tables["wavenumber"]`/`["redshift"]`), deliberately
     bypassing `object_get`/`build`, because `build`'s serial-number path requires a live
     `SerialPoolBroker` Ray actor that a single-process harness cannot easily provide, and that
     machinery is orthogonal to what this prompt changed.
   - Calls `ds.read_table("wavenumber", units=units, is_source=True)` and
     `ds.read_table("redshift", is_source=True)`, and separately queries the same SQLite file with
     raw `sqlite3` (`SELECT serial, k_inv_Mpc FROM wavenumber WHERE source=1`, and the redshift
     equivalent).
   - **Result:** both queries returned exactly the 3 rows inserted with `source=True` (of 4 total
     rows, one `response`-only), and the `store_id`/`serial` sets from `read_table` and from the
     direct SQL query were identical, both times. This is the "honest check … against a direct SQL
     query" the prompt asks for, given that the pre-change code could not construct a pool at all so
     no true before/after comparison is possible.
   - **Negative case 1 (unconfigured class):** `ds.read_table("GkSource")` on a `Datastore` with the
     real two-entry `read_table_config` raised
     `RuntimeError: Datastore: the read_table service is not available for objects of class
     "GkSource"`.
   - **Negative case 2 (service not configured at all):** a second `Datastore` instance constructed
     with no `read_table_config` argument (defaults to `None`) raised
     `RuntimeError: Datastore: the read_table service is not configured` on any `read_table` call.
   - Scratch database files removed after the run; harness script itself left in the scratchpad
     directory (not committed, per instructions), not under version control.
6. **Behavioural — `ShardedPool.read_table` negative cases (actually run):** the prompt's two
   required negative cases were checked against a minimally-constructed `ShardedPool` object (created
   via `__new__`, per the prompt's own fallback — a full `ShardedPool` needs a live multi-shard Ray
   cluster, which is out of proportion to what these two checks need), with only `_read_table_config`
   and `_sharded_tables` populated from the real `config/sharding.py` values:
   - `pool.read_table("GkSource")` (a sharded class) raised
     `RuntimeError: ShardedPool: the read_table service is only available for replicated tables, but
     "GkSource" is configured as a sharded table`.
   - `pool.read_table("LambdaCDM")` (replicated but not in `read_table_config`) raised
     `RuntimeError: ShardedPool: the read_table service is not available for objects of class
     "LambdaCDM"`.
7. **Not run:** end-to-end execution of any of the six `extract_*.py` scripts against a real,
   current-schema `ShardedPool` with multiple live Ray-actor shards (the audit §8 "same
   wavenumber/redshift arrays as before" checklist item, insofar as it implies a live multi-shard
   run). This needs `ray.init()` with actual shard actors and a populated multi-shard datastore,
   which is a real pipeline run, not a unit-style check. Recorded as an open item below for prompt
   10, consistent with how prompts 01/02/03 handled the same class of gap.

## Observations not acted on

- The `[02-shard-config-reader]` `defaults`/`config.defaults` import break in
  `LiouvilleGreen/WKBtools.py:6` (already tracked, still open) was hit again and worked around with
  the same `sys.modules` stub. This prompt was explicitly called out in that issue's own text as
  needing a real (non-stubbed) import "at the latest" — confirmed again that the workaround
  generalises, but the underlying fix is still deferred, per the existing issue's guidance, to
  whichever prompt (05, 09 or 10) the user or a later prompt decides should carry it. Not fixed here:
  it is not one of this prompt's tracked items (only B3 is), and fixing an unrelated import bug
  inside a "replace the read-table methods" commit would blur the revert boundary.
- `Datastore/SQL/__init__.py` does `from .Datastore import Datastore`, which means
  `import Datastore.SQL.Datastore as X` binds `X` to the `@ray.remote`-wrapped actor *class* rather
  than to the submodule object. Harmless and pre-existing (confirmed unrelated to this prompt's
  diff), but worth recording since it tripped up this prompt's own verification harness and would
  trip up the next person who tries the same import pattern; the workaround is to read
  `sys.modules["Datastore.SQL.Datastore"]` directly when the module object itself is needed.

## State handed to the next prompt

- `read_table` is now a real, validated service on both `Datastore` and `ShardedPool`. Prompt 05
  (E1, `persist_handler` split) touches `RayWorkPool.py`/`main.py`/the same six `extract_*.py`
  scripts but not `read_table` itself — no interaction expected, though both touch the same six
  files, so prompt 05 should re-grep before editing rather than assume line numbers from the audit.
- Prompt 06 (`inventory()` plumbing) is explicitly modelled on `read_table`'s shape
  (`README.md` §4: "Writing `inventory` against a settled `read_table` keeps them consistent").
  `Datastore.read_table` and `ShardedPool.read_table` as they now stand are the reference shape to
  follow: validate config-present → validate class-is-configured → (for `ShardedPool`) validate
  sharded/replicated routing → dispatch. Prompt 06 should **not** copy the `tables_arg`-style kwarg
  injection unless `inventory` genuinely needs the same mechanism — check what upstream's `inventory`
  actually needs before assuming it matches `read_table`'s.
- `IMPLEMENTATION_STATE.md` §5 note 11 (new) sets the convention for prompts 05–10: do not embed a
  commit's own SHA in that same commit's log/status-board entry. Follow it to avoid re-opening
  `[commit-sha-links-stale]`.
- The `[02-shard-config-reader]` issue remains open, now confirmed hit by four consecutive prompts
  (02, 03, 04, and implicitly whichever of 05/09/10 needs the first *unstubbed* import). Still
  nobody's tracked item; still needs either a user decision or an explicit owning prompt.
- Audit §8 checklist items 5-6 (behavioural confirmation of `read_table` under a real multi-shard Ray
  run, referenced in this prompt's header) are not discharged by this prompt — added to prompt 10's
  inputs via a new §3 issue in `IMPLEMENTATION_STATE.md`.
