# Prompt 02 — the sweep makes its store through the registry, and checks shards through the resolver

**Campaign:** [`README.md`](README.md) · **Board items:** **R4**, **R5** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** none. This prompt depends on no decision in README §6.2, and may be dispatched before 01.
**Closes:** `datastore-portability`'s `[03-atol-sweep-prepare-writes-its-store-sidecar-by-hand]` and
`[01-atol-sweep-check-expects-absolute-shard-records]`, both assigned to this campaign on
2026-09-25 (`docs/OPEN_ISSUES.md` §1.14).
**Opens:** anything out of scope that you find (§6), **without fixing it**.
**Recommended model:** **Sonnet** is enough for the code. The judgement is in keeping the edit to
two functions and a flag, in a script that is the record of a measurement.

**Read first:**

1. [`README.md`](README.md): §0, §1, §3, §5 and §6.3.
2. [`docs/store-retirement-audit.md`](../../docs/store-retirement-audit.md) §2.3, §2.6 and §2.12.
3. The two issues on the `datastore-portability` board, §3: their text, and their "Next step",
   which this prompt carries out.
4. `docs/handover/quadsource_atol_sweep.py`: the module docstring, `shard_paths` and
   `assert_store_is_self_consistent` (`:583-620`), `prepare()` (`:623-686`), `run_build`'s resume
   branch (`:444-462`), `run_child`'s call at `:372`, and `main()`'s `--prepare` / `--force`
   handling (`:879-917`).
5. `RunRegistry/stores.py`: `copy_store` (`:700-736`) and `_prepare` (`:644-679`).
6. `Datastore/shard_paths.py` `resolve_shard_path`.
7. `RunRegistry/tests/test_store_fingerprint.py` test 12 (`:797-842`). It counts this script's
   `run.finish(` calls, and must stay passing.
8. `RunRegistry/tests/store_fixtures.py` and `Datastore/tests/shard_store_fixtures.py`, for building
   a store with a registry sidecar in a temporary directory.

---

## 1. What is wanted

The sweep store is made by `prepare()`, which copies the A3 baseline's five files by hand, rewrites
the copy's `shards` rows to absolute paths with its own `UPDATE`, and writes a legacy sidecar with
`Path.write_text`. Two consequences now matter.

- **`prepare()` would overwrite a tombstone.** It judges that the sweep store exists from the
  store's files alone. After the sweep store is retired (prompt 05), none of those files exists, so
  a plain `--prepare` re-copies the baseline and **replaces the retired sidecar** (audit §2.6). A
  tombstone that a script in the tree silently destroys is not a record.
- **`--build --resume` refuses every modern store.** `assert_store_is_self_consistent` requires the
  `shards` rows to be the absolute paths of the expected siblings. Every store created since
  `datastore-portability` prompt 01 records bare names, so the check refuses them. That includes
  the A3 v2 store, which `run_build` checks before resuming (audit §2.6, probed).

Both issues' own next step is the fix: "replace its hand copy, its `UPDATE` and its sidecar with
one `RunRegistry.stores.copy_store` call", and "compare through
`Datastore.shard_paths.resolve_shard_path` instead of against literal absolute paths". This is the
"other reason" those entries waited for.

`RunRegistry.stores.copy_store` refuses a destination whose sidecar name exists (`_prepare`,
`:667-678`). After this prompt, therefore, `--prepare` refuses at a retired name, with no
retirement code yet. That is the behaviour a tombstone needs (README D6).

---

## 2. What to change

**R4 — `prepare()` makes the sweep store with one `copy_store` call.**
- Replace the hand copy of the shards and the primary, the `UPDATE`, and the hand-written sidecar
  with `RunRegistry.stores.copy_store(BASELINE_STORE, SWEEP_STORE, purpose=…)`.
- The purpose is the existing sidecar purpose text, **verbatim**.
- Import `RunRegistry` inside `prepare()`, as the script already imports it where it registers
  runs, so that the module's own imports do not change.
- Keep the call to `assert_store_is_self_consistent(SWEEP_STORE)` after the copy, and the success
  message.
- A refusal from `copy_store` comes through with its text intact. Prefix it with
  `quadsource_atol_sweep:` if you like, but lose nothing of it.
- `prepare()`'s docstring and its inline comments say what it now does. The comment explaining
  why the copied primary had to be re-pointed goes, because `copy_store` rewrites the rows.

**`--force` (*prompt's choice*, README §6.3).** `copy_store` never overwrites, so `--force` can no
longer mean "replace". Keep the flag, so that an old command line fails loudly rather than being
parsed as something else. With it, `prepare()` refuses before anything is read. The message says
that a sweep store is no longer replaced. Once the registry can retire a store (prompt 03), the
old one is retired. Until then, and in any case, the new one needs a name that has never held a
store, which is a change to `SWEEP_STORE`. Update `--force`'s `help` to say the same. If you find a
reason to remove the flag instead, log it as a deviation.

**R5 — `assert_store_is_self_consistent` compares through the resolver.**
- It passes when each `shards` row, resolved by `resolve_shard_path(primary, stored)`, is the
  expected sibling `shard_paths(primary)[serial]`. Compare serial by serial, not as sets of
  strings, and require exactly the serials `0 … SHARDS-1`.
- It refuses, as now, naming what it expected and what it found (the stored records and what
  they resolve to), when any serial is missing or extra, or any resolved path differs.
- It accepts a bare-name store and a legacy store whose absolute records name its own siblings,
  alike.
- It still opens the primary `mode=ro` only.
- Its docstring is updated. The claim that `ShardedPool` reads shard records "as absolute paths"
  is out of date: since `datastore-portability` prompt 01 it reads every record through the
  resolver. Keep the account of the 2026-09-23 accident, which is still why the check exists.
- Its refusal says what to do for **each** of its three callers, or says it neutrally. "Re-run
  --prepare --force" is wrong for a build and, after R4, for `prepare()` too.

**Nothing else in the script changes**: no measurement function, no case table, no `run.finish(`,
no other flag or message. The script's SHA-256 changes, and future run manifests record the new
value. Nothing gates on it: the sweep uses no unit ledger (`known()`), and
`QUADSOURCE-TOLERANCE-SWEEP.md` cites the script at `51cf304`, which git keeps. Say so in the log.

---

## 3. Tests — a new module in `RunRegistry/tests/`, no Ray, nothing under `var/`

Import the script by path with `importlib.util` (its module-level imports are light: `argparse`,
`json`, `numpy`, the standard library), and point `BASELINE_STORE` and `SWEEP_STORE` at stores in a
temporary directory. Make sure that the running-run check reads a temporary runs root, never
`var/runs/`. Monkeypatching `RunRegistry.stores.DEFAULT_ROOT` is one way; say which you used. The
source store is a fixture store with a problem-free registry sidecar. At minimum:

1. **A fresh sweep name.** `prepare()` makes a store whose:
   - `shards` rows are bare names;
   - sidecar is a problem-free registry sidecar, with a new `store_id`, `copied_from` naming the
     source's `store_id`, the purpose verbatim, and a last history entry `copy`;
   - self-consistency check passes.

   The source's files and sidecar are unchanged by `tree_state`.
2. **A tombstone at the sweep name.** The one this prompt exists for. Only a sidecar exists at the
   sweep name, with no primary and no shard. Use a registry sidecar whose primary has been
   removed, which is what a retired store looks like to today's reader (audit §2.1). `prepare()`
   refuses. The sidecar is **byte-identical** afterwards, no store file was created, and the source
   is unchanged. Do it again with `force=True`.
3. **An existing sweep store.** `prepare()` refuses with and without `force=True`, and nothing
   under the temporary root changed.
4. **The check.**
   - It accepts a bare-name store and a legacy store whose absolute records name its own siblings.
   - It also accepts a legacy store whose absolute records name the same file names in **another
     existing directory**. The resolver reads that store's own siblings, which is the backup's shape,
     and the log says why accepting it is correct.
   - It refuses a record whose name is not the expected sibling's, a missing serial and an extra
     serial, each with a message naming the expected and the found.
5. **The build-resume path.** `run_build`'s pre-resume check, `assert_store_is_self_consistent` on
   a store written by `write_new_store`, passes. It refused before this prompt; the audit's probe
   is the before. Do not run a build.
6. **No Ray**, and test 12 of `test_store_fingerprint.py` still passes unmodified.

**Deliberate breakage.** Show that each of these makes the tests written against it fail, then
restore it:
- (i) call `ShardedPool.copy_store` directly, and write the sidecar by hand as before. Test 2 must
  fail: the tombstone is overwritten;
- (ii) compare in the check against literal absolute paths, as before. Tests 4 and 5 must fail;
- (iii) let `force=True` fall through to the copy. Test 2's `force` case or test 3 must fail;
- (iv) compare the check as sets of resolved paths, ignoring serials. A test with two serials'
  records swapped must fail. Add one if 4 does not already have it.

Put each mutation in the log as a short diff, **exactly as applied**, for `git apply`. Name the
tests that failed. Mutations are never committed.

**A read-only check on the real primaries.** This is the one exception to README §3's rule that
prompts 01–04 open nothing under `var/`. Run the new `assert_store_is_self_consistent` against the
three real primaries in `var/datastores/`: the live A3 store, the sweep store and the backup. Each
must pass, because each holds legacy absolute records naming its own siblings by name. Check that
the check opened every file `mode=ro`: compare each primary's size, mtime and SHA-256 before and
after, and list each directory before and after. **Never call `prepare()`, `copy_store`, `--build`
or anything that writes against a real store or name.** Throwaway scripts go in your scratch
space.

---

## 4. Acceptance

1. §3's tests exist, need no Ray and open nothing under `var/`. Every existing test passes
   unmodified, test 12 included.
2. The deliberate-breakage record: (i)–(iv), each with its diff and the tests that failed.
3. The read-only check on the three real primaries: each passes, and each primary and directory
   is unchanged.
4. `git diff` of the script touches only `prepare()`, `assert_store_is_self_consistent`, the
   `--force` flag's handling and help, and the docstrings and messages of those.
5. `black --check` clean.
6. On this board: the §1 row for 02, items R4–R5, and §5 baselines.
7. On `datastore-portability`'s board: both issues moved from §3 to §4, each with a closing note
   naming this prompt and its commit. `[01-…]`'s note records the understated impact (audit
   §2.6).
8. `docs/OPEN_ISSUES.md`: both rows deleted from §1.14, and the count and date corrected. All in
   the same commit.

---

## 5. Stop conditions — stop and ask the user

- `copy_store` refuses the A3 baseline's sidecar as a source for a reason that is not a fixture
  artefact. That would mean `--prepare` cannot run at all against the real store.
- The check would refuse any of the three real primaries.
- Any change is needed outside §4 item 4's list, in the script or anywhere else.
- A test would need anything under `var/`.

---

## 6. What this prompt does not do

- It does not retire, delete or rename anything, and it does not touch the real sweep store.
- It does not change `copy_store`. In particular, it does not change what a copy carries from its
  source. A copy of the A3 baseline carries its `backup`, `restart` and `run_history` fields
  (audit §2.12, `[00-a-copy-carries-its-sources-present-tense-fields]`). Record in the log what
  the new sweep sidecar would carry, and fix nothing.
- It does not change `shard_paths`, `SHARDS`, the second shard-naming copy at `:711`, or any
  measurement code.

---

## 7. The log and the board

`logs/02-the-sweep-prepares-through-the-registry.md`, using the template in README §5.1. In
addition:
- the diff of the script, in full, since it is short and is a measurement record;
- why the check accepts the backup's shape;
- the read-only check on the real primaries, with the before-and-after table;
- the deliberate-breakage record, with diffs.

This board: the §1 row for 02, items R4–R5, and §5's baselines. The `datastore-portability`
board and `docs/OPEN_ISSUES.md`, as §4 says. Leave 03–05's held rows as they are.
