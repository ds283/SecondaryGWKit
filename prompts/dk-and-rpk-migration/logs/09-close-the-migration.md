# Log — prompt 09, close the migration

**Prompt:** [`../09-close-the-migration.md`](../09-close-the-migration.md) ·
**Board items:** Z1–Z7 · **Date:** 2026-10-12 · **Model:** Sonnet 5.5 ·
**Commit:** this commit · **Result:** landed.

**What it did:**
- **Z1.** Three inherited issues moved to their boards' §4, **transferred to DatastoreKit, still open
  there**; the fourth, `[01-cross-filesystem-move-advice-says-delete-by-hand]`, **narrowed** to SGK's
  half, at the user's decision.
- **Z2.** The 23 prose lines of `[04-comments-name-the-layers-old-module-paths]` name what is current.
  The regex printed 23 lines before and **none** after; all 18 modules are `ast`-equal with docstrings
  removed.
- **Z7** (chosen by the user). The six `docs/` probes import `factories` from `config.datastore`; each
  now gets past its imports, shown without running a probe.
- **Z3–Z6.** `CLAUDE.md`'s test command; the final measurements; verification §4 with both G2 records
  drafted; the board, README §2, `prompts/INDEX.md` and the index (**108 → 103**).
- The six roots are at 08's counts. Nothing was opened.

**Where it ran.** Dispatched at `b95bf19` ("Write the orchestrator note for dk-and-rpk-migration prompt
09"), whose code is `f539015`'s (`ede4062` and `b95bf19` touch only `prompts/`).
- `git status --short` was empty, on branch `handover-remedial`.
- `RunRegistry list`: 29 runs, 27 finished, 2 `unknown`, none `running`. `RAY_ADDRESS` unset; no Ray
  process (`pgrep -lf 'gcs_server|raylet|ray::'` found none).
- DatastoreKit clean at `83068cb`, RayPoolKit clean at `c9b0e11`; both stayed clean (read only).
- The venv holds `datastorekit` 0.2.2 and `raypoolkit` 0.2.0, neither editable; nothing was installed or
  uninstalled.
- Load at the suites' starts, 1 min: 44–54 (the first four roots), 24.8 (ComputeTargets), 7–8
  (Datastore). The machine was shared with other sessions.

The record is [`docs/dk-and-rpk-migration-verification.md`](../../../docs/dk-and-rpk-migration-verification.md)
§4. Scratch that must outlive the session is under `var/dk-and-rpk-migration-09/` (gitignored): the two
scripts below, the regex's output before and after, the import checks, the tree-commit measurements and
the Z2/Z7 diff.

## What shipped

- **Z2: 18 files, 23 lines**, comment and docstring text only. By file, from `git diff -U0`:
  - `ComputeTargets/tests/`: `test_cosmology_representation_key.py:32`, `test_gauss_order_key.py:60`,
    `test_numeric_break_point_key.py:31`, `test_run_identity.py:41`;
  - `Datastore/tests/`: `real_store_fixtures.py:403`, `test_database_errors_reach_the_caller.py:37`,
    `test_exact_identity_lookups.py:29`, `test_integral_read_batch.py:26`, `test_quadsource_policy_key.py:28`,
    `test_read_path_mechanics.py:25`, `test_value_reads_parent_key.py:27`, `test_wkb_numeric_parent_key.py:35`;
  - `RunRegistry/stores.py:107`, `:114`, `:209`, `:1155`; `RunRegistry/tests/test_quadsource_atol_sweep_prepare.py:7`;
  - `config/datastore.py:4-5`, `:18`; `extract_common.py:157`; `main.py:283`; `tools/inventory_report.py:5`.

  Three of them were re-wrapped, where the longer name passed 100 columns (`test_cosmology_representation_key.py`,
  `test_run_identity.py`, `config/datastore.py`).
- **Z7: six files, one statement each:** `docs/datastore-integrity-audit/fk_sweep_probe.py:8`,
  `lookup_keys/common.py:18`, `lookup_keys/probe_03_column_existence.py:25` (its `# noqa: E402` stays),
  `tq_fk_probe.py:8`; `docs/tolerance-convergence/inventory.py:196` and `order_audit.py:1253` (both nested,
  indentation kept).
- **Z1:** `prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md` (two entries to §4),
  `prompts/datastore-integrity/IMPLEMENTATION_STATE.md` (one entry to §4),
  `prompts/store-retirement/IMPLEMENTATION_STATE.md` (one entry narrowed, in place). Each board gets
  only that.
- **Z3:** `CLAUDE.md`'s test command (the one fenced block and the sentence leading it).
- **Z4:** `prompts/dk-and-rpk-migration/measure/out/09-*`: both scripts' outputs, `09-pip-freeze.txt`,
  `09-package-provenance.json` and the nine suites' outputs.
- **Z5:** verification §4 appended (§1–§3 untouched).
- **Z6:** this campaign's board and README §2; `prompts/INDEX.md`; `docs/OPEN_ISSUES.md`; this log.
- **No code, test, tool or rehearsal script was added or changed**, nor `requirements.txt`; nothing in
  DatastoreKit or RayPoolKit.

## Deviations from the prompt

Each correction of the orchestrator's note (`orchestrator/prompt-09.md` §0) is logged, since each governs.

1. **Z7's check is not 04's method.** `STRUCTURALLY REQUIRED`. The prompt says "five with no arguments,
   `order_audit.py` through `d3_tables()`", which ran the probes up to their imports. Once repaired, a run
   goes on into the probe's body (`inventory.py` with no arguments runs its `main()`; the others build
   schemas and print), and the prompt's §5 says 09 opens no store and runs no build. Note, correction 1: **no probe was run.** The check executes each
   file's import statements alone (the script below). Before the repair, each of the six failed on its
   `_factories` line; after it, each is `all OK`.
2. **The move advice is narrowed, not read-and-decided.** `IMPLEMENTATION CHOICE`, the user's decision
   (note §0, "The user's decisions", and correction 2). The prompt says to read the path and transfer or
   narrow it. The reading is below (Verification 5), and the entry is narrowed.
3. **`config/datastore.py:4-5` in file-path form.** `IMPLEMENTATION CHOICE` (note, correction 3). The
   prompt says ``Datastore/SQL/`` in `config/datastore.py:4` → `datastorekit.SQL`, dotted; all three names
   in that parenthesis are file paths, so all three are ``datastorekit/SQL/``, ``datastorekit/store_reader.py``
   and ``datastorekit/store_inventory.py``.
4. **Z2 is 04's 23 lines and no more.** `IMPLEMENTATION CHOICE` (note, correction 4). The prompt adds "and
   any 07's log recorded"; 07 recorded none, and the regex finds none.
5. **README §2 marks all nine prompts landed**, each with its commit. `IMPLEMENTATION CHOICE` (note,
   correction 5). The prompt assumed 01–04 were already marked; every row said "written 2026-10-11".
6. **DatastoreKit's "0 imports" is "0 outside `prompts/`".** `STRUCTURALLY REQUIRED` (note, correction 6).
   The script counts `prompts/` too: 51 statements in 31 files, 20 strings and 1 `import_module` call, all
   there. The saved output and verification §4.2.1 say both numbers.
7. **No pin moved, though `v0.2.3` exists.** `IMPLEMENTATION CHOICE` (note, correction 7). DatastoreKit's
   checkout is at `83068cb`, and `v0.2.3` was released 2026-10-11. SGK stays at `v0.2.2`, which
   DatastoreKit's G2 admits. `git diff v0.2.2 HEAD -- docs/extraction/measure_client_imports.py` in
   DatastoreKit is empty, so running its script from the checkout is running the tag's.
8. **A transfer is not a resolution.** `IMPLEMENTATION CHOICE` (note, correction 8). Each of the three
   entries keeps its text verbatim, gains "transferred 2026-10-12 to DatastoreKit by dk-and-rpk-migration
   09" in its heading note, and a line saying it is **still open there**, indexed in DatastoreKit's
   `docs/OPEN_ISSUES.md` §1.2. No §3 lead-in on the three boards states a count (`a3-v2-readiness` §3.1/§3.2,
   `datastore-integrity` §3.2: none), so none was corrected. `store-retirement`'s "Four remain open" stays
   true.
9. **The G2 drafts' form.** `IMPLEMENTATION CHOICE` (note, correction 9). DatastoreKit's: its G2 cell,
   `✅ 2026-10-12: holds …` then the existing notes verbatim, mapped to its checklist §9's three parts.
   RayPoolKit's: G1's form, `**holds** 2026-10-12: …`, keeping the checklist pointer, mapped to items 1-5
   of its §8. Each is a fenced block in verification §4.4 and §4.5.
10. **Z4's measurements are at `b95bf19`, not at this commit.** `STRUCTURALLY REQUIRED`: both scripts read
    the committed tree through `git show`, and a commit cannot name its own SHA in its own content. As an
    additional check (an `IMPLEMENTATION CHOICE`), `git stash create` made a commit object of the working
    tree (`43458c2`, tracked changes only; no ref, no change to the tree), and both scripts were run on it:
    each differs from the saved output by the commit line only. The orchestrator re-runs both at this
    commit.
11. **Three lead-in sentences added to `docs/OPEN_ISSUES.md`** (§1.14, §1.15, §1.17), beside the note's
    §1.20 lead-in. `IMPLEMENTATION CHOICE`. Each says in one sentence what 09 did to a row of that
    section, because the section's prose says how many rows "stay open" and would otherwise read as the
    opposite. No issue content is copied.
12. **`CLAUDE.md`'s lead-in is reworded**, not only its fenced block: the sentence before the block now
    says there are six roots and why Datastore goes in four groups, and the sentence after (`:110-111`
    of the old file) is as it was. `IMPLEMENTATION CHOICE`. The prompt names "one sentence saying why the
    groups exist".
13. **The two scripts below are not committed** (acceptance 5, "09 adds no code and no test"), and the
    log quotes them verbatim. `IMPLEMENTATION CHOICE` (note, correction 1; §0 "The import check").

## Verification performed

### 1. §2.1's regex, before and after Z2

The orchestrator's implementation, as run from the repository root with `./venv/bin/python -I` (`stale_paths.py`),
quoted verbatim:

```python
"""09 §2.1: tracked files outside prompts/ and docs/ naming a deleted layer file or RayTools as a path."""
import re, subprocess, sys
mods = ["object", "contract", "replication", "shard_paths", "store_reader", "store_inventory",
        "SQL/schema", "SQL/ShardedPool", "SQL/Datastore", "SQL/ClientPool", "SQL/SerialPoolBroker",
        "SQL/ProfileAgent", "SQL/ObjectFactories/base"]
alts = []
for m in mods:
    alts.append(re.escape("Datastore/" + m) + r"(?:\.py)?\b")
    alts.append(re.escape("Datastore." + m.replace("/", ".")) + r"\b")
for t in ["sharded_store", "shard_key_audit"]:
    alts += [re.escape("tools/" + t) + r"\b", re.escape("tools." + t) + r"\b"]
alts += [r"ObjectFactories/base\b", r"RayTools"]
pat = re.compile(r"(?<![\w./])(?:" + "|".join(alts) + ")")
skip = re.compile(r"^\s*(?:from\s+\S+\s+import\b|import\s+\S)|import_module\(")
files = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True, check=True).stdout.split("\0")
n = 0
for f in files:
    if not f or f.startswith(("prompts/", "docs/")):
        continue
    try:
        lines = open(f, encoding="utf-8").read().splitlines()
    except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
        continue
    for i, line in enumerate(lines, 1):
        if pat.search(line) and not skip.search(line):
            print(f"{f}:{i}: {line.strip()}"); n += 1
print(f"-- {n} lines", file=sys.stderr)
```

**Before Z2, at `b95bf19`: 23 lines**, exactly the board's:

```text
ComputeTargets/tests/test_cosmology_representation_key.py:32: factories' own ``register()`` output in the shape ``Datastore.SQL.Datastore._build_schema`` builds
ComputeTargets/tests/test_gauss_order_key.py:60: ``register()`` output in the shape ``Datastore.SQL.Datastore._build_schema`` builds them, and the
ComputeTargets/tests/test_numeric_break_point_key.py:31: ``register()`` output in the shape ``Datastore.SQL.Datastore._build_schema`` builds them, and the
ComputeTargets/tests/test_run_identity.py:41: factories' own ``register()`` output in the shape ``Datastore.SQL.Datastore._build_schema`` builds
Datastore/tests/real_store_fixtures.py:403: # inventory (Datastore/store_inventory.py inventory_classes), with tags wherever the class has an
Datastore/tests/test_database_errors_reach_the_caller.py:37: ``build_schema(MetaData(), Datastore.SQL.Datastore._factories)`` in memory, as in
Datastore/tests/test_exact_identity_lookups.py:29: ``build_schema(MetaData(), Datastore.SQL.Datastore._factories)``, exactly as
Datastore/tests/test_integral_read_batch.py:26: `build_schema(MetaData(), Datastore.SQL.Datastore._factories)`, the same construction
Datastore/tests/test_quadsource_policy_key.py:28: ``build_schema(MetaData(), Datastore.SQL.Datastore._factories)``, and every call is to the real
Datastore/tests/test_read_path_mechanics.py:25: ``build_schema(MetaData(), Datastore.SQL.Datastore._factories)`` -- the same construction
Datastore/tests/test_value_reads_parent_key.py:27: `build_schema(MetaData(), Datastore.SQL.Datastore._factories)`, as in
Datastore/tests/test_wkb_numeric_parent_key.py:35: a temporary directory) with ``build_schema(MetaData(), Datastore.SQL.Datastore._factories)``, and
RunRegistry/stores.py:107: structured inventory (`Datastore.store_inventory.read_inventory`): per class and per tag set a
RunRegistry/stores.py:114: `Datastore.store_inventory` inside the fingerprint functions only, so that importing this module,
RunRegistry/stores.py:209: # both separators, on every platform, as `Datastore/shard_paths.py` treats a shard record
RunRegistry/stores.py:1155: # A fingerprint digests the structured inventory, `Datastore.store_inventory.read_inventory`. A
RunRegistry/tests/test_quadsource_atol_sweep_prepare.py:7: `shards` row, serial by serial, through `Datastore.shard_paths.resolve_shard_path`, which is what
config/datastore.py:4: The generic datastore layer (``Datastore/SQL/``, ``Datastore/store_reader.py`` and
config/datastore.py:5: ``Datastore/store_inventory.py``) does not import this project's registries: it is given them, and
config/datastore.py:18: tables it is given in the order ``Datastore.SQL.schema.drop_order`` derives from their foreign
extract_common.py:157: ``Datastore.store_inventory.read_inventory(pool.primary, pool.factories)`` (the registry the
main.py:283: # --inventory reads the closed store read-only (Datastore.store_inventory.read_inventory), before
tools/inventory_report.py:5: that ``Datastore.store_inventory.read_inventory`` returns. It reads nothing from a store itself:
```

**After Z2 and Z7: 0 lines** (`-- 0 lines`, and no stdout). The regex is not vacuous, since it printed the
23 above on the same tree before the sweep, and it printed 0 only after. It skips lines that are import
statements or hold `import_module(`, and its lookbehind `(?<![\w./])` keeps `datastorekit.tools.sharded_store`
and `datastorekit/store_inventory.py` out. It reads only files outside `prompts/` and `docs/`; run over
`docs/` alone it finds 128 lines, 25 of them in this campaign's own verification record, the rest dated
audits and rehearsal outputs, which are the record of the tree they describe (README §6.2 C10), and
`docs/tolerance-convergence/inventory.py:1315`, a dated string Z7 leaves.

### 2. Z2 changes prose only

For every one of the 18 Python files, `ast.dump` of the module with docstrings removed (the module's, and
every class's and function's first string statement) is identical at the working tree and at `HEAD`:

```python
import ast, subprocess
files = subprocess.run(["git","diff","--name-only"],capture_output=True,text=True).stdout.split()
def strip(t):
    for n in ast.walk(t):
        if isinstance(n,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.body and isinstance(n.body[0],ast.Expr) and isinstance(getattr(n.body[0],"value",None),ast.Constant) and isinstance(n.body[0].value.value,str):
            n.body=n.body[1:] or [ast.Pass()]
    return ast.dump(t)
for f in files:
    if not f.endswith(".py") or f.startswith("docs/"): continue
    old=subprocess.run(["git","show","HEAD:"+f],capture_output=True,text=True).stdout
    print(("IDENTICAL " if strip(ast.parse(old))==strip(ast.parse(open(f).read())) else "DIFFERS   ")+f)
```

Result: `IDENTICAL` for all 18, `files: 18 differs: 0`. Comments are not in the AST, so each file's changed
lines are named above from `git diff -U0`, and the 23 `-`/`+` pairs differ by a path only (the diff is
`var/dk-and-rpk-migration-09/z2_z7_diff_U0.patch`). `RunRegistry/stores.py:114`'s claim holds under the
new name: in a fresh `python -I`, `import RunRegistry.stores` loads none of `ray`, `sqlalchemy` or
`datastorekit.store_inventory`. The Z2 edit was made by `z2_apply.py` (in `var/`, not committed): it rewrites,
on the 23 lines and nothing else, the patterns `Datastore.SQL.Datastore._factories` → `config.datastore.factories`,
`Datastore.SQL.Datastore._build_schema`, `Datastore.store_inventory`, `Datastore.shard_paths`, `Datastore.SQL.schema`
→ `datastorekit.<same>`, and `Datastore/store_inventory.py`, `Datastore/store_reader.py`, `Datastore/shard_paths.py`,
``Datastore/SQL/`` → `datastorekit/…`; the three re-wraps were made by hand.

### 3. Z7: each probe gets past its imports

**The script**, `imports_only.py`, quoted verbatim. It executes every absolute import statement of a file, in
source order, and nothing else, from a copy outside the repository's tracked files, as
`./venv/bin/python -I imports_only.py . <file>` from the repository root:

```python
"""Execute every absolute import statement of a Python file, and nothing else.

usage: python -I imports_only.py <repo root> <file>
Prints each import and OK, or the first exception, and exits 0 / 1.
"""
import ast, sys, traceback
from pathlib import Path
root, path = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
src = path.read_text()
sys.path[:0] = [str(root), str(path.parent)]
nodes = [n for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))
         and not (isinstance(n, ast.ImportFrom) and n.level)]
nodes.sort(key=lambda n: n.lineno)
ns = {"__name__": "imports_only", "__file__": str(path)}
for n in nodes:
    stmt = ast.unparse(n)
    try:
        exec(compile(ast.Module([n], []), str(path), "exec"), ns)
    except Exception as e:
        print(f"{path.relative_to(root)}:{n.lineno}: {stmt}  ->  {type(e).__name__}: {e}")
        sys.exit(1)
print(f"{path.relative_to(root)}: {len(nodes)} import statements, all OK")
```

**Before the repair**, on the six: each exits 1 on its `_factories` line (this is the non-vacuity):

```text
docs/datastore-integrity-audit/fk_sweep_probe.py:8: from datastorekit.SQL.Datastore import _factories  ->  ImportError: cannot import name '_factories' from 'datastorekit.SQL.Datastore' (/Users/ds283/Documents/Code/SecondaryGWKit/venv/lib/python3.12/site-packages/datastorekit/SQL/Datastore.py)
docs/datastore-integrity-audit/lookup_keys/common.py:18: from datastorekit.SQL.Datastore import _factories  ->  ImportError: cannot import name '_factories' from 'datastorekit.SQL.Datastore' (/Users/ds283/Documents/Code/SecondaryGWKit/venv/lib/python3.12/site-packages/datastorekit/SQL/Datastore.py)
docs/datastore-integrity-audit/lookup_keys/probe_03_column_existence.py:25: from datastorekit.SQL.Datastore import _factories  ->  ImportError: cannot import name '_factories' from 'datastorekit.SQL.Datastore' (/Users/ds283/Documents/Code/SecondaryGWKit/venv/lib/python3.12/site-packages/datastorekit/SQL/Datastore.py)
docs/datastore-integrity-audit/tq_fk_probe.py:8: from datastorekit.SQL.Datastore import _factories  ->  ImportError: cannot import name '_factories' from 'datastorekit.SQL.Datastore' (/Users/ds283/Documents/Code/SecondaryGWKit/venv/lib/python3.12/site-packages/datastorekit/SQL/Datastore.py)
docs/tolerance-convergence/inventory.py:196: from datastorekit.SQL.Datastore import _factories  ->  ImportError: cannot import name '_factories' from 'datastorekit.SQL.Datastore' (/Users/ds283/Documents/Code/SecondaryGWKit/venv/lib/python3.12/site-packages/datastorekit/SQL/Datastore.py)
docs/tolerance-convergence/order_audit.py:1253: from datastorekit.SQL.Datastore import _factories  ->  ImportError: cannot import name '_factories' from 'datastorekit.SQL.Datastore' (/Users/ds283/Documents/Code/SecondaryGWKit/venv/lib/python3.12/site-packages/datastorekit/SQL/Datastore.py)
```

**After it**, on the six and on the six that fail through `lookup_keys/common.py`, in that order:

```text
docs/datastore-integrity-audit/fk_sweep_probe.py: 4 import statements, all OK
docs/datastore-integrity-audit/lookup_keys/common.py: 6 import statements, all OK
docs/datastore-integrity-audit/lookup_keys/probe_03_column_existence.py: 6 import statements, all OK
docs/datastore-integrity-audit/tq_fk_probe.py: 4 import statements, all OK
docs/tolerance-convergence/inventory.py: 10 import statements, all OK
docs/tolerance-convergence/order_audit.py: 25 import statements, all OK
docs/datastore-integrity-audit/lookup_keys/probe_01_tagged_read_batch.py: 14 import statements, all OK
docs/datastore-integrity-audit/lookup_keys/probe_01b_fix_shape.py: 6 import statements, all OK
docs/datastore-integrity-audit/lookup_keys/probe_04_exit_time_and_solver.py: 9 import statements, all OK
docs/datastore-integrity-audit/lookup_keys/probe_04b_exact_match_fix_shape.py: 7 import statements, all OK
docs/datastore-integrity-audit/lookup_keys/probe_05_value_lookups_and_read_batch.py: 14 import statements, all OK
docs/datastore-integrity-audit/lookup_keys/probe_06_oneloop_tags_and_redshift_zero.py: 7 import statements, all OK
```

The statement counts 4, 6, 6, 4, 10 and 25, and 14, 6, 9, 7, 14 and 7, are the orchestrator's predictions.
`probe_01b` and `probe_04b` import `probe_01` and `probe_04` as modules, and both guard their bodies with
`if __name__ == "__main__"`, so nothing ran. `git status` stayed clean of everything but the intended
edits (no `__pycache__` is tracked).

### 4. Suites

README §7's command, output to a file, the verdict grepped; each root in the foreground, serially, Datastore
in four groups. At `b95bf19` plus 09's changes.

| Root | Ran | 08's | Result | Load (1 min) |
|---|---|---|---|---|
| AdaptiveLevin | 32 | 32 | OK | 43.8 |
| CosmologyModels | 39 | 39 | OK | 43.8 |
| LiouvilleGreen | 148 | 148 | OK (skipped=1) | 44.1 |
| RunRegistry | 286 | 286 | OK | 54.4 |
| ComputeTargets | 693 | 693 | FAILED (failures=1) | 24.8 |
| Datastore `[a-f]` / `[g-o]` / `[p-r]` / `[s-z]` | 54 / 103 / 101 / 62 = 320 | 320 | OK | 8.2 / 7.9 / 7.3 / 7.5 |

The one failure is `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, `0.0613 not less than or equal
to 0.06`, README §7's known wall-clock flake. Re-run alone at load 8.2, three times, it passed twice (the run
took 0.43 s and 0.40 s) and failed once (`0.0658 not less than or equal to 0.06`), so alone it is no more
than a coin that favours the pass; 08's review asked for the re-run alone at low load before reporting it. It passed in R1's and R2's
root runs, and 09 changed nothing it reaches. The counts are unchanged, and no test imports the six probes.
`black --check` on the 24 Python files touched (18 + 6): `24 files would be left unchanged`.

### 5. Z1's reading of the move advice

- **The text is the package's.** `ShardedPool._failure_message` (installed
  `datastorekit/SQL/ShardedPool.py:3000`) appends, on `EXDEV` (`:3026-3030`): "…copy it instead, and then
  delete the source by hand". `move_store`'s docstring says the same (`:2683-2684`). The package has
  `ShardedPool.delete_store` (`:2704`).
- **SGK relays it.** `RunRegistry.stores.move_store` (`RunRegistry/stores.py:1109`) calls
  `_store_files("move", ShardedPool.move_store, src, dst)` at `:1123`. `_store_files` (`:1052-1060`)
  re-raises `RuntimeError(f"{e}. No sidecar was written or moved: …")`, the package's message verbatim with
  the sidecar listing appended. SGK adds and shapes no advice.
- **The harm is SGK's rule.** A registered store's source is retired with `store retire`, never deleted by
  hand (store-retirement decision 6.1.1), and the package cannot know a store is registered.
- **The test** `Datastore/tests/test_copy_move_store.py`, which asserted the text and which the entry's
  "Next step" named, is not in SGK's tree at `HEAD` (04 deleted it); `git show HEAD:Datastore/tests/test_copy_move_store.py`
  fails, and the package has `datastorekit/tests/test_copy_move_store.py`. So the constraint goes with the
  package's half.
- DatastoreKit's `docs/OPEN_ISSUES.md` §1.2 lists all four issues, as the note says, so no stop condition
  of the prompt's §4 fires.

### 6. The index, by counting

- `grep -c '^| \`\[' docs/OPEN_ISSUES.md` = **103** (108 before); header "103 open across twenty
  campaigns", "Last updated 2026-10-12".
- By section: **§1.14 4, §1.15 5** (was 6), **§1.17 7** (was 9), **§1.20 2** (was 4). Three rows deleted
  (the transfers), two deleted (the closures), one rewritten (`[01-cross-filesystem-…]`'s hook, now
  "Narrowed 2026-10-12", SGK's half). No section empties, so "twenty campaigns" stands.
- `prompts/INDEX.md`: "25 campaigns: 3 live, 1 seeded, 21 closed or complete"; this campaign's row is
  "closed 9 / 9, 2026-10-12", "2026-10-11 → 2026-10-12", "§1.20; 2"; its taken-on date is 2026-10-12.
- This campaign's board §3 holds 2 entries and §4 holds 2, agreeing with §1.20's two rows.

### 7. Z4 and the other measurements

See verification §4.2. In short: DatastoreKit's script at `b95bf19` gives 15 layer files absent, 423 `.py`
files read, **51 statements in 31 files, 20 strings and 1 `import_module` call, all in `prompts/` and
0 outside**, and 99 factory imports; RayPoolKit's gives `RayTools/` absent, 0 imports, 0 `import_module`
calls, 2 strings, **54 calls, `constructs` 54**, 1496 tracked files, 793 non-`.py` lines. Neither
repository's status changed. `pip freeze` `diff`s empty against `var/dk-and-rpk-migration-06/freeze-after.txt`;
`pip check` is clean; `package_provenance()` is 08's record. Each is saved under `measure/out/09-*`.

## The deliberate-breakage record

None. The prompt adds no code and no test (acceptance 5), and §2.1's regex run before Z2 is its only
non-vacuity run (Verification 1); the six probes' failing import checks before Z7 are the other
(Verification 3).

## Observations not acted on

1. **README §7's baselines are stale** ("last measured at `b510bc9`": RunRegistry 280, Datastore 739), and
   §0.2's table with them. They are the record of the plan, and this board's §5 carries the current counts.
2. **Prose naming the layer's `Datastore` class, not a path, remains** in some docstrings, e.g.
   `ComputeTargets/tests/test_run_identity.py` "the shape of ``Datastore._insert``" and "the project's
   ``Datastore`` machinery". The regex does not reach them (they name no module path), the class is the
   package's `datastorekit.SQL.Datastore.Datastore`, and the prompt asked for paths only.
3. **The store-retirement entry's own text** still names `RunRegistry/stores.py:753` and
   `Datastore/tests/test_copy_move_store.py`; the entry's text is the record, and the "Narrowed" line gives
   the current lines.
4. **`test_tk_wkb_phase.TestCost.test_wall_time_per_object` failed in one of three runs alone at load 8**
   (0.0658 s against 0.06 s). README §7 says it passes alone; at this load it is a coin that favours the
   pass. Not an issue of this campaign's (the test's margin is about one), and already known.
5. **The inherited transfers' wording in DatastoreKit's §1.2** says SGK's measurements "stay on SGK's
   boards"; they do, in §4, text unchanged. DatastoreKit's list has no "narrowed" mark for the fourth
   issue, whose SGK half it does not know about; the user's G2 commit there may want a line.

## State handed to the orchestrator

- **Review 09** against its commit: V1 (the sweep is prose only), V2 (the probes get past their imports),
  V3 (the hand-over, by counting), V4 (the close, with both measurements re-run at this commit), V5 (the
  records), V6 (the six roots). "This commit" stands for the SHA wherever it appears on this board, in
  README §2, in verification §4.4 and §4.5 (the two drafts) and in this log.
- **The user's, after the review:** retiring the six rehearsal stores; the two G2 commits in DatastoreKit
  and RayPoolKit (the drafts are in verification §4.4 and §4.5, ready to paste); renaming the `Datastore`
  package; moving to `datastorekit` `v0.2.3`.
- The registry holds 29 runs, none `running`. The venv is unchanged.
