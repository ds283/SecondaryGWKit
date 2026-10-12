# RunRegistry extraction audit — moving the run registry into a package of its own

**Date:** 2026-10-12 · **Tree:** `handover-remedial` at `d03a643`, clean · **Author:** Claude Opus
5.5 · **Status:** read-only in this repository: no file was changed to write it except this one.
Under `var/`, the run directories and the store sidecars were read and nothing was written. The
two client campaign plans named in §7 were amended in their own repositories, at the user's
request: ChamPBH's in a commit of its own, SI's in its still-uncommitted plan (§7.3).

**Why it exists.** `RunRegistry/` was built here, for this repository's long jobs
(`prompts/run-registry/README.md` §0). ChamPBH and StochasticInstantons (SI) need the same thing:
both of their `dk-and-rpk-migration` plans record that they have "no run registry and no
fingerprint", and ChamPBH's P5 built its provenance helper "in the shape of SGK's
`package_provenance()` … so that the move is a substitution", because the user noted that ChamPBH
will eventually import the registry. Now that `datastorekit` and `raypoolkit` are extracted and
adopted, the user asked, before the A3 v2 compute run, how the registry should be shared. This
audit records the decisions taken on 2026-10-12, measures what in `RunRegistry/` belongs to this
repository rather than to the registry, and sets out the order in which three repositories have to
move.

**How to read the citations.** **[checked]** marks a claim re-read in the source while writing this
document. **[probed]** marks one also run, read-only. **[agent]** marks a claim reported by a
read-only search agent and spot-checked, not re-derived line by line.

---

## 0. Decisions taken with the user, 2026-10-12

| | Decided | Rejected, and why |
|---|---|---|
| **D1** — where it goes | **A repository of its own, `RunRegistryKit`** (package `runregistrykit`), beside DatastoreKit and RayPoolKit | **DatastoreKit:** the registry's core is about jobs, not stores, and imports only the standard library (§1); SI's parameter sweeps write CSVs and should not need `ray` and `sqlalchemy` to get a heartbeat; DatastoreKit's charter is the standard library, `ray`, `sqlalchemy` and itself; `datastore-portability` §6.3 deliberately kept the sidecar out of the datastore layer; and a registry change would force a `datastorekit` release and three pin moves. **A copy in each client:** three copies of a convention drift apart, and the convention is the point. **One combined toolkit:** ties three unrelated release cadences together |
| **D2** — when | **Before the A3 v2 compute run** | After it. The extraction is a move, but A3 v2's run manifest and store sidecar are cleaner as provenance if a tagged release writes them, and no registry change should land while an A3 v2 store might still need resuming |
| **D3** — the fingerprint | **Split by layer (§3).** The encoding and the digest algorithm go to DatastoreKit, with a version of their own. What goes into a fingerprint, and the client's format number, stay in the client, **grouped in one module**. RunRegistryKit is given a client-provided fingerprinter and treats its result as opaque, apart from the format, the overall digest and the differences `compare` returns | The algorithm in RunRegistryKit: it would bind RunRegistryKit to DatastoreKit's `StoreInventory` shapes. A digest callable written by each client: the same algorithm three times, which is where divergence bites |
| **D4** — a client with no fingerprinter | **None is supported.** The store layer requires a fingerprinter. The core/`stores` split (§5) is a dependency split, not a degraded mode | Making the store layer work without one: the user's ruling, "that is solving the wrong problem". ChamPBH and SI grow `inventory_spec()` in their own migrations (§7) and adopt RunRegistryKit after them |
| **D5** — spec coverage in the clients | **ChamPBH prompt 09 and SI prompt 10 gain a coverage acceptance** (§7.3): every registered class declares a spec, is read through an owner's spec, or is excluded with a recorded reason, and a test enforces it | Leaving the report-coverage criterion as it is: a report needs only what it shows, but a fingerprint must not silently omit a class that holds state |

D1 and D2 were the planner's recommendation, accepted ("Order of work: agree"). D3 was the
planner's recommendation after the user asked what a fingerprint format currently specifies, and
the user added the requirement that the client's version and its content specification be grouped
together client-side. D4 is the user's correction of the planner's first proposal. D5 is the
planner's, accepted; the user asked for the client plans to be amended now rather than left to a
later campaign.

---

## 1. What `RunRegistry/` is today

Measured at `d03a643`. `python -m RunRegistry list` reports 29 runs, 27 finished and 2 `unknown`
(`a3-pilot`, `a3-pilot-resume`, which predate the registry); nothing is `running`. **[probed]**

| File | Lines | Imports beyond the standard library | What it is |
|---|---|---|---|
| `RunRegistry/__init__.py` | 710 | none at module scope; `RunRegistry.stores` lazily, for `Run.finish(fingerprint=True)` | the run directory, manifest, `status.json`, the checkpoint ledger, heartbeat, liveness, provenance, `begin`, `list_runs` |
| `RunRegistry/stores.py` | 2185 | `datastorekit.SQL.ShardedPool` (×3) and `datastorekit.store_inventory` (×3), all lazy; **`config.datastore.factories`** (`:1494`), lazy | the store sidecar: create, adopt, copy, move, fingerprint, compare, listing, retire, amend |
| `RunRegistry/__main__.py` | 515 | none | the CLI: `list` and the `store` subcommands |

**[checked]** It was built by `run-registry` (01–03, 2026-09-22) and grown by `datastore-portability`
(the sidecar, copy and move, `f53598f`), `store-fingerprint` (`ef46154`), `store-retirement`
(retire, `854e2ae`; amend, `66617c9`; `938be2b`, `211d2c4`), `datastore-integrity` (fingerprint
formats 2–6), `datastore-generic` (the registries given rather than imported, `19f07ee`) and
`dk-and-rpk-migration` (`packages` in the manifest, `f54f1c5`; the `datastorekit` imports,
`0c9e2cf`).

**The two layers are separable, and only one touches a datastore.** The core imports nothing but
the standard library; every `datastorekit` import is in `stores.py` and is lazy, so that
`import RunRegistry` loads neither `ray` nor `sqlalchemy` (the docstring of `fingerprint_store`,
`stores.py:1454-1455`). The one call from the core into the store layer is
`Run.finish(..., fingerprint=True)` (`__init__.py:366-410`). **[checked]**

### 1.1 The public surface **[agent]**

- **Core** (`__init__.py`): `begin` (`:542`), `list_runs` (`:674`), `class Run` (`:290`: `status`,
  `heartbeat`, `finish`, `record`, `known`, and the path properties), `liveness`, `heartbeat_age`,
  `pid_alive`, `now_iso`, `run_id`, `script_sha256`, `git_provenance`, `package_provenance`,
  `read_json`, `write_json_atomic`; constants `TERMINAL_STATES`, `DEFAULT_STALE_AFTER` (900 s),
  `REPO_ROOT`, `DEFAULT_ROOT`, `RECORDED_DISTRIBUTIONS`.
- **Store layer** (`stores.py`): `sidecar_path`, `new_store_id`, `read_sidecar` and
  `SidecarReading`, `create_sidecar`, `adopt_sidecar`, `runs_naming`, `copy_store`, `move_store`,
  `fingerprint_of`, `compare_fingerprints`, `listing_lines`, `check_listing_path`, `write_listing`,
  `fingerprint_store`, `retire_store`, `amend_sidecar`; the format constants `SIDECAR_FORMAT`,
  `REQUIRED_FIELDS`, `KNOWN_FIELDS`, the operation names, `FINGERPRINT_FORMAT`, `TAKEN_KEYS`.
- **CLI** (`__main__.py:372`): `list [--root] [--stale-after]`; `store show | create | adopt |
  copy | move | fingerprint [--write] [--listing] | retire --reason [--without-fingerprint]
  [--dry-run] [--stores-root] | amend --field --reason [--json | --remove]`, each with
  `--runs-root` where a running run can refuse it.

**Two formats are versioned, one is not.** A sidecar carries `"sidecar_format": 1`; a fingerprint
carries `fingerprint_format`; a run manifest carries no format field (§8, Q7). **[checked]**

---

## 2. What in it belongs to this repository

Each row is something a shared package cannot keep as it is. **[checked]** unless marked.

| # | Where | What it assumes | What it becomes |
|---|---|---|---|
| C1 | `REPO_ROOT`, `__init__.py:57` | the repository root is the parent of the package's own directory. **Installed into `site-packages`, this points into the venv**, so `var/runs/` would be created there: a silent breach of the durability rule (`CLAUDE.md`, rule 5) | the root found from where the code runs, through a client config file at the root that also marks it (§5.2) |
| C2 | `DEFAULT_ROOT = REPO_ROOT/var/runs`, `:58` | as C1 | the client config, defaulting to `var/runs` under the found root |
| C3 | `DEFAULT_STORES_ROOT = REPO_ROOT/var/datastores`, `stores.py:185` | as C1; it is where `retire` looks for sidecars that reference a store | the client config |
| C4 | `_repo_path`, `_resolve`, `:176-185` | paths in manifests and sidecars are written relative to `REPO_ROOT` (e.g. `"script": "docs/gktk-remedial/scoped_pipeline_run.py"`, `"results": "var/datastores/…"`) | unchanged in meaning, relative to the found root. **This is what keeps every existing record readable** |
| C5 | `RECORDED_DISTRIBUTIONS = ("datastorekit", "raypoolkit")`, `:119` | the packages SGK's `dk-and-rpk-migration` adopted | the client config. `runregistrykit` itself is always recorded |
| C6 | `from config.datastore import factories`, `stores.py:1494` | SGK's registry | the client's fingerprinter (§3, D3) |
| C7 | `FINGERPRINT_FORMAT = 6`, `stores.py:206`, and its history comment, `:194-205` | versions SGK's key changes (§3.2) | moves to the client's fingerprint module (§3.3) |
| C8 | docstrings citing SGK campaigns (`prompts/run-registry` §0, `datastore-portability` §6.5, `store-retirement` §6.2, …) | the reader can open SGK's `prompts/` | SGK's history, reachable through a `PROVENANCE.md` import commit, as RayPoolKit does; the rules themselves are restated in the package README |
| C9 | git provenance, `git_provenance(cwd=REPO_ROOT)`, `:98` | the git tree is the one the package lives in | the found root's tree. **Also the run's script hash and SHA** (`CLAUDE.md` rule 6) |

**A hazard C1 already has here.** `REPO_ROOT` follows whichever copy of `RunRegistry/` was imported.
A run launched from a git worktree of this repository therefore records into the worktree's
`var/runs/`, and is lost when the worktree is cleaned up. None exists today (`git worktree list`
shows the main checkout only) **[probed]**, but campaigns here do run in worktrees
(the venv-symlink memory). The extraction has to decide this explicitly (§8, Q3): the found root
can be `git rev-parse --git-common-dir`'s parent, which is the main checkout from any worktree.

### 2.1 Who calls it **[agent; `main.py` probed]**

**`main.py` does not.** It contains no reference to the registry (`grep -ci RunRegistry main.py` is
0). A pipeline run is registered by the drivers that `exec` it, which is the design of
`run-registry` prompt 02 ("register the pipeline driver without re-plumbing it", `ee46e5c`):

| Caller | What it uses |
|---|---|
| `docs/gktk-remedial/scoped_pipeline_run.py` — **the launcher A3 v2 will use** | `begin(..., results=…, heartbeat_means=…)` (`:541-554`); a pre-Ray SIGTERM/SIGINT handler that calls `finish("killed")` (`:557-564`); `heartbeat(stage=…)` from its stream wrapper (`:253`); `finish(..., fingerprint=True)` after the exec (`:476-496`) |
| `docs/handover/quadsource_atol_sweep.py` | `begin` (`:556`, `:864`); `heartbeat`; `finish(..., fingerprint=True)` (`:952-961`); `stores.copy_store` (`:763`) |
| `ObjectFactories/tests/test_quadsource_policy_key.py:411` | `from RunRegistry.stores import fingerprint_of` — layer C, so after D3 it imports from DatastoreKit |
| `docs/a3-v2-readiness/{sigterm_handler_probe,fingerprint_time_probe}.py`, `docs/datastore-integrity-verification/{compare_stores,interrupt_when}.py`, `docs/dk-and-rpk-migration/rehearsal/{compare_with_recorded,stage_counts}.py` | the rehearsal and probe tooling: `begin`, `liveness`, `pid_alive`, `read_json`, `fingerprint_store`, `compare_fingerprints`, `FINGERPRINT_FORMAT`; and `stage_counts.py` uses the private `RunRegistry._slug` |
| `prompts/datastore-generic/orchestrator/measure-0{5,6,7,8,9}/scripts/*.py` (6 files) | `fingerprint_of`. Campaign records: left as written |

The live callers are few and concentrated: two drivers, one factory test, and the rehearsal
tooling that the SGK adoption campaign will itself run. `extract_common.py` and `tools/` do not
import it.

---

## 3. The fingerprint

### 3.1 What specifies a fingerprint today

A fingerprint is produced in three layers, which live in three places. **[checked]**

| Layer | What it decides | Where |
|---|---|---|
| **A. Content** | which classes, and each class's identity: leaf columns, parent references (`Parent`, polymorphic or not, nullable, cross-shard), parent sets, tag and value tables, the validated column | **the client**: each factory's static `inventory_spec()`, and the registry `config.datastore.factories`, from which `inventory_classes` derives the classes and their order |
| **B. Encoding** | how a row becomes a record and a canonical string: `canonical` (a float as `float.hex` of the stored value), `canonical_json`, `reference_digest`, `Record.as_json` (`key`, `tags`, `validated`, `value_count`), the parent-set digest, which shard's copy of a replicated class is used, the problem names | **DatastoreKit**: `datastorekit/store_inventory.py` |
| **C. Fingerprint** | SHA-256 over each class's canonical record lines, newline-terminated; the same per distinct tag set; the overall digest, SHA-256 of `canonical_json({"fingerprint_format", "classes": {name: {"count", "digest"}}})`; problems counted by kind, outside every digest; `taken`, outside every digest; and the two readers of that structure, `compare_fingerprints` and `listing_lines` | **this repository**: `RunRegistry/stores.py:1164-1390` (`_sha256_lines`, `_tag_sets`, `fingerprint_of`, `compare_fingerprints`, `listing_lines`) |

Layer C is a pure function of a `StoreInventory`: it reads nothing, names no client class, and
imports only `canonical_json` from layer B. Layer A is data the client declares.

### 3.2 What `FINGERPRINT_FORMAT` has versioned

`FINGERPRINT_FORMAT` is the one version, and it enters the overall digest. Its history, from each
commit's diff **[probed]**:

| Format | Commit | What changed | Factories changed (layer A) | `store_inventory.py` changed (layer B) |
|---|---|---|---|---|
| 1 | `ef46154`, store-fingerprint | the fingerprint introduced | — | — |
| 2 | `a171ced`, datastore-integrity 07 | WKB rows keyed on their numeric parent | `GkWKBIntegration`, `TkWKBIntegration` | yes (a `nullable` parent) |
| 3 | `b18bb0b`, datastore-integrity 08 | `QuadSourceIntegral` keyed on its policy | `QuadSourceIntegral`, `QuadSourcePolicy` | no |
| 4 | `dbf2cf2`, datastore-integrity 09a | `QuadSource` keyed on `Tq`, `Tr` | `QuadSource` | yes (a `cross_shard` parent) |
| 5 | `3d7b606`, datastore-integrity 09b | `GkSource` keyed on its parent set | `GkSource`, `GkNumericIntegration`, `GkWKBIntegration`, `TkWKBIntegration` | yes (`ParentSet`) |
| 6 | `014be95`, datastore-integrity 09c | `QuadSourceIntegral` keyed on its six parents | `QuadSource`, `QuadSourceIntegral` | no |

**Every bump was driven by a change to what SGK puts in a key** — layer A. Three of them also
extended layer B, because the key needed something the encoding could not yet express (a nullable
parent, a cross-shard parent, a parent set) **[probed]**. So the one number has been doing two jobs, and the
history shows both: a client's content changing, and the shared encoding growing to carry it.

**Layer B has no version of its own.** Since format 6, `store_inventory.py` has changed in this
repository (`bc8bffc`, `6646b06`, `4c34f13`) and moved to DatastoreKit (`0c9e2cf`); `4c34f13`
records that "the fingerprint … [is] unchanged", and the rehearsal fingerprint `39809dca` has been
reproduced across every pin move since. That is what has protected it: an acceptance test, not a
version. A DatastoreKit release that changed `canonical` would change every digest while the format
stayed 6, and `compare_fingerprints` would then report a content difference in every class, not a
format mismatch. In a shared package that serves three clients, that gap matters more.

### 3.3 The split (D3)

| Layer | Owner after the extraction | Versioned by |
|---|---|---|
| A — content | **the client**: the `inventory_spec()`s on its factories (unchanged), its registry (unchanged), and **one new module** (for SGK, say `config/fingerprint.py`) holding the format number, the history of what each bump changed (today `stores.py:194-205`), and the fingerprinter wiring: `read_inventory(primary, factories)` handed to DatastoreKit's fingerprint function | **the client's format number**, `6` for SGK |
| B — encoding | DatastoreKit, unchanged | **a new DatastoreKit version** covering B and C together, since C reads B's structures and changes with them |
| C — algorithm | **DatastoreKit**, moved from `stores.py` into a module of its own beside `store_inventory` (`fingerprint_of`, `compare_fingerprints`, `listing_lines`, and their helpers) | the same DatastoreKit version |
| when and where a fingerprint is taken and recorded | **RunRegistryKit**: refused while a `running` run names the store; taken at `Run.finish(fingerprint=True)`; required to match before a retirement; written to the sidecar's `fingerprint` field; the `taken` stamp | the sidecar's `sidecar_format` |

RunRegistryKit reads a fingerprint only through `fingerprint_format`, `digest`, and the list of
differences that `compare` returns; it never looks inside `classes`. Its client config names the
fingerprinter (§5.2).

`compare` refuses — "fingerprints of different formats are not compared" — when **either** version
differs: the client's (what goes in) or DatastoreKit's (how it is encoded and digested).

This does not reverse `datastore-portability` §6.3. That decision keeps the *sidecar* out of
DatastoreKit, and it still is. A fingerprint is a statement about what a store contains, which is
DatastoreKit's own subject; the sidecar is only where RunRegistryKit records it.

### 3.4 The digest must not move

The overall digest hashes `{"fingerprint_format": 6, "classes": …}`. The extraction keeps that
object exactly: the client's format number stays the integer in that field, and DatastoreKit's new
version is recorded **beside** the digest (say `inventory_format`), not inside the hashed object.
Its acceptance is that the rehearsal fingerprint `39809dca` reproduces, as it has at every
rehearsal since `a3-v2-readiness` 07.

**There is no live sidecar to amend.** All 13 sidecars under `var/datastores/` are tombstones
(`retired` present): eleven at format 6 (digests `39809dca` and `10b8725f`) and the two
`handover-*` stores at format 1. **[probed]** The planner's earlier remark to the user, that "13
sidecars already hold a fingerprint, and retiring a store requires a fresh fingerprint to match the
recorded one", was true of the count but not of the consequence: a tombstone is never fingerprinted
again. So the new field needs no `store amend` and no reader that tolerates its absence (the
standing rule against version-aware readers). The first live store under the new layout will be
A3 v2's (D2).

What still depends on byte-identity is the **rehearsal chain**: `39809dca` is the acceptance of
`a3-v2-readiness` 07, `datastore-generic` 09 and `dk-and-rpk-migration` 02, 05 and 08, and it is
recorded on those boards and in those tombstones. Keeping it reproducible is what lets the
extraction be judged by the same test.

### 3.5 A guard the grouping makes possible

Today a bump is remembered, not enforced: a key change is caught only because a golden fingerprint
in `RunRegistry/tests/data/` stops matching, and the author then bumps by hand. With layer A and its
number in one client module, a client test can pin, for each format number, a digest of the
canonical form of every `inventory_spec()` in the registry. Changing a spec without bumping the
number then fails a test that says which. Everything in an `InventorySpec` is a string, a boolean
or a mapping of them (`store_inventory.py:207-332`) **[checked]**, with one exception: a
polymorphic `Parent.types` maps a type **value** (not necessarily a string) to a class name, and
`canonical_json` refuses a non-string key. The guard has to stringify those keys deliberately.
Whether to build it, and where, is for the SGK adoption prompt (§8, Q5).

---

## 4. The tests

`RunRegistry/tests/` holds 286 tests in 14 modules **[agent; the per-module counts sum to the
total]**. Classified by whether they would move with the package:

| Class | Tests | Modules |
|---|---|---|
| **Generic** — move unchanged, apart from `REPO_ROOT` (C1) | 31 | `test_run_registry` (29; one, `test_the_recorded_names_are_the_two_packages`, `:393`, pins C5 and becomes a config test); `test_store_fingerprint.TestFormats` (2) |
| **Generic, needs a fixture** — the behaviour is the package's, but the store is built with this repository's fixtures | 154 | `test_store_sidecar` (17), `test_store_copy_move` (27), `test_store_command_line` (2), `test_store_amend` (21), `test_store_amend_residuals` (9), `test_store_retire` (41), `test_store_retire_references` (10), `test_store_fingerprint` except `TestGolden`, `TestDrivers` and `TestFormats` (27) |
| **This repository's** — they test SGK's drivers, SGK's golden fingerprint, or `main.py`'s arguments | 101 | `test_pipeline_adoption` (15), `test_killed_handler` (26), `test_prune_on_resume` (18), `test_build_rehearsal_scope` (28), `test_quadsource_atol_sweep_prepare` (10); `test_store_fingerprint.TestGolden` (3) and `TestDrivers` (1) |

**The fixture split decides how hard the move is.**

- `StoreFixtures/shard_store_fixtures.py` imports only the standard library and `datastorekit`
  **[agent]**. The sidecar, copy, move and amend tests (76) need only this, and it can be carried
  into RunRegistryKit's tests as it is.
- `StoreFixtures/real_store_fixtures.py` imports `config.datastore`, `config.sharding` and
  `ObjectFactories.GkSource` **[agent]**. The retire, retire-references and fingerprint tests (78)
  build a full SGK store from it and fingerprint it, through `stores.py:1494`. In RunRegistryKit
  they need **a small fingerprintable store with no client**: stand-in factories that declare
  `inventory_spec()`, and a stand-in fingerprinter. DatastoreKit's tests already build stores with
  no client, so the DatastoreKit release of §6 may offer such a fixture; if not, RunRegistryKit
  writes its own.

**The 101 stay here.** They are tests of `docs/` drivers and of SGK's fingerprint, and live under
`RunRegistry/tests/` only because that is where registry adoption was tested. They need a test
root of their own when `RunRegistry/` is deleted (§8, Q6). `RunRegistry/tests/data/` holds six
golden fingerprints, formats 1–6, of SGK's full fixture store; only `…-09c.json` (format 6) is read
(`test_store_fingerprint.py:77`), and the others are kept by the rule in `stores.py:194-205`. All
six are SGK's and stay with SGK's fingerprint module.

The coupling between modules has to move intact: `store_fixtures` builds on
`test_run_registry.RegistryTestCase`, and `test_store_command_line.CommandLineTestCase` is the
base of the amend, retire and fingerprint tests. On SGK's side, `test_killed_handler` imports
from `test_pipeline_adoption`, and `test_build_rehearsal_scope` from `test_prune_on_resume`.
**[agent]**

---

## 5. RunRegistryKit

### 5.1 The repository

Modelled on RayPoolKit, the smaller of the two existing extractions **[checked]**: a `pyproject.toml`
with setuptools, `requires-python = ">=3.12"`, the package found by `include = ["runregistrykit*"]`
and its tests excluded; a `PROVENANCE.md` naming the import commit, the file map, and the
differences allowed at `v0.1.0`; a `CLAUDE.md` with the cross-repository conventions; and a guard
test that the package names no client — DatastoreKit's rule, "the package knows nothing about any
client", applied here. A fresh import, as RayPoolKit's (its U2): SGK's history stays SGK's, reached
through the import commit.

### 5.2 What a client gives it

**A config file at the client's root**, read by the package, which also marks the root (C1):

- the runs root (default `var/runs`) and the stores root (default `var/datastores`);
- the distributions to record in a manifest (C5);
- the fingerprinter, as an import path to an object (for SGK, in `config/fingerprint.py`);
- the root policy for worktrees (§2, §8 Q3).

Which file — a `[tool.runregistrykit]` table in a `pyproject.toml`, or a file of its own — is §8 Q2.
None of the three clients has a `pyproject.toml` today, and a config file that is also the root
marker is simplest as a file of its own.

### 5.3 Dependencies

The core: the standard library only, as now. The store layer: an optional extra,
`runregistrykit[stores]`, needing `datastorekit` **by name and minimum version, not by git URL**.
`datastorekit` is not on PyPI; each client installs it from its own pinned tag
(`requirements.txt:17` here). A git-URL requirement inside RunRegistryKit would conflict with the
client's pin, whereas a bare `datastorekit>=…` is satisfied by the installed tag. The extra's floor
is the DatastoreKit release that carries layer C (§6).

### 5.4 The rules travel with the code

The six rules in `CLAUDE.md` ("Long-running jobs — the run registry") are as much the product as
the code (`prompts/run-registry` README §0.3: "a registry that nobody reads is worse than none").
Their full text, with the five failures of §0 that each one answers, moves to RunRegistryKit's
README. Each client's `CLAUDE.md` keeps a short binding section that names the command to run at
the start of a session and points there.

---

## 6. DatastoreKit's part

DatastoreKit has one campaign in progress, `object-timestamp`, whose prompt 02 is the release
`v0.3.0`; SI's `datastorekit-v0.2.4-brief.md` requested that change. **[checked]** Layer C is a
separate change, made by a campaign written in DatastoreKit (its `CLAUDE.md`: "a client's change is
not made here, and this repository's change is not made in a client"). This audit can only state
what SGK needs of it:

1. `fingerprint_of`, `compare_fingerprints`, `listing_lines`, `_sha256_lines` and `_tag_sets`,
   moved from `RunRegistry/stores.py:1164-1390` into a DatastoreKit module beside
   `store_inventory`, with the client's format number as a parameter where `FINGERPRINT_FORMAT` is
   read today.
2. A version for layers B and C together, recorded in a fingerprint beside the digest (§3.4), and a
   rule in DatastoreKit's contract for when it is bumped.
3. Tests with no client: a fingerprint of a store built by DatastoreKit's own fixtures, golden, with
   its own format history.
4. The overall digest unchanged for an unchanged inventory and client format: SGK's `39809dca`
   reproduces when SGK adopts the release.

The brief to DatastoreKit is written by SGK's campaign (§9), as ChamPBH's and SI's campaigns wrote
theirs.

---

## 7. The clients

### 7.1 Where they stand

| Client | Its DatastoreKit migration | The prompt that writes the specs | The registry |
|---|---|---|---|
| ChamPBH | `dk-and-rpk-migration`, live: 01–03 landed; 04 in progress on 2026-10-12 | 09, "Port the inventory" | none. P5's helper `utilities.installed_package_provenance` (`utilities.py:177`) is `package_provenance` in shape, so that adopting the registry "is a substitution" |
| SI | `dk-and-rpk-migration`, planned: plan and board written, U3 ruled, the rest awaiting the user; not yet committed | 10, "Port the inventory" | none |

Neither has a factory that declares `inventory_spec()` today; both registries are the module-level
`_factories` in `Datastore/SQL/Datastore.py`, which each campaign's 05 moves to
`config/datastore.py`. **[agent]** So both reach a fingerprintable registry inside their own
migration, and both adopt RunRegistryKit after it closes (D4).

### 7.2 Is the planned coverage enough for a fingerprint?

Both prompts scope the specs as "each factory whose class the report covers", plus the classes
those specs depend on. A fingerprint needs more: every registered class either declares a spec, is
read through an owner's (a tag, value or parent-set table), or is excluded for a stated reason.
`inventory_classes` takes only the classes that declare a spec and passes over the rest without a
word, so a gap is an omission, not an error. SGK meets the stricter test: 38 registered classes,
21 with a spec, and the other 17 all tag, value or parent-set tables read through an owner.
**[probed]**

Measured read-only on 2026-10-12 **[agent; the SI omissions checked against its README]**:

| | Registered | Owners | Read through an owner | Not reachable by a spec | The planned list misses |
|---|---|---|---|---|---|
| ChamPBH (`Datastore.py:94-122`) | 27 | 21 | 6 (tag and value tables of `ScalarModel`, `AdiabaticHistory`, `BBNData`) | none | **nothing** |
| SI (`Datastore.py:96-127`) | 30 | 24 | 5 value tables | **1**: `GradientCoupledInstanton` has two child tables, `…Value` and `…Profile`, and a spec has one `values` slot (`store_inventory.py:293-303`) | **`redshift`** and **`IntegrationSolver`**: registered, written by no driver (SI's README §4.1), and referenced by no spec, so the dependency rule does not bring them in |

`…Profile` is written for every `GradientCoupledInstanton` row and holds the radial compaction
profile, which the factory calls the science output; `…Value` is written only when
`store_full_values` is set. **[agent]** SI's gradient-coupled code is halted (its README §1), but
its tables are registered and its stores exist.

### 7.3 The amendments (D5)

**ChamPBH**, committed `64ef9aa` on its `dk-and-rpk-migration` branch, touching only
`prompts/dk-and-rpk-migration/README.md` and `IMPLEMENTATION_STATE.md` (prompt 04's staged work in
that tree was left staged and untouched; ChamPBH's `CLAUDE.md` provides for planning commits
interleaved with a prompt's):
- 09 gains **acceptance item 5**: a test that puts every registered class in exactly one of three
  places — a spec, an owner's slot, or a named exclusion with a reason — and fails on a class in
  none and on an exclusion naming a class no longer registered, with two deliberate breakages;
- README **§6.5** records the ruling and the measurement (the exclusion table starts empty);
- the board's Decisions section records it.

**SI**, in its plan's working files, **not committed**: the whole `dk-and-rpk-migration` folder is
still untracked there, awaiting the user's review of the plan, so the amendment goes in with it:
- 10 gains the same **acceptance item 5**, naming the three cases it has to settle and record:
  a spec (preferred) or an exclusion for `redshift` and `IntegrationSolver`; and which of
  `GradientCoupledInstanton`'s two child tables takes the slot, the other being excluded and opened
  as an issue;
- README **§6.1** gains **U8**, ruled; **§6.3** gains a question for the user at 10: whether to ask
  DatastoreKit for more than one value table per spec;
- the board records U8 in its Decisions table, its status line and a new §5.2.

Neither amendment brings RunRegistryKit into those campaigns' scope.

---

## 8. Open questions for the user

Each has a recommendation, given first. **Ruled by the user on 2026-10-12: all eight as
recommended** ("Accept all §8 recommendations, write the campaign"). The campaign that carries them
is [`prompts/runregistry-extraction/`](../prompts/runregistry-extraction/README.md), whose README
§6.1 records them beside D1–D5.

| | Recommended | The alternative |
|---|---|---|
| **Q1** — names | Repository `RunRegistryKit`, distribution and package `runregistrykit`, CLI `python -m runregistrykit`, matching the other two | Keeping the name `RunRegistry`: shorter, but it breaks the pattern, and the package name would then differ from the repository's only in case |
| **Q2** — the client config | **A file of its own at the root**, read with the standard library's `tomllib`, which is also the root marker (C1). None of the three clients has a `pyproject.toml` | A `[tool.runregistrykit]` table in a `pyproject.toml`, which each client would first have to create |
| **Q3** — a run launched from a worktree | **Recorded in the main checkout**: the root is the parent of `git rev-parse --git-common-dir`, so `var/runs/` survives the worktree (`CLAUDE.md` rule 5). The run's git SHA and script hash are still the worktree's, which is what provenance needs | Recorded in the worktree, as today by accident (§2). Lost with it |
| **Q4** — one distribution or two | **One, with the store layer as the extra `[stores]`** (§5.3). `Run.finish(fingerprint=True)` is the one call from the core into it, and it already imports lazily | Two distributions: a cleaner dependency graph, two release cadences for one convention |
| **Q5** — the spec-digest guard (§3.5) | **Build it in SGK's adoption**, in the client fingerprint module's tests, and offer the pattern to the other two clients through RunRegistryKit's adoption checklist | Leave the bump to the golden fingerprint, as now |
| **Q6** — a home for SGK's 101 driver tests (§4) | **A test root of their own**, for the `docs/` drivers that launch registered runs, added to `CLAUDE.md`'s list of roots | Folding them into `tools/tests/`, which is for `tools/` |
| **Q7** — a format field in the run manifest | **Not during the extraction**, which is a move. Proposed separately, after it: three clients and a package of its own make a version on the one record that has none worth having | Adding it during the move: a new field in every new manifest, while the acceptance is that nothing changes |
| **Q8** — more than one value table per spec | **Left to SI's prompt 10** (§7.3), which measures whether it is needed. If the user then asks DatastoreKit for it, it rides with §6's release or the one after | Asking DatastoreKit now, before SI has settled which table matters |

---

## 9. Order of work

Sequenced by what each step needs to exist, not by convenience.

1. **DatastoreKit: layer C and its version** (§6). A campaign written in DatastoreKit, from a brief
   that SGK's campaign writes, and released after `object-timestamp`'s `v0.3.0`. It needs nothing
   from RunRegistryKit; RunRegistryKit's store layer needs it.
2. **RunRegistryKit `v0.1.0`: a fresh import** of `RunRegistry/` from a frozen SGK commit, as
   RayPoolKit was imported (`PROVENANCE.md`, the file map, the differences allowed): C1–C9 made
   configuration, layer C replaced by the DatastoreKit call and a client fingerprinter, the 185
   generic tests carried with a client-free fingerprintable fixture (§4), a no-client guard test,
   and the rules in its README. **SGK's `RunRegistry/` is frozen from the import commit until SGK
   adopts**, so that the two never diverge.
3. **SGK adopts both**: the DatastoreKit pin moves from `v0.2.2` past `v0.2.3` and `v0.3.0` to the
   release of step 1, in a commit of its own; `runregistrykit` is pinned at its tag; SGK's
   fingerprint module is written (format `6`, its history, the wiring, the six golden files); the
   live callers of §2.1 are rewritten; the 101 tests get their root; `RunRegistry/` is deleted;
   `CLAUDE.md`'s section becomes a pointer plus the binding rules. **Acceptance:** the rehearsal
   fingerprint reproduces `39809dca` through both packages; `list` gives the same runs and states
   as at `d03a643`; every manifest and tombstone under `var/` reads without a problem.
4. **A3 v2**, through the packages (D2). Its store is the first live store in the new layout.
5. **ChamPBH and SI**, each after its `dk-and-rpk-migration` closes, in a campaign of its own: the
   config file, a fingerprint module over the specs that 09 and 10 wrote, and the substitution of
   P5's helper by `package_provenance`.

Steps 1–3 follow this repository's campaign conventions and the user's audit-then-prompts pattern:
the SGK campaign, [`prompts/runregistry-extraction/`](../prompts/runregistry-extraction/README.md), owns step 3 and writes the briefs for
steps 1 and 2, whose campaigns are written in their own repositories, as DatastoreKit's and
RayPoolKit's `extraction` campaigns were.

---

## 10. Observations not acted on

- **A fingerprint counts value rows; it does not hash them.** `Record.value_count` is a count per
  owner (`store_inventory.py`, the module docstring), so a fingerprint proves identities, tags,
  validated flags and row counts, not computed values. This is by design ("a computed-values field
  can be added to `Record` later") and is not changed by anything here. It bears on §7.2: even the
  `GradientCoupledInstanton` table that takes the slot is counted, not hashed.
- **Class coverage is not key completeness.** The §7.3 test proves that every class is read; it
  does not prove that a spec's key holds every column that distinguishes two rows. ChamPBH's
  `ScalarModel` has three polymorphic references with no foreign key (cosmology, potential,
  coupling) **[agent]**, which 09's list does not mention, though its charter derives leaves and
  parents from each factory's `build` lookup, which would include them.
- `docs/dk-and-rpk-migration/rehearsal/stage_counts.py` reaches into the private
  `RunRegistry._slug` **[agent]**: the rewrite in step 3 should use a public name.
- No issue is opened on this repository's boards or in `docs/OPEN_ISSUES.md` by this audit: the
  gaps it finds are the work of the campaigns in §9, which open them on their own boards.
