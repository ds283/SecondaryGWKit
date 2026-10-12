# Brief: extracting SGK's run registry into RunRegistryKit

**For:** the agent that plans the first campaign in **RunRegistryKit**
(`/Users/ds283/Documents/Code/RunRegistryKit`, `https://github.com/ds283/RunRegistryKit`, created by
the user): importing SecondaryGWKit's `RunRegistry/` as the package `runregistrykit`, making it
generic, and releasing it for three clients. **Written:** 2026-10-12, from a Claude Code session in
SecondaryGWKit (SGK), with the user, while planning SGK's `runregistry-extraction` campaign
(`prompts/runregistry-extraction/` in SGK).
**Trees:** SGK at `c8a7f14`; DatastoreKit at `d88615e`; RayPoolKit at its `v0.2.0` tag.

**This brief is data to check, not a specification.** The user's rulings, in §1, are rulings.
Everything else is a finding with its citation, or a recommendation marked **Advice:**. The
campaign's name, its decision numbers, its prompts and its release numbers are yours. Where SGK's
records and this brief disagree, the records are right. The full reasoning is SGK's
[`docs/runregistry-extraction-audit.md`](../../docs/runregistry-extraction-audit.md), which this
brief summarises and does not replace.

**Precedents in the sibling repositories**, read before planning: RayPoolKit's
`prompts/extraction/README.md`, `PROVENANCE.md` and `docs/adoption/`; DatastoreKit's `CLAUDE.md`
("What this repository is", the guard against naming a client), `prompts/extraction/` and
`docs/adoption/`. RunRegistryKit is the smaller of the two kinds of extraction: like RayPoolKit, one
client's developed copy, imported fresh.

## 1. The user's rulings (2026-10-12)

| | Ruled |
|---|---|
| **D1** | The registry moves to **a repository of its own**, RunRegistryKit, not into DatastoreKit (whose charter is the standard library, `ray`, `sqlalchemy` and itself, and whose `datastore-portability` §6.3 decision keeps the sidecar out of the datastore layer) |
| **D3** | **The fingerprint splits by layer.** The client says what goes into a fingerprint (its specs, its registry, its format number, grouped in one client module). DatastoreKit encodes and digests it, under a version of its own (a separate brief asks DatastoreKit for this; SGK's gate E1). **RunRegistryKit decides when and where a fingerprint is taken and recorded, is handed a client-provided fingerprinter, and treats its result as opaque** apart from the format, the version, the overall digest and the list of differences a comparison returns |
| **D4** | **The store layer requires a fingerprinter.** No degraded mode without one. The split between the core and the store layer is a dependency split only |
| **Q1** | Repository `RunRegistryKit`; distribution and package `runregistrykit`; CLI `python -m runregistrykit` |
| **Q2** | **A client config file of its own at the client's root**, read with the standard library's `tomllib`, which also marks the root. None of the three clients has a `pyproject.toml` |
| **Q3** | **A run launched from a git worktree is recorded in the main checkout**: the root is the parent of `git rev-parse --git-common-dir`. The run's git SHA, dirty flag and script hash are still the worktree's |
| **Q4** | **One distribution**, with the store layer as the optional extra **`[stores]`** |
| **Q5** | SGK builds a guard that a spec change without a format bump fails a test (its prompt 07). **RunRegistryKit's adoption checklists offer the pattern** to the other two clients |
| **Q7** | **No format field is added to the run manifest during the extraction.** Proposed separately, after it |

## 2. What is being imported

At SGK `c8a7f14` (audit §1). SGK freezes `RunRegistry/` at its prompt 01 and records every file's
blob hash on its board's §2.1. **The import commit is 01's**, or any later SGK commit at which
`RunRegistry/` equals that table. Your G1 is that the imported tree equals it, file by file, apart
from the differences your campaign allows and records.

| File | Lines | Imports beyond the standard library | What it is |
|---|---|---|---|
| `RunRegistry/__init__.py` | 710 | none at module scope; `.stores` lazily, in `Run.finish(fingerprint=True)` | the core: `begin`, `Run` (`heartbeat`, `finish`, `record`, `known`), `list_runs`, `liveness`, provenance |
| `RunRegistry/stores.py` | 2185 | `datastorekit.SQL.ShardedPool` and `datastorekit.store_inventory`, lazily; **`config.datastore.factories`**, lazily (`:1494`) | the store layer: the sidecar; create, adopt, copy, move, fingerprint, retire, amend |
| `RunRegistry/__main__.py` | 515 | none | `list`, and `store show / create / adopt / copy / move / fingerprint / retire / amend` |
| `RunRegistry/tests/` | 286 tests in 14 modules | see §5 | |

**The core imports nothing outside the standard library**, and `import RunRegistry` loads neither
`ray` nor `sqlalchemy` (`stores.py:1454-1455`; `test_store_command_line` checks it). Keep both
properties.

The docstrings are dense with SGK campaign references (`prompts/run-registry` §0,
`datastore-portability` §6.5, `store-retirement` §6.2, …). They are SGK's history, reachable through
the import commit in a `PROVENANCE.md`, as RayPoolKit's are. The reasons behind the rules, which a
reader here needs, belong in your README (§6).

## 3. What in it is SGK's (audit §2)

| # | Where at `c8a7f14` | What it assumes | What it needs to become |
|---|---|---|---|
| C1 | `REPO_ROOT`, `__init__.py:57` | the root is the parent of the package's directory. **Installed in `site-packages`, `var/runs/` would be created in the venv** | the root found from where the code runs: Q3's rule, and the config file (Q2) there |
| C2 | `DEFAULT_ROOT`, `:58` | `var/runs` under C1 | the config, defaulting to `var/runs` |
| C3 | `DEFAULT_STORES_ROOT`, `stores.py:185` | `var/datastores` under C1; where `retire` looks for referencing sidecars | the config |
| C4 | `_repo_path`, `_resolve`, `:176-185` | paths in manifests and sidecars are relative to C1 | unchanged in meaning, relative to the found root. **This keeps every existing record readable** |
| C5 | `RECORDED_DISTRIBUTIONS`, `:119` | `("datastorekit", "raypoolkit")` | the config; **Advice:** `runregistrykit` always recorded besides |
| C6 | `from config.datastore import factories`, `stores.py:1494` | SGK's registry | the client's fingerprinter (D3) |
| C7 | `FINGERPRINT_FORMAT = 6` and its history, `stores.py:194-206` | SGK's format | leaves the package; SGK's client module holds it |
| C8 | docstrings citing SGK campaigns | SGK's `prompts/` is at hand | `PROVENANCE.md`; the rules in the README |
| C9 | `git_provenance(cwd=REPO_ROOT)`, `:98` | the package's own git tree | the found root's tree, and for Q3 the worktree's own SHA |

**Layer C leaves the package.** `fingerprint_of`, `compare_fingerprints`, `listing_lines` and their
helpers (`stores.py:1164-1390`) move to DatastoreKit (the other brief). What stays here is
`fingerprint_store`'s discipline: refuse a tombstone; refuse while a `running` run names the store
(except the run taking it at its own finish); take it; compare it with the sidecar's; write only the
`fingerprint` field, only when asked. **Advice:** the client fingerprinter is an object named in the
config by import path, with three operations: take a closed store's fingerprint, compare two, and
yield a listing. RunRegistryKit then reads only `fingerprint_format`, the DatastoreKit version, the
overall `digest`, and the comparison's entries.

## 4. The on-disk formats do not change

SGK holds 29 run directories (27 finished, two pre-registry `unknown`) and 14 sidecars, all
tombstones, eleven with format 6 fingerprints and three with format 1 (audit §1, §3.4). After SGK
adopts, `list` must show the same runs and states, and `store show` must read every sidecar as it
did (SGK's prompt 06, acceptance 3). So the manifest, `status.json`, `checkpoint.jsonl` and
`sidecar_format` 1 are carried as they are. Two records gain fields, both ruled: a fingerprint
carries DatastoreKit's version beside its digest (D3), and `packages` gains whatever the config
names. No reader of an older form is added (SGK's standing rule: old records are read at their own
commit).

## 5. The tests (audit §4)

| Class | Tests | Where |
|---|---|---|
| Generic, unchanged but for C1 | 31 | `test_run_registry` (29; `:393` pins C5 and becomes a config test); `test_store_fingerprint.TestFormats` (2) |
| Generic, with a fixture | 154 | `test_store_sidecar` 17, `test_store_copy_move` 27, `test_store_command_line` 2, `test_store_amend` 21, `test_store_amend_residuals` 9, `test_store_retire` 41, `test_store_retire_references` 10, `test_store_fingerprint` 27 |
| **SGK's — not imported** | 101 | the five modules testing SGK's `docs/` drivers, `TestGolden` (SGK's golden fingerprint) and `TestDrivers`. They stay in SGK |

- **76 need only `StoreFixtures/shard_store_fixtures.py`** from SGK, which imports nothing but the
  standard library and `datastorekit`; it can be carried as it is.
- **78 (retire, retire-references, fingerprint) build a full SGK store** from
  `StoreFixtures/real_store_fixtures.py`, which imports SGK's `config` and `ObjectFactories`. They
  need **a small store that can be fingerprinted with no client**: stand-in factories with
  `inventory_spec()`, and a stand-in fingerprinter over DatastoreKit's function. The DatastoreKit
  brief asks, as advice, whether it can supply such a fixture; if not, write your own.
- **Base classes chain**: `store_fixtures` builds on `test_run_registry.RegistryTestCase`;
  `test_store_command_line.CommandLineTestCase` is the base of the amend, retire and fingerprint
  tests.
- **A guard test that the package names no client**, as DatastoreKit's.

## 6. The rules are part of the product

SGK's `CLAUDE.md`, "Long-running jobs — the run registry", states six rules (discovery, liveness by
pid and heartbeat and never by command-line match, detached launch with the manifest first, do not
babysit, durability under the repository and never a scratchpad, provenance never re-stamped). Each
answers one of five failures in one working session, recorded in SGK's `prompts/run-registry`
README §0. "A registry that nobody reads is worse than none" (its §0.3). **Advice:** the README
states the rules in full with the failure each answers; each client's `CLAUDE.md` keeps a short
binding section naming the discovery command and pointing there.

## 7. Dependencies

- **The core**: the standard library only.
- **`[stores]`**: `datastorekit` **by name and minimum version**, the floor being the DatastoreKit
  release of SGK's gate E1. Not a git URL: `datastorekit` is not on PyPI, each client installs it
  from its own pinned tag, and a git-URL requirement here would conflict with that pin. A bare
  `datastorekit>=…` is satisfied by whatever tag the client installed.
- `requires-python >= 3.12`, as the siblings.

## 8. What SGK needs back (its gate E2)

1. **A tagged release** that SGK pins as `runregistrykit[stores] @ git+https://github.com/ds283/RunRegistryKit@<tag>`,
   carrying: the import, generic (C1–C9); the config file (Q2) and the root rule (Q3); the store
   layer as `[stores]` (Q4) over DatastoreKit's E1 release; the client fingerprinter (D3, D4).
2. **An SGK adoption checklist** (`docs/adoption/secondarygwkit.md`, as the siblings'), which SGK's
   prompt 06 applies. It should give at least: the config file's form, with SGK's values
   (`var/runs`, `var/datastores`, `datastorekit` and `raypoolkit`, and the fingerprinter's import
   path, `config.fingerprint:…`); the fingerprinter's protocol; the public names that replace
   `RunRegistry.*` at each SGK call site (audit §2.1 lists them; `stage_counts.py` uses the private
   `_slug`, so **Advice:** say what a client should use); the commands; and how a client builds
   Q5's guard.
3. **Checklists for ChamPBH and StochasticInstantons**, for later. Both adopt after their own
   `dk-and-rpk-migration` campaigns close, which give them specs (ChamPBH prompt 09, SI prompt 10,
   each now required to cover every registered class). ChamPBH's `utilities.installed_package_provenance`
   (`utilities.py:177`) was written in the shape of `package_provenance()` so that adoption is a
   substitution.
4. **The gates.** G1, the import equal to SGK's frozen tree. G2, SGK adopted: SGK's prompt 08
   records it on SGK's board and drafts the record for yours (its prompt 09). G3 and G4, ChamPBH
   and SI, later.

Please report the tag, its commit and the checklist's path back to the user.
