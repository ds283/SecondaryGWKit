# Campaign — runregistry-extraction

**Written:** 2026-10-12 on branch `handover-remedial` at `c8a7f14`, by Claude Opus 5.5. It is
planned from [`docs/runregistry-extraction-audit.md`](../../docs/runregistry-extraction-audit.md)
and from the user's decisions of the same day: D1–D5, taken while the audit was written, and Q1–Q8,
the audit's §8, accepted as recommended (§6.1).
- **Briefs, written by this plan for the two other repositories:**
  [`datastorekit-fingerprint-brief.md`](datastorekit-fingerprint-brief.md) and
  [`runregistrykit-extraction-brief.md`](runregistrykit-extraction-brief.md).

§0.2 measures the start tree. §2 fixes each prompt's charter and acceptance. **01–08 are written**:
03 and 04 on 2026-10-14 after DatastoreKit's `v0.4.0` (E1), and 05–08 on 2026-10-15 after
RunRegistryKit's `v0.1.0` and its SGK checklist (E2); 09 follows 08's review (§2.1).

## 0. Why this campaign exists

`RunRegistry/` was built here for this repository's long jobs (`prompts/run-registry` README §0).
ChamPBH and StochasticInstantons (SI) need the same thing, and the user decided on 2026-10-12 that
it moves to a repository of its own, **RunRegistryKit** (package `runregistrykit`), beside
DatastoreKit and RayPoolKit (audit §0, D1), **before the A3 v2 compute run** (D2).

The move is not a copy. The audit (§2) finds nine places where the registry assumes it lives here,
and one design question it settles with the user (§3, D3): the store fingerprint splits by layer.
- **What goes into a fingerprint** stays in the client: the `inventory_spec()`s on the factories,
  the registry, and a **format number grouped with them in one client module** (for SGK,
  `config/fingerprint.py`, format `6`).
- **How it is encoded and digested** goes to DatastoreKit, beside `store_inventory`, under a
  **version of its own**. Today that encoding has no version, and every one of the six format bumps
  came from a key change here (audit §3.2).
- **When and where a fingerprint is taken and recorded** is RunRegistryKit's. It is handed a
  client-provided fingerprinter, and a client without one cannot use its store layer (D4).

So three repositories move, in an order fixed by what each needs (audit §9):

```
SGK 01 freeze ─ SGK 02 R0 ─┬─ [E1: DatastoreKit's fingerprint release] ─ SGK 03 pin DK ─ SGK 04 R1 ─┐
                           └─ [E2: RunRegistryKit's first release, imported from 01's tree] ────────┴─ SGK 05 pin RRK ─ 06 adopt ─ 07 guard ─ 08 R2 ─ 09 close ─ A3 v2
```

E1 and E2 are campaigns in DatastoreKit and RunRegistryKit, planned there from this folder's two
briefs (§7). This campaign makes no change in either repository (§5 rule 12). RunRegistryKit's
store layer needs E1's release, so E2's release follows E1's; how RunRegistryKit stages its own
work is its campaign's to decide.

### 0.1 Why three rehearsals

The reference fingerprint is format 6, digest
`39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, 21 classes
(`docs/dk-and-rpk-migration-verification.md` §3). It is the acceptance of every rehearsal since
`a3-v2-readiness` 07.
- **R0 (02) is needed.** The last rehearsal was `dk-and-rpk-migration` 08, at `f539015`. Since then
  `511ef5f` moved the object factories to `ObjectFactories/`, which changes the module path of every
  class Ray pickles, and no build has run through it (§0.2). If R1 missed without R0, nothing could
  say whether the move or the DatastoreKit pin caused it.
- **R1 (04) isolates the DatastoreKit pin**, which moves three releases (`v0.2.2` → the release of
  E1, through `v0.2.3` and `v0.3.0`). R1 also takes each store's fingerprint a second way, through
  DatastoreKit's new code with SGK's format 6, and requires the two to be equal, before anything
  here depends on that code (§6.2 C3).
- **R2 (08) isolates the RunRegistryKit swap**: the code that registers the runs, takes the
  fingerprints and writes the sidecars.

### 0.2 Reconciliation against `c8a7f14`

Measured on 2026-10-12, read-only.

**The registry.** `python -m RunRegistry list` shows 29 runs: 27 finished and 2 `unknown` (the
pre-registry `a3-pilot` pair), none `running`. All 14 sidecars under `var/datastores/` are
tombstones: eleven at format 6 (digests `39809dca…` and `10b8725f…`) and three at format 1 (the
`handover-*` stores and the A3 backup's, in `backup-pre-resume-20260921T091011/`). There is no live store (audit §3.4).

**The venv.** `datastorekit` 0.2.2 and `raypoolkit` 0.2.0, as pinned at `requirements.txt:17` and
`:76`.

**The seven test roots**, by discovery (`countTestCases`):

| Root | Tests |
|---|---|
| `AdaptiveLevin/tests` | 32 |
| `ComputeTargets/tests` | 693 |
| `CosmologyModels/tests` | 39 |
| `LiouvilleGreen/tests` | 148 |
| `ObjectFactories/tests` | 271 |
| `RunRegistry/tests` | 286 |
| `tools/tests` | 49 |

These are discovery counts, not runs. 01 runs every root and records the results.

**Since the last rehearsal** (`f539015`): `5943492` closed `dk-and-rpk-migration`; `718e650`
recorded the retirement of its six stores; `511ef5f` moved the factories; `7d76d3e` gave the store
fixtures and the inventory-report tests their own homes; `d03a643` deleted the orphaned schema
snapshots; `a97a59c` and `c8a7f14` are the audit. Only `511ef5f` and `7d76d3e` touch code that a
build or a test runs. `511ef5f`'s message records that the roots pass and a Ray task round-trips
all 38 factories, but not a build.

**Who calls the registry** (audit §2.1): `main.py` does not. Two drivers register the runs:
`docs/gktk-remedial/scoped_pipeline_run.py`, which A3 v2 will use, and
`docs/handover/quadsource_atol_sweep.py`, which the rehearsal uses. One factory test imports
`fingerprint_of`. The rest are rehearsal and probe tooling under `docs/`, and campaign scripts under
`prompts/`.

## 1. Scope

**In scope:**
- freezing `RunRegistry/` from the commit RunRegistryKit imports (01);
- the three rehearsals, R0, R1 and R2, and their verification document (02, 04, 08);
- `requirements.txt`: the `datastorekit` pin moved, and the `runregistrykit` pin added (03, 05);
- SGK's client side of RunRegistryKit (06): the config file at the root (Q2), its fingerprint
  module `config/fingerprint.py`, every `RunRegistry` import outside `prompts/` rewritten,
  `RunRegistry/` deleted, its 101 SGK tests moved to their own roots (Q6), and `CLAUDE.md`;
- the spec-digest guard (07, Q5);
- the two issues this plan opens (§6.2 C8), and the gates recorded (§2, E1, E2; 08's RunRegistryKit
  G2).

**Out of scope:**
- **What the code writes.** No table, column, key or stored value changes, and the fingerprint's
  digest must not move (§5 rule 9).
- **DatastoreKit and RunRegistryKit themselves** (§5 rule 12). Their work is the briefs' (§7).
- **RayPoolKit**, untouched.
- **ChamPBH and StochasticInstantons.** Their spec coverage is ruled in their own plans (audit §7.3);
  their adoption of RunRegistryKit is theirs, after their migrations close.
- **`prompts/`** and **dated `docs/` prose**: records of the tree they describe (§6.2 C7).
- **Every store under `var/`** but the six this campaign builds (§3).
- **The A3 v2 build itself.**

## 2. Prompts

| # | Prompt | Covers | Model | Status |
|---|---|---|---|---|
| 01 | [`01-freeze-the-run-registry.md`](01-freeze-the-run-registry.md) | The freeze, in `CLAUDE.md`; the frozen tree recorded file by file; the baselines 06 and 08 compare with | Sonnet | written 2026-10-12 |
| 02 | [`02-rehearse-the-start-tree.md`](02-rehearse-the-start-tree.md) | **R0.** §4.1 at the start tree; every fingerprint `39809dca…`. Closes the `f539015` → `511ef5f` gap (§0.1) | Opus | written 2026-10-12 |
| 02a | [`02a-the-rehearsal-lock-helper.md`](02a-the-rehearsal-lock-helper.md) | **The lock helper**: §4.1's lock steps as `tools/rehearsal_lock.py`, with a check for this repository's Ray processes; proved against SI's and ChamPBH's scripts and on the real lock; `CLAUDE.md`'s pointer | Sonnet | written 2026-10-12 |
| 03 | [`03-pin-datastorekit-at-v0.4.0.md`](03-pin-datastorekit-at-v0.4.0.md) | The pin moved from `v0.2.2` to `v0.4.0`, in a commit of its own; the venv installed from it | Sonnet | written 2026-10-14 |
| 04 | [`04-rehearse-through-the-new-datastorekit.md`](04-rehearse-through-the-new-datastorekit.md) | **R1**; the fingerprint taken both ways and equal | Opus | written 2026-10-14; held: 03 |
| 05 | [`05-pin-runregistrykit-at-v0.1.0.md`](05-pin-runregistrykit-at-v0.1.0.md) | The pin, with the `[stores]` extra, at `v0.1.0`, in a commit of its own | Sonnet | written 2026-10-15 |
| 06 | [`06-adopt-runregistrykit.md`](06-adopt-runregistrykit.md) | **The swap**: the config file, `config/fingerprint.py`, the imports, `RunRegistry/` deleted, the 101 tests moved, `CLAUDE.md` | Opus | written 2026-10-15; held: 05 |
| 07 | [`07-guard-the-fingerprint-format.md`](07-guard-the-fingerprint-format.md) | **Q5**: a test that a spec change without a format bump fails | Sonnet | written 2026-10-15; held: 06 |
| 08 | [`08-rehearse-through-both-packages.md`](08-rehearse-through-both-packages.md) | **R2**; **RunRegistryKit's G2** | Opus | written 2026-10-15; held: 07 |
| 09 | Close the campaign | the records, the gates drafted for the two repositories, the hand-over to A3 v2 | Sonnet | held: 08 |

### 2.1 When each prompt file is written

The charters below fix every prompt's task now. The files are written in batches, each when what it
must cite exists (the user's pattern: a prompt is written once its task is determined, and held
while an earlier step establishes its method):
- **01–02 are written now.** Their task depends on nothing outside this repository.
- **02a** was written after R0, when the user adopted the shared rehearsal lock (§4.1). It depends
  on nothing held, and lands before 04, the first rehearsal to take the lock.
- **03–04** are written when E1's release is tagged: they cite its tag, its commit and its
  SGK-facing note.
- **05–08** are written when E2's release is tagged and its SGK adoption checklist exists: 06 is
  that checklist applied, and its counts and names come from it. (Written 2026-10-15, at
  `14bcac0`, from `v0.1.0` and the checklist at RunRegistryKit's `aaab408`; board §5.10.)
- **09** follows 08's review.

E1 and E2 run in parallel with 02. Neither waits on R0, but 03 does: a miss at R0 is SGK's to
explain before any pin moves.

### 2.2 The charters

**01 — Freeze `RunRegistry/` while it moves to RunRegistryKit** (Sonnet).
- **What it changes.**
  - `CLAUDE.md` gains a short section, beside "Long-running jobs — the run registry": `RunRegistry/`
    is frozen from this commit until 06, because RunRegistryKit imports it; a defect found in it is
    recorded on this board and fixed in RunRegistryKit, not here; the rules of the registry section
    still bind.
  - The board records **the frozen tree**: every file under `RunRegistry/` with its blob hash at
    01's commit, so that E2's import can be checked file by file (its G1).
  - **The baselines** for 06 and 08, saved under `measure/out/`: `python -m RunRegistry list`; `store
    show` on each of the 14 sidecars; and every root run once, with its result.
- **Acceptance.**
  1. `git diff --stat c8a7f14 <01's commit> -- RunRegistry/` is empty: the freeze changes no file
     it freezes.
  2. The frozen-tree table lists every tracked file under `RunRegistry/`, and `git ls-tree -r` at
     01's commit agrees with it.
  3. `list` shows §0.2's 29 runs; each `store show` reads its sidecar as a tombstone with no
     problem.
  4. The seven roots pass at §0.2's counts, the known wall-clock flake aside (§8).
- **Out.** Any change under `RunRegistry/`. Any brief or other repository.

**02 — Rehearse the start tree, R0** (Opus).
- **What it does.** §4.1's protocol, slug prefix `runregistry-extraction-02`, at 01's tree.
- **Acceptance.**
  1. L1–L3 hold. Both stores are `39809dca…`, format 6, all 21 classes, equal to each other and to
     `dk-and-rpk-migration` 08's recorded fingerprints (its R2 `reference` and `resumed-c` run
     directories).
  2. Every manifest's `packages` names `datastorekit` 0.2.2 and `raypoolkit` 0.2.0, not editable,
     and every `git_dirty` is `false`.
  3. L4 against R2 of `dk-and-rpk-migration`: the reference's stage counts equal, the resumed
     chain's identities hold, and the policy statistics' counts equal, compared as sorted lines
     (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`).
  4. `docs/runregistry-extraction-verification.md` is created, with §1, R0.
- **Out.** Any code or test. Retiring a store.

**02a — A helper for the shared rehearsal lock** (Sonnet). Added 2026-10-12, after R0.
- **What it changes.**
  - `tools/rehearsal_lock.py`, standard library only, with five commands: `acquire`, `set-pid`,
    `release`, `status` and `ray-processes`. Together they are §4.1's lock steps, following SI's
    brief §2. The helper never waits for, retries or breaks a lock, and never starts or stops Ray.
  - Its tests, in `tools/tests/`. They use no Ray and never touch the real lock.
  - `CLAUDE.md` gains a short section on the shared lock.
- **Acceptance.**
  1. Each command refuses what the contract forbids, and its tests catch six named mutations.
  2. SI's and ChamPBH's scripts, at `8e04a7c` and `288b385`, and the helper each respect the
     other's lock, under a temporary `HOME`.
  3. On the real lock, with a one-CPU in-process probe, `release` is refused while the probe is
     alive and succeeds once its processes are gone.
  4. The seven roots pass; `tools` gains the new tests.
- **Out.** Any rehearsal tool, driver or `extract_*.py`; the `auto` default (assigned to
  `handover`); `RunRegistry/`; other repositories.

**03 — Pin `datastorekit` at E1's release** (Sonnet).
- **What it changes.** `requirements.txt:17`, the one line, from `v0.2.2` to E1's tag. The venv is
  installed from it after a dry run.
- **Acceptance.**
  1. The dry run moves only `datastorekit`; after install it is at the tag's commit, not editable.
  2. The seven roots pass at 02a's counts (§0.2's, with `tools` 91), apart from any count E1's
     SGK-facing note says changes, and why. (`v0.4.0`'s note says none does.)
  3. The SGK-facing notes of `v0.2.3`, `v0.3.0` and E1's release are each read, and every item they
     ask of SGK is either done here, by name, or shown not to apply.
- **Out.** Any use of DatastoreKit's new fingerprint code (06).

**04 — Rehearse through the new DatastoreKit, R1** (Opus).
- **What it does.** §4.1's protocol, slug prefix `runregistry-extraction-04`, at 03's tree.
- **Acceptance.**
  1. L1–L3 hold, `39809dca…` as at 02, and every manifest's `packages` names E1's release.
  2. **The fingerprint both ways.** For each store, DatastoreKit's new fingerprint function, given
     `read_inventory(primary, factories)` and SGK's format 6, gives the same overall digest, the
     same per-class and per-tag-set digests, and the same counts as `RunRegistry.stores.fingerprint_of`
     (§6.2 C3): every field equal but `encoding_format`, which the package adds and SGK's code
     lacks. On R1's two stores and R0's two, which are still live. Run as a verification script
     under `docs/runregistry-extraction/`, not a test.
  3. L4 against R0.
- **Stop.** A miss in (2) with (1) holding is a DatastoreKit defect (§5 rule 11).

**05 — Pin `runregistrykit` at E2's release** (Sonnet).
- **What it changes.** `requirements.txt` gains `runregistrykit[stores] @ git+https://github.com/ds283/RunRegistryKit@v0.1.0`,
  in a commit of its own. The venv is installed from it after a dry run.
- **Acceptance.**
  1. The dry run adds only `runregistrykit`; `datastorekit` stays at 03's release, which satisfies
     the extra's floor (audit §5.3).
  2. Nothing imports `runregistrykit` yet; the seven roots pass at 03's counts.
- **Out.** Any import of it (06).

**06 — Adopt `runregistrykit`** (Opus).
- **What it changes.** E2's SGK checklist, applied. At least:
  - **the config file** at the root (Q2), whose form E2 defines, naming `var/runs`,
    `var/datastores`, the distributions `datastorekit` and `raypoolkit`, and the fingerprinter;
  - **`config/fingerprint.py`**: `FINGERPRINT_FORMAT = 6`; the history of formats 1–6, moved
    verbatim from `RunRegistry/stores.py:194-205`; and the fingerprinter, which hands
    `read_inventory(primary, factories)` and the format to DatastoreKit's fingerprint function;
  - **every `RunRegistry` import outside `prompts/` rewritten** (audit §2.1): the two drivers,
    `ObjectFactories/tests/test_quadsource_policy_key.py:411` (to DatastoreKit's `fingerprint_of`),
    and the probes and rehearsal tooling under `docs/`, `stage_counts.py`'s private `_slug`
    included;
  - **the 101 SGK tests moved** (§6.2 C5): `TestGolden` and the six golden files to
    `ObjectFactories/tests/`; the driver tests (`test_pipeline_adoption`, `test_killed_handler`,
    `test_prune_on_resume`, `test_build_rehearsal_scope`, `test_quadsource_atol_sweep_prepare`, and
    `TestDrivers`) to a new root, `RunDrivers/tests/`;
  - **`RunRegistry/` deleted**, its other 185 tests now RunRegistryKit's;
  - **`CLAUDE.md`**: the registry section becomes the binding rules and a pointer to
    RunRegistryKit's README, with `python -m runregistrykit list` as the discovery command; the
    freeze section goes; the package section names RunRegistryKit and its pin; the test roots list
    `RunDrivers` in place of `RunRegistry`.
- **Acceptance.**
  1. No `RunRegistry` import, string literal or `python -m RunRegistry` remains outside `prompts/`
     and dated `docs/` prose (§6.2 C6).
  2. The roots pass: `RunRegistry` gone; `RunDrivers` with the 98 driver tests (97 + `TestDrivers`'
     1) and one new test of the sweep's runs-root seam (C12), 99; `ObjectFactories` 271 + 3 + the
     3 tests of `config/fingerprint.py` (C18), 277; the rest at 05's counts. *(Amended 2026-10-15,
     at E2: the seam's test and C18's three.)*
  3. `python -m runregistrykit list` reads every run as `python -m RunRegistry list` did just
     before the swap: 37 at E2, 01's 29 as 01's baseline has them, and R0's and R1's 8. `store show`
     on each of the 18 sidecars (01's 14 and R0's and R1's 4 tombstones) reads it as `RunRegistry`
     did. *(Amended 2026-10-15, at E2: 01's counts were taken before R0 and R1.)*
  4. **From a worktree** of this repository, `begin()` of a throwaway run records it under the main
     checkout's `var/runs/` (Q3), which the run directory then shows; the run is ended with
     `finish("done")` and left as a record, never deleted. The worktree is at a probe commit of
     06's staged tree, and is removed after (C15).
  5. A fingerprint taken through the new path carries DatastoreKit's version beside the digest, and
     its digest equals `fingerprint_of`'s on 06's own test store.
  6. Closes `[00-a-run-launched-from-a-worktree-records-into-the-worktree]` and
     `[00-the-fingerprint-encoding-has-no-version-of-its-own]` (§6.2 C8).
  7. Closes `[00-recorded-fingerprints-lack-the-encoding-format]`: before any call of the package's
     comparison, no sidecar under `var/datastores/` is a live store whose recorded fingerprint lacks
     `encoding_format` (R0's and R1's retired under D6, §3). The addendum's items 1, 2 and 5 are
     06's changes, named (§5 rule 7).
- **Out.** The guard (07). Any change in what is stored.

**07 — Guard the fingerprint format against spec changes** (Sonnet).
- **What it changes.** A test in `ObjectFactories/tests/` that pins, for format 6, the SHA-256 of
  the canonical form of every `inventory_spec()` in the registry, in registry order. A polymorphic
  `Parent.types` key is stringified explicitly, since `canonical_json` refuses a non-string key
  (audit §3.5). `config/fingerprint.py` holds the pin beside the format number.
- **Acceptance.**
  1. The test passes at 06's tree.
  2. **Deliberate breakages:** a leaf added to one spec, with no bump, fails it and names the class;
     the format bumped with no new pin fails it; a class added to the registry with a spec fails it.
- **Out.** Any spec change.

**08 — Rehearse through both packages, R2** (Opus).
- **What it does.** §4.1's protocol, slug prefix `runregistry-extraction-08`, at 07's tree, through
  RunRegistryKit and DatastoreKit's fingerprint code.
- **Acceptance.**
  1. L1–L3 hold, `39809dca…`, equal to R1's and R0's recorded fingerprints; each fingerprint
     carries DatastoreKit's version. The recorded ones are SGK's and lack `encoding_format`, so they
     are compared field by field with it set aside, not through the package's
     `compare_fingerprints`, which gives one `encoding` entry for them
     (`[00-recorded-fingerprints-lack-the-encoding-format]`).
  2. Every manifest's `packages` names `datastorekit`, `raypoolkit` and `runregistrykit`, each at
     its pin, none editable.
  3. L4 against R1.
  4. `python -m runregistrykit list` shows 01's 29 runs and this campaign's fourteen (the twelve
     rehearsal launches, 06's worktree check and the run of its breakage), and nothing `running`.
     *(Amended 2026-10-15, at E2: 06's breakage of the check leaves a run of its own.)*
  5. **The board records RunRegistryKit's G2 as holding** (§6.2 C11).

**09 — Close the campaign** (Sonnet).
- **What it does.**
  - Records the final measurements at `HEAD`: the roots, `pip freeze` against 05's, and every
    `RunRegistry` reference outside `prompts/` and dated `docs/`.
  - Drafts the gate records for RunRegistryKit's board (G2) and DatastoreKit's (the fingerprint
    release adopted), in each board's form, in verification §4 (§6.2 C11).
  - Drafts, for the user to hand to RunRegistryKit, the SGK issues whose defect lies in code that
    moved there (`[00-open-issues-about-the-registry-code-stay-on-sgk-boards]`, opened at E2), each
    located in `runregistrykit` at `v0.1.0`, and with them
    `[06-runregistrykit-list-into-a-closed-pipe-ends-in-a-traceback]`, a defect in the package itself
    seen in 06's review (added 2026-10-15, at the user's request).
  - Closes the board, README §2, `prompts/INDEX.md` and `docs/OPEN_ISSUES.md` §1.21.
  - States, for the A3 v2 campaign, what it inherits: the launcher, the commands, and that its
    store is the first live one in the new layout.

## 3. Datastores

**The campaign builds six throwaway stores and touches no other.** Each rehearsal builds two,
`var/datastores/runregistry-extraction-NN-{reference,resumed}.sqlite`, for NN = 02, 04 and 08, each
with a sidecar (`store create`) and a written fingerprint.

**No prompt retires a store.** They stay for each review, and R0's and R1's are compared with later
ones through their recorded fingerprints, which their run directories keep.
- **R0's and R1's four are retired after 04's review** (the user's ruling D6, 2026-10-14, §6.1). Their
  recorded fingerprints are SGK's, without `encoding_format`, and from 06 the package's comparison
  refuses them (`[00-recorded-fingerprints-lack-the-encoding-format]`). The orchestrator retires
  each with `store retire`, after a dry run, while SGK's own code still compares them, in a commit
  of its own, as `dk-and-rpk-migration`'s were at `718e650`.
- **R2's two** the user retires after 09's review, or not.

**Before touching anything,** a prompt runs the registry's `list` and reports anything `running` or
stale. Before each launch it checks that no Ray process of this repository is up, waiting by pid
for a killed build's agents (`[07-a-killed-builds-ray-agents-outlive-the-driver]`).

## 4. The interfaces between prompts

- **01 ships** the freeze section of `CLAUDE.md`; the frozen-tree table on the board; and
  `measure/out/01-*` baselines: `01-list.txt`, `01-store-show.txt` and `01-roots.txt`. 06 and 08
  compare with them.
- **02 ships** `docs/runregistry-extraction-verification.md` with §1, R0, and saved outputs under
  `docs/runregistry-extraction/rehearsal/` named with `02`. It runs `dk-and-rpk-migration`'s
  rehearsal tooling under `docs/dk-and-rpk-migration/rehearsal/` unchanged. 04 and 08 add §2 and §3,
  and 09 §4.
- **02a ships** `tools/rehearsal_lock.py`, which 04 and 08 run for §4.1's lock steps, and its
  proofs, `measure/out/02a-interop.txt` and `02a-real.txt`.
- **03 and 05 ship the pins**, each in `requirements.txt` and no other file.
- **04 ships** `docs/runregistry-extraction/fingerprint_both_ways.py <primary>...` (W1). It compares
  two live implementations on one inventory: SGK's `fingerprint_of` and the package's. 06 deletes
  the first, and R0's and R1's stores were retired after 04's review (§3). So **08 does not run the
  script as 04 left it; it repeats W1's claim across stores** (amended 2026-10-14, after 04's
  review):
  - **What 08 repeats.** The package's fingerprint of each of R2's two stores equals, in every field
    but `encoding_format`, the fingerprints SGK's code recorded for R0 and R1 in their `reference`
    and `resumed-c` run directories. This is 08's acceptance 1. W1 showed that SGK's fingerprint and
    the package's agree on the same store. So this equality carries W1's claim to the tree where only
    the package's implementation is left.
  - **06 rewrites the script for it**, under C6, since its `RunRegistry` imports must go (06's
    acceptance 1). The new form is `fingerprint_both_ways.py <primary>... --against <run dir>...`.
    - It takes each store's fingerprint through `config/fingerprint.py`'s fingerprinter.
    - It compares that fingerprint field by field with each run directory's recorded one,
      `encoding_format` set aside, and prints every differing path.
    - It keeps W1's checks that still have two sides:
      - the store's `encoding_format` is the package's `ENCODING_FORMAT`;
      - the recorded fingerprint lacks it;
      - the package's `compare_fingerprints(recorded, current)` is exactly one `encoding` entry;
      - the sidecar's fingerprint, now the package's, compares empty.
    - 06 names the script's breakages (§5 rule 8), as 04 did.
  - **`compare_with_recorded.py` is not 08's comparison with R1 and R0.** After 06 it compares
    through the package, which gives one `encoding` entry for every fingerprint SGK's code recorded
    (`[00-recorded-fingerprints-lack-the-encoding-format]`). At 08 it is run on R2's own stores and
    sidecars only.
- **06 ships** the config file, `config/fingerprint.py`, `RunDrivers/tests/`, the moved and new
  golden files under `ObjectFactories/tests/data/`, the rewritten tooling,
  `docs/runregistry-extraction/worktree_run_check.py`, and `measure/out/06-*`. From 06 on, §4.1's commands name `runregistrykit` where they named `RunRegistry`; 06
  states the replacements in its log, and 08 runs them.

### 4.1 The rehearsal protocol

**`dk-and-rpk-migration`'s protocol, unchanged but for names and the lock below** (its README §4.1,
as its prompts 02, 05 and 08 ran it, and `docs/dk-and-rpk-migration-verification.md` §1–§3 record
it). NN is the prompt's number:
- stores `var/datastores/runregistry-extraction-NN-{reference,resumed}.sqlite`;
- `--campaign runregistry-extraction --prompt NN --job-name runregistry-extraction-NN`;
- launch logs `var/runs/runregistry-extraction-NN-<slug>.launch.log`;
- reader outputs under `var/runregistry-extraction-NN/extract/`.

The four launches are `reference` straight through; `resumed-a`, killed by `SIGTERM` in
`CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT`; `resumed-b`, with `--resume`, killed in
`CALCULATE QUADRATIC SOURCE INTEGRALS`; and `resumed-c`, with `--resume`, to `done`. **After each
launch, one check, then end the turn** (`CLAUDE.md`, run registry rules 3–4); the orchestrator waits
once per launch on the run's liveness.

**What must hold** is L1–L4 as there: every launch ends as recorded (L1); the resume equals the
reference, `39809dca…`, and passes the read-only integrity checks (L2); every reader runs read-only
on the closed `resumed` store (L3); and the counts equal the previous rehearsal's (L4: this
campaign's previous, or for R0 `dk-and-rpk-migration`'s R2). A difference in L4 alone is a finding
to explain; the fingerprint is the acceptance.

**If a fingerprint misses**, the prompt stops, records both fingerprints and `compare_fingerprints`,
and changes no code. R0's miss is SGK's; R1's is DatastoreKit's; R2's is RunRegistryKit's or
DatastoreKit's (§5 rule 11).

**The rehearsal lock** (added 2026-10-12, after R0, at the user's request). Three projects on this
machine, SGK, ChamPBH and StochasticInstantons (SI), run Ray builds. The user ruled that only one
rehearsal or long build runs at a time, under one shared lock. The contract is SI's
`.prompts/dk-and-rpk-migration/rehearsal-lock-brief.md` §2 and §5, at SI's `d6d550b`, with SI's
handover note at `5fb4a1d`. SGK's builds already start a private in-process Ray instance, so SGK
takes the lock and nothing else.
- **The lock** is the directory `~/ScienceData/.rehearsal-lock`. Its file `owner` has seven
  `key: value` lines, in this order:
  - `project: SecondaryGWKit`;
  - `purpose: runregistry-extraction NN (R<n>)`;
  - `acquired:` ISO 8601, local time with its offset;
  - `host:` the output of `hostname -s`;
  - `ray_temp_dir: in-process` and `ray_address: in-process`;
  - `pid:` as below.

  `owner` is always written to `owner.tmp` first, then renamed.
- **The helper.** From 02a on, each step below is a command of `tools/rehearsal_lock.py`, run as
  `PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock <command>`. The helper writes the owner
  file and refuses any step that would break the contract. Its `ray-processes` command is §3's
  check for Ray processes of this repository. A machine-wide `pgrep 'gcs_server|raylet|ray::'`,
  as R0 ran, now also finds SI's and ChamPBH's clusters, and is not the test.
- **Take it before the `reference` launch**, with
  `acquire --purpose "runregistry-extraction NN (R<n>)"`. Exit 3 means the lock is held: the
  helper prints `owner`, and the prompt launches nothing and **stops and asks the user**. Never
  wait in a loop, retry, or remove the lock. Exit 4 means a Ray process of this repository is
  still alive; the helper has released the lock again, and the prompt stops in the same way.
- **Hold it through all four launches and L3's readers.** It is one lock for the whole rehearsal,
  not one per launch: other projects must not start between the launches either.
- **At each launch, set `pid`** with `set-pid <pid>`, using that run's driver pid from its
  `status.json`, in the same check that confirms the run started. The readers leave it alone. Between launches the pid is
  the previous driver's and is dead. The lock is still held then: only the user removes a lock,
  by hand, after looking.
- **Release it after the readers**, with `release`. The helper refuses, exit 1, while the
  recorded pid is alive or `ray-processes` lists anything: wait by pid for a killed build's agents,
  as §3 says, then release. It also refuses an owner that is not `project: SecondaryGWKit` with
  `ray_temp_dir: in-process`, and prints both: stop and ask the user.
- **A prompt that stops early** releases the lock the same way once no Ray process of this
  repository is alive. If it cannot, it says so in its report, so the user can remove the lock.

**Ray addresses.** Every Ray driver this protocol runs gets an explicit address and starts its own
instance:
- `quadsource_atol_sweep.py --build` replaces `main.py`'s `ray.init` with `address="local"`;
- `run_readers.sh` passes `--ray-address local`.

Nothing in a rehearsal runs `main.py` or an `extract_*.py` at its default, `--ray-address auto`. The
reason is `[00-main-and-the-readers-default-to-ray-address-auto]`: `auto` attaches to any Ray
instance it finds on the machine. Every `ray` CLI call names `--address`. **`ray stop` is never
run.** It stops every Ray process on the machine, including other projects' builds.

### 4.2 Rollback

Each prompt is one commit, reverted by `git revert`. Reverting 03 or 05 also needs the venv put
back (`pip install` of the previous pin, or `pip uninstall -y runregistrykit`). Reverting 06 alone
needs nothing in the venv. The campaign runs in the main checkout: a worktree here symlinks the main
checkout's venv, so a pin installed from one lands in the other. 06's worktree check (acceptance 4)
is the one use of a worktree, and installs nothing.

## 5. The rules this campaign runs under

The project-wide ones in `CLAUDE.md`, plus:

1. **One commit per prompt.** The commit boundary is the rollback boundary.
2. **Every prompt writes a log** to `logs/NN-<name>.md`, using §5.1, classifying every deviation as
   `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.
3. **Every prompt updates `IMPLEMENTATION_STATE.md` in its own commit,** and `docs/OPEN_ISSUES.md`
   whenever a §3 or §4 entry moves, in the same commit.
4. **Do not fix things the prompt did not ask for.** Record them, and open a §3 issue here.
5. **Commit messages** in `CLAUDE.md`'s form, ending with `Co-Authored-By:` naming the model.
6. **Verification documents are additive.**
7. **Existing tests are not modified**, except where a prompt names the test and the reason. 06's
   moves and import rewrites are named.
8. **Deliberate breakage.** Each prompt that adds code or a test names mutations its tests must
   catch. The log records each as a diff, exactly as applied. Mutations are never committed.
9. **Nothing the code stores changes.** No table, column, key or stored value; the fingerprint's
   overall digest stays `39809dca…`. Two records gain fields, by D3 and the pins: a fingerprint
   carries DatastoreKit's version beside its digest (06), and a manifest's `packages` names
   `runregistrykit` (06). A prompt that cannot keep the digest stops.
10. **No test needs Ray, and no test opens anything under `var/`.** A rehearsal's reads of its own
    stores, and 01's and 06's reads of the registry and the sidecars, are verification steps, not
    tests.
11. **A defect in a package is not patched or worked around here.** The prompt stops and records
    it on this board and in `docs/OPEN_ISSUES.md`. The user reports it to that package's repository,
    where it is fixed in a release, and a pin prompt moves SGK to it.
12. **No change is made in DatastoreKit, RunRegistryKit or RayPoolKit** by a prompt of this
    campaign. Their repositories are read, and their scripts run from their roots with their own
    venvs. The briefs in this folder are handed to them by the user (§7).
13. **Pins.** Each pin is a tag, in a commit of its own, never an editable install. SGK keeps its own
    `ray` and `SQLAlchemy` pins.
14. **`RunRegistry/` is frozen from 01 until 06** (§2.2, 01).
15. **Long-running jobs** follow `CLAUDE.md`'s run-registry rules: discovery first, detached launch,
    one check, end the turn.

### 5.1 The log template

- the subject, commit and result;
- **What shipped**;
- **Deviations from the prompt**, each classified;
- **Verification performed**, with every count against §0.2's or the previous prompt's;
- **The deliberate-breakage record**;
- **Observations not acted on**;
- **State handed to the next prompt**.

## 6. Decisions

### 6.1 Made by the user (2026-10-12), and not reopened here

From the audit, §0 and §8.

| | Decided | Rejected |
|---|---|---|
| **D1** — where it goes | **A repository of its own, RunRegistryKit**, beside DatastoreKit and RayPoolKit | DatastoreKit: the core is about jobs, imports only the standard library, and `datastore-portability` §6.3 kept the sidecar out of the datastore layer. A copy in each client. One combined toolkit |
| **D2** — when | **Before the A3 v2 compute run** | After it |
| **D3** — the fingerprint | **Split by layer**: encoding and digest to DatastoreKit with a version of its own; the content specification and the format number grouped client-side in one module; RunRegistryKit takes a client fingerprinter and treats the result as opaque | The algorithm in RunRegistryKit; a digest callable written by each client |
| **D4** — no fingerprinter | **Not supported.** The store layer requires one; the core/`[stores]` split is a dependency split, not a degraded mode | A store layer that works without one: "solving the wrong problem" |
| **D5** — spec coverage in the clients | **ChamPBH 09 and SI 10 prove it**, by a test (ChamPBH `64ef9aa`; SI's uncommitted plan) | The report-coverage criterion alone |
| **Q1** — names | `RunRegistryKit`, `runregistrykit`, `python -m runregistrykit` | `RunRegistry` |
| **Q2** — the client config | **A file of its own at the root**, read with `tomllib`, which also marks the root | A `[tool.runregistrykit]` table in a `pyproject.toml` |
| **Q3** — a worktree's run | **Recorded in the main checkout** (the parent of `git rev-parse --git-common-dir`); its SHA and script hash are still the worktree's | Recorded in the worktree, and lost with it |
| **Q4** — distributions | **One, with the store layer as the extra `[stores]`** | Two |
| **Q5** — the spec-digest guard | **Built in SGK's adoption** (07), and offered to the other clients through RunRegistryKit's checklist | The golden fingerprint alone |
| **Q6** — the 101 driver tests | **A test root of their own** (§6.2 C5) | `tools/tests/` |
| **Q7** — a manifest format field | **Not during the move**; proposed separately after it | Added during the move |
| **Q8** — two value tables per spec | **Left to SI's prompt 10** | Asking DatastoreKit now |

**Made by the user on 2026-10-14**, at E1, from DatastoreKit's `v0.4.0` addendum (item 3):

| | Decided | Rejected |
|---|---|---|
| **D6** — the fingerprints SGK holds, which lack `encoding_format` | **Option (c): retire R0's and R1's four stores after 04's review**, while SGK's own code still compares them (§3) | (a) recomputing each with `--write`, replacing the recorded fingerprint; (b) keeping SGK's comparison beside the package's, a version-aware reader |

### 6.2 Choices of method made in writing this plan

The orchestrator's, recorded so that the user can reverse any of them.
- **C1 — the branch.** `handover-remedial`, where A3 v2 runs. The main checkout (§4.2).
- **C2 — the rehearsals** are `dk-and-rpk-migration`'s full protocol, run by the agent, as that
  campaign's U2 ruled for it: three, so that each change is isolated (§0.1).
- **C3 — R1 takes the fingerprint both ways.** DatastoreKit's tests cannot name SGK, so its release
  cannot itself show that SGK's digest is unchanged. R1 shows it, on real stores, before 06 makes SGK
  depend on it. The brief asks DatastoreKit to keep the digest; R1 is where that is checked.
- **C4 — the freeze is 01, and the import commit is 01's** or any later commit at which
  `RunRegistry/` equals the frozen tree. `CLAUDE.md` says so until 06.
- **C5 — where the 101 go.** `TestGolden` (3) and the six golden files test SGK's fingerprint of
  SGK's fixture store, a function of the specs, so they go to `ObjectFactories/tests/` beside the
  inventory tests, with the golden files under `ObjectFactories/tests/data/`. The other 98 test the
  `docs/` drivers that launch registered runs, so they go to a new root, **`RunDrivers/tests/`**,
  which `CLAUDE.md`'s list of roots gains. The name is the plan's; 06 may propose another before it
  acts.
- **C6 — the rewrite rule.** Every `RunRegistry` import, string literal and command outside
  `prompts/` is rewritten, including `docs/` scripts (as `dk-and-rpk-migration` C2), but not dated
  `docs/` prose. The six `prompts/datastore-generic/orchestrator/measure-*` scripts are records.
- **C7 — stale prose.** A statement that a prompt's own commit makes false is corrected in that
  commit: `CLAUDE.md` (01, 06), the registry's mentions in `CLAUDE.md`'s test list and mechanics
  (06). Dated audits, verification records and `prompts/` are left as written.
- **C8 — two issues opened by this plan** (board §3), both assigned to 06:
  - `[00-a-run-launched-from-a-worktree-records-into-the-worktree]`, SGK's (audit §2, C1);
  - `[00-the-fingerprint-encoding-has-no-version-of-its-own]`, a DatastoreKit defect, reported by
    [`datastorekit-fingerprint-brief.md`](datastorekit-fingerprint-brief.md). It closes here when SGK
    records the version (06), as the package's half closes there with E1.
- **C9 — the briefs.** Two, in this folder, handed by the user to a session in each repository.
  RunRegistryKit's repository is created by the user, locally and on GitHub, before its session
  starts (§6.3).
- **C10 — the rehearsal tooling.** `docs/dk-and-rpk-migration/rehearsal/` is run as it stands at 01's
  tree, through 04. 06 rewrites its two `RunRegistry` imports (C6), and 08 runs it as 06 left it.
- **C11 — the gates.** This board records E1, E2, R0–R2 and RunRegistryKit's G2 as each holds.
  Recording a gate on RunRegistryKit's or DatastoreKit's board is a commit in that repository: 09
  drafts the text, and the user makes the commits there, or has a session there make them.

**Made at E2 (2026-10-15), in writing 05–08 from RunRegistryKit's SGK checklist**, where the
checklist leaves a choice to SGK:
- **C12 — the drivers' runs root.** Two staying tests redirected a driver's runs by patching
  `DEFAULT_ROOT`, which `v0.1.0` deletes. `scoped_pipeline_run.register` and
  `quadsource_atol_sweep.prepare` each gain a `runs_root=None` keyword, passed to `begin(root=)` and
  `copy_store(runs_root=)`, whose default changes nothing; the tests pass a temporary root. This is
  the checklist's advice. The alternatives patch `runregistrykit.begin`, which the package does not
  promise, or let a test read the real `var/runs/`. Neither seam was tested before: the registration
  test gains an assertion, and the sweep a test (06 §3.2).
- **C13 — the golden fingerprint.** SGK's golden files lack `encoding_format`. 06 captures a new
  golden file beside the six, `full_store_fingerprint_runregistry-extraction-06.json`, through the
  config's fingerprinter; a test pins it as `09c`'s with `encoding_format` added. This is the format
  history's own convention, a new golden beside the old and never over it. The format is not
  bumped, since the digest has not moved. Setting `encoding_format` aside in the golden comparison
  instead would be a version-aware comparison in a test.
- **C14 — the test helpers.** `v0.1.0` ships no test support (its U21). SGK keeps its own copy of
  what its staying tests use of `StoreTestCase` and `RegistryTestCase`, in
  `RunDrivers/tests/registry_fixtures.py`, taken from the frozen tree.
- **C15 — the probe commit.** 06's worktree check and the checklist's measure need 06's tree as a
  commit, before 06's one commit exists. 06 makes one with `git commit-tree` from its staged tree,
  which no branch names. The worktree runs there, clean, while the main checkout is dirty, which
  shows that the run's provenance is the worktree's. Its commit then differs from the probe by the
  records alone.
- **C16 — the rewrite boundary.** C6 made precise: every tracked file outside `prompts/` is
  rewritten, `.py` and `.sh` under `docs/` included, docstrings and comments too. `.md` and `.txt`
  under `docs/` are dated records and recorded output, and `docs/OPEN_ISSUES.md`'s rows are the
  boards'. `CLAUDE.md` may keep one sentence naming the registry's former home.
- **C17 — `config/fingerprint.py`.** The fingerprinter is named `fingerprinter`, an instance of
  `DatastoreKitFingerprinter`. The format history moves verbatim, followed by one dated paragraph:
  where the golden files now are, and that the encoding's version is DatastoreKit's and does not
  bump SGK's format.
- **C18 — `config/fingerprint.py` is tested.** It is new code (§5 rule 8): three tests beside
  `TestGolden`, on the config's naming of it, its take of the fixture store, and the new golden.

### 6.3 Left for the user

- **Before E2:** creating the RunRegistryKit repository, at `/Users/ds283/Documents/Code/RunRegistryKit`
  and `https://github.com/ds283/RunRegistryKit`, and handing each brief to a session in its
  repository.
- **At the close:** retiring R2's two rehearsal stores, or not (§3); the two gate records (C11).

## 7. The briefs, and what this campaign needs back

| Brief | For | What SGK needs back | Gate |
|---|---|---|---|
| [`datastorekit-fingerprint-brief.md`](datastorekit-fingerprint-brief.md) | a campaign in DatastoreKit | a release with the fingerprint algorithm and a version for the encoding and the algorithm; its SGK-facing note | **E1**, for 03 |
| [`runregistrykit-extraction-brief.md`](runregistrykit-extraction-brief.md) | the extraction campaign in a new RunRegistryKit repository | a release imported from 01's frozen tree, with the client config, the store layer as an extra over E1's release, and SGK's adoption checklist | **E2**, for 05 |

Each brief is evidence for the receiving campaign to check, not a specification of it. Where its
own records disagree, they are right, and the disagreement comes back here.

## 8. Baselines

Re-measure every root at `HEAD`, clean, before every dispatch:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s <package>/tests -t . > <file> 2>&1
grep -E '^(Ran|OK|FAILED)|^(FAIL|ERROR):' <file>
```

**Last counted at `c8a7f14`** (§0.2): AdaptiveLevin 32, ComputeTargets 693, CosmologyModels 39,
LiouvilleGreen 148, ObjectFactories 271, RunRegistry 286, tools 49.

**Under load, four tests are known to fail on wall clock or chance:**
`test_tk_wkb_phase.TestCost.test_wall_time_per_object`; `test_background_tau.test_build_cost`;
`test_numeric_phase_cut.test_the_print_is_gated_both_ways`; LiouvilleGreen's
`test_3bessel_analytic.test_JJJ`. Re-run a failure alone before calling it real.
