# Prompt 08 — rehearse through both packages (R2)

**Campaign:** [`README.md`](README.md) · **Board items:** **L1**–**L4** (R2), **G-RPK** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-11, with prompts 05–07 and 09, after the review of 04 (`0b6565e`).
**Gate:**
- 07 landed and reviewed: `RayTools/` gone, 54 sites constructing (board, 07's row);
- `git status` is clean, and no commit after 07's touches code (anything outside `prompts/`);
- `python -m RunRegistry list` shows nothing `running`;
- the venv holds `datastorekit` 0.2.2 at `6d8ee9ae…` and `raypoolkit` 0.2.0 at `eaef0ce…`, neither
  editable;
- R1's two stores and four run directories exist as 05 left them.

**Closes:** nothing. **Opens:** anything you find, **without fixing it**.
**Gate it decides:** **RayPoolKit's G2** (README §0, §6.2 C12).
**Recommended model:** **Opus**. The same four builds as R1, now with the work queue the package's
too, and one behavioural change to measure (C8).

**Read first:**

1. `CLAUDE.md`, its package section and "Long-running jobs — the run registry", in full.
2. [`README.md`](README.md): §0.1, §2.2 (08's charter), §3, **§4.1**, §5 and §6.2 (C8, C12).
3. **[`05-rehearse-through-datastorekit.md`](05-rehearse-through-datastorekit.md), in full.** This
   prompt runs its protocol, with the names and differences below. Where this prompt is silent, 05
   governs, and through it 02.
4. [`docs/dk-and-rpk-migration-verification.md`](../../docs/dk-and-rpk-migration-verification.md)
   §1 and **§2 (R1)**, which R2 is compared with.
5. [`logs/05-rehearse-through-datastorekit.md`](logs/05-rehearse-through-datastorekit.md) and
   [`logs/07-adopt-raypoolkit.md`](logs/07-adopt-raypoolkit.md), "State handed to the next prompt"
   in each.
6. RayPoolKit's SGK checklist, `/Users/ds283/Documents/Code/RayPoolKit/docs/adoption/secondarygwkit.md`,
   items 5 (`main.py:3247`) and 8 (G2). Read only.
7. The board's §3 entry `[02-the-policy-statistics-block-orders-its-lines-by-completion]`.

---

## 0. What this prompt is

R1 rebuilt the reference store through `datastorekit` with SGK's own `RayTools/` and reproduced
`39809dca…`. 07 then replaced `RayTools/` with `raypoolkit` and changed nothing else that runs.
**R2 is R1 again, through both packages.** If R2 differs from R1, the work queue is the only cause.

R2 also measures the one behavioural change RayPoolKit's checklist names (C8): at `main.py:3247`
the `APPLY GKSOURCE POLICIES` queue's `post_handler` now receives the persisted object, not the
local copy. By reading, the statistics it counts are the same. R2's statistics block, against
R1's, is the measurement.

**What this prompt is not.** It changes no code and no test, and until the last launch has ended
it creates no file in the repository outside `var/` (`git_dirty: false`). Working notes go under
`var/dk-and-rpk-migration-08/`.

---

## 1. The protocol, with this prompt's names

**Run 05's §1 as written**, with `05` replaced by `08` in every name: stores
`var/datastores/dk-and-rpk-migration-08-{reference,resumed}.sqlite`, `--prompt 08`,
`--job-name dk-and-rpk-migration-08`, launch logs `var/runs/dk-and-rpk-migration-08-<slug>.launch.log`,
reader outputs under `var/dk-and-rpk-migration-08/extract/`, saved outputs named with `08`. Each
`--purpose` names R2, both packages and their commits, and the fingerprint it must equal.

Differences from 05:
- **The pre-launch checks** find `datastorekit` 0.2.2 and `raypoolkit` 0.2.0 installed.
- **The interruption points are R2's own**, chosen from R2's `reference` `stdout.log` by the same
  reasoning. `interrupt_when.py` reads the run's `status.json`, whose `stage` the registry takes
  from the pool's printed lines: if the interrupter fails to see a stage it saw at R1, that is a
  change in the printed lines and a stop (§4).
- **After a killed launch**, wait by pid for its Ray agents, as at R1, and compare how long they
  took with R1's.

---

## 2. What to verify

**L1–L3 are 05's §2**, on this prompt's stores and runs, with these differences.
- **The manifests.** Every `packages` is exactly
  `{"datastorekit": {"version": "0.2.2", "url": "https://github.com/ds283/DatastoreKit", "commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "requested_revision": "v0.2.2", "editable": false}, "raypoolkit": {"version": "0.2.0", "url": "https://github.com/ds283/RayPoolKit", "commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "requested_revision": "v0.2.0", "editable": false}}`,
  and every `git_dirty` is `false`.
- **The fingerprints.** Both stores `39809dca…`, format 6, 21 classes, each equal to its sidecar's;
  `compare_fingerprints` between them empty; `compare_with_recorded.py` against **R1's `reference`
  run directory**, R0's, 07's and `datastore-generic` 09's, each empty. R1's and R0's stores are
  not opened; record their files' SHA-256 and mtimes before and after.
- **What runs is the package.** `import RayTools` raises `ModuleNotFoundError` at the dispatch
  `HEAD`, and `raypoolkit.__file__` lies in the venv's `site-packages`.
- **What is new in the logs** is compared with **R1's** same slug, as 05 compared with R0's.

**L4 — equal to R1.** Against verification §2:
- **The statistics block, which measures `main.py:3247` (C8).** `policy_statistics.py` on R2's
  `reference`, against R1's: line for line and, where only the order differs, as sorted lines,
  saying so (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`). README §2.2 says
  "R1's, line for line"; that issue shows the order follows Ray's completion order on any one tree,
  so **the comparison is of the counts and of the sorted lines**. Log this as a `STRUCTURALLY
  REQUIRED` deviation. **Every count must be R1's** (291; numeric 109; mixed 26, 25 complete and
  1 minimal; WKB 156). Do the same for every resumed run that prints the block (at R0 both
  `resumed-b` and `resumed-c` did), against its R1 counterpart.
  - **A count that differs is the measurement C8 exists for**, not a stop: the fingerprint is the
    acceptance. Record it, explain it from `main.py:2871-2893` and the package's
    `RayWorkPool.py:529-530`, and open a §3 issue.
- **The reference run's stage counts** equal R1's, stage by stage.
- **The resumed chain's identities** hold, as 05 §2 states them.
- **The stage sequence**, the `** ` banner lines of the `reference` `stdout.log` in order, equals
  R1's.
- **The printed lines are an interface** (RayPoolKit checklist item 5; README §2.2, 08's acceptance
  4). For each of R2's four runs against R1's same slug:
  - the progress and completion lines of every stage have R1's form, times and counts aside: the
    same prefix, `n/m work items remaining = p% complete`, the `inflight: … | completed: …` line,
    and `ALL WORK ITEMS COMPLETE in time …`;
  - the registry read them: each run's `status.json` `stage` at its end, and each killed run's
    `stage` when it was killed, is the stage R1's same slug had.

  A line form `StageTracker` no longer parses is a package defect (§4).
- **The stage times**, as verification §1.5.5, beside R1's. Not compared.

**G-RPK — RayPoolKit's G2** (checklist item 8; board §2). It holds when:
1. `RayTools/` is gone (`git ls-tree` at 07's commit);
2. RayPoolKit's script finds 0 imports and 0 `import_module` calls naming it, and every one of the
   54 sites `constructs` (07's measurement, re-run here at the dispatch `HEAD` and saved as
   `measure/out/08-raypoolkit-measure_client_pools.md`);
3. the pin is in place and the venv installed from it (06; this prompt's manifests);
4. the six roots pass (§3 below);
5. R2 reproduces `39809dca…` (§2 above).

Say which commit and which run each part rests on.

---

## 3. Acceptance

1. **L1–L3** hold as §2 says. Both stores equal each other, `39809dca…`, and R1's, R0's, 07's and
   `datastore-generic` 09's recorded fingerprints.
2. Every manifest's `packages` is §2's record, and every `git_dirty` is `false`.
3. **L4** against R1: the statistics counts equal (or a difference recorded and opened as an
   issue), the reference stage counts equal, the chain's identities hold, the stage sequence
   equal, and the printed lines and registry fields as R1's.
4. Every difference in the logs from R1's is listed and explained.
5. **The six roots** pass at the dispatch `HEAD`, with 07's counts: Datastore 320 (54 + 103 + 101 +
   62), RunRegistry 286, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels 39, LiouvilleGreen
   148 (1 skipped).
6. **The board records RayPoolKit's G2 as holding**, naming 06's, 07's and this prompt's commits
   and R2's two runs that finished `done`.
7. **No deliberate breakage**: 08 adds no code and no test.

---

## 4. Stop conditions — stop and ask

**05 §4's conditions apply**, with R1 in place of R0, and:
- **A fingerprint differs from `39809dca…` or from R1's.** Only the work queue changed, so it is a
  `raypoolkit` defect (README §5 rule 11). Record, change nothing, relaunch nothing, stop.
- **A printed line `StageTracker` or `interrupt_when.py` cannot read**, a `status.json` stage that
  R1 had and R2 does not, or a registry field left empty that R1 filled.
- **A pool raises** anything R1's did not: a `ValueError` from a constructor, a traceback in a
  `stderr.log`.
- A manifest's `packages` is not §2's record, or a `git_dirty` is `true`.
- R1's or R0's store files change.

---

## 5. What this prompt does not do

- It changes no tracked file but the records of §6.
- It writes no store under `var/` but its two, retires none, and reads R0's and R1's only through
  their recorded fingerprints.
- It does not fix what it finds.

---

## 6. The log, the verification record, and the board

**The files you may change, and nothing else:**
- `docs/dk-and-rpk-migration-verification.md`: a new **§3, "R2 — through both packages"**, with
  §2's subsections in the same order, and one more, **"`main.py:3247`"**, holding C8's measurement.
  §1 and §2 are not edited.
- `docs/dk-and-rpk-migration/rehearsal/`: saved outputs named with `08`.
- `measure/out/08-raypoolkit-measure_client_pools.md`.
- the log, this board and `docs/OPEN_ISSUES.md`.

**The log**, at `logs/08-rehearse-through-both-packages.md`, in README §5.1's form, with what 05's
log records, against R1 in place of R0, plus C8's measurement and G-RPK part by part.

**The board:** the §1 row for 08, items L1–L4 (R2) and G-RPK, the §2 row "RayPoolKit G2", and
`### 5.8 After prompt 08 (<date>)`; a §3 entry for anything opened, with `docs/OPEN_ISSUES.md`
(§1.20) in the same commit, the count and date corrected by counting.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
