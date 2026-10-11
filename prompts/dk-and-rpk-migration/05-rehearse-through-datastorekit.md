# Prompt 05 — rehearse through `datastorekit` (R1)

**Campaign:** [`README.md`](README.md) · **Board items:** **L1**–**L4** (R1), **G-DK** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-11, with prompts 06–09, after the review of 04 (`0b6565e`).
**Gate:**
- 04 landed (`0c9e2cf`) and reviewed, with R0's reference store fingerprinting `39809dca…` through
  the package (board §5.4);
- `git status` is clean, and no commit after `0c9e2cf` touches code (anything outside `prompts/`);
- `python -m RunRegistry list` shows nothing `running`;
- the venv holds `datastorekit` 0.2.2 at `6d8ee9ae…`, not editable, and no `raypoolkit`;
- R0's two stores and four run directories exist as 02 left them.

**Closes:** nothing. **Opens:** anything you find, **without fixing it**.
**Gate it decides:** **DatastoreKit's G2** (README §0, §6.2 C12).
**Recommended model:** **Opus**. Four registered builds and the readers under Ray, through code
this repository no longer owns. The work is in reading what they did against R0.

**Read first:**

1. `CLAUDE.md`, "The datastore layer is `datastorekit`" and "Long-running jobs — the run registry",
   in full. The second's rules bind every launch here.
2. [`README.md`](README.md): §0.1, §2.2 (05's charter), §3, **§4.1 (the protocol)**, §5 (rules 9–11)
   and §6.2 (C12, C14).
3. **[`02-rehearse-the-start-tree.md`](02-rehearse-the-start-tree.md), in full.** This prompt runs
   its protocol. Where this prompt is silent, 02 governs, and through it `a3-v2-readiness` 07.
4. **[`docs/dk-and-rpk-migration-verification.md`](../../docs/dk-and-rpk-migration-verification.md)
   §1**, R0: how each launch went, the commands verbatim, the interrupter's firings and the reasons
   for its points (§1.1.2), and L4's tables (§1.5). Your §2 has the same shape.
5. [`logs/02-rehearse-the-start-tree.md`](logs/02-rehearse-the-start-tree.md) and
   [`logs/04-adopt-datastorekit.md`](logs/04-adopt-datastorekit.md), "State handed to the next
   prompt" in each.
6. `docs/dk-and-rpk-migration/rehearsal/`: the six scripts and R0's saved outputs (`*_02*.txt`),
   which are what yours should look like.
7. The board's §3 entries `[02-the-policy-statistics-block-orders-its-lines-by-completion]` and
   `[02-07s-store-soundness-calls-read-inventory-without-its-registry]`.

---

## 0. What this prompt is

R0 rebuilt the reference store at the start tree, with SGK's own copy of the layer, and reproduced
`39809dca…`. 04 then replaced that copy with the installed package and changed nothing else that
runs: no factory, no `config/`, no call. **R1 is R0 again, through the package.** So if R1 differs
from R0, the package is the only cause.

R1 is the first build, the first killed build and the first resume run through `datastorekit`.
Its code differs from R0's layer by DatastoreKit's `v0.2.1` fixes and `v0.2.2`'s release of actor
names on close (04's log, "State handed to the next prompt"). None should change what a build
writes. The fingerprint is the measurement.

**What this prompt is not.** It changes no code and no test. **Until the last build launch has
ended, it modifies and creates no file in the repository** outside `var/`, so every manifest stamps
`git_dirty: false`. Working notes go under `var/dk-and-rpk-migration-05/` until then.

---

## 1. The protocol, with this prompt's names

**Run 02's §1 as written, with these substitutions and differences.**

### 1.1 Names

- **Stores:** `var/datastores/dk-and-rpk-migration-05-reference.sqlite` and
  `var/datastores/dk-and-rpk-migration-05-resumed.sqlite`. Before the first launch, `store show`
  on each gives kind `absent` and no runs; if not, stop.
- **Runs:** `--campaign dk-and-rpk-migration --prompt 05`, slugs `reference`, `resumed-a`,
  `resumed-b`, `resumed-c`, and `--job-name dk-and-rpk-migration-05` on every launch. Each
  `--purpose` names R1, this campaign, the package and its commit, and the fingerprint it must
  equal.
- **Launch logs:** `var/runs/dk-and-rpk-migration-05-<slug>.launch.log`.
- **Reader outputs:** `var/dk-and-rpk-migration-05/extract/`.
- **Saved outputs** you commit carry `05` in their names, beside R0's `02` ones.

### 1.2 What holds from 02

- **The launch command** is README §4.1's, verbatim but for the store, the slug, the purpose and
  `--resume` (verification §1.1).
- **The pre-launch checks** of 02 §1.3, with one change: `./venv/bin/pip show datastorekit`
  finds 0.2.2 and `pip show raypoolkit` finds nothing.
- **One check after each launch, then end your turn.** The orchestrator waits on the run's
  liveness and resumes you. Never poll; never wait with `pgrep -f`.
- **After a killed launch**, wait by pid for its Ray agents before the next launch, and record how
  long they took. Under `v0.2.2` a pool kills its named actors on close, so a killed driver may
  leave fewer behind than R0's did. Record what you see.
- **The interrupter's two dummy tests** run before the first launch, into new directories outside
  `var/` and the repository, exactly as verification §1.1.2 gives them.
- **The interruption points are R1's own.** Choose them from R1's `reference` `stdout.log`, by
  verification §1.1.2's reasoning: `resumed-a` some 10–15 s into `CALCULATE GREENS FUNCTIONS FOR
  SOURCE REDSHIFT`, and `resumed-b` about a third of the way into `CALCULATE QUADRATIC SOURCE
  INTEGRALS`, with the QSI pattern not anchored at its end. Record each point and its reason.
- **Sidecars** as R0's: `store create` on each closed primary, and `store fingerprint --write` on
  the reference after `reference`.

### 1.3 The scripts

S1's six scripts in `docs/dk-and-rpk-migration/rehearsal/` run **as 04 left them**. The only change
since R0 is `store_soundness.py:30`, which imports `datastorekit.store_inventory` (README §4, "04
ships"). 07's originals under `docs/a3-v2-readiness/rehearsal/` are records; `store_soundness.py`
there still fails as the board's issue says, and is not run.
- `run_readers.sh <primary> <output dir>`, once with `--models LambdaCDM` (its default) and once
  with `nomodels extract_Gk_data`, as R0 ran it;
- `stage_counts.py` over the four run directories;
- `policy_statistics.py` on each run directory that has the block;
- `prune_check.py <closed_store_state output> <primary>`, its first argument
  `closed_store_state.py`'s output saved after each killed launch and before the next, run once the
  next launch has opened the store;
- `compare_with_recorded.py <primary>... --against <run dir>...`;
- `store_soundness.py <primary>...`.

`closed_store_state.py`, `prune_serials.py`, `replicated_serial_census.py` and `interrupt_when.py`
run from where R0 ran them. **If any script fails in a way that its R0 run did not, stop**: that is
either a package defect or a script 04 should have rewritten, and both are the user's to decide.

---

## 2. What to verify

**L1–L3 are 02's §2**, on this prompt's stores and runs, with these differences.

- **The manifests.** Every manifest has:
  - `git_dirty: false`, and `git_head` the dispatch `HEAD`;
  - `packages` exactly
    `{"datastorekit": {"version": "0.2.2", "url": "https://github.com/ds283/DatastoreKit", "commit": "6d8ee9ae2c056b8b1d0cf9cd460d867c47057ec6", "requested_revision": "v0.2.2", "editable": false}, "raypoolkit": null}`.
- **The fingerprints.** Both stores are format 6,
  `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`, all 21 classes, each equal
  to its sidecar's, and `compare_fingerprints` between them is empty. Then
  `compare_with_recorded.py` against **R0's `reference` run directory**
  (`var/runs/dk-and-rpk-migration-02-reference-20261011T143003`), and against 07's and
  `datastore-generic` 09's, as 02 ran it: empty for each.
- **R0's stores are not opened.** R1 is compared with R0 through R0's recorded fingerprints, not by
  reading R0's files. Record their SHA-256 and mtimes before your first launch and after your last
  step: they must not change.
- **What runs is the package.** State, with the evidence, that no layer module of SGK's can be
  imported at the dispatch `HEAD` (each of the 15 gives `ModuleNotFoundError` in a fresh process,
  as 04's review showed), and that `datastorekit.__file__` lies in the venv's `site-packages`.
- **What is new in the logs.** For each of the four runs, compare its `stderr.log` and launch log
  with R0's same slug. List every line kind that R1 has and R0 has not, or the reverse, with times,
  pids and paths stripped, and explain each. A warning or traceback that R0 did not print is a
  finding; one that indicates a failure is a stop.
- **The readers** run through `run_readers.sh`, with every store file byte-identical around each,
  as at R0. `extract_tensor_source_data.py` exits 0.
- **The guard** runs once, after `resumed-c`, on the resumed store, as 02 ran it.

**L4 — equal to R0.** Against verification §1.5:
- **The statistics block.** `policy_statistics.py` on R1's `reference`: compare with R0's
  `policy_statistics_02_reference.txt` line for line and, where only the order differs, as sorted
  lines, saying so (`[02-the-policy-statistics-block-orders-its-lines-by-completion]`). **The
  counts must be equal**: 291 instances; numeric 109, all complete; mixed 26, 25 complete and 1
  minimal; WKB 156, all complete. A difference in any count is a finding to explain, not a stop:
  the fingerprint is the acceptance (README §4.1).
- **The reference run's stage counts** equal R0's `reference` column of `stage_counts_02.txt`,
  stage by stage.
- **The resumed chain, by identity, not by count.** README §2.2 asks that "what the resume
  recomputed" be R0's. The raw counts cannot be, since they depend on where each launch was
  killed, and R1 chooses its own points. So L4 compares what holds whatever the point:
  - every stage finished before a launch was killed computes 0 on the next launch;
  - the `GkSource` rows `resumed-b` pruned at open are exactly those `resumed-a` left unvalidated
    (`prune_check.py`), and `resumed-b` computed 291 − (stored by `resumed-a`) + (pruned);
  - `resumed-c` computed 864 − (stored by `resumed-b`) `QuadSourceIntegral` rows, and nothing
    else;
  - the totals per class over the chain equal the reference's.

  Log this reading as a `STRUCTURALLY REQUIRED` deviation from README §2.2's wording.
- **The stage sequence.** The `** ` banner lines of R1's `reference` `stdout.log`, in order, equal
  R0's. Times differ and are not compared.
- **The stage times**, as verification §1.5.5, recorded for 08 to choose its points from. Not
  compared.

**G-DK — DatastoreKit's G2** (README §0; board §2). It holds when:
1. the 17 files are gone, two emptied (04, reviewed);
2. the pin is in place and the venv installed from it (03, reviewed; this prompt's manifests);
3. the six roots pass (§3 below);
4. R1 reproduces `39809dca…` (§2 above).

Say in the log and the verification record which commit and which run each part rests on.

---

## 3. Acceptance

1. **L1–L3** hold as §2 says. Both stores' fingerprints equal each other, `39809dca…`, and R0's,
   07's and `datastore-generic` 09's recorded ones.
2. Every manifest's `packages` is §2's record, and every `git_dirty` is `false`.
3. **L4** against R0, recorded in tables beside §1.5's: the statistics counts equal, the reference
   stage counts equal, the resumed chain's identities hold, and the stage sequence equal.
4. Every difference in the logs from R0's is listed and explained.
5. **The six roots** pass at the dispatch `HEAD`, serially, after the last launch: Datastore 320
   (54 + 103 + 101 + 62), RunRegistry 286, ComputeTargets 693, AdaptiveLevin 32, CosmologyModels
   39, LiouvilleGreen 148 (1 skipped). `black --check` is clean on any Python file added.
6. **The board records DatastoreKit's G2 as holding** (§2, row "DatastoreKit G2"), naming 04's and
   this prompt's commits and R1's two runs that finished `done`.
7. **No deliberate breakage** (README §5 rule 8): 05 adds no code and no test. The fingerprint
   comparison is the check, and R0 showed it can fail (02's review, check 6, compared against
   recorded fingerprints from other trees).

---

## 4. Stop conditions — stop and ask

**02 §4's conditions apply**, with these changes:
- **A fingerprint differs from `39809dca…` or from R0's.** This is the stop R1 exists for. Since
  only the package changed, it is a package defect (README §5 rule 11). Record both fingerprints
  and `compare_fingerprints`' output, classify each difference by class, change no code, relaunch
  nothing, and stop. The user reports it to DatastoreKit; the campaign waits for a release and a
  new pin prompt (C14).
- **The package raises anything the layer did not at R0**: a refused open, a traceback in a
  `stderr.log`, a reader that refuses or fails, a `ReadOnlyMiss` or `ReadOnlyWrite`.
- **A manifest's `packages` is not §2's record**, or a `git_dirty` is `true`.
- **An S1 script fails where its R0 run did not** (§1.3).
- R0's store files change.

---

## 5. What this prompt does not do

- It changes no tracked file but the records of §6.
- It writes no store under `var/` but its two, retires neither, and reads R0's only through their
  recorded fingerprints.
- It does not fix what it finds. Every finding goes in the log's *Observations not acted on*, and
  opens a §3 issue if it is a defect.
- It does not touch `RayTools/`, the pin, or anything in DatastoreKit.

---

## 6. The log, the verification record, and the board

**The files you may change, and nothing else:**
- `docs/dk-and-rpk-migration-verification.md`: a new **§2, "R1 — through `datastorekit`"**, with
  §1's subsections in the same order, every command verbatim. §1 is not edited (README §5 rule 6).
  The header's first paragraph already names §2.
- `docs/dk-and-rpk-migration/rehearsal/`: saved outputs small enough to commit, named with `05`.
  No script there changes; a new script of yours, if one is needed, goes beside them and is named
  in the log.
- the log, this board and `docs/OPEN_ISSUES.md`.

**The log**, at `logs/05-rehearse-through-datastorekit.md`, in README §5.1's form. It also records:
- every run's id, slug, pid, state, exit code, driver exit status, wall time and load;
- the interrupter's dummy tests, both points with their reasons, and both firings;
- the time the killed launches' Ray agents took to exit;
- L4's tables beside R0's;
- the log differences of §2 and their explanations;
- G-DK, part by part;
- the six suites beside the dispatch baselines.

**The board:** the §1 row for 05, items L1–L4 (R1) and G-DK, the §2 row "DatastoreKit G2", and
`### 5.5 After prompt 05 (<date>)`; a §3 entry for anything opened, with `docs/OPEN_ISSUES.md`
(§1.20) in the same commit, the count and date corrected by counting.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
