# Prompt 02 — rehearse the start tree (R0)

**Campaign:** [`README.md`](README.md) · **Board items:** **S1**, **L1**–**L4** (R0) ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-11, with prompts 01, 03 and 04, after the user reviewed the plan (`4532188`).
**Gate:**
- 01 landed and reviewed (board §1);
- `git status` is clean;
- `python -m RunRegistry list` shows nothing `running`;
- the venv holds neither `datastorekit` nor `raypoolkit` (`./venv/bin/pip show` finds neither).

**Closes:** nothing. **Opens:** anything you find, **without fixing it**.
**Recommended model:** **Opus**. It runs four registered builds and the readers under Ray on two
real stores, and the work is in reading what they did.

**Read first:**

1. `CLAUDE.md`, the section "Long-running jobs — the run registry", in full. Its rules bind every
   launch in this prompt.
2. [`README.md`](README.md): §0.1 (why R0 exists), §2.2 (02's charter), §3, **§4.1 (the
   protocol)**, §5 and §6.
3. **[`prompts/a3-v2-readiness/07-rehearse-the-a3-v2-launch.md`](../a3-v2-readiness/07-rehearse-the-a3-v2-launch.md),
   in full.** This prompt runs its protocol, §1.3–§3, with the names and differences of §1 below.
   Where this prompt is silent, 07 governs.
4. [`docs/a3-v2-readiness-verification.md`](../../docs/a3-v2-readiness-verification.md): how 07's
   runs went, its commands verbatim (§3), and the two issues it opened that bear on a launch:
   - `[07-a-killed-builds-ray-agents-outlive-the-driver]`: wait for the agents by pid;
   - `[07-stage-counts-cannot-label-a-run-of-any-prompt-but-10]`: why 07 wrote
     `stage_counts_07.py`.
5. `docs/a3-v2-readiness/rehearsal/`: `run_readers.sh`, `stage_counts_07.py`,
   `closed_store_state.py`, `prune_serials.py`, `store_soundness.py`, and the saved outputs, which
   are what yours should look like.
6. 01's log, for the manifest's `packages` field.

---

## 0. What this prompt is

The last rebuild that reproduced the reference fingerprint was `datastore-generic`'s 09 rehearsal,
at `53d4e18`. Three commits have changed the layer since:
- `fee1fbf`, the drop refusal;
- `4c34f13`, `resolve()`'s `None`;
- `086c81a`, the closed-store prefix.

None of them was rehearsed, and DatastoreKit's import commit `6f7f291` comes after all three. So
the package is a copy of a tree that no rehearsal has run. **R0 runs it.** R1 (05) is then
compared with R0. If R1 differs, the package is the only cause, since nothing else changed.

At R0 the layer is `b510bc9`'s, byte for byte, and the pool is the frozen `RayTools/`. The tree
differs from `b510bc9` only by the plan, 01's manifest field and the orchestrator's records.

**What this prompt is not.** It changes no production code and no test. **Until the last build
launch has ended, it modifies and creates no file in the repository** outside `var/`, so every
manifest stamps `git_dirty: false`. Keep working notes under `var/dk-and-rpk-migration-02/`, which
is gitignored, until then. The rehearsal scripts (S1) are written after the builds, before the
readers.

---

## 1. The protocol, with this prompt's names

**Run 07's §1.3–§3 as written, with these substitutions and differences.**

### 1.1 Names

- **Stores:** `var/datastores/dk-and-rpk-migration-02-reference.sqlite` and
  `var/datastores/dk-and-rpk-migration-02-resumed.sqlite`. Before the first launch, show that
  nothing exists by either name (`store show`: kind `absent`, no runs), as 07 §1.2. If anything
  does, stop.
- **Runs:** `--campaign dk-and-rpk-migration --prompt 02`, with slugs `reference`, `resumed-a`,
  `resumed-b` and `resumed-c`, and `--job-name dk-and-rpk-migration-02` on every launch.
  - Each `--purpose` is one line a stranger can read. It names R0, this campaign, and the
    fingerprint it must equal.
- **Launch logs:** `var/runs/dk-and-rpk-migration-02-<slug>.launch.log`.
- **Reader outputs:** `var/dk-and-rpk-migration-02/extract/`.

### 1.2 What 07 decided that holds here

- **D7-1:** two interruptions, as README §4.1 gives them: `resumed-a` in
  `CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT`, and `resumed-b` in
  `CALCULATE QUADRATIC SOURCE INTEGRALS`. Choose each point from the reference's `stdout.log`, as
  07 §1.4 says. 07's own choices are in its verification §3.
- **D7-2:** both stores stay, with their sidecars. No store is retired (README §3).
- The pair is `1e-32 / 1e-7`, with `--rehearsal`, `--cpus 6` and `main.py`'s default pruning.

### 1.3 The launches

- **The pre-launch checks of 07 §1.3** apply unchanged:
  - nothing `running`;
  - no Ray process from this repository's `venv`. The user's ChamPBH cluster may be up, and is
    left alone;
  - `RAY_ADDRESS` unset;
  - the load average recorded;
  - and **`git status --short` empty**, which is stricter than 07 (§0).
- **After each launch:** one check, then end your turn, exactly as 07 §1.3 says. The orchestrator
  waits on the run's liveness and resumes you. Do not poll, and never use `pgrep -f` to wait.
- **After a killed launch**, before the next, wait by pid for its orphaned Ray agents
  (`[07-a-killed-builds-ray-agents-outlive-the-driver]`). Record how long they took.
- **The interrupter's two dummy tests** (07 §3) run before the first launch, each into a new
  directory outside `var/` and outside the repository.

### 1.4 S1 — the rehearsal scripts

07's `run_readers.sh` hardcodes its store and output directory. Its `stage_counts_07.py` splits
run ids on `-07-`. After `resumed-c` has ended, and before the readers, write parameterised copies
under **`docs/dk-and-rpk-migration/rehearsal/`**:
- **`run_readers.sh`**, taking the store's primary and the output directory as arguments. Keep
  07's behaviour: the six readers with `--ray-address local --models LambdaCDM`, and each file's
  SHA-256, mtime and size, plus any journal, before and after each reader.
- **`stage_counts.py`**, labelling each column by the run id's slug. Take it from the manifest's
  own `campaign`, `prompt` and slug, not from a split on a prompt number. Read through
  `docs/datastore-integrity-verification/stage_counts.read` as 07's does.
- **`policy_statistics.py`**, which prints the `** GkSourcePolicyData SUMMARY STATISTICS` block
  from a run's `stdout.log` verbatim, and nothing else. It exits non-zero when the block is absent
  or appears more than once.

**05 and 08 run these unchanged** (README §4), so write them to be reused:
- usage in each docstring;
- no prompt number inside;
- every path an argument or derived from one.

07's originals are records, and are left as they are.

---

## 2. What to verify

**L1–L3 are 07's §2.1–§2.3**, on this prompt's stores and runs, with these differences:
- **The manifests (07 §2.1).** Also check:
  - every `packages` is `{"datastorekit": null, "raypoolkit": null}`;
  - every `git_dirty` is `false`;
  - every `git_head` is the dispatch `HEAD`.
- **The comparison (07 §2.2).** `compare_fingerprints` of the two stores is empty, and:
  - each is format 6, digest `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`,
    all 21 classes equal, each equal to its sidecar's;
  - **each also equals 07's reference.** `compare_fingerprints` against the fingerprint recorded
    in `var/runs/a3-v2-readiness-07-reference-20261004T012909/status.json` is empty. 07's stores
    are retired; its run directory is not, and is read only.

  Replace 07's comparison "against `datastore-integrity` 10" by this one.
- **The readers (07 §2.3)** run through S1's `run_readers.sh`. 07's run "without `--models`" is
  repeated, once. `extract_tensor_source_data.py` is expected to exit 0 (07a fixed it).
- **The guard (07 §2.1)** is run once, after `resumed-c`, on the resumed store, exactly as 07
  ran it.
- **07 §2.4, A3 v2's checklist, is not repeated.** Do not write one.

**L4 — the campaign's baseline.** R0 has no earlier rehearsal to equal. It records, for 05 and 08
to compare against:
- the reference run's `GkSourcePolicyData SUMMARY STATISTICS` block, through
  `policy_statistics.py`, verbatim;
- **against `datastore-generic`'s 09 rehearsal.** That block from
  `var/runs/datastore-generic-09-rehearsal-20261006T152452/stdout.log`, through the same script,
  equals R0's reference block line for line. If it does not, say what differs. It is a finding,
  not a stop: the fingerprint is the acceptance;
- the stage-count table for all four runs (`stage_counts.py`);
- what each resumed launch pruned and recomputed, by class and count.

---

## 3. Acceptance

1. **L1–L3** hold as §2 says. The two stores' fingerprints equal each other, `39809dca…`, and
   07's reference.
2. Every manifest has `packages` both `null` and `git_dirty: false`.
3. **L4** is recorded in the verification document, in tables that 05 and 08 can be compared with
   line for line.
4. **S1**'s three scripts exist, with usage in their docstrings. The readers, the stage counts
   and the statistics were produced by them.
5. Every existing test passes unmodified. The six suites run serially on the staged tree, at the
   dispatch baselines, and `black --check` is clean on every Python file added.
6. The verification record holds every measurement, transcribed. The stores are throwaway, so the
   numbers live in the document.

---

## 4. Stop conditions — stop and ask

**07 §5's conditions apply**, with "the two fingerprints differ" widened as follows. In addition:
- **A fingerprint differs from `39809dca…`, or from 07's reference.** This is the stop R0 exists
  for. The regression is then in SGK's own tree, between `53d4e18` and the dispatch `HEAD`, and is
  not this campaign's to fix (README §2.2).
  - Record both fingerprints and `compare_fingerprints`' output, and classify each difference by
    class.
  - Change no code, relaunch nothing, and stop. Nothing after 02 is dispatched until the user
    decides.
- `git status --short` is not empty before a launch, or a manifest stamps `git_dirty: true`.
- A manifest's `packages` is not both `null`.

---

## 5. What this prompt does not do

- It changes no tracked file but the records of §6.
- It writes no store under `var/` but its two, retires neither, and opens no other store except
  to read the two run directories §2 names.
- It does not fix what it finds. Every finding goes in the log's *Observations not acted on*, and
  opens a §3 issue if it is a defect.

---

## 6. The log, the verification record, and the board

**The files you may change, and nothing else:**
- `docs/dk-and-rpk-migration-verification.md`, new. Its first paragraph names the campaign, the
  tree, the stores and the runs. Its **§1, "R0 — the start tree"**, holds §2's measurements, with
  every command quoted verbatim;
- `docs/dk-and-rpk-migration/rehearsal/`, new: S1's scripts, any other script of yours, and saved
  outputs small enough to commit, under names that carry `02` where they are R0's. Large outputs
  stay under `var/`, and the record names them;
- the log, this board and `docs/OPEN_ISSUES.md`.

**The log**, at `logs/02-rehearse-the-start-tree.md`, in README §5.1's form. In addition, it
records:
- every run's id, slug, pid, state, exit code, driver exit status, wall time and load average;
- the interrupter's dummy tests and both real firings;
- the time the killed launches' Ray agents took to exit;
- L4's tables;
- the six suites beside the dispatch baselines.

**The board:**
- the §1 row for 02, items S1 and L1–L4, the §2 row for R0, and `### 5.2 After prompt 02 (<date>)`;
- a §3 entry for anything opened, with `docs/OPEN_ISSUES.md` in the same commit. The index gains
  this campaign's section when its first issue is opened. Correct the count and date by counting.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
