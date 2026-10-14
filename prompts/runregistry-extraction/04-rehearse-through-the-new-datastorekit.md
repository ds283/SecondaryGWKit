# Prompt 04 — rehearse through the new DatastoreKit (R1)

**Campaign:** [`README.md`](README.md) · **Board items:** **L1**–**L4** (R1), **W1** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-14, with prompt 03, at `a5a6f6d`, after E1 was recorded as holding (board §2).
**Gate:**
- 03 landed and reviewed: `requirements.txt:17` pins `v0.4.0`, and the venv holds `datastorekit`
  0.4.0 at `c8570e849334d09085bbd84876095fc9e22bd005`, not editable, with `raypoolkit` 0.2.0 at
  `eaef0ce6…` and no `runregistrykit`;
- `git status` is clean, and no commit after 03's review touches code (anything outside `prompts/`
  and `docs/`);
- `python -m RunRegistry list` shows nothing `running`;
- R0's two stores, `var/datastores/runregistry-extraction-02-{reference,resumed}.sqlite`, are still
  live (not retired): this prompt fingerprints them both ways too;
- the rehearsal lock is free (`PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock status`), and
  `ray-processes` lists nothing.

**Closes:** nothing. **Opens:** anything you find, **without fixing it**.
**Gate it decides:** **R1** (board §2).
**Recommended model:** **Opus**. Four registered builds and the readers under Ray, the first
rehearsal under the shared lock, and one verification script.

**Read first:**

1. `CLAUDE.md`: "Long-running jobs — the run registry", the freeze section, and **"The shared
   rehearsal lock"**, in full.
2. [`README.md`](README.md): §0.1, §2.2 (04's charter), §3, **§4.1** (the protocol and the lock),
   §5 (rules 8–11, 14, 15) and §6.2 (C3).
3. **[`02-rehearse-the-start-tree.md`](02-rehearse-the-start-tree.md)**, which this prompt runs again
   with the names and differences below, and through it `dk-and-rpk-migration`'s
   [`08-rehearse-through-both-packages.md`](../dk-and-rpk-migration/08-rehearse-through-both-packages.md).
   Where this prompt is silent, 02 governs, and through it 08.
4. **[`docs/runregistry-extraction-verification.md`](../../docs/runregistry-extraction-verification.md)
   §1**, R0: how each launch went, the commands verbatim, the interrupter's points, and L4's tables.
   Your §2 has the same shape and is compared with it.
5. `docs/runregistry-extraction/rehearsal/`: R0's saved outputs (`*_02*.txt`), which are what yours
   should look like.
6. **DatastoreKit's `v0.4.0`:** `datastorekit/store_fingerprint.py` as installed, and its checklist
   addendum for SGK, items 1–7 (`git -C /Users/ds283/Documents/Code/DatastoreKit show
   v0.4.0:docs/adoption/secondarygwkit.md`, "Addendum: v0.4.0").
7. The board: §3, `[00-recorded-fingerprints-lack-the-encoding-format]`; §5.2 (R0's review); 03's
   log, "State handed to the next prompt".
8. [`tools/rehearsal_lock.py`](../../tools/rehearsal_lock.py)'s module docstring, and 02a's log
   "State handed to the next prompt".

---

## 0. What this prompt is

R0 (02) reproduced `39809dca…` at the start tree, through `datastorekit` 0.2.2. 03 moved the pin to
`v0.4.0` and changed nothing else. **R1 is R0 again at 03's tree.** If it reproduces `39809dca…`,
the three releases between the pins move nothing a build does. A miss with the build otherwise sound
is DatastoreKit's (README §5 rule 11).

**R1 also takes every fingerprint a second way** (README §6.2 C3). SGK's `fingerprint_of` is still
the one the registry calls. The package's `fingerprint_of`, given the same inventory and SGK's
format 6, must give the same fingerprint in every field but one: `encoding_format`, which the
package adds and SGK's code does not have. This is the check DatastoreKit's own tests cannot make,
since they cannot name SGK. It is made here, on real stores, before 06 makes SGK depend on that code.

**What this prompt is not.** It changes no code and no test. The one file it adds is the
verification script of §2, under `docs/`, written **after the last launch has ended**. Until then
it creates no file in the repository outside `var/`, so that every manifest records
`git_dirty: false`. Working notes go under `var/runregistry-extraction-04/`.

---

## 1. The protocol, with this prompt's names

**Run 02's §1 as written**, with these substitutions:
- **Stores:** `var/datastores/runregistry-extraction-04-reference.sqlite` and
  `var/datastores/runregistry-extraction-04-resumed.sqlite`. Before the first launch, `store show`
  on each gives kind `absent` and no runs; if not, stop.
- **Runs:** `--campaign runregistry-extraction --prompt 04`, slugs `reference`, `resumed-a`,
  `resumed-b`, `resumed-c`, and `--job-name runregistry-extraction-04` on every launch. Each
  `--purpose` names R1 of this campaign, the tree's commit, `datastorekit` 0.4.0 at `c8570e8`, and
  the fingerprint it must equal.
- **Launch logs:** `var/runs/runregistry-extraction-04-<slug>.launch.log`.
- **Reader outputs:** `var/runregistry-extraction-04/extract/`.
- **Saved outputs** you commit go to `docs/runregistry-extraction/rehearsal/`, named as R0's with
  `04` (`prelaunch_checks_04.txt`, `stage_counts_04.txt`, …).

**What is new since R0:**
- **The lock**, README §4.1, through the helper, from the repository root:
  - **Before the `reference` launch:** `PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock
    acquire --purpose "runregistry-extraction 04 (R1)"`. **Exit 3 (held) or 4 (a Ray process of
    this repository alive): launch nothing, stop and ask.** Never wait, retry or remove it.
  - **At each launch**, in the one check that confirms the run started, `set-pid <pid>` with the
    driver's pid from that run's `status.json`.
  - **Held through all four launches and L3's readers.** One lock for the whole rehearsal.
  - **After the readers:** `release`. It refuses, exit 1, while the recorded pid is alive or
    `ray-processes` lists anything; then wait by pid for the agents, as README §3 says, and release.
    A refusal naming another owner: stop and ask.
  - **If you stop early**, release the same way once no Ray process of this repository is alive; if
    you cannot, say so in your report.
  - Save every helper command and its exit status in `lock_04.txt`.
- **The Ray check before each launch** is `ray-processes` (exit 0, nothing listed), **not** a
  machine-wide `pgrep`, which now also finds other projects' clusters (README §4.1). Record both if
  you like; only the helper's is the test. A refusal lists agents by pid and a cut command line;
  name them with `ps -p <pid> -o command=`.
- **Ray addresses** as README §4.1: the driver forces `address="local"`, `run_readers.sh` passes
  `--ray-address local`. Nothing runs `main.py` or an `extract_*.py` at its default. **`ray stop`
  is never run.**
- **The pre-launch checks** find `datastorekit` 0.4.0 at `c8570e84…`, `raypoolkit` 0.2.0 at
  `eaef0ce6…`, both `editable: false`, and no `runregistrykit`.

What holds from 02, unchanged: the launch command (README §4.1, verbatim but for the store, the
slug, the purpose and `--resume`); **one check after each launch, then end your turn**, the
orchestrator waiting on the run's liveness and resuming you, never polling, never `pgrep -f`; the
wait by pid for a killed launch's agents, timed beside R0's; the interrupter's two dummy tests
before the first launch; interruption points of R1's own, chosen from R1's `reference`
`stdout.log` by R0's reasoning, and recorded with it (R0's 40 % hold on `GkSource`, which made the
prune at open a measured clause, is the precedent to follow); the sidecars as R0's; and the six
scripts of `docs/dk-and-rpk-migration/rehearsal/` run as they stand, from where they are. A script
that fails where its R0 run did not: stop.

---

## 2. The fingerprint both ways (W1)

**After the last launch has ended and the readers have run**, add one verification script,
**`docs/runregistry-extraction/fingerprint_both_ways.py <primary>...`** (README §4, 04's line; 08
runs it again). It is a script, not a test (README §5 rule 10).

**For each primary it is given**, read-only and with no Ray:
1. `inventory = read_inventory(primary, factories)`, with `read_inventory` from
   `datastorekit.store_inventory` and `factories` from `config.datastore`, as
   `RunRegistry.stores.fingerprint_store` reads them;
2. `sgk = RunRegistry.stores.fingerprint_of(inventory)`;
3. `pkg = datastorekit.store_fingerprint.fingerprint_of(inventory, RunRegistry.stores.FINGERPRINT_FORMAT)`,
   SGK's format passed from SGK's constant, not typed;
4. **the comparison**, each line printed with its result:
   - `pkg["encoding_format"] == datastorekit.store_fingerprint.ENCODING_FORMAT`, and `== 1`;
   - `"encoding_format" not in sgk`;
   - **with `encoding_format` removed from `pkg`, `pkg == sgk`** as dictionaries: the same keys,
     `fingerprint_format` 6, the overall `digest`, `problems`, `taken` (`None` on both), and for
     every class, in the same order, `count`, `digest` and every tag set's `tags`, `count` and
     `digest`. On a difference, print each differing path (class, tag set, field) with both values;
   - the overall digest printed, and checked equal to `39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`;
   - **the listings**: SGK's `RunRegistry.stores.listing_lines(inventory, sgk)` and the package's
     `listing_lines(inventory, pkg)` are equal line for line but the first, and the package's first
     line is SGK's with ` encoding_format 1` appended (addendum item 4);
   - **the comparisons, as the addendum states them**: the package's `compare_fingerprints(pkg, pkg)`
     is empty; SGK's `compare_fingerprints(sgk, sgk)` is empty; and the package's
     `compare_fingerprints(sgk, pkg)` is exactly one entry, of kind `encoding`. This last is the
     reason for `[00-recorded-fingerprints-lack-the-encoding-format]`, measured, not a failure.
5. **against the sidecar**: SGK's `compare_fingerprints` of the sidecar's recorded fingerprint with
   `sgk` is empty.

It prints one block per store and a summary line, and **exits 0 if and only if every check holds on
every store**. It writes nothing, anywhere. It imports `RunRegistry` and does not change it.

**Run it on all four live stores**, R1's two and R0's two:

```bash
PYTHONPATH=. ./venv/bin/python docs/runregistry-extraction/fingerprint_both_ways.py \
    var/datastores/runregistry-extraction-04-reference.sqlite \
    var/datastores/runregistry-extraction-04-resumed.sqlite \
    var/datastores/runregistry-extraction-02-reference.sqlite \
    var/datastores/runregistry-extraction-02-resumed.sqlite
```

saved whole as `fingerprint_both_ways_04.txt`, with every store file's SHA-256, size and mtime
before and after, equal.

**Deliberate breakage** (README §5 rule 8). On a copy of the script in your scratch directory, never
committed, each of these must make it exit non-zero and say which check failed:
1. `encoding_format` is not removed from `pkg` before the dictionary comparison;
2. one class's `digest` in `pkg` is altered after it is taken: the output names that class;
3. format `5` is passed to the package's `fingerprint_of`: the overall digest differs, and
   `fingerprint_format` is named;
4. one tag set's `count` in `sgk` is incremented: the output names the class and the tags;
5. the listing check compares the first lines unmodified.

The log records each as a diff, exactly as applied, with the exit status and the line that names
the failure.

---

## 3. What to verify

**L1–L3 are 02's §2**, on this prompt's stores and runs, with these differences:
- **The manifests.** Every `packages` names `datastorekit` 0.4.0, commit
  `c8570e849334d09085bbd84876095fc9e22bd005`, `requested_revision` `v0.4.0`, and `raypoolkit`
  exactly as R0's, both `editable: false`; every `git_dirty` is `false`; every `git_head` the
  dispatch `HEAD`.
- **The fingerprints.** Both stores `39809dca…`, format 6, 21 classes, each equal to its sidecar's;
  `compare_fingerprints` between them empty; `compare_with_recorded.py` against **R0's run
  directories** `var/runs/runregistry-extraction-02-reference-20261012T091921` and
  `…-02-resumed-c-20261012T101145`, each empty. Both sides are SGK's fingerprints, so SGK's
  comparison applies.
- **What is new in the logs** is compared with R0's same slug. A new line that names
  `datastorekit`, a `DeprecationWarning`, or the `_timestamp` of `v0.3.0`'s `DatastoreObject` is a
  finding to explain.

**L4 — equal to R0.** Against verification §1.5, as 02's §2 states it against R2: the statistics
block's counts (291; numeric 109; mixed 26, 25 complete and 1 minimal; WKB 156), as counts and as
sorted lines; the reference stage counts; the resumed chain's identities; the stage sequence; the
printed lines and the registry's fields; the stage times beside R0's, not compared. A difference in
L4 with L1–L3 holding is a finding to explain from the three releases, and to open as a §3 issue.

**W1** is §2: `fingerprint_both_ways.py` exits 0 on all four stores.

---

## 4. Acceptance

1. **L1–L3** hold as §3 says: both stores `39809dca…`, equal to each other and to R0's recorded
   fingerprints; every manifest's `packages` names `v0.4.0` at `c8570e84…`; every `git_dirty`
   `false`.
2. **The fingerprint both ways**: `fingerprint_both_ways.py` exits 0 on R1's two stores and R0's
   two, every field equal but `encoding_format`, and the five breakages of §2 each caught.
3. **L4** against R0, or each difference recorded, explained and opened.
4. **The lock** was taken before `reference`, its `pid` set at each launch, held through the
   readers, and released, every command and exit status in `lock_04.txt`; afterwards `status`
   reports no lock and `ray-processes` lists nothing.
5. **The seven roots** pass at the dispatch `HEAD`, at 03's counts.
6. `docs/runregistry-extraction-verification.md` gains **§2, "R1 — through DatastoreKit `v0.4.0`"**,
   in §1's shape, with a subsection for W1. §1 is not edited.
7. **The board records R1 as holding**, naming the two runs that finished `done`.

---

## 5. Stop conditions — stop and ask

**02 §4's conditions apply**, with R0 in place of R2, and:
- **A fingerprint differs from `39809dca…`.** Record both fingerprints and `compare_fingerprints`,
  change nothing, relaunch nothing, stop. At R1 the cause is DatastoreKit's (README §4.1).
- **`fingerprint_both_ways.py` finds a difference** other than `encoding_format`, on any store,
  with L1 holding. That is a DatastoreKit defect (README §5 rule 11, §6.2 C3): record the paths and
  both values, stop.
- **The lock is held** (exit 3), a Ray process of this repository is alive at `acquire` (exit 4),
  or `release` refuses on another owner.
- A manifest's `packages` is not as §3 says, or a `git_dirty` is `true`.
- A rehearsal script fails where its R0 run did not.

---

## 6. What this prompt does not do

- It changes no tracked file but the records of §7 and the one new script.
- It writes no store under `var/` but its two, and **retires none**, R0's included: the
  orchestrator retires R0's and R1's four after this prompt's review (board §5.5).
- It does not call the package's fingerprint code from SGK code; the script is a verification step.
- It does not fix what it finds.

---

## 7. The log, the verification record, and the board

**The files you may change, and nothing else:**
- `docs/runregistry-extraction/fingerprint_both_ways.py`, new, after the last launch (`black`-clean);
- `docs/runregistry-extraction-verification.md`: **§2** added, and the header's list of sections
  left as it is unless a line of it is now false;
- `docs/runregistry-extraction/rehearsal/`: saved outputs named with `04`, `lock_04.txt` and
  `fingerprint_both_ways_04.txt` among them;
- the log, this board and `docs/OPEN_ISSUES.md`.

**The log**, at `logs/04-rehearse-through-the-new-datastorekit.md`, in README §5.1's form, with what
02's log records, against R0, and §2's five breakages as diffs.

**The board:** the §1 row for 04, items L1–L4 (R1) and W1, the §2 row R1, and
`### 5.7 After prompt 04 (<date>)`; a §3 entry for anything opened, with `docs/OPEN_ISSUES.md`
§1.21 in the same commit, the count and date corrected by counting.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
