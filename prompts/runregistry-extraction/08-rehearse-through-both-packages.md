# Prompt 08 — rehearse through both packages (R2)

**Campaign:** [`README.md`](README.md) · **Board items:** **L1**–**L4** (R2), **G-RRK** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-15, with prompts 05–07, at `14bcac0`, after E2 was recorded as holding (board
§2, §5.10).
**Gate:**
- 07 landed and reviewed: the guard passes, and the roots pass at 07's counts;
- the venv holds `runregistrykit` 0.1.0 at `9f8ebe43…`, `datastorekit` 0.4.0 at `c8570e84…` and
  `raypoolkit` 0.2.0 at `eaef0ce6…`, none editable;
- `git status` is clean, and no commit after 07's review touches code (anything outside `prompts/`
  and `docs/`);
- `./venv/bin/python -m runregistrykit list` shows nothing `running`, and 39 runs: 01's 29, R0's and
  R1's 8, and 06's two (`worktree-check`, and `worktree-check-breakage` from its breakage 8);
- R0's and R1's run directories hold their recorded fingerprints:
  `var/runs/runregistry-extraction-02-reference-20261012T091921`, `…-02-resumed-c-20261012T101145`,
  `…-04-reference-20261014T154651` and `…-04-resumed-c-20261014T161739`. Their stores are retired
  (board §5.8); the run directories are what R2 is compared with;
- the rehearsal lock is free (`PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock status`), and
  `ray-processes` lists nothing.

**Closes:** nothing. **Opens:** anything you find, **without fixing it**.
**Gates it decides:** **R2**, and **RunRegistryKit's G2** (board §2).
**Recommended model:** **Opus**. Four registered builds and the readers under Ray, through both
packages, under the shared lock.

**Read first:**

1. `CLAUDE.md`: "Long-running jobs — the run registry", "The shared rehearsal lock", and the
   packages section, all as 06 left them.
2. [`README.md`](README.md): §0.1 (why R2 isolates the swap), §2.2 (08's charter), §3, **§4** (06's
   and 08's lines: what 08 repeats of W1), **§4.1** (the protocol and the lock), §5 (rules 8–11,
   15), and §6.2 (C3, C10, C11).
3. **[`04-rehearse-through-the-new-datastorekit.md`](04-rehearse-through-the-new-datastorekit.md)**,
   which this prompt runs again with the names and differences below, and through it
   [`02-rehearse-the-start-tree.md`](02-rehearse-the-start-tree.md) and `dk-and-rpk-migration`'s
   [`08-rehearse-through-both-packages.md`](../dk-and-rpk-migration/08-rehearse-through-both-packages.md).
   Where this prompt is silent, 04 governs, and through it 02, and through it that 08.
4. **[`docs/runregistry-extraction-verification.md`](../../docs/runregistry-extraction-verification.md)
   §2**, R1: how each launch went, the commands verbatim, the interrupter's points, the lock, W1, and
   L4's tables. Your §3 has the same shape and is compared with it.
5. `docs/runregistry-extraction/rehearsal/`: R1's saved outputs (`*_04*.txt`), which are what yours
   should look like.
6. **06's log**: "State handed to the next prompt", and the commands it states for README §4.1
   (`runregistrykit` in place of `RunRegistry`), which this prompt runs. Also
   `docs/runregistry-extraction/fingerprint_both_ways.py` as 06 rewrote it, and
   `docs/dk-and-rpk-migration/rehearsal/` as 06 left it.
7. RunRegistryKit's checklist for SGK, item 9 ("Its acceptance"), at `aaab408`: what G2 asks.
8. The board: §3, `[00-recorded-fingerprints-lack-the-encoding-format]` (in §4 after 06); §5.7
   (R1's review, and its note on the waits); 07's log, "State handed to the next prompt".

---

## 0. What this prompt is

R1 (04) reproduced `39809dca…` through `datastorekit` 0.4.0, with SGK's own registry. 05 installed
RunRegistryKit, 06 made it the registry, and 07 added a test. **R2 is R1 again at 07's tree**,
through both packages: `runregistrykit` registers the runs, takes the fingerprints at finish and
writes the sidecars, through `config/fingerprint.py`'s fingerprinter over DatastoreKit's code. If R2
reproduces `39809dca…`, the swap moves nothing a build writes or a fingerprint reads. A miss with
the build otherwise sound is RunRegistryKit's or DatastoreKit's (README §5 rule 11).

**What R2 repeats of W1** (README §4, amended 2026-10-14). 04 showed SGK's fingerprint and the
package's agree on the same store. SGK's is deleted, and R0's and R1's stores are retired. So R2
carries W1's claim across stores: the package's fingerprint of each R2 store equals, in every field
but `encoding_format`, the fingerprints SGK's code recorded for R0 and R1. That is acceptance 1,
through 06's `fingerprint_both_ways.py --against`. It is not `compare_with_recorded.py`, which now
compares through the package and gives one `encoding` entry for every fingerprint SGK's code
recorded; that script is run on R2's own stores and sidecars only.

**What this prompt is not.** It changes no code and no test. It writes no file in the repository
outside `var/` until the last launch has ended, so that every manifest records `git_dirty: false`.
Working notes go under `var/runregistry-extraction-08/`.

---

## 1. The protocol, with this prompt's names

**Run 04's §1 as written**, with these substitutions:
- **Stores:** `var/datastores/runregistry-extraction-08-reference.sqlite` and
  `var/datastores/runregistry-extraction-08-resumed.sqlite`. Before the first launch, `store show`
  on each gives kind `absent` and no runs; if not, stop.
- **Runs:** `--campaign runregistry-extraction --prompt 08`, slugs `reference`, `resumed-a`,
  `resumed-b`, `resumed-c`, and `--job-name runregistry-extraction-08` on every launch. Each
  `--purpose` names R2 of this campaign, the tree's commit, `runregistrykit` 0.1.0 at `9f8ebe4` and
  `datastorekit` 0.4.0 at `c8570e8`, and the fingerprint it must equal.
- **Launch logs:** `var/runs/runregistry-extraction-08-<slug>.launch.log`.
- **Reader outputs:** `var/runregistry-extraction-08/extract/`.
- **Saved outputs** you commit go to `docs/runregistry-extraction/rehearsal/`, named as R1's with
  `08` (`prelaunch_checks_08.txt`, `lock_08.txt`, `stage_counts_08.txt`, …).
- **The lock:** `acquire --purpose "runregistry-extraction 08 (R2)"`, and the rest of 04's lock
  steps unchanged.

**What is new since R1:**
- **Every registry command is `runregistrykit`'s**, as 06's log states them: `./venv/bin/python -m
  runregistrykit list`, `… store show`, `… store create`, `… store fingerprint [--write]`. Commands
  that also import SGK's own modules (the drivers, the readers, the rehearsal scripts) keep
  `PYTHONPATH=.`.
- **The pre-launch checks** find `runregistrykit` 0.1.0 at `9f8ebe43…`, `datastorekit` 0.4.0 at
  `c8570e84…`, `raypoolkit` 0.2.0 at `eaef0ce6…`, all `editable: false`, through
  `runregistrykit.package_provenance(("runregistrykit", "datastorekit", "raypoolkit"))` or as 06's
  log says; and `runregistrykit.toml` read by `find_client_root()` from the root gives `var/runs`,
  `var/datastores` and `config.fingerprint:fingerprinter`.
- **The waits end on the pid's death**, not on `state == "finished"`: a killed run's `status.json`
  says `killed` some seconds before its fingerprint and final heartbeat are written (board §5.7).

What holds from 04, unchanged: the launch command (README §4.1); **one check after each launch,
then end your turn**, the orchestrator waiting on the run's liveness and resuming you, never
polling, never `pgrep -f`; the lock, held from before `reference` through L3's readers; `set-pid` at
each launch; the wait by pid for a killed launch's agents; the interrupter's two dummy tests before
the first launch; interruption points of R2's own, chosen from R2's `reference` `stdout.log` by
R1's reasoning (the 40 % hold on `GkSource`, a third of the QSI stage), and recorded with it; the
sidecars as R1's; the Ray addresses (`local`, never `auto`; **`ray stop` is never run**); and the
six scripts of `docs/dk-and-rpk-migration/rehearsal/` run as 06 left them. A script that fails
where its R1 run did not: stop.

---

## 2. What to verify

**L1–L3 are 04's §3**, on this prompt's stores and runs, with these differences:
- **The manifests.** Every `packages` names, in this order, `runregistrykit` 0.1.0 at
  `9f8ebe431503fb8c79ee1407cc460501f1ca2d92` (`requested_revision` `v0.1.0`), `datastorekit` 0.4.0
  at `c8570e849334d09085bbd84876095fc9e22bd005` (`v0.4.0`) and `raypoolkit` 0.2.0 at `eaef0ce6…`
  (`v0.2.0`), all `editable: false`; every `git_dirty` is `false`; every `git_head` the dispatch
  `HEAD`. Compared as JSON with what the pre-launch check printed.
- **The fingerprints.** Both stores format 6, `39809dca…`, 21 classes, **each with
  `encoding_format` 1**, each equal to its sidecar's; the package's `compare_fingerprints` between
  them empty; and each run's fingerprint in its `status.json` carrying `encoding_format`.
- **`compare_with_recorded.py`** on both primaries **against R2's own `reference` and `resumed-c`
  run directories only**: each comparison empty. Not against R0's or R1's (README §4).
- **What is new in the logs** is compared with R1's same slug. A new line that names
  `runregistrykit`, `datastorekit` outside a path, a `DeprecationWarning`, or a `ConfigError` is a
  finding to explain.

**W1 across stores** (README §4; acceptance 1):

```bash
PYTHONPATH=. ./venv/bin/python docs/runregistry-extraction/fingerprint_both_ways.py \
    var/datastores/runregistry-extraction-08-reference.sqlite \
    var/datastores/runregistry-extraction-08-resumed.sqlite \
    --against \
    var/runs/runregistry-extraction-02-reference-20261012T091921 \
    var/runs/runregistry-extraction-02-resumed-c-20261012T101145 \
    var/runs/runregistry-extraction-04-reference-20261014T154651 \
    var/runs/runregistry-extraction-04-resumed-c-20261014T161739
```

run after the last launch has ended and the readers have run, saved whole as
`fingerprint_both_ways_08.txt`, with every store file's SHA-256, size and mtime before and after,
equal. It must exit 0: in each of the eight pairs every field equal but `encoding_format` (and
`taken`), the recorded fingerprint without `encoding_format`, the package's comparison exactly one
`encoding` entry, and each store equal to its sidecar. This prompt adds no breakage of the script:
06 named them, on the same code.

**L4 — equal to R1.** Against verification §2.5, as 04's §3 states it against R0: the statistics
block's counts (291; numeric 109; mixed 26, 25 complete and 1 minimal; WKB 156), as counts and as
sorted lines; the reference stage counts; the resumed chain's identities; the stage sequence; the
printed lines and the registry's fields; the stage times beside R1's, not compared. A difference in
L4 with L1–L3 holding is a finding to explain from the swap, and to open as a §3 issue.

**The registry after R2** (acceptance 4): `./venv/bin/python -m runregistrykit list` shows 01's 29
runs and this campaign's fourteen (R0's, R1's and R2's twelve launches, and 06's two worktree
runs), 43 in all, and nothing `running`. Saved as
`run_records_08.txt`, with the counts by state.

**G2** (acceptance 5). RunRegistryKit's checklist, item 9, against what 06, 07 and this prompt
show, each item with its evidence: no `RunRegistry` outside `prompts/` and dated `docs/` prose (06
§7.6, re-run here at the dispatch `HEAD`); the roots (06, 07, and here); `list` and `store show`
(06 §7.1); the worktree run (06 §7.4); `encoding_format` beside the digest (here); the pin and the
manifests' `packages` (here). If every item holds, the board records G2 as holding.

---

## 3. Acceptance

1. **L1–L3** hold as §2 says: both stores `39809dca…`, format 6, `encoding_format` 1, equal to each
   other and to their sidecars; **`fingerprint_both_ways.py` exits 0 against R0's and R1's four
   recorded fingerprints**, every field equal but `encoding_format`.
2. **Every manifest's `packages`** names `runregistrykit`, `datastorekit` and `raypoolkit` at their
   pins, none editable; every `git_dirty` `false`.
3. **L4** against R1, or each difference recorded, explained and opened.
4. **`list`** shows 01's 29 runs and this campaign's fourteen, 43 in all, nothing `running`.
5. **The board records RunRegistryKit's G2 as holding** (README §6.2 C11), item by item, and the
   R2 row of §2.
6. **The lock** was taken before `reference`, its `pid` set at each launch, held through the
   readers, and released, every command and exit status in `lock_08.txt`; afterwards `status`
   reports no lock and `ray-processes` lists nothing.
7. **The seven roots** pass at the dispatch `HEAD`, at 07's counts.
8. `docs/runregistry-extraction-verification.md` gains **§3, "R2 — through both packages"**, in
   §2's shape, with a subsection for W1 across stores and one for G2. §1 and §2 are not edited.

## 4. Stop conditions — stop and ask

**04 §5's conditions apply**, with R1 in place of R0, and:
- **A fingerprint differs from `39809dca…`.** Record both fingerprints and the package's
  `compare_fingerprints`, change nothing, relaunch nothing, stop. At R2 the cause is RunRegistryKit's
  or DatastoreKit's (README §4.1).
- **`fingerprint_both_ways.py` finds a difference** other than `encoding_format`, in any pair, with
  L1 holding: record the paths and both values, stop.
- A manifest's `packages` is not as §2 says, or a `git_dirty` is `true`.
- `runregistrykit` refuses a command 06's log says works (a `ConfigError`, a missing
  fingerprinter), or writes a record anywhere but `var/runs/` and `var/datastores/`.
- The lock is held (exit 3), a Ray process of this repository is alive at `acquire` (exit 4), or
  `release` refuses on another owner.
- A rehearsal script fails where its R1 run did not.

## 5. What this prompt does not do

- It changes no tracked file but the records of §6.
- It writes no store under `var/` but its two, and **retires none**: R2's two are the user's to
  retire after 09's review, or not (README §3).
- It does not record G2 on RunRegistryKit's board: 09 drafts that record, and the user, or a session
  there, commits it (README §6.2 C11).
- It does not fix what it finds.

## 6. The log, the verification record, and the board

**The files you may change, and nothing else:**
- `docs/runregistry-extraction-verification.md`: **§3** added, and the header's list of sections
  left as it is unless a line of it is now false;
- `docs/runregistry-extraction/rehearsal/`: saved outputs named with `08`, `lock_08.txt` and
  `fingerprint_both_ways_08.txt` among them;
- the log, this board and `docs/OPEN_ISSUES.md`.

**The log**, at `logs/08-rehearse-through-both-packages.md`, in README §5.1's form, with what 04's
log records, against R1.

**The board:** the §1 row for 08, items L1–L4 (R2) and G-RRK, the §2 rows R2 and RunRegistryKit
G2, and `### 5.14 After prompt 08 (<date>)`; a §3 entry for anything opened, with
`docs/OPEN_ISSUES.md` §1.21 in the same commit, the count and date corrected by counting.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
