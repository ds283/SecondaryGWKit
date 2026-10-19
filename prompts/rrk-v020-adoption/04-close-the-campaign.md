# Prompt 04 — close the campaign

**Campaign:** [`README.md`](README.md) · **Board items:** **Z1**–**Z5** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-19, after the review of 03 (`ebb70a3`), as README §2.1 holds it. Its
orchestrator note, written at the same time, hands it the counts and the facts checked at
`ebb70a3`, and governs where it differs from this file.
**Gate:**
- 03 landed (`17916cb`) and was reviewed (`ebb70a3`), with **R3 and HK holding** on this board's
  §2;
- `git status` is clean;
- nothing is `running` or stale in the registry, the rehearsal lock is free, and no Ray process of
  this repository is up.

**Closes:** nothing. **Narrows:** `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` (§1,
Z4). **Opens:** nothing expected.
**Recommended model:** **Sonnet**. Records, prose and measurements, and two sentences of
`CLAUDE.md`; no code.

**Read first:**

1. [`README.md`](README.md): §0, §1, §2.2 (04's charter, with its amendment of 2026-10-19), §3,
   §5, §6.1 (**U4**) and §6.3.
2. This board: the header, §1, §2, §3 and §4, and §5.1–§5.3 with their reviews.
3. [`docs/rrk-v020-adoption-verification.md`](../../docs/rrk-v020-adoption-verification.md): the
   header, §1's and §2's opening paragraphs, §1.2.3, §2.3, §2.8 and §2.10.
4. [`logs/03-rehearse-through-v0.2.0.md`](logs/03-rehearse-through-v0.2.0.md): "Observations not
   acted on" and "State handed to the next prompt".
5. **The precedent.** `runregistry-extraction`'s close:
   [`../runregistry-extraction/09-close-the-campaign.md`](../runregistry-extraction/09-close-the-campaign.md),
   and what it wrote, [`docs/runregistry-extraction-verification.md`](../../docs/runregistry-extraction-verification.md)
   §4, in particular **§4.3.1** (a gate drafted for RunRegistryKit) and **§4.5** (what A3 v2
   inherits). This prompt follows its shape where it does not say otherwise.
6. **RunRegistryKit**, `/Users/ds283/Documents/Code/RunRegistryKit`, read with `git`, never
   changed, at the commit the orchestrator's note names:
   - `prompts/interrupted-runs/IMPLEMENTATION_STATE.md`: the header, §2 (the gates), §3's entries
     `[04-a-killed-process-not-yet-reaped-answers-kill-0]` and
     `[03-a-run-whose-manifest-cannot-be-read-names-no-store]`, and §4's entries
     `[04-runs-do-not-say-which-machine-produced-them]` and
     `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]`, each to its
     **Resolved** paragraph;
   - `prompts/interrupted-runs/README.md` §7 (the gates);
   - `prompts/extraction/IMPLEMENTATION_STATE.md` §2, the **G2** cell: the form of a gate recorded
     for SGK;
   - `docs/OPEN_ISSUES.md`;
   - at the tag: `docs/releases/v0.2.0.md` and `docs/adoption/v0.2.0/secondarygwkit.md`.
7. `CLAUDE.md`: the open-issues rule, the run-registry rules (rule 5 above all) and the rehearsal
   lock; [`prompts/INDEX.md`](../INDEX.md); [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md)
   §1.11, §1.14, §1.17, §1.21 and §1.22.
8. The two SGK board entries Z3 touches:
   `[04-runs-do-not-say-which-machine-produced-them]` in
   `prompts/run-registry/IMPLEMENTATION_STATE.md` §4, and
   `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]` in
   `prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md` §4.

---

## 1. What is wanted

The pin is moved, the route after a hard kill is adopted and has been shown on the A3 v2 launcher.
04 leaves the repository saying so; drafts the gate record the user hands to RunRegistryKit;
corrects what the index says of the issues RunRegistryKit has since closed; and says what the A3 v2
build inherits, in the verification record and, for the one step R3 added, in `CLAUDE.md`.

- **Z1 — the final measurements**, at your `HEAD` before you edit anything, each saved whole with
  the command that made it, under `measure/out/04-*`:
  - `04-roots.txt`: the seven roots, serially, against 02's counts (README §8's command);
  - `04-pip.txt`: `pip freeze`, compared with 01's after-install freeze (`01-pip.txt`, the block
    headed `pip freeze … after the install`), which it must equal; `pip check`; and
    `runregistrykit.package_provenance(("runregistrykit", "datastorekit", "raypoolkit"))`, which
    must equal, as JSON, every R3 manifest's `packages`;
  - `04-list.txt`: `./venv/bin/python -m runregistrykit list`, its exit status and its stderr;
  - `04-store-show.txt`: `store show` on **every** sidecar under `var/datastores/`, found with
    `find`, not a glob of the top level. The 21 tombstones must read as `01-store-show.txt`'s
    after-blocks; R3's three must read `kind: registry`, `problems: none`, with the fingerprint
    `39809dca…`;
  - `04-disk.txt`: `df -h /`, for Z4.
- **Z2 — RunRegistryKit's G1, drafted.** In verification §3.3, a fenced block ready to paste: the
  **status cell** of the G1 row in RunRegistryKit's `interrupted-runs` board §2, which today reads
  `not yet (SGK adopts `v0.2.0`)`.
  - **Its form** is the `extraction` board's G2 cell: `**holds** <date>: …`, naming SGK's commits
    and the runs each part rests on, then what was read and what SGK met. Nothing else on that
    board changes for it (its §2: a gate is recorded "in a records commit of its own").
  - **What it says.** The gate's own words, from its README §7: SGK "has adopted `v0.2.0` in a
    campaign of its own, its pin moved in a commit of its own". So: the campaign; 01's commit, the
    one line of `requirements.txt` and the install's provenance; then what the adoption went on to
    show: 02's rules and test, 03's R3 and hard-kill route, each with its commit and runs; then 04.
  - **Map the adoption note.** For each of `docs/adoption/v0.2.0/secondarygwkit.md` §1–§8, one
    clause giving SGK's evidence, by log, verification section or measurement. For §9's points
    that bear on the route (the defect, `end`, `begin`'s refusal, the resumed run's fingerprint),
    say where R3 showed each on the real launcher.
  - **What SGK met.** No defect of the package, if that is what the records show. R3's one finding,
    `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`, is Ray's process behaviour after a
    `SIGKILL`, and not the package's: say so in one clause, so that it is not read as a report.
  - **The date** the gate held is the day the pin moved in a commit of its own (01).

  It is **not** committed in RunRegistryKit. The user makes that commit, or has a session there make
  it (README §6.3).
- **Z3 — the issues RunRegistryKit has since closed.** Two of the registry issues SGK handed over
  on 2026-10-16 were closed in RunRegistryKit by `interrupted-runs` and released in `v0.2.0`, which
  SGK now runs: `[04-runs-do-not-say-which-machine-produced-them]` and
  `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]`. SGK's records still
  say both are open there.
  - **Confirm it** from RunRegistryKit's board: each entry in its §4, with its **Resolved** line,
    date, prompt and commit; and that `v0.2.0`'s release notes name each change.
  - **`docs/OPEN_ISSUES.md`.** Correct each present-tense statement that either issue "stays open"
    in RunRegistryKit, in §1.11, §1.17 and §1.21's lead-in, to say where and when it was closed,
    released in `v0.2.0` and adopted by SGK in this campaign (01). Where a sentence names the
    transferred issues as a group (§1.21's "each still open in RunRegistryKit"), say which are
    still open there and which closed. Nothing else in those paragraphs changes.
  - **The two boards.** Each entry, in its board's §4, gains **one** dated line, after its
    **Transferred** paragraph: `**Closed in RunRegistryKit (<date>, by interrupted-runs <NN>,
    <commit>):**`, then released in `v0.2.0`, and adopted by SGK in `rrk-v020-adoption` 01
    (`0a27b0c`); where RunRegistryKit's change differs from the entry's next step, one clause says
    so, and that RunRegistryKit's board is right. Nothing else in the entry changes.
  - The other transferred issues are **not** touched: `[00-a-copy-carries-…]`,
    `[01-cross-filesystem-move-advice-…]` and `[06-runregistrykit-list-into-a-closed-pipe-…]` are
    still open in RunRegistryKit's index. Confirm that, and say so.
- **Z4 — what A3 v2 inherits.**
  - **Verification §3.5**, for `handover`, which owns the A3 v2 launch. Start from
    `runregistry-extraction`'s §4.5 and state what is now different, from the tree and the records,
    not from memory. Each item points at its record:
    - **the launcher**, unchanged: its path, its SHA-256 at your `HEAD` against every R3 manifest's
      `script_sha256`, its last commit, the flags R3 ran it with and the one a production build
      drops (`--rehearsal`), and the `address="local"` it forces;
    - **the pins**: `runregistrykit[stores]` `v0.2.0`, `datastorekit` `v0.4.0`, `raypoolkit`
      `v0.2.0`, by line of `requirements.txt`;
    - **the commands**: §4.5's, in their current spelling, with `end` added, and what `list` now
      prints (`!!` on a stale run, `foreign`, "*N* with no host recorded");
    - **the route after a hard kill, as R3 ran it**, step by step from verification §2: `list`
      shows the run `stale`, `!!`; the resume's `begin` is refused and creates nothing; a person
      checks and runs `end` on the host it ran on; `ray-processes` (below); `store create` if the
      store has no sidecar yet; then the resume, whose fingerprint is recorded;
    - **Ray after a `SIGKILL`**, as R3 found it (verification §2.3): the raylet, workers and
      actors gone within 20 s; the GCS server left reparented to pid 1, inside its own shutdown,
      which a `SIGTERM` cannot end and a `SIGKILL` did. Until it is gone `ray-processes` is not
      empty, so the lock's `release` refuses and a next `acquire` refuses too (say with which exit
      status, from `tools/rehearsal_lock.py`). It is one observation of one kill;
    - **Ray after a `SIGTERM`**: `[07-a-killed-builds-ray-agents-outlive-the-driver]`, still open,
      and R3's §1.2.3, where nothing outlived either `SIGTERM`ed driver;
    - **what the hard kill did not test** (03's log, observation 2): no row was left unvalidated
      and no journal was left, so neither the prune at open nor DatastoreKit's journal rollback ran
      after a `SIGKILL`. A kill inside the `GkSource` stage, or inside a write, would test them;
    - **RunRegistryKit's open issues that bear on a long run**, each with what it means for A3 v2:
      `[04-a-killed-process-not-yet-reaped-answers-kill-0]` (the detached `nohup sh -c` launch is
      reaped by its `sh`) and `[03-a-run-whose-manifest-cannot-be-read-names-no-store]` (the
      manifest is written once, by `begin`);
    - **the issues that no longer bear on it**: `[a3-v2-a-hard-killed-run-blocks-its-store]`
      (closed by 03), `[00-main-and-the-readers-default-to-ray-address-auto]` (closed by `handover`
      05, `359f3fd`) and the two of Z3;
    - **its store**: the first live store meant to be kept in this layout. Every sidecar under
      `var/datastores/` is a tombstone but R3's three, which are throwaway, and the user's to
      retire (README §3, §6.3);
    - **the disk**: `04-disk.txt` against R2's 97 % and R3's 93 %, and Ray's 95 % spilling
      threshold.
  - **`CLAUDE.md`, rule 5** (the user's ruling, README §6.1 **U4**). After its sentence on `end`
    and the resume, add what R3 found, in the rule's own voice, at most two sentences. The
    substance, which is the user's ruling and is not to be weakened:
    - after `end`, check `tools.rehearsal_lock ray-processes`;
    - a `SIGKILL`ed driver's GCS server may be left alive, reparented to pid 1, stuck in its own
      shutdown, where a `SIGTERM` cannot end it (`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`);
    - one still listed ten minutes after the kill is ended by a person with `kill -KILL <pid>`, by
      its pid, never by pattern and never with `ray stop`;
    - until then the rehearsal lock cannot be released.

    Nothing else in `CLAUDE.md` changes. Quote the rule as it was and as it is in the log.
  - **The issue narrows.** `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` stays in this
    board's §3 and gains `**Narrowed (<date>, prompt 04):**`: what A3 v2 inherits is stated
    (verification §3.5, `CLAUDE.md` rule 5); what remains is the cause and any fix, which are Ray's
    or the launcher's; it has no owner. Its index row's hook says the same, in place of "**Next
    step: 04**".
- **Z5 — the records.**
  - **Verification §3, "The close"**, appended:
    - §3.1, R3's six runs as one table, read from each run directory's `manifest.json` and
      `status.json`: slug, run id, `git_head`, packages, end state and exit code, digest and
      `encoding_format`, `fingerprint_sidecar`;
    - §3.2, Z1;
    - §3.3, Z2's draft;
    - §3.4, Z3: the two issues, with RunRegistryKit's lines, and the three still open there;
    - §3.5, Z4, ending with `CLAUDE.md` rule 5's new sentences quoted.
  - **This board closes**: the status line; §1's row 04 and the items Z1–Z5; §2 unchanged but for
    a closing line; §3 as Z4 leaves it; and `### 5.4 After prompt 04 (<date>)`, listing the issue
    this campaign leaves open, with where it stands, and what is left to the user (§1, below).
  - **README §2's table**: 04's row landed, with "this commit", in 01–03's form. Nothing else in the
    README.
  - **`prompts/INDEX.md`**: this campaign's row closed, in the form of its closed neighbours, with
    its open-issue column taken by counting, and its "Opened → last board update" cell as
    `2026-10-18 → <date>`; the header's campaign counts, its date, and the sentence of dates on
    which counts were taken.
  - **`docs/OPEN_ISSUES.md`**: Z3's corrections; the narrowed row's hook; §1.22's lead-in gaining
    one sentence on 04; the header's date, and its count by counting, which stays 100.

**Left for the user, not done here** (README §6.3): retiring R3's three stores, or not; the G1
commit in RunRegistryKit; C3, if the clean error is wanted after all; and whether
`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` is assigned to a campaign.

---

## 2. Verification

1. **Z1's measurements** are each saved whole, with the command and your `HEAD`. They are taken
   before any edit, at the tree you started from, which this prompt changes only in records and
   `CLAUDE.md`: say so.
2. **Z2 and Z3** cite RunRegistryKit by file and line at the commit the note names, read with
   `git show <commit>:<path>`; every SGK commit the draft names exists, and says what is claimed
   (`git show --stat`).
3. **The boards.** For `run-registry` and `a3-v2-readiness`, `git diff -U0` holds the one added
   line, in the named entry in §4, and nothing else.
4. **The index.** `docs/OPEN_ISSUES.md`'s header count equals its rows by counting
   (`git grep -c '^| \`\['`, which reads the committed tree: count the working file with Python
   before committing). No row is added or deleted; one hook changes. `prompts/INDEX.md`'s row
   agrees with this board, and its header's counts with its rows.
5. **Z4 and §3.1** are read from the tree and the run directories, each figure with its source.
6. **`CLAUDE.md`** differs from `HEAD` in rule 5 alone, and says nothing that RunRegistryKit's
   README at the tag contradicts.
7. **The suites**, all seven, at 02's counts.

---

## 3. Acceptance

1. **Z1–Z5** as §1 says.
2. `pip freeze` equals 01's after-install freeze; the provenance equals R3's manifests'.
3. `list` shows 51 runs, 49 finished and 2 `unknown`, "45 with no host recorded", nothing `running`
   or stale, exit 0, stderr empty. Every `store show` reports `problems: none`.
4. The index's count and date agree with its rows; every board edited agrees with the index.
5. The seven roots pass at 02's counts.
6. **No code, no test, no breakage**: 04 changes no file a build, a test or a script reads.
   `CLAUDE.md` is read by sessions, not by code.
7. **One commit**, with the log `logs/04-close-the-campaign.md` in README §5.1's form. Its
   deliberate-breakage record says "none: no code or test changes", and why.

---

## 4. Stop conditions — stop and ask

- R3 or HK does not hold on this board when you start, or the registry shows anything `running` or
  stale, or the lock is held, or `ray-processes` lists anything.
- Any root count moves, or `pip freeze` differs from 01's, or a `store show` reports a problem.
- Either Z3 issue is **not** in RunRegistryKit's §4 as resolved, or one of the three Z3 leaves alone
  is not still open there, or one of Z4's two RunRegistryKit issues is closed.
- Anything would need an edit in RunRegistryKit, DatastoreKit or RayPoolKit, or a change to code,
  a test or a script here.
- Rule 5's addition cannot be written without changing another rule of `CLAUDE.md`.

---

## 5. What this prompt does not do

- **Files it changes:** `docs/rrk-v020-adoption-verification.md` (§3 appended); `CLAUDE.md` (rule
  5); `docs/OPEN_ISSUES.md`; `prompts/run-registry/IMPLEMENTATION_STATE.md` and
  `prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md` (one line each); README §2's table; this board;
  `prompts/INDEX.md`. **Files it adds:** `measure/out/04-*` and the log. Nothing else.
- It runs no build, opens no store but through `store show`, retires nothing, takes no lock and
  starts no Ray.
- It edits nothing in RunRegistryKit, DatastoreKit or RayPoolKit (README §5 rule 12), and installs
  nothing.
- It does not touch the `handover` board or A3 v2's runbook (README §1): §3.5 is written for
  `handover` to read.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
