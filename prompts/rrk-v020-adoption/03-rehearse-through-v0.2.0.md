# Prompt 03 — rehearse through `v0.2.0` (R3), and the hard-kill route

**Campaign:** [`README.md`](README.md) · **Board items:** **L1**–**L4** (R3), **H1**, **H2** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-18, with prompts 01–02, at `79434bf`.
**Gate:**
- 02 landed and reviewed: the roots pass at 02's counts, and both interrupter dummy tests pass,
  case D included;
- the venv holds `runregistrykit` 0.2.0 at `0bae3a3c…`, `datastorekit` 0.4.0 at `c8570e84…` and
  `raypoolkit` 0.2.0 at `eaef0ce6…`, none editable;
- `git status` is clean, and no commit after 02's review touches code (anything outside `prompts/`
  and `docs/`);
- `./venv/bin/python -m runregistrykit list` shows README §0.2's 45 runs and nothing `running` or
  stale;
- R2's run directories hold their recorded fingerprints:
  `var/runs/runregistry-extraction-08-reference-20261015T235716` and
  `var/runs/runregistry-extraction-08-resumed-c-20261016T000959`. R2's stores are retired
  (`5db42bf`); these run directories are what R3 is compared with;
- the three stores `var/datastores/rrk-v020-adoption-03-{reference,resumed,hardkill}.sqlite` are
  `absent` to `store show`, with no runs;
- the rehearsal lock is free (`PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock status`), and
  `ray-processes` lists nothing.

**Closes:** `[a3-v2-a-hard-killed-run-blocks-its-store]` (`handover` board §3; README §6.2 C6),
when acceptance 3 holds. **Opens:** anything you find, **without fixing it**.
**Gates it decides:** **R3** and **HK** (board §2).
**Recommended model:** **Opus**. Six registered builds, one refused launch, an `end`, and the
readers under Ray, under the shared lock.

**Read first:**

1. `CLAUDE.md`: "Long-running jobs — the run registry" as 02 left it (the route after a hard kill),
   "The shared rehearsal lock", and the packages section.
2. [`README.md`](README.md): §0.1 (why one rehearsal, and why the hard kill has a store of its
   own), §2.2 (03's charter), §3, **§4.1** (the protocol and the hard-kill pair), §5 (rules 8–11,
   14, 15) and §6 (U2, U3, C6–C8).
3. **`runregistry-extraction`'s
   [`08-rehearse-through-both-packages.md`](../runregistry-extraction/08-rehearse-through-both-packages.md)**,
   which this prompt runs again with the names and differences below. Through it, read that
   campaign's `04-…` and `02-…`, and `dk-and-rpk-migration`'s `08-…`. Where this prompt is silent,
   that 08 governs, and through it the chain it names.
4. **[`docs/runregistry-extraction-verification.md`](../../docs/runregistry-extraction-verification.md)
   §3**, R2: how each launch went, the commands verbatim (§3.1), the interrupter's points (§3.1.2),
   the lock (§3.1.3), the waits for Ray's processes (§3.2.3), and L4's tables (§3.5). Your §1 has
   the same shape and is compared with it. §4.5 there is the launcher A3 v2 uses, which is the
   launcher you run.
5. `docs/runregistry-extraction/rehearsal/`: R2's saved outputs (`*_08.txt`), which yours should
   look like. `docs/dk-and-rpk-migration/rehearsal/`: the six scripts and `run_readers.sh`, as they
   stand.
6. **RunRegistryKit at the tag**, read with `git -C /Users/ds283/Documents/Code/RunRegistryKit show
   v0.2.0:<path>`: `docs/adoption/v0.2.0/README.md` §4 (the route after a hard kill) and
   `docs/adoption/v0.2.0/secondarygwkit.md` §4 (the refusal SGK's launcher will print);
   `docs/releases/v0.2.0.md` §2 (the `host` and `ended` records).
7. The `handover` board's §3, `[a3-v2-a-hard-killed-run-blocks-its-store]`, whose six-step
   measurement on a fixture store, under `v0.1.0`, this prompt repeats on the real launcher under
   `v0.2.0`.
8. 01's and 02's logs, "State handed to the next prompt": 01's `this_host()` reading, and 02's
   `interrupt_when.py --signal`.

---

## 0. What this prompt is

R2 reproduced `39809dca…` through `runregistrykit` 0.1.0. 01 moved the pin to `v0.2.0`, and 02
wrote the rule and taught the interrupter to send `SIGKILL`. **R3 is two things on one tree, under
one lock:**
- **R2 again, through `v0.2.0`**: the four-launch protocol unchanged. If it reproduces `39809dca…`,
  the pin moves nothing a build writes or a fingerprint reads. The `SIGTERM`'d launches also run
  `v0.2.0`'s new status writer under the drivers' handlers. A miss with the build otherwise sound is
  RunRegistryKit's or DatastoreKit's (README §5 rule 11).
- **The first hard kill of a rehearsal build.** It is run on a third store, so that its result
  cannot be confused with the pin's (README §0.1, U2). It shows A3 v2's route after a hard kill on
  A3 v2's own launcher, which is what closes `[a3-v2-a-hard-killed-run-blocks-its-store]`. It also
  measures two things no rehearsal has:
  - whether a store a build left with no handler having run resumes to the reference;
  - whether Ray's processes outlive a driver that had no chance to shut them down.

**What this prompt is not.** It changes no code and no test. It writes no file in the repository
outside `var/` until the last launch has ended, so that every manifest records `git_dirty: false`.
Working notes go under `var/rrk-v020-adoption-03/`.

---

## 1. The four launches, with this prompt's names

**Run R2's §1 as written** (`runregistry-extraction` 08 §1), with these substitutions:
- **Stores:** `var/datastores/rrk-v020-adoption-03-reference.sqlite` and
  `var/datastores/rrk-v020-adoption-03-resumed.sqlite`.
- **Runs:** `--campaign rrk-v020-adoption --prompt 03`, slugs `reference`, `resumed-a`,
  `resumed-b` and `resumed-c`, and `--job-name rrk-v020-adoption-03` on every launch. Each
  `--purpose` names:
  - R3 of this campaign;
  - the tree's commit;
  - `runregistrykit` 0.2.0 at `0bae3a3`;
  - the fingerprint it must equal, and R2's run that recorded it.
- **Launch logs:** `var/runs/rrk-v020-adoption-03-<slug>.launch.log`.
- **Reader outputs:** `var/rrk-v020-adoption-03/extract/`.
- **Saved outputs** you commit go to `docs/rrk-v020-adoption/rehearsal/`, named as R2's with `03`
  (`prelaunch_checks_03.txt`, `lock_03.txt`, `stage_counts_03.txt`, …).
- **The lock:** `acquire --purpose "rrk-v020-adoption 03 (R3)"`. It is held from before `reference`
  until after the hard-kill pair's checks (§2), and `set-pid` is run at each launch that begins a
  run.

**What is new since R2:**
- **The pre-launch checks** find `runregistrykit` 0.2.0 at `0bae3a3c…`, `requested_revision`
  `v0.2.0`, beside `datastorekit` and `raypoolkit` as at R2, all `editable: false`. They record
  `this_host()`, whose `id` must equal 01's reading and whose `boot` is today's: no reboot since 01,
  or say so.
- **Every manifest and every status carries a `host`** (release notes §2.1). Each manifest's
  `host.id` equals `this_host()["id"]`; each status's `host.id` and `host.boot` equal the
  pre-launch reading. Record each.
- **`list`'s summary** gains "*N* with no host recorded". R3's runs are not among the *N*.
- **A `SIGTERM`'d launch's `killed`.** Under `v0.2.0` the handler's `finish` writes through a
  temporary name of its own (release notes §2.3). Record whether any
  `status.json.<pid>-<n>-<hex>.tmp` is left in a killed run's directory: none is expected from a
  handler that returns.
- **The comparison with R2** is through the package. R2's recorded fingerprints carry
  `encoding_format` 1, so `compare_with_recorded.py` applies to them directly (README §6.2 C7).
  `fingerprint_both_ways.py` is not run.

What holds from R2, unchanged:
- the launch command (`runregistry-extraction` README §4.1; verification §4.5);
- **one check after each launch, then end your turn**, the orchestrator waiting on the run's pid
  and resuming you, never polling, never `pgrep -f`;
- the waits end on the pid's death, not on `state`;
- the wait by pid for a killed launch's Ray agents;
- the interrupter's two dummy tests before the first launch, which now include 02's case D;
- interruption points of R3's own, chosen from R3's `reference` `stdout.log` by R2's reasoning, and
  recorded with it;
- the sidecars made as R2's were;
- the Ray addresses (`local`, never `auto`; **`ray stop` is never run**);
- the scripts of `docs/dk-and-rpk-migration/rehearsal/` run as they stand. A script that fails
  where its R2 run did not: stop.

---

## 2. The hard-kill pair

After `resumed-c` has ended, its last Ray process is gone, and L3's readers have run. Under the
same lock. README §4.1 lists the steps; here they are with what each must record. Save every
command and its output in `docs/rrk-v020-adoption/rehearsal/hardkill_03.txt`, in order.

**2.1 `hardkill-a`.**
- **The launch.** A fresh build on `var/datastores/rrk-v020-adoption-03-hardkill.sqlite`, with no
  `--resume`, `--register hardkill-a`, and otherwise `reference`'s flags. It is launched detached,
  with the same `nohup sh -c '…; echo "** driver exit status $?"'` wrapper, so that the `sh` is the
  driver's parent and reaps it.
- **The interrupter.** Right after the launch, the interrupter with **`--signal KILL`**, at the
  point you chose for `resumed-b` in `CALCULATE QUADRATIC SOURCE INTEGRALS`, with the same hold.
- **One check, then end your turn.** Run `set-pid` with the driver's pid.

**2.2 After the kill.** Record each of these:
- **The launch log** ends `** driver exit status 137`.
- **The interrupter's log** says it sent `SIGKILL`, at a stage that matches the regex and a
  percentage at or above the point.
- **`status.json`.** It has `state` `running`, the driver's pid and `exit_code` `null`, and its
  `host` and `boot` are this machine's. Record its heartbeat and its age.
- **The run directory's file set.** Record any leftover `status.json.*.tmp` file. A kill in the
  middle of a status write can leave one, and nothing reads it (release notes §2.3).
- **`kill -0 <pid>`** says no such process.

**2.3 The wait for Ray (U3).**
- Run `ray-processes` and record what it lists. Then wait for each process **by pid**, as for a
  `SIGTERM`'d launch's agents (`[07-a-killed-builds-ray-agents-outlive-the-driver]`). Record each
  process's command and the time it was gone.
- **If any process of this repository's venv is alive ten minutes after the driver's death,
  stop.** Report each pid, its command and its parent. Do not signal it, do not run `ray stop`,
  and do not release the lock: the user decides (README §6.1 U3).
- If none is alive, record how long the last took to go. 04 hands this to A3 v2.

**2.4 The evidence before `end`.**
- `./venv/bin/python -m runregistrykit list`. Its row for `hardkill-a` is `stale`, marked `!!`,
  and its stderr names `python -m runregistrykit end RUN_ID --reason TEXT`.
- `store show` on the `hardkill` primary. There is no sidecar yet (README §6.2 C8), so record
  what it reports, including the runs that name the store.
- The SHA-256, size and mtime of the primary and of every shard.
- The number of directories under `var/runs/`.

**2.5 The refused resume.**
- **The launch.** The same build with `--resume --register hardkill-b`, launched detached as every
  launch is, into `var/runs/rrk-v020-adoption-03-hardkill-b-refused.launch.log`. It runs the
  `--resume` pre-check, which reads the primary `mode=ro`, then calls `begin`, which must raise
  before Ray is imported.
- **One check, then end your turn.** The check is that the `sh` has exited. Do not run `set-pid`
  for this launch: it begins no run.
- **Then record:**
  - the launch log: the `--resume` line, the traceback ending in `RuntimeError: Cannot begin run
    rrk-v020-adoption-03-hardkill-b-<timestamp> with results "<primary>": run <hardkill-a's id> is
    running (stale, pid <pid>) …`, the sentence naming `python -m runregistrykit end RUN_ID
    --reason TEXT`, `No run directory was created`, and a non-zero driver exit status;
  - `var/runs/`'s directory count, equal to 2.4's;
  - every store file's SHA-256, size and mtime, equal to 2.4's;
  - `ray-processes`, which lists nothing.
- **If `begin` accepts** and a run directory appears, that is a stop. The run is a live build under
  the lock: record its id, leave it to run, run `set-pid` with its pid, end your turn, and report.
  Do not kill it.

**2.6 `end`.**
- **The command.** `./venv/bin/python -m runregistrykit end <hardkill-a's id> --reason
  "rrk-v020-adoption 03: SIGKILLed by interrupt_when.py --signal KILL at <stage>; pid <pid> checked
  gone with kill -0 at <time>"`. It must exit 0 and print that the run is now `lost`, verified,
  naming its pid.
- **The record.** Save `status.json` after it. Then check:
  - `state` is `lost` and `exit_code` is `null`;
  - `ended.verified` is `true` and `ended.not_verified_because` is `null`;
  - `ended.evidence` has `host` `this`, `boot` `this`, `pid_answers` `false` and `state`
    `running`;
  - `ended.git_head` is the dispatch `HEAD`;
  - every key the status held before is kept;
  - the manifest's SHA-256 is unchanged.
- **A second `end`** on the same run must be refused, with a non-zero exit, because the run has
  already ended. Its `status.json` is byte-identical before and after.
- **`list` again.** `hardkill-a` is no longer `stale`, and is counted among the finished.

**2.7 The sidecar.** `./venv/bin/python -m runregistrykit store create --purpose "<one line>"`
on the closed `hardkill` primary, as R2's sidecars were made. Then `store show`.

**2.8 `hardkill-b`.**
- **The launch.** `--resume --register hardkill-b`, launched detached into
  `var/runs/rrk-v020-adoption-03-hardkill-b.launch.log`, run to `done`. One check, then `set-pid`,
  then end your turn.
- **At open,** record the prune lines of what `hardkill-a` left unvalidated, as `resumed-b`'s are
  recorded (verification §3.3.2 there).
- **At finish,** record its `status.json`: `state` `done`, `exit_code` `0`, a `fingerprint` with
  `encoding_format` 1, and `fingerprint_sidecar` `written`. This is what `v0.1.0` refused
  (`[a3-v2-a-hard-killed-run-blocks-its-store]`, step 4).

**2.9 The checks, read-only, on the `hardkill` store**, after `hardkill-b` and its Ray processes
are gone:
- `compare_with_recorded.py` against R3's `reference` and `resumed-c` run directories and R2's two:
  every comparison empty;
- `store_soundness.py`, `prune_check.py` and the census, as on `resumed`;
- `PRAGMA integrity_check` on the primary and every shard, opened `mode=ro`.

Then **release the lock** (`release`), as 08's protocol says.

---

## 3. What to verify

**L1–L3 are R2's §2**, on this prompt's two stores and four runs, with these differences:
- **The manifests.** Every `packages` names, in this order:
  - `runregistrykit` 0.2.0 at `0bae3a3c501a8264ee4be5bc24f06422af38720f` (`requested_revision`
    `v0.2.0`);
  - `datastorekit` 0.4.0 at `c8570e849334d09085bbd84876095fc9e22bd005` (`v0.4.0`);
  - `raypoolkit` 0.2.0 at `eaef0ce6…` (`v0.2.0`).

  All are `editable: false`. Every `git_dirty` is `false`, every `git_head` is the dispatch `HEAD`,
  and every `host` is as §1 says. Compare each as JSON with what the pre-launch check printed. This
  covers all six runs.
- **The fingerprints.** Both stores are format 6, `39809dca…`, 21 classes, `encoding_format` 1, and
  each equal to its sidecar's. The package's `compare_fingerprints` between them is empty, and
  each run's `status.json` fingerprint carries `encoding_format`.
- **`compare_with_recorded.py`** on both primaries, against R3's own `reference` and `resumed-c` run
  directories **and R2's two** (Gate): every comparison empty. Save it as
  `fingerprint_against_recorded_03.txt`.
- **What is new in the logs** is compared with R2's same slug. A new line that names
  `runregistrykit` outside a path, a `DeprecationWarning`, or a refusal is a finding to explain.

**L4 — equal to R2.** Against `runregistry-extraction` verification §3.5, as R2 stated it against
R1:
- the statistics block's counts, as counts and as sorted lines;
- the reference stage counts;
- the resumed chain's identities;
- the stage sequence;
- the printed lines and the registry's fields;
- the stage times, beside R2's and not compared.

A difference in L4 with L1–L3 holding is a finding to explain, and to open as a §3 issue.

**The hard-kill route** (H1) is §2, each step against its expectation.

**Ray after the `SIGKILL`** (H2) is §2.3.

**The registry after R3.** `./venv/bin/python -m runregistrykit list` shows §0.2's 45 runs and
R3's six, which are the four launches, `hardkill-a` and `hardkill-b`. The refused launch makes
none. That is 51 in all, with nothing `running` or stale. `hardkill-a` is `lost` and counted as
finished. Save this as `run_records_03.txt`, with the counts by state.

---

## 4. Acceptance

1. **L1–L3 hold**, as §3 says. Both stores are `39809dca…`, format 6, `encoding_format` 1, equal
   to each other, to their sidecars and to R2's recorded fingerprints.
2. **Every manifest's `packages`** names the three packages at their pins, none editable. Every
   `git_dirty` is `false`, and every manifest and status `host` is this machine's.
3. **The hard-kill route holds on the real launcher** (§2):
   - `hardkill-a` dies of signal 9 with its status `running`, and `list` marks it stale, `!!`;
   - the resume is refused at `begin`, naming the dead run and `end`, with no run directory created
     and every store file byte-identical;
   - `end` exits 0, verified; a second `end` is refused;
   - `hardkill-b` ends `done` with its fingerprint recorded and the sidecar `written`;
   - the `hardkill` store is `39809dca…`, equal to `reference`, and passes §2.9's checks.
4. **No Ray process of this repository outlived `hardkill-a`** past §2.3's ten minutes. Each
   process's time to go is recorded.
5. **L4** against R2, or each difference recorded, explained and opened.
6. **`list`** shows 51 runs, nothing `running` or stale.
7. **The lock** was taken before `reference` and set at each launch that began a run. It was held
   through the readers and the hard-kill pair, and released. Every command and exit status is in
   `lock_03.txt`. Afterwards `status` reports no lock and `ray-processes` lists nothing.
8. **The seven roots** pass at the dispatch `HEAD`, at 02's counts.
9. **`docs/rrk-v020-adoption-verification.md` is created**, with these sections:
   - **§1, "R3 — through `runregistrykit` 0.2.0"**, in R2's §3 shape;
   - **§2, "The hard-kill route on the A3 v2 launcher"**, with a subsection for each of §2's steps
     and one for Ray after the kill.
10. **`[a3-v2-a-hard-killed-run-blocks-its-store]` closes**, if 3 holds:
    - its entry moves from the `handover` board's §3 to its §4, with a closing paragraph citing
      this prompt's commit, verification §2, and RunRegistryKit `v0.2.0`'s `end` and `begin`;
    - this board's §3 entry moves to its §4;
    - its row leaves `docs/OPEN_ISSUES.md` §1.22, with the count and date corrected by counting,
      in this prompt's commit.

## 5. Stop conditions — stop and ask

**`runregistry-extraction` 08 §4's conditions apply**, with R2 in place of R1, and:
- **A fingerprint of the four-launch stores differs from `39809dca…`.** Record both fingerprints
  and the package's `compare_fingerprints`, change nothing, relaunch nothing, and stop. The cause
  is the pin's: RunRegistryKit's or DatastoreKit's (README §5 rule 11).
- **The `hardkill` store's fingerprint differs** while the four-launch stores hold. Record it the
  same way and stop. The cause is the hard kill, and is SGK's or DatastoreKit's to find. Do not
  rebuild it.
- A Ray process of this repository outlives the killed driver by ten minutes (§2.3).
- **`begin` accepts the resume of a store a stale run names** (§2.5); **`end` refuses**
  `hardkill-a` with its pid gone; or **`end` accepts** a run whose pid answers. Each is
  RunRegistryKit's behaviour at the tag, against its own README.
- `hardkill-b`'s `finish` records `fingerprint_error` with `hardkill-a` ended.
- A manifest's `packages` or `host` is not as §3 says, or a `git_dirty` is `true`.
- The lock is held (exit 3), a Ray process of this repository is alive at `acquire` (exit 4), or
  `release` refuses on another owner.
- A rehearsal script fails where its R2 run did not.

## 6. What this prompt does not do

- It changes no tracked file but the records of §7.
- It writes no store under `var/` but its three, and **retires none**. R3's three are the user's
  to retire after 04's review, or not (README §3).
- It runs `end` on `hardkill-a` alone, once accepted and once refused. It calls `Run.finish` on no
  run (README §5 rule 15).
- It does not record G1 on RunRegistryKit's board: 04 drafts that record.
- It does not fix what it finds.

## 7. The log, the verification record, and the board

**The files you may change, and nothing else:**
- `docs/rrk-v020-adoption-verification.md` (new): §1 and §2, with a header saying what the
  document is and that 04 adds §3;
- `docs/rrk-v020-adoption/rehearsal/`: saved outputs named with `03`, among them `lock_03.txt`,
  `hardkill_03.txt`, `fingerprint_against_recorded_03.txt` and `run_records_03.txt`;
- the log, this board, the `handover` board's entry for
  `[a3-v2-a-hard-killed-run-blocks-its-store]` (acceptance 10), and `docs/OPEN_ISSUES.md`.

**The log**, at `logs/03-rehearse-through-v0.2.0.md`, in README §5.1's form. It records what R2's
log records, against R2, and the hard-kill pair step by step.

**The board:**
- the §1 row for 03, and items L1–L4 (R3), H1 and H2;
- the §2 rows R3 and HK;
- §3 and §4 for the closed issue;
- `### 5.3 After prompt 03 (<date>)`;
- a §3 entry for anything opened, with `docs/OPEN_ISSUES.md` §1.22 in the same commit, the count
  and date corrected by counting.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
