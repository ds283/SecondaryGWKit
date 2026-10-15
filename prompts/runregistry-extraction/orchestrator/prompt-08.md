# Orchestrator — prompt 08, rehearse through both packages (R2)

Read [`../README.md`](../README.md) first: §0.1, §2.2 (08's charter), §3, §4 (06's and 08's lines:
what 08 repeats of W1), §4.1 (the protocol and the lock), §5 (rules 8–11, 15) and §6.2 (C3, C10,
C11). Read also `CLAUDE.md`'s "Long-running jobs — the run registry", the packages section and "The
shared rehearsal lock", whose rules bind you as much as the agent. Then
[`prompt-04.md`](prompt-04.md), the note for R1, whose protocol, wait and review this note carries
over except where §0 below says otherwise, and the board's §5.7 (R1 and its review), §5.12 (06) and
§5.13 (07).

**You do not write code.** You may:
- run the interrupter's dummy tests, and the rehearsal scripts on a scratch fixture store, in the
  scratchpad;
- run the one wait per launch that §2.1 describes;
- read R2's two stores read-only once the agent's last reader has run, fingerprint them, and run the
  census, `store_soundness.py` and `fingerprint_both_ways.py` on them;
- read R0's and R1's run directories, and hash the sidecars under `var/datastores/`;
- run `tools.rehearsal_lock status` and `ray-processes`, never `acquire`, `set-pid` or `release`
  while the agent holds the lock;
- run RunRegistryKit's `measure_client_registry.py` from its root with its venv, which reads this
  repository through git only;
- run the suites, and a comparison script from the session scratchpad, never committing one;
- fix small residue in a follow-up commit of your own (§4).

**While a build is running you write nothing in the repository**, outside `var/` and the session
scratchpad. A file you create makes the next manifest stamp `git_dirty: true`, which is the
prompt's stop. This note is committed before dispatch for that reason.

**The prompt:** [`08-rehearse-through-both-packages.md`](../08-rehearse-through-both-packages.md)
**Board items:** L1–L4 (R2), G-RRK · **Gates it decides:** R2, RunRegistryKit G2 · **Closes:**
nothing · **Narrows:** nothing · **Opens:** anything found · **Model:** Opus, as the prompt
recommends.

**Gate:**
- 07 landed (`a9ed928`) and was reviewed (`319cad0`). The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches no code (anything outside
  `prompts/` and `docs/`) and no script under `docs/`. If a later commit touches either, re-check
  every fact below that it could move, and hand the agent the corrections as an addendum.
- **Never dispatched concurrently with anything else in this checkout**, and never while any Ray
  instance of this repository is alive or the rehearsal lock is held.

## 0. What makes this prompt unusual

**It is R1 again, through `runregistrykit`.** Between R1's tree (`89ff8b7`) and the dispatch `HEAD`,
what a build runs changed in one place: the registry. `git diff --stat 89ff8b7 319cad0 -- .
':!prompts'` lists `RunRegistry/` deleted, `runregistrykit.toml` and `config/fingerprint.py` added,
the `RunDrivers/` tests, `requirements.txt`'s one line, and twelve `docs/` scripts with their
`RunRegistry` names rewritten. `main.py`, every `extract_*.py`, `run_readers.sh`, every factory and
`StoreFixtures/` are untouched. The launcher's diff since R1 is its two `import runregistrykit` and
`runregistrykit.begin` calls, its printed `list it with:` line, and `prepare`'s `runs_root`, which
the `--build` path does not reach. **So a miss is RunRegistryKit's or DatastoreKit's** (README §4.1,
§5 rule 11): the first suspects are the fingerprinter's `take` and `Run.finish`'s sidecar write.

**It is the first rehearsal whose fingerprints carry `encoding_format`, and the last with a recorded
fingerprint to compare against.** R0's and R1's stores are retired (board §5.8). What R2 is compared
with is their four run directories' recorded fingerprints, field by field, through 06's
`fingerprint_both_ways.py --against` (README §4). `compare_with_recorded.py` is run on R2's own
stores against R2's own run directories only.

**It decides two gates.** R2, and RunRegistryKit's G2, whose evidence is spread over 06, 07 and this
prompt. The board records G2 as holding; 09 drafts the record for RunRegistryKit's board.

**It runs for half an hour or more, and no conversation may own it.** As at 04: the agent launches
each build detached, through the registry, checks once, and ends its turn. You run **one** wait per
launch (§2.1) and resume it when the driver's pid is dead. Four launches, four waits.

**Corrections to the prompt.** The orchestrator checked the prompt against the tree at `319cad0`.
Each correction below governs, and the agent logs it as an `IMPLEMENTATION CHOICE` citing this note,
or `STRUCTURALLY REQUIRED` where the prompt cannot be followed as written.

1. **04's corrections carry over**, as 04's log records them, with R1 in place of R0 and
   `var/runregistry-extraction-08/` in place of `-04/`: `prune_serials.py` is not run, and
   `prune_check.py` runs on R2's own `closed_store_state.py` output; the six scripts of
   `docs/dk-and-rpk-migration/rehearsal/` are the ones run, as 06 left them; the other tools run
   from where R1 ran them (verification §2.1.2, §2.3, §2.4), none copied; nothing is written into
   `docs/runregistry-extraction/` or anywhere else in the repository until `resumed-c` has ended;
   the pre-launch Ray check is the helper's `ray-processes`, exit 0.
2. **The `resumed-a` point.** Size the hold at **about 40 % of R2's own `reference` `GkSource` stage
   time**, rounded to whole seconds, and not below 10 s. R1's stage took 102.9 s (hold 41 s), R0's
   78 s (31 s). Pattern `'^CALCULATE GREENS FUNCTIONS FOR SOURCE REDSHIFT$'`, `--interval 5`.
   - If `resumed-a` stores no `GkSource` row, that is recorded, not a stop: the run was killed in the
     right stage, and the prune clause holds vacuously.
   - If the stage completes during the hold and the build runs on, that is a missed interruption
     and a stop.
3. **The `resumed-b` point.** Pattern `'^CALCULATE QUADRATIC SOURCE INTEGRALS'`, not anchored at its
   end, `--interval 5`, `--hold` about a third of R2's `reference` QSI stage **and at most 250 s**,
   inside the queue's 300 s minimum notify interval (`main.py:119`). R1's stage took 494.9 s (hold
   165 s).
4. **A progress line may or may not appear.** R1's reference and `resumed-c` QSI stages printed one
   each; R0's reference three; `dk-and-rpk-migration` R2's none. Whether R2's does is the 300 s
   threshold and the load, not a finding. Compare any that appear with R1's (verification §2.5.6).
5. **`packages` is compared as JSON, not in order.** The prompt's §2 says every `packages` names the
   three distributions "in this order". Every record is written with `sort_keys`, so on disk the
   keys are alphabetical (`datastorekit`, `raypoolkit`, `runregistrykit`), whatever order the code
   built them in (06's log, deviation 8). Compare each manifest's `packages` with §0's record as
   parsed dictionaries; key order is not a check. `STRUCTURALLY REQUIRED`.
6. **The launcher's hash moved, and one printed line with it.** Every R2 manifest's `script_sha256`
   is `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d`, not R1's `b7935877…`: 06
   rewrote the launcher's registry calls, and 06's review (`7af17bd`) its `--force` message. Its
   banner's `list it with:` line is now
   `**   list it with: ./venv/bin/python -m runregistrykit list`, where R1's said
   `PYTHONPATH=. ./venv/bin/python -m RunRegistry list`, in every `stdout.log` and launch log.
   That line is the one expected new line naming `runregistrykit`; the log comparison lists it and
   cites 06. Any other line naming `runregistrykit`, or `datastorekit` outside a path, is the
   prompt's finding to explain.
7. **`fingerprint_both_ways.py`'s count.** For each store it makes four checks of its own and four
   against each recorded fingerprint: with R0's and R1's four run directories, **20 of 20** per
   store, and a summary line saying every check holds. Run it as the prompt's §2 gives it; its
   `--expect-digest` stays at its default, `39809dca…`.
8. **G2's item 1 is RunRegistryKit's measure, re-run.** Run from RunRegistryKit's root, with its
   venv, after the last launch: `./venv/bin/python docs/adoption/measure_client_registry.py sgk
   --commit <the dispatch HEAD>`, saved whole as `docs/runregistry-extraction/rehearsal/g2_measure_08.txt`.
   It reads this repository through git only, and changes nothing there (README §5 rule 12). It must
   report 0 import statements, 0 string literals and 0 `-m RunRegistry` in live code and in the
   staying tests. The grep `git grep -nP 'RunRegistry(?!Kit)' -- . ':!prompts' ':!docs/*.md'
   ':!docs/**/*.md' ':!docs/**/*.txt'` prints `CLAUDE.md:82` and `config/fingerprint.py:33` (06's
   ruling 1; the second was `:31` at 06, moved by 07's docstring bullet) and nothing else.
9. **The verification document's §3** has §2's subsections §3.1–§3.9 in the same order, §3.1.3 the
   lock; **§3.10 is W1 across stores** and **§3.11 is G2**, item by item with its evidence. The
   header's list already names §3 as R2's; it is left as it is.
10. **The board's header** says 9 of 10 landed once the agent commits; `prompts/INDEX.md` is the
    orchestrator's.

**What moves on purpose:**
- Added: `docs/runregistry-extraction/rehearsal/*_08*` outputs, `lock_08.txt`,
  `fingerprint_both_ways_08.txt`, `run_records_08.txt` and `g2_measure_08.txt` among them; the log.
- Modified: `docs/runregistry-extraction-verification.md` (§3 appended; §1, §2 and the header not
  edited); the board; `docs/OPEN_ISSUES.md` only if an issue is opened.
- **README does not move**, and neither does `prompts/INDEX.md` (the orchestrator's, §4). No script
  anywhere changes, and no code or test.
- Under `var/`: R2's two stores and their sidecars, four run directories and their launch logs, and
  `var/runregistry-extraction-08/`. **Nothing else under `var/` changes**: the 18 tombstones and every
  earlier run directory included.
- Under `~/ScienceData/`: `.rehearsal-lock` created and removed again.
- The index is 106, and moves only if an issue is opened.

**The facts, checked by the orchestrator on 2026-10-15 at `319cad0`.** Pass them on.

- **The tree and the machine.** Branch `handover-remedial`, `git status --short` empty, this
  checkout alone in `git worktree list`. `git log a9ed928..319cad0` is `319cad0` alone, which touches
  `prompts/` only. The code's last change is `a9ed928`; the last change to a `docs/` script is
  `7af17bd`. `RAY_ADDRESS` unset. 10 CPUs. The one-minute load was 15 at 23:46 and 5 at 23:52.
  **38 GB free on `var/`'s volume (96 % used)**, against 74 GB at R1; a rehearsal's stores are about
  140 MB each.
- **The lock and Ray.** `tools.rehearsal_lock status`: `lock: free`, exit 0. `ray-processes`: none,
  exit 0. `~/ScienceData/` holds `ChamPBH-rehearsals/` alone.
- **The registry.** `./venv/bin/python -m runregistrykit list`: 39 runs, 37 finished, 2 `unknown`
  (the `a3-pilot` pair); none `running` or stale. The newest two are 06's worktree runs.
- **The venv, and what every manifest should hold.** `pip check` clean.
  `runregistrykit.package_provenance(("runregistrykit", "datastorekit", "raypoolkit"))` gives
  `{"runregistrykit": {"version": "0.1.0", "url": "https://github.com/ds283/RunRegistryKit",
  "commit": "9f8ebe431503fb8c79ee1407cc460501f1ca2d92", "requested_revision": "v0.1.0", "editable":
  false}, "datastorekit": {"version": "0.4.0", "url": "https://github.com/ds283/DatastoreKit",
  "commit": "c8570e849334d09085bbd84876095fc9e22bd005", "requested_revision": "v0.4.0", "editable":
  false}, "raypoolkit": {"version": "0.2.0", "url": "https://github.com/ds283/RayPoolKit", "commit":
  "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "requested_revision": "v0.2.0", "editable": false}}`.
  06's worktree-check manifest's `packages` equals it as JSON. `datastorekit`'s and `raypoolkit`'s
  entries are R1's exactly.
- **The config.** `runregistrykit.find_client_root()` from the root: `config_file`
  `runregistrykit.toml`, `runs_root` `var/runs`, `stores_root` `var/datastores`, `distributions`
  `("datastorekit", "raypoolkit")`, `stores` `{"fingerprinter": "config.fingerprint:fingerprinter"}`.
- **The records' fields.** A runregistrykit manifest has R1's nineteen keys, no more and no fewer
  (06's worktree check against R1's `reference`). R1's `status.json` keys are `exit_code`,
  `fingerprint`, `fingerprint_sidecar`, `heartbeat`, `pid`, `stage`, `state`, `units_done`,
  `units_total`; R2's should be the same, its `fingerprint` gaining `encoding_format`.
- **The names are free.** `runregistrykit store show` on `var/datastores/runregistry-extraction-08-reference.sqlite`
  and `…-resumed.sqlite`: kind `absent`, problems none, no runs naming either, exit 0. Nothing under
  `var/runs/` contains `runregistry-extraction-08`; `var/runregistry-extraction-08/` does not exist.
- **What is under `var/datastores/`**: 18 files, all tombstone sidecars (one under
  `backup-pre-resume-20260921T091011/`), no `.sqlite`. `find var/datastores -type f -print0 | sort -z
  | xargs -0 shasum -a 256 | shasum -a 256` is `1fbaf5b766d2a006…`. The agent records the full list
  with SHA-256, size and mtime before its first launch and after its last step: the 18 unchanged, and
  the only new files R2's ten store files and two sidecars.
- **The recorded fingerprints to compare against.** `status.json` of
  `var/runs/runregistry-extraction-02-reference-20261012T091921`, `…-02-resumed-c-20261012T101145`,
  `…-04-reference-20261014T154651` and `…-04-resumed-c-20261014T161739`: each `done`/0, format 6,
  `39809dcac5c7b231…`, 21 classes, problems `{}`, **no `encoding_format`**, and all four equal once
  `taken` is set aside.
- **The scripts, run at `319cad0`.**
  - `stage_counts.py` on R1's four run directories reproduces `stage_counts_04.txt` byte for byte;
    `policy_statistics.py` on R1's `reference`, `resumed-b` and `resumed-c` reproduces each of the
    three `policy_statistics_04_*.txt`. (It takes the run directory, not the log.)
  - On two scratch fixture stores (`StoreFixtures.real_store_fixtures.build_full_store`, each given a
    sidecar by `runregistrykit store create` and `store fingerprint --write`):
    `closed_store_state.py`, `replicated_serial_census.py` (`=> no divergence`), `compare_stores.py`
    (`verdict: equal`), `store_soundness.py` (`=> clean`), and `compare_with_recorded.py` against a
    scratch run directory holding the package's fingerprint (0 differences, `verdict: equal`), each
    exit 0.
  - `fingerprint_both_ways.py` on the fixture store against a scratch run directory holding the
    `09c` golden, `--expect-digest 6576b85d…`: exit 0, 8 of 8. Given R1's `reference` run directory
    as well, it reads that record and fails, as it must on a different store: the digest and 98
    differing paths, 10 of 12.
  - So a script that fails in the agent's run fails on the build's store, not on the swap.
- **The interrupter passes at `319cad0`.** Both dummy tests, into the scratchpad, exit 0, 40 and 20
  lines, each equal to R1's once times, pids and paths are stripped. The agent runs them again.
- **G2's item 1, measured.** RunRegistryKit's `measure_client_registry.py sgk --commit 319cad0`,
  from its root (`caad864`, clean): exit 0; 0 import statements, 0 names, 0 calls bound, 0 string
  literals, 0 `-m RunRegistry`, in live code and in the staying tests. The grep of correction 8 gives
  its two lines.
- **What R1's logs held**, against which the log comparison is made (verification §2.2.5): each
  killed `stderr.log` the handler's two lines (`SystemExit(143)`, then `SystemExit(1)`); the SciPy
  `IntegrationWarning` in `reference` and `resumed-c`; the prune's integrity warnings in `resumed-b`;
  one progress line in each of `reference`'s and `resumed-c`'s QSI stages. No line names
  `datastorekit` in any R1 build log but through paths. Correction 6's line is the one expected
  difference.
- **The stage times.** R1's reference (verification §2.5.5), at load 11–100: `GkSource` **102.9 s**,
  QSI **494.9 s**, the build 13 m 57 s. R0's: 78 s, 1230.1 s, 29 m 19 s.
- **The commands are R1's**, verbatim in verification §2.1, §2.1.2, §2.1.3 and §2.3–§2.4, with
  `runregistry-extraction-08`, `--campaign runregistry-extraction --prompt 08`, the store names of
  the prompt's §1, and every registry command `./venv/bin/python -m runregistrykit …` as 06's log
  §7.8 states them.
- **No Ray process left.** `ray-processes` is the test. A killed build's agents are waited for **by
  pid**, never by pattern, and how long they took is recorded beside R1's (none outlived either
  killed driver).
- **Shell traps.**
  - `stat` here is GNU coreutils: use `/usr/bin/stat -f '%m %N'` or Python's `os.stat`.
  - `grep` may wrap `ugrep`: use `git grep` or Python for any regex the log relies on.
  - The shell is `zsh`: an unquoted `$var` holding spaces is **not** split into words (use
    `${=var}` or an array), and a bare word beginning `=` is command expansion. Quote it.
- **Expected counts**, at `319cad0`, serially, load 17 falling to 5 (one-minute):

  | Root | Tests | Result | Wall |
  |---|---|---|---|
  | AdaptiveLevin | 32 | OK | 1 s |
  | CosmologyModels | 39 | OK | 3 s |
  | ObjectFactories | 279 | OK | 14 s |
  | tools | 91 | OK | 11 s |
  | LiouvilleGreen | 148 | OK (1 skipped) | 24 s |
  | RunDrivers | 99 | OK | 3 s |
  | ComputeTargets | 693 | OK | 138 s |

  Every count is 07's. No known flake fired. A failure of one of README §8's four is re-run alone,
  not a stop.
- **The index:** 106.

**What the review exists to establish.** All else is mechanics.
- **(R1) Every launch ended as recorded.** `done`/0 and `killed`/143, each with a fingerprint
  carrying `encoding_format`, each driver's exit status in its launch log, and no Ray process of this
  repository left.
- **(R2) R2 is R1.** Both stores equal each other, `39809dca…`, and their sidecars; field by field,
  `encoding_format` and `taken` aside, they equal the four fingerprints R0 and R1 recorded. Nothing
  stored was recomputed; the prune removed exactly the unvalidated rows, if there were any.
- **(R3) The stores are sound**, by `store_soundness.py` and the census.
- **(R4) Every reader works, and none touches the store.**
- **(R5) The manifests say what ran.** Every `packages` is §0's record as JSON, every `git_dirty`
  `false`, every `git_head` the dispatch `HEAD`, and every `script_sha256` `2a985303…`.
- **(R6) L4 equals R1's**: the statistics counts, the reference stage counts, the chain's
  identities, the stage sequence, the printed lines and registry fields. Every log difference from
  R1 is listed and explained.
- **(R7) W1 across stores.** `fingerprint_both_ways.py` exits 0, 20 of 20 on each store, the stores
  untouched by it.
- **(R8) The lock.** Taken before `reference`, `pid` set at each launch, held through the readers,
  released; free now.
- **(R9) Nothing else under `var/` moved.** The 18 tombstones keep their hashes; `list` shows 43.
- **(R10) G2.** Each of RunRegistryKit's checklist item 9's six clauses has its evidence, named.

**Conventions.** 04's, carried. All bind the agent, and the brief says so.
- **Staging and reporting.** Stage by explicit path only, and show `git diff --cached --name-only`
  before committing. Count, then paste. A log states only what the record shows.
- **The SHA.** The agent writes "this commit", and the orchestrator substitutes the SHA in its
  follow-up.
- **Suites.** README §8's command, output to a file in the agent's scratch directory, the verdict
  grepped from it. Each **in the foreground**, serially, within the turn, and **only after the last
  launch has ended and the lock is released**. Under load, the four tests of README §8 are known to
  fail: re-run a failure alone before calling it real.
- **Scratch.** The agent's own subdirectory of the session scratchpad. Anything that must outlive
  the session, the working notes and between-launch captures included, goes under
  `var/runregistry-extraction-08/` (`CLAUDE.md` rule 5). No scratch file goes anywhere else in the
  repository, even for a moment, while a build may yet be launched.
- **The venv.** The agent installs and uninstalls nothing.
- **DatastoreKit, RunRegistryKit and RayPoolKit.** Read only; nothing changed there. The one thing
  run in another repository is correction 8's measure, from RunRegistryKit's root with its own
  venv (README §5 rule 12).
- **The lock.** Every helper command and its exit status in `lock_08.txt`. Never wait for, retry or
  remove a lock; exit 3 or 4 at `acquire`, or a refusal naming another owner, is a stop.
- **Ray addresses.** The driver forces `address="local"`; `run_readers.sh` passes `local`. Nothing
  runs `main.py` or an `extract_*.py` at its default. **`ray stop` is never run.**
- **Never touch a process you did not start.**
- **Stop conditions are real.** A stop leaves the stores, the run directories and the tree as they
  are: no relaunch, no deletion, no fix. It goes to the user through the orchestrator, and the
  agent resumes with `SendMessage`. If it stops while holding the lock, it releases once no Ray
  process of this repository is alive, or says it cannot.
- **Usage limits.** A usage stop does not stop a detached build. Resume the agent with
  `SendMessage`, saying exactly where it stopped. If it commits but sends no report, review from the
  commit, its log and its outputs.

## 1. Before you dispatch

Done on 2026-10-15 at `319cad0`, before this note was committed. Repeat steps 1–3 at the dispatch
`HEAD` if anything lands in between.

1. **The tree.** Branch and `HEAD`; `git status --short` and `git diff --cached` empty; `git worktree
   list` this checkout alone; `git log --oneline a9ed928..HEAD -- . ':!prompts' ':!docs'` empty, and
   no `docs/` script changed.
2. **The registry.** Nothing `running` or stale; 39 runs.
3. **The lock and Ray.** `status`: free; `ray-processes`: none. `RAY_ADDRESS` unset. Record the load,
   and say so if `var/` has less than 10 GB free.
4. **The venv.** `package_provenance()` gives §0's record; `pip check` is clean.
5. **Baselines**, as §0's table.
6. **The index.** `grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives 106.
7. **The names are free**, and `var/datastores/` holds the 18 files, hashing as §0.
8. **The interrupter.** Both dummy tests at `HEAD`, into the scratchpad. Both pass.
9. **The scripts and the fixture prototype**, as §0; the four recorded fingerprints read.
10. **G2's measure**, as §0.
11. **The `var/` marker**: immediately before dispatch, `touch` a marker in the scratchpad.

## 2. Dispatch

One fresh-context subagent, on **Opus**. Give it:
- the prompt, and through it 04, 02 and `dk-and-rpk-migration`'s 08;
- the campaign README and this board;
- `CLAUDE.md`'s three sections named above;
- 04's log, 06's log (§7.8 and its hand-off) and 07's hand-off, and verification §2;
- `HEAD`, the baselines and the index count;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after every launch, the readers, the lock's release, W1, G2's
  measure and the suites. Its log is `logs/08-rehearse-through-both-packages.md`, in README §5.1's
  form, with what 04's log records, against R1.
- **The hand-off.**
  - After each launch: one check, including `set-pid`, then a report giving the run id, the Python
    pid, the `sh` pid, the interrupter's pid and arguments if there is one, and the lock's `status`,
    then the end of its turn.
  - You resume it with the run's final `status.json` (less the fingerprint's per-class detail), the
    last lines of its launch log, and the wait's verdict.
  - It never waits, polls or sleeps on a build itself. Waiting by pid for a killed build's orphaned
    Ray agents, before the next launch or before `release`, is the one bounded wait it runs.
- **The files it may change** are the prompt's §6 list, nothing else. **It touches none of them, and
  creates nothing in the repository outside `var/`, until the last build launch has ended**, so
  that every manifest stamps `git_dirty: false`.
- **It retires no store.** R2's two are the user's, after 09's review (README §3).
- **Stop and ask** on any of the prompt's §4 conditions. A fingerprint miss, or a `fingerprint_both_ways`
  difference other than `encoding_format` and `taken`, is the one R2 exists for: record, change
  nothing, stop. **A statistics count that differs, with the fingerprint holding, is not a stop**:
  it is a finding, recorded, explained if it can be, and opened as an issue.

### 2.1 The one wait per launch

When the agent reports a launch, run **one** background command (`run_in_background`) that ends
when that run's driver pid is dead, or after the cap. It ends on the pid's death, **not** on
`state == "finished"`: a killed run's `status.json` says `killed` 5–6 s before its fingerprint and
final heartbeat are written (board §5.7).

```bash
./venv/bin/python - <<'EOF'
import json, time
import runregistrykit
run_dir, pid = "var/runs/<run id>", <python pid>
status = runregistrykit.read_json(f"{run_dir}/status.json") or {}
assert status.get("pid") == pid, ("status.json names another pid", status.get("pid"))
start, cap, seen_stale, capped = time.time(), 2 * 3600, False, False
while runregistrykit.pid_alive(pid):
    status = runregistrykit.read_json(f"{run_dir}/status.json") or {}
    seen_stale = seen_stale or runregistrykit.liveness(status, 900) == "stale"
    if time.time() - start > cap:
        capped = True
        break
    time.sleep(30)
if not capped:
    time.sleep(10)  # the sh wrapper's exit-status line
status = runregistrykit.read_json(f"{run_dir}/status.json") or {}
fp = status.get("fingerprint") or {}
brief = {k: v for k, v in status.items() if k != "fingerprint"}
brief["fingerprint"] = {k: fp.get(k) for k in ("fingerprint_format", "encoding_format", "digest", "problems")}
brief["fingerprint"]["classes"] = len(fp.get("classes") or {})
print("CAP" if capped else "ENDED", status.get("state"), json.dumps(brief))
print("heartbeat went stale at least once" if seen_stale else "heartbeat never stale")
EOF
```

- It identifies the run by its pid and heartbeat, never by a command-line match (`CLAUDE.md`
  rule 2). It exits when the driver exits. On `CAP` it leaves the build running.
- When it ends, send the agent its printed lines and the launch log's last lines, and nothing else.
  - On `ENDED`, it carries on.
  - On `CAP`, the two-hour stop is met. Tell the agent, and ask the user before anything else
    happens.
- Do not run a second wait while one is alive, and do not check on the build between its start and
  its end.

## 3. The review — sixteen checks

1. **Scope.**
   - `git diff --name-status HEAD~1 HEAD`: `A` for the log and the `*_08*` outputs; `M` for the
     verification document, the board and, if an issue was opened, `docs/OPEN_ISSUES.md`. README,
     `prompts/INDEX.md` and `orchestrator/` are not modified. No code, test, tool or script.
   - `git diff HEAD~1 HEAD -- docs/runregistry-extraction-verification.md` adds §3 and nothing else:
     §1, §2 and the header are byte-identical.
2. **(R1) The runs.** Read every manifest, `status.json`, `stderr.log`, interrupter log and launch
   log under `var/runs/runregistry-extraction-08-*` yourself.
   - `reference` and `resumed-c` are `done`/0; `resumed-a` and `resumed-b` are `killed`/143, each
     with a `fingerprint` carrying `encoding_format` 1 and no `fingerprint_error`.
   - Each launch log ends `** driver exit status` 0 or 143, matching.
   - Each killed `stderr.log` has the handler's lines and no traceback.
   - Each interrupter log shows one `SIGTERM`, with the pid and stage it acted on. The holds are
     those corrections 2 and 3 give, computed from R2's own reference.
   - The next launch's counts show the killed stage had stored some rows and not all, or, for
     `resumed-a`, none, as the log says.
   - No process the agent started is alive, and `ray-processes` lists nothing.
3. **(R5) The manifests.** For each: `git_head` the dispatch `HEAD`, `git_dirty` `false`; `packages`
   §0's record, compared as JSON; `script_sha256` `2a985303…` in all four; `argv` holds `--build
   --rehearsal`, with `--resume` on `resumed-b` and `-c` only; `scope` begins `REHEARSAL build, not
   A3 v2:`; `results` names the right store; each `--purpose` names R2 of this campaign, the tree's
   commit, `runregistrykit` 0.1.0 at `9f8ebe4`, `datastorekit` 0.4.0 at `c8570e8`, and `39809dca`.
4. **(R2) Nothing stored was recomputed.** Run `stage_counts.py` on the four run directories
   yourself, and compare it with the log's table and with `stage_counts_04.txt`.
   - The `reference` column equals R1's, stage by stage.
   - Every stage completed before an interruption computes 0 on the next launch.
   - `resumed-b`'s `GkSource` = 291 − (stored by `resumed-a`) + (pruned); `resumed-c`'s QSI = 864 −
     (stored by `resumed-b`), and nothing else. The totals per class over the chain equal the
     reference's.
5. **(R2) The prune.** Run `prune_check.py` on the agent's saved `closed_store_state.py` output after
   `resumed-a`: every listed serial absent now (or reused by a later row), read `mode=ro`.
   `read_inventory` reports no `duplicate`. `resumed-b`'s integrity warnings count the same rows,
   Ray's folding included. If `resumed-a` stored none, the clause holds vacuously, and the log says
   so.
6. **(R2) The fingerprints.** Fingerprint both closed stores yourself, read-only:
   - `compare_stores.py <reference> <resumed>`: equal, each equal to its sidecar;
   - `compare_with_recorded.py` on both primaries `--against` R2's own `reference` and `resumed-c`
     run directories: 0 differences each;
   - `runregistrykit store fingerprint` on each: format 6, `encoding_format` 1, `39809dca…`, 21
     classes, `recorded: matches`.
7. **(R3) Integrity.** Run `store_soundness.py` and `replicated_serial_census.py` on both primaries
   yourself: `integrity_check` `ok`, `foreign_key_check` empty, one `version` row per shard (label
   `2025.1.1`), no census divergence, no inventory problem, no journal file.
8. **(R4) The readers.** Each reader's log under `var/runregistry-extraction-08/extract/`: exit 0, no
   `ReadOnlyMiss`, `ReadOnlyWrite`, `ModuleNotFoundError`, unpickling error or traceback; hashes and
   mtimes equal around each; the `nomodels extract_Gk_data` run printed the skip line. Each log
   compared with R1's same reader as line kinds. Run one reader yourself only if the lock is free
   and you take it for the run, and release it after.
9. **(R6) L4.**
   - Run `policy_statistics.py` on R2's `reference`, `resumed-b` and `resumed-c` yourself: 14 lines
     each; the counts R1's (291; 109 / 26 (25 complete, 1 minimal) / 156); against R1's same slug,
     equal line for line or equal sorted, as the log says.
   - The `** ` banner lines of each run, in order, equal R1's same slug (numbers and the campaign
     prefix stripped, correction 6's line aside).
   - The resumed-chain identities of check 4 are in the log as a table beside R1's verification
     §2.5.3.
10. **(R6) The printed lines and the log differences.** For each slug, against R1's: the counts of
    `ALL WORK ITEMS COMPLETE in time`, `Queue summary:` and `(no compute items were evaluated`
    lines, each form equal with numbers stripped; any progress line in correction 4's terms; each
    final `status.json` `stage`. Then R2's `stderr.log`, launch log and `stdout.log` against R1's as
    line kinds, times, pids, ports, paths, numbers and the campaign prefix stripped: every kind in
    one and not the other is in the log with an explanation that holds, correction 6's line among
    them. Search all twelve build logs and the reader logs yourself for `runregistrykit`,
    `datastorekit` outside paths, `DeprecationWarning`, `ConfigError`, `Traceback`,
    `ModuleNotFoundError`.
11. **(R7) W1 across stores, run here.** `fingerprint_both_ways.py` with the prompt's §2 command:
    exit 0, 20 of 20 on each store, output equal to `fingerprint_both_ways_08.txt` but for times;
    every store file byte-identical around it.
12. **(R8) The lock.** `lock_08.txt`: `acquire` exit 0 before `reference`'s manifest was created;
    `set-pid` exit 0 at each launch, with that run's driver pid; `release` exit 0 after the last
    reader's log ends, and any refusal before it explained. Now `status`: free; `ray-processes`:
    none. `~/ScienceData/` holds what it held before dispatch.
13. **(R9) `var/`.** The file list under `var/datastores/`, with hashes, before the agent's first
    launch (its record) and now: the 18 unchanged, the only additions R2's twelve. A3 v2's guard
    refused and changed nothing. `find var -newer <marker>` lists only R2's runs, launch logs,
    stores, sidecars and working directory. `runregistrykit list`: 43 runs, 41 finished and 2
    `unknown`, nothing `running`; `run_records_08.txt` agrees.
14. **(R10) G2.** `g2_measure_08.txt` is correction 8's command at the dispatch `HEAD`, all zeros;
    re-run it here at the agent's commit. Verification §3.11 takes item 9's six clauses in order,
    each with evidence that is in a committed record (06's `measure/out/06-*`, 07's and this
    prompt's roots, this prompt's manifests and fingerprints), not asserted. The board's §2 G2 row
    says holding, naming 06 and 08.
15. **The records.** The board: the header (9 landed), the §1 row for 08, L1–L4 (R2) and G-RRK ✅,
    the §2 rows R2 (naming the two runs that finished `done`) and RunRegistryKit G2,
    `### 5.14 After prompt 08`; §3 entries and the index for anything opened, the count right by
    counting. Verification §3 has §3.1–§3.11 as correction 9 says, every command verbatim, and its
    numbers match the run directories. The log classifies §0's ten corrections.
16. **The suites.** All seven roots, serially, at the commit, every count as §0.

## 4. After the review

- **Record it on the board** as an *Orchestrator review of prompt 08* paragraph in §5.14, in 04's
  form. Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA, on the board and in the log's header;
  - the board's §2 rows, if the agent left either ⬜ with every part holding;
  - this note added to the board's list of notes, marked "used for 08";
  - `prompts/INDEX.md`'s row: 9 of 10 landed, R2 and G2 hold.
- Anything the review finds that is not residue is opened on the board's §3 and in the index.
- **Report to the user:** R1–R10, each in a line; the fingerprints against `39809dca…` and R0's and
  R1's four records; the statistics counts against R1's; where each interruption fell and what the
  resume pruned and recomputed; every log difference from R1 and its explanation; the lock; G2;
  anything opened.
- **Then 09**, the close. Its prompt is written after this review (README §2.1). R2's two stores
  stay live until the user decides, after 09's review, whether to retire them (README §3).
