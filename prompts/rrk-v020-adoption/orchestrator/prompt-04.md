# Orchestrator — prompt 04, close the campaign

Read [`../README.md`](../README.md) first: §1, §2.2 (04's charter and its amendment of 2026-10-19),
§3, §5, §6.1 (**U4**) and §6.3. Then the board's §2, §3 and §5.3 with the *Orchestrator review of
prompt 03*. `runregistry-extraction`'s
[`orchestrator/prompt-09.md`](../../runregistry-extraction/orchestrator/prompt-09.md) is this note's
model for a close that drafts a gate for RunRegistryKit and states what A3 v2 inherits; what
differs is said here. The conventions are [`prompt-01.md`](prompt-01.md) §0's, carried over.

**You do not write code.** You may:
- run the suites, `python -m runregistrykit list` and `store show`, `tools.rehearsal_lock status`
  and `ray-processes`;
- read RunRegistryKit with `git`, never changing, fetching or committing there;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`04-close-the-campaign.md`](../04-close-the-campaign.md)
**Board items:** Z1–Z5 · **Closes:** nothing · **Narrows:**
`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` · **Opens:** nothing expected ·
**Model:** Sonnet, as the prompt recommends.

**Gate:**
- 03 landed (`17916cb`) and was reviewed (`ebb70a3`). R3 and HK hold on the board's §2. The tree is
  clean.
- `HEAD` is the commit that adds the prompt and this note, or a later one that touches nothing
  outside `prompts/`.
- Nothing is `running` or stale in the registry; the lock is free; no Ray process of this
  repository is up. 04 starts no Ray, but the suites must not share the machine with a build.

## 0. What makes this prompt unusual

**It is records, a draft for another repository, and one rule of `CLAUDE.md`.** No code changes.
What the review must establish is that every record is true, by counting and by reading
RunRegistryKit's board at its commit, and that nothing is claimed as done that only the user can
do: the G1 commit is RunRegistryKit's, and the stores are the user's.

**One part was added after the charter was written.** At 03's review the GCS server deadlock
(`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`) was found to leave the rehearsal lock
unreleasable after a hard kill. On 2026-10-19 the user ruled (**U4**) that `CLAUDE.md` rule 5 gains
the step, rather than only the verification record. The README's §2.2 amendment and §6.1 record
it. The rule's wording is the agent's; its substance is the user's, and the review checks that it
is not weakened: a **person** ends a listed GCS server, with `SIGKILL`, **by its pid**, never by
pattern and never with `ray stop`.

**Rulings on the prompt** (the orchestrator's, made in writing this note; the user may reverse any).
Pass them on; the agent logs each as the classification given.
1. **The G1 date is 2026-10-18**, the day `0a27b0c` moved the pin in a commit of its own: that is
   when the gate's condition held. The cell says the campaign closed at 04, as "this commit"; the
   orchestrator substitutes the SHA in its follow-up, in the draft too, before the user pastes it.
   `IMPLEMENTATION CHOICE`.
2. **The draft names the runs** by their ids, from the run directories: R3's
   `rrk-v020-adoption-03-reference-20261018T232838` and `-resumed-c-20261018T233941` (`done`,
   `39809dca…`), and the route's `-hardkill-a-20261018T234619` (`lost`, verified) and
   `-hardkill-b-20261019T001145` (`done`, `fingerprint_sidecar` `written`). `IMPLEMENTATION CHOICE`.
3. **Z3's index corrections rewrite the present-tense clause, not the history.** In §1.11 and
   §1.17 the clause "where it stays open (`docs/OPEN_ISSUES.md` §1.2 there)" becomes where and when
   it was closed there, released in `v0.2.0`, and adopted by SGK at `0a27b0c`. "Its row is deleted
   here; the board's §4 keeps the entry as the record" stays. In §1.21's lead-in, "each still open
   in RunRegistryKit (§1.11, §1.14, §1.17)" says which two are still open (§1.14) and which two
   closed (§1.11, §1.17). Each is a correction of a statement now false, as README §6.2 C4 allows
   the index (README §1, "One exception"). `IMPLEMENTATION CHOICE`.
4. **The two boards' added lines go after the `Transferred (2026-10-16)` paragraph**, inside the
   entry: `prompts/run-registry/IMPLEMENTATION_STATE.md:377` begins that paragraph, a top-level
   paragraph of the entry; `prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md:1685` begins it as a
   list item, so the new line is a list item at the same depth. Each is one line in the sense of
   one paragraph, wrapped at the board's width. The two entries' difference from RunRegistryKit's
   change is real for the machine issue: SGK's entry asked for `machine` in the manifest; the
   package records a `host` with an `id` and a `boot`, and `foreign` liveness. Say so in a clause.
   For the heartbeat issue RunRegistryKit already corrected the next step on hand-over (the entry
   says so). `IMPLEMENTATION CHOICE`.
5. **`prompts/INDEX.md`'s header** reads "27 campaigns: 2 live, 1 planned, 1 seeded, 23 closed or
   complete". By its rows, the "1 planned" is this campaign (its row says **in progress**); `handover`
   and `datastore-readback` are the two live. After 04: "27 campaigns: 2 live, 1 seeded, 24 closed
   or complete". The row's "Opened → last board update" cell is `2026-10-19` alone, which is residue
   from 03's follow-up; it becomes `2026-10-18 → 2026-10-19`. `IMPLEMENTATION CHOICE`.
6. **The index header's other clauses are not recounted.** "Across twenty-two campaigns" and "Two
   of these hold no open issue" do not reproduce by counting (03's log, observation 5; 03's review).
   That reconciliation is not this campaign's; 04 changes only the date, and the count if it moved,
   which it must not. `IMPLEMENTATION CHOICE`.
7. **The GCS issue's narrowing** does not assign it. Whether it goes to `handover` (whose launcher a
   parent-death watch would change) is the user's (README §6.3). The board entry keeps its text
   and gains the **Narrowed** line; the index hook's "**Next step: 04**, which states what A3 v2
   inherits." becomes what is now true, keeping the hook to a few lines. `IMPLEMENTATION CHOICE`.
8. **R3's three sidecars are live stores**, `kind: registry`, which `store show` reads with their
   fingerprint. 04-store-show.txt is 24 blocks: the 21 tombstones, which here equal 01's
   after-blocks line for line (orchestrator's check below), and R3's three. `find var/datastores
   -name '*.manifest.json'` gives 24. `IMPLEMENTATION CHOICE`.
9. **A known wall-clock test failing under load is recorded, not a stop.** If one of README §8's
   four fails, re-run it alone; if it still fails, record its assertion, the one-minute load and
   the heaviest process (`ps -Ao pcpu,comm -r | head`), and carry on: 04 changes no code a test
   reads, so such a failure cannot be 04's. Any other failure, or a count that moves, is §4's stop.
   `IMPLEMENTATION CHOICE`.

**What moves on purpose:**
- **Added:** `measure/out/04-{roots,pip,list,store-show,disk}.txt`; `logs/04-close-the-campaign.md`.
- **Modified:** `docs/rrk-v020-adoption-verification.md` (§3 appended, nothing above it changed);
  `CLAUDE.md` (rule 5); `docs/OPEN_ISSUES.md`; `prompts/run-registry/IMPLEMENTATION_STATE.md` and
  `prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md` (one paragraph each); this board; README §2's
  table (row 04 only); `prompts/INDEX.md`.
- **Not modified:** any code, test, script, `requirements.txt`, `runregistrykit.toml`, `config/`,
  `orchestrator/`, the `handover` board, every other campaign's board; anything in RunRegistryKit,
  DatastoreKit or RayPoolKit; anything under `var/`.
- **The index stays at 100.** One hook changes (§1.22); three lead-ins change (§1.11, §1.17,
  §1.21) and one gains a sentence (§1.22); the date stays 2026-10-19 unless the agent's day differs.

**The facts, checked by the orchestrator on 2026-10-19 at `ebb70a3`.** Pass them on.
- **The tree and the machine.** Branch `handover-remedial`; `git status --short` empty; one
  worktree. `git log 17916cb..ebb70a3` is `ebb70a3` alone, which touches `prompts/` only.
  `RAY_ADDRESS` unset. Load **22–40** (one-minute and five-minute; a Time Machine backup) from other work on the machine
  during the baselines; the lock was free. `df -h /`: **927G, 859G used, 69G free, 93 %**.
- **The lock and Ray.** `tools.rehearsal_lock status`: `lock: free`, exit 0. `ray-processes`:
  none, exit 0. `~/ScienceData/` holds `ChamPBH-rehearsals/` alone.
- **The registry.** `list`: exit 0, stderr empty, last line `51 run(s) under …/var/runs: 49
  finished, 2 unknown; 45 with no host recorded.` None `running` or stale.
- **The venv.** `pip freeze` (101 lines) equals `01-pip.txt`'s after-install block (lines 217–318,
  blank lines dropped) exactly. `pip check`: "No broken requirements found."
  `package_provenance(("runregistrykit", "datastorekit", "raypoolkit"))` is
  `{"datastorekit": {"commit": "c8570e849334d09085bbd84876095fc9e22bd005", "editable": false, "requested_revision": "v0.4.0", "url": "https://github.com/ds283/DatastoreKit", "version": "0.4.0"}, "raypoolkit": {"commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RayPoolKit", "version": "0.2.0"}, "runregistrykit": {"commit": "0bae3a3c501a8264ee4be5bc24f06422af38720f", "editable": false, "requested_revision": "v0.2.0", "url": "https://github.com/ds283/RunRegistryKit", "version": "0.2.0"}}`,
  verification §1's record. `TERMINAL_STATES` `('done', 'failed', 'killed', 'lost')`;
  `this_host()` `id` `8d2b247824882486`, `boot` `376461AF-4985-4F42-8E41-F4E914F5E3E6`: no reboot
  since 01.
- **The stores.** `var/datastores/` holds 39 files: the 21 tombstone sidecars, and R3's three stores
  (primary and four shards each) with their three sidecars. `store show` on each of the 21
  tombstones equals its `01-store-show.txt` after-block line for line (the `exit=` lines aside);
  R3's three read `kind: registry`, `problems: none`.
- **The launcher.** `shasum -a 256 docs/handover/quadsource_atol_sweep.py` is
  `2a98530369f530cfc6f33ae91ff78fb51d3cbc89dea56f8bcfb9cb9ee012676d`, every R3 manifest's
  `script_sha256`; its last commit is `7af17bd`. Its line numbers in `runregistry-extraction`
  verification §4.5 are still right, since the file has not changed.
- **The lock helper's exits after a hard kill** (`tools/rehearsal_lock.py`): `release` with a Ray
  process listed prints the processes and returns **1** (`:376-380`); `acquire` with one listed
  removes the lock again and returns **4** (`:293-299`).
- **RunRegistryKit**, `HEAD` `f6d4b80`, clean. Read at `f6d4b80` with `git show f6d4b80:<path>`,
  so that a later commit there does not move the lines:
  - `prompts/interrupted-runs/IMPLEMENTATION_STATE.md` §2 (`:1176-1188`): G1's cell is `not yet
    (SGK adopts `v0.2.0`)`, and the italic close note says a gate is recorded "in a records commit
    of its own". §4: `[04-runs-do-not-say-which-machine-produced-them]` at `:1523`, **Resolved
    (2026-10-16, by 01, `387c58f`)** at `:1597`;
    `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]` at `:1641`,
    **Resolved (2026-10-17, by 02, `2bf9c72`)** at `:1763`. §3: `[03-a-run-whose-manifest-…]` at
    `:1345`, `[04-a-killed-process-not-yet-reaped-…]` at `:1452`, both "open, owner none" at the
    close.
  - `prompts/interrupted-runs/README.md` §7 (`:388-402`): G1's condition, word for word as README
    §2.2 quotes it.
  - `prompts/extraction/IMPLEMENTATION_STATE.md:1128`: the G2 cell, the form.
  - `docs/OPEN_ISSUES.md`: 11 open; §1.2 lists `[00-a-copy-…]`, `[01-cross-filesystem-move-…]` and
    `[06-runregistrykit-list-into-a-closed-pipe-…]`; §1.3 lists the six, the two of Z4 among them.
  - The adoption note at `v0.2.0`, `docs/adoption/v0.2.0/secondarygwkit.md`: §1 the pin, §2 its
    runs, §3 `liveness`'s values, §4 `begin` with `results` and `--build --resume` after a hard
    kill, §5 the handlers' `finish("killed", …)`, §6 `TERMINAL_STATES`, §7 refusal fragments and
    two runs on one store, §8 `pid_alive` and key sets, §9 the stale-run brief point by point.
- **SGK's records Z3 touches.** `docs/OPEN_ISSUES.md`: §1.11's note `:674-676`, §1.17's
  `:836-839`, §1.21's lead-in `:913-917`, §1.22 `:920-931`. The boards: `run-registry` `:377`,
  `a3-v2-readiness` `:1685`, each the `Transferred (2026-10-16)` paragraph, each saying the issue
  "is **still open there**".
- **`prompts/INDEX.md`.** 27 rows: `handover` and `datastore-readback` **live**,
  `perturbation-sound-speed` **seeded**, this campaign **in progress**, 23 closed or complete.
- **The index:** 100, by `git grep -c '^| \`\[' docs/OPEN_ISSUES.md`.
- **Shell traps.** `stat` is GNU coreutils here; `grep` may wrap `ugrep`, so use `git grep` or
  Python for anything the log relies on; the shell is `zsh`, where a bare word beginning `=` fails
  and an unmatched glob is an error.
- **The roots** at `ebb70a3`, serially, load 22–40:

  | Root | Ran | Result | Wall |
  |---|---|---|---|
  | `AdaptiveLevin/tests` | 32 | OK | 1 s |
  | `CosmologyModels/tests` | 39 | OK | 3 s |
  | `ObjectFactories/tests` | 279 | OK | 19 s |
  | `tools/tests` | 91 | OK | 14 s |
  | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 37 s |
  | `RunDrivers/tests` | 103 | OK | 14 s |
  | `ComputeTargets/tests` | 700 | FAILED (failures=1): `test_tk_wkb_phase.TestCost.test_wall_time_per_object` | 446 s |

  Every count is 02's. The one failure is README §8's first known wall-clock test. Re-run alone at
  load 40 it failed again (`0.0954 … not less than or equal to 0.05`): the machine was running a
  Time Machine backup (`backupd` at ~200 % CPU). It passed at 03's review at load 4–6, and no code
  has changed since. See ruling 9.

**What the review exists to establish.**
- **(V1) The measurements are true** at the agent's `HEAD`, and equal to this note's.
- **(V2) The G1 draft** is in the G2 cell's form, true in every clause, maps each of the adoption
  note's §1–§8 to SGK evidence that exists, and claims no defect of the package.
- **(V3) Z3 is true and minimal**: both issues resolved in RunRegistryKit as stated; the three left
  alone still open there; the index's four paragraphs and the two boards' entries changed by the
  rulings' amounts and no more.
- **(V4) Z4 is read from the tree and the records**, and rule 5's new sentences carry U4's substance
  unweakened, contradict nothing in RunRegistryKit's README at the tag, and change no other rule.
- **(V5) The close**: the board, README §2, `prompts/INDEX.md`, §5.4 naming the open issue and
  every decision left to the user.
- **(V6) The roots** pass at 02's counts.

**Conventions.** 03's, carried. All bind the agent, and the brief says so.
- **Staging.** By explicit path only; show `git diff --cached --name-only` before the commit.
  Count, then paste. A log states only what the record shows.
- **The SHA.** "this commit" where 04's hash belongs; the orchestrator substitutes it.
- **Suites.** Each in the foreground, serially, output to a file in the agent's scratch directory,
  the verdict grepped from it.
- **Scratch.** The agent's own subdirectory of the session scratchpad. Nothing written under
  `var/`; no scratch file in the repository.
- **The venv.** Nothing installed or uninstalled.
- **RunRegistryKit.** Read only, with `git show f6d4b80:<path>` or `git show v0.2.0:<path>`.
- **No build, no Ray, no lock**, and no store opened but by `store show`.
- **Stop conditions are real.** A stop leaves the tree as it is, and goes to the user through the
  orchestrator; the agent resumes with `SendMessage`.

## 1. Before you dispatch

Done at `ebb70a3` on 2026-10-19; the results are §0's facts. Repeat 1–2 at the dispatch `HEAD`.
1. **The tree.** Clean; one worktree; nothing outside `prompts/` since `ebb70a3`.
2. **The registry, the lock and Ray.** 51 runs, none `running` or stale; lock free;
   `ray-processes` none.
3. **The venv.** `pip freeze` against 01's; `package_provenance` against R3's manifests.
4. **The 24 `store show`s**, and the 21 against 01's after-blocks.
5. **Baselines.** The seven roots, as §0's table.
6. **The index.** 100.
7. **The `var/` marker**: immediately before dispatch, `touch` a marker in the orchestrator's scratch
   directory.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it the prompt, the README, the board, `CLAUDE.md`,
`HEAD`, the baselines, the index count, and §0 of this note up to "## 1. Before you dispatch", as a
copy in the session scratchpad. The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after every check and the suites, staged by explicit path from
  §0's list.
- **The order.** Z1's measurements first, before any edit; then Z3; then Z2; then Z4, with
  `CLAUDE.md`; then Z5; then the suites.
- **The rulings of §0**, each logged as an `IMPLEMENTATION CHOICE` citing this note, or
  `STRUCTURALLY REQUIRED` where the prompt cannot be followed as written.
- **Nothing closes, nothing is committed in RunRegistryKit, nothing is retired.**
- **Stop and ask** on any of the prompt's §4 conditions.

## 3. The review — ten checks

1. **Scope.** `git show --name-status <04>`: `A` for the log and the five `measure/out/04-*`; `M` for
   exactly §0's list. Verification: one hunk appending §3, no removed line. README: row 04 alone.
2. **(V1) The measurements,** re-run here at `<04>`: `pip freeze` against 01's; `package_provenance`;
   `list` (51, none `running`); the 24 `store show`s; `df`. Each equals its saved output but for
   the commit line and the disk's free bytes.
3. **(V2) The G1 draft,** read whole against RunRegistryKit at `f6d4b80`: the cell's form; the date
   (ruling 1); every SGK commit and run id it names exists and says what is claimed; each of the
   adoption note's §1–§8 mapped to a log, a verification section or a measurement that exists; the
   GCS finding said to be Ray's, not the package's.
4. **(V3) Z3.** RunRegistryKit's §4 **Resolved** lines and commits as §0; the release notes name
   each change; the index's three paragraphs by `git diff -U0`, each change a present-tense clause
   (ruling 3); the two boards, one paragraph added each, in place (ruling 4), nothing else; the
   three left alone still in RunRegistryKit's §1.2.
5. **(V4) Z4.** Verification §3.5 against the tree: the launcher's hash, last commit and forced
   address; the pins by line; the commands in their spelling, `end` among them; the route's steps
   against verification §2; the lock helper's exits as §0; the two RunRegistryKit issues with their
   meaning for A3 v2; the issues that no longer bear on it, each closed where it says; the disk.
6. **(V4) `CLAUDE.md`.** `git diff` touches rule 5 alone. The added sentences say: check
   `ray-processes` after `end`; a GCS server may be left, stuck, unmoved by `SIGTERM`; a **person**
   ends one still listed, with `SIGKILL`, by its pid, never by pattern or `ray stop`; the lock waits
   for it. Read against RunRegistryKit's README at the tag and the lock section of `CLAUDE.md`: no
   contradiction.
7. **(V5) The records.** The board: status closed, §1 row 04 ✅, Z1–Z5 ✅, §3 holding the one
   narrowed entry, §5.4 listing it and every §6.3 decision. The index: 100 by counting; the hook;
   §1.22's sentence; the date. `prompts/INDEX.md`: the row closed (ruling 5), the header's counts
   by counting its rows.
8. **(V5) Verification §3.1** against the six run directories: ids, `git_head`, packages, states,
   exit codes, digests, `encoding_format`, `fingerprint_sidecar`.
9. **Nothing else moved.** `find var -newer <marker>` is empty; RunRegistryKit's checkout is
   unchanged at `f6d4b80`, clean; `git status --short` clean after the agent's commit.
10. **(V6) The suites.** All seven roots, serially, at the commit, every count as §0.

## 4. After the review

- **Record it on the board** as an *Orchestrator review of prompt 04* paragraph in §5.4, in 03's
  form. Fix small residue in a follow-up commit of its own:
  - "this commit" → the SHA, on the board, in the log's header, in README §2's row, in the
    verification document and **in the G1 draft**;
  - this note listed as "used for 04";
  - `prompts/INDEX.md`'s row, if the agent left it short of the board.
- Anything the review finds that is not residue is opened on the board's §3 and in the index.
- **Report to the user:** V1–V6, each in a line; the G1 draft, ready for RunRegistryKit; what A3 v2
  inherits in four lines; rule 5's new sentences quoted; and what is theirs (README §6.3): R3's
  three stores, the G1 commit, C3, and the GCS issue's owner.
