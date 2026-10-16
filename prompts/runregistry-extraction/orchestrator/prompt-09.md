# Orchestrator — prompt 09, close the campaign

Read [`../README.md`](../README.md) first: §2.2 (09's charter), §3, §5, §6.2 (C11) and §6.3. Then the
board's §2, §3 and §5.14 with the *Orchestrator review of prompt 08*.
`dk-and-rpk-migration`'s [`orchestrator/prompt-09.md`](../../dk-and-rpk-migration/orchestrator/prompt-09.md)
is this note's model for a close that hands issues to a package; what differs is said here.

**You do not write code.** You may:
- run RunRegistryKit's measure from its root, with its own venv, at any SGK commit;
- run the suites, `python -m runregistrykit list` and `store show`;
- read RunRegistryKit and DatastoreKit with `git`, never changing them;
- fix small residue in a follow-up commit of your own (§4).

**The prompt:** [`09-close-the-campaign.md`](../09-close-the-campaign.md)
**Board items:** Z1–Z5 · **Closes:** nothing · **Narrows:**
`[00-open-issues-about-the-registry-code-stay-on-sgk-boards]` and
`[06-runregistrykit-list-into-a-closed-pipe-ends-in-a-traceback]` here, and, each on its own board,
`[04-runs-do-not-say-which-machine-produced-them]`, `[00-a-copy-carries-its-sources-present-tense-fields]`,
`[01-cross-filesystem-move-advice-says-delete-by-hand]` and
`[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]` · **Opens:** nothing
expected · **Model:** Sonnet, as the prompt recommends.

**Gate:**
- 08 landed (`957d658`) and was reviewed (`33a7346`). R2 and RunRegistryKit's G2 hold on the
  board's §2. The tree is clean.
- `HEAD` is the commit that adds this note, or a later one that touches nothing outside `prompts/`.
  `c07ab17` wrote the prompt and touches `prompts/` alone.
- Nothing is `running` in the registry; no Ray process of this repository is up; the lock is free.
  09 starts no Ray, but the suites must not share the machine with a build.

## 0. What makes this prompt unusual

**It is records, and a hand-over that cannot be completed inside it.** No code changes. What the
review must establish is that every record is true, by counting and by reading the code at the tag,
and that nothing is claimed as done that only the user can do.

**A drafted issue is not a transferred one.** At `dk-and-rpk-migration` 09, DatastoreKit's
`docs/OPEN_ISSUES.md` already listed the four layer issues, so 09 moved them to their boards' §4.
RunRegistryKit lists none of these five (its index holds its own two). So here each stays open, on
its own board, with a line saying where it went, and moves to §4 in a records commit after the user
has handed the brief over. The prompt says this (§1, Z3); the agent records the rule, not a choice.

**Rulings on the prompt** (the orchestrator's, made in writing this note; the user may reverse any).
Pass them on; the agent logs each as the classification given.
1. **The two SIGTERM issues are SGK's.** `terminal_state`, `install_killed_handler` and
   `finish_after_exec` are `docs/gktk-remedial/scoped_pipeline_run.py:256`, `:412` and `:429`, and
   `runregistrykit` at `v0.1.0` defines none of the three (`git grep -nP "def
   (terminal_state|finish_after_exec|install_killed_handler)\("` there is empty). So
   `[00-a-sigterm-pipeline-run-is-recorded-as-failed]` and
   `[04-a-sigterm-during-ray-init-meets-rays-handler-not-the-drivers]` stay where they are,
   untouched; the brief names them as SGK's, with these lines. The agent re-derives this from the
   code and records it; a different reading is §4's stop.
2. **`[04-a-signal-inside-a-heartbeat-write-…]` is the package's, though SGK's handler triggers
   it.** The defect is the fixed temporary name in `write_json_atomic`; the handler that calls
   `finish` inside a heartbeat is SGK's driver, but nothing in SGK's code could avoid the shared
   inode. The entry's own next step says "In `RunRegistry`". It is handed over whole.
3. **`[01-cross-filesystem-move-advice-…]` is handed over as SGK's half only.** The entry was
   narrowed at `dk-and-rpk-migration` 09: the advice text is DatastoreKit's, indexed there in §1.2,
   and the half that stayed SGK's is `store move` relaying it for a registered store
   (`_store_files`, `move_store`). That half's code is now the package's. The brief's entry says
   both halves, and that only the relaying half is RunRegistryKit's to fix.
4. **`[04-runs-do-not-say-which-machine-…]` is a decided change, not a defect.** The brief carries
   the user's decision of 2026-09-24 as the entry states it (`machine` in the manifest, `list`
   refusing liveness for a foreign record, only terminal run directories transferred), and says it
   is SGK's decision for SGK's runs: whether RunRegistryKit adopts it for every client is its own
   question.
5. **The G2 draft replaces the cell only.** RunRegistryKit's board §2 says "a gate is recorded on
   this board when it holds, in a records commit of its own; nothing else on the board changes for
   it". So the draft is the G2 row's status cell, in E1's form (`**holds** 2026-10-16: …`), keeping
   "its checklist is [`docs/adoption/secondarygwkit.md`](…) (07, `aaab408`)". Nothing else.
6. **The DatastoreKit draft is a dated italic note** appended to its `fingerprint` board's §2
   paragraph, in the form of the `*(2026-10-14: …)*` already there. It reports R1's both-ways result
   first (what that paragraph waits for), then the adoption: 03's pin, 06's fingerprinter, R2's W1
   across stores. "No difference" is the finding; no issue is opened there.
7. **README §2's table.** Each of 01–08's status cells is "written <date>" and sometimes "held: …";
   none says landed. Each becomes "written <date>; landed `<sha>`", from the board's §1, and 09's
   "written 2026-10-16; landed this commit".
8. **`store show` on twenty sidecars.** There are twenty under `var/datastores/`, one in
   `backup-pre-resume-20260921T091011/`: eighteen tombstones and R2's two live stores. Use `find`,
   not a glob of the top level, which finds nineteen.

**What moves on purpose:**
- **Added:** `runregistrykit-inherited-issues-brief.md`; `measure/out/09-*`;
  `logs/09-close-the-campaign.md`.
- **Modified:** `docs/runregistry-extraction-verification.md` (§4 appended); the boards of
  `run-registry`, `store-retirement` and `a3-v2-readiness` (one line per entry, four entries); this
  board; README §2's table; `prompts/INDEX.md`; `docs/OPEN_ISSUES.md`.
- **Not modified:** any code, test, script, `requirements.txt`, `CLAUDE.md`, `runregistrykit.toml`,
  `config/`, `orchestrator/`, every other campaign's board; anything in RunRegistryKit, DatastoreKit
  or RayPoolKit; anything under `var/`.
- **The index stays at 106.** Six rows change their hook: §1.11 one, §1.14 two, §1.17 one, §1.21 two.
  Its date becomes 2026-10-16, and §1.21's lead-in gains a sentence on 09.

**The facts, checked by the orchestrator on 2026-10-16 at `33a7346`.** Pass them on.
- **The tree and the machine.** Branch `handover-remedial`; clean; one worktree. `git diff --stat
  b9daf44 33a7346 -- . ':!prompts' ':!docs/runregistry-extraction-verification.md'
  ':!docs/runregistry-extraction'` is empty: the code is R2's. `RAY_ADDRESS` unset; lock free;
  `ray-processes` none. Load 6–24. **The volume is 97 % used, 35 GB free** (`df -h /`), down from
  40–44 GB at R2.
- **The registry.** 43 runs: 41 finished, 2 `unknown`; none `running` or stale. `store show` on the
  twenty sidecars: each `problems: none`, eighteen `retired`.
- **The venv.** `pip freeze` equals 05's "before" list plus its one added line,
  `runregistrykit @ git+https://github.com/ds283/RunRegistryKit@9f8ebe431503fb8c79ee1407cc460501f1ca2d92`
  (the `diff` against lines 4–103 of `05-pip.txt` is that `81a82` line alone). `pip check` clean.
  `package_provenance(["datastorekit", "raypoolkit", "runregistrykit"])` equals, as JSON, R2's
  `reference` manifest's `packages`: 0.4.0 at `c8570e84…`, 0.2.0 at `eaef0ce6…`, 0.1.0 at
  `9f8ebe43…`, none editable. (`package_provenance` takes the names; with none it raises
  `TypeError`.)
- **The measure.** RunRegistryKit's `docs/adoption/measure_client_registry.py sgk --commit 33a7346`,
  from its root at `caad864`, clean, exit 0: 422 `.py` files parsed; every live-code and
  staying-test figure 0; campaign records 6 files importing it, 7 naming it. The `git grep` of
  verification §3.11 prints `CLAUDE.md:82` and `config/fingerprint.py:33` alone.
- **RunRegistryKit.** `HEAD` `caad864`, clean; its extraction campaign closed (`3aab658`).
  `git diff --stat v0.1.0 caad864 -- runregistrykit pyproject.toml` is empty, so the tag's lines
  are its `HEAD`'s. Its `docs/OPEN_ISSUES.md` holds two issues, both its own.
- **The package's lines at `v0.1.0`**, for the agent's cross-check (it reads them itself):
  - `write_json_atomic`, `runregistrykit/__init__.py:242-251`; the fixed `tmp = path + ".tmp"` at
    `:245`.
  - `copy_store`, `stores.py:1138`; its docstring: every other field, "unknown ones included, is
    the source's, verbatim"; `fields = _copy.deepcopy(reading.fields)` at `:1160`.
  - `_store_files`, `stores.py:1119-1129`, re-raising `f"{e}. No sidecar was written or moved: …"`;
    `move_store` at `:1179`, calling `_store_files("move", ShardedPool.move_store, src, dst)` at
    `:1195`.
  - `begin`, `__init__.py:604`; the manifest's `git_head` at `:152`, `argv` and `cwd` at
    `:712-713`; `pid_alive` `:258`, `liveness` `:294`. No `machine` and no `platform` anywhere in
    the package's code.
  - `sys.stdout.flush()`, `__main__.py:563`, unguarded, in `list`'s path.
- **The SGK board entries:** `prompts/run-registry/IMPLEMENTATION_STATE.md:61`;
  `prompts/store-retirement/IMPLEMENTATION_STATE.md:471` (SIGTERM, SGK's), `:508` (copy), `:525`
  (move); `prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md:1375` (heartbeat), `:1392` (`ray.init`,
  SGK's). Index rows: `docs/OPEN_ISSUES.md:676`, `:745`, `:746`, `:838`, `:907`, `:908`.
- **DatastoreKit.** `HEAD` `b01737f`; its `fingerprint` board §2 is one paragraph with one dated
  italic note (`prompts/fingerprint/IMPLEMENTATION_STATE.md:259-268`), saying SGK's R1 result is
  recorded there "when SGK reports it". Its `docs/OPEN_ISSUES.md` §1.2 and
  `prompts/inherited-issues/README.md` are the precedent for a hand-over board.
- **The twelve rehearsal runs**, read from each run directory:

  | Run | State | Exit | Digest | `encoding_format` | `git_head` | Packages |
  |---|---|---|---|---|---|---|
  | `…-02-reference-20261012T091921` | done | 0 | `39809dca` | — | `fe16d8e` | DK 0.2.2, RPK 0.2.0 |
  | `…-02-resumed-a-20261012T095041` | killed | 143 | `9bb753bd` | — | `fe16d8e` | same |
  | `…-02-resumed-b-20261012T100353` | killed | 143 | `a0d053af` | — | `fe16d8e` | same |
  | `…-02-resumed-c-20261012T101145` | done | 0 | `39809dca` | — | `fe16d8e` | same |
  | `…-04-reference-20261014T154651` | done | 0 | `39809dca` | — | `89ff8b7` | DK 0.4.0, RPK 0.2.0 |
  | `…-04-resumed-a-20261014T160226` | killed | 143 | `a410cce0` | — | `89ff8b7` | same |
  | `…-04-resumed-b-20261014T161222` | killed | 143 | `8146dab8` | — | `89ff8b7` | same |
  | `…-04-resumed-c-20261014T161739` | done | 0 | `39809dca` | — | `89ff8b7` | same |
  | `…-08-reference-20261015T235716` | done | 0 | `39809dca` | 1 | `b9daf44` | DK 0.4.0, RPK 0.2.0, RRK 0.1.0 |
  | `…-08-resumed-a-20261016T000342` | killed | 143 | `87e55e8f` | 1 | `b9daf44` | same |
  | `…-08-resumed-b-20261016T000708` | killed | 143 | `8978b72e` | 1 | `b9daf44` | same |
  | `…-08-resumed-c-20261016T000959` | done | 0 | `39809dca` | 1 | `b9daf44` | same |

  §4.1 states each tree's code commit beside its dispatch commit (R0 `a45aefc`'s, R1 `5b81f2e`'s,
  R2 `a9ed928`'s), from verification §1–§3.
- **The launcher.** `docs/handover/quadsource_atol_sweep.py`, SHA-256 `2a985303…676d`, equal to
  every R2 manifest's `script_sha256`: it has not changed since R2.
- **`prompts/INDEX.md`.** Header: "26 campaigns: 3 live, 1 seeded, 22 closed or complete", then the
  sentence of dates on which counts were taken. After 09: 2 live, 23 closed or complete; the row
  "closed 10 / 10, 2026-10-16", "2026-10-12 → 2026-10-16", "§1.1, §1.21; 3", the count taken
  2026-10-16 (§1.21's two, and §1.1's `[00-main-and-the-readers-default-to-ray-address-auto]`).
- **Shell traps.** `stat` is GNU coreutils here; `grep` may wrap `ugrep`, so use `git grep` or
  Python for anything the log relies on; the shell is `zsh`, where a bare word beginning `=` fails.
- **The roots** at `33a7346`, serially, load 5–24:

  | Root | Ran | Result | Wall |
  |---|---|---|---|
  | `AdaptiveLevin/tests` | 32 | OK | 1 s |
  | `CosmologyModels/tests` | 39 | OK | 2 s |
  | `ObjectFactories/tests` | 279 | OK | 15 s |
  | `tools/tests` | 91 | OK | 12 s |
  | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 25 s |
  | `RunDrivers/tests` | 99 | OK | 4 s |
  | `ComputeTargets/tests` | 693 | OK | 139 s |

**What the review exists to establish.**
- **(V1) The measurements are true** at the agent's commit, and equal to this note's.
- **(V2) The placement is right**: five in the package at the lines above, two SGK's, each shown
  from the code.
- **(V3) The brief is complete**: each entry readable without SGK's boards, in RunRegistryKit's
  form, with lines at the tag, impact and next step; nothing claimed as RunRegistryKit's that is
  SGK's.
- **(V4) Nothing is claimed transferred.** Every entry stays in §3 with one added line; the index
  count is 106 by counting; each changed hook agrees with its board.
- **(V5) The drafts** are in their boards' forms and map every checklist clause to evidence.
- **(V6) The close**: the board, README §2, `prompts/INDEX.md`, §5.15 naming every open issue and
  every decision left to the user; the A3 v2 hand-over read from the tree.
- **(V7) The roots** pass at 08's counts.

**Conventions.** 08's, carried. All bind the agent, and the brief says so.
- **Staging.** By explicit path only; show `git diff --cached --name-only` before the commit.
  Count, then paste. A log states only what the record shows.
- **The SHA.** "this commit" where 09's hash belongs; the orchestrator substitutes it. The
  measurements are taken at the agent's `HEAD` before it commits, and the measure reads the
  committed tree: say so.
- **Suites.** Each in the foreground, serially, output to a file in the agent's scratch directory,
  the verdict grepped from it. A known flake (README §8) is re-run alone and recorded, not a stop.
- **Scratch.** The agent's own subdirectory of the session scratchpad. Nothing written under
  `var/`; no scratch file in the repository.
- **The venv.** Nothing installed or uninstalled.
- **RunRegistryKit and DatastoreKit.** Read only, with `git`; the measure is run from
  RunRegistryKit's root with its own venv; nothing there is changed, fetched or committed.
- **No build, no Ray, no lock**, and no store opened but by `store show`.
- **Stop conditions are real.** A stop leaves the tree as it is, and goes to the user through the
  orchestrator; the agent resumes with `SendMessage`.

## 1. Before you dispatch

Done at `33a7346` on 2026-10-16; the results are §0's facts. Repeat 1–3 at the dispatch `HEAD`.
1. **The tree.** Clean; one worktree; nothing outside `prompts/` since `33a7346`.
2. **The registry, the lock and Ray.** 43 runs, none `running`; lock free; `ray-processes` none.
3. **The venv.** `pip freeze` against 05's; `package_provenance` against R2's manifests.
4. **The measure, the grep, the twenty `store show`s** and the package's lines, as §0.
5. **Baselines.** The seven roots, as §0's table.
6. **The index.** `git grep -c '^| \`\[' docs/OPEN_ISSUES.md` gives 106.
7. **The `var/` marker**, for check 9: immediately before dispatch, `touch` a marker in the
   orchestrator's scratch directory.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**. Give it the prompt, the README, the board, `CLAUDE.md`,
`HEAD`, the baselines, the index count, and §0 of this note up to "## 1. Before you dispatch", as a
copy in the session scratchpad. The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after every check and the suites, staged by explicit path from
  §0's list.
- **The order.** Z1's measurements first, before any edit; then Z3's placement, the brief and the
  board lines; then Z2; then Z4; then Z5; then the suites.
- **The rulings of §0**, each logged as an `IMPLEMENTATION CHOICE` citing this note, or
  `STRUCTURALLY REQUIRED` where the prompt cannot be followed as written.
- **Nothing is transferred and nothing closes.**
- **Stop and ask** on any of the prompt's §4 conditions.

## 3. The review — eleven checks

1. **Scope.** `git show --name-status <09>`: `A` for the brief, the log and `measure/out/09-*`; `M`
   for exactly §0's list. Verification: one hunk appending §4, no removed line.
2. **(V1) The measurements,** re-run here at `<09>`: the measure (every figure 0; it parses 422 files
   unless the commit adds `.py`, which it must not); the `git grep`; `pip freeze` against 05's;
   `package_provenance`; `list` (43 runs, none `running`); the twenty `store show`s. Each equals its
   saved output but for the commit line.
3. **(V2) The placement,** against §0's lines, read here at the tag: five package, two SGK's. Every
   file and line in the brief and in the board lines exists at `v0.1.0` and says what is claimed.
4. **(V3) The brief,** read whole. Five entries, each in RunRegistryKit's board-entry form, with
   its SGK name and board, opening date, defect, package lines, measurement, impact and next step;
   rulings 3 and 4 as stated; the two SGK issues named with their lines; five proposed index rows in
   RunRegistryKit's form; placement left to RunRegistryKit. No claim that RunRegistryKit has them.
5. **(V4) The boards.** `git diff -U0 <09>~1 <09>` on each of the three other boards: one added line
   per entry, in its §3, and nothing else; the two SIGTERM entries untouched. This board's two §3
   entries gain the narrowing line, and remain in §3.
6. **(V4) The index.** 106 by counting; date 2026-10-16; the six hooks each agreeing with its board,
   no `RunRegistry/` code path left in them; §1.21's lead-in true. No row deleted or added.
7. **(V5) The drafts.** The G2 cell in E1's form (ruling 5), every clause of RunRegistryKit's
   checklist §9 mapped to verification §3.11 and Z1, naming 06, 07, 08 and R2's two `done` runs.
   The DatastoreKit note in its section's form (ruling 6), naming 03, 04 (and §2.10's 20 of 20),
   06 and 08 (§3.10). Nothing claimed that the records do not show.
8. **(V6) Verification §4.1 and the hand-over.** The twelve runs against §0's table; §4.5 against
   the tree: the launcher's path, hash and forced address, the commands in their current spelling,
   the issues by name with their boards, and the disk measured at the agent's run.
9. **(V6) The records.** The board: status closed, §1 row 09 ✅, Z1–Z5 ✅, §3 holding the three it
   held, §5.15 naming every open issue (this board's three, the four on other boards awaiting the
   hand-over) and the three things left to the user. README §2: ten rows landed. `prompts/INDEX.md`
   as §0. `find var -newer <marker>` empty.
10. **(V7) The roots,** re-run here at `<09>`, serially, at §0's counts; a known flake re-run
    alone. `09-roots.txt` agrees.
11. **The log.** README §5.1's sections; the rulings classified; the placement with its lines; the
    measurements with their commands; observations, none needing an issue unless stated.

## 4. After the review

- **Record it on the board** as an *Orchestrator review of prompt 09* paragraph in §5.15, with
  residue fixed in a follow-up commit: "this commit" → the SHA on the board, in the log, in
  verification §4, in the two drafts and in the brief; this note listed as "used for 09";
  `prompts/INDEX.md` if its row needs the SHA.
- **Report to the user:**
  - V1–V7, each in a line;
  - the hand-over: five drafted for RunRegistryKit, two found SGK's, nothing transferred yet;
  - where the two gate drafts and the brief are, and that committing them in RunRegistryKit and
    DatastoreKit is the user's (C11);
  - what is left: R2's two stores, the two gate commits, handing over the brief; and the records
    commit that follows the hand-over (the four to §4 as transferred, this board's two closed);
  - the disk, for A3 v2.
