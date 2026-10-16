# Log — runregistry-extraction prompt 09, close the campaign

- **Subject:** [`09-close-the-campaign.md`](../09-close-the-campaign.md), board items Z1–Z5.
- **Commit:** this commit (parent `ef34ed0`).
- **Result:** landed. The final measurements are saved, the two gate records are drafted, the six registry issues
  are placed and the five that are the package's are drafted for RunRegistryKit, what A3 v2 inherits is stated,
  and the board, README §2, `prompts/INDEX.md` and `docs/OPEN_ISSUES.md` close the campaign. **Nothing is
  transferred and nothing closes**: every issue stays in its board's §3, and the index holds 106 rows. No code, test
  or script changed. No stop condition of §4 fired.
- **Model:** Sonnet 5.5.
- **Date:** 2026-10-16.

The orchestrator's note was not read as a file in `orchestrator/`; its §0, handed over in writing, is cited as "the
note", and its eight rulings as "ruling N".

**Where it ran.** At `ef34ed0` ("Write the orchestrator note for runregistry-extraction prompt 09"), clean, on
`handover-remedial`, whose code is `a9ed928`'s. The registry showed none `running` or stale, the lock was free
and `RAY_ADDRESS` unset. The measure and the `git grep` read the committed tree, so they measured the tree 09
started from, which 09 changes only in records (verification §4).

## What shipped

- **Z1** (verification §4.2; `measure/out/09-*`): `09-roots.txt`, `09-pip.txt`, `09-measure.txt`, `09-list.txt`,
  `09-store-show.txt`, `09-disk.txt` and `09-runs.txt`, each whole with its command.
- **Z2** (verification §4.3): RunRegistryKit's G2 status cell, and DatastoreKit's dated italic note, each in a
  fenced block, with the six checklist clauses mapped to evidence. **Neither is committed in its repository.**
- **Z3** (verification §4.4):
  - [`runregistrykit-inherited-issues-brief.md`](../runregistrykit-inherited-issues-brief.md), five entries, each in
    RunRegistryKit's board-entry form with a proposed `docs/OPEN_ISSUES.md` row;
  - one `**Drafted for RunRegistryKit (2026-10-16, runregistry-extraction prompt 09):**` line on each of four
    entries, on the `run-registry`, `store-retirement` (two) and `a3-v2-readiness` boards, and nothing else on
    those boards (`git diff -U0` holds 20 added lines, one of them blank, and no removed one);
  - a `**Narrowed (2026-10-16, prompt 09):**` line on this board's two entries.
- **Z4** (verification §4.5): the launcher at `2a985303…`, the commands, the store, what bears on it.
- **Z5**: verification §4 (§4.1 the twelve runs); this board (status line, §1 row 09 and items Z1–Z5, a closing line
  under §2, §3's two narrowings, §5.15); README §2's table; `prompts/INDEX.md`; `docs/OPEN_ISSUES.md` (the six
  hooks, §1.21's lead-in, the date).

**The placement** (file:line at `v0.1.0` for the package, at `ef34ed0` for SGK):

| Issue | Whose | Lines |
|---|---|---|
| `[04-runs-do-not-say-which-machine-produced-them]` | package; a decided change | `__init__.py:604` (`begin`), `:707-723` (manifest), `:258` (`pid_alive`), `:294` (`liveness`); no `machine` anywhere |
| `[00-a-copy-carries-its-sources-present-tense-fields]` | package | `stores.py:1138` (`copy_store`), `:1160` (`deepcopy`) |
| `[01-cross-filesystem-move-advice-says-delete-by-hand]` | package, the relaying half only | `stores.py:1119-1129` (`_store_files`), `:1179` (`move_store`), `:1195` |
| `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]` | package | `__init__.py:242-251`, `tmp` at `:245`; `_update` `:398-403` |
| `[06-runregistrykit-list-into-a-closed-pipe-ends-in-a-traceback]` | package | `__main__.py:563` |
| `[00-a-sigterm-pipeline-run-is-recorded-as-failed]` | SGK | `docs/gktk-remedial/scoped_pipeline_run.py:256`, `:412`, `:429`; the package defines none of the three |
| `[04-a-sigterm-during-ray-init-meets-rays-handler-not-the-drivers]` | SGK | the same file, `install_killed_handler` `:412`, after `ray.init` |

## Deviations from the prompt

1. **The two SIGTERM issues are SGK's** (ruling 1). Re-derived from the code, not taken from the note: `git grep -nP
   "def (terminal_state|finish_after_exec|install_killed_handler)\(" v0.1.0` in RunRegistryKit is empty, and the
   package has no `SIGTERM` or `signal.` outside its tests; the three are at the lines above in SGK. They stay where
   they are, untouched, and the brief names them as SGK's. **IMPLEMENTATION CHOICE**, by the note.
2. **`[04-a-signal-inside-a-heartbeat-write-…]` is the package's** (ruling 2), though SGK's handler triggers it: the
   defect is the fixed temporary name (`__init__.py:245`), shared by `heartbeat` and `finish` through `_update`. Handed
   over whole. **IMPLEMENTATION CHOICE**, by the note.
3. **`[01-cross-filesystem-move-advice-…]` is handed over as the relaying half only** (ruling 3). The advice text is
   DatastoreKit's (installed `ShardedPool.py:3029`, its `docs/OPEN_ISSUES.md` §1.2). The brief says both halves and that
   only the relaying one is RunRegistryKit's. **IMPLEMENTATION CHOICE**, by the note.
4. **`[04-runs-do-not-say-which-machine-…]` is a decided change, not a defect** (ruling 4). The brief carries the user's
   decision of 2026-09-24 as the entry states it, says it is SGK's decision for SGK's runs, and asks for
   RunRegistryKit's own decision for every client. **IMPLEMENTATION CHOICE**, by the note.
5. **The G2 draft replaces the status cell only** (ruling 5), in E2's form, keeping the checklist pointer, because
   RunRegistryKit's board §2 says nothing else on the board changes for a gate. **IMPLEMENTATION CHOICE**, by the note.
6. **DatastoreKit's draft is a dated italic note** appended to its `fingerprint` board §2 paragraph (ruling 6), R1's
   both-ways result first, then the adoption. "No difference" is the finding; no issue is opened. **IMPLEMENTATION
   CHOICE**, by the note.
7. **README §2's table** says "written <date>; landed `<sha>`" for 01–08 (ruling 7), taken from the board's §1, and 09's
   "written 2026-10-16; landed this commit". The "held: …" clauses on 04 and 06–08 are dropped: each hold ended when its
   prompt landed. **IMPLEMENTATION CHOICE**, by the note.
8. **`store show` on twenty sidecars** (ruling 8), found with `find var/datastores -name '*.manifest.json'`, not a glob of
   the top level (nineteen). Eighteen `retired` and R2's two live, each `problems: none`. **IMPLEMENTATION CHOICE**, by
   the note.
9. **Extra files in `measure/out/`**: `09-runs.txt` (verification §4.1's source) and `09-disk.txt` (the disk figure Z4
   asks to be measured again, with the lock and `RAY_ADDRESS`) beside Z1's list. The prompt's §5 allows `measure/out/09-*`.
   **IMPLEMENTATION CHOICE.**
10. **The roots ran twice** at the same code: once before the records were written, in the scratchpad, to have counts for
    §4, and once after the last record edit, saved as `09-roots.txt`. Both were all-OK at 08's counts. **IMPLEMENTATION
    CHOICE.**
11. **The brief proposes the issue rows in the form of RunRegistryKit's `docs/OPEN_ISSUES.md`** (Issue, Owner, Hook),
    with owner `none`, and suggests a first step for two entries (a line in `copy`'s output; waiting for DatastoreKit's
    reword). They are proposals for RunRegistryKit to take or leave. **IMPLEMENTATION CHOICE.**
12. **The board's "Orchestrator notes" paragraph does not list `orchestrator/prompt-09.md`**: the prompt told this session
    not to read `orchestrator/`, and the orchestrator records a note as used in its review, as it did for 08.
    **IMPLEMENTATION CHOICE.**
13. **`store show` was first run on the extension-less primary path**, which made every block report a
    problem ("does not name the primary beside it"). That first file was overwritten before anything relied on it, and
    the saved `09-store-show.txt` passes the `.sqlite` path, as `01-store-show.txt` does. **UNINTENDED DRIFT**, corrected
    in the same session; no record cites the discarded output.

## Verification performed

**The acceptance items** (prompt §3):
1. **Z1–Z5** as §1 says: above.
2. **The measure** reports 0 in every live-code and staying-test figure (422 files), and the `git grep` prints
   `CLAUDE.md:82` and `config/fingerprint.py:33` alone (`09-measure.txt`).
3. **`pip freeze` equals 05's after-install freeze** (its "before" list plus the one `runregistrykit` line; the `diff`
   against the "before" list is `81a82` alone), `pip check` is clean, and **the provenance equals each of R2's four
   manifests' `packages`** as JSON (`09-pip.txt`).
4. **The index** is 106 by `git grep -c '^| \`\[' docs/OPEN_ISSUES.md`, its date is 2026-10-16, and the six changed
   rows agree with their boards. `prompts/INDEX.md` says closed 10 / 10, 2 live and 23 closed or complete, 26 rows.
5. **The seven roots** pass at 08's counts: AdaptiveLevin 32, CosmologyModels 39, ObjectFactories 279, tools 91,
   LiouvilleGreen 148 (1 skipped), RunDrivers 99, ComputeTargets 693. No flake.
6. **No code, no test, no breakage**: `git status` lists records only (the 8 modified and 9 added files below), and no
   `.py` file; `black` was not run.
7. **One commit**, with this log.

**The prompt's §2:**
- **Z1's measurements** are each saved whole with their command. The measure and the `git grep` read the committed tree,
  so they measured the tree 09 started from, `ef34ed0` (said in `09-measure.txt` and verification §4).
- **Z3's placement** is shown with its lines: the package's from `git show v0.1.0:<path>`, SGK's from the tree.
- **The three boards**: `git diff -U0` on `prompts/run-registry`, `prompts/store-retirement` and
  `prompts/a3-v2-readiness` holds only added lines, one block per entry, none removed.
- **Z4 and §4.1** read from the tree and the run directories: the twelve runs' `manifest.json` and `status.json`
  (`09-runs.txt`), the launcher's hash with `sha256sum`, the flags from the script's `argparse` and verification §3.1.
- **State of the world.** `list`: 43 runs, 23 `done`, 15 `killed`, 3 `failed`, 2 `unknown`, none `running`; 01's 29 each
  in its recorded state. `store show`: 20 sidecars. `/` 97 % used, 36 GB free.

**Files in the commit** (staged by explicit path): modified `docs/OPEN_ISSUES.md`,
`docs/runregistry-extraction-verification.md`, `prompts/INDEX.md`, `prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md`,
`prompts/run-registry/IMPLEMENTATION_STATE.md`, `prompts/runregistry-extraction/IMPLEMENTATION_STATE.md`,
`prompts/runregistry-extraction/README.md`, `prompts/store-retirement/IMPLEMENTATION_STATE.md`; added the brief, this
log and `measure/out/09-{disk,list,measure,pip,roots,runs,store-show}.txt`.

## The deliberate-breakage record

None. This prompt adds no code, no test and no script (README §5 rule 8).

## Observations not acted on

1. **The disk is 97 % used, 36 GB free**, past Ray's 95 % threshold, down from 40–44 GB at R2. It is recorded in
   verification §4.5 as something A3 v2 inherits and not opened as an issue, as 08's review ruled.
2. **The load was 8–28** during the measurements, with other work on the machine; no run depended on timing but the
   suites, which passed.
3. **Verification §3.11 says "all 18 tombstones"; there are now twenty sidecars**, the 18 and R2's two live. Both are
   true of their own dates; §4.2 states the twenty. Per README §5 rule 6, §3.11 is not rewritten.
4. **The brief's two suggested first steps** (a line in `copy`'s output naming the source-bearing fields it carries;
   waiting for DatastoreKit's reword rather than matching its message text) are the session's reading, not
   measurements. RunRegistryKit may choose differently.
5. **`[04-a-signal-inside-a-heartbeat-write-…]` is read, not measured**, in the board entry and in the brief. 09 did not
   reproduce it: it would write a status file, and 09 touches nothing under `var/`.
6. **The G2 draft's sentence on defects** says none was met in the rehearsal, and names the closed-pipe issue as the one
   found in adoption: true of 08's record and 06's review, not re-measured.

## State handed to the next prompt

None: this closes the campaign. What is left, none of it done here (README §6.3, board §5.15):
- **The user's three decisions**: retire R2's two stores or not; commit the two gate records in RunRegistryKit and
  DatastoreKit; hand the brief to a session in RunRegistryKit.
- **After the hand-over**, a records commit here moves the four entries to their boards' §4 as transferred, closes this
  board's two, and corrects `docs/OPEN_ISSUES.md` (106 → 100 when the six rows go) and
  `prompts/INDEX.md`.
- **A3 v2** (`handover`) inherits verification §4.5, including
  `[00-main-and-the-readers-default-to-ray-address-auto]`, assigned to it.
- **The index** holds 106 rows.
