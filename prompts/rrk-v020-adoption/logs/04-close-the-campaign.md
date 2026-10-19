# Log — rrk-v020-adoption prompt 04, close the campaign

- **Subject:** [`04-close-the-campaign.md`](../04-close-the-campaign.md), board items Z1–Z5.
- **Commit:** `34bf4f7` (parent `aa0b57c`).
- **Result:** landed. Z1–Z5 pass; no stop condition fired. The one test failure, in `ComputeTargets`, is
  README §8's known wall-clock test, recorded under the orchestrator's ruling 9.
- **Model:** Sonnet 5.5.
- **Date:** 2026-10-19.

**What 04 shows:** the pin is `v0.2.0`, the venv is exactly 01's, and R3's manifests name the packages
the venv holds. The registry has 51 runs and nothing `running` or stale; all 24 stores read
`problems: none`. The record for RunRegistryKit's G1 is drafted. Two issues RunRegistryKit has closed
and released are corrected in the index and on two boards. `CLAUDE.md` rule 5 says what to do about a
GCS server left by a `SIGKILL`.

The record is [`docs/rrk-v020-adoption-verification.md`](../../../docs/rrk-v020-adoption-verification.md) §3,
and this log points into it. The orchestrator's note for this prompt is cited as "the note", by ruling
number.

## What shipped

- **`measure/out/04-{roots,pip,list,store-show,disk}.txt`** (new), each whole, with its command and
  `HEAD` `aa0b57c`, **taken before any edit** (Z1). 04 changes records and `CLAUDE.md` rule 5 only, so
  the tree they measure is the tree 04 leaves, apart from those files.
- **`docs/rrk-v020-adoption-verification.md`**: §3 appended (§3.1–§3.5). Nothing above it changed
  (`git diff` shows 0 deleted lines).
- **`CLAUDE.md`**: rule 5, two sentences (below).
- **`docs/OPEN_ISSUES.md`**: §1.11, §1.17 and §1.21's present-tense statements corrected; §1.22's
  lead-in gains a sentence; the GCS row's hook is narrowed. The date stays 2026-10-19 and the count 100.
- **`prompts/run-registry/IMPLEMENTATION_STATE.md`** and **`prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md`**:
  one dated paragraph each, in the entry's §4.
- **This board**: the header, §1's row 04 and Z1–Z5, §2's closing line, §3's `Narrowed` line, §5.4.
- **README §2**: row 04. **`prompts/INDEX.md`**: this campaign's row and the header.
- **This log.**

### `CLAUDE.md` rule 5, as it was and as it is

As it was (the end of the rule, at `aa0b57c`):

> … The one case `end` refuses that is not a live run is a killed process its parent has not yet
> reaped, which still answers `kill -0` (RunRegistryKit's
> `[04-a-killed-process-not-yet-reaped-answers-kill-0]`); a detached launch through `nohup sh -c`
> is reaped by its `sh`.

As it is (the added sentences follow the unchanged ones, which are not repeated):

> After `end`, check
> `PYTHONPATH=. ./venv/bin/python -m tools.rehearsal_lock ray-processes`: a `SIGKILL`ed driver's GCS
> server may be left alive, reparented to pid 1, stuck in its own shutdown where a `SIGTERM` cannot
> end it (`[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]`). One still listed ten
> minutes after the kill is ended **by a person** with `kill -KILL <pid>`, by its pid, never by
> pattern and never with `ray stop`; until then the rehearsal lock cannot be released.

`git diff -U0 CLAUDE.md` is one hunk, in rule 5. No line is over the file's width.

## Deviations from the prompt

Each of the note's nine rulings is logged as the classification the note gives. The rest are 04's own.

1. **The G1 date is 2026-10-18** (the day `0a27b0c` moved the pin), the campaign's close is "this
   commit", and the orchestrator substitutes the SHA, in the draft too (the note's ruling 1).
   `IMPLEMENTATION CHOICE`.
2. **The draft names the runs by id** from the run directories: R3's `reference` and `resumed-c`, and the
   route's `hardkill-a` (`lost`, verified) and `hardkill-b` (`done`, `fingerprint_sidecar` `written`)
   (ruling 2). `IMPLEMENTATION CHOICE`.
3. **The index's corrections rewrite the present-tense clause, not the history**: §1.11 and §1.17 say
   where and when each issue closed in RunRegistryKit, that `v0.2.0` released it and that SGK adopted it
   at `0a27b0c`, and keep "Its row is deleted here; the board's §4 keeps the entry as the record"; §1.21
   says which two of the four transferred issues are still open (§1.14) and which two closed (ruling 3).
   `IMPLEMENTATION CHOICE`.
4. **The boards' added paragraphs come after the `Transferred (2026-10-16)` paragraph**, inside the
   entry: a top-level paragraph in `run-registry`, a list item in `a3-v2-readiness`. The machine
   entry says its next step differs from RunRegistryKit's change (a `host` with `id`, `boot` and
   `foreign` liveness, in place of a `machine` field) and the heartbeat entry says the same of the
   discard of a superseded write; each says RunRegistryKit's board is right (ruling 4).
   `IMPLEMENTATION CHOICE`.
5. **`prompts/INDEX.md`'s header** reads "27 campaigns: 2 live, 1 seeded, 24 closed or complete" (by its
   rows: 2, 1 and 24, counted), and the row's "Opened → last board update" cell is `2026-10-18 →
   2026-10-19` (ruling 5). `IMPLEMENTATION CHOICE`.
6. **The index header's other clauses are not recounted** ("Across twenty-two campaigns" in
   `docs/OPEN_ISSUES.md`, and "Two of these hold no open issue"); only the date was kept (ruling 6).
   `IMPLEMENTATION CHOICE`.
7. **The GCS issue is narrowed and not assigned** (ruling 7): the board gains the `Narrowed` line, and
   the row's hook now says what is stated and what remains, in place of "**Next step: 04**".
   `IMPLEMENTATION CHOICE`.
8. **`04-store-show.txt` has 24 blocks**: the 21 tombstones, equal to 01's after-blocks line for line,
   and R3's three, live (ruling 8). `IMPLEMENTATION CHOICE`.
9. **A known wall-clock test failed under load and is recorded** (ruling 9): in the full `ComputeTargets`
   run `test_tk_wkb_phase.TestCost.test_wall_time_per_object` failed (`0.1397 not less than or equal to
   0.06`); re-run alone it failed again (`0.0759 not less than or equal to 0.05`) at one-minute load 21.8
   with `WindowServer`, `backupd` and `mds_stores` the heaviest processes. Its output is in
   `04-roots.txt`. `IMPLEMENTATION CHOICE`.
10. **The G1 draft is the whole row** (`| **G1** | **holds** … |`), not the status cell alone, as
    `runregistry-extraction`'s §4.3.1 gave G2: the user pastes it over `| **G1** | not yet (…) |`. The
    prompt asks for the "status cell". `IMPLEMENTATION CHOICE`.
11. **The seven roots were run twice**: once as Z1, before any edit, and once more after the edits, as
    the note's order says (its last step). Both are against 02's counts. The second set is not saved
    under `measure/out/` (Z1's is the saved measurement); its verdicts are in "Verification performed".
    `IMPLEMENTATION CHOICE`.
12. **Rule 5's addition follows the sentences on the unreaped process**, which close the rule, and not
    the sentence on `end` and the resume that precedes them: that sentence's own qualification
    (`[04-a-killed-process-not-yet-reaped-answers-kill-0]`) is about `end`, and the new sentences are
    about what follows `end`. `IMPLEMENTATION CHOICE`.
13. **My first `store show` pass gave sidecar paths** (`…manifest.json`) where the command takes the
    store (`….sqlite`): `store show` answered `kind: absent`, `problems: none` for a sidecar's
    sidecar. The output was discarded and the pass redone with the store paths, as 01 gave them. Nothing
    was written under `var/`, and the saved `04-store-show.txt` is the second pass. `UNINTENDED DRIFT`.
    (The behaviour of `store show` on a path it was not meant to take is an observation, below.)

## Verification performed

Against the prompt's §2 and §3.

1. **Z1** is saved whole, with commands and `HEAD`, before any edit (`git status` showed only the
   `measure/out/04-*` files untracked until Z3).
   - Roots: AdaptiveLevin 32 OK, CosmologyModels 39 OK, ObjectFactories 279 OK, tools 91 OK,
     LiouvilleGreen 148 OK (skipped=1), RunDrivers 103 OK, ComputeTargets 700 with one failure
     (deviation 9). All counts are 02's.
   - `pip freeze`: 101 lines, `diff` against 01's after-install block exit 0. `pip check`: no broken
     requirements. `package_provenance`: equal as JSON to all six R3 manifests' `packages`.
   - `list`: exit 0, stderr empty, 51 runs, 49 finished (27 `done`, 18 `killed`, 3 `failed`, 1 `lost`)
     and 2 `unknown`, "45 with no host recorded". None `running`, stale or `foreign`.
   - `store show`: 24 sidecars by `find`, 24 blocks, 24 `kind: registry`, 24 `problems: none`, 24
     `exit=0`; 21 `retired` blocks equal to 01's after-blocks; R3's three at `39809dca…`.
   - `df -h /`: 927G, 863G used, 64G free, 94 % (the note read 859G, 69G and 93 %).
2. **Z2.** Every SGK commit the draft names exists and says what is claimed: `0a27b0c`, `2818780`,
   `17916cb` (`git show --stat`). RunRegistryKit was read with `git show f6d4b80:<path>` and
   `v0.2.0:<path>` only: the adoption note, the release notes, the board's §2–§4, the README at the tag.
   The adoption note's §1–§8 each map to a clause (verification §3.3).
3. **Z3.** Both issues are in RunRegistryKit's §4 as resolved: `:1597` (01, `387c58f`, 2026-10-16) and
   `:1763` (02, `2bf9c72`, 2026-10-17); both commits are ancestors of `v0.2.0`; the release notes name
   each change (§2.1 and §2.4; §2.3). The three left alone are in its `docs/OPEN_ISSUES.md` §1.2 at
   `f6d4b80`; the two of Z4 are open there (§1.3, `:45` and `:49`). `git diff -U0` of
   `run-registry/IMPLEMENTATION_STATE.md` and `a3-v2-readiness/IMPLEMENTATION_STATE.md`: one hunk each,
   additions only.
4. **The index.** `docs/OPEN_ISSUES.md` has 100 rows by a Python count of the working file, and the
   header says 100; `git grep -c '^| \`\['` at `HEAD` also says 100. No row is added or deleted; one
   hook changes. By its Board column, one row names `rrk-v020-adoption`. `prompts/INDEX.md`'s 27 rows
   count 2 live, 1 seeded, 24 closed or complete, and its header says so.
5. **Z4 and §3.1** are read from the tree and the run directories: the launcher's SHA-256 equals all
   six `script_sha256` (checked with Python), its last commit is `7af17bd`, the lines cited are the
   file's (`:556`, `:610`, `:989-993`, `:1000`, `:1014`), the lock's exits are at
   `tools/rehearsal_lock.py:376-380` and `:293-299`.
6. **`CLAUDE.md`** differs from `HEAD` in rule 5 alone (one hunk) and says nothing RunRegistryKit's
   README at the tag contradicts: the README's "Ending a run whose end was never recorded" says what
   `end` does and does not do, and has nothing on Ray.
7. **The suites,** all seven, after the edits, serially, in the foreground, output to files in the
   scratch directory, verdicts grepped:

   | Root | Ran | Result | Load (1 min) at start |
   |---|---|---|---|
   | `AdaptiveLevin/tests` | 32 | OK | 21.1 |
   | `CosmologyModels/tests` | 39 | OK | 20.6 |
   | `ObjectFactories/tests` | 279 | OK | 62.9 |
   | `tools/tests` | 91 | OK | 39.8 |
   | `LiouvilleGreen/tests` | 148 | OK (skipped=1) | 27.6 |
   | `RunDrivers/tests` | 103 | OK | 24.9 |
   | `ComputeTargets/tests` | 700 | FAILED (failures=1), 773 s | 29.6 |

   Every count is 02's. The one failure is again `test_tk_wkb_phase.TestCost.test_wall_time_per_object`
   (`0.0771 not less than or equal to 0.05` in the full run); re-run alone at load 26.7 it failed
   again (`0.0664 not less than or equal to 0.05`), with `corespotlightd` (109 %), `WindowServer`
   (70 %) and `backupd` (44 %) the heaviest processes. The machine was under a Time Machine backup and
   Spotlight indexing throughout. 04 changes no code a test reads, so the failure is not 04's (ruling 9).
   No other test failed.

## The deliberate-breakage record

None: no code or test changes. 04 edits records (a verification document, boards, indexes, a log),
`CLAUDE.md` rule 5, which sessions read and no code or test does, and adds saved measurements. There is
nothing for a test to catch.

## Observations not acted on

1. **`store show` on a sidecar's path answers `kind: absent`, `problems: none`.** Given
   `var/datastores/X.manifest.json` it looked for `X.manifest.manifest.json` and reported an absent
   store without error (deviation 13). The command takes the store (`X.sqlite`); the answer is
   correct for the argument, but "absent, no problems" for a path that is a sidecar is easy to misread.
   Not RunRegistryKit's to change on this record: it is a usability note only, not opened.
2. **The wall-clock test keeps failing under this machine's current load** (Time Machine, Spotlight).
   At 01 and 02 it passed at load 8–11; at `ebb70a3` and here it failed at load 22–60, alone as well.
   Its limits (`COST_COLD_WALL_TIME_LIMIT` 0.06 s cold, 0.05 s warm) have about a 1x margin. Not 04's.
3. **The disk read 94 %, 64 GB free**, against the note's 93 %, 69 GB, 21 hours earlier: the machine's.
   The headroom to Ray's 95 % spilling threshold is about 17 GB, which verification §3.5 states for A3 v2.
4. **The index header's "Across twenty-two campaigns" and "Two of these hold no open issue"** still
   cannot be reproduced by counting (03's observation 5; the note's ruling 6). Not recounted.
5. **`docs/OPEN_ISSUES.md` §1.22's lead-in** still says "Its plan opened no issue"; the sentence is
   correct about its plan, and the new sentence adds the narrowing. Left.
6. **The `a3-v2-readiness` board's "Transferred" bullet** says "the entry leaves §3" while sitting in
   §4, as written by `runregistry-extraction`; the new item follows it as the prompt says, and the
   wording is not changed.
7. **R3's three stores are live** and 04 retires none (README §3). The user's.

## State handed to the next prompt

There is none: this is the campaign's last prompt. The orchestrator's review (note §3) should check:
- **"this commit"** wherever 04's hash belongs, to be replaced by the orchestrator: this log (the header
  and the "Commit" line), this board (§1 row 04 and its Z1–Z5 cells, §2's closing line, the header's
  neighbours, §5.4), README §2's row 04 and the verification document's §3.3 (inside the G1 draft, at
  "04 (this commit, 2026-10-19)") and §3.5 if it names it.
- **The G1 draft** in verification §3.3 is the text for the user, with the hash substituted, to paste
  into RunRegistryKit's `prompts/interrupted-runs/IMPLEMENTATION_STATE.md` §2. It is not committed there.
- **What stays open:** `[03-a-sigkilled-drivers-gcs-server-deadlocks-in-shutdown]` (board §3, index
  §1.22), narrowed, with no owner.
- **For the user** (README §6.3): retiring R3's three stores; the G1 commit in RunRegistryKit; C3; and
  whether the GCS issue is assigned to a campaign.
- **The registry** holds 51 runs, none `running` or stale. The lock is free (never taken by 04) and no
  Ray process of this repository was started by it.
