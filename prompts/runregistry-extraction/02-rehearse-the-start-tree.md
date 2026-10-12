# Prompt 02 — rehearse the start tree (R0)

**Campaign:** [`README.md`](README.md) · **Board items:** **L1**–**L4** (R0) ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-12, with the plan and prompt 01, at `c8a7f14`.
**Gate:**
- 01 landed and reviewed: `RunRegistry/` frozen, the baselines saved (board, 01's row and §2.1);
- `git status` is clean, and no commit after 01's touches code (anything outside `prompts/`);
- `python -m RunRegistry list` shows nothing `running`;
- the venv holds `datastorekit` 0.2.2 at `6d8ee9ae…` and `raypoolkit` 0.2.0 at `eaef0ce…`, neither
  editable, and no `runregistrykit`.

**Closes:** nothing. **Opens:** anything you find, **without fixing it**.
**Gate it decides:** **R0** (board §2).
**Recommended model:** **Opus**. Four registered builds and the readers under Ray. The work is in
reading what they did against `dk-and-rpk-migration`'s R2.

**Read first:**

1. `CLAUDE.md`, "Long-running jobs — the run registry" and the freeze section 01 added, in full. The
   registry's rules bind every launch here.
2. [`README.md`](README.md): §0.1, §0.2 ("Since the last rehearsal"), §2.2 (02's charter), §3,
   **§4.1**, §5 (rules 9–11, 14, 15).
3. **`dk-and-rpk-migration`'s rehearsal prompts, which this prompt runs:**
   [`08-rehearse-through-both-packages.md`](../dk-and-rpk-migration/08-rehearse-through-both-packages.md)
   in full, and through it
   [`05-rehearse-through-datastorekit.md`](../dk-and-rpk-migration/05-rehearse-through-datastorekit.md)
   §1–§2 and [`02-rehearse-the-start-tree.md`](../dk-and-rpk-migration/02-rehearse-the-start-tree.md)
   §1. Where this prompt is silent, 08 governs, and through it 05 and 02.
4. **[`docs/dk-and-rpk-migration-verification.md`](../../docs/dk-and-rpk-migration-verification.md)
   §3**, its R2: how each launch went, the commands verbatim, the interrupter's points and reasons,
   and L4's tables. Your §1 has the same shape, and is compared with it.
5. `docs/dk-and-rpk-migration/rehearsal/`: the six scripts, and R2's saved outputs (`*_08*.txt`),
   which are what yours should look like.
6. That campaign's board, §3: `[02-the-policy-statistics-block-orders-its-lines-by-completion]` and
   `[02-07s-store-soundness-calls-read-inventory-without-its-registry]`; and this repository's
   `[07-a-killed-builds-ray-agents-outlive-the-driver]` (`docs/OPEN_ISSUES.md`).
7. 01's log, "State handed to the next prompt".

---

## 0. What this prompt is

`dk-and-rpk-migration`'s R2, at `f539015`, rebuilt the reference store through both packages and
reproduced `39809dca…`. Since then `511ef5f` moved the object factories from
`Datastore/SQL/ObjectFactories/` to `ObjectFactories/`, and `7d76d3e` moved the store fixtures. No
build has run through either (README §0.2). The factory move changes the module path of every class
that Ray pickles and every factory the pool is handed.

**R0 is that R2 again, at 01's tree.** If R0 reproduces `39809dca…`, the start tree is sound, and
R1 can isolate the DatastoreKit pin. If it does not, the miss is this repository's, and nothing in
this campaign moves until it is explained.

**What this prompt is not.** It changes no code and no test. Until the last launch has ended it
creates no file in the repository outside `var/`, so that every manifest records
`git_dirty: false`. Working notes go under `var/runregistry-extraction-02/`.

---

## 1. The protocol, with this prompt's names

**Run `dk-and-rpk-migration` 08's §1 as written** (and through it 05's §1 and 02's §1), with these
substitutions:
- **Stores:** `var/datastores/runregistry-extraction-02-reference.sqlite` and
  `var/datastores/runregistry-extraction-02-resumed.sqlite`. Before the first launch, `store show`
  on each gives kind `absent` and no runs; if not, stop.
- **Runs:** `--campaign runregistry-extraction --prompt 02`, slugs `reference`, `resumed-a`,
  `resumed-b`, `resumed-c`, and `--job-name runregistry-extraction-02` on every launch. Each
  `--purpose` names R0 of this campaign, the start tree's commit, and the fingerprint it must equal.
- **Launch logs:** `var/runs/runregistry-extraction-02-<slug>.launch.log`.
- **Reader outputs:** `var/runregistry-extraction-02/extract/`.
- **Saved outputs** you commit go to **`docs/runregistry-extraction/rehearsal/`**, named as
  `dk-and-rpk-migration`'s with `02` (`prelaunch_checks_02.txt`, `stage_counts_02.txt`, …).

What holds from 08, unchanged:
- **The launch command** is README §4.1's (`dk-and-rpk-migration` README §4.1's, verbatim but for
  the store, the slug, the purpose and `--resume`).
- **The pre-launch checks**, finding `datastorekit` 0.2.2 and `raypoolkit` 0.2.0 and no
  `runregistrykit`.
- **One check after each launch, then end your turn.** The orchestrator waits on the run's
  liveness and resumes you. Never poll; never wait with `pgrep -f`.
- **After a killed launch**, wait by pid for its Ray agents before the next launch, and record how
  long they took, beside R2's.
- **The interrupter's two dummy tests** run before the first launch, as at R2.
- **The interruption points are R0's own**, chosen from R0's `reference` `stdout.log` by the same
  reasoning, and recorded with it.
- **Sidecars** as R2's: `store create` on each closed primary, and `store fingerprint --write` on
  the reference after `reference`.

**The scripts.** `docs/dk-and-rpk-migration/rehearsal/`'s six scripts run **as they stand at 01's
tree**, from where they are; they are not copied (README §6.2 C10). If one fails in a way its R2 run
did not, stop: a script broken by `511ef5f` or `7d76d3e` is a finding for the user, and is not
repaired here.

---

## 2. What to verify

**L1–L3 are 08's §2**, on this prompt's stores and runs, with these differences.
- **The manifests.** Every `packages` is exactly R2's record (08 §2: `datastorekit` 0.2.2 at
  `6d8ee9ae…`, `raypoolkit` 0.2.0 at `eaef0ce…`, both `editable: false`), and every `git_dirty` is
  `false`.
- **The fingerprints.** Both stores `39809dca…`, format 6, 21 classes, each equal to its sidecar's;
  `compare_fingerprints` between them empty; `compare_with_recorded.py` against **R2's run
  directories** `var/runs/dk-and-rpk-migration-08-reference-20261011T224347` and
  `…-08-resumed-c-20261011T230717`, each empty. R2's stores are retired (`718e650`); their run
  directories' `status.json` hold the recorded fingerprints.
- **What is new in the logs** is compared with R2's same slug.

**L4 — equal to `dk-and-rpk-migration`'s R2.** Against its verification §3:
- **The statistics block**: `policy_statistics.py` on R0's `reference`, against R2's, compared as
  counts and as sorted lines (the issue in Read first, item 6). **Every count must be R2's** (291;
  numeric 109; mixed 26, 25 complete and 1 minimal; WKB 156). The same for each resumed run that
  prints the block.
- **The reference run's stage counts** equal R2's, stage by stage.
- **The resumed chain's identities** hold, as 05 §2 states them.
- **The stage sequence**, the `** ` banner lines of the `reference` `stdout.log` in order, equals
  R2's.
- **The printed lines and the registry's fields** as R2's (08 §2, "The printed lines are an
  interface").
- **The stage times**, beside R2's. Not compared.

A difference in L4 with L1–L3 holding is a finding to explain from `511ef5f` or `7d76d3e`, and to
open as a §3 issue. The fingerprint is the acceptance.

---

## 3. Acceptance

1. **L1–L3** hold as §2 says. Both stores equal each other, `39809dca…`, and R2's recorded
   fingerprints.
2. Every manifest's `packages` is R2's record, and every `git_dirty` is `false`.
3. **L4** against R2: the statistics counts, the reference stage counts, the chain's identities,
   the stage sequence, and the printed lines and registry fields, as R2's, or each difference
   recorded, explained and opened as an issue.
4. **The seven roots** pass at the dispatch `HEAD`, with 01's counts.
5. `docs/runregistry-extraction-verification.md` exists, with **§1, "R0 — the start tree"**, in the
   shape of `docs/dk-and-rpk-migration-verification.md` §3.
6. **The board records R0 as holding**, naming the two runs that finished `done`.
7. **No deliberate breakage**: 02 adds no code and no test.

---

## 4. Stop conditions — stop and ask

**08 §4's conditions apply**, with R2 in place of R1, and:
- **A fingerprint differs from `39809dca…`.** Record both fingerprints and `compare_fingerprints`,
  change nothing, relaunch nothing, stop. At R0 the cause is this repository's; the board's §3 gets
  an entry naming the classes that differ.
- **A build or a reader fails on an import or an unpickling** of a factory or a class: the factory
  move's signature.
- **A rehearsal script fails** where its R2 run did not.
- A manifest's `packages` is not R2's record, or a `git_dirty` is `true`.

---

## 5. What this prompt does not do

- It changes no tracked file but the records of §6.
- It writes no store under `var/` but its two, and retires none.
- It does not fix what it finds.

---

## 6. The log, the verification record, and the board

**The files you may change, and nothing else:**
- `docs/runregistry-extraction-verification.md`, new, with a short header (campaign, purpose, how it
  grows: §1 R0 here; §2 R1 at 04; §3 R2 at 08; §4 the close at 09) and **§1**, with the subsections
  of `docs/dk-and-rpk-migration-verification.md` §3 in the same order;
- `docs/runregistry-extraction/rehearsal/`: saved outputs named with `02`;
- the log, this board and `docs/OPEN_ISSUES.md`.

**The log**, at `logs/02-rehearse-the-start-tree.md`, in README §5.1's form, with what
`dk-and-rpk-migration` 08's log records, against R2.

**The board:** the §1 row for 02, items L1–L4 (R0), the §2 row R0, and
`### 5.2 After prompt 02 (<date>)`; a §3 entry for anything opened, with `docs/OPEN_ISSUES.md`
§1.21 in the same commit, the count and date corrected by counting.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
