# Prompt 09 — close the campaign

**Campaign:** [`README.md`](README.md) · **Board items:** **Z1**–**Z5** (replacing the board's one
Z1 row) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-16, after the review of 08 (`33a7346`), as README §2.1 holds it. Its
orchestrator note, written at the same time, hands it the counts and the facts checked at
`33a7346`, and governs where it differs from this file.
**Gate:**
- 08 landed and reviewed, with **R2 and RunRegistryKit's G2 holding** on this board's §2;
- `git status` is clean;
- nothing is `running` in the registry, and no Ray process of this repository is up.

**Closes:** nothing. **Narrows:** `[00-open-issues-about-the-registry-code-stay-on-sgk-boards]` and
`[06-runregistrykit-list-into-a-closed-pipe-ends-in-a-traceback]` (§1, Z3), and the four issues the
first names, each on its own board. **Opens:** nothing expected.
**Recommended model:** **Sonnet**. Records, prose and measurements; no code.

**Read first:**

1. [`README.md`](README.md): §2.2 (09's charter), §3, §5, §6.2 (C11) and §6.3.
2. This board: the header, §1, §2, §3 and §4, and §5.10–§5.14 with their reviews.
3. [`docs/runregistry-extraction-verification.md`](../../docs/runregistry-extraction-verification.md):
   the header, §1–§3's opening paragraphs, §2.10, §3.10 and §3.11.
4. The four issues `[00-open-issues-about-the-registry-code-stay-on-sgk-boards]` names, each on its
   own board:
   - `[04-runs-do-not-say-which-machine-produced-them]`: `prompts/run-registry/IMPLEMENTATION_STATE.md` §3;
   - `[00-a-copy-carries-its-sources-present-tense-fields]` and
     `[01-cross-filesystem-move-advice-says-delete-by-hand]`:
     `prompts/store-retirement/IMPLEMENTATION_STATE.md` §3;
   - `[04-a-signal-inside-a-heartbeat-write-can-leave-status-json-unparseable]`:
     `prompts/a3-v2-readiness/IMPLEMENTATION_STATE.md` §3.

   And two more on those boards that name registry behaviour, to be **read and placed** (Z3):
   `[00-a-sigterm-pipeline-run-is-recorded-as-failed]` (store-retirement) and
   `[04-a-sigterm-during-ray-init-meets-rays-handler-not-the-drivers]` (a3-v2-readiness).
5. **RunRegistryKit**, `/Users/ds283/Documents/Code/RunRegistryKit`, read with `git`, never
   changed: the package at the tag `v0.1.0`; its `docs/OPEN_ISSUES.md`; its extraction board's
   header, §2 (the gates) and one §3 entry, for their form; and its SGK checklist,
   `docs/adoption/secondarygwkit.md`, §9.
6. **DatastoreKit**, `/Users/ds283/Documents/Code/DatastoreKit`, read the same way: its `fingerprint`
   campaign's board §2 and README §7; and, as the precedent for a hand-over of issues,
   `docs/OPEN_ISSUES.md` §1.2 and `prompts/inherited-issues/README.md`.
7. `CLAUDE.md`: the open-issues rule, the package section and the run-registry rules;
   [`prompts/INDEX.md`](../INDEX.md); [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) §1.11,
   §1.14, §1.17 and §1.21.

---

## 1. What is wanted

The packages are adopted and every gate holds. 09 leaves the repository saying so; drafts what the
user hands to RunRegistryKit and DatastoreKit; and says what the A3 v2 build inherits.

- **Z1 — the final measurements**, at your `HEAD`, saved under `measure/out/09-*`:
  - the seven roots, serially, against 08's counts;
  - `pip freeze` against 05's after-install freeze (`measure/out/05-pip.txt`: its "before" list
    plus the one `runregistrykit` line its diff adds), `pip check`, and
    `runregistrykit.package_provenance(["datastorekit", "raypoolkit", "runregistrykit"])`, which
    must equal, as JSON, R2's manifests' `packages`;
  - every `RunRegistry` reference outside `prompts/` and dated `docs/` prose: RunRegistryKit's
    `docs/adoption/measure_client_registry.py sgk --commit <HEAD>`, run from its root with its own
    venv, and verification §3.11's `git grep`, each with its output;
  - `python -m runregistrykit list`, and `store show` on every sidecar under `var/datastores/`.
- **Z2 — the two gate records, drafted** (README §6.2 C11), in verification §4, each a fenced block
  ready to paste, in its board's own form, naming the SGK commits and runs each part rests on:
  - **RunRegistryKit's G2**, the cell of its extraction board's §2 row, in the form of that table's
    G1 and E1 cells, keeping the checklist pointer. Map each of its checklist §9's clauses to
    verification §3.11's evidence and Z1's.
  - **DatastoreKit's**, the record its `fingerprint` board §2 says it takes "when SGK reports it":
    R1's fingerprint both ways (04, verification §2.10), and what adopted the release since: the
    pin (03), every fingerprint SGK takes going through `datastorekit.store_fingerprint` with
    `encoding_format` recorded (06), and R2's W1 across stores (08, verification §3.10). In the form
    of the dated italic note already in that section.

  Neither is committed in its repository. The user makes those commits, or has a session there
  make them.
- **Z3 — the registry issues, handed over.**
  - **Place each.** For each of the four and the two of "Read first" item 4, show from the code
    whether the defect is in `runregistrykit` at `v0.1.0` or in SGK's own code (for instance
    `docs/gktk-remedial/scoped_pipeline_run.py`), and if in the package, where, by file and line at
    the tag. Say whether it is still present there. The two of item 4 are expected to be SGK's; if
    either is the package's, it joins the hand-over.
  - **Draft the hand-over** as a new file in this folder,
    `runregistrykit-inherited-issues-brief.md`, for the user to hand to a session in RunRegistryKit.
    It holds the package's issues, the four and
    `[06-runregistrykit-list-into-a-closed-pipe-ends-in-a-traceback]`, each **complete**, in
    RunRegistryKit's board-entry form: its SGK name and board, when it was opened, the defect with
    the package's lines at `v0.1.0`, the measurement it rests on, its impact, and its next step. It
    proposes one `docs/OPEN_ISSUES.md` row for each, in RunRegistryKit's form. It says which of the
    six were found to be SGK's and why, so that they are not asked for. Where they go in
    RunRegistryKit (a board, a campaign) is RunRegistryKit's to decide; the brief may cite
    DatastoreKit's `inherited-issues` board as the precedent, and no more.
  - **Record it on SGK's boards.** A drafted issue is **not** transferred: RunRegistryKit does not
    yet list it. So each of the four entries **stays in its board's §3**, and gains one line,
    `**Drafted for RunRegistryKit (2026-10-16, runregistry-extraction prompt 09):**`, giving the
    package's file and line at `v0.1.0` and the brief, and saying that the entry moves to §4 as
    transferred when RunRegistryKit's `docs/OPEN_ISSUES.md` lists it. Nothing else in the entry
    changes. Its index row's hook gains the same fact in a clause, and any path in the hook that
    names `RunRegistry` code now names the package's.
  - **This board's two** stay in §3, each with a `**Narrowed (2026-10-16, prompt 09):**` line: what
    09 did, and that what remains is the user's hand-over, after which both close here and the
    four move to their boards' §4. Their `**Assigned**` lines stay as written; the narrowing says the
    assignment is discharged. Their index rows' hooks say the same.
- **Z4 — what A3 v2 inherits**, a subsection of verification §4, for the campaign that runs it
  (`handover`, which owns the A3 v2 launch, `a3-v2-readiness` README §1). From the tree and the
  records, not from memory:
  - **the launcher**: `docs/handover/quadsource_atol_sweep.py --build --register`, at its hash at
    your `HEAD`, the flags R2 ran it with that a production build drops (`--rehearsal`), and the
    `address="local"` it forces;
  - **the commands**, as R2 ran them (verification §3.1 and §3.2): discovery,
    `store create`, `store fingerprint --write`, `store show`, `store retire`, and the lock's five
    commands, each in `runregistrykit`'s or `tools.rehearsal_lock`'s spelling;
  - **its store** is the first live store recorded in the new layout meant to be kept: every
    sidecar under `var/datastores/` but R2's two is a tombstone, and R2's two are throwaway
    (README §3);
  - **what bears on it**, each pointing at its record: the issues open on boards that touch its
    launch (at least `[00-main-and-the-readers-default-to-ray-address-auto]`, assigned to
    `handover`, and `[07-a-killed-builds-ray-agents-outlive-the-driver]`), and the machine's disk,
    which 08's review found past Ray's 95 % spilling threshold (board §5.14), measured again here.
- **Z5 — the records.**
  - **Verification §4, "The close"**: §4.1, the three rehearsals as one table (twelve runs: tree,
    packages, fingerprint and end state, read from each run directory's `status.json` and
    `manifest.json`); §4.2, Z1; §4.3, Z2's drafts; §4.4, Z3's placement and a pointer to the brief;
    §4.5, Z4.
  - **This board closes**: the status line; §1's row 09 and the items table, its Z1 row replaced
    by Z1–Z5; §2 unchanged but for any closing line; §3 as Z3 leaves it; and a final
    `### 5.15 After prompt 09 (2026-10-16)` listing every issue this campaign leaves open, with
    where each now stands, and what is left to the user (README §6.3): R2's two stores, the two
    gate commits, and the issue hand-over.
  - **README §2's table** marks every prompt landed, with its commit, in the board's §1 form.
  - **`prompts/INDEX.md`**: this campaign's row closed, in the form of its closed neighbours, with
    its open-issue column taken by counting; the header's campaign counts and date.
  - **`docs/OPEN_ISSUES.md`**: the six rows Z3 changes, §1.21's lead-in saying what 09 did, and the
    header's count and date, corrected by counting.

**Left for the user, not done here** (README §6.3): retiring R2's two stores, or not; the two gate
commits; handing the brief to RunRegistryKit. After the hand-over, a records commit here moves the
four to their boards' §4 as transferred and closes this board's two.

---

## 2. Verification

1. **Z1's measurements** are each saved whole, with the command that made it, under
   `measure/out/09-*`. The measure and the `git grep` read the committed tree, so taken at your
   `HEAD` they measure the tree you started from, which this prompt changes only in records: say so.
2. **Z3's placement** is shown with the lines it rests on, at `v0.1.0` for the package (`git show
   v0.1.0:<path>` in RunRegistryKit) and at your `HEAD` for SGK.
3. **The boards.** For each of the three other boards, `git diff -U0` holds the one added line per
   entry and nothing else.
4. **The index.** `docs/OPEN_ISSUES.md`'s header count equals its rows by counting
   (`grep -c '^| \`\['`), and each changed row agrees with its board. `prompts/INDEX.md`'s row
   agrees with this board.
5. **Z4 and §4.1** are read from the tree and the run directories, each figure with its source.
6. **The suites**, all seven, at 08's counts.

---

## 3. Acceptance

1. **Z1–Z5** as §1 says.
2. The measure reports 0 in every live-code and staying-test figure, and the `git grep` prints only
   `CLAUDE.md`'s and `config/fingerprint.py`'s lines naming the old package's history.
3. `pip freeze` equals 05's after-install freeze; the provenance equals R2's manifests'.
4. The index's count and date agree with its rows; every board edited agrees with the index.
5. The seven roots pass at 08's counts.
6. **No code, no test, no breakage**: 09 changes no file a build, a test or a script reads.
7. **One commit**, with the log `logs/09-close-the-campaign.md` in README §5.1's form.

---

## 4. Stop conditions — stop and ask

- R2 or RunRegistryKit's G2 does not hold on this board when you start.
- An issue's defect can be placed in neither the package nor SGK, or is in both in a way a single
  entry cannot say.
- One of the four is no longer present in `runregistrykit` at `v0.1.0`.
- Any count moves, or `pip freeze` differs from 05's.
- Anything would need an edit in RunRegistryKit, DatastoreKit or RayPoolKit, or a change to code
  here.

---

## 5. What this prompt does not do

- **Files it changes:** `docs/runregistry-extraction-verification.md` (§4 appended); the three
  other boards (`run-registry`, `store-retirement`, `a3-v2-readiness`), one line per entry;
  `docs/OPEN_ISSUES.md`; README §2's table; this board; `prompts/INDEX.md`. **Files it adds:** the
  brief, the `measure/out/09-*` files and the log. Nothing else.
- It runs no build, opens no store but through `store show`, retires nothing, takes no lock and
  starts no Ray.
- It edits nothing in RunRegistryKit, DatastoreKit or RayPoolKit (README §5 rule 12), and installs
  nothing.

**One commit**, in `CLAUDE.md`'s form, staged by explicit path only, after showing
`git diff --cached --name-only`.
