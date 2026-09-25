# Prompt 05 — retire the two stores, and correct the one claim that retirement falsifies

**Campaign:** [`README.md`](README.md) · **Board items:** **R10**–**R12** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** prompts 01–04 have landed (`ff65f9f`, `226889f`, `854e2ae`, `66617c9`), each reviewed.
**Closes:** nothing. **Opens:** anything you find (§6), **without fixing it**.
**Recommended model:** **Opus**. This prompt writes no code. It is the campaign's one contact with
real stores, and the only irreversible step in it. The judgement is in checking, before and after,
that what is deleted is exactly what should be, and that the live A3 store is untouched.

**Written** 2026-09-25 at `00e0ec6`, against what 01–04 shipped. Every command and field name
below is as shipped. Where this text and a log disagree, the log is right. Stop and say so.

**Read first:**

1. [`README.md`](README.md), in full: §3, §5 rule 10, and §6.
2. [`docs/store-retirement-audit.md`](../../docs/store-retirement-audit.md) §1, §2.7 and §3.
3. The logs for prompts 03 and 04, `logs/03-retire-a-store.md` and
   `logs/04-amend-an-unknown-field.md`. Read "What shipped", the `retired` shape, the interruption
   table and "State handed to the next prompt".
4. The board's orchestrator reviews of prompts 03 and 04 (`IMPLEMENTATION_STATE.md` §1). They
   carry two points for this prompt.
5. `python -m RunRegistry store retire --help` and `store amend --help`.

---

## 0. The rule this prompt runs under

**You never write, delete or amend anything under `var/`.** README §5 rule 10: the **user** runs
every `store retire` and the `store amend`. You prepare each command, show what it will do, and
check what it did.

What you may run against `var/` is read-only:
- `python -m RunRegistry list`;
- `python -m RunRegistry store show PRIMARY`;
- `python -m RunRegistry store retire PRIMARY --reason … --dry-run`;
- reading sidecars and run manifests as JSON;
- `stat`, and a SHA-256 of any file.

A dry run takes a fresh fingerprint `mode=ro`. That is a read, and it is how prompt 03 meant this
prompt to show the user the command before the user runs it (README §4). Nothing else touches
`var/`. In particular you never run these against it:
- `store fingerprint --write`, `store adopt`, `store create`, `store copy` or `store move`;
- `store retire` without `--dry-run`, or `store amend`;
- `RunRegistry.begin`;
- `quadsource_atol_sweep.py` in any mode;
- `main.py`;
- anything that opens a store for writing.

**Never pass `--without-fingerprint`.** Both stores carry a fingerprint that matched on
2026-09-25 (`store-fingerprint` log 05, Phase B). The fingerprint code has not changed since it
was taken: `Datastore/store_inventory.py` and `fingerprint_of` / `compare_fingerprints` are
unchanged since `50a24ac`. If a dry run refuses on the fingerprint, or suggests the flag, stop
(§5).

---

## 1. What is wanted

Two stores are retired, and nothing else changes except one field of one sidecar.
- **The sweep store**, `var/datastores/handover-atol-sweep.sqlite`, with its four shards (R10).
  `docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md:15-17` then becomes true, and is not edited.
- **The backup**,
  `var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.sqlite`, with
  its four shards (R11). Its primary's `shards` rows name the **live** A3 store's shards by
  absolute path (audit §2.7). Prompt 01's resolver reads them as its own siblings, and the dry run
  must show that before the user runs anything.
- **The live A3 sidecar's `backup` field** is corrected with `store amend`, once the backup is
  retired (R11, README D5). It is the one present-tense claim a retirement makes false.
- **The records** (R12): this campaign's board, the `run-registry` board and the `handover`
  board. They are updated by additive, dated notes. The tombstones are committed as evidence,
  because `var/` is gitignored.

**The live A3 store**, `var/datastores/handover-A3-baseline-lambdacdm.sqlite` and its four shards,
**stays, byte-identical**. Its sidecar changes only by the amendment. This is the property the
whole prompt is checked against.

**Why these two may go** (audit §3):
- **The sweep store** is declared disposable in its own `purpose`. Every number it produced is
  transcribed in `QUADSOURCE-TOLERANCE-SWEEP.md`. The A3 store is to be regenerated from scratch
  (the user's decision of 2026-09-23), so none of its rows is reused.
- **The backup:** `store-fingerprint` log 05 found nothing only in the backup. Its retention
  condition, "until a resume completes cleanly", can never be met.

**What the sidecars say today,** read on 2026-09-25 while this prompt was written:

| Store | `store_id` | Recorded fingerprint digest | History |
|---|---|---|---|
| sweep | `04198f22f8704c52a252a72566af94ed` | `2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18` | one `adopt` |
| backup | `4c2ce77b0bf24671bcebd8dda3441041` | `eedcdfb2bbd12f95237e118231128c19ff793b49ab08664e06fb9072888db222` | one `adopt` |
| live A3 | `5f58ac536362424499a4e94c6242928e` | `433b7fc3cad94a971020c0493e4062e4d3a7a590a6f084c6a8e8f451f2b648a3` | one `adopt` |

**What the references reports should find.** This was predicted while writing this prompt, by
running the shipped matcher (`stores._strings`, `stores._names_one_of`) over every sidecar under
`var/datastores/`, and `runs_naming` over `var/runs/`. No store file was opened.
- **The sweep:** four runs, each finished and matched by `results`:
  - `handover--quadsource-atol-sweep-20260923T110622`;
  - `handover--quadsource-atol-sweep-lowz-20260923T132309`;
  - `handover--quadsource-atol-sweep-seam-20260923T225928`;
  - `handover--quadsource-atol-sweep-lowz-check-20260924T001705`.

  No sidecar. Prompt 03's review warned that this report could be broad, because the sweep store
  sits directly in `var/datastores/`. Today no sidecar string resolves to that directory.
- **The backup:** no run. One sidecar, the live A3 sidecar, at `$.backup.path`.

A report that differs from this is not wrong by definition, but you must explain every
difference before the user runs anything (§5).

---

## 2. Phase A — prepare and show (you)

Record everything below in the log draft, `logs/05-retire-the-two-stores.md`, **as you go**. Keep
it in the tree, never in a scratchpad. It is committed at the end of Phase C, and until then it is
the only record of what was measured.

1. **The tree.** Record the branch and `HEAD`. `git status` is clean apart from untracked paths
   that are not this campaign's (`docs/datastore-integrity-audit*`,
   `prompts/datastore-integrity/`). Leave those alone.
2. **Nothing running.** Run `python -m RunRegistry list`. Nothing may be `running`. Record the
   listing.
3. **The before-picture.** For every file under `var/datastores/`, recursively, record the size,
   the mtime and the SHA-256. Record the directory listings. This is the table Phases B and C are
   compared against. The live A3 store's five files and its sidecar are the rows that matter
   most. Also record the output of `python -m RunRegistry store show` on each of the three
   primaries.
4. **The reasons.** D3 requires one per retirement, and only a person can give it. Draft each
   from the text below, which the user may reword. Keep a reason free of single quotes, so that
   the command can single-quote it. **The reason recorded in a tombstone is final.** A completion
   refuses a different one (log 03), so the dry run, the real run and any completion must use the
   same text, character for character. Each draft below is wrapped for reading. Joined into one
   line with single spaces, it is the text.
   - **Sweep:** `Disposable working copy of the A3 baseline for
     docs/handover/quadsource_atol_sweep.py. Every number it produced is transcribed in
     docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md, and the A3 store is to be regenerated from
     scratch (user decision 2026-09-23), so none of its rows is reused. store-retirement prompt
     05.`
   - **Backup:** `Pre-resume backup of the A3 baseline. store-fingerprint log 05 found nothing in
     it that is not also in the live A3 store, and its retention condition, a clean resume, can
     never be met because the A3 store is to be regenerated, not resumed. Its purpose field, which
     says Keep, was inherited from the live store it copies. store-retirement prompt 05.`
5. **The dry runs.** For each store, run:
   ```bash
   ./venv/bin/python -m RunRegistry store retire <PRIMARY> --reason '<REASON>' --dry-run
   ```
   Run it from the repository root, with the default roots, `var/runs/` and `var/datastores/`.
   Each must exit 0, and you record its output verbatim. Check each of these, and record the
   check:
   - **`files to be deleted:`** is exactly the store's four shards, in ascending serial, then its
     primary. **For the backup, every path is inside `backup-pre-resume-20260921T091011/`.** Any
     path naming `var/datastores/handover-A3-baseline-lambdacdm*` is the audit §2.7 danger, and is
     a stop. The resolver's `!!` notice, "reading them as siblings in … instead", is expected on
     the backup's output and should be quoted in the log.
   - **`fingerprint check:`** is `matched`, with the recorded digest in the table above.
   - **`references:`** is §1's prediction, and names both roots.
   - **`tombstone:`** carries the reason exactly, state `retiring`, the five files, and the
     references.
6. **Nothing changed.** Re-take step 3's before-picture. It must be identical, including every
   mtime. A dry run that changed a byte under `var/` is a stop.
7. **Hand back.** End Phase A by returning, for the orchestrator to show the user:
   - the two dry-run outputs;
   - the two reasons;
   - the two exact commands, the dry-run commands without `--dry-run`, **sweep first**. The sweep
     goes first because it is disposable and has no present-tense reference, so the tool's first
     real use is on the store with the least at stake. The backup goes second, after the sweep's
     result has been checked;
   - what each command will delete, and that the live A3 store is not among it.

   If the user rewords a reason, run that dry run again with the final text before the user runs
   the command.

---

## Between A and B — the user retires the two stores

The orchestrator shows the user Phase A's result. **The user**, not you:
1. runs the sweep's `store retire`. The orchestrator checks the result against the before-picture:
   the sweep's five files are gone, its sidecar changed, and nothing else changed;
2. runs the backup's `store retire`, with the same check. The backup's five files are gone, its
   sidecar changed, and nothing else changed. **The live A3 store's five files are byte-identical.**

A command that exits 1 with "Nothing was written or deleted" is a refusal. Nothing happened, and
the sequence stops there. One that exits 1 with "failed at step" is an interrupted retirement. Its
tombstone reads `retiring`, and the remedy is the same command with the same reason (log 03,
interruption table). That remedy is the user's to run, after you have read the state in Phase B.

---

## 3. Phase B — check the retirements, and prepare the amendment (you)

1. **Nothing running,** again: `python -m RunRegistry list`.
2. **The tombstones.** Run `store show` on each retired primary. Each must exit 0, begin with
   `retirement:`, and read as a **completed tombstone**, which means:
   - state `retired`, with `completed` set;
   - the reason, exactly as given;
   - the fingerprint condition `matched`, with the digest above;
   - `files` equal to the dry run's `files to be deleted:`;
   - `references` equal to the dry run's;
   - a history that ends in one `retire` entry after the `adopt`.

   Read each sidecar as JSON too. Its `fingerprint` field, and every field but `history` and
   `retired`, must be value-identical to the before-picture. Record the check, field by field.
3. **The files.** Compare the after-picture with the before-picture. The only differences allowed
   are:
   - the ten store files, which are gone;
   - the two retired sidecars, which have changed;
   - the directory entries that follow from those.

   The live A3 store's five files and its sidecar are **byte-identical, mtimes included**. The
   backup's directory now holds only its sidecar. Anything else is a stop.
4. **An interrupted retirement,** if either reads `retiring`. Record exactly what `store show`
   names as remaining. Hand back the one remedy: the same command, with the same reason. Do not
   go on to step 5 until both are completed tombstones.
5. **The amendment.** Draft the command that corrects the live A3 sidecar's `backup` field. Today
   the field reads:
   ```json
   {"path": "var/datastores/backup-pre-resume-20260921T091011", "reason": "The resume did NOT succeed. Backup kept until a resume completes cleanly.", "retained": true}
   ```
   What it should say is the user's judgement (D5). Offer this as the draft:
   - **keep `path`**, so that a reader who follows it lands on the tombstone;
   - set `retained` to `false`;
   - replace `reason` with why the backup is gone;
   - add `retired`, the backup tombstone's `retired.completed`;
   - add `tombstone`, the backup sidecar's repository path.

   Nothing is lost: the old value is recorded verbatim in the `amend` entry's `before` (log 04).
   Fill in the real timestamp, and keep the JSON free of single quotes:
   ```bash
   ./venv/bin/python -m RunRegistry store amend var/datastores/handover-A3-baseline-lambdacdm.sqlite --field backup --json '{"path": "var/datastores/backup-pre-resume-20260921T091011", "retained": false, "retired": "<retired.completed>", "tombstone": "var/datastores/backup-pre-resume-20260921T091011/handover-A3-baseline-lambdacdm.manifest.json", "reason": "Retired with store retire (store-retirement prompt 05). The resume it guarded will never happen, because the A3 store is to be regenerated, not resumed, and store-fingerprint log 05 found nothing in it that is not also in this store."}' --reason 'The backup this field describes was retired with store retire, so retained true is no longer true. store-retirement prompt 05.'
   ```
   There is no dry run for `store amend`. Check that the `--json` argument parses with
   `json.loads`, and that the value differs from the current one. Then hand back the command for
   the orchestrator to show the user.

---

## Between B and C — the user amends the live sidecar

**The user** runs the amendment, and the orchestrator checks that exactly one file changed: the
live A3 sidecar. It exits 0 and prints `>> amended:`. A refusal (exit 1) writes nothing, and the
sequence stops there.

---

## 4. Phase C — check the amendment, and record (you)

1. **The amendment.** Read the live A3 sidecar as JSON, and compare it with the before-picture:
   - `backup` is the value the user gave;
   - `history` is the old history plus one `amend` entry, with `field` `backup`, the reason, a
     `before` of `{"present": true, "value": <the old backup, verbatim>}`, and the new value in
     `after`;
   - every other field is value-identical, the `fingerprint` included;
   - the live store's five files are byte-identical to the before-picture, mtimes included.

   Run `store show` on it. It exits 0, and prints `kind:     registry` and `problems: none`.
2. **The final picture.** Re-take the before-picture. The differences from Phase A are exactly
   Phase B's plus the live sidecar. Record the final hashes of the live store's five files beside
   Phase A's.
3. **The evidence file.** Write `logs/05-tombstones.json`, holding verbatim, as read from disk:
   - the sweep's tombstone sidecar;
   - the backup's tombstone sidecar;
   - the live A3 sidecar after the amendment.

   `var/` is gitignored, so this file is what survives in git, as `store-fingerprint`'s
   `logs/05-fingerprints.json` does for the fingerprints.
4. **The records (R12).** Each is an additive, dated note. Never rewrite what a board already
   says: it was true for the tree it was written on (`CLAUDE.md` campaign invariant 6).
   - **This board.** The §1 row for 05, items R10–R12, and §5 (no code changed, so the suites are
     not re-run; say so). The header's status says the campaign is complete.
   - **The `run-registry` board**, `prompts/run-registry/IMPLEMENTATION_STATE.md`. Add a dated note
     beside the paragraph that describes the retained backup (the one beginning "A second
     instance, found when the campaign was planned").
     - The note says that the backup was retired on this date, and that its directory now holds
       only its tombstone.
     - It explains `var/runs/a3-pilot/BACKUP_PATH`. That file names the backup's directory. It is
       a pre-registry record of the past, and it is not edited: `var/runs/` is out of this
       campaign's scope (README §1), and D5 keeps records of the past. Following it now lands on
       the tombstone.
   - **The `handover` board**, `prompts/handover/IMPLEMENTATION_STATE.md`. Add a dated note to the
     `[a3-baseline-quadsource-integrals-are-1680-short]` entry, after its "Superseded" paragraph.
     It says two things:
     - The sweep store, whose first run wrote the 54 rows, has been retired.
     - `docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md:15-17` ("deleted after this document was
       written") is now true, and was not edited.
   - **`docs/OPEN_ISSUES.md`**, only if you open an issue.
5. **Commit.** One commit, holding:
   - the log;
   - `logs/05-tombstones.json`;
   - the three boards;
   - the index, if an issue was opened.

   Use the `CLAUDE.md` form. The subject is a short imperative phrase, for example "Retire the
   sweep store and the A3 backup". Stage by name, never with `git add -A`.

---

## 5. Stop conditions — stop and hand back

At any of these, stop, write nothing more, and report what you saw:
- anything is `running`, at any phase;
- a dry run refuses, for any reason, or its fingerprint check is not `matched` with the recorded
  digest, or it suggests `--without-fingerprint`. The fingerprint code is unchanged, so a mismatch
  means a store's content changed since 2026-09-25, and that must be explained before anything is
  deleted;
- the backup's `files to be deleted:` names any file outside its own directory;
- a references report differs from §1's prediction in a way you cannot explain from the sidecars
  and manifests;
- a dry run changes anything under `var/`;
- after a user step, anything under `var/` differs from what that step should change. Above all,
  any change to the live A3 store's five files;
- a tombstone reads as anything but a completed tombstone after its remedy has been run;
- this prompt and a log disagree about a command, a field or a behaviour.

---

## 6. What this prompt does not do

- It writes no code, and so no tests and no mutations. The log's deliberate-breakage section says
  so, and says why.
- It changes no run manifest, no file in `var/runs/`, and no committed document outside the three
  boards and the index. `QUADSOURCE-TOLERANCE-SWEEP.md` is **not** edited.
- It does not amend the retired sidecars. They are tombstones, and amend refuses them. Their
  unknown fields are records of the past, including the backup's `restart.command`, which names
  the live store.
- It does not close `[00-a-copy-carries-its-sources-present-tense-fields]` or any other open
  issue.
- It does not touch the live A3 store, apart from the one amendment to its sidecar, which the
  user runs.

---

## 7. The log

`logs/05-retire-the-two-stores.md`, using the template in README §5.1. In addition, it records:
- Phase A:
  - the before-picture, with the live A3 store's rows in full;
  - the `list` output;
  - the two dry-run outputs, verbatim;
  - the check against §1's prediction;
- the user's commands as the user ran them, with their exit codes;
- Phase B:
  - the `store show` outputs;
  - the field-by-field check;
  - the after-picture's differences;
- Phase C:
  - the amendment check;
  - the final hashes of the live A3 store's five files, beside Phase A's.
