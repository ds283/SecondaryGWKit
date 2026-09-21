# Prompt 01 — the `QuadSourceIntegral` read-back, and the guard that should have caught it

**Campaign:** [`README.md`](README.md) · **Board item:** **R1** ·
**Board:** `IMPLEMENTATION_STATE.md` — **does not exist; this prompt creates it** (§8).
**Closes:** nothing yet indexed — this defect is found by this campaign and closed by it.
**Opens:** whatever the §3 audit finds, **without fixing any of it**.
**Recommended model:** **Opus**. The fix is one line. The guard is the work, and deciding what it
may not catch is the judgement.

**Read first:**

1. [`README.md`](README.md) **§0 in full** — the failure, how it was found, and why the one-line
   fix is not the point.
2. `Datastore/SQL/ObjectFactories/QuadSourceIntegral.py` — **all of it**. In particular the
   `SELECT` at `:234`, `build()` at `:324`, and the *other* `SELECT` at `:530`, which does include
   the column. Understand why one has it and the other does not before you change either.
3. `Datastore/SQL/Datastore.py` `object_get` (`:512`) — the caller, and the path a resume takes.
4. `ComputeTargets/tests/test_main_plumbing.py` — `load_main_py_functions`, for the **`ast`
   precedent**: this tree already reads source statically in a test rather than executing it.
5. `CLAUDE.md` — repository mechanics, in particular that tests live in `<package>/tests/`, run
   from the repository root, and **must not need Ray or a datastore**.

---

## 1. The defect, stated precisely

`sqla.select(...)` at `:234` builds `row_data`. `build()` at `:324` reads
`row_data.numeric_quad`. The column is in the table, is written correctly, and is absent from that
`SELECT`'s column list. Any read-back of a stored `QuadSourceIntegral` therefore raises
`NoSuchColumnError`, and always has.

The evidence it has never been exercised: a 10 h 33 m run wrote 7,552 such rows without difficulty,
and the first attempt to resume against them failed in under two minutes.

---

## 2. What to fix

Add the missing column to the `:234` `SELECT`. **That is the whole fix.** Do not reorganise the
query, do not factor the two `SELECT`s together, do not touch the schema, and do not alter what
`build()` reads.

Then confirm it, against the real store, **without starting a long run**: read back one stored
`QuadSourceIntegral` row through the same `object_get` path the resume takes, and show that it
now succeeds and returns the stored `numeric_quad`. A short throwaway script under your scratch
directory is the right vehicle; it is not a deliverable and must not be committed.

**The datastore is `var/datastores/handover-A3-baseline-lambdacdm.sqlite`, 4 shards.** Read it,
do not write to it. **Do not delete `var/datastores/backup-pre-resume-20260921T091011`**, and do
not start the pipeline: finishing that run is the user's, after this lands.

---

## 3. The guard, which is the actual work

A test that would have caught this, in a new `Datastore/tests/`. **No Ray. No datastore.** The
constraint is real and it shapes the answer.

Two candidate shapes. Choose one, implement it, and justify the choice in the log:

- **(a) Static consistency.** For each object factory, collect the attribute names `build()` reads
  off its `row_data`, collect the columns its `SELECT` requests, and assert the first is a subset
  of the second. `ast` over the module source, in the manner of `load_main_py_functions`. Catches
  the whole class, needs no database, and is the reason this prompt exists.
- **(b) In-memory round trip.** Build the factory's own table in an in-memory SQLite, insert a row,
  read it back through `build()`. Direct and unambiguous for one factory; does not generalise
  without writing it 12 times.

**(a) is recommended.** It is the one that would have caught this defect *before* it cost a day,
and the one that catches the next one. But it is a static approximation and you must say plainly,
in the test's own docstring and in the log, **what it cannot see** — for example attributes read
through `getattr`, columns added by `add_columns` at `:632`/`:641` on a conditional path, or a
`build()` whose `row_data` comes from somewhere other than the module's own `SELECT`. A guard that
oversells its coverage is worse than none.

If you implement (a) and it cannot be made to work honestly — the static analysis is too
approximate to be sound, or it produces false positives you would have to special-case away — say
so and fall back to (b) **for `QuadSourceIntegral` only**, recording the failure of (a) as an
observation. Do not ship a static check you have had to weaken until it passes.

---

## 4. The audit — open, do not fix

Run your guard across all 22 factories. **12 read `row_data` attributes** and are the candidates.

For every additional defect it finds: **do not fix it.** Record it in the log and open a §3 issue
naming the file, the attribute, the `SELECT` that omits it, and whether the path is reachable
(is there a resume or an `object_get` that would hit it?). A sweep that quietly repairs eleven
factories in the same commit cannot be reviewed, and this campaign's README §4 forbids it.

If the guard finds **nothing** beyond `QuadSourceIntegral`, that is a result and should be stated:
one defect, now fixed, with a guard against recurrence.

---

## 5. What this prompt does not do

- It does not change the schema or migrate anything. The stored data is correct.
- It does not fix any other factory.
- It does not touch `ComputeTargets/`, `main.py`, `config/`, `docs/spec/`, or any physics.
- It does not run the pipeline, finish the interrupted run, or delete the backup.
- It does not refactor the two `SELECT`s into one, however tempting. That is a separate change with
  its own risk, and it would obscure a one-line fix in the diff that proves it.

---

## 6. Acceptance

1. The `:234` `SELECT` requests `numeric_quad`; the diff of that file is one line.
2. One stored `QuadSourceIntegral` row reads back through `object_get`, demonstrated, with the
   `numeric_quad` value quoted.
3. A test exists in `Datastore/tests/` that **fails on the unfixed code and passes on the fixed
   code** — demonstrate both directions, as a deliberate-breakage record in the log. A guard not
   shown to fail on the bug it was written for is not known to work.
4. The test needs no Ray and no datastore.
5. Its docstring and the log state what it cannot see.
6. The audit has been run across all 22 factories; findings opened as §3 issues, **none fixed**.
7. `ComputeTargets` 552, `CosmologyModels` 39, `LiouvilleGreen` 148 (skipped=1), `AdaptiveLevin` at
   its baseline — all unchanged. `Datastore/tests` rises from nothing by exactly the tests you add.
8. `black --check` clean. Board created, `docs/OPEN_ISSUES.md` updated in the same commit.

---

## 7. Stop conditions — stop and ask the user

- **The fix is not one line** — the column cannot simply be added, or adding it changes behaviour
  elsewhere. That means the diagnosis in README §0 is wrong and should be re-stated, not worked
  around.
- The read-back still fails after the fix, or returns a `numeric_quad` that disagrees with the
  stored value.
- The audit finds a defect on a path that is **currently reachable in a running pipeline** rather
  than only on resume. That is more urgent than this prompt and the user should hear about it
  immediately rather than in a log.
- The static guard cannot be made honest **and** the round-trip fallback also proves impractical.
- Anything would require writing to the datastore or deleting the backup.

---

## 8. The log and the board

`logs/01-quadsourceintegral-readback.md`, template as README §5.1. Beyond it:

- the **deliberate-breakage record** of §6 item 3: the guard failing on the unfixed code and
  passing on the fixed code, naming the test;
- which guard shape you chose and why, and **what it cannot see**;
- the audit result across all 22 factories, listed, with the 12 candidates named even where clean.

`IMPLEMENTATION_STATE.md`: create it, with a §1 prompt table, a §2 item table, §3 Active and
unresolved issues, §4 Resolved issues, and the maintenance-rule blockquote —
`prompts/handover/IMPLEMENTATION_STATE.md` is the model. `docs/OPEN_ISSUES.md`: add a §1 subsection
for this campaign, add any issue the audit opens, and correct the count and the date.
