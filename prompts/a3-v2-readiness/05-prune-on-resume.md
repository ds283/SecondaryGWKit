# Prompt 05 — prune unvalidated rows when a build resumes

**Campaign:** [`README.md`](README.md) · **Board items:** **P1**–**P2** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** none. Independent of 00–04, but not dispatched while 04's change to
`quadsource_atol_sweep.py` is uncommitted in the tree; `git status` must be clean.
**Closes:** `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]`, on the
`datastore-integrity` board's §4, with its index row deleted. **Opens:** nothing expected.
**Recommended model:** **Sonnet**. A flag removed from two argument lists, and a test that keeps it
removed.

**Read first:**

1. [`README.md`](README.md): §0 item 1, §1, §5.
2. The board entry for `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]` on
   `prompts/datastore-integrity/IMPLEMENTATION_STATE.md` §3.3, including its **Assigned** lines.
3. `docs/datastore-integrity-verification.md` §2.1.4: the duplicate rows, and the pruned resume
   (`resumed-d`) whose fingerprint equals the reference's.
4. `docs/handover/quadsource_atol_sweep.py`: `run_child`, `run_build`, `main`.
5. `main.py`'s `--prune-unvalidated` argument (a `BooleanOptionalAction`, default `True`).
6. `RunRegistry/tests/test_quadsource_atol_sweep_prepare.py`: how this script is tested without Ray.

---

## 1. What is wanted

Pruning unvalidated rows at startup is the designed recovery for a row stored but not validated
when a build was interrupted; the user said on 2026-10-01 that the flag is the bug. Both of the
sweep script's `main.py` invocations pass `--no-prune-unvalidated`, so an interrupted A3 v2 build
resumed with `--build --resume` stores a second row beside every such row. Prompt 10 of
`datastore-integrity` measured both halves already: under the flag, 45 `GkSource` rows twice; one
further launch at `main.py`'s default pruned them and left a store whose fingerprint equals the
uninterrupted reference's. So this prompt needs no new run: it removes the flag and makes sure it
stays removed.

---

## 2. What to change

**P1 — the flag.**
- Remove `--no-prune-unvalidated` from `run_build`'s `sys.argv`, and from `run_child`'s. The child
  writes into a sweep store, a disposable copy whose unvalidated rows are exactly what pruning is
  for; there is no case for keeping it there.
- So that the argument list can be tested, the two lists are built by one function (or two, one per
  mode; *prompt's choice*) that takes the parsed arguments and the database path and returns
  `main.py`'s argv, with no Ray and no `exec`. `run_build` and `run_child` call it.
- Remove the flag from the usage lines in the module docstrings of
  `docs/gktk-remedial/scoped_pipeline_run.py` and
  `docs/source-remediation-verification/scoped_pipeline_run.py`, which teach it to anyone
  launching a pipeline by copying them. Change nothing else in either file.

**P2 — keep it removed.** A test module under `RunRegistry/tests/`, beside the existing sweep-script
test, needing no Ray:
- the build's argv, with and without `--resume`, and the child's, contain no
  `--no-prune-unvalidated`, and nothing else in them disables pruning;
  *Amended 2026-10-03:* that includes an abbreviation. `main.py`'s parser is
  `argparse.ArgumentParser()` with `allow_abbrev` at its default, so `--no-prune` resolves to
  `--no-prune-unvalidated`. No other option of `main.py` begins `--no-p`;
- `main.py`'s `--prune-unvalidated` defaults to true, read from `main.py`'s source with `ast` (it
  cannot be imported; `CLAUDE.md`), so that the build's silence means pruning;
- the rest of each argv is exactly what it was before this commit, element by element, so that the
  refactor moved nothing else.

---

## 3. Verification

1. The new tests pass; the RunRegistry suite with every existing test unchanged.
2. `substitute_main_py(filter_cases=False)` still matches each of its four substitutions once.
3. `quadsource_atol_sweep.py --build --database <a path in a temporary directory> --atol 1e-32
   --rtol 1e-7` is **not** run. Instead, show the argv the function returns for that command line.
4. **Deliberate breakage.** Each must make a test fail; then restore it:
   - (i) put `--no-prune-unvalidated` back into the build's list;
   - (ii) put it back only when `--resume` is given;
   - (iii) change `main.py`'s default to `False` (in the test's view of the source: a mutation of
     the file, restored afterwards);
   - (iv) drop `--source-samples-log10z 100` from the build's list;
   - (v) *added 2026-10-03:* put `--no-prune` into the child's list.

   Each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. The flag gone from both invocations and both usage lines; one function building each argv.
2. The new test module, and the deliberate-breakage record (i)–(v).
3. `[10-a3-v2-launch-resumes-without-pruning-unvalidated-rows]` moved to the `datastore-integrity`
   board's §4 with a closing note naming this commit and citing prompt 10's `resumed-d` as the
   measurement; its index row deleted; the count and date corrected. *Amended 2026-10-03:* the
   note also says that `resumed-d` ran before prompt 02 (`b647c0b`), which moved the prune of the
   replicated classes into the pool. The 45 `GkSource` rows are a sharded class, which an actor
   still prunes, so the measurement stands for them. 07's L2 measures a pruned resume at the tree
   A3 v2 will run.
4. `black --check` clean. On this board: the §1 row, items P1–P2, §5. All in the same commit.

---

## 5. Stop conditions — stop and ask the user

- Removing the flag changes anything else `main.py` does at startup that a resume relies on.
- An existing test pins the old argv.

---

## 6. What this prompt does not do

- It does not touch `main.py` or `ShardedPool`'s prune.
- It does not touch the six extraction scripts' `prune_unvalidated=False`, which is right for a
  reader and is 03's to make structural.
- It does not run a build.

---

## 7. The log and the board

`logs/05-prune-on-resume.md`, using README §5.1. In addition: both argvs before and after.

`IMPLEMENTATION_STATE.md`: the §1 row for 05, items P1–P2, §5.
