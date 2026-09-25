# Orchestrator — prompt 04, the fingerprint

Read [`../README.md`](../README.md) first. **You do not write code.** You may *replay* a mutation the
log records, with `git apply`, and then revert it.

**The prompt:** [`04-the-fingerprint.md`](../04-the-fingerprint.md)
**Board items:** F10–F12 · **Closes:** `run-registry`'s
`[04-a-runs-product-is-named-but-never-fingerprinted]`

## 0. What makes this prompt unusual

This is the prompt the campaign exists for, and the first to write into a sidecar since the
sidecars were adopted. The review turns on five things.

- **Localisation.** The user's point in asking for per-class, per-tag-set digests is that a
  mismatch names what differs. The discriminator in §4 step 6 must name **exactly one** class and
  tag set. A fingerprint that only says "different" fails the charter, even if every digest is
  correct.
- **Exactly one field.** `--write` and a run's finish replace the sidecar's `fingerprint` and
  nothing else, unknown fields included. Compare the bytes yourself, not the agent's summary.
- **A fingerprint never changes how a run ended.** Read `finish`: a refusal or error is recorded
  and swallowed, and the state and exit code written are the ones given. The signal handlers
  keep `fingerprint=False`, because the pool is still open when they run.
- **The format is pinned.** The golden fingerprint must be a committed file compared by a test,
  so that a later change to a key cannot go unnoticed. Check that mutation (viii) fails it.
- **The layering holds.** `import RunRegistry`, `list` and `store show` load neither `ray` nor
  `sqlalchemy`. `Datastore/` is untouched. The drivers change only in their `finish` calls (D2).

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`**, and record the SHA. `HEAD` must be the commit that writes
   this prompt, or a later commit of this campaign. Prompt 03 may have landed first; if it has,
   say so to the agent. Tell the agent the tree is not its own.
2. `python -m RunRegistry list`: nothing `running`.
3. **Check the three real sidecars** read-only: none holds a field named `fingerprint` (the
   prompt's first §7 condition). Record their SHA-256.
4. **Baseline every suite.** Re-measure.
5. **Snapshot, read-only,** the three stores and their sidecars, and `var/runs/`, as in prompt
   01's orchestrator §1.4, under an `orch_` prefix.
6. **Free disk space:** at least 1 GB. The demonstration makes two copies.
7. `git status` is clean, and `var/` holds nothing new.

## 2. Dispatch

One fresh-context subagent. Give it:
- the prompt, the campaign README and the audit;
- prompt 02's log, and prompt 03's log if 03 has landed;
- the SHA and the baselines.

Nothing else. Tell it plainly:
- **one commit.** The log goes at `logs/04-the-fingerprint.md`;
- update this board, `run-registry`'s board and `docs/OPEN_ISSUES.md` in the same commit;
- run `black`;
- **never point new code at an original store, the backup, or a real sidecar.** No fingerprint is
  written into a real sidecar: that is prompt 05. Work only on copies under
  `var/store-fingerprint-check-04/`, with a runs root inside it, and delete the directory
  afterwards;
- **never begin a run under `var/runs/`**;
- mutations are recorded as diffs, exactly as applied, and never committed;
- do not touch `orch_*` files;
- the prompt's §7 stop conditions mean *stop and ask*.

## 3. The review — eleven checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only `RunRegistry/`, its tests and test data,
   the two drivers, the log, the two boards and the index. `git diff HEAD~1 HEAD -- Datastore/`
   is empty. In `git diff HEAD~1 HEAD -- docs/`, every changed line is a `run.finish(` call.
2. **The format.** Read `fingerprint_of`. It digests `record.canonical_json()` lines in the
   inventory's order, per class and per tag set. The overall digest covers the format version.
   `taken` and `problems` are outside every digest. There is one canonical JSON: nothing in
   `RunRegistry/` calls `json.dumps` to form a digest.
3. **Localisation.** Test 2 covers all four changes, and asserts that no other digest moves.
   `compare_fingerprints` refuses to compare two formats.
4. **The golden.** It is a committed file, the test compares it, and its format equals
   `FINGERPRINT_FORMAT`. Run the test twice; it is deterministic.
5. **The sidecar.** `fingerprint` is in `KNOWN_FIELDS`, and `_registry_problems` checks its shape.
   `copy_store` and `move_store` carry it, `taken` included. `_update_sidecar`'s docstring says
   three places, and names them.
6. **The refusal.** `fingerprint_store` refuses a store any `running` run names, alive or stale,
   by path or by `store_id`, except the `taken_by` run. It imports the inventory inside the
   function.
7. **`finish`.** `fingerprint=False` is the default. An error becomes `fingerprint_error`, and the
   terminal state is the one given. In each driver, read every `run.finish(` call. There are
   twelve at the base: three in `terminal` handlers, which do not pass `fingerprint=True`, and
   nine after a pipeline returns or raises, three in each registered path, which do.
8. **Mutations reproduce.** Replay (ii) (a tag-set digest over the whole class), (iv) (a write
   without `--write`) and (vii) (a fingerprint error raised) from the log. Run the named tests,
   confirm that they fail, and revert. `git status` is clean afterwards.
9. **No Ray, no `var/`, in the tests.** Nothing initialises Ray. Every store and runs root is in
   a temporary directory. The existing import test is unchanged and passes.
10. **The demonstration discriminates.**
    - Step 4: the sidecar's bytes differ from step 2's only in `fingerprint`. Diff the JSON
      yourself.
    - Step 5: the registry copy says `matches`.
    - Step 6: exactly one difference, `QuadSourceIntegral`, 7 706 against 7 705.
    - Step 7: `status.json`'s fingerprint equals the sidecar's, and the second, `running` run
      made `store fingerprint` refuse.
    - Re-derive the per-class counts of two classes from your own snapshot of the original sweep
      store.
11. **The originals are untouched, and the suites and the boards are right.**
    - Re-take §1.5, including the real sidecars' SHA-256, and confirm that
      `var/store-fingerprint-check-04/` is gone and nothing new is under `var/runs/`.
    - Every suite matches its baseline, and `RunRegistry/tests` rises by exactly the tests added.
    - `black --check` is clean.
    - This board: the §1 row, F10–F12. `run-registry`'s board: the issue in §4 with a closure
      line. The index: the row gone, and the count and date right.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate.

- Any of the prompt's §7.
- Any check in §3 fails. **Report it; do not repair it.**
- The agent proposes writing a real sidecar, changing the inventory, a key or a schema, or
  changing a driver beyond its `finish` calls.

## 5. After it lands

Report:
- the commit;
- the fingerprint's format as shipped, with the demonstration's real example;
- every *prompt's choice* the agent changed;
- the demonstration's numbers: the overall digest, the per-class table, the discriminator's one
  difference, and the size, time and memory;
- the mutations you replayed;
- the state of the originals and the real sidecars, before and after;
- the suite counts.

Then stop. Prompt 05 is written next, against the format 04 shipped (README §2).
