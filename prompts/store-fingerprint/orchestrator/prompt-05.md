# Orchestrator — prompt 05, fingerprint the real stores

Read [`../README.md`](../README.md) first. **You do not write code, and you do not write a
sidecar.** You may read the stores, read-only, to check what the agent did.

**The prompt:** [`05-fingerprint-the-real-stores.md`](../05-fingerprint-the-real-stores.md)
**Board item:** F13 · **Closes:** nothing · **Narrows:** `[00-replicated-writes-can-diverge-across-shards]`

## 0. What makes this prompt unusual

It is the only prompt of the campaign that writes a real file, and the first that points campaign
code at an original store. There is no code, so there are no tests and no mutations. The review
turns on four things.

- **Exactly one field, and a rollback that is exact.** All three sidecars are in the writer's byte
  form (the prompt's §2). So after each write, deleting `fingerprint` and re-serialising must give
  back the old bytes **exactly**. Check that yourself, against your own copy of the bytes taken
  before dispatch, not the agent's.
- **The reads did not write.** Every store file except the three sidecars must be byte-identical
  before and after, mtimes included. Compare with your own snapshot.
- **The backup read its own shards.** Its primary names the live store's shard paths. Its
  `QuadSourceIntegral` count, 7 552 against the live 7 654, is the direct check.
- **Findings are not stops, but a doubt about the fingerprint is.** The prompt's §3 choice 2 and §7
  draw the line. Check that the agent drew it where the prompt does. It must not have written after
  a stop condition, and must not have stopped on a finding.

## 1. Before you dispatch

1. **Confirm the branch and `HEAD`**, and record the SHA. `HEAD` must be the commit that writes
   this prompt, or a later commit of this campaign. Tell the agent the tree is not its own, and
   that it must stay clean until the writes are done (the prompt's §3 choice 3).
2. `python -m RunRegistry list`: nothing `running`.
3. **The three sidecars.** For each, record:
   - its SHA-256 and size, which must equal the prompt's §2 table;
   - that it has no `fingerprint` field;
   - that `store show` says registry, with problems `none`;
   - that it re-serialises to its own bytes.

   **Copy their bytes** into your scratchpad under an `orch_` prefix. This is your independent
   record for the rollback check. If any differs from the prompt's table, stop and ask: something
   wrote it after the prompt was written.
4. **Baseline every suite.** Re-measure.
5. **Snapshot, read-only,** every file under `var/datastores/` and the backup's directory:
   listing, size, `st_mtime_ns`, SHA-256, and per-table row counts (`sqlite3` `mode=ro`). Also
   every entry under `var/runs/`, with the SHA-256 of each `manifest.json` and `status.json`. Keep
   it under an `orch_` prefix. Take it once; do not re-read the stores while the agent works.
6. **No fourth store:** `find var -name '*.sqlite'` shows only the three stores' fifteen files.
7. **Free disk space:** at least 1 GB.
8. `git status` is clean, and `var/` holds nothing new.

## 2. Dispatch

One fresh-context subagent. Give it:
- the prompt, the campaign README;
- prompt 04's log, and prompt 02's log;
- the SHA and the baselines.

Nothing else. Tell it plainly:
- **one commit.** The log goes at `logs/05-fingerprint-the-real-stores.md`, and the fingerprints
  as written at `logs/05-fingerprints.json`;
- update this board and `docs/OPEN_ISSUES.md` in the same commit, and README §2's row for 05;
- **the tree stays clean until Phase B's writes are done.** Draft the log outside the repository;
- **the only files it may write outside `var/store-fingerprint-check-05/` are the three sidecars'
  `fingerprint` field**, through `store fingerprint --write`, and only in Phase B;
- **never begin a run**, under `var/runs/` or anywhere;
- delete `var/store-fingerprint-check-05/` at the end;
- do not touch `orch_*` files;
- the prompt's §7 stop conditions mean *stop and ask*, **before any further write**.

## 3. The review — ten checks

1. **Scope.** `git diff HEAD~1 HEAD --stat` touches only the log, `logs/05-fingerprints.json`, this
   board, the README and `docs/OPEN_ISSUES.md`. No `.py` file changed. In the README, only §2's
   row for 05 changed.
2. **The sidecars changed in one field.** For each of the three, using your own `orch_` copy of
   its bytes:
   - its keys are the old keys plus `fingerprint`;
   - deleting `fingerprint` and re-serialising (`json.dumps(indent=2, sort_keys=True)` plus a
     newline) gives your copy's bytes exactly.
3. **The recorded fingerprint is right.** For each store, run
   `python -m RunRegistry store fingerprint <primary>` read-only. It must say `matches`, naming a
   person and the base SHA with no `(dirty)`. `taken.run_id` is null and `taken.git_dirty` is
   false. Check that the sweep's overall digest is
   `2c2dde68659a138e8bd1454f0d3c35f600a09f73c6e7f55c385a9472c4da5b18`.
4. **The committed JSON is the sidecars' value.** Each entry of `logs/05-fingerprints.json` parses
   equal to the matching sidecar's `fingerprint`.
5. **Counts.** For each store, check the fingerprint's class counts against your own row counts
   from §1.5. The backup's `QuadSourceIntegral` is 7 552 and its `tolerance` 6.
6. **The known differences.** In a scratch script, call `compare_fingerprints` on the backup's and
   the live A3's recorded fingerprints, then on the live A3's and the sweep's. Your entries must
   equal the log's. Check the log's named records against this:
   - the added `tolerance` records account for 6 and 1;
   - the added `QuadSourceIntegral` records account for 102 and 52, by `(atol, rtol)`;
   - the number only in the older store is stated.

   Re-derive one `log10_tol` from its `float.hex` yourself.
7. **Findings and stops.** Every difference beyond the two expected classes, every removal and
   every named problem is in the log's findings, and each unexplained one is a §3 issue with an
   `OPEN_ISSUES` row. No write happened after a stop condition. The replicated-divergence entry has
   the measurement for the A3 store and the backup.
8. **The originals.** Re-take §1.5. Every file except the three sidecars is identical: bytes,
   size, `st_mtime_ns`, row counts. The listings match, with no `-journal` or new file anywhere.
   `var/store-fingerprint-check-05/` is gone. `var/runs/` is identical, and `RunRegistry list`
   shows the same runs.
9. **The registry copy.** The log shows Phase C: a new `store_id`, the fingerprint carried
   verbatim, and `matches`. Its directory is gone.
10. **Suites and boards.** Every suite matches its baseline, and `black --check` is clean. This
    board: the §1 row for 05, F13, §3, §5. The index: the count and date right, and every row the
    log's §3 changes added or updated.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate.

- Any of the prompt's §7.
- Any check in §3 fails. **Report it; do not repair it.** Above all, do not "fix" a sidecar. If one
  is wrong, report its bytes and the exact rollback; restoring it is the user's decision.
- The agent proposes changing code, a store, or any sidecar field but `fingerprint`.

## 5. After it lands

Report:
- the commit;
- for each store: the overall digest, the class counts, the problem counts, and the time and
  memory;
- the two comparisons, and the records they named: which tolerance values, which `(atol, rtol)`
  pairs, and any removal;
- every finding, and every issue opened;
- every *prompt's choice* the agent changed;
- each sidecar's SHA-256 before and after, and that the rollback is exact;
- the state of the originals, before and after;
- the suite counts.

Then stop. The campaign's five prompts will have landed. Closing the campaign is the user's
decision.
