# Orchestrator — prompt 03, one inventory service

Read [`../README.md`](../README.md) first. **You do not write code.** You may *replay* a mutation the
log records, with `git apply`, and then revert it.

**The prompt:** [`03-one-inventory-service.md`](../03-one-inventory-service.md)
**Board items:** F8–F9 · **Closes:** `[00-inventory-run-prunes-unvalidated-rows-by-default]`, and
`qcd-background-audit`'s `[03-qcd-inventory-does-not-report-the-representation]`

## 0. What makes this prompt unusual

It is mostly deletion, and it is the first prompt of the campaign that changes what a user sees.
The review turns on four things.

- **The display cannot hide a difference.** A display that leaves one key field out of one class
  makes two records render alike, and nothing else would notice. Read §3 test 1's
  "distinct records render as distinct lines" check, and confirm that it runs over **every**
  class in verbose mode, not over a sample.
- **`--inventory` really is read-only.** The issue it closes is a default that **deletes data**.
  The test must run `main.py` itself, through the `ray.init` guard, on a store holding
  unvalidated rows, and find those rows still there afterwards. A test of the formatter alone
  proves nothing about `main.py`.
- **The retirement is complete.** No caller of the old service may survive, including the two
  `docs/` scripts that import `inventory_config`. Run the `git grep` yourself.
- **D4, and a claim that changes meaning.** The retired QCD test asserted two labels for two rows
  with different `name`s. Under the new service those rows share a key, so they are one key held
  twice, a named `duplicate`. Check that the log says so plainly, rather than quietly weakening
  the test.

## 1. Before you dispatch

1. **D4 was decided on 2026-09-25** (README §6.2): retire the QCD inventory test module and
   re-express its claims. Confirm that the prompt's F9 item 6 still says so.
2. **Confirm the branch and `HEAD`**, and record the SHA. `HEAD` must be the commit that writes
   this prompt, or a later commit of this campaign. Prompt 04 may have landed first; if it has,
   say so to the agent. Tell the agent the tree is not its own.
3. `python -m RunRegistry list`: nothing `running`. **No Ray cluster is running** either
   (`ray status` fails). The demonstration's step 3 starts one and must stop it.
4. **Baseline every suite.** Re-measure.
5. **Snapshot, read-only,** the three stores and their sidecars, and `var/runs/`, as in prompt
   01's orchestrator §1.4, under an `orch_` prefix.
6. **Free disk space:** at least 1 GB. The demonstration makes two copies.
7. `git status` is clean, and `var/` holds nothing new.

## 2. Dispatch

One fresh-context subagent. Give it:
- the prompt, the campaign README and the audit;
- prompt 02 and its log;
- the SHA and the baselines.

Nothing else. Tell it plainly:
- **one commit.** The log goes at `logs/03-one-inventory-service.md`;
- update this board, `qcd-background-audit`'s board and `docs/OPEN_ISSUES.md` in the same commit;
- run `black`;
- **never point new code, or `main.py`, at an original store or the backup.** Work only on copies
  under `var/store-fingerprint-check-03/`, and delete the directory afterwards;
- **§4 step 3 runs the old report before any code changes**, and the Ray it starts is stopped
  afterwards;
- mutations are recorded as diffs, exactly as applied, and never committed;
- do not touch `orch_*` files;
- the prompt's §7 stop conditions mean *stop and ask*. A caller of the old service that the
  prompt does not name is a stop. So is a disagreement between the old and new reports that the
  definitions do not explain.

## 3. The review — eleven checks

1. **The retirement is complete.** Run
   `git grep -nE '\.inventory\(|def inventory\(|inventory_config|_merge_queue|InventoryConfigType' -- '*.py'`.
   It prints nothing. Count the deleted `def inventory(` lines in
   `git diff HEAD~1 HEAD -- Datastore/SQL/ObjectFactories/`: **28**.
2. **Nothing else in a factory changed.** In that diff, every removed line lies inside an
   `inventory()` method or is an import line, and no line is added except a rewritten import.
   `inventory_records`, `build`, `store`, `read_batch`, `validate`, `validate_on_startup` and
   `register` are byte-identical.
3. **Scope.** `git diff HEAD~1 HEAD --stat` lists only the files of the prompt's §5.5.
   - In `ShardedPool.py`, the diff is the deletions and one `primary` property.
   - In `Datastore.py`, it is `inventory` and `InventoryConfigType`.
   - In the two `docs/` scripts, it is `inventory_config` only.
   - `RunRegistry/` is untouched.
   - If `store_inventory.py` changed, every change is additive, and `Record`, `canonical`,
     `canonical_json` and `reference_digest` are untouched.
4. **`main.py`.** The `if args.inventory:` branch is before `ray.init`, and nothing between
   `parse_args` and it constructs anything. Read those lines yourself. The old branch after the
   pool is gone. `--inventory` with `--drop` is refused.
5. **The display drops nothing.** The distinct-rendering test covers every class of the full
   store in verbose mode. Floats are typed by the schema, not by the shape of a string; find the
   code that decides it. Verbose floats use `repr`.
6. **The run labels.** `available_run_labels` keeps its signature, reads `pool.primary`, refuses
   a `store_tag` problem, and returns what it returned before on the demonstration copy.
   `extract_*.py` is untouched, and `test_run_identity` passes unchanged.
7. **Mutations reproduce.** Replay (i) (the branch after `ray.init`), (ii) (`source_grid_digest`
   dropped) and (v) (a parent by digest) from the log. Run the named tests, confirm that they
   fail, and revert. `git status` is clean afterwards.
8. **No Ray, no `var/`, in the tests.** Read the `main.py` test: it guards `ray.init`, and its
   store is in a temporary directory. The unvalidated rows are counted with `sqlite3` `mode=ro`
   before and after.
9. **The demonstration discriminates.**
   - The old and new reports agree, class by class, on count, validated split and value-row
     total. Any difference is explained.
   - The new report on the copy left the copy byte-identical.
   - Re-derive two classes' counts from your own snapshot of the original sweep store.
10. **D4 and the qcd closure.** The retired module's three claims are each re-expressed, and the
    changed meaning of the third is stated. On `qcd-background-audit`'s board, the issue is in
    §4 with a closure line. The row is gone from `docs/OPEN_ISSUES.md`.
11. **The originals are untouched, and the suites and the boards are right.**
    - Re-take §1.5, and confirm that `var/store-fingerprint-check-03/` is gone and no Ray is
      running.
    - Every suite matches its baseline, with the prompt's §5.6 changes exactly.
    - `black --check` is clean.
    - This board: the issue in §4, the §1 row, F8–F9. The index: both rows gone, and the count
      and date right.

## 4. Stop and ask the user

Relay verbatim; do not adjudicate.

- Any of the prompt's §7.
- Any check in §3 fails. **Report it; do not repair it.**
- The agent proposes changing a key, a lookup, a schema, an extract script, `resolve_run_selection`,
  or what counts as a run.

## 5. After it lands

Report:
- the commit;
- what the display looks like: its header, one class, the `BackgroundModel` record;
- every *prompt's choice* the agent changed, and F8 item 2's choice;
- the old and new report counts side by side, and any difference;
- the run labels found, and any `Run_` label no record carries;
- the mutations you replayed;
- the state of the originals, before and after;
- the suite counts.

Then stop. Prompt 04 may be dispatched next, if it has not landed already.
