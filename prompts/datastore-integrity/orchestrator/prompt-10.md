# Orchestrator — prompt 10, the throwaway store end to end

Read [`../README.md`](../README.md) first: §0.1, §2, §3, §4 and §5. Read also `CLAUDE.md`'s
"Long-running jobs — the run registry", whose rules bind you as much as the agent.

**You do not write code.** You may:
- read the two new stores read-only, and fingerprint them;
- run the census, `PRAGMA integrity_check` and `foreign_key_check`, `read_inventory` and
  `main.py --inventory` on them;
- replay the logged mutation with `git apply`, in a throwaway export, and then revert it;
- run the one wait per launch that §2.1 describes.

**The prompt:** [`10-build-a-throwaway-store-end-to-end.md`](../10-build-a-throwaway-store-end-to-end.md)
**Board items:** V1, V2, V3 · **Closes:** `[09d-distinct-lines-test-is-satisfied-by-the-label-it-cannot-police]` ·
**Model:** Opus
**Gate:** 09d landed and reviewed. The relevant commits:
- `9540314`: 09d;
- `96fb98d`: its remnants;
- `16dba2d`: its review;
- `ec44ae0`: the issue assigned to 10;
- `a2cc9d9`: the index's 09d sentence;
- the commit that writes this note and the prompt.

**Never dispatched concurrently with anything else in this checkout**, and never while any other
Ray instance of this session is alive.

## 0. What makes this prompt unusual

**It is the campaign's only real run.** Every earlier prompt was tested on stand-ins and fixture
stores without Ray, and every log says so in its last observation. This one builds two stores under
`var/datastores/`, the one exception that README §3 and §5 rule 10 allow. It interrupts one of
them twice and resumes it, then reads both back with everything that reads a store.

**It runs for hours, and no conversation may own it.** The agent launches each build detached,
through the registry, checks once that it started, and ends its turn. You wait, with **one** wait
per launch (§2.1), and resume the agent when the run is no longer alive. Neither of you polls.
Four launches, so four waits: `reference`, then `resumed-a`, `-b` and `-c`.

**The user's decisions come first.** The prompt's §1.1 states D10-1 to D10-5, each with a
recommendation. They are the user's to take before dispatch.
- Commit them as a decision record on the board, as `9a18063` did for 09c, before you dispatch.
- Give them to the agent in the brief, and say where any differs from the prompt's recommendation.
- **D10-1's evidence was measured by the orchestrator on 2026-10-01, at `ec44ae0`.** The A3
  baseline's four shards were opened `mode=ro` with `sqlite3`, and their SHA-256 were unchanged
  afterwards. The query:
  - joined `QuadSourceIntegral`'s three `*_wavenumber_exit_serial`s to `wavenumber.k_inv_Mpc`, and
    its `z_response_serial` to `redshift.z`;
  - found 7,654 rows over eight $k$ values.

  For $k, q, r \le 1.1\times10^6$ there are 1,060 rows, 6,315 s of `compute_time` in total, a
  maximum of 533 s, and 18 rows over 60 s. Their lowest $z_{\rm response}$ is 132–230 by triangle.
  For $\le 3.2\times10^6$: 1,899 rows, 60,069 s, a maximum of 3,876 s. For $\le 10^7$: 2,984 rows,
  303,590 s, a maximum of 12,714 s.

**What the review exists to establish.** All else is mechanics.
- **(R1) The resume is the reference.**
  - Every stage finished before an interruption computes nothing on the next launch.
  - The resumed store's fingerprint equals the reference's, class by class, with no difference.
- **(R2) Integrity holds on real shards.** On both stores:
  - the census finds no divergence;
  - `integrity_check` is `ok`, and `foreign_key_check` is empty;
  - the inventory has no problem.

  The four fault copies refuse, or repair, exactly as prompt 02 designed.
- **(R3) Every reader works, and none changes the content.** All six `extract_*.py` scripts and
  `main.py --inventory`, each leaving the fingerprint where it was.
- **(R4) The test fix.** The amended test is caught by the mutation that the old one was not.

**Carried from the reviews of 08 and 09a–09d** (board §1). All bind the agent, and the brief says so.
- **Staging and reporting.**
  - Stage by explicit path only, never `git add -A`, `.` or `commit -a`. Show
    `git diff --cached --name-only` before committing.
  - Another session may write in this checkout; stop and ask if one modifies a file the agent needs.
  - **Count, then paste.** Paste `-v` output unfiltered.
  - A log states only what the diff shows, and its line numbers come from `git show`, not a
    working-tree view.
- **Mutations.**
  - Each is pasted as `git diff`, and replayed twice: before the commit from the final log file,
    and after it from `git show HEAD:<log>`, reported with no amend.
  - Replays run in a `git archive` export **made into its own git repository**, so that the tests
    that call `git` pass there.
- **Load-sensitive tests.** The machine has been heavily loaded by an unrelated job in another of
  the user's workspaces, with load averages of 35–290 during 09 and 80–160 during 09d. Then these
  fail:
  - the known flake `test_tk_wkb_phase.TestCost.test_wall_time_per_object`;
  - `test_background_tau.test_build_cost`;
  - `test_numeric_phase_cut.test_the_print_is_gated_both_ways`.

  Re-run a failure alone, and on an export of the parent commit, before calling it real. **Never
  touch a process you did not start.**
- **New files.** `test_inventory_retired`'s `git grep` sees only tracked files, so a name clash
  shows only once the file is staged (log 08, deviation 8; log 09b, deviation 15; log 09d, where
  a helper named `inventory` tripped it). Run the suites on the staged tree.
- **Adjacent prose and attribution.**
  - A docstring or comment beside a changed line that the change makes false must be fixed.
  - A comment citing a decision names it as the user's.
- **Residue and scratch.**
  - Small residue found in review is the orchestrator's, fixed in a follow-up commit (the user,
    2026-09-29).
  - The agent works in its own scratch subdirectory, never the session scratchpad's top level.
    Long-running results go under `var/`, never a scratchpad (`CLAUDE.md` rule 5).
- **Usage limits.** Agents of 09b and 09c stopped on API usage limits mid-task. Resume one with
  `SendMessage`, saying exactly where it stopped. If an agent commits but sends no report, review
  from the commit, its log and its scratch outputs. A usage stop **does not stop a detached build**,
  which is the point of detaching it. Your wait still ends when the build does.

## 1. Before you dispatch

1. **The tree.** Record the branch and `HEAD`, which is the commit that writes this note, or a
   later one that touches no code; say which.
   - `git status --short` and `git diff --cached` are both empty.
   - `git worktree list` shows this checkout alone.
   - If another session has uncommitted work here, do not dispatch; ask the user.
2. **The registry.** `python -m RunRegistry list`: nothing `running`. Report any stale or unknown
   runs, without acting on them.
3. **Ray and the machine.**
   - `ray status` finds no cluster, and `RAY_ADDRESS` is unset.
   - Record `df -h var/`: 47 GB free on 2026-10-01, at 96 % used. The two stores should need well
     under 1 GB, but say so if free space is below 10 GB.
   - Record the load average and `sysctl -n hw.ncpu` (10 on 2026-10-01).
4. **The snapshot.** Snapshot `var/datastores/`, `var/runs/` and the root
   `physics-test-n20-lambdacdm-zend0p1*.sqlite` read-only, with `orch_04_snapshot.py` in the session
   scratchpad. Write it again if it is missing: the SHA-256, size and mtime of every file, and the
   directory list. Open files only as bytes, never as databases, and never put the script in the
   tree. The 09d session's snapshot held 46 files and 10 directories.
   - **This prompt adds files, by design.** The review's check is that every file in this snapshot
     is byte-identical afterwards. It also checks that every new path is one of:
     - the two stores' primaries, shards and sidecars;
     - `var/runs/datastore-integrity-10-*`;
     - `var/datastore-integrity-10/`.
5. **Baselines** (board §5.15): AdaptiveLevin 32, ComputeTargets 665, CosmologyModels 39,
   Datastore 551, LiouvilleGreen 148 (1 skipped), RunRegistry 199. Re-measure all six on `HEAD`,
   serially, with README §7's command.
6. **The index.** Count its rows (`grep -c '^| \`\[' docs/OPEN_ISSUES.md`) and tell the agent the
   number. It is **99** at `a2cc9d9`.
7. **The names are free.** `python -m RunRegistry store show` on both store paths reports no
   sidecar and no tombstone, and no file exists by either name.
8. **The decisions** are committed (§0), and the brief carries them.

## 2. Dispatch

One fresh-context subagent, on Opus. Give it:
- the prompt;
- the README's §3, §4, §5 and §5.1;
- `CLAUDE.md`'s run-registry section;
- logs 09c and 09d;
- `HEAD`, the baselines and the index count;
- the user's decisions;
- §0 of this note, up to "## 1. Before you dispatch" only, as a copy in the session scratchpad.
  The agent does not read `orchestrator/`.

Tell it plainly:
- **One commit, at the very end**, after all four launches, every check and the suites. Its log is
  at `logs/10-build-a-throwaway-store-end-to-end.md`, in README §5.1's form, with the prompt's §7
  additions.
- **The hand-off.**
  - After each launch: one check, a report giving the run id, the pid and the interrupter's pid if
    there is one, then the end of its turn.
  - You resume it with the run's final `status.json`, and with the wait's own verdict.
  - It never waits, polls or sleeps on a build itself.
- **The files it may change** are exactly the prompt's §7 list. **Nothing else**, and in
  `Datastore/tests/test_inventory_report.py` only the one test.
- **Stop and ask** on any of the prompt's §5 conditions. A stop leaves the stores, the run
  directories and the tree as they are: no relaunch, no deletion, no fix.

### 2.1 The one wait per launch

When the agent reports a launch, run **one** background command that ends when that run is no
longer alive, or after the cap. For example:

```bash
PYTHONPATH=. ./venv/bin/python - <<'EOF'
import json, sys, time
import RunRegistry
status_path = "var/runs/<run id>/status.json"
start, cap, seen_stale = time.time(), 4 * 3600, False
while True:
    status = RunRegistry.read_json(status_path) or {}
    state = RunRegistry.liveness(status, 900)
    if state == "finished" or not RunRegistry.pid_alive(status.get("pid")):
        print("ENDED", state, json.dumps(status)); break
    seen_stale = seen_stale or state == "stale"
    if time.time() - start > cap:
        print("CAP", state, json.dumps(status)); break
    time.sleep(120)
print("heartbeat went stale at least once" if seen_stale else "heartbeat never stale")
EOF
```

- It identifies the run by its pid and heartbeat, never by a command-line match (`CLAUDE.md`
  rule 2).
- It exits when the build exits, so it cannot outlive what it waits on. On `CAP`, it leaves the
  build running.
- Check that `RunRegistry.read_json` exists before relying on it. If it does not, read the file
  with `json`.
- When it ends, send the agent the run's final `status.json` and the verdict line, and nothing
  else.
  - On `ENDED`, it carries on.
  - On `CAP`, the prompt's §5 four-hour condition is met. Tell the agent, and ask the user before
    anything else happens.
- Do not run a second wait while one is alive, and do not check on the build between its start and
  its end.

## 3. The review — eleven checks

1. **Scope.**
   - `git diff HEAD~1 HEAD --stat` touches only the prompt's §7 files.
   - `--diff-filter=M` over `*/tests/*` is `Datastore/tests/test_inventory_report.py` alone. Its
     hunks are inside `test_verbose_renders_every_distinct_record_as_a_distinct_line`, plus at most
     one import or one module-level regex.
   - The diff over every factory, `ComputeTargets/`, `main.py`, `tools/`, `RunRegistry/`,
     `config/`, both scoped drivers, `quadsource_atol_sweep.py`, `extract_*.py`,
     `docs/datastore-integrity-audit/` and `orchestrator/` is empty.
2. **The runs.**
   - Read every manifest and `status.json` under `var/runs/datastore-integrity-10-*`.
   - States: `reference` and `resumed-c` are `done` with exit code 0; `resumed-a` and `-b` are
     `killed`.
   - Each manifest's `git_head` is the dispatch `HEAD`, and its `results` names the right store.
   - Where `git_dirty` is true, the log's `git status --short` at that launch shows only the
     agent's own new verification files.
   - Each interrupter's log shows one `SIGTERM`, with the pid and stage it acted on. The dummy test
     is in the log.
   - No process the agent started is alive: neither interrupter, nor any Ray instance
     (`ray status`).
3. **(R1) Nothing stored was recomputed.**
   - Parse the lookup / compute / store counts from the four `stdout.log`s yourself, not from the
     log's tables, and compare.
   - Every stage completed before an interruption computes 0 on the next launch.
   - Summed over the three launches, each stage stores what `reference` stored. Any excess equals
     the rows in flight at the `SIGTERM`, as the log says.
4. **(R1) The fingerprints.**
   - Fingerprint both closed stores yourself with `RunRegistry.stores.fingerprint_store`
     (read-only). `compare_fingerprints` gives nothing, both are at format 6, and each equals its
     sidecar's recorded fingerprint.
5. **(R2) Integrity, independently.** On every shard of both stores, opened `mode=ro`:
   - `integrity_check` is `ok`, and `foreign_key_check` is empty;
   - `replicated_serial_census.py` finds no divergence;
   - `read_inventory` has no problem in any class;
   - the primary's `replication_in_flight` is empty.

   Then compare the census's table list with `config/sharding.py` yourself.
6. **(R3) The readers.**
   - Each `extract_*.py` script's output exits 0 and names its run selection.
   - The fingerprint after the last one equals the sidecar's.
   - Say whether `extract_QuadSourceIntegral_data.py`'s write
     (`[08-extraction-script-object-get-of-gksourcepolicy-writes-to-the-store-it-reads]`) was
     observable at all.
7. **(R3) The inventory report.**
   - Render `main.py --inventory` and `--inventory-verbose` yourself on both stores; this needs no
     Ray.
   - With your own regex, not the agent's: every label printed has exactly one defining line, and
     no 64-hex digest appears.
   - The two stores' reports are identical once header timestamp ranges are masked.
   - Record the sizes and the longest record line per class. This is the first measurement of 09d's
     report on real rows.
8. **(R2) The fault copies.**
   - Each case's message is in the log verbatim, and names the class, shard and serial where the
     case has them.
   - The repair case's census after the open is clean.
   - The temporary directory is gone, and nothing under `var/` was a fault copy.
9. **(R4) The test fix.** Replay the mutation from `git show HEAD:<log>` with plain `git apply`, in
   a git-initialised export of `HEAD`:
   - the amended test fails for exactly the logged classes, sub-tests counted;
   - with `HEAD~1`'s test file in place, it passes;
   - the export's tracked diff is empty after the revert.
10. **The records.**
    - The board: the §1 row for 10, V1–V3, the issue on §4 with its closing note, §3.3 for anything
      opened, and §5.16. The header reads "the campaign's work is done; it is closed by the user
      after the orchestrator's review of 10", and does not say it is closed.
    - The index: the row deleted, any opened added. The count is 98 plus any opened, by counting,
      and §1.15 has its sentence.
    - The verification record quotes every command verbatim, and its numbers match the run
      directories.
    - Its §2.6 checklist names every open issue on this board.
11. **`var/`, suites, `black`, `main.py`.**
    - Every file in §1's snapshot is byte-identical, and every new path is on §1's list.
    - All six suites run serially in the checkout. Datastore is unchanged in count, and so are the
      others. Treat load failures as §0 says.
    - `black --check` is clean on every Python file added or changed.
    - `main.py` is byte-identical to `HEAD~1`.

**Audit probes are not run.** This prompt changes no code they import. Check that the commit's diff
over `Datastore/`, `ComputeTargets/` and `tools/` holds nothing but the one test file.

## 4. After the review

- **Record it on the board**, after the 09d review, in that review's form. Fix any small residue in
  a follow-up commit of its own.
- **Then report to the user.** Say:
  - whether R1–R4 hold;
  - what the run found that no stand-in could, and every issue it opened;
  - the §2.6 checklist for A3 v2.
- **The user decides two things**:
  - whether the campaign closes;
  - D10-5's question, whether the two stores are retired. `store retire` is theirs to run.
- **On the user's word, write the closure** in its own commit, as `store-retirement` did after its
  prompt 06:
  - README's header paragraph and §2 row;
  - the board's header status, now "closed";
  - one sentence in the index's §1.15.

  Issues still open stay on this board's §3, each with its index row, for their owners.
