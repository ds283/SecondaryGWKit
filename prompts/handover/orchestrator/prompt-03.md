# Orchestrator — prompt 03, the policy-geometry census

Read [`README.md`](README.md) in this directory first, and [`prompt-11.md`](prompt-11.md), whose
arrangement this prompt reuses. **You do not write code.**

**The prompt:** [`prompts/handover/03-policy-geometry-census.md`](../03-policy-geometry-census.md)
**Board item:** A3 · **Closes:** README §7 **D5** · **Opens:** the three README §2 (o)/(p) issues ·
**Model:** Sonnet, as the prompt recommends. You, the orchestrator, review on whatever model you
were started with.

Written 2026-10-25, when the prompt was refreshed. Before then this file did not exist: A3 was to
run on the laptop against the A3 baseline store, which the refresh replaced with A3 v2.

## 0. What makes this dispatch unusual

**You run on the Mac Studio, as 10's and 11's orchestrators did.** You are a Claude Code session over
the saved `MacStudio` SSH connection, in `/Users/ds283/Documents/Code/SecondaryGWKit`. You dispatch
the executing agent **in that checkout**, review there, and **commit nothing**. The bookkeeping
commit is made on the laptop (§4).

**Two other prompts may run at the same time.** On 2026-10-25 the user agreed that A3 runs in
parallel with `levin-phase-floor` 04 (on this host, in a worktree at `98e68d3`) and with `handover`
A4 (on the laptop, in a worktree at `98e68d3`). A4 never touches the Mac Studio. `levin-phase-floor`
04 does:
- it writes under `var/levin-phase-floor-04/` in this main checkout, and its code runs from a
  worktree;
- it opens A3 v2 `mode=ro`, uses no Ray and no lock, and runs as one registered detached run;
- it records the main checkout's `HEAD` as the commit its records belong to.

So:
- **The laptop orchestrator fast-forwards the main checkout once, before either session starts.**
  If `levin-phase-floor` 04 is already running when you are started, the checkout is **not** moved:
  dispatch at its current `HEAD` provided that `HEAD` contains the refresh commit and the prompt's
  Gate diff is empty, and otherwise stop and ask.
- **The agent runs no git command that writes**, `fetch` included, and writes only under
  `var/handover-03/`.
- **The Gate's registry check allows exactly one exception**: a `levin-phase-floor-04-*` run whose
  pid answers `kill -0`. The agent records it as a `STRUCTURALLY REQUIRED` deviation.

**Facts checked on 2026-10-25 by the session that refreshed the prompt,** over
`ssh macstudio-tunnel`, read-only. Re-check them; do not trust them.
- **The checkout:** `handover-remedial` at `8a02f0c` (R8's tree), clean.
- **`var/`** holds `datastores`, `handover-06`, `-07`, `-08`, `-10`, `-11`, `-12`,
  `levin-phase-floor-01`, `-03`, `py313-venv-02` and `runs`. There is **no `var/handover-03/`**.
- **The store:** `var/datastores/handover-A3-v2-lambdacdm.sqlite`, four shards and the sidecar
  `handover-A3-v2-lambdacdm.manifest.json`. Fingerprint `0e32f24a…`. The sidecar's `GkSource` tag set
  (1160 rows) includes `SourceGridConstruction_2`, `SourceRedshiftGrid_1740_8d1d43b6` and
  `ResponseRedshiftGrid_145_c51d43ac`.
- **What 11 measured** (`A3-V2-VERIFICATION.md` §5.3): `numeric`/`complete` 363, `mixed`/`complete`
  71, `mixed`/`minimal` 1, `WKB`/`complete` 725; no `fail`, no `incomplete`. The `minimal` row is
  shard 2, serial 460.
- **B1 did not touch the policy:** `git diff 98e68d3 HEAD --stat` on the laptop names no
  `GkSourcePolicyData` or `GkSource` file, in `ComputeTargets/` or in `ObjectFactories/`.
- **Shell.** Every command that takes arguments from a variable runs through `bash`, not zsh.

## 1. Before you dispatch

1. **The Gate**, re-read by you, exactly as the prompt states it. Anything else is a stop. Do not
   fetch.
2. **The `var/` marker:** `touch` a file in your session scratchpad immediately before dispatch.
3. **No suite baselines.** The prompt changes no tracked file.

## 2. Dispatch

One fresh-context subagent, on **Sonnet**, in the Mac Studio's checkout. Give it:
- the prompt;
- the dispatch SHA;
- §0 of this note.

It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.**
- **Commit nothing, and run no git command that writes**, `fetch` included.
- **Write only under `var/handover-03/`.** A write anywhere else is a stop, and that includes
  `var/levin-phase-floor-04/`, the worktree, the store's files, its sidecar and every run directory.
- **Save every command and its output.**
- **Stop and ask** at any of prompt §7's conditions.
- **Do not fix** the three §2 (o)/(p) defects, and **do not recommend** a threshold.

## 3. The review — on the Mac Studio, after the agent reports

1. **Scope.**
   - `git status --short` is empty, and `HEAD` is the dispatch SHA.
   - `find var -newer <marker>` lists only `var/handover-03/`, and `var/levin-phase-floor-04/` and
     its run directory if that session is running. It does not list the store's files or its
     sidecar.
2. **The store.**
   - `diff` `store_files_before.txt` against `store_files_after.txt`: identical.
   - Re-run `store fingerprint` without `--write` yourself; it matches `0e32f24a…`.
   - There is no journal beside any store file.
3. **The script.**
   - It imports nothing from this repository's packages, and opens every file `mode=ro`. Read it.
   - Its header's SHA-256 is the file's. `black --check` is clean on it.
   - **It counts intervals by node index.** Read the counting code: matches by redshift serial, and
     `crossover_z` located by exact equality with a node's stored `z`. A `log(1+z)` width divided by
     a step, or any `exp`/`expm1` back to $z$ in a comparison, is a stop.
   - It takes the exit-time row from `wavenumber_exit_serial`, not by tolerance.
4. **The counts, recomputed by you.**
   - The type/quality count equals 11's in every cell.
   - **Pick three `mixed` rows**, the `minimal` row among them, and recount `numeric_intervals` and
     `WKB_intervals` yourself with a few lines of `sqlite3` against the shards. They agree with the
     agent's table.
   - The distributions are over 72 `mixed` rows.
5. **The §5.5 comparison.** It is stated as a comparison of shape, with the $10^5$–$10^7$
   restriction, and every differing cell is attributed to one of §2.2's three differences with the
   table that shows it, or recorded as unexplained. A fourth reason invented to explain a cell is a
   stop.
6. **§2.1.** Item 1 against the three landmarks on every `mixed` row; item 2 stated as a proxy, with
   the vacuity of the `incomplete` test said in so many words.
7. **The records.**
   - `log.md` has every heading of `logs/11-a3-v2-seam-record.md` and classifies each deviation.
   - `POLICY-GEOMETRY.md` has §0 in one paragraph, the paragraph set beside 11, and the closing
     "what this does not cover". Its links are relative to `docs/handover/`.
   - `issues.md` drafts the three issues under the prompt's names, with file:line evidence that you
     spot-check against the dispatch tree, and proposes no fix.

Write your verdict on each check to `var/handover-03/review.md`, with the commands you ran and what
they printed. End the turn with a report:
- pass or stop;
- §0's paragraph, verbatim;
- the rows with fewer than two intervals on a side, or "none";
- the §5.5 shape comparison's headline;
- §2.1 item 1's verdict;
- the fingerprint.

## 4. After the review — on the laptop

The laptop session is woken when this one ends. It copies `var/handover-03/` back with `rsync -a`,
gzipping any CSV over 10 MB, and re-checks check 4's three recounted rows over
`ssh macstudio-tunnel` itself. Then, in **one commit**, pushed, and separate from `levin-phase-floor`
04's and A4's:
- `out/`, `scripts/`, `state/`, `issues.md` and `review.md` to
  `docs/handover/a3-v2/policy-geometry/`, and `POLICY-GEOMETRY.md` to `docs/handover/`;
- `log.md` to `logs/03-policy-geometry-census.md`;
- the board: A3's row and item; the three issues opened in §3, and the paragraph "Not on this board,
  deliberately" removed;
- `docs/OPEN_ISSUES.md`: the three §1.1 rows now point at the board, the header's sentence about
  them is corrected, and the count and date are corrected by counting;
- README §7 **D5** marked answered, with a pointer to `POLICY-GEOMETRY.md`;
- this directory's README row for 03.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. **Never write to the store, its sidecar, another session's
`var/` directory or the worktree**, and never run a git command that writes.
