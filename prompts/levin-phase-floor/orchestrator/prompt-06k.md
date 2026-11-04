# Orchestrator — prompt 06k, three ways to stop refining the slow leaves

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the measurements are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06k-slow-leaf-trust-routes.md`](../06k-slow-leaf-trust-routes.md)
**Board item:** 06k · **Closes:** nothing. Its records measure three remedies for
`[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`; which, if any, becomes code
is for the user.
**Model:** Opus.

It is a measurement on the Mac Studio only, with no code commit. The machinery is 06j's note §B: an
orchestrator session on the Mac Studio, through `ssh macstudio-tunnel`, dispatches and reviews.
What is new is three census runs, each with a scratch variant of the Levin module loaded in place
of the tree's, as 06c's (b′) was.

**Dispatched on the user's word.**

## 0. The order

1. **This prompt's commit is the dispatch SHA.** Push it. The laptop then makes the worktree
   `SecondaryGWKit-lpf06k-<short SHA>` on the Mac Studio, detached, with no venv.
   `git diff --stat 1c5336e <dispatch SHA> -- . ':!docs' ':!prompts'` shows only
   `tools/filon_reference.py`, `tools/tests/test_filon_reference.py` and `CLAUDE.md`.
2. **06k runs** on the Mac Studio: Gate K0, Part L, then the three censuses one after another.
   One agent; the Mac Studio orchestrator reviews it.
3. **The laptop** copies the records back and makes 06k's records commit.
4. **The user decides** from 06k's results whether a remedy prompt implements a route, and which.
   Then 07 resumes. Neither starts here.

## B. On the Mac Studio

### B.0 Facts, checked by the orchestrator on the laptop, 2026-11-04

- **06g's census** `levin-phase-floor-06g-census-20261101T114315` is kept in full on the Mac Studio.
  - Its checkpoint is `03adcbba8a78d82d9d6b899b8eee8b9ae1659763f52deb066d2840af3d82c0fa`, with 16
    region pickles.
  - It took 5 h 40 min. 06h's took 2 h 33 min.
  - `census/out/v8_calls.json` (`1e9a62b0…`) lists V8's lost calls.
- **The references** are committed:
  - 06g's: `docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json`, `rows` (102:
    V3's 90 and V8's 12) and `V1` (12);
  - 06j's: `docs/levin-phase-floor/06j/out/w_calls.csv`, 22 calls on 17 references.
  06g's reference pickles were deleted on 2026-11-03. No route needs a new reading.
- **06j's 46 captures were deleted on 2026-11-03.** They are reproducible: run 1, which made them,
  took 11 minutes, and J0 showed each equal to 06g's census bit for bit.
- **06j's replays** took 0.2–10 s per call (103 s for all 46).
- **The scratch copy** `var/levin-phase-floor-06/wider/state/a3v2_copy/` is certified; its
  fingerprint is `0e32f24a…`.
- **The Mac Studio** had 89 GB free on 2026-11-03. Three census run directories at about 4.5 GB each
  need about 14 GB.
- **06g's four worktrees** `SecondaryGWKit-lpf06g-{1c5336e,4f4f9ed,b5626ec,ff3e839}` stay, on the
  user's word; neither you nor the agent touches them.
- **No Ray, no lock, no store built.** If anything needs Ray, that is a stop.

### B.1 Before you dispatch

1. `ssh macstudio-tunnel`; `runregistrykit list` shows nothing `running` or stale; the main
   checkout is clean; the worktree is at the dispatch SHA and clean, including ignored files.
2. **SHA-256 checks.** The scratch copy matches its certification, with no `-wal` or `-shm`. 06g's
   census checkpoint, its 16 region pickles and `v8_calls.json` match theirs.
3. **Disk.** At least 40 GB free.
4. `touch` a marker; record the `var/` listing, as 06j's preflight did.

### B.2 Dispatch

One fresh-context agent, on **Opus**, in the worktree. It is given the prompt, the dispatch SHA and
§B.0, and it does not read `orchestrator/`. It writes only under `var/levin-phase-floor-06k/` and,
for registered runs, `var/runs/`. It never touches:
- `var/levin-phase-floor-07*`, or 06g's and 06j's records;
- the scratch copy's directory, beyond reading the copy;
- any store.

It makes no commit and runs no `git` write. The variants live under its own `scripts/`, never in the
tree.

**Two planned stops, each checked here before the agent goes on:**
1. **After Gate K0 and Part L, before the first census:**
   - **The variants' diffs:** read each one and run your own AST comparison against the tree's
     module. Only `_adaptive_levin` may differ, and within it only what prompt §0.2 names.
   - **K0's records:** the identity variant is bit-identical on all 100 units, counters included,
     and each route matches 06g wherever its firing count is 0.
   - **The census plan:** the cost projection, from 06g's and 06h's walls, and the backstop. The
     backstop stops a run only at a projected day or more, counting the units in flight (06g's
     review, Slips).
2. **Between censuses:** one check that each finished run is `done`, 18,560 of 18,560, before the
   next starts.

Wait on each detached run with a background `kill -0` loop on its pid, not by polling.

### B.3 The review — on the Mac Studio

1. **Scope:**
   - nothing under `var/` newer than the marker outside `var/levin-phase-floor-06k/` and the 06k
     runs;
   - the scratch copy and 06g's checkpoint unchanged by SHA-256;
   - the worktree clean, including ignored files;
   - no Ray started; nothing new in `var/datastores/`.
2. **The variants and K0.**
   - Run your own AST comparison of each variant against the tree's module.
   - **Re-run two of K0's units yourself,** each under the identity variant and under one route, and
     check their bit-identity with 06g's census record.
3. **The samples and predictions:** the K0 draw rule, its seed, and L1, L2, C1, C2, H1, S1 and K1,
   copied verbatim into the log before the first number, by its timestamps.
4. **Part L, re-checked:** for three of 06j's slow leaves and two 06h carrying leaves, recompute
   each route's decision yourself: $\nu_1$ with the halves' floors, $\Delta_{1c}$ and $r_c$ from
   the solves' $E$, and $D$ against $\eta$ times the request.
5. **Part C, re-checked:**
   - for each route, recompute C1's and S1's counts from the route's census checkpoint and 06g's,
     with your own reader;
   - **re-run one recovered call yourself** under its route, and check its value, abserr, regions,
     `converged` and firing count against the census record.
6. **The references:** for two calls per route, recompute
   $|v_{\rm route} - {\rm ref}|/e_{\rm route}$ from the checkpoint and the reference.
7. **The records:**
   - README §5.1's headings, with every deviation classified;
   - `results.md` proposes no code;
   - provenance headers;
   - what is kept on the Mac Studio is listed with SHA-256.

Write `var/levin-phase-floor-06k/review.md`, with the verdict in its first line, and report.

### B.4 On the laptop, after the review

1. Copy `var/levin-phase-floor-06k/` back without pickles or files over 2 MB, checking every file
   by SHA-256. Copy the run directories' small files (manifest, status, logs) to the laptop's
   `var/runs/`.
2. Re-score the predictions from the copied tables with your own reader.
3. Make the records commit:
   - `docs/levin-phase-floor/06k/`, with its `KEPT_ON_MAC_STUDIO.md`;
   - the log in `logs/`, and `VERIFICATION.md` §6k;
   - the board and `docs/OPEN_ISSUES.md`;
   - the CLAUDE.md black list for the copied scripts and variants.
4. Report the results per route with what each implies for a remedy and for 07. Ask the user about
   the worktree, the three census run directories and the pickles.

## 5. Stop and ask the user

As 06j's note §5, with prompt 06k's §7. **Never touch 07's stashes or saved work.**
