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

## B.5 Addendum, 2026-11-05: the run, the review and the records

**Dispatched** 2026-11-04 on the user's word: *"Yes, please push and dispatch on the Mac Studio"*.
The laptop pushed `89370609` and made the worktree `SecondaryGWKit-lpf06k-8937060`, detached at
`89370609`, on the Mac Studio, whose main checkout was clean at `28ebaf0`. It then started a Mac
Studio orchestrator session in the worktree.

**The run and the review on the Mac Studio.**
- **§B.1's preflight passed.** The marker was set at 08:48:43Z; 89 GB were free.
- **Planned stop 1** (after Gate K0 and Part L, 09:16–10:10Z):
  - The orchestrator's AST comparison and its K0 checks held, and the census plan was accepted.
  - It asked the user two questions. **Q1** was whether route 2 should also be measured as §0.2 reads
    literally. The user: *"Q1: I don't actually see why the literal reading cannot hold. It just
    says every leaf is decided and declared on the corrected numbers. That seems perfectly
    reasonable."* Told that the literal reading withdraws trust from 326 leaves, 265 of them at
    $r < 1/4$: *"OK, so this isn't really a contradiction, it's just that the note said that trust
    would be preserved if the leaf had `r < 1/4`, and it might not be. We had better run that
    fourth census."* **Q2** was whether route 3 should also rank an untrusted leaf at `depth_max`:
    *"Q2: stick with what has been built."*
  - So a fourth variant, route 2 literal, was built, checked as at stop 1 (its K0 and draws in run
    `levin-phase-floor-06k-route2lit-20261104T101930`), and given a fourth census.
- **The censuses**, one at a time, each cleared by the orchestrator (planned stop 2) and waited on by
  a background `kill -0` loop on its pid:

  | Route | Run | Wall | Projection |
  |---|---|---|---|
  | 1 | `levin-phase-floor-06k-census-route1-20261104T103009` | 16,758 s | 16,183 s |
  | 2 | `levin-phase-floor-06k-census-route2-20261104T151120` | 13,621 s | 14,332 s |
  | 2 literal | `levin-phase-floor-06k-census-route2lit-20261104T185927` | 13,968 s | 15,037 s |
  | 3 | `levin-phase-floor-06k-census-route3-20261104T225339` | 7,626 s | 11,866 s |

  Each was `done`, 18,560 of 18,560, exit 0, with stderr empty. The backstop reported once (route 1)
  and stopped nothing.
- **A §7 stop, "a reference cannot be matched to its call."** V1's 12 references are calls of 06g's
  rehearsal store R24, not of A3 v2's census. Listing them in prompt 06k §3 was this orchestrator's
  error: 06g's `VERIFICATION.md` §6g.4 records them as captured at R24's tree. The agent offered (a),
  scoring H1 on the 124 matched references, or (b), re-running V1's calls on R24 under each variant.
  The orchestrator recommended (a), with V1 dropped as a prompt error so that H1 is scored, not left
  undecided. The user: *"(a) as recommended, please"*. It is the log's deviation 24, and nothing more
  was computed.
- **The review** (`docs/levin-phase-floor/06k/review.md`) passed all seven checks. It re-ran two K0
  units and one recovered call, recomputed Part L on five leaves, re-counted C1, C2 and S1 from the
  raw checkpoints and recomputed eight reference ratios, each with its own scripts. Its three
  observations change no verdict: S1's "rows" are census units; the four `out/partC/<route>.json`
  have no header of their own; and its check 4 did not re-solve the children. The first two are the
  log's deviations 29 and 30.

**The results:** K0 holds, and no route passes as a candidate remedy. Route 1 fails only C2 (7 calls
`node_limited` at `1e-8`) and K1. Routes 2 and 2 literal fail L2, C2, S1 and K1. Route 3 fails L1,
C2, H1 and S1. `VERIFICATION.md` §6k has the numbers.

**The laptop's checks.**
- **The copy.** `var/levin-phase-floor-06k/` was copied back by `rsync` without pickles or files over
  2 MB: 1,154 files, each equal to the Mac Studio's by SHA-256. The seven run directories'
  manifests, statuses and logs, and the three small runs' checkpoints, were copied to `var/runs/`
  and checked the same way (31 files).
- **The re-scoring**, by the orchestrator's own reader from the copied tables:
  - L1 and L2: each route's decision recomputed from the leaf's numbers on all 2,046 Part L leaves,
    with no disagreement with the probes;
  - H1 from `h1_calls.csv`, with 06g's ratio checked against 06g's `scores.json` and 06j's
    `w_calls.csv` on all 124 references;
  - C1, C2, S1 and K1 from `out/partC/<route>.json`.

  Every verdict and number is the same, with S1's rows counted shard-serial (5, 5 and 22).
- **The CSV tables.** Three were added past the laptop's `*.csv` exclusion.

**Issues:**
- Narrowed: `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`.
- Opened: none. The agent proposed `[06k-route3-left-leaf-declaration-understates]` and
  `[06k-route2-corrected-declaration-omits-the-node-error-of-q]`. Both describe scratch variants
  that are not in the tree, so they are recorded under the slow-leaves issue as constraints on a
  remedy.

`docs/OPEN_ISSUES.md` stays at 126 open.

**For the user:**
- whether a remedy prompt implements a route, and which;
- the worktree `SecondaryGWKit-lpf06k-8937060`;
- the four census run directories (17.1 GB in all);
- 06k's pickles and large files (1.1 GB).

07 resumes after that.
