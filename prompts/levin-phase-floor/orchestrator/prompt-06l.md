# Orchestrator — prompt 06l, the trust yardstick with the halves' node errors signed

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code**, and the measurements are the agent's, not yours.

**The prompt:** [`prompts/levin-phase-floor/06l-signed-trust-yardstick.md`](../06l-signed-trust-yardstick.md)
**Board item:** 06l · **Closes:** nothing. Its records measure route B for
`[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]` beside 06k's route 1; which
route 06m implements is for the user.
**Model:** Opus.

It is a measurement on the Mac Studio only, with no code commit. The machinery is 06k's note §B: an
orchestrator session on the Mac Studio, through `ssh macstudio-tunnel`, dispatches and reviews. What
differs is one variant, one census, and a second "before": route 1's census.

**Dispatched on the user's word.**

## 0. The order

1. **This prompt's commit is the dispatch SHA.** Push it. The laptop then makes the worktree
   `SecondaryGWKit-lpf06l-<short SHA>` on the Mac Studio, detached, with no venv.
   `git diff --stat 1c5336e <dispatch SHA> -- . ':!docs' ':!prompts'` shows only
   `tools/filon_reference.py`, `tools/tests/test_filon_reference.py` and `CLAUDE.md`.
2. **06l runs** on the Mac Studio: Gate K0, Part L with Gate L3, then the census. One agent; the Mac
   Studio orchestrator reviews it.
3. **The laptop** copies the records back and makes 06l's records commit.
4. **The user decides** from 06l's results whether 06m implements B or A (route 1 with the
   Levin-parent exclusion). Then 07 resumes. Neither starts here.

## B. On the Mac Studio

### B.0 Facts, checked by the orchestrator on the laptop, 2026-11-05

- **06g's census** `levin-phase-floor-06g-census-20261101T114315` is kept in full on the Mac Studio:
  checkpoint `03adcbba8a78d82d9d6b899b8eee8b9ae1659763f52deb066d2840af3d82c0fa`, 16 region pickles,
  20,361 s. `census/out/v8_calls.json` is `1e9a62b0…`.
- **Route 1's census** `levin-phase-floor-06k-census-route1-20261104T103009` is kept in full:
  checkpoint `36e3dd4c9125f6ea620b80e1bc15d656827b5dd2e449c983d479d1577c6bed38`, 16 region pickles,
  16,758 s. Its firings carry `is_direct`, $\phi_P$, $\phi_L$, $\phi_R$ and $\Delta_1$, but not the
  halves' kinds or $E$.
- **06k's compact extracts** `var/levin-phase-floor-06k/state/extract/g.pkl` and `route1.pkl` are
  kept; the other four were deleted on 2026-11-05.
- **The orchestrator's count from route 1's checkpoint** (prompt §0.1): 60,609 firings on Levin
  parents (48,131 at $\Delta_1/\phi_P \ge 5.8$), 206,969 on Clenshaw–Curtis parents (56,531). The
  scan took 22 s.
- **06k's leaf table** `docs/levin-phase-floor/06k/out/partL_leaves.csv` has $E_P$, $E_L$, $E_R$ and
  the floors for every Part L leaf, but not `phase_err_less_node`. The orchestrator's estimate of B
  from it (prompt §0.2) used $\phi_h - |E_h|$ in its place: 85 of 142 slow leaves trusted, 7 of them
  at $\Delta_1/\phi_P \ge 5.8$; none of L-b; all of L-c.
- **06j's 46 captures** are reproducible: 06k re-captured them in its run `k0partL` and checked each
  against 06g's census bit for bit. 06k's captures were deleted on 2026-11-05.
- **The references** are committed: 06g's `rows` (102) and 06j's `w_calls.csv` (22). V1's 12 are R24
  calls and are not used.
- **The scratch copy** `var/levin-phase-floor-06/wider/state/a3v2_copy/` is certified; its fingerprint
  is `0e32f24a…`.
- **The Mac Studio** had 87 GB free on 2026-11-05.
- **06g's four worktrees** `SecondaryGWKit-lpf06g-{1c5336e,4f4f9ed,b5626ec,ff3e839}` stay, on the
  user's word; neither you nor the agent touches them.
- **No Ray, no lock, no store built.** If anything needs Ray, that is a stop.

### B.1 Before you dispatch

1. `ssh macstudio-tunnel`; `runregistrykit list` shows nothing `running` or stale; the main
   checkout is clean; the worktree is at the dispatch SHA and clean, including ignored files.
2. **SHA-256 checks.** The scratch copy matches its certification, with no `-wal` or `-shm`. 06g's
   census checkpoint, its 16 region pickles and `v8_calls.json` match theirs; so do route 1's
   checkpoint and its 16 region pickles.
3. **Disk.** At least 20 GB free.
4. `touch` a marker; record the `var/` listing, as 06k's preflight did.

### B.2 Dispatch

One fresh-context agent, on **Opus**, in the worktree. It is given the prompt, the dispatch SHA and
§B.0, and it does not read `orchestrator/`. It writes only under `var/levin-phase-floor-06l/` and,
for registered runs, `var/runs/`. It never touches:
- `var/levin-phase-floor-07*`, or 06g's, 06j's and 06k's records and runs, which it reads only;
- the scratch copy's directory, beyond reading the copy;
- any store.

It makes no commit and runs no `git` write. The variants live under its own `scripts/`, never in the
tree.

**Two planned stops, each checked here before the agent goes on:**
1. **After Gate K0, Part L and Gate L3, before the census:**
   - **The variant's diff:** read it and run your own AST comparison against the tree's module. Only
     `_adaptive_levin` may differ, and within it only `_check_leaf`'s noise clause and the firing
     record.
   - **$\nu_B$ read against prompt §0.2:** each of the four parent and halves cases, the Levin
     parent's exclusion of a Clenshaw–Curtis half, $\phi^{-}$ from `phase_err_less_node`, and the
     signed sum over the Clenshaw–Curtis halves only.
   - **K0's and L3's records:** the identity variant bit-identical on all 100 units, B equal to 06g
     wherever it never fires, and $\nu_B \le \nu_1$ with B's trust inside route 1's on every leaf.
   - **L2's leaves:** all 75 first-population leaves trusted, or the agent's account of each that is
     not.
   - **The census plan:** the projection, between route 1's 16,758 s and 06g's 20,361 s, and the
     backstop, which stops a run only at a projected day or more, counting the units in flight.
2. **After the census:** one check that it is `done`, 18,560 of 18,560, before the agent scores it.

Wait on each detached run with a background `kill -0` loop on its pid, not by polling.

### B.3 The review — on the Mac Studio

1. **Scope:**
   - nothing under `var/` newer than the marker outside `var/levin-phase-floor-06l/` and the 06l
     runs;
   - the scratch copy, 06g's checkpoint and route 1's checkpoint unchanged by SHA-256;
   - the worktree clean, including ignored files;
   - no Ray started; nothing new in `var/datastores/`.
2. **The variant and K0:**
   - your own AST comparison of the variant against the tree's module;
   - **re-run two K0 units yourself,** each under the identity variant and under B, and check their
     bit-identity with 06g's census record.
3. **The predictions:** L1, L2, C1, C2, H1, S1 and K1 copied verbatim into the log before the first
   number, by its timestamps.
4. **Part L, re-checked:** for three first-population slow leaves, two second-population slow leaves
   and one 06h carrying leaf, recompute $\nu_B$ yourself from the captured halves' solves
   (`phase_err_less_node`, `node_error_signed`, each half's kind), and B's decision.
5. **Part C, re-checked:**
   - recompute C1's, C2's and S1's counts from B's census checkpoint and 06g's, with your own reader;
   - recompute the count of calls route 1 converges and B does not, from the two checkpoints;
   - **re-run one call** that B converges and 06g does not, under B, and check its value, abserr,
     regions, `converged` and firing count against the census record.
6. **The references:** for three calls, recompute $|v_B - {\rm ref}|/e_B$ from the checkpoint and the
   reference.
7. **The records:**
   - README §5.1's headings, with every deviation classified;
   - `results.md` proposes no code;
   - provenance headers on every output, the per-route tables included (06k's deviation 30);
   - rows counted shard-serial, units apart (06k's deviation 29);
   - what is kept on the Mac Studio is listed with SHA-256.

Write `var/levin-phase-floor-06l/review.md`, with the verdict in its first line, and report.

### B.4 On the laptop, after the review

1. Copy `var/levin-phase-floor-06l/` back without pickles or files over 2 MB, checking every file
   by SHA-256. Copy the run directories' small files (manifest, status, logs) to the laptop's
   `var/runs/`.
2. Re-score the predictions from the copied tables with your own reader.
3. Make the records commit:
   - `docs/levin-phase-floor/06l/`, with its `KEPT_ON_MAC_STUDIO.md`;
   - the log in `logs/`, and `VERIFICATION.md` §6l;
   - the board and `docs/OPEN_ISSUES.md`;
   - the CLAUDE.md black list for the copied scripts and the variant.
4. Report B beside route 1 and A: what each recovers, on what grounds its trust rests, and what each
   implies for 06m and for 07. Ask the user which 06m implements, and about the worktree, the census
   run directory and the pickles.

## 5. Stop and ask the user

As 06k's note §5, with prompt 06l's §7. **Never touch 07's stashes or saved work.**

## B.5 Addendum, 2026-11-05: the run, the review and the records

**Dispatched** 2026-11-05 on the user's word: *"Yes, please push and dispatch"*. The laptop pushed
`af4c4dc1` and made the worktree `SecondaryGWKit-lpf06l-af4c4dc`, detached at `af4c4dc1`, on the Mac
Studio, whose main checkout was clean at `28ebaf07`. The orchestrator session could not start until
the user trusted the worktree's folder in the app; it then started in the worktree.

**The run and the review on the Mac Studio.**
- **§B.1's preflight passed.** The marker was set at 09:06:04Z; 85 GB were free.
- **The first run failed.** `levin-phase-floor-06l-k0partL-20261105T091942` ended `failed` (exit 2):
  its capture check refused all 46 captures because `run_census.py`, 06k's edited copy, was not on
  its allowed list of harness hashes. Every capture's values equalled 06g's census, which the
  orchestrator checked itself, so it was not a §7 stop. The agent fixed the list and re-ran the
  check, slicing and probes as `levin-phase-floor-06l-partL-20261105T095407`, 48 of 48 (log
  deviation 11). K0 and the draws were scored from the first run, whose commands for them exited 0.
- **Planned stop 1** (`review/stop1.md`): the orchestrator's AST comparison, $\nu_B$ read against
  §0.2, K0 and Gate L3 recomputed by its own reader all held. L2 was found to fail, 63 of 75, as a
  scored prediction, not a stop. The census plan (17,347 s) was accepted.
- **The census** `levin-phase-floor-06l-census-routeB-20261105T100126` was waited on by a background
  `kill -0` loop on its pid. It was `done`, 18,560 of 18,560, exit 0, with stderr empty, in
  18,340 s. The backstop reported twice and stopped nothing. Planned stop 2 passed.
- **The agent asked no question**, and no §7 stop arose.
- **The review** (`docs/levin-phase-floor/06l/review/review.md`) passed all seven checks. It re-ran two
  K0 units and one recovered call, re-made the seven solves of six Part L leaves, re-counted C1, C2,
  S1, K1 and the comparison with route 1 from the raw checkpoints, and recomputed five reference
  ratios, each with its own scripts. Its two observations change no verdict: twelve harness outputs
  carry no header of their own (the log's deviation 22), and the agent's final report miscounted
  its deviations by class.

**The results:** Gates K0 and L3 hold. L1, C1, C2, H1, S1 and K1 hold; L2 fails, so by prompt §4's
rule B does not pass as a candidate remedy. `VERIFICATION.md` §6l has the numbers.

**The orchestrator's error in L2.** L2 asked B to trust all 75 first-population slow leaves, on
§0.2's triangle-inequality argument. 06k's committed table already had route 1 trusting 63 of them,
and Gate L3, set by the same prompt, makes route 1's trust an upper bound for B's. B trusts exactly
those 63. The argument is first-order only: where $E_P$ and $E_L + E_R$ have opposite signs, it is
tight, and $\Delta_1$ exceeds it by 0.2–0.6 %. Checking L2 against 06k's table before dispatch would
have caught it. L2 is scored as written; it does not separate B from route 1 or A.

**The laptop's checks.**
- **The copy.** `var/levin-phase-floor-06l/` was copied back by `rsync` without pickles or files over
  2 MB: 824 files, each equal to the Mac Studio's by SHA-256. The three run directories' small files
  (143) were copied to `var/runs/` and checked the same way.
- **The re-scoring**, by the orchestrator's own reader from the copied tables:
  - L1, L2 and L3: $\nu_B$ and B's decision recomputed from each leaf's numbers on all 2,046 Part L
    leaves, with no disagreement, and B's 63 checked equal to route 1's;
  - H1 from `h1_calls.csv`;
  - C1, C2, S1 and K1 from `out/partC/summary.json`, equal to the review's count from the raw
    checkpoints.

  Every verdict and number is the same.
- **The CSV tables.** Two were added past the laptop's `*.csv` exclusion.

**Issues:**
- Narrowed: `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]`.
- Opened: none.

`docs/OPEN_ISSUES.md` stays at 126 open.

**For the user:**
- whether 06m implements B, B with a margin, A, or none;
- the worktree `SecondaryGWKit-lpf06l-af4c4dc`;
- route B's census run directory (4.4 GB);
- 06l's pickles and large files (0.4 GB).

07 resumes after that.

**The user's decisions, 2026-11-05.** On 06m: *"I think B as measured is the right choice, as far as
I can see."* On the Mac Studio: *"yes, please clean up. Remove the worktree. For now keep the census
run directory. Remove the pickles and other large files."* The laptop removed the worktree
`SecondaryGWKit-lpf06l-af4c4dc` (clean with `--ignored`; 06g's four worktrees untouched), and deleted
the 183 pickles and large files under `var/levin-phase-floor-06l/`, each first checked against its
SHA-256 (`docs/levin-phase-floor/06l/KEPT_ON_MAC_STUDIO.md`). Route B's census run directory and the
two small runs are kept.
