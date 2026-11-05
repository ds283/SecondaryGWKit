# levin-phase-floor 06l — the trust yardstick with the halves' node errors signed (log, in progress)

## Predictions, copied before the first number

Copied verbatim from the prompt's §4 at 2026-11-05T09:07:54Z, before Gate K0's first unit. The timestamped copy is `predictions.md` (SHA-256 be2e6544e3ab5bb7f970e9dafa0d962aea458120dd3679019b5944f4b399df6d); this block is not rewritten.

> ## 4. Predictions, fixed before the run
> 
> README §5 rule 3. Copy them into the log unchanged. Each is scored **holds**, **fails** or
> **undecided** against the rule as written. None is a stop. "Outside the 720" means outside the 720
> stationary-point rows.
> 
> - **L1 — B does not trust 06h's kind of leaf.** **Rule:** of the 17 06h carrying leaves (L-b), B
>   trusts none.
> - **L2 — B trusts every first-population slow leaf.** **Rule:** B trusts all 75 of 06j's sampled slow
>   leaves with $\Delta_1/\phi_P < 2$.
> - **C1 — convergence comes back.** **Rule:** at least 90 % of V8's lost calls outside the 720, at
>   each rtol, converge under B.
> - **C2 — nothing converged is lost but to the node-limited stop.** **Rule:** every call outside the
>   720 that 06g converges and B does not ends `node_limited`.
> - **H1 — no referenced call gets worse.** **Rule:** on every one of the 115 matched references
>   outside the 720, $|v_B - {\rm ref}|/e_B \le \max(1,\ |v_g - {\rm ref}|/e_g)$.
> - **S1 — no move past both errors.** **Rule:** no call outside the 720 moves by more than $e_B + e_g$
>   against 06g's census.
> - **K1 — B trusts Levin parents only at noise level.** **Rule:** every trust firing on a Levin
>   parent, outside the 720, has $\Delta_1/\phi_P < 5.8$.
> 
> **Descriptive, not scored:** the same counts for Clenshaw–Curtis parents, by $\Delta_1/\phi_P$ and by
> $\Delta_1/(|E_P| + |E_L + E_R|)$; how many of 06j's second population B trusts, and why.
> 
> **The basis.**
> - **L1, H1, S1 and K1** are 06k's safety constraint.
>   - L1: 06h's leaves are Levin over Levin, where $\nu_B = \nu_1$ and route 1 trusted none
>     ($\Delta_1/\nu_1 \ge 5.7$).
>   - K1: on a Levin parent, $\nu_B$ holds only Levin floors. In 06k's L-c sample a Levin half's
>     floor is at most 0.96 of its parent's, so $\nu_B \le 3\phi_P$ there.
>   - K1 is scored on Levin parents only. On a Clenshaw–Curtis parent $\phi_P$ understates the
>     rounding in $\Delta_1$ (§0.1), so $\Delta_1/\phi_P$ against 5.8, a threshold taken from Levin
>     leaves, is not a test of noise there.
> - **L2** is §0.2's guarantee, checked. 06j measured $r$ at 0.489–0.506 on these leaves, so the
>   $r < 1$ condition holds.
> - **C1** keeps 06k's rule for comparison. The leaf-level estimate (about 60 % of sampled slow leaves
>   trusted, against route 1's 80 %) makes a shortfall likely, and the recovered fraction itself is the
>   measure.
> - **C2** follows the user's ruling that losses to the node-limited stop are
>   `[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`'s, not the route's.
> 
> B passes as a candidate remedy if L1, L2, C2, H1, S1 and K1 hold. C1 and the comparison with route 1
> say how much it recovers, and what A's extra recovery rests on.

<!-- the predictions block above is not edited; the log's other sections are added below as the work goes -->

## Working notes (in progress; folded into README §5.1's sections at the end)

1. **09:07–09:19Z.** Predictions written (`predictions.md`, 09:07:47Z) and copied here before any number. Gate checks
   `out/cmd/00`–`07`: every clause holds (worktree `af4c4dc1` clean with `--ignored`; `34437951` and `af4c4dc1` on
   `origin/handover-remedial`; production diff since `1c5336e` only `CLAUDE.md` and `tools/filon_reference.py` with its
   test; main checkout `28ebaf07` clean; 55 runs, all finished; lock free; scratch copy EQUAL to its certification,
   fingerprint `0e32f24a…`; 06g's checkpoint `03adcbba…` with 16 pickles, `v8_calls.json` `1e9a62b0…`; route 1's
   checkpoint `36e3dd4c…` with 16 pickles and both extracts matching 06k's `KEPT_ON_MAC_STUDIO.txt`; 85 GB free;
   no Ray process).
2. Scripts copied from `docs/levin-phase-floor/06k/scripts/` (variants excluded); a read-only copy is
   `state/scripts_06k_reference/`. Variants built (`out/cmd/10c`, AST `11c`: CONFINED): identity `d89c959e…` (the
   tree's file), route B `5aa6178a…`. 06k's four routes rebuilt in memory have 06k's recorded SHA-256.
3. Development unit `g-1e-7-2-1449` (outside K0's 100) under route B (`out/cmd/14`, `15`): 19 calls with no firing
   equal 06g's; 2 calls fire.
4. **09:19:42Z** launched `levin-phase-floor-06l-k0partL-20261105T091942` (captures, K0 identity and route B, Part L
   probes, LiouvilleGreen draws).
5. **09:52Z, run k0partL `failed`.** `31_check_captures` exited 1: all 46 captures were refused on the harness-script
   check alone (`scripts_sha256_differing` = prov.py, run_census.py, variant_loader.py, worker.py; the copied check
   allows all but run_census.py). Every capture's region arrays and call record equal 06g's census (no region or call
   key differing on any of the 46). run_census.py is 06k's final copy (`c19d9327…`, with 06k's `--variant` and
   backstop), which 06k changed after its own captures; worker.py hashes it in SCRIPTS and does not import it. So
   `32_slice` sliced none and the 46 probes were skipped. Captures, K0 (both configurations and both comparisons) and
   the draws exited 0. A fault of my harness (deviation, UNINTENDED DRIFT, a failed attempt). Fixed in the copies:
   check_captures.py allows run_census.py (hashes recorded); slice_06j.py takes an optional OUT_JSON; launch_06l.py
   gains plan `partL`. The failed run's records are left as written.
6. **09:54:07Z** launched `levin-phase-floor-06l-partL-20261105T095407` (check again into
   `out/check_captures_b.json`, slicing into `state/inputs/` and `out/inputs_b.json`, the 46 probes).
7. **09:55Z, run partL `done`** (48/48, exit 0): 46 of 46 captures reproduce 06g's census bit for bit
   (`out/check_captures_b.json`), 46 sliced (`out/inputs_b.json`), 46 probes (`out/partL/`).
8. **Scored before stop 1** (`out/cmd/80`–`87`):
   - **Gate K0 holds** (`out/k0/compare_identity.json`, `compare_routeB.json`, `summary.json`): identity 100/100 units
     equal whole (counters included); route B equal to 06g on all 1,862 calls with no firing (16 units whole); 248 calls
     fire; 0 withdrawals.
   - **Part L** (`out/partL_leaves.csv`, `out/partL_summary.json`): 2,046 leaves; all 42 of 06k's columns equal 06k's
     table on every row; L1 holds (0 of 17); **L2 fails** (63 of 75; account `out/partL_l2/account.json`: on the 12,
     E_P and E_L + E_R have opposite signs, so the triangle bound is tight, and delta1 exceeds the node-error
     difference by 0.20–0.61 %, more than the rest of nu_B, about 0.15–0.27 %, covers; delta1/nu_B 1.0004–1.0041);
     **Gate L3 holds** on all 2,046 (nu_B/nu1 at most 1 + 2.2e-16; no leaf B trusts that route 1 does not).
   - **Draws** (`out/lg_draws/compare.txt`): identity equals the tree on all 4,320 Levin calls; route B fires in 122
     calls, changes no convergence and moves no case between honest and dishonest.
   - **Census plan** (`out/census_projection.json`): rho (route B / identity on K0) 0.730; P = 17,347 s.
9. **10:01:26Z** launched the census `levin-phase-floor-06l-census-routeB-20261105T100126` (route B `5aa6178a…`,
   run_census.py `280fe637…`, projection 17,347 s; sh pid 54724, driver pid 54726), after the orchestrator accepted
   stop 1. Six 1e-7 workers started at the one check (`state/launch_census_routeB_check.txt`).
