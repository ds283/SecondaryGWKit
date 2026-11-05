# levin-phase-floor 06l — the trust yardstick with the halves' node errors signed, measured on A3 v2 (a measurement)

**Subject:** prompt 06l (`prompts/levin-phase-floor/06l-signed-trust-yardstick.md`), on the Mac Studio (mps026298),
2026-11-05, from the worktree `SecondaryGWKit-lpf06l-af4c4dc` at the dispatch SHA `af4c4dc1` (production code
`1c5336e`'s, 06g's).
**Commit:** none; records committed by the orchestrator from the laptop.
**Result:** Gates K0 and L3 hold. L1, C1, C2, H1, S1 and K1 hold; **L2 fails** (63 of 75), so route B does not pass
as a candidate remedy by §4's rule as written. Its census holds every safety rule.

## Predictions and verdicts

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

<!-- the block above is the predictions as copied at 09:07:54Z, unchanged (its body equals predictions.md, written
09:07:47Z, SHA-256 be2e6544…); the in-progress log with it is state/log_md_before_final.md; verdicts follow -->

| Prediction | verdict | the number it was scored on |
|---|---|---|
| **L1** | **holds** | 0 of the 17 distinct 06h carrying leaves trusted (22 evaluations, none) |
| **L2** | **fails** | 63 of 75 first-population slow leaves trusted; the 12 others at Δ1/ν_B 1.0004–1.0041 |
| **C1** | **holds** | 749 of 801 (93.5 %) at 1e-7, 773 of 830 (93.1 %) at 1e-8 |
| **C2** | **holds** | 4 calls outside the 720 that 06g converges and B does not, all `node_limited`, all at 1e-8 |
| **H1** | **holds** | 115 of 115 matched references outside the 720 at ≤ max(1, 06g's ratio); worst v03 1.182 (06g 1.203) |
| **S1** | **holds** | 0 calls outside the 720 move by more than e_B + e_g |
| **K1** | **holds** | 9,839 trust firings on Levin parents outside the 720, none at Δ1/φ_P ≥ 5.8 (max 2.85) |

**Notes on the scoring.**
- **L2's 12** are Clenshaw–Curtis over Clenshaw–Curtis leaves. On each, E_P and E_L + E_R have opposite signs, so
  §0.2's triangle bound is tight, and Δ1 exceeds the node-error difference by 0.20–0.61 %, which the rest of ν_B does
  not cover (`out/partL_l2/account.json`). The rule is scored as written. Route 1 trusts the same 63 of 75.
- **H1** is scored on the 124 references the census can match (115 outside the 720), as 06k did. V1's 12 are R24 calls
  and are not used, per the prompt.
- **C2** is scored on group and analytic calls; every loss is a group call.
- **K1** counts B's `fired` records on Levin parents outside the 720, in group and analytic calls. Inside the 720:
  1,901, none at ≥ 5.8.
- **S1:** units and shard-serial rows are counted apart, and are 0 and 0 outside the 720.
- **The candidate rule** (L1, L2, C2, H1, S1 and K1): **B does not pass, on L2 alone.**
- **Where the numbers are:** `out/results.md`, `out/partC/summary.json`, `out/partC/routeB.json`,
  `out/partC/against_A.json`, `out/partC/h1.json`, `out/partL_summary.json` and `out/k0/summary.json`.

## What shipped

No code. The records are in `var/levin-phase-floor-06l/`: `log.md` (this file), `predictions.md`, `verification.md`,
`issues.md`, `KEPT_ON_MAC_STUDIO.txt`, and the directories below.

**`out/`**
- every command with its output (`out/cmd/`, `out/commands.log`);
- the variants' diffs (`out/variant_identity_diff.txt` is empty; `out/variant_routeB_diff.txt`;
  `out/variant_routeB_vs_route1_diff.txt`) and AST check (`out/cmd/11c`);
- K0's records (`out/k0/`): `compare_identity.json`, `compare_routeB.json`, `summary.json`;
- Part L's tables: `out/partL_leaves.csv`, `out/partL_summary.json`, `out/partL/` (46 probes) and
  `out/partL_l2/account.json`;
- the captures' checks (`out/check_captures.json` from the failed attempt, `out/check_captures_b.json`) and
  `out/inputs_b.json`;
- the draws (`out/lg_draws/`), the census projection (`out/census_projection.json`), and Part C (`out/partC/`):
  `routeB.json`, `summary.json`, `against_A.json`, `h1.json`, `h1_calls.csv`;
- `out/results.md`;
- the scripts' diffs against 06k's copies (`out/scripts_diff_at_stop1.txt`, `out/scripts_diff_final.txt`).

**`scripts/`** holds 06k's scripts, edited where marked "prompt 06l", and 06l's own. `scripts/variants/` holds the
identity variant (`d89c959e…`, the tree's file) and route B (`5aa6178a…`), both read-only.

**`state/`**
- the copy of 06k's scripts (`scripts_06k_reference/`, read-only);
- the earlier builds' scripts and the first route B build;
- the plans and the launch transcripts with their checks;
- the captures (`state/capture`, `state/inputs`), K0's worker directories (`state/k0/`), the development unit
  (`state/dev/`), `state/index_06g.json` and the copy of 06g's `rows.json`;
- route B's census extract (`state/extract/routeB.pkl`);
- the in-progress log (`state/log_md_before_final.md`).

**Three registered runs, every one finished:**
- `levin-phase-floor-06l-k0partL-20261105T091942`: `failed`, for its capture check, slicing and probes alone
  (deviation 11). Its captures, K0 and draws exited 0.
- `levin-phase-floor-06l-partL-20261105T095407`: `done`, 48/48.
- `levin-phase-floor-06l-census-routeB-20261105T100126`: `done`, 18,560/18,560, 18,340 s. Checkpoint `5cb3d3c0…`.

## The stages, in order

1. **09:07–09:20Z.**
   - The brief, the prompt and its reading list read.
   - Predictions written (09:07:47Z) and copied into this log (09:07:54Z).
   - The Gate checks (`out/cmd/00`–`07`).
   - Scripts copied; the variants built and AST-checked; the harness edited.
   - One development unit run outside the scored set.
   - Run `k0partL` launched.
2. **09:52Z.** Run `k0partL` failed in its capture check alone. The check was fixed, and run `partL` launched at
   09:54Z and finished at 09:55Z.
3. **09:55–10:00Z.** K0, Part L, L3 and the draws scored; the census planned; planned stop 1. The orchestrator
   accepted all of it.
4. **10:01–15:07Z.** The census, launched on the orchestrator's word. Planned stop 2: done, 18,560/18,560.
5. **15:09–15:20Z.** Part C scored, the references scored, the after-checks run, and the records written.

## Deviations from the prompt

1. **`runregistrykit list` was run twice for Gate check 01** (`out/commands.log` holds both; the cmd file holds the
   second). **UNINTENDED DRIFT** (cosmetic).
2. **A file list was written to `/tmp/x` while the scripts were copied**, then removed. It was not a record.
   **UNINTENDED DRIFT.**
3. **The copied scripts' changes,** each marked "prompt 06l". The diffs are `out/scripts_diff_at_stop1.txt` and
   `out/scripts_diff_final.txt`. **STRUCTURALLY REQUIRED.**
   - `prov.py`, `refs_common.py` and `calibration/prov.py`: the tree and the records directory.
   - `worker.py`: route B's result key `route_06l`, kept in the call summary and the pickle record.
   - `check_captures.py` and `compare_k0.py`: `route_06l` excluded as `route_06k` is, and `compare_k0.py` reads the
     firing count from it.
   - `run_k0.py`, `run_census.py` and `calibration/lg_seeded_draws.py`: the `routeB` variant or mode.
   - `probe_l.py`: B's numbers from the same seven solves.
   - `ast_check_variants.py`: the `_06l` suffix.
   - `index_06g.py`: the title.
   - `extract_census.py`: route B's record.
   - `slice_06j.py`: an optional output path (deviation 11).
4. **Route B was built three times before any scored run** (`out/cmd/10`, `10b`, `10c`). The first build failed the
   AST check's name rule, on a comprehension variable `h`, and was rebuilt with `h_06l` and `x_06l`. The first build's
   script and variant are `state/make_variants_as_first_run.py` and `state/levin_quadrature_routeB_first_build.py`.
   **UNINTENDED DRIFT** (a failed attempt).
5. **The third build added the record kind `withdrawn`:** a leaf 06g's rule trusts that B does not, which only a NaN
   floor could produce. It was recorded so that any such case would be seen. It occurred 0 times in K0, the draws and
   the census. The second build's script is `state/make_variants_as_second_run.py`. **IMPLEMENTATION CHOICE.**
6. **B's record also holds the kind `untrusted`:** a leaf B leaves untrusted that route 1's yardstick would trust, or
   that sits at `depth_max`, with every field of a firing. It is there for §3's comparison with route 1, which needs
   the halves' kinds of those leaves, and route 1's census does not record them. `n_firings` counts only `fired`. The
   value returned is unchanged, and the AST check places the record inside `_check_leaf`'s trust statement.
   **IMPLEMENTATION CHOICE.**
7. **`make_variants.py` writes only the identity variant and route B.** 06k's four routes are rebuilt in memory and
   hashed beside 06k's recorded SHA-256 (all equal), not written. **IMPLEMENTATION CHOICE.**
8. **K0 has no tree-module control** (06k's deviation 6). The identity variant is the tree's file byte for byte.
   **IMPLEMENTATION CHOICE.**
9. **`state/index_06g.json` was rebuilt** after a provenance comment was added to `index_06g.py` (`out/cmd/13`,
   `13b`). The two builds' units are equal; the first is kept as `state/index_06g_first_build_superseded.json`.
   **UNINTENDED DRIFT.**
10. **A development unit was run outside the scored set:** `g-1e-7-2-1449` under route B (`out/cmd/14`, `15`). Its
    19 calls with no firing are equal to 06g's, and 2 calls fire. It also exercised the draws' header
    (`out/cmd/16`). None of it is scored. **IMPLEMENTATION CHOICE.**
11. **Run `k0partL`'s capture check refused all 46 captures** (`out/cmd/k0partL/31_check_captures.txt`).
    - **The cause:** the check refused them on the harness-script hashes alone. `run_census.py` was not on its allowed
      list. It is 06k's final copy, changed after 06k's own captures had run; worker.py hashes it and does not import
      it. Every capture's region arrays and call record equalled 06g's census.
    - **What followed:** `32_slice` sliced none, and the 46 probes were skipped.
    - **The fix:** `check_captures.py` now allows `run_census.py`, with its two hashes recorded, and `slice_06j.py`
      takes an output path.
    - **The re-run:** the check, slicing and probes alone, as the registered run `partL` (`out/cmd/partL/`): 46 of 46
      reproduce, all sliced and probed.
    - The failed run's records are left as written.

    **UNINTENDED DRIFT** (a failed attempt).
12. **K0 and the draws were scored from run `k0partL`,** whose commands for them exited 0. Only its capture check,
    slicing and probes were re-run (06k's deviation 8 is the precedent). The orchestrator checked and accepted this at
    planned stop 1. **IMPLEMENTATION CHOICE.**
13. **New scoring scripts:** `k0_summary_06l.py`, `score_l_06l.py`, `l2_account_06l.py`,
    `calibration/lg_compare_06l.py`, `census_projection_06l.py`, `analyse_c_06l.py`, `against_a_06l.py` and
    `score_h1_06l.py` (the last copied from 06k's `score_h1.py` with its routes changed). They stand in for 06k's
    `k0_summary.py`, `score_l.py`, `lg_compare_06k.py`, `analyse_c.py` and `score_h1.py`:
    - each applies 06k's comparisons rule for rule to B;
    - each adds what §2–§4 ask of 06l;
    - they compare with 06g's census or route 1's, since 06l has no tree control and 06k's other routes' extracts are
      deleted.

    **IMPLEMENTATION CHOICE.**
14. **`score_l_06l.py` compares this run's 06k columns with 06k's committed `partL_leaves.csv`,** as text, on all
    2,046 rows. All are equal. **IMPLEMENTATION CHOICE.**
15. **The census projection** (`out/census_projection.json`):
    - It is 06k's form, P = 06h + (06g − 06h)·ρ. Here ρ = route B / identity on K0, today, in one run (0.730), which
      gives 17,347 s, inside §3's bracket (16,758–20,361 s).
    - The census took 18,340 s, 5.7 % more. The backstop reported twice and stopped nothing.
    - Today's identity run took 1.11× 06g's census walls on the same units (host load about 15).

    **IMPLEMENTATION CHOICE** (recorded for the cost).
16. **Part C reads compact extracts:**
    - route B's, made here by `extract_census.py` (extended; `out/cmd/100`);
    - 06k's of 06g's census and route 1's (`var/levin-phase-floor-06k/state/extract/g.pkl` and `route1.pkl`, read only;
      each records its checkpoint's SHA-256, which `analyse_c_06l.py` checks against `03adcbba…` and `36e3dd4c…`).

    `against_a_06l.py` reads route 1's checkpoint itself, hashing it in the same pass. **IMPLEMENTATION CHOICE.**
17. **06h's cost** (census wall, unit wall, group-call regions) is taken from 06k's committed `out/partC/summary.json`.
    06h's extract was deleted on 2026-11-05, and the cost needs nothing else of 06h's. **STRUCTURALLY REQUIRED.**
18. **`movers_c.py` was not run.** S1 has no mover outside the 720. The 720's one moving call, `0-2316` #1 at both
    rtols, is reported from `routeB.json` without its carrying cells: it is a stationary-point row, reported apart and
    not scored. **IMPLEMENTATION CHOICE.**
19. **A's difference is inferred, not measured** (§0.3: no census of A). A's decision is computed for each of route 1's
    firings in the calls route 1 converges and B does not:
    - **matching:** the firing is matched by its exact span to a leaf B withheld trust from, which gives the halves'
      kinds;
    - **A's decision:** from route 1's own numbers, φ_P plus the Levin halves' φ on a Levin parent, and route 1's
      decision on a Clenshaw–Curtis parent;
    - **classes:** each call is classed by what A would withhold.

    It shows what route 1's recovery rests on. It does not say A's outcome on the 75 calls where A withholds part of
    route 1's trust. **IMPLEMENTATION CHOICE.**
20. **The scopes the rules leave open:**
    - C2 and S1 count group and analytic calls alike.
    - K1 counts B's `fired` records whose parent is a Levin solve; the 6 with a Clenshaw–Curtis half are included.
    - H1's 22 from 06j include the 1e-8 twins f05–f09, each scored as its own call, as 06j and 06k did.

    **IMPLEMENTATION CHOICE.**
21. **Labels in two JSON tallies.** `out/k0/summary.json` and `out/partC/summary.json` label the top band of Δ1/ν_B
    "<1", where its edge is 1 + 10⁻¹². `results.md` states the intervals. **UNINTENDED DRIFT** (cosmetic).

## Verification performed

`verification.md` has the full text. In brief:
- **The Gate:** every clause passes (`out/cmd/00`–`07`, `06b`).
- **The store and the inputs.**
  - The scratch copy is EQUAL to its certification before, after, and after the final fingerprint (`03`/`04`,
    `110`/`111`, `113`/`114`). Its fingerprint is `0e32f24a…` before and after (`05`, `112`).
  - These 39 files are equal by SHA-256 before and after (`06`, `115`, `115c`):
    - 06g's census checkpoint and its 16 region pickles;
    - `v8_calls.json`, `v8_slow_leaves.json` and 06g's `rows.json`;
    - route 1's checkpoint and its 16 pickles;
    - both 06k extracts.
  - `var/datastores/` is unchanged: its listing's md5 is `3eb83380…` before and after, with 51 entries.
- **No Ray.**
  - All 16 census workers report Ray not initialised and no foreign module; their tree evidence holds with route B's
    file (`120`).
  - The same holds for all 15 workers of each K0 configuration (`out/cmd/k0partL/41`, `42`) and for every probe.
  - The lock is free, and no Ray process is listed (`118`, `119`).
- **The trees.**
  - The worktree is at `af4c4dc1`, clean with `--ignored`, before and after (`00`, `116`). No `__pycache__` was
    written anywhere.
  - The main checkout is at `28ebaf07`, clean.
  - The registry holds 58 runs, all finished, three of them 06l's (`117`).
- **The variants.**
  - Only `_adaptive_levin` differs from the tree's file. Inside it are the records list, `_check_leaf`'s trust
    statement with B's record, and the return key (`11c`).
  - The identity variant is the tree's file, byte for byte.
- **K0, Part L, L3 and the census:** as in the Result. The census ran with the variant whose SHA-256 every worker
  recorded (`5aa6178a…`).

## Observations not acted on

1. **§0.2's guarantee holds to first order only.**
   - On L2's 12 leaves, Δ1 exceeds |E_P − (E_L + E_R)| = |E_P| + |E_L + E_R| by 0.2–0.6 %. ν_B's remaining terms,
     about 0.15–0.27 % of it, do not cover that.
   - In the census, 52,169 of the CC + CC firings sit at Δ1/(|E_P| + |E_L + E_R|) in [1, 1.01).
   - The 231 Clenshaw–Curtis-parent decisions where B withholds route 1's trust miss ν_B by 0.0004 % to 3× (median
     1 %).
   - A yardstick with a margin for the node term's second-order remainder would reach these leaves. That is not
     measured here.
2. **What holds B's unconverged calls is a deeper leaf.** In the 91 calls route 1 converges and B does not, the
   leaves that hold B at `depth_max` are all Clenshaw–Curtis over Clenshaw–Curtis at depth 20, with r median 1.46.
   167 of 199 are leaves route 1's yardstick would not trust either. They come from refining below a leaf B withheld
   trust from.
3. **Clenshaw–Curtis parents are still trusted far above φ_P:** up to Δ1/φ_P = 862, with 64,869 firings at ≥ 5.8
   outside the 720. 110 CC + CC firings have Δ1 ≥ 5.8 × (|E_P| + |E_L + E_R|). As §4 says, φ_P is not a noise scale
   there. These leaves are bounded by ν_B, which every one of them meets.
4. **The 4 C2 losses are `node_limited` at 1e-8,** with abserr 1.0009–1.032× 06g's
   (`[06g-node-limited-stop-can-end-a-call-refinement-would-converge]`'s mechanism). Route 1 lost 7.
   - Two calls are lost under both routes: `g-1e-8-0-12591` #4 and `g-1e-8-1-35186` #6.
   - B converges the other 5 of route 1's 7.
   - Of the 6 calls B converges and route 1 does not, all `node_limited` under route 1, five are those five. The
     sixth, `g-1e-8-1-2229` #4, is unconverged under 06g too.
5. **In the 720, row `0-2316`'s `G+q+r` call #1 moves by 2,513× and 7,406× (e_B + e_g).** B converges it in 68 and
   73 regions, where 06g does not converge in 201 and 206. Route 1 agrees with B on it within both errors. It is a
   stationary-point row: 07's, not scored here.
6. **This census's unit walls carry the host's load.** On K0, today's identity run took 1.11× 06g's census walls.
   So the 0.86 unit-wall ratio overstates B's cost against 06g's and route 1's censuses, which ran on quieter hosts.
   The region ratio, 0.84 against route 1's 0.78, is free of load.

## State handed on

- Nothing is running. The lock is free, and no Ray process is listed.
- **Kept on the Mac Studio:** the census run directory, the capture, K0 and development pickles, and route B's extract,
  listed with SHA-256 and sizes in `KEPT_ON_MAC_STUDIO.txt`.
- **The board:**
  - `[06g-slow-leaves-at-half-cost-convergence-outside-stationary-points]` gains 06l's measurement of B;
  - `issues.md` proposes a narrowing and no new issue.
- Which route, if any, becomes code (B, A or neither) is for the user. The remedy prompt is 06m.
