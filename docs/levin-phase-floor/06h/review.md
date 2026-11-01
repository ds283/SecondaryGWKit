**Verdict: Part B is complete and its records stand. No §7 stop fired. Gate H holds. H0, H1, H2 and H3 hold; H4 fails at rtol 1e-8 on one call (`g-1e-8-3-1072` #13, analytic J1), which the note makes a finding for the user, not a stop. All ten review checks pass; the slips found are cosmetic and recorded.**

# levin-phase-floor 06h, Part B — the orchestrator's review (Mac Studio)

Written 2026-11-01 by the orchestrator on `mps026298.phys.susx.ac.uk`. Prompt
`prompts/levin-phase-floor/06h-group-phase-rounding.md`, the note `orchestrator/prompt-06h.md` §B,
whose machinery is 06e's note §B and 07's note §B.2–§B.3. Every command I ran myself, with its
output, is under `review/checks/`; my readers are under `review/scripts/`; the briefs are under
`review/` with their SHA-256 in `review/dispatch_sha256.txt`. I wrote no code in the trees and ran
no `git` command that writes.

## The trees and the runs

| | Tree | Run | Result |
|---|---|---|---|
| R21 | parent `2c325f5` (06e's code) | `levin-phase-floor-06h-r21-parent-20261101T001852` | done, exit 0, `1b332ffa…` |
| R22 | `c1ec3b2` (code `7172a8f` + tests-only `ac1c6f4`) | `levin-phase-floor-06h-r22-groupphase-20261101T002423` | done, exit 0, `1b332ffa…` |
| Census | h = `c1ec3b2` | `levin-phase-floor-06h-census-20261101T005037` | done, exit 0, 18,560/18,560 units, 2 h 33 min wall |
| Phase | p = `2c325f5` and h = `c1ec3b2` | `levin-phase-floor-06h-phase-20261101T033514` | done, exit 0, 84/84 units |

Worktrees: `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06h-2c325f5` and
`/Users/ds283/Documents/Code/SecondaryGWKit-lpf06h-c1ec3b2`, both clean including ignored files at
the end. The main checkout stayed at `28ebaf0`, clean.

**The gate test on this host** (`preflight/test_group_phase_rounding.txt`):
`LiouvilleGreen/tests/test_group_phase_rounding.py`, 8 tests OK. **ε_exp on this host:** test 5
measured the worst relative error of `np.exp` as 1.114619e-16 = 1.0040 u, identical to the
laptop's; `EXP_RELERR` = 2.23e-16 is twice that, rounded up. Scalar and array paths bit-identical.
Python 3.13.16, numpy 2.2.4.

## The hypotheses

| Item | Verdict | Numbers (my re-score agrees in each case) |
|---|---|---|
| **H0** (§4.3, on `total`) | **holds**, trivially | 864 rows scored, 0 violate, 0 non-finite, 0 stationary-point rows in the store; `total` and `total_abserr` bit-identical in 864/864. The same form on the analytic oracle (`analytic_rad` against `analytic.abserr`): 0 violate, moved in 464 rows, worst ratio 2.485e-5 |
| **Gate H** | **holds** | P1 `low-G+r-1` (`2-38302`) and P2 `ctrl-low-1` (`2-2095`) at 1e-7, P3 `0-1971` at 1e-8: at the parent equal to 06e's census on `total`, `total_abserr`, `numeric_quad`, `WKB_Levin`, `analytic_rad` and every call's value, abserr, regions, depth and convergence (23, 23 and 21 calls); git clause: only `CLAUDE.md` changes `6d81fad`..`2c325f5`; `g-1e-7-0-14164` deterministic at h |
| **H1** | **holds** | 21,177 points (1,177 nodes of every returned leaf and 20,000 dense) on 06f's 20 M2 calls; **0 violate**; worst \|δθ\|/theta_abserr = **0.4933** at `g-1e-7-0-12689` #19 (J4), CC leaf 1 node 11, u = `-0x1.e633fe4760d6fp+2`, δθ = 4.5027e-13 against 9.128e-13 (constituents' term 9.009e-13, abscissa 5.46e-15, φ_eval 6.46e-15). Nodes alone: 0 of 1,177, same worst. The parent on the same points: 561 violate (53 nodes), worst **376.5** (#19 dense point 996) |
| **H2** | **holds** | 27 returned leaves (23 CC, 4 Levin), **0 violate**; worst \|δI\|/declared = **0.268** at #19 Levin leaf 3, median 4.5e-5. The parent: 2 violate, worst 41.6 on the same leaf |
| **H3** as written (`total`) | **holds** | 8,560 rows scored at each rtol outside 06a's 720; 0 violate; every `total` bit-identical to 06e's (06h cannot reach `total`: `BesselPhaseGroup` reaches only the analytic oracle) |
| **H3**, second reading (the oracle) | **holds** | 8,560 rows per rtol, 0 violate; worst ratio 0.0756 at 1e-7 (`2-38300`), 1.35e-4 at 1e-8 (`1-20510`); 06e's oracle recomputed in each h unit equals 06e's recorded `analytic_rad` bit for bit on all 8,560 rows per rtol. The two readings agree, so no stop |
| **H4** | **holds at 1e-7, fails at 1e-8** | Per rtol 95,896 analytic-block calls and 33,725 group calls matched outside the 720; none gained convergence. One lost: `g-1e-8-3-1072` #13 "analytic J1" — 06e 2 regions, depth 1, abserr 7.16e-27, converged; h 1 region, depth 0, abserr 2.36e-26 against a request of 1.63e-26, unconverged. The row's `total` and oracle value are bit-identical to 06e's |

**H4's mechanism** (the census agent's reading, inferred from the premise counters and the budget
record; the halves' floors at h are not recorded): at h the root's floor rose from 1.94e-26 to
2.36e-26 with the declared phase term (4.14e-27, mostly the abscissa), so the round-off exit's
premise held at the root and it was not split; under 06e the premise failed and the bisected
halves converged. So it fails through the round-off exit's premise, not through the declared
term's size against the request. Per the note's header, "H0, H3 or H4 failing is a finding for the
user, not by itself a reason to keep [`06f-theta-abserr-omits-the-product-rounding`] open."

**The closing condition** for `[06f-theta-abserr-omits-the-product-rounding]` — prompt §2's tests
and §5's H1 and H2 hold — is met on this host's evidence. Closing it is the records commit's.

## The descriptive items

- **Cost.** Unit wall (`wall_s`, the timed section): 2.223 h at 1e-7 against 06e's 2.282 h
  (×0.974), 2.919 h at 1e-8 against 2.974 h (×0.981). The run's wall, 2 h 33 min against 06e's
  1 h 47 min, is the post-timed instrumentation (chiefly the recomputed 06e oracle), which runs
  after the deadline is cleared and is outside both the budget and `wall_s`.
- **Regions and solves** (all calls): 1e-7 regions 625,878 → 576,425, solves 4.862 M → 4.467 M;
  1e-8 regions 799,186 → 738,153, solves 6.210 M → 5.722 M. All of the change is in the analytic
  blocks (×0.876 and ×0.870 in regions); group calls' depths, regions and solves are identical to
  06e's.
- **Convergence and depth.** Converged calls 144,732 = 144,732 at 1e-7; 144,726 → 144,725 at 1e-8
  (H4's call). Group calls unconverged 9 and 15, rows with `total_converged` false 9 and 15, calls
  at depth 20 11 and 18 — all as 06e. Analytic calls got shallower: at 1e-7 depth 3 holds 25,353
  against 9,101 and depth 4 15,520 against 30,546.
- **The 720 stationary-point rows.** No `total` moved at either rtol; the oracle moved in 462 and
  421 rows, all within the second reading's bound; no call changed convergence.
- **The declared term** reaches at most 0.341 of a call's request (median 5.3e-6 at 1e-7, 6.3e-5 at
  1e-8). On the Clenshaw–Curtis branch the declared-dominated calls roughly doubled (2,493 → 5,137
  and 2,785 → 5,939). The Levin branch has no 06e column: 06e's checkpoint does not separate it, so
  h's constituents part there is a counterfactual (the census agent's open point, not adjudicated by
  me).
- **The near-singular class** (the census agent's rule, fixed before the run): 435 rows, all b = 0,
  all (+,+,−) with q = r ≫ k; nothing moved there (ratio at most 5e-14 of its bound).
- **The phase re-measurement**, split by dominant term at h: constituents 20,446 points (median
  6.8e-4, max 0.493), abscissa 731 (median 0.134, max 0.459), φ_eval 0. `K_lo` non-zero on 9 of 20
  calls, at most 4.66e-9.

## The ten checks

1. **Scope — passes.** `var/` against the pre-dispatch listing (67,284 files) by path: 0 removed,
   4,350 added, 1 changed, every one under `var/levin-phase-floor-06h/`,
   `var/runs/levin-phase-floor-06h-*` or `var/datastores/levin-phase-floor-06h-*`
   (`checks/var_scope.txt`). 07's work (`var/levin-phase-floor-07*`) untouched. A3 v2's listing
   equal; the scratch copy equal to its certification by SHA-256 and listing; 06e's checkpoint
   `a0dc4d7f…` unchanged; HEADs as dispatched, all clean (`checks/scope.txt`).
2. **The runs — pass.** Registry 40 runs, none running or stale; the four 06h runs `done`, exit 0,
   host this one; R21's and R22's manifests carry their tree's `git_head`, clean; the census's and
   phase's carry the main checkout's `28ebaf0` (their cwd), as their purposes state — their records
   carry the trees (`checks/manifests.txt`, `checks/registry_final.txt`).
3. **The fingerprints — pass.** `runregistrykit store fingerprint`, no `--write`, from the worktree
   `c1ec3b2`: R21 and R22 both `1b332ffa…`, problems none. (From the main checkout it refuses —
   that code does not declare `theta_residual` — and reads nothing; `checks/fingerprint_note.txt`.)
   Store hashes unchanged by my reads.
4. **The values — pass.** My `tools.store_value_diff --json` is byte-identical to the agent's
   (`68b9c378…`); 05's classifier (`26690e61…`) on it: outside `QuadSourceIntegral` the seven value
   tables identical (360,462 rows), the rest timing, labels, `parents_digest` and `rho_reuse` only,
   0 unclassified. `orch_h0.py` reproduces H0 and the oracle form (`checks/orch_h0.txt`).
5. **The readers and the lock — pass.** Readers 6/6 exit 0, R22 byte-identical before and after.
   My own `status` and `ray-processes`, each a command of its own, after the rehearsals, after the
   census and at the end: lock free, no Ray process of `/tmp/ray-sgk` (`checks/lock_*.txt`).
6. **Gate H — passes**, checked by my own reader against 06e's checkpoint (`checks/gate_h_mine.txt`).
7. **The trees — pass.** All 18,560 census records: tree `c1ec3b2`, status clean, 102 own-tree
   modules, 0 foreign, 0 modules from another SecondaryGWKit tree, one script hash `e11dcd72…`
   (`checks/census_provenance.txt`). Phase captures: workers report the right tree, Ray not
   initialised.
8. **The predictions — pass.** Census `state/predictions.md` 00:49:57Z before its manifest
   00:50:37Z; phase `state/predictions_20261101T033443Z.md` 03:34:43Z before 03:35:14Z. H0 is fixed
   by the prompt at `664def1`.
9. **The verdicts — pass.** `orch_census.py` on the two checkpoints reproduces H3 (both readings),
   H4 (the one lost call), and every descriptive count above (`checks/orch_census.txt`). H1 and H2
   recounted over every point and leaf of the phase tables (`checks/orch_h1h2_recount.txt`).
   `orch_dtheta.py`, mine, loads the captured group from its pickle and recomputes δθ in mpmath at
   the exact e^u: at h's worst node 4.5026800156217862e-13, ratio 0.4933, at 30 and 50 digits; at
   the parent's worst point the parent gives 376.47 and h 0.387 (`checks/orch_dtheta_{h,p}.txt`).
10. **The records — pass.** Each agent wrote `log.md`, `verification.md`, `issues.md` and
    `KEPT_ON_MAC_STUDIO.txt`, with every deviation classified (rehearsal 10, census 23, phase 22).

## Slips, and what they touched

All `UNINTENDED DRIFT` and harmless: a `runregistrykit list | head` BrokenPipeError and a wrong
redirect path (rehearsal); a `cp` to `/dev/null`, a `sed` pipe on a lock listing, a BSD `stat` on
this host's GNU `stat` (census); a `tail` pipe on `ray-processes`, a refused redirect, and the phase
driver leaving its stage string at "refcheck running" after the last unit (phase). Mine: one
refused redirect in `checks/scope.txt` that wrote nothing. The rehearsal launch logs sit at
`var/runs/levin-phase-floor-06h-r2{1,2}.launch.log`, beside the run directories rather than in them.

## For the user

- **The rehearsal agent's question** (its deviation 2): "The readers start Ray. Each reader calls
  `ray.init(address="local", _temp_dir=/tmp/ray-sgk)` … The conflict: the common brief says only
  R21's and R22's drivers may start a Ray session. Prompt §4.4 and this agent's brief both require
  the readers, with the lock held through them. What was done: they ran under the lock, as 06e's,
  06c's and 06b's did … Recorded so that the orchestrator can confirm the reading." The conflict is
  my common brief's imprecision.
- **The census agent's question:** "H4 fails on one call through the round-off exit's premise, not
  through the size of the declared term. Should 06h's records commit treat H4 as failing, so it is a
  finding for the user as the prompt says, and which board entry should carry D1?" Its candidates:
  amend `[07-roundoff-exit-assumes-bisection-cannot-help]` or the 06b premise-check family, or open
  `[06h-declaration-can-trip-the-roundoff-exit-premise]`.
- **Candidate issues:** H0 and H3 as written score `total`, which 06h cannot reach (rehearsal);
  D3, the declared term dominating twice as many CC calls (census: amend
  `[06e-cc-declared-phase-term-is-a-crude-proxy]`); the BrokenPipeError of `runregistrykit list |
  head`.
- **Kept on this host** until the user says otherwise: R21's and R22's stores
  (`var/datastores/levin-phase-floor-06h-r2{1,2}*.sqlite`); `rehearsal/out/store_value_diff.json`
  (41 MB) and `rehearsal/extract/` (172 MB); the census run directory in full (2.3 GB, checkpoint
  `3104c18e…`, 16 region pickles — 06g's before); the census's Gate H and smoke pickles and
  `state/rows.json`; the phase's 80 capture pickles. Each agent's `KEPT_ON_MAC_STUDIO.txt` lists
  them with SHA-256; everything else in the three agent directories is for the records commit.
