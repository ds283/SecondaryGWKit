# levin-phase-floor 07b §6 — the census agent's issues (proposed, for the laptop and the user to decide)

Nothing here was acted on. Each item is an observation, with the place its evidence is kept.

1. **The chase in production** (`[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]`, opened at §9.6; the
   evidence is in `out/scores/summary.json` `newly_unconverged`).
   - 198 group calls go from converged at 07a to unconverged at 07b: 97 at 1e-7, 101 at 1e-8.
   - In 192 of them every leaf at depth 20 (668 leaves) is a Clenshaw–Curtis leaf below a leaf the rate rules refused.
   - 139 rows lose `total_converged`, against 4 that gain it. Rows at depth 20 rise from 77 / 83 to 182 / 185.
   - This is the issue's next-step population at census scale.
2. **605 group calls moved beyond both codes' abserrs** (`out/scores/summary.json` `calls_moved_beyond_abserr_07a`).
   - 603 of them are converged under both codes, so in each one of the two declarations does not bound its error.
   - Examples: `h-1e-7-0-12710` `G-q-r` moved 63.6× 07a's abserr (1 region at 07a, 52 at 07b); `h-1e-7-0-14297` `G-q-r`
     moved 25.6×.
   - None of them is referenced. A reference sample drawn from them would say which code is right and how often.
3. **abserr grew by more than 100× in 738 acted calls**, up to 4.6×10⁵ (e.g. `h-1e-7-1-10964` analytic J4).
   - They are analytic calls of one region at 07a, two at 07b.
   - Whether 07a's one-region declarations understated is not measured.
4. **The edge rule's reach** (§9.3):
   - it refused only on `G+q+r` root windows (868 refusals, 674 calls, 636 at depth 0);
   - it refused none of 2,802 evaluations on analytic windows;
   - it never acted on a minima window (07a's 2,360 newly located minima-window calls are all untouched).
   - What §9.3 says its basis does not reach, the census confirms it does not act on.
5. **Cost.** The census wall is 1.93× 07a's: 45,016 s against 23,284 s. Regions are 2.08×. Unit walls are 2.09× and 1.92×.
   §9.4's estimate of about 5× was pessimistic. This census becomes prompt 08's "before".
6. **The analytic Levin calls at 1e-8.** The census records 107,416 analytic calls at each rtol; §9.3 lists them at 1e-7
   only. The range rule acts on 87% of them at both rtols.
7. **Six newly unconverged calls with no depth-20 leaf**:
   - `h-1e-8-0-33997` #4, `h-1e-8-1-2229` #4, `h-1e-8-3-13615` #2/#3, `h-1e-8-3-13490` #2/#3;
   - no untrusted leaf, abserr 1.001–1.33× 07a's: the budget not met.
8. **A records note:** 07b's census and k0 run directories are kept on the Mac Studio only (`KEPT_ON_MAC_STUDIO.txt`), the
   census as prompt 08's "before".
