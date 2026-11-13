# levin-phase-floor 07a §6 — the census agent's issues (proposed, for the laptop and the user to decide)

Nothing here was acted on. Each item is an observation, with the place its evidence is kept.

1. **06g's census records are no longer on this host** (`var/levin-phase-floor-06g/` is absent).
   - 07's harness named 06g's `rows.json` (`aa523d28…`) as `refs_common.ROWS`. 07a's harness reads 07's copy of it, which
     has the same SHA-256 (`out/cmd/07` failed on the old path; `out/cmd/07b`).
   - **06m's census checkpoint was trimmed on 2026-11-10** (`TRIMMED_2026-11-10.txt` in its run directory). The disagreement
     list against 06m therefore takes 06m's values from 07's cross-check output.
   - Any later prompt that reads either census on the Mac Studio will meet the same absence. This is a records-retention
     note, not a code defect.
2. **07's worktrees `SecondaryGWKit-lpf07-*` are removed.** 07a's K0 run executes the parent's module from
   `state/variants/levin_quadrature_parent.py` in tree h, as the dispatch directs. `out/w0/` shows it is the parent's code.
3. **W7 fails on 25 lone-`G` minima-window calls at 1e-8** (`out/scores/w7_detail.json`).
   - Under 07a's value test, min(1e-5, rtol), a minima window that stopped at 07 (iroff1 6) no longer stops (iroff1 4–5).
   - The value moves by 0.0066–0.145 of the call's abserr. Abserr and `converged` are unchanged.
   - §9.5's threshold, 1e-5 of abserr, rested on Part A's 430 calls, all `G+q+r`; no minima window was among them.
   - This bears on `[07-iroff1-window-stop-can-leave-a-call-dishonest]`, whose closing condition is "W7 and §5 hold".
4. **W5's node-partition count on forced solves** (`out/scores/w5_forced.json`; W5 is scored on gated solves by the user's
   ruling, `state/ruling_w5.md`).
   - In 4,416 of the 4,804 root-located calls, the window piece whose end node is the root has θ′ changing sign on its nodes,
     so its V takes the node-partition sum. So do its forced halves and quarters: 18,236 solves in all, 3–10 per call.
   - The V of a forced solve routes nothing, but it is the `phase_span` reported for that region.
5. **Shrinking declarations on calls whose only route changes are auxiliary** (`out/scores/summary.json` `aux_only`;
   `out/scores/moves_beyond.json`).
   - 47 aux_only calls' abserr shrank by more than 2× at h.
   - At `3-678` (`G+q-r` and `G-q+r`, both rtols) it shrank to 0.090 of 07's, with regions going 12 → 1, while the value
     moved by 2.0× the new abserr.
   - A trust-check quarter or comparison half re-routed to Clenshaw–Curtis or to least squares can change the trust test's
     ratio and the trusted value Q.
   - **The user's reference for `3-678`** (`out/refs/report.md`): h's declaration is honest, at 0.491 of its abserr. 07
     was at 0.135 of its own.
   - The other 46 such calls are unreferenced.
6. **07 and h disagree beyond both errors at `1-34599`** (`G+q-r`, `G-q+r`, 1e-7 and 1e-8; class own, one own C2 route
   change).
   - **The user's reference** (`out/refs/report.md`): h is honest, at 0.020 (1e-7) and 0.015 (1e-8) of its abserr. **07 was
     not**, at 4.03× its abserr.
   - The own C2 route change (a region V < 6π ≤ estimate, now to Clenshaw–Curtis; 6 → 16/17 regions) corrected a dishonest
     07 call.
   - It was a converged, dishonest call at 07, with no window, that no earlier check had caught. Whether other own-class calls
     are like it is unmeasured: 5,314 own calls moved.
7. **`[07-c2-gate-threshold-was-never-measured]`, for that issue:**
   - Every changed C2 decision in the census went to Clenshaw–Curtis: 10,014 solves, estimate/V 1.00001–2.29.
   - Every changed LU decision went to least squares: 14,545 solves.
   - No call lost convergence for it (§10.4's set is empty in the census).
   - The smallest unchanged margin to 6π is 4.8e-8 rad, a trust-check quarter of `*-1-14543`.
