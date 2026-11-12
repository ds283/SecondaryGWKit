# levin-phase-floor 07a Part A — issues proposed (not opened)

Measuring agent, Mac Studio (mps026298), worktree `SecondaryGWKit-lpf07a-f93aafd` at `f93aafd0`. Each item is a
proposal for the orchestrator; none is opened here. Items 1-4 are as written at the first stop (2026-11-10). Each carries an
update from the section 2.5 stop (2026-11-11), and items 5-6 are new.

1. **A Levin leaf beside the window can understate its error (H3).** On v22, the one failure attributed so far
   (`out/attr/attr_tables_interim_1.md`), the whole excess error of the call, +2.04e-33 against its abserr 8.08e-34
   (2.53x), lies in one Levin leaf just right of the window, [9.2321, 9.7509] at depth 5: true error 2.04e-33 against
   its e' 4.37e-34 (4.7x). The window's 15 leaves together are wrong by about 1e-40. If the other four failures
   attribute the same way, the window's early stop is not what makes them dishonest, and none of section 2.3's stop
   variants can fix them. Proposed name: `[07a-levin-leaf-beside-a-window-understates-its-error]`.
2. **Window leaves are bisected because they are untrusted, not because the budget asks.** In the 34 calls of section
   2.1 items 1-2 and the no-window check, all 382 bisections of a window leaf have an untrusted parent (none trusted,
   none with a widened declaration), and 299 of them happen with the call's budget already met
   (`out/replays/a/*.instrument.json`). dqage's iroff1 counts bisections that a budget forced; here the trust check
   forces them. Departures 1-3 of the prompt's section 0.1 never act on these calls (no unresolved half, e' equal to the
   raw estimate, value changes 1e-9-1e-12 relative), so variants (c) and (d) count exactly as 07 on all 34. Recorded
   for the orchestrator's reading of section 2.4; it bears on what an `iroff1` port can do here.
3. **With no stop, window leaves are chased to depth_max, and the call stays unconverged.** Variant (b) on v08 and v12:
   regions 86, 200, 472, 1,125, 2,718, 6,691 at depth_max 6-16 (about 2.4-2.7x per two levels), every one unconverged
   with window leaves at depth_max, and abserr not falling (v08 1.94e-36 at depth_max 10-16, as at 07). 07's earlier
   attempts found the same on a different driver (`docs/levin-phase-floor/07/earlier-attempts/log-parts-a-1-to-4.md`).
   A sequel of `[07-iroff1-window-stop-can-leave-a-call-dishonest]`, not a new defect: recorded so the cost of "no stop"
   is on file.
4. **The 100x rule for a leaf reference cannot be met where a leaf's declared error is below the integrand's own
   noise.** The Filon readings of a leaf agree between resolutions to about 1e-15 of the local magnitude (the captured
   double-precision callables); 80 of v22's 124 leaves, 32 of v21's 62 and 83 of v03's 161 declare e' below 100x that.
   No method that evaluates the captured integrand can do better. The parts' summed accuracies are 1e-5-1e-7 of each
   call's abserr, so the attribution is resolved at the part level. A note for the prompt's rule, not a code defect.

## Updates at the section 2.5 stop (2026-11-11)

- **Item 1, now on all five failures** (`out/attr/attr_tables_final.md`, `out/o9/o9_tables.md`). In every failure, H3
  holds the whole excess: v08 3.16, v09 2.21, v12 3.08, v17 2.38, v22 2.53 of abserr. The windows' parts are 1e-8 to
  1e-7 of abserr, and the call's excess sits almost entirely in one H3 leaf (share 0.999 to 1.008):
  - in v08, v12 and v17 that leaf is the **depth-0 seed region** from the window's upper edge to the end of the span,
    about [9.23, 27-28]. It was solved once, as a Levin leaf, and never bisected; its true error / e' is 3.63, 7.44 and
    2.94;
  - in v09 it is a depth-1 leaf, [12.84, 24.69], at 2.66;
  - in v22 it is a depth-5 leaf, [9.23, 9.75], at 4.67.

  The O9 check finds no route change on these leaves, their parents or their siblings, and no strict sign change of
  theta'. The true variation is 0.47-1.06x 07's estimate on the main leaves (0.47-1.39x over every region checked), far above 20 pi on all of them. So the gate did not
  misroute them on phase grounds.
- **Item 3:** variant (g) is no different. (g2) equals (b2) bit for bit on all 34 calls, and both leave every one of
  the 29 window calls unconverged: 764,283 regions against (a)'s 2,386.
5. **(new) The tolerance cross-check's ratio, scored by r_g, fails at 07's code on both pairs.** p14501 has
   r_g 1.39 and p35006 1.08, both members converged (`out/tables_final_rg.md`). This is the known cross-check finding,
   now confirmed by replays. Variants (e) and (f) pass only by leaving a member unconverged. Proposed for the
   orchestrator to merge with the issue that already holds the cross-check, not to open separately.
6. **(new) The H2 part's ratio exceeds 1 on two controls**, v02 (|true| / declared 1.31) and v10 (1.04). In both,
   the part is about 1e-7 of its call's abserr, and both calls are honest. Recorded so it is on file; it changes no
   verdict.
