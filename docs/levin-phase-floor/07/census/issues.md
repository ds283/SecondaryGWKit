# levin-phase-floor 07 §6, the census agent: issues proposed

These are proposals for the laptop and the user to decide. None is adjudicated here. The numbers are in `log.md` and
`out/`.

1. **Windows stopped by §1.3's `iroff1` rule while leaves were still falling can leave a converged call outside its
   declared error.** (Proposed name: `[07-iroff1-window-stop-can-leave-a-call-dishonest]`.)
   - V4 fails on five of its 24 calls: v08, v09, v12, v17 and v22, at 2.2–3.2× abserr_g.
   - Each is a converged `G+q+r` call. Its window was stopped at `iroff1` 6 while 4–13 of its leaves were still falling;
     the windows' `max_depth_at_stop` was 5–6.
   - Every call's readings agree to 0.004 of the tolerance or better.
   - 06m was 497–30,400× its abserr on the same calls.
   - This is O8 and the note's B.5 met on real calls. Over the census, 1,153 (1e-7) and 1,192 (1e-8) windows stopped with
     a leaf still falling.
2. **V5: located calls still reach `depth_max`.** 23 `G+q+r` calls locate their root and hold 2–54 leaves at depth 20; 20
   of them are unconverged. This reopens G2's L2 (Decisions, **G2 L2, decided**), as the prompt says a V5 failure does.
3. **O9's calls in the census** (for `[07-c2-gate-threshold-was-never-measured]`).
   - 28 analytic calls (J4 and Y4 of 7 rows, at both rtols) have a phase Rule A skipped (V 15.6–18.6). The C2 gate's node
     estimate sent a region with a root to the Levin solve, and the nodal guard caught it, one region each. All 28 are
     converged.
   - A further 154 analytic calls at 1e-7 with a skipped phase show a scan sign change that the gate sent to
     Clenshaw–Curtis, unguarded.
   - This is a proposed addition to the existing issue, not a new one.
4. **The cross-check's undecided calls** (interim; for the user's sample of references). 786 `G+q+r` calls (396 rows)
   where both codes are self-consistent across rtols and differ beyond both errors. All five of V4's failures are among
   them. References for all of them project at 4.5e6 s FCC-only; 145 rows project at 30 minutes or less.
5. **Observation, not proposed as an issue:** lone-`G` group calls locate 6 or 19 minima of |θ′| with λ about 1e-7–1e-6
   (236 calls at each rtol). All are converged; nothing measured is wrong.
