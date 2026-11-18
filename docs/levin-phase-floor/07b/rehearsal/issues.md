# Issues and observations of the 07b rehearsal agent (R31), for the orchestrator

Written 2026-11-16, at the end of Part C §5. Nothing here was acted on.

1. **R30 had converged and dishonest group calls in the rehearsal store, which (h) corrects.**
   - In all 7 group records whose \|v31 − v30\| exceeds e30 + e31, R30's converged call is 2.91 to 59.3 of its abserr
     from the reference. R31's call is 0.0014 to 0.298 of its own (`out/refs/understates.txt`).
   - In each, R31's call had 1–3 range-rule refusals and no edge refusal. R30's carried 1–8 regions, 58 in one case;
     R31's carried 38–112.
   - All 7 are G−q±r at rtol 1e-7, with no located point. This is the "away from windows" mechanism of §9.2, in a store
     no Part A reference covered.
   - It bears on `[06b-moved-calls-declare-less-than-their-error]` and on 07a's honesty generally. These are seven more
     cases of a converged Levin call understating, found by references, not by the census's own checks.
   - The 144 group records with \|Δv\| > e30, beyond these 7, were not referenced. By the same argument, they are
     candidates for R30 understating.
2. **The one record (h) makes unconverged** is G+q+r at (1e5, 1e6, 1e6), z 144627.7.
   - R30: 2 regions, converged. R31: 143 regions to depth 20, 4 range refusals, unconverged.
   - \|Δv\| is 0.62 of e30 and 0.90 of e31.
   - It looks like `[07b-rate-range-rule-chases-noise-leaves-beyond-the-window-stop]` (R2-28's pattern). It was not
     referenced (the user asked about the seven), and its capture is kept (`state/inputs/ref-G+q+r-h-3-20363-1.pkl`), ready
     for that issue's measurement.
3. **`phase_limited` flips False → True** on 5,336 analytic-block records and 348 group records, never the other way.
   Read as the range rule refining the coarse leaves to the phase floor, §10.1's finding. Descriptive only.
4. **The edge rule refused nothing** in any of the 4,295 replayed calls (located calls included). The rehearsal's scope
   (three k, zend 250) has no window-edge carrier. The edge rule's reach is untested by R31; the census measures it.
5. **Cost:** `QuadSourceIntegral`'s stage took 1.52× R30's, and WKB regions 1.33×. Analytic-block regions in the
   differing records went 25,707 → 62,854.
6. **`out/commands.log` is 11 MB** because Part A's `run_06j.py` appends every command's output to `out/commands.log`
   (deviation 17). The orchestrator may prefer to copy the agent's own lines only.
7. **The readers' outputs are not comparable file for file across rehearsals** (random k and z subsamples), as 07a's
   item 4 recorded.
