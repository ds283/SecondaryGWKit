# Issues and observations of the 07a rehearsal agent (R30), for the orchestrator

Written 2026-11-12, at the end of Part C §5. Nothing here was acted on. (The stop-1 draft's item 1 is superseded by
item 1 below, which the replay settles.)

1. **49 Levin group records whose abserr moved while value, regions and location did not.** Seen in R30 against R29,
   in records with no location in either store (`out/qsi_summary_r29_r30.txt`). The relative change runs from
   −1.9e-1 to +1.8, and is mostly about 1e-7.
   - **The orchestrator's reading, settled from the replay** (`out/trust_reading.txt`, from `route_07a` in
     `out/replay_calls.checks.json`):
     - Every one of the 49 reproduces R30 bit for bit.
     - In each one, **exactly one** region solve has a C2 route or LU choice that differs between V and 07's estimate.
       07's estimate is recomputed with the parent's expression (e182bc7a) on the same `theta_prime_Cheb`.
     - **That solve is never a region's own solve** (`Levin subregion`). It is always an auxiliary solve of the trust
       check:

       | Auxiliary solve | What differs | Records |
       |---|---|---|
       | trust-check quarter | route | 19 |
       | trust-check quarter | LU choice | 22 |
       | comparison half (`Levin left/right-comparison region`) | LU choice | 7 |
       | comparison half | route | 1 |

     - In 29 of the 49, the changed solve's span is also the span of a returned region. That region's own solve is
       unchanged.
   - **So the reading holds in substance, refined:**
     - The declared error moved because a solve that feeds the trust check changed its route or LU choice: a quarter,
       or a comparison half used as `data_L`/`data_R`. Each returned region's own solve, and so the value, is
       unchanged.
     - It is not, as the reading put it, a visited and bisected parent's *own* solve.
     - That the matching solve at R29 differed only in that route is inferred from the recomputed estimate, not
       measured. R29's tree no longer exists.
     - None of this says whether the moved abserr leaves any call honest or dishonest. S1–S4 and the census decide
       that.
   - **For §6 (census) W1.** §0.3 says that a call with no window and no *region* whose route or LU choice changes is
     bit-identical to 07's, with `phase_span` set aside.
     - Here, calls in which every region's own solve is unchanged still move in abserr, through an auxiliary solve.
     - §6's diagnostics count "the regions visited whose route or LU choice would differ". If those exclude the trust
       check's quarters and comparison halves, W1 will see calls move that it predicts are bit-identical.
     - So the diagnostics should count those solves, or W1's rule should say how they are classed. This is for the
       orchestrator; not acted on here.
2. **Analytic block records do not store their values.** They hold only abserr, regions, converged, phase_limited and
   the four counts. Inside QSI, a block's value change shows only through `analytic_rad`, so S4 needed a replay of the
   four blocks, as R29's did. This is not a defect; it is recorded so the next rehearsal plans for it.
3. **`route_07a` over every replayed call** (4,614 calls in 223 rows; `out/replay_check.txt`) classes 427,437 region
   solves:
   - 312 have a C2 route that differs between V and 07's estimate, and 430 an LU choice that differs. Most of these are
     trust-check quarters and comparison halves.
   - Of the regions' own solves (`Levin subregion`), 15 differ in route (none returned) and 43 in LU choice (36 of them
     returned).
   - This is descriptive only; the census measures it.
4. **The readers' output is not comparable file for file across rehearsals.**
   - `extract_Gk_data.py` (line 416) and `extract_GkSource_data.py` (line 861) draw random k and z subsamples.
   - So R30's `extract/` holds 360 and 162 files for those two readers, against R29's 387 and 170, and the sizes differ.
   - The readers' rule (exit 0, store byte-identical) does not depend on this.
5. **Not this agent's** (from the dispatch's list, met in passing): the stale comments in the `bessel_phase()`
   docstring, in `three_bessel_integrals.py`'s `residual_log_deriv` docstring, and at
   `ComputeTargets/tests/test_quadsource_integral.py:2121–2122`. Not touched.
