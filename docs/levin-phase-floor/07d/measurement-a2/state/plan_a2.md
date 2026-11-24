# levin-phase-floor 07d Part A': the stages, fixed before anything long runs (2026-11-23)

1. **Gate** (out/cmd/01_gate.txt): holds.
2. **v4's rule** (state/refs_rule_leaf_v4.txt) and the seeds (state/seeds_a2.json), written before any reading or draw.
3. **The projection** (out/projection/), reported to the orchestrator; nothing long runs before its go-ahead.
4. **The instrumented variant** `scripts/variants/levin_quadrature_a2_I6.py`: the tree plus pass-through records of every
   leaf made (span, depth, route, phase_span, window, guarded) and every `_check_leaf` (span, depth, route, forced,
   delta1, delta2, r, nu_B, trusted, rate-rule refusal). Diff saved. Run on all 4,220 of Part A's calls at 6 pi as a
   registered run; every call must equal Part A's tree replay (value, abserr, converged, regions, depth, the chase counts),
   else stop.
5. **The reproduction of `_check_leaf`** (`scripts/checkrepro_a2.py`) compared bit for bit with (4) on at least 200 checked
   Levin leaves, group and analytic, drawn by a fixed rule; any difference is a stop (prompt section 9.3).
6. **Item 3's population**: the leaves (4) records as made, Levin route, V in [20 pi, 32 pi), on Part A's `d:` calls,
   permuted by BANDS2 and saved before any is solved.
7. **The leaf run** (`scripts/leaf_a2.py`, registered): every leaf of Part A's 44 strata and the new band's, in seeded order,
   filled as Part A's (40 scored or exhausted): v4; sigma_hat; CC_47, CC_24, CC_93, CC_185 and their ingredients; item 1's
   checks at T = 4, 6, 8, 10, 12, 16, 20 pi under the CC_47 and CC_93 fallbacks; item 4's knots by kind and splits (group
   leaves); item 5's costs.
8. **Tables 1-6**; CC_93's candidates fixed (kappa from the 93-sample plateau against sigma_hat) before scoring.
9. **The variants** (at most eight, registered run), chosen by items 1, 2 and 4 as section 9.2 item 6 says; then new call
   references only where an (Rt) stratum is empty, projected before reading.
10. **Table 7**, the records, KEPT_ON_MAC_STUDIO.txt last; the stop.
