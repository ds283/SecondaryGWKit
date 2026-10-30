# Issues found by the rehearsal agent, levin-phase-floor 06c §4 (Mac Studio)

These are proposals for the orchestrator. The board and `docs/OPEN_ISSUES.md` are not edited
here.

## Proposed

### N1. `[06c-rehearsal-totals-disagree-beyond-both-declared-errors]`: Y1 fails

**Where.** R17 (06b's code) against R18 (06c + 06d), on R8's rehearsal scope at rtol 1e-7. 30 of
the 838 `QuadSourceIntegral` rows with finite `total_abserr` in both stores have
$|T_{18}-T_{17}| > e_{17}+e_{18}$.

**The 12 large ones** (336–16,763× the summed declared errors):
- **Where:** all equilateral ($k=q=r$ at $10^5$, $3.16\times10^5$ and $10^6$; $z$ 250–6,905).
- **The cause:** each is one call, the `G+q+r` phase group of the all-oscillatory sub-interval.
  Its value moves by 0.03–1.0 % between the codes (0.64–1.04 % in eleven), while each code
  declares at most about $2\times10^{-7}$ of it.
- **The size:** at least one code misdeclares that call by up to $3.4\times10^5$.
- **The flags:** both codes call it converged in 11 of the 12.
- **The direction is mixed.** R18 uses fewer regions in ten (for example 13→2) and more in two
  (2→21, 1→66). So the comparison cannot say which code is wrong.

**The other 18** violate by 1.007–20.4×.

**Evidence:** `06c/rehearsal/out/qsi_tabulation.txt` (every row in `float.hex`),
`out/y1_violations_groups.txt`, `out/y1_violations_worst_calls.txt` and `out/qsi_rows.csv`.

**Impact.** Prompt 06c §4's Y1 fails. Under addendum B.6, the header issues close only "if 06c's
§4 holds and X0–X4 hold; otherwise ask the user". If R18 is the dishonest side, this is production
evidence for `[06c-budget-can-stop-on-a-pre-asymptotic-residual]` and/or
`[06b-moved-calls-declare-less-than-their-error]` on a call class X1's sample may not contain
(`G+q+r` on equilateral triangles at low $z$). If R17 is the dishonest side, 06b's code was wrong
there and 06c corrects it.

**Next step.** Evaluate the 12 `G+q+r` calls against an independent reference, for example §5's
W2: Levin forced to $2^9$ and $2^{10}$ equal pieces, each piece's floor reported. That decides
which code is honest. It is the user's to decide whether that belongs to 06c's records, to the
census agent's X1, or to a follow-up.

## Observed, no issue proposed

- **`QuadSourceIntegral` persists neither `num_untrusted_regions`, the trust check's per-leaf
  fields, nor a per-call `max_depth`.** Only the row's maximum depth is stored. The census records
  them per call; a store cannot show them. This is the same kind of gap as 06b's N2.
- **One R18 row reaches depth 20 and stays unconverged:** $k=10^6$, $q=3.16\times10^5$, $r=10^6$,
  $z=3971.4$, with 31→394 regions. Y1 holds there.
