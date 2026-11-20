# levin-phase-floor 07c Part C §6 (census agent): issues proposed, for the laptop and the user

Proposed only; none is opened here. Numbers from `log.md` ("Y1–Y9 scored") and `out/scores/summary.json`.

1. **The stop acts mostly on Levin leaves and on analytic calls, beyond §9.3's basis.**
   - 630,595 of the census's 977,289 stopped leaves are Levin leaves, mostly at depth 3–4, 413,499 of them untrusted.
   - 57,659 of the 82,172 calls the stop acts in are analytic: 429,832 stopped Levin leaves.
   - §9.3's basis was D1's group-call chase: a Levin leaf "rarely at the noise level" (2,460 of 15,445), "the analytic
     calls have no chase in the sample".
   - No analytic call is unconverged, and no referenced call is converged and dishonest but R2-28. But a stopped Levin
     leaf's declaration on an analytic call is scored only through 07b's R2's 21 and R3's referenced analytic calls.
   - Next step: a reference sample of analytic calls the stop acts in, stratified by the carrying stopped leaf's route,
     trust and share.
2. **A stopped untrusted leaf's 1/(1 − r) declaration can overstate by orders of magnitude.**
   - On `h-1e-7-1-14470` analytic #7 (J1, converged under both codes), an untrusted stopped Levin leaf at depth 3 with
     r = 0.99986 declares 2.46×10⁻²⁰ from δ₁ = 3.5×10⁻²⁴, carrying 0.998 of the call's abserr. The call's abserr is 242×
     07b's.
   - 24 acted analytic calls at 1e-7 have abserr above 2× 07b's.
   - This is the LiouvilleGreen draws' pattern (Part B: stopped Clenshaw–Curtis leaves' declarations, r 0.81–0.97), at
     census scale and on Levin leaves. It costs accuracy claims, not honesty. §1.1 makes a stopped leaf declare as at
     `depth_max`.
3. **Item 6's moved calls are unchanged in kind.** 328 group calls move past the sum of 07c's and 07a's abserrs (07b:
   326); none moves past the sum against 07b. They remain 07d's (with R5's verdicts).
4. **compare_07c.py's whole-unit categorisation files `route_07a_post_unit` under `call`.** 07b's regex does not name it.
   13 untouched-in-every-call units differ from 07b's only there and in `analytic_06h.parent_counterfactual`, the
   harness's own post-section re-solves. Not a census output; the harness's categories should name it.
