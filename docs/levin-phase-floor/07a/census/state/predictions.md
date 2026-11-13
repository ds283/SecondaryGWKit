# levin-phase-floor 07a §6 — the census's predictions, fixed before the census

Provenance (the writer's header):

```
levin-phase-floor 07a section 6: the census's predictions, fixed before the census
written 2026-11-12T13:16:44Z on mps026298.phys.susx.ac.uk
script write_predictions_07a.py sha256 61d989f391cd8310ecc20be790d55580845d81c1a2386cae00310c57901fa911
tree h: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3 HEAD 4bf50d3426dc4c9ba190760e6d30e5b7d6dd0b03 (status --short: empty)
main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
input /Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3/prompts/levin-phase-floor/07a-window-stop-and-phase-variation.md sha256 854dd5fc7d131c1f512c8ae4ef25f55641c09e14c6e385fc48eb56d89360de3b
input /Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3/docs/levin-phase-floor/07/census/out/v4/v4_scores.json sha256 2d0268e607e0cb17b4d4247831a7b933c2df0db61c248b8959ecdf9f091649b8
```

Written after Gate W0 (all four clauses: out/w0/, out/k0/compare_w0b.json, out/w0c3/compare_instr.json, out/w0c4/compare_det.json) and before the census is launched. Not to be rewritten: a correction is a new file and a deviation.

Source: `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3/prompts/levin-phase-floor/07a-window-stop-and-phase-variation.md` (file SHA-256 854dd5fc7d131c1f512c8ae4ef25f55641c09e14c6e385fc48eb56d89360de3b). Part 1 is lines 525-556 (§6, the predictions and the descriptive items); part 2 is lines 716-731 (§9.5, the additions: W4, W6 on r_g, W7); part 3 is lines 823-825 (§10.4's descriptive item). Each is copied unchanged, word for word, by line range. Part 4 (descriptive additions) and part 5 (how each rule is read, with the two questions left open for the user) are this agent's. Part 6 fixes V4's 33 calls.

## 1. Prompt §6, the predictions and the descriptive items (verbatim)

**Predictions, fixed before the run** (README §5 rule 3), copied into `state/predictions.md`
unchanged, with §9's additions:
- **W1 — nothing else moves.** Every call with no window at 07 or at $h$, and no region whose route
  or LU choice differs, equals 07's call bit for bit, with `phase_span` set aside. Every row all of
  whose calls are such equals 07's in `total` and `total_abserr`. **Holds** if true at both rtols.
- **W2 — no Levin leaf is returned across a stationary point** (07's V2): the scan finds no returned
  Levin leaf holding a sign change of $\theta'$.
- **W3 — every phase with something to locate is located.** The location agrees with the scan both
  ways (07's V3), with no phase set aside except one whose rate is identically zero; every skipped
  phase is listed with its call, and each is of that kind.
- **W4 — honest where it is referenced.** V4's 33 calls (z00, z12, Z3's seven and the 24),
  re-scored from the census against the readings 07 scored them on, by 07's rule. **Holds** if
  every scored call is honest, or is unconverged under a rule of §9's amendment.
- **W5 — the gate is consistent, and the guard is a backstop.** In every call: no Clenshaw–Curtis
  region has a half sent to the Levin solve; no region takes the node-partition sum; no region is
  guarded (`num_guarded_regions` 0). **Holds** if all three are zero. A non-zero count is a finding,
  with its calls listed.
- **W6 — the pairs.** The cross-check's $r$ re-computed at $h$ over all 2,484 pairs: every pair has
  $r \le 1$, or a member unconverged under §9's rule. `3-14501` and `3-35006` are listed.

**Descriptive, not scored:**
- **Route changes:** the regions and calls that changed route or LU choice, by kind, band, $k$ and
  rtol, in which direction, and how far the old estimate stood from $V$.
- **Moves:** totals and calls against 07's, relative and against `total_abserr`$_{07}$.
- **The newly located:** calls whose phase Rule A skipped at 07, with their points and window
  pieces; any call locating more than 64 points.
- **The windows:** windows stopped, leaves falling at the stop, and calls left unconverged by the
  stop, against 07's 1,157/1,153 and 1,195/1,192.
- **Convergence and the flag,** in the 720 and outside, against 07's; and the depth distribution.
- **The disagreement list:** 07's cross-check list re-made at $h$ against 06m and against 07, by
  class.
- **Cost:** regions and wall against 07's. Cost is not an argument (README §5 rule 4).

## 2. Prompt §9.5, the additions to §6's predictions (verbatim)

### 9.5 Additions to §6's predictions

- **W4 is predicted to fail** on exactly v08, v09, v12, v17 and v22, at 3.16, 2.21, 3.08, 2.38 and
  2.53 to three figures, and to hold on the other 28 calls.
  - Basis: §9.2. Neither §1.2 nor O9 reaches the leaf that carries the excess
    (`[07a-levin-leaf-beside-a-window-understates-its-error]`).
  - A move of any of the five is reported with the regions it changed.
- **W6 is scored on $r_g$** (§9.1). It is predicted to fail on `3-14501` ($r_g$ 1.39) and `3-35006`
  (1.08), as at 07, and to hold on the other 2,482 pairs.
- **W7 — the stop's change is small.** The rule covers every call that has a window at 07 and at
  $h$, no region whose route or LU choice differs (the diagnostics), and no newly located phase.
  - Each such call moves from 07's value by at most $10^{-5}$ of 07's abserr, and keeps 07's
    `converged`.
  - **Holds** if true at both rtols.
  - Basis: (d) against (a) on 430 calls, largest move $1.76\times10^{-7}$ of abserr and no flag
    changed. The threshold is 57 times that.

## 3. Prompt §10.4, the descriptive item (verbatim)

**Into §6's descriptive items:** the census lists every call that goes from converged to
unconverged with a region whose route changed and no window, with its $V$, 07's estimate and the
abserr under each code, for that issue.

## 4. This agent's additions, descriptive and not scored (the dispatch brief)

- **§10.4's item** (part 3), as the census lists it: every call that goes from converged (07) to unconverged (h) with a
  solve whose route changed and no window at 07 or at h, with its V, 07's estimate, and the abserr under each code. "A
  region whose route changed" is listed under both readings of part 5's question 1 (an own solve; any solve).
- **The largest |theta| sampled at a gated region's end**, by kind (group, analytic) and rtol: the largest of
  |theta(a)|, |theta(b)| over every non-forced solve's builds (section 6's diagnostics, ``theta_end_max_nonforced``),
  with its ulp.
- **The smallest margins |V - 6 pi| and |V - 20 pi|** among solves whose route and LU choice did not change (the
  diagnostics' ``min_margin_6pi_same`` and ``min_margin_20pi_same_levin``, the latter among solves on the Levin route),
  by kind and rtol, with their V and 07's estimate.

## 5. How each rule is read (this agent's, fixed here before the census; the open questions are the user's)

**The diagnostics** (worker.py, ``route_07a``; out/w0/ shows they change nothing). For every subregion solve the driver
makes in a call (each call of ``_adaptive_levin_subregion``), they record its span, its role, whether it was forced to
the nested Clenshaw-Curtis pair, and, for each build of its Levin data, V (h's), 07's estimate mean|theta'| * width from
the same returned theta_prime_Cheb and grid by the parent's expression, and whether V took the node-partition sum. A
solve's **decision differs** if at some build its C2 route differs (V < 6 pi against the estimate < 6 pi), or both
routes are Levin and its LU choice differs (> 20 pi). A forced solve (a window piece, a window leaf's halves and
quarters, a guarded leaf at depth_max) makes no route decision and is never counted as differing. Each solve is
classed by **role**:
- ``own``: its span is a region the driver made (a ``used_interval``: returned, or made and later bisected), so its
  value and error became, at some point, a region's own;
- otherwise by the description the driver gave it: ``quarter`` (a trust-check quarter: "trust check: ..."),
  ``comparison_half`` (a Levin leaf's step-(4) halves, "Levin left/right-comparison region", data_L/data_R),
  ``fallback_half`` (a fallback leaf's halves, solved when it is chosen), ``other``.
The harness's own re-solves after the timed section (06b's W3, 06c's deep lines, 06e's floor terms, 06h's declared terms
and its analytic counterfactual) are not the call's and are tallied apart (``route_07a_post``).

**Question 1 (open; the user's): what W1's "region whose route or LU choice differs" and W7's counts.** R30's 49
abserr-only records each had exactly one changed solve, never a region's own: a trust-check quarter or a comparison half.
Such a solve's value enters the call's declared error (the trust check's delta1 and delta2, a Levin leaf's residual), not
its value. Two readings, both scored:
- **reading A:** every solve the driver makes counts (own and auxiliary): W1 covers only calls in which no solve of any
  role changed;
- **reading B:** only a call's own region solves count: W1 also covers calls whose only changes are auxiliary solves,
  and predicts them bit-identical (R30 says they are not, in abserr).
The same choice applies to W7's "no region whose route or LU choice differs". The census's comparison classes every call
without a window as ``untouched`` (no solve changed), ``aux_only`` (only auxiliary solves changed) or ``own`` (an own
solve changed), and reports W1 and W7 under each reading. This agent does not choose.

**Question 2 (open; the user's): what W1's "equals 07's call bit for bit" compares.** Every call is diffed against 07's on
everything 07's comparison takes as the call's, and each differing path is put in one category:
- ``call``: the call's own outputs (value, abserr, converged, regions, leaves, its solve counts, its trust checks' and
  premise checks' records, guard_07's windows and counts, its scans);
- ``location``: guard_07's ``location_skipped_phases``, ``unlocated_minima`` and ``n_unlocated_minima``: the location's
  own report. §1.1 item 1 changes it by construction for every call whose phase Rule A skipped at 07: 07's entry
  carries "variation", h's (only for a rate identically zero) "reason", and a phase Rule A skipped is now scanned, so it
  can newly report an unlocated minimum even where it locates nothing and makes no window;
- ``harness``: the harness's own re-solves after the timed section (06e's floor terms, 06h's declared terms, 06c's and
  06e's deep lines, 06b's W3), which re-solve spans the driver may not have visited, under h's routing.
W1 is scored three ways: **strict** (no differing path), **as written for phase_span** (the ``location`` and ``harness``
categories set aside as phase_span is: each is a record, not a computed value of the call, and each differing call's
``route_07a_post`` or location record is shown to explain it), and the counts in each category. This agent does not
choose. (``phase_span`` itself is not a key of any record of the harness at 07 or at h: compare_07a.py counts its
occurrences in every unit compared, and sets it aside by showing it is absent.)

**W1's rows.** A unit all of whose calls are of W1's classes (under the reading) is scored on ``result.total`` and
``result.total_abserr``, hex, as written; its whole-unit diff is reported with its categories (the unit's 06h analytic
counterfactual is ``harness``).

**W2** is 07's V2 on the census's scan_07 (every group call at both rtols, every analytic call at 1e-7), as 07 scored it.

**W3** is 07's V3 on the scanned calls, as 07 scored it, with 07 §10's set-aside replaced by 07a's: a phase is set aside
only if its entry in ``location_skipped_phases`` says ``reason: zero_rate``; each such call is listed and its rate is
checked to be exactly zero on the scan (scan_07: no sign change, and min |theta'| / max |theta'| of 0 or no finite
non-zero sample). A skipped phase of any other kind fails W3.

**W4** is scored by 07's rule (score_v4_07.py's) on 07's readings, with "the before" now 07's census and "the new" h's:
the tolerance min(abserr_07, abserr_h)/100, the readings' agreement test unchanged in form, honest iff |v_h - ref| <=
abserr_h. A call whose readings do not agree under the new tolerance is reported, not scored, and its verdict under 07's
own tolerance is shown beside it. §9's amendment adopted no rule that leaves a call unconverged (§9.3), so W4's escape
clause "or is unconverged under a rule of §9's amendment" admits no call. The prediction of §9.5 (fails on exactly v08,
v09, v12, v17, v22 at 3.16, 2.21, 3.08, 2.38, 2.53 to three figures; holds on the other 28) is scored as written: to three
significant figures.

**W5:** the three counts are made on every solve of every call (forced included), each broken down by role, and the
node-partition count is also given for the gated (non-forced) solves alone, which is how Part B's review counted it
("493,516 regions gated"). "A Clenshaw-Curtis region with a half sent to the Levin solve": a non-forced solve that took
the nested pair (is_direct) and a visited non-forced solve of one of its halves (its break point (a + b)/2) that took
the Levin solve. The harness's own re-solves are not regions of the call and are not counted. ``num_guarded_regions`` is
guard_07's, as 07 recorded it.

**W6** is the cross-check's r_g at h, r_g = |v_h(1e-7) - v_h(1e-8)| / e_h(1e-7) (state/xcheck_rule_20261109.md's
part A, 07's values replaced by h's), on 07's 2,484 pairs (the calls 07 marked as locating or guarding), matched as 07
matched them. "A member unconverged under §9's rule": none can be, as for W4. The pairs h itself marks (newly located
calls included) are reported beside it, not scored.

**W7** is scored on the calls with a window at 07 and at h (guard_07 ``n_windows`` > 0 in both), no differing decision
(under each reading of question 1), and no newly located phase (a phase in 07's ``location_skipped_phases`` that h
located): |v_h - v_07| <= 1e-5 abserr_07 (exactly, as Fractions) and converged_h == converged_07.

**The over-budget pass:** a unit over budget in both passes is counted as not scored for W1 and W7 and listed.

## 6. V4's 33 calls, fixed by name (07's set, from 07's records; W4 is scored on 07's readings)

From `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3/docs/levin-phase-floor/07/census/out/v4/v4_scores.json` (SHA-256 2d0268e607e0cb17b4d4247831a7b933c2df0db61c248b8959ecdf9f091649b8), its `calls` (the 24) and `z` (z00, z12 and Z3's seven). The unit is 07's name (`g-...`); at h it is `h-...`. Each call is the unit's Levin call of that index (group: `calls`; analytic: `other_calls`), matched at h by index, label and span as compare_07a.py matches calls.

| call | unit | rtol | kind | index | 07's basis | 07's ref (30 digits) | 07's |v-ref|/abserr | 07's verdict |
|---|---|---|---|---|---|---|---|---|
| v00 | g-1e-7-3-37343 | 1e-7 | group | 1 | four readings | 1.828064505735055e-27 | 0.01967 | honest |
| v01 | g-1e-7-2-34834 | 1e-7 | group | 1 | FCC r1 and r2 | 4.875138519044203e-26 | 0.03252 | honest |
| v02 | g-1e-8-3-39137 | 1e-8 | group | 1 | four readings | 1.7794421219243247e-24 | 0.005717 | honest |
| v03 | g-1e-8-1-34611 | 1e-8 | group | 1 | four readings | -1.3417123142647598e-24 | 0.001751 | honest |
| v04 | g-1e-7-3-2326 | 1e-7 | group | 1 | four readings | -1.697835445994946e-21 | 0.1377 | honest |
| v05 | g-1e-7-0-38985 | 1e-7 | group | 1 | four readings | -7.226924936085433e-26 | 0.001238 | honest |
| v06 | g-1e-7-1-2120 | 1e-7 | group | 1 | FCC r1 and r2 | 1.3253807037988687e-24 | 0.0002783 | honest |
| v07 | g-1e-7-3-14592 | 1e-7 | group | 1 | FCC r1 and r2 | 1.5858416698368707e-27 | 0.0004815 | honest |
| v08 | g-1e-7-1-38834 | 1e-7 | group | 1 | FCC r1 and r2 | -3.4105380540062642e-31 | 3.16 | NOT honest |
| v09 | g-1e-8-3-34962 | 1e-8 | group | 1 | four readings | -2.478150779742095e-25 | 2.211 | NOT honest |
| v10 | g-1e-8-0-14314 | 1e-8 | group | 1 | four readings | 7.981860679850132e-26 | 0.1924 | honest |
| v11 | g-1e-8-2-2189 | 1e-8 | group | 1 | FCC r1 and r2 | 3.901361900446531e-26 | 0.007617 | honest |
| v12 | g-1e-8-0-14299 | 1e-8 | group | 1 | FCC r1 and r2 | 5.316056019237085e-29 | 3.08 | NOT honest |
| v13 | g-1e-8-2-34841 | 1e-8 | group | 1 | FCC r1 and r2 | 1.8336335617074055e-32 | 0.309 | honest |
| v14 | g-1e-7-3-13816 | 1e-7 | group | 1 | four readings | 3.232725490521406e-23 | 0.001021 | honest |
| v15 | g-1e-7-0-1973 | 1e-7 | group | 1 | four readings | -4.745014484607306e-27 | 0.002042 | honest |
| v16 | g-1e-7-2-2092 | 1e-7 | group | 1 | FCC r1 and r2 | -5.1548456372611367e-26 | 0.03819 | honest |
| v17 | g-1e-7-0-1772 | 1e-7 | group | 1 | four readings | -1.4185437200041364e-29 | 2.382 | NOT honest |
| v18 | g-1e-7-2-13649 | 1e-7 | group | 1 | FCC r1 and r2 | 2.923289549764128e-32 | 0.5289 | honest |
| v19 | g-1e-8-3-1921 | 1e-8 | group | 1 | four readings | 2.257993965209488e-23 | 0.04621 | honest |
| v20 | g-1e-8-0-1896 | 1e-8 | group | 1 | four readings | 5.637990679304912e-25 | 0.01414 | honest |
| v21 | g-1e-8-1-1747 | 1e-8 | group | 1 | four readings | -3.8423689484434757e-25 | 0.008619 | honest |
| v22 | g-1e-8-3-14096 | 1e-8 | group | 1 | FCC r1 and r2 | -5.977741175450714e-28 | 2.53 | NOT honest |
| v23 | g-1e-8-1-34772 | 1e-8 | group | 1 | FCC r1 and r2 | -1.9402737752240047e-30 | 0.7671 | honest |
| z00 | g-1e-7-1-2174 | 1e-7 | group | 1 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1', 'qawo_r2'] | 1.3549296929382802e-26 | 0.7181 | honest |
| z12 | g-1e-8-1-2174 | 1e-8 | group | 1 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1', 'qawo_r2'] | 1.3549296929382802e-26 | 0.7181 | honest |
| z07 | g-1e-7-2-34718 | 1e-7 | analytic | 11 | 06g's readings: ['fcc_r1', 'fcc_r2'] | -2.433000368553533e-13 | 1.525e-06 | honest |
| z16 | g-1e-8-2-34708 | 1e-8 | analytic | 5 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1'] | -9.618793963065276e-12 | 6.543e-06 | honest |
| z18 | g-1e-8-1-2368 | 1e-8 | analytic | 9 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 6.967220717606601e-13 | 7.388e-07 | honest |
| z21 | g-1e-8-1-14572 | 1e-8 | analytic | 8 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 2.3690345571332306e-13 | 3.482e-07 | honest |
| z24 | g-1e-8-2-34718 | 1e-8 | analytic | 5 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 5.111888717223634e-15 | 0.002074 | honest |
| z28 | g-1e-7-2-34708 | 1e-7 | analytic | 5 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1'] | -9.618793963065276e-12 | 6.543e-06 | honest |
| z29 | g-1e-7-1-14572 | 1e-7 | analytic | 9 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 6.967220717614978e-13 | 2.342e-07 | honest |

24 + 9 = 33 calls.

