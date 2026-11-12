# STOP: question (section 2.2's cost rule, and the cost of variants (b) and (g)) — 07a Part A, stop 1

Measuring agent, Mac Studio, 2026-11-10. Records: `/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/measurement/`
(`log.md` written first; `verification.md`, `issues.md`, `out/tables_interim_1.md`, `out/attr/attr_tables_interim_1.md`,
`out/b_scaling/`). Nothing recommended.

## Where it stands

- **The Gate holds** on the Mac Studio. **Item 2's captures check**, 9 of 9. **Every replay at 07's code reproduces its
  census record bit for bit**: 430 of 430 at the tree and at (a). That includes the 396 at 1e-8; 390 of them were captured
  at 1e-7 and replayed with the 1e-8 call's rtol and atol.
- **Variants (a), (c), (d), (e), (f) have run on all 430 calls.** The no-window check passes, 5 of 5 each.
  - (c) and (d) count exactly as 07 on all 34 calls of items 1–2. In their 382 window bisections every parent is
    untrusted, none has a widened declaration, no half is "unresolved", and the value changes are 1e-9–1e-12 relative.
  - (d) moves 58 of the 396.
  - (e) and (f) unconverge v03, the two 1e-8 pair members and 35 of the 396.
  - **None of (a) and (c)–(f) changes any of the five failures**: v08, v09, v12, v17 and v22 stay converged at 3.16,
    2.21, 3.08, 2.38 and 2.53. In those five the call's request was met when the window stopped.
- **Attribution, three calls so far.** All three close (|Σref_i − ref|/tol 1.5e-4–1.8e-3).
  - **v22: all of the excess error is H3.** It sits in one Levin leaf just right of the window, [9.2321, 9.7509], with
    true error +2.04e-33 against e′ 4.37e-34. The window's 15 leaves are wrong by about 1e-40 in all.
  - v21 and v03 (controls) are honest in every part.
- **(b) at depth_max 20 on v12: 16,210 regions, 609 s against 07's 0.5 s.** It is unconverged, with 11,490 window
  leaves at depth_max. Its value and abserr are the same as 07's to about 1e-13 relative (ratio 3.08).
- **(b) on v08:** 86, 200, 472, 1,125, 2,718, 6,691 and 16,609 regions at depth_max 6–18, taking 669 s at 18. It is
  unconverged at every depth, and abserr stays at 1.94e-36 throughout.

## Question 1 — section 2.2's cost rule

The pilot (v22) projects the remaining attributions at **9.4 h**, under 12 h. But v21 and v03 then cost about 1.0× 07's
own reading CPU, where v22 cost 0.37×, so the realistic projection is **about 25 h** on six workers.
- The cost is dominated by **v13** and **v18**: 144,000 and 149,000 CPU-s at 07, together 54% of the total. Both are
  honest controls (0.309, 0.529).
- **To fit 12 h, drop v13 and v18** (about 11.7 h); dropping **v23** as well gives about 9.5 h.
- The five failures and z12 are kept in either case.
- Held: the 20 remaining attributions. Still running, under any answer: the serial run `attr-1` of the six cheapest
  controls.

## Question 2 — variants (b) and (g) on the 396

The prompt gives no bound for these.
- (b) costs about 10–40 min per call, so the 396 at (b) alone come to roughly **24 h** at six processes; (g) is likely
  similar.
- On every call probed, (b) is unconverged at depth_max 20. If that holds, **item 3 scores none of the 396**: it scores
  only where (b) converges with abserr_b ≤ abserr/10, so every call would be undecided.
- Running now, needed under any answer: (b2) and (g2) on items 1–2 and the five no-window calls (34 calls, registered run
  `levin-phase-floor-07a-bg2-items12-20261110T223735`). They are b and g with an instrument cost removed (log deviation 10).

**Choices for the user.** Order is not a preference.
- **(A)** Run (b2) and (g2) on all 396 as written.
- **(B)** Run (b2) and (g2) on the 396 only if (b2) converges on some call of items 1–2. Otherwise report item 3 as
  undecided by construction.
- **(C)** Another bound the user names.
