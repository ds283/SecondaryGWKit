# STOP: planned §2.5 — 07a Part A, the measurement's report

Measuring agent, Mac Studio (mps026298), 2026-11-11. Worktree `SecondaryGWKit-lpf07a-f93aafd` at `f93aafd0` (07's
code); records in `/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/measurement/`. `log.md` was written first (22 deviations), then `verification.md` and `issues.md`
(6 proposals, none opened). Nothing is recommended here.

## Summary

- The **Gate** holds before and after. Item 2's captures check, 9 of 9. Every replay at 07's code reproduces its census
  record bit for bit, 430 of 430. The **no-window check** passes, 5 of 5, for every variant: (a), (b2), (c), (d), (e),
  (f), (g2).
- **No variant changes any of the five failures.** v08, v09, v12, v17 and v22 keep |v - ref| / abserr = 3.16, 2.21,
  3.08, 2.38 and 2.53 at every variant:
  - converged and dishonest at the tree, (a), (c), (d), (e) and (f);
  - unconverged at (b2) and (g2).
- **The attribution, 24 calls, all closing (6.2e-6 to 3.3e-3 of tol).**
  - **In all five failures the whole excess is in H3**, the Levin leaves outside the windows. H1 and H2 are each 1e-9
    to 5e-7 of abserr.
  - The excess sits almost entirely in one H3 leaf next to the window (share 0.999 to 1.008).
  - In v08, v12 and v17 that leaf is the depth-0 seed region from the window's upper edge to the end of the span, about
    [9.23, 27-28], solved once as a Levin leaf and never bisected; its true error / e' is 3.63, 7.44 and 2.94.
  - In v09 the leaf is at depth 1 (2.66), in v22 at depth 5 (4.67).
- **The O9 check** (added at the user's word) covers 120 regions: each H3 leaf carrying excess, plus the top three
  H3 leaves by |true error|, with every leaf's parent and sibling.
  - No route change, and no strict sign change of theta', on any of them.
  - The true variation is 0.47-1.39x 07's `phase_span`, and 0.47-1.06x on the main leaves. Every main leaf is far
    above 20 pi, so it took the LU route on both measures.
- **Qualification (item 4)** is scored on r_g, with item 3 undecided by construction.
  - **Only (b2) and (g2) qualify**, and only because every one of the 29 window calls of items 1-2 is unconverged:
    none is "converged and dishonest".
  - Their cost on items 1-2 is 764,283 regions and about 50,500-51,500 s, against (a)'s 2,386 regions and 48 s.
  - (b2) and (g2) are equal bit for bit on all 34 calls: (g)'s rule never stops a window here.
  - (c), (d), (e) and (f) fail on the five failures, and (c) and (d) also on both pairs. (e) and (f) pass the pairs
    only by leaving a member unconverged.
- **Item 3 is undecided by construction**, by the user's ruling: (b) is not run on the 396, and (b2) converged on none
  of the 29 window calls.

## Item 4: qualification and cost (items 2 and 4 scored on r_g)

## 4. Qualification and cost

| variant | qualifies (items 1-2) | conv. & dishonest V4 | conv. & dishonest 396 | pairs failing | 396 scored / undecided | unconverged V4 / pairs / 396 | regions | wall s | regions 1-2 | wall s 1-2 | windows stopped | leaves falling at stop | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tree | no | v08, v09, v12, v17, v22 | 0 | p14501, p35006 | 0 / 396 | 1 / 0 / 4 | 29555 | 615 | 2386 | 48 | 408 | 4325 | 0 |
| a | no | v08, v09, v12, v17, v22 | 0 | p14501, p35006 | 0 / 396 | 1 / 0 / 4 | 29555 | 625 | 2386 | 48 | 408 | 4325 | 0 |
| b2 | yes | - | n/a | - | 0 / 396 | 25 / 4 / not run | 764283 | 50555 | 764283 | 50555 | 0 | 0 | 0 |
| c | no | v08, v09, v12, v17, v22 | 0 | p14501, p35006 | 0 / 396 | 1 / 0 / 4 | 29555 | 628 | 2386 | 48 | 408 | 4325 | 0 |
| d | no | v08, v09, v12, v17, v22 | 0 | p14501, p35006 | 0 / 396 | 1 / 0 / 4 | 29827 | 629 | 2386 | 48 | 408 | 4543 | 0 |
| e | no | v08, v09, v12, v17, v22 | 0 | - | 0 / 396 | 2 / 2 / 39 | 29555 | 617 | 2386 | 49 | 408 | 4325 | 0 |
| f | no | v08, v09, v12, v17, v22 | 0 | - | 0 / 396 | 2 / 2 / 39 | 29555 | 619 | 2386 | 49 | 408 | 4325 | 0 |
| g2 | yes | - | n/a | - | 0 / 396 | 25 / 4 / not run | 764283 | 51494 | 764283 | 51494 | 0 | 0 | 0 |

## Item 1: honesty on V4's 25

## 1. Honesty on V4's 25: |v - ref| / abserr (* converged and dishonest; u unconverged)

| call | tree | a | b2 | c | d | e | f | g2 |
|---|---|---|---|---|---|---|---|---|
| cal_z12 | 0.718 | 0.718 | 0.718 u | 0.718 | 0.718 | 0.718 | 0.718 | 0.718 u |
| v00 | 0.0197 | 0.0197 | 0.0197 u | 0.0197 | 0.0197 | 0.0197 | 0.0197 | 0.0197 u |
| v01 | 0.0325 | 0.0325 | 0.0325 u | 0.0325 | 0.0325 | 0.0325 | 0.0325 | 0.0325 u |
| v02 | 0.00572 | 0.00572 | 0.00572 u | 0.00572 | 0.00572 | 0.00572 | 0.00572 | 0.00572 u |
| v03 | 0.00175 | 0.00175 | 0.00175 u | 0.00175 | 0.00175 | 0.00175 u | 0.00175 u | 0.00175 u |
| v04 | 0.138 u | 0.138 u | 0.138 u | 0.138 u | 0.138 u | 0.138 u | 0.138 u | 0.138 u |
| v05 | 0.00124 | 0.00124 | 0.00124 u | 0.00124 | 0.00124 | 0.00124 | 0.00124 | 0.00124 u |
| v06 | 0.000278 | 0.000278 | 0.000278 u | 0.000278 | 0.000278 | 0.000278 | 0.000278 | 0.000278 u |
| v07 | 0.000481 | 0.000481 | 0.000482 u | 0.000481 | 0.000481 | 0.000481 | 0.000481 | 0.000482 u |
| v08 (failure) | 3.16* | 3.16* | 3.16 u | 3.16* | 3.16* | 3.16* | 3.16* | 3.16 u |
| v09 (failure) | 2.21* | 2.21* | 2.21 u | 2.21* | 2.21* | 2.21* | 2.21* | 2.21 u |
| v10 | 0.192 | 0.192 | 0.192 u | 0.192 | 0.192 | 0.192 | 0.192 | 0.192 u |
| v11 | 0.00762 | 0.00762 | 0.00762 u | 0.00762 | 0.00762 | 0.00762 | 0.00762 | 0.00762 u |
| v12 (failure) | 3.08* | 3.08* | 3.08 u | 3.08* | 3.08* | 3.08* | 3.08* | 3.08 u |
| v13 | 0.309 | 0.309 | 0.309 u | 0.309 | 0.309 | 0.309 | 0.309 | 0.309 u |
| v14 | 0.00102 | 0.00102 | 0.00102 u | 0.00102 | 0.00102 | 0.00102 | 0.00102 | 0.00102 u |
| v15 | 0.00204 | 0.00204 | 0.00205 u | 0.00204 | 0.00204 | 0.00204 | 0.00204 | 0.00205 u |
| v16 | 0.0382 | 0.0382 | 0.0382 u | 0.0382 | 0.0382 | 0.0382 | 0.0382 | 0.0382 u |
| v17 (failure) | 2.38* | 2.38* | 2.38 u | 2.38* | 2.38* | 2.38* | 2.38* | 2.38 u |
| v18 | 0.529 | 0.529 | 0.529 u | 0.529 | 0.529 | 0.529 | 0.529 | 0.529 u |
| v19 | 0.0462 | 0.0462 | 0.0462 u | 0.0462 | 0.0462 | 0.0462 | 0.0462 | 0.0462 u |
| v20 | 0.0141 | 0.0141 | 0.0141 u | 0.0141 | 0.0141 | 0.0141 | 0.0141 | 0.0141 u |
| v21 | 0.00862 | 0.00862 | 0.00863 u | 0.00862 | 0.00862 | 0.00862 | 0.00862 | 0.00863 u |
| v22 (failure) | 2.53* | 2.53* | 2.53 u | 2.53* | 2.53* | 2.53* | 2.53* | 2.53 u |
| v23 | 0.767 | 0.767 | 0.767 u | 0.767 | 0.767 | 0.767 | 0.767 | 0.767 u |

## Item 2: the pairs

## 2. The pairs, scored on r_g = |v_7 - v_8| / e_7, the cross-check's; in brackets the prompt's r = |v_7 - v_8| / (e_7 + e_8), for the record (u: a member unconverged; * converged and r_g > 1)

| pair | tree | a | b2 | c | d | e | f | g2 |
|---|---|---|---|---|---|---|---|---|
| p14501 | r_g 1.39 (r 0.727)* | r_g 1.39 (r 0.727)* | r_g 1.39 (r 0.727) u | r_g 1.39 (r 0.727)* | r_g 1.39 (r 0.727)* | r_g 1.39 (r 0.727) u | r_g 1.39 (r 0.727) u | r_g 1.39 (r 0.727) u |
| p35006 | r_g 1.08 (r 0.794)* | r_g 1.08 (r 0.794)* | r_g 1.08 (r 0.794) u | r_g 1.08 (r 0.794)* | r_g 1.08 (r 0.794)* | r_g 1.08 (r 0.794) u | r_g 1.08 (r 0.794) u | r_g 1.08 (r 0.794) u |

## Item 3: the 396

## 3. The 396 against (b): undecided by construction ((b) not run on the 396, the user's ruling at stop 1; (b2) converged on none of the 29 window calls of items 1-2)

| variant | scored | undecided | honest | converged & dishonest | unconverged & dishonest | max ratio |
|---|---|---|---|---|---|---|
| tree | 0 | 396 | 0 | 0 | 0 | nan |
| a | 0 | 396 | 0 | 0 | 0 | nan |
| b2 | 0 | 396 | 0 | 0 | 0 | nan |
| c | 0 | 396 | 0 | 0 | 0 | nan |
| d | 0 | 396 | 0 | 0 | 0 | nan |
| e | 0 | 396 | 0 | 0 | 0 | nan |
| f | 0 | 396 | 0 | 0 | 0 | nan |
| g2 | 0 | 396 | 0 | 0 | 0 | nan |

The 396 at the variants that ran there, against (a):
- (d) changes 58 calls (value, abserr or regions);
- (e) and (f) leave 35 more unconverged (39 against 4);
- (c) changes none.

## Item 5: the attribution (H1-H3) per call, with the closure check

Each part is given as its true error / the call's abserr, with |true| / its own declared error in brackets. Full
per-leaf tables: `out/attr/attr_tables_final.md`.

| call | abs(v - ref) / abserr | H1 falling: true / call abserr (true/decl) | H2 other window | H3 outside | closure / tol | leaves below 100x rule |
|---|---|---|---|---|---|---|
| v08 (failure) | +3.160 | +3.64e-08 (0.228) | -4.51e-07 (0.817) | +3.16e+00 (3.16) | 3.29e-03 | 29 of 31 |
| v09 (failure) | +2.211 | +1.77e-08 (0.22) | +8.80e-09 (0.166) | +2.21e+00 (2.21) | 6.16e-06 | 36 of 39 |
| v12 (failure) | +3.080 | +5.37e-09 (0.0223) | +1.15e-08 (0.0936) | +3.08e+00 (3.08) | 2.97e-05 | 22 of 25 |
| v17 (failure) | +2.382 | -3.42e-08 (0.265) | -1.96e-07 (0.487) | -2.38e+00 (2.38) | 3.02e-06 | 20 of 22 |
| v22 (failure) | +2.530 | -2.88e-08 (0.23) | -8.82e-08 (0.486) | +2.53e+00 (2.53) | 1.49e-04 | 80 of 124 |
| cal_z12 | +0.718 | -2.30e-06 (0.376) | -1.49e-07 (0.0712) | +7.18e-01 (0.718) | 3.16e-04 | 62 of 70 |
| v00 | +0.020 | -2.45e-09 (0.00145) | +5.81e-08 (0.105) | -1.97e-02 (0.0197) | 5.63e-04 | 32 of 37 |
| v01 | +0.033 | +1.05e-07 (0.0557) | +6.53e-09 (0.005) | +3.25e-02 (0.0325) | 2.64e-04 | 45 of 51 |
| v02 | +0.006 | -2.28e-07 (0.396) | -3.69e-07 (1.31) | +5.71e-03 (0.00571) | 4.87e-04 | 90 of 152 |
| v03 | +0.002 | -6.86e-07 (0.0726) | -2.37e-06 (0.507) | -1.75e-03 (0.00175) | 1.03e-04 | 83 of 161 |
| v05 | +0.001 | -4.91e-09 (0.0253) | +4.78e-08 (0.477) | -1.24e-03 (0.00124) | 6.25e-05 | 66 of 138 |
| v06 | +0.000 | -1.73e-07 (0.178) | -1.50e-07 (0.137) | -2.78e-04 (0.000278) | 3.37e-05 | 68 of 118 |
| v07 | +0.000 | +9.46e-08 (0.047) | +1.72e-07 (0.62) | -4.82e-04 (0.000482) | 2.12e-05 | 61 of 97 |
| v10 | +0.192 | -1.88e-08 (0.254) | +8.48e-07 (1.04) | +1.92e-01 (0.192) | 1.78e-05 | 35 of 39 |
| v11 | +0.008 | -5.79e-07 (0.216) | +1.87e-06 (0.401) | -7.61e-03 (0.00761) | 3.54e-04 | 43 of 78 |
| v13 | +0.309 | -7.23e-08 (0.246) | +4.03e-08 (0.132) | +3.09e-01 (0.309) | 3.93e-04 | 31 of 33 |
| v14 | +0.001 | +7.20e-09 (0.051) | -2.13e-07 (0.898) | +1.02e-03 (0.00102) | 3.92e-05 | 31 of 46 |
| v15 | +0.002 | +3.96e-06 (0.0911) | +1.21e-05 (0.34) | -2.07e-03 (0.00207) | 8.56e-04 | 94 of 166 |
| v16 | +0.038 | -4.78e-07 (0.138) | -7.45e-07 (0.409) | -3.82e-02 (0.0382) | 4.21e-04 | 74 of 137 |
| v18 | +0.529 | +3.39e-08 (0.063) | +4.75e-09 (0.0275) | +5.29e-01 (0.529) | 1.23e-04 | 23 of 25 |
| v19 | +0.046 | +2.35e-08 (0.00513) | +3.08e-07 (0.0157) | +4.62e-02 (0.0462) | 2.04e-04 | 83 of 161 |
| v20 | +0.014 | -2.11e-06 (0.155) | -9.35e-06 (0.506) | -1.41e-02 (0.0141) | 1.94e-03 | 82 of 153 |
| v21 | +0.009 | +6.77e-06 (0.651) | +3.09e-06 (0.261) | -8.61e-03 (0.00861) | 1.79e-03 | 32 of 62 |
| v23 | +0.767 | +6.12e-08 (0.0706) | -6.61e-07 (0.856) | -7.67e-01 (0.767) | 1.62e-05 | 30 of 32 |

The 100x accuracy rule cannot be met on leaves whose e' is below about 100x the integrand's own double-precision noise
(log deviation 12, issues.md item 4). The parts' summed reference accuracies are 8.5e-8 to 1.7e-4 of each call's abserr, so every part is resolved.

## The O9 check on the H3 leaves of the five failures

| call | N | H3 leaves / carrying excess (share of H3 true error) | regions checked | route changes | theta' sign changes | true / estimate (min-max) | main leaf: depth, span, true/declared, share | its phase_span (07) / true variation | route (07 = true) | parent phase_span / route | sibling phase_span / route | input sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v08 | 24 | 16 / 4 (1.001) | 16 | 0 | 0 | 0.730-0.999 | d0, [9.23151, 28.1227], 3.63, 1.001 | 4.061e+08 / 2.965e+08 | Levin (LU tried first) | none (the leaf is a seed region) | none (the leaf is a seed region) | be977557779b |
| v09 | 24 | 20 / 8 (1.000) | 30 | 0 | 0 | 0.466-1.389 | d1, [12.8372, 24.6909], 2.66, 1.000 | 9.146e+06 / 4.258e+06 | Levin (LU tried first) | 2.227e+09 / Levin (LU tried first) | 1.818e+09 / Levin (LU tried first) | 7365c0d45f57 |
| v12 | 24 | 14 / 3 (1.008) | 13 | 0 | 0 | 0.752-0.998 | d0, [9.23164, 26.9711], 7.44, 1.008 | 1.257e+08 / 9.448e+07 | Levin (LU tried first) | none (the leaf is a seed region) | none (the leaf is a seed region) | d651bfa7dcd3 |
| v17 | 24 | 8 / 4 (1.000) | 16 | 0 | 0 | 0.752-0.999 | d0, [9.23164, 26.9711], 2.94, 1.000 | 1.257e+08 / 9.448e+07 | Levin (LU tried first) | none (the leaf is a seed region) | none (the leaf is a seed region) | 15aea2d1afa3 |
| v22 | 24 | 109 / 14 (1.000) | 45 | 0 | 0 | 0.915-1.121 | d5, [9.23186, 9.75095], 4.67, 0.999 | 4.033e+06 / 4.275e+06 | Levin (LU tried first) | 1.026e+07 / Levin (LU tried first) | 7.139e+06 / Levin (LU tried first) | b79bff109063 |

Full tables, every leaf with its parent and sibling: `out/o9/o9_tables.md`.

## Per-call output paths

- Replays: `out/replays/<variant>/<sid>.json`, for variant in tree, a, b2, c, d, e, f, g2 (b2 and g2 on items 1-2 and
  n0-n4 only). Instrument records: `out/replays/<v>/<sid>.instrument.json` (items 1-2, at a and c-f).
- Attribution: `out/attr/<sid>.json` (v22: `out/attr/pilot_v22.json`), each with its log `out/attr/<sid>.log`.
- O9: `out/o9/<sid>.{json,md}`.
- Tables: `out/tables_final_rg.{json,md}` (of record), `out/tables_final.{json,md}`, `out/tables_interim_1.*`,
  `out/attr/attr_tables_final.{json,md}`.
- (b) cost probe: `out/b_scaling/`.

## What section 2.3 does not cover

- **The H3 leaves.** Every variant of section 2.3 changes only what happens inside a window. The excess error of all
  five failures lies outside every window, in a Levin leaf next to one. In three of the five that leaf is a seed
  region that was never bisected.
- **Requests met at the stop.** In all five failures the call's request was met when the window stopped. (e) and (f)
  ("request unmet at the stop") therefore never act on them.
- **Trust, not budget, drives the bisections.** In all 382 window bisections of items 1-2 the parent leaf is
  untrusted, and 299 of them happen with the budget already met. dqage's iroff1 premise, a count of budget-forced
  bisections, does not describe these calls, and (c)'s and (d)'s departures never act (issues.md item 2).
- **(b) and (g) do not converge.** Neither converges on any window call of items 1-2, so neither yields a reference for
  item 3.

## State

- Nothing is running. 07a's four registered runs are `done`, and the lock was not used.
- 07's captures (78 of 78) and the scratch copy (5 of 5) match before and after; 07's census checkpoint is `03dfe786...`.
- The worktree is clean, including ignored files. The main checkout is at `28ebaf07`, clean.
- `KEPT_ON_MAC_STUDIO.txt` lists 61 files (every pickle, and every file over 2 MB).
