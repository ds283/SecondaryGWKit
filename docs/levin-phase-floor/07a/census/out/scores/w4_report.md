# levin-phase-floor 07a: W4 scored (V4's 33 calls at h, on 07's readings)
# written 2026-11-12T20:08:50Z on mps026298.phys.susx.ac.uk
# script score_v4_07a.py sha256 da39b6f763a658f1fa6e668d733a2242502b82c35fc7366345b83c802c83ab32
# tree h: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3 HEAD 4bf50d3426dc4c9ba190760e6d30e5b7d6dd0b03 (status --short: empty)
# main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/census/state/plan_references.json sha256 93506ccbee960f79741fedac6ad5679df97274eaf762a9a08a6c8bfd169d78fe
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/census/state/v4_reading_list_20261109.json sha256 ce594095f08e0010a955fcd8c4a7027acd905662350850616ddfe9217a9f6e82
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/census/out/v4/inputs.json sha256 55b855d4f71feba93f4a9d578530251d38d17b11c63267cee2241f109a222b11
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/census/state/v4_selection.json sha256 15a4391b8e49c547a0557bc20829faf5c2323f0e826420ef0e21d694b134d6c8
# input /Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3/docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json sha256 73c18680d266b1a0fdf11123c104d20e822658047dd7c36e1c814db9ffcad912
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07a-census-20261112T132744/checkpoint.jsonl sha256 461df2ec66969a113e9f6d590f05a4d5b300d7c1e0cb62d05c7385d23e8de56b
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/census/state/index_07.json sha256 17b08699957c244d404579372b718600396a65d7dee807dd9b05bed3f074c367
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/census/out/v4/v4_scores.json sha256 2d0268e607e0cb17b4d4247831a7b933c2df0db61c248b8959ecdf9f091649b8

# V4 scored

V4: FAILS: 28 of 33 scored calls honest; not honest ['v08', 'v09', 'v12', 'v17', 'v22']; reported, not scored [].
Totals: {"24": {"NOT honest": 5, "honest": 19}, "z": {"honest": 9}, "all": {"honest": 28, "NOT honest": 5}, "basis_24": {"FCC r1 and r2": 11, "four readings": 13}}

| call | unit (07's name) | rtol | basis | |v_h - ref|/abserr_h | |v_07 - ref|/abserr_07 | spread/tol | verdict | notes |
|---|---|---|---|---|---|---|---|---|
| v00 | g-1e-7-3-37343 | 1e-7 | four readings | 0.01967 | 0.01967 | 4e-05 | honest |  |
| v01 | g-1e-7-2-34834 | 1e-7 | FCC r1 and r2 | 0.03252 | 0.03252 | 0.000162 | honest |  |
| v02 | g-1e-8-3-39137 | 1e-8 | four readings | 0.005717 | 0.005717 | 9.38e-05 | honest |  |
| v03 | g-1e-8-1-34611 | 1e-8 | four readings | 0.001751 | 0.001751 | 0.000614 | honest |  |
| v04 | g-1e-7-3-2326 | 1e-7 | four readings | 0.1377 | 0.1377 | 1.19e-05 | honest |  |
| v05 | g-1e-7-0-38985 | 1e-7 | four readings | 0.001238 | 0.001238 | 5.47e-05 | honest |  |
| v06 | g-1e-7-1-2120 | 1e-7 | FCC r1 and r2 | 0.0002783 | 0.0002783 | 0.000168 | honest |  |
| v07 | g-1e-7-3-14592 | 1e-7 | FCC r1 and r2 | 0.0004815 | 0.0004815 | 8.37e-06 | honest |  |
| v08 | g-1e-7-1-38834 | 1e-7 | FCC r1 and r2 | 3.16 | 3.16 | 3.93e-05 | NOT honest |  |
| v09 | g-1e-8-3-34962 | 1e-8 | four readings | 2.211 | 2.211 | 0.00053 | NOT honest |  |
| v10 | g-1e-8-0-14314 | 1e-8 | four readings | 0.1924 | 0.1924 | 0.000209 | honest |  |
| v11 | g-1e-8-2-2189 | 1e-8 | FCC r1 and r2 | 0.007617 | 0.007617 | 0.000303 | honest |  |
| v12 | g-1e-8-0-14299 | 1e-8 | FCC r1 and r2 | 3.08 | 3.08 | 9.48e-06 | NOT honest |  |
| v13 | g-1e-8-2-34841 | 1e-8 | FCC r1 and r2 | 0.309 | 0.309 | 1.56e-05 | honest |  |
| v14 | g-1e-7-3-13816 | 1e-7 | four readings | 0.001021 | 0.001021 | 2.77e-05 | honest |  |
| v15 | g-1e-7-0-1973 | 1e-7 | four readings | 0.002042 | 0.002042 | 0.00135 | honest |  |
| v16 | g-1e-7-2-2092 | 1e-7 | FCC r1 and r2 | 0.03819 | 0.03819 | 0.000276 | honest |  |
| v17 | g-1e-7-0-1772 | 1e-7 | four readings | 2.382 | 2.382 | 3.75e-05 | NOT honest |  |
| v18 | g-1e-7-2-13649 | 1e-7 | FCC r1 and r2 | 0.5289 | 0.5289 | 7.53e-06 | honest |  |
| v19 | g-1e-8-3-1921 | 1e-8 | four readings | 0.04621 | 0.04621 | 0.00051 | honest |  |
| v20 | g-1e-8-0-1896 | 1e-8 | four readings | 0.01414 | 0.01414 | 0.000819 | honest |  |
| v21 | g-1e-8-1-1747 | 1e-8 | four readings | 0.008619 | 0.008619 | 0.00085 | honest |  |
| v22 | g-1e-8-3-14096 | 1e-8 | FCC r1 and r2 | 2.53 | 2.53 | 2.13e-05 | NOT honest |  |
| v23 | g-1e-8-1-34772 | 1e-8 | FCC r1 and r2 | 0.7671 | 0.7671 | 0.000104 | honest |  |
| z00 | g-1e-7-1-2174 | 1e-7 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1', 'qawo_r2'] | 0.7181 | 0.7181 | 0.00047 | honest |  |
| z12 | g-1e-8-1-2174 | 1e-8 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1', 'qawo_r2'] | 0.7181 | 0.7181 | 0.00047 | honest |  |
| z07 | g-1e-7-2-34718 | 1e-7 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 1.525e-06 | 1.525e-06 | 1.57e-06 | honest |  |
| z16 | g-1e-8-2-34708 | 1e-8 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1'] | 6.543e-06 | 6.543e-06 | 0.000114 | honest |  |
| z18 | g-1e-8-1-2368 | 1e-8 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 7.388e-07 | 7.388e-07 | 1.69e-06 | honest |  |
| z21 | g-1e-8-1-14572 | 1e-8 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 3.482e-07 | 3.482e-07 | 1.34e-06 | honest |  |
| z24 | g-1e-8-2-34718 | 1e-8 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 0.002074 | 0.002074 | 0.00025 | honest |  |
| z28 | g-1e-7-2-34708 | 1e-7 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1'] | 6.543e-06 | 6.543e-06 | 0.000114 | honest |  |
| z29 | g-1e-7-1-14572 | 1e-7 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 2.342e-07 | 2.342e-07 | 3.4e-07 | honest |  |

The before (07's census): honest on 28 of 33 scored calls.

Beside 07's own scores (07's v4_scores.json): value and abserr at h equal 07's (hex), 07's verdict and ratio:

- v00: value equal True, abserr equal True; 07: honest, 0.01967
- v01: value equal True, abserr equal True; 07: honest, 0.03252
- v02: value equal True, abserr equal True; 07: honest, 0.005717
- v03: value equal True, abserr equal True; 07: honest, 0.001751
- v04: value equal True, abserr equal True; 07: honest, 0.1377
- v05: value equal True, abserr equal True; 07: honest, 0.001238
- v06: value equal True, abserr equal True; 07: honest, 0.0002783
- v07: value equal True, abserr equal True; 07: honest, 0.0004815
- v08: value equal True, abserr equal True; 07: NOT honest, 3.16
- v09: value equal True, abserr equal True; 07: NOT honest, 2.211
- v10: value equal True, abserr equal True; 07: honest, 0.1924
- v11: value equal True, abserr equal True; 07: honest, 0.007617
- v12: value equal True, abserr equal True; 07: NOT honest, 3.08
- v13: value equal True, abserr equal True; 07: honest, 0.309
- v14: value equal True, abserr equal True; 07: honest, 0.001021
- v15: value equal True, abserr equal True; 07: honest, 0.002042
- v16: value equal True, abserr equal True; 07: honest, 0.03819
- v17: value equal True, abserr equal True; 07: NOT honest, 2.382
- v18: value equal True, abserr equal True; 07: honest, 0.5289
- v19: value equal True, abserr equal True; 07: honest, 0.04621
- v20: value equal True, abserr equal True; 07: honest, 0.01414
- v21: value equal True, abserr equal True; 07: honest, 0.008619
- v22: value equal True, abserr equal True; 07: NOT honest, 2.53
- v23: value equal True, abserr equal True; 07: honest, 0.7671
- z00: value equal True, abserr equal True; 07: honest, 0.7181
- z12: value equal True, abserr equal True; 07: honest, 0.7181
- z07: value equal True, abserr equal True; 07: honest, 1.525e-06
- z16: value equal True, abserr equal True; 07: honest, 6.543e-06
- z18: value equal True, abserr equal True; 07: honest, 7.388e-07
- z21: value equal True, abserr equal True; 07: honest, 3.482e-07
- z24: value equal True, abserr equal True; 07: honest, 0.002074
- z28: value equal True, abserr equal True; 07: honest, 6.543e-06
- z29: value equal True, abserr equal True; 07: honest, 2.342e-07
