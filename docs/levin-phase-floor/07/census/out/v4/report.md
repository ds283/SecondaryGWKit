# levin-phase-floor 07: V4 scored
# written 2026-11-10T02:33:42Z on mps026298.phys.susx.ac.uk
# script score_v4_07.py sha256 c976be1554809ebafb127c2407e97f37226b1bd60420dea940cd01cefbd72af9
# tree g: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07-e182bc7 HEAD e182bc7a827142cd4f42594f1a9d8cf85ccc3a5f (status --short: empty)
# tree p: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07-91472db HEAD 91472dbee1daba8374b0018db1195cf6c7898bbb (status --short: empty)
# main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/census/state/plan_references.json sha256 93506ccbee960f79741fedac6ad5679df97274eaf762a9a08a6c8bfd169d78fe
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/census/state/v4_reading_list_20261109.json sha256 ce594095f08e0010a955fcd8c4a7027acd905662350850616ddfe9217a9f6e82
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/census/out/v4/inputs.json sha256 55b855d4f71feba93f4a9d578530251d38d17b11c63267cee2241f109a222b11
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/census/state/v4_selection.json sha256 15a4391b8e49c547a0557bc20829faf5c2323f0e826420ef0e21d694b134d6c8
# input /Users/ds283/Documents/Code/SecondaryGWKit-lpf07-e182bc7/docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json sha256 73c18680d266b1a0fdf11123c104d20e822658047dd7c36e1c814db9ffcad912
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07-census-20261108T171137/checkpoint.jsonl sha256 03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10

# V4 scored

V4: FAILS: 28 of 33 scored calls honest; not honest ['v08', 'v09', 'v12', 'v17', 'v22']; reported, not scored [].
Totals: {"24": {"NOT honest": 5, "honest": 19}, "z": {"honest": 9}, "all": {"honest": 28, "NOT honest": 5}, "basis_24": {"FCC r1 and r2": 11, "four readings": 13}}

| call | unit | rtol | basis | |v_g - ref|/abserr_g | |v_06m - ref|/abserr_06m | spread/tol | verdict | notes |
|---|---|---|---|---|---|---|---|---|
| v00 | g-1e-7-3-37343 | 1e-7 | four readings | 0.01967 | 0.0005483 | 0.00117 | honest |  |
| v01 | g-1e-7-2-34834 | 1e-7 | FCC r1 and r2 | 0.03252 | 0.02342 | 0.00118 | honest |  |
| v02 | g-1e-8-3-39137 | 1e-8 | four readings | 0.005717 | 0.009684 | 9.98e-05 | honest |  |
| v03 | g-1e-8-1-34611 | 1e-8 | four readings | 0.001751 | 0.032 | 0.000614 | honest |  |
| v04 | g-1e-7-3-2326 | 1e-7 | four readings | 0.1377 | 2.323e+05 | 1.64e-05 | honest |  |
| v05 | g-1e-7-0-38985 | 1e-7 | four readings | 0.001238 | 355.8 | 0.000126 | honest |  |
| v06 | g-1e-7-1-2120 | 1e-7 | FCC r1 and r2 | 0.0002783 | 0.1271 | 0.000319 | honest |  |
| v07 | g-1e-7-3-14592 | 1e-7 | FCC r1 and r2 | 0.0004815 | 1.238e+04 | 8.86e-06 | honest |  |
| v08 | g-1e-7-1-38834 | 1e-7 | FCC r1 and r2 | 3.16 | 3.043e+04 | 0.00388 | NOT honest |  |
| v09 | g-1e-8-3-34962 | 1e-8 | four readings | 2.211 | 638.6 | 0.00053 | NOT honest |  |
| v10 | g-1e-8-0-14314 | 1e-8 | four readings | 0.1924 | 1295 | 0.000209 | honest |  |
| v11 | g-1e-8-2-2189 | 1e-8 | FCC r1 and r2 | 0.007617 | 0.1257 | 0.000303 | honest |  |
| v12 | g-1e-8-0-14299 | 1e-8 | FCC r1 and r2 | 3.08 | 9088 | 0.000227 | NOT honest |  |
| v13 | g-1e-8-2-34841 | 1e-8 | FCC r1 and r2 | 0.309 | 1.845e+04 | 0.00187 | honest |  |
| v14 | g-1e-7-3-13816 | 1e-7 | four readings | 0.001021 | 1.353e+04 | 2.77e-05 | honest |  |
| v15 | g-1e-7-0-1973 | 1e-7 | four readings | 0.002042 | 5.189e+05 | 0.00137 | honest |  |
| v16 | g-1e-7-2-2092 | 1e-7 | FCC r1 and r2 | 0.03819 | 0.04411 | 0.00116 | honest |  |
| v17 | g-1e-7-0-1772 | 1e-7 | four readings | 2.382 | 497.4 | 3.75e-05 | NOT honest |  |
| v18 | g-1e-7-2-13649 | 1e-7 | FCC r1 and r2 | 0.5289 | 1.608e+04 | 0.000744 | honest |  |
| v19 | g-1e-8-3-1921 | 1e-8 | four readings | 0.04621 | 1.927e+05 | 0.00051 | honest |  |
| v20 | g-1e-8-0-1896 | 1e-8 | four readings | 0.01414 | 1.604e+05 | 0.0012 | honest |  |
| v21 | g-1e-8-1-1747 | 1e-8 | four readings | 0.008619 | 0.002056 | 0.000962 | honest |  |
| v22 | g-1e-8-3-14096 | 1e-8 | FCC r1 and r2 | 2.53 | 1.671e+04 | 0.000152 | NOT honest |  |
| v23 | g-1e-8-1-34772 | 1e-8 | FCC r1 and r2 | 0.7671 | 1.227e-05 | 0.000104 | honest |  |
| z00 | g-1e-7-1-2174 | 1e-7 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1', 'qawo_r2'] | 0.7181 | 0.01828 | 0.00047 | honest |  |
| z12 | g-1e-8-1-2174 | 1e-8 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1', 'qawo_r2'] | 0.7181 | 1681 | 0.00047 | honest |  |
| z07 | g-1e-7-2-34718 | 1e-7 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 1.525e-06 | 1.525e-06 | 1.57e-06 | honest |  |
| z16 | g-1e-8-2-34708 | 1e-8 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1'] | 6.543e-06 | 6.543e-06 | 0.000114 | honest |  |
| z18 | g-1e-8-1-2368 | 1e-8 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 7.388e-07 | 7.388e-07 | 1.69e-06 | honest |  |
| z21 | g-1e-8-1-14572 | 1e-8 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 3.482e-07 | 3.482e-07 | 1.34e-06 | honest |  |
| z24 | g-1e-8-2-34718 | 1e-8 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 0.002074 | 0.002074 | 0.00025 | honest |  |
| z28 | g-1e-7-2-34708 | 1e-7 | 06g's readings: ['fcc_r1', 'fcc_r2', 'qawo_r1'] | 6.543e-06 | 6.543e-06 | 0.000114 | honest |  |
| z29 | g-1e-7-1-14572 | 1e-7 | 06g's readings: ['fcc_r1', 'fcc_r2'] | 2.342e-07 | 2.342e-07 | 3.4e-07 | honest |  |

The stationary-point calls before: 06m honest on 17 of 33 scored calls.
