# levin-phase-floor 07d Part A': the projection (section 9.2)
# written 2026-11-23T02:19:06Z on mps026298.phys.susx.ac.uk
# script projection_a2.py sha256 748e3f952bdee8df77fd324cbe051deeeb3e020080a3159daede3b7be919f495
# tree: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-c2c57a3 HEAD c2c57a36cbad3780986baceb807dcb447975b498 (status --short: empty)
# main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2/out/projection/probe_v4_cost.json sha256 32027ce078864ba29fde7e5f8c37b0fa22ad231def07723aa09b9e921bc84adb
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/out/leaves-v3_results.json sha256 571e11585babaeef600816f903d7e31c3345fb9c3adff439cf096a1eb7d20e23
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/out/variants_results.json sha256 9f69124229dcdf2a1e950851b636764513fd95d3af809a91817b8bcb9628a4ac
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/out/refs_results.json sha256 fc6af0961662c10ac6151d83a02bd2172de11263c846e3645deb33ece0f83e61

| item | elapsed (s) |
|---|---|
| instrumented run at 6 pi (4,220 calls), with the new band's population | 351 |
| item 0: v4 on Part A's D-E and D-T leaves (2873 distinct of 2873 drawn), central | 864 |
| item 0: the same at the probe's largest evaluations a piece | 3,053 |
| item 3: the new band, 4 strata x 80 leaves, v4 at the largest | 455 |
| items 1, 2, 4, 5 on every leaf | 4,398 |
| item 6: eight variants x 4,220 calls | 8,436 |
| new call references: at most 30 readings, two at a time | 12,869 |

central 27,373 s; with the v4 high case 29,561 s; x1.5 margin: 11.4 h central, 12.3 h high, against the 36 h limit
model: {"c_eval_s": 5.775000072068631e-05, "evals_per_piece_median": 1649.5467999231214, "evals_per_piece_max": 12096.0, "t_sigma_s": 1.0836790408939123, "leafcc_cpu_s_per_leaf": 1.0331518119774834, "items_factor": 8.0, "tree_replay_wall_s": 292.1000251333222, "ref_reading_max_s": 857.944149017334, "margin": 1.5, "workers": 6}
knots on the distinct Part A leaves: {'0': 1026, '1-4': 703, '31-100': 152, '5-30': 480, '>100': 15, 'unknown': 497}
