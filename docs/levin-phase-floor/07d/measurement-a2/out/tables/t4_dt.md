# levin-phase-floor 07d Part A': t4_dt
# written 2026-11-23T02:58:38Z on mps026298.phys.susx.ac.uk
# script tables_a2.py sha256 3e6f64886e10dbc494a0a1d57166ff7343aa8122a5fc1ce952a892df9b78ecb8
# tree: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-c2c57a3 HEAD c2c57a36cbad3780986baceb807dcb447975b498 (status --short: empty)
# main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/state/candidates_07d.json sha256 d73b208e76d77043a05605a9f44de15af31337f34960b6dd629ce92dd2718bee
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/out/tables/d_e.json sha256 c8269db1a296d2a895eccd0f42e47b1ce33563cf4981324dd7ba2a93fa23a4eb
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/out/tables/d_t.json sha256 379d2b585b347492bfeb1c9c9c0fb9405b3b80cdad16e19efea708b352d09404
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2/state/rules_tables_a2.md sha256 ab600807938b4545d7cd295d52c3bfa91af59384b6fb347b73cda07216850330

Per T, call kind and pairing (Levin side's fallback, CC side's declaration); returned leaves only (band_20_32v, visited not returned, is in the rows, never pooled here). Each side: verdict, scored, understates.

| T/pi | kind | pairing | Levin side (V >= T) | CC side (V < T) |
|---|---|---|---|---|
| 4 | group | tree | fails (718, 3) | fails (164, 34) |
| 4 | group | nest3N4 on CC_47 | fails (718, 3) | bounded (525, 0) |
| 4 | group | CC_93 nest3N4 | fails (710, 2) | bounded (525, 0) |
| 4 | group | CC_93 pair93 | fails (710, 2) | bounded (523, 0) |
| 4 | group | CC_93 chop93 | fails (710, 2) | bounded (526, 0) |
| 4 | group | CC_93 nest93 | fails (710, 2) | bounded (523, 0) |
| 4 | group | CC_93 tail93 | fails (710, 2) | bounded (523, 0) |
| 4 | analytic | tree | bounded (210, 0) | bounded (279, 0) |
| 4 | analytic | nest3N4 on CC_47 | bounded (210, 0) | bounded (279, 0) |
| 4 | analytic | CC_93 nest3N4 | bounded (235, 0) | bounded (279, 0) |
| 4 | analytic | CC_93 pair93 | bounded (235, 0) | bounded (279, 0) |
| 4 | analytic | CC_93 chop93 | bounded (235, 0) | bounded (279, 0) |
| 4 | analytic | CC_93 nest93 | bounded (235, 0) | bounded (279, 0) |
| 4 | analytic | CC_93 tail93 | bounded (235, 0) | bounded (279, 0) |
| 6 | group | tree | fails (719, 3) | fails (239, 47) |
| 6 | group | nest3N4 on CC_47 | fails (719, 3) | bounded (802, 0) |
| 6 | group | CC_93 nest3N4 | fails (700, 2) | bounded (801, 0) |
| 6 | group | CC_93 pair93 | fails (700, 2) | bounded (800, 0) |
| 6 | group | CC_93 chop93 | fails (700, 2) | bounded (803, 0) |
| 6 | group | CC_93 nest93 | fails (700, 2) | bounded (800, 0) |
| 6 | group | CC_93 tail93 | fails (700, 2) | bounded (800, 0) |
| 6 | analytic | tree | bounded (160, 0) | bounded (379, 0) |
| 6 | analytic | nest3N4 on CC_47 | bounded (160, 0) | bounded (379, 0) |
| 6 | analytic | CC_93 nest3N4 | bounded (155, 0) | bounded (379, 0) |
| 6 | analytic | CC_93 pair93 | bounded (155, 0) | bounded (379, 0) |
| 6 | analytic | CC_93 chop93 | bounded (155, 0) | bounded (379, 0) |
| 6 | analytic | CC_93 nest93 | bounded (155, 0) | bounded (379, 0) |
| 6 | analytic | CC_93 tail93 | bounded (155, 0) | bounded (379, 0) |
| 8 | group | tree | bounded (598, 0) | fails (359, 52) |
| 8 | group | nest3N4 on CC_47 | bounded (598, 0) | bounded (922, 0) |
| 8 | group | CC_93 nest3N4 | bounded (587, 0) | bounded (921, 0) |
| 8 | group | CC_93 pair93 | bounded (587, 0) | bounded (920, 0) |
| 8 | group | CC_93 chop93 | bounded (587, 0) | bounded (923, 0) |
| 8 | group | CC_93 nest93 | bounded (587, 0) | bounded (920, 0) |
| 8 | group | CC_93 tail93 | bounded (587, 0) | bounded (920, 0) |
| 8 | analytic | tree | bounded (158, 0) | bounded (471, 0) |
| 8 | analytic | nest3N4 on CC_47 | bounded (158, 0) | bounded (471, 0) |
| 8 | analytic | CC_93 nest3N4 | bounded (136, 0) | bounded (471, 0) |
| 8 | analytic | CC_93 pair93 | bounded (136, 0) | bounded (471, 0) |
| 8 | analytic | CC_93 chop93 | bounded (136, 0) | bounded (471, 0) |
| 8 | analytic | CC_93 nest93 | bounded (136, 0) | bounded (471, 0) |
| 8 | analytic | CC_93 tail93 | bounded (136, 0) | bounded (471, 0) |
| 10 | group | tree | bounded (481, 0) | fails (479, 52) |
| 10 | group | nest3N4 on CC_47 | bounded (481, 0) | bounded (1042, 0) |
| 10 | group | CC_93 nest3N4 | bounded (474, 0) | bounded (1041, 0) |
| 10 | group | CC_93 pair93 | bounded (474, 0) | bounded (1040, 0) |
| 10 | group | CC_93 chop93 | bounded (474, 0) | bounded (1043, 0) |
| 10 | group | CC_93 nest93 | bounded (474, 0) | bounded (1040, 0) |
| 10 | group | CC_93 tail93 | bounded (474, 0) | bounded (1040, 0) |
| 10 | analytic | tree | bounded (126, 0) | bounded (575, 0) |
| 10 | analytic | nest3N4 on CC_47 | bounded (126, 0) | bounded (575, 0) |
| 10 | analytic | CC_93 nest3N4 | bounded (118, 0) | bounded (575, 0) |
| 10 | analytic | CC_93 pair93 | bounded (118, 0) | bounded (575, 0) |
| 10 | analytic | CC_93 chop93 | bounded (118, 0) | bounded (575, 0) |
| 10 | analytic | CC_93 nest93 | bounded (118, 0) | bounded (575, 0) |
| 10 | analytic | CC_93 tail93 | bounded (118, 0) | bounded (575, 0) |
| 12 | group | tree | bounded (375, 0) | fails (599, 52) |
| 12 | group | nest3N4 on CC_47 | bounded (375, 0) | bounded (1162, 0) |
| 12 | group | CC_93 nest3N4 | bounded (380, 0) | bounded (1161, 0) |
| 12 | group | CC_93 pair93 | bounded (380, 0) | bounded (1160, 0) |
| 12 | group | CC_93 chop93 | bounded (380, 0) | bounded (1163, 0) |
| 12 | group | CC_93 nest93 | bounded (380, 0) | bounded (1160, 0) |
| 12 | group | CC_93 tail93 | bounded (380, 0) | bounded (1160, 0) |
| 12 | analytic | tree | bounded (86, 0) | bounded (675, 0) |
| 12 | analytic | nest3N4 on CC_47 | bounded (86, 0) | bounded (675, 0) |
| 12 | analytic | CC_93 nest3N4 | bounded (88, 0) | bounded (675, 0) |
| 12 | analytic | CC_93 pair93 | bounded (88, 0) | bounded (675, 0) |
| 12 | analytic | CC_93 chop93 | bounded (88, 0) | bounded (675, 0) |
| 12 | analytic | CC_93 nest93 | bounded (88, 0) | bounded (675, 0) |
| 12 | analytic | CC_93 tail93 | bounded (88, 0) | bounded (675, 0) |
| 16 | group | tree | bounded (244, 0) | fails (759, 52) |
| 16 | group | nest3N4 on CC_47 | bounded (244, 0) | bounded (1322, 0) |
| 16 | group | CC_93 nest3N4 | bounded (254, 0) | bounded (1321, 0) |
| 16 | group | CC_93 pair93 | bounded (254, 0) | bounded (1320, 0) |
| 16 | group | CC_93 chop93 | bounded (254, 0) | bounded (1323, 0) |
| 16 | group | CC_93 nest93 | bounded (254, 0) | bounded (1320, 0) |
| 16 | group | CC_93 tail93 | bounded (254, 0) | bounded (1320, 0) |
| 16 | analytic | tree | bounded (68, 0) | bounded (779, 0) |
| 16 | analytic | nest3N4 on CC_47 | bounded (68, 0) | bounded (779, 0) |
| 16 | analytic | CC_93 nest3N4 | fails (78, 1) | bounded (779, 0) |
| 16 | analytic | CC_93 pair93 | fails (78, 1) | bounded (779, 0) |
| 16 | analytic | CC_93 chop93 | fails (78, 1) | bounded (779, 0) |
| 16 | analytic | CC_93 nest93 | fails (78, 1) | bounded (779, 0) |
| 16 | analytic | CC_93 tail93 | fails (78, 1) | bounded (779, 0) |
| 20 | group | tree | bounded (124, 0) | fails (916, 52) |
| 20 | group | nest3N4 on CC_47 | bounded (124, 0) | bounded (1479, 0) |
| 20 | group | CC_93 nest3N4 | bounded (133, 0) | bounded (1478, 0) |
| 20 | group | CC_93 pair93 | bounded (133, 0) | bounded (1476, 0) |
| 20 | group | CC_93 chop93 | bounded (133, 0) | bounded (1480, 0) |
| 20 | group | CC_93 nest93 | bounded (133, 0) | bounded (1477, 0) |
| 20 | group | CC_93 tail93 | bounded (133, 0) | bounded (1477, 0) |
| 20 | analytic | tree | bounded (17, 0) | bounded (877, 0) |
| 20 | analytic | nest3N4 on CC_47 | bounded (17, 0) | bounded (877, 0) |
| 20 | analytic | CC_93 nest3N4 | fails (41, 1) | bounded (877, 0) |
| 20 | analytic | CC_93 pair93 | fails (41, 1) | bounded (877, 0) |
| 20 | analytic | CC_93 chop93 | fails (41, 1) | bounded (877, 0) |
| 20 | analytic | CC_93 nest93 | fails (41, 1) | bounded (877, 0) |
| 20 | analytic | CC_93 tail93 | fails (41, 1) | bounded (877, 0) |

### Rows by T, route, declaration, kind, family and knots

| T/pi | route | decl/fallback | kind | family | knots | scored | bounded | understates | undecided | untrusted | own CC/guarded | D/te min, median, max | te/request median | te<=request | Levin own decl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | cc | cc47:nest3N4 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.01e+03, 2.42e+03, 5.67e+05 | 7.38e-06 | 5 |  |
| 10 | cc | cc47:nest3N4 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.45e+04, 3.36e+05, 2.3e+06 | 7.45e-07 | 5 |  |
| 10 | cc | cc47:nest3N4 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 1.13e+03, 9.39e+04, 2.02e+07 | 5.8e-07 | 43 |  |
| 10 | cc | cc47:nest3N4 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 160, 3.03e+05, 1.23e+07 | 1.39e-06 | 39 |  |
| 10 | cc | cc47:nest3N4 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 9.63e+04, 4.02e+06, 1.58e+09 | 6.64e-06 | 13 |  |
| 10 | cc | cc47:nest3N4 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 1.61e+06, 1.02e+07, 4.5e+08 | 2.22e-05 | 6 |  |
| 10 | cc | cc47:nest3N4 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 3.47e+05, 2e+07, 5.8e+08 | 8.13e-06 | 38 |  |
| 10 | cc | cc47:nest3N4 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 1.37e+05, 3.41e+07, 9.11e+08 | 3.87e-06 | 47 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.72e+03, 1.53e+04 | 3.1e-07 | 4 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 34.8, 229 | 1.48e-07 | 63 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.38, 58.3, 3.97e+03 | 7.36e-07 | 33 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 375, 2.18e+03, 6.69e+04 | 4.89e-07 | 4 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.89, 31.4, 554 | 1.69e-07 | 51 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 7.33, 61.1, 1.44e+04 | 8.34e-07 | 39 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 612, 3.48e+03, 7.52e+03 | 1.3e-06 | 13 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.03e+03, 3.01e+03 | 1.97e-05 | 4 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.63, 54.3, 1.87e+05 | 5.13e-07 | 32 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.42, 414, 1.12e+04 | 6.24e-07 | 49 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 473, 2.44e+03, 5.95e+03 | 2.8e-06 | 12 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 739, 739, 739 | 2.96e-05 | 2 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12, 266, 911 | 3.5e-07 | 36 |  |
| 10 | cc | cc47:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 15.1, 172, 1.5e+03 | 1.52e-06 | 37 |  |
| 10 | cc | cc47:nest3N4 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.43, 6.64, 2.16e+03 | 1.62e-06 | 32 |  |
| 10 | cc | cc47:nest3N4 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 1.69, 140, 2.43e+03 | 0.00104 | 80 |  |
| 10 | cc | cc47:nest3N4 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 2.1, 5.49, 47.1 | 1.3e-06 | 7 |  |
| 10 | cc | cc47:nest3N4 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 2.16, 38.3, 1.36e+03 | 0.000232 | 26 |  |
| 10 | cc | cc47:nest3N4 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 3.49, 178, 1.01e+04 | 0.00127 | 89 |  |
| 10 | cc | cc47:nest3N4 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.54, 3.54, 6.23 | 5.27e-05 | 3 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.3, 10.2, 2.03e+03 | 0.00096 | 37 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 1.57, 37.9, 934 | 0.041 | 40 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.26, 1.59, 123 | 4.84e-05 | 106 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 2.03, 6.75, 76.1 | 0.000109 | 55 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.6, 34.1, 202 | 0.0339 | 31 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.73, 4.74, 108 | 3.35e-05 | 127 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.95, 5.23, 221 | 3.86e-06 | 65 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 1.44, 41.3, 356 | 0.026 | 47 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.92, 3.94, 650 | 8.03e-05 | 78 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.51, 3.54, 51.4 | 7.31e-05 | 79 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 2.38, 47.4, 796 | 0.00112 | 26 |  |
| 10 | cc | cc47:nest3N4 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 2.03, 3.52, 217 | 2.63e-05 | 94 |  |
| 10 | cc | cc47:tree | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.66e+06, 7.6e+06, 2.9e+08 | 7.38e-06 | 5 |  |
| 10 | cc | cc47:tree | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 7.56e+07, 1.05e+08, 2.71e+08 | 7.45e-07 | 5 |  |
| 10 | cc | cc47:tree | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 4.33e+06, 7.99e+07, 2.59e+10 | 5.8e-07 | 43 |  |
| 10 | cc | cc47:tree | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 4.23e+05, 8.25e+07, 2.47e+09 | 1.39e-06 | 39 |  |
| 10 | cc | cc47:tree | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 2.31e+06, 1.97e+08, 5.41e+10 | 6.64e-06 | 13 |  |
| 10 | cc | cc47:tree | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 9.02e+06, 7.26e+07, 9.77e+09 | 2.22e-05 | 6 |  |
| 10 | cc | cc47:tree | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 6.38e+07, 2.82e+08, 2.38e+10 | 8.13e-06 | 38 |  |
| 10 | cc | cc47:tree | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 1.21e+07, 4.89e+08, 3e+10 | 3.87e-06 | 47 |  |
| 10 | cc | cc47:tree | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.69e+03, 1.52e+04 | 3.1e-07 | 4 |  |
| 10 | cc | cc47:tree | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 5.46, 26.7, 177 | 1.48e-07 | 63 |  |
| 10 | cc | cc47:tree | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 2.63, 45.5, 3.96e+03 | 7.36e-07 | 33 |  |
| 10 | cc | cc47:tree | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 373, 2.17e+03, 6.68e+04 | 4.89e-07 | 4 |  |
| 10 | cc | cc47:tree | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 6.96, 28, 3.93e+04 | 1.69e-07 | 51 |  |
| 10 | cc | cc47:tree | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 5.85, 48.7, 1.43e+04 | 8.34e-07 | 39 |  |
| 10 | cc | cc47:tree | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 609, 3.46e+03, 1.37e+05 | 1.3e-06 | 13 |  |
| 10 | cc | cc47:tree | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.88e+04, 6.06e+04 | 1.97e-05 | 4 |  |
| 10 | cc | cc47:tree | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.05, 2.41e+04, 2.29e+07 | 5.13e-07 | 32 |  |
| 10 | cc | cc47:tree | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.01, 2.69e+03, 1.12e+07 | 6.24e-07 | 49 |  |
| 10 | cc | cc47:tree | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 5.29e+04, 3.41e+05, 5.87e+05 | 2.8e-06 | 12 |  |
| 10 | cc | cc47:tree | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 8.23e+04, 8.23e+04, 8.23e+04 | 2.96e-05 | 2 |  |
| 10 | cc | cc47:tree | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 1.45e+04, 3.51e+05, 6.87e+06 | 3.5e-07 | 36 |  |
| 10 | cc | cc47:tree | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 7.32e+03, 7.41e+05, 1.21e+07 | 1.52e-06 | 37 |  |
| 10 | cc | cc47:tree | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 10.5, 938, 2.59e+04 | 1.62e-06 | 32 |  |
| 10 | cc | cc47:tree | group | band_6_8 | 5+ | 81 | 76 | 5 | 0 | 0 | 0/0 | 0.0385, 62.5, 2.64e+04 | 0.00104 | 80 |  |
| 10 | cc | cc47:tree | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 5.24, 102, 1.84e+03 | 1.3e-06 | 7 |  |
| 10 | cc | cc47:tree | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 264, 6.34e+04, 3.12e+06 | 0.000232 | 26 |  |
| 10 | cc | cc47:tree | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 37.3, 1.89e+03, 6.48e+06 | 0.00127 | 89 |  |
| 10 | cc | cc47:tree | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 4.18e+03, 4.18e+03, 9.48e+04 | 5.27e-05 | 3 |  |
| 10 | cc | cc47:tree | group | gate_q1 | 1-4 | 21 | 21 | 0 | 17 | 0 | 0/0 | 1.08, 9.02, 629 | 0.00261 | 20 |  |
| 10 | cc | cc47:tree | group | gate_q1 | 5+ | 45 | 27 | 18 | 0 | 0 | 0/0 | 0.0613, 4.6, 286 | 0.041 | 40 |  |
| 10 | cc | cc47:tree | group | gate_q1 | none | 7 | 7 | 0 | 107 | 0 | 0/0 | 1.28, 2.62, 20.2 | 3.56e-05 | 7 |  |
| 10 | cc | cc47:tree | group | gate_q2 | 1-4 | 10 | 9 | 1 | 45 | 0 | 0/0 | 0.776, 13.6, 26 | 0.0269 | 10 |  |
| 10 | cc | cc47:tree | group | gate_q2 | 5+ | 32 | 25 | 7 | 0 | 0 | 0/0 | 0.0137, 4.62, 47 | 0.0339 | 31 |  |
| 10 | cc | cc47:tree | group | gate_q2 | none | 11 | 11 | 0 | 116 | 0 | 0/0 | 1.84, 2.49, 6.78 | 4.24e-05 | 11 |  |
| 10 | cc | cc47:tree | group | gate_q3 | 1-4 | 10 | 10 | 0 | 55 | 0 | 0/0 | 4.22, 18.8, 67.3 | 0.0826 | 10 |  |
| 10 | cc | cc47:tree | group | gate_q3 | 5+ | 49 | 32 | 17 | 0 | 0 | 0/0 | 0.00255, 5.37, 101 | 0.026 | 47 |  |
| 10 | cc | cc47:tree | group | gate_q3 | none | 4 | 4 | 0 | 74 | 0 | 0/0 | 1.52, 2.02, 2.51 | 0.000895 | 4 |  |
| 10 | cc | cc47:tree | group | gate_q4 | 1-4 | 16 | 16 | 0 | 63 | 0 | 0/0 | 1.52, 4.29, 36.9 | 0.000454 | 16 |  |
| 10 | cc | cc47:tree | group | gate_q4 | 5+ | 27 | 23 | 4 | 0 | 0 | 0/0 | 0.171, 7.07, 392 | 0.00112 | 26 |  |
| 10 | cc | cc47:tree | group | gate_q4 | none | 7 | 7 | 0 | 87 | 0 | 0/0 | 1.7, 4.66, 13.7 | 5.82e-06 | 7 |  |
| 10 | cc | cc93:chop93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 10 | cc | cc93:chop93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 10 | cc | cc93:chop93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 38.7, 551, 2.16e+04 | 3.59e-07 | 43 |  |
| 10 | cc | cc93:chop93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 56.5, 472, 9.43e+03 | 7.91e-07 | 39 |  |
| 10 | cc | cc93:chop93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 10 | cc | cc93:chop93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 10 | cc | cc93:chop93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 26, 121, 626 | 1.79e-06 | 38 |  |
| 10 | cc | cc93:chop93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 29.9, 714, 6.21e+03 | 1.11e-06 | 47 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.59e+04, 1.59e+04, 1.59e+04 | 2.98e-07 | 4 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 154, 3.22e+03, 7.35e+06 | 2.23e-07 | 63 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 50.6, 2.92e+03, 3.51e+05 | 9.31e-07 | 33 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.73e+03, 3.49e+03 | 2.39e-06 | 4 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 108, 4.99e+03, 1.75e+05 | 9.35e-08 | 51 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 639, 4.44e+03, 2.32e+05 | 4.37e-07 | 39 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 1.01e+03, 6.03e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 71.1, 557, 1.84e+04 | 3.11e-07 | 32 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 87.4, 2.16e+03, 1.95e+04 | 6.33e-07 | 49 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 78.3, 308, 3.26e+03 | 4.09e-07 | 36 |  |
| 10 | cc | cc93:chop93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 95.2, 620, 6.89e+03 | 1.27e-06 | 37 |  |
| 10 | cc | cc93:chop93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.44, 450, 3.99e+04 | 2.6e-06 | 32 |  |
| 10 | cc | cc93:chop93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 5.35, 101, 7.94e+05 | 6.24e-05 | 81 |  |
| 10 | cc | cc93:chop93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 184, 1.02e+03, 1.77e+03 | 6.17e-06 | 7 |  |
| 10 | cc | cc93:chop93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 4.01, 547, 3.83e+04 | 0.000124 | 26 |  |
| 10 | cc | cc93:chop93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 6.3, 84.1, 1.17e+03 | 0.00013 | 91 |  |
| 10 | cc | cc93:chop93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 123, 123, 1.51e+03 | 4.35e-05 | 3 |  |
| 10 | cc | cc93:chop93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 4.29, 1.53e+03, 1.88e+08 | 0.0004 | 38 |  |
| 10 | cc | cc93:chop93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 1.09e+07 | 0.00275 | 45 |  |
| 10 | cc | cc93:chop93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 382, 1.28e+04, 3.83e+07 | 4.52e-05 | 107 |  |
| 10 | cc | cc93:chop93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 63.2, 4.34e+03, 2.08e+05 | 0.000135 | 55 |  |
| 10 | cc | cc93:chop93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 6.59e+04 | 0.00128 | 32 |  |
| 10 | cc | cc93:chop93 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 85.2, 7.81e+03, 8.74e+05 | 1.28e-05 | 127 |  |
| 10 | cc | cc93:chop93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 32, 1.71e+03, 1.29e+05 | 2.16e-06 | 65 |  |
| 10 | cc | cc93:chop93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 10 | cc | cc93:chop93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 280, 2.36e+03, 2.39e+05 | 6.44e-05 | 78 |  |
| 10 | cc | cc93:chop93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 26.8, 1.66e+03, 4.34e+04 | 9.55e-05 | 79 |  |
| 10 | cc | cc93:chop93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 21.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 10 | cc | cc93:chop93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 65.5, 1.24e+03, 2.37e+05 | 3.11e-05 | 93 |  |
| 10 | cc | cc93:nest3N4 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.51e+03, 5.22e+03, 5.94e+05 | 7.36e-06 | 5 |  |
| 10 | cc | cc93:nest3N4 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 4.41e+04, 4.9e+05, 5.74e+06 | 5.11e-07 | 5 |  |
| 10 | cc | cc93:nest3N4 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 1.41e+03, 1.06e+05, 2.98e+07 | 3.59e-07 | 43 |  |
| 10 | cc | cc93:nest3N4 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 541, 5.09e+05, 4.24e+07 | 7.91e-07 | 39 |  |
| 10 | cc | cc93:nest3N4 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 3.84e+05, 1.04e+07, 1.33e+09 | 3.57e-06 | 13 |  |
| 10 | cc | cc93:nest3N4 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 1.95e+06, 4.97e+08, 5.16e+09 | 2.19e-06 | 6 |  |
| 10 | cc | cc93:nest3N4 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 6.83e+05, 1.72e+07, 5.62e+09 | 1.79e-06 | 38 |  |
| 10 | cc | cc93:nest3N4 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 8.3e+04, 6.88e+07, 4.75e+09 | 1.11e-06 | 47 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 50.6, 229 | 2.23e-07 | 63 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.69, 56.4, 5.48e+03 | 9.31e-07 | 33 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.21, 83.1, 645 | 9.35e-08 | 51 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.6, 99.8, 5.01e+04 | 4.37e-07 | 39 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 604, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 6.91, 79.5, 1.4e+05 | 3.11e-07 | 32 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.85, 427, 9.95e+03 | 6.33e-07 | 49 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.34e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.07e+03, 1.07e+03, 1.07e+03 | 2.05e-05 | 2 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 17.4, 240, 1.23e+03 | 4.09e-07 | 36 |  |
| 10 | cc | cc93:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 10.2, 517, 3.43e+03 | 1.27e-06 | 37 |  |
| 10 | cc | cc93:nest3N4 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.8, 19.2, 5.82e+03 | 2.6e-06 | 32 |  |
| 10 | cc | cc93:nest3N4 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 20.4, 1.25e+03, 1.26e+05 | 6.24e-05 | 81 |  |
| 10 | cc | cc93:nest3N4 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.47, 6.85, 20.5 | 6.17e-06 | 7 |  |
| 10 | cc | cc93:nest3N4 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 2.91, 28.5, 1.72e+04 | 0.000124 | 26 |  |
| 10 | cc | cc93:nest3N4 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 52.1, 1.94e+03, 7.6e+04 | 0.00013 | 91 |  |
| 10 | cc | cc93:nest3N4 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 4.29, 4.29, 17.3 | 4.35e-05 | 3 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.29, 14, 4.99e+04 | 0.0004 | 38 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 33.7, 558, 1.48e+04 | 0.00275 | 45 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.24, 1.62, 435 | 4.5e-05 | 106 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.98, 5.68, 6.7e+03 | 0.000135 | 55 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 78.3, 835, 3.45e+04 | 0.00128 | 32 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.71, 7.16, 317 | 1.28e-05 | 127 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 2.03, 5.99, 2.58e+03 | 2.16e-06 | 65 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 9.29, 846, 3.12e+04 | 0.000486 | 49 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.97, 4.49, 446 | 6.44e-05 | 78 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.5, 4.85, 147 | 9.55e-05 | 79 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12, 1.68e+03, 1.03e+05 | 6.52e-05 | 27 |  |
| 10 | cc | cc93:nest3N4 | group | gate_q4 | none | 93 | 93 | 0 | 1 | 0 | 0/0 | 1.82, 5.16, 731 | 3.59e-05 | 92 |  |
| 10 | cc | cc93:nest93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 10 | cc | cc93:nest93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 10 | cc | cc93:nest93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 10 | cc | cc93:nest93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 10 | cc | cc93:nest93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 10 | cc | cc93:nest93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 10 | cc | cc93:nest93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 26, 201 | 1.79e-06 | 38 |  |
| 10 | cc | cc93:nest93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 10 | cc | cc93:nest93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 10 | cc | cc93:nest93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.19, 7.74, 206 | 2.6e-06 | 32 |  |
| 10 | cc | cc93:nest93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 4.65, 89, 5.85e+03 | 6.24e-05 | 81 |  |
| 10 | cc | cc93:nest93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 2.83, 4.41, 18.3 | 6.17e-06 | 7 |  |
| 10 | cc | cc93:nest93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.48, 518 | 0.000124 | 26 |  |
| 10 | cc | cc93:nest93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 5.62, 74.4, 1.11e+03 | 0.00013 | 91 |  |
| 10 | cc | cc93:nest93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 14.9 | 4.35e-05 | 3 |  |
| 10 | cc | cc93:nest93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 5.15e+03 | 0.0004 | 38 |  |
| 10 | cc | cc93:nest93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15.2, 86.7, 1.95e+04 | 0.00275 | 45 |  |
| 10 | cc | cc93:nest93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 310 | 4.52e-05 | 107 |  |
| 10 | cc | cc93:nest93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.45, 348 | 0.000135 | 55 |  |
| 10 | cc | cc93:nest93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.01, 144, 5.57e+03 | 0.00128 | 32 |  |
| 10 | cc | cc93:nest93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.47, 201 | 1.7e-05 | 124 |  |
| 10 | cc | cc93:nest93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.35, 189 | 2.16e-06 | 65 |  |
| 10 | cc | cc93:nest93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.57, 103, 1.69e+03 | 0.000486 | 49 |  |
| 10 | cc | cc93:nest93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.11, 402 | 6.44e-05 | 78 |  |
| 10 | cc | cc93:nest93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.16, 80.8 | 9.55e-05 | 79 |  |
| 10 | cc | cc93:nest93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 143, 7.39e+03 | 6.52e-05 | 27 |  |
| 10 | cc | cc93:nest93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.53, 4.37, 448 | 3.11e-05 | 93 |  |
| 10 | cc | cc93:pair93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 10 | cc | cc93:pair93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 10 | cc | cc93:pair93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 10 | cc | cc93:pair93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 10 | cc | cc93:pair93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 10 | cc | cc93:pair93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 10 | cc | cc93:pair93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 17.4, 89.3 | 1.79e-06 | 38 |  |
| 10 | cc | cc93:pair93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 10 | cc | cc93:pair93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 10 | cc | cc93:pair93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.67, 7.74, 206 | 2.6e-06 | 32 |  |
| 10 | cc | cc93:pair93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 4.65, 66.2, 5.5e+03 | 6.24e-05 | 81 |  |
| 10 | cc | cc93:pair93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.03, 4.41, 17.5 | 6.17e-06 | 7 |  |
| 10 | cc | cc93:pair93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.48, 396 | 0.000124 | 26 |  |
| 10 | cc | cc93:pair93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 5.59, 68.7, 825 | 0.00013 | 91 |  |
| 10 | cc | cc93:pair93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 14.9 | 4.35e-05 | 3 |  |
| 10 | cc | cc93:pair93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 4.11e+03 | 0.0004 | 38 |  |
| 10 | cc | cc93:pair93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 12.8, 64.1, 1.76e+04 | 0.00275 | 45 |  |
| 10 | cc | cc93:pair93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.44, 307 | 4.52e-05 | 107 |  |
| 10 | cc | cc93:pair93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 5.11, 259 | 0.000135 | 55 |  |
| 10 | cc | cc93:pair93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.25, 107, 3.71e+03 | 0.00128 | 32 |  |
| 10 | cc | cc93:pair93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.39, 196 | 1.7e-05 | 124 |  |
| 10 | cc | cc93:pair93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.66, 5.16, 189 | 2.16e-06 | 65 |  |
| 10 | cc | cc93:pair93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 4.25, 72.3, 1.21e+03 | 0.000486 | 49 |  |
| 10 | cc | cc93:pair93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 3.94, 338 | 6.44e-05 | 78 |  |
| 10 | cc | cc93:pair93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.14, 80.8 | 9.55e-05 | 79 |  |
| 10 | cc | cc93:pair93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 116, 5.23e+03 | 6.52e-05 | 27 |  |
| 10 | cc | cc93:pair93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.4, 4.27, 448 | 3.11e-05 | 93 |  |
| 10 | cc | cc93:tail93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 10 | cc | cc93:tail93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 10 | cc | cc93:tail93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 10 | cc | cc93:tail93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 10 | cc | cc93:tail93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 10 | cc | cc93:tail93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 10 | cc | cc93:tail93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 17.4, 89.3 | 1.79e-06 | 38 |  |
| 10 | cc | cc93:tail93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 10 | cc | cc93:tail93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 10 | cc | cc93:tail93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.84, 8.49, 206 | 2.6e-06 | 32 |  |
| 10 | cc | cc93:tail93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 5.35, 101, 6.24e+03 | 6.24e-05 | 81 |  |
| 10 | cc | cc93:tail93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.26, 4.42, 17.6 | 6.17e-06 | 7 |  |
| 10 | cc | cc93:tail93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.49, 623 | 0.000124 | 26 |  |
| 10 | cc | cc93:tail93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 6.3, 84.1, 1.17e+03 | 0.00013 | 91 |  |
| 10 | cc | cc93:tail93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 21.8 | 4.35e-05 | 3 |  |
| 10 | cc | cc93:tail93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 12.5, 6.04e+03 | 0.0004 | 38 |  |
| 10 | cc | cc93:tail93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 2.66e+04 | 0.00275 | 45 |  |
| 10 | cc | cc93:tail93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 361 | 4.52e-05 | 107 |  |
| 10 | cc | cc93:tail93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.98, 378 | 0.000135 | 55 |  |
| 10 | cc | cc93:tail93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 5.43e+03 | 0.00128 | 32 |  |
| 10 | cc | cc93:tail93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.52, 197 | 1.7e-05 | 124 |  |
| 10 | cc | cc93:tail93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.78, 250 | 2.16e-06 | 65 |  |
| 10 | cc | cc93:tail93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 10 | cc | cc93:tail93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.93, 613 | 6.44e-05 | 78 |  |
| 10 | cc | cc93:tail93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.21, 104 | 9.55e-05 | 79 |  |
| 10 | cc | cc93:tail93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 10 | cc | cc93:tail93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.79, 4.99, 471 | 3.11e-05 | 93 |  |
| 10 | levin | cc47 | analytic | band_10_12 | 1-4 | 12 | 12 | 0 | 0 | 5 | 0/0 | 134, 239, 2.39e+03 | 8.21e-07 | 12 | {"bounded": 17} |
| 10 | levin | cc47 | analytic | band_10_12 | 5+ | 3 | 3 | 0 | 0 | 8 | 0/0 | 77.7, 77.7, 323 | 4.76e-06 | 3 | {"bounded": 11} |
| 10 | levin | cc47 | analytic | band_10_12 | none | 13 | 13 | 0 | 0 | 17 | 0/0 | 1.41, 8.92, 169 | 7.4e-07 | 13 | {"bounded": 30} |
| 10 | levin | cc47 | analytic | band_10_12 | unknown | 22 | 22 | 0 | 0 | 20 | 0/0 | 2.95, 81.5, 783 | 4.41e-06 | 22 | {"bounded": 42} |
| 10 | levin | cc47 | analytic | band_12_16 | 1-4 | 1 | 1 | 0 | 0 | 5 | 0/0 | 52.1, 52.1, 52.1 | 2.74e-05 | 1 | {"bounded": 6} |
| 10 | levin | cc47 | analytic | band_12_16 | 5+ | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 10 | levin | cc47 | analytic | band_12_16 | none | 6 | 6 | 0 | 3 | 27 | 0/0 | 3.33, 16.1, 38.4 | 1.17e-07 | 6 | {"bounded": 33, "undecided": 3} |
| 10 | levin | cc47 | analytic | band_12_16 | unknown | 16 | 16 | 0 | 0 | 40 | 0/0 | 2.5, 37.6, 603 | 1.81e-06 | 16 | {"bounded": 53, "undecided": 3} |
| 10 | levin | cc47 | analytic | band_16_20 | 1-4 | 3 | 3 | 0 | 0 | 15 | 0/0 | 60.5, 60.5, 326 | 7.09e-06 | 3 | {"bounded": 18} |
| 10 | levin | cc47 | analytic | band_16_20 | 5+ | 5 | 5 | 0 | 0 | 11 | 0/0 | 53.2, 60.5, 83.4 | 4.11e-06 | 5 | {"bounded": 16} |
| 10 | levin | cc47 | analytic | band_16_20 | none | 2 | 2 | 0 | 1 | 11 | 0/0 | 6.43, 9.03, 11.6 | 5.39e-05 | 2 | {"bounded": 13, "undecided": 1} |
| 10 | levin | cc47 | analytic | band_16_20 | unknown | 7 | 7 | 0 | 0 | 43 | 0/0 | 2.48, 21.7, 495 | 6.37e-06 | 7 | {"bounded": 45, "undecided": 5} |
| 10 | levin | cc47 | analytic | band_20_32 | 1-4 | 4 | 4 | 0 | 0 | 7 | 0/0 | 48.6, 510, 863 | 7.55e-07 | 4 | {"bounded": 11} |
| 10 | levin | cc47 | analytic | band_20_32 | 5+ | 7 | 7 | 0 | 0 | 26 | 0/0 | 12.8, 78.6, 123 | 1.39e-05 | 7 | {"bounded": 33} |
| 10 | levin | cc47 | analytic | band_20_32 | none | 3 | 3 | 0 | 9 | 39 | 0/0 | 8.26, 16.3, 54.5 | 9.56e-07 | 3 | {"undecided": 21, "bounded": 30} |
| 10 | levin | cc47 | analytic | band_20_32 | unknown | 22 | 22 | 0 | 10 | 71 | 0/0 | 2.13, 17.9, 389 | 5.45e-06 | 22 | {"bounded": 77, "undecided": 26} |
| 10 | levin | cc47 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 4, "understates": 2} |
| 10 | levin | cc47 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 10 | levin | cc47 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 10 | levin | cc47 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 64, "bounded": 31, "understates": 3} |
| 10 | levin | cc47 | group | band_10_12 | 1-4 | 24 | 24 | 0 | 16 | 0 | 0/0 | 1.33, 22.1, 1.45e+04 | 7.73e-06 | 24 | {"understates": 8, "bounded": 13, "undecided": 19} |
| 10 | levin | cc47 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 25.1, 4.06e+03, 2e+06 | 8.66e-07 | 73 | {"bounded": 33, "undecided": 14, "understates": 26} |
| 10 | levin | cc47 | group | band_10_12 | none | 2 | 2 | 0 | 5 | 0 | 0/0 | 1.31, 1.31, 1.31 | 0.000115 | 2 | {"bounded": 5, "undecided": 2} |
| 10 | levin | cc47 | group | band_12_16 | 1-4 | 44 | 44 | 0 | 2 | 27 | 0/0 | 5.38, 49.8, 1.27e+04 | 6.44e-07 | 44 | {"understates": 16, "undecided": 40, "bounded": 17} |
| 10 | levin | cc47 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 11, 2.52e+03, 1.71e+06 | 1.08e-06 | 85 | {"undecided": 20, "understates": 28, "bounded": 37} |
| 10 | levin | cc47 | group | band_12_16 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"undecided": 2} |
| 10 | levin | cc47 | group | band_16_20 | 1-4 | 36 | 36 | 0 | 2 | 28 | 0/0 | 5.77, 33.1, 5.67e+03 | 1.11e-06 | 36 | {"understates": 10, "undecided": 42, "bounded": 14} |
| 10 | levin | cc47 | group | band_16_20 | 5+ | 83 | 83 | 0 | 0 | 2 | 0/0 | 13.8, 1.36e+03, 6.24e+05 | 5.02e-07 | 83 | {"undecided": 18, "bounded": 37, "understates": 30} |
| 10 | levin | cc47 | group | band_16_20 | none | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 10 | levin | cc47 | group | band_20_32 | 1-4 | 36 | 36 | 0 | 8 | 53 | 0/0 | 3.28, 14.5, 4.38e+03 | 2.5e-06 | 36 | {"undecided": 85, "understates": 8, "bounded": 4} |
| 10 | levin | cc47 | group | band_20_32 | 5+ | 98 | 98 | 0 | 0 | 3 | 0/0 | 5.81, 420, 1.3e+05 | 8.38e-06 | 98 | {"bounded": 50, "understates": 42, "undecided": 9} |
| 10 | levin | cc47 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 10 | levin | cc47 | group | band_20_32v | 1-4 | 8 | 8 | 0 | 2 | 86 | 0/0 | 1.65, 9.1, 190 | 1.65e-06 | 8 | {"bounded": 15, "undecided": 72, "understates": 9} |
| 10 | levin | cc47 | group | band_20_32v | 5+ | 11 | 11 | 0 | 0 | 91 | 0/0 | 1.02, 3.11e+03, 3.57e+04 | 0.00525 | 9 | {"bounded": 64, "understates": 36, "undecided": 2} |
| 10 | levin | cc93 | analytic | band_10_12 | 1-4 | 12 | 12 | 0 | 0 | 5 | 0/0 | 138, 175, 1.34e+04 | 4.49e-07 | 12 | {"bounded": 17} |
| 10 | levin | cc93 | analytic | band_10_12 | 5+ | 5 | 5 | 0 | 0 | 6 | 0/0 | 104, 106, 3.71e+03 | 2.98e-06 | 5 | {"bounded": 11} |
| 10 | levin | cc93 | analytic | band_10_12 | none | 8 | 8 | 0 | 0 | 22 | 0/0 | 3.88, 16, 25.3 | 2.18e-06 | 8 | {"bounded": 30} |
| 10 | levin | cc93 | analytic | band_10_12 | unknown | 19 | 19 | 0 | 0 | 23 | 0/0 | 3.91, 68.5, 174 | 4.02e-06 | 19 | {"bounded": 42} |
| 10 | levin | cc93 | analytic | band_12_16 | 1-4 | 2 | 2 | 0 | 0 | 4 | 0/0 | 76.1, 221, 365 | 1.01e-05 | 2 | {"bounded": 6} |
| 10 | levin | cc93 | analytic | band_12_16 | 5+ | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 10 | levin | cc93 | analytic | band_12_16 | none | 11 | 11 | 0 | 2 | 23 | 0/0 | 4.39, 24.4, 65.3 | 1.44e-07 | 11 | {"bounded": 33, "undecided": 3} |
| 10 | levin | cc93 | analytic | band_12_16 | unknown | 13 | 13 | 0 | 0 | 43 | 0/0 | 3.01, 148, 196 | 1.16e-06 | 13 | {"bounded": 53, "undecided": 3} |
| 10 | levin | cc93 | analytic | band_16_20 | 1-4 | 2 | 2 | 0 | 0 | 16 | 0/0 | 70.8, 74, 77.2 | 6.82e-06 | 2 | {"bounded": 18} |
| 10 | levin | cc93 | analytic | band_16_20 | 5+ | 5 | 5 | 0 | 0 | 11 | 0/0 | 13.2, 69.7, 81.2 | 2.06e-06 | 5 | {"bounded": 16} |
| 10 | levin | cc93 | analytic | band_16_20 | none | 1 | 1 | 0 | 0 | 13 | 0/0 | 6.64, 6.64, 6.64 | 5.42e-05 | 1 | {"bounded": 13, "undecided": 1} |
| 10 | levin | cc93 | analytic | band_16_20 | unknown | 6 | 6 | 0 | 1 | 43 | 0/0 | 8.62, 52.3, 206 | 4.24e-06 | 6 | {"bounded": 45, "undecided": 5} |
| 10 | levin | cc93 | analytic | band_20_32 | 1-4 | 2 | 2 | 0 | 0 | 9 | 0/0 | 176, 314, 451 | 3.71e-06 | 2 | {"bounded": 11} |
| 10 | levin | cc93 | analytic | band_20_32 | 5+ | 9 | 9 | 0 | 0 | 24 | 0/0 | 17.3, 117, 451 | 5.07e-06 | 9 | {"bounded": 33} |
| 10 | levin | cc93 | analytic | band_20_32 | none | 7 | 7 | 0 | 7 | 37 | 0/0 | 3.11, 6.51, 132 | 1.34e-07 | 7 | {"undecided": 21, "bounded": 30} |
| 10 | levin | cc93 | analytic | band_20_32 | unknown | 16 | 16 | 0 | 9 | 78 | 0/0 | 4.13, 29.9, 38.6 | 5.81e-06 | 16 | {"bounded": 77, "undecided": 26} |
| 10 | levin | cc93 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 10 | levin | cc93 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 10 | levin | cc93 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 10 | levin | cc93 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 66, "bounded": 30, "understates": 2} |
| 10 | levin | cc93 | group | band_10_12 | 1-4 | 19 | 19 | 0 | 9 | 12 | 0/0 | 1.55, 27.6, 6.94e+03 | 2.66e-05 | 19 | {"understates": 5, "bounded": 11, "undecided": 24} |
| 10 | levin | cc93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 40.5, 9.52e+03, 1.74e+06 | 1.99e-07 | 73 | {"undecided": 23, "bounded": 19, "understates": 31} |
| 10 | levin | cc93 | group | band_10_12 | none | 2 | 2 | 0 | 2 | 3 | 0/0 | 1.56, 1.56, 1.56 | 0.000537 | 2 | {"bounded": 4, "undecided": 3} |
| 10 | levin | cc93 | group | band_12_16 | 1-4 | 45 | 45 | 0 | 1 | 27 | 0/0 | 5.81, 95.7, 1.88e+04 | 5.99e-07 | 45 | {"undecided": 48, "understates": 13, "bounded": 12} |
| 10 | levin | cc93 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 16.4, 4.8e+03, 2.96e+06 | 4.85e-07 | 85 | {"undecided": 39, "bounded": 32, "understates": 14} |
| 10 | levin | cc93 | group | band_12_16 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"undecided": 2} |
| 10 | levin | cc93 | group | band_16_20 | 1-4 | 36 | 36 | 0 | 3 | 27 | 0/0 | 5.44, 37.5, 4.42e+03 | 1.03e-06 | 36 | {"undecided": 47, "understates": 7, "bounded": 12} |
| 10 | levin | cc93 | group | band_16_20 | 5+ | 83 | 83 | 0 | 0 | 2 | 0/0 | 16.5, 3.89e+03, 2.24e+06 | 1.48e-07 | 83 | {"bounded": 25, "undecided": 30, "understates": 30} |
| 10 | levin | cc93 | group | band_16_20 | none | 0 | 0 | 0 | 1 | 5 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 10 | levin | cc93 | group | band_20_32 | 1-4 | 33 | 33 | 0 | 10 | 54 | 0/0 | 4.54, 24.8, 170 | 2.07e-06 | 33 | {"undecided": 85, "understates": 8, "bounded": 4} |
| 10 | levin | cc93 | group | band_20_32 | 5+ | 98 | 98 | 0 | 0 | 3 | 0/0 | 10.5, 804, 2.27e+05 | 5.43e-06 | 98 | {"bounded": 48, "understates": 42, "undecided": 11} |
| 10 | levin | cc93 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 10 | levin | cc93 | group | band_20_32v | 1-4 | 11 | 11 | 0 | 2 | 83 | 0/0 | 2.12, 11.6, 74.4 | 2.66e-07 | 11 | {"bounded": 15, "undecided": 72, "understates": 9} |
| 10 | levin | cc93 | group | band_20_32v | 5+ | 11 | 11 | 0 | 0 | 91 | 0/0 | 1, 1.43e+04, 4.72e+06 | 0.00077 | 9 | {"bounded": 62, "understates": 38, "undecided": 2} |
| 12 | cc | cc47:nest3N4 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 6.59e+06, 2.55e+07, 5.93e+07 | 0.00018 | 17 |  |
| 12 | cc | cc47:nest3N4 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 7.64e+06, 8.91e+07, 3.67e+09 | 6.42e-05 | 11 |  |
| 12 | cc | cc47:nest3N4 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 1.19e+07, 7.59e+07, 2.45e+09 | 6.01e-05 | 30 |  |
| 12 | cc | cc47:nest3N4 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 8.57e+06, 6.24e+07, 4.05e+08 | 5.61e-05 | 42 |  |
| 12 | cc | cc47:nest3N4 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.01e+03, 2.42e+03, 5.67e+05 | 7.38e-06 | 5 |  |
| 12 | cc | cc47:nest3N4 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.45e+04, 3.36e+05, 2.3e+06 | 7.45e-07 | 5 |  |
| 12 | cc | cc47:nest3N4 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 1.13e+03, 9.39e+04, 2.02e+07 | 5.8e-07 | 43 |  |
| 12 | cc | cc47:nest3N4 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 160, 3.03e+05, 1.23e+07 | 1.39e-06 | 39 |  |
| 12 | cc | cc47:nest3N4 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 9.63e+04, 4.02e+06, 1.58e+09 | 6.64e-06 | 13 |  |
| 12 | cc | cc47:nest3N4 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 1.61e+06, 1.02e+07, 4.5e+08 | 2.22e-05 | 6 |  |
| 12 | cc | cc47:nest3N4 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 3.47e+05, 2e+07, 5.8e+08 | 8.13e-06 | 38 |  |
| 12 | cc | cc47:nest3N4 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 1.37e+05, 3.41e+07, 9.11e+08 | 3.87e-06 | 47 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.72e+03, 1.53e+04 | 3.1e-07 | 4 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 34.8, 229 | 1.48e-07 | 63 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.38, 58.3, 3.97e+03 | 7.36e-07 | 33 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 375, 2.18e+03, 6.69e+04 | 4.89e-07 | 4 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.89, 31.4, 554 | 1.69e-07 | 51 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 7.33, 61.1, 1.44e+04 | 8.34e-07 | 39 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 612, 3.48e+03, 7.52e+03 | 1.3e-06 | 13 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.03e+03, 3.01e+03 | 1.97e-05 | 4 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.63, 54.3, 1.87e+05 | 5.13e-07 | 32 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.42, 414, 1.12e+04 | 6.24e-07 | 49 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 473, 2.44e+03, 5.95e+03 | 2.8e-06 | 12 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 739, 739, 739 | 2.96e-05 | 2 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12, 266, 911 | 3.5e-07 | 36 |  |
| 12 | cc | cc47:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 15.1, 172, 1.5e+03 | 1.52e-06 | 37 |  |
| 12 | cc | cc47:nest3N4 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 5.4, 187, 9.15e+04 | 3.42e-06 | 40 |  |
| 12 | cc | cc47:nest3N4 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 5.94, 433, 4.28e+04 | 0.000104 | 73 |  |
| 12 | cc | cc47:nest3N4 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 29.2, 1.79e+03, 3.88e+04 | 3.91e-06 | 7 |  |
| 12 | cc | cc47:nest3N4 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.43, 6.64, 2.16e+03 | 1.62e-06 | 32 |  |
| 12 | cc | cc47:nest3N4 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 1.69, 140, 2.43e+03 | 0.00104 | 80 |  |
| 12 | cc | cc47:nest3N4 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 2.1, 5.49, 47.1 | 1.3e-06 | 7 |  |
| 12 | cc | cc47:nest3N4 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 2.16, 38.3, 1.36e+03 | 0.000232 | 26 |  |
| 12 | cc | cc47:nest3N4 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 3.49, 178, 1.01e+04 | 0.00127 | 89 |  |
| 12 | cc | cc47:nest3N4 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.54, 3.54, 6.23 | 5.27e-05 | 3 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.3, 10.2, 2.03e+03 | 0.00096 | 37 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 1.57, 37.9, 934 | 0.041 | 40 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.26, 1.59, 123 | 4.84e-05 | 106 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 2.03, 6.75, 76.1 | 0.000109 | 55 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.6, 34.1, 202 | 0.0339 | 31 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.73, 4.74, 108 | 3.35e-05 | 127 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.95, 5.23, 221 | 3.86e-06 | 65 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 1.44, 41.3, 356 | 0.026 | 47 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.92, 3.94, 650 | 8.03e-05 | 78 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.51, 3.54, 51.4 | 7.31e-05 | 79 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 2.38, 47.4, 796 | 0.00112 | 26 |  |
| 12 | cc | cc47:nest3N4 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 2.03, 3.52, 217 | 2.63e-05 | 94 |  |
| 12 | cc | cc47:tree | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 6.95e+06, 1.1e+08, 3.25e+09 | 0.00018 | 17 |  |
| 12 | cc | cc47:tree | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 1.9e+06, 1.21e+09, 3.23e+10 | 6.42e-05 | 11 |  |
| 12 | cc | cc47:tree | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 8.3e+05, 2.39e+08, 2.12e+10 | 6.01e-05 | 30 |  |
| 12 | cc | cc47:tree | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 6.86e+05, 6.32e+08, 6.18e+09 | 5.61e-05 | 42 |  |
| 12 | cc | cc47:tree | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.66e+06, 7.6e+06, 2.9e+08 | 7.38e-06 | 5 |  |
| 12 | cc | cc47:tree | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 7.56e+07, 1.05e+08, 2.71e+08 | 7.45e-07 | 5 |  |
| 12 | cc | cc47:tree | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 4.33e+06, 7.99e+07, 2.59e+10 | 5.8e-07 | 43 |  |
| 12 | cc | cc47:tree | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 4.23e+05, 8.25e+07, 2.47e+09 | 1.39e-06 | 39 |  |
| 12 | cc | cc47:tree | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 2.31e+06, 1.97e+08, 5.41e+10 | 6.64e-06 | 13 |  |
| 12 | cc | cc47:tree | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 9.02e+06, 7.26e+07, 9.77e+09 | 2.22e-05 | 6 |  |
| 12 | cc | cc47:tree | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 6.38e+07, 2.82e+08, 2.38e+10 | 8.13e-06 | 38 |  |
| 12 | cc | cc47:tree | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 1.21e+07, 4.89e+08, 3e+10 | 3.87e-06 | 47 |  |
| 12 | cc | cc47:tree | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.69e+03, 1.52e+04 | 3.1e-07 | 4 |  |
| 12 | cc | cc47:tree | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 5.46, 26.7, 177 | 1.48e-07 | 63 |  |
| 12 | cc | cc47:tree | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 2.63, 45.5, 3.96e+03 | 7.36e-07 | 33 |  |
| 12 | cc | cc47:tree | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 373, 2.17e+03, 6.68e+04 | 4.89e-07 | 4 |  |
| 12 | cc | cc47:tree | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 6.96, 28, 3.93e+04 | 1.69e-07 | 51 |  |
| 12 | cc | cc47:tree | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 5.85, 48.7, 1.43e+04 | 8.34e-07 | 39 |  |
| 12 | cc | cc47:tree | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 609, 3.46e+03, 1.37e+05 | 1.3e-06 | 13 |  |
| 12 | cc | cc47:tree | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.88e+04, 6.06e+04 | 1.97e-05 | 4 |  |
| 12 | cc | cc47:tree | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.05, 2.41e+04, 2.29e+07 | 5.13e-07 | 32 |  |
| 12 | cc | cc47:tree | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.01, 2.69e+03, 1.12e+07 | 6.24e-07 | 49 |  |
| 12 | cc | cc47:tree | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 5.29e+04, 3.41e+05, 5.87e+05 | 2.8e-06 | 12 |  |
| 12 | cc | cc47:tree | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 8.23e+04, 8.23e+04, 8.23e+04 | 2.96e-05 | 2 |  |
| 12 | cc | cc47:tree | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 1.45e+04, 3.51e+05, 6.87e+06 | 3.5e-07 | 36 |  |
| 12 | cc | cc47:tree | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 7.32e+03, 7.41e+05, 1.21e+07 | 1.52e-06 | 37 |  |
| 12 | cc | cc47:tree | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 3.64e+04, 2.96e+06, 3.24e+07 | 3.42e-06 | 40 |  |
| 12 | cc | cc47:tree | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 1.46, 1.61e+05, 1.92e+07 | 0.000104 | 73 |  |
| 12 | cc | cc47:tree | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 4.63e+05, 4.9e+06, 9.87e+07 | 3.91e-06 | 7 |  |
| 12 | cc | cc47:tree | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 10.5, 938, 2.59e+04 | 1.62e-06 | 32 |  |
| 12 | cc | cc47:tree | group | band_6_8 | 5+ | 81 | 76 | 5 | 0 | 0 | 0/0 | 0.0385, 62.5, 2.64e+04 | 0.00104 | 80 |  |
| 12 | cc | cc47:tree | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 5.24, 102, 1.84e+03 | 1.3e-06 | 7 |  |
| 12 | cc | cc47:tree | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 264, 6.34e+04, 3.12e+06 | 0.000232 | 26 |  |
| 12 | cc | cc47:tree | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 37.3, 1.89e+03, 6.48e+06 | 0.00127 | 89 |  |
| 12 | cc | cc47:tree | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 4.18e+03, 4.18e+03, 9.48e+04 | 5.27e-05 | 3 |  |
| 12 | cc | cc47:tree | group | gate_q1 | 1-4 | 21 | 21 | 0 | 17 | 0 | 0/0 | 1.08, 9.02, 629 | 0.00261 | 20 |  |
| 12 | cc | cc47:tree | group | gate_q1 | 5+ | 45 | 27 | 18 | 0 | 0 | 0/0 | 0.0613, 4.6, 286 | 0.041 | 40 |  |
| 12 | cc | cc47:tree | group | gate_q1 | none | 7 | 7 | 0 | 107 | 0 | 0/0 | 1.28, 2.62, 20.2 | 3.56e-05 | 7 |  |
| 12 | cc | cc47:tree | group | gate_q2 | 1-4 | 10 | 9 | 1 | 45 | 0 | 0/0 | 0.776, 13.6, 26 | 0.0269 | 10 |  |
| 12 | cc | cc47:tree | group | gate_q2 | 5+ | 32 | 25 | 7 | 0 | 0 | 0/0 | 0.0137, 4.62, 47 | 0.0339 | 31 |  |
| 12 | cc | cc47:tree | group | gate_q2 | none | 11 | 11 | 0 | 116 | 0 | 0/0 | 1.84, 2.49, 6.78 | 4.24e-05 | 11 |  |
| 12 | cc | cc47:tree | group | gate_q3 | 1-4 | 10 | 10 | 0 | 55 | 0 | 0/0 | 4.22, 18.8, 67.3 | 0.0826 | 10 |  |
| 12 | cc | cc47:tree | group | gate_q3 | 5+ | 49 | 32 | 17 | 0 | 0 | 0/0 | 0.00255, 5.37, 101 | 0.026 | 47 |  |
| 12 | cc | cc47:tree | group | gate_q3 | none | 4 | 4 | 0 | 74 | 0 | 0/0 | 1.52, 2.02, 2.51 | 0.000895 | 4 |  |
| 12 | cc | cc47:tree | group | gate_q4 | 1-4 | 16 | 16 | 0 | 63 | 0 | 0/0 | 1.52, 4.29, 36.9 | 0.000454 | 16 |  |
| 12 | cc | cc47:tree | group | gate_q4 | 5+ | 27 | 23 | 4 | 0 | 0 | 0/0 | 0.171, 7.07, 392 | 0.00112 | 26 |  |
| 12 | cc | cc47:tree | group | gate_q4 | none | 7 | 7 | 0 | 87 | 0 | 0/0 | 1.7, 4.66, 13.7 | 5.82e-06 | 7 |  |
| 12 | cc | cc93:chop93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 423, 741, 3.18e+04 | 1.4e-06 | 17 |  |
| 12 | cc | cc93:chop93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 8.4e+03, 9.83e+04 | 3.8e-07 | 11 |  |
| 12 | cc | cc93:chop93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 30.9, 131, 1.69e+03 | 1.89e-06 | 30 |  |
| 12 | cc | cc93:chop93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 32.4, 684, 7.08e+04 | 1.6e-06 | 42 |  |
| 12 | cc | cc93:chop93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 12 | cc | cc93:chop93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 12 | cc | cc93:chop93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 38.7, 551, 2.16e+04 | 3.59e-07 | 43 |  |
| 12 | cc | cc93:chop93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 56.5, 472, 9.43e+03 | 7.91e-07 | 39 |  |
| 12 | cc | cc93:chop93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 12 | cc | cc93:chop93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 12 | cc | cc93:chop93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 26, 121, 626 | 1.79e-06 | 38 |  |
| 12 | cc | cc93:chop93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 29.9, 714, 6.21e+03 | 1.11e-06 | 47 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.59e+04, 1.59e+04, 1.59e+04 | 2.98e-07 | 4 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 154, 3.22e+03, 7.35e+06 | 2.23e-07 | 63 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 50.6, 2.92e+03, 3.51e+05 | 9.31e-07 | 33 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.73e+03, 3.49e+03 | 2.39e-06 | 4 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 108, 4.99e+03, 1.75e+05 | 9.35e-08 | 51 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 639, 4.44e+03, 2.32e+05 | 4.37e-07 | 39 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 1.01e+03, 6.03e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 71.1, 557, 1.84e+04 | 3.11e-07 | 32 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 87.4, 2.16e+03, 1.95e+04 | 6.33e-07 | 49 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 78.3, 308, 3.26e+03 | 4.09e-07 | 36 |  |
| 12 | cc | cc93:chop93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 95.2, 620, 6.89e+03 | 1.27e-06 | 37 |  |
| 12 | cc | cc93:chop93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 981, 1.12e+04 | 2.13e-06 | 40 |  |
| 12 | cc | cc93:chop93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.48, 84.7, 1.48e+04 | 8.54e-06 | 73 |  |
| 12 | cc | cc93:chop93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 179, 1.9e+03, 7.05e+03 | 3.17e-05 | 7 |  |
| 12 | cc | cc93:chop93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.44, 450, 3.99e+04 | 2.6e-06 | 32 |  |
| 12 | cc | cc93:chop93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 5.35, 101, 7.94e+05 | 6.24e-05 | 81 |  |
| 12 | cc | cc93:chop93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 184, 1.02e+03, 1.77e+03 | 6.17e-06 | 7 |  |
| 12 | cc | cc93:chop93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 4.01, 547, 3.83e+04 | 0.000124 | 26 |  |
| 12 | cc | cc93:chop93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 6.3, 84.1, 1.17e+03 | 0.00013 | 91 |  |
| 12 | cc | cc93:chop93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 123, 123, 1.51e+03 | 4.35e-05 | 3 |  |
| 12 | cc | cc93:chop93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 4.29, 1.53e+03, 1.88e+08 | 0.0004 | 38 |  |
| 12 | cc | cc93:chop93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 1.09e+07 | 0.00275 | 45 |  |
| 12 | cc | cc93:chop93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 382, 1.28e+04, 3.83e+07 | 4.52e-05 | 107 |  |
| 12 | cc | cc93:chop93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 63.2, 4.34e+03, 2.08e+05 | 0.000135 | 55 |  |
| 12 | cc | cc93:chop93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 6.59e+04 | 0.00128 | 32 |  |
| 12 | cc | cc93:chop93 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 85.2, 7.81e+03, 8.74e+05 | 1.28e-05 | 127 |  |
| 12 | cc | cc93:chop93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 32, 1.71e+03, 1.29e+05 | 2.16e-06 | 65 |  |
| 12 | cc | cc93:chop93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 12 | cc | cc93:chop93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 280, 2.36e+03, 2.39e+05 | 6.44e-05 | 78 |  |
| 12 | cc | cc93:chop93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 26.8, 1.66e+03, 4.34e+04 | 9.55e-05 | 79 |  |
| 12 | cc | cc93:chop93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 21.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 12 | cc | cc93:chop93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 65.5, 1.24e+03, 2.37e+05 | 3.11e-05 | 93 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 2.13e+07, 9.12e+09, 3.23e+11 | 1.4e-06 | 17 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 2.8e+08, 2.01e+10, 4.76e+11 | 3.8e-07 | 11 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 1.73e+08, 1.97e+09, 9.43e+10 | 1.89e-06 | 30 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 4.02e+06, 1.99e+09, 2.46e+13 | 1.6e-06 | 42 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.51e+03, 5.22e+03, 5.94e+05 | 7.36e-06 | 5 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 4.41e+04, 4.9e+05, 5.74e+06 | 5.11e-07 | 5 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 1.41e+03, 1.06e+05, 2.98e+07 | 3.59e-07 | 43 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 541, 5.09e+05, 4.24e+07 | 7.91e-07 | 39 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 3.84e+05, 1.04e+07, 1.33e+09 | 3.57e-06 | 13 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 1.95e+06, 4.97e+08, 5.16e+09 | 2.19e-06 | 6 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 6.83e+05, 1.72e+07, 5.62e+09 | 1.79e-06 | 38 |  |
| 12 | cc | cc93:nest3N4 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 8.3e+04, 6.88e+07, 4.75e+09 | 1.11e-06 | 47 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 50.6, 229 | 2.23e-07 | 63 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.69, 56.4, 5.48e+03 | 9.31e-07 | 33 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.21, 83.1, 645 | 9.35e-08 | 51 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.6, 99.8, 5.01e+04 | 4.37e-07 | 39 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 604, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 6.91, 79.5, 1.4e+05 | 3.11e-07 | 32 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.85, 427, 9.95e+03 | 6.33e-07 | 49 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.34e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.07e+03, 1.07e+03, 1.07e+03 | 2.05e-05 | 2 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 17.4, 240, 1.23e+03 | 4.09e-07 | 36 |  |
| 12 | cc | cc93:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 10.2, 517, 3.43e+03 | 1.27e-06 | 37 |  |
| 12 | cc | cc93:nest3N4 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 3.55, 355, 2.04e+04 | 2.13e-06 | 40 |  |
| 12 | cc | cc93:nest3N4 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 110, 4.37e+03, 1.25e+05 | 8.54e-06 | 73 |  |
| 12 | cc | cc93:nest3N4 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 11, 27.5, 6.24e+04 | 3.17e-05 | 7 |  |
| 12 | cc | cc93:nest3N4 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.8, 19.2, 5.82e+03 | 2.6e-06 | 32 |  |
| 12 | cc | cc93:nest3N4 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 20.4, 1.25e+03, 1.26e+05 | 6.24e-05 | 81 |  |
| 12 | cc | cc93:nest3N4 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.47, 6.85, 20.5 | 6.17e-06 | 7 |  |
| 12 | cc | cc93:nest3N4 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 2.91, 28.5, 1.72e+04 | 0.000124 | 26 |  |
| 12 | cc | cc93:nest3N4 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 52.1, 1.94e+03, 7.6e+04 | 0.00013 | 91 |  |
| 12 | cc | cc93:nest3N4 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 4.29, 4.29, 17.3 | 4.35e-05 | 3 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.29, 14, 4.99e+04 | 0.0004 | 38 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 33.7, 558, 1.48e+04 | 0.00275 | 45 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.24, 1.62, 435 | 4.5e-05 | 106 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.98, 5.68, 6.7e+03 | 0.000135 | 55 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 78.3, 835, 3.45e+04 | 0.00128 | 32 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.71, 7.16, 317 | 1.28e-05 | 127 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 2.03, 5.99, 2.58e+03 | 2.16e-06 | 65 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 9.29, 846, 3.12e+04 | 0.000486 | 49 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.97, 4.49, 446 | 6.44e-05 | 78 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.5, 4.85, 147 | 9.55e-05 | 79 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12, 1.68e+03, 1.03e+05 | 6.52e-05 | 27 |  |
| 12 | cc | cc93:nest3N4 | group | gate_q4 | none | 93 | 93 | 0 | 1 | 0 | 0/0 | 1.82, 5.16, 731 | 3.59e-05 | 92 |  |
| 12 | cc | cc93:nest93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 585, 1.39e+03, 3.18e+04 | 1.4e-06 | 17 |  |
| 12 | cc | cc93:nest93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 6.23e+04, 9.83e+04 | 3.8e-07 | 11 |  |
| 12 | cc | cc93:nest93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 11.1, 90.8, 7.91e+03 | 1.89e-06 | 30 |  |
| 12 | cc | cc93:nest93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 8.07, 684, 2.86e+06 | 1.6e-06 | 42 |  |
| 12 | cc | cc93:nest93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 12 | cc | cc93:nest93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 12 | cc | cc93:nest93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 12 | cc | cc93:nest93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 12 | cc | cc93:nest93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 12 | cc | cc93:nest93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 12 | cc | cc93:nest93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 26, 201 | 1.79e-06 | 38 |  |
| 12 | cc | cc93:nest93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 12 | cc | cc93:nest93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 12 | cc | cc93:nest93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 12.3, 224 | 2.13e-06 | 40 |  |
| 12 | cc | cc93:nest93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.25, 75.6, 2.24e+03 | 8.54e-06 | 73 |  |
| 12 | cc | cc93:nest93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 4.67, 7.73, 138 | 3.17e-05 | 7 |  |
| 12 | cc | cc93:nest93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.19, 7.74, 206 | 2.6e-06 | 32 |  |
| 12 | cc | cc93:nest93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 4.65, 89, 5.85e+03 | 6.24e-05 | 81 |  |
| 12 | cc | cc93:nest93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 2.83, 4.41, 18.3 | 6.17e-06 | 7 |  |
| 12 | cc | cc93:nest93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.48, 518 | 0.000124 | 26 |  |
| 12 | cc | cc93:nest93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 5.62, 74.4, 1.11e+03 | 0.00013 | 91 |  |
| 12 | cc | cc93:nest93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 14.9 | 4.35e-05 | 3 |  |
| 12 | cc | cc93:nest93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 5.15e+03 | 0.0004 | 38 |  |
| 12 | cc | cc93:nest93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15.2, 86.7, 1.95e+04 | 0.00275 | 45 |  |
| 12 | cc | cc93:nest93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 310 | 4.52e-05 | 107 |  |
| 12 | cc | cc93:nest93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.45, 348 | 0.000135 | 55 |  |
| 12 | cc | cc93:nest93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.01, 144, 5.57e+03 | 0.00128 | 32 |  |
| 12 | cc | cc93:nest93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.47, 201 | 1.7e-05 | 124 |  |
| 12 | cc | cc93:nest93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.35, 189 | 2.16e-06 | 65 |  |
| 12 | cc | cc93:nest93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.57, 103, 1.69e+03 | 0.000486 | 49 |  |
| 12 | cc | cc93:nest93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.11, 402 | 6.44e-05 | 78 |  |
| 12 | cc | cc93:nest93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.16, 80.8 | 9.55e-05 | 79 |  |
| 12 | cc | cc93:nest93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 143, 7.39e+03 | 6.52e-05 | 27 |  |
| 12 | cc | cc93:nest93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.53, 4.37, 448 | 3.11e-05 | 93 |  |
| 12 | cc | cc93:pair93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 423, 741, 3.18e+04 | 1.4e-06 | 17 |  |
| 12 | cc | cc93:pair93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 8.4e+03, 9.83e+04 | 3.8e-07 | 11 |  |
| 12 | cc | cc93:pair93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 8.88, 18.5, 212 | 1.89e-06 | 30 |  |
| 12 | cc | cc93:pair93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 7.91, 468, 7.08e+04 | 1.6e-06 | 42 |  |
| 12 | cc | cc93:pair93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 12 | cc | cc93:pair93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 12 | cc | cc93:pair93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 12 | cc | cc93:pair93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 12 | cc | cc93:pair93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 12 | cc | cc93:pair93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 12 | cc | cc93:pair93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 17.4, 89.3 | 1.79e-06 | 38 |  |
| 12 | cc | cc93:pair93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 12 | cc | cc93:pair93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 12 | cc | cc93:pair93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 12.9, 161 | 2.13e-06 | 40 |  |
| 12 | cc | cc93:pair93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.35, 57.4, 2.21e+03 | 8.54e-06 | 73 |  |
| 12 | cc | cc93:pair93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 4.82, 6.78, 138 | 3.17e-05 | 7 |  |
| 12 | cc | cc93:pair93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.67, 7.74, 206 | 2.6e-06 | 32 |  |
| 12 | cc | cc93:pair93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 4.65, 66.2, 5.5e+03 | 6.24e-05 | 81 |  |
| 12 | cc | cc93:pair93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.03, 4.41, 17.5 | 6.17e-06 | 7 |  |
| 12 | cc | cc93:pair93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.48, 396 | 0.000124 | 26 |  |
| 12 | cc | cc93:pair93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 5.59, 68.7, 825 | 0.00013 | 91 |  |
| 12 | cc | cc93:pair93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 14.9 | 4.35e-05 | 3 |  |
| 12 | cc | cc93:pair93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 4.11e+03 | 0.0004 | 38 |  |
| 12 | cc | cc93:pair93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 12.8, 64.1, 1.76e+04 | 0.00275 | 45 |  |
| 12 | cc | cc93:pair93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.44, 307 | 4.52e-05 | 107 |  |
| 12 | cc | cc93:pair93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 5.11, 259 | 0.000135 | 55 |  |
| 12 | cc | cc93:pair93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.25, 107, 3.71e+03 | 0.00128 | 32 |  |
| 12 | cc | cc93:pair93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.39, 196 | 1.7e-05 | 124 |  |
| 12 | cc | cc93:pair93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.66, 5.16, 189 | 2.16e-06 | 65 |  |
| 12 | cc | cc93:pair93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 4.25, 72.3, 1.21e+03 | 0.000486 | 49 |  |
| 12 | cc | cc93:pair93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 3.94, 338 | 6.44e-05 | 78 |  |
| 12 | cc | cc93:pair93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.14, 80.8 | 9.55e-05 | 79 |  |
| 12 | cc | cc93:pair93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 116, 5.23e+03 | 6.52e-05 | 27 |  |
| 12 | cc | cc93:pair93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.4, 4.27, 448 | 3.11e-05 | 93 |  |
| 12 | cc | cc93:tail93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 423, 741, 3.18e+04 | 1.4e-06 | 17 |  |
| 12 | cc | cc93:tail93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 8.4e+03, 9.83e+04 | 3.8e-07 | 11 |  |
| 12 | cc | cc93:tail93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 8.88, 18.5, 212 | 1.89e-06 | 30 |  |
| 12 | cc | cc93:tail93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 7.91, 468, 7.08e+04 | 1.6e-06 | 42 |  |
| 12 | cc | cc93:tail93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 12 | cc | cc93:tail93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 12 | cc | cc93:tail93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 12 | cc | cc93:tail93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 12 | cc | cc93:tail93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 12 | cc | cc93:tail93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 12 | cc | cc93:tail93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 17.4, 89.3 | 1.79e-06 | 38 |  |
| 12 | cc | cc93:tail93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 12 | cc | cc93:tail93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 12 | cc | cc93:tail93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 14.1, 226 | 2.13e-06 | 40 |  |
| 12 | cc | cc93:tail93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.48, 84.7, 3.05e+03 | 8.54e-06 | 73 |  |
| 12 | cc | cc93:tail93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 5.52, 9.92, 155 | 3.17e-05 | 7 |  |
| 12 | cc | cc93:tail93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.84, 8.49, 206 | 2.6e-06 | 32 |  |
| 12 | cc | cc93:tail93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 5.35, 101, 6.24e+03 | 6.24e-05 | 81 |  |
| 12 | cc | cc93:tail93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.26, 4.42, 17.6 | 6.17e-06 | 7 |  |
| 12 | cc | cc93:tail93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.49, 623 | 0.000124 | 26 |  |
| 12 | cc | cc93:tail93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 6.3, 84.1, 1.17e+03 | 0.00013 | 91 |  |
| 12 | cc | cc93:tail93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 21.8 | 4.35e-05 | 3 |  |
| 12 | cc | cc93:tail93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 12.5, 6.04e+03 | 0.0004 | 38 |  |
| 12 | cc | cc93:tail93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 2.66e+04 | 0.00275 | 45 |  |
| 12 | cc | cc93:tail93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 361 | 4.52e-05 | 107 |  |
| 12 | cc | cc93:tail93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.98, 378 | 0.000135 | 55 |  |
| 12 | cc | cc93:tail93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 5.43e+03 | 0.00128 | 32 |  |
| 12 | cc | cc93:tail93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.52, 197 | 1.7e-05 | 124 |  |
| 12 | cc | cc93:tail93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.78, 250 | 2.16e-06 | 65 |  |
| 12 | cc | cc93:tail93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 12 | cc | cc93:tail93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.93, 613 | 6.44e-05 | 78 |  |
| 12 | cc | cc93:tail93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.21, 104 | 9.55e-05 | 79 |  |
| 12 | cc | cc93:tail93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 12 | cc | cc93:tail93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.79, 4.99, 471 | 3.11e-05 | 93 |  |
| 12 | levin | cc47 | analytic | band_12_16 | 1-4 | 1 | 1 | 0 | 0 | 5 | 0/0 | 52.1, 52.1, 52.1 | 2.74e-05 | 1 | {"bounded": 6} |
| 12 | levin | cc47 | analytic | band_12_16 | 5+ | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 12 | levin | cc47 | analytic | band_12_16 | none | 11 | 11 | 0 | 2 | 23 | 0/0 | 3.33, 16.1, 71.2 | 1.06e-07 | 11 | {"bounded": 33, "undecided": 3} |
| 12 | levin | cc47 | analytic | band_12_16 | unknown | 20 | 20 | 0 | 0 | 36 | 0/0 | 2.5, 37.6, 603 | 1.81e-06 | 20 | {"bounded": 53, "undecided": 3} |
| 12 | levin | cc47 | analytic | band_16_20 | 1-4 | 4 | 4 | 0 | 0 | 14 | 0/0 | 15.4, 60.5, 326 | 8.95e-06 | 4 | {"bounded": 18} |
| 12 | levin | cc47 | analytic | band_16_20 | 5+ | 6 | 6 | 0 | 0 | 10 | 0/0 | 44.3, 60.5, 83.4 | 8.08e-06 | 6 | {"bounded": 16} |
| 12 | levin | cc47 | analytic | band_16_20 | none | 2 | 2 | 0 | 1 | 11 | 0/0 | 6.43, 9.03, 11.6 | 5.39e-05 | 2 | {"bounded": 13, "undecided": 1} |
| 12 | levin | cc47 | analytic | band_16_20 | unknown | 13 | 13 | 0 | 0 | 37 | 0/0 | 2.48, 38.5, 495 | 6.37e-06 | 13 | {"bounded": 45, "undecided": 5} |
| 12 | levin | cc47 | analytic | band_20_32 | 1-4 | 4 | 4 | 0 | 0 | 7 | 0/0 | 48.6, 458, 863 | 1.41e-06 | 4 | {"bounded": 11} |
| 12 | levin | cc47 | analytic | band_20_32 | 5+ | 7 | 7 | 0 | 0 | 26 | 0/0 | 12.8, 61.2, 112 | 1.39e-05 | 7 | {"bounded": 33} |
| 12 | levin | cc47 | analytic | band_20_32 | none | 1 | 1 | 0 | 7 | 43 | 0/0 | 6.55, 6.55, 6.55 | 2.61e-07 | 1 | {"undecided": 21, "bounded": 30} |
| 12 | levin | cc47 | analytic | band_20_32 | unknown | 17 | 17 | 0 | 7 | 79 | 0/0 | 2.13, 27.8, 353 | 4.98e-06 | 17 | {"bounded": 77, "undecided": 26} |
| 12 | levin | cc47 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 4, "understates": 2} |
| 12 | levin | cc47 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 12 | levin | cc47 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 12 | levin | cc47 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 64, "bounded": 31, "understates": 3} |
| 12 | levin | cc47 | group | band_12_16 | 1-4 | 44 | 44 | 0 | 2 | 27 | 0/0 | 5.38, 49.8, 1.27e+04 | 6.44e-07 | 44 | {"understates": 16, "undecided": 40, "bounded": 17} |
| 12 | levin | cc47 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 11, 2.52e+03, 1.71e+06 | 1.08e-06 | 85 | {"undecided": 20, "understates": 28, "bounded": 37} |
| 12 | levin | cc47 | group | band_12_16 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"undecided": 2} |
| 12 | levin | cc47 | group | band_16_20 | 1-4 | 36 | 36 | 0 | 2 | 28 | 0/0 | 5.77, 33.1, 5.67e+03 | 1.11e-06 | 36 | {"understates": 10, "undecided": 42, "bounded": 14} |
| 12 | levin | cc47 | group | band_16_20 | 5+ | 84 | 84 | 0 | 0 | 1 | 0/0 | 13.8, 1.25e+03, 6.24e+05 | 5.32e-07 | 84 | {"undecided": 20, "bounded": 36, "understates": 29} |
| 12 | levin | cc47 | group | band_16_20 | none | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 12 | levin | cc47 | group | band_20_32 | 1-4 | 30 | 30 | 0 | 4 | 63 | 0/0 | 6.1, 17.8, 4.21e+03 | 3.22e-06 | 30 | {"undecided": 77, "understates": 9, "bounded": 11} |
| 12 | levin | cc47 | group | band_20_32 | 5+ | 96 | 96 | 0 | 0 | 5 | 0/0 | 4.46, 445, 1.3e+05 | 8.38e-06 | 96 | {"understates": 37, "bounded": 45, "undecided": 19} |
| 12 | levin | cc47 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 12 | levin | cc47 | group | band_20_32v | 1-4 | 13 | 13 | 0 | 12 | 71 | 0/0 | 1.65, 9.6, 181 | 8.02e-08 | 13 | {"bounded": 31, "undecided": 55, "understates": 10} |
| 12 | levin | cc47 | group | band_20_32v | 5+ | 21 | 21 | 0 | 0 | 81 | 0/0 | 5.73, 1.27e+03, 3.32e+04 | 6.95e-05 | 21 | {"bounded": 55, "understates": 41, "undecided": 6} |
| 12 | levin | cc93 | analytic | band_12_16 | 1-4 | 2 | 2 | 0 | 0 | 4 | 0/0 | 76.1, 221, 365 | 1.01e-05 | 2 | {"bounded": 6} |
| 12 | levin | cc93 | analytic | band_12_16 | 5+ | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 12 | levin | cc93 | analytic | band_12_16 | none | 18 | 18 | 0 | 0 | 18 | 0/0 | 4.66, 15.8, 65.3 | 1.73e-07 | 18 | {"bounded": 33, "undecided": 3} |
| 12 | levin | cc93 | analytic | band_12_16 | unknown | 14 | 14 | 0 | 0 | 42 | 0/0 | 3.01, 69.4, 196 | 1.16e-06 | 14 | {"bounded": 53, "undecided": 3} |
| 12 | levin | cc93 | analytic | band_16_20 | 1-4 | 3 | 3 | 0 | 0 | 15 | 0/0 | 17.5, 55.8, 77.2 | 6.32e-05 | 3 | {"bounded": 18} |
| 12 | levin | cc93 | analytic | band_16_20 | 5+ | 6 | 6 | 0 | 0 | 10 | 0/0 | 13.2, 69.7, 81.2 | 5.58e-06 | 6 | {"bounded": 16} |
| 12 | levin | cc93 | analytic | band_16_20 | none | 1 | 1 | 0 | 0 | 13 | 0/0 | 6.64, 6.64, 6.64 | 5.42e-05 | 1 | {"bounded": 13, "undecided": 1} |
| 12 | levin | cc93 | analytic | band_16_20 | unknown | 12 | 12 | 0 | 1 | 37 | 0/0 | 8.62, 82.4, 667 | 1.98e-06 | 12 | {"bounded": 45, "undecided": 5} |
| 12 | levin | cc93 | analytic | band_20_32 | 1-4 | 2 | 2 | 0 | 0 | 9 | 0/0 | 104, 140, 176 | 4.28e-06 | 2 | {"bounded": 11} |
| 12 | levin | cc93 | analytic | band_20_32 | 5+ | 10 | 10 | 0 | 0 | 23 | 0/0 | 17.3, 117, 451 | 2.8e-06 | 10 | {"bounded": 33} |
| 12 | levin | cc93 | analytic | band_20_32 | none | 4 | 4 | 0 | 5 | 42 | 0/0 | 2.53, 9.67, 15.1 | 1.48e-06 | 4 | {"undecided": 21, "bounded": 30} |
| 12 | levin | cc93 | analytic | band_20_32 | unknown | 16 | 16 | 0 | 6 | 81 | 0/0 | 4.07, 32.9, 200 | 7.5e-06 | 16 | {"bounded": 77, "undecided": 26} |
| 12 | levin | cc93 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 12 | levin | cc93 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 12 | levin | cc93 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 12 | levin | cc93 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 66, "bounded": 30, "understates": 2} |
| 12 | levin | cc93 | group | band_12_16 | 1-4 | 45 | 45 | 0 | 1 | 27 | 0/0 | 5.81, 95.7, 1.88e+04 | 5.99e-07 | 45 | {"undecided": 48, "understates": 13, "bounded": 12} |
| 12 | levin | cc93 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 16.4, 4.8e+03, 2.96e+06 | 4.85e-07 | 85 | {"undecided": 39, "bounded": 32, "understates": 14} |
| 12 | levin | cc93 | group | band_12_16 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"undecided": 2} |
| 12 | levin | cc93 | group | band_16_20 | 1-4 | 36 | 36 | 0 | 3 | 27 | 0/0 | 5.44, 37.5, 4.42e+03 | 1.03e-06 | 36 | {"undecided": 47, "understates": 7, "bounded": 12} |
| 12 | levin | cc93 | group | band_16_20 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 16.5, 3.89e+03, 2.24e+06 | 2.14e-07 | 85 | {"bounded": 21, "undecided": 38, "understates": 26} |
| 12 | levin | cc93 | group | band_16_20 | none | 0 | 0 | 0 | 1 | 5 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 12 | levin | cc93 | group | band_20_32 | 1-4 | 32 | 32 | 0 | 5 | 60 | 0/0 | 3.35, 23.9, 152 | 1.61e-06 | 32 | {"undecided": 84, "understates": 5, "bounded": 8} |
| 12 | levin | cc93 | group | band_20_32 | 5+ | 97 | 97 | 0 | 0 | 4 | 0/0 | 12, 703, 2.27e+05 | 5.68e-06 | 97 | {"understates": 38, "bounded": 38, "undecided": 25} |
| 12 | levin | cc93 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 12 | levin | cc93 | group | band_20_32v | 1-4 | 12 | 12 | 0 | 4 | 80 | 0/0 | 2.12, 12.5, 71.5 | 1.15e-07 | 12 | {"bounded": 26, "undecided": 61, "understates": 9} |
| 12 | levin | cc93 | group | band_20_32v | 5+ | 22 | 22 | 0 | 0 | 80 | 0/0 | 15.5, 2.1e+03, 4.38e+06 | 5.27e-07 | 22 | {"bounded": 52, "understates": 44, "undecided": 6} |
| 16 | cc | cc47:nest3N4 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 6.59e+06, 2.55e+07, 5.93e+07 | 0.00018 | 17 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 7.64e+06, 8.91e+07, 3.67e+09 | 6.42e-05 | 11 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 1.19e+07, 7.59e+07, 2.45e+09 | 6.01e-05 | 30 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 8.57e+06, 6.24e+07, 4.05e+08 | 5.61e-05 | 42 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 3.1e+06, 1.35e+07, 7.24e+08 | 0.136 | 6 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 2.97e+06, 9.5e+06, 2.64e+07 | 0.422 | 4 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 1.12e+06, 7.95e+06, 3.48e+08 | 0.0566 | 28 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 6.57e+05, 1.42e+07, 2.02e+08 | 0.236 | 39 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.01e+03, 2.42e+03, 5.67e+05 | 7.38e-06 | 5 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.45e+04, 3.36e+05, 2.3e+06 | 7.45e-07 | 5 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 1.13e+03, 9.39e+04, 2.02e+07 | 5.8e-07 | 43 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 160, 3.03e+05, 1.23e+07 | 1.39e-06 | 39 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 9.63e+04, 4.02e+06, 1.58e+09 | 6.64e-06 | 13 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 1.61e+06, 1.02e+07, 4.5e+08 | 2.22e-05 | 6 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 3.47e+05, 2e+07, 5.8e+08 | 8.13e-06 | 38 |  |
| 16 | cc | cc47:nest3N4 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 1.37e+05, 3.41e+07, 9.11e+08 | 3.87e-06 | 47 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.72e+03, 1.53e+04 | 3.1e-07 | 4 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 34.8, 229 | 1.48e-07 | 63 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.38, 58.3, 3.97e+03 | 7.36e-07 | 33 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 375, 2.18e+03, 6.69e+04 | 4.89e-07 | 4 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.89, 31.4, 554 | 1.69e-07 | 51 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 7.33, 61.1, 1.44e+04 | 8.34e-07 | 39 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 612, 3.48e+03, 7.52e+03 | 1.3e-06 | 13 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.03e+03, 3.01e+03 | 1.97e-05 | 4 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.63, 54.3, 1.87e+05 | 5.13e-07 | 32 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.42, 414, 1.12e+04 | 6.24e-07 | 49 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 473, 2.44e+03, 5.95e+03 | 2.8e-06 | 12 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 739, 739, 739 | 2.96e-05 | 2 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12, 266, 911 | 3.5e-07 | 36 |  |
| 16 | cc | cc47:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 15.1, 172, 1.5e+03 | 1.52e-06 | 37 |  |
| 16 | cc | cc47:nest3N4 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 5.4, 187, 9.15e+04 | 3.42e-06 | 40 |  |
| 16 | cc | cc47:nest3N4 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 5.94, 433, 4.28e+04 | 0.000104 | 73 |  |
| 16 | cc | cc47:nest3N4 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 29.2, 1.79e+03, 3.88e+04 | 3.91e-06 | 7 |  |
| 16 | cc | cc47:nest3N4 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 455, 1.54e+06, 1.02e+09 | 5.31e-06 | 73 |  |
| 16 | cc | cc47:nest3N4 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 87.6, 2.58e+05, 5.23e+07 | 0.000101 | 83 |  |
| 16 | cc | cc47:nest3N4 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 670, 6.66e+04, 1.33e+05 | 0.000579 | 2 |  |
| 16 | cc | cc47:nest3N4 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.43, 6.64, 2.16e+03 | 1.62e-06 | 32 |  |
| 16 | cc | cc47:nest3N4 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 1.69, 140, 2.43e+03 | 0.00104 | 80 |  |
| 16 | cc | cc47:nest3N4 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 2.1, 5.49, 47.1 | 1.3e-06 | 7 |  |
| 16 | cc | cc47:nest3N4 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 2.16, 38.3, 1.36e+03 | 0.000232 | 26 |  |
| 16 | cc | cc47:nest3N4 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 3.49, 178, 1.01e+04 | 0.00127 | 89 |  |
| 16 | cc | cc47:nest3N4 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.54, 3.54, 6.23 | 5.27e-05 | 3 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.3, 10.2, 2.03e+03 | 0.00096 | 37 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 1.57, 37.9, 934 | 0.041 | 40 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.26, 1.59, 123 | 4.84e-05 | 106 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 2.03, 6.75, 76.1 | 0.000109 | 55 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.6, 34.1, 202 | 0.0339 | 31 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.73, 4.74, 108 | 3.35e-05 | 127 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.95, 5.23, 221 | 3.86e-06 | 65 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 1.44, 41.3, 356 | 0.026 | 47 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.92, 3.94, 650 | 8.03e-05 | 78 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.51, 3.54, 51.4 | 7.31e-05 | 79 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 2.38, 47.4, 796 | 0.00112 | 26 |  |
| 16 | cc | cc47:nest3N4 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 2.03, 3.52, 217 | 2.63e-05 | 94 |  |
| 16 | cc | cc47:tree | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 6.95e+06, 1.1e+08, 3.25e+09 | 0.00018 | 17 |  |
| 16 | cc | cc47:tree | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 1.9e+06, 1.21e+09, 3.23e+10 | 6.42e-05 | 11 |  |
| 16 | cc | cc47:tree | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 8.3e+05, 2.39e+08, 2.12e+10 | 6.01e-05 | 30 |  |
| 16 | cc | cc47:tree | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 6.86e+05, 6.32e+08, 6.18e+09 | 5.61e-05 | 42 |  |
| 16 | cc | cc47:tree | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 4.01e+05, 3.42e+06, 9.56e+08 | 0.136 | 6 |  |
| 16 | cc | cc47:tree | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 4.64e+03, 3.57e+05, 4.57e+06 | 0.422 | 4 |  |
| 16 | cc | cc47:tree | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 3.66e+03, 3.82e+05, 2.63e+08 | 0.0566 | 28 |  |
| 16 | cc | cc47:tree | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 1.28e+03, 5.66e+05, 2.54e+08 | 0.236 | 39 |  |
| 16 | cc | cc47:tree | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.66e+06, 7.6e+06, 2.9e+08 | 7.38e-06 | 5 |  |
| 16 | cc | cc47:tree | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 7.56e+07, 1.05e+08, 2.71e+08 | 7.45e-07 | 5 |  |
| 16 | cc | cc47:tree | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 4.33e+06, 7.99e+07, 2.59e+10 | 5.8e-07 | 43 |  |
| 16 | cc | cc47:tree | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 4.23e+05, 8.25e+07, 2.47e+09 | 1.39e-06 | 39 |  |
| 16 | cc | cc47:tree | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 2.31e+06, 1.97e+08, 5.41e+10 | 6.64e-06 | 13 |  |
| 16 | cc | cc47:tree | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 9.02e+06, 7.26e+07, 9.77e+09 | 2.22e-05 | 6 |  |
| 16 | cc | cc47:tree | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 6.38e+07, 2.82e+08, 2.38e+10 | 8.13e-06 | 38 |  |
| 16 | cc | cc47:tree | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 1.21e+07, 4.89e+08, 3e+10 | 3.87e-06 | 47 |  |
| 16 | cc | cc47:tree | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.69e+03, 1.52e+04 | 3.1e-07 | 4 |  |
| 16 | cc | cc47:tree | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 5.46, 26.7, 177 | 1.48e-07 | 63 |  |
| 16 | cc | cc47:tree | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 2.63, 45.5, 3.96e+03 | 7.36e-07 | 33 |  |
| 16 | cc | cc47:tree | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 373, 2.17e+03, 6.68e+04 | 4.89e-07 | 4 |  |
| 16 | cc | cc47:tree | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 6.96, 28, 3.93e+04 | 1.69e-07 | 51 |  |
| 16 | cc | cc47:tree | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 5.85, 48.7, 1.43e+04 | 8.34e-07 | 39 |  |
| 16 | cc | cc47:tree | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 609, 3.46e+03, 1.37e+05 | 1.3e-06 | 13 |  |
| 16 | cc | cc47:tree | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.88e+04, 6.06e+04 | 1.97e-05 | 4 |  |
| 16 | cc | cc47:tree | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.05, 2.41e+04, 2.29e+07 | 5.13e-07 | 32 |  |
| 16 | cc | cc47:tree | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.01, 2.69e+03, 1.12e+07 | 6.24e-07 | 49 |  |
| 16 | cc | cc47:tree | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 5.29e+04, 3.41e+05, 5.87e+05 | 2.8e-06 | 12 |  |
| 16 | cc | cc47:tree | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 8.23e+04, 8.23e+04, 8.23e+04 | 2.96e-05 | 2 |  |
| 16 | cc | cc47:tree | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 1.45e+04, 3.51e+05, 6.87e+06 | 3.5e-07 | 36 |  |
| 16 | cc | cc47:tree | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 7.32e+03, 7.41e+05, 1.21e+07 | 1.52e-06 | 37 |  |
| 16 | cc | cc47:tree | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 3.64e+04, 2.96e+06, 3.24e+07 | 3.42e-06 | 40 |  |
| 16 | cc | cc47:tree | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 1.46, 1.61e+05, 1.92e+07 | 0.000104 | 73 |  |
| 16 | cc | cc47:tree | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 4.63e+05, 4.9e+06, 9.87e+07 | 3.91e-06 | 7 |  |
| 16 | cc | cc47:tree | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 1.82e+06, 5.92e+07, 1.09e+09 | 5.31e-06 | 73 |  |
| 16 | cc | cc47:tree | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 1.49e+03, 8.44e+06, 1.5e+08 | 0.000101 | 83 |  |
| 16 | cc | cc47:tree | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 4.74e+06, 7.61e+06, 1.05e+07 | 0.000579 | 2 |  |
| 16 | cc | cc47:tree | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 10.5, 938, 2.59e+04 | 1.62e-06 | 32 |  |
| 16 | cc | cc47:tree | group | band_6_8 | 5+ | 81 | 76 | 5 | 0 | 0 | 0/0 | 0.0385, 62.5, 2.64e+04 | 0.00104 | 80 |  |
| 16 | cc | cc47:tree | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 5.24, 102, 1.84e+03 | 1.3e-06 | 7 |  |
| 16 | cc | cc47:tree | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 264, 6.34e+04, 3.12e+06 | 0.000232 | 26 |  |
| 16 | cc | cc47:tree | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 37.3, 1.89e+03, 6.48e+06 | 0.00127 | 89 |  |
| 16 | cc | cc47:tree | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 4.18e+03, 4.18e+03, 9.48e+04 | 5.27e-05 | 3 |  |
| 16 | cc | cc47:tree | group | gate_q1 | 1-4 | 21 | 21 | 0 | 17 | 0 | 0/0 | 1.08, 9.02, 629 | 0.00261 | 20 |  |
| 16 | cc | cc47:tree | group | gate_q1 | 5+ | 45 | 27 | 18 | 0 | 0 | 0/0 | 0.0613, 4.6, 286 | 0.041 | 40 |  |
| 16 | cc | cc47:tree | group | gate_q1 | none | 7 | 7 | 0 | 107 | 0 | 0/0 | 1.28, 2.62, 20.2 | 3.56e-05 | 7 |  |
| 16 | cc | cc47:tree | group | gate_q2 | 1-4 | 10 | 9 | 1 | 45 | 0 | 0/0 | 0.776, 13.6, 26 | 0.0269 | 10 |  |
| 16 | cc | cc47:tree | group | gate_q2 | 5+ | 32 | 25 | 7 | 0 | 0 | 0/0 | 0.0137, 4.62, 47 | 0.0339 | 31 |  |
| 16 | cc | cc47:tree | group | gate_q2 | none | 11 | 11 | 0 | 116 | 0 | 0/0 | 1.84, 2.49, 6.78 | 4.24e-05 | 11 |  |
| 16 | cc | cc47:tree | group | gate_q3 | 1-4 | 10 | 10 | 0 | 55 | 0 | 0/0 | 4.22, 18.8, 67.3 | 0.0826 | 10 |  |
| 16 | cc | cc47:tree | group | gate_q3 | 5+ | 49 | 32 | 17 | 0 | 0 | 0/0 | 0.00255, 5.37, 101 | 0.026 | 47 |  |
| 16 | cc | cc47:tree | group | gate_q3 | none | 4 | 4 | 0 | 74 | 0 | 0/0 | 1.52, 2.02, 2.51 | 0.000895 | 4 |  |
| 16 | cc | cc47:tree | group | gate_q4 | 1-4 | 16 | 16 | 0 | 63 | 0 | 0/0 | 1.52, 4.29, 36.9 | 0.000454 | 16 |  |
| 16 | cc | cc47:tree | group | gate_q4 | 5+ | 27 | 23 | 4 | 0 | 0 | 0/0 | 0.171, 7.07, 392 | 0.00112 | 26 |  |
| 16 | cc | cc47:tree | group | gate_q4 | none | 7 | 7 | 0 | 87 | 0 | 0/0 | 1.7, 4.66, 13.7 | 5.82e-06 | 7 |  |
| 16 | cc | cc93:chop93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 423, 741, 3.18e+04 | 1.4e-06 | 17 |  |
| 16 | cc | cc93:chop93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 8.4e+03, 9.83e+04 | 3.8e-07 | 11 |  |
| 16 | cc | cc93:chop93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 30.9, 131, 1.69e+03 | 1.89e-06 | 30 |  |
| 16 | cc | cc93:chop93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 32.4, 684, 7.08e+04 | 1.6e-06 | 42 |  |
| 16 | cc | cc93:chop93 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 238, 467, 818 | 2e-05 | 6 |  |
| 16 | cc | cc93:chop93 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 854, 7.47e+03, 3.8e+04 | 8.77e-07 | 6 |  |
| 16 | cc | cc93:chop93 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 13.1, 57.1, 962 | 6.82e-07 | 36 |  |
| 16 | cc | cc93:chop93 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 10.3, 474, 1.79e+04 | 1.71e-06 | 56 |  |
| 16 | cc | cc93:chop93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 16 | cc | cc93:chop93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 16 | cc | cc93:chop93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 38.7, 551, 2.16e+04 | 3.59e-07 | 43 |  |
| 16 | cc | cc93:chop93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 56.5, 472, 9.43e+03 | 7.91e-07 | 39 |  |
| 16 | cc | cc93:chop93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 16 | cc | cc93:chop93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 16 | cc | cc93:chop93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 26, 121, 626 | 1.79e-06 | 38 |  |
| 16 | cc | cc93:chop93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 29.9, 714, 6.21e+03 | 1.11e-06 | 47 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.59e+04, 1.59e+04, 1.59e+04 | 2.98e-07 | 4 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 154, 3.22e+03, 7.35e+06 | 2.23e-07 | 63 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 50.6, 2.92e+03, 3.51e+05 | 9.31e-07 | 33 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.73e+03, 3.49e+03 | 2.39e-06 | 4 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 108, 4.99e+03, 1.75e+05 | 9.35e-08 | 51 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 639, 4.44e+03, 2.32e+05 | 4.37e-07 | 39 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 1.01e+03, 6.03e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 71.1, 557, 1.84e+04 | 3.11e-07 | 32 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 87.4, 2.16e+03, 1.95e+04 | 6.33e-07 | 49 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 78.3, 308, 3.26e+03 | 4.09e-07 | 36 |  |
| 16 | cc | cc93:chop93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 95.2, 620, 6.89e+03 | 1.27e-06 | 37 |  |
| 16 | cc | cc93:chop93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 981, 1.12e+04 | 2.13e-06 | 40 |  |
| 16 | cc | cc93:chop93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.48, 84.7, 1.48e+04 | 8.54e-06 | 73 |  |
| 16 | cc | cc93:chop93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 179, 1.9e+03, 7.05e+03 | 3.17e-05 | 7 |  |
| 16 | cc | cc93:chop93 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 2.92, 294, 2.98e+04 | 4.46e-06 | 73 |  |
| 16 | cc | cc93:chop93 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.7, 76.5, 4.31e+03 | 5.91e-06 | 85 |  |
| 16 | cc | cc93:chop93 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 89.2, 280, 471 | 0.000302 | 2 |  |
| 16 | cc | cc93:chop93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.44, 450, 3.99e+04 | 2.6e-06 | 32 |  |
| 16 | cc | cc93:chop93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 5.35, 101, 7.94e+05 | 6.24e-05 | 81 |  |
| 16 | cc | cc93:chop93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 184, 1.02e+03, 1.77e+03 | 6.17e-06 | 7 |  |
| 16 | cc | cc93:chop93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 4.01, 547, 3.83e+04 | 0.000124 | 26 |  |
| 16 | cc | cc93:chop93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 6.3, 84.1, 1.17e+03 | 0.00013 | 91 |  |
| 16 | cc | cc93:chop93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 123, 123, 1.51e+03 | 4.35e-05 | 3 |  |
| 16 | cc | cc93:chop93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 4.29, 1.53e+03, 1.88e+08 | 0.0004 | 38 |  |
| 16 | cc | cc93:chop93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 1.09e+07 | 0.00275 | 45 |  |
| 16 | cc | cc93:chop93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 382, 1.28e+04, 3.83e+07 | 4.52e-05 | 107 |  |
| 16 | cc | cc93:chop93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 63.2, 4.34e+03, 2.08e+05 | 0.000135 | 55 |  |
| 16 | cc | cc93:chop93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 6.59e+04 | 0.00128 | 32 |  |
| 16 | cc | cc93:chop93 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 85.2, 7.81e+03, 8.74e+05 | 1.28e-05 | 127 |  |
| 16 | cc | cc93:chop93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 32, 1.71e+03, 1.29e+05 | 2.16e-06 | 65 |  |
| 16 | cc | cc93:chop93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 16 | cc | cc93:chop93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 280, 2.36e+03, 2.39e+05 | 6.44e-05 | 78 |  |
| 16 | cc | cc93:chop93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 26.8, 1.66e+03, 4.34e+04 | 9.55e-05 | 79 |  |
| 16 | cc | cc93:chop93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 21.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 16 | cc | cc93:chop93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 65.5, 1.24e+03, 2.37e+05 | 3.11e-05 | 93 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 2.13e+07, 9.12e+09, 3.23e+11 | 1.4e-06 | 17 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 2.8e+08, 2.01e+10, 4.76e+11 | 3.8e-07 | 11 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 1.73e+08, 1.97e+09, 9.43e+10 | 1.89e-06 | 30 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 4.02e+06, 1.99e+09, 2.46e+13 | 1.6e-06 | 42 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 6.9e+09, 2.07e+11, 6e+11 | 2e-05 | 6 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 7.34e+11, 1.02e+13, 3.87e+13 | 8.77e-07 | 6 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 1.67e+09, 1.11e+12, 1.38e+14 | 6.82e-07 | 36 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 9.48e+09, 1.73e+12, 6.79e+14 | 1.71e-06 | 56 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.51e+03, 5.22e+03, 5.94e+05 | 7.36e-06 | 5 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 4.41e+04, 4.9e+05, 5.74e+06 | 5.11e-07 | 5 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 1.41e+03, 1.06e+05, 2.98e+07 | 3.59e-07 | 43 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 541, 5.09e+05, 4.24e+07 | 7.91e-07 | 39 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 3.84e+05, 1.04e+07, 1.33e+09 | 3.57e-06 | 13 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 1.95e+06, 4.97e+08, 5.16e+09 | 2.19e-06 | 6 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 6.83e+05, 1.72e+07, 5.62e+09 | 1.79e-06 | 38 |  |
| 16 | cc | cc93:nest3N4 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 8.3e+04, 6.88e+07, 4.75e+09 | 1.11e-06 | 47 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 50.6, 229 | 2.23e-07 | 63 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.69, 56.4, 5.48e+03 | 9.31e-07 | 33 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.21, 83.1, 645 | 9.35e-08 | 51 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.6, 99.8, 5.01e+04 | 4.37e-07 | 39 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 604, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 6.91, 79.5, 1.4e+05 | 3.11e-07 | 32 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.85, 427, 9.95e+03 | 6.33e-07 | 49 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.34e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.07e+03, 1.07e+03, 1.07e+03 | 2.05e-05 | 2 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 17.4, 240, 1.23e+03 | 4.09e-07 | 36 |  |
| 16 | cc | cc93:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 10.2, 517, 3.43e+03 | 1.27e-06 | 37 |  |
| 16 | cc | cc93:nest3N4 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 3.55, 355, 2.04e+04 | 2.13e-06 | 40 |  |
| 16 | cc | cc93:nest3N4 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 110, 4.37e+03, 1.25e+05 | 8.54e-06 | 73 |  |
| 16 | cc | cc93:nest3N4 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 11, 27.5, 6.24e+04 | 3.17e-05 | 7 |  |
| 16 | cc | cc93:nest3N4 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 908, 2.83e+06, 8.41e+08 | 4.46e-06 | 73 |  |
| 16 | cc | cc93:nest3N4 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 1.36e+03, 4.12e+06, 2.59e+09 | 5.91e-06 | 85 |  |
| 16 | cc | cc93:nest3N4 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 696, 1.29e+05, 2.58e+05 | 0.000302 | 2 |  |
| 16 | cc | cc93:nest3N4 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.8, 19.2, 5.82e+03 | 2.6e-06 | 32 |  |
| 16 | cc | cc93:nest3N4 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 20.4, 1.25e+03, 1.26e+05 | 6.24e-05 | 81 |  |
| 16 | cc | cc93:nest3N4 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.47, 6.85, 20.5 | 6.17e-06 | 7 |  |
| 16 | cc | cc93:nest3N4 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 2.91, 28.5, 1.72e+04 | 0.000124 | 26 |  |
| 16 | cc | cc93:nest3N4 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 52.1, 1.94e+03, 7.6e+04 | 0.00013 | 91 |  |
| 16 | cc | cc93:nest3N4 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 4.29, 4.29, 17.3 | 4.35e-05 | 3 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.29, 14, 4.99e+04 | 0.0004 | 38 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 33.7, 558, 1.48e+04 | 0.00275 | 45 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.24, 1.62, 435 | 4.5e-05 | 106 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.98, 5.68, 6.7e+03 | 0.000135 | 55 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 78.3, 835, 3.45e+04 | 0.00128 | 32 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.71, 7.16, 317 | 1.28e-05 | 127 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 2.03, 5.99, 2.58e+03 | 2.16e-06 | 65 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 9.29, 846, 3.12e+04 | 0.000486 | 49 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.97, 4.49, 446 | 6.44e-05 | 78 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.5, 4.85, 147 | 9.55e-05 | 79 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12, 1.68e+03, 1.03e+05 | 6.52e-05 | 27 |  |
| 16 | cc | cc93:nest3N4 | group | gate_q4 | none | 93 | 93 | 0 | 1 | 0 | 0/0 | 1.82, 5.16, 731 | 3.59e-05 | 92 |  |
| 16 | cc | cc93:nest93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 585, 1.39e+03, 3.18e+04 | 1.4e-06 | 17 |  |
| 16 | cc | cc93:nest93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 6.23e+04, 9.83e+04 | 3.8e-07 | 11 |  |
| 16 | cc | cc93:nest93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 11.1, 90.8, 7.91e+03 | 1.89e-06 | 30 |  |
| 16 | cc | cc93:nest93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 8.07, 684, 2.86e+06 | 1.6e-06 | 42 |  |
| 16 | cc | cc93:nest93 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 238, 2.07e+04, 1.94e+05 | 2e-05 | 6 |  |
| 16 | cc | cc93:nest93 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 2.45e+05, 1.34e+06, 2.42e+06 | 8.77e-07 | 6 |  |
| 16 | cc | cc93:nest93 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 49, 3.15e+05, 1.69e+07 | 6.82e-07 | 36 |  |
| 16 | cc | cc93:nest93 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 48.3, 1.15e+05, 8.88e+08 | 1.71e-06 | 56 |  |
| 16 | cc | cc93:nest93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 16 | cc | cc93:nest93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 16 | cc | cc93:nest93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 16 | cc | cc93:nest93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 16 | cc | cc93:nest93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 16 | cc | cc93:nest93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 16 | cc | cc93:nest93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 26, 201 | 1.79e-06 | 38 |  |
| 16 | cc | cc93:nest93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 16 | cc | cc93:nest93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 16 | cc | cc93:nest93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 12.3, 224 | 2.13e-06 | 40 |  |
| 16 | cc | cc93:nest93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.25, 75.6, 2.24e+03 | 8.54e-06 | 73 |  |
| 16 | cc | cc93:nest93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 4.67, 7.73, 138 | 3.17e-05 | 7 |  |
| 16 | cc | cc93:nest93 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 2.09, 12, 824 | 4.46e-06 | 73 |  |
| 16 | cc | cc93:nest93 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.7, 69.2, 3.94e+03 | 5.91e-06 | 85 |  |
| 16 | cc | cc93:nest93 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 2.9, 3.52, 4.14 | 0.000302 | 2 |  |
| 16 | cc | cc93:nest93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.19, 7.74, 206 | 2.6e-06 | 32 |  |
| 16 | cc | cc93:nest93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 4.65, 89, 5.85e+03 | 6.24e-05 | 81 |  |
| 16 | cc | cc93:nest93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 2.83, 4.41, 18.3 | 6.17e-06 | 7 |  |
| 16 | cc | cc93:nest93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.48, 518 | 0.000124 | 26 |  |
| 16 | cc | cc93:nest93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 5.62, 74.4, 1.11e+03 | 0.00013 | 91 |  |
| 16 | cc | cc93:nest93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 14.9 | 4.35e-05 | 3 |  |
| 16 | cc | cc93:nest93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 5.15e+03 | 0.0004 | 38 |  |
| 16 | cc | cc93:nest93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15.2, 86.7, 1.95e+04 | 0.00275 | 45 |  |
| 16 | cc | cc93:nest93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 310 | 4.52e-05 | 107 |  |
| 16 | cc | cc93:nest93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.45, 348 | 0.000135 | 55 |  |
| 16 | cc | cc93:nest93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.01, 144, 5.57e+03 | 0.00128 | 32 |  |
| 16 | cc | cc93:nest93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.47, 201 | 1.7e-05 | 124 |  |
| 16 | cc | cc93:nest93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.35, 189 | 2.16e-06 | 65 |  |
| 16 | cc | cc93:nest93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.57, 103, 1.69e+03 | 0.000486 | 49 |  |
| 16 | cc | cc93:nest93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.11, 402 | 6.44e-05 | 78 |  |
| 16 | cc | cc93:nest93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.16, 80.8 | 9.55e-05 | 79 |  |
| 16 | cc | cc93:nest93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 143, 7.39e+03 | 6.52e-05 | 27 |  |
| 16 | cc | cc93:nest93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.53, 4.37, 448 | 3.11e-05 | 93 |  |
| 16 | cc | cc93:pair93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 423, 741, 3.18e+04 | 1.4e-06 | 17 |  |
| 16 | cc | cc93:pair93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 8.4e+03, 9.83e+04 | 3.8e-07 | 11 |  |
| 16 | cc | cc93:pair93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 8.88, 18.5, 212 | 1.89e-06 | 30 |  |
| 16 | cc | cc93:pair93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 7.91, 468, 7.08e+04 | 1.6e-06 | 42 |  |
| 16 | cc | cc93:pair93 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 238, 467, 818 | 2e-05 | 6 |  |
| 16 | cc | cc93:pair93 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 854, 7.47e+03, 3.8e+04 | 8.77e-07 | 6 |  |
| 16 | cc | cc93:pair93 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 6.83, 25.9, 962 | 6.82e-07 | 36 |  |
| 16 | cc | cc93:pair93 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 6.38, 406, 1.79e+04 | 1.71e-06 | 56 |  |
| 16 | cc | cc93:pair93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 16 | cc | cc93:pair93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 16 | cc | cc93:pair93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 16 | cc | cc93:pair93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 16 | cc | cc93:pair93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 16 | cc | cc93:pair93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 16 | cc | cc93:pair93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 17.4, 89.3 | 1.79e-06 | 38 |  |
| 16 | cc | cc93:pair93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 16 | cc | cc93:pair93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 16 | cc | cc93:pair93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 12.9, 161 | 2.13e-06 | 40 |  |
| 16 | cc | cc93:pair93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.35, 57.4, 2.21e+03 | 8.54e-06 | 73 |  |
| 16 | cc | cc93:pair93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 4.82, 6.78, 138 | 3.17e-05 | 7 |  |
| 16 | cc | cc93:pair93 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 2.09, 11.5, 825 | 4.46e-06 | 73 |  |
| 16 | cc | cc93:pair93 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.7, 56.2, 2.87e+03 | 5.91e-06 | 85 |  |
| 16 | cc | cc93:pair93 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 3.53, 3.84, 4.14 | 0.000302 | 2 |  |
| 16 | cc | cc93:pair93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.67, 7.74, 206 | 2.6e-06 | 32 |  |
| 16 | cc | cc93:pair93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 4.65, 66.2, 5.5e+03 | 6.24e-05 | 81 |  |
| 16 | cc | cc93:pair93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.03, 4.41, 17.5 | 6.17e-06 | 7 |  |
| 16 | cc | cc93:pair93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.48, 396 | 0.000124 | 26 |  |
| 16 | cc | cc93:pair93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 5.59, 68.7, 825 | 0.00013 | 91 |  |
| 16 | cc | cc93:pair93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 14.9 | 4.35e-05 | 3 |  |
| 16 | cc | cc93:pair93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 4.11e+03 | 0.0004 | 38 |  |
| 16 | cc | cc93:pair93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 12.8, 64.1, 1.76e+04 | 0.00275 | 45 |  |
| 16 | cc | cc93:pair93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.44, 307 | 4.52e-05 | 107 |  |
| 16 | cc | cc93:pair93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 5.11, 259 | 0.000135 | 55 |  |
| 16 | cc | cc93:pair93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.25, 107, 3.71e+03 | 0.00128 | 32 |  |
| 16 | cc | cc93:pair93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.39, 196 | 1.7e-05 | 124 |  |
| 16 | cc | cc93:pair93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.66, 5.16, 189 | 2.16e-06 | 65 |  |
| 16 | cc | cc93:pair93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 4.25, 72.3, 1.21e+03 | 0.000486 | 49 |  |
| 16 | cc | cc93:pair93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 3.94, 338 | 6.44e-05 | 78 |  |
| 16 | cc | cc93:pair93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.14, 80.8 | 9.55e-05 | 79 |  |
| 16 | cc | cc93:pair93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 116, 5.23e+03 | 6.52e-05 | 27 |  |
| 16 | cc | cc93:pair93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.4, 4.27, 448 | 3.11e-05 | 93 |  |
| 16 | cc | cc93:tail93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 423, 741, 3.18e+04 | 1.4e-06 | 17 |  |
| 16 | cc | cc93:tail93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 8.4e+03, 9.83e+04 | 3.8e-07 | 11 |  |
| 16 | cc | cc93:tail93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 8.88, 18.5, 212 | 1.89e-06 | 30 |  |
| 16 | cc | cc93:tail93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 7.91, 468, 7.08e+04 | 1.6e-06 | 42 |  |
| 16 | cc | cc93:tail93 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 238, 467, 818 | 2e-05 | 6 |  |
| 16 | cc | cc93:tail93 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 854, 7.47e+03, 3.8e+04 | 8.77e-07 | 6 |  |
| 16 | cc | cc93:tail93 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 6.83, 25.9, 962 | 6.82e-07 | 36 |  |
| 16 | cc | cc93:tail93 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 6.38, 406, 1.79e+04 | 1.71e-06 | 56 |  |
| 16 | cc | cc93:tail93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 16 | cc | cc93:tail93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 16 | cc | cc93:tail93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 16 | cc | cc93:tail93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 16 | cc | cc93:tail93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 16 | cc | cc93:tail93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 16 | cc | cc93:tail93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 17.4, 89.3 | 1.79e-06 | 38 |  |
| 16 | cc | cc93:tail93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 16 | cc | cc93:tail93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 16 | cc | cc93:tail93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 14.1, 226 | 2.13e-06 | 40 |  |
| 16 | cc | cc93:tail93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.48, 84.7, 3.05e+03 | 8.54e-06 | 73 |  |
| 16 | cc | cc93:tail93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 5.52, 9.92, 155 | 3.17e-05 | 7 |  |
| 16 | cc | cc93:tail93 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 2.09, 13.6, 907 | 4.46e-06 | 73 |  |
| 16 | cc | cc93:tail93 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.7, 76.5, 4.31e+03 | 5.91e-06 | 85 |  |
| 16 | cc | cc93:tail93 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 3.29, 3.72, 4.14 | 0.000302 | 2 |  |
| 16 | cc | cc93:tail93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.84, 8.49, 206 | 2.6e-06 | 32 |  |
| 16 | cc | cc93:tail93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 5.35, 101, 6.24e+03 | 6.24e-05 | 81 |  |
| 16 | cc | cc93:tail93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.26, 4.42, 17.6 | 6.17e-06 | 7 |  |
| 16 | cc | cc93:tail93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.49, 623 | 0.000124 | 26 |  |
| 16 | cc | cc93:tail93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 6.3, 84.1, 1.17e+03 | 0.00013 | 91 |  |
| 16 | cc | cc93:tail93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 21.8 | 4.35e-05 | 3 |  |
| 16 | cc | cc93:tail93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 12.5, 6.04e+03 | 0.0004 | 38 |  |
| 16 | cc | cc93:tail93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 2.66e+04 | 0.00275 | 45 |  |
| 16 | cc | cc93:tail93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 361 | 4.52e-05 | 107 |  |
| 16 | cc | cc93:tail93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.98, 378 | 0.000135 | 55 |  |
| 16 | cc | cc93:tail93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 5.43e+03 | 0.00128 | 32 |  |
| 16 | cc | cc93:tail93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.52, 197 | 1.7e-05 | 124 |  |
| 16 | cc | cc93:tail93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.78, 250 | 2.16e-06 | 65 |  |
| 16 | cc | cc93:tail93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 16 | cc | cc93:tail93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.93, 613 | 6.44e-05 | 78 |  |
| 16 | cc | cc93:tail93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.21, 104 | 9.55e-05 | 79 |  |
| 16 | cc | cc93:tail93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 16 | cc | cc93:tail93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.79, 4.99, 471 | 3.11e-05 | 93 |  |
| 16 | levin | cc47 | analytic | band_16_20 | 1-4 | 6 | 6 | 0 | 0 | 12 | 0/0 | 15.4, 57.9, 326 | 4.54e-06 | 6 | {"bounded": 18} |
| 16 | levin | cc47 | analytic | band_16_20 | 5+ | 8 | 8 | 0 | 0 | 8 | 0/0 | 25, 56.9, 83.4 | 1.16e-05 | 8 | {"bounded": 16} |
| 16 | levin | cc47 | analytic | band_16_20 | none | 2 | 2 | 0 | 0 | 12 | 0/0 | 3.26, 3.26, 3.26 | 4.56e-07 | 2 | {"bounded": 13, "undecided": 1} |
| 16 | levin | cc47 | analytic | band_16_20 | unknown | 23 | 23 | 0 | 0 | 27 | 0/0 | 6.57, 34.5, 1.05e+03 | 6.37e-06 | 23 | {"bounded": 46, "undecided": 4} |
| 16 | levin | cc47 | analytic | band_20_32 | 1-4 | 6 | 6 | 0 | 0 | 5 | 0/0 | 17.1, 675, 6.43e+04 | 1.57e-06 | 6 | {"bounded": 11} |
| 16 | levin | cc47 | analytic | band_20_32 | 5+ | 6 | 6 | 0 | 0 | 27 | 0/0 | 12.8, 42.8, 81.6 | 2.58e-05 | 6 | {"bounded": 33} |
| 16 | levin | cc47 | analytic | band_20_32 | none | 5 | 5 | 0 | 4 | 42 | 0/0 | 1.6, 7.42, 7.42 | 1.38e-06 | 5 | {"undecided": 21, "bounded": 30} |
| 16 | levin | cc47 | analytic | band_20_32 | unknown | 12 | 12 | 0 | 6 | 85 | 0/0 | 2.67, 13.6, 353 | 1.49e-05 | 12 | {"bounded": 77, "undecided": 26} |
| 16 | levin | cc47 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 4, "bounded": 2} |
| 16 | levin | cc47 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 16 | levin | cc47 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 16 | levin | cc47 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 62, "bounded": 33, "understates": 3} |
| 16 | levin | cc47 | group | band_16_20 | 1-4 | 36 | 36 | 0 | 2 | 28 | 0/0 | 5.77, 33.1, 5.67e+03 | 1.11e-06 | 36 | {"understates": 10, "undecided": 42, "bounded": 14} |
| 16 | levin | cc47 | group | band_16_20 | 5+ | 84 | 84 | 0 | 0 | 1 | 0/0 | 13.8, 1.25e+03, 6.24e+05 | 5.32e-07 | 84 | {"undecided": 20, "bounded": 36, "understates": 29} |
| 16 | levin | cc47 | group | band_16_20 | none | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 16 | levin | cc47 | group | band_20_32 | 1-4 | 28 | 28 | 0 | 9 | 60 | 0/0 | 5.59, 16.8, 4.21e+03 | 4.33e-06 | 28 | {"undecided": 66, "bounded": 21, "understates": 10} |
| 16 | levin | cc47 | group | band_20_32 | 5+ | 96 | 96 | 0 | 0 | 5 | 0/0 | 6.43, 416, 1.28e+05 | 8.38e-06 | 96 | {"bounded": 46, "undecided": 31, "understates": 24} |
| 16 | levin | cc47 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 16 | levin | cc47 | group | band_20_32v | 1-4 | 16 | 16 | 0 | 13 | 67 | 0/0 | 2.53, 11.2, 217 | 2.8e-06 | 16 | {"undecided": 57, "understates": 8, "bounded": 31} |
| 16 | levin | cc47 | group | band_20_32v | 5+ | 68 | 68 | 0 | 0 | 34 | 0/0 | 5.73, 274, 3.32e+04 | 7.32e-06 | 68 | {"bounded": 42, "understates": 42, "undecided": 18} |
| 16 | levin | cc93 | analytic | band_16_20 | 1-4 | 7 | 7 | 0 | 0 | 11 | 0/0 | 17.5, 55.8, 83 | 6.25e-06 | 7 | {"bounded": 18} |
| 16 | levin | cc93 | analytic | band_16_20 | 5+ | 8 | 8 | 0 | 0 | 8 | 0/0 | 21.5, 81, 596 | 6.32e-06 | 8 | {"bounded": 16} |
| 16 | levin | cc93 | analytic | band_16_20 | none | 2 | 2 | 0 | 0 | 12 | 0/0 | 1.76, 1.94, 2.13 | 2.45e-06 | 2 | {"bounded": 13, "undecided": 1} |
| 16 | levin | cc93 | analytic | band_16_20 | unknown | 21 | 21 | 0 | 1 | 28 | 0/0 | 8.62, 43.2, 667 | 5.71e-06 | 21 | {"bounded": 45, "undecided": 5} |
| 16 | levin | cc93 | analytic | band_20_32 | 1-4 | 5 | 5 | 0 | 0 | 6 | 0/0 | 77.4, 89.3, 104 | 2.1e-06 | 5 | {"bounded": 11} |
| 16 | levin | cc93 | analytic | band_20_32 | 5+ | 9 | 9 | 0 | 0 | 24 | 0/0 | 33.7, 117, 451 | 4.14e-06 | 9 | {"bounded": 33} |
| 16 | levin | cc93 | analytic | band_20_32 | none | 5 | 4 | 1 | 3 | 43 | 0/0 | 0.357, 4.65, 15.1 | 2.29e-06 | 5 | {"undecided": 21, "bounded": 30} |
| 16 | levin | cc93 | analytic | band_20_32 | unknown | 21 | 21 | 0 | 6 | 76 | 0/0 | 4.07, 36, 203 | 3.09e-06 | 21 | {"bounded": 77, "undecided": 26} |
| 16 | levin | cc93 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 16 | levin | cc93 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 16 | levin | cc93 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 16 | levin | cc93 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 66, "bounded": 30, "understates": 2} |
| 16 | levin | cc93 | group | band_16_20 | 1-4 | 36 | 36 | 0 | 3 | 27 | 0/0 | 5.44, 37.5, 4.42e+03 | 1.03e-06 | 36 | {"undecided": 47, "understates": 7, "bounded": 12} |
| 16 | levin | cc93 | group | band_16_20 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 16.5, 3.89e+03, 2.24e+06 | 2.14e-07 | 85 | {"bounded": 21, "undecided": 38, "understates": 26} |
| 16 | levin | cc93 | group | band_16_20 | none | 0 | 0 | 0 | 1 | 5 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 16 | levin | cc93 | group | band_20_32 | 1-4 | 33 | 33 | 0 | 4 | 60 | 0/0 | 3.2, 23.1, 128 | 1.97e-06 | 33 | {"undecided": 79, "understates": 3, "bounded": 15} |
| 16 | levin | cc93 | group | band_20_32 | 5+ | 100 | 100 | 0 | 0 | 1 | 0/0 | 12, 618, 2.27e+05 | 6.28e-06 | 100 | {"understates": 22, "bounded": 29, "undecided": 50} |
| 16 | levin | cc93 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 16 | levin | cc93 | group | band_20_32v | 1-4 | 15 | 15 | 0 | 3 | 78 | 0/0 | 2.82, 12.9, 170 | 3.49e-06 | 15 | {"undecided": 67, "bounded": 23, "understates": 6} |
| 16 | levin | cc93 | group | band_20_32v | 5+ | 76 | 76 | 0 | 0 | 26 | 0/0 | 8.41, 1.15e+03, 4.38e+06 | 1.3e-06 | 76 | {"bounded": 37, "undecided": 33, "understates": 32} |
| 20 | cc | cc47:nest3N4 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 6.59e+06, 2.55e+07, 5.93e+07 | 0.00018 | 17 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 7.64e+06, 8.91e+07, 3.67e+09 | 6.42e-05 | 11 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 1.19e+07, 7.59e+07, 2.45e+09 | 6.01e-05 | 30 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 8.57e+06, 6.24e+07, 4.05e+08 | 5.61e-05 | 42 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 3.1e+06, 1.35e+07, 7.24e+08 | 0.136 | 6 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 2.97e+06, 9.5e+06, 2.64e+07 | 0.422 | 4 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 1.12e+06, 7.95e+06, 3.48e+08 | 0.0566 | 28 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 6.57e+05, 1.42e+07, 2.02e+08 | 0.236 | 39 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_16_20 | 1-4 | 18 | 18 | 0 | 0 | 0 | 0/0 | 5.33e+05, 6.27e+06, 2.46e+07 | 4.38 | 6 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_16_20 | 5+ | 16 | 16 | 0 | 0 | 0 | 0/0 | 2.66e+06, 3.42e+06, 1.26e+07 | 35.3 | 0 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_16_20 | none | 14 | 14 | 0 | 0 | 0 | 0/0 | 1.16e+05, 9.41e+05, 2.24e+07 | 212 | 0 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_16_20 | unknown | 50 | 50 | 0 | 0 | 0 | 0/0 | 3.43e+05, 3.66e+06, 1.24e+08 | 14.2 | 6 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.01e+03, 2.42e+03, 5.67e+05 | 7.38e-06 | 5 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.45e+04, 3.36e+05, 2.3e+06 | 7.45e-07 | 5 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 1.13e+03, 9.39e+04, 2.02e+07 | 5.8e-07 | 43 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 160, 3.03e+05, 1.23e+07 | 1.39e-06 | 39 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 9.63e+04, 4.02e+06, 1.58e+09 | 6.64e-06 | 13 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 1.61e+06, 1.02e+07, 4.5e+08 | 2.22e-05 | 6 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 3.47e+05, 2e+07, 5.8e+08 | 8.13e-06 | 38 |  |
| 20 | cc | cc47:nest3N4 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 1.37e+05, 3.41e+07, 9.11e+08 | 3.87e-06 | 47 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.72e+03, 1.53e+04 | 3.1e-07 | 4 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 34.8, 229 | 1.48e-07 | 63 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.38, 58.3, 3.97e+03 | 7.36e-07 | 33 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 375, 2.18e+03, 6.69e+04 | 4.89e-07 | 4 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.89, 31.4, 554 | 1.69e-07 | 51 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 7.33, 61.1, 1.44e+04 | 8.34e-07 | 39 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 612, 3.48e+03, 7.52e+03 | 1.3e-06 | 13 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.03e+03, 3.01e+03 | 1.97e-05 | 4 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.63, 54.3, 1.87e+05 | 5.13e-07 | 32 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.42, 414, 1.12e+04 | 6.24e-07 | 49 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 473, 2.44e+03, 5.95e+03 | 2.8e-06 | 12 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 739, 739, 739 | 2.96e-05 | 2 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12, 266, 911 | 3.5e-07 | 36 |  |
| 20 | cc | cc47:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 15.1, 172, 1.5e+03 | 1.52e-06 | 37 |  |
| 20 | cc | cc47:nest3N4 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 5.4, 187, 9.15e+04 | 3.42e-06 | 40 |  |
| 20 | cc | cc47:nest3N4 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 5.94, 433, 4.28e+04 | 0.000104 | 73 |  |
| 20 | cc | cc47:nest3N4 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 29.2, 1.79e+03, 3.88e+04 | 3.91e-06 | 7 |  |
| 20 | cc | cc47:nest3N4 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 455, 1.54e+06, 1.02e+09 | 5.31e-06 | 73 |  |
| 20 | cc | cc47:nest3N4 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 87.6, 2.58e+05, 5.23e+07 | 0.000101 | 83 |  |
| 20 | cc | cc47:nest3N4 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 670, 6.66e+04, 1.33e+05 | 0.000579 | 2 |  |
| 20 | cc | cc47:nest3N4 | group | band_16_20 | 1-4 | 66 | 66 | 0 | 0 | 0 | 0/0 | 1.4e+07, 1.07e+08, 2.92e+09 | 9.67e-05 | 66 |  |
| 20 | cc | cc47:nest3N4 | group | band_16_20 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 8.69e+04, 8.07e+07, 8.55e+10 | 7.03e-05 | 84 |  |
| 20 | cc | cc47:nest3N4 | group | band_16_20 | none | 6 | 6 | 0 | 0 | 0 | 0/0 | 2.14e+07, 3.13e+08, 7.29e+09 | 1.09e-05 | 6 |  |
| 20 | cc | cc47:nest3N4 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.43, 6.64, 2.16e+03 | 1.62e-06 | 32 |  |
| 20 | cc | cc47:nest3N4 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 1.69, 140, 2.43e+03 | 0.00104 | 80 |  |
| 20 | cc | cc47:nest3N4 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 2.1, 5.49, 47.1 | 1.3e-06 | 7 |  |
| 20 | cc | cc47:nest3N4 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 2.16, 38.3, 1.36e+03 | 0.000232 | 26 |  |
| 20 | cc | cc47:nest3N4 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 3.49, 178, 1.01e+04 | 0.00127 | 89 |  |
| 20 | cc | cc47:nest3N4 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.54, 3.54, 6.23 | 5.27e-05 | 3 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.3, 10.2, 2.03e+03 | 0.00096 | 37 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 1.57, 37.9, 934 | 0.041 | 40 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.26, 1.59, 123 | 4.84e-05 | 106 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 2.03, 6.75, 76.1 | 0.000109 | 55 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.6, 34.1, 202 | 0.0339 | 31 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.73, 4.74, 108 | 3.35e-05 | 127 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.95, 5.23, 221 | 3.86e-06 | 65 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 1.44, 41.3, 356 | 0.026 | 47 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.92, 3.94, 650 | 8.03e-05 | 78 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.51, 3.54, 51.4 | 7.31e-05 | 79 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 2.38, 47.4, 796 | 0.00112 | 26 |  |
| 20 | cc | cc47:nest3N4 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 2.03, 3.52, 217 | 2.63e-05 | 94 |  |
| 20 | cc | cc47:tree | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 6.95e+06, 1.1e+08, 3.25e+09 | 0.00018 | 17 |  |
| 20 | cc | cc47:tree | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 1.9e+06, 1.21e+09, 3.23e+10 | 6.42e-05 | 11 |  |
| 20 | cc | cc47:tree | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 8.3e+05, 2.39e+08, 2.12e+10 | 6.01e-05 | 30 |  |
| 20 | cc | cc47:tree | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 6.86e+05, 6.32e+08, 6.18e+09 | 5.61e-05 | 42 |  |
| 20 | cc | cc47:tree | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 4.01e+05, 3.42e+06, 9.56e+08 | 0.136 | 6 |  |
| 20 | cc | cc47:tree | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 4.64e+03, 3.57e+05, 4.57e+06 | 0.422 | 4 |  |
| 20 | cc | cc47:tree | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 3.66e+03, 3.82e+05, 2.63e+08 | 0.0566 | 28 |  |
| 20 | cc | cc47:tree | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 1.28e+03, 5.66e+05, 2.54e+08 | 0.236 | 39 |  |
| 20 | cc | cc47:tree | analytic | band_16_20 | 1-4 | 18 | 18 | 0 | 0 | 0 | 0/0 | 2.35e+04, 1.6e+05, 4.01e+05 | 4.38 | 6 |  |
| 20 | cc | cc47:tree | analytic | band_16_20 | 5+ | 16 | 16 | 0 | 0 | 0 | 0/0 | 4.96e+04, 8.32e+04, 3.02e+05 | 35.3 | 0 |  |
| 20 | cc | cc47:tree | analytic | band_16_20 | none | 14 | 14 | 0 | 0 | 0 | 0/0 | 1.16e+03, 2.86e+04, 1.36e+06 | 212 | 0 |  |
| 20 | cc | cc47:tree | analytic | band_16_20 | unknown | 50 | 50 | 0 | 0 | 0 | 0/0 | 1.13e+03, 8.11e+04, 3.88e+06 | 14.2 | 6 |  |
| 20 | cc | cc47:tree | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.66e+06, 7.6e+06, 2.9e+08 | 7.38e-06 | 5 |  |
| 20 | cc | cc47:tree | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 7.56e+07, 1.05e+08, 2.71e+08 | 7.45e-07 | 5 |  |
| 20 | cc | cc47:tree | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 4.33e+06, 7.99e+07, 2.59e+10 | 5.8e-07 | 43 |  |
| 20 | cc | cc47:tree | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 4.23e+05, 8.25e+07, 2.47e+09 | 1.39e-06 | 39 |  |
| 20 | cc | cc47:tree | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 2.31e+06, 1.97e+08, 5.41e+10 | 6.64e-06 | 13 |  |
| 20 | cc | cc47:tree | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 9.02e+06, 7.26e+07, 9.77e+09 | 2.22e-05 | 6 |  |
| 20 | cc | cc47:tree | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 6.38e+07, 2.82e+08, 2.38e+10 | 8.13e-06 | 38 |  |
| 20 | cc | cc47:tree | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 1.21e+07, 4.89e+08, 3e+10 | 3.87e-06 | 47 |  |
| 20 | cc | cc47:tree | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.69e+03, 1.52e+04 | 3.1e-07 | 4 |  |
| 20 | cc | cc47:tree | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 5.46, 26.7, 177 | 1.48e-07 | 63 |  |
| 20 | cc | cc47:tree | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 2.63, 45.5, 3.96e+03 | 7.36e-07 | 33 |  |
| 20 | cc | cc47:tree | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 373, 2.17e+03, 6.68e+04 | 4.89e-07 | 4 |  |
| 20 | cc | cc47:tree | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 6.96, 28, 3.93e+04 | 1.69e-07 | 51 |  |
| 20 | cc | cc47:tree | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 5.85, 48.7, 1.43e+04 | 8.34e-07 | 39 |  |
| 20 | cc | cc47:tree | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 609, 3.46e+03, 1.37e+05 | 1.3e-06 | 13 |  |
| 20 | cc | cc47:tree | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.88e+04, 6.06e+04 | 1.97e-05 | 4 |  |
| 20 | cc | cc47:tree | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.05, 2.41e+04, 2.29e+07 | 5.13e-07 | 32 |  |
| 20 | cc | cc47:tree | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.01, 2.69e+03, 1.12e+07 | 6.24e-07 | 49 |  |
| 20 | cc | cc47:tree | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 5.29e+04, 3.41e+05, 5.87e+05 | 2.8e-06 | 12 |  |
| 20 | cc | cc47:tree | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 8.23e+04, 8.23e+04, 8.23e+04 | 2.96e-05 | 2 |  |
| 20 | cc | cc47:tree | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 1.45e+04, 3.51e+05, 6.87e+06 | 3.5e-07 | 36 |  |
| 20 | cc | cc47:tree | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 7.32e+03, 7.41e+05, 1.21e+07 | 1.52e-06 | 37 |  |
| 20 | cc | cc47:tree | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 3.64e+04, 2.96e+06, 3.24e+07 | 3.42e-06 | 40 |  |
| 20 | cc | cc47:tree | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 1.46, 1.61e+05, 1.92e+07 | 0.000104 | 73 |  |
| 20 | cc | cc47:tree | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 4.63e+05, 4.9e+06, 9.87e+07 | 3.91e-06 | 7 |  |
| 20 | cc | cc47:tree | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 1.82e+06, 5.92e+07, 1.09e+09 | 5.31e-06 | 73 |  |
| 20 | cc | cc47:tree | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 1.49e+03, 8.44e+06, 1.5e+08 | 0.000101 | 83 |  |
| 20 | cc | cc47:tree | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 4.74e+06, 7.61e+06, 1.05e+07 | 0.000579 | 2 |  |
| 20 | cc | cc47:tree | group | band_16_20 | 1-4 | 66 | 66 | 0 | 0 | 0 | 0/0 | 1.06e+03, 3.31e+06, 6.31e+08 | 9.67e-05 | 66 |  |
| 20 | cc | cc47:tree | group | band_16_20 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 1.23e+03, 3.33e+06, 1.13e+10 | 7.03e-05 | 84 |  |
| 20 | cc | cc47:tree | group | band_16_20 | none | 6 | 6 | 0 | 0 | 0 | 0/0 | 7.07e+05, 9.24e+06, 7.35e+08 | 1.09e-05 | 6 |  |
| 20 | cc | cc47:tree | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 10.5, 938, 2.59e+04 | 1.62e-06 | 32 |  |
| 20 | cc | cc47:tree | group | band_6_8 | 5+ | 81 | 76 | 5 | 0 | 0 | 0/0 | 0.0385, 62.5, 2.64e+04 | 0.00104 | 80 |  |
| 20 | cc | cc47:tree | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 5.24, 102, 1.84e+03 | 1.3e-06 | 7 |  |
| 20 | cc | cc47:tree | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 264, 6.34e+04, 3.12e+06 | 0.000232 | 26 |  |
| 20 | cc | cc47:tree | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 37.3, 1.89e+03, 6.48e+06 | 0.00127 | 89 |  |
| 20 | cc | cc47:tree | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 4.18e+03, 4.18e+03, 9.48e+04 | 5.27e-05 | 3 |  |
| 20 | cc | cc47:tree | group | gate_q1 | 1-4 | 21 | 21 | 0 | 17 | 0 | 0/0 | 1.08, 9.02, 629 | 0.00261 | 20 |  |
| 20 | cc | cc47:tree | group | gate_q1 | 5+ | 45 | 27 | 18 | 0 | 0 | 0/0 | 0.0613, 4.6, 286 | 0.041 | 40 |  |
| 20 | cc | cc47:tree | group | gate_q1 | none | 7 | 7 | 0 | 107 | 0 | 0/0 | 1.28, 2.62, 20.2 | 3.56e-05 | 7 |  |
| 20 | cc | cc47:tree | group | gate_q2 | 1-4 | 10 | 9 | 1 | 45 | 0 | 0/0 | 0.776, 13.6, 26 | 0.0269 | 10 |  |
| 20 | cc | cc47:tree | group | gate_q2 | 5+ | 32 | 25 | 7 | 0 | 0 | 0/0 | 0.0137, 4.62, 47 | 0.0339 | 31 |  |
| 20 | cc | cc47:tree | group | gate_q2 | none | 11 | 11 | 0 | 116 | 0 | 0/0 | 1.84, 2.49, 6.78 | 4.24e-05 | 11 |  |
| 20 | cc | cc47:tree | group | gate_q3 | 1-4 | 10 | 10 | 0 | 55 | 0 | 0/0 | 4.22, 18.8, 67.3 | 0.0826 | 10 |  |
| 20 | cc | cc47:tree | group | gate_q3 | 5+ | 49 | 32 | 17 | 0 | 0 | 0/0 | 0.00255, 5.37, 101 | 0.026 | 47 |  |
| 20 | cc | cc47:tree | group | gate_q3 | none | 4 | 4 | 0 | 74 | 0 | 0/0 | 1.52, 2.02, 2.51 | 0.000895 | 4 |  |
| 20 | cc | cc47:tree | group | gate_q4 | 1-4 | 16 | 16 | 0 | 63 | 0 | 0/0 | 1.52, 4.29, 36.9 | 0.000454 | 16 |  |
| 20 | cc | cc47:tree | group | gate_q4 | 5+ | 27 | 23 | 4 | 0 | 0 | 0/0 | 0.171, 7.07, 392 | 0.00112 | 26 |  |
| 20 | cc | cc47:tree | group | gate_q4 | none | 7 | 7 | 0 | 87 | 0 | 0/0 | 1.7, 4.66, 13.7 | 5.82e-06 | 7 |  |
| 20 | cc | cc93:chop93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 423, 741, 3.18e+04 | 1.4e-06 | 17 |  |
| 20 | cc | cc93:chop93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 8.4e+03, 9.83e+04 | 3.8e-07 | 11 |  |
| 20 | cc | cc93:chop93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 30.9, 131, 1.69e+03 | 1.89e-06 | 30 |  |
| 20 | cc | cc93:chop93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 32.4, 684, 7.08e+04 | 1.6e-06 | 42 |  |
| 20 | cc | cc93:chop93 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 238, 467, 818 | 2e-05 | 6 |  |
| 20 | cc | cc93:chop93 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 854, 7.47e+03, 3.8e+04 | 8.77e-07 | 6 |  |
| 20 | cc | cc93:chop93 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 13.1, 57.1, 962 | 6.82e-07 | 36 |  |
| 20 | cc | cc93:chop93 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 10.3, 474, 1.79e+04 | 1.71e-06 | 56 |  |
| 20 | cc | cc93:chop93 | analytic | band_16_20 | 1-4 | 18 | 18 | 0 | 0 | 0 | 0/0 | 283, 552, 1.73e+04 | 4.07e-06 | 18 |  |
| 20 | cc | cc93:chop93 | analytic | band_16_20 | 5+ | 16 | 16 | 0 | 0 | 0 | 0/0 | 184, 877, 1.2e+04 | 8.04e-06 | 16 |  |
| 20 | cc | cc93:chop93 | analytic | band_16_20 | none | 14 | 14 | 0 | 0 | 0 | 0/0 | 6.01, 45.5, 1.59e+05 | 3.52e-06 | 14 |  |
| 20 | cc | cc93:chop93 | analytic | band_16_20 | unknown | 50 | 50 | 0 | 0 | 0 | 0/0 | 19.2, 532, 7.16e+04 | 3.55e-06 | 50 |  |
| 20 | cc | cc93:chop93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 20 | cc | cc93:chop93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 20 | cc | cc93:chop93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 38.7, 551, 2.16e+04 | 3.59e-07 | 43 |  |
| 20 | cc | cc93:chop93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 56.5, 472, 9.43e+03 | 7.91e-07 | 39 |  |
| 20 | cc | cc93:chop93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 20 | cc | cc93:chop93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 20 | cc | cc93:chop93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 26, 121, 626 | 1.79e-06 | 38 |  |
| 20 | cc | cc93:chop93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 29.9, 714, 6.21e+03 | 1.11e-06 | 47 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.59e+04, 1.59e+04, 1.59e+04 | 2.98e-07 | 4 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 154, 3.22e+03, 7.35e+06 | 2.23e-07 | 63 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 50.6, 2.92e+03, 3.51e+05 | 9.31e-07 | 33 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.73e+03, 3.49e+03 | 2.39e-06 | 4 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 108, 4.99e+03, 1.75e+05 | 9.35e-08 | 51 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 639, 4.44e+03, 2.32e+05 | 4.37e-07 | 39 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 1.01e+03, 6.03e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 71.1, 557, 1.84e+04 | 3.11e-07 | 32 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 87.4, 2.16e+03, 1.95e+04 | 6.33e-07 | 49 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 78.3, 308, 3.26e+03 | 4.09e-07 | 36 |  |
| 20 | cc | cc93:chop93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 95.2, 620, 6.89e+03 | 1.27e-06 | 37 |  |
| 20 | cc | cc93:chop93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 981, 1.12e+04 | 2.13e-06 | 40 |  |
| 20 | cc | cc93:chop93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.48, 84.7, 1.48e+04 | 8.54e-06 | 73 |  |
| 20 | cc | cc93:chop93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 179, 1.9e+03, 7.05e+03 | 3.17e-05 | 7 |  |
| 20 | cc | cc93:chop93 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 2.92, 294, 2.98e+04 | 4.46e-06 | 73 |  |
| 20 | cc | cc93:chop93 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.7, 76.5, 4.31e+03 | 5.91e-06 | 85 |  |
| 20 | cc | cc93:chop93 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 89.2, 280, 471 | 0.000302 | 2 |  |
| 20 | cc | cc93:chop93 | group | band_16_20 | 1-4 | 66 | 66 | 0 | 0 | 0 | 0/0 | 2.3, 125, 9.03e+04 | 1.82e-06 | 66 |  |
| 20 | cc | cc93:chop93 | group | band_16_20 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.33, 97.8, 7.15e+03 | 2.1e-06 | 85 |  |
| 20 | cc | cc93:chop93 | group | band_16_20 | none | 6 | 6 | 0 | 0 | 0 | 0/0 | 41.6, 278, 1.56e+03 | 6.06e-07 | 6 |  |
| 20 | cc | cc93:chop93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.44, 450, 3.99e+04 | 2.6e-06 | 32 |  |
| 20 | cc | cc93:chop93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 5.35, 101, 7.94e+05 | 6.24e-05 | 81 |  |
| 20 | cc | cc93:chop93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 184, 1.02e+03, 1.77e+03 | 6.17e-06 | 7 |  |
| 20 | cc | cc93:chop93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 4.01, 547, 3.83e+04 | 0.000124 | 26 |  |
| 20 | cc | cc93:chop93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 6.3, 84.1, 1.17e+03 | 0.00013 | 91 |  |
| 20 | cc | cc93:chop93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 123, 123, 1.51e+03 | 4.35e-05 | 3 |  |
| 20 | cc | cc93:chop93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 4.29, 1.53e+03, 1.88e+08 | 0.0004 | 38 |  |
| 20 | cc | cc93:chop93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 1.09e+07 | 0.00275 | 45 |  |
| 20 | cc | cc93:chop93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 382, 1.28e+04, 3.83e+07 | 4.52e-05 | 107 |  |
| 20 | cc | cc93:chop93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 63.2, 4.34e+03, 2.08e+05 | 0.000135 | 55 |  |
| 20 | cc | cc93:chop93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 6.59e+04 | 0.00128 | 32 |  |
| 20 | cc | cc93:chop93 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 85.2, 7.81e+03, 8.74e+05 | 1.28e-05 | 127 |  |
| 20 | cc | cc93:chop93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 32, 1.71e+03, 1.29e+05 | 2.16e-06 | 65 |  |
| 20 | cc | cc93:chop93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 20 | cc | cc93:chop93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 280, 2.36e+03, 2.39e+05 | 6.44e-05 | 78 |  |
| 20 | cc | cc93:chop93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 26.8, 1.66e+03, 4.34e+04 | 9.55e-05 | 79 |  |
| 20 | cc | cc93:chop93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 21.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 20 | cc | cc93:chop93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 65.5, 1.24e+03, 2.37e+05 | 3.11e-05 | 93 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 2.13e+07, 9.12e+09, 3.23e+11 | 1.4e-06 | 17 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 2.8e+08, 2.01e+10, 4.76e+11 | 3.8e-07 | 11 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 1.73e+08, 1.97e+09, 9.43e+10 | 1.89e-06 | 30 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 4.02e+06, 1.99e+09, 2.46e+13 | 1.6e-06 | 42 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 6.9e+09, 2.07e+11, 6e+11 | 2e-05 | 6 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 7.34e+11, 1.02e+13, 3.87e+13 | 8.77e-07 | 6 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 1.67e+09, 1.11e+12, 1.38e+14 | 6.82e-07 | 36 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 9.48e+09, 1.73e+12, 6.79e+14 | 1.71e-06 | 56 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_16_20 | 1-4 | 18 | 18 | 0 | 0 | 0 | 0/0 | 3.65e+11, 1.31e+13, 5.39e+13 | 4.07e-06 | 18 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_16_20 | 5+ | 16 | 16 | 0 | 0 | 0 | 0/0 | 1.19e+12, 1.46e+13, 3.13e+14 | 8.04e-06 | 16 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_16_20 | none | 14 | 14 | 0 | 0 | 0 | 0/0 | 1.23e+13, 7.45e+13, 9.12e+14 | 3.52e-06 | 14 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_16_20 | unknown | 50 | 50 | 0 | 0 | 0 | 0/0 | 3.4e+11, 1.54e+13, 1.57e+15 | 3.55e-06 | 50 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.51e+03, 5.22e+03, 5.94e+05 | 7.36e-06 | 5 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 4.41e+04, 4.9e+05, 5.74e+06 | 5.11e-07 | 5 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 1.41e+03, 1.06e+05, 2.98e+07 | 3.59e-07 | 43 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 541, 5.09e+05, 4.24e+07 | 7.91e-07 | 39 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 3.84e+05, 1.04e+07, 1.33e+09 | 3.57e-06 | 13 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 1.95e+06, 4.97e+08, 5.16e+09 | 2.19e-06 | 6 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 6.83e+05, 1.72e+07, 5.62e+09 | 1.79e-06 | 38 |  |
| 20 | cc | cc93:nest3N4 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 8.3e+04, 6.88e+07, 4.75e+09 | 1.11e-06 | 47 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 50.6, 229 | 2.23e-07 | 63 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.69, 56.4, 5.48e+03 | 9.31e-07 | 33 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.21, 83.1, 645 | 9.35e-08 | 51 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.6, 99.8, 5.01e+04 | 4.37e-07 | 39 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 604, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 6.91, 79.5, 1.4e+05 | 3.11e-07 | 32 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.85, 427, 9.95e+03 | 6.33e-07 | 49 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.34e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.07e+03, 1.07e+03, 1.07e+03 | 2.05e-05 | 2 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 17.4, 240, 1.23e+03 | 4.09e-07 | 36 |  |
| 20 | cc | cc93:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 10.2, 517, 3.43e+03 | 1.27e-06 | 37 |  |
| 20 | cc | cc93:nest3N4 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 3.55, 355, 2.04e+04 | 2.13e-06 | 40 |  |
| 20 | cc | cc93:nest3N4 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 110, 4.37e+03, 1.25e+05 | 8.54e-06 | 73 |  |
| 20 | cc | cc93:nest3N4 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 11, 27.5, 6.24e+04 | 3.17e-05 | 7 |  |
| 20 | cc | cc93:nest3N4 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 908, 2.83e+06, 8.41e+08 | 4.46e-06 | 73 |  |
| 20 | cc | cc93:nest3N4 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 1.36e+03, 4.12e+06, 2.59e+09 | 5.91e-06 | 85 |  |
| 20 | cc | cc93:nest3N4 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 696, 1.29e+05, 2.58e+05 | 0.000302 | 2 |  |
| 20 | cc | cc93:nest3N4 | group | band_16_20 | 1-4 | 66 | 66 | 0 | 0 | 0 | 0/0 | 3.55e+07, 7.56e+09, 2.78e+14 | 1.82e-06 | 66 |  |
| 20 | cc | cc93:nest3N4 | group | band_16_20 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 2.98e+05, 3.68e+09, 1.49e+12 | 2.1e-06 | 85 |  |
| 20 | cc | cc93:nest3N4 | group | band_16_20 | none | 6 | 6 | 0 | 0 | 0 | 0/0 | 3.51e+07, 4.73e+08, 3.07e+10 | 6.06e-07 | 6 |  |
| 20 | cc | cc93:nest3N4 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.8, 19.2, 5.82e+03 | 2.6e-06 | 32 |  |
| 20 | cc | cc93:nest3N4 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 20.4, 1.25e+03, 1.26e+05 | 6.24e-05 | 81 |  |
| 20 | cc | cc93:nest3N4 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.47, 6.85, 20.5 | 6.17e-06 | 7 |  |
| 20 | cc | cc93:nest3N4 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 2.91, 28.5, 1.72e+04 | 0.000124 | 26 |  |
| 20 | cc | cc93:nest3N4 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 52.1, 1.94e+03, 7.6e+04 | 0.00013 | 91 |  |
| 20 | cc | cc93:nest3N4 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 4.29, 4.29, 17.3 | 4.35e-05 | 3 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.29, 14, 4.99e+04 | 0.0004 | 38 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 33.7, 558, 1.48e+04 | 0.00275 | 45 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.24, 1.62, 435 | 4.5e-05 | 106 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.98, 5.68, 6.7e+03 | 0.000135 | 55 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 78.3, 835, 3.45e+04 | 0.00128 | 32 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.71, 7.16, 317 | 1.28e-05 | 127 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 2.03, 5.99, 2.58e+03 | 2.16e-06 | 65 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 9.29, 846, 3.12e+04 | 0.000486 | 49 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.97, 4.49, 446 | 6.44e-05 | 78 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.5, 4.85, 147 | 9.55e-05 | 79 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12, 1.68e+03, 1.03e+05 | 6.52e-05 | 27 |  |
| 20 | cc | cc93:nest3N4 | group | gate_q4 | none | 93 | 93 | 0 | 1 | 0 | 0/0 | 1.82, 5.16, 731 | 3.59e-05 | 92 |  |
| 20 | cc | cc93:nest93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 585, 1.39e+03, 3.18e+04 | 1.4e-06 | 17 |  |
| 20 | cc | cc93:nest93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 6.23e+04, 9.83e+04 | 3.8e-07 | 11 |  |
| 20 | cc | cc93:nest93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 11.1, 90.8, 7.91e+03 | 1.89e-06 | 30 |  |
| 20 | cc | cc93:nest93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 8.07, 684, 2.86e+06 | 1.6e-06 | 42 |  |
| 20 | cc | cc93:nest93 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 238, 2.07e+04, 1.94e+05 | 2e-05 | 6 |  |
| 20 | cc | cc93:nest93 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 2.45e+05, 1.34e+06, 2.42e+06 | 8.77e-07 | 6 |  |
| 20 | cc | cc93:nest93 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 49, 3.15e+05, 1.69e+07 | 6.82e-07 | 36 |  |
| 20 | cc | cc93:nest93 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 48.3, 1.15e+05, 8.88e+08 | 1.71e-06 | 56 |  |
| 20 | cc | cc93:nest93 | analytic | band_16_20 | 1-4 | 18 | 18 | 0 | 0 | 0 | 0/0 | 1.48e+04, 2.49e+06, 9.14e+07 | 4.07e-06 | 18 |  |
| 20 | cc | cc93:nest93 | analytic | band_16_20 | 5+ | 16 | 16 | 0 | 0 | 0 | 0/0 | 2e+05, 4.28e+06, 7.14e+07 | 8.04e-06 | 16 |  |
| 20 | cc | cc93:nest93 | analytic | band_16_20 | none | 14 | 14 | 0 | 0 | 0 | 0/0 | 2.13e+06, 6.84e+07, 3.32e+09 | 3.52e-06 | 14 |  |
| 20 | cc | cc93:nest93 | analytic | band_16_20 | unknown | 50 | 50 | 0 | 0 | 0 | 0/0 | 2.74e+03, 6.09e+06, 4.25e+09 | 3.55e-06 | 50 |  |
| 20 | cc | cc93:nest93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 20 | cc | cc93:nest93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 20 | cc | cc93:nest93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 20 | cc | cc93:nest93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 20 | cc | cc93:nest93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 20 | cc | cc93:nest93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 20 | cc | cc93:nest93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 26, 201 | 1.79e-06 | 38 |  |
| 20 | cc | cc93:nest93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 20 | cc | cc93:nest93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 20 | cc | cc93:nest93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 12.3, 224 | 2.13e-06 | 40 |  |
| 20 | cc | cc93:nest93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.25, 75.6, 2.24e+03 | 8.54e-06 | 73 |  |
| 20 | cc | cc93:nest93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 4.67, 7.73, 138 | 3.17e-05 | 7 |  |
| 20 | cc | cc93:nest93 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 2.09, 12, 824 | 4.46e-06 | 73 |  |
| 20 | cc | cc93:nest93 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.7, 69.2, 3.94e+03 | 5.91e-06 | 85 |  |
| 20 | cc | cc93:nest93 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 2.9, 3.52, 4.14 | 0.000302 | 2 |  |
| 20 | cc | cc93:nest93 | group | band_16_20 | 1-4 | 66 | 66 | 0 | 0 | 0 | 0/0 | 1.87, 84.8, 1.78e+06 | 1.82e-06 | 66 |  |
| 20 | cc | cc93:nest93 | group | band_16_20 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.7, 120, 2.85e+04 | 2.1e-06 | 85 |  |
| 20 | cc | cc93:nest93 | group | band_16_20 | none | 6 | 6 | 0 | 0 | 0 | 0/0 | 2.58, 6.01, 681 | 6.06e-07 | 6 |  |
| 20 | cc | cc93:nest93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.19, 7.74, 206 | 2.6e-06 | 32 |  |
| 20 | cc | cc93:nest93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 4.65, 89, 5.85e+03 | 6.24e-05 | 81 |  |
| 20 | cc | cc93:nest93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 2.83, 4.41, 18.3 | 6.17e-06 | 7 |  |
| 20 | cc | cc93:nest93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.48, 518 | 0.000124 | 26 |  |
| 20 | cc | cc93:nest93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 5.62, 74.4, 1.11e+03 | 0.00013 | 91 |  |
| 20 | cc | cc93:nest93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 14.9 | 4.35e-05 | 3 |  |
| 20 | cc | cc93:nest93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 5.15e+03 | 0.0004 | 38 |  |
| 20 | cc | cc93:nest93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15.2, 86.7, 1.95e+04 | 0.00275 | 45 |  |
| 20 | cc | cc93:nest93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 310 | 4.52e-05 | 107 |  |
| 20 | cc | cc93:nest93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.45, 348 | 0.000135 | 55 |  |
| 20 | cc | cc93:nest93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.01, 144, 5.57e+03 | 0.00128 | 32 |  |
| 20 | cc | cc93:nest93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.47, 201 | 1.7e-05 | 124 |  |
| 20 | cc | cc93:nest93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.35, 189 | 2.16e-06 | 65 |  |
| 20 | cc | cc93:nest93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.57, 103, 1.69e+03 | 0.000486 | 49 |  |
| 20 | cc | cc93:nest93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.11, 402 | 6.44e-05 | 78 |  |
| 20 | cc | cc93:nest93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.16, 80.8 | 9.55e-05 | 79 |  |
| 20 | cc | cc93:nest93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 143, 7.39e+03 | 6.52e-05 | 27 |  |
| 20 | cc | cc93:nest93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.53, 4.37, 448 | 3.11e-05 | 93 |  |
| 20 | cc | cc93:pair93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 423, 741, 3.18e+04 | 1.4e-06 | 17 |  |
| 20 | cc | cc93:pair93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 8.4e+03, 9.83e+04 | 3.8e-07 | 11 |  |
| 20 | cc | cc93:pair93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 8.88, 18.5, 212 | 1.89e-06 | 30 |  |
| 20 | cc | cc93:pair93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 7.91, 468, 7.08e+04 | 1.6e-06 | 42 |  |
| 20 | cc | cc93:pair93 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 238, 467, 818 | 2e-05 | 6 |  |
| 20 | cc | cc93:pair93 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 854, 7.47e+03, 3.8e+04 | 8.77e-07 | 6 |  |
| 20 | cc | cc93:pair93 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 6.83, 25.9, 962 | 6.82e-07 | 36 |  |
| 20 | cc | cc93:pair93 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 6.38, 406, 1.79e+04 | 1.71e-06 | 56 |  |
| 20 | cc | cc93:pair93 | analytic | band_16_20 | 1-4 | 18 | 18 | 0 | 0 | 0 | 0/0 | 283, 552, 1.73e+04 | 4.07e-06 | 18 |  |
| 20 | cc | cc93:pair93 | analytic | band_16_20 | 5+ | 16 | 16 | 0 | 0 | 0 | 0/0 | 184, 877, 1.2e+04 | 8.04e-06 | 16 |  |
| 20 | cc | cc93:pair93 | analytic | band_16_20 | none | 14 | 14 | 0 | 0 | 0 | 0/0 | 6.01, 45.5, 1.52e+05 | 3.52e-06 | 14 |  |
| 20 | cc | cc93:pair93 | analytic | band_16_20 | unknown | 50 | 50 | 0 | 0 | 0 | 0/0 | 19.2, 532, 6.96e+04 | 3.55e-06 | 50 |  |
| 20 | cc | cc93:pair93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 20 | cc | cc93:pair93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 20 | cc | cc93:pair93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 20 | cc | cc93:pair93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 20 | cc | cc93:pair93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 20 | cc | cc93:pair93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 20 | cc | cc93:pair93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 17.4, 89.3 | 1.79e-06 | 38 |  |
| 20 | cc | cc93:pair93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 20 | cc | cc93:pair93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 20 | cc | cc93:pair93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 12.9, 161 | 2.13e-06 | 40 |  |
| 20 | cc | cc93:pair93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.35, 57.4, 2.21e+03 | 8.54e-06 | 73 |  |
| 20 | cc | cc93:pair93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 4.82, 6.78, 138 | 3.17e-05 | 7 |  |
| 20 | cc | cc93:pair93 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 2.09, 11.5, 825 | 4.46e-06 | 73 |  |
| 20 | cc | cc93:pair93 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.7, 56.2, 2.87e+03 | 5.91e-06 | 85 |  |
| 20 | cc | cc93:pair93 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 3.53, 3.84, 4.14 | 0.000302 | 2 |  |
| 20 | cc | cc93:pair93 | group | band_16_20 | 1-4 | 66 | 66 | 0 | 0 | 0 | 0/0 | 2.01, 9.04, 4.65e+04 | 1.82e-06 | 66 |  |
| 20 | cc | cc93:pair93 | group | band_16_20 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.39, 68.4, 5.99e+03 | 2.1e-06 | 85 |  |
| 20 | cc | cc93:pair93 | group | band_16_20 | none | 5 | 5 | 0 | 1 | 0 | 0/0 | 5.64, 5.82, 83.9 | 6.64e-08 | 5 |  |
| 20 | cc | cc93:pair93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.67, 7.74, 206 | 2.6e-06 | 32 |  |
| 20 | cc | cc93:pair93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 4.65, 66.2, 5.5e+03 | 6.24e-05 | 81 |  |
| 20 | cc | cc93:pair93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.03, 4.41, 17.5 | 6.17e-06 | 7 |  |
| 20 | cc | cc93:pair93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.48, 396 | 0.000124 | 26 |  |
| 20 | cc | cc93:pair93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 5.59, 68.7, 825 | 0.00013 | 91 |  |
| 20 | cc | cc93:pair93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 14.9 | 4.35e-05 | 3 |  |
| 20 | cc | cc93:pair93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 4.11e+03 | 0.0004 | 38 |  |
| 20 | cc | cc93:pair93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 12.8, 64.1, 1.76e+04 | 0.00275 | 45 |  |
| 20 | cc | cc93:pair93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.44, 307 | 4.52e-05 | 107 |  |
| 20 | cc | cc93:pair93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 5.11, 259 | 0.000135 | 55 |  |
| 20 | cc | cc93:pair93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.25, 107, 3.71e+03 | 0.00128 | 32 |  |
| 20 | cc | cc93:pair93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.39, 196 | 1.7e-05 | 124 |  |
| 20 | cc | cc93:pair93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.66, 5.16, 189 | 2.16e-06 | 65 |  |
| 20 | cc | cc93:pair93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 4.25, 72.3, 1.21e+03 | 0.000486 | 49 |  |
| 20 | cc | cc93:pair93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 3.94, 338 | 6.44e-05 | 78 |  |
| 20 | cc | cc93:pair93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.14, 80.8 | 9.55e-05 | 79 |  |
| 20 | cc | cc93:pair93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 116, 5.23e+03 | 6.52e-05 | 27 |  |
| 20 | cc | cc93:pair93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.4, 4.27, 448 | 3.11e-05 | 93 |  |
| 20 | cc | cc93:tail93 | analytic | band_10_12 | 1-4 | 17 | 17 | 0 | 0 | 0 | 0/0 | 423, 741, 3.18e+04 | 1.4e-06 | 17 |  |
| 20 | cc | cc93:tail93 | analytic | band_10_12 | 5+ | 11 | 11 | 0 | 0 | 0 | 0/0 | 819, 8.4e+03, 9.83e+04 | 3.8e-07 | 11 |  |
| 20 | cc | cc93:tail93 | analytic | band_10_12 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 8.88, 18.5, 212 | 1.89e-06 | 30 |  |
| 20 | cc | cc93:tail93 | analytic | band_10_12 | unknown | 42 | 42 | 0 | 0 | 0 | 0/0 | 7.91, 468, 7.08e+04 | 1.6e-06 | 42 |  |
| 20 | cc | cc93:tail93 | analytic | band_12_16 | 1-4 | 6 | 6 | 0 | 0 | 0 | 0/0 | 238, 467, 818 | 2e-05 | 6 |  |
| 20 | cc | cc93:tail93 | analytic | band_12_16 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 854, 7.47e+03, 3.8e+04 | 8.77e-07 | 6 |  |
| 20 | cc | cc93:tail93 | analytic | band_12_16 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 6.83, 25.9, 962 | 6.82e-07 | 36 |  |
| 20 | cc | cc93:tail93 | analytic | band_12_16 | unknown | 56 | 56 | 0 | 0 | 0 | 0/0 | 6.38, 406, 1.79e+04 | 1.71e-06 | 56 |  |
| 20 | cc | cc93:tail93 | analytic | band_16_20 | 1-4 | 18 | 18 | 0 | 0 | 0 | 0/0 | 283, 552, 1.73e+04 | 4.07e-06 | 18 |  |
| 20 | cc | cc93:tail93 | analytic | band_16_20 | 5+ | 16 | 16 | 0 | 0 | 0 | 0/0 | 184, 877, 1.2e+04 | 8.04e-06 | 16 |  |
| 20 | cc | cc93:tail93 | analytic | band_16_20 | none | 14 | 14 | 0 | 0 | 0 | 0/0 | 6.01, 45.5, 1.59e+05 | 3.52e-06 | 14 |  |
| 20 | cc | cc93:tail93 | analytic | band_16_20 | unknown | 50 | 50 | 0 | 0 | 0 | 0/0 | 19.2, 532, 7.16e+04 | 3.55e-06 | 50 |  |
| 20 | cc | cc93:tail93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 20 | cc | cc93:tail93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 20 | cc | cc93:tail93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 20 | cc | cc93:tail93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 20 | cc | cc93:tail93 | analytic | band_8_10 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 833, 1.56e+03, 2.21e+04 | 3.57e-06 | 13 |  |
| 20 | cc | cc93:tail93 | analytic | band_8_10 | 5+ | 6 | 6 | 0 | 0 | 0 | 0/0 | 801, 3.05e+03, 9.49e+04 | 2.19e-06 | 6 |  |
| 20 | cc | cc93:tail93 | analytic | band_8_10 | none | 38 | 38 | 0 | 0 | 0 | 0/0 | 5.08, 17.4, 89.3 | 1.79e-06 | 38 |  |
| 20 | cc | cc93:tail93 | analytic | band_8_10 | unknown | 47 | 47 | 0 | 0 | 0 | 0/0 | 11.9, 85, 6.21e+03 | 1.11e-06 | 47 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 20 | cc | cc93:tail93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 20 | cc | cc93:tail93 | group | band_10_12 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.6, 14.1, 226 | 2.13e-06 | 40 |  |
| 20 | cc | cc93:tail93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 3.48, 84.7, 3.05e+03 | 8.54e-06 | 73 |  |
| 20 | cc | cc93:tail93 | group | band_10_12 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 5.52, 9.92, 155 | 3.17e-05 | 7 |  |
| 20 | cc | cc93:tail93 | group | band_12_16 | 1-4 | 73 | 73 | 0 | 0 | 0 | 0/0 | 2.09, 13.6, 907 | 4.46e-06 | 73 |  |
| 20 | cc | cc93:tail93 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.7, 76.5, 4.31e+03 | 5.91e-06 | 85 |  |
| 20 | cc | cc93:tail93 | group | band_12_16 | none | 2 | 2 | 0 | 0 | 0 | 0/0 | 3.29, 3.72, 4.14 | 0.000302 | 2 |  |
| 20 | cc | cc93:tail93 | group | band_16_20 | 1-4 | 66 | 66 | 0 | 0 | 0 | 0/0 | 2.3, 11.8, 5.87e+04 | 1.82e-06 | 66 |  |
| 20 | cc | cc93:tail93 | group | band_16_20 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.33, 88.4, 7.15e+03 | 2.1e-06 | 85 |  |
| 20 | cc | cc93:tail93 | group | band_16_20 | none | 6 | 6 | 0 | 0 | 0 | 0/0 | 3.31, 6.31, 88.5 | 6.06e-07 | 6 |  |
| 20 | cc | cc93:tail93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.84, 8.49, 206 | 2.6e-06 | 32 |  |
| 20 | cc | cc93:tail93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 5.35, 101, 6.24e+03 | 6.24e-05 | 81 |  |
| 20 | cc | cc93:tail93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.26, 4.42, 17.6 | 6.17e-06 | 7 |  |
| 20 | cc | cc93:tail93 | group | band_8_10 | 1-4 | 26 | 26 | 0 | 0 | 0 | 0/0 | 3.26, 9.49, 623 | 0.000124 | 26 |  |
| 20 | cc | cc93:tail93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 6.3, 84.1, 1.17e+03 | 0.00013 | 91 |  |
| 20 | cc | cc93:tail93 | group | band_8_10 | none | 3 | 3 | 0 | 0 | 0 | 0/0 | 3.22, 3.22, 21.8 | 4.35e-05 | 3 |  |
| 20 | cc | cc93:tail93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 12.5, 6.04e+03 | 0.0004 | 38 |  |
| 20 | cc | cc93:tail93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 2.66e+04 | 0.00275 | 45 |  |
| 20 | cc | cc93:tail93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 361 | 4.52e-05 | 107 |  |
| 20 | cc | cc93:tail93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.98, 378 | 0.000135 | 55 |  |
| 20 | cc | cc93:tail93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 5.43e+03 | 0.00128 | 32 |  |
| 20 | cc | cc93:tail93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.52, 197 | 1.7e-05 | 124 |  |
| 20 | cc | cc93:tail93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.78, 250 | 2.16e-06 | 65 |  |
| 20 | cc | cc93:tail93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 20 | cc | cc93:tail93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.93, 613 | 6.44e-05 | 78 |  |
| 20 | cc | cc93:tail93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.21, 104 | 9.55e-05 | 79 |  |
| 20 | cc | cc93:tail93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 20 | cc | cc93:tail93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.79, 4.99, 471 | 3.11e-05 | 93 |  |
| 20 | levin | cc47 | analytic | band_20_32 | 1-4 | 5 | 5 | 0 | 0 | 6 | 0/0 | 17.1, 1.3e+03, 6.43e+04 | 8.36e-07 | 5 | {"bounded": 11} |
| 20 | levin | cc47 | analytic | band_20_32 | 5+ | 3 | 3 | 0 | 0 | 30 | 0/0 | 12.8, 51.5, 81.6 | 5.2e-06 | 3 | {"bounded": 33} |
| 20 | levin | cc47 | analytic | band_20_32 | none | 4 | 4 | 0 | 2 | 45 | 0/0 | 1.6, 7.42, 7.42 | 7.56e-06 | 4 | {"bounded": 40, "undecided": 11} |
| 20 | levin | cc47 | analytic | band_20_32 | unknown | 5 | 5 | 0 | 3 | 95 | 0/0 | 5.39, 10.3, 14.7 | 2.25e-05 | 5 | {"bounded": 93, "undecided": 10} |
| 20 | levin | cc47 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 20 | levin | cc47 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 29, "undecided": 2, "understates": 3} |
| 20 | levin | cc47 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 19, "undecided": 33} |
| 20 | levin | cc47 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"bounded": 50, "undecided": 48} |
| 20 | levin | cc47 | group | band_20_32 | 1-4 | 28 | 28 | 0 | 9 | 60 | 0/0 | 5.59, 16.8, 4.21e+03 | 4.33e-06 | 28 | {"undecided": 66, "bounded": 21, "understates": 10} |
| 20 | levin | cc47 | group | band_20_32 | 5+ | 96 | 96 | 0 | 0 | 5 | 0/0 | 6.43, 416, 1.28e+05 | 8.38e-06 | 96 | {"bounded": 45, "undecided": 31, "understates": 25} |
| 20 | levin | cc47 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 20 | levin | cc47 | group | band_20_32v | 1-4 | 16 | 16 | 0 | 13 | 67 | 0/0 | 2.53, 11.2, 217 | 2.8e-06 | 16 | {"undecided": 57, "understates": 8, "bounded": 31} |
| 20 | levin | cc47 | group | band_20_32v | 5+ | 79 | 79 | 0 | 0 | 23 | 0/0 | 5.73, 360, 8.23e+04 | 8.24e-06 | 79 | {"bounded": 43, "understates": 40, "undecided": 19} |
| 20 | levin | cc93 | analytic | band_20_32 | 1-4 | 5 | 5 | 0 | 0 | 6 | 0/0 | 77.4, 89.3, 104 | 2.1e-06 | 5 | {"bounded": 11} |
| 20 | levin | cc93 | analytic | band_20_32 | 5+ | 9 | 9 | 0 | 0 | 24 | 0/0 | 33.7, 117, 451 | 4.14e-06 | 9 | {"bounded": 33} |
| 20 | levin | cc93 | analytic | band_20_32 | none | 3 | 2 | 1 | 3 | 45 | 0/0 | 0.357, 5.57, 7.11 | 2.22e-05 | 3 | {"undecided": 21, "bounded": 30} |
| 20 | levin | cc93 | analytic | band_20_32 | unknown | 24 | 24 | 0 | 5 | 74 | 0/0 | 2.49, 31.1, 203 | 1.86e-06 | 24 | {"bounded": 77, "undecided": 26} |
| 20 | levin | cc93 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 4, "bounded": 2} |
| 20 | levin | cc93 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 20 | levin | cc93 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 38, "understates": 4} |
| 20 | levin | cc93 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 62, "bounded": 35, "understates": 1} |
| 20 | levin | cc93 | group | band_20_32 | 1-4 | 33 | 33 | 0 | 4 | 60 | 0/0 | 3.2, 23.1, 128 | 1.97e-06 | 33 | {"undecided": 79, "understates": 3, "bounded": 15} |
| 20 | levin | cc93 | group | band_20_32 | 5+ | 100 | 100 | 0 | 0 | 1 | 0/0 | 12, 618, 2.27e+05 | 6.28e-06 | 100 | {"understates": 22, "bounded": 28, "undecided": 51} |
| 20 | levin | cc93 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 20 | levin | cc93 | group | band_20_32v | 1-4 | 15 | 15 | 0 | 3 | 78 | 0/0 | 2.82, 12.9, 170 | 3.49e-06 | 15 | {"undecided": 67, "bounded": 23, "understates": 6} |
| 20 | levin | cc93 | group | band_20_32v | 5+ | 89 | 89 | 0 | 0 | 13 | 0/0 | 8.41, 1.33e+03, 4.38e+06 | 1.45e-06 | 89 | {"bounded": 37, "undecided": 34, "understates": 31} |
| 4 | cc | cc47:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.72e+03, 1.53e+04 | 3.1e-07 | 4 |  |
| 4 | cc | cc47:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 34.8, 229 | 1.48e-07 | 63 |  |
| 4 | cc | cc47:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.38, 58.3, 3.97e+03 | 7.36e-07 | 33 |  |
| 4 | cc | cc47:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 375, 2.18e+03, 6.69e+04 | 4.89e-07 | 4 |  |
| 4 | cc | cc47:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.89, 31.4, 554 | 1.69e-07 | 51 |  |
| 4 | cc | cc47:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 7.33, 61.1, 1.44e+04 | 8.34e-07 | 39 |  |
| 4 | cc | cc47:nest3N4 | analytic | gate_q3 | 1-4 | 8 | 8 | 0 | 0 | 0 | 0/0 | 612, 2.84e+03, 7.52e+03 | 9.98e-07 | 8 |  |
| 4 | cc | cc47:nest3N4 | analytic | gate_q3 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.36e+03, 1.63e+03, 1.91e+03 | 2.81e-05 | 2 |  |
| 4 | cc | cc47:nest3N4 | analytic | gate_q3 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 5.63, 38.3, 6.96e+03 | 6.6e-07 | 30 |  |
| 4 | cc | cc47:nest3N4 | analytic | gate_q3 | unknown | 45 | 45 | 0 | 0 | 0 | 0/0 | 6.42, 414, 1.12e+04 | 6.19e-07 | 45 |  |
| 4 | cc | cc47:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.3, 10.2, 2.03e+03 | 0.00096 | 37 |  |
| 4 | cc | cc47:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 1.57, 37.9, 934 | 0.041 | 40 |  |
| 4 | cc | cc47:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.26, 1.59, 123 | 4.84e-05 | 106 |  |
| 4 | cc | cc47:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 2.03, 6.75, 76.1 | 0.000109 | 55 |  |
| 4 | cc | cc47:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.6, 34.1, 202 | 0.0339 | 31 |  |
| 4 | cc | cc47:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.73, 4.74, 108 | 3.35e-05 | 127 |  |
| 4 | cc | cc47:nest3N4 | group | gate_q3 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.09, 6.16, 157 | 1.33e-06 | 40 |  |
| 4 | cc | cc47:nest3N4 | group | gate_q3 | 5+ | 31 | 31 | 0 | 0 | 0 | 0/0 | 2.46, 47, 356 | 0.0272 | 30 |  |
| 4 | cc | cc47:nest3N4 | group | gate_q3 | none | 44 | 44 | 0 | 0 | 0 | 0/0 | 1.92, 3.96, 650 | 3.22e-05 | 44 |  |
| 4 | cc | cc47:tree | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.69e+03, 1.52e+04 | 3.1e-07 | 4 |  |
| 4 | cc | cc47:tree | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 5.46, 26.7, 177 | 1.48e-07 | 63 |  |
| 4 | cc | cc47:tree | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 2.63, 45.5, 3.96e+03 | 7.36e-07 | 33 |  |
| 4 | cc | cc47:tree | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 373, 2.17e+03, 6.68e+04 | 4.89e-07 | 4 |  |
| 4 | cc | cc47:tree | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 6.96, 28, 3.93e+04 | 1.69e-07 | 51 |  |
| 4 | cc | cc47:tree | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 5.85, 48.7, 1.43e+04 | 8.34e-07 | 39 |  |
| 4 | cc | cc47:tree | analytic | gate_q3 | 1-4 | 8 | 8 | 0 | 0 | 0 | 0/0 | 609, 2.83e+03, 7.48e+03 | 9.98e-07 | 8 |  |
| 4 | cc | cc47:tree | analytic | gate_q3 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.36e+03, 1.63e+03, 1.9e+03 | 2.81e-05 | 2 |  |
| 4 | cc | cc47:tree | analytic | gate_q3 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 5.05, 1.57e+04, 7.89e+06 | 6.6e-07 | 30 |  |
| 4 | cc | cc47:tree | analytic | gate_q3 | unknown | 45 | 45 | 0 | 0 | 0 | 0/0 | 5.01, 2.69e+03, 6.64e+05 | 6.19e-07 | 45 |  |
| 4 | cc | cc47:tree | group | gate_q1 | 1-4 | 21 | 21 | 0 | 17 | 0 | 0/0 | 1.08, 9.02, 629 | 0.00261 | 20 |  |
| 4 | cc | cc47:tree | group | gate_q1 | 5+ | 45 | 27 | 18 | 0 | 0 | 0/0 | 0.0613, 4.6, 286 | 0.041 | 40 |  |
| 4 | cc | cc47:tree | group | gate_q1 | none | 7 | 7 | 0 | 107 | 0 | 0/0 | 1.28, 2.62, 20.2 | 3.56e-05 | 7 |  |
| 4 | cc | cc47:tree | group | gate_q2 | 1-4 | 10 | 9 | 1 | 45 | 0 | 0/0 | 0.776, 13.6, 26 | 0.0269 | 10 |  |
| 4 | cc | cc47:tree | group | gate_q2 | 5+ | 32 | 25 | 7 | 0 | 0 | 0/0 | 0.0137, 4.62, 47 | 0.0339 | 31 |  |
| 4 | cc | cc47:tree | group | gate_q2 | none | 11 | 11 | 0 | 116 | 0 | 0/0 | 1.84, 2.49, 6.78 | 4.24e-05 | 11 |  |
| 4 | cc | cc47:tree | group | gate_q3 | 1-4 | 5 | 5 | 0 | 35 | 0 | 0/0 | 4.22, 18.8, 33.1 | 0.0938 | 5 |  |
| 4 | cc | cc47:tree | group | gate_q3 | 5+ | 31 | 23 | 8 | 0 | 0 | 0/0 | 0.00255, 9.58, 46.6 | 0.0272 | 30 |  |
| 4 | cc | cc47:tree | group | gate_q3 | none | 2 | 2 | 0 | 42 | 0 | 0/0 | 2.51, 2.51, 2.51 | 3.73e-06 | 2 |  |
| 4 | cc | cc93:chop93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.59e+04, 1.59e+04, 1.59e+04 | 2.98e-07 | 4 |  |
| 4 | cc | cc93:chop93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 154, 3.22e+03, 7.35e+06 | 2.23e-07 | 63 |  |
| 4 | cc | cc93:chop93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 50.6, 2.92e+03, 3.51e+05 | 9.31e-07 | 33 |  |
| 4 | cc | cc93:chop93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.73e+03, 3.49e+03 | 2.39e-06 | 4 |  |
| 4 | cc | cc93:chop93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 108, 4.99e+03, 1.75e+05 | 9.35e-08 | 51 |  |
| 4 | cc | cc93:chop93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 639, 4.44e+03, 2.32e+05 | 4.37e-07 | 39 |  |
| 4 | cc | cc93:chop93 | analytic | gate_q3 | 1-4 | 8 | 8 | 0 | 0 | 0 | 0/0 | 1.01e+03, 5.66e+03, 1.25e+04 | 3.68e-07 | 8 |  |
| 4 | cc | cc93:chop93 | analytic | gate_q3 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 2.37e+03, 2.39e+03, 2.41e+03 | 1.66e-05 | 2 |  |
| 4 | cc | cc93:chop93 | analytic | gate_q3 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 71.1, 454, 1.84e+04 | 3.35e-07 | 30 |  |
| 4 | cc | cc93:chop93 | analytic | gate_q3 | unknown | 45 | 45 | 0 | 0 | 0 | 0/0 | 87.4, 2.16e+03, 1.68e+04 | 6e-07 | 45 |  |
| 4 | cc | cc93:chop93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 4.29, 1.53e+03, 1.88e+08 | 0.0004 | 38 |  |
| 4 | cc | cc93:chop93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 1.09e+07 | 0.00275 | 45 |  |
| 4 | cc | cc93:chop93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 382, 1.28e+04, 3.83e+07 | 4.52e-05 | 107 |  |
| 4 | cc | cc93:chop93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 63.2, 4.34e+03, 2.08e+05 | 0.000135 | 55 |  |
| 4 | cc | cc93:chop93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 6.59e+04 | 0.00128 | 32 |  |
| 4 | cc | cc93:chop93 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 85.2, 7.81e+03, 8.74e+05 | 1.28e-05 | 127 |  |
| 4 | cc | cc93:chop93 | group | gate_q3 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 65.4, 2.39e+03, 1.29e+05 | 1.07e-06 | 40 |  |
| 4 | cc | cc93:chop93 | group | gate_q3 | 5+ | 31 | 31 | 0 | 0 | 0 | 0/0 | 7.12, 74.9, 997 | 0.00066 | 31 |  |
| 4 | cc | cc93:chop93 | group | gate_q3 | none | 44 | 44 | 0 | 0 | 0 | 0/0 | 280, 2.78e+03, 2.39e+05 | 3.45e-05 | 44 |  |
| 4 | cc | cc93:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 4 | cc | cc93:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 50.6, 229 | 2.23e-07 | 63 |  |
| 4 | cc | cc93:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.69, 56.4, 5.48e+03 | 9.31e-07 | 33 |  |
| 4 | cc | cc93:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 4 | cc | cc93:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.21, 83.1, 645 | 9.35e-08 | 51 |  |
| 4 | cc | cc93:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.6, 99.8, 5.01e+04 | 4.37e-07 | 39 |  |
| 4 | cc | cc93:nest3N4 | analytic | gate_q3 | 1-4 | 8 | 8 | 0 | 0 | 0 | 0/0 | 604, 3.37e+03, 1.25e+04 | 3.68e-07 | 8 |  |
| 4 | cc | cc93:nest3N4 | analytic | gate_q3 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 2.37e+03, 2.39e+03, 2.41e+03 | 1.66e-05 | 2 |  |
| 4 | cc | cc93:nest3N4 | analytic | gate_q3 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 6.91, 63.5, 6.26e+04 | 3.35e-07 | 30 |  |
| 4 | cc | cc93:nest3N4 | analytic | gate_q3 | unknown | 45 | 45 | 0 | 0 | 0 | 0/0 | 5.85, 427, 8.65e+03 | 6e-07 | 45 |  |
| 4 | cc | cc93:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.29, 14, 4.99e+04 | 0.0004 | 38 |  |
| 4 | cc | cc93:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 33.7, 558, 1.48e+04 | 0.00275 | 45 |  |
| 4 | cc | cc93:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.24, 1.62, 435 | 4.5e-05 | 106 |  |
| 4 | cc | cc93:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.98, 5.68, 6.7e+03 | 0.000135 | 55 |  |
| 4 | cc | cc93:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 78.3, 835, 3.45e+04 | 0.00128 | 32 |  |
| 4 | cc | cc93:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.71, 7.16, 317 | 1.28e-05 | 127 |  |
| 4 | cc | cc93:nest3N4 | group | gate_q3 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 2.03, 8.42, 1.77e+03 | 1.07e-06 | 40 |  |
| 4 | cc | cc93:nest3N4 | group | gate_q3 | 5+ | 31 | 31 | 0 | 0 | 0 | 0/0 | 9.29, 846, 8.62e+03 | 0.00066 | 31 |  |
| 4 | cc | cc93:nest3N4 | group | gate_q3 | none | 44 | 44 | 0 | 0 | 0 | 0/0 | 1.97, 4.24, 247 | 3.45e-05 | 44 |  |
| 4 | cc | cc93:nest93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 4 | cc | cc93:nest93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 4 | cc | cc93:nest93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 4 | cc | cc93:nest93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 4 | cc | cc93:nest93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 4 | cc | cc93:nest93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 4 | cc | cc93:nest93 | analytic | gate_q3 | 1-4 | 8 | 8 | 0 | 0 | 0 | 0/0 | 603, 3.37e+03, 1.25e+04 | 3.68e-07 | 8 |  |
| 4 | cc | cc93:nest93 | analytic | gate_q3 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 2.37e+03, 2.39e+03, 2.41e+03 | 1.66e-05 | 2 |  |
| 4 | cc | cc93:nest93 | analytic | gate_q3 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.35e-07 | 30 |  |
| 4 | cc | cc93:nest93 | analytic | gate_q3 | unknown | 45 | 45 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6e-07 | 45 |  |
| 4 | cc | cc93:nest93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 5.15e+03 | 0.0004 | 38 |  |
| 4 | cc | cc93:nest93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15.2, 86.7, 1.95e+04 | 0.00275 | 45 |  |
| 4 | cc | cc93:nest93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 310 | 4.52e-05 | 107 |  |
| 4 | cc | cc93:nest93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.45, 348 | 0.000135 | 55 |  |
| 4 | cc | cc93:nest93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.01, 144, 5.57e+03 | 0.00128 | 32 |  |
| 4 | cc | cc93:nest93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.47, 201 | 1.7e-05 | 124 |  |
| 4 | cc | cc93:nest93 | group | gate_q3 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 1.88, 6.42, 189 | 1.07e-06 | 40 |  |
| 4 | cc | cc93:nest93 | group | gate_q3 | 5+ | 31 | 31 | 0 | 0 | 0 | 0/0 | 6.57, 70.2, 919 | 0.00066 | 31 |  |
| 4 | cc | cc93:nest93 | group | gate_q3 | none | 44 | 44 | 0 | 0 | 0 | 0/0 | 1.6, 3.9, 160 | 3.45e-05 | 44 |  |
| 4 | cc | cc93:pair93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 4 | cc | cc93:pair93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 4 | cc | cc93:pair93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 4 | cc | cc93:pair93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 4 | cc | cc93:pair93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 4 | cc | cc93:pair93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 4 | cc | cc93:pair93 | analytic | gate_q3 | 1-4 | 8 | 8 | 0 | 0 | 0 | 0/0 | 603, 3.37e+03, 1.25e+04 | 3.68e-07 | 8 |  |
| 4 | cc | cc93:pair93 | analytic | gate_q3 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 2.37e+03, 2.39e+03, 2.41e+03 | 1.66e-05 | 2 |  |
| 4 | cc | cc93:pair93 | analytic | gate_q3 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.35e-07 | 30 |  |
| 4 | cc | cc93:pair93 | analytic | gate_q3 | unknown | 45 | 45 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6e-07 | 45 |  |
| 4 | cc | cc93:pair93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 4.11e+03 | 0.0004 | 38 |  |
| 4 | cc | cc93:pair93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 12.8, 64.1, 1.76e+04 | 0.00275 | 45 |  |
| 4 | cc | cc93:pair93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.44, 307 | 4.52e-05 | 107 |  |
| 4 | cc | cc93:pair93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 5.11, 259 | 0.000135 | 55 |  |
| 4 | cc | cc93:pair93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.25, 107, 3.71e+03 | 0.00128 | 32 |  |
| 4 | cc | cc93:pair93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.39, 196 | 1.7e-05 | 124 |  |
| 4 | cc | cc93:pair93 | group | gate_q3 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 1.88, 6.16, 189 | 1.07e-06 | 40 |  |
| 4 | cc | cc93:pair93 | group | gate_q3 | 5+ | 31 | 31 | 0 | 0 | 0 | 0/0 | 4.25, 55.2, 790 | 0.00066 | 31 |  |
| 4 | cc | cc93:pair93 | group | gate_q3 | none | 44 | 44 | 0 | 0 | 0 | 0/0 | 1.6, 3.9, 160 | 3.45e-05 | 44 |  |
| 4 | cc | cc93:tail93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 4 | cc | cc93:tail93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 4 | cc | cc93:tail93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 4 | cc | cc93:tail93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 4 | cc | cc93:tail93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 4 | cc | cc93:tail93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 4 | cc | cc93:tail93 | analytic | gate_q3 | 1-4 | 8 | 8 | 0 | 0 | 0 | 0/0 | 603, 3.37e+03, 1.25e+04 | 3.68e-07 | 8 |  |
| 4 | cc | cc93:tail93 | analytic | gate_q3 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 2.37e+03, 2.39e+03, 2.41e+03 | 1.66e-05 | 2 |  |
| 4 | cc | cc93:tail93 | analytic | gate_q3 | none | 30 | 30 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.35e-07 | 30 |  |
| 4 | cc | cc93:tail93 | analytic | gate_q3 | unknown | 45 | 45 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6e-07 | 45 |  |
| 4 | cc | cc93:tail93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 12.5, 6.04e+03 | 0.0004 | 38 |  |
| 4 | cc | cc93:tail93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 2.66e+04 | 0.00275 | 45 |  |
| 4 | cc | cc93:tail93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 361 | 4.52e-05 | 107 |  |
| 4 | cc | cc93:tail93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.98, 378 | 0.000135 | 55 |  |
| 4 | cc | cc93:tail93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 5.43e+03 | 0.00128 | 32 |  |
| 4 | cc | cc93:tail93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.52, 197 | 1.7e-05 | 124 |  |
| 4 | cc | cc93:tail93 | group | gate_q3 | 1-4 | 40 | 40 | 0 | 0 | 0 | 0/0 | 1.88, 7.4, 250 | 1.07e-06 | 40 |  |
| 4 | cc | cc93:tail93 | group | gate_q3 | 5+ | 31 | 31 | 0 | 0 | 0 | 0/0 | 7.12, 74.9, 997 | 0.00066 | 31 |  |
| 4 | cc | cc93:tail93 | group | gate_q3 | none | 44 | 44 | 0 | 0 | 0 | 0/0 | 1.6, 4.63, 186 | 3.45e-05 | 44 |  |
| 4 | levin | cc47 | analytic | band_10_12 | 1-4 | 12 | 12 | 0 | 0 | 5 | 0/0 | 368, 716, 2.24e+03 | 5.67e-07 | 12 | {"bounded": 17} |
| 4 | levin | cc47 | analytic | band_10_12 | 5+ | 3 | 3 | 0 | 0 | 8 | 0/0 | 24.4, 24.4, 1.49e+04 | 7.24e-06 | 3 | {"bounded": 11} |
| 4 | levin | cc47 | analytic | band_10_12 | none | 0 | 0 | 0 | 0 | 30 | 0/0 | —, —, — | — | 0 | {"bounded": 30} |
| 4 | levin | cc47 | analytic | band_10_12 | unknown | 20 | 20 | 0 | 0 | 22 | 0/0 | 4.97, 82, 195 | 1.07e-06 | 20 | {"bounded": 42} |
| 4 | levin | cc47 | analytic | band_12_16 | 1-4 | 1 | 1 | 0 | 0 | 5 | 0/0 | 68.6, 68.6, 68.6 | 2.08e-05 | 1 | {"bounded": 6} |
| 4 | levin | cc47 | analytic | band_12_16 | 5+ | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 4 | levin | cc47 | analytic | band_12_16 | none | 7 | 7 | 0 | 2 | 27 | 0/0 | 3.85, 33.3, 148 | 8.34e-08 | 7 | {"bounded": 33, "undecided": 3} |
| 4 | levin | cc47 | analytic | band_12_16 | unknown | 6 | 6 | 0 | 0 | 50 | 0/0 | 3.97, 26.6, 1.39e+03 | 2.9e-06 | 6 | {"bounded": 53, "undecided": 3} |
| 4 | levin | cc47 | analytic | band_16_20 | 1-4 | 4 | 4 | 0 | 0 | 14 | 0/0 | 6.56, 52.4, 151 | 1.1e-05 | 4 | {"bounded": 18} |
| 4 | levin | cc47 | analytic | band_16_20 | 5+ | 6 | 6 | 0 | 0 | 10 | 0/0 | 17.8, 53, 313 | 6.19e-06 | 6 | {"bounded": 16} |
| 4 | levin | cc47 | analytic | band_16_20 | none | 0 | 0 | 0 | 0 | 14 | 0/0 | —, —, — | — | 0 | {"bounded": 13, "undecided": 1} |
| 4 | levin | cc47 | analytic | band_16_20 | unknown | 11 | 11 | 0 | 1 | 38 | 0/0 | 11.4, 24.4, 371 | 4.26e-06 | 11 | {"bounded": 45, "undecided": 5} |
| 4 | levin | cc47 | analytic | band_20_32 | 1-4 | 2 | 2 | 0 | 0 | 9 | 0/0 | 1.15e+03, 1.15e+03, 1.15e+03 | 2.16e-07 | 2 | {"bounded": 11} |
| 4 | levin | cc47 | analytic | band_20_32 | 5+ | 6 | 6 | 0 | 0 | 27 | 0/0 | 14.7, 131, 175 | 6.7e-06 | 6 | {"bounded": 33} |
| 4 | levin | cc47 | analytic | band_20_32 | none | 1 | 1 | 0 | 5 | 45 | 0/0 | 44, 44, 44 | 1.13e-07 | 1 | {"undecided": 21, "bounded": 30} |
| 4 | levin | cc47 | analytic | band_20_32 | unknown | 12 | 12 | 0 | 6 | 85 | 0/0 | 8.79, 27.3, 265 | 1.39e-05 | 12 | {"bounded": 77, "undecided": 26} |
| 4 | levin | cc47 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 4 | levin | cc47 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 4 | levin | cc47 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 4 | levin | cc47 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 66, "bounded": 32} |
| 4 | levin | cc47 | analytic | band_6_8 | 1-4 | 3 | 3 | 0 | 0 | 2 | 0/0 | 153, 169, 203 | 6.96e-06 | 3 | {"bounded": 5} |
| 4 | levin | cc47 | analytic | band_6_8 | 5+ | 1 | 1 | 0 | 0 | 4 | 0/0 | 1.18e+03, 1.18e+03, 1.18e+03 | 5.74e-06 | 1 | {"bounded": 5} |
| 4 | levin | cc47 | analytic | band_6_8 | none | 6 | 6 | 0 | 0 | 37 | 0/0 | 4.58, 27.1, 222 | 1.06e-06 | 6 | {"bounded": 43} |
| 4 | levin | cc47 | analytic | band_6_8 | unknown | 13 | 13 | 0 | 0 | 26 | 0/0 | 9.73, 18.4, 4.67e+03 | 5.35e-07 | 13 | {"bounded": 39} |
| 4 | levin | cc47 | analytic | band_8_10 | 1-4 | 5 | 5 | 0 | 0 | 8 | 0/0 | 157, 922, 925 | 1.12e-05 | 5 | {"bounded": 13} |
| 4 | levin | cc47 | analytic | band_8_10 | 5+ | 3 | 3 | 0 | 0 | 3 | 0/0 | 65.5, 738, 738 | 1.33e-06 | 3 | {"bounded": 6} |
| 4 | levin | cc47 | analytic | band_8_10 | none | 10 | 10 | 0 | 0 | 28 | 0/0 | 5.3, 9.27, 765 | 1.2e-06 | 10 | {"bounded": 38} |
| 4 | levin | cc47 | analytic | band_8_10 | unknown | 10 | 10 | 0 | 0 | 37 | 0/0 | 54.6, 185, 822 | 1.03e-06 | 10 | {"bounded": 47} |
| 4 | levin | cc47 | analytic | gate_q3 | 1-4 | 4 | 4 | 0 | 0 | 1 | 0/0 | 364, 929, 1.5e+03 | 4.9e-06 | 4 | {"bounded": 5} |
| 4 | levin | cc47 | analytic | gate_q3 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 299, 367, 435 | 2.48e-05 | 2 | {"bounded": 2} |
| 4 | levin | cc47 | analytic | gate_q3 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"bounded": 2} |
| 4 | levin | cc47 | analytic | gate_q3 | unknown | 2 | 2 | 0 | 0 | 2 | 0/0 | 17.2, 206, 394 | 3.43e-06 | 2 | {"bounded": 4} |
| 4 | levin | cc47 | analytic | gate_q4 | 1-4 | 6 | 6 | 0 | 0 | 6 | 0/0 | 77.7, 478, 478 | 5.18e-06 | 6 | {"bounded": 12} |
| 4 | levin | cc47 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 939, 939, 939 | 4.16e-06 | 2 | {"bounded": 2} |
| 4 | levin | cc47 | analytic | gate_q4 | none | 27 | 27 | 0 | 0 | 9 | 0/0 | 7.49, 22.8, 213 | 4.63e-07 | 27 | {"bounded": 36} |
| 4 | levin | cc47 | analytic | gate_q4 | unknown | 25 | 25 | 0 | 0 | 12 | 0/0 | 5.31, 47.9, 900 | 1.54e-06 | 25 | {"bounded": 37} |
| 4 | levin | cc47 | group | band_10_12 | 1-4 | 18 | 18 | 0 | 0 | 22 | 0/0 | 4.06, 28.5, 1.5e+04 | 1.34e-05 | 18 | {"undecided": 34, "understates": 5, "bounded": 1} |
| 4 | levin | cc47 | group | band_10_12 | 5+ | 64 | 64 | 0 | 0 | 9 | 0/0 | 38.6, 6.63e+03, 2e+06 | 1.16e-06 | 64 | {"bounded": 44, "undecided": 3, "understates": 26} |
| 4 | levin | cc47 | group | band_10_12 | none | 0 | 0 | 0 | 0 | 7 | 0/0 | —, —, — | — | 0 | {"undecided": 7} |
| 4 | levin | cc47 | group | band_12_16 | 1-4 | 54 | 54 | 0 | 4 | 15 | 0/0 | 3.44, 36.3, 1.61e+04 | 2.21e-06 | 54 | {"undecided": 63, "understates": 5, "bounded": 5} |
| 4 | levin | cc47 | group | band_12_16 | 5+ | 81 | 81 | 0 | 0 | 4 | 0/0 | 9.9, 1.06e+03, 1.72e+06 | 1.21e-06 | 81 | {"undecided": 5, "understates": 45, "bounded": 35} |
| 4 | levin | cc47 | group | band_12_16 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"undecided": 2} |
| 4 | levin | cc47 | group | band_16_20 | 1-4 | 45 | 45 | 0 | 11 | 10 | 0/0 | 7.14, 131, 5.77e+04 | 3.33e-07 | 45 | {"undecided": 57, "understates": 3, "bounded": 6} |
| 4 | levin | cc47 | group | band_16_20 | 5+ | 84 | 84 | 0 | 0 | 1 | 0/0 | 42.2, 590, 7.72e+04 | 1.18e-06 | 84 | {"bounded": 36, "understates": 42, "undecided": 7} |
| 4 | levin | cc47 | group | band_16_20 | none | 0 | 0 | 0 | 1 | 5 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 4 | levin | cc47 | group | band_20_32 | 1-4 | 45 | 43 | 2 | 34 | 18 | 0/0 | 0.0013, 51.8, 1.11e+03 | 4.97e-07 | 45 | {"undecided": 85, "understates": 8, "bounded": 4} |
| 4 | levin | cc47 | group | band_20_32 | 5+ | 101 | 101 | 0 | 0 | 0 | 0/0 | 18.1, 334, 1.42e+04 | 8.75e-06 | 101 | {"bounded": 48, "understates": 42, "undecided": 11} |
| 4 | levin | cc47 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 4 | levin | cc47 | group | band_20_32v | 1-4 | 13 | 13 | 0 | 29 | 54 | 0/0 | 5.07, 22.1, 88.8 | 1.33e-07 | 13 | {"bounded": 15, "undecided": 73, "understates": 8} |
| 4 | levin | cc47 | group | band_20_32v | 5+ | 9 | 9 | 0 | 0 | 93 | 0/0 | 1.5, 326, 7.36e+03 | 0.00293 | 9 | {"bounded": 65, "understates": 35, "undecided": 2} |
| 4 | levin | cc47 | group | band_6_8 | 1-4 | 16 | 15 | 1 | 13 | 3 | 0/0 | 0.116, 43.7, 3.21e+05 | 2.77e-05 | 16 | {"bounded": 9, "understates": 6, "undecided": 17} |
| 4 | levin | cc47 | group | band_6_8 | 5+ | 74 | 74 | 0 | 0 | 7 | 0/0 | 36.7, 4.08e+03, 6.13e+05 | 2.59e-06 | 74 | {"bounded": 35, "understates": 39, "undecided": 7} |
| 4 | levin | cc47 | group | band_6_8 | none | 0 | 0 | 0 | 6 | 1 | 0/0 | —, —, — | — | 0 | {"bounded": 5, "undecided": 2} |
| 4 | levin | cc47 | group | band_8_10 | 1-4 | 11 | 11 | 0 | 1 | 14 | 0/0 | 37.4, 399, 1.07e+05 | 2.68e-05 | 11 | {"understates": 7, "undecided": 15, "bounded": 4} |
| 4 | levin | cc47 | group | band_8_10 | 5+ | 74 | 74 | 0 | 0 | 17 | 0/0 | 61.6, 4.84e+03, 6.63e+05 | 2.1e-06 | 74 | {"bounded": 46, "understates": 45} |
| 4 | levin | cc47 | group | band_8_10 | none | 0 | 0 | 0 | 0 | 3 | 0/0 | —, —, — | — | 0 | {"undecided": 3} |
| 4 | levin | cc47 | group | gate_q3 | 1-4 | 5 | 5 | 0 | 0 | 20 | 0/0 | 1.61, 3.39e+03, 1.54e+04 | 0.00033 | 5 | {"bounded": 12, "undecided": 11, "understates": 2} |
| 4 | levin | cc47 | group | gate_q3 | 5+ | 18 | 18 | 0 | 0 | 0 | 0/0 | 78.3, 1.43e+03, 3.76e+04 | 3.37e-05 | 18 | {"understates": 14, "undecided": 2, "bounded": 2} |
| 4 | levin | cc47 | group | gate_q3 | none | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"undecided": 26, "bounded": 8} |
| 4 | levin | cc47 | group | gate_q4 | 1-4 | 0 | 0 | 0 | 3 | 76 | 0/0 | —, —, — | — | 0 | {"undecided": 44, "bounded": 35} |
| 4 | levin | cc47 | group | gate_q4 | 5+ | 26 | 26 | 0 | 0 | 1 | 0/0 | 28.6, 3.52e+03, 1.99e+05 | 2.59e-06 | 26 | {"understates": 11, "bounded": 15, "undecided": 1} |
| 4 | levin | cc47 | group | gate_q4 | none | 2 | 2 | 0 | 6 | 86 | 0/0 | 1.15, 1.43, 1.71 | 6.78e-05 | 2 | {"undecided": 61, "bounded": 33} |
| 4 | levin | cc93 | analytic | band_10_12 | 1-4 | 8 | 8 | 0 | 0 | 9 | 0/0 | 578, 746, 1.09e+03 | 5.04e-07 | 8 | {"bounded": 17} |
| 4 | levin | cc93 | analytic | band_10_12 | 5+ | 5 | 5 | 0 | 0 | 6 | 0/0 | 21.5, 92.8, 512 | 8.22e-06 | 5 | {"bounded": 11} |
| 4 | levin | cc93 | analytic | band_10_12 | none | 3 | 3 | 0 | 0 | 27 | 0/0 | 15, 24.7, 24.7 | 7.82e-08 | 3 | {"bounded": 30} |
| 4 | levin | cc93 | analytic | band_10_12 | unknown | 21 | 21 | 0 | 0 | 21 | 0/0 | 4.79, 145, 354 | 1.4e-06 | 21 | {"bounded": 42} |
| 4 | levin | cc93 | analytic | band_12_16 | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 4 | levin | cc93 | analytic | band_12_16 | 5+ | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 4 | levin | cc93 | analytic | band_12_16 | none | 11 | 11 | 0 | 3 | 22 | 0/0 | 4.91, 16.1, 60.8 | 7.55e-08 | 11 | {"bounded": 33, "undecided": 3} |
| 4 | levin | cc93 | analytic | band_12_16 | unknown | 14 | 14 | 0 | 0 | 42 | 0/0 | 2.59, 23.9, 1.5e+03 | 8.71e-07 | 14 | {"bounded": 53, "undecided": 3} |
| 4 | levin | cc93 | analytic | band_16_20 | 1-4 | 5 | 5 | 0 | 0 | 13 | 0/0 | 6.62, 84.8, 255 | 5.78e-06 | 5 | {"bounded": 18} |
| 4 | levin | cc93 | analytic | band_16_20 | 5+ | 6 | 6 | 0 | 0 | 10 | 0/0 | 18, 73.2, 152 | 8.1e-06 | 6 | {"bounded": 16} |
| 4 | levin | cc93 | analytic | band_16_20 | none | 1 | 1 | 0 | 0 | 13 | 0/0 | 62, 62, 62 | 5.8e-06 | 1 | {"bounded": 13, "undecided": 1} |
| 4 | levin | cc93 | analytic | band_16_20 | unknown | 9 | 9 | 0 | 0 | 41 | 0/0 | 11.1, 43.5, 626 | 3.64e-06 | 9 | {"bounded": 45, "undecided": 5} |
| 4 | levin | cc93 | analytic | band_20_32 | 1-4 | 4 | 4 | 0 | 0 | 7 | 0/0 | 139, 217, 266 | 8.63e-07 | 4 | {"bounded": 11} |
| 4 | levin | cc93 | analytic | band_20_32 | 5+ | 5 | 5 | 0 | 0 | 28 | 0/0 | 13.7, 128, 189 | 8.55e-06 | 5 | {"bounded": 33} |
| 4 | levin | cc93 | analytic | band_20_32 | none | 3 | 3 | 0 | 5 | 43 | 0/0 | 10.5, 132, 132 | 3.82e-08 | 3 | {"undecided": 21, "bounded": 30} |
| 4 | levin | cc93 | analytic | band_20_32 | unknown | 13 | 13 | 0 | 5 | 85 | 0/0 | 3.37, 23.7, 270 | 6.02e-06 | 13 | {"bounded": 77, "undecided": 26} |
| 4 | levin | cc93 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 4 | levin | cc93 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 4 | levin | cc93 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 4 | levin | cc93 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 67, "bounded": 31} |
| 4 | levin | cc93 | analytic | band_6_8 | 1-4 | 2 | 2 | 0 | 0 | 3 | 0/0 | 178, 193, 208 | 2.84e-05 | 2 | {"bounded": 5} |
| 4 | levin | cc93 | analytic | band_6_8 | 5+ | 1 | 1 | 0 | 0 | 4 | 0/0 | 388, 388, 388 | 1.74e-05 | 1 | {"bounded": 5} |
| 4 | levin | cc93 | analytic | band_6_8 | none | 7 | 7 | 0 | 0 | 36 | 0/0 | 7.94, 28.5, 673 | 2.73e-07 | 7 | {"bounded": 43} |
| 4 | levin | cc93 | analytic | band_6_8 | unknown | 14 | 14 | 0 | 0 | 25 | 0/0 | 8.7, 26.9, 840 | 2.71e-07 | 14 | {"bounded": 39} |
| 4 | levin | cc93 | analytic | band_8_10 | 1-4 | 7 | 7 | 0 | 0 | 6 | 0/0 | 167, 1.38e+03, 2.88e+03 | 1.23e-06 | 7 | {"bounded": 13} |
| 4 | levin | cc93 | analytic | band_8_10 | 5+ | 3 | 3 | 0 | 0 | 3 | 0/0 | 177, 804, 804 | 1.22e-06 | 3 | {"bounded": 6} |
| 4 | levin | cc93 | analytic | band_8_10 | none | 14 | 14 | 0 | 0 | 24 | 0/0 | 5.02, 10.7, 12.5 | 1.15e-06 | 14 | {"bounded": 38} |
| 4 | levin | cc93 | analytic | band_8_10 | unknown | 13 | 13 | 0 | 0 | 34 | 0/0 | 17.9, 55.5, 1.81e+03 | 1.53e-06 | 13 | {"bounded": 47} |
| 4 | levin | cc93 | analytic | gate_q3 | 1-4 | 4 | 4 | 0 | 0 | 1 | 0/0 | 406, 1.7e+03, 2.99e+03 | 4.27e-06 | 4 | {"bounded": 5} |
| 4 | levin | cc93 | analytic | gate_q3 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 201, 334, 466 | 3.64e-05 | 2 | {"bounded": 2} |
| 4 | levin | cc93 | analytic | gate_q3 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"bounded": 2} |
| 4 | levin | cc93 | analytic | gate_q3 | unknown | 2 | 2 | 0 | 0 | 2 | 0/0 | 15.9, 439, 862 | 1.77e-06 | 2 | {"bounded": 4} |
| 4 | levin | cc93 | analytic | gate_q4 | 1-4 | 4 | 4 | 0 | 0 | 8 | 0/0 | 531, 531, 531 | 3.46e-06 | 4 | {"bounded": 12} |
| 4 | levin | cc93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.49e+03, 1.49e+03, 1.49e+03 | 2.62e-06 | 2 | {"bounded": 2} |
| 4 | levin | cc93 | analytic | gate_q4 | none | 22 | 22 | 0 | 0 | 14 | 0/0 | 14.7, 21.2, 1.2e+03 | 3.34e-07 | 22 | {"bounded": 36} |
| 4 | levin | cc93 | analytic | gate_q4 | unknown | 30 | 30 | 0 | 0 | 7 | 0/0 | 5.97, 39.1, 1.62e+03 | 1.41e-06 | 30 | {"bounded": 37} |
| 4 | levin | cc93 | group | band_10_12 | 1-4 | 18 | 18 | 0 | 0 | 22 | 0/0 | 2.64, 30.6, 7.19e+03 | 1.49e-05 | 18 | {"undecided": 34, "understates": 5, "bounded": 1} |
| 4 | levin | cc93 | group | band_10_12 | 5+ | 65 | 65 | 0 | 0 | 8 | 0/0 | 3.32, 1.59e+04, 1.95e+06 | 2.31e-07 | 65 | {"bounded": 45, "undecided": 3, "understates": 25} |
| 4 | levin | cc93 | group | band_10_12 | none | 0 | 0 | 0 | 0 | 7 | 0/0 | —, —, — | — | 0 | {"undecided": 7} |
| 4 | levin | cc93 | group | band_12_16 | 1-4 | 50 | 50 | 0 | 3 | 20 | 0/0 | 3.56, 53.7, 5.43e+04 | 5.99e-07 | 50 | {"undecided": 63, "understates": 5, "bounded": 5} |
| 4 | levin | cc93 | group | band_12_16 | 5+ | 81 | 81 | 0 | 0 | 4 | 0/0 | 15.8, 1.38e+03, 2.96e+06 | 5.38e-07 | 81 | {"undecided": 5, "understates": 45, "bounded": 35} |
| 4 | levin | cc93 | group | band_12_16 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"undecided": 2} |
| 4 | levin | cc93 | group | band_16_20 | 1-4 | 45 | 45 | 0 | 11 | 10 | 0/0 | 7.14, 136, 5.77e+04 | 3.33e-07 | 45 | {"undecided": 57, "understates": 3, "bounded": 6} |
| 4 | levin | cc93 | group | band_16_20 | 5+ | 84 | 84 | 0 | 0 | 1 | 0/0 | 42.3, 590, 6.94e+04 | 9.99e-07 | 84 | {"bounded": 36, "understates": 42, "undecided": 7} |
| 4 | levin | cc93 | group | band_16_20 | none | 0 | 0 | 0 | 1 | 5 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 4 | levin | cc93 | group | band_20_32 | 1-4 | 45 | 43 | 2 | 34 | 18 | 0/0 | 0.0013, 51.8, 1.11e+03 | 4.97e-07 | 45 | {"undecided": 85, "understates": 8, "bounded": 4} |
| 4 | levin | cc93 | group | band_20_32 | 5+ | 101 | 101 | 0 | 0 | 0 | 0/0 | 18.1, 334, 1.42e+04 | 8.75e-06 | 101 | {"bounded": 48, "understates": 42, "undecided": 11} |
| 4 | levin | cc93 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 4 | levin | cc93 | group | band_20_32v | 1-4 | 13 | 13 | 0 | 29 | 54 | 0/0 | 5.07, 22.1, 88.8 | 1.33e-07 | 13 | {"bounded": 15, "undecided": 73, "understates": 8} |
| 4 | levin | cc93 | group | band_20_32v | 5+ | 9 | 9 | 0 | 0 | 93 | 0/0 | 1.67, 326, 7.36e+03 | 0.00293 | 9 | {"bounded": 65, "understates": 35, "undecided": 2} |
| 4 | levin | cc93 | group | band_6_8 | 1-4 | 14 | 14 | 0 | 7 | 11 | 0/0 | 5.48, 76.8, 3.75e+03 | 2.48e-05 | 14 | {"bounded": 10, "undecided": 17, "understates": 5} |
| 4 | levin | cc93 | group | band_6_8 | 5+ | 72 | 72 | 0 | 0 | 9 | 0/0 | 20.2, 1.74e+04, 2.96e+06 | 1.03e-06 | 72 | {"undecided": 13, "understates": 29, "bounded": 39} |
| 4 | levin | cc93 | group | band_6_8 | none | 0 | 0 | 0 | 1 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 5, "bounded": 2} |
| 4 | levin | cc93 | group | band_8_10 | 1-4 | 13 | 13 | 0 | 1 | 12 | 0/0 | 5.63, 527, 2.79e+07 | 2.2e-05 | 13 | {"understates": 7, "undecided": 15, "bounded": 4} |
| 4 | levin | cc93 | group | band_8_10 | 5+ | 74 | 74 | 0 | 0 | 17 | 0/0 | 57.5, 3.37e+04, 2.25e+06 | 4.44e-07 | 74 | {"bounded": 46, "understates": 45} |
| 4 | levin | cc93 | group | band_8_10 | none | 0 | 0 | 0 | 0 | 3 | 0/0 | —, —, — | — | 0 | {"undecided": 3} |
| 4 | levin | cc93 | group | gate_q3 | 1-4 | 3 | 3 | 0 | 0 | 22 | 0/0 | 3.84e+03, 9.54e+03, 1.06e+04 | 0.000566 | 3 | {"undecided": 18, "bounded": 7} |
| 4 | levin | cc93 | group | gate_q3 | 5+ | 18 | 18 | 0 | 0 | 0 | 0/0 | 50.3, 8.26e+03, 3.5e+05 | 5.28e-06 | 18 | {"bounded": 12, "understates": 6} |
| 4 | levin | cc93 | group | gate_q3 | none | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"undecided": 25, "bounded": 9} |
| 4 | levin | cc93 | group | gate_q4 | 1-4 | 0 | 0 | 0 | 1 | 78 | 0/0 | —, —, — | — | 0 | {"undecided": 44, "bounded": 35} |
| 4 | levin | cc93 | group | gate_q4 | 5+ | 26 | 26 | 0 | 0 | 1 | 0/0 | 359, 1.29e+04, 3.45e+05 | 1e-06 | 26 | {"understates": 12, "undecided": 3, "bounded": 12} |
| 4 | levin | cc93 | group | gate_q4 | none | 1 | 1 | 0 | 11 | 82 | 0/0 | 1.36, 1.36, 1.36 | 0.00114 | 1 | {"undecided": 58, "bounded": 36} |
| 6 | cc | cc47:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.72e+03, 1.53e+04 | 3.1e-07 | 4 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 34.8, 229 | 1.48e-07 | 63 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.38, 58.3, 3.97e+03 | 7.36e-07 | 33 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 375, 2.18e+03, 6.69e+04 | 4.89e-07 | 4 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.89, 31.4, 554 | 1.69e-07 | 51 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 7.33, 61.1, 1.44e+04 | 8.34e-07 | 39 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 612, 3.48e+03, 7.52e+03 | 1.3e-06 | 13 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.03e+03, 3.01e+03 | 1.97e-05 | 4 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.63, 54.3, 1.87e+05 | 5.13e-07 | 32 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.42, 414, 1.12e+04 | 6.24e-07 | 49 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 473, 2.44e+03, 5.95e+03 | 2.8e-06 | 12 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 739, 739, 739 | 2.96e-05 | 2 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12, 266, 911 | 3.5e-07 | 36 |  |
| 6 | cc | cc47:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 15.1, 172, 1.5e+03 | 1.52e-06 | 37 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.3, 10.2, 2.03e+03 | 0.00096 | 37 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 1.57, 37.9, 934 | 0.041 | 40 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.26, 1.59, 123 | 4.84e-05 | 106 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 2.03, 6.75, 76.1 | 0.000109 | 55 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.6, 34.1, 202 | 0.0339 | 31 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.73, 4.74, 108 | 3.35e-05 | 127 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.95, 5.23, 221 | 3.86e-06 | 65 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 1.44, 41.3, 356 | 0.026 | 47 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.92, 3.94, 650 | 8.03e-05 | 78 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.51, 3.54, 51.4 | 7.31e-05 | 79 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 2.38, 47.4, 796 | 0.00112 | 26 |  |
| 6 | cc | cc47:nest3N4 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 2.03, 3.52, 217 | 2.63e-05 | 94 |  |
| 6 | cc | cc47:tree | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.69e+03, 1.52e+04 | 3.1e-07 | 4 |  |
| 6 | cc | cc47:tree | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 5.46, 26.7, 177 | 1.48e-07 | 63 |  |
| 6 | cc | cc47:tree | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 2.63, 45.5, 3.96e+03 | 7.36e-07 | 33 |  |
| 6 | cc | cc47:tree | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 373, 2.17e+03, 6.68e+04 | 4.89e-07 | 4 |  |
| 6 | cc | cc47:tree | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 6.96, 28, 3.93e+04 | 1.69e-07 | 51 |  |
| 6 | cc | cc47:tree | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 5.85, 48.7, 1.43e+04 | 8.34e-07 | 39 |  |
| 6 | cc | cc47:tree | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 609, 3.46e+03, 1.37e+05 | 1.3e-06 | 13 |  |
| 6 | cc | cc47:tree | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.88e+04, 6.06e+04 | 1.97e-05 | 4 |  |
| 6 | cc | cc47:tree | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.05, 2.41e+04, 2.29e+07 | 5.13e-07 | 32 |  |
| 6 | cc | cc47:tree | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.01, 2.69e+03, 1.12e+07 | 6.24e-07 | 49 |  |
| 6 | cc | cc47:tree | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 5.29e+04, 3.41e+05, 5.87e+05 | 2.8e-06 | 12 |  |
| 6 | cc | cc47:tree | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 8.23e+04, 8.23e+04, 8.23e+04 | 2.96e-05 | 2 |  |
| 6 | cc | cc47:tree | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 1.45e+04, 3.51e+05, 6.87e+06 | 3.5e-07 | 36 |  |
| 6 | cc | cc47:tree | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 7.32e+03, 7.41e+05, 1.21e+07 | 1.52e-06 | 37 |  |
| 6 | cc | cc47:tree | group | gate_q1 | 1-4 | 21 | 21 | 0 | 17 | 0 | 0/0 | 1.08, 9.02, 629 | 0.00261 | 20 |  |
| 6 | cc | cc47:tree | group | gate_q1 | 5+ | 45 | 27 | 18 | 0 | 0 | 0/0 | 0.0613, 4.6, 286 | 0.041 | 40 |  |
| 6 | cc | cc47:tree | group | gate_q1 | none | 7 | 7 | 0 | 107 | 0 | 0/0 | 1.28, 2.62, 20.2 | 3.56e-05 | 7 |  |
| 6 | cc | cc47:tree | group | gate_q2 | 1-4 | 10 | 9 | 1 | 45 | 0 | 0/0 | 0.776, 13.6, 26 | 0.0269 | 10 |  |
| 6 | cc | cc47:tree | group | gate_q2 | 5+ | 32 | 25 | 7 | 0 | 0 | 0/0 | 0.0137, 4.62, 47 | 0.0339 | 31 |  |
| 6 | cc | cc47:tree | group | gate_q2 | none | 11 | 11 | 0 | 116 | 0 | 0/0 | 1.84, 2.49, 6.78 | 4.24e-05 | 11 |  |
| 6 | cc | cc47:tree | group | gate_q3 | 1-4 | 10 | 10 | 0 | 55 | 0 | 0/0 | 4.22, 18.8, 67.3 | 0.0826 | 10 |  |
| 6 | cc | cc47:tree | group | gate_q3 | 5+ | 49 | 32 | 17 | 0 | 0 | 0/0 | 0.00255, 5.37, 101 | 0.026 | 47 |  |
| 6 | cc | cc47:tree | group | gate_q3 | none | 4 | 4 | 0 | 74 | 0 | 0/0 | 1.52, 2.02, 2.51 | 0.000895 | 4 |  |
| 6 | cc | cc47:tree | group | gate_q4 | 1-4 | 16 | 16 | 0 | 63 | 0 | 0/0 | 1.52, 4.29, 36.9 | 0.000454 | 16 |  |
| 6 | cc | cc47:tree | group | gate_q4 | 5+ | 27 | 23 | 4 | 0 | 0 | 0/0 | 0.171, 7.07, 392 | 0.00112 | 26 |  |
| 6 | cc | cc47:tree | group | gate_q4 | none | 7 | 7 | 0 | 87 | 0 | 0/0 | 1.7, 4.66, 13.7 | 5.82e-06 | 7 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.59e+04, 1.59e+04, 1.59e+04 | 2.98e-07 | 4 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 154, 3.22e+03, 7.35e+06 | 2.23e-07 | 63 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 50.6, 2.92e+03, 3.51e+05 | 9.31e-07 | 33 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.73e+03, 3.49e+03 | 2.39e-06 | 4 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 108, 4.99e+03, 1.75e+05 | 9.35e-08 | 51 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 639, 4.44e+03, 2.32e+05 | 4.37e-07 | 39 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 1.01e+03, 6.03e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 71.1, 557, 1.84e+04 | 3.11e-07 | 32 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 87.4, 2.16e+03, 1.95e+04 | 6.33e-07 | 49 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 78.3, 308, 3.26e+03 | 4.09e-07 | 36 |  |
| 6 | cc | cc93:chop93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 95.2, 620, 6.89e+03 | 1.27e-06 | 37 |  |
| 6 | cc | cc93:chop93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 4.29, 1.53e+03, 1.88e+08 | 0.0004 | 38 |  |
| 6 | cc | cc93:chop93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 1.09e+07 | 0.00275 | 45 |  |
| 6 | cc | cc93:chop93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 382, 1.28e+04, 3.83e+07 | 4.52e-05 | 107 |  |
| 6 | cc | cc93:chop93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 63.2, 4.34e+03, 2.08e+05 | 0.000135 | 55 |  |
| 6 | cc | cc93:chop93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 6.59e+04 | 0.00128 | 32 |  |
| 6 | cc | cc93:chop93 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 85.2, 7.81e+03, 8.74e+05 | 1.28e-05 | 127 |  |
| 6 | cc | cc93:chop93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 32, 1.71e+03, 1.29e+05 | 2.16e-06 | 65 |  |
| 6 | cc | cc93:chop93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 6 | cc | cc93:chop93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 280, 2.36e+03, 2.39e+05 | 6.44e-05 | 78 |  |
| 6 | cc | cc93:chop93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 26.8, 1.66e+03, 4.34e+04 | 9.55e-05 | 79 |  |
| 6 | cc | cc93:chop93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 21.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 6 | cc | cc93:chop93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 65.5, 1.24e+03, 2.37e+05 | 3.11e-05 | 93 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 50.6, 229 | 2.23e-07 | 63 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.69, 56.4, 5.48e+03 | 9.31e-07 | 33 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.21, 83.1, 645 | 9.35e-08 | 51 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.6, 99.8, 5.01e+04 | 4.37e-07 | 39 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 604, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 6.91, 79.5, 1.4e+05 | 3.11e-07 | 32 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.85, 427, 9.95e+03 | 6.33e-07 | 49 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.34e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.07e+03, 1.07e+03, 1.07e+03 | 2.05e-05 | 2 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 17.4, 240, 1.23e+03 | 4.09e-07 | 36 |  |
| 6 | cc | cc93:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 10.2, 517, 3.43e+03 | 1.27e-06 | 37 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.29, 14, 4.99e+04 | 0.0004 | 38 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 33.7, 558, 1.48e+04 | 0.00275 | 45 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.24, 1.62, 435 | 4.5e-05 | 106 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.98, 5.68, 6.7e+03 | 0.000135 | 55 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 78.3, 835, 3.45e+04 | 0.00128 | 32 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.71, 7.16, 317 | 1.28e-05 | 127 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 2.03, 5.99, 2.58e+03 | 2.16e-06 | 65 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 9.29, 846, 3.12e+04 | 0.000486 | 49 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.97, 4.49, 446 | 6.44e-05 | 78 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.5, 4.85, 147 | 9.55e-05 | 79 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12, 1.68e+03, 1.03e+05 | 6.52e-05 | 27 |  |
| 6 | cc | cc93:nest3N4 | group | gate_q4 | none | 93 | 93 | 0 | 1 | 0 | 0/0 | 1.82, 5.16, 731 | 3.59e-05 | 92 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 6 | cc | cc93:nest93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 6 | cc | cc93:nest93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 5.15e+03 | 0.0004 | 38 |  |
| 6 | cc | cc93:nest93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15.2, 86.7, 1.95e+04 | 0.00275 | 45 |  |
| 6 | cc | cc93:nest93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 310 | 4.52e-05 | 107 |  |
| 6 | cc | cc93:nest93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.45, 348 | 0.000135 | 55 |  |
| 6 | cc | cc93:nest93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.01, 144, 5.57e+03 | 0.00128 | 32 |  |
| 6 | cc | cc93:nest93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.47, 201 | 1.7e-05 | 124 |  |
| 6 | cc | cc93:nest93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.35, 189 | 2.16e-06 | 65 |  |
| 6 | cc | cc93:nest93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.57, 103, 1.69e+03 | 0.000486 | 49 |  |
| 6 | cc | cc93:nest93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.11, 402 | 6.44e-05 | 78 |  |
| 6 | cc | cc93:nest93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.16, 80.8 | 9.55e-05 | 79 |  |
| 6 | cc | cc93:nest93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 143, 7.39e+03 | 6.52e-05 | 27 |  |
| 6 | cc | cc93:nest93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.53, 4.37, 448 | 3.11e-05 | 93 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 6 | cc | cc93:pair93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 6 | cc | cc93:pair93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 4.11e+03 | 0.0004 | 38 |  |
| 6 | cc | cc93:pair93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 12.8, 64.1, 1.76e+04 | 0.00275 | 45 |  |
| 6 | cc | cc93:pair93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.44, 307 | 4.52e-05 | 107 |  |
| 6 | cc | cc93:pair93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 5.11, 259 | 0.000135 | 55 |  |
| 6 | cc | cc93:pair93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.25, 107, 3.71e+03 | 0.00128 | 32 |  |
| 6 | cc | cc93:pair93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.39, 196 | 1.7e-05 | 124 |  |
| 6 | cc | cc93:pair93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.66, 5.16, 189 | 2.16e-06 | 65 |  |
| 6 | cc | cc93:pair93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 4.25, 72.3, 1.21e+03 | 0.000486 | 49 |  |
| 6 | cc | cc93:pair93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 3.94, 338 | 6.44e-05 | 78 |  |
| 6 | cc | cc93:pair93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.14, 80.8 | 9.55e-05 | 79 |  |
| 6 | cc | cc93:pair93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 116, 5.23e+03 | 6.52e-05 | 27 |  |
| 6 | cc | cc93:pair93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.4, 4.27, 448 | 3.11e-05 | 93 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 6 | cc | cc93:tail93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 6 | cc | cc93:tail93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 12.5, 6.04e+03 | 0.0004 | 38 |  |
| 6 | cc | cc93:tail93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 2.66e+04 | 0.00275 | 45 |  |
| 6 | cc | cc93:tail93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 361 | 4.52e-05 | 107 |  |
| 6 | cc | cc93:tail93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.98, 378 | 0.000135 | 55 |  |
| 6 | cc | cc93:tail93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 5.43e+03 | 0.00128 | 32 |  |
| 6 | cc | cc93:tail93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.52, 197 | 1.7e-05 | 124 |  |
| 6 | cc | cc93:tail93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.78, 250 | 2.16e-06 | 65 |  |
| 6 | cc | cc93:tail93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 6 | cc | cc93:tail93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.93, 613 | 6.44e-05 | 78 |  |
| 6 | cc | cc93:tail93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.21, 104 | 9.55e-05 | 79 |  |
| 6 | cc | cc93:tail93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 6 | cc | cc93:tail93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.79, 4.99, 471 | 3.11e-05 | 93 |  |
| 6 | levin | cc47 | analytic | band_10_12 | 1-4 | 8 | 8 | 0 | 0 | 9 | 0/0 | 134, 168, 192 | 3.46e-06 | 8 | {"bounded": 17} |
| 6 | levin | cc47 | analytic | band_10_12 | 5+ | 1 | 1 | 0 | 0 | 10 | 0/0 | 323, 323, 323 | 4.76e-06 | 1 | {"bounded": 11} |
| 6 | levin | cc47 | analytic | band_10_12 | none | 5 | 5 | 0 | 0 | 25 | 0/0 | 14.6, 17.3, 17.9 | 6.53e-07 | 5 | {"bounded": 30} |
| 6 | levin | cc47 | analytic | band_10_12 | unknown | 20 | 20 | 0 | 0 | 22 | 0/0 | 4.43, 91.6, 783 | 1.26e-06 | 20 | {"bounded": 42} |
| 6 | levin | cc47 | analytic | band_12_16 | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 6 | levin | cc47 | analytic | band_12_16 | 5+ | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 6 | levin | cc47 | analytic | band_12_16 | none | 7 | 7 | 0 | 3 | 26 | 0/0 | 3.85, 4.64, 16.4 | 7.51e-07 | 7 | {"bounded": 33, "undecided": 3} |
| 6 | levin | cc47 | analytic | band_12_16 | unknown | 11 | 11 | 0 | 0 | 45 | 0/0 | 3.97, 40.4, 420 | 2.98e-06 | 11 | {"bounded": 53, "undecided": 3} |
| 6 | levin | cc47 | analytic | band_16_20 | 1-4 | 1 | 1 | 0 | 0 | 17 | 0/0 | 94.6, 94.6, 94.6 | 5.53e-06 | 1 | {"bounded": 18} |
| 6 | levin | cc47 | analytic | band_16_20 | 5+ | 3 | 3 | 0 | 0 | 13 | 0/0 | 34.8, 164, 209 | 4.3e-06 | 3 | {"bounded": 16} |
| 6 | levin | cc47 | analytic | band_16_20 | none | 1 | 1 | 0 | 1 | 12 | 0/0 | 9.8, 9.8, 9.8 | 8.73e-05 | 1 | {"bounded": 13, "undecided": 1} |
| 6 | levin | cc47 | analytic | band_16_20 | unknown | 11 | 11 | 0 | 1 | 38 | 0/0 | 2.49, 33.9, 277 | 4.6e-06 | 11 | {"bounded": 45, "undecided": 5} |
| 6 | levin | cc47 | analytic | band_20_32 | 1-4 | 4 | 4 | 0 | 0 | 7 | 0/0 | 192, 384, 1.49e+03 | 6.49e-07 | 4 | {"bounded": 11} |
| 6 | levin | cc47 | analytic | band_20_32 | 5+ | 6 | 6 | 0 | 0 | 27 | 0/0 | 36.4, 56.3, 204 | 5.87e-06 | 6 | {"bounded": 33} |
| 6 | levin | cc47 | analytic | band_20_32 | none | 0 | 0 | 0 | 0 | 51 | 0/0 | —, —, — | — | 0 | {"undecided": 21, "bounded": 30} |
| 6 | levin | cc47 | analytic | band_20_32 | unknown | 7 | 7 | 0 | 0 | 96 | 0/0 | 4.64, 8.18, 101 | 1.13e-05 | 7 | {"bounded": 77, "undecided": 26} |
| 6 | levin | cc47 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 4, "understates": 2} |
| 6 | levin | cc47 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 6 | levin | cc47 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 6 | levin | cc47 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 64, "bounded": 31, "understates": 3} |
| 6 | levin | cc47 | analytic | band_6_8 | 1-4 | 1 | 1 | 0 | 0 | 4 | 0/0 | 153, 153, 153 | 6.96e-06 | 1 | {"bounded": 5} |
| 6 | levin | cc47 | analytic | band_6_8 | 5+ | 1 | 1 | 0 | 0 | 4 | 0/0 | 1.18e+03, 1.18e+03, 1.18e+03 | 5.74e-06 | 1 | {"bounded": 5} |
| 6 | levin | cc47 | analytic | band_6_8 | none | 18 | 18 | 0 | 0 | 25 | 0/0 | 4.58, 11.8, 222 | 8.39e-07 | 18 | {"bounded": 43} |
| 6 | levin | cc47 | analytic | band_6_8 | unknown | 7 | 7 | 0 | 0 | 32 | 0/0 | 9.58, 11.3, 4.67e+03 | 3.95e-07 | 7 | {"bounded": 39} |
| 6 | levin | cc47 | analytic | band_8_10 | 1-4 | 4 | 4 | 0 | 0 | 9 | 0/0 | 157, 618, 1.23e+03 | 6.22e-06 | 4 | {"bounded": 13} |
| 6 | levin | cc47 | analytic | band_8_10 | 5+ | 2 | 2 | 0 | 0 | 4 | 0/0 | 193, 193, 193 | 2.78e-06 | 2 | {"bounded": 6} |
| 6 | levin | cc47 | analytic | band_8_10 | none | 20 | 20 | 0 | 0 | 18 | 0/0 | 5.3, 9.79, 765 | 8.11e-07 | 20 | {"bounded": 38} |
| 6 | levin | cc47 | analytic | band_8_10 | unknown | 22 | 22 | 0 | 0 | 25 | 0/0 | 4.05, 71.9, 1.51e+03 | 9.83e-07 | 22 | {"bounded": 47} |
| 6 | levin | cc47 | group | band_10_12 | 1-4 | 24 | 24 | 0 | 16 | 0 | 0/0 | 1.33, 22.1, 1.45e+04 | 7.73e-06 | 24 | {"understates": 8, "bounded": 13, "undecided": 19} |
| 6 | levin | cc47 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 20.9, 4.06e+03, 2e+06 | 8.66e-07 | 73 | {"bounded": 33, "undecided": 13, "understates": 27} |
| 6 | levin | cc47 | group | band_10_12 | none | 2 | 2 | 0 | 5 | 0 | 0/0 | 1.31, 1.31, 1.31 | 0.000115 | 2 | {"bounded": 5, "undecided": 2} |
| 6 | levin | cc47 | group | band_12_16 | 1-4 | 54 | 54 | 0 | 4 | 15 | 0/0 | 3.44, 35.3, 1.29e+04 | 2.21e-06 | 54 | {"undecided": 63, "understates": 5, "bounded": 5} |
| 6 | levin | cc47 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 9.9, 2.67e+03, 1.72e+06 | 1.08e-06 | 85 | {"undecided": 5, "understates": 45, "bounded": 35} |
| 6 | levin | cc47 | group | band_12_16 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"undecided": 2} |
| 6 | levin | cc47 | group | band_16_20 | 1-4 | 46 | 46 | 0 | 2 | 18 | 0/0 | 3.51, 23.9, 6.54e+03 | 7.67e-07 | 46 | {"undecided": 57, "understates": 3, "bounded": 6} |
| 6 | levin | cc47 | group | band_16_20 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 3.8, 1.27e+03, 9.05e+05 | 5.62e-07 | 85 | {"bounded": 36, "understates": 42, "undecided": 7} |
| 6 | levin | cc47 | group | band_16_20 | none | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 6 | levin | cc47 | group | band_20_32 | 1-4 | 46 | 44 | 2 | 30 | 21 | 0/0 | 0.0013, 19.6, 4.38e+03 | 1.02e-06 | 46 | {"undecided": 85, "understates": 8, "bounded": 4} |
| 6 | levin | cc47 | group | band_20_32 | 5+ | 101 | 101 | 0 | 0 | 0 | 0/0 | 7.33, 240, 5.68e+04 | 1.26e-05 | 101 | {"bounded": 48, "understates": 42, "undecided": 11} |
| 6 | levin | cc47 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 6 | levin | cc47 | group | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 96 | 0/0 | —, —, — | — | 0 | {"bounded": 15, "undecided": 73, "understates": 8} |
| 6 | levin | cc47 | group | band_20_32v | 5+ | 5 | 5 | 0 | 0 | 97 | 0/0 | 326, 3.11e+03, 3.56e+04 | 0.0183 | 5 | {"bounded": 65, "understates": 35, "undecided": 2} |
| 6 | levin | cc47 | group | band_6_8 | 1-4 | 16 | 15 | 1 | 15 | 1 | 0/0 | 0.116, 43.7, 3.21e+05 | 2.77e-05 | 16 | {"bounded": 11, "understates": 6, "undecided": 15} |
| 6 | levin | cc47 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 21.5, 3.32e+03, 6.13e+05 | 2.92e-06 | 81 | {"bounded": 36, "understates": 38, "undecided": 7} |
| 6 | levin | cc47 | group | band_6_8 | none | 0 | 0 | 0 | 6 | 1 | 0/0 | —, —, — | — | 0 | {"bounded": 5, "undecided": 2} |
| 6 | levin | cc47 | group | band_8_10 | 1-4 | 15 | 15 | 0 | 9 | 2 | 0/0 | 1.92, 264, 1.07e+05 | 7.74e-05 | 15 | {"undecided": 14, "bounded": 10, "understates": 2} |
| 6 | levin | cc47 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 50.6, 4.01e+03, 6.6e+05 | 7.01e-06 | 91 | {"understates": 42, "bounded": 44, "undecided": 5} |
| 6 | levin | cc47 | group | band_8_10 | none | 0 | 0 | 0 | 2 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1, "bounded": 2} |
| 6 | levin | cc93 | analytic | band_10_12 | 1-4 | 8 | 8 | 0 | 0 | 9 | 0/0 | 178, 204, 1.34e+04 | 6.39e-07 | 8 | {"bounded": 17} |
| 6 | levin | cc93 | analytic | band_10_12 | 5+ | 3 | 3 | 0 | 0 | 8 | 0/0 | 104, 104, 3.71e+03 | 2.98e-06 | 3 | {"bounded": 11} |
| 6 | levin | cc93 | analytic | band_10_12 | none | 6 | 6 | 0 | 0 | 24 | 0/0 | 8.94, 17.1, 17.7 | 4.61e-07 | 6 | {"bounded": 30} |
| 6 | levin | cc93 | analytic | band_10_12 | unknown | 15 | 15 | 0 | 0 | 27 | 0/0 | 6.98, 81.5, 174 | 1.72e-06 | 15 | {"bounded": 42} |
| 6 | levin | cc93 | analytic | band_12_16 | 1-4 | 1 | 1 | 0 | 0 | 5 | 0/0 | 69, 69, 69 | 7.75e-06 | 1 | {"bounded": 6} |
| 6 | levin | cc93 | analytic | band_12_16 | 5+ | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 6 | levin | cc93 | analytic | band_12_16 | none | 14 | 14 | 0 | 0 | 22 | 0/0 | 4.91, 16.3, 60.8 | 8.11e-08 | 14 | {"bounded": 33, "undecided": 3} |
| 6 | levin | cc93 | analytic | band_12_16 | unknown | 9 | 9 | 0 | 0 | 47 | 0/0 | 2.59, 18.6, 153 | 9.18e-07 | 9 | {"bounded": 53, "undecided": 3} |
| 6 | levin | cc93 | analytic | band_16_20 | 1-4 | 3 | 3 | 0 | 0 | 15 | 0/0 | 73.7, 434, 434 | 2.81e-07 | 3 | {"bounded": 18} |
| 6 | levin | cc93 | analytic | band_16_20 | 5+ | 3 | 3 | 0 | 0 | 13 | 0/0 | 66.9, 68.2, 110 | 4.89e-06 | 3 | {"bounded": 16} |
| 6 | levin | cc93 | analytic | band_16_20 | none | 1 | 1 | 0 | 0 | 13 | 0/0 | 113, 113, 113 | 7.28e-06 | 1 | {"bounded": 13, "undecided": 1} |
| 6 | levin | cc93 | analytic | band_16_20 | unknown | 12 | 12 | 0 | 3 | 35 | 0/0 | 10.6, 75.5, 1.54e+03 | 2.98e-06 | 12 | {"bounded": 45, "undecided": 5} |
| 6 | levin | cc93 | analytic | band_20_32 | 1-4 | 1 | 1 | 0 | 0 | 10 | 0/0 | 827, 827, 827 | 1.87e-07 | 1 | {"bounded": 11} |
| 6 | levin | cc93 | analytic | band_20_32 | 5+ | 5 | 5 | 0 | 0 | 28 | 0/0 | 51.7, 90.9, 188 | 6.12e-06 | 5 | {"bounded": 33} |
| 6 | levin | cc93 | analytic | band_20_32 | none | 3 | 3 | 0 | 5 | 43 | 0/0 | 10.5, 132, 132 | 3.82e-08 | 3 | {"undecided": 21, "bounded": 30} |
| 6 | levin | cc93 | analytic | band_20_32 | unknown | 12 | 12 | 0 | 4 | 87 | 0/0 | 3.37, 11.1, 531 | 6.09e-06 | 12 | {"bounded": 77, "undecided": 26} |
| 6 | levin | cc93 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 6 | levin | cc93 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 6 | levin | cc93 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 6 | levin | cc93 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 66, "bounded": 30, "understates": 2} |
| 6 | levin | cc93 | analytic | band_6_8 | 1-4 | 0 | 0 | 0 | 0 | 5 | 0/0 | —, —, — | — | 0 | {"bounded": 5} |
| 6 | levin | cc93 | analytic | band_6_8 | 5+ | 1 | 1 | 0 | 0 | 4 | 0/0 | 388, 388, 388 | 1.74e-05 | 1 | {"bounded": 5} |
| 6 | levin | cc93 | analytic | band_6_8 | none | 17 | 17 | 0 | 0 | 26 | 0/0 | 4.28, 13, 673 | 5.82e-07 | 17 | {"bounded": 43} |
| 6 | levin | cc93 | analytic | band_6_8 | unknown | 10 | 10 | 0 | 0 | 29 | 0/0 | 22, 75.8, 4.67e+03 | 1.69e-07 | 10 | {"bounded": 39} |
| 6 | levin | cc93 | analytic | band_8_10 | 1-4 | 4 | 4 | 0 | 0 | 9 | 0/0 | 167, 322, 435 | 2.05e-05 | 4 | {"bounded": 13} |
| 6 | levin | cc93 | analytic | band_8_10 | 5+ | 3 | 3 | 0 | 0 | 3 | 0/0 | 177, 225, 225 | 4.35e-06 | 3 | {"bounded": 6} |
| 6 | levin | cc93 | analytic | band_8_10 | none | 9 | 9 | 0 | 0 | 29 | 0/0 | 3.9, 10.7, 12.5 | 1.12e-06 | 9 | {"bounded": 38} |
| 6 | levin | cc93 | analytic | band_8_10 | unknown | 15 | 15 | 0 | 0 | 32 | 0/0 | 4.19, 75.8, 219 | 1.3e-06 | 15 | {"bounded": 47} |
| 6 | levin | cc93 | group | band_10_12 | 1-4 | 19 | 19 | 0 | 9 | 12 | 0/0 | 1.55, 27.6, 6.94e+03 | 2.66e-05 | 19 | {"understates": 5, "bounded": 11, "undecided": 24} |
| 6 | levin | cc93 | group | band_10_12 | 5+ | 71 | 71 | 0 | 0 | 2 | 0/0 | 40.5, 1.05e+04, 1.76e+06 | 1.99e-07 | 71 | {"undecided": 20, "bounded": 25, "understates": 28} |
| 6 | levin | cc93 | group | band_10_12 | none | 2 | 2 | 0 | 2 | 3 | 0/0 | 1.56, 1.56, 1.56 | 0.000537 | 2 | {"bounded": 4, "undecided": 3} |
| 6 | levin | cc93 | group | band_12_16 | 1-4 | 50 | 50 | 0 | 3 | 20 | 0/0 | 3.56, 51.2, 1.91e+04 | 5.99e-07 | 50 | {"undecided": 63, "understates": 5, "bounded": 5} |
| 6 | levin | cc93 | group | band_12_16 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 15.8, 4.8e+03, 2.96e+06 | 4.85e-07 | 85 | {"undecided": 5, "understates": 45, "bounded": 35} |
| 6 | levin | cc93 | group | band_12_16 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"undecided": 2} |
| 6 | levin | cc93 | group | band_16_20 | 1-4 | 43 | 43 | 0 | 2 | 21 | 0/0 | 5.49, 36.7, 5.1e+03 | 1.03e-06 | 43 | {"undecided": 57, "understates": 3, "bounded": 6} |
| 6 | levin | cc93 | group | band_16_20 | 5+ | 85 | 85 | 0 | 0 | 0 | 0/0 | 4.74, 4.04e+03, 2.24e+06 | 2.14e-07 | 85 | {"bounded": 36, "understates": 42, "undecided": 7} |
| 6 | levin | cc93 | group | band_16_20 | none | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 6 | levin | cc93 | group | band_20_32 | 1-4 | 45 | 43 | 2 | 29 | 23 | 0/0 | 0.0013, 28.6, 148 | 8.5e-07 | 45 | {"undecided": 85, "understates": 8, "bounded": 4} |
| 6 | levin | cc93 | group | band_20_32 | 5+ | 101 | 101 | 0 | 0 | 0 | 0/0 | 9.91, 321, 9.6e+04 | 9.56e-06 | 101 | {"bounded": 48, "understates": 42, "undecided": 11} |
| 6 | levin | cc93 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 6 | levin | cc93 | group | band_20_32v | 1-4 | 3 | 3 | 0 | 1 | 92 | 0/0 | 10.5, 12.7, 13.9 | 2.66e-07 | 3 | {"bounded": 15, "undecided": 73, "understates": 8} |
| 6 | levin | cc93 | group | band_20_32v | 5+ | 5 | 5 | 0 | 0 | 97 | 0/0 | 326, 3.11e+03, 4.71e+06 | 0.0398 | 5 | {"bounded": 65, "understates": 35, "undecided": 2} |
| 6 | levin | cc93 | group | band_6_8 | 1-4 | 14 | 14 | 0 | 7 | 11 | 0/0 | 5.48, 76.8, 3.75e+03 | 2.48e-05 | 14 | {"bounded": 10, "undecided": 17, "understates": 5} |
| 6 | levin | cc93 | group | band_6_8 | 5+ | 79 | 79 | 0 | 0 | 2 | 0/0 | 20.2, 1.42e+04, 2.96e+06 | 1.15e-06 | 79 | {"undecided": 14, "understates": 29, "bounded": 38} |
| 6 | levin | cc93 | group | band_6_8 | none | 0 | 0 | 0 | 1 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 5, "bounded": 2} |
| 6 | levin | cc93 | group | band_8_10 | 1-4 | 15 | 15 | 0 | 2 | 9 | 0/0 | 6.33, 350, 2.35e+07 | 1.68e-05 | 15 | {"undecided": 13, "bounded": 13} |
| 6 | levin | cc93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 47.8, 2.88e+04, 2.12e+06 | 1.27e-06 | 91 | {"bounded": 36, "understates": 33, "undecided": 22} |
| 6 | levin | cc93 | group | band_8_10 | none | 0 | 0 | 0 | 0 | 3 | 0/0 | —, —, — | — | 0 | {"bounded": 1, "undecided": 2} |
| 8 | cc | cc47:nest3N4 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.01e+03, 2.42e+03, 5.67e+05 | 7.38e-06 | 5 |  |
| 8 | cc | cc47:nest3N4 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.45e+04, 3.36e+05, 2.3e+06 | 7.45e-07 | 5 |  |
| 8 | cc | cc47:nest3N4 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 1.13e+03, 9.39e+04, 2.02e+07 | 5.8e-07 | 43 |  |
| 8 | cc | cc47:nest3N4 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 160, 3.03e+05, 1.23e+07 | 1.39e-06 | 39 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.72e+03, 1.53e+04 | 3.1e-07 | 4 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 34.8, 229 | 1.48e-07 | 63 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.38, 58.3, 3.97e+03 | 7.36e-07 | 33 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 375, 2.18e+03, 6.69e+04 | 4.89e-07 | 4 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.89, 31.4, 554 | 1.69e-07 | 51 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 7.33, 61.1, 1.44e+04 | 8.34e-07 | 39 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 612, 3.48e+03, 7.52e+03 | 1.3e-06 | 13 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.03e+03, 3.01e+03 | 1.97e-05 | 4 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.63, 54.3, 1.87e+05 | 5.13e-07 | 32 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.42, 414, 1.12e+04 | 6.24e-07 | 49 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 473, 2.44e+03, 5.95e+03 | 2.8e-06 | 12 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 739, 739, 739 | 2.96e-05 | 2 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12, 266, 911 | 3.5e-07 | 36 |  |
| 8 | cc | cc47:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 15.1, 172, 1.5e+03 | 1.52e-06 | 37 |  |
| 8 | cc | cc47:nest3N4 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.43, 6.64, 2.16e+03 | 1.62e-06 | 32 |  |
| 8 | cc | cc47:nest3N4 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 1.69, 140, 2.43e+03 | 0.00104 | 80 |  |
| 8 | cc | cc47:nest3N4 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 2.1, 5.49, 47.1 | 1.3e-06 | 7 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.3, 10.2, 2.03e+03 | 0.00096 | 37 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 1.57, 37.9, 934 | 0.041 | 40 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.26, 1.59, 123 | 4.84e-05 | 106 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 2.03, 6.75, 76.1 | 0.000109 | 55 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.6, 34.1, 202 | 0.0339 | 31 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.73, 4.74, 108 | 3.35e-05 | 127 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.95, 5.23, 221 | 3.86e-06 | 65 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 1.44, 41.3, 356 | 0.026 | 47 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.92, 3.94, 650 | 8.03e-05 | 78 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.51, 3.54, 51.4 | 7.31e-05 | 79 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 2.38, 47.4, 796 | 0.00112 | 26 |  |
| 8 | cc | cc47:nest3N4 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 2.03, 3.52, 217 | 2.63e-05 | 94 |  |
| 8 | cc | cc47:tree | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.66e+06, 7.6e+06, 2.9e+08 | 7.38e-06 | 5 |  |
| 8 | cc | cc47:tree | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 7.56e+07, 1.05e+08, 2.71e+08 | 7.45e-07 | 5 |  |
| 8 | cc | cc47:tree | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 4.33e+06, 7.99e+07, 2.59e+10 | 5.8e-07 | 43 |  |
| 8 | cc | cc47:tree | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 4.23e+05, 8.25e+07, 2.47e+09 | 1.39e-06 | 39 |  |
| 8 | cc | cc47:tree | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 156, 7.69e+03, 1.52e+04 | 3.1e-07 | 4 |  |
| 8 | cc | cc47:tree | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 5.46, 26.7, 177 | 1.48e-07 | 63 |  |
| 8 | cc | cc47:tree | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 2.63, 45.5, 3.96e+03 | 7.36e-07 | 33 |  |
| 8 | cc | cc47:tree | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 373, 2.17e+03, 6.68e+04 | 4.89e-07 | 4 |  |
| 8 | cc | cc47:tree | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 6.96, 28, 3.93e+04 | 1.69e-07 | 51 |  |
| 8 | cc | cc47:tree | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 5.85, 48.7, 1.43e+04 | 8.34e-07 | 39 |  |
| 8 | cc | cc47:tree | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 609, 3.46e+03, 1.37e+05 | 1.3e-06 | 13 |  |
| 8 | cc | cc47:tree | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.36e+03, 2.88e+04, 6.06e+04 | 1.97e-05 | 4 |  |
| 8 | cc | cc47:tree | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 5.05, 2.41e+04, 2.29e+07 | 5.13e-07 | 32 |  |
| 8 | cc | cc47:tree | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.01, 2.69e+03, 1.12e+07 | 6.24e-07 | 49 |  |
| 8 | cc | cc47:tree | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 5.29e+04, 3.41e+05, 5.87e+05 | 2.8e-06 | 12 |  |
| 8 | cc | cc47:tree | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 8.23e+04, 8.23e+04, 8.23e+04 | 2.96e-05 | 2 |  |
| 8 | cc | cc47:tree | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 1.45e+04, 3.51e+05, 6.87e+06 | 3.5e-07 | 36 |  |
| 8 | cc | cc47:tree | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 7.32e+03, 7.41e+05, 1.21e+07 | 1.52e-06 | 37 |  |
| 8 | cc | cc47:tree | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 10.5, 938, 2.59e+04 | 1.62e-06 | 32 |  |
| 8 | cc | cc47:tree | group | band_6_8 | 5+ | 81 | 76 | 5 | 0 | 0 | 0/0 | 0.0385, 62.5, 2.64e+04 | 0.00104 | 80 |  |
| 8 | cc | cc47:tree | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 5.24, 102, 1.84e+03 | 1.3e-06 | 7 |  |
| 8 | cc | cc47:tree | group | gate_q1 | 1-4 | 21 | 21 | 0 | 17 | 0 | 0/0 | 1.08, 9.02, 629 | 0.00261 | 20 |  |
| 8 | cc | cc47:tree | group | gate_q1 | 5+ | 45 | 27 | 18 | 0 | 0 | 0/0 | 0.0613, 4.6, 286 | 0.041 | 40 |  |
| 8 | cc | cc47:tree | group | gate_q1 | none | 7 | 7 | 0 | 107 | 0 | 0/0 | 1.28, 2.62, 20.2 | 3.56e-05 | 7 |  |
| 8 | cc | cc47:tree | group | gate_q2 | 1-4 | 10 | 9 | 1 | 45 | 0 | 0/0 | 0.776, 13.6, 26 | 0.0269 | 10 |  |
| 8 | cc | cc47:tree | group | gate_q2 | 5+ | 32 | 25 | 7 | 0 | 0 | 0/0 | 0.0137, 4.62, 47 | 0.0339 | 31 |  |
| 8 | cc | cc47:tree | group | gate_q2 | none | 11 | 11 | 0 | 116 | 0 | 0/0 | 1.84, 2.49, 6.78 | 4.24e-05 | 11 |  |
| 8 | cc | cc47:tree | group | gate_q3 | 1-4 | 10 | 10 | 0 | 55 | 0 | 0/0 | 4.22, 18.8, 67.3 | 0.0826 | 10 |  |
| 8 | cc | cc47:tree | group | gate_q3 | 5+ | 49 | 32 | 17 | 0 | 0 | 0/0 | 0.00255, 5.37, 101 | 0.026 | 47 |  |
| 8 | cc | cc47:tree | group | gate_q3 | none | 4 | 4 | 0 | 74 | 0 | 0/0 | 1.52, 2.02, 2.51 | 0.000895 | 4 |  |
| 8 | cc | cc47:tree | group | gate_q4 | 1-4 | 16 | 16 | 0 | 63 | 0 | 0/0 | 1.52, 4.29, 36.9 | 0.000454 | 16 |  |
| 8 | cc | cc47:tree | group | gate_q4 | 5+ | 27 | 23 | 4 | 0 | 0 | 0/0 | 0.171, 7.07, 392 | 0.00112 | 26 |  |
| 8 | cc | cc47:tree | group | gate_q4 | none | 7 | 7 | 0 | 87 | 0 | 0/0 | 1.7, 4.66, 13.7 | 5.82e-06 | 7 |  |
| 8 | cc | cc93:chop93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 8 | cc | cc93:chop93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 8 | cc | cc93:chop93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 38.7, 551, 2.16e+04 | 3.59e-07 | 43 |  |
| 8 | cc | cc93:chop93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 56.5, 472, 9.43e+03 | 7.91e-07 | 39 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.59e+04, 1.59e+04, 1.59e+04 | 2.98e-07 | 4 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 154, 3.22e+03, 7.35e+06 | 2.23e-07 | 63 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 50.6, 2.92e+03, 3.51e+05 | 9.31e-07 | 33 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.73e+03, 3.49e+03 | 2.39e-06 | 4 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 108, 4.99e+03, 1.75e+05 | 9.35e-08 | 51 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 639, 4.44e+03, 2.32e+05 | 4.37e-07 | 39 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 1.01e+03, 6.03e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 71.1, 557, 1.84e+04 | 3.11e-07 | 32 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 87.4, 2.16e+03, 1.95e+04 | 6.33e-07 | 49 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 78.3, 308, 3.26e+03 | 4.09e-07 | 36 |  |
| 8 | cc | cc93:chop93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 95.2, 620, 6.89e+03 | 1.27e-06 | 37 |  |
| 8 | cc | cc93:chop93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.44, 450, 3.99e+04 | 2.6e-06 | 32 |  |
| 8 | cc | cc93:chop93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 5.35, 101, 7.94e+05 | 6.24e-05 | 81 |  |
| 8 | cc | cc93:chop93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 184, 1.02e+03, 1.77e+03 | 6.17e-06 | 7 |  |
| 8 | cc | cc93:chop93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 4.29, 1.53e+03, 1.88e+08 | 0.0004 | 38 |  |
| 8 | cc | cc93:chop93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 1.09e+07 | 0.00275 | 45 |  |
| 8 | cc | cc93:chop93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 382, 1.28e+04, 3.83e+07 | 4.52e-05 | 107 |  |
| 8 | cc | cc93:chop93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 63.2, 4.34e+03, 2.08e+05 | 0.000135 | 55 |  |
| 8 | cc | cc93:chop93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 6.59e+04 | 0.00128 | 32 |  |
| 8 | cc | cc93:chop93 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 85.2, 7.81e+03, 8.74e+05 | 1.28e-05 | 127 |  |
| 8 | cc | cc93:chop93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 32, 1.71e+03, 1.29e+05 | 2.16e-06 | 65 |  |
| 8 | cc | cc93:chop93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 8 | cc | cc93:chop93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 280, 2.36e+03, 2.39e+05 | 6.44e-05 | 78 |  |
| 8 | cc | cc93:chop93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 26.8, 1.66e+03, 4.34e+04 | 9.55e-05 | 79 |  |
| 8 | cc | cc93:chop93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 21.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 8 | cc | cc93:chop93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 65.5, 1.24e+03, 2.37e+05 | 3.11e-05 | 93 |  |
| 8 | cc | cc93:nest3N4 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 1.51e+03, 5.22e+03, 5.94e+05 | 7.36e-06 | 5 |  |
| 8 | cc | cc93:nest3N4 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 4.41e+04, 4.9e+05, 5.74e+06 | 5.11e-07 | 5 |  |
| 8 | cc | cc93:nest3N4 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 1.41e+03, 1.06e+05, 2.98e+07 | 3.59e-07 | 43 |  |
| 8 | cc | cc93:nest3N4 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 541, 5.09e+05, 4.24e+07 | 7.91e-07 | 39 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 6.57, 50.6, 229 | 2.23e-07 | 63 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.69, 56.4, 5.48e+03 | 9.31e-07 | 33 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.21, 83.1, 645 | 9.35e-08 | 51 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.6, 99.8, 5.01e+04 | 4.37e-07 | 39 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 604, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 6.91, 79.5, 1.4e+05 | 3.11e-07 | 32 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.85, 427, 9.95e+03 | 6.33e-07 | 49 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.34e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.07e+03, 1.07e+03, 1.07e+03 | 2.05e-05 | 2 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 17.4, 240, 1.23e+03 | 4.09e-07 | 36 |  |
| 8 | cc | cc93:nest3N4 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 10.2, 517, 3.43e+03 | 1.27e-06 | 37 |  |
| 8 | cc | cc93:nest3N4 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.8, 19.2, 5.82e+03 | 2.6e-06 | 32 |  |
| 8 | cc | cc93:nest3N4 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 20.4, 1.25e+03, 1.26e+05 | 6.24e-05 | 81 |  |
| 8 | cc | cc93:nest3N4 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.47, 6.85, 20.5 | 6.17e-06 | 7 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.29, 14, 4.99e+04 | 0.0004 | 38 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 33.7, 558, 1.48e+04 | 0.00275 | 45 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q1 | none | 113 | 113 | 0 | 1 | 0 | 0/0 | 1.24, 1.62, 435 | 4.5e-05 | 106 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.98, 5.68, 6.7e+03 | 0.000135 | 55 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 78.3, 835, 3.45e+04 | 0.00128 | 32 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q2 | none | 127 | 127 | 0 | 0 | 0 | 0/0 | 1.71, 7.16, 317 | 1.28e-05 | 127 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 2.03, 5.99, 2.58e+03 | 2.16e-06 | 65 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 9.29, 846, 3.12e+04 | 0.000486 | 49 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.97, 4.49, 446 | 6.44e-05 | 78 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.5, 4.85, 147 | 9.55e-05 | 79 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12, 1.68e+03, 1.03e+05 | 6.52e-05 | 27 |  |
| 8 | cc | cc93:nest3N4 | group | gate_q4 | none | 93 | 93 | 0 | 1 | 0 | 0/0 | 1.82, 5.16, 731 | 3.59e-05 | 92 |  |
| 8 | cc | cc93:nest93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 8 | cc | cc93:nest93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 8 | cc | cc93:nest93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 8 | cc | cc93:nest93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 8 | cc | cc93:nest93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 8 | cc | cc93:nest93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.19, 7.74, 206 | 2.6e-06 | 32 |  |
| 8 | cc | cc93:nest93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 4.65, 89, 5.85e+03 | 6.24e-05 | 81 |  |
| 8 | cc | cc93:nest93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 2.83, 4.41, 18.3 | 6.17e-06 | 7 |  |
| 8 | cc | cc93:nest93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 5.15e+03 | 0.0004 | 38 |  |
| 8 | cc | cc93:nest93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15.2, 86.7, 1.95e+04 | 0.00275 | 45 |  |
| 8 | cc | cc93:nest93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 310 | 4.52e-05 | 107 |  |
| 8 | cc | cc93:nest93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.45, 348 | 0.000135 | 55 |  |
| 8 | cc | cc93:nest93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.01, 144, 5.57e+03 | 0.00128 | 32 |  |
| 8 | cc | cc93:nest93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.47, 201 | 1.7e-05 | 124 |  |
| 8 | cc | cc93:nest93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.35, 189 | 2.16e-06 | 65 |  |
| 8 | cc | cc93:nest93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 6.57, 103, 1.69e+03 | 0.000486 | 49 |  |
| 8 | cc | cc93:nest93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.11, 402 | 6.44e-05 | 78 |  |
| 8 | cc | cc93:nest93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.16, 80.8 | 9.55e-05 | 79 |  |
| 8 | cc | cc93:nest93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 143, 7.39e+03 | 6.52e-05 | 27 |  |
| 8 | cc | cc93:nest93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.53, 4.37, 448 | 3.11e-05 | 93 |  |
| 8 | cc | cc93:pair93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 8 | cc | cc93:pair93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 8 | cc | cc93:pair93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 8 | cc | cc93:pair93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 8 | cc | cc93:pair93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 8 | cc | cc93:pair93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.67, 7.74, 206 | 2.6e-06 | 32 |  |
| 8 | cc | cc93:pair93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 4.65, 66.2, 5.5e+03 | 6.24e-05 | 81 |  |
| 8 | cc | cc93:pair93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.03, 4.41, 17.5 | 6.17e-06 | 7 |  |
| 8 | cc | cc93:pair93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 11.6, 4.11e+03 | 0.0004 | 38 |  |
| 8 | cc | cc93:pair93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 12.8, 64.1, 1.76e+04 | 0.00275 | 45 |  |
| 8 | cc | cc93:pair93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.44, 307 | 4.52e-05 | 107 |  |
| 8 | cc | cc93:pair93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 5.11, 259 | 0.000135 | 55 |  |
| 8 | cc | cc93:pair93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.25, 107, 3.71e+03 | 0.00128 | 32 |  |
| 8 | cc | cc93:pair93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.39, 196 | 1.7e-05 | 124 |  |
| 8 | cc | cc93:pair93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.66, 5.16, 189 | 2.16e-06 | 65 |  |
| 8 | cc | cc93:pair93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 4.25, 72.3, 1.21e+03 | 0.000486 | 49 |  |
| 8 | cc | cc93:pair93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 3.94, 338 | 6.44e-05 | 78 |  |
| 8 | cc | cc93:pair93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.14, 80.8 | 9.55e-05 | 79 |  |
| 8 | cc | cc93:pair93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 116, 5.23e+03 | 6.52e-05 | 27 |  |
| 8 | cc | cc93:pair93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.4, 4.27, 448 | 3.11e-05 | 93 |  |
| 8 | cc | cc93:tail93 | analytic | band_6_8 | 1-4 | 5 | 5 | 0 | 0 | 0 | 0/0 | 950, 2e+03, 2.86e+03 | 7.36e-06 | 5 |  |
| 8 | cc | cc93:tail93 | analytic | band_6_8 | 5+ | 5 | 5 | 0 | 0 | 0 | 0/0 | 3.24e+03, 4.81e+03, 1.15e+04 | 5.11e-07 | 5 |  |
| 8 | cc | cc93:tail93 | analytic | band_6_8 | none | 43 | 43 | 0 | 0 | 0 | 0/0 | 11.4, 48, 2.11e+03 | 3.59e-07 | 43 |  |
| 8 | cc | cc93:tail93 | analytic | band_6_8 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 15.1, 50.9, 5.41e+03 | 7.91e-07 | 39 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q1 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 157, 157, 157 | 2.98e-07 | 4 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q1 | none | 63 | 63 | 0 | 0 | 0 | 0/0 | 7.01, 46, 237 | 2.23e-07 | 63 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q1 | unknown | 33 | 33 | 0 | 0 | 0 | 0/0 | 3.57, 63.1, 5.48e+03 | 9.31e-07 | 33 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q2 | 1-4 | 4 | 4 | 0 | 0 | 0 | 0/0 | 445, 754, 3.49e+03 | 2.39e-06 | 4 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q2 | none | 51 | 51 | 0 | 0 | 0 | 0/0 | 8.46, 80.6, 629 | 9.35e-08 | 51 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q2 | unknown | 39 | 39 | 0 | 0 | 0 | 0/0 | 10.7, 103, 5.01e+04 | 4.37e-07 | 39 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q3 | 1-4 | 13 | 13 | 0 | 0 | 0 | 0/0 | 603, 5.15e+03, 1.25e+04 | 5.43e-07 | 13 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q3 | 5+ | 4 | 4 | 0 | 0 | 0 | 0/0 | 2.37e+03, 3.39e+03, 4.79e+03 | 9.48e-06 | 4 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q3 | none | 32 | 32 | 0 | 0 | 0 | 0/0 | 7.05, 26.7, 432 | 3.11e-07 | 32 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q3 | unknown | 49 | 49 | 0 | 0 | 0 | 0/0 | 5.72, 218, 8.66e+03 | 6.33e-07 | 49 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q4 | 1-4 | 12 | 12 | 0 | 0 | 0 | 0/0 | 625, 1.33e+03, 2.38e+04 | 9.45e-07 | 12 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q4 | 5+ | 2 | 2 | 0 | 0 | 0 | 0/0 | 1.06e+03, 1.06e+03, 1.06e+03 | 2.05e-05 | 2 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q4 | none | 36 | 36 | 0 | 0 | 0 | 0/0 | 12.8, 28.7, 544 | 4.09e-07 | 36 |  |
| 8 | cc | cc93:tail93 | analytic | gate_q4 | unknown | 37 | 37 | 0 | 0 | 0 | 0/0 | 9.81, 46.6, 2.65e+03 | 1.27e-06 | 37 |  |
| 8 | cc | cc93:tail93 | group | band_6_8 | 1-4 | 32 | 32 | 0 | 0 | 0 | 0/0 | 2.84, 8.49, 206 | 2.6e-06 | 32 |  |
| 8 | cc | cc93:tail93 | group | band_6_8 | 5+ | 81 | 81 | 0 | 0 | 0 | 0/0 | 5.35, 101, 6.24e+03 | 6.24e-05 | 81 |  |
| 8 | cc | cc93:tail93 | group | band_6_8 | none | 7 | 7 | 0 | 0 | 0 | 0/0 | 3.26, 4.42, 17.6 | 6.17e-06 | 7 |  |
| 8 | cc | cc93:tail93 | group | gate_q1 | 1-4 | 38 | 38 | 0 | 0 | 0 | 0/0 | 1.28, 12.5, 6.04e+03 | 0.0004 | 38 |  |
| 8 | cc | cc93:tail93 | group | gate_q1 | 5+ | 45 | 45 | 0 | 0 | 0 | 0/0 | 15, 99.9, 2.66e+04 | 0.00275 | 45 |  |
| 8 | cc | cc93:tail93 | group | gate_q1 | none | 114 | 114 | 0 | 0 | 0 | 0/0 | 1.18, 1.46, 361 | 4.52e-05 | 107 |  |
| 8 | cc | cc93:tail93 | group | gate_q2 | 1-4 | 55 | 55 | 0 | 0 | 0 | 0/0 | 1.71, 4.98, 378 | 0.000135 | 55 |  |
| 8 | cc | cc93:tail93 | group | gate_q2 | 5+ | 32 | 32 | 0 | 0 | 0 | 0/0 | 3.39, 149, 5.43e+03 | 0.00128 | 32 |  |
| 8 | cc | cc93:tail93 | group | gate_q2 | none | 124 | 124 | 0 | 3 | 0 | 0/0 | 1.58, 5.52, 197 | 1.7e-05 | 124 |  |
| 8 | cc | cc93:tail93 | group | gate_q3 | 1-4 | 65 | 65 | 0 | 0 | 0 | 0/0 | 1.88, 5.78, 250 | 2.16e-06 | 65 |  |
| 8 | cc | cc93:tail93 | group | gate_q3 | 5+ | 49 | 49 | 0 | 0 | 0 | 0/0 | 7.12, 115, 1.82e+03 | 0.000486 | 49 |  |
| 8 | cc | cc93:tail93 | group | gate_q3 | none | 78 | 78 | 0 | 0 | 0 | 0/0 | 1.6, 4.93, 613 | 6.44e-05 | 78 |  |
| 8 | cc | cc93:tail93 | group | gate_q4 | 1-4 | 79 | 79 | 0 | 0 | 0 | 0/0 | 1.94, 4.21, 104 | 9.55e-05 | 79 |  |
| 8 | cc | cc93:tail93 | group | gate_q4 | 5+ | 27 | 27 | 0 | 0 | 0 | 0/0 | 12.2, 160, 7.5e+03 | 6.52e-05 | 27 |  |
| 8 | cc | cc93:tail93 | group | gate_q4 | none | 94 | 94 | 0 | 0 | 0 | 0/0 | 1.79, 4.99, 471 | 3.11e-05 | 93 |  |
| 8 | levin | cc47 | analytic | band_10_12 | 1-4 | 12 | 12 | 0 | 0 | 5 | 0/0 | 134, 192, 286 | 1.61e-06 | 12 | {"bounded": 17} |
| 8 | levin | cc47 | analytic | band_10_12 | 5+ | 3 | 3 | 0 | 0 | 8 | 0/0 | 77.7, 77.7, 323 | 4.76e-06 | 3 | {"bounded": 11} |
| 8 | levin | cc47 | analytic | band_10_12 | none | 11 | 11 | 0 | 0 | 19 | 0/0 | 1.41, 14.6, 169 | 7.31e-07 | 11 | {"bounded": 30} |
| 8 | levin | cc47 | analytic | band_10_12 | unknown | 25 | 25 | 0 | 0 | 17 | 0/0 | 2.95, 81.5, 783 | 5.15e-06 | 25 | {"bounded": 42} |
| 8 | levin | cc47 | analytic | band_12_16 | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 8 | levin | cc47 | analytic | band_12_16 | 5+ | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 8 | levin | cc47 | analytic | band_12_16 | none | 5 | 5 | 0 | 3 | 28 | 0/0 | 3.33, 4.63, 16.4 | 7.51e-07 | 5 | {"bounded": 33, "undecided": 3} |
| 8 | levin | cc47 | analytic | band_12_16 | unknown | 8 | 8 | 0 | 0 | 48 | 0/0 | 11.1, 97.6, 158 | 1.81e-06 | 8 | {"bounded": 53, "undecided": 3} |
| 8 | levin | cc47 | analytic | band_16_20 | 1-4 | 3 | 3 | 0 | 0 | 15 | 0/0 | 60.5, 60.5, 326 | 7.09e-06 | 3 | {"bounded": 18} |
| 8 | levin | cc47 | analytic | band_16_20 | 5+ | 5 | 5 | 0 | 0 | 11 | 0/0 | 53.2, 60.5, 83.4 | 4.11e-06 | 5 | {"bounded": 16} |
| 8 | levin | cc47 | analytic | band_16_20 | none | 2 | 2 | 0 | 1 | 11 | 0/0 | 6.43, 6.93, 7.42 | 5.41e-05 | 2 | {"bounded": 13, "undecided": 1} |
| 8 | levin | cc47 | analytic | band_16_20 | unknown | 7 | 7 | 0 | 1 | 42 | 0/0 | 2.49, 21.7, 495 | 6.37e-06 | 7 | {"bounded": 45, "undecided": 5} |
| 8 | levin | cc47 | analytic | band_20_32 | 1-4 | 3 | 3 | 0 | 0 | 8 | 0/0 | 70.4, 157, 192 | 1.59e-06 | 3 | {"bounded": 11} |
| 8 | levin | cc47 | analytic | band_20_32 | 5+ | 8 | 8 | 0 | 0 | 25 | 0/0 | 11.1, 66, 204 | 1.12e-05 | 8 | {"bounded": 33} |
| 8 | levin | cc47 | analytic | band_20_32 | none | 3 | 3 | 0 | 3 | 45 | 0/0 | 8.26, 16.3, 54.5 | 9.56e-07 | 3 | {"undecided": 21, "bounded": 30} |
| 8 | levin | cc47 | analytic | band_20_32 | unknown | 15 | 15 | 0 | 8 | 80 | 0/0 | 2.13, 18.7, 389 | 2.91e-06 | 15 | {"bounded": 77, "undecided": 26} |
| 8 | levin | cc47 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 4, "understates": 2} |
| 8 | levin | cc47 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 8 | levin | cc47 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 8 | levin | cc47 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 64, "bounded": 31, "understates": 3} |
| 8 | levin | cc47 | analytic | band_8_10 | 1-4 | 3 | 3 | 0 | 0 | 10 | 0/0 | 157, 925, 1.23e+03 | 1.12e-05 | 3 | {"bounded": 13} |
| 8 | levin | cc47 | analytic | band_8_10 | 5+ | 2 | 2 | 0 | 0 | 4 | 0/0 | 193, 193, 193 | 2.78e-06 | 2 | {"bounded": 6} |
| 8 | levin | cc47 | analytic | band_8_10 | none | 14 | 14 | 0 | 0 | 24 | 0/0 | 2.18, 9.79, 765 | 8.16e-07 | 14 | {"bounded": 38} |
| 8 | levin | cc47 | analytic | band_8_10 | unknown | 29 | 29 | 0 | 0 | 18 | 0/0 | 4.05, 71.6, 7.48e+03 | 6.79e-07 | 29 | {"bounded": 47} |
| 8 | levin | cc47 | group | band_10_12 | 1-4 | 24 | 24 | 0 | 16 | 0 | 0/0 | 1.33, 22.1, 1.45e+04 | 7.73e-06 | 24 | {"understates": 8, "bounded": 13, "undecided": 19} |
| 8 | levin | cc47 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 25.1, 4.06e+03, 2e+06 | 8.66e-07 | 73 | {"bounded": 33, "undecided": 14, "understates": 26} |
| 8 | levin | cc47 | group | band_10_12 | none | 2 | 2 | 0 | 5 | 0 | 0/0 | 1.31, 1.31, 1.31 | 0.000115 | 2 | {"bounded": 5, "undecided": 2} |
| 8 | levin | cc47 | group | band_12_16 | 1-4 | 45 | 45 | 0 | 2 | 26 | 0/0 | 5.38, 44.6, 1.27e+04 | 8.14e-07 | 45 | {"understates": 16, "undecided": 40, "bounded": 17} |
| 8 | levin | cc47 | group | band_12_16 | 5+ | 84 | 84 | 0 | 0 | 1 | 0/0 | 11, 2.6e+03, 1.71e+06 | 1.15e-06 | 84 | {"undecided": 18, "understates": 32, "bounded": 35} |
| 8 | levin | cc47 | group | band_12_16 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"undecided": 2} |
| 8 | levin | cc47 | group | band_16_20 | 1-4 | 46 | 46 | 0 | 2 | 18 | 0/0 | 3.51, 23.9, 6.54e+03 | 7.67e-07 | 46 | {"undecided": 57, "understates": 3, "bounded": 6} |
| 8 | levin | cc47 | group | band_16_20 | 5+ | 83 | 83 | 0 | 0 | 2 | 0/0 | 14.4, 1.51e+03, 9.05e+05 | 5.02e-07 | 83 | {"bounded": 37, "understates": 41, "undecided": 7} |
| 8 | levin | cc47 | group | band_16_20 | none | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 8 | levin | cc47 | group | band_20_32 | 1-4 | 36 | 36 | 0 | 8 | 53 | 0/0 | 3.28, 14.5, 4.38e+03 | 2.5e-06 | 36 | {"undecided": 85, "understates": 8, "bounded": 4} |
| 8 | levin | cc47 | group | band_20_32 | 5+ | 99 | 99 | 0 | 0 | 2 | 0/0 | 5.81, 339, 1.3e+05 | 8.79e-06 | 99 | {"bounded": 48, "understates": 42, "undecided": 11} |
| 8 | levin | cc47 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 8 | levin | cc47 | group | band_20_32v | 1-4 | 6 | 6 | 0 | 2 | 88 | 0/0 | 1.65, 5.18, 18.7 | 2.79e-05 | 6 | {"bounded": 15, "undecided": 73, "understates": 8} |
| 8 | levin | cc47 | group | band_20_32v | 5+ | 12 | 11 | 1 | 0 | 90 | 0/0 | 0.914, 2.9e+03, 3.56e+04 | 0.00278 | 10 | {"bounded": 65, "understates": 35, "undecided": 2} |
| 8 | levin | cc47 | group | band_8_10 | 1-4 | 15 | 15 | 0 | 9 | 2 | 0/0 | 1.92, 264, 1.07e+05 | 7.74e-05 | 15 | {"undecided": 14, "bounded": 10, "understates": 2} |
| 8 | levin | cc47 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 50.6, 4.01e+03, 6.6e+05 | 7.01e-06 | 91 | {"understates": 42, "bounded": 44, "undecided": 5} |
| 8 | levin | cc47 | group | band_8_10 | none | 0 | 0 | 0 | 2 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1, "bounded": 2} |
| 8 | levin | cc93 | analytic | band_10_12 | 1-4 | 12 | 12 | 0 | 0 | 5 | 0/0 | 145, 191, 1.34e+04 | 6.67e-07 | 12 | {"bounded": 17} |
| 8 | levin | cc93 | analytic | band_10_12 | 5+ | 5 | 5 | 0 | 0 | 6 | 0/0 | 104, 106, 3.71e+03 | 2.98e-06 | 5 | {"bounded": 11} |
| 8 | levin | cc93 | analytic | band_10_12 | none | 4 | 4 | 0 | 0 | 26 | 0/0 | 17.7, 21.5, 25.3 | 2.18e-06 | 4 | {"bounded": 30} |
| 8 | levin | cc93 | analytic | band_10_12 | unknown | 22 | 22 | 0 | 0 | 20 | 0/0 | 3.91, 81.5, 174 | 4.15e-06 | 22 | {"bounded": 42} |
| 8 | levin | cc93 | analytic | band_12_16 | 1-4 | 1 | 1 | 0 | 0 | 5 | 0/0 | 365, 365, 365 | 1.46e-06 | 1 | {"bounded": 6} |
| 8 | levin | cc93 | analytic | band_12_16 | 5+ | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"bounded": 6} |
| 8 | levin | cc93 | analytic | band_12_16 | none | 11 | 11 | 0 | 2 | 23 | 0/0 | 4.39, 19.2, 60.8 | 1.44e-07 | 11 | {"bounded": 33, "undecided": 3} |
| 8 | levin | cc93 | analytic | band_12_16 | unknown | 9 | 9 | 0 | 0 | 47 | 0/0 | 2.82, 148, 153 | 7.33e-07 | 9 | {"bounded": 53, "undecided": 3} |
| 8 | levin | cc93 | analytic | band_16_20 | 1-4 | 2 | 2 | 0 | 0 | 16 | 0/0 | 70.8, 74, 77.2 | 6.82e-06 | 2 | {"bounded": 18} |
| 8 | levin | cc93 | analytic | band_16_20 | 5+ | 5 | 5 | 0 | 0 | 11 | 0/0 | 13.2, 69.7, 81.2 | 2.06e-06 | 5 | {"bounded": 16} |
| 8 | levin | cc93 | analytic | band_16_20 | none | 1 | 1 | 0 | 0 | 13 | 0/0 | 6.64, 6.64, 6.64 | 5.42e-05 | 1 | {"bounded": 13, "undecided": 1} |
| 8 | levin | cc93 | analytic | band_16_20 | unknown | 6 | 6 | 0 | 3 | 41 | 0/0 | 8.62, 52.3, 206 | 4.24e-06 | 6 | {"bounded": 45, "undecided": 5} |
| 8 | levin | cc93 | analytic | band_20_32 | 1-4 | 1 | 1 | 0 | 0 | 10 | 0/0 | 451, 451, 451 | 3.42e-07 | 1 | {"bounded": 11} |
| 8 | levin | cc93 | analytic | band_20_32 | 5+ | 8 | 8 | 0 | 0 | 25 | 0/0 | 18.3, 90.9, 458 | 3.55e-06 | 8 | {"bounded": 33} |
| 8 | levin | cc93 | analytic | band_20_32 | none | 4 | 4 | 0 | 7 | 40 | 0/0 | 6.51, 72.1, 132 | 2.15e-07 | 4 | {"undecided": 21, "bounded": 30} |
| 8 | levin | cc93 | analytic | band_20_32 | unknown | 9 | 9 | 0 | 5 | 89 | 0/0 | 4.13, 5.69, 94.4 | 5.81e-07 | 9 | {"bounded": 77, "undecided": 26} |
| 8 | levin | cc93 | analytic | band_20_32v | 1-4 | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 8 | levin | cc93 | analytic | band_20_32v | 5+ | 0 | 0 | 0 | 0 | 34 | 0/0 | —, —, — | — | 0 | {"bounded": 26, "undecided": 3, "understates": 5} |
| 8 | levin | cc93 | analytic | band_20_32v | none | 0 | 0 | 0 | 0 | 52 | 0/0 | —, —, — | — | 0 | {"bounded": 10, "undecided": 40, "understates": 2} |
| 8 | levin | cc93 | analytic | band_20_32v | unknown | 0 | 0 | 0 | 0 | 98 | 0/0 | —, —, — | — | 0 | {"undecided": 66, "bounded": 30, "understates": 2} |
| 8 | levin | cc93 | analytic | band_8_10 | 1-4 | 3 | 3 | 0 | 0 | 10 | 0/0 | 167, 260, 384 | 3.97e-05 | 3 | {"bounded": 13} |
| 8 | levin | cc93 | analytic | band_8_10 | 5+ | 3 | 3 | 0 | 0 | 3 | 0/0 | 177, 225, 225 | 4.35e-06 | 3 | {"bounded": 6} |
| 8 | levin | cc93 | analytic | band_8_10 | none | 7 | 7 | 0 | 0 | 31 | 0/0 | 3.9, 10.7, 192 | 1.35e-06 | 7 | {"bounded": 38} |
| 8 | levin | cc93 | analytic | band_8_10 | unknown | 23 | 23 | 0 | 0 | 24 | 0/0 | 3.19, 46.2, 900 | 1.15e-06 | 23 | {"bounded": 47} |
| 8 | levin | cc93 | group | band_10_12 | 1-4 | 19 | 19 | 0 | 9 | 12 | 0/0 | 1.55, 27.6, 6.94e+03 | 2.66e-05 | 19 | {"understates": 5, "bounded": 11, "undecided": 24} |
| 8 | levin | cc93 | group | band_10_12 | 5+ | 73 | 73 | 0 | 0 | 0 | 0/0 | 40.5, 9.52e+03, 1.74e+06 | 1.99e-07 | 73 | {"undecided": 23, "bounded": 19, "understates": 31} |
| 8 | levin | cc93 | group | band_10_12 | none | 2 | 2 | 0 | 2 | 3 | 0/0 | 1.56, 1.56, 1.56 | 0.000537 | 2 | {"bounded": 4, "undecided": 3} |
| 8 | levin | cc93 | group | band_12_16 | 1-4 | 45 | 45 | 0 | 1 | 27 | 0/0 | 5.81, 95.7, 1.88e+04 | 5.99e-07 | 45 | {"undecided": 49, "understates": 12, "bounded": 12} |
| 8 | levin | cc93 | group | band_12_16 | 5+ | 84 | 84 | 0 | 0 | 1 | 0/0 | 16.4, 6.43e+03, 2.96e+06 | 5.11e-07 | 84 | {"undecided": 34, "bounded": 33, "understates": 18} |
| 8 | levin | cc93 | group | band_12_16 | none | 0 | 0 | 0 | 0 | 2 | 0/0 | —, —, — | — | 0 | {"undecided": 2} |
| 8 | levin | cc93 | group | band_16_20 | 1-4 | 43 | 43 | 0 | 2 | 21 | 0/0 | 5.49, 36.7, 5.1e+03 | 1.03e-06 | 43 | {"undecided": 57, "understates": 3, "bounded": 6} |
| 8 | levin | cc93 | group | band_16_20 | 5+ | 83 | 83 | 0 | 0 | 2 | 0/0 | 16.5, 4.04e+03, 2.24e+06 | 1.48e-07 | 83 | {"bounded": 37, "understates": 41, "undecided": 7} |
| 8 | levin | cc93 | group | band_16_20 | none | 0 | 0 | 0 | 0 | 6 | 0/0 | —, —, — | — | 0 | {"undecided": 6} |
| 8 | levin | cc93 | group | band_20_32 | 1-4 | 33 | 33 | 0 | 10 | 54 | 0/0 | 4.54, 24.8, 170 | 2.07e-06 | 33 | {"undecided": 85, "understates": 8, "bounded": 4} |
| 8 | levin | cc93 | group | band_20_32 | 5+ | 99 | 99 | 0 | 0 | 2 | 0/0 | 9.91, 583, 2.27e+05 | 6.88e-06 | 99 | {"bounded": 48, "understates": 42, "undecided": 11} |
| 8 | levin | cc93 | group | band_20_32 | none | 0 | 0 | 0 | 0 | 1 | 0/0 | —, —, — | — | 0 | {"undecided": 1} |
| 8 | levin | cc93 | group | band_20_32v | 1-4 | 9 | 9 | 0 | 2 | 85 | 0/0 | 2.12, 10.5, 17.3 | 3.49e-06 | 9 | {"bounded": 15, "undecided": 73, "understates": 8} |
| 8 | levin | cc93 | group | band_20_32v | 5+ | 10 | 10 | 0 | 0 | 92 | 0/0 | 1, 9.59e+03, 4.71e+06 | 0.0014 | 8 | {"bounded": 64, "understates": 36, "undecided": 2} |
| 8 | levin | cc93 | group | band_8_10 | 1-4 | 15 | 15 | 0 | 2 | 9 | 0/0 | 6.33, 350, 2.35e+07 | 1.68e-05 | 15 | {"undecided": 13, "bounded": 13} |
| 8 | levin | cc93 | group | band_8_10 | 5+ | 91 | 91 | 0 | 0 | 0 | 0/0 | 47.8, 2.88e+04, 2.12e+06 | 1.27e-06 | 91 | {"bounded": 34, "understates": 35, "undecided": 22} |
| 8 | levin | cc93 | group | band_8_10 | none | 0 | 0 | 0 | 0 | 3 | 0/0 | —, —, — | — | 0 | {"bounded": 1, "undecided": 2} |
