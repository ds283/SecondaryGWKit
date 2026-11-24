# levin-phase-floor 07d Part A': table 7 (honesty and convergence) and the qualification
# written 2026-11-23T05:00:02Z on mps026298.phys.susx.ac.uk
# script score_a2.py sha256 8f61f20cc57662f1199486b474cd93e57d958890175572aa804933244a4ca7be
# tree: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-c2c57a3 HEAD c2c57a36cbad3780986baceb807dcb447975b498 (status --short: empty)
# main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/out/tables/scores_07d.json sha256 e3e332d1f7bff669841e15b1e2b53f99cd6c1555f911df5c11e2ee1c2016d421
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2/state/variants_a2.json sha256 f1ec25356f1da37b98c2dde534ddd87ea23469013737328202ff452dd5217193
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2/out/tables/t3_cc93.json sha256 aa749e9aab3dfd3423f888f0896c76f21258ea05716f5bb3d385921fc1226c23
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2/out/tables/t4_dt.json sha256 7e2f1a8f50bdd26274e85ebf320dc48e9371c5fd4436a41cd95d452a9d51e229

### Honesty (referenced calls, scored)

| code | honest | dishonest | unconverged |
|---|---|---|---|
| tree | 410 | 4 | 3 |
| V93 | 412 | 4 | 1 |
| V93tail | 412 | 4 | 1 |
| V93T8 | 412 | 4 | 1 |
| V93T10 | 412 | 4 | 1 |
| V93T12 | 411 | 5 | 1 |
| K | 413 | 3 | 1 |
| G | 416 | 0 | 1 |
| KT | 413 | 3 | 1 |

Part A's (its own codes, its scoring): {"tree": {"honest": 410, "dishonest": 4, "unconverged": 3}, "Eg": {"honest": 412, "dishonest": 4, "unconverged": 1}, "Ea": {"honest": 412, "dishonest": 4, "unconverged": 1}, "ET20": {"honest": 385, "unconverged": 32}, "ET12": {"honest": 413, "dishonest": 1, "unconverged": 3}, "Ep": {"honest": 412, "dishonest": 4, "unconverged": 1}, "T20": {"honest": 373, "unconverged": 43, "dishonest": 1}}

### By set

- Ra: tree {'honest': 60}; V93 {'honest': 60}; V93tail {'honest': 60}; V93T8 {'honest': 60}; V93T10 {'honest': 60}; V93T12 {'honest': 60}; K {'honest': 60}; G {'honest': 60}; KT {'honest': 60}
- Rt: tree {'honest': 60}; V93 {'honest': 60}; V93tail {'honest': 60}; V93T8 {'honest': 60}; V93T10 {'honest': 60}; V93T12 {'honest': 60}; K {'honest': 60}; G {'honest': 60}; KT {'honest': 60}
- regression: tree {'honest': 290, 'unconverged': 3}; V93 {'honest': 292, 'unconverged': 1}; V93tail {'honest': 292, 'unconverged': 1}; V93T8 {'honest': 292, 'unconverged': 1}; V93T10 {'honest': 292, 'unconverged': 1}; V93T12 {'honest': 291, 'dishonest': 1, 'unconverged': 1}; K {'honest': 289, 'dishonest': 3, 'unconverged': 1}; G {'honest': 292, 'unconverged': 1}; KT {'honest': 289, 'dishonest': 3, 'unconverged': 1}
- regression R2-28: tree {'dishonest': 4}; V93 {'dishonest': 4}; V93tail {'dishonest': 4}; V93T8 {'dishonest': 4}; V93T10 {'dishonest': 4}; V93T12 {'dishonest': 4}; K {'honest': 4}; G {'honest': 4}; KT {'honest': 4}

### Converged and dishonest calls

- R:a:R2-28 (regression R2-28): tree dishonest 1.14, V93 dishonest 1.13, V93tail dishonest 1.12, V93T8 dishonest 1.11, V93T10 dishonest 1.1, V93T12 dishonest 1.1, K honest 0.053, G honest 4.82e-08, KT honest 0.0522
- R:a:R2-28~o (regression R2-28): tree dishonest 1.14, V93 dishonest 1.13, V93tail dishonest 1.12, V93T8 dishonest 1.11, V93T10 dishonest 1.1, V93T12 dishonest 1.1, K honest 0.053, G honest 5.13e-08, KT honest 0.0522
- R:a:W2-0-12977-5 (regression): tree honest 0.0485, V93 honest 0.0486, V93tail honest 0.0469, V93T8 honest 0.0538, V93T10 honest 0.0498, V93T12 dishonest 2.79, K honest 0.0181, G honest 0.000263, KT honest 0.014
- R:c-1e-7-0-13757-5 (regression R2-28): tree dishonest 1.14, V93 dishonest 1.13, V93tail dishonest 1.12, V93T8 dishonest 1.11, V93T10 dishonest 1.1, V93T12 dishonest 1.1, K honest 0.053, G honest 4.82e-08, KT honest 0.0522
- R:c-1e-7-1-1759-5 (regression): tree honest 0.000747, V93 honest 0.00153, V93tail honest 0.000677, V93T8 honest 0.00152, V93T10 honest 0.0108, V93T12 honest 0.0104, K dishonest 1.19, G honest 2.27e-08, KT dishonest 1.19
- R:c-1e-8-0-13757-5 (regression R2-28): tree dishonest 1.14, V93 dishonest 1.13, V93tail dishonest 1.12, V93T8 dishonest 1.11, V93T10 dishonest 1.1, V93T12 dishonest 1.1, K honest 0.053, G honest 5.13e-08, KT honest 0.0522
- R:c-1e-8-1-1759-5 (regression): tree honest 0.000745, V93 honest 0.00153, V93tail honest 0.000674, V93T8 honest 0.00152, V93T10 honest 0.0108, V93T12 honest 0.0104, K dishonest 1.19, G honest 2.27e-08, KT dishonest 1.19
- R:m-1e-8-3-35006-g4 (regression): tree honest 0.0201, V93 honest 0.0165, V93tail honest 0.0233, V93T8 honest 0.0197, V93T10 honest 0.0266, V93T12 honest 0.00521, K dishonest 1.56, G honest 0.000119, KT dishonest 1.24

### Convergence, reach and cost (all 4,220 calls)

| variant | touched | untouched | conv lost | conv gained | unconverged (tree) | regions/tree | solves/tree | wall/tree | route changes (calls) | knot gates (calls) | knot breaks (calls) | chase stops tree/var | stopped leaves tree/var | stopped decl tree/var | stopped-window leaves differ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V93 | 3397 | 823 | 0 | 7 | 2 (9) | 0.912 | 0.908 | 2.03 | 0 (0) | 0 (0) | 0 (0) | 1644/1547 | 22822/19482 | 4.16e-19/9.79e-21 | 208 |
| V93tail | 3397 | 823 | 1 | 7 | 3 (9) | 0.992 | 0.966 | 1.52 | 0 (0) | 0 (0) | 0 (0) | 1644/1602 | 22822/21673 | 4.16e-19/9.79e-21 | 208 |
| V93T8 | 3513 | 707 | 0 | 7 | 2 (9) | 0.921 | 0.916 | 2.23 | 22490 (2388) | 0 (0) | 0 (0) | 1644/1542 | 22822/20712 | 4.16e-19/9.78e-21 | 208 |
| V93T10 | 3625 | 595 | 0 | 7 | 2 (9) | 0.907 | 0.901 | 2.32 | 37574 (3158) | 0 (0) | 0 (0) | 1644/1546 | 22822/21192 | 4.16e-19/9.75e-21 | 208 |
| V93T12 | 3687 | 533 | 0 | 7 | 2 (9) | 0.888 | 0.884 | 2.27 | 48669 (3328) | 0 (0) | 0 (0) | 1644/1545 | 22822/21007 | 4.16e-19/9.63e-21 | 208 |
| K | 3406 | 814 | 0 | 7 | 2 (9) | 0.789 | 0.789 | 1.57 | 0 (0) | 0 (0) | 220326 (721) | 1644/1461 | 22822/13772 | 4.16e-19/9.79e-21 | 208 |
| G | 3406 | 814 | 0 | 7 | 2 (9) | 4.519 | 4.550 | 10.35 | 0 (0) | 305221 (686) | 0 (0) | 1644/1080 | 22822/7284 | 4.16e-19/9.79e-21 | 208 |
| KT | 3522 | 698 | 1 | 7 | 3 (9) | 0.791 | 0.788 | 1.68 | 17010 (2379) | 0 (0) | 218166 (721) | 1644/1461 | 22822/14317 | 4.16e-19/9.78e-21 | 208 |

### Qualification (section 9.3)

- **V93**: does not qualify; new converged-dishonest 0 []; declaration pair93 qualifies True; threshold 6 pi bounded both sides False {"group": [["fails", 700, 2], ["bounded", 800, 0]], "analytic": [["bounded", 155, 0], ["bounded", 379, 0]]}
- **V93tail**: does not qualify; new converged-dishonest 0 []; declaration tail93 qualifies True; threshold 6 pi bounded both sides False {"group": [["fails", 700, 2], ["bounded", 800, 0]], "analytic": [["bounded", 155, 0], ["bounded", 379, 0]]}
- **V93T8**: QUALIFIES; new converged-dishonest 0 []; declaration pair93 qualifies True; threshold 8 pi bounded both sides True {"group": [["bounded", 587, 0], ["bounded", 920, 0]], "analytic": [["bounded", 136, 0], ["bounded", 471, 0]]}
- **V93T10**: QUALIFIES; new converged-dishonest 0 []; declaration pair93 qualifies True; threshold 10 pi bounded both sides True {"group": [["bounded", 474, 0], ["bounded", 1040, 0]], "analytic": [["bounded", 118, 0], ["bounded", 575, 0]]}
- **V93T12**: does not qualify; new converged-dishonest 1 ['R:a:W2-0-12977-5']; declaration pair93 qualifies True; threshold 12 pi bounded both sides True {"group": [["bounded", 380, 0], ["bounded", 1160, 0]], "analytic": [["bounded", 88, 0], ["bounded", 675, 0]]}
- **K**: does not qualify; new converged-dishonest 3 ['R:c-1e-7-1-1759-5', 'R:c-1e-8-1-1759-5', 'R:m-1e-8-3-35006-g4']; declaration pair93 qualifies True; threshold 6 pi bounded both sides False {"group": [["fails", 700, 2], ["bounded", 800, 0]], "analytic": [["bounded", 155, 0], ["bounded", 379, 0]]}; knot bisection (K) leaves the gate at its V threshold
- **G**: does not qualify; new converged-dishonest 0 []; declaration pair93 qualifies True; threshold 6 pi bounded both sides False {"group": [["fails", 700, 2], ["bounded", 800, 0]], "analytic": [["bounded", 155, 0], ["bounded", 379, 0]]}; the knot gate (G) has no table-4 row of its own; judged at its V threshold
- **KT**: does not qualify; new converged-dishonest 3 ['R:c-1e-7-1-1759-5', 'R:c-1e-8-1-1759-5', 'R:m-1e-8-3-35006-g4']; declaration pair93 qualifies True; threshold 8 pi bounded both sides True {"group": [["bounded", 587, 0], ["bounded", 920, 0]], "analytic": [["bounded", 136, 0], ["bounded", 471, 0]]}; knot bisection (K) leaves the gate at its V threshold
