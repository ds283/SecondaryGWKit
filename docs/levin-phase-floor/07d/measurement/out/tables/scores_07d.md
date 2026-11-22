levin-phase-floor 07d Part A: tables 4 and 5, and the qualification
written 2026-11-22T12:57:18Z on mps026298.phys.susx.ac.uk
script score_07d.py sha256 b37808c2a5ad55bd9b28e14403644a9ad186ee49f7faa1af90be44f0473b7d93
tree h: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-49ce964 HEAD 49ce96429da797efe55f546d65a23ebcd0099f33 (status --short: empty)
main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07c/measurement/out/tables/scores_07c.json sha256 f2f10ea3fb451669ccbf6de136c67322413f59615330594d5a87ac3beeedb3c5
input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/out/tables/d_e.json sha256 c8269db1a296d2a895eccd0f42e47b1ce33563cf4981324dd7ba2a93fa23a4eb
input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/out/tables/d_t.json sha256 379d2b585b347492bfeb1c9c9c0fb9405b3b80cdad16e19efea708b352d09404

Referenced calls: 425, scored 417.

## Table 4 (honesty), every scored call

| code | converged & honest | converged & dishonest | unconverged |
|---|---|---|---|
| tree | 410 | 4 | 3 |
| Eg | 412 | 4 | 1 |
| Ea | 412 | 4 | 1 |
| ET20 | 385 | 0 | 32 |
| ET12 | 413 | 1 | 3 |
| Ep | 412 | 4 | 1 |
| T20 | 373 | 1 | 43 |

### by set

| set | code | honest | dishonest | unconverged |
|---|---|---|---|---|
| Ra | tree | 60 | 0 | 0 |
| Ra | Eg | 60 | 0 | 0 |
| Ra | Ea | 60 | 0 | 0 |
| Ra | ET20 | 60 | 0 | 0 |
| Ra | ET12 | 60 | 0 | 0 |
| Ra | Ep | 60 | 0 | 0 |
| Ra | T20 | 59 | 0 | 1 |
| Rt | tree | 60 | 0 | 0 |
| Rt | Eg | 60 | 0 | 0 |
| Rt | Ea | 60 | 0 | 0 |
| Rt | ET20 | 58 | 0 | 2 |
| Rt | ET12 | 60 | 0 | 0 |
| Rt | Ep | 60 | 0 | 0 |
| Rt | T20 | 56 | 0 | 4 |
| regression | tree | 290 | 0 | 3 |
| regression | Eg | 292 | 0 | 1 |
| regression | Ea | 292 | 0 | 1 |
| regression | ET20 | 267 | 0 | 26 |
| regression | ET12 | 289 | 1 | 3 |
| regression | Ep | 292 | 0 | 1 |
| regression | T20 | 254 | 1 | 38 |
| regression R2-28 | tree | 0 | 4 | 0 |
| regression R2-28 | Eg | 0 | 4 | 0 |
| regression R2-28 | Ea | 0 | 4 | 0 |
| regression R2-28 | ET20 | 0 | 0 | 4 |
| regression R2-28 | ET12 | 4 | 0 | 0 |
| regression R2-28 | Ep | 0 | 4 | 0 |
| regression R2-28 | T20 | 4 | 0 | 0 |

### every converged-and-dishonest or unconverged scored call

| sid | set | tree | Eg | Ea | ET20 | ET12 | Ep | T20 |
|---|---|---|---|---|---|---|---|---|
| R:a:R1-x247-7 | regression | hones 0.00597 | hones 0.00579 | hones 0.00579 | uncon 0.00576 | hones 0.0239 | hones 0.00555 | uncon 5.01e-07 |
| R:a:R1-x247-8 | regression | hones 0.00597 | hones 0.00579 | hones 0.00579 | uncon 0.00576 | hones 0.0239 | hones 0.00555 | uncon 5.05e-07 |
| R:a:R1-x310-7 | regression | hones 0.00186 | hones 0.000525 | hones 0.000525 | hones 0.000514 | hones 0.0278 | hones 0.00052 | uncon 1.7e-08 |
| R:a:R1-x310-8 | regression | hones 0.00186 | hones 0.000525 | hones 0.000524 | hones 0.000514 | hones 0.0278 | hones 0.000519 | uncon 1.64e-08 |
| R:a:R1-x353-7 | regression | hones 0.00772 | hones 0.00754 | hones 0.00754 | hones 0.00562 | hones 0.00729 | hones 0.00747 | uncon 0.00087 |
| R:a:R1-x353-8 | regression | hones 0.00772 | hones 0.00754 | hones 0.00754 | hones 0.00562 | hones 0.00729 | hones 0.00747 | uncon 0.00087 |
| R:a:R2-08~o | regression | hones 0.805 | hones 0.0141 | hones 0.0141 | hones 0.00769 | hones 0.0179 | hones 0.0132 | disho 1.12 |
| R:a:R2-12 | regression | hones 0.0435 | hones 0.043 | hones 0.043 | hones 5.72e-05 | hones 0.00803 | hones 0.0431 | uncon 6.35e-07 |
| R:a:R2-12~o | regression | hones 0.0435 | hones 0.043 | hones 0.043 | hones 5.72e-05 | hones 0.00803 | hones 0.0431 | uncon 6.35e-07 |
| R:a:R2-16 | regression | hones 0.00709 | hones 0.00708 | hones 0.00708 | uncon 2.16e-09 | hones 0.0385 | hones 0.00709 | hones 1.22e-05 |
| R:a:R2-16~o | regression | hones 0.00709 | hones 0.00708 | hones 0.00708 | uncon 2.16e-09 | hones 0.0385 | hones 0.00709 | hones 1.22e-05 |
| R:a:R2-28 | regression R2-28 | disho 1.14 | disho 1.13 | disho 1.13 | uncon 7.44e-08 | hones 0.724 | disho 1.13 | hones 0.000164 |
| R:a:R2-28~o | regression R2-28 | disho 1.14 | disho 1.13 | disho 1.13 | uncon 7.44e-08 | hones 0.724 | disho 1.13 | hones 0.000164 |
| R:a:R3a-3-39138-g2 | regression | hones 0.0136 | hones 0.00385 | hones 0.00385 | hones 0.00692 | hones 0.0038 | hones 0.00383 | uncon 7.73e-07 |
| R:a:R3a-3-39138-g3 | regression | hones 0.0136 | hones 0.00385 | hones 0.00385 | hones 0.00692 | hones 0.0038 | hones 0.00383 | uncon 7.73e-07 |
| R:a:W2-0-12977-5 | regression | hones 0.0485 | hones 0.0396 | hones 0.0396 | hones 0.0362 | disho 3.25 | hones 0.0366 | hones 0.0412 |
| R:a:p14501_7 | regression | hones 0.000582 | hones 0.000584 | hones 0.000584 | hones 0.000733 | hones 0.00393 | hones 0.000584 | uncon 1.35e-05 |
| R:a:p14501_8 | regression | hones 0.00107 | hones 0.00877 | hones 0.00877 | hones 0.00732 | hones 0.00454 | hones 0.00465 | uncon 0.000274 |
| R:a:p35006_7 | regression | hones 0.00074 | hones 0.000742 | hones 0.000742 | hones 0.00294 | hones 0.005 | hones 0.000742 | uncon 2.1e-06 |
| R:a:p35006_8 | regression | hones 0.00155 | hones 0.00837 | hones 0.00837 | uncon 0.00196 | hones 0.00355 | hones 0.00463 | uncon 1.15e-07 |
| R:a:v00 | regression | hones 0.0121 | hones 0.0104 | hones 0.0104 | hones 2.28e-05 | hones 0.00508 | hones 0.0103 | uncon 1.87e-07 |
| R:a:v03 | regression | hones 0.00174 | hones 0.00158 | hones 0.00158 | hones 0.00638 | hones 0.00146 | hones 0.00151 | uncon 0.00351 |
| R:a:v04 | regression | uncon 0.000156 | hones 0.000789 | hones 0.000789 | hones 0.000628 | hones 0.000837 | hones 0.000734 | uncon 7.8e-05 |
| R:a:v09 | regression | hones 0.00112 | hones 0.00484 | hones 0.00484 | hones 0.00285 | hones 0.00406 | hones 0.0078 | uncon 0.00013 |
| R:c-1e-7-0-13757-5 | regression R2-28 | disho 1.14 | disho 1.13 | disho 1.13 | uncon 7.44e-08 | hones 0.724 | disho 1.13 | hones 0.000164 |
| R:c-1e-7-0-13757-6 | regression | hones 0.138 | hones 0.105 | hones 0.105 | hones 2.33e-06 | hones 0.00108 | hones 0.098 | uncon 7.22e-08 |
| R:c-1e-7-1-1759-5 | regression | hones 0.000747 | hones 0.00106 | hones 0.00106 | uncon 6.49e-09 | hones 0.0102 | hones 0.00104 | uncon 4.59e-09 |
| R:c-1e-7-1-1761-6 | regression | hones 0.00134 | hones 0.00134 | hones 0.00134 | uncon 5.14e-09 | hones 0.00339 | hones 0.00134 | hones 9.7e-09 |
| R:c-1e-7-1-1841-5 | regression | hones 0.00137 | hones 0.000942 | hones 0.000942 | hones 9.87e-08 | hones 0.000273 | hones 0.000711 | uncon 1.02e-08 |
| R:c-1e-7-1-2217-1 | regression | hones 0.0119 | hones 0.00452 | hones 0.00452 | uncon 2.02e-06 | hones 0.138 | hones 0.00407 | hones 1.14e-06 |
| R:c-1e-7-1-34591-1 | regression | hones 0.0305 | hones 0.0305 | hones 0.0305 | hones 0.00158 | hones 0.0943 | hones 0.0305 | uncon 7.2e-08 |
| R:c-1e-8-0-13757-5 | regression R2-28 | disho 1.14 | disho 1.13 | disho 1.13 | uncon 7.44e-08 | hones 0.724 | disho 1.13 | hones 0.000164 |
| R:c-1e-8-0-13757-6 | regression | hones 0.138 | hones 0.105 | hones 0.105 | hones 2.33e-06 | hones 0.00108 | hones 0.098 | uncon 7.22e-08 |
| R:c-1e-8-1-1759-5 | regression | hones 0.000745 | hones 0.00106 | hones 0.00106 | uncon 6.48e-09 | hones 0.0102 | hones 0.00103 | uncon 4.57e-09 |
| R:c-1e-8-1-1761-6 | regression | hones 0.00134 | hones 0.00134 | hones 0.00134 | uncon 5.14e-09 | hones 0.00339 | hones 0.00134 | hones 9.69e-09 |
| R:c-1e-8-1-1841-5 | regression | hones 0.00137 | hones 0.000942 | hones 0.000942 | hones 9.87e-08 | hones 0.000273 | hones 0.000711 | uncon 1.02e-08 |
| R:c-1e-8-1-2217-1 | regression | hones 0.0119 | hones 0.00452 | hones 0.00452 | uncon 2.02e-06 | hones 0.138 | hones 0.00407 | hones 1.14e-06 |
| R:c-1e-8-1-34591-1 | regression | hones 0.0305 | hones 0.0305 | hones 0.0305 | hones 0.00158 | hones 0.0943 | hones 0.0305 | uncon 8.03e-08 |
| R:c-1e-8-3-13490-2 | regression | uncon 0.00859 | uncon 0.0086 | uncon 0.0086 | uncon 0.0127 | uncon 0.00831 | uncon 0.0206 | uncon 0.0102 |
| R:c-1e-8-3-13615-3 | regression | uncon 0.00834 | hones 0.0203 | hones 0.0203 | hones 0.0137 | hones 0.0203 | hones 0.0265 | hones 0.0108 |
| R:m-1e-7-0-1941-g1 | regression | hones 0.00839 | hones 0.00886 | hones 0.00886 | uncon 6.85e-05 | hones 0.0102 | hones 0.00886 | hones 0.000325 |
| R:m-1e-7-0-2016-g1 | regression | hones 0.00781 | hones 0.00831 | hones 0.00831 | uncon 4.97e-05 | hones 0.00979 | hones 0.00831 | hones 0.000296 |
| R:m-1e-7-2-2186-g1 | regression | hones 0.022 | hones 0.0208 | hones 0.0208 | uncon 0.00771 | hones 0.0375 | hones 0.0206 | hones 0.01 |
| R:m-1e-7-2-38635-g6 | regression | hones 0.000287 | hones 0.000245 | hones 0.000245 | hones 0.000162 | hones 0.00105 | hones 0.000237 | uncon 4.01e-05 |
| R:m-1e-7-3-13615-g4 | regression | hones 0.11 | hones 0.0938 | hones 0.0938 | hones 0.0734 | hones 0.0715 | hones 0.0922 | uncon 8.09e-05 |
| R:m-1e-8-0-1941-g1 | regression | hones 0.00839 | hones 0.00886 | hones 0.00886 | uncon 6.85e-05 | hones 0.0102 | hones 0.00886 | hones 0.000325 |
| R:m-1e-8-0-2016-g1 | regression | hones 0.00781 | hones 0.00831 | hones 0.00831 | uncon 4.97e-05 | hones 0.00979 | hones 0.00831 | hones 0.000296 |
| R:m-1e-8-0-33996-g4 | regression | hones 0.00245 | hones 0.00184 | hones 0.00184 | hones 0.00102 | hones 0.000847 | hones 0.00168 | uncon 8.52e-05 |
| R:m-1e-8-1-11062-g6 | regression | hones 0.00542 | hones 0.00726 | hones 0.00726 | hones 0.00472 | hones 0.00287 | hones 0.00536 | uncon 0.00111 |
| R:m-1e-8-1-1728-g5 | regression | hones 0.0539 | hones 0.00898 | hones 0.00898 | uncon 0.000657 | uncon 0.000771 | hones 0.0088 | hones 0.0665 |
| R:m-1e-8-1-37676-g2 | regression | hones 0.00139 | hones 0.003 | hones 0.003 | uncon 0.00621 | hones 0.00415 | hones 0.00268 | hones 0.0117 |
| R:m-1e-8-1-37676-g3 | regression | hones 0.00139 | hones 0.003 | hones 0.003 | uncon 0.00621 | hones 0.00415 | hones 0.00268 | hones 0.0117 |
| R:m-1e-8-2-2186-g1 | regression | hones 0.022 | hones 0.0208 | hones 0.0208 | uncon 0.00771 | hones 0.0375 | hones 0.0206 | hones 0.01 |
| R:m-1e-8-2-38635-g6 | regression | hones 0.000286 | hones 0.000244 | hones 0.000244 | hones 0.000163 | hones 0.00105 | hones 0.000236 | uncon 4.02e-05 |
| R:m-1e-8-3-35020-g1 | regression | hones 0.0149 | hones 0.0201 | hones 0.0201 | uncon 0.00102 | uncon 0.00116 | hones 0.0226 | hones 0.00579 |
| R:x-1e-7-0-12660-6 | regression | hones 0.0015 | hones 0.00151 | hones 0.00151 | uncon 0.000209 | hones 0.00112 | hones 0.00151 | uncon 0.000118 |
| R:x-1e-7-0-14318-3 | regression | hones 2.64e-05 | hones 2.52e-05 | hones 2.52e-05 | uncon 0.00629 | hones 0.000695 | hones 2.64e-05 | hones 0.246 |
| R:x-1e-7-1-1728-5 | regression | hones 0.0062 | hones 0.00617 | hones 0.00617 | uncon 0.00373 | hones 0.011 | hones 0.00619 | uncon 4.32e-05 |
| R:x-1e-7-1-37676-2 | regression | hones 0.000289 | hones 0.000873 | hones 0.000873 | hones 0.303 | hones 0.00245 | hones 0.000862 | uncon 2.07e-05 |
| R:x-1e-7-1-37676-3 | regression | hones 0.000289 | hones 0.000873 | hones 0.000873 | hones 0.303 | hones 0.00245 | hones 0.000862 | uncon 2.07e-05 |
| R:x-1e-7-3-13490-2 | regression | hones 0.0395 | hones 0.0368 | hones 0.0368 | uncon 0.00112 | hones 0.000543 | hones 0.00309 | hones 0.00145 |
| R:x-1e-7-3-34802-4 | regression | hones 0.00187 | hones 0.00187 | hones 0.00187 | hones 0.00961 | hones 0.000808 | hones 0.00187 | uncon 2.24e-05 |
| R:x-1e-7-3-35006-4 | regression | hones 0.00219 | hones 0.00219 | hones 0.00219 | hones 0.0289 | hones 0.00715 | hones 0.00219 | uncon 2.27e-05 |
| d:h-1e-7-0-12851|a20 | Ra | hones 0.0726 | hones 0.0726 | hones 0.0726 | hones 0.000277 | hones 0.0726 | hones 0.0726 | uncon 2.1e-06 |
| d:h-1e-7-0-1940|g5 | Rt~o | hones 0.162 | hones 0.135 | hones 0.135 | uncon 3.49e-08 | hones 0.031 | hones 0.146 | uncon 2.62e-07 |
| d:h-1e-7-1-38744|g3 | Rt | hones 0.0423 | hones 0.0419 | hones 0.0419 | hones 2.22e-07 | hones 0.0521 | hones 0.0419 | uncon 3.13e-08 |
| d:h-1e-8-0-1940|g5 | Rt | hones 0.162 | hones 0.135 | hones 0.135 | uncon 3.49e-08 | hones 0.031 | hones 0.146 | uncon 2.62e-07 |
| d:h-1e-8-1-38744|g3 | Rt~o | hones 0.0423 | hones 0.0419 | hones 0.0419 | hones 2.22e-07 | hones 0.0521 | hones 0.0419 | uncon 3.13e-08 |

## Table 5 (convergence, reach, cost) against the tree

| variant | calls | touched | conv gained | conv lost | regions / tree | wall / tree | route-changed solves (calls) | CC solves re-declared (noise term larger) | window-stop calls changed | chase stops / leaves / bisections (tree) | stopped leaves' declaration / tree | Rule A changes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Eg | 4220 | 1964 | 7 | 0 | 0.969 | 1.309 | 0 (0) | 197510 (197256) | 110 | 1566/20483/20763 (1644/22822/21683) | 0.0236 | 0 |
| Ea | 4220 | 2156 | 7 | 0 | 0.945 | 1.327 | 0 (0) | 218202 (217917) | 176 | 1566/20483/20763 (1644/22822/21683) | 0.0236 | 0 |
| ET20 | 4220 | 3177 | 7 | 74 | 1.677 | 2.674 | 120147 (3500) | 557938 (556728) | 186 | 1605/26212/20543 (1644/22822/21683) | 204 | 0 |
| ET12 | 4220 | 2672 | 7 | 2 | 0.960 | 1.511 | 49628 (3328) | 272039 (271483) | 178 | 1576/23060/20846 (1644/22822/21683) | 0.0292 | 0 |
| Ep | 4220 | 2156 | 7 | 0 | 0.960 | 0.972 | 0 (0) | 231446 (223558) | 176 | 1604/21810/21088 (1644/22822/21683) | 0.13 | 0 |
| T20 | 4220 | 2575 | 4 | 114 | 8.759 | 5.760 | 127988 (3500) | 0 (0) | 180 | 1779/28036/21457 (1644/22822/21683) | 1.55e+03 | 0 |

## Qualification

| variant | new converged & dishonest | estimate (qualifies on table 2) | threshold (bounded both sides on table 3) | qualifies |
|---|---|---|---|---|
| Eg | 0 [] | nest3N4 (yes) | 6pi (no) | no |
| Ea | 0 [] | nest3N4 (yes) | 6pi (no) | no |
| ET20 | 0 [] | nest3N4 (yes) | 20pi (yes, Levin side vacuous: no leaf above it) | yes (vacuously on its threshold) |
| ET12 | 1 ['R:a:W2-0-12977-5'] | nest3N4 (yes) | 12pi (no) | no |
| Ep | 0 [] | pairN4 (no) | 6pi (no) | no |
| T20 | 1 ['R:a:R2-08~o'] | tree (no) | 20pi (no, Levin side vacuous: no leaf above it) | no |

## (Ra) at 07c's code (beside the table, not folded in)

| sid | set | scored | spread | converged | ratio | verdict |
|---|---|---|---|---|---|---|
| d:h-1e-7-0-12767|a20 | Ra | True | 6.1e-29 | True | 0.000172 | honest |
| d:h-1e-7-0-12851|a20 | Ra | True | 1.5e-26 | True | 0.0726 | honest |
| d:h-1e-7-0-12859|a20 | Ra | True | 2.2e-25 | True | 0.0165 | honest |
| d:h-1e-7-0-1778|a16 | Ra | True | 9.9e-29 | True | 0.000208 | honest |
| d:h-1e-7-0-1824|a12 | Ra | True | 1.2e-27 | True | 0.00205 | honest |
| d:h-1e-7-0-2098|a20 | Ra | True | 7.7e-29 | True | 0.0747 | honest |
| d:h-1e-7-0-33914|a14 | Ra~o | True | 1.3e-29 | True | 0.0746 | honest |
| d:h-1e-7-0-34041|a11 | Ra | True | 8.9e-29 | True | 0.000112 | honest |
| d:h-1e-7-0-38350|a18 | Ra | True | 4.5e-27 | True | 0.000784 | honest |
| d:h-1e-7-0-38461|a16 | Ra | True | 1.1e-28 | True | 0.062 | honest |
| d:h-1e-7-1-14573|a10 | Ra | True | 8.8e-30 | True | 0.0172 | honest |
| d:h-1e-7-1-34528|a16 | Ra | True | 5.2e-29 | True | 0.000629 | honest |
| d:h-1e-7-1-34528|a18 | Ra | True | 5.2e-29 | True | 0.000629 | honest |
| d:h-1e-7-1-35127|a6 | Ra | True | 1.5e-28 | True | 0.0141 | honest |
| d:h-1e-7-1-35261|a14 | Ra | True | 1.6e-28 | True | 0.00509 | honest |
| d:h-1e-7-1-38143|a19 | Ra~o | True | 8.4e-28 | True | 0.158 | honest |
| d:h-1e-7-1-38741|a18 | Ra | True | 7.5e-30 | True | 0.0064 | honest |
| d:h-1e-7-2-13626|a15 | Ra~o | True | 1.3e-29 | True | 0.0649 | honest |
| d:h-1e-7-2-13628|a7 | Ra | True | 3.7e-30 | True | 0.0533 | honest |
| d:h-1e-7-2-1577|a18 | Ra | True | 7.5e-29 | True | 0.00267 | honest |
| d:h-1e-7-2-1701|a17 | Ra | True | 2.7e-29 | True | 0.423 | honest |
| d:h-1e-7-2-34576|a11 | Ra~o | True | 5.5e-27 | True | 0.0386 | honest |
| d:h-1e-7-2-35084|a10 | Ra | True | 2.3e-29 | True | 0.0285 | honest |
| d:h-1e-7-3-1227|a14 | Ra | True | 3.4e-28 | True | 0.0712 | honest |
| d:h-1e-7-3-1275|a8 | Ra~o | True | 2.3e-27 | True | 1.78e-05 | honest |
| d:h-1e-7-3-13482|a7 | Ra | True | 1.7e-26 | True | 0.00681 | honest |
| d:h-1e-7-3-13521|a12 | Ra | True | 4.5e-24 | True | 0.00127 | honest |
| d:h-1e-7-3-35995|a21 | Ra | True | 1.5e-26 | True | 0.102 | honest |
| d:h-1e-7-3-37359|a18 | Ra | True | 3.1e-28 | True | 0.000277 | honest |
| d:h-1e-7-3-38240|a5 | Ra | True | 7.4e-30 | True | 0.157 | honest |
| d:h-1e-8-0-12767|a20 | Ra~o | True | 6.1e-29 | True | 0.000251 | honest |
| d:h-1e-8-0-12851|a20 | Ra~o | True | 1.5e-26 | True | 0.0848 | honest |
| d:h-1e-8-0-12859|a20 | Ra~o | True | 2.2e-25 | True | 0.0168 | honest |
| d:h-1e-8-0-1778|a16 | Ra~o | True | 9.9e-29 | True | 0.0005 | honest |
| d:h-1e-8-0-1824|a12 | Ra~o | True | 1.2e-27 | True | 0.000396 | honest |
| d:h-1e-8-0-2098|a20 | Ra~o | True | 7.7e-29 | True | 0.017 | honest |
| d:h-1e-8-0-33914|a14 | Ra | True | 1.3e-29 | True | 0.0746 | honest |
| d:h-1e-8-0-34041|a11 | Ra~o | True | 8.9e-29 | True | 0.000172 | honest |
| d:h-1e-8-0-38350|a18 | Ra~o | True | 4.5e-27 | True | 0.00322 | honest |
| d:h-1e-8-0-38461|a16 | Ra~o | True | 1.1e-28 | True | 0.00978 | honest |
| d:h-1e-8-1-14573|a10 | Ra~o | True | 8.8e-30 | True | 0.00319 | honest |
| d:h-1e-8-1-34528|a16 | Ra~o | True | 5.2e-29 | True | 0.000572 | honest |
| d:h-1e-8-1-34528|a18 | Ra~o | True | 5.2e-29 | True | 0.000572 | honest |
| d:h-1e-8-1-35127|a6 | Ra~o | True | 1.5e-28 | True | 0.0112 | honest |
| d:h-1e-8-1-35261|a14 | Ra~o | True | 1.6e-28 | True | 0.00785 | honest |
| d:h-1e-8-1-38143|a19 | Ra | True | 8.4e-28 | True | 0.117 | honest |
| d:h-1e-8-1-38741|a18 | Ra~o | True | 7.5e-30 | True | 0.00112 | honest |
| d:h-1e-8-2-13626|a15 | Ra | True | 1.3e-29 | True | 0.0169 | honest |
| d:h-1e-8-2-13628|a7 | Ra~o | True | 3.7e-30 | True | 0.00543 | honest |
| d:h-1e-8-2-1577|a18 | Ra~o | True | 7.5e-29 | True | 0.00121 | honest |
| d:h-1e-8-2-1701|a17 | Ra~o | True | 2.7e-29 | True | 0.0033 | honest |
| d:h-1e-8-2-34576|a11 | Ra | True | 5.5e-27 | True | 0.0244 | honest |
| d:h-1e-8-2-35084|a10 | Ra~o | True | 2.3e-29 | True | 0.00681 | honest |
| d:h-1e-8-3-1227|a14 | Ra~o | True | 3.4e-28 | True | 0.0794 | honest |
| d:h-1e-8-3-1275|a8 | Ra | True | 2.3e-27 | True | 1.78e-05 | honest |
| d:h-1e-8-3-13482|a7 | Ra~o | True | 1.7e-26 | True | 0.00681 | honest |
| d:h-1e-8-3-13521|a12 | Ra~o | True | 4.5e-24 | True | 0.00127 | honest |
| d:h-1e-8-3-35995|a21 | Ra~o | True | 1.5e-26 | True | 0.107 | honest |
| d:h-1e-8-3-37359|a18 | Ra~o | True | 3.1e-28 | True | 2.88e-05 | honest |
| d:h-1e-8-3-38240|a5 | Ra~o | True | 7.4e-30 | True | 0.000539 | honest |
