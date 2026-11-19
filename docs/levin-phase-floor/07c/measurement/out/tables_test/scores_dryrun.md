## Table 3: honesty on every referenced call (07's rule; scored if the reference's spread <= min(abserr)/100)

| variant | referenced | not scored | converged & honest | converged & dishonest | unconverged (ratio) |
|---|---|---|---|---|---|
| tree | 184 | 6 | 173 | 0 | 5 (a:R2-28 1.11, a:R2-28~o 1.11, a:v04 0.000156, c-1e-7-0-13757-5 1.11, c-1e-8-0-13757-5 1.11) |
| S | 184 | 6 | 173 | 4 (a:R2-28, a:R2-28~o, c-1e-7-0-13757-5, c-1e-8-0-13757-5) | 1 (a:v04 0.000156) |
| S2 | 184 | 6 | 173 | 4 (a:R2-28, a:R2-28~o, c-1e-7-0-13757-5, c-1e-8-0-13757-5) | 1 (a:v04 0.000156) |
| N | 184 | 6 | 174 | 0 | 4 (a:R2-28 1.11, a:R2-28~o 1.11, c-1e-7-0-13757-5 1.11, c-1e-8-0-13757-5 1.11) |
| Eg | 184 | 6 | 168 | 0 | 10 (a:R1-x029-7 0.004, a:R1-x029-8 0.0146, a:R1-x098-7 0.000498, a:R1-x098-8 0.000502, a:R2-28 1.1, a:R2-28~o 1.1, a:v04 0.000148, a:v05 0.00114, c-1e-7-0-13757-5 1.1, c-1e-8-0-13757-5 1.1) |
| Ea | 184 | 6 | 168 | 0 | 10 (a:R1-x029-7 0.004, a:R1-x029-8 0.0146, a:R1-x098-7 0.000498, a:R1-x098-8 0.000502, a:R2-28 1.1, a:R2-28~o 1.1, a:v04 0.000148, a:v05 0.00114, c-1e-7-0-13757-5 1.1, c-1e-8-0-13757-5 1.1) |
| SEg | 184 | 6 | 168 | 4 (a:R2-28, a:R2-28~o, c-1e-7-0-13757-5, c-1e-8-0-13757-5) | 6 (a:R1-x029-7 0.004, a:R1-x029-8 0.0146, a:R1-x098-7 0.000498, a:R1-x098-8 0.000502, a:v04 0.000148, a:v05 0.00114) |
| SEa | 184 | 6 | 168 | 4 (a:R2-28, a:R2-28~o, c-1e-7-0-13757-5, c-1e-8-0-13757-5) | 6 (a:R1-x029-7 0.004, a:R1-x029-8 0.0146, a:R1-x098-7 0.000499, a:R1-x098-8 0.000503, a:v04 0.000148, a:v05 0.00114) |

## Table 4: the chase (198 calls and item 2's three)

| variant | calls | unconverged | of which every depth-20 leaf below a refused leaf | regions | wall s |
|---|---|---|---|---|---|
| tree | 201 | 201 | 195 | 27939 | 604 |
| S | 201 | 5 | 0 | 17488 | 357 |
| S2 | 201 | 6 | 0 | 19295 | 399 |
| N | 201 | 21 | 19 | 23625 | 505 |
| Eg | 201 | 186 | 182 | 40518 | 970 |
| Ea | 201 | 186 | 182 | 40546 | 967 |
| SEg | 201 | 4 | 0 | 24737 | 592 |
| SEa | 201 | 4 | 0 | 24765 | 592 |

## Table 5: cost and reach on every call of section 2.1

| variant | calls | regions | regions / tree | wall s | wall / tree | calls touched |
|---|---|---|---|---|---|---|
| tree | 454 | 50407 | 1.000 | 1055 | 1.000 | 0 |
| S | 454 | 37539 | 0.745 | 766 | 0.726 | 326 |
| S2 | 454 | 40142 | 0.796 | 815 | 0.773 | 309 |
| N | 454 | 44546 | 0.884 | 927 | 0.879 | 429 |
| Eg | 454 | 66161 | 1.313 | 1513 | 1.435 | 385 |
| Ea | 454 | 66123 | 1.312 | 1511 | 1.432 | 396 |
| SEg | 454 | 46670 | 0.926 | 1058 | 1.003 | 395 |
| SEa | 454 | 46632 | 0.925 | 1093 | 1.036 | 401 |

## Qualification

| variant | new converged & dishonest | fewer chase calls unconverged | qualifies |
|---|---|---|---|
| S | a:R2-28, a:R2-28~o, c-1e-7-0-13757-5, c-1e-8-0-13757-5 | yes | no |
| S2 | a:R2-28, a:R2-28~o, c-1e-7-0-13757-5, c-1e-8-0-13757-5 | yes | no |
| N | - | yes | yes |
| Eg | - | yes | yes |
| Ea | - | yes | yes |
| SEg | a:R2-28, a:R2-28~o, c-1e-7-0-13757-5, c-1e-8-0-13757-5 | yes | no |
| SEa | a:R2-28, a:R2-28~o, c-1e-7-0-13757-5, c-1e-8-0-13757-5 | yes | no |

## R5's verdicts at 07a's and 07b's code (beside the table)

| call | 07a | ratio | 07b | ratio |
|---|---|---|---|---|

script sha256 579ed08ed486a815cbf4287ae8c949fafcfcbeaca785498621058a27eff46808
