# levin-phase-floor 07d Part A': the understatements re-scored on the second reference
# written 2026-11-23T05:28:10Z on mps026298.phys.susx.ac.uk
# script rescore_ref2_a2.py sha256 47dd660b7f4d590db32c723474a3108c5e12536004f5074d097cc51c4bc85bc1
# tree: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-c2c57a3 HEAD c2c57a36cbad3780986baceb807dcb447975b498 (status --short: empty)
# main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2/out/repro/second_ref_a2.json sha256 5708114c61ed9e254e6b53d348b27cf9b1b4c1826ff0d43cdda7ee4f169fd9cd
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2/out/tables/t2_de_v4.json sha256 35701cabbf335a007b19362ddeb63a3d2bd6c5390cc91ef52304482d14cc604c
# input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement-a2/out/tables/t4_dt.json sha256 7e2f1a8f50bdd26274e85ebf320dc48e9371c5fd4436a41cd95d452a9d51e229

```
{
 "table 2": {
  "n": 74,
  "verdict_ref2": {
   "understates": 74
  },
  "refs_disagree": 3,
  "cause_reference": 0
 },
 "table 4": {
  "n": 13,
  "verdict_ref2": {
   "undecided": 10,
   "understates": 1,
   "bounded": 2
  },
  "refs_disagree": 12,
  "cause_reference": 12
 },
 "table 2 by v4 cause": {
  "common": {
   "understates": 19
  },
  "knots": {
   "understates": 55
  }
 }
}
```

| table | leaf | what | D | te v4 | eps v4 | te ref2 | eps2 | refs agree | verdict on ref2 |
|---|---|---|---|---|---|---|---|---|---|
| 2 | gate_q1|group|1e-7#53 | tree CC_47 max(pair, floor) | 3.77e-37 | 7.96e-37 | 7.33e-44 | 7.96e-37 | 4.23e-45 | False | understates |
| 2 | gate_q1|group|1e-7#93 | tree CC_47 max(pair, floor) | 1.32e-37 | 1.47e-36 | 3.24e-43 | 1.47e-36 | 3.14e-44 | False | understates |
| 2 | window|group|1e-7#43 | tree CC_47 max(pair, floor) | 9.82e-33 | 1.62e-32 | 1.6e-40 | 1.62e-32 | 1.87e-41 | False | understates |
| 4 | band_20_32|group|1e-7#63 | Levin Q, T = 4 pi, fallback cc47 | 9.66e-44 | 7.18e-42 | 5.6e-42 | 4.84e-45 | 4.67e-43 | False | undecided |
| 4 | band_20_32|group|1e-7#63 | Levin Q, T = 4 pi, fallback cc93 | 9.66e-44 | 7.18e-42 | 5.6e-42 | 4.84e-45 | 4.67e-43 | False | undecided |
| 4 | band_20_32|group|1e-7#63 | Levin Q, T = 6 pi, fallback cc47 | 9.66e-44 | 7.18e-42 | 5.6e-42 | 4.84e-45 | 4.67e-43 | False | undecided |
| 4 | band_20_32|group|1e-7#63 | Levin Q, T = 6 pi, fallback cc93 | 9.66e-44 | 7.18e-42 | 5.6e-42 | 4.84e-45 | 4.67e-43 | False | undecided |
| 4 | band_20_32v|group|1e-7#79 | Levin Q, T = 8 pi, fallback cc47 | 2.39e-40 | 2.62e-40 | 1.55e-45 | 2.62e-40 | 1.26e-46 | True | understates |
| 4 | band_6_8|group|1e-8#40 | Levin Q, T = 4 pi, fallback cc47 | 2.99e-42 | 2.58e-41 | 2.18e-41 | 9.52e-43 | 2.9e-42 | False | undecided |
| 4 | band_6_8|group|1e-8#40 | Levin Q, T = 6 pi, fallback cc47 | 2.99e-42 | 2.58e-41 | 2.18e-41 | 9.52e-43 | 2.9e-42 | False | undecided |
| 4 | band_20_32|analytic|1e-8#21 | Levin Q, T = 16 pi, fallback cc93 | 3.44e-27 | 9.65e-27 | 5.63e-27 | 2.6e-27 | 8.05e-28 | False | bounded |
| 4 | band_20_32|analytic|1e-8#21 | Levin Q, T = 20 pi, fallback cc93 | 3.44e-27 | 9.65e-27 | 5.63e-27 | 2.6e-27 | 8.05e-28 | False | bounded |
| 4 | band_20_32|group|1e-8#52 | Levin Q, T = 4 pi, fallback cc47 | 7.47e-47 | 5.75e-44 | 4.84e-44 | 7.59e-46 | 8.04e-45 | False | undecided |
| 4 | band_20_32|group|1e-8#52 | Levin Q, T = 4 pi, fallback cc93 | 7.47e-47 | 5.75e-44 | 4.84e-44 | 7.59e-46 | 8.04e-45 | False | undecided |
| 4 | band_20_32|group|1e-8#52 | Levin Q, T = 6 pi, fallback cc47 | 7.47e-47 | 5.75e-44 | 4.84e-44 | 7.59e-46 | 8.04e-45 | False | undecided |
| 4 | band_20_32|group|1e-8#52 | Levin Q, T = 6 pi, fallback cc93 | 7.47e-47 | 5.75e-44 | 4.84e-44 | 7.59e-46 | 8.04e-45 | False | undecided |
