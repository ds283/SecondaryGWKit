# Addendum A.4: the near-edge checks on ref48 and the smooth-phase reference

```
{
 "final": {
  "visited|levin|bounded": 23,
  "returned|levin|bounded": 334,
  "returned|levin|undecided": 76,
  "returned|cc|bounded": 24,
  "visited|levin|undecided": 1,
  "visited|levin|understates": 1,
  "returned|levin|understates": 2
 },
 "at_noise_floor": 77,
 "ref48": {
  "visited|levin|bounded": 21,
  "returned|levin|bounded": 319,
  "returned|levin|undecided": 93,
  "returned|cc|bounded": 24,
  "visited|levin|undecided": 3,
  "visited|levin|understates": 1
 },
 "v4_ref48_disagree_checks": 34,
 "smooth_taken_leaves": 27
}
```

## Counts by set, T, route and what (final verdict; and ref48's)

| key | counts |
|---|---|
| returned|T10|cc|CC_93, pair93 | {"bounded": 3, "ref48: bounded": 3} |
| returned|T10|cc|CC_93, tail93 | {"bounded": 3, "ref48: bounded": 3} |
| returned|T10|levin|Q, fallback cc47 | {"bounded": 44, "ref48: bounded": 41, "undecided (noise floor)": 12, "ref48: undecided": 15} |
| returned|T10|levin|Q, fallback cc93 | {"bounded": 42, "ref48: bounded": 41, "undecided (noise floor)": 3, "ref48: undecided": 4} |
| returned|T12|cc|CC_93, pair93 | {"bounded": 3, "ref48: bounded": 3} |
| returned|T12|cc|CC_93, tail93 | {"bounded": 3, "ref48: bounded": 3} |
| returned|T12|levin|Q, fallback cc47 | {"undecided (noise floor)": 2, "ref48: undecided": 2, "bounded": 23, "ref48: bounded": 23} |
| returned|T12|levin|Q, fallback cc93 | {"bounded": 21, "ref48: bounded": 21, "undecided (noise floor)": 1, "ref48: undecided": 1} |
| returned|T6|cc|CC_93, pair93 | {"bounded": 3, "ref48: bounded": 3} |
| returned|T6|cc|CC_93, tail93 | {"bounded": 3, "ref48: bounded": 3} |
| returned|T6|levin|Q, fallback cc47 | {"bounded": 64, "ref48: bounded": 61, "undecided (noise floor)": 29, "ref48: undecided": 33, "understates": 1} |
| returned|T6|levin|Q, fallback cc93 | {"bounded": 53, "ref48: bounded": 49, "ref48: undecided": 18, "undecided (noise floor)": 14} |
| returned|T8|cc|CC_93, pair93 | {"bounded": 3, "ref48: bounded": 3} |
| returned|T8|cc|CC_93, tail93 | {"bounded": 3, "ref48: bounded": 3} |
| returned|T8|levin|Q, fallback cc47 | {"bounded": 46, "ref48: bounded": 43, "undecided (noise floor)": 13, "ref48: undecided": 17, "understates": 1} |
| returned|T8|levin|Q, fallback cc93 | {"bounded": 41, "ref48: bounded": 40, "undecided (noise floor)": 2, "ref48: undecided": 3} |
| visited|T10|levin|Q, fallback cc47 | {"bounded": 2, "ref48: bounded": 2} |
| visited|T10|levin|Q, fallback cc93 | {"bounded": 2, "ref48: bounded": 2} |
| visited|T12|levin|Q, fallback cc47 | {"bounded": 11, "ref48: bounded": 9, "ref48: undecided": 3, "undecided (noise floor)": 1} |
| visited|T12|levin|Q, fallback cc93 | {"bounded": 3, "ref48: bounded": 3, "understates": 1, "ref48: understates": 1} |
| visited|T6|levin|Q, fallback cc93 | {"bounded": 1, "ref48: bounded": 1} |
| visited|T8|levin|Q, fallback cc47 | {"bounded": 2, "ref48: bounded": 2} |
| visited|T8|levin|Q, fallback cc93 | {"bounded": 2, "ref48: bounded": 2} |

## Every check

| leaf | set | T/pi | what | D | te v4 / eps v4 | te48 / eps48 (dominant) | v48 | te smooth / eps_s | final | |D - te| | 2 s_47 | noise floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gate_q2|group|1e-7#19 | returned | 6 | CC_93, pair93 | 3.12e-42 | 2.35e-42 / 1.2e-42 | 1.65e-42 / 5.54e-44 (noise) | bounded | — | bounded | 1.46e-42 | 1.53e-42 |  |
| gate_q2|group|1e-7#19 | returned | 6 | CC_93, tail93 | 3.12e-42 | 2.35e-42 / 1.2e-42 | 1.65e-42 / 5.54e-44 (noise) | bounded | — | bounded | 1.46e-42 | 1.53e-42 |  |
| gate_q2|group|1e-8#122 | returned | 6 | CC_93, pair93 | 2.62e-42 | 1.91e-42 / 1.07e-42 | 1.31e-42 / 5.1e-44 (noise) | bounded | — | bounded | 1.31e-42 | 1.44e-42 |  |
| gate_q2|group|1e-8#122 | returned | 6 | CC_93, tail93 | 2.62e-42 | 1.91e-42 / 1.07e-42 | 1.31e-42 / 5.1e-44 (noise) | bounded | — | bounded | 1.31e-42 | 1.44e-42 |  |
| gate_q2|group|1e-8#76 | returned | 6 | CC_93, pair93 | 2.36e-42 | 1.66e-42 / 1.01e-42 | 1.06e-42 / 5.08e-44 (noise) | bounded | — | bounded | 1.3e-42 | 1.41e-42 |  |
| gate_q2|group|1e-8#76 | returned | 6 | CC_93, tail93 | 2.38e-42 | 1.66e-42 / 1.01e-42 | 1.06e-42 / 5.08e-44 (noise) | bounded | — | bounded | 1.32e-42 | 1.41e-42 |  |
| band_10_12|group|1e-7#12 | returned | 6 | Q, fallback cc47 | 1.14e-43 | 1.65e-43 / 1.79e-43 | 1.1e-43 / 1.51e-44 (noise) | undecided | 1.16e-43 / 1.04e-44 | undecided | 1.95e-45 | 6.31e-43 | yes |
| band_10_12|group|1e-7#14 | returned | 6 | Q, fallback cc47 | 2.79e-44 | 1.07e-44 / 4.91e-44 | 2.27e-44 / 5.82e-45 (noise) | undecided | 2.47e-44 / 6.37e-45 | undecided | 3.19e-45 | 2.09e-43 | yes |
| band_10_12|group|1e-7#18 | returned | 6 | Q, fallback cc47 | 3.92e-35 | 3.7e-35 / 1e-35 | 3.87e-35 / 1.26e-36 (noise) | undecided | 3.86e-35 / 1.02e-36 | undecided | 6.05e-37 | 5.39e-35 | yes |
| band_10_12|group|1e-7#18 | returned | 6 | Q, fallback cc93 | 3.97e-35 | 3.38e-35 / 1e-35 | 3.54e-35 / 1.26e-36 (noise) | bounded | 3.53e-35 / 1.02e-36 | bounded | 4.33e-36 | 5.39e-35 |  |
| band_10_12|group|1e-7#22 | returned | 6 | Q, fallback cc47 | 3.18e-42 | 3e-42 / 6.14e-41 | 2.59e-42 / 5.09e-42 (noise) | undecided | 3.17e-42 / 6.12e-42 | undecided | 1.34e-44 | 2.03e-40 | yes |
| band_10_12|group|1e-7#24 | returned | 6 | Q, fallback cc47 | 1.12e-45 | 1.26e-45 / 7.29e-46 | 1.14e-45 / 2.44e-46 (noise) | undecided | — | undecided | 2.19e-47 | 6.77e-45 | yes |
| band_10_12|group|1e-7#24 | returned | 6 | Q, fallback cc93 | 2.41e-45 | 1.94e-45 / 7.29e-46 | 1.82e-45 / 2.44e-46 (noise) | bounded | — | bounded | 5.84e-46 | 6.77e-45 |  |
| band_10_12|group|1e-7#33 | returned | 6 | Q, fallback cc47 | 5.93e-46 | 7.05e-46 / 2.46e-45 | 5.26e-46 / 1.89e-46 (noise) | undecided | 4.75e-46 / 1.15e-46 | bounded | 1.18e-46 | 7.71e-45 |  |
| band_10_12|group|1e-7#35 | returned | 6 | Q, fallback cc47 | 1.2e-40 | 7.91e-41 / 7.64e-41 | 8.41e-41 / 5.9e-42 (noise) | bounded | 8.45e-41 / 2.08e-42 | bounded | 3.6e-41 | 2.07e-40 |  |
| band_10_12|group|1e-7#35 | returned | 6 | Q, fallback cc93 | 7.51e-41 | 6.76e-41 / 7.64e-41 | 7.25e-41 / 5.9e-42 (noise) | undecided | 7.3e-41 / 2.08e-42 | bounded | 2.14e-42 | 2.07e-40 |  |
| band_10_12|group|1e-7#45 | returned | 6 | Q, fallback cc47 | 2.29e-40 | 1.69e-40 / 1.03e-40 | 1.75e-40 / 1.84e-41 (noise) | bounded | — | bounded | 5.38e-41 | 6.01e-40 |  |
| band_10_12|group|1e-7#47 | returned | 6 | Q, fallback cc47 | 3.45e-43 | 1.64e-43 / 9.83e-43 | 1.03e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.42e-43 | 2.17e-42 |  |
| band_10_12|group|1e-7#47 | returned | 6 | Q, fallback cc93 | 3.53e-43 | 1.78e-43 / 9.83e-43 | 1.17e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.36e-43 | 2.17e-42 |  |
| band_10_12|group|1e-7#52 | returned | 6 | Q, fallback cc47 | 8.96e-43 | 9.07e-43 / 1.07e-41 | 5.01e-43 / 7.1e-43 (noise) | undecided | 3.39e-43 / 3.07e-43 | bounded | 5.57e-43 | 2.19e-41 |  |
| band_10_12|group|1e-7#7 | returned | 6 | Q, fallback cc93 | 6.73e-41 | 7.34e-41 / 2.41e-41 | 6.77e-41 / 7.62e-42 (noise) | undecided | — | undecided | 3.92e-43 | 2.08e-40 | yes |
| band_10_12|group|1e-8#14 | returned | 6 | Q, fallback cc47 | 2.29e-40 | 1.69e-40 / 1.03e-40 | 1.75e-40 / 1.84e-41 (noise) | bounded | — | bounded | 5.38e-41 | 6.01e-40 |  |
| band_10_12|group|1e-8#20 | returned | 6 | Q, fallback cc47 | 1.77e-40 | 1.67e-40 / 1.15e-40 | 1.77e-40 / 1.7e-41 (noise) | undecided | — | undecided | 2.99e-43 | 5.96e-40 | yes |
| band_10_12|group|1e-8#20 | returned | 6 | Q, fallback cc93 | 3.13e-40 | 2.26e-40 / 1.15e-40 | 2.36e-40 / 1.7e-41 (noise) | bounded | — | bounded | 7.72e-41 | 5.96e-40 |  |
| band_10_12|group|1e-8#21 | returned | 6 | Q, fallback cc47 | 5.1e-42 | 1.81e-41 / 1.17e-40 | 1.38e-42 / 5.32e-42 (noise) | undecided | 1.26e-42 / 5.19e-42 | undecided | 3.84e-42 | 2.01e-40 | yes |
| band_10_12|group|1e-8#24 | returned | 6 | Q, fallback cc93 | 6.73e-41 | 7.34e-41 / 2.41e-41 | 6.77e-41 / 7.62e-42 (noise) | undecided | — | undecided | 3.92e-43 | 2.08e-40 | yes |
| band_10_12|group|1e-8#27 | returned | 6 | Q, fallback cc47 | 1.12e-45 | 1.26e-45 / 7.29e-46 | 1.14e-45 / 2.44e-46 (noise) | undecided | — | undecided | 2.19e-47 | 6.77e-45 | yes |
| band_10_12|group|1e-8#27 | returned | 6 | Q, fallback cc93 | 2.41e-45 | 1.94e-45 / 7.29e-46 | 1.82e-45 / 2.44e-46 (noise) | bounded | — | bounded | 5.84e-46 | 6.77e-45 |  |
| band_10_12|group|1e-8#29 | returned | 6 | Q, fallback cc47 | 3.45e-43 | 1.64e-43 / 9.83e-43 | 1.03e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.42e-43 | 2.17e-42 |  |
| band_10_12|group|1e-8#29 | returned | 6 | Q, fallback cc93 | 3.53e-43 | 1.78e-43 / 9.83e-43 | 1.17e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.36e-43 | 2.17e-42 |  |
| band_10_12|group|1e-8#3 | returned | 6 | Q, fallback cc47 | 6.41e-42 | 1.84e-42 / 9.46e-42 | 1.83e-43 / 4.23e-43 (noise) | bounded | — | bounded | 6.22e-42 | 2e-41 |  |
| band_10_12|group|1e-8#3 | returned | 6 | Q, fallback cc93 | 5.2e-42 | 1.83e-42 / 9.46e-42 | 1.94e-43 / 4.23e-43 (noise) | bounded | — | bounded | 5e-42 | 2e-41 |  |
| band_10_12|group|1e-8#35 | returned | 6 | Q, fallback cc47 | 1.14e-43 | 1.65e-43 / 1.79e-43 | 1.1e-43 / 1.51e-44 (noise) | undecided | 1.16e-43 / 1.04e-44 | undecided | 1.95e-45 | 6.31e-43 | yes |
| band_10_12|group|1e-8#48 | returned | 6 | Q, fallback cc47 | 1.43e-37 | 1.11e-37 / 7.41e-38 | 1.1e-37 / 4.23e-39 (noise) | bounded | — | bounded | 3.36e-38 | 1.84e-37 |  |
| band_10_12|group|1e-8#48 | returned | 6 | Q, fallback cc93 | 1.64e-37 | 9.63e-38 / 7.41e-38 | 9.49e-38 / 4.23e-39 (noise) | bounded | — | bounded | 6.91e-38 | 1.84e-37 |  |
| band_10_12|group|1e-8#49 | returned | 6 | Q, fallback cc47 | 7.86e-42 | 7.67e-42 / 7.86e-42 | 5.58e-42 / 7.18e-43 (noise) | bounded | — | bounded | 2.27e-42 | 2.32e-41 |  |
| band_10_12|group|1e-8#59 | returned | 6 | Q, fallback cc47 | 1.93e-42 | 1.69e-42 / 8.02e-43 | 1.94e-42 / 2.03e-43 (noise) | undecided | — | undecided | 5.54e-45 | 5.99e-42 | yes |
| band_10_12|group|1e-8#7 | returned | 6 | Q, fallback cc47 | 3.41e-39 | 2.78e-39 / 8.65e-40 | 2.77e-39 / 2.65e-40 (noise) | bounded | — | bounded | 6.46e-40 | 6.33e-39 |  |
| band_12_16|analytic|1e-7#27 | returned | 6 | Q, fallback cc47 | 1.06e-26 | 6.13e-27 / 6.33e-27 | 9.09e-28 / 4.91e-28 (noise) | bounded | — | bounded | 9.73e-27 | 1.16e-26 |  |
| band_12_16|analytic|1e-7#45 | returned | 6 | Q, fallback cc47 | 9.3e-25 | 6.69e-25 / 4.09e-25 | 8.4e-26 / 4.6e-26 (noise) | bounded | — | bounded | 8.46e-25 | 1.09e-24 |  |
| band_12_16|analytic|1e-8#30 | returned | 6 | Q, fallback cc47 | 9.3e-25 | 6.69e-25 / 4.09e-25 | 8.4e-26 / 4.6e-26 (noise) | bounded | — | bounded | 8.46e-25 | 1.09e-24 |  |
| band_12_16|group|1e-7#34 | returned | 6 | Q, fallback cc47 | 2.08e-42 | 3.35e-42 / 2.36e-42 | 3.11e-43 / 1.52e-43 (noise) | bounded | — | bounded | 1.77e-42 | 6.77e-42 |  |
| band_12_16|group|1e-7#34 | returned | 6 | Q, fallback cc93 | 2.07e-42 | 3.34e-42 / 2.36e-42 | 3.07e-43 / 1.52e-43 (noise) | bounded | — | bounded | 1.77e-42 | 6.77e-42 |  |
| band_12_16|group|1e-7#58 | returned | 6 | Q, fallback cc93 | 1.47e-40 | 2.06e-41 / 1.37e-40 | 6.43e-42 / 6.18e-42 (noise) | bounded | — | bounded | 1.4e-40 | 2.91e-40 |  |
| band_12_16|group|1e-7#69 | returned | 6 | Q, fallback cc47 | 1.03e-42 | 1.34e-42 / 4.24e-42 | 1.86e-43 / 5.76e-43 (noise) | bounded | — | bounded | 8.48e-43 | 2.49e-41 |  |
| band_12_16|group|1e-7#7 | returned | 6 | Q, fallback cc47 | 2.14e-36 | 1.12e-36 / 2.58e-36 | 4.15e-37 / 1.69e-37 (noise) | bounded | — | bounded | 1.73e-36 | 7.38e-36 |  |
| band_12_16|group|1e-8#75 | returned | 6 | Q, fallback cc47 | 1.29e-42 | 6.27e-43 / 2.63e-42 | 5.44e-44 / 2.04e-43 (noise) | bounded | — | bounded | 1.24e-42 | 7.13e-42 |  |
| band_12_16|group|1e-8#75 | returned | 6 | Q, fallback cc93 | 1.37e-42 | 5.66e-43 / 2.63e-42 | 6.66e-45 / 2.04e-43 (noise) | bounded | — | bounded | 1.36e-42 | 7.13e-42 |  |
| band_16_20|analytic|1e-7#43 | returned | 6 | Q, fallback cc47 | 1.82e-27 | 2.36e-27 / 1.5e-27 | 5.17e-28 / 6.06e-29 (noise) | bounded | — | bounded | 1.3e-27 | 2.81e-27 |  |
| band_16_20|analytic|1e-7#8 | returned | 6 | Q, fallback cc93 | 1.14e-23 | 5.23e-24 / 1.04e-23 | 4.39e-25 / 7.54e-25 (noise) | bounded | — | bounded | 1.1e-23 | 1.46e-23 |  |
| band_16_20|analytic|1e-8#26 | returned | 6 | Q, fallback cc93 | 1.14e-23 | 5.23e-24 / 1.04e-23 | 4.39e-25 / 7.54e-25 (noise) | bounded | — | bounded | 1.1e-23 | 1.46e-23 |  |
| band_16_20|analytic|1e-8#27 | returned | 6 | Q, fallback cc47 | 9.94e-27 | 7.5e-27 / 9.52e-27 | 3.26e-27 / 8.66e-28 (noise) | bounded | — | bounded | 6.68e-27 | 1.56e-26 |  |
| band_16_20|analytic|1e-8#27 | returned | 6 | Q, fallback cc93 | 9.94e-27 | 5.28e-27 / 9.52e-27 | 1.03e-27 / 8.66e-28 (noise) | bounded | — | bounded | 8.9e-27 | 1.56e-26 |  |
| band_16_20|group|1e-8#15 | returned | 6 | Q, fallback cc47 | 3.94e-43 | 1.1e-43 / 7.68e-43 | 8.69e-44 / 9.23e-44 (noise) | bounded | — | bounded | 3.07e-43 | 3.74e-42 |  |
| band_16_20|group|1e-8#15 | returned | 6 | Q, fallback cc93 | 3.41e-43 | 6.98e-44 / 7.68e-43 | 4.69e-44 / 9.23e-44 (noise) | bounded | — | bounded | 2.94e-43 | 3.74e-42 |  |
| band_16_20|group|1e-8#59 | returned | 6 | Q, fallback cc47 | 1.18e-40 | 4.16e-42 / 1.29e-40 | 6.32e-42 / 7.21e-42 (noise) | bounded | — | bounded | 1.12e-40 | 3.08e-40 |  |
| band_16_20|group|1e-8#59 | returned | 6 | Q, fallback cc93 | 1.21e-40 | 6.15e-42 / 1.29e-40 | 8.31e-42 / 7.21e-42 (noise) | bounded | — | bounded | 1.13e-40 | 3.08e-40 |  |
| band_20_32|analytic|1e-7#0 | returned | 6 | Q, fallback cc93 | 3.2e-24 | 1.71e-24 / 3.57e-24 | 2.26e-25 / 2.86e-25 (noise) | bounded | — | bounded | 2.98e-24 | 7.49e-24 |  |
| band_20_32|analytic|1e-7#19 | returned | 6 | Q, fallback cc93 | 3.48e-24 | 8.35e-25 / 3.86e-24 | 3.23e-27 / 2.67e-25 (noise) | bounded | — | bounded | 3.47e-24 | 6.34e-24 |  |
| band_20_32|analytic|1e-7#23 | returned | 6 | Q, fallback cc93 | 1.21e-27 | 3.11e-27 / 2.65e-27 | 3.16e-30 / 1.85e-28 (noise) | bounded | — | bounded | 1.21e-27 | 4.31e-27 |  |
| band_20_32|analytic|1e-7#59 | returned | 6 | Q, fallback cc93 | 3.48e-24 | 8.35e-25 / 3.86e-24 | 3.23e-27 / 2.67e-25 (noise) | bounded | — | bounded | 3.47e-24 | 6.34e-24 |  |
| band_20_32|analytic|1e-7#71 | returned | 6 | Q, fallback cc93 | 1.73e-22 | 4.92e-23 / 1.96e-22 | 1.24e-24 / 1.18e-23 (noise) | bounded | — | bounded | 1.72e-22 | 2.38e-22 |  |
| band_20_32|analytic|1e-8#36 | returned | 6 | Q, fallback cc93 | 1.21e-27 | 3.11e-27 / 2.65e-27 | 3.16e-30 / 1.85e-28 (noise) | bounded | — | bounded | 1.21e-27 | 4.31e-27 |  |
| band_20_32|analytic|1e-8#6 | returned | 6 | Q, fallback cc93 | 3.48e-24 | 8.35e-25 / 3.86e-24 | 3.23e-27 / 2.67e-25 (noise) | bounded | — | bounded | 3.47e-24 | 6.34e-24 |  |
| band_20_32|analytic|1e-8#60 | returned | 6 | Q, fallback cc93 | 3.2e-24 | 1.71e-24 / 3.57e-24 | 2.26e-25 / 2.86e-25 (noise) | bounded | — | bounded | 2.98e-24 | 7.49e-24 |  |
| band_20_32|analytic|1e-8#63 | returned | 6 | Q, fallback cc93 | 3.48e-24 | 8.35e-25 / 3.86e-24 | 3.23e-27 / 2.67e-25 (noise) | bounded | — | bounded | 3.47e-24 | 6.34e-24 |  |
| band_20_32|group|1e-7#25 | returned | 6 | Q, fallback cc47 | 4.09e-45 | 7.73e-46 / 4.4e-45 | 6.2e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.47e-45 | 1.42e-44 |  |
| band_20_32|group|1e-7#25 | returned | 6 | Q, fallback cc93 | 3.72e-45 | 4.92e-46 / 4.4e-45 | 3.39e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.38e-45 | 1.42e-44 |  |
| band_20_32|group|1e-7#27 | returned | 6 | Q, fallback cc47 | 1.96e-39 | 7.2e-40 / 1.38e-39 | 1.53e-41 / 7.31e-41 (noise) | bounded | — | bounded | 1.94e-39 | 2.84e-39 |  |
| band_20_32|group|1e-7#27 | returned | 6 | Q, fallback cc93 | 1.96e-39 | 7.2e-40 / 1.38e-39 | 1.53e-41 / 7.31e-41 (noise) | bounded | — | bounded | 1.94e-39 | 2.84e-39 |  |
| band_20_32|group|1e-7#40 | returned | 6 | Q, fallback cc47 | 2.71e-42 | 4.24e-41 / 5e-41 | 1.8e-43 / 3.81e-42 (noise) | undecided | 1.56e-42 / 3.55e-42 | undecided | 1.15e-42 | 1.51e-40 | yes |
| band_20_32|group|1e-7#40 | returned | 6 | Q, fallback cc93 | 2.71e-42 | 4.24e-41 / 5e-41 | 1.8e-43 / 3.81e-42 (noise) | undecided | 1.56e-42 / 3.55e-42 | undecided | 1.15e-42 | 1.51e-40 | yes |
| band_20_32|group|1e-7#52 | returned | 6 | Q, fallback cc47 | 4.84e-46 | 7.31e-45 / 4.66e-44 | 6.72e-46 / 4.82e-45 (noise) | undecided | 5.75e-46 / 9.34e-46 | undecided | 9.04e-47 | 1.75e-43 | yes |
| band_20_32|group|1e-7#52 | returned | 6 | Q, fallback cc93 | 4.84e-46 | 7.31e-45 / 4.66e-44 | 6.72e-46 / 4.82e-45 (noise) | undecided | 5.75e-46 / 9.34e-46 | undecided | 9.04e-47 | 1.75e-43 | yes |
| band_20_32|group|1e-7#7 | returned | 6 | Q, fallback cc47 | 2.21e-39 | 7.75e-39 / 8.64e-39 | 3.14e-41 / 3.03e-40 (noise) | bounded | — | bounded | 2.18e-39 | 1.6e-38 |  |
| band_20_32|group|1e-7#7 | returned | 6 | Q, fallback cc93 | 2.21e-39 | 7.75e-39 / 8.64e-39 | 3.14e-41 / 3.03e-40 (noise) | bounded | — | bounded | 2.18e-39 | 1.6e-38 |  |
| band_20_32|group|1e-7#78 | returned | 6 | Q, fallback cc47 | 5.57e-45 | 6.35e-44 / 5.57e-43 | 6.7e-45 / 5.77e-44 (noise) | undecided | — | undecided | 1.12e-45 | 1.89e-42 | yes |
| band_20_32|group|1e-7#78 | returned | 6 | Q, fallback cc93 | 5.57e-45 | 6.35e-44 / 5.57e-43 | 6.7e-45 / 5.77e-44 (noise) | undecided | — | undecided | 1.12e-45 | 1.89e-42 | yes |
| band_20_32|group|1e-7#79 | returned | 6 | Q, fallback cc47 | 5.22e-44 | 3.63e-42 / 3.47e-41 | 7.71e-44 / 1.62e-42 (noise) | undecided | — | undecided | 2.49e-44 | 5.08e-41 | yes |
| band_20_32|group|1e-7#79 | returned | 6 | Q, fallback cc93 | 5.22e-44 | 3.63e-42 / 3.47e-41 | 7.71e-44 / 1.62e-42 (noise) | undecided | — | undecided | 2.49e-44 | 5.08e-41 | yes |
| band_20_32|group|1e-7#90 | returned | 6 | Q, fallback cc47 | 6.35e-42 | 2.82e-42 / 4.79e-42 | 3.5e-43 / 2.7e-43 (noise) | bounded | — | bounded | 6e-42 | 1.41e-41 |  |
| band_20_32|group|1e-7#92 | returned | 6 | Q, fallback cc47 | 1.82e-42 | 8.17e-43 / 1.26e-42 | 2.94e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.53e-42 | 4.26e-42 |  |
| band_20_32|group|1e-7#92 | returned | 6 | Q, fallback cc93 | 1.6e-42 | 6.61e-43 / 1.26e-42 | 1.38e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.46e-42 | 4.26e-42 |  |
| band_20_32|group|1e-7#94 | returned | 6 | Q, fallback cc47 | 3.85e-48 | 1.87e-45 / 3.34e-45 | 1.8e-47 / 1.39e-46 (noise) | undecided | — | undecided | 1.42e-47 | 5.69e-45 | yes |
| band_20_32|group|1e-7#94 | returned | 6 | Q, fallback cc93 | 3.85e-48 | 1.87e-45 / 3.34e-45 | 1.8e-47 / 1.39e-46 (noise) | undecided | — | undecided | 1.42e-47 | 5.69e-45 | yes |
| band_20_32|group|1e-8#10 | returned | 6 | Q, fallback cc47 | 5.57e-45 | 6.35e-44 / 5.57e-43 | 6.7e-45 / 5.77e-44 (noise) | undecided | — | undecided | 1.12e-45 | 1.89e-42 | yes |
| band_20_32|group|1e-8#10 | returned | 6 | Q, fallback cc93 | 5.57e-45 | 6.35e-44 / 5.57e-43 | 6.7e-45 / 5.77e-44 (noise) | undecided | — | undecided | 1.12e-45 | 1.89e-42 | yes |
| band_20_32|group|1e-8#12 | returned | 6 | Q, fallback cc47 | 1.35e-39 | 5.55e-40 / 2.78e-39 | 4.47e-41 / 3.2e-40 (noise) | bounded | — | bounded | 1.31e-39 | 1.41e-38 |  |
| band_20_32|group|1e-8#12 | returned | 6 | Q, fallback cc93 | 1.35e-39 | 5.55e-40 / 2.78e-39 | 4.47e-41 / 3.2e-40 (noise) | bounded | — | bounded | 1.31e-39 | 1.41e-38 |  |
| band_20_32|group|1e-8#16 | returned | 6 | Q, fallback cc47 | 4.56e-44 | 4.51e-45 / 4.57e-44 | 5.32e-46 / 2.71e-45 (noise) | bounded | — | bounded | 4.5e-44 | 1.78e-43 |  |
| band_20_32|group|1e-8#16 | returned | 6 | Q, fallback cc93 | 4.56e-44 | 4.51e-45 / 4.57e-44 | 5.32e-46 / 2.71e-45 (noise) | bounded | — | bounded | 4.5e-44 | 1.78e-43 |  |
| band_20_32|group|1e-8#21 | returned | 6 | Q, fallback cc47 | 1.98e-45 | 1.2e-43 / 4.5e-43 | 2.04e-44 / 2.36e-43 (noise) | undecided | — | undecided | 1.84e-44 | 5.75e-42 | yes |
| band_20_32|group|1e-8#21 | returned | 6 | Q, fallback cc93 | 1.98e-45 | 1.2e-43 / 4.5e-43 | 2.04e-44 / 2.36e-43 (noise) | undecided | — | undecided | 1.84e-44 | 5.75e-42 | yes |
| band_20_32|group|1e-8#22 | returned | 6 | Q, fallback cc47 | 3.07e-39 | 9.29e-40 / 6.01e-39 | 1.46e-41 / 4.02e-40 (noise) | bounded | — | bounded | 3.06e-39 | 1.58e-38 |  |
| band_20_32|group|1e-8#22 | returned | 6 | Q, fallback cc93 | 3.07e-39 | 9.29e-40 / 6.01e-39 | 1.46e-41 / 4.02e-40 (noise) | bounded | — | bounded | 3.06e-39 | 1.58e-38 |  |
| band_20_32|group|1e-8#27 | returned | 6 | Q, fallback cc47 | 2.08e-41 | 2.26e-41 / 7.79e-41 | 1.14e-43 / 3.43e-42 (noise) | bounded | — | bounded | 2.07e-41 | 1.42e-40 |  |
| band_20_32|group|1e-8#27 | returned | 6 | Q, fallback cc93 | 2.08e-41 | 2.26e-41 / 7.79e-41 | 1.14e-43 / 3.43e-42 (noise) | bounded | — | bounded | 2.07e-41 | 1.42e-40 |  |
| band_20_32|group|1e-8#30 | returned | 6 | Q, fallback cc47 | 9.99e-40 | 1.23e-39 / 5.58e-39 | 2.41e-41 / 3.36e-40 (noise) | bounded | — | bounded | 9.75e-40 | 1.84e-38 |  |
| band_20_32|group|1e-8#30 | returned | 6 | Q, fallback cc93 | 9.99e-40 | 1.23e-39 / 5.58e-39 | 2.41e-41 / 3.36e-40 (noise) | bounded | — | bounded | 9.75e-40 | 1.84e-38 |  |
| band_20_32|group|1e-8#33 | returned | 6 | Q, fallback cc47 | 4.34e-42 | 2.63e-42 / 2.24e-41 | 3.22e-43 / 1.08e-42 (noise) | bounded | — | bounded | 4.01e-42 | 5.03e-41 |  |
| band_20_32|group|1e-8#33 | returned | 6 | Q, fallback cc93 | 4.34e-42 | 2.63e-42 / 2.24e-41 | 3.22e-43 / 1.08e-42 (noise) | bounded | — | bounded | 4.01e-42 | 5.03e-41 |  |
| band_20_32|group|1e-8#38 | returned | 6 | Q, fallback cc47 | 2.61e-45 | 1.62e-45 / 4.5e-45 | 5.98e-47 / 3.47e-46 (noise) | bounded | — | bounded | 2.55e-45 | 1.78e-44 |  |
| band_20_32|group|1e-8#38 | returned | 6 | Q, fallback cc93 | 2.61e-45 | 1.62e-45 / 4.5e-45 | 5.98e-47 / 3.47e-46 (noise) | bounded | — | bounded | 2.55e-45 | 1.78e-44 |  |
| band_20_32|group|1e-8#4 | returned | 6 | Q, fallback cc47 | 3.58e-44 | 5.36e-42 / 1.12e-41 | 5.62e-44 / 1.21e-42 (noise) | undecided | — | undecided | 2.04e-44 | 4.77e-41 | yes |
| band_20_32|group|1e-8#4 | returned | 6 | Q, fallback cc93 | 3.58e-44 | 5.36e-42 / 1.12e-41 | 5.62e-44 / 1.21e-42 (noise) | undecided | — | undecided | 2.04e-44 | 4.77e-41 | yes |
| band_20_32|group|1e-8#45 | returned | 6 | Q, fallback cc47 | 1.66e-41 | 9.11e-40 / 2.44e-39 | 1.87e-41 / 1.73e-40 (noise) | undecided | — | undecided | 2.12e-42 | 7.44e-39 | yes |
| band_20_32|group|1e-8#45 | returned | 6 | Q, fallback cc93 | 1.66e-41 | 9.11e-40 / 2.44e-39 | 1.87e-41 / 1.73e-40 (noise) | undecided | — | undecided | 2.12e-42 | 7.44e-39 | yes |
| band_20_32|group|1e-8#48 | returned | 6 | Q, fallback cc47 | 2.49e-42 | 4.09e-42 / 6.34e-42 | 1.78e-43 / 1.11e-42 (noise) | bounded | — | bounded | 2.32e-42 | 3.86e-41 |  |
| band_20_32|group|1e-8#48 | returned | 6 | Q, fallback cc93 | 2.33e-42 | 4.24e-42 / 6.34e-42 | 3.33e-43 / 1.11e-42 (noise) | bounded | — | bounded | 1.99e-42 | 3.86e-41 |  |
| band_20_32|group|1e-8#50 | returned | 6 | Q, fallback cc47 | 1.82e-42 | 8.17e-43 / 1.26e-42 | 2.94e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.53e-42 | 4.26e-42 |  |
| band_20_32|group|1e-8#50 | returned | 6 | Q, fallback cc93 | 1.6e-42 | 6.61e-43 / 1.26e-42 | 1.38e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.46e-42 | 4.26e-42 |  |
| band_20_32|group|1e-8#57 | returned | 6 | Q, fallback cc47 | 5.49e-38 | 2.07e-39 / 6.1e-38 | 4.42e-40 / 3.14e-39 (noise) | bounded | — | bounded | 5.44e-38 | 1.42e-37 |  |
| band_20_32|group|1e-8#57 | returned | 6 | Q, fallback cc93 | 5.49e-38 | 2.07e-39 / 6.1e-38 | 4.42e-40 / 3.14e-39 (noise) | bounded | — | bounded | 5.44e-38 | 1.42e-37 |  |
| band_20_32|group|1e-8#58 | returned | 6 | Q, fallback cc47 | 4.5e-44 | 6.03e-42 / 2.52e-41 | 9.74e-44 / 1.71e-42 (noise) | undecided | — | undecided | 5.24e-44 | 4.78e-41 | yes |
| band_20_32|group|1e-8#58 | returned | 6 | Q, fallback cc93 | 4.5e-44 | 6.03e-42 / 2.52e-41 | 9.74e-44 / 1.71e-42 (noise) | undecided | — | undecided | 5.24e-44 | 4.78e-41 | yes |
| band_20_32|group|1e-8#67 | returned | 6 | Q, fallback cc47 | 2.58e-45 | 8.45e-47 / 6.59e-45 | 4.7e-49 / 3.04e-46 (noise) | bounded | — | bounded | 2.58e-45 | 1.58e-44 |  |
| band_20_32|group|1e-8#67 | returned | 6 | Q, fallback cc93 | 2.58e-45 | 8.45e-47 / 6.59e-45 | 4.7e-49 / 3.04e-46 (noise) | bounded | — | bounded | 2.58e-45 | 1.58e-44 |  |
| band_20_32|group|1e-8#69 | returned | 6 | Q, fallback cc47 | 1.19e-41 | 1.7e-42 / 2.03e-41 | 7.53e-44 / 7.94e-43 (noise) | bounded | — | bounded | 1.19e-41 | 4.63e-41 |  |
| band_20_32|group|1e-8#69 | returned | 6 | Q, fallback cc93 | 1.19e-41 | 1.7e-42 / 2.03e-41 | 7.53e-44 / 7.94e-43 (noise) | bounded | — | bounded | 1.19e-41 | 4.63e-41 |  |
| band_20_32|group|1e-8#75 | returned | 6 | Q, fallback cc47 | 2.96e-46 | 1.49e-45 / 3.6e-45 | 2.16e-47 / 4.33e-46 (noise) | undecided | 3.44e-47 / 2.48e-46 | bounded | 2.62e-46 | 1.87e-44 |  |
| band_20_32|group|1e-8#75 | returned | 6 | Q, fallback cc93 | 2.96e-46 | 1.49e-45 / 3.6e-45 | 2.16e-47 / 4.33e-46 (noise) | undecided | 3.44e-47 / 2.48e-46 | bounded | 2.62e-46 | 1.87e-44 |  |
| band_20_32|group|1e-8#94 | returned | 6 | Q, fallback cc47 | 4.09e-45 | 7.73e-46 / 4.4e-45 | 6.2e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.47e-45 | 1.42e-44 |  |
| band_20_32|group|1e-8#94 | returned | 6 | Q, fallback cc93 | 3.72e-45 | 4.92e-46 / 4.4e-45 | 3.39e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.38e-45 | 1.42e-44 |  |
| band_20_32|group|1e-8#97 | returned | 6 | Q, fallback cc47 | 3.01e-44 | 8.84e-45 / 3.78e-44 | 9.86e-46 / 2.79e-45 (noise) | bounded | — | bounded | 2.91e-44 | 1.69e-43 |  |
| band_20_32|group|1e-8#97 | returned | 6 | Q, fallback cc93 | 3.01e-44 | 8.84e-45 / 3.78e-44 | 9.86e-46 / 2.79e-45 (noise) | bounded | — | bounded | 2.91e-44 | 1.69e-43 |  |
| band_6_8|group|1e-7#10 | returned | 6 | Q, fallback cc47 | 9.96e-40 | 5.33e-41 / 2.27e-39 | 3.45e-41 / 9.64e-41 (noise) | bounded | — | bounded | 9.61e-40 | 3.78e-39 |  |
| band_6_8|group|1e-7#10 | returned | 6 | Q, fallback cc93 | 9.55e-40 | 1.1e-41 / 2.27e-39 | 7.72e-42 / 9.64e-41 (noise) | bounded | — | bounded | 9.47e-40 | 3.78e-39 |  |
| band_6_8|group|1e-7#17 | returned | 6 | Q, fallback cc47 | 4.69e-41 | 4.3e-41 / 1.8e-41 | 3.62e-41 / 5.87e-42 (noise) | bounded | 3.75e-41 / 2.38e-42 | bounded | 9.43e-42 | 1.19e-40 |  |
| band_6_8|group|1e-7#17 | returned | 6 | Q, fallback cc93 | 3.65e-41 | 3.96e-41 / 1.8e-41 | 3.28e-41 / 5.87e-42 (noise) | undecided | 3.41e-41 / 2.38e-42 | bounded | 2.38e-42 | 1.19e-40 |  |
| band_6_8|group|1e-7#20 | returned | 6 | Q, fallback cc47 | 3.76e-41 | 3.06e-41 / 1.85e-41 | 3.68e-41 / 1.76e-42 (noise) | undecided | 3.72e-41 / 6.61e-43 | undecided | 4.22e-43 | 4.7e-41 | yes |
| band_6_8|group|1e-7#21 | returned | 6 | Q, fallback cc47 | 8.31e-45 | 7.7e-45 / 3.16e-45 | 6.52e-45 / 3.89e-46 (noise) | bounded | — | bounded | 1.79e-45 | 1.58e-44 |  |
| band_6_8|group|1e-7#3 | returned | 6 | Q, fallback cc47 | 1.01e-42 | 9.66e-43 / 4.93e-43 | 9.84e-43 / 5.91e-44 (noise) | undecided | 9.65e-43 / 7.05e-44 | undecided | 4.51e-44 | 1.71e-42 | yes |
| band_6_8|group|1e-7#33 | returned | 6 | Q, fallback cc47 | 1.3e-40 | 1.41e-40 / 4.57e-40 | 7.17e-41 / 4.84e-41 (noise) | bounded | — | bounded | 5.79e-41 | 1.91e-39 |  |
| band_6_8|group|1e-7#33 | returned | 6 | Q, fallback cc93 | 9.19e-41 | 1e-40 / 4.57e-40 | 3.11e-41 / 4.84e-41 (noise) | bounded | — | bounded | 6.08e-41 | 1.91e-39 |  |
| band_6_8|group|1e-7#39 | returned | 6 | Q, fallback cc47 | 1.7e-45 | 2.48e-45 / 3.57e-45 | 1.71e-45 / 4.41e-46 (noise) | undecided | — | undecided | 1.26e-47 | 1.59e-44 | yes |
| band_6_8|group|1e-7#40 | returned | 6 | Q, fallback cc47 | 4.22e-45 | 4.34e-45 / 8.32e-46 | 4.26e-45 / 1.28e-46 (noise) | undecided | 4.25e-45 / 8.83e-47 | undecided | 2.91e-47 | 4.07e-45 | yes |
| band_6_8|group|1e-7#40 | returned | 6 | Q, fallback cc93 | 3.9e-45 | 3.98e-45 / 8.32e-46 | 3.9e-45 / 1.28e-46 (noise) | undecided | 3.89e-45 / 8.83e-47 | undecided | 4.11e-48 | 4.07e-45 | yes |
| band_6_8|group|1e-7#46 | returned | 6 | Q, fallback cc47 | 4.17e-41 | 3.76e-41 / 1.62e-41 | 3.14e-41 / 1.1e-42 (noise) | bounded | — | bounded | 1.03e-41 | 4.71e-41 |  |
| band_6_8|group|1e-7#46 | returned | 6 | Q, fallback cc93 | 1.3e-41 | 1.38e-41 / 1.62e-41 | 7.52e-42 / 1.1e-42 (noise) | bounded | — | bounded | 5.52e-42 | 4.71e-41 |  |
| band_6_8|group|1e-7#49 | returned | 6 | Q, fallback cc47 | 5.76e-45 | 3.75e-45 / 3.27e-45 | 3.3e-45 / 6.96e-46 (noise) | bounded | — | bounded | 2.46e-45 | 1.45e-44 |  |
| band_6_8|group|1e-8#13 | returned | 6 | Q, fallback cc47 | 3.76e-41 | 3.06e-41 / 1.85e-41 | 3.68e-41 / 1.76e-42 (noise) | undecided | 3.72e-41 / 6.61e-43 | undecided | 4.22e-43 | 4.7e-41 | yes |
| band_6_8|group|1e-8#15 | returned | 6 | Q, fallback cc47 | 7.79e-45 | 1.83e-45 / 7.09e-45 | 5.07e-45 / 4.58e-46 (noise) | bounded | — | bounded | 2.72e-45 | 1.5e-44 |  |
| band_6_8|group|1e-8#21 | returned | 6 | Q, fallback cc47 | 6.94e-43 | 6.9e-43 / 2.18e-43 | 6.39e-43 / 4.42e-44 (noise) | bounded | — | bounded | 5.49e-44 | 1.85e-42 |  |
| band_6_8|group|1e-8#26 | returned | 6 | Q, fallback cc47 | 2.86e-42 | 2.39e-42 / 1.28e-42 | 2.43e-42 / 1.35e-43 (noise) | bounded | — | bounded | 4.27e-43 | 4.38e-42 |  |
| band_6_8|group|1e-8#3 | returned | 6 | Q, fallback cc47 | 3.13e-45 | 1.25e-45 / 2.85e-45 | 1.89e-45 / 1.31e-46 (noise) | bounded | — | bounded | 1.25e-45 | 4.03e-45 |  |
| band_6_8|group|1e-8#33 | returned | 6 | Q, fallback cc47 | 8.31e-45 | 7.7e-45 / 3.16e-45 | 6.52e-45 / 3.89e-46 (noise) | bounded | — | bounded | 1.79e-45 | 1.58e-44 |  |
| band_6_8|group|1e-8#44 | returned | 6 | Q, fallback cc47 | 1.86e-42 | 5.13e-42 / 5.23e-42 | 1.1e-42 / 4.32e-43 (noise) | bounded | — | bounded | 7.58e-43 | 1.36e-41 |  |
| band_6_8|group|1e-8#44 | returned | 6 | Q, fallback cc93 | 2.82e-42 | 6.22e-42 / 5.23e-42 | 2.19e-42 / 4.32e-43 (noise) | bounded | — | bounded | 6.28e-43 | 1.36e-41 |  |
| band_6_8|group|1e-8#58 | returned | 6 | Q, fallback cc47 | 7.62e-45 | 2.5e-45 / 9.95e-45 | 4.51e-45 / 3.71e-46 (noise) | bounded | — | bounded | 3.11e-45 | 1.47e-44 |  |
| band_6_8|group|1e-8#59 | returned | 6 | Q, fallback cc47 | 4.22e-45 | 4.34e-45 / 8.32e-46 | 4.26e-45 / 1.28e-46 (noise) | undecided | 4.25e-45 / 8.83e-47 | undecided | 2.91e-47 | 4.07e-45 | yes |
| band_6_8|group|1e-8#59 | returned | 6 | Q, fallback cc93 | 3.9e-45 | 3.98e-45 / 8.32e-46 | 3.9e-45 / 1.28e-46 (noise) | undecided | 3.89e-45 / 8.83e-47 | undecided | 4.11e-48 | 4.07e-45 | yes |
| band_6_8|group|1e-8#6 | returned | 6 | Q, fallback cc47 | 1.95e-42 | 2.1e-43 / 2.14e-42 | 3.45e-43 / 9.86e-44 (noise) | bounded | — | bounded | 1.6e-42 | 4.37e-42 |  |
| band_6_8|group|1e-8#8 | returned | 6 | Q, fallback cc47 | 1.54e-42 | 1.04e-42 / 5.66e-43 | 1e-42 / 4.3e-44 (noise) | bounded | 1e-42 / 2.83e-44 | bounded | 5.37e-43 | 1.67e-42 |  |
| band_6_8|group|1e-8#8 | returned | 6 | Q, fallback cc93 | 1.21e-42 | 1.21e-42 / 5.66e-43 | 1.18e-42 / 4.3e-44 (noise) | undecided | 1.18e-42 / 2.83e-44 | bounded | 3.35e-44 | 1.67e-42 |  |
| band_8_10|group|1e-7#23 | returned | 6 | Q, fallback cc47 | 7.96e-40 | 5.21e-40 / 5.15e-40 | 4.4e-40 / 3.2e-41 (noise) | bounded | — | bounded | 3.56e-40 | 1.54e-39 |  |
| band_8_10|group|1e-7#32 | returned | 6 | Q, fallback cc47 | 3.06e-39 | 2.72e-39 / 2.21e-39 | 2.09e-39 / 1.4e-40 (noise) | bounded | — | bounded | 9.72e-40 | 5.95e-39 |  |
| band_8_10|group|1e-7#32 | returned | 6 | Q, fallback cc93 | 3.9e-39 | 2.4e-39 / 2.21e-39 | 1.77e-39 / 1.4e-40 (noise) | bounded | — | bounded | 2.13e-39 | 5.95e-39 |  |
| band_8_10|group|1e-7#40 | returned | 6 | Q, fallback cc47 | 2.74e-39 | 2.07e-39 / 1.46e-39 | 2.36e-39 / 1.14e-40 (noise) | bounded | — | bounded | 3.8e-40 | 4.79e-39 |  |
| band_8_10|group|1e-7#41 | returned | 6 | Q, fallback cc47 | 3.29e-41 | 2.95e-41 / 1.12e-41 | 3.28e-41 / 1.25e-42 (noise) | undecided | — | undecided | 4.12e-44 | 5.09e-41 | yes |
| band_8_10|group|1e-7#53 | returned | 6 | Q, fallback cc47 | 8.83e-43 | 1.11e-42 / 8.27e-43 | 8.88e-43 / 4.08e-44 (noise) | undecided | 9.01e-43 / 2.98e-44 | undecided | 1.81e-44 | 1.64e-42 | yes |
| band_8_10|group|1e-7#55 | returned | 6 | Q, fallback cc47 | 4.87e-41 | 4.46e-41 / 1.82e-41 | 4.47e-41 / 1.64e-42 (noise) | bounded | — | bounded | 3.98e-42 | 4.78e-41 |  |
| band_8_10|group|1e-8#0 | returned | 6 | Q, fallback cc47 | 3.97e-42 | 3.92e-43 / 4.24e-42 | 6.92e-43 / 4.16e-43 (noise) | bounded | — | bounded | 3.27e-42 | 1.68e-41 |  |
| band_8_10|group|1e-8#23 | returned | 6 | Q, fallback cc47 | 1.65e-43 | 9.58e-43 / 1.77e-42 | 3.02e-43 / 1.6e-43 (noise) | undecided | 3.25e-43 / 1.15e-43 | understates | 1.6e-43 | 6.14e-42 |  |
| band_8_10|group|1e-8#45 | returned | 6 | Q, fallback cc47 | 4.87e-41 | 4.46e-41 / 1.82e-41 | 4.47e-41 / 1.64e-42 (noise) | bounded | — | bounded | 3.98e-42 | 4.78e-41 |  |
| band_8_10|group|1e-8#58 | returned | 6 | Q, fallback cc47 | 3.29e-41 | 2.95e-41 / 1.12e-41 | 3.28e-41 / 1.25e-42 (noise) | undecided | — | undecided | 4.12e-44 | 5.09e-41 | yes |
| band_8_10|group|1e-8#8 | returned | 6 | Q, fallback cc47 | 7.96e-40 | 5.21e-40 / 5.15e-40 | 4.4e-40 / 3.2e-41 (noise) | bounded | — | bounded | 3.56e-40 | 1.54e-39 |  |
| band_8_10|group|1e-8#9 | returned | 6 | Q, fallback cc93 | 2.09e-38 | 9.59e-39 / 1.82e-38 | 7.4e-39 / 1.22e-39 (noise) | bounded | — | bounded | 1.35e-38 | 5.21e-38 |  |
| gate_q2|group|1e-7#19 | returned | 8 | CC_93, pair93 | 3.12e-42 | 2.35e-42 / 1.2e-42 | 1.65e-42 / 5.54e-44 (noise) | bounded | — | bounded | 1.46e-42 | 1.53e-42 |  |
| gate_q2|group|1e-7#19 | returned | 8 | CC_93, tail93 | 3.12e-42 | 2.35e-42 / 1.2e-42 | 1.65e-42 / 5.54e-44 (noise) | bounded | — | bounded | 1.46e-42 | 1.53e-42 |  |
| gate_q2|group|1e-8#122 | returned | 8 | CC_93, pair93 | 2.62e-42 | 1.91e-42 / 1.07e-42 | 1.31e-42 / 5.1e-44 (noise) | bounded | — | bounded | 1.31e-42 | 1.44e-42 |  |
| gate_q2|group|1e-8#122 | returned | 8 | CC_93, tail93 | 2.62e-42 | 1.91e-42 / 1.07e-42 | 1.31e-42 / 5.1e-44 (noise) | bounded | — | bounded | 1.31e-42 | 1.44e-42 |  |
| gate_q2|group|1e-8#76 | returned | 8 | CC_93, pair93 | 2.36e-42 | 1.66e-42 / 1.01e-42 | 1.06e-42 / 5.08e-44 (noise) | bounded | — | bounded | 1.3e-42 | 1.41e-42 |  |
| gate_q2|group|1e-8#76 | returned | 8 | CC_93, tail93 | 2.38e-42 | 1.66e-42 / 1.01e-42 | 1.06e-42 / 5.08e-44 (noise) | bounded | — | bounded | 1.32e-42 | 1.41e-42 |  |
| band_10_12|group|1e-7#12 | returned | 8 | Q, fallback cc47 | 1.14e-43 | 1.65e-43 / 1.79e-43 | 1.1e-43 / 1.51e-44 (noise) | undecided | 1.16e-43 / 1.04e-44 | undecided | 1.95e-45 | 6.31e-43 | yes |
| band_10_12|group|1e-7#14 | returned | 8 | Q, fallback cc47 | 2.79e-44 | 1.07e-44 / 4.91e-44 | 2.27e-44 / 5.82e-45 (noise) | undecided | 2.47e-44 / 6.37e-45 | undecided | 3.19e-45 | 2.09e-43 | yes |
| band_10_12|group|1e-7#18 | returned | 8 | Q, fallback cc47 | 3.92e-35 | 3.7e-35 / 1e-35 | 3.87e-35 / 1.26e-36 (noise) | undecided | 3.86e-35 / 1.02e-36 | undecided | 6.05e-37 | 5.39e-35 | yes |
| band_10_12|group|1e-7#18 | returned | 8 | Q, fallback cc93 | 3.97e-35 | 3.38e-35 / 1e-35 | 3.54e-35 / 1.26e-36 (noise) | bounded | 3.53e-35 / 1.02e-36 | bounded | 4.33e-36 | 5.39e-35 |  |
| band_10_12|group|1e-7#22 | returned | 8 | Q, fallback cc47 | 3.18e-42 | 3e-42 / 6.14e-41 | 2.59e-42 / 5.09e-42 (noise) | undecided | 3.17e-42 / 6.12e-42 | undecided | 1.34e-44 | 2.03e-40 | yes |
| band_10_12|group|1e-7#24 | returned | 8 | Q, fallback cc47 | 1.12e-45 | 1.26e-45 / 7.29e-46 | 1.14e-45 / 2.44e-46 (noise) | undecided | — | undecided | 2.19e-47 | 6.77e-45 | yes |
| band_10_12|group|1e-7#24 | returned | 8 | Q, fallback cc93 | 2.41e-45 | 1.94e-45 / 7.29e-46 | 1.82e-45 / 2.44e-46 (noise) | bounded | — | bounded | 5.84e-46 | 6.77e-45 |  |
| band_10_12|group|1e-7#33 | returned | 8 | Q, fallback cc47 | 5.93e-46 | 7.05e-46 / 2.46e-45 | 5.26e-46 / 1.89e-46 (noise) | undecided | 4.75e-46 / 1.15e-46 | bounded | 1.18e-46 | 7.71e-45 |  |
| band_10_12|group|1e-7#35 | returned | 8 | Q, fallback cc47 | 1.2e-40 | 7.91e-41 / 7.64e-41 | 8.41e-41 / 5.9e-42 (noise) | bounded | 8.45e-41 / 2.08e-42 | bounded | 3.6e-41 | 2.07e-40 |  |
| band_10_12|group|1e-7#35 | returned | 8 | Q, fallback cc93 | 7.51e-41 | 6.76e-41 / 7.64e-41 | 7.25e-41 / 5.9e-42 (noise) | undecided | 7.3e-41 / 2.08e-42 | bounded | 2.14e-42 | 2.07e-40 |  |
| band_10_12|group|1e-7#45 | returned | 8 | Q, fallback cc47 | 2.29e-40 | 1.69e-40 / 1.03e-40 | 1.75e-40 / 1.84e-41 (noise) | bounded | — | bounded | 5.38e-41 | 6.01e-40 |  |
| band_10_12|group|1e-7#47 | returned | 8 | Q, fallback cc47 | 3.45e-43 | 1.64e-43 / 9.83e-43 | 1.03e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.42e-43 | 2.17e-42 |  |
| band_10_12|group|1e-7#47 | returned | 8 | Q, fallback cc93 | 3.53e-43 | 1.78e-43 / 9.83e-43 | 1.17e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.36e-43 | 2.17e-42 |  |
| band_10_12|group|1e-7#52 | returned | 8 | Q, fallback cc47 | 8.96e-43 | 9.07e-43 / 1.07e-41 | 5.01e-43 / 7.1e-43 (noise) | undecided | 3.39e-43 / 3.07e-43 | bounded | 5.57e-43 | 2.19e-41 |  |
| band_10_12|group|1e-7#7 | returned | 8 | Q, fallback cc93 | 6.73e-41 | 7.34e-41 / 2.41e-41 | 6.77e-41 / 7.62e-42 (noise) | undecided | — | undecided | 3.92e-43 | 2.08e-40 | yes |
| band_10_12|group|1e-8#14 | returned | 8 | Q, fallback cc47 | 2.29e-40 | 1.69e-40 / 1.03e-40 | 1.75e-40 / 1.84e-41 (noise) | bounded | — | bounded | 5.38e-41 | 6.01e-40 |  |
| band_10_12|group|1e-8#20 | returned | 8 | Q, fallback cc47 | 1.77e-40 | 1.67e-40 / 1.15e-40 | 1.77e-40 / 1.7e-41 (noise) | undecided | — | undecided | 2.99e-43 | 5.96e-40 | yes |
| band_10_12|group|1e-8#20 | returned | 8 | Q, fallback cc93 | 3.13e-40 | 2.26e-40 / 1.15e-40 | 2.36e-40 / 1.7e-41 (noise) | bounded | — | bounded | 7.72e-41 | 5.96e-40 |  |
| band_10_12|group|1e-8#21 | returned | 8 | Q, fallback cc47 | 5.1e-42 | 1.81e-41 / 1.17e-40 | 1.38e-42 / 5.32e-42 (noise) | undecided | 1.26e-42 / 5.19e-42 | undecided | 3.84e-42 | 2.01e-40 | yes |
| band_10_12|group|1e-8#24 | returned | 8 | Q, fallback cc93 | 6.73e-41 | 7.34e-41 / 2.41e-41 | 6.77e-41 / 7.62e-42 (noise) | undecided | — | undecided | 3.92e-43 | 2.08e-40 | yes |
| band_10_12|group|1e-8#27 | returned | 8 | Q, fallback cc47 | 1.12e-45 | 1.26e-45 / 7.29e-46 | 1.14e-45 / 2.44e-46 (noise) | undecided | — | undecided | 2.19e-47 | 6.77e-45 | yes |
| band_10_12|group|1e-8#27 | returned | 8 | Q, fallback cc93 | 2.41e-45 | 1.94e-45 / 7.29e-46 | 1.82e-45 / 2.44e-46 (noise) | bounded | — | bounded | 5.84e-46 | 6.77e-45 |  |
| band_10_12|group|1e-8#29 | returned | 8 | Q, fallback cc47 | 3.45e-43 | 1.64e-43 / 9.83e-43 | 1.03e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.42e-43 | 2.17e-42 |  |
| band_10_12|group|1e-8#29 | returned | 8 | Q, fallback cc93 | 3.53e-43 | 1.78e-43 / 9.83e-43 | 1.17e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.36e-43 | 2.17e-42 |  |
| band_10_12|group|1e-8#3 | returned | 8 | Q, fallback cc47 | 6.41e-42 | 1.84e-42 / 9.46e-42 | 1.83e-43 / 4.23e-43 (noise) | bounded | — | bounded | 6.22e-42 | 2e-41 |  |
| band_10_12|group|1e-8#3 | returned | 8 | Q, fallback cc93 | 5.2e-42 | 1.83e-42 / 9.46e-42 | 1.94e-43 / 4.23e-43 (noise) | bounded | — | bounded | 5e-42 | 2e-41 |  |
| band_10_12|group|1e-8#35 | returned | 8 | Q, fallback cc47 | 1.14e-43 | 1.65e-43 / 1.79e-43 | 1.1e-43 / 1.51e-44 (noise) | undecided | 1.16e-43 / 1.04e-44 | undecided | 1.95e-45 | 6.31e-43 | yes |
| band_10_12|group|1e-8#48 | returned | 8 | Q, fallback cc47 | 1.43e-37 | 1.11e-37 / 7.41e-38 | 1.1e-37 / 4.23e-39 (noise) | bounded | — | bounded | 3.36e-38 | 1.84e-37 |  |
| band_10_12|group|1e-8#48 | returned | 8 | Q, fallback cc93 | 1.64e-37 | 9.63e-38 / 7.41e-38 | 9.49e-38 / 4.23e-39 (noise) | bounded | — | bounded | 6.91e-38 | 1.84e-37 |  |
| band_10_12|group|1e-8#49 | returned | 8 | Q, fallback cc47 | 7.86e-42 | 7.67e-42 / 7.86e-42 | 5.58e-42 / 7.18e-43 (noise) | bounded | — | bounded | 2.27e-42 | 2.32e-41 |  |
| band_10_12|group|1e-8#59 | returned | 8 | Q, fallback cc47 | 1.93e-42 | 1.69e-42 / 8.02e-43 | 1.94e-42 / 2.03e-43 (noise) | undecided | — | undecided | 5.54e-45 | 5.99e-42 | yes |
| band_10_12|group|1e-8#7 | returned | 8 | Q, fallback cc47 | 3.41e-39 | 2.78e-39 / 8.65e-40 | 2.77e-39 / 2.65e-40 (noise) | bounded | — | bounded | 6.46e-40 | 6.33e-39 |  |
| band_12_16|analytic|1e-7#27 | returned | 8 | Q, fallback cc47 | 1.06e-26 | 6.63e-27 / 6.33e-27 | 1.4e-27 / 4.91e-28 (noise) | bounded | — | bounded | 9.24e-27 | 1.16e-26 |  |
| band_12_16|analytic|1e-7#45 | returned | 8 | Q, fallback cc47 | 9.3e-25 | 7.06e-25 / 4.09e-25 | 4.69e-26 / 4.6e-26 (noise) | bounded | — | bounded | 8.84e-25 | 1.09e-24 |  |
| band_12_16|analytic|1e-7#45 | returned | 8 | Q, fallback cc93 | 9.3e-25 | 7.84e-25 / 4.09e-25 | 3.07e-26 / 4.6e-26 (noise) | bounded | — | bounded | 9e-25 | 1.09e-24 |  |
| band_12_16|analytic|1e-8#30 | returned | 8 | Q, fallback cc47 | 9.3e-25 | 7.06e-25 / 4.09e-25 | 4.69e-26 / 4.6e-26 (noise) | bounded | — | bounded | 8.84e-25 | 1.09e-24 |  |
| band_12_16|analytic|1e-8#30 | returned | 8 | Q, fallback cc93 | 9.3e-25 | 7.84e-25 / 4.09e-25 | 3.07e-26 / 4.6e-26 (noise) | bounded | — | bounded | 9e-25 | 1.09e-24 |  |
| band_12_16|group|1e-7#34 | returned | 8 | Q, fallback cc47 | 2.13e-42 | 3.35e-42 / 2.36e-42 | 3.11e-43 / 1.52e-43 (noise) | bounded | — | bounded | 1.82e-42 | 6.77e-42 |  |
| band_12_16|group|1e-7#34 | returned | 8 | Q, fallback cc93 | 2.02e-42 | 3.34e-42 / 2.36e-42 | 3.07e-43 / 1.52e-43 (noise) | bounded | — | bounded | 1.72e-42 | 6.77e-42 |  |
| band_12_16|group|1e-7#58 | returned | 8 | Q, fallback cc47 | 1.1e-40 | 1.81e-41 / 1.37e-40 | 3.23e-41 / 6.18e-42 (noise) | bounded | — | bounded | 7.73e-41 | 2.91e-40 |  |
| band_16_20|analytic|1e-7#43 | returned | 8 | Q, fallback cc47 | 1.82e-27 | 2.54e-27 / 1.5e-27 | 7.04e-28 / 6.06e-29 (noise) | bounded | — | bounded | 1.11e-27 | 2.81e-27 |  |
| band_16_20|analytic|1e-7#8 | returned | 8 | Q, fallback cc93 | 1.14e-23 | 5.23e-24 / 1.04e-23 | 4.39e-25 / 7.54e-25 (noise) | bounded | — | bounded | 1.1e-23 | 1.46e-23 |  |
| band_16_20|analytic|1e-8#26 | returned | 8 | Q, fallback cc93 | 1.14e-23 | 5.23e-24 / 1.04e-23 | 4.39e-25 / 7.54e-25 (noise) | bounded | — | bounded | 1.1e-23 | 1.46e-23 |  |
| band_16_20|analytic|1e-8#27 | returned | 8 | Q, fallback cc47 | 9.94e-27 | 7.5e-27 / 9.52e-27 | 3.26e-27 / 8.66e-28 (noise) | bounded | — | bounded | 6.68e-27 | 1.56e-26 |  |
| band_16_20|analytic|1e-8#27 | returned | 8 | Q, fallback cc93 | 9.94e-27 | 5.28e-27 / 9.52e-27 | 1.03e-27 / 8.66e-28 (noise) | bounded | — | bounded | 8.9e-27 | 1.56e-26 |  |
| band_16_20|group|1e-8#15 | returned | 8 | Q, fallback cc47 | 3.94e-43 | 1.1e-43 / 7.68e-43 | 8.69e-44 / 9.23e-44 (noise) | bounded | — | bounded | 3.07e-43 | 3.74e-42 |  |
| band_16_20|group|1e-8#15 | returned | 8 | Q, fallback cc93 | 3.41e-43 | 6.98e-44 / 7.68e-43 | 4.69e-44 / 9.23e-44 (noise) | bounded | — | bounded | 2.94e-43 | 3.74e-42 |  |
| band_16_20|group|1e-8#59 | returned | 8 | Q, fallback cc47 | 1.18e-40 | 4.16e-42 / 1.29e-40 | 6.32e-42 / 7.21e-42 (noise) | bounded | — | bounded | 1.12e-40 | 3.08e-40 |  |
| band_16_20|group|1e-8#59 | returned | 8 | Q, fallback cc93 | 1.21e-40 | 6.15e-42 / 1.29e-40 | 8.31e-42 / 7.21e-42 (noise) | bounded | — | bounded | 1.13e-40 | 3.08e-40 |  |
| band_20_32|analytic|1e-7#0 | returned | 8 | Q, fallback cc93 | 3.2e-24 | 1.71e-24 / 3.57e-24 | 2.26e-25 / 2.86e-25 (noise) | bounded | — | bounded | 2.98e-24 | 7.49e-24 |  |
| band_20_32|analytic|1e-7#1 | returned | 8 | Q, fallback cc47 | 1.25e-23 | 5.68e-24 / 1.29e-23 | 9.01e-25 / 7.09e-25 (noise) | bounded | — | bounded | 1.16e-23 | 1.72e-23 |  |
| band_20_32|analytic|1e-7#1 | returned | 8 | Q, fallback cc93 | 8.98e-24 | 5.65e-24 / 1.29e-23 | 9.33e-25 / 7.09e-25 (noise) | bounded | — | bounded | 8.04e-24 | 1.72e-23 |  |
| band_20_32|analytic|1e-7#15 | returned | 8 | Q, fallback cc47 | 3.2e-24 | 3.06e-24 / 1.62e-24 | 5.11e-25 / 2.53e-25 (noise) | bounded | — | bounded | 2.69e-24 | 7.35e-24 |  |
| band_20_32|analytic|1e-7#17 | returned | 8 | Q, fallback cc47 | 1.86e-24 | 1.79e-24 / 2e-24 | 2.26e-25 / 1.26e-25 (noise) | bounded | — | bounded | 1.64e-24 | 3.44e-24 |  |
| band_20_32|analytic|1e-7#17 | returned | 8 | Q, fallback cc93 | 1.86e-24 | 1.76e-24 / 2e-24 | 2.04e-25 / 1.26e-25 (noise) | bounded | — | bounded | 1.66e-24 | 3.44e-24 |  |
| band_20_32|analytic|1e-7#23 | returned | 8 | Q, fallback cc93 | 1.21e-27 | 3.11e-27 / 2.65e-27 | 3.16e-30 / 1.85e-28 (noise) | bounded | — | bounded | 1.21e-27 | 4.31e-27 |  |
| band_20_32|analytic|1e-7#31 | returned | 8 | Q, fallback cc47 | 3.24e-27 | 2.37e-27 / 6.29e-27 | 1.68e-27 / 2.29e-28 (noise) | bounded | — | bounded | 1.55e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-7#31 | returned | 8 | Q, fallback cc93 | 3.24e-27 | 2.95e-27 / 6.29e-27 | 1.1e-27 / 2.29e-28 (noise) | bounded | — | bounded | 2.14e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-8#11 | returned | 8 | Q, fallback cc47 | 7.66e-26 | 1.07e-25 / 1.14e-25 | 5.13e-26 / 8.63e-27 (noise) | bounded | — | bounded | 2.53e-26 | 2.07e-25 |  |
| band_20_32|analytic|1e-8#11 | returned | 8 | Q, fallback cc93 | 7.66e-26 | 9.22e-26 / 1.14e-25 | 3.68e-26 / 8.63e-27 (noise) | bounded | — | bounded | 3.98e-26 | 2.07e-25 |  |
| band_20_32|analytic|1e-8#16 | returned | 8 | Q, fallback cc47 | 3.24e-27 | 2.9e-28 / 3.77e-27 | 1.29e-27 / 3.38e-28 (noise) | bounded | — | bounded | 1.95e-27 | 7.38e-27 |  |
| band_20_32|analytic|1e-8#35 | returned | 8 | Q, fallback cc47 | 3.39e-27 | 8.87e-28 / 3.42e-27 | 3.12e-27 / 2.9e-28 (noise) | undecided | 3.14e-27 / 2.11e-29 | bounded | 2.58e-28 | 7.66e-27 |  |
| band_20_32|analytic|1e-8#35 | returned | 8 | Q, fallback cc93 | 3.05e-27 | 1.63e-27 / 3.42e-27 | 6.12e-28 / 2.9e-28 (noise) | bounded | 6.25e-28 / 2.11e-29 | bounded | 2.42e-27 | 7.66e-27 |  |
| band_20_32|analytic|1e-8#36 | returned | 8 | Q, fallback cc93 | 1.21e-27 | 3.11e-27 / 2.65e-27 | 3.16e-30 / 1.85e-28 (noise) | bounded | — | bounded | 1.21e-27 | 4.31e-27 |  |
| band_20_32|analytic|1e-8#38 | returned | 8 | Q, fallback cc47 | 1.86e-24 | 1.79e-24 / 2e-24 | 2.26e-25 / 1.26e-25 (noise) | bounded | — | bounded | 1.64e-24 | 3.44e-24 |  |
| band_20_32|analytic|1e-8#38 | returned | 8 | Q, fallback cc93 | 1.86e-24 | 1.76e-24 / 2e-24 | 2.04e-25 / 1.26e-25 (noise) | bounded | — | bounded | 1.66e-24 | 3.44e-24 |  |
| band_20_32|analytic|1e-8#49 | returned | 8 | Q, fallback cc47 | 3.85e-24 | 1.29e-24 / 3.32e-24 | 1.08e-24 / 3.36e-25 (noise) | bounded | — | bounded | 2.77e-24 | 7.64e-24 |  |
| band_20_32|analytic|1e-8#51 | returned | 8 | Q, fallback cc93 | 2.04e-27 | 7.98e-28 / 1.65e-27 | 6.94e-29 / 2.99e-28 (noise) | bounded | — | bounded | 1.97e-27 | 6.92e-27 |  |
| band_20_32|analytic|1e-8#60 | returned | 8 | Q, fallback cc93 | 3.2e-24 | 1.71e-24 / 3.57e-24 | 2.26e-25 / 2.86e-25 (noise) | bounded | — | bounded | 2.98e-24 | 7.49e-24 |  |
| band_20_32|analytic|1e-8#84 | returned | 8 | Q, fallback cc47 | 3.24e-27 | 2.37e-27 / 6.29e-27 | 1.68e-27 / 2.29e-28 (noise) | bounded | — | bounded | 1.55e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-8#84 | returned | 8 | Q, fallback cc93 | 3.24e-27 | 2.95e-27 / 6.29e-27 | 1.1e-27 / 2.29e-28 (noise) | bounded | — | bounded | 2.14e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-8#96 | returned | 8 | Q, fallback cc47 | 3.2e-24 | 3.06e-24 / 1.62e-24 | 5.11e-25 / 2.53e-25 (noise) | bounded | — | bounded | 2.69e-24 | 7.35e-24 |  |
| band_20_32|group|1e-7#25 | returned | 8 | Q, fallback cc47 | 4.09e-45 | 7.73e-46 / 4.4e-45 | 6.2e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.47e-45 | 1.42e-44 |  |
| band_20_32|group|1e-7#25 | returned | 8 | Q, fallback cc93 | 3.72e-45 | 4.92e-46 / 4.4e-45 | 3.39e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.38e-45 | 1.42e-44 |  |
| band_20_32|group|1e-7#27 | returned | 8 | Q, fallback cc93 | 2.01e-39 | 1.13e-39 / 1.38e-39 | 4.23e-40 / 7.31e-41 (noise) | bounded | — | bounded | 1.58e-39 | 2.84e-39 |  |
| band_20_32|group|1e-7#69 | returned | 8 | Q, fallback cc47 | 1.5e-39 | 5.59e-40 / 1.48e-39 | 3.61e-40 / 6.99e-41 (noise) | bounded | — | bounded | 1.14e-39 | 3.11e-39 |  |
| band_20_32|group|1e-7#90 | returned | 8 | Q, fallback cc47 | 6.35e-42 | 2.82e-42 / 4.79e-42 | 3.5e-43 / 2.7e-43 (noise) | bounded | — | bounded | 6e-42 | 1.41e-41 |  |
| band_20_32|group|1e-7#92 | returned | 8 | Q, fallback cc47 | 1.82e-42 | 8.17e-43 / 1.26e-42 | 2.94e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.53e-42 | 4.26e-42 |  |
| band_20_32|group|1e-7#92 | returned | 8 | Q, fallback cc93 | 1.6e-42 | 6.61e-43 / 1.26e-42 | 1.38e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.46e-42 | 4.26e-42 |  |
| band_20_32|group|1e-8#16 | returned | 8 | Q, fallback cc93 | 4.48e-44 | 2.32e-45 / 4.57e-44 | 2.73e-45 / 2.71e-45 (noise) | bounded | — | bounded | 4.21e-44 | 1.78e-43 |  |
| band_20_32|group|1e-8#48 | returned | 8 | Q, fallback cc47 | 2.49e-42 | 4.09e-42 / 6.34e-42 | 1.78e-43 / 1.11e-42 (noise) | bounded | — | bounded | 2.32e-42 | 3.86e-41 |  |
| band_20_32|group|1e-8#48 | returned | 8 | Q, fallback cc93 | 2.33e-42 | 4.24e-42 / 6.34e-42 | 3.33e-43 / 1.11e-42 (noise) | bounded | — | bounded | 1.99e-42 | 3.86e-41 |  |
| band_20_32|group|1e-8#50 | returned | 8 | Q, fallback cc47 | 1.82e-42 | 8.17e-43 / 1.26e-42 | 2.94e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.53e-42 | 4.26e-42 |  |
| band_20_32|group|1e-8#50 | returned | 8 | Q, fallback cc93 | 1.6e-42 | 6.61e-43 / 1.26e-42 | 1.38e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.46e-42 | 4.26e-42 |  |
| band_20_32|group|1e-8#57 | returned | 8 | Q, fallback cc47 | 5.55e-38 | 1.31e-39 / 6.1e-38 | 1.2e-39 / 3.14e-39 (noise) | bounded | — | bounded | 5.43e-38 | 1.42e-37 |  |
| band_20_32|group|1e-8#57 | returned | 8 | Q, fallback cc93 | 5.59e-38 | 3.03e-39 / 6.1e-38 | 5.23e-40 / 3.14e-39 (noise) | bounded | — | bounded | 5.53e-38 | 1.42e-37 |  |
| band_20_32|group|1e-8#67 | returned | 8 | Q, fallback cc93 | 3.21e-45 | 5.03e-46 / 6.59e-45 | 5.88e-46 / 3.04e-46 (noise) | bounded | — | bounded | 2.62e-45 | 1.58e-44 |  |
| band_20_32|group|1e-8#69 | returned | 8 | Q, fallback cc93 | 1.45e-41 | 1.21e-42 / 2.03e-41 | 2.98e-42 / 7.94e-43 (noise) | bounded | — | bounded | 1.15e-41 | 4.63e-41 |  |
| band_20_32|group|1e-8#94 | returned | 8 | Q, fallback cc47 | 4.09e-45 | 7.73e-46 / 4.4e-45 | 6.2e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.47e-45 | 1.42e-44 |  |
| band_20_32|group|1e-8#94 | returned | 8 | Q, fallback cc93 | 3.72e-45 | 4.92e-46 / 4.4e-45 | 3.39e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.38e-45 | 1.42e-44 |  |
| band_8_10|group|1e-7#23 | returned | 8 | Q, fallback cc47 | 7.96e-40 | 5.21e-40 / 5.15e-40 | 4.4e-40 / 3.2e-41 (noise) | bounded | — | bounded | 3.56e-40 | 1.54e-39 |  |
| band_8_10|group|1e-7#32 | returned | 8 | Q, fallback cc47 | 3.06e-39 | 2.72e-39 / 2.21e-39 | 2.09e-39 / 1.4e-40 (noise) | bounded | — | bounded | 9.72e-40 | 5.95e-39 |  |
| band_8_10|group|1e-7#32 | returned | 8 | Q, fallback cc93 | 3.9e-39 | 2.4e-39 / 2.21e-39 | 1.77e-39 / 1.4e-40 (noise) | bounded | — | bounded | 2.13e-39 | 5.95e-39 |  |
| band_8_10|group|1e-7#40 | returned | 8 | Q, fallback cc47 | 2.74e-39 | 2.07e-39 / 1.46e-39 | 2.36e-39 / 1.14e-40 (noise) | bounded | — | bounded | 3.8e-40 | 4.79e-39 |  |
| band_8_10|group|1e-7#41 | returned | 8 | Q, fallback cc47 | 3.29e-41 | 2.95e-41 / 1.12e-41 | 3.28e-41 / 1.25e-42 (noise) | undecided | — | undecided | 4.12e-44 | 5.09e-41 | yes |
| band_8_10|group|1e-7#53 | returned | 8 | Q, fallback cc47 | 8.83e-43 | 1.11e-42 / 8.27e-43 | 8.88e-43 / 4.08e-44 (noise) | undecided | 9.01e-43 / 2.98e-44 | undecided | 1.81e-44 | 1.64e-42 | yes |
| band_8_10|group|1e-7#55 | returned | 8 | Q, fallback cc47 | 4.87e-41 | 4.46e-41 / 1.82e-41 | 4.47e-41 / 1.64e-42 (noise) | bounded | — | bounded | 3.98e-42 | 4.78e-41 |  |
| band_8_10|group|1e-8#0 | returned | 8 | Q, fallback cc47 | 3.97e-42 | 3.92e-43 / 4.24e-42 | 6.92e-43 / 4.16e-43 (noise) | bounded | — | bounded | 3.27e-42 | 1.68e-41 |  |
| band_8_10|group|1e-8#23 | returned | 8 | Q, fallback cc47 | 1.65e-43 | 9.58e-43 / 1.77e-42 | 3.02e-43 / 1.6e-43 (noise) | undecided | 3.25e-43 / 1.15e-43 | understates | 1.6e-43 | 6.14e-42 |  |
| band_8_10|group|1e-8#45 | returned | 8 | Q, fallback cc47 | 4.87e-41 | 4.46e-41 / 1.82e-41 | 4.47e-41 / 1.64e-42 (noise) | bounded | — | bounded | 3.98e-42 | 4.78e-41 |  |
| band_8_10|group|1e-8#58 | returned | 8 | Q, fallback cc47 | 3.29e-41 | 2.95e-41 / 1.12e-41 | 3.28e-41 / 1.25e-42 (noise) | undecided | — | undecided | 4.12e-44 | 5.09e-41 | yes |
| band_8_10|group|1e-8#8 | returned | 8 | Q, fallback cc47 | 7.96e-40 | 5.21e-40 / 5.15e-40 | 4.4e-40 / 3.2e-41 (noise) | bounded | — | bounded | 3.56e-40 | 1.54e-39 |  |
| band_8_10|group|1e-8#9 | returned | 8 | Q, fallback cc93 | 2.09e-38 | 9.59e-39 / 1.82e-38 | 7.4e-39 / 1.22e-39 (noise) | bounded | — | bounded | 1.35e-38 | 5.21e-38 |  |
| gate_q2|group|1e-7#19 | returned | 10 | CC_93, pair93 | 3.12e-42 | 2.35e-42 / 1.2e-42 | 1.65e-42 / 5.54e-44 (noise) | bounded | — | bounded | 1.46e-42 | 1.53e-42 |  |
| gate_q2|group|1e-7#19 | returned | 10 | CC_93, tail93 | 3.12e-42 | 2.35e-42 / 1.2e-42 | 1.65e-42 / 5.54e-44 (noise) | bounded | — | bounded | 1.46e-42 | 1.53e-42 |  |
| gate_q2|group|1e-8#122 | returned | 10 | CC_93, pair93 | 2.62e-42 | 1.91e-42 / 1.07e-42 | 1.31e-42 / 5.1e-44 (noise) | bounded | — | bounded | 1.31e-42 | 1.44e-42 |  |
| gate_q2|group|1e-8#122 | returned | 10 | CC_93, tail93 | 2.62e-42 | 1.91e-42 / 1.07e-42 | 1.31e-42 / 5.1e-44 (noise) | bounded | — | bounded | 1.31e-42 | 1.44e-42 |  |
| gate_q2|group|1e-8#76 | returned | 10 | CC_93, pair93 | 2.36e-42 | 1.66e-42 / 1.01e-42 | 1.06e-42 / 5.08e-44 (noise) | bounded | — | bounded | 1.3e-42 | 1.41e-42 |  |
| gate_q2|group|1e-8#76 | returned | 10 | CC_93, tail93 | 2.38e-42 | 1.66e-42 / 1.01e-42 | 1.06e-42 / 5.08e-44 (noise) | bounded | — | bounded | 1.32e-42 | 1.41e-42 |  |
| band_10_12|group|1e-7#12 | returned | 10 | Q, fallback cc47 | 1.14e-43 | 1.65e-43 / 1.79e-43 | 1.1e-43 / 1.51e-44 (noise) | undecided | 1.16e-43 / 1.04e-44 | undecided | 1.95e-45 | 6.31e-43 | yes |
| band_10_12|group|1e-7#14 | returned | 10 | Q, fallback cc47 | 2.79e-44 | 1.07e-44 / 4.91e-44 | 2.27e-44 / 5.82e-45 (noise) | undecided | 2.47e-44 / 6.37e-45 | undecided | 3.19e-45 | 2.09e-43 | yes |
| band_10_12|group|1e-7#18 | returned | 10 | Q, fallback cc47 | 3.92e-35 | 3.7e-35 / 1e-35 | 3.87e-35 / 1.26e-36 (noise) | undecided | 3.86e-35 / 1.02e-36 | undecided | 6.05e-37 | 5.39e-35 | yes |
| band_10_12|group|1e-7#18 | returned | 10 | Q, fallback cc93 | 3.97e-35 | 3.38e-35 / 1e-35 | 3.54e-35 / 1.26e-36 (noise) | bounded | 3.53e-35 / 1.02e-36 | bounded | 4.33e-36 | 5.39e-35 |  |
| band_10_12|group|1e-7#22 | returned | 10 | Q, fallback cc47 | 3.18e-42 | 3e-42 / 6.14e-41 | 2.59e-42 / 5.09e-42 (noise) | undecided | 3.17e-42 / 6.12e-42 | undecided | 1.34e-44 | 2.03e-40 | yes |
| band_10_12|group|1e-7#24 | returned | 10 | Q, fallback cc47 | 1.12e-45 | 1.26e-45 / 7.29e-46 | 1.14e-45 / 2.44e-46 (noise) | undecided | — | undecided | 2.19e-47 | 6.77e-45 | yes |
| band_10_12|group|1e-7#24 | returned | 10 | Q, fallback cc93 | 2.41e-45 | 1.94e-45 / 7.29e-46 | 1.82e-45 / 2.44e-46 (noise) | bounded | — | bounded | 5.84e-46 | 6.77e-45 |  |
| band_10_12|group|1e-7#33 | returned | 10 | Q, fallback cc47 | 5.93e-46 | 7.05e-46 / 2.46e-45 | 5.26e-46 / 1.89e-46 (noise) | undecided | 4.75e-46 / 1.15e-46 | bounded | 1.18e-46 | 7.71e-45 |  |
| band_10_12|group|1e-7#35 | returned | 10 | Q, fallback cc47 | 1.2e-40 | 7.91e-41 / 7.64e-41 | 8.41e-41 / 5.9e-42 (noise) | bounded | 8.45e-41 / 2.08e-42 | bounded | 3.6e-41 | 2.07e-40 |  |
| band_10_12|group|1e-7#35 | returned | 10 | Q, fallback cc93 | 7.51e-41 | 6.76e-41 / 7.64e-41 | 7.25e-41 / 5.9e-42 (noise) | undecided | 7.3e-41 / 2.08e-42 | bounded | 2.14e-42 | 2.07e-40 |  |
| band_10_12|group|1e-7#45 | returned | 10 | Q, fallback cc47 | 2.29e-40 | 1.69e-40 / 1.03e-40 | 1.75e-40 / 1.84e-41 (noise) | bounded | — | bounded | 5.38e-41 | 6.01e-40 |  |
| band_10_12|group|1e-7#47 | returned | 10 | Q, fallback cc47 | 3.45e-43 | 1.64e-43 / 9.83e-43 | 1.03e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.42e-43 | 2.17e-42 |  |
| band_10_12|group|1e-7#47 | returned | 10 | Q, fallback cc93 | 3.53e-43 | 1.78e-43 / 9.83e-43 | 1.17e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.36e-43 | 2.17e-42 |  |
| band_10_12|group|1e-7#52 | returned | 10 | Q, fallback cc47 | 8.96e-43 | 9.07e-43 / 1.07e-41 | 5.01e-43 / 7.1e-43 (noise) | undecided | 3.39e-43 / 3.07e-43 | bounded | 5.57e-43 | 2.19e-41 |  |
| band_10_12|group|1e-7#7 | returned | 10 | Q, fallback cc93 | 6.73e-41 | 7.34e-41 / 2.41e-41 | 6.77e-41 / 7.62e-42 (noise) | undecided | — | undecided | 3.92e-43 | 2.08e-40 | yes |
| band_10_12|group|1e-8#14 | returned | 10 | Q, fallback cc47 | 2.29e-40 | 1.69e-40 / 1.03e-40 | 1.75e-40 / 1.84e-41 (noise) | bounded | — | bounded | 5.38e-41 | 6.01e-40 |  |
| band_10_12|group|1e-8#20 | returned | 10 | Q, fallback cc47 | 1.77e-40 | 1.67e-40 / 1.15e-40 | 1.77e-40 / 1.7e-41 (noise) | undecided | — | undecided | 2.99e-43 | 5.96e-40 | yes |
| band_10_12|group|1e-8#20 | returned | 10 | Q, fallback cc93 | 3.13e-40 | 2.26e-40 / 1.15e-40 | 2.36e-40 / 1.7e-41 (noise) | bounded | — | bounded | 7.72e-41 | 5.96e-40 |  |
| band_10_12|group|1e-8#21 | returned | 10 | Q, fallback cc47 | 5.1e-42 | 1.81e-41 / 1.17e-40 | 1.38e-42 / 5.32e-42 (noise) | undecided | 1.26e-42 / 5.19e-42 | undecided | 3.84e-42 | 2.01e-40 | yes |
| band_10_12|group|1e-8#24 | returned | 10 | Q, fallback cc93 | 6.73e-41 | 7.34e-41 / 2.41e-41 | 6.77e-41 / 7.62e-42 (noise) | undecided | — | undecided | 3.92e-43 | 2.08e-40 | yes |
| band_10_12|group|1e-8#27 | returned | 10 | Q, fallback cc47 | 1.12e-45 | 1.26e-45 / 7.29e-46 | 1.14e-45 / 2.44e-46 (noise) | undecided | — | undecided | 2.19e-47 | 6.77e-45 | yes |
| band_10_12|group|1e-8#27 | returned | 10 | Q, fallback cc93 | 2.41e-45 | 1.94e-45 / 7.29e-46 | 1.82e-45 / 2.44e-46 (noise) | bounded | — | bounded | 5.84e-46 | 6.77e-45 |  |
| band_10_12|group|1e-8#29 | returned | 10 | Q, fallback cc47 | 3.45e-43 | 1.64e-43 / 9.83e-43 | 1.03e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.42e-43 | 2.17e-42 |  |
| band_10_12|group|1e-8#29 | returned | 10 | Q, fallback cc93 | 3.53e-43 | 1.78e-43 / 9.83e-43 | 1.17e-43 / 4.26e-44 (noise) | bounded | — | bounded | 2.36e-43 | 2.17e-42 |  |
| band_10_12|group|1e-8#3 | returned | 10 | Q, fallback cc47 | 6.41e-42 | 1.84e-42 / 9.46e-42 | 1.83e-43 / 4.23e-43 (noise) | bounded | — | bounded | 6.22e-42 | 2e-41 |  |
| band_10_12|group|1e-8#3 | returned | 10 | Q, fallback cc93 | 5.2e-42 | 1.83e-42 / 9.46e-42 | 1.94e-43 / 4.23e-43 (noise) | bounded | — | bounded | 5e-42 | 2e-41 |  |
| band_10_12|group|1e-8#35 | returned | 10 | Q, fallback cc47 | 1.14e-43 | 1.65e-43 / 1.79e-43 | 1.1e-43 / 1.51e-44 (noise) | undecided | 1.16e-43 / 1.04e-44 | undecided | 1.95e-45 | 6.31e-43 | yes |
| band_10_12|group|1e-8#48 | returned | 10 | Q, fallback cc47 | 1.43e-37 | 1.11e-37 / 7.41e-38 | 1.1e-37 / 4.23e-39 (noise) | bounded | — | bounded | 3.36e-38 | 1.84e-37 |  |
| band_10_12|group|1e-8#48 | returned | 10 | Q, fallback cc93 | 1.64e-37 | 9.63e-38 / 7.41e-38 | 9.49e-38 / 4.23e-39 (noise) | bounded | — | bounded | 6.91e-38 | 1.84e-37 |  |
| band_10_12|group|1e-8#49 | returned | 10 | Q, fallback cc47 | 7.86e-42 | 7.67e-42 / 7.86e-42 | 5.58e-42 / 7.18e-43 (noise) | bounded | — | bounded | 2.27e-42 | 2.32e-41 |  |
| band_10_12|group|1e-8#59 | returned | 10 | Q, fallback cc47 | 1.93e-42 | 1.69e-42 / 8.02e-43 | 1.94e-42 / 2.03e-43 (noise) | undecided | — | undecided | 5.54e-45 | 5.99e-42 | yes |
| band_10_12|group|1e-8#7 | returned | 10 | Q, fallback cc47 | 3.41e-39 | 2.78e-39 / 8.65e-40 | 2.77e-39 / 2.65e-40 (noise) | bounded | — | bounded | 6.46e-40 | 6.33e-39 |  |
| band_12_16|analytic|1e-7#27 | returned | 10 | Q, fallback cc47 | 1.06e-26 | 6.63e-27 / 6.33e-27 | 1.4e-27 / 4.91e-28 (noise) | bounded | — | bounded | 9.24e-27 | 1.16e-26 |  |
| band_12_16|analytic|1e-7#45 | returned | 10 | Q, fallback cc47 | 9.3e-25 | 7.06e-25 / 4.09e-25 | 4.69e-26 / 4.6e-26 (noise) | bounded | — | bounded | 8.84e-25 | 1.09e-24 |  |
| band_12_16|analytic|1e-7#45 | returned | 10 | Q, fallback cc93 | 9.3e-25 | 7.84e-25 / 4.09e-25 | 3.07e-26 / 4.6e-26 (noise) | bounded | — | bounded | 9e-25 | 1.09e-24 |  |
| band_12_16|analytic|1e-8#30 | returned | 10 | Q, fallback cc47 | 9.3e-25 | 7.06e-25 / 4.09e-25 | 4.69e-26 / 4.6e-26 (noise) | bounded | — | bounded | 8.84e-25 | 1.09e-24 |  |
| band_12_16|analytic|1e-8#30 | returned | 10 | Q, fallback cc93 | 9.3e-25 | 7.84e-25 / 4.09e-25 | 3.07e-26 / 4.6e-26 (noise) | bounded | — | bounded | 9e-25 | 1.09e-24 |  |
| band_12_16|group|1e-7#34 | returned | 10 | Q, fallback cc47 | 2.13e-42 | 3.35e-42 / 2.36e-42 | 3.11e-43 / 1.52e-43 (noise) | bounded | — | bounded | 1.82e-42 | 6.77e-42 |  |
| band_12_16|group|1e-7#34 | returned | 10 | Q, fallback cc93 | 2.02e-42 | 3.34e-42 / 2.36e-42 | 3.07e-43 / 1.52e-43 (noise) | bounded | — | bounded | 1.72e-42 | 6.77e-42 |  |
| band_12_16|group|1e-7#58 | returned | 10 | Q, fallback cc47 | 1.1e-40 | 1.81e-41 / 1.37e-40 | 3.23e-41 / 6.18e-42 (noise) | bounded | — | bounded | 7.73e-41 | 2.91e-40 |  |
| band_16_20|analytic|1e-7#43 | returned | 10 | Q, fallback cc47 | 1.82e-27 | 2.54e-27 / 1.5e-27 | 7.04e-28 / 6.06e-29 (noise) | bounded | — | bounded | 1.11e-27 | 2.81e-27 |  |
| band_16_20|analytic|1e-8#27 | returned | 10 | Q, fallback cc93 | 1.14e-26 | 4.18e-27 / 9.52e-27 | 6.31e-29 / 8.66e-28 (noise) | bounded | — | bounded | 1.13e-26 | 1.56e-26 |  |
| band_16_20|group|1e-8#15 | returned | 10 | Q, fallback cc93 | 5.32e-43 | 6.98e-44 / 7.68e-43 | 4.69e-44 / 9.23e-44 (noise) | bounded | — | bounded | 4.86e-43 | 3.74e-42 |  |
| band_16_20|group|1e-8#32 | returned | 10 | Q, fallback cc47 | 4.94e-42 | 5.18e-42 / 3.93e-42 | 4.9e-42 / 2.38e-43 (noise) | undecided | 4.79e-42 / 3.1e-43 | undecided | 1.5e-43 | 1.09e-41 | yes |
| band_16_20|group|1e-8#32 | returned | 10 | Q, fallback cc93 | 4.03e-42 | 4.25e-42 / 3.93e-42 | 3.97e-42 / 2.38e-43 (noise) | undecided | 3.86e-42 / 3.1e-43 | undecided | 1.69e-43 | 1.09e-41 | yes |
| band_16_20|group|1e-8#59 | returned | 10 | Q, fallback cc93 | 1.2e-40 | 6.15e-42 / 1.29e-40 | 8.31e-42 / 7.21e-42 (noise) | bounded | — | bounded | 1.12e-40 | 3.08e-40 |  |
| band_16_20|group|1e-8#60 | returned | 10 | Q, fallback cc47 | 5.96e-40 | 7.45e-40 / 1.55e-39 | 4.95e-40 / 2.77e-40 (noise) | undecided | 4.24e-40 / 1.86e-40 | undecided | 1.72e-40 | 1.09e-38 | yes |
| band_16_20|group|1e-8#71 | returned | 10 | Q, fallback cc93 | 2e-44 | 1.59e-44 / 4.83e-44 | 1.55e-44 / 3.3e-45 (noise) | bounded | — | bounded | 4.53e-45 | 1.05e-43 |  |
| band_20_32|analytic|1e-7#0 | returned | 10 | Q, fallback cc93 | 3.2e-24 | 1.73e-24 / 3.57e-24 | 2.46e-25 / 2.86e-25 (noise) | bounded | — | bounded | 2.96e-24 | 7.49e-24 |  |
| band_20_32|analytic|1e-7#1 | returned | 10 | Q, fallback cc47 | 1.25e-23 | 5.68e-24 / 1.29e-23 | 9.01e-25 / 7.09e-25 (noise) | bounded | — | bounded | 1.16e-23 | 1.72e-23 |  |
| band_20_32|analytic|1e-7#1 | returned | 10 | Q, fallback cc93 | 8.98e-24 | 5.65e-24 / 1.29e-23 | 9.33e-25 / 7.09e-25 (noise) | bounded | — | bounded | 8.04e-24 | 1.72e-23 |  |
| band_20_32|analytic|1e-7#10 | returned | 10 | Q, fallback cc47 | 7.32e-27 | 5.92e-27 / 7.41e-27 | 1.13e-27 / 8.92e-28 (noise) | bounded | — | bounded | 6.19e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-7#10 | returned | 10 | Q, fallback cc93 | 6.16e-27 | 5.43e-27 / 7.41e-27 | 6.44e-28 / 8.92e-28 (noise) | bounded | — | bounded | 5.52e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-7#17 | returned | 10 | Q, fallback cc47 | 1.86e-24 | 1.79e-24 / 2e-24 | 2.26e-25 / 1.26e-25 (noise) | bounded | — | bounded | 1.64e-24 | 3.44e-24 |  |
| band_20_32|analytic|1e-7#17 | returned | 10 | Q, fallback cc93 | 1.86e-24 | 1.76e-24 / 2e-24 | 2.04e-25 / 1.26e-25 (noise) | bounded | — | bounded | 1.66e-24 | 3.44e-24 |  |
| band_20_32|analytic|1e-7#23 | returned | 10 | Q, fallback cc47 | 1.21e-27 | 3.53e-27 / 2.65e-27 | 4.29e-28 / 1.85e-28 (noise) | bounded | — | bounded | 7.84e-28 | 4.31e-27 |  |
| band_20_32|analytic|1e-7#31 | returned | 10 | Q, fallback cc47 | 3.24e-27 | 2.37e-27 / 6.29e-27 | 1.68e-27 / 2.29e-28 (noise) | bounded | — | bounded | 1.55e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-7#31 | returned | 10 | Q, fallback cc93 | 3.24e-27 | 2.95e-27 / 6.29e-27 | 1.1e-27 / 2.29e-28 (noise) | bounded | — | bounded | 2.14e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-7#39 | returned | 10 | Q, fallback cc47 | 7.93e-27 | 2.02e-27 / 7.26e-27 | 2.63e-27 / 1.11e-27 (noise) | bounded | — | bounded | 5.31e-27 | 2.46e-26 |  |
| band_20_32|analytic|1e-7#49 | returned | 10 | Q, fallback cc93 | 2.04e-27 | 1.6e-27 / 3.26e-27 | 3.59e-28 / 3.03e-28 (noise) | bounded | — | bounded | 1.68e-27 | 6.58e-27 |  |
| band_20_32|analytic|1e-7#52 | returned | 10 | Q, fallback cc47 | 1.22e-27 | 8.28e-28 / 2.39e-27 | 8.01e-28 / 1.47e-28 (noise) | bounded | — | bounded | 4.18e-28 | 4.73e-27 |  |
| band_20_32|analytic|1e-7#54 | returned | 10 | Q, fallback cc47 | 7.32e-27 | 5.92e-27 / 7.41e-27 | 1.13e-27 / 8.92e-28 (noise) | bounded | — | bounded | 6.19e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-7#54 | returned | 10 | Q, fallback cc93 | 6.16e-27 | 5.43e-27 / 7.41e-27 | 6.44e-28 / 8.92e-28 (noise) | bounded | — | bounded | 5.52e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-7#74 | returned | 10 | Q, fallback cc93 | 2.04e-27 | 1.6e-27 / 3.26e-27 | 3.59e-28 / 3.03e-28 (noise) | bounded | — | bounded | 1.68e-27 | 6.58e-27 |  |
| band_20_32|analytic|1e-8#11 | returned | 10 | Q, fallback cc47 | 7.66e-26 | 1.07e-25 / 1.14e-25 | 5.13e-26 / 8.63e-27 (noise) | bounded | — | bounded | 2.53e-26 | 2.07e-25 |  |
| band_20_32|analytic|1e-8#11 | returned | 10 | Q, fallback cc93 | 7.66e-26 | 9.22e-26 / 1.14e-25 | 3.68e-26 / 8.63e-27 (noise) | bounded | — | bounded | 3.98e-26 | 2.07e-25 |  |
| band_20_32|analytic|1e-8#16 | returned | 10 | Q, fallback cc47 | 3.24e-27 | 2.9e-28 / 3.77e-27 | 1.29e-27 / 3.38e-28 (noise) | bounded | — | bounded | 1.95e-27 | 7.38e-27 |  |
| band_20_32|analytic|1e-8#23 | returned | 10 | Q, fallback cc47 | 7.93e-27 | 2.02e-27 / 7.26e-27 | 2.63e-27 / 1.11e-27 (noise) | bounded | — | bounded | 5.31e-27 | 2.46e-26 |  |
| band_20_32|analytic|1e-8#35 | returned | 10 | Q, fallback cc47 | 3.39e-27 | 8.87e-28 / 3.42e-27 | 3.12e-27 / 2.9e-28 (noise) | undecided | 3.14e-27 / 2.11e-29 | bounded | 2.58e-28 | 7.66e-27 |  |
| band_20_32|analytic|1e-8#35 | returned | 10 | Q, fallback cc93 | 3.05e-27 | 1.63e-27 / 3.42e-27 | 6.12e-28 / 2.9e-28 (noise) | bounded | 6.25e-28 / 2.11e-29 | bounded | 2.42e-27 | 7.66e-27 |  |
| band_20_32|analytic|1e-8#36 | returned | 10 | Q, fallback cc47 | 1.21e-27 | 3.53e-27 / 2.65e-27 | 4.29e-28 / 1.85e-28 (noise) | bounded | — | bounded | 7.84e-28 | 4.31e-27 |  |
| band_20_32|analytic|1e-8#38 | returned | 10 | Q, fallback cc47 | 1.86e-24 | 1.79e-24 / 2e-24 | 2.26e-25 / 1.26e-25 (noise) | bounded | — | bounded | 1.64e-24 | 3.44e-24 |  |
| band_20_32|analytic|1e-8#38 | returned | 10 | Q, fallback cc93 | 1.86e-24 | 1.76e-24 / 2e-24 | 2.04e-25 / 1.26e-25 (noise) | bounded | — | bounded | 1.66e-24 | 3.44e-24 |  |
| band_20_32|analytic|1e-8#49 | returned | 10 | Q, fallback cc47 | 3.85e-24 | 1.8e-24 / 3.32e-24 | 1.6e-24 / 3.36e-25 (noise) | bounded | — | bounded | 2.26e-24 | 7.64e-24 |  |
| band_20_32|analytic|1e-8#51 | returned | 10 | Q, fallback cc47 | 2.04e-27 | 9.62e-28 / 1.65e-27 | 2.34e-28 / 2.99e-28 (noise) | bounded | — | bounded | 1.81e-27 | 6.92e-27 |  |
| band_20_32|analytic|1e-8#60 | returned | 10 | Q, fallback cc93 | 3.2e-24 | 1.73e-24 / 3.57e-24 | 2.46e-25 / 2.86e-25 (noise) | bounded | — | bounded | 2.96e-24 | 7.49e-24 |  |
| band_20_32|analytic|1e-8#78 | returned | 10 | Q, fallback cc47 | 7.32e-27 | 5.92e-27 / 7.41e-27 | 1.13e-27 / 8.92e-28 (noise) | bounded | — | bounded | 6.19e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-8#78 | returned | 10 | Q, fallback cc93 | 6.16e-27 | 5.43e-27 / 7.41e-27 | 6.44e-28 / 8.92e-28 (noise) | bounded | — | bounded | 5.52e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-8#84 | returned | 10 | Q, fallback cc47 | 3.24e-27 | 2.37e-27 / 6.29e-27 | 1.68e-27 / 2.29e-28 (noise) | bounded | — | bounded | 1.55e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-8#84 | returned | 10 | Q, fallback cc93 | 3.24e-27 | 2.95e-27 / 6.29e-27 | 1.1e-27 / 2.29e-28 (noise) | bounded | — | bounded | 2.14e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-8#86 | returned | 10 | Q, fallback cc47 | 6.94e-27 | 1.84e-27 / 8.53e-27 | 1.51e-28 / 9.74e-28 (noise) | bounded | — | bounded | 6.79e-27 | 2.16e-26 |  |
| band_20_32|analytic|1e-8#86 | returned | 10 | Q, fallback cc93 | 6.94e-27 | 3.77e-27 / 8.53e-27 | 1.78e-27 / 9.74e-28 (noise) | bounded | — | bounded | 5.17e-27 | 2.16e-26 |  |
| band_20_32|analytic|1e-8#94 | returned | 10 | Q, fallback cc93 | 2.04e-27 | 1.6e-27 / 3.26e-27 | 3.59e-28 / 3.03e-28 (noise) | bounded | — | bounded | 1.68e-27 | 6.58e-27 |  |
| band_20_32|group|1e-7#25 | returned | 10 | Q, fallback cc47 | 4.09e-45 | 7.73e-46 / 4.4e-45 | 6.2e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.47e-45 | 1.42e-44 |  |
| band_20_32|group|1e-7#25 | returned | 10 | Q, fallback cc93 | 3.72e-45 | 4.92e-46 / 4.4e-45 | 3.39e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.38e-45 | 1.42e-44 |  |
| band_20_32|group|1e-7#27 | returned | 10 | Q, fallback cc93 | 2.01e-39 | 1.13e-39 / 1.38e-39 | 4.23e-40 / 7.31e-41 (noise) | bounded | — | bounded | 1.58e-39 | 2.84e-39 |  |
| band_20_32|group|1e-7#69 | returned | 10 | Q, fallback cc47 | 1.5e-39 | 5.59e-40 / 1.48e-39 | 3.61e-40 / 6.99e-41 (noise) | bounded | — | bounded | 1.14e-39 | 3.11e-39 |  |
| band_20_32|group|1e-7#90 | returned | 10 | Q, fallback cc47 | 6.35e-42 | 2.82e-42 / 4.79e-42 | 3.5e-43 / 2.7e-43 (noise) | bounded | — | bounded | 6e-42 | 1.41e-41 |  |
| band_20_32|group|1e-7#92 | returned | 10 | Q, fallback cc47 | 1.82e-42 | 8.17e-43 / 1.26e-42 | 2.94e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.53e-42 | 4.26e-42 |  |
| band_20_32|group|1e-7#92 | returned | 10 | Q, fallback cc93 | 1.6e-42 | 6.61e-43 / 1.26e-42 | 1.38e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.46e-42 | 4.26e-42 |  |
| band_20_32|group|1e-8#16 | returned | 10 | Q, fallback cc93 | 4.48e-44 | 2.32e-45 / 4.57e-44 | 2.73e-45 / 2.71e-45 (noise) | bounded | — | bounded | 4.21e-44 | 1.78e-43 |  |
| band_20_32|group|1e-8#48 | returned | 10 | Q, fallback cc47 | 2.49e-42 | 4.09e-42 / 6.34e-42 | 1.78e-43 / 1.11e-42 (noise) | bounded | — | bounded | 2.32e-42 | 3.86e-41 |  |
| band_20_32|group|1e-8#48 | returned | 10 | Q, fallback cc93 | 2.33e-42 | 4.24e-42 / 6.34e-42 | 3.33e-43 / 1.11e-42 (noise) | bounded | — | bounded | 1.99e-42 | 3.86e-41 |  |
| band_20_32|group|1e-8#50 | returned | 10 | Q, fallback cc47 | 1.82e-42 | 8.17e-43 / 1.26e-42 | 2.94e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.53e-42 | 4.26e-42 |  |
| band_20_32|group|1e-8#50 | returned | 10 | Q, fallback cc93 | 1.6e-42 | 6.61e-43 / 1.26e-42 | 1.38e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.46e-42 | 4.26e-42 |  |
| band_20_32|group|1e-8#57 | returned | 10 | Q, fallback cc47 | 5.55e-38 | 1.31e-39 / 6.1e-38 | 1.2e-39 / 3.14e-39 (noise) | bounded | — | bounded | 5.43e-38 | 1.42e-37 |  |
| band_20_32|group|1e-8#57 | returned | 10 | Q, fallback cc93 | 5.59e-38 | 3.03e-39 / 6.1e-38 | 5.23e-40 / 3.14e-39 (noise) | bounded | — | bounded | 5.53e-38 | 1.42e-37 |  |
| band_20_32|group|1e-8#67 | returned | 10 | Q, fallback cc93 | 3.21e-45 | 5.03e-46 / 6.59e-45 | 5.88e-46 / 3.04e-46 (noise) | bounded | — | bounded | 2.62e-45 | 1.58e-44 |  |
| band_20_32|group|1e-8#69 | returned | 10 | Q, fallback cc93 | 1.45e-41 | 1.21e-42 / 2.03e-41 | 2.98e-42 / 7.94e-43 (noise) | bounded | — | bounded | 1.15e-41 | 4.63e-41 |  |
| band_20_32|group|1e-8#94 | returned | 10 | Q, fallback cc47 | 4.09e-45 | 7.73e-46 / 4.4e-45 | 6.2e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.47e-45 | 1.42e-44 |  |
| band_20_32|group|1e-8#94 | returned | 10 | Q, fallback cc93 | 3.72e-45 | 4.92e-46 / 4.4e-45 | 3.39e-46 / 3.48e-46 (noise) | bounded | — | bounded | 3.38e-45 | 1.42e-44 |  |
| gate_q2|group|1e-7#19 | returned | 12 | CC_93, pair93 | 3.12e-42 | 2.35e-42 / 1.2e-42 | 1.65e-42 / 5.54e-44 (noise) | bounded | — | bounded | 1.46e-42 | 1.53e-42 |  |
| gate_q2|group|1e-7#19 | returned | 12 | CC_93, tail93 | 3.12e-42 | 2.35e-42 / 1.2e-42 | 1.65e-42 / 5.54e-44 (noise) | bounded | — | bounded | 1.46e-42 | 1.53e-42 |  |
| gate_q2|group|1e-8#122 | returned | 12 | CC_93, pair93 | 2.62e-42 | 1.91e-42 / 1.07e-42 | 1.31e-42 / 5.1e-44 (noise) | bounded | — | bounded | 1.31e-42 | 1.44e-42 |  |
| gate_q2|group|1e-8#122 | returned | 12 | CC_93, tail93 | 2.62e-42 | 1.91e-42 / 1.07e-42 | 1.31e-42 / 5.1e-44 (noise) | bounded | — | bounded | 1.31e-42 | 1.44e-42 |  |
| gate_q2|group|1e-8#76 | returned | 12 | CC_93, pair93 | 2.36e-42 | 1.66e-42 / 1.01e-42 | 1.06e-42 / 5.08e-44 (noise) | bounded | — | bounded | 1.3e-42 | 1.41e-42 |  |
| gate_q2|group|1e-8#76 | returned | 12 | CC_93, tail93 | 2.38e-42 | 1.66e-42 / 1.01e-42 | 1.06e-42 / 5.08e-44 (noise) | bounded | — | bounded | 1.32e-42 | 1.41e-42 |  |
| band_12_16|analytic|1e-7#45 | returned | 12 | Q, fallback cc47 | 9.3e-25 | 7.06e-25 / 4.09e-25 | 4.69e-26 / 4.6e-26 (noise) | bounded | — | bounded | 8.84e-25 | 1.09e-24 |  |
| band_12_16|analytic|1e-8#30 | returned | 12 | Q, fallback cc47 | 9.3e-25 | 7.06e-25 / 4.09e-25 | 4.69e-26 / 4.6e-26 (noise) | bounded | — | bounded | 8.84e-25 | 1.09e-24 |  |
| band_12_16|group|1e-7#34 | returned | 12 | Q, fallback cc47 | 2.13e-42 | 3.35e-42 / 2.36e-42 | 3.11e-43 / 1.52e-43 (noise) | bounded | — | bounded | 1.82e-42 | 6.77e-42 |  |
| band_12_16|group|1e-7#34 | returned | 12 | Q, fallback cc93 | 2.02e-42 | 3.34e-42 / 2.36e-42 | 3.07e-43 / 1.52e-43 (noise) | bounded | — | bounded | 1.72e-42 | 6.77e-42 |  |
| band_12_16|group|1e-7#58 | returned | 12 | Q, fallback cc47 | 1.1e-40 | 1.81e-41 / 1.37e-40 | 3.23e-41 / 6.18e-42 (noise) | bounded | — | bounded | 7.73e-41 | 2.91e-40 |  |
| band_16_20|analytic|1e-7#43 | returned | 12 | Q, fallback cc47 | 1.82e-27 | 2.54e-27 / 1.5e-27 | 7.04e-28 / 6.06e-29 (noise) | bounded | — | bounded | 1.11e-27 | 2.81e-27 |  |
| band_16_20|analytic|1e-8#27 | returned | 12 | Q, fallback cc93 | 1.14e-26 | 4.18e-27 / 9.52e-27 | 6.31e-29 / 8.66e-28 (noise) | bounded | — | bounded | 1.13e-26 | 1.56e-26 |  |
| band_16_20|group|1e-8#15 | returned | 12 | Q, fallback cc93 | 5.32e-43 | 6.98e-44 / 7.68e-43 | 4.69e-44 / 9.23e-44 (noise) | bounded | — | bounded | 4.86e-43 | 3.74e-42 |  |
| band_16_20|group|1e-8#32 | returned | 12 | Q, fallback cc47 | 4.94e-42 | 5.18e-42 / 3.93e-42 | 4.9e-42 / 2.38e-43 (noise) | undecided | 4.79e-42 / 3.1e-43 | undecided | 1.5e-43 | 1.09e-41 | yes |
| band_16_20|group|1e-8#32 | returned | 12 | Q, fallback cc93 | 4.03e-42 | 4.25e-42 / 3.93e-42 | 3.97e-42 / 2.38e-43 (noise) | undecided | 3.86e-42 / 3.1e-43 | undecided | 1.69e-43 | 1.09e-41 | yes |
| band_16_20|group|1e-8#59 | returned | 12 | Q, fallback cc93 | 1.2e-40 | 6.15e-42 / 1.29e-40 | 8.31e-42 / 7.21e-42 (noise) | bounded | — | bounded | 1.12e-40 | 3.08e-40 |  |
| band_16_20|group|1e-8#60 | returned | 12 | Q, fallback cc47 | 5.96e-40 | 7.45e-40 / 1.55e-39 | 4.95e-40 / 2.77e-40 (noise) | undecided | 4.24e-40 / 1.86e-40 | undecided | 1.72e-40 | 1.09e-38 | yes |
| band_16_20|group|1e-8#71 | returned | 12 | Q, fallback cc93 | 2e-44 | 1.59e-44 / 4.83e-44 | 1.55e-44 / 3.3e-45 (noise) | bounded | — | bounded | 4.53e-45 | 1.05e-43 |  |
| band_20_32|analytic|1e-7#0 | returned | 12 | Q, fallback cc93 | 3.2e-24 | 1.73e-24 / 3.57e-24 | 2.46e-25 / 2.86e-25 (noise) | bounded | — | bounded | 2.96e-24 | 7.49e-24 |  |
| band_20_32|analytic|1e-7#10 | returned | 12 | Q, fallback cc47 | 7.32e-27 | 5.92e-27 / 7.41e-27 | 1.13e-27 / 8.92e-28 (noise) | bounded | — | bounded | 6.19e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-7#10 | returned | 12 | Q, fallback cc93 | 6.16e-27 | 5.43e-27 / 7.41e-27 | 6.44e-28 / 8.92e-28 (noise) | bounded | — | bounded | 5.52e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-7#17 | returned | 12 | Q, fallback cc93 | 1.86e-24 | 2.06e-24 / 2e-24 | 5.01e-25 / 1.26e-25 (noise) | bounded | — | bounded | 1.36e-24 | 3.44e-24 |  |
| band_20_32|analytic|1e-7#23 | returned | 12 | Q, fallback cc47 | 1.21e-27 | 3.53e-27 / 2.65e-27 | 4.29e-28 / 1.85e-28 (noise) | bounded | — | bounded | 7.84e-28 | 4.31e-27 |  |
| band_20_32|analytic|1e-7#31 | returned | 12 | Q, fallback cc47 | 3.77e-27 | 1.63e-27 / 6.29e-27 | 2.42e-27 / 2.29e-28 (noise) | bounded | — | bounded | 1.35e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-7#31 | returned | 12 | Q, fallback cc93 | 3.24e-27 | 2.7e-27 / 6.29e-27 | 1.34e-27 / 2.29e-28 (noise) | bounded | — | bounded | 1.89e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-7#39 | returned | 12 | Q, fallback cc47 | 7.93e-27 | 2.02e-27 / 7.26e-27 | 2.63e-27 / 1.11e-27 (noise) | bounded | — | bounded | 5.31e-27 | 2.46e-26 |  |
| band_20_32|analytic|1e-7#52 | returned | 12 | Q, fallback cc47 | 1.22e-27 | 8.28e-28 / 2.39e-27 | 8.01e-28 / 1.47e-28 (noise) | bounded | — | bounded | 4.18e-28 | 4.73e-27 |  |
| band_20_32|analytic|1e-7#54 | returned | 12 | Q, fallback cc47 | 7.32e-27 | 5.92e-27 / 7.41e-27 | 1.13e-27 / 8.92e-28 (noise) | bounded | — | bounded | 6.19e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-7#54 | returned | 12 | Q, fallback cc93 | 6.16e-27 | 5.43e-27 / 7.41e-27 | 6.44e-28 / 8.92e-28 (noise) | bounded | — | bounded | 5.52e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-8#11 | returned | 12 | Q, fallback cc47 | 7.66e-26 | 8.07e-26 / 1.14e-25 | 2.52e-26 / 8.63e-27 (noise) | bounded | — | bounded | 5.13e-26 | 2.07e-25 |  |
| band_20_32|analytic|1e-8#11 | returned | 12 | Q, fallback cc93 | 1.13e-25 | 6.76e-26 / 1.14e-25 | 1.21e-26 / 8.63e-27 (noise) | bounded | — | bounded | 1.01e-25 | 2.07e-25 |  |
| band_20_32|analytic|1e-8#16 | returned | 12 | Q, fallback cc47 | 3.24e-27 | 8.46e-28 / 3.77e-27 | 1.84e-27 / 3.38e-28 (noise) | bounded | — | bounded | 1.39e-27 | 7.38e-27 |  |
| band_20_32|analytic|1e-8#23 | returned | 12 | Q, fallback cc47 | 7.93e-27 | 2.02e-27 / 7.26e-27 | 2.63e-27 / 1.11e-27 (noise) | bounded | — | bounded | 5.31e-27 | 2.46e-26 |  |
| band_20_32|analytic|1e-8#35 | returned | 12 | Q, fallback cc47 | 3.05e-27 | 3.49e-28 / 3.42e-27 | 1.89e-27 / 2.9e-28 (noise) | bounded | 1.9e-27 / 2.11e-29 | bounded | 1.15e-27 | 7.66e-27 |  |
| band_20_32|analytic|1e-8#36 | returned | 12 | Q, fallback cc47 | 1.21e-27 | 3.53e-27 / 2.65e-27 | 4.29e-28 / 1.85e-28 (noise) | bounded | — | bounded | 7.84e-28 | 4.31e-27 |  |
| band_20_32|analytic|1e-8#38 | returned | 12 | Q, fallback cc93 | 1.86e-24 | 2.06e-24 / 2e-24 | 5.01e-25 / 1.26e-25 (noise) | bounded | — | bounded | 1.36e-24 | 3.44e-24 |  |
| band_20_32|analytic|1e-8#49 | returned | 12 | Q, fallback cc47 | 3.85e-24 | 1.8e-24 / 3.32e-24 | 1.6e-24 / 3.36e-25 (noise) | bounded | — | bounded | 2.26e-24 | 7.64e-24 |  |
| band_20_32|analytic|1e-8#60 | returned | 12 | Q, fallback cc93 | 3.2e-24 | 1.73e-24 / 3.57e-24 | 2.46e-25 / 2.86e-25 (noise) | bounded | — | bounded | 2.96e-24 | 7.49e-24 |  |
| band_20_32|analytic|1e-8#78 | returned | 12 | Q, fallback cc47 | 7.32e-27 | 5.92e-27 / 7.41e-27 | 1.13e-27 / 8.92e-28 (noise) | bounded | — | bounded | 6.19e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-8#78 | returned | 12 | Q, fallback cc93 | 6.16e-27 | 5.43e-27 / 7.41e-27 | 6.44e-28 / 8.92e-28 (noise) | bounded | — | bounded | 5.52e-27 | 2.28e-26 |  |
| band_20_32|analytic|1e-8#84 | returned | 12 | Q, fallback cc47 | 3.77e-27 | 1.63e-27 / 6.29e-27 | 2.42e-27 / 2.29e-28 (noise) | bounded | — | bounded | 1.35e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-8#84 | returned | 12 | Q, fallback cc93 | 3.24e-27 | 2.7e-27 / 6.29e-27 | 1.34e-27 / 2.29e-28 (noise) | bounded | — | bounded | 1.89e-27 | 7.23e-27 |  |
| band_20_32|analytic|1e-8#86 | returned | 12 | Q, fallback cc93 | 6.94e-27 | 2.99e-27 / 8.53e-27 | 9.97e-28 / 9.74e-28 (noise) | bounded | — | bounded | 5.95e-27 | 2.16e-26 |  |
| band_20_32|group|1e-7#27 | returned | 12 | Q, fallback cc93 | 2.01e-39 | 1.13e-39 / 1.38e-39 | 4.23e-40 / 7.31e-41 (noise) | bounded | — | bounded | 1.58e-39 | 2.84e-39 |  |
| band_20_32|group|1e-7#69 | returned | 12 | Q, fallback cc47 | 1.5e-39 | 5.59e-40 / 1.48e-39 | 3.61e-40 / 6.99e-41 (noise) | bounded | — | bounded | 1.14e-39 | 3.11e-39 |  |
| band_20_32|group|1e-7#92 | returned | 12 | Q, fallback cc47 | 1.75e-42 | 8.17e-43 / 1.26e-42 | 2.94e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.45e-42 | 4.26e-42 |  |
| band_20_32|group|1e-8#16 | returned | 12 | Q, fallback cc93 | 4.48e-44 | 2.32e-45 / 4.57e-44 | 2.73e-45 / 2.71e-45 (noise) | bounded | — | bounded | 4.21e-44 | 1.78e-43 |  |
| band_20_32|group|1e-8#50 | returned | 12 | Q, fallback cc47 | 1.75e-42 | 8.17e-43 / 1.26e-42 | 2.94e-43 / 7.87e-44 (noise) | bounded | — | bounded | 1.45e-42 | 4.26e-42 |  |
| band_20_32|group|1e-8#57 | returned | 12 | Q, fallback cc47 | 5.55e-38 | 1.31e-39 / 6.1e-38 | 1.2e-39 / 3.14e-39 (noise) | bounded | — | bounded | 5.43e-38 | 1.42e-37 |  |
| band_20_32|group|1e-8#57 | returned | 12 | Q, fallback cc93 | 5.59e-38 | 3.03e-39 / 6.1e-38 | 5.23e-40 / 3.14e-39 (noise) | bounded | — | bounded | 5.53e-38 | 1.42e-37 |  |
| band_20_32|group|1e-8#67 | returned | 12 | Q, fallback cc93 | 3.21e-45 | 5.03e-46 / 6.59e-45 | 5.88e-46 / 3.04e-46 (noise) | bounded | — | bounded | 2.62e-45 | 1.58e-44 |  |
| band_20_32|group|1e-8#69 | returned | 12 | Q, fallback cc93 | 1.45e-41 | 1.21e-42 / 2.03e-41 | 2.98e-42 / 7.94e-43 (noise) | bounded | — | bounded | 1.15e-41 | 4.63e-41 |  |
| band_20_32v|group|1e-8#16 | visited | 6 | Q, fallback cc93 | 9.53e-40 | 1.63e-39 / 4.03e-39 | 6.27e-41 / 2.45e-40 (noise) | bounded | — | bounded | 8.91e-40 | 1.26e-38 |  |
| band_20_32v|group|1e-7#4 | visited | 8 | Q, fallback cc47 | 1.19e-39 | 2.42e-39 / 4.19e-39 | 5.84e-40 / 3.77e-40 (noise) | bounded | — | bounded | 6.06e-40 | 1.75e-38 |  |
| band_20_32v|group|1e-8#16 | visited | 8 | Q, fallback cc93 | 9.53e-40 | 1.63e-39 / 4.03e-39 | 6.27e-41 / 2.45e-40 (noise) | bounded | — | bounded | 8.91e-40 | 1.26e-38 |  |
| band_20_32v|group|1e-8#4 | visited | 8 | Q, fallback cc47 | 7.27e-40 | 7.27e-40 / 3.9e-40 | 3.34e-40 / 6.23e-41 (noise) | bounded | — | bounded | 3.93e-40 | 2.01e-39 |  |
| band_20_32v|group|1e-8#4 | visited | 8 | Q, fallback cc93 | 5.96e-40 | 5.83e-40 / 3.9e-40 | 1.91e-40 / 6.23e-41 (noise) | bounded | — | bounded | 4.05e-40 | 2.01e-39 |  |
| band_20_32v|group|1e-7#4 | visited | 10 | Q, fallback cc47 | 1.19e-39 | 2.42e-39 / 4.19e-39 | 5.84e-40 / 3.77e-40 (noise) | bounded | — | bounded | 6.06e-40 | 1.75e-38 |  |
| band_20_32v|group|1e-8#16 | visited | 10 | Q, fallback cc93 | 9.53e-40 | 1.63e-39 / 4.03e-39 | 6.27e-41 / 2.45e-40 (noise) | bounded | — | bounded | 8.91e-40 | 1.26e-38 |  |
| band_20_32v|group|1e-8#4 | visited | 10 | Q, fallback cc47 | 7.27e-40 | 7.27e-40 / 3.9e-40 | 3.34e-40 / 6.23e-41 (noise) | bounded | — | bounded | 3.93e-40 | 2.01e-39 |  |
| band_20_32v|group|1e-8#4 | visited | 10 | Q, fallback cc93 | 5.96e-40 | 5.83e-40 / 3.9e-40 | 1.91e-40 / 6.23e-41 (noise) | bounded | — | bounded | 4.05e-40 | 2.01e-39 |  |
| band_20_32v|group|1e-7#4 | visited | 12 | Q, fallback cc47 | 1.19e-39 | 2.42e-39 / 4.19e-39 | 5.84e-40 / 3.77e-40 (noise) | bounded | — | bounded | 6.06e-40 | 1.75e-38 |  |
| band_20_32v|group|1e-7#43 | visited | 12 | Q, fallback cc93 | 1.49e-42 | 1.24e-42 / 3.69e-42 | 1.97e-42 / 3.14e-43 (noise) | understates | — | understates | 4.8e-43 | 1.3e-41 |  |
| band_20_32v|group|1e-7#46 | visited | 12 | Q, fallback cc47 | 1.1e-42 | 1.28e-42 / 8.5e-43 | 9.71e-43 / 6.96e-44 (noise) | bounded | — | bounded | 1.31e-43 | 2.43e-42 |  |
| band_20_32v|group|1e-7#62 | visited | 12 | Q, fallback cc47 | 4.73e-43 | 1.56e-43 / 4.29e-43 | 2.8e-43 / 2.42e-44 (noise) | bounded | — | bounded | 1.93e-43 | 1.21e-42 |  |
| band_20_32v|group|1e-7#65 | visited | 12 | Q, fallback cc47 | 3.31e-36 | 3.38e-36 / 1.25e-36 | 3.08e-36 / 3.78e-37 (noise) | undecided | 3.12e-36 / 1.62e-37 | bounded | 1.89e-37 | 1.35e-35 |  |
| band_20_32v|group|1e-7#76 | visited | 12 | Q, fallback cc47 | 4.93e-38 | 3e-38 / 2.73e-38 | 2.59e-38 / 2.23e-39 (noise) | bounded | — | bounded | 2.34e-38 | 1.22e-37 |  |
| band_20_32v|group|1e-8#1 | visited | 12 | Q, fallback cc93 | 6.55e-45 | 4.02e-45 / 6.2e-45 | 1.9e-45 / 3.31e-46 (noise) | bounded | — | bounded | 4.66e-45 | 1.5e-44 |  |
| band_20_32v|group|1e-8#13 | visited | 12 | Q, fallback cc93 | 9.15e-37 | 3.24e-37 / 7.98e-37 | 2.78e-37 / 6.38e-38 (noise) | bounded | — | bounded | 6.37e-37 | 3.63e-36 |  |
| band_20_32v|group|1e-8#4 | visited | 12 | Q, fallback cc47 | 7.27e-40 | 7.27e-40 / 3.9e-40 | 3.34e-40 / 6.23e-41 (noise) | bounded | — | bounded | 3.93e-40 | 2.01e-39 |  |
| band_20_32v|group|1e-8#4 | visited | 12 | Q, fallback cc93 | 5.96e-40 | 5.83e-40 / 3.9e-40 | 1.91e-40 / 6.23e-41 (noise) | bounded | — | bounded | 4.05e-40 | 2.01e-39 |  |
| band_20_32v|group|1e-8#55 | visited | 12 | Q, fallback cc47 | 3.31e-36 | 3.38e-36 / 1.25e-36 | 3.08e-36 / 3.78e-37 (noise) | undecided | 3.12e-36 / 1.62e-37 | bounded | 1.89e-37 | 1.35e-35 |  |
| band_20_32v|group|1e-8#66 | visited | 12 | Q, fallback cc47 | 5.74e-44 | 3.79e-44 / 7.81e-44 | 1.06e-44 / 9.24e-45 (noise) | bounded | — | bounded | 4.69e-44 | 4.3e-43 |  |
| band_20_32v|group|1e-8#82 | visited | 12 | Q, fallback cc47 | 4.73e-43 | 1.56e-43 / 4.29e-43 | 2.8e-43 / 2.42e-44 (noise) | bounded | — | bounded | 1.93e-43 | 1.21e-42 |  |
| band_20_32v|group|1e-8#83 | visited | 12 | Q, fallback cc47 | 4.07e-39 | 2.71e-39 / 3.42e-39 | 2.52e-39 / 2.91e-40 (noise) | bounded | — | bounded | 1.55e-39 | 1.27e-38 |  |
| band_20_32v|group|1e-8#86 | visited | 12 | Q, fallback cc47 | 1.73e-40 | 1.54e-40 / 1.86e-40 | 1.56e-40 / 1.34e-41 (noise) | bounded | — | bounded | 1.7e-41 | 4.41e-40 |  |
| band_20_32v|group|1e-8#88 | visited | 12 | Q, fallback cc47 | 1.47e-41 | 9.23e-42 / 9.35e-42 | 1.31e-41 / 2.4e-42 (noise) | undecided | 1.43e-41 / 5.59e-43 | undecided | 3.36e-43 | 4.54e-41 | yes |
