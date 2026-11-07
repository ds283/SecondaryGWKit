# 07 section 2.2 tables

    levin-phase-floor 07: section 2.2, Phi_w and Lambda_c (run)
    written 2026-11-07T09:12:41+00:00 on mps026088.phys.susx.ac.uk
    python 3.13.16 (/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python)
    script calibrate.py sha256 4f4290c5d7de782658dcc4cab48c80cd734f2f6bcdbfa36960d1ca478e3b72d9
    prov.py sha256 9e2adc86dd624417822eb68ef554a6e5c23c817a824a8f778df712055d449c76
    AdaptiveLevin/levin_quadrature.py sha256 13294b2ac3e3e3a480a6735a263ea8abe205ccdaada12939c13cd20edad884a4
    input docs/levin-phase-floor/07/calibration/scripts/reference.py sha256 43283037c7bad69b7624e05a6084067cf1e0f4d34da4a4a7c5e02cbe097e0a26
    input docs/levin-phase-floor/07/calibration/scripts/analysis07.py sha256 1f2999bfb671bd564c3b4669214ebad490cc1579c06fcc09e996d4343f850239
    HEAD bd21bf23da006aa21c5aa670cf32ea2ed726cb6b
    tree dirty: yes; git status --short:
         M AdaptiveLevin/levin_quadrature.py
         M ComputeTargets/QuadSourceIntegral.py
        ?? docs/levin-phase-floor/07/
    uptime at start: 9:12  up 7 days, 11:33, 1 user, load averages: 9.96 14.43 15.79

## Phi_w on family (R)

(R)'s two references, by erf and by contour: largest relative disagreement 2.19e-37.

| Phi_w | case | true error | abserr | err/abserr | abserr/request | honest | converged | node_limited | regions | window pieces | max depth | theta'^2/theta'' at edges | point - (x_s - omega/c) | point - x_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2pi | R c=1e+04 x_s=0.3820 omega=0 | 6.94e-18 | 9.59e-13 | 7.2e-06 | 0.496 | True | True | False | 20 | 5 | 7 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+04 x_s=0.3820 omega=1 | 4.16e-17 | 1.63e-13 | 0.00026 | 0.083 | True | True | False | 18 | 9 | 4 | 12.57, 12.57 | 0.0e+00 | -1.0e-04 |
| 2pi | R c=1e+04 x_s=0.7236 omega=0 | 4.81e-17 | 3.78e-14 | 0.0013 | 0.325 | True | True | False | 21 | 4 | 6 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+06 x_s=0.3820 omega=0 | 5.14e-17 | 4.01e-14 | 0.0013 | 0.206 | True | True | False | 34 | 6 | 11 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+06 x_s=0.3820 omega=1 | 8.54e-17 | 1.74e-14 | 0.0049 | 0.088 | True | True | False | 24 | 9 | 7 | 12.57, 12.57 | 0.0e+00 | -1.0e-06 |
| 2pi | R c=1e+06 x_s=0.7236 omega=0 | 4.27e-17 | 5.85e-15 | 0.0073 | 0.409 | True | True | False | 27 | 8 | 10 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+08 x_s=0.3820 omega=0 | 1.66e-16 | 5.84e-15 | 0.028 | 0.301 | True | True | False | 59 | 6 | 15 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+08 x_s=0.3820 omega=1 | 9.06e-17 | 1.54e-14 | 0.0059 | 0.78 | True | True | False | 55 | 6 | 15 | 12.57, 12.57 | 0.0e+00 | -1.0e-08 |
| 2pi | R c=1e+08 x_s=0.7236 omega=0 | 3.61e-16 | 1.75e-15 | 0.21 | 1.24 | True | False | True | 290 | 66 | 20 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+08 x_s=0.7236 omega=1 | 4.27e-18 | 7.31e-15 | 0.00058 | 0.649 | True | True | False | 69 | 10 | 15 | 12.57, 12.57 | 0.0e+00 | -1.0e-08 |
| 8pi | R c=1e+04 x_s=0.3820 omega=0 | 5.90e-17 | 1.18e-13 | 0.0005 | 0.0612 | True | True | False | 40 | 34 | 5 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+04 x_s=0.3820 omega=1 | 2.78e-17 | 6.12e-13 | 4.5e-05 | 0.312 | True | True | False | 70 | 64 | 9 | 50.27, 50.27 | 0.0e+00 | -1.0e-04 |
| 8pi | R c=1e+04 x_s=0.7236 omega=0 | 2.01e-16 | 9.27e-14 | 0.0022 | 0.797 | True | True | False | 18 | 12 | 3 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+06 x_s=0.3820 omega=0 | 3.45e-17 | 3.95e-14 | 0.00087 | 0.203 | True | True | False | 31 | 18 | 6 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+06 x_s=0.3820 omega=1 | 3.12e-17 | 2.11e-14 | 0.0015 | 0.107 | True | True | False | 36 | 23 | 6 | 50.27, 50.27 | 0.0e+00 | -1.0e-06 |
| 8pi | R c=1e+06 x_s=0.7236 omega=0 | 2.55e-16 | 7.74e-15 | 0.033 | 0.54 | True | True | False | 45 | 16 | 11 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+08 x_s=0.3820 omega=0 | 6.84e-17 | 1.33e-14 | 0.0051 | 0.685 | True | True | False | 37 | 18 | 9 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+08 x_s=0.3820 omega=1 | 4.73e-17 | 2.13e-15 | 0.022 | 0.107 | True | True | False | 38 | 18 | 9 | 50.27, 50.27 | 0.0e+00 | -1.0e-08 |
| 8pi | R c=1e+08 x_s=0.7236 omega=0 | 2.27e-16 | 1.69e-15 | 0.13 | 1.2 | True | False | True | 467 | 313 | 20 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+08 x_s=0.7236 omega=1 | 2.83e-16 | 1.12e-14 | 0.025 | 0.998 | True | True | False | 75 | 16 | 16 | 50.27, 50.27 | 0.0e+00 | -1.0e-08 |
| 32pi | R c=1e+04 x_s=0.3820 omega=0 | 1.04e-17 | 5.96e-13 | 1.7e-05 | 0.308 | True | True | False | 111 | 109 | 7 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+04 x_s=0.3820 omega=1 | 9.71e-17 | 2.77e-14 | 0.0035 | 0.0141 | True | True | False | 122 | 119 | 11 | 201.1, 201.1 | 0.0e+00 | -1.0e-04 |
| 32pi | R c=1e+04 x_s=0.7236 omega=0 | 3.25e-16 | 4.30e-14 | 0.0076 | 0.37 | True | True | False | 92 | 88 | 6 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+06 x_s=0.3820 omega=0 | 2.61e-16 | 1.35e-13 | 0.0019 | 0.694 | True | True | False | 74 | 65 | 6 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+06 x_s=0.3820 omega=1 | 1.32e-16 | 3.70e-14 | 0.0036 | 0.187 | True | True | False | 78 | 68 | 5 | 201.1, 201.1 | 0.0e+00 | -1.0e-06 |
| 32pi | R c=1e+06 x_s=0.7236 omega=0 | 1.22e-16 | 1.23e-14 | 0.0099 | 0.859 | True | True | False | 91 | 80 | 5 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+08 x_s=0.3820 omega=0 | 1.06e-16 | 1.20e-14 | 0.0088 | 0.616 | True | True | False | 96 | 80 | 7 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+08 x_s=0.3820 omega=1 | 1.78e-16 | 5.31e-15 | 0.033 | 0.268 | True | True | False | 79 | 62 | 8 | 201.1, 201.1 | 0.0e+00 | -1.0e-08 |
| 32pi | R c=1e+08 x_s=0.7236 omega=0 | 4.90e-18 | 1.43e-15 | 0.0034 | 1.02 | True | False | True | 168821 | 168780 | 20 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+08 x_s=0.7236 omega=1 | 8.88e-17 | 1.04e-14 | 0.0085 | 0.927 | True | True | False | 135 | 117 | 9 | 201.1, 201.1 | 0.0e+00 | -1.0e-08 |
| 128pi | R c=1e+04 x_s=0.3820 omega=0 | 8.67e-17 | 1.24e-13 | 0.0007 | 0.0643 | True | True | False | 348 | 346 | 9 | 804.2, 804.2 | 0.0e+00 | 0.0e+00 |
| 128pi | R c=1e+04 x_s=0.3820 omega=1 | 1.60e-16 | 1.34e-13 | 0.0012 | 0.0684 | True | True | False | 335 | 333 | 10 | 804.2, 804.2 | 0.0e+00 | -1.0e-04 |
| 128pi | R c=1e+04 x_s=0.7236 omega=0 | 7.30e-16 | 9.94e-14 | 0.0073 | 0.856 | True | True | False | 272 | 271 | 7 | 804.2, 763.9 | 0.0e+00 | 0.0e+00 |
| 128pi | R c=1e+06 x_s=0.3820 omega=0 | 9.32e-17 | 7.58e-14 | 0.0012 | 0.39 | True | True | False | 330 | 323 | 6 | 804.2, 804.2 | 0.0e+00 | 0.0e+00 |
| 128pi | R c=1e+06 x_s=0.3820 omega=1 | 2.19e-16 | 1.24e-13 | 0.0018 | 0.626 | True | True | False | 271 | 264 | 6 | 804.2, 804.2 | 0.0e+00 | -1.0e-06 |
| 128pi | R c=1e+06 x_s=0.7236 omega=0 | 7.97e-18 | 1.44e-14 | 0.00055 | 1 | True | False | True | 15717 | 15706 | 17 | 804.2, 804.2 | 0.0e+00 | 0.0e+00 |
| 128pi | R c=1e+08 x_s=0.3820 omega=0 | 1.05e-16 | 1.93e-14 | 0.0054 | 0.995 | True | True | False | 345 | 330 | 7 | 804.2, 804.2 | 0.0e+00 | 0.0e+00 |
| 128pi | R c=1e+08 x_s=0.3820 omega=1 | 2.84e-17 | 1.95e-14 | 0.0015 | 0.986 | True | True | False | 299 | 284 | 7 | 804.2, 804.2 | 0.0e+00 | -1.0e-08 |
| 128pi | R c=1e+08 x_s=0.7236 omega=0 | did not finish (3600 s) | | | | | | | | | | | |
| 128pi | R c=1e+08 x_s=0.7236 omega=1 | 1.21e-17 | 1.13e-14 | 0.0011 | 1 | True | False | True | 30651 | 30634 | 20 | 804.2, 804.2 | 0.0e+00 | -1.0e-08 |
| 512pi | R c=1e+04 x_s=0.3820 omega=0 | 5.76e-16 | 2.22e-13 | 0.0026 | 0.115 | True | True | False | 964 | 963 | 9 | 1459, 3217 | 0.0e+00 | 0.0e+00 |
| 512pi | R c=1e+04 x_s=0.3820 omega=1 | 4.51e-16 | 2.37e-13 | 0.0019 | 0.121 | True | True | False | 895 | 894 | 8 | 1458, 3217 | 0.0e+00 | -1.0e-04 |
| 512pi | R c=1e+04 x_s=0.7236 omega=0 | 8.74e-16 | 1.16e-13 | 0.0075 | 0.998 | True | True | False | 777 | 776 | 11 | 3217, 763.9 | 0.0e+00 | 0.0e+00 |
| 512pi | R c=1e+06 x_s=0.3820 omega=0 | 3.14e-16 | 1.40e-13 | 0.0022 | 0.722 | True | True | False | 1113 | 1108 | 8 | 3217, 3217 | 0.0e+00 | 0.0e+00 |
| 512pi | R c=1e+06 x_s=0.3820 omega=1 | 2.37e-16 | 1.05e-13 | 0.0023 | 0.53 | True | True | False | 963 | 958 | 13 | 3217, 3217 | 0.0e+00 | -1.0e-06 |
| 512pi | R c=1e+06 x_s=0.7236 omega=0 | did not finish (3601 s) | | | | | | | | | | | |
| 512pi | R c=1e+08 x_s=0.3820 omega=0 | 3.58e-17 | 1.94e-14 | 0.0018 | 1 | True | False | True | 247707 | 247692 | 20 | 3217, 3217 | 0.0e+00 | 0.0e+00 |
| 512pi | R c=1e+08 x_s=0.3820 omega=1 | 3.19e-17 | 1.98e-14 | 0.0016 | 1 | True | False | True | 213935 | 213920 | 20 | 3217, 3217 | 0.0e+00 | -1.0e-08 |
| 512pi | R c=1e+08 x_s=0.7236 omega=0 | did not finish (3601 s) | | | | | | | | | | | |
| 512pi | R c=1e+08 x_s=0.7236 omega=1 | did not finish (3601 s) | | | | | | | | | | | |

**Phi_w by the rule:** none

## Lambda_c on family (C) at Phi_w = 2pi -- SUPPLEMENTARY, not the rule (no Phi_w rung qualified)

No minimum located (Lambda_c = 0):

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima found, not located (lambda) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 1.04e-17 | 2.84e-13 | 3.7e-05 | True | True | False | 12 | 0 | 5 |  | 0.25 |
| C b=1e+04 lambda=1 | 1 | 7.29e-17 | 2.30e-13 | 0.00032 | True | True | False | 11 | 0 | 5 |  | 1 |
| C b=1e+04 lambda=4 | 4 | 3.86e-17 | 5.48e-14 | 0.0007 | True | True | False | 13 | 0 | 5 |  | 4 |
| C b=1e+04 lambda=16 | 16 | 1.21e-16 | 4.72e-15 | 0.026 | True | True | False | 17 | 0 | 6 |  | 16 |
| C b=1e+04 lambda=64 | 64 | 6.42e-17 | 2.40e-14 | 0.0027 | True | True | False | 5 | 0 | 3 |  | 64 |
| C b=1e+04 lambda=256 | 256 | 3.79e-19 | 7.83e-16 | 0.00048 | True | True | False | 4 | 0 | 2 |  | 256 |
| C b=1e+04 lambda=1024 | 1024 | 3.33e-17 | 1.82e-14 | 0.0018 | True | True | False | 2 | 0 | 1 |  | 1024 |
| C b=1e+08 lambda=0.25 | 0.25 | 5.85e-17 | 7.49e-14 | 0.00078 | True | True | False | 20 | 0 | 10 |  | 0.25 |
| C b=1e+08 lambda=1 | 1 | 1.90e-17 | 1.77e-14 | 0.0011 | True | True | False | 23 | 0 | 11 |  | 1 |
| C b=1e+08 lambda=4 | 4 | 2.47e-17 | 4.80e-15 | 0.0051 | True | True | False | 20 | 0 | 11 |  | 4 |
| C b=1e+08 lambda=16 | 16 | 2.67e-17 | 4.71e-16 | 0.057 | True | False | False | 58 | 0 | 12 |  | 16 |
| C b=1e+08 lambda=64 | 64 | 6.64e-18 | 4.40e-16 | 0.015 | True | False | False | 58 | 0 | 12 |  | 64 |
| C b=1e+08 lambda=256 | 256 | 6.53e-18 | 3.51e-16 | 0.019 | True | False | False | 15 | 0 | 7 |  | 256 |
| C b=1e+08 lambda=1024 | 1024 | 9.95e-18 | 3.42e-16 | 0.029 | True | False | False | 13 | 0 | 6 |  | 1024 |

**Lambda_c by the rule:** 4.0

The check, lambda < 4, minimum located:

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima not located |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 9.37e-17 | 1.64e-12 | 5.7e-05 | True | True | False | 23 | 18 | 6 | minimum 0.25 |  |
| C b=1e+04 lambda=1 | 1 | 1.73e-17 | 9.34e-13 | 1.9e-05 | True | True | False | 24 | 16 | 5 | minimum 1 |  |
| C b=1e+08 lambda=0.25 | 0.25 | 7.91e-17 | 1.00e-13 | 0.00079 | True | True | False | 30 | 9 | 12 | minimum 0.25 |  |
| C b=1e+08 lambda=1 | 1 | 5.55e-17 | 2.69e-14 | 0.0021 | True | True | False | 26 | 11 | 7 | minimum 1 |  |

**The check holds:** True

## Lambda_c on family (C) at Phi_w = 8pi -- SUPPLEMENTARY, not the rule (no Phi_w rung qualified)

No minimum located (Lambda_c = 0):

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima found, not located (lambda) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 1.04e-17 | 2.84e-13 | 3.7e-05 | True | True | False | 12 | 0 | 5 |  | 0.25 |
| C b=1e+04 lambda=1 | 1 | 7.29e-17 | 2.30e-13 | 0.00032 | True | True | False | 11 | 0 | 5 |  | 1 |
| C b=1e+04 lambda=4 | 4 | 3.86e-17 | 5.48e-14 | 0.0007 | True | True | False | 13 | 0 | 5 |  | 4 |
| C b=1e+04 lambda=16 | 16 | 1.21e-16 | 4.72e-15 | 0.026 | True | True | False | 17 | 0 | 6 |  | 16 |
| C b=1e+04 lambda=64 | 64 | 6.42e-17 | 2.40e-14 | 0.0027 | True | True | False | 5 | 0 | 3 |  | 64 |
| C b=1e+04 lambda=256 | 256 | 3.79e-19 | 7.83e-16 | 0.00048 | True | True | False | 4 | 0 | 2 |  | 256 |
| C b=1e+04 lambda=1024 | 1024 | 3.33e-17 | 1.82e-14 | 0.0018 | True | True | False | 2 | 0 | 1 |  | 1024 |
| C b=1e+08 lambda=0.25 | 0.25 | 5.85e-17 | 7.49e-14 | 0.00078 | True | True | False | 20 | 0 | 10 |  | 0.25 |
| C b=1e+08 lambda=1 | 1 | 1.90e-17 | 1.77e-14 | 0.0011 | True | True | False | 23 | 0 | 11 |  | 1 |
| C b=1e+08 lambda=4 | 4 | 2.47e-17 | 4.80e-15 | 0.0051 | True | True | False | 20 | 0 | 11 |  | 4 |
| C b=1e+08 lambda=16 | 16 | 2.67e-17 | 4.71e-16 | 0.057 | True | False | False | 58 | 0 | 12 |  | 16 |
| C b=1e+08 lambda=64 | 64 | 6.64e-18 | 4.40e-16 | 0.015 | True | False | False | 58 | 0 | 12 |  | 64 |
| C b=1e+08 lambda=256 | 256 | 6.53e-18 | 3.51e-16 | 0.019 | True | False | False | 15 | 0 | 7 |  | 256 |
| C b=1e+08 lambda=1024 | 1024 | 9.95e-18 | 3.42e-16 | 0.029 | True | False | False | 13 | 0 | 6 |  | 1024 |

**Lambda_c by the rule:** 4.0

The check, lambda < 4, minimum located:

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima not located |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 3.12e-17 | 1.67e-13 | 0.00019 | True | True | False | 38 | 35 | 6 | minimum 0.25 |  |
| C b=1e+04 lambda=1 | 1 | 5.03e-17 | 6.07e-13 | 8.3e-05 | True | True | False | 28 | 25 | 4 | minimum 1 |  |
| C b=1e+08 lambda=0.25 | 0.25 | 1.21e-17 | 2.82e-14 | 0.00043 | True | True | False | 38 | 25 | 6 | minimum 0.25 |  |
| C b=1e+08 lambda=1 | 1 | 2.02e-17 | 1.63e-14 | 0.0012 | True | True | False | 31 | 18 | 6 | minimum 1 |  |

**The check holds:** True

## Lambda_c on family (C) at Phi_w = 32pi -- SUPPLEMENTARY, not the rule (no Phi_w rung qualified)

No minimum located (Lambda_c = 0):

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima found, not located (lambda) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 1.04e-17 | 2.84e-13 | 3.7e-05 | True | True | False | 12 | 0 | 5 |  | 0.25 |
| C b=1e+04 lambda=1 | 1 | 7.29e-17 | 2.30e-13 | 0.00032 | True | True | False | 11 | 0 | 5 |  | 1 |
| C b=1e+04 lambda=4 | 4 | 3.86e-17 | 5.48e-14 | 0.0007 | True | True | False | 13 | 0 | 5 |  | 4 |
| C b=1e+04 lambda=16 | 16 | 1.21e-16 | 4.72e-15 | 0.026 | True | True | False | 17 | 0 | 6 |  | 16 |
| C b=1e+04 lambda=64 | 64 | 6.42e-17 | 2.40e-14 | 0.0027 | True | True | False | 5 | 0 | 3 |  | 64 |
| C b=1e+04 lambda=256 | 256 | 3.79e-19 | 7.83e-16 | 0.00048 | True | True | False | 4 | 0 | 2 |  | 256 |
| C b=1e+04 lambda=1024 | 1024 | 3.33e-17 | 1.82e-14 | 0.0018 | True | True | False | 2 | 0 | 1 |  | 1024 |
| C b=1e+08 lambda=0.25 | 0.25 | 5.85e-17 | 7.49e-14 | 0.00078 | True | True | False | 20 | 0 | 10 |  | 0.25 |
| C b=1e+08 lambda=1 | 1 | 1.90e-17 | 1.77e-14 | 0.0011 | True | True | False | 23 | 0 | 11 |  | 1 |
| C b=1e+08 lambda=4 | 4 | 2.47e-17 | 4.80e-15 | 0.0051 | True | True | False | 20 | 0 | 11 |  | 4 |
| C b=1e+08 lambda=16 | 16 | 2.67e-17 | 4.71e-16 | 0.057 | True | False | False | 58 | 0 | 12 |  | 16 |
| C b=1e+08 lambda=64 | 64 | 6.64e-18 | 4.40e-16 | 0.015 | True | False | False | 58 | 0 | 12 |  | 64 |
| C b=1e+08 lambda=256 | 256 | 6.53e-18 | 3.51e-16 | 0.019 | True | False | False | 15 | 0 | 7 |  | 256 |
| C b=1e+08 lambda=1024 | 1024 | 9.95e-18 | 3.42e-16 | 0.029 | True | False | False | 13 | 0 | 6 |  | 1024 |

**Lambda_c by the rule:** 4.0

The check, lambda < 4, minimum located:

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima not located |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 1.35e-16 | 9.96e-13 | 0.00014 | True | True | False | 70 | 68 | 4 | minimum 0.25 |  |
| C b=1e+04 lambda=1 | 1 | 3.82e-17 | 2.26e-13 | 0.00017 | True | True | False | 105 | 103 | 7 | minimum 1 |  |
| C b=1e+08 lambda=0.25 | 0.25 | 5.75e-17 | 5.54e-14 | 0.001 | True | True | False | 90 | 79 | 5 | minimum 0.25 |  |
| C b=1e+08 lambda=1 | 1 | 4.11e-17 | 6.06e-14 | 0.00068 | True | True | False | 91 | 80 | 5 | minimum 1 |  |

**The check holds:** True

## Lambda_c on family (C) at Phi_w = 128pi -- SUPPLEMENTARY, not the rule (no Phi_w rung qualified)

No minimum located (Lambda_c = 0):

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima found, not located (lambda) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 1.04e-17 | 2.84e-13 | 3.7e-05 | True | True | False | 12 | 0 | 5 |  | 0.25 |
| C b=1e+04 lambda=1 | 1 | 7.29e-17 | 2.30e-13 | 0.00032 | True | True | False | 11 | 0 | 5 |  | 1 |
| C b=1e+04 lambda=4 | 4 | 3.86e-17 | 5.48e-14 | 0.0007 | True | True | False | 13 | 0 | 5 |  | 4 |
| C b=1e+04 lambda=16 | 16 | 1.21e-16 | 4.72e-15 | 0.026 | True | True | False | 17 | 0 | 6 |  | 16 |
| C b=1e+04 lambda=64 | 64 | 6.42e-17 | 2.40e-14 | 0.0027 | True | True | False | 5 | 0 | 3 |  | 64 |
| C b=1e+04 lambda=256 | 256 | 3.79e-19 | 7.83e-16 | 0.00048 | True | True | False | 4 | 0 | 2 |  | 256 |
| C b=1e+04 lambda=1024 | 1024 | 3.33e-17 | 1.82e-14 | 0.0018 | True | True | False | 2 | 0 | 1 |  | 1024 |
| C b=1e+08 lambda=0.25 | 0.25 | 5.85e-17 | 7.49e-14 | 0.00078 | True | True | False | 20 | 0 | 10 |  | 0.25 |
| C b=1e+08 lambda=1 | 1 | 1.90e-17 | 1.77e-14 | 0.0011 | True | True | False | 23 | 0 | 11 |  | 1 |
| C b=1e+08 lambda=4 | 4 | 2.47e-17 | 4.80e-15 | 0.0051 | True | True | False | 20 | 0 | 11 |  | 4 |
| C b=1e+08 lambda=16 | 16 | 2.67e-17 | 4.71e-16 | 0.057 | True | False | False | 58 | 0 | 12 |  | 16 |
| C b=1e+08 lambda=64 | 64 | 6.64e-18 | 4.40e-16 | 0.015 | True | False | False | 58 | 0 | 12 |  | 64 |
| C b=1e+08 lambda=256 | 256 | 6.53e-18 | 3.51e-16 | 0.019 | True | False | False | 15 | 0 | 7 |  | 256 |
| C b=1e+08 lambda=1024 | 1024 | 9.95e-18 | 3.42e-16 | 0.029 | True | False | False | 13 | 0 | 6 |  | 1024 |

**Lambda_c by the rule:** 4.0

The check, lambda < 4, minimum located:

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima not located |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 3.64e-16 | 1.09e-12 | 0.00033 | True | True | False | 209 | 208 | 5 | minimum 0.25 |  |
| C b=1e+04 lambda=1 | 1 | 2.86e-16 | 7.10e-13 | 0.0004 | True | True | False | 260 | 259 | 10 | minimum 1 |  |
| C b=1e+08 lambda=0.25 | 0.25 | 3.91e-16 | 5.90e-14 | 0.0066 | True | True | False | 310 | 301 | 6 | minimum 0.25 |  |
| C b=1e+08 lambda=1 | 1 | 4.30e-17 | 4.19e-14 | 0.001 | True | True | False | 309 | 300 | 7 | minimum 1 |  |

**The check holds:** True

## Lambda_c on family (C) at Phi_w = 512pi -- SUPPLEMENTARY, not the rule (no Phi_w rung qualified)

No minimum located (Lambda_c = 0):

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima found, not located (lambda) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 1.04e-17 | 2.84e-13 | 3.7e-05 | True | True | False | 12 | 0 | 5 |  | 0.25 |
| C b=1e+04 lambda=1 | 1 | 7.29e-17 | 2.30e-13 | 0.00032 | True | True | False | 11 | 0 | 5 |  | 1 |
| C b=1e+04 lambda=4 | 4 | 3.86e-17 | 5.48e-14 | 0.0007 | True | True | False | 13 | 0 | 5 |  | 4 |
| C b=1e+04 lambda=16 | 16 | 1.21e-16 | 4.72e-15 | 0.026 | True | True | False | 17 | 0 | 6 |  | 16 |
| C b=1e+04 lambda=64 | 64 | 6.42e-17 | 2.40e-14 | 0.0027 | True | True | False | 5 | 0 | 3 |  | 64 |
| C b=1e+04 lambda=256 | 256 | 3.79e-19 | 7.83e-16 | 0.00048 | True | True | False | 4 | 0 | 2 |  | 256 |
| C b=1e+04 lambda=1024 | 1024 | 3.33e-17 | 1.82e-14 | 0.0018 | True | True | False | 2 | 0 | 1 |  | 1024 |
| C b=1e+08 lambda=0.25 | 0.25 | 5.85e-17 | 7.49e-14 | 0.00078 | True | True | False | 20 | 0 | 10 |  | 0.25 |
| C b=1e+08 lambda=1 | 1 | 1.90e-17 | 1.77e-14 | 0.0011 | True | True | False | 23 | 0 | 11 |  | 1 |
| C b=1e+08 lambda=4 | 4 | 2.47e-17 | 4.80e-15 | 0.0051 | True | True | False | 20 | 0 | 11 |  | 4 |
| C b=1e+08 lambda=16 | 16 | 2.67e-17 | 4.71e-16 | 0.057 | True | False | False | 58 | 0 | 12 |  | 16 |
| C b=1e+08 lambda=64 | 64 | 6.64e-18 | 4.40e-16 | 0.015 | True | False | False | 58 | 0 | 12 |  | 64 |
| C b=1e+08 lambda=256 | 256 | 6.53e-18 | 3.51e-16 | 0.019 | True | False | False | 15 | 0 | 7 |  | 256 |
| C b=1e+08 lambda=1024 | 1024 | 9.95e-18 | 3.42e-16 | 0.029 | True | False | False | 13 | 0 | 6 |  | 1024 |

**Lambda_c by the rule:** 4.0

The check, lambda < 4, minimum located:

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima not located |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 0.00e+00 | 7.25e-13 | 0 | True | True | False | 360 | 360 | 8 | minimum 0.25 |  |
| C b=1e+04 lambda=1 | 1 | 2.05e-16 | 1.23e-13 | 0.0017 | True | True | False | 390 | 390 | 9 | minimum 1 |  |
| C b=1e+08 lambda=0.25 | 0.25 | 7.31e-17 | 1.35e-13 | 0.00054 | True | True | False | 1162 | 1155 | 10 | minimum 0.25 |  |
| C b=1e+08 lambda=1 | 1 | 5.23e-16 | 7.52e-14 | 0.007 | True | True | False | 1140 | 1132 | 7 | minimum 1 |  |

**The check holds:** True

## Section 2.5 on these cases: calls that do not converge or reach depth_max

| part | Phi_w | Lambda_c | case | converged | node_limited | max depth | abserr/request | leaves by class: count, sum e', at depth_max | QUADPACK iroff1, iroff2, ier=2 at bisection |
|---|---|---|---|---|---|---|---|---|---|
| R (depth_max 20) | 2.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | True | 20 | 1.24 | Clenshaw-Curtis 207, 8.55e-16, 40; Levin 17, 2.62e-16, 0; window 66, 2.84e-16, 0 | 69, 72, 30 |
| R (depth_max 20) | 8.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | True | 20 | 1.2 | Clenshaw-Curtis 136, 3.77e-16, 6; Levin 18, 2.61e-16, 0; window 313, 7.67e-16, 0 | 123, 122, 34 |
| R (depth_max 20) | 32.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | True | 20 | 1.02 | Clenshaw-Curtis 21, 1.03e-16, 0; Levin 20, 2.60e-16, 0; window 168780, 1.04e-15, 1454 | 67695, 65657, 30 |
| R (depth_max 20) | 128.0 | 64.0 | R c=1e+06 x_s=0.7236 omega=0 | False | True | 17 | 1 | Levin 11, 2.33e-16, 0; window 15706, 1.41e-14, 0 | 8843, 6726, 50 |
| R (depth_max 20) | 128.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=1 | False | True | 20 | 1 | Levin 17, 2.58e-16, 0; window 30634, 1.10e-14, 2 | 13555, 13278, 43 |
| R (depth_max 20) | 512.0 | 64.0 | R c=1e+08 x_s=0.3820 omega=0 | False | True | 20 | 1 | Levin 15, 2.45e-16, 0; window 247692, 1.92e-14, 122 | 103322, 95680, 41 |
| R (depth_max 20) | 512.0 | 64.0 | R c=1e+08 x_s=0.3820 omega=1 | False | True | 20 | 1 | Levin 15, 2.45e-16, 0; window 213920, 1.95e-14, 90 | 89127, 82349, 99 |
| C0 (depth_max 20) | 2.0 | 0.0 | C b=1e+08 lambda=16 | False | False | 12 | 361 | Clenshaw-Curtis 43, 1.23e-16, 0; Levin 15, 3.47e-16, 0 | 6, 5, 51 |
| C0 (depth_max 20) | 2.0 | 0.0 | C b=1e+08 lambda=256 | False | False | 7 | 854 | Levin 15, 3.51e-16, 0 | 0, 0, None |
| C0 (depth_max 20) | 2.0 | 0.0 | C b=1e+08 lambda=64 | False | False | 12 | 54.7 | Clenshaw-Curtis 42, 7.97e-17, 0; Levin 16, 3.49e-16, 0 | 5, 5, None |
| C0 (depth_max 20) | 2.0 | 0.0 | C b=1e+08 lambda=1024 | False | False | 6 | 336 | Levin 13, 3.42e-16, 0 | 0, 0, None |
| C0 (depth_max 20) | 8.0 | 0.0 | C b=1e+08 lambda=16 | False | False | 12 | 361 | Clenshaw-Curtis 43, 1.23e-16, 0; Levin 15, 3.47e-16, 0 | 6, 5, 51 |
| C0 (depth_max 20) | 8.0 | 0.0 | C b=1e+08 lambda=256 | False | False | 7 | 854 | Levin 15, 3.51e-16, 0 | 0, 0, None |
| C0 (depth_max 20) | 8.0 | 0.0 | C b=1e+08 lambda=64 | False | False | 12 | 54.7 | Clenshaw-Curtis 42, 7.97e-17, 0; Levin 16, 3.49e-16, 0 | 5, 5, None |
| C0 (depth_max 20) | 8.0 | 0.0 | C b=1e+08 lambda=1024 | False | False | 6 | 336 | Levin 13, 3.42e-16, 0 | 0, 0, None |
| C0 (depth_max 20) | 32.0 | 0.0 | C b=1e+08 lambda=16 | False | False | 12 | 361 | Clenshaw-Curtis 43, 1.23e-16, 0; Levin 15, 3.47e-16, 0 | 6, 5, 51 |
| C0 (depth_max 20) | 32.0 | 0.0 | C b=1e+08 lambda=256 | False | False | 7 | 854 | Levin 15, 3.51e-16, 0 | 0, 0, None |
| C0 (depth_max 20) | 32.0 | 0.0 | C b=1e+08 lambda=64 | False | False | 12 | 54.7 | Clenshaw-Curtis 42, 7.97e-17, 0; Levin 16, 3.49e-16, 0 | 5, 5, None |
| C0 (depth_max 20) | 32.0 | 0.0 | C b=1e+08 lambda=1024 | False | False | 6 | 336 | Levin 13, 3.42e-16, 0 | 0, 0, None |
| C0 (depth_max 20) | 128.0 | 0.0 | C b=1e+08 lambda=16 | False | False | 12 | 361 | Clenshaw-Curtis 43, 1.23e-16, 0; Levin 15, 3.47e-16, 0 | 6, 5, 51 |
| C0 (depth_max 20) | 128.0 | 0.0 | C b=1e+08 lambda=64 | False | False | 12 | 54.7 | Clenshaw-Curtis 42, 7.97e-17, 0; Levin 16, 3.49e-16, 0 | 5, 5, None |
| C0 (depth_max 20) | 128.0 | 0.0 | C b=1e+08 lambda=256 | False | False | 7 | 854 | Levin 15, 3.51e-16, 0 | 0, 0, None |
| C0 (depth_max 20) | 128.0 | 0.0 | C b=1e+08 lambda=1024 | False | False | 6 | 336 | Levin 13, 3.42e-16, 0 | 0, 0, None |
| C0 (depth_max 20) | 512.0 | 0.0 | C b=1e+08 lambda=16 | False | False | 12 | 361 | Clenshaw-Curtis 43, 1.23e-16, 0; Levin 15, 3.47e-16, 0 | 6, 5, 51 |
| C0 (depth_max 20) | 512.0 | 0.0 | C b=1e+08 lambda=64 | False | False | 12 | 54.7 | Clenshaw-Curtis 42, 7.97e-17, 0; Levin 16, 3.49e-16, 0 | 5, 5, None |
| C0 (depth_max 20) | 512.0 | 0.0 | C b=1e+08 lambda=256 | False | False | 7 | 854 | Levin 15, 3.51e-16, 0 | 0, 0, None |
| C0 (depth_max 20) | 512.0 | 0.0 | C b=1e+08 lambda=1024 | False | False | 6 | 336 | Levin 13, 3.42e-16, 0 | 0, 0, None |
| parent (depth_max 20) | None | None | C b=1e+08 lambda=16 | False | False | 12 | 361 | Clenshaw-Curtis 43, 1.23e-16, 0; Levin 15, 3.47e-16, 0 | 6, 5, 51 |
| parent (depth_max 20) | None | None | C b=1e+08 lambda=256 | False | False | 7 | 854 | Levin 15, 3.51e-16, 0 | 0, 0, None |
| parent (depth_max 20) | None | None | C b=1e+08 lambda=64 | False | False | 12 | 54.7 | Clenshaw-Curtis 42, 7.97e-17, 0; Levin 16, 3.49e-16, 0 | 5, 5, None |
| parent (depth_max 20) | None | None | C b=1e+08 lambda=1024 | False | False | 6 | 336 | Levin 13, 3.42e-16, 0 | 0, 0, None |
| bounded:R (depth_max 6) | 128.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | False | 6 | 26.7 | Levin 14, 3.75e-14, 4; window 8192, 1.51e-16, 8192 | 2687, 2552, 55 |
| bounded:R (depth_max 8) | 128.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | False | 8 | 3.11 | Levin 17, 2.58e-16, 2; window 32768, 4.11e-15, 32768 | 12352, 12094, 57 |
| bounded:R (depth_max 6) | 512.0 | 64.0 | R c=1e+06 x_s=0.7236 omega=0 | False | False | 6 | 1.83 | Levin 8, 2.15e-16, 0; window 32896, 2.60e-14, 32896 | 27915, 12667, 57 |
| bounded:R (depth_max 10) | 128.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | False | 10 | 1.85 | Levin 17, 2.58e-16, 0; window 131072, 2.35e-15, 131072 | 56846, 53305, 57 |
| bounded:R (depth_max 8) | 512.0 | 64.0 | R c=1e+06 x_s=0.7236 omega=0 | False | False | 8 | 1.02 | Levin 8, 2.15e-16, 0; window 131584, 1.43e-14, 131584 | 116942, 52919, 57 |
| bounded:R (depth_max 6) | 512.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | False | 6 | 3.37 | Levin 14, 2.83e-16, 4; window 32896, 4.44e-15, 32896 | 12141, 11590, 140 |
| bounded:R (depth_max 8) | 512.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | False | 8 | 1.61 | Levin 15, 2.55e-16, 0; window 131584, 2.01e-15, 131584 | 53512, 49385, 141 |
| bounded:R (depth_max 12) | 128.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | False | 12 | 1.16 | Levin 17, 2.58e-16, 0; window 524288, 1.38e-15, 524288 | 219328, 201004, 57 |
| bounded:R (depth_max 10) | 512.0 | 64.0 | R c=1e+06 x_s=0.7236 omega=0 | False | False | 10 | 1.02 | Levin 8, 2.15e-16, 0; window 526336, 1.44e-14, 526336 | 469492, 204990, 57 |
| bounded:R (depth_max 6) | 512.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=1 | False | False | 6 | 1.79 | Levin 13, 2.61e-16, 2; window 32768, 1.99e-14, 32768 | 11191, 10384, 20 |
| bounded:R (depth_max 8) | 512.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=1 | False | False | 8 | 1.56 | Levin 14, 2.55e-16, 0; window 131072, 1.73e-14, 131072 | 51092, 47464, 20 |
| bounded:R (depth_max 10) | 512.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | False | 10 | 1.19 | Levin 15, 2.55e-16, 0; window 526336, 1.42e-15, 526336 | 225885, 205650, 141 |
| bounded:R (depth_max 10) | 512.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=1 | False | False | 10 | 1.34 | Levin 14, 2.55e-16, 0; window 524288, 1.48e-14, 524288 | 228477, 207457, 20 |
| bounded:R (depth_max 12) | 512.0 | 64.0 | R c=1e+06 x_s=0.7236 omega=0 | False | False | 12 | 1.03 | Levin 8, 2.15e-16, 0; window 2105344, 1.46e-14, 2105344 | 1875771, 878988, 57 |
| bounded:R (depth_max 12) | 512.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | False | 12 | 5.58 | Levin 15, 2.55e-16, 0; window 2105344, 7.59e-15, 2105344 | 916437, 839010, 141 |
| bounded:R (depth_max 12) | 512.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=1 | False | False | 12 | 1.16 | Levin 14, 2.55e-16, 0; window 2097152, 1.28e-14, 2097152 | 912129, 836813, 20 |

Units that did not finish within the unit limit:

- R commit Phi_w 128.0 pi, Lambda_c 64.0, R c=1e+08 x_s=0.7236 omega=0, depth_max 20
- R commit Phi_w 512.0 pi, Lambda_c 64.0, R c=1e+06 x_s=0.7236 omega=0, depth_max 20
- R commit Phi_w 512.0 pi, Lambda_c 64.0, R c=1e+08 x_s=0.7236 omega=0, depth_max 20
- R commit Phi_w 512.0 pi, Lambda_c 64.0, R c=1e+08 x_s=0.7236 omega=1, depth_max 20

Units that ended in an error:


### The bar lifted (window and guarded leaves allowed the floor-limited exit), where window leaves reached depth_max

| part | Phi_w | case | window leaves floor-limited | converged | node_limited | max depth | regions | err/abserr | abserr/request | honest |
|---|---|---|---|---|---|---|---|---|---|---|
| barlifted:R | 32.0 | R c=1e+08 x_s=0.7236 omega=0 | 34 | False | True | 9 | 79 | 0.16 | 1.19 | True |
| barlifted:R | 128.0 | R c=1e+08 x_s=0.7236 omega=1 | 56 | False | True | 7 | 214 | 0.029 | 1.02 | True |
| barlifted:R | 512.0 | R c=1e+08 x_s=0.3820 omega=0 | 282 | False | True | 6 | 712 | 0.014 | 1.06 | True |
| barlifted:R | 512.0 | R c=1e+08 x_s=0.3820 omega=1 | 266 | False | True | 6 | 695 | 0.027 | 1.08 | True |
| barlifted:R | 128.0 | R c=1e+08 x_s=0.7236 omega=0 | 372 | False | True | 8 | 502 | 0.21 | 1.27 | True |
| barlifted:R | 512.0 | R c=1e+06 x_s=0.7236 omega=0 | 408 | False | True | 3 | 654 | 0.032 | 1.04 | True |
| barlifted:R | 512.0 | R c=1e+08 x_s=0.7236 omega=0 | 1989 | False | False | 10 | 2004 | 0.094 | 3.64 | True |
| barlifted:R | 512.0 | R c=1e+08 x_s=0.7236 omega=1 | 423 | False | True | 6 | 781 | 0.021 | 1.07 | True |

## For context: the parent's code (bd21bf23, 06m) on every case (not part of either rule)

| case | true error | abserr | err/abserr | abserr/request | honest | converged | node_limited | regions | max depth |
|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 1.04e-17 | 2.84e-13 | 3.7e-05 | 0.0966 | True | True | False | 12 | 5 |
| C b=1e+04 lambda=1 | 7.29e-17 | 2.30e-13 | 0.00032 | 0.155 | True | True | False | 11 | 5 |
| C b=1e+04 lambda=1024 | 3.33e-17 | 1.82e-14 | 0.0018 | 0.618 | True | True | False | 2 | 1 |
| C b=1e+04 lambda=16 | 1.21e-16 | 4.72e-15 | 0.026 | 0.159 | True | True | False | 17 | 6 |
| C b=1e+04 lambda=256 | 3.79e-19 | 7.83e-16 | 0.00048 | 0.0317 | True | True | False | 4 | 2 |
| C b=1e+04 lambda=4 | 3.86e-17 | 5.48e-14 | 0.0007 | 0.308 | True | True | False | 13 | 5 |
| C b=1e+04 lambda=64 | 6.42e-17 | 2.40e-14 | 0.0027 | 0.279 | True | True | False | 5 | 3 |
| C b=1e+08 lambda=0.25 | 5.85e-17 | 7.49e-14 | 0.00078 | 0.527 | True | True | False | 20 | 10 |
| C b=1e+08 lambda=1 | 1.90e-17 | 1.77e-14 | 0.0011 | 0.235 | True | True | False | 23 | 11 |
| C b=1e+08 lambda=1024 | 9.95e-18 | 3.42e-16 | 0.029 | 336 | True | False | False | 13 | 6 |
| C b=1e+08 lambda=16 | 2.67e-17 | 4.71e-16 | 0.057 | 361 | True | False | False | 58 | 12 |
| C b=1e+08 lambda=256 | 6.53e-18 | 3.51e-16 | 0.019 | 854 | True | False | False | 15 | 7 |
| C b=1e+08 lambda=4 | 2.47e-17 | 4.80e-15 | 0.0051 | 0.569 | True | True | False | 20 | 11 |
| C b=1e+08 lambda=64 | 6.64e-18 | 4.40e-16 | 0.015 | 54.7 | True | False | False | 58 | 12 |
| R c=1e+04 x_s=0.3820 omega=0 | 3.47e-17 | 1.01e-12 | 3.4e-05 | 0.524 | True | True | False | 14 | 8 |
| R c=1e+04 x_s=0.3820 omega=1 | 3.47e-18 | 1.09e-12 | 3.2e-06 | 0.558 | True | True | False | 14 | 8 |
| R c=1e+04 x_s=0.7236 omega=0 | 1.60e-17 | 9.42e-14 | 0.00017 | 0.811 | True | True | False | 32 | 8 |
| R c=1e+06 x_s=0.3820 omega=0 | 2.21e-17 | 4.64e-14 | 0.00048 | 0.239 | True | True | False | 16 | 9 |
| R c=1e+06 x_s=0.3820 omega=1 | 4.47e-17 | 1.31e-13 | 0.00034 | 0.663 | True | True | False | 15 | 9 |
| R c=1e+06 x_s=0.7236 omega=0 | 3.37e-17 | 1.06e-14 | 0.0032 | 0.741 | True | True | False | 27 | 11 |
| R c=1e+08 x_s=0.3820 omega=0 | 3.44e-17 | 7.37e-15 | 0.0047 | 0.38 | True | True | False | 31 | 14 |
| R c=1e+08 x_s=0.3820 omega=1 | 2.58e-17 | 1.60e-14 | 0.0016 | 0.808 | True | True | False | 31 | 14 |
| R c=1e+08 x_s=0.7236 omega=0 | 2.26e-17 | 1.35e-15 | 0.017 | 0.959 | True | True | False | 68 | 15 |
| R c=1e+08 x_s=0.7236 omega=1 | 5.63e-17 | 7.64e-15 | 0.0074 | 0.679 | True | True | False | 39 | 15 |

Wall time 8559 s; uptime at end: 11:35  up 7 days, 13:56, 1 user, load averages: 32.93 27.70 23.51

