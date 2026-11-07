# 07 section 2.2 tables

    levin-phase-floor 07: section 2.2, Phi_w and Lambda_c (run)
    written 2026-11-07T21:51:30+00:00 on mps026088.phys.susx.ac.uk
    python 3.13.16 (/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python)
    script calibrate.py sha256 03a47ec30ec09ccf3ea8693b248dcdc6bd094ed4362219cc7f745a09e0af3df6
    prov.py sha256 2999c22ba523324fe7a87aa3c9cafc481f5afb345bbc9b70b6eaae6f29af174e
    AdaptiveLevin/levin_quadrature.py sha256 14293c4b2329e8c122ad21922a0b3d0d2fac51aeb13d019a02c2adca022e500b
    input docs/levin-phase-floor/07/calibration/scripts/reference.py sha256 43283037c7bad69b7624e05a6084067cf1e0f4d34da4a4a7c5e02cbe097e0a26
    input docs/levin-phase-floor/07/calibration/scripts/analysis07.py sha256 623901c0a00162d848a90d0b1f0aa677ee2ad6338b575c7efe1de4d3c0825619
    HEAD 91472dbee1daba8374b0018db1195cf6c7898bbb
    tree dirty: yes; git status --short:
         M AdaptiveLevin/levin_quadrature.py
         M ComputeTargets/QuadSourceIntegral.py
        ?? docs/levin-phase-floor/07/calibration/
        ?? docs/levin-phase-floor/07/earlier-attempts/
        ?? prompts/levin-phase-floor/logs/07-stationary-point-guard.md
    uptime at start: 21:51  up 8 days, 12 mins, 1 user, load averages: 8.37 5.83 4.89

## Phi_w on family (R)

(R)'s two references, by erf and by contour: largest relative disagreement 2.19e-37.

| Phi_w | case | true error | abserr | err/abserr | abserr/request | honest | converged | node_limited | regions | window pieces | windows stopped, of them with a leaf falling | max depth | theta'^2/theta'' at edges | point - (x_s - omega/c) | point - x_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2pi | R c=1e+04 x_s=0.3820 omega=0 | 1.73e-17 | 8.02e-12 | 2.2e-06 | 0.415 | True | True | False | 13 | 5 | 0, 0 | 3 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+04 x_s=0.3820 omega=1 | 4.86e-17 | 4.78e-12 | 1e-05 | 0.244 | True | True | False | 17 | 9 | 0, 0 | 4 | 12.57, 12.57 | 0.0e+00 | -1.0e-04 |
| 2pi | R c=1e+04 x_s=0.7236 omega=0 | 4.81e-17 | 3.78e-14 | 0.0013 | 0.0325 | True | True | False | 21 | 4 | 0, 0 | 6 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+06 x_s=0.3820 omega=0 | 5.14e-17 | 4.01e-14 | 0.0013 | 0.0206 | True | True | False | 34 | 6 | 0, 0 | 11 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+06 x_s=0.3820 omega=1 | 8.54e-17 | 1.74e-14 | 0.0049 | 0.0088 | True | True | False | 24 | 9 | 0, 0 | 7 | 12.57, 12.57 | 0.0e+00 | -1.0e-06 |
| 2pi | R c=1e+06 x_s=0.7236 omega=0 | 6.98e-17 | 8.03e-14 | 0.00087 | 0.56 | True | True | False | 23 | 8 | 0, 0 | 7 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+08 x_s=0.3820 omega=0 | 5.02e-17 | 7.31e-14 | 0.00069 | 0.377 | True | True | False | 48 | 6 | 0, 0 | 15 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+08 x_s=0.3820 omega=1 | 1.72e-16 | 1.03e-13 | 0.0017 | 0.521 | True | True | False | 37 | 6 | 0, 0 | 14 | 12.57, 12.57 | 0.0e+00 | -1.0e-08 |
| 2pi | R c=1e+08 x_s=0.7236 omega=0 | 3.68e-16 | 1.25e-14 | 0.03 | 0.885 | True | True | False | 81 | 4 | 0, 0 | 14 | 12.57, 12.57 | 0.0e+00 | 0.0e+00 |
| 2pi | R c=1e+08 x_s=0.7236 omega=1 | 4.01e-17 | 4.01e-14 | 0.001 | 0.356 | True | True | False | 60 | 10 | 0, 0 | 15 | 12.57, 12.57 | 0.0e+00 | -1.0e-08 |
| 8pi | R c=1e+04 x_s=0.3820 omega=0 | 1.98e-16 | 1.95e-12 | 0.0001 | 0.101 | True | True | False | 28 | 23 | 1, 1 | 3 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+04 x_s=0.3820 omega=1 | 1.87e-16 | 3.84e-12 | 4.9e-05 | 0.196 | True | True | False | 29 | 24 | 1, 1 | 3 | 50.27, 50.27 | 0.0e+00 | -1.0e-04 |
| 8pi | R c=1e+04 x_s=0.7236 omega=0 | 2.01e-16 | 9.27e-14 | 0.0022 | 0.0797 | True | True | False | 18 | 12 | 0, 0 | 3 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+06 x_s=0.3820 omega=0 | 6.31e-17 | 8.34e-13 | 7.6e-05 | 0.429 | True | True | False | 26 | 13 | 0, 0 | 6 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+06 x_s=0.3820 omega=1 | 3.12e-17 | 2.11e-14 | 0.0015 | 0.0107 | True | True | False | 36 | 23 | 0, 0 | 6 | 50.27, 50.27 | 0.0e+00 | -1.0e-06 |
| 8pi | R c=1e+06 x_s=0.7236 omega=0 | 1.82e-16 | 6.80e-14 | 0.0027 | 0.475 | True | True | False | 37 | 16 | 0, 0 | 10 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+08 x_s=0.3820 omega=0 | 6.84e-17 | 1.33e-14 | 0.0051 | 0.0685 | True | True | False | 37 | 18 | 0, 0 | 9 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+08 x_s=0.3820 omega=1 | 4.53e-17 | 2.31e-14 | 0.002 | 0.117 | True | True | False | 37 | 18 | 0, 0 | 9 | 50.27, 50.27 | 0.0e+00 | -1.0e-08 |
| 8pi | R c=1e+08 x_s=0.7236 omega=0 | 9.29e-17 | 1.24e-14 | 0.0075 | 0.881 | True | True | False | 53 | 25 | 1, 1 | 15 | 50.27, 50.27 | 0.0e+00 | 0.0e+00 |
| 8pi | R c=1e+08 x_s=0.7236 omega=1 | 2.61e-17 | 1.40e-14 | 0.0019 | 0.125 | True | True | False | 35 | 16 | 0, 0 | 9 | 50.27, 50.27 | 0.0e+00 | -1.0e-08 |
| 32pi | R c=1e+04 x_s=0.3820 omega=0 | 2.88e-16 | 5.95e-13 | 0.00048 | 0.0308 | True | True | False | 61 | 59 | 1, 1 | 3 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+04 x_s=0.3820 omega=1 | 2.26e-16 | 4.82e-12 | 4.7e-05 | 0.246 | True | True | False | 60 | 58 | 1, 1 | 3 | 201.1, 201.1 | 0.0e+00 | -1.0e-04 |
| 32pi | R c=1e+04 x_s=0.7236 omega=0 | 2.93e-16 | 4.32e-13 | 0.00068 | 0.371 | True | True | False | 57 | 54 | 1, 1 | 2 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+06 x_s=0.3820 omega=0 | 2.32e-16 | 1.35e-13 | 0.0017 | 0.0693 | True | True | False | 67 | 58 | 1, 1 | 4 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+06 x_s=0.3820 omega=1 | 2.22e-16 | 3.71e-13 | 0.0006 | 0.187 | True | True | False | 72 | 63 | 1, 1 | 4 | 201.1, 201.1 | 0.0e+00 | -1.0e-06 |
| 32pi | R c=1e+06 x_s=0.7236 omega=0 | 9.90e-17 | 9.91e-14 | 0.001 | 0.692 | True | True | False | 78 | 68 | 1, 1 | 5 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+08 x_s=0.3820 omega=0 | 1.59e-17 | 2.84e-14 | 0.00056 | 0.146 | True | True | False | 71 | 56 | 1, 1 | 7 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+08 x_s=0.3820 omega=1 | 4.70e-17 | 4.95e-14 | 0.00095 | 0.25 | True | True | False | 71 | 55 | 1, 1 | 7 | 201.1, 201.1 | 0.0e+00 | -1.0e-08 |
| 32pi | R c=1e+08 x_s=0.7236 omega=0 | 1.69e-16 | 2.46e-14 | 0.0069 | 1.74 | True | False | False | 98 | 57 | 1, 1 | 18 | 201.1, 201.1 | 0.0e+00 | 0.0e+00 |
| 32pi | R c=1e+08 x_s=0.7236 omega=1 | 1.34e-16 | 5.50e-14 | 0.0024 | 0.488 | True | True | False | 70 | 54 | 1, 1 | 8 | 201.1, 201.1 | 0.0e+00 | -1.0e-08 |
| 128pi | R c=1e+04 x_s=0.3820 omega=0 | 1.46e-16 | 1.17e-13 | 0.0012 | 0.00604 | True | True | False | 188 | 186 | 1, 1 | 2 | 804.2, 804.2 | 0.0e+00 | 0.0e+00 |
| 128pi | R c=1e+04 x_s=0.3820 omega=1 | 8.01e-16 | 1.23e-13 | 0.0065 | 0.00629 | True | True | False | 173 | 171 | 1, 1 | 2 | 804.2, 804.2 | 0.0e+00 | -1.0e-04 |
| 128pi | R c=1e+04 x_s=0.7236 omega=0 | 3.14e-16 | 9.34e-14 | 0.0034 | 0.0804 | True | True | False | 176 | 175 | 1, 1 | 2 | 804.2, 763.9 | 0.0e+00 | 0.0e+00 |
| 128pi | R c=1e+06 x_s=0.3820 omega=0 | 6.74e-17 | 1.04e-12 | 6.5e-05 | 0.536 | True | True | False | 191 | 185 | 1, 1 | 2 | 804.2, 804.2 | 0.0e+00 | 0.0e+00 |
| 128pi | R c=1e+06 x_s=0.3820 omega=1 | 3.10e-16 | 1.82e-12 | 0.00017 | 0.92 | True | True | False | 190 | 184 | 1, 1 | 2 | 804.2, 804.2 | 0.0e+00 | -1.0e-06 |
| 128pi | R c=1e+06 x_s=0.7236 omega=0 | 1.24e-17 | 1.11e-13 | 0.00011 | 0.778 | True | True | False | 182 | 175 | 1, 1 | 3 | 804.2, 804.2 | 0.0e+00 | 0.0e+00 |
| 128pi | R c=1e+08 x_s=0.3820 omega=0 | 1.17e-16 | 4.92e-14 | 0.0024 | 0.254 | True | True | False | 164 | 151 | 1, 1 | 6 | 804.2, 804.2 | 0.0e+00 | 0.0e+00 |
| 128pi | R c=1e+08 x_s=0.3820 omega=1 | 1.77e-16 | 7.27e-14 | 0.0024 | 0.367 | True | True | False | 175 | 162 | 1, 1 | 6 | 804.2, 804.2 | 0.0e+00 | -1.0e-08 |
| 128pi | R c=1e+08 x_s=0.7236 omega=0 | 4.36e-16 | 4.55e-14 | 0.0096 | 3.23 | True | False | False | 186 | 169 | 1, 1 | 8 | 804.2, 804.2 | 0.0e+00 | 0.0e+00 |
| 128pi | R c=1e+08 x_s=0.7236 omega=1 | 3.25e-16 | 3.72e-14 | 0.0087 | 0.33 | True | True | False | 176 | 163 | 1, 1 | 6 | 804.2, 804.2 | 0.0e+00 | -1.0e-08 |
| 512pi | R c=1e+04 x_s=0.3820 omega=0 | 2.60e-16 | 1.77e-13 | 0.0015 | 0.00914 | True | True | False | 432 | 431 | 1, 1 | 2 | 1459, 3217 | 0.0e+00 | 0.0e+00 |
| 512pi | R c=1e+04 x_s=0.3820 omega=1 | 3.57e-16 | 1.98e-13 | 0.0018 | 0.0101 | True | True | False | 409 | 408 | 1, 1 | 2 | 1458, 3217 | 0.0e+00 | -1.0e-04 |
| 512pi | R c=1e+04 x_s=0.7236 omega=0 | 7.89e-16 | 2.30e-13 | 0.0034 | 0.198 | True | True | False | 344 | 343 | 1, 1 | 2 | 3217, 763.9 | 0.0e+00 | 0.0e+00 |
| 512pi | R c=1e+06 x_s=0.3820 omega=0 | 3.36e-16 | 4.37e-13 | 0.00077 | 0.225 | True | True | False | 561 | 558 | 1, 1 | 2 | 3217, 3217 | 0.0e+00 | 0.0e+00 |
| 512pi | R c=1e+06 x_s=0.3820 omega=1 | 1.26e-15 | 6.53e-13 | 0.0019 | 0.33 | True | True | False | 560 | 557 | 1, 1 | 2 | 3217, 3217 | 0.0e+00 | -1.0e-06 |
| 512pi | R c=1e+06 x_s=0.7236 omega=0 | 9.14e-16 | 1.09e-13 | 0.0084 | 0.759 | True | True | False | 571 | 566 | 1, 1 | 2 | 3217, 3217 | 0.0e+00 | 0.0e+00 |
| 512pi | R c=1e+08 x_s=0.3820 omega=0 | 1.09e-15 | 1.29e-13 | 0.0085 | 0.663 | True | True | False | 556 | 546 | 1, 1 | 4 | 3217, 3217 | 0.0e+00 | 0.0e+00 |
| 512pi | R c=1e+08 x_s=0.3820 omega=1 | 8.05e-16 | 1.69e-13 | 0.0048 | 0.854 | True | True | False | 612 | 602 | 1, 1 | 4 | 3217, 3217 | 0.0e+00 | -1.0e-08 |
| 512pi | R c=1e+08 x_s=0.7236 omega=0 | 1.54e-15 | 8.37e-14 | 0.018 | 5.95 | True | False | False | 591 | 576 | 1, 1 | 7 | 3217, 3217 | 0.0e+00 | 0.0e+00 |
| 512pi | R c=1e+08 x_s=0.7236 omega=1 | 4.09e-16 | 9.16e-14 | 0.0045 | 0.814 | True | True | False | 535 | 526 | 1, 1 | 4 | 3217, 3217 | 0.0e+00 | -1.0e-08 |

**Phi_w by the rule:** 2 pi

## Lambda_c on family (C) at Phi_w = 2pi

No minimum located (Lambda_c = 0):

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima found, not located (lambda) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 1.04e-17 | 2.38e-11 | 4.4e-07 | True | True | False | 11 | 0 | 5 |  | 0.25 |
| C b=1e+04 lambda=1 | 1 | 7.29e-17 | 1.85e-12 | 3.9e-05 | True | True | False | 10 | 0 | 5 |  | 1 |
| C b=1e+04 lambda=4 | 4 | 3.86e-17 | 5.48e-14 | 0.0007 | True | True | False | 13 | 0 | 5 |  | 4 |
| C b=1e+04 lambda=16 | 16 | 1.19e-16 | 1.55e-13 | 0.00077 | True | True | False | 15 | 0 | 6 |  | 16 |
| C b=1e+04 lambda=64 | 64 | 6.42e-17 | 2.40e-14 | 0.0027 | True | True | False | 5 | 0 | 3 |  | 64 |
| C b=1e+04 lambda=256 | 256 | 8.67e-19 | 6.25e-14 | 1.4e-05 | True | True | False | 3 | 0 | 2 |  | 256 |
| C b=1e+04 lambda=1024 | 1024 | 3.33e-17 | 1.82e-14 | 0.0018 | True | True | False | 2 | 0 | 1 |  | 1024 |
| C b=1e+08 lambda=0.25 | 0.25 | 1.84e-17 | 4.66e-13 | 4e-05 | True | True | False | 18 | 0 | 10 |  | 0.25 |
| C b=1e+08 lambda=1 | 1 | 1.95e-17 | 5.52e-13 | 3.5e-05 | True | True | False | 19 | 0 | 11 |  | 1 |
| C b=1e+08 lambda=4 | 4 | 2.47e-17 | 4.80e-15 | 0.0051 | True | True | False | 20 | 0 | 11 |  | 4 |
| C b=1e+08 lambda=16 | 16 | 2.67e-17 | 4.71e-16 | 0.057 | True | False | False | 58 | 0 | 12 |  | 16 |
| C b=1e+08 lambda=64 | 64 | 6.64e-18 | 4.40e-16 | 0.015 | True | False | False | 58 | 0 | 12 |  | 64 |
| C b=1e+08 lambda=256 | 256 | 6.53e-18 | 3.51e-16 | 0.019 | True | False | False | 15 | 0 | 7 |  | 256 |
| C b=1e+08 lambda=1024 | 1024 | 9.95e-18 | 3.42e-16 | 0.029 | True | False | False | 13 | 0 | 6 |  | 1024 |

**Lambda_c by the rule:** 4.0

The check, lambda < 4, minimum located:

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | node_limited | regions | window pieces | max depth | located | minima not located |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 2.78e-17 | 1.64e-12 | 1.7e-05 | True | True | False | 19 | 14 | 5 | minimum 0.25 |  |
| C b=1e+04 lambda=1 | 1 | 3.82e-17 | 1.56e-12 | 2.4e-05 | True | True | False | 17 | 12 | 3 | minimum 1 |  |
| C b=1e+08 lambda=0.25 | 0.25 | 7.91e-17 | 1.00e-13 | 0.00079 | True | True | False | 30 | 9 | 12 | minimum 0.25 |  |
| C b=1e+08 lambda=1 | 1 | 4.33e-17 | 2.68e-14 | 0.0016 | True | True | False | 26 | 11 | 7 | minimum 1 |  |

**The check holds:** True

## Section 2.5 on these cases: calls that do not converge or reach depth_max

| part | Phi_w | Lambda_c | case | converged | node_limited | max depth | abserr/request | leaves by class: count, sum e', at depth_max | QUADPACK iroff1, iroff2, ier=2 at bisection |
|---|---|---|---|---|---|---|---|---|---|
| R (depth_max 20) | 32.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | False | 18 | 1.74 | Clenshaw-Curtis 21, 1.03e-16, 0; Levin 20, 2.60e-16, 0; window 57, 2.40e-14, 0 | 15, 16, 39 |
| R (depth_max 20) | 128.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | False | 8 | 3.23 | Levin 17, 2.58e-16, 0; window 169, 4.48e-14, 0 | 6, 6, 53 |
| R (depth_max 20) | 512.0 | 64.0 | R c=1e+08 x_s=0.7236 omega=0 | False | False | 7 | 5.95 | Levin 15, 2.55e-16, 0; window 576, 8.19e-14, 0 | 6, 6, 71 |
| C0 (depth_max 20) | 2.0 | 0.0 | C b=1e+08 lambda=16 | False | False | 12 | 36.1 | Clenshaw-Curtis 43, 1.23e-16, 0; Levin 15, 3.47e-16, 0 | 6, 5, 51 |
| C0 (depth_max 20) | 2.0 | 0.0 | C b=1e+08 lambda=256 | False | False | 7 | 85.4 | Levin 15, 3.51e-16, 0 | 0, 0, None |
| C0 (depth_max 20) | 2.0 | 0.0 | C b=1e+08 lambda=64 | False | False | 12 | 5.47 | Clenshaw-Curtis 42, 7.97e-17, 0; Levin 16, 3.49e-16, 0 | 5, 5, None |
| C0 (depth_max 20) | 2.0 | 0.0 | C b=1e+08 lambda=1024 | False | False | 6 | 33.6 | Levin 13, 3.42e-16, 0 | 0, 0, None |
| parent (depth_max 20) | None | None | C b=1e+08 lambda=1024 | False | False | 6 | 33.6 | Levin 13, 3.42e-16, 0 | 0, 0, None |
| parent (depth_max 20) | None | None | C b=1e+08 lambda=16 | False | False | 12 | 36.1 | Clenshaw-Curtis 43, 1.23e-16, 0; Levin 15, 3.47e-16, 0 | 6, 5, 51 |
| parent (depth_max 20) | None | None | C b=1e+08 lambda=256 | False | False | 7 | 85.4 | Levin 15, 3.51e-16, 0 | 0, 0, None |
| parent (depth_max 20) | None | None | C b=1e+08 lambda=64 | False | False | 12 | 5.47 | Clenshaw-Curtis 42, 7.97e-17, 0; Levin 16, 3.49e-16, 0 | 5, 5, None |

Windows stopped by section 1.3's rule while one of their leaves' errors fell at its last bisection (a finding for the user, not a stop; section 9 point 4):

- R commit Phi_w 8.0 pi, R c=1e+04 x_s=0.3820 omega=0: window [0.3110678572, 0.4528641653], 12 of 23 leaves falling, stopped after split 18, deepest 3
- R commit Phi_w 8.0 pi, R c=1e+08 x_s=0.7236 omega=0: window [0.7228978162, 0.7243157793], 16 of 25 leaves falling, stopped after split 34, deepest 2
- R commit Phi_w 8.0 pi, R c=1e+04 x_s=0.3820 omega=1: window [0.3109678572, 0.4527641653], 14 of 24 leaves falling, stopped after split 19, deepest 3
- R commit Phi_w 32.0 pi, R c=1e+04 x_s=0.7236 omega=0: window [0.5818104897, 0.8654031058], 26 of 54 leaves falling, stopped after split 23, deepest 2
- R commit Phi_w 32.0 pi, R c=1e+04 x_s=0.3820 omega=0: window [0.2401697032, 0.5237623193], 32 of 59 leaves falling, stopped after split 26, deepest 3
- R commit Phi_w 32.0 pi, R c=1e+06 x_s=0.3820 omega=0: window [0.3677863804, 0.3961456421], 30 of 58 leaves falling, stopped after split 33, deepest 3
- R commit Phi_w 32.0 pi, R c=1e+06 x_s=0.7236 omega=0: window [0.7094271669, 0.7377864286], 43 of 68 leaves falling, stopped after split 43, deepest 3
- R commit Phi_w 32.0 pi, R c=1e+08 x_s=0.3820 omega=0: window [0.3805480482, 0.3833839743], 28 of 56 leaves falling, stopped after split 36, deepest 3
- R commit Phi_w 32.0 pi, R c=1e+08 x_s=0.7236 omega=0: window [0.7221888347, 0.7250247608], 32 of 57 leaves falling, stopped after split 39, deepest 3
- R commit Phi_w 32.0 pi, R c=1e+04 x_s=0.3820 omega=1: window [0.2400697032, 0.5236623193], 33 of 58 leaves falling, stopped after split 26, deepest 3
- R commit Phi_w 32.0 pi, R c=1e+06 x_s=0.3820 omega=1: window [0.3677853804, 0.3961446421], 33 of 63 leaves falling, stopped after split 36, deepest 3
- R commit Phi_w 32.0 pi, R c=1e+08 x_s=0.3820 omega=1: window [0.3805480382, 0.3833839643], 24 of 55 leaves falling, stopped after split 36, deepest 3
- R commit Phi_w 32.0 pi, R c=1e+08 x_s=0.7236 omega=1: window [0.7221888247, 0.7250247508], 24 of 54 leaves falling, stopped after split 34, deepest 3
- R commit Phi_w 128.0 pi, R c=1e+04 x_s=0.3820 omega=0: window [0.09837339511, 0.6655586274], 101 of 186 leaves falling, stopped after split 57, deepest 2
- R commit Phi_w 128.0 pi, R c=1e+08 x_s=0.7236 omega=0: window [0.7207708716, 0.7264427239], 70 of 169 leaves falling, stopped after split 53, deepest 1
- R commit Phi_w 128.0 pi, R c=1e+04 x_s=0.7236 omega=0: window [0.4400141816, 1], 84 of 175 leaves falling, stopped after split 49, deepest 2
- R commit Phi_w 128.0 pi, R c=1e+08 x_s=0.3820 omega=0: window [0.3791300851, 0.3848019374], 28 of 151 leaves falling, stopped after split 32, deepest 2
- R commit Phi_w 128.0 pi, R c=1e+06 x_s=0.7236 omega=0: window [0.6952475361, 0.7519660594], 79 of 175 leaves falling, stopped after split 52, deepest 2
- R commit Phi_w 128.0 pi, R c=1e+06 x_s=0.3820 omega=0: window [0.3536067496, 0.4103252729], 96 of 185 leaves falling, stopped after split 59, deepest 2
- R commit Phi_w 128.0 pi, R c=1e+04 x_s=0.3820 omega=1: window [0.09827339511, 0.6654586274], 71 of 171 leaves falling, stopped after split 43, deepest 2
- R commit Phi_w 128.0 pi, R c=1e+08 x_s=0.3820 omega=1: window [0.3791300751, 0.3848019274], 48 of 162 leaves falling, stopped after split 43, deepest 2
- R commit Phi_w 128.0 pi, R c=1e+08 x_s=0.7236 omega=1: window [0.7207708616, 0.7264427139], 53 of 163 leaves falling, stopped after split 44, deepest 2
- R commit Phi_w 128.0 pi, R c=1e+06 x_s=0.3820 omega=1: window [0.3536057496, 0.4103242729], 98 of 184 leaves falling, stopped after split 60, deepest 2
- R commit Phi_w 512.0 pi, R c=1e+04 x_s=0.7236 omega=0: window [0.1564215655, 1], 37 of 343 leaves falling, stopped after split 25, deepest 2
- R commit Phi_w 512.0 pi, R c=1e+04 x_s=0.3820 omega=0: window [0, 0.9491512435], 100 of 431 leaves falling, stopped after split 57, deepest 2
- R commit Phi_w 512.0 pi, R c=1e+08 x_s=0.7236 omega=0: window [0.7179349454, 0.7292786501], 112 of 576 leaves falling, stopped after split 71, deepest 1
- R commit Phi_w 512.0 pi, R c=1e+06 x_s=0.7236 omega=0: window [0.6668882745, 0.780325321], 91 of 566 leaves falling, stopped after split 54, deepest 2
- R commit Phi_w 512.0 pi, R c=1e+06 x_s=0.3820 omega=0: window [0.325247488, 0.4386845345], 74 of 558 leaves falling, stopped after split 45, deepest 2
- R commit Phi_w 512.0 pi, R c=1e+08 x_s=0.3820 omega=0: window [0.3762941589, 0.3876378636], 53 of 546 leaves falling, stopped after split 42, deepest 2
- R commit Phi_w 512.0 pi, R c=1e+04 x_s=0.3820 omega=1: window [0, 0.9490512435], 55 of 408 leaves falling, stopped after split 34, deepest 2
- R commit Phi_w 512.0 pi, R c=1e+06 x_s=0.3820 omega=1: window [0.325246488, 0.4386835345], 76 of 557 leaves falling, stopped after split 46, deepest 2
- R commit Phi_w 512.0 pi, R c=1e+08 x_s=0.7236 omega=1: window [0.7179349354, 0.7292786401], 16 of 526 leaves falling, stopped after split 21, deepest 2
- R commit Phi_w 512.0 pi, R c=1e+08 x_s=0.3820 omega=1: window [0.3762941489, 0.3876378536], 166 of 602 leaves falling, stopped after split 98, deepest 2
- Ccheck commit Phi_w 2.0 pi, C b=1e+04 lambda=1: window [0.2757465999, 0.4881854226], 3 of 12 leaves falling, stopped after split 13, deepest 3
- Ccheck commit Phi_w 2.0 pi, C b=1e+04 lambda=0.25: window [0.2653512845, 0.498580738], 5 of 14 leaves falling, stopped after split 14, deepest 5
- Ccheck commit Phi_w 2.0 pi, C b=1e+08 lambda=1: window [0.3770357429, 0.3868962796], 3 of 11 leaves falling, stopped after split 22, deepest 4

Units that did not finish within the unit limit:


Units that ended in an error:


### The bar lifted (window and guarded leaves allowed the floor-limited exit), where window leaves reached depth_max

| part | Phi_w | case | window leaves floor-limited | converged | node_limited | max depth | regions | err/abserr | abserr/request | honest |
|---|---|---|---|---|---|---|---|---|---|---|

## For context: the parent's code (bd21bf23, 06m) on every case (not part of either rule)

| case | true error | abserr | err/abserr | abserr/request | honest | converged | node_limited | regions | max depth |
|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 1.04e-17 | 2.38e-11 | 4.4e-07 | 0.811 | True | True | False | 11 | 5 |
| C b=1e+04 lambda=1 | 7.29e-17 | 1.85e-12 | 3.9e-05 | 0.125 | True | True | False | 10 | 5 |
| C b=1e+04 lambda=1024 | 3.33e-17 | 1.82e-14 | 0.0018 | 0.0618 | True | True | False | 2 | 1 |
| C b=1e+04 lambda=16 | 1.19e-16 | 1.55e-13 | 0.00077 | 0.521 | True | True | False | 15 | 6 |
| C b=1e+04 lambda=256 | 8.67e-19 | 6.25e-14 | 1.4e-05 | 0.253 | True | True | False | 3 | 2 |
| C b=1e+04 lambda=4 | 3.86e-17 | 5.48e-14 | 0.0007 | 0.0308 | True | True | False | 13 | 5 |
| C b=1e+04 lambda=64 | 6.42e-17 | 2.40e-14 | 0.0027 | 0.0279 | True | True | False | 5 | 3 |
| C b=1e+08 lambda=0.25 | 1.84e-17 | 4.66e-13 | 4e-05 | 0.328 | True | True | False | 18 | 10 |
| C b=1e+08 lambda=1 | 1.95e-17 | 5.52e-13 | 3.5e-05 | 0.734 | True | True | False | 19 | 11 |
| C b=1e+08 lambda=1024 | 9.95e-18 | 3.42e-16 | 0.029 | 33.6 | True | False | False | 13 | 6 |
| C b=1e+08 lambda=16 | 2.67e-17 | 4.71e-16 | 0.057 | 36.1 | True | False | False | 58 | 12 |
| C b=1e+08 lambda=256 | 6.53e-18 | 3.51e-16 | 0.019 | 85.4 | True | False | False | 15 | 7 |
| C b=1e+08 lambda=4 | 2.47e-17 | 4.80e-15 | 0.0051 | 0.0569 | True | True | False | 20 | 11 |
| C b=1e+08 lambda=64 | 6.64e-18 | 4.40e-16 | 0.015 | 5.47 | True | False | False | 58 | 12 |
| R c=1e+04 x_s=0.3820 omega=0 | 3.12e-17 | 7.16e-12 | 4.4e-06 | 0.37 | True | True | False | 11 | 6 |
| R c=1e+04 x_s=0.3820 omega=1 | 3.47e-18 | 1.62e-11 | 2.1e-07 | 0.828 | True | True | False | 11 | 6 |
| R c=1e+04 x_s=0.7236 omega=0 | 1.52e-17 | 5.39e-13 | 2.8e-05 | 0.463 | True | True | False | 27 | 7 |
| R c=1e+06 x_s=0.3820 omega=0 | 5.29e-17 | 3.08e-13 | 0.00017 | 0.158 | True | True | False | 15 | 9 |
| R c=1e+06 x_s=0.3820 omega=1 | 4.47e-17 | 1.31e-13 | 0.00034 | 0.0663 | True | True | False | 15 | 9 |
| R c=1e+06 x_s=0.7236 omega=0 | 3.45e-17 | 9.46e-14 | 0.00036 | 0.661 | True | True | False | 20 | 10 |
| R c=1e+08 x_s=0.3820 omega=0 | 3.30e-17 | 3.36e-14 | 0.00098 | 0.173 | True | True | False | 29 | 14 |
| R c=1e+08 x_s=0.3820 omega=1 | 2.57e-17 | 2.29e-14 | 0.0011 | 0.116 | True | True | False | 28 | 14 |
| R c=1e+08 x_s=0.7236 omega=0 | 2.25e-17 | 1.06e-14 | 0.0021 | 0.75 | True | True | False | 58 | 15 |
| R c=1e+08 x_s=0.7236 omega=1 | 5.05e-17 | 3.60e-14 | 0.0014 | 0.32 | True | True | False | 25 | 14 |

Wall time 49 s; uptime at end: 21:52  up 8 days, 13 mins, 1 user, load averages: 12.50 7.26 5.46

