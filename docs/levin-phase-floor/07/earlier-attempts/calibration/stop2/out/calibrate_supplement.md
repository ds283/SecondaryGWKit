# 07 section 2, supplementary: (C) at every rung (not the prompt's rule)

script sha256 `d0a55df541253a7beb0888101ecb2db1c077b70aa13e7f11ddcc7466d6d73e52`, levin_quadrature.py sha256 `7d6921acb8eb751d15c9bcda66f9df3b4bf0fcc88117be173f21765f56960d30`, HEAD `cf3efac8fccc56723ad83fb11ff2514d1d2551f8`, tree dirty: True

With Lambda_c = 0 nothing is located on (C), so its runs are the same at every rung; they are tabulated once, from the first rung.

## (C), no minimum located (Lambda_c = 0)

| case | lambda (actual) | true error | abserr | err/abserr | honest | converged | regions | max depth | minima found, lambda |
|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 0.25 | 2.79e-13 | 2.88e-13 | 0.97 | True | True | 11 | 5 | 0.25 |
| C b=1e+04 lambda=1 | 1 | 6.38e-14 | 9.18e-14 | 0.69 | True | True | 14 | 6 | 1 |
| C b=1e+04 lambda=4 | 4 | 3.74e-14 | 1.26e-13 | 0.3 | True | True | 11 | 5 | 4 |
| C b=1e+04 lambda=16 | 16 | 1.41e-15 | 1.19e-13 | 0.012 | True | False | 14 | 6 | 16 |
| C b=1e+04 lambda=64 | 64 | 2.01e-14 | 2.39e-14 | 0.84 | True | True | 5 | 3 | 64 |
| C b=1e+04 lambda=256 | 256 | 5.89e-16 | 7.82e-16 | 0.75 | True | True | 4 | 2 | 256 |
| C b=1e+04 lambda=1024 | 1024 | 1.76e-14 | 1.82e-14 | 0.97 | True | True | 2 | 1 | 1024 |
| C b=1e+08 lambda=0.25 | 0.25 | 7.03e-16 | 5.99e-15 | 0.12 | True | True | 21 | 10 | 0.25 |
| C b=1e+08 lambda=1 | 1 | 2.80e-16 | 4.26e-15 | 0.066 | True | True | 20 | 10 | 1 |
| C b=1e+08 lambda=4 | 4 | 2.35e-15 | 3.61e-15 | 0.65 | True | True | 20 | 10 | 4 |
| C b=1e+08 lambda=16 | 16 | 1.29e-15 | 2.43e-14 | 0.053 | True | False | 17 | 9 | 16 |
| C b=1e+08 lambda=64 | 64 | 5.72e-15 | 6.48e-15 | 0.88 | True | False | 13 | 8 | 64 |
| C b=1e+08 lambda=256 | 256 | 1.08e-16 | 7.75e-16 | 0.14 | True | False | 12 | 7 | 256 |
| C b=1e+08 lambda=1024 | 1024 | 1.67e-16 | 8.35e-16 | 0.2 | True | False | 10 | 6 | 1024 |

## Phi_w = 2pi: Lambda_c by the rule = 4.0; the check holds: True

| case | true error | abserr | err/abserr | honest | converged | regions | window pieces | max depth | located (kind, lambda) |
|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 1.35e-16 | 5.13e-13 | 0.00026 | True | True | 14 | 4 | 5 | minimum 0.25 |
| C b=1e+04 lambda=1 | 6.08e-13 | 7.74e-13 | 0.79 | True | True | 12 | 4 | 5 | minimum 1 |
| C b=1e+08 lambda=0.25 | 4.07e-15 | 2.78e-14 | 0.15 | True | True | 43 | 5 | 9 | minimum 0.25 |
| C b=1e+08 lambda=1 | 1.63e-14 | 2.88e-14 | 0.57 | True | True | 23 | 4 | 7 | minimum 1 |

## Phi_w = 8pi: Lambda_c by the rule = 4.0; the check holds: True

| case | true error | abserr | err/abserr | honest | converged | regions | window pieces | max depth | located (kind, lambda) |
|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 1.46e-13 | 2.19e-12 | 0.067 | True | True | 13 | 10 | 1 | minimum 0.25 |
| C b=1e+04 lambda=1 | 1.18e-13 | 4.04e-13 | 0.29 | True | True | 13 | 10 | 1 | minimum 1 |
| C b=1e+08 lambda=0.25 | 4.16e-15 | 3.18e-14 | 0.13 | True | True | 69 | 11 | 11 | minimum 0.25 |
| C b=1e+08 lambda=1 | 1.82e-15 | 2.04e-14 | 0.089 | True | True | 50 | 10 | 6 | minimum 1 |

## Phi_w = 32pi: Lambda_c by the rule = 4.0; the check holds: True

| case | true error | abserr | err/abserr | honest | converged | regions | window pieces | max depth | located (kind, lambda) |
|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 3.24e-15 | 1.95e-12 | 0.0017 | True | True | 39 | 37 | 2 | minimum 0.25 |
| C b=1e+04 lambda=1 | 2.83e-15 | 3.13e-13 | 0.009 | True | True | 36 | 34 | 1 | minimum 1 |
| C b=1e+08 lambda=0.25 | 1.95e-15 | 3.07e-14 | 0.064 | True | True | 47 | 35 | 5 | minimum 0.25 |
| C b=1e+08 lambda=1 | 1.61e-15 | 1.64e-14 | 0.098 | True | True | 71 | 37 | 7 | minimum 1 |

## Phi_w = 128pi: Lambda_c by the rule = 4.0; the check holds: True

| case | true error | abserr | err/abserr | honest | converged | regions | window pieces | max depth | located (kind, lambda) |
|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 2.43e-17 | 7.42e-13 | 3.3e-05 | True | True | 100 | 99 | 1 | minimum 0.25 |
| C b=1e+04 lambda=1 | 1.79e-16 | 6.72e-13 | 0.00027 | True | True | 106 | 105 | 1 | minimum 1 |
| C b=1e+08 lambda=0.25 | 3.32e-17 | 3.02e-14 | 0.0011 | True | True | 149 | 135 | 6 | minimum 0.25 |
| C b=1e+08 lambda=1 | 8.28e-16 | 2.01e-14 | 0.041 | True | True | 149 | 134 | 7 | minimum 1 |

## Phi_w = 512pi: Lambda_c by the rule = 4.0; the check holds: True

| case | true error | abserr | err/abserr | honest | converged | regions | window pieces | max depth | located (kind, lambda) |
|---|---|---|---|---|---|---|---|---|---|
| C b=1e+04 lambda=0.25 | 4.23e-16 | 2.78e-13 | 0.0015 | True | True | 166 | 166 | 2 | minimum 0.25 |
| C b=1e+04 lambda=1 | 6.26e-16 | 3.17e-13 | 0.002 | True | True | 162 | 162 | 1 | minimum 1 |
| C b=1e+08 lambda=0.25 | 1.27e-16 | 3.25e-14 | 0.0039 | True | True | 779 | 770 | 4 | minimum 0.25 |
| C b=1e+08 lambda=1 | 7.19e-16 | 2.28e-14 | 0.032 | True | True | 787 | 762 | 8 | minimum 1 |
