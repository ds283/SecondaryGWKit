levin-phase-floor 07d Part A: D-N, the integrand's noise
written 2026-11-22T10:27:45Z on mps026298.phys.susx.ac.uk
script d_n_table_07d.py sha256 038a4472dc4f2f76834d61b5b692cb0b6e7b4489915b873ef7252c2d5e89c3bc
tree h: /Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-49ce964 HEAD 49ce96429da797efe55f546d65a23ebcd0099f33 (status --short: empty)
main checkout: /Users/ds283/Documents/Code/SecondaryGWKit HEAD 28ebaf0743e0555fb2c4512ce714884005a60eff (status --short: empty)
input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/out/leaves/fill_state.json sha256 266ad8af9284955fc76dcf3d1b86dbc6e4fb11103bec0e8a426b05d56f8d54c4
input /Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/state/selection_leaves_07d.json sha256 c4c01873c2eb874be25ac4d51f3fac6562b0e10b9c6e39fb827d2beb04ba8327

Leaves: 1442 (every Clenshaw-Curtis stratum's fill prefix).

## by all

| class | n | chopped (fine) | sigma/mean|g| median (p10-p90) | sigma(d_lo)/sigma(d_hi) | plateau47/LS | plateau185/LS | theta_raw sigma / u|theta| | phase-pred/sigma_g | amp-pred/sigma_g | acf1 g | acf1 theta | shift spread / s_CC | declared/sigma_g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 1442 | 932 | 1.47e-12 (3.2e-15-7.1e-12) | 1 | 1.08 | 1.08 | 0.811 | 1.43 | 0.00965 | -0.00765 | -0.00555 | 0.789 | 4.1 |

## by kind

| class | n | chopped (fine) | sigma/mean|g| median (p10-p90) | sigma(d_lo)/sigma(d_hi) | plateau47/LS | plateau185/LS | theta_raw sigma / u|theta| | phase-pred/sigma_g | amp-pred/sigma_g | acf1 g | acf1 theta | shift spread / s_CC | declared/sigma_g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| analytic | 448 | 314 | 5.61e-15 (1.3e-15-9.6e-15) | 1 | 1.7 | 3.52 | 6.56 | 1.58 | 0.449 | 0.0292 | 0.183 | 0.61 | 4.1 |
| group | 994 | 618 | 2.75e-12 (3e-14-9.5e-12) | 1 | 0.98 | 1.02 | 0.69 | 1.42 | 0.00124 | -0.0164 | -0.0161 | 0.862 | - |

## by label

| class | n | chopped (fine) | sigma/mean|g| median (p10-p90) | sigma(d_lo)/sigma(d_hi) | plateau47/LS | plateau185/LS | theta_raw sigma / u|theta| | phase-pred/sigma_g | amp-pred/sigma_g | acf1 g | acf1 theta | shift spread / s_CC | declared/sigma_g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| analytic J1 | 11 | 6 | 9.33e-15 (7.5e-15-1.9e-14) | 1 | 1.19 | 2 | 10.3 | 2.67 | 0.689 | 0.031 | 0.464 | 0.449 | 135 |
| analytic J2 | 83 | 53 | 4.39e-15 (1.6e-15-7.7e-15) | 0.998 | 1.89 | 4.03 | 6.81 | 1.63 | 0.664 | 0.0515 | 0.256 | 0.738 | 2.98 |
| analytic J3 | 50 | 22 | 5.8e-15 (1.6e-15-1.1e-14) | 1 | 1.78 | 3.36 | 9.31 | 1.74 | 0.446 | 0.0508 | 0.367 | 0.351 | 2.84 |
| analytic J4 | 96 | 77 | 4.97e-15 (1.1e-15-9e-15) | 1 | 1.71 | 3.95 | 6.15 | 1.47 | 0.739 | 0.0345 | 0.11 | 0.892 | 11.1 |
| analytic Y1 | 25 | 17 | 8.08e-15 (3.9e-15-1.5e-14) | 0.999 | 1.25 | 2.5 | 6.85 | 1.52 | 0.165 | 0.00549 | 0.105 | 0.913 | 83.5 |
| analytic Y2 | 55 | 37 | 6.43e-15 (3.1e-15-9.5e-15) | 0.999 | 1.64 | 3.09 | 6.13 | 1.57 | 0.25 | 0.00906 | 0.181 | 0.24 | 170 |
| analytic Y3 | 59 | 43 | 6.43e-15 (2.8e-15-9.9e-15) | 0.999 | 1.39 | 3.21 | 6.07 | 1.56 | 0.283 | 0.0264 | 0.174 | 0.264 | 2.62 |
| analytic Y4 | 69 | 59 | 2.01e-15 (6e-16-7.5e-15) | 1 | 2.78 | 8.1 | 5.49 | 1.47 | 0.517 | 0.0166 | 0.0689 | 1.16 | 599 |
| group G | 59 | 0 | 1.3e-14 (2.2e-15-5.6e-14) | 1.01 | - | 6.17e+05 | 0.541 | 0.476 | 1.14 | -0.0495 | -0.0654 | 2.06e+05 | - |
| group G+q+r | 343 | 189 | 3.84e-12 (5.9e-14-7.2e-09) | 1 | 1.01 | 1.02 | 0.702 | 1.42 | 0.000615 | -0.0146 | -0.0143 | 0.83 | - |
| group G+q-r | 136 | 98 | 3.07e-12 (4.9e-13-4.7e-12) | 1 | 0.936 | 0.991 | 0.616 | 1.42 | 0.0014 | -0.0104 | -0.0123 | 0.732 | - |
| group G+r | 18 | 7 | 1.65e-13 (2.2e-14-5.1e-12) | 1 | 1.15 | 1.07 | 0.795 | 1.43 | 0.019 | -0.0329 | -0.0406 | 46.5 | - |
| group G-q+r | 183 | 138 | 2.6e-12 (4.9e-14-7.2e-12) | 1 | 0.993 | 1.01 | 0.64 | 1.43 | 0.0011 | -0.0172 | -0.0155 | 0.477 | - |
| group G-q-r | 211 | 171 | 2.66e-12 (7.7e-13-7.1e-12) | 1 | 0.943 | 1.03 | 0.724 | 1.42 | 0.000739 | -0.0169 | -0.016 | 0.541 | - |
| group G-r | 42 | 15 | 2.48e-13 (5.9e-14-1.2e-12) | 1 | 1.06 | 1.12 | 0.818 | 1.41 | 0.0539 | -0.0296 | -0.0257 | 80.8 | - |
| group q+r | 2 | 0 | 4.1e-14 (4.1e-14-4.1e-14) | 1.01 | - | - | 75.2 | 1.5 | 0.122 | -0.239 | -0.24 | 3.95e+04 | - |

## by k_band

| class | n | chopped (fine) | sigma/mean|g| median (p10-p90) | sigma(d_lo)/sigma(d_hi) | plateau47/LS | plateau185/LS | theta_raw sigma / u|theta| | phase-pred/sigma_g | amp-pred/sigma_g | acf1 g | acf1 theta | shift spread / s_CC | declared/sigma_g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| analytic|k0 | 128 | 95 | 5.44e-15 (1.6e-15-8.5e-15) | 1 | 1.71 | 3.57 | 6.48 | 1.53 | 0.54 | 0.0354 | 0.2 | 0.913 | 2.96 |
| analytic|k1 | 92 | 70 | 5.3e-15 (1.2e-15-9.8e-15) | 0.999 | 1.67 | 3.91 | 6.33 | 1.53 | 0.396 | 0.0288 | 0.14 | 0.563 | 143 |
| analytic|k2 | 103 | 75 | 5.8e-15 (1.2e-15-1.3e-14) | 1 | 1.8 | 3.23 | 8.59 | 1.62 | 0.449 | 0.0158 | 0.112 | 0.61 | 3.1 |
| analytic|k3 | 125 | 74 | 5.75e-15 (1.6e-15-8.3e-15) | 0.999 | 1.7 | 3.52 | 6.75 | 1.75 | 0.442 | 0.0364 | 0.263 | 0.271 | 5.17 |
| group|k0 | 191 | 111 | 2.6e-12 (8.1e-15-8.3e-12) | 1 | 0.937 | 1.01 | 0.607 | 1.4 | 0.00164 | -0.0185 | -0.0191 | 1.09 | - |
| group|k1 | 313 | 205 | 3.35e-12 (1.1e-13-1.3e-11) | 1 | 0.986 | 1.02 | 0.677 | 1.42 | 0.000657 | -0.0131 | -0.014 | 0.82 | - |
| group|k2 | 202 | 106 | 2.47e-12 (4.5e-14-5.7e-11) | 1 | 1 | 1.04 | 0.73 | 1.42 | 0.00299 | -0.0205 | -0.0179 | 0.679 | - |
| group|k3 | 288 | 196 | 2.56e-12 (2.4e-14-5.6e-12) | 1 | 0.973 | 1.01 | 0.713 | 1.41 | 0.00171 | -0.0175 | -0.0193 | 1.02 | - |

## by route

| class | n | chopped (fine) | sigma/mean|g| median (p10-p90) | sigma(d_lo)/sigma(d_hi) | plateau47/LS | plateau185/LS | theta_raw sigma / u|theta| | phase-pred/sigma_g | amp-pred/sigma_g | acf1 g | acf1 theta | shift spread / s_CC | declared/sigma_g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| analytic|chase | 48 | 46 | 5.75e-15 (2.7e-15-9.1e-15) | 0.999 | 1.39 | 3.35 | 5.47 | 1.53 | 0.277 | 0.0178 | 0.114 | 0.391 | 3.36 |
| analytic|gate | 320 | 192 | 6.12e-15 (1.7e-15-1e-14) | 1 | 1.67 | 3.26 | 6.81 | 1.64 | 0.433 | 0.0323 | 0.257 | 0.461 | 2.98 |
| analytic|window | 80 | 76 | 1.69e-15 (6.4e-16-8.2e-15) | 1 | 3.1 | 8.85 | 2.73 | 1.11 | 0.798 | 0.0197 | 0.0102 | 1.2 | 859 |
| group|chase | 178 | 147 | 2.86e-12 (1.5e-12-5.5e-12) | 1 | 0.986 | 1.02 | 0.697 | 1.42 | 0.00143 | -0.014 | -0.0144 | 0.196 | - |
| group|gate | 713 | 454 | 2.54e-12 (3e-14-7e-12) | 1 | 0.978 | 1.02 | 0.705 | 1.42 | 0.00145 | -0.0177 | -0.0176 | 0.958 | - |
| group|window | 103 | 17 | 3.32e-10 (4.4e-15-6.1e-08) | 1 | 0.941 | 0.991 | 0.644 | 1.07 | 2.93e-05 | -0.01 | -0.0139 | 1.26 | - |

## by knots

| class | n | chopped (fine) | sigma/mean|g| median (p10-p90) | sigma(d_lo)/sigma(d_hi) | plateau47/LS | plateau185/LS | theta_raw sigma / u|theta| | phase-pred/sigma_g | amp-pred/sigma_g | acf1 g | acf1 theta | shift spread / s_CC | declared/sigma_g |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| analytic|0 | 189 | 132 | 3.89e-15 (1.2e-15-8e-15) | 1 | 2.06 | 4.45 | 6.81 | 1.6 | 0.54 | 0.035 | 0.227 | 0.647 | 2.97 |
| analytic|1 | 23 | 23 | 8.81e-15 (5.8e-15-1.5e-14) | 0.999 | 0.914 | 2 | 8.82 | 0.396 | 1.91 | 0.493 | 0.0604 | 1.12 | 177 |
| analytic|2-4 | 24 | 3 | 7.45e-15 (4.4e-15-9.1e-15) | 1 | 1.27 | 2.49 | 22.1 | 3.37 | 1.26 | 0.277 | 0.887 | 0.449 | 210 |
| analytic|5+ | 4 | 0 | 5.3e-15 (5.1e-15-6.6e-15) | 1 | - | 3.51 | 6.07 | 2.6 | 0.644 | 0.0323 | 0.682 | 1.2 | 296 |
| analytic|unknown | 208 | 156 | 6.12e-15 (1.2e-15-1e-14) | 1 | 1.67 | 3.29 | 6.18 | 1.55 | 0.3 | 0.0108 | 0.135 | 0.535 | 6.98 |
| group|0 | 532 | 406 | 4.07e-12 (1.8e-12-1.6e-10) | 1 | 0.975 | 1.01 | 0.704 | 1.42 | 0.000447 | -0.012 | -0.0135 | 0.567 | - |
| group|1 | 258 | 203 | 2.5e-12 (1.1e-12-4.9e-12) | 1 | 0.994 | 1.03 | 0.703 | 1.42 | 0.00575 | -0.0123 | -0.0136 | 0.316 | - |
| group|2-4 | 42 | 9 | 1.32e-13 (1.6e-14-1.5e-12) | 1 | 1.22 | 1.22 | 0.723 | 1.39 | 0.0781 | -0.0189 | -0.0202 | 117 | - |
| group|5+ | 162 | 0 | 2.81e-14 (2.3e-15-1.2e-13) | 1.01 | - | 8.58e+04 | 0.636 | 1.4 | 0.1 | -0.0912 | -0.107 | 1.08e+04 | - |

