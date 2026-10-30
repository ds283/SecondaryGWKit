# 06d's round-off floor on 07's (R) cases: term by term

tables.py run: worktree HEAD `e5764a6708757cb11704b95bcd66825baab4ba29`, dirty False, levin_quadrature.py sha256 `ed0648dcbff012e11f32acd8bb0baccd804c8ec24368ee87577f670415440cd1`, host mps026088.phys.susx.ac.uk, 2026-10-29T17:33:29.846575+00:00, numpy 2.2.4, scipy 1.16.3, mpmath 1.3.0.

Inputs: driver_run.json (header: HEAD `e5764a6708757cb11704b95bcd66825baab4ba29`, dirty False, 2026-10-29T17:21:24.689891+00:00), node_chain.json (2026-10-29T17:22:32.372486+00:00), scaling.json.

Script sha256: check_scheme.py `add9e0381b1a8d4702f458bb14aee2212b5bfef00c0583f61f8565937ed9bb3e`, common.py `13cdad64547fb5ddc14e313dc8dd7d2f2dfeecb10aef4b9223c046472b5c261d`, driver_run.py `f6ccb1995654606c3d45f1667a8472b4ef6ec784bb98ffc2f84d079a411be6c8`, levin_node_attribution.py `bf35fe4a6f15149f4d5701c233e4561f5a9fc68ee976f3f9802011d2c9566dad`, node_chain.py `3f5498e4636424ce7eb02dc2df4ed4a097f971fe4456014774f636193991cb93`, probe_levin_leaf.py `10b485435004ac578290920b1cee75637a09ae156a020238b3a8f9267a79e084`, scaling.py `df2e7549cfa31b53d56b6d91419ebceec3de5f0f9d62a9b80f4b8564610458a0`, show_leaves.py `20e561ccd25ec8925d57f59d5b185120579e97d23416538472c322f15d9e5eb7`, tables.py `7ad4483b1ec25c4fd35a5f68996e68f1620320c42b169c68ed04eac2efc2f431`

07's scripts (read, not modified) sha256: calibrate.py `0b316b6d0fbdd021b2be20c54b6dc96914824ed974c48992a551c7391447e0e9`, lg_location.py `4d42a22dc1936d4cec927219cd910598a2cedb3d967e3b733065e2353e3f4424`, lg_repeat.py `f6651cf53e0b7af1e582ea0852652e3c4ac1a406731f8ddbf488e6f850320925`, parent_compare.py `03fca21d4d4cb5275092d1034ac320b280ac6a77d24836e0470f065524ab243a`, prov.py `5860fe90ab5026409473858119f16c40b8ba42d898451c68d19a2988f0b05ebd`, reference.py `77617ac573307985dff4409793f4131a0ab19741703283826c2655e888feaeba`, stop4_diagnose.py `5a5e6ccf4ff19ff7a36d7f6324837a6a4273e126681c06036de24509cb3b756e`, stop4_lg_seed0.py `b313bf83a5f081a13e5967e002e3b29dc5bd8074491924207bc031745a2e2ef2`, validate_reference.py `9c791c8497f9fdf269b2a1a48c7fa83d20ca236e2c2f4f96ebef469572626d9c`, validate_reference_precision.py `62f139f5230092e373aac28cc2fd7666dd6e78b0c62cfe9ca59f3251731a7412`

## 1-2. The driver's own run at 06d: the floor's terms summed over the leaves, against the round-off

| | c=1e6 w=0 | c=1e8 w=0 | c=1e8 w=1 |
|---|---|---|---|
| leaves (Levin / CC), all floor-limited | 15 / 16, 31 | 18 / 45, 63 | 18 / 51, 69 |
| request max(atol, rtol abs(V)) | 1.43e-14 | 1.41e-15 | 1.13e-14 |
| **declared abserr** | 3.75e-14 | 1.33e-13 | 1.50e-13 |
| **true error** (V - closed form) | 8.87e-17 | 1.77e-17 | 1.61e-17 |
| declared / true | 422 | 7523 | 9279 |
| floor term L1_eps_f_h2, summed | 5.52e-17 | 5.59e-17 | 5.59e-17 |
| floor term L2_G1_over_G0, summed | 1.51e-16 | 1.60e-16 | 1.60e-16 |
| floor term L3_maxG1k2_over_G0, summed | 1.58e-16 | 1.60e-16 | 1.60e-16 |
| floor term C1_n4_uS, summed | 4.55e-17 | 8.85e-18 | 9.52e-18 |
| floor term C2_argument, summed | 2.64e-17 | 2.03e-17 | 2.43e-17 |
| floor term C3_node, summed | 3.70e-14 | 1.33e-13 | 1.49e-13 |
| C3_node share of the floor | 0.9884 | 0.9970 | 0.9973 |
| actual round-off, sum abs per leaf | 1.97e-16 | 4.15e-16 | 5.56e-16 |
| actual round-off, abs of sum | 1.03e-16 | 1.63e-17 | 1.52e-17 |
|   CC node rounding: sum abs / abs sum | 1.74e-16 / 1.01e-16 | 3.61e-16 / 4.20e-18 | 4.98e-16 / 1.53e-18 |
|   Levin endpoint-phase rounding: sum abs / abs sum | 2.30e-17 / 3.63e-18 | 5.39e-17 / 2.05e-17 | 5.73e-17 / 1.34e-17 |
|   the rest (solve, sums, f, weights): sum abs / abs sum | 3.91e-18 / 1.44e-18 | 7.89e-19 / 4.70e-21 | 1.42e-18 / 2.95e-19 |
| truncation (mp scheme - exact): sum abs / abs sum | 1.57e-17 / 1.44e-17 | 1.61e-18 / 1.34e-18 | 1.50e-18 / 9.39e-19 |
| C3 / sum abs(node part) | 212 | 367 | 300 |
| C3 / abs(sum node part) | 367 | 31562 | 97373 |
| leaves with abs(round-off) > own floor | 0 | 0 | 2 |

### The CC node term's pessimism, factor by factor (sums over the driver's CC leaves)

A = C3 as coded; B = A with the bound delta = 5u(max(|a|,|b|) + h) replaced by each node's actual |x^_k - x_k|; C = B with max theta' replaced by theta'(x_k); D = sum |w||dx||g'| with the true g'; E = sum w dx g' (signed, first order); R = measured node part.

| | c=1e6 w=0 | c=1e8 w=0 | c=1e8 w=1 |
|---|---|---|---|
| A / B (bound on dx vs actual dx) | 14.1 | 21.5 | 21.7 |
| B / C (max theta' vs theta' at the node) | 1.13 | 1.07 | 1.06 |
| C / D (theta' sum abs f + sum abs f' vs abs g') | 2.23 | 2.19 | 2.22 |
| D / sum abs E (cancellation inside a leaf) | 6.0 | 7.3 | 5.9 |
| sum abs E / abs sum E (cancellation across leaves) | 1.7 | 86.1 | 325.0 |
| E / R (first-order model vs measured; summed) | 1.000000 | 1.000000 | 1.000000 |
| max abs(dx) over all CC nodes, in u | 0.480 | 0.490 | 0.490 |
| the bound delta, in u | 3.638 | 3.615 | 3.616 |

## 3. Scaling with h: [0, 1] cut into 2^d equal leaves, each solved by 06d's subregion solve

### c=1e6 omega=0

| d | h | Levin / CC | summed floor | L1 sum | L2 sum | L3 sum | C1 sum | C2 sum | C3 sum | med floor Levin | med floor CC | med L3 | med C3 | med abs(ro) Levin (rest) | med abs(ro) CC (node) | sum abs(ro) | abs(sum ro) | leaves ro>floor (Levin rest>floor) | worst Levin rest/floor | node pattern(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1.6e-02 | 64 / 0 | 2.02e-16 | 4.65e-17 | 6.47e-17 | 9.07e-17 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.44e-18 | -- | 8.43e-19 | -- | 4.46e-18 (5.15e-22) | -- (--) | 5.83e-16 | 1.39e-17 | 44 (0) | 0.2 | 0 |
| 8 | 3.9e-03 | 254 / 2 | 1.31e-14 | 4.58e-17 | 4.75e-17 | 7.75e-17 | 2.00e-17 | 2.71e-18 | 1.29e-14 | 6.02e-19 | 6.47e-15 | 2.06e-19 | 6.46e-15 | 3.55e-18 (5.64e-22) | 2.97e-17 (2.97e-17) | 2.09e-15 | 5.95e-17 | 211 (0) | 0.31 | 1 |
| 10 | 9.8e-04 | 984 / 40 | 1.70e-13 | 4.46e-17 | 4.48e-17 | 1.35e-16 | 1.01e-16 | 2.89e-16 | 1.69e-13 | 1.88e-19 | 4.23e-15 | 6.87e-20 | 4.22e-15 | 4.04e-18 (6.68e-22) | 1.55e-17 (1.54e-17) | 9.48e-15 | 3.23e-16 | 906 (0) | 0.94 | 1 |
| 12 | 2.4e-04 | 3464 / 632 | 2.54e-12 | 4.03e-17 | 4.03e-17 | 3.14e-16 | 3.99e-16 | 1.77e-14 | 2.52e-12 | 8.60e-20 | 3.88e-15 | 6.13e-20 | 3.86e-15 | 5.04e-18 (1.37e-21) | 2.58e-18 (2.28e-18) | 3.81e-14 | 2.51e-16 | 3400 (0) | 0.48 | 1 |
| 14 | 6.1e-05 | 6796 / 9588 | 3.67e-11 | 2.33e-17 | 2.33e-17 | 3.83e-16 | 1.48e-15 | 9.46e-13 | 3.58e-11 | 6.40e-20 | 2.51e-15 | 5.70e-20 | 2.43e-15 | 7.99e-18 (4.96e-21) | 4.92e-18 (2.10e-18) | 1.82e-13 | 4.42e-16 | 6749 (0) | 0.51 | 2 |

### c=1e8 omega=0

| d | h | Levin / CC | summed floor | L1 sum | L2 sum | L3 sum | C1 sum | C2 sum | C3 sum | med floor Levin | med floor CC | med L3 | med C3 | med abs(ro) Levin (rest) | med abs(ro) CC (node) | sum abs(ro) | abs(sum ro) | leaves ro>floor (Levin rest>floor) | worst Levin rest/floor | node pattern(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1.6e-02 | 64 / 0 | 1.76e-16 | 4.65e-17 | 6.47e-17 | 6.47e-17 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.44e-18 | -- | 8.43e-19 | -- | 4.29e-18 (4.74e-24) | -- (--) | 5.46e-16 | 2.06e-17 | 42 (0) | 5.2e-05 | 0 |
| 8 | 3.9e-03 | 256 / 0 | 1.76e-16 | 4.61e-17 | 6.49e-17 | 6.49e-17 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 5.55e-19 | -- | 1.87e-19 | -- | 4.09e-18 (5.58e-24) | -- (--) | 2.12e-15 | 2.07e-17 | 223 (0) | 0.0064 | 0 |
| 10 | 9.8e-04 | 1024 / 0 | 1.63e-16 | 4.60e-17 | 5.01e-17 | 6.66e-17 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.34e-19 | -- | 4.50e-20 | -- | 3.29e-18 (5.04e-24) | -- (--) | 8.00e-15 | 6.24e-17 | 977 (1) | 3.2 | 0 |
| 12 | 2.4e-04 | 4090 / 6 | 3.03e-14 | 4.60e-17 | 4.61e-17 | 5.00e-17 | 3.78e-18 | 1.53e-18 | 3.02e-14 | 3.39e-20 | 5.03e-15 | 1.14e-20 | 5.03e-15 | 3.61e-18 (5.30e-24) | 1.53e-18 (1.53e-18) | 3.39e-14 | 1.82e-17 | 4021 (11) | 16 | 1 |
| 14 | 6.1e-05 | 16283 / 101 | 4.09e-13 | 4.58e-17 | 4.58e-17 | 6.05e-17 | 1.60e-17 | 1.14e-16 | 4.09e-13 | 8.93e-21 | 4.02e-15 | 3.05e-21 | 4.01e-15 | 4.08e-18 (5.48e-24) | 2.44e-18 (2.44e-18) | 1.41e-13 | 6.13e-17 | 16159 (7) | 1.3 | 1 |

### c=1e8 omega=1

| d | h | Levin / CC | summed floor | L1 sum | L2 sum | L3 sum | C1 sum | C2 sum | C3 sum | med floor Levin | med floor CC | med L3 | med C3 | med abs(ro) Levin (rest) | med abs(ro) CC (node) | sum abs(ro) | abs(sum ro) | leaves ro>floor (Levin rest>floor) | worst Levin rest/floor | node pattern(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | 1.6e-02 | 64 / 0 | 1.76e-16 | 4.65e-17 | 6.47e-17 | 6.47e-17 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 2.44e-18 | -- | 8.43e-19 | -- | 5.58e-18 (5.33e-24) | -- (--) | 5.77e-16 | 1.33e-17 | 43 (0) | 3.6e-05 | 0 |
| 8 | 3.9e-03 | 256 / 0 | 1.76e-16 | 4.61e-17 | 6.48e-17 | 6.48e-17 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 5.55e-19 | -- | 1.87e-19 | -- | 3.56e-18 (5.37e-24) | -- (--) | 2.17e-15 | 1.35e-17 | 224 (0) | 0.048 | 0 |
| 10 | 9.8e-04 | 1024 / 0 | 1.63e-16 | 4.60e-17 | 5.01e-17 | 6.66e-17 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 1.34e-19 | -- | 4.50e-20 | -- | 3.45e-18 (5.06e-24) | -- (--) | 8.12e-15 | 5.86e-17 | 966 (2) | 2.1 | 0 |
| 12 | 2.4e-04 | 4090 / 6 | 3.03e-14 | 4.60e-17 | 4.61e-17 | 5.00e-17 | 3.82e-18 | 1.66e-18 | 3.02e-14 | 3.39e-20 | 5.03e-15 | 1.14e-20 | 5.03e-15 | 3.63e-18 (5.26e-24) | 4.62e-18 (4.62e-18) | 3.41e-14 | 1.59e-17 | 4031 (10) | 13 | 1 |
| 14 | 6.1e-05 | 16283 / 101 | 4.09e-13 | 4.58e-17 | 4.58e-17 | 6.05e-17 | 1.60e-17 | 1.14e-16 | 4.09e-13 | 8.93e-21 | 4.02e-15 | 3.05e-21 | 4.01e-15 | 4.01e-18 (5.36e-24) | 2.68e-18 (2.74e-18) | 1.41e-13 | 3.94e-17 | 16162 (6) | 1.5 | 1 |

### Local power of h of each summed term, sum ~ h^p, between consecutive depths

| case | d -> d' | L1_eps_f_h2 | L2_G1_over_G0 | L3_maxG1k2_over_G0 | C3_node | summed floor | abs(sum ro) |
|---|---|---|---|---|---|---|---|
| c=1e6 omega=0 | 6 -> 8 | +0.01 | +0.22 | +0.11 | -- | -3.01 | -1.05 |
| c=1e6 omega=0 | 8 -> 10 | +0.02 | +0.04 | -0.40 | -1.86 | -1.85 | -1.22 |
| c=1e6 omega=0 | 10 -> 12 | +0.07 | +0.08 | -0.61 | -1.95 | -1.95 | +0.18 |
| c=1e6 omega=0 | 12 -> 14 | +0.40 | +0.40 | -0.14 | -1.91 | -1.93 | -0.41 |
| c=1e8 omega=0 | 6 -> 8 | +0.01 | -0.00 | -0.00 | -- | +0.00 | -0.01 |
| c=1e8 omega=0 | 8 -> 10 | +0.00 | +0.19 | -0.02 | -- | +0.06 | -0.80 |
| c=1e8 omega=0 | 10 -> 12 | +0.00 | +0.06 | +0.21 | -- | -3.77 | +0.89 |
| c=1e8 omega=0 | 12 -> 14 | +0.00 | +0.00 | -0.14 | -1.88 | -1.88 | -0.88 |
| c=1e8 omega=1 | 6 -> 8 | +0.01 | -0.00 | -0.00 | -- | +0.00 | -0.01 |
| c=1e8 omega=1 | 8 -> 10 | +0.00 | +0.19 | -0.02 | -- | +0.06 | -1.06 |
| c=1e8 omega=1 | 10 -> 12 | +0.00 | +0.06 | +0.21 | -- | -3.77 | +0.94 |
| c=1e8 omega=1 | 12 -> 14 | +0.00 | +0.00 | -0.14 | -1.88 | -1.88 | -0.66 |

