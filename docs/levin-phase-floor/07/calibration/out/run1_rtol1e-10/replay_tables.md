# 07 section 2.3: the real calls replayed

    levin-phase-floor 07: section 2.3, the real calls replayed (run)
    written 2026-11-07T09:36:35+00:00 on mps026088.phys.susx.ac.uk
    python 3.13.16 (/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python)
    script replay.py sha256 43d8d7ea5d1b59da515748cb93348987512a572462f8f9b15e073a504cb45133
    prov.py sha256 9e2adc86dd624417822eb68ef554a6e5c23c817a824a8f778df712055d449c76
    AdaptiveLevin/levin_quadrature.py sha256 13294b2ac3e3e3a480a6735a263ea8abe205ccdaada12939c13cd20edad884a4
    input docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json sha256 73c18680d266b1a0fdf11123c104d20e822658047dd7c36e1c814db9ffcad912
    input docs/levin-phase-floor/06g/references/out/capture.json sha256 62b91d27bbd6a4e6a5fc10f64e212e4a491baeb043c1651e1722b0ebf5922a49
    input docs/levin-phase-floor/06g/references/v1/out/v1_capture.json sha256 71e5b9da1ec4b9b4ee465c0cd0531a34d48ba46f26fb47bd4cf417af9a9ad19a
    input docs/levin-phase-floor/06l/out/partC/h1_calls.csv sha256 641d497a3451ea24f614a9ce5a598cd382c8a646b65ca42df9561acccc764270
    input docs/levin-phase-floor/07/calibration/scripts/analysis07.py sha256 1f2999bfb671bd564c3b4669214ebad490cc1579c06fcc09e996d4343f850239
    input docs/levin-phase-floor/07/calibration/scripts/variant_loader.py sha256 3206d8f892cdc31bdcd1ce8be719d1dc1c4361a294fc5c86dad8bd043f2d8b33
    input docs/levin-phase-floor/07/calibration/scripts/calibrate.py sha256 4f4290c5d7de782658dcc4cab48c80cd734f2f6bcdbfa36960d1ca478e3b72d9
    input var/levin-phase-floor-06g-fcc-laptop/inputs/z12.pkl sha256 7b8792e57d4aa769dcd16dbaea2a107c17b139a0d4aea63d0dc2cf20cb7f4882
    input var/levin-phase-floor-06g-fcc-laptop/inputs/z00.pkl sha256 1b870b17582834a0a066412b636919fa0951331e514a9118bc76dc18fa8c6114
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w00.pkl sha256 1012bf434706cb008547fe3c5e4c854f6943130f70224ae93d18f6ee88ed4242
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w01.pkl sha256 cc57610233d2513d71817b81ae968159b75a030613c95767156a5d4f08a93152
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w02.pkl sha256 84e7775c0c378c0a919e06cdc93058592320f9eba6cbeac7b17d90c481d0851d
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w03.pkl sha256 00ca5cc90aede4dcdbe8d49b36acab696aa943334a417207ddc243f11928fe9c
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w04.pkl sha256 b1f33ae9c0fd7d62e28f3b6a7ceca2eb3616990dac666a9b64c8dc6aa3c38d63
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w05.pkl sha256 5d780514a4593046a2697a71e885b5c5f5701bd9155e499a07de685331af80f7
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w06.pkl sha256 7e86a8e9d16a6acde7ef3a8b3020d8b7420f853f361c61d54fcbca7b1dd90dcb
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w07.pkl sha256 88a11cf7575f07b5c0294c2d3b0b6b8dae84f8b67c93254a0dd14fcdb373d2db
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w08.pkl sha256 34d3f38a4c750de96570dc99c7850cf9a799bb83a406559ebd38f895c3753803
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w09.pkl sha256 825b510465594c256fafc9020d1c0affd7ff852ca186d7346a28307894a610cc
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w10.pkl sha256 fe99a2bdf8ffa18e4a75d70de794f0cc1e2e1a1b63862ddd3f84ea9e08bdb28b
    input var/levin-phase-floor-06g-fcc-laptop/inputs/w11.pkl sha256 e7e441399564f9d3ac6f2f5a34f11bf6b350f0ff6ecead37b3379c74a425ae15
    input var/levin-phase-floor-06g-fcc-laptop/inputs/r55.pkl sha256 f0df8234e588c53168407d09ee8d8fbbbd7e3565ab8e605685bef9aecc20bd9e
    HEAD bd21bf23da006aa21c5aa670cf32ea2ed726cb6b
    tree dirty: yes; git status --short:
         M AdaptiveLevin/levin_quadrature.py
         M ComputeTargets/QuadSourceIntegral.py
        ?? docs/levin-phase-floor/07/
    uptime at start: 9:36  up 7 days, 11:57, 1 user, load averages: 185.40 191.93 148.09

06g's tree: /private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/194e48fe-b717-47ef-90f9-bf58f0c23da9/scratchpad/agent07/wt-g06-3ae6f3a4 HEAD 3ae6f3a499a2c7c9d0b38d1f418cc8fe85e66054 (status --short: empty)

## The replay check

| call | 06g (3ae6f3a4) reproduces | differences | parent (route B) reproduces | differences |
|---|---|---|---|---|
| z12 | True | {} | True | {} |
| z00 | True | {} | True | {} |
| w00 | True | {} | no record |  |
| w01 | True | {} | no record |  |
| w02 | True | {} | no record |  |
| w03 | True | {} | no record |  |
| w04 | True | {} | no record |  |
| w05 | True | {} | no record |  |
| w06 | True | {} | no record |  |
| w07 | True | {} | no record |  |
| w08 | True | {} | no record |  |
| w09 | True | {} | no record |  |
| w10 | True | {} | no record |  |
| w11 | True | {} | no record |  |
| r55 | True | {} | True | {} |

**All reproduce:** True

## Every call and rung

| call | mode | |v - ref| / abserr | honest | converged | node_limited | regions | max depth | abserr/request | window pieces | located (kind x) | windows (lo, hi, pieces) | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| z12 | g06 | 1681 | False | True | False | 135 | 18 | 0.952 | -- |  |  | 6.9 |
| z12 | parent | 1681 | False | True | False | 112 | 17 | 0.952 | -- |  |  | 5.3 |
| z12 | barlifted 128pi | did not finish or error | | | | | | | | | | |
| z12 | barlifted 2pi | did not finish or error | | | | | | | | | | |
| z12 | barlifted 32pi | did not finish or error | | | | | | | | | | |
| z12 | barlifted 512pi | did not finish or error | | | | | | | | | | |
| z12 | barlifted 8pi | did not finish or error | | | | | | | | | | |
| z12 | commit 128pi | did not finish or error | | | | | | | | | | |
| z12 | commit 128pi depth_max 12 | 0.9084 | True | False | False | 9757 | 12 | 0.407 | 9715 | root 5.636713876 | 5.6347453, 5.6386834, 128 | 437.7 |
| z12 | commit 128pi depth_max 8 | 0.9084 | True | False | False | 2755 | 8 | 0.407 | 2723 | root 5.636713876 | 5.6347453, 5.6386834, 128 | 93.8 |
| z12 | commit 2pi | did not finish or error | | | | | | | | | | |
| z12 | commit 2pi depth_max 12 | 0.7179 | True | False | False | 943 | 12 | 0.52 | 892 | root 5.636713876 | 5.6364678, 5.63696, 2 | 30.8 |
| z12 | commit 2pi depth_max 8 | 1.749 | False | False | False | 632 | 8 | 15.3 | 512 | root 5.636713876 | 5.6364678, 5.63696, 2 | 5.0 |
| z12 | commit 32pi | did not finish or error | | | | | | | | | | |
| z12 | commit 32pi depth_max 12 | 0.7623 | True | False | False | 5747 | 12 | 0.489 | 5651 | root 5.636713876 | 5.6357295, 5.6376985, 32 | 227.2 |
| z12 | commit 32pi depth_max 8 | 0.7626 | True | False | False | 1201 | 8 | 0.488 | 1163 | root 5.636713876 | 5.6357295, 5.6376985, 32 | 37.6 |
| z12 | commit 512pi | did not finish or error | | | | | | | | | | |
| z12 | commit 512pi depth_max 12 | did not finish or error | | | | | | | | | | |
| z12 | commit 512pi depth_max 8 | 0.9299 | True | False | False | 7905 | 8 | 0.393 | 7877 | root 5.636713876 | 5.6327776, 5.6406537, 512 | 321.3 |
| z12 | commit 8pi | did not finish or error | | | | | | | | | | |
| z12 | commit 8pi depth_max 12 | 0.001201 | True | False | False | 1602 | 12 | 0.906 | 1461 | root 5.636713876 | 5.6362216, 5.6372062, 8 | 56.4 |
| z12 | commit 8pi depth_max 8 | 1.432 | False | False | False | 2168 | 8 | 1.08 | 2048 | root 5.636713876 | 5.6362216, 5.6372062, 8 | 18.2 |
| z00 | g06 | 0.01831 | True | False | False | 249 | 20 | 0.696 | -- |  |  | 10.4 |
| z00 | parent | 0.01828 | True | False | False | 225 | 20 | 0.696 | -- |  |  | 8.6 |
| z00 | barlifted 128pi | 0.9084 | True | False | False | 18668 | 20 | 0.407 | 18618 | root 5.636713876 | 5.6347453, 5.6386834, 128 | 849.7 |
| z00 | barlifted 2pi | did not finish or error | | | | | | | | | | |
| z00 | barlifted 32pi | did not finish or error | | | | | | | | | | |
| z00 | barlifted 512pi | did not finish or error | | | | | | | | | | |
| z00 | barlifted 8pi | 0.001211 | True | False | False | 19172 | 20 | 0.906 | 18996 | root 5.636713876 | 5.6362216, 5.6372062, 8 | 887.9 |
| z00 | commit 128pi | did not finish or error | | | | | | | | | | |
| z00 | commit 128pi depth_max 12 | 0.9084 | True | False | False | 9761 | 12 | 0.407 | 9715 | root 5.636713876 | 5.6347453, 5.6386834, 128 | 345.4 |
| z00 | commit 128pi depth_max 8 | 0.9084 | True | False | False | 2755 | 8 | 0.407 | 2723 | root 5.636713876 | 5.6347453, 5.6386834, 128 | 79.6 |
| z00 | commit 2pi | did not finish or error | | | | | | | | | | |
| z00 | commit 2pi depth_max 12 | 0.7179 | True | False | False | 944 | 12 | 0.52 | 892 | root 5.636713876 | 5.6364678, 5.63696, 2 | 29.8 |
| z00 | commit 2pi depth_max 8 | 1.749 | False | False | False | 634 | 8 | 15.3 | 512 | root 5.636713876 | 5.6364678, 5.63696, 2 | 5.0 |
| z00 | commit 32pi | did not finish or error | | | | | | | | | | |
| z00 | commit 32pi depth_max 12 | 0.7623 | True | False | False | 5750 | 12 | 0.489 | 5651 | root 5.636713876 | 5.6357295, 5.6376985, 32 | 179.2 |
| z00 | commit 32pi depth_max 8 | 0.7626 | True | False | False | 1201 | 8 | 0.488 | 1163 | root 5.636713876 | 5.6357295, 5.6376985, 32 | 31.6 |
| z00 | commit 512pi | did not finish or error | | | | | | | | | | |
| z00 | commit 512pi depth_max 12 | 0.9299 | True | False | False | 20672 | 12 | 0.393 | 20638 | root 5.636713876 | 5.6327776, 5.6406537, 512 | 874.0 |
| z00 | commit 512pi depth_max 8 | 0.9299 | True | False | False | 7905 | 8 | 0.393 | 7877 | root 5.636713876 | 5.6327776, 5.6406537, 512 | 256.0 |
| z00 | commit 8pi | did not finish or error | | | | | | | | | | |
| z00 | commit 8pi depth_max 12 | 0.001201 | True | False | False | 1604 | 12 | 0.906 | 1461 | root 5.636713876 | 5.6362216, 5.6372062, 8 | 42.6 |
| z00 | commit 8pi depth_max 8 | 1.432 | False | False | False | 2170 | 8 | 1.08 | 2048 | root 5.636713876 | 5.6362216, 5.6372062, 8 | 14.2 |
| w00 | g06 | 0.005418 | True | False | False | 141 | 20 | 0.994 | -- |  |  | 4.2 |
| w00 | parent | 0.005418 | True | False | False | 123 | 20 | 0.994 | -- |  |  | 3.5 |
| w00 | barlifted 128pi | 0.05916 | True | True | False | 267 | 11 | 0.715 | 219 | root 9.231349907 | 9.1915236, 9.272473, 128 | 7.1 |
| w00 | barlifted 32pi | 0.0007812 | True | True | False | 168 | 15 | 0.875 | 69 | root 9.231349907 | 9.2112804, 9.2517434, 32 | 4.3 |
| w00 | barlifted 512pi | 0.7332 | True | True | False | 980 | 12 | 0.434 | 977 | root 9.231349907 | 9.1529054, 9.3149916, 512 | 28.8 |
| w00 | barlifted 8pi | 0.0009297 | True | True | False | 119 | 11 | 0.607 | 74 | root 9.231349907 | 9.2212754, 9.2415054, 10 | 3.2 |
| w00 | commit 128pi | 0.05915 | True | False | False | 2322 | 20 | 0.715 | 2274 | root 9.231349907 | 9.1915236, 9.272473, 128 | 86.6 |
| w00 | commit 2pi | 0.03617 | True | True | False | 74 | 13 | 0.652 | 18 | root 9.231349907 | 9.2263026, 9.2364175, 2 | 2.3 |
| w00 | commit 32pi | 0.0007817 | True | False | False | 1562 | 20 | 0.875 | 1463 | root 9.231349907 | 9.2112804, 9.2517434, 32 | 60.9 |
| w00 | commit 512pi | 0.7328 | True | False | False | 5033 | 20 | 0.434 | 5030 | root 9.231349907 | 9.1529054, 9.3149916, 512 | 190.4 |
| w00 | commit 8pi | 0.0009299 | True | False | False | 757 | 20 | 0.607 | 712 | root 9.231349907 | 9.2212754, 9.2415054, 10 | 28.7 |
| w01 | g06 | 0.8174 | True | True | False | 45 | 15 | 0.0278 | -- |  |  | 0.8 |
| w01 | parent | 0.8174 | True | True | False | 38 | 15 | 0.0278 | -- |  |  | 0.6 |
| w01 | barlifted 128pi | 0.02275 | True | True | False | 439 | 11 | 0.227 | 413 | root 9.231349907 | 9.1915236, 9.272473, 129 | 12.7 |
| w01 | barlifted 2pi | 0.1285 | True | True | False | 36 | 10 | 0.586 | 10 | root 9.231349907 | 9.2263026, 9.2364175, 2 | 0.8 |
| w01 | barlifted 32pi | 0.001213 | True | True | False | 257 | 16 | 0.295 | 174 | root 9.231349907 | 9.2112804, 9.2517434, 32 | 7.3 |
| w01 | barlifted 512pi | 0.001187 | True | True | False | 1109 | 12 | 0.13 | 1080 | root 9.231349907 | 9.1529054, 9.3149916, 513 | 35.2 |
| w01 | barlifted 8pi | 0.08556 | True | True | False | 111 | 13 | 0.864 | 46 | root 9.231349907 | 9.2212754, 9.2415054, 8 | 2.8 |
| w01 | commit 128pi | 0.02275 | True | False | False | 4715 | 20 | 0.227 | 4689 | root 9.231349907 | 9.1915236, 9.272473, 129 | 177.4 |
| w01 | commit 2pi | 0.1285 | True | False | False | 73 | 20 | 0.586 | 47 | root 9.231349907 | 9.2263026, 9.2364175, 2 | 1.9 |
| w01 | commit 32pi | 0.001213 | True | False | False | 1623 | 20 | 0.295 | 1540 | root 9.231349907 | 9.2112804, 9.2517434, 32 | 54.9 |
| w01 | commit 512pi | 0.001184 | True | False | False | 9402 | 20 | 0.13 | 9373 | root 9.231349907 | 9.1529054, 9.3149916, 513 | 379.1 |
| w01 | commit 8pi | 0.08556 | True | False | False | 751 | 20 | 0.864 | 686 | root 9.231349907 | 9.2212754, 9.2415054, 8 | 23.5 |
| w02 | g06 | 0.003166 | True | False | False | 84 | 20 | 0.573 | -- |  |  | 1.7 |
| w02 | parent | 0.003166 | True | True | False | 75 | 17 | 0.573 | -- |  |  | 1.6 |
| w02 | barlifted 128pi | 0.009747 | True | True | False | 456 | 13 | 0.206 | 369 | root 9.231349907 | 9.1915236, 9.272473, 129 | 12.9 |
| w02 | barlifted 32pi | 0.006926 | True | True | False | 219 | 15 | 0.411 | 78 | root 9.231349907 | 9.2112804, 9.2517434, 32 | 5.4 |
| w02 | barlifted 512pi | 0.6976 | True | True | False | 1006 | 10 | 0.247 | 1003 | root 9.231349907 | 9.1529054, 9.3149916, 513 | 32.6 |
| w02 | barlifted 8pi | 0.06099 | True | True | False | 134 | 14 | 0.43 | 67 | root 9.231349907 | 9.2212754, 9.2415054, 8 | 3.5 |
| w02 | commit 128pi | 0.009746 | True | False | False | 3784 | 20 | 0.206 | 3697 | root 9.231349907 | 9.1915236, 9.272473, 129 | 119.2 |
| w02 | commit 2pi | 0.03499 | True | True | False | 76 | 13 | 0.369 | 4 | root 9.231349907 | 9.2263026, 9.2364175, 2 | 1.6 |
| w02 | commit 32pi | 0.006925 | True | False | False | 768 | 20 | 0.411 | 627 | root 9.231349907 | 9.2112804, 9.2517434, 32 | 21.3 |
| w02 | commit 512pi | 0.6973 | True | False | False | 6640 | 20 | 0.247 | 6637 | root 9.231349907 | 9.1529054, 9.3149916, 513 | 240.6 |
| w02 | commit 8pi | 0.06099 | True | False | False | 679 | 20 | 0.43 | 612 | root 9.231349907 | 9.2212754, 9.2415054, 8 | 20.1 |
| w03 | g06 | 0.5732 | True | False | False | 202 | 20 | 0.0989 | -- |  |  | 6.2 |
| w03 | parent | 0.5731 | True | False | False | 139 | 20 | 0.0989 | -- |  |  | 3.8 |
| w03 | barlifted 128pi | 0.06936 | True | True | False | 371 | 11 | 0.198 | 341 | root 9.231349907 | 9.2088002, 9.2543095, 130 | 11.7 |
| w03 | barlifted 32pi | 0.01605 | True | True | False | 191 | 14 | 0.772 | 133 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 5.2 |
| w03 | barlifted 512pi | 0.1063 | True | True | False | 1282 | 11 | 0.717 | 1280 | root 9.231349907 | 9.1866443, 9.277696, 512 | 43.4 |
| w03 | barlifted 8pi | 0.0136 | True | True | False | 73 | 13 | 0.443 | 24 | root 9.231349907 | 9.2256747, 9.2370507, 10 | 1.7 |
| w03 | commit 128pi | 0.06936 | True | False | False | 4575 | 20 | 0.198 | 4545 | root 9.231349907 | 9.2088002, 9.2543095, 130 | 142.2 |
| w03 | commit 2pi | 0.07263 | True | True | False | 185 | 16 | 0.109 | 9 | root 9.231349907 | 9.2285091, 9.2341971, 2 | 4.7 |
| w03 | commit 32pi | 0.01605 | True | False | False | 1560 | 20 | 0.772 | 1502 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 48.1 |
| w03 | commit 512pi | 0.1063 | True | False | False | 11738 | 20 | 0.717 | 11736 | root 9.231349907 | 9.1866443, 9.277696, 512 | 426.1 |
| w03 | commit 8pi | 0.0136 | True | False | False | 761 | 20 | 0.443 | 712 | root 9.231349907 | 9.2256747, 9.2370507, 10 | 22.9 |
| w04 | g06 | 0.001819 | True | False | False | 105 | 20 | 0.178 | -- |  |  | 1.9 |
| w04 | parent | 0.001819 | True | False | False | 93 | 20 | 0.178 | -- |  |  | 1.6 |
| w04 | barlifted 128pi | 0.01718 | True | True | False | 573 | 13 | 0.178 | 489 | root 9.231349907 | 9.2088002, 9.2543095, 130 | 19.0 |
| w04 | barlifted 32pi | 0.009476 | True | True | False | 259 | 14 | 0.438 | 121 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 7.5 |
| w04 | barlifted 512pi | 0.03064 | True | True | False | 1050 | 13 | 0.113 | 1023 | root 9.231349907 | 9.1866443, 9.277696, 512 | 33.7 |
| w04 | barlifted 8pi | 0.002084 | True | True | False | 94 | 15 | 0.534 | 24 | root 9.231349907 | 9.2256747, 9.2370507, 10 | 2.5 |
| w04 | commit 128pi | 0.01718 | True | False | False | 5363 | 20 | 0.178 | 5279 | root 9.231349907 | 9.2088002, 9.2543095, 130 | 160.9 |
| w04 | commit 2pi | 0.0003702 | True | True | False | 133 | 16 | 0.15 | 6 | root 9.231349907 | 9.2285091, 9.2341971, 2 | 2.8 |
| w04 | commit 32pi | 0.009476 | True | False | False | 2807 | 20 | 0.438 | 2669 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 78.8 |
| w04 | commit 512pi | 0.03062 | True | False | False | 11753 | 20 | 0.113 | 11726 | root 9.231349907 | 9.1866443, 9.277696, 512 | 428.5 |
| w04 | commit 8pi | 0.002084 | True | False | False | 527 | 20 | 0.534 | 457 | root 9.231349907 | 9.2256747, 9.2370507, 10 | 13.2 |
| w05 | g06 | 0.05871 | True | False | False | 115 | 20 | 0.192 | -- |  |  | 2.1 |
| w05 | parent | 0.05871 | True | False | False | 95 | 20 | 0.192 | -- |  |  | 1.6 |
| w05 | barlifted 128pi | 0.005369 | True | True | False | 564 | 14 | 0.187 | 453 | root 9.231349907 | 9.2088002, 9.2543095, 129 | 18.9 |
| w05 | barlifted 32pi | 0.008007 | True | True | False | 196 | 16 | 0.567 | 146 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 5.9 |
| w05 | barlifted 512pi | 0.1511 | True | False | False | 1929 | 20 | 0.0946 | 1913 | root 9.231349907 | 9.1866443, 9.277696, 512 | 66.2 |
| w05 | barlifted 8pi | 0.0713 | True | True | False | 211 | 15 | 0.284 | 84 | root 9.231349907 | 9.2256747, 9.2370507, 9 | 6.7 |
| w05 | commit 128pi | 0.005367 | True | False | False | 7005 | 20 | 0.187 | 6894 | root 9.231349907 | 9.2088002, 9.2543095, 129 | 224.9 |
| w05 | commit 2pi | 0.01868 | True | True | False | 51 | 9 | 0.763 | 4 | root 9.231349907 | 9.2285091, 9.2341971, 2 | 0.9 |
| w05 | commit 32pi | 0.008007 | True | False | False | 2001 | 20 | 0.567 | 1951 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 54.3 |
| w05 | commit 512pi | 0.151 | True | False | False | 13854 | 20 | 0.0947 | 13838 | root 9.231349907 | 9.1866443, 9.277696, 512 | 532.6 |
| w05 | commit 8pi | 0.0713 | True | False | False | 844 | 20 | 0.284 | 717 | root 9.231349907 | 9.2256747, 9.2370507, 9 | 21.3 |
| w06 | g06 | 0.1162 | True | True | False | 39 | 11 | 0.579 | -- |  |  | 0.4 |
| w06 | parent | 0.1162 | True | True | False | 38 | 11 | 0.579 | -- |  |  | 0.4 |
| w06 | barlifted 128pi | 0.002847 | True | True | False | 353 | 14 | 0.536 | 274 | root 9.231349907 | 9.2088002, 9.2543095, 128 | 10.6 |
| w06 | barlifted 32pi | 0.005276 | True | True | False | 208 | 12 | 0.656 | 143 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 6.1 |
| w06 | barlifted 512pi | 0.001312 | True | True | False | 1294 | 13 | 0.359 | 1276 | root 9.231349907 | 9.1866443, 9.277696, 514 | 43.2 |
| w06 | barlifted 8pi | 0.0009972 | True | True | False | 66 | 12 | 0.526 | 20 | root 9.231349907 | 9.2256747, 9.2370507, 8 | 1.9 |
| w06 | commit 128pi | 0.002845 | True | False | False | 2740 | 20 | 0.536 | 2661 | root 9.231349907 | 9.2088002, 9.2543095, 128 | 80.6 |
| w06 | commit 2pi | 0.06701 | True | True | False | 76 | 11 | 0.743 | 8 | root 9.231349907 | 9.2285091, 9.2341971, 2 | 1.3 |
| w06 | commit 32pi | 0.005277 | True | False | False | 1683 | 20 | 0.656 | 1618 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 47.3 |
| w06 | commit 512pi | 0.001307 | True | False | False | 12259 | 20 | 0.359 | 12241 | root 9.231349907 | 9.1866443, 9.277696, 514 | 454.4 |
| w06 | commit 8pi | 0.0009973 | True | True | False | 136 | 20 | 0.526 | 90 | root 9.231349907 | 9.2256747, 9.2370507, 8 | 3.3 |
| w07 | g06 | 0.9424 | True | False | False | 208 | 20 | 0.241 | -- |  |  | 4.7 |
| w07 | parent | 0.9424 | True | False | False | 175 | 20 | 0.241 | -- |  |  | 3.8 |
| w07 | barlifted 128pi | 0.001011 | True | True | False | 532 | 11 | 0.701 | 489 | root 9.231349907 | 9.2186199, 9.2442095, 128 | 17.2 |
| w07 | barlifted 2pi | 0.002971 | True | True | False | 130 | 13 | 0.76 | 61 | root 9.231349907 | 9.2297516, 9.2329502, 4 | 3.5 |
| w07 | barlifted 32pi | 0.002938 | True | True | False | 227 | 15 | 0.663 | 155 | root 9.231349907 | 9.2249689, 9.2377633, 34 | 7.0 |
| w07 | barlifted 512pi | 0.001522 | True | True | False | 2296 | 16 | 0.497 | 2209 | root 9.231349907 | 9.2060166, 9.2572017, 514 | 61.0 |
| w07 | barlifted 8pi | 0.003618 | True | True | False | 171 | 11 | 0.577 | 100 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 4.9 |
| w07 | commit 128pi | 0.001009 | True | False | False | 8309 | 20 | 0.701 | 8266 | root 9.231349907 | 9.2186199, 9.2442095, 128 | 281.9 |
| w07 | commit 2pi | 0.002971 | True | False | False | 932 | 20 | 0.76 | 863 | root 9.231349907 | 9.2297516, 9.2329502, 4 | 24.5 |
| w07 | commit 32pi | 0.002939 | True | False | False | 4927 | 20 | 0.663 | 4855 | root 9.231349907 | 9.2249689, 9.2377633, 34 | 147.0 |
| w07 | commit 512pi | did not finish or error | | | | | | | | | | |
| w07 | commit 512pi depth_max 12 | 0.001517 | True | False | False | 8472 | 12 | 0.497 | 8391 | root 9.231349907 | 9.2060166, 9.2572017, 514 | 272.9 |
| w07 | commit 512pi depth_max 8 | 0.001541 | True | False | False | 4797 | 8 | 0.497 | 4744 | root 9.231349907 | 9.2060166, 9.2572017, 514 | 138.7 |
| w07 | commit 8pi | 0.003618 | True | False | False | 549 | 20 | 0.577 | 478 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 13.5 |
| w08 | g06 | 0.01027 | True | False | False | 176 | 20 | 0.159 | -- |  |  | 4.4 |
| w08 | parent | 0.01027 | True | False | False | 151 | 20 | 0.159 | -- |  |  | 3.6 |
| w08 | barlifted 128pi | 0.2082 | True | True | False | 663 | 13 | 0.304 | 643 | root 9.231349907 | 9.2186199, 9.2442095, 128 | 22.6 |
| w08 | barlifted 32pi | 0.113 | True | True | False | 117 | 10 | 0.657 | 101 | root 9.231349907 | 9.2249689, 9.2377633, 32 | 3.6 |
| w08 | barlifted 512pi | 0.09994 | True | True | False | 1653 | 18 | 0.264 | 1609 | root 9.231349907 | 9.2060166, 9.2572017, 513 | 53.4 |
| w08 | barlifted 8pi | 0.1248 | True | True | False | 89 | 10 | 0.658 | 55 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 2.0 |
| w08 | commit 128pi | 0.2082 | True | False | False | 8798 | 20 | 0.304 | 8778 | root 9.231349907 | 9.2186199, 9.2442095, 128 | 298.9 |
| w08 | commit 2pi | 0.08832 | True | True | False | 31 | 9 | 0.953 | 3 | root 9.231349907 | 9.2297516, 9.2329502, 2 | 0.6 |
| w08 | commit 32pi | 0.1129 | True | False | False | 1640 | 20 | 0.657 | 1624 | root 9.231349907 | 9.2249689, 9.2377633, 32 | 45.6 |
| w08 | commit 512pi | 0.09992 | True | False | False | 15894 | 20 | 0.264 | 15850 | root 9.231349907 | 9.2060166, 9.2572017, 513 | 692.0 |
| w08 | commit 8pi | 0.1248 | True | False | False | 537 | 20 | 0.658 | 503 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 14.6 |
| w09 | g06 | 0.06182 | True | False | False | 109 | 20 | 0.88 | -- |  |  | 2.4 |
| w09 | parent | 0.06182 | True | False | False | 107 | 20 | 0.88 | -- |  |  | 2.3 |
| w09 | barlifted 128pi | 0.2017 | True | True | False | 731 | 14 | 0.304 | 711 | root 9.231349907 | 9.2186199, 9.2442095, 128 | 24.6 |
| w09 | barlifted 2pi | 0.1256 | True | True | False | 31 | 9 | 0.642 | 6 | root 9.231349907 | 9.2297516, 9.2329502, 2 | 0.7 |
| w09 | barlifted 32pi | 0.07397 | True | True | False | 110 | 11 | 0.885 | 94 | root 9.231349907 | 9.2249689, 9.2377633, 32 | 3.5 |
| w09 | barlifted 512pi | 0.1264 | True | True | False | 1695 | 15 | 0.23 | 1658 | root 9.231349907 | 9.2060166, 9.2572017, 512 | 54.8 |
| w09 | barlifted 8pi | 0.141 | True | True | False | 58 | 10 | 0.56 | 24 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 1.4 |
| w09 | commit 128pi | 0.2017 | True | False | False | 9169 | 20 | 0.304 | 9149 | root 9.231349907 | 9.2186199, 9.2442095, 128 | 348.6 |
| w09 | commit 2pi | 0.1256 | True | False | False | 2058 | 20 | 0.642 | 2033 | root 9.231349907 | 9.2297516, 9.2329502, 2 | 55.4 |
| w09 | commit 32pi | 0.07397 | True | False | False | 2915 | 20 | 0.885 | 2899 | root 9.231349907 | 9.2249689, 9.2377633, 32 | 87.5 |
| w09 | commit 512pi | 0.1264 | True | False | False | 14226 | 20 | 0.23 | 14189 | root 9.231349907 | 9.2060166, 9.2572017, 512 | 619.1 |
| w09 | commit 8pi | 0.141 | True | False | False | 168 | 20 | 0.56 | 134 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 4.0 |
| w10 | g06 | 0.3912 | True | False | False | 137 | 20 | 0.326 | -- |  |  | 3.5 |
| w10 | parent | 0.3912 | True | False | False | 126 | 20 | 0.326 | -- |  |  | 3.3 |
| w10 | barlifted 128pi | 0.2238 | True | True | False | 597 | 13 | 0.276 | 578 | root 9.231349907 | 9.2186199, 9.2442095, 130 | 18.8 |
| w10 | barlifted 2pi | 0.09232 | True | True | False | 29 | 8 | 0.854 | 3 | root 9.231349907 | 9.2297516, 9.2329502, 2 | 0.8 |
| w10 | barlifted 32pi | 0.1006 | True | True | False | 225 | 15 | 0.695 | 210 | root 9.231349907 | 9.2249689, 9.2377633, 32 | 7.4 |
| w10 | barlifted 512pi | 0.1346 | True | True | False | 1274 | 15 | 0.221 | 1235 | root 9.231349907 | 9.2060166, 9.2572017, 512 | 40.8 |
| w10 | barlifted 8pi | 0.1013 | True | True | False | 68 | 10 | 0.774 | 35 | root 9.231349907 | 9.2281554, 9.2345525, 10 | 1.8 |
| w10 | commit 128pi | 0.2238 | True | False | False | 8052 | 20 | 0.276 | 8033 | root 9.231349907 | 9.2186199, 9.2442095, 130 | 305.5 |
| w10 | commit 2pi | 0.09232 | True | False | False | 1634 | 20 | 0.854 | 1608 | root 9.231349907 | 9.2297516, 9.2329502, 2 | 48.6 |
| w10 | commit 32pi | 0.1006 | True | False | False | 3471 | 20 | 0.695 | 3456 | root 9.231349907 | 9.2249689, 9.2377633, 32 | 113.1 |
| w10 | commit 512pi | 0.1346 | True | False | False | 11903 | 20 | 0.221 | 11864 | root 9.231349907 | 9.2060166, 9.2572017, 512 | 497.7 |
| w10 | commit 8pi | 0.1013 | True | False | False | 396 | 20 | 0.774 | 363 | root 9.231349907 | 9.2281554, 9.2345525, 10 | 11.1 |
| w11 | g06 | 0.0472 | True | False | False | 121 | 20 | 0.219 | -- |  |  | 2.6 |
| w11 | parent | 0.0472 | True | False | False | 111 | 20 | 0.219 | -- |  |  | 2.5 |
| w11 | barlifted 128pi | 0.1894 | True | True | False | 563 | 15 | 0.344 | 544 | root 9.231349907 | 9.2186199, 9.2442095, 129 | 17.1 |
| w11 | barlifted 32pi | 0.1024 | True | True | False | 159 | 13 | 0.764 | 128 | root 9.231349907 | 9.2249689, 9.2377633, 34 | 4.9 |
| w11 | barlifted 512pi | 0.06542 | True | True | False | 1781 | 14 | 0.326 | 1763 | root 9.231349907 | 9.2060166, 9.2572017, 513 | 58.6 |
| w11 | barlifted 8pi | 0.1182 | True | True | False | 57 | 10 | 0.742 | 27 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 1.5 |
| w11 | commit 128pi | 0.1894 | True | False | False | 5648 | 20 | 0.344 | 5629 | root 9.231349907 | 9.2186199, 9.2442095, 129 | 199.6 |
| w11 | commit 2pi | 0.2176 | True | True | False | 82 | 19 | 0.414 | 57 | root 9.231349907 | 9.2297516, 9.2329502, 4 | 2.1 |
| w11 | commit 32pi | 0.1024 | True | False | False | 2339 | 20 | 0.764 | 2308 | root 9.231349907 | 9.2249689, 9.2377633, 34 | 76.0 |
| w11 | commit 512pi | 0.06542 | True | False | False | 17333 | 20 | 0.326 | 17315 | root 9.231349907 | 9.2060166, 9.2572017, 513 | 779.7 |
| w11 | commit 8pi | 0.1182 | True | False | False | 324 | 20 | 0.742 | 294 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 9.5 |
| r55 | g06 | 0.02665 | True | True | False | 1 | 0 | 0.00112 | -- |  |  | 0.0 |
| r55 | parent | 0.02665 | True | True | False | 1 | 0 | 0.00112 | -- |  |  | 0.0 |
| r55 | commit 128pi | 0.02665 | True | True | False | 1 | 0 | 0.00112 | 0 |  |  | 0.0 |
| r55 | commit 2pi | 0.02665 | True | True | False | 1 | 0 | 0.00112 | 0 |  |  | 0.0 |
| r55 | commit 32pi | 0.02665 | True | True | False | 1 | 0 | 0.00112 | 0 |  |  | 0.1 |
| r55 | commit 512pi | 0.02665 | True | True | False | 1 | 0 | 0.00112 | 0 |  |  | 0.0 |
| r55 | commit 8pi | 0.02665 | True | True | False | 1 | 0 | 0.00112 | 0 |  |  | 0.0 |

## Section 2.5 on these calls

| call | mode | converged | node_limited | max depth | leaves by class: count, sum e', at depth_max | QUADPACK iroff1, iroff2, ier=2 at bisection |
|---|---|---|---|---|---|---|
| z12 | commit 128pi depth_max 12 | False | False | 12 | Levin 42, 6.56e-34, 4; window 9715, 1.38e-37, 5258 | 5154, 5131, 19 |
| z12 | commit 128pi depth_max 8 | False | False | 8 | Levin 32, 6.56e-34, 8; window 2723, 1.40e-37, 1476 | 1390, 1385, 19 |
| z12 | commit 2pi depth_max 12 | False | False | 12 | Levin 51, 8.38e-34, 6; window 892, 8.10e-39, 628 | 501, 500, 36 |
| z12 | commit 2pi depth_max 8 | False | False | 8 | Clenshaw-Curtis 32, 3.95e-36, 32; Levin 88, 2.47e-32, 68; window 512, 2.32e-39, 512 | 302, 307, 47 |
| z12 | commit 32pi depth_max 12 | False | False | 12 | Clenshaw-Curtis 47, 2.66e-37, 47; Levin 49, 7.87e-34, 5; window 5651, 1.17e-37, 3676 | 3065, 3057, 22 |
| z12 | commit 32pi depth_max 8 | False | False | 8 | Levin 38, 7.87e-34, 14; window 1163, 1.19e-37, 752 | 610, 614, 22 |
| z12 | commit 512pi depth_max 8 | False | False | 8 | Levin 28, 6.34e-34, 4; window 7877, 1.88e-37, 3358 | 3903, 3878, 18 |
| z12 | commit 8pi depth_max 12 | False | False | 12 | Clenshaw-Curtis 62, 2.10e-34, 59; Levin 79, 1.25e-33, 3; window 1461, 1.36e-38, 1014 | 863, 868, 18 |
| z12 | commit 8pi depth_max 8 | False | False | 8 | Clenshaw-Curtis 32, 3.93e-36, 32; Levin 88, 1.73e-33, 68; window 2048, 4.48e-39, 2048 | 1136, 1137, 18 |
| z00 | barlifted 128pi | False | False | 20 | Clenshaw-Curtis 8, 1.08e-38, 0; Levin 42, 6.56e-34, 0; window 18618, 1.28e-37, 7266 | 8774, 8722, 19 |
| z00 | barlifted 8pi | False | False | 20 | Clenshaw-Curtis 98, 2.10e-34, 0; Levin 78, 1.25e-33, 0; window 18996, 1.28e-38, 11652 | 9780, 9710, 18 |
| z00 | commit 128pi depth_max 12 | False | False | 12 | Levin 46, 6.56e-34, 6; window 9715, 1.38e-37, 5258 | 5157, 5134, 19 |
| z00 | commit 128pi depth_max 8 | False | False | 8 | Levin 32, 6.56e-34, 10; window 2723, 1.40e-37, 1476 | 1391, 1386, 19 |
| z00 | commit 2pi depth_max 12 | False | False | 12 | Levin 52, 8.38e-34, 6; window 892, 8.10e-39, 628 | 503, 502, 36 |
| z00 | commit 2pi depth_max 8 | False | False | 8 | Clenshaw-Curtis 32, 3.95e-36, 32; Levin 90, 2.47e-32, 70; window 512, 2.32e-39, 512 | 305, 310, 47 |
| z00 | commit 32pi depth_max 12 | False | False | 12 | Clenshaw-Curtis 47, 2.66e-37, 47; Levin 52, 7.87e-34, 5; window 5651, 1.17e-37, 3676 | 3068, 3060, 22 |
| z00 | commit 32pi depth_max 8 | False | False | 8 | Levin 38, 7.87e-34, 16; window 1163, 1.19e-37, 752 | 611, 615, 22 |
| z00 | commit 512pi depth_max 12 | False | False | 12 | Levin 34, 6.34e-34, 0; window 20638, 1.88e-37, 8878 | 10802, 10718, 18 |
| z00 | commit 512pi depth_max 8 | False | False | 8 | Levin 28, 6.34e-34, 4; window 7877, 1.88e-37, 3358 | 3903, 3878, 18 |
| z00 | commit 8pi depth_max 12 | False | False | 12 | Clenshaw-Curtis 62, 2.10e-34, 59; Levin 81, 1.25e-33, 3; window 1461, 1.36e-38, 1014 | 865, 870, 18 |
| z00 | commit 8pi depth_max 8 | False | False | 8 | Clenshaw-Curtis 32, 3.93e-36, 32; Levin 90, 1.73e-33, 70; window 2048, 4.48e-39, 2048 | 1138, 1139, 18 |
| z00 | g06 | False | False | 20 | Clenshaw-Curtis 162, 2.68e-34, 78; Levin 87, 8.53e-34, 0 | 131, 129, 7 |
| z00 | parent | False | False | 20 | Clenshaw-Curtis 139, 2.68e-34, 76; Levin 86, 8.53e-34, 0 | 125, 123, 7 |
| w00 | commit 128pi | False | False | 20 | Clenshaw-Curtis 9, 1.61e-29, 0; Levin 39, 2.42e-29, 0; window 2274, 4.30e-33, 304 | 1192, 1183, 15 |
| w00 | commit 32pi | False | False | 20 | Clenshaw-Curtis 45, 1.21e-29, 0; Levin 54, 3.72e-29, 0; window 1463, 1.19e-33, 200 | 808, 803, 17 |
| w00 | commit 512pi | False | False | 20 | Levin 3, 2.44e-29, 0; window 5030, 4.27e-32, 498 | 2256, 2230, 90 |
| w00 | commit 8pi | False | False | 20 | Clenshaw-Curtis 10, 1.39e-29, 0; Levin 35, 2.03e-29, 0; window 712, 8.07e-34, 112 | 389, 379, 20 |
| w00 | g06 | False | False | 20 | Clenshaw-Curtis 88, 1.89e-29, 4; Levin 53, 3.71e-29, 0 | 64, 64, 18 |
| w00 | parent | False | False | 20 | Clenshaw-Curtis 73, 1.89e-29, 2; Levin 50, 3.71e-29, 0 | 55, 55, 18 |
| w01 | commit 128pi | False | False | 20 | Clenshaw-Curtis 3, 9.83e-30, 0; Levin 23, 2.91e-29, 0; window 4689, 4.81e-33, 854 | 2445, 2425, 9 |
| w01 | commit 2pi | False | False | 20 | Levin 26, 1.01e-28, 0; window 47, 4.62e-35, 12 | 28, 28, 29 |
| w01 | commit 32pi | False | False | 20 | Clenshaw-Curtis 41, 1.42e-29, 0; Levin 42, 3.64e-29, 0; window 1540, 1.74e-33, 178 | 861, 857, 14 |
| w01 | commit 512pi | False | False | 20 | Clenshaw-Curtis 3, 1.09e-29, 0; Levin 26, 1.13e-29, 0; window 9373, 2.30e-32, 1612 | 4576, 4545, 11 |
| w01 | commit 8pi | False | False | 20 | Clenshaw-Curtis 31, 9.50e-34, 0; Levin 34, 1.48e-28, 0; window 686, 5.17e-34, 82 | 390, 387, 23 |
| w02 | commit 128pi | False | False | 20 | Clenshaw-Curtis 61, 1.81e-33, 0; Levin 26, 1.90e-29, 0; window 3697, 4.89e-33, 878 | 1941, 1928, 12 |
| w02 | commit 32pi | False | False | 20 | Clenshaw-Curtis 65, 5.36e-30, 0; Levin 76, 3.26e-29, 0; window 627, 1.04e-33, 72 | 386, 385, 16 |
| w02 | commit 512pi | False | False | 20 | Levin 3, 2.28e-29, 0; window 6637, 2.70e-32, 620 | 3096, 3074, 103 |
| w02 | commit 8pi | False | False | 20 | Clenshaw-Curtis 13, 5.96e-30, 0; Levin 54, 3.38e-29, 0; window 612, 1.09e-33, 78 | 359, 358, 14 |
| w02 | g06 | False | False | 20 | Clenshaw-Curtis 30, 9.91e-30, 2; Levin 54, 4.30e-29, 0 | 43, 41, 21 |
| w03 | commit 128pi | False | False | 20 | Levin 30, 1.05e-30, 0; window 4545, 1.79e-34, 1034 | 2378, 2360, 13 |
| w03 | commit 32pi | False | False | 20 | Clenshaw-Curtis 16, 1.02e-35, 0; Levin 42, 4.12e-30, 0; window 1502, 4.02e-35, 172 | 830, 823, 16 |
| w03 | commit 512pi | False | False | 20 | Levin 2, 3.82e-30, 0; window 11736, 6.98e-34, 1142 | 5933, 5888, 64 |
| w03 | commit 8pi | False | False | 20 | Clenshaw-Curtis 6, 1.04e-36, 0; Levin 43, 2.36e-30, 0; window 712, 1.69e-35, 224 | 419, 416, 17 |
| w03 | g06 | False | False | 20 | Clenshaw-Curtis 189, 3.02e-35, 38; Levin 13, 5.27e-31, 0 | 89, 91, 27 |
| w03 | parent | False | False | 20 | Clenshaw-Curtis 126, 4.96e-35, 8; Levin 13, 5.27e-31, 0 | 63, 65, 28 |
| w04 | commit 128pi | False | False | 20 | Clenshaw-Curtis 35, 3.45e-35, 0; Levin 49, 9.01e-31, 0; window 5279, 1.59e-34, 1074 | 2779, 2760, 14 |
| w04 | commit 32pi | False | False | 20 | Clenshaw-Curtis 67, 1.14e-31, 0; Levin 71, 2.10e-30, 0; window 2669, 4.62e-35, 646 | 1484, 1474, 17 |
| w04 | commit 512pi | False | False | 20 | Clenshaw-Curtis 6, 2.21e-31, 0; Levin 21, 3.50e-31, 0; window 11726, 1.01e-33, 2286 | 5844, 5798, 10 |
| w04 | commit 8pi | False | False | 20 | Clenshaw-Curtis 22, 1.46e-31, 0; Levin 48, 2.56e-30, 0; window 457, 8.02e-36, 22 | 281, 279, 18 |
| w04 | g06 | False | False | 20 | Clenshaw-Curtis 43, 1.16e-32, 4; Levin 62, 8.90e-31, 0 | 60, 59, 8 |
| w04 | parent | False | False | 20 | Clenshaw-Curtis 32, 1.16e-32, 2; Levin 61, 8.90e-31, 0 | 50, 49, 8 |
| w05 | barlifted 512pi | False | False | 20 | Clenshaw-Curtis 2, 3.50e-31, 0; Levin 14, 1.26e-31, 0; window 1913, 3.31e-34, 372 | 675, 672, 13 |
| w05 | commit 128pi | False | False | 20 | Clenshaw-Curtis 71, 3.15e-31, 0; Levin 40, 6.28e-31, 0; window 6894, 1.68e-34, 1358 | 3639, 3602, 15 |
| w05 | commit 32pi | False | False | 20 | Clenshaw-Curtis 3, 2.67e-31, 0; Levin 47, 2.59e-30, 0; window 1951, 8.99e-35, 552 | 1060, 1057, 14 |
| w05 | commit 512pi | False | False | 20 | Clenshaw-Curtis 2, 3.50e-31, 0; Levin 14, 1.26e-31, 0; window 13838, 7.01e-34, 2850 | 7046, 7001, 13 |
| w05 | commit 8pi | False | False | 20 | Clenshaw-Curtis 74, 2.42e-31, 0; Levin 53, 1.19e-30, 0; window 717, 7.56e-36, 234 | 434, 431, 17 |
| w05 | g06 | False | False | 20 | Clenshaw-Curtis 56, 1.51e-35, 18; Levin 59, 9.68e-31, 0 | 52, 53, 18 |
| w05 | parent | False | False | 20 | Clenshaw-Curtis 37, 5.95e-35, 2; Levin 58, 9.68e-31, 0 | 47, 48, 18 |
| w06 | commit 128pi | False | False | 20 | Clenshaw-Curtis 28, 2.81e-31, 0; Levin 51, 6.17e-31, 0; window 2661, 1.97e-34, 262 | 1363, 1356, 14 |
| w06 | commit 32pi | False | False | 20 | Clenshaw-Curtis 14, 2.77e-31, 0; Levin 51, 8.24e-31, 0; window 1618, 6.72e-35, 360 | 896, 889, 16 |
| w06 | commit 512pi | False | False | 20 | Clenshaw-Curtis 2, 2.66e-31, 0; Levin 16, 3.35e-31, 0; window 12241, 6.38e-34, 2376 | 6149, 6111, 16 |
| w06 | commit 8pi | True | False | 20 | Clenshaw-Curtis 11, 2.71e-31, 0; Levin 35, 6.11e-31, 0; window 90, 8.61e-36, 2 | 64, 64, 18 |
| w07 | commit 128pi | False | False | 20 | Clenshaw-Curtis 2, 3.53e-33, 0; Levin 41, 2.55e-32, 0; window 8266, 7.63e-36, 1916 | 4401, 4368, 13 |
| w07 | commit 2pi | False | False | 20 | Clenshaw-Curtis 10, 3.32e-33, 0; Levin 59, 2.81e-32, 0; window 863, 9.77e-38, 300 | 494, 498, 26 |
| w07 | commit 32pi | False | False | 20 | Clenshaw-Curtis 16, 3.44e-33, 0; Levin 56, 2.40e-32, 0; window 4855, 1.09e-36, 1828 | 2643, 2622, 26 |
| w07 | commit 512pi depth_max 12 | False | False | 12 | Clenshaw-Curtis 30, 5.66e-33, 12; Levin 51, 1.49e-32, 0; window 8391, 2.78e-35, 2124 | 4221, 4190, 16 |
| w07 | commit 512pi depth_max 8 | False | False | 8 | Clenshaw-Curtis 3, 5.65e-33, 0; Levin 50, 1.49e-32, 14; window 4744, 2.77e-35, 1440 | 2201, 2184, 16 |
| w07 | commit 8pi | False | False | 20 | Clenshaw-Curtis 2, 3.37e-33, 0; Levin 69, 2.05e-32, 0; window 478, 3.56e-37, 50 | 277, 276, 24 |
| w07 | g06 | False | False | 20 | Clenshaw-Curtis 142, 2.55e-36, 22; Levin 66, 9.96e-33, 0 | 103, 104, 22 |
| w07 | parent | False | False | 20 | Clenshaw-Curtis 110, 2.47e-36, 8; Levin 65, 9.96e-33, 0 | 88, 89, 22 |
| w08 | commit 128pi | False | False | 20 | Levin 20, 5.11e-32, 0; window 8778, 4.09e-36, 1830 | 4633, 4604, 49 |
| w08 | commit 32pi | False | False | 20 | Levin 16, 1.11e-31, 0; window 1624, 1.20e-36, 350 | 854, 849, 28 |
| w08 | commit 512pi | False | False | 20 | Clenshaw-Curtis 16, 5.56e-37, 0; Levin 28, 4.44e-32, 0; window 15850, 4.53e-35, 3620 | 8216, 8159, 18 |
| w08 | commit 8pi | False | False | 20 | Levin 34, 1.11e-31, 0; window 503, 2.90e-37, 54 | 274, 275, 18 |
| w08 | g06 | False | False | 20 | Clenshaw-Curtis 115, 4.76e-37, 22; Levin 61, 2.68e-32, 0 | 77, 80, 15 |
| w08 | parent | False | False | 20 | Clenshaw-Curtis 91, 1.75e-36, 14; Levin 60, 2.68e-32, 0 | 66, 69, 15 |
| w09 | commit 128pi | False | False | 20 | Levin 20, 5.17e-32, 0; window 9149, 6.77e-36, 1882 | 4855, 4817, 19 |
| w09 | commit 2pi | False | False | 20 | Levin 25, 1.09e-31, 0; window 2033, 8.45e-38, 1304 | 1067, 1059, 25 |
| w09 | commit 32pi | False | False | 20 | Levin 16, 1.50e-31, 0; window 2899, 1.34e-36, 790 | 1529, 1514, 21 |
| w09 | commit 512pi | False | False | 20 | Clenshaw-Curtis 14, 7.21e-37, 0; Levin 23, 3.91e-32, 0; window 14189, 3.41e-35, 2202 | 7345, 7300, 33 |
| w09 | commit 8pi | False | False | 20 | Levin 34, 9.51e-32, 0; window 134, 2.81e-37, 34 | 83, 86, 17 |
| w09 | g06 | False | False | 20 | Clenshaw-Curtis 85, 3.68e-37, 18; Levin 24, 1.49e-31, 0 | 46, 46, 24 |
| w09 | parent | False | False | 20 | Clenshaw-Curtis 83, 5.92e-37, 14; Levin 24, 1.49e-31, 0 | 46, 46, 24 |
| w10 | commit 128pi | False | False | 20 | Levin 19, 4.69e-32, 0; window 8033, 7.57e-36, 1656 | 4256, 4232, 41 |
| w10 | commit 2pi | False | False | 20 | Levin 26, 1.45e-31, 0; window 1608, 6.33e-38, 678 | 868, 866, 23 |
| w10 | commit 32pi | False | False | 20 | Levin 15, 1.18e-31, 0; window 3456, 1.06e-36, 1196 | 1817, 1806, 25 |
| w10 | commit 512pi | False | False | 20 | Clenshaw-Curtis 16, 5.78e-37, 0; Levin 23, 3.75e-32, 0; window 11864, 2.42e-35, 1790 | 6020, 5973, 32 |
| w10 | commit 8pi | False | False | 20 | Levin 33, 1.32e-31, 0; window 363, 2.19e-37, 100 | 207, 206, 16 |
| w10 | g06 | False | False | 20 | Clenshaw-Curtis 123, 5.19e-37, 22; Levin 14, 5.54e-32, 0 | 65, 64, 17 |
| w10 | parent | False | False | 20 | Clenshaw-Curtis 112, 1.03e-36, 14; Levin 14, 5.54e-32, 0 | 58, 57, 17 |
| w11 | commit 128pi | False | False | 20 | Levin 19, 5.60e-32, 0; window 5629, 6.94e-36, 1516 | 2946, 2924, 31 |
| w11 | commit 32pi | False | False | 20 | Clenshaw-Curtis 4, 2.97e-37, 0; Levin 27, 1.25e-31, 0; window 2308, 1.17e-36, 388 | 1228, 1220, 22 |
| w11 | commit 512pi | False | False | 20 | Levin 18, 5.31e-32, 0; window 17315, 1.78e-35, 3630 | 8955, 8897, 25 |
| w11 | commit 8pi | False | False | 20 | Levin 30, 1.21e-31, 0; window 294, 3.08e-37, 68 | 172, 173, 21 |
| w11 | g06 | False | False | 20 | Clenshaw-Curtis 62, 2.56e-37, 24; Levin 59, 3.56e-32, 0 | 66, 67, 9 |
| w11 | parent | False | False | 20 | Clenshaw-Curtis 58, 3.98e-37, 22; Levin 53, 3.56e-32, 0 | 60, 61, 9 |

Wall time 9048 s; uptime at end: 12:07  up 7 days, 14:28, 1 user, load averages: 4.82 7.20 10.70

