# 07 section 2.3: the real calls replayed

    levin-phase-floor 07: section 2.3, the real calls replayed (run)
    written 2026-11-08T11:52:57+00:00 on mps026088.phys.susx.ac.uk
    python 3.13.16 (/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python)
    script replay.py sha256 f65260446369c3cfd558525c707a007c8179536632252792acf02a99f598a385
    prov.py sha256 9eb229e7c7ef3b922efc784a5cb53cd0faa601b921e0483f0bf92dcd9f9a5ac2
    AdaptiveLevin/levin_quadrature.py sha256 4e5da371f51e6e3dc20e73579eb45e0e985e2a3b18571ee29eb86b3d3e9c93fc
    input docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json sha256 73c18680d266b1a0fdf11123c104d20e822658047dd7c36e1c814db9ffcad912
    input docs/levin-phase-floor/06g/references/out/capture.json sha256 62b91d27bbd6a4e6a5fc10f64e212e4a491baeb043c1651e1722b0ebf5922a49
    input docs/levin-phase-floor/06g/references/v1/out/v1_capture.json sha256 71e5b9da1ec4b9b4ee465c0cd0531a34d48ba46f26fb47bd4cf417af9a9ad19a
    input docs/levin-phase-floor/06l/out/partC/h1_calls.csv sha256 641d497a3451ea24f614a9ce5a598cd382c8a646b65ca42df9561acccc764270
    input docs/levin-phase-floor/07/calibration/rule-a/scripts/analysis07.py sha256 784ad891688736328ac753190615995304f9307620a219f0bcbebd77f0e30fc4
    input docs/levin-phase-floor/07/calibration/rule-a/scripts/variant_loader.py sha256 3206d8f892cdc31bdcd1ce8be719d1dc1c4361a294fc5c86dad8bd043f2d8b33
    input docs/levin-phase-floor/07/calibration/rule-a/scripts/calibrate.py sha256 03a47ec30ec09ccf3ea8693b248dcdc6bd094ed4362219cc7f745a09e0af3df6
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
    HEAD 2ddd33066b1da9929988d23f3b42ecfa061ac7cc
    tree dirty: yes; git status --short:
         M AdaptiveLevin/levin_quadrature.py
         M AdaptiveLevin/tests/test_levin_stationary.py
         M AdaptiveLevin/tests/test_levin_trust_yardstick.py
         M CLAUDE.md
         M docs/OPEN_ISSUES.md
        ?? docs/levin-phase-floor/07/calibration/rule-a/
    uptime at start: 11:52  up 8 days, 14:13, 1 user, load averages: 48.63 29.18 26.21

06g's tree: /private/tmp/claude-35086/-Users-ds283-Documents-Code-SecondaryGWKit/194e48fe-b717-47ef-90f9-bf58f0c23da9/scratchpad/rulea/wt-g06-3ae6f3a4 HEAD 3ae6f3a499a2c7c9d0b38d1f418cc8fe85e66054 (status --short: empty)

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
| z12 | g06 | 1681 | False | True | False | 135 | 18 | 0.952 | -- |  |  | 3.7 |
| z12 | parent | 1681 | False | True | False | 112 | 17 | 0.952 | -- |  |  | 3.1 |
| z12 | commit 128pi | 0.9084 | True | True | False | 207 | 13 | 0.407 | 161 | root 5.636713876 | 5.6347453, 5.6386834, 128 | 5.4 |
| z12 | commit 2pi | 0.7181 | True | True | False | 70 | 13 | 0.52 | 15 | root 5.636713876 | 5.6364678, 5.63696, 2 | 1.8 |
| z12 | commit 32pi | 0.7621 | True | True | False | 175 | 14 | 0.489 | 47 | root 5.636713876 | 5.6357295, 5.6376985, 32 | 5.5 |
| z12 | commit 512pi | 0.9299 | True | True | False | 587 | 11 | 0.393 | 553 | root 5.636713876 | 5.6327776, 5.6406537, 512 | 16.5 |
| z12 | commit 8pi | 0.001209 | True | True | False | 194 | 15 | 0.906 | 20 | root 5.636713876 | 5.6362216, 5.6372062, 8 | 5.4 |
| z00 | g06 | 0.01831 | True | False | False | 249 | 20 | 0.696 | -- |  |  | 6.5 |
| z00 | parent | 0.01828 | True | False | False | 225 | 20 | 0.696 | -- |  |  | 5.5 |
| z00 | commit 128pi | 0.9084 | True | True | False | 211 | 13 | 0.407 | 161 | root 5.636713876 | 5.6347453, 5.6386834, 128 | 5.8 |
| z00 | commit 2pi | 0.7181 | True | True | False | 71 | 13 | 0.52 | 15 | root 5.636713876 | 5.6364678, 5.63696, 2 | 1.5 |
| z00 | commit 32pi | 0.7621 | True | True | False | 178 | 14 | 0.489 | 47 | root 5.636713876 | 5.6357295, 5.6376985, 32 | 4.9 |
| z00 | commit 512pi | 0.9299 | True | True | False | 587 | 11 | 0.393 | 553 | root 5.636713876 | 5.6327776, 5.6406537, 512 | 16.4 |
| z00 | commit 8pi | 0.001209 | True | True | False | 196 | 15 | 0.906 | 20 | root 5.636713876 | 5.6362216, 5.6372062, 8 | 5.0 |
| w00 | g06 | 0.005418 | True | False | False | 141 | 20 | 0.994 | -- |  |  | 3.7 |
| w00 | parent | 0.005418 | True | False | False | 123 | 20 | 0.994 | -- |  |  | 3.0 |
| w00 | commit 128pi | 0.05915 | True | True | False | 221 | 11 | 0.715 | 173 | root 9.231349907 | 9.1915236, 9.272473, 128 | 7.3 |
| w00 | commit 2pi | 0.03617 | True | True | False | 70 | 13 | 0.652 | 14 | root 9.231349907 | 9.2263026, 9.2364175, 2 | 1.4 |
| w00 | commit 32pi | 0.0007818 | True | True | False | 161 | 15 | 0.875 | 62 | root 9.231349907 | 9.2112804, 9.2517434, 32 | 4.4 |
| w00 | commit 512pi | 0.733 | True | True | False | 604 | 3 | 0.434 | 601 | root 9.231349907 | 9.1529054, 9.3149916, 512 | 18.7 |
| w00 | commit 8pi | 0.0009305 | True | True | False | 78 | 11 | 0.607 | 33 | root 9.231349907 | 9.2212754, 9.2415054, 10 | 2.1 |
| w01 | g06 | 0.8174 | True | True | False | 45 | 15 | 0.0278 | -- |  |  | 0.7 |
| w01 | parent | 0.8174 | True | True | False | 38 | 15 | 0.0278 | -- |  |  | 0.6 |
| w01 | commit 128pi | 0.02275 | True | True | False | 189 | 8 | 0.227 | 163 | root 9.231349907 | 9.1915236, 9.272473, 129 | 5.5 |
| w01 | commit 2pi | 0.1285 | True | True | False | 46 | 11 | 0.586 | 20 | root 9.231349907 | 9.2263026, 9.2364175, 2 | 1.1 |
| w01 | commit 32pi | 0.001212 | True | True | False | 149 | 16 | 0.295 | 66 | root 9.231349907 | 9.2112804, 9.2517434, 32 | 4.1 |
| w01 | commit 512pi | 0.001186 | True | True | False | 596 | 8 | 0.13 | 567 | root 9.231349907 | 9.1529054, 9.3149916, 513 | 16.8 |
| w01 | commit 8pi | 0.08556 | True | True | False | 88 | 13 | 0.864 | 23 | root 9.231349907 | 9.2212754, 9.2415054, 8 | 2.3 |
| w02 | g06 | 0.003166 | True | False | False | 84 | 20 | 0.573 | -- |  |  | 1.8 |
| w02 | parent | 0.003166 | True | True | False | 75 | 17 | 0.573 | -- |  |  | 1.6 |
| w02 | commit 128pi | 0.009746 | True | True | False | 262 | 13 | 0.206 | 175 | root 9.231349907 | 9.1915236, 9.272473, 129 | 7.2 |
| w02 | commit 2pi | 0.03499 | True | True | False | 76 | 13 | 0.369 | 4 | root 9.231349907 | 9.2263026, 9.2364175, 2 | 1.7 |
| w02 | commit 32pi | 0.006926 | True | True | False | 202 | 15 | 0.411 | 61 | root 9.231349907 | 9.2112804, 9.2517434, 32 | 5.2 |
| w02 | commit 512pi | 0.6975 | True | True | False | 618 | 2 | 0.247 | 615 | root 9.231349907 | 9.1529054, 9.3149916, 513 | 18.5 |
| w02 | commit 8pi | 0.06099 | True | True | False | 88 | 14 | 0.43 | 21 | root 9.231349907 | 9.2212754, 9.2415054, 8 | 2.0 |
| w03 | g06 | 0.5732 | True | False | False | 202 | 20 | 0.0989 | -- |  |  | 7.4 |
| w03 | parent | 0.5731 | True | False | False | 139 | 20 | 0.0989 | -- |  |  | 4.1 |
| w03 | commit 128pi | 0.06936 | True | True | False | 206 | 9 | 0.198 | 176 | root 9.231349907 | 9.2088002, 9.2543095, 130 | 5.5 |
| w03 | commit 2pi | 0.07263 | True | True | False | 185 | 16 | 0.109 | 9 | root 9.231349907 | 9.2285091, 9.2341971, 2 | 5.2 |
| w03 | commit 32pi | 0.01605 | True | True | False | 117 | 14 | 0.772 | 59 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 3.0 |
| w03 | commit 512pi | 0.1063 | True | True | False | 578 | 2 | 0.717 | 576 | root 9.231349907 | 9.1866443, 9.277696, 512 | 17.7 |
| w03 | commit 8pi | 0.0136 | True | True | False | 76 | 13 | 0.443 | 27 | root 9.231349907 | 9.2256747, 9.2370507, 10 | 2.0 |
| w04 | g06 | 0.001819 | True | False | False | 105 | 20 | 0.178 | -- |  |  | 2.5 |
| w04 | parent | 0.001819 | True | False | False | 93 | 20 | 0.178 | -- |  |  | 2.0 |
| w04 | commit 128pi | 0.01718 | True | True | False | 255 | 13 | 0.178 | 171 | root 9.231349907 | 9.2088002, 9.2543095, 130 | 8.0 |
| w04 | commit 2pi | 0.0003702 | True | True | False | 133 | 16 | 0.15 | 6 | root 9.231349907 | 9.2285091, 9.2341971, 2 | 3.5 |
| w04 | commit 32pi | 0.009475 | True | True | False | 184 | 14 | 0.438 | 46 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 4.4 |
| w04 | commit 512pi | 0.03063 | True | True | False | 649 | 13 | 0.113 | 622 | root 9.231349907 | 9.1866443, 9.277696, 512 | 18.6 |
| w04 | commit 8pi | 0.002084 | True | True | False | 95 | 15 | 0.534 | 25 | root 9.231349907 | 9.2256747, 9.2370507, 10 | 2.5 |
| w05 | g06 | 0.05871 | True | False | False | 115 | 20 | 0.192 | -- |  |  | 2.6 |
| w05 | parent | 0.05871 | True | False | False | 95 | 20 | 0.192 | -- |  |  | 2.1 |
| w05 | commit 128pi | 0.005365 | True | True | False | 266 | 13 | 0.187 | 155 | root 9.231349907 | 9.2088002, 9.2543095, 129 | 6.9 |
| w05 | commit 2pi | 0.01868 | True | True | False | 51 | 9 | 0.763 | 4 | root 9.231349907 | 9.2285091, 9.2341971, 2 | 1.2 |
| w05 | commit 32pi | 0.008007 | True | True | False | 96 | 9 | 0.567 | 46 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 2.0 |
| w05 | commit 512pi | 0.1511 | True | True | False | 549 | 7 | 0.0946 | 533 | root 9.231349907 | 9.1866443, 9.277696, 512 | 15.0 |
| w05 | commit 8pi | 0.0713 | True | True | False | 152 | 15 | 0.284 | 25 | root 9.231349907 | 9.2256747, 9.2370507, 9 | 3.5 |
| w06 | g06 | 0.1162 | True | True | False | 39 | 11 | 0.579 | -- |  |  | 0.5 |
| w06 | parent | 0.1162 | True | True | False | 38 | 11 | 0.579 | -- |  |  | 0.4 |
| w06 | commit 128pi | 0.002846 | True | True | False | 227 | 14 | 0.536 | 148 | root 9.231349907 | 9.2088002, 9.2543095, 128 | 6.4 |
| w06 | commit 2pi | 0.06701 | True | True | False | 76 | 11 | 0.743 | 8 | root 9.231349907 | 9.2285091, 9.2341971, 2 | 1.5 |
| w06 | commit 32pi | 0.005276 | True | True | False | 122 | 12 | 0.656 | 57 | root 9.231349907 | 9.2200249, 9.2427774, 32 | 3.1 |
| w06 | commit 512pi | 0.001305 | True | True | False | 602 | 7 | 0.359 | 584 | root 9.231349907 | 9.1866443, 9.277696, 514 | 19.7 |
| w06 | commit 8pi | 0.0009969 | True | True | False | 66 | 12 | 0.526 | 20 | root 9.231349907 | 9.2256747, 9.2370507, 8 | 1.5 |
| w07 | g06 | 0.9424 | True | False | False | 208 | 20 | 0.241 | -- |  |  | 5.5 |
| w07 | parent | 0.9424 | True | False | False | 175 | 20 | 0.241 | -- |  |  | 6.4 |
| w07 | commit 128pi | 0.001007 | True | True | False | 213 | 8 | 0.701 | 170 | root 9.231349907 | 9.2186199, 9.2442095, 128 | 5.3 |
| w07 | commit 2pi | 0.002971 | True | True | False | 84 | 13 | 0.76 | 15 | root 9.231349907 | 9.2297516, 9.2329502, 4 | 3.7 |
| w07 | commit 32pi | 0.002942 | True | True | False | 126 | 15 | 0.663 | 54 | root 9.231349907 | 9.2249689, 9.2377633, 34 | 2.6 |
| w07 | commit 512pi | 0.001518 | True | True | False | 677 | 14 | 0.497 | 590 | root 9.231349907 | 9.2060166, 9.2572017, 514 | 19.2 |
| w07 | commit 8pi | 0.003618 | True | True | False | 94 | 11 | 0.577 | 23 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 2.0 |
| w08 | g06 | 0.01027 | True | False | False | 176 | 20 | 0.159 | -- |  |  | 4.2 |
| w08 | parent | 0.01027 | True | False | False | 151 | 20 | 0.159 | -- |  |  | 3.5 |
| w08 | commit 128pi | 0.2082 | True | True | False | 190 | 10 | 0.304 | 170 | root 9.231349907 | 9.2186199, 9.2442095, 128 | 6.5 |
| w08 | commit 2pi | 0.08832 | True | True | False | 31 | 9 | 0.953 | 3 | root 9.231349907 | 9.2297516, 9.2329502, 2 | 0.6 |
| w08 | commit 32pi | 0.1129 | True | True | False | 67 | 6 | 0.657 | 51 | root 9.231349907 | 9.2249689, 9.2377633, 32 | 2.2 |
| w08 | commit 512pi | 0.09996 | True | True | False | 625 | 15 | 0.263 | 581 | root 9.231349907 | 9.2060166, 9.2572017, 513 | 18.3 |
| w08 | commit 8pi | 0.1248 | True | True | False | 55 | 10 | 0.658 | 21 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 1.3 |
| w09 | g06 | 0.06182 | True | False | False | 109 | 20 | 0.88 | -- |  |  | 3.3 |
| w09 | parent | 0.06182 | True | False | False | 107 | 20 | 0.88 | -- |  |  | 3.3 |
| w09 | commit 128pi | 0.2017 | True | True | False | 206 | 10 | 0.304 | 186 | root 9.231349907 | 9.2186199, 9.2442095, 128 | 5.2 |
| w09 | commit 2pi | 0.1256 | True | True | False | 36 | 9 | 0.642 | 11 | root 9.231349907 | 9.2297516, 9.2329502, 2 | 0.8 |
| w09 | commit 32pi | 0.07397 | True | True | False | 70 | 6 | 0.885 | 54 | root 9.231349907 | 9.2249689, 9.2377633, 32 | 1.9 |
| w09 | commit 512pi | 0.1265 | True | True | False | 597 | 15 | 0.23 | 560 | root 9.231349907 | 9.2060166, 9.2572017, 512 | 16.2 |
| w09 | commit 8pi | 0.141 | True | True | False | 55 | 10 | 0.56 | 21 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 1.3 |
| w10 | g06 | 0.3912 | True | False | False | 137 | 20 | 0.326 | -- |  |  | 3.6 |
| w10 | parent | 0.3912 | True | False | False | 126 | 20 | 0.326 | -- |  |  | 3.4 |
| w10 | commit 128pi | 0.2238 | True | True | False | 203 | 10 | 0.276 | 184 | root 9.231349907 | 9.2186199, 9.2442095, 130 | 6.2 |
| w10 | commit 2pi | 0.09232 | True | True | False | 39 | 8 | 0.854 | 13 | root 9.231349907 | 9.2297516, 9.2329502, 2 | 0.8 |
| w10 | commit 32pi | 0.1006 | True | True | False | 67 | 6 | 0.695 | 52 | root 9.231349907 | 9.2249689, 9.2377633, 32 | 1.9 |
| w10 | commit 512pi | 0.1346 | True | True | False | 657 | 15 | 0.221 | 618 | root 9.231349907 | 9.2060166, 9.2572017, 512 | 20.9 |
| w10 | commit 8pi | 0.1013 | True | True | False | 61 | 10 | 0.774 | 28 | root 9.231349907 | 9.2281554, 9.2345525, 10 | 1.3 |
| w11 | g06 | 0.0472 | True | False | False | 121 | 20 | 0.219 | -- |  |  | 3.2 |
| w11 | parent | 0.0472 | True | False | False | 111 | 20 | 0.219 | -- |  |  | 2.4 |
| w11 | commit 128pi | 0.1894 | True | True | False | 176 | 10 | 0.344 | 157 | root 9.231349907 | 9.2186199, 9.2442095, 129 | 6.6 |
| w11 | commit 2pi | 0.2176 | True | True | False | 38 | 9 | 0.414 | 13 | root 9.231349907 | 9.2297516, 9.2329502, 4 | 0.9 |
| w11 | commit 32pi | 0.1024 | True | True | False | 86 | 13 | 0.764 | 55 | root 9.231349907 | 9.2249689, 9.2377633, 34 | 3.5 |
| w11 | commit 512pi | 0.06538 | True | True | False | 641 | 8 | 0.326 | 623 | root 9.231349907 | 9.2060166, 9.2572017, 513 | 19.2 |
| w11 | commit 8pi | 0.1182 | True | True | False | 56 | 10 | 0.742 | 26 | root 9.231349907 | 9.2281554, 9.2345525, 8 | 1.2 |
| r55 | g06 | 0.02665 | True | True | False | 1 | 0 | 0.00112 | -- |  |  | 0.0 |
| r55 | parent | 0.02665 | True | True | False | 1 | 0 | 0.00112 | -- |  |  | 0.0 |
| r55 | commit 128pi | 0.02665 | True | True | False | 1 | 0 | 0.00112 | 0 |  |  | 0.0 |
| r55 | commit 2pi | 0.02665 | True | True | False | 1 | 0 | 0.00112 | 0 |  |  | 0.0 |
| r55 | commit 32pi | 0.02665 | True | True | False | 1 | 0 | 0.00112 | 0 |  |  | 0.0 |
| r55 | commit 512pi | 0.02665 | True | True | False | 1 | 0 | 0.00112 | 0 |  |  | 0.0 |
| r55 | commit 8pi | 0.02665 | True | True | False | 1 | 0 | 0.00112 | 0 |  |  | 0.0 |

## Section 2.5 on these calls

| call | mode | converged | node_limited | max depth | leaves by class: count, sum e', at depth_max | QUADPACK iroff1, iroff2, ier=2 at bisection |
|---|---|---|---|---|---|---|
| z00 | g06 | False | False | 20 | Clenshaw-Curtis 162, 2.68e-34, 78; Levin 87, 8.53e-34, 0 | 131, 129, 7 |
| z00 | parent | False | False | 20 | Clenshaw-Curtis 139, 2.68e-34, 76; Levin 86, 8.53e-34, 0 | 125, 123, 7 |
| w00 | g06 | False | False | 20 | Clenshaw-Curtis 88, 1.89e-29, 4; Levin 53, 3.71e-29, 0 | 64, 64, 18 |
| w00 | parent | False | False | 20 | Clenshaw-Curtis 73, 1.89e-29, 2; Levin 50, 3.71e-29, 0 | 55, 55, 18 |
| w02 | g06 | False | False | 20 | Clenshaw-Curtis 30, 9.91e-30, 2; Levin 54, 4.30e-29, 0 | 43, 41, 21 |
| w03 | g06 | False | False | 20 | Clenshaw-Curtis 189, 3.02e-35, 38; Levin 13, 5.27e-31, 0 | 89, 91, 27 |
| w03 | parent | False | False | 20 | Clenshaw-Curtis 126, 4.96e-35, 8; Levin 13, 5.27e-31, 0 | 63, 65, 28 |
| w04 | g06 | False | False | 20 | Clenshaw-Curtis 43, 1.16e-32, 4; Levin 62, 8.90e-31, 0 | 60, 59, 8 |
| w04 | parent | False | False | 20 | Clenshaw-Curtis 32, 1.16e-32, 2; Levin 61, 8.90e-31, 0 | 50, 49, 8 |
| w05 | g06 | False | False | 20 | Clenshaw-Curtis 56, 1.51e-35, 18; Levin 59, 9.68e-31, 0 | 52, 53, 18 |
| w05 | parent | False | False | 20 | Clenshaw-Curtis 37, 5.95e-35, 2; Levin 58, 9.68e-31, 0 | 47, 48, 18 |
| w07 | g06 | False | False | 20 | Clenshaw-Curtis 142, 2.55e-36, 22; Levin 66, 9.96e-33, 0 | 103, 104, 22 |
| w07 | parent | False | False | 20 | Clenshaw-Curtis 110, 2.47e-36, 8; Levin 65, 9.96e-33, 0 | 88, 89, 22 |
| w08 | g06 | False | False | 20 | Clenshaw-Curtis 115, 4.76e-37, 22; Levin 61, 2.68e-32, 0 | 77, 80, 15 |
| w08 | parent | False | False | 20 | Clenshaw-Curtis 91, 1.75e-36, 14; Levin 60, 2.68e-32, 0 | 66, 69, 15 |
| w09 | g06 | False | False | 20 | Clenshaw-Curtis 85, 3.68e-37, 18; Levin 24, 1.49e-31, 0 | 46, 46, 24 |
| w09 | parent | False | False | 20 | Clenshaw-Curtis 83, 5.92e-37, 14; Levin 24, 1.49e-31, 0 | 46, 46, 24 |
| w10 | g06 | False | False | 20 | Clenshaw-Curtis 123, 5.19e-37, 22; Levin 14, 5.54e-32, 0 | 65, 64, 17 |
| w10 | parent | False | False | 20 | Clenshaw-Curtis 112, 1.03e-36, 14; Levin 14, 5.54e-32, 0 | 58, 57, 17 |
| w11 | g06 | False | False | 20 | Clenshaw-Curtis 62, 2.56e-37, 24; Levin 59, 3.56e-32, 0 | 66, 67, 9 |
| w11 | parent | False | False | 20 | Clenshaw-Curtis 58, 3.98e-37, 22; Levin 53, 3.56e-32, 0 | 60, 61, 9 |

Wall time 208 s; uptime at end: 11:56  up 8 days, 14:17, 1 user, load averages: 57.03 52.61 37.77

