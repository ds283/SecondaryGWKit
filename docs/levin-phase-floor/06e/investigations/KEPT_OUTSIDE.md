# Investigation records kept outside this directory

`floor-breakdown/` and `07-leaves/` are the laptop orchestrator's investigations of 2026-10-29 and
2026-10-30 that prompt 06e rests on: the floor breakdown on 06d's code, and the leaf study of 07's
stuck cases on 07's rebased tree (`07-leaves/07-staged.patch`, SHA-256
`722c7a477f8038946dc3bfb18ed3bbc41b3a574317022012aecc28203207e365`, 07's work as rebased onto 06d's records commit `e5764a6`). They were
run on the laptop and copied to the Mac Studio's `var/`. Six files over 2 MB, the per-leaf
dumps of the scaling runs, were left out; they stay in
`var/levin-phase-floor-floor-breakdown/` on both machines:

```
393bd4c8520b5d83745ca1d0d2293e12e76e7f4d0bd9309d5bc06f0ddaf162cf  scaling_leaves_c1e6_omega0_d12.json
808f1374ce38e61ee241ead45a6507ded62a209abce64a566a4b278d707ad410  scaling_leaves_c1e6_omega0_d14.json
a1379fdf0ccada098789c5991a4bd4e530547fc88b25b1bd0bad4ab0a3503021  scaling_leaves_c1e8_omega0_d12.json
ee322646a69a1964ab5fa7b0f69c7d497043354a4b93d0bdac13fb7c65a5e6d4  scaling_leaves_c1e8_omega0_d14.json
0b7d891a62e4dcb2330fb97b9cda59d100fc7969b05e2da9c71a8e30c851343f  scaling_leaves_c1e8_omega1_d12.json
082fd838b0f260800994dd0e0b1e2eeb3da31b407734221b72de9c044fdf1356  scaling_leaves_c1e8_omega1_d14.json
```
