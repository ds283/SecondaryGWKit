# Files of prompt 06e's Part B kept outside this directory

These stay in the Mac Studio's main checkout, under
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06e/` (paths below are relative
to it), and those in the hashed list below also in the laptop's `var/levin-phase-floor-06e/`. They
were left out of this directory as 06c's were: every pickle, every file over 2 MB, and the readers'
`rehearsal/extract/`.

**Kept on the Mac Studio only, never copied to the laptop:**
- `rehearsal/extract/` (179 MB);
- the captured-input pickles under `census/state/Z0/{G1,G2,P1,P2,P3}` and `census/state/smoke`;
- Z3's 13 captured inputs, `census/references/inputs/sNN.pkl`, listed with their SHA-256 below. The
  orchestrator's copies of samples 1, 2, 8, 9, 13, 22 and 25 are also in the laptop's
  `var/levin-phase-floor-06c/depth20-tiling/orchestrator_check/inputs/`, identical by SHA-256.

**The census's run directory** `var/runs/levin-phase-floor-06e-census-20261030T081318/` (1.5 GB)
stays there in full: the checkpoint, 16 region pickles, the events and the worker logs. Prompt
07's §6 compares against it. Its `checkpoint.jsonl` has SHA-256
`a0dc4d7f5c47eb28d8691b450559d005a76e1d9f3bd41442e249a419fb15904c`. The references run's
`levin-phase-floor-06e-references-20261030T192730/checkpoint.jsonl` has SHA-256
`51c20020d086ae68969fcfd5d56a360f41878ce311c91070364ff1f27685eff5`.

**The rehearsal stores** `var/datastores/levin-phase-floor-06e-r19*` and `-r20*` were retired by
the user on 2026-10-31 (`runregistrykit store retire`). Their sidecars, which record the
fingerprint `1b332ffa…` and the retirement, stay in `var/datastores/`; the run directories
`levin-phase-floor-06e-r19-parent-20261030T074440` and
`levin-phase-floor-06e-r20-endnodes-20261030T075034` stay in `var/runs/`.

**Added past the laptop's `*.csv` exclusion:** `rehearsal/out/qsi_rows.csv`, the per-row
tabulation of all 864 rows.

The scratch copy of A3 v2 that the census and the references read,
`var/levin-phase-floor-06/wider/state/a3v2_copy/`, is 06's (`06/KEPT_ON_MAC_STUDIO.md`), and is
still kept until the user says otherwise.

**Left out of this directory, on both machines**, by SHA-256 taken on the laptop on 2026-10-31:

```
cdafa6fcb04c9070c66df98cfdb0bbdce5e0d19d0df699b316861f1c4070dcae  rehearsal/out/store_value_diff.json
aa523d283f1dae26a5127434557ea04ca8217f8c4de058a7d415d736f17f6417  census/state/rows.json
```

**Z3's inputs, on the Mac Studio only**, by SHA-256 taken there on 2026-10-31:

```
7f162791c9b19f1cf15ef0b02ccd35910d0da83c8681aba7558642dfefbf8bb0  census/references/inputs/s01.pkl
0100f98a9e46f4907547d3d6bb3e1e21fb16d1dc00d4e69ddb33db0d54f3a183  census/references/inputs/s02.pkl
81cff8d83b285ca9f1851e5d56e964e5fdb05293b07202b0d3eb9799b1cf4892  census/references/inputs/s03.pkl
861bc871a2ff585df2c688288c8e8953440edee13dbd427ed806fcce5188dc80  census/references/inputs/s08.pkl
cbe9cb72607d7c19ae7d65a4b50726e9b0a0915998b5a3ca8f715e4a6d8285d3  census/references/inputs/s09.pkl
08f40073cbfe11d70d1f4fec19ee6ebc1876c95145375eb0823533bfe0bbd5af  census/references/inputs/s10.pkl
54bd746d5ac6a1fd51f31822e374a2fe60a1143fc04579a7f80cbbc4ddf96b56  census/references/inputs/s13.pkl
9570b1794986758ad0413315bb300a9cec4f13715f14fa85d0a0ef2a017eb942  census/references/inputs/s14.pkl
e9ca60604f8cb8cae4689d16362b97b438bd0e1ed103bff099f485756e786abd  census/references/inputs/s15.pkl
ad6a6a065ead3616ef8dc96d162a7fa7f4ee902ca59d5f41580d72608b03e9b1  census/references/inputs/s22.pkl
1ffb073daec0dbdb8c4961bf3508218f5c25d5ce1bbaa75f41954acee134d5ba  census/references/inputs/s23.pkl
696768378bfba3f57e0b464b887933bb828315bfe31c1a2dd540da9064969c42  census/references/inputs/s25.pkl
d5389183fa3d15c084d15d2939f4923c151541255ada9ce3fff26be7ee1fca1c  census/references/inputs/s27.pkl
```
