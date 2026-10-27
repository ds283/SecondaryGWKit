# Files of prompt 06a kept on the Mac Studio only

These stay in the Mac Studio's main checkout, under
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06a/` (paths below are relative
to it). They were left out of this directory as 06's were: every pickle, and every file over 2 MB.
The run directory `var/runs/levin-phase-floor-06a-census-20261027T023143/` (210 MB, 133 files: the
checkpoint, the 16 per-worker pickle streams of region arrays, the events and the worker logs)
stays there as well. Its `checkpoint.jsonl` has SHA-256
`c13685b4afaf1dad3609fc4c5f56f1a2936ab98f25115ceda1020a300453427a`.

`out/scan_listed_regions.csv` (141 kB, the 720 regions accepted across a stationary point, for
prompt 07) is in this directory, added past the laptop's `*.csv` exclusion.

The scratch copy of A3 v2 that 06a read, `var/levin-phase-floor-06/wider/state/a3v2_copy/`, is 06's
(`06/KEPT_ON_MAC_STUDIO.md`), and is still kept until the user says otherwise.

The SHA-256 of each was taken on the Mac Studio on 2026-10-27, when the records were copied back.

```
569750d1a179065bbc085e60b651865bc0e189876692ca9851367d8257b77303  out/cmd/43_convergence_kinds_rerun.txt
8bd380fbb46ecd489c4f9274023f06b9f6cebaacb859fb420782d4513a01943d  out/commands.log
aa523d283f1dae26a5127434557ea04ca8217f8c4de058a7d415d736f17f6417  state/rows.json
6ca6e6f96c29cc11f490114946dd4678756521ba6eb9ac0cb21c84ad5d02fdaa  state/C0/W1/regions_W1.pkl
a2d89e4c4551f29a7471540db062ced0c06b9ed215d4e0f5f35add2eabbd4e02  state/C0/W2/regions_W2.pkl
5dc0c14949e5a7b67d98d1d2aa71a5505a46d8ae32a479997341605455f0a082  state/C0/W3/regions_W3.pkl
4549b5644b36c284c2716b632eafe47c1e542c12b489a05b9fedaa1b7a495daa  state/dev/smoke1/regions_1e-7-k2_302.pkl
ad1617ff38f18678735236eb97a9387cfa9b545b7ee21412bb433703366f8e64  state/pilot/regions_pilot.pkl
9125966624692b80aef1c92f47b5c7e89b3a35f8eaa1ff3d78a0922d76bdc38c  state/smoke/regions_1e-7-k2_302.pkl
c21f749e0defa3fff9c51772bb7830d290da14b263c0b2192982d826c1574b15  state/smoke/regions_1e-7-k3_201.pkl
430f850f949fd94f49b29e39f9bd15bcedfc62ad0c0c347da54c02c7a70e3353  state/smoke/regions_1e-8-k2_302.pkl
6bc5ce1133c22b4e143cbf50302b047b71035b0ecc0348dd03a5a89d1b04e5f4  state/smoke/regions_1e-8-k3_201.pkl
```

`out/cmd/43` is the full output of the changed `convergence_kinds_06a.py`. This directory's
`out/convergence_kinds_h.txt` holds the same output without its per-record input lines.
