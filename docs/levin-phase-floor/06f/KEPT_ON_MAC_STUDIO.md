# Files of prompt 06f kept outside this directory

These stay in the Mac Studio's main checkout, under
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06f/` (paths below are relative
to it). They were left out of this directory as 06e's were: every pickle and every file over 2 MB.

**Kept on the Mac Studio only, never copied to the laptop:** the 90 files that
[`KEPT_ON_MAC_STUDIO.txt`](KEPT_ON_MAC_STUDIO.txt) lists with SHA-256 and size, 34.9 MB in all.
The Mac Studio review re-checked every hash and size against the files, with 0 mismatches
(`review/checks/kept.txt`). They are:
- the 82 capture pickles of the 41 sampled calls, `state/capture/C00/` to `C40/`
  (`inputs_CNN.pkl` and `regions_CNN.pkl`), which `x1_capture.py` wrote and which unpickle with
  06e's census `scripts/` on `PYTHONPATH`;
- the 6 development-capture pickles under `state/dev/cap1/` and `state/dev/cap2/`;
- Z2's per-leaf tables `out/replay/g-1e-8-0-33995_5/leaves.json.gz` and
  `out/replay/g-1e-8-0-33995_5_det2/leaves.json.gz`, 2.4 MB each, identical
  (`8741e4fa…`). Their `result.json` and `solves_cc.json.gz` are in this directory.

**Added past the laptop's `*.csv` exclusion:** the six tables `out/f0.csv`, `m1_calls.csv`,
`m1_z2_by_depth.csv`, `m1_z2_subtrees.csv`, `m2_calls.csv` and `m2_leaves.csv`. Python's `csv`
wrote them with CRLF line endings, and git stores them with LF (`core.autocrlf=input`, as for 06e's
`qsi_rows.csv`). So the committed bytes differ from the files whose SHA-256 `out/cmd/43` prints;
those originals are still on the Mac Studio, and the laptop's working copies are byte-identical to
them.

**The run directory** `var/runs/levin-phase-floor-06f-measure-20261031T204803/` (44 KB) stays there:
the manifest, `status.json` and the driver's logs. The run was registered with `runregistrykit`
and finished `done`, 44 of 44 units, exit 0.

**Read, not written, by 06f:** 06e's census checkpoint
`var/runs/levin-phase-floor-06e-census-20261030T081318/checkpoint.jsonl`
(`a0dc4d7f5c47eb28d8691b450559d005a76e1d9f3bd41442e249a419fb15904c`, unchanged before and after,
`06e/KEPT_ON_MAC_STUDIO.md`), and 06's scratch copy of A3 v2,
`var/levin-phase-floor-06/wider/state/a3v2_copy/`, equal to its certification by SHA-256 before and
after (`06/KEPT_ON_MAC_STUDIO.md`), still kept until the user says otherwise.

**Checked from the laptop on 2026-10-31,** over SSH, on the kept files: Z2's
`leaves.json.gz` and `solves_cc.json.gz`, read by the laptop orchestrator's own reader, give
$\sum|E| =$ `0x1.b4186452c4a69p-110` over the 8,600 returned Clenshaw–Curtis leaves, equal bit for
bit to the census checkpoint's `cc_floor.sums_all_cc_leaves.abs_E` for `g-1e-8-0-33995` call 5,
and $|\sum E| =$ `0x1.028b75b5aee8cp-115`, a ratio of 53.975.

**The worktree** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06f-f4770d1` (detached at
`f4770d13`) is kept until the user says to remove it.
