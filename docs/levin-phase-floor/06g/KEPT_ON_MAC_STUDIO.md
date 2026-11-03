# Files of prompt 06g's Part B kept outside this directory

These stay in the Mac Studio's main checkout,
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06g/`, and everything in this
directory is also in the laptop's `var/levin-phase-floor-06g/`. They were left out as 06h's were:
every pickle, every file over 2 MB, and the readers' `rehearsal/extract/`. Each agent's own list,
with SHA-256 and sizes, is its `KEPT_ON_MAC_STUDIO.txt`:

| List | SHA-256 |
|---|---|
| [`rehearsal/`](rehearsal/KEPT_ON_MAC_STUDIO.txt) | `f1489bf0…` |
| [`census/`](census/KEPT_ON_MAC_STUDIO.txt) | `c8e24de6…` |
| [`references/`](references/KEPT_ON_MAC_STUDIO.txt) | `284034e7…` |

**Kept on the Mac Studio only:**
- **The R23 and R24 stores,** `var/datastores/levin-phase-floor-06g-r2{3,4}*.sqlite`, ten files.
  Their primaries are `88cc355f…` and `6e4815c8…`. They are kept until the user says otherwise.
- **`rehearsal/extract/`,** the readers' output: 3,187 files, 182 MB. These are unseeded random
  subsamples, not reproducible file for file.
- **`rehearsal/out/store_value_diff.json`** (`36cd46a4…`, 42 MB).
- **The census's large files and pickles:**
  - `census/out/v8_calls.json` (`1e9a62b0…`);
  - `census/out/v8_slow_leaves.json` (`b4643033…`);
  - `census/state/rows.json` (`aa523d28…`);
  - the Gate V and smoke region pickles.
- **The references' pickles and files over 2 MB** (218 MB). These are the capture pickles under
  `references/state/capture/` and `references/state/inputs/`, V1's under
  `references/v1/state/inputs/`, and `references/out/compare13.json`.

**Not on the list, brought over compressed:** `review/checks/var_files_after.txt` (9.2 MB, SHA-256
`5e7c09e1…`) is in this directory as `review/checks/var_files_after.txt.gz`. Its uncompressed
SHA-256 equals the Mac Studio's.

**The run directories.** The census's,
`var/runs/levin-phase-floor-06g-census-20261101T114315/` (149 files, 4.5 GB), stays on the Mac
Studio in full: the checkpoint `03adcbba…`, 16 region pickles, the events and the worker logs. It is
07's "before" and the references' source. These run directories are copied to the laptop's
`var/runs/`:
- R23's and R24's, with their launch logs;
- the references' parts 1 and 2;
- the rerun batch.

The laptop's FCC batch, `var/runs/levin-phase-floor-06g-fcc-laptop-20261102T233229/`, is in the
laptop's registry. Its readings are in `references/out/readings_fcc_b5626ec_laptop/`. Its scripts and
check inputs are in [`fcc-laptop/`](fcc-laptop/), and its 14 input captures are copies of the Mac
Studio's, each matched by SHA-256.

**Worktrees** (`/Users/ds283/Documents/Code/SecondaryGWKit-lpf06g-…`). On the Mac Studio,
`4f4f9ed`, `1c5336e`, `ff3e839` and `b5626ec` stay for the references' reproducibility until the
user says otherwise. The 71 stray `.pyc` files the laptop's slope survey left in `ff3e839`, and the
one in `references/scripts/__pycache__/`, were deleted on the user's word on 2026-11-03. The laptop
keeps its `b5626ec` worktree.
