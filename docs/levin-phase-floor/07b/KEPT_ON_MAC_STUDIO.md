# Files of prompt 07b's Part C kept outside this directory

These stay in the Mac Studio's main checkout, `/Users/ds283/Documents/Code/SecondaryGWKit/`. Everything in this
directory's `rehearsal/`, `census/` and `review/` is also in the laptop's `var/levin-phase-floor-07b/`. They were
left out as 07a's were: every pickle, every file over 2 MB, and the readers' `rehearsal/extract/`. The copy-back list
was made on the Mac Studio before anything was copied. The 763 files copied, and the 20 run-directory files copied
into the laptop's `var/runs/`, match the Mac Studio's SHA-256. The 3,302 files left out agree with the agents' own
lists: the census's 54 file by file, the rehearsal's 51 each named or inside a directory its list keeps whole, the review's 7
named below, the readers' 3,189 and one regenerable module. Each agent's list, with SHA-256 and sizes, is its
`KEPT_ON_MAC_STUDIO.txt`:

| List | SHA-256 |
|---|---|
| [`rehearsal/`](rehearsal/KEPT_ON_MAC_STUDIO.txt) (R31) | `1454e8eb…` |
| [`census/`](census/KEPT_ON_MAC_STUDIO.txt) | `98f66fd0…` |

Part A's list is [`measurement/KEPT_ON_MAC_STUDIO.txt`](measurement/KEPT_ON_MAC_STUDIO.txt).

**Kept on the Mac Studio only:**
- **The R31 store,** `var/datastores/levin-phase-floor-07b-r31*.sqlite`, five files, primary `5981563b…`. It is
  kept until the user says otherwise, with R29's (`levin-phase-floor-07-r29*`, 07's list) and R30's
  (`levin-phase-floor-07a-r30*`, 07a's list).
- **07b's census run directory,** `var/runs/levin-phase-floor-07b-census-20261117T010200/`, 150 files, checkpoint
  `396784bc…`. **It is prompt 08's "before"**, in place of 07a's, which is also kept. The census's list gives every
  file.
- **The readers' extract,** `rehearsal/extract/`: 3,189 files, the readers' output on R31. They are unseeded random
  subsamples, so they cannot be reproduced file for file.
- **The rehearsal's large files and captures** (51, inside `rehearsal/state/capture/`, `rehearsal/state/inputs/` and
  `rehearsal/out/`): the replay's and the references' captures, the value-diff JSON of R30 against R31
  (`b7d508c1…`, which the Mac Studio review reproduced byte for byte) with its copy whose job names are normalised,
  `out/qsi_records_r30_r31.csv` and the agent's 11 MB `out/commands.log` (Part A's run driver appends each command's
  output to it). Also `rehearsal/state/variants/levin_quadrature_1ec734d7.py`, `git show
  1ec734d7:AdaptiveLevin/levin_quadrature.py`, reproducible from git.
- **The census's pickles and large files** (54 under `census/`): K0's, the smoke runs' and U0's region pickles,
  the index of 07a's checkpoint, `state/score_07b_units.json`, `state/rows.json`, and stop 3's comparison by call,
  `out/stop3/compare_census_calls.calls.jsonl` (599 MB) with its `.json` (15 MB).
- **The Mac Studio review's large files** (7): `review/stop1/svd.json`, its re-run of the value diff; four region
  pickles under `review/stop2/`; and `review/stop3/`'s indices of the two checkpoints, `index07a.json` and
  `index07b.json`, with their SHA-256 in `stop3/sha256.txt`.
- **The R31, R31-replay, R31-references and k0 run directories,** with R31's launch log, are copied whole into the
  laptop's `var/runs/`.

**Also kept, from earlier prompts,** until the user says otherwise: 07a's census run directory
(`levin-phase-floor-07a-census-20261112T132744`), 07's census run directory and A3 v2's scratch copy (for 08).
