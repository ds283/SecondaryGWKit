# Files of prompt 07a's Part C kept outside this directory

These stay in the Mac Studio's main checkout, `/Users/ds283/Documents/Code/SecondaryGWKit/`. Everything in this
directory's `rehearsal/`, `census/` and `review/` is also in the laptop's `var/levin-phase-floor-07a/`. They were
left out as 07's were: every pickle, every file over 2 MB, and the readers' `rehearsal/extract/`. The copy-back list
was made on the Mac Studio before anything was copied. The 781 files copied, and the 15 run-directory files copied
into the laptop's `var/runs/`, match the Mac Studio's SHA-256. The 3,219 files left out agree with the agents' own
lists. Each agent's list, with SHA-256 and sizes, is its `KEPT_ON_MAC_STUDIO.txt`:

| List | SHA-256 |
|---|---|
| [`rehearsal/`](rehearsal/KEPT_ON_MAC_STUDIO.txt) (R30) | `28bdf1be…` |
| [`census/`](census/KEPT_ON_MAC_STUDIO.txt) | `0ff1610a…` |

Part A's list is [`measurement/KEPT_ON_MAC_STUDIO.txt`](measurement/KEPT_ON_MAC_STUDIO.txt).

**Kept on the Mac Studio only:**
- **The R30 store,** `var/datastores/levin-phase-floor-07a-r30*.sqlite`, five files, primary `e5f8b443…`. It is
  kept until the user says otherwise, with R29's (`levin-phase-floor-07-r29*`, 07's list).
- **07a's census run directory,** `var/runs/levin-phase-floor-07a-census-20261112T132744/`, checkpoint
  `461df2ec…`. **It is prompt 08's "before"** (Decisions, **07a**). The census's list gives every file.
- **The readers' extract,** `rehearsal/extract/`: 3,091 files, the readers' output on R30. They are unseeded random
  subsamples, so they cannot be reproduced file for file.
- **The rehearsal's large files** (28 under `rehearsal/state/`, 2 under `rehearsal/out/`): the replay's captures and
  inputs, and the value-diff JSON of R29 against R30 (`f6de2ef1…`, which the Mac Studio review reproduced byte for
  byte) with its copy whose job names are normalised.
- **The census's pickles and large files** (86 under `census/state/`, 5 under `census/out/`): K0's and the
  references' captures, the index of 07's checkpoint, `state/score_07a_units.json` (94 MB), and stop 3's
  comparison by call, `out/stop3/compare_census_calls.calls.jsonl` (334 MB) with its `.json` (22 MB).
- **The Mac Studio review's large files** (7): `review/stop1/svd_r29_r30.json`, its re-run of the value diff;
  four region pickles under `review/stop2/`, named in its runs' `units.jsonl`; and `review/stop3/`'s indices of the
  two checkpoints, with their SHA-256 in `stop3/sha256.txt`.
- **The k0 and references run directories** are copied whole into the laptop's `var/runs/`, with R30's run
  directory and launch log.

**Also kept, from earlier prompts,** until the user says otherwise: 07's census run directory
(`levin-phase-floor-07-census-20261108T171137`, the "before" of this census) and A3 v2's scratch copy (for 08).
