# Files of prompt 07's Part B kept outside this directory

These stay in the Mac Studio's main checkout,
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/`. Everything in this
directory's `rehearsal/`, `census/`, `review/` and `review.md` is also in the laptop's
`var/levin-phase-floor-07/`. They were left out as 06m's were: every pickle, every file over 2 MB,
and the readers' `rehearsal/extract/` and `rehearsal/r29/extract/`. The 1,183 files copied match
the Mac Studio's SHA-256. Each agent's own list, with SHA-256 and sizes, is its
`KEPT_ON_MAC_STUDIO.txt`:

| List | SHA-256 |
|---|---|
| [`rehearsal/`](rehearsal/KEPT_ON_MAC_STUDIO.txt) (R27, R28) | `f27b4bc3…` |
| [`rehearsal/r29/`](rehearsal/r29/KEPT_ON_MAC_STUDIO.txt) (R29) | `473b7783…` |
| [`census/`](census/KEPT_ON_MAC_STUDIO.txt) | `3d7348c1…` |

**Kept on the Mac Studio only:**
- **The R29 store,** `var/datastores/levin-phase-floor-07-r29*.sqlite`, five files, primary
  `28a83aa0…`. It is kept until the user says otherwise.
- **The readers' extracts** (unseeded random subsamples, not reproducible file for file):
  - `rehearsal/extract/`, the readers' output on R28: 3,039 files, 105 MB;
  - `rehearsal/r29/extract/`, the readers' output on R29: 3,126 files, 110 MB.
- **The rehearsal's value-diff JSONs:**
  - `rehearsal/out/store_value_diff.json`, R27 against R28 (`927d4e70…`, 43 MB);
  - `rehearsal/out/store_value_diff_r26_r27.json` (`458bbce2…`, 40 MB) and its
    `.labels_normalised.json` (`77102dfe…`, 27 MB), S0;
  - `rehearsal/r29/out/store_value_diff_r27_r29.json` (`c511b582…`, 43 MB), which the review
    reproduced byte for byte;
  - `rehearsal/r29/out/store_value_diff_r28_r29.json` (`924458d5…`, 40 MB).
- **The rehearsal's captures:** `rehearsal/state/capture/` (54 files, 131 MB) and
  `rehearsal/state/inputs/` (the eight captured calls, 6.1 MB); `rehearsal/r29/state/capture/`
  (57 files, 80 MB) and `rehearsal/r29/state/inputs/` (the twelve block captures, 288 KB); and S2's
  two reading JSONs, `rehearsal/out/p1_references/p1_row{1,2}.json` (2.5 MB each), with their
  `.log` copies.
- **The census's pickles and large files,** 152 files, 1.07 GB:
  - the cross-check's captures, `census/state/xcheck/` (28 files, 710 MB);
  - V4's captures, `census/state/v4/` (75 files, 75 MB), and K0's, `census/state/k0/` (33 files,
    85 MB), with the smaller `state/dev/`, `state/smoke/` and `state/v0c/`;
  - `census/out/stop3/compare_census_calls.json` (`3bf49204…`, 147 MB), V1's comparison by call;
  - `census/out/scores/summary.json` (`eab6935c…`, 6.0 MB), every verdict's detail and every
    stopped window;
  - `census/out/xcheck/xcheck.json` (`533b2b07…`, 4.3 MB), the cross-check;
  - `census/state/score_07_units.json` (26 MB), `census/state/index_06m.json` (10 MB) and
    `census/state/rows.json` (`aa523d28…`, 06g's copy).

  `VERIFICATION.md` §7 and the census agent's `log.md` link the three JSONs under `out/`; they are
  here only.

**The run directories.** The census's, `var/runs/levin-phase-floor-07-census-20261108T171137/`
(150 files, 4.81 GB), stays on the Mac Studio in full: the checkpoint
`03dfe786f69343a0e86f436936161ace728630f26d687061e186d0883dd5cb10`, the region pickles, the events
and the worker logs. **It is 08's "before"**, in place of 06m's census. The references run
directory, `var/runs/levin-phase-floor-07-references-20261109T005328/` (5 files, checkpoint
`40a9d871…`), also stays whole on the Mac Studio. These small run directories and launch logs are
copied to the laptop's `var/runs/`, and match by SHA-256 (29 files):
- R27's, R28's and R29's, with `levin-phase-floor-07-r2{7,8,9}.launch.log`;
- `p1-references`' (S2's readings), `levin-phase-floor-07-p1-references-20261108T103132/`;
- K0's (Gate V0), `levin-phase-floor-07-k0-20261108T135240/`;
- the references', `levin-phase-floor-07-references-20261109T005328/`.

**Read only and unchanged** (hashed before and after, `review/checks/` and `review/b3/`): A3 v2,
its scratch copy, R23–R26's stores and 06m's census run directory (checkpoint `ea5bc3ab…`).

**Worktrees** (`/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-…`): `91472db` (R27 and K0 at the
parent), `7460035` (R28) and `e182bc7` (R29, the census and the references), each clean with
`--ignored` at the review.

**After Part B, on 2026-11-10,** in the user's housekeeping
([`../housekeeping-2026-11-10/README.md`](../housekeeping-2026-11-10/README.md)), on the user's word
(*"Yes, please go ahead with the cleanup as listed."*):
- **the R27 and R28 stores were retired** with `runregistrykit store retire`, each sidecar kept as
  its record, each fingerprint `1b332ffa…` recorded first. The symlinks to their shards under
  `rehearsal/state/stores/` and `rehearsal/r29/state/stores/` now dangle;
- **the three 07 worktrees were removed,** all clean.

**What stays:** R29's store, the census and references run directories,
`var/levin-phase-floor-07/` (this directory's source, with everything listed above) and A3 v2's
scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/`, for 08.
