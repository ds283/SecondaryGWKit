# Files of prompt 06m's Part B kept outside this directory

These stay in the Mac Studio's main checkout,
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06m/`. Everything in this
directory's `rehearsal/`, `census/` and `review/` is also in the laptop's `var/levin-phase-floor-06m/`.
They were left out as 06g's were: every pickle, every file over 2 MB, and the readers'
`rehearsal/extract/`. The 393 files copied match the Mac Studio's SHA-256. Each agent's own list,
with SHA-256 and sizes, is its `KEPT_ON_MAC_STUDIO.txt`:

| List | SHA-256 |
|---|---|
| [`rehearsal/`](rehearsal/KEPT_ON_MAC_STUDIO.txt) | `5ce60bdc…` |
| [`census/`](census/KEPT_ON_MAC_STUDIO.txt) | `2efec35d…` |

**Kept on the Mac Studio only:**
- **The R25 and R26 stores,** `var/datastores/levin-phase-floor-06m-r2{5,6}*.sqlite`, ten files.
  Their primaries are `d74ac31d…` and `e90e26c2…`. They are kept until the user says otherwise.
- **`rehearsal/extract/`,** the readers' output on R26: 3,171 files, 179 MB. These are unseeded
  random subsamples, not reproducible file for file.
- **The rehearsal's three value-diff JSONs** (`rehearsal/out/`):
  - `store_value_diff.json`, R25 against R26 (`d67e696f…`, 41 MB);
  - `store_value_diff_r24_r25.json` (`d9445335…`, 40 MB);
  - `store_value_diff_r24_r25.labels_normalised.json` (`117f0991…`, 27 MB).
- **The census's pickles and large files,** 19 files, 98 MB:
  - `census/out/gate_m/compare_m.json` (41 MB), Gate M's full comparison;
  - `census/state/index_routeB.json` (10 MB), route B's census indexed by call;
  - `census/state/rows.json` (`aa523d28…`, 06g's copy);
  - `census/state/k0/tree/units.jsonl` and the K0 region pickles.
- **The review's two re-run pickles** (`review/stop2/rerun/`), from the Mac Studio orchestrator's
  re-run of two K0 units at planned stop 2:
  - `tree-1e-7-k3_1/regions_tree-1e-7-k3_1.pkl` (`e0e9c6f4…`, 579,268 bytes);
  - `tree-1e-8-k2_302/regions_tree-1e-8-k2_302.pkl` (`ef20cf82…`, 344,771 bytes).

**The run directories.** The census's, `var/runs/levin-phase-floor-06m-census-20261106T102153/`
(150 files, 4.41 GB), stays on the Mac Studio in full: the checkpoint `ea5bc3ab…`, the region
pickles, the events and the worker logs. **It is 07's "before"**, in place of 06g's census. These
small run directories and launch logs are copied to the laptop's `var/runs/`, and match by SHA-256:
- R25's and R26's, with `levin-phase-floor-06m-r2{5,6}.launch.log`;
- K0's (Gate M0), `levin-phase-floor-06m-k0-20261106T101118/`.

**Read only and unchanged** (hashed before and after, `review/hashes_before/` and
`review/hashes_after/`): A3 v2's scratch copy, R23's and R24's stores and run directories, route B's
census (`5cb3d3c0…`) and 06g's census (`03adcbba…`).

**Worktrees** (`/Users/ds283/Documents/Code/SecondaryGWKit-lpf06m-…`). On the Mac Studio,
`5ff8a5e` (R25) and `38f6a82` (R26, K0 and the census) stay until the user says otherwise. Both are
clean with `--ignored`. 06g's four worktrees are unchanged.
