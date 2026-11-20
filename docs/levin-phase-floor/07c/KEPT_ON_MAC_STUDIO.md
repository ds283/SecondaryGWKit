# Files of prompt 07c kept outside this directory

These stay in the Mac Studio's main checkout, `/Users/ds283/Documents/Code/SecondaryGWKit/`.

## Part A

The copy-back list was made on the Mac Studio before anything was copied: every file under
`var/levin-phase-floor-07c/measurement/` and the six registered run directories
`var/runs/levin-phase-floor-07c-*/` except pickles and files over 2 MB. Every copied file matches the Mac
Studio's SHA-256.

**Left out by the agent's rule (468 files):** every pickle and every file over 2 MB. These are listed with SHA-256
and sizes in the agent's [`measurement/KEPT_ON_MAC_STUDIO.txt`](measurement/KEPT_ON_MAC_STUDIO.txt) (`366cad95…`).
The files left out are exactly the ones it lists. One listed file, `out/commands.log`, grew after the list was
written, as the extension appended its commands. Its first 3,177,402 bytes, the length listed, still hash to the
listed `beab0152…`.

**Left out by the orchestrator (7,234 files, 1,093,862,743 bytes):** fourteen folders of `out/replays/`. The user
asked whether the replay JSON was needed, and these folders are not:
- the other variants' plain replays: `N/`, `S2/`, `Eg/`, `Ea/`, `SEg/` and `SEa/`;
- every instrumented copy: `S_i/`, `S2_i/`, `N_i/`, `Eg_i/`, `Ea_i/`, `SEg_i/` and `SEa_i/`;
- the instrument's own replays, `instr/`.

The tables they produced are committed: `out/tables/scores_07c.{json,md}`, `out/d1/`, `out/variants/reach.{json,md}`,
and each folder's `*_summary_*.json` beside it. Each is listed with SHA-256 and size in
[`KEPT_REPLAYS_ON_MAC_STUDIO.txt`](KEPT_REPLAYS_ON_MAC_STUDIO.txt) (`efa0886f…`). All 7,234 were re-hashed on the
Mac Studio against that list on 2026-11-19, and all matched.

**Kept in the repository:**
- `out/replays/tree/`: the 454 replays at 07b's code, each reproducing its census record;
- `out/replays/S/`: the replay records of the chosen fix, which §9.5's Y8 and Part B's review compare the code with
  bit for bit.

**Also kept on the Mac Studio, from earlier prompts:** R29's, R30's and R31's stores; 07a's and 07b's census run
directories; A3 v2's scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/`, for 08.

## Part C

The copy-back list was made on the Mac Studio on 2026-11-20 before anything was copied: every file under
`var/levin-phase-floor-07c/{rehearsal,census,review}/` except what the three agents' lists keep, and the R32, R32-replay
and k0 run directories with R32's launch log. **688 record files** came into the laptop's `var/levin-phase-floor-07c/`
and are in this directory's `rehearsal/`, `census/` and `review/` byte for byte; **15 run-directory files** came into the
laptop's `var/runs/`. Every one matches the Mac Studio's SHA-256.

**Left out by the agents' lists (3,307 files under the three directories):** the rehearsal's 3,178, the census's 95 and
the review's 34. Each list, with SHA-256 and size, is the agent's `KEPT_ON_MAC_STUDIO.txt`, and every hash it gives was
re-checked on the Mac Studio before the copy (325 files, all equal):

| List | SHA-256 |
|---|---|
| [`rehearsal/`](rehearsal/KEPT_ON_MAC_STUDIO.txt) (R32) | `ae5e0ca6…` |
| [`census/`](census/KEPT_ON_MAC_STUDIO.txt) | `3a97e742…` |
| [`review/`](review/KEPT_ON_MAC_STUDIO.txt) | `8f12e902…` |

- **The rehearsal's:** the readers' `extract/` (3,122 files, unseeded random subsamples), `state/capture/` and
  `state/inputs/` whole, the value diff of R31 against R32 (`c42c6d77…`, 40 MB) with its copy whose job names are
  normalised, and the protected listings.
- **The census's:** K0's, the smoke runs' and Y0's pickles and raw worker streams, `state/index_07b.json`,
  `state/rows.json`, `state/score_07c_units.json`, and stop 3's comparison by call,
  `out/stop3/compare_census_calls.calls.jsonl` (668 MB) with its `.json`.
- **The review's:** the four checkpoints' extracts and byte indices (`final/`), `stop3/calls_07{b,c}.jsonl`, its re-run
  of the value diff (`stop1/svd_r31_r32.json`, equal to the rehearsal's), three identical hash listings of the
  protected files (`checks/extra_*.sha256`) and stop 2's raw re-runs.

**Left out by the orchestrator (4 files):**
- three of the census's four listings of the files it protects. Their bodies (17,801 lines) equal
  [`census/out/runs_hashed_before.txt`](census/out/runs_hashed_before.txt)'s line for line, and each header's verdict is
  ALL HOLD; only the header's time differs:

  | File | Bytes | SHA-256 |
  |---|---|---|
  | `census/out/runs_hashed_stop2.txt` | 2,083,081 | `d27e8eb199d7d30623063214bbb9e1642a85ab10025b48ea87d78b94853a21e7` |
  | `census/out/runs_hashed_after.txt` | 2,083,081 | `9c867bf16213e2112d484ddffe5a3aa53c6a246f44ed14dc8be647364748e286` |
  | `census/out/runs_hashed_end.txt` | 2,083,079 | `1668003a3d9a60a485c03e6225db5c9f98bb3407102839b12f831302f1eccaf7` |

- `census/state/variants/levin_quadrature_parent.py` (334,493 bytes, `8956c7b6…`), which is
  `git show cff4237f:AdaptiveLevin/levin_quadrature.py` byte for byte.

**Kept on the Mac Studio only:**
- **The R32 store,** `var/datastores/levin-phase-floor-07c-r32*.sqlite`, five files, primary `4dd39837…`, fingerprint
  `1b332ffa…`, listed in the rehearsal's list.
- **07c's census run directory,** `var/runs/levin-phase-floor-07c-census-20261119T201419/`, 149 files, 10.66 GB,
  checkpoint `33d94ce2…`. **It is prompt 07d's "before".** The census's list gives every file.

**Also kept, from earlier prompts,** until the user says otherwise: R29's, R30's and R31's stores; 07a's and 07b's census
run directories; A3 v2's scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/`, for 08.
