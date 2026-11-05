# Files of prompt 06l kept outside this directory

These are kept in the Mac Studio's main checkout,
`/Users/ds283/Documents/Code/SecondaryGWKit/`, under `var/levin-phase-floor-06l/` (where the rest of
06l's records also stay) and `var/runs/`. They were left out of this directory as 06k's were:
every pickle and every file over 2 MB.

> **Mostly removed 2026-11-05, on the user's word** (*"yes, please clean up. Remove the worktree. For
> now keep the census run directory. Remove the pickles and other large files."*). The 183 files
> listed below under `var/levin-phase-floor-06l/` (the 178 in `KEPT_ON_MAC_STUDIO.txt`'s first
> section and the review's five pickles) were checked against their SHA-256, all 183 equal, and
> deleted, with the directories they left empty: 479,656 KB down to 62,428 KB. What remains under
> `var/levin-phase-floor-06l/` is the 824 files committed here.
>
> **Still kept:** route B's census run directory in full (4.4 GB, checkpoint `5cb3d3c0…`), and the
> two small runs (`k0partL` and `partL`). Everything deleted can be recreated from 06g's census, the
> scratch copy and the variant under `scripts/variants/`, all still kept; route B's compact extract
> `state/extract/routeB.pkl` from the census checkpoint by `scripts/extract_census.py`. The worktree
> was removed the same day.

**Under `var/levin-phase-floor-06l/`: 183 files, never copied to the laptop.**
- **178 are listed with SHA-256 and size** in [`KEPT_ON_MAC_STUDIO.txt`](KEPT_ON_MAC_STUDIO.txt)
  (SHA-256 `8ee0abb0…`, written by `scripts/kept_list_06l.py`), 426,383,808 bytes in all:
  - the 92 capture pickles of 06j's 46 calls, `state/capture/C00/` to `C45/`, and the 46 per-call
    inputs under `state/inputs/`;
  - Gate K0's emitted records and region pickles under `state/k0/` (`identity` and `routeB`);
  - route B's compact census extract `state/extract/routeB.pkl` (154 MB), the development unit's
    pickle under `state/dev/`, `state/index_06g.json` (and its superseded first build) and
    `state/rows.json`;
  - the LiouvilleGreen draws' record file `out/lg_draws/lg_seeded_draws.jsonl`.
- **5 are not in that list:** the region pickles of the review's own re-runs under `review/checks/`,
  which the list leaves out with the rest of `review/`:

  | SHA-256 | bytes | file |
  |---|---|---|
  | `0facd7da70232016b6b5b80258ee1eb930fc1ae18516c51b6f0be50f3832f73e` | 109,142 | `k0rerun/identity/identity-1e-7-k0_202/regions_identity-1e-7-k0_202.pkl` |
  | `90c2156ff34412e0b5083f3b3648f5ec2e058ffbb1ce19861ad57f8462069cd4` | 112,119 | `k0rerun/identity/identity-1e-8-k2_2/regions_identity-1e-8-k2_2.pkl` |
  | `9919986ef90278dcd4bc632c81c77cdbeb6566186fffc8fda20becb52fbd2954` | 110,126 | `k0rerun/routeB/routeB-1e-7-k0_202/regions_routeB-1e-7-k0_202.pkl` |
  | `31342420df00af11b03efb0c4d37ef44037f84e64b85b0bf67f3536e38bb6b3a` | 113,021 | `k0rerun/routeB/routeB-1e-8-k2_2/regions_routeB-1e-8-k2_2.pkl` |
  | `ca9c864e623a685829155a5db5c67670bb9cedd183f6f1271158414aa8a47a7e` | 73,070 | `recovered_rerun/routeB/routeB-1e-8-k3_1/regions_routeB-1e-8-k3_1.pkl` |

  Their SHA-256 were taken by the laptop orchestrator on the Mac Studio on 2026-11-05.

The Mac Studio review checked all 338 entries of the list against the files, by SHA-256 and size,
with no mismatch (`review/checks/c7_kept_check.txt`).

**The three run directories** are kept on the Mac Studio in full; every file of each is listed in
`KEPT_ON_MAC_STUDIO.txt`:

| Run | State | Files | Bytes | Checkpoint SHA-256 |
|---|---|---|---|---|
| `levin-phase-floor-06l-k0partL-20261105T091942` | `failed` (log deviation 11) | 5 | 21,796 | — |
| `levin-phase-floor-06l-partL-20261105T095407` | `done`, 48/48 | 5 | 18,721 | — |
| `levin-phase-floor-06l-census-routeB-20261105T100126` | `done`, 18,560/18,560 | 150 | 4,771,509,464 | `5cb3d3c0e6e628720aa122a1d23d4a90f140b116ef9e58e2f66a3bffb9dd0d39` |

The two small runs' manifests, statuses, logs and checkpoints, and the census's every file but its
checkpoint and its 16 region pickles, were copied to the laptop's `var/runs/`.

**What it would take to recreate them.** Everything here can be recreated from 06g's census, the
scratch copy and the variant `scripts/variants/levin_quadrature_routeB.py`, all still kept. Route
B's census is the reference for a remedy built on $\nu_B$: such a remedy should equal it on every
call where its change does not bite, as route 1's census (kept from 06k) is for one built on
$\nu_1$.
