# Files of prompt 06k kept outside this directory

These are kept in the Mac Studio's main checkout,
`/Users/ds283/Documents/Code/SecondaryGWKit/`, under `var/levin-phase-floor-06k/` (where the rest of
06k's records also stay) and `var/runs/`. They were left out of this directory as 06j's were:
every pickle and every file over 2 MB.

> **Mostly removed 2026-11-05, on the user's word** (*"I assume the census runs, pickles, and other
> large files have no value now, unless we're likely to need to revisit the route 1 census for
> comparison purposes. If that seems possible, we should return that census and any supporting
> files, but clean out the others."*). It is possible: a remedy built on route 1 should equal route
> 1's census on every call where its change does not bite, so route 1's census is kept as the
> reference for that comparison. 315 files were checked against their SHA-256 (all 315 equal) and
> deleted:
> - every file under `var/levin-phase-floor-06k/` listed below except the two checkpoint extracts
>   `state/extract/route1.pkl` and `state/extract/g.pkl` (06g's), which are kept: 259 files;
> - the review's five re-run pickles under `review/checks/b3/`;
> - the `checkpoint.jsonl` and 16 `regions_*.pkl` of each of the censuses of route 2, route 2
>   literal and route 3: 51 files. Each of those run directories keeps its manifest, status,
>   logs, events, specs and worker records (17 MB each), which are the registry's record of the run.
>
> **Still kept:** route 1's census run directory in full (4.1 GB, checkpoint `36e3dd4c…`), the two
> extracts, and the small records under `var/levin-phase-floor-06k/` that are also committed here.
> The other three runs are kept whole. Everything deleted can be recreated from 06g's census, the
> scratch copy and the variants under `scripts/variants/`, all still kept. The worktree was removed
> the same day.

**Under `var/levin-phase-floor-06k/`: 266 files, never copied to the laptop.**
- **261 are listed with SHA-256 and size** in [`KEPT_ON_MAC_STUDIO.txt`](KEPT_ON_MAC_STUDIO.txt)
  (SHA-256 `963ea70b…`, written by `scripts/kept_list_06k.py`), 1,100,872,852 bytes in all:
  - the 92 capture pickles of 06j's 46 calls, `state/capture/C00/` to `C45/`, and the 46 per-call
    inputs under `state/inputs/`;
  - Gate K0's emitted records and region pickles under `state/k0/` (`tree`, `identity` and the four
    routes);
  - the six compact checkpoint extracts under `state/extract/`, the development unit's files under
    `state/dev/`, `state/index_06g.json` and `state/rows.json`;
  - the LiouvilleGreen draws' two record files under `out/lg_draws/`.
- **5 are not in that list:** the region pickles of the review's own re-runs under
  `review/checks/b3/`, which the list leaves out with the rest of `review/`:

  | SHA-256 | bytes | file |
  |---|---|---|
  | `ba943d2d9686ea8fe0bab1ea1515a54968b6fb21b8452942e3ff6e746abcea7d` | 94,501 | `k0rerun/identity/identity-1e-7-k3_201/regions_identity-1e-7-k3_201.pkl` |
  | `0fe1becf6f8999914bb976d8f81b9cc1a197310dba45315340c019f798f58719` | 99,343 | `k0rerun/identity/identity-1e-8-k2_2/regions_identity-1e-8-k2_2.pkl` |
  | `cf942ef3ce1b704ed8674d061d51b55eb44d9a953eb284fc0e49a3d1c6a6b894` | 95,082 | `k0rerun/route2/route2-1e-7-k3_201/regions_route2-1e-7-k3_201.pkl` |
  | `700d5900f7bd3e7dfdfbe994c98e193b4bd3b2c0c6ec720de73f22526c786208` | 99,924 | `k0rerun/route2/route2-1e-8-k2_2/regions_route2-1e-8-k2_2.pkl` |
  | `f804b20e6e08c570fbbb587058195dd1a17c681c1cc4100cabbc46126369c422` | 76,060 | `recovered_rerun/route1/route1-1e-7-k3_1/regions_route1-1e-7-k3_1.pkl` |

  Their SHA-256 were taken by the laptop orchestrator on the Mac Studio on 2026-11-05.

The Mac Studio review checked a seeded sample of 12 entries of the list, and the four census
checkpoints in full, against the files, with no mismatch (`review/checks/b3/c7_kept_sample.txt`,
`c7_census_checkpoints.sha256`).

**The seven run directories** are kept on the Mac Studio in full; every file of each is listed in
`KEPT_ON_MAC_STUDIO.txt`:

| Run | State | Files | Bytes | Checkpoint SHA-256 |
|---|---|---|---|---|
| `levin-phase-floor-06k-k0partL-20261104T091640` | `failed` (log deviation 8) | 5 | 23,030 | — |
| `levin-phase-floor-06k-lgdraws-20261104T095904` | `done` | 5 | 4,044 | — |
| `levin-phase-floor-06k-route2lit-20261104T101930` | `done` | 5 | 4,913 | — |
| `levin-phase-floor-06k-census-route1-20261104T103009` | `done`, 18,560/18,560 | 150 | 4,442,324,735 | `36e3dd4c9125f6ea620b80e1bc15d656827b5dd2e449c983d479d1577c6bed38` |
| `levin-phase-floor-06k-census-route2-20261104T151120` | `done`, 18,560/18,560 | 149 | 4,193,782,343 | `8363ae821ce41cd6a9d79c47457d0fad1c5cd5faf2ebf2f0874d6133f720463e` |
| `levin-phase-floor-06k-census-route2lit-20261104T185927` | `done`, 18,560/18,560 | 149 | 4,858,051,010 | `7fb9c41e996ef03309b03e83b3ee72180f2fd5f9d3874009a6a1fcd12914c156` |
| `levin-phase-floor-06k-census-route3-20261104T225339` | `done`, 18,560/18,560 | 149 | 3,570,599,114 | `89afd6da13fe7549d7bd714cf0109e17194ac8224b5c4524c04ef16729d81d0d` |

Their `manifest.json`, `status.json`, `stdout.log` and `stderr.log`, and the three small runs'
`checkpoint.jsonl`, were copied to the laptop's `var/runs/` and checked by SHA-256 (31 files, all
equal).

**Added past the laptop's `*.csv` exclusion:** the three tables `out/partL_leaves.csv`,
`out/partL_route2lit_withdrawals.csv` and `out/partC/h1_calls.csv`. Python's `csv` wrote them with
CRLF line endings, and git stores them with LF (`core.autocrlf=input`, as for 06j's tables). So the
committed bytes differ from the files whose SHA-256 the command transcripts print. Those originals
are still on the Mac Studio, and the laptop's working copies are byte-identical to them.

**Read, not written, by 06k:**
- 06g's census checkpoint
  `var/runs/levin-phase-floor-06g-census-20261101T114315/checkpoint.jsonl` (`03adcbba…`) and its
  16 region pickles;
- 06h's census checkpoint (`3104c18e…`);
- 06g's `census/out/v8_calls.json` (`1e9a62b0…`) and `v8_slow_leaves.json`;
- 06's scratch copy of A3 v2, `var/levin-phase-floor-06/wider/state/a3v2_copy/` (fingerprint
  `0e32f24a…`).

Each was unchanged by SHA-256 before and after (`out/cmd/110`–`115b`, `review/checks/b3/`), and
each is still kept until the user says otherwise (`06g/KEPT_ON_MAC_STUDIO.md`,
`06/KEPT_ON_MAC_STUDIO.md`).

**Checked from the laptop on 2026-11-05:**
- The 1,154 files copied back equal the Mac Studio's by SHA-256.
- Every verdict was re-scored from the copied tables by the laptop orchestrator's own reader:
  - L1 and L2 by recomputing each route's decision on all 2,046 Part L leaves from the leaf's
    numbers (no disagreement with the probe's);
  - H1 from `out/partC/h1_calls.csv`, with 06g's ratio checked against 06g's `scores.json` and
    06j's `w_calls.csv` on all 124 references (equal);
  - C1, C2, S1 and K1 from `out/partC/<route>.json`.

  Every verdict and count is the same.

**The worktree** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06k-8937060` (detached at
`89370609`, clean with `--ignored`) was removed on 2026-11-05 with `git worktree remove`, on the
user's word (*"Clean up the Mac Studio: worktrees yes."*).
