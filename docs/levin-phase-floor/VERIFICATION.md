# levin-phase-floor — verification

Each prompt of [`prompts/levin-phase-floor`](../../prompts/levin-phase-floor/README.md) writes its
own section here and nothing else (campaign README §4.1). Its records are in `NN/` beside this
file. **The records are additive:** a later re-run adds a subsection, and does not rewrite one.

**§1 was corrected on committing** by the laptop orchestrator, on the user's decision of
2026-10-25 (review check 9). The script's SHA-256 is now the final run's. The paths are links
relative to this file. The text is otherwise the agent's.

## 1. The phase-group census

**Prompt 01, run 2026-10-25 on the Mac Studio** at `757ef01` (`handover-remedial`, clean), with
no Ray, no lock and no store write. It read A3 v2 (`var/datastores/handover-A3-v2-lambdacdm.sqlite`
and four shards, SQLite `mode=ro`), fingerprint
`0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`.

- **Script:** [`scripts/group_census.py`](01/scripts/group_census.py), SHA-256 `aac0b4ed…` in the header of every final output. *(Corrected on committing: the agent's text gave `068ce023…`, the header of the superseded run 2 in `state/superseded_run2/`; review check 9.)* It imports
  $\eta(z)$ and $\varepsilon$ from [`docs/handover/a3-v2/findings/depth20_floor.py`](../handover/a3-v2/findings/depth20_floor.py) (SHA-256
  `c82f72b0…`) and does not copy them.
- **Outputs:**
  - [`out/census_01.txt`](01/out/census_01.txt), which holds T1–T5 and (d);
  - `out/groups_01.csv`, with 37,325 group records, 12.9 MB. It is **not committed**: this checkout's `.git/info/exclude` excludes every `*.csv`. It stays at `var/levin-phase-floor-01/out/` on the Mac Studio;
  - [`out/candidates_01.txt`](01/out/candidates_01.txt);
  - [`out/recon_check_01.txt`](01/out/recon_check_01.txt) and [`out/recon_diff_01.txt`](01/out/recon_diff_01.txt), for step 5.
- **The store** was byte-identical before and after ([`before`](01/state/store_files_before.txt), [`after`](01/state/store_files_after.txt)).
  `store fingerprint`, run without `--write`, matches the sidecar ([`out/fingerprint_01.txt`](01/out/fingerprint_01.txt)).

### 1.1 Population and records

**Rows.** There are 9280 `QuadSourceIntegral` rows at `(1e-32, 1e-7)`, 2320 per shard, and none at
any other pair. Every row's `metadata` parses and has a `WKB_Levin` key.
- 6791 rows have at least one Levin sub-interval, and 2489 have none.
- 2265 rows are at depth 20, and every one of them has group records.

**Geometry.**
- `r == k` with $q<r$: 4060 rows.
- `q == r == k`: 1160.
- Other: 4060.
- There are **no** `q == k` rows with $r \ne k$, and no `r == k` rows with $q > r$.

**Records.**
- **Labels.** `G` 6368, `G±r` 2965 each, the four `G±q±r` 6053 each, `q±r` 392 each, and `r` 31.
- **Records without $G$.** There are 815: the `qr` and `r` regimes, in the seam band. They are in
  neither class, and none sits on a depth-20 row.

**The $\eta$ check.** Against the stored `eta_response`, `depth20_floor.py`'s $\eta(z_{\rm response})$
is 1.0009–1.0012 in the low band and 1.0012–1.0015 in the mid band. That is findings §2.1's
0.09–0.13 %.
- **Above $z \approx 10^{12}$ the ratio departs from 1.** `depth20_floor.py` integrates only up to
  $z = 10^{15}$, so above that its $\eta$ is negative ([`out/eta_check_01.txt`](01/out/eta_check_01.txt)).
- **No group record is affected.** Those rows have no Levin sub-interval, and no record in
  `groups_01.csv` has $F \le 0$.

### 1.2 T1 — cost by (regime, label)

The depth-20 rows hold 13,115 records, 285.65 CPU-hours and 145,104,757 regions. (All rows: 285.96
CPU-hours.)

| regime | label | CPU-hr | share | regions | share | records |
|---|---|---|---|---|---|---|
| Gr | G+r | 149.835 | 52.5 % | 71,933,987 | 49.6 % | 903 |
| Gr | G−r | 94.978 | 33.2 % | 45,337,612 | 31.2 % | 903 |
| G | G | 18.329 | 6.4 % | 20,808,506 | 14.3 % | 2265 |
| Gqr | G+q+r | 10.370 | 3.6 % | 3,226,384 | 2.2 % | 2261 |
| Gqr | G−q+r | 4.765 | 1.7 % | 1,490,398 | 1.0 % | 2261 |
| Gqr | G−q−r | 3.945 | 1.4 % | 1,235,658 | 0.9 % | 2261 |
| Gqr | G+q−r | 3.430 | 1.2 % | 1,072,212 | 0.7 % | 2261 |

The all-rows column is in `census_01.txt`. It differs from this one by 0.31 CPU-hours.

### 1.3 T2 — the most expensive group of each depth-20 row

| geometry | regime | label | rows | cancelling |
|---|---|---|---|---|
| other | Gqr | G+q+r | 781 | no |
| r==k, q<r | Gr | G+r | 665 | yes |
| q==r==k | Gqr | G+q+r | 241 | yes |
| r==k, q<r | Gr | G−r | 163 | no |
| other | G | G | 116 | no |
| other | Gqr | G−q−r | 93 | no |
| r==k, q<r | Gqr | G+q+r | 50 | yes |
| q==r==k | G | G | 47 | no |
| other | Gqr | G−q+r / G+q−r | 32 / 30 | no |
| q==r==k | Gqr | G−q−r | 21 | no |
| r==k, q<r | G | G | 14 | no |
| r==k, q<r | Gqr | G−q+r / G+q−r / G−q−r | 7 / 3 / 1 | yes / no / no |
| q==r==k | Gqr | G+q−r | 1 | yes |

- **By class:** the top group is non-cancelling in 1301 rows and cancelling in 964. By row count
  the cheap rows dominate, which is why `other`/`G+q+r` leads.
- **The top group's share** of its row's Levin `elapsed`:
  - median 0.485, with 10th and 90th percentiles 0.293 and 0.786;
  - 0.548 on rows over 60 s, and 0.541 on rows over 100 s.

### 1.4 T3 — depth-20 cost by band and by $k$

**By band:**
- **low:** 726 rows, 249.03 CPU-hr. `G+r` 52.1 %, `G−r` 33.2 %, `G` 6.7 %, `Gqr` 7.9 %.
- **mid:** 945 rows, 36.53 CPU-hr. `G+r` 54.8 %, `G−r` 33.4 %, `G` 4.4 %, `Gqr` 7.4 %.
- **seam:** 594 rows, 0.086 CPU-hr.

**By $k$** (CPU-hr; the share of `G+r`, `G−r` and `G`):

| $k$ (/Mpc) | rows | CPU-hr | G+r | G−r | G |
|---|---|---|---|---|---|
| 1.00e5 | 379 | 0.124 | — | — | 56.6 % |
| 3.14e5 | 396 | 0.639 | 9.6 % | 6.2 % | 33.8 % |
| 9.85e5 | 465 | 8.087 | 24.5 % | 19.9 % | 23.2 % |
| 3.09e6 | 340 | 43.280 | 36.4 % | 34.7 % | 9.7 % |
| 9.70e6 | 300 | 110.307 | 47.7 % | 37.1 % | 7.4 % |
| 3.05e7 | 225 | 107.646 | 62.1 % | 32.1 % | 3.4 % |
| 9.56e7 | 158 | 15.566 | 80.8 % | 18.0 % | 0.6 % |
| 3.00e8 | 2 | 0.002 | 99.3 % | 0.1 % | 0.4 % |

**Descriptive only:** `G+r`'s share of the cost rises with $k$, and `G−r`'s does not.

### 1.5 T4 — the P1–P3 statistics

- **P1.** $S$ = **0.4385**: 125.27 of 285.65 depth-20 CPU-hours are on non-cancelling records.
  - No depth-20 record lacks $G$.
  - Within the low band $S$ = 0.444, and within the mid band 0.399.
- **P2.** The depth-20 records in the mid and low bands with $e_g$ finite and non-zero. The fit is
  $\log_{10} e_g$ on $\log_{10} F$, over $\log_{10} F \in [-9.9, -3.2]$:

  | class | n | slope | median $e_g/F$ | 10th | 90th |
  |---|---|---|---|---|---|
  | cancelling | 2946 | 0.968 | 2.117 | 0.257 | 13.52 |
  | non-cancelling | 6943 | 0.822 | 1.489 | 0.148 | 20.42 |

  - The ratio of the two medians is 1.42.
  - Ten records were excluded for a zero or non-finite $e_g$: 6 cancelling and 4 non-cancelling.
  - **Descriptive, by band** (slope / median $e_g/F$):
    - cancelling: low 1.289 / 1.87, mid 0.891 / 2.19;
    - non-cancelling: low 1.237 / 0.70, mid 0.760 / 2.61.
  - **Descriptive, by label:**
    - the lone `G`: slope 0.649, median 0.34;
    - `G−q+r` (non-cancelling): slope 0.447, median 0.72;
    - `G+r`: slope 0.715, median 1.25;
    - `G−r`: slope 0.656, median 1.91.
- **P3.** The share of `elapsed` on records with $F > 10^{-7}$:
  - non-cancelling: **0.9991** (125.16 of 125.27 CPU-hr);
  - cancelling: **0.9997** (160.34 of 160.38).

### 1.6 T5 — where in source time the top group's sub-interval sits

These are the quartiles of $\log(1+z)$ for the top group's sub-interval.
- **`Gr` (`G±r`).** The sub-intervals sit at $\log(1+z_{\max}) \approx 24.7$–$28.1$ and
  $\log(1+z_{\min}) \approx 21.3$–$24.7$, i.e. $z' \sim 10^{9}$–$10^{12}$, deep in radiation
  domination. That holds in every band, although $z_{\rm response}$ is below 9 in the low band.
- **The lone `G`.** Its sub-intervals are at $\log(1+z) \approx 23.6$–$37.3$, before T's hand-over.
- **`Gqr`.** Its sub-intervals reach down to $z_{\rm response}$: in the low band
  $\log(1+z_{\min})$ has median 0.3–0.8, and in the mid band 3.8–6.8.

The full table is in `census_01.txt`. It is descriptive and scores nothing.

### 1.7 Findings §2.5, confirmed on the store

**Item 1, recomputed on the store and reproduced exactly:**
- 1086 depth-20 rows are flagged `total_converged = 1`;
- 257 of them have `total_abserr/|total|` > 1e-6, 197 > 1e-5 and 132 > 1e-4;
- the largest is **5.03e-2**;
- 1179 depth-20 rows are unconverged, and none has a zero `total`.

**At group level.** These are the records flagged `converged` with $e_g > 10^{-6}$ and with
$e_g > 10^{-4}$:
- **depth-20 rows:** 8408 converged of 13,115 records. 2541 exceed 1e-6 and 1004 exceed 1e-4. The
  largest finite $e_g$ is 0.95. By class:
  - cancelling: 827 and 380;
  - non-cancelling: 1714 and 624.
- **all rows:** 30,580 converged of 37,325. 15,560 exceed 1e-6 and 7635 exceed 1e-4. The largest
  is 37.7.

**Item 2: non-finite `total_abserr`.** This **reaches far more than the five rows findings §2.5
counted.**
- **558 rows** store `total_abserr` = +inf: **5 at depth 20** and **553 at depth 1–19**.
  - By band: 392 in the seam, 44 in the mid band and 122 in the low band.
  - All 558 are `total_converged = 0` and `total_phase_limited = 0`.
- **How SQLite returns them.** `typeof` is `real` and `cast(... as text)` gives `Inf`. Python reads
  `inf`. The `metadata` JSON holds the literal `Infinity`. `numeric_quad.abserr` is finite on all
  558, and `WKB_Levin.abserr` is `inf`.
- **The five depth-20 rows** are those of findings §2.5, all with $q = r = 9.56\times10^7$ and
  under 2 s each: (0, 14201), (1, 2368), (1, 14572), (1, 14578) and (3, 34990).
- **The groups that carry it.** 803 group records have `abserr = inf`. Each has a finite non-zero
  `value`, `converged = False` and `phase_limited = False`. 631 of them have `regions = 1`.
  - `qr`/`q−r`: 392, every one in the seam band. On $q = r$ the $q-r$ phase is identically zero.
  - `Gqr`: `G−q+r` 149, `G+q−r` 130, `G+q+r` 89 and `G−q−r` 43.
  - 708 of the 803 are on $q = r$ rows. Of all 6822 `G+q−r`/`G−q+r` records on $q = r$ rows, 234
    are inf.
- Every row is listed one by one, with its sub-interval and groups, in `census_01.txt` (d).

### 1.8 The reconnaissance, re-derived

[`scripts/recon_check.py`](01/scripts/recon_check.py) re-derives [`measure/out/recon_groups.txt`](../../prompts/levin-phase-floor/measure/out/recon_groups.txt) from `groups_01.csv`, on the
same population: depth-20 rows over 100 s.
- **It reproduces the reconnaissance line for line**, apart from the header ([`out/recon_diff_01.txt`](01/out/recon_diff_01.txt)
  is empty):
  - 629 rows and 282.93 CPU-hr;
  - per-group `elapsed` of 282.88 CPU-hr;
  - `G+r` 52.7 %, `G−r` 33.5 %, `G` 6.3 % and `Gqr` 7.6 %;
  - the same top-group table and band table.
- **README §0.2's prose disagrees with that output on one point.** It says *"229.7 of the 282.9
  CPU-hours"* are in the low band. The recon output's own low-band lines sum to **247.88**.

### 1.9 Verdicts

| | H-cancel | H-anchor |
|---|---|---|
| **P1** ($S$ = 0.4385) | **fails** ($S \ge 0.25$) | **holds**: no prediction on $S$, and it holds because P2 holds |
| **P2** (slopes 0.968 / 0.822; medians 2.117 / 1.489; ratio 1.42; n = 2946 / 6943) | **fails**: the non-cancelling class meets the conditions too | **holds** |
| **P3** (shares 0.9991 / 0.9997) | descriptive | **holds** |

`log.md` "Predictions and verdicts" gives each rule's words. It also gives the caveats on how
strongly P3 and P2 discriminate, and the code reading on how far `G+r` actually cancels.

### 1.10 Candidates for 03

[`out/candidates_01.txt`](01/out/candidates_01.txt) lists at least two rows per class, each with its per-group records:

| class | rows listed | picked from |
|---|---|---|
| `G+r` top on `r == k` | 8 | 67 mid-band rows of 100–800 s |
| `G−r` top on `r == k` | 8 | 14 such rows |
| lone `G` top | 3 | all 3 such rows |
| a `Gqr` group top | 8 | 15 such rows |

The `G+r`, `G−r` and `Gqr` lists were thinned to spread over $k$. Every class had mid-band rows, so
no class needed the low-band fallback.
