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

## 2. Phase-noise injection — stopped at the probe

**Prompt 02, run 2026-10-25 on the laptop** (mps026088) from dispatch SHA `3f75bbf`, with no Ray,
no lock and no store. **The sweep was never run.** The prompt stopped at its step 3, and the user
decided the same day that it is superseded by prompt 03.

- **Script:** [`noise_injection.py`](noise_injection.py), SHA-256
  `6ddaaeca85976b2d5f58af12d03ed703d02caa91784fd976d568da8f042656a7`. Every probe record carries
  this hash. The script was uncommitted when the probe ran, so every record says `git_dirty: true`
  at HEAD `3f28791`, the HEAD at launch; another session had committed it while 02 worked.
- **The run:** `levin-phase-floor-02-noise-probe-20261025T183809`, registered and detached.
  - It began at 18:38:09Z.
  - It ended **`killed`, exit 143**, at 19:47:59Z, after 16 of its 19 units, stopped by SIGTERM
    on the user's decision.
  - Its records are copied to [`02/state/`](02/state/) (`checkpoint.jsonl`, `manifest.json`,
    `status.json`, `stdout.log`, `stderr.log`). The run directory stays in `var/runs/`.
- **The report:** [`02/out/probe_02.txt`](02/out/probe_02.txt), written by `--probe-report` from
  the copied ledger alone (SHA-256 `37e01a26…`).
- **Load:** the one-minute load average was 27.8–30.3 during the δ = 0 cells, on 10 cores.

### 2.1 Step 2: the wrapper at δ = 0 is bit-identical to no wrapper

Each of the eight cases was run twice at δ = 0:
- once **unwrapped**;
- once **wrapped**: through the noising wrapper on `QSI.build_phase_groups`, white noise at
  δ = 0, and the pass-through observer on `QSI.adaptive_levin_sincos`.

The cases are {`exact`, `realistic`} × {`together`, `q-smooth`} × $x_{\rm resp}$ ∈ {980, 1e4},
with b = 0 and the seam closed, at atol 1e-45 and rtol `DEFAULT_QUADRATURE_RTOL`.

**On all eight cases the two runs agree exactly**, by `repr` of every float, in:
- `total`, `total_abserr`, `numeric_quad`, `WKB_Levin` and `analytic_rad`;
- `max_depth`, `num_regions`, `num_simple_regions` and `evaluations`;
- `total_converged` and `total_phase_limited`;
- every per-group record of every Levin sub-interval except its wall-clock `elapsed`. That is 48
  group records over 22 sub-intervals.

Two checks made this, and they agree:
- the script's own comparison, in `probe_02.txt`;
- a field-by-field re-check read from the checkpoint while writing this section.

**No δ = 0 cell reached the cap.** `max_depth` was 0–2 in the exact flavour and 6–8 in the
realistic one, so the §5 stop on the baseline did not fire.

**The δ = 1e-4 check never ran.** It was the probe's last unit, so the evidence that the injection
changes `total` was never taken.

### 2.2 Step 3: the timing probe, and why the sweep is infeasible

The first noised cell was `exact`/`together`/980 with white noise at δ = 1e-5. Its δ = 0 cell
takes 0.4 s, with 7 regions.
- It had run **about 69 minutes**, from the last recorded unit at 18:38:39Z to the SIGTERM at
  19:47:59Z, with CPU about equal to wall, without finishing.
- An earlier scratch timing of the same cell had also not finished in 540 s.
- The realistic timing cell and the δ = 1e-4 cell never started.
- There is therefore no projection in `probe_02.txt`.

**The user's reading, on stopping it.** The noise is width-independent and larger than `rtol`, so
no region can pass the driver's per-region relative test. Every Levin call therefore bisects the
full depth-20 tree. On the order of a hundred noised cells, each at that cost, put the sweep at
weeks to months on either machine.

### 2.3 Superseded by 03

The user decided on 2026-10-25 that 02's injection sweep is not run in any form. Prompt 03 (§3)
measured the mechanism directly on seven A3 v2 rows:
- it found the jitter in every scored group, cancelling or not;
- it found a discrepancy that is flat with depth;
- it found the discrepancy at round-off once the jitter is removed.

The prompt's predictions now stand as follows:
- **N1 and N3** are answered by 03 on real rows.
- **N4** moves to prompt 04, whose jitter-removed re-run gives a reference `total` per row.
- **N2's original form** is not tested. The user judged that this does not matter.

## 3. Seven rows, instrumented

**Prompt 03, run 2026-10-25 on the Mac Studio** at `98e68d3` (`handover-remedial`, clean), with no store write.

**Route (a).** A3 v2 was read through a real `datastorekit` `ShardedPool(read_only=True)` whose actors were stood in by `StoreFixtures.standin_pool`.
- `ray.put`, `compute_QuadSource_integral.remote` and the stand-in's serialisation were patched to `ray.cloudpickle` equivalents inside the harness.
- Ray was never initialised, and no lock was taken.
- The route was first proven on a `real_store_fixtures` store ([`state/fixture_proof_final.txt`](03/state/fixture_proof_final.txt)).

**The run.** The seven rows were re-run in-process by the registered, detached run `levin-phase-floor-03-instrument-20261025T194950`, which ended `done`, 7/7, exit 0. Its pickles stay in that run directory on the Mac Studio. The store was fingerprint `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56` before and after.

- **Scripts** (all in [`scripts/`](03/scripts/)):
  - [`rebuild.py`](03/scripts/rebuild.py): step 4, `main.py`'s lookups by the same keys and tags;
  - [`run_instrument.py`](03/scripts/run_instrument.py): the wrappers, M0 and the pickles;
  - [`harness.py`](03/scripts/harness.py): route (a), and the pickler;
  - [`launch.py`](03/scripts/launch.py): the registered launch;
  - [`analyse.py`](03/scripts/analyse.py): step 7.

  Each output's header records the SHA-256 of the script that wrote it.
- **Outputs:**
  - [`out/m0_03.txt`](03/out/m0_03.txt): M0, field by field;
  - [`out/instrument_03.txt`](03/out/instrument_03.txt): every group, then M1–M5;
  - [`out/fingerprint_03.txt`](03/out/fingerprint_03.txt).
- **The store** was byte-identical at step 2, at launch, after the run and after the fingerprint: [`before`](03/state/store_files_before.txt), [`after`](03/state/store_files_after.txt). `store fingerprint`, run without `--write`, matches the sidecar.

### 3.1 M0 — the re-runs are production

On all seven rows, everything step 5 lists equals the stored row bit for bit, under `==` and by IEEE-754 bit pattern:
- `total`, `total_abserr`, `total_converged`;
- `WKB_Levin`, `numeric_quad`;
- `WKB_Levin_data.max_depth` and `num_regions`;
- every field of every group record, and of every sub-interval record, in `metadata["WKB_Levin"]["subintervals"]`.

`analytic_rad` is equal too, on every row. The one field not gated is each group record's `elapsed`, a wall-clock time (all 45 differ; the re-run took 0.76–0.96 of the stored time, median 0.83).

The inputs were rebuilt as `main.py` builds them:
- the grid tags came out as `SourceRedshiftGrid_1740_8d1d43b6` and `ResponseRedshiftGrid_145_c51d43ac`;
- every rebuilt z equals the stored z;
- the six parents equal the stored row's;
- the existence query returns each row by its serial.

The wrappers on `adaptive_levin_sincos` and `build_phase_groups` are transparent. Each Levin call was matched to its `PhaseGroup` by the label parsed from `notify_label` and confirmed by object identity.

**Fidelity of the analysis.** From the pickles alone, the driver's own single-region routine reproduces the recorded `abserr` and type of all 580 sampled depth-20 regions exactly.

### 3.2 What the bisection looks like

**Every depth-20 region is a Clenshaw–Curtis fallback region**: 189,196 of 189,196 in the 29 scored groups (at least 10 regions at depth ≥ 18). None of the 45 groups has a Levin region at depth 20.

| row | top group | F | scored groups | median σ_θ/F | median D/σ_θ | slope (real) | slope (control) | D₂₀ real/control |
|---|---|---|---|---|---|---|---|---|
| R1 | `G+r` | 1.69e-6 | 5 | 0.28–0.29 | 0.112 | 0.059 | 0.068 | 3.1e6 |
| R2 | `G+r` | 4.60e-6 | 3 | 0.31–0.32 | 0.093 | 0.067 | −0.020 | 8.8e6 |
| R3 | `G+r` | 9.19e-5 | 2 | 0.26–0.29 | 0.078 | 0.089 | 0.069 | 2.4e8 |
| R4 | `G−r` | 2.60e-6 | 5 | 0.25–0.26 | 0.099 | 0.054 | 0.148 | 5.8e6 |
| R5 | `G−r` | 6.49e-6 | 4 | 0.22–0.23 | 0.100 | 0.051 | −0.002 | 1.1e7 |
| R6 | lone `G` | 1.96e-6 | 5 | 0.34 | 0.113 | 0.027 | 0.023 | 2.0e7 |
| R7 | `G+q+r` | 2.60e-6 | 5 | 0.26 | 0.106 | 0.049 | −0.017 | 1.0e7 |

How to read the table:
- σ_θ/F is the range over the row's scored groups. D/σ_θ, the slopes and the ratio are medians over the row's scored groups.
- The slopes are of ln D on ln h over depths 12–20, from 20 sampled depth-20 regions per group.
- In the control, `theta_mod_2pi` is replaced on each depth-10 span by a degree-12 polynomial fit to it, and nothing else changes.

**The composed phase carries non-smooth jitter at about 0.2–0.35 F**, in every scored group:
- cancelling or not: the class medians are 0.270 (cancelling, 18 groups) against 0.262 (non-cancelling, 11), a ratio of 1.03;
- `G±r`, the lone `G`, and all four `Gqr` signs alike.

**The discrepancy that drives the bisection does not fall with refinement.** D is flat from depth 10 to 20 (slopes −0.005 to 0.17), at about 0.1 σ_θ.

**Remove the jitter and it is gone.** The depth-20 discrepancy falls by 3e6 to 2e8, to the round-off of the nested pair (D ≈ 1e-16 to 1e-13). In the control it is already at that level at depth 10, so it cannot fall further with h.

**The driver cannot see the floor.** `phase_err` reaches `abserr` in 0.85 % of the depth-20 regions (pooled; 0.26–2.1 % by row).

**How far `G+r` cancels.** On R1–R5, |θ'_{G+r}|/|θ'_{G−r}| is 0.26795–0.26885 at five abscissae of the `Gr` sub-interval. The prediction (1−c_s)/(1+c_s) is 0.267949, with c_s² = 0.333333. That is agreement within 0.34 %, so `G+r` oscillates at about 0.27 of `G−r`'s rate and is not stationary. R6 and R7 have no `Gr` sub-interval.

### 3.3 Verdicts

| | Number | Verdict |
|---|---|---|
| **M0** | 7/7 rows bit-identical | **holds** |
| **M1** | 29/29 scored groups in [0.1, 10]; 0.224–0.342 | **holds** |
| **M2** | both parts on 7/7 rows (slopes 0.027–0.089; D/σ_θ 0.078–0.113) | **holds** |
| **M3** | control slope ≥ 2 on 0/7 rows (−0.020 to 0.148); ratio ≥ 100 on 7/7 rows (3.1e6 to 2.4e8) | **fails**, on the slope clause; the clause is **badly posed**, recorded as such and not re-posed (03's `issues.md` I1) |
| **M4** | 1611/189,196 = 0.0085 < 0.1; Clenshaw–Curtis share 1.000 | **holds** |
| **M5** | R1–R5 within 0.34 %; inapplicable on R7 (no `Gr`) | **holds** on R1–R5 |

**The mechanism**, "a non-smooth phase floor, invisible to the driver, drives the bisection to the cap", holds if M1–M4 hold.
- **As written, it does not hold, because M3 fails.** The prompt's clause "M2 holds and M3 fails ⇒ refuted" fires by its letter.
- **What M3 measured points the other way.** Removing the jitter, and nothing else, removes the discrepancy by six to eight orders of magnitude. The slope clause cannot be met by a discrepancy that is already at round-off from depth 10.
- That reading is the user's to make at G1. 03's `log.md` note (a) gives it in full.

**H-anchor holds.** M1 and M2 hold in `G−r` on R4 and R5, in the lone `G` on R6, and in `G+r`; indeed in every scored group.

**H-cancel's condition is met** in the cancelling groups. But the non-cancelling groups carry the same floor at the same level and the same depth, so **it does not account for them**. As in §1, it fails as the whole account.

**For §2.7 layer 2.** The regions at the cap are all in the Clenshaw–Curtis branch (03's `issues.md` I2), and that branch's acceptance test is the one a declared phase floor must reach.

## 4. The floor removed

**Prompt 04, run 2026-10-25 on the Mac Studio.** The code ran from a git worktree at `98e68d3`, the tree 03 ran at. The records are in the main checkout at `c8ca155` (`handover-remedial`, clean). There was no store write, no Ray and no lock (route (a), as in §3).

**The run.** `levin-phase-floor-04-remove-20261025T231110` ended `done`, 28/28 units, exit 0. Its checkpoints and pickles stay in that run directory on the Mac Studio.

**The store** was byte-identical at every check: step 2, the stop, the resume, before launch, after the run, and after the fingerprint ([`before`](04/state/store_files_before.txt), [`after`](04/state/store_files_after.txt)). `store fingerprint`, run without `--write`, gives `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56` and `problems: none` ([`out/fingerprint_04.txt`](04/out/fingerprint_04.txt)).

**A stop, and an amendment.** The first dispatch stopped at step 5, where E1 (c) and E2 failed at round-off: 4e-16 to 6.5e-14 rad, against an F of 1.7e-6 to 9.2e-5 ([`out/phases_04_first.txt`](04/out/phases_04_first.txt)).
- The bounds left out two round-off terms.
  - The first is the abscissa's own rounding, z = expm1(x), at about ε|dθ/du|. Predicted alone, it matches X's residual with a correlation of 0.94–1.00 ([`out/diag_abscissa_rounding_04.txt`](04/out/diag_abscissa_rounding_04.txt)).
  - The second is the ulp of arm A's unreduced remainder.
- The user amended the two gates (prompt §8, commit `d170e6d`). The scored predictions were left unchanged, and nothing had been re-run.

- **Scripts** (all in [`scripts/`](04/scripts/)):
  - [`phases.py`](04/scripts/phases.py): the arms P, A and X as one dispatcher at `PrimitivePhase.theta_mod_2pi`, with Δ^ext in mpmath at 512 bits, Φ^arm and F^arm;
  - [`check_phases.py`](04/scripts/check_phases.py): step 5, the gates E1 and E2;
  - [`run_remove.py`](04/scripts/run_remove.py) and [`launch.py`](04/scripts/launch.py): the registered run;
  - [`analyse.py`](04/scripts/analyse.py): step 8, from the checkpoints;
  - [`harness.py`](04/scripts/harness.py), [`rebuild.py`](04/scripts/rebuild.py) and [`prov.py`](04/scripts/prov.py): 03's route (a), with both trees' provenance.

  Each output's header records the SHA-256 of the script that wrote it, and both trees' `HEAD`s.
- **Outputs:**
  - [`out/phases_04.txt`](04/out/phases_04.txt): E1, E2, the reproduction of 03, F^arm per group, A1's coverage, σ_φ and the modulus;
  - [`out/remove_04.txt`](04/out/remove_04.txt): every unit, every group, and the scores.
- **Predictions** were fixed in [`state/predictions.md`](04/state/predictions.md) at 23:10:41Z, before the 23:11:10Z launch.

### 4.1 The gates

| | Number | Verdict |
|---|---|---|
| **E1 (a)** | worst \|float(Δ^ext) − delta()\| / 2 ulp = 1.000, over 80,000 abscissae | holds |
| **E1 (b)** | worst \|wrap(ρ^X − ρ^P)\| / (4εΦ^P + 1e-15) = 0.870 | holds |
| **E1 (c)**, amended | worst \|wrap(ρ^A − ρ^X)\| / (4εΦ^A + 4·2⁻⁴⁹) = 0.379 | holds |
| **E2**, amended | 29/29 scored groups; σ_θ / bound 0.22–0.34 (P), 0.10–0.77 (A), 0.05–0.20 (X); X's σ_θ ≤ 6.5e-14 ≪ 1e-2 F | holds |
| **M0** | P bit-identical to the stored row on 7/7 rows (`analytic_rad` too; only `elapsed` differs) | holds |

- **ρ^X is exact.** It equals the correctly rounded exact remainder at all 80,000 E1 abscissae (checked with `fractions`).
- **P reproduces 03.** It reproduces 03's printed medians of σ_θ and σ_θ/F in all 29 groups, to 03's digits.

### 4.2 What removing the floor does

| row | P regions | X regions (1e-7) | X/P | P wall | X wall | most at depth ≥ 18 in a scored group, X | A = X? |
|---|---|---|---|---|---|---|---|
| R1 | 34,147 | 133 | 0.0039 | 181.8 s | 1.5 s | 0 | yes |
| R2 | 29,782 | 89 | 0.0030 | 196.9 s | 1.2 s | 0 | yes |
| R3 | 33,314 | 66 | 0.0020 | 219.7 s | 0.8 s | 0 | yes |
| R4 | 52,909 | 167 | 0.0032 | 321.6 s | 2.1 s | 2 | yes |
| R5 | 26,273 | 130 | 0.0049 | 181.0 s | 1.9 s | 3 | yes |
| R6 | 22,147 | 89 | 0.0040 | 124.3 s | 1.4 s | 3 | yes |
| R7 | 23,338 | 88 | 0.0038 | 136.5 s | 1.4 s | 3 | yes |

**With the jitter gone, the cap goes.**
- No group under X or A at rtol 1e-7 reaches depth 20.
- The few regions at depth 18–19 are in `Gqr` groups on R4–R7.
- No X or A unit came near its budget: 0.4–3.0 s, against budgets of 373–965 s.

**Layer 1(a) alone does what the exact limit does, on these rows.** A, which takes kτ_R out of G's phase and leaves T's remainders as production forms them, gives the same region count as X on every row, group by group. The totals agree to ≤ 1.2e-12 relative. That includes the `Gqr` groups, whose predicted F^A_g (a maximum over the sub-interval) is 1.1–1.3 F (04's `issues.md` I7).

**At rtol 1e-8**, X reaches depth 20 in a single short chain: 2 regions at depth 20 per affected group, 4–10 per row, on R1 and R4–R7 (04's `issues.md` I6).

**The phase that is left** ([`out/phases_04.txt`](04/out/phases_04.txt), descriptive). These errors are smooth, and P, A and X share them:
- **The modulus error of G** is 0.175 F on every row.
- **G's residual φ is noisy.** In the stored samples it carries noise at 0.15–0.21 F (median), rebuilt in mpmath from the stored div/mod. Production's quantised φ_G reaches 0.46–0.78 F in its worst windows.
- So X is the jitter-free limit, not the truth (04's `issues.md` I2).

### 4.3 Verdicts

| | Rule | Number | Verdict |
|---|---|---|---|
| **R1** | under X at 1e-7, every scored group < 10 regions at depth ≥ 18, and regions ≤ P/10; on ≥ 6 of 7 rows | 7/7 rows; X/P 0.0020–0.0049; at most 3 regions at depth ≥ 18 | **holds** |
| **A1** | under A at 1e-7, every scored group with F^A_g ≤ 1e-8 has < 10 regions at depth ≥ 18 | 14 covered groups (every scored lone `G` and `G±r`); 0 at depth ≥ 18 in all 14 | **holds** |
| **N4** | \|total_P − total_X(1e-8)\| ≤ total_abserr_P on all seven | ratio 0.440, 0.0795, **1.161**, 0.245, 0.445, **1.533**, **1.942** (R1–R7) | **fails** on R3, R6, R7 |
| **R2** | \|total_X(1e-7) − total_X(1e-8)\| ≤ total_abserr_X(1e-7) on all seven | ratio 0.405, 2.1e-4, 0, 0.305, 8.9e-7, 0.474, 2.5e-3 | **holds** |

**Q5's number,** \|total_X(1e-7) − total_X(1e-8)\| / \|total_X(1e-8)\|, beside the 2026-09 sweep's 1.9e-6 (mid band), 2.84e-6 (low) and 8.3e-7 (seam):

| R1 | R2 | R3 | R4 | R5 | R6 | R7 |
|---|---|---|---|---|---|---|
| 5.36e-7 | 3.16e-9 | 0 | 6.73e-6 | 1.41e-11 | 8.21e-8 | 3.04e-10 |

**Findings §2.6 item 3 is answered.** Its refutation clause, "item 3 leaves the depth at 20", does not fire: the depth falls from 20 to at most 19, and the region count to under 0.5 % of P's.

**Its second prediction, that `total` moves by less than the row's old declared error, holds on four rows only.** That prediction is N4's rule.
- It holds on R1, R2, R4 and R5.
- It fails on R3, R6 and R7, where the stored `total_abserr` understates the jitter's effect by 1.16–1.94×. In relative terms that is 2.7e-4 (R3, a cancelling total of 7.0e-29) and 6.6–7.3e-7 (R6, R7).

**Q5 measures self-consistency.** With the floor gone, the cost of `rtol = 1e-7` against `1e-8` is at most 6.7e-6 relative (R4), and below 1e-6 on six rows. It does not bound the shared φ error.

## 5. Layer 1(a)

**Prompt 05, 2026-10-26, in three parts.** The code and the comparison tool were written on the
laptop; R9 and R10 and the wider re-measurement ran on the Mac Studio.

- **The code** is `120fdfe`. $G$'s phase remainder is now formed node by node:
  - for each table node, a constant formed exactly from the node limbs and $k$, reduced against
    $2\pi$ itself at 256 bits, and memoised;
  - plus a small part, formed in double and reduced with `WKB_mod_2pi`.

  No double of size $k\tau_R$ is formed on the way. $T$'s remainder, `raw_theta` and `theta_deriv`
  are unchanged. The tests and the four breakages are in `prompts/levin-phase-floor/logs/05-reanchor-g-phase.md`.
  On the test fixture, σ_θ falls from 0.297 $F$ to 8.8e-10 $F$.
- **The comparison tool** is `tools/store_value_diff.py`, also in `120fdfe`. It compares two closed
  stores' values, matched by physical identity, never by serial (`CLAUDE.md`, campaign conventions
  rule 7).
- **Its first use** is R9 (the parent) against R10 (the code), in §5.1.
  - Every value table of every class is bit-identical.
  - Outside `QuadSourceIntegral`, row columns differ only in timings, timestamped labels, the
    serial-derived `GkSource.parents_digest`, and GkWKB work counters in 25 rows. The user ruled
    that the prompt was wrongly worded in counting those (`[05-store-value-diff-compares-volatile-columns]`).
  - In `QuadSourceIntegral`, 542 of 864 totals move, by a median 8.7e-5 of the declared error. One
    row moves by 28×.
- **The wider re-measurement** is §5.2, on 39 A3 v2 rows. On the 11 depth-20 rows still capped
  under the parent, the change removes the cap on every one: 644 s → 4.4 s. L1, L1q and L3 fail as
  written, but only on rows that `handover` B1 had already uncapped, and on R5's short chain. L2
  holds. B1 by itself moves 36 of the 39 stored totals beyond their stored error
  (`[05-b1-moves-stored-a3v2-totals-beyond-their-error]`).

**Records** are in [`05/`](05/):
- `rehearsal/` and `wider/`, each holding `out/`, `scripts/`, `state/`, the agent's `log.md`,
  `verification.md` as first written, and `issues.md`;
- the Mac Studio orchestrator's [`review.md`](05/review.md) and its `review/` outputs and scratch
  scripts.

The scripts are byte-identical to the runs and are left unformatted (`CLAUDE.md`).

**Set in on committing** by the laptop orchestrator, from the two agents' `verification.md`:
- the subsections are numbered 5.1 and 5.2;
- three links to files kept on the Mac Studio only are made plain text.

The text is otherwise the agents'.

**The laptop's re-checks** (orchestrator note §B.4), read-only over SSH:
- the 28× row reads the tabulated totals bit for bit in both stores;
- on `low-G-r-2`, `QuadSource` recomputed at the parent equals A3 v2's stored values at all 666
  nodes and 9 columns, and adds B1's 3 nodes;
- A3 v2 is unchanged.

### 5.1 Rehearsals R9 and R10, and the value comparison (prompt §5)

**Prompt 05 §5, run 2026-10-26, 01:46–02:10 UTC, on the Mac Studio (mps026298, arm64, Darwin
27.0.0).** The records were written in the main checkout at `53084dc` (`handover-remedial`, clean
throughout). Its code is the code commit `120fdfe`'s. Python 3.13.16 from the main checkout's
venv: `ray` 2.59.0, `SQLAlchemy` 2.0.55, `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit`
0.2.0. The load average was about 5–7 on 10 cores.

**Records** are in [`05/rehearsal/`](05/rehearsal/): `out/`, `scripts/` and `state/`, the agent's
`log.md`, this section as first written, and `issues.md`. Every command is in
[`out/commands.log`](05/rehearsal/out/commands.log). The scripts' SHA-256 are in
[`out/script_sha256.txt`](05/rehearsal/out/script_sha256.txt). The readers' `extract/` was not
copied back.

#### The two trees

| | R9 (before) | R10 (after) |
|---|---|---|
| tree | worktree `SecondaryGWKit-lpf05-e648951`, detached at `e648951` (the code commit's parent) | main checkout at `53084dc` (code = `120fdfe`) |
| how it ran | cwd and `PYTHONPATH` the worktree; the main checkout's venv python by absolute path; `PYTHONDONTWRITEBYTECODE=1` | R8's launch as written: `PYTHONPATH=.`, `./venv/bin/python` |
| run | `levin-phase-floor-05-r9-parent-20261026T014729` | `levin-phase-floor-05-r10-reanchor-20261026T015120` |
| `sh` / driver pid | 47763 / 47770 | 49749 / 49752 |
| wall | 01:47:29 → 01:50:23 | 01:51:20 → 01:54:10 |
| manifest | `git_head` `e6489519…`, `git_dirty` false, cwd the worktree | `git_head` `53084dcb…`, `git_dirty` false, cwd the main checkout |
| launcher SHA-256 | `5b984c75…` (R8's) | `5b984c75…` |
| end | `done`, exit 0, no Traceback, `stderr.log` empty | `done`, exit 0, no Traceback, `stderr.log` empty |
| store | `var/datastores/levin-phase-floor-05-r9.sqlite` + 4 shards | `var/datastores/levin-phase-floor-05-r10.sqlite` + 4 shards |

Both builds were R8's rehearsal scope: 3 k over 1e5–1e6/Mpc, zend 250, atol 1e-32, rtol 1e-7, 6
CPUs, the same `--job-name levin-phase-floor-05`, and `--campaign levin-phase-floor --prompt 05`.

At R9's launch, `lsof` showed the cwd of the driver and of a Ray `Datastore` actor to be the
worktree ([`out/r9_launch_check.txt`](05/rehearsal/out/r9_launch_check.txt)). After both runs, the
worktree's `git status --short --ignored` was empty, and `find -newer` listed nothing in it.

#### The lock

[`out/lock.log`](05/rehearsal/out/lock.log). Each command was a shell command of its own, output by `>>`.

| # | command | output | exit |
|---|---|---|---|
| 1 | `acquire --project SecondaryGWKit --purpose "levin-phase-floor prompt 05, rehearsals R9 … and R10 …" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 2 | `set-pid … 47770` (R9) | recorded | 0 |
| 3 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after R9) | nothing | 0 |
| 4 | `set-pid … 49752` (R10) | recorded | 0 |
| 5 | `status --ray-temp-dir /tmp/ray-sgk` (while stopped for the question below) | HELD; owner 49752 not alive; no Ray processes | 0 |
| 6 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after the readers) | nothing | 0 |
| 7 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 8 | `status --ray-temp-dir /tmp/ray-sgk` | free; no Ray processes | 0 |

#### The fingerprints

| | digest | problems | per-class |
|---|---|---|---|
| R8 (reference, `handover-12-r8-reference-20261025T210551`) | `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` | none | 21 classes |
| **R9** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 digests equal R8's |
| **R10** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 equal R8's; the whole `classes` object equals R9's |

The fingerprints are the ones the launcher took at the end of each build, read from `status.json`
([`out/r9_end_check.txt`](05/rehearsal/out/r9_end_check.txt),
[`out/r10_end_check.txt`](05/rehearsal/out/r10_end_check.txt)). There is no sidecar: `--build`
writes none.

#### The value comparison (prompt §5.4)

`PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff var/datastores/levin-phase-floor-05-r9.sqlite var/datastores/levin-phase-floor-05-r10.sqlite --json`,
run in the main checkout at `53084dc`. It exited **1** in 16 s. The text form, `--max-rows 1000000`,
also exited 1. Outputs: `out/store_value_diff.json` (41 MB, on the Mac Studio only),
`out/store_value_diff.txt` (10 MB, on the Mac Studio only),
[`out/store_value_diff_summary.txt`](05/rehearsal/out/store_value_diff_summary.txt).

- **Structure.** 21 classes were compared, and `not_compared` is empty. In every class, 0 rows are
  in one store only, 0 rows cannot be matched by identity and 0 columns are unresolvable.
- **Every value table of every class is bit-identical between R9 and R10**, as the tool's JSON
  shows:

  | value table | rows identical | differ / only in one / unmatched |
  |---|---|---|
  | BackgroundModelValue | 1159 | 0 / 0 / 0 |
  | TkNumericValue | 1478 | 0 / 0 / 0 |
  | TkWKBValue | 2209 | 0 / 0 / 0 |
  | GkNumericValue | 35874 | 0 / 0 / 0 |
  | GkWKBValue | 147642 | 0 / 0 / 0 |
  | GkSourceValue | 169653 | 0 / 0 / 0 |
  | QuadSourceValue | 2447 | 0 / 0 / 0 |

- **Row columns outside `QuadSourceIntegral`.** 12 classes are reported value-identical. They are
  `version`, `store_tag`, `redshift`, `wavenumber`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`,
  `IntegrationSolver`, `GkSourcePolicy`, `QuadSourcePolicy`, `GkSourcePolicyData` and
  `OneLoopIntegral`, which has no rows. Eight are reported not value-identical, in their row
  columns only. [`scripts/classify_non_qsi.py`](05/rehearsal/scripts/classify_non_qsi.py) assigns
  every one of their differing columns to one of four kinds, and **0 are left unclassified**
  ([`out/classify_non_qsi.txt`](05/rehearsal/out/classify_non_qsi.txt)):

  | kind | columns | classes (rows) |
  |---|---|---|
  | wall-clock timing | `compute_time`, `stage_1_compute_time`, `{min,mean,max}_RHS_time` | wavenumber_exit_time (3), BackgroundModel (1), TkNumericIntegration (3), TkWKBIntegration (3), GkNumericIntegration (1368), GkWKBIntegration (3474), QuadSource (6) |
  | `label` | `<job>-…-<datetime.now()>`; equal once the trailing timestamp is removed | TkNumeric (3), TkWKB (3), GkNumeric (1368), GkWKB (3477), GkSource (291), QuadSource (6) |
  | `GkSource.parents_digest` | see below | GkSource (291 of 291) |
  | GkWKB work counters | `metadata` keys `rho_evals`, `rho_reused`, and `stage_1_RHS_evaluations` | GkWKBIntegration (25 of 3477) |

  **`parents_digest`.** It is the SHA-256 (hex) of the compact JSON
  (`json.dumps(…, separators=(",", ":"))`) of the GkSource's parent set. That set is the list of
  triples (z_source **serial**, GkNumericIntegration **serial** or None, GkWKBIntegration
  **serial** or None), sorted by z_source serial (`ObjectFactories/GkSource.py`
  `GkSource_parents_digest`; `ComputeTargets/GkSource.py` `GkSource_parent_set`). It is a function
  of row serials, and of nothing else.
  [`scripts/parent_serials.py`](05/rehearsal/scripts/parent_serials.py)
  ([`out/parent_serials.txt`](05/rehearsal/out/parent_serials.txt)) checks this on the stores:
  - the redshift serials are the same in both stores (0 of 1159 differ);
  - the GkNumericIntegration serials differ in 1346 of 1368 rows, and the GkWKBIntegration serials
    in 3466 of 3477. Rows were paired by label without its timestamp;
  - in each store, recomputing the digest from that GkSource's own `GkSource_parents` rows
    reproduces the stored column for 291 of 291 rows.

  The tool matched every GkSource row by identity, so the parent sets agree by identity, and only
  the serials differ.

  **The GkWKB work counters**
  ([`scripts/gkwkb_reuse_rows.py`](05/rehearsal/scripts/gkwkb_reuse_rows.py),
  [`out/gkwkb_reuse_rows.txt`](05/rehearsal/out/gkwkb_reuse_rows.txt)). In each of the 25 rows,
  one store reused the per-process residual table (`rho_reused` true, `rho_evals` 0 or 4) and the
  other built it (`rho_evals` 4632–4636). The swaps go both ways within every $k$: 11 rows were
  reused in R9 only, and 14 in R10 only. No other metadata key differs in those rows. Every
  `GkWKBValue` row, and every other GkWKB value column (`sin_coeff`, `cos_coeff`, `G_init`, …), is
  identical.

- **The user's decision (2026-10-26), relayed:** "Those columns cannot be expected to be identical.
  So I think that's just an incorrectly-worded prompt." The four kinds above are therefore not
  prompt §8's stop. A difference outside `QuadSourceIntegral` of any other kind would still be a
  stop, and there is none.

#### `QuadSourceIntegral`, tabulated per row

[`scripts/tabulate_qsi.py`](05/rehearsal/scripts/tabulate_qsi.py) reads both stores (`mode=ro`) and
the tool's JSON. Its outputs are [`out/qsi_tabulation.txt`](05/rehearsal/out/qsi_tabulation.txt)
and one row per integral in `out/qsi_rows.csv` (on the Mac Studio only). A = R9 (parent),
B = R10 (code).

- Rows were paired by physical (k, q, r, z_response), giving 864 = 864. The pairing has the same
  key set as the tool's 864 identity-matched rows, and gives 0 mismatches in `total`'s `float.hex`
  against the tool.
- All 864 rows differ in some column. Wall-clock columns differ in every row, and `label` in
  every row.

| | rows |
|---|---|
| `total` differs | **542** of 864 (322 bit-identical) |
| `WKB_Levin` differs | 542, exactly the rows where `total` differs |
| `numeric_quad` / `analytic_rad` differ | **0** / **0** |
| `total_abserr` differs | 544 |
| rows with no `WKB_Levin` part (`max_depth` None) | 291, none of which differs in `total` |

Over the 542 rows whose `total` differs:

| statistic | median | 90 % | max |
|---|---|---|---|
| $\lvert\Delta{\rm total}\rvert/\lvert{\rm total}_A\rvert$ | 1.43e-11 | 7.02e-09 | 4.03e-05 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ | 8.71e-05 | 3.62e-02 | 28.2 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_B$ | 8.71e-05 | 3.66e-02 | 29.4 |

- Rows with $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ > 0.1: 22. Above 1: **one**, at
  $(k,q,r,z_{\rm resp}) = (10^6, 3.162\times10^5, 10^6, 329.64)$. There the ratio is 28.2, the
  relative change 4.0e-5, the depth 6 → 7, the regions 36 → 39, and `total_converged` is 0 in both.
- **`total_converged`:** no row changes (0→0: 355, 1→1: 509). **`total_phase_limited`:** no row
  changes (0→0: 833, 1→1: 31).
- **`WKB_Levin_max_depth`:** 15 rows change: 6→7 ×5, 7→6 ×3, 8→7 ×5, 8→6 ×1, 14→12 ×1. No row
  is at depth ≥ 15 in either store.
- **`WKB_Levin_num_regions`:** 42 rows change, with the total going from 13,726 to 13,652. The
  largest falls are on $(10^6, 10^5, 10^6)$: 79 → 55, 67 → 50, 63 → 56 and 60 → 53.
- **By depth under A**, the median $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ rises with
  depth: 3e-5 to 9e-5 at depths 5–7, 4.8e-4 at depth 8, and 1.2e-2 to 6.4e-2 at depths 10–14.
- **Metadata.** The non-wall-clock keys that differ are the `WKB_Levin` `value`/`abserr` tree:
  `WKB_Levin.abserr` in 544 rows, `subintervals[i].value` and `.abserr`, and
  `subintervals[i].groups[j].value` and `.abserr`. In a few rows `groups[j].regions`,
  `.simple_regions` and `.evaluations` also differ. There are 41 distinct keys, in 546 rows. In the
  other 318 rows only wall-clock keys (`…elapsed`, `…compute_time`) differ.

These are recorded, not judged. Prompt §6 scores the effect on the depth-20 rows of A3 v2.

#### The readers (prompt §5.5)

`bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh var/datastores/levin-phase-floor-05-r10.sqlite var/levin-phase-floor-05/rehearsal/extract`,
with the lock held, from 02:06:12 to 02:08:36 UTC ([`out/readers.txt`](05/rehearsal/out/readers.txt)).
Overall exit **0**.

| reader | exit | wall | lookups | files written (R8) | store files after |
|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 26 s | 116 | 450 (531) | identical |
| `extract_GkWKB_data.py` | 0 | 48 s | 348 | 2436 (2436) | identical |
| `extract_GkSource_data.py` | 0 | 19 s | 10 | 142 (160) | identical |
| `extract_TkWKB_data.py` | 0 | 16 s | 3 | 27 (27) | identical |
| `extract_tensor_source_data.py` | 0 | 14 s | 6 | 27 (27) | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 15 s | 24 | 74 (76) | identical |

- The lookups equal R8's.
- File counts differ only where a reader draws an unseeded subsample
  (`[10-the-readers-draw-their-work-lists-unseeded]`).
- After each reader, 0 Ray processes of this venv were left, and no reader log has a Traceback.

**R10's store, before and after the readers**
([`state/r10_store_sha256_before_reads.txt`](05/rehearsal/state/r10_store_sha256_before_reads.txt),
[`state/r10_store_sha256_after_reads.txt`](05/rehearsal/state/r10_store_sha256_after_reads.txt);
`diff` empty; no `-journal`, `-wal` or `-shm` file):

```
557066beae8bf5e5416947f07687cd33771e126c31407fef2016ec826afb50cc  levin-phase-floor-05-r10.sqlite            36864
ee4552a062a0adbd7c192ffc79a0d9e33527df8a6acd6e2405054d891173062e  levin-phase-floor-05-r10-shard0000.sqlite  24076288
83c0e9b27b76d479495a19388f23dc65d864254fd28aa3a8966432666cb7cadb  levin-phase-floor-05-r10-shard0001.sqlite  24289280
5b80e1bda653a68e1c3e5d1b1f877e9f5d0c723f8555cee52892a8b6e9a96689  levin-phase-floor-05-r10-shard0002.sqlite   1077248
9b90fb154fd8051e0b3ffe2cd07fa0a484d436deb496808a400929c171a8f8d6  levin-phase-floor-05-r10-shard0003.sqlite  23543808
```

**Both stores**, all ten files, are byte-identical from before the comparison to the end
([`state/store_sha256_before_diff.txt`](05/rehearsal/state/store_sha256_before_diff.txt),
[`state/stores_sha256_final.txt`](05/rehearsal/state/stores_sha256_final.txt)). R9's equal its
hashes taken just after its build
([`state/r9_store_sha256_after_build.txt`](05/rehearsal/state/r9_store_sha256_after_build.txt)).
R9's primary is `cfba7ac4c218b7c25f0fef9114537b556530b5f71c6b96c6699818cc5fb7d2de`.

#### Against prompt §7.2

- R9 and R10 both end `done` with fingerprint `1b332ffa…`: **yes**.
- Every non-`QuadSourceIntegral` value is identical: **yes, on the user's reading.** Every value
  table is bit-identical. The row-column differences are only wall-clock timings, timestamped
  labels, the serial-derived `parents_digest`, and the GkWKB residual-reuse counters.
- The `QuadSourceIntegral` differences are tabulated: **yes**. 542 of 864 `total`s moved, all
  through `WKB_Levin`, with `numeric_quad` and `analytic_rad` untouched. R9 and R10 are not
  value-identical in `QuadSourceIntegral`, so each tree ran the code it names.
- The readers exit 0: **yes**, and the store is byte-identical.
- The lock is released: **yes**.

### 5.2 The wider re-measurement (prompt 05 §6)

**Prompt 05 §6, run 2026-10-26 on the Mac Studio (mps026298).** There was no Ray, no lock and no
store write (route (a), as in §3 and §4). Two trees took part:
- $p$, the code commit's parent `e648951` (B1's code without 05's change), in the git worktree
  `SecondaryGWKit-lpf05-e648951`;
- $c$, the code commit `120fdfe`, in the main checkout at `53084dc`, which adds only records.

Each worker loaded its packages from its own tree only. The `hello` records in
[`state/L0/ledger.jsonl`](05/wider/state/L0/ledger.jsonl) and the run's `worker_*_hello.json` show
it. Under $c$, every $G$ phase had `exact_remainder=True` and every $T$ phase `False`.

**The run.** `levin-phase-floor-05-wider-20261026T020005` ended `done`, 109/109 units, exit 0, in
12 minutes. Its checkpoints and region pickles stay in that run directory on the Mac Studio.
- Every record carries `run_wider.py`'s SHA-256 (`491d2608…`), and its tree's `HEAD` and empty
  `git status --short`.
- No unit went over budget, and none raised.

**The store** was byte-identical before, before launch, after the run and after the fingerprint
([`before`](05/wider/state/store_files_before.txt),
[`after`](05/wider/state/store_files_after_fingerprint.txt)). No journal, `-wal` or `-shm` file
appeared. `store fingerprint`, run without `--write`, gives
`0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56` and `problems: none`
([`out/fingerprint_05_wider.txt`](05/wider/out/fingerprint_05_wider.txt)).

**The rows** ([`state/sample.md`](05/wider/state/sample.md), fixed before any row ran;
[`out/sample.txt`](05/wider/out/sample.txt)). There are 39:
- 24 depth-20 rows, 3 per cell of {`G+r`, `G−r`, lone `G`, `Gqr`} × {low, mid}, the cheapest at
  distinct $k$ with stored `compute_time` ≤ 900 s;
- 03's R1–R7;
- 8 controls: two per band with depth < 20 and a group containing $G$, and two with `NULL` depth.

The `NULL` controls come from the store (read-only), because the census holds no `NULL` rows. The
rule prefers cheap rows.

**`QuadSource` is recomputed in-process**, as `main.py` computes it at this tree. The read-back is
formed as the factory forms it, with no production change
([`scripts/rebuild.py`](05/wider/scripts/rebuild.py)).

**Scripts** ([`scripts/`](05/wider/scripts/)):
- `worker.py`, `harness.py`, `rebuild.py` and `dispatch.py` are 04's harness, adapted;
- `run_wider.py` and `launch.py` are the registered run;
- `l0.py` is gate L0;
- `analyse.py` scores the run.

**Outputs:**
- [`out/L0.txt`](05/wider/out/L0.txt) and [`out/L0_final.txt`](05/wider/out/L0_final.txt);
- [`out/wider_05.txt`](05/wider/out/wider_05.txt), which holds every unit, every group and the
  scores;
- [`out/cmd/`](05/wider/out/cmd/), every command with its output.

#### Gate L0

Taken on `low-G+r-1` (2, 38302) and `ctrl-low-1` (2, 2095), under $p$. Everything was compared by
`==` and `float.hex`.

| | Number | Verdict |
|---|---|---|
| **(a)** `QuadSource` at $p$ against the stored row | 360 of 360 stored nodes bit-identical in all nine value columns; 3 nodes added, the switch node and its 2 overlap nodes, which is exactly B1's; none removed. The same holds in all 109 units of the run | holds |
| **(b)** two in-process runs of the same unit | the result, every group call and every region array (by bytes) identical, on both rows | holds |

L0 was re-run with the final scripts after one fix to the worker's bookkeeping, and gave an
identical body ([`out/L0_final.txt`](05/wider/out/L0_final.txt)).
[`state/predictions.md`](05/wider/state/predictions.md) was written at 01:57:36Z, after L0 held and
before the 02:00:05Z launch.

#### Verdicts, scored as written

| | Rule | Number | Verdict |
|---|---|---|---|
| **L1** | every depth-20 row of the sample topped by `G+r`, `G−r` or lone `G`: under $c$ at 1e-7, $p$'s scored groups each < 10 regions at depth ≥ 18, and regions ≤ $p$'s / 10 | 9 of 24 rows. Clause 1 holds on all 24. Clause 2 fails on 15: 14 rows not capped under $p$, where $c/p$ is 0.73–1.09, and R5 at 0.345 | **fails** |
| **L1q** | the same, on the `Gqr`-topped rows | 1 of 7 (R7, $c/p$ = 0.0016). The 6 failures are the drawn `Gqr` rows, none capped under $p$: $c/p$ 1.00–1.35. None has a region at depth ≥ 18 under $c$ or $p$ | **fails** |
| **L2** | every control: $\lvert{\rm total}_c - {\rm total}_p\rvert \le$ `total_abserr`$_p$ | 8 of 8; largest ratio 0.652 (`ctrl-low-2`); the two `NULL` controls bit-identical | **holds** |
| **L3** | every depth-20 row with both $c$ units in budget: $\lvert c(10^{-7}) - c(10^{-8})\rvert \le$ `total_abserr`$_c(10^{-7})$ | 29 of 31; fails on `mid-G-r-3` (1.29) and `mid-G-3` (1.70), neither capped under $p$; holds on all 11 rows capped under $p$ (largest 0.166) | **fails** |

No unit went over budget.

#### The rows capped under $p$ (not scored; the reading the user asked for)

$p$ is B1's code without 05's change. 11 of the 31 depth-20 rows are still at depth 20 under it.
The other 20 are not capped under $p$, R2 among them (05's `issues.md` W2).

| row | class | band | $p$: depth, regions | $c$ 1e-7: depth, regions | $c/p$ | $p$'s scored groups (regions at depth ≥ 18, $p$ → $c$) | L1 clause 1 | clause 2 |
|---|---|---|---|---|---|---|---|---|
| `low-G+r-3` | `G+r` | low | 20, 1340 | 8, 33 | 0.0246 | `G` 1199 → 0 | holds | holds |
| `low-G-r-2` | `G−r` | low | 20, 15068 | 7, 42 | 0.0028 | `G` 11741 → 0 | holds | holds |
| `low-G-r-3` | `G−r` | low | 20, 6822 | 8, 40 | 0.0059 | `G` 1671 → 0; `G−r` 4840 → 0 | holds | holds |
| `low-G-1` | `G` | low | 20, 3902 | 11, 34 | 0.0087 | `G` 3676 → 0 | holds | holds |
| `low-G-3` | `G` | low | 20, 831 | 7, 39 | 0.0469 | `G` 689 → 0 | holds | holds |
| R1 | `G+r` | mid | 20, 22852 | 8, 63 | 0.0028 | `G` 8908, `G+r` 6597, `G−r` 6075 → 0 | holds | holds |
| R3 | `G+r` | mid | 20, 10457 | 5, 18 | 0.0017 | `G+r` 9635 → 0 | holds | holds |
| R4 | `G−r` | mid | 20, 42830 | 8, 83 | 0.0019 | `G` 7545, `G+r` 15539, `G−r` 17307 → 0 | holds | holds |
| R5 | `G−r` | mid | 20, 55 | 5, 19 | 0.345 | none (`G+r` has 7) | holds | **fails** |
| R6 | `G` | mid | 20, 5130 | 8, 28 | 0.0055 | `G` 3115, `G+q+r` 1725 → 0 | holds | holds |
| R7 | `Gqr` | mid | 20, 16660 | 8, 27 | 0.0016 | `G` 11393, `G+q+r` 4647 → 0 | holds | holds |

**On these rows, the change removes the cap everywhere.**
- No group of any capped row keeps a region at depth ≥ 18 under $c$, at 1e-7.
- The maximum depth under $c$ is 5–11 at 1e-7, and 5–11 at 1e-8.
- L1's rule holds on 9 of 10 and L1q's on 1 of 1. R5 misses the tenfold clause because its cap
  under $p$ was one chain of about 50 regions (05's `issues.md` W4).
- L3's rule holds on all 11.

**Cost on these rows.** The $p$ wall time is 0.4–262 s per row, 644 s in all; under $c$ it is
0.2–0.55 s per row, 4.4 s in all. The budgets were never approached.

#### Descriptive

- **$\lvert{\rm total}_c - {\rm total}_p\rvert$ / `total_abserr`$_p$** (04's N4, wider, now with the
  modulus correction too):
  - on the capped rows: 0.0225–0.634, and 2.48 on `low-G-3` (05's `issues.md` W6);
  - on the uncapped rows: ≤ 0.932.
  - At $p$, R3, R6 and R7 are within their error: 0.406, 0.403 and 0.17. At the pre-B1 tree, 04
    found 1.16, 1.53 and 1.94.
- **By band:**

  | Rows | Capped under $p$ | Regions, $p$ → $c$ | Wall, $p$ → $c$ |
  |---|---|---|---|
  | Low band, drawn | 5 of 12 | 28,054 → 288 | 101 s → 4.5 s |
  | Mid band, drawn | 0 of 12 | 352 → 321 | 3.7 s → 3.5 s |
  | R1–R7 (mid) | 6 of 7 | 97,999 → 250 | 545 s → 2.4 s |

  The controls are unchanged: 42 regions under both trees.
- **`total_converged`:**
  - On the depth-20 rows it is False under $c$ on 21 of 31 at 1e-7 and 19 of 31 at 1e-8.
  - On the capped rows, 9 of 11 stay unconverged although the cap is gone (05's `issues.md` W7).
  - 3 rows have an infinite `total_abserr` under both trees.
- **B1 against the stored rows** (05's `issues.md` W1): $\lvert{\rm total}_p - {\rm total}_{\rm stored}\rvert$
  exceeds the stored `total_abserr` on 36 of 39 rows. The ratios are 5.2 to 1.5e5. The three rows
  within error are `low-Gqr-3` (stored error infinite) and the two `NULL` controls (bit-identical).
