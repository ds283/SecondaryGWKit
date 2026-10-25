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
