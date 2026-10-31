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

## 5a. Layer 1(a) for $T$

**Prompt 05a, 2026-10-26, in three parts**, as 05 ran. The code was written on the laptop; R11 and
R12 and the re-measurement of 05's 39 rows ran on the Mac Studio.

- **The code** is `1fc349c`. `TkSourceFunctions` gains the keyword `exact_remainder`, default on,
  passed to its WKB phase's `PrimitivePhase`. So $T$'s remainder is formed node by node and reduced
  against $2\pi$, by the construction 05 built for $G$. $T$'s anchor `z_init` is a root, so the
  anchor's end partial is non-zero in production for the first time. `raw_theta`, `theta_deriv`
  and $G$'s phase are unchanged, and `primitive_phase.py`'s code did not move. The tests and the
  four breakages are in `prompts/levin-phase-floor/logs/05a-reanchor-t-phase.md`.
  - On the test fixture, σ_θ sampled in $z$ at a table node falls from 0.24 to
    $4.8\times10^{-8}\,\varepsilon|\theta_T|$.
  - `bd38e50` corrected three comments the code commit left stale.
- **T1 was amended before §6 ran** (prompt §9, `2db3f9e`, the user's decision). The code's tests
  showed that a probe window of relative width $10^{-6}$ is not resolved by its 65 samples at
  large $|\theta_T|$, and that between table nodes the remainder keeps $\varepsilon k|p_z|$. §9
  narrows each window until it is resolved, and admits $\varepsilon(k\max|p_z| + 2\pi)$ in the bound
  under $t$.
- **R11 against R12** is §5a.1.
  - Both give `1b332ffa…`, and every value table of every class is bit-identical.
  - Outside `QuadSourceIntegral`, only the four volatile kinds of
    `[05-store-value-diff-compares-volatile-columns]` differ.
  - 424 of 864 totals move, by at most $2.9\times10^{-11}$ relative and $8.8\times10^{-5}$ of the
    declared error. No depth, region count or convergence flag changes.
- **The re-measurement** is §5a.2.
  - T0 holds: all 39 of 05a's $c$ units equal 05's $c$ units bit for bit.
  - **T1 holds by the true step** (70 of 70 windows; under $t$, σ_θ is at most
    $1.3\times10^{-5}\,\varepsilon|\theta_T|$, against 0.20–0.37 under $c$). Read literally it
    fails in 36, all aliased.
  - T2 and T3 hold. $T$'s two errors moved a total by at most $4.7\times10^{-6}$ of its declared
    error, and changed no Levin call's regions, depth or flag.
  - T4 fails on the two rows on which 05's L3 failed, at the same ratios.
  - Sampled in $u$, as production calls it, the rounding of $z = {\rm expm1}(u)$ remains: 0.14–0.29
    of $\varepsilon\tfrac{z}{1+z}|d\theta_T/du|$ (`[05a-t-phase-abscissa-rounding-remains]`).

**T1's reading, and when it was decided.** §9's rule "the largest step of the unwrapped samples"
was meant as the phase's true step per sample. Read literally, it is the folded step, which admits
aliased windows. The re-measurement agent fixed both readings in `predictions.md` at 12:30:03Z,
before the run, with the literal one to be scored, and asked which §9 meant. Both were scored at
12:35:49Z. The question reached the laptop while the run was reported as going, and the laptop
told the user that no T1 number existed yet. That was wrong. The user chose the true step, without
having seen either number, and the ruling reached the Mac Studio at about 13:23Z. After the laptop
corrected the record of the timing, the user confirmed the true step as T1's verdict of record.
The literal reading's 36 failures stay beside it.

**The issues.** On the user's word, `[05-t-remainder-keeps-the-anchoring-floor]` and `[04-g-phase-is-reduced-against-the-double-two-pi]`
are closed. §B.4 of the orchestrator note asks the user first whenever T1 or T4 fails, and T4 fails.
That failure is 05's, unchanged by $T$.

**Records** are in [`05a/`](05a/):
- `rehearsal/` and `wider/`, each holding `out/`, `scripts/`, `state/`, the agent's `log.md`,
  `verification.md` as first written, and `issues.md`;
- the Mac Studio orchestrator's [`review.md`](05a/review.md), with its scratch readers reproduced
  in full.

The scripts are byte-identical to the runs and are left unformatted (`CLAUDE.md`). The readers'
`extract/`, the pickles, the CSVs and the tool's 41 MB and 10 MB outputs stay on the Mac Studio.

**Set in on committing** by the laptop orchestrator, from the two agents' `verification.md`:
- the subsections are numbered 5a.1 and 5a.2;
- three links to files kept on the Mac Studio only are made plain text.

The text is otherwise the agents'.

**The laptop's re-checks** (orchestrator note §B.4), read-only over SSH:
- on the row of largest $|\Delta{\rm total}|/$`total_abserr` ($k = q = r = 3.162\times10^5$,
  $z_{\rm response} = 6904.72$), R11 and R12 read the tabulated totals bit for bit
  (`0x1.2b4c92b9fb462p-60`, `0x1.2b4c92b9ffc9cp-60`);
- on `low-G-r-2`, 05a's $c$ unit at `1e-7` equals 05's $c$ unit in all 16 fields by bit pattern;
- A3 v2's five files hash as they did before the run.

### 5a.1 Rehearsals R11 and R12, and the value comparison (prompt 05a §5)

**Prompt 05a §5, run 2026-10-26, 12:20–12:33 UTC, on the Mac Studio** (mps026298, arm64, Darwin
27.0.0).
- **The checkout.** The records were written in the main checkout at `149c393`
  (`handover-remedial`, clean throughout). Its code is the code commit `1fc349c`'s, up to
  `bd38e50`'s comment-only changes.
- **The software.** Python 3.13.16 from the main checkout's venv: `ray` 2.59.0, `SQLAlchemy`
  2.0.55, `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit` 0.2.0.
- **The load** was about 5–6 on 10 cores. Prompt §6's Ray-free re-measurement ran beside the
  readers, from 12:30:47.

**Records** are in [`05a/rehearsal/`](05a/rehearsal/): `out/`, `scripts/` and `state/`, the
agent's `log.md`, this section as first written, and `issues.md`.
- Every command is in [`out/commands.log`](05a/rehearsal/out/commands.log).
- The scripts' SHA-256 are in [`out/script_sha256.txt`](05a/rehearsal/out/script_sha256.txt). All
  four (`classify_non_qsi.py`, `tabulate_qsi.py`, `tools/store_value_diff.py`, `run_readers.sh`)
  equal 05's.
- The readers' `extract/` was not copied back.

#### The two trees

| | R11 (before) | R12 (after) |
|---|---|---|
| tree | worktree `SecondaryGWKit-lpf05a-c4eaefd`, detached at `c4eaefd` (the code commit's parent) | main checkout at `149c393` (code = `1fc349c`) |
| how it ran | cwd and `PYTHONPATH` the worktree; the main checkout's venv python by absolute path; `PYTHONDONTWRITEBYTECODE=1` | R10's launch: `PYTHONPATH=.`, `./venv/bin/python` |
| run | `levin-phase-floor-05a-r11-parent-20261026T122041` | `levin-phase-floor-05a-r12-reanchor-t-20261026T122434` |
| `sh` / driver pid | 22677 / 22684 | 24894 / 24897 |
| wall | 12:20:41 → 12:23:39 | 12:24:34 → 12:27:25 |
| manifest | `git_head` `c4eaefda…`, `git_dirty` false, cwd the worktree | `git_head` `149c3938…`, `git_dirty` false, cwd the main checkout |
| launcher SHA-256 | `5b984c75…` (R8's) | `5b984c75…` |
| end | `done`, exit 0, no Traceback, `stderr.log` empty | `done`, exit 0, no Traceback, `stderr.log` empty |
| store | `var/datastores/levin-phase-floor-05a-r11.sqlite` + 4 shards | `var/datastores/levin-phase-floor-05a-r12.sqlite` + 4 shards |

**The scope.** Both builds were R8's rehearsal scope:
- 3 k over 1e5–1e6/Mpc, zend 250;
- atol 1e-32, rtol 1e-7, 6 CPUs;
- the same `--job-name levin-phase-floor-05a`, and `--campaign levin-phase-floor --prompt 05a`.

**The worktree.** At R11's launch, `lsof` showed that the driver's cwd, and that of the Ray
`Datastore` actor 22808, was the worktree
([`out/r11_launch_check.txt`](05a/rehearsal/out/r11_launch_check.txt)). After both runs, the
worktree's `git status --short --ignored` was empty, and `find -newer` listed nothing in it.

#### The lock

[`out/lock.log`](05a/rehearsal/out/lock.log). Each command was a shell command of its own, output by
`>>`.

| # | command | output | exit |
|---|---|---|---|
| 1 | `acquire --project SecondaryGWKit --purpose "levin-phase-floor prompt 05a, rehearsals R11 … and R12 …" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 2 | `set-pid … 22684` (R11) | recorded | 0 |
| 3 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after R11) | nothing | 0 |
| 4 | `set-pid … 24897` (R12) | recorded | 0 |
| 5 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after the readers) | nothing | 0 |
| 6 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 7 | `status --ray-temp-dir /tmp/ray-sgk` | free; no Ray processes | 0 |
| 8 | `ray-processes --ray-temp-dir /tmp/ray-sgk` | nothing | 0 |

#### The fingerprints

| | digest | problems | per-class |
|---|---|---|---|
| R8 (reference, `handover-12-r8-reference-20261025T210551`) | `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` | none | 21 classes |
| **R11** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 digests equal R8's |
| **R12** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 equal R8's; the whole `classes` object equals R11's |

The fingerprints are the ones the launcher took at the end of each build, read from `status.json`
([`out/r11_end_check.txt`](05a/rehearsal/out/r11_end_check.txt),
[`out/r12_end_check.txt`](05a/rehearsal/out/r12_end_check.txt); the files themselves are
[`out/r11_status.json`](05a/rehearsal/out/r11_status.json) and
[`out/r12_status.json`](05a/rehearsal/out/r12_status.json)). There is no sidecar, because
`--build` writes none.

#### The value comparison (prompt §5.4)

`PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff var/datastores/levin-phase-floor-05a-r11.sqlite var/datastores/levin-phase-floor-05a-r12.sqlite --json`,
run in the main checkout at `149c393`. It exited **1**, as expected, in 17 s. The text form, with
`--max-rows 1000000`, also exited 1.

Outputs:
- `out/store_value_diff.json` (41 MB, on the Mac Studio only);
- `out/store_value_diff.txt` (10 MB, on the Mac Studio only);
- [`out/store_value_diff_summary.txt`](05a/rehearsal/out/store_value_diff_summary.txt);
- [`out/store_value_diff.meta.txt`](05a/rehearsal/out/store_value_diff.meta.txt).

- **Structure.** 21 classes were compared, and `not_compared` is empty. In every class:
  - 0 rows are in one store only;
  - 0 rows cannot be matched by identity;
  - 0 columns are unresolvable.
- **Every value table of every class is bit-identical between R11 and R12**, as the tool's JSON
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

- **Row columns outside `QuadSourceIntegral`.** As in 05, 12 classes are reported value-identical:
  `version`, `store_tag`, `redshift`, `wavenumber`, `tolerance`, `LambdaCDM`, `QCD_Cosmology`,
  `IntegrationSolver`, `GkSourcePolicy`, `QuadSourcePolicy`, `GkSourcePolicyData`, and
  `OneLoopIntegral`, which has no rows.
- **Eight classes differ, in row columns only.** 05's
  [`scripts/classify_non_qsi.py`](05a/rehearsal/scripts/classify_non_qsi.py), copied unchanged
  (`26690e61…`), assigns every differing column to one of the four volatile kinds of
  `[05-store-value-diff-compares-volatile-columns]`. **0 are left unclassified**
  ([`out/classify_non_qsi.txt`](05a/rehearsal/out/classify_non_qsi.txt)):

  | kind | columns | classes (rows) | column-differences |
  |---|---|---|---|
  | wall-clock timing | `compute_time`, `stage_1_compute_time`, `{min,mean,max}_RHS_time` | wavenumber_exit_time (3), BackgroundModel (1), TkNumericIntegration (3), TkWKBIntegration (3), GkNumericIntegration (1368), GkWKBIntegration (3474), QuadSource (6) | 8607 |
  | `label` | equal once the trailing `datetime.now()` stamp is removed | TkNumeric (3), TkWKB (3), GkNumeric (1368), GkWKB (3477), GkSource (291), QuadSource (6) | 5148 |
  | `GkSource.parents_digest` | the SHA-256 of the parents' serials (05 §5.x) | GkSource (291 of 291) | 291 |
  | GkWKB work counters | `metadata` keys `rho_evals`, `rho_reused`; `stage_1_RHS_evaluations` | GkWKBIntegration (31 of 3477) | 62 |

  Nothing outside `QuadSourceIntegral` differs in any other way, so prompt §8's stop does not fire.

#### `QuadSourceIntegral`, tabulated per row

05's [`scripts/tabulate_qsi.py`](05a/rehearsal/scripts/tabulate_qsi.py), copied unchanged
(`b7508351…`), reads both stores (`mode=ro`) and the tool's JSON. Its docstring still names R9 and
R10; **here A = R11 (parent `c4eaefd`) and B = R12 (code `1fc349c`)**. Its outputs are
[`out/qsi_tabulation.txt`](05a/rehearsal/out/qsi_tabulation.txt), with one row per integral in
`out/qsi_rows.csv` (on the Mac Studio only). Further counts from the CSV are in
[`out/qsi_extra_counts.txt`](05a/rehearsal/out/qsi_extra_counts.txt).

- **The pairing.** Rows were paired by physical (k, q, r, z_response), giving 864 = 864, with one
  atol and one rtol, (−32, −7). The pairing has the same key set as the tool's 864 identity-matched
  rows, and 0 mismatches in `total`'s `float.hex` against the tool.
- **All 864 rows differ in some column.** Wall-clock columns and `label` differ in every row, and
  so does `metadata` (below).

| | rows |
|---|---|
| `total` differs | **424** of 864 (440 bit-identical) |
| `WKB_Levin` differs | 435. Every row whose `total` differs is among them; in 11 more, the change rounds away in `total` |
| `numeric_quad` / `analytic_rad` differ | **0** / **0** |
| `total_abserr` differs | 476 |
| rows with no `WKB_Levin` part (`max_depth` None) | 291, none of which differs in `total` |

Over the 424 rows whose `total` differs:

| statistic | median | 90 % | max |
|---|---|---|---|
| $\lvert\Delta{\rm total}\rvert/\lvert{\rm total}_A\rvert$ | 1.38e-15 | 2.84e-14 | 2.92e-11 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ | 8.25e-09 | 4.36e-08 | 8.81e-05 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_B$ | 8.25e-09 | 4.36e-08 | 8.81e-05 |

- Rows with $\lvert\Delta{\rm total}\rvert/$`total_abserr` > 0.1: **0** (and so 0 above 1).
- **The largest ratio** is at $(k,q,r,z_{\rm resp}) = (3.162\times10^5, 3.162\times10^5,
  3.162\times10^5, 6904.72)$. There the ratio is 8.8e-5 and the relative change 3.5e-12, at depth
  11, with 32 regions and `total_converged` 0 in both stores. The 15 largest ratios are all on the
  equal-$k$ diagonals $k = q = r = 10^5$ and $3.162\times10^5$, at depths 10–12.
- **`total_converged`:** no row changes (0→0: 355, 1→1: 509). **`total_phase_limited`:** no row
  changes (0→0: 833, 1→1: 31).
- **`WKB_Levin_max_depth`:** no row changes. **`WKB_Levin_num_regions`:** no row changes, with
  13,652 regions in both stores.
- **By depth under A,** the median $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ is:
  - 2e-9 to 1.1e-8 at depths 2–8;
  - 1.5e-6 to 1.7e-5 at depths 10–14;
  - the maximum anywhere is 8.8e-5, at depth 11.
- **`total_abserr` = inf in both stores:** 26 rows, at depths 0–8. 12 of their `total`s differ, by
  1.1e-16 to 1.0e-15 relative, and the tabulation prints their ratio as 0. This is the open issue
  `[04-roundoff-floor-can-be-infinite]`.
- **Metadata.** The non-wall-clock keys that differ, 29 distinct, are two families:
  - `partition.handover.T{q,r}.handover_continuity[i].d{T,Tprime}_rel`, **in every one of the 864
    rows**. This is expected (§B.5). The diagnostic takes $\sin$ and $\cos$ of $T$'s
    `theta_mod_2pi`, which is what this prompt changes. In 292 rows these are the only
    non-wall-clock keys that differ: the 291 rows with no WKB part, and one at depth 0;
  - the `WKB_Levin` `value`/`abserr` tree (`WKB_Levin.abserr` in 476 rows, `subintervals[i]`, and
    `subintervals[i].groups[j]`), in 572 rows. No `regions`, `simple_regions` or `evaluations` key
    differs.
- **R11 and R12 are not value-identical in `QuadSourceIntegral`.** So each tree ran the code it
  names.

These are recorded, not judged. Prompt §6 scores the effect on A3 v2's rows.

**Beside 05's R9/R10, for orientation only.** 05 changed $G$'s phase, and moved 542 `total`s by up
to 4.0e-5 relative and 28× `total_abserr`. It changed 15 depths and 42 region counts. This prompt
changes $T$'s phase, and moves 424 `total`s by at most 2.9e-11 relative and 8.8e-5 ×
`total_abserr`. It changes no depth, region count or convergence flag.

#### The readers (prompt §5.5)

`bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh var/datastores/levin-phase-floor-05a-r12.sqlite var/levin-phase-floor-05a/rehearsal/extract`,
with the lock held, from 12:29:51 to 12:32:26 UTC
([`out/readers.txt`](05a/rehearsal/out/readers.txt)). Overall exit **0**.

| reader | exit | wall | lookups | files written (05's R10) | store files after |
|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 28 s | 116 | 387 (450) | identical |
| `extract_GkWKB_data.py` | 0 | 57 s | 348 | 2436 (2436) | identical |
| `extract_GkSource_data.py` | 0 | 20 s | 10 | 162 (142) | identical |
| `extract_TkWKB_data.py` | 0 | 15 s | 3 | 27 (27) | identical |
| `extract_tensor_source_data.py` | 0 | 14 s | 6 | 27 (27) | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 17 s | 24 | 73 (74) | identical |

- The lookups equal 05's and R8's.
- File counts differ only where a reader draws an unseeded subsample
  (`[10-the-readers-draw-their-work-lists-unseeded]`).
- After each reader, 0 Ray processes of this venv were left, and no reader log has a Traceback.

**R12's store, before and after the readers.** The `diff` is empty, and there is no `-journal`,
`-wal` or `-shm` file
([`state/r12_store_sha256_before_reads.txt`](05a/rehearsal/state/r12_store_sha256_before_reads.txt),
[`state/r12_store_sha256_after_reads.txt`](05a/rehearsal/state/r12_store_sha256_after_reads.txt)):

```
94dae1cca60ef5d58e9ef298d8fdd4fbf937a7eebbdd3f1495e9ca0b6300c749  levin-phase-floor-05a-r12.sqlite            36864
b42aa513f7d75b563ed196e4e8d6065c20606e8357565ebec912d36dbacb4e04  levin-phase-floor-05a-r12-shard0000.sqlite  24076288
a5d54bbbe8d6b1a64c47ea252f5b11be1dad640e2d061746493276e55254a73f  levin-phase-floor-05a-r12-shard0001.sqlite  24203264
af01699e932eeda9b24d8e4661fb284c3bec8ad1f0531f6c4f8bcb4d4897b273  levin-phase-floor-05a-r12-shard0002.sqlite   1077248
5b249c8510b5bd91ece9cb67bc060e17cd12bbac475c71246836bb861835c9da  levin-phase-floor-05a-r12-shard0003.sqlite  23560192
```

**Both stores, all ten files, are byte-identical from before the comparison to the end**
([`state/store_sha256_before_diff.txt`](05a/rehearsal/state/store_sha256_before_diff.txt),
[`state/stores_sha256_final.txt`](05a/rehearsal/state/stores_sha256_final.txt)). R11's files equal
the hashes taken just after its build
([`state/r11_store_sha256_after_build.txt`](05a/rehearsal/state/r11_store_sha256_after_build.txt)).
R11's primary is `8553800554d70f69d0fd90655899ed819bb5bf2456d62456da293f660e1a7cb8`.

#### Against prompt §7.2

- R11 and R12 both end `done` with fingerprint `1b332ffa…`: **yes**.
- Every value table outside `QuadSourceIntegral` is identical, and every other difference there is
  of the four kinds: **yes**.
  - All 7 value tables are bit-identical: 360,462 rows.
  - 0 differences are unclassified, and 0 rows are in one store only.
- The `QuadSourceIntegral` differences are tabulated: **yes**.
  - 424 of 864 `total`s moved, all through `WKB_Levin`, by at most 2.9e-11 relative and 8.8e-5 ×
    `total_abserr`.
  - No depth, region count or convergence flag changed.
  - `handover_continuity` moved in every row.
- The readers exit 0: **yes**, and the store is byte-identical.
- The lock is released: **yes**.

### 5a.2 The wider re-measurement for $T$ (prompt 05a §6, with §9's T1)

**Prompt 05a §6, with T1 as amended by §9, run 2026-10-26 on the Mac Studio (mps026298).** There
was no Ray, no lock and no store write (route (a), as in §3–§5). Two trees took part:
- $c$, the code commit's parent `c4eaefd`, whose code is 05's $c$: $G$ node-anchored, $T$ not. It
  ran from the git worktree `SecondaryGWKit-lpf05a-c4eaefd`.
- $t$, the code commit `1fc349c`: $T$'s remainder node-anchored and reduced against $2\pi$. It ran
  from the main checkout at `149c393`, whose code is `1fc349c`'s up to `bd38e50`'s comments.

The rows are 05's 39, from 05's `sample.json` copied unchanged
([`state/sample.json`](05a/wider/state/sample.json), SHA-256 `73aa7876…`, equal to the
`sample_sha256` 05's run recorded). The harness is 05's
([`scripts/`](05a/wider/scripts/); its diff against 05's copy is
[`out/scripts_diff_against_05_final.txt`](05a/wider/out/scripts_diff_against_05_final.txt)).
Its trees are renamed $c$/$t$, and it adds the phase probe ([`scripts/probe.py`](05a/wider/scripts/probe.py)).

Each worker loaded its packages from its own tree only, as recorded per unit. Under $c$, every $T$
phase had `exact_remainder` False and every $G$ phase True. Under $t$, both were True.

**The run.** `levin-phase-floor-05a-wider-20261026T123047` ended `done`, 109/109 units, exit 0,
in under two minutes. Its checkpoints and pickles stay in that run directory on the Mac Studio.
- The units ran in order: $c$@1e-7 on all 39 rows, $t$@1e-7 on all 39, and $t$@1e-8 on the 31
  depth-20 rows. The budget of the $t$ units was 3× the row's $c$ wall time.
- Every record carries `run_wider.py`'s SHA-256 (`af656ad4…`) and its tree's `HEAD` and empty
  `git status --short`.
- No unit went over budget (the largest wall/budget is 0.861), none raised, and the probe raised
  nowhere.

**The store** was byte-identical before, before the launch, after the run and after the
fingerprint ([`before`](05a/wider/state/store_files_before.txt),
[`after`](05a/wider/state/store_files_after_fingerprint.txt)). No journal, `-wal` or `-shm` file
appeared. `store fingerprint`, run without `--write`, gives
`0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56` and `problems: none`
([`out/fingerprint_05a_wider.txt`](05a/wider/out/fingerprint_05a_wider.txt)).

**The phase probe** reads phase objects only and runs no quadrature.
- **Where.** It covers every Levin sub-interval of every row with an oscillatory $T$ factor, and
  every such factor: 91 windows on 37 rows. Each window starts at the sub-interval's low-$z$ end
  $z_{\min}$ and is $[z_{\min}, z_{\min}(1+w)]$ with $w = 10^{-6}/2^n$.
- **What each tree computes at every level** until its 65 samples stop being distinct doubles:
  - $\sigma_\theta$ of `theta_mod_2pi` sampled in $z$, as §3 test 3 computes it;
  - the same sampled in $u$;
  - the largest unwrapped step;
  - $\max|p_z|$, from `cs_tau.end_partial`.
- **The level used** is the first at which both trees' largest unwrapped step is below $\pi/2$.
  No window with $\varepsilon|\theta_T| \ge 10^{-11}$ was unresolved.
- **Where it is kept.** Every level of both ladders, and the samples as `float.hex` at the level
  used, are in [`state/probe_windows.json`](05a/wider/state/probe_windows.json).
  [`scripts/rescore_t1.py`](05a/wider/scripts/rescore_t1.py) re-scores T1 from that file alone,
  recomputing every $\sigma$ from the samples, and agrees in every window
  ([`out/cmd/24_rescore_t1.txt`](05a/wider/out/cmd/24_rescore_t1.txt)).

**Outputs:**
- [`out/T0.txt`](05a/wider/out/T0.txt);
- [`out/wider_05a.txt`](05a/wider/out/wider_05a.txt): every unit, every group, every probe window
  under both readings, and the scores;
- [`out/convergence_kinds_t.txt`](05a/wider/out/convergence_kinds_t.txt);
- [`out/commands.log`](05a/wider/out/commands.log), which has every command with its output.

#### Gate T0

| | Number | Verdict |
|---|---|---|
| **(a)** `git diff --stat 120fdfe c4eaefd -- . ':!docs' ':!prompts'` | `CLAUDE.md` alone | holds |
| **(b)** $c$@1e-7 against 05's $c$ unit, on `low-G+r-1` and `ctrl-low-1` | `total`, `total_abserr`, and all 7 calls' value and abserr: 16 doubles per row, identical by `==` and `float.hex`. So are the whole result summary, the call summaries and every region array of 05's pickles | holds |
| **(c)** two in-process runs of the same $t$ unit, on both rows | result, calls, regions, probe summaries and every probe sample bit-identical | holds |

[`state/predictions.md`](05a/wider/state/predictions.md) was written at 12:30:03Z, after T0 held
(12:28:33Z) and before the 12:30:47Z launch. It quotes §6's predictions with §9's T1 in place of
§6's, and fixes how each rule is read.

#### Verdicts, scored as written

*(T1's scored verdict was later taken by the true step, by the user's decision: it holds. See
below the table. The T1 row here is the literal reading, kept beside it.)*

| | Rule | Number | Verdict |
|---|---|---|---|
| **T1** (§9) | in every resolved window with $\varepsilon\lvert\theta_T\rvert \ge 10^{-11}$ rad, sampled in $z$: $\sigma_c \ge 0.1\,\varepsilon\lvert\theta_T\rvert$ and $\sigma_t \le \max(10^{-3}\varepsilon\lvert\theta_T\rvert,\ \varepsilon(k\max\lvert p_z\rvert+2\pi),\ 10^{-13})$ | 70 scored windows, 0 unresolved. The $c$ clause holds in all 70: $\sigma_c$ = 0.196–0.374 $\varepsilon\lvert\theta_T\rvert$. The $t$ clause holds in 34 ($\sigma_t \le 1.3\times10^{-5}\,\varepsilon\lvert\theta_T\rvert$) and fails in 36 ($\sigma_t$ 0.016–0.23 $\varepsilon\lvert\theta_T\rvert$). Those 36 are exactly the windows the literal step rule accepts while the phase steps 4.9–352 rad per sample (below) | **fails** |
| **T2** | every row (39), $t$@1e-7: no group has a region at depth ≥ 18 | 39 of 39; largest depth 11 | **holds** |
| **T3** | every row (39): $\lvert{\rm total}_t - {\rm total}_c\rvert \le$ `total_abserr`$_c$ at 1e-7 | 39 of 39; largest ratio $4.66\times10^{-6}$ (`low-G-1`); largest relative change $3.6\times10^{-12}$ (R4); 13 totals bit-identical | **holds** |
| **T4** | every depth-20 row (31) with both $t$ units in budget: $\lvert t(10^{-7}) - t(10^{-8})\rvert \le$ `total_abserr`$_t(10^{-7})$ | 29 of 31; fails on `mid-G-r-3` (1.29) and `mid-G-3` (1.70), the same rows and ratios as 05's L3 under $c$; holds on all 11 rows capped under B1 (largest 0.166) | **fails** |

**T1's failure is the probe's step rule, not the phase.**
- §9 halves a window while "the largest step of the unwrapped samples" is $\pi/2$ or more. Read
  as written, that is the folded step, which can be small while the samples advance by many cycles.
- On those windows, `numpy.unwrap` removes whole cycles per index. The rounding of the abscissae
  then reads as jitter of $0.13$–$0.30\,\varepsilon z|d\theta_T/dz|$, whatever the remainder.
- **By the true step** (unscored; both readings were fixed before the run, and both are kept), the
  same rule holds in **70 of 70** windows:
  - $\sigma_t$ is $4.4\times10^{-10}$–$1.31\times10^{-5}\,\varepsilon|\theta_T|$, at most 0.013 of
    its bound;
  - $\sigma_c$ is 0.198–0.374.
  - The reading by the true step means the first level at which $|d\theta_T/dz|(b-a)/64 < \pi/2$
    also holds in both trees.
- The question of which reading §9 means is the user's (05a's `issues.md` X1).

**T1's scored verdict, by the user's decision after the run: holds.**
- The coordinator relayed the decision to the Mac Studio session at about 13:23Z. The user chose
  "True step (Recommended)": *"T1 is scored on windows whose true step |dθ/dz|(b−a)/64 is below
  π/2, narrowing further where it isn't. This is what §9 meant by 'resolved'. The literal
  unwrapped-step reading is reported beside it, unscored."*

| | Rule | Number | Verdict |
|---|---|---|---|
| **T1** (§9, scored by the true step) | as above, on windows resolved by $\lvert d\theta_T/dz\rvert(b-a)/64 < \pi/2$ under both $c$ and $t$ | 91 windows, 70 with $\varepsilon\lvert\theta_T\rvert \ge 10^{-11}$, 0 unresolved, true in 70. $\sigma_c$ 0.198–0.374 $\varepsilon\lvert\theta_T\rvert$; $\sigma_t$ $4.4\times10^{-10}$–$1.31\times10^{-5}\,\varepsilon\lvert\theta_T\rvert$, at most 0.0131 of its bound ([`out/cmd/26_T1_true_step_scored.txt`](05a/wider/out/cmd/26_T1_true_step_scored.txt)) | **holds** |
| T1 (§9, literal step; now unscored) | the table above | 36 of 70 false, all aliased | (fails) |

- **No re-probe was needed.** The kept levels resolve every window by the true step, and the level
  chosen by the true step alone is, in all 91 windows, the level whose samples were saved.
- **The timing, as the files show it,** is in
  [`state/predictions_note_T1_reading.md`](05a/wider/state/predictions_note_T1_reading.md):
  - `predictions.md` fixed both readings at 12:30:03Z;
  - both were first scored at 12:35:49Z;
  - the decision reached the Mac Studio session at about 13:23Z.

**§9's unscored items.**
- The term $\varepsilon(k\max|p_z| + 2\pi)$ is the largest of the three in **0** windows: the
  windows sit at or near table nodes, with $k\max|p_z| \le 10^{-6}|\theta_T|$.
- **Sampled in $u$**, as production calls the phase, $\sigma_t$ is 0.14–0.29 of the abscissa term
  $\varepsilon\tfrac{z}{1+z}|d\theta_T/du|$ everywhere.
  - It exceeds $0.1\,\varepsilon|\theta_T|$ in 14 windows at $z \gg 1$, at 0.10–0.22.
  - Under $c$ it is 0.21–0.40.
  - At $z < 1$ it falls to 0.0025–0.0096 under $t$.
  - This is §0.1's prediction: for $T$ the abscissa term is the size of the removed floor wherever
    $z \gg 1$. An issue is proposed (X2).

#### Descriptive

- **$t$ against $c$:**
  - all 219 Levin group calls are the same sequence, with the same region count, maximum depth and
    convergence flag. That includes the 140 `Gqr` groups and the regions at each sub-interval's
    low-$z$ end;
  - `total_converged` is unchanged on every row;
  - the totals move by at most $4.7\times10^{-6}$ of `total_abserr`$_c$.

  On these rows, $T$'s two errors were never what the integrator saw.
- **Cost:** the regions are unchanged (low band 288, mid band 321, R1–R7 250, the 11 capped rows
  426 at 1e-7). The wall times are 0.06–1.15 s per unit and carry the host's load.
- **The kinds of non-convergence** under $t$ ([`out/convergence_kinds_t.txt`](05a/wider/out/convergence_kinds_t.txt);
  `convergence_kinds.py` unchanged, SHA-256 `49c83a72…`):

  | | resolution | bound only | phase-limited |
  |---|---|---|---|
  | 1e-7 | 18 | 32 | 17 |
  | 1e-8 | 21 | 24 | 27 |

  These equal 05's $c$ (`prompts/levin-phase-floor/measure/out/convergence_kinds.txt`).

## 6. The stored phase

**Prompt 06, 2026-10-26 and 27, in three parts**, as 05 and 05a ran. The code was written on the
laptop; R13 and R14 and the re-measurement of 05's 39 rows ran on the Mac Studio; finding P1 was
investigated from the laptop afterwards, on the Mac Studio's stores.

- **The code** is `dc9d28c`. Each `GkWKBIntegration` and `TkWKBIntegration` value stores its
  residual $r = \Delta\theta - \rho.\delta$, formed from those two small terms, in a new
  `theta_residual` column (on `GkSourceValue` too). Its `theta_mod_2pi` and `theta_div_2pi` are
  the exact remainder and cycle count of the true phase against $2\pi$, formed in integer
  arithmetic from the table's limbs and partials. The consumers build $\varphi$ from $r$:
  $\varphi_T = r$, and $\varphi_G = r + k\,\tau.\delta(z_s, z_{\rm init})$, unwrapped in
  `_build_phase` and checked against the rectifier's shifts. The tests and the five breakages are
  in `prompts/levin-phase-floor/logs/06-producers-store-the-residual.md`.
  - On the test geometry every stored remainder is within 0.500 ulp($2\pi$) of the exact one at
    $k$ up to $3\times10^8$, and the consumers' $\varphi$ has window RMS at most
    $1.8\times10^{-10}\,{\rm ulp}(\theta)$, against 0.37–3.9 for the old construction.
- **The prompt was amended before and during dispatch** (its §9, points 1–6). Points 1–4 and 6
  were the user's decisions; S1 is scored on pure-WKB windows only.
- **R13 against R14** is §6.1. Both give `1b332ffa…`. Outside the five classes 06 changes, every
  value table is identical. Inside them, `theta_div_2pi` changes in no row and the remainder by a
  median 0.3 $\varepsilon|\theta|$; 554 of 864 `QuadSourceIntegral` totals move, by a median
  $2.8\times10^{-11}$ relative. Two totals at $z_{\rm response} = 250$ move by $1.8\times10^4$ and
  484 times R13's declared error: that is P1, §6.3. No disagreement warning; the readers exit 0.
- **The re-measurement** is §6.2. U0 holds. **S1 holds**: in all 6,700 pure-WKB $G$ windows on
  R1–R7, $\varphi_G$'s noise falls from production's quantised value to at most
  $8.8\times10^{-14}F$. S3 holds (largest depth 11). S2 fails in 2,115 of 12,862 $T$ windows, all
  at $z \ge 1.75\times10^7$, where the metric measures $\varphi_T$'s own shape equally under both
  trees. S4 fails on 5 of 39 rows and S5 on 3 of 31, both as findings about the declared error.
- **P1** is §6.3: a defect of `AdaptiveLevin`, not of 06.

**The decisions, each the user's.** Route (B) for reading A3 v2 under the new code; S1's reading
(I), `z_init == z_source`, ruled with no S1 number in existence; check 4 passed with the 185
`UNCLASSIFIED` all inside the five classes. On 2026-10-27,
`[02-consumer-phi-below-the-storage-granularity]` was closed on S1 and S3, with S2's failure read
as the metric's (W4); P1 was opened as an `AdaptiveLevin` defect for this campaign, to be fixed
generically; R13, R14 and the scratch copy of A3 v2 are kept.

**Records** are in [`06/`](06/):
- `rehearsal/` and `wider/`, each holding `out/`, `scripts/`, `state/`, the agent's `log.md`,
  `verification.md` as first written, and `issues.md`;
- the Mac Studio orchestrator's [`review.md`](06/review.md), with its scratch readers in `review/`;
- `p1/`, the investigation of P1: [`report.md`](06/p1/report.md), `scripts/` and `out/`;
- `laptop_rechecks/`, the laptop's two re-checks;
- [`KEPT_ON_MAC_STUDIO.md`](06/KEPT_ON_MAC_STUDIO.md), the SHA-256 of every file left on the Mac
  Studio for size.

The scripts are byte-identical to the runs and are left unformatted (`CLAUDE.md`).

**Set in on committing** by the laptop orchestrator, from the two agents' `verification.md`: the
subsections are numbered 6.1 and 6.2. The text is otherwise the agents'. §6.3 is the laptop's.

**The laptop's re-checks** (orchestrator note §B.4), read-only over SSH:
- on the changed `GkWKBValue` of largest $|\theta|$, each store holds exactly one row with the
  tabulated cycle count and remainder (`-0x1.8cf0843e97b54p+1` and `-0x1.8cf083e8164e1p+1`), and
  $|\Delta{\rm mod}| = 0.237\,\varepsilon|\theta|$
  ([`recheck4.txt`](06/laptop_rechecks/recheck4.txt));
- on `mid-G-1`, the $t_5$ unit at `1e-7` equals 05a's $t$ unit in total, `total_abserr` and all
  five calls, by bit pattern ([`recheck6.txt`](06/laptop_rechecks/recheck6.txt)).

### 6.1 Rehearsals R13 and R14, and the value comparison (prompt 06 §5)

**Prompt 06 §5, run 2026-10-26, 21:55–22:15 UTC, on the Mac Studio** (mps026298, arm64, Darwin
27.0.0).
- **The trees.** Both builds ran from git worktrees, so that the main checkout did not move:
  - R13 ran in `SecondaryGWKit-lpf06-9f66622`, detached at `9f66622`, the parent of the code commit
    `dc9d28c`;
  - R14 ran in `SecondaryGWKit-lpf06-f5cee4c`, detached at Part B's dispatch SHA `f5cee4c`. Its code
    is `dc9d28c`'s; `dc9d28c..f5cee4c` touches only `prompts/`.
- **The main checkout** stayed at `28ebaf0` and clean throughout. It holds the venv, the stores,
  the registry and these records.
- **The software.** Python 3.13.16 from the main checkout's venv: `ray` 2.59.0, `SQLAlchemy`
  2.0.55, `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit` 0.2.0, `numpy` 2.2.4 and
  `scipy` 1.16.3.
- **The load** was about 5–10 on 10 cores. Prompt §6's Ray-free re-measurement ran beside these
  builds.

**Records** are in [`06/rehearsal/`](06/rehearsal/): `out/`, `scripts/` and `state/`, the agent's
`log.md`, this section as first written, and `issues.md`.
- Every command is in [`out/commands.log`](06/rehearsal/out/commands.log).
- The scripts' SHA-256 are in [`out/script_sha256.txt`](06/rehearsal/out/script_sha256.txt):
  - `classify_non_qsi.py` (`26690e61…`) and `tabulate_qsi.py` (`b7508351…`) are 05's, copied
    unchanged and byte-identical (`cmp`);
  - `run_readers.sh` (`7c63a5ba…`) and the launcher (`5b984c75…`) equal 05's;
  - `tools/store_value_diff.py` is `7997af85…` in the code worktree. That is the code commit's
    §1.5 extension. The parent worktree and the main checkout still hold 05's `5c92d8e3…`.
- The readers' `extract/` (176 MB) was not copied back, and neither was the 1.09 GB
  `out/store_value_diff.json` (see the log's observations).

#### The two builds

| | R13 (before) | R14 (after) |
|---|---|---|
| tree | worktree at `9f66622` (parent of `dc9d28c`) | worktree at `f5cee4c` (code `dc9d28c`) |
| how it ran | cwd and `PYTHONPATH` the worktree; the main checkout's venv python by absolute path; `PYTHONDONTWRITEBYTECODE=1` | the same, in its own worktree |
| run | `levin-phase-floor-06-r13-parent-20261026T215550` | `levin-phase-floor-06-r14-residual-20261026T215945` |
| `sh` / driver pid | 97736 / 97740 | 99934 / 99938 |
| wall | 21:55:50 → 21:58:44 (heartbeat) | 21:59:45 → 22:02:40 (heartbeat) |
| manifest | `git_head` `9f66622d…`, `git_dirty` false, cwd the parent worktree | `git_head` `f5cee4c7…`, `git_dirty` false, cwd the code worktree |
| cwd of the driver and of the `Datastore` actor (`lsof`) | the parent worktree (97740, 97834) | the code worktree (99938, 133) |
| end | `done`, exit 0; no Traceback; `stderr.log` empty | `done`, exit 0; no Traceback; `stderr.log` empty |
| store | `var/datastores/levin-phase-floor-06-r13.sqlite` + 4 shards | `var/datastores/levin-phase-floor-06-r14.sqlite` + 4 shards |

**The scope.** Both builds were R8's rehearsal scope, launched as 05a launched R11:
- 3 k over 1e5–1e6/Mpc, zend 250;
- atol 1e-32, rtol 1e-7, 6 CPUs;
- `--job-name levin-phase-floor-06` and `--campaign levin-phase-floor --prompt 06`.

**The worktrees were not written.** After both builds, after the readers, and at the end, each
worktree's `git status --short --ignored` was empty, and `find -newer` the pre-R13 marker listed
nothing in either ([`out/r13_end_check.txt`](06/rehearsal/out/r13_end_check.txt),
[`out/r14_end_check.txt`](06/rehearsal/out/r14_end_check.txt),
[`out/final_check.txt`](06/rehearsal/out/final_check.txt)).

#### The repairs' agreement, and `_check_phase_sign` (prompt §0.2, §8; dispatch §B.5)

**R14 printed no `!! WARNING (GkSourcePolicyData` line.** It printed no other `!! WARNING` line
either, and `_check_phase_sign` did not raise.
- The count is 0 in R14's launch log, `stdout.log` and `stderr.log`, for each of these patterns:
  `!! WARNING (GkSourcePolicyData`, any `!! WARNING`, `_check_phase_sign`, `Error`, `Exception`
  and `Traceback`.
- **Ray's own worker logs.** Only the `Datastore` actor's output reaches the launch log, so the
  same patterns were also searched in Ray's worker logs for both sessions: 12 `worker-*.out` and 12
  `worker-*.err` each, under `/tmp/ray-sgk/session_*_97740` and `…_99938`. Every count is 0
  ([`out/ray_worker_logs_check.txt`](06/rehearsal/out/ray_worker_logs_check.txt)).
- Those logs record the same tasks in both builds: 291 `apply_GkSource_policy`, 291
  `assemble_GkSource_values`, 3480 `WKB_phase_function`, 1371 `numeric_with_phase_cut`, 864
  `compute_QuadSource_integral`, 6 `compute_quad_source`, 3 `find_horizon_exit_time` and 1
  `compute_background`.

#### The lock

[`out/lock.log`](06/rehearsal/out/lock.log). Each lock command was a shell command of its own, with
its output appended by `>>`.

| # | command | output | exit |
|---|---|---|---|
| 1 | `acquire --project SecondaryGWKit --purpose "levin-phase-floor prompt 06 section 5, rehearsals R13 … and R14 …" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 2 | `set-pid … 97740` (R13) | recorded | 0 |
| 3 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after R13) | nothing | 0 |
| 4 | `set-pid … 99938` (R14) | recorded | 0 |
| 5 | `ray-processes` (after R14) | nothing | 0 |
| 6 | `ray-processes` (after the readers) | nothing | 0 |
| 7 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 8 | `status --ray-temp-dir /tmp/ray-sgk` | free; no Ray processes | 0 |
| 9 | `ray-processes --ray-temp-dir /tmp/ray-sgk` | nothing | 0 |

#### The fingerprints (prompt §5.2, §5.3)

| | digest | problems | per-class |
|---|---|---|---|
| R8 (reference, `handover-12-r8-reference-20261025T210551`) | `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` | none | 21 classes |
| **R13** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 digests equal R8's |
| **R14** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | all 21 equal R8's; the whole `classes` object equals R13's |

The fingerprints are the ones the launcher took at the end of each build, read from `status.json`
(fingerprint format 6, encoding format 1;
[`out/r13_status.json`](06/rehearsal/out/r13_status.json),
[`out/r14_status.json`](06/rehearsal/out/r14_status.json)). There is no sidecar, because `--build`
writes none. As §5.3 says, the added column does not enter the fingerprint.

#### The value comparison (prompt §5.4)

The comparison was run from the code worktree, so that `-m` imports that tree's tool:

```
cd <code worktree> && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<code worktree> <main>/venv/bin/python -B -m tools.store_value_diff <main>/var/datastores/levin-phase-floor-06-r13.sqlite <main>/var/datastores/levin-phase-floor-06-r14.sqlite --json
```

It exited **1**, as the addendum expects, in 29 s, and wrote 1.09 GB of JSON
(`out/store_value_diff.json`, SHA-256 `7e42eadc…`).
- The summary is [`out/store_value_diff_summary.txt`](06/rehearsal/out/store_value_diff_summary.txt),
  from [`scripts/summarise_diff.py`](06/rehearsal/scripts/summarise_diff.py).
- The exit status is in [`out/store_value_diff.meta.txt`](06/rehearsal/out/store_value_diff.meta.txt).
- R13 was read without `theta_residual`. The tool reports that column under `columns_only_in_b` in
  `GkWKBValue`, `TkWKBValue` and `GkSourceValue`, and in no other table.

**Structure.** 21 classes were compared, and `not_compared` is empty. In every class:
- 0 rows are in one store only;
- 0 rows cannot be matched by identity;
- 0 columns are unresolvable.

The value tables:

| value table | class | rows identical | differ | only in one / unmatched | column in one store only |
|---|---|---|---|---|---|
| BackgroundModelValue | outside | 1159 | 0 | 0 / 0 | — |
| TkNumericValue | outside | 1478 | 0 | 0 / 0 | — |
| GkNumericValue | outside | 35874 | 0 | 0 / 0 | — |
| QuadSourceValue | outside | 2447 | 0 | 0 / 0 | — |
| TkWKBValue | **inside** | 47 | 2162 | 0 / 0 | `theta_residual` (R14) |
| GkWKBValue | **inside** | 1391 | 146251 | 0 / 0 | `theta_residual` (R14) |
| GkSourceValue | **inside** | 23402 | 146251 | 0 / 0 | `theta_residual` (R14) |

There are 360,462 value rows in all. **R13 and R14 are not value-identical in `GkWKBIntegration`'s
values**, so the dispatch's stop does not fire: each tree ran its own code.

##### Outside the five classes

- **Every value table is bit-identical:** `BackgroundModelValue`, `TkNumericValue`,
  `GkNumericValue` and `QuadSourceValue`, 40,958 rows.
- **10 classes are value-identical:** `version`, `store_tag`, `redshift`, `wavenumber`,
  `tolerance`, `LambdaCDM`, `QCD_Cosmology`, `IntegrationSolver`, `GkSourcePolicy` and
  `QuadSourcePolicy`. `OneLoopIntegral` has no rows.
- **Five classes differ in row columns only.** 05's
  [`scripts/classify_non_qsi.py`](06/rehearsal/scripts/classify_non_qsi.py), copied unchanged,
  sorts each of their differences into one of the volatile kinds
  ([`out/classify_non_qsi.txt`](06/rehearsal/out/classify_non_qsi.txt)). **0 are unclassified:**

  | class | rows | columns (kind) |
  |---|---|---|
  | wavenumber_exit_time | 3 | `compute_time` (timing) |
  | BackgroundModel | 1 | `compute_time`, `{min,mean,max}_RHS_time` (timing) |
  | TkNumericIntegration | 3 | `compute_time`, `*_RHS_time` (timing); `label` |
  | GkNumericIntegration | 1368 | `compute_time`, `*_RHS_time` (timing; `min_RHS_time` in 1047); `label` |
  | QuadSource | 6 | `compute_time` (timing); `label` |

  Nothing outside the five classes differs in any other way, so prompt §8's stop does not fire.
- **The script exits 1, reporting 185 problems.** All 185 are inside the five classes, because it
  exempts only `QuadSourceIntegral` (log, deviation 6):
  - 3 are the value tables of `TkWKBIntegration`, `GkWKBIntegration` and `GkSource`;
  - 182 are `GkSourcePolicyData` rows whose `metadata` differs.
- **Its row-column kinds inside the five classes** are the same volatile kinds as before:
  - `TkWKBIntegration`: 3 `label` and 3 `stage_1_compute_time`;
  - `GkWKBIntegration`: 3477 `label`, 3474 `stage_1_compute_time`, and 25 rows with the
    residual-memo counters `rho_evals`, `rho_reused` and `stage_1_RHS_evaluations`;
  - `GkSource`: 291 `label` and 291 `parents_digest`.

##### Inside the five classes, tabulated per class

The new script is [`scripts/tabulate_phase_classes.py`](06/rehearsal/scripts/tabulate_phase_classes.py)
(`81effdbb…`).
- **How it reads.** It pairs rows with the code worktree's own `tools.store_value_diff.read_store`,
  which is the tool's identity matching, and so has every column of both sides.
- **The cross-check.** Its differing-row counts per column equal the tool's JSON for all three
  value tables.
- **Outputs:** [`out/phase_classes_tabulation.txt`](06/rehearsal/out/phase_classes_tabulation.txt),
  with one row per value row in `out/phase_classes_rows.csv`.
- **The outliers** come from [`scripts/examine_outliers.py`](06/rehearsal/scripts/examine_outliers.py),
  in [`out/examine_outliers.txt`](06/rehearsal/out/examine_outliers.txt).
- **Notation:** $\varepsilon = 2^{-52}$, and $|\theta| = |{\rm div}\cdot2\pi + {\rm mod}|$ from
  R14's stored cycle count and remainder.

**`theta_residual`, present in R14 only:**
- `GkWKBValue`: non-null in all 147,642 rows; max $|r|$ 1.571.
- `TkWKBValue`: non-null in all 2,209 rows; max $|r|$ 1.764.
- `GkSourceValue`: non-null in all 147,642 rows with a WKB phase; max $|r|$ 1.571.
- In no table is $|r|$ greater than $\pi + 2\times10^{-3}$.
- In every table, R14's `theta_mod_2pi` lies in $(-2\pi, 0]$.

**`theta_mod_2pi`: $|\Delta|$ against $\varepsilon|\theta|$**, over the rows whose cycle count did
not change, which is every row:

| table | rows | bit-identical | $\lvert\theta\rvert$ median / max | $\lvert\Delta\rvert$ max (rad) | $\lvert\Delta\rvert/\varepsilon\lvert\theta\rvert$ median / 90 % / 99 % / max | > 1 | > 4 |
|---|---|---|---|---|---|---|---|
| GkWKBValue | 147642 | 1391 | 7.65e5 / 7.67e8 | 2.28e-7 | 0.324 / 0.790 / 1.26 / 272 | 5442 | 52 |
| TkWKBValue | 2209 | 47 | 7.01e4 / 2.11e8 | 5.29e-8 | 0.395 / 0.877 / 1.43 / 8.41 | 120 | 2 |
| GkSourceValue | 147642 | 1391 | 7.65e5 / 7.67e8 | 2.28e-7 | 0.309 / 0.766 / 1.19 / 7.18 | 4553 | 2 |

- **All 54 producer rows above 4** (52 $G$, 2 $T$) have $|\theta| < 1$ rad, and
  $|\Delta| \le 8.9\times10^{-16}$, which is one ulp of $2\pi$.
- **Where they lie.** These are samples close to the producer's own anchor. There $\varepsilon
  |\theta|$ is smaller than the granularity of the remainder itself, and smaller than the error of
  the old cancelling sum $k\,L.\delta$, whose terms are of size $k\tau$.
- **Against `GkSource`'s rectified cycle count.** The same remainders sit at $|\theta|$ of 0.09–26
  rad there. Against that, every ratio is at most 7.2.
- This is recorded, not judged (log, observations).

**`theta_div_2pi` changes: none.**
- There are 0 in `GkWKBValue`, 0 in `TkWKBValue` and 0 in `GkSourceValue`, and 0 changes of
  `GkSourceValue.raw_theta_div_2pi`.
- So no change needs to be tested against the $4\varepsilon|\theta|$ bound of §5.4 (§9 point 3),
  and none is listed on its own.

**`G_WKB` and `T_WKB`.** Here $A = \sqrt{H_{\rm ratio}/\omega}\,{\rm hypot}(s, c)$, times
$e^{\rm friction}$ for $T$, is the Liouville–Green amplitude.

| column | rows differ | $\lvert\Delta\rvert/\lvert a\rvert$ median / 90 % / 99 % / max | $\lvert\Delta\rvert/(A\,\lvert\Delta\theta_{\rm mod}\rvert)$ median / 90 % / max |
|---|---|---|---|
| GkWKBValue.G_WKB | 145772 | 3.7e-11 / 1.1e-8 / 2.0e-7 / 3.6e-4 | 0.70 / 0.99 / 2.73 |
| TkWKBValue.T_WKB | 2154 | 5.0e-12 / 3.2e-9 / 8.0e-8 / 7.7e-6 | 0.72 / 0.99 / 1.08 |
| GkSourceValue.G_WKB | 145772 | as GkWKBValue | as GkWKBValue |
| GkSourceValue.new_G_WKB | 145772 | as GkWKBValue | as GkWKBValue |

- The largest relative changes are where $G$ is near a zero of its oscillation.
- **Exceeding $A|\Delta\theta_{\rm mod}|$.** 615 `GkWKBValue` rows and 14 `TkWKBValue` rows exceed
  it. Every one is within $A|\Delta\theta| + 4\varepsilon A$: that excess is the rounding of $G$
  itself, and most of these rows have $|\Delta\theta| \le 10^{-14}$. So every change to `G_WKB`
  and `T_WKB` is accounted for by the change of the remainder.
- No other value column of the three tables differs.

**`GkSourcePolicyData`'s changed fields:**
- `Levin_z`, `crossover_z`, `type` and `quality` are unchanged in all 291 rows.
- Only `metadata` differs, in 182 rows, and only in its key `Levin_z_dtheta_dlogz`:
  - $|\Delta|$ has median 3.0e-10 and max 4.0e-6;
  - $|\Delta|/|a|$ has median 2.0e-10 and max 2.6e-6;
  - no value goes to or from null.

##### `QuadSourceIntegral`, tabulated per row

05's [`scripts/tabulate_qsi.py`](06/rehearsal/scripts/tabulate_qsi.py), copied unchanged, reads both
stores (`mode=ro`) and the tool's JSON. Its docstring still names R9 and R10; **here A = R13
(parent `9f66622`) and B = R14 (code `dc9d28c`)**.
- Its output is [`out/qsi_tabulation.txt`](06/rehearsal/out/qsi_tabulation.txt), with one row per
  integral in `out/qsi_rows.csv`.
- Further counts are in [`out/qsi_extra_counts.txt`](06/rehearsal/out/qsi_extra_counts.txt).

**The pairing.** Rows were paired by physical (k, q, r, z_response): 864 = 864, with one atol and
one rtol, (−32, −7). It has the same key set as the tool's, with 0 mismatches in `total`'s
`float.hex` against the tool.

| | rows |
|---|---|
| `total` differs | **554** of 864 (310 bit-identical) |
| `WKB_Levin` differs | 568. Every row whose `total` differs is among them |
| `numeric_quad` / `analytic_rad` differ | **0** / **0** |
| rows with no `WKB_Levin` part (`max_depth` None) | 291, none of which differs in `total` |

Over the 554 rows whose `total` differs:

| statistic | median | 90 % | max |
|---|---|---|---|
| $\lvert\Delta{\rm total}\rvert/\lvert{\rm total}_A\rvert$ | 2.79e-11 | 1.37e-08 | 1.57e-03 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ | 1.90e-04 | 9.82e-02 | 1.82e+04 |
| $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_B$ | 1.90e-04 | 9.58e-02 | 1.28e+04 |

- **Above `total_abserr`.** 4 rows exceed $\lvert\Delta{\rm total}\rvert/$`total_abserr`$_A$ > 1,
  and 2 exceed it against `total_abserr`$_B$. 55 rows exceed 0.1 against $A$, and 54 against $B$.
- **The two largest are at $z_{\rm resp} = 250$, the end of the range:**

  | $(k, q, r)$ | rel. change | $/$`total_abserr`$_A$ | depth | regions |
  |---|---|---|---|---|
  | $(3.162\times10^5, 3.162\times10^5, 3.162\times10^5)$ | 1.6e-3 | 1.8e4 | 12 → 6 | 31 → 14 |
  | $(10^5, 3.162\times10^5, 3.162\times10^5)$ | 2.8e-5 | 484 | 14 → 6 | 30 → 15 |

  Both rows have `total_converged` 0 in both stores. The other two rows above 1 are at
  $k = q = r = 10^6$: $z_{\rm resp}$ 755.7 at 1.25, and 996.4 at 1.02. Both are below 1 against
  `total_abserr`$_B$.
- **`WKB_Levin_max_depth`:** 3 rows change (12 → 6, 14 → 6 and 4 → 3).
- **`WKB_Levin_num_regions`:** 59 rows change, and the total falls from 13,652 to 13,495.
- **`total_converged`:** no row changes (0→0: 355, 1→1: 509).
- **`total_phase_limited`:** 13 rows change (4 0→1, 9 1→0).
- **By depth under A,** the median $\lvert\Delta\rvert/$`total_abserr`$_A$ is:
  - 1e-5 to 2.4e-4 at depths 5–8;
  - 0.04 to 0.13 at depths 10–12.
- **`total_abserr` = inf in both stores:** 26 rows. 9 of their `total`s differ. This is the open
  issue `[04-roundoff-floor-can-be-infinite]`.
- **Metadata.** 60 distinct non-wall-clock keys differ, and every row has at least one. They form
  two families:
  - `partition.handover.T{q,r}.handover_continuity[i].d{T,Tprime}_rel`, from $T$'s changed
    remainder, as in 05a;
  - the `WKB_Levin` `value`/`abserr` tree.

These are recorded, not judged. Prompt §6 scores the effect on A3 v2's rows.

#### The readers (prompt §5.5)

**Why they did not run from `run_readers.sh` itself.** The script changes directory to its own
tree and calls `./venv/bin/python`. The code worktree has no venv, and the main checkout's copy
would run the main checkout's code (`28ebaf0`), not the code commit's.
- So they ran through [`scripts/run_readers_worktree.sh`](06/rehearsal/scripts/run_readers_worktree.sh),
  a copy changed in three lines only ([`out/run_readers_worktree.diff`](06/rehearsal/out/run_readers_worktree.diff)).
- Each reader's cwd was the code worktree, run with the main venv's python, `PYTHONPATH=.` and
  `PYTHONDONTWRITEBYTECODE=1`.
- The lock was held, from 22:12:15 to 22:14:35 UTC
  ([`out/readers.txt`](06/rehearsal/out/readers.txt)). Overall exit **0**.

| reader | exit | wall | lookups | files written (05a's R12) | store files after |
|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 25 s | 116 | 396 (387) | identical |
| `extract_GkWKB_data.py` | 0 | 47 s | 348 | 2436 (2436) | identical |
| `extract_GkSource_data.py` | 0 | 19 s | 10 | 176 (162) | identical |
| `extract_TkWKB_data.py` | 0 | 14 s | 3 | 27 (27) | identical |
| `extract_tensor_source_data.py` | 0 | 14 s | 6 | 27 (27) | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 16 s | 24 | 70 (73) | identical |

- The lookups equal 05's and 05a's.
- File counts differ only where a reader draws an unseeded subsample
  (`[10-the-readers-draw-their-work-lists-unseeded]`).
- After each reader, 0 Ray processes of the venv were left.
- No reader log has a Traceback. Each has one line `!! WARNING: number of shards read from
  database (=N) does not match specified number of shards (=N)`, the readers' own note on the shard
  count.

**R14's store, before and after the readers.** The `diff` is empty, and there is no `-journal`,
`-wal` or `-shm` file
([`state/r14_store_sha256_before_reads.txt`](06/rehearsal/state/r14_store_sha256_before_reads.txt),
[`state/r14_store_sha256_after_reads.txt`](06/rehearsal/state/r14_store_sha256_after_reads.txt)):

```
1c0f3cf63cc95208c53e558d152255f34fa6b17c9b82f9270a2b43f2f126f7da  levin-phase-floor-06-r14-shard0000.sqlite  25038848
866511b7ef81f27da9bcf5306bd9a6efa9c367475f3c7959a23a78d07c943fcd  levin-phase-floor-06-r14-shard0001.sqlite  25255936
dfaad9499d3d7f7364532cd5cb8091b1f32a51d1fed0a1a43f10213fdaff656b  levin-phase-floor-06-r14-shard0002.sqlite   1077248
410bf34b370512c5426428f1c6f3c6ea6008e3bd9b0d371f60e08bbce9d12a57  levin-phase-floor-06-r14-shard0003.sqlite  24485888
fff6cf7fac9106897efba04547710ec99ce051f4f2457108dc8cdb6fd9689acd  levin-phase-floor-06-r14.sqlite              36864
```

**Both stores, all ten files, are byte-identical from before the comparison to the end**
([`state/store_sha256_before_diff.txt`](06/rehearsal/state/store_sha256_before_diff.txt),
[`state/stores_sha256_final.txt`](06/rehearsal/state/stores_sha256_final.txt)). R13's files equal
the hashes taken just after its build
([`state/r13_store_sha256_after_build.txt`](06/rehearsal/state/r13_store_sha256_after_build.txt)).
R13's primary is `766839d657e7f943f2e3ef1f3058268544b42ffcfeb1d8f753a68d9c768989c9`.

#### Against prompt §7.2

- **R13 and R14 both end `done` with fingerprint `1b332ffa…`: yes.** Each has 21/21 class digests
  equal to R8's.
- **Outside the classes of §0.3, every value table is identical and every other difference is of
  the volatile kinds: yes.**
  - The 4 value tables there are bit-identical: 40,958 rows.
  - 0 differences there are unclassified.
  - 0 rows are in one store only, and no column is in one store only.
- **Inside them, the differences are tabulated: yes.**
  - `theta_residual` is in R14 only, and non-null wherever a remainder is stored.
  - `theta_mod_2pi` moved in 146,251 $G$ and 2,162 $T$ rows. The 99th percentile of the move is
    1.2–1.4 $\varepsilon|\theta|$.
  - `theta_div_2pi`: no change.
  - `G_WKB` and `T_WKB` moved by the remainder's change only.
  - `GkSourcePolicyData`: only `Levin_z_dtheta_dlogz`, by at most 2.6e-6 relative.
  - `QuadSourceIntegral`: 554/864 `total`s moved, two by $10^2$–$10^4$ × `total_abserr`$_A$, with
    their depth halved.
- **The repairs agree in R14:** 0 `!! WARNING (GkSourcePolicyData` lines, and
  `_check_phase_sign` did not raise.
- **The readers exit 0: yes,** and the store is byte-identical.
- **The lock is released: yes.**

### 6.2 The re-measurement on A3 v2, with the producers re-run in-process (prompt 06 §6)

Run `levin-phase-floor-06-wider-20261026T223634` on the Mac Studio (mps026298), Ray-free, no lock,
22:36:34Z–23:23:16Z: `done`, 109/109, exit 0, no unit over budget, no error. Two trees, both
worktrees: **t5** = `9f66622` (05a's t; its consumers and A3 v2's stored producer samples) and
**s** = `f5cee4c` (code `dc9d28c`; its producers re-run on every `GkWKBIntegration` and
`TkWKBIntegration` each row uses, the `GkSource` re-assembled and the `GkSourcePolicyData`
recomputed from them, and its consumers). Records: [`06/wider/`](06/wider/).

**Route (B), the user's decision.** `s` cannot open A3 v2 (its factories declare `theta_residual`,
which A3 v2 lacks: [`06/wider/out/cmd/06_dev_smoke_s.txt`](06/wider/out/cmd/06_dev_smoke_s.txt)). It
read, read-only, an unregistered copy of A3 v2 with the three nullable columns added, all NULL:
every one of 3,411,578 rows identical by bit pattern, schema differing only by those columns
([`06/wider/out/cmd/12_certify_copy_rerun.txt`](06/wider/out/cmd/12_certify_copy_rerun.txt)),
fingerprint `0e32f24a…`, problems none
([`06/wider/out/cmd/15_fingerprint_copy_at_s_rerun.txt`](06/wider/out/cmd/15_fingerprint_copy_at_s_rerun.txt)).
t5 read A3 v2 itself.

**Gate U0** ([`06/wider/out/U0_a.txt`](06/wider/out/U0_a.txt), [`06/wider/out/U0.txt`](06/wider/out/U0.txt)): holds.

| Clause | Result | |
|---|---|---|
| (a) | `git diff --stat 1fc349c 9f66622 -- . ':!docs' ':!prompts'`: `CLAUDE.md`, `primitive_phase.py`, `test_g_phase_reanchor.py`; both ASTs without docstrings equal `1fc349c`'s | holds |
| (b) | t5@1e-7 on `low-G+r-1`, `ctrl-low-1` against 05a's `t-1e-7-*`: total, abserr, every call, the whole result summary and region arrays bit-identical | holds |
| (c) | producers re-run at the parent on A3 v2 (3,156 G objects, 241,146 values; 4 T objects, 5,172 values): `theta_div_2pi`, `theta_mod_2pi`, `G_WKB`, `T_WKB` and every coefficient bit-identical; beyond the clause the re-assembled GkSource, the recomputed GkSourcePolicyData and each unit's result and regions reproduce the stored-input unit bit for bit | holds |
| (d) | two in-process `s` runs per row: result, calls, regions, the re-run's record, the policy, the probe and every `deltaTheta` bit-identical | holds |

**Predictions** ([`06/wider/state/predictions.md`](06/wider/state/predictions.md); §6 as amended,
quoted unchanged, written after U0 and before the launch; the pure-WKB tag read two ways in a
section added before the launch, the user ruling for (I) with no S1 number in existence):

| | Number | Verdict |
|---|---|---|
| **S1**, reading (I) of record (`z_init == z_source`) | 6,700 pure-WKB G windows on R1–R7, all matched, true in 6,700. `sigma_t5` is exactly 0 in all 6,700 (production's `phi_G` quantised to zero); `sigma_s` ≤ 8.8e-14 F (median 5.7e-16 F); largest `sigma_s`/bound 8.8e-8 | **holds** |
| (S1, reading (L), unscored) | no window has a null `z_init` (A3 v2 stores none) | (undecided) |
| **S2** | 12,862 T windows, all matched; true in 10,747, **false in 2,115**, all at z ≥ 1.75e7 with `\|theta_T\|` 0.05–1.5e3 rad, where `sigma_s/sigma_t5` = 0.952–1.04 and `sigma` = 5–1.5e7 `eps\|theta\|`: `phi_T`'s own structure, not storage noise. Elsewhere `sigma_s/F` median 7e-10 | **fails** |
| **S3** | 39 of 39; largest depth under s@1e-7 is 11 (11 on the rows capped under B1) | **holds** |
| **S4** | 34 of 39; false on `low-G-3` 10.8, `ctrl-low-1` 3.90, `ctrl-low-2` 3.57, `mid-G-2` 1.62, `R2` 1.49 (× `total_abserr_t5`); median ratio 0.29, median relative change 1.3e-6; bit-identical on the two NULL controls | **fails** |
| **S5** | 28 of 31, none excluded; false on `mid-G-3` 1.57, `mid-G-r-3` 1.32, `mid-G-1` 1.31; true on all 11 capped under B1 (largest 0.823). 05a's T4 under t5: 29 of 31 | **fails** |

**Numerically seeded G windows (not scored):** 3,594; `sigma_s` 2.1e-4 F at the median (max 1.2e-3
F) against `sigma_t5` 0.27 F (max 0.78 F); the re-run `deltaTheta`'s window RMS is 2.8e-5 F at the
median, about 1/8 of `sigma_s`.

**Descriptive:**
- No unit over budget; largest s wall/budget 0.715 (`s-1e-8-R2`). No `!! WARNING
  (GkSourcePolicyData` line in any unit.
- Regions 901 (t5) → 742 (s@1e-7), 1,053 at 1e-8; R1–R7 250 → 200; the 11 capped 426 → 336. 48 of
  219 group calls change regions, depth or convergence. Unconverged: 21 → 20 at 1e-7, 19 at 1e-8.
- `theta_deriv` (log) at the window centres: T relative change ≤ 4.6e-14; G median 3.7e-14, up to
  4.9 in seeded windows.
- `Levin_z` changes on one row (`low-G+r-1`); type, quality and crossover on none.
- `convergence_kinds.py` under s ([`06/wider/out/convergence_kinds_s.txt`](06/wider/out/convergence_kinds_s.txt)):
  1e-7 resolution 14, bound only 32, phase-limited 23; 1e-8 20, 25, 32 (05a's t: 18, 32, 17 and 21, 24, 27).
- The re-run values against A3 v2: 17 G cycle changes, each one cycle at a remainder within
  `4 eps|theta|` of a boundary; none for T.

**The stores.** A3 v2's six files: SHA-256, size and mtime identical before, before the launch,
after the run and after the fingerprint; no journal, `-wal` or `-shm`; fingerprint `0e32f24a…`,
problems none, matching the one taken at `b90752d`
([`06/wider/out/cmd/36_fingerprint_A3v2.txt`](06/wider/out/cmd/36_fingerprint_A3v2.txt)). The copy's
five files are identical from certification to the end, and its fingerprint is again `0e32f24a…`.

### 6.3 P1: a stationary point inside one Levin region

**Investigated 2026-10-27** by a laptop subagent, on the Mac Studio, Ray-free and read-only on the
R13/R14 stores (their SHA-256 unchanged). Report: [`06/p1/report.md`](06/p1/report.md).

- **The rows.** $(k, q, r) = (3.162\times10^5)^3$ and $(10^5, 3.162\times10^5, 3.162\times10^5)$
  /Mpc at $z_{\rm response} = 250$. Each tree rebuilt both from its own store and reproduced every
  stored value bit for bit (54 items per row).
- **Where they differ.** Only the group G+q+r on $z \in [250, 5.42\times10^9]$; every other group
  agrees to about $10^{-8}$. R13 took it in 18 and 16 regions, converged; R14 accepted it as a
  single region of 3 evaluations.
- **Why.** The group's phase has one interior stationary point: $z_* = 10211.3$, where
  $c_s = 1/2$, and $z_* = 275.0$, where $2qc_s = k$. A single Levin region misses that point's
  contribution, and so do its two halves, so the step-(4) residual cannot see it. R14's estimate
  misses by exactly the leading stationary-phase term, to $10^{-5}$ and $5\times10^{-5}$. R13's
  noisier phase lifted the root's residual over rtol, by factors 2.05 and 1.008, through the
  unrelated high-$z$ half; the bisection then found the point.
- **The reference.** Forced subdivision (up to $2^{12}$ pieces) and a Gauss–Legendre window around
  $z_*$ with Levin outside agree, good to about $10^{-12}$ relative, in both trees.
  - **R13's totals are right:** within 0.0095 and 0.0089 of their declared error.
  - **R14's are wrong:** $1.28\times10^4$ and 555 times their declared error away (the group by
    $6.8\times10^{-3}$ and $1.2\times10^{-3}$).
  - Under R14 row 1 stays wrong at rtol $10^{-9}$ and $10^{-11}$: the root is then accepted as
    round-off-limited.
- **06's integrand is the more accurate.** The amplitudes are bit-identical in both trees, and the
  phases differ by 1–2 ulp of $k\tau$, R13 being the noisy side. Integrated correctly, the trees
  agree to $7\times10^{-9}$ and $4\times10^{-11}$ relative.

**What it means.** `AdaptiveLevin`'s error control is not valid for a Levin region that contains an
interior stationary point. The C2 fix gates only regions whose total phase variation is below
$6\pi$, and its regression case puts the stationary point at the first bisection point. In this
pipeline almost every resonant triangle has such a point; it is the Kohri–Terada resonance.
Opened as `[06-levin-accepts-a-region-across-a-stationary-point]`, for G2's layer 3.

## 6a. The depth census

*Prompt 06a, on the Mac Studio, 2026-10-27, Ray-free, no lock. The records are under
[`06a/`](06a/). The pickles, the run directory and every file over 2 MB stay on the Mac Studio.*

**What was measured.** Every `QuadSourceIntegral` row of A3 v2, 9,280 of them, was re-run
in-process at the current code, at `rtol = 1e-7` and at `1e-8` (`atol` 1e-32), and each Levin
call's depth was recorded.
- **The tree:** $h$, the worktree at the dispatch SHA `e05c34f`. Its code outside `docs/` and
  `prompts/` is 06's $s$ tree's (`f5cee4c`), byte for byte.
- **The store:** 06's certified scratch copy of A3 v2, opened read-only.
- **The method:** 06's harness, with each producer re-run once per worker instead of once per row
  ([`06a/scripts/`](06a/scripts/); diff against 06's in
  [`06a/out/scripts_diff_against_06_final.txt`](06a/out/scripts_diff_against_06_final.txt)).
- **The run:** `levin-phase-floor-06a-census-20261027T023143`, 02:31:43Z–03:17:10Z (45.5 min). It
  ended `done`, 18,560 of 18,560 units, exit 0. No unit went over budget (300 s) or errored, and
  `stderr.log` is empty. The run directory is 220 MB.

### 6a.1 Gate C0 (before the launch): holds

| | | Evidence |
|---|---|---|
| (a) | The code is 06's: `git diff --stat f5cee4c e05c34f -- . ':!docs' ':!prompts'` shows `CLAUDE.md` alone | [`out/cmd/07`](06a/out/cmd/07_gate_diff_stat.txt) |
| (b) | The copy's five files equal `copy_files_certified.txt`; `fingerprint_copy.py` at $h$ gives `0e32f24a…` with no problems | [`02`](06a/out/cmd/02_copy_vs_certified.txt), [`15`](06a/out/cmd/15_C0b_fingerprint_copy_at_h.txt) |
| (c) | Census units bit-identical to 06's `s` units: `low-G+r-1` at 1e-7 and at 1e-8, `ctrl-low-1` at 1e-7 (total, total_abserr, every call's value and abserr, the region arrays) | [`18`](06a/out/cmd/18_C0_check_rerun.txt) |
| (d) | `low-G+r-1` run first in a fresh worker, and run last after the seven other rows of its block, is identical in every computed value; so are the seven | [`18`](06a/out/cmd/18_C0_check_rerun.txt) |
| (e) | No session directory appeared under `/tmp/ray-sgk` (59 entries before and after, here and for the run); no worker initialised Ray | [`18`](06a/out/cmd/18_C0_check_rerun.txt), [`44`](06a/out/cmd/44_final_state.txt) |

**The pilot** ([`19`](06a/out/cmd/19_pilot_run.txt), [`20`](06a/out/cmd/20_pilot_projection.txt)):
three blocks of k = 9.70e6/Mpc, at both rtols, in one worker. It projected 0.84 h of wall time and
266 MB. The run took 0.76 h and 220 MB.

**The predictions** were written to [`state/predictions.md`](06a/state/predictions.md):
sha256 `3ad9b43d…`, mtime 02:28:40Z, byte-equal to the prompt's §4, before the launch at 02:31:43Z.

### 6a.2 Predictions and verdicts

| | Rule (prompt §4) | Number | Verdict |
|---|---|---|---|
| **C1** | at `1e-7`, no row has a region at depth ≥ 18 | 0 of 9,280 rows; largest row depth 12; no unit over budget, none errored | **holds** |
| **C2** | the same at `1e-8` | 0 of 9,280; largest row depth 13; none over budget, none errored | **holds** |
| **C3** | at each rtol, no row's depth exceeds 11 | 19 rows at 1e-7 (all at depth 12); 29 at 1e-8 (28 at 12, one at 13) | **fails** |
| **C4** | 05's 39 rows at `1e-7`, and its 31 depth-20 rows at `1e-8`, bit-identical to 06's `s` units | 70 of 70 | **holds** |

C3's failure is characterised in [`06a/issues.md`](06a/issues.md) X1:
- every one of the 48 deeper rows holds an interior stationary point;
- their regions at depth ≥ 11 lie within 1–3.5 grid spacings of it;
- every region at depth ≥ 12 is in a `Gqr` call.

### 6a.3 Descriptive (§5), from [`out/census_summary.txt`](06a/out/census_summary.txt)

1. **Row depth.** 2,489 rows have no Levin call.

   | depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
   | 1e-7 | 291 | 530 | 1,786 | 654 | 99 | 96 | 511 | 2,010 | 732 | 29 | 3 | 31 | 19 | 0 |
   | 1e-8 | 284 | 516 | 1,790 | 567 | 69 | 91 | 396 | 529 | 1,565 | 923 | 8 | 24 | 28 | 1 |

   - The 2,265 rows stored at depth 20 fall to depths 1–12 at 1e-7; 1,248 of them are at depth 7.
   - The largest depth by band is low 12, mid 12, seam 9 at 1e-7; low 12, mid 13, seam 10 at 1e-8.
2. **Depth ≥ 12 by branch** (Levin / Clenshaw–Curtis): 13 / 25 at 1e-7; 17 / 42, plus 0 / 2 at
   depth 13, at 1e-8. All are in `Gqr` calls.
3. **The deepest rows:** listed in the summary with every call. They are mostly equilateral
   q = r = k at k = 1e5 and 3.14e5/Mpc, mid band, class `Gqr`.
4. **Regions.** 108,879 at 1e-7 and 157,229 at 1e-8, against 145,313,734 stored.
5. **Convergence.**
   - `total_converged` is False on 2,578 rows at 1e-7 and 2,638 at 1e-8 (2,262 in the store).
   - Kinds of unconverged call: bound only 4,014 / 3,745; resolution 1,737 / 2,182; phase-limited
     1,213 / 1,720; capped 0 / 0.
   - 634 rows have `total_abserr = +inf` at both rtols (558 in the store).
6. **Totals against A3 v2 at 1e-7.**
   - The change exceeds the stored error on 6,187 of 8,722 rows: 5,677 beyond 10× and 4,818 beyond
     100×; median 286×.
   - `Levin_z` differs on 22 of 1,160 `GkSourcePolicyData`; type and quality on none.
7. **Stationary points (1e-7).**
   - 720 rows, in 720 calls (all `Gqr`/`G+q+r`), hold an interior stationary point. In every one a
     region was accepted across it: 623 of them at depth 0 (the whole call accepted as one region),
     677 Levin, 238 phase-limited.
   - The grid spacing h runs from 1.7e-5 to 7.1e-3 (median 2.5e-3). A pair closer than h is not
     seen.
8. **Cost.** Levin wall sums 0.67 h at 1e-7 and 0.77 h at 1e-8; the median unit takes 0.25 s and
   0.27 s. Each k's producer re-run takes 53–58 s. A3 v2 stored 286 h of compute time.
9. **Disagreement warnings:** 0 (unit counters, the 2,320 assemblies, and a grep of the logs).

### 6a.4 The stores

A3 v2's six files and the copy's five keep the same SHA-256, size and mtime at every check: before
the gate, before the launch, after the run and after the fingerprints
([`38`](06a/out/cmd/38_closing_compare.txt)). No journal, `-wal` or `-shm` file appeared.
- `store fingerprint` on A3 v2 gives `0e32f24a…`, problems none
  ([`34`](06a/out/cmd/34_fingerprint_A3v2.txt)).
- `fingerprint_copy.py` on the copy gives `0e32f24a…`, problems none
  ([`35`](06a/out/cmd/35_fingerprint_copy_after.txt)).

The worktree and the main checkout are clean at the end ([`44`](06a/out/cmd/44_final_state.txt)).

### 6a.5 The review, and the user's ruling

The Mac Studio orchestrator's review ([`06a/review.md`](06a/review.md)) passed nine of its ten
checks and stopped on check 9 alone. `out/cmd/17_C0_check.txt` records a `c0.py` SHA-256,
`b33229d8…`, that no kept file has. It was the first `c0.py`, superseded under deviation 15. C0's
verdict rests on [`18`](06a/out/cmd/18_C0_check_rerun.txt), whose `c0.py` (`2c1b8067…`) is the file
in `scripts/`. **The user ruled on 2026-10-27 that the record is accepted as it stands**: no number
depends on the superseded check, and re-stamping it would falsify it.

The laptop re-checked two more of 05's rows against 06's units, `mid-G-3` at `1e-7` and `R4` at
`1e-8`, and one line of `census_rows.csv.gz`, `0-1971` at `1e-8`, against its checkpoint record.
All agree in `float.hex` ([`06a/laptop_rechecks/`](06a/laptop_rechecks/)).

## 6b. The round-off-limited exit

*Prompt 06b, 2026-10-28: the code on the laptop, the rehearsals and the census on the Mac Studio.
The records are under [`06b/`](06b/). The pickles, the census's run directory and every file over
2 MB stay on the Mac Studio ([`06b/KEPT_ON_MAC_STUDIO.md`](06b/KEPT_ON_MAC_STUDIO.md)).*

### 6b.1 The code

`_adaptive_levin` accepts a region as round-off-limited only if its halves' eq. (151) floors sum to
at least $\rho = \tfrac12$ of its own (`_LEVIN_ROUNDOFF_EXIT_MIN_FLOOR_RATIO`,
`_roundoff_exit_premise_holds`), on both branches, checked only where the exit would otherwise
fire. Code commit `4ddc3c2`, reviewed on the laptop against eight checks. The tests, the three
breakages and the 20 seeded LiouvilleGreen draws are in the log,
[`logs/06b-roundoff-exit-premise.md`](../../prompts/levin-phase-floor/logs/06b-roundoff-exit-premise.md),
with the draws' records in [`06b/calibration/`](06b/calibration/).

### 6b.2 Rehearsals R15 and R16, and the value comparison (prompt 06b §4)

**Prompt 06b §4, run 2026-10-28, 13:52–14:12 UTC, on the Mac Studio** (mps026298, Darwin 27.0.0).
- **The trees.** Both builds ran from git worktrees, so the main checkout did not move:
  - **R15** ran in `SecondaryGWKit-lpf06b-da5c24d`, detached at `da5c24d`, the parent of the code
    commit `4ddc3c2`;
  - **R16** ran in `SecondaryGWKit-lpf06b-92bc62f`, detached at Part B's dispatch SHA `92bc62f`.
    Its code is `4ddc3c2`'s: `4ddc3c2..92bc62f` is empty outside `docs/` and `prompts/`.
  - **On code paths,** `da5c24d..92bc62f` touches only `AdaptiveLevin/levin_quadrature.py` and
    `AdaptiveLevin/tests/test_levin_roundoff_exit.py`.
- **The main checkout** stayed at `28ebaf0` and clean. It holds the venv, the stores, the registry
  and these records.
- **The software.** Python 3.13.16 from the main checkout's venv: `ray` 2.59.0, `SQLAlchemy`
  2.0.55, `datastorekit` 0.5.0, `raypoolkit` 0.3.0 and `runregistrykit` 0.2.0.
- **The load** was about 9 on 10 cores, all of it macOS and desktop processes.

**Records** are in [`06b/rehearsal/`](06b/rehearsal/): `out/`, `scripts/`, `state/`, the agent's
`log.md`, this section as first written, and `issues.md`.
- **Commands.** Every command is in [`out/commands.log`](06b/rehearsal/out/commands.log).
- **Scripts, by SHA-256** (from [`out/final_check.txt`](06b/rehearsal/out/final_check.txt)):
  - `classify_non_qsi.py` is `26690e61…`. It is 05's, copied unchanged, and byte-identical to 05's
    and 06's copies;
  - `run_readers_worktree.sh` is `ec96cf09…`, 06's, copied unchanged;
  - `run_readers.sh` is `7c63a5ba…`;
  - the launcher is `5b984c75…` in both worktrees and in the main checkout;
  - `tools/store_value_diff.py` is `7997af85…` in both worktrees;
  - the new `tabulate_qsi_phase_limited.py` is `ecde056d…`. Its failed first attempt is kept as
    `state/tabulate_qsi_phase_limited.attempt1.py`, `03176ba4…`.
- **Not copied back:** the readers' `extract/` (180 MB).

#### The two builds

| | R15 (before) | R16 (after) |
|---|---|---|
| tree | worktree at `da5c24d` (parent of `4ddc3c2`) | worktree at `92bc62f` (code `4ddc3c2`) |
| how it ran | cwd and `PYTHONPATH` the worktree; the main checkout's venv python by absolute path; `-B -u` and `PYTHONDONTWRITEBYTECODE=1`; `nohup sh -c`, detached | the same, in its own worktree |
| run | `levin-phase-floor-06b-r15-parent-20261028T135349` | `levin-phase-floor-06b-r16-premise-20261028T135745` |
| `sh` / driver pid | 96490 / 96498 | 97356 / 97360 |
| wall | 13:53:48 launch → 13:56:59 last heartbeat; driver gone 13:57:07 | 13:57:43 → 14:01:04; driver gone 14:01:13 |
| manifest | `git_head` `da5c24de…`, `git_dirty` false, cwd the parent worktree, script SHA `5b984c75…` | `git_head` `92bc62f0…`, `git_dirty` false, cwd the code worktree, script SHA `5b984c75…` |
| cwd of the driver (`lsof`) | the parent worktree | the code worktree |
| end | `done`, exit 0; no Traceback; `stderr.log` empty; no `!! WARNING` in the launch log | the same |
| store | `var/datastores/levin-phase-floor-06b-r15.sqlite` + 4 shards | `var/datastores/levin-phase-floor-06b-r16.sqlite` + 4 shards |

**The scope** was R8's rehearsal scope, launched as R13 and R14 were:
- 3 k over 1e5–1e6/Mpc, zend 250;
- atol 1e-32, rtol 1e-7, 6 CPUs;
- `--job-name levin-phase-floor-06b`, `--register r15-parent` / `r16-premise`, and
  `--campaign levin-phase-floor --prompt 06b`.

**The worktrees were not written.** After each build, after the readers and at the end, each
worktree's `git status --short --ignored` was empty, and `find -newer` the pre-R15 marker listed
nothing in either.

#### The lock

[`state/lock.log`](06b/rehearsal/state/lock.log). Each lock command was a shell command of its
own, with its output appended by `>>`.

| # | command | output | exit |
|---|---|---|---|
| 1 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (before) | nothing | 0 |
| 2 | `acquire --project SecondaryGWKit --purpose "levin-phase-floor prompt 06b section 4, rehearsals R15 … and R16 …" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 3 | `set-pid … 96498` (R15) | recorded | 0 |
| 4 | `ray-processes` (after R15) | nothing | 0 |
| 5 | `set-pid … 97360` (R16) | recorded | 0 |
| 6 | `ray-processes` (after R16) | nothing | 0 |
| 7 | `ray-processes` (after the readers) | nothing | 0 |
| 8 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 9 | `status --ray-temp-dir /tmp/ray-sgk` | free; no Ray processes | 0 |
| 10 | `ray-processes --ray-temp-dir /tmp/ray-sgk` | nothing | 0 |

#### The fingerprints (prompt §4.2)

| | digest | problems | per-class |
|---|---|---|---|
| R8 (reference, `handover-12-r8-reference-20261025T210551`) | `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` | none | 21 classes |
| **R15** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | `classes` equals R8's |
| **R16** | **`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`** | none | `classes` equals R8's and R15's |

These are the fingerprints the launcher took at the end of each build, read from `status.json`:
[`out/r15_status.json`](06b/rehearsal/out/r15_status.json) and
[`out/r16_status.json`](06b/rehearsal/out/r16_status.json). R15's was taken at `da5c24d` and
R16's at `92bc62f`, both with `git_dirty` false. There is no sidecar, because `--build` writes
none.

#### The value comparison (prompt §4.3)

The comparison was run from the code worktree, so that `-m` imports that tree's tool:

```
cd <code worktree> && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<code worktree> <main>/venv/bin/python -B -m tools.store_value_diff <main>/var/datastores/levin-phase-floor-06b-r15.sqlite <main>/var/datastores/levin-phase-floor-06b-r16.sqlite --json
```

- **The run.** It exited **1**, in 18 s, and wrote 40.3 MB of JSON (`out/store_value_diff.json`,
  SHA-256 `51b9b94a…`; [`out/store_value_diff.meta.txt`](06b/rehearsal/out/store_value_diff.meta.txt)).
- **The structure.** `not_compared` is empty. Every class has 0 rows in one store only, 0 rows
  that cannot be matched and 0 unresolvable columns.
- **The stores did not change.** Both stores' ten files are byte-identical before and after it
  ([`state/store_sha256_before_diff.txt`](06b/rehearsal/state/store_sha256_before_diff.txt),
  [`state/store_sha256_after_diff.txt`](06b/rehearsal/state/store_sha256_after_diff.txt)).

##### Outside `QuadSourceIntegral`

The source is 05's [`scripts/classify_non_qsi.py`](06b/rehearsal/scripts/classify_non_qsi.py),
unchanged, with output in [`out/classify_non_qsi.txt`](06b/rehearsal/out/classify_non_qsi.txt). It
exits 0 and reports **0 unclassified differences or structural problems**.

- **Every value table is bit-identical:** 7 tables, 360,462 rows.

  | value table | rows identical | differ |
  |---|---|---|
  | BackgroundModelValue | 1,159 | 0 |
  | TkNumericValue | 1,478 | 0 |
  | TkWKBValue | 2,209 | 0 |
  | GkNumericValue | 35,874 | 0 |
  | GkWKBValue | 147,642 | 0 |
  | GkSourceValue | 169,653 | 0 |
  | QuadSourceValue | 2,447 | 0 |

- **11 classes are row-identical:** `version`, `store_tag`, `redshift`, `wavenumber`,
  `tolerance`, `LambdaCDM`, `QCD_Cosmology`, `IntegrationSolver`, `GkSourcePolicy`,
  `QuadSourcePolicy` and `GkSourcePolicyData`. `OneLoopIntegral` has no rows.
- **Eight classes differ in row columns of the volatile kinds only:**

  | class | rows | columns (kind) |
  |---|---|---|
  | wavenumber_exit_time | 3 | `compute_time` (timing) |
  | BackgroundModel | 1 | `compute_time`, `{min,mean,max}_RHS_time` (timing) |
  | TkNumericIntegration | 3 | `compute_time`, `*_RHS_time` (timing); `label` |
  | TkWKBIntegration | 3 | `stage_1_compute_time` (timing); `label` |
  | GkNumericIntegration | 1368 | `compute_time`, `*_RHS_time` (timing; `min_RHS_time` in 1022); `label` |
  | GkWKBIntegration | 3477 | `stage_1_compute_time` (timing, 3474); `label`; `metadata` (`rho_evals`/`rho_reused` only) and `stage_1_RHS_evaluations` in 23 rows (the residual-memo work counter) |
  | GkSource | 291 | `label`; `parents_digest` |
  | QuadSource | 6 | `compute_time` (timing); `label` |

##### Inside `QuadSourceIntegral`: where the phase-limited flags are stored

The flags were established from `ComputeTargets/QuadSourceIntegral.py` and
`ObjectFactories/QuadSourceIntegral.py` at `92bc62f`, and confirmed on the stores. All three live
in the shard table `QuadSourceIntegral`.

| flag | column | encoding | set by |
|---|---|---|---|
| **(a)** `total_phase_limited` | `total_phase_limited` | SQLAlchemy `Boolean`, nullable. Stored as SQLite `integer` 0/1: `typeof` is `integer` on all 864 rows of both stores | `evaluate_QuadSource_integral`: the OR of (b) over every Levin sub-interval |
| **(b)** each group record's flag | `metadata` (a JSON string, `json.dumps` / `json.loads`), key `WKB_Levin.subintervals[i].groups[j].phase_limited` | JSON `true`/`false` | `phase_group_Levin_integral`: one record per phase group, holding `label`, `value`, `abserr`, `converged`, `phase_limited`, `regions`, `simple_regions`, `evaluations` and `elapsed` |
| **(c)** each analytic block's flag | `metadata`, key `analytic.{0pt5,2pt5}.{J,Y}{1..4}_data.phase_limited` (three-Bessel integral all Levin) or `analytic.{0pt5,2pt5}.Levin.{J,Y}{1..4}_data.phase_limited` (split quad + Levin) | JSON `true`/`false` | `_three_bessel_Levin`: one record per block, holding `elapsed`, `regions`, `abserr`, `converged` and `phase_limited` |

- **The ORs of (b)** are `WKB_Levin.phase_limited` and `WKB_Levin.subintervals[i].phase_limited`.
  The tabulation asserts, on every row of both stores, that each is the OR of its groups and that
  (a) is the OR of all (b).
- **The ORs of (c)** are not stored.
- **No other** metadata leaf named `phase_limited` exists.
- **"Phase-limited in R15"** is (a) or any (b) or any (c), on R15's row.

##### Inside `QuadSourceIntegral`: per row

The new script is
[`scripts/tabulate_qsi_phase_limited.py`](06b/rehearsal/scripts/tabulate_qsi_phase_limited.py).
- **How it reads.** It opens every store file with sqlite `mode=ro`, and pairs rows by physical
  $(k, q, r, z_{\rm resp})$.
- **The cross-checks.** The pairing gives the tool's key set, and `total`'s `float.hex` agrees with
  the tool on all 864 rows. Both stores have one tolerance pair, (atol, rtol) = (1e-32, 1e-7), and
  their metadata key sets are equal on every row.
- **Outputs:** [`out/qsi_tabulation.txt`](06b/rehearsal/out/qsi_tabulation.txt), with one line per
  row in `out/qsi_rows.csv`.
- **Bit-identity** is by `float.hex` for floats, and by type and `==` otherwise.

**Field kinds:**
- **Not compared:**
  - timings: columns ending `compute_time`, `RHS_time` or `elapsed`, and metadata leaves `elapsed`
    and `compute_time`;
  - `label`;
  - store identities: `serial`, every `*_serial`, and `timestamp`.
- **Work counters,** compared and reported separately:
  - the columns `WKB_Levin_{num_regions, evaluations, simple_regions, SVD_errors, order_changes,
    chebyshev_min_order, max_depth}`, `numeric_quad_compute_steps` and
    `numeric_quad_RHS_evaluations`;
  - the metadata leaves `regions`, `simple_regions` and `evaluations`.
- **Values:** everything else, every metadata leaf included.

**The counts:**

| | R15 | R16 |
|---|---|---|
| rows phase-limited (any flag) | **82** of 864 | **0** |
| (a) `total_phase_limited` true | 26 | 0 |
| rows with any (b) | 26 | 0 |
| (b) flags set, of 3,105 group records | 48 | 0 |
| rows with any (c) | 70 | 0 |
| (c) flags set, of 10,072 analytic blocks | 143 | 0 |

Of the 82 rows, 12 are (a/b) only, 56 are (c) only and 14 are both.

**The rule: a row with no phase-limited flag in R15 is bit-identical.** It holds on all 782 such
rows:
- **The named set:** 0 rows differ in `total`, `total_abserr`, `analytic_rad`, any `analytic`
  metadata leaf or any field of any group record.
- **Every value field** of the row, columns and metadata: 0 rows differ.
- **Every work counter:** 0 rows differ.

**The other 82 rows are tabulated in full** in [`out/qsi_tabulation.txt`](06b/rehearsal/out/qsi_tabulation.txt).
- **`total`** moved in exactly the 26 rows with a (b) flag in R15, and in no other
  ([`out/qsi_set_check.txt`](06b/rehearsal/out/qsi_set_check.txt)).
- **`analytic_rad`** moved in exactly the 70 rows with a (c) flag.
- **The 56 (c)-only rows** keep `total` and `total_abserr` bit for bit.

The 26 rows whose total moved:

| (k, q, r) /Mpc | z_resp | flags R15 a/b/c | `total_abserr` R15 → R16 | R16/R15 | \|ΔT\|/err15 | \|ΔT\|/err16 | Levin regions | max depth | converged |
|---|---|---|---|---|---|---|---|---|---|
| (1e5, 1e5, 1e5) | 755.687 | 1/1/0 | 3.561e-25 → 3.507e-25 | 0.985 | 0.00182 | 0.00185 | 27 → 29 | 11 → 11 | 0 → 0 |
| (1e5, 3.16e5, 3.16e5) | 755.687 | 1/2/0 | 5.089e-26 → 4.608e-26 | 0.905 | 0.0642 | 0.0709 | 15 → 17 | 6 → 6 | 0 → 0 |
| (1e5, 3.16e5, 3.16e5) | 36286.8 | 1/1/0 | 2.174e-24 → 2.200e-24 | 1.012 | 0.00661 | 0.00653 | 23 → 28 | 7 → 7 | 0 → 0 |
| (1e5, 3.16e5, 3.16e5) | 83186 | 1/1/0 | 5.036e-24 → 5.044e-24 | 1.002 | 0.0048 | 0.0048 | 21 → 24 | 7 → 7 | 0 → 0 |
| (1e5, 1e6, 1e6) | 434.651 | 1/2/3 | 3.054e-27 → 2.579e-27 | 0.844 | 0.0492 | 0.0582 | 15 → 17 | 6 → 6 | 0 → 1 |
| (1e5, 1e6, 1e6) | 755.687 | 1/1/3 | 5.664e-27 → 5.336e-27 | 0.942 | 0.024 | 0.0255 | 14 → 15 | 6 → 6 | 0 → 0 |
| (1e5, 1e6, 1e6) | 1313.84 | 1/2/3 | 8.130e-27 → 7.024e-27 | 0.864 | 0.0435 | 0.0503 | 10 → 12 | 3 → 3 | 0 → 1 |
| (1e5, 1e6, 1e6) | 3011.93 | 1/2/2 | 1.517e-26 → 1.349e-26 | 0.889 | 0.0399 | 0.0448 | 15 → 17 | 6 → 6 | 0 → 1 |
| (1e5, 1e6, 1e6) | 9104.3 | 1/1/1 | 4.172e-26 → 4.057e-26 | 0.973 | 0.000935 | 0.000962 | 15 → 17 | 6 → 6 | 0 → 0 |
| (3.16e5, 3.16e5, 3.16e5) | 329.64 | 1/3/0 | 3.248e-27 → 1.653e-27 | 0.509 | 0.0439 | 0.0863 | 32 → 35 | 12 → 12 | 0 → 1 |
| (3.16e5, 3.16e5, 3.16e5) | 1732.38 | 1/1/0 | 1.504e-26 → 1.425e-26 | 0.947 | 0.00699 | 0.00738 | 33 → 34 | 12 → 12 | 0 → 0 |
| (3.16e5, 3.16e5, 3.16e5) | 2284.25 | 1/1/0 | 1.828e-26 → 1.735e-26 | 0.949 | 0.00127 | 0.00134 | 32 → 35 | 12 → 12 | 0 → 0 |
| (3.16e5, 3.16e5, 3.16e5) | 3011.93 | 1/1/0 | 2.345e-26 → 2.246e-26 | 0.958 | 0.00356 | 0.00372 | 30 → 32 | 12 → 12 | 0 → 0 |
| (3.16e5, 1e6, 1e6) | 250 | 1/2/2 | 8.930e-28 → 5.711e-28 | 0.640 | 0.36 | 0.563 | 12 → 14 | 6 → 6 | 0 → 0 |
| (3.16e5, 1e6, 1e6) | 573.115 | 1/1/1 | 1.454e-27 → 1.408e-27 | 0.968 | 0.0351 | 0.0362 | 14 → 15 | 7 → 7 | 0 → 0 |
| (3.16e5, 1e6, 1e6) | 755.687 | 1/3/3 | 2.362e-27 → 1.882e-27 | 0.797 | 0.124 | 0.155 | 14 → 17 | 7 → 7 | 0 → 0 |
| (3.16e5, 1e6, 1e6) | 1732.38 | 1/4/1 | 4.007e-27 → 2.921e-27 | 0.729 | 0.152 | 0.208 | 12 → 20 | 6 → 6 | 0 → 1 |
| (3.16e5, 1e6, 1e6) | 2284.25 | 1/4/0 | 4.882e-27 → 3.653e-27 | 0.748 | 0.107 | 0.143 | 12 → 18 | 6 → 6 | 0 → 1 |
| (3.16e5, 1e6, 1e6) | 3011.93 | 1/4/1 | 6.076e-27 → 4.916e-27 | 0.809 | 0.0896 | 0.111 | 12 → 16 | 6 → 6 | 0 → 1 |
| (3.16e5, 1e6, 1e6) | 15828.8 | 1/2/0 | 3.035e-26 → 2.953e-26 | 0.973 | 0.00275 | 0.00283 | 20 → 25 | 6 → 6 | 0 → 0 |
| (1e6, 3.16e5, 1e6) | 250 | 1/1/3 | 5.501e-28 → 3.678e-28 | 0.668 | 0.046 | 0.0688 | 35 → 36 | 7 → 7 | 0 → 0 |
| (1e6, 3.16e5, 1e6) | 329.64 | 1/1/1 | 6.076e-28 → 4.110e-28 | 0.677 | 0.0249 | 0.0367 | 39 → 40 | 7 → 7 | 0 → 0 |
| (1e6, 3.16e5, 1e6) | 573.115 | 1/1/2 | 1.153e-27 → 9.348e-28 | 0.811 | 0.0405 | 0.05 | 31 → 32 | 6 → 6 | 0 → 0 |
| (1e6, 1e6, 1e6) | 329.64 | 1/3/1 | 5.416e-28 → 1.310e-28 | 0.242 | 0.144 | 0.597 | 16 → 27 | 7 → 7 | 0 → 0 |
| (1e6, 1e6, 1e6) | 755.687 | 1/1/0 | 8.133e-28 → 5.230e-28 | 0.643 | 0.333 | 0.517 | 14 → 15 | 7 → 7 | 0 → 0 |
| (1e6, 1e6, 1e6) | 20871.2 | 1/2/0 | 6.169e-27 → 6.072e-27 | 0.984 | 0.0138 | 0.014 | 17 → 25 | 8 → 8 | 0 → 0 |

**Summaries of the 82 flagged rows:**
- **The 26 rows whose total moved:**
  - $|\Delta T|$ is at most 0.36 × `total_abserr`$_{R15}$ and at most 0.597 ×
    `total_abserr`$_{R16}$;
  - `total_abserr` fell in 24, by a factor of 0.242–0.985, and rose in 2, by a factor of 1.002 and
    1.012;
  - `WKB_Levin_max_depth` did not change in any of them;
  - their Levin regions rose from 530 to 612, and their `WKB_Levin_evaluations` from 1314 to 1638;
  - `total_converged` went 0 → 1 in 7, and stayed 0 in 19.
- **The 70 rows whose `analytic_rad` moved:** $|\Delta$`analytic_rad`$|$ against R15's analytic
  `abserr` has a median of 0.058 and a maximum of 0.748. Their analytic-block regions rose from
  2203 to 2355.
- **Over all 82:** `total_converged` is 0 → 0 in 71, 0 → 1 in 7 and 1 → 1 in 4.

#### The readers (prompt §4.4)

- **Where they ran.** On R16 from the code worktree, with the lock held, through 06's
  `run_readers_worktree.sh`. That script is `run_readers.sh` with only its tree and its python
  changed ([`out/run_readers_worktree.diff`](06b/rehearsal/out/run_readers_worktree.diff), equal
  to 06's).
- **The results** are in [`out/readers.txt`](06b/rehearsal/out/readers.txt) and
  [`out/readers_check.txt`](06b/rehearsal/out/readers_check.txt).

| reader | exit | wall | files | store before = after |
|---|---|---|---|---|
| extract_Gk_data | 0 | 27 s | 468 | yes |
| extract_GkWKB_data | 0 | 50 s | 2436 | yes |
| extract_GkSource_data | 0 | 20 s | 180 | yes |
| extract_TkWKB_data | 0 | 15 s | 27 | yes |
| extract_tensor_source_data | 0 | 14 s | 27 | yes |
| extract_QuadSourceIntegral_data | 0 | 18 s | 76 | yes |

- **The outcome:** 0 Tracebacks, and no Ray process left after any reader.
- **The warning.** Each reader prints one
  `!! WARNING: number of shards read from database (=4) does not match specified number of shards
  (=10)`, as in 06.
- **The store.** R16's five files are byte-identical before and after the readers
  ([`state/r16_store_sha256_before_reads.txt`](06b/rehearsal/state/r16_store_sha256_before_reads.txt),
  [`state/r16_store_sha256_after_reads.txt`](06b/rehearsal/state/r16_store_sha256_after_reads.txt)),
  and equal to the hashes taken before the comparison.
- **The file counts** differ from 06's run on R14. The readers draw their subsample with unseeded
  `random.sample`.

#### At the end

- **The stores** are kept; their ten files' final hashes are
  [`state/stores_sha256_final.txt`](06b/rehearsal/state/stores_sha256_final.txt), equal to those
  before the comparison.
- **The registry** holds 27 runs, all finished.
- **The lock** is free, and `ray-processes` lists nothing.
- **The trees:** the main checkout is at `28ebaf0` and clean, and both worktrees are clean.

**§6.2 acceptance holds:**
- R15 and R16 are `done` with `1b332ffa…`;
- outside `QuadSourceIntegral`, every value table is identical;
- inside it, every row with no phase-limited flag in R15 is bit-identical, and the other 82 are
  tabulated;
- the readers exit 0, and the lock is released.

### 6b.3 The census: 06a's census re-run at the code commit (Mac Studio, 2026-10-28)

**What was run.** Run `levin-phase-floor-06b-census-20261028T161212` re-ran every A3 v2
`QuadSourceIntegral` row (9,280) at `rtol` 1e-7 and 1e-8, in-process and Ray-free.
- **Tree:** $b$, the worktree at `92bc62f`, whose code is `4ddc3c2`'s.
- **Store:** 06's certified scratch copy of A3 v2, read-only.
- **Comparison:** against 06a's run `levin-phase-floor-06a-census-20261027T023143`.
- **Outcome:** 18,560 of 18,560 units `done`, none over budget, none in error. 16:12–17:03Z.
- **Harness:** 06a's, changed as the log's deviations 1–8 say. The diff is
  [`06b/census/out/scripts_diff_against_06a_final.txt`](06b/census/out/scripts_diff_against_06a_final.txt).

**Gate W0 holds** ([`06b/census/out/cmd/14`](06b/census/out/cmd/14_W0_check_rerun.txt),
[`15`](06b/census/out/cmd/15_W0_clause1.txt)):
1. **The parent's code is 06a's.** `e05c34f..da5c24d` changes:
   - `CLAUDE.md`;
   - three files whose ASTs without docstrings are equal: `primitive_phase.py`, `main.py` and
     `test_primitive_phase.py`;
   - four test modules that the harness does not import.
2. **The harness reproduces 06a.** At `da5c24d`, `low-G+r-1` and `ctrl-low-1` at 1e-7, and `0-1971`
   at 1e-8, are identical to 06a's units in every computed value.
3. **The new code is deterministic.** `b-1e-7-1-34599` is identical in two fresh workers.

**The predictions.** [`06b/census/state/predictions.md`](06b/census/state/predictions.md), sha256
`8df51796…`, was written at 14:13:55Z, before the launch. Part A is the prompt's W1–W4 byte for
byte. Part B fixes W2's selection rule and the readings, including the analytic-block reading the
user accepted.

| | Measured | Verdict |
|---|---|---|
| **W1** | 36,112 / 35,605 group calls (1e-7 / 1e-8) not phase-limited in 06a: all equal to 06a's bit for bit (value, abserr, regions, depth). 6,777 / 6,177 rows all of whose calls are such: all equal in `total` and `total_abserr`. | **holds** |
| **W2** | 24 calls (12 low, 12 mid). Every re-run from the saved inputs reproduces b. $\|S_9-S_{10}\| \le 0.0066$ `abserr`$_b$, so none is dropped. $\|{\rm value}_b - S_{10}\| \le$ `abserr`$_b$ on 7 and fails on 17, by up to 19.7×. | **fails** |
| **W3** | 44 regions accepted as phase-limited at 1e-7, all in analytic-block calls. Recomputed floors: own equals stored on all 44. Halves' sum / own is 0.504–0.891. 0 violate the rule. | **holds** |
| **W4** | Three moved analytic-block calls at 1e-8 reach depth 20 (up to 31,885 regions; unit wall up to 159 s). 0 units over budget. | **fails** |

**The store.** The copy's five files equal `copy_files_certified.txt` before W0, before the launch,
after the run and after the fingerprint, with no journal file. It gives `0e32f24a…` before and
after.

**What W2's failure means** ([`out/cmd/38`](06b/census/out/cmd/38_w2_crosscheck.txt),
[`39`](06b/census/out/cmd/39_w2_request.txt)):
- **The reference holds up.** $S_{11}$ agrees with $S_{10}$. On one call the adaptive driver at
  `rtol` 1e-11 agrees too.
- **On two others the driver at `rtol` 1e-11 returns b's value unchanged,** with the same regions.
  Their residuals and floors sit well below the true error.
- **b's values are better than 06a's, but its error is understated.** Against the request, b's
  error exceeds it on 4 of 24 calls and 06a's on 23 of 24. b's declared `abserr` is below its true
  error on 17 of 24.

**Descriptive items** ([`out/cmd/37`](06b/census/out/cmd/37_analyse06b_report_rerun.txt)):
- **Phase-limited group calls** fall from 1,213 (on 585 rows) to 0 at 1e-7, and from 1,720 (745)
  to 0 at 1e-8. Every one of 06a's phase-limited regions fails the premise.
- **Analytic blocks:** 12,633 / 17,722 of their calls reached the old exit under 06a. 32 / 800 are
  still phase-limited under b.
- **Row depth:** largest 14 under b, against 12 / 13 under 06a; with the analytic blocks, 18 / 20.
- **`total_converged` false:** 2,578 → 2,349 rows at 1e-7, and 2,638 → 2,388 at 1e-8.
- **Wall time:** 51 min against 45.5.

**Records:**
- [`06b/census/log.md`](06b/census/log.md) and [`06b/census/issues.md`](06b/census/issues.md)
  (C1–C4);
- W3's per-region floors: [`06b/census/out/w3_regions.csv`](06b/census/out/w3_regions.csv);
- W2's sample: [`06b/census/state/w2_selection.json`](06b/census/state/w2_selection.json);
- W2's references: [`06b/census/out/w2_reference.json`](06b/census/out/w2_reference.json).

The checkpoint, the region pickles and the 975 captured calls' inputs stay on the Mac Studio, in
`var/runs/levin-phase-floor-06b-census-20261028T161212/` (checkpoint sha256 `f934ea24…`).

### 6b.4 The review, and the user's rulings

The Mac Studio orchestrator's review ([`06b/review.md`](06b/review.md)) passed checks 1–9, re-scoring
W1–W4 from both checkpoints with its own readers, and stopped on check 10 alone: the census log has
no **What shipped** section, and its deviation 19 is unclassified. The laptop re-checked, over
`ssh macstudio-tunnel` with readers of its own:
- **one more moved rehearsal row,** $(k, q, r) = (10^6, 3.162\times10^5, 10^6)$ at
  $z_{\rm response} = 573.1$: `total` is `0x1.7ee4bd7f61ebap-69` in R15 (shard 1, serial 20087)
  and `0x1.7ee4bdf5de672p-69` in R16 (serial 20139), as tabulated;
- **one census unit against 06a's,** `1e-8-2-32862`: its 82 compared leaves differ only in
  provenance (script hashes, pickle hash and length, rebuild time, the unit label) and in the
  fields 06b added; every computed value is bit-identical.

**The user's rulings, 2026-10-28:**
- **Check 10:** *"If the only issue is that the agent didn't say whether anything went over budget,
  then I don't care. Just accept it."*
- **The header issue** `[07-roundoff-exit-assumes-bisection-cannot-help]` stays open: *"seems that
  we should keep it open until we've fixed W2 at least."*
- **W2's failure** is opened as `[06b-moved-calls-declare-less-than-their-error]`. W4's three calls
  narrow `[06b-premise-check-can-chase-a-flat-residual-to-depth-max]` rather than opening a
  duplicate. The agents' other proposals (C3, C4, N1, N2) are not opened.

## 6c. Global error control, with 06d's Clenshaw–Curtis floor

*Prompts 06c and 06d: the code on the laptop (06c `ad9a5f4`, 06d `3d70db9`), the rehearsals and
the census on the Mac Studio on 2026-10-29, on the two together (06c's orchestrator note §B.6). The
records are under [`06c/`](06c/); what stays on the Mac Studio is in
[`06c/KEPT_ON_MAC_STUDIO.md`](06c/KEPT_ON_MAC_STUDIO.md).*

### 6c.1 The code

06c: `_adaptive_levin` keeps every leaf of its partition and stops when
$\sum e_i \le \max(\texttt{atol}, \texttt{rtol}|\sum v_i|)$, bisecting the refinable leaf with the
largest error, with a trust check one level down when the budget passes. 06d: the Clenshaw–Curtis
fallback's round-off floor is derived from the rule, $\phi_{\rm CC}$ = sum, argument and node terms,
with no $1/\theta'$, and the depth-18 log is guarded. The tests, breakages and LiouvilleGreen draws
are in [`logs/06c-global-error-control.md`](../../prompts/levin-phase-floor/logs/06c-global-error-control.md)
and [`logs/06d-cc-roundoff-floor.md`](../../prompts/levin-phase-floor/logs/06d-cc-roundoff-floor.md),
with the draws' records in [`06c/calibration/`](06c/calibration/) and
[`06d/calibration/`](06d/calibration/).

### 6c.2 Rehearsals R17 and R18 (prompt 06c §4)

R17 `levin-phase-floor-06c-r17-parent-20261029T123720` at `fef401c` (06b's code) and R18
`levin-phase-floor-06c-r18-global-20261029T124058` at `e5764a6`, both `done`, both fingerprinted
`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. Outside `QuadSourceIntegral`, 7
value tables, 360,462 rows, bit-identical. Inside, of 864 rows, `total` is bit-identical in 295;
`total_converged` 516 → 863; regions 13,577 → 34,892; maximum row depth 12 → 20.
- **Y1 fails** on 30 of 838 rows. The 12 worst, all equilateral, hold a stationary point of
  `G+q+r` at $z \approx 10211$ ($c_s = 1/2$), where one code or the other is wrong by
  $4.5\times10^3$–$1.6\times10^6$× its declared error while reporting converged; 07's. On the other
  18, against the orchestrator's references, R18 is within its error (0.006–0.37) and R17 is not
  (1.5–68×) ([`06c/y1-references/`](06c/y1-references/)).
- The agent's full record: [`06c/rehearsal/verification.md`](06c/rehearsal/verification.md).

### 6c.3 The census: 06b's census re-run at the code (prompt 06c §5)

`levin-phase-floor-06c-census-20261029T130214`: 18,560 of 18,560 units `done`, none in error, in
2 h 30 min against 06b's 51 min; nine units ran past 300 s unmarked.
- **X0 holds** (06b's units reproduced bit for bit at the parent; $g$ deterministic). **X3 holds.**
- **X1 fails:** 22 of 23 calls within their error; `g-1e-7-0-14127` call 3 at 1.375×, a 7-leaf
  answer the trust check passed, its error reachable by refinement.
- **X2 fails:** 53 of 20,425 and 127 of 22,442 of 06b's budget misses stay unconverged, most at
  depth 20.
- **X4 fails:** 263 and 302 of 8,092 rows. On the four worst rows' calls, referenced, 06b's error
  is 116–445× its declared error, 06c's 0.11–0.61×.
- **X5 fails:** 109 and 153 moved calls reach depth 20. On their last five levels the residual
  dominates and sits a median $2\times10^4$ above the floor.
- The agent's full record: [`06c/census/verification.md`](06c/census/verification.md).

### 6c.4 The depth-20 calls: the Chebyshev end nodes (2026-10-29)

On the Mac Studio, a check of X5's 262 depth-20 calls ([`06c/depth20-tiling/results.md`](06c/depth20-tiling/results.md),
its orchestrator's re-check in [`orchestrator_check/`](06c/depth20-tiling/orchestrator_check/)):
over their last five levels, 1,179 of the 1,222 Levin levels whose residual exceeds 10× its floor
have end nodes that do not tile, against a base rate of 93% over all levels. On 30 sampled calls,
pinning only the end nodes to the span brings 127 of 138 flagged levels within 10× of the floor;
maximum depth 20 → 1–7 on 29, regions 10,181 → 274, converged 1 → 29. Honesty after pinning is
shown on one call (0.998 of its declared error); sample 25's references do not settle at the pinned
error. This is prompt 06e's case (`[06c-chebyshev-end-nodes-miss-the-span]`).

### 6c.5 The review, and the user's rulings

The Mac Studio orchestrator's review ([`06c/review.md`](06c/review.md)) passed all ten checks,
re-scoring Y1 and X1–X5 with its own readers. It corrected the census agent's reading of X1. The
user, 2026-10-30, on closing the four header issues of 06c and 06d despite the failed predictions:
*"If there's a real, substantive reason to keep them open, I don't mind. If this is just an
argument about words, I don't care."* None of the failures is any of the four (the board's
Decisions, **06c Part B lands**), and they close:
`[07-per-region-acceptance-misses-a-cancelling-total]`,
`[06b-resolved-region-keeps-a-floor-bisection-would-cut]`,
`[07-cc-solves-use-the-levin-roundoff-floor]` and `[07-depth18-diagnostic-divides-by-zero]`.

## 6e. Exact end nodes, and a node term that measures the displacement

*Prompt 06e: the code on the laptop (`f4d8e17`), the rehearsals, the census and Z3's references on
the Mac Studio on 2026-10-30 and 2026-10-31. The records are under [`06e/`](06e/); what stays on
the Mac Studio is in [`06e/KEPT_ON_MAC_STUDIO.md`](06e/KEPT_ON_MAC_STUDIO.md).*

### 6e.1 The code

`chebyshev_matrices` sets x[0] := b and x[−1] := a, so a parent's ends and its halves' tile. The
Clenshaw–Curtis node term becomes $\phi_{\rm node} = |E| + R_g + R_2 + R_c$, with
$E = \sum_k w_k\,g'(\hat x_k)\,\Delta_k$ from the exact per-node displacements, computed by
error-free transformations against 160-bit base corrections. The tests, breakages and
LiouvilleGreen draws are in
[`logs/06e-exact-end-nodes.md`](../../prompts/levin-phase-floor/logs/06e-exact-end-nodes.md), with
the draws' records in [`06e/calibration/`](06e/calibration/). P1 holds: the median of 06d's term
over $\phi_{\rm node}$ on 90 window cells is 848.

### 6e.2 Rehearsals R19 and R20 (prompt 06e §4)

R19 `levin-phase-floor-06e-r19-parent-20261030T074440` at `50277f3` (06d's code) and R20
`levin-phase-floor-06e-r20-endnodes-20261030T075034` at `6d81fad`, both `done`, both fingerprinted
`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`. Outside `QuadSourceIntegral`
every value table is identical. Inside, of 864 rows, `total` is bit-identical in 361;
`total_converged` 863 → 864; regions 34,892 → 31,216; rows at depth ≥ 12 46 → 11.
- **Z1 holds:** 864 rows, no violation; the largest $|\Delta T|/(e_{19} + e_{20})$ is 0.00312.
- The agent's full record: [`06e/rehearsal/verification.md`](06e/rehearsal/verification.md).

### 6e.3 The census: 06c's census re-run at the code (prompt 06e §5)

`levin-phase-floor-06e-census-20261030T081318`: 18,560 of 18,560 units `done`, none in error or
over budget, in about 1 h 47 min against 06c's 2 h 30 min. Unit wall falls ×0.711 at `1e-7` and
×0.515 at `1e-8`; calls at depth 20 fall 109 → 11 and 153 → 18, and outside 06a's 720
stationary-point rows 84 → 0 and 133 → 1.
- **Z0 holds** (06c's units reproduced bit for bit at the parent; $g$ deterministic). **Z4 holds**
  (8,560 rows at each rtol, no violation).
- **Z2 and Z5 fail** on one call, `g-1e-8-0-33995` call 5, unconverged at depth 20 with 8,706
  regions. Its floor, $1.33\times10^{-33}$ against atol $6.7\times10^{-34}$, is 06e's node term
  summed in absolute value over 8,600 Clenshaw–Curtis leaves; the sum does not fall under
  splitting.
- **Z3 fails** on one of six counted references: sample 3 at 1.245× its declared error (06c's code
  1.197×). The others are at 0.78–0.9995.
- The agent's full record: [`06e/census/verification.md`](06e/census/verification.md).

### 6e.4 The references, and the trusted leaf's value (2026-10-31)

Z3's references for samples 3 and 15 ran in `levin-phase-floor-06e-references-20261030T192730`, 9 h
35 min; the short eight ran 10:02–19:25 UTC on 2026-10-30, 2–3.5× the prompt's estimates. The Mac
Studio review re-ran sample 2's reference at P = 96 with 80/112 nodes and agreed to
$2.3\times10^{-6}$ of the error.

On the laptop the orchestrator broke down the near-1 ratios of samples 2, 8 and 9
([`06e/investigations/d3-leaf-values/`](06e/investigations/d3-leaf-values/)): in each, one trusted
leaf with $r \le 0.0014$ carries the error, and the trust check returns that leaf's own value while
declaring $\max(\epsilon_P, \Delta_1)/(1-r)$, which is that value's true error to within $r$. Its
quarters sum, already computed, is 650–2,200× closer. Sample 3's leaf has $r = 0.507$, and its
quarters sum is itself $1.7\times10^{-34}$ off: the error falls more slowly than the extrapolation
assumes.

### 6e.5 The review, and the user's rulings

The Mac Studio orchestrator's review ([`06e/review.md`](06e/review.md)) passed, re-scoring Z1–Z5
with its own readers and re-fingerprinting both stores. The user, 2026-10-31: *"OK. Close both
header issues. We need to open a new issue for D4."* None of the failures is either header issue
(the board's Decisions, **06e Part B lands**), and they close:
`[06c-chebyshev-end-nodes-miss-the-span]` and `[06d-cc-node-term-is-pessimistic]`. Three issues
open: `[06e-trusted-leaf-returns-its-coarsest-value]`,
`[06e-cc-declared-phase-term-is-a-crude-proxy]` and `[06e-node-errors-add-in-absolute-value]`.
