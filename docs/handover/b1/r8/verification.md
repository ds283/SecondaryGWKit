## 5. Rehearsal R8 — the Mac Studio, 2026-10-25

**Host:** mps026298 (Mac Studio, arm64, Darwin 27.0.0), checkout
`/Users/ds283/Documents/Code/SecondaryGWKit` at `8a02f0c` (B1's code `049c7c8`; the commits between
touch only markdown), `git status --short` empty throughout. Python 3.13.16, `ray` 2.59.0,
`SQLAlchemy` 2.0.55, `datastorekit` 0.5.0 (`f47b9a2`), `raypoolkit` 0.3.0 (`67a75b9`),
`runregistrykit` 0.2.0 (`0bae3a3`). Load average about 4.6 on 10 cores at the start.
**Outputs:** `var/handover-12/out/`, `state/`, `scripts/` and `extract/` on the Mac Studio.

### 5.1 The Gate and the roots (prompt §6.1)

Re-checked read-only before anything was written, as §B.5 of the orchestrator's prompt gives it:
`HEAD` `8a02f0c`; 11 runs in the registry, all finished, none `running` or stale; `~/.raypoolkit/`
empty and `~/ScienceData/.rehearsal-lock` absent; no `handover-12` store and no `var/handover-12/`;
R3's two recorded runs at `var/handover-06/recorded/`; `~/.ray/auth_token` mode `-rw-------`;
117 GiB free.

| root | laptop, Part A review | Mac Studio, `8a02f0c` |
|---|---|---|
| ComputeTargets | 742 OK | **742 OK** (137.8 s) |
| ObjectFactories | 280 OK | **280 OK** (12.0 s) |

`out/roots_ComputeTargets.txt`, `out/roots_ObjectFactories.txt`. No `FAIL:` or `ERROR:` line; the
wall-time flake did not fire.

### 5.2 The lock and the build (prompt §6.2, §6.3)

The lock transcript, `state/lock.log`, each command its own shell invocation, output by `>>`:

| # | command | output | exit |
|---|---|---|---|
| 1 | `acquire --project SecondaryGWKit --purpose "handover prompt 12, rehearsal R8 on the Mac Studio (mps026298), tree 8a02f0c (B1's code 049c7c8)" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | acquired | 0 |
| 2 | `set-pid --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk 21718` | recorded pid 21718 | 0 |
| 3 | `ray-processes --ray-temp-dir /tmp/ray-sgk` (after the readers) | (nothing listed) | 0 |
| 4 | `release --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | released | 0 |
| 5 | `status --project SecondaryGWKit --ray-temp-dir /tmp/ray-sgk` | usage error: `status` takes no `--project` | 2 |
| 6 | `status --ray-temp-dir /tmp/ray-sgk` | `lock: free …`; `Ray processes of /tmp/ray-sgk: none` | 0 |

Row 5 is a mistyped look, not a lock operation (log, deviation 5).

**The launch** is §B.1's command, verbatim, with `--register r8-reference --campaign handover
--prompt 12`, `--atol 1e-32 --rtol 1e-7 --cpus 6`, into
`var/datastores/handover-12-reference.sqlite`, output to `var/runs/handover-12-reference.launch.log`.
The `sh` wrapper was pid 21715 (`$!`); the driver, its child by `pgrep -P 21715`, pid **21718**, the
pid `status.json` records.

**The run** `handover-12-r8-reference-20261025T210551`:
- created 21:05:51 UTC, fingerprint taken 21:08:41, driver gone by 21:09:24; launch log ends
  `** driver exit status 0`; no `Traceback` in the launch log or the run's `stdout.log`;
  `stderr.log` empty;
- `status.json`: `done`, `exit_code` 0, pid 21718, host and boot recorded;
- manifest: campaign `handover`, prompt `12`, `git_head` `8a02f0c4…`, `git_dirty` false, Python
  3.13.16, script `RunDrivers/quadsource_atol_sweep.py`, SHA-256 `5b984c75…` (R7's launcher);
- stages: Tk numeric 0.99 s, Tk WKB 0.29 s, QuadSource 0.92 s, Gk numeric 23.1 s, Gk WKB 63.0 s,
  GkSource 23 s, policies 8.8 s, QuadSourceIntegral 40.5 s (864 of 1728 triples after the triangle
  filter, as R3);
- `fingerprint_sidecar`: "not written … there is none". The store has no sidecar, by design of the
  launcher's `--build`; the fingerprint is in `status.json`.

### 5.3 The fingerprint (prompt §6.4)

**R8's fingerprint, the rehearsal reference from here on:**
`1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7` (format 6, encoding 1, 21
classes, problems none), taken by `handover-12-r8-reference-20261025T210551` at `8a02f0c`,
`git_dirty` false.

`compare_with_recorded.py var/datastores/handover-12-reference.sqlite --against
var/handover-06/recorded/rrk-v020-adoption-03-reference-20261018T232838`
(`out/fingerprint_against_R3.txt`), exit **1**, verdict `DIFFERENT`:

```
compare_fingerprints(…rrk-v020-adoption-03-reference-20261018T232838/status.json, store): 1 difference(s)
  QuadSource: tag set [LargestSourceRedshift_6.8788e+13, OutsideHorizonEfolds_e3, Run_default,
  SmallestSourceRedshift_250, SourceGridConstruction_2, SourceRedshiftGrid_1159_7139cb9c,
  TkOneLoopDensity] differs: 6 recorded, 6 now
```

**One class differs, `QuadSource`; no other class differs. `QuadSourceIntegral` does not differ.**
Every class other than `QuadSource` has R3's digest.

Why `QuadSourceIntegral` cannot differ in the fingerprint: format 6's `inventory_spec()` for
`QuadSourceIntegral` (`ObjectFactories/QuadSourceIntegral.py`) is its identity only (parents and
tags, "No validated flag and no value table"), so it does not see `total`, `metadata` or the
partition. Its `source` parent resolves cross-shard to the `QuadSource` row's reference digest,
which is `reference_digest(key, tags)` (`datastorekit/store_inventory.py:167`), and does not cover
that row's value count. The `QuadSource` rows' identities are unchanged; only their
`QuadSourceValue` count moved, which `QuadSource`'s own spec covers (`values=("QuadSourceValue",
…)`). So the fingerprint records B1's `QuadSource` change and cannot record its `QuadSourceIntegral`
change. §5.4 shows the integrals did change.

### 5.4 R8's rows, read-only (prompt §6.5)

`scripts/r8_rows.py`, every store file opened `mode=ro` (`out/r8_rows.txt`, exit 0, problems 0).
R3's store is retired. The `QuadSource` comparison is therefore against the R4 store's
`extract_tensor_source_data` CSVs (`var/handover-06/extract/…`, handover 06's resumed store, whose
fingerprint equalled R3's `39809dca`).

**`QuadSourceIntegral`.** 864 rows; **864** carry `partition.handover` with `Tq` and `Tr`, each with
`z_init`, `z_switch` and `handover_continuity`. **0** rows have any metadata key containing `clamp`,
so none has `clamp_gaps_log1pz` or `max_clamp_gap_log1pz`. Each $k$'s continuity record is the same
in every row that carries it.

**The switch node.** For each $k$, `z_switch` is the first source-grid node below `z_init`, with no
source-grid node between them:

| $k$ / Mpc | $z_{\rm init}$ | $z_{\rm sw}$ |
|---|---|---|
| 1e5 | 1724800979.6868515 | 1712616453.6720948 |
| 3.1623e5 | 5454303327.635345 | 5420993175.39622 |
| 1e6 | 17248025285.92749 | 17159222629.609928 |

**`handover_continuity`**, recorded, not judged. Node 0 is $z_{\rm sw}$; nodes 1 and 2 are the
overlap nodes:

| $k$ / Mpc | node | $z$ | `dT_rel` | `dTprime_rel` |
|---|---|---|---|---|
| 1e5 | 0 | 1712616453.6720948 | -3.153510e-07 | 5.674049e-06 |
| 1e5 | 1 | 1673600269.6012518 | -5.545501e-06 | 2.165710e-05 |
| 1e5 | 2 | 1635472937.5650687 | -1.603827e-05 | 3.079759e-05 |
| 3.1623e5 | 0 | 5420993175.39622 | -2.354245e-07 | 4.915088e-06 |
| 3.1623e5 | 1 | 5297494147.272005 | -5.207091e-06 | 2.110052e-05 |
| 3.1623e5 | 2 | 5176808627.568508 | -1.552583e-05 | 3.059522e-05 |
| 1e6 | 0 | 17159222629.609928 | -1.672076e-07 | 4.151410e-06 |
| 1e6 | 1 | 16768307671.85978 | -4.878320e-06 | 2.053187e-05 |
| 1e6 | 2 | 16386298391.685555 | -1.501807e-05 | 3.037574e-05 |

By node, over $k$: node 0, $|\Delta T_{\rm rel}|$ 1.67e-07 to 3.15e-07 and $|\Delta T'_{\rm rel}|$
4.15e-06 to 5.67e-06; node 1, 4.88e-06 to 5.55e-06 and 2.05e-05 to 2.17e-05; node 2, 1.50e-05 to
1.60e-05 and 3.04e-05 to 3.08e-05. Every `dT_rel` is negative and every `dTprime_rel` positive, and
both grow monotonically away from the switch node.

**`QuadSource`**, R8 against R4:

| (q, r) / Mpc | R8 nodes (`z_samples`) | R4 nodes | added | removed | common nodes differing | R8 lowest $z$ | R4 lowest $z$ |
|---|---|---|---|---|---|---|---|
| (1e5, 1e5) | 478 (478) | 475 | 3 | 0 | 0 of 475 | 1635472937.5650687 | 1752542211.3413053 |
| (1e5, 3.1623e5) | 428 (428) | 425 | 3 | 0 | 0 of 425 | 5176808627.568508 | 5547371302.491309 |
| (1e5, 1e6) | 371 (371) | 368 | 3 | 0 | 0 of 368 | 16386298391.685555 | 17559250880.554787 |
| (3.1623e5, 3.1623e5) | 428 (428) | 425 | 3 | 0 | 0 of 425 | 5176808627.568508 | 5547371302.491309 |
| (3.1623e5, 1e6) | 371 (371) | 368 | 3 | 0 | 0 of 368 | 16386298391.685555 | 17559250880.554787 |
| (1e6, 1e6) | 371 (371) | 368 | 3 | 0 | 0 of 368 | 16386298391.685555 | 17559250880.554787 |

The top of every row is unchanged, at $z$ = 68787986538016.445. Every row gained exactly
`HANDOVER_OVERLAP_NODES` + 1 = 3 nodes, 18 in all. They are the larger-$k$ mode's $z_{\rm sw}$ and
its two overlap nodes. At every node R4 already had, all nine value columns are bit-identical. All
six rows are `validated`.

**Supplement: `QuadSourceIntegral` values against R4** (`scripts/compare_qsi_extract.py`, reading
only the readers' CSVs; `out/qsi_values_R8_vs_R4.txt`). The readers draw their triples unseeded, so
only one triple has an `integral_data.csv` in both extracts: $(k,q,r) = (3.1623{\rm e}5, 1{\rm e}5,
3.1623{\rm e}5)$, 96 `z_response` rows.

| column | bit-identical | largest $\lvert\Delta\rvert$ / max$\lvert$value$\rvert$ over z_response | where |
|---|---|---|---|
| `total` | 33/96 | 1.82e-03 | $z_{\rm resp}$ = 5297494147.27 (an overlap node) |
| `numeric_quad` | 93/96 | 1.89e-03 | the same |
| `WKB_Levin` | 34/96 | 1.83e-02 | $z_{\rm resp}$ = 1329131612.30 |
| `analytic_rad` | 96/96 | 0 | |

The largest pointwise relative difference, 0.80, is in `total` and `WKB_Levin` at $z_{\rm resp}$ =
47846.4, where the integral is small. The 33 bit-identical `total`s are exactly the rows with
$z_{\rm resp} \geq$ 9210251401.997; every row with $z_{\rm resp} \leq$ 6985073578.495 differs.
That includes the one row above both $z_{\rm init}$ (5.45e9) and below 9.2e9, $z_{\rm resp}$ =
6.985e9, where `numeric_quad` moves by 3.0e-13 of max|value|. That is consistent with the source
spline's coefficients changing a little above the added nodes, but it was not traced. So B1 changed
the integrals, as intended, and the fingerprint (§5.3) does not show it. These are recorded, not
judged.

### 5.5 The readers (prompt §6.6)

`bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh var/datastores/handover-12-reference.sqlite
var/handover-12/extract`, with the lock held (`out/readers.txt`), 21:12:47 to about 21:15:02 UTC.
Overall exit **0**.

| reader | exit | wall | lookups | files written (R7; R3) | Traceback | store files after |
|---|---|---|---|---|---|---|
| `extract_Gk_data.py` | 0 | 26 s | 116 | 531 (441; 468) | none | identical |
| `extract_GkWKB_data.py` | 0 | 44 s | 348 | 2436 (2436; 2436) | none | identical |
| `extract_GkSource_data.py` | 0 | 18 s | 10 | 160 (180; 156) | none | identical |
| `extract_TkWKB_data.py` | 0 | 15 s | 3 | 27 (27; 27) | none | identical |
| `extract_tensor_source_data.py` | 0 | 13 s | 6 | 27 (27; 27) | none | identical |
| `extract_QuadSourceIntegral_data.py` | 0 | 16 s | 24 | 76 (73; 70) | none | identical |

The lookups equal R3's and R7's. File counts differ only where a reader draws an unseeded subsample
(`[10-the-readers-draw-their-work-lists-unseeded]`). After each reader, 0 Ray processes of this
checkout's venv were left. No log has an authentication or token line.

**The store, before every read and after every read** (`state/store_sha256_before_reads.txt`,
`state/store_sha256_after_reads.txt`; `diff` empty):

```
531073a9940233ee260f7fb0bb9289fa1ceea15e0bbfcaa8c142646ebd5d3759  handover-12-reference.sqlite        36864
c8cb8d7c07936bf384474d68f470256fa9e6ae3fc50ab454ec28b99f7ba3c8d5  handover-12-reference-shard0000.sqlite  24096768
836df59f6067688fba9f37e4444fdd8722605e809279a244ce6a0957507d5b78  handover-12-reference-shard0001.sqlite  24207360
82119003fb3d2b834c1660b2899442697e884fc21282cde2d6766aed4df7ea74  handover-12-reference-shard0002.sqlite   1077248
fdf375f5c674d21056dac8cb6cd018fb83f7e4f03e4d37403c120a5326ff6d59  handover-12-reference-shard0003.sqlite  23633920
```

No `-journal`, `-wal` or `-shm` file at any point. "Before" was taken before
`compare_with_recorded.py`, so it covers §6.4, §6.5 and §6.6.

### 5.6 The lock released (prompt §6.7)

`ray-processes` listed nothing (exit 0). `release` exited 0. `status` then said
`lock: free (/Users/ds283/.raypoolkit/rehearsal-lock does not exist)` and
`Ray processes of /tmp/ray-sgk: none` (exit 0). The registry ends with 12 runs, all finished;
R8 is `done`.

### 5.7 Against prompt §7.3

- R8 ends `done`: **yes**.
- It differs from R3 in `QuadSource` and `QuadSourceIntegral` only: **yes**, in the sense that it
  differs in `QuadSource` alone, which is within that set. The dispatch expected `QuadSourceIntegral`
  to differ as well (§5.3; log, deviation 7).
- No row records a clamp gap: **yes**, 0 of 864.
- The readers exit 0: **yes**, and the store is byte-identical.
- The lock is released: **yes**.
