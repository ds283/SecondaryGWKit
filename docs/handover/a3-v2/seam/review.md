# Review — handover prompt 11, A3 v2's seam record

**Reviewer:** the orchestrator session on the Mac Studio (`mps026298`), Opus 5.5, 2026-10-25.
**Dispatch SHA:** `757ef016fc37be996c342d162c05f17b3c430e4d`. **Executing agent:** one fresh-context
subagent on Sonnet. **Verdict: PASS.** All seven checks of `orchestrator/prompt-11.md` §3 hold. No
stop condition of prompt §3 was met, and nothing was repaired.

## Before dispatch (note §1)

The Gate, re-read by me before dispatch:

- `git rev-parse HEAD` returned `757ef016fc37be996c342d162c05f17b3c430e4d`.
- `git status --short` returned empty (rc 0).
- `./venv/bin/python -m runregistrykit list` showed 10 runs, all finished: "10 finished; 0 with no
  host recorded". Nothing was `running` or stale.
- `runregistrykit store show var/datastores/handover-A3-v2-lambdacdm.sqlite` reported kind
  `registry`, problems none, fingerprint digest
  `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`, and the run
  `handover-07-a3-v2-20261022T211155` `done`.
- `ls -d var/handover-11` returned "No such file or directory".

The marker was `touch`ed in the orchestrator's scratchpad at 2026-10-25 18:12:07 UTC, immediately
before dispatch. No suite baselines were taken (note §1.3).

**An incident during the wait, recorded for the laptop.** At about 18:12, a message in the form of
the agent's completion notice reached this session, before the agent had finished. It carried a
full set of results:
- `Tq` 8 990, `Tr` 9 090, `source` 576 non-zero sub-intervals;
- 3712 seam rows;
- the `minimal` row at $k = 3\times10^8$, $z_{\rm response} = 0.1$.

When I checked, `var/handover-11/` did not yet exist. Its first files appeared at 18:12–18:13, and
held only the Gate, step 1 and the policy count. I treated that message as not genuine and used
none of its figures. The agent's actual report arrived later, and agrees with what is on disk.
**Every figure in that early message is wrong.** Every figure below was read from the records or
recomputed by me.

## Check 1 — Scope: PASS

```
$ git rev-parse HEAD; git status --short
757ef016fc37be996c342d162c05f17b3c430e4d
(empty; rc 0)
$ find var -newer <marker> | awk -F/ 'NF>=2{print $1"/"$2}' | sort | uniq -c
     23 var/handover-11
     31 var/levin-phase-floor-01
$ find var -newer <marker> | grep -vE '^var/(handover-11|levin-phase-floor-01)(/|$)'
var
```

The only other entry is the directory `var` itself, whose mtime changed when the two
subdirectories were created. No store file, sidecar, or file under the A3 v2 run directory is
newer than the marker. I did not open `var/levin-phase-floor-01/` beyond counting its entries.

## Check 2 — The store and the log: PASS

```
$ diff state/store_files_before.txt state/store_files_after.txt
(no output; rc 0)
```

Both files hash seven files with SHA-256, size and mtime: the primary, the four shards, the sidecar
and `stdout.log`. I re-hashed all seven with `shasum -a 256` and `/usr/bin/stat`; each matches the
before file:

- primary `1542e05c…6522`, 36864 B, mtime 2026-10-22 21:12:03;
- shard0000 `b311698f…e6f5`, shard0001 `c5e341ce…d0e`, shard0002 `175a6731…63c6` and shard0003
  `76b83fb3…0cb1`, all with mtime 2026-10-24 21:56:16;
- sidecar `3201f6a5…def6`, 10519 B;
- `stdout.log` `4926b6fc…7945`, 19545 B, mtime 2026-10-24 21:56:27.

```
$ ./venv/bin/python -m runregistrykit store fingerprint var/datastores/handover-A3-v2-lambdacdm.sqlite
digest:   0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56
problems: none
recorded: matches the fingerprint taken 2026-10-25T00:34:14+01:00 by a person, at b90752dda46e57d588f8107cbd4d4557134fadb2
(rc 0; run without --write, 10.7 s)
$ ls var/datastores/ | grep -E 'journal|wal|shm'
(no output; rc 1)
```

The agent's log records one piece of unintended drift here (log item 6). Its first
`storefiles.sh` run used GNU `stat` and wrote garbage into `store_files_before.txt`. It was
corrected and re-run at 18:13, before the census ran at 18:15. The store files' mtimes all predate
this session, and my own hashes match, so the before record stands.

## Check 3 — The policy summary: PASS

I compared `out/policy_11.txt` lines 89–102 with `sed -n 85,105p` of the log: verbatim, line for
line. The block runs from `** GkSourcePolicyData SUMMARY STATISTICS` at line 89 to
`complete: 725 instances = 100.00% of this type, 62.50% of total` at line 102.

I made an independent table count myself, `mode=ro` on the four shards, grouped by
`(type, quality)` and decoded with `ObjectFactories/GkSourcePolicyData.py:13-41`:

| type | quality | log | agent's table | my count |
|---|---|---:|---:|---:|
| numeric | complete | 363 | 363 | 363 |
| mixed | complete | 71 | 71 | 71 |
| mixed | minimal | 1 | 1 | 1 |
| WKB | complete | 725 | 725 | 725 |
| **total** | | **1160** | **1160** | **1160** |

There is no `fail` and no `incomplete` row.

The one non-`complete` row is shard 2, `GkSourcePolicyData` serial 460. My own join, through
`GkSource.z_response_serial` and `wavenumber_exit_time.wavenumber_serial`, gives:
- $k = 3091682.0425413917$/Mpc;
- $z_{\rm response} = 39637290381.164085$;
- `crossover_z` $= 42472907249.1825$.

That equals the agent's identification.

## Check 4 — The population: PASS

I counted the distinct `(shard, serial)` pairs in `subintervals_11.csv` myself, with Python's
`csv` module: **9280**. The CSV has 19 289 records, plus its header. By band: seam 6080, mid 2112,
low 1088. `seam_census_11.txt` (a) gives 9280 rows at (1e-32, 1e-7), 9280 partitions parsed, and
2320 per shard.

## Check 5 — The gap census: PASS

**The field.** I read `scripts/seam_census.py`. The gap is
`s["clamp_gaps_log1pz"][factor]` (lines 211, 256–264, 276, 286, 297). `max_clamp_gap_log1pz` is
read once, as `allow` (line 144). It is used only as the allowance: the over-allowance test (213),
the near test (264), and `step = allow / HANDOVER_CLAMP_MAX_GRID_STEPS` (166). The constant is read
from `ComputeTargets/QuadSourceIntegral.py` by regex (line 57), giving 1.5.

**My recomputation from the CSV** agrees with `seam_census_11.txt` lines 48–50 in every figure:

| factor | non-zero sub-intervals | rows ≥1 | median | max | census file |
|---|---:|---:|---:|---:|---|
| `Tq` | 12 880 | 6440 | 1.1516e-02 | 2.2593e-02 | 12880 / 6440 / 1.152e-02 / 2.259e-02 ✓ |
| `Tr` | 13 582 | 6791 | 1.1516e-02 | 2.2593e-02 | 13582 / 6791 / 1.152e-02 / 2.259e-02 ✓ |
| `source` | 423 | 423 | 1.4759e-02 | 2.2593e-02 | 423 / 423 / 1.476e-02 / 2.259e-02 ✓ |

- Gaps above the row's allowance: **0**. Gaps at or within 1 % of it: **0**. The allowance ranges
  from 3.4132e-02 to 3.4548e-02.
- In mean source-grid steps, the median is 0.50 for `Tq` and 0.51 for `Tr`, and the maximum is
  0.981 for all three factors.
- The 2026-09 figures stand beside the new ones in `seam_census_11.txt` and in `verification.md`'s
  comparison table:
  - `Tq` on 1666, `Tr` on 1798;
  - median 1.2e-02, max 2.2e-02;
  - about 1.0 mean grid step;
  - `source` on 108.

**Observation, not adjudicated.** In $\log(1+z)$ the median and maximum reproduce 2026-09's. In
mean grid steps they do not: 2026-09's "about 1.0 step" sits beside a median of about 0.5 and a
maximum of 0.98 here. `verification.md` line 127 sets the two side by side without saying which
2026-09 statistic "about 1.0" referred to. Whether the grid step differs between the two
populations is for the laptop to settle when it narrows `[12-…]`.

## Check 6 — The oracle comparison: PASS

The comparison is in `seam_census_11.txt` (d).

- **Seam band only.** The population is the 6080 rows with $z_{\rm response} > 10^5$. All 6080
  were used.
- **Strata as step 4 (d) says.** Each is tabulated by $z_{\rm response}$ decade, with n, median and
  max:
  - Stratum 1b counts the clamped factors among `Tq` and `Tr` (0, 1, 2), which is the prompt's
    stratum.
  - Stratum 2 counts the oscillatory factors in the last Levin sub-interval (0–3).

  The agent also added, as a declared IMPLEMENTATION CHOICE (log item 3):
  - Stratum 1, which counts `source` as well (0–3);
  - a crossing of the two strata;
  - the grid values nearest 7.63e9 and 1.45e9.
- **Exclusions are counted.** "excluded (analytic_rad zero/non-finite or total non-finite): 0".
- **The limits are stated** in the output: constant-$w$ oracle at $b = 0$; its own deficit towards
  equality; the LG-phase term; the quadrature error. Decades $10^5$–$10^7$ are marked `*` and
  "reported, not read as the clamp's error". `verification.md` states the same limits.
- **Headline:**
  - The overall median is 5.31e-04 and the maximum 96.9.
  - Rows with no clamped factor: n = 2489, median 4.98e-06.
  - Rows with Tq and Tr both clamped: n = 3240, median 5.50e-03.
  - Rows with one of Tq, Tr clamped: n = 351, median 6.57e-02.
  - At $z_{\rm response} = 8.37\times10^9$, the grid value nearest 2026-09's 7.63e9:
    - unclamped 9.6e-06;
    - one clamp 1.6e-01;
    - two clamps 9.0e-03.

  The agent did not interpret this. It records in `issues.md` item 2 that the no-clamp stratum is
  the one-sub-interval rows, which are present only at $z_{\rm response} \ge 10^9$, so the strata
  are tied to $z_{\rm response}$.

## Check 7 — The records: PASS

```
$ diff <(grep -E '^#' prompts/handover/logs/10-reread-a3-v2.md) <(grep -E '^#' var/handover-11/log.md)
1c1   (the title line only; every other heading is identical)
```

`log.md` classifies seven deviations:
1. STRUCTURALLY REQUIRED: step 1 became a check, for the parallel dispatch. This is the step 1
   no-op.
2. IMPLEMENTATION CHOICE: a separate `policy_11.py`.
3. IMPLEMENTATION CHOICE: extra strata and columns.
4. IMPLEMENTATION CHOICE: how the census is run, and where it finds the constant.
5. IMPLEMENTATION CHOICE: extra read-only checks.
6. UNINTENDED DRIFT: the first `storefiles.sh` run wrote garbage.
7. UNINTENDED DRIFT: a stray `diff rc 0` line at the start of `out/store_after_11.txt`.

`verification.md` opens `## 5. The seam record`. All eleven of its links have the
`a3-v2/seam/…` form, matching §4's `a3-v2/reread/…`.

`issues.md` exists, with five items:
1. the `minimal` row, which narrows `[03-…]` / §2 (p);
2. the strata being tied to $z_{\rm response}$;
3. the narrowing of `[12-handover-clamp-error-in-production]`;
4. 3798 rows with a `Tq` `skipped` breakpoint, of `log_width` 0.0, at $q = r$;
5. the large oracle maxima.

**One point for the laptop.** The Gate's "pushed" item was checked without fetching. It used the
local `origin/handover-remedial` ref and `git branch -r --contains`, and also
`git ls-remote origin refs/heads/handover-remedial`. That is a network read, not a write, and it
writes no ref. It is recorded in `out/gate_11.txt` line 20 and `log.md` deviation 1.

## For the laptop (note §4)

- `out/subintervals_11.csv` is 5 050 755 B. It is under 10 MB, so no gzip is needed.
- `extract/` does not exist for this prompt; copy everything.
