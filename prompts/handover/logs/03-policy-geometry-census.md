# Log 03 — The policy-geometry census on A3 v2

> **Written on the Mac Studio, committed on the laptop** by the orchestrator (prompt §6). A path
> `var/handover-03/…` below is that directory on the Mac Studio; its `out/`, `scripts/` and `state/`, with
> `issues.md`, are committed at `docs/handover/a3-v2/policy-geometry/`, and `POLICY-GEOMETRY.md` is
> `docs/handover/POLICY-GEOMETRY.md`.

**Prompt:** prompts/handover/03-policy-geometry-census.md (read at the dispatch SHA `c8ca155`)
**Commit:** none on the Mac Studio (prompt header; dispatch rules). The checkout was **not** moved: it was at
`c8ca1559fcc53fdb88998f170257635431fb85ca` at the start, after the census and at the end (deviation 1).
**Model:** Claude Sonnet 5.5, in a Claude Code session on the Mac Studio (`mps026298`), as a dispatched
sub-agent.
**Date:** 2026-10-25 (22:36–22:56 UTC).
**Result:** COMPLETE WITH DEVIATIONS. No stop condition of §7 was hit, and acceptance 1–9 hold. Two readings are
flagged for the reviewer, because the prompt leaves them to judgement and the evidence is near a stop:
(a) §7's second stop condition (the shape against verification §5.5): the census accounts for the difference
as a denominator (§3 of `POLICY-GEOMETRY.md`, and "The census against §5.5" below) and does not treat it as a stop;
(b) §7's first condition on §2.1 item 1: the half-e-fold band is present on all 72 `mixed` rows, so the stated
prediction is not falsified, but the overlap is 9 to 17 e-folds wide on 71 of the 72, not half an e-fold on all but
"some", and the `complete` seam is at one node near the top of the grid (items 4 and 5 of `issues.md`).

The record is `POLICY-GEOMETRY.md`.

## What shipped

Nothing tracked, and no commit. Under `var/handover-03/`:
- `out/`:
  - `gate_03.txt`, the Gate (with a note appended at its end);
  - `pre_census_03.txt`, `HEAD`, `git status --short`, the registry and the script's SHA-256 just before the run;
  - `policy_geometry_03.txt`, **the census** (the command, its output and its runtime): 1890 lines, the
    1160-row table last;
  - `step_check_03.txt`, a supplement (deviation 4);
  - `exploration_03.txt`, the two schema-reading scripts, re-run and recorded (deviation 8);
  - `black_03.txt`, `black --check --diff` on every script;
  - `store_after_03.txt`: the before/after compare, the fingerprint (no `--write`), the registry and `HEAD`;
  - `dev_runs/`: the stdout of the three earlier versions of the census script, and the records that went with the
    last of them (deviation 7).
- `state/`: `store_files_before.txt`, `store_files_after.txt`.
- `scripts/`: `policy_geometry.py` (SHA-256 `fa010147f92397853a6d19049424117f6d29afd4b5b378cbf4942985db390b4c`,
  the script that ran), `step_check.py`, `exploration/explore.py` and `explore2.py`, `gate_03.sh`, `rec.sh`,
  `storefiles.sh`, and the document's three inputs: `POLICY-GEOMETRY.template.md`, `answer.md` and
  `assemble_doc.py`.
- This log, `POLICY-GEOMETRY.md` and `issues.md`.

**Outside `var/handover-03/`:** nothing was written. In particular nothing under `var/levin-phase-floor-04/` (not
read either), nothing in the worktree `SecondaryGWKit-lpf04-98e68d3`, none of the store's files, its sidecar, or any
run directory. The store's SQLite files were opened `mode=ro`. No git command that writes was run. No Ray, no lock.

## The datastore, its fingerprint and the grid tags the census ran on

- `var/datastores/handover-A3-v2-lambdacdm.sqlite`, four shards and the sidecar `…manifest.json`; fingerprint
  `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`, `store show` kind `registry`, problems none,
  and recomputed after the census and again at the end by `store fingerprint` without `--write`: equal to the sidecar's.
- **Grid tags carried by all 1160 `GkSource` parents** (`GkSource_tags`, and the sidecar's tag set): `GkOneLoopDensity`,
  `LargestSourceRedshift_2.0636e+16`, `ResponseRedshiftGrid_145_c51d43ac`, `ResponseSparsenessZ_12`, `Run_default`,
  `SmallestSourceRedshift_0.1`, **`SourceGridConstruction_2`**, `SourceRedshiftGrid_1740_8d1d43b6`. The `BackgroundModel`
  rows they name carry `source_grid_construction = 2`.
- `CosmologyConcepts/wavenumber.py:61` `SOURCE_GRID_CONSTRUCTION_VERSION = 2`, whose comment above it says version 2 is
  version 1 plus the base density set by the measured curvature criterion of prompt 15 of `qcd-background-audit`. The
  script reads the constant by `ast` from the text and prints it.

## Deviations from the prompt

1. **STRUCTURALLY REQUIRED — the checkout check replaces 10's and 11's step 1.**
   - **The rule:** the Gate says the checkout was fast-forwarded by the laptop orchestrator before dispatch; the
     prompt §8 says "since the checkout was moved for you".
   - **What was so:** `levin-phase-floor` 04 had already started, so the checkout was **not** moved (the dispatch's
     correction of the orchestrator note: it was at `c8ca155…`, not `8a02f0c`). The Gate was therefore a check only.
   - **Done:** `HEAD` is `c8ca155…` and `git status --short` is empty at the start (`gate_03.txt`), before the
     census (`pre_census_03.txt`) and after it (`store_after_03.txt`). The refresh commit `f1a3888` is an ancestor of
     `HEAD` (`git merge-base --is-ancestor`). "The dispatch commit is pushed" was checked without fetching:
     `git branch -r --contains HEAD` lists `origin/handover-remedial`, and `git rev-parse @{u}` equals `HEAD`.
     `git diff 98e68d3 HEAD --stat -- <the four files>` is empty.
2. **The Gate's registry exception for `levin-phase-floor` 04 did not arise.** `runregistrykit list` showed 12 runs,
   12 finished, nothing running or stale and no `levin-phase-floor-04-*` run, at the Gate, before the census and after
   it. So no `STRUCTURALLY REQUIRED` deviation was needed for it. This is recorded because the prompt says to record
   it "if it arose".
3. **IMPLEMENTATION CHOICE — what the script imports and how it reads constants.**
   - **The rule:** "`sqlite3` and `json` only, nothing imported from this repository's packages", "following
     `seam_census.py`".
   - **Done:** `seam_census.py` also uses `numpy`, and so does this script (percentiles, `interp`); it is in the venv.
     Nothing from the repository is imported. The constants are read from the source **text** with `ast.parse` and
     `ast.literal_eval`, where `seam_census.py` uses a regex: `MIN_SPLINE_DATA_POINTS`, `CLEARANCE_*` and
     `CLASSIFICATION_BANDS` (`ComputeTargets/GkSourcePolicyData.py`), `_type_deserialize` and `_quality_deserialize`
     (`ObjectFactories/GkSourcePolicyData.py`), `SOURCE_GRID_CONSTRUCTION_VERSION`
     (`CosmologyConcepts/wavenumber.py`), `DEFAULT_FLOAT_PRECISION` (`config/defaults.py`). All are printed.
4. **IMPLEMENTATION CHOICE — what the census reports beyond the prompt's table.** Each is read-only, and each answers
   something the prompt asks or something the table cannot say alone:
   - the band ladder of `_classify_crossover` **replicated** on the stored geometry, and checked against the stored
     `crossover_z` node and `quality` (72 of 72), for "the tie to §3.2";
   - `numeric_smallest_z` and `primary_WKB_largest_z` **recomputed** from the stored value rows and checked against
     the GkSource row (1160 of 1160), because the whole census counts from them;
   - the spline data points the policy counted (against `MIN_SPLINE_DATA_POINTS`), and holes on each side;
   - the thresholds expressed in intervals, the binned relation of §4 and a Spearman line (which carries no weight:
     71 rows tie), for "the relation between the two clearances";
   - table (f) and the probe table (g) in the shape comparison, and the k-decade, per-k and per-z-decade tables, for
     §2.2's "say which of the three accounts for it";
   - the hole details (the node where each hole starts and where the island begins, in e-folds), and the
     `N(z_response)` window of the `mixed` rows per $k$;
   - the check that `primary_WKB_largest_z` is the top node on 71 of the 72 rows, and that `crossover_z` takes one
     value on the 71 `complete` rows;
   - `step_check.py` (`out/step_check_03.txt`), a small separate script that says where the source grid is finer than
     its usual step, because the interval counts depend on it.
5. **IMPLEMENTATION CHOICE — the `band` column.** The prompt asks for "which of the ten bands the stored `quality`
   implies". A quality implies a set (`acceptable` is bands 1–3, `marginal` 4–8), so the column is the first band the
   replicated ladder reaches, and the script checks that the stored `quality` is one of the qualities that band set
   implies (72 of 72). In the per-row table a point whose own first-satisfied band differs from the ladder's would be
   shown as `(point: n)`; none is.
6. **IMPLEMENTATION CHOICE — three definitions.**
   - **E-folds.** The prompt says to convert redshifts to e-folds only as $z \to \log(1+z)$ by interpolating the stored
     `z_exit_subh_eN` columns. That is how the landmarks' $N$ is computed (through `z_exit_suph_e5` to `z_exit_subh_e6`,
     with `<-5` and `>6` for the outside of the table). For the **overlap width**, $\log(1+z_{\rm WKB}) - \log(1+z_{\rm num})$
     is reported directly as "e-folds", after printing that each landmark step $\log((1+z_{eN})/(1+z_{e(N+1)}))$ is 1.00000
     for all eight $k$ (the stored columns make a step of $\log(1+z)$ an e-fold here).
   - **The lowest decile** is "value $\le$ the 10th percentile (numpy, linear)". It is degenerate on this store (82 is the
     10th percentile, 71 rows tie, all 72 qualify), and the output and the document say so.
   - **The band presence test** (§2.1 item 1) counts source-grid nodes with $z_{e4} \le z < \sqrt{z_{e3}z_{e4}}$ that carry
     both a numeric and a WKB value. `main.py:2056` includes $z_{e4}$ itself (`z_source.z > z_e4 - DEFAULT_FLOAT_PRECISION`), so
     the lower edge is inclusive; the upper edge is exclusive as `main.py:466-468`'s `>=` on the other side makes it.
7. **UNINTENDED DRIFT — the census script was edited after its first runs, and only the last version's run is the
   record.** Black was run before the first run, as the prompt asks. Three earlier versions of the script were then run
   as development runs, to the session scratchpad (stdout), then the first two copied to `out/dev_runs/`
   (`dry1_first_version.txt`, `dry2_second_version.txt`; each has its own script SHA-256 in its header) and the third
   (the first "recorded" run, SHA `577a4166…`) moved there as `dry3_third_version_577a4166.txt` with its pre-run, black and
   store-after records, when I added the e-fold, hole-detail and top-of-grid outputs and a probe table. `out/policy_geometry_03.txt`
   is the run of the final script, `fa010147…`, **byte-identical to the file in `scripts/`**, which is `black --check`
   clean. The stores were not touched by any of them (`store_files_before.txt` equals `store_files_after.txt`).
8. **UNINTENDED DRIFT — unrecorded reads, and the repair.** The two schema-reading scripts
   (`scripts/exploration/explore.py`, `explore2.py`) were run first into the conversation only. They were black-formatted
   and **re-run with their output saved** (`out/exploration_03.txt`). The `sqlite3` command-line schema dumps (`.schema`,
   `pragma table_info`) and the `sed`/`grep` reads of source files and campaign documents were not saved: they are
   reads of the tree and of the store's schema, not measurements.
9. **IMPLEMENTATION CHOICE — how `POLICY-GEOMETRY.md` was made.** Its tables are blocks of the census's stdout, cut by
   markers the script prints (`<!-- section:NAME -->`), and whole script sections for the grid, tags, ladder tie, hole
   census, exit-time check and stop-condition list. `scripts/assemble_doc.py` does the cutting from
   `POLICY-GEOMETRY.template.md` (the prose, written by hand) and `answer.md` (§0). Each block carries a
   `<!-- generated by … -->` marker with the script's SHA-256.
10. **UNINTENDED DRIFT — one `Edit` call to a mistyped path** (`…/SecondaryGWKid/…`, identical old and new strings) was
    rejected and wrote nothing. Noted for completeness.
11. **IMPLEMENTATION CHOICE — the Gate's "there is no `var/handover-03/`".** Checked first (`ls` answered "No such file or
    directory"), then the directory was created for the Gate's own output, so `gate_03.txt`'s closing `ls var` lists it. A
    note at the end of `gate_03.txt` says so.

## Verification performed

All of the following was run, and its output is saved.

- **Gate** (`gate_03.txt`): host `mps026298`; `HEAD` `c8ca155…`, branch `handover-remedial`, clean; pushed (above);
  `f1a3888` an ancestor; the four-file diff against `98e68d3` empty; 11's board row reads "reviewed"; registry 12 runs, 12
  finished; `store show` kind `registry`, problems none, fingerprint `0e32f24a…`; no `var/handover-03/` at the start.
- **Acceptance 1.** `./venv/bin/python -I var/handover-03/scripts/policy_geometry.py
  var/datastores/handover-A3-v2-lambdacdm.sqlite`, with its output and runtime, in `out/policy_geometry_03.txt`
  (runtime 1.6 s, real 1.76 s; "Hard checks failed: 0").
- **Acceptance 2.** `state/store_files_before.txt` and `store_files_after.txt` are identical (`diff` rc 0; the
  after-file was taken twice, the second after `step_check.py` and the document); `runregistrykit store fingerprint`
  (no `--write`) gives `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`, problems none, "matches the
  fingerprint taken 2026-10-25T00:34:14+01:00".
- **Acceptance 3.** The type/quality count equals prompt 11's table in every cell: `numeric`/`complete` 363, `mixed`/`complete`
  71, `mixed`/`minimal` 1, `WKB`/`complete` 725, total 1160; `fail` or `incomplete`: 0. The shape against verification §5.5
  is in the next section.
- **Acceptance 4.** `numeric_intervals` over the 72 `mixed` rows: min 6, 5th 308, median 481, 95th 664, max 664;
  `WKB_intervals`: 1, 82, 82, 82, 82. **One row has fewer than two on a side: shard 2, serial 460, the `minimal` row.** Its
  record, including its stored metadata, is tabulated individually.
- **Acceptance 5.** `POLICY-GEOMETRY.md` exists; §0 answers D5 in one paragraph; §10 is "what this does not cover".
- **Acceptance 6.** §2.1 item 1 and item 2 are answered, each with README §2 (q)'s wording quoted beside the
  measurement, item 2 as a proxy (A3 v2 has no `incomplete` row).
- **Acceptance 7.** The non-contiguity count and the exit-time check (below, and §7 of the document).
- **Acceptance 8.** `black --check --diff` is clean on the script and on the other scripts (`out/black_03.txt`); the script
  that ran is the script recorded (SHA-256 above, in the header of the output and by `shasum` in `pre_census_03.txt`).
- **Acceptance 9.** The records are under `var/handover-03/`.
- **Self-checks the script makes** (each prints `!! STOP-CHECK FAILED` and exits 1 on failure; none fired): the sidecar
  fingerprint begins `0e32f24a`; ten bands; construction 2; 1160 rows; the four shards agree on `redshift` and
  `wavenumber_exit_time`; every GkSource's grid is the slice of the 1740-node grid from its `z_response` upward and
  `z_samples` equals its length; all rows carry `SourceGridConstruction_2`, and construction 2 on the model; the
  type/quality cells; 72 `mixed` rows; every `mixed` row's `crossover_z` matches exactly one node; policy row and GkSource parent name
  the same exit-time row; the recomputed summary serials equal the stored ones; no integration row named by
  `GkSource_parents` has a different exit-time row.
- **The test suites were not run.** The prompt does not ask for them and nothing here changes code.

## The census against verification §5.5, cell by cell

Full tables are in `POLICY-GEOMETRY.md` §3 (and the script's §2.1 (a) to (g)). §5.5's figures are 270 / 123 / 61 / 1 / 7 of 462.

| cell | §5.5 | A3 v2, 1160 rows | A3 v2, 1e5 ≤ k ≤ 1e7 (725 rows) | probe: also $z_{\rm response} \ge 10^7$ (390 rows) |
|---|---:|---:|---:|---:|
| `numeric`/`complete` | 58.44 % | 31.29 % (363) | 35.59 % (258) | 66.15 % |
| `WKB`/`complete` | 26.62 % | 62.50 % (725) | 58.07 % (421) | 22.05 % |
| `mixed`/`complete` | 13.20 % | 6.12 % (71) | 6.21 % (45) | 11.54 % |
| `mixed`/`minimal` | 0.22 % | 0.09 % (1) | 0.14 % (1) | 0.26 % |
| `fail`/`incomplete` | 1.52 % (7) | 0 | 0 | 0 |

- **Difference 3 (the k span)** accounts for 4.3 of the 27.2 points of the `numeric` shift and 4.4 of the 35.9 of the `WKB` shift, and
  for none of the `mixed` or `fail` shift (full and restricted agree).
- **The remainder is a number of response redshifts per $k$** (A3 v2 145; §5.5's 462/7 = 66, an inference from a division, not a recorded
  fact): `mixed` is 9.2 rows per $k$ on A3 v2 (restricted) against 8.86 from §5.5's (61 + 1)/7; `numeric` 51.6 against 38.6;
  `WKB` 84.2 against 17.6. On A3 v2 the `mixed` rows lie in a window $N(z_{\rm response}) = 3.59$ to $5.99$ that is the same at all eight $k$, with
  every row of that window `mixed`; 8 to 10 rows per $k$. That belongs to differences 1 and 2 jointly: the old store's own configuration is
  gone, and the response grid is every twelfth node of a source grid whose construction changed. The census cannot separate them.
- **The probe** (restricting to $z_{\rm response} \ge 10^7$, a lower limit §5.5's own TkWKB table suggests for the old grid) restores §5.5's
  rank order. It is a hypothesis, labelled as one.
- **Unexplained:** the 7 `fail`/`incomplete` rows of §5.5 and their absence here. Difference 3 does not touch it; differences 1 and 2 are the
  only candidates and cannot be told apart. A finding, not a failure.
- **No re-baselining.** §5.5's figures are not altered anywhere; `docs/source-remediation-verification.md` is not touched.
- *(Added on committing, 2026-10-25, on the user's ruling at review check 5: the old run's configuration is recorded in
  `docs/source-remediation-verification.md:224-238`, as `--k-count 7 … --zend 1e7`, with "66 response redshifts", and `z_end = 1e7` was
  chosen deliberately. 462 = 7 × 66 is therefore a fact, the probe's $10^7$ is that run's `z_end`, and the difference is a fourth known
  one, the response range, which the prompt's §2.2 omitted. Check 5 is cleared on it.)*

## The non-contiguity counts

From the stored value rows of all 1160 GkSource rows, applying the warning's test (`GkSource.py:283-305`, a WKB node whose next-lower node has none,
after one was seen):

| rows | count | hole, $z_{\rm response} < z_{e4}$ (the warning's condition) | hole, $z_{\rm response} \ge z_{e4}$ | WKB nodes above the primary run | no WKB at the bottom node | hole events |
|---|---:|---:|---:|---:|---:|---:|
| all | 1160 | **0** | **12** | 17 | 360 | 22 |
| `mixed` | 72 | 0 | 1 | 1 | 0 | 1 |
| `numeric` | 363 | 0 | 11 | 16 | 360 | 21 |
| `WKB` | 725 | 0 | 0 | 0 | 0 | 0 |

The 12 are 1.0 % of the rows, none at $z_{\rm response} < z_{e4}$; not a "material fraction", and not a stop. The one `mixed` row with a hole is the
`minimal` row (a hole at N = 3.484 above a WKB run ending at N = 3.507; WKB resumes at N = 3.438). `primary_WKB_largest_z`
recomputed from the value rows equals the stored serial on 1160 of 1160 rows.

## The exit-time check (prompt §3 item 4)

Not looked up by tolerance. The policy row's `wavenumber_exit_serial` equals its GkSource parent's on **all 1160 rows**. Every integration row the GkSource rows were
assembled from (`GkSource_parents`: 146,445 `GkNumericIntegration`, 916,937 `GkWKBIntegration`) names the same exit-time row as its GkSource: 0 differ. The exit-time
rows' tolerance pair is atol serial 1 (log10 −10), rtol serial 2 (log10 −9) on all 1160. That the rows used are the ones the policy was computed with rests on these
foreign keys and not on any value lookup.

## Observations not acted on

Each is in `issues.md`, for the orchestrator to open or drop. Items 1 to 3 there are the three issues of prompt §5, drafted and not fixed; items 4 to 7 are:

1. **The `complete` seam is one node near the top of the grid** on 71 of 72 rows ($3.121867\times10^{15}$, 82 intervals below the top, super-horizon),
   where `maximize-WKB` meets `CLEARANCE_GOOD` at the grid top.
2. **README §2 (q)'s "adds above it on some"** against 71 of 72: the band is present on all 72, the overlap above it on 71.
3. **The `minimal` row's overlap is cut by a WKB hole**, and the hole census (12 rows) is not empty.
4. **Verification §5.5's shape**, accounted for as a denominator, with one inference that cannot be checked.

Also, not issues but facts a later prompt will need:
- **The line numbers of README §2 (q) and the mechanism document have drifted.** The numeric cut L1 is now
  `main.py:2056` (`if z_source.z > k_exit.z_exit_subh_e4 - DEFAULT_FLOAT_PRECISION`; the README says `main.py:1888`); L2, the
  initial-data switch, is `main.py:466-468` (`z_source_limit = sqrt(k_exit.z_exit_subh_e3 * k_exit.z_exit_subh_e4)`; the README
  says `main.py:2120`). The `GkSourcePolicyData.py` citations (`:457`, `:465`, `:472-475`, `:287-504`, `:638-752`) still hold.
- **The source grid has half-step runs** at $z = 1.2387\times10^{10}$ to $1.3582\times10^{10}$ and $3.9183\times10^{10}$ to $4.2965\times10^{10}$, and 700
  fine intervals below $z = 1.0043\times10^{6}$; the `minimal` row lies in the second run. The interval counts of the 37 rows whose numeric
  side crosses a run are in half-steps there.
- **`mixed` per $k$ is 10 at $k = 3.09\times10^{6}$**, one more than elsewhere (9 or 8): nine `complete` rows and the `minimal` row, in the same
  $N(z_{\rm response})$ window as the others.
- **16 `numeric` rows have both representations at some node** (88 rows in all, 72 of them `mixed`); the type is decided by
  `MIN_SPLINE_DATA_POINTS` and the region ends, not by the presence of both values.

## State handed to the next step

- **The checkout** is at `c8ca1559fcc53fdb88998f170257635431fb85ca`, clean, not moved.
- **The store** `var/datastores/handover-A3-v2-lambdacdm.sqlite` (with its four shards and sidecar) is byte-identical; its fingerprint is
  `0e32f24a…`, equal to the sidecar's.
- **The registry** holds 12 runs, all finished. No run was registered by this prompt. No lock was taken, no Ray process started.
- **`var/handover-03/`** holds the records above, ready to be copied to `docs/handover/a3-v2/policy-geometry/` and `docs/handover/POLICY-GEOMETRY.md`
  by the laptop orchestrator. It is 1 MB; the longest file is `out/policy_geometry_03.txt`.
