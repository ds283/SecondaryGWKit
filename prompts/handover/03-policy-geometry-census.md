# Prompt 03 — the policy-geometry census

**Campaign:** [`README.md`](README.md) · **Board item:** **A3** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-09-19. **Refreshed 2026-10-25**, never having run: re-pointed from the A3 baseline
store to A3 v2 on the Mac Studio, reconciled with what prompt 11 has already measured there
(§0.1), and moved to the arrangement of prompts 10 and 11. The refresh changed the datastore, the
host, the records and the read-only pattern. It did not change the charter: count the source-grid
intervals each side of `crossover_z`, by node index, and open README §2 (o) and (p)'s issues without
fixing them.
**Where it runs:** on the Mac Studio, in a Claude Code session there, orchestrated there as 10 and
11 were ([`orchestrator/prompt-03.md`](orchestrator/prompt-03.md)). **It commits nothing there.**
Everything it writes is under `var/handover-03/` in the Mac Studio's main checkout.
**At which tree:** the dispatch SHA on `handover-remedial`, i.e. `HEAD`. B1 (`049c7c8`) changed
`ComputeTargets/QuadSource.py`, `QuadSourceIntegral.py`, `TkSourceFunctions.py`, the new
`ComputeTargets/handover.py` and three test modules. It did not touch `GkSourcePolicyData.py`,
`GkSource.py` or either factory (`git diff 98e68d3 HEAD --stat`). And this census imports nothing
from the pipeline in any case (§2).
**Uses:** no Ray, no lock, no store write. SQLite `mode=ro` on A3 v2's primary and four shards
(`var/datastores/handover-A3-v2-lambdacdm.sqlite`, fingerprint `0e32f24a…`), and a read of its
sidecar.
**Closes:** README §7 **D5** (the $G_k$ overlap width, asked by the follow-up note §3 item 1 and
never measured). **Narrows** nothing by itself. It is the instrument **D1** is scored on, and the
only thing that can tell **E2** whether the $G$ seam is protected at the depth it chooses.
**Opens:** the three §3 issues README §2 (o) and (p) record. They are already known and are **not**
to be fixed here (§5).
**Recommended model:** **Sonnet**. This is a census, not an argument. The rows are read
mechanically, and the only judgement is in the honesty of §4's "what this does not cover".

**Read first:**

1. [`docs/handover/HANDOVER-MECHANISM.md`](../../docs/handover/HANDOVER-MECHANISM.md) **in full**.
   In particular §0.1 (whose object is whose; the $z_{\rm source}$/$z_{\rm response}$ transposition
   at `GkSource` is where this prompt goes wrong if it goes wrong), §2.1a, §3.2 and §6.
2. `ComputeTargets/GkSourcePolicyData.py`: `_classify_crossover` (`:287-504`) and
   `_create_functions` (`:638-752`). Read both in full; you are measuring the geometry they assume.
3. `ComputeTargets/GkSource.py`: `assemble_GkSource_values` (`:170`), for how
   `primary_WKB_largest_z` and `numeric_smallest_z` are found (`:195-200`, `:310-316`, `:382-385`,
   `:458-459`) and when the non-contiguity warning fires (`:283-305`).
4. The tables you read: `ObjectFactories/GkSourcePolicyData.py:13-80` (the integer codes of `type`
   and `quality`, and the columns), `ObjectFactories/GkSource.py` (the `GkSource` row's
   `numeric_smallest_z_serial` and `primary_WKB_largest_z_serial`, `:274-288`; its value table, keyed
   by `z_source_serial`, `:880-905`), and `ObjectFactories/wavenumber.py:135-180` (the exit-time
   row's `z_exit_subh_eN` columns).
5. [`docs/handover/a3-v2/seam/scripts/seam_census.py`](../../docs/handover/a3-v2/seam/scripts/seam_census.py),
   prompt 11's census: **the pattern to follow**, not a file to edit. Read-only and Ray-free,
   `sqlite3` and `json` only, nothing from the pipeline imported, a constant read from the source
   text rather than by import, and a header recording its own SHA-256, each store file's SHA-256,
   the fingerprint and `git HEAD`.
6. [`docs/handover/A3-V2-VERIFICATION.md`](../../docs/handover/A3-V2-VERIFICATION.md) **§5**, in
   particular §5.3: what prompt 11 has already measured on this store. §0.1 below says what follows
   from it.
7. [`docs/source-remediation-verification.md`](../../docs/source-remediation-verification.md) §5.5 —
   the 462-row type/quality census (audit §4.2's table: `numeric`/`complete` 270,
   `WKB`/`complete` 123, `mixed`/`complete` 61, `mixed`/`minimal` 1, `fail`/`incomplete` 7).
   **It cannot be reproduced on the same datastore, and you must not try.** See §2.2.
8. Campaign [`README.md`](README.md) §0.4, §2 (b), (f), (o), (p), (q), §5 and §7 **D5** and
   **D7**. README §2 (o)–(q) cite `main.py` and `MetadataConcepts/` by line as they stood in
   September. Those numbers have drifted. Every file:line you write down is re-read at the
   dispatch tree, not copied from the README.
9. [`docs/lg-phase-and-handover-followup-2026-09.md`](../../docs/lg-phase-and-handover-followup-2026-09.md)
   §1.3 and §3 item 1 — the question, in the words in which it was first asked.

**It changes no tracked file, adds no test and runs no git command that writes.** It writes a
census script, its output and a document under `var/handover-03/`. The orchestrator copies them
back to the laptop, and the bookkeeping commit is made there (§6).

**Gate** (check, and save to `out/gate_03.txt`):
- 11's board row is "reviewed", and the dispatch commit is pushed (checked without fetching, as
  11's Gate was);
- `HEAD` is the dispatch SHA, and `git status --short` is empty. **The checkout was fast-forwarded by
  the laptop orchestrator before dispatch**; you do not move it;
- `git diff 98e68d3 HEAD --stat -- ComputeTargets/GkSourcePolicyData.py ComputeTargets/GkSource.py
  ObjectFactories/GkSourcePolicyData.py ObjectFactories/GkSource.py` is empty;
- `runregistrykit list` shows nothing `running` or stale, **except** a `levin-phase-floor-04-*` run
  whose pid answers `kill -0`. That is another session's (§0.2). Record it as a `STRUCTURALLY
  REQUIRED` deviation, not a stop. Any other `running` or stale run is a stop;
- `store show` gives A3 v2 live, fingerprint `0e32f24a…`;
- there is no `var/handover-03/`.

---

## 0. Before anything

### 0.1 What prompt 11 already measured, and what this prompt therefore does not repeat

Prompt 11 ran on this store on 2026-10-25 and passed review. Its §5.3 transcribed the build's
`GkSourcePolicyData SUMMARY STATISTICS` block and checked it against a count of the table, cell by
cell:

| `type` | `quality` | rows | share |
|---|---|---:|---:|
| numeric | complete | 363 | 31.29 % |
| mixed | complete | 71 | 6.12 % |
| mixed | **minimal** | **1** | 0.09 % |
| WKB | complete | 725 | 62.50 % |
| | **total** | **1160** | |

**No `fail` and no `incomplete` row exists on A3 v2.** The one `minimal` row is shard 2, serial 460,
$k = 3.091682\times10^6$/Mpc, $z_{\rm response} = 3.963729\times10^{10}$,
`crossover_z` $= 4.2472907\times10^{10}$.

So, of the 2026-09 plan:
- **The type/quality census is not re-taken.** Your script counts the rows by decoded
  `(type, quality)` as a consistency check, and the count must equal the table above in every cell.
  A disagreement is a stop. What this prompt adds is the **comparison of shape** against §5.5
  (§2.2), which 11 did not make, and the breakdowns §2.2 asks for.
- **The `fail`/`incomplete` rows of the old §2 item 2 do not exist here.** There is nothing to
  attribute to `_classify_crossover`'s failure branches and nothing to look up in the
  `QuadSourceIntegral` table. Say so in one line, citing 11. The `minimal` row takes their place:
  it is tabulated individually, with its `metadata["comment"]` (§2).
- **§2.1 item 2 cannot be answered as it was asked** — there is no `incomplete` row to cluster.
  §2.1 says what is measured instead.
- **The deliverable is unchanged and has never been measured:** `numeric_intervals` and
  `WKB_intervals` on every `mixed` row. Nothing in 11 counts grid intervals.

### 0.2 Who else is on the Mac Studio

`levin-phase-floor` prompt 04 may be running at the same time on the same host. Its code runs in a
git worktree at `98e68d3`. It writes under `var/levin-phase-floor-04/` in this same main checkout,
opens A3 v2 `mode=ro`, and runs as a registered detached run with no Ray and no lock. So the two
cannot interfere **except through git and `var/`**:
- **You run no git command that writes**, `fetch` included.
- **You write only under `var/handover-03/`.** You never read or write
  `var/levin-phase-floor-04/`, the worktree, the store's files, its sidecar or any run directory.
- Your census takes seconds. You need not wait for the other run, and you do not end, signal or
  inspect it beyond the Gate's `kill -0`.

---

## 1. The question, stated precisely

The follow-up note §1.3 says the $G$ seam is protected *"if the overlap is at least two grid
intervals on each side of `crossover_z`"*, and §3 item 1 asks for that to be measured. README
**D5** has carried it since 2026-09-19. It has never been taken.

**`_classify_crossover` cannot answer it.** Its clearance is a *fraction of $\log(1+z)$* —

$$
c_{\rm num} = \frac{\log(1+z) - \log(1+z_{\rm num,low})}{\log(1+z_{\rm num,low})},
\qquad
c_{\rm WKB} = \frac{\log(1+z_{\rm WKB,high}) - \log(1+z)}{\log(1+z_{\rm WKB,high})}
$$

— thresholded at 0.05 / 0.025 / 0.01 (`GkSourcePolicyData.py:410-416`, `:267-284`). Grid spacing
does not enter the function anywhere. At $z\sim10^{13}$, $\log(1+z)\approx30$, so `CLEARANCE_GOOD`
means about 1.5 e-folds; at $z\sim10$ it means about 0.12. **So the stored `quality` label does not
tell you how many grid intervals the crossover has on either side, and nothing else does either.**
On A3 v2, 1159 of 1160 labels are `complete` (§0.1). That says the clearance fractions passed, and
nothing about the intervals.

Two consequences make this worth an instrument rather than a guess:

- **D1** (`[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]`) measures the
  numeric $G$ spline's interpolation error at **1.6e-04 to 9.4e-03** of the envelope against
  **2.6e-07** for the solver. A cubic spline loses one to two orders in its **last two intervals**
  (`[05-numeric-region-is-now-the-accuracy-floor]`). Whether the consumed range touches those
  intervals is exactly the D5 question, and it decides whether D1's number is the floor or an
  understatement.
- **`maximize-WKB` takes the candidate with the *least* WKB clearance in the winning band**
  (mechanism §3.2 point 2). In the `complete` band that is bounded; below it, it is not. A3 v2 has
  71 `mixed` rows `complete` and one `minimal`, on one cosmology and one grid.

---

## 2. What to build

`var/handover-03/scripts/policy_geometry.py`, following `seam_census.py` (Read first item 5) and
**not an edit of it**. Read-only and Ray-free throughout: every shard and the primary opened
`file:…?mode=ro`, `sqlite3` and `json` only, **nothing imported from this repository's packages**.
Where it needs a constant — `CLEARANCE_GOOD`/`ACCEPTABLE`/`MARGINAL`, `CLASSIFICATION_BANDS`,
`MIN_SPLINE_DATA_POINTS` — it reads it from `ComputeTargets/GkSourcePolicyData.py`'s text and
records the value, as `seam_census.py` reads `HANDOVER_CLAMP_MAX_GRID_STEPS`. Its header records its
own SHA-256, each store file's SHA-256, the fingerprint and `git HEAD`. It prints markdown tables to
stdout. **Run `black` on it before its first run**, so that the committed record is both
black-clean and byte-identical to what ran.

For **every** stored `GkSourcePolicyData` row (1160), report:

| Column | Meaning |
|---|---|
| `k`, `z_response`, `shard`, policy `serial`, `source_serial` | the row |
| `type`, `quality` | as stored, decoded |
| `crossover_z` | as stored |
| `overlap_log_width` | $\log(1+\texttt{numeric\_smallest\_z})$ to $\log(1+\texttt{primary\_WKB\_largest\_z})$ |
| `overlap_intervals` | the same, **in source-grid intervals** — the count of grid nodes strictly inside the overlap, plus one |
| `numeric_intervals` | grid intervals between `crossover_z` and `numeric_smallest_z` |
| `WKB_intervals` | grid intervals between `primary_WKB_largest_z` and `crossover_z` |
| `c_num`, `c_WKB` | the two stored-geometry clearances recomputed, for the tie to §3.2 |
| `band` | which of the ten `CLASSIFICATION_BANDS` the stored `quality` implies |

The geometry comes from the policy row's `GkSource` parent (`source_serial`): its
`numeric_smallest_z_serial` and `primary_WKB_largest_z_serial`, and its value rows, whose
`z_source_serial`s are the source grid that `GkSource` was assembled on. `crossover_z` is stored as
a float, not a serial. Locate it on the grid by **exact** equality with a node's stored `z`
(`_classify_crossover` assigns it from a node, `crossover_z.z`), and stop if any `mixed` row's
`crossover_z` matches no node exactly.

**`numeric_intervals` and `WKB_intervals` are the deliverable.** Everything else is context for
them. Report each as a distribution (min, 5th, median, 95th, max) over all 72 `mixed` rows, and
**tabulate the rows with fewer than two on either side individually** — those are the rows the
follow-up note says are unprotected. If there are none, say so in exactly those terms. **The
`minimal` row is tabulated individually whatever its counts**, with its `metadata["comment"]`
verbatim.

Also report, because they are free once the rows are open and both are recorded gaps in the
mechanism document §6:

1. **The type/quality count**, as the consistency check of §0.1, and the comparison of shape against
   verification §5.5 of §2.2.
2. **The non-contiguity count** (§3 item 2).

### 2.1 The two questions B4 and D7 need answered

README §2 (q) derives the overlap's structure from the producers and names seven levers. It is a
derivation from the code, **not a measurement**, and this prompt is where it gets checked.

1. **Where the overlap actually sits, in e-folds.** For every `mixed` row report
   `numeric_smallest_z` and `primary_WKB_largest_z` as **e-folds sub-horizon**, against the three
   landmarks $z_{e3}$, $\sqrt{z_{e3}z_{e4}}$ and $z_{e4}$ that §2 (q) says bound it. Take the
   landmarks from the exit-time row the policy row itself names (`wavenumber_exit_serial`), and
   check that its `GkSource` parent names the same one (§3 item 4). Convert redshifts to e-folds
   only in the direction $z\to\log(1+z)$, by interpolating the row's stored `z_exit_subh_eN`
   columns. The prediction to test is that the band $z_{e4} < z_{\rm source} < \sqrt{z_{e3}z_{e4}}$
   — **half an e-fold, the same for every $k$** — is present on every row, with the upper
   `GkWKBIntegration` branch adding above it on some. **If the measurement disagrees with the
   derivation, the measurement wins and §2 (q) is corrected** — by the user, after this prompt
   (§7). That is the single most valuable thing this prompt can find, because B4 plans to move L1
   and L2 on the strength of it.
2. **Whether the thin end of the overlap is a statement about $z_{\rm response}$ or about $k$.**
   §2 (q) argues that the guaranteed band is fixed in e-folds, so the $z_{\rm response}$ dependence
   lives entirely in L3 (`min(z_e3, z_init)`) and L4 ($0.85\,z_{e6}$ and the `"stop"` event), and
   that `incomplete` rows should therefore cluster in $z_{\rm response}$ and be spread in $k$.
   **A3 v2 has no `incomplete` row** (§0.1), so that test is vacuous here: say so. Measure instead
   the nearest thing the store holds:
   - `overlap_log_width` in e-folds as a table over $k$ decade × $z_{\rm response}$ decade, for the
     `mixed` rows. The map predicts a floor of half an e-fold in every cell, with any excess varying
     along $z_{\rm response}$.
   - The rows in the lowest decile of $\min(\texttt{numeric\_intervals}, \texttt{WKB\_intervals})$,
     plus the `minimal` row, tabulated both by $k$ and by $z_{\rm response}$. Say which they cluster
     in, if either.
   - The type mix (`numeric` / `mixed` / `WKB`) over the same $k \times z_{\rm response}$ grid. That
     says where the `mixed` rows are at all.

   This is a proxy, not the quantity §2 (q) made its prediction about. A clustering in $k$ is
   reported as evidence against the lever map. It is **not** a stop condition, unlike the absence of
   the half-e-fold band in item 1.

Report both against the **grid-interval** columns of §2, not instead of them. The e-fold picture
says where the overlap is. The interval count says whether it is usable (`MIN_SPLINE_DATA_POINTS`,
and the end-interval question of the mechanism document §6).

### 2.2 The datastore, and what §5.5 can and cannot be compared against

The store is `var/datastores/handover-A3-v2-lambdacdm.sqlite` on the Mac Studio, with four shards
and its sidecar `handover-A3-v2-lambdacdm.manifest.json` (`var/` is gitignored). **Read the sidecar
first.** It records the fingerprint, the history and the tag sets. Treat the store as read-only.

**The source-grid construction.** The sidecar's tag sets for `GkSource` include
`SourceGridConstruction_2`, `SourceRedshiftGrid_1740_8d1d43b6` and
`ResponseRedshiftGrid_145_c51d43ac`. Confirm by reading `CosmologyConcepts/wavenumber.py` that
`SOURCE_GRID_CONSTRUCTION_VERSION = 2` is the curvature criterion `qcd-background-audit` prompt 15
introduced, and record the tags the `GkSource` rows carry. The census is meaningless without saying
which grid it was taken on.

**Verification §5.5's census cannot be re-run, and you must not attempt a like-for-like
comparison.** Three things differ, all of them known in advance:

1. **Its datastore no longer exists.** It was written into a session scratchpad, which has since
   been reaped. Nothing about audit §4.2's 462 rows can be re-derived; only the published table
   survives.
2. **The grid criterion changed.** `qcd-background-audit` prompt 15 replaced the base density with
   the measured curvature criterion, so interval *counts* are expected to differ. That is the whole
   reason a new store was needed.
3. **The $k$ span is wider.** §5.5's run covered $10^5$–$10^7$/Mpc — two decades, the bottom of the
   range. A3 v2 spans the full production $10^5$–$3\times10^8$, so it includes hand-over geometry
   at $k$ the earlier census never saw, and the type mix may legitimately shift.

So the comparison is **of shape**: do the same types appear, in roughly the same rank order, with
the pathological classes still rare? **The shape has already shifted, and you know by how much
before you start.** On §4.2's 462 rows `numeric` led (58.44 %), then `WKB` (26.62 %), then `mixed`
(13.42 %), with 1.52 % `fail`. On A3 v2's 1160, `WKB` leads (62.50 %), then `numeric` (31.29 %),
then `mixed` (6.21 %), with no `fail`. The rank order of the two pure types has reversed, and
`mixed` is at less than half its share.

To separate the three differences, tabulate the type/quality census **also restricted to
$10^5 \le k \le 10^7$/Mpc**, which removes difference 3, and by $k$ decade across the full span.
For each cell that differs from §4.2's, say which of the three accounts for it, with the table that
shows it, or record it as unexplained. **An unexplained difference is a finding to state, not a
failure.** But §7's second stop condition is for a difference the three cannot account for, and you
should reach for that rather than inventing a fourth reason.

§4.2's figures, for reference: `numeric`/`complete` 270 (58.44 %), `WKB`/`complete` 123 (26.62 %),
`mixed`/`complete` 61 (13.20 %), `mixed`/`minimal` 1 (0.22 %), `fail`/`incomplete` 7 (1.52 %), of
462 rows.

---

## 3. The four things that will go wrong

1. **The transposition.** `GkSource` is at fixed $z_{\rm response}$ over a $z_{\rm source}$ grid,
   while `GkWKBIntegration` is at fixed $z_{\rm source}$ over a $z_{\rm response}$ grid
   (mechanism §0.1). The grid whose intervals you are counting is the **source** grid of the
   `GkSource` — its value rows' `z_source_serial`s, which is `source.z_sample` in memory — not any
   response grid. Getting this backwards produces a plausible table of wrong numbers.
2. **`primary_WKB_largest_z` is contiguous-from-the-bottom**, not simply the largest $z$ with a WKB
   value (`GkSource.py:195-197`, `:312-316`, `:382-385`). A WKB island above a hole is excluded, and
   `assemble_GkSource_values` prints a warning when it sees one with
   $z_{\rm response} < z_{e4}$ (`GkSource.py:283-305`). From the stored value rows, count the
   `GkSource` rows whose WKB values have a hole above the primary region, split by whether
   $z_{\rm response} < z_{e4}$ (the warning's condition), and report both numbers. If either is
   non-zero, that is a finding in its own right.
3. **`log(1+z) → z` is lossy and must not appear in an equality-like comparison** (campaign README
   §2 (n), `CLAUDE.md`). Count grid intervals by **node index in the stored grid**, matching on
   redshift `serial`, not by converting a $\log(1+z)$ width back to a redshift and dividing. The
   redshifts here reach $\sim10^{14}$, where recoverable $1+z$ has a granularity of ~0.02. The one
   float match this prompt makes, `crossover_z` against a node's stored `z` (§2), is exact equality
   on a value copied from that node, not a round trip.
4. **The exit-time row.** `[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]`
   records two verification scripts that looked the exit time up under a tolerance pair other than
   the sector's own. **Do not look it up by tolerance at all.** The `GkSourcePolicyData` row and its
   `GkSource` parent each carry a `wavenumber_exit_serial`. Use that row. Check that the two agree on
   every row, record the exit-time row's `atol`/`rtol` serials and their `log10_tol`, and say in the
   document how you confirmed the rows you used are the ones the policy was computed with.

---

## 4. What to write down

`var/handover-03/POLICY-GEOMETRY.md`, which the orchestrator commits as
`docs/handover/POLICY-GEOMETRY.md`, in the shape of
[`docs/tolerance-convergence/QUADSOURCE-READONLY.md`](../../docs/tolerance-convergence/QUADSOURCE-READONLY.md):

- a header block naming the tree SHA, the datastore and its fingerprint, the model, and **the
  source-grid construction tags** the rows carry — a census of overlap widths is meaningless without
  saying which grid;
- **§0 — the answer to D5, in one paragraph**, in the form "on 72 `mixed` rows of A3 v2,
  `crossover_z` has *(distribution)* grid intervals of numeric clearance and *(distribution)* of
  WKB clearance; M rows have fewer than two on one side; they are *(list)*";
- the tables, generated by the script, with a `<!-- generated by ... -->` marker. Links in the
  document are written relative to `docs/handover/`, as 10's and 11's sections are (the
  `a3-v2/policy-geometry/…` form);
- **§ — the relation between the two clearances.** The stored $c_{\rm num}$, $c_{\rm WKB}$ against
  the interval counts, as a binned table. The question a reader will ask is whether the band
  thresholds happen to track grid intervals on the grid that actually ships. **State what you
  find. Do not assert a relationship you have not measured, and do not recommend a threshold
  change** — README §0.4 and §5 rule 8; this document measures and hands over;
- **§ — set beside prompt 11.** One short paragraph citing `A3-V2-VERIFICATION.md` §5.3 for the
  type/quality count and the absence of `fail`/`incomplete`, and saying what this census adds;
- **§ — what this does not cover**, the honest closing section. At minimum: one datastore, one grid
  construction, one cosmology (ΛCDM); the $T_k$ seam is *not* covered (it has no overlap to measure
  — mechanism §5 — and B1 has since changed its construction); §2.1 item 2 was answered by a proxy,
  because A3 v2 has no `incomplete` row; and the census says nothing about whether the end-interval
  error *matters*, only whether the geometry exposes it.

---

## 5. What this prompt does not do

- It does not touch any tracked file. Not `GkSourcePolicyData.py`, not `main.py`, not `config/`,
  not `MetadataConcepts/`.
- **It does not fix the three defects of README §2 (o) and (p)**, although it is reading exactly that
  code. It drafts them as §3 issues in `var/handover-03/issues.md`, with file:line evidence re-read
  at the dispatch tree and nothing more, under these names:
  - `[03-gksource-policy-accepts-a-value-that-raises]` (§2 (o) item 1);
  - `[03-quadsource-policy-vocabulary-differs]` (§2 (o) item 2, with the warning messages that lose
    the offending value);
  - `[03-gk-classification-is-diagnosed-and-ignored]` (§2 (p)), carrying 11's measurement that the
    shipping grid has no `fail` or `incomplete` row, so the fail-late contract is unenforced and
    unexercised.

  The orchestrator opens them on the board (§6). Workstream **C2** owns the first two, **B3** the
  third. Fixing them here would destroy the revert-per-prompt property for a change that has no
  measurement attached.
- It does not change any threshold, band or policy value, and does not recommend one.
- It does not re-run the source integral, the Green's function or anything else that computes, and
  it does not rebuild `GkSourceFunctions`. It reads stored rows.
- It does not edit `seam_census.py`, `analyse_greens_and_source.py` or
  `docs/source-remediation-verification.md`. They are records.
- It adds no test. Suite counts are untouched, because no tracked file changes.

---

## 6. Acceptance

1. One command, from the repository root,
   `./venv/bin/python -I var/handover-03/scripts/policy_geometry.py <primary>`, recorded with its
   output and runtime in `out/policy_geometry_03.txt`.
2. Every store file and the sidecar are byte-identical before and after (`state/store_files_before.txt`,
   `state/store_files_after.txt`), and `runregistrykit store fingerprint` (without `--write`) gives
   `0e32f24a…`.
3. The type/quality count equals §0.1's table in every cell, and is compared against verification
   §5.5's **as a shape on a different store**, per §2.2, with the $10^5$–$10^7$ restriction, each
   difference either accounted for or recorded as unexplained.
4. `numeric_intervals` and `WKB_intervals` are reported as distributions over all 72 `mixed` rows;
   every row with fewer than two on either side is tabulated individually, and so is the `minimal`
   row.
5. `POLICY-GEOMETRY.md` exists, its §0 answers D5 in one paragraph, and it has a closing "what this
   does not cover".
6. §2.1's two questions are answered: the overlap's position in e-folds against the three
   landmarks, and the proxy for the $z_{\rm response}$-versus-$k$ question, stated as a proxy. Both
   are reported as measurements with the derivation of README §2 (q) quoted beside them.
7. The non-contiguity count and the exit-time check of §3 are reported.
8. `black --check` is clean on the script, and the script that ran is the script recorded.
9. Records under `var/handover-03/`: `out/`, `scripts/`, `state/`, `POLICY-GEOMETRY.md`, `log.md`
   and `issues.md`.

The orchestrator then, on the laptop and in **one commit**:
- copies `out/`, `scripts/`, `state/` and `issues.md` to `docs/handover/a3-v2/policy-geometry/`,
  and `POLICY-GEOMETRY.md` to `docs/handover/`;
- writes `log.md` to `logs/03-policy-geometry-census.md`;
- the board: A3's row and item, and the three §3 issues of §5 opened, with the paragraph "Not on this
  board, deliberately" removed;
- `docs/OPEN_ISSUES.md`: the three §1.1 rows that have had no board row now point at the board, and
  the header's sentence about them is corrected; the count and date by counting;
- README §7 **D5** marked answered, with a pointer to the new document. **D5 closes on the campaign
  README**, not on the index — it is a decision, not an issue.

---

## 7. Stop conditions — stop and ask the user

- **Any Gate item is not as stated**, or the store is not A3 v2 at `0e32f24a…`.
- **The rows do not carry `SourceGridConstruction_2`**, or construction 2 is not the curvature
  criterion. A census on a superseded grid answers a question nobody asked. Report which grid the
  rows carry and stop.
- **The type/quality count disagrees with §0.1's table in any cell.**
- **The type/quality census differs from verification §5.5's shape in a way §2.2's three known
  differences do not account for.** The full-store shift of §2.2 is known in advance and is not by
  itself this condition. It is this condition if, restricted to $10^5$–$10^7$/Mpc, `mixed` rows are
  still materially scarcer than §4.2's 13.42 % and neither the grid change nor the loss of the old
  store accounts for it. Report which cells differ, and do not re-baseline.
- **A `mixed` row's `crossover_z` matches no source-grid node exactly**, or a policy row and its
  `GkSource` parent name different exit-time rows.
- **More than a few percent of `mixed` rows have fewer than two grid intervals on a side.** That
  would mean the $T_k$ end-interval analysis applies to $G$ as well, with larger absolute errors
  (follow-up §3 item 1), which changes D1's scope and E2's target. Report it and stop; do not
  propose a remedy.
- **`primary_WKB_largest_z` is non-contiguous on a material fraction of rows.** That is a producer
  problem, not a policy one, and it is outside this prompt.
- **§2.1 item 1 falsifies the lever map** — the half-e-fold band of README §2 (q) is not present on
  every `mixed` row. B4 is planned on that derivation and §7 **D7** is deferred pending it, so a
  disagreement changes two pieces of the campaign plan. Report it and stop; do not correct §2 (q)
  yourself.
- Any store file, or the sidecar, changes.

---

## 8. The log

`var/handover-03/log.md`, which becomes `logs/03-policy-geometry-census.md`: template as
`qcd-background-audit` README §5.1, with every heading of `logs/11-a3-v2-seam-record.md`. Beyond it:
the datastore, its fingerprint and the grid tags the census ran on; the census against verification
§5.5 cell by cell, with the $10^5$–$10^7$ restriction; the non-contiguity counts; the exit-time
check (§3 item 4); and the deviations of the refresh's arrangement that apply — the Gate's
exception for `levin-phase-floor` 04, if it arose, and the checkout check that replaces 10's and
11's step 1, since the checkout was moved for you.
