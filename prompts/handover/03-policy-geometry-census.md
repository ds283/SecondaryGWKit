# Prompt 03 — the policy-geometry census

**Campaign:** [`README.md`](README.md) · **Board item:** **A3** ·
**Board:** `IMPLEMENTATION_STATE.md` — **created by prompt 01**; if 01 has not landed, stop and say
so rather than creating it here.
**Closes:** README §7 **D5** (the $G_k$ overlap width, asked by the follow-up note §3 item 1 and
never measured). **Narrows** nothing by itself — it is the instrument **D1** is scored on and the
only thing that can tell **E2** whether the $G$ seam is protected at the depth it is choosing.
**Opens:** the two §3 issues of README §2 (o), which are already known and are **not** to be fixed
here.
**Recommended model:** **Sonnet**. This is a census, not an argument: the reading of the stored
rows is mechanical and the only judgement is in the honesty of §5's "what this does not cover".

**Read first:**

1. [`docs/handover/HANDOVER-MECHANISM.md`](../../docs/handover/HANDOVER-MECHANISM.md) **in full** —
   in particular §0.1 (whose object is whose; the $z_{\rm source}$/$z_{\rm response}$ transposition
   at `GkSource` is where this prompt goes wrong if it goes wrong), §3.2 and §6.
2. `ComputeTargets/GkSourcePolicyData.py` — `_classify_crossover` (`:287-504`) and
   `_create_functions` (`:638-752`). Read both in full; you are measuring the geometry they assume.
3. [`docs/source-remediation-verification/analyse_greens_and_source.py`](../../docs/source-remediation-verification/analyse_greens_and_source.py)
   — the **pattern to follow**, not a file to edit. Its `gk_continuity` and the audit §4.2 census
   are the closest existing work; its module docstring states the read-only discipline this prompt
   inherits.
4. [`docs/source-remediation-verification.md`](../../docs/source-remediation-verification.md) §5.5 —
   the 462-row type/quality census (§4.2's table: `numeric`/`complete` 270, `WKB`/`complete` 123,
   `mixed`/`complete` 61, `mixed`/`minimal` 1, `fail`/`incomplete` 7).
   **It cannot be reproduced on the same datastore, and you must not try.** That store was written
   into a session scratchpad and no longer exists; it also predates `qcd-background-audit`
   prompt 15, and it covered only $10^5$–$10^7$/Mpc where the current baseline store spans the full
   production $10^5$–$3\times10^8$. The comparison is therefore **of shape, on a different store**,
   and the three differences above are the ones you account for. See §2.2.
5. Campaign [`README.md`](README.md) §0.4, §2 (b), (f), (o), §5, §7 **D5**.
6. [`docs/lg-phase-and-handover-followup-2026-09.md`](../../docs/lg-phase-and-handover-followup-2026-09.md)
   §1.3 and §3 item 1 — the question, in the words in which it was first asked.

**It changes no production code and adds no test.** It adds a `docs/` script and a document.

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

— thresholded at 0.05 / 0.025 / 0.01 (`GkSourcePolicyData.py:415-416`, `:266-284`). Grid spacing
does not enter the function anywhere. At $z\sim10^{13}$, $\log(1+z)\approx30$, so `CLEARANCE_GOOD`
means about 1.5 e-folds; at $z\sim10$ it means about 0.12. **So the stored `quality` label does not
tell you how many grid intervals the crossover has on either side, and nothing else does either.**

Two consequences make this worth an instrument rather than a guess:

- **D1** (`[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]`) measures the
  numeric $G$ spline's interpolation error at **1.6e-04 to 9.4e-03** of the envelope against
  **2.6e-07** for the solver. A cubic spline loses one to two orders in its **last two intervals**
  (`[05-numeric-region-is-now-the-accuracy-floor]`). Whether the consumed range touches those
  intervals is exactly the D5 question, and it decides whether D1's number is the floor or an
  understatement.
- **`maximize-WKB` takes the candidate with the *least* WKB clearance in the winning band**
  (mechanism §3.2 point 2). In the `complete` band that is bounded; below it, it is not. On the
  prompt-12 run 61 of 62 `mixed` rows were `complete` and one was `minimal` — but that is one run,
  one cosmology, one grid, and the grid has since been replaced by `qcd-background-audit`
  prompt 15's curvature criterion.

---

## 2. What to build

`docs/handover/policy_geometry.py` — a sibling of `analyse_greens_and_source.py`, **not an edit of
it**. Read-only throughout: shards opened `mode=ro`, `prune_unvalidated=False`, no `object_store`
and no `object_validate` call anywhere. Command-line shape as that script's
(`--database`, `--shards-glob`, `--shards`, `--model-label`), printing markdown tables to stdout.

For **every** stored `GkSourcePolicyData` row, report:

| Column | Meaning |
|---|---|
| `k`, `z_response`, policy `store_id` | the row |
| `type`, `quality` | as stored |
| `crossover_z` | as stored |
| `overlap_log_width` | $\log(1+\texttt{numeric\_smallest\_z})$ to $\log(1+\texttt{primary\_WKB\_largest\_z})$ |
| `overlap_intervals` | the same, **in source-grid intervals** — the count of grid nodes strictly inside the overlap, plus one |
| `numeric_intervals` | grid intervals between `crossover_z` and `numeric_smallest_z` |
| `WKB_intervals` | grid intervals between `primary_WKB_largest_z` and `crossover_z` |
| `c_num`, `c_WKB` | the two stored-geometry clearances recomputed, for the tie to §3.2 |
| `band` | which of the ten `CLASSIFICATION_BANDS` the stored `quality` implies |

**`numeric_intervals` and `WKB_intervals` are the deliverable.** Everything else is context for
them. Report each as a distribution (min, 5th, median, 95th, max) over all `mixed` rows, and
**tabulate the rows with fewer than two on either side individually** — those are the rows the
follow-up note says are unprotected, and if there are none, say so in exactly those terms.

Also report, because they are free once the rows are open and both are recorded gaps in the
mechanism document §6:

1. **The type/quality census**, in the shape of verification §5.5's table, so the two can be
   compared directly.
2. **The `fail`/`incomplete` rows**: how many, at which $(k, z_{\rm response})$, and — for each —
   which of `_classify_crossover`'s four failure branches produced it, read from
   `metadata["comment"]`. Verification §5.5 found 7 of 462 and recorded them as an observation
   only. State whether `build_partition` would raise on them by checking whether the
   $(k, z_{\rm response})$ pair appears in the `QuadSourceIntegral` table; **do not** run the
   integral to find out.

### 2.1 The two questions B4 and D7 need answered

README §2 (q) derives the overlap's structure from the producers and names seven levers. It is a
derivation from the code, **not a measurement**, and this prompt is where it gets checked. Two
additions to the tables above, both cheap once the rows are open:

1. **Where the overlap actually sits, in e-folds.** For every `mixed` row report
   `numeric_smallest_z` and `primary_WKB_largest_z` as **e-folds sub-horizon**, against the three
   landmarks $z_{e3}$, $\sqrt{z_{e3}z_{e4}}$ and $z_{e4}$ that §2 (q) says bound it. The
   prediction to test is that the band $z_{e4} < z_{\rm source} < \sqrt{z_{e3}z_{e4}}$ — **half an
   e-fold, the same for every $k$** — is present on every row, with the upper `GkWKBIntegration`
   branch adding above it on some. **If the measurement disagrees with the derivation, the
   measurement wins and the derivation is corrected in §2 (q).** That is the single most valuable
   thing this prompt can find, because B4 plans to move L1 and L2 on the strength of it.
2. **Whether `incomplete` is a statement about $z_{\rm response}$ or about $k$.** §2 (q) argues
   the guaranteed band is fixed in e-folds, so the $z_{\rm response}$ dependence lives entirely in
   L3 (`min(z_e3, z_init)`) and L4 ($0.85\,z_{e6}$ and the `"stop"` event) — and therefore that
   the `fail`/`incomplete` rows should cluster in $z_{\rm response}$ and be spread in $k$. Tabulate
   them both ways and say which. A clustering in $k$ instead would falsify the lever map and is a
   stop condition.

Report both against the **grid-interval** columns of §2, not instead of them: the e-fold picture
says where the overlap is, the interval count says whether it is usable (`MIN_SPLINE_DATA_POINTS`,
and the end-interval question of the mechanism document §6).

### 2.2 The datastore, and what §5.5 can and cannot be compared against

The baseline store is `var/datastores/handover-A3-baseline-lambdacdm.sqlite` (4 shards, `var/` is
gitignored). **Read its manifest beside it first** — it records the scope, the tree SHA, the grid
criterion, the run history and a backup you must not delete. Treat the store as read-only.

**Verification §5.5's census cannot be re-run, and you must not attempt a like-for-like
comparison.** Three things differ, all of them known in advance:

1. **Its datastore no longer exists.** It was written into a session scratchpad, which has since
   been reaped. Nothing about §4.2's 462 rows can be re-derived; only the published table survives.
2. **The grid criterion changed.** `qcd-background-audit` prompt 15 replaced the base density with
   the measured curvature criterion, so interval *counts* are expected to differ. That is the whole
   reason a new store was needed.
3. **The $k$ span is wider.** §5.5's run covered $10^5$–$10^7$/Mpc — two decades, the bottom of the
   range. This store spans the full production $10^5$–$3\times10^8$, so it includes hand-over
   geometry at $k$ the earlier census never saw, and the type mix may legitimately shift.

So the comparison you make is **of shape**: do the same types appear, in roughly the same rank
order, with the pathological classes still rare? For each cell that differs, say which of the three
accounts for it, or record it as unexplained. **An unexplained difference is a finding to state,
not a failure** — but §7's second stop condition is for a difference the three cannot account for,
and you should reach for that rather than inventing a fourth reason.

§4.2's figures, for reference: `numeric`/`complete` 270 (58.44 %), `WKB`/`complete` 123 (26.62 %),
`mixed`/`complete` 61 (13.20 %), `mixed`/`minimal` 1 (0.22 %), `fail`/`incomplete` 7 (1.52 %), of
462 rows.

---

## 3. The four things that will go wrong

1. **The transposition.** `GkSource` is at fixed $z_{\rm response}$ over a $z_{\rm source}$ grid,
   while `GkWKBIntegration` is at fixed $z_{\rm source}$ over a $z_{\rm response}$ grid
   (mechanism §0.1). The grid whose intervals you are counting is the **source** grid of the
   `GkSource` — `source.z_sample` — not any response grid. Getting this backwards produces a
   plausible table of wrong numbers.
2. **`primary_WKB_largest_z` is contiguous-from-the-bottom**, not simply the largest $z$ with a WKB
   value (`GkSource.py:243-246, 315-316`). A WKB island above a hole is excluded, and
   `assemble_GkSource_values` prints a warning when it sees one (`GkSource.py:150-164`). Count the
   non-contiguity warnings you would have triggered and report the number; if it is non-zero, that
   is a finding in its own right.
3. **`log(1+z) → z` is lossy and must not appear in an equality-like comparison** (campaign README
   §2 (n), `CLAUDE.md`). Count grid intervals by **node index in the stored `z_sample`**, matching
   on `store_id`, not by converting a $\log(1+z)$ width back to a redshift and dividing. The
   redshifts here reach $\sim10^{14}$, where recoverable $1+z$ has a granularity of ~0.02.
4. **The exit-time lookup.** `[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]`
   records that `analyse_greens_and_source.py:447` builds its `wavenumber_exit_time` lookup from
   `DEFAULT_ABS_TOLERANCE`/`DEFAULT_REL_TOLERANCE` rather than the sector's own pair. **Do not copy
   that line.** Use the same pair `main.py` uses for this target, and say in the document which
   pair you used and how you confirmed the rows you were served are the rows you asked for.

---

## 4. What to write down

`docs/handover/POLICY-GEOMETRY.md`, in the shape of `QUADSOURCE-READONLY.md`:

- a header block naming the tree SHA, the datastore, the model, and **the source-grid construction
  tag** the rows carry — a census of overlap widths is meaningless without saying which grid;
- **§0 — the answer to D5, in one paragraph**, in the form "on N `mixed` rows of this datastore,
  `crossover_z` has *(distribution)* grid intervals of numeric clearance and *(distribution)* of
  WKB clearance; M rows have fewer than two on one side; they are *(list)*";
- the tables, generated by the script, with a `<!-- generated by ... -->` marker;
- **§ — the relation between the two clearances.** The stored $c_{\rm num}$, $c_{\rm WKB}$ against
  the interval counts, as a scatter or a binned table. The question a reader will ask is whether
  the band thresholds happen to track grid intervals on the grid that actually ships. **State what
  you find. Do not assert a relationship you have not measured, and do not recommend a threshold
  change** — README §0.4 and §5 rule 8; this document measures and hands over.
- **§ — what this does not cover**, the honest closing section. At minimum: one datastore, one grid
  construction, whichever cosmologies it holds; the $T_k$ seam is *not* covered (it has no overlap
  to measure — mechanism §5); and the census says nothing about whether the end-interval error
  *matters*, only whether the geometry exposes it.

---

## 5. What this prompt does not do

- It does not touch any production file. Not `GkSourcePolicyData.py`, not `main.py`, not
  `config/`, not `MetadataConcepts/`.
- **It does not fix the two defects of README §2 (o)**, although it is reading exactly that code.
  Open them as §3 issues on the board, named
  `[03-gksource-policy-accepts-a-value-that-raises]` and
  `[03-quadsource-policy-vocabulary-differs]`, with the file:line evidence and nothing more.
  Workstream **C2** owns the fix. Fixing them here would destroy the revert-per-prompt property
  for a change that has no measurement attached.
- It does not change any threshold, band or policy value, and does not recommend one.
- It does not re-run the source integral, the Green's function or anything else that computes. It
  reads stored rows and, where it must, rebuilds `GkSourceFunctions` from them as
  `analyse_greens_and_source.py`'s `gk_continuity` does.
- It does not edit `analyse_greens_and_source.py` or `docs/source-remediation-verification.md`.
  That campaign is closed and its documents are its record.
- It adds no test. Suite counts must be **unchanged**, not risen.

---

## 6. Acceptance

1. One command, from the repository root, recorded with its datastore and runtime.
2. The type/quality census is compared against verification §5.5's **as a shape on a different
   store**, per §2.2, with each difference either accounted for or recorded as unexplained. An
   unexplained difference is a finding to state, not a failure — but it must be stated.
3. `numeric_intervals` and `WKB_intervals` are reported as distributions over all `mixed` rows,
   and every row with fewer than two on either side is tabulated individually.
4. `docs/handover/POLICY-GEOMETRY.md` exists, its §0 answers D5 in one paragraph, and it has a
   closing "what this does not cover".
5. §2.1's two questions are answered: the overlap's position in e-folds against the three
   landmarks, and whether `incomplete` clusters in $z_{\rm response}$ or in $k$. Both are reported
   as measurements with the derivation of README §2 (q) quoted beside them.
6. The two §3 issues are opened on the board and indexed in `docs/OPEN_ISSUES.md`, and **no
   production file is in the diff**.
7. `ComputeTargets` and `CosmologyModels` counts unchanged from the orchestrator's baselines.
8. `black --check` clean on the new script.
9. Board and `docs/OPEN_ISSUES.md` updated in the same commit; README §7 **D5** marked answered
   with a pointer to the new document.

---

## 7. Stop conditions — stop and ask the user

- **No datastore is available**, or the only one available was written before
  `qcd-background-audit` prompt 15 replaced the grid density criterion. A census on the superseded
  grid answers a question nobody asked. Report which grid the rows carry and stop.
- **The type/quality census differs from verification §5.5's shape in a way §2.2's three known
  differences do not account for** — in particular a `fail`/`incomplete` fraction materially above
  §4.2's 1.52 %, or `mixed` rows materially scarcer than its 13.42 %. Report which cells differ,
  and do not re-baseline. A shape that differs *consistently with* a wider $k$ span or the prompt-15
  grid is not this condition; a shape that differs beyond them is.
- **More than a few percent of `mixed` rows have fewer than two grid intervals on a side.** That
  would mean the $T_k$ end-interval analysis applies to $G$ as well and with larger absolute errors
  (follow-up §3 item 1), which changes D1's scope and E2's target. Report it and stop; do not
  propose a remedy.
- **`primary_WKB_largest_z` is non-contiguous on a material fraction of rows.** That is a producer
  problem, not a policy one, and it is outside this prompt.
- **§2.1's lever map is falsified** — the half-e-fold band of README §2 (q) is not present on every
  row, or the `fail`/`incomplete` rows cluster in $k$ rather than in $z_{\rm response}$. B4 is
  planned on that derivation and §7 **D7** is deferred pending it, so a disagreement changes two
  pieces of the campaign plan. Report it and stop; do not correct §2 (q) yourself.

---

## 8. The log

`logs/03-policy-geometry-census.md`, template as `qcd-background-audit` README §5.1. Beyond it:
the datastore and grid tag the census ran on; the census against verification §5.5 cell by cell;
the count of non-contiguity warnings; and the exit-time pair used, with how it was confirmed
(§3 item 4).

Board: this prompt's row, the item-level entry for **A3**, and the two §3 issues of §5. On
`docs/OPEN_ISSUES.md`: the two new rows, the count and the date. **D5 closes on the campaign
README**, not on the index — it is a decision, not an issue.
