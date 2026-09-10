# Orchestrator prompt — Workstream F (prompt 13)

You are orchestrating Workstream F of the Gk/Tk WKB phase remedial campaign in the repository at
`/Users/ds283/Documents/Code/SecondaryGWKit` (branch `gktk-remedial`). You do not write code
yourself.

One prompt: re-measure the review's production-path tables through the shipped code, run the
pipeline scoped on a fresh datastore for both models, and write the verification document and the
additive notes. **Nothing in production code may change here**; a defect found becomes a §3 issue.
The review's numbers are additive documents — nothing above §14 in the review is rewritten.

## What to read

`../README.md` §4.1, §4.2 item 3, §4.3, §5, **§6 (every row is re-measured here)**;
`../RECONCILIATION.md` §0; `../IMPLEMENTATION_STATE.md` — the whole board and every §3 entry;
`orchestrator/README.md`; every log's "State handed to the next prompt"; `CLAUDE.md` (verification
documents are additive; `main.py` cannot be imported; the scoped-run script).

## Preconditions

`git status` clean; rows 01–12 ✅/⚠️ (or the user has explicitly accepted running verification
with a named row incomplete — record it). The full offline suite passes:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t . 2>&1 | tail -3
PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_phase_spline LiouvilleGreen.tests.test_range_reduce -v
```

Confirm a scratch location for two fresh datastores exists and that Ray can start locally
(`docs/source-remediation-verification/scoped_pipeline_run.py --help`).

## Dispatching

Standard dispatch text (`workstream-A.md`). Model: **13 → Opus**.

## Reviewing prompt 13

Structural checks; allowed files: `docs/gktk-remedial-verification.md`, scripts under
`docs/gktk-remedial/`, the review (**appended §14 only** — `git diff HEAD~1 -- docs/gk-wkb-review-fable-2026-09-09.md`
must show additions after §13 and no deletions), `docs/spec/02-greens-function.md` (one dated
parenthesis in §0.1), `docs/OPEN_ISSUES.md`, log, board. **No production module, no `main.py`, no
`docs/lg-phase-and-handover-followup-2026-09.md`** (`transfer-remedial` 09's).

4. The Layer 1 script runs from the repository root and its tables match the document:
   ```bash
   PYTHONPATH=. ./venv/bin/python docs/gktk-remedial/verify_production_path.py 2>&1 | tail -40
   ```
6. **Every README §6 row appears with a measured value** in the document's §3, with the
   (model, $k$, $z$) of the maximum and a pass/miss against target *and* floor. A missed row with
   the Result still `COMPLETE` is a stop; a missed row recorded as a §3 issue with
   `COMPLETE WITH DEVIATIONS` is the correct outcome and you report it. **No target was loosened**
   (compare the document's targets to README §6 verbatim).
7. Layer 2 ran for LambdaCDM on a fresh datastore with the commands quoted verbatim; QCD ran or
   the reason is a §3 issue. The `WKB_phase_spline_chunks` column reads 1; stored rows agree with
   Layer 1 bit-for-bit at shared $(k, z_s, z_r)$; the `has_unresolved_osc` warning count is
   reported (D2 in production terms).
8. `docs/OPEN_ISSUES.md`: count and date correct; every row deleted corresponds to a board §4
   entry; the four planning follow-ups that remain open are still listed unless the user closed
   them.
9. The `transfer-remedial` Bessel-stage comment is recorded as theirs, not edited.

## Continue or stop

Stop on the campaign-wide conditions, on check 6's silent miss, or on any production-code change.

## Completion criterion

Row 13 ✅/⚠️. Report: "Campaign complete; the tree is at `<SHA>`. Verification document at
`docs/gktk-remedial-verification.md`. Open follow-ups: `<list from the board §3>`. Every datastore
predating `<03's SHA>` must be regenerated." Include the before/after headline numbers for
$\theta_G$ and $\theta_T$ at $z=0.1$, $k=10^5$ and $3\times10^8$, and the per-object cost.
