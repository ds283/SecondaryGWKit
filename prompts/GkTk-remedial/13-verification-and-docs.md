# Prompt 13 — Verification on both models, and the close-out documents

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §4 and §12.3 (the production-path measurements to repeat), §13.5 (the six
planning points to close)
**Depends on:** everything (01–12).
**Recommended model:** Opus
**Files you may touch:** new `docs/gktk-remedial-verification.md`, new scripts under
`docs/gktk-remedial/`, **dated additive notes** in `docs/gk-wkb-review-fable-2026-09-09.md` (a
short §14 pointing at the verification document; nothing above it rewritten),
`docs/spec/02-greens-function.md` §0 (one dated sentence where it says `BackgroundModel`
"integrates $d(a_0\eta)/dz=-1/H$"), `docs/OPEN_ISSUES.md`, plus the log and the status board.
**Do not touch:** production code (any defect found here is a §3 issue, not a fix);
`docs/lg-phase-and-handover-followup-2026-09.md` (`transfer-remedial` prompt 09 edits it);
`main.py`.

Read first: README §6 (every row is re-measured here), §4.2 item 3; `CLAUDE.md` ("verification
documents are additive"; `docs/source-remediation-verification/scoped_pipeline_run.py` for the
scoped run); every log's "State handed to the next prompt"; `docs/source-remediation-verification.md`
as the model for the document's shape.

---

## 1. Layer 1 — production-path measurements, offline

`docs/gktk-remedial/verify_production_path.py`: through the **production** functions (undecorated
`WKB_phase_function`, `compute_background`, the `store()` algebra of both WKB classes,
`PrimitivePhase`, `TkSourceFunctions`), on `LambdaCDMModel` and `QCDModel`, for
$k\in\{10^5, 10^7, 3\times10^8\}$:

1. The review's §4 and §12.3 tables re-measured: phase error at the same six/seven $z$ rows
   against prompt 01's references, for $G_k$ and $T_k$; cost per object (wall time, integrand
   evaluations). Present the "before" (review) and "after" columns side by side.
2. The consumer at production $x$: `PrimitivePhase` (Green's function at fixed $z_r=0.1$ over the
   full source grid, from stand-in `GkSourceValue`s built by the production `store()` algebra)
   against the reference phase at 10 points per interval; same for `TkSourceFunctions.phase`.
3. $F$ and $\rho_T$ as stored, against the references.
4. Every row of README §6 with its measured value, the (model, $k$, $z$) of the maximum, and
   whether the target and the floor were respected.

## 2. Layer 2 — a scoped pipeline run per model

With `docs/source-remediation-verification/scoped_pipeline_run.py`, on a **fresh** datastore
(the schema changed in 03/04 and the $T_k$ tolerance key in 12), for **each** of `LambdaCDM` and
`QCD_Cosmology` separately, a small $k$ set (e.g. `--k-min 1e5 --k-max 1e7 --k-count 5`) at the
production `--zend 0.1` and `--source-samples-log10z 100`, through at least the background,
$T_k$ numeric/WKB and $G_k$ numeric/WKB stages (use whatever stage-selection flags `main.py`
offers; read its argument parser). Record:

- wall time of the $G_k$ WKB and $T_k$ WKB stages (review §4: ~13 CPU-hours per $k$ at
  $3\times10^8$ before — you will not reach that $k$; compare at the $k$ you ran using the review's
  per-object costs);
- the number of `has_unresolved_osc` warnings printed (D2 in production terms);
- the `BackgroundModel` row's `tau_Mpc`/`tau_lo_Mpc` at three nodes against the references;
- a sample of stored `GkWKBValue`/`TkWKBValue` rows against Layer 1 at the same $(k, z_s, z_r)$
  (they must agree bit-for-bit — the same code ran);
- that `GkSourcePolicyData` and `QuadSourceIntegral` stages complete on the new phase objects, if
  the scoped run reaches them, with the `WKB_phase_spline_chunks` column now `1`.

If the QCD run fails for a reason unrelated to this campaign, record it as a §3 issue and run the
LambdaCDM half.

## 3. `docs/gktk-remedial-verification.md`

Shape: §0 what was verified and the SHAs; §1 Layer 1 tables (before/after); §2 Layer 2 (commands
run verbatim, datastore paths, timings, warnings); §3 the README §6 acceptance table with measured
values; §4 what remains open (every §3 board issue, with a one-line status); §5 reproduction.

## 4. Documentation notes (additive)

- `docs/gk-wkb-review-fable-2026-09-09.md`: append **§14 — Outcome (dated)** with the after-column
  headline numbers and a link to the verification document. Do not edit §§0–13.
- `docs/spec/02-greens-function.md` §0.1 item (1): where it says `ComputeTargets/BackgroundModel.py`
  "integrates $d(a_0\eta)/dz=-1/H$", add a dated parenthesis: since commit `<03's SHA>` the
  conformal time is a per-interval Gauss–Legendre table; the convention $\tau=a_0\eta$ is
  unchanged.
- `docs/OPEN_ISSUES.md`: close every row this campaign resolved (delete rows; correct the count and
  date); leave the hand-over rows and the follow-ups opened by planning that remain open
  (`[00-consumer-anchoring-floor]`, `[00-tk-lg-truncation-floor]`, `[00-tk-superhorizon-ic-series]`,
  and `[00-unresolved-osc-print-policy]` unless the user has decided it).
- `main.py:516-519` (the Bessel $Q$ comment): **not** this campaign's — note in the verification
  document that `transfer-remedial` prompt 06 owns it.

## 5. Verification and acceptance

- Layer 1 script runs from the repository root and reproduces every Layer 1 table.
- Every README §6 row is reported with a measured value; any miss is a §3 issue and the log's
  Result is `COMPLETE WITH DEVIATIONS` at best — **do not loosen a target here**.
- Layer 2 ran for LambdaCDM (and QCD, or the reason it did not).
- `docs/OPEN_ISSUES.md` count and date correct; every closed board item is in the board's §4.

## 6. Log and commit

The log's "State handed to the next prompt" section is titled "State of the tree at close" and
lists: every SHA of the campaign; the datastore regeneration requirement; the open issues and who
owns them.

Commit subject, or something equally specific: `Verify the Gk/Tk WKB remediation against both background models`.
