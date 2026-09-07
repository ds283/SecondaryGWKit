# Prompt 12 — Verification pass (audit §4; whole campaign)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** §4 (four UNVERIFIED items), §7; every prompt's "Verification performed"
**Depends on:** 01–11 (hard)
**Recommended model:** Opus (a live Ray run with judgement about what is and is not worth
running; not design work)
**Files you may touch:** new `docs/source-remediation-verification.md`, new scripts under
`docs/source-remediation-verification/` (or `tools/`), the campaign log and status board. **No
production code.** If verification reveals a defect, open a §3 issue with the reproduction and stop;
do not fix it in this prompt.

Read first: `docs/backport-modules-verification.md` and `prompts/backport-modules/10-verification.md`
— the format and the "scoped-down driver" approach to a live run that this prompt follows; every
log in `logs/`; `main.py`'s CLI (`:80-230`) to see which queues can be enabled individually.

---

## 1. Character of this commit

Two layers of verification, then a written record.

**Layer 1 — everything offline, all at once.** Run the full unit-test tree
(`CosmologyModels/tests`, `ComputeTargets/tests`, `AdaptiveLevin/tests`, plus any test module
prompt 10 added) and all 29 audit scripts in `docs/spec-code-audit/scripts/`; several of those
scripts measured the *defects* (e.g. `QS_03_spline_error.py`, `QS_04_coverage_and_jacobian.py`,
`TK_05/06/07`, `QI_04`) and must now show them fixed or be documented as measuring pre-fix
behaviour by design. Tabulate script → expectation → observed.

**Layer 2 — one live, scoped pipeline run.** The backport campaign's verification (§5 there)
established that a real `RayWorkPool`/`ShardedPool` run needs a driver outside `main.py` because
the wavenumber sample size is hardcoded at 50+50. Write such a driver (`tools/` or the
verification directory), on a plain `LambdaCDM` (Planck 2018) model **and**, if time allows, a
`LambdaCDM_GenericEOS` with the radiation EOS (which exercises prompts 01 and 03), with:

- 6–8 log-spaced wavenumbers over two decades, chosen so that at least one triangle-closing triple
  falls in each of the three shapes prompt 08 tested ($q\approx r\approx k$; $q\approx r\gg k$;
  $q\ll k\approx r$);
- the production redshift density (100 per decade) but a shorter range if needed to keep the run
  under ~1–2 hours on the available cores;
- all queues enabled in pipeline order: background → Tk numeric → Tk WKB → QuadSource → Gk numeric
  → Gk WKB → GkSource/policy → QuadSourceIntegral;
- a fresh datastore (existing ones are stale/unreadable after prompts 06 and 09; say so).

If a Ray cluster or a writable datastore location is not available, **stop and ask** (README §4.1);
do not fake Layer 2 with stubs.

## 2. What Layer 2 must measure

For each `QuadSourceIntegral` row produced:

1. `total` vs `analytic_rad` (the model is constant-$w$ where the source is; the radiation era is
   the relevant regime — restrict the comparison to `z_response` deep in radiation domination, or
   report the matter-era discrepancy separately as expected) — relative difference and the ratio to
   `total_abserr`. This is the end-to-end closure of A2 and A4: **before this campaign the numeric
   branch could not have matched the oracle sub-horizon.** Quote the pre-campaign figure if any
   pre-campaign database with `analytic_rad` is available to you (`test-qcd-db*.sqlite` in the
   repository root may predate the schema change; the backport verification says they use the old
   shard schema — check, and do not spend long on it).
2. The regime mix from `metadata["partition"]`: how many sub-intervals of each README §6 row
   occurred. Confirm the "$G$ smooth, both $T$ oscillatory" row occurs for the $q\approx r\gg k$
   shape.
3. **Audit §4.1 — continuity of $G$ at `crossover_z`.** For every `GkSourcePolicyData` of type
   `mixed`, evaluate `functions.numeric_Gk` and `functions.WKB_Gk` at `crossover_z` and report the
   relative difference distribution (median, worst, with the offending $(k, z_{\rm resp})$).
4. **Audit §4.2 — reachability of A6.** Count `GkSourcePolicyData` rows with `quality == "minimal"`
   and report whether any exist.
5. **Audit §4.3 — `has_WKB_violation`.** Count `TkWKBIntegration`/`GkWKBIntegration` rows with it
   set, list the `(k, WKB_violation_efolds_subh)` pairs, and — for one such $k$ if any — compare
   `T_WKB` against `analytic_T_rad` just past the violation point. Report only; no policy change.
6. **Audit §4.4 — A7 on a GenericEOS run** (only if the GenericEOS run happened): sample
   `BackgroundModelValue` at the two grid ends and compare `epsilon`, its derivatives and
   `d2_wPerturbations_dz2` with the analytic `LambdaCDM` values at the same $z$ (the radiation EOS
   makes them identical models).
7. Prompt 06 §3.4's observation: the numeric-spline residual near the hand-over, measured on real
   `QuadSource` rows against `analytic_source_rad`.
8. Cost: wall time per `QuadSourceIntegral`, split by regime, and the Levin evaluation counts from
   the aggregated `WKB_Levin_*` columns; compare with prompt 08's offline estimates.

## 3. The written record

`docs/source-remediation-verification.md`, in the format of `docs/backport-modules-verification.md`:
what was run, exact commands, the tables from §1 and §2, and an explicit list of what remains
unverified (with a cost estimate) — e.g. a QCD-EOS run, the full 50+50 grid, anything Layer 2 could
not reach. Close or narrow each audit §4 item on the board with a pointer into this document.

## 4. Log and commit

Log to `logs/12-verification.md`. Board: row 12; items §4.1–§4.4; **Progress** line; move resolved
§3 issues to §4; set the campaign-complete note if everything passed, or list what did not. One
commit containing the verification document, the driver script(s) and the board; the body
summarises the headline `total` vs `analytic_rad` agreement and anything that failed.
