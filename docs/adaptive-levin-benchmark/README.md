# AdaptiveLevin performance campaign — handoff for correctness review

This folder was exported from a Claude Science project to inform a re-examination of
`AdaptiveLevin/levin_quadrature.py` for correctness, in light of empirical findings from a
benchmarking campaign. Start here:

1. **`ADAPTIVE-LEVIN-REVIEW.md`** — the pre-campaign static code review that motivated the
   benchmarking (written before any measurement).
2. **`LEVIN-PERFORMANCE-REPORT.md`** — the campaign report (final version), with figures
   embedded via relative paths (`figs/`) so it renders in-repo and on GitHub.
3. **`HANDOFF-PROVENANCE.md`** — full provenance: file inventory with checksums, dataset/
   figure lineage, software environment, and version history.

## The one caveat that matters most for a correctness review

**The measured code does not correspond to current `HEAD`, and it does not correspond to
any single commit.** Commit `7f5b0a5` is the best inferred proxy for the state that was
benchmarked (see §3 of `HANDOFF-PROVENANCE.md` for the reasoning). Current `HEAD` (`b76570e`)
was authored *after* the report and already implements part of the report's own §7
recommendations — so diffing `HEAD` against the report's findings is a before/after
comparison, not a check of what was measured. If you need to reproduce a number from the
report exactly, check out `7f5b0a5` into a clean worktree first.

## Key findings to weigh against the code (see report for detail)

- The Levin core is frequency-independent to ~1e-16 given an exact phase; all high-frequency
  error degradation traces to double-precision endpoint phase evaluation, not the Levin
  collocation itself.
- The per-region error estimator's endpoint-phase rounding is common-mode and cancels
  exactly — the estimator measures collocation error alone. This is consistent with Chen,
  Serkh & Bremer (arXiv:2211.13400v3, p.30), not a defect.
- `eps * theta_max` is a good empirical health-check bound (0.25 margin across 35 tested
  high-frequency cells) — empirical, not a fitted law.
- Levin's advantage is nonlinear-phase problems; on linear phase, scipy's QAWO rule beats it
  by 7–9 orders of magnitude at ω=1e12.
- Recommended-but-not-yet-implemented-at-campaign-close (check current `HEAD` — some of this
  may already be done): return `sum(max(r.abserr, r.phase_err))` plus a `phase_limited` flag;
  raise the default Chebyshev order above 12 (accuracy is order-independent, so this only
  buys cheaper subdivision); A/B a rank-revealing QR against the SVD/pinv solve per the
  paper's Remark 2 (~5× reported, no apparent accuracy loss).

## Supporting data

- `results/` — the ten per-experiment CSVs plus their concatenation. Filter `status=="ok"`
  before computing accuracy statistics; only `tierA_ladder.csv` has quotable wall-clock
  timings (run serially, alone) — other tables' `time_s`/`phase_build_s`/`total_s` were
  measured concurrently with a background scan and are not valid timing comparisons.
- `figs/` — the four report figures (PNG).
- `levin_bench/` — the harness that produced the tables and figures (`python -m
  levin_bench.campaign figures` redraws the figures from the CSVs alone).
- `provenance/plan_benchmark-adaptivelevin.json` — the approved experiment plan the campaign
  executed against.

Not included here: the source paper (arXiv:2211.13400v3, Chen/Serkh/Bremer, CC BY-NC-ND
4.0) — fetch it directly from arXiv rather than vendoring a third-party PDF into this repo.
