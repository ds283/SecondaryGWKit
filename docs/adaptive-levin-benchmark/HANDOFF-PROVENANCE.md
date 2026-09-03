# Handoff and provenance record — AdaptiveLevin performance campaign

**Purpose.** This document is the provenance record for the `levin_bench` performance
campaign and its report, prepared so the work can be re-seeded into a new project on a
different account without loss of traceability. Everything described here is present in
the accompanying bundle; nothing needs to be reconstructed from conversation history.

Prepared 2026-09-03. Source project `proj_90530d5e557f`.

---

## 1. Origin of the work

All measurements, code and figures were produced in a single working session
(`6d00fbd3-9fd1-42e6-ab00-63301cf7c0c1`, 2026-09-02 12:03Z – 20:42Z). Two documents
predate the campaign proper and are included because the report depends on them:

| item | role |
|---|---|
| `ADAPTIVE-LEVIN-REVIEW.md` | static code review of `AdaptiveLevin/levin_quadrature.py` that motivated the campaign; written 2026-09-02 12:01Z, before any measurement |
| `provenance/plan_benchmark-adaptivelevin.json` | the approved experiment plan the campaign executed against; useful as a statement of intent, superseded by the report |
| `reference/2211.13400v3.pdf` | Bremer, Chen & Yang, arXiv:2211.13400v3 — the source paper for the algorithm. Third-party material, uploaded by the user and included only so page-number citations in the report can be checked; re-obtainable from arXiv and not for redistribution |

The report itself went through three versions in-session; only the final one (v3,
2026-09-02 20:42Z) is bundled. The v1→v3 rewrite is substantive rather than cosmetic and
is described in §9.

---

## 2. Inventory

26 files, 3,598,413 bytes. Every file was copied out of the artifact store and its
SHA-256 re-computed after copying; all 26 match the checksum recorded in the store, so the
bundle is byte-identical to the artifacts as saved.

| path | bytes | sha-256 (first 12) | role |
|---|---|---|---|
| `levin_bench/problems.py` | 9,822 | `c7b488d80e6f` | 5 synthetic integrands with closed-form oracles (validated to 60 digits vs mpmath); `select_omega` builds the frequency ladder |
| `levin_bench/runners.py` | 10,350 | `133df100e8cd` | uniform `run_levin`/`run_quad`/`run_qawo` wrappers, common record schema, wall-clock budget guard |
| `levin_bench/sweeps.py` | 10,704 | `71afdb076863` | tier-A experiments: ladder, Pareto, tolerance, Chebyshev order, phase floor, reduction cost, estimator fidelity |
| `levin_bench/bessel_tier.py` | 11,305 | `adc97e517b56` | tier-B experiments against the 7 closed-form three-Bessel oracles |
| `levin_bench/campaign.py` | 3,923 | `734bc2e641b8` | driver and the shared `_write` CSV helper |
| `levin_bench/figures.py` | 13,021 | `deb23b4f296a` | redraws all four report figures standalone from the CSVs |
| `levin_bench/README.md` | 3,588 | `80ee091d881f` | harness documentation: requirements, entry points, conventions, timing caveat |
| `results/tierA_ladder.csv` | 51,374 | `a8497097028a` | frequency ladder — the only source of quotable wall-clock numbers |
| `results/tierA_pareto.csv` | 49,137 | `a254826460d3` | accuracy-cost Pareto scan |
| `results/tierA_tolerance.csv` | 29,761 | `221099a4f54c` | tolerance map (atol/rtol response) |
| `results/tierA_order.csv` | 21,264 | `3c48f0e9f5e7` | Chebyshev order scan |
| `results/tierA_estimator.csv` | 18,292 | `41b962cd7c27` | estimator fidelity: internal estimate vs true error |
| `results/tierA_phase_floor.csv` | 14,616 | `e0f791a81c24` | phase-evaluation floor experiment (mechanism, report SS3) |
| `results/tierA_reduction_cost.csv` | 1,166 | `2555f7be8394` | cost of range reduction vs plain fmod |
| `results/tierB_truncation.csv` | 18,397 | `3be112d06e23` | three-Bessel truncation scan |
| `results/tierB_production.csv` | 5,452 | `8b4f875b97a1` | production-scale wavenumber scan |
| `results/tierB_head_to_head.csv` | 5,641 | `a21d50f6758e` | Levin vs brute-force `quad` on three-Bessel products (12 quad timeouts) |
| `results/levin_bench_results.csv` | 264,620 | `c3538e3fabe7` | concatenation of all ten tables above; convenience view, not an independent measurement |
| `figs/fig1_ladder.png` | 481,650 | `2ce8d65d9928` | report figure 1 — frequency ladder |
| `figs/fig2_estimator.png` | 433,743 | `dcb2ec26a933` | report figure 2 — estimator reliability |
| `figs/fig3_bessel_cost.png` | 308,222 | `da076fc8c311` | report figure 3 — Bessel tier and cost |
| `figs/fig4_mechanism.png` | 230,633 | `f13a5d6f1ad3` | report figure 4 — mechanism (error is in the phase) |
| `LEVIN-PERFORMANCE-REPORT.md` | 23,963 | `700ae82f71f5` | the report (v3, final) |
| `ADAPTIVE-LEVIN-REVIEW.md` | 32,485 | `d8eb52771a5c` | pre-campaign static code review |
| `reference/2211.13400v3.pdf` | 1,533,838 | `02b1b6ac3f29` | source paper, arXiv:2211.13400v3 (third-party; do not redistribute) |
| `provenance/plan_benchmark-adaptivelevin.json` | 11,446 | `385a1e197a58` | approved experiment plan |

---

## 3. The one dependency that is *not* in the bundle

`levin_bench` is self-contained as a harness but it measures code that lives in the
user's own repository, which the bundle does not and should not vendor:

```
problems.py     → LiouvilleGreen.range_reduce_mod_2pi
runners.py      → AdaptiveLevin.adaptive_levin_sincos
sweeps.py       → AdaptiveLevin.adaptive_levin_sincos, LiouvilleGreen.range_reduce_mod_2pi
bessel_tier.py  → LiouvilleGreen.bessel_phase, LiouvilleGreen.three_bessel_integrals,
                  LiouvilleGreen.tests.test_3bessel_analytic (Jintegrals, Yintegrals)
```

Repository: `SecondaryGWKit`, `https://github.com/ds283/SecondaryGWKit.git`,
branch `fix-bessel-phase-and-optimize-levin`.

**The measured source state does not correspond to any commit, and it is not the current
HEAD.** This is the single most important caveat in this document, because re-running the
campaign against HEAD will not reproduce the numbers in the report. The evidence:

- Last measurement written: 2026-09-02 19:44Z (tier B) / 15:59Z (tier A).
- Commits `4f7ce8a`, `03c6bf7`, `ae88de4`, `b28d3c1`, `cc64ae4`, `7f5b0a5` were all
  authored between 20:12:48Z and 20:14:05Z — six commits in 77 seconds, ~28 minutes after
  the last measurement. That timing pattern is the signature of a batch commit of
  working-tree content that already existed, not of fresh editing. The inference (not a
  provable fact) is that **`7f5b0a5` is the closest committed proxy for the code that was
  measured.** The nearest committed *ancestor* is `11fa287` (2025-12-15), which is
  certainly *not* what ran: `11fa287..7f5b0a5` changes 467 lines across
  `levin_quadrature.py`, `bessel_phase.py`, `three_bessel_integrals.py` and the
  three-Bessel tests.
- HEAD is now `b76570e` (2026-09-03 01:52+0100), authored *after* the report. It adds 337
  lines to `AdaptiveLevin/levin_quadrature.py` under the message "report an error estimate,
  account for phase rounding, attribute the algorithm" — i.e. it implements part of the
  report's own §7 change list. Re-running `campaign all` against `b76570e` therefore
  measures a *different, already-improved* integrator, and any comparison with the report's
  tables is a before/after comparison, not a reproduction.

Recommended handling in the new project: record `7f5b0a5` as the measurement baseline,
re-run against `b76570e` only as a deliberate before/after, and if a true reproduction is
ever needed, check out `7f5b0a5` into a clean worktree.

---

## 4. Software environment

The campaign ran in a conda environment named `python` with Python 3.11.16, on macOS
(darwin, 10 cores, 16 GiB). Versions of the packages the harness imports, as installed in
that environment:

| package | version at campaign time |
|---|---|
| python | 3.11.16 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pandas | 2.3.3 |
| matplotlib | 3.11.1 |
| mpmath | required by `problems.py` / `sweeps.py`; **not present in that environment now** — the repo's own `./venv` carries `mpmath==1.3.0`, which is the version to assume |

Two environment notes that matter for re-running:

- `scipy` version is load-bearing for tier A: `quad`, `qawo` and their self-reported error
  columns (`quad_reported_err`, `quad_warned`) are the comparison baseline in report §2 and
  §4, and QUADPACK wrapper behaviour has changed across scipy releases.
- The repository's `requirements.txt` pins an older stack (`numpy==2.2.4`, `scipy==1.15.2`,
  `matplotlib==3.10.1`, `pandas==2.2.3`, `mpmath==1.3.0`) than the environment that ran the
  campaign. The oracles are closed-form and validated to 60 digits, so accuracy columns are
  insensitive to this; wall-clock columns are not.

---

## 5. Dataset provenance

Ten per-experiment tables, plus one concatenation of all ten.

`results/levin_bench_results.csv` (892 rows × 62 columns) is exactly the row-wise
concatenation of the ten per-experiment tables — verified: per-file row counts sum to 892
and its `source_table` column reproduces each file's row count exactly. It is a
convenience view, not an independent measurement; the per-experiment tables are canonical.
Its column union is sparse by construction (a column present only in the tolerance sweep is
empty for ladder rows).

| table | rows | columns |
|---|---|---|
| `levin_bench_results.csv` | 892 | 62 |
| `tierA_ladder.csv` | 216 | 34 |
| `tierA_pareto.csv` | 195 | 35 |
| `tierA_tolerance.csv` | 117 | 29 |
| `tierA_order.csv` | 90 | 28 |
| `tierA_phase_floor.csv` | 80 | 15 |
| `tierB_truncation.csv` | 77 | 23 |
| `tierA_estimator.csv` | 60 | 32 |
| `tierB_head_to_head.csv` | 24 | 26 |
| `tierB_production.csv` | 21 | 24 |
| `tierA_reduction_cost.csv` | 12 | 5 |

Composition of the master table:

- `source_table` — `tierA_ladder` 216, `tierA_pareto` 195, `tierA_tolerance` 117,
  `tierA_order` 90, `tierA_phase_floor` 80, `tierB_truncation` 77, `tierA_estimator` 60,
  `tierB_head_to_head` 24, `tierB_production` 21, `tierA_reduction_cost` 12.
- `method` — `levin` 572, `quad` 147, `qawo` 81.
- `status` — `ok` 788, `timeout` 12, empty 92. The 12 timeouts are all in
  `tierB_head_to_head` (brute-force `quad` on three-Bessel products, 60 s guard). The 92
  empty values are the whole of `tierA_phase_floor` (80) and `tierA_reduction_cost` (12),
  which are derived/diagnostic tables that carry no status column of their own.
- `experiment` — empty for all 216 `tierA_ladder` rows; use `source_table`, not
  `experiment`, as the grouping key.

**Reading conventions inherited from the harness** (also stated in `levin_bench/README.md`):

- Filter `status == "ok"` before computing any accuracy statistic.
- Relative error is against the closed-form oracle. The `grz` problem family passes through
  accidental zeros of the integral at certain ω, where relative error is meaningless; read
  `abs_err` for those cells.
- Keys with a leading underscore (e.g. `_regions`, the per-region diagnostics) are
  deliberately excluded from the CSVs by `campaign._write`, to keep non-scalar payloads out
  of the tables. If per-region data is needed again it must be re-run, not recovered from
  these files.
- **Timing fidelity.** Only `tierA_ladder` yields quotable wall-clock numbers, and only
  because it was run alone and serially. The other sweeps were run concurrently with a
  background Bessel scan; their `time_s`, `phase_build_s` and `total_s` columns must not be
  cited as timing results. This constraint is a property of how the data was collected and
  cannot be recovered by re-analysis.

---

## 6. Figure provenance

Each figure is drawn by one function in `levin_bench/figures.py`, and that source is the
authoritative mapping — the platform's recorded dependency edges are incomplete here
(captured for `fig3` in full, partially for `fig2`, not at all for `fig1` and `fig4`, whose
extraction had not completed at save time). Both are given so the gap is visible rather
than silent:

| figure | drawn by | tables read (from `figures.py`) | recorded lineage edges |
|---|---|---|---|
| `fig1_ladder.png` | `figures.fig1_ladder()` | `tierA_ladder.csv` | _none captured_ |
| `fig2_estimator.png` | `figures.fig2_estimator()` | `tierA_estimator.csv`, `tierA_tolerance.csv` | `tierA_tolerance.csv` |
| `fig3_bessel_cost.png` | `figures.fig3_bessel_cost()` | `tierA_order.csv`, `tierA_reduction_cost.csv`, `tierB_truncation.csv` | `tierA_order.csv`, `tierA_reduction_cost.csv`, `tierB_truncation.csv` |
| `fig4_mechanism.png` | `figures.fig4_mechanism()` | `tierA_phase_floor.csv` | _none captured_ |

Note that `fig3_bessel_cost.png` is named for its subject but is drawn from two tier-A
tables as well as the tier-B truncation scan.

`levin_bench/figures.py` redraws all four figures standalone from the CSVs
(`python -m levin_bench.campaign figures`), so figures are regenerable from the bundle alone
without the repository or a re-run.

---

## 7. Version history

Ten of the 26 artifacts have more than one version. The four figures each have four
versions, but the pixel content changed only once (v1→v2, a restyle); v2, v3 and v4 are
byte-identical for `fig1`, and for `fig2`–`fig4` v3 and v4 are byte-identical to v2. The
repeated saves are re-versionings that accompanied each report revision, not new renders.
Full history:

| artifact | v | bytes | saved (UTC) | sha-256 (first 12) | note |
|---|---|---|---|---|---|
| `LEVIN-PERFORMANCE-REPORT.md` | 1 | 15,978 | 2026-09-02T19:54:52Z | `89b97f03fe18` |  |
| `LEVIN-PERFORMANCE-REPORT.md` | 2 | 18,760 | 2026-09-02T20:09:15Z | `3788d5a064dd` |  |
| `LEVIN-PERFORMANCE-REPORT.md` | 3 | 23,963 | 2026-09-02T20:42:30Z | `700ae82f71f5` | current |
| `bessel_tier.py` | 1 | 11,305 | 2026-09-02T19:44:45Z | `adc97e517b56` |  |
| `bessel_tier.py` | 2 | 11,305 | 2026-09-02T19:54:52Z | `adc97e517b56` | current; byte-identical to v1 |
| `campaign.py` | 1 | 4,583 | 2026-09-02T15:59:12Z | `2547df515f9d` |  |
| `campaign.py` | 2 | 3,923 | 2026-09-02T19:54:52Z | `734bc2e641b8` | current |
| `fig1_ladder.png` | 1 | 481,650 | 2026-09-02T15:58:15Z | `2ce8d65d9928` | current |
| `fig1_ladder.png` | 2 | 456,010 | 2026-09-02T19:55:05Z | `2630b0d670aa` |  |
| `fig1_ladder.png` | 3 | 456,010 | 2026-09-02T20:09:21Z | `2630b0d670aa` | byte-identical to v2 |
| `fig1_ladder.png` | 4 | 456,010 | 2026-09-02T20:42:54Z | `2630b0d670aa` | byte-identical to v2 |
| `fig2_estimator.png` | 1 | 433,743 | 2026-09-02T15:58:15Z | `dcb2ec26a933` | current |
| `fig2_estimator.png` | 2 | 416,238 | 2026-09-02T19:55:05Z | `bbb619ffecc8` |  |
| `fig2_estimator.png` | 3 | 416,238 | 2026-09-02T20:09:21Z | `bbb619ffecc8` | byte-identical to v2 |
| `fig2_estimator.png` | 4 | 416,238 | 2026-09-02T20:42:55Z | `bbb619ffecc8` | byte-identical to v2 |
| `fig3_bessel_cost.png` | 1 | 308,222 | 2026-09-02T15:58:15Z | `da076fc8c311` | current |
| `fig3_bessel_cost.png` | 2 | 298,304 | 2026-09-02T19:55:05Z | `fef551f8dfa4` |  |
| `fig3_bessel_cost.png` | 3 | 298,304 | 2026-09-02T20:09:21Z | `fef551f8dfa4` | byte-identical to v2 |
| `fig3_bessel_cost.png` | 4 | 298,304 | 2026-09-02T20:42:54Z | `fef551f8dfa4` | byte-identical to v2 |
| `fig4_mechanism.png` | 1 | 230,633 | 2026-09-02T19:44:34Z | `f13a5d6f1ad3` | current |
| `fig4_mechanism.png` | 2 | 211,969 | 2026-09-02T19:55:05Z | `8b8fd45e71fd` |  |
| `fig4_mechanism.png` | 3 | 211,969 | 2026-09-02T20:09:21Z | `8b8fd45e71fd` | byte-identical to v2 |
| `fig4_mechanism.png` | 4 | 211,969 | 2026-09-02T20:42:54Z | `8b8fd45e71fd` | byte-identical to v2 |
| `problems.py` | 1 | 9,822 | 2026-09-02T15:59:12Z | `c7b488d80e6f` |  |
| `problems.py` | 2 | 9,822 | 2026-09-02T19:54:52Z | `c7b488d80e6f` | current; byte-identical to v1 |
| `runners.py` | 1 | 10,350 | 2026-09-02T15:59:12Z | `133df100e8cd` |  |
| `runners.py` | 2 | 10,350 | 2026-09-02T19:54:52Z | `133df100e8cd` | current; byte-identical to v1 |
| `tierA_reduction_cost.csv` | 1 | 1,166 | 2026-09-02T15:59:05Z | `2555f7be8394` |  |
| `tierA_reduction_cost.csv` | 2 | 1,166 | 2026-09-02T19:44:34Z | `2555f7be8394` | current; byte-identical to v1 |

The remaining 16 artifacts have a single version.

---

## 8. Re-seeding the new project

The bundle is laid out as a working directory, so the fastest route is to unzip it, then
upload the whole tree into the new project. Two mechanical points:

**Figure embeds in the report will not resolve.** `LEVIN-PERFORMANCE-REPORT.md` embeds its
four figures by artifact ID, which are specific to the old project:

| embed caption | old artifact ID | filename to point at |
|---|---|---|
| Frequency ladder | `art_8a274d76-c25a-4f5c-a256-754e89c725f1` | `figs/fig1_ladder.png` |
| Mechanism | `art_312a4634-165e-4dcc-8f74-d733121cdd7d` | `figs/fig4_mechanism.png` |
| Estimator reliability | `art_d9207193-46e5-427c-8f06-889914ce65c3` | `figs/fig2_estimator.png` |
| Bessel tier and cost | `art_98007c3f-063f-40ba-ad64-da94e1f6f7d4` | `figs/fig3_bessel_cost.png` |

After uploading the figures into the new project, replace each `art_…` ID in the report with
the new project's artifact ID for the same filename. The bundle also contains
`REPORT-relative-paths.md`, identical in text but with the four embeds rewritten
to relative paths (`figs/fig1_ladder.png`, …); that variant renders correctly in any
markdown viewer, on GitHub, and inside the repository, and is the better choice if the
report's long-term home is the repo rather than a project workspace.

**What cannot be carried over.** Three things live in the platform rather than in files and
will not survive the account move:

1. *Artifact lineage.* The dependency edges in §6 and the per-artifact reproduction code are
   platform metadata. Re-uploaded files arrive with no lineage; §6 of this document is the
   durable replacement.
2. *Project memory.* 30 recorded facts about this project. The load-bearing ones are
   distilled in §9 below — that section exists so they can be re-stated to a fresh session
   in one paste.
3. *Conversation history.* The report, the review document and this record are the intended
   substitutes; no result in the report depends on transcript content.

---

## 9. Findings and working conventions worth re-establishing

Condensed from the project's recorded knowledge. These are conclusions and conventions, not
raw data; each is either stated in the report or directly supports reading it.

**On the integrator (the substance of the report).**

- The Levin core is frequency-independent to machine precision (≈1e-16 at ω=1e12) given an
  exact phase. All high-frequency degradation lives in the double-precision endpoint phase:
  a region's interior depends only on θ′, while θ itself enters only at region endpoints.
- The per-region error estimate is a parent-vs-children difference, and because a region's
  value is a difference of endpoint quantities only, bisection cancels the interior point and
  parent and children are built from identical θ(a), θ(b). Endpoint phase rounding is
  therefore common-mode and subtracts out exactly, so the estimator measures collocation
  error alone — which is why delivered error is identical from Chebyshev order 4 to 32.
- This is *correct behaviour*, not a defect. Chen et al. v3 p.30 explicitly anticipates the
  conditioning loss (condition number growing with |g|, principal loss at the endpoint
  formula (171), absolute error predicted to stay flat with frequency) and their eq. 151
  bounds only the discretization error. §4 of the report was rewritten to say this; an
  earlier draft called the convergence test an invalid estimate and an extraction-blocking
  defect, and both characterisations were wrong. Measurements were unchanged by the rewrite —
  only their interpretation.
- `eps·θ_max` is the recommended health-check predicate: it bounded delivered relative error
  with 0.25 margin in all 35 high-frequency cells tested. Empirical upper bound, not a fitted
  law — the ratio varies by several decades across problems.
- Levin's defensible territory is nonlinear phase. Where the phase is linear, scipy's QAWO
  rule beats adaptive Levin by 7–9 orders of magnitude at ω=1e12. Against brute-force `quad`
  on three-Bessel products, `quad` converged in 0 of 12 cells (60 s timeouts) while Levin gave
  8–9 digits in under 1 s.
- In the three-Bessel application the scalability frontier is `bessel_phase`
  (Liouville–Green) construction, not the Levin core: 92% of total cost at κ=100, and a
  κ=1000 build did not complete in ~25 minutes. It also sets a ≈2e-8 accuracy floor uniform
  across all seven closed-form oracles — a property of this repository's phase construction,
  not of Levin quadrature in general.

**Recommended but not implemented at the time the campaign closed** (note that HEAD
`b76570e` has since implemented at least the first): return an aggregate
`sum(max(r.abserr, r.phase_err))` plus a `phase_limited` flag, framed as an extension for
safe reuse rather than a bug fix; raise the default Chebyshev order above 12, since accuracy
is order-independent and higher order cuts subdivision work for free; and A/B a
rank-revealing QR against the SVD/pinv solve, following the paper's Remark 2, where the
authors report ~5× with no apparent accuracy loss.

**House conventions to re-state to a fresh session.**

- For oscillatory phases, pass the raw large argument to libm (Payne–Hanek reduction against
  multi-hundred-bit π); never pre-reduce with `fmod` against a 53-bit `TWO_PI`, whose error
  grows with the number of cycles removed. Reduce only when a (cycle count, bounded
  remainder) representation is genuinely needed, e.g. for splining.
- Published numerical-analysis work is a strong prior. When a measurement appears to
  contradict a paper's error estimate, read the paper's prose discussion — limitations are
  often documented outside the formal bounds — and distinguish what the paper claims from the
  regime the measurement probes, before concluding the estimate is invalid.
- Page-number citations in the report are keyed to arXiv:2211.13400**v3**; numbering differs
  in v1 and v2.
- Repository practicalities: no `pytest` in `./venv` (use `./venv/bin/python -m unittest`),
  `PYTHONPATH=.` is required or imports fail, and importing
  `Quadrature.integrators.WKB_phase_function` directly raises a pre-existing circular
  `ImportError` — import `ComputeTargets` first.

---

## 10. Verifying the bundle

`MANIFEST.csv` lists every file with its source artifact ID, version number, byte size and
SHA-256. To confirm nothing was altered in transit:

```sh
cd <unzipped bundle>
python3 - <<'EOF'
import csv, hashlib
bad = []
for r in csv.DictReader(open("MANIFEST.csv")):
    h = hashlib.sha256(open(r["path"], "rb").read()).hexdigest()
    if h != r["sha256"]:
        bad.append(r["path"])
print("mismatches:", bad or "none")
EOF
```

The same SHA-256 values are the store-recorded checksums of the source artifacts, so a clean
run of the above establishes the chain from the new project's files back to the artifacts
this campaign produced.
