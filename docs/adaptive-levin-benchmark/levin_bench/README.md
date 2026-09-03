# `levin_bench` — performance harness for adaptive Levin quadrature

A self-contained benchmark and regression suite for the adaptive Levin
integrator (`AdaptiveLevin/levin_quadrature.py`). Built to characterise accuracy
and cost against brute-force adaptive quadrature as oscillation frequency grows;
intended to travel with the integrator when it is extracted into a standalone
package, where it doubles as a regression benchmark.

Findings are written up in `../LEVIN-PERFORMANCE-REPORT.md`.

## Requirements

- The `SecondaryGWKit` checkout on `PYTHONPATH` (for `AdaptiveLevin`,
  `LiouvilleGreen`, and the closed-form Bessel oracles in
  `LiouvilleGreen/tests/test_3bessel_analytic.py`).
- `numpy`, `scipy`, `mpmath`, `pandas`, `matplotlib`.

## Running

```sh
export PYTHONPATH=/path/to/SecondaryGWKit:$PWD
python -m levin_bench.campaign all       # everything (~40 min)
python -m levin_bench.campaign tierA     # synthetic problems only
python -m levin_bench.campaign tierB     # Bessel-product tier only
python -m levin_bench.campaign figures   # redraw figures from existing CSVs
```

Individual experiments can be run directly:

```sh
python -m levin_bench.sweeps ladder pareto tolerance order phase_floor \
                             reduction_cost fidelity
python -m levin_bench.bessel_tier truncation production head
```

Tables land in `../results/`, figures in `../figs/`.

## Timing fidelity

**Only the frequency ladder (`sweeps ladder`) produces quotable wall-clock
numbers, and only when run alone.** Every other experiment is designed for
accuracy and region-structure data, and may be run concurrently. If you re-run
the ladder to compare timings, run nothing else at the same time — the numbers
in the report were taken that way.

## Layout

| file | contents |
|---|---|
| `problems.py` | five synthetic integrands with closed-form oracles, each validated against 60-digit `mpmath` quadrature; `select_omega` builds the frequency ladder |
| `runners.py` | uniform wrappers `run_levin` / `run_quad` / `run_qawo`, all returning the same record schema (value, errors, timings, evaluation counts, solver diagnostics, status) plus a wall-clock budget guard |
| `sweeps.py` | tier-A experiments: frequency ladder, accuracy–cost Pareto, tolerance map, Chebyshev-order scan, phase-evaluation floor, range-reduction cost, estimator fidelity |
| `bessel_tier.py` | tier-B experiments against the seven closed-form three-Bessel oracles: truncation scan, production-scale wavenumber scan, head-to-head vs `quad` |
| `campaign.py` | driver plus the shared `_write` CSV helper |
| `figures.py` | redraws every report figure from the saved CSVs |

## Conventions

- Every runner returns the same flat dict, so records concatenate directly into
  one CSV per experiment. Keys with a leading underscore (e.g. `_regions`,
  the retained per-region diagnostics) are deliberately excluded from CSVs by
  `_write` to keep non-scalar payloads out of the tables.
- `status` is `ok`, `budget` (wall-clock guard tripped), or an exception name.
  Filter on `status == "ok"` before computing accuracy statistics.
- Relative error is computed against the closed-form oracle. Where an oracle
  passes through an accidental zero of the integral (the `grz` family does, at
  certain ω), relative error is not meaningful; those cells are marked and
  should be read via `abs_err`.
- `quad_reported_err` and `quad_warned` record what `scipy` said about its own
  result, so the honesty of each method's self-assessment can be compared —
  this is what §2 and §4 of the report rest on.
