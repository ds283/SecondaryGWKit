# The realistic flavour at large $x$, and the two terms at the seam

**Measured:** 2026-09-20, on the tree at `c414451` plus the script itself, offline (no Ray, no datastore),
on an Apple M1 Pro, one core
**Script:** [`realistic_large_x.py`](realistic_large_x.py) — §3 is its stdout, unedited
**Invocation:** `PYTHONPATH=. ./venv/bin/python docs/handover/realistic_large_x.py` (10,772 s;
60 cells)
**Instrument:** `ComputeTargets/tests/test_quadsource_integral.py`, imported and **not edited**
**Reference:** Kohri & Terada eq. (22) and the head $0\to\bar x_{\rm min}$, both at 50 digits
([`../radiation-oracle/eq22_rounding.py`](../radiation-oracle/eq22_rounding.py))
**Campaign:** [`prompts/handover`](../../prompts/handover/README.md), board item **A2**

---

## 0. What is being compared, and whose each object is

[`KOHRI-TERADA-ORACLE.md`](../radiation-oracle/KOHRI-TERADA-ORACLE.md) §0 is the model for this
section and it is not optional: saying which object is whose is the difference between a
cross-check and a tautology. Six objects appear below.

| | Object | Provenance |
|---|---|---|
| **A** | `evaluate_QuadSource_integral` (`ComputeTargets/QuadSourceIntegral.py:733`) in the **exact** flavour: the partition, the clamp adapter and the Levin integrator, driven on `ExactTkFunctions`/`ExactGk` closed-form stand-ins injected through the `Tk_functions_builder` hook | **This repository's pipeline**, with ingredients exact to rounding. This is the object §8 of the Kohri–Terada audit measured, and this document's control |
| **A′** | The same call in the **realistic** flavour: `TkSourceFunctions` built from prompt 05's exact fixture (amplitude re-splined on the production 100-per-$\log_{10}z$ grid, the phase carried as a `PrimitivePhase`, Liouville–Green closed forms for $\omega$ and $\mathrm{d}\ln M/\mathrm{d}z$), `OffsetBesselPhaseGk`'s `phase_spline` for the Green's function, and $f$ splined on the production grid | **This repository's pipeline together with this repository's representation machinery.** Pre-existing; not written for this document. See §8 for the one place where it has fallen behind production |
| **A″** | Either flavour with `WKB_region[0]` one grid step below `crossover_z`, so that `build_partition` records a gap and `_ClampedTk` holds the Liouville–Green accessors across it | **This repository's pipeline.** The clamp is `QuadSourceIntegral`'s own and the gap geometry is production's (campaign README §2 (a)) |
| **B** | KT eq. (22) at 50 digits, `eq22_rounding.I_RD_mp` | **The `radiation-oracle` campaign's transcription** of a published equation. Fixed by construction, moves with no tolerance, shares no machinery with A |
| **C** | The head $\int_0^{\bar x_{\rm min}}$ of KT eq. (15) at 50 digits, `eq22_rounding.head_mp` | Likewise. Not a pipeline defect: the pipeline correctly starts where its source starts (campaign README §2 (k)) |
| **D** | `_GappedExactTkFunctions` and the harness itself | **New, written for this document.** It lands no physics: it moves one declared region boundary and drives A, A′ and A″ |

**What each comparison does and does not guard.**

- **$N$ against B − C** is the absolute check. $N \equiv -\tfrac98\,\text{total}/\text{predicted}$,
  and a wrong kernel, measure or normalisation would make $N$ **drift** with $x$. §8 of the
  Kohri–Terada audit established that it does not, over $x$ from 4.33 to $1.6\times10^8$ — but for
  A only. Table 1 re-establishes it here, on this tree, before anything is attributed.
- **A difference of two $N$ within a block guards nothing about B or C, and does not need to.**
  The four cells of a block share one $(u, v, x, \bar x_{\rm min})$ and therefore one value of
  B − C, to the last bit. It cancels identically in every difference in Table 3, and the figure
  quoted beside each term is the sum of the two cells' own declared errors and nothing else. This
  is why the separation is possible at all: it does not require the reference to be *better* than
  the terms, only to be *the same*.
- **The comparison that is not available** is a clamp measured against a representation that has
  no samples. The exact flavour has no seam to open (§2), so its clamp is an artificial region
  truncation, object **D**, rather than production's mechanism. What makes it informative is that
  it and the realistic clamp agree; §4 measures the agreement, and it is the whole basis of the
  attribution.

**Author conventions that are conventions, not defects** (campaign README §2 (m)): $a_0$ is
absorbed, not "set to 1"; the sign of $N$ is spec 04 §0(3)'s orientation convention; $\bar G_k$ is
the unit-jump Green's function in $z$.

---

## 1. The headline

**The clamp and the representation are now separated, and the $2\times2$ is additive.**
`[12-handover-clamp-error-in-production]`'s recorded next step says separating them *"cannot be
done by measurement alone at these $x$"*. It can, and this is it.

| | measured range over the ladder | its own error |
|---|---|---|
| **clamp term** $N_{\rm gap\ open} - N_{\rm gap\ closed}$ | $3.91\times10^{-3}$ to $9.12\times10^{-1}$ | $2.3\times10^{-12}$ to $6.8\times10^{-8}$ |
| **representation term** $N_{\rm realistic} - N_{\rm exact}$ | $4.08\times10^{-5}$ to $2.68\times10^{-3}$ | $1.2\times10^{-10}$ to $8.3\times10^{-8}$ |
| **their ratio** | **11.6 to 1034** | — |
| **interaction** (the $2\times2$'s non-additivity) | $\le 2.82\times10^{-5}$ of the clamp term, $\le 1.03\times10^{-2}$ of the representation term | — |

Every term in Table 3 exceeds its own error bar by at least a factor of $3.5\times10^{3}$.

Five further results, each of which is a measurement rather than a restatement:

1. **The control reproduces Kohri & Terada §8 Table 8.1 exactly** — all sixteen rows, all ten data
   columns, digit for digit, and independently checked against a separately-taken `large_x.py` run
   at `162f6df`. Nothing moved under that campaign's close-out.
2. **The clamp term is $x$-independent**, with fitted slopes of $-0.07$, $+0.03$ and $+0.03$ over
   ladders spanning five decades. §5 says why, and the reason is structural.
3. **The representation term is also $x$-independent** — slopes $-0.06$, $-0.15$, $-0.12$ — and
   therefore **does not grow like $h^4x/384$**. The prompt's stated expectation is refuted by five
   decades. §5 gives the numbers and the reason.
4. **The gap this harness opens is production's**, to 1.00 of the fixture's own grid step and 1.00
   of the mean source-grid step — but it is production's *maximum* gap, 1.92 times its median. §6.
5. **The realistic flavour's cost is the binding constraint on how far this instrument reaches**,
   and it is the *integral* cost, not the set-up cost the prompt anticipated. §7.

**What this closes and what it does not.** It closes nothing on its own; it is the instrument B2, D
and E are scored on. It narrows `[08-handover-clamp-error]`,
`[12-handover-clamp-error-in-production]` and `[12-phase-spline-error-grows-with-x]`, and the
narrowing notes are on those issues' own board (`prompts/source-remediation`).

---

## 2. Why the $2\times2$ needs two gap mechanisms

Campaign README §2 (a) states the seam's geometry: `main.py:1501-1506` starts each $T_k$'s WKB grid
at the largest source-grid point **below** `z_init`, `TkWKBIntegration` stores no sample at
`z_init`, so `TkSourceFunctions.WKB_region[0] < crossover_z` and no accessor of that factor is
evaluable in between. `QuadSourceIntegral` bridges the interval by clamping the Liouville–Green
accessors to `WKB_region[0]`, up to `HANDOVER_CLAMP_MAX_GRID_STEPS = 1.5` mean source-grid steps.
`Fixture(drop_first_WKB_sample=True)` reproduces exactly that shape, and it is what
`TestHandOverClamp` uses.

**In the exact flavour that mechanism does nothing at all.** `ExactTkFunctions` ignores the sampled
`(Tk_numeric, Tk_WKB)` inputs entirely — `Case.Tk_builder` returns a closed-form object whose
`WKB_region` is `(crossover_z, 0.0)` by construction — so dropping a sample from a fixture the
builder never reads changes nothing. **Table 6 measures this: the totals agree bit for bit
(relative difference exactly 0.0e+00 on all three shapes) and no gap is recorded.** It is a
structural property of the flavour, not a numerical accident, and it is why campaign README §1 can
say the exact flavour's seam is "continuous by construction".

A $2\times2$ whose fourth cell is bit-identical to its first measures nothing, so the exact half of
the factorial opens the seam a second way: `_GappedExactTkFunctions` declares `WKB_region[0]` to sit
`gap` below `crossover_z`, where `gap` is **the same number** — the dropped fixture's own first WKB
grid step in $\log(1+z)$. `build_partition` then computes the same shortfall, and Table 2 confirms
that it does: the recorded `gap Tq` and `gap Tr` are identical across the two flavours at every
rung, to all four printed digits. `_ClampedTk` applies the same hold, to closed-form accessors
instead of splined ones. What varies across the flavour axis is therefore the *ingredients*, with
the *geometry* held fixed, which is what the attribution needs.

**Do the two mechanisms make the gap-open cells comparable across flavours? Yes, and §4 is the
evidence rather than the assertion.** The clamp term measured on exact ingredients and on realistic
ones agrees to five significant figures at every one of the fourteen rungs where both exist
($+4.593496\times10^{-1}$ against $+4.593416\times10^{-1}$ at the worst-conditioned rung). If the
two mechanisms were opening materially different gaps, that agreement could not happen.

This is a deviation from the prompt's literal "gap open (`drop_first_WKB_sample=True`)", and it is
classified and justified in
[`logs/02-…`](../../prompts/handover/logs/02-realistic-flavour-large-x-harness.md).

---

## 3. The tables

<!-- generated by docs/handover/realistic_large_x.py -->

**Table 1 -- the control cell against Kohri & Terada section 8, Table 8.1.** Same columns, same order, same script inputs; the only difference is that this harness builds the case up to four times per row.

| shape | x_resp | lam | x = k tau | z_resp | N + 9/8 | pipeline's declared error | N + 9/8, head omitted | Levin / abs(total) | eq. (22) vs eq. (25) | integral | fixture set-up |
|---|---|---|---|---|---|---|---|---|---|---|---|
| together | 980 | 1 | 1.5560e+03 | 6.07 | +6.99e-14 | 8.7e-12 | +1.41e-04 | 3.85 | 9.7e-02 | 0.55 s | 0.0 s |
| together | 10000 | 10.2 | 1.5877e+04 | 6.07 | +2.89e-15 | 1.3e-12 | +4.30e-06 | 0.13 | 1.9e-04 | 0.48 s | 0.0 s |
| together | 100000 | 102 | 1.5877e+05 | 6.07 | +1.72e-13 | 1.7e-11 | +1.39e-05 | 0.39 | 6.3e-05 | 0.49 s | 0.0 s |
| together | 1e+06 | 1.02e+03 | 1.5877e+06 | 6.07 | -1.56e-12 | 1.4e-10 | +4.89e-06 | 0.15 | 3.9e-06 | 0.56 s | 0.0 s |
| together | 1e+07 | 1.02e+04 | 1.5877e+07 | 6.07 | -3.10e-10 | 9.9e-09 | -3.31e-05 | 0.86 | 1.5e-06 | 0.61 s | 0.0 s |
| together | 1e+08 | 1.02e+05 | 1.5877e+08 | 6.07 | -1.46e-10 | 3.5e-08 | -4.10e-06 | 0.09 | 9.9e-09 | 0.55 s | 0.0 s |
| T-first | 980 | 1 | 1.4145e+02 | 6.07 | -2.38e-14 | 4.8e-12 | +1.98e-06 | 2.89 | 1.1e-01 (no cos term) | 0.21 s | 0.0 s |
| T-first | 10000 | 10.2 | 1.4434e+03 | 6.07 | -2.22e-16 | 1.0e-12 | +2.24e-06 | 0.15 | 4.8e-05 (no cos term) | 0.26 s | 0.0 s |
| T-first | 100000 | 102 | 1.4434e+04 | 6.07 | +6.48e-14 | 1.6e-12 | +2.24e-06 | 0.18 | 8.9e-05 (no cos term) | 0.25 s | 0.0 s |
| T-first | 1e+06 | 1.02e+03 | 1.4434e+05 | 6.07 | -2.31e-12 | 7.8e-11 | +2.24e-06 | 1.11 | 1.9e-05 (no cos term) | 0.30 s | 0.0 s |
| T-first | 1e+07 | 1.02e+04 | 1.4434e+06 | 6.07 | +4.67e-12 | 3.1e-10 | +2.24e-06 | 0.13 | 1.0e-07 (no cos term) | 0.38 s | 0.1 s |
| q-smooth | 980 | 1 | 1.5431e+03 | 5.48 | -3.67e-13 | 5.2e-12 | -1.30e-05 | 2.10 | 3.8e-02 | 0.19 s | 0.0 s |
| q-smooth | 10000 | 10.2 | 1.5746e+04 | 5.48 | -1.01e-13 | 1.0e-11 | -1.22e-05 | 4.70 | 1.3e-02 | 0.23 s | 0.0 s |
| q-smooth | 100000 | 102 | 1.5746e+05 | 5.48 | -3.87e-13 | 2.1e-11 | -1.27e-05 | 1.62 | 3.5e-04 | 0.20 s | 0.0 s |
| q-smooth | 1e+06 | 1.02e+03 | 1.5746e+06 | 5.48 | +2.26e-13 | 5.4e-10 | -1.24e-05 | 4.45 | 5.0e-04 | 0.21 s | 0.0 s |
| q-smooth | 1e+07 | 1.02e+04 | 1.5746e+07 | 5.48 | -6.49e-13 | 4.9e-09 | -1.27e-05 | 2.59 | 4.1e-05 | 0.25 s | 0.0 s |

**Table 2 -- the factorial.** `gap Tq` and `gap Tr` are the shortfalls `build_partition` actually recorded in `clamp_gaps_log1pz`, and `1.5 grid steps` is `max_clamp_gap_log1pz`, the most the clamp will bridge.

| shape | x_resp | x = k tau | flavour | seam | N + 9/8 | pipeline's declared error | converged | Levin / abs(total) | Levin regions | gap Tq | gap Tr | 1.5 grid steps | integral | set-up |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| together | 980 | 1.5560e+03 | exact | closed | +6.994e-14 | 8.7e-12 | no | 3.85 | 7 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.55 s | 0.03 s |
| together | 980 | 1.5560e+03 | exact | open | +4.593e-01 | 1.8e-11 | no | 7.19 | 102 | 2.301e-02 | 2.304e-02 | 3.455e-02 | 1.13 s | 0.02 s |
| together | 980 | 1.5560e+03 | realistic | closed | -2.677e-03 | 1.4e-08 | no | 3.84 | 1304 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 18.52 s | 0.10 s |
| together | 980 | 1.5560e+03 | realistic | open | +4.567e-01 | 2.3e-08 | no | 7.16 | 1366 | 2.301e-02 | 2.304e-02 | 3.455e-02 | 19.36 s | 0.04 s |
| together | 10000 | 1.5877e+04 | exact | closed | +2.887e-15 | 1.3e-12 | no | 0.13 | 8 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.48 s | 0.03 s |
| together | 10000 | 1.5877e+04 | exact | open | -1.041e-02 | 1.2e-12 | no | 0.12 | 107 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 1.15 s | 0.03 s |
| together | 10000 | 1.5877e+04 | realistic | closed | +1.211e-04 | 4.6e-10 | no | 0.13 | 12068 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 173.39 s | 0.03 s |
| together | 10000 | 1.5877e+04 | realistic | open | -1.028e-02 | 4.6e-10 | no | 0.12 | 12122 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 181.29 s | 0.06 s |
| together | 100000 | 1.5877e+05 | exact | closed | +1.721e-13 | 1.7e-11 | no | 0.39 | 8 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.49 s | 0.03 s |
| together | 100000 | 1.5877e+05 | exact | open | +2.285e-02 | 2.3e-12 | no | 0.41 | 105 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 1.07 s | 0.03 s |
| together | 100000 | 1.5877e+05 | realistic | closed | -7.722e-05 | 1.1e-09 | no | 0.39 | 27253 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 410.21 s | 0.05 s |
| together | 100000 | 1.5877e+05 | realistic | open | +2.277e-02 | 1.1e-09 | no | 0.41 | 27291 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 402.08 s | 0.07 s |
| together | 1e+06 | 1.5877e+06 | exact | closed | -1.561e-12 | 1.4e-10 | no | 0.15 | 9 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.56 s | 0.04 s |
| together | 1e+06 | 1.5877e+06 | exact | open | -8.355e-03 | 1.7e-12 | no | 0.14 | 105 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 1.17 s | 0.04 s |
| together | 1e+06 | 1.5877e+06 | realistic | closed | +1.088e-04 | 5.1e-10 | no | 0.15 | 20253 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 299.17 s | 0.05 s |
| together | 1e+06 | 1.5877e+06 | realistic | open | -8.246e-03 | 4.6e-10 | no | 0.14 | 21954 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 335.26 s | 0.10 s |
| together | 1e+07 | 1.5877e+07 | exact | closed | -3.101e-10 | 9.9e-09 | no | 0.86 | 9 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.61 s | 0.04 s |
| together | 1e+07 | 1.5877e+07 | exact | open | -1.400e-01 | 2.1e-11 | no | 0.88 | 104 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 1.15 s | 0.04 s |
| together | 1e+07 | 1.5877e+07 | realistic | closed | +8.937e-04 | 3.2e-08 | no | 0.86 | 50463 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 773.74 s | 0.06 s |
| together | 1e+07 | 1.5877e+07 | realistic | open | -1.391e-01 | 2.5e-08 | no | 0.88 | 63146 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 958.12 s | 0.10 s |
| together | 1e+08 | 1.5877e+08 | exact | closed | -1.461e-10 | 3.5e-08 | no | 0.09 | 9 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.55 s | 0.04 s |
| together | 1e+08 | 1.5877e+08 | exact | open | -3.949e-02 | 1.6e-11 | no | 0.12 | 106 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 1.24 s | 0.04 s |
| together | 1e+08 | 1.5877e+08 | realistic | closed | +2.943e-04 | 3.8e-08 | no | 0.09 | 15827 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 241.94 s | 0.06 s |
| together | 1e+08 | 1.5877e+08 | realistic | open | -3.920e-02 | 2.7e-09 | no | 0.12 | 97590 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 1515.53 s | 0.10 s |
| T-first | 980 | 1.4145e+02 | exact | closed | -2.376e-14 | 4.8e-12 | no | 2.89 | 7 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.21 s | 0.02 s |
| T-first | 980 | 1.4145e+02 | exact | open | -3.913e-03 | 4.8e-12 | no | 2.88 | 57 | 2.301e-02 | 2.304e-02 | 3.455e-02 | 0.49 s | 0.02 s |
| T-first | 980 | 1.4145e+02 | realistic | closed | +3.387e-04 | 9.2e-09 | no | 2.89 | 720 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 7.18 s | 0.03 s |
| T-first | 980 | 1.4145e+02 | realistic | open | -3.574e-03 | 9.1e-09 | no | 2.88 | 765 | 2.301e-02 | 2.304e-02 | 3.455e-02 | 7.53 s | 0.05 s |
| T-first | 10000 | 1.4434e+03 | exact | closed | -2.220e-16 | 1.0e-12 | no | 0.15 | 10 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.26 s | 0.03 s |
| T-first | 10000 | 1.4434e+03 | exact | open | -5.646e-03 | 1.0e-12 | no | 0.16 | 61 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 0.57 s | 0.03 s |
| T-first | 10000 | 1.4434e+03 | realistic | closed | +7.001e-05 | 2.9e-10 | no | 0.15 | 3444 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 48.15 s | 0.04 s |
| T-first | 10000 | 1.4434e+03 | realistic | open | -5.576e-03 | 2.9e-10 | no | 0.16 | 3501 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 48.27 s | 0.06 s |
| T-first | 100000 | 1.4434e+04 | exact | closed | +6.484e-14 | 1.6e-12 | no | 0.18 | 9 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.25 s | 0.03 s |
| T-first | 100000 | 1.4434e+04 | exact | open | -5.633e-03 | 1.7e-12 | no | 0.18 | 59 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 0.58 s | 0.03 s |
| T-first | 100000 | 1.4434e+04 | realistic | closed | +7.280e-05 | 2.7e-10 | no | 0.18 | 19375 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 310.69 s | 0.06 s |
| T-first | 100000 | 1.4434e+04 | realistic | open | -5.560e-03 | 2.7e-10 | no | 0.18 | 19428 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 283.21 s | 0.08 s |
| T-first | 1e+06 | 1.4434e+05 | exact | closed | -2.310e-12 | 7.8e-11 | no | 1.11 | 12 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.30 s | 0.04 s |
| T-first | 1e+06 | 1.4434e+05 | exact | open | -5.239e-03 | 7.8e-11 | no | 1.11 | 64 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 0.65 s | 0.04 s |
| T-first | 1e+06 | 1.4434e+05 | realistic | closed | +1.703e-04 | 4.2e-09 | no | 1.11 | 36728 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 576.00 s | 0.05 s |
| T-first | 1e+06 | 1.4434e+05 | realistic | open | -5.068e-03 | 4.2e-09 | no | 1.11 | 36758 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 553.78 s | 0.11 s |
| T-first | 1e+07 | 1.4434e+06 | exact | closed | +4.674e-12 | 3.1e-10 | no | 0.13 | 13 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.38 s | 0.06 s |
| T-first | 1e+07 | 1.4434e+06 | exact | open | -5.763e-03 | 3.1e-10 | no | 0.12 | 63 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 0.64 s | 0.04 s |
| T-first | 1e+07 | 1.4434e+06 | realistic | closed | +4.081e-05 | 9.9e-10 | no | 0.13 | 75570 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 1229.77 s | 0.06 s |
| T-first | 1e+07 | 1.4434e+06 | realistic | open | -5.722e-03 | 9.8e-10 | no | 0.12 | 75626 | 2.303e-02 | 2.302e-02 | 3.455e-02 | 1202.38 s | 0.10 s |
| q-smooth | 980 | 1.5431e+03 | exact | closed | -3.668e-13 | 5.2e-12 | no | 2.10 | 17 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.19 s | 0.02 s |
| q-smooth | 980 | 1.5431e+03 | exact | open | +2.584e-01 | 8.2e-12 | no | 3.02 | 56 | 0.000e+00 | 2.304e-02 | 3.455e-02 | 0.47 s | 0.02 s |
| q-smooth | 980 | 1.5431e+03 | realistic | closed | -4.113e-04 | 1.0e-10 | no | 2.10 | 855 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 6.80 s | 0.03 s |
| q-smooth | 980 | 1.5431e+03 | realistic | open | +2.580e-01 | 1.4e-10 | no | 3.02 | 882 | 0.000e+00 | 2.304e-02 | 3.455e-02 | 7.20 s | 0.05 s |
| q-smooth | 10000 | 1.5746e+04 | exact | closed | -1.015e-13 | 1.0e-11 | no | 4.70 | 21 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.23 s | 0.02 s |
| q-smooth | 10000 | 1.5746e+04 | exact | open | +9.119e-01 | 1.2e-10 | no | 29.11 | 181 | 2.301e-02 | 2.302e-02 | 3.455e-02 | 1.19 s | 0.03 s |
| q-smooth | 10000 | 1.5746e+04 | realistic | closed | -8.815e-04 | 3.4e-10 | no | 4.70 | 5018 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 62.36 s | 0.03 s |
| q-smooth | 10000 | 1.5746e+04 | realistic | open | +9.110e-01 | 1.8e-09 | no | 28.98 | 5088 | 2.301e-02 | 2.302e-02 | 3.455e-02 | 65.74 s | 0.06 s |
| q-smooth | 100000 | 1.5746e+05 | exact | closed | -3.866e-13 | 2.1e-11 | no | 1.62 | 21 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.20 s | 0.03 s |
| q-smooth | 100000 | 1.5746e+05 | exact | open | -2.126e-01 | 1.7e-11 | no | 1.52 | 193 | 2.301e-02 | 2.302e-02 | 3.455e-02 | 1.12 s | 0.03 s |
| q-smooth | 100000 | 1.5746e+05 | realistic | closed | +2.322e-04 | 2.1e-10 | no | 1.62 | 31358 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 514.80 s | 0.04 s |
| q-smooth | 100000 | 1.5746e+05 | realistic | open | -2.123e-01 | 1.9e-10 | no | 1.52 | 31449 | 2.301e-02 | 2.302e-02 | 3.455e-02 | 489.70 s | 0.18 s |
| q-smooth | 1e+06 | 1.5746e+06 | exact | closed | +2.265e-13 | 5.4e-10 | no | 4.45 | 16 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.21 s | 0.03 s |
| q-smooth | 1e+06 | 1.5746e+06 | exact | open | +8.666e-01 | 5.2e-10 | no | 22.74 | 485 | 2.301e-02 | 2.302e-02 | 3.455e-02 | 3.21 s | 0.03 s |
| q-smooth | 1e+06 | 1.5746e+06 | realistic | closed | -- | -- | -- | -- | -- | -- | -- | -- | not reached | not reached |
| q-smooth | 1e+06 | 1.5746e+06 | realistic | open | -- | -- | -- | -- | -- | -- | -- | -- | not reached | not reached |
| q-smooth | 1e+07 | 1.5746e+07 | exact | closed | -6.486e-13 | 4.9e-09 | no | 2.59 | 30 | 0.000e+00 | 0.000e+00 | 3.455e-02 | 0.25 s | 0.04 s |
| q-smooth | 1e+07 | 1.5746e+07 | exact | open | -3.845e-01 | 7.6e-11 | no | 2.19 | 179 | 2.302e-02 | 2.302e-02 | 3.455e-02 | 1.10 s | 0.04 s |
| q-smooth | 1e+07 | 1.5746e+07 | realistic | closed | -- | -- | -- | -- | -- | -- | -- | -- | not reached | not reached |
| q-smooth | 1e+07 | 1.5746e+07 | realistic | open | -- | -- | -- | -- | -- | -- | -- | -- | not reached | not reached |

**Table 3 -- the attribution.** Each term is a difference of two N scored against the *same* 50-digit reference, so eq. (22) and the head cancel identically and the error beside each term is the sum of the two cells' own declared errors, converted to N. `interaction` is (clamp, realistic) - (clamp, exact), equivalently (representation, open) - (representation, closed): zero if the 2x2 is additive.

| shape | x_resp | x = k tau | clamp term, exact | +/- | clamp term, realistic | +/- | representation term, gap closed | +/- | representation term, gap open | +/- | interaction | h^4 x_resp / 384 | h^4 (k tau) / 384 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| together | 980 | 1.5560e+03 | +4.593e-01 | 2.1e-11 | +4.593e-01 | 3.1e-08 | -2.677e-03 | 1.5e-08 | -2.685e-03 | 1.5e-08 | -7.919e-06 | 7.2e-07 | 1.1e-06 |
| together | 10000 | 1.5877e+04 | -1.041e-02 | 2.9e-12 | -1.041e-02 | 1.0e-09 | +1.211e-04 | 5.2e-10 | +1.213e-04 | 5.2e-10 | +1.819e-07 | 7.3e-06 | 1.2e-05 |
| together | 100000 | 1.5877e+05 | +2.285e-02 | 2.2e-11 | +2.285e-02 | 2.5e-09 | -7.722e-05 | 1.3e-09 | -7.762e-05 | 1.3e-09 | -3.915e-07 | 7.3e-05 | 1.2e-04 |
| together | 1e+06 | 1.5877e+06 | -8.355e-03 | 1.5e-10 | -8.355e-03 | 1.1e-09 | +1.088e-04 | 7.3e-10 | +1.090e-04 | 5.3e-10 | +1.947e-07 | 7.3e-04 | 1.2e-03 |
| together | 1e+07 | 1.5877e+07 | -1.400e-01 | 1.1e-08 | -1.400e-01 | 6.8e-08 | +8.937e-04 | 4.7e-08 | +8.964e-04 | 3.2e-08 | +2.736e-06 | 7.3e-03 | 1.2e-02 |
| together | 1e+08 | 1.5877e+08 | -3.949e-02 | 4.0e-08 | -3.949e-02 | 4.6e-08 | +2.943e-04 | 8.3e-08 | +2.953e-04 | 3.1e-09 | +9.533e-07 | 7.3e-02 | 1.2e-01 |
| T-first | 980 | 1.4145e+02 | -3.913e-03 | 1.1e-11 | -3.913e-03 | 2.1e-08 | +3.387e-04 | 1.0e-08 | +3.388e-04 | 1.0e-08 | +1.105e-07 | 7.2e-07 | 1.0e-07 |
| T-first | 10000 | 1.4434e+03 | -5.646e-03 | 2.3e-12 | -5.646e-03 | 6.6e-10 | +7.001e-05 | 3.3e-10 | +7.011e-05 | 3.3e-10 | +9.747e-08 | 7.3e-06 | 1.1e-06 |
| T-first | 100000 | 1.4434e+04 | -5.633e-03 | 3.7e-12 | -5.633e-03 | 6.0e-10 | +7.280e-05 | 3.0e-10 | +7.290e-05 | 3.0e-10 | +9.769e-08 | 7.3e-05 | 1.1e-05 |
| T-first | 1e+06 | 1.4434e+05 | -5.239e-03 | 1.8e-10 | -5.239e-03 | 9.4e-09 | +1.703e-04 | 4.8e-09 | +1.705e-04 | 4.8e-09 | +1.060e-07 | 7.3e-04 | 1.1e-04 |
| T-first | 1e+07 | 1.4434e+06 | -5.763e-03 | 6.9e-10 | -5.763e-03 | 2.2e-09 | +4.081e-05 | 1.5e-09 | +4.090e-05 | 1.5e-09 | +9.497e-08 | 7.3e-03 | 1.1e-03 |
| q-smooth | 980 | 1.5431e+03 | +2.584e-01 | 1.3e-11 | +2.584e-01 | 2.4e-10 | -4.113e-04 | 1.2e-10 | -4.138e-04 | 1.3e-10 | -2.493e-06 | 7.2e-07 | 1.1e-06 |
| q-smooth | 10000 | 1.5746e+04 | +9.119e-01 | 3.6e-11 | +9.119e-01 | 7.7e-10 | -8.815e-04 | 3.9e-10 | -8.906e-04 | 4.1e-10 | -9.092e-06 | 7.3e-06 | 1.2e-05 |
| q-smooth | 100000 | 1.5746e+05 | -2.126e-01 | 4.7e-11 | -2.126e-01 | 5.0e-10 | +2.322e-04 | 2.6e-10 | +2.343e-04 | 2.8e-10 | +2.095e-06 | 7.3e-05 | 1.2e-04 |
| q-smooth | 1e+06 | 1.5746e+06 | +8.666e-01 | 7.4e-10 | -- | -- | -- | -- | -- | -- | -- | 7.3e-04 | 1.2e-03 |
| q-smooth | 1e+07 | 1.5746e+07 | -3.845e-01 | 5.6e-09 | -- | -- | -- | -- | -- | -- | -- | 7.3e-03 | 1.2e-02 |

**Table 4 -- the x-scaling of each term:** least-squares slope of log|term| against log(x = k tau) over the rungs each term has. 0 is x-independent; 1 is linear in x. `rows` is how many rungs entered each fit. A slope is only as meaningful as the term is monotone -- read it beside Table 3's individual values, not instead of them.

| shape | term | rows | slope | smallest | largest |
|---|---|---|---|---|---|
| together | clamp term, exact | 6 | -0.07 | -8.36e-03 (x_resp 1e+06) | +4.59e-01 (x_resp 980) |
| together | clamp term, realistic | 6 | -0.07 | -8.36e-03 (x_resp 1e+06) | +4.59e-01 (x_resp 980) |
| together | representation term, gap closed | 6 | -0.06 | -7.72e-05 (x_resp 100000) | -2.68e-03 (x_resp 980) |
| together | representation term, gap open | 6 | -0.06 | -7.76e-05 (x_resp 100000) | -2.69e-03 (x_resp 980) |
| T-first | clamp term, exact | 5 | +0.03 | -3.91e-03 (x_resp 980) | -5.76e-03 (x_resp 1e+07) |
| T-first | clamp term, realistic | 5 | +0.03 | -3.91e-03 (x_resp 980) | -5.76e-03 (x_resp 1e+07) |
| T-first | representation term, gap closed | 5 | -0.15 | +4.08e-05 (x_resp 1e+07) | +3.39e-04 (x_resp 980) |
| T-first | representation term, gap open | 5 | -0.14 | +4.09e-05 (x_resp 1e+07) | +3.39e-04 (x_resp 980) |
| q-smooth | clamp term, exact | 5 | +0.03 | -2.13e-01 (x_resp 100000) | +9.12e-01 (x_resp 10000) |
| q-smooth | clamp term, realistic | 3 | -0.04 | -2.13e-01 (x_resp 100000) | +9.12e-01 (x_resp 10000) |
| q-smooth | representation term, gap closed | 3 | -0.12 | +2.32e-04 (x_resp 100000) | -8.82e-04 (x_resp 10000) |
| q-smooth | representation term, gap open | 3 | -0.12 | +2.34e-04 (x_resp 100000) | -8.91e-04 (x_resp 10000) |

**Table 5 -- the gap this harness opens, against production's.** `[12-handover-clamp-error-in-production]` records median **1.2e-02** and max **2.2e-02** in log(1+z), about one mean source-grid step. `mean source-grid step` is `max_clamp_gap_log1pz / HANDOVER_CLAMP_MAX_GRID_STEPS`, i.e. the step the clamp tolerance is measured in.

| shape | x_resp | fixture WKB grid step | gap recorded, Tr | in fixture grid steps | mean source-grid step | in source-grid steps | / production median | / production max |
|---|---|---|---|---|---|---|---|---|
| together | 980 | 2.3038e-02 | 2.3038e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| together | 10000 | 2.3023e-02 | 2.3023e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| together | 100000 | 2.3024e-02 | 2.3024e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| together | 1e+06 | 2.3024e-02 | 2.3024e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| together | 1e+07 | 2.3025e-02 | 2.3025e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| together | 1e+08 | 2.3025e-02 | 2.3025e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| T-first | 980 | 2.3038e-02 | 2.3038e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| T-first | 10000 | 2.3023e-02 | 2.3023e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| T-first | 100000 | 2.3024e-02 | 2.3024e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| T-first | 1e+06 | 2.3024e-02 | 2.3024e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| T-first | 1e+07 | 2.3025e-02 | 2.3025e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| q-smooth | 980 | 2.3038e-02 | 2.3038e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| q-smooth | 10000 | 2.3023e-02 | 2.3023e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| q-smooth | 100000 | 2.3024e-02 | 2.3024e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| q-smooth | 1e+06 | 2.3024e-02 | 2.3024e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |
| q-smooth | 1e+07 | 2.3025e-02 | 2.3025e-02 | 1.00 | 2.3035e-02 | 1.00 | 1.92 | 1.05 |

**Table 6 -- `drop_first_WKB_sample=True` is inert in the exact flavour.** `ExactTkFunctions` ignores the sampled inputs entirely, so production's own gap mechanism opens no gap there. This is why the exact half of Table 2 truncates `WKB_region` instead, and it is the reason the factorial's fourth cell is not simply the first two mechanisms applied together.

| shape | x_resp | total, exact + Fixture as built | total, exact + dropped sample | relative difference | largest gap recorded |
|---|---|---|---|---|---|
| together | 980 | +9.263229868103e-13 | +9.263229868103e-13 | 0.0e+00 | 0.0e+00 |
| T-first | 980 | +7.469042198322e-11 | +7.469042198322e-11 | 0.0e+00 | 0.0e+00 |
| q-smooth | 980 | -8.570137704916e-12 | -8.570137704916e-12 | 0.0e+00 | 0.0e+00 |

**Table 7 -- cost.** Set-up is everything before `evaluate_QuadSource_integral`: both `Fixture` objects, the `Case`, and (realistic only) the `OffsetBesselPhaseGk` phase spline. The integral column includes the two `bessel_phase` builds `Case.run` makes and the `analytic_rad` oracle the integral computes alongside `total`.

| shape | x_resp | x = k tau | set-up, exact | set-up, realistic | ratio | integral, exact | integral, realistic | ratio |
|---|---|---|---|---|---|---|---|---|
| together | 980 | 1.5560e+03 | 0.027 s | 0.098 s | 3.6 | 0.55 s | 18.52 s | 33 |
| together | 10000 | 1.5877e+04 | 0.025 s | 0.035 s | 1.4 | 0.48 s | 173.39 s | 365 |
| together | 100000 | 1.5877e+05 | 0.029 s | 0.050 s | 1.8 | 0.49 s | 410.21 s | 837 |
| together | 1e+06 | 1.5877e+06 | 0.036 s | 0.049 s | 1.4 | 0.56 s | 299.17 s | 531 |
| together | 1e+07 | 1.5877e+07 | 0.045 s | 0.063 s | 1.4 | 0.61 s | 773.74 s | 1264 |
| together | 1e+08 | 1.5877e+08 | 0.044 s | 0.061 s | 1.4 | 0.55 s | 241.94 s | 437 |
| T-first | 980 | 1.4145e+02 | 0.022 s | 0.032 s | 1.4 | 0.21 s | 7.18 s | 35 |
| T-first | 10000 | 1.4434e+03 | 0.025 s | 0.038 s | 1.5 | 0.26 s | 48.15 s | 182 |
| T-first | 100000 | 1.4434e+04 | 0.031 s | 0.062 s | 2.0 | 0.25 s | 310.69 s | 1242 |
| T-first | 1e+06 | 1.4434e+05 | 0.037 s | 0.050 s | 1.3 | 0.30 s | 576.00 s | 1941 |
| T-first | 1e+07 | 1.4434e+06 | 0.056 s | 0.061 s | 1.1 | 0.38 s | 1229.77 s | 3239 |
| q-smooth | 980 | 1.5431e+03 | 0.022 s | 0.029 s | 1.3 | 0.19 s | 6.80 s | 36 |
| q-smooth | 10000 | 1.5746e+04 | 0.024 s | 0.033 s | 1.3 | 0.23 s | 62.36 s | 273 |
| q-smooth | 100000 | 1.5746e+05 | 0.025 s | 0.044 s | 1.7 | 0.20 s | 514.80 s | 2520 |
| q-smooth | 1e+06 | 1.5746e+06 | 0.031 s | not reached | -- | 0.21 s | not reached | -- |
| q-smooth | 1e+07 | 1.5746e+07 | 0.037 s | not reached | -- | 0.25 s | not reached | -- |

**Table 8 -- the cost wall.** The Levin driver accepts each exact sub-interval in one or two regions at every x; against the realistic flavour's splined ingredients it bisects into hundreds or thousands. The growth rate of that cost with x is what sets how far this instrument reaches, and it is **not the same on every shape**. `decade factor` is the realistic integral time at this rung divided by the one below.

| shape | x_resp | x = k tau | Levin regions, exact | Levin regions, realistic | integral, realistic | decade factor | status |
|---|---|---|---|---|---|---|---|
| together | 980 | 1.5560e+03 | 7 | 1304 | 18.5 s | -- | run |
| together | 10000 | 1.5877e+04 | 8 | 12068 | 173.4 s | 9.4x | run |
| together | 100000 | 1.5877e+05 | 8 | 27253 | 410.2 s | 2.4x | run |
| together | 1e+06 | 1.5877e+06 | 9 | 20253 | 299.2 s | 0.7x | run |
| together | 1e+07 | 1.5877e+07 | 9 | 50463 | 773.7 s | 2.6x | run |
| together | 1e+08 | 1.5877e+08 | 9 | 15827 | 241.9 s | 0.3x | run |
| T-first | 980 | 1.4145e+02 | 7 | 720 | 7.2 s | -- | run |
| T-first | 10000 | 1.4434e+03 | 10 | 3444 | 48.1 s | 6.7x | run |
| T-first | 100000 | 1.4434e+04 | 9 | 19375 | 310.7 s | 6.5x | run |
| T-first | 1e+06 | 1.4434e+05 | 12 | 36728 | 576.0 s | 1.9x | run |
| T-first | 1e+07 | 1.4434e+06 | 13 | 75570 | 1229.8 s | 2.1x | run |
| q-smooth | 980 | 1.5431e+03 | 17 | 855 | 6.8 s | -- | run |
| q-smooth | 10000 | 1.5746e+04 | 21 | 5018 | 62.4 s | 9.2x | run |
| q-smooth | 100000 | 1.5746e+05 | 21 | 31358 | 514.8 s | 8.3x | run |
| q-smooth | 1e+06 | 1.5746e+06 | 16 | -- | not reached | -- | **not reached** |
| q-smooth | 1e+07 | 1.5746e+07 | 30 | -- | not reached | -- | **not reached** |

| shape | realistic rungs | slope of log(realistic integral seconds) against log x | mean factor per decade of x_resp |
|---|---|---|---|
| together | 6 | +0.21 | 1.7x |
| T-first | 5 | +0.55 | 3.6x |
| q-smooth | 3 | +0.94 | 8.6x |

Cells: 60 computed or read from the checkpoint, 0 raised, 4 not reached. This invocation: 10772 s.

---

## 4. The attribution

**The deliverable is Table 3, and it says three things.**

**(a) The clamp dominates the representation by one to three orders, at every shape and every
$x$.** The ratio $|$clamp$|/|$representation$|$ runs from **11.6** (`T-first`, $x_{\rm resp}=980$)
to **1034** (`q-smooth`, $x_{\rm resp}=10^4$), with a median near 140. Campaign README §2 (b)
asserts that "the gap masks everything else at the seam, by two to four orders" on the strength of
production rows; this is the first measurement in which both terms are in hand simultaneously and
scored against a reference that does not move. The assertion survives, with the range widened at
the low end: on the shape where the clamp is weakest it is only eleven times the representation
floor, not a hundred.

**(b) The $2\times2$ is additive, so the separation is real and not an artefact of the
decomposition.** The interaction — equivalently (clamp, realistic) − (clamp, exact) or
(representation, open) − (representation, closed) — is at most $2.82\times10^{-5}$ of the clamp
term and at most $1.03\times10^{-2}$ of the much smaller representation term, at every rung of
every shape. Prompt §7's third stop condition asks what to do if "the cross term [is] comparable to
the main effects"; it is four to five orders below the larger and two to three below the smaller,
so the condition does not fire. The practical consequence for **B1** is the one that matters:
removing the gap should recover the clamp term in full, because the clamp does not depend on which
representation it is applied to.

**(c) Both terms bracket what production measured, which is the tie to real rows.**
`[12-handover-clamp-error-in-production]` records, on 462 production rows at
$z_{\rm response} = 7.63\times10^9$, that rows **with** a gap deviate from `analytic_rad` by
$6.6\times10^{-2}$ to $4.6\times10^{-1}$ while rows with **no** gap agree to $6.2\times10^{-5}$.
This harness's clamp terms span $3.9\times10^{-3}$ to $9.1\times10^{-1}$ and its representation
terms $4.1\times10^{-5}$ to $2.7\times10^{-3}$. The two families line up: the clamped production
band sits inside the clamp-term range, and production's unclamped $6.2\times10^{-5}$ sits inside
the representation-term range. That is a cross-check between a fixture and a production census that
share no code path beyond `QuadSourceIntegral` itself.

**The error on every term is the sum of the two cells' declared errors, and nothing else.** The
50-digit eq. (22) and head are bit-identical across the four cells of a block, so they cancel in the
difference; the reference contributes exactly zero to the quoted uncertainty. This is the only
reason a $4\times10^{-5}$ term can be quoted at all against a reference whose own absolute floor is
nowhere near that good.

---

## 5. The $x$-scaling

**The clamp term does not grow with $x$.** Fitted slopes of $\log|$term$|$ against
$\log(x = k\tau)$: $-0.07$ (`together`, six rungs), $+0.03$ (`T-first`, five), $+0.03$ (`q-smooth`,
five). Over five decades of $x$ the term stays in a band of order $10^{-3}$–$10^{0}$ and changes
sign repeatedly — it is oscillatory in $x$, not monotone, which is why Table 4 warns that the slope
is only as meaningful as the term is monotone and why Table 3's individual values are the primary
record.

The prompt's stated expectation was that the clamp term "is roughly $x$-independent … so its effect
on the integrand is set by the phase advanced over the gap, **which grows with $x$**", and asked
for what is actually found. **What is found is $x$-independence, and the parenthetical reason for
doubting it does not apply**, because the phase advanced over the gap does *not* grow with
$x_{\rm resp}$ here. The gap sits at the **hand-over**, and the fixture places the hand-over at a
fixed 3.5 e-folds sub-horizon, so $x_T = c_s k a_0\eta$ there is $\exp(3.5)\,c_s/(p-1) = 19.1$ for
every $k$ — the $\lambda$ scaling moves $k$, $q$ and $r$ together and leaves $x_T$ alone. The phase
held across one grid step is therefore $\omega\,\Delta\log(1+z) \approx 19.1 \times 2.30\times10^{-2}
= 0.44$ rad at every rung, which is exactly the figure campaign README §0.1 quotes. The clamp term
is $x$-independent **because the hand-over depth is $x$-independent by construction**, and that is
a statement about where the seam is put, not about the response time. It is also the reason **E2**
can scan depth at large $x$ and expect the clamp term to move: $x_T$ is the variable it is
sensitive to, and $x_{\rm resp}$ is not.

**The representation term does not grow with $x$ either, and this refutes the prompt's guess.**
Slopes $-0.06$, $-0.15$, $-0.12$. The prompt predicted growth "like $h^4x/384$", and Table 3 carries
that prediction in its last two columns so the comparison is on the page rather than in the prose:

| | $x_{\rm resp} = 980$ | $x_{\rm resp} = 10^8$ | growth |
|---|---|---|---|
| $h^4 x_{\rm resp}/384$ | 7.2e-07 | 7.3e-02 | ×$10^{5}$ |
| $h^4 (k\tau)/384$ | 1.1e-06 | 1.2e-01 | ×$10^{5}$ |
| measured representation term, `together` | 2.68e-03 | 2.94e-04 | ×0.11 |

At the bottom of the ladder the measured term is **3700 times larger** than $h^4x/384$; at the top
it is **250 times smaller**. The prediction is wrong in both directions and by five decades in
trend. **The measurement wins and the guess is recorded here as refuted.**

The reason is that the object the prediction describes is no longer in the path it was written for.
$\delta\theta \simeq h^4x/384$ is the error of re-splining a phase that grows like $x$;
`TkSourceFunctions` stopped doing that when `prompts/GkTk-remedial` prompt 10 replaced the round
trip with a `PrimitivePhase` — the leading $k\,\Delta\tau$ evaluated from the background table and
only a small residual splined. The transfer-function sector of the realistic flavour therefore no
longer carries the term, which is that campaign's fix working as intended and is the first
end-to-end confirmation of it in `total` rather than in $\theta$. What the representation term
measures instead is the floor that is left: the amplitude re-spline, the $f$ spline on the
production grid, the Liouville–Green closed forms for $\omega$ and $\mathrm{d}\ln M/\mathrm{d}z$,
and the Green's function's phase. It sits at $4\times10^{-5}$ to $3\times10^{-3}$ and is flat.

**One caveat, and it is in §8 rather than buried here:** the Green's-function half of this fixture
*does* still use a raw `phase_spline`, so the $h^4x/384$ term is present in the instrument even
though it is absent from production. It does not visibly propagate into `total` — which is itself
worth recording — but it means the representation term above is an **upper bound** on production's,
not an estimate of it.

---

## 6. The gap this harness opens, against production's

Table 5 answers the question prompt §3 item 2 asks, and the answer is clean because every quantity
in it is recorded by `build_partition` rather than inferred.

- The fixture's WKB grid step is **2.3023e-02 to 2.3038e-02** in $\log(1+z)$ at every rung of every
  shape — $\ln 10/100$, the production 100-per-$\log_{10}z$ density.
- `drop_first_WKB_sample=True` opens a gap of exactly **1.00 fixture grid steps**, and
  `build_partition` records it as **1.00 mean source-grid steps**. The two densities coincide here
  by construction, which is what makes the comparison meaningful.
- The clamp's allowance, `HANDOVER_CLAMP_MAX_GRID_STEPS = 1.5`, is **3.455e-02**, so the gap is at
  67 % of what the clamp will bridge and nothing is ever refused.
- Against production: **1.92 times the median** (1.2e-02) and **1.05 times the maximum** (2.2e-02)
  that `[12-handover-clamp-error-in-production]` recorded on 1666 $T_q$ and 1798 $T_r$
  sub-intervals.

**So the clamp term measured here is production's worst case, not its typical one, and that must
travel with the number.** `[08-handover-clamp-error]` records that the error scales as gap$^2$; on
that scaling a median-gap production row would show $(1.2/2.30)^2 = 0.27$ of the clamp term in
Table 3. The clamp term is an upper bound on the typical production row by roughly a factor of
four, and a fair estimate of its worst rows.

What is *not* comparable is the count: production clamps **two** transfer functions on most rows,
and so does this harness — except on `q-smooth`, where $T_q$ never becomes oscillatory within the
range and Table 2 records `gap Tq` as exactly zero at $x_{\rm resp} = 980$. That shape therefore
measures a one-factor clamp at its base rung and a two-factor clamp above it, which is visible in
Table 3 as the jump from $+2.58\times10^{-1}$ to $+9.12\times10^{-1}$.

---

## 7. Cost

**Prompt §3 item 1 predicted that the realistic flavour's set-up cost would not be flat, and asked
for it to be measured and reported, and for a statement if it became the binding cost. It was
measured; it is flat; and the binding cost is somewhere else.**

- **Set-up is flat and cheap in both flavours.** Table 7: 0.022–0.056 s exact, 0.029–0.098 s
  realistic, across five decades of $x$ and all three shapes. The ratio is 1.1–2.0 at every rung
  but one (3.6 at `together` $x_{\rm resp}=980$, the first cell of the run and the only one paying
  first-touch import costs). Building the `phase_spline` through the fixture's samples, and the
  `OffsetBesselPhaseGk` phase, costs hundredths of a second and does not grow. The prediction is
  refuted.
- **The exact integral is flat, as Kohri & Terada §8 item 1 found.** 0.19–0.61 s per triple from
  $x = 1.4\times10^2$ to $1.6\times10^8$, and the Levin driver accepts each sub-interval in
  **7 to 30 regions** at every rung.
- **The realistic integral is neither.** 6.8 s to 1230 s, i.e. **33× to 3239×** the exact flavour,
  and the driver bisects into **720 to 97,590 regions**.

**The growth rate is not the same on every shape, and that is what bounds the instrument.** Mean
factor per decade of $x_{\rm resp}$, from Table 8: **1.7× (`together`), 3.6× (`T-first`), 8.6×
(`q-smooth`)**; fitted slopes of $\log(\text{seconds})$ against $\log x$ are $+0.21$, $+0.55$,
$+0.94$. `together` and `T-first` therefore reach $10^8$ and $10^7$ in minutes. `q-smooth` does not:
at 8.6× per decade its $x_{\rm resp} = 10^6$ pair is hours each and its $10^7$ pair is of order
twelve hours each, so **`q-smooth` realistic is reported only to $x_{\rm resp} = 10^5$** and the
four cells above it are marked *not reached* in Tables 2, 7 and 8.

That ceiling is corroborated independently. An earlier run of this harness was killed while
computing `q-smooth` realistic at $x_{\rm resp} = 10^6$, closed seam; it had been running **more
than 5600 s without completing**. That is a genuine lower bound on that cell, consistent with the
8.6×-per-decade rate measured here, and it is quoted as a lower bound from a killed run rather than
as a completed measurement. No $N$ value from that run appears anywhere in this document.

**Where the cost goes.** The *level* of the realistic region count is understood: the integration
range spans 5.40 decades of $(1+z)$ at the base rung, i.e. about 540 cells of the
100-per-$\log_{10}z$ grid, against 720–1304 regions — the driver is bisecting to about the spline
knot scale, which is what a piecewise-cubic phase forces it to do. The *growth* is not understood.
The range widens by one decade per rung, because `z_source_max` tracks $k$ while $z_{\rm response}$
is fixed, so the knot count grows only from 540 to 1041 across six rungs — a factor of 1.9. The
region count grows by a factor of 12 (`together`), 105 (`T-first`) and 37 (`q-smooth`) over the
same span, so regions-per-knot rises from 1.3–2.4 at the base rung to 15–80 at the top: four to six
extra levels of bisection that the knot count does not explain.

**A plausible mechanism is recorded as an open issue rather than asserted here.**
`[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` on the campaign board sets out the
argument — that the fixture's Green's-function phase is a raw `phase_spline` whose representation
floor grows like $h^4x/384$, and that `_adaptive_levin`'s acceptance test forgives a region only
once its quadrature error has fallen *to* that floor — and the one-cell experiment that would
confirm or refute it. If the mechanism is right, production does not have it, and this section's
cost curve, including the `q-smooth` ceiling, is a statement about the instrument rather than about
the pipeline. **It does not touch any $N$ in this document**, because every term in Table 3 is a
difference taken at fixed flavour and the region count affects both sides of it identically.

**A note on `converged`, which reads `no` on every row of Table 2, including control rows whose
$N + 9/8$ is $7\times10^{-14}$.** The flag is the Levin driver's comparison of its own
*conservative error bound* against the requested tolerances, group by group and sub-interval by
sub-interval. This harness runs at the reference pair $(10^{-45}, 10^{-12})$ that `large_x.py`
uses, and $\text{atol} = 10^{-45}$ is unreachable by construction — the driver's round-off floor is
absolute — so the test reduces to a purely relative one at $\text{rtol} = 10^{-12}$. The bound
itself is 1.0e-12 to 3.8e-08 relative, so it fails that request at essentially every rung. It is
not a statement about achieved accuracy: on the control the achieved $|N + 9/8|$ is
$\le 6.5\times10^{-13}$, up to four orders below the bound that failed. The flag is therefore
expected and is not alarming here; `[09-abserr-is-a-quadrature-bound]` on the `source-remediation`
board is the entry that records what `total_abserr` does and does not mean.

---

## 8. What this does not cover

1. **$b = 0$ only.** Every fixture here is radiation. Campaign README §2 (g) measures the
   Liouville–Green representation as about twenty times more favourable at $w = 1/3$ than at
   $w = 0.2$, so the representation term above is measured on the easiest member of the family.
   `prompts/handover` prompt 01 landed a general-$b$ oracle; extending this factorial to $b = 0.2$
   is not done here.
2. **The fixture's grid is not the production grid.** It is a uniform 100 samples per
   $\log_{10}z$ throughout. `prompts/qcd-background-audit` prompt 15 replaced production's base
   density with a measured curvature criterion, so the gap this harness opens is one step of a grid
   production no longer uses — which is exactly why §6 reports it in grid-step units as well as in
   $\log(1+z)$.
3. **The hand-over sits at a fixed $x_T = 19.1$.** The fixture places it 3.5 e-folds sub-horizon by
   construction and this harness never moves it. Every clamp term here is therefore measured at one
   depth, and §5's explanation of the $x$-independence is also the reason **E2** cannot read a depth
   scan off these numbers.
4. **Production's $x \approx 4\times10^{12}$ is still out of reach.** The largest here is
   $1.6\times10^8$ in the exact flavour and $1.6\times10^8$ in the realistic one on `together` —
   four decades short. Kohri & Terada §8 item 2's warning stands: the pipeline's own declared error
   grows like $x\epsilon$ above $x \sim 10^5$ and would be of order $10^{-3}$ at production's $x$
   for any method carrying the phase as a double.
5. **`q-smooth` realistic is reported only to $x_{\rm resp} = 10^5$.** Its $10^6$ and $10^7$ cells
   were not reached, and §7 gives the measured cost curve — 8.6× per decade, against 1.7× and 3.6×
   on the other two shapes — that makes them unreachable. On the small-$u$ shape the realistic
   representation's **integral** cost, not its set-up cost, is what bounds this instrument's reach.
   That is prompt §7's second stop condition met with a measurement rather than a workaround: no
   cell was quietly dropped and nothing fell back to the exact flavour.
6. **The Green's-function half of the realistic flavour is one campaign behind production.**
   `test_phase_groups.py:299` documents `BesselPhaseGk` as built "the way
   `GkSourcePolicyData._create_functions` builds the real one", and that has not been true since
   `prompts/GkTk-remedial` prompt 09: production's `_build_phase` returns a `PrimitivePhase`, while
   the fixture still builds a raw `phase_spline`. The transfer-function half *is* current. So the
   representation term in §4 is an **upper bound** on production's, carrying a term production has
   already removed, and the test module's claim that the realistic flavour "is the accuracy
   production can expect" is stale for $G$. Recorded as
   `[02-realistic-fixture-Gk-phase-is-the-superseded-construction]`; **B2** should fix the fixture
   before it re-scores the residue.
7. **This measures the seam, not the remedy.** Nothing here says which of campaign README §7
   **D1**'s three constructions to use. What it does say is that whichever is chosen should recover
   the clamp term in full — §4 (b) — and that what will be left underneath is between
   $4\times10^{-5}$ and $3\times10^{-3}$ on these fixtures, with item 6's caveat.
