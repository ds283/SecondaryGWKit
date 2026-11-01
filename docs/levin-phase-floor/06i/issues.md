# 06i issues (for the board's §3 and `docs/OPEN_ISSUES.md`, applied by the orchestrator)

Per the user's ruling of 2026-11-01:
- `[06i-z3-r0-integrand-rule-has-no-declaration-to-meet]`, proposed in the stop report, is
  **dropped**. It was the prompt's error, and the orchestrator records it as a prompt amendment.
- The jitter finding is **not** a new issue. It narrows the existing one below.

## Proposed narrowing of `[05a-t-phase-abscissa-rounding-remains]`

**Add to the board entry, as a measurement by 06i** (`out/r0_jitter.txt`, `out/r0_integrand.txt`,
and the Route B readings' `jitter` blocks in `out/readings/`):

- **Measured on Z3's six cosmological-group calls (samples 1, 2, 3, 13, 14, 15)** at 06e's census
  tree, `6d81fad`. The group's `theta_mod_2pi`, about its least-squares smooth phase on each segment
  between knots, carries white jitter of σ ≈ 4.6×10⁻¹⁷ |dθ/du|. The median over 40 segments per call
  is 4.58–4.73×10⁻¹⁷, the range 3.2–6.5×10⁻¹⁷. The jitter is uncorrelated down to 16-ulp spacing in u.
  This is the rms of the rounding of z = expm1(u) times |dθ/du|.
- **It is in G's phase as well as T's.** On sample 2, the group is G−q+r with q = r, so the two T
  phases cancel bit for bit. The jitter there, 4.7×10⁻¹⁷|θ'|, is G's alone. Both factors' phases are
  `PrimitivePhase` objects evaluated at the rounded expm1(u). The issue's title and text name only T.
- **It floors any reference that samples the captured integrand.** The floor is estimated as
  sqrt(∫|f|²σ²Δx dx) at 06e's Gauss–Legendre node density (P = 128 rad, 96 nodes). Over Z3's
  counting threshold min(abserr)/100 it is:

  | sample | 1 | 2, 14 | 3 | 13 | 15 |
  |---|---|---|---|---|---|
  | floor / threshold | 12 | 2.7×10⁻⁵ | 0.20 | 161 | 0.13 |
  | 06e's observed GL scatter / threshold | 96 | 5×10⁻⁴ | 0.83 | 1,760 | 0.89 |

  The ranking is the same, and the observed scatter is 4–10× higher. So Gauss–Legendre's dropping
  of samples 1 and 13 is the integrand's own jitter, not the reference method. 06i's Route B
  integrates the smooth phase instead. Each Route B reading also reports the jittered value at the
  same nodes and its 1σ floor, so that an R3 violation can be split between Levin's error and the
  jitter (the Route B readings' `jitter`; `out/results.txt`).
- **Impact, narrowed:** beyond "an accuracy floor in T", the jitter is an undeclared,
  node-dependent term in every cosmological-group Levin call's integrand. Where the cancellation is
  deep, it is the floor of any reference built by sampling that integrand: for Z3's samples 1 and
  13, 12× and 161× the counting threshold at Gauss–Legendre's density. **Next step**, unchanged:
  a phase evaluated in u without the expm1 round trip, a producer-side change for G1 Q3's 06+.
  Possibly also a declaration of the term for cosmological groups, as 06h declares the abscissa
  term (c) for the Bessel groups. Not 06i's to decide.
