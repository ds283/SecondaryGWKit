# Issues from prompt 11 (for the orchestrator to open or drop)

No stop condition was hit, and nothing was repaired.

1. **A `minimal` Green's-function row on the shipping grid.** `GkSourcePolicyData` shard 2 serial 460:
   $k = 3.091682\times10^6$/Mpc, $z_{\rm response} = 3.963729\times10^{10}$, `mixed`, `minimal`,
   `crossover_z` $4.2472907\times10^{10}$. There is no `fail` and no `incomplete` row among the 1160.
   Narrows `[03-gk-classification-is-diagnosed-and-ignored]` (README §2 (p)): the question "does any
   `fail`/`incomplete` row exist on the shipping grid" is answered, no; what a `minimal` row costs the
   integral that uses it is not measured here.
2. **The oracle comparison's strata are tied to $z_{\rm response}$.** The rows with no clamped factor
   (2489) are the rows with one sub-interval, and occur only at $z_{\rm response}\ge10^9$. A "number of
   clamped factors" stratum is therefore also a $z_{\rm response}$ stratum, and the comparison cannot
   separate the clamp's error from the oracle's own deficit or the LG phase representation term. The
   separation needs a design where the gap is closed (a rebuild), or the fixtures of prompt 02.
   Bears on `[12-handover-clamp-error-in-production]`.
3. **`[12-handover-clamp-error-in-production]`, narrowed on A3 v2's data.** The gap in $\log(1+z)$ matches
   the 2026-09 figures: median 1.152e-02 (1.2e-02), maximum 2.259e-02 (2.2e-02); 6791 of 9280 rows
   have a clamp; no gap is within 1 % of the allowance (the maximum is 0.981 mean source-grid
   steps, against the 1.5-step allowance). The `source` spline is clamped on 423 sub-intervals (2026-09: 108),
   all in the seam band.
4. **Skipped breakpoints.** 3798 rows carry one `skipped` entry, each for `Tq`, `log_width` 0.0, and
   each with $q = r$ (equal hand-over redshifts for `Tq` and `Tr`). Nothing is wrong that this record
   shows; it is noted because the stored `partition` records them.
5. **Large deviations from the oracle in the seam band.** Maximum $|{\rm total}/{\tt analytic\_rad} - 1|$
   is 96.9 over 6080 rows (20.5 at the $10^8$ decade; 17.8 and 96.9 at $10^{10}$). The medians are
   far smaller (5.3e-04 overall). These are tabulated and not read here; they are the rows to look at if
   the clamp's effect is to be separated.
