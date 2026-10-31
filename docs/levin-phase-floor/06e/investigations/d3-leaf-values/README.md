# D3: which value a trusted leaf returns

The orchestrator's check of the census agent's D3 (`../../census/issues.md`), on the laptop,
2026-10-31, at `6d81fad` (`f4d8e17`'s code). On samples 2, 8, 9 and 14 of Z3, $|v - {\rm ref}| /
{\rm abserr}$ is 0.998–0.9995 under both 06c's census code and 06e's. Samples 2 and 14 are the same
call at two rtols.

**The hypothesis.** 06c's trust check (`_check_leaf` in `AdaptiveLevin/levin_quadrature.py`)
computes a leaf's halves and quarters, sets $\Delta_1 = |V_P - H|$, $\Delta_2 = |H - Q|$ and
$r = \Delta_2/\Delta_1$, and declares $\max(\epsilon_P, \Delta_1)/(1-r)$, but keeps the leaf's own
value $V_P$ ("Its value is unchanged"). If $Q$ is much better than $V_P$, $V_P$'s true error is about
$\Delta_1$, and the ratio sits just under 1.

**The probe.** `d3_probe.py` replays a captured call (the Z3 inputs, from
`var/levin-phase-floor-06c/depth20-tiling/orchestrator_check/inputs/` on the laptop, and
`var/levin-phase-floor-06e/census/references/inputs/s03.pkl` from the Mac Studio, SHA-256 in each
output's header) with `AdaptiveLevin.levin_quadrature._adaptive_levin_subregion` wrapped to record
every solve's value. For each returned leaf it prints $V_P - H$, $H - Q$ and $V_P - Q$, signed, and
then the call's ratio with each trusted leaf's value replaced by its quarters sum. The reference is
each sample's S128(P64) (`../../census/references/out/`; sample 9's from
`docs/levin-phase-floor/06c/depth20-tiling/orchestrator_check/04-ref-s09-P64.txt`). Each replay
reproduces the census's value and abserr.

Run as `PYTHONPATH=.:var/levin-phase-floor-06e/census/scripts ./venv/bin/python -B d3_probe.py
PICKLE REF_HEX` from the repository root; the census harness is on the path because the captured
calls' pickles name its functions.

| Sample | Ratio | The leaf carrying the error | $r$ | With quarters |
|---|---|---|---|---|
| 2 | 0.998477 | $[19.43, 29.27]$, depth 1: $V_P - Q = -4.683\times10^{-35}$ of $v - {\rm ref} = -4.688\times10^{-35}$ | 0.00137 | 0.0012 |
| 8 | 0.999518 | $[-11.66, -2.95]$, depth 1 | 0.00017 | 0.00046 |
| 9 | 0.997859 | $[-11.21, -7.76]$, depth 2 | 0.00024 | 0.0015 |
| 3 | 1.245046 | $[21.78, 25.84]$, depth 2: $V_P - Q = 2.56\times10^{-34}$ of $v - {\rm ref} = 4.30\times10^{-34}$ | 0.507 | 0.50 |

**Holds on samples 2, 8 and 9.** One trusted leaf with small $r$ carries the call's error, its
returned value is $V_P$, and the quarters sum it has already computed is 650–2,200× closer to the
reference. **Sample 3 is different:** its leaf converges slowly ($r = 0.507$), and $Q$ is still
$1.7\times10^{-34}$ off, so the error falls more slowly than the extrapolation assumes
(`[06c-budget-can-stop-on-a-pre-asymptotic-residual]`). Returning $Q$ would still halve its error.

Opened as `[06e-trusted-leaf-returns-its-coarsest-value]` on the board.
