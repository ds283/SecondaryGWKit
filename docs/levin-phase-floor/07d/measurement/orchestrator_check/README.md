# 07d Part A — the orchestrator's checks at the stop

The laptop orchestrator's own readers, run over `ssh macstudio-tunnel` against Part A's records in
`var/levin-phase-floor-07d/measurement/` on 2026-11-22, read-only, for the orchestrator note's §A.3.
They are the scripts as they ran, byte for byte, and are left out of `black`.

**How each ran.** Either from the main checkout on the Mac Studio, as `ssh macstudio-tunnel
'<main checkout>/venv/bin/python -I -' < script`, or with its cwd in the worktree
`SecondaryGWKit-lpf07d-49ce964`, with `PYTHONPATH` set to the worktree and `-B` (any script that
imports `AdaptiveLevin` or `tools`).

**Outputs.** Three outputs are kept, `mycc93.out`, `mylevintc.out` and `mylevknots.out`. In them,
ssh's three-line post-quantum key-exchange warning is removed and nothing else is changed. The other
scripts' outputs were read in the session and not kept. Their results are below, and in prompt §9.1.

| script | §A.3 item | what it checks | result |
|---|---|---|---|
| `kept_check.py` | 1 | the files of §A.0 and A3 v2's scratch copy against their `KEPT_ON_MAC_STUDIO.txt` lines and certification | all equal, except two `commands.log` files appended before the marker, and R27's and R28's stores, retired earlier |
| `myreplay.py`, `ck.py` | 2 | three calls (`d:h-1e-8-1-10922\|g3`, `d:h-1e-8-0-2275\|a7`, `R:x-1e-7-3-13490-2`) replayed at 07c's code against 07c's census record (`ck.py` reads the record's layout) | equal in `float.hex` |
| `mynoise.py` | 3 | $\sigma$ by eighth-order finite differences on named leaves | 0.90 and 0.95 of the agent's on two group leaves; 0.57 on an analytic leaf, consistent with its "structure" finding |
| `mydraw.py` | 4 | the leaf strata re-drawn from their seeds; table 2's verdicts re-scored | all 44 strata in the same order; every verdict equal |
| `mytrue.py`, `mytrue2.py` | 4 | an independent leaf reference: 24-point Gauss–Legendre on the leaf split at its knots, 8 to 256 panels a piece | disagrees with v3 beyond $\epsilon_{\rm ref}$ on one-knot leaves (prompt §9.1 item 2) |
| `myclose.py` | 4 | nest3N4's 25 closest leaves against `mytrue2.py`'s reference | all bounded, with more margin |
| `myknot.py`, `myone.py` | 4 | one-knot leaves dissected by panel count | the GL value moves $4\times10^{-41}$ between 16 and 64 panels on `band_10_12\|group\|1e-7` #3 |
| `mycount.py` | 4 | understatements by knot count and by whether the reference check passed | — |
| `mylevin.py` | 5 | table 3's worst Levin understatements re-referenced | 12 of the 14 worst are reference artefacts |
| `mylevintc.py` → `mylevintc.out` | 5 | the 37 trusted, or $r<1$, band leaves `levin_tc` counts as understating, re-referenced | 14 still understate; prompt §9.1 items 3–5 account for them |
| `myredraw.py` | 6 | (Ra) and (Rt) re-drawn from their seeds | identical |
| `myscore.py` | 6 | the call readings re-scored in exact `Fraction`s | 425 referenced, 417 scored; no disagreement |
| `pickvar.py`, `myvar.py` | 7 | two calls re-run under Ea and ET20 | equal to the agent's replays |
| `myqual.py` | 7 | §2.7's qualification by an independent reader | only ET20 qualifies, vacuously; agrees with the agent |
| `mycc93.py` → `mycc93.out` | after the stop | ${\rm CC}_{93}$ as the returned value; the routes by band | prompt §9.1 item 6 |
| `mylevknots.py` → `mylevknots.out` | after the stop | $V_P$ against ${\rm CC}_{93}$ on group band leaves by knot count; knots a leaf | prompt §9.1 item 6 |
| `copyback_07d.py` | §A.4 | the copy-back list: 8,072 files copied, 29,459 replay files left out, the agent's 246 kept files re-hashed | no anomaly; every copied file checked by SHA-256 on the laptop |
