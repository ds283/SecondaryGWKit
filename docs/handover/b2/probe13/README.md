# Probe for `[13-the-tlg-term-depends-on-the-levin-subdivision]` (2026-10-27)

A laptop probe on B2's recorded cells (`../scripts/b2_cells.py`, imported, not edited), outside
the registered run. `../scripts/probe13.py` is the script as last run: the "noderiv" arm alone,
with the `QuadSourceIntegral` module fetched by `importlib.import_module`.

- `probe13_out.txt` — the first run (about 13:00–14:11 GMT): arms `base`, `phase` and `noderiv`
  at four places, same script but iterating all three arms, and with
  `import ComputeTargets.QuadSourceIntegral as QSI`. That name resolves to the *class*
  (`ComputeTargets/__init__.py` re-exports it), so setting `LEVIN_USE_THETA_DERIV` on it did
  nothing: **that run's `noderiv` lines equal its `base` lines and are void.**
- `probe13_noderiv_out.txt` — the `noderiv` arm re-run with the module handle (14:12–15:26). The two
  `together` places did not finish inside its 40-minute limit.
- `probe13_clean_out.txt` — `q-smooth` 980, all three arms, re-run in a clean detached worktree at
  `4cef62a`. Both runs above used the main checkout, where another session had uncommitted edits
  to `AdaptiveLevin/levin_quadrature.py` (`levin-phase-floor` 07) at some point during them. At
  the committed driver every printed figure is identical to the two runs above.
