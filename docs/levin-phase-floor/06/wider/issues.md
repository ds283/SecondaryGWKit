# levin-phase-floor 06 §6 — issues and observations (the re-measurement)

Proposed for the board; the laptop decides which become §3 entries.

- **W1 — `s` cannot read A3 v2 (any store written before `dc9d28c`).** `dc9d28c` declares
  `theta_residual` on `GkWKBValue`, `TkWKBValue` and `GkSourceValue`, and `datastorekit` refuses to
  open a store whose tables lack a declared column (`out/cmd/06_dev_smoke_s.txt`). This is as the
  prompt intends ("an old store is read at the SHA that wrote it"), but §6 presumed `s` could read
  A3 v2. **Resolved for this run by the user's decision, route (B):** a certified scratch copy with
  the three nullable all-NULL columns (`state/a3v2_copy/`, 408 MB, fingerprint `0e32f24a…`), kept
  until the records are committed and then deleted on the user's word. Any later re-measurement of
  a pre-06 store at a post-06 SHA meets the same wall.
- **W2 — A3 v2's diagnostic `analytic_*` WKB columns predate `1c19998`.** Re-run at the parent,
  every value reproduces `theta_div_2pi`, `theta_mod_2pi`, `G_WKB`, `T_WKB` and every other column
  except `analytic_G_w`/`analytic_Gprime_w` (stored overflowing values, now `None`) and some
  `analytic_G_rad` (104 + 99 of 241,146 on U0's two rows); one `GkSourceValue.analytic_G_rad` is
  `0.0` against a stored `-0.0`. No consumer reads them. Not acted on.
- **W3 — §6's pure-WKB tag rests on a false premise.** §6 and §0.2 say `GkSource` stores
  `z_init` as null for a pure-WKB object. A3 v2 stores `z_init == z_source` for every pure-WKB
  `GkWKBIntegration` (9,552 of 9,552) and no `GkSourceValue` has a null `z_init` (917,937 rows)
  (`out/cmd/28`). Read literally, S1 has no window. Fixed before the launch as two readings; the
  user ruled for (I), `z_init == z_source` (log, deviation 17). The same premise is in §0.2's account
  of `_phi_samples`' second term; with `z_init == z_s` the code forms `k tau.delta(z_s, z_s)` for a
  pure-WKB value rather than taking its `None` branch. Whether that is exactly zero was not checked
  here (S1's pure windows show `sigma_s <= 8.8e-14 F`). Not acted on (consumer code).
- **W4 — S2's rule is badly posed where `|theta_T|` is small.** S2 fails in 2,115 of 12,862 T
  windows, all at `z >= 1.75e7` (1,982 of them above `1e8`), where `|theta_T|` is 0.05–1,500 rad. In
  every false window `sigma_s/sigma_t5` is 0.952–1.04 and `sigma` is 5–1.5e7 `eps|theta|`
  (1.7e-12–1.7e-10 rad): it is `phi_T`'s own structure over 9 samples, identical in both trees, not
  storage noise, which there is far below it. The rule's `1e-3 sigma_t5` assumes `sigma_t5` is
  storage noise. Recorded against the rule as written; not re-posed.
- **W5 — S4 fails on 5 of 39 rows: the stored noise cost more than the declared error.**
  `|total_s - total_t5| / total_abserr_t5`: `low-G-3` 10.8, `ctrl-low-1` 3.90, `ctrl-low-2` 3.57,
  `mid-G-2` 1.62, `R2` 1.49 (relative 4.0e-6, 2.5e-2, 5.4e-5, 5.7e-7, 1.7e-5). A finding about the
  rows before 06 (§6). `ctrl-low-1` (k = 3e8, z_R 8.3) moves by 2.5 % of its total.
- **W6 — S5 fails on 3 of 31:** `mid-G-3` 1.57, `mid-G-r-3` 1.32 (both also failed 05a's T4, at
  1.70 and 1.29) and `mid-G-1` 1.31 (0.615 under t5). True on all 11 rows capped under B1 (largest
  0.823).
- **W7 — more phase-limited non-convergence under `s`.** `convergence_kinds.py` under `s`: at
  1e-7 resolution 14, bound only 32, phase-limited 23 (05a's t: 18, 32, 17); at 1e-8 20, 25, 32
  (21, 24, 27). Unconverged rows 20 at 1e-7 and 19 at 1e-8 (t5: 21 at 1e-7).
- **W8 — numerically seeded G windows.** `sigma_s` is 2.1e-4 F at the median (max 1.2e-3 F) against
  `sigma_t5` 0.27 F (max 0.78 F). It is about 8× (median) the window RMS of the re-run `deltaTheta`
  alone, so `deltaTheta` is not the whole of what remains: `phi` there is `r + k tau.delta(z_s,
  z_init)`, and `z_init` moves with the numeric stop point. 56 windows have a `deltaTheta` RMS above
  1e-3 rad (jumps that are not whole cycles). Descriptive.
- **W9 — the re-run values against A3 v2 (descriptive, `s`@1e-7, 3,960,030 G and 73,015 T
  values).** 17 G cycle changes, every one a single cycle with the new remainder within
  `4 eps|theta|` of a boundary; no T cycle change. The largest `|d theta_mod_2pi|/(eps|theta|)` is
  1.9e3 (G) and 67 (T). That ratio is dominated by samples of small `|theta|` near `z_init`, where
  the old double addition of `deltaTheta` erred by `ulp(2 pi)`, not `eps|theta|`; R14's comparison
  is the place for the absolute tabulation.
- **W10 — `Levin_z` (diagnostic) changes on one row:** `low-G+r-1`, `0x1.56a9ce9670b63p+46 ->
  0x1.4edc7c892d90fp+46`; type, quality and crossover unchanged on all 39.
- **W11 — 15 scipy `IntegrationWarning`s in `stderr.log`, as in 05 and 05a.**
- **W12 — `python -m runregistrykit store fingerprint` takes its fingerprinter from the main
  checkout's `runregistrykit.toml` root, not `PYTHONPATH`**, so a store whose schema is a later
  SHA's cannot be fingerprinted from a worktree that way (`out/cmd/13`); the tree's own
  `config.fingerprint.fingerprinter` was called directly (`scripts/fingerprint_copy.py`). Possibly
  intended behaviour of RunRegistryKit; noted for its maintainers, not worked round in the package.
