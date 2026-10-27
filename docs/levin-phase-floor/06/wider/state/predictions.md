# levin-phase-floor 06 §6 — predictions, fixed before the run

Written on the Mac Studio after gate U0 held (`out/U0_a.txt`, `out/U0.txt`, 22:32:07Z) and before
the launch. The block below is prompt 06 §6's "Predictions, fixed before the run" and "Descriptive,
not scored", **quoted unchanged** from `prompts/levin-phase-floor/06-producers-store-the-residual.md`
at Part B's dispatch SHA `f5cee4c` (lines 379–409; `state/prompt_section6_predictions.raw`). That
text is the prompt as amended: §9 point 4 amended S1 in place before dispatch (quoted after it,
lines 460–464; `state/prompt_section9_point4.raw`). Nothing in either quotation is re-posed.

---

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
- **S1 — $\varphi_G$'s noise goes.**
  - **The rule:** in every **pure-WKB** probe window of $G$ on R1–R7,
    $\sigma_s \le \max(10^{-3}\,\sigma_{t_5},\ 10^{-6}F)$.
  - **Holds** if true in every pure-WKB window. **Undecided** if there is none.
  - Numerically seeded windows are not scored: their $\sigma_s$ is reported beside the window RMS
    of $\Delta\theta$ alone (the probe, above).
  - 04 measured $\sigma$ at 0 at the median, quantised, and 0.46–0.78 $F$ at the maximum, for
    production's $\varphi_G$.
- **S2 — $\varphi_T$'s too.** S1's rule for every probe window of a $T$ factor. $T$ has one
  object per $k$, so its $\Delta\theta$ is a constant and every $T$ window is scored.
- **S3 — the cap does not return.** On every row (all 39), under $s$ at `1e-7`, no group has a
  region at depth ≥ 18. An over-budget unit counts as false.
- **S4 — the stored noise was within the declared error.**
  - **The rule:** on every row (all 39), $|{\rm total}_s - {\rm total}_{t_5}| \le$
    `total_abserr`$_{t_5}$, at `1e-7`.
  - A failure measures what the stored noise cost a total beyond what it declared. That is a
    finding about the rows before this prompt, not a fault in it.
- **S5 — the declared error is honest after the change.**
  - **The rule:** on every one of the 31 depth-20 rows with both $s$ units in budget,
    $|{\rm total}_s(10^{-7}) - {\rm total}_s(10^{-8})| \le$ `total_abserr`$_s(10^{-7})$.
  - 05a's T4, the same rule under $t_5$, held on 29 of 31.

**Descriptive, not scored:**
- $|{\rm total}_s - {\rm total}_{t_5}|$, relative and against `total_abserr`$_{t_5}$, on every row;
- regions and maximum depth per group, $t_5$ against $s$;
- `total_converged` under both trees;
- `theta_deriv`'s change at the Levin windows: $\varphi'$ was the term 04 and
  `phase-representation` 02 found worse than omitting $\varphi$;
- `GkSourcePolicyData`'s `Levin_z` under both trees;
- `convergence_kinds.py` under $s$, as 05a ran it.

---

§9 point 4, as amended before dispatch (for the record; S1 above already carries it):

4. **§6, S1.** A numerically seeded `GkWKBIntegration` takes its $\Delta\theta$ from the
   `GkNumericIntegration` at `rtol` $10^{-8}$, a separate integration for each source redshift. Its
   error varies from sample to sample and is far above $10^{-6}F$, and storing $r$ exactly does not
   remove it. S1 is scored on pure-WKB windows only; the others are reported beside the noise of
   $\Delta\theta$ alone. S2 is unchanged.

---

## Readings fixed before the run

These say how the rules above are computed. They do not change any rule.

- **ε** is `2**-52`. **F** is 04's: `ε · k · η(z_response)`, with `k` the row's `k` and `η` from
  `docs/handover/a3-v2/findings/depth20_floor.py` (`eta`), as `docs/levin-phase-floor/04/scripts/check_phases.py`
  formed it.
- **σ (a window's RMS)** is 04's `window_rms`, copied unchanged into `scripts/phase_probe06.py`: the
  RMS residual about a degree-4 fit in `t ∈ [-1, 1]` over 9 consecutive samples of `φ`, in
  `u = log1p(z)`. The samples are the factor's `PrimitivePhase.phi_samples` at its `z_samples`, the
  phase object the unit's own Levin calls used (G: `GkSourcePolicyData.functions.phase`; T:
  `TkSourceFunctions.phase`).
- **A probe window** of a Levin sub-interval is a window whose centre (its 5th sample) lies in that
  sub-interval's `[log(1 + z_min), log(1 + z_max)]`, for each factor oscillatory there (G if
  `regime[0]`, T_q if `regime[1]`, T_r if `regime[2]`). R1–R7 are 04's rows, which `sample.json`
  names `R1`…`R7`.
- **Matching across trees.** A window is scored when the same (row, sub-interval index, factor,
  start index) exists under both trees with its 9 `z` samples bit-identical. A window found in one
  tree only, or with differing `z`, is reported and not scored; if any such window exists and no
  scored window is false, S1 (or S2) is **undecided**.
- **Pure-WKB tag** (S1): all 9 of a G window's samples have `z_init` None in the `GkSource` value
  they come from, under **both** trees (t5 from A3 v2, s from the re-assembled source). A window
  whose tag differs between the trees is reported and not scored, with the same undecided rule.
- **S1/S2 inequality**: `σ_s <= max(1e-3 * σ_t5, 1e-6 * F)`, evaluated in double, per window, with
  the row's F. S2 covers T_q and T_r windows alike (both are counted when `q == r`).
- **S3**: the maximum region depth of every group call of each `s`@1e-7 unit, from its pickled
  region arrays (equal to the checkpoint's `max_depth`); a unit not `done` (over budget or error)
  counts as false on its row.
- **S4**: `|total_s - total_t5| <= total_abserr_t5` in IEEE double as written (an infinite
  `total_abserr_t5` satisfies it unless the difference is NaN). A row whose `s`@1e-7 unit is not
  `done` makes S4 **undecided** on that row (the t5 unit has no budget).
- **S5**: over the 31 rows whose `role` is `d20` (the reading 05a used); a row whose `s`@1e-7 or
  `s`@1e-8 unit is not `done` is excluded and listed. The 11 rows capped under B1 are reported apart,
  as in 05 and 05a.
- **Numerically seeded G windows** (not scored): reported with `σ_s` beside the window RMS of the
  re-run producers' `Δθ` for the same 9 samples, on the same fit, after `numpy.unwrap`
  (`dtheta_rms_unwrapped`), and without it (`dtheta_rms_raw`).
- **The trees**: t5 reads A3 v2 itself; s reads, read-only, the certified scratch copy of route (B)
  (`state/a3v2_copy/`, fingerprint `0e32f24a…`, every row identical, three added all-NULL columns;
  `out/cmd/10`–`16`), and re-runs every producer it uses.

## Added before the launch: the pure-WKB tag (S1), read two ways

Found after U0 and the predictions above, in the driver smoke test (`out/cmd/27`, rows `R7` and
`ctrl-NULL-1`), before the launch. The smoke's output that was looked at: R7's probe window counts
per factor and tag, and the range of the G windows' RMS over all of them (t5 0 to 1.4e-6, s 5e-23
to 2.1e-9); no window was scored and no S1 number exists.

Prompt §6 tags a G window **pure-WKB** "if every one of its 9 source samples comes from a pure-WKB
object (`GkSource`'s `z_init` null)", and §0.2 says `GkSource` "stores `z_init` as null there".
**A3 v2 does not**: every pure-WKB `GkWKBIntegration` (no numeric parent, `numeric_serial` NULL)
stores `z_init` equal to its `z_source`, bit for bit (9,552 of 9,552 over the four shards), every
seeded one a `z_init` below its `z_source`, and no `GkSourceValue` with WKB data has a null
`z_init` (917,937 rows) (`out/cmd/28_pure_wkb_zinit.txt`). The re-run carries the stored `z_init`,
so the same holds under `s`. Read literally, then, no window is pure-WKB, and S1 is undecided by
its own clause ("Undecided if there is none").

Both readings are computed from the saved samples (each window's 9 `z` and 9 `z_init`, under both
trees; under `s` also each sample's re-run object's parent):
- **(I), scored:** a sample comes from a pure-WKB object iff its value's `z_init` equals its source
  redshift exactly (`z_init == z_source`; under `s` also its re-run object has no numeric parent,
  which must agree). A window is pure-WKB iff all 9 samples are, under both trees.
- **(L), reported beside it, unscored:** the parenthetical as written (`z_init` null).

Which reading is S1's verdict of record is put to the user in the launch report; (I) is scored
until they rule. Nothing above this section is changed; the file as first written is kept as
`state/predictions_as_first_written.md`.
