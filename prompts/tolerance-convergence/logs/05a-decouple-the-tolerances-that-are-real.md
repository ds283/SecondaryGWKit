# Log 05a — Decouple the tolerances that are real

**Prompt:** `prompts/tolerance-convergence/05a-decouple-the-tolerances-that-are-real.md`
**Commit:** *(this prompt's own commit)* — "Give each solver-reaching target its own tolerance"
**Model:** Opus 5
**Date:** 2026-09-18
**Result:** **DONE.** The three targets whose `(atol, rtol)` reaches a solver each carry a pair of
their own, with the measurement that chose it beside the value in `config/defaults.py`, and the
fourth — `QuadSourceIntegral`, decoupled since `source-remediation` — is guarded for the first
time. Six constants ship, **every one of them settled by the user under D1 on 2026-09-17 and none
of them chosen here**: `DEFAULT_HEXIT_ABS_TOLERANCE = 1e-10`, `DEFAULT_HEXIT_REL_TOLERANCE = 1e-9`,
`DEFAULT_GK_NUMERIC_ABS_TOLERANCE = 1e-10`, `DEFAULT_GK_NUMERIC_REL_TOLERANCE = 1e-8`,
`DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` (already existed) and
`DEFAULT_TK_NUMERIC_REL_TOLERANCE = 3e-11`. Two values move — the exit time's `rtol` from `1e-8`
to `1e-9` and the $T_k$ numeric's from `1e-8` to `3e-11` — and the other four are new *names* for
values that do not. **After this commit no object type in the pipeline keys on
`DEFAULT_ABS_TOLERANCE` or `DEFAULT_REL_TOLERANCE`**, whose values are unchanged at 1e-10 and 1e-8
and whose names survive (§3 below).

**11 sites in `main.py`** moved to their target's own constant, the twelfth — `QuadSource`'s — lost
a pair that reached nothing, and the `build_missing_GkSource` conditional is preserved. **All six
`extract_*.py` readers follow**, which closes
`[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]`, live since `GkTk-remedial` prompt 12.
The `ast` guard no longer filters on `endswith("Integration")`: it **enumerates all seventeen
object types `main.py` looks up**, fails on a class it has not been told about and on a site whose
tolerance disagrees with its class, and a second guard cross-checks the six readers against
`main.py` by `config/defaults.py` constant. Five negative controls were run and all five fail the
suite (§7).

**No schema change**: no file under `Datastore/SQL/ObjectFactories/` is in the diff, and the six
factories' column lists read exactly as at `a351a50`. **The four Gauss orders are still 4** and
nothing under prompt 05b's grant moved. **The three published source-grid digests are unmoved** —
`3bef2c06`, `60a3205a`, `21ffc126` — and the reason is measured rather than assumed: the production
anchor solve is **bit-identical** at `(1e-10, 1e-8)` and `(1e-10, 1e-9)` on both production
cosmologies (§6). `ComputeTargets` **521, OK**; `CosmologyModels` **39, OK**;
`test_numeric_break_point_key.py` and `test_gauss_order_key.py` both pass **unedited**.
**9 production files** in the diff — `config/defaults.py`, `main.py`,
`CosmologyConcepts/wavenumber.py` and the six `extract_*.py` readers — plus one test module,
`ComputeTargets/tests/test_main_plumbing.py`.

Board items **T8** and **T10**. `[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]` and
part (ii) of `[02a-grid-digest-not-reproducible]` are closed;
`[02-shared-atol-doubles-as-a-float-comparison-epsilon]`,
`[02-oneloopintegral-is-a-ninth-keyed-object-type]` and
`[03a-tk-numeric-excursion-is-sporadic-in-rtol]` stay open, as the prompt requires. One issue is
opened, `[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]`.

---

## 1. Rule 9 — the provenance of every constant shipped

README §5 rule 9 makes the five §1.2 fields an acceptance condition, and prompt §2 requires them by
**citation** rather than restatement. They are in **log 03a's "State handed to the next prompt"**
(entries (a), (b), (c), (d)) for `wavenumber_exit_time` and `TkNumericIntegration`, and in
**`docs/tolerance-convergence/GK-NUMERIC-SWEEP.md` §§5.2 and 7** with **log 03's** hand-off for
`GkNumericIntegration`. Nothing below re-derives a number; what follows is the index, and the full
text of each field is in `config/defaults.py` beside the value, where the next reader will find it.

| constant | value | status | provenance (cited, not restated) |
|---|---|---|---|
| `DEFAULT_HEXIT_ABS_TOLERANCE` | 1e-10 | new name, unchanged value; **inert, unchosen and coupled** | log 03a (d); §1.2's closing rule applies |
| `DEFAULT_HEXIT_REL_TOLERANCE` | **1e-9** | new name, **changed** from 1e-8; **chosen** | log 03a (c); `TK-NUMERIC-AND-EXIT-TIME.md` §7.6 |
| `DEFAULT_GK_NUMERIC_ABS_TOLERANCE` | 1e-10 | new name, unchanged value; **inert and unchosen** | log 03; `GK-NUMERIC-SWEEP.md` |
| `DEFAULT_GK_NUMERIC_REL_TOLERANCE` | 1e-8 | new name, unchanged value; **chosen** | log 03; `GK-NUMERIC-SWEEP.md` §5.2, §7 |
| `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` | 1e-13 | **existed**; unchanged and **re-characterised** | the user 2026-09-12; log 03a (b) |
| `DEFAULT_TK_NUMERIC_REL_TOLERANCE` | **3e-11** | new name, **changed** from 1e-8; **chosen** | log 03a (a); `TK-NUMERIC-AND-EXIT-TIME.md` §4.1, §5 |

Each comment in `config/defaults.py` carries, in the standard `DEFAULT_TK_NUMERIC_ABS_TOLERANCE`
already set: the object types it keys and their per-model object count; the measurement that chose
it, **with its source-grid generation and anchor** (README §5 rule 6) and its reference's
convergence (rule 5); the floor it competes against; and the cost at the setting, with one step
either side for the one setting that moves a sector's cost.

### 1.1 The three qualifications that ship *with* the numbers

Prompt §2 requires all three in the comments, because a reader who finds only the value will
mis-read them. In my own words:

1. **`3e-11` is a measured setting and not a bound.** The $T_k$ maximum is not monotone in `rtol`:
   from `1e-9` down exactly one of the 150 runs exceeds the 3e-6 target at each of `1e-9`, `3e-10`
   and `1e-10`, and it is a *different* run each time. So `3e-11` is the loosest setting that
   cleared the floor **in prompt 03a's sweep**; it is not a setting at which the excursion has been
   shown to be impossible, and `[03a-tk-numeric-excursion-is-sporadic-in-rtol]` is open and was not
   closed by the acceptance. A future reader who sees `3e-11` and infers "errors below 3.88e-08 are
   guaranteed" has read the value without the caveat.
2. **`DEFAULT_HEXIT_ABS_TOLERANCE` is coupled, not merely inert.** Brent stops at
   `xtol + rtol*|u|`. At the largest production `|u| = 38.04` an `xtol` of 1e-10 floors the pair at
   ~1e-10 relative *however far `rtol` is tightened*, taking over below `rtol ≈ 2.6e-12`. It binds
   at 0 of 150 (k, offset) pairs at the setting shipped, so it does nothing today; what it does is
   cap what any future tightening of its partner can buy. That is said where it is defined.
3. **The exit-time pair was accepted on the *guarantee* reading.** `1e-9` is the loosest `rtol`
   whose Brent *bound* — 3.81e-08 at `|u| = 38.04` — clears
   `DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7`. The *achieved* displacement at the old `1e-8` is
   7.86e-08, which already clears 1e-7 by 1.3×, so on the achieved reading README §6.1's rule gives
   `unchanged` (log 03a, deviation 4). The user took the guarantee reading; the comment says so, so
   that the decision and not only its output is in the file.

---

## 2. The one design question (prompt §3): the shared names survive, and say what they are

**Decision: keep `DEFAULT_ABS_TOLERANCE` and `DEFAULT_REL_TOLERANCE`, at their values and under
their names, and give them a comment block saying what is left of them.** The prompt asks that
this not be assumed, so here is the argument that was actually weighed.

**What renaming would have to reach.** After this commit the two names are read in five modules,
and *four of the five are outside this prompt's grant*:

| site | what it is | in my grant? |
|---|---|---|
| `ComputeTargets/GkSource.py:96`, `:104`, `:275` | `fabs(a − b) <` float comparisons on redshifts, and a `\|G_WKB\| <` magnitude test | **no** (§6; and 05/05b own the file) |
| `Quadrature/integrators/numeric_with_phase_cut.py:618`, `:737`, `:790` | the same, on `z_init` against `z_sample.min` and on a phase difference | **no** (§6) |
| `LiouvilleGreen/WKBtools.py:83` | `theta_mod_2pi >` on a phase remainder | **no** (§6) |
| `numeric_with_phase_cut.py:536-537` | the signature defaults of `integrate_numeric_with_phase_cut`, which production always overrides from `main.py` | **no** |
| `docs/source-remediation-verification/` ×2, `docs/transfer-remedial/measure_bessel_phase.py` | verification scripts | **no** |

So renaming is not a rename: it is either an edit to three modules the prompt forbids, or a
compatibility alias, which leaves two names for one constant and is strictly worse than one. The
prompt's own §6 is explicit that
`[02-shared-atol-doubles-as-a-float-comparison-epsilon]` must stay open *and* that keeping the
value fixed is how the seven sites are protected — and a rename that left the value alone would
still touch every one of those files.

**The argument for renaming, and why it loses.** It is real: a constant called
`DEFAULT_ABS_TOLERANCE` that is no longer any target's absolute tolerance is a trap, and this is
the one commit in the campaign where the question is live. But the name is not the defect. The
defect the issue names is that *one constant is doing two unrelated jobs*, and the right repair is
to give the float-comparison sites an epsilon of their own — which is a decision about comparing
two redshifts, taken in modules that are three other campaigns' subjects, and is not made cheaper
by being taken here under a different label. Renaming without splitting would rename the trap.

**What I did instead**, which is the part that is cheap and is done: `config/defaults.py` now opens
the pair with a block saying, in terms, that they are the accuracy parameter of **none** of the
eight keyed object types; that what is left of `DEFAULT_ABS_TOLERANCE` is seven float comparisons
and a set of signature defaults, enumerated by file and line; that this is
`[02-shared-atol-doubles-as-a-float-comparison-epsilon]`, open and deliberately not closed by
moving these values; and that neither is to be retuned as though it were a solver tolerance. A
reader who reaches for them now finds the reason not to.

---

## 3. The site inventory (prompt §4)

Checked against log 05's "State handed to the next prompt" table, which prompt 05b left untouched.
Every count matches it.

| target | sites | tolerance before | tolerance now |
|---|---|---|---|
| `wavenumber_exit_time` | `build_k_exit_work` — **1** | `atol`, `rtol` | `hexit_atol`, `hexit_rtol` |
| `TkNumericIntegration` | `build_Tk_numeric_work`'s query batch, its direct `object_get`, `build_Tk_WKB_work`'s numeric lookup batch, `build_QuadSource_work`'s numeric lookup batch, `build_QuadSourceIntegral_batch`'s `Tk_numeric_lookup_batch` — **5** | `Tk_numeric_atol`, `rtol` | `Tk_numeric_atol`, `Tk_numeric_rtol` |
| `GkNumericIntegration` | `build_Gk_numeric_work`'s query batch, its direct `object_get`, `build_Gk_WKB_work`'s numeric lookup batch — **3** | `atol`, `rtol` | `Gk_numeric_atol`, `Gk_numeric_rtol` |
| `GkNumericIntegration` | `build_missing_GkSource`'s `object_read_batch` — **1, conditional** | `{"atol": atol, "rtol": rtol}` if `cls_name == "GkNumericValue"` | the same conditional, naming `Gk_numeric_atol` / `Gk_numeric_rtol` |
| `QuadSourceIntegral` | the missing-instance batch and the direct `object_get` — **2** | `quad_atol`, `quad_rtol` | unchanged |
| `QuadSource` | `build_tensor_source_work`'s query batch — **1** | `atol`, `rtol` | **removed** |

**The `QuadSource` pair was dead payload and is gone.** `ComputeTargets/QuadSource.py` names
neither `atol` nor `rtol`, its factory has no tolerance column, and its other two lookups
(`build_QuadSource_work`'s own `object_get` and `build_QuadSourceIntegral_batch`'s
`QuadSource_lookup_batch`) never carried the pair — so the one query that did was inconsistent with
the other two as well as with the table. The edit is in `main.py` alone.

**The `build_missing_GkSource` conditional is preserved exactly**, including prompt 05's comment
explaining why one payload serves two classes and the pair is added only for `GkNumericValue`. Only
the two names inside it changed.

**Two further things moved in `main.py`, both consequences rather than choices.** The `ray.get`
block that built five tolerance objects now builds eight, in one call as the prompt requires; and
`run_pipeline`'s `atol` / `rtol` **parameters were removed**, because after the switch nothing in
the function reads them. They are recorded as deviation 1.

---

## 4. The guard (prompt §4, board item **T10**)

**Before.** `test_main_plumbing.py` filtered candidate sites with
`first.value.endswith("Integration")`. That matched four class names —
`GkNumericIntegration`, `TkNumericIntegration`, `GkWKBIntegration`, `TkWKBIntegration` — of the
seventeen object types `main.py` looks up. It never saw `wavenumber_exit_time`,
`BackgroundModel`, `GkSource`, `QuadSource` or `QuadSourceIntegral`, the last because it ends in
"Integral": **the one target that was already decoupled was the one target never checked.**

**After.** `OBJECT_TYPE_TOLERANCES` enumerates every object type `main.py` looks up through
`object_get` / `object_get_vectorized` — nine compute targets and eight metadata/concept types —
each mapped either to the `(atol, rtol)` names its sites must carry or to `NO_TOLERANCE`. The
predicate on the call is now "the first argument is a string", so an unknown class reaches the
assertions instead of being filtered away. Four of the seventeen carry a pair; thirteen carry none.
`EXPECTED_SITES` gives the per-class site count for all seventeen, which is `NO_TOLERANCE_INTEGRATIONS`
and the two `EXPECTED_*_SITES` counters folded into one table — the counters exist because a finder
that silently matched nothing would pass every assertion, and that reason is unchanged. The finder
also reads `rtol` now, not only `atol`, since a decoupled `rtol` is exactly what half of this
prompt ships.

**What it fails on.** (i) A site whose class is not in the enumeration — the case that catches a
new target, which would otherwise be plumbed with no guard at all. (ii) A site whose tolerance
disagrees with its class's classification — the case that catches a missed switch, *and* a
tolerance reappearing on one of prompt 05's four. (iii) A class whose tolerance cannot be read at
all. (iv) A per-class site count that has moved.

**How `OneLoopIntegral` is represented: by deliberate absence, with a test that says so.** It is
the ninth keyed object type, with `atol_serial` / `rtol_serial` columns it genuinely filters on,
which `main.py` never builds and whose object count is 0. Adding it to the enumeration with a
classification would be answering `[02-oneloopintegral-is-a-ninth-keyed-object-type]`, which is
unassigned and the user's. So it is **out of the enumeration**, and
`test_oneloopintegral_is_not_looked_up_by_main_py` asserts both that it is absent from the
enumeration and that no site names it — meaning that if `main.py` ever looks it up, the guard fails
and the question is put rather than pre-empted.

**What it would have caught that the old one would not.** Any mis-wiring of `wavenumber_exit_time`,
`QuadSourceIntegral`, `GkSource`, `BackgroundModel` or `QuadSource` — none of which it could see —
and, on the four it could see, any mis-wiring of `rtol`, which it never read. Concretely: the
`QuadSource` dead pair removed above was invisible to the old predicate, and so was every
`extract_*.py` fault.

**One site is checked by inspection rather than structurally**, and the module says so:
`build_missing_GkSource`'s read batch goes through `object_read_batch`, whose object name is a loop
variable over `["GkNumericValue", "GkWKBValue"]` and whose pair is added by a conditional `**{...}`
expansion. The finder reads `object_get` and `object_get_vectorized` only. Extending it to a
third method whose class name is not a literal is a different piece of work and was not in the
prompt; it is recorded in §8 rather than done.

---

## 5. The readers (prompt §5): which changed, and which failure mode each was in

All six build their tolerance objects at the top of `run_pipeline` from `config/defaults.py`
constants, and **all six** query `wavenumber_exit_time` in a `create_k_exit_work` helper. Working
out which target each lookup reaches, site by site:

| script | lookups that carry a tolerance | was | now | failure mode it was in |
|---|---|---|---|---|
| `extract_Gk_data.py` | `wavenumber_exit_time`; `GkNumericIntegration` (`query_payload` at `:405`) | shared pair for both | `hexit_*`; `Gk_numeric_*` | exit time: **silent reuse**. `GkNumericIntegration`: **neither** — its value does not move, so this is a rename with no behavioural change |
| `extract_GkWKB_data.py` | `wavenumber_exit_time`; `GkNumericIntegration` (`:451`) | shared pair for both | `hexit_*`; `Gk_numeric_*` | as above |
| `extract_TkWKB_data.py` | `wavenumber_exit_time`; `TkNumericIntegration` (`:444`) | shared pair for both | `hexit_*`; `Tk_numeric_*` | `TkNumericIntegration`: **miss — and it was already missing before this commit**, which is `[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]`. Exit time: silent reuse |
| `extract_GkSource_data.py` | `wavenumber_exit_time` only | shared pair | `hexit_*` | silent reuse |
| `extract_tensor_source_data.py` | `wavenumber_exit_time` only | shared pair | `hexit_*` | silent reuse |
| `extract_QuadSourceIntegral_data.py` | `wavenumber_exit_time`; `QuadSourceIntegral` (`:1091`, already `quad_*`) | shared pair for the first | `hexit_*`; `quad_*` unchanged | silent reuse |

**The prompt's instruction to check the other five for the same fault was carried out and found one
more thing rather than none.** No reader other than `extract_TkWKB_data.py` queried
`TkNumericIntegration` at all, so the split constant's absence hurt only that one — but the
`GkNumericIntegration` lookups in `extract_Gk_data.py` and `extract_GkWKB_data.py` were in the same
*shape*: a shared constant standing in for a target-specific one, harmless only because the two
happened to have the same value. Both now name their own constant, so the coincidence is no longer
load-bearing. `extract_GkSource_data.py` and `extract_tensor_source_data.py` look up
`BackgroundModel` and `GkSource` with no tolerance at all, which is right since prompt 05.

**The two failure modes, and why the distinction matters.** For the three equality-keyed targets a
stale reader **misses**: `atol_serial ==` does not match, nothing is found, and the script says so.
For `wavenumber_exit_time` the key is an **inequality**
(`[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`) — the lookup takes the loosest stored
row at least as tight as the request — so a reader still asking for `1e-8` does **not** miss: it is
served the new `1e-9` row and then reports the *stored* pair, not the requested one. Tightening the
production value therefore makes every stale exit-time reader quietly succeed with different
provenance, which is why §8 opens an issue for the two verification scripts outside this prompt's
file list rather than leaving them unremarked.

**§7's cross-check was buildable without a datastore and is built.** `ReaderToleranceAgreementTestCase`
resolves, for `main.py` and for each reader, the `config/defaults.py` constant behind every
tolerance a lookup passes — through the `ray.get([pool.object_get("tolerance", tol=CONSTANT), …])`
tuple each module builds, and through `**query_payload` dict literals — and requires the reader's
constant for a class to be the one `main.py` uses for that class. It compares **constants, not
local variable names**, because the scripts may call their locals what they like and the datastore
key is the value. It would have caught
`[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]` the day it landed.

---

## 6. The digest check (prompt §9, stop condition 4)

**Done before anything else was concluded, as the prompt requires, and nothing moved.**

`z_init` is the output of `_solve_horizon_exit(cosmology, k = 3e8/Mpc, offset = −5)`, so a change
to the exit-time pair is exactly the sensitivity `[02a-grid-digest-not-reproducible]` describes.
Measured directly, old pair against new, on both production cosmologies:

| cosmology | `z_init` at `(1e-10, 1e-8)` | `z_init` at `(1e-10, 1e-9)` | relative move |
|---|---|---|---|
| `LambdaCDM(Planck2018)` | `2.0636395964154036e+16` | `2.0636395964154036e+16` | **0.000e+00** |
| `QCD_Cosmology` | `3.30033444460513e+16` | `3.30033444460513e+16` | **0.000e+00** |

**Bit-identical.** Brent converged to the same double at both settings, so there is no displacement
to propagate and the question of what it would have done to the grid does not arise. The version-2
grids built at those anchors, and at the two published anchors, confirm it:

| grid | samples | digest | published |
|---|---|---|---|
| `RadiationModel` at its own anchor `44523947729.77296` | 2,306 | `3bef2c06` | `3bef2c06` ✓ |
| `LambdaCDMModel` at `PRODUCTION_Z_INIT_LAMBDACDM` | 1,778 | `60a3205a` | `60a3205a` ✓ |
| `QCDModel` at `PRODUCTION_Z_INIT_QCD` | 2,034 | `21ffc126` | `21ffc126` ✓ |

The radiation anchor is built by `wkb_reference.horizon_exit_z`, a test-tree solve at
`xtol = rtol = 1e-14` that this prompt does not touch, so that row is a control rather than a
result.

**One thing this measurement says that the campaign record did not**, and it is recorded rather
than acted on: the *live* LambdaCDM anchor solve returns `2.0636395964154036e+16` while
`wkb_reference.PRODUCTION_Z_INIT_LAMBDACDM` is `2.0636395964161516e+16`, 3.6e-13 apart. That is
`[02a-grid-digest-not-reproducible]`'s own figure, confirmed again here, and it is why the
published digests are pinned to a hard-coded constant rather than to a re-solve. It is unchanged by
this commit and is not caused by it.

**Part (ii) of `[02a-grid-digest-not-reproducible]` closes on the ordering, and the log says
exactly what closes.** The issue's mechanism is that the anchor was pinned only to
`xtol + rtol*|u| = 3.8e-7` relative — *coarser* than the `1e-7` at which
`Datastore/SQL/ObjectFactories/redshift.py` matches an existing row — so the tag could turn over
while every redshift row was re-matched and reused. At `rtol = 1e-9` the guarantee is **3.81e-08**,
which is **inside** the row-match tolerance: the anchor is now pinned finer than the precision at
which its consequences are compared, so one design tolerance now governs both. What is **not**
claimed, and must not be read into this: the digest itself is still bit-exact and still turns over
under a 1e-14 relative perturbation of the anchor, which is
`[03a-qcd-v2-grid-sample-count-is-not-reproducible]` and **stays open**.

---

## 7. Verification performed

1. **`ComputeTargets` suite: 521 tests, OK, 182.2 s** — against 516 at the parent commit, which is
   the acceptance floor. The arithmetic: the old `TkNumericToleranceWiringTestCase` (6 tests) is
   replaced by `ObjectGetToleranceWiringTestCase` (7) and `ReaderToleranceAgreementTestCase` (4).
   `516 − 6 + 11 = 521`. The wall-clock flake of board note 16,
   `test_tk_wkb_phase.TestCost.test_wall_time_per_object`, did not fire.
2. **`CosmologyModels` suite: 39 tests, OK.**
3. **`ComputeTargets/tests/test_numeric_break_point_key.py` and `test_gauss_order_key.py` pass
   unedited** — neither is in the diff, and both are inside the 521.
4. **Five negative controls**, each applied to a pristine copy of the tree, run, and reverted. All
   five fail; the guard is not vacuous:

   | control | result |
   |---|---|
   | one `TkNumericIntegration` site left on `Gk_numeric_rtol` | **FAILED (3)** |
   | a tolerance pair added back to a `GkWKBIntegration` lookup | **FAILED (3)** |
   | one `object_get` renamed to an unenumerated class (`OneLoopIntegral`) | **FAILED (2)** |
   | `extract_TkWKB_data.py` put back on the exit-time pair for `TkNumericIntegration` | **FAILED (1)** |
   | the `QuadSource` dead pair restored | **FAILED (2)** |

   Two earlier attempts at controls 2 and 3 passed and were **discarded as bad controls, not as
   evidence**: the first inserted a duplicate `atol` key into a dict that already had the same
   name, the second matched no text in `main.py`. Both are recorded here because a negative control
   that passes is worth explaining rather than deleting.
5. **The digest check of §6**, run before any conclusion was drawn about the exit-time pair.
6. **`black --check` clean on all ten `.py` files in the diff** (nine production, one test). (`black --check .` reports 54
   pre-existing offenders under `docs/gk-wkb-review-fable-2026-09-09/` and
   `docs/adaptive-levin-benchmark/`; none is in this diff and none was introduced here.)
7. **No schema change, verified by construction**: `git diff --name-only` contains no path under
   `Datastore/SQL/ObjectFactories/`, and no `ComputeTargets/` compute class is in the diff.
8. **The four Gauss orders are still 4** and nothing under prompt 05b's grant
   (`BackgroundModel.py`, `GkWKBIntegration.py`, `TkWKBIntegration.py`, `GkSource.py`,
   `phase_residual.py`, `WKB_phase_function.py`) is in the diff.
9. **The seven float-comparison sites are unchanged and still read `DEFAULT_ABS_TOLERANCE = 1e-10`**
   (prompt §6): `GkSource.py:96`, `:104`, `:275`, `numeric_with_phase_cut.py:618`, `:737`, `:790`,
   `WKBtools.py:83`. None is in the diff and the value did not move.

---

## 8. Deviations from the prompt

1. **`run_pipeline`'s `atol` / `rtol` parameters were removed** — *IMPLEMENTATION CHOICE.* After
   the switch nothing in the function reads them: the eleven sites all name a target-specific
   object, and those objects are module-level names built in `main.py`'s `with` block, which is how
   `Tk_numeric_atol`, `quad_atol` and `quad_rtol` already reached `run_pipeline` before this
   commit. Leaving two unused parameters called `atol` and `rtol` in the signature of the function
   that holds every lookup in the pipeline is an invitation to the exact failure this prompt
   exists to close, so they were dropped along with the two arguments at the single call site. The
   alternative — promoting the six new objects to parameters — would have been a larger edit that
   broke with the convention the file already follows for three of the eight.
2. **`config/defaults.py` gained a comment block on `DEFAULT_ABS_TOLERANCE` /
   `DEFAULT_REL_TOLERANCE`** — *IMPLEMENTATION CHOICE.* The prompt's file list says "the constants
   of §2, **and nothing else in it**", and these two are not §2 constants. No value moved and no
   name changed; what was added is the §3 decision's substance, which would otherwise exist only in
   this log. §3 asks that the choice be implemented as well as justified, and for "leave them" the
   implementation *is* the note that says why they are still there and what they are now for.
3. **`CosmologyConcepts/wavenumber.py` gained a comment above the `root_scalar` call** — *IMPLEMENTATION
   CHOICE.* The prompt grants "`_solve_horizon_exit`'s tolerance defaults and the paths that reach
   them". Changing the defaults without recording that `xtol` binds nowhere and that `rtol*|u|` is
   what pins the root leaves the coupling discoverable only from `config/defaults.py`, two modules
   away from the `brentq` call it is about. No algorithm, bracketing or `DEFAULT_HEXIT_TOLERANCE`
   change.
4. **The guard's finder now reads `rtol` as well as `atol`** — *STRUCTURALLY REQUIRED.* Prompt §4
   asks it to fail on "a site whose tolerance disagrees with its class's classification", and half
   of what this prompt ships is a decoupled `rtol`: a finder that read only `atol` would have
   passed a site carrying `Tk_numeric_atol` with the wrong `rtol`, which is precisely negative
   control 1.
5. **`EXPECTED_TK_NUMERIC_SITES` and `EXPECTED_NO_TOLERANCE_SITES` became one `EXPECTED_SITES`
   table over all seventeen classes** — *STRUCTURALLY REQUIRED*, and explicitly invited by prompt
   §4 ("fold … into whatever shape that takes"). The reason the counters exist is unchanged and is
   restated in the module.
6. **The old test class name `TkNumericToleranceWiringTestCase` is gone**, replaced by
   `ObjectGetToleranceWiringTestCase` — *IMPLEMENTATION CHOICE.* The class no longer guards one
   constant on one target and a name that says it does would mislead. Nothing outside the module
   imports it (checked).

**No `UNINTENDED DRIFT`.**

---

## 9. Observations not acted on

1. **Two verification scripts outside the file list still query `wavenumber_exit_time` under the
   shared pair.** `docs/source-remediation-verification/analyse_greens_and_source.py:447-465` and
   `run_quadsource_integrals.py:140-157` build `atol`/`rtol` from `DEFAULT_ABS_TOLERANCE` /
   `DEFAULT_REL_TOLERANCE` and use them for a `wavenumber_exit_time` lookup — and, in the first
   case, for `GkNumericIntegration` as well. The `GkNumericIntegration` half is harmless: the value
   did not move. The exit-time half is the **silent reuse** mode of §5 — they will be served the
   new `rtol = 1e-9` row and report a pair they did not ask for. Neither file is in this prompt's
   grant. Opened as
   `[05a-two-verification-scripts-still-query-the-exit-time-under-the-shared-pair]`.
2. **`ComputeTargets/QuadSourceIntegral.py:1550` still says the pipeline supplies
   `DEFAULT_QUADRATURE_ATOL = 1e-25`.** It has been 1e-32 since `source-remediation` prompt 12.
   Recorded by `RECONCILIATION.md` §2.9 and not fixed there either; the file is out of bounds under
   README §0.4.
3. **`build_missing_GkSource`'s read batch is the one tolerance site no structural test can see.**
   §4 explains why. A finder extended to `object_read_batch` would have to resolve a loop variable
   over a literal list and a conditional dict expansion; it is buildable but it is a different
   piece of work from the one asked for, and inventing it here would have put an untested
   `ast` walker in the commit that ships the campaign's parameters.
4. **`main.py`'s Bessel `phase_atol=1e-12` / `amplitude_rtol=1e-12` (`:1197`, `:1205`) are still
   hard-coded literals** and are not in any key. They are on prompt 06's provenance checklist
   (README §1.2) as parameters nobody has ever chosen; this prompt neither moved nor named them.

---

## 10. State handed to the next prompt

**Prompt 06 is next** and assembles `docs/TOLERANCE-PROVENANCE.md` from the campaign's logs
(README §1.2). What this commit settles, and what it leaves:

**(a) Decoupled and measured — four targets, eight constants.** Each has the five §1.2 fields in
`config/defaults.py` at the point of use and in the logs cited in §1 above.

| target | pair | where the measurement is |
|---|---|---|
| `wavenumber_exit_time` | `DEFAULT_HEXIT_ABS_TOLERANCE = 1e-10` (inert, unchosen, **coupled**), `DEFAULT_HEXIT_REL_TOLERANCE = 1e-9` (chosen, on the **guarantee** reading) | log 03a (c), (d); `TK-NUMERIC-AND-EXIT-TIME.md` §7.6 |
| `GkNumericIntegration` | `DEFAULT_GK_NUMERIC_ABS_TOLERANCE = 1e-10` (inert, unchosen), `DEFAULT_GK_NUMERIC_REL_TOLERANCE = 1e-8` (chosen) | log 03; `GK-NUMERIC-SWEEP.md` §5.2, §7 |
| `TkNumericIntegration` | `DEFAULT_TK_NUMERIC_ABS_TOLERANCE = 1e-13` (step-selection knob, re-characterised), `DEFAULT_TK_NUMERIC_REL_TOLERANCE = 3e-11` (chosen, **not a bound**) | the user 2026-09-12; log 03a (a), (b) |
| `QuadSourceIntegral` | `DEFAULT_QUADRATURE_ATOL = 1e-32` (chosen), `DEFAULT_QUADRATURE_RTOL = 1e-8` (confirmed non-binding) | `source-remediation` log 12; prompt 06 re-measures |

**(b) Order-governed, not tolerance-governed — four targets, five orders.** `BackgroundModel`
($N_\tau$, $N_{c_s\tau}$, $N_F$), `GkWKBIntegration` and `TkWKBIntegration` ($N_\rho$), all 4 and
all now in the lookup key (prompt 05) and reported as built (prompt 05b); `GkSource` has no
accuracy parameter at all. Provenance in log 04's hand-off and `ORDER-AUDIT.md`.

**(c) Inherited and unmeasured, or measured elsewhere.** `RESIDUAL_WKB_REGION_MARGIN = 0.5`
(`unchanged` under §6.1 rule 6, `ORDER-AUDIT.md` §7.2); the three
`LambdaCDM_GenericEOS.py` / `T_z_reference.py` root solves, settled by
`background-solver-robustness` and lifted by prompt 02 rather than re-derived;
`SOURCE_GRID_MAX_GUARDED_FRACTION = 0.05` (prompt 02a, measured at 64.8× the worst band).

**(d) Still unexplained — README §1.2's closing list, scored.** These are what prompt 06 must say
"the provenance cannot be established from the record" about, in those words:

- `DEFAULT_ABS_TOLERANCE = 1e-10` and `DEFAULT_REL_TOLERANCE = 1e-8` — **no longer any target's
  tolerance**, but still seven float comparisons' epsilon and a set of signature defaults.
  `[02-shared-atol-doubles-as-a-float-comparison-epsilon]`, open.
- `find_phase_extremum`'s `xtol=1e-6, rtol=1e-4` at `LiouvilleGreen/integration_tools.py:95` — the
  numeric→WKB hand-over's, `[11-stop-point-root-tolerance]`, out of this campaign.
- `main.py`'s Bessel `phase_atol=1e-12`, `amplitude_rtol=1e-12` (`:1197`, `:1205`).
- `DEFAULT_LEVIN_THRESHOLD = 1.0`.
- `DEFAULT_HEXIT_TOLERANCE = 1e-2` — a **bracket width**, not a tolerance, despite the name; the
  post-hoc guard on `|q_root|`.
- The two *inert and unchosen* halves shipped here, `DEFAULT_HEXIT_ABS_TOLERANCE` and
  `DEFAULT_GK_NUMERIC_ABS_TOLERANCE`: both are `DEFAULT_ABS_TOLERANCE`'s 1e-10 inherited, and the
  record does not say who chose 1e-10 or for what. Their notes already say so; prompt 06 should not
  invent a justification for them.
- `OneLoopIntegral`'s `atol_serial` / `rtol_serial`, which key a target that computes nothing —
  `[02-oneloopintegral-is-a-ninth-keyed-object-type]`, unassigned and the user's.

**(e) The two caveats that must survive into the provenance note**, because a note that records
`3e-11` or `1e-9` without the reading it rests on has not recorded the decision: the $T_k$ setting
is the loosest that cleared *in prompt 03a's sweep* and not a bound
(`[03a-tk-numeric-excursion-is-sporadic-in-rtol]`, open), and the exit-time setting applies README
§6.1's rule to Brent's *guarantee* rather than to the achieved displacement, on which reading the
answer would have been `unchanged`.

**(f) The datastore.** Two values moved, so every existing `TkNumericIntegration` row and every
`wavenumber_exit_time` row is now keyed differently — the first by an equality that will miss, the
second by an inequality that will miss *upward* (a request for 1e-9 is not satisfied by a stored
1e-8). That is D2, settled 2026-09-12, and the intended outcome rather than a cost to be minimised.
`GkNumericIntegration` and `QuadSourceIntegral` rows are untouched, their values being unchanged.
