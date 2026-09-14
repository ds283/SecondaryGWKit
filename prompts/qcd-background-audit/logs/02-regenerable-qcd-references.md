# Log 02 — Make the QCD reference fixture regenerable, and map what depends on it

**Prompt:** prompts/qcd-background-audit/02-regenerable-qcd-references.md
**Commit:** *(this commit)* — Make the QCD reference fixture regenerable before the background moves
**Model:** Claude Sonnet 5
**Date:** 2026-09-14
**Result:** COMPLETE WITH DEVIATIONS

---

## What shipped

**No production file was touched, no existing test was touched, and
`ComputeTargets/tests/wkb_reference_data.json` is unchanged byte for byte** (`git status --porcelain`
on it is empty both before and after this prompt's work). `T_Z_REPRESENTATION_VERSION` still does
not exist at this commit (introduced by prompt 03); the regenerator handles that gracefully (see
below) rather than assuming it.

### `docs/qcd-background-audit/generate_qcd_references.py` (new, ~330 lines)

The QCD-only reduction of `docs/gktk-remedial/generate_references.py`, per README §2 (e) and this
prompt's own charter. It:

- imports `docs/gktk-remedial/reference_lib.py` (generic Gauss/quad support, not campaign-specific
  state) exactly as `generate_references.py` itself does, via `sys.path.insert`;
- reproduces `generate_references.py`'s `build_adaptive_model` for the QCD block only, at the same
  tolerances (`QUAD_PRIMARY_EPSREL = 1.5e-14`, `QUAD_CROSSCHECK_EPSREL = 1e-12`,
  `GAUSS_CROSSCHECK_RTOL = 1e-14`, `GAUSS_CROSSCHECK_MAX_LEVEL = 8`) and on the **same grid** —
  `QCDModel` is built on `LambdaCDMModel`'s horizon-exit `z_init`, not a QCD-specific one, matching
  `generate_references.py`'s `main()` exactly (this detail is what makes bit-for-bit reproduction
  possible; getting it wrong would have produced a different, merely close, grid);
- defines `SCIENCE_KEYS` — `tau_minus_top`, `cs_tau_minus_top`, `friction_F_minus_top`, `rho_G`,
  `rho_T`, `rho_anchor_z`, `primitives_at_rho_anchor`, `short_baseline`, `reference_floor`, `grid`,
  `z_top`, `checkpoints` — the fields compared for "did the representation move"; `method` and the
  two timing fields (`build_time_seconds`, `generation_time_seconds`) are excluded because they
  differ on every run regardless of whether the representation moved;
- `science_diff(existing_block, new_block)` compares each `SCIENCE_KEYS` entry via a JSON
  round-trip (so `np.float64` vs `float` and tuple vs list do not register as spurious changes) and
  reports which keys differ;
- on **no change** (this run): writes nothing at all — not even to refresh `generated` or `method`
  — so the acceptance test (`git status --porcelain` empty) holds trivially;
- on a **change** (what prompts 04–07 will see): rewrites `payload["models"]["QCDModel"]` in
  place, preserving every other top-level key (`RadiationModel`, `LambdaCDMModel`, `convergence`,
  `baselines`, …) untouched; sets the top-level `generated` date; and **appends** (never replaces)
  a provenance line to the block's own `method` string naming the script, the date, and
  `T_Z_REPRESENTATION_VERSION` (or, before prompt 03, a note that the constant does not exist yet);
- `--dry-run` performs the same recomputation and comparison, prints the diff, and exits `1` if
  anything would change, `0` if not — writing nothing in either case.

### `docs/qcd-background-audit/REFERENCE-FIXTURE.md` (new, ~230 lines)

The map. Verified the grep the prompt names
(`grep -rln --include="*.py" "QCD\|qcd" ComputeTargets/tests/ CosmologyModels/tests/ LiouvilleGreen/tests/`)
returns exactly the eleven modules the prompt lists (plus `wkb_reference.py`, the harness, and
prompt 01's two new `CosmologyModels/tests/` modules; `LiouvilleGreen/tests/` has zero hits).
Read every one of those eleven files in full (`test_background_derivatives.py` turned out to have
zero QCD-dependent tests despite one docstring mention — verified by reading it, not assumed) and
classified **45 test methods** that depend on a QCD background number, each against: what it
asserts, its tolerance, whether it is scored against the JSON `models.QCDModel` block, the JSON's
separate `convergence` block, a closed form, or another part of the tree (self-consistency), and
which of {`T(z)` node value, representation shape, break-point set, none} would move it. §5 tallies
the four "moved by" categories (a test can appear in more than one) and calls out, by name, the two
assertions that will need **editing** rather than mere re-measurement once prompt 07 lands.

## Deviations from the prompt

### 1. Imported `docs/gktk-remedial/reference_lib.py` rather than copying it — IMPLEMENTATION CHOICE

The prompt says to take `generate_references.py` "as the starting point and reduce it," and to
write a fresh script rather than edit the original if it cannot be reduced in place. I reduced
`build_adaptive_model` into a new, QCD-only function (`build_qcd_block`), but for the ~340 lines
of generic Gauss-Legendre/adaptive-quadrature support in `reference_lib.py` (which carries no
QCD-audit-specific logic — checkpoint selection, panel bisection, the double-precision integrands)
I imported the module directly, exactly as `generate_references.py` itself does. Alternatives
considered: (a) copy the ~340 lines into the new script, which risks the two copies drifting
silently if `GkTk-remedial` is ever revisited; (b) import it, accepting a dependency on a file
outside this campaign's directory. I chose (b): the prompt's "do not touch" list names
`generate_references.py` specifically (the file whose *history* the `GkTk-remedial` board depends
on), not `reference_lib.py`; nothing here modifies either file; and `reference_lib.py` is already a
shared dependency of the campaign's own new module (`docs/qcd-background-audit/generate_qcd_references.py`
now sits alongside `docs/gktk-remedial/*.py` the same way `docs/gk-wkb-review-fable-2026-09-09/common.py`
already is shared across scripts, which is the precedent `reference_lib.py`'s own docstring names).

### 2. "Science content" excludes `method` and the two timing fields — IMPLEMENTATION CHOICE

The prompt's acceptance test is that `--dry-run` "must report no change" on this tree. A
byte-for-byte comparison of the *whole* QCD block would never satisfy that: `generation_time_seconds`
and `build_time_seconds` are wall-clock measurements that differ on every invocation regardless of
whether the representation moved (confirmed: 0.604 s / 111.683 s on this run, and no two runs of
`generate_references.py` itself would agree on these either). I therefore defined `SCIENCE_KEYS`
to exclude them, and excluded `method` because the prompt says the generator must "update" it and
"append" a provenance note whenever the block is regenerated — appending unconditionally would
make every re-run, including a true no-op, report a change. The alternative (comparing the whole
block, accepting that `--dry-run` would then never report "no change") was rejected because it
directly contradicts the prompt's own acceptance test in §3. A consequence, stated in the script's
docstring: on a run where the science content is unchanged, the tool writes **nothing at all**,
not even to refresh the timing figures or append a no-op provenance note — the simplest way to
guarantee the acceptance test holds and to keep "regeneration diff shows only the numbers that
moved" true in the strongest sense (no diff at all when no number moved).

### 3. `T_Z_REPRESENTATION_VERSION` handled defensively — STRUCTURALLY REQUIRED

The prompt's §2 item 1 says the provenance note must record "at what `T_Z_REPRESENTATION_VERSION`"
the regeneration happened, but that constant is introduced by prompt 03, which has not yet run.
`_t_z_representation_version()` looks it up with `getattr(QCD_Cosmology, "T_Z_REPRESENTATION_VERSION", None)`
and substitutes an explicit "not yet introduced (prompt 03)" note when it is absent, rather than
raising or guessing a value. This path was exercised on this tree (no change, so the provenance
note was never written) but not on a tree with the constant present, since prompt 03 has not run;
the substitution logic itself was verified by a standalone call.

### 4. Two new §3 issues opened rather than folded into G1's existing tracking — IMPLEMENTATION CHOICE

While building the map I found two things worth recording that the board did not already carry:
the JSON's `convergence` block has its own, separate generator
(`docs/gktk-remedial/residual_convergence.py`) that this prompt's script does not touch, and two
specific test assertions are pinned to today's ~404-knot count and will need editing (not just
re-measuring) once prompt 07 lands. Both are already implied in outline by G1's tracking (owned by
prompts 07/08), but neither was previously recorded at the level of a specific file and assertion.
I opened them as two new, separate `[qcd-background-audit]` issues
(`[01-convergence-block-has-a-separate-generator]`, `[02-fixture-tests-pinned-to-todays-break-point-artefact]`)
rather than appending a note to G1's existing row, on the view that a later reader of
`IMPLEMENTATION_STATE.md` §3 should be able to find "which two tests will break" without first
reading `REFERENCE-FIXTURE.md` end to end. `docs/OPEN_ISSUES.md`'s count and date are updated in
this same commit (56 → 58).

## Verification performed

**`--dry-run` reproduction** (the prompt's stated acceptance test):

```
$ PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py --dry-run
   QCDModel: compute_background + splines took 0.604 s
   reference quadrature took 111.683 s
** QCD block science-content comparison against the shipped file:
  tau_minus_top: unchanged
  cs_tau_minus_top: unchanged
  friction_F_minus_top: unchanged
  rho_G: unchanged
  rho_T: unchanged
  rho_anchor_z: unchanged
  primitives_at_rho_anchor: unchanged
  short_baseline: unchanged
  reference_floor: unchanged
  grid: unchanged
  z_top: unchanged
  checkpoints: unchanged
No change: the QCD block's science content (12 keys) is bit-identical to the shipped file.
Total wall time 112.297 s (build 0.604 s, reference quadrature 111.683 s).
```

Exit code `0`. `git status --porcelain ComputeTargets/tests/wkb_reference_data.json` printed
nothing, both immediately before this run and immediately after.

**Suite counts** (PYTHONPATH=. ./venv/bin/python -m unittest discover, from the repository root),
baseline vs. after this prompt — **unchanged, as expected, since no test file was touched**:

| Suite | Baseline (`README.md`) | After this prompt |
|---|---|---|
| `CosmologyModels` | 18 | 18, `OK`, 0.111 s |
| `ComputeTargets` | 339 | 339, `OK`, 156.7 s |
| `LiouvilleGreen` (fast set) | 143 | 143, `OK`, 17.2 s |

`LiouvilleGreen` was run as the **fast set**: all eleven modules except
`test_3bessel_analytic` (`test_bessel_compatibility`, `test_bessel_near_region`,
`test_bessel_phase`, `test_bessel_reference`, `test_bessel_tail`, `test_bessel_two_region`,
`test_phase_spline`, `test_range_reduce`, `test_scipy_bessel_domain`, `test_three_bessel`,
`test_wkbtools`), matching the baseline's own count for that set (143 in ~15 s) — this prompt
touches nothing `LiouvilleGreen/` imports, so the excluded module was not expected to move either,
but the full 148-test run (`test_3bessel_analytic` included, ~1400 s) was not completed within
this session and is not claimed here.

**The two floor issues** (prompt §2 item 4), re-measured by reading the shipped JSON's own
`convergence` block (not independently recomputed — see Deviation 4/`[01-convergence-block-has-a-separate-generator]`
above for why this script does not itself own that computation):

- `[02-qcd-reference-floor]`: `convergence.models.QCDModel["branch+knots"]["tau"]["json_vs_reference_max_rel"]
  = 1.8785409360313507e-14`; the `cs_tau` equivalent is `1.8866533733829712e-14`. Both match the
  board's quoted "~1.9e-14" to the digit.
- `[03-qcd-short-baseline-reference-endpoint-rounding]`: not a JSON field; ~1e-13 relative on the
  37% fractions near z=1e6, per `test_background_tau.py`'s own docstring and `GkTk-remedial` log
  03 (shipped table agrees with an exact-endpoint `quad` to ≤8.8e-16). This prompt's regenerated
  `short_baseline` block reproduces the shipped values exactly (see the "unchanged" line above), so
  it carries the identical rounding; the figure itself is unchanged by this prompt.

**Map exhaustiveness**: verified the file list by running the prompt's own `grep` command (§3 of
`REFERENCE-FIXTURE.md`) rather than trusting the prompt's quoted list, and by reading each of the
eleven `ComputeTargets/tests/` modules plus prompt 01's two `CosmologyModels/tests/` modules in
full — not by grepping for "QCD" inside method bodies alone, which under-counts tests that
reference a QCD fixture built in `setUpClass` without repeating the word "QCD" in every method
(`test_omega_eff_split.py`'s six tests were found this way).

## Observations not acted on

1. **The `convergence` block is a second, uncovered generator.** Recorded as
   `[01-convergence-block-has-a-separate-generator]` (§3 above and the board). Not fixed here: this
   prompt's charter is the QCD-only `models.QCDModel` block, and the `convergence` block belongs to
   a different script that prompt 08's own charter already re-measures.
2. **Two assertions are pinned to today's break-point artefact and will need editing, not just
   re-measuring.** Recorded as `[02-fixture-tests-pinned-to-todays-break-point-artefact]`. Named
   precisely (`test_numeric_break_points.py::TestDeclaration::test_kind_selects_knots_or_jumps`,
   `test_background_tau.py::TestBackgroundTau::test_qcd_break_points`) so prompt 07 does not have to
   re-discover them.
3. **`TestManyBreakPoints`'s docstring in `test_numeric_break_points.py`** states, as fact, that
   "the T(z) spline's knots are 2.85e-02 apart in log(1+z)" — true today, and it will become a
   stale description (not a failing assertion — it is prose, not a threshold) once prompt 07
   removes the knots. Same class of drift as the already-tracked
   `[19-cosmologymodels-docstrings-predate-per-sector-policy]`; not given its own issue, since it
   is documentation rather than a test that can fail, but worth a line in prompt 07's log.
4. **`test_qcd_break_points`'s `expected` count reads `convergence.geometry.QCDModel.T_spline_knots_in_range`.**
   Even after that field is correctly re-measured to reflect the collapsed break-point set, the
   test's own logic (`expected = T_spline_knots_in_range + len(branch_boundaries)`) still needs the
   `convergence` block regenerated in step with the representation, or it reads a stale count from
   the wrong script. Folded into observation 1/2 above rather than opened as a third issue.

## State handed to the next prompt

- **Exact regeneration command:**
  `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/generate_qcd_references.py` (append
  `--dry-run` to check without writing). No Ray, no datastore.
- **Wall time:** 112.3 s total on this machine (0.6 s to build `QCDModel` via `compute_background`
  on the ~1,732-point production grid; 111.7 s for the converged adaptive-quadrature reference
  itself, dominated by the per-interval `scipy.quad` + Gauss-40-bisection cross-checks). Prompts
  04–07 should expect a similar wall time each time they regenerate the fixture.
- **`--dry-run` result on this tree:** no change, all 12 `SCIENCE_KEYS` bit-identical (quoted in
  full above).
- **The map:** `docs/qcd-background-audit/REFERENCE-FIXTURE.md`. 45 QCD-dependent test methods;
  by "moved by": node 15, shape 18, breaks 15, none 20 (a test may count in more than one bucket).
  Two assertions need editing rather than re-measuring once prompt 07 lands (named above and in
  the map's §5).
- **The two floor figures**, as the `convergence` block (not this prompt's script) reports them:
  `[02-qcd-reference-floor]` = 1.8785409360313507e-14 (tau) / 1.8866533733829712e-14 (cs_tau);
  `[03-qcd-short-baseline-reference-endpoint-rounding]` ≈ 1e-13 (not a JSON field; see above).
- **`T_Z_REPRESENTATION_VERSION`**: still does not exist at this commit. Unchanged by this prompt.
- **A caution for prompts 04–07**: regenerating the QCD block does **not** regenerate the
  `convergence` block (`[01-convergence-block-has-a-separate-generator]`). If a
  `convergence`-scored test's accuracy figure needs to move (not just a break-point count), the
  prompt that moves it must say so explicitly and either re-run
  `docs/gktk-remedial/residual_convergence.py` or note why it did not.
