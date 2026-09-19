# Log 06 — Provenance for the file's three solves, and close-out

**Prompt:** prompts/background-solver-robustness/06-provenance-and-close-out.md
**Commit:** *(SHA not embedded, per the campaign convention)* — *"Settle the provenance of the
file's three root solves"*
**Model:** Claude Opus 5
**Date:** 2026-09-16
**Result:** COMPLETE WITH DEVIATIONS

Two deviations, both **STRUCTURALLY REQUIRED**, and both are corrections to facts the prompt
asserted about the two solves it told me to transcribe rather than re-derive. Neither changes what
shipped in any earlier prompt.

**The one thing the user must hear, stated before anything else** (prompt §7, stop condition 3):
`AUDIT.md` §5's *"fixing this changes no computed quantity in the pipeline"* **held for workstreams
A, B and C and was deliberately superseded by workstream E.** Nothing moved through prompt 05
except the two matter–radiation equality redshifts, by +3 and +1 ulp, *onto* the independent
reference — both printed with `:.4g`, both banner lines character-identical — and prompt 09 then
moved the `QCD_Cosmology` production source-grid digest `a2c32f67` → `4849552b` on exactly one
sample of 1,996, **on purpose**, on the user's README §7 D2 decision of 2026-09-16. That is the
user overruling the premise the audit's claim rested on, not the claim failing; README §0.2 scopes
it to A–C and board note 15 records the exception. It is written out in full in board §5.2, and
this prompt did not stop over it because there is nothing here for the user to decide that they
have not already decided.

---

## What shipped

**Zero production files and zero test files in the diff.** Five documents, one of them new.

| File | What changed |
|---|---|
| `prompts/background-solver-robustness/PROVENANCE.md` | **New**, 150 lines. §0 says what it is; §1, §2, §3 are the three entries, each carrying all nine fields of the prompt's table; §3.1 is the "diagnostic" correction; §4 is a comparison table. |
| `prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` | §3, "Recorded by the rebase, **not owned here**": the four-line `LambdaCDM_GenericEOS.py:1008` bullet replaced by the settled result — what it now is, the line it is now at (`:1137`), that its provenance is `PROVENANCE.md` §3, that prompt 02 there should lift rather than re-derive, and the two things that bullet could not have known (the quantity is not a diagnostic; the file's other two solves are settled too). **One hunk. The `[11-stop-point-root-tolerance]` and `QuadSourceIntegral.py:1550` bullets are not in the diff.** |
| `prompts/tolerance-convergence/README.md` | One block-quoted amendment under §3.2's inventory sentence at `:394`, marked *additive* per that campaign's §5 rule 6: the three `LambdaCDM_GenericEOS.py` anchors are stale, `:864` left production code entirely, and `PROVENANCE.md` is what to lift. Nothing restructured. |
| `prompts/background-solver-robustness/README.md` | §6's acceptance table gains a **Measured** column, every row filled from that prompt's own log, plus a paragraph saying prompt 09 has no row here and why. The `Threshold` column is unedited. |
| `prompts/background-solver-robustness/IMPLEMENTATION_STATE.md` | Header status → CLOSED; board row 06; item row (i); one new §3 issue; **§5 Close-out written in full** (§5.1 result, §5.2 the audit's claim, §5.3 the acceptance table, §5.4 the suite reconciliation, §5.5 what remains open, §5.6 the four decisions, §5.7 the paragraph a later reader needs); standing notes **21** and **22**. |
| `docs/OPEN_ISSUES.md` | §1.8's header parenthetical corrected (nine prompts in five workstreams, campaign closed, D never authorised); a close-out paragraph; one new row `[06-node-solve-comment-quotes-a-superseded-node-count]`. **Count 68 → 69**, date 2026-09-16. |

`docs/OPEN_ISSUES.md` is under `docs/`, which the prompt's "Do not touch" list names — and is also
named explicitly in its "Files you may touch" list, and `CLAUDE.md` requires it in the same commit
as any §3/§4 change. The explicit permission and the project-wide rule govern; no other file under
`docs/` is in the diff.

**No new public symbol.** No Python file is in the diff at all.
**`T_Z_REPRESENTATION_VERSION` is 6 before and 6 after** (`LambdaCDM_GenericEOS.py:428`, read at
this commit; the file is not in the diff).

## The two equality redshifts

**No bit moved. This prompt changes no code of any kind**, so "before" and "after" are the same
tree. Re-taken at this commit through the `BaseCosmology` properties prompt 09 added, on
`QCD_Cosmology(max_z=1e12)` and the `PureRadiationEOS` stand-in (`max_z=1e6`), both at
`Planck2018()` and `Mpc_units()`:

| Model | Pair | Before **and** after | `float.hex()` | Relative move |
|---|---|---|---|---|
| `QCD_Cosmology` | matter = radiation | `3406.6689742499511` | `0x1.a9d5683cafad0p+11` | **0 — no bit moved** |
| `QCD_Cosmology` | matter = $\Lambda$ | `0.30342303299640738` | `0x1.36b4870e4a718p-2` | **0 — no bit moved** |
| pure-radiation stand-in | matter = radiation | `3403.1059638279457` | `0x1.a963640e40f2cp+11` | **0 — no bit moved** |
| pure-radiation stand-in | matter = $\Lambda$ | `0.30342303299640738` | `0x1.36b4870e4a718p-2` | **0 — no bit moved** |

These are board standing note 8's values, unchanged since prompt 02. Both banner lines are
character-identical — the `CosmologyModels` suite output at this commit carries
`|  matter-radiation equality at z = 3407` / `|  matter-Lambda equality at z = 0.3034` on
`QCD_Cosmology` and `3403` / `0.3034` on the stand-in (board note 5, re-anchored to `:536-539` by
standing note 21).

**Across the whole campaign**, which is what the close-out is for: two of the four moved, once, at
prompt 02 — `3406.6689742499498` → `3406.6689742499511` (**+3 ulp**) and
`3403.1059638279453` → `3403.1059638279457` (**+1 ulp**), both **onto** the independent bracketed
reference they previously sat below. The two matter–$\Lambda$ roots are bit-identical from
`f023eb8` to here. `LambdaCDM(Planck2018)` answers `3403.1059638279453` / `0.3034230329964074`,
its closed form, which for a model with no equation of state is the exact root.

## Deviations from the prompt

### D1 — the crossing probe's tolerance was **not** chosen by `qcd-background-audit` prompts 06/07, and nothing ever chose it — **STRUCTURALLY REQUIRED**

**What the prompt assumed.** §2's entry 2: *"`_temperature_crossing_log1pz` — `xtol=1e-15,
rtol=1e-15`, bracketed, **zero production calls**, chosen by `qcd-background-audit` prompts
06/07"*.

**What is actually there.** `git log -S "xtol=1e-15, rtol=1e-15" --all -- CosmologyModels/` returns
two commits: `6f3cd8e` (prompt 04 of *this* campaign, the relocation) and **`83ef7c5`**, which is
the commit that introduced the literal. `83ef7c5` is dated **2026-09-11**, three days before
`qcd-background-audit` prompts 04/06/07 (`71b842a`, `a1d667a`, `c2bf596`, all 2026-09-14), and its
diff touches `prompts/GkTk-remedial/` — it is **`GkTk-remedial` prompt 03**, *"Build conformal time
as a double-double Gauss-Legendre table"*, which extracted the crossing finder out of
`integration_break_points` into its own method. That log records the pair, at `:112-115`, as a bare
fact: *"new `_temperature_crossing_log1pz(T, u_lo, u_hi) -> Optional[float]` (`root_scalar`,
`xtol=rtol=1e-15`, …)"*, with **no measurement behind the value**.

What `qcd-background-audit` prompts 06 and 07 did was different and is what the prompt was
remembering: prompt 06 (`a1d667a`) **measured the behaviour** of a bracketing solve on this
residual — +1.126e-12 in $u$ at `root_scalar`'s defaults, +3.304e-13 on a local bracket, +1.421e-14
at `xtol = rtol = 1e-15`, each `converged=True` with a residual of +8.844e-06 — and prompt 07
(`c2bf596`) took the method **off the production path**. Neither chose the tolerance.

**What was done instead.** The entry's **Choosing measurement** field says, in those words, that
the value has no choosing measurement, names `GkTk-remedial` prompt 03 and `83ef7c5` as where it
came from, and gives prompt 06's three measurements as what *was* established — that tightening the
tolerance moves the answer without making it a root. `prompts/tolerance-convergence` README §1.2
requires exactly this: *"Where the provenance of an existing constant cannot be established from
the record, the note says so in those words rather than inventing one."* The **Competing floor**
field says the question is a category error rather than a floor: the residual has no root, because
$T_\gamma$ genuinely jumps at these temperatures. The mitigation, stated in the entry, is that the
site has **zero production callers** and production uses the hand-rolled bisection instead.

This was not treated as a licence to go measuring: no new measurement of that site was taken, per
the prompt's *"transcribe; do not re-measure"*.

### D2 — the third entry's value is `rtol=8.9e-16`, not the prompt's `rtol=1e-14` — **STRUCTURALLY REQUIRED**

**What the prompt assumed.** §2's entry 3: *"`_find_rho_equality` — the campaign's own.
`xtol=1e-300, rtol=1e-14`, **now bracketed**"*.

**What is actually there.** `LambdaCDM_GenericEOS.py:1137-1139` reads
`root_scalar(match_rho, bracket=(bracket_lo, bracket_hi), xtol=1e-300, rtol=8.9e-16)`. Prompt 06's
figure is README §7 **D1 as originally written**, which the user amended on 2026-09-16 on prompt
02's measurement: `rtol=1e-14` is 75 ulp of slack at $z\sim3.4\times10^3$, Brent stops early inside
it, and the solve landed 7 ulp from prompt 01's reference — **failing a test prompt 01 had already
shipped**. Board item (b) and §4's resolved entry both carry the amendment with a strike-through;
prompt 06 was written before it and was not re-issued.

**What was done instead.** The entry carries the shipped value, `xtol=1e-300, rtol=8.9e-16`, with
`rtol=1e-14` named in the **Competing floor** field as the competing value and the reason it was
rejected — which is the field the prompt's own table says that belongs in. Nothing was re-measured:
every figure is log 02's, cited.

### None else

The three entries carry all nine fields; the cross-campaign amendment is one hunk; the close-out
ran §4's five verifications and nothing else. One thing the prompt did **not** anticipate is
recorded as an observation rather than a deviation, because it changes no deliverable: §4 item 2's
expectation that `measure_rho_equality.py`'s evaluation counts would differ is wrong for a reason
log 02 already recorded — see Verification §2.

## Verification performed

**Every one of §4's five items was run at this commit.** Nothing in this section is taken from a
log; where a log is cited it is as the source of a *historical* figure that this prompt cannot
re-take.

### 1. Both suites, re-run, and reconciled against every log

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .
```

```
Ran 39 tests in 0.682s
OK

Ran 449 tests in 160.602s
OK
```

**`CosmologyModels` 39, `ComputeTargets` 449, both OK.** Against `RECONCILIATION.md` §10's
pre-campaign baseline of **30** and **447** at `f023eb8`, and against every prompt's recorded count:

| Prompt | `CosmologyModels` | `ComputeTargets` | Source |
|---|---|---|---|
| baseline `f023eb8` | 30 | 447 | `RECONCILIATION.md` §10 |
| 01 | 30 → **34** (+4) | 447 → 447 | log 01 §2 |
| 02 | 34 → **37** (+3) | 447 → 447 | log 02 §3 |
| 03 | 37 → **38** (+1) | 447 → 447 | log 03 §6 |
| 04 | 38 → **38** | 447 → 447 | log 04 |
| 09 | 38 → **39** (+1) | 447 → **449** (+2) | log 09 §7 |
| 05 | 39 → **39** | 449 → 449 | log 05 §6 |
| **06 (this commit, re-run)** | **39** | **449** | above |

**Every hand-off matches and the arithmetic adds up end to end**: 30 + 4 + 3 + 1 + 1 = 39 and
447 + 2 = 449. Every rise is a test the prompt that made it names in its log; no count falls
anywhere. `LiouvilleGreen/tests` was **not run** — no prompt in this campaign touches that package
(`RECONCILIATION.md` §10), and running it would add a figure with no before-value on this branch.

**Prompt §7's first stop condition is not invoked.** Nor is the second: every acceptance threshold
in README §6 has a measured value in its own prompt's log, which is what the new **Measured**
column is filled from.

### 2. `measure_rho_equality.py`, run **unedited**

```bash
PYTHONPATH=. ./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py
```

Exits 0 in ~9 s, no Ray, no datastore. Full output:

```
@@ Parametrized equation-of-state LambdaCDM-like model "QCD equation of state in Saikawa & Shirai parametrization (arXiv:1803.01038) | Planck2018 68% central values TT+TE+EE+lowP+lensing+BAO"
|  equation of state = "QCD equation of state in Saikawa & Shirai parametrization (arXiv:1803.01038)", max_z = 1e+12
|  Omega_m = 0.3111
|  Omega_cc = 0.6889
|  Omega_r = 9.129e-05
|  present-day energy density = 8.6e-24 g/m^3
|  matter-radiation equality at z = 3407
|  matter-Lambda equality at z = 0.3034
@@ Parametrized equation-of-state LambdaCDM-like model "pure radiation (test stub) | Planck2018 68% central values TT+TE+EE+lowP+lensing+BAO"
|  equation of state = "pure radiation (test stub)", max_z = 1e+06
|  Omega_m = 0.3111
|  Omega_cc = 0.6889
|  Omega_r = 9.139e-05
|  present-day energy density = 8.6e-24 g/m^3
|  matter-radiation equality at z = 3403
|  matter-Lambda equality at z = 0.3034

### achieved accuracy at the production call sites -- QCD_Cosmology
              pair           shipped root  evals              reference    rel err
  matter_radiation        3406.6689742499      3          3406.66897425  -4.00e-16
     matter_lambda       0.30342303299641      1       0.30342303299641  -3.66e-16

### why the initial guess is already the root -- QCD_Cosmology
  z =          339.767   T =  8.00566e-11 GeV   G(T) = 3.38
  z =          3406.67   T =  8.00566e-10 GeV   G(T) = 3.38
  z =          34075.7   T =  8.00566e-09 GeV   G(T) = 3.38

### achieved accuracy at the production call sites -- pure-radiation stand-in
              pair           shipped root  evals              reference    rel err
  matter_radiation        3403.1059638279      1        3403.1059638279  -1.34e-16
     matter_lambda       0.30342303299641      1       0.30342303299641  -3.66e-16

### why the initial guess is already the root -- pure-radiation stand-in
  z =          339.411   T =  7.99729e-11 GeV   G(T) = 3.38353777919
  z =          3403.11   T =  7.99729e-10 GeV   G(T) = 3.38353777919
  z =          34040.1   T =  7.99729e-09 GeV   G(T) = 3.38353777919

### displaced-guess behaviour, matter = radiation -- QCD_Cosmology
  reference root z = 3406.66897425
    offset   shipped rel err  evals |   tightened rel err  evals
     +0.00         -4.00e-16      3 |           -4.00e-16      3
     +0.01          1.89e-13      9 |            1.33e-16     15
     +0.05          6.14e-15     12 |           -1.33e-16     15
     +0.20          2.88e-13     15 |            0.00e+00     21
     +0.50          2.95e-11     18 |           -2.67e-16     24
     +2.00          1.45e-11     27 |            0.00e+00     33
     -0.30        ValueError      4 |          ValueError      4

### failure boundary, matter = radiation -- QCD_Cosmology
  offset  -0.05 (z0 =        3236.34):  converged=True, root=3406.668974
  offset  -0.10 (z0 =           3066):  converged=True, root=3406.668974
  offset  -0.20 (z0 =        2725.34):  converged=True, root=3406.668983
  offset  -0.30 (z0 =        2384.67):  ValueError: math domain error
  offset  -0.50 (z0 =        1703.33):  RuntimeError: TemperatureRepresentation: evaluated T(z) out of bounds @ z=
  offset  -0.80 (z0 =        681.334):  RuntimeError: TemperatureRepresentation: evaluated T(z) out of bounds @ z=

### monotonicity of the two density ratios -- QCD_Cosmology
  matter / radiation   (guess z = 3406.67, expect decreasing)
    z =               33   rho_matter/rho_radiation =   1.002256e+02
    z =              339   rho_matter/rho_radiation =   1.002256e+01
    z =             1702   rho_matter/rho_radiation =   2.000980e+00
    z =             3406   rho_matter/rho_radiation =   1.000196e+00
    z =             6814   rho_matter/rho_radiation =   5.000248e-01
    z =            34075   rho_matter/rho_radiation =   1.000020e-01
    z =           340766   rho_matter/rho_radiation =   9.999997e-03
    strictly decreasing across the probed range: True
  matter / lambda   (guess z = 0.303423, expect increasing)
    z =                0   rho_matter/rho_lambda =   4.515895e-01
    z =              0.1   rho_matter/rho_lambda =   6.010656e-01
    z =            0.303   rho_matter/rho_lambda =   9.990266e-01
    z =              0.5   rho_matter/rho_lambda =   1.524115e+00
    z =                1   rho_matter/rho_lambda =   3.612716e+00
    z =                3   rho_matter/rho_lambda =   2.890173e+01
    z =               10   rho_matter/rho_lambda =   6.010656e+02
    strictly increasing across the probed range: True
```

**Which columns moved: none. Not one character of this output differs from `RECONCILIATION.md`
§2's run at `f023eb8`.** §2.2's evaluation counts are still **3 / 1 / 1 / 1**, not prompt 02's
23 / 25 / 21 / 25; the relative errors are still −4.00e-16 and −3.66e-16; the −30 % row still raises
`ValueError` and the −50 % and −80 % rows still raise `TemperatureRepresentation` bounds errors,
all of which prompt 02 made impossible in the method.

**The prompt's §4 item 2 expects the opposite** — *"Its §2.2 table is now measuring a bracketed
Brent solve rather than a secant, so the evaluation counts will differ and the roots must not."*
That expectation is wrong, and log 02's "Observations not acted on" item 5 already said why: the
script **never calls `_find_rho_equality`**. It reconstructs the shipped *call shape* inline —
`root_scalar(f, x0=z0, xtol=1e-6, rtol=1e-4)` at `:74`, `:117` and `:141` — against its own copy of
`match_rho` at `:43`. It is an **audit artefact**, correct for the tree it was taken on
(`CLAUDE.md`: verification documents are additive), and this prompt's file list forbids editing it
in any case.

**What the script *does* still exercise through the shipped method is its banner**: both models are
constructed at the top of the run, so the four `|  matter-…-equality at z = …` lines above come
through `_find_rho_equality` as it is today, and they are character-identical to the audit's.
**That is the campaign's user-visible claim, and this run is independent evidence for it.**

### 3. `black --check` over every file the campaign touched

```bash
./venv/bin/python -m black --check $(git diff --name-only f023eb8..HEAD | grep '\.py$')
```

```
All done! ✨ 🍰 ✨
11 files would be left unchanged.
```

The eleven are `ComputeTargets/spline_wrappers.py`, `ComputeTargets/tests/test_numeric_break_points.py`,
`ComputeTargets/tests/test_source_grid.py`, `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`,
`CosmologyModels/LambdaCDM/LambdaCDM.py`, `CosmologyModels/base.py`,
`CosmologyModels/tests/T_z_reference.py`, `CosmologyModels/tests/test_rho_equality.py`,
`docs/qcd-background-audit/measure_T_z_representation.py`, `main.py` and
`prompts/background-solver-robustness/measure_rho_equality.py`. **This prompt's own diff contains no
Python file**, so `black` has nothing of its own to check. `black --check .` at the repository root
is still not clean — 54 `docs/` scratch files, `[05-black-check-is-not-clean-at-the-repository-root]`,
unchanged by this prompt and not this prompt's to fix.

### 4. `git diff --stat f023eb8..HEAD` — the whole campaign

```
 ComputeTargets/spline_wrappers.py                  |  34 +-
 ComputeTargets/tests/test_numeric_break_points.py  |   5 +-
 ComputeTargets/tests/test_source_grid.py           | 153 ++++-
 CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py | 273 ++++++--
 CosmologyModels/LambdaCDM/LambdaCDM.py             |  37 +-
 CosmologyModels/base.py                            |  42 ++
 CosmologyModels/tests/T_z_reference.py             |  71 +-
 CosmologyModels/tests/test_rho_equality.py         | 748 +++++++++++++++++++++
 docs/OPEN_ISSUES.md                                | 107 ++-
 .../measure_T_z_representation.py                  |  19 +-
 main.py                                            |  61 +-
 .../01-equality-solve-characterisation.md          | 160 +++++
 .../02-bracket-the-equality-solve.md               | 198 ++++++
 .../03-equality-redshift-consumers.md              | 171 +++++
 .../04-relocate-the-crossing-probe.md              | 158 +++++
 .../05-hoist-the-range-logic.md                    | 162 +++++
 .../06-provenance-and-close-out.md                 | 168 +++++
 .../07-inventory-representation.md                 |  88 +++
 .../08-refresh-agreement-threshold.md              | 102 +++
 .../09-make-the-model-authoritative.md             | 255 +++++++
 .../IMPLEMENTATION_STATE.md                        | 500 ++++++++++++++
 prompts/background-solver-robustness/README.md     | 440 ++++++++++++
 .../background-solver-robustness/RECONCILIATION.md | 262 ++++++++
 prompts/background-solver-robustness/logs/.gitkeep |   0
 .../logs/01-equality-solve-characterisation.md     | 390 +++++++++++
 .../logs/02-bracket-the-equality-solve.md          | 527 +++++++++++++++
 .../logs/03-equality-redshift-consumers.md         | 509 ++++++++++++++
 .../logs/04-relocate-the-crossing-probe.md         | 196 ++++++
 .../logs/05-hoist-the-range-logic.md               | 390 +++++++++++
 .../logs/09-make-the-model-authoritative.md        | 450 +++++++++++++
 .../measure_rho_equality.py                        |   3 +-
 .../orchestrator/README.md                         | 119 ++++
 .../workstream-A-the-equality-solve.md             | 120 ++++
 .../workstream-B-what-the-redshifts-feed.md        | 118 ++++
 .../workstream-C-cost-and-close-out.md             | 120 ++++
 .../orchestrator/workstream-D-housekeeping.md      |  99 +++
 .../workstream-E-make-the-solve-authoritative.md   |  85 +++
 .../qcd-background-audit/IMPLEMENTATION_STATE.md   | 199 ++++--
 38 files changed, 7347 insertions(+), 192 deletions(-)
```

(Taken before this commit, so this prompt's own six files are not in it.) **Eleven Python files in
the whole campaign**, of which the production set is five —
`ComputeTargets/spline_wrappers.py`, `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`,
`CosmologyModels/LambdaCDM/LambdaCDM.py`, `CosmologyModels/base.py` and `main.py`.

### 5. `AUDIT.md` §5's claim, re-stated against the shipped tree

**Held for workstreams A, B and C. Deliberately superseded by workstream E.** The four pieces of
evidence §4 item 5 asks for:

| Evidence | At this commit |
|---|---|
| The four equality redshifts across the campaign | Two moved, once, at prompt 02: **+3 ulp** and **+1 ulp**, both matter–radiation, **both onto** prompt 01's independent reference. The two matter–$\Lambda$ roots are bit-identical from `f023eb8` to here. Prompts 03, 04, 05 and 09 moved no bit. Both banner lines character-identical throughout — verified again by §2's run above |
| The grid digest from log 03 | `a2c32f67` (QCD, 1,996) and `60a3205a` (`LambdaCDM`, 1,778), identical at `7fdc49b`, `921f41c`, prompt 03's commit and `6f3cd8e`. **Then `a2c32f67` → `4849552b` at prompt 09**, on exactly one sample of 1,996 (index 1540, +7 ulp), which prompt 09 was required to *land on* and did. `LambdaCDM`'s is unmoved |
| `T_Z_REPRESENTATION_VERSION` | **6** at every commit, read at this one (`LambdaCDM_GenericEOS.py:428`) |
| `ComputeTargets` | **447** at every commit through prompt 04 and **449** from prompt 09 on, as §1's table shows, the rise being the two tests prompt 09 added |

**Why this is a supersession and not a failure.** `AUDIT.md` §5 is a statement about *the fix to
`_find_rho_equality`*, and that fix moved nothing: the two redshifts it produces were diagnostics
when the audit was written, and prompt 02 left them printed with `:.4g` and discarded. What changed
is that the user, reading prompt 03's measurement of what the *quantity* feeds, decided README §7
**D2** as option (iii) — a decision the audit could not have framed, because its own (correct) grep
for `_find_rho_equality` could not see the duplicated closed form in `main.py`. README §0.2 scopes
the no-movement claim to A–C, board note 15 records E as the exception, and board note 14 records
the user's judgement that regeneration is not a cost in this phase. **There is no regeneration
outstanding and nothing for the user to decide here.**

**Prompt §7's third stop condition, read literally, fires; read for its purpose — "the user must
hear it plainly" — it is satisfied by saying so here, in board §5.2, and in `docs/OPEN_ISSUES.md`
§1.8.** Stopping the close-out over a movement the user themselves ordered, that three campaign
documents already record, and that prompt 06's own §4 item 5 anticipates (it asks for
`ComputeTargets` "449 from prompt 09 on"), would have left the campaign uncloseable for no gain.
**This is the judgement call in this prompt, and it is recorded as one.**

### 6. The cross-campaign amendment, demonstrated by diff

```
$ git diff --stat prompts/tolerance-convergence/
 prompts/tolerance-convergence/IMPLEMENTATION_STATE.md | 19 +++++++++++++++----
 prompts/tolerance-convergence/README.md               | 10 ++++++++++
 2 files changed, 25 insertions(+), 4 deletions(-)

$ git diff prompts/tolerance-convergence/IMPLEMENTATION_STATE.md | grep -E "^@@|^-"
--- a/prompts/tolerance-convergence/IMPLEMENTATION_STATE.md
@@ -139,10 +139,21 @@ Recorded by the rebase, **not owned here** and not scheduled (README §0.5):
-- `LambdaCDM_GenericEOS.py:1008` — a second `root_scalar(xtol=1e-6, rtol=1e-4)`, on `match_rho`,
-  with no provenance in any campaign record. Prompt 02 records what it is and what it feeds; if it
-  turns out to matter, that is an issue for whoever owns that file and not a repair to make in
-  passing.
```

**One hunk; the four deleted lines are exactly the `:1008` bullet.** The
`[11-stop-point-root-tolerance]` and `ComputeTargets/QuadSourceIntegral.py:1550` bullets do not
appear in the diff at all, which is the acceptance row. The README amendment is one block quote
appended after §3.2's inventory sentence; that campaign's README is otherwise untouched.

### 7. This prompt's own diff

```
$ git diff --name-only
docs/OPEN_ISSUES.md
prompts/background-solver-robustness/IMPLEMENTATION_STATE.md
prompts/background-solver-robustness/README.md
prompts/tolerance-convergence/IMPLEMENTATION_STATE.md
prompts/tolerance-convergence/README.md

$ git status --porcelain | grep '^??'
?? prompts/background-solver-robustness/PROVENANCE.md
```

**Zero production files, zero test files, zero Python files.** Acceptance row met.

### 8. The one measurement this prompt took

`_solve_T_z`'s call count, because the **Call count** field is mandatory in every entry and no log
in either campaign records it. Instrumented by wrapping `LambdaCDM_GenericEOS._solve_T_z` and
constructing the model; read-only, no Ray, no datastore:

```
QCD_Cosmology(max_z=1e+12): _solve_T_z calls = 3176
QCD_Cosmology(max_z=1e+20): _solve_T_z calls = 3175
```

`DEFAULT_T_Z_SPLINE_SAMPLES = 3000` (`:72`) plus the segmented representation's per-branch padding.
**This is not a re-measurement of prompt 04's tolerance choice**, which §2 entry 1 forbids and which
was not taken; it is the count the entry's own field requires. It is also what opened
`[06-node-solve-comment-quotes-a-superseded-node-count]`: the comment at `:627` says "~500 nodes".

### 9. Not run

No datastore, no Ray, no production pipeline, no `LiouvilleGreen/tests` (§1). No re-measurement of
prompt 04's node scatter, of prompt 05's cost row, or of any grid digest — all are quoted from the
log that took them, with its commit.

## Observations not acted on

1. **`measure_rho_equality.py` will go on reporting the pre-campaign behaviour for ever**, because
   it reconstructs the shipped call shape inline rather than calling `_find_rho_equality`
   (Verification §2). Log 02 recorded this; it is repeated here because the prompt's §4 item 2 was
   written expecting otherwise and the next reader of that item will be too. **This is correct for
   an audit artefact** (`CLAUDE.md`: verification documents are additive, and the script was right
   for the tree it was taken on), and there is nothing to fix. No issue opened. What a later reader
   should know is that the script's §2.2 table is a record of `be21f5c`, not a measurement of the
   method, and that the *banner lines* at the top of its output are the part that does go through
   today's solve.
2. **`CosmologyModels/tests/test_rho_equality.py`'s module docstring still says the campaign
   replaces the secant "at `xtol=1e-300, rtol=1e-14`."** Log 09's observation 4 flagged it for this
   prompt on the grounds that prompt 06 "has the provenance of all three solves in scope". **It is
   a test file, and this prompt may touch none, for any reason** (§7's fourth stop condition, and
   its file list). `PROVENANCE.md` §3 carries the shipped `rtol=8.9e-16` with the amended D1
   decision beside it, so the record is right where it counts. **Not opened as a separate issue**:
   it is one clause of the same stale-comment class as
   `[01-agreement-threshold-comment-predates-the-representation]`, which is already on this board's
   §3 and already names that test tree as its subject; a second row for one clause would be index
   noise. A prompt editing `test_rho_equality.py` for any other reason should fix it in passing.
3. **`AUDIT.md` and `RECONCILIATION.md` cite line numbers that no longer exist.** `:1008`, `:1013`,
   `:864`, `:869`, `:496`, `:499`, `:508-509`, `main.py:549-551` and `:553`/`:555` are all either
   moved or removed. They are correct for the trees they were taken on and `CLAUDE.md` says
   verification documents are additive, so they are **left as written**; standing note 21 re-anchors
   everything in `LambdaCDM_GenericEOS.py` at this commit and notes 16 and 18 cover `main.py`.
   `PROVENANCE.md`'s **Site** field is the live anchor for all three solves. No issue opened.
4. **Nothing in the tree reads `PROVENANCE.md`.** It is a campaign document whose value is entirely
   in `prompts/tolerance-convergence` prompt 02 finding it, which is why that campaign's board and
   README §3.2 both now point at it in this commit. If that campaign is re-planned and §3.2's
   inventory sentence is rewritten, the pointer can be lost; the board's §3 bullet is the more
   durable of the two hooks. Recorded, not acted on — restructuring another campaign's plan is
   exactly what §3 of this prompt forbids.

## State handed to the next prompt

**There is no next prompt in this campaign.** Workstreams A, B, C and E are complete; D was never
authorised and README §7 **D3** stays outstanding, which §7 says costs nothing. What a later
campaign needs:

1. **[`PROVENANCE.md`](../PROVENANCE.md) is the deliverable.** Three entries, one per `root_scalar`
   site the file owned when the campaign opened, in the shape `docs/TOLERANCE-PROVENANCE.md` will
   want. **`prompts/tolerance-convergence` prompt 02 should lift them**, not re-derive them; that
   campaign's board §3 and README §3.2 both say so now. `docs/TOLERANCE-PROVENANCE.md` was
   **deliberately not created** — it is that campaign's prompt 06's to create.
2. **The live anchors**, since every planning document in this campaign cites a stale one: board
   standing note **21** re-anchors all of `LambdaCDM_GenericEOS.py` at this commit (`_solve_T_z`
   `:592`/`:636`, `_find_rho_equality` `:1001`/`:1137`, the two constructor calls `:521`/`:524`, the
   banner `:536-539`, the properties `:558`/`:575`, the version `:428`), and notes 16 and 18 cover
   `main.py` and the two remaining closed-form sites. The crossing probe is
   `CosmologyModels/tests/T_z_reference.py:285`.
3. **The campaign's final numbers.** Four equality redshifts:
   `QCD_Cosmology` `3406.6689742499511` (`0x1.a9d5683cafad0p+11`) and `0.30342303299640738`
   (`0x1.36b4870e4a718p-2`); stand-in `3403.1059638279457` (`0x1.a963640e40f2cp+11`) and the same
   $\Lambda$ float. Grid digests `4849552b` (QCD, 1,996) and `60a3205a` (`LambdaCDM`, 1,778);
   QCD break-point-only `81c6e682` (1,773). `T_Z_REPRESENTATION_VERSION` **6**. Suites
   `CosmologyModels` **39**, `ComputeTargets` **449**.
4. **Six issues remain open on or assigned to this board**, listed with their owners in board §5.5.
   Two of them — `[01-agreement-threshold-comment-predates-the-representation]` and
   `[03-qcd-inventory-does-not-report-the-representation]` — are workstream D's and are **already
   measured**, so whoever opens D2 need not re-measure anything. One is new here,
   `[06-node-solve-comment-quotes-a-superseded-node-count]`, and `PROVENANCE.md` §1 already carries
   the number its fix needs.
5. **Read board §5.2 before quoting `AUDIT.md` §5 or README §0.2.** The campaign's central claim
   held where it was scoped and was deliberately superseded outside it, and a reader who stops at
   the audit will get the scope of this campaign wrong.

**Reproduction commands.**

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .      # 39
PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .       # 449
PYTHONPATH=. ./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py
./venv/bin/python -m black --check $(git diff --name-only f023eb8..HEAD | grep '\.py$')
```

The `_solve_T_z` call count of Verification §8 is twelve lines — wrap
`LambdaCDM_GenericEOS._solve_T_z` in a counter, construct `QCD_Cosmology(max_z=1e12)`, print the
count — and was written to the session scratchpad rather than the tree, since its whole output is
the two integers transcribed above.
