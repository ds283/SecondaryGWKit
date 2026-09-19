# Reconciliation — `AUDIT.md` against the tree, and against `docs/OPEN_ISSUES.md`

**Taken:** 2026-09-16 at `f023eb8` (`tolerance-convergence`).
**Scores:** every claim of [`AUDIT.md`](AUDIT.md) (written at `be21f5c`) against the code at
`f023eb8`, and the audit's subject against the project-wide index
[`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md).
**Reproduction:** `PYTHONPATH=. ./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py`
(~9 s, no Ray, no datastore). **Note the `PYTHONPATH=.`** — the audit's own header omits it and the
script does not run without it (R0 below).

This document exists because `AUDIT.md` was correct when it was written and two of its statements
are no longer the whole truth. It is the record of what changed, what was already wrong, and what
the campaign's scope has to be as a result. **Where this document and `AUDIT.md` disagree, this
document is the later measurement; where they agree, the audit is not restated here.**

---

## 0. Summary

| # | Audit claim | Verdict |
|---|---|---|
| R0 | The reproduction command in the audit header | **CORRECTION** — needs `PYTHONPATH=.`; the script's own docstring is fixed in the planning commit |
| R1 | Every figure in §2.2, §2.3, §3.1, §3.2, §4.1 | **STANDS** — reproduced to the digit printed |
| R2 | The three solves are at `:578`, `:864`, `:1008` | **STANDS, RE-ANCHORED** — `+5` lines since `be21f5c` |
| R3 | §6 observation 1 (the spline error's class name) | **CLOSED** at `f023eb8`; the audit is already amended |
| R4 | §2.1 *"the blast radius of this solve is two banner lines"* | **NARROWED — true of the solve, false of the quantity** |
| R5 | §6 observation 2 (the caller derives the guess) | **WIDENED** — the closed form exists in **three** places, one of them production |
| R6 | — | **NEW** — `main.py:526`'s "4e-13 relative" is stale by three orders |
| R7 | §3.3 (no campaign record states the provenance) | **STANDS, AND THE INDEX HAS A GAP** — `:1008` has no row in `docs/OPEN_ISSUES.md` at all |
| R8 | §4.3 (what must stay out of scope) | **STANDS** — and the index adds five adjacent issues that *are* this campaign's |

---

## 1. R0 — the reproduction command

`AUDIT.md`'s header and `measure_rho_equality.py`'s own docstring both give

```
./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py
```

which fails with `ModuleNotFoundError: No module named 'CosmologyModels'`. The command that works
is the repository convention (`CLAUDE.md`, "Tests"):

```bash
PYTHONPATH=. ./venv/bin/python prompts/background-solver-robustness/measure_rho_equality.py
```

Trivial, and recorded only because "re-run it rather than trusting this document" is the audit's
own instruction and the first person to follow it hits this. **The script's docstring is corrected
in the planning commit**; `AUDIT.md`'s header is left as written, because a campaign's verification
documents are additive and this file is the correction layer (`CLAUDE.md`).

## 2. R1 — the measurements reproduce

Re-run at `f023eb8`. Every number in the audit's tables is reproduced **to the digit printed**:

- §2.2: `3406.6689742499` / 3 evals / **−4.00e-16**; `0.30342303299641` / 1 / **−3.66e-16**;
  the pure-radiation stand-in `3403.1059638279` / 1 / **−1.34e-16** and `0.30342303299641` / 1 /
  **−3.66e-16**.
- §2.3: $G(T) = 3.38$ at all three probes, a decade either side of $z_{\rm eq}$.
- §3.1: the displaced-guess ladder, all six rows, both columns, and the evaluation counts.
- §3.2: the failure boundary, all six rows — **with the one textual change of R3**.
- §4.1: both ratios strictly monotone over their probe ranges.

**No reconciliation of the numbers is needed.** The audit's §2 and §3 may be read as written.

## 3. R2 — line numbers

`f023eb8` added five net lines to `TemperatureRepresentation`'s docstring, which sits above all
three solves. Re-anchored at `f023eb8`:

| Audit cites | Now | Symbol |
|---|---|---|
| `:578` | **`:583`** (`def` at **`:539`**) | `_solve_T_z`'s `root_scalar` |
| `:864` | **`:869`** (`def` at **`:828`**) | `_temperature_crossing_log1pz`'s `root_scalar` |
| `:1008` | **`:1013`** (`def` at **`:999`**) | `_find_rho_equality`'s `root_scalar` — **the subject** |
| `:496`, `:499` | **`:501`**, **`:504`** | the two call sites in `__init__` |
| `:508-509` | **`:516-517`** | the two `print` statements |
| `:1010` | **`:1015`** | the `if not root.converged` guard |

Also `_bisect_temperature_crossing_log1pz` at **`:759`** — the production locator, which the audit
does not count among the three `root_scalar` sites because it is a hand-rolled bisection, not a
`scipy` solve. **It is in the file and it is sound**; nothing in this campaign touches it.

## 4. R3 — §6 observation 1 is closed

`f023eb8` ("Make the out-of-bounds spline error name its own class") replaced the literal
`GkSource.function:` prefix with `type(self).__name__` at **six** sites across **three** classes —
`ZSplineWrapper` and `GkWKBSplineWrapper` in `ComputeTargets/spline_wrappers.py`, and
`TemperatureRepresentation` in `LambdaCDM_GenericEOS.py` — and amended the docstring sentence that
recorded the duplication as deliberate. `AUDIT.md` §6 observation 1 was struck through and
corrected in the same commit.

The one consequence for the audit's text: §3.2's last two rows now read

```
RuntimeError: TemperatureRepresentation: evaluated T(z) out of bounds @ z=-0.25719
```

rather than `GkSource. ...`. The audit's §3.2 item 2 — *"a `LambdaCDM_GenericEOS` constructor that
dies this way reports a temperature-spline bounds error with no mention of equality-finding"* — is
**still true and still the point**: the message now names the right class, but it still does not
name `_find_rho_equality`, the species pair, or the range searched, and
`_find_rho_equality`'s own guard at `:1015` is still dead on every path that actually fails.

## 5. R4 — the blast radius is not two banner lines

**This is the substantive finding of the reconciliation, and it changes the campaign's shape.**

`AUDIT.md` §2.1 establishes, correctly, that a repository-wide grep for `_find_rho_equality`
returns three hits, that both results land in local variables printed at `:516-517` and discarded,
and that nothing downstream reads them. Every one of those statements is true at `f023eb8`.

What does not follow is that the *quantity* is a diagnostic. It is not. `main.py` computes the same
two redshifts from the same closed forms and feeds them into the production source grid:

```
main.py:506   def cosmology_feature_redshifts(cosmology, z_end, z_init)
main.py:549       feature_z.append(float(omega_m / omega_r - 1.0))
main.py:551       feature_z.append(float(pow(omega_cc / omega_m, 1.0 / 3.0) - 1.0))
main.py:903   break_z, feature_z = cosmology_feature_redshifts(model_cosmology, zend, z_init_grid)
main.py:928       feature_z=feature_z,
              -> CosmologyConcepts/wavenumber.py:287 build_z_sample(..., feature_z=())
              -> CosmologyConcepts/wavenumber.py:350   forced into the grid
              -> Datastore/SQL/ObjectFactories/BackgroundModel.py:225  source_grid_digest = z_sample.digest()
              -> :182, :251, :273   a BackgroundModel lookup-key column, filtered on
```

So on any cosmology that declares break points — which today means `QCD_Cosmology`, the path being
gated on exactly that (`main.py:530-537`) — **the two equality redshifts are production sample
locations, and they are inside a datastore identity.** They arrive there through a *duplicated
closed form*, not through `_find_rho_equality`, which is precisely why the grep the audit ran
finds nothing and why the audit's conclusion is what it is.

**Consequences for the campaign:**

1. The audit's §5 — *"fixing this changes no computed quantity in the pipeline"* — **remains
   true**, because the fix is to `_find_rho_equality` and nothing reads it. §5 may be kept as the
   campaign's framing.
2. But the obvious tidy that §6 observation 2 gestures at — have the method derive its own guess,
   expose the result, and have `main.py` import it instead of recomputing — is **not** a tidy. If
   it moved either redshift by one ulp it would move a grid sample, hence the content digest, hence
   the `BackgroundModel` lookup key, and invalidate every stored object of the eight types
   `qcd-background-audit` log 11 §5 priced. That is a decision for the user, not a refactor to
   make in passing, and this campaign's README §7 **D2** puts it to them.
3. The campaign therefore needs a prompt whose whole job is to *establish and document* this,
   so that the next reader of `AUDIT.md` §2.1 does not act on "two banner lines". That is
   **prompt 03**.

## 6. R5 — the closed form exists three times

| Site | Form | What it feeds |
|---|---|---|
| `LambdaCDM_GenericEOS.py:501-506` | closed form, as `init_z` to `_find_rho_equality` | two `print`s at `:516-517` |
| `LambdaCDM/LambdaCDM.py:73-74` | closed form, no solve at all | two `print`s at `:81-82` |
| `main.py:549-551` | closed form, inline | `feature_z` → the production source grid → a datastore key (R4) |

Three copies of $1+z_{\rm eq} = \Omega_m/\Omega_r$ and $1+z_\Lambda = (\Omega_\Lambda/\Omega_m)^{1/3}$,
in three packages, one of which is load-bearing. `main.py:522-528` documents the duplication and
says why prompt 11 of `qcd-background-audit` made it — *"recomputed here … rather than imported
from the model, which computes them in its constructor and discards them … and which prompt 11 may
not modify"* — so it was a deliberate, scoped choice, not an oversight. It is still three copies,
and nothing scores them against each other.

## 7. R6 — `main.py:526`'s agreement figure is stale by three orders

`main.py:525-527` says:

> The closed form agrees with `LambdaCDM_GenericEOS`'s own root solve to 4e-13 relative in z on
> `QCD_Cosmology` at production parameters, which is far below a grid interval.

Measured at `f023eb8`, `QCD_Cosmology(max_z=1e12)` at Planck2018, against an independent `brentq`
at `xtol=1e-300, rtol=8.9e-16`:

| Pair | closed form | `brentq` reference | relative |
|---|---|---|---|
| matter = radiation | `3406.668974249948` | `3406.668974249951` | **−9.34e-16** |
| matter = $\Lambda$ | `0.3034230329964074` | `0.3034230329964075` | **−3.66e-16** |

Three orders tighter than the docstring claims. The claim is a safe over-estimate and is used only
to argue "far below a grid interval", so **nothing is wrong today**; but it is an unverified figure
in a docstring that justifies a production grid choice, which is the class of thing this project
does not leave standing. Prompt 03 re-takes it and corrects the sentence.

## 8. R7 — `:1008` has no row in the project-wide index

`docs/OPEN_ISSUES.md` (68 open at `f023eb8`) carries **no entry** for this site. Its only record
anywhere is `prompts/tolerance-convergence/IMPLEMENTATION_STATE.md` §3, under *"Recorded by the
rebase, **not owned here**"*:

> `LambdaCDM_GenericEOS.py:1008` — a second `root_scalar(xtol=1e-6, rtol=1e-4)`, on `match_rho`,
> with no provenance in any campaign record. Prompt 02 records what it is and what it feeds; if it
> turns out to matter, that is an issue for whoever owns that file and not a repair to make in
> passing.

That bullet is not a §3 issue with a tag, so `CLAUDE.md`'s maintenance rule never fired and the
index never learned about it. **The audit is the answer to that conditional** and this campaign is
"whoever owns that file". The planning commit therefore opens the issue properly — a `§1.8`
subsection in the index for this board — and the campaign closes it.

## 9. R8 — scope, in and out

### 9.1 Out, and the audit is right about all four

- **`_solve_T_z` (`:583`) and `_temperature_crossing_log1pz` (`:869`).** Audited by
  `qcd-background-audit` prompts 04, 06 and 07; both bracketed; both at the representable floor;
  both carrying the comment that chose the value. **Their tolerances are not re-opened here.**
  (Prompt 04 *moves* `_temperature_crossing_log1pz` without changing a character of its solve —
  §9.2.)
- **`_bisect_temperature_crossing_log1pz` (`:759`).** The production locator. Not a `scipy` solve;
  correct by construction; untouched.
- **`find_phase_extremum`, `LiouvilleGreen/integration_tools.py:92`.** Confirmed still
  `root_scalar(xtol=1e-6, rtol=1e-4)` at `f023eb8`. It is `[11-stop-point-root-tolerance]`, owned
  by the hand-over campaign (`docs/OPEN_ISSUES.md` §1.1), and it sets a computed quantity with a
  datastore regeneration attached. **This campaign must not absorb it**, and README §0.5 says so.
- **`ComputeTargets/QuadSourceIntegral.py:1550`'s stale `DEFAULT_QUADRATURE_ATOL` comment.**
  Already recorded on the `tolerance-convergence` board; that campaign's README §0.4 puts the file
  out of bounds and so does this one.

### 9.2 In, and the index says they are waiting for exactly this campaign

Five rows of `docs/OPEN_ISSUES.md` §1.7 name, as their next step, "whichever prompt next has these
files in scope". This campaign is the first to have them. Each is adopted with a dated
`**Assigned**` line on the `qcd-background-audit` board and its index row moved into §1.8:

| Issue | Index next step | Prompt |
|---|---|---|
| `[08-temperature-crossing-solver-is-test-only]` | *"move it into `CosmologyModels/tests/T_z_reference.py`, or delete it and have that one test bisect, for whichever prompt next has both files in scope"* | **04** |
| `[07-t-photon-range-logic-recomputes-its-bounds]` | *"Hoisting them into `__init__` is numerically null. Out of scope for prompt 06"* | **05** |
| `[06-t-photon-call-cost-needs-a-quiet-machine]` | *"hoist the two loop-invariant `_outward` calls (`[07-…]`) … which lands at ~2.49 µs; if that does not clear it, the row itself is what to put to the user"* | **05** |
| `[09-audit-script-section-5-prose-counts-the-wrong-set]` | *"in whichever prompt next has `docs/qcd-background-audit/` in scope"* | **05** (which re-runs that script for its §6 cost row) |
| `[03-qcd-inventory-does-not-report-the-representation]` | *"One line in `inventory()`; out of scope for prompt 03 … and out of scope for prompt 09"* | **07** (gated, workstream D) |

**Nothing else is adopted.** In particular `[00-eos-branch-joins-do-not-match]` is a question for
the equation of state's authors (`qcd-background-audit` README §7 D6) and stays where it is;
`[04-unsplit-tk-run-now-meets-the-criterion]` and the two prompt-13/15 rows are about the source
grid and the numeric split, which this campaign does not touch.

### 9.3 One thing found while reconciling, not adopted

`CosmologyModels/tests/test_wPerturbations.py:34-41` describes the T(z) inversion as *"tabulated on
a 500-point spline in log(1+z)"* and quotes *"~1.3e-9 relative with max_z = 1e4 and ~4e-7 with the
default max_z = 1e20"*, setting `AGREEMENT_RTOL = 1.0e-8`. Since `qcd-background-audit` prompts 05
and 06 the representation is a **segmented entropy factor over 3,000 nodes at order 5**
(`DEFAULT_T_Z_SPLINE_SAMPLES = 3000`, `DEFAULT_T_Z_SPLINE_ORDER = 5`), so both the description and
both figures predate the tree by two replacements. Same class as
`[10-transfer-remedial-tolerance-comments-stale]`. Every assertion still passes.
**Prompt 01 measures it and opens it as an issue; prompt 08 (workstream D, gated) is where it gets
fixed if the user wants it fixed.**

## 10. Baseline

Taken at `f023eb8`, on the tree this plan was written against:

| Suite | Count |
|---|---|
| `CosmologyModels/tests` | **30**, OK, 0.62 s |
| `ComputeTargets/tests` | **447** (recorded at `acd5b8e` by `be21f5c`; re-run and confirmed at `f023eb8`) |
| `LiouvilleGreen/tests` | 148 full set / 143 fast set (`qcd-background-audit` log 15) — **not re-run**; no prompt here touches `LiouvilleGreen/` |

Every prompt records all counts it runs, before and after. A count that falls is a stop.
