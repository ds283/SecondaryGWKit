# Implementation state — the background solver robustness campaign

**Campaign:** [`README.md`](README.md) · **Audit:** [`AUDIT.md`](AUDIT.md) ·
**Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Logs:** [`logs/`](logs/) ·
**Orchestrator:** [`orchestrator/`](orchestrator/)
**Planned:** 2026-09-16 at `f023eb8` (`tolerance-convergence`)
**Baseline commit:** `f023eb8` — suites green and re-run at planning time:
`CosmologyModels` **30**, `ComputeTargets` **447**
**Target branch:** `background-solver-robustness`, cut from `f023eb8` (README §4)
**Last updated:** 2026-09-16 · **Status: workstreams A, B and E complete — prompts 01, 02, 03, 04
and 09 done. README §7 D2 is implemented: the model is authoritative for its own equality
redshifts, `main.py` computes nothing, and the `QCD_Cosmology` production source-grid digest has
moved `a2c32f67` → **`4849552b`** on exactly one sample, on purpose (§6 note 15). Workstream C
(05, 06) is next.**

> **The impact is zero change to any computed quantity, and that is the point.** `AUDIT.md` §5 and
> README §0.2 are the campaign's framing: the two redshifts `_find_rho_equality` produces are
> printed with `:.4g` and discarded, they are already right to the double-precision floor, and no
> prompt in workstreams A–C moves a stored number or implies a regeneration. What the campaign buys
> is a solve that is correct **by construction** instead of because its caller hands it the answer,
> a reachable convergence guard, a comprehensible failure message, and a provenance entry
> `docs/TOLERANCE-PROVENANCE.md` can state.
>
> **The reconciliation changed one thing about the audit's scope.** `AUDIT.md` §2.1's *"the blast
> radius of this solve is two banner lines"* is true of the **solve** and false of the
> **quantity**: `main.py:549-551` recomputes both equality redshifts from the same closed forms and
> forces them into the production source grid, whose content digest is a `BackgroundModel`
> lookup-key column. The tidy that audit §6 observation 2 gestures at would therefore invalidate
> every stored object of eight types. That is **prompt 03** and README §7 **D2**, and it is the
> single most important thing this campaign writes down.
>
> **Five issues are adopted from `docs/OPEN_ISSUES.md` §1.7**, each of which names as its next step
> "whichever prompt next has these files in scope". This is the first campaign that does.
> (`RECONCILIATION.md` §9.2.)

---

## 1. The board

| # | Prompt | Workstream | Covers | Model | Status | Commit | Log |
|---|---|---|---|---|---|---|---|
| 01 | [Equality-solve characterisation test](01-equality-solve-characterisation.md) | A | README §2 (a), (d) | Opus | ⚠️ | *"Characterise the equality solve before bracketing it"* (SHA not embedded, per the campaign convention) | [`logs/01-equality-solve-characterisation.md`](logs/01-equality-solve-characterisation.md) |
| 02 | [Bracket the equality solve](02-bracket-the-equality-solve.md) | A | README §2 (b), (c), (e); audit §4.2 | Opus | ⚠️ | *"Bracket the equality solve and make its guard reachable"* (SHA not embedded, per the campaign convention) | [`logs/02-bracket-the-equality-solve.md`](logs/02-bracket-the-equality-solve.md) |
| 03 | [What the equality redshifts feed](03-equality-redshift-consumers.md) | B | README §2 (f); §7 D2 | Opus | ⚠️ | *"Write down what the equality redshifts actually feed"* (SHA not embedded, per the campaign convention) | [`logs/03-equality-redshift-consumers.md`](logs/03-equality-redshift-consumers.md) |
| 04 | [Relocate the crossing probe](04-relocate-the-crossing-probe.md) | B | README §2 (g) | Sonnet | ✅ | *"Relocate the crossing probe to test machinery"* (SHA not embedded, per the campaign convention) | [`logs/04-relocate-the-crossing-probe.md`](logs/04-relocate-the-crossing-probe.md) |
| 05 | [Hoist the range logic](05-hoist-the-range-logic.md) | C | README §2 (h); §7 D4 | Opus | ⬜ | | |
| 06 | [Provenance and close-out](06-provenance-and-close-out.md) | C | README §2 (i); §0.4 | Opus | ⬜ | | |
| 07 | [Report the representation in the inventory](07-inventory-representation.md) | **D — gated** | — | Sonnet | 🔒 | | |
| 08 | [Refresh the agreement threshold](08-refresh-agreement-threshold.md) | **D — gated** | — | Sonnet | 🔒 | | |
| 09 | [Make the model authoritative](09-make-the-model-authoritative.md) | **E** | README §2 (l); §7 D2 as decided | Opus | ⚠️ | *"Make the cosmology authoritative for its equality redshifts"* (SHA not embedded, per the campaign convention) | [`logs/09-make-the-model-authoritative.md`](logs/09-make-the-model-authoritative.md) |

Status key: ⬜ not started · 🔄 in flight · ✅ complete · ⚠️ complete with a recorded caveat ·
❌ blocked · 🔒 gated on README §7 D3.

**Ordering.** A (01 → 02) → B (03 → 04) → **E (09)** → C (05 → 06). 03 depends on 02's corrected
solve; 04 and 05 touch the same file as 02 and are sequenced behind it for that reason alone.
**09 is placed before C, not after it**, for two reasons: it moves the source-grid digest, and
prompt 06 is the campaign close-out, which would be stale the moment 09 landed behind it; and 05
touches `LambdaCDM_GenericEOS.py`, which 09 also edits, so 09 going first keeps 05's bit-identity
acceptance clean. 06 depends on everything. 07 and 08 are independent of the chain and may run at
any point after the gate opens, except that 08 needs 01's log.

---

## 2. Item-level state

One row per thing the campaign claims to establish. A row whose evidence is one model or one pair
is **not** ✅ — both equality pairs on both models is the standard, because the two roots are four
decades apart and nothing about one of them predicts the other.

| Item | Kind | Statement | Prompt | Status |
|---|---|---|---|---|
| (a) | **MACHINERY** | The two equality redshifts, on `QCD_Cosmology` and the pure-radiation stand-in, both pairs, scored against an independent `brentq` at Brent's own $4\varepsilon$ floor — ~~to ≤ 2 ulp~~ **to ≤ 4 ulp**, in a test that needs no Ray and no datastore | 01 | ⚠️ |
| (b) | **FIX** | `_find_rho_equality` is bracketed and Brent, at `xtol=1e-300`, ~~`rtol=1e-14`~~ **`rtol=8.9e-16`** (the user's amended README §7 D1, 2026-09-16; log 02 D1), with the comment that chose the values at the point of use and to the standard of `:569-583` | 02 | ⚠️ |
| (c) | **FIX** | A failure to bracket raises `_find_rho_equality`'s own `RuntimeError` naming the species pair and the range searched — not a `ValueError` and not a `TemperatureRepresentation` bounds error from two frames down | 02 | ✅ |
| (d) | **MEASUREMENT** | The monotonicity the bracket rests on is a standing test, not a paragraph in an audit: $\rho_m/\rho_r$ strictly decreasing on $z\in[33,3.4\times10^5]$, $\rho_m/\rho_\Lambda$ strictly increasing on $z\in[0,10]$ | 01 | ✅ |
| (e) | **DISCIPLINE** | Every behaviour-change assertion is shown **failing on `HEAD~1`**, with the output quoted in the log | 02 | ✅ |
| (f) | **MEASUREMENT** | What the equality redshifts actually feed, written down: three closed-form sites scored against each other and against the corrected solve on three models — ~~`RadiationModel`~~ **the pure-radiation stand-in, `RadiationModel` exposing no $\Omega$s at all (log 03, D1)** — the chain from `feature_z` to the `BackgroundModel` lookup key, and `main.py:526`'s stale 4e-13 corrected | 03 | ✅ |
| (g) | **HYGIENE** | `_temperature_crossing_log1pz` is test machinery in the test tree, with its docstring and its `xtol=1e-15, rtol=1e-15` carried across unchanged | 04 | ✅ |
| (h) | **MEASUREMENT** | The four loop-invariant `_outward` bounds hoisted in all three classes, **bit-identical** returns and character-identical messages demonstrated, and the `T_photon` cost row closed at ≤ 2.5 µs or escalated to the user | 05 | ⬜ |
| (l) | **FIX** | The model is authoritative for its own equality redshifts: `BaseCosmology` declares them, `LambdaCDM` answers with the closed form (exact for it), `LambdaCDM_GenericEOS` answers with the solve its constructor already runs, `main.py` computes nothing and has **no fallback** — QCD digest `a2c32f67` → `4849552b` on exactly one moved sample (index 1540, +7 ulp), `LambdaCDM` unmoved — **and the QCD break-point-only grid `303f9ce7` → `81c6e682`, which the prompt did not name and which carries the same one sample (log 09, D1)** | 09 | ✅ |
| (i) | **PROVENANCE** | An entry for **all three** `root_scalar` sites in `LambdaCDM_GenericEOS.py` — value, method, what it sets, call count, choosing measurement **and its commit**, competing floor, cost, citation — in the shape `docs/TOLERANCE-PROVENANCE.md` will want | 06 | ⬜ |

---

## 3. Active and unresolved issues

Issues opened here must be added to [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) **in the
same commit** (`CLAUDE.md`), with the count corrected.

Opened by **prompt 09** (2026-09-16):

- **[09-retire-tag-test-docstring-cites-a-superseded-digest]** *(prompt 09; no prompt assigned)* —
  `ComputeTargets/tests/test_retire_samples_per_decade_tag.py:29-32`'s module docstring says that
  `test_source_grid.py::TestTheConstructionVersionNamesThisAlgorithm` *"pins the production grids
  to 1,996 samples / digest `a2c32f67` (QCD) and 1,778 / `60a3205a` (LambdaCDM)"*. Since prompt 09
  the QCD digest is **`4849552b`**; the `LambdaCDM` half is still right, and so is the sample
  count.
  **Impact:** none mechanically. That file asserts only `SOURCE_GRID_CONSTRUCTION_VERSION` and
  `T_Z_REPRESENTATION_VERSION` — the digest appears in prose explaining *why* it does not need to
  assert the grid — and the whole `ComputeTargets` suite is green at 449. It is a stale figure in
  a docstring that argues a prompt moved no grid, which is the class of thing this project does
  not leave standing. The file is not in prompt 09's permitted list (README §5 rule 5).
  **Next step:** one word, in whichever prompt next has
  `ComputeTargets/tests/test_retire_samples_per_decade_tag.py` in scope.

Opened by **prompt 03** (2026-09-16) and **resolved by prompt 09** — moved to §4:
`[03-main-py-cites-a-stale-line-for-the-equality-solve]`.

Opened by **prompt 02** (2026-09-16):

- **[02-bracketed-reference-is-not-the-exact-root]** *(prompt 02; measured; no prompt assigned)* —
  README §3.1 makes the bracketed `brentq` reference "the anchor every measurement is scored
  against" and item (a) is written as agreement with it. On the one pair where an exact oracle
  exists, **the anchor is the less accurate of the two things being compared.** `match_rho` for
  matter = $\Lambda$ is exactly $\rho_{m0}(1+z)^3-\rho_\Lambda$, with no temperature dependence, so
  the root follows in closed form from the model's own float constants. Evaluated at 60 decimal
  digits from `QCD_Cosmology`'s `rho_m0 = 6.894224419906764e+105` and
  `rho_cc = 1.52665741011693e+106`, the exact root is `0.303423032996407410561312801228`. The
  closed form, and the solve at both `7fdc49b` and this commit, give `0.30342303299640738` —
  **−0.506 ulp, the nearest double**. `bracketed_reference` gives `0.30342303299640749` —
  **+1.494 ulp, the second-nearest, on the wrong side**. So `LAMBDA_CLOSED_FORM_ULP = 2` is
  measuring the reference's own error, and the "−2.0 ulp" that prompt 01 and prompt 02 both report
  for that pair is the reference being wrong, not the solve.
  **Impact:** none on anything asserted — every test passes, and the direction of the finding is
  that the shipped answer is *better* than what scores it. It matters because three more prompts
  score figures in ulp against this anchor. **Next step:** prompt 03 must read this before scoring
  the three closed-form sites; whether README §3.1 and item (a) should be re-worded around an
  exact oracle for the $\Lambda$ pair is a planning question and a decision for the user, since
  changing the reference would change what prompt 01's four tests assert.

Opened by the **2026-09-16 planning commit** and **resolved by prompt 09** — moved to §4:
`[00-equality-redshift-closed-form-is-duplicated-three-times]`. Its measurement, the three priced
options and the user's D2 decision are carried there verbatim.

Also opened by the planning commit, found while reconciling (`RECONCILIATION.md` §9.3) and
**measured by prompt 01**, which has the file open for another reason:

- **[01-agreement-threshold-comment-predates-the-representation]** *(planning; **measured by prompt
  01** 2026-09-16; assigned to prompt 08, gated)* — `CosmologyModels/tests/test_wPerturbations.py:34-41`
  describes the $T(z)$ inversion as a "500-point spline" and quotes ~1.3e-9 and ~4e-7 to justify
  `AGREEMENT_RTOL = 1.0e-8`. Since `qcd-background-audit` prompts 05 and 06 the representation is a
  segmented entropy factor at **3,000 nodes of order 5** and what is tabulated is not $T$.
  **Measured at `3e820eb`** (log 01 §4.3), on exactly what `test_agrees_with_LambdaCDM` compares —
  the `PureRadiationEOS` stand-in against `LambdaCDM.wPerturbations` over that test's own probe set
  $z\in\{0,0.5,1,2,10,10^3\}$: the worst relative disagreement is **8.8818e-16** at
  `max_z = 1e4` (claimed ~1.3e-9) and **6.6613e-16** at `max_z = 1e20` (claimed ~4e-7) — **seven to
  nine orders tighter, and the `max_z` dependence has gone entirely**. On a constant-$g_*$ equation
  of state the tabulated entropy factor is exactly constant, so $T(z) = T_{\rm CMB}(1+z)$ is
  recovered to rounding at any `max_z`; the old figures describe a representation that no longer
  exists. **Impact:** none — every assertion passes and `AGREEMENT_RTOL` is now a ceiling eight
  orders above what the code delivers. Same class as
  `[10-transfer-remedial-tolerance-comments-stale]`. **Next step:** prompt 08 rewrites the comment
  and may retighten the constant, if README §7 D3 opens workstream D. Prompt 01 was forbidden to
  edit that file and did not.

### 3.1 Adopted from other boards

Each is that board's issue; each names "whichever prompt next has these files in scope" as its next
step, and this is that campaign (`RECONCILIATION.md` §9.2). Each carries an `**Assigned
(2026-09-16):**` line on its owning board, and each closes on **that** board's §4 with its row
deleted from `docs/OPEN_ISSUES.md`.

| Issue | Owning board | Assigned to | Why here |
|---|---|---|---|
| ~~`[08-temperature-crossing-solver-is-test-only]`~~ | qcd-background-audit | prompt **04** | **Closed by prompt 04, 2026-09-16** — moved to `CosmologyModels/tests/T_z_reference.temperature_crossing_log1pz`; resolved entry on the `qcd-background-audit` board's §4, row deleted from `docs/OPEN_ISSUES.md` |
| `[07-t-photon-range-logic-recomputes-its-bounds]` | qcd-background-audit | prompt **05** | Out of scope for that campaign's prompt 06, whose prompt did not cover prompt 05's range logic. Numerically null; 0.056 µs × 2 of a ~2.5 µs call |
| `[06-t-photon-call-cost-needs-a-quiet-machine]` | qcd-background-audit | prompt **05** | Its own next step **is** the hoist above. Confirmed miss at 2.596 µs against a 2.5 µs target; predicted landing ~2.49 µs. If it does not clear, README §7 **D4** puts the row to the user |
| `[09-audit-script-section-5-prose-counts-the-wrong-set]` | qcd-background-audit | prompt **05** | Needs `docs/qcd-background-audit/` in scope; prompt 05 re-runs `measure_T_z_representation.py` for its §6 cost row and so has it |
| `[03-qcd-inventory-does-not-report-the-representation]` | qcd-background-audit | prompt **07** (gated) | Orphaned by two prompts of that campaign on scope. Not this campaign's subject either, which is why it is behind README §7 **D3** |

### 3.2 Recorded, **not owned here**, and not scheduled (README §0.5)

- `[11-stop-point-root-tolerance]` — `find_phase_extremum`,
  `LiouvilleGreen/integration_tools.py:92`, confirmed still `root_scalar(xtol=1e-6, rtol=1e-4)` at
  `f023eb8`. It **looks identical to this campaign's subject and is not the same problem**: it is
  bracketed, it is on the production path, and it sets a computed quantity — the numeric→WKB
  hand-over — with a datastore regeneration attached. Owned by the hand-over campaign
  (`docs/OPEN_ISSUES.md` §1.1). `AUDIT.md` §4.3 forbids absorbing it. **No prompt here may touch
  it.**
- `ComputeTargets/QuadSourceIntegral.py:1550`'s stale `DEFAULT_QUADRATURE_ATOL = 1e-25` comment
  (the constant has been `1e-32` since `source-remediation` prompt 12). Already on the
  `tolerance-convergence` board; that campaign's README §0.4 puts the file out of bounds and so
  does this one. Repeated here only so it is not re-discovered as new.
- `_solve_T_z:583` and `_temperature_crossing_log1pz:869`'s **tolerances**. Audited, bracketed, at
  the representable floor, commented. Prompt 04 *moves* the second without changing a character of
  its solve; nothing else in this campaign touches either.

---

## 4. Resolved issues

- **[00-equality-redshift-closed-form-is-duplicated-three-times]** — **RESOLVED by prompt 09,
  2026-09-16**, by implementing README §7 **D2** as the user decided it.

  **What the issue was.** $1+z_{\rm eq}=\Omega_m/\Omega_r$ and
  $1+z_\Lambda=(\Omega_\Lambda/\Omega_m)^{1/3}$ were computed at three sites in three packages:
  `LambdaCDM_GenericEOS.py:502`/`:507` (a solver guess, feeding two prints),
  `LambdaCDM/LambdaCDM.py:73-74` (two prints) and `main.py:553`/`:555`. **The third was
  load-bearing:** it became `feature_z`, was forced into the production source grid by
  `CosmologyConcepts/wavenumber.py:350`, and the grid's content digest is a `BackgroundModel`
  lookup-key column (`Datastore/SQL/ObjectFactories/BackgroundModel.py:182`, `:225`, `:251`). On
  `QCD_Cosmology` the two equality redshifts were therefore **production sample locations inside a
  datastore identity**, reached by a duplicated closed form rather than by `_find_rho_equality` —
  which is why `AUDIT.md` §2.1's grep found nothing and concluded the blast radius was two banner
  lines. `main.py:522-528` recorded the duplication as `qcd-background-audit` prompt 11's
  deliberate, scoped choice.

  **What prompt 03 measured, and the three options it priced** (log 03 §3 item 3). The three sites
  were the **same float** on `QCD_Cosmology`, the pure-radiation stand-in and
  `LambdaCDM(Planck2018)`, on both pairs, although `LambdaCDM.py` reaches `math.pow` through
  `from math import sqrt, pow` where the other two get the builtin; the closed form and the
  corrected solve differed by **7 ulp** (QCD, matter–radiation), **1 ulp** (stand-in) and **not at
  all** at matter–$\Lambda$ on either model. **(i)** leave all three — cost zero, guarded by a new
  test; **(ii)** unify on the closed form — safe only while the spellings agree bit for bit, which
  is measured but not guaranteed by the language, and it makes `main.py` import a cosmology model;
  **(iii)** unify on the solve — measured to take the `QCD_Cosmology` production grid digest from
  `a2c32f67` to `4849552b` on a 7-ulp move in one sample of 1,996. **The campaign's recommendation
  was (i).**

  **README §7 D2 — decided by the user, 2026-09-16: option (iii).** Against the campaign's
  recommendation, and the campaign's recommendation was wrong. $\Omega_r$ is a **present-day**
  density parameter, so $1+z_{\rm eq}=\Omega_m/\Omega_r$ is exact only while
  $\rho_r\propto(1+z)^4$ holds from today back to equality. On `QCD_Cosmology` it does, to 7 ulp,
  for one reason — all the $g_*(T)$ structure sits at $z\sim10^{12}$, twelve orders above
  $z_{\rm eq}$. A cosmology with entropy injection below $z_{\rm eq}$, a decaying species, or extra
  relativistic species appearing late breaks it **by far more than ulps, and silently**. The
  accuracy at that site was a property of where the QCD transition happens to sit, not of the code
  — which is, in terms, the argument **prompt 02 already wrote into the solve** when it refused to
  short-circuit on a guess that is already the root.

  **The mechanism is not D2 option (iii)'s wording, and prompt 09 shipped the mechanism.**
  `main.py` does not import `_find_rho_equality`. The distinction that matters is not *has a
  solver* but *is the closed form valid for this model*, and only the model knows.
  `CosmologyModels/base.py` now declares two abstract properties,
  **`z_matter_radiation_equality`** and **`z_matter_lambda_equality`**; `LambdaCDM` implements them
  with the closed form, which for a model with no equation of state is the exact root and not an
  approximation to it, and its `:73-74` banner is now a *use* of them; `LambdaCDM_GenericEOS`
  implements them from the two `_find_rho_equality` calls its constructor already ran and used to
  discard; and `cosmology_feature_redshifts` reads them and computes nothing. **The contract, in
  the user's words:** it is the cosmology's business to supply the correct value, and if a subclass
  chooses to do something incoherent, that is its problem. `BaseCosmology` declares the obligation;
  it does not police it.

  **All three of the decision's consequences were carried.** The two **initial guesses** at
  `:502`/`:507` are **character-identical** and stay closed forms; there is **no fallback** in
  `main.py` — a cosmology that declares break points but cannot answer raises a `RuntimeError`
  naming it, its class, which of the two was missing and the attribute it was looked for under,
  and `_BrokenCosmology` in `test_source_grid.py` (which *has* the $\Omega$s, so a fallback would
  have succeeded on it) asserts that; and prompt 03's guard was **reworked, not deleted**, as
  `test_the_remaining_closed_form_sites_agree`, which now also asserts that `LambdaCDM`'s property
  returns exactly the expression the module transcribes (log 09, D2 and D3).

  **What moved, on purpose** (README §0.2 is scoped to workstreams A–C; this is E, §6 note 15).
  `QCD_Cosmology` production grid digest **`a2c32f67` → `4849552b`**, 1,996 samples unchanged, on
  **exactly one** sample — index **1540**, the matter–radiation feature, **+7.0 ulp**,
  `3406.668974249948` → `3406.668974249951`. matter–$\Lambda$ is the same double from either route
  (`0x1.36b4870e4a718p-2`). `LambdaCDM(Planck2018)` is **`60a3205a`** at 1,778, **unmoved** — its
  closed form is its answer and it never reaches the feature path, declaring no break points. The
  QCD break-point-only grid (no spacing profile) moved with it, `303f9ce7` → **`81c6e682`** at
  1,773, which the prompt did not name (log 09, D1). **There is no regeneration to do** (§6 note
  14). `T_Z_REPRESENTATION_VERSION` **6**; suites `CosmologyModels` 38 → **39**, `ComputeTargets`
  447 → **449**, both OK. The two new behaviour tests were shown **failing on `6f3cd8e`** with the
  output in log 09 §2.

- **[03-main-py-cites-a-stale-line-for-the-equality-solve]** — **RESOLVED by prompt 09,
  2026-09-16**, incidentally and within scope. The citation lived in the sentence of
  `cosmology_feature_redshifts`'s docstring that said the two equality redshifts are *"recomputed
  here … rather than imported from the model, which computes them in its constructor and discards
  them (``LambdaCDM.py:73``, ``LambdaCDM_GenericEOS.py:483``)"*. Prompt 09's §2.5 required that
  whole paragraph to be rewritten — the model no longer discards them and `main.py` no longer
  recomputes anything — so the sentence carrying the wrong line number is gone rather than
  corrected. Neither citation survives, and the replacement paragraph names no line numbers at all,
  which is why this cannot go stale the same way again. Prompt 03 was right to leave it: the fix
  really was one line, and it belonged to the prompt that had the paragraph in scope.

- **[00-main-py-equality-agreement-figure-is-stale]** — **RESOLVED by prompt 03, 2026-09-16.**
  `main.py`'s `cosmology_feature_redshifts` docstring no longer claims the closed form agrees with
  the model's root solve *"to 4e-13 relative in z"*. Re-taken at `921f41c` against **prompt 02's
  corrected solve** rather than against the secant `RECONCILIATION.md` §6 measured:

  | Model | Pair | closed form | corrected solve | relative | ulp |
  |---|---|---|---|---|---|
  | `QCD_Cosmology` | matter = radiation | `3406.668974249948` | `3406.668974249951` | **−9.344e-16** | −7.0 |
  | `QCD_Cosmology` | matter = $\Lambda$ | `0.3034230329964074` | `0.3034230329964074` | **+0.000e+00** | 0.0 |
  | pure-radiation stand-in | matter = radiation | `3403.1059638279453` | `3403.1059638279457` | **−1.336e-16** | −1.0 |
  | pure-radiation stand-in | matter = $\Lambda$ | `0.3034230329964074` | `0.3034230329964074` | **+0.000e+00** | 0.0 |

  The sentence now quotes **−9.3e-16 (7 ulp)** and **the same float**, names `921f41c` as the tree
  they were taken on, and says that 4e-13 was a safe over-estimate — unverified rather than wrong,
  cited only to make the "far below a grid interval" argument that the measured figures make three
  orders more comfortably. The matter–$\Lambda$ figure has gone to *exactly zero* because prompt 02
  moved the solve onto the closed form's float; it is the `brentq` reference that now sits 2 ulp
  away, which is `[02-bracketed-reference-is-not-the-exact-root]`.

  **Zero executable lines of `main.py` are in the diff.** The correction is four lines longer than
  the sentence it replaces, so everything below it shifts by +4 — see standing note 11.

- **[00-equality-solve-is-unbracketed-and-loose]** — **RESOLVED by prompt 02, 2026-09-16.**
  `LambdaCDM_GenericEOS._find_rho_equality` no longer runs an unbracketed secant. It clamps the
  guess into the $T(z)$ representation's tabulated range, expands a bracket about it
  multiplicatively in $1+z$ by $\sqrt2$ (capped at 140 steps, clamped to the representation's own
  bounds at both ends) until the residual changes sign, and calls
  `root_scalar(..., bracket=..., xtol=1e-300, rtol=8.9e-16)`. The `converged` guard is kept and
  now names the species pair and the bracket, and a **failure to bracket** raises the method's own
  `RuntimeError` naming both species, the guess, the clamped guess, both endpoints and both
  residuals. The comment at the point of use carries the value, the competing floor, the measured
  cost and `AUDIT.md` §2.3's finding, to the standard of `:569-583`.

  **The tolerance is `rtol=8.9e-16`, Brent's own $4\varepsilon$ floor, not README §7 D1's
  `rtol=1e-14`** — the user's amended D1 decision of 2026-09-16, taken on prompt 02's measurement
  (log 02, D1): the residual is a cancellation between two densities of order $10^{112}$, so the
  root is a band a few ulp wide and `rtol=1e-14` (75 ulp of slack at $z\sim3.4\times10^3$) stopped
  **7 ulp** from the independent reference and **failed prompt 01's test**. The same decision
  amended prompt 02 §4's first acceptance row from "bit-identical / ≤ 1 ulp" — which is
  unattainable by any solver here — to "≤ 4 ulp against prompt 01's reference".

  **What moved:** the two matter–radiation roots, by **+3 ulp** (`QCD_Cosmology`,
  `3406.6689742499498` → `3406.6689742499511`) and **+1 ulp** (stand-in, `3403.1059638279453` →
  `3403.1059638279457`), **both onto** the independent reference they previously sat below. Both
  matter–$\Lambda$ roots are **bit-identical**. Both printed banner lines are character-identical.
  Cost: **3, 1, 1, 1 → 23, 25, 21, 25** `_rho_fluid` evaluations, twice per model construction;
  `AUDIT.md` §3.1's "+6 to +9" is for tightening the secant and does not survive bracketing.
  Pinned by three new tests in `CosmologyModels/tests/test_rho_equality.py`, all three shown
  failing on `7fdc49b` with the output in the log. Suites `CosmologyModels` 34 → **37**,
  `ComputeTargets` **447** unchanged, `T_Z_REPRESENTATION_VERSION` **6**.

---

## 5. Close-out

Written by prompt 06. Empty until then.

---

## 6. Standing notes

1. **A test that passes both before and after proves nothing** (README §2 (e)). This campaign's
   other stop condition is "nothing moved", which is also what a test that tests nothing reports.
   Where a prompt says "show it fails on `HEAD~1`", the orchestrator **runs that check itself**.
2. **`T_Z_REPRESENTATION_VERSION` is 6 at every commit of this campaign.** A prompt that changes it
   has left its scope.
3. **`ComputeTargets` is 447 at every commit.** Nothing in that package reads `_find_rho_equality`;
   a move there means something else changed. **Amended by prompt 09: it is 449 from that commit
   on.** 09 added the two `test_source_grid.py` tests that assert `feature_z` is the model's own
   answer and that a cosmology which cannot answer raises (README §2 (l)). The rule is unchanged in
   substance — a count that *falls* is still a stop, and a rise must be a test a prompt says it
   added.
4. **A figure without the tree it was taken on is not a measurement** (README §5 rule 9).
5. **The two printed banner lines at `LambdaCDM_GenericEOS.py:516-517` are character-identical
   throughout.** They are the user-visible surface of the whole campaign, and the campaign's claim
   is that they do not change. **Still true at prompt 09, and re-anchored by it to `:527-530`:**
   09 made the two values instance attributes rather than locals, so the `print` calls now read
   `self._z_matter_radiation_equality` and `self._z_matter_lambda_equality` and `black` split the
   first over three source lines. **The text they emit is unchanged, and so is `LambdaCDM.py`'s
   pair, now at `:83-84` (was `:81-82`).**
6. **An agent must never assume `HEAD` is its own.** Planning and orchestration commits land on the
   same branch.
7. **"≤ 2 ulp" in README §6 and prompt 01 §5 is arithmetically incompatible with the audit's own
   figures, and the shipped threshold is 4 ulp** (log 01, deviation D1). `AUDIT.md` §2.2's
   **−4.00e-16** at $z = 3406.67$ *is* −3.0 ulp — one ulp there is $1.335\times10^{-16}$ relative —
   and `RECONCILIATION.md` §6's −9.34e-16 for the closed form is 7 ulp. Measured separations from
   the bracketed reference at `3e820eb`: solve **3 / 2 / 1 / 2 ulp**, $\Lambda$ closed form
   **2 / 2**, matter–radiation closed form **7** (QCD) and **1** (stand-in). The test module ships
   `SOLVE_VS_REFERENCE_ULP = 4`, `LAMBDA_CLOSED_FORM_ULP = 2` and
   `MATTER_RADIATION_CLOSED_FORM_ULP = 8`; the fourth ulp is the **reference's own** bracket
   sensitivity, measured, not slack for the solve. **A later prompt converting a relative figure in
   `AUDIT.md` or `RECONCILIATION.md` into ulp must do the arithmetic rather than copy the "2".**
   Prompt 02 did not widen any of the three.
8. **Two of the four equality redshifts moved at prompt 02, and log 02's table is the current
   one.** `QCD_Cosmology` matter = radiation is `3406.6689742499511` (`0x1.a9d5683cafad0p+11`) and
   the stand-in's is `3403.1059638279457` (`0x1.a963640e40f2cp+11`), both **+3 and +1 ulp above**
   log 01's values and both **on** the bracketed reference. The two matter–$\Lambda$ roots did not
   move. **Prompt 03 scores the three closed-form sites against log 02's table, not log 01's** —
   in particular `main.py:526`'s correction, since the QCD matter–radiation closed form now sits
   7 ulp below the solve.
9. **Bit-identity is not a property this solve can have** (log 02, D1). The residual is a
   cancellation between two densities of order $10^{112}$, quantised at ~$7\times10^{100}$ near the
   root, so its sign change spans several floats: on `QCD_Cosmology` at matter–radiation equality
   it is exactly zero one ulp above the closed form, non-zero either side, and changes sign six to
   seven ulp higher; on the matter–$\Lambda$ pair it is exactly zero across five consecutive
   floats. A later prompt must not write "bit-identical" as an acceptance for anything downstream
   of this root without re-taking that measurement.
10. **This machine was not quiet at prompt 02.** Ten orphaned `while :; do :; done` shells from a
    12:55 loaded-machine benchmark were still saturating every core at 14:20; the
    `ComputeTargets` suite took 215 s against log 01's 181 s on the same tree. No pass/fail or
    float is affected. **Prompt 05 must check for them before it measures `T_photon`**, because
    `[06-t-photon-call-cost-needs-a-quiet-machine]` is a five-run mean on a quiet machine and
    README §7 **D4** turns on 4 % (log 02, "Observations not acted on" item 4).
11. **`main.py`'s line numbers moved at prompt 03, by +4 below `:528`.** The corrected docstring
    sentence is four lines longer than the one it replaced. Re-anchored: the two closed-form
    expressions are **`:553`** and **`:555`** (were `:549`, `:551`); the early return is `:545-546`;
    the `getattr` block is `:549-551`; `break_z, feature_z = cosmology_feature_redshifts(...)` is
    **`:907`** (was `:903`); `feature_z=feature_z` is **`:932`** (was `:928`); the corrected
    sentence itself is `:525-531`. `AUDIT.md`, `RECONCILIATION.md` §5 and README §0.3 all cite the
    old numbers and are left as written — they were correct for the trees they were taken on
    (`CLAUDE.md`) — so a prompt following one of those citations must add 4.
    **Superseded by note 16: prompt 09 moved them again, and there is no longer a closed-form
    expression in `main.py` to anchor.**
12. **The production source-grid digests are `a2c32f67` (`QCD_Cosmology`, 1,996 samples) and
    `60a3205a` (`LambdaCDM(Planck2018)`, 1,778)**, measured identical at `7fdc49b`, `921f41c` and
    prompt 03's commit. Any later prompt in workstreams A–C that moves either has left its scope
    (**workstream E is the exception — note 15**).
    Reproduce with `ComputeTargets/tests/test_source_grid._production_grid(cosmology)` and
    `_to_redshift_array(grid.z_values).digest()`; `QCD_Cosmology` must be built at the production
    `max_z = 1e20`, not at the test modules' `1e12`, or the spacing profile leaves the tabulated
    range. **Superseded for `QCD_Cosmology` by note 17: prompt 09 moved it to `4849552b`, on
    purpose. `LambdaCDM`'s `60a3205a` is unchanged and this note still binds there**, as does the
    reproduction recipe, which is unchanged.
13. **The three closed-form sites are one float, and that is now a test.**
    `CosmologyModels/tests/test_rho_equality.test_the_three_closed_form_sites_agree` asserts it on
    three models and both pairs. A prompt that edits any of `main.py:553`/`:555`,
    `LambdaCDM_GenericEOS.py:502`/`:507` or `LambdaCDM.py:73-74` will hear about it there first. **Prompt
    09 reworks it** — after 09, `main.py` has no closed-form site and the test's subject is the two
    that remain; it may not simply be deleted. **Done — superseded by note 18.**
14. **Datastore regeneration is not a cost in this project's current phase** (the user,
    2026-09-16): this is the build phase of a science code, there is no legacy data to curate, and
    what matters is a code that does the right thing. **This changes the calculus of several
    entries written before it.** README §0.2's "no prompt moves a stored number or implies a
    regeneration", and every board entry that prices an option by how many object types it
    invalidates, were weighing a cost the user does not have. **There is no regeneration to do**:
    no prompt here should schedule one, hedge for one, or carry an outstanding action for one. A
    later prompt must not treat "it would require a regeneration" as an argument against doing the
    correct thing.
15. **Workstream E is outside README §0.2.** Prompt 09 moves the `QCD_Cosmology` source-grid digest
    `a2c32f67` → `4849552b` **on purpose**, on exactly one sample. Note 12's "any later prompt that
    moves either has left its scope" is scoped to A–C and does not bind 09; `LambdaCDM`'s
    `60a3205a` is unmoved by 09 and note 12 still binds there. **Done, and it landed on the
    predicted digest** — see notes 17 and 18.
16. **`main.py`'s line numbers moved again at prompt 09, by +29 below `:558`** (note 11 is
    superseded). `cosmology_feature_redshifts` is `:507-586` (was `:507-557`): the rewritten
    docstring paragraph is longer, and the body now raises rather than computing. Re-anchored:
    the early return is **`:563-564`**; the property loop is **`:566-584`**;
    `pre_grid_background_proxy` is **`:589`** (was `:560`);
    `break_z, feature_z = cosmology_feature_redshifts(...)` is **`:936`** (was `:907`);
    `feature_z=feature_z` is **`:961`** (was `:932`). **There is no closed-form expression in
    `main.py` any more**, so note 11's `:553`/`:555` anchors have no successor — a document citing
    them is citing something that was removed, not something that moved. In
    `LambdaCDM_GenericEOS.py` the two `init_z=` guesses are **`:513`** and **`:518`** (were `:502`,
    `:507`) and the two new properties are at **`:549`** and **`:566`**; in `LambdaCDM.py` the two
    closed forms are now inside properties, at **`:121`** and **`:132`** (were `:73`, `:74`).
17. **The production source-grid digests are `4849552b` (`QCD_Cosmology`, 1,996 samples) and
    `60a3205a` (`LambdaCDM(Planck2018)`, 1,778)** from prompt 09's commit. The QCD value was
    `a2c32f67` at `7fdc49b`, `921f41c`, prompt 03's commit and `6f3cd8e`; it moved on **exactly
    one** sample, index **1540**, **+7.0 ulp**, when `feature_z[0]` became the model's own solve.
    The QCD **break-point-only** grid — `build_z_sample` with `break_z` and `feature_z` and no
    spacing profile — moved with it, `303f9ce7` → **`81c6e682`** at 1,773 samples; both literals
    are asserted in `ComputeTargets/tests/test_source_grid.py`. The bare lattice (`0960e169`,
    1,732) and both `LambdaCDM` grids are untouched. Reproduction is note 12's, unchanged.
18. **There are two closed-form sites, not three, and the test is renamed.** `main.py` computes
    neither redshift; `LambdaCDM_GenericEOS.py:513`/`:518` (the solver's initial guesses) and
    `LambdaCDM.py:121`/`:132` (that model's answer) are what remain, and
    `CosmologyModels/tests/test_rho_equality.`**`test_the_remaining_closed_form_sites_agree`**
    asserts they are one float on three models and both pairs — plus, on `LambdaCDM`, that its
    property returns exactly that expression. `main_py_closed_form` and its `CLOSED_FORM_SITES`
    entry are gone. The companion is
    **`test_each_model_answers_for_its_own_equality_redshifts`**, which asserts that
    `LambdaCDM_GenericEOS` answers with its solve and that on `QCD_Cosmology` that is **not** the
    closed form (7 ulp). A prompt editing either site, or either model's properties, hears about it
    in those two tests and in `test_source_grid.py`'s
    `TestTheModelIsAuthoritativeForItsEqualityRedshifts`.
