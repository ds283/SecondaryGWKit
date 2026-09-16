# Implementation state — the background solver robustness campaign

**Campaign:** [`README.md`](README.md) · **Audit:** [`AUDIT.md`](AUDIT.md) ·
**Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Logs:** [`logs/`](logs/) ·
**Orchestrator:** [`orchestrator/`](orchestrator/)
**Planned:** 2026-09-16 at `f023eb8` (`tolerance-convergence`)
**Baseline commit:** `f023eb8` — suites green and re-run at planning time:
`CosmologyModels` **30**, `ComputeTargets` **447**
**Target branch:** `background-solver-robustness`, cut from `f023eb8` (README §4)
**Last updated:** 2026-09-16 · **Status: workstream A complete; workstream B under way — prompts
01, 02 and 03 done; README §7 D2 answered by the user — option (ii), §3.**

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
| 04 | [Relocate the crossing probe](04-relocate-the-crossing-probe.md) | B | README §2 (g) | Sonnet | ⬜ | | |
| 05 | [Hoist the range logic](05-hoist-the-range-logic.md) | C | README §2 (h); §7 D4 | Opus | ⬜ | | |
| 06 | [Provenance and close-out](06-provenance-and-close-out.md) | C | README §2 (i); §0.4 | Opus | ⬜ | | |
| 07 | [Report the representation in the inventory](07-inventory-representation.md) | **D — gated** | — | Sonnet | 🔒 | | |
| 08 | [Refresh the agreement threshold](08-refresh-agreement-threshold.md) | **D — gated** | — | Sonnet | 🔒 | | |

Status key: ⬜ not started · 🔄 in flight · ✅ complete · ⚠️ complete with a recorded caveat ·
❌ blocked · 🔒 gated on README §7 D3.

**Ordering.** A (01 → 02) → B (03 → 04) → C (05 → 06). 03 depends on 02's corrected solve; 04 and
05 touch the same file as 02 and are sequenced behind it for that reason alone; 06 depends on
everything. 07 and 08 are independent of the chain and may run at any point after the gate opens,
except that 08 needs 01's log.

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
| (g) | **HYGIENE** | `_temperature_crossing_log1pz` is test machinery in the test tree, with its docstring and its `xtol=1e-15, rtol=1e-15` carried across unchanged | 04 | ⬜ |
| (h) | **MEASUREMENT** | The four loop-invariant `_outward` bounds hoisted in all three classes, **bit-identical** returns and character-identical messages demonstrated, and the `T_photon` cost row closed at ≤ 2.5 µs or escalated to the user | 05 | ⬜ |
| (i) | **PROVENANCE** | An entry for **all three** `root_scalar` sites in `LambdaCDM_GenericEOS.py` — value, method, what it sets, call count, choosing measurement **and its commit**, competing floor, cost, citation — in the shape `docs/TOLERANCE-PROVENANCE.md` will want | 06 | ⬜ |

---

## 3. Active and unresolved issues

Issues opened here must be added to [`docs/OPEN_ISSUES.md`](../../docs/OPEN_ISSUES.md) **in the
same commit** (`CLAUDE.md`), with the count corrected.

Opened by **prompt 03** (2026-09-16):

- **[03-main-py-cites-a-stale-line-for-the-equality-solve]** *(prompt 03; no prompt assigned)* —
  `main.py:524-525`'s docstring says the two equality redshifts are computed and discarded by the
  model's constructor and cites ``LambdaCDM.py:73`` and ``LambdaCDM_GenericEOS.py:483``. The first
  is right; the second is not. The constructor's two `_find_rho_equality` calls are at
  **`:501-508`**, and `:483` is now inside the `_break_point_crossings_log1pz` cache assignment,
  eighteen lines above them.
  **Impact:** none mechanically — it is a pointer in prose and nothing parses it — but it is in the
  one docstring that justifies a production grid choice, and it sends a reader looking for the
  solve to the wrong method. Prompt 03's file list permits **one** docstring sentence, `:525-527`,
  and this citation is in the sentence before it; fixing it would have been out of scope
  (README §5 rule 5). **Next step:** one line, in whichever prompt next has `main.py` in scope.

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

Opened by the **2026-09-16 planning commit**:

- **[00-equality-redshift-closed-form-is-duplicated-three-times]** *(planning; measured by prompt
  03; the decision is README §7 D2 and the user's)* — $1+z_{\rm eq}=\Omega_m/\Omega_r$ and
  $1+z_\Lambda=(\Omega_\Lambda/\Omega_m)^{1/3}$ are computed at three sites in three packages:
  `LambdaCDM_GenericEOS.py:501-506` (as a solver guess, feeding two prints),
  `CosmologyModels/LambdaCDM/LambdaCDM.py:73-74` (two prints), and `main.py:549-551`. **The third
  is load-bearing:** it becomes `feature_z`, is forced into the production source grid by
  `CosmologyConcepts/wavenumber.py:350`, and the grid's content digest is a `BackgroundModel`
  lookup-key column (`Datastore/SQL/ObjectFactories/BackgroundModel.py:182`, `:225`, `:251`). So on
  `QCD_Cosmology` the two equality redshifts are **production sample locations inside a datastore
  identity**, reached by a duplicated closed form rather than by `_find_rho_equality` — which is
  why `AUDIT.md` §2.1's grep found nothing and concluded the blast radius was two banner lines.
  **Impact:** unifying the three, which is the obvious tidy and what `AUDIT.md` §6 observation 2
  gestures at, would move a grid sample if it moved either value by one ulp, and invalidate every
  stored object of the eight types `qcd-background-audit` log 11 §5 prices. `main.py:522-528`
  records the duplication as a deliberate, scoped choice by that campaign's prompt 11.
  **Narrowed by prompt 03 (2026-09-16), measured and priced — it does not close.** The three sites
  are the **same float** on `QCD_Cosmology`, the pure-radiation stand-in and `LambdaCDM(Planck2018)`,
  on both pairs, although `LambdaCDM.py` reaches `math.pow` through `from math import sqrt, pow`
  where the other two get the builtin; the closed form and the corrected solve differ by **7 ulp**
  (QCD, matter–radiation), **1 ulp** (stand-in) and **not at all** at matter–$\Lambda$ on either
  model. The three options, priced (log 03 §3 item 3): **(i)** leave all three — **cost zero**, and
  now guarded by `CosmologyModels/tests/test_rho_equality.test_the_three_closed_form_sites_agree`,
  which a one-ulp edit at any site fails on all three models; **(ii)** unify on the closed form —
  safe only while the spellings agree bit for bit, which is measured but not guaranteed by the
  language, and it makes `main.py` import a cosmology model, the coupling prompt 11 avoided;
  **(iii)** unify on the solve — **measured** to take the `QCD_Cosmology` production grid digest
  from `a2c32f67` to `4849552b` on a 7-ulp move in one sample of 1,996, invalidating every stored
  object of the eight types log 11 §5 prices. **The campaign's recommendation is (i).**
  Line numbers re-anchored by prompt 03: the sites are `LambdaCDM_GenericEOS.py:502`
  and `:507`, `LambdaCDM.py:73-74`, and `main.py:553` and `:555`.

  **README §7 D2 — decided by the user, 2026-09-16: option (ii), unify on the closed form.**
  Put to them by the workstream B orchestrator on prompt 03's report, against the campaign's
  recommendation of (i), which is the user's to overrule. What (ii) means here is fixed by prompt
  03's measurement and not by an agent's judgement: the resulting float must be **bit-identical to
  what `main.py:553`/`:555` produce today** — `3406.668974249948` and `0.3034230329964074` on
  `QCD_Cosmology` at production parameters — and the QCD production source-grid digest must stay
  **`a2c32f67`** at 1,996 samples, with `60a3205a` at 1,778 for `LambdaCDM(Planck2018)`. The helper
  must therefore keep the arithmetic *spelling* of the three sites — `omega_m / omega_r - 1.0` and
  `pow(omega_cc / omega_m, 1.0 / 3.0) - 1.0` — not an algebraically equal rewrite; log 03 §3 item 3
  names `(omega_m - omega_r)/omega_r`, `expm1(log(...)/3)` and `**` as forms that would move the
  grid. `test_the_three_closed_form_sites_agree` becomes the test that the unified helper still
  reproduces all three sites, and the two digests are the acceptance.
  **Next step:** **a new prompt, not an amendment to 03** (orchestrator
  `workstream-B-what-the-redshifts-feed.md` §2). It is not one of this campaign's eight, and it is
  not scheduled; it needs a home and a slot before anything writes the helper. **Until that prompt
  runs the issue stays open, and nothing may unify the three sites.**

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
| `[08-temperature-crossing-solver-is-test-only]` | qcd-background-audit | prompt **04** | Needs `LambdaCDM_GenericEOS.py` and `CosmologyModels/tests/` in scope together. Prompt 04 has both, and the method is one of the three `root_scalar` sites prompt 06 must write provenance for |
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
   a move there means something else changed.
4. **A figure without the tree it was taken on is not a measurement** (README §5 rule 9).
5. **The two printed banner lines at `LambdaCDM_GenericEOS.py:516-517` are character-identical
   throughout.** They are the user-visible surface of the whole campaign, and the campaign's claim
   is that they do not change.
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
12. **The production source-grid digests are `a2c32f67` (`QCD_Cosmology`, 1,996 samples) and
    `60a3205a` (`LambdaCDM(Planck2018)`, 1,778)**, measured identical at `7fdc49b`, `921f41c` and
    prompt 03's commit. Any later prompt in workstreams A–C that moves either has left its scope.
    Reproduce with `ComputeTargets/tests/test_source_grid._production_grid(cosmology)` and
    `_to_redshift_array(grid.z_values).digest()`; `QCD_Cosmology` must be built at the production
    `max_z = 1e20`, not at the test modules' `1e12`, or the spacing profile leaves the tabulated
    range.
13. **The three closed-form sites are one float, and that is now a test.**
    `CosmologyModels/tests/test_rho_equality.test_the_three_closed_form_sites_agree` asserts it on
    three models and both pairs. A prompt that edits any of `main.py:553`/`:555`,
    `LambdaCDM_GenericEOS.py:502`/`:507` or `LambdaCDM.py:73-74` will hear about it there first.
