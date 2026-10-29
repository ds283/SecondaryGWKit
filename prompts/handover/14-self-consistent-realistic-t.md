# Prompt 14 — B2a: give the realistic $T$ production's Liouville–Green construction

**Campaign:** [`README.md`](README.md) · **Board item:** **B2a**, added 2026-10-29 between B2 and
E1 (not in the original plan) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-29, on the user's decision of the same day: a small prompt ahead of E1.
**Where it runs:** on the laptop, in a git worktree of its own, Ray-free, with no datastore and no
lock, like B2. Orchestrated from the laptop ([`orchestrator/prompt-14.md`](orchestrator/prompt-14.md)).
**In two parts, two commits**, as B2:
- **Part A (§1): the fixture.** Test fixtures only. One commit.
- **Part B (§2): B2's attribution re-taken** on that fixture with the $T$ terms re-cut. A
  registered run of about 10 minutes, then one records commit.

**Closes, in the records commit, if §2's acceptance holds:**
`[13-the-tlg-term-depends-on-the-levin-subdivision]`.
**Narrows:** `[00-tk-lg-truncation-floor]` (`GkTk-remedial`) and
`[07-lg-derivative-truncation-at-handover]` (`source-remediation`).
**Recommended model:** **Opus**.

**Read first.** Every path is read at the dispatch SHA.
1. This board's §3: `[13-the-tlg-term-depends-on-the-levin-subdivision]` **with its narrowing of
   2026-10-27**, which is the reason for this prompt. `GkTk-remedial`'s
   `[00-tk-lg-truncation-floor]`, with its narrowing by B2.
2. [`docs/handover/SEAM-RESIDUE.md`](../../docs/handover/SEAM-RESIDUE.md): §0, §1, §4 and **§4.7**.
   Then [`docs/handover/b2/probe13/README.md`](../../docs/handover/b2/probe13/README.md) and the
   probe's three outputs.
3. [`prompts/handover/13-score-the-unmasked-residue.md`](13-score-the-unmasked-residue.md) §2, and
   [`logs/13-score-the-unmasked-residue.md`](logs/13-score-the-unmasked-residue.md): its
   deviations and its "Records at their own SHA".
4. `docs/handover/b2/scripts/`: `b2_cells.py` (above all `HybridT`, `apply_hybrid`, `build_case`,
   `run_cell`), `b2_checks.py`, `b2_driver.py`, `b2_launch.py`, `b2_tables.py`, and
   `probe13.py`. Part B's scripts are successors of these.
5. `ComputeTargets/tests/test_tk_source_functions.py`: the module docstring (the "exact" and "LG"
   fixtures and why they differ), `Fixture`, `exact_functions`, `LG_functions`,
   `closed_form_residual`, `test_omega_matches_phase_derivative` and
   `test_LG_truncation_error_of_the_exact_envelope`.
6. `ComputeTargets/tests/test_tk_wkb_phase.py`: `build_and_store`, `_Shared.build_radiation` and
   `TestRadiationValue`. **This runs production's own `TkWKBIntegration` producer offline**, on
   exact initial data, and is the construction Part A should reach for (§1.1).
7. `ComputeTargets/TkWKBIntegration.py`: the class docstring and `store()`. The phase is
   $\theta_T = -[k\,\tau_s + \rho_T]$ from $z_{\rm init}$. The amplitude comes from `friction_F`.
   `sin_coeff`, `cos_coeff` and `deltaTheta` are matched to the numeric $T$, $T'$ at
   $z_{\rm init}$. The stored `theta_residual` follows `levin-phase-floor` 06.
8. `ComputeTargets/TkSourceFunctions.py`: `M`, `dlnM_dz`, `omega`, `phase`, and the module
   docstring's account of the amplitude and the phase. `ComputeTargets/primitive_phase.py`:
   `theta_deriv`.
9. `ComputeTargets/tests/test_quadsource_integral.py`: `Case`, `production_fixture` and
   `exact_with_the_geometry_of`, as B2 left them.
10. Campaign [`README.md`](README.md) §0.4, §5 and §5.1; `CLAUDE.md`, "Long-running jobs — the run
    registry".

**It changes no production code.** Part A edits test fixtures; Part B adds `docs/` scripts and a
section of a document.

**Gate** (check, and save the output in the log):
- you are in the worktree the orchestrator names, at the dispatch SHA, with `git status --short`
  empty there;
- `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale on this host;
- `docs/handover/b2/scripts/b2_cells.py` hashes `0a8ae0eb…`;
- the seven roots' counts, taken by the orchestrator at the dispatch SHA, are in hand.

---

## 0. Why this exists

B2 attributed the seam's residue on a realistic fixture whose $T$ is **internally inconsistent**:
- its stored amplitude and phase are the **exact** Bessel ones (`Fixture.exact_functions`);
- its `omega` and `dlnM_dz` are the **Liouville–Green** closed forms.

The Levin driver oscillates with the stored phase but takes its derivative from `omega`. Where the
two disagree, its result depends on how finely it subdivides, and its error estimate cannot see
that. That is `[13-…]`, confirmed by a probe on 2026-10-27 (SEAM-RESIDUE.md §4.7). Taking the
derivative from the phase removes every interaction, to $\le4\times10^{-10}$. About
$1.5\times10^{-4}$ of B2's `together`-980 residue was the artefact.

**Production does not have the inconsistency.** `TkWKBIntegration` stores
$\theta_T = -[k\,\tau_s + \rho_T]$ with $\rho_T$ integrated from the same $\omega_{\rm eff}$ that
`TkSourceFunctions.omega` returns, and its amplitude from the same friction integral that
`dlnM_dz` differentiates. Its $T$ differs from the exact one by the **Liouville–Green truncation**,
in amplitude and phase as well as in the derivatives: `[00-tk-lg-truncation-floor]`'s $x_i^{-3}$
law. B2 could not measure that half, because its fixture's amplitude and phase were exact.

So this prompt:
- gives the realistic $T$ production's construction, consistent by construction (Part A);
- re-takes B2's attribution on it, with $T$'s Liouville–Green terms re-cut so that **every
  hybrid is itself consistent** (Part B).

E1 and E2 trade the source spline against exactly this truncation, so they need this number.

### 0.1 Since B2

The Levin driver has changed: `levin-phase-floor` 06b, 06c and 06d landed between B2 and this
prompt, and 07 may have landed by dispatch. B2's numbers are therefore context, not a column:
everything here is computed at the dispatch SHA. §2.4's controls say how far the driver change
alone moved the exact cells.

---

## 1. Part A — production's $T$ in the realistic flavour

Work only in the worktree, with `PYTHONPATH=.` on every command.

### 1.1 The construction

`Case(exact=False)`'s transfer functions get their WKB samples **from production's construction**,
not from the exact Bessel decomposition:
- the phase $\theta_T = -[k\,\tau_s + \rho_T](z_{\rm init}\to z) + \Delta\theta$, from the same
  $\omega_{\rm eff}$ as `TkSourceFunctions.omega`;
- the amplitude from the model's own `friction_F`, the one `dlnM_dz` differentiates. The
  realistic $T$ is therefore read against the fixture's `FakeModel` itself, not against
  `exact_envelope_model()`;
- `sin_coeff`, `cos_coeff` and $\Delta\theta$ matched to the exact $T$ and $T'$ at $z_{\rm init}$,
  as `TkWKBIntegration.store()` matches them to the numeric solution there;
- `theta_residual` as production stores it since `levin-phase-floor` 06.

**The preferred route is production's own code**: drive `TkWKBIntegration`'s `compute()` payload
and `store()` offline, as `test_tk_wkb_phase.build_and_store` does, and capture the stored values
as the fixture's `Tk_WKB`. If `Case`'s `FakeModel` cannot drive the producer, there are two
fallbacks:
- give the fixture a background the producer can run on;
- reproduce the producer's formulas in the fixture, and show at one configuration that they agree
  with `build_and_store` to its own declared accuracy.

Say which route was taken, and why. `LG_functions` is **not** the construction: its `sin_coeff` is
arbitrary and its phase starts at 0. It is not matched.

**Unchanged:** the numeric $T$ samples (exact), production's seam (`drop_first_WKB_sample=True`),
the table `cs_tau` with $z_{\rm init}$ off a node, and Part A of B2's $G$. `Fixture.exact_functions`
and its tests stay as they are; the exact-envelope fixture is still what tests the amplitude's
*assembly*.

### 1.2 Tests

1. **New tests**, each named in the log:
   - (a) **consistency:** on one cheap `Case(exact=False)`, over each realistic $T$'s WKB region,
     $\max|\,\texttt{phase.theta\_deriv}(\log) - \omega\,(1+z)\,| / |\omega\,(1+z)|$ and the same
     for $\texttt{dlnM\_dz}$ against a finite difference of $\log M$. Bound each from its
     provenance (the residual spline's derivative error on this grid), and quote the measured
     value. **This is the fixture's measure of production's own mismatch.** It is what the
     2026-10-27 narrowing could not give;
   - (b) **the matching:** at $z_{\rm init}$, the realistic $T$'s Liouville–Green value and
     derivative equal the exact $T$ and $T'$, to the matching's arithmetic;
   - (c) **the truncation is the physical one:** against the exact $T$, the realistic $T$'s
     error over its WKB region, as a fraction of the envelope, is what
     `TestRadiationValue`'s $x_i^{-3}$ law gives at the fixture's hand-over $x_T\approx19.1$,
     within a stated factor.
2. **Existing tests whose numbers move** are listed with old and new values. **A test that fails
   is a stop**; do not change its tolerance, report the margin.
3. **Deliberate breakage**, each recorded and reverted:
   - build the realistic $T$'s phase from the exact Bessel phase again: (a) must fail;
   - perturb $\Delta\theta$ by $10^{-3}$: (b) must fail;
   - read the realistic $T$ against `exact_envelope_model()`: (a)'s amplitude half or (c) must
     fail. Say which.
4. **The seven roots** at the end, serially. A count that falls is a stop; a rise equals the tests
   added.

### 1.3 Part A's commit

`black` on every touched `.py` file. **One commit on the worktree's branch**, staged by explicit
path: test modules under `ComputeTargets/tests/`, the log
`logs/14-self-consistent-realistic-t.md` (Part A's half), and the board (B2a's row and item,
"Part A landed"; no issue moves). No push, no merge, no rebase. Then **stop and report**: the
orchestrator reviews Part A before Part B starts.

---

## 2. Part B — B2's attribution, re-taken

### 2.1 The script

Successors of B2's scripts, under the worktree's `var/handover-14/scripts/` (gitignored), copied
back to `docs/handover/b2a/scripts/` byte-identical in the records commit. They import what still
holds from `docs/handover/b2/scripts/b2_cells.py` and from `realistic_large_x.py`, and build every
case through `Case` at the dispatch SHA. A4's scripts cannot be imported at `HEAD` (B2's log).

### 2.2 The objects and the terms

B2's objects (E, E′, R, five add-one, five remove-one: 13 per rung) and B2's ladder (A2's `RUNS`,
16 rungs: **208 cells**). The terms change in one place: **$T$'s Liouville–Green accessors are
split by what is consistent with what**, so that no hybrid pairs a phase with another phase's
derivative, or an amplitude with another amplitude's log-derivative:

| term | accessors | was, in B2 |
|---|---|---|
| `f` | `QuadSource.functions.source` | `f` |
| `Tnum` | `T.T`, `T.dT_dz` | `Tnum` |
| `Tphase` | `T.phase`, `T.omega` | `phase` in `Trep`, `omega` in `TLG` |
| `Tamp` | `T.M`, `T.dlnM_dz` | `M` in `Trep`, `dlnM_dz` in `TLG` |
| `G` | `Gk.phase`, `Gk.sin_amplitude` | `G` |

`Tphase` and `Tamp` each carry both the Liouville–Green truncation and the consumer's re-spline of
the stored samples. B2 measured the re-spline below $10^{-10}$ in $N$; say if that no longer holds.
Completeness and disjointness are proven as B2 proved them, bit for bit (prompt 13 §2.2).

### 2.3 What is computed

As prompt 13 §2.3: $D$, $A_i$, $B_i$, both interactions, and the geometry control $N_{E'}-N_E$,
each with its error. §2.3's additivity rule is unchanged. **The expectation**, fixed here before
the run: with every hybrid consistent, $A_i = B_i$ to within the declared errors for every term
at every rung. If that fails anywhere, report where; do not explain it away.

### 2.4 Before the launch

Saved under `var/handover-14/state/`:
1. **Construction:** R's $T$ objects are Part A's, at every rung; R's $G$ is B2's.
2. **Completeness and disjointness** at `together` 980.
3. **Determinism** and **transparency**, as B2.
4. **Controls:**
   - E against B2's E cells (its checkpoint, `var/runs/handover-13-residue-20261027T110832/`,
     in the main checkout), rung by rung. Any difference is the Levin driver's change since B2.
     Record it against the declared errors, without judging it;
   - E against A2's exact column, as B2 did;
   - the geometry control at `together` 980.
5. **The stationary-point scan,** as B2. A sign change is a stop.
6. **`state/terms.md`:** §2.2's mapping and §2.3's rule and expectation, written last, before the
   launch. **Run no cell of the ladder before it exists**, smoke tests included. Test the scripts
   on a rung outside the ladder instead, for example `together` at $x_{\rm resp} = 300$, and say
   which.

### 2.5 The run

Registered as B2's was: campaign `handover`, prompt `14`, slug `residue-lg`. Detached; at most two
cells at once; a one-hour budget per cell. **Verify in one check that it started, and stop.**

### 2.6 What to write down

**`docs/handover/SEAM-RESIDUE.md` gains §8**, dated, additive: B2's §0–§7 stay as written. §8
contains:
- what changed (the fixture, the term split, the driver since B2) and the controls;
- the table of §1, re-taken: $D$ and the five terms per shape, with errors, and additivity per
  rung against the expectation of §2.3;
- `together` at 980 again: what carries its residue now;
- **the Liouville–Green truncation in $N$** (`Tphase` + `Tamp`) against the `f` term, per shape
  and rung. That is the comparison E1 and E2 are planned on;
- test (a)'s mismatch, as the bound on production's;
- what this still does not cover. B2's §7 list stands, less its items on the stored $T$ being
  exact Bessel and on `TLG`'s per-term value.

Beside it, under `docs/handover/b2a/`: `scripts/`, `out/` (tables and state).

---

## 3. What this prompt does not do

- No production code: not `TkWKBIntegration.py`, `TkSourceFunctions.py`, `QuadSourceIntegral.py`,
  `phase_groups.py` or `AdaptiveLevin/`. **Supplying Levin's derivative from the phase in
  production is not this prompt's business**: production is consistent by construction, and test
  (a) measures by how much.
- No real-store measurement. Production's mismatch on `TkWKBValue` rows of A3 v2 needs the Mac
  Studio. If test (a) bounds it below $10^{-8}$ relative, the records commit records that the
  real-row check is not needed for E1's purpose; otherwise it opens a §3 issue for it.
- It does not edit B2's scripts, B2's sections of SEAM-RESIDUE.md, or any other record.
- No hand-over depth, grid density or remedy is chosen.

---

## 4. Acceptance

**Part A**
1. The realistic $T$ is production's construction, by the route §1.1 names.
2. Tests (a)–(c) pass, with (a)'s measured mismatch quoted. The breakages are caught; every moved
   number is listed.
3. The seven roots: none falls; `ComputeTargets` rises by the tests added.

**Part B**
4. The run is registered and `done`; every cell is `ok` or `not reached`. `state/terms.md` is
   older than the manifest, and no ladder cell ran before it existed.
5. Completeness holds bit for bit; the controls are recorded.
6. $A_i = B_i$ for every term at every rung within the declared errors, or every exception is
   reported.
7. SEAM-RESIDUE.md §8; board, issues and `docs/OPEN_ISSUES.md` per §5.

---

## 5. Stop conditions, the log, the board

**Stop and ask** if:
- the Gate fails;
- an existing test fails after Part A;
- the producer cannot be driven offline **and** its formulas cannot be shown equal to it;
- completeness fails, or a hybrid cannot be built without a production change;
- the stationary-point scan finds a sign change;
- a cell raises for any reason other than its budget;
- anything needs Ray, a datastore, the lock or the Mac Studio.

**The log:** `logs/14-self-consistent-realistic-t.md`, in `qcd-background-audit` README §5.1's
form, with every deviation classified. Part A writes its half and Part B completes it.

**The records commit**, on the worktree's branch after the run, holds:
- `docs/handover/b2a/` and SEAM-RESIDUE.md §8; the log completed;
- the board: B2a's row and item, §3 and §4. `[13-…]` moves to §4 if acceptance item 6 holds;
  otherwise it is narrowed with what failed;
- dated narrowing notes on `[00-tk-lg-truncation-floor]` (`GkTk-remedial`) and
  `[07-lg-derivative-truncation-at-handover]` (`source-remediation`), pointing at §8;
- `docs/OPEN_ISSUES.md` corrected by counting;
- a §3 issue for anything found and not fixed, the real-row check included if §3 requires it.

The copied scripts are records. If they are not `black`-clean, say so in the log: the
orchestrator lists `docs/handover/b2a/scripts/` in `CLAUDE.md` with the user's agreement. Commit
messages per `CLAUDE.md`. No push, no merge.
