# Prompt 05a — rows 1 and 2 for $T$: the node-anchored remainder, against $2\pi$

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-26, after 05 landed, on the user's decision the same day (the board's
Decisions, **05a**): give $T$'s consumer phase what 05 gave $G$'s, in exactly the same way.
- **Row 1, the anchoring floor.** $T$'s remainder is formed by reducing a double of size
  $k\,\tau_s$, so it carries rounding of about $\varepsilon k\tau_s$ that is not smooth in $z$.
- **Row 2, the modulus.** That reduction is against the double `TWO_PI`, not $2\pi$: a smooth
  error of about $0.18\,\varepsilon|\theta_T|$.

Row 3, the noise in the producers' stored samples (`[02-consumer-phi-below-the-storage-granularity]`),
is G1 Q3's later producer work for $G$ and $T$ alike (06+). It is not this prompt's.

**Where it runs:** in three parts, as 05 did.
- **§1–§4, the code:** on the laptop, by a fresh-context subagent, in the main checkout at `HEAD`.
  One commit.
- **§5, rehearsals R11 and R12:** on the Mac Studio, orchestrated there, after the code commit is
  pushed, under the lock.
- **§6, the re-measurement:** on the Mac Studio, Ray-free, reading A3 v2. It may run beside §5's
  builds, since it takes no lock.

The records of §5 and §6 are committed on the laptop afterwards, in 05a's records commit.

**Closes, when §5 and §6 hold:**
- `[05-t-remainder-keeps-the-anchoring-floor]` (this campaign);
- `[04-g-phase-is-reduced-against-the-double-two-pi]` (this campaign). 05 did its $G$ half; this
  prompt does the $T$ half.

**Recommended model:** **Opus**.

**Read first:**
1. This prompt's predecessor, [`05-reanchor-g-phase.md`](05-reanchor-g-phase.md), in full, and its
   log [`logs/05-reanchor-g-phase.md`](logs/05-reanchor-g-phase.md): its deviations, its breakage
   (iii) and its two stops.
2. The board's Decisions (G1 and 05a) and §3's entries for the two issues above;
   [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §4
   and §5; `docs/levin-phase-floor/04/issues.md` I2, I3 and I7.
3. `ComputeTargets/primitive_phase.py`, in full: the module docstring's "Anchoring, and the floor
   it implies" and "Who uses which", the constructor's `exact_remainder` branch (`:296-321`),
   `raw_theta` (`:376`) and `theta_mod_2pi` (`:385`).
4. `ComputeTargets/TkSourceFunctions.py`: the comment on `TK_PHASE_SIGN` (`:207-214`),
   `_sound_horizon_rate`, the constructor (`:276-`), the `PrimitivePhase(...)` in `_build_WKB`
   (`:494-504`), and the two uses of `theta_mod_2pi`: `handover_continuity` (`:630`) and `T_WKB`
   (`:772-781`).
5. `ComputeTargets/BackgroundModel.py:448-461`, where `cs_tau` is built as a `CumulativeTable`
   shifted by `cs_tau_init`, and `ComputeTargets/handover.py:60-70`: $T$'s anchor `z_init` is a
   root, not a grid node.
6. `ComputeTargets/QuadSourceIntegral.py:615-640`: `Tk_functions_builder`, which production never
   passes.
7. `ComputeTargets/tests/test_g_phase_reanchor.py`, in full: 05's tests, its `_TableCsTauFixture`
   (`:446`) and the test this prompt replaces (`:511`).

**Gate:**
- the main checkout's `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

### 0.1 The defect

$T$'s phase is $\theta_T(z) = +k\,\Delta_{\tau_s}(z, z_{\rm init}) + \phi(u)$, with $u = \log(1+z)$.
Its anchor is `z_init`, at the top of the WKB region. `theta_mod_2pi` forms it as one double, of
size about $k\tau_s(z)$, the sound horizon at $z$, and reduces that with `WKB_mod_2pi`. So:
- **Row 1.** Every remainder carries rounding of about $\varepsilon k\tau_s(z)$, not smooth in $z$.
  It is small at early source times and largest at late ones. With $G$ already clean, 04's I7
  put its predicted size, $\varepsilon\sum|\theta_T|$ at the late end of the `Gqr` sub-intervals,
  at 1.1–1.3 $F$: two $T$ factors there. That is a bound from the phase's size, not a measured
  $\sigma_\theta$.
- **Row 2.** The reduction against `TWO_PI` costs a smooth $0.18\,\varepsilon|\theta_T|$: 1.9e-4 to
  0.051 $F$ on 04's rows (I3).

**What the fix cannot remove: the abscissa's own rounding.** A phase evaluated at $u$ is evaluated
at $z = {\rm expm1}(u)$ rounded to a double. That puts about
$\varepsilon\,\tfrac{z}{1+z}\,|d\theta/du|$ of jitter into it (04's I1, where it was the whole of
arm X's residual). The node-anchored remainder does not touch it.
- **For $G$ it was negligible where it mattered.** $|d\theta_G/du| = k(1+z')/H \approx k\tau(z')$,
  far below $k\eta_R$ at early source times.
- **For $T$ it is not.** $|d\theta_T/du| = k c_s(1+z)/H$, and $|\theta_T| \approx k c_s\tau(z)$. In
  radiation domination $\tau = (1+z)/H$, so the two are **equal**. Wherever $z \gg 1$, the
  abscissa term is the same size as the floor this prompt removes.
- **So the expected gain from row 1 for $T$ in production is a factor of order one,** and larger
  only at $z \lesssim 1$, where $z/(1+z)$ falls. That is the low band. Row 2 is a clean gain
  everywhere.

This is arithmetic, not a measurement. §3 test 3 and §6's probe separate the two terms by sampling
in $z$, and they report the $u$-sampled jitter beside it. The abscissa term is not this prompt's to
fix.

Neither of the two errors has been seen to drive the Levin bisection. 04's arm A left $T$ as
production forms it and matched the exact limit region for region, and 05 removed the cap on all
11 capped rows without touching $T$. **This prompt is for correctness, not cost** (README §5 rule 4). It removes two
known errors from a factor that enters the integrand on the same footing as $G$, and it measures
on real rows what they cost.

### 0.2 The construction

There is nothing new to build. 05 built the node-anchored remainder in `PrimitivePhase`, for any
leading primitive with node accessors and for either sign (`exact_remainder=True`). Production's
`cs_tau` is a `TablePrimitive` over a `CumulativeTable`, so it has them. With the keyword on, $T$'s
remainder is
$$c_{n_z} + {\rm rem}(v(z)),\qquad c_n = \big(k\,[T(n_R) - T(n) + p_R]\big) \bmod 2\pi,\qquad v(z) = -k\,p_z + \phi(u),$$
with $n_R$, $p_R$ the node and end partial of `z_init`. Each $c_n$ is formed exactly from the limbs
and reduced against $2\pi$ at 256 bits, as for $G$.

**The one thing that is new in production is $p_R \neq 0$.** $G$'s anchor $z_R$ is a table node in
production, so its $p_R$ is zero (`primitive_phase.py:316-317`). $T$'s `z_init` is a root, not a
node (`handover.py:67`), so its $p_R$ is a double from the table's own quadrature, and enters
$c_n$ exactly as that double. The tests below cover it on and off a node.

### 0.3 How it is switched on

`TkSourceFunctions` gains a constructor keyword, **`exact_remainder: bool = True`**, passed to the
`PrimitivePhase` of `_build_WKB`.
- **Production never passes it.** `QuadSourceIntegral` builds $T$ through `Tk_functions_builder`,
  called with four positional arguments. That builder is itself a parameter that "production never
  passes" (`QuadSourceIntegral.py:637-639`), which is the precedent for a keyword tests use.
- **A test fixture whose `cs_tau` is a `ClosedFormPrimitive`** passes `exact_remainder=False`
  explicitly. A closed form has no table nodes, so `PrimitivePhase` raises, as 05 §2 requires.
- **No fallback.** `TkSourceFunctions` does not switch the keyword off when the primitive lacks
  accessors. That would be the silent fallback 05 §2 forbade, one level up.

### 0.4 What it does not change

- **$G$'s phase**, bit for bit.
- **`raw_theta` and `theta_deriv` of $T$**, bit for bit.
- **`PrimitivePhase` and `CumulativeTable`.** If the construction needs a change to either, stop.
- **The producers, any schema, `AdaptiveLevin/`, and any stored value of any class except
  `QuadSourceIntegral`'s.** `QuadSource` reads `TkNumericIntegration` only
  (`QuadSource.py:110-146`), and no `extract_*` reader builds a `TkSourceFunctions`.
- **Row 3**, the stored samples' noise.

## 1. `TkSourceFunctions`

- The keyword of §0.3, documented in the constructor's docstring.
- The comment on `TK_PHASE_SIGN` and the module docstring say what is now true of $T$'s remainder.
- In `primitive_phase.py`, rewrite only the module docstring's "Who uses which": both factors now
  use the node-anchored remainder, by the user's decision of 2026-10-26 (05a), and $T$'s anchor
  carries a non-zero end partial. No code in `primitive_phase.py` changes.

## 2. The fixtures

Every existing test fixture that builds a `TkSourceFunctions` on a `ClosedFormPrimitive` `cs_tau`
passes `exact_remainder=False`. The candidates are the constructions in
`ComputeTargets/tests/test_tk_source_functions.py`, the `Tk_builder` of
`test_quadsource_integral.py:425` and `test_phase_groups.py:653`. List every site changed in the
log, by file and line.

**No other existing assertion changes,** with one exception:
05's `TestNothingElseMoves.test_transfer_function_remainders_are_unchanged`
(`test_g_phase_reanchor.py:511`). It asserts G1's decision that $T$ stays as it was, which the user
has now reversed. Replace it with §3 test 4(b), and say so in its place. If any other existing
assertion has to change, stop.

## 3. Tests

A new module, `ComputeTargets/tests/test_t_phase_reanchor.py`, on the stand-in pattern of
`test_g_phase_reanchor.py`, with no Ray and no datastore. Its `cs_tau` is a real `TablePrimitive`,
as in `_TableCsTauFixture`. Import that fixture rather than copying it, if it can be imported
cleanly.

1. **Production's configuration.**
   - `TkSourceFunctions` on a table `cs_tau`, with no keyword, has `phase.exact_remainder` true.
   - On a `ClosedFormPrimitive` `cs_tau`, with no keyword, it raises the `ValueError` that names
     the missing accessors.
   - On a `ClosedFormPrimitive`, with `exact_remainder=False`, it builds.
2. **Exactness against $2\pi$.** At $k = 10^5$, $3\times10^6$, $10^8$ and $3\times10^8$, with
   `z_init` **on a node and off one**, over abscissae spanning the WKB range from `z_init` to its
   bottom: the remainder is within $4\varepsilon\,(k|p_z| + 2\pi)$ of the exact remainder of
   $\theta_T$ against $2\pi$. The reference is computed in `mpmath` from the same limbs, the same
   $p_R$ and $p_z$, and the same $\phi$.
3. **No jitter.** Take a window of relative width $10^{-6}$ in $z$, at a source redshift where
   $k|\Delta_{\tau_s}(z, z_{\rm init})| \ge 10^9$ rad; choose it and record $z$ and the phase.
   **Sample it on 65 doubles in $z$, call `theta_mod_2pi(z, x_is_log=False)`, and fit in $z$,** so
   that no `expm1` rounding enters (§0.1). Otherwise use 04's $\sigma_\theta$: `numpy.unwrap`, a
   degree-4 fit, and the RMS residual.
   - With the keyword off it is at least $0.1\,\varepsilon|\theta_T|$.
   - With it on it is at most $10^{-3}$ of that.
   - **Also record, unscored,** $\sigma_\theta$ of the same window sampled in $u$ (as `T_WKB` and
     the phase groups call it), keyword off and on, beside
     $\varepsilon\,\tfrac{z}{1+z}|d\theta_T/du|$.
4. **Nothing else moves.**
   - **(a)** $T$'s `raw_theta` and `theta_deriv` are bit-identical with the keyword on and off.
   - **(b)** With the keyword off, `theta_mod_2pi` is bit-identical to
     `WKB_mod_2pi(raw_theta)[1]`. This replaces 05's test (§2).
   - **(c)** $G$'s phase, built by `GkSourcePolicyData._build_phase`, still has `exact_remainder`
     true, and 05's tests pass unchanged.
5. **$T$ itself.** `T_WKB`, `TkSourceFunctions`' own value of $T_k$ in the WKB region, moves
   between the keyword off and on by no more than its amplitude times the old error,
   $4\varepsilon k|\Delta_{\tau_s}| + 0.18\,\varepsilon|\theta_T|$, plus $10^{-14}$ relative.
6. **End to end.** Use `test_phase_groups.py`'s realistic fixture at one cheap configuration of
   regime `Gqr`, with $G$'s phase as production builds it (05's `_ProductionPhaseGk`) and both $T$
   factors on a table `cs_tau` with the keyword on:
   - `evaluate_sum` against the direct product $G f/H^2$ stays within the bound the module's
     existing test states;
   - each group's composed remainder differs from the keyword-off one by at most the sum of the two
     $T$ factors' old errors, plus $10^{-14}$.

**Deliberate breakages,** each recorded in the log with the test that catches it, then reverted:
- **(i)** the keyword's default `False`: test 1 must fail;
- **(ii)** `TkSourceFunctions` falls back to `False` when the primitive lacks accessors: test 1's
  raise must fail;
- **(iii)** `_build_phase` turns the node-anchored remainder off for $G$: test 4(c) must fail;
- **(iv)** in `PrimitivePhase`'s anchor, drop the end partial ($p_R \to 0$): test 2 must fail off a
  node. This is the one breakage inside `primitive_phase.py`, and it is reverted like the others.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added, less the one
replaced.

## 4. Records of the code

- `black` on the files changed.
- `logs/05a-reanchor-t-phase.md`, in README §5.1's form. Its **Predictions and verdicts** section
  holds §6's predictions, copied before §6 runs.
- The board's 05a row.
- **One commit.** The issues of the header move only in 05a's records commit, after §5 and §6.

## 5. Rehearsals R11 and R12 — on the Mac Studio, after the code commit is pushed

As 05 §5, with what 05 learned. Records under `var/levin-phase-floor-05a/`.
1. **The lock:** `acquire`, then `set-pid` at each launch, and `release` after the readers. Each
   lock command is a shell command of its own (`CLAUDE.md`).
2. **R11** is at the code commit's parent, in a worktree. It runs `RunDrivers/quadsource_atol_sweep.py
   --build --rehearsal` as R8, R9 and R10 ran it, registered (`--campaign levin-phase-floor
   --prompt 05a`, slug `r11-parent`). Its fingerprint must equal `1b332ffa…`.
3. **R12** is the same at the code commit (slug `r12-reanchor-t`). Its fingerprint must also equal
   `1b332ffa…`.
4. **The value comparison:** `python -m tools.store_value_diff` on R11 and R12, at the code commit,
   with `--json`, saved to `out/`.
   - **The tool reports columns that are not computed values** (`[05-store-value-diff-compares-volatile-columns]`):
     wall-clock timings, the timestamped `label`, `GkSource.parents_digest` and the GkWKB memo
     counters. Classify every difference outside `QuadSourceIntegral` with 05's
     `docs/levin-phase-floor/05/rehearsal/scripts/classify_non_qsi.py`, copied unchanged into
     `scripts/` and recorded by its SHA-256.
   - **Outside `QuadSourceIntegral`,** every value table of every class is identical, every
     difference falls into those four kinds, and no row is in one store only. Anything else is a
     stop.
   - **In `QuadSourceIntegral`,** tabulate per row with 05's `tabulate_qsi.py`, copied the same way:
     - $|\Delta{\rm total}|/|{\rm total}|$ and $|\Delta{\rm total}|/$`total_abserr`;
     - the changes in `total_converged`, regions and maximum depth;
     - which metadata keys differ.
5. **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`) on R12: every one exits 0,
   and the store is byte-identical before and after.
6. **Release the lock.** `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing.
7. **The stores are kept** until the user says otherwise, as 05's were.

## 6. The re-measurement — on the Mac Studio, Ray-free, reading A3 v2

05's harness on 05's rows, so that every number here compares row for row with 05's.
- **The harness** is `docs/levin-phase-floor/05/wider/scripts/`, copied into
  `var/levin-phase-floor-05a/scripts/`. Run it in two trees: the code commit's parent ($c$, whose
  code is 05's $c$) and the code commit ($t$).
  - The scripts name 05's paths, its trees `p` and `c`, and its unit sets. Change only those, and
    whatever the probe below needs.
  - Save the diff against 05's copy in the records, and record each change as a deviation.
- **The rows are 05's 39**, from `var/levin-phase-floor-05/wider/state/sample.json` on the Mac
  Studio, copied unchanged. Its SHA-256 must equal the `sample_sha256` that 05's run recorded.
  There is no new draw.
- **"Depth-20 row"** has two readings, and both are reported, as 05's were (its second stop): the
  31 sampled rows that A3 v2 stores at depth 20, and the 11 of them capped under B1's code
  (VERIFICATION §5.2's table). Each scored rule below names its reading.
- **The store is never written.** Hash every store file before and after, and check the
  fingerprint `0e32f24a…` at the end.

**The units.** One registered, detached run (slug `wider`), with one checkpoint per (tree, rtol,
row):
1. $c$ at `rtol = 1e-7`, on every row: the baseline, and each row's budget;
2. $t$ at `1e-7`, on every row;
3. $t$ at `1e-8`, on the 31 depth-20 rows.

Units 2 and 3 have a budget of 3× the row's $c$ time, enforced as 04 and 05 enforced it. An
over-budget unit is a result.

**The phase probe.** For every Levin sub-interval of every row in which a $T$ factor is oscillatory,
and for each such factor: a window of relative width $10^{-6}$ at the sub-interval's low-$z$ end,
where $|\theta_T|$ is largest. Record there:
- $|\theta_T|$, $|d\theta_T/du|$ and $z$;
- $\sigma_\theta$ of that factor's `theta_mod_2pi`, under $c$ and under $t$, sampled two ways:
  - **in $z$**, as §3 test 3 does: this is what T1 scores;
  - **in $u$**, as production calls it: this is descriptive, and shows what is left (§0.1).

This reads the phase objects only and runs no quadrature.

**Gate T0, before the run:**
- `git diff --stat 120fdfe <code commit's parent> -- . ':!docs' ':!prompts'` shows nothing but
  `CLAUDE.md`. At writing, that holds at `HEAD`. If another session has changed code since, stop
  and ask: 05's $c$ is then not this $c$.
- $c$ at `1e-7` reproduces 05's $c$ unit (`levin-phase-floor-05-wider-20261026T020005`) **bit for
  bit** on one depth-20 row and one control: `total`, `total_abserr`, and every call's value and
  abserr.
- Two in-process runs of the same $t$ unit are bit-identical.

If T0 fails, stop.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
- **T1 — the floor is there, and goes.** *(Replaced before the run by §9, 2026-10-26.)*
  - **The rule:** in every probe window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad, sampled in
    $z$: $\sigma_\theta \ge 0.1\,\varepsilon|\theta_T|$ under $c$, and
    $\sigma_\theta \le \max(10^{-3}\varepsilon|\theta_T|,\ 10^{-13}\ {\rm rad})$ under $t$.
  - **Holds** if true in every such window.
  - **The $u$-sampled $\sigma_\theta$ under $t$** is not scored. Wherever it exceeds
    $0.1\,\varepsilon|\theta_T|$, report it against $\varepsilon\,\tfrac{z}{1+z}|d\theta_T/du|$, and
    propose an issue for the abscissa term in the log's "Observations not acted on".
- **T2 — the cap does not return.**
  - **The rule:** on every row (all 39), under $t$ at `1e-7`, no group has a region at depth ≥ 18.
  - **Holds** if true on every row; an over-budget unit counts as false.
- **T3 — $T$'s errors were within the declared error.**
  - **The rule:** on every row (all 39), $|{\rm total}_t - {\rm total}_c| \le$ `total_abserr`$_c$,
    at `1e-7`.
  - A failure measures what $T$'s two errors cost a total beyond what it declared. That is a
    finding about the rows before this prompt, not a fault in it.
- **T4 — the declared error is honest after the change.**
  - **The rule:** on every one of the 31 depth-20 rows with both $t$ units in budget,
    $|{\rm total}_t(10^{-7}) - {\rm total}_t(10^{-8})| \le$ `total_abserr`$_t(10^{-7})$.
  - 05's L3, the same rule under $c$, held on 29 of 31.

**Descriptive, not scored:**
- $|{\rm total}_t - {\rm total}_c|/|{\rm total}_c|$ and $/$`total_abserr`$_c$ on every row;
- regions and maximum depth per group, $c$ against $t$, with the `Gqr` groups and the low-$z$ end
  of each sub-interval shown apart;
- `total_converged` under $c$ and $t$;
- **the kinds of non-convergence** under $t$ at both tolerances, by the orchestrator's
  `prompts/levin-phase-floor/measure/convergence_kinds.py`, copied unchanged, recorded by SHA-256,
  and run as `convergence_kinds.py <this run> --tree t --capped-run <05's run>`. It reads the
  `checkpoint.jsonl` and `unit_*.pkl` layout of 05's run, which the harness keeps. The same
  measurement under 05's $c$ is `prompts/levin-phase-floor/measure/out/convergence_kinds.txt`.

## 7. Acceptance

1. §1–§3 as written, with every test passing, the four breakages caught and the seven roots
   rising only by the tests added, less the one replaced.
2. §5:
   - R11 and R12 both end `done` with fingerprint `1b332ffa…`;
   - every value table outside `QuadSourceIntegral` is identical, and every other difference
     there is of the four kinds;
   - the `QuadSourceIntegral` differences are tabulated;
   - the readers exit 0;
   - the lock is released.
3. §6: T0 holds; T1–T4 are each scored holds, fails or undecided against the rule as written, with
   their numbers; the store is unchanged.
4. **Records.**
   - `logs/05a-reanchor-t-phase.md`, with §5 and §6's Mac Studio record appended by the
     orchestrator;
   - `docs/levin-phase-floor/VERIFICATION.md` §5a "Layer 1(a) for $T$";
   - the records in `docs/levin-phase-floor/05a/`.

## 8. Stop conditions — stop and ask

- The Gate fails.
- The construction needs a change to `PrimitivePhase`, `CumulativeTable`, $G$'s phase, a producer,
  a schema or `AdaptiveLevin/`.
- `raw_theta` or `theta_deriv` of $T$ cannot be kept bit-identical.
- An existing assertion other than the one §2 names has to change.
- A seven-roots count falls.
- R11 or R12 does not give `1b332ffa…`.
- Outside `QuadSourceIntegral`, a value table differs, or a column differs in a way that is not one
  of the four kinds.
- T0 fails.
- Any store file changes.
- The lock is held, or will not release.

## 9. T1 amended before §6 runs (added 2026-10-26)

§1–§4 landed as `1fc349c` and passed the orchestrator's review. In writing test 3, the agent found
that T1's rule cannot be scored as written, for two reasons that are not the phase. Its log records
both, as observations 1 and 2.
- **An unresolved window measures `numpy.unwrap`.** A window of relative width $10^{-6}$ moves
  $\theta_T$ by about $10^{-6}\tfrac{z}{1+z}|d\theta_T/du|$. Where $|\theta_T| \gtrsim 10^9$ and
  $z \gtrsim 1$ that is hundreds to thousands of rad across 65 samples. `unwrap` then folds by whole
  cycles per index, and the rounding of the sampled doubles turns the folds into
  $\sigma_\theta \approx 0.01$–$0.1\,\varepsilon|\theta_T|$. The remainder was exact to
  $8\times10^{-14}$ rad when this was measured.
- **Between table nodes, the node-anchored remainder keeps $\varepsilon k|p_z|$.** That is the
  rounding of its small part $v = -k p_z + \phi$, which test 2's bound
  $4\varepsilon(k|p_z| + 2\pi)$ already admits. For $G$ it was negligible beside
  $\varepsilon k\eta_R$. For $T$, $|p_z|$ reaches half a table interval of $\tau_s$, about
  $0.0115\,|d\theta_T/du|/k$, so it is a fixed fraction of $\varepsilon|\theta_T|$: measured at
  $7.5\times10^{-3}$ midway at $z \approx 10^4$, and $2.8\times10^{-4}$ midway at $z \approx 0.12$.
  T1's $10^{-3}$ is below the first. Test 3 holds as written because its window sits on a node
  (log, deviation 4). It stands.

**The user's decision (2026-10-26): amend T1 before §6 runs.** No §6 number exists: R11, R12 and
the re-measurement have not started, and no probe of the 39 rows has been made. The amendment
changes **T1 only**. T0, T2, T3, T4 and the descriptive items are unchanged.

**The probe window, amended.** At each sub-interval's low-$z$ end, the window starts at relative
width $10^{-6}$ in $z$, as §6 says. Then:
- it is sampled under both $c$ and $t$. If the largest step of the unwrapped samples is $\pi/2$ or
  more under either, the width is halved and the window sampled again;
- this repeats until both trees' largest step is below $\pi/2$. The width used and that step are
  recorded per window;
- if the 65 samples would stop being distinct doubles first, the window is **unresolved**. It is
  reported and not scored;
- the $u$-sampled $\sigma_\theta$ uses the same width, in $\log(1+z)$.

Record $\max|p_z|$ over the window's 65 samples beside it: under $t$, from $T$'s `cs_tau.end_partial`
at each sample.

**T1, amended.** It replaces §6's T1. The $u$-sampled clause is unchanged.
- **The rule:** in every resolved probe window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad,
  sampled in $z$: $\sigma_\theta \ge 0.1\,\varepsilon|\theta_T|$ under $c$, and
  $\sigma_\theta \le \max\big(10^{-3}\varepsilon|\theta_T|,\ \varepsilon(k\max|p_z| + 2\pi),\ 10^{-13}\ {\rm rad}\big)$
  under $t$.
- **Holds** if true in every such window. **Fails** if false in any. **Undecided** if none is
  false but some window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad is unresolved.
- **Also report, unscored,** in how many windows the term $\varepsilon(k\max|p_z| + 2\pi)$ is the
  largest of the three, and $\sigma_\theta/(\varepsilon|\theta_T|)$ under $t$ in those windows.

**In the records.** `state/predictions.md` quotes §6's predictions with this section's T1 in place
of §6's, and says so. The log's **Predictions and verdicts** keeps §6's T1, marked "replaced before
the run by §9", with this section's T1 after it.
