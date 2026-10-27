# Prompt 13 — B2: score the residue the gap was masking

**Campaign:** [`README.md`](README.md) · **Board item:** **B2** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-27, on the user's request of the same day. B1 and A2, its two prerequisites
(README §4 item 2), have landed; so has A4, which re-took A2's representation term at `HEAD` and
left it for B2, without a verdict.
**Where it runs:** on the laptop, in a git worktree of its own, Ray-free, with no datastore and no
lock. **The Mac Studio is not needed.** With production's $G$ phase, A4's arm P took 0.5–13 s of
integral per realistic cell over A2's whole ladder (`GK-PHASE-COST-GROWTH.md` §2.3), where A2's
`phase_spline` cells took hours. Orchestrated from the laptop
([`orchestrator/prompt-13.md`](orchestrator/prompt-13.md)).
**In two parts, two commits** (B1's precedent: a code commit and a records commit):
- **Part A (§1): make the realistic flavour production's.** Test fixtures and comments only. One
  commit.
- **Part B (§2–§3): the measurement.** One registered, detached run, then one records commit.

**Closes, in the records commit:**
- `[02-realistic-fixture-Gk-phase-is-the-superseded-construction]`, by Part A;
- `[04-table-cs-tau-fixture-does-not-cover-z-init-when-the-first-sample-is-dropped]`, by Part A;
- `[12-realistic-large-x-reads-the-retired-clamp-records]`, by Part A's comments and Part B's
  successor script (§1.5).

**Narrows:** `[04-production-phase-gk-tau-table-has-a-zero-width-bottom-interval]` (its fixture
half; whether `CumulativeTable` should refuse such a pair is not B2's). On their own boards:
`[05-numeric-region-is-now-the-accuracy-floor]`, `[06-source-spline-residual-vs-handover]` and
`[07-lg-derivative-truncation-at-handover]` (`source-remediation`), and
`[00-tk-lg-truncation-floor]` (`GkTk-remedial`). These four are the README §3 table's B2 row.
**Recommended model:** **Opus**.

**Read first.** Every path is read at the dispatch SHA.
1. Campaign [`README.md`](README.md) §0.4, §2 (b), (g), (j), (k), (m), (n), §3 (workstreams B and
   E), §4, §5 and §5.1.
2. [`docs/handover/REALISTIC-LARGE-X.md`](../../docs/handover/REALISTIC-LARGE-X.md): §0, §1, §4, §5
   and §8. A2's factorial, its two terms, and why its representation term was an upper bound.
3. [`docs/handover/HANDOVER-B1.md`](../../docs/handover/HANDOVER-B1.md) §2, in full. **§2.2 is the
   reason for this prompt**: after B1, `together` at $x_{\rm resp}=980$ is left at
   $N+9/8 = +3.47\times10^{-3}$, larger than A2's representation term, and "not attributed here.
   It is B2's to attribute".
4. [`docs/handover/GK-PHASE-COST-GROWTH.md`](../../docs/handover/GK-PHASE-COST-GROWTH.md) §0, §5
   (Table H) and §6. Table H shows that **$G$'s phase construction moves $N$ by at most about
   $2\times10^{-5}$** (S against P) while the residue is $10^{-5}$–$3.6\times10^{-3}$. So most of
   what is left is not $G$'s phase, and B2 has to say what it is.
5. This board's §3: the three issues B2 closes and the one it narrows, in full, with their
   narrowings. `docs/handover/a4/issues.md`.
6. `ComputeTargets/tests/test_quadsource_integral.py`: the module docstring, `FakeQuadSource`
   (`:146`), `OffsetBesselPhaseGk` (`:217`), `ExactTkFunctions`, `exact_Tk_inputs`, `Case`
   (`:340`) and `production_seam_inputs` (`:934`).
7. `ComputeTargets/tests/test_phase_groups.py`: `_ExactPhase`, `ExactTk`, `ExactGk` and
   `BesselPhaseGk` (`:300`, whose docstring is the false claim).
8. `ComputeTargets/tests/test_g_phase_reanchor.py`: `_TableCsTauFixture` (`:448`), `_Value` and
   `_ProductionPhaseGk` (`:664`). `ComputeTargets/tests/test_t_phase_reanchor.py`:
   `_table_fixture_functions` (`:288`).
9. `docs/handover/a4/scripts/a4_cells.py`, **all of it**: `_UnionTableCsTauFixture` (the table
   construction the user approved at A4's stops), the $G$ grid's last node set exactly to
   $z_{\rm resp}$, `below_anchor_check`, the wrapper of `phase_group_Levin_integral`, and the
   reads of `partition["handover"]`. Then `a4_launch.py` and `a4_driver.py`: **the pattern to
   follow** for a registered, detached, Ray-free run with one child process per cell.
10. `ComputeTargets/GkSourcePolicyData.py`, `_build_phase`. `ComputeTargets/TkSourceFunctions.py`:
    the module docstring and the public accessors (`T`, `dT_dz`, `M`, `dlnM_dz`, `omega`,
    `T_WKB`, `phase`, `sin_coeff`, and the region properties). Both consume the producers'
    stored residual since `levin-phase-floor` 06 (`dc9d28c`).
11. `ComputeTargets/QuadSourceIntegral.py`: `build_partition`, `evaluate_QuadSource_integral` and
    `phase_group_Levin_integral`; and `ComputeTargets/phase_groups.py`. You need to know **which
    accessor of which factor is called in which sub-interval regime** (§2.2).
12. `prompts/levin-phase-floor/IMPLEMENTATION_STATE.md`,
    `[06-levin-accepts-a-region-across-a-stationary-point]`, for §2.5's scan.
13. `CLAUDE.md`, "Long-running jobs — the run registry", rules 1–7.

**It changes no production code.** Part A edits test fixtures and four comments; Part B adds
`docs/` scripts and one document.

**Gate** (check, and save the output in the log):
- you are in the worktree the orchestrator names, at the dispatch SHA, and `git status --short`
  is empty there;
- `./venv/bin/python -m runregistrykit list` shows nothing `running` or stale on this host;
- `sha256sum docs/handover/realistic_large_x.py` begins `c1cd3598c23a`. It is A2's record, and
  nothing here edits it;
- the seven roots' counts, taken by the orchestrator at the dispatch SHA, are in hand.

---

## 0. What this prompt is

### 0.1 The question

A2 split the error at the seam into two terms: the clamp, $3.9\times10^{-3}$–$0.91$, and a
representation term, $4\times10^{-5}$–$2.7\times10^{-3}$. B1 removed the clamp. **What is left is
the representation term, now unmasked, and nobody has said what it is made of.** README §3 asks B2
to "re-run A2 at the same configurations and attribute what is left, by term".

"The representation term" is a single number per cell, $N_{\rm realistic}-N_{\rm exact}$: every
way in which the realistic flavour's ingredients differ from the exact ones, at once. Its parts
are the open issues the README's B2 row names, and they have different remedies:

| part | where it lives | issue | its remedy, if it dominates |
|---|---|---|---|
| the source spline | `QuadSource`'s spline of $f$ | `[06-source-spline-residual-vs-handover]` | denser source grid, or E1/E2's depth |
| the numeric $T$ spline | `TkSourceFunctions`' numeric branch | `[05-numeric-region-is-now-the-accuracy-floor]` | `source_samples_per_log10z` |
| the stored WKB representation of $T$ | its amplitude and its `PrimitivePhase` | (05a, 06) | — |
| the Liouville–Green closed forms | $\omega$ and ${\rm d}\ln M/{\rm d}z$ | `[07-lg-derivative-truncation-at-handover]`, `[00-tk-lg-truncation-floor]` | a deeper hand-over only (E2) |
| $G$'s WKB representation | its phase and amplitude | `[02-realistic-fixture-…]` | — |

B2 measures each part. **That is the deliverable**, and it is what E1 and E2 are planned on.

### 0.2 Why the instrument has to be fixed first

The realistic flavour is documented as "the accuracy production can expect", and at the dispatch
SHA it is not production's in three ways:
- its $G$ phase is a raw `phase_spline`, the construction `_build_phase` abandoned at
  `GkTk-remedial` 09 (`[02-realistic-fixture-…]`);
- its $T$ is built with `exact_remainder=False` on a closed-form `cs_tau`, where production since
  `levin-phase-floor` 05a builds it on a table `cs_tau` with the keyword at its default (A4 §1.1);
- its seam is not production's: `Case` uses `Fixture`'s default, with a Liouville–Green sample at
  $z_{\rm init}$, where production's first WKB sample is the grid node below it
  (`drop_first_WKB_sample=True`, as `production_seam_inputs` builds).

A4 built production's $G$ and $T$ in its own script and found two defects in the test fixtures on
the way (the two `[04-…]` issues). Part A moves that construction into the fixtures, so that
`Case(exact=False)` **is** production's construction, and Part B's script builds on `Case` rather
than carrying a third copy.

### 0.3 What is already known, and what is not

- After B1, at $x_{\rm resp}=980$: `T-first` $-6.0\times10^{-5}$, `q-smooth` $-5.4\times10^{-5}$,
  `together` $+3.47\times10^{-3}$ (HANDOVER-B1.md §2.2). B1's column-by-column table moves one
  thing at a time and already shows the two spline end intervals' share at the seam.
- At `HEAD` with production's $G$ (A4 Table H, before 06): the residue over A2's ladder is
  $8\times10^{-6}$–$3.6\times10^{-3}$, and S against P is at most $\sim2\times10^{-5}$.
- **Not known:** how `together`'s $3.5\times10^{-3}$ divides among the five parts; how each part
  scales with $x_{\rm resp}$; whether the parts add.

---

## 1. Part A — make the realistic flavour production's

Work only in the worktree. Every command runs there, with `PYTHONPATH=.`.

### 1.1 $G$

- **`Case(exact=False)`'s Green's function** has its phase built by `GkSourcePolicyData._build_phase`
  itself, as `_ProductionPhaseGk` builds `.phase`: a `PrimitivePhase` on a `CumulativeTable` of
  $1/H$, anchored at $z_{\rm resp}$, from stored-sample stand-ins carrying `theta_residual`.
  Its amplitude stays `bessel_phase`'s modulus, and `numeric_Gk` stays `compute_analytic_G`;
  record both as what is still not production's (§3, §6 of the document).
- **The zero-width interval** (`[04-production-phase-gk-tau-table-…]`). The table's nodes must be
  distinct in $\log(1+z)$. Fix it in the fixture, by making the grid's last node exactly
  $z_{\rm resp}$ (A4's run-2 fix) or by de-duplicating in $\log(1+z)$; state which. A point one ulp
  below the anchor must evaluate. **`CumulativeTable` is not changed.**
- **`BesselPhaseGk` and `OffsetBesselPhaseGk`.** The issue's next step is to rebuild them on
  `PrimitivePhase` as `_build_phase` does. Do that. `OffsetBesselPhaseGk` exists only because a
  `phase_spline` refuses mixed-sign samples, so it may become unnecessary. If a test exercises the
  `phase_spline` construction **as such**, keep that construction for it under a name that says
  it is the superseded one, and say so in the log. `_ProductionPhaseGk` reuses the shared
  construction or is reduced to it; `old_phase` (the keyword off) stays available, since
  `levin-phase-floor` 05's tests read it.
- **The docstrings.** `test_phase_groups.py:300-306` and `test_quadsource_integral.py`'s module
  docstring say what the realistic flavour now is, and name what is still not production's:
  the amplitude, `numeric_Gk`, the uniform 100-per-decade grid, $f$ from exact samples, and the
  fixed hand-over depth.

### 1.2 $T$

- **`Case(exact=False)`'s transfer functions** are built as production builds them since 05a:
  `TkSourceFunctions(fixture.exact_envelope_model(), k, Tn, Tw)` with **no keyword**, on a
  table `cs_tau`.
- **`_TableCsTauFixture`** (`[04-table-cs-tau-…]`) builds its table on nodes that cover
  $z_{\rm init}$ with $z_{\rm init}$ **off a node**, as production's background grid does. A4's
  `_UnionTableCsTauFixture` is the construction the user approved at A4's stops: the union of the
  fixture's numeric and WKB nodes without the one numeric node equal to $z_{\rm init}$. Adopt it
  into `_TableCsTauFixture`, or justify another in the log. The `levin-phase-floor` 05a tests that
  use the fixture must still pass.

### 1.3 The seam

`Case(exact=False)` uses production's seam: the Liouville–Green samples begin at the largest grid
node below $z_{\rm init}$ (`drop_first_WKB_sample=True`), and the numeric samples continue past
it (`Fixture`'s default since B1). **`Fixture`'s own default is not changed**, as B1 required.

### 1.4 Tests

The stand-in pattern; no Ray, no datastore.
1. **New tests**, each named in the log:
   - (a) **construction:** on one cheap `Case(exact=False)`, $G$'s phase is a `PrimitivePhase`
     with `exact_remainder` true; each $T$'s phase is a `PrimitivePhase` on a `TablePrimitive` with
     `exact_remainder` true; each $T$'s `WKB_region[0]` lies strictly below its `z_init`;
   - (b) **the table covers $z_{\rm init}$:** `_TableCsTauFixture` builds, and its $T$ evaluates at
     $z_{\rm init}$, at one configuration A4 recorded raising (A2's ladder, $10^4$–$10^6$; choose
     the cheapest), and $z_{\rm init}$ is not a node of its table;
   - (c) **no zero-width interval:** the $G$ construction on a grid whose last node is one ulp
     below $z_{\rm resp}$ builds, and evaluates one ulp below the anchor.
2. **Existing tests whose numbers move.** Every test that prints or asserts a realistic-flavour
   number is listed in the log with old and new values, as HANDOVER-B1.md §4 lists B1's. **A test
   that fails is a stop.** Do not change its tolerance: report the margin.
3. **Deliberate breakage**, each recorded and reverted:
   - put `Case`'s $G$ back on a `phase_spline`: (a) must fail;
   - build `_TableCsTauFixture`'s table on the dropped `z_WKB` again: (b) must fail;
   - restore the duplicate node: (c) must fail.
4. **The seven roots** at the end, serially. A count that falls is a stop; a rise equals the tests
   added.

### 1.5 Comments, and A2's harness

- **Four stale comments** name the clamp B1 removed. Correct them, **comments only**:
  `ComputeTargets/primitive_phase.py:113` and `:175` (`QuadSourceIntegral._ClampedPhase`),
  `ComputeTargets/tests/test_primitive_phase.py:416` (the same), and `main.py:1890` ("stores end
  at the larger of the two Tk hand-over redshifts"). Re-run
  `grep -rn '_ClampedPhase\|_ClampedTk\|_ClampedSource\|clamp_gaps\|max_clamp_gap\|HANDOVER_CLAMP' --include='*.py' .`
  outside `venv/`, `docs/` and `prompts/`. Any hit left is explained in the log; the existing
  `assertNotIn` checks in `test_quadsource_integral.py` are expected.
- **`docs/handover/realistic_large_x.py` is not edited.** It is A2's record: its tables'
  checkpoint carries its SHA-256 (`c1cd3598…`), and half its factorial, the open seam, cannot be
  built at `HEAD`, where B1's strict coverage check refuses a gap. It runs at A2's tree
  (`0d7c05c`). Part B's script is its successor at `HEAD` and carries the `partition["handover"]`
  reads. The same holds for `docs/levin-phase-floor/noise_injection.py` and `docs/handover/a4/`'s
  scripts, which import the fixtures Part A changes: each is a record of its own SHA. List them
  in the log.

### 1.6 Part A's commit

`black` on every touched `.py` file. **One commit on the worktree's branch**, staged by explicit
path: the test modules under `ComputeTargets/tests/`, the four comment edits, the log
`logs/13-score-the-unmasked-residue.md` (Part A's half), and the board (B2's row and item, marked
"Part A landed"). **No issue moves in this commit**; they move in the records commit. No push, no
merge, no rebase. Then **stop and report**: the orchestrator reviews Part A before Part B starts.

---

## 2. Part B — the measurement

### 2.1 The script

A successor of `realistic_large_x.py` at `HEAD`, written under the worktree's
`var/handover-13/scripts/` (gitignored, so the tree stays clean at launch) and copied back to
`docs/handover/b2/scripts/` byte-identical in the records commit. It **imports** from
`realistic_large_x.py` what still holds (`RUNS`, `scaled_shape`, `fixture_x_max`, `seed_fixtures`,
`reference_of`, `REF_ATOL`, `REF_RTOL`) and builds every case through `Case`. It reads
`partition["handover"]` where A2 read the clamp records. Its docstring says that after landing it is
a record of its own SHA-256. Cell, driver, launcher and tables in the shape of A4's four scripts.

### 2.2 The objects, and the five terms

Every cell at a rung shares one shape, one $\lambda$ scaling, one set of fixtures, one 50-digit
reference (`reference_of`) and the tolerances `REF_ATOL = 1e-45`, `REF_RTOL = 1e-12`.

| object | what it is |
|---|---|
| **E** | `Case(exact=True)`: A2's exact flavour, its own geometry. The control |
| **R** | `Case(exact=False)` after Part A: production's construction, production's seam |
| **E′** | R's objects, with **every** accessor that differs from the exact flavour replaced by the exact one. R's geometry (regions, switch nodes, partition), exact ingredients |
| **E′ + i** | E′ with term $i$'s accessors realistic, the rest exact |
| **R − i** | R with term $i$'s accessors exact, the rest realistic |

**The five terms.** A hybrid is R's object with a chosen set of accessors delegated to the exact
counterpart (`ExactTkFunctions`, `ExactGk`, `FakeQuadSource(exact=True)`), applied to $T_q$ and
$T_r$ together. For $G$, `_replace` on the policy's `GkSourceFunctions`, as A4 did.

| term | made realistic | accessors, by default | bears on |
|---|---|---|---|
| `f` | the source spline | `QuadSource.functions.source` | `[06-source-spline-…]` |
| `Tnum` | the numeric $T$ branch | `T`, `dT_dz` where the factor is numeric | `[05-numeric-region-…]` |
| `Trep` | $T$'s stored WKB representation | `M`, `phase`, `sin_coeff` | 05a, 06 |
| `TLG` | the Liouville–Green closed forms | `omega`, `dlnM_dz` | `[07-lg-derivative-…]`, `[00-tk-lg-truncation-floor]` |
| `G` | $G$'s WKB representation | `phase`, `sin_amplitude` | `[02-realistic-fixture-…]` |

The "accessors" column is a starting point, not a specification. **Read Read-first item 11 and
fix the mapping from the code**: which accessor each regime calls, and whether an accessor such as
`T` dispatches to the WKB form below the switch node, in which case `Tnum`'s delegation is by
region as well as by name. Record the final mapping, by accessor and region, in
`state/terms.md` (§2.6). **Two checks prove it, and both are bit-for-bit in `total` and
`total_abserr`:**
- **completeness:** E′ + all five ≡ R, and R − all five ≡ E′;
- **disjointness:** no accessor is in two terms (shown from the mapping, by listing).

A term that cannot be isolated by delegation without a production change is a **stop**: do not
merge it into another and keep both names.

### 2.3 What is computed from the cells

At each rung, with $N \equiv -\tfrac98\,{\rm total}/{\rm predicted}$ as A2 defines it:
- the **residue** $D = N_R - N_{E'}$;
- the **add-one** term $A_i = N_{E'+i} - N_{E'}$ and the **remove-one** term
  $B_i = N_R - N_{R-i}$;
- the **interactions** $\sum_i A_i - D$ and $\sum_i B_i - D$;
- the **geometry control** $N_{E'} - N_E$: R's partition with exact ingredients against the exact
  flavour's own.

Each with its error, the sum of the cells' declared errors converted to $N$ (A2's `term`). The
50-digit reference is common to every cell of a rung and cancels in every difference (A2 §0), so
it contributes nothing to these errors. $N_R$ and $N_E$ against the reference carry its own error
beside them (README §5 rule 8).

**No threshold.** A rung where $|\sum_i A_i - D|$ or $|\sum_i B_i - D|$ exceeds the largest
$|A_i|$ is reported as **not additive at that rung**, and the document does not give a per-term
attribution there without saying so.

### 2.4 The cells

**The ladder** is A2's `RUNS`, all of it: `together` at $980$, $10^4$, $10^5$, $10^6$, $10^7$ and
$10^8$; `T-first` at $980$ to $10^7$; `q-smooth` at $980$ to $10^7$, including the two rungs A2
could not reach. That is 16 rungs × 13 objects (E, E′, R, five add-one, five remove-one) = **208
cells**.

**Each cell runs in a child process**, with a wall-clock budget of **one hour**. A cell over budget
is killed and recorded `not reached`, with its elapsed time: a result, not a failure. **At most two
cells at once**: the laptop carries other projects' work. Times are indicative only, with the load
average beside each; $N$, region counts and depths are the measurement. Schedule cheapest first:
the exact objects, then rung by rung upward.

**What each cell records.** A4's fields: per phase group and sub-interval, `num_regions`,
`max_depth`, `num_phase_limited_regions`, `converged`, value and abserr (through the same transparent
wrapper of `phase_group_Levin_integral`); per cell `Levin_regions`, the maximum depth and its group,
`total`, `total_abserr`, the reference, `setup_time`, `integral_time` and the load; the
`partition["handover"]` record; and **which accessors were delegated**, with each factor's phase
class and `exact_remainder`.

### 2.5 Before the launch

Each saved under `var/handover-13/state/`:
1. **Construction:** R's objects are what Part A's test (a) asserts, at every rung.
2. **Completeness and disjointness** (§2.2), at one cheap rung, `together` at 980.
3. **Determinism:** two in-process runs of E, E′ and R at `together` 980 are bit-identical in
   `total`, `total_abserr` and every group's region count and depth.
4. **The wrapper is transparent**, as A4 showed it.
5. **The controls:**
   - E reproduces A2's exact closed column (REALISTIC-LARGE-X.md Table 2) at every rung, within the
     sum of the two declared errors. Report the digits that agree;
   - $N_{E'} - N_E$ at `together` 980 is within the sum of the declared errors. If it is not, stop:
     R's partition then contributes a term of its own, and §2.3's decomposition does not hold;
   - R at `together` 980 against A4's arm P ($+3.623\times10^{-3}$, at `28ebaf0`, before 06):
     recorded, not judged. The difference is `levin-phase-floor` 06's and Part A's together.
6. **The stationary-point scan.** For every rung, every Levin sub-interval of R's partition and
   every phase group in it, the group's $\theta'$ from the exact closed forms (`_ExactPhase.theta_deriv`
   of each factor, combined with the group's signs), sampled densely over the sub-interval: the
   minimum and maximum of $|\theta'|$, and **any sign change**. At $b=0$ the sound speed is
   constant, and the expectation is none; `T-first`'s $G\pm q\mp r$ groups are the nearest
   approach. **A sign change is a stop**: that cell would fall under `levin-phase-floor`'s
   `[06-levin-accepts-a-region-across-a-stationary-point]`, which is not fixed.
7. **`state/terms.md`:** §2.2's final mapping and §2.3's rule for additivity, written after checks
   1–6 and **before** the launch. Its mtime against the run's manifest is the review's evidence that
   the terms were fixed before any number was seen.

### 2.6 The run

Registered, as `CLAUDE.md` rules 1–7 require and as A4's launcher does:
- `runregistrykit.begin` before anything runs, campaign `handover`, prompt `13`, slug `residue`.
  A run launched from a worktree is recorded in the main checkout's `var/runs/`;
- detached (`nohup sh -c '…; exit $?'`, a new session), stdout and stderr in the run directory,
  the driver's pid recorded with `Run.heartbeat(pid=…)`;
- one checkpoint record per cell, carrying the script's SHA-256, `HEAD` and `git status --short`.

**Verify in one check that it started, and end the turn.** Do not poll. The orchestrator resumes
you when it has ended.

---

## 3. What to write down

`docs/handover/SEAM-RESIDUE.md`, written in the worktree after the run, in the shape of
REALISTIC-LARGE-X.md:
- **§0 — the objects,** in `KOHRI-TERADA-ORACLE.md` §0's shape: E, R, E′ and the hybrids, whose
  each is, the tree, and what each comparison guards and does not.
- **§1 — the headline:** the residue and its five terms, in one table of ranges over the ladder,
  each with its error; whether they add; which term carries `together`'s $3.5\times10^{-3}$.
- **§2 — the controls** of §2.5 item 5, and the stationary-point scan.
- **§3 — the tables,** generated by the tables script with a `<!-- generated by … -->` marker:
  $N+9/8$ per object and cell; $D$, $A_i$, $B_i$ and both interactions per rung, with errors; the
  cells `not reached`.
- **§4 — the attribution,** per term and per shape: its size against the residue, its sign, and
  its $x_{\rm resp}$-scaling (a fitted slope where the term is monotone, as A2's Table 4; say where
  it is not). Then, for each issue of the header's "Narrows", one paragraph: what this measures of
  it, on this fixture.
- **§5 — against the earlier records,** each labelled with its tree: A2's representation term
  (pre-B1), HANDOVER-B1.md §2.2 (B1 at 980), A4 Table H (pre-06). Context, not columns of the
  measurement.
- **§6 — cost,** recorded without a reading: R's regions and depths per rung beside A4's arm P.
- **§7 — what this does not cover.** At minimum: $b=0$ only; the hand-over at the fixture's fixed
  $x_T \approx 19.1$, so nothing here is a depth scan (E2); the fixture's uniform 100-per-decade
  grid, not production's curvature-criterion grid; $G$'s numeric branch exact
  (`compute_analytic_G`) and its crossover at the shape's `G_cross_x`, not production's
  super-horizon node (`[03-the-g-crossover-is-one-super-horizon-node-on-every-complete-row]`), so
  D1's consumer spline is not measured; $G$'s amplitude from `bessel_phase`; $f$ splined from exact
  samples; $x_{\rm resp} \le 10^8$, not production's $\sim4\times10^{12}$; the Levin driver before
  `levin-phase-floor` 07.

Beside it, under `docs/handover/b2/`: `scripts/`, `out/` (the tables' output and the state files),
and `issues.md` if anything is opened.

---

## 4. What this prompt does not do

- No production code changes beyond §1.5's four comments: not `QuadSourceIntegral.py`,
  `TkSourceFunctions.py`, `GkSourcePolicyData.py`, `CumulativeTable`, `AdaptiveLevin/`, `config/`
  or a factory.
- It does not edit `realistic_large_x.py`, `REALISTIC-LARGE-X.md`, `HANDOVER-B1.md`,
  `GK-PHASE-COST-GROWTH.md` or any script under `docs/` that is a record.
- It does not choose a hand-over depth, a grid density or a remedy for any term. That is E1, E2
  and D, on these numbers.
- It does not run $b = 0.2$. A1's oracle makes that possible; it is not B2's configuration.
- No Ray, no datastore, no lock, no write to the Mac Studio.

---

## 5. Acceptance

**Part A**
1. §1.1–§1.3 as written; the docstrings say what the realistic flavour is and what it is not.
2. Tests (a)–(c) pass; the three breakages are caught and reverted; every moved number is listed.
3. The seven roots: none falls, and `ComputeTargets` rises by the tests added.
4. §1.5's grep is clean or explained; `realistic_large_x.py` still hashes `c1cd3598…`.

**Part B**
5. One command launches the run; it is registered and ends `done`, every cell `ok` or
   `not reached`.
6. §2.5's checks were saved before the launch, and `state/terms.md` is older than the manifest.
7. Completeness holds bit for bit; the geometry control holds; E reproduces A2's exact column.
8. $D$, every $A_i$ and $B_i$, and both interactions per rung, with errors; additivity reported per
   rung; `together`'s 980 residue attributed or reported not additive.
9. `SEAM-RESIDUE.md` with §0–§7.
10. Board, issues and `docs/OPEN_ISSUES.md` per §7.

---

## 6. Stop conditions — stop and ask

- The Gate fails.
- An existing test fails after Part A. Report the margin; do not loosen it.
- A fixture cannot be made production's without a production change.
- A term cannot be isolated by delegation (§2.2), or completeness fails.
- The geometry control fails, or E does not reproduce A2's exact column within the declared errors.
- The stationary-point scan finds a sign change.
- A determinism check fails, or the wrapper is not transparent.
- A cell raises for any reason other than its budget. Record it, and stop before that shape's next
  rung.
- Anything would need Ray, a datastore, the lock or the Mac Studio.

---

## 7. The log, the board and the commits

`logs/13-score-the-unmasked-residue.md`, in `qcd-background-audit` README §5.1's form, classifying
every deviation (`STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE`, `UNINTENDED DRIFT`). Part A writes
its half; Part B completes it. Beyond the template:
- the Gate's output; the seven roots before and after, with `uptime`;
- Part A's moved numbers, the breakage record, and the records-at-their-SHA list of §1.5;
- §2.5's outputs, `state/terms.md`'s time against the launch, and the run directory;
- the cells `not reached`, with their elapsed times.

**The records commit**, in the worktree, after the run, on the same branch:
- `docs/handover/b2/` and `docs/handover/SEAM-RESIDUE.md`; the log completed;
- the board: B2's row and item, §3 and §4. The three closed issues move to §4. The narrowing of
  `[04-production-phase-gk-tau-table-…]` is recorded on it;
- narrowing notes on `[05-…]`, `[06-…]` and `[07-…]` on the `source-remediation` board and on
  `[00-tk-lg-truncation-floor]` on the `GkTk-remedial` board, each dated and pointing at
  SEAM-RESIDUE.md §4;
- `docs/OPEN_ISSUES.md`: rows deleted, narrowed or added, the count and the date corrected **by
  counting**;
- a §3 issue for anything found and not fixed.

`black --check` on every new `.py` file except the copied scripts, which are records and are
listed in `CLAUDE.md`'s formatting paragraph in the same commit if they are not `black`-clean.
Commit messages per `CLAUDE.md`. No push, no merge: landing is the orchestrator's.
