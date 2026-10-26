# Prompt 04 — does A2's Levin cost growth survive production's phases?

**Campaign:** [`README.md`](README.md) · **Board item:** **A4** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md). Campaign README §3 carries A4 too;
do not add it there again.
**Written:** 2026-09-21 as a decision test on A2's numbers. Refreshed 2026-10-25. **Rewritten
2026-10-26, never having run, on the user's decision the same day** (§1.1):
- it runs at `HEAD`, not at the pre-B1 tree;
- it measures the cost growth with three constructions of $G$'s phase;
- it fixes two readings of the result before the run, and no threshold for a "material" change.

**Where it runs:** on the Mac Studio, Ray-free, in the main checkout at the dispatch SHA,
orchestrated there ([`orchestrator/prompt-04.md`](orchestrator/prompt-04.md)). It commits nothing
there: its records go under `var/handover-04/`, and the laptop copies them back and commits them.
It takes no lock, so it may run beside another Ray-free prompt, or beside a build that holds the
lock.

**Uses:** no Ray, no datastore, no lock. One registered, detached run (§3).
**Closes:** nothing by itself. Its result decides whether
`[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` closes or narrows (§4). That is the
user's call, made in the records commit. It narrows
`[02-realistic-fixture-Gk-phase-is-the-superseded-construction]` and
`[12-realistic-large-x-reads-the-retired-clamp-records]`.
**Recommended model:** **Opus**.

**Read first.** Every path is read at the dispatch SHA.
1. [`docs/handover/REALISTIC-LARGE-X.md`](../../docs/handover/REALISTIC-LARGE-X.md): §5 (the
   $x$-scaling), §7 (cost), Table 8 and §8 items 5 and 6. Those are A2's cost curves, taken before
   B1.
2. This campaign's board §3: `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` in full,
   with both narrowings and the planning note; `[02-realistic-fixture-Gk-phase-is-the-superseded-construction]`;
   and `[12-realistic-large-x-reads-the-retired-clamp-records]`.
3. `prompts/levin-phase-floor/IMPLEMENTATION_STATE.md`: the status block and the Decisions G1,
   05a, 05a §9 and 05a T1. Then
   [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §4,
   §5 and §5a. They are why there are three arms, not two.
4. `docs/handover/realistic_large_x.py`, **all of it**, in particular `run_cell` (`:303`), its reads
   of the retired clamp records (`:332`, `:356`, and `:666` in the tables), `open_the_seam` and `seed_fixtures`.
5. `ComputeTargets/tests/test_quadsource_integral.py`: the module docstring, `OffsetBesselPhaseGk`
   (`:218`), `Case` (`:340`, the realistic branch at `:411-427` and `:450-466`) and
   `production_seam_inputs` (`:934`).
6. `ComputeTargets/tests/test_g_phase_reanchor.py`: `_TableCsTauFixture` (`:448`) and
   `_ProductionPhaseGk` (`:652`), which builds $G$'s phase both as production does (`.phase`) and with the
   keyword off (`.old_phase`). They are the two `PrimitivePhase` arms below, already built.
7. `ComputeTargets/GkSourcePolicyData.py`, `_build_phase` and its docstring; then
   `ComputeTargets/primitive_phase.py`'s module docstring, "Anchoring, and the floor it implies"
   and "Who uses which".
8. `ComputeTargets/QuadSourceIntegral.py`, `phase_group_Levin_integral`: its `groups` and
   `Levin_results`. Then `AdaptiveLevin/levin_quadrature.py`: what `adaptive_levin_sincos`
   returns (`max_depth`, `num_regions`, `num_phase_limited_regions`, `regions`).
9. For the registered run: `CLAUDE.md`, "Long-running jobs — the run registry", and
   `docs/levin-phase-floor/05/wider/scripts/launch.py`, a launcher of a registered detached
   Ray-free run on this host. **The pattern to follow.**
10. Campaign [`README.md`](README.md) §0.4, §2 (b), (l), §5 and §5.1.

**It changes no production code and no test file.** It adds `docs/` scripts and one document.
It edits neither `realistic_large_x.py` nor `REALISTIC-LARGE-X.md`, which are A2's records.

**Gate** (check, and save the output):
- the main checkout is at the dispatch SHA, and `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale;
- there is no `var/handover-04/`;
- `sha256sum docs/handover/realistic_large_x.py` begins `c1cd3598c23a`, the hash A2's cells
  carry. The file is imported, never edited.

---

## 1. Why this exists

A2 found the realistic flavour's Levin cost **growing steeply in $x$** where the exact flavour's is
flat (board §3, `[02-levin-cost-growth-…]`, from `REALISTIC-LARGE-X.md` Table 8):

| shape | exact flavour, regions | realistic flavour, regions |
|---|---|---|
| `together` | 7 → 9 over six rungs | 1,304 → 50,463 at $10^7$, 15,827 at $10^8$ |
| `T-first` | 7 → 13 | 720 → 75,570 |
| `q-smooth` | 17 → 30 | 855 → 31,358 by $10^5$, where it had to stop |

Two things are in that table, and they are separate:
- **a level:** the realistic flavour is already about 100× the exact one at the bottom rung. The
  board explains this by the spline knot scale;
- **a growth:** the realistic region count rises 12–105× across the ladder, while the knot count
  rises only 1.9×.

A2's fixture built $G$'s phase as a raw `phase_spline`, which production abandoned at
`GkTk-remedial` 09. That phase's floor, $h^4x/384$, grows linearly in $x$. So the growth might be
the fixture's, not production's. That is the hypothesis this prompt tests.

### 1.1 What has changed, and why three arms

- **B1 landed** (prompt 12, `049c7c8`). The seam is overlap by construction, and the clamp
  records A2's harness reads are gone, so `run_cell` raises at `HEAD`. A comparison with A2's
  recorded cells would now mix B1's change with the phase's. **So every cell here is computed at
  `HEAD`, every arm on the same tree, and A2's numbers are context only.**
- **`levin-phase-floor` 05 and 05a landed.** Production's $G$ and $T$ phases are now node-anchored
  remainders, reduced against $2\pi$ (`120fdfe`, `1fc349c`). Those prompts showed that the
  anchoring floor of a `PrimitivePhase` phase, about $\varepsilon k\eta$, sends the Levin bisection
  to its depth cap on A3 v2. So $G$'s phase has **two** candidate causes of growth, not one:
  1. the `phase_spline` floor, $h^4x/384$;
  2. the anchoring floor of a `PrimitivePhase` built without the node anchor, about
     $\varepsilon x$. The board's 2026-10-25 narrowing puts it above the harness's
     `REF_RTOL = 1e-12` from $x_{\rm resp} \approx 4.5\times10^3$.

  Three arms separate them (§2).
- **$T$ is now production's in every realistic arm.** At `HEAD` the fixture's `Tk_builder` passes
  `exact_remainder=False` on a closed-form `cs_tau` (`test_quadsource_integral.py:425-427`). That
  $T$ carries its own anchoring floor, which the board's narrowing names as a third explanation.
  Here every realistic arm builds $T$ on a table `cs_tau` with the keyword at its default, as
  production does since 05a, so that the arms differ in $G$ alone.
- **No threshold for a "material" change** (the user, 2026-10-26). The earlier prompt was a
  decision test on A2's record. At `HEAD` that decision no longer arises:
  - A2's numbers stay a pre-B1 record either way;
  - the fixture's $G$ is to be made production's in any case, because it is documented as "the
    accuracy production can expect".

  What remains is a measurement, with two readings fixed before it is taken (§4).

---

## 2. What to build

A sibling of `realistic_large_x.py`, written under `var/handover-04/scripts/` and copied back to
`docs/handover/a4/scripts/` byte-identical. It **imports** from `realistic_large_x.py` (`RUNS`,
`scaled_shape`, `fixture_x_max`, `reference_of`, `REF_ATOL`, `REF_RTOL`, and the like) and from
the test modules, and replaces only what it must:
- **`run_cell`'s clamp reads,** with reads of `partition["handover"]`, recording what B1
  records there;
- **the case construction**, per arm, as below.

State in its docstring that after landing it is a record of its own SHA-256.

### 2.1 The arms

Every arm runs the same shapes, the same $\lambda$ scaling, the same fixtures, the same 50-digit
reference (`reference_of`) and the same tolerances, `REF_ATOL = 1e-45` and `REF_RTOL = 1e-12`.

| arm | $G$'s phase | $T_q$, $T_r$ | seam |
|---|---|---|---|
| **E** | `ExactGk`, closed form | `ExactTkFunctions`, closed form | as the exact flavour |
| **S** | `OffsetBesselPhaseGk`'s `phase_spline`, A2's construction | production's: `TkSourceFunctions` on a table `cs_tau`, keyword at its default | production's |
| **P₀** | a `PrimitivePhase` on a `CumulativeTable` of $1/H$, keyword **off** (production before 05) | as S | production's |
| **P** | the same `PrimitivePhase` built by `_build_phase`, keyword **on** (production since 05) | as S | production's |

- **E** is the flat reference, recomputed at `HEAD`.
- **S against P₀** isolates the `phase_spline` floor. **P₀ against P** isolates the anchoring floor.
- **P's and P₀'s $G$** are built as `_ProductionPhaseGk` builds `.phase` and `.old_phase`, on the
  realistic `Case`'s own $G$ grid and samples. `Case` uses `OffsetBesselPhaseGk` because a
  `phase_spline` refuses samples of mixed sign at $z_{\rm response}$. If `_ProductionPhaseGk`'s own
  `phase_spline` (built by its `super().__init__`) raises on that grid, build the arm without it.
  The offset of whole cycles is immaterial to a `PrimitivePhase`. Only `.phase` may differ between
  S, P₀ and P: the amplitude and every other accessor are the same object.
- **$T$ in S, P₀ and P** is built as `test_t_phase_reanchor._table_fixture_functions` builds it:
  - the fixture is a `_TableCsTauFixture` seeded as `seed_fixtures` seeds a `Fixture`, with
    `drop_first_WKB_sample=True`;
  - its samples are captured with `exact_Tk_inputs`;
  - the object is built by `TkSourceFunctions(fixture.exact_envelope_model(), k, Tn, Tw)`, with
    no keyword.
- **The seam** in S, P₀ and P is production's since B1: the WKB samples begin at the largest grid
  node below $z_{\rm init}$ (`production_seam_inputs`, `test_quadsource_integral.py:934`). A2's
  closed and open seams no longer correspond to production's, and A2 found its cost the same on
  both, so one seam is run.
- **Each built object is checked before any cell runs, and the check saved:**
  - S's $G$ phase is a `phase_spline`;
  - P₀'s and P's are `PrimitivePhase`s with `exact_remainder` false and true;
  - every realistic $T$ phase is a `PrimitivePhase` on a `TablePrimitive` with `exact_remainder`
    true.

### 2.2 What every cell records

`run_cell`'s fields, with the clamp reads replaced, and from the driver's own per-call results:
- per phase group, per sub-interval: the group label, `num_regions`, **`max_depth`**,
  `num_phase_limited_regions`, `converged`, value and abserr;
- per cell: `Levin_regions` (the sum), the maximum depth and **which group carries it**, and the
  $z$-range of its deepest regions;
- `total`, its declared error and the reference, so that the representation term
  $N_{\rm arm} - N_{\rm E}$ can be formed for every arm;
- `setup_time`, `integral_time`, the host's load average, and each phase object's class and
  `exact_remainder`.

**How the per-group fields are taken.** Wrap `ComputeTargets.QuadSourceIntegral.phase_group_Levin_integral`
at its module attribute, which is how `evaluate_QuadSource_integral` calls it. The wrapper calls
the original unchanged and keeps its `groups` and `Levin_results` side by side. Do not instrument
`AdaptiveLevin/`. Show once, on one cell, that the wrapped and unwrapped runs give bit-identical
`total` and `total_abserr`.

---

## 3. The cells, and how they run

**The ladder** is A2's (`RUNS`), one seam, in every arm:
- `together` at $980$, $10^4$, $10^5$, $10^6$, $10^7$ and $10^8$;
- `T-first` at $980$, $10^4$, $10^5$, $10^6$ and $10^7$;
- `q-smooth` at $980$, $10^4$, $10^5$ **and $10^6$**, the rung A2 could not reach.

That is 15 × 4 = 60 cells. In A2, arm S's analogue took about 1.3 h over the first fourteen rungs.
`q-smooth` at $10^6$ was estimated at hours, from a run killed after 5,600 s. P₀ and P are
expected to cost at most what S costs.

**The budget.** Each cell runs in a child process of its own, with a wall-clock budget of
**3 hours**. A cell over budget is killed and recorded as `not reached`, with its elapsed time.
That is a result, not a failure. Up to **4 cells** run at once. Times are then indicative only,
and the load is recorded beside each; region counts and depths are the measurement.
Cells are scheduled cheapest first: arm E, then rung by rung upward.

**Before the launch:**
- **Determinism:** two in-process runs of one cheap cell per arm (`together` at 980) are
  bit-identical in `total`, `total_abserr` and every group's region count and depth.
- **The wrapper is transparent**, as §2.2 says.
- **The readings of §4** are copied into `var/handover-04/state/readings.md`, after those checks
  and before the launch. Its mtime is the review's evidence.

**The run** is registered (`CLAUDE.md` rules 1–7):
- a launcher in the pattern of Read first item 9;
- `runregistrykit.begin` before anything runs, with campaign `handover`, prompt `04` and slug
  `cost-growth`;
- started detached, with stdout and stderr in the run directory and the driver's pid recorded;
- one checkpoint record per cell, carrying the script's SHA-256 and the tree's `HEAD` and
  `git status --short`.

Verify in **one** check that it started, and end the turn. Resume when it has ended.

---

## 4. The readings, fixed before the run

There is no threshold. The measure is the **growth across the ladder** in each shape: the region
count at the highest rung reached over the count at 980, for each arm, set beside arm E's. The
level at 980 is reported apart from it. Copy this section into `state/readings.md` unchanged before
the launch.

- **Reading 1:** if arm P's region count is flat in $x$, like arm E's, the growth was the
  fixture's $G$, and `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` closes.
- **Reading 2:** if arm P's still grows, the $G$ construction is not the cause. The issue is
  narrowed, and the cause is looked for elsewhere.
- **The attribution:** S against P₀ measures what the `phase_spline` floor adds; P₀ against P
  measures what the anchoring floor adds. Each is stated per shape, as a ratio of growths, with the
  group that carries the deepest regions.
- **If the numbers support neither reading cleanly,** say so. An example would be P growing by a
  factor between E's and S's, or the shapes disagreeing. Give the numbers, and leave the reading to
  the user. Do not choose a reading after seeing the numbers.

---

## 5. What to write down

`var/handover-04/GK-PHASE-COST-GROWTH.md`, for the laptop to commit as
`docs/handover/GK-PHASE-COST-GROWTH.md`. It is short: a measurement memo, not a survey.
- **§0 — the objects,** in `KOHRI-TERADA-ORACLE.md` §0's shape: the four arms, whose each
  construction is, the tree (the dispatch SHA), and why A2's recorded cells are context and not a
  column.
- **§1 — the readings,** quoted from §4, with `state/readings.md`'s time beside the launch's.
- **§2 — the tables:**
  - regions and maximum depth per cell and arm, and their growth per shape;
  - the level at 980;
  - `integral_time` with the load;
  - the cells `not reached`;
  - A2's recorded curves beside them, labelled pre-B1.
- **§3 — what the numbers support:** reading 1, reading 2, or neither (§4), with the attribution.
- **§4 — the depth record:** per arm and shape, the group that carries the deepest regions and
  where in $z$ they are, and the phase-limited counts.
- **§5 — the representation term, recorded, no verdict:** $N_{\rm arm} - N_{\rm E}$ for S, P₀ and
  P per cell, each with both cells' declared errors. B2 re-takes it properly.
- **§6 — what this does not cover.** At minimum:
  - $b = 0$ only;
  - one seam;
  - the fixture's $T$ made production's in every realistic arm, so $T$'s own floor is not
    measured here (05a did that on A3 v2);
  - $x_{\rm resp} \le 10^8$, not production's $4\times10^{12}$;
  - the producers' stored phase samples (`levin-phase-floor` 06+), which the fixture does not
    model.

Beside it: `log.md` (campaign README §5.1), `issues.md`, and `out/`, `scripts/` and `state/`. The
memo is the verification record; there is no separate `verification.md`.

---

## 6. What this prompt does not do

- It does not edit `realistic_large_x.py`, `REALISTIC-LARGE-X.md` or any test module.
  - The false claim at `test_phase_groups.py:301` and `Case`'s superseded $G$ stay as they are:
    fixing the fixture is B2's, by `[02-realistic-fixture-…]`'s own next step.
  - The sibling script does carry `[12-realistic-large-x-…]`'s fix, in its own copy.
- It touches no production code, `AdaptiveLevin/`, `main.py`, `config/` or
  `docs/radiation-oracle/`.
- It uses no Ray, no datastore and no lock, and it runs no git command that writes.
- It does not close any issue. Its result is the input to the user's decision in the records
  commit.

---

## 7. Acceptance

1. One command launches the run from the main checkout's root. The run is registered and ended
   `done`, with every cell recorded, `ok` or `not reached`.
2. The construction check (§2.1), determinism and transparency (§3) were saved before the launch,
   and `state/readings.md` is older than the run's manifest.
3. Every arm's phase objects are what §2.1 says, by class and `exact_remainder`, in every cell's
   record.
4. §2.2's fields for every cell, the growth per shape and arm, and the level at 980.
5. §4's readings answered against the numbers: reading 1, reading 2 or neither, with the
   attribution.
6. The representation term recorded for S, P₀ and P, with no verdict.
7. No file outside `var/handover-04/`, the run directory and its launch log was written;
   `realistic_large_x.py` still hashes `c1cd3598…`; the suite was not touched.

---

## 8. Stop conditions — stop and ask the user

- Any Gate item is not as stated.
- **An arm cannot be built as §2.1 says** offline. Examples: P's or P₀'s $G$ needs a production
  change, or a realistic $T$ will not build on a table `cs_tau`. Do not substitute an
  approximation and give it the arm's name.
- The wrapper is not transparent, or a determinism check fails.
- A cell raises for a reason other than its budget. Record it, and stop before the next rung of
  that shape and arm.
- Anything would need a change to production code, a test file, `AdaptiveLevin/` or a datastore.

---

## 9. The log and the board

`log.md`, template as campaign README §5.1, classifying every deviation. Beyond it:
- the construction check, determinism and transparency outputs;
- when `state/readings.md` was written, relative to the launch;
- the cells `not reached` and their elapsed times;
- the run directory.

**It commits nothing.** The laptop copies `var/handover-04/` back, without checkpoints, and makes
one records commit:
- the scripts and the memo;
- this campaign's log, `logs/04-gk-phase-decision-test.md`;
- the board's A4 row and item;
- the narrowings of the two `02-` issues and of `[12-realistic-large-x-…]`;
- `docs/OPEN_ISSUES.md`.

Whether `[02-levin-cost-growth-…]` closes is the user's decision, on §4's reading.
