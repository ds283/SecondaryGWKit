## Your brief: the census agent — prompt 07 §6 (Gate V0, the census $g$, V1–V5, V4's references)

Read `/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/review/dispatch_common.md`
first, in full; it binds you, **except where this brief overrides it** (the next section). Then read
`/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/review/preflight/B6_B7_verbatim.md`:
the orchestrator note's §B.6 and §B.7, verbatim, written after the common brief. Those two files and
this brief are the only things you read under `var/levin-phase-floor-07/review/`.

**Your scope is prompt 07 §6, as amended by prompt §10** (its last bullet: V3 sets aside a phase Rule
A skipped), and only it. §5, the rehearsals, belongs to the rehearsal agent, and it has finished.
You need no lock, and you start no Ray session.

### What overrides the common brief

The common brief was written for Part B's first dispatch. Part B was then held, prompt 07 was
amended (§10, Rule A), and Part B resumed at a new dispatch SHA. So:

- **Part B's dispatch SHA is now `e182bc7a827142cd4f42594f1a9d8cf85ccc3a5f`** (`e182bc7a`). Its code is
  `adffa21b`'s, "Skip locating a phase whose variation is below the C2 gate", on 07's `ad732887`.
  Between `74600350` and `e182bc7a` the commits are `2ddd3306` (prompt §10), `adffa21b` (the code)
  and `e182bc7a` (the note's §B.7). `git diff --stat 74600350 e182bc7a -- . ':!docs' ':!prompts'`
  touches only `AdaptiveLevin/levin_quadrature.py`, `AdaptiveLevin/tests/test_levin_stationary.py`,
  `AdaptiveLevin/tests/test_levin_trust_yardstick.py` and `CLAUDE.md`.
- **The census tree $g$ is the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-e182bc7`,**
  detached at `e182bc7a`, clean, no venv. Your census, its determinism check (V0's third clause), the
  captures for V4 and V4's references run here. **Read the prompt, the code log and `CLAUDE.md` here.**
  - Module hashes at `e182bc7a`, as measured: `AdaptiveLevin/levin_quadrature.py` `4e5da371…`,
    `ComputeTargets/QuadSourceIntegral.py` `33cb161e…` (unchanged from `74600350`).
- **The parent worktree** `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-91472db` (`91472dbe`,
  `levin_quadrature.py` `beaf0e23…`, 06m's code) is unchanged. **V0's K0 run runs here.**
- **The old code worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-7460035`** and the stores
  R27, R28 and R29 are records. **You run nothing in `-7460035`, and you do not open those stores**
  (you have no need to).
- **Where you write:** nothing outside these:
  - `var/levin-phase-floor-07/census/`, with `out/`, `scripts/` and `state/`, and your `log.md`,
    `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt`;
  - your registered runs' directories in `var/runs/`: slugs `k0`, `census` and `references`, each
    `--campaign levin-phase-floor --prompt 07` (so `levin-phase-floor-07-k0-…`, `…-census-…`,
    `…-references-…`), and nothing else unless you report it as a deviation.

  Everything else the common brief forbids stays forbidden. In particular you never write to
  `var/levin-phase-floor-07/rehearsal/` or `review/`, A3 v2, its scratch copy, 06m's census, any
  store, or `var/levin-phase-floor-07-leaves/`.
- **Six cores:** the rehearsal agent's work is done. You are the only computation on this host.
- **Read the code log in the new worktree in full:** `prompts/levin-phase-floor/logs/07-stationary-point-guard.md`,
  including its sections "Amendment §10 (Rule A)" to "Amendment §10: state handed on". Rule A's result
  key is `location_skipped_phases` (the log's deviation 39); `QuadSourceIntegral.py` does not store it.

### What Part B has found so far (facts for your context, not instructions)

The rehearsal agent's records are in `var/levin-phase-floor-07/rehearsal/` (stop 1) and
`rehearsal/r29/` (R29, at `e182bc7a`). You may read them; you do not write there. In brief:
- **At `74600350` (R28),** location treated an identically zero rate (`q-r` group calls with
  $q = r$) as 4,097 roots, one per scan sample. The user ruled that Rule A fixes it (prompt §10).
- **At `e182bc7a` (R29),** against R28: 90 Levin records changed, all calls R28 located and R29 does
  not: the 26 `q-r` calls ($V = 0$), 60 analytic `2pt5` J4/Y4 blocks ($V$ 0.49–15.9), each equal to
  R27's; and **4 analytic blocks where Rule A skipped a phase with a root ($V$ 15.11 and 15.90, below
  `SIX_PI`) that the C2 gate's node estimate (19.16 and 20.20, above `SIX_PI`) still sent to the Levin
  solve, and the nodal guard guarded one leaf**. All four are honest against references in all three
  stores. **The user's rulings:** these are 07's intended behaviour (§10 assigns such a region to the
  guard), findings, not a stop; this case (O9, `[07-c2-gate-threshold-was-never-measured]`) is dealt
  with after the census. So `num_guarded_regions` may be nonzero in your census: prompt §6 lists it as
  descriptive ("expected 0 since `theta_deriv` is supplied"), and you report what you find there.
- **Rule A skips widely:** in R29's replay, 423 of 470 replayed calls report a skipped phase, with $V$
  from 0 to 18.70.
- **S1 and S2 hold** on R28 and R29; R29's 44 `G+q+r` calls locate a root at $z \approx 10211.318$
  (42) or $275.009$ (2).

### The user's words so far in Part B, verbatim

- At stop 1: *"I approve the replay. Also, add a reference to check 2 or 3 of the `q-r` calls, to
  settle whether R28 is honest. Hold Part B before the census."*
- At the hold: *"Yes, rule A alone, please. Open a separate issue for the `6pi` gate. We'll deal with
  this afterwards, as you say. Record the resonance issues as belonging to the one-loop build and
  point at the planning document. Please make the fix as an amendment of 07."*
- On landing the amendment: *"Yes, please push both, then write and push §B.7, and send to the Mac
  Studio to be executed."*
- On R29's four guarded blocks: *"Yes, please proceed as you recommend."*
- At stop 1′, dispatching you: *"Yes, please go ahead. There seems no reason not to. We may have to
  come back to the O9 issue but that is best done after the census anyway."*

### Read also

- 06m's census records, `docs/levin-phase-floor/06m/census/` in the new worktree: `log.md` (all of
  it, its deviations above all), `verification.md`, `predictions.md`, `issues.md`,
  `KEPT_ON_MAC_STUDIO.txt`, `out/`, `state/`, and its harness `scripts/` (`run_census.py`,
  `run_k0.py`, `compare_k0.py`, `compare_06m.py`, `worker.py`, `harness.py`, `dispatch.py`,
  `rerun.py`, `prov.py`, `refs_common.py`, the scan, `census_projection_06m.py`, and whatever these
  import). 06m's census is `levin-phase-floor-06m-census-20261106T102153` (18,560 units, wall
  18,465 s, checkpoint `ea5bc3ab…`); it is your "before".
- 06g's references for V4: `docs/levin-phase-floor/06g/references/` (`log.md`, `verification.md`,
  `out/scores_stage3/scores.json` and its `report.md`, `v1/`), `tools/filon_reference.py` and its
  tests. z00, z12 and Z3's seven analytic calls are scored against 06g's references (`rows`) as §6
  says.
- The rehearsal agent's capture harness, `var/levin-phase-floor-07/rehearsal/scripts/capture/` and
  `r29/scripts/capture/`, for how a call was captured and checked bit for bit at this code, if useful
  for V4's captures. Copy what you use into your `scripts/`, recording its hash and diff.

### The harness

- **Copy** `docs/levin-phase-floor/06m/census/scripts/` (from the new worktree) into
  `var/levin-phase-floor-07/census/scripts/` (prompt §6), and save `diff -r` of the copy against
  06m's in `out/` before and after your changes. **Change only what prompt §6 lists** (the tree and
  its paths; §1.5's diagnostics per call and per returned leaf, and Rule A's `location_skipped_phases`
  per call, which you need for V3; the scan on every group call at both rtols and every analytic call
  at `1e-7`, against the returned leaves; the comparison with 06m's census, setting the new keys
  aside; V4's references). Mark each change in the code ("prompt 07"), and record each as a deviation.
- **No variant is installed.** Each worker imports the tree's own `AdaptiveLevin/levin_quadrature.py`;
  the harness records the module's path and SHA-256 (`4e5da371…` at $g$, `beaf0e23…` at the parent),
  and you check it in the records.
- **Every checkpoint record** carries its script's SHA-256, its tree's `HEAD` and
  `git status --short`, and each unit's evidence of where its modules were loaded from.
- **The store** is the scratch copy `var/levin-phase-floor-06/wider/state/a3v2_copy/`, re-checked
  against its certification (`var/levin-phase-floor-06/wider/state/copy_files_certified.txt`, and
  06e's `docs/levin-phase-floor/06e/review/checks/a3v2_copy_after.sha256`), opened read-only, hashed
  before and after. 06m's census run directory is read only, each file checked against
  `docs/levin-phase-floor/06m/census/KEPT_ON_MAC_STUDIO.txt` before use, hashed before and after.
- **The budget:** 300 s per unit, with 06a's over-budget pass (a re-run once at 3,600 s, after both
  sets), as 06m's.

### The comparison with 06m's census (V0's second clause and V1)

06m's comparison (`compare_k0.py` / `compare_06m.py`), with only the new keys set aside: §1.5's six
result keys, the four counts, `location_skipped_phases`, and anything else 07 adds. **Say exactly
which keys and fields you exclude, and why each is new or timing.** Floats by bit pattern, arrays by
dtype, shape and bytes.

### The order

1. **Before any number:** the checks you rely on (the three trees' `HEAD`s and `status`, module
   hashes, the scratch copy against its certification, 06m's census against its kept list, the
   registry), and the scratch copy and 06m's census hashed (`out/`).
2. **Gate V0, all three clauses** (prompt §6):
   - **(a)** `git diff --stat 48cf87c0 91472dbe -- . ':!docs' ':!prompts'` shows nothing the harness
     imports;
   - **(b)** one registered run, slug `k0`, **in the parent worktree**, on the 100 units of
     `docs/levin-phase-floor/06l/out/k0/units.json`, at most six processes, then all 100 compared with
     06m's census by the comparison above. **Holds** if all 100 are equal;
   - **(c)** two runs of one $g$ unit on a stationary-point row (one whose calls locate a point at
     $g$), bit-identical, in the new worktree. Say how you chose the row.

   If a run may exceed about ten minutes, launch it detached, make one check, and end your turn with
   the launch report; the orchestrator resumes you when its pid exits. **A V0 failure is a stop.**
3. **The predictions,** after V0 and before the census's launch: copy into `state/predictions.md`
   prompt §6's "Predictions, fixed before the run" (V1–V5) **unchanged**, word for word, then prompt
   §10's last bullet on V3, word for word, then **V4's selection rule and its 24 calls**, fixed (each
   call by unit, rtol, kind, index and label, with how it meets the strata: band, $k$, rtol, 06m's
   depth, the deepest four). The selection must be decidable from 06m's census and $g$'s rows
   without $g$'s results; say how. Do not rewrite the file afterwards (a correction is a new file and
   a deviation). Its mtime is the review's evidence.
4. **Planned stop 2,** after V0 and the predictions, before the census. End your turn and report:
   - V0's three clauses, with numbers (units, calls and region records compared and differing, and
     every differing path if any) and where the outputs are;
   - the `k0` run id, where each unit's emitted record and region pickle are, and **the exact
     commands that run a chosen list of units into a directory of the caller's choosing, at the
     parent and at $g$**, with the form of the units file and how to compare their output with 06m's
     census (the orchestrator re-runs two K0 units at the parent itself and compares them with 06m's
     census, and runs one stationary-point unit at $g$ twice);
   - the predictions file's path and SHA-256, and V4's 24 calls;
   - the harness diff against 06m's, every change classified;
   - **the census plan:** the launch command (registered, slug `census`, detached, six workers, 300 s
     per unit with 06a's over-budget pass); the cost projection from 06m's wall (18,465 s) and from
     your own unit timings at $g$ (location costs time: B.5 says the ComputeTargets root took 442 s
     against 291 s), with the load as you find it; and the backstop: a run past its projection is
     reported with its progress and projected finish, and **stopped only at a projected day or more,
     counting the units in flight, not only those finished**;
   - your deviations so far.

   Launch no census before the orchestrator resumes you.
5. **The census.** Launch it as planned, make one check, and end your turn with the launch report.
   Do not poll it. Nothing else computes while it runs.
6. **Planned stop 3.** You are resumed after the census's process has exited and the orchestrator has
   checked that it is `done`, 18,560 of 18,560. Then, **first and alone: every Levin call that locates
   nothing and guards nothing, compared with 06m's** by the comparison above, at both rtols. **A
   single such call that differs from 06m's is a stop, before anything else is scored:** end your
   turn with the first differing units and fields, and the count. A unit that ended in an error (not
   over budget) is a stop too. If none, end your turn and report planned stop 3: units done, errors,
   over-budget units (with wall against 06m's), and the comparison's counts.
7. **When resumed after stop 3:** score V1, V2, V3 (with §10's set-aside: every phase Rule A skipped
   set aside, its calls counted and listed, with any sign change the scan finds in them), V5, and the
   descriptive items (prompt §6, and B.5's: for each call that locates a point, every window
   `stationary_windows` records as stopped, with `leaves_falling_at_stop`, by rtol, and whether the
   call converged; over-budget units with their wall time against 06m's). If V2 finds a returned leaf
   holding a sign change, check whether it is guarded or a window leaf before scoring (B.5). Then
   **capture V4's 24 calls at $g$**, each checked bit for bit against the census record, and
   **project the references' wall time**.
8. **Planned stop 4,** before the references run. End your turn and report: V1, V2, V3, V5 as
   scored, with numbers; the 24 calls, their captures and their checks; the projection (beyond a day
   is a stop); the launch command for slug `references` (from $g$, at most six workers, four readings:
   QAWO and FCC at two resolutions, FCC at resolution 2 the reference; the 30-minute reading limit a
   check, not a cutoff).
9. **When resumed:** launch the references (detached if over about ten minutes; one check; end your
   turn). When resumed after them, score V4 (z00, z12, Z3's seven against 06g's; the 24 against their
   own references, with the tolerance $\min({\rm abserr}_{06m}, {\rm abserr}_g)/100$; a call whose
   readings do not agree is reported, not scored), and "the stationary-point calls before".
10. **After:** hash the scratch copy and 06m's census again; each must equal its before. Then write
    `log.md`, `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt` (the census run directory
    in full, every pickle, every file over 2 MB). **The census's checkpoint and region records stay on
    the Mac Studio in full; 08 compares against them.**

**Your final report:** V0–V5 each holds, fails or undecided with its numbers; the descriptive items
(depth at both rtols against 06m's, windows, totals in the 720, locations, `num_guarded_regions`,
convergence with 06m's flag-cost table beside it, §1.3's firings, cost and over-budget units against
06m's); the before/after hashes; every run id and outcome; every deviation classified; anything in
`issues.md`; and anything left running (there should be nothing).
