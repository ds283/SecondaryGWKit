# Orchestrator — prompt 02, phase-noise injection

Read [`README.md`](README.md) in this directory first: it holds the rules that bind you.
**You do not write code.**

**The prompt:** [`prompts/levin-phase-floor/02-phase-noise-injection.md`](../02-phase-noise-injection.md)
**Board item:** 02 · **Closes:** nothing · **Narrows:**
`[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` and
`[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`, both on the `handover`
board · **Model:** Opus, as the prompt recommends.

**Gate**, checked by the orchestrator on 2026-10-25 at `6af798f`, and again at `757ef01`
(01's and `handover` 11's dispatch notes, from another session):
- `handover-remedial` clean; `git worktree list` shows the main checkout only.
- Nothing `running` or stale in the registry (59 runs: 57 finished, 2 `unknown` pre-registry).
- `docs/levin-phase-floor/` does not exist: 01 has not landed. It is running on the Mac Studio,
  orchestrated from another laptop session, which will commit its records to this branch.
- The lock is free. It is not needed: 02 starts no Ray process.

## 0. What makes this prompt unusual

It changes no code and runs no test root, so there are no suite baselines. What it produces is a
**measurement whose verdicts were fixed before it ran** (campaign README §5 rule 3). A tidy report
is not enough for the review. It turns on three things the report cannot show by itself:
- the wrapper is **bit-identical at δ = 0**, so the baseline is production's integrand and the
  injection is the only change;
- every checkpoint record carries the **committed** script's SHA-256, so the report is of the code
  that landed;
- N1–N4 are scored against §4 **as written**, with the numbers, and nothing is re-posed once the
  numbers are in.

**Facts checked by the orchestrator at `6af798f`; none of them is changed by `757ef01`.** Pass them on.
- **The injection point.** `ComputeTargets/QuadSourceIntegral.py:15` imports `build_phase_groups`
  into the module, and `phase_group_Levin_integral` calls it through that name at `:1045`. Wrapping
  `ComputeTargets.QuadSourceIntegral.build_phase_groups` therefore reaches every Levin sub-interval.
  The wrapper must be restored after each cell (`try`/`finally`), so a δ = 0 cell never runs
  through a noised wrapper left behind.
- **`PhaseGroup`** is `@dataclass(frozen=True)` at `ComputeTargets/phase_groups.py:176`, with
  `label`, `theta`, `theta_mod_2pi` and `theta_deriv` among its fields (`:186-191`). The composed
  remainder is `_composed_theta_mod_2pi` (`:542`). All three phase callables take `log(1+z)`.
- **The depth cap.** `DEFAULT_LEVIN_MAX_DEPTH = 20` is at `AdaptiveLevin/levin_quadrature.py:154`
  and is `adaptive_levin_sincos`'s default `depth_max` (`:1701`). `QuadSourceIntegral.py` passes no
  `depth_max`, so the fixtures run under production's cap. `DEFAULT_QUADRATURE_RTOL = 1e-7` is at
  `config/defaults.py:173`; import it.
- **`Case.run`** (`ComputeTargets/tests/test_quadsource_integral.py:511`) defaults to
  `atol=1e-25, rtol=1e-8`. Pass both explicitly. Its result carries `WKB_Levin_data`, a `LevinData`
  aggregated over sub-intervals (`QuadSourceIntegral.py:931`), whose `max_depth` is the maximum over
  them (`:676`); it can be `None` if a case has no WKB sub-interval.
- **`realistic_large_x.run_cell`** (`docs/handover/realistic_large_x.py:303`) calls
  `case.run(atol=REF_ATOL, rtol=REF_RTOL)` with `REF_RTOL = 1e-12`, so it cannot be called as it
  stands. The build is its lines before that call: `w_of_b(0.0)`, `scaled_shape`, `seed_fixtures`
  (which writes the class-level cache `Case._fixtures`), and `Case(b=0.0, shape=…, x_resp=…,
  exact=…)`, with the seam closed by not calling `open_the_seam`. Copying them is an
  `IMPLEMENTATION CHOICE` to record.
- **Importing `realistic_large_x`.** `docs/handover/` is not a package. On import the module puts
  `docs/radiation-oracle/` on `sys.path`, hashes itself and names its checkpoint
  `var/runs/realistic_large_x_cells.jsonl`. It writes nothing unless run as `__main__`. That
  checkpoint must not change.
- **A2's cost, for scale** (`REALISTIC-LARGE-X.md` Tables 7–8, at `rtol = 1e-12`): exact cells
  under 1 s; realistic `together` 18.5 s at 980 and 173 s at 1e4; realistic `q-smooth` 6.8 s and
  62 s. At `rtol = 1e-7` the δ = 0 cells should be cheaper. The noised cells at the cap are the
  unknown the probe exists to measure.
- **N1's clause is asymmetric.** It **holds** if all three conditions hold in all eight cases, and
  **fails** if any is false in a `realistic` case. A condition false only in an `exact` case is
  neither, so the verdict is **undecided**, and the log says which case and which condition.
- **The manifest will say `git_dirty: true`**: the sweep runs on the uncommitted script, before the
  prompt's one commit. That is `STRUCTURALLY REQUIRED`, as in `handover` 09's rehearsal.
- **Load.** At 18:12 the laptop's load average was ~130 on 10 cores, from a ChronoSync backup.
  Record `uptime` beside the probe's timings. A projection taken under such load says so.
- **Disk.** 56 GB free.
- **Other sessions write to this checkout.** A laptop session is orchestrating 01 and `handover`
  11, both running on the Mac Studio, and will commit their records here: 01's touches this
  campaign's board, `docs/OPEN_ISSUES.md` and `docs/levin-phase-floor/`; 11's touches the `handover`
  board and `docs/OPEN_ISSUES.md`. 02 narrows two issues on the `handover` board, so it can meet
  either.

## 1. Before you dispatch

1. The Gate, as above.
2. **The `var/` marker:** `touch` a file in the session scratchpad immediately before dispatch, for
   check 9.
3. **Commit this note and `README.md`**; that commit is the dispatch SHA.

## 2. Dispatch

One fresh-context subagent, on **Opus**, in the main checkout. Give it the prompt, the dispatch SHA
and §0 of this note. It does not read `orchestrator/`. Tell it plainly:
- **`HEAD` is not its own.** Other sessions commit to this branch while it works.
- **One commit, at the end**, staged by explicit path. It covers `docs/levin-phase-floor/`
  (the script, `02/`, `VERIFICATION.md`), its log `prompts/levin-phase-floor/logs/02-*.md`, this
  campaign's board, the `handover` board and `docs/OPEN_ISSUES.md`. A path outside that list is a
  stop.
- **The shared files last, in one burst.** The board, the `handover` board, `docs/OPEN_ISSUES.md`
  and `VERIFICATION.md` are edited only at step 6–7, immediately before the commit. First
  `git status --short` and `git log --oneline <dispatch SHA>..HEAD`: if 01 has landed, read its log
  and merge with its rows; if any of those files has uncommitted changes that are not the agent's,
  **stop and ask**.
- **Launch, one check, end the turn.** Report the run id, the run directory, the detached Python
  pid, the `sh` pid if `nohup sh -c` is used, and the first checkpoint line. The orchestrator waits
  on the pid, once, in the background, and resumes the agent with the run's final `status.json`.
  The agent does not wait or poll.
- **Stop and ask** at any of prompt §5's conditions, and if the probe projects more than 24 hours,
  with the proposal the prompt asks for.

## 3. The review — nine checks

1. **Scope.** `git show --stat <02>`: only §2's paths. Nothing under `AdaptiveLevin/`,
   `ComputeTargets/`, `LiouvilleGreen/`, `CosmologyModels/`, `ObjectFactories/`, `config/`,
   `RunDrivers/` or `main.py`; `docs/handover/realistic_large_x.py` byte-identical;
   `var/runs/realistic_large_x_cells.jsonl` unchanged.
2. **The construction, read.** In the script: noise on `theta_mod_2pi` only, through
   `dataclasses.replace`, `theta` and `theta_deriv` untouched; the white $\xi$ exactly as §1 (blake2b
   of `label.encode() + struct.pack("<d", x)`, first eight bytes over $2^{64}$, $2u-1$); the
   staircase with $\Delta = 2\delta$ on the group's own `theta`; `rtol` imported, `atol = 1e-45`,
   $b = 0$, seam closed; the wrapper restored in `finally`.
3. **Step 2, re-run here.** The bit-identity check at δ = 0 on the cheapest case, by the script's
   own entry point or by importing it from a scratch file: `==` on `total`, `total_abserr` and the
   group records, not `assertAlmostEqual`. The δ = 1e-4 cell differs.
4. **Provenance.** Recompute the committed script's SHA-256; every checkpoint record in
   `02/state/` carries it, and `noise_02.txt`'s header does too, with the dispatch-era `git HEAD`.
   Re-run `--report` from the copied checkpoint into the scratchpad: it reproduces `noise_02.txt`
   apart from any timestamp.
5. **The run.** `runregistrykit list` shows it `done`; its manifest names `levin-phase-floor`, `02`,
   `noise` and the launch pid; stdout and stderr are in the run directory and copied to
   `02/state/`.
6. **Coverage.** Count the checkpoint: arm A 80 cells, arm C 24, arm B (labels × 3 δ × 2 models)
   as the log states the label count. Every missing cell is named in the log with a reason. No
   δ = 0 cell at the cap; if one is, the agent stopped (§5).
7. **The verdicts, re-scored here.** From the checkpoint, by a scratch reader: $\delta^*$ per case
   and N1's three conditions; N2's two ratios per shape; N3 per label at white $10^{-5}$, with
   slow/fast from `theta_deriv`; N4's inequality per cell. Each verdict in the log agrees, its rule
   is §4's text unchanged, and N1's asymmetry is respected.
8. **The records.** The log has §5.1's headings and classifies every deviation (§0's `git_dirty`
   and the copied build lines among them). `VERIFICATION.md` §2 written, and nothing else in it
   but §1's placeholder if 01 had not landed. This board's 02 row; the two `handover` issues
   narrowed, not closed, with N2's and N1's outcomes; `docs/OPEN_ISSUES.md` hooks narrowed to
   match, count unchanged, date correct. `black --check` clean on the script. The commit message
   follows `CLAUDE.md`'s form.
9. **Nothing else moved.** `git status --short` is clean. `find var -newer <marker>` lists the run
   directory and nothing else of the agent's; anything else is attributed to another session or
   reported.

## 4. After the review

If it passes: residue first, in a follow-up commit of the orchestrator's own (the commit hash on
the board and in the log; this directory's README row). Then report to the user: the four
verdicts with their numbers, what N1–N3 say about H-cancel and H-anchor, and anything the log
leaves for 03.

## 5. Stop and ask the user

Relay verbatim; do not adjudicate. The prompt's §5 conditions, or any failed check in §3: report
it, do not repair it.

## 6. What happened (2026-10-25)

**The user's decisions, in order:**
1. **A projection anywhere near 24 hours runs on the Mac Studio**, not the laptop: it is faster,
   has more memory, and does not sleep or move. It became moot.
2. **Stop the probe.** The agent made the probe a registered run of its own
   (`levin-phase-floor-02-noise-probe-20261025T183809`), because one noised cell had already run
   more than nine minutes in a scratch test. Its first noised cell, `exact`/`together`/980 at white
   δ = 1e-5, ran ~69 minutes without finishing. Noise that does not shrink with region width
   defeats the driver's per-region relative test at every depth once it exceeds `rtol`, so each
   Levin call bisects the full depth-20 tree, and the sweep would have taken weeks on either
   machine. SIGTERM at 19:47:59Z; the registry records it `killed`, exit 143, 16/19 units.
3. **02 is superseded by 03** (`d5f5a2b`), which measured the mechanism on A3 v2's rows directly.
   N1 and N3 are answered there; N4 moves to 04; N2's original form is not tested, which the user
   judged does not matter.

**The review of the records commit `87813ed`.** The §3 checks were written for a completed sweep;
those that apply to a stopped one were run, and pass:
- **Scope:** ten paths, all under `docs/levin-phase-floor/` and this campaign's log and board.
  `docs/OPEN_ISSUES.md`, the `handover` board and `realistic_large_x.py` unchanged. The one edit to
  `VERIFICATION.md` outside §2 replaces 01's "§2 is reserved" placeholder (the log's deviation 8).
- **The construction, read:** noise on `theta_mod_2pi` only, by `dataclasses.replace`; white $\xi$
  from blake2b of `label + pack('<d', x)`; `rtol` imported, `atol = 1e-45`; wrappers restored in
  `finally`.
- **Step 2, re-checked here** from the run's own ledger before the commit: on all eight cases the
  wrapped and unwrapped δ = 0 cells agree in every scalar field and every group record, `elapsed`
  excepted. No δ = 0 cell came near the cap (depth 0–2 exact, 6–8 realistic). The δ = 1e-4 check
  never ran.
- **Provenance:** the committed script hashes to `6ddaaeca…`, the hash all 16 ledger records carry;
  `black --check` clean. The five files in `02/state/` are byte-identical to the run directory's.
- **The records:** the log has §5.1's headings, prompt §4 verbatim (checked by substring), N1–N4
  each "not scored" with where it now stands, and every deviation classified. The board's 02 row
  and status line are updated. The commit message follows `CLAUDE.md`'s form.
- **Not applicable:** coverage and the re-scoring of N1–N4 (no sweep), and `--report`
  reproduction (the probe report was written from the copied ledger).

**Residue, in the orchestrator's follow-up commit:** `87813ed` into the board and the log; the
board's legend gains ⛔, and H-anchor's "Scored by" drops 02 N3; README §2 marks 02 superseded and
gives 04 N4; README §6.3 gains **Q5**, whether `rtol = 1e-7` is right for accuracy, at the user's
request.
