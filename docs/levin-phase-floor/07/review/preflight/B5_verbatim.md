### B.5 Addendum: Part B's dispatch

*Written 2026-11-07, at Part A's landing.*

**Part A** is the code commit **`ad732887`**, whose parent is **`91472dbe`**. It was reviewed in
§A.3's eleven checks, all passing, and pushed on the user's word: *"Yes, please land it. Push. Then,
write, commit, and push the dispatch addendum for Part B."* Two orchestrator commits came before it,
both pushed:
- `945c8668` opened `[07-cc-interior-nodes-shift-with-a-rounded-midpoint]`;
- `91472dbe` amended the prompt at §2's stop (its §9).

**The user's rulings at §2's stop** are prompt §9 points 1–4, with the user's words, and the board's
Decisions row **07 §2 stop**:
- §2.2 ran at `rtol` $10^{-9}$, and its rules gave **$\Phi_w = 2\pi$, $\Lambda_c = 4$**;
- §1.3 is QUADPACK's `iroff1` counted per window: limit 6, value tolerance $10^{-5}$, error ratio 0.99.
  A stopped window's leaves are returned floor-limited;
- a key-set assertion admits §1.5's six new result keys;
- §2 was re-run once, on the code with §1.3 built.

**The trees, by SHA:**
- **R27 and V0's K0 run** at **`91472dbe`**, in the worktree `SecondaryGWKit-lpf07-91472db`. Its
  production code is the dispatch SHA `bd21bf23`'s, which is 06m's: `git diff --stat bd21bf23
  91472dbe -- . ':!docs' ':!prompts'` is empty. Against R26's tree `38f6a828` it differs only in
  `CLAUDE.md`. So S0 (R27 against R26) is a reproducibility check, as written.
- **R28, the census $g$, V4's references, the comparisons and the readers** at **this addendum's
  commit**, in the worktree `SecondaryGWKit-lpf07-<this commit's short SHA>`. Its code is
  `ad732887`'s. `git diff --stat 91472dbe ad732887 -- . ':!docs' ':!prompts'` touches only:
  - `AdaptiveLevin/levin_quadrature.py`;
  - `AdaptiveLevin/tests/test_levin_stationary.py` (new) and `test_levin_trust_yardstick.py`;
  - `ComputeTargets/QuadSourceIntegral.py`, the four counts;
  - `ComputeTargets/tests/test_quadsource_levin_guard_counts.py` (new);
  - `CLAUDE.md`.

**What Part A's review found that Part B must carry:**
- **Every number in the review was re-run on the laptop:**
  - the seven roots, at 103, 781, 39, 156, 280, 127 and 64, all OK;
  - §2.2's 124 units, bit-identical on the committed module;
  - z12 and w03 replayed bit for bit. z12 is honest at 0.718 of its error, against 1,681× at the
    parent.
- **Windows stop while some of their leaves are still falling** (the agent's O8). This happens on 8
  of §2.3's 14 calls at $2\pi$, and all 8 are honest and converged. **The census reports it, as
  descriptive:** for each call that locates a point, every window `stationary_windows` records as
  stopped, with `leaves_falling_at_stop`, by rtol, and whether the call converged.
- **Location costs time** (the agent's O5). A rate that does not vectorise is evaluated 4,097 times
  per call. The ComputeTargets root took 442 s here against 291 s at dispatch. The census's 300 s
  per-unit budget may be reached more often than in 06m's, which had none over. Over-budget units
  are counted under V5 as the prompt says, and **reported with their wall time against 06m's**. That
  is a finding, not a stop.
- **The bar on window leaves protects no measured value.** Under breakage (iii) test 5's value stays
  honest (0.0057 of its error). Window leaves carry the Clenshaw–Curtis floor, which has no
  $1/\theta'$ in it. Nothing for Part B to do. If V2 finds a returned leaf holding a sign change,
  check whether it is guarded or a window leaf before scoring.
- **The value tables differ from R27 only inside `QuadSourceIntegral`,** and there only in rows
  whose group or analytic calls located a point or guarded a leaf, plus the four new counts. §0.1
  already excludes those counts from "value-identical".
- **Not measured here:** `[07-cc-interior-nodes-shift-with-a-rounded-midpoint]`. Production spans
  are non-dyadic, so the parent's leaves carry that floor too. The census does not score it.

**§B.0 and §B.1, re-checked read-only on the Mac Studio, 22:47–22:50 UTC, 2026-11-07:**
- **The main checkout** is `handover-remedial` at `28ebaf07`, with `git status --short` empty.
  `git worktree list` shows the main checkout alone; 06g's and 06m's worktrees were removed
  (`a8c7854c`).
- **The registry** holds 62 runs, all finished, nothing `running` or stale.
- **The lock and Ray:**
  - `rehearsal_lock status`: free (`~/.raypoolkit/rehearsal-lock` does not exist);
  - `ray-processes --ray-temp-dir /tmp/ray-sgk`: none. `/tmp/ray-sgk` holds only session
    directories from 2026-10-25.
- **Load and disk:** the load is 2.75 / 3.04 / 3.17, and 136 GiB is free (85 % used).
- **No 07 paths yet:** there is no `var/levin-phase-floor-07/` and no
  `var/datastores/levin-phase-floor-07-*`. **But `var/levin-phase-floor-07-leaves/` exists** (28
  files, 8 MB, last written 2026-10-29). It is the working directory of 06e's leaf study of 07's
  fourth stop, whose records are committed under `docs/levin-phase-floor/06e/investigations/07-leaves/`.
  It is not Part B's, and Part B's paths (`var/levin-phase-floor-07/…`) do not collide with it.
  **No agent writes or reads it.** Whether it is kept is asked at 07's records commit.
- **The inputs, by SHA-256 against their recorded lists,** 160 files, all OK:
  - the scratch copy's five files, against `06e/review/checks/a3v2_copy_after.sha256`, with no
    `-wal` or `-shm`;
  - 06m's census run directory `levin-phase-floor-06m-census-20261106T102153`, all 150 files
    against `06m/census/KEPT_ON_MAC_STUDIO.txt`, checkpoint `ea5bc3ab…` among them;
  - R26's five store files against `06m/rehearsal/KEPT_ON_MAC_STUDIO.txt`, primary `e90e26c2…`.

  R25's and R26's run directories are present.
- **06g's captures:** `var/levin-phase-floor-06g/references/state/capture/` holds 102 entries,
  19 MB.
- **The venv** is Python 3.13.16, with `datastorekit` 0.5.0, `raypoolkit` 0.3.0, `runregistrykit`
  0.2.0, `ray` 2.59.0, `SQLAlchemy` 2.0.55, `numpy` 2.2.4, `scipy` 1.16.3 and `mpmath` 1.3.0. The
  laptop's are the same.

**The worktrees** are made by the laptop on the user's word to dispatch, after this commit is
pushed, detached and with no venv: `SecondaryGWKit-lpf07-91472db` and
`SecondaryGWKit-lpf07-<this commit's short SHA>`. The Mac Studio's main checkout fetches first,
which moves no working tree. §B.0's checks are repeated then. The Mac Studio orchestrator session
is then started in the code worktree, through the saved `MacStudio` connection, with this note as
its brief. Its folder must be trusted in the app first.
