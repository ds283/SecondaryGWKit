## Your brief: the references agent — prompt 06g §5, the references (V3, V4 and the correction $-\sum E$)

**Your scope is prompt 06g §5's "The references"**, with V3 and V4 scored from them, and the
descriptive item "The correction $-\sum E$". These parts are not yours:
- §5's census, Gate V, V2, V5–V8 and the other descriptive items belong to the census agent;
- §4, R23 and R24, belongs to the rehearsal agent.

**Read, besides the common brief's list:**
- prompt 06g §0.2 (e), §1's item on `tools/filon_reference.py`, §2's test 8 and §5 in full;
- `tools/filon_reference.py` and `tools/tests/test_filon_reference.py` in the code worktree, and the
  code log's deviations 10–12;
- prompt 06i, `prompts/levin-phase-floor/06i-filon-references.md`, in full, and its records
  `docs/levin-phase-floor/06i/` (`verification.md`, `issues.md`, `scripts/`, `out/`, and
  `inputs_SHA256SUMS.txt`). Its counting rule, its agreement rule between routes and its handling
  of sample 8 matter;
- Z3's calls, `docs/levin-phase-floor/06c/depth20-tiling/results.json` (`calls`: 30 entries with
  `sample_no`, `rtol`, `unit`, `index`), and 06e's Z3 record in `docs/levin-phase-floor/06e/census/`;
- 06e's leaf breakdown `docs/levin-phase-floor/06e/investigations/d3-leaf-values/` (V4's basis);
- the capture scripts: 06h's `x1_capture.py` and `make_capture.py`
  (`docs/levin-phase-floor/06h/census/scripts/`) and the phase agent's use of them
  (`docs/levin-phase-floor/06h/phase/`).

**Where you write:**
- only `var/levin-phase-floor-06g/references/` (with `out/`, `scripts/` and `state/`) and your own
  run directories in `var/runs/`;
- you take no lock;
- no process of yours initialises Ray (`ray.cloudpickle` as an unpickler is fine);
- nothing under `var/datastores/` is created.

**06i's 13 inputs on this host** are `var/levin-phase-floor-06e/census/references/inputs/s*.pkl`
(06e's captures, made at 06e's census tree `6d81fad` with 06e's census harness,
`docs/levin-phase-floor/06e/census/scripts/`). The orchestrator checked today that their SHA-256
equal `docs/levin-phase-floor/06i/inputs_SHA256SUMS.txt` line for line. **Read them only**; copy
them into your `state/` if a tool needs them beside a harness, and record the copies' SHA-256.

**The sample, fixed before any reading** (prompt §5):
- **Z3's 30 calls,** matched in the $g$ census by row (`unit`), rtol and call index.
  - **06i's 13** (samples 1, 2, 3, 8, 9, 10, 13, 14, 15, 22, 23, 25, 27) keep their references
    (`docs/levin-phase-floor/06i/out/readings/`) **if** the captured groups at $g$ equal those 06i
    read: coefficients, orders, signs and spans, bit for bit. Say what you compared and how.
  - **The 17 others** are captured at $g$ and referenced.
- **A random sample of 60 calls** from 06h's census, outside the 720 stationary-point rows
  (06a's `spscan.py`) and Z3's 30: 30 at each rtol, at each rtol 15 analytic-block calls and 15
  cosmological-group calls, each drawn uniformly among calls with finite `abserr`. Captured at $g$
  and referenced.
  - **The seed** is chosen by you and written into your predictions file **before** the draw.
  - The draw script is saved in `scripts/` and its output, the 60 calls, in `out/`.
- **Every capture at $g$** (the code worktree, $g$'s census harness on `PYTHONPATH`) is checked bit
  for bit against the $g$ census's checkpoint record of the same call: value, abserr and regions.
  The census agent's run, `var/runs/levin-phase-floor-06g-census-*`, will be complete before you
  compute; read it only. A capture that does not reproduce its census record is a stop.

**The tool's check on 06i's 13** (prompt §7; the code log's deviation 12): run
`tools/filon_reference.py` on 06i's 13 calls and compare its readings with 06i's. **Fix the
agreement rule in your predictions file before running it**, taken from 06i's own rule between
routes and resolutions (prompt 06i, its §on agreement: within the sum of the two scatters plus 1/100
of the minimum abserr, or as 06i's records define it), and say which captures you feed it (06i's
inputs at 06e's tree, or your $g$ captures where the groups are bit-identical) and why. **A reading
that disagrees with 06i's on one of its 13 is a stop.** 06i's sample 8 was accepted on the user's
ruling; read how, and apply the same reading.

**The references:**
- `python -m tools.filon_reference PICKLE --tolerance T --unpickler ray.cloudpickle --workers 6`,
  with $g$ and the census harness on `PYTHONPATH` (the captures name the harness's functions), as
  the addendum says. Route B's default edges walk the captured integrand's closures as 06i did; a
  capture whose closures differ raises, and its edges must be passed (each such case a deviation).
- **One registered, detached run** with slug `references` (`--campaign levin-phase-floor --prompt
  06g`), at most 6 processes in all. The driver that loops over the samples is a script of yours in
  `scripts/`, its launch as 06h's phase agent launched `launch_06h.py`.
- **A reference counts** when its four readings agree to within 1/100 of
  $\min({\rm abserr}_{06h}, {\rm abserr}_g)$. Those that do not are reported with their scatter and
  dropped. **A reading over 30 minutes is stopped,** reported, and that sample's reference dropped:
  the driver enforces this, and records each reading's wall time.
- **Fewer than half of the 77 new samples counting** (fewer than 39) is a stop (prompt §7).
- Every reading is saved in `out/readings/` in 06i's form, with its route, implementation,
  resolution, wall time and value in `float.hex`.

**The predictions:**
- Copy prompt §5's V3 and V4 verbatim into `log.md` and into a timestamped file under `state/`,
  **before** any number is computed, together with the seed, the agreement rule for 06i's 13, and
  how you compute ${\rm abserr}_{06h}$, $v_{06h}$, ${\rm abserr}_g$, $v_g$ and $\sum E$ for each
  call (from which checkpoint fields).
- Do not edit that file afterwards.

**Scoring:**
- **V3:** holds, fails or undecided against its rule as written. Report the number counted (of 90:
  13 + 17 + 60), every violation with $v_g$, ref, ${\rm abserr}_g$ in `float.hex`, the worst
  $|v_g - {\rm ref}|/{\rm abserr}_g$, and $|v_{06h} - {\rm ref}|/{\rm abserr}_{06h}$ beside each.
- **V4:** holds, fails or undecided against its rule as written, on Z3's samples 2, 8, 9 and 14,
  with each ratio and 06e's breakdown's number beside it.
- **Descriptive, the correction $-\sum E$:** on every counted reference with a Clenshaw–Curtis
  leaf, $|v_g - \sum E - {\rm ref}|$ against $|v_g - {\rm ref}|$ and against ${\rm abserr}_g$.

**Sequencing:**
1. **Now, while the rehearsals and the census run:**
   1. read;
   2. copy the scripts you will use, and write their diffs against their sources;
   3. write your draw script, capture script and driver;
   4. plan the captures and the 06i comparison.

   **Compute nothing:** no capture, no draw, no Filon reading, no hashing of the store or of a
   checkpoint, no unpickling of 06i's inputs beyond reading their SHA-256 list. Other stages hold
   the six cores. End your turn reporting that you are ready, with the commands you will run.
2. **When the orchestrator resumes you,** after the census is scored:
   1. hash the scratch copy and the $g$ census's checkpoint;
   2. write the predictions, with the seed;
   3. draw the random sample;
   4. capture the 77 new calls and check each against the census record; compare 06i's 13 groups;
   5. run the tool on 06i's 13 and compare with 06i's readings (in session if under about ten
      minutes in all, otherwise in the registered run's first stage, ending your turn at launch);
   6. launch the references run, make one check, and end your turn with the launch report.
3. **When resumed with the run's final `status.json`:**
   1. hash the scratch copy again;
   2. score V3, V4 and the correction;
   3. write your records;
   4. report in full.

**Your final report** gives:
- the seed and the 60 calls drawn;
- the captures' bit-for-bit checks against the census, and the 13 groups' comparison;
- the tool's readings on 06i's 13 against 06i's;
- every sample's four readings, scatter, wall time and whether it counted;
- V3 and V4, each holds, fails or undecided, with numbers;
- the correction $-\sum E$;
- the store's hashes before and after;
- your deviations;
- anything in `issues.md`.
