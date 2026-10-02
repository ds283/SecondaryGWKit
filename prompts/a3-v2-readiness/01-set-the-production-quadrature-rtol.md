# Prompt 01 — set the production quadrature `rtol` to `1e-7`

**Campaign:** [`README.md`](README.md) · **Board items:** **T1**–**T2** ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Gate:** none. Independent of 00; `git status` must be clean.
**Closes:** `[00-quadsource-extraction-reads-at-the-default-rtol]`, on this board's §4, with its
index row deleted. **Opens:** nothing expected.
**Recommended model:** **Sonnet**. One constant and its record. The care is in finding every
place that states the old value, and in not editing the measurement records that used it.

**Read first:**

1. [`README.md`](README.md): §0 item 7, §1, §5 and §6.1 (D1, D2).
2. `docs/handover/QUADSOURCE-TOLERANCE-SWEEP.md` §0 and §7: why `1e-7`.
3. `config/defaults.py`: `DEFAULT_QUADRATURE_RTOL`, `DEFAULT_QUADRATURE_ATOL` and their comments.
4. `docs/TOLERANCE-PROVENANCE.md`: the `DEFAULT_QUADRATURE_RTOL` entry and standing note 26.
5. `ComputeTargets/tests/test_main_plumbing.py`: the test that pins which constants each reader
   queries `QuadSourceIntegral` under.

---

## 1. What is wanted

A3 v2 is built at `(atol, rtol) = (1e-32, 1e-7)` (D1). Today only the build knows that: it gets
there by substituting `tol=SWEEP_QUAD_RTOL` for `tol=DEFAULT_QUADRATURE_RTOL` in `main.py`'s source,
while `config.defaults` still says `1e-8`. So `extract_QuadSourceIntegral_data.py`, which looks its
rows up at `DEFAULT_QUADRATURE_RTOL`, cannot find a single `QuadSourceIntegral` row in A3 v2, and
`main.py` run directly would build a different store. The user decided (D2) that the default
itself moves, so that every consumer shares one constant.

---

## 2. What to change

**T1 — the constant.**
- `config/defaults.py`: `DEFAULT_QUADRATURE_RTOL = 1e-7`. Give it a comment of its own, in the
  style of the comments around it, saying: the value; that it is the loosest rung the 2026-09-23/24
  sweep found converged in all three response-redshift bands, worst deviation from each band's
  tightest rung 9.97e-07 / 1.9e-06 / 2.84e-06, against 1.1e-02 for `1e-6` in the mid band; that
  low $z_{\rm response}$ is representation-limited, so a tighter rung there buys cost and no
  accuracy; that it was `1e-8` until this commit; the convergence check for a science output is one
  rung tighter, `1e-8` (D1); and a pointer to `QUADSOURCE-TOLERANCE-SWEEP.md`. The existing comment
  below it, which is about `atol`, stays where it is and says what it says.
- **Every comment or docstring in production code that states the quadrature `rtol`'s value** is
  corrected to name the constant rather than a number, or to the new number where a number is the
  point. Known: `ComputeTargets/QuadSourceIntegral.py`'s comment in `analytic_rad`'s neighbourhood
  that says the pipeline supplies `1e-25` and `1e-8` (both are stale). Find the rest with a search
  for `1e-8` and `1e-08` near "quadrature", "rtol" or "QuadSource", and list every hit in the log
  with what was done to it.

**T2 — the record.**
- `docs/TOLERANCE-PROVENANCE.md`: a new subsection beneath the `DEFAULT_QUADRATURE_RTOL` entry,
  dated, recording the change, the sweep it rests on and D1. The entry above it is not rewritten
  (README §5 rule 6).
- The `handover` board's `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]`:
  a dated line saying the shipped `rtol` is now `1e-7`. **It does not close that issue**: the dead
  `local_atol` branch and the depth cap it describes are unchanged; only the pair is the one the
  sweep found converged.
- A test, in a new module under `ComputeTargets/tests/`, pinning `DEFAULT_QUADRATURE_RTOL == 1e-7`
  and `DEFAULT_QUADRATURE_ATOL == 1e-32`, whose docstring says why (the sweep, D1) and that
  changing either is a decision that also changes which rows every store's readers find. A value
  pin is weak on its own; its job is to make a change deliberate.

---

## 3. Verification

1. All six suites at `HEAD` with every existing test unchanged.
2. `substitute_main_py(filter_cases=False)` still matches each of its four substitutions once
   (`main.py` is not edited, so this is a check, not a fix).
3. **Deliberate breakage.** Each must make a test fail; then restore it:
   - (i) `DEFAULT_QUADRATURE_RTOL = 1e-8`;
   - (ii) `DEFAULT_QUADRATURE_ATOL = 1e-25`;
   - (iii) make `extract_QuadSourceIntegral_data.py` look its `rtol` up under
     `DEFAULT_GK_NUMERIC_REL_TOLERANCE` instead. `test_main_plumbing` already catches this; show it.

   Each in the log as a diff, exactly as applied, with the failing tests.

---

## 4. Acceptance

1. The constant, its comment, and every stale statement of its value, each listed in the log.
2. The provenance subsection and the `handover` board line.
3. The new test, and the deliberate-breakage record (i)–(iii).
4. `[00-quadsource-extraction-reads-at-the-default-rtol]` moved to this board's §4 with a closing
   note naming this commit; its index row deleted; the count and date corrected.
5. `black --check` clean. On this board: the §1 row, items T1–T2, §3, §4, §5. All in the same
   commit.

---

## 5. Stop conditions — stop and ask the user

- An existing test fails at `1e-7` because a pinned value moved. That test is measuring something
  at the old default; whether it is re-pinned or given an explicit `rtol` is the user's call.
- A production path uses `DEFAULT_QUADRATURE_RTOL` for something other than a quadrature
  tolerance. (A region guard once did — the comment above `MIN_SUBINTERVAL_LOG_WIDTH` in
  `ComputeTargets/QuadSourceIntegral.py` records its removal; check that none does now.)

---

## 6. What this prompt does not do

- It does not touch `main.py`, the sweep script or its `--build`'s explicit `--atol`/`--rtol`.
  After this commit a build at any other pair is again unreadable by the extractor; record that in
  the log's **Observations not acted on**, and do not fix it here.
- It does not re-run or edit the measurement records that read the default at their own tree:
  `docs/tolerance-convergence/quadsource_readonly.py` and `QUADSOURCE-READONLY.md`,
  `docs/tolerance-convergence/inventory.py` and `TOLERANCE-INVENTORY.md`. Name each in the log as a
  record whose re-run would now measure at `1e-7`.
- It does not change `DEFAULT_QUADRATURE_ATOL` or `DEFAULT_LEVIN_MAX_DEPTH`.

---

## 7. The log and the board

`logs/01-set-the-production-quadrature-rtol.md`, using README §5.1. In addition: every hit of the
§2 T1 search, with what was done to it.

`IMPLEMENTATION_STATE.md`: the §1 row for 01, items T1–T2, §3, §4 and §5.
