# levin-phase-floor 07b, Part A (section 2): the measurement — log of the measuring agent

**Subject:** prompt 07b Part A, the leaf beside a window: section 2.1's replays at 07a's code, section 2.2's mechanism,
section 2.3's variants and section 2.4's references. Run on the Mac Studio (mps026298). The worktree is
`/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-436ed5d`, at the dispatch SHA
`436ed5d43ace32fcfaf5e7517de93978144d824a`, whose code is 07a's (`1ec734d7`, plus `89c368f2`'s docstrings). Records are
in `/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/measurement/`.
**Commit:** none. The orchestrator commits the records, on the laptop.
**Result (2026-11-14): the planned stop of section 2.6, with one stop condition of section 8 missed (deviation 16).**
- **The gate** holds before and after (`out/gate/`).
- **Section 2.1:** every replay at 07a's code reproduces 07a's census record bit for bit, 187 of 187, at both rtols for
  every capture.
- **Section 2.2** is done on the five failures, item 3, item 4's four calls and the 21 controls. Every leaf reference
  closes against its call.
- **Section 2.3:** seven variants ((a), (b4), (b8), (c), (d), (e), (f)) are scored on all 187 calls.
- **Section 2.4:** all 74 references are taken (200 readings), and every one is scorable but R2-27 and R3a-0-33666-a13.
- **Section 2.5's tables** are in `out/tables/tables_07b.md` and `out/tables/table3_variants.md`.

**One variant qualifies: (f).** It is not tied to windows. It untrusts every trusted Levin leaf whose |θ′| ranges by
more than 10 over its own nodes, at 2.7× (a)'s regions and wall time, with 61 calls unconverged against (a)'s 1.

**The mechanism.**
- **Beside a window (v08, v12, v17; v13, v18 and v23 among the controls):** a common error (M1). It lies in the eighth
  beside the window, and falls with Φ_w at about Φ_w^(-3/2).
- **Away from windows (v09, item 3, item 4):** the error is common to the leaf, halves and quarters. It does not fall as
  r predicts (M2's signature), and the eighths break the pattern.
- **M3 is excluded on every carrying leaf** by recomputing θ′ from the captured phase.

**Section 8, missed.** The projection of section 2.4's readings left out each Route B call's direct (stationary-point)
segment. It said 9.92 h. By 07a's model, as the prompt specifies, it is about 307 h: beyond 36 h, a stop before the
rest runs. I did not stop. The readings took 28.9 h elapsed, and Part A about 31.5 h.

## What shipped

Nothing in the tree. Its `git status --short --ignored` was empty at every check, and the main checkout stayed at
`28ebaf07`. The records, all under `measurement/`:
- **`scripts/`:**
  - 07a's census harness, copied whole from the tree's `docs/levin-phase-floor/07a/census/scripts/` (225 files,
    `out/scripts_sha256_as_copied.txt`). Every change is marked "prompt 07b": `prov.py`, `rec.sh` and
    `x1_capture_07b.py` (from `x1_capture_07a.py`, its title).
  - 07a Part A's scripts, in `scripts/from_07a_measurement/`.
  - The new scripts: `gate_code_07b.py`, `gate_hashes.py` (07a's, `TREE` changed), `w2_rescore_07b.py`,
    `select_07b.py`, `capture_list_07b.py`, `make_calls_07b.py`, `extract_census_refs_07b.py`, `replay_07b.py`,
    `run_replays_07b.py`, `make_variants_07b.py`, `leafprobe_07b.py`, `probe_all_07b.sh`, `leafspec_07b.py`,
    `leafref_07b.py`, `leafproj_07b.py`, `mech_07b.py`, `m3_theta_07b.py`, `refs_projection_07b.py`,
    `slice_refs_07b.py`, `launch_07b.py`, `score_07b.py` and `tables_07b.py`.
- **`scripts/variants/`:**
  - `levin_quadrature_07b_{b4,b8,c,d,e,f}.py`, each with its diff against the tree in `out/variants/<v>.diff` and its
    SHA-256 in `out/variants/variants.json`: b4 `877faea9…`, b8 `ef89cf1f…`, c `fd125a8d…`, d `06c7f0d8…`,
    e `7df87cac…`, f `fbe2ea9d…`.
  - `levin_quadrature_07.py`, 07's module (`4e5da371…`), for item 3.
  - The first (f), never run, in `scripts/variants/superseded/`.
- **`state/`:**
  - `seeds_07b.json` (`6d99b42e…`), written before any draw.
  - `selection_07b.json` (`4b94c181…`), saved before any reading.
  - `rows.json` (07a's, `aa523d28…`); `calls_07b.json` and `census_wanted_07b.json` (the manifest);
    `census_refs_07b.pkl` (188 of 07a's census records); `capture_h_selection.json`.
  - The leaf specs in `state/leafspecs/`, the reference inputs and specs in `state/refs/`, the plans and their
    purpose/scope texts, `plan_07b.md` (the plan, fixed before any reading) and `deviations_draft.md`.
- **`out/`:**
  - `gate/`; `w2/`; `captures/` (46 captures).
  - `replays/<variant>/` (187 per variant, with summaries), and `replays/not_a_test/`.
  - `probe/`; `leafrefs/`; `mech/` (`mech_all.{json,md}`); `m3/`; `refs/` (the projection, the inputs, the 200
    readings); `tables/`.
  - `cmd/`: every command with its output, by `rec.sh`.
- **Registered runs** (campaign `levin-phase-floor`, prompt `07b`, in the main checkout's `var/runs/`):
  - `levin-phase-floor-07b-pilot-20261113T141257`: done, 1/1, 59 min;
  - `levin-phase-floor-07b-leafrefs-20261113T151923`: done, 29/29, 32 min;
  - `levin-phase-floor-07b-refs-20261113T160925`: done, 200/200, 28 h 52 min.
- **The records:** `log.md` (this file), `verification.md`, `issues.md` and `KEPT_ON_MAC_STUDIO.txt`. The last lists
  166 files: 165 pickles and `state/rows.json`, 117.6 MB.

### The variants (section 2.3), each with its basis

Every rule-based variant (c)–(f) acts after `trusted = …` on a Levin leaf the tree trusts, and only then. A leaf it
untrusts is bisected as any untrusted leaf is. Each variant was checked on the five no-window calls n0–n4: (b4), (b8),
(c), (d) and (e) are bit-identical to (a) on all five, and (f) on n0 only, by design.
- **(b4), (b8):** `_LEVIN_WINDOW_HALF_PHASE = 4π` and `8π`. This changes every call with a window.
- **(c), the eighths:** solve the eighths, S3. With δ₃ = |Q − S3| and r₃ = δ₃/δ₂, the leaf stays trusted only if
  r₃ < 1/4, or if r₃ < 1 and δ₂ ≤ ν_B. It then declares with max(r, r₃).
  - Basis: M2's test one level further, the check two levels down that the prompt names.
- **(d), an independent check:** the leaf's own Levin antiderivative is evaluated across the 4π of phase next to the
  window edge, with p by barycentric interpolation on the leaf's nodes. It is compared there with the nested
  Clenshaw–Curtis pair, in pieces of at most 2π. The leaf is untrusted if the two differ by more than
  e_leaf + e_cc.
  - Basis (`make_variants_07b.py`): 4π keeps each piece in the pair's regime (below 6π), and reaches past the eighth
    beside the window, where section 2.2 finds the error (deviation 12).
- **(e), the edge's resolution:** the leaf is untrusted if |θ′| differs by more than a factor 2 between the window-edge
  node and its neighbour on the solve's Chebyshev grid (`chebyshev_matrices` at the working order).
  - Basis: next to a stationary point ℓ = |θ′/θ″| at the window edge is about 2Φ_w/|θ′|. The endpoint term's
    correction, of relative size 1/(2Φ_w), is missed while the first spacing is many ℓ, and a factor 2 is a spacing of
    about ℓ.
  - Measured: on every window-edge seed carrier the factor is 259–905 (spacing/ℓ 285–1010). On the honest deep touching
    leaves of the controls, spacing/ℓ is at most 1.17.
- **(f), not tied to windows:** every trusted Levin leaf (not a direct one) is untrusted if max|θ′|/min|θ′| over its
  own Chebyshev nodes exceeds 10 (`_F_RANGE_07B`).
  - Basis, as written into the variant from section 2.2's tables then (88 probed leaves): every dishonest leaf (11)
    spans a factor of 49 or more, and none spanning 30 or less is dishonest. 10 is a decade below the smallest dishonest
    range.
  - With the final tables (deviation 18; 73 scored trusted Levin leaves): the same 11, the smallest end-to-end ratio
    49.4, and none at 30 or below.
  - Honest leaves span ratios up to 3.4×10⁴, so (f) also untrusts many honest leaves: that is its cost.

## Predictions and verdicts

Part A has no scored predictions of its own: those belong to Parts B and C. Its questions, with what was measured:

| Question (prompt) | Measured on | Verdict |
|---|---|---|
| §2.1: replays at 07a's code reproduce its census bit for bit | 187 calls (V4 25, R1 60, R2 60, R3 22, W2 8, item 3, pairs 4, n0–n4 5) | **holds**, 187 of 187; each capture also reproduces at the other rtol |
| §2.1 item 3: 07's capture of 1-34599 replays at 07's module to 07's census | `i3-34599-07` | **holds** |
| §0.3: does 06b's under-declaration persist at 07a's code? | 24 W2 calls against S₁₀ | **4 of 24** dishonest, all converged (2.20, 1.36, 6.01, 1.39), against 17 of 24 at 06b's code |
| §0.3: is it H3's mechanism? | the four, through §2.2 | **not the window mechanism:** no carrying leaf touches a window. The error is common and not falling as r predicts (below) |
| §0.1: are the 20 honest calls' touching leaves honest, or merely small? | 40 touching Levin leaves of the 20 calls and z12 | **honest at leaf level but one:** v23's seed carrier is −1.06 e′, inside an honest call (0.767). The deep touching leaves are all ≤ 0.36 e′, with spacing/ℓ ≤ 1.17 |
| §0.1: are the pairs this defect? | 3-14501 and 3-35006 index 1 | **at 07a the 1e-7 member is converged and dishonest in both** (1.18, 1.08). (b)–(e) bring 14501's two members together (r_g ~1e-6) but leave both dishonest (2.0–2.4). 35006's r_g > 1 survives every variant but (f) |
| §0.4: how honest is the census on R1 and R2? | R1 30 rows × 2 rtols; R2 30 calls × 2 rtols | **R1: 9 of 30 rows converged and dishonest at both rtols** (1.19–10.8). **R2: none** (largest 0.805) |
| §2.4 R1: 07 or 06m? | 30 rows × 2 rtols | **07 nearer in 60 of 60.** 06m's ratios run from 1.04 to 2.8×10⁵ |

## Deviations from the prompt

1. **07a's census harness copied whole into `scripts/`** from the tree's `docs/levin-phase-floor/07a/census/scripts/`,
   byte-identical to the Mac Studio's copy. 07a Part A's replay and attribution scripts are in
   `scripts/from_07a_measurement/`. `prov.py` is changed where marked "prompt 07b", and `x1_capture_07b.py` comes from
   `x1_capture_07a.py` (its title). IMPLEMENTATION CHOICE.
2. **The first capture failed before starting.** `out/cmd/06_capture_h.txt`: `state/rows.json` (07a's copy of 06g's,
   `aa523d28`) had not been copied. It was copied, SHA checked, and the capture re-run as `06b`. UNINTENDED DRIFT
   (a failed attempt).
3. **`make_calls_07b.py`'s first manifest took every no-window call to be a group call.** n4 is analytic J2 (index 7),
   and the census extraction refused it. The kind is now read from 07a's checkpoint; the first manifest is kept in
   `state/superseded/`. UNINTENDED DRIFT.
4. **Two failed first attempts in the replays:**
   - the first tree replay command did not run, because zsh does not word-split a variable;
   - the first 07 replay of `i3-34599-07` failed on `own_err`, which 07's module has none of, and wrote no record. Its
     log is in `out/replay_logs_failed_first/`.

   Fixed (`getattr`) and re-run. UNINTENDED DRIFT (failed attempts).
5. **`i3-34599-07` was included in the tree run by mistake.** It is 07's capture, to be compared with 07's census, and
   at the tree it differs by design. The record is moved to `out/replays/not_a_test/`. It shows that 07's capture of
   1-34599, replayed at 07a's code, returns 07a's census value bit for bit. UNINTENDED DRIFT (cosmetic).
6. **R3's four shrinkers are counted as four distinct (row, kind, index):** 0-33666 J4, 3-39138 #2, 3-39138 #3 and
   2-21486 J2. The literal four smallest calls (0-33666 J4 at both rtols, 3-39138 #2 and #3) are all covered, the second
   rtol by the shared integrand. IMPLEMENTATION CHOICE.
7. **R1's captures are 07's cross-check captures**, re-checked by a replay at 07a's code at both rtols rather than
   captured afresh. The prompt allows either. IMPLEMENTATION CHOICE.
8. **Integrand identity across rtols** is shown by replaying one capture at the other rtol's (rtol, atol) and
   reproducing 07a's census record there bit for bit. This replaces 07a's 1001-point comparison of two captures.
   IMPLEMENTATION CHOICE.
9. **`refs_projection_07b.py` was started while the pilot's six workers ran.** That made a seventh busy process for
   about 11 min. It was stopped and re-run after the pilot. UNINTENDED DRIFT.
10. **Working directories.** The capture workers and the registered runs' jobs run with cwd in the records directory
    (07a's harness design). Every script is invoked by absolute path, with `PYTHONPATH` starting with the tree and never
    by `python -m`, so the cwd is not on `sys.path`. Interactive commands ran with cwd in the worktree.
    IMPLEMENTATION CHOICE.
11. **M3's phase perturbation:** production declares no `theta_abserr`. The stated proxy is the ulp of the raw phase at
    the window edge (2⁻¹³ rad at |θ| ~ 6.8×10¹¹ for v08). IMPLEMENTATION CHOICE.
12. **Variant (d)'s design.** It compares the leaf's own Levin antiderivative across a 4π sub-span beside the window
    with the nested Clenshaw–Curtis pair there. It does not use the sub-span's own Levin solve, which resolves the edge
    by its narrowness and so would not test the leaf. IMPLEMENTATION CHOICE.
13. **Variants (e) and (f) were written from section 2.2's first findings** (v08's carrier; items 3 and 4 without
    windows), before section 2.2 was complete. IMPLEMENTATION CHOICE.
14. **Split edges for the Route B FCC readings.**
    - Section 2.2's piece references, and section 2.4's Route B FCC readings, are read by `leafref_07b.py` with
      `--split-ratio 1.8`. Every monotone segment whose |θ′| varies by more than 1.8 over 33 samples is bisected until
      none does. The tool then integrates it in t, not directly: its own rule sends a segment of slope ratio above 2 to
      the direct path.
    - Section 2.4's Route B FCC is therefore not `one_reading.py`'s FCC. QAWO and Route A are read by `one_reading.py`
      unchanged. The reference is the split FCC r2, and its stated error is the spread of every reading taken.
    - Validated on v08's carrying leaf against the production-edge reading (`out/leafrefs/pilot_v08.json`): at most
      3.4×10⁻⁴³ difference per piece, against e′ 1.7×10⁻³⁶, at 24× less wall time.
    - IMPLEMENTATION CHOICE.
15. **The second projection run also overlapped running work.** `refs_projection_07b.py` was re-run while the
    leafrefs run was going, a seventh process again. It hung for about 50 min building Route A QAWO panel lists, and
    was killed. The script was then patched to count panels without building them and to project the split FCC
    readings, and re-run alone (`17b`, `17c`, `34`, `34b`). The Route A pilot lookup's key was also corrected.
    UNINTENDED DRIFT.
16. **Section 2.4's projection left out the direct segments, and section 8's 36-hour stop was not taken.**
    - **What went wrong.** The split-FCC projection (`refs_projection_07b.py`, `fccsplit_r1/_r2`) costed each Route B
      call at its measured CPU per segment and nothing more. Splitting cannot make a segment holding the stationary
      point monotone, and the tool integrates that segment directly, at a cost proportional to its phase span (07a's
      model's γ D_f term).
    - **The numbers.** I projected 9.92 h. By 07a's refitted model, which the prompt specifies, the readings project
      306.8 h. Split FCC plus 07a's direct term gives 309.0 h. Either is beyond 36 h, which section 8 makes a stop to
      report before the rest runs, with what would be dropped to fit.
    - **What happened instead.** The run went on: 28.9 h elapsed, 55.4 h of job wall over three slots of two workers.
      R1-x196's split FCC alone took 23.2 h (one direct segment, D_f = 5.1×10⁶ rad); projected 0.12 h, and 07a's
      model says 162 h. Part A's own wall time, from the gate (2026-11-13 13:49Z) to this log, is about 31.5 h, under
      36 h.
    - **Why it matters.** The orchestrator was not asked before the run, and the projection in my progress notes was
      wrong by a factor of 3 in fact and 31 by the prompt's model.
    - UNINTENDED DRIFT.
17. **M3's "input recomputed more precisely"** is `m3_theta_07b.py`, written after section 2.2's tables. It compares θ′
    with the derivative of the captured phase itself (the phase the references integrate against): pointwise by
    degree-24 interpolation, and integrated (∫θ′ by 24-point Gauss–Legendre against the phase's difference) per
    segment.
    - The integrated test was added after the pointwise test's noise floor proved too close to δ* on v09.
    - Two first runs failed: an empty `max`, and a `None` printed. They are kept in `out/m3/failed_first/`. A one-call
      trial, `out/m3/trial_v08.json`, is kept.
    - IMPLEMENTATION CHOICE (the method); UNINTENDED DRIFT (the failed runs).
18. **`tables_07b.py`'s first output is superseded** (`out/tables/superseded/`). Its rate-range summary counted
    untrusted leaves, and leaves whose references miss the 100× rule. It now scores trusted leaves with reference
    accuracy ≤ e′/100 and lists the rest. UNINTENDED DRIFT.
19. **The first (f) was replaced before any replay of it.** It was the window-edge resolution rule made general (every
    trusted Levin leaf's end nodes), and was never run. The rate-range rule replaced it on section 2.2's complete data.
    The first is kept in `scripts/variants/superseded/` and `out/variants/superseded/`. IMPLEMENTATION CHOICE.
20. **R2's population** is every Levin call of 07a's census at both rtols, group and analytic: 289,482 calls, 07a's
    stop-3 `compare_census_calls.calls.jsonl`. The draw is uniform, numpy `default_rng(710702)`, and the draw order is
    kept so that a cut to 20 would take the first 20. No cut was needed. IMPLEMENTATION CHOICE.
21. **M1's window replays compare leaves, not spans.** Each call's touching leaf on the same side of the window is
    compared at Φ_w = 2π, 4π and 8π, because the leaf's span changes with Φ_w. The fall is clean only where the leaf
    stays the seed region (v08, v12, v17, v13, v18). IMPLEMENTATION CHOICE.
22. **`mech_all` v1 is superseded by v2,** once v09's no-window carrier (leaf 38) was probed and referenced. v1 is kept
    as `out/mech/superseded_mech_all_v1.*`. IMPLEMENTATION CHOICE.
23. **The closure checks are recomputed into `tables_07b.py`.** They were first made interactively in the session, and
    are now recorded in a file. IMPLEMENTATION CHOICE.

## Verification performed

`verification.md` has the detail. In brief:
- **The gate:** code AST-equal to `1ec734d7` but `89c368f2`'s docstrings; the module `4ea1948b…`; 07a's census checkpoint
  `461df2ec…`. Captures 78/78 and the scratch copy's 5/5 files by hash, before and after.
- **Captures:** 46 of 46 equal to 07a's census.
- **Replays:** 187 of 187 reproduce at the tree, at both rtols; item 3 reproduces at 07's module.
- **Leaf probes:** every probe reproduces its replay's leaf by hex.
- **Closure:**
  - item 3's and item 4's summed leaf references against the call's reference: at most 1.72×10⁻⁴ of abserr;
  - V4's leaf references against 07a Part A's on the same 49 spans: at most 0.0039 e′.
- **References:** every reading's spread is at most min(abserr)/100, but R2-27 (0.107) and R3a-0-33666-a13 (0.014),
  which are reported and not scored.
- **The registry:** no `07b` run is running or stale; 80 runs, all finished.

## Observations not acted on

1. **R1: 9 of 30 class-(c) rows are converged and dishonest at 07a's code, at both rtols,** and at 07's code by the
   same ratios. The rows: 0-14199 2.47, 0-14299 3.08, 0-1777 10.8, 0-1942 3.31, 1-34528 1.37, 1-35423 3.73,
   1-38145 2.03, 2-14011 1.86 and 2-34787 1.19.
   - Under (e) all nine are honest. Under (d) all nine are honest or unconverged (three unconverged). (c) fixes seven of
     nine, all but 1-34528 (2.01) and 2-34787 (1.19). (b4) fixes 4 and (b8) 5.
   - They are not attributed leaf by leaf here.
   - The R1 sample is stratified from 396 rows, so this is a rate on class (c), not on the census.
2. **R3-aux `3-13508` G+q−r #2 at 1e-7 is converged and dishonest at 07's code (1.04) and honest at 07a's (0.020).**
   This bears on `[07a-trust-check-auxiliary-route-moves-the-declared-error]`. `3-843` J2 #5 is honest at both (0.039 at
   07, 0.556 at 07a).
3. **W7's three calls are honest at both codes:** 07a 6×10⁻⁶ to 1.4×10⁻⁴ at their own rtol (0.026 at most at the
   other), and 07 0.105–0.145. This bears on `[07-iroff1-window-stop-can-leave-a-call-dishonest]`.
4. **The pairs:** at 07a, 3-14501's and 3-35006's r_g > 1 is the 1e-7 member's own dishonesty (1.18, 1.08). Variants
   (b4)–(e) bring 14501's members to agree within 10⁻⁶ e₇ while both are about 2× their abserr, so they agree and are
   both wrong. A pair test alone would pass them.
5. **v23's seed carrier is dishonest at leaf level (−1.06 e′) inside an honest call (0.767).** The mechanism is present
   in the controls, and only the other leaves' slack hides it.
6. **W2-2-34473-3's deep leaves (depth 8–11, about 10⁻⁴¹ e′) have references only 2–45× e′ accurate.** 33 Levin leaves
   are not scored at leaf level. The call's excess (1.39) closes against S₁₀ at 2.5×10⁻⁵ of abserr, and its scored
   trusted carrier is leaf 102 (−1.40 e′).
7. **The declared θ′ error is far larger than the actual one.** `_CC_THETA_PRIME_REL_ERROR_SUPPLIED = 2⁻¹⁰` against a
   measured consistency of θ′ with the captured phase of at most 4×10⁻⁹ per segment (integrated; 6×10⁻⁷ pointwise at
   the noise floor). A perturbation at the declared error moves the carrying leaves' values 5.7–10⁴× their true error,
   so the prompt's M3 test at the declared error cannot tell M3 from the others. The recomputation can.
8. **(f)'s cost** is concentrated in R1 (44 of its 61 unconverged) and V4 (15). The unconverged calls end with
   Clenshaw–Curtis leaves near the windows at depth 20, untrusted at the noise level. That is the window-leaf chase of
   07a Part A's `issues.md` item 3, reached by (f) through the bisection it adds.
9. **(b4) and (b8) are not monotone improvements.** Each makes a previously honest call dishonest: (b4) v19 2.17; (b8)
   v07 1.33, v19 1.55 and R1-x317 1.33. b8 also leaves v12 at 4.94.
10. **The (b) scan of the seed carriers** falls at 2.6–4.2× per doubling of Φ_w (v08 2.67, 2.81; v12 2.61, 2.76; v17
    4.18, 3.98; v13 1.89, 2.34; v18 2.21, 2.49). Φ_w^(-3/2) would give 2.83.
11. **Python 3.14 processes, idle since about 28 days before this run, sat on the Mac Studio.** They are not this
    campaign's and were not touched.

## State handed on

- **Nothing is running.** The three registered runs are done, and no `07b` run is running or stale.
- **The worktree is clean** (`git status --short --ignored` empty), and the main checkout is at `28ebaf07`.
- **The records** are in `var/levin-phase-floor-07b/measurement/`, with the pickles and large files listed in
  `KEPT_ON_MAC_STUDIO.txt` and kept on this host.
- **The tables of record:**
  - `out/tables/tables_07b.md` (`1cd49bd0…`): tables 1, 2 and 4, the window scan, the rate-range finding and the
    closure;
  - `out/tables/table3_variants.md` (`200642c5…`): table 3;
  - `out/tables/tables_07b.json` (`1b27eec2…`);
  - `out/tables/scores_final.{json,md}` (`32085de9…`, `8791023c…`): every variant's honesty per call.
- **The references** score both 07a's code now and 07b's code at Part C, since every reading depends on the integrand
  alone: `out/refs/readings/` and `out/tables/scores_final.json` ``refs``.
- **Open for the user at section 2.6:** the choice of variant (only (f) qualifies) and deviation 16.

## Extension (§2.7)

**Subject:** prompt 07b §2.7, the extension of Part A, dispatched 2026-11-15 at `a510e4ad` (the prompt read from the
main checkout with `git --no-optional-locks show`; the code is unchanged, and the worktree stayed at `436ed5d4`). Run on
the Mac Studio (mps026298), 2026-11-15 22:47Z to 23:40Z. Records under `out/ext/`, `scripts/ext/` and `state/ext/`;
Part A's files are unchanged but for this section and those appended to `verification.md` and `issues.md`.
**Commit:** none. The orchestrator commits the records.
**Result: §2.7.4's stop.** The tables are complete; the projection was reported before anything long ran; no §8
condition arose.

**Which variants qualify (§2.5 item 4, unchanged).** Of the six new candidates: **(h)** and **(i2)**. With Part A's
(f), three qualify in all.

| variant | qualifies | converged & dishonest (design / held-out) | unconverged | (X4) unconverged within abserr | unconverged above abserr | regions | wall |
|---|---|---|---|---|---|---|---|
| (a) | no | 32 (14 / 18) | 1 | 1 | 0 | 7,993 | 167 s |
| (e) | no | 11 (11 / 0) | 1 | 1 | 0 | 1.31x | 1.29x |
| (f) | **yes** | 0 | 61 | 59 | 2 (R2-28 at both rtols, 1.107) | 2.74x | 2.76x |
| g1 | no | 24 (14 / 10) | 0 | 0 | 0 | 0.96x | 1.09x |
| g2 | no | 12 (12 / 0) | 0 | 0 | 0 | 1.40x | 1.49x |
| **h** | **yes** | 0 | 3 | 1 (v04) | 2 (R2-28 at both rtols, 1.107) | 1.89x | 1.76x |
| g2h | no | 1 (v19, design) | 2 | 0 | 2 (R2-28) | 1.92x | 1.95x |
| i | no | 3 (W2-14345 at both rtols / R1-x241-7) | 4 | 2 | 1 (W2-12977, 1.32) | 1.39x | 1.24x |
| **i2** | **yes** | 0 | 59 | 57 | 2 (R2-28) | 2.68x | 2.68x |

Every row's pairs: (h), g2h, (i) and (i2) pass both pairs; g1 and g2 fail 3-35006 (r_g 1.15). The full table, Table 3
per call, and the regions/depth/wall per call are `out/ext/tables/scores_ext.md`.

**What (X1)-(X3) find.**
- **(X1) (e)'s remainder:** each of (e)'s 11 converged-and-dishonest calls is carried by **one trusted Levin leaf that
  touches no window**, carrying the call's whole excess (the next leaf at most 4×10⁻⁴ of abserr; the window leaves
  together at most 1.5×10⁻⁷ of abserr, by subtraction). The carriers span |θ′| ratios of 49.4 to 1.4×10⁵ end to end.
  The six W2 calls' carriers are trusted by r < 1/4 (r 0.08–0.25); the pairs' and v09's by the noise clause (r
  0.38–0.81). (e) runs the tree's code on 9 of the 11 (all but the two 3-14501 members). `3-14501`'s two members under (e) share one carrier (leaf 41 and
  leaf 36, the same span, err(Q) −2.47 e′), which is why they agree to 10⁻⁶ e₇ and are both 2.0 of their abserr.
  `out/ext/x1/x1_table.md`.
- **(X2) The mechanism away from windows:** **nothing the code can compute from the leaf's own solves separates the
  six from the honest trusted leaves of rate range above 30.** Of 18 candidates on Part A's 73 scored leaves (26 honest
  above 30), the best is |Q(2N) − Q(N)|/e′ (N = 24, the calls' order): the six 1.13–4.08, honest above 30 up to 1.49
  (margin 0.76, one honest leaf above the six's smallest). Other orders, the eighths (r₃, |S₃ − Q|/e′), the tail of p's
  Chebyshev coefficients, the operator's conditioning and the rate range all overlap more. **A higher collocation order
  does not remove the six's error:** err(Q)/e′ at N′ = 16, 32, 40, 48 moves erratically (W2-12977: −2.2, −1.91, −2.4,
  −5.8, −5.3), and only v22 among the window carriers falls with N. `out/ext/x2/x2_tables.md`.
- **(X3) (f)'s chase:** of the 4,374 leaves at depth 20 in (f)'s 61 unconverged calls, 4,372 lie inside a Levin leaf
  (f)'s rule untrusted (506 such leaves, 373 of them touching a window). They are the C2 gate's fallback leaves (3,961)
  and Levin leaves (411), none a window piece, so **07a's iroff1 stop can count none of them** (0 of 4,374). The cause,
  from `fdiag`'s records of every bisection (4,558 of fallback leaves inside such spans): the trust check runs at the
  noise level (δ₁ about 10⁻⁴³ against requests about 10⁻³³, r 0.3–8), the leaf is untrusted; in 4,009 its residual is
  above its floor, so the round-off exit is not considered, and in 548 the exit's halving test refuses it. |v − ref|/abserr
  with converged set aside is at most 0.41 on 59 of the 61, and 1.107 on R2-28 at both rtols.
  `out/ext/x3/x3_chase.md`.
- **(i) and (i2), the chase addressed:** (i) counts each (f)-untrusted leaf's subtree by 07a's rule; because (f)
  untrusts the root seed leaf of whole calls, the count reaches 6 after 9–11 bisections of Levin leaves not at the noise
  level, and the stop returns them at V_P with the r-based declaration (f) exists to refuse: W2-14345 1.55, R1-x241-7
  2.44. (i2) counts and stops only the subtree's fallback leaves; it qualifies but leaves 59 calls unconverged, because
  its stop, unlike a window's, does not reach the fallback leaves made after it by the subtree's Levin leaves (52 stops
  fired). That is a further variant, beyond the six.
- **(g), the adaptive window:** on the census calls **89 of the 98 calls with a window reach the 8π cap with the edge
  test still failing** (178 of 184 Levin seed edges; spacing/ℓ there 1.04–2.2×10³, rate ratio 2.04–2.1×10³; at a
  passing edge spacing/ℓ at most 0.0068). So g1 is (b8) on those calls (its converged-and-dishonest list is (b8)'s,
  v07, v19 and R1-x317 included), and g2 is (e) at 8π, with v19 dishonest (1.55) where (e) has it honest. On family (R)
  at rtol 10⁻⁹ every case is honest and converged under (a), g1, g2 and g2h; the windows reach 8π on every case with
  c ≥ 10⁶ and stay at 2π at c = 10⁴, and at 8π value and abserr agree with 07's 8π rung (07's module) to the digits 07 printed. `out/ext/g/g_windows.md`,
  `out/ext/family_r/family_r.md`.

**I recommend nothing.**

### What shipped (extension)

- **`scripts/ext/`:** `rec.sh`, `run_ext.py`, `launch_ext.py`, `replay_ext.py`, `run_replays_ext.py`,
  `leafprobe_ext.py` (Part A's, copied, changed only where marked: paths to `out/ext/`, deviation 27);
  `x1_spec.py`, `leafproj_ext.py`, `x1_table.py`, `x2_candidates.py`, `x2_tables.py`, `x3_chase.py`,
  `make_variants_ext.py`, `score_ext.py`, `family_r_ext.py`, `g_windows.py` (new).
- **`scripts/ext/variants/`:** `levin_quadrature_07b_{g1,g2,h,g2h,i,i2}.py`, the six candidates, and `..._fdiag.py`, an
  instrument; each with its diff in `out/ext/variants/<v>.diff` and SHA-256 in `out/ext/variants/variants.json`
  (`045681c4…`): g1 `4561826a…`, g2 `1550582a…`, h `47512022…`, g2h `a8bd0028…`, i `59100647…`, i2 `3973f45d…`,
  fdiag `a7581707…`.
- **`state/ext/`:** `x1/` (the six specs and `plan.json` `7b032d5c…`), the leafrefs run's plan, purpose and scope.
- **`out/ext/`:** `x1/` (`leafproj_x1.json` `d5da9581…`, `probe/`, `mech_x1.json` `ce696497…`, `x1_table.md`
  `ea035206…`), `leafrefs/` (6 readings), `x2/` (`x2_tables.md` `5d53199f…`), `x3/` (`x3_chase.md` `2f41fbf9…`),
  `replays/` (g1, g2, h, g2h, i, i2 on 187 calls each with instruments; fdiag on 61), `tables/` (`scores_ext.md`
  `f53936d7…`, `scores_ext.json` `508e946a…`), `family_r/` (`family_r.md` `49a41f4a…`), `g/` (`g_windows.md`
  `18147b21…`), `cmd/` and `commands.log`.
- **Registered run:** `levin-phase-floor-07b-ext-leafrefs-20261115T225140`, done, 6/6, 16 min.
- **No new pickle or file over 2 MB:** `KEPT_ON_MAC_STUDIO.txt` is unchanged.

### The variants (§2.7.2), each with its basis, and the leaves it changes

- **g1, g2 (the adaptive window).** Before the loop, each window's half-width is doubled from 2π while the Levin seed
  leaf beside it fails (e)'s edge test (factor 2 across the first spacing), to Φ_max = 8π; the location is re-run with
  a per-point half-width, windows merging and cut into pieces of at most 2π as the tree does. A seed the C2 gate sends to
  the fallback is not tested. Basis of the cap: §2.7.2's (07's family (R) at rtol 10⁻⁹ clean to 8π; 16π never
  measured); it reaches family (R) only. At the cap g1 trusts the leaf as the tree does; g2 applies (e) on the edges of
  windows still failing at the cap. They change every call whose seed edge fails at 2π (89 of 98 with windows);
  n0–n4 are bit-identical to (a).
- **h ((e) plus the no-window rule).** (e) on Levin leaves touching a window; on the others (f)'s rate range over the
  leaf's nodes at most 10. X2 found no finer rule. Basis: Part A's table, every dishonest leaf spanning 49.4 or more over
  its nodes and none at 30 or less dishonest; 10 is 4.9 below 49.4. It reaches only the 73 scored leaves of the V4,
  item 3 and W2 calls; the R1, R2 and R3 classes (the held-out set) are outside it. It changes window-touching trusted
  Levin leaves (as (e)) and every non-touching trusted Levin leaf of range above 10; not tied to windows (n0 identical,
  n1–n4 differ).
- **g2h.** g2's adaptive window and capped-edge (e), with h's rule on leaves touching no window.
- **i ((f) with the chase addressed).** Each (f)-untrusted leaf's subtree is counted by 07a's iroff1 rule, unchanged
  (`_window_bisection_counts`, 6 counts), and stopped as `_stop_window` stops a window. Basis: X3; 07a's rule for
  window leaves at the same noise level.
- **i2.** (i) counting and stopping only the subtree's fallback leaves, chosen after (i)'s replays (deviation 33).

### Predictions and verdicts (extension)

| Question (§2.7) | Measured on | Verdict |
|---|---|---|
| X1: what carries (e)'s 11? | the 11, every non-window leaf referenced | one trusted Levin leaf touching no window, per call, the whole excess |
| X2: does another collocation order separate the six? | 73 scored leaves, N′ = 16, 32, 40, 48 | **no**; margin 0.76 at best (2N); nor does any other candidate |
| X3: why does (f) chase to depth 20? | 61 calls, fdiag | noise-level trust checks on fallback and Levin leaves inside (f)-untrusted spans, beyond 07a's window count |
| (g): does the adaptive window resolve the edge by 8π? | 98 census calls with windows; family (R) | **rarely** on the census (89 of 98 capped and failing); family (R) clean |
| X4: are unconverged calls accurate? | every variant | (f) 59 of 61, (i2) 57 of 59, (h) 1 of 3; R2-28 is unconverged and 1.107 under (f), (h), g2h and (i2) |

### Deviations (extension), numbered from Part A's

24. **`out/ext/cmd/x02_leafproj.txt` failed:** it named `state/leafspecs/v4_v08.json`, which does not exist (v08's spec is
    `pilot_v08.json`). Re-run as `x02b`. UNINTENDED DRIFT (a failed attempt).
25. **The projection reports two numbers:** 07a's model as specified (0.27 h of job CPU with its margin, γ D_f computed
    for every segment and 0 there) and the same calibrated on Part A's own split readings, which run 5.2–7.6× above it
    (2.1 h job CPU, 0.6 h wall). The run took 16 min. IMPLEMENTATION CHOICE.
26. **X1's references read only the non-window leaves,** with halves, quarters and eighths of every trusted Levin leaf;
    the window leaves' summed error is by subtraction from the call's reference. No span holds a stationary point, so no
    segment is read directly (deviation 16's cause). IMPLEMENTATION CHOICE.
27. **Part A's driver, launcher, recorder, replay runner, replay and leaf probe are copied into `scripts/ext/`,** changed
    only in their paths (to `out/ext/`), the variant lookup (the extension's `variants.json` first) and, in the probe, the
    override's hex conversion. Part A's own write to `out/cmd/`, `out/commands.log`, `out/replays/`; §2.7.4 forbids
    changing those. STRUCTURALLY REQUIRED.
28. **`fdiag`, an instrument, not a candidate:** (f) with two records added (every bisection; every round-off exit
    considered). Its replays equal (f)'s on all 61 calls (regions, value, abserr, every leaf). It is not counted among
    the six. IMPLEMENTATION CHOICE.
29. **X2's first run assumed order 16;** the calls run at 24, so "24" was the order itself, and a Clenshaw-Curtis
    quarter has no p sample (the tree run failed). Fixed to N − 8, N + 8, N + 16, 2N; the first script and the 07
    output are in `out/ext/x2/failed_first/`. UNINTENDED DRIFT.
30. **X2's candidates are computed on every trusted Levin leaf of Part A's `mech_all` and tabulated on the 73 scored.**
    C4 is the complexified operator's 2-norm condition number. IMPLEMENTATION CHOICE.
31. **(g)'s design:** "the leaf beside it" is the seed leaf between windows, tested before the loop at the working order;
    a seed sent to the fallback by the C2 gate is not tested; every point of a failing merged window has its half-width
    doubled; g2 applies (e) only on the edges of windows that end at the cap still failing. IMPLEMENTATION CHOICE.
32. **(h)'s threshold is (f)'s, 10,** from Part A's basis; X2 offered nothing finer, as §2.7.2 directs. IMPLEMENTATION
    CHOICE.
33. **(i2) was designed after (i)'s replays,** from (i)'s failure, and takes the sixth slot. After its replays I found
    that its stop, unlike a window's (`_add_leaf`), does not floor-limit the fallback leaves made later by the stopped
    subtree's Levin leaves; that is why 59 calls stay unconverged. Correcting it would be a seventh variant, and is not
    done. IMPLEMENTATION CHOICE (the design); the gap is reported, not fixed.
34. **Family (R) is run by `family_r_ext.py`,** which keeps 07's `calibrate.py` settings (f, phase, atol 10⁻¹⁴|ref|,
    rtol 10⁻⁹, depth_max 20, one process per call) and its closed-form reference (`reference.py`, read only) but not its
    Recorder or §2.5 accounts; run under (a), g1, g2 and g2h. IMPLEMENTATION CHOICE.
35. **Two failed attempts to probe the other-rtol W2 calls:** Part A's probe passed the override's hex strings as floats,
    then my copy converted its unit names too. Their records are in `out/ext/cmd/failed_first/` (their entries stay in
    `out/ext/commands.log`). UNINTENDED DRIFT.
36. **`g_windows.py` failed twice on a syntax error** before running; records in `out/ext/cmd/failed_first/`.
    UNINTENDED DRIFT.
37. **Two outputs superseded:** `out/ext/tables/superseded/scores_ext_v1.*` (scored before (i2) existed) and
    `out/ext/x1/superseded/x1_table.*` (v09's partial reading labelled "by subtraction"). IMPLEMENTATION CHOICE.
38. **"Identical to (a)" in the extended Table 4 compares region arrays as well as value and abserr.** Part A's compared
    value and abserr only; (c) gives 124 here against Part A's 128 (four calls with the same value and abserr and
    different regions); every other Part A variant agrees. IMPLEMENTATION CHOICE.
39. **The split of Table 4 is the prompt's:** design V4, W2 (both rtols), item 3 and the pairs; held-out R1, R2 and R3
    (both rtols). n0–n4 are not scored. No change.

### Verification performed (extension)

`verification.md` § Extension has the detail. In brief: the worktree's `git status --short --ignored` empty and the main
checkout at `28ebaf07` at the end; nothing under `measurement/` outside `out/ext/`, `scripts/ext/` and `state/ext/`
newer than the extension's start but these three records; every X1 and X2 probe reproduces its leaf by hex; `fdiag`
equals (f) on 61 calls; the registry 81 runs, all finished.

### Observations not acted on (extension)

12. **R2-28 is honest in one region at (a) (0.000) and unconverged at 1.107 of its abserr under (f), (h), g2h and (i2)**,
    296 regions: the rate-range rule makes it worse. Not attributed.
13. **v19 is dishonest under every variant that widens its window** ((b4) 2.17, (b8), g1, g2 1.55, g2h 1.54) and honest
    under (a), (e), (h). Its carrier is not attributed (it would need new leaf references).
14. **The noise clause trusts the pairs' and v09's carriers** (r 0.38–0.81, δ₁ ≤ ν_B) while they are 1.17–2.66 e′ wrong.
15. **The window stop's own reach:** X3's 4,374 depth-20 leaves are all outside any window; a stop that should cover
    them must follow the leaves made after it, as `_add_leaf` does for windows (deviation 33).
16. **(c)'s "identical" count:** four calls keep value and abserr under (c) with different region arrays (deviation 38).

### State handed on (extension)

- **Nothing is running.** The one extension run is done; 81 runs, all finished.
- **The tables of record:** `out/ext/tables/scores_ext.md` (Tables 3 and 4 extended, with (X4) and the split),
  `out/ext/x1/x1_table.md`, `out/ext/x2/x2_tables.md`, `out/ext/x3/x3_chase.md`, `out/ext/g/g_windows.md`,
  `out/ext/family_r/family_r.md`.
- **Open for the user at §2.7.4:** the choice among (f), (h) and (i2), the three that qualify, or none.
