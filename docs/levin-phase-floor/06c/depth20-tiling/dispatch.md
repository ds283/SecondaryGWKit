# Depth-20 lines against the end-node tiling mechanism: dispatch brief

- **From:** the Mac Studio orchestrator, 2026-10-29, host mps026298.
- **To:** a fresh agent on the Mac Studio. This brief is binding.
- **Origin:** the user asked for this through the laptop orchestrator. The user's words: "When that happens, I think we should first check the production depth-20 lines to see if their issue is the same tiling problem."

## Background

**The census.** Prompt 06c's census ran 06b's census rows under R18's code, tree `e5764a6`. That code is 06c's global error budget plus 06d's φ_CC floor.
- Its run directory, read only: `var/runs/levin-phase-floor-06c-census-20261029T130214/`, whose `checkpoint.jsonl` holds one record per unit.
- Its records, read only: `var/levin-phase-floor-06c/census/`.
- 262 moved Levin calls reached `depth_max = 20`: 109 at rtol 1e-7 and 153 at 1e-8, on 144 units.

**The deepest lineage of every depth-20 call** is in `var/levin-phase-floor-06c/census/out/deep_lines_06c.json`.
- Each entry gives the unit, the call `index`, the `notify_label`, and the deepest leaf's `leaf_start` and `leaf_end` in hex.
- Each entry also gives `levels`, the line from the root to that leaf, one item per level. Work out the fields' meaning from the census script that wrote them (`census/scripts/analyse06c.py` or `worker.py`). They include the branch, the step-(4) residual, the floor, the truncation, the error e and the ratio (e_L+e_R)/e_P.

**The orchestrator's review** is in `var/levin-phase-floor-06c/review/checks/check9_deep_lines.txt`. Over the last five levels of the 262 lines, 1,310 levels in all:
- the residual dominates e in 1,179;
- residual/floor has a median of 1.97e4.

**The candidate mechanism.** A leaf study on the laptop calls it "tiling". Read its records in `var/levin-phase-floor-07-leaves/`: `out/tables.md`, especially its last two sections, `scripts/endnodes.py`, `scripts/q3_decompose.py` and `scripts/lc.py`. Its companion is `var/levin-phase-floor-floor-breakdown/`, for the mpmath re-solve method. The mechanism:

1. **The end nodes miss.** `chebyshev_matrices` in `AdaptiveLevin/levin_quadrature.py` forms the grid's end nodes as (a+b)/2 ± (b−a)/2·1. In about a third of spans that misses a or b by one ulp, although its docstring says x[0] is the upper endpoint.
2. **So the leaf shifts.** A Levin leaf then integrates a slightly shifted interval, which costs about |f|·ulp(x).
3. **The residual sees the shift.** The step-(4) residual |V_P − (V_L + V_R)| picks up that offset whenever the computed end nodes of the parent and its halves fail to tile. Tiling means:
   - the left half's bottom node equals the parent's;
   - the right half's top node equals the parent's;
   - the left half's top node equals the right half's bottom node.

   In the study, 301 of the 302 flat-residual leaves did not tile.
4. **The floor does not contain it.** The eq. (151) floor does not include the offset. So such a leaf is never floor-limited, its residual never falls, and it is bisected towards `depth_max`.

**Note:** the leaf study ran on `e5764a6` plus prompt 07's staged patch, and on analytic test problems. **You run on `e5764a6` unpatched, on production integrands**, which are double-precision interpolants.

**Also relevant:** the Y1 study just done in `var/levin-phase-floor-06c/y1-references/`. Read its `results.md`. It found stationary points of θ′, at z≈10211 in equilateral `G+q+r` calls, that a coarse region swallows. Its `scripts/refquad.py` is an independent Gauss–Legendre reference that you may copy.

## The task

**Step 0: fix the sample.** First, before looking at any result beyond the inventory, fix the sample for the expensive parts.
- Write the rule to `sampling_rule.md` and record its SHA-256 in `log.md`.
- The sample has **at least 30 calls**.
- It covers **both tolerances**, and **every source block present** at each. The blocks are the group labels `G+q+r`, `G+q-r`, `G-q+r`, `G-q-r` and the analytic labels `J1`–`J4` and `Y1`–`Y4`, as `notify_label` names them; the census has depth-20 calls in all twelve at both tolerances.
- The rule is mechanical, for example: in each tolerance × block cell, the call with the smallest `(unit, index)` under a stated hash order, plus a stated stratified fill.
- Prefer units with a short census `wall_s`, but only if the rule says so in advance.

**Step 1: the cheap pass, over all 262 calls if possible.** From the lineage records and the census checkpoint, for each level of the deepest lineage:
- the parent's span and its halves' spans, reconstructed exactly. The halves are split at `(b + a) / 2.0`, the driver's own midpoint; check the driver for the exact expression. The call's `x_span` is needed: find it in the checkpoint or the capture.
- whether the parent's and its halves' **computed** end nodes tile, calling `chebyshev_matrices` at the call's Chebyshev order from `e5764a6`;
- whether the parent's computed end nodes equal a and b;
- residual / floor.

Confirm that your reconstruction reproduces the recorded leaf, `leaf_start` and `leaf_end`, bit for bit.

If the spans cannot be reconstructed without a capture, say so, and do this pass on the sample only.

**Step 2: per sampled call, the inputs.**
- **Use the census's captured inputs where they exist:** `census/state/X0`, `X1capture` and `X4capture`, with `x1_captured.json` and `x4_captured.json` as their index.
- **Otherwise re-capture by the census's own route:** `census/scripts/x1_capture.py` with `worker.py` and `harness.py`, re-running the unit in a fresh census worker at `e5764a6`. Copy the scripts into your own `scripts/` and change them there.
  - That route reads 06's certified scratch copy of A3 v2, `var/levin-phase-floor-06/wider/state/a3v2_copy/`, **read only**, through a read-only pool, as the census did.
  - Before the first capture and after the last, record that copy's hash and fingerprint the way the census and review did (`census/scripts/hash_store.py`, `fingerprint_copy.py`; the review's `check6_copy_fp.txt` holds `0e32f24a…`). Show it unchanged.
  - Never write to it, and never open A3 v2 itself.
- **A capture counts only if** replaying it in a fresh process at `e5764a6` reproduces the census call bit for bit: value and abserr by `float.hex`, plus `num_regions`, `max_depth` and `converged`.

**Step 3: per sampled call, classify it.**

- **(a) Tiling.** On the deepest lineage, the leaves whose residual exceeds 10× their floor fail to tile, and a re-solve attributes the residual to the end-node offset.
  - **The re-solve.** Re-solve the parent and its halves, in double and in mpmath at ≥ 50 digits, once with the computed end nodes and once with the end nodes pinned exactly to a and b.
  - **What stays fixed.** Keep every other node the same. Evaluate f and θ in double at whatever double nodes each grid holds, because the integrands are double interpolants.
  - **Attribution** means: with the ends pinned, the residual falls to ≤ 10× the floor, and the residual's change matches the offset terms.
  - The leaf study's `q3_decompose.py` and the floor-breakdown scripts show the scheme. Adapt them, and say exactly what you changed.
- **(b) Stationary point.** θ′ changes sign, or |θ′| comes within 1e-6 of its maximum over the span, inside the region of the deepest lineage, as in Y1's `G+q+r` at z≈10211. Give x, z, θ′ and θ″ there.
- **(c) Other.** Name the component that holds the residual up.

A call may be both (a) and (b). Say so, and classify it by which one holds the deepest levels.

**Step 4: per sampled call, the counterfactual.** Re-run the captured call with the grid's end nodes pinned exactly to a and b.
- **How to pin.** Patch `chebyshev_matrices` inside your measurement script only, with `unittest.mock.patch` or a module-attribute swap in your process. **Never edit the tree.**
- **What the patch changes.** Only x[0] := b and x[−1] := a; every other node and every matrix stays the same. Show that the patch changes nothing else: on several spans, compare its output with the original, element by element.
- **Report, against R18's census call:**
  - depth reached, number of regions or leaves, and wall time;
  - value and declared abserr, both codes' in hex.
- **Where a reference exists, give the true error both ways.**
  - None of the Y1 references are census calls. So compute an independent Gauss–Legendre reference with `refquad.py` from y1-references, or your own, for as many sampled calls as the budget allows.
  - Converge it as Y1's were: two resolutions, differing by ≤ 1/100 of the smaller declared error.
  - Say which calls have one.

  This answers whether pinning the ends stops the chase without making the declared error dishonest.

**Step 5: totals.**
- counts in (a), (b) and (c) at each tolerance, and by block;
- the median and maximum of residual/floor;
- the depth distribution before and after pinning;
- the pinned runs' honesty, |value − ref| / abserr, where a reference exists.

## Trees and commands

- **The tree:** R18's code, `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06c-e5764a6`, at `e5764a6`, detached and clean.
- **Other trees, read only:** `fef401c`, the main checkout's `28ebaf0`, and git history.
- **Every Python command:** `/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python -B`, with `PYTHONPATH=<tree>:<your scripts dir>`. The census pickles name `harness`, so put a scripts directory holding `harness.py` on the path.

**Records** go only in `var/levin-phase-floor-06c/depth20-tiling/`:
- `scripts/`;
- `out/`, holding every command's output, numbered;
- `state/`;
- `log.md`;
- `sampling_rule.md`;
- `results.md`, holding the tables and the one-paragraph answer;
- `results.json`, holding the per-call numbers in hex.

**Every output starts with:**
- the host;
- the UTC time;
- the tree's `git rev-parse HEAD` and `git status --short`;
- the script's SHA-256;
- the SHA-256 of every input.

## Binding rules

- **No git command that writes, in any tree. That includes fetch.** Nothing is committed.
- **No Ray.** The `ray` package may be imported, but never initialised. Never `ray stop`, and never touch the rehearsal lock.
- **No store writes.** The A3 v2 scratch copy is read only, through a read-only pool, and only by the capture route. A3 v2 itself is never opened.
- **Never write to:**
  - any `var/runs/` directory;
  - any earlier prompt's records;
  - `var/levin-phase-floor-06c/{census,rehearsal,review,y1-references}`;
  - `var/levin-phase-floor-07-leaves/`;
  - `var/levin-phase-floor-floor-breakdown/`.

  Read them only.
- **At most 6 CPUs at once:** six worker processes at most.
- **Wall time:** run anything over about two minutes in the background, and wait on it once. **Never poll, never `pgrep -f`.**
  - Size each computation before launching it. If any single one would take longer than about 45 minutes, **stop before launching it**, and report the estimate and the command in your final message. Such a job must be registered in the run registry, and the laptop orchestrator told first.
  - Batches each under 45 minutes are fine.
- **Write no production code.** Your scripts are readers, captures, re-solves and quadratures.

## Deviations and stopping

**Log deviations.** Record every deviation from this brief in `log.md`. Classify each as `STRUCTURALLY REQUIRED`, `IMPLEMENTATION CHOICE` or `UNINTENDED DRIFT`.

**Stop and report verbatim if:**
- a capture's replay fails to reproduce the census call;
- the reconstructed lineage fails to reproduce the recorded leaf;
- the scratch copy's hash changes;
- a reference fails to converge;
- anything else would force you outside these rules.

## Your final message

Your final message goes to the orchestrator, who will re-check your numbers. It holds:
- the classification table, per sampled call and in total;
- the counterfactual table;
- the cheap pass's totals;
- the one-paragraph answer: is the depth-20 chase the tiling mechanism, and does pinning stop it without dishonesty?
- how each check was made;
- every deviation;
- the record paths.
