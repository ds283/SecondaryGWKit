# Prompt 04 — Tighten `_solve_T_z`: fix the p90 (T2)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** `[02-qcd-T-z-spline-node-tolerance]` on the
[`GkTk-remedial` board](../GkTk-remedial/IMPLEMENTATION_STATE.md) §3
**Implements:** audit §8 recommendation **1** — *"One line, build-time cost only. Largest
accuracy-per-line in the audit."*
**Measurements:** audit §3 (T2), §4 · **Map:** `docs/qcd-background-audit/REFERENCE-FIXTURE.md`
**Depends on:** 01 (the harness scores it), 02 (the fixture regenerates), 03 (the key exists).
**Recommended model:** **Sonnet** — the production change is one argument list. The work is the
re-scoring, and prompt 02's map tells you exactly what has to be re-scored.

**Files you may touch:** `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` (`_solve_T_z`'s
`root_scalar` call and its comment, and the version constant), `ComputeTargets/tests/wkb_reference_data.json`
(**via the prompt-02 generator only** — never by hand), the test modules prompt 02's map identifies
as scored against a QCD background number (**tolerances, thresholds and comments only**),
`CosmologyModels/tests/test_T_z_representation.py` (thresholds 1 and 2), plus this campaign's log
and board, the `GkTk-remedial` board entry, and `docs/OPEN_ISSUES.md`.

**Do not touch:** `_build_T_z_spline` (prompt 05), `integration_break_points` (prompt 07), the
`RESIDUAL_WKB_REGION_MARGIN` constant or anything else in `ComputeTargets/phase_residual.py`
(§2 item 3 says measure it, not change it), `QCD_EOS.py`, any other production file.

**Read first:** `LambdaCDM_GenericEOS.py:144-188` (`_solve_T_z`), audit §3's T2 paragraph,
`docs/qcd-background-audit/REFERENCE-FIXTURE.md`, `ComputeTargets/phase_residual.py:210-230` (the
comment that attributes a $\pm0.1$ scatter in $\omega^2/\omega_0^2$ to this issue), and the
`GkTk-remedial` board entry `[02-qcd-T-z-spline-node-tolerance]`.

---

## 1. What is wrong

`_solve_T_z` (`:179`) ends:

```python
root = root_scalar(T_equation, bracket=(bracket_lo, bracket_hi), xtol=1e-6, rtol=1e-4)
```

`rtol=1e-4` is four digits. `_build_T_z_spline` calls this 500 times, once per node, and **each
converges independently**, so adjacent nodes carry uncorrelated errors of up to **2.496e-05**. A
cubic through scattered nodes scatters, and its derivative scatters worse. Measured on the audit's
probe set, the shipped node solve is max 2.496e-05 / p90 1.246e-05 / median 1.222e-08, and that
scatter is the **p90** of the whole representation.

`xtol=1e-6` is worse than useless: it is an *absolute* tolerance on a dimensionful $T$ that spans
twenty decades over the tabulated range, so it binds at the cold end and is unreachable at the hot
end. The audit's reference uses `xtol=1e-300, rtol=1e-14` — a tolerance that is purely relative at
every temperature.

**The runtime cost of fixing this is zero.** The solve is build-time only, at 8.3 µs per node: 500
nodes is ~4 ms, once per cosmology object.

## 2. The change

1. **`root_scalar(T_equation, bracket=(bracket_lo, bracket_hi), xtol=1e-300, rtol=1e-14)`**, or an
   equally defensible pair. Justify both numbers in the log: `rtol` must stay above Brent's floor
   of roughly $4\varepsilon\approx8.9\times10^{-16}$, and `xtol` must be small enough never to bind
   at any temperature on the tabulated range. If SciPy rejects `xtol=1e-300` on this version, say
   what it accepted and why.

   Replace the surrounding comment with one that says what the tolerances are *for* — the node
   values of an interpolant, built once, whose scatter between neighbours is the error that matters
   — and that the cost is build-time.

2. **Bump `T_Z_REPRESENTATION_VERSION` to 2**, and add this prompt's row to the constant's table
   (prompt 03 §2 item 1).

3. **Measure, and do not change, two things this defect was blamed for.**

   a. **`[02-qcd-T-z-spline-node-tolerance]`'s own claim** — the node values are up to 2.1e-5
      relative from a tight re-solve, ~4e-5 in $H$. Re-measure both before and after.

   b. **`ComputeTargets/phase_residual.py:220`'s $\pm0.1$ scatter.** That comment says the sign of
      $\omega^2$ near the turning point is not resolved by the background — *"at k = 3e8 the ratio
      `omega^2/omega_0^2` scatters by ±0.1 between neighbouring nodes around z ~ 4e15
      (`[02-qcd-T-z-spline-node-tolerance]`)"* — and `RESIDUAL_WKB_REGION_MARGIN = 0.5` exists to
      survive it. **Re-measure that scatter with the tightened nodes and report the number.**
      Do **not** change the margin, the comment, or anything else in that file: the margin is an
      unkeyed configuration axis (`[20-wkb-gauss-orders-not-in-lookup-key]`) and changing it is
      that issue's owner's decision. If the scatter is gone, that is a **narrowing**, recorded as
      an observation and a note on the board — it is the single most useful thing this prompt can
      hand forward.

4. **Regenerate the QCD reference fixture** with prompt 02's generator, in this commit, and quote
   the largest relative move per key. **Do not edit the JSON by hand.**

5. **Re-score, do not loosen.** Every test prompt 02's map lists as scored against a QCD background
   number will move. For each: quote the old tolerance, the new achieved value, and the new
   tolerance. A tolerance may only *tighten* or stay; a tolerance that must **loosen** is a
   finding — record it, say which number got worse and why, and the Result is
   `COMPLETE WITH DEVIATIONS`. If more than one has to loosen, **stop and ask**.

## 3. Acceptance

| Quantity | Before | Target |
|---|---|---|
| `_solve_T_z` node error, max on the probe set | 2.496e-05 | **≤ 1e-14** |
| `T(z)` representation error, **p90** | 1.323e-05 | **≤ 2.0e-07** (audit §4 measures 1.936e-07) |
| `T(z)` representation error, max | 7.177e-04 | ~7.26e-04 — **unchanged, and that is correct**: the max is pinned at the jump height until prompt 06 |
| `T(z)` representation error, median | 1.890e-07 | ~1.07e-07 |
| $\int\mathrm{d}z/H$ relative error (prompt 01's T1 guard) | 3.461e-08 | **recorded**; it is *not* required to improve much — T1 is prompt 06's |
| `_build_T_z_spline` wall time | ~1 ms | **≤ 100 ms**, quoted |
| `T_photon` cost per call | 2.26–2.44 µs | **unchanged** — this prompt does not touch the evaluation path |

Tighten prompt 01's test thresholds 1 and 2 to the achieved values (README §6.1's "After 04"
column), and **show they fail on `HEAD~1`**:

```bash
git stash list; git show HEAD --stat
git checkout HEAD~1 -- CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py
PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t . 2>&1 | tail -5
git checkout HEAD -- CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py
```

The middle command **must fail**, naming the node-solve test. Confirm `git status` is clean
afterwards. This is the check that matters most: a threshold that passes both before and after has
measured nothing (README §0.2).

All three suites pass; counts quoted and not falling. `black` clean.

## 4. Log and commit

Log to `logs/04-tighten-node-solve.md` per README §5.1. Move
`[02-qcd-T-z-spline-node-tolerance]` from the `GkTk-remedial` board's §3 to that board's §4 with
your measured before/after, update `docs/OPEN_ISSUES.md` (the row leaves §1.7 and the count falls
by one), and update this campaign's board §1 version table and §2 row T2.

**"State handed to the next prompt"** must carry: the tolerance pair chosen and why; the version
constant's new value; the largest reference move per key; the re-measured $\omega^2/\omega_0^2$
scatter with and without the fix; and the complete list of tests whose tolerances moved.

Commit subject, or something equally specific:
`Solve the T(z) spline nodes to a useful tolerance`
