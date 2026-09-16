# Prompt 05 — Hoist the range logic, and settle the `T_photon` cost row

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Establishes:** README §2 **(h)** · **Reports on:** README §7 **D4**
**Closes:** `[07-t-photon-range-logic-recomputes-its-bounds]` and
`[09-audit-script-section-5-prose-counts-the-wrong-set]` on the
[`qcd-background-audit` board](../qcd-background-audit/IMPLEMENTATION_STATE.md) §3.
**Closes or escalates:** `[06-t-photon-call-cost-needs-a-quiet-machine]` — see §4.
**Depends on:** 04 (same file; sequencing only)
**Recommended model:** **Opus** — the code change is four expressions moved into two `__init__`s.
The work is a microsecond-scale timing protocol with a stop-or-escalate rule, on a hot path shared
by three classes, with an acceptance of **bit-identity** that has to be demonstrated rather than
argued.

**Files you may touch:** `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py`
(`TemperatureRepresentation.__init__` and `__call__`, `:274-355`),
`ComputeTargets/spline_wrappers.py` (`ZSplineWrapper` and `GkWKBSplineWrapper`, `:55-140`),
`docs/qcd-background-audit/measure_T_z_representation.py` (**§5's prose only**, `:399-408`), plus
this campaign's log, both boards and `docs/OPEN_ISSUES.md`.

**Do not touch:** `_outward` itself (`spline_wrappers.py:14`) — its semantics are pinned by
`ComputeTargets/tests/test_spline_wrappers.py`; the `RuntimeError` message text, which `f023eb8`
settled three weeks of argument about; the segment dispatch; `SPLINE_BOUND_SLACK`; the spline
order or node count; `_solve_T_z`; `_find_rho_equality`.

**Read first:** `ComputeTargets/spline_wrappers.py:1-140` in full;
`LambdaCDM_GenericEOS.py:235-355` (`TemperatureRepresentation`, its docstring's "range logic"
paragraph at `:261-269`, and `__call__`);
`docs/qcd-background-audit/measure_T_z_representation.py:395-435` (§5's prose and §6's cost loop);
and the index rows for `[06-…]`, `[07-…]` and `[09-…]` in `docs/OPEN_ISSUES.md` §1.7.

---

## 1. What is wrong

### 1.1 The range logic recomputes its own bounds on every call

`TemperatureRepresentation.__call__` (`:317`) and `ZSplineWrapper.__call__`
(`spline_wrappers.py:~60`) and `GkWKBSplineWrapper.__call__` (`~:120`) each evaluate

```python
if log_z > _outward(self._max_log_z, +1):   ...
if log_z < _outward(self._min_log_z, -1):   ...
```

on **every** call, and `_outward(self._max_z, -1)` / `_outward(self._min_z, +1)` again inside the
two f-strings. All four bounds are fixed at construction. Measured by `qcd-background-audit`:
**0.056 µs each**, of a ~2.5 µs call. Hoisting them into `__init__` is **numerically null** — the
same `_outward` of the same bound, computed once instead of per call.

Note the asymmetry the code already has: the *comparisons* use the **log** bounds and the *messages*
use the **raw** bounds. Four hoisted values per class, not two. Name them so a reader sees which is
which.

### 1.2 The cost row is a confirmed miss

`qcd-background-audit` README §6.2 set ≤ 2.5 µs per `T_photon` call. Measured on a quiet machine
after prompt 06, restated by prompt 09: **2.596 µs mean, range 2.505–2.671 over five runs** —
about **3.8 % over**. All of the excess is the order-5 `BSpline.__call__`, and order 5 is not
optional (a cubic needs ~25,000 nodes for the required p90). The index row's own next step is:

> hoist the two loop-invariant `_outward` calls (`[07-…]`, a measured 0.11 µs, numerically null),
> which lands at ~2.49 µs; if that does not clear it, the row itself is what to put to the user.

### 1.3 The audit script's §5 prose counts the wrong set

`docs/qcd-background-audit/measure_T_z_representation.py:401-404` prints

> Of the BREAK_POINT_ALL points, 2411 are knots of the T(z) spline itself

but computes `inside = knots[(knots > u_nodes.min()) & (knots < u_nodes.max())]` — the
tabulation's knots **inside the production range**, never the intersection with the declared set,
which since prompt 07 is **0**. The table above it (3 / 2) is correct; only the sentence lies.
It is in scope here because §3 re-runs this script for its §6 cost row.

## 2. The change

1. **Hoist, in all three classes.** Compute the four bounds in `__init__`, store them under names
   that say what they are (the two comparison thresholds and the two message bounds), and have
   `__call__` read them. Add a one-line comment at the `__init__` site saying they are loop
   invariants of a hot path, citing `[07-t-photon-range-logic-recomputes-its-bounds]` and this
   campaign — not at the `__call__` site, where it would be noise.

2. **Keep the message text character-identical.** `f023eb8` made the prefix `type(self).__name__`
   so the three classes share one expression while each names itself; the rest of the string is
   unchanged since before that. A hoisted bound must be formatted by the same `:.5g` and appear in
   the same position. **Verify by raising from each class and diffing the strings**, not by reading.

3. **Fix §1.3's prose.** Intersect the knots with the **declared** break points before printing,
   and word the sentence so it reads correctly at 0 as well as at a positive count. The table above
   it does not change. **Run the script and quote the new line.**

## 3. The measurement protocol

This is the substance of the prompt. Follow it literally.

1. **A quiet machine.** Close everything else. The figure this replaces was taken on a quiet
   machine and a comparison against a loaded one is worthless. If you cannot get a quiet machine,
   **say so and stop** — an unquiet number is worse than no number.
2. **Use `docs/qcd-background-audit/measure_T_z_representation.py` §6**, which already times
   `T_photon` over a 200-point probe set taking the best of 3. Do not write a new harness; the
   comparison must be against the same instrument.
3. **Five runs at `HEAD~1`, five at `HEAD`**, alternating, all four rows of the §6 table each time.
   Report **mean and range** for the shipped-representation row at each tree, and the difference.
4. **Report the three controls too** — the entropy-factor, segmented and accurate-root rows. If a
   row that this change cannot touch moves by more than the measurement's own spread, the machine
   was not quiet and the run is void.
5. **Also give the ratio**, not only the absolute: `HEAD` / `HEAD~1` for the shipped row. The
   absolute target is a machine-dependent number, the ratio is not, and the ratio is what shows
   the hoist did what it claims.

## 4. The decision rule — read this before you measure

Let $\bar t$ be the mean over five runs at `HEAD`.

- **$\bar t \le 2.5$ µs** → `[06-t-photon-call-cost-needs-a-quiet-machine]` **closes**. Move its
  row to the `qcd-background-audit` board's §4, delete it from `docs/OPEN_ISSUES.md`, and quote
  the five figures in both the log and the board entry.
- **$\bar t > 2.5$ µs** → **escalate. Do not go looking for another optimisation.** Write the
  row up for the user as README §7 **D4** requires: the measured mean and range at both trees, the
  ratio, the share attributable to the order-5 `BSpline.__call__`, and the statement that order 5
  is not optional. The two honest options are that the 2.5 µs target was set ~4 % too tight or that
  the cost is accepted; name both and recommend one. `[06-…]` is **narrowed, not closed**, and its
  index row is updated with the new figure and a next step of "the user's decision".

**In both cases `[07-t-photon-range-logic-recomputes-its-bounds]` closes** — it is about the
recomputation, not about the target, and the hoist discharges it regardless of what the clock says.

## 5. Acceptance

| Check | Threshold |
|---|---|
| Every value the three classes return over a probe set spanning the full tabulated range of each | **bit-identical** to `HEAD~1`, demonstrated by `float.hex()` comparison, count quoted |
| Both `RuntimeError` messages from each of the three classes | **character-identical** to `HEAD~1`, demonstrated by diff |
| `T_photon` cost, five runs each tree | mean, range and ratio quoted; §4's rule applied |
| The three control rows of §6 | move by less than the measurement spread |
| `measure_T_z_representation.py` §5 | prints the intersection (**0**), new line quoted; §2–§4 and §6 output otherwise unchanged |
| `CosmologyModels` suite | unchanged count, OK |
| `ComputeTargets` suite | **447 → 447**, OK (`test_spline_wrappers.py` is the guard on `_outward`) |
| `T_Z_REPRESENTATION_VERSION` | **6** before and after |
| `black --check` | clean |

## 6. Stop conditions

- **Any returned value differs by a bit.** The change is numerically null by construction; a
  difference means a bound was hoisted from the wrong expression. Stop.
- **Any message text differs by a character.** Same.
- **A control row moved.** The machine was not quiet. Re-run; if it persists, stop and say so.
- **You want to optimise something else** to clear 2.5 µs — inline `bisect_right`, cache on $z$,
  lower the order. **You may not.** §4 says escalate, and README §7 D4 says the same. Every one of
  those is a change to a representation two campaigns settled.

## 7. Deliverables

1. The hoist, in all three classes, commented once per `__init__`.
2. The corrected §5 prose in `measure_T_z_representation.py`.
3. `logs/05-hoist-the-range-logic.md` per README §5.1 — with §3's full ten-run table, the
   bit-identity and message-identity demonstrations, and §4's decision applied and justified.
4. Board row 05, item row (h); `[07-…]` and `[09-…]` moved to the **`qcd-background-audit` board's
   §4** with their rows deleted from `docs/OPEN_ISSUES.md`; `[06-…]` closed or narrowed per §4 —
   count and date corrected, in the same commit.
5. One commit, README §5 rule 2.
