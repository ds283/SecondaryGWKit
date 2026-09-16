# Log 05 — Hoist the range logic, and settle the `T_photon` cost row

**Prompt:** prompts/background-solver-robustness/05-hoist-the-range-logic.md
**Commit:** *(SHA not embedded, per the campaign convention)* — *"Hoist the range logic out of the
spline wrappers' hot path"*
**Model:** Opus (claude-opus-5)
**Date:** 2026-09-16
**Result:** COMPLETE WITH DEVIATIONS

---

## What shipped

Three classes, one shape. In each, the four `_outward` calls that `__call__` evaluated on every
call are computed once in `__init__` and stored; `__call__` reads the attributes.

| Site | Before | After |
|---|---|---|
| `ComputeTargets/spline_wrappers.py` — `ZSplineWrapper.__init__` | ended at `:50` with `_min_log_z` / `_max_log_z` | + the comment and the four hoisted bounds at **`:52-59`** |
| `spline_wrappers.py:64`, `:74` → **`:73`**, **`:83`** — `ZSplineWrapper.__call__`'s two comparisons | `if log_z > _outward(self._max_log_z, +1)` / `if log_z < _outward(self._min_log_z, -1)` | `> self._reject_above_log_z` / `< self._reject_below_log_z` |
| `spline_wrappers.py:66`, `:76` → **`:75`**, **`:85`** — the two f-strings | `{_outward(self._max_z, -1):.5g}` / `{_outward(self._min_z, +1):.5g}` | `{self._recommended_max_z:.5g}` / `{self._recommended_min_z:.5g}` |
| `spline_wrappers.py` — `GkWKBSplineWrapper.__init__` | ended at `:112` | + the same block at **`:123-130`** |
| `spline_wrappers.py:123`, `:133` → **`:141`**, **`:151`**, and the f-strings at **`:143`**, **`:153`** | as above | as above |
| `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` — `TemperatureRepresentation.__init__` | ended at `:305` | + the same block at **`:307-314`** |
| `LambdaCDM_GenericEOS.py:324`, `:337` → **`:333`**, **`:346`**, and the f-strings at **`:335`**, **`:348`** | as above | as above |
| `docs/qcd-background-audit/measure_T_z_representation.py:401-408` → **`:401-417`** | `inside = knots[(knots > u_nodes.min()) & (knots < u_nodes.max())]`, printed as "Of the BREAK_POINT_ALL points, 2411 are knots …" | a comment, `all_points` from `integration_break_points(..., kind=BREAK_POINT_ALL)`, `shared = np.intersect1d(all_points, knots)`, printed as "0 of those 3 BREAK_POINT_ALL points are also knots of the T(z) tabulation …" |

**The four new attributes, identically named in all three classes** (no new public symbol; all four
are private instance attributes set in `__init__` and never mutated):

| Attribute | Value | Used by |
|---|---|---|
| `_reject_above_log_z` | `_outward(self._max_log_z, +1)` | the upper **comparison**, in `log(1+z)` |
| `_reject_below_log_z` | `_outward(self._min_log_z, -1)` | the lower **comparison**, in `log(1+z)` |
| `_recommended_max_z` | `_outward(self._max_z, -1)` | the upper **message**, raw `z` |
| `_recommended_min_z` | `_outward(self._min_z, +1)` | the lower **message**, raw `z` |

The asymmetry prompt §1.1 names — comparisons on the **log** bounds, messages on the **raw** bounds
— is carried in the names, which is why there are four and not two. The one-line comment sits at
each `__init__` site only, citing `[07-t-photon-range-logic-recomputes-its-bounds]` and this
campaign; there is no new comment at any `__call__` site.

`_outward` itself, `SPLINE_BOUND_SLACK`, the `RuntimeError` text, the segment dispatch, the spline
order and node count, `_solve_T_z` and `_find_rho_equality` are untouched. The
`TemperatureRepresentation` docstring's "range logic" paragraph (`:260-270`) is unchanged: it says
`_outward` is imported rather than re-derived and that the message text is character-for-character
`ZSplineWrapper`'s, and both statements remain true.

**`T_Z_REPRESENTATION_VERSION` is 6 before and 6 after** (`LambdaCDM_GenericEOS.py:428`).

## The two equality redshifts

Not required by README §5.1 for workstream C, given because the campaign's central claim is that no
computed quantity moves and this prompt edits the hot path every $T(z)$ evaluation goes through.
Measured on both models at `max_z = 1e12`, through the `BaseCosmology` properties prompt 09 added,
at `82f4e8c` (before) and on the shipped tree (after):

| Model | Pair | Before | After | `float.hex()` | Relative move |
|---|---|---|---|---|---|
| `QCD_Cosmology` | matter = radiation | `3406.668974249951` | `3406.668974249951` | `0x1.a9d5683cafad0p+11` | **0** |
| `QCD_Cosmology` | matter = $\Lambda$ | `0.3034230329964074` | `0.3034230329964074` | `0x1.36b4870e4a718p-2` | **0** |
| pure-radiation stand-in | matter = radiation | `3403.1059638279457` | `3403.1059638279457` | `0x1.a963640e40f2cp+11` | **0** |
| pure-radiation stand-in | matter = $\Lambda$ | `0.3034230329964074` | `0.3034230329964074` | `0x1.36b4870e4a718p-2` | **0** |

**No bit moved.** The four hex strings are identical at both trees, and both banner lines are
character-identical (`matter-radiation equality at z = 3407`, `matter-Lambda equality at
z = 0.3034` on `QCD_Cosmology`; `3403` and `0.3034` on the stand-in). These are board standing
note 8's values, unchanged.

## Deviations from the prompt

### D1 — the `__init__` comment is four lines, not one — IMPLEMENTATION CHOICE

§2 item 1 says "a one-line comment at the `__init__` site … citing
`[07-t-photon-range-logic-recomputes-its-bounds]` and this campaign". The shipped comment is four
lines. The citation alone is 95 characters; adding to it the two things a later reader needs — that
the four are loop invariants of a hot path, and *why* hoisting is numerically null (the bounds are
never mutated after construction) — does not fit on one line at this project's ~95-column comment
width without dropping one of them.

**Alternatives considered.** (i) One line carrying only the issue tag, with the reasoning left to
the issue: rejected, because the issue closes in this same commit and a reader following the tag
lands on a §4 entry rather than on a live issue. (ii) Two lines, dropping "never mutated after
construction": rejected, because that clause is the whole argument for the hoist being safe, and
the next person to add a setter for `_max_z` is exactly who needs to read it. (iii) Put the
reasoning in each class's docstring instead: rejected, because `ZSplineWrapper` and
`GkWKBSplineWrapper` have no class docstring and adding two would be a larger edit than the one
declined. The prompt's stated intent — "not at the `__call__` site, where it would be noise" — is
about placement, and placement is as the prompt asked.

The identical four lines appear in all three classes, matching the existing convention in this file
set that the three classes share one expression of the range logic.

### D2 — §5's replacement prose also drops the stale "500-point interpolant" — IMPLEMENTATION CHOICE

§2 item 3 asks for the count to be intersected with the declared set and the sentence worded so it
reads correctly at 0. The sentence that had to be rewritten around the corrected count also
described the tabulation as "a uniform lattice of an auxiliary 500-point interpolant". Since
`qcd-background-audit` prompts 05 and 06 the tabulation is a segmented entropy factor at 3,000
nodes of order 5, so keeping that clause would have meant shipping a freshly-written sentence
containing a figure known to be wrong.

**Alternatives considered.** (i) Carry the clause across verbatim: rejected for the reason above.
(ii) Correct it to "3,000-point": rejected as scope creep — the sentence does not need a node count
to make its point, and quoting one commits the script to tracking `DEFAULT_T_Z_SPLINE_SAMPLES`.
(iii) What shipped: drop the node count entirely and speak about *a representation that forces its
own interpolation lattice into the break-point set* versus one that does not, which is the
distinction the sentence was always making and is true at any node count. The §5 table above it is
byte-identical, as §5 of the prompt requires.

### D3 — the §3 protocol was run twice; the first run is void — STRUCTURALLY REQUIRED

§3 item 1 requires a quiet machine and §3 item 4 makes a moving control row a void run. The first
ten-run pass was taken at load average ~4.5–5.3 and **is void by that rule**: the
entropy-factor control moved +2.0 % and the segmented control +0.8 % between the two trees, which
is the same order as the difference being measured, and the shipped row's own spread was 2.509–2.828
(before) and 2.389–2.894 (after). It is reported in "Verification performed" §3.0 and **no figure
from it is used anywhere**. The second pass, after a five-minute settle at load average ~3.0–3.9,
has all three controls inside ±2 % against per-tree spreads of 4–10 %, and is the measurement §4's
rule is applied to. Nothing about the code differed between the two passes.

## Verification performed

### 1. Bit-identity of every returned value — **ran it**

`probe_range_logic.py` (scratchpad; not committed — it needs no repository fixture and its result
is the two hashes below) builds eight wrappers spanning the three classes and exercises each over a
probe set covering the **full tabulated range** plus the slack band and points outside it:

- `ZSplineWrapper` × 4 — `(min_z, max_z, log_z, deriv)` = `(-0.24, 1e4, True, False)`,
  `(-0.24, 1e4, True, True)`, `(0.1, 1e4, True, False)`, `(0.1, 1e4, False, False)`. The negative
  `min_z` is `test_spline_wrappers.py`'s discriminating case.
- `GkWKBSplineWrapper` × 2 — `(-0.24, 1e4)` and `(0.1, 1e6)`, over a real `phase_spline`.
- `TemperatureRepresentation` × 2 — the production `QCD_Cosmology` and the pure-radiation stand-in,
  both at `max_z = 1e12`, over their own `[_min_z, _max_z]`.

251 abscissae per wrapper (241 uniform in $u$ across the declared range, plus both bounds, both
slack thresholds, the midpoints of both slack bands, the two `nextafter` neighbours of the
thresholds, and two points well outside), each entered through **both** entry points
(`z_is_log=True` and `z_is_log=False`). Every result recorded as `float.hex()`.

```
values=3979 raises=37
MD5 (probe_before.txt) = 7c16ba495fe0ffa72e1262406fbe7bb2
MD5 (probe_after.txt)  = 7c16ba495fe0ffa72e1262406fbe7bb2
```

**3,979 values, bit-identical by `float.hex()`; 37 in-probe rejections, identical in position and
in text; `diff` empty.** Re-taken against the final tree after the `black` pass: still identical.

Per-wrapper counts, so a later reader can see none was skipped:

| Wrapper | values | raises |
|---|---|---|
| `ZSplineWrapper[0]` (`min_z=-0.24`) | 497 | 5 |
| `ZSplineWrapper[1]` (`deriv=True`) | 497 | 5 |
| `ZSplineWrapper[2]` (`min_z=0.1`) | 498 | 4 |
| `ZSplineWrapper[3]` (`log_z=False`) | 498 | 4 |
| `GkWKBSplineWrapper[0]` | 497 | 5 |
| `GkWKBSplineWrapper[1]` | 498 | 4 |
| `TemperatureRepresentation[QCD]` | 497 | 5 |
| `TemperatureRepresentation[stand-in]` | 497 | 5 |

### 2. Character-identity of all six messages — **ran it**

Raised from each of the three classes at both ends and diffed the strings, as §2 item 2 requires
("verify by raising from each class and diffing the strings, not by reading"). Identical at both
trees, inside the same `diff` as §1:

```
ZSplineWrapper low:  ZSplineWrapper: evaluated probe-Z out of bounds @ z=-0.24492 (min allowed z=-0.24, recommended limit is z >= -0.2376)
ZSplineWrapper high: ZSplineWrapper: evaluated probe-Z out of bounds @ z=12035 (max allowed z=10000, recommended limit is z <= 9900)
GkWKBSplineWrapper low:  GkWKBSplineWrapper: evaluated probe-Gk out of bounds @ z=-0.24492 (min allowed z=-0.24, recommended limit is z >= -0.2376)
GkWKBSplineWrapper high: GkWKBSplineWrapper: evaluated probe-Gk out of bounds @ z=12035 (max allowed z=10000, recommended limit is z <= 9900)
TemperatureRepresentation low:  TemperatureRepresentation: evaluated T(z) out of bounds @ z=-0.24492 (min allowed z=-0.24, recommended limit is z >= -0.2376)
TemperatureRepresentation high: TemperatureRepresentation: evaluated T(z) out of bounds @ z=1.8283e+12 (max allowed z=1.05e+12, recommended limit is z <= 1.0395e+12)
```

Each hoisted bound is formatted by the same `:.5g` in the same position, and the `type(self).__name__`
prefix `f023eb8` settled is untouched.

### 3. The cost measurement

**The instrument.** `docs/qcd-background-audit/measure_T_z_representation.py` §6 — the 200-point
probe set, best of 3 — unmodified except for the §5 prose, which is above §6 and does not enter the
timing. **The same file was in place at both trees**: only
`ComputeTargets/spline_wrappers.py` and `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` were
swapped between runs, so the comparison is against literally the same instrument, as §3 item 2
requires. Reproduce either tree with

```bash
PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py
```

#### 3.0 The void first pass (D3) — recorded, not used

Load average 4.50–5.27 (1-minute), 10 cores. Shipped row: before 2.509 / 2.828 / 2.533 / 2.603 /
2.669 (mean 2.628), after 2.389 / 2.516 / 2.894 / 2.392 / 2.553 (mean 2.549). **Controls:**
entropy-factor mean 2.141 → 2.184 (**+2.0 %**), segmented 2.277 → 2.328 (+2.2 %), accurate root
solve 8.469 → 8.578 (+1.3 %) — all moving in a direction the change cannot cause, by the same order
as the difference under test. **Void by §3 item 4.** No figure from this pass is used.

#### 3.1 The measurement — five runs at each tree, alternating

Machine: 10 cores, load average 2.83–3.88 (1-minute) across the ten runs after a five-minute
settle, 5 s of slack between runs; no orphaned busy shells (board standing note 10 — checked with
`ps` before measuring, and there were none). Every run is a full script invocation; the four rows
below are the four rows §6 prints, in order.

| Run | tree | shipped | entropy 2000 | segmented | accurate root | load avg |
|---|---|---|---|---|---|---|
| 1 | `HEAD~1` | 2.532 | 2.097 | 2.233 | 8.154 | 3.88 |
| 1 | `HEAD` | 2.495 | 2.206 | 2.316 | 8.526 | 3.73 |
| 2 | `HEAD~1` | 2.856 | 2.263 | 2.384 | 8.717 | 3.37 |
| 2 | `HEAD` | 2.418 | 2.110 | 2.226 | 8.404 | 3.26 |
| 3 | `HEAD~1` | 2.684 | 2.211 | 2.474 | 8.860 | 3.28 |
| 3 | `HEAD` | 2.477 | 2.154 | 2.299 | 8.629 | 3.26 |
| 4 | `HEAD~1` | 2.622 | 2.081 | 2.291 | 8.139 | 2.99 |
| 4 | `HEAD` | 2.546 | 2.175 | 2.340 | 8.616 | 2.83 |
| 5 | `HEAD~1` | 2.678 | 2.206 | 2.324 | 8.658 | 3.15 |
| 5 | `HEAD` | 2.491 | 2.159 | 2.299 | 8.491 | 3.38 |

All figures µs/call.

| Row | `HEAD~1` mean | range | `HEAD` mean | range | ratio | change |
|---|---|---|---|---|---|---|
| **shipped spline** (`T_photon`) | **2.6744** | 2.532–2.856 | **2.4854** | 2.418–2.546 | **0.9293** | **−7.07 %** |
| entropy factor, 2000 pts *(control)* | 2.1716 | 2.081–2.263 | 2.1608 | 2.110–2.206 | 0.9950 | −0.50 % |
| segmented entropy factor *(control)* | 2.3412 | 2.233–2.474 | 2.2960 | 2.226–2.340 | 0.9807 | −1.93 % |
| accurate root solve *(control)* | 8.5056 | 8.139–8.860 | 8.5332 | 8.404–8.629 | 1.0032 | +0.32 % |

**The three controls hold** (§3 item 4, §5 row 4). None of the three goes through
`TemperatureRepresentation.__call__` — they are the script's own candidate objects and its
`brentq` reference — and they move by −0.50 %, −1.93 % and +0.32 %, against per-tree spreads of
8.4 %, 10.3 % and 8.5 % at `HEAD~1` and 4.4 %, 5.0 % and 2.6 % at `HEAD`. Every control move is
well inside the measurement's own spread, so the run stands.

**The ratio** (§3 item 5): **0.9293**, an absolute saving of **0.189 µs/call**. The issue predicted
0.11 µs (2 × 0.056 µs, the two `_outward` calls on the non-raising path). The measured saving is
larger because what is removed is not two arithmetic operations but two Python-level function calls
with a branch inside each, replaced by two attribute loads. The prediction was of the right sign and
the right order and was conservative.

**One honest caveat about the absolute number.** This machine reads high on this instrument: the
`HEAD~1` mean of **2.6744** is **+3.0 %** against the 2.596 µs that `[06-…]` records for the *same
code* on the machine that took the confirmed miss. The ratio is machine-independent and the controls
say the run is sound, but if this machine's +3 % offset is applied to the `HEAD` figure the
corresponding reading there would be ~2.41 µs. **Both readings clear 2.5 µs**, which is why the
decision below does not turn on resolving the offset.

#### 3.2 §4's decision rule, applied

$\bar t$ = **2.4854 µs** over five runs at `HEAD` on a machine whose controls are steady.
**$\bar t \le 2.5$ µs, so `[06-t-photon-call-cost-needs-a-quiet-machine]` closes.**

Stated without flattery: the margin is **0.6 %**, and one of the five runs (2.546) is above the
target on its own. The rule §4 states is on the mean over five runs and the mean clears it; the
range is quoted here so that the next person to measure knows it straddles. The row is moved to the
`qcd-background-audit` board's §4 with all five figures, both trees, and this caveat.

§4's second branch — escalate to README §7 **D4** — is **not taken**, and no second optimisation was
looked for (§6 stop condition 4).

### 4. The audit script's §5 — **ran it**

Against the unedited script at `82f4e8c`, the **only** difference in the script's entire output
above §6 is the four prose lines replaced by five:

```
< Of the BREAK_POINT_ALL points, 2411 are knots of the T(z) spline itself --
< a uniform lattice of an auxiliary 500-point interpolant, not a feature of the
< cosmology. A representation that does not need that lattice removes them from the
< break-point set, leaving only the genuine crossings of section 2.
---
> 0 of those 3 BREAK_POINT_ALL points are also knots of the T(z)
> tabulation. A representation that forces its own interpolation lattice into the
> break-point set contributes knots here, which are an artefact of how T(z) is
> approximated and not a feature of the cosmology; a representation that does not,
> contributes none, leaving only the genuine crossings of section 2.
```

The corrected count is **0**, which is what `qcd-background-audit` prompt 07 made it. §5's table
above it still prints `all 3 points in range … 215.34 x the grid spacing` and
`discontinuity 2 points in range … 430.69 x the grid spacing`, unchanged. The sentence now reads
correctly at 0 (the count leads: "0 of those 3 …") and would read correctly at a positive count.

### 5. Sections 0–5 are byte-identical across all ten runs and both trees — **ran it**

An accidental but strong confirmation of numerical nullity, and worth recording: §0's accuracy
table, §1's branch joins, §2's discontinuity probe, §3's candidate comparison, §4's $H(z)$ and
$\int\mathrm{d}z/H$ figures and §5 all go through `TemperatureRepresentation.__call__`. Diffed with
`run_before_1.txt` as the reference, **all nine other runs — five at `HEAD~1`, five at `HEAD` —
match it byte for byte down to the `6. COST PER CALL` banner.** In particular §0's
`shipped T(z) spline … max 6.807e-11 p90 3.237e-15 median 1.765e-16` and §4's
`integral dz/H … = 1.3320002507788795e+03` are the same digits at both trees.

### 6. Suites — **ran them**

| Suite | `HEAD~1` | `HEAD` |
|---|---|---|
| `CosmologyModels/tests` | **39**, OK, 0.677 s | **39**, OK, 0.668 s |
| `ComputeTargets/tests` | **449**, OK, 160.3 s | **449**, OK, 158.6 s |

Both run with `PYTHONPATH=. ./venv/bin/python -m unittest discover -s <pkg>/tests -t .` from the
repository root. `ComputeTargets/tests/test_spline_wrappers.py` — the guard on `_outward` and on
the bounds behaviour — is inside the 449 and passes at both trees unchanged. No test was added:
this prompt claims no behaviour change, and §5's acceptance is bit-identity against `HEAD~1`, which
is a property of a diff and not of a tree. There is therefore no "fails on `HEAD~1`" demonstration
to give, and README §2 (e) does not apply to a prompt that asserts nothing new.

### 7. `black` — **ran it**

`./venv/bin/python -m black ComputeTargets/spline_wrappers.py
CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py
docs/qcd-background-audit/measure_T_z_representation.py` → *"3 files left unchanged"*. All three
files in the diff are clean under `--check`.

### 8. `T_Z_REPRESENTATION_VERSION` — **read it**

`LambdaCDM_GenericEOS.py:428`: `T_Z_REPRESENTATION_VERSION: int = 6`, before and after. Untouched
by this prompt, as board standing note 2 requires.

## Observations not acted on

1. **`black --check .` is not clean at the repository root, and was not before this prompt
   either.** 54 files would be reformatted, every one of them under `docs/` in a per-review or
   per-benchmark scratch directory (`docs/gk-wkb-review-fable-2026-09-09/`,
   `docs/adaptive-levin-benchmark/levin_bench/`, and others). None is a production file, none is in
   this prompt's permitted list, and `docs/qcd-background-audit/measure_T_z_representation.py` —
   the one `docs/` file this prompt touches — is **not** among them. README §5 rule 7's "the tree
   is clean under `--check`" is therefore already false of the whole tree and true of everything
   any campaign has touched. Recorded rather than fixed: reformatting 54 files nobody asked about
   would swamp this diff. Opened as a §3 issue on this campaign's board.

2. **The hoist's saving is 1.7× the figure `[07-…]` predicted**, 0.189 µs against 0.11 µs. Not a
   problem — it is the issue's own cheap-saving argument coming out better than costed — but a
   later reader comparing the two numbers should know the difference is a Python function call
   (two per evaluation), not two floating-point operations, and that `[07-…]`'s 0.056 µs each was
   measured as the cost of `_outward` rather than as the difference the hoist makes.

3. **§5's production source grid is the 1,732-sample bare lattice**, not the 1,996-sample
   production grid of board standing note 17: the script builds its grid from
   `ComputeTargets/tests/wkb_reference.production_source_grid`, which takes no `feature_z`. Nothing
   in §5's argument depends on which of the two it is — it is comparing a break-point spacing
   against a median grid spacing — and prompt 09 moved neither. Recorded so that a reader holding
   note 17 beside the script's output does not think one of them is wrong.

4. **Nothing else in the tree hoists these bounds.** `_min_log_z` / `_max_log_z` / `_outward` are
   read only inside the three `__call__`s changed here and inside `test_spline_wrappers.py`; a
   repository-wide grep is in the commit's verification. No other class reproduces the range logic.

## State handed to the next prompt

Prompt 06 is provenance and close-out and may not touch production code. What it needs from here:

1. **No production number moved.** 3,979 `float.hex()` values across the three classes, 37 in-probe
   rejections, six `RuntimeError` messages, and the whole of `measure_T_z_representation.py`'s
   output above §6 are byte-identical at `HEAD~1` and `HEAD`. The two equality redshifts are board
   standing note 8's values, unmoved: `QCD_Cosmology` `3406.668974249951` /
   `0.3034230329964074`, stand-in `3403.1059638279457` / `0.3034230329964074`.

2. **The four hoisted attributes** are `_reject_above_log_z`, `_reject_below_log_z`,
   `_recommended_max_z`, `_recommended_min_z`, identically named and identically derived in
   `ZSplineWrapper`, `GkWKBSplineWrapper` and `TemperatureRepresentation`, set in `__init__` and
   never mutated. A prompt adding a setter for `_min_z` or `_max_z` on any of the three must update
   all four or the range logic goes stale silently — there is no guard against that beyond the
   comment.

3. **`[06-t-photon-call-cost-needs-a-quiet-machine]` is closed at 2.4854 µs**, mean of five runs at
   `HEAD` (range 2.418–2.546, ratio 0.9293 against `HEAD~1`'s 2.6744, controls within ±2 %).
   README §7 **D4** is **not** invoked and needs no answer from the user. The margin is 0.6 % and
   one run of five was above target; if a later campaign changes the representation, the row is
   worth re-measuring rather than assumed.

4. **This machine reads +3.0 % high** on `measure_T_z_representation.py` §6 against the machine
   that took `[06-…]`'s 2.596 µs figure, measured on identical code. A later prompt comparing an
   absolute µs figure to either number must use the ratio, or re-take both ends itself.

5. **The reproduction commands.**
   - Cost, both trees: `PYTHONPATH=. ./venv/bin/python docs/qcd-background-audit/measure_T_z_representation.py`
     (§6's four rows; ~3 s per run, no Ray, no datastore). Five runs at each tree, alternating,
     after a settle; a control row moving more than the spread voids the run.
   - Suites: `PYTHONPATH=. ./venv/bin/python -m unittest discover -s CosmologyModels/tests -t .`
     (39) and `… -s ComputeTargets/tests -t .` (449).
   - Bit-identity: the scratchpad probe described in §1 above. It is not committed; it is eight
     wrapper constructions and a `float.hex()` dump, and re-deriving it is cheaper than carrying a
     file whose only output is a hash.

6. **`T_Z_REPRESENTATION_VERSION` is 6**; suites `CosmologyModels` **39**, `ComputeTargets` **449**,
   both OK at both trees. Workstream C's remaining prompt is 06.
