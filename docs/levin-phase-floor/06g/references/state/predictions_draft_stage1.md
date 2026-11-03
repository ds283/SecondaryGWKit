# levin-phase-floor 06g §5, the references — predictions and fixed readings (DRAFT, stage 1)

> **Draft, written in stage 1 (2026-11-01, before any capture, draw, reading or hash).** At stage 2,
> before any number is computed, it is copied into `state/predictions_<UTC>.md` (then made
> read-only and never edited) with the one item marked **[stage 2]** filled from the census agent's
> final harness, and the same text goes into `log.md`. Nothing in it may depend on a measured number.

## Prompt 06g §5, verbatim

- **A reference counts** when its four readings agree to within 1/100 of
  $\min({\rm abserr}_{06h}, {\rm abserr}_g)$. Those that do not are reported with their scatter and
  dropped. A reading over 30 minutes is stopped, reported, and the sample's reference dropped.

- **V3 — honest where referenced.** **Rule:** $|v_g - {\rm ref}| \le {\rm abserr}_g$ for every
  counted reference, Z3's and the random sample's. $|v_{06h} - {\rm ref}|/{\rm abserr}_{06h}$ is
  reported beside it. **Holds** if no counted reference violates it; fewer than 45 of the 90 counted is
  **undecided**.
- **V4 — the trusted leaf's value.** **Rule:** on Z3's samples 2, 8, 9 and 14,
  $|v_g - {\rm ref}|/{\rm abserr}_g \le 0.01$ (06e's breakdown, with the quarters: 0.0012, 0.00046,
  0.0015). **Holds** if true on all four.

**Descriptive, not scored:**
- **The correction $-\sum E$:** on every counted reference with a Clenshaw–Curtis leaf,
  $|v_g - \sum E - {\rm ref}|$ against $|v_g - {\rm ref}|$ and against ${\rm abserr}_g$.

Prompt 06g §7, the references' stop: fewer than half of the 77 new samples count, or a reading
disagrees with 06i's on one of its 13.

## The seed

seed: 20261101

Chosen in stage 1, before the 06h checkpoint was read for the draw; the draw (`scripts/draw_refs.py`)
refuses unless this line is in the predictions file it is given.

## The sample

- **Z3's 30 calls**, `docs/levin-phase-floor/06c/depth20-tiling/results.json`'s `calls`, ids
  `z00`–`z29` by `sample_no`. 06i's 13 are z01, z02, z03, z08, z09, z10, z13, z14, z15, z22, z23,
  z25, z27; the 17 others are z00, z04, z05, z06, z07, z11, z12, z16, z17, z18, z19, z20, z21, z24,
  z26, z28, z29. A Z3 call is matched in each census by the unit's `<rtol>-<shard>-<serial>` and the
  call `index`, with the same `notify_label`.
- **The 60 drawn**, `r00`–`r59`, by `scripts/draw_refs.py` (its docstring is the rule): from 06h's
  census (checkpoint `3104c18e…`), outside the 720 stationary-point rows (06a's scan; the row
  excluded at both rtols) and Z3's 30 calls; strata 1e-7 analytic, 1e-7 group, 1e-8 analytic, 1e-8
  group, 15 each; an analytic-block call is a unit's `other_calls` entry, a cosmological-group call a
  `calls` entry; **finite abserr in 06h's census record** (the population named is 06h's census);
  uniform without replacement, `random.Random(20261101).sample` over each stratum sorted by
  (shard, serial, index) as integers, one generator, the strata in that order.

## How each number is formed

All arithmetic is exact: census doubles are read from their hex strings; 30-digit reading strings
are parsed in `mpmath` at 50 digits; comparisons are made there.

- $v_g$, ${\rm abserr}_g$: the g census's checkpoint (the census agent's run
  `levin-phase-floor-06g-census-*`, read only), the unit's `done` record, the call's entry in `calls`
  (group) or `other_calls` (analytic) with the same index, label and span: `value.hex`,
  `abserr.hex`. (Every `done` record of the unit holding the call must agree.)
- $v_{06h}$, ${\rm abserr}_{06h}$: the same from 06h's census checkpoint (`3104c18e…`).
- $\sum E$: the g census record's per-call signed node-error sum, the driver's
  `node_error_signed_sum` as the census harness records it: **[stage 2: the field name and form in
  the census agent's final `worker.py`]**. A call **has a Clenshaw–Curtis leaf** iff its record's
  `leaves.cc` > 0 (06c's per-call summary, `_summarise_call_06c`).
- **ref**: for 06i's 13, 06i's reference, its FCC res-2 reading (`docs/levin-phase-floor/06i/out/
  scores.json`'s `ref_30`, equal to `readings/sNN-fcc-r2.json`'s `value_30`); for the 77, the FCC
  res-2 reading of `tools.filon_reference` at g (`out/readings/<sid>-fcc-r2.json`, `value_30`), the
  tool's own choice (`reference()`'s value).
- **The four readings** of a sample: QAWO and FCC, each at resolutions 1 and 2, by
  `tools.filon_reference.reading(call, impl, res, route=None, edges=None, workers=6,
  call_source=("tools.filon_reference", "_load_pickle", (PICKLE, "ray.cloudpickle")))`, exactly
  what the command line's `reference()` calls, one reading per process (`scripts/one_reading.py`) so
  that each can be stopped at 30 minutes and timed.

## The count rule, as applied

A sample's reference **counts** iff all hold:
1. its four readings exist (none stopped at 30 minutes, none failed);
2. their scatter, largest minus smallest of the four `value_30`, is at most
   $\min({\rm abserr}_{06h}, {\rm abserr}_g)/100$ (a non-finite ${\rm abserr}_g$ leaves
   ${\rm abserr}_{06h}$);
3. on Route B, every reading reports `tmap.all_monotone` true. The tool's docstring: a segment that
   contains a stationary point of the phase "is reported (`all_monotone` False) and the reference
   must not be used".

For **06i's 13** the readings are 06i's four (the references kept), scored by the same rule
against 06g's tolerance, because V3 counts "of the 90" under one rule; 06i counted them against
$\min({\rm abserr}_{06c}, {\rm abserr}_{06e})/100$, which is reported beside it. Kept only if the
group comparison (below) holds.

## The scores

- **V3:** over the counted references, a **violation** is $|v_g - {\rm ref}| > {\rm abserr}_g$.
  **Fails** if any counted reference violates it; otherwise **undecided** if fewer than 45 of the 90
  count; otherwise **holds**. Reported: the number counted; every violation with $v_g$, ref and
  ${\rm abserr}_g$ in `float.hex` (ref also at 30 digits); the worst $|v_g - {\rm ref}|/{\rm abserr}_g$;
  $|v_{06h} - {\rm ref}|/{\rm abserr}_{06h}$ beside each.
- **V4:** on z02, z08, z09 and z14, $|v_g - {\rm ref}|/{\rm abserr}_g$. **Holds** if all four count
  and each is at most 0.01; **fails** if any counted one exceeds 0.01; otherwise **undecided** (a
  sample whose reference does not count cannot be shown "true"). 06e's breakdown's numbers beside
  them: 0.0012 (z02 and z14, the same call at two rtols), 0.00046 (z08), 0.0015 (z09).
- **The stop:** fewer than 39 of the 77 new samples counting is a stop (§7), reported before V3 is
  scored.
- **The correction $-\sum E$ (descriptive):** on every counted reference whose g record has a
  Clenshaw–Curtis leaf: $|v_g - \sum E - {\rm ref}|$, $|v_g - {\rm ref}|$, their ratio, and each over
  ${\rm abserr}_g$.

## 06i's 13

**Which captures the tool reads:** the captures at g (`state/inputs/zNN.pkl`), once
`scripts/compare13.py` finds each group equal to 06i's input. Why:
- 06i read its inputs at 06e's tree `6d81fad`, in a worktree. No such worktree is on this host, and
  making one is a git write, which the brief forbids. Loading 06i's inputs at g would mix closures
  pickled at 06c's tree with g's classes (06i: the captured groups "predate 06h"): neither tree's
  integrand.
- The captures at g are what the 77 new references are read from, so the check exercises the tool
  exactly as it is used.
- `git diff --stat` from 06c's capture tree `e5764a6` through `6d81fad` and `c1ec3b2` to `1c5336e`,
  outside docs/ and prompts/, touches `AdaptiveLevin/`, `tools/`, `LiouvilleGreen/bessel_phase.py`
  (a docstring) and `LiouvilleGreen/three_bessel_integrals.py` (06h: `sin_cos`, `theta_abserr`,
  `K_lo`; not the residual, the amplitudes, `K`'s value or the coefficients), and tests. So where the
  groups are equal the references should be too.

**The group comparison** (`scripts/compare13.py`, its docstring): bit for bit, the route, the span,
and Route A's signs, coefficients and orders, or Route B's signs and each active factor's phase
spline and leading table (the criterion as written); and the integrand as the tool reads it (both:
$f_{\sin}$, $f_{\cos}$ at 65 points; A: `K_exact`, C, `u_breaks`, the residual at the 65 points; B:
`theta`, `theta_mod_2pi`, `theta_deriv` at the 65 points and `production_edges`). A difference in
either is a stop.

**The agreement rule** (06i's R1 rule, applied reading by reading): for each of the 13 and each of
its four readings $k$,
$$D_k = |{\rm tool}_k - {\rm 06i}_k| \le s_{\rm tool} + s_{06i} + m/100,$$
with $s$ each set's four-reading scatter and $m$ 06i's $\min({\rm abserr}_{06c}, {\rm abserr}_{06e})$
(06i's `min_abserr`). A sample with any $k$ outside it disagrees: **a stop**. Reported beside it,
not used: bit-identity with 06i's readings, $D_k$ over the allowance and over $m$, the structure
(panels, segments, tables, $K$, $C$, breaks).

**Sample 8,** as 06i read it on the user's ruling (*"I accept sample 8."*): 06i's §8 statistic,
$|{\rm FCC}_{r2} - {\rm QAWO}_{r2}|$ over the larger implementation's own res-1/res-2 scatter, was
5.1 there, and the reference was accepted because its four readings agree within the counting rule;
the statistic is neither a counting nor an agreement criterion. So it is computed and reported for
all 13, tool and 06i, and not used; z08 is treated like the other twelve.

**The command line:** on z25 (Route A) and z13 (Route B) the tool's own command line,
`python -m tools.filon_reference PICKLE --tolerance T --unpickler ray.cloudpickle --workers 6 --out
JSON`, with T = min(abserr_06h, abserr_g)/100; each of its four readings must equal
`one_reading.py`'s bit for bit. A difference is a stop.

## Other stops (each ends the run before any reference reading)

- a capture that does not reproduce its g census record (value, abserr, label, converged,
  num_regions, max_depth, the region arrays), or that ran a harness other than the census's;
- one of 06i's inputs not equal to 06i's SHA-256 list;
- a capture whose Route B default edges cannot be found (`production_edges` raises);
- any of the 52 readings on 06i's 13 failing or exceeding 30 minutes.
