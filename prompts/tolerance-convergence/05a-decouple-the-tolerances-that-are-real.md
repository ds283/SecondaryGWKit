# Prompt 05a — decouple the tolerances that are real

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** board items **T8** and **T10**, and with them
`[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]` and part (ii) of
`[02a-grid-digest-not-reproducible]`
**Depends on:** prompt 05 (`90d0114`) and prompt 05b (`a351a50`). **D1 closed 2026-09-17**, so every
number this prompt ships is already settled and none of them is yours to choose.
**Recommended model:** **Opus** — the numbers are given; the judgement is in the plumbing. What
makes this prompt succeed or fail is whether *every* lookup of a retuned target moves with it, in
`main.py` and in six reader scripts, and whether the guard that is supposed to catch a missed one
can actually see all of them.

**This is the tolerance half of what §3.5 used to be** (README §7 **D9**, 2026-09-18), and the
second of the two halves to run (**D10** put 05b between them). **§5 rule 8 names this prompt as
the one prompt in the campaign that may change a parameter.** Every other prompt reads the
constants and measures; this one moves them.

**It changes no schema.** No column is added, renamed, retyped or dropped. The six factories' column
lists must read exactly as they do at `a351a50`, `break_point_kind` and the four Gauss orders
included. A retuned tolerance changes which `tolerance` row a key points at, not what the key is
made of.

**Files you may create or touch:**
`config/defaults.py` — the constants of §2, and **nothing else in it**;
`main.py` — the `ray.get` tolerance block, every `object_get` of a retuned target, and the dead
`QuadSource` pair of §4;
`CosmologyConcepts/wavenumber.py` — `_solve_horizon_exit`'s tolerance defaults and the paths that
reach them, **and no change to the root-solve algorithm, the bracketing or `DEFAULT_HEXIT_TOLERANCE`**
(which is a bracket width, not a tolerance, despite the name);
the six `extract_*.py` scripts of §5;
`ComputeTargets/tests/test_main_plumbing.py` — the guard of §4;
`ComputeTargets/GkNumericIntegration.py`, `TkNumericIntegration.py` — **comments and docstrings
only**, where they name the shared constant and would now be wrong;
plus this campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** the schema of any table, or any factory under `Datastore/SQL/ObjectFactories/`;
the four Gauss orders, `ComputeTargets/phase_residual.py`, `Quadrature/integrators/WKB_phase_function.py`,
`ComputeTargets/BackgroundModel.py`, `GkWKBIntegration.py`, `TkWKBIntegration.py`, `GkSource.py`
and `ComputeTargets/tests/test_gauss_order_key.py` — prompts 05 and 05b own all of that and none of
those four targets carries a tolerance any more;
**the seven float-comparison sites of §6** — `GkSource.py:96`, `:104`, `:275`,
`numeric_with_phase_cut.py:618`, `:737`, `:790`, `WKBtools.py:83`;
**`OneLoopIntegral`** and its factory — §6 again, and it is the user's;
`DEFAULT_QUADRATURE_ATOL` and `DEFAULT_QUADRATURE_RTOL` — `QuadSourceIntegral` is already
decoupled and its two constants are `source-remediation`'s, not this campaign's;
`ComputeTargets/QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`, `AdaptiveLevin/`
(README §0.4); `break_point_kind`, the source grid, the band, `RESIDUAL_WKB_REGION_MARGIN`,
`BREAK_POINT_KIND` (README §0.5); `ComputeTargets/tests/test_numeric_break_point_key.py`.

**Read first:** README §1.2 (the five provenance fields), §2 (a), (e) and (g); §5 rules 5, 8 and
**9**; §7 **D1** in full, **D9** and **D10**; board items **T8** and **T10**;
`prompts/tolerance-convergence/logs/03a-tk-numeric-and-exit-time.md`'s "State handed to the next
prompt", which holds the five provenance fields for all four pairs and **which you cite rather than
restate**; `prompts/tolerance-convergence/logs/05-replace-the-vestigial-key-columns.md`'s "State
handed to the next prompt", whose site-by-site table is your index into `main.py` and which
prompt 05b left standing untouched; and the four issues named in §6.

---

## 1. Why this prompt exists

Prompt 05 removed the tolerance pair from the four object types that never used one. **Four are
left, and theirs are real** — each reaches a solver and sets the accuracy of what is stored:

| target | pair today | reaches |
|---|---|---|
| `wavenumber_exit_time` | `DEFAULT_ABS_TOLERANCE`, `DEFAULT_REL_TOLERANCE` | `brentq`'s `xtol`/`rtol` in `_solve_horizon_exit` |
| `GkNumericIntegration` | `DEFAULT_ABS_TOLERANCE`, `DEFAULT_REL_TOLERANCE` | DOP853 |
| `TkNumericIntegration` | `DEFAULT_TK_NUMERIC_ABS_TOLERANCE`, `DEFAULT_REL_TOLERANCE` | DOP853 |
| `QuadSourceIntegral` | `DEFAULT_QUADRATURE_ATOL`, `DEFAULT_QUADRATURE_RTOL` | the Levin/quadrature stack |

Three of the four share `DEFAULT_ABS_TOLERANCE` or `DEFAULT_REL_TOLERANCE` with each other and with
things that are not tolerances at all. **One constant cannot be right for a Green's function whose
$|G|\sim10^{10}$, a transfer function whose $|T|\sim10^{-5}$ and a root solve in $u=\log(1+z)$**,
and the campaign has now measured all three separately: prompt 03 for $G_k$, prompt 03a for $T_k$
and the exit time. `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` already exists and already carries the
comment standard the rest must match — it is the pattern, not an exception.

`QuadSourceIntegral` is the one of the four that is **already** decoupled, and it stays exactly as
it is. It appears here only so that the guard of §4 can classify it.

## 2. The constants (board item **T8**)

**Every number below was settled by the user under D1 on 2026-09-17 and none of them is yours.**
You are shipping them, with their provenance, not choosing them.

| constant | value | status | provenance |
|---|---|---|---|
| `DEFAULT_GK_NUMERIC_ABS_TOLERANCE` | **1e-10** | new name, **unchanged** value; **inert and unchosen** | prompt 03, `GK-NUMERIC-SWEEP.md` |
| `DEFAULT_GK_NUMERIC_REL_TOLERANCE` | **1e-8** | new name, **unchanged** value; **chosen** | prompt 03, `GK-NUMERIC-SWEEP.md` §5.2 and §7 |
| `DEFAULT_TK_NUMERIC_ABS_TOLERANCE` | **1e-13** | **exists already**; unchanged and **re-characterised** | user 2026-09-12; re-characterised by prompt 03a |
| `DEFAULT_TK_NUMERIC_REL_TOLERANCE` | **3e-11** | new name, **changed** from 1e-8; **chosen** | prompt 03a, `TK-NUMERIC-AND-EXIT-TIME.md` §4.1 and §5 |
| `DEFAULT_HEXIT_ABS_TOLERANCE` | **1e-10** | new name, **unchanged** value; **inert and unchosen**, and **coupled** | prompt 03a |
| `DEFAULT_HEXIT_REL_TOLERANCE` | **1e-9** | new name, **changed** from 1e-8; **chosen** | prompt 03a, `TK-NUMERIC-AND-EXIT-TIME.md` §7.6 |

**Rule 9 is an acceptance condition here, not advice.** Each constant carries, in
`config/defaults.py` at the point of use, a comment of the standard
`DEFAULT_TK_NUMERIC_ABS_TOLERANCE` already sets: the measurement that chose it, the floor it is
competing against, and its cost. The five §1.2 fields for all four pairs are in **log 03a's** "State
handed to the next prompt" — **cite them; do not restate them**, and do not re-derive a number.

**Three qualifications ship *with* the numbers and must appear in the comments**, because a reader
who finds only the value will mis-read all three:

1. **`3e-11` is a measured setting, not a bound.** `[03a-tk-numeric-excursion-is-sporadic-in-rtol]`
   is open and the acceptance did not close it: the $T_k$ excursion is *not monotone* in `rtol`, and
   from `1e-9` down exactly one of 150 runs exceeds 3e-6 at each setting and it is a **different**
   run each time. `3e-11` is the loosest that clears *in that sweep*.
2. **`DEFAULT_HEXIT_ABS_TOLERANCE` is coupled to its partner**, not merely inert: `brentq` stops at
   `xtol + rtol*|u|`, so at `u ≈ 37.6` an `xtol` of 1e-10 floors the pair at ~1e-10 relative and no
   `rtol` below ~2.6e-12 can do anything. Say so where it is defined.
3. **The exit-time pair was accepted on the *guarantee* reading** (log 03a, deviation 4) — `1e-9` is
   the loosest whose *bound* clears `DEFAULT_REDSHIFT_RELATIVE_PRECISION = 1e-7`, not the loosest
   whose *achieved* displacement does.

**`DEFAULT_ABS_TOLERANCE` and `DEFAULT_REL_TOLERANCE` keep their values, 1e-10 and 1e-8.** §6
explains why, and it is not negotiable in this prompt. Whether they keep their *names* is §3's
design question.

## 3. The one design question, and it is yours

After the switch, **no object type keys on `DEFAULT_ABS_TOLERANCE` or `DEFAULT_REL_TOLERANCE`.**
What is left of `DEFAULT_ABS_TOLERANCE` is seven bare `fabs(a − b) <` float comparisons in modules
this prompt may not edit (§6) — a *different quantity* wearing a tolerance's name, which is exactly
the confusion `[02-shared-atol-doubles-as-a-float-comparison-epsilon]` was opened to record.

So: **do the two shared names survive this prompt, and as what?** Leaving them untouched is
defensible — the seven sites are outside your grant and renaming reaches them. Renaming them to say
what they now are is also defensible, and might be the only moment in the campaign when it is cheap.
**Choose, implement it, and justify the choice in the log**, against the constraint that the seven
sites are not yours and that their *value* must not move whatever you do. Do not assume the answer
is "leave them".

## 4. The plumbing, and the guard that should catch a missed site (board item **T10**)

**Every `object_get` of a retuned target moves with it.** Log 05's "State handed to the next prompt"
enumerates them and prompt 05b changed not one line of `main.py`, so that table stands exactly as
written: `wavenumber_exit_time` ×1, `TkNumericIntegration` ×5, `GkNumericIntegration` ×4,
`QuadSourceIntegral` ×2, plus the conditional pair in `build_missing_GkSource`'s
`object_read_batch`, which is carried for `GkNumericValue` and not for `GkWKBValue` — **keep that
conditional and change only which tolerance object it names.** Build the new tolerance objects in
the same `ray.get` (`main.py:3542`) as the existing five.

**A missed site does not crash.** The tolerance is part of the key, so a lookup under the old value
simply fails to find the row and the pipeline recomputes it — silently, at full cost. That is the
batch-dispatch hazard `GkTk-remedial` prompt 12 hit, and it is why the guard exists.

**`QuadSource` carries a pair that reaches nothing, and it is not a schema question.**
`main.py`'s `build_tensor_source_work` query batch passes `"atol"`/`"rtol"` into an
`object_get_vectorized("QuadSource", …)`, and **`QuadSource` names neither `atol` nor `rtol`
anywhere** — not in `ComputeTargets/QuadSource.py`, not in its factory, and it has no tolerance
column to drop. They are dead payload keys. Remove them; that edit is in `main.py` alone and
touches no §0.4 file. Then classify `QuadSource` in the guard as carrying no tolerance.

**The guard's predicate is too narrow to do its job, and widening it is half of T10.** As prompt 05
left it, `test_main_plumbing.py:559` filters candidate sites with
`first.value.endswith("Integration")`. That matches **four** class names. It does not match
`BackgroundModel`, `GkSource`, `wavenumber_exit_time`, `QuadSource` or `QuadSourceIntegral` — note
that the last ends in "Integral", not "Integration", so the one already-decoupled target has never
been guarded at all. Replace the predicate with an **explicit enumeration of every object type
`main.py` looks up**, each classified as carrying a tolerance or not, and make the guard fail on

- a site whose class is **not in the enumeration** (the case that catches a new target), and
- a site whose tolerance **disagrees with its class's classification** (the case that catches a
  missed switch, and a tolerance reappearing on one of prompt 05's four).

Fold `NO_TOLERANCE_INTEGRATIONS` and the `EXPECTED_*_SITES` counters into whatever shape that takes;
the counters exist because a finder that silently matched nothing would pass every assertion, and
that reason still holds however you restructure it. The finder already walks both `object_get` and
`object_get_vectorized`, which is why the `QuadSource` site above is visible to it at all.

## 5. The readers, and two different failure modes

All six `extract_*.py` scripts build `atol`/`rtol` from the shared constants at module import and
use them in lookups that this prompt retunes. **Every one of them must follow**, and you must work
out site by site which target each lookup reaches — the characteristic error here is changing a
pair that feeds the wrong class, as prompt 05's `extract_Gk_data.py:407` trap showed.

Two anchors you are given, because they are already recorded:

- **All six** query `wavenumber_exit_time` in a `create_k_exit_work` helper, with the shared pair.
- `extract_TkWKB_data.py`'s `TkNumericIntegration` lookup is
  `[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]`, live since `GkTk-remedial` prompt
  12: it already queries under `DEFAULT_ABS_TOLERANCE` while `main.py` writes under
  `DEFAULT_TK_NUMERIC_ABS_TOLERANCE`, so it cannot match a production row **today**. This prompt
  closes it. **None of the six imports the split constant**, so check the others for the same fault
  before you assume it is the only one.

**The two failure modes are not the same and the log must distinguish them.** For the three
equality-keyed targets a stale reader **misses**: it finds nothing and says so. For
`wavenumber_exit_time` it does not miss — `[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`
records that its lookup takes *the loosest row at least as tight as the request*, so a reader still
asking for `1e-8` will **silently reuse** the new `1e-9` row and then report the stored pair rather
than the requested one. Tightening the production value makes every stale exit-time reader quietly
succeed with different provenance. Say which readers you changed and which mode each was in.

## 6. What this prompt does not do

- **It does not touch the seven float-comparison sites.** `DEFAULT_ABS_TOLERANCE` is used as a bare
  `fabs(a − b) <` epsilon at `GkSource.py:96`, `:104`, `:275`, `numeric_with_phase_cut.py:618`,
  `:737`, `:790` and `WKBtools.py:83`, with no connection to any ODE.
  `[02-shared-atol-doubles-as-a-float-comparison-epsilon]` **stays open** and this prompt must not
  close it: the hazard it names is precisely that a prompt retuning the constant moves all seven, in
  modules outside its file list. Keeping the value fixed is how you avoid that. Record in the log
  what the seven sites are now reading.
- **It does not decouple `OneLoopIntegral`.** It is a ninth keyed object type — `atol_serial` and
  `rtol_serial` columns that it genuinely filters on — which `main.py` never builds and whose object
  count is 0. `[02-oneloopintegral-is-a-ninth-keyed-object-type]` is **unassigned and explicitly the
  user's**: prompt 02 §8 asks whether decoupling a target that computes nothing is premature. Leave
  it, and do not add it to the guard's enumeration as though the question were settled — if your
  enumeration needs to mention it, mention it as *out of scope pending the user*, and say so in the
  log.
- **It changes no schema, no Gauss order, and no algorithm.** Not the root solve's bracketing, not
  `DEFAULT_HEXIT_TOLERANCE` (a bracket width of 1e-2 that is not a tolerance), not DOP853's
  selection, not the quadrature stack.

## 7. The tests

The guard of §4 is the main one and it is an existing module. Beyond it:

- **A test that fails if a target's `object_get` sites disagree with each other.** All five
  `TkNumericIntegration` sites must name the same tolerance; so must all four `GkNumericIntegration`
  sites. The existing `TkNumericToleranceWiringTestCase` is the pattern.
- **A test that the six readers agree with `main.py`.** A reader querying a target under a different
  constant from the one `main.py` writes it with is exactly
  `[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]`, and it went unnoticed from
  `GkTk-remedial` prompt 12 until prompt 02 found it by reading. An `ast` check over the six scripts
  that compares each lookup's constant against `main.py`'s for the same class would have caught it
  the day it landed. **Build it if you can do so without a datastore**; if you conclude you cannot,
  say why in the log rather than shipping a weaker test silently.

Neither may need Ray or a datastore (README §5, `CLAUDE.md`): `test_main_plumbing.py` reads
`main.py` with `ast` and never executes it, which is the pattern.

## 8. Acceptance

1. The six constants of §2 exist with exactly those values, each with a §1.2-standard comment, and
   the three qualifications appear where §2 requires.
2. `DEFAULT_ABS_TOLERANCE` = 1e-10 and `DEFAULT_REL_TOLERANCE` = 1e-8 in value, whatever §3 decides
   about their names.
3. Every `object_get` of a retuned target names its own constant; sites of one target agree with
   each other; the `build_missing_GkSource` conditional is preserved.
4. `QuadSource`'s dead payload pair is gone from `main.py`, and no file under §0.4 is touched.
5. The guard enumerates every object type `main.py` looks up, fails on an unclassified class and on
   a site disagreeing with its class, and its counters are updated. `QuadSourceIntegral` is guarded
   for the first time.
6. All six `extract_*.py` follow, each lookup matched to the target it reaches; all compile; the log
   separates the miss cases from the exit-time silent-reuse case.
7. **No schema change**: the six factories' column lists read exactly as at `a351a50`, and no file
   under `Datastore/SQL/ObjectFactories/` is in the diff.
8. The four Gauss orders are still 4 and nothing under prompt 05b's grant moved.
9. `ComputeTargets` **must not fall below 516** and should rise; `CosmologyModels` **39**. Both OK.
   `test_numeric_break_point_key.py` and `test_gauss_order_key.py` both pass **unedited**.
10. The three published source-grid digests are unmoved: `3bef2c06`, `60a3205a`, `21ffc126`.
11. `black --check` clean on every `.py` in the diff.
12. Board rows 05a, items **T8** and **T10**, §3/§4, and `docs/OPEN_ISSUES.md` with its count and
    date corrected — all in the **same commit**.
    `[02-extract-tkwkb-queries-tk-numeric-under-the-shared-atol]` closes and part (ii) of
    `[02a-grid-digest-not-reproducible]` closes; `[02-shared-atol-doubles-as-a-float-comparison-epsilon]`,
    `[02-oneloopintegral-is-a-ninth-keyed-object-type]` and
    `[03a-tk-numeric-excursion-is-sporadic-in-rtol]` stay **open**.

## 9. Stop conditions

Stop and report rather than working around any of these:

- **You believe one of §2's numbers is wrong.** D1 closed on 2026-09-17 and every value is the
  user's. Report the argument; do not ship a different number.
- **You find a target that should be retuned and is not in §1's four**, or one of the four that
  should keep the shared pair.
- **You want to decouple `OneLoopIntegral`**, or to touch the seven float-comparison sites, or any
  factory, column or Gauss order.
- **Changing the exit-time pair moves a source-grid digest.** `z_init` is a root-solve output and
  `[02a-grid-digest-not-reproducible]` is exactly about that sensitivity; a moved digest is a stop
  under README §4.3 even though this prompt is the one entitled to move the tolerance that causes
  it. **Check the digests before you conclude anything else**, and if one moves, report the measured
  displacement rather than adjusting a threshold.
- **A reader cannot be matched to a single target**, or you cannot build §7's cross-check without a
  datastore — say so rather than shipping a weaker test quietly.
- **`ComputeTargets` falls at all.**

## 10. The log

`logs/05a-decouple-the-tolerances-that-are-real.md`, on the campaign's template, classifying every
deviation. Beyond the template:

- **Rule 9 in full**: the five §1.2 fields for every constant shipped, by **citation** to log 03a
  and the campaign documents, plus the three qualifications of §2 in your own words.
- **Your §3 decision** on the two shared names, and the argument for it.
- **The site inventory**: every `object_get` you changed, by target, checked against log 05's table,
  and the `QuadSource` pair removed.
- **The guard**: its new predicate, what it now enumerates, how `OneLoopIntegral` is represented,
  and what it would have caught that the old one would not.
- **The readers**: which of the six changed, which target each lookup reaches, and which failure
  mode each was in — miss, or exit-time silent reuse.
- **The digest check**, explicitly, with the numbers.
- **"State handed to the next prompt"** — prompt 06 is next and assembles
  `docs/TOLERANCE-PROVENANCE.md` from the campaign's logs. Say which accuracy parameters in the
  pipeline are now decoupled and measured, which are inherited and unmeasured, and which remain
  unexplained; §1.2's closing list of the ones nobody has ever chosen is the checklist.
