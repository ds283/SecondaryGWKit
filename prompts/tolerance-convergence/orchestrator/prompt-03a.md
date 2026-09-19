# Orchestrator — prompt 03a, `TkNumericIntegration` and `wavenumber_exit_time`

**Campaign:** [`../README.md`](../README.md) · **Rules:** [`README.md`](README.md) ·
**Prompt:** [`../03a-tk-numeric-and-exit-time.md`](../03a-tk-numeric-and-exit-time.md)
**Model:** Opus. **Production code changed:** none. **Test code changed:**
`ComputeTargets/tests/convergence_reference.py`, **additive only**, and only if §2.5's exit-time
helper needs it — T5 should need nothing at all.

Read [`README.md`](README.md) first — the seven binding rules, the suite checks and the dispatch
template are there and are not repeated here.

**This prompt is the second half of the charter split on 2026-09-17** (campaign README §7 **D7**).
Prompt 03 took `GkNumericIntegration` and the consumer-spline floor, board item **T4**; **03a takes
`TkNumericIntegration` and `wavenumber_exit_time`**, board items **T5** and **T6**. No item was
renumbered.

**It closes D1 if the user accepts it.** The $G_k$ half was accepted on 2026-09-17; these are the
remaining halves, and **prompt 05 may not start until they are accepted too**. So this prompt, like
03, ends in a **hand-back**.

---

## 1. Before you dispatch

Record `git rev-parse --short HEAD` and the three baselines. At `8143f0c` they are **491**
(`ComputeTargets`), **39** (`CosmologyModels`) and **32** (`test_convergence_reference`).
`ComputeTargets` must not fall below **452**, the campaign floor, and should be at 491 here.

**The facility count matters again and for a wider reason than at 03.** Prompt 03 was the first
prompt permitted to touch `convergence_reference.py`; this is the second, and it must now preserve
**two** prompts' published figures rather than one:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_convergence_reference 2>&1 | grep -E "^(OK|FAILED|Ran )"
```

Any movement in that count after the agent's commit is a **stop** under §3.6.

**Precondition:** prompts 01, 02, 02a and **03** have landed, and **the user has accepted D1's $G_k$
half** (`8143f0c`). That acceptance matters to this prompt specifically: it is what makes
`GkNumericIntegration` closed rather than merely measured, so an agent that finds its own sector
behaving differently has no licence to reopen it.

## 2. Dispatch

Use [`README.md`](README.md)'s template with `NN-<name>` = `03a-tk-numeric-and-exit-time`, model
**Opus**. **The template's standing parameter-freeze sentence applies unchanged** — this prompt
changes no production code and has no D6 carve-out. Add the same sentence prompt 03 carried:

> **You may add to `ComputeTargets/tests/convergence_reference.py` and may not change an existing
> line of it.** Prompts 01's and 03's published figures must both reproduce bit-identically
> afterwards, and `ComputeTargets/tests/test_convergence_reference.py` is what checks that. If the
> check your prompt §2.5 requires cannot be built additively, stop and say so rather than editing.

**Do not tell the agent what to expect, and be more careful of this than you were at 03.** At 03 the
hazard was repeating the prompt's own "`unchanged` is the likely outcome". Here the hazard is
sharper and runs the other way: prompt 03 has just landed a large, well-evidenced `unchanged` in an
adjacent sector, and an orchestrator who mentions that — or who hints that the $T_k$ sector is
"where something might actually move" — has told the agent which of two opposite answers to come
back with. The prompt's §4.3 already forbids carrying prompt 03's conclusions over; your dispatch
must not undo that by putting them in the agent's head before it starts.

## 3. The review — seven checks

1. **Was `BREAK_POINT_ALL` actually passed?**

   This is the single most likely silent failure in the whole prompt, because `tk_run` **defaults**
   to `BREAK_POINT_DISCONTINUITY` and an agent that omits the argument gets a clean, plausible,
   wrong sweep that reproduces prompt 17 rather than production.

   ```bash
   grep -n "break_point_kind\|BREAK_POINT" docs/tolerance-convergence/tk_numeric_exit_sweep.py
   ```

   Every `tk_run` call that feeds a published `TkNumericIntegration` figure must pass
   `BREAK_POINT_ALL` explicitly. A sweep that relies on the default is a **stop**, and so is a
   document that does not say which policy each table was taken under. If the agent also re-took
   anything under `BREAK_POINT_DISCONTINUITY` for the §6-question-1 comparison, that is correct and
   expected — check the two are labelled and never mixed in one table.

2. **Is the $T_k$ measurement its own, or prompt 03's with the labels changed?**

   Prompt §4.3 permits citing `GK-NUMERIC-SWEEP.md` and forbids carrying its conclusions. The three
   findings most likely to be borrowed are "`atol` is inert", "10.1 per decade" and "the floor
   dominates, so `unchanged`". For each one that appears in the $T_k$ document, check there is a
   $T_k$ table behind it and that `tk_numeric_exit_sweep.py` produces that table. **A number that
   traces to prompt 03's script is a stop.**

   Check too that the `atol` off-axis points of §4.1 are there and are *bounded* — the prompt asks
   for one decade either side at two or three `rtol` settings to test separability, not a second
   full axis, and not an `atol` recommendation. Either failure is a deviation worth reading the log
   for: too few and question 2 is unanswered, too many and D1's settled half is being reopened.

3. **Did the exit-time sweep go through `_solve_horizon_exit`, and not the store?**

   ```bash
   grep -n "_solve_horizon_exit\|wavenumber_exit_time\|object_get" docs/tolerance-convergence/tk_numeric_exit_sweep.py
   ```

   Direct calls only. **Any route through `wavenumber_exit_time` or the datastore is a stop**: that
   lookup is an inequality (`[02-wavenumber-exit-time-tolerance-is-an-inequality-key]`), so a
   loosened sweep point silently reuses a tighter stored row and the sweep measures the store.
   Check the three offsets of prompt §2.3 are all present — crossing, `suph_e5`, `subh_e4` — since
   those are the ones a grid is actually built from.

4. **Was the $u \to z$ recovery addressed, or quietly ignored?**

   `_solve_horizon_exit` returns `exp(log_z_root) - 1.0`, and at $u \approx 37.6$ that conversion is
   lossy at a level `CLAUDE.md`'s redshift note quantifies. The document must say how much of the
   root's precision survives it. **An exit-time accuracy quoted in $z$ with no statement about the
   conversion is a defect**, not because the figure is necessarily wrong but because it is the one
   error in this target that cannot be tightened away and a reader will otherwise attribute it to
   the tolerance.

   Check also that the agent identified **which term of Brent's `xtol + rtol*|u|` binds**, by
   measurement. `[02a-grid-digest-not-reproducible]` says `rtol` binds at 3.8e-7 and `xtol` never
   does; the prompt requires that to be re-taken rather than inherited, so look for the agent's own
   figure beside the review's.

5. **Rule 6, applied honestly or dodged in both directions.**

   `wavenumber_exit_time` is the campaign's likeliest rule-6 row, and there are two opposite
   failures:
   - **a recommended pair with no floor behind it** — a number picked because a recommendation
     looks more finished than a finding. Check what bound is claimed and what measured it.
   - **rule 6 invoked to avoid the work** — "no floor could be established" written before the
     displacement-to-grid question of prompt §2.4 was actually attempted.

   Either way the document must say what displacement the grid can absorb, or say in rule 6's own
   words that the record does not fix one. Both are complete answers; neither may be silent.

6. **Was §6.1's target rule applied to $T_k$, in the direction it actually points?**

   The same two failure modes as at prompt 03, and they are stops for the same reasons:
   - the agent recommends **tightening** — check it recommended the **loosest** `rtol` that clears
     the floor, not the tightest it measured (rule 3);
   - the agent recommends **`unchanged`** — check the word is in the cell and the **factor by which
     the floor dominates** is beside it (rule 4).

   Cost must be in **evaluations times objects**, at the setting and one step either side, and
   **never in wall time** (§2 (i)). Check the object count is this sector's — **50 per model** — and
   that the document says so; a reader arriving from prompt 03's ~65,000 will otherwise weigh a
   decade here as though it cost the same.

   Check finally that the **3e-6 excursion count** is carried at each setting. That is the statistic
   `[12-tk-numeric-atol-largest-k-excursion]` is written in, and a re-take that does not report it
   cannot close or narrow that issue however good its other tables are.

7. **The facility, the suites and the bookkeeping.**

   ```bash
   git diff --numstat HEAD~1 HEAD -- ComputeTargets/tests/convergence_reference.py
   ```

   Deletions must be **zero** — and the file may well be untouched, which is the better outcome.
   Re-run `test_convergence_reference` and compare with §1's count. `ComputeTargets` **must not fall
   below 452** and should read 491 or more; `CosmologyModels` must read **39**. Board rows 03a, item
   rows **T5** and **T6**, and `docs/OPEN_ISSUES.md` in the **same commit** with its count and date
   corrected if §3 or §4 moved. `black --check` clean on every `.py` in the diff.

   `[12-tk-numeric-atol-largest-k-excursion]` is the one existing issue this prompt can legitimately
   close or narrow. If the agent claims either, check the claim is against **its own** re-take under
   `BREAK_POINT_ALL` and not against prompt 17's figures.

## 4. What a good outcome looks like

- A $T_k$ sweep under the **production** break-point policy on the version-2 grid, with the
  difference from prompt 17's configuration quantified rather than asserted, and the two changes —
  policy and grid generation — separated if they can be and declared inseparable if they cannot.
- An answer to "does `atol = 1e-13` bind in this sector?" that is a **measurement**, taken without
  reopening the value the user settled on 2026-09-12.
- An exit-time answer that is honest about being a **location rather than a value**: the residual,
  the displacement from a converged re-solve, the oracle check on radiation, and either the grid's
  absorbable displacement or rule 6's words.
- A recommendation for D1 written so the user can take a decision from it, **and a statement that
  accepting it closes D1**.

**What a good outcome does *not* look like:** a $T_k$ recommendation that mirrors prompt 03's
`unchanged` because the adjacent sector's floor dominated, or one that tightens because this
sector's tightening is cheap. Both are answers imported from somewhere other than this sector's
measurement. The cost being trivial is a reason the decision is easy, not a reason to make it.

## 5. Stop and ask the user

Beyond [`README.md`](README.md)'s standing list:

- **An accuracy reported below a floor the agent itself just measured.** §2 (f) and §6.1 rule 5 —
  an arithmetic error, never a discovery, and campaign-wide.
- **A reference did not converge** at any $(model, k)$, for either target. Prompt 17's error. Not a
  wavenumber to drop and not a criterion to loosen.
- **A moved grid digest**, for either published grid, at any point and for any reason.
- **The agent proposes to touch `CosmologyConcepts/wavenumber.py`**, the digest, the quantisation,
  `SOURCE_GRID_MIN_SEPARATION`, `DEFAULT_REDSHIFT_RELATIVE_PRECISION`, the density guard's mask
  ordering, the band, the grid or any parameter. Each is named out of scope in prompt §7, and the
  digest-reproducibility fix in particular will look like the obvious thing to do — it is
  **prompt 05's**, and this prompt supplies only its prerequisite measurement.
- **The agent proposes to reopen `GkNumericIntegration`** because its own sector behaves
  differently. T4 is measured and **accepted**; a genuine inconsistency is a §3 issue and a question
  for the user, not a re-measurement.
- **The agent asks you to choose a target, a floor, a tolerance or an absorbable displacement.**
  Relay verbatim (rule 5); do not pick a number, and in particular do not tell it what prompt 03
  found or what `[02a-grid-digest-not-reproducible]` recommends.
- **Any subagent that reports `PARTIAL`, `BLOCKED`, or an `UNINTENDED DRIFT` it kept.**

## 6. After it lands

**Hand back to the user**, because **D1 is theirs and this is what closes it**. Report, in this
order:

1. **What `BREAK_POINT_ALL` changed** — §6 question 1, in one line, with the excursion count and
   worst figure beside prompt 17's 3 / 13 / 8 and 8.64e-4.
2. **Whether `atol = 1e-13` binds** in this sector, with the figure, and the $T_k$ initial-condition
   floor as re-confirmed on this tree beside its inherited 2.52e-6.
3. **The `TkNumericIntegration` recommendation** — a setting, or the word `unchanged` — with the
   cost at that setting and one step either side in evaluations times **50 objects per model**.
4. **The `wavenumber_exit_time` answer** — a pair with its floor, or rule 6's words — together with
   what binds Brent's criterion and how much precision survives the $u \to z$ recovery.
5. **That accepting 3 and 4 closes D1**, and that prompt 05 unblocks on that acceptance.

Then stop. **Prompt 04 is next and is not written yet**; campaign README §3.4 fixes its charter and
§6.2 its acceptance rows, and D5 is already settled *yes*, so it may write the fixture. Write it
after the user has read this. Do not dispatch anything further, and do not start prompt 05 on the
strength of a partial acceptance — D1 closes on **both** of this prompt's targets, not on the $T_k$
half alone.
