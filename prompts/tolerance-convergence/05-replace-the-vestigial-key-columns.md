# Prompt 05 — replace the vestigial key columns with the Gauss orders

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** board item **T9**, and with it `[20-wkb-gauss-orders-not-in-lookup-key]`
**Depends on:** the user's settlement of **D3** (2026-09-18) and on nothing else. Prompt 04
measured the orders, prompt 04b landed the evidence; this prompt puts them where a lookup can see
them.
**Recommended model:** **Opus** — the edit is mechanical in four factories and the judgement is
not: what makes this prompt succeed or fail is whether the order ends up *in the key* and whether a
row can still be written claiming an order it was not computed at.

**This is the first of the two prompts §3.5 was split into** (README §7 **D9**, 2026-09-18). It is
the **schema** half and it changes **no number**. The tolerance half — per-target constants in
`config/defaults.py`, the tolerance objects in `main.py`, every `object_get` switched, the `ast`
guard widened — is **prompt 05a**, and it runs after this one. **Do not do 05a's work here**, and
in particular do not touch `config/defaults.py`.

**Files you may create or touch:**
`Datastore/SQL/ObjectFactories/BackgroundModel.py`, `GkWKBIntegration.py`, `TkWKBIntegration.py`,
`GkSource.py` — the schema, the build query, `store()`, the inventory and validation paths;
`ComputeTargets/BackgroundModel.py`, `GkWKBIntegration.py`, `TkWKBIntegration.py`, `GkSource.py` —
whatever the constructor and payload change requires, **and no algorithm**;
`main.py` — the four `object_get` sites of those targets, **and nothing else in that file**;
the six `extract_*.py` scripts of §5;
`ComputeTargets/tests/test_run_identity.py`, and any other test module that names the columns;
plus this campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `config/defaults.py` — **not one line**, and no constant in it changes in this
prompt; `ComputeTargets/GkNumericIntegration.py`, `TkNumericIntegration.py`,
`CosmologyConcepts/wavenumber.py` or their factories — the three targets whose pair *does* reach a
solver keep it, and so does `QuadSourceIntegral`; `break_point_kind` on either numeric factory
(`GkTk-remedial` prompt 20 put it there and it is not this prompt's business); the source grid, the
band, `RESIDUAL_WKB_REGION_MARGIN`, `BREAK_POINT_KIND` (README §0.5);
`ComputeTargets/QuadSourceIntegral.py`, `QuadSource.py`, `phase_groups.py`, `AdaptiveLevin/`
(README §0.4); `ComputeTargets/tests/wkb_reference_data.json` — prompt 04b landed it and it is
finished.

**Read first:** README §2 (a) and (g); §5 rules 3, 4, 8 and 9; §7 **D2**, **D3** and **D9**; board
item **T9** and the issue `[20-wkb-gauss-orders-not-in-lookup-key]`;
`docs/tolerance-convergence/TOLERANCE-INVENTORY.md` §2.4, which is the measurement of the defect;
`prompts/tolerance-convergence/logs/04-order-governed-targets.md`'s "State handed to the next
prompt" for the five provenance fields of the four orders, **which you cite and do not restate**;
and `ComputeTargets/tests/test_numeric_break_point_key.py` in full — it is the pattern for §3, and
`GkTk-remedial` prompt 20 wrote it to prove exactly the property this prompt must prove.

---

## 1. Why this prompt exists

**The defect is live, and it is a silent one.** Four object types carry `atol_serial` and
`rtol_serial` as part of their datastore lookup key, and for all four the pair reaches no solver:

- `ComputeTargets/BackgroundModel.py:409-410` — "``atol`` and ``rtol`` are accepted for signature
  compatibility with ``BackgroundModel.compute`` and because they remain part of the datastore
  lookup key; the table has no tolerances."
- `ComputeTargets/GkWKBIntegration.py:333-336` — "The phase comes from the background model's
  conformal-time table and a residual table …, which have no tolerances. `self._atol` and
  `self._rtol` are kept because they are part of the datastore lookup key …; they are not passed
  here."

Meanwhile the integers that **do** set those objects' accuracy — `TAU_GAUSS_ORDER`,
`CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER` (`ComputeTargets/BackgroundModel.py:35,43,44`) and
`RHO_GAUSS_ORDER` (`ComputeTargets/phase_residual.py:90`) — are module constants in no column.

**What that costs, concretely.** Raise `TAU_GAUSS_ORDER` from 4 to 6 today and re-run. The lookup
key does not move, the pipeline finds the existing order-4 row, and serves it. The order **is**
recorded — `main.py:3593` fetches an `IntegrationSolver` with
`stepping=BackgroundModel.TAU_GAUSS_ORDER` and `main.py:3602` does the same for the phase solver
with `PHASE_SOLVER_STEPPING`, which is `RHO_GAUSS_ORDER`
(`Quadrature/integrators/WKB_phase_function.py:86`), so the row's `solver_label` reads
`cumulative-GL-stepping4` — but **`solver_serial` is joined and never filtered on**:
`Datastore/SQL/ObjectFactories/BackgroundModel.py:255-271` joins
`solver_table` into the `select_from` and its `.filter(...)` constrains
`cosmology_type`, `cosmology_serial`, `atol_serial`, `rtol_serial` and nothing else.
`GkWKBIntegration.py:230-242` and `TkWKBIntegration.py` are the same shape. So the truth about the
computation is in the row, in a column no lookup consults, behind a key made of two numbers that
describe nothing.

**The user settled what replaces it on 2026-09-18** (README §7 **D3**), with a standing instruction
that governs this prompt and every later schema question in the campaign: *there is nothing to
rebuild and nothing to backfill, a schema change costs nothing, and a schema change that is needed
is always the right answer.* **You may not cite migration cost, row counts, or the absence of a
backfill value as an argument anywhere in this prompt's work** — not in the code, not in the log,
not as a reason to prefer one design. What may be costed is **reader breakage**, because a script
that queries a column that no longer exists simply stops working; §5 is that list and it is longer
than D3's parenthetical says.

## 2. What the schema becomes

| Table | Loses | Gains |
|---|---|---|
| `BackgroundModel` | `atol_serial`, `rtol_serial` | `tau_gauss_order`, `cs_tau_gauss_order`, `friction_F_gauss_order` |
| `GkWKBIntegration` | `atol_serial`, `rtol_serial` | `rho_gauss_order` |
| `TkWKBIntegration` | `atol_serial`, `rtol_serial` | `rho_gauss_order` |
| `GkSource` | `atol_serial`, `rtol_serial` | — |

**This table is exhaustive.** It is the user's settled D3 and you implement it rather than
extending it.

`BackgroundModel` gains **three** columns and not one, because the three primitives are three
independent integrals — $1/H$, $c_s/H$ and $-\tfrac32(1+c_s^2)/(1+z)$ — each with its own
constant and each independently movable. A single `gauss_order` column would be a claim that they
must move together, which is false and which the next person to move one would discover the hard
way. Two of the three are in a worse position than $\tau$ to begin with:
`TOLERANCE-INVENTORY.md` §2.4 measured that only $\tau$'s order reaches a solver *label* at all, so
**`CS_TAU_GAUSS_ORDER` and `FRICTION_F_GAUSS_ORDER` leave no trace anywhere in the store today** —
not in a key, not in a label, not in a tag.

`GkSource` gains nothing and that is the whole answer for it: `compute()` calls
`assemble_GkSource_values`, which stitches the numeric and WKB results together and **integrates
nothing**. There is no order to record because there is no quadrature. Say so in the code where the
columns used to be.

**`RESIDUAL_WKB_REGION_MARGIN` is the fifth axis `[20-wkb-gauss-orders-not-in-lookup-key]` names,
and it does not become a column.** Not by oversight: prompt 04 swept it from 0.05 to 0.9 and found
the stored residual **bit-identical throughout** (`ORDER-AUDIT.md`; board **T7**), so it is a
reachability bound on where the residual may be evaluated and not an accuracy axis of the object
that is stored. A key column exists to separate rows that differ; this one cannot produce two rows
that differ. **Verify that reading against `ORDER-AUDIT.md` yourself rather than taking this
paragraph for it**, record what you found in the log, and if you conclude it *should* be keyed,
**stop and say so** — that is a D3 amendment and not yours to make.

## 3. In the key, not merely in the table

**Adding a column is the easy half and it is not the point.** A `tau_gauss_order` column that
`store()` writes and `build()` selects but does not filter on reproduces the exact defect §1
describes, one identifier later. The property this prompt must deliver is:

> **When the constant moves, the key moves.** Change `TAU_GAUSS_ORDER` from 4 to 6 and the
> `BackgroundModel` lookup must miss the order-4 row and compute a new one. Likewise
> `CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER` and, in both WKB sectors, `RHO_GAUSS_ORDER`.

and, with it:

> **A row cannot claim an order it was not computed at.** Whatever mechanism you choose, there must
> be no path — no keyword, no payload key, no default — by which the value written into the key
> differs from the order the computation actually used.

**How you achieve the second is your design decision and the prompt does not make it for you.**
Passing the order in from `main.py` the way `atol` was passed is explicit and mirrors what is
there; having the compute class or factory read its own module constant makes the divergence
impossible to express. Both can be made correct and both can be got wrong. **Choose, implement it,
and justify the choice in the log against the invariant above** — that justification is what the
review reads.

**`ComputeTargets/tests/test_numeric_break_point_key.py` is the pattern**, and you should follow it
rather than invent one. `GkTk-remedial` prompt 20 faced this same question for `break_point_kind`
and answered it with `_capture_build_query`, which compiles the `whereclause` and asserts the column
is *"filtered on, not merely selected"* (that module's own words, at its line 356). Do the same for
each of the four orders on each of the three factories that gain them, and assert the negative for
`GkSource` — that no tolerance survives in its criteria.

**That module must keep passing unedited.** It asserts that `atol_serial` and `rtol_serial` are in
the `GkNumericIntegration` and `TkNumericIntegration` criteria, and those two targets keep their
pair: their tolerances reach a DOP853 solver and are real. If you find yourself editing it, you have
changed a target this prompt does not own.

## 4. No number changes in this prompt

`TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER` and `RHO_GAUSS_ORDER` are all
**4** and all four stay **4**. Prompt 04 measured every one of them `unchanged` under README §6.1
rule 4, the floors dominating the loosest order swept by ×3.03e5, ×1.99e5, ×48.4 and ×2.35e6, and
prompt 04b landed that evidence. This prompt records those values in a key; it does not revisit
them.

`config/defaults.py` is **byte-identical** at the end of this prompt. `DEFAULT_ABS_TOLERANCE` and
`DEFAULT_REL_TOLERANCE` keep their present values and their present names — three live targets
still use them, and re-pointing those is **prompt 05a**.

So: `git diff HEAD~1 HEAD -- config/defaults.py` is empty, and that is an acceptance condition, not
an observation.

## 5. The readers

**D3's parenthetical says three `extract_*.py` scripts and the tree says six.** Every one of them
performs a `BackgroundModel` lookup with `atol=`/`rtol=`, and two also look up `GkSource`:

| script | `BackgroundModel` | `GkSource` | a WKB target |
|---|---|---|---|
| `extract_Gk_data.py` | :305 | — | — |
| `extract_GkWKB_data.py` | :347 | — | `GkWKBIntegration`, batch dict :446 |
| `extract_TkWKB_data.py` | :371 | — | `TkWKBIntegration`, batch dict :439 |
| `extract_GkSource_data.py` | :760 | :872 | — |
| `extract_tensor_source_data.py` | :303 | — | — |
| `extract_QuadSourceIntegral_data.py` | :959 | :1098 | — |

All six must follow, or they query a column that no longer exists. `extract_Gk_data.py:407`'s batch
dict feeds **`GkNumericIntegration`**, which keeps its pair — leave it alone; getting that one wrong
in either direction is the characteristic error here, so check what each dict is dispatched into
before editing it. Correct D3's count in the README **only** by noting it in your log and opening
nothing: the README is the orchestrator's file, not yours.

`docs/tolerance-convergence/inventory.py` and `order_audit.py` read the column names as *data* and
are this campaign's own measurement scripts. **Leave both alone.** Their published output was
correct for the tree it was taken on (README §5 rule 7) and prompt 06 restates the inventory. Note
in the log that their next run will report differently.

## 6. The tests

`ComputeTargets/tests/test_run_identity.py` builds its stand-in schema **from the factory's own
`register()`** (its line 110's docstring says why: so that the test cannot drift from the factory),
so most of it follows the change for free. What will not follow is where it names the columns
explicitly — the raw `sqla.insert` at :731-745 that writes a pre-prompt-14 row, and the module
docstring at :9 describing the key as `(cosmology_type, cosmology_serial, atol_serial,
rtol_serial)`. Repair both. That module is the one that will catch a half-done schema change, so
**read its failure carefully rather than editing until it is quiet.**

Add the key tests §3 asks for. They must need neither Ray nor a datastore (README §5, `CLAUDE.md`):
follow the stand-in pattern `test_numeric_break_point_key.py` and `test_run_identity.py` already
use.

## 7. Acceptance

1. The four tables have the schema of §2 — the pair gone from all four, three orders on
   `BackgroundModel`, one on each WKB factory, nothing new on `GkSource`.
2. For each of the four orders, a test proves it is **filtered on, not merely selected**, in the
   build query of every factory that carries it; and a test proves no tolerance survives in
   `GkSource`'s criteria.
3. A test proves the §3 invariant directly: with the constant moved, the compiled `whereclause`
   changes. **A test that only checks the column exists does not satisfy this row.**
4. No row can be written whose recorded order differs from the order used. The log states the
   mechanism and why it makes that unexpressible.
5. `config/defaults.py` is byte-identical, and all four orders are still 4.
6. `break_point_kind` is untouched on both numeric factories, and
   `ComputeTargets/tests/test_numeric_break_point_key.py` passes **unedited**.
7. All six `extract_*.py` scripts of §5 import and parse (`python -m py_compile`, and an `ast`
   read for the `object_get` payloads), with no reference to `atol_serial`/`rtol_serial` for any
   of the four retuned targets, and `extract_Gk_data.py:407`'s `GkNumericIntegration` dict
   unchanged.
8. `ComputeTargets` **must not fall below 452** and should read **491 or more** — this prompt adds
   test modules, so a rise is expected and a fall is not; `CosmologyModels` **39**. Both OK.
9. `black --check` clean on every `.py` in the diff.
10. Board row 05, item **T9**, §3/§4, and `docs/OPEN_ISSUES.md` with its count and date corrected —
    all in the **same commit**. `[20-wkb-gauss-orders-not-in-lookup-key]` closes, with the
    `RESIDUAL_WKB_REGION_MARGIN` half answered on §2's evidence rather than dropped.
11. The three published source-grid digests are unmoved: `3bef2c06`, `60a3205a`, `21ffc126`.

## 8. Stop conditions

Stop and report rather than working around any of these:

- **You cannot put an order in the key without changing what the computation does.** That is a
  charter question, not an implementation one.
- **You conclude `RESIDUAL_WKB_REGION_MARGIN` must be keyed** (§2), or that any table needs a
  column D3's table does not list, or that one of the four should keep its pair.
- **A target outside the four changes its verdict**, or you find yourself editing
  `GkNumericIntegration`, `TkNumericIntegration`, `wavenumber_exit_time`, `QuadSourceIntegral`,
  `config/defaults.py`, or `test_numeric_break_point_key.py`.
- **A source-grid digest moves.**
- **You find yourself wanting to change an integrator's algorithm** (README §0.5) to make a
  schema change land.
- **`main.py` needs a change beyond the four `object_get` sites.** The tolerance objects, the
  `ray.get` block and the guard are **prompt 05a's**, and touching them here destroys the split.

## 9. The log

`logs/05-replace-the-vestigial-key-columns.md`, on the campaign's template, classifying every
deviation. Beyond the template:

- **The mechanism** you chose for §3's second invariant, and the argument that a row cannot claim
  an order it was not computed at.
- **The `RESIDUAL_WKB_REGION_MARGIN` finding** of §2, in your own words, against `ORDER-AUDIT.md`.
- **The reader set**: which of the six scripts you changed and what each one was querying, and the
  note that D3's count of three was low.
- **"State handed to the next prompt"** — prompt 05a is next and needs to know exactly which
  targets still carry a tolerance after this commit, which `object_get` sites in `main.py` still
  pass `atol=`/`rtol=`, and whether `DEFAULT_ABS_TOLERANCE` and `DEFAULT_REL_TOLERANCE` now have
  fewer consumers than they did. The four orders' five provenance fields (README §1.2) are in
  **log 04's** "State handed to the next prompt" and **must not be restated here**; cite them, and
  add only what this commit changes about them: that they are now in a key.
