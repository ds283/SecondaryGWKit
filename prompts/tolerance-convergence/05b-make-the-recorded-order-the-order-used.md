# Prompt 05b — make the recorded order the order that was used

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Closes:** board item **T15**, and with it
`[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]`
**Depends on:** prompt 05 (`90d0114`), which put the four orders in the lookup key. Nothing else.
**Recommended model:** **Opus** — the edit is small and the invariant is subtle. What makes this
prompt succeed or fail is whether an object *reports the order it was actually built at* on both
the compute path and the rehydration path, and the second is the one that is wrong today in a way
no test currently sees.

**This prompt runs next, before 05a** (README §7 **D10**, 2026-09-18, the user's decision). The
letter records insertion, not run order — a departure from the 02a/03a convention, made because
this is a **correctness** defect and 05a layers further plumbing onto the same objects.

**This prompt changes no number.** `config/defaults.py` is byte-identical and all four
`*_GAUSS_ORDER` constants are still **4** when you finish. It changes no algorithm and no
quadrature: every table this prompt touches must produce bit-identical values on the production
path.

**Files you may create or touch:**
`ComputeTargets/phase_residual.py`, `ComputeTargets/BackgroundModel.py`,
`ComputeTargets/GkWKBIntegration.py`, `ComputeTargets/TkWKBIntegration.py`;
`Quadrature/integrators/WKB_phase_function.py` — **only** what carrying the order out of the
residual table requires, and **no** change to the phase algorithm, the anchor, the node selection
or the leading term;
`Datastore/SQL/ObjectFactories/BackgroundModel.py`, `GkWKBIntegration.py`, `TkWKBIntegration.py` —
the `build()` paths, and **not** the schema, which prompt 05 settled;
`ComputeTargets/tests/test_gauss_order_key.py` and any other test module that asserts on the
orders; plus this campaign's log, board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `config/defaults.py` — **not one line**; the schema of any table — no column is
added, renamed, retyped or dropped in this prompt, and `GkSource` gains nothing, still;
`ComputeTargets/GkNumericIntegration.py`, `TkNumericIntegration.py`,
`CosmologyConcepts/wavenumber.py`, `ComputeTargets/QuadSourceIntegral.py`, `QuadSource.py`,
`phase_groups.py`, `AdaptiveLevin/` (README §0.4, §0.5); `break_point_kind`;
`ComputeTargets/tests/test_numeric_break_point_key.py`; the source grid, the band,
`RESIDUAL_WKB_REGION_MARGIN`, `BREAK_POINT_KIND`; `main.py` — **nothing in it**, because the order
is not the caller's to supply and this prompt does not make it so;
`docs/tolerance-convergence/inventory.py` and `order_audit.py`, and the verification scripts under
`docs/qcd-background-audit/` and `docs/gktk-remedial/` — see §4, which is about **not** breaking
them.

**Read first:** README §2 (a) and (g); §5 rules 3, 4, 7 and 8; §7 **D3**, **D9** and **D10**; board
item **T15**; `prompts/tolerance-convergence/logs/05-replace-the-vestigial-key-columns.md` in full,
especially its account of the mechanism and its "State handed to the next prompt";
`ComputeTargets/tests/test_gauss_order_key.py`, which is prompt 05's proof and which this prompt
extends rather than replaces.

---

## 1. Why this prompt exists

Prompt 05 put the four Gauss orders in the lookup key and proved each one is *filtered on, not
merely selected*. It established the mechanism as **one name resolved at call time**: the compute
path reads the module constant, the object's property returns the module constant, `store()` writes
that property, `build()` filters on the same module attribute.

**That mechanism is right about the key and wrong about the object.** A property that re-reads a
module constant does not report what the object *is*; it reports what the module *currently says*.
Those differ in two places.

**(a) The residual table can be built at an order the object will not record.**
`phase_residual.build_phase_residual` (`:164`), `phase_residual_cache_key` (`:375`) and
`cached_phase_residual` (`:396`) each take `order: int = RHO_GAUSS_ORDER`. The table honours it —
`CumulativeTable` stores it and exposes `.order` (`ComputeTargets/cumulative_table.py:421`) — but
`GkWKBIntegration.rho_gauss_order` (`:161`) and `TkWKBIntegration.rho_gauss_order` (`:175`) return
`phase_residual.RHO_GAUSS_ORDER` regardless. A caller passing `order=6` computes at 6 and persists
4. No production caller does so today, which is why the defect is silent and why nothing fails.

**(b) A rehydrated `BackgroundModel` reassembles its tables at the current constant, not at its
own.** `_build_tau_primitive` (`:913`), `_build_cs_tau_primitive` (`:948`) and
`_build_friction_F_primitive` (`:976`) take the persisted `hi`/`lo` limbs off the row and construct
a `CumulativeTable` whose order is `TAU_GAUSS_ORDER` / `CS_TAU_GAUSS_ORDER` /
`FRICTION_F_GAUSS_ORDER` **as the module reads them now**. The row's own
`tau_gauss_order`, `cs_tau_gauss_order` and `friction_F_gauss_order` columns are selected by
`build()` (`Datastore/SQL/ObjectFactories/BackgroundModel.py:340-342`) and **never passed to the
constructor** (`:542`). Every off-grid partial `delta()` later evaluated against that table uses
the current order over nodes that were integrated at the stored one.

**Both are masked today by the same accident**: `build()` filters on the constant, so a row that is
served always happens to carry the constant's value. Correctness rests on an argument about the
filter rather than on construction. That is the defect — not the numbers, which are all 4 and all
agree.

**This is a correctness prompt, not a hardening prompt.** Do not argue anywhere in this work that
the impact is zero and the change therefore optional. It is zero *today*, and the campaign's
standing instruction about schema applies in spirit here too: what is needed is right.

## 2. The invariant

> **An object reports the order it was built at, on every path by which it can come into
> existence.** Computed fresh, it reports the order its tables were actually constructed with.
> Rehydrated from a row, it reports the row's stored order, and rebuilds its tables at that order.

and, unchanged from prompt 05 and **not to be weakened**:

> **`build()` filters on the current module constant.** That is the lookup semantics — *give me a
> row computed at the order this run is configured for* — and it is what makes the key work. It
> does **not** become a filter on the object's own value.

These two together are the whole point. Once an object records what it used, the filter and the
record no longer have to agree by luck: a table built at 6 stores 6, and a run configured at 4
simply never sees it. **Say in the log why that is the correct behaviour rather than a miss.**

**A consequence you must reason about explicitly:** with the record made faithful, the `order=`
parameter on the three `phase_residual` entry points stops being a lie, because whatever it is
given is what gets recorded. **Whether it keeps its default is therefore your design decision, not
a correctness one** — and §4 is the constraint that should inform it. Choose, and justify the
choice in the log against the invariant above. Do not assume the answer is "make it required".

## 3. What that requires, by path

You are not being given the edit. You are being given the invariant and the three places it fails.
Determine the rest by reading. What the tree already gives you:

- `CumulativeTable.order` exists and is faithful (`cumulative_table.py:421`).
- `WKB_phase_function`'s payload already carries a `metadata` dict with `rho_nodes`, `rho_evals`
  and `rho_reused` (`:359-361`), built from the very `rho` table whose order is wanted.
- `GkWKBIntegration` already stores `self._metadata = payload["metadata"]` on the compute path
  (`:113`) and sets it to `None` on the build path (`:94`) — so **metadata is not a channel the
  rehydration path can use**, and the stored order must reach the constructor another way. That
  asymmetry is the heart of the edit and the `atol_serial`/`rtol_serial` pair is the tree's own
  precedent for it, as it stood before prompt 05.
- `compute_background` takes no order parameter at all and reads the three constants at its three
  `CumulativeTable` constructions (`:428`, `:450`, `:467`), echoing them into the payload as
  `tau_order`, `cs_tau_order`, `friction_F_order` (`:613-619`). **The compute path of
  `BackgroundModel` is already correct** and you should not add a parameter to it; what is missing
  is that the payload's values, not the module's, are what the object reports.

## 4. Do not break the readers, and do not rewrite their documents

Five call sites outside production pass no `order` today:
`docs/qcd-background-audit/grid_density_criterion.py:214`,
`docs/qcd-background-audit/consumer_knot_scheme_scan.py:423`, and
`docs/gktk-remedial/verify_production_path.py:735`, `:947`, `:1061`.
`docs/tolerance-convergence/order_audit.py:864` **does** pass one, because sweeping the order is
what prompt 04 built it to do — so the parameter cannot simply be deleted.

Those five belong to other campaigns' verification directories. README §5 rule 7 makes their
**published output** immutable; it does not make their scripts exempt from compiling. **Whatever
you choose in §2, every one of those six scripts must still import and parse afterwards**, and you
must check it (`python -m py_compile`) rather than assume it. If your design would break them, that
is evidence about the design, not a licence to edit six files across three campaigns.

## 5. No number changes, and no value moves

`TAU_GAUSS_ORDER`, `CS_TAU_GAUSS_ORDER`, `FRICTION_F_GAUSS_ORDER` and `RHO_GAUSS_ORDER` are all
**4** and all stay **4**. `config/defaults.py` is byte-identical.

**And this prompt must move no computed value at all.** It changes which integer an object reports
and which integer a rehydrated table is constructed with; on the production path both are 4 before
and after, so every table, every phase, every residual and every stored quantity is bit-identical.
The three published source-grid digests are unmoved: `3bef2c06`, `60a3205a`, `21ffc126`. If
anything you do moves a value, you have changed the computation and that is a stop (§8).

## 6. The tests

Extend `ComputeTargets/tests/test_gauss_order_key.py` rather than starting a module beside it — it
is prompt 05's proof and this prompt strengthens the same property. Its 16 tests must all still
pass, repaired only where they assert the *old* mechanism (a property that re-reads the module) as
though it were the invariant.

The tests this prompt turns on, both of which must fail against `90d0114`:

1. **A rehydrated object reports its row, not the module.** Build a stand-in row whose stored order
   is **not** the current constant, rehydrate it through the factory's `build()`, and assert the
   object reports the stored value — and, for `BackgroundModel`, that its reassembled primitives
   carry that order rather than the constant.
2. **A table built at a non-default order is recorded at that order.** Drive the WKB path with an
   order that is not `RHO_GAUSS_ORDER` and assert the value `store()` would write is the order the
   table was actually built at.

Neither may need Ray or a datastore (README §5, `CLAUDE.md`): follow the stand-in pattern
`test_gauss_order_key.py` and `test_numeric_break_point_key.py` already use. **Confirm both tests
fail at `90d0114` and say so in the log** — a test that passes before the fix is not testing the
fix.

## 7. Acceptance

1. The §2 invariant holds on both paths for all four orders, proved by the §6 tests.
2. `build()` still filters on the current module constant for all four orders, and
   `test_gauss_order_key.py`'s "filtered on, not merely selected" assertions still pass.
3. No schema change: the six factories' columns read exactly as they do at `90d0114`, `GkSource`
   included, `break_point_kind` included.
4. `config/defaults.py` byte-identical; all four orders still 4.
5. All six scripts of §4 compile.
6. `main.py` is **unchanged** — zero lines.
7. `ComputeTargets` **must not fall below 508** and should rise; `CosmologyModels` **39**. Both OK.
   `test_numeric_break_point_key.py` passes **unedited**.
8. The three published source-grid digests are unmoved.
9. `black --check` clean on every `.py` in the diff.
10. Board row 05b, item **T15**, §3/§4, and `docs/OPEN_ISSUES.md` with its count and date
    corrected — all in the **same commit**.
    `[05-rho-gauss-order-reaches-the-residual-table-through-a-default-argument]` closes, and the
    rehydration half of the defect — which that issue does **not** name, because prompt 05 did not
    find it — is recorded as closed with it.

## 8. Stop conditions

Stop and report rather than working around any of these:

- **Any computed value moves**, or a source-grid digest moves.
- **You cannot satisfy §2 without changing what `build()` filters on**, or without a schema change.
  Either is a charter question: prompt 05 settled the schema under D3 and this prompt does not
  reopen it.
- **You find yourself editing** `config/defaults.py`, `main.py`, a numeric target,
  `wavenumber_exit_time`, `QuadSourceIntegral`, `test_numeric_break_point_key.py`, or more than the
  `WKB_phase_function` plumbing §3 describes.
- **Your design requires editing the five doc-script call sites of §4.**
- **You conclude the defect is not real** — that the property re-reading the module is correct
  because the filter guarantees agreement. Say so and stop; do not implement a no-op. That argument
  is exactly what this prompt exists to reject, and if you believe it the user needs to hear it
  rather than receive an empty commit.

## 9. The log

`logs/05b-make-the-recorded-order-the-order-used.md`, on the campaign's template, classifying every
deviation. Beyond the template:

- **The mechanism**, and specifically how the stored order reaches a rehydrated object given that
  `_metadata` is `None` on that path.
- **Your decision on the `order=` default** (§2's consequence), and the §4 constraint's part in it.
- **Confirmation that both §6 tests fail at `90d0114`**, with the failure messages.
- **The argument that no computed value moved**, and how you checked it rather than assumed it.
- **"State handed to the next prompt"** — prompt 05a is next, and inherits log 05's list of which
  `object_get` sites still pass a tolerance, which this prompt does not change. Say explicitly that
  `main.py` is untouched here, so that list still stands as 05 left it.
