# Prompt 20 — Put the numeric break-point policy in the datastore lookup key

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Covers:** §3 `[18-numeric-solver-not-in-lookup-key]`
**Review sections:** none — this is a datastore-keying prompt, not a numerics one. Read instead
`docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §9.4 and §10.5, which measure how far the two
policies move the stored answer, and `logs/19-per-sector-break-point-policy.md`, "State handed to
the next prompt".
**Design facts:** README §2 (h) — the numeric region is sound and its diagnostic is a live warning.
Nothing in §2 is touched by this prompt: **no computed value may move.**
**Depends on:** 19 (which made the policy a per-sector choice), 18, 12.
**Recommended model:** Opus — production datastore code, five factories audited, and a schema
change with no migration.
**Files you may touch:** `Datastore/SQL/ObjectFactories/GkNumericIntegration.py` and
`TkNumericIntegration.py`; `ComputeTargets/GkNumericIntegration.py` and `TkNumericIntegration.py`
(the policy constant and its plumbing, *not* the integration); `main.py`, and **only** the
`object_get` sites for the two numeric targets; `ComputeTargets/tests/test_numeric_break_points.py`
and/or a new test module; plus the log, the status board and `docs/OPEN_ISSUES.md`.
**Do not touch:** `Quadrature/integrators/numeric_with_phase_cut.py` — **the integrator does not
change in this prompt**; its `break_point_kind` argument, its default, and every number it produces
stay exactly as prompt 19 left them. `config/defaults.py`, the `tolerance` objects and every
tolerance value. `CosmologyModels/`. The `IntegrationSolver` table, the `solvers` menu in
`main.py`, and every `solver_serial` column — see §1, they are explicitly *not* the fix.
`BackgroundModel`, `GkWKBIntegration` and `TkWKBIntegration`'s factories: §5 asks you to **audit
and report** on them, not to change them. Everything README §5 rule 8 lists.

Read first: README §2 (h), §5, §6; `IMPLEMENTATION_STATE.md`'s
`[18-numeric-solver-not-in-lookup-key]` entry in §3; `logs/19-per-sector-break-point-policy.md`,
"State handed to the next prompt"; `docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md` §9.4 and §10.5.

---

## 1. What is wrong, and what the fix is *not*

Since prompt 19 the numeric integrator takes a `break_point_kind`, and the two sectors pass
different values: `TkNumericIntegration` asks for `BREAK_POINT_ALL`, `GkNumericIntegration` for
`BREAK_POINT_DISCONTINUITY`. That argument is a genuine degree of freedom — it changes the
integration, and on `QCDModel` it moves the stored transfer function by up to
$1.61\times10^{-4}$ of the envelope (§10.5), on top of the $2.82\times10^{-4}$ prompt 18 already
moved it (§9.4). **It is in neither numeric lookup key.**
`Datastore/SQL/ObjectFactories/GkNumericIntegration.py:221-227` and
`TkNumericIntegration.py:225-231` filter on `validated`, `wavenumber_exit_serial`, `model_serial`,
`atol_serial` and `rtol_serial` (plus the source/init redshift when supplied) and on nothing else.
So a row computed under one policy is indistinguishable by key from a row computed under the other,
and a pipeline run that finds the older row takes it.

**The obvious fix is the wrong one, and this section exists so that you do not implement it.**
The issue was opened against the *solver*: `solver_serial` is stored by both numeric factories and
matched by neither. Adding it to the two queries would be a no-op, because the value cannot vary:

- `numeric_with_phase_cut` hard-codes `method="DOP853"` at lines 358 and 658, takes no method or
  solver argument, and returns the constant string `"solve_ivp+DOP853-stepping0"` (line 823).
- `main.py:3033`'s `solvers` dict is never indexed anywhere in `main.py`. It is passed whole to six
  `object_get` sites, and its only consumer is `self._solver_labels[data["solver_label"]]` in the
  compute target, *after* the integration has run. `RK45`, `Radau`, `BDF` and `LSODA` are
  registered in the datastore and are unreachable.
- So every numeric row ever stored points at one and the same `IntegrationSolver` serial.
  Filtering on it would exclude nothing while making the query read as though it were sound.

`IntegrationSolver` cannot be made to carry the policy cheaply either: its own lookup is
`label == label AND stepping >= stepping` (`integration_metadata.py:32`), so `stepping` is an
ordered *quality* parameter — which is why it can hold a Gauss order — and a break-point policy is
categorical.

**The user's decision (2026-09-13):** key the configuration and leave the solver as provenance.
`break_point_kind` is configuration in exactly the sense `atol` and `rtol` are, and it belongs in
the key beside them. This follows prompt 12's precedent, which closed the same class of hazard for
the $T_k$ `atol` by giving it its own keyed `tolerance` object rather than by relabelling anything.

## 2. What to build

### 2.1 One source of truth for each sector's policy

Today the policy is written at the call site as a literal argument
(`break_point_kind=BREAK_POINT_ALL` at `ComputeTargets/TkNumericIntegration.py:414`,
`BREAK_POINT_DISCONTINUITY` at `GkNumericIntegration.py:383`). Once it is also a key field it will
be written in at least three places — passed to the integrator, stored, and queried — and three
copies of a constant is three chances to drift.

Give each compute target **one** declaration that all three uses read from, in the shape prompt 06
established for the WKB phase solver (`GkWKBIntegration.PHASE_SOLVER_LABEL_BASE` and friends, at
`ComputeTargets/GkWKBIntegration.py:40-42`). A class constant is the obvious form. The integrator
call site then passes *that*, not a literal, and the factory queries *that*. If you find a shape
you prefer, take it and record the alternatives in the log — but a test must be able to show that
the value passed to `numeric_with_phase_cut` and the value used in the key cannot differ.

### 2.2 The key field itself

Add the policy to both numeric tables and to both `build()` filters.

The value is categorical with two members today, so a plain column on each numeric table is
adequate and is the simplest thing that works; a serial to a small shared table would be more
idiomatic for this codebase but buys nothing while the vocabulary has two entries. **Pick one,
implement it, and give the alternatives and your reason in the log** — this is left to you
deliberately. What is *not* left to you: the field must be filtered on in `build()`, not merely
selected, and it must be `nullable=False`.

Store the vocabulary's own strings (`BREAK_POINT_ALL` / `BREAK_POINT_DISCONTINUITY` from
`CosmologyModels/GenericEOS/GenericEOS.py`, re-exported through `ComputeTargets/BackgroundModel.py`).
Do not invent a second vocabulary, and do not let a temperature, a model or an equation of state
appear in `Datastore/` — the generic property prompts 18 and 19 established holds here too.

### 2.3 Old datastores fail loudly; there is no migration

A datastore built before this commit has no such column, and its rows were computed under a policy
nobody recorded — for `QCDModel` that may be "no split at all" (pre-prompt-18), "jumps only"
(prompt 18) or "jumps and kinks" (prompt 19), and the three differ by up to
$2.8\times10^{-4}$ of the envelope. **There is no defensible default**, so do not supply one.

Follow the precedent this codebase already set for exactly this situation when prompts 03 and 04
added columns to `BackgroundModelValue`: detect the missing column and raise a `RuntimeError`
naming the prompt and saying the datastore must be regenerated
(`Datastore/SQL/ObjectFactories/BackgroundModel.py:300-309` is the model to copy, including its
tone). A silent miss is not good enough here — the whole defect being repaired is a stale row that
looked like a hit.

### 2.4 `main.py`

Change only what the two numeric `object_get` sites need in order to pass the policy. If they need
nothing — because the compute target supplies it from §2.1's constant — then change nothing, and
say so. Every other `main.py` hunk is out of scope (README §5 rule 8, and the campaign's
`main.py` rule).

## 3. What must be true afterwards, and must be shown

1. **No computed value moves.** This is a keying change. Every existing test passes with its
   expectation unchanged, and a numeric run under the new code produces a payload bit-identical to
   one under `HEAD~1` — including the QCD $T_k$ figures prompt 19 measured. If anything moves, you
   have changed the integration, which this prompt forbids.
2. **A lookup under the wrong policy misses and under the right policy hits.** This is the whole
   point of the prompt and it must be demonstrated, not asserted.
3. **The two uses of the constant cannot drift** (§2.1).
4. **An old-schema datastore raises the §2.3 error**, with a message that names this prompt.

## 4. Testing, under the campaign's constraint

README §5 rule 7 forbids tests that need Ray or a datastore, and **no factory in this tree is
tested against a live one** — the nearest precedent, `test_quadsource.py:420`, *mimics* what a
factory does on read. So do not stand up SQLite. Three approaches are available and you may use
any combination:

- **Compile the query without a connection.** SQLAlchemy will build a `select()` and let you
  inspect its `whereclause` with no database behind it. This can show item 2 of §3 directly: the
  compiled filter mentions the policy column, and two different policies compile to two different
  criteria.
- **Read the source with `ast`**, as
  `test_numeric_phase_cut.test_both_integrators_pass_warn_unresolved_osc_False` and
  prompt 19's `test_each_production_call_site_passes_the_kind_its_sector_decided_on` already do
  for the call sites. This is the natural way to show §3 item 3.
- **Mimic the factory's logic** over an in-memory stand-in, in the style of `test_quadsource.py`.

`ComputeTargets/tests/test_main_plumbing.load_main_py_functions` reads `main.py` with `ast` rather
than executing it; use it if you need to assert anything about §2.4.

## 5. The audit the user asked for (report only — change nothing)

The user's scope decision of 2026-09-13 was "all five compute targets". Under the design they then
chose — key the configuration, not the solver — four of the five need no key change today, because
they have no equivalent free parameter. **Confirm or refute that**, and report it; do not act on it.

`BackgroundModel`, `GkWKBIntegration` and `TkWKBIntegration` (which has *two* solver columns,
`phase_solver_serial` and `friction_solver_serial`) all store a solver serial and none of them
filters on it. For each, say in the log: what its lookup key actually is; whether it has any
configuration axis that can vary between runs and is not in that key; and whether its solver is
hard-coded in the same way the numeric one is. If any of them does have such an axis, **open a §3
issue** with the measurement or the reason you could not take one — do not fix it here.

One is already visible and should be recorded whatever you find: the WKB targets take their
initial data from the numeric stop point, so a numeric row that is recomputed under a different
policy silently invalidates the WKB rows built on it, and those are keyed independently. Say
whether that is true, and what it means for prompt 13.

## 6. Verification and acceptance

- The suite passes: `discover -s ComputeTargets/tests -t .`, **328 before, plus your new cases**;
  none removed, none changed in expectation. If an existing expectation moves, stop — see §3
  item 1.
- `black --check` clean on every file touched.
- `git diff HEAD~1 --stat` touches nothing outside the allowed list — in particular **not**
  `Quadrature/`, **not** `config/defaults.py`, **not** `CosmologyModels/`, and no
  `IntegrationSolver` or `solver_serial` code.
- The §3 item 1 bit-identity check is run and quoted, not reasoned about.

## 7. Log and commit

Close `[18-numeric-solver-not-in-lookup-key]` only if §3 items 1–4 all hold; otherwise narrow it
with what you found. Either way update `docs/OPEN_ISSUES.md` in the same commit — and note that
the issue's recorded "next step" is **wrong** and was corrected by the user's decision of
2026-09-13: adding `solver_serial` to the queries is a no-op, for the reasons in §1. Say so in the
entry, so that a later reader does not re-propose it. Update board row 20.

"State handed to the next prompt": what prompt 13 must regenerate and what it may keep, now that
an old-schema datastore fails loudly rather than silently; the shape of the key field and the
constant, since both are public; whether anything in §5 turned up a second instance; and whether
the WKB-rows-consume-numeric-values hazard is real.

Commit subject, or something equally specific:
`Put the numeric break-point policy in the datastore lookup key`.
