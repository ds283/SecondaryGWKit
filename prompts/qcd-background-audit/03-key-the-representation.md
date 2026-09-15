# Prompt 03 — Put the `T(z)` representation in the datastore lookup key

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Design facts:** README §2 (d) · **Decision:** README §7 **D1** — the prompt proposes, the
orchestrator reports, the user may overrule
**Depends on:** 02 (for a quiet tree). **Blocks 04**: no number may move before this lands.
**Recommended model:** **Opus** — a schema change whose whole justification is a failure mode
nobody can see.

**Files you may touch:** `CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py` (the constant and its
comment **only**), `Datastore/SQL/ObjectFactories/QCD_Cosmology.py`,
`ComputeTargets/tests/` — one new module, `test_cosmology_representation_key.py` — plus this
campaign's log and board and `docs/OPEN_ISSUES.md`.

**Do not touch:** `_solve_T_z`, `_build_T_z_spline`, `integration_break_points`, `rho`, `Hubble`,
`wBackground`, `wPerturbations`, or any other computation. **This prompt moves no number.** Nor any
other `Datastore/SQL/ObjectFactories/` file, `ComputeTargets/`, `Quadrature/`, `main.py`.

**Read first:** `Datastore/SQL/ObjectFactories/QCD_Cosmology.py` in full — `register()`'s column
list and `build()`'s `select ... filter(...)`; `Datastore/SQL/ObjectFactories/BackgroundModel.py`'s
**module docstring**, which is the precedent for the "regenerate; there is no migration" shape;
`ComputeTargets/TkNumericIntegration.py:98-118` (`BREAK_POINT_KIND`), which is the precedent for a
policy constant that is declared once and keyed; `ComputeTargets/tests/test_numeric_break_point_key.py`,
which is the precedent for the test; board entry `[20-wkb-rows-consume-numeric-initial-data]`.

---

## 1. What is wrong

`sqla_QCDCosmology_factory.build()` looks a cosmology up by seven parameter values and
`log10_max_z`. **Nothing in that key mentions how `T(z)` is represented.** So when prompts 04, 05
and 06 change the representation:

- the same cosmology row is returned, with the same serial;
- every `BackgroundModel` keyed on that serial is found and deserialised;
- its `tau`, `cs_tau` and `friction_F` limbs are the **old** background's;
- and every `Gk`/`Tk` numeric and WKB row built on it is served, silently, against a background
  that no longer exists in the code.

There is no exception, no warning and no column that differs. This is
`[20-wkb-rows-consume-numeric-initial-data]` one level up and considerably worse, because there the
mismatch at least moves $z_{\rm init}$ by 4.59e5 and the lookup misses; here nothing moves.

It has to land **before** prompt 04, because prompt 04 is the first prompt that changes a number.

## 2. The change

1. **`T_Z_REPRESENTATION_VERSION`, a class constant on `LambdaCDM_GenericEOS`**, initial value
   **1**, inherited by `QCD_Cosmology`. Follow `TkNumericIntegration.BREAK_POINT_KIND`'s shape and
   the style of its comment block: a single declaration, readable from the class without an
   instance, with a comment saying

   - what it identifies — **everything about the background that the parameter key does not
     otherwise capture**: the node solve's tolerance, the quantity splined, the node count, the
     spline order, the segmentation, and the break-point set;
   - that **every prompt in `prompts/qcd-background-audit/` that changes any of those bumps it**,
     and which ones did, with the version each landed at (a short table the later prompts append
     to);
   - that a stale row is otherwise undetectable, which is the whole reason it exists.

2. **A `T_z_representation` integer column** on the QCD cosmology table, in `register()`, and an
   equality filter on it in `build()`'s `select`. Not a float: this is an identifier, and
   `DEFAULT_FLOAT_PRECISION` has no business near it. Insert it in `insert_data` alongside the
   others.

3. **A datastore written before this column exists must say so.** Follow
   `sqla_BackgroundModel_factory`'s precedent: `build()` raises with a message that names this
   campaign, says there is no migration, and says the QCD half of the datastore must be
   regenerated — rather than failing with a SQLAlchemy attribute error a reader cannot interpret.
   Decide by inspection whether the missing column actually reaches `build()` as an exception or
   as a silently absent attribute, and handle what actually happens; say which in the log.

4. **Nothing else.** In particular do **not** add the version to `name`, to a tag, to the
   `BackgroundModel` key, or to any other factory. The cosmology serial is what everything
   downstream hangs from; keying it is sufficient and keying anything else is scope creep.

**If you conclude that option (ii) of README §7 D1 — no schema change, a documented regeneration
requirement — is the better shape, stop and report rather than taking it.** The campaign's default
is the column and the user reserves this decision.

## 3. Tests

New module `ComputeTargets/tests/test_cosmology_representation_key.py`, on the pattern of
`test_numeric_break_point_key.py`, and **without Ray**. It needs a datastore, which
`test_numeric_break_point_key.py` shows how to stand up in-memory; if it cannot be done without
one, use `ast` to assert the *structure* instead — the column is in `register()`, the filter is in
`build()`, the constant is read rather than a literal — and say in the log which you did and why.

1. **Two versions, two serials.** The same parameter block with `T_Z_REPRESENTATION_VERSION` 1 and
   2 produces two distinct rows; the same version twice reuses one.
2. **The filter reads the constant, not a literal.** `ast`, following
   `test_numeric_break_point_key.test_each_production_call_site_passes_the_kind_its_sector_decided_on`.
   A literal `1` in `build()` would silently stop tracking the constant the moment prompt 04 bumps
   it, and that is the exact failure this prompt exists to prevent.
3. **A table without the column raises the campaign's message**, not an opaque error.
4. **`LambdaCDM` is untouched.** `Datastore/SQL/ObjectFactories/LambdaCDM.py` has no such column and
   must not acquire one; assert its `register()` column list is unchanged.

## 4. Verification and acceptance

- All three suites pass; counts quoted, `ComputeTargets` rises by the new cases.
- **No number moves.** Demonstrate it: `T_photon`, `Hubble` and `rho` at a dense set of $z$ are
  **bit-identical** to `HEAD~1` on `QCD_Cosmology`, `LambdaCDM` and a `LambdaCDM_GenericEOS` built
  on `PureRadiationEOS`. Quote the comparison, do not assert it in prose.
- `docs/qcd-background-audit/generate_qcd_references.py --dry-run` still reports **no change**.
- `black` clean.

## 5. Log and commit

Log to `logs/03-key-the-representation.md` per README §5.1. It must state plainly, in "What
shipped": **the value of `T_Z_REPRESENTATION_VERSION` (1), that prompts 04, 05, 06 and 07 each bump
it, and exactly what a pre-campaign datastore now does** — which error, from which call, with which
message. Update this campaign's board §1's version table (the row for prompt 03).

Commit subject, or something equally specific:
`Key the QCD cosmology on its temperature representation`
