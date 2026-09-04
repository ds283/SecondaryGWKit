# Log 07 — `inventory()` on the replicated-table object factories (F2b)

**Prompt:** prompts/backport-modules/07-inventory-replicated-factories.md
**Commit:** *(SHA intentionally not embedded — see `IMPLEMENTATION_STATE.md` §5 note 11)*
**Date:** 2026-09-04
**Result:** COMPLETE

## What shipped

All 13 replicated-table factories now have `@staticmethod def inventory(conn, table, tables, *args,
**kwargs)`, per the prompt's contract. None of them consult `tables` (no cross-table joins are
needed for any of these shapes), but the parameter is accepted on every method as required.

### Group A — scalar value types (8 classes)

- **`redshift.py`** (`sqla_redshift_factory`) — near-verbatim of the prompt's model shape:
  `earliest_timestamp`/`latest_timestamp` via `func.min`/`func.max` on `table.c.timestamp`, `values`
  = all `z`, ordered.
- **`wavenumber.py`** (`sqla_wavenumber_factory`) — accepts an explicit optional `units:
  Optional[UnitsLike] = None` parameter (not routed through generic `*args`, since `read_table`'s own
  `units` parameter on the same class is likewise named and typed explicitly rather than left
  anonymous). Always returns `values` = raw stored `k_inv_Mpc` plus `values_unit: "1/Mpc (comoving)"`.
  When `units` is supplied, additionally returns `values_physical` (each raw value converted via
  `k_inv_Mpc / units.Mpc`, matching `CosmologyConcepts.wavenumber.k`'s own conversion) and
  `values_physical_unit` = `units.system_name`. See "Decisions" below for why both forms are returned
  rather than one or the other.
- **`tolerance.py`** (`sqla_tolerance_factory`) — `values` = `10 ** log10_tol` for every row (i.e. the
  tolerance itself, not its log), since that is the quantity a caller of this table actually asked
  for; the stored `log10_tol` is an implementation detail of how the table avoids float underflow.
- **`store_tag.py`** (`sqla_store_tag_factory`) — `values` = sorted list of tag labels.
- **`version.py`** (`sqla_version_factory`) — registered `"timestamp": False`. Returns `{"values":
  [...]}` only — `earliest_timestamp`/`latest_timestamp` are omitted entirely rather than returned as
  `None`, per the prompt's explicit instruction.
- **`LambdaCDM.py`** (`sqla_LambdaCDM_factory`) — a per-row label plus timestamp range: `values` is a
  list of `{"name", "omega_m", "omega_cc", "h"}` dicts (the parameters that actually distinguish one
  row from another; `f_baryon`/`T_CMB_Kelvin`/`Neff` are stored but omitted from the inventory as
  secondary detail — this table is small, so the omission is for readability, not cost).
- **`QCD_Cosmology.py`** (`sqla_QCDCosmology_factory`) — identical shape to `LambdaCDM`, plus
  `log10_max_z` (the parameter that distinguishes a QCD cosmology from a plain `LambdaCDM` one).
- **`integration_metadata.py`** (`sqla_IntegrationSolver_factory`) — `values` = list of
  `{"label", "stepping"}` dicts, ordered by label then stepping.

### Group B — `wavenumber_exit_time` (1 class)

`wavenumber.py` (`sqla_wavenumber_exit_time_factory`) — no per-row label list (this table can hold
one row per (wavenumber, cosmology, tolerance pair) combination, which is not bounded the way the
Group A configuration tables are). Returns `earliest_timestamp`/`latest_timestamp`, `count` (`SELECT
count(*)`), and `distinct_wavenumbers` (`SELECT count(DISTINCT wavenumber_serial)`).

### Group C — `BackgroundModel` (1 class)

`BackgroundModel.py` (`sqla_BackgroundModelFactory`) — labelled shape:

```python
{
    "validated": {"labels": [...], "earliest_timestamp": ..., "latest_timestamp": ...},
    "unvalidated": {"labels": [...], "earliest_timestamp": ..., "latest_timestamp": ...},
}
```

`labels` is the sorted, deduplicated set of non-null `label` values in each bucket; the timestamp
range is computed per-bucket with `func.min`/`func.max` under the same `WHERE` condition, not by
iterating the label query's own rows. The `"unvalidated"` bucket's condition is `validated == False
OR validated IS NULL` — matching `validate_on_startup`'s own defensive treatment of a legacy `NULL`
in this column (the column is `nullable=False` going forward, but `validate_on_startup` in this same
file still guards against `NULL`, so the inventory does too rather than silently dropping such rows
from both buckets). No `"versions": []` field is included (upstream's unpopulated field, per the
prompt's explicit instruction not to carry it across unless populated).

### Group D — `BackgroundModelValue` (1 class)

`BackgroundModel.py` (`sqla_BackgroundModelValue_factory`) — `{"count": <row count>}` only, via
`select(func.count()).select_from(table)`, no row iteration.

## Deviations from the prompt

### `wavenumber`'s `units` handling — IMPLEMENTATION CHOICE

The prompt explicitly leaves this open ("Your call — but prompt 09 has to format these, so record
the decision clearly"). Considered: (a) raw values only, no `units` parameter at all; (b) accept
`units` and return *only* the converted physical values, dropping the raw ones; (c) accept `units`
optionally and return both raw and physical values side by side. Chose (c): the raw `k_inv_Mpc`
values are always present under `values` (so a caller with no `units` on hand — or prompt 09's report
running against a class it does not construct units for — still gets a complete, well-formed
inventory), and the physical form is added only when `units` is actually supplied, under distinctly
named keys (`values_physical`, `values_physical_unit`) so the two forms are never confused with each
other. This differs from the prompt's alternative of returning *either* raw *or* physical depending on
whether `units` was passed (same key, different meaning depending on caller) — that alternative was
rejected because a shape that changes meaning based on an argument is a sharper edge for prompt 09 to
have to detect than a shape that only ever grows an extra key.

**No merge-policy consequence**: `wavenumber` is replicated, not sharded (`config/sharding.py`), so
`ShardedPool.inventory` returns whichever single shard's dict was picked, unmodified — nothing here
is merged, and this dict's varying key set (with vs. without `units`) is therefore never exercised by
`_merge_queue`'s shape-sniff logic. This is stated explicitly because it is the one class in this
prompt whose returned key set is not fixed across calls.

### `tolerance` reports `tol`, not `log10_tol` — IMPLEMENTATION CHOICE

The table stores `log10_tol` (chosen upstream and in this tree to avoid float underflow for very
small tolerances), but `values` reports `10 ** log10_tol` — the tolerance value itself, which is what
a caller/reader of this inventory actually wants to see, not the storage encoding. No information is
lost (the stored column already keeps full precision; re-deriving `log10_tol` from the reported value
would be exact to the same precision `pow`/`log10` round-trip through at these magnitudes).

### `LambdaCDM`/`QCD_Cosmology` omit some stored columns from `values` — IMPLEMENTATION CHOICE

Both tables store `f_baryon`, `T_CMB_Kelvin`, `Neff` in addition to the columns reported. These were
left out of the per-row dict as secondary detail once `name`/`omega_m`/`omega_cc`/`h` (and
`log10_max_z` for the QCD variant) already distinguish one configuration from another for a human
reading the inventory. Not a cost-discipline concession (`register()` shows these are small,
one-row-per-configuration tables — the audit's own note on this) — purely a readability choice. If
prompt 09 (or a user) wants the omitted columns surfaced, they can be added to the `select()` and the
per-row dict with no shape implications elsewhere, since `LambdaCDM`/`QCD_Cosmology` are also
replicated, not sharded.

### `BackgroundModel`'s unvalidated bucket treats `NULL` as unvalidated — IMPLEMENTATION CHOICE

Not explicitly requested by the prompt's Group C text (which only sketches the two-bucket shape), but
taken directly from this same file's own `validate_on_startup` method (`or_(table.c.validated ==
False, table.c.validated == None)`), on the reasoning that an inventory of "unvalidated" models
should agree with what `validate_on_startup` itself considers unvalidated, rather than silently
omitting a legacy `NULL` row from both buckets.

### No other deviations

Every other class follows Group A/B/C/D's default shape as described in the prompt, with no
structural surprises: every `register()` was read directly from the file before writing the
corresponding `inventory` method, and every column referenced was confirmed present.

## Verification performed

1. **All 11 touched files parse.** `./venv/bin/python3 -m py_compile` on all 11 files — exit 0.
2. **`black --check` clean.** `./venv/bin/python3 -m black` on all 11 files — two files needed one
   reformat pass on first write (`wavenumber.py`, `BackgroundModel.py`, both for a wrapped multi-line
   list comprehension); `black --check` on all 11 afterwards — "11 files would be left unchanged."
3. **`grep -c "def inventory" Datastore/SQL/ObjectFactories/*.py` accounts for exactly 13** across
   the 11 touched files (`BackgroundModel.py` and `wavenumber.py` each contribute 2, matching their
   two factory classes apiece; the other 9 files contribute 1 each) — confirmed by summing the
   per-file counts.
4. **`grep -B1 "def inventory" ... | grep -c staticmethod` = 13** — every `inventory` method is
   immediately preceded by `@staticmethod`, matching the count in check 3 exactly (no missing
   decorator).
5. **`grep -rn "def inventory(self" ...` = no matches** — the upstream-copy failure mode the prompt
   warns about is confirmed absent.
6. **Every method run against a real (in-memory SQLite) database, not just reasoned about** —
   throwaway harness `scratchpad/verify_inventory_07.py` (not committed), run under
   `./venv/bin/python3`, hand-builds a minimal schema matching each factory's `register()` (serial +
   the columns each method actually selects) and calls every one of the 13 methods twice: once
   against empty tables, once after inserting a small number of rows.
   - **Empty-table results, all sensible**: every scalar-value class returned `earliest_timestamp:
     None, latest_timestamp: None, values: []`; `version` returned `{"values": []}` with no timestamp
     keys at all; `wavenumber_exit_time` returned `count: 0, distinct_wavenumbers: 0` alongside `None`
     timestamps; `BackgroundModel` returned both buckets with empty `labels` lists and `None`
     timestamps; `BackgroundModelValue` returned `{"count": 0}`. No method raised, and nothing
     returned a Python exception disguised as a value (e.g. a `scalar()` call against zero rows
     correctly returns `None`, not an error, for every aggregate used).
   - **Populated-table results, spot-checked against the inserted data**: `redshift` returned `values:
     [1.0, 2.0]` for two inserted rows, in ascending order; `wavenumber` returned raw `values: [10.0]`
     and, when a fake `units` object (`Mpc = 2.0`) was passed, additionally `values_physical: [5.0]`
     (`10.0 / 2.0`) and `values_physical_unit: "fake-units"` — confirming the conversion arithmetic
     and the key-set growth described above; `tolerance` returned `values: [1e-06]` for an inserted
     `log10_tol = -6.0` row (`10 ** -6 == 1e-06`); `store_tag` returned `values: ['tag_a', 'tag_b']`
     for rows inserted in the reverse order, confirming the `ORDER BY label` sort; `version` returned
     `{"values": ["v1"]}`; `LambdaCDM` and `IntegrationSolver` and `GkSourcePolicy` each returned a
     single correctly-populated dict matching the inserted row; `BackgroundModel` returned `labels:
     ["model_a"]` under `"validated"` and `labels: ["model_b"]` under `"unvalidated"` for one row of
     each, with correct non-`None` timestamps in both buckets; `BackgroundModelValue` returned
     `{"count": 3}` for three inserted rows.
   - All results printed and inspected directly; script output is quoted inline above. Scratch script
     left in the scratchpad directory (not committed), per the campaign's established convention (see
     prompt 06's log for the same pattern).
7. **Not run**: `Datastore.inventory(...)` / `ShardedPool.inventory(...)` dispatch through the real
   `Datastore`/`ShardedPool` classes onto these factory methods. Prompt 06's log already exercised
   that dispatch machinery directly (with no factory yet implementing `inventory`, so only the
   error-path branches were reachable at that time); this prompt's job is the factory methods
   themselves, and check 6 exercises them directly against the same kind of schema the real dispatch
   would hand them (`conn`, `table`, `tables` — here `tables={}` since none of these 13 methods
   consult it). Wiring a full `Datastore` instance with all 13 factories registered and no other
   factory would require constructing significant unrelated schema (foreign keys into tables this
   prompt does not touch) for no additional confidence beyond what check 6 already establishes; left
   for prompt 09 (the first real end-to-end exercise, per the README's own framing) or prompt 10.

## Observations not acted on

- **`LambdaCDM`/`QCD_Cosmology` inventories omit `f_baryon`, `T_CMB_Kelvin`, `Neff`** from the
  per-row `values` dicts (see "Deviations" above) — a readability choice, not a cost one. If a future
  caller wants the full parameter set surfaced, it is a one-line addition to each `select()` and dict
  literal.
- **No cross-check that every `BackgroundModel` row has a non-null `label`.** The `label` column is
  `nullable=True` in `register()`, and rows with a null label are silently excluded from both buckets'
  `labels` lists (via `.where(table.c.label.isnot(None))`) rather than surfaced as, say, an
  `"unlabelled_count"`. This matches the Group C sketch's `"labels": [...]` field literally, and an
  unlabelled `BackgroundModel` is presumably an edge case (every `build()`/`store()` call site passes
  a `label` in practice), but it is a silent omission worth knowing about if prompt 09's report looks
  like it is missing models a user expects to see.
- **`wavenumber_exit_time`'s `distinct_wavenumbers` count was judged "cheap" without measuring an
  actual populated table** (none exists in this tree — see `IMPLEMENTATION_STATE.md` §5 note 1). A
  `COUNT(DISTINCT ...)` over an indexed foreign-key column is a standard cheap aggregate for any
  database size in practice, but this is inference from the schema, not from a measured large-table
  run.

## State handed to the next prompt

- **The two field-name contracts prompt 08 must match exactly** (per the prompt's own emphasis that
  this log is the authority prompt 08 depends on):
  - **Labelled compute-target shape (Group C)**: top-level keys are the label strings themselves
    (here `"validated"` and `"unvalidated"`), each mapping to `{"labels": [...], "earliest_timestamp":
    ..., "latest_timestamp": ...}`. Prompt 08's sharded compute targets (`TkNumericIntegration`,
    `TkWKBIntegration`, `QuadSource`, `GkNumericIntegration`, `GkWKBIntegration`, `GkSource`) must use
    these same three field names inside each label's sub-dict, since `_merge_queue` is applied once
    per label to whatever fields that label's dict contains (per prompt 06's log) — if prompt 08 adds
    extra fields inside a label's dict (e.g. a compute-time statistic), it must assign each of them a
    merge policy consistent with `ShardedPool._merge_queue`'s vocabulary (`"extend"` for `labels`,
    `"earliest"`/`"latest"` for the two timestamp fields — a numeric field would need `"sum"`, `"min"`,
    or `"max"`).
  - **Count field name (Group D)**: `"count"`, a bare top-level key mapping to an integer. Prompt 08's
    high-volume sharded value tables (`TkNumericValue`, `TkWKBValue`, `QuadSourceValue`,
    `GkNumericValue`, `GkWKBValue`, `GkSourceValue` — the six tables `IMPLEMENTATION_STATE.md` §5 note
    8 identifies as `"timestamp": False`) should return `{"count": <row count>}` in the same shape,
    since `inventory_config` for each will need exactly one merge policy entry, `"count": "sum"`.
- **`wavenumber` takes `units` as an explicit optional third positional parameter** (`units:
  Optional[UnitsLike] = None`, following `conn, table, tables`), not through generic `*args` — and
  returns raw values unconditionally plus physical values additionally when `units` is supplied. This
  is the one class in this sub-campaign whose top-level key set is caller-dependent; harmless here
  because `wavenumber` is replicated and never merged, but a sharded class with a similarly
  caller-dependent key set would break `ShardedPool.inventory`'s label/shape sniff (per prompt 06's
  log, which sniffs shape from `data_queue[0]` alone) — prompt 08 should not replicate this pattern for
  any sharded class.
- **No sharded-table factory or `config/sharding.py` entry was touched** — prompt 08's scope is
  untouched by this commit, and `inventory_config` remains exactly as prompt 06 left it (unpopulated,
  since no sharded class implements `inventory` yet).
- `IMPLEMENTATION_STATE.md` §5 note 11's no-self-referential-SHA convention followed: no SHA embedded
  in this log's header or in the status board's prompt-07 row.
