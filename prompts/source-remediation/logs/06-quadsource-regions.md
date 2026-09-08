# Log 06 — Restrict `QuadSource` to the both-numeric region (A3, A2 part 2)

**Prompt:** prompts/source-remediation/06-quadsource-regions.md
**Commit:** *(this commit)* — "Restrict QuadSource to the region where both T_k are numeric"
(the SHA cannot be embedded in the commit that contains this file; see log 01 deviation 4)
**Model:** Claude Opus 5
**Date:** 2026-09-08
**Result:** COMPLETE WITH DEVIATIONS

## What shipped

### `ComputeTargets/QuadSource.py`

- **`QuadSourceFunctions`** (`:20` before → `:28` after) gains a second field:
  `namedtuple("QuadSourceFunctions", ["source", "numeric_region"])`, with `numeric_region`
  the `(z_max, z_min)` pair the spline is valid over. Nothing constructs this namedtuple
  positionally outside `_create_functions`; `extract_*.py` and
  `docs/spec-code-audit/scripts/QI_03_measure.py` already fill unknown fields generically.
- **new module function `numeric_crossover_z(Tk) -> Optional[float]`** (`:31-53`). Returns
  `float(Tk.z_exit) - float(Tk.stop_deltaz_subh)`, i.e. exactly what `main.py:694` uses as
  the `TkWKBIntegration` starting redshift and what `TkSourceFunctions` validates against
  `TkWKBIntegration.z_init`. Returns `None` when the integration was not run in `mode="stop"`
  (the `stop_deltaz_subh` property raises `RuntimeError` in that case; caught).
- **new private helper `_z_above(z, limit)`** (`:56-61`): `z >= limit` with a relative
  tolerance of `DEFAULT_FLOAT_PRECISION`, so a grid point that *is* the limit is not dropped.
- **`compute_quad_source`** (`:55-187` before → `:96-273` after) rewritten in three ways:
  1. `Tq_zsample = Tq.z_sample` / `Tr_zsample = Tr.z_sample` (`:63-64`) replaced by
     `store_id → value` dicts built from `Tq.values` / `Tr.values`. The value list, not
     `z_sample`, is the authoritative record of coverage: in `mode="stop"` a
     `TkNumericIntegration` holds fewer values than its `z_sample`
     (`TkNumericIntegration.py:424-426`).
  2. The grid is truncated. `crossover_z_q`/`crossover_z_r` come from `numeric_crossover_z`;
     `z_floor = max(crossover_z_q, crossover_z_r, min sampled z of Tq, min sampled z of Tr)`
     (the last two terms are deviation 1); `region = [z for z in z_sample if
     _z_above(z.z, z_floor)]`. An empty region raises `RuntimeError` quoting all four inputs.
     The loop runs over `region`, not `range(len(z_sample))`.
  3. The index-walking alignment (`:82-102`, the `q_idx`/`r_idx` pair and the `q_idx > 0`
     test) is gone. A local `_lookup(z, label, value_map, grid_z_max)` returns the stored
     value, or `None` if `z` is above the start of that factor's grid — in which case the
     caller substitutes the exact super-horizon `T=1, T'=0`, unchanged from `:116-121` — or
     raises `RuntimeError` naming the factor (`"Tq"` / `"Tr"`), the redshift, its `store_id`
     and the factor's `z_max` if the redshift is inside the sampled range but absent from it.
  The returned dict gains `"z_store_ids"`, `"crossover_z_q"` and `"crossover_z_r"`.
- **`QuadSource.__init__`** initialises `self._crossover_z_q = self._crossover_z_r = None`.
- **new properties** `crossover_z_q`, `crossover_z_r` (both `Optional[float]`, `None` after a
  datastore round-trip) and `numeric_region` (`Optional[Tuple[float, float]]`, derived from
  `self._z_sample`, so it *is* recoverable after a round-trip).
- **`_create_functions`** (`:294-309` before): unchanged except that the spline bounds are
  hoisted into `z_max`/`z_min` locals and also passed as `numeric_region`. `ZSplineWrapper`
  and prompt 02's `"quadratic source"` label are untouched.
- **`store()`**: sets `_crossover_z_q`/`_crossover_z_r` from the payload, then rebuilds
  `self._z_sample` as a `redshift_array` over exactly the returned `z_store_ids` (raising if
  one is not part of the grid the object was constructed with), checks
  `len(source) == len(region_z)`, and builds each `QuadSourceValue` against `region_z[i]`
  rather than `self._z_sample[i]`. So `z_sample`, `values`, `numeric_region` and the spline
  all describe the same range both before and after persistence.

### `main.py`

- A seven-line comment at the `pool.object_get("QuadSource", ...)` site (`:905` before →
  `:905-911` after) recording that the full source grid is still the right thing to pass
  because `QuadSource` truncates internally, and that the stored values therefore end at the
  larger of the two `Tk` hand-over redshifts. No code change; `z_sample=z_source_sample` and
  all six tags are as before.

### `ComputeTargets/tests/test_quadsource.py` (new, 10 tests, 1.3 s)

Offline stand-ins (`FakeTk`, `FakeTkValue`, `FakeModelProxy`, `FakeWavenumberExit`,
`source_grid`) shaped like the real objects, with exact constant-$w$ transfer functions from
`ComputeTargets/analytic_Tk.py` on an exact constant-$w$ background, following
`docs/spec-code-audit/scripts/QS_02_deriv_and_f.py` and `QS_03_spline_error.py`. Real
`redshift`/`redshift_array` objects are used rather than the audit script's `Z` stand-in, so
that `z_sample.max`/`.min` and the descending-order invariant are the production ones.
`FakeTk` deliberately keeps a `z_sample` and `__getitem__` so the *pre-commit* loop can be
driven from the same fixture.

### No changes

`Datastore/SQL/ObjectFactories/QuadSource.py`, `QuadSourceIntegral.py`, `source_function`,
`QuadSourceValue`, the `QuadSource`/`QuadSourceValue` schemas, and
`docs/spec-code-audit/scripts/`.

## Deviations from the prompt

### 1. The region floor is clamped to the sampled coverage as well as the hand-over — STRUCTURALLY REQUIRED

§2.1 says "Define the both-numeric region as $z' \ge \max(z^{\rm X}_q, z^{\rm X}_r)$". That
alone is not safe, because the hand-over redshift can fall *below* a factor's last stored
sample. `main.py:505-507` truncates the `Tk` grid at `0.85 · z_exit_subh_e6` and
`TkNumericIntegration.py:130-131` sets the `mode="stop"` search window to
`[z_exit_subh_e3, z_exit_subh_e6]`. `numeric_with_phase_cut.py:113-117` terminates the
integration at the *bottom* of that window, so the last stored sample is the smallest grid
point at or above `z_exit_subh_e6`; but `find_phase_minimum` may return a root anywhere in
the window, including below that grid point. In that case a grid point satisfying
`z >= max(z^X_q, z^X_r)` can be absent from a factor's values, and `_lookup` would raise —
turning a benign one-grid-step effect into a run-blocking error, which is the very failure
mode this prompt exists to remove.

Shipped: `z_floor = max(crossover_z_q, crossover_z_r, Tq_z_min, Tr_z_min)` where `Tq_z_min`
is the smallest sampled redshift of `Tq`. In production the two agree (the phase minimum
found by stepping down from `z_exit_subh_e3` is normally reached well above
`z_exit_subh_e6`), and the tests assert equality in that case; the clamp only ever *shortens*
the region, never lengthens it, so it cannot admit a redshift where a factor is already
oscillatory. `test_region_is_clamped_to_the_sampled_coverage` pins the behaviour with a
fixture whose nominal hand-over is two grid steps below the last sample.

The interior-gap `RuntimeError` §2.1 asks to keep is kept and is now *stronger* than before:
the old code only raised for a mismatch once `q_idx > 0`, and could not distinguish a hole
from a trailing shortfall at all.

### 2. `store()` truncates `self._z_sample`; the compute task returns the redshifts it used — IMPLEMENTATION CHOICE

§2.1 asks only that the object "record ... the region actually used". The prompt does not say
what happens to `QuadSource.z_sample`, which `__init__` receives as the *full* grid from
`main.py` and which `_create_functions` reads for the spline bounds (`:305-306`) and `store()`
read for each value's redshift (`:388`). Leaving it alone would have produced an object whose
`z_sample` (full grid) disagreed with its `values` (truncated) and whose spline claimed a
range extending below its data — and, after a datastore round-trip, the factory rebuilds
`z_sample` from the value rows, so the in-memory and reloaded objects would have disagreed
too.

Alternatives considered: (a) truncate in `compute()` before dispatch, duplicating the region
calculation on the driver; (b) return only a sample count and take a prefix of `z_sample`
(valid, since `redshift_array` is descending and the truncation is at the low-$z$ end, and
this is what `TkNumericIntegration.store()` does at `:424-426`); (c) return the `store_id`
list and rebuild. Shipped (c): the region is computed in exactly one place, the mapping from
value to redshift is explicit rather than positional, and a mismatch raises instead of
silently mis-labelling a value. Cost: three extra keys in the task's return dict and a
`len(region)`-sized list of integers over the Ray boundary.

### 3. `numeric_region` is derived, not stored; `crossover_z_q`/`_r` are `None` after a round-trip — IMPLEMENTATION CHOICE (anticipated by the prompt)

§2.1 says "If you find the object cannot recover them after a datastore round-trip without
the `Tk` objects, expose them as `None` in that case and document it". That is the case for
the two hand-over redshifts — they are floats on the `Tk` objects, not on any `QuadSource`
column, and §2 forbids a schema change. But `numeric_region` *is* recoverable, because the
factory rebuilds `z_sample` from the stored value rows
(`Datastore/SQL/ObjectFactories/QuadSource.py:236-261`), so it is defined as
`(z_sample.max.z, z_sample.min.z)` and works on both paths. Prompt 08 can therefore always
read `numeric_region`, and should call `numeric_crossover_z(Tq)` itself (it holds the `Tk`
objects) rather than relying on `crossover_z_q`.

### 4. `numeric_crossover_z` is a module function, not a method — IMPLEMENTATION CHOICE

The hand-over definition is needed by the Ray task (which has no `QuadSource` instance), by
the tests, and by prompt 08. A module-level function is importable from all three and keeps
one definition of the formula. `TkSourceFunctions` computes the same quantity as
`float(Tk_WKB.z_init)` and validates it against this formula
(`TkSourceFunctions.py:173-181`); the two are not shared, because `QuadSource` never sees a
`TkWKBIntegration`. Alternative considered: put the helper in `TkNumericIntegration` as a
property. Not done — that file is not in this prompt's allowed set.

### 5. Two tests beyond §3's list — IMPLEMENTATION CHOICE

§3 lists four tests. Shipped ten, the extras being: `test_crossover_helper`,
`test_source_function_is_symmetric` (QS-9, one line), `test_region_is_clamped_to_the_sampled_coverage`
(deviation 1), `test_store_truncates_z_sample_and_the_spline_range` and
`test_datastore_round_trip_shape` (deviations 2 and 3 — the parts of the change §3's four
tests do not reach), and `test_A3_regression_pre_commit_code_would_have_raised`, which
re-executes the old alignment loop inline so that the reason the loop changed stays in the
test suite rather than only in this log.

`test_store_truncates_z_sample_and_the_spline_range` drives the real `QuadSource.store()`
with `ComputeTargets.QuadSource.ray` patched to a `MagicMock` whose `wait`/`get` return the
payload computed directly from the task body. No Ray runtime is initialised, so README §5
item 7 is respected; the alternative was to leave `store()` untested, since it is the only
place the truncation of `z_sample` happens.

## Verification performed

**Unit tests.** `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .`
from the repository root: **27 tests, 1.29 s, OK** (10 new, 17 pre-existing from prompts 03
and 05). `./venv/bin/black` reports all three touched files formatted.

**§3.1 — A3 regression (ran, passed).** Full source grid of 300 points (z = 1e5 → 1e2, 100
per log10 z); `Tq` covering grid indices 60–240, `Tr` covering 20–180, both with
`stop_deltaz_subh` set so that `z_exit - stop_deltaz_subh` lands exactly on the last covered
grid point. `compute_quad_source` completes and returns **181** values (indices 0–180); the
last redshift is `grid[180]`, which is `max(z^X_q, z^X_r) = z^X_r = 1563.08`; the first 20
values (above the start of both numeric grids) equal
`source_function(1, 1, 0, 0, z, 1/3)` to $10^{-14}$ relative, and between the two grid starts
only $q$ is on its default. `crossover_z_q` and `crossover_z_r` match `grid[240].z` and
`grid[180].z` to 10 decimal places.

**§3.1 — the same inputs on the pre-commit code (ran).** Two independent checks.
(i) In the test suite, `test_A3_regression_pre_commit_code_would_have_raised` replays the old
`:82-102` index walk on the fixture above and asserts `IndexError`.
(ii) Out of tree, `git show HEAD:ComputeTargets/QuadSource.py` loaded as a module and driven
with the audit's own 12-point fixture (`Tk` grid missing 2 leading and 4 trailing samples):

```
old code RAISED IndexError: list index out of range
  File ".../old_quadsource.py", line 87, in compute_quad_source
    Tq_z: redshift = Tq_zsample[q_idx]
```

reproducing `QS-6` verbatim.

**§4 — `QS_04_coverage_and_jacobian.py` part (a) (ran, on an adapted copy).** The audit script
as committed now fails with `AttributeError: 'MockTk' object has no attribute 'values'`,
because its stand-in predates the new protocol; §4 of the prompt allows adapting a copy. The
adapted copy (same 12 redshifts, same "drop 2 leading and 4 trailing" `Tk` grid, plus
`z_exit`/`stop_deltaz_subh` giving a hand-over at `grid[7]`) prints:

```
(a) len(z_sample) = 12   len(Tq.values) = 6
    returned 8 source values
    region ends at store_id 107 -> z = 31.622776601683793
    crossover_z_q = 31.62277660168379   crossover_z_r = 31.62277660168379
    first three values (T=1,T'=0 default): [1.5, 1.5, 1.5]
```

No `IndexError`; the region ends at the hand-over; and `1.5` is exactly the audit's own
control value for the super-horizon default at $w=1/3$
($f = (5+3w)/(3(1+w)) = 6/4$). The audit script itself was **not** modified — it is the
record of the pre-fix state (README §5 item 5).

**§3.2 — interior gap (ran, passed).** Deleting one value from the middle of `Tq`'s coverage
inside the region raises

```
RuntimeError: QuadSource: redshift z=6251.5344 (store_id=1120) is missing from the sampled
values of Tq, but lies inside its sampled range (z_max=25003.069). The source and
transfer-function grids are expected to be nested, so this is a gap in the Tq data.
```

and the same hole in `Tr` names `Tr`.

**§3.3 — kernel unchanged (ran, passed).** `source_function` against the independent
transcription of spec 03 R22 copied from `QS_01_sympy_f.py`/`QS_02_deriv_and_f.py`, over
$w \in \{1/3, 0.1, 0.2, 0.5\}$, four $(q,r)$ pairs and six redshifts (96 points): **worst
relative difference 2.802e-16**, i.e. float round-off, comfortably inside the $10^{-14}$ the
prompt asks for. `source == undiff + diff` to $10^{-14}$ at every point, and
$f(q,r) = f(r,q)$ bit-for-bit.

**§3.4 — spline residual inside the both-numeric region (ran; reported, not asserted).**
Exact analytic radiation source, $q = r = k$, 100 samples per log10 z, region running from 5
e-folds super-horizon down to the stated hand-over, 39 interior points per node interval,
error normalised to the local oscillation envelope (the `QS_03_spline_error.py` measure). In
radiation with $a_0$ absorbed, $k/(aH) = k/(1+z)$ and $x = k c_s a_0\eta = k/(\sqrt3(1+z))$,
so $x = e^N/\sqrt3$ at $N$ e-folds inside the horizon, **independently of $k$**:

| hand-over | $x$ there | cycles of $f$ | max err / envelope | at $z$ |
|---|---|---|---|---|
| `z_exit_subh_e3` (where the phase minimum is normally found) | 11.6 | 3.7 | **4.493e-04** | 501 |
| `z_exit_subh_e6` | 232.9 | 74.1 | 1.220e+00 | 24.2 |
| `0.85 · z_exit_subh_e6` (latest possible) | 272.1 | 86.6 | 1.129e+00 | 56.4 |

Consistent with the audit's QS-5 table ($1.8\times10^{-4}$ at 10 cycles, 0.88 at 95). Two
things follow, and both are recorded in `IMPLEMENTATION_STATE.md` §3 as issue
`[06-source-spline-residual-vs-handover]`:

- At the *realistic* hand-over the residual is **4.5e-04 of envelope**, just inside the
  prompt's $\sim10^{-3}$ threshold. This is the accuracy floor of the all-smooth region of
  the source integral, and it is one to two orders worse than either branch of
  `TkSourceFunctions` (log 05: 6.1e-06 LG, 7.4e-06 numeric) — because $f$ is quadratic in
  $T$ and so oscillates at *twice* the transfer-function phase, halving the nodes per
  half-cycle.
- If the hand-over were ever to fall near the bottom of `main.py`'s search window the
  residual would be $O(1)$. The knob is the window in `TkNumericIntegration.py:130-131` (or
  the grid density), **not** this prompt.

**§2.4 — datastore round-trip (ran a shape test; reasoned about the SQL).** Not exercised
against a live database (no datastore in a unit test). Read and reasoned:
`Datastore/SQL/ObjectFactories/QuadSource.py:236-261` rebuilds `z_sample` from the
`QuadSourceValue` rows themselves (`imported_z_sample = redshift_array(z_points)`, ordered by
`redshift.z DESC`), and `:313` writes the `z_samples` column as `len(obj.values)`, so the
consistency check at `:263-267` and `validate()` at `:365-377` both compare a truncated list
against a count derived from the same truncated list. `available` (`Datastore/object.py:20`)
depends only on `store_id`. So a truncated value list reads back with a consistent `z_sample`
and `available == True`, and **no factory change is needed** — §2.4's stop condition did not
trigger. `test_datastore_round_trip_shape` constructs a `QuadSource` with exactly the payload
the factory produces and asserts `available`, `len(z_sample) == len(values)`,
`numeric_region[1] == z^X_r`, and that both `crossover_z_*` are `None`.

**Not verified here, and needing a pipeline run (prompt 12).** That a real
`TkNumericIntegration` pair satisfies `crossover_z == last stored sample` (deviation 1's
clamp is a no-op) on the production grid; and that the `--quad-source-queue` stage now
completes end to end. Both need stored data.

## Observations not acted on

1. **`QuadSourceIntegral` will now raise below the both-numeric region.** All three regions
   read `source_f.source(log_z_source, z_is_log=True)`
   (`QuadSourceIntegral.py:958, 1028, 1102`), and `ZSplineWrapper` raises once more than 1 %
   (in $\log(1+z)$) below `min_z` (`spline_wrappers.py:50-53`). Before this commit that call
   returned a meaningless spline value; now it raises. This is the boundary the prompt
   describes ("everything below that ... is handled by the phase-group machinery of prompts
   07–08") and README §4's stopping-point note for 06, but it means the
   `--quad-source-integral-queue` stage cannot complete between this commit and prompt 08 for
   any $(k,q,r,z_{\rm resp})$ whose integration range reaches below the hand-over. Recorded
   in `IMPLEMENTATION_STATE.md` §3 as `[06-qsi-blocked-until-08]`. Not acted on: fixing it
   *is* prompts 07–08.
   `QuadSourceIntegral.py:1424` checks only `source.z_sample.max.z`, which truncation does
   not change, so the ingredient compatibility check still passes.
2. **`docs/spec-code-audit/scripts/QS_04_coverage_and_jacobian.py` part (a) no longer runs**
   against the current tree (its `MockTk` has no `.values`). Left alone deliberately: the
   audit scripts are the reproduction record of the pre-fix state, and README §5 item 5
   forbids fixing what the prompt did not ask for. Anyone re-running it should use the
   adapted copy described above. Part (b), the sympy Jacobian check, still passes
   (difference = 0).
3. **`ZSplineWrapper`'s error messages still say `GkSource.function:`**
   (`spline_wrappers.py:42, 52`), so a range error from the source spline reads
   `GkSource.function: evaluated quadratic source out of bounds`. Already recorded as log 05
   observation 1; audit B10 fixed the label passed in, not the hard-coded prefix. One line in
   a file no prompt in this campaign owns.
4. **`QuadSource` still consumes `TkNumericIntegration` only.** It never sees a
   `TkWKBIntegration` and does not construct a `TkSourceFunctions`. That is deliberate under
   this campaign's split: `QuadSource` is now only the smooth part, and the oscillatory part
   is assembled inside `QuadSourceIntegral` from `TkSourceFunctions` (prompts 07–08), never
   sampled and splined. Consequently nothing in `main.py`'s QuadSource stage needs the
   `TkWKBIntegration` objects, and none were plumbed in.
5. **The `SourceZGridSizeTag` family still describes the grid that was *requested*, not the
   grid that was stored.** `main.py` passes the same six tags as before, and the object is
   still queried by them, which is what makes lookups work; but two `QuadSource` rows with
   the same tags can now legitimately have different `z_samples` counts, because the count
   depends on $q$ and $r$ through the hand-over. That is fine for the factory (which filters
   on `(model, q_exit, r_exit, tags)` and takes the count from the row), but a human reading
   the tag would be misled. No change made — the tags are shared with the `Tk` stages and
   renaming them is outside this prompt.
6. **`analytic_source_w` / `analytic_source_rad` are truncated along with `source`.** They
   are oracle columns computed from the same values by the same kernel (audit QS-8), so they
   inherit the region without further thought; nothing reads them below the hand-over.

## State handed to the next prompt

Prompt 08 programs against the following.

```python
from ComputeTargets.QuadSource import QuadSource, numeric_crossover_z

src: QuadSource

src.numeric_region      # (z_max, z_min); Optional[Tuple[float, float]].
                        # Always available, before or after a datastore round-trip.
src.crossover_z_q       # Optional[float]; None after a datastore round-trip
src.crossover_z_r       # Optional[float]; None after a datastore round-trip

f = src.functions       # QuadSourceFunctions
f.source(z, z_is_log=False)   # ZSplineWrapper, valid only on f.numeric_region
f.numeric_region              # (z_max, z_min), == src.numeric_region

numeric_crossover_z(Tk)       # float or None; the hand-over of one TkNumericIntegration,
                              # == k_exit.z_exit - Tk.stop_deltaz_subh
```

1. **`numeric_region` is the only region in which `f.source` may be evaluated.** It is
   `(z_source_sample.max.z, z_floor)` with
   `z_floor = max(crossover_z_q, crossover_z_r, Tq_z_min, Tr_z_min)`. `ZSplineWrapper` gives
   a 1 % cushion in $\log(1+z)$ at each end and then raises `RuntimeError`; do not rely on
   the cushion. **Partition the source integral at `numeric_region[1]`**: above it use
   `f.source`, below it assemble the integrand from `TkSourceFunctions` (log 05) for both
   factors and never touch this spline.
2. **Prefer `numeric_crossover_z(Tq)` over `src.crossover_z_q`.** Prompt 08 holds the `Tk`
   objects, and the two hand-over redshifts are not persisted on `QuadSource`
   (deviation 3). `numeric_crossover_z(Tk)` is the same formula
   `TkSourceFunctions` validates against `TkWKBIntegration.z_init`, so
   `numeric_crossover_z(Tq) == TkSourceFunctions(model, q, Tq, Tq_WKB).crossover_z` for a
   consistent pair; asserting this is a cheap consistency check.
3. **`numeric_region[1]` can be strictly above `max(crossover_z_q, crossover_z_r)`** by up
   to one grid step (deviation 1's clamp). Trust `numeric_region`, not the crossovers, for
   where the spline stops. In the gap — if there is one — both factors are already in their
   LG representation as far as `TkSourceFunctions` is concerned
   (`WKB_region[0] <= crossover_z`), so partitioning on `numeric_region[1]` leaves no hole.
4. **The accuracy floor of the smooth region is 4.5e-04 of the local envelope** at the
   realistic hand-over (100 samples per log10 z, $q = r$, $x \approx 11.6$, ~3.7 cycles of
   $f$) — set by $f$ oscillating at twice the transfer-function phase. This is one to two
   orders *worse* than either `TkSourceFunctions` branch (6.1e-06 LG, 7.4e-06 numeric,
   log 05), so **the smooth region, not the oscillatory one, dominates prompt 08's error
   budget**, and shortening the numeric region (log 05 item 5) would help twice over. Do not
   set a tolerance on the all-smooth region below ~1e-3.
5. **`QuadSourceFunctions` is now a two-field namedtuple** `("source", "numeric_region")`.
   Construct it by keyword.
6. **Existing `QuadSource` rows are stale** — they were written before this commit with a
   value per source redshift, and the new code writes fewer. `IMPLEMENTATION_STATE.md` §5
   note 2 covers this; rebuild rather than migrate. The **schema is unchanged**, so old rows
   are still *readable*; they are simply wrong for the new consumer, and a row whose
   `z_samples` count exceeds the current region will silently give the spline a range that
   extends below the both-numeric region.
7. **`QuadSourceIntegral` is blocked until prompt 08** for any integration range reaching
   below `numeric_region[1]` (observation 1). That is expected, not a regression to fix in
   prompt 07.
8. **Test fixtures are reusable.** `ComputeTargets/tests/test_quadsource.py` exposes
   `Hubble`, `tau`, `f_spec`, `f_exact`, `FakeModelProxy`, `FakeTk`, `FakeTkValue`,
   `FakeWavenumberExit` and `source_grid(z_init, z_end, samples_per_log10z)`. `FakeTk` takes
   `(z_grid, k, w, z_exit, stop_deltaz_subh)` and supplies `.values`, `.z_sample`, `.z_exit`,
   `.stop_deltaz_subh`, `.available` and `__getitem__`. Prompt 08 can build a
   `(q, r)` pair at two different $k$ with hand-overs on chosen grid points, and pair them
   with prompt 05's `Fixture` for the LG side.
9. **No schema change, no `Datastore` change, no `MetadataConcepts/QuadSourcePolicy` change.**
