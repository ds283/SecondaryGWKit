# Issues from prompt 03 (for the orchestrator to open or drop)

No stop condition was hit, and nothing was repaired. Items 1 to 3 are the three issues that prompt 03 §5
names, drafted from the code **re-read at the dispatch tree** (`c8ca1559fcc53fdb88998f170257635431fb85ca`;
`ComputeTargets/GkSourcePolicyData.py`, `GkSource.py` and the two factories are unchanged since `98e68d3`,
the Gate's empty diff). `main.py` has moved a long way since the campaign README cited it (it is now
about 4,350 lines), so every `main.py` line below is the line found there, not the README's. None of the
three was fixed, and no recommendation is made. Items 4 to 7 are observations of the census itself, offered
as candidate §3 rows. **Do not read items 4 to 7 as instructions**; they say what was measured and no more.

## 1. `[03-gksource-policy-accepts-a-value-that-raises]` (README §2 (o) item 1; workstream C2)

- **The set the constructor accepts.** `MetadataConcepts/GkSourcePolicy.py:6`:
  `_allowed_numeric_policies = ["maximize-numeric", "maximize-WKB"]`. The check is at `:25-30`.
- **The set the consumer branches on.** `ComputeTargets/GkSourcePolicyData.py:457` tests
  `policy_name == "maximize-WKB"`; `:465` tests `policy_name == "minimize-WKB"`; the `else` at `:472-475`
  raises `RuntimeError(f'Unknown GkSourcePolicy numeric policy "{policy_name}"')`. The comment at `:464`
  calls the `minimize-WKB` branch "the maximize-numeric policy".
- **So:** `"maximize-numeric"` is constructed without complaint, is persisted (`numeric_policy` is a column of
  `GkSourcePolicy`, `ObjectFactories/GkSourcePolicy.py`, and part of the datastore key of `GkSourcePolicyData`),
  and `_classify_crossover` raises on it. `"minimize-WKB"`, the only spelling the branch at `:465` answers
  to, fails the constructor's check and is replaced by the default. So **the branch at `:465-470` cannot be
  reached by any constructible value**.
- **When the raise fires.** The comparison sits inside the band loop, after `filtered_values` is non-empty
  (`:444-451`), so it fires only on a `mixed` row that has an overlap and a non-empty band. A `numeric` or
  `WKB` row never reads the name. A `mixed` row whose bands are all empty takes the fallback at `:477-`
  (`crossover_z = primary_WKB_largest_z`, `quality = "minimal"`) without reading it either.
- **Reachability today.** Every policy the repository constructs passes `"maximize-WKB"`:
  `main.py:4321-4323`, `:4327-4329`, `:4342-4344`, `extract_QuadSourceIntegral_data.py:1189-1197`,
  `config/defaults.py:190-202`. The test fixture `StoreFixtures/real_store_fixtures.py:464` and `:480` write
  rows with `"maximize-numeric"` into a fixture store; nothing computes on them.
- **Impact:** latent. The only value that validates and is not the default raises at compute time inside a
  Ray remote, on the one kind of row (`mixed`, 72 of 1160 on A3 v2) that reads it.

## 2. `[03-quadsource-policy-vocabulary-differs]` (README §2 (o) item 2; workstream C2)

- `MetadataConcepts/QuadSourcePolicy.py:6`: `_allowed_numeric_policies = ["maximize_numeric", "maximize-WKB"]`,
  with an underscore, against `GkSourcePolicy.py:6`'s hyphen. Three spellings of one idea exist in the tree
  (`maximize-numeric`, `maximize_numeric`, `minimize-WKB`).
- **Nothing reads the field.** The class docstring (`QuadSourcePolicy.py:10-43`) says so. It is persisted and
  keys `QuadSourceIntegral` (`quad_policy_serial`), and neither `QuadSource.py` nor `QuadSourceIntegral.py`
  consults it.
- **The two warning messages that lose the offending value.** Both constructors reassign `numeric_policy` to
  the default **before** formatting the message, so the message prints the default twice:
  - `MetadataConcepts/GkSourcePolicy.py:25-30`: line 26 assigns, line 28 formats
    `!! Unknown GkSource numeric policy "{numeric_policy}"`;
  - `MetadataConcepts/QuadSourcePolicy.py:60-65`: line 61 assigns, line 63 formats
    `!! Unknown Quadsource numeric policy "{numeric_policy}"` (also spelling "Quadsource" with a lower-case s).
- **Impact:** none on a run today (no caller passes anything but the default). A person who passes a wrong
  value is told that the wrong value was the default.

## 3. `[03-gk-classification-is-diagnosed-and-ignored]` (README §2 (p); workstream B3, invariant B4)

- **The classification** `_classify_crossover` (`GkSourcePolicyData.py:287-504`) has routes to
  `quality = "incomplete"` (`:358` numeric, `:374` WKB, `:390` mixed with no overlap) and to
  `type = "fail"` (`:342-344`).
- **Where `quality` or `type` is read, found at this tree, and what each does:**
  - `extract_common.py:698-703`: prints `Quality: {fns.quality}` on a plot.
  - `main.py:2894-2895` (inside `postprocess_GkSourcePolicy`, `:2883-`), and `:3277-3300`: counted into the
    `GkSourcePolicyData SUMMARY STATISTICS` block.
  - `main.py:3132` builds the list `incomplete` from `obj.quality == "incomplete"`; `:3140` uses it only
    `if dump_incomplete`.
  - `main.py:3224-3225` (`validate_GkSourcePolicy_work`): `if obj.quality == "incomplete":
    dump_incomplete_GkSourcePolicy.remote(obj)`.
  - `ComputeTargets/GkSourcePolicyData.py:224` (`_classify_Levin`): returns early
    (`Levin_z` is diagnostic only).
  None of them stops or filters anything.
- **No filter on the way to the source integral.** `main.py:385` takes `GkPolicy` from `Gk_cache` for the
  work item with no test of `type` or `quality`; the lookups that fill the cache
  (`main.py:3830-3860`, and `:3560-3590`) check only `obj.available`.
- **Where it ends.** `ComputeTargets/QuadSourceIntegral.py:325-345`: for a Green's function that is
  oscillatory (smooth) on a sub-interval but has no WKB (numeric) representation, `build_partition` raises
  `RuntimeError("... has no WKB representation (type=..., quality=...)")` /
  `("... no numeric representation ...")`, and `_check_region_covers` (`:444-`) is strict, with no clamp.
- **The ungated dump.** `dump_incomplete = args.dump_incomplete is not None` (`main.py:2801`) and
  `dump_incomplete_path = ... if dump_incomplete else None` (`:2802-2804`). `validate_GkSourcePolicy_work`
  (`:3224-3225`) calls `dump_incomplete_GkSourcePolicy.remote(obj)` **without** testing `dump_incomplete`.
  With the flag unset, `dump_incomplete_path` is `None`, so the remote function's
  `dump_incomplete_path / f"store_id=..."` (`:2812-2814`) is `None / str`, a `TypeError` raised in a Ray task
  whose reference the caller discards.
- **No test.** `ComputeTargets/tests/test_gk_source_policy.py` is 214 lines and defines three tests
  (`:155`, `:174`, `:195`), every one of them about `_classify_Levin`. No module under
  `ComputeTargets/tests/` mentions `_classify_crossover`.
- **What prompt 11 measured, carried here as the prompt asks** (`A3-V2-VERIFICATION.md` §5.3; reproduced by
  this census, `out/policy_geometry_03.txt` §2): the shipping grid (A3 v2, `SourceGridConstruction_2`) has **no**
  `fail` and **no** `incomplete` row among its 1160 `GkSourcePolicyData` rows: `numeric`/`complete` 363,
  `mixed`/`complete` 71, `mixed`/`minimal` 1, `WKB`/`complete` 725. So the fail-late contract is unenforced
  **and** has not been exercised on the grid that ships: none of the routes above has been taken by a row of
  this store. The one non-`complete` row is `minimal`, which is not a failure route (shard 2, serial 460,
  $k = 3.091682\times10^6$/Mpc, $z_{\rm response} = 3.963729\times10^{10}$).

## 4. Candidate: the `complete` seam sits at one node near the top of the grid, on 71 of 72 `mixed` rows

Measured by this census (`POLICY-GEOMETRY.md` §3, §6): on all 71 `mixed`/`complete` rows `crossover_z` is
$3.121867\times10^{15}$, the same grid node for every $k$ and every $z_{\rm response}$; `primary_WKB_largest_z`
is the top of the source grid ($2.0636\times10^{16}$) on those rows, `maximize-WKB` therefore takes the first
node whose WKB clearance exceeds `CLEARANCE_GOOD` = 0.05, and that node is 82 grid intervals below the top
(WKB_intervals / (0.05 log(1+z_WKB) / mean step) = 1.0055 on every one of the 71 rows). In e-folds sub-horizon the node is N = -3.11 for $k = 3\times10^8$/Mpc, N = -4.26 for $k = 9.56\times10^7$/Mpc and `<-5`
(outside the exit-time table) for every smaller $k$, so it is super-horizon on every one of them.
Numeric $G$ is consumed above that node and the WKB $G$ below it, so on these rows the consumed range of the
numeric spline is the top 82 intervals of the grid. Stated, not read: nothing here says whether that matters.

## 5. Candidate: README §2 (q)'s "upper branch adds above the band on some rows" against 71 of 72

Measured (`POLICY-GEOMETRY.md` §6): the half-e-fold band ($z_{e4} \le z < \sqrt{z_{e3}z_{e4}}$) carries both
representations on **all 72** `mixed` rows, so the prediction README §2 (q) states is **not** contradicted. The
overlap above it is not a feature of "some": `primary_WKB_largest_z` is the grid top on 71 of 72 rows, so
the overlap runs from $z_{e4}$ to the grid top, 8.8 to 17.0 in $\log(1+z)$ (about 9 to 17 e-folds), not 0.5. The
one row where it is not is the `minimal` row (item 6). The correction of §2 (q), if one is wanted, is the user's.

## 6. Candidate: the `minimal` row's overlap is cut by a WKB hole, and the hole census is not empty

Measured (`POLICY-GEOMETRY.md` §8): from the stored value rows, no GkSource row has a WKB hole with
$z_{\rm response} < z_{e4}$ (the condition of the warning at `GkSource.py:283-305`), and **12** of 1160 rows have
one with $z_{\rm response} \ge z_{e4}$: 11 `numeric` rows and 1 `mixed` row. The `mixed` row is the `minimal` row
(shard 2, serial 460). Its WKB run, contiguous from $z_{\rm response}$, ends at N = 3.507 (z = 4.2965e10), its first
hole is the node at N = 3.484 (z = 4.3966e10) and WKB values resume at N = 3.438 (z = 4.6039e10). That row's
overlap is 7 grid intervals, 6 numeric and 1 WKB, with `c_WKB` = 4.7e-04. The first hole of each of the 12
lies at N between 3.35 and 3.50 on 4 rows (the `minimal` row among them, next to $\sqrt{z_{e3}z_{e4}}$ at N = 3.5), between
2.40 and 2.73 on 7 rows and at 0.99 to 1.01 on 1 row. 17 rows (16 `numeric`, 1 `mixed`)
have WKB nodes above their primary run.

## 7. Candidate: verification §5.5's shape, accounted for, with one inference that cannot be checked

`POLICY-GEOMETRY.md` §5. The restricted comparison ($10^5 \le k \le 10^7$/Mpc) leaves `mixed` at 6.34 %
against §4.2's 13.42 %. Counted per $k$, the `mixed` rows are 9.2 on A3 v2 against 8.86 from §5.5's
(61 + 1)/7, if its seven $k$ carried equal counts; what differs is the number of non-`mixed` rows per $k$ (A3 v2
145; §5.5's 462/7 = 66). Restricting also to $z_{\rm response} \ge 10^7$, a lower limit §5.5's own TkWKB table
suggests but does not establish for the old grid, gives `numeric` 66.15 %, `WKB` 22.05 %, `mixed` 11.80 %. The
old store is gone (prompt §2.2 difference 1), so the old store's response grid cannot be checked.
