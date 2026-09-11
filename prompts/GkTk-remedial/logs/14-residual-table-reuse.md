# Log 14 — Build the phase residual once per $(model, k, sector)$

**Prompt:** prompts/GkTk-remedial/14-residual-table-reuse.md
**Commit:** (this commit) — Build the WKB phase residual once per wavenumber
**Model:** Claude Opus 5
**Date:** 2026-09-11
**Result:** COMPLETE WITH DEVIATIONS

**Out of sequence.** This prompt is numbered 14 because the campaign's numbers are append-only,
but it runs **between 06 and 07**: prompt 07 inherits the same producer
(`WKB_phase_function`) and would otherwise double the exposure to
`[06-residual-table-per-object]`. The board's row 14 sits between 06 and 07 for that reason.

---

## What shipped

### `ComputeTargets/phase_residual.py` (additive only)

`build_phase_residual`, `phase_residual_integrand`, `RHO_GAUSS_ORDER` and
`RHO_ADAPTIVE_FALLBACK_REQUIRED` are untouched — signature and behaviour both.
`ComputeTargets/tests/test_phase_residual.py` passes unchanged (`git diff HEAD~1 --stat` does not
name it).

New module-level names:

```python
RESIDUAL_WKB_REGION_MARGIN = 0.5        # the fraction of omega_0^2 a retained node must keep
RESIDUAL_CACHE_MAX_ENTRIES = 256        # LRU cap on the cache

def residual_node_range(model, k: float, z_grid: Sequence[float], sector: str,
                        margin: float = RESIDUAL_WKB_REGION_MARGIN) -> np.ndarray
def phase_residual_cache_key(model, k: float, sector: str,
                             store_id: Optional[int] = None,
                             order: int = RHO_GAUSS_ORDER) -> tuple
def cached_phase_residual(model, k: float, z_grid: Sequence[float], sector: str,
                          order: int = RHO_GAUSS_ORDER,
                          store_id: Optional[int] = None) -> Tuple[CumulativeTable, bool]
def clear_phase_residual_cache() -> None
def phase_residual_cache_size() -> int
```

* **`residual_node_range`** (`:250-320`) is the node rule: the background model's own grid, cut at
  the bottom to the lowest grid node and at the top to the highest node at which the
  Liouville–Green frequency still keeps `RESIDUAL_WKB_REGION_MARGIN` of its leading term. It walks
  **upwards from the lowest node** — deep inside the horizon, where the frequency is positive by
  construction — and stops at the first node where `leading < 0` or
  `leading + correction < margin * leading`, tested through the same
  `*_omegaEff_sq_leading`/`*_omegaEff_sq_correction` pair the integrand uses. The range is
  therefore the largest run of retained nodes reaching the bottom of the grid, not everything
  below the highest positive node; see Deviation 2 for why that distinction is load-bearing.
* **`cached_phase_residual`** (`:391-...`) memoises one `CumulativeTable` per
  `(model, k, sector, order)` in a module-level `OrderedDict` with LRU eviction, and returns
  `(table, reused)`. The nodes are computed only on a miss. `z_grid` is a property of the model,
  read on a build and not compared on a hit.
* **The key** is `("store", store_id, k, sector, order)` when the model has a datastore id, and
  `("obj", id(model), k, sector, order)` when it does not — `ModelProxy.store_id` is `None` for an
  unavailable model (`BackgroundModel.py:963`) and for every offline stand-in, so `None` is never
  itself a key. `_ResidualCacheEntry.serves(model)` confirms an identity-keyed hit through a
  `weakref`, so a recycled `id()` cannot be served another model's table.
* **Memory.** 0.395 MB per 1,732-node table, measured with `tracemalloc` over ten tables on the
  production grid. A production run builds one background model (`main.py:471`) over ~50
  wavenumbers in two sectors: 100 keys, ~40 MB per worker. Both production models in one worker
  would be 200 keys and ~79 MB. The cap of 256 is above that, so it never binds in production —
  reuse cannot silently degrade into thrashing — and still bounds a longer walk at ~101 MB.

### `Quadrature/integrators/WKB_phase_function.py`

* `residual_nodes(grid, z_sample, z_init)` (`:93-114` before) is **deleted**; in its place
  `nearest_table_node(table: CumulativeTable, z: float) -> float` (`:93-116`) returns the node of a
  table nearest to `z` in `u = log(1+z)`, mirroring `CumulativeTable`'s own nearest-node rule
  (exact-`z` lookup first, `log(1+z)` only on the off-grid branch).
* `:234-235` before — `nodes = residual_nodes(...)`, `rho = build_phase_residual(...)` — becomes
  `rho, rho_reused = cached_phase_residual(model, k_float, leading_table.z_nodes, sector,
  store_id=getattr(model_proxy, "store_id", None))`.
* A new guard raises `RuntimeError` naming both ranges if `z_init` or any sample falls outside
  `[rho.z_nodes[-1], rho.z_nodes[0]]`, so `delta` only ever forms interior partials
  (`CumulativeTable` refuses an endpoint more than one grid interval beyond either end) — asserted
  rather than assumed, as the prompt requires.
* The anchor is split off at a node once per object:
  `rho_anchor_node = nearest_table_node(rho, z_init)`, `rho_anchor = rho.delta(z_init,
  rho_anchor_node)`, and the sample loop forms `rho_anchor + rho.delta(rho_anchor_node, z)`, which
  is free for an on-grid sample. Exactly `0.0` with no integrand call when `z_init` is itself a
  node.
* `metadata` gains **`"rho_reused"`** (`bool`). `metadata["rho_evals"]` now counts the residual
  integrand evaluations **this call actually spent** — `rho.total_evaluations` differenced across
  the call, so the table build (none on a reuse) plus this object's anchor partial — and
  `stage_1_data.RHS_evaluations` is `rho_evals + lead_evals` as before. `rho_nodes` is the cached
  table's node count. `rho_end` is taken from the sample loop rather than recomputed, so it costs
  nothing and cannot escape the evaluation count.
* The module docstring's "**The anchor is off the grid**" paragraph is rewritten for the split, and
  the header sentence for `cached_phase_residual`.

### `ComputeTargets/tests/test_gk_wkb_phase.py` (only what §3.1 allows)

`git diff` shows three hunks: the imports; `TestOffGridAnchor.test_residual_nodes_put_the_anchor_on_top`
replaced by `test_residual_node_range_is_the_grid_cut_at_the_wkb_region`; and `"rho_reused"` added
to the metadata-key list of `TestPayloadContract.test_keys_and_stage_data`. Every accuracy, cost
and sweep test is untouched and passes at its existing threshold.

### `ComputeTargets/tests/test_residual_table_reuse.py` (new, 10 tests)

Prompt 14 §3.2 items 1–5, plus two additions (Deviation 7). Fixtures are prompt 06's, imported
from `test_gk_wkb_phase`; `_ProxyWithStoreId` adds the `store_id` attribute the production
`ModelProxy` carries. `prompt_06_residual_nodes` is a local copy of the deleted per-object rule,
kept as the control item 3 scores against.

---

## Deviations from the prompt

### 1. `residual_nodes` replaced, and the new rule lives in `phase_residual.py` — IMPLEMENTATION CHOICE

The prompt leaves the choice open ("keep it with the new semantics, replace it, or delete it") and
asks it to be recorded. The old function's three arguments were exactly the per-object inputs the
change removes, so keeping the name with new semantics would have been a trap. It is replaced by
`residual_node_range(model, k, z_grid, sector)`.

It sits in `phase_residual.py` rather than in `WKB_phase_function.py` because the rule is a
statement about where the residual *integrand* is defined: it is tested with the same
`_SECTOR_FUNCTIONS` pair the integrand uses, so the test and the integrand cannot drift apart.
`phase_residual.py` may be touched additively and this is additive. The alternative — leaving the
rule in the producer — would have meant importing the private `_SECTOR_FUNCTIONS`, or duplicating
the leading/correction dispatch.

### 2. The top is cut where `omega^2 >= 0.5 * omega_0^2`, not where `omega^2 > 0` — STRUCTURALLY REQUIRED

The prompt says "restricted at the top to the highest node at which $\omega^2(k,z)>0$". That rule
does not survive contact with `QCD_Cosmology`, and it was implemented literally first and failed:

* **The sign is not monotone in $z$.** Implemented as "the highest node at which $\omega^2>0$",
  the rule returned the whole grid for QCD at $k=3\times10^8$, because the Green's-function
  frequency is positive again at the top of the production grid ($z\ge1.9\times10^{16}$) while
  negative just below; the build then raised at $z=1.84\times10^{16}$.
* **Re-implemented as the largest run reaching the bottom of the grid, it still failed**: the
  build raised at $z=3.6118639\times10^{15}$, an *interior Gauss abscissa* of a panel whose two
  nodes are both positive (`leading = 2.898e-34`, `correction = -2.904e-34`,
  $\omega^2=-5.9\times10^{-37}$).
* The reason is `[02-qcd-T-z-spline-node-tolerance]`: near the turning point $\omega^2$ is a small
  difference of two nearly equal quantities, and the ratio $\omega^2/\omega_0^2$ scatters over
  neighbouring nodes — measured, at $k=3\times10^8$ around $z\sim4\times10^{15}$: $+0.106$,
  $-0.029$, $+0.0025$, $+0.213$, $+0.125$, $+0.0088$, $+0.031$, $+0.201$. The sign of the
  Liouville–Green frequency in that band is simply not resolved by the background.

A margin is therefore required, and one half is the value. It is not arbitrary, and it does not
narrow what the producer can do:

| model / sector / $k$ | cut node (margin 0.5) | highest node with $\|d\ln\omega/dz\|/\omega\le1$ |
|---|---|---|
| LambdaCDM Gk $10^5$ / $3\times10^8$ | $z=2.06\times10^{16}$ (nothing cut) | $2.31\times10^{10}$ / $6.82\times10^{13}$ |
| LambdaCDM Tk $10^5$ / $3\times10^8$ | $1.33\times10^{10}$ / $3.93\times10^{13}$ | $9.61\times10^{9}$ / $2.91\times10^{13}$ |
| QCD Gk $10^5$ / $3\times10^8$ | $2.65\times10^{11}$ / $1.06\times10^{15}$ | $2.53\times10^{10}$ / $1.08\times10^{14}$ |
| QCD Tk $10^5$ / $3\times10^8$ | $1.46\times10^{10}$ / $6.08\times10^{13}$ | $1.05\times10^{10}$ / $4.40\times10^{13}$ |

In every case the cut sits **above**, in $z$, the highest node at which the WKB validity criterion
holds — and `WKB_phase_function` has refused an anchor violating that criterion since before this
campaign (`:192-196`, unchanged). So no anchor the producer will accept can lie above the table.
Production anchors are three e-folds inside the horizon, where the ratio exceeds $0.99$; the cut
lands about 1.25 e-folds inside the horizon in the `Tk` sector and outside it in `Gk`.

**Acceptance:** 80 table builds — both production models, both sectors, 20 wavenumbers
geometrically spaced over $10^5$–$3\times10^8$ — all succeed at margin 0.5. Recorded as
`[14-residual-range-top-margin]`.

### 3. A reusing object spends zero *build* evaluations, not zero residual evaluations — STRUCTURALLY REQUIRED

§3.2 item 5 asks the test to "assert the second and subsequent objects spend **zero**
residual-integrand evaluations". An object whose anchor is off the grid cannot: reaching it costs
one Gauss panel of the residual integrand, which is the same off-grid partial §2.1 of the prompt
names as "precisely the term that accessor exists for". What is asserted instead, and measured:

* the second and every subsequent object adds **no** evaluation to the table's build count
  (`CumulativeTable.evaluations` is identical before and after), and reports `rho_reused: true`;
* an object whose anchor is **on** the grid spends exactly **zero** residual-integrand evaluations
  (measured: two of the four objects of `TestTableIsShared`);
* an object whose anchor is off the grid spends exactly `RHO_GAUSS_ORDER = 4` — one panel, paid
  once for the object rather than once per sample, because the anchor is split at the nearest
  node (Deviation 4).

### 4. The anchor is split at the nearest table node — IMPLEMENTATION CHOICE

The prompt says the anchor is "reached through `CumulativeTable.delta`'s off-grid partial" but
does not say how often. Calling `rho.delta(z_init, z)` per sample, the obvious reading, costs
`order` evaluations *per sample* — 464 per object at $k=3\times10^8$ over the production response
grid, which would have left a third of the per-object cost in place. Writing it as
`rho.delta(z_init, n) + rho.delta(n, z)` with `n` the nearest node pays the partial once and makes
the second term free for an on-grid sample.

The price is one extra double-double difference of a quantity below 0.1 rad, i.e. a rounding of
order $10^{-17}$ rad. Measured against the per-object table (item 3.3): $\rho$ moves by at most
$1.4\times10^{-17}$ rad and $\theta$ itself is **bit-identical at every sample**.

`nearest_table_node` duplicates `CumulativeTable._nearest_node`'s rule rather than calling it,
because `cumulative_table.py` is in this prompt's "do not touch" list and the method is private.
The alternative — splitting at an arbitrary nearby node — would have widened the partial panel.

### 5. Item 3.3 is scored on the unreduced phase, and on $\rho$ — IMPLEMENTATION CHOICE

The payload's `div*2pi + mod` carries the $\varepsilon k\tau$ representation floor,
$9.8\times10^{-4}$ rad at $k=3\times10^8$ (README §2 (d)), which would swamp a $10^{-9}$ rad
acceptance. The test therefore forms both phases unreduced, as doubles, from the shared and the
per-object table, and reports the difference in $\theta$ **and** in $\rho$ alone; and it separately
asserts that the payload reproduces the unreduced $\theta$ it scored, to
$8\varepsilon\max|\theta|$. Scoring $\rho$ as well matters because $\theta$'s own ulp at
$k=3\times10^8$ is $4.9\times10^{-4}$ rad, so "the phase is bit-identical" would hold even for a
much larger change in $\rho$; the $\rho$ figure is the sharp one.

### 6. The `rho_evals` metadata key changes meaning — IMPLEMENTATION CHOICE (required by §2.3)

§2.3 requires `stage_1_data.RHS_evaluations` to count only this call's evaluations. Rather than add
a second key — the `metadata` column has 29 characters of headroom — `rho_evals` was redefined
from "the table's build cost" to "the residual evaluations this call spent", which leaves
`RHS_evaluations == rho_evals + lead_evals` true and `test_keys_and_stage_data`'s assertion
unchanged. A reader of an old row cannot tell the difference; the `rho_reused` flag beside it is
what disambiguates.

### 7. Two tests beyond the prompt's five items — IMPLEMENTATION CHOICE

`TestTableIsShared.test_metadata_still_fits_the_column` asserts the JSON payload against
`DEFAULT_STRING_LENGTH`, because `[06-metadata-column-headroom]` is live and §2.3 required a new
key. `TestAnchoringIsFree.test_phase_is_unchanged_in_the_transfer_function_sector` repeats item 3.3
in the `Tk` sector, where $\rho_T\approx-0.09$ rad is six orders larger than $\rho_G$ and an
anchoring change would show first — and where prompt 07 inherits the code.

---

## Verification performed

Everything below was run; nothing here is reasoning in place of a run.

### §3.1 — every threshold met, at prompt 06's numbers

`PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_gk_wkb_phase` — **21 tests,
OK**.

| | Threshold | Prompt 06 | Measured now |
|---|---|---|---|
| radiation span $10^7$ rad | ≤1e-8 | 3.7253e-09 | **3.7253e-09** (at $z=0.229289$) |
| radiation span $10^9$ rad | ≤1e-6 | 3.5763e-07 | **3.5763e-07** (at $z=0.398558$) |
| LambdaCDM $k=10^5$ | ≤1e-5 | 1.1921e-07 | **1.1921e-07** (at $z=10.0123$) |
| LambdaCDM $k=3\times10^8$ | ≤5e-3 | 9.7656e-04 | **9.7656e-04** (at $z=1.00062$) |
| QCD $k=3\times10^8$ | ≤5e-3 | 9.7656e-04 | **9.7656e-04** (at $z=1.00062$) |
| off-grid anchor (37 % through an interval) | ≤1e-8 | 3.7253e-09 | **3.7253e-09** (at $z=0.1$) |
| cost per object, $k=3\times10^8$ | ≤0.05 s | 0.0309 s | **0.0010 s** (116 samples; 468 integrand evaluations = 4 residual + 464 leading) |
| cross-object sweep, 990 objects | 0 rebase offsets; jumps = transitions | 90 = 90 | **90 = 90**, integer defect 3.97e-12 rad, `apply_phase_offset` defect 4.44e-16 rad |

Every figure is identical to prompt 06's to the digits it published. The cost row is the only one
that moves, and it moves down by 31× because `TestCost`'s three repeats now reuse the table built
by `TestRealBackground`.

`git diff HEAD~1 --stat` does not name `ComputeTargets/tests/test_phase_residual.py`; it passes
unchanged inside the discover run below.

### §3.2 — the new module

`PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_residual_table_reuse` —
**10 tests, OK, 4.8 s**.

1. **The table is shared.** LambdaCDM, $k=10^7$, sector `Gk`: one table of 1,732 nodes and 6,924
   integrand evaluations serves four objects with anchors
   $2.25583\times10^{11}$ (on grid), $2.22333\times10^{11}$ (37 % through the interval below it),
   $8.84906\times10^{10}$ (off grid) and $3.57348\times10^{10}$ (on grid) and 155, 83, 55, 40
   samples. Every call resolves to the same table *object* (`assertIs`), the cache holds one
   entry, the build count never changes, and the four objects' `rho_evals` are **[0, 4, 4, 0]** —
   the anchor partials alone.
2. **The key discriminates.** A different $k$, a different sector and a different `store_id` each
   force a rebuild and give a different object; the `Tk` table is the shorter one (its frequency
   turns over inside the horizon). Two offline stand-ins with `store_id=None` (`RadiationModel`
   with $H_0=1$ and $H_0=2$) get **separate** tables and each gets its own back;
   $\rho_T$ over the range is $-8.634090\times10^{-2}$ and $-1.733378\times10^{-1}$ rad, so they
   really are different tables and not merely different objects.
3. **The answer did not move.** Shared table against a table built prompt 06's way (anchored at
   $z_{\rm init}$), `Gk` sector, nine (model, $k$) cases —
   `RadiationModel`/`LambdaCDMModel`/`QCDModel` × $k\in\{10^5,10^7,3\times10^8\}$, 40–117 samples
   each: **$\theta$ is bit-identical, $0.000\times10^{0}$ rad, at every sample of every case.**
   In $\rho$ alone the worst is **5.421e-19 rad at (`QCDModel`, $k=3\times10^8$,
   $z=3.661255\times10^{12}$)**; per model, 0 (radiation, exactly), 2.827e-19 (LambdaCDM,
   $k=10^7$), 5.421e-19 (QCD). In the `Tk` sector, six cases: $\theta$ again bit-identical,
   $\rho$ worst **1.388e-17 rad at (`LambdaCDMModel`, $k=10^5$, $z=6.947540\times10^{5}$)**.
   Threshold 1e-9 rad; met with eight orders of margin.
4. **Radiation control survives.** Through the shared table, every `hi` and every `lo` limb of
   $\rho_G$ is bit-exactly `0.0`, the off-grid anchor partial is bit-exactly `0.0`, the payload's
   `rho_end` is `0.0`, and the range is the whole grid ($\omega^2=(k/H)^2$ keeps all of its
   leading term). $|\theta + k\,\Delta\tau| \le 3.725\times10^{-9}$ rad for both an on-grid and an
   off-grid anchor.
5. **Cost.** 50 objects of one wavenumber, each with its own off-grid anchor, $k=3\times10^8$:

   | | table build | per later object | amortised over 50 | before prompt 14 |
   |---|---|---|---|---|
   | LambdaCDM | 6,924 evaluations, 1,732 nodes | 4 (its anchor partial), 0 build | **142.5** | 6,924 per object |
   | QCD | 7,908 evaluations, 1,603 nodes | 4, 0 build | **163.0** | 7,908 per object |

   **49× fewer residual-integrand evaluations per object** on both models.

### The metadata column

`[06-metadata-column-headroom]` re-measured with `rho_reused` present, `json.dumps` of the
payload's `metadata`: **227 characters** worst case (LambdaCDM, $k=3\times10^8$, `Gk`, first call),
against `String(DEFAULT_STRING_LENGTH) = String(256)` — **29 characters of headroom**. The four
measured variants: 226 (`Gk` built), 222 (`Gk` reused), 223 (`Tk` built), 219 (`Tk` reused); the
`initial_data_only` payload is 82. Prompt 06 recorded 206 without the key, so the key costs 20–21
characters. Asserted by a test, so a later prompt that adds one cannot overflow it silently.

### Table build cost and node ranges (20 wavenumbers per model per sector, $10^5$–$3\times10^8$)

| model / sector | nodes | evaluations per table | time per table |
|---|---|---|---|
| LambdaCDM `Gk` | 1732 (nothing cut) | 6,924 | 44 ms |
| LambdaCDM `Tk` | 1113–1460 | 5,143 mean | 27 ms |
| QCD `Gk` | 1243–1603 | 7,023 mean | 149 ms |
| QCD `Tk` | 1117–1479 | 6,371 mean | 267 ms |

All 80 builds succeed. Against `[06-residual-table-per-object]`'s "roughly 0.5 min (LambdaCDM) to
~4 h (QCD) of repeated identical quadrature per model per run", the whole set of tables for a
50-wavenumber run in both sectors now costs **3.6 s (LambdaCDM)** and **21 s (QCD)** per worker
that touches every wavenumber.

### Memory

`tracemalloc` over ten tables on the production grid: **0.395 MB per 1,732-node table**. 100 keys
(one model, 50 wavenumbers, two sectors) ≈ **40 MB per worker**; 200 keys (both models) ≈ 79 MB;
the cap of 256 bounds it at ≈ 101 MB.

### Suites and formatting

* `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` — **209 tests,
  OK, 133 s**.
* `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase` — 6 tests, OK.
  `LiouvilleGreen.tests.test_range_reduce` — 4 tests, OK.
* `./venv/bin/python -m black --check` on the four touched files — clean. (The repository as a
  whole is not clean under `black --check`: 54 files under `docs/` would be reformatted, all of
  them pre-existing and none of them touched here.)

---

## Observations not acted on

1. **The persisted `RHS_evaluations` of a WKB integration is now build-order dependent.** §2.3
   requires the column to count what the call spent, so the first object of a given
   $(model, k, sector)$ in a worker stores ~7,000 and every later one stores a few hundred (the
   leading table's partials). Which object is "first" depends on how Ray schedules the tasks, so
   two runs over the same datastore can store different values for the same object. The column is
   payload data and not part of any lookup key, so nothing misses; but anyone reading it as
   "the cost of this object" is misled, and prompt 13's timing of the scoped pipeline run should
   sum it rather than sample it. Opened as `[14-rhs-evaluations-depend-on-build-order]`.
2. **The residual table's top cut is a measured constant.** `RESIDUAL_WKB_REGION_MARGIN = 0.5`
   (Deviation 2). It is inert today because the cut is above the WKB-criterion boundary in all
   eight measured (model, sector, $k$) cases, so `WKB_phase_function` refuses such an anchor
   before the table is consulted. A producer that ever anchored within about one e-fold of horizon
   crossing would meet the new range guard's `RuntimeError` instead of a silently built table.
   Opened as `[14-residual-range-top-margin]`.
3. **`CumulativeTable` has no public nearest-node accessor.** `nearest_table_node` in
   `WKB_phase_function.py` duplicates the private `_nearest_node` rule because
   `cumulative_table.py` is out of scope here. If a second caller ever needs it — prompt 09's
   `PrimitivePhase` is the candidate, since it will want to split its own anchor the same way —
   the rule should move onto `CumulativeTable` as a public method and the copy should go. Not
   opened as an issue: it is a one-function refactor with no measurement attached, and it is
   recorded here and in the function's docstring.
4. **The `Gk` residual table on LambdaCDM covers the whole grid, including the super-horizon
   region.** The Green's-function correction $C$ vanishes identically in exact radiation, so
   $\omega^2=(k/H)^2>0$ even 13 e-folds *outside* the horizon and nothing is cut; the table
   therefore accumulates $\rho_G$ from $z=2.06\times10^{16}$, where the Liouville–Green
   representation has no physical meaning. This costs 6,924 rather than ~5,500 evaluations per
   table and is otherwise harmless: the contribution above the anchor is common to both endpoints
   of every `delta` and cancels exactly (measured: cumulative $\rho_G$ at the bottom of the grid is
   $-5.53\times10^{-7}$ rad against $-2.59\times10^{-7}$ from the anchor, and the two tables'
   `delta` agree to 5.3e-23 rad). Left alone rather than tightened with a second, physical cut,
   which would be a new constant with nothing to fix.

---

## State handed to the next prompt

**The node-range rule, and how the top is found.** The residual table is built on
`residual_node_range(model, k, z_grid, sector)` in `ComputeTargets/phase_residual.py`: the
background model's own grid (`leading_table.z_nodes`), from its lowest node up to the highest node
at which the Liouville–Green frequency still keeps `RESIDUAL_WKB_REGION_MARGIN = 0.5` of its
leading term, `leading + correction >= 0.5 * leading`, tested through the same
`*_omegaEff_sq_leading` / `*_omegaEff_sq_correction` pair the integrand uses. The search walks
**upwards from the lowest node** and stops at the first failure, so the range is the largest
retained run reaching the bottom of the grid. The bare `omega^2 > 0` rule the prompt asked for does
not work on `QCD_Cosmology` — the sign is not monotone in $z$ and a panel with two positive nodes
can hold an abscissa with $\omega^2<0$ (measured at $z=3.6118639\times10^{15}$, $k=3\times10^8$);
see log Deviation 2. The cut lies **above** the highest node at which
$|d\ln\omega/dz|/\omega\le1$ on both production models, both sectors, at $k=10^5$ and
$3\times10^8$, and `WKB_phase_function` already refuses an anchor that violates that criterion, so
no anchor a producer can accept lies outside the table. Node counts on the production grid (1,732
nodes): LambdaCDM `Gk` 1732 (nothing cut), LambdaCDM `Tk` 1113–1460, QCD `Gk` 1243–1603, QCD `Tk`
1117–1479.

**The cache and its lifetime.**

```python
from ComputeTargets.phase_residual import (
    cached_phase_residual,      # -> (CumulativeTable, reused: bool)
    residual_node_range,
    clear_phase_residual_cache, # tests only
    phase_residual_cache_size,
    RESIDUAL_WKB_REGION_MARGIN, # 0.5
    RESIDUAL_CACHE_MAX_ENTRIES, # 256
)

rho, reused = cached_phase_residual(model, k_float, leading_table.z_nodes, sector,
                                    store_id=getattr(model_proxy, "store_id", None))
```

Key: `("store", store_id, k, sector, order)` when the model has a datastore id, else
`("obj", id(model), k, sector, order)` with a `weakref` guard, because `ModelProxy.store_id` is
`None` for an unavailable model and for every offline stand-in. Lifetime: the worker process; LRU
cap 256 entries at 0.395 MB each (~40 MB for a production run's 100 keys). `z_grid` is read only
on a build, never compared on a hit. **Call it exactly as the `Gk` path does** — pass
`leading_table.z_nodes` and the proxy's `store_id`, nothing derived from the object.

**The anchor split — what prompt 07 must not undo.** `rho.delta(z_init, z)` is **not** called per
sample. Per object:

```python
rho_anchor_node = nearest_table_node(rho, z_init)      # Quadrature/integrators/WKB_phase_function.py
rho_anchor = rho.delta(z_init, rho_anchor_node)        # one Gauss panel, or exactly 0.0 on-grid
...
rho_delta = rho_anchor + rho.delta(rho_anchor_node, z) # free for an on-grid sample
```

Calling `rho.delta(z_init, z)` per sample instead costs `RHO_GAUSS_ORDER` evaluations per sample
and puts a third of the old per-object cost back.

**The new `metadata` key and the payload's size.** `metadata["rho_reused"]` is `True` when the
table came from the cache. `metadata["rho_evals"]` now means *the residual-integrand evaluations
this call spent* — the build (zero on a reuse) plus this object's anchor partial — so
`stage_1_data.RHS_evaluations == rho_evals + lead_evals` still holds. A reusing object's
`RHS_evaluations` is 4 + `lead_evals` (468 at $k=3\times10^8$ on the LambdaCDM response grid);
a building object's is ~6,900 more. Longest `json.dumps(metadata)`: **227 characters** against the
`String(256)` column, **29 characters of headroom** — `[06-metadata-column-headroom]` is still
live and `test_metadata_still_fits_the_column` now guards it. **Count before adding a key.**

**What became of `residual_nodes`.** Deleted from `WKB_phase_function.py` and replaced by
`residual_node_range` in `phase_residual.py` (Deviation 1). `nearest_table_node(table, z)` is the
new public helper in `WKB_phase_function.py`. Prompt 06's rule survives only as
`prompt_06_residual_nodes` inside `ComputeTargets/tests/test_residual_table_reuse.py`, where it is
the control.

**Measured agreement (item 3.3).** Against the per-object table, over
`RadiationModel`/`LambdaCDMModel`/`QCDModel` × $k\in\{10^5,10^7,3\times10^8\}$ in `Gk` and the two
production models in `Tk`: **$\theta$ is bit-identical at every sample of all fifteen cases**;
$\rho$ moves by at most **5.421e-19 rad** (`Gk`, QCD, $k=3\times10^8$, $z=3.661\times10^{12}$) and
**1.388e-17 rad** (`Tk`, LambdaCDM, $k=10^5$, $z=6.948\times10^{5}$). Threshold 1e-9 rad.

**Measured cost (item 3.5).** Over 50 objects of one wavenumber at $k=3\times10^8$: **142.5**
residual-integrand evaluations per object on LambdaCDM (table build 6,924) and **163.0** on QCD
(7,908) — 49× on both. The second and every subsequent object adds nothing to the build; an
on-grid anchor spends exactly 0 and an off-grid anchor exactly 4. Wall time per
`GkWKBIntegration` object at $k=3\times10^8$ on LambdaCDM over the full response grid: **0.0010 s**
against prompt 06's 0.0309 s and the ODE's 63.7 s. One table costs 44 ms (LambdaCDM `Gk`), 27 ms
(LambdaCDM `Tk`), 149 ms (QCD `Gk`), 267 ms (QCD `Tk`).

**What prompt 07 must do differently for the `Tk` sector: nothing.** `WKB_phase_function` is
shared, and `TkWKBIntegration.compute()` already calls it with `sector="Tk"` (prompt 06), so the
`Tk` tables are cached on the same key with `sector` discriminating them; the `Tk` node range and
the `Tk` anchor split are exercised by
`TestAnchoringIsFree.test_phase_is_unchanged_in_the_transfer_function_sector` and by
`TestKeyDiscriminates`. Two things to know. First, the `Tk` range **is** cut at the top (1113–1479
of the 1,732 nodes, about 1.25 e-folds inside the horizon), unlike `Gk` on LambdaCDM, so an
anchor handed to the `Tk` sector must be inside the horizon by more than that — every production
$z_{\rm init}$ is, at three e-folds. Second, the friction path is untouched:
`friction_sample[i] = friction_F.delta(z_init, z_i)` is still evaluated directly from the
background table, per sample, and is not cached or split, because `friction_F` is a per-model
table with no per-$k$ build to amortise.
