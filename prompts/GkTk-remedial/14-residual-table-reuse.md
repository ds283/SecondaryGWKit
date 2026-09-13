# Prompt 14 — Build the phase residual once per $(model, k, sector)$

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Covers:** §3 `[06-residual-table-per-object]`
**Review sections:** none new — §6 (the residual and its size) and §7 (the primitive) are the
standing background; this prompt changes *when* the residual table is built, not what it is.
**Design facts:** README §2 (a), (c); §5 standing note 3.
**Depends on:** 05, 06.
**Recommended model:** Opus
**Files you may touch:** `Quadrature/integrators/WKB_phase_function.py`, `ComputeTargets/phase_residual.py`
(**additively only** — `build_phase_residual`'s existing signature and behaviour must not change),
new `ComputeTargets/tests/test_residual_table_reuse.py`, `ComputeTargets/tests/test_gk_wkb_phase.py`
(**only** the tests that pin the per-object anchoring convention — see §3.1), plus the log, the
status board and `docs/OPEN_ISSUES.md`.
**Do not touch:** `ComputeTargets/GkWKBIntegration.py`, `TkWKBIntegration.py`, `main.py`,
`LiouvilleGreen/WKBtools.py`, `BackgroundModel.py`, `cumulative_table.py`, the Datastore factories,
`ComputeTargets/tests/test_phase_residual.py`.

**Out of sequence.** This prompt is numbered 14 because the campaign's numbers are append-only,
but it runs **between 06 and 07**: prompt 07 inherits the same producer and would otherwise double
the exposure. Note this in your log.

Read first: README §2 (a), (c), §4.3 stop conditions, §5, §6; `RECONCILIATION.md` §1 item 11,
§2 item 5; `IMPLEMENTATION_STATE.md` §3 `[06-residual-table-per-object]` and §5 notes 2, 3, 12;
`logs/05-…` and `logs/06-…` "State handed to the next prompt".

---

## 1. What is wrong

`WKB_phase_function` calls `build_phase_residual(model, k_float, nodes, sector)` on **every
object** (`WKB_phase_function.py:234-235`), although the residual
$\rho=\int C/(\omega+\omega_0)\,dz$ depends only on $(model, k, sector)$. Measured by prompt 06:
5,536 of the 6,000 integrand evaluations and ~92 % of the 0.031 s per object at $k=3\times10^8$ on
LambdaCDM; 5,088–6,900 evaluations and 82–275 ms per table on `QCD_Cosmology` (log 05). Across
~1,700 source redshifts per $k$ that is roughly 0.5 min (LambdaCDM) to ~4 h (QCD) of repeated
identical quadrature per model per run.

It is not shared today because of prompt 06's deviation 4: the table is built with the object's own
$z_{\rm init}$ as its **top node**, and $z_{\rm init}$ — the numeric stop point, a `root_scalar`
root — differs per object (`RECONCILIATION.md` §2 item 5). So the fix is not memoisation alone:
the table must be anchored at a **fixed node range per $(model, k, sector)$**, and $z_{\rm init}$
reached through `CumulativeTable.delta`'s off-grid partial — which is precisely the term that
accessor exists for.

This is a cost defect, not a correctness defect. **Nothing about the phase's value may change
beyond the last few digits.**

## 2. The change

1. **Anchor the residual table on the background grid, not on the object.** Replace the
   per-object `residual_nodes(grid, z_sample, z_init)` convention with a node set that depends
   only on $(model, k, sector)$ and covers every object of that $k$: the background model's own
   grid, restricted at the top to the highest node at which $\omega^2(k,z)>0$ and at the bottom to
   the lowest grid node. Two constraints you must respect, both already recorded:
   - **The integrand raises `ValueError` where $\omega^2\le0$** (log 05). Above horizon crossing
     that is the case for the smaller $k$, so the top of the range is $k$-dependent and must be
     found, not assumed. State how you find it.
   - **`CumulativeTable` refuses an evaluation more than one grid interval beyond either end**
     (log 03). Every object's $z_{\rm init}$ and every sample lies strictly inside this range, so
     `delta` only ever forms interior partials — but assert it rather than assume it.

   Every production sample is already a node (`RECONCILIATION.md` §1 item 11), so no sample needs
   adding. Say in the log what you did with `residual_nodes`: keep it with the new semantics,
   replace it, or delete it. Whichever you choose is an `IMPLEMENTATION CHOICE` to record.

2. **Build once and reuse.** Cache the table on $(model, k, sector)$. The mechanism is yours —
   a module-level dict in the Ray worker is the obvious one — but:
   - **The key must not collide between models.** `ModelProxy.store_id` is `None` for an
     unavailable (offline, test) model (`BackgroundModel.py:963`), so a `None` store id must not
     be used as a cache key for two different stand-ins in one process. Handle this explicitly;
     a test must cover it.
   - **Bound the memory, or show it does not need bounding.** Quote the per-table footprint and
     the worst-case total for a production run (50 $k$ × 2 sectors × both models).
   - The cache must be transparent: two calls that differ only in $z_{\rm init}$ or `z_sample`
     must return the same table object, and a call with a different $k$, sector or model must not.

3. **Report the reuse in the payload.** `metadata` gains a key recording whether the table was
   built or reused on this call (name it yourself), and `stage_1_data.RHS_evaluations` must then
   count only the integrand evaluations **this call actually spent** — a reused table spends none.
   Say in the log what a reusing object's `RHS_evaluations` now is.
   `[06-metadata-column-headroom]` is live: the `metadata` column is `String(256)` and prompt 06's
   JSON is already 206–233 characters. **Count the new payload's length and report it**; if adding
   the key would exceed 256, say so and stop rather than silently overflowing.

## 3. Tests

### 3.1 `test_gk_wkb_phase.py` — what may change and what may not

You may edit **only** the tests that pin the per-object anchoring convention — at minimum
`TestOffGridAnchor.test_residual_nodes_put_the_anchor_on_top`, and the `metadata` key expectations
if item 2.3 changes them. **Every accuracy, cost and sweep test must pass with its thresholds
unchanged**, and `git diff` of that file must show nothing else. In particular these must still
hold, at the same numbers prompt 06 measured:

| | Threshold | Prompt 06 measured |
|---|---|---|
| radiation span $10^7$ / $10^9$ rad | ≤1e-8 / ≤1e-6 | 3.7253e-09 / 3.5763e-07 |
| LambdaCDM $k=10^5$ / $3\times10^8$ | ≤1e-5 / ≤5e-3 | 1.1921e-07 / 9.7656e-04 |
| QCD $k=3\times10^8$ | ≤5e-3 | 9.7656e-04 |
| off-grid anchor (37 % through an interval) | ≤1e-8 | 3.7253e-09 |
| cost per object, $k=3\times10^8$ | ≤0.05 s | 0.0309 s |
| cross-object sweep, 990 objects | 0 rebase offsets; jumps = transitions | 90 = 90 |

`test_phase_residual.py` must pass **unchanged** (`git diff HEAD~1 --stat`).

### 3.2 `test_residual_table_reuse.py` (new)

1. **The table is shared.** For one $(model, k, sector)$, build phases for several objects with
   *different* $z_{\rm init}$ (including one off-grid at 37 % through an interval) and different
   `z_sample`, and assert the residual table was built once and reused thereafter — by the
   metadata key of item 2.3 and by the integrand-evaluation count.
2. **The key discriminates.** A different $k$, a different sector, and a different model each
   force a rebuild. Include the `store_id is None` stand-in case of item 2.1: two distinct
   offline models must not share a table.
3. **The answer did not move.** For each of `RadiationModel`, `LambdaCDMModel` and `QCDModel`,
   three $k$, compare $\theta$ from the shared table against $\theta$ from a table built the old
   way (anchored at $z_{\rm init}$): agreement ≤ **1e-9 rad** at every sample. Quote the measured
   maximum and its $(model, k, z)$. This is the test that the anchoring change is free; 1e-9 is
   two orders below the $\rho$ acceptance of README §6 and far below the $\varepsilon k\tau$ floor.
4. **Radiation control survives.** $\rho_G$ is still bit-exactly zero through the shared table.
5. **Cost.** The amortised integrand evaluations per object over $N$ objects of one $k$: report
   the measured figure for $N=50$ on LambdaCDM and on QCD, and assert the second and subsequent
   objects spend **zero** residual-integrand evaluations.

## 4. Verification and acceptance

- `test_residual_table_reuse.py` and `test_gk_wkb_phase.py` pass; `discover -s ComputeTargets/tests -t .`
  passes; `LiouvilleGreen.tests.test_range_reduce`, `test_bessel_phase` pass.
- Every row of the §3.1 table met at its stated threshold.
- `black --check` clean.
- `[06-residual-table-per-object]` moved to §4 (Resolved) on the board, with the measured
  before/after, and its row deleted from `docs/OPEN_ISSUES.md` with the count corrected.

## 5. Log and commit

"State handed to the next prompt", verbatim: the new node-range rule and how the top is found;
the cache key and its lifetime; the new `metadata` key and the payload's character count; what
`residual_nodes` became; the measured agreement of item 3.3; the amortised cost of item 3.5; and
**what prompt 07 must do differently**, if anything, to get the same reuse for the `Tk` sector.

Commit subject, or something equally specific: `Build the WKB phase residual once per wavenumber`.
