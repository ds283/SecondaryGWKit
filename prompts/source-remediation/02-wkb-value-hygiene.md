# Prompt 02 — WKB value, policy and label hygiene (B1, B2, B3, B4, A6, B9, B10)

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Audit:** §0.2 A6; §0.3 B1–B4, B9, B10; `TK-report.md` TK-2/3/4; `GK-report.md` GK-1/2/3/8/10;
`QS-report.md` QS-10
**Depends on:** nothing (ordering only: run after 01)
**Recommended model:** Sonnet
**Files you may touch:** `ComputeTargets/TkWKBIntegration.py`, `ComputeTargets/GkWKBIntegration.py`,
`ComputeTargets/GkSourcePolicyData.py`, `ComputeTargets/QuadSource.py` (one string), plus the log
and the status board.

---

## Character of this commit

Seven small, mutually independent edits, none of which changes a physical number. They are
bundled because each is one to three lines. **If any turns out to be larger than described, drop
it, note it in the log, and open a §3 issue** rather than letting it swell the change.

Work through them in order. Confirm each line reference against the file before editing — the
audit was written at `e9a5539` and prompt 01 does not touch these files, but check anyway.

## B1 — `TkWKBValue` analytic-oracle accessors

`ComputeTargets/TkWKBIntegration.py:630-636`:

```python
    @property
    def analytic_T_w(self) -> Optional[float]:
        return self._analytic_T_rad          # should be self._analytic_T_w

    @property
    def analytic_Tprime_w(self) -> Optional[float]:
        return self._analytic_Tprime_rad     # should be self._analytic_Tprime_w
```

The constructor stores `_analytic_T_w` / `_analytic_Tprime_w` correctly (`:551-554, 571-572`).
Fix the two accessors. Consequence to record: the `analytic_T_w`/`analytic_Tprime_w` columns of
every existing `TkWKBValue` row hold radiation values (detectable as equal to the `_rad` columns).

## B2 — `GkWKBValue` analytic-oracle accessors

Same slip, `ComputeTargets/GkWKBIntegration.py:582-588` (`analytic_G_w`, `analytic_Gprime_w`
return the `_rad` members). Fix both. Consequence: the same columns in `GkWKBValue` and, via
`GkSource.assemble_GkSource_values` (`GkSource.py:359-368`), in `GkSourceValue` for the WKB-only
region, hold radiation values. Audit GK-1 measured ~15 % difference at $w=0.2$.

## B3 — dead WKB-criterion warnings

`ComputeTargets/TkWKBIntegration.py:356-362` and `ComputeTargets/GkWKBIntegration.py:312-318`
compute `WKB_criterion_init = d_ln_omega_init / sqrt(omega_sq_init)` and warn if `> 1.0`. Every
other evaluation of this criterion takes `fabs` (`TkWKBIntegration.py:442-444, 490-492`,
`GkWKBIntegration.py:456`, `Quadrature/integrators/WKB_phase_function.py:89, 285, 662`), and the
audit measured `d ln ω/dz < 0` throughout, so both warnings are unreachable. Add `fabs(...)` to the
numerator in both places so the warning means what it says. This makes a previously silent
condition print; that is the intended behaviour, not a regression — but note in the log that
`WKB_phase_function.py:662` already raises on the same condition, so the warning is advisory.

## B4 — attribute typo

`ComputeTargets/TkWKBIntegration.py:115` and `ComputeTargets/GkWKBIntegration.py:81` initialise
`self._init_efolds_suph = None` where the property and `store()` use `_init_efolds_subh`. Rename
to `_init_efolds_subh` in both so the intended `RuntimeError("... has not yet been populated")`
fires instead of `AttributeError`. Confirm by grep that nothing reads `_init_efolds_suph`.

## A6 — `WKB_minimal` tests the wrong clearance

`ComputeTargets/GkSourcePolicyData.py:363-364`:

```python
                    "numeric_minimal": numeric_clearance > 0.0,
                    "WKB_minimal": numeric_clearance > 0.0,     # should be WKB_clearance
```

Change to `WKB_clearance > 0.0`. This can change the stored `crossover_z` (hence `Levin_z`) for a
`(k, z_response)` that falls through all nine earlier `CLASSIFICATION_BANDS`; the audit could not
determine whether any shipped configuration does. Record that; prompt 12 measures it.

## B9 — evaluate the θ-spline chunking inconsistency

`GkSourcePolicyData._classify_Levin` (`:159-172`) builds the θ spline it uses to pick `Levin_z`
with `chunk_step=None, chunk_logstep=None` (single chunk; comment: chunking risks edge effects in
the derivative), while `_create_functions` (`:662-671`) builds the θ spline that is *evaluated*
with `chunk_logstep=125`. Two acceptable outcomes:

- **Leave as-is** if you judge the single-chunk derivative is the better choice for a threshold
  test and the chunked spline the better choice for evaluation (the two uses genuinely differ:
  one needs a smooth derivative, one needs `mod 2π` precision). Then add a one-line comment at
  `:669` cross-referencing `:170` so the next reader knows it is deliberate.
- **Make them consistent** only if you can show, with a short script on a synthetic phase (e.g.
  $\theta = -\alpha(1+z)^{\beta}$ sampled on the production grid), that the `Levin_z` picked by
  the two constructions differs by more than one grid point.

Record which you did and why, as an IMPLEMENTATION CHOICE. Do not touch `phase_spline` itself.

## B10 — wrong label on the source spline wrapper

`ComputeTargets/QuadSource.py:304` constructs `ZSplineWrapper(..., "T_k", ...)`. Change the label
to `"quadratic source"` (or similar) so out-of-bounds errors name the right object. String only;
prompt 06 rewrites the surrounding function and must not be pre-empted here.

## Verification

- `PYTHONPATH=. ./venv/bin/python -c "import ComputeTargets"` succeeds.
- Existing tests still pass: `PYTHONPATH=. ./venv/bin/python -m unittest discover -s AdaptiveLevin/tests -t .`
  and `-s CosmologyModels/tests` (from prompt 01).
- For B1/B2: a three-line script constructing a `TkWKBValue`/`GkWKBValue` with distinct `_rad`
  and `_w` arguments and asserting the accessors return the `_w` ones. Paste the output in the log.
- For B3: quote one value of the criterion at a representative `z_init` (the audit's
  `TK_08_criterion_sign.py` prints them) so the log shows the sign.

## Log and commit

Log to `logs/02-wkb-value-hygiene.md`. Board: row 02; items B1, B2, B3, B4, A6, B9, B10. One
commit; the body lists the seven items in one line each and states that no physical number
changes and which stored diagnostic columns are known-bad in existing datastores.
