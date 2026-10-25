# Prompt 09 — `analytic_G_w` outside its domain: store `NULL`, and plot only what is sensible

**Campaign:** [`README.md`](README.md) · **Board item:** **09** (outside workstreams A–E) ·
**Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25 at `6591a8e`, on the user's decision of the same day (§0.2).
**Where it runs:** on the laptop, in a fresh-context subagent, as prompts 01–05 did.
**Closes:** `[08-the-gkwkb-reader-fails-plotting-a-full-range-green-function-on-the-a3-v2-store]`
**in code** (prompt 10 closes it by re-reading A3 v2), and
`[08-extract-gk-data-writes-the-radiation-form-into-its-w-columns]`.
**Recommended model:** **Opus**.

**Read first:**

1. This board's §3 entry `[08-the-gkwkb-reader-fails-plotting-…]` in full, including its
   **Diagnosed** paragraph, and `[08-extract-gk-data-writes-the-radiation-form-into-its-w-columns]`.
2. [`docs/spec/02-greens-function.md`](../../docs/spec/02-greens-function.md) **R2** (`:258` on):
   the constant-$w$ background, $b = (1-3w)/(1+3w)$, $a(\eta) \propto \eta^{1+b}$, and the
   Green's function that `ComputeTargets/analytic_Gk.py` evaluates. And `:61`, which names that
   file as its implementation. **Author conventions are not defects** (`CLAUDE.md`).
3. `ComputeTargets/analytic_Gk.py` (both functions), and their callers:
   `ComputeTargets/GkNumericIntegration.py:465-495` and `ComputeTargets/GkWKBIntegration.py:535-560`.
   `ComputeTargets/tests/convergence_reference.py:490-510` calls them at a model's own constant
   $w$; `docs/` scripts call them too and are records, not edited.
4. The four readers that plot `analytic_G_w`: `extract_GkWKB_data.py` (`plot_Gk`, `:140-300`),
   `extract_Gk_data.py` (`:130-300`), `extract_GkSource_data.py` (`:250-400`) and
   `extract_QuadSourceIntegral_data.py` (`:700-740`); and `extract_common.py`'s `safe_fabs`
   (`:399`) and `set_loglog_axes` (`:426`).
5. `ComputeTargets/tests/test_reader_read_only.py` and `test_main_plumbing.py`: the `ast` pattern
   for scripts that call `ray.init` at module scope. `extract_common` itself imports `ray` but does
   not start it, and `tools/tests/test_inventory_consumers.py` already imports from it.

---

## 0. What this prompt is

### 0.1 The defect

`analytic_G_w` and `analytic_Gprime_w` are a comparison aid: the constant-$w$ Bessel form of spec
02 R2, evaluated at each response point with that point's background `wBackground`. The form's
own derivation needs $1 + 3w > 0$: there $a \propto \eta^{2/(1+3w)}$ with $\eta > 0$ growing. As
$w \to -1/3^{+}$ the Bessel order $\tfrac12 + b$ diverges, and below $-1/3$ the form does not
describe the background at all. In LambdaCDM $w(z)$ crosses $-1/3$ at late times ($z \approx 0.65$
for Planck 2018), and A3 v2 stores `analytic_G_w` up to **7.57e+243** there: 293 `GkWKBValue` rows
above 1e200. `extract_GkWKB_data.py` plots it on log–log axes beside $|G|$, matplotlib proposes a
tick past the largest double, and the reader dies with `OverflowError`. The column feeds no
computation.

Separately, `extract_Gk_data.py:282-283` fills its CSV columns `analytic_G_w` and
`analytic_Gprime_w` from `analytic_G_rad` and `analytic_Gprime_rad`.

### 0.2 The user's decision (2026-10-25)

**Both fixes**: plot only where the value is sensible, **and** store `NULL` in future runs outside
the form's valid range. "Changing new store's fingerprints can never be an argument not to make a
useful fix." A3 v2 keeps what it stores; the reader fix is what reads it.

---

## 1. The compute side

In `ComputeTargets/analytic_Gk.py`:

1. **The domain.** `compute_analytic_G` and `compute_analytic_Gprime` return `None` when
   $1 + 3w \le 0$, **before** evaluating any Bessel function. Use exactly that test, with no
   tolerance band: the boundary is the derivation's, and a margin would be a tuning choice nobody
   has measured. Name it once, as a module-level predicate (for example
   `constant_w_form_applies(w)`), which both functions and the tests use.
2. **Overflow.** Inside the domain, return `None` when the result is not finite (`math.isfinite`).
   A value of exactly 0 is finite and is returned.
3. **Nothing else moves.** Inside the domain with a finite result, the value is **bit-identical** to
   the current code's: same expression, same order of operations. The return annotation becomes
   `Optional[float]`, and the docstrings cite spec 02 R2 and say when `None` is returned and why.
4. **The callers** pass the result through unchanged. Their value classes already take
   `Optional[float]`, and the stored columns (`GkWKBValue.analytic_G_w`,
   `GkNumericValue.analytic_G_w`, and the two `Gprime` columns) are already nullable. Confirm both
   by reading; change nothing there unless something is not as stated, which is a stop.
   `analytic_G_rad` is evaluated at $w = 1/3$ and is untouched.

No schema changes. New stores hold `NULL` where the form does not apply, and their fingerprints
change accordingly. That is intended (§0.2).

---

## 2. The reader side

1. **One helper** in `extract_common.py`, beside `safe_fabs`:
   `sensible_log_points(points, reference_values, decades=ANALYTIC_PLOT_DECADES)`. It returns the
   `(x, |y|)` pairs whose `y` is not `None`, is finite, is non-zero, and has $|y|$ no more than
   `10**decades` times the largest finite non-zero $|$value$|$ in `reference_values`, the quantity
   being compared against ($|G|$ on the same plot). With an empty or all-`None` reference it keeps
   every finite, non-zero point. `ANALYTIC_PLOT_DECADES = 3`, a module constant with a comment
   saying it is a plotting choice: a comparison curve more than three decades above the function it
   is compared with carries no information on a shared log axis.
2. **Every plotted `analytic_G_w` and `analytic_Gprime_w` series** in the four readers goes through
   it, with that plot's $|G|$ (or $|G'|$) as the reference. The radiation-form series may go through
   it too, for uniformity; say which you did.
3. **`extract_GkWKB_data.py` uses `safe_fabs`, not `fabs`, for every analytic column** in its plot
   points and its CSV columns (`:168-230`): with §1, new stores hold `None` there, and
   `fabs(None)` raises `TypeError`. A `None` is written to the CSV as an empty cell, as `pandas`
   writes it.
4. **`extract_Gk_data.py:282-283`**: `analytic_G_w_column` and `analytic_Gprime_w_column` are filled
   from `value.analytic_G_w` and `value.analytic_Gprime_w`.
5. Nothing else in the readers changes: no axis, label, file name or other column.

---

## 3. Tests

No Ray, no datastore. Each new test names the defect it guards.

- **`ComputeTargets/tests/test_analytic_gk_domain.py`**:
  - `None` from both functions at $w = -1/3$ (as the float `-1.0 / 3.0`, and say what
    `1 + 3w` evaluates to for it), $w = -1/3 - 10^{-12}$, $-1/2$ and $-1$;
  - inside the domain ($w = 1/3$, $0$, $0.9$, and $w = -1/3 + 10^{-3}$ where finite), values
    **bit-identical** to the expression as it stood, evaluated inline in the test from `scipy.special`;
  - `None` for an in-domain case that overflows: choose $(k, \tau_s, \tau)$ so that `yv` returns
    `inf`, and say why it does;
  - the radiation form, $w = 1/3$, unchanged: equal to the closed form of spec 02 for $b = 0$ to a
    tolerance stated with its source.
- **`ComputeTargets/tests/test_sensible_log_points.py`**: `None`, `nan`, `inf`, `-inf` and `0`
  dropped; a point at `10**3.0001` × the reference dropped and one at `10**2.9999` kept; an empty
  reference; negative values as their modulus; the order of the input kept.
- **An `ast` test** (in the same module or beside `test_reader_read_only`) that, in each of the four
  readers, every expression that builds plot points from `analytic_G_w` or `analytic_Gprime_w`
  passes through `sensible_log_points`, that `extract_GkWKB_data.py` has no bare `fabs(` applied to
  an `analytic_` attribute, and that `extract_Gk_data.py`'s CSV dict maps `"analytic_G_w"` to
  `value.analytic_G_w`.

**The breakage record.** Deliberately: (a) remove the domain test from `compute_analytic_G`; (b)
widen it to $1 + 3w < 0$; (c) make the helper keep `inf`; (d) restore one bare `fabs(` in the GkWKB
reader; (e) restore the CSV mislabel. Run the modules after each, record which named test failed,
restore. In the log.

---

## 4. Verification beyond the suite

1. **The seven roots** pass. Counts before, from the dispatch `HEAD`: record them. After: each root
   unchanged except `ComputeTargets`, up by exactly the tests added.
2. **A rehearsal reference build on the laptop**, under the rehearsal lock, by
   `RunDrivers/quadsource_atol_sweep.py --build --rehearsal` as R3 ran it (`rrk-v020-adoption` 03
   §1; `docs/rrk-v020-adoption-verification.md` §3.5), `--register r5-reference --campaign handover
   --prompt 09`, into `var/datastores/handover-09-reference.sqlite`. The rehearsal stops at
   `--zend 250`, where $w(z) > -1/3$ at every point, so §1 must change no stored value there: **its
   fingerprint must equal `39809dca…`**. That is the evidence that §1 is confined to the domain. One
   check after launch, a background wait on the pid, then `store create`, the comparison
   (`docs/dk-and-rpk-migration/rehearsal/compare_with_recorded.py` against R3's recorded
   `var/runs/rrk-v020-adoption-03-reference-20261018T232838`), `release`. Run commands with variable
   arguments through `bash`.
3. **The readers on that store**, `bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh
   var/datastores/handover-09-reference.sqlite var/handover-09/extract`: all exit 0, the store
   unchanged. (A3 v2 itself is prompt 10's, on the Mac Studio.)
4. `black --check` clean on every touched `.py` file.

---

## 5. What this prompt does not do

- It does not touch A3 v2, the Mac Studio, any schema, `ObjectFactories/`, `analytic_G_rad`, or the
  `docs/` scripts that call `compute_analytic_G`.
- It does not change any reader's other plots, axes or files, and adds no plotting for
  `analytic_G_rad`.
- It does not retire `handover-09-reference`; that is the user's, after the review.

## 6. Acceptance

1. §1: `None` outside $1 + 3w > 0$ and on overflow, bit-identical inside; callers and columns
   unchanged and confirmed nullable.
2. §2: the helper, used for every plotted `analytic_G_w`/`analytic_Gprime_w` series; `safe_fabs` in
   the GkWKB reader; the `extract_Gk_data` CSV columns correct.
3. §3's tests pass, and each deliberate breakage was caught by a named test.
4. §4: the seven roots; the rehearsal store `39809dca…`; the readers exit 0 on it.
5. One commit, in `CLAUDE.md`'s form. The log, the board and `docs/OPEN_ISSUES.md` go in it:
   `[08-extract-gk-data-writes-the-radiation-form-into-its-w-columns]` moves to §4, closed;
   `[08-the-gkwkb-reader-fails-…]` is **narrowed** ("fixed in code at <this commit>; closes when
   prompt 10 has re-read A3 v2"), not closed.

## 7. Stop conditions — stop and ask

- A value class or stored column that §1.4 says is nullable is not.
- A value inside the domain is not bit-identical to the old expression.
- The rehearsal fingerprint differs from `39809dca…`: record both and `compare_fingerprints`, and
  stop. It would mean a rehearsal point has $w \le -1/3$ or overflows, which contradicts §4.2.
- The lock is held (exit 3), or a Ray process of this checkout is alive at `acquire`.
- A test anywhere depends on `compute_analytic_G` never returning `None`.

## 8. The log and the board

`logs/09-analytic-g-w-outside-its-domain.md`, in `qcd-background-audit` README §5.1's form, with the
breakage record, the counts, and the rehearsal's fingerprint. The board: the 09 row and item; §3/§4
as §6.5 says; `docs/OPEN_ISSUES.md`, count and date by counting.
