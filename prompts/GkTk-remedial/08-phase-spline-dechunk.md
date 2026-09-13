# Prompt 08 — Remove chunking from `phase_spline`, keeping its signature

**Campaign:** [`README.md`](README.md) · **Reconciliation:** [`RECONCILIATION.md`](RECONCILIATION.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Review sections:** §5 (all), §9 "Does `phase_spline` chunking achieve anything? No."
**Design facts:** README §2 (g); decision §7 D4.
**Depends on:** nothing (independent). Recommended before 09 and 10.
**Recommended model:** Sonnet
**Files you may touch:** `LiouvilleGreen/phase_spline.py`, new `LiouvilleGreen/tests/test_phase_spline.py`,
plus the log and the status board.
**Do not touch:** any caller. `bessel_phase.py`, `GkSourcePolicyData.py`, `TkSourceFunctions.py`
and the test fixtures that pass `chunk_logstep=125` must keep working **without edits** — that is
this prompt's acceptance test, and `bessel_phase.py` belongs to another campaign (README §0.2).

Read first: README §2 (g), §0.2, §7 D4; `RECONCILIATION.md` §1 items 5, 6, §2 item 13; review §5.

---

## 1. What the review measured (why this is deletion, not redesign)

Consumer geometry (`GkSourcePolicyData._create_functions`), exact radiation, 100 and 300 per
decade (review §5 table): interpolation error is **bit-for-bit identical** with one chunk and two
($h^4x/384$: $8.26\times10^{-5}$ at $k=10^6$, $8.26\times10^{-3}$ at $10^8$); with two chunks the
spline ordinates reach $6.44\times10^8$ rad against a $9.99\times10^6$ rad data span (rebased to the
chunk's far boundary), knot residuals worsen from $9.3\times10^{-9}$ to $2.7\times10^{-7}$, and the
hard switch between chunks introduces a $-1.4\times10^{-4}$ rad jump in $\theta$ and
$3.3\times10^{-8}$ relative in $\theta'$ (100/decade, $k=10^8$). The `MINIMUM_SPLINE_DATA_POINTS`
merge removes any bound on a chunk's span, so "≤125 cycles per chunk" was never true. And
`_build_log_chunks_positive` cannot make progress for `chunk_logstep < 2` starting at cycle 1.

None of this is a precision *mechanism*: a single global rebase already bounds ordinates by the
data span. Chunking goes. **The $h^4x/384$ error does not go with it** — that is prompts 09 and 10.

## 2. What to build

`LiouvilleGreen/phase_spline.py`:

1. **One spline.** `phase_spline.__init__` keeps its signature exactly:
   `(x_sample, div_2pi_sample, mod_2pi_sample, x_is_log=False, x_is_redshift=False,
   chunk_step=DEFAULT_CHUNK_SIZE, chunk_logstep=None, increasing=True)`. `chunk_step` and
   `chunk_logstep` are **accepted and ignored** (docstring: deprecated, no effect since this
   commit, retained so that existing callers need not change). Build a single `_chunk_spline`
   (rename to `_rebased_spline` if you like; keep it private) over all the data, rebased to the
   `div_2pi` of the sample with the **smallest** $|{\rm div}|$ — or the median; state the choice
   — so that ordinates are bounded by the data span, never inflated.
2. **Delete** `_build_linear_chunks`, `_build_log_chunks`, `_build_log_chunks_positive`,
   `_build_log_chunks_negative`, `_match_chunk`, `MINIMUM_SPLINE_DATA_POINTS`, the chunk merge
   logic, and the chunk-order checks. `num_chunks` returns `1` (`QuadSourceIntegral.py:912-914`
   reads it via `getattr` into metadata; the persisted `WKB_phase_spline_chunks` becomes 1).
3. **Keep** the public methods and their semantics: `raw_theta(x, x_is_log=False)`,
   `theta_mod_2pi(x, x_is_log=False)`, `theta_deriv(x, x_is_log=False, log_derivative=False)`;
   the range checks with `SPLINE_TOP_BOTTOM_CUSHION` and the safe-range warnings; `min_x`/`max_x`
   attributes if any consumer reads them (grep). `increasing` may keep ordering the data or become
   a no-op; state which.
4. Module docstring: what the object is (a cubic spline of a globally rebased phase), what it is
   **not** (a cure for the $h^4x/384$ growth — cite review §5 and point at `PrimitivePhase`, which
   prompt 09 adds), and the deprecation of the chunk arguments.

## 3. Tests (`LiouvilleGreen/tests/test_phase_spline.py`) — the module's first

1. **Interpolation law.** Exact radiation phase samples ($\theta=k(1/s_i-1/s)$ reduced with
   `WKB_mod_2pi`) on 100/decade, $k=10^6$, $s\in[10,10^4]$, `x_is_log=True, x_is_redshift=True,
   increasing=False` (the `GkSourcePolicyData` geometry): interior max error at 10 points per
   interval against the exact phase between $7\times10^{-5}$ and $1\times10^{-4}$ rad (review §5:
   $8.26\times10^{-5}$; predicted $7.3\times10^{-5}$). This pins the law so that prompt 09 can show
   it beating it.
2. **Ordinates bounded by the span**: the spline's internal `y` values have
   $\max|y|\le$ (data span in rad) $+2\pi$.
3. **No switch**: `theta_deriv` is continuous — finite-difference the spline derivative across every
   interval boundary and assert no jump larger than the interior variation (review: the two-chunk
   build had $3.3\times10^{-8}$ relative at the switch).
4. **Signature compatibility**: constructing with `chunk_logstep=125` and with `chunk_step=200` gives
   objects **identical** in `raw_theta` at 50 points to the `chunk_logstep=None, chunk_step=None`
   build; `num_chunks == 1` for all three.
5. **Both orientations**: `increasing=True` (the `TkSourceFunctions` geometry, $\theta$ decreasing
   towards low $z$) and `increasing=False` both interpolate to the same law.
6. **Range behaviour unchanged**: evaluation just inside the cushion returns the boundary value;
   outside raises `RuntimeError` — replicate the current behaviour by reading the code before
   deleting anything, and assert it.
7. **The old progress defect cannot recur**: construct with `chunk_logstep=1.5` and data starting
   at cycle 1 — must return promptly (a regression guard for the deleted code path).

## 4. Verification and acceptance

- `PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_phase_spline -v` passes.
- **Callers untouched and passing:** `git diff HEAD~1 --stat` shows only the two allowed files, the
  log and the board; and
  ```bash
  PYTHONPATH=. ./venv/bin/python -m unittest LiouvilleGreen.tests.test_bessel_phase -v
  PYTHONPATH=. ./venv/bin/python -m unittest ComputeTargets.tests.test_phase_groups ComputeTargets.tests.test_tk_source_functions ComputeTargets.tests.test_gk_source_policy ComputeTargets.tests.test_quadsource_integral -v
  ```
  all pass. (`test_3bessel_analytic.py` and `test_three_bessel.py` are very slow — the
  `transfer-remedial` board §5 note 13 — run `test_three_bessel` if it finishes in ten minutes,
  otherwise record that it was not run.)
- `grep -n "_match_chunk\|_build_log_chunks\|MINIMUM_SPLINE_DATA_POINTS" LiouvilleGreen/phase_spline.py`
  empty.
- `black --check` clean.

## 5. Log and commit

"State handed to the next prompt": the rebase choice; the measured interpolation error of test 1
(prompt 09 quotes it as the number it beats); the list of attributes kept for consumers.

Commit subject, or something equally specific: `Drop the chunked phase spline in favour of one rebased spline`.
