# Prompt 06 — the producers' stored phase: store the small part, reduce the large one exactly

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-26, after 05 and 05a landed, on the user's word the same day. Its charter is
G1 Q3 (the board's Decisions): *"The noise in the producers' stored phase samples is not removed by
regenerating the store. It needs its own method, and this campaign takes it up for both
`GkWKBIntegration` and `TkWKBIntegration` after layer 1, before the next production build."* This
is row 3 of 05a's list, for $G$ and $T$ alike.

**Where it runs:** in three parts, as 05 and 05a did.
- **§1–§4, the code:** on the laptop, by a fresh-context subagent, in the main checkout at `HEAD`.
  One commit.
- **§5, rehearsals R13 and R14:** on the Mac Studio, after the code commit is pushed, under the
  lock. R13 runs in a worktree at the code commit's parent.
- **§6, the re-measurement:** on the Mac Studio, Ray-free, reading A3 v2, beside §5's builds.

**`handover` A4 may be running on the Mac Studio** in the main checkout, Ray-free, when Part B is
dispatched. Part B then runs **both** of its trees from worktrees, so the main checkout does not
move under A4 (orchestrator note §B).

The records of §5 and §6 are committed on the laptop afterwards, in 06's records commit.

**Closes, when §5 and §6 hold:** `[02-consumer-phi-below-the-storage-granularity]`
(`phase-representation`; assigned here at G1). It narrows nothing else by design.

**Recommended model:** **Opus**.

**Read first:**
1. 05's and 05a's prompts and logs, in full:
   - [`05-reanchor-g-phase.md`](05-reanchor-g-phase.md) and [`logs/05-reanchor-g-phase.md`](logs/05-reanchor-g-phase.md);
   - [`05a-reanchor-t-phase.md`](05a-reanchor-t-phase.md), §9 included, and
     [`logs/05a-reanchor-t-phase.md`](logs/05a-reanchor-t-phase.md).

   Their constructions, their two stops and their amended rule are why this prompt reads as it does.
2. The board's Decisions (G1, 05a, 05a §9, 05a T1). Then `[02-consumer-phi-below-the-storage-granularity]`
   on the `phase-representation` board, with its narrowing. Then
   [`docs/levin-phase-floor/04/issues.md`](../../docs/levin-phase-floor/04/issues.md) I2 and
   `docs/levin-phase-floor/04/scripts/check_phases.py` (`window_rms`, `:106`), whose metric §6
   repeats.
3. `Quadrature/integrators/WKB_phase_function.py`, **in full**. Every sample's phase is formed
   there as one double, `theta = -(k * leading.delta(z_init, z) + rho_delta)` (`:313`), and
   reduced with `WKB_mod_2pi` against `TWO_PI` (`:314`). Then `ComputeTargets/phase_residual.py`'s
   module docstring: $\rho$ is at most about 1.5e-3 rad.
4. The two producers' `store()`: `ComputeTargets/GkWKBIntegration.py` (`:428-595`, the offset
   `deltaTheta` at `:504` and `apply_phase_offset` at `:516`, the values after it) and
   `ComputeTargets/TkWKBIntegration.py` (`:437-609`, `deltaTheta` at `:518`,
   `apply_phase_offset` at `:532`). Then `LiouvilleGreen/WKBtools.py`:
   `WKB_mod_2pi` (`:15`) and `apply_phase_offset` (`:132`).
5. `ComputeTargets/GkSource.py`, `assemble_GkSource_values` (`:170-`), **including the rectifier**
   that repairs the $2\pi$ wraps of `deltaTheta` between neighbouring source redshifts.
6. The consumers:
   - `ComputeTargets/GkSourcePolicyData.py`, `_build_phase`;
   - `ComputeTargets/TkSourceFunctions.py`, `_build_WKB` (`:478`) and `_check_phase_sign` (`:583`);
   - `ComputeTargets/primitive_phase.py`, `build_phi_samples` (`:511`) and the constructor's
     node-anchored branch.

   Each rebuilds $\theta$ as `theta_div_2pi * TWO_PI + theta_mod_2pi` and subtracts
   `sign * k * leading.delta`, so $\varphi$ is a difference of two numbers of size $k\tau$.
7. The value tables: `ObjectFactories/GkWKBIntegration.py`, `ObjectFactories/TkWKBIntegration.py`
   and `ObjectFactories/GkSource.py` (the value columns at `:885-915`). Then `CLAUDE.md`,
   "The datastore layer", for what is this repository's to change.
8. `tools/store_value_diff.py`, and 05a's
   [`docs/levin-phase-floor/05a/rehearsal/`](../../docs/levin-phase-floor/05a/rehearsal/) and
   [`…/05a/wider/`](../../docs/levin-phase-floor/05a/wider/) scripts, which §5 and §6 reuse.

**Gate:**
- the main checkout's `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale on the laptop;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

### 0.1 The defect

Every stored phase sample loses its small part before anyone can use it.
- **The producer** forms $\theta = -(k\,L.\delta(z_{\rm init}, z) + \rho.\delta(z_{\rm init}, z))$
  as one double. $L$ is `tau` for $G$ and `cs_tau` for $T$. That double has size $k\tau$, up to
  about $10^{12}$ rad. It is reduced against the double `TWO_PI`, and the offset $\Delta\theta$ is
  then added in double (`apply_phase_offset`).
- **The small part is lost there.** $\rho$ is at most 1.5e-3 rad, and $\Delta\theta$ lies in
  $(-\pi, \pi]$. Both are exact before the sum, but in the stored sample they survive only to
  ${\rm ulp}(k\tau)$, about $10^{-4}$ rad at the top.
- **The consumer** rebuilds $\theta$ as `div * TWO_PI + mod`, a second rounding of size
  $\varepsilon k\tau$. It then subtracts `sign * k * L.delta`, a third. So
  $\varphi = \theta - {\rm sign}\,k\,L.\delta$ is quantised to ${\rm ulp}(\theta)$.
  - On A3 v2 (04's I2), production's $\varphi_G$ is exactly zero in over half its 9-sample
    windows.
  - Rebuilt in `mpmath` from the stored samples, $\varphi_G$ still carries 0.15–0.21 $F$ at the
    median, with $F = \varepsilon k\eta_R$.
  - On `QCD_Cosmology` the derivative of $\varphi$ is 3–10× worse than omitting it.
- **The stored remainder is also reduced against `TWO_PI`.** That costs the stored `G_WKB` and
  `T_WKB` values $0.18\,\varepsilon|\theta|$, smooth, as 04's I3 measured for the consumer.

05 and 05a removed rows 1 and 2 at the consumer. They could not remove this one, because the
information is gone from the samples before the consumer reads them.

### 0.2 The construction

**What is stored.** Each producer value gains one column, **`theta_residual`**:
$$r(z) = \theta(z) + k\,L.\delta(z_{\rm init}, z) = \Delta\theta - \rho.\delta(z_{\rm init}, z).$$
It is formed **from its two small terms**, never by subtracting two large numbers. $\Delta\theta$ is
the object's offset, and $\rho.\delta$ the residual table's interval. So $r$ is good to a few ulp of
a number below about $\pi$.

**The stored remainder is exact.** `theta_mod_2pi` becomes the correctly rounded remainder of the
true $\theta$ against $2\pi$, in $(-2\pi, 0]$. `theta_div_2pi` is its cycle count.
- **How it is formed:** the large term $k\,L.\delta(z_{\rm init}, z)$ comes exactly from the
  table's node limbs and partials, with $k$ as a `Fraction`. $r$ is added exactly, and the sum is
  reduced against `mpmath`'s $2\pi$ at no fewer than 40 digits (05's constant used 256 bits).
- **The same machinery as 05:** node limbs from `node_limbs`, partials from `end_partial`, and
  memoisation per node. If a helper is factored out of `PrimitivePhase` so that producer and
  consumer reduce identically, that is in scope (§1.4). A change to what `PrimitivePhase` computes
  is not.
- **No cross-sample rebase.** README §2 (e) of `GkTk-remedial` holds: two objects that share a
  physical phase store the same cycle count.

**The consumers read $r$.** None forms $\varphi$ by subtracting two numbers of size $k\tau$.
- **$T$:** the producer's anchor is the consumer's anchor `z_init`, with the opposite sign of
  $L.\delta$, so $\varphi_T(z) = r(z)$, up to the sign convention `_check_phase_sign` confirms.
- **$G$:** a `GkSource` gathers one value per source redshift $z'$ at a fixed response $z_R$. The
  value comes from the producer object for $z'$, anchored at that object's own `z_init`. With
  `_build_phase`'s leading term $-k\,\tau.\delta(z', z_R)$,
  $$\varphi_G(z') = r(z'; z_R) + k\,\tau.\delta(z_{\rm init}(z'), z'),$$
  up to the sign conventions the agent confirms.
  - **The second term is small:** zero for a pure-WKB object, where `z_init` is $z'$; and the
    numeric span's phase, tens of rad, for a numerically seeded one.
  - **It is formed exactly** from the table's limbs and partials, or as a double of that small
    size. Never as a difference of two values of $\tau$ times $k$.
- **The $2\pi$ wraps of $\Delta\theta$** between neighbouring source redshifts, which the
  rectifier repairs today on `(div, mod)`, are repaired on $\varphi_G$ by whole multiples of
  $2\pi$. That is an unwrap of a small number.
  - The rectifier's own `theta_div_2pi` output stays, for the stored rows and the readers.
  - **The two repairs must agree:** the multiples of $2\pi$ found on $\varphi_G$ equal the
    rectifier's block shifts, sample for sample. A disagreement is a stop.

**`GkSource`'s value table** carries what `_build_phase` needs: $r$ for each source redshift, and
the `z_init` it already stores. Where $\varphi_G$ is formed, in `assemble_GkSource_values` or in
`_build_phase`, is the agent's choice, recorded in the log. It is formed once.

### 0.3 What this changes, and what it does not

- **It changes the stored values of** `GkWKBIntegration`, `TkWKBIntegration`, `GkSource` and,
  through them, `GkSourcePolicyData` and `QuadSourceIntegral`. Schema changes are free
  (`CLAUDE.md`; the user's standing rule): there is nothing to backfill. An old store is read
  at the SHA that wrote it, and no reader is made version-aware.
- **It does not change** `raw_theta` or `theta_deriv`'s leading term, the background tables, the
  residual table $\rho$, the numeric producers, `QuadSource` (which reads `TkNumericIntegration`
  only), `AdaptiveLevin/`, or 05's and 05a's consumer remainders. Only the $\varphi$ samples they
  spline change.
- **It does not touch** the abscissa's rounding, `[05a-t-phase-abscissa-rounding-remains]`. That
  is in the consumer's evaluation at $z = {\rm expm1}(u)$, not in what is stored.

## 1. The code

### 1.1 The phase function
`WKB_phase_function` returns, per sample:
- the exact `(theta_div_2pi, theta_mod_2pi)`;
- the residual's two small terms: $\rho.\delta(z_{\rm init}, z)$, and enough for the producer to
  add $\Delta\theta$ exactly.

The payload names them, and its docstring says how each was formed.

### 1.2 The producers
`GkWKBIntegration.store()` and `TkWKBIntegration.store()`:
- form $\Delta\theta$ as today;
- form `theta_mod_2pi` and `theta_div_2pi` exactly, with $\Delta\theta$ inside the exact sum. This
  replaces `apply_phase_offset`'s double addition on this path;
- form `theta_residual` from its small terms;
- compute `G_WKB` and `T_WKB` from the exact remainder.

`apply_phase_offset` stays for its other callers, if any. Name them in the log.

### 1.3 The value tables
- Add `theta_residual` (`Float(64)`, nullable) to the value tables of `GkWKBIntegration`,
  `TkWKBIntegration` and `GkSource`, through their factories in `ObjectFactories/`.
- Add each to the matching value object's constructor and accessors.
- No other column changes.

### 1.4 The consumers
- **`assemble_GkSource_values`** carries $r$ through, and forms $\varphi_G$ there if that is the
  agent's choice. It also runs the agreement check of §0.2.
- **`_build_phase`** builds its $\varphi$ samples from $r$ (§0.2), not from `build_phi_samples` on
  a rebuilt $\theta$.
- **`TkSourceFunctions._build_WKB`** builds $\varphi_T$ from $r$, and `_check_phase_sign` checks
  the same convention on the new samples.
- **`build_phi_samples`** is kept, if anything still calls it, and its docstring says what it is
  now for. If nothing calls it, it is removed, and the log names where it was used.
- **A helper in `primitive_phase.py`** for the exact reduction of $k\,L.\delta(a, b) + s$ against
  $2\pi$, if factored out (§0.2). `PrimitivePhase`'s results must not move by a bit: §3 test 6.

### 1.5 `tools/store_value_diff.py`, only if needed
R13 and R14's value tables differ by a column. If the tool cannot compare two stores whose value
tables differ by an added column, extend it, with a test. It should report the column as present
in one store only, and compare every other column as before. If it can already, leave it alone,
and say so in the log.

## 2. The fixtures

Test stand-ins that model a producer's stored values (`FakeWKBValue` and the like, in
`test_tk_source_functions.py`, `test_phase_groups.py`, `test_quadsource_integral.py`,
`test_g_phase_reanchor.py`, `test_t_phase_reanchor.py`) gain the residual. It is formed as the new
producer forms it, from its small terms.
- **List every site in the log, by file and line.**
- **No existing assertion changes**, with one class of exception: an assertion that pins a value
  this prompt is meant to move, such as a stored `theta_mod_2pi`, a `G_WKB` or a $\varphi$ sample.
  Each such change is listed with its old and new value, and shown to be within the bound its
  test states. Any other change to an existing assertion is a stop.

## 3. Tests

A new module, `ComputeTargets/tests/test_stored_phase_residual.py`, on the stand-in pattern, with no
Ray and no datastore, on a real `CumulativeTable` as 05's and 05a's tests use.

1. **The stored remainder is exact.** At $k = 10^5$, $3\times10^6$, $10^8$ and $3\times10^8$, with
   `z_init` on a node and off one, over the samples of one object in each sector:
   - `theta_mod_2pi` is within $2\,{\rm ulp}(2\pi)$ of the exact remainder of the true $\theta$
     against $2\pi$, computed in `mpmath` from the same limbs, partials, $\rho$ and $\Delta\theta$;
   - `theta_div_2pi` is its exact cycle count.
2. **The residual is the small part.** `theta_residual` is within $4\varepsilon(|\Delta\theta| +
   |\rho.\delta|)$ of $\Delta\theta - \rho.\delta(z_{\rm init}, z)$ computed in `mpmath`. With the
   exact large term added back and reduced in `mpmath`, it reproduces `theta_mod_2pi`.
3. **$\varphi$ is noise-free.** Use the same objects, $k = 3\times10^8$ and $|\theta| \ge 10^{11}$
   rad, with 04's `window_rms` over windows of 9 consecutive samples.
   - $\varphi_G$ and $\varphi_T$ as the consumers now form them have window RMS at most
     $10^{-3}\,{\rm ulp}(\theta)$ in every window.
   - **Record, unscored,** the same metric for $\varphi$ formed the old way, from
     `div * TWO_PI + mod` minus `sign * k * L.delta`.
4. **The two repairs agree.** On a `GkSource` assembled from stand-in producer values whose
   `deltaTheta` wraps between neighbouring source redshifts (build one, at least two wraps):
   - the $2\pi$ multiples found on $\varphi_G$ equal the rectifier's block shifts at every sample;
   - $\varphi_G$ is continuous across them, to within the bound of test 3.
5. **The consumers.** `_build_phase` and `TkSourceFunctions` build their `PrimitivePhase` on the
   new $\varphi$. Their `raw_theta` at the sample redshifts is within
   $4\varepsilon(k|L.\delta| + |\varphi|)$ of the exact $\theta$, and 05's and 05a's tests pass.
6. **Nothing else moves.**
   - **(a)** With the same $\varphi$ samples, `PrimitivePhase`'s `raw_theta`, `theta_deriv` and
     `theta_mod_2pi` are bit-identical before and after any helper factored out of it.
   - **(b)** The background tables, $\rho$ and the numeric producers are untouched (by the diff,
     and by one bit-identity check each).
   - **(c)** `QuadSource`'s functions are bit-identical: it reads no WKB value.
7. **End to end.** Use `test_phase_groups.py`'s realistic fixture in regime `Gqr`, with both factors
   on stand-ins carrying the new residual. `evaluate_sum` against the direct product stays within
   the module's own bound.

**Deliberate breakages,** each recorded in the log with the test that catches it, then reverted:
- **(i)** the producer reduces against `TWO_PI` again: test 1 must fail;
- **(ii)** `theta_residual` formed as $\theta + k\,L.\delta$ in double: test 2 or 3 must fail;
- **(iii)** `_build_phase` builds $\varphi_G$ from `build_phi_samples` on `div * TWO_PI + mod`
  again: test 3 must fail;
- **(iv)** the $\varphi_G$ unwrap removed: test 4 must fail;
- **(v)** `TkSourceFunctions` builds $\varphi_T$ the old way: test 3 must fail.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 4. Records of the code

- `black` on the files changed.
- `logs/06-producers-store-the-residual.md`, in README §5.1's form. Its **Predictions and
  verdicts** holds §6's predictions, copied before §6 runs, marked not yet scored.
- The board's 06 row.
- **One commit.** The header's issue moves only in 06's records commit, after §5 and §6.

## 5. Rehearsals R13 and R14 — on the Mac Studio, after the code commit is pushed

As 05a §5, with what 05 and 05a learned. Records under `var/levin-phase-floor-06/rehearsal/`.
1. **The lock:** `acquire`, then `set-pid` at each launch, and `release` after the readers. Each
   lock command is a shell command of its own (`CLAUDE.md`).
2. **R13** is at the code commit's parent, in a worktree. It runs `RunDrivers/quadsource_atol_sweep.py
   --build --rehearsal` as R8–R12 ran it, registered with `--campaign levin-phase-floor --prompt 06`
   and slug `r13-parent`. Its fingerprint must equal `1b332ffa…`.
3. **R14** is the same at the code commit, in a worktree too (§ header), with slug
   `r14-residual`. Its fingerprint is **reported**.
   - It may differ from `1b332ffa…` only in the classes of §0.3, and only if the added column
     enters what the fingerprint digests.
   - Any other class whose digest differs is a stop.
4. **The value comparison:** `tools.store_value_diff` on R13 and R14, with `--json`, saved to
   `out/`. 05's `classify_non_qsi.py` is copied unchanged and recorded by SHA-256. It sorts every
   difference outside the classes of §0.3 into the four volatile kinds of
   `[05-store-value-diff-compares-volatile-columns]`.
   - **Outside** `GkWKBIntegration`, `TkWKBIntegration`, `GkSource`, `GkSourcePolicyData` and
     `QuadSourceIntegral`: every value table identical, every other difference of the four
     kinds, and no row in one store only. Anything else is a stop.
   - **Inside them,** tabulate per class:
     - `theta_residual` present in R14 only;
     - $|\Delta|$ of `theta_mod_2pi` against $\varepsilon|\theta|$, where $|\theta|$ is formed from
       the stored cycle count;
     - `theta_div_2pi` changes, which should be none, or a cycle at a remainder within one ulp of
       $0$ or $-2\pi$;
     - `G_WKB` and `T_WKB` relative changes;
     - `GkSourcePolicyData`'s changed fields (`Levin_z` among them);
     - and `QuadSourceIntegral` per row, as 05's `tabulate_qsi.py` tabulates it, copied
       unchanged.
5. **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`) on R14: every one exits 0,
   and the store is byte-identical before and after.
6. **Release the lock.** `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing.
7. **The stores are kept** until the user says otherwise.

## 6. The re-measurement — on the Mac Studio, Ray-free, reading A3 v2

05a's harness on 05's 39 rows, with the producers re-run in-process. Every number then compares
with 05's and 05a's.
- **The trees:**
  - $t_5$ is the code commit's parent, whose code is 05a's $t$: its consumers, and the stored
    producer samples;
  - $s$ is the code commit: its producers re-run on each row's objects, and its consumers.

  Both run from worktrees.
- **The harness** is 05a's, from `docs/levin-phase-floor/05a/wider/scripts/`, copied into
  `var/levin-phase-floor-06/wider/scripts/`. Change only:
  - the tree names and paths;
  - the re-run of the producers;
  - what the probe below needs.

  Save the diff against 05a's copy, and record each change as a deviation.
- **Re-running the producers.** For each row, the `GkWKBIntegration` and `TkWKBIntegration` objects
  its `GkSource` and transfer functions are built from are re-run in-process under $s$. Each is
  re-run from its stored `z_init`, initial data, `z_sample` and model, through the producer's own
  code, with the `ray.remote` stood in as 05's harness stood in `QuadSource`'s. The `GkSource` is
  then assembled from them by `assemble_GkSource_values`' own body. Nothing is written to the store.
- **The rows are 05's 39**, from 05's `sample.json` (SHA-256 `73aa7876…`), copied unchanged. "Depth-20
  row" is read as 05a read it.
- **The store is never written.** Hash every store file before and after, and check the
  fingerprint `0e32f24a…` at the end.

**The units.** One registered, detached run (slug `wider`), one checkpoint per (tree, rtol, row):
1. $t_5$ at `rtol = 1e-7`, on every row: the baseline, and each row's budget;
2. $s$ at `1e-7`, on every row;
3. $s$ at `1e-8`, on the 31 depth-20 rows.

Units 2 and 3 have a budget of 3× the row's $t_5$ time, raised from the dispatcher as before. An
over-budget unit is a result.

**The phase probe.** For 04's seven rows R1–R7, which are among the 39, and for every factor of
every Levin sub-interval: 04's `window_rms` of $\varphi$ over windows of 9 consecutive samples,
centred inside the sub-interval, under $t_5$ and under $s$. Record $F = \varepsilon k\eta_R$ for
each row, and each window's $|\theta|$.

**Gate U0, before the run:**
- `git diff --stat 1fc349c <code commit's parent> -- . ':!docs' ':!prompts'` shows `CLAUDE.md`
  and two comment-only files from `bd38e50` (`primitive_phase.py`, `test_g_phase_reanchor.py`).
  Their ASTs, docstrings removed, equal `1fc349c`'s. If another session has changed code since,
  stop and ask.
- $t_5$ at `1e-7` reproduces 05a's $t$ unit (`levin-phase-floor-05a-wider-20261026T123047`) **bit
  for bit** on one depth-20 row and one control: `total`, `total_abserr`, and every call's value
  and abserr.
- **The producer re-run is faithful.** At the **parent's** code, re-running one row's producer
  objects reproduces A3 v2's stored `theta_div_2pi`, `theta_mod_2pi`, `G_WKB` and `T_WKB` bit for
  bit. That validates the re-run before $s$ is trusted with it.
- Two in-process runs of the same $s$ unit are bit-identical.

If U0 fails, stop.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
- **S1 — $\varphi_G$'s noise goes.**
  - **The rule:** in every probe window of $G$ on R1–R7,
    $\sigma_s \le \max(10^{-3}\,\sigma_{t_5},\ 10^{-6}F)$.
  - **Holds** if true in every window.
  - 04 measured $\sigma$ at 0 at the median, quantised, and 0.46–0.78 $F$ at the maximum, for
    production's $\varphi_G$.
- **S2 — $\varphi_T$'s too.** S1's rule for every probe window of a $T$ factor.
- **S3 — the cap does not return.** On every row (all 39), under $s$ at `1e-7`, no group has a
  region at depth ≥ 18. An over-budget unit counts as false.
- **S4 — the stored noise was within the declared error.**
  - **The rule:** on every row (all 39), $|{\rm total}_s - {\rm total}_{t_5}| \le$
    `total_abserr`$_{t_5}$, at `1e-7`.
  - A failure measures what the stored noise cost a total beyond what it declared. That is a
    finding about the rows before this prompt, not a fault in it.
- **S5 — the declared error is honest after the change.**
  - **The rule:** on every one of the 31 depth-20 rows with both $s$ units in budget,
    $|{\rm total}_s(10^{-7}) - {\rm total}_s(10^{-8})| \le$ `total_abserr`$_s(10^{-7})$.
  - 05a's T4, the same rule under $t_5$, held on 29 of 31.

**Descriptive, not scored:**
- $|{\rm total}_s - {\rm total}_{t_5}|$, relative and against `total_abserr`$_{t_5}$, on every row;
- regions and maximum depth per group, $t_5$ against $s$;
- `total_converged` under both trees;
- `theta_deriv`'s change at the Levin windows: $\varphi'$ was the term 04 and
  `phase-representation` 02 found worse than omitting $\varphi$;
- `GkSourcePolicyData`'s `Levin_z` under both trees;
- `convergence_kinds.py` under $s$, as 05a ran it.

## 7. Acceptance

1. §1–§3 as written: every test passing, the five breakages caught, and the seven roots rising
   only by the tests added.
2. §5:
   - R13 ends `done` with `1b332ffa…`, and R14 `done` with its fingerprint reported as §5.3 says;
   - outside the classes of §0.3, every value table is identical and every other difference is of
     the four kinds;
   - inside them, the differences are tabulated;
   - the readers exit 0, and the lock is released.
3. §6: U0 holds; S1–S5 are each scored holds, fails or undecided against the rule as written, with
   their numbers; the store is unchanged.
4. **Records:**
   - `logs/06-producers-store-the-residual.md`, with §5 and §6's Mac Studio record appended by the
     orchestrator;
   - `docs/levin-phase-floor/VERIFICATION.md` §6 "The stored phase";
   - the records in `docs/levin-phase-floor/06/`.

## 8. Stop conditions — stop and ask

- The Gate fails.
- The construction needs a change to what `PrimitivePhase` computes, to `CumulativeTable`, to the
  background tables, to $\rho$, to the numeric producers or to `AdaptiveLevin/`.
- The two repairs of §0.2 disagree anywhere, on a fixture or in R14.
- An existing assertion has to change outside §2's exception.
- A seven-roots count falls.
- R13 does not give `1b332ffa…`, or R14's fingerprint differs outside the classes of §0.3.
- Outside those classes, a value table differs, or a column differs in a way that is not one of
  the four kinds.
- U0 fails.
- Any store file changes.
- The lock is held, or will not release.
