# Prompt 05 — layer 1(a): take $k\tau_R$ out of $G$'s phase

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-26, on the user's answers at gate G1 (the board's Decisions):
- **Q1:** H-anchor;
- **Q2:** layer 1(a), with the large constant reduced against the true $2\pi$ and a wider
  re-measurement;
- **Q3:** it lands here, consumer-only, as early as possible;
- **Order:** no fingerprint change. The fingerprint certifies identities, tags and value-row counts,
  never values, so this prompt certifies its values directly (`CLAUDE.md`, campaign conventions
  rule 7) with a tool it adds, `tools/store_value_diff.py` (§3A).

**Where it runs:** in three parts.
- **§1–§4, the code and the comparison tool:** on the laptop, by a fresh-context subagent, in the
  main checkout at `HEAD`. One commit.
- **§5, rehearsals R9 and R10:** on the Mac Studio, orchestrated there, after the code commit is
  pushed, under the lock.
- **§6, the wider re-measurement:** on the Mac Studio, Ray-free, reading A3 v2. It may run beside
  §5's builds, since it takes no lock.

The records of §5 and §6 are committed on the laptop afterwards, in 05's records commit.

**Closes, when §5 and §6 hold:** `[00-consumer-anchoring-floor]` (`GkTk-remedial`), for $G$.
**Narrows:**
- `[04-g-phase-is-reduced-against-the-double-two-pi]`, for $G$'s large constant;
- `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` (`handover`).

**Recommended model:** **Opus**.

**Read first:**
1. The board's Decisions (G1) and §2; [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md)
   §3 and §4; `docs/levin-phase-floor/04/issues.md` I2, I3 and I7.
2. [`LiouvilleGreen/range_reduce_mod_2pi.py`](../../LiouvilleGreen/range_reduce_mod_2pi.py)'s module
   docstring: why a remainder taken against `TWO_PI` is a remainder against the wrong modulus.
3. `ComputeTargets/primitive_phase.py`, in full: `raw_theta` (`:276`), `theta_mod_2pi` (`:285`),
   `theta_deriv` and `build_phi_samples` (`:359`).
4. `ComputeTargets/cumulative_table.py`: the module docstring, the node limbs `_hi_list`/`_lo_list`
   (`:172-173`), `_locate` (`:336`), `_integral` (`:293`) and `delta` (`:360-401`).
   `BackgroundModel.TablePrimitive` (`ComputeTargets/BackgroundModel.py:638-670`) exposes the table
   as `.table`.
5. `ComputeTargets/GkSourcePolicyData.py`: `_build_phase` (`:145-195`) and its two callers
   (`:237`, `:726`).
6. Every consumer of a $G$ remainder:
   - `ComputeTargets/phase_groups.py:542-549`, the composition;
   - `ComputeTargets/spline_wrappers.py:160-172`, `GkWKBSplineWrapper`, which the readers use;
   - `AdaptiveLevin/levin_quadrature.py:1106-1123`, where the driver takes `sin`/`cos` of a
     remainder.
7. 04's harness, `docs/levin-phase-floor/04/scripts/phases.py`. Arm X there computed exactly what
   this prompt builds for $G$, at runtime in `mpmath`.

**Gate:**
- the main checkout's `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale;
- the seven roots' counts are taken at the dispatch `HEAD` (`CLAUDE.md`).

---

## 0. What this prompt is

### 0.1 The defect

$G$'s phase is $\theta_G(z) = -k\,\Delta_\tau(z, z_R) + \phi(u)$, with $u = \log(1+z)$.
`PrimitivePhase.theta_mod_2pi` forms that as **one double**, of size about $k\eta_R$ (up to
$\sim10^{12}$ rad), and only then reduces it. So every remainder carries rounding of about
$\varepsilon k\eta_R = F$, and that rounding is not smooth in $z$.
- 03 found it in every scored group, at 0.22–0.34 F.
- 04 removed it and the cliff went (R1). Removing it from $G$ alone did as much (A1, and I7).
- The reduction is also against `TWO_PI`, which costs a smooth 0.175 F (04's I3).

### 0.2 The construction

**The aim.** No double of size $k\eta_R$ is ever formed on the way to $G$'s remainder.
`CumulativeTable.delta` is a node term plus two end partials:
$$\Delta_\tau(z, z_R) = \underbrace{\big[T(n_R) - T(n_z)\big]}_{\text{node term, from the limbs}} + p_R - p_z,$$
where $n_z$ is the table node `_locate` assigns to $z$, $p_z$ is the local Gauss partial from $z$
to $n_z$ (zero on a node), and likewise at $z_R$. Then:

- **The constant.** For each node $n$, the remainder
  $$c_n = \Big( -k\,\big[T(n_R) - T(n) + p_R\big] \Big) \bmod 2\pi$$
  is computed **once**, in extended precision. The node term is taken exactly from the four limbs,
  $k$ is multiplied in exactly, and the reduction is against $2\pi$ itself (for instance `mpmath`
  at no fewer than 40 digits, with `mpmath.pi`). The result is rounded to a double in
  $(-2\pi, 0]$. It is computed lazily and memoised per node. $p_R$ is a constant of the object.
- **The variable part.** $v(z) = k\,p_z + \phi(u)$. It is small: at most $k$ times half a table
  interval of conformal time, plus O(1) rad. It is formed in double and reduced with
  `WKB_mod_2pi`. Its modulus error is about $0.18\,\varepsilon|v|$, which is negligible.
- **The remainder** is $c_{n_z} + {\rm rem}(v(z))$, wrapped once into $(-2\pi, 0]$ by an exact add
  or subtract of `TWO_PI`. A single subtraction of a double close to $2\pi$ from a sum in
  $(-4\pi, 0]$ is exact by Sterbenz's lemma, so the wrap costs nothing.

This is 04's arm X for $G$, precomputed per node instead of evaluated in `mpmath` at every call.
It also reduces against $2\pi$, where X kept `TWO_PI`. **It is layer 1(a):** the constant
$k\tau_R$, and every other large constant, leaves the double before it is rounded.
- **Residual jitter:** about $\varepsilon k|p_z|$. 04 measured the equivalent in X at no more
  than $4\times10^{-8}F$.
- **The switch between nodes.** At the midpoint between two nodes, $n_z$ changes. The two
  representations differ there by rounding alone, about $10^{-15}$ rad. That is far below anything
  the driver resolves; 04's E2 bounds the same effect.

### 0.3 What it does not change

- **$T$'s phase** is unchanged, by G1 Q2. Every `TkSourceFunctions` remainder must stay
  bit-identical.
- **`raw_theta` and `theta_deriv`**, of $G$ and $T$ alike, are unchanged bit for bit.
  `raw_theta` stays a double of size $k\eta_R$. It is used for the driver's total-variation gate
  and for diagnostics, where that size is harmless.
- **The producers, any schema, any stored value of any class except `QuadSourceIntegral`'s, and
  `AdaptiveLevin/`** are unchanged. **The noise in $\phi$'s stored samples**
  (`[02-consumer-phi-below-the-storage-granularity]`) is untouched: it is G1's later producer
  work.
- **No re-anchoring of $T$**, even where §6 shows it would help. That is a finding for the
  board, not a change for this prompt.

## 1. `CumulativeTable`

An interval accessor that keeps the node term exact. For example,
`delta_parts(z_a, z_b) -> (j_a, j_b, ends)`, with `ends = p_b - p_a` as `delta` forms it, and
`node_limbs(j) -> (hi, lo)`. The name and shape are your choice. It must satisfy two conditions:
- `delta(z_a, z_b)` is unchanged bit for bit, and is `table + ends` with `table` formed as now;
- the exact node term, $(\text{hi}_b - \text{hi}_a) + (\text{lo}_b - \text{lo}_a)$ evaluated in
  `mpmath` or as a `Fraction`, rounds to `delta`'s `table` term to within one ulp.

`TablePrimitive` passes the accessor through.

## 2. `PrimitivePhase`

An opt-in, node-anchored remainder, as §0.2 gives it. For example, a constructor keyword
`exact_remainder=False`. **The default reproduces today's behaviour bit for bit.** With the keyword
on, `theta_mod_2pi` returns $c_{n_z} + {\rm rem}(v(z))$, wrapped into $(-2\pi, 0]$. It needs the
leading primitive's table, through §1's accessor.
- Without the accessor it raises. It does not fall back.
- The abscissa is taken through `_get_x` and `_clamped_u` exactly as `raw_theta` takes it.
- `sign` enters $c_n$ and $v$ as it enters `raw_theta`. The construction is written for any sign,
  and it is tested at both.

`_build_phase` in `GkSourcePolicyData` turns it on for $G$. `TkSourceFunctions` does not.

Update the module docstring's "Anchoring, and the floor it implies" to say what is now true for
$G$, what remains for $T$, and why, with the G1 citation.

## 3. Tests

Use the stand-in pattern of `test_primitive_phase.py` and `test_cumulative_table.py`, with no Ray
and no datastore.
1. **The accessor.** On a table built as `functions.tau` is built, at 1000 random pairs of
   abscissae, on and off nodes: `delta` is bit-identical to before (compare against a copy of
   today's formula, written into the test), and the exact node term rounds to `table` within one
   ulp.
2. **Exactness.** At $k = 10^5$, $3\times10^6$, $10^8$ and $3\times10^8$, with $z_R$ at
   $0.1$ and $30$, over abscissae spanning the WKB range: the remainder is within
   $4\varepsilon\,(k|p_z| + 2\pi)$ of the exact remainder of $\theta$ **against $2\pi$**. The
   reference is computed in `mpmath` from the same limbs and the same $\phi$.
3. **No jitter.** On a region of width $10^{-6}$ in $u$ at $z' \sim 10^{10}$, $k = 3\times10^6$:
   use 04's $\sigma_\theta$ (65 points, `numpy.unwrap`, degree-4 fit, RMS residual).
   - With the keyword off it is at least $0.1\,\varepsilon k\eta_R$.
   - With it on it is at most $10^{-3}$ of that.
4. **Nothing else moves.**
   - `raw_theta` and `theta_deriv` are bit-identical with the keyword on and off.
   - A `TkSourceFunctions` phase built as production builds it has bit-identical remainders before
     and after this prompt.
   - Both signs are tested.
5. **The default is today's.** With the keyword off, `theta_mod_2pi` is bit-identical to
   `WKB_mod_2pi(raw_theta)[1]`.
6. **End to end.** Use `test_phase_groups.py`'s realistic fixture at one cheap configuration, with
   $G$'s phase built as production builds it:
   - `evaluate_sum` against the direct product $G f/H^2$ stays within the bound the module's
     existing test states;
   - the composed remainder differs from the old one by at most the old floor
     ($4\varepsilon k\eta_R$) plus $10^{-14}$.

**Deliberate breakages,** each recorded in the log with the test that catches it, then reverted:
- **(i)** reduce $c_n$ against `TWO_PI` instead of $2\pi$: test 2 must fail at $k = 3\times10^8$;
- **(ii)** form the node term from `delta` (a double) instead of the limbs: test 3 must fail;
- **(iii)** turn the keyword on for $T$ as well: test 4 must fail.

**The seven roots,** before and after. A fall is a stop. A rise equals the tests added.

## 3A. The comparison tool, `tools/store_value_diff.py`

`CLAUDE.md`'s campaign conventions rule 7 asks for a tool, and this prompt is the first that needs
it. It compares the **values** of two closed stores, matched by **physical identity**: the
inventory key `datastorekit.store_inventory` builds for each row, with no serial in it. It never
matches by serial. This repository's code, and no change to `datastorekit`.
- **Read only.** It opens both stores as the readers and `store fingerprint` do, with no Ray.
- **What is compared, per class:**
  - every column of the row itself, except its serial, its timestamp and the columns that name
    another row. A column naming another row is compared through that row's identity, which the
    key already holds;
  - every row of each of its value tables, matched by its parent's identity and by the identity
    of every other row it names (a value row's redshift, for example), never by serial.
- **How values are compared:** floats bit for bit (`float.hex`), with `NaN` equal to `NaN` and
  `+inf` to `+inf`; JSON columns canonicalised as JSON; everything else by `==`.
- **What it reports:**
  - per class: rows identical, rows that differ, and rows present in one store only;
  - per differing column: the count, and for floats the largest $|\Delta|$ and $|\Delta|/|a|$;
  - for a JSON column, the keys that differ.

  It writes text for a person and, with `--json`, a machine-readable form. It exits 0 when the
  stores are value-identical and 1 otherwise. `--classes` restricts the classes it compares.
- **A class whose value rows cannot be matched by identity** is reported as such, by name. It is
  never compared by serial instead. If a production class is such a class, stop and ask.
- Run as `PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff A.sqlite B.sqlite`.

**Its tests**, in `tools/tests/test_store_value_diff.py`, on stores built by `StoreFixtures`
(`real_store_fixtures`), with no Ray:
- **(t1)** two builds of the same fixture are reported identical, and the exit code is 0;
- **(t2)** one float in a copy changed by one ulp, in a row's own column and then in a value-table
  row, is reported precisely: class, identity, column and $\Delta$;
- **(t3)** a row deleted from a copy is reported as present in one store only;
- **(t4)** a copy whose serials are renumbered, with the same identities, is reported identical.
  This is the test that it never matches by serial;
- **(t5)** a JSON column with the same content in a different key order is identical, and a changed
  value in it is reported by key.

**One more deliberate breakage, (iv):** match by serial instead of identity. Test (t4) must fail.
Record it in the log, then revert it.

The `tools` root is one of the seven; its count rises by these tests.

## 4. Records of the code

- `black` on the files changed.
- `logs/05-reanchor-g-phase.md`, in README §5.1's form. Its **Predictions and verdicts** section
  holds §6's predictions, copied before §6 runs.
- The board's 05 row.
- **One commit.** The issues of the header move only in 05's records commit, after §5 and §6.

## 5. Rehearsals R9 and R10 — on the Mac Studio, after the code commit is pushed

Orchestrated there, as `handover` 12's R8 was, with the records under `var/levin-phase-floor-05/`.
**A fingerprint certifies identities, tags and value-row counts, never values** (`CLAUDE.md`,
campaign conventions rule 7). So this pair is compared value by value, with §3A's tool.
1. **The lock:** `acquire`, then `set-pid` at each launch, and `release` after the readers. Each
   lock command is a shell command of its own (`CLAUDE.md`).
2. **R9** is at the code commit's parent, in a worktree. It runs `RunDrivers/quadsource_atol_sweep.py
   --build --rehearsal` as R8 ran it, registered (`--campaign levin-phase-floor --prompt 05`, slug
   `r9-parent`). Its fingerprint must equal R8's, `1b332ffa…`: it reproduces R8 at the same code.
3. **R10** is the same at the code commit (slug `r10-reanchor`). Its fingerprint must also equal
   `1b332ffa…`, because identities and counts are unchanged.
4. **The value comparison:** `python -m tools.store_value_diff` on R9 and R10, at the code commit,
   with `--json`, saved to `out/`.
   - **Every class other than `QuadSourceIntegral` is reported value-identical**, with no row in
     one store only.
   - In `QuadSourceIntegral`, from the tool's JSON and a short script of your own in `scripts/`,
     tabulate per row:
     - $|\Delta{\rm total}|/|{\rm total}|$ and $|\Delta{\rm total}|/$`total_abserr`;
     - the changes in `total_converged`, regions and maximum depth;
     - which metadata keys differ.
5. **The readers** (`docs/dk-and-rpk-migration/rehearsal/run_readers.sh`) on R10: every one exits 0,
   and the store is byte-identical before and after.
6. **Release the lock.** `ray-processes --ray-temp-dir /tmp/ray-sgk` lists nothing.

## 6. The wider re-measurement — on the Mac Studio, Ray-free, reading A3 v2

04 measured seven mid-band rows. G1 asked for a wider sample before G2.

**The harness** is 04's (`docs/levin-phase-floor/04/scripts/`, route (a)), run at `HEAD`, in two
trees: the code commit's parent ($p$) and the code commit ($c$).
- **`QuadSource` is recomputed in-process.** B1's strict check refuses A3 v2's stored `QuadSource`
  rows at this tree. So each row's `QuadSource` is computed in-process from the stored $T_k$ rows,
  as `main.py` would compute it at this tree, with no Ray.
  - If that cannot be done without a production change, stop.
- **Everything else** is rebuilt from the store as 03 and 04 rebuilt it.
- **The store is never written.** Hash every store file before and after, and check the
  fingerprint `0e32f24a…` at the end.

**The rows.** Draw them from `var/levin-phase-floor-01/out/groups_01.csv` on the Mac Studio, by a
rule fixed in `state/sample.md` before any row runs.
- **Depth-20 rows.** In each cell of {top group `G+r`, `G−r`, lone `G`, `Gqr`} × {low band, mid
  band}, up to 3 rows: those of smallest stored `compute_time` at distinct $k$, with
  `compute_time` ≤ 900 s. That is at most 24 rows. Also 03's seven rows, R1–R7.
- **Controls.** 8 rows with `WKB_Levin_max_depth` < 20 or `NULL`, two per band (low, mid and seam)
  and two with `NULL`, each with `compute_time` ≤ 60 s.
- Record the rule's bias: it prefers cheap rows.

**The units.** One registered, detached run (slug `wider`), with one checkpoint per (tree, rtol,
row):
1. $p$ at `rtol = 1e-7`, on every row: the baseline, and each row's budget;
2. $c$ at `1e-7`, on every row;
3. $c$ at `1e-8`, on the depth-20 rows.

Units 2 and 3 have a budget of 3× the row's $p$ time, enforced as 04 enforced it: raise from the
dispatcher, not by a signal inside `mpmath` (04's I9). An over-budget unit is a result.

**Gate L0, before the run** (on one depth-20 row and one control):
- `QuadSource` recomputed at $p$ equals A3 v2's stored `QuadSource` **bit for bit at every node
  the stored row holds**, and adds exactly B1's overlap nodes. That is how R8 found B1's change;
- two in-process runs of the same unit at $p$ are bit-identical.

If L0 fails, stop.

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged. "Scored
group" is 03's definition: at least 10 regions at depth 18 or more, here under $p$.
- **L1 — the cap goes where $G$'s anchor was the floor.**
  - **The rule:** on every depth-20 row whose top group is `G+r`, `G−r` or the lone `G`, under $c$
    at `1e-7`, every group under $p$'s scored set has fewer than 10 regions at depth ≥ 18, and the
    row's regions are at most a tenth of $p$'s.
  - **Holds** if true on every such row; an over-budget unit counts as false.
- **L1q — the same in the `Gqr` rows.**
  - **The rule:** L1's rule, on the `Gqr`-topped rows.
  - **Holds** if true on every such row. A failure is evidence that $T$'s phase carries a floor
    that matters there, which is the case for re-anchoring $T$. Report the depth-18 regions'
    $z'$ against $z_{\rm response}$ on any row that fails.
- **L2 — the controls do not move beyond their error.**
  - **The rule:** on every control row, $|{\rm total}_c - {\rm total}_p| \le$ `total_abserr`$_p$.
- **L3 — the declared error is honest after the change.**
  - **The rule:** on every depth-20 row with both $c$ units in budget,
    $|{\rm total}_c(10^{-7}) - {\rm total}_c(10^{-8})| \le$ `total_abserr`$_c(10^{-7})$.

**Descriptive, not scored:**
- $|{\rm total}_c - {\rm total}_p| /$ `total_abserr`$_p$ on the depth-20 rows. This is 04's N4,
  wider, and now with the modulus correction in it too;
- cost and regions by class and band;
- `total_converged` before and after;
- the low band against the mid band.

## 7. Acceptance

1. §1–§3A as written, with every test passing, the four breakages caught and the seven roots
   rising only by the tests added.
2. §5:
   - R9 and R10 both end `done` with fingerprint `1b332ffa…`;
   - every non-`QuadSourceIntegral` value is identical;
   - the `QuadSourceIntegral` differences are tabulated;
   - the readers exit 0;
   - the lock is released.
3. §6: L0 holds; L1, L1q, L2 and L3 are each scored holds, fails or undecided against the rule as
   written, with their numbers; the store is unchanged.
4. **Records.**
   - `logs/05-reanchor-g-phase.md`, with §5 and §6's Mac Studio record appended by the orchestrator;
   - `docs/levin-phase-floor/VERIFICATION.md` §5 "Layer 1(a)";
   - the records in `docs/levin-phase-floor/05/`.

## 8. Stop conditions — stop and ask

- The Gate fails.
- The construction needs a change to `TkSourceFunctions`'s phase, to a producer, to a schema or to
  `AdaptiveLevin/`.
- `delta` or `raw_theta` cannot be kept bit-identical.
- A seven-roots count falls.
- R9 does not reproduce `1b332ffa…`, or R10 differs from it.
- Any value outside `QuadSourceIntegral` differs between R9 and R10.
- A production class's value rows cannot be matched by identity (§3A).
- L0 fails, or `QuadSource` cannot be recomputed in-process without a production change.
- Any store file changes.
- The lock is held, or will not release.
