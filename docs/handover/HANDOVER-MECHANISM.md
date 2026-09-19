# The numeric→Liouville–Green hand-over, end to end, for one $k$ mode

**Campaign:** [`prompts/handover`](../../prompts/handover/README.md) · **Board item:** none — this
document was written by a review of the campaign documents (2026-09-19, Claude Opus 5) and lands no
code and no measurement.
**Status:** an **explanation**, not a record. Every number quoted here is quoted from somewhere
else and carries a pointer; nothing is measured here. Where this document and a campaign board
disagree, **the board is right** (`CLAUDE.md`).
**Reading position:** campaign README §8 item 0 — read this before the seam issues of
[`docs/OPEN_ISSUES.md`](../OPEN_ISSUES.md) §1.1, because those issues are named in the vocabulary
this document defines.

---

## 0. Why this document exists

The seam is spread across seven files and two independent object graphs, and the words
`crossover_z`, `z_init`, "hand-over", "overlap" and "gap" all mean slightly different things on the
$T_k$ side and the $G_k$ side. Before this document, a reader had to assemble that from
[`QuadSourcePolicy`'s class docstring](../../MetadataConcepts/QuadSourcePolicy.py),
[`TkSourceFunctions`' module docstring](../../ComputeTargets/TkSourceFunctions.py),
[`build_partition`'s docstring](../../ComputeTargets/QuadSourceIntegral.py), `GK-report.md`,
`QI-report.md` and the follow-up note — none of which is wrong, and no one of which is the account.

The load-bearing sentence, which as far as I can find is written down nowhere else, is §5:

> **$T_k$ clips its regions at the hand-over and therefore has a gap. $G_k$ does not clip and
> therefore has an overlap. That single difference is why $T_k$ needs a clamp, why $G_k$ needs a
> policy, and why there is no policy object for the transfer function.**

### 0.1 Whose object is whose

This is the trap. Get it wrong and nothing below parses.

| Object | Fixed | Varies over | Built by |
|---|---|---|---|
| `TkNumericIntegration` | $k$ | $z$ (one grid) | `main.py:1294-1318`, `mode="stop"` |
| `TkWKBIntegration` | $k$ | $z$ (one grid) | `main.py:1506-1520` |
| `TkSourceFunctions` | $k$ | $z$ | assembled on read; **not persisted** |
| `GkNumericIntegration` | $k$, $z_{\rm source}$ | $z_{\rm response}$ | `main.py:1885-1930`, `mode="stop"` |
| `GkWKBIntegration` | $k$, $z_{\rm source}$ | $z_{\rm response}$ | `main.py:2150-2192` |
| `GkSource` | $k$, $z_{\rm response}$ | **$z_{\rm source}$** | `assemble_GkSource_values`, `GkSource.py` |
| `GkSourcePolicyData` | $k$, $z_{\rm response}$, policy | — | `apply_GkSource_policy` |

The transposition at `GkSource` is the point. The Green's function is *produced* at fixed
$z_{\rm source}$ over a response grid and *consumed* at fixed $z_{\rm response}$ over a source
grid. So a single `GkSource` gathers values from many `GkWKBIntegration` objects, one per
$z_{\rm source}$, and whether a WKB value exists at a given $z_{\rm source}$ depends on whether
*that* object's initial condition was found above this $z_{\rm response}$. The transfer function
has no such transposition, and that is the root of every asymmetry below.

---

## 1. One $k$ mode, $T_k$: there is no choice to make

### 1.1 The producers

`main.py:1294-1318` submits one `TkNumericIntegration` per $k$ with `mode="stop"`, starting at the
top of the source grid. In `"stop"` mode the ODE does not run to the bottom of `z_sample`: it walks
down and terminates at a **phase extremum** located by
[`find_phase_extremum`](../../LiouvilleGreen/integration_tools.py), searching between
`z_exit_subh_e3` and `z_exit_subh_e6` (`TkNumericIntegration.py:148-149`). The object stores
`stop_deltaz_subh`, `stop_T`, `stop_Tprime` at that point, and holds **fewer values than its
`z_sample`** (`TkNumericIntegration.py:507`).

`main.py:1501-1506` then starts the WKB object at exactly that redshift:

```python
T_init      = Tk.stop_T
Tprime_init = Tk.stop_Tprime
z_init      = k_exit.z_exit - Tk.stop_deltaz_subh

max_source    = min(k_exit.z_exit_subh_e3, z_init)
source_sample = z_source_sample.truncate(max_source, keep="lower")
```

The last two lines are the whole of the structural defect. `z_init` is a root of a `root_scalar`
solve and is **not a source-grid point**; `truncate(..., keep="lower")` keeps the grid points at or
below it. So `TkWKBIntegration`'s first sample sits up to one grid step *below* `z_init`, and
**no sample is stored at `z_init` itself**.

### 1.2 The consumer-side assembly

[`TkSourceFunctions`](../../ComputeTargets/TkSourceFunctions.py) reads both objects, sets
`crossover_z = Tk_WKB.z_init`, and cross-checks it against
`k_exit.z_exit - Tk_numeric.stop_deltaz_subh` to `DEFAULT_FLOAT_PRECISION`
(`TkSourceFunctions.py:285-300`). It then **clips** both regions at the hand-over:

```
numeric_region = (largest numeric sample z,  smallest numeric sample z >= crossover_z)
WKB_region     = (largest WKB sample z <= crossover_z,  smallest WKB sample z)
```

so `WKB_region[0] <= crossover_z <= numeric_region[1]`, with equality only if the grid happens to
contain the hand-over point. In production it does not, so:

```
          numeric_region[1]      crossover_z         WKB_region[0]
   ────────────●──────────────────── × ─────────────────●──────────────►  decreasing z
               │                      │                 │
               └──── numeric valid ───┘   NOTHING       └── LG valid ────
                                          IS EVALUABLE HERE
```

That interval — median 1.2e-02, max 2.2e-02 in $\log(1+z)$ on real rows — is **the gap**. It is
structural, not a bug in any one line: it follows from `z_init` not being a grid point and from
both regions being clipped at it.

`TkSourceFunctions` is deliberately not a compute target and is not persisted; every number in it
is a re-reading of stored values plus closed forms.

### 1.3 Why no policy object

There is no overlap, so there is nothing to choose. `crossover_z` is a single already-determined
redshift recoverable from either stored object. This is stated, correctly and at length, in
[`QuadSourcePolicy`'s class docstring](../../MetadataConcepts/QuadSourcePolicy.py) and in
`TkSourceFunctions`' module docstring. See §3.3.

---

## 2. One $k$ mode, $G_k$: the overlap is engineered on purpose

### 2.1 The producers, and where the overlap comes from

Two cuts, deliberately staggered, both with explanatory comments in `main.py`:

- **Numeric.** `main.py:1888` submits a `GkNumericIntegration` for every $z_{\rm source}$ down to
  `z_exit_subh_e4` — **four** e-folds inside the horizon. Its response grid is truncated at
  `0.85 * z_exit_subh_e6`. The comment at `main.py:1878-1884` says why the cut is at four rather
  than three: *"we don't find enough overlap between the numeric and WKB regions to allow a smooth
  handover"*.
- **WKB.** `main.py:2120` computes
  $z_{\rm source\ limit} = \sqrt{z_{e3}\,z_{e4}}$, the geometric mean of the 3- and 4-e-fold
  points. Above it, the WKB object takes its initial data from the numeric one; below it, a purely
  WKB analysis is used. The comment at `main.py:2144-2149` states the intent in terms: *"Then we
  should have safely overlapping WKB and numeric solutions around the 4-efold point."*

So for $G$ the overlap is a designed quantity, not an accident — but note **its width is not
controlled by either cut directly**. What a given `GkSource` sees depends on $z_{\rm response}$,
because `GkWKBIntegration` produces no value for a $z_{\rm response}$ lying between its
$z_{\rm source}$ and its own `z_init` (`main.py:2136-2141`, `GkSource.py:174-181`).

### 2.1a What sizes the overlap, and the levers that move it

There are **two** `GkWKBIntegration` creation sites, and the difference between them is what makes
the overlap exist. At `main.py:2149` the branch is on $z_{\rm source}$ against
$z_{\rm source\ limit} = \sqrt{z_{e3}z_{e4}}$ (the geometric mean, i.e. **3.5 e-folds** sub-horizon):

| Branch | $z_{\rm source}$ | Initial data | Response grid |
|---|---|---|---|
| `main.py:2170` | $\ge \sqrt{z_{e3}z_{e4}}$ | from the numeric row: `G_init = Gk.stop_G`, `z_init = z_exit - Gk.stop_deltaz_subh` | **truncated** at $\min(z_{e3}, z_{\rm init})$ (`:2162`) |
| `main.py:2218` | $< \sqrt{z_{e3}z_{e4}}$ | the unit jump itself: `G_init = 0.0`, `Gprime_init = 1.0`, applied at $z_{\rm source}$ | **untruncated** |

The lower branch needs nothing from the numeric solution, so it produces a WKB value at **every**
$z_{\rm response}$ in its pool. Numeric $G$, meanwhile, exists for every $z_{\rm source}$ down to
`z_exit_subh_e4` — **4 e-folds** (`main.py:1888`). So on the band

$$
z_{e4} \;<\; z_{\rm source} \;<\; \sqrt{z_{e3}z_{e4}}
\qquad\text{i.e. between 4 and 3.5 e-folds sub-horizon}
$$

**both** representations exist for any $z_{\rm response}$ that both grids reach. That half e-fold
is the overlap the design intends, and it is the same half e-fold in e-folds for every $k$,
because both ends are per-$k$ horizon-crossing quantities. The overlap can be *wider* — the upper
branch adds to it whenever $z_{\rm response} \le \min(z_{e3}, z_{\rm init})$ — but half an e-fold
is what is there by construction.

**The levers.** These are the knobs that move the overlap, in the order in which they bite:

| # | Lever | Where | What it moves |
|---|---|---|---|
| **L1** | the numeric $z_{\rm source}$ cut, `z_exit_subh_e4` | `main.py:1888` | the **bottom** of the guaranteed overlap. Deepening it lowers `numeric_smallest_z` directly, at the cost of more `GkNumericIntegration` rows where $G$ oscillates fastest |
| **L2** | the WKB initial-data switch, $\sqrt{z_{e3}z_{e4}}$ | `main.py:2120` | the **top** of the guaranteed overlap. Raising it extends the unit-jump branch — the one that needs nothing from the numeric side — to higher $z_{\rm source}$ |
| **L3** | the upper branch's response-grid top, $\min(z_{e3}, z_{\rm init})$ | `main.py:2162` | whether the overlap extends *above* L2 at this $z_{\rm response}$. This is the $z_{\rm response}$-dependent part, and the source of the structural hole (`main.py:2136-2141`) |
| **L4** | the numeric response-grid bottom, $0.85\,z_{e6}$ and the `"stop"` event | `main.py:1901-1902` | the lowest $z_{\rm response}$ at which numeric $G$ exists at all |
| **L5** | `MIN_SPLINE_DATA_POINTS = 5` | `GkSourcePolicyData.py:17` | converts a geometric overlap into a *usable* one. An overlap carrying four nodes is classified `fail`, not `mixed` |
| **L6** | the source-grid density criterion | `qcd-background-audit` prompt 15 | how many nodes a given width contains. Interacts with L5, and is **out of scope** for this campaign (README §0.3) |
| **L7** | `GkNumericIntegration`'s own `mode="stop"` point | its stop search | sets L3's $z_{\rm init}$ per $z_{\rm source}$ |

**L1 and L2 are the two that size the guaranteed overlap, and both are single expressions in
`main.py`.** Neither has ever been varied, and no measurement of the resulting width — in grid
intervals, which is the unit that matters for L5 and for the end-interval question of §6 — exists.
That measurement is campaign prompt 03.

Two things follow that are worth stating before anyone reaches for a lever. First, **the
$z_{\rm response}$ dependence lives in L3 and L4, not in L1 or L2**: the guaranteed band is fixed
in e-folds, and what varies with $z_{\rm response}$ is how much is added above it and whether the
numeric grid reaches down far enough. So an `incomplete` classification is a statement about
$z_{\rm response}$, not about $k$ — which prompt 03 can confirm or refute directly. Second,
**L1 and L2 move the overlap in opposite directions but are not equivalent**: L1 buys width with
numeric rows in the fastest-oscillating region, where `[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]`
already measures the consumer spline at ×631–×37,700 the solver's error, while L2 buys it with
unit-jump WKB rows in a region one half e-fold shallower, where the LG truncation is larger. They
are not the same trade and should not be treated as one.

### 2.2 Assembly

`assemble_GkSource_values` walks the source grid at fixed $z_{\rm response}$ and records, per
$z_{\rm source}$, whichever of the numeric and WKB values exist. Two summary redshifts come out:

- **`numeric_smallest_z`** — the lowest $z_{\rm source}$ carrying a numeric $G$ (`GkSource.py:318-320`);
- **`primary_WKB_largest_z`** — the top of the *contiguous* run of WKB values counted **upward from
  the bottom of the grid** (`GkSource.py:181, 243-246, 315-316`). "Primary" means contiguous-from-
  the-bottom: a WKB island higher up, separated by a hole, is not counted, and a non-contiguous WKB
  region prints a warning (`GkSource.py:150-164`).

The **overlap** is $(\texttt{primary\_WKB\_largest\_z}, \texttt{numeric\_smallest\_z})$ — non-empty
whenever `numeric_smallest_z < primary_WKB_largest_z`.

### 2.3 What the policy chooses, and what it does not

`GkSourcePolicyData._create_functions` (`GkSourcePolicyData.py:638-752`) builds **two splines in
$\log(1+z_{\rm source})$**:

| Spline | Range | Note |
|---|---|---|
| `numeric_Gk` | `(z_sample.max, numeric_smallest_z)` | cubic `make_interp_spline` over source-grid nodes |
| `WKB_Gk` | `(primary_WKB_largest_z, z_sample.min)` | amplitude splined; phase from `PrimitivePhase`, **not** splined |

**Neither is clipped at `crossover_z`.** Both cross it. `crossover_z` is purely a *consumer-side
switch*: a redshift at which the consumer changes which of two everywhere-defined objects it
evaluates. This is the exact opposite of the $T_k$ arrangement and is the reason $G$ needs no
clamp.

One $G$-specific asymmetry, flagged by the follow-up note §1.3 and worth repeating: the WKB
**amplitude** `sin_coeff * sqrt(H_ratio / sqrt(omega_WKB_sq))` **is** splined
(`GkSourcePolicyData.py:709-721`), whereas `TkSourceFunctions.M` is assembled from closed forms and
a table lookup. So the $G$ WKB branch has an end interval of its own that the $T$ branch does not.

---

## 3. The policy objects

### 3.1 `GkSourcePolicy` — two fields, one of them live

[`MetadataConcepts/GkSourcePolicy.py`](../../MetadataConcepts/GkSourcePolicy.py) is a persisted
metadata object with `Levin_threshold` and `numeric_policy`. It is part of the datastore key of
`GkSourcePolicyData`, so changing either field forks the stored rows rather than overwriting them.

- **`numeric_policy`** — read exactly once, at `GkSourcePolicyData.py:398`. Live.
- **`Levin_threshold`** — sets `Levin_z`, which has been **diagnostic only** since
  `source-remediation` prompt 10. `QuadSourceIntegral` reads it into
  `metadata["partition"]["Levin_z_unused"]` and routes on nothing. The comment at
  `GkSourcePolicyData.py:50-57` records this.

`main.py:3630-3645` creates two policies per run, at `Levin_threshold` 1.5 and 5.0, both with
`numeric_policy="maximize-WKB"`. **No other value has ever been used**, in `main.py` or in any
`extract_*.py` or verification script.

### 3.2 `_classify_crossover` — the decision procedure

`GkSourcePolicyData.py:287-504`. Four outcomes, in order:

| Condition | `type` | `quality` | `crossover_z` |
|---|---|---|---|
| neither region splineable (< 5 points) | `fail` | `incomplete` | `None` |
| no WKB region | `numeric` | `complete` iff numeric reaches `z_sample.min` | `None` |
| no numeric region | `WKB` | `complete` iff WKB reaches `z_sample.max` | `None` |
| both present, no overlap | `mixed` | `incomplete` | `numeric_smallest_z` |
| both present, overlap | `mixed` | band reached | chosen below |

For the last case, each grid point in the overlap is scored by two **relative clearances in
$\log(1+z)$** (`GkSourcePolicyData.py:415-416`):

$$
c_{\rm num} = \frac{\log(1+z) - \log(1+z_{\rm num,low})}{\log(1+z_{\rm num,low})},
\qquad
c_{\rm WKB} = \frac{\log(1+z_{\rm WKB,high}) - \log(1+z)}{\log(1+z_{\rm WKB,high})}
$$

and thresholded at `CLEARANCE_GOOD = 0.05`, `CLEARANCE_ACCEPTABLE = 0.025`,
`CLEARANCE_MARGINAL = 0.01`, plus `*_minimal` meaning "strictly greater than zero". Ten
`CLASSIFICATION_BANDS` (`GkSourcePolicyData.py:273-284`) are then tried in order — `complete`,
three `acceptable`, five `marginal`, one `minimal` — and the **first band with any qualifying point
wins**. Within the winning band, `maximize-WKB` sorts descending in $z$ and takes the largest.
If every band is empty, the fallback sets `crossover_z = primary_WKB_largest_z` and
`quality = "minimal"` with an explanatory `metadata["comment"]`.

Three properties of this worth stating plainly, because they are what §6 turns on:

1. **The clearance is a fraction of $\log(1+z)$, not a count of grid intervals and not e-folds.**
   At $z \sim 10^{13}$, $\log(1+z) \approx 30$, so `CLEARANCE_GOOD` means ~1.5 e-folds; at
   $z \sim 10$ it means ~0.12. `GK-report.md` GK-2's closing note calls this *"a deliberate choice,
   but it makes the thresholds z-range-dependent."*
2. **`maximize-WKB` takes the candidate with the *least* WKB clearance in the winning band.**
   Larger $z$ is nearer `primary_WKB_largest_z`. Within `complete` this is bounded by the band
   floor and is harmless; in `marginal` or `minimal` it places the consumer's LG evaluation in the
   WKB amplitude spline's top end interval.
3. **The bands degrade silently.** Only `quality` records which was reached; nothing raises and no
   threshold is a floor.

### 3.3 `QuadSourcePolicy` — persisted, threaded, and read by nothing

[`MetadataConcepts/QuadSourcePolicy.py`](../../MetadataConcepts/QuadSourcePolicy.py) has the same
two fields and **no consumer**. Its class docstring is the authority and should be read in full;
in summary:

- `numeric_policy` would choose a crossover inside an overlap, and the transfer function has none
  (§1.3);
- `Levin_threshold` would gate quadrature, and since `source-remediation` prompt 08 every
  oscillatory sub-interval goes to `adaptive_levin_sincos`, whose own total-variation gate decides
  per region with strictly more information — it can see the composed phase
  $\theta_G \pm \theta_q \pm \theta_r$, which a threshold here cannot.

The object stays persisted so the schema and run signatures do not churn. Note the standing
instruction on `[10-levin-wholesale-cc-fallback]`: ***"Do not re-add a threshold in
`QuadSourcePolicy` — that is defect A4 in weaker form."***

---

## 4. The consumer: `build_partition`

`QuadSourceIntegral.py:326-560` partitions $[z_{\rm response}, z_{\rm source\ max}]$ at the
hand-over of all three factors of $G_k(z,z')\,f(z'|q,r)/H(z')^2$:

| factor | smooth above | oscillatory below | breakpoint |
|---|---|---|---|
| $G$ | `Gk_f.numeric_Gk` | `Gk_f.sin_amplitude`, `Gk_f.phase` | `GkPolicy.crossover_z` (type `mixed`) |
| $T_q$ | `Tq_f.T`, `dT_dz` | `Tq_f.M`, `dlnM_dz`, `omega`, `phase` | `Tq_f.crossover_z` |
| $T_r$ | likewise | likewise | `Tr_f.crossover_z` |

Each sub-interval carries a regime triple; all-smooth goes to ordinary quadrature, anything else to
Levin. Then the asymmetry of §5 appears in the code, in two adjacent blocks:

- **$G$ (`:461-479`)** — `_check_region_covers`, **strict**. The comment says why:
  *"GkSourcePolicyData guarantees its regions cover the crossover with clearance, so no clamping is
  offered."*
- **$T_q$, $T_r$ (`:481-509`)** — `_check_gap` against
  `HANDOVER_CLAMP_MAX_GRID_STEPS * _mean_source_grid_step(source)`, currently **1.5** mean grid
  steps, and the accessors are wrapped in `_ClampedTk`, which holds $\log(1+z)$ at the nearest end
  of the evaluable range. A larger shortfall raises.

Clamping the LG accessors means **holding the phase constant across the gap** — up to 0.44 rad on
real rows. Gaps are recorded per sub-interval in `metadata["partition"]["clamp_gaps_log1pz"]`.

One identity check worth knowing about: `QuadSourceIntegral.py:781-790` refuses a `QuadSource`
whose recorded `crossover_z_q` / `crossover_z_r` disagree with the supplied `TkSourceFunctions`.
There is **no** analogous foreign key tying a `TkWKBIntegration` row to the numeric row it came
from — that is `[20-wkb-rows-consume-numeric-initial-data]`, campaign workstream C1.

---

## 5. The asymmetry, in one table

| | $T_k$ | $G_k$ |
|---|---|---|
| producers share a grid? | yes (`z_source_sample`) | no — WKB produced over $z_{\rm response}$, consumed over $z_{\rm source}$ |
| hand-over redshift | `TkWKBIntegration.z_init`, a `root_scalar` root | `crossover_z`, a **grid point** in the overlap |
| who decides it | nobody — it is determined | `_classify_crossover` + `numeric_policy` |
| regions clipped at it? | **yes**, by `TkSourceFunctions` | **no** — both splines cross it |
| consequence | a **gap** of up to one grid step | an **overlap** of many grid steps |
| consumer treatment | `_ClampedTk`, ≤ 1.5 mean grid steps | strict `_check_region_covers` |
| WKB amplitude | closed form + table lookup | **splined** (own end interval) |
| policy object | none needed, and `QuadSourcePolicy` is inert | `GkSourcePolicy.numeric_policy`, live |
| continuity at the seam | measured by `source-remediation` prompt 12 | measured: median 7.4e-08, worst 4.0e-06 |

---

## 6. What the procedure guarantees, and what it does not

**Guaranteed.** `crossover_z` is a grid point strictly inside the overlap, carrying a recorded
quality label; both splines are defined on both sides of it; the consumer's $G$ range checks are
strict and cannot silently clamp; `numeric_Gk` and `WKB_Gk` are the same unit-jump $\bar G_k$ on
both sides (`GK-report.md` GK-11), and their measured mismatch at `crossover_z` is 7.4e-08 median
over 40 `mixed` rows.

**Not guaranteed, and this is the gap between what the code does and what the campaign documents
assume it does:**

1. **That the consumed range avoids either spline's end interval.** The follow-up note §1.3 and §3
   item 1, and campaign README **D5**, both ask whether `crossover_z` is at least *two grid
   intervals* inside both regions. **No band tests that.** The clearance is a fraction of
   $\log(1+z)$; grid spacing does not enter `_classify_crossover` anywhere. The question is
   therefore not answerable from the stored `quality` label, which is why D5 is still open —
   campaign prompt 03 is the instrument that answers it.
2. **Any WKB-validity criterion.** `GK-report.md` GK-6: the crossover is chosen on spline geometry
   alone. R38's $Q \to -1$ is never used as a criterion, and `has_WKB_violation` is stored but
   never used to reject data. Closed as *moot in this configuration* — 0 of 7 `TkWKBIntegration`
   and 0 of 5488 `GkWKBIntegration` rows carry the flag — which is a statement about the rows that
   exist, not about the criterion.
3. **Any agreement test between the two representations.** `QI-7` made this point and was closed by
   *measuring* the agreement once, on one run's rows. The code still does not test it, so a future
   configuration in which the two disagree at `crossover_z` would be selected just as confidently.
4. **That the machinery is exercised at all.** On the `source-remediation` prompt 12 run, of 462
   policy rows: 270 `numeric/complete` (58 %), 123 `WKB/complete` (27 %), 61 `mixed/complete`
   (13 %), **1 `mixed/minimal`**, **7 `fail/incomplete`**. The band ladder runs on ~13 % of rows,
   and the 7 `fail` rows would raise in `build_partition` if the source integral reached them.

### 6.1 `incomplete` is diagnosed and then ignored

This is the sharpest of the six, and it is the subject of campaign workstream **B3**.

`quality` is read in exactly four places, and **none of them gates any work**:

| Site | What it does |
|---|---|
| `extract_common.py:478` | prints it on a plot |
| `main.py:2982-2994` | end-of-stage summary statistics |
| `main.py:2841`, `:2929` | triggers a diagnostic JSON dump |
| `GkSourcePolicyData.py:221-224` | `_classify_Levin` returns early — and `Levin_z` is diagnostic-only |

The row is then stored, and `main.py:3175-3205` looks `GkSourcePolicyData` up and feeds it into
the source-integral work queue **with no filter on `type` or `quality`**. An `incomplete` row is
served to `compute_QuadSource_integral` exactly like a `complete` one.

Every `incomplete` route ends in a raise if that $(k, z_{\rm response})$ pair is reached:

- **`mixed`/`incomplete`** sets `crossover_z = numeric_smallest_z`, which by that branch's own test
  sits *above* `primary_WKB_largest_z`. Below the crossover `build_partition` needs the WKB branch,
  and $G$ has **no clamp** — `_check_region_covers` is strict, unlike the $T_q$/$T_r$ path (§4).
- **`numeric`/`incomplete`** and **`WKB`/`incomplete`** fail region coverage at the bottom and top
  of the range respectively, for the same reason.

So the classification is a **fail-late** contract: diagnosed at policy time, ignored, and surfaced
much later as a message about *regions* rather than about the policy. Verification §5.5's remark
that the 7 `fail` rows *"are not among the $(k, z_{\rm response})$ pairs the source integral
reached here"* records luck, not design.

Two supporting gaps. **`_classify_crossover` has no test**:
`ComputeTargets/tests/test_gk_source_policy.py` is 214 lines and every one is about
`_classify_Levin`. And `main.py:2929` calls `dump_incomplete_GkSourcePolicy.remote(obj)` **without
checking the `dump_incomplete` flag**, so on a run without `--dump-incomplete` the dump path is
`None`, the Ray task dies of `TypeError`, and nothing notices because the reference is discarded
(`main.py:2524-2527`).

---

## 7. Known defects and open questions

| # | What | Where | Status |
|---|---|---|---|
| GK-2 / A6 | `"WKB_minimal"` tested `numeric_clearance` | `GkSourcePolicyData.py` | **fixed**, `source-remediation` prompt 02; reachability confirmed (1 of 462 rows) |
| GK-6 | no WKB-validity criterion | `GkSourcePolicyData` | closed as **moot** — 0 rows flagged |
| QI-7 | continuity at `crossover_z` not measured | `GkSourcePolicyData` | closed — 7.4e-08 median, 4.0e-06 worst |
| §4.2 obs. | 7 of 462 rows are `fail/incomplete`; `build_partition` would raise | `_classify_crossover` | recorded as an observation only, never opened as an issue |
| C2 (a) | `numeric_policy="maximize-numeric"` validates, persists, then raises at compute time | `GkSourcePolicy.py:6` vs `GkSourcePolicyData.py:465, 472-475` | campaign README §2 (o), workstream **C2** |
| C2 (b) | `QuadSourcePolicy` spells the same value `"maximize_numeric"` | `QuadSourcePolicy.py:6` | campaign README §2 (o), workstream **C2** |
| B3 | `incomplete`/`fail` is diagnosed and then ignored; no filter, and the raise lands in `build_partition` | §6.1 | campaign README §2 (p), workstream **B3** |
| B3 | `_classify_crossover` has no test; `main.py:2929` fires the dump ungated | §6.1 | same |
| D5 | is `crossover_z` two grid intervals inside both regions? | — | **unmeasured**; campaign prompt 03 |
| D7 | what to do when the geometry is irrecoverably bad for some $(k, z_{\rm response})$ | §2.1a L3 | campaign README §7 **D7**, deferred pending prompt 03 |
| D1 | the numeric $G$ consumer spline dominates near the hand-over | `GkSourcePolicyData.py` | `[03-numeric-g-consumer-spline-is-the-dominant-error-near-the-hand-over]` |

Three citations in the campaign README had drifted and were refreshed against this tree on
2026-09-19: §2 (a)'s `main.py:695-697` → `main.py:1501-1506`; §3 **E2**'s
`TkNumericIntegration.py:130-131` → `:148-149`; §3 **D1**'s `GkSourcePolicyData.py:654-680` →
`:638-752`, numeric branch `:667-681`. The measurements attached to each were not touched.

One distinction a reader will otherwise conflate: the `mode="stop"` **search window** is
`z_exit_subh_e3` → `z_exit_subh_e6` (`TkNumericIntegration.py:148-149`), while the factor `0.85`
that appears alongside it in `main.py` is a separate truncation of the *sample grid*
(`main.py:1292` for $T$, `main.py:1902` for $G$), which the comment there explains is not
what it appears to be: in `"stop"` mode the ODE terminates on an event and the samples between
$z_{e6}$ and $0.85\,z_{e6}$ are never produced.

---

## 8. Where the measurements live

Nothing here is measured. The numbers quoted above come from:

| Quantity | Source |
|---|---|
| gap widths (1.2e-02 median, 2.2e-02 max in $\log(1+z)$); 6.6e-02–4.6e-01 vs 6.2e-05 | `prompts/source-remediation/IMPLEMENTATION_STATE.md` §3, `[12-handover-clamp-error-in-production]` |
| 0.44 rad held phase | same, `[08-handover-clamp-error]` |
| $G$ continuity 7.4e-08 / 4.0e-06; the 462-row census; `has_WKB_violation` zero counts | [`docs/source-remediation-verification.md`](../source-remediation-verification.md) §5.5 |
| numeric $G$ consumer spline ×631–×37,700 | `prompts/tolerance-convergence/IMPLEMENTATION_STATE.md`, `[03-…]` |
| LG truncation 1.4e-04 at $x_T = 15.5$, 4e-06 at $x_T = 50$ | `prompts/GkTk-remedial/IMPLEMENTATION_STATE.md`, `[00-tk-lg-truncation-floor]` |
| stop-point root tolerance $\sim10^{-4}z$ | same, `[11-stop-point-root-tolerance]` |
| overlap width in grid intervals | **nowhere — this is campaign prompt 03** |
