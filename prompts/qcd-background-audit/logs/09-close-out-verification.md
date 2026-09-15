# Log 09 — Close-out: the consumer tables under a corrected background

**Prompt:** prompts/qcd-background-audit/09-close-out-verification.md
**Commit:** *(this commit)* — Verify the QCD background remediation against both consumers
**Model:** Claude Opus 5
**Date:** 2026-09-14
**Result:** COMPLETE WITH DEVIATIONS

---

## What shipped

**No production file was touched**, and none is in the diff. `git diff --name-only` against
`4804aac` is: `docs/qcd-background-verification.md` (new),
`docs/qcd-background-audit/consumer_break_point_profile.py` (new),
`docs/gktk-remedial-verification.md` (**+121 / −0**, a §9 appended, nothing at or above §8
touched), this log, the campaign board, `prompts/GkTk-remedial/IMPLEMENTATION_STATE.md` (one dated
block appended to the `[13-...]` entry it owns — deviation 7) and `docs/OPEN_ISSUES.md`.

**`T_Z_REPRESENTATION_VERSION` is 5 before and 5 after.** Prompt 09 is a verification prompt, it
moves no number, and no bump was due.

### `docs/qcd-background-verification.md` (new, 704 lines)

The campaign's verification document, written once and additive thereafter, structured on
`docs/gktk-remedial-verification.md`. Eight sections: §0 what was run and the framing, §1 the
representation, §2 **T1 closed** (the headline, and the first result), §3 the consumers, §4 the
break-point set and what it cost, §5 cost, §6 what the campaign did not establish, §7 reproduction.

### `docs/qcd-background-audit/consumer_break_point_profile.py` (new, 176 lines)

`dlnH/du` differenced on production-grid spacing over 25 intervals centred on an equation-of-state
crossing, reported as a deviation from the local median, with the peak, the **share of the total
deviation inside the crossing's own interval** and the number of intervals carrying > 10 % of the
peak. It takes `--root` so the identical measurement can be taken on a `git worktree` at the
campaign base, and `--crossing {T_LO,EOS_T_LO,T_120_MEV}`. No Ray, no datastore, ~1 s.

It exists because §3.3 is the document's one contentious finding and prose was not enough: it is
the evidence that the two consumer rows which rose did so because the **corrected background
presents a sharper feature**, not because anything regressed.

### `docs/gktk-remedial-verification.md` §9 (appended)

Dated 2026-09-14, five subsections: §9.1 what §8.3 warned about has happened, §9.2 §3.5 row by row,
§9.3 §3.6, §9.4 everything else in that document that moved, §9.5 the caveat discharged and the one
added. It says in terms that §§1–7 were correct for the tree they were taken on and that the
background underneath them has since changed — the fact its own §6 could not have seen.

---

## Deviations from the prompt

### 1. The base run was reused at `2a5e0fa`, not re-taken at `e8f746d` — IMPLEMENTATION CHOICE

Prompt §2 item 1 says to take the `verify_production_path.py` run "at the campaign's **base**
(`e8f746d`)". The runs used are the orchestrator's, taken at **`2a5e0fa`**, the planning commit.

They were reused rather than re-taken because the two trees are **numerically identical**, which is
checkable rather than asserted: `git diff --name-only e8f746d 2a5e0fa` is 21 files and **every one
of them is under `docs/` or `prompts/`** — `docs/OPEN_ISSUES.md`, the campaign's twelve prompt
files, its README, board, log index and four orchestrator prompts. No Python file, no test module,
no fixture, and **neither of the two scripts this prompt runs unedited**
(`docs/gktk-remedial/verify_production_path.py` and
`docs/qcd-background-audit/measure_T_z_representation.py`, the latter committed at `e375c9f`,
before `e8f746d`). Re-taking them would have produced the same numbers and a second, differently
-timed set of wall clocks to choose between; the campaign's rule is that a measurement is reused or
re-taken, never mixed. **Both base files are used throughout and neither was supplemented.**

The same base tree was also used, as a `git worktree` with the main checkout's `venv` symlinked, for
the §3.3 profile and for the byte-identity check of deviation 3.

### 2. "Nothing got worse" could not be established: **two of §3.5's twelve rows and two of §3.6's rose** — STRUCTURALLY REQUIRED

Prompt §1 and README §6.4 state what must be established, and the first clause is "that **nothing
got worse**". Measured, that is false, and it is recorded as a miss rather than argued away
(README §6: "A miss is an issue and `COMPLETE WITH DEVIATIONS`, never a rewritten threshold").

| table | row | base | `HEAD` | factor |
|---|---|---|---|---|
| §3.5 | QCD $G_k$, $k=10^5$ | 1.9073e-06 rad (8.00 ulp) | **8.1062e-06 (34.00 ulp)** | **4.25×** |
| §3.5 | QCD $T_k$, $k=10^5$ | 3.1859e-06 rad (427.60 ulp) | **1.3982e-05 (1876.61 ulp)** | **4.39×** |
| §3.6 | QCD $G_k$, $k=10^5$ | 5.1283e-07 | **1.0232e-06** | 2.00× |
| §3.6 | QCD $T_k$, $k=10^5$ | 1.0998e-06 | **3.0630e-06** | 2.79× |

All four have their maximum at the `T_LO` crossing — §3.5's at $z=4.24388\times10^7$, *inside* the
single production grid interval that contains it, §3.6's at $z=4.2885\times10^7$, the node
immediately above it.

**The cause was measured, not inferred**, and it is that the corrected background is *harder* for
the consumer's cubic, not worse:

| `dlnH/du` across the `T_LO` crossing, on production-grid spacing | base | `HEAD` |
|---|---|---|
| peak deviation from the local median | +3.071237e-02 | **+8.544581e-02** |
| total $\sum\lvert$deviation$\rvert$ over 25 intervals | 1.201851e-01 | **8.557937e-02** |
| share of that total inside the crossing's own interval | **25.55 %** | **99.84 %** |
| intervals carrying > 10 % of the peak | **8** | **1** |

The pre-campaign 500-node order-3 `T`-against-`u` spline had a knot spacing of 9.2936e-02 in $u$,
**4.04× the production grid spacing**, so it smeared the equation of state's genuine step over about
four grid intervals and added ±3e-03 of its own scatter either side. The shipped representation is
segmented *at* the crossing: the step is the genuine one, 2.78× taller, and 99.84 % of it is inside
one interval. The measured consumer factors, 4.25× and 4.39×, are consistent with the 2.78× in
step height carrying a further ~1.5× from the concentration; that decomposition is a plausibility
argument, and the two halves of the table are the measurement.

**This was recorded and not fixed.** It is `[13-consumer-spline-crosses-eos-break-points]` — opened
on the `GkTk-remedial` board, narrowed by `prompts/phase-representation` prompt 02 and again by this
campaign's prompt 07, assigned to **prompt 10** — and prompt 09 may not touch production code. The
board entry is narrowed with the new magnitudes and the mechanism; nothing was repaired and no
threshold moved. **For scale and not as an excuse:** 8.1e-06 and 1.4e-05 rad sit 70–120× below the
~1e-03 rad Liouville–Green truncation floor that bounds any QCD phase claim, where they sat
300–500× below it before; and they were already above README §6's 1e-06 rad consumer target at the
base.

### 3. `LambdaCDM_GenericEOS(PureRadiationEOS)` is **not** bit-identical to the base — STRUCTURALLY REQUIRED

Prompt §3's acceptance table has a row "`RadiationModel` and stand-in figures | **bit-identical**".
Measured over 1,001 redshifts from 0 to $10^{19}$ as exact `float.hex`, base against `HEAD`:

| model | lines | verdict |
|---|---|---|
| `LambdaCDM(Planck2018)` — `Hubble`, `rho`, `T_photon`, `wBackground`, `wPerturbations` | 5,005 | **byte-identical**, MD5 `157b1c61436106ba52058ddf3533d2df` |
| `RadiationModel` — `Hubble`, `tau` | 2,002 | **byte-identical**, MD5 `265454eb751f9503a965f204e9e68bf4` |
| `LambdaCDM_GenericEOS(PureRadiationEOS)` — the same five | 5,005 | **differs on 4,617** |

The third is **required** to move and README §2 (g) says so in terms: on a constant-$g_s$ equation
of state $F(u)$ is identically constant and the new representation is **exact** — "that is an
acceptance test in prompt 05, not a hope". Unlike `LambdaCDM` and `RadiationModel`, which have no
`T(z)` representation at all, `PureRadiationEOS` is a `LambdaCDM_GenericEOS` and does build one; it
declares no *break points*, which is the thing README §2 (g)'s opening sentence is about.

What matters is the **direction**, and it was measured against the closed form $T=T_{\rm CMB}(1+z)$
over 2,001 points in $z\in[10^{-3},10^{16}]$:

| | max | p90 | median |
|---|---|---|---|
| base | 2.270323e-07 | 1.904049e-07 | 1.006789e-07 |
| **`HEAD`** | **3.330669e-16** | **2.220446e-16** | **0.000000e+00** |

6.8e8× towards exactness, and bit-exact at the median. **Not a stop**: the two models that take the
unchanged code path entirely are byte-identical across the whole nine-commit span, which is the
statement README §0.5 makes a stop condition.

### 4. `[01-convergence-block-has-a-separate-generator]` was re-pointed at prompt 09 and prompt 09 also cannot take it — STRUCTURALLY REQUIRED

Prompt 08's log and the board entry name prompt 09 as "the first prompt after this one with the
JSON and that test module naturally in scope". **They are not in scope.** Prompt 09's "Files you may
touch" is `docs/qcd-background-verification.md`, `docs/qcd-background-audit/`,
`docs/gktk-remedial-verification.md` (§9 only), the log, the board and `docs/OPEN_ISSUES.md`; its
"Do not touch" is "**any production file**", with the explicit reason that a verification prompt
changing production code is what created `prompts/phase-representation`. It includes neither
`ComputeTargets/tests/wkb_reference_data.json` nor `ComputeTargets/tests/test_background_tau.py`,
and closing the issue needs `docs/gktk-remedial/residual_convergence.py` to **write** that JSON.

**Decision: not done.** Regenerating the block and taking `QCD_BREAK_POINT_ALIGNMENT_TOL` back from
1.5e-04 would be a test-file edit on the production side of the line and scope creep under README
§5 rule 5. The figure is unchanged and re-measured twice already (1.418851e-04 at `T_120_MEV`, by
prompts 06 and 07, identical to every digit), and the whole of it is the block's age: the tree's
crossing is the genuine jump at $u = 27.485391822$ while the block still records where a smooth
interpolant passed through 0.12 GeV, at $u = 27.485249937$. The entry is **left open and pointed
forward** to whichever prompt next has both files in scope, with the reason recorded there.
`[02-qcd-reference-floor]` on the `GkTk-remedial` board waits on the same run.

### 5. One script was added under `docs/qcd-background-audit/` — IMPLEMENTATION CHOICE

Prompt §2 item 4 permits "`docs/qcd-background-audit/` (any script it needs)".
`consumer_break_point_profile.py` was added rather than left as a scratch script because §3.3 is
the one finding in the document that a later reader is likely to dispute, and a prose recipe is not
a reproduction. It is 1 s, takes `--root` so the base tree can be measured with the identical code,
and its docstring states the claim it is measuring and the two ways the delivery of the step
changed.

### 6. The `LiouvilleGreen` suite was run **full**, not on the fast set — IMPLEMENTATION CHOICE

Prompts 02–08 ran the fast set (every module except `test_3bessel_analytic`, which is ~1,400 s of a
~1,420 s total) because nothing they touched could reach it. Prompt 09 touches no code at all, so
the full set is affordable in wall time it was going to spend anyway and is the stronger statement
against the base's own 148. **Reported as the full set**, and the figure below is the full set.

### 7. `prompts/GkTk-remedial/IMPLEMENTATION_STATE.md` was edited, and it is not on prompt 09's file list — IMPLEMENTATION CHOICE

Prompt §4 says to update "this campaign's board … and every issue the campaign closed", and prompt
§2 item 4 lists the files. `prompts/GkTk-remedial/IMPLEMENTATION_STATE.md` is not among them, and it
was edited: a dated "**Narrowed again**" block was appended to
`[13-consumer-spline-crosses-eos-break-points]`, its owning board's §3 entry.

**Why.** That entry contains a testable prediction, written by `prompts/phase-representation`'s
close-out: the "±3 grid interval" structure prompt 02 attributed to $\varphi$ itself "is very likely
`[02-qcd-T-z-spline-node-tolerance]` after all — **testable by rebuilding `T(z)` and
re-measuring**". Prompt 09 is the run of that test, and **it falsifies the prediction in the
interesting direction**: `T(z)` was rebuilt and the consumer got *worse*, because the old
representation was smearing the defect rather than causing it. Leaving the owning board silent
about a measurement that refutes a prediction written in it — and leaving prompt 10 to design
against prompt 02's kink fit, which was taken on the smeared step — was the alternative, and it was
rejected. `CLAUDE.md` and README §5 rule 4 both make the owning board the place a measurement goes,
and prompt 07 of this campaign set the precedent by editing the same file to record its own
narrowing of the same entry.

**What was not done.** Nothing on that board was rewritten, nothing was closed there, no other
entry was touched, and the block is dated and additive (`git diff --numstat` is **44 / 1**, the one
deletion being its `Last updated` line, which that board's own convention requires to be corrected
and which keeps its previous entry). The campaign-board copy of the row and the
`docs/OPEN_ISSUES.md` index row were updated in the same commit.

### 8. `docs/OPEN_ISSUES.md`'s count and date were already correct — no change was due

Prompt §4 asks for "the count, the date, the `Boards` line …, and §1.7". Prompt 09 **opened no
issue and closed none** — four were narrowed (`[13-consumer-spline-crosses-eos-break-points]`,
`[01-convergence-block-has-a-separate-generator]`,
`[06-t-photon-call-cost-needs-a-quiet-machine]`,
`[03-qcd-inventory-does-not-report-the-representation]`) — so the count stays at **61**, verified by
counting the index's issue rows (`grep -c '^| \`\['` = 61, no duplicates), and the date was already
2026-09-14. The `Boards` line already carries `qcd-background-audit`; checked, correct, unchanged.
§1.7 was restructured: its prose now records the close at 9 / 12, and its single issue table is
**split in two** — "Owned by workstream D", which holds
`[13-consumer-spline-crosses-eos-break-points]` alone, and "Opened by this campaign and owned by no
prompt of it", which holds the other eight. That is prompt §4's "§1.7, which should now hold only
what workstream D still owns", made true without deleting a row that no other section indexes.

### None else

`docs/gktk-remedial/verify_production_path.py` and
`docs/qcd-background-audit/measure_T_z_representation.py` were run **unedited** (`git status` clean
for both, before and after). Nothing at or above §8 of `docs/gktk-remedial-verification.md` was
edited: `git diff --numstat` reads **121 insertions, 0 deletions**. No threshold anywhere in the
tree was loosened, because no test file was touched.

---

## Verification performed

Everything below was **run**. Commands were executed from the repository root with `PYTHONPATH=.`.
The machine was under erratic external load all day (load average peaked above 140 during prompt
07); every wall time here carries the load average it was taken at, and no conclusion rests on one.

### Prompt §3's acceptance table

| Quantity | Requirement | Measured | |
|---|---|---|---|
| Every LambdaCDM row of §3.5, §3.6, §3.7 | bit-identical | **bit-identical** — all six §3.5 rows, all six §3.6 rows including their per-sample decay ladders, all six §3.7 rows, and §1/§2's producer tables | ✓ |
| `RadiationModel` and stand-in figures | bit-identical | `RadiationModel` and `LambdaCDM` **byte-identical** over 7,007 `float.hex` lines; `PureRadiationEOS` moves **towards exactness**, by design (deviation 3) | ✓ / see deviation 3 |
| Prompt 01's $\int\mathrm{d}z/H$ guard | ≤ 1e-15, phases below the three floors | **0.0** — bit-identical at all 17 digits; **0.000e+00 rad** at all three wavenumbers | ✓ |
| Every moved QCD number | has a stated cause | every line of the 274-line diff is accounted for in `qcd-background-verification.md` §§2–5 | ✓ |
| Audit script §1 and §2 (the equation of state) | unchanged | **character-for-character identical** between the two runs | ✓ |
| Three suite counts | no fall from base | 11 → **30**, 339 → **359**, 148 → **148** (full set) | ✓ |

### The T1 guard, in full, at `HEAD`

```
[conformal time] int dz/H over z in [1e+02, 1e+12], 3 interior jumps given to the integrator
  shipped background = 1.3320002507788795e+03
  exact   background = 1.3320002507788795e+03
  relative error in tau = 0.0000e+00   (bit-identical: True)
    k =     1e+05 /Mpc:   0.000e+00 rad   against a 1-ulp floor of 3.050e-07 rad
    k =     1e+07 /Mpc:   0.000e+00 rad   against a 1-ulp floor of 3.050e-05 rad
    k =     3e+08 /Mpc:   0.000e+00 rad   against a 1-ulp floor of 9.150e-04 rad
```

against `3.4605051e-08` / 4.751e+01 / 4.751e+03 / 1.425e+05 rad at the base. Run as
`python -m unittest CosmologyModels.tests.test_T_z_representation -v`, 19 tests in 0.421 s, OK.

### `verify_production_path.py`, unedited, base against `HEAD`

45.8 s at `HEAD` (load ~6), 48.8 s at the base. The diff is **274 lines**, and the classification
is:

| class | lines | verdict |
|---|---|---|
| wall-clock timings only (build times, µs/call, total) | 12 | noise; the evaluation counts beside them are quoted instead |
| QCD primitives (§0) | 5 | `tau`/`cs_tau` **improve 9.3× / 9.5×**; `friction_F`, $\Delta\tau$ unchanged; `rho` 3.608e-16 → 2.248e-15 rad against a 1e-6 target, both at the round-off floor, reference regenerated |
| QCD producer tables (§1, §2 of the script) | ~90 | **all six rows improve**; all six were above the script's printed `eps*|theta|_max` floor at the base (1.07× to 5.94×) and at `HEAD` three are below it and three are within 11 % of it, i.e. 1.5–2 ulp of the accumulated phase |
| QCD `F`/`rho_T` as stored (§3 of the script) | 3 | `F` 1.345e-15 → 6.442e-16 and 1.782e-15 → 8.314e-16; 2.665e-14 → 6.252e-14 at $k=3\times10^8$; `rho_T` residual improves 3.1× / 1.4× / 3.2× |
| QCD consumers (§3.5, §3.6) | ~40 | four §3.5 rows unchanged in value, **two 4.25× / 4.39× worse** (deviation 2); §3.6 one row **1,413× better**, two worse, three within 30 % |
| per-object cost (§3.9) | 4 | QCD $G_k$ **8,380 → 6,892** build evaluations, $T_k$ **12,896 → 11,532**; LambdaCDM and all four cached counts exactly unchanged |
| residual-table margin (§4 of the script) | 2 | QCD only, see below |
| bulk accessor (§5 of the script) | 2 | QCD `raw_theta` off-grid **4.27 → 4.00** integrand evaluations per call |
| **LambdaCDM numbers** | **0** | — |

**Not one LambdaCDM value moved.** Every LambdaCDM line in the diff is a wall clock.

### The residual-table margin (§4), the one row whose cause needed looking up

QCD `Gk` worst cut/anchor 47.48 → **14.88** (3.860 → 2.700 e-folds), QCD `Tk` 5.666 → **4.677**
(1.735 → 1.543); LambdaCDM unchanged. Both are a *minimum over 50 wavenumbers* and the argmin
relocated ($k$ 1.1599e6 → 2.2298e6 and 3.6957e5 → 6.9985e6). The cut $z$ at `HEAD` is
**8.3841e+11 in both sectors**, which is production source grid node **439**,
$z = 8.384143\times10^{11}$, **1.33 grid intervals below the `T_120_MEV` crossing** at
$z = 8.644781\times10^{11}$ — i.e. `phase_residual`'s region selector now stops just short of the
genuine discontinuity instead of wandering with the old representation's scatter. The node
identification was measured (`production_source_z_values()` against the three crossings). Both
margins remain far above the ~1 e-fold the script itself names as the level that would make
`RESIDUAL_WKB_REGION_MARGIN` worth revisiting, and the constant was not touched.

### `measure_T_z_representation.py`, unedited

1.0 s at `HEAD` (load ~5), 1.4 s at the base. The whole diff is 14 lines, in §0, §3, §4, §5 and §6.
**§1 and §2 — the equation of state's branch joins, the forced steps in $T(z)$ and the jump
geometry — are character-for-character identical.** That is README §0.5's boundary held, measured.

| section | base | `HEAD` |
|---|---|---|
| §0 `_solve_T_z` | 2.496e-05 / 1.246e-05 / 1.222e-08 | **0.0 / 0.0 / 0.0** |
| §0, §3 shipped `T(z)` | 7.177e-04 / 1.323e-05 / 1.890e-07 | **6.807e-11 / 3.237e-15 / 1.765e-16** |
| §4 $H(z)$ | 1.278e-03 / 2.895e-05 / 3.424e-07 | **1.690e-10 / 6.276e-15 / 2.804e-16** |
| §4 $\int\mathrm{d}z/H$ | 1.3320002968728165e+03 | **1.3320002507788795e+03** = the exact one |
| §4 phase, three $k$ | 3.461e-08 → 4.751e+01 / 4.751e+03 / 1.425e+05 rad | **0.0 → 0.000e+00 at all three** |
| §5 `BREAK_POINT_ALL` in range | 407, spacing 4.04× the grid | **3**, spacing 215.34× |
| §5 `BREAK_POINT_DISCONTINUITY` | 2 | **2** |
| §6 `T_photon` | 2.454 µs/call | **2.596 µs/call** (a **confirmed miss**; §5.1 of the document) |

§5's prose line still reads "Of the BREAK_POINT_ALL points, 2411 are knots of the T(z) spline
itself", which is `[09-audit-script-section-5-prose-counts-the-wrong-set]`: the table above it is
right (`all 3 points in range`) and the script must be run unedited, so it was not fixed.

### The byte-identity harness, base against `HEAD`

`Hubble`, `rho`, `T_photon`, `wBackground`, `wPerturbations` for `LambdaCDM(Planck2018)` and for
`LambdaCDM_GenericEOS(PureRadiationEOS)`, plus `Hubble`, `tau` for `RadiationModel`, as exact
`float.hex` on 1,001 redshifts from 0 to $10^{19}$ — 12,012 lines, driven by one script against a
`git worktree` at `2a5e0fa` and against this tree. Results in deviation 3.

### The §3.3 profile

`docs/qcd-background-audit/consumer_break_point_profile.py`, on this tree and on the base worktree
with `--root`. Numbers in deviation 2. The measurement is 25 intervals of `dlnH/du` on
production-grid spacing around $u_c = 17.565806941870026$.

### Suite counts and wall times

| suite | base (`e8f746d`) | `HEAD` | wall | load |
|---|---|---|---|---|
| `CosmologyModels/tests` | 11 OK | **30 OK** | 0.573 s | ~5 |
| `ComputeTargets/tests` | 339 OK | **359 OK** | 152.470 s | ~5.8 |
| `LiouvilleGreen/tests` (**full**) | 148 OK | **148 OK** | 997.573 s | ~5.8 → 11.5 |

No count falls. `CosmologyModels` rises by 19 across the campaign (prompt 01's seven, prompt 06's
twelve) and `ComputeTargets` by 20 (prompt 03's fifteen, prompt 07's five).

### `black`

`./venv/bin/python -m black --check docs/qcd-background-audit/` — 5 files, clean. No other Python
file in the tree was touched.

---

## Observations not acted on

1. **§3.6's QCD $G_k$ row at $k=3\times10^8$ has its three windows collapse onto one number.**
   Base `3.309e-04 / 3.309e-04 / 1.748e-04` (max / `[3:-3]` / deep interior) becomes
   `2.542e-04` in all three, so the maximum improved 1.30× while the **deep interior got 1.45×
   worse**. That is what a quantity dominated by a three-valued staircase looks like when the
   noise that used to differentiate the windows is removed — `[02-consumer-phi-below-the-storage-granularity]`,
   which audit §9 and README §0.5 both put out of scope here. Not opened as a new issue: it is that
   issue seen more clearly, and the existing entry already carries the measurement
   (2.0 ulp, three distinct values over 1,377 samples).

2. **The `[06-t-photon-call-cost-needs-a-quiet-machine]` miss is now measured and the issue's own
   "next step" is discharged, but the miss stands.** 2.596 µs mean, range 2.505–2.671 over five
   runs on a quiet machine, against ≤ 2.5 µs — about 3.8 % over. The board entry is **narrowed**
   rather than closed, with the next step re-pointed at
   `[07-t-photon-range-logic-recomputes-its-bounds]`, which is a measured 0.11 µs and would bring it
   to ~2.49 µs. Hoisting those two `_outward` calls is a production edit and prompt 09 may not make
   it.

3. **`docs/gktk-remedial/verify_production_path.py` prints its sections out of numerical order**
   (0, 1, 2, 3, 4, 6, 5), which makes a `diff` of two runs harder to read than it needs to be and
   makes the mapping onto `gktk-remedial-verification.md`'s own §3.x numbering non-obvious. Not
   opened as an issue: it is cosmetic, the script must be run unedited, and it is another campaign's
   file.

4. **`[03-qcd-inventory-does-not-report-the-representation]` was not taken either**, although its
   own entry says it is "worth doing before prompt 09, which is the first prompt likely to look at a
   datastore holding rows at two representations". Prompt 09 looked at no datastore — nothing it
   runs needs one — and `Datastore/SQL/ObjectFactories/QCD_Cosmology.py` is a production file. Left
   open, unchanged.

5. **The campaign leaves `[13-consumer-spline-crosses-eos-break-points]` measurably larger than it
   found it**, which is deviation 2 and is the strongest argument in the tree for running prompt 10.
   It is recorded on the board rather than acted on, because workstream D is gated on README §7 D7.

---

## State handed to the next prompt

**The ungated chain 01–09 is complete and the campaign is CLOSED at 9 / 12.** Prompts 10–12 are
gated on README §7 **D7** and are the user's to release.

- **`T_Z_REPRESENTATION_VERSION` is 5**, unchanged by this prompt and unchanged since prompt 07.
  A prompt that moves a QCD background number must bump it on `LambdaCDM_GenericEOS` and add a row
  to the table in the comment block above the declaration.
- **The verification document is `docs/qcd-background-verification.md`** and it is **additive**: a
  later re-measurement appends a section and rewrites nothing. So is
  `docs/gktk-remedial-verification.md`, which now has a §9 and whose §§1–7 and §8 must not be
  touched.
- **For prompt 10, the case is now quantified.** §3.5's two QCD $k=10^5$ rows are **8.1062e-06 rad**
  ($G_k$, 34.00 ulp) and **1.3982e-05 rad** ($T_k$, 1876.61 ulp), both at $z=4.24388\times10^7$,
  and §3.6's are **1.0232e-06** and **3.0630e-06** at $z=4.2885\times10^7$ — 4.25×, 4.39×, 2.00×
  and 2.79× the figures `gktk-remedial-verification.md` §3.5/§3.6 record. The reason is measured:
  99.84 % of the step in `dlnH/du` is now inside the single grid interval containing the crossing,
  against 25.55 % before, and the step is 2.78× taller. Prompt 07 removed the blocker
  (a multiplicity-`spline_order` knot vector constructs on all six production geometries,
  `TestConsumerKnotVectorConstructs`), so the remedy is now available as well as needed.
  **`docs/qcd-background-audit/consumer_break_point_profile.py` is the tool** for re-taking the
  background side of that measurement.
- **Two tools are owed and neither could be run here.**
  `docs/gktk-remedial/residual_convergence.py` regenerates the JSON's `convergence` block and
  closes `[01-convergence-block-has-a-separate-generator]` and, with it,
  `QCD_BREAK_POINT_ALIGNMENT_TOL = 1.5e-04` (measured 1.418851e-04) and
  `[02-qcd-reference-floor]` on the `GkTk-remedial` board. One line in
  `sqla_QCDCosmology_factory.inventory()` closes
  `[03-qcd-inventory-does-not-report-the-representation]`. Both need a prompt that may touch
  production or test files.
- **Three cheap, numerically null production edits are queued**, each with its measurement already
  taken: hoist the two loop-invariant `_outward` calls out of
  `TemperatureRepresentation.__call__` and `ZSplineWrapper.__call__` (0.11 µs of a 2.596 µs
  `T_photon` call, `[07-...]`); move `_temperature_crossing_log1pz` into
  `CosmologyModels/tests/T_z_reference.py` (`[08-temperature-crossing-solver-is-test-only]`);
  intersect `knots` with the declared points in `measure_T_z_representation.py`'s §5 print
  (`[09-audit-script-section-5-prose-counts-the-wrong-set]`).
- **`prompts/tolerance-convergence` is unblocked.** README §0.4 held this campaign in front of it
  because its QCD half would have been measured against a background about to move by 3.5e-08. The
  background has moved and is now bit-identical to the exact one; references taken against it will
  not be invalidated the day they are taken.
- **README §7 D5 is still open and is the user's.** Neither `BREAK_POINT_KIND` was changed; log 08
  recommends leaving both, at a cost of +0.99 % of the $T_k$ evaluations, against a regeneration of
  every stored QCD $T_k$ row and everything below it if they move.
- **Reproduction, one command each:** the campaign's verification document §7 lists every one,
  with their wall times and the worktree recipe for the §3.3 profile.
