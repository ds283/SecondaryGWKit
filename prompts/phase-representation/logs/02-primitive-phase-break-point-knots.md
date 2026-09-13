# Log 02 — Break-point knots for `PrimitivePhase`: stopped on §2 item 2

**Prompt:** [`prompts/phase-representation/02-primitive-phase-break-point-knots.md`](../02-primitive-phase-break-point-knots.md)
**Commit:** *(this commit)* — Stop prompt 02: the repeated-knot vector does not construct
**Model:** Claude Opus 5
**Date:** 2026-09-13
**Baseline:** `9daa2cb` · **Parent:** `0c61799` (prompt 01)
**Result:** BLOCKED

**Why.** Prompt §2 item 2: *"If the knot vector cannot be made to construct on a real production
grid, stop and report. Do not fall back silently; README §7 D2 reserves that decision."* It cannot.
At the kind the prompt names — `BREAK_POINT_ALL` — a repeated (multiplicity-`spline_order`) knot
vector is **singular on all six production grids**, and where a repeated knot vector *does*
construct, at the one or two declared discontinuities, it makes the measured consumer error
**two to three times worse** in both sectors. No production file was changed.

The measurements below also **narrow the issue this prompt was to close**: the kink in $\varphi$ at
`QCD_EOS`'s declared discontinuity is worth **1.6e-08 rad** ($G_k$) and **1.4e-07 rad** ($T_k$) of
the measured 1.907e-06 / 3.186e-06 rad — **1 % and 4 %** — so the knots were never going to close
it, and the `theta_deriv` misses at the two larger wavenumbers turn out to belong to neither
candidate the prompt's §4 offers.

---

## What shipped

**No production code.** `ComputeTargets/primitive_phase.py`, `GkSourcePolicyData.py`,
`TkSourceFunctions.py` and `BackgroundModel.py` are byte-identical to `0c61799`; no test was added
or changed. What this commit contains is this log, this campaign's board, the narrowed
`GkTk-remedial` §3 entry, and `docs/OPEN_ISSUES.md`.

### The design decision, as prompt §5 requires it to be stated

**Neither construction was taken.** Prompt §2 item 2 offers a repeated-knot vector or per-segment
splines and does not dictate which. Both are excluded, by measurement rather than by preference:

- **The repeated-knot vector** (a knot of multiplicity `spline_order` = 3, so $C^0$, at each break
  point) is what the `GkTk-remedial` board's "next step" names. At `BREAK_POINT_ALL` it does not
  construct on any production grid (§1 below). At `BREAK_POINT_DISCONTINUITY` it constructs and is
  **2.0× worse** on $G_k$ and **2.1× worse** on $T_k$ at $k=10^5$ (§3).
- **Per-segment splines** are strictly freer than a multiplicity-3 knot — they drop even $C^0$ —
  so they cannot do better than the construction that was measured to be twice as bad, and they
  would have to manufacture the shared value at the break that §3 item 3 asks to be demonstrated
  absent. They were not implemented; taking them here would have been exactly the silent fallback
  §2 item 2 and README §7 D2 forbid.

Two non-repeated variants were measured as controls and are **not** proposals (§3): a
multiplicity-1 knot at every `BREAK_POINT_ALL` point, and multiplicity 1 or 2 at the
discontinuities. The first is the only scheme that improves $k=10^5$ — and it **breaks two of the
ten rows that prompt §4 requires to stay at 1.00 ulp**, and makes QCD $G_k$ `theta_deriv` at
$k=10^7$ **150× worse**. There is no scheme in the family that is a strict improvement.

---

## Deviations from the prompt

1. **`STRUCTURALLY REQUIRED` — the `GkTk-remedial` §3 entry
   `[13-consumer-spline-crosses-eos-break-points]` was *not* moved to that board's §4.** Prompt §5
   says to move it. The issue is not resolved, so moving it would have recorded a fix that does not
   exist. It stays in §3, **narrowed** with the measurements below and with a rewritten "Next step"
   that says what a future attempt must do differently. `docs/OPEN_ISSUES.md` keeps its row, with
   the hook corrected to the measured attribution.

2. **`STRUCTURALLY REQUIRED` — prompt §4's "re-run
   `docs/gktk-remedial/verify_production_path.py` … before and after" was run only as a
   *before*, and the QCD $G_k$ half of §3.5 could not have shown an "after" in any case.** That
   script builds its $G_k$ consumer by calling `PrimitivePhase(...)` directly (`:557`, `:1173`), so
   the `break_points` argument prompt §2 item 1 specifies — passed by the two production call sites
   — would never have reached it. Its $T_k$ half goes through `TkSourceFunctions` and would have
   moved. The measurements in §3 below were therefore taken on a harness that reproduces that
   script's geometry exactly (same fixtures, same band, same 10 points per production interval,
   same reference) and varies only the knot vector. Opened as
   `[02-verify-script-builds-its-own-Gk-consumer]`.

3. **`IMPLEMENTATION CHOICE` — two new issues opened** (§"Observations", and §3 of this
   campaign's board): `[02-consumer-phi-below-the-storage-granularity]` and
   `[02-verify-script-builds-its-own-Gk-consumer]`. Both are findings of the measurement this
   prompt was told to take, not repairs; nothing was fixed for either.

Nothing is tagged `UNINTENDED DRIFT`. No README §2 design fact was touched: $\varphi$ alone is
still what is splined, `num_chunks` still returns 1, `spline_order` is still 3, nothing was
chunked, and no `GkWKBValue`/`TkWKBValue` column, `*_omegaEff_sq` return value or `GkSource`
rectifier logic was approached.

---

## Verification performed

Everything below is `QCD_Cosmology` at the production geometry of
`docs/gktk-remedial-verification.md` §3.5 — the pure-WKB source band at the production
`--zend 0.1` for $G_k$, the whole WKB region below the hand-over for $T_k$, both scored at ten
points per production grid interval against the production producer evaluated at those same points.
LambdaCDM declares no break points at all (measured: **0** on every band), so every scheme below is
the identity there and no LambdaCDM number can move.

### 1. The repeated-knot vector does not construct

Per grid: the number of `BREAK_POINT_ALL` / `BREAK_POINT_DISCONTINUITY` points strictly inside the
sample range, the number of the resulting segments that hold **no** sample at all, and whether a
multiplicity-3 knot vector satisfies the Schoenberg–Whitney conditions against the sites present.

| sector | $k$ | samples | ALL | DISC | empty segments | segments < 3 samples | ALL × 3 | DISC × 3 |
|---|---|---|---|---|---|---|---|---|
| $G_k$ | 1e5 | 1016 | 226 | 1 | 1 | 1 | **SW fails at 3 sites** | constructs |
| $T_k$ | 1e5 | 1040 | 233 | 1 | 2 | 3 | **SW fails at 5 sites** | constructs |
| $G_k$ | 1e7 | 1218 | 278 | 1 | 2 | 3 | **SW fails at 9 sites** | constructs |
| $T_k$ | 1e7 | 1242 | 284 | 1 | 2 | 4 | **SW fails at 8 sites** | constructs |
| $G_k$ | 3e8 | 1377 | 318 | 2 | 1 | 4 | **SW fails at 10 sites** | constructs |
| $T_k$ | 3e8 | 1401 | 325 | 2 | 3 | 5 | **SW fails at 14 sites** | constructs |

Handing the failing vectors to `make_interp_spline` gives
`LinAlgError: Colocation matrix is singular.` on all six.

**Why it is structural and not a placement that could be improved.** With multiplicity 3 at
consecutive break points $b_j < b_{j+1}$ and no data site between them, the B-spline whose support
is exactly $[b_j, b_{j+1}]$ sees no interpolation condition; Schoenberg–Whitney fails for **any**
placement of the remaining knots. `BREAK_POINT_ALL` on the production range is one break point per
4.5 samples (median segment occupancy **4**, minimum **0**), so empty segments are not an accident
of one grid — they appear on all six.

**And the payment cannot be moved away from the break.** A multiplicity-3 knot costs three
interior knots, and the interpolating knot vector has a fixed length $n+k+1$. Starting from the
shipped default (knots at the data sites, $t_i = u_{i-2}$), Schoenberg–Whitney tolerates at most
**one** net removal below any site and one above, so the three knots a repeated knot consumes must
be removed **locally**, from the intervals adjacent to the break. Measured: keeping the two knots
that bracket the break and paying elsewhere fails SW at exactly the site adjacent to the break
(index 861 on $G_k$, 862 on $T_k$, $k=10^5$); paying two knots in the smooth interior fails at 50
and 32 sites. So a $C^0$ knot at a break necessarily coarsens the spline in the three intervals
around that break — which is exactly where $\varphi$ is least smooth.

**Every declared break point lies strictly inside a grid interval.** Measured on all six grids and
both kinds: **0 exact coincidences** with a sample; fractional position within its interval
0.0011–0.9999 (`BREAK_POINT_ALL`), 0.6431 and 0.3230 (`BREAK_POINT_DISCONTINUITY`). The production
grid is the cosmology's source grid and knows nothing about the equation of state, so nothing
resolves a break point from either side.

### 2. The kink the issue names is 1–4 % of the error it is charged with

The declared discontinuity in range at $k=10^5$ is `QCD_EOS`'s `T_LO` crossing, $z=4.25344\times10^7$
— where $H(z)$ jumps by 4.4e-04 (`GkTk-remedial` log 02). $\varphi$ was reconstructed from the
dense reference (ten points per interval, the producer's own $\theta$ minus the same leading term
the consumer uses) and cubics were fitted to the ten reference points on each side of the break, in
the one grid interval either side — the finest one-sided estimate the reference supports:

| sector | $k$ | $[\varphi]$ | $[\varphi']$ | kink term $\|[\varphi']\|h/8$ | measured consumer error |
|---|---|---|---|---|---|
| $G_k$ | 1e5 | +1.88e-07 | −5.55e-06 | **1.60e-08 rad** | 1.907e-06 rad |
| $T_k$ | 1e5 | −1.59e-07 | −4.99e-05 | **1.44e-07 rad** | 3.186e-06 rad |

So the slope discontinuity at the declared break accounts for **1 %** of the $G_k$ figure and
**4 %** of the $T_k$ figure. (Widening the one-sided window to two intervals moves $[\varphi']$ by
two orders — 6.8e-04 and 1.2e-03 — which is the signature of smooth-but-unresolved data, not of a
slope discontinuity: a real kink gives a window-independent jump.)

What the error actually is: the interpolation error of the base spline around the break is a
**string of arches, one per grid interval, growing towards the break and decaying away from it** —
1.74e-06, 2.70e-06, **3.19e-06** (the interval holding the break), 2.90e-06, 2.00e-06 on $T_k$ at
$k=10^5$ — against 8.60e-07 over the rest of the range. A single kink at a point decays by
$(\sqrt3-2)\approx0.27$ per interval; this decays by ~0.7. The non-smoothness is **spread over
$\pm3$ grid intervals**, which is the scale on which $\varphi$'s own structure lives there (the
inferred $|\varphi''''|\approx2.4$ implies a variation scale of ~3.7 grid intervals, and the
`BREAK_POINT_ALL` knots are one per 4.5 samples).

### 3. What each scheme actually does at the production geometry

Consumer phase error, and `theta_deriv` against $\omega$, for every constructible scheme.
`base (today)` is the shipped `make_interp_spline` default and reproduces
`docs/gktk-remedial-verification.md` §3.5 and §3.6 to the printed digits, which is the harness's own
check that it is measuring the same thing.

**QCD $G_k$, $k=10^5$** (span 1.3728e+09 rad, 1 ulp = 2.3842e-07 rad; target ≤1e-06 rad **and**
≤2 ulp)

| scheme | max \|θ−ref\| | ulp | at $z$ | `theta_deriv` deep interior |
|---|---|---|---|---|
| base (today) | 1.9073e-06 | 8.00 | 4.2439e+07 | 2.3123e-07 |
| ALL × 1 | 1.4305e-06 | 6.00 | 4.1473e+07 | 1.9061e-07 |
| DISC × 1 | 1.6689e-06 | 7.00 | 4.2439e+07 | 2.0033e-07 |
| DISC × 2 | 1.4305e-06 | 6.00 | 4.1473e+07 | 1.9372e-07 |
| **DISC × 3 ($C^0$, the repeated knot)** | **3.8147e-06** | **16.00** | 4.2528e+07 | 1.4413e-07 |
| ALL × 3 ($C^0$) | — | — | — | **does not construct** |

**QCD $T_k$, $k=10^5$** (span 6.1744e+07 rad, 1 ulp = 7.4506e-09 rad)

| scheme | max \|θ−ref\| | ulp | at $z$ | `theta_deriv` deep interior |
|---|---|---|---|---|
| base (today) | 3.1859e-06 | 427.60 | 4.2439e+07 | 6.9922e-07 |
| ALL × 1 | 2.6138e-06 | 350.82 | 4.3337e+07 | 6.6353e-07 |
| DISC × 1 | 2.6660e-06 | 357.82 | 4.3337e+07 | 6.6443e-07 |
| DISC × 2 | 2.6314e-06 | 353.18 | 4.3337e+07 | 6.6443e-07 |
| **DISC × 3 ($C^0$, the repeated knot)** | **6.7904e-06** | **911.40** | 4.2528e+07 | 6.6443e-07 |
| ALL × 3 ($C^0$) | — | — | — | **does not construct** |

**The ten rows that must stay at 1.00 ulp.** `ALL × 1` — the only scheme that improves $k=10^5$ —
moves two of them:

| case | base | ALL × 1 |
|---|---|---|
| QCD $G_k$ $k=10^7$ | 1.5259e-05 rad (**1.00** ulp) | 4.5776e-05 rad (**3.00** ulp) |
| QCD $T_k$ $k=3\times10^8$ | 3.0518e-05 rad (**1.00** ulp) | 9.1553e-05 rad (**3.00** ulp) |

and its `theta_deriv` at QCD $G_k$ $k=10^7$ goes **1.0984e-05 → 1.7113e-03**, a factor 156. Prompt
§4 makes those rows a requirement and README §4 makes a regression there a stop condition, so
`ALL × 1` is not a candidate either, whatever it buys at $k=10^5$.

**Best case over the whole family: 6.00 ulp / 1.4305e-06 rad ($G_k$) and 350.82 ulp /
2.6138e-06 rad ($T_k$), against targets of 2 ulp and 1e-06 rad — and only at the cost of a
regression elsewhere.** The target is not reachable by a knot vector.

### 4. `theta_deriv` against $\omega$ — the two contributions prompt §4 asks to be separated

Prompt §4 asks how much of the 2.3e-07 – 3.3e-04 relative miss is the knots' and how much is
`[02-qcd-T-z-spline-node-tolerance]`'s scatter of $\omega^2$ between neighbouring nodes. Measured
on the deep interior (`[20:-20]`), with a third column that is the decisive control: the same
identity scored with the $\varphi$ spline's derivative **removed**, i.e. $|k\,{\rm rate}(z)|$
against $\omega$ alone.

| sector | $k$ | today | $\varphi'$ set to 0 | $\|d\varphi/dz\|/\omega$ | range of stored $\varphi$ |
|---|---|---|---|---|---|
| $G_k$ | 1e5 | 2.3123e-07 | 6.8497e-06 | 6.7610e-06 | 1112 ulp |
| $T_k$ | 1e5 | 6.9922e-07 | 3.0018e-03 | 3.0022e-03 | 1.18e+07 ulp |
| **$G_k$** | **1e7** | **5.9403e-06** | **1.9982e-06** | 7.9385e-06 | **6.0 ulp** |
| $T_k$ | 1e7 | 2.7355e-07 | 3.2091e-03 | 3.2088e-03 | 9.31e+04 ulp |
| **$G_k$** | **3e8** | **1.7475e-04** | **1.7392e-05** | 1.8377e-04 | **2.0 ulp** |
| $T_k$ | 3e8 | 8.4188e-06 | 3.1342e-03 | 3.1341e-03 | 3.01e+03 ulp |

**The separation, and it is neither of the two the prompt offers.** The two rows that miss the
$10^{-6}$ target are QCD $G_k$ at $10^7$ and $3\times10^8$. At those wavenumbers the recovered
$\varphi$ has a dynamic range of **6.0** and **2.0 ulp of the stored phase** — it takes **7** and
**3 distinct values** over 1218 and 1377 samples, and every value is an exact multiple of
${\rm ulp}(\theta)$ (measured: $\max|\varphi - {\rm round}(\varphi/{\rm ulp})\cdot{\rm ulp}| = 0$).
$\varphi$ there is the rounding of $\theta$ and nothing else, and differentiating that staircase is
**3× and 10× worse than contributing nothing at all**. So:

- **the knots' share** is what the best constructible knot vector recovers: at $G_k$ $k=10^5$,
  2.3123e-07 → 1.4413e-07 (`DISC × 3`), i.e. at most **38 %**, and that scheme costs 8 → 16 ulp of
  consumer phase; at $10^7$ and $3\times10^8$ every scheme leaves the figure unchanged to the
  printed digits;
- **`[02-qcd-T-z-spline-node-tolerance]`'s share** is not what dominates either: on the three
  $T_k$ rows the identity is missed by 3.0e-03–3.2e-03 relative when $\varphi'$ is dropped, and the
  $\varphi$ spline recovers all but 2.7e-07–8.4e-06 of it, i.e. the spline is doing its job;
- **what dominates the two misses is the $\varepsilon k\tau$ storage granularity of $\varphi$
  itself** — `[00-consumer-anchoring-floor]`, the follow-up `primitive_phase.py`'s module docstring
  already names, because $\varphi$ is recovered as a difference of two numbers of size $k\tau$.
  Opened as `[02-consumer-phi-below-the-storage-granularity]`.

At $k=10^5$, where $\varphi$ has 1112 ulp of range, the spline's derivative genuinely earns its
place: 6.8497e-06 → 2.3123e-07, a factor 30.

### 5. Cost

Not measured, and the row is vacuous: nothing about the build changed, so the
`GkTk-remedial` prompt-13 figure of **0.0010 s / 468 integrand evaluations per $G_k$ object**
stands unaltered. (Had a knot vector shipped, the extra work would have been one
`_cosmology_break_points` call and one `np.argsort` per object.)

### 6. Tests and the tree

| suite | result |
|---|---|
| `PYTHONPATH=. ./venv/bin/python -m unittest discover -s ComputeTargets/tests -t .` | **Ran 339 tests in 258.8 s — FAILED (failures=1)** |
| `PYTHONPATH=. ./venv/bin/python -m unittest discover -s LiouvilleGreen/tests -t .` | not re-run |

**The count did not fall: 339, as at `0c61799`.** The one failure is
`test_tk_wkb_phase.TestCost.test_wall_time_per_object` —
`AssertionError: 0.06374904210679233 not less than or equal to 0.06` — and it is **not this
commit's**: `git status` shows no `.py` file modified, so the code under test is byte-identical to
`0c61799`, where the orchestrator measured 339 OK. Run alone on an otherwise idle machine it still
fails, four times in four: **0.0652, 0.0979, 0.0638, 0.0639 s** against the 0.06 s limit, each a
best-of-3 minimum. This is the already-open `[07-tk-per-object-cost-is-all-setup]`, whose own text
records the figure as "0.0494–0.0516 s across seven timed runs, straddling prompt 07 §3 item 6's
≤0.05 s rather than clearing it"; the threshold in the test is 0.06, and on this machine today the
measurement has drifted above it. Prompt 01's `WKB_mod_2pi` change costs +48 ns per reduction and
this object performs 1,384 of them — 66 µs, four orders below the 4 ms margin — so it is not the
cause either. Not repaired: `test_tk_wkb_phase.py` is not on this prompt's file list, the issue is
open and owned, and loosening a wall-clock threshold is not this prompt's business.

`LiouvilleGreen/` contains no file this commit touches — this commit touches no `.py` file at all —
and the orchestrator measured **148 OK** at `0c61799`, which is this commit's parent and the tree
the code is unchanged from. `black --check` is unaffected for the same reason; the two Markdown
directories it already fails on (`docs/`, `AdaptiveLevin/`) are untouched (prompt 01's observation 5).

---

## Observations not acted on

1. **`docs/gktk-remedial/verify_production_path.py` builds its own $G_k$ consumer.** `:557` and
   `:1173` call `PrimitivePhase(...)` directly rather than going through `GkSourcePolicyData`, so
   six of the twelve §3.5 rows and both `theta_deriv` $G_k$ columns are blind to anything a
   production call site passes. The $T_k$ half does go through `TkSourceFunctions`. Anyone
   re-attempting this prompt needs to know that half its acceptance table cannot move. Opened as
   `[02-verify-script-builds-its-own-Gk-consumer]`; not fixed, the file is not on this prompt's
   list and it is another campaign's verification driver
   (cf. `[13-scoped-run-driver-k-grid-literal]`).

2. **The recovered $\varphi$ is below the storage granularity at large $k$.** Three distinct values
   over 1377 samples at QCD $G_k$ $3\times10^8$; seven over 1218 at $10^7$. Opened as
   `[02-consumer-phi-below-the-storage-granularity]`. The obvious remedy — do not add a spline
   derivative whose ordinates span a couple of ulp — is a production change this prompt may not
   make, and it belongs with `[00-consumer-anchoring-floor]` rather than on its own.

3. **`docs/gktk-remedial-verification.md` §3.5's stated mechanism is not what the measurement
   supports.** It reads "The residual $\varphi$ has a kink there and a cubic spline of it does not";
   §2 above puts that kink at 1 % and 4 % of the error. Nothing above §7 of that document may be
   edited (README §0.2, §4) and this is not a close-out, so it was left alone; the correction lives
   in the narrowed board entry and in this log. A future §8 should carry it.

4. **`BREAK_POINT_ALL` is one break point per 4.5 production samples on QCD.** 226–325 of them
   inside a 1016–1401 sample range. Any future consumer that wants to respect them needs either a
   sample grid built around them or a representation that is not an interpolating spline on the
   cosmology's own source grid. That is a bigger decision than a knot vector and it is not opened
   as an issue, because it is a design question, not a defect.

5. **`test_tk_wkb_phase.TestCost.test_wall_time_per_object` fails at `0c61799` on this machine.**
   0.0638–0.0979 s against a 0.06 s limit, four runs out of four, with no `.py` file changed. It is
   `[07-tk-per-object-cost-is-all-setup]`, already open and already described as straddling its
   threshold; what is new is that on this machine today it straddles the *test's* 0.06 s as well as
   prompt 07's 0.05 s. No new issue opened, and the existing entry was not edited — it is not the
   entry this prompt is allowed to touch.

6. **The `GkTk-remedial` board's "Next step" for this issue names the wrong remedy.** It says "give
   `PrimitivePhase` a knot vector that repeats a knot at each `integration_break_points` value
   inside the sample range — the remedy prompts 02 and 03 built for the quadrature and prompts 18
   and 19 for the ODE". The quadrature and the ODE **choose their own abscissae** and can therefore
   put a panel edge or a step boundary exactly at a break; an interpolating spline is stuck with
   the samples it is given. That is the asymmetry the plan missed, and it is now written into the
   narrowed entry.

---

## State handed to the next prompt

**There is no next prompt in this campaign.** Prompt 02 was the last of two, and it is `BLOCKED`
rather than complete, so the campaign closes at **1 / 2** unless the user directs otherwise.

1. **The tree is `0c61799` plus documentation.** No `.py` file differs from prompt 01's commit.
   Reverting this commit costs nothing but the record.
2. **`[13-consumer-spline-crosses-eos-break-points]` stays open**, on the `GkTk-remedial` board §3,
   narrowed: the declared-discontinuity kink is measured at 1.6e-08 / 1.4e-07 rad, 1 % and 4 % of
   the 1.907e-06 / 3.186e-06 rad it is charged with; a repeated-knot vector is singular at
   `BREAK_POINT_ALL` on all six production grids and 2× worse at `BREAK_POINT_DISCONTINUITY`; and
   the residue is $\varphi$'s own structure on a $\pm3$-interval scale, which the production
   sample grid does not resolve. Its **Assigned** line now says this campaign tried and stopped.
3. **The decision README §7 D2 reserves is live and is the user's.** The three ways forward, in
   the order this log would rank them: (a) accept the issue as a property of the sample grid and
   close it as `INERT` at the QCD Liouville–Green truncation floor of ~1e-03 rad, which is 300×
   above the worst figure; (b) re-open it against the *sample grid* rather than the spline — split
   the production source grid at `integration_break_points` so that a break is resolved from both
   sides, which is a `main.py`/`BackgroundModel` change and a different campaign; (c) attack
   `[00-consumer-anchoring-floor]` first, since §4 above shows the $\varphi$ storage granularity,
   not the knots, is what limits `theta_deriv` at $k\ge10^7$.
4. **Two new issues** are on this campaign's board §3 and in `docs/OPEN_ISSUES.md`:
   `[02-verify-script-builds-its-own-Gk-consumer]` (§4 of the index — verification debt) and
   `[02-consumer-phi-below-the-storage-granularity]` (§1.6 — the phase-representation campaign).
   The index count goes 53 → 55.
5. **What a re-attempt must not repeat.** Do not measure a knot scheme at $k=10^5$ alone: the two
   schemes that help there are the two that regress $10^7$ and $3\times10^8$. Score all six QCD
   (sector, $k$) cases and all six LambdaCDM ones, and score `theta_deriv` as well as the phase —
   `DISC × 3` improves `theta_deriv` while doubling the phase error, and a one-number acceptance
   test would have shipped it.
