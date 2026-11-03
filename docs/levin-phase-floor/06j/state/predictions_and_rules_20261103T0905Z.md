# levin-phase-floor 06j — predictions, sampling rules, seeds and operational definitions

Written 2026-11-03 (UTC) on mps026298 by the 06j agent, **before any probe or reading** and before the
samples are drawn. This file is not rewritten afterwards: a correction is a new file, recorded as a
deviation. The samples themselves are written by `scripts/draw_06j.py` into `out/sample/` (each with
a provenance header and the seeds below).

Tree: the worktree `/Users/ds283/Documents/Code/SecondaryGWKit-lpf06j-dfa19d9` at `dfa19d98`
(production code `1c5336e`'s, the tool `b5626ecb`'s). Inputs (read only): 06g's census run
`levin-phase-floor-06g-census-20261101T114315` (checkpoint `03adcbba…`, 16 region pickles), 06h's census
`levin-phase-floor-06h-census-20261101T005037` (checkpoint `3104c18e…`), 06g's
`census/out/v8_calls.json` (`1e9a62b0…`), `census/out/v8_slow_leaves.json` (`b4643033…`),
`census/out/v2_rows.csv.gz`, 06a's census (the 720 stationary-point rows, `spscan` rule).

## 1. Predictions, copied verbatim from prompt 06j §4

README §5 rule 3. Copy them into the log unchanged. Each is scored **holds**, **fails** or
**undecided** against the rule as written. None is a stop.

**Part S:**
- **S1 — the slow leaves sit on seams.** **Rule:** at least 80 % of the sampled slow leaves hold a
  seam in their span, and the fraction is at least five times the null's.
- **S2 — $r$ stays at 1/2 where the seam is kept.** **Rule:** on at least 80 % of the sampled leaves
  that hold a seam, $r$ stays within $[0.4, 0.6]$ at every one of depths 21–24 along the branch
  holding the seam.
- **S3 — the slow leaves are not noise.** **Rule:** on at least 80 % of the sampled leaves,
  $\Delta_1$ exceeds ten times the larger of S-c's two scatters.
- **S4 — splitting at the seam restores trust.** **Rule:** on at least 80 % of the sampled leaves
  that hold a seam, both sides of the split would be trusted under rule (b).

**Part W:**
- **W1 — the references count.** **Rule:** on every W call, the four readings agree within the
  count tolerance.
- **W2 — 06g is honest on V2's rows.** **Rule:** on every W-5 call, $|v_g - {\rm ref}| \le e_g$.
- **W3 — the parent is the one in error.** **Rule:** on every W-5 call, $|v_{06h} - {\rm ref}| > e_{06h}$.
- **W4 — the pattern is general.** **Rule:** over the W-12 calls, 06g is within its error on at
  least 11, and 06h is outside its error on at least 6.

**The basis of the 80 % and the factors.** S1–S4 are each read as a test of (J) against (N), and
80 % leaves room for a minority of leaves of the other kind. The factor of five over the null and
of ten over the scatter are chosen generously, so that a pass is not a near-miss: README §5's
practice for a cut-off that decides between remedies. Every number behind each verdict is reported,
so that a different reading can be made from the record.

## 2. Part S — the sample (prompt §2), seed 20261103

**Eligible calls.** At each rtol: the lost calls of `v8_calls.json` (`"lost"`) with
`stationary_point_row` False, in the four groups `G+q+r`, `G+q-r`, `G-q+r`, `G-q-r` (the `G±q±r`
groups that have them; the one `q+r` call and the `G-r` calls are not in a `G±q±r` group), that return
at least one scoped slow leaf in `v8_slow_leaves.json` (its scope is `v8_slow_leaves.py`'s: returned,
untrusted as returned (`trusted_final` 0), $1/4 \le r < 1$, $\Delta_1 > \phi_P$). A lost call with no
scoped leaf (its untrusted leaves all $r \ge 1$) has nothing to probe and is not eligible.

**Strata.** The number of slow-leaf clusters per call, by `v8_slow_leaves.py`'s clustering rule
(depth-20 scoped leaves sorted by start; consecutive ones in one cluster if the gap is at most 64
depth-20 widths of the call, $64 (b-a) 2^{-20}$): $c = 1$; $c \in \{2, 3\}$; $c \ge 4$. Four groups
times three strata is twelve cells per rtol.

**The draw.** One `random.Random(20261103)` stream. For rtol in (`1e-7`, `1e-8`), group in
(`G+q+r`, `G+q-r`, `G-q+r`, `G-q-r`), stratum in ($c=1$, $c\in\{2,3\}$, $c\ge4$), in that order: the
cell's eligible calls sorted by (row, index) are shuffled by the stream into the cell's ranked list;
the cell's sampled call is the first in its list whose row (`shard-serial`) has not been sampled
already, at either rtol (so no two sampled calls share a row: a row's `G+q-r` and `G-q+r` calls can
return identical leaves). If a cell has no such call, the group's other strata's lists are tried in
the order ($c=1$, $c\in\{2,3\}$, $c\ge4$), taking the first unused entry. 12 calls per rtol, 24 in all.

**Replacement (J0).** A call that fails J0 is replaced by the next unused entry of its own cell's
ranked list (same rtol, group and stratum), under the same row rule; every replacement is listed.

**The leaves.** In each sampled call, every scoped slow leaf if there are at most 10; otherwise 10,
chosen across its clusters: the clusters in order of position (centre ascending), each cluster's
leaves in an order drawn by `random.Random(20261103 + 1000 * k)` (k the call's 0-based position in
the sample), then round-robin over the clusters (one leaf from each cluster in turn, skipping
exhausted clusters) until 10 are taken. Leaves are identified by their exact (start, end) doubles
from the census's region pickle `checked_leaves_06g` (checked equal to `v8_slow_leaves.json`'s).

## 3. Part S — the measurements, as they will be made (no code change; probes in the harness's process)

The **leaf rule** is `_adaptive_levin`'s `_check_leaf`, evaluated in isolation on a span $[a, b]$ by
calling the tree's own `AdaptiveLevin.levin_quadrature._adaptive_levin_subregion` with the call's own
arguments (`f`, `_Basis_SinCos(theta)`, `chebyshev_order`, `rtol` from the capture's `kw`, a fresh
vectorise cache per call, `diagnostics_path` under this records directory): $V_P$ on $[a,b]$; $c =
(b+a)/2$; $V_L$, $V_R$; $c_L = (c+a)/2$, $c_R = (b+c)/2$; the four quarters. $\Delta_1 = |V_P - (V_L +
V_R)|$, $\Delta_2 = |(V_L+V_R) - ((V_{LL}+V_{LR}) + (V_{RL}+V_{RR}))|$, $r = \Delta_2/\Delta_1$ (0 if
both are 0, $+\infty$ if only $\Delta_1$ is 0 or either is not finite), $\phi_P$ the parent solve's
`phase_err`. **Trusted under rule (b):** $r < 1/4$, or $r < 1$ and $\Delta_1 \le \phi_P$. A probe of the
census leaf's own span must reproduce the census's $\Delta_1$, $\Delta_2$, $r$ and $\phi_P$ bit for
bit; a leaf where it does not is reported, and its probes are not scored.

**Seams (S-a).** The call's seams are `tools.filon_reference.production_edges`'s points **before**
merging, each with its kind, by walking the same closures the tool walks: `phase_knot` (each active
factor's phase spline knots `fac.phase._spline.t`), `amp_knot` (the Green's function amplitude spline's
knots `G._Gk.sin_amplitude._spline.t`), `table_node` (each distinct cumulative table's `_u_asc`),
`switch_mid` (its nearest-node switch midpoints `asc[:-1] + (asc[1:] - asc[:-1])/2`), `break` (its
`_breaks`); with which table or factor. The merged set (the tool's own 1e-9 rule) is checked equal to
`production_edges(call)[0]`. A leaf **holds a seam** if $a < x_s < b$ for some seam $x_s$ (strict, on the
doubles). Otherwise its distance to the nearest seam is reported in units of its width $w = b - a$,
both from its nearer end and from its centre. **The null:** for each call, the fraction of the
$2^{20}$ depth-20 cells $[a_0 + i W 2^{-20}, a_0 + (i+1) W 2^{-20})$ ($W$ the call's span) that hold
at least one seam; the sample's null is the mean of its calls' nulls weighted by the call's sampled
leaves. **The comparison:** the call's returned leaves trusted as returned (`trusted_final` 1) at
depth $\ge 15$, the same seam test, with each one's own null at its depth.

**S1 scoring.** Fraction of sampled slow leaves holding a seam, against 0.8 and against 5 times the
sample's null.

**S-b.** For every sampled leaf, the leaf rule on its span (depth 20) and down two branches to depth
24, at every node of a branch the rule evaluated on both children:
- for a leaf that holds a seam: the **seam branch**, at each level the child that holds a seam (if
  both do, the one with the larger $\Delta_1$); and the **off-seam branch**, at depth 21 the child that
  holds no seam (if both do, recorded as having none), then at each level the child with the larger
  $\Delta_1$ (the stricter choice: an unlisted non-smooth point would be followed);
- for a leaf that holds no seam: the **max branch** (at each level the child with the larger
  $\Delta_1$) and the **min branch** (the smaller).

Each node records depth, span, whether it holds a seam, branch (Levin or Clenshaw–Curtis), $V_P$,
$\Delta_1$, $\Delta_2$, $r$, $\phi_P$, $\Delta_1/\phi_P$ and trusted under (b). **S2 scoring:** over the
leaves that hold a seam, the fraction whose seam-branch $r$ is in $[0.4, 0.6]$ at each of depths 21,
22, 23 and 24. A leaf whose seam branch ends early (the seam falls on a midpoint, so no child holds
it) is excluded from the denominator and listed; if more than 20 % are excluded, S2 is undecided.

**S-c.** For every sampled leaf, two perturbation sets of 16 draws each, applied to the leaf's own
solve $V_P$ and, descriptively, to its halves:
- **(i) the nodes:** a pass-through at the module attribute `chebyshev_matrices` (which both branches
  call for their grids) returns the grid with every node moved by $k$ ulp, $k$ uniform in
  $\{-4, \dots, 4\}$, independently per node and per call, from `numpy.random.Generator(PCG64(SEED))`
  with SEED = 2026110301 + 100 d (d = 0..15 the draw), a fresh generator per leaf and draw; the
  differentiation matrix and weights are unchanged;
- **(ii) the integrand's inputs at the nodes:** `f_sin`, `f_cos` and the phase dict's `theta`,
  `theta_mod_2pi` and `theta_deriv` each evaluated at $x + k(x)\,{\rm ulp}(x)$, with
  $k(x) = ({\rm int}(\text{SHA-256}(\text{"06j-ii"}, d, x.{\rm hex}())[:4]) \bmod 9) - 4$, so that one
  node moves the same way in every callable; the rule's own nodes unchanged.

**The scatter** of a set is the rms over its 16 draws of $V_P^{(d)} - V_P$ (the unperturbed value).
Reported against $\Delta_1$ (the census leaf's) and against $\phi_P$; also, descriptively, the rms of
$(V_L + V_R)^{(d)} - (V_L+V_R)$ and of $\Delta_1^{(d)} - \Delta_1$. **S3 scoring:** the fraction of sampled
leaves with $\Delta_1 > 10 \max({\rm scatter}_{(i)}, {\rm scatter}_{(ii)})$.

**S-d.** For every sampled leaf that holds a seam: the leaf rule on the sub-spans either side of the
seam, $[a, x_s]$ and $[x_s, b]$; a leaf holding $k > 1$ seams is split at all of them, into $k+1$
sub-spans. Each records $r$, $\Delta_1/\phi_P$ and trusted under (b). **S4 scoring:** the fraction of
seam-holding leaves on which every sub-span is trusted under (b).

**S-e.** At each seam held by a sampled leaf: the integrand $F = f_{\sin}\sin\theta + f_{\cos}\cos\theta$
(sin and cos of `theta_mod_2pi`, as the rule evaluates it) and each ingredient (`f_sin`, `f_cos`,
`theta`, `theta_mod_2pi` with differences reduced into $(-\pi, \pi]$, `theta_deriv`) at $x_s \pm j h$,
$j = 1, 2, 3$, for $h = w\,2^{-2}, w\,2^{-4}, w\,2^{-6}, w\,2^{-8}$ ($w$ the leaf's width); on each
side the quadratic through the three points gives the one-sided limit, first and second derivative
at $x_s$; the jump is right minus left. The same stencil at a control point (the midpoint between
the seam and the leaf's farther end, if no seam lies within $3 h_{\max}$ of it) gives the stencil's
noise. Each jump is reported against the control's and against $|F|$; the representation that
changes there is named from the seam's kind and table or factor, with $z = e^{x_s} - 1$.

## 4. Part W — the sample (prompt §3)

**W-5.** The rows of `v2_rows.csv.gz` with `reading` = `total` and `violates` True, at each rtol (5
expected at each), their call index 1 (`G+r`). If a row's captured integrand at `1e-7` equals its
integrand at `1e-8` by 06g's criterion (`compare13.describe`: route, span, the active factors' signs,
each factor's phase spline and leading table by bytes; the integrand at 65 points and
`production_edges`), one reference, read on the `1e-7` capture, serves both, and the equality is
recorded.

**The call-level set,** recomputed from the two checkpoints by `values_beyond06g.py`'s rule: every
matched call outside the 720 with $|v_g - v_{06h}| > e_g + e_{06h}$ (finite errors; exact
arithmetic). It must reproduce 62 calls at `1e-7` and 80 at `1e-8`.

**W-12** (no seed: the rule is deterministic). At each rtol, the candidates are the call-level set less
W-5's calls; at `1e-8` also less every (row, index) taken at `1e-7`. Sort by the ratio
$|v_g - v_{06h}|/(e_g + e_{06h})$ ascending (ties by row, then index); $n$ candidates, ranks
$0..n-1$. For $j = 0..5$ in order, the target rank is $p_j = (j + 1/2)/6 \cdot (n - 1)$; the pick is
the unpicked candidate nearest $p_j$ in rank (ties to the lower rank) among those whose group has
fewer than two picks at this rtol so far. 6 per rtol, 12 in all.

**Values.** $v_g$, $e_g$, regions, depth and converged from 06g's census checkpoint; $v_{06h}$,
$e_{06h}$, regions, depth and converged from 06h's census checkpoint (`3104c18e…`, checked).

## 5. Part W — what is measured for each call (prompt §3)

- **The reference:** `tools.filon_reference.reading` for (qawo, fcc) × (1, 2), each in its own process
  by 06g's `one_reading.py` with six workers; usable per the tool's rule (`tmap.all_monotone`, or every
  non-monotone segment integrated directly). The count tolerance is
  $\min(e_g, e_{06h})/100$; the spread is the largest minus the smallest reading (`tools.filon_reference
  .scatter`). The reference value is the FCC reading at resolution 2 (06g's and 06i's convention).
- **The codes against it:** $|v_g - {\rm ref}|/e_g$ and $|v_{06h} - {\rm ref}|/e_{06h}$, exactly
  (fractions of the 30-digit reading), with each code's regions, depth and converged.
- **What the codes returned:** 06g's returned leaves from its census region pickle (depth, $r$,
  trusted flag, value, $V_P$) with S-a's seam test on the capture's seams; 06h's from its census
  region pickle (depth, $r$, trusted) where recorded. 06h's leaves' values are not in its records; they
  are recomputed by solving each 06h leaf's span with the tree's `_adaptive_levin_subregion` (whose
  value path is 06h's, `git diff 4f4f9ed 1c5336e`), and accepted only if their exact sum reproduces
  06h's recorded value bit for bit.
- **The constant move** (the `G+r` calls): over the common refinement of the two partitions (every
  endpoint of either), each code's partial sum on each cell of the coarser partition that contains
  it — for each 06h leaf, $v_{06h}$(leaf) against the sum of 06g's returned leaves inside it (and the
  reverse wherever 06g's partition is the coarser); the move is localised if one such cell carries
  at least 90 % of $|v_g - v_{06h}|$.

## 6. Scoring conventions, fixed now

- S1–S4 are scored on the leaves for which the measurement could be made; a leaf whose probe does
  not reproduce its census numbers is excluded and listed; if more than 20 % of the leaves a
  prediction counts are excluded, it is **undecided**.
- W1: **holds** if every W call has four usable readings agreeing within its tolerance; **fails** if
  any call with four usable readings does not agree; otherwise (some call lacks usable readings, and
  none that has four disagrees) **undecided**.
- W2–W4 are scored against the FCC resolution-2 reading of every W call that has one, with the
  reading count and spread stated beside it ("every reference is scored with the readings it has");
  a call without a usable FCC r2 reading makes its prediction **undecided** unless the others already
  decide it.
