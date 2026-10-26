# levin-phase-floor 05a §6 (the wider re-measurement): found and not acted on

These are for the orchestrator to place on the boards. Nothing here was changed. The numbers are in
`out/wider_05a.txt` and `state/probe_windows.json`, from the run
`levin-phase-floor-05a-wider-20261026T123047`. In the text:
- $c$ is the code commit's parent `c4eaefd`, which is 05's $c$;
- $t$ is the code commit `1fc349c`, run from the main checkout at `149c393`.

## X1. §9's halving rule admits aliased windows, and T1 as written fails on exactly those

**Proposed tag:** `[05a-t1-halving-rule-admits-aliased-windows]`. It bears on T1's verdict. The
question was put to the user through the orchestrator after the launch and before any scoring. It
is not yet answered.

**What was measured.**
- §9 halves the window while "the largest step of the unwrapped samples" is $\pi/2$ or more in
  either tree. Read as written, that is `max|diff(numpy.unwrap(θ))|`, the quantity §3 test 3's
  guard computes. `state/predictions.md` fixed this reading before the run.
- After unwrapping, a step is folded into $(-\pi, \pi]$. So a window whose samples advance by many
  cycles each can still pass the rule.
- Of the 70 scored windows, 36 pass the rule at a level where the phase's own step per sample,
  $|d\theta_T/dz|(b-a)/64$, is 4.9 to 352 rad. These 36 windows sit at $z$ = 2.09 to 4800.
- On those 36, $\sigma_\theta$ under $t$ is 0.016 to 0.23 $\varepsilon|\theta_T|$, which is 0.13
  to 0.30 of $\varepsilon z|d\theta_T/dz|$. This is `numpy.unwrap` folding the samples whole
  cycles per index, so that the rounding of the sample abscissae appears as jitter. It is what the
  code log's observation 1 described.
- **T1 as written fails in exactly these 36 windows.** In each, the $t$ clause is false and the
  $c$ clause is true.
- **Every other window passes.** In the 34 other scored windows, $\sigma_\theta$ under $t$ is at
  most $1.3\times10^{-5}\,\varepsilon|\theta_T|$.
- **Read by the true step, T1 holds in all 70 windows.** That reading also requires
  $|d\theta/dz|(b-a)/64 < \pi/2$, and is reported unscored. Under it, $\sigma_t$ is
  $4.4\times10^{-10}$ to $1.31\times10^{-5}\,\varepsilon|\theta_T|$, at most 0.013 of the bound,
  and $\sigma_c$ is 0.198 to 0.374.

**Impact.** As written, T1's verdict measures the unwrap and not the phase, on half the scored
windows. Every level of both trees' ladders is saved, so T1 can be re-scored under either reading
without re-running: `state/probe_windows.json`, `scripts/rescore_t1.py` and `out/cmd/24_rescore_t1.txt`.

**Next step.** The user's ruling on which reading is the rule's.

**Added after the ruling.** It was relayed to this session at about 13:23Z: the user chose "True
step (Recommended)".
- T1 is now scored on windows resolved by the true step. It **holds in 70 of 70**:
  - $\sigma_t$ is $4.4\times10^{-10}$–$1.31\times10^{-5}\,\varepsilon|\theta_T|$, at most 0.0131
    of its bound;
  - $\sigma_c$ is 0.198–0.374.
- No re-probe was needed (`out/cmd/26_T1_true_step_scored.txt`). The literal reading is reported
  beside it, unscored.
- The laptop amends §9's wording in 05a's records commit.
- What remains of X1 is the record that the wording as first written admitted aliased windows.
  Whether to open the proposed tag is the laptop's call.
- `state/predictions_note_T1_reading.md` gives the timing as the files show it.

## X2. The abscissa term remains in T's phase as production samples it ($u$)

**Proposed tag:** `[05a-t-phase-abscissa-rounding-remains]`. T1's $u$-sampled clause asks for this
proposal wherever $\sigma_\theta$ sampled in $u$ under $t$ exceeds $0.1\,\varepsilon|\theta_T|$.

**What was measured.** All numbers are at the resolved, true-step level.
- **14 of 70 windows exceed $0.1\,\varepsilon|\theta_T|$, at 0.10 to 0.22.**
  - The windows are the $z \gtrsim 10^2$ ends of the mid-band rows `mid-G±r-*`, `mid-Gqr-2` and
    `ctrl-mid-2`.
  - There the abscissa term $\varepsilon\tfrac{z}{1+z}|d\theta_T/du|$ is 0.69 to 0.98
    $\varepsilon|\theta_T|$.
- **Over all 70 windows,** the $u$-sampled $\sigma$ under $t$ is 0.14 to 0.29 of the abscissa
  term. Under $c$ it is 0.21 to 0.40 $\varepsilon|\theta_T|$.
- **At $z < 1$ the keyword does reduce it.** The six windows at $z$ = 0.13 to 0.69 give 0.0025 to
  0.0096 $\varepsilon|\theta_T|$ under $t$.
- §0.1 predicted all of this: for $T$, the rounding of $z = {\rm expm1}(u)$ is the same size as
  the floor removed wherever $z \gg 1$.
- In the literal reading, 27 windows exceed 0.1. Those numbers include the aliased windows of X1.

**Impact.** What `T_WKB` and the phase groups see is still about
$0.1$–$0.2\,\varepsilon|\theta_T|$ at $z \gg 1$. The fix does not touch this, and the prompt
did not ask it to (§0.1).

**Next step.** For the board: whether a phase evaluated in $u$ without an `expm1` round trip is
wanted. That would mean a leading term in $u$, a producer change, and G1 Q3's territory.

## X3. T4 fails on the same two rows as 05's L3, with the same ratios

- **`mid-G-r-3`:** $|t(10^{-7}) - t(10^{-8})|$ = 1.29 × `total_abserr`$_t(10^{-7})$.
- **`mid-G-3`:** 1.70 × `total_abserr`$_t(10^{-7})$.

05's L3 under $c$ gave 1.29 and 1.70 on these rows, and every one of the 31 rows' ratios agrees
with 05's to the digits printed. Neither row is capped under B1. On the 11 capped rows, T4 holds
with a largest ratio of 0.166.

**Bearing.** This is 05's W5, unchanged by $T$'s fix, and narrows nothing. It belongs with the
open `[05-declared-error-at-1e-7-misses-the-1e-8-change-on-uncapped-rows]`, if that was opened, and
with G2's Q5.

## X4. On these 39 rows, T's two errors cost nothing the integrator can see

This is descriptive. On every row:
- the region count, the maximum depth and the convergence flag of all 219 Levin group calls are
  identical under $c$ and $t$;
- `total_converged` is identical too.

The totals move by at most $4.7\times10^{-6}$ of `total_abserr`$_c$ (`low-G-1`), and by at most
$3.6\times10^{-12}$ relative (R4). On 13 rows the total is bit-identical. T3 holds with that
margin. This agrees with §0.1's statement that neither error was seen to drive the bisection.

## X5. §9's $p_z$ term never binds on the probe's windows

The term $\varepsilon(k\max|p_z| + 2\pi)$ is the largest of the three bounds in 0 of 70 windows.
`1e-3 ε|θ|` is the largest in 69, and `1e-13` in 1. $k\max|p_z|/|\theta_T|$ is at most
$9.8\times10^{-7}$, and every window at its level used lies within one table node's interval.

So the low-$z$ ends of the Levin sub-intervals sit at or very near background-table nodes. The
code log's observation 2, $\varepsilon k|p_z|$ up to $10^{-2}\varepsilon|\theta_T|$ midway between
nodes, is not exercised by §6's window placement. This is descriptive.

## X6. scipy `IntegrationWarning`s, as in 05 (05's W9)

`stderr.log` holds 15 warnings from production's numeric quadrature: 9 "maximum number of
subdivisions" and 6 "probably divergent". 05's run gave 15 too. No unit failed. Which rows raise
them was not investigated.

## X7. The kinds of non-convergence under $t$ equal 05's under $c$

`convergence_kinds.py` gives, under $t$:
- at `1e-7`: resolution 18, bound only 32, phase-limited 17. Restricted to the rows capped under
  B1: 12, 21, 10;
- at `1e-8`: 21, 24 and 27; on the capped rows, 17, 13 and 18.

These are the same tallies as `prompts/levin-phase-floor/measure/out/convergence_kinds.txt` under
05's $c$. The unconverged `Gqr` groups ("bound only", "phase-limited", 1 region at depth 0) are
untouched by $T$'s re-anchoring. This bears on G2.
