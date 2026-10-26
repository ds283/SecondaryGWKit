# levin-phase-floor 05a §6 — the predictions, fixed before the run

Prompt 05a §6's predictions, quoted word for word, **with §9's T1 in place of §6's T1**, as §9's
"In the records" says. Taken from `prompts/levin-phase-floor/05a-reanchor-t-phase.md` at `149c393`
(`git show 149c3938:… | sed -n …`): §6's predictions are lines 297–328 (the same bytes are
`state/prompt_section6_predictions.raw`); §9's T1 is lines 401–410
(`state/prompt_section9_T1.raw`); §9's amended probe window, on which that T1 is scored, is lines
391–399 (`state/prompt_section9_window.raw`). Written after gate T0 held (`out/T0.txt`) and before
the run is launched.

What is spliced, and nothing else: §6's line 297 (the heading line); then §9's T1 (lines 401–410)
**in place of §6's T1** (lines 298–305), which is not scored; then §6's T1 sub-bullet on the
$u$-sampled $\sigma_\theta$ (lines 303–305), which §9 keeps ("The $u$-sampled clause is
unchanged"); then §6's T2, T3, T4 and its descriptive items (lines 306–328). Each quoted block is
byte-identical to the prompt's lines. The rules below the first rule are the prompt's.

---

**Predictions, fixed before the run** (README §5 rule 3). Copy them into the log unchanged.
**T1, amended.** It replaces §6's T1. The $u$-sampled clause is unchanged.
- **The rule:** in every resolved probe window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad,
  sampled in $z$: $\sigma_\theta \ge 0.1\,\varepsilon|\theta_T|$ under $c$, and
  $\sigma_\theta \le \max\big(10^{-3}\varepsilon|\theta_T|,\ \varepsilon(k\max|p_z| + 2\pi),\ 10^{-13}\ {\rm rad}\big)$
  under $t$.
- **Holds** if true in every such window. **Fails** if false in any. **Undecided** if none is
  false but some window with $\varepsilon|\theta_T| \ge 10^{-11}$ rad is unresolved.
- **Also report, unscored,** in how many windows the term $\varepsilon(k\max|p_z| + 2\pi)$ is the
  largest of the three, and $\sigma_\theta/(\varepsilon|\theta_T|)$ under $t$ in those windows.


*§9's T1 replaces §6's T1. §6's T1 clause on the $u$-sampled $\sigma_\theta$, which §9 keeps unchanged (§6 lines 303–305):*

  - **The $u$-sampled $\sigma_\theta$ under $t$** is not scored. Wherever it exceeds
    $0.1\,\varepsilon|\theta_T|$, report it against $\varepsilon\,\tfrac{z}{1+z}|d\theta_T/du|$, and
    propose an issue for the abscissa term in the log's "Observations not acted on".

*§6's T2, T3, T4 and descriptive items (lines 306–328):*

- **T2 — the cap does not return.**
  - **The rule:** on every row (all 39), under $t$ at `1e-7`, no group has a region at depth ≥ 18.
  - **Holds** if true on every row; an over-budget unit counts as false.
- **T3 — $T$'s errors were within the declared error.**
  - **The rule:** on every row (all 39), $|{\rm total}_t - {\rm total}_c| \le$ `total_abserr`$_c$,
    at `1e-7`.
  - A failure measures what $T$'s two errors cost a total beyond what it declared. That is a
    finding about the rows before this prompt, not a fault in it.
- **T4 — the declared error is honest after the change.**
  - **The rule:** on every one of the 31 depth-20 rows with both $t$ units in budget,
    $|{\rm total}_t(10^{-7}) - {\rm total}_t(10^{-8})| \le$ `total_abserr`$_t(10^{-7})$.
  - 05's L3, the same rule under $c$, held on 29 of 31.

**Descriptive, not scored:**
- $|{\rm total}_t - {\rm total}_c|/|{\rm total}_c|$ and $/$`total_abserr`$_c$ on every row;
- regions and maximum depth per group, $c$ against $t$, with the `Gqr` groups and the low-$z$ end
  of each sub-interval shown apart;
- `total_converged` under $c$ and $t$;
- **the kinds of non-convergence** under $t$ at both tolerances, by the orchestrator's
  `prompts/levin-phase-floor/measure/convergence_kinds.py`, copied unchanged, recorded by SHA-256,
  and run as `convergence_kinds.py <this run> --tree t --capped-run <05's run>`. It reads the
  `checkpoint.jsonl` and `unit_*.pkl` layout of 05's run, which the harness keeps. The same
  measurement under 05's $c$ is `prompts/levin-phase-floor/measure/out/convergence_kinds.txt`.

---

*§9's amended probe window (lines 391–399), on which §9's T1 is scored:*

  more under either, the width is halved and the window sampled again;
- this repeats until both trees' largest step is below $\pi/2$. The width used and that step are
  recorded per window;
- if the 65 samples would stop being distinct doubles first, the window is **unresolved**. It is
  reported and not scored;
- the $u$-sampled $\sigma_\theta$ uses the same width, in $\log(1+z)$.

Record $\max|p_z|$ over the window's 65 samples beside it: under $t$, from $T$'s `cs_tau.end_partial`
at each sample.

---

## Readings fixed before the run

These say how each rule is read where the text leaves a choice. They are fixed here, before any
unit of the run exists, and change no rule.
- **"The largest step of the unwrapped samples"** (§9's halving) is
  `max |numpy.diff(numpy.unwrap(theta))|` over the 65 samples **in $z$** (the samples T1 scores),
  exactly as §3 test 3's guard computes it (`test_t_phase_reanchor._sigma_theta`). The step the
  phase would take without folding, $|d\theta_T/dz|\,(b-a)/64$, is recorded beside it at every
  level and is not used by the rule. T0's own windows show that the two can differ: a window can
  pass the literal rule while its samples step by many cycles (out/T0.txt's workers' ledgers). The
  rule is scored as written; a reading by the true step is reported apart, unscored.
- **The window** at a sub-interval's low-$z$ end $z_{\min}$ is $[z_{\min},\ z_{\min} + z_{\min} w]$,
  inside the sub-interval, at relative width $w = 10^{-6}/2^n$, sampled at
  `numpy.linspace(a, b, 65)`; in $u$ at `numpy.linspace(log1p(a), log1p(b), 65)`. The level used is
  the first $n$ at which both trees' largest step is below $\pi/2$. **Unresolved:** the 65 $z$
  samples stop being distinct doubles (`numpy.diff > 0` fails) before such a level.
- $\varepsilon = 2^{-52}$. $|\theta_T|$ is $|$`raw_theta`$(z_{\min})|$ under $t$ (bit-identical
  under $c$, §0.4); $k$ is the factor's own wavenumber; $\max|p_z|$ is over the 65 $z$ samples
  of the level used, from the $t$ factor's `cs_tau.end_partial`.
- **"Depth-20 row":** T2 and T3 are over all 39 rows, as written; T4 over the 31 rows A3 v2
  stores at depth 20, as written. The 11 rows capped under B1's code (05's $p$ at depth 20, read
  from 05's run) are reported apart, unscored, as 05 did.
- **Over-budget units:** T2 counts one as false, as written. T4 excludes a row whose either $t$
  unit is over budget, as written. T3 does not say: a row whose $t$@`1e-7` unit is over budget has
  no total, so T3 cannot be evaluated there; T3 is then **undecided** if no row is false, and
  **fails** if any is.
- The inequalities of T3 and T4 are evaluated exactly on the doubles (`fractions.Fraction`), as
  05's analysis did; an infinite `total_abserr` makes the inequality true.
