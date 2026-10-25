# levin-phase-floor 01 — found and not acted on

These are for the orchestrator to place on the boards. Nothing here was changed.

## I1. The "cancelling" group's leading phase does not cancel: G's leading rate is k/H, T's is k c_s/H

**Proposed tag:** `[01-slow-group-leading-terms-do-not-cancel]`.

**What the code does.** It confirms the sign convention prompt §1 states: $\theta_G$ decreases with
$z'$, and $\theta_q$ and $\theta_r$ increase. But the two leading primitives differ.
- **$G$ uses `tau`, at rate $1/H$.** `GK_PHASE_SIGN = -1` is at `ComputeTargets/GkSourcePolicyData.py:142`
  and is used at `:184` and `:193`. `PrimitivePhase`'s default rate is $1/H$
  (`ComputeTargets/primitive_phase.py:291-310`), and `WKB_Gk.py:29-42` gives the leading frequency
  as $(k/H)^2$.
- **$T$ uses the sound horizon `cs_tau`, at rate $c_s/H$.** `TK_PHASE_SIGN = +1` is at
  `ComputeTargets/TkSourceFunctions.py:184` and is used at `:393` and `:403`.
  `_sound_horizon_rate` (`:205-236`) gives the rate. `WKB_Tk.py:35-51` gives the leading frequency
  as $c_s^2 (k/H)^2$. `phase_groups._OscillatoryT.theta_deriv` (`phase_groups.py:395-397`)
  returns $\omega(1+z)$ with $\omega = \sqrt{\texttt{Tk\_omegaEff\_sq}}$.

**What follows.** On `r == k` the leading $d\Psi/dz$ of the "slow" group `G+r` is
$-k(1-c_s)/H$, not zero. That of `G−r` is $-k(1+c_s)/H$.
- **The radiation era.** T5 puts the top group's sub-interval at $z' \sim 10^{9}$–$10^{12}$. With
  $c_s^2 \approx 1/3$ there, `G+r` oscillates at about $0.42\,k/H$, against $1.58\,k/H$ for
  `G−r`. It is slower by a factor of about 3.7, but it is not stationary.
- **Two documents claim more than this.** Findings §2.2 says the $\tau_s$ terms *"cancel
  identically"*. README §0.2 says *"the $\tau_s$ terms of $G$ and $T_r$ cancel"*.
- **Layer 1 would be affected too.** If findings §2.7 layer 1 composes $(s_G k + s_r r)\tau_s$, then
  on `r == k` the slow group's $\tau_s$ coefficient is exactly zero only if both factors share one
  primitive. They do not.

**Impact.** This does not change prompt 01's definitions or its stop condition, and every verdict
stands as scored.
- **It bears on H-cancel's mechanism.** "A failed cancellation leaves a slow group" is a partial
  cancellation by a factor of about 3.7, not a cancellation to zero. Each group's own jitter is
  still about $\varepsilon k\eta$.
- **It bears on the remedy's shape**, at G1 Q2(b).

**Next step.** Check this numerically in 03: `theta_deriv` of `G+r` and `G−r` on one candidate row.
Then the orchestrator decides whether findings §2.2 and §2.7 need a correction note.

## I2. Non-finite `total_abserr` reaches 558 rows of A3 v2, not 5

**Narrows** `[04-roundoff-floor-can-be-infinite]` (`levin-refactor`, in OPEN_ISSUES §1.3).

- **The rows.** 558 of 9280 rows store `total_abserr` = +inf: SQLite `real` `Inf`, Python `inf`,
  JSON `Infinity`.
  - Only **5** are at depth 20; those are the five of findings §2.5. The other **553** are at
    depth 1–19.
  - By band: 392 seam, 44 mid and 122 low. All 558 are `total_converged = 0`.
- **The groups.** 803 group records carry `abserr = inf`. Each has a finite non-zero `value` and is
  flagged unconverged; 631 of them have `regions = 1`.
  - 392 are `qr`/`q−r` records in the seam band, where on $q = r$ the $q-r$ phase is identically
    zero.
  - The other 411 are `Gqr` records, 708 of the 803 being on $q = r$ rows.
- **What the finding confirms.** The +inf rows reach stored data. They are not confined to
  expensive rows: the cheap ones far outnumber them.
- **Impact.** Any consumer that sums or ratios `total_abserr` must handle `inf`. A
  `total_abserr/|total|` gate, as findings §2.5 proposes, would reject these 558 rows. Whether that
  is right depends on whether the `inf` is a real bound or an artefact of the round-off floor on a
  zero-phase group. That was not examined.
- **Next step.** For the issue's owner, read one seam `q−r` record and one `Gqr` record in
  process, and see where the `inf` comes from (`_roundoff_floor` with $\theta' = 0$?). That is not
  this campaign's scope (README §1, out of scope).

## I3. README §0.2's low-band figure does not match the reconnaissance output

README §0.2 says *"Most of the cost is in the low band: 229.7 of the 282.9 CPU-hours."*
`measure/out/recon_groups.txt`'s own band table sums to **247.88** CPU-hours in the low band, and
01 reproduces that table exactly.
- **Impact:** a prose figure only. No verdict uses it.
- **Next step:** a correction note on the README when the orchestrator next edits it.

## I4. `depth20_floor.py`'s $\eta(z)$ is wrong above $z \approx 10^{12}$ and negative above $10^{15}$

- **The cause.** It integrates from $z$ to $10^{15}$, so above that limit $\eta$ is negative. Against
  the stored `eta_response`, the ratio departs from 1 from $z \approx 1.2\times10^{13}$
  (`out/eta_check_01.txt`).
- **Impact here: none.** No group record lies at those $z_{\rm response}$, and the findings script
  excluded the seam band.
- **Impact elsewhere.** Anyone reusing that $\eta$ for seam rows would get a wrong $F$.
- **Next step:** a docstring note, or use `eta_response`, which the row stores. Neither is in this
  prompt's scope.

## I5. P3 discriminates weakly

Both classes have 99.9 % of their depth-20 cost at $F > 10^{-7}$. That is a property of which rows
reach depth 20, and not of the groups: $F$ is per row, shared by every group in it. Any hypothesis
in which depth-20 rows are those with $F > {\rm rtol}$ predicts the same.
- **The rule is not badly posed:** it tests what it says. But its "holds" adds little to P2.
- **Recorded** so that G1 weighs it.
