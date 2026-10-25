# Prompt 04 — remove the floor from seven depth-20 rows of A3 v2

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25, after 03 landed (`d5f5a2b`). 03 decided what the constructions are.
**Where it runs:** on the Mac Studio, orchestrated there, as 03 was. **It commits nothing there.**
Everything it writes is under `var/levin-phase-floor-04/` in the Mac Studio's checkout (README §4.1).
**At which tree:** the code runs at **`98e68d3`**, the tree 03 ran at, in a git worktree the laptop
makes before dispatch. `handover` B1 (`049c7c8`) changed `QuadSource` and `QuadSourceIntegral`, and
its strict coverage check refuses A3 v2's `QuadSource` rows, so no A3 v2 row can be re-run at `HEAD`
(`[12-realistic-large-x-reads-the-retired-clamp-records]`).
**Uses:** no store write, and no Ray (03's route (a)). The re-runs are one registered, detached run.
**Closes:** nothing. **Narrows:**
- `[00-consumer-anchoring-floor]` (`GkTk-remedial`);
- `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` (`handover`);
- `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]` (`handover`), through N4;
- this campaign's `[03-m3-slope-clause-is-badly-posed-at-round-off]`, by giving G1 the full re-run
  that M3's control stood in for.

**Recommended model:** **Opus**.

**Read first:**
1. [`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §2.6 item 3 and §2.7.
2. This campaign's [`README.md`](README.md) §0.2, §5 and §6.3 (Q2, Q3 and Q5), and the board's §2
   and §3.
3. [`docs/levin-phase-floor/VERIFICATION.md`](../../docs/levin-phase-floor/VERIFICATION.md) §3, and
   `docs/levin-phase-floor/03/issues.md`. I3 and I4 are the two workarounds this prompt inherits.
4. `ComputeTargets/primitive_phase.py`, all of it. `raw_theta` (`:276`), `theta_mod_2pi` (`:285`)
   and `build_phi_samples` (`:359`) are where the floor is made.
5. `ComputeTargets/cumulative_table.py`: the module docstring, `delta` (`:360-401`), `_locate`
   (`:336`), `_integral` (`:293`), and the node limbs `_hi_list` and `_lo_list` (`:172-173`).
   `BackgroundModel.TablePrimitive` (`ComputeTargets/BackgroundModel.py:638-670`) exposes the table
   as `.table`.
6. `ComputeTargets/phase_groups.py`: the module docstring's "Phase composition", and
   `_composed_theta_mod_2pi` (`:542`). A group's `theta_mod_2pi` is the signed sum of its factors'
   remainders, each from `PrimitivePhase.theta_mod_2pi`.
7. Where the two phases are built, **at `98e68d3`**:
   - $G$: `GkSourcePolicyData._build_phase` (`ComputeTargets/GkSourcePolicyData.py:145-195`),
     anchored at $z_{\rm response}$, `sign = -1`, leading primitive `functions.tau`;
   - $T$: `TkSourceFunctions.__init__` (`git show 98e68d3:ComputeTargets/TkSourceFunctions.py`,
     `:380-405`), anchored at its `z_init`, `sign = +1`, leading primitive `functions.cs_tau`.
     The file has changed since, so read it at `98e68d3`.
8. `LiouvilleGreen/WKBtools.py`, `WKB_mod_2pi` (`:15`): `fmod` against the double `TWO_PI`.
9. 03's scripts, `docs/levin-phase-floor/03/scripts/`: `rebuild.py`, `harness.py`,
   `run_instrument.py`, `launch.py` and `analyse.py`. This prompt reuses them.

**Gate:**
- 03 landed and recorded (`d5f5a2b` or later);
- the dispatch commit pushed;
- on the Mac Studio, the main checkout at the dispatch SHA with `git status --short` empty, and the
  worktree at `98e68d3` with `git status --short` empty;
- `runregistrykit list` shows nothing `running` or stale;
- the lock is free;
- `store show` gives A3 v2 live, fingerprint `0e32f24a…`;
- 03's run directory, `var/runs/levin-phase-floor-03-instrument-20261025T194950/`, holds its
  pickles.

---

## 0. The question

03 found a non-smooth jitter in every scored group's composed phase, at 0.22–0.34 of
$F = \varepsilon k\eta_R$, cancelling or not. It found a discrepancy that does not fall with depth,
and that falls to round-off when the jitter is fitted away on a depth-10 span. It did not re-run a
row with the jitter gone.

This prompt does. Two questions:
- **Does the cap go when the floor goes?** Findings §2.6 item 3: with the floor removed, the depth
  falls well below 20, and `total` moves by less than the row's old declared error.
- **Does the cheapest layer 1 do it?** Findings §2.7 layer 1(a): take $k\tau_R$ out of $G$'s phase.
  README §6.3 Q2 asks the user whether that, or more, is wanted.

It also answers two questions that need a jitter-free re-run:
- **N4** (02's, moved here): does each row's stored `total_abserr` cover the jitter's effect on
  `total`?
- **Q5**: what does `rtol = 1e-7` cost in accuracy against `1e-8` on a depth-20 row, once the floor
  no longer decides the depth?

## 1. Why the floor is where it is

Each oscillatory factor $i \in \{G, q, r\}$ is a `PrimitivePhase`:

$$\theta_i(z) = s_i\,k_i\,\Delta_i(z, a_i) + \phi_i(u), \qquad u = \log(1+z),$$

with $\Delta_i$ the leading primitive's interval accessor, $a_i$ its anchor and $\phi_i$ a spline of
an O(1) residual. For $G$: $s = -1$, $\Delta$ on `tau`, $a = z_{\rm response}$. For $T$: $s = +1$,
$\Delta$ on `cs_tau`, $a = z_{\rm init}$.

`theta_mod_2pi` forms $\theta_i$ as **one double**, then reduces it exactly with `fmod`. The double
is rounded to about $\varepsilon\,|\theta_i|$, and that rounding is not a smooth function of $z$.
Write $\Phi_i(z) = |\theta_i(z)|$ for the magnitude being rounded.

01's T5 puts the `G±r` sub-intervals at $z' \sim 10^9$–$10^{12}$ and the lone `G`'s above them. There,
$\Delta_G(z, z_{\rm response}) \approx \eta_R$, so $\varepsilon\Phi_G \approx F$. $T$'s phase,
anchored at its own `z_init` just above, is small by comparison. So H-anchor names $G$'s anchor.
The `Gqr` sub-intervals reach down towards $z_{\rm response}$, where $\Phi_G$ is small and $T$'s
$\Phi$ is not.

## 2. The arms

Every arm changes **one thing**: how `PrimitivePhase.theta_mod_2pi` turns $\theta_i$ into a
remainder. `raw_theta`, `theta_deriv`, the amplitudes, the partition, the driver and every
tolerance are production's. Each arm is installed by patching
`ComputeTargets.primitive_phase.PrimitivePhase.theta_mod_2pi` at the class attribute, in the
harness's own process, around each unit's `evaluate_QuadSource_integral` call only, and restoring it
in a `finally`. The clamp of `QuadSourceIntegral._ClampedPhase` therefore applies unchanged, since it
calls through to the method. A $G$ phase is told from a $T$ phase by its `sign`. At `98e68d3` the only
constructions of a `PrimitivePhase` are $G$'s (`GkSourcePolicyData.py:187`) and $T$'s (in
`TkSourceFunctions.__init__`). The dispatcher passes any other instance through unchanged, and counts
any it sees.

Write $\Delta^{\rm ext}(z_a, z_b)$ for `CumulativeTable.delta` re-formed in `mpmath` at no fewer than
40 digits. The node limbs are summed exactly, $(\text{hi}_b - \text{hi}_a) + (\text{lo}_b - \text{lo}_a)$,
and the two end partials are production's own floats, from `_integral`. Every reduction below
is against the double constant `TWO_PI` taken exactly, as `fmod` takes it, into production's
$(-\texttt{TWO\_PI}, 0]$ convention, and then rounded to a double once.

- **P — production.** The method as it stands, called through the patch in pass-through mode.
- **A — $k\tau_R$ out of $G$'s phase** (findings §2.7 layer 1(a)). $G$ is re-anchored at $z_{\rm top}$,
  the node of the `tau` table at or nearest above `max_x`, the top of its own $\phi$ samples:
  $$\rho^A_G(z) = \texttt{WKB\_mod\_2pi}\big(-k\,\Delta_\tau(z, z_{\rm top}) + \phi_G(u)\big)[1] + c_G,$$
  with $c_G$ the remainder of $-k\,\Delta^{\rm ext}_\tau(z_{\rm top}, z_{\rm response})$, a constant
  computed once per $G$ object. In real arithmetic this is $G$'s phase exactly, since $\Delta$ is
  additive. The variable part is now $\Phi^A_G(z) = |-k\Delta_\tau(z, z_{\rm top}) + \phi_G|$,
  small where $z' \gg z_{\rm response}$. The sum may lie in $(-4\pi, 0]$ and is not re-reduced; the
  composition does not re-reduce either. $T$'s remainders are production's. Production could build
  this in `GkSourcePolicyData` and `PrimitivePhase` alone, without touching `AdaptiveLevin/`.
- **X — every remainder exact.** For every factor, $G$ and $T$ alike:
  $$\rho^X_i(z) = \big(s_i\,k_i\,\Delta^{\rm ext}_i(z, a_i) + \phi_i(u)\big) \bmod \texttt{TWO\_PI}.$$
  What is left is the rounding of the end partials, $\Phi^X_i(z) = k_i\,|p_i(z)| + 2\pi$, with $p_i$
  the two partials' sum. That is far below $F$ except within a grid interval or so of an anchor.
  X is the limit that any phase layer could reach: per-region anchoring, or a composition before
  reduction done exactly. It is the reference for N4. It is **not the truth**: it keeps P's $\phi$
  and P's modulus, so $P - X$ is the jitter's effect alone.

**Not run as an arm: layer 1(b), the group composed from its leading terms in double before
reduction.** `[01-slow-group-leading-terms-do-not-cancel]` shows that the leading coefficients do
not cancel, because $T$'s primitive is `cs_tau`. At the `G±r` sub-intervals the leading terms that
(b) would compose are those A already makes small. Near $z_{\rm response}$, composing them in double
leaves $T$'s magnitude where it is. X bounds what any composition could do. This is the orchestrator's
choice C7 (README §6.2), and the user can reverse it.

**The predicted floor of a group.** For a group $g$ and an arm, $F^{\rm arm}_g$ is $\varepsilon$
times the largest value, over the group's sub-interval, of $\sum_{i \in g} \Phi^{\rm arm}_i(z)$.
Sample it at 1000 points equispaced in $\log(1+z')$, at the abscissae production would pass, so
after the clamp. $\varepsilon$ is the one 01 and 03 used for $F$.

## 3. The rows, the rungs and the units

The rows are 03's, unchanged:

| # | (shard, serial) | top group | $k$ | $q$ | $r$ | $z_{\rm response}$ | stored s | $F$ |
|---|---|---|---|---|---|---|---|---|
| R1 | (2, 37527) | `G+r` | 3.09e6 | 1.00e5 | 3.09e6 | 33.16 | 200 | 1.69e-6 |
| R2 | (0, 1797) | `G+r` | 3.05e7 | 3.09e6 | 3.05e7 | 302.6 | 220 | 4.60e-6 |
| R3 | (1, 37670) | `G+r` | 9.56e7 | 9.85e5 | 9.56e7 | 10.98 | 240 | 9.19e-5 |
| R4 | (2, 37720) | `G-r` | 3.09e6 | 1.00e5 | 3.09e6 | 14.47 | 355 | 2.60e-6 |
| R5 | (0, 37966) | `G-r` | 3.05e7 | 3.09e6 | 3.05e7 | 174.1 | 200 | 6.49e-6 |
| R6 | (2, 34728) | lone `G` | 3.09e6 | 3.09e6 | 3.09e6 | 25.16 | 134 | 1.96e-6 |
| R7 | (2, 34820) | `G+q+r` | 3.09e6 | 3.09e6 | 3.09e6 | 14.47 | 150 | 2.60e-6 |

**The scored groups are 03's 29**, as `docs/levin-phase-floor/03/out/instrument_03.txt` lists them.
Report every group of every row, scored or not.

**The units**, in this order, one checkpoint each, all at `atol = 1e-32`:
1. **P** at `rtol = 1e-7`, all seven rows. This is M0, and it sets each row's budget.
2. **X** at `rtol = 1e-7`, all seven rows.
3. **A** at `rtol = 1e-7`, all seven rows.
4. **X** at `rtol = 1e-8`, all seven rows.

**The budget.** Each X or A unit may run for 3 times that row's P unit's wall time. The run enforces
this itself, with `signal.setitimer` or equivalent. An over-budget unit is recorded `over budget`
with its elapsed time, and the run continues to the next unit. **An over-budget unit is a result,
not a failure of the run**: it is what an arm that does not remove the floor looks like. P units have
no budget. The worst case is about 25 minutes plus 3 × 75 minutes.

**What each unit keeps.** The row's result dict, and for each Levin call: the group label, parsed from
`notify_label` as 03 did; `value`, `abserr`, `converged`, `phase_limited`; the region count; the
depth histogram split by type (Levin or Clenshaw–Curtis); `evaluations`; and `elapsed`. **Do not
pickle P's region lists again**, since 03's pickles hold them. For X and A, keep the region lists if
they are small enough to be useful, and say what was kept.

## 4. Steps

Make `var/levin-phase-floor-04/{out,scripts,state}` in the main checkout. Save every command and its
output. Run commands with variable arguments through `bash`.

1. **Check the trees.** No git command that writes is run, in either tree:
   - the main checkout's `HEAD` is the dispatch SHA, and `git status --short` is empty;
   - the worktree's `HEAD` is `98e68d3`, and `git status --short` is empty;
   - record `pip freeze` once.
2. **Hash the store** into `state/store_files_before.txt`: the SHA-256, size and mtime of the
   primary, the four shards and the sidecar.
3. **Take 03's scripts.** Copy them from `docs/levin-phase-floor/03/scripts/` at the dispatch SHA
   into `scripts/`, and adapt them:
   - **code from the worktree:** every import of this repository's packages, and `main.py` for
     `load_main_py_functions`, resolves in the worktree. Run with `PYTHONPATH` set to the worktree
     and the main checkout's venv by its absolute path;
   - **data from the main checkout:** the store, `var/`, 03's run directory and the registry;
   - **provenance:** every output header records the SHA-256 of the script that wrote it, the
     worktree's `HEAD` (the code that ran) and the main checkout's `HEAD` (the records). 03's
     `git_head()` reads one tree, so it must be changed.

   Prove route (a) again, on one row, before anything else. It worked for 03 at this tree; if it
   does not now, stop.
4. **Write `scripts/phases.py`:** the three arms of §2 as one dispatcher patched at the class
   attribute, $\Delta^{\rm ext}$, $\Phi^{\rm arm}_i$ and $F^{\rm arm}_g$.
5. **The phase checks,** from the rebuilt inputs alone, Ray-free and with no Levin call, into
   `out/phases_04.txt`. E1 and E2 (§5) are gates.
   - **E1, the arms compute the same phase.** On each row, for each oscillatory factor of each
     Levin sub-interval, at 2000 abscissae equispaced in $\log(1+z')$ and clamped as production
     clamps:
     - $|{\rm float}(\Delta^{\rm ext}) - \texttt{delta}()| \le 2$ ulp of `delta()`;
     - $|{\rm wrap}(\rho^X - \rho^P)| \le 4\varepsilon\Phi^P_i(z) + 10^{-15}$;
     - $|{\rm wrap}(\rho^A - \rho^X)| \le 4\varepsilon\Phi^A_i(z) + 10^{-15}$, for $G$.

     Here wrap maps onto $(-\pi, \pi]$ with period `TWO_PI`.
   - **E2, the floor model.** On the depth-20 regions 03 sampled, at most 20 per scored group and
     read from 03's pickles, measure $\sigma_\theta$ of the group's composed `theta_mod_2pi` under P,
     A and X. Use 03's estimator unchanged: 65 equispaced points, `numpy.unwrap`, a degree-4 fit in
     the local coordinate, and the RMS residual. P must reproduce 03's medians to the digits 03
     printed.
   - **Descriptive, not scored:**
     - $F^P_g$, $F^A_g$ and $F^X_g$ for every group, beside 03's $F$ and its scored or unscored
       status;
     - **$\sigma_\phi$, the residual's own sampling noise.** For each factor, take $\phi_i$ at its
       stored samples as production builds it (`build_phi_samples`). Take it again rebuilt in
       `mpmath` from the stored `theta_div_2pi` and `theta_mod_2pi`, with the stored phase formed
       as `theta_div_2pi` × `TWO_PI` + `theta_mod_2pi` exactly, and with $\Delta^{\rm ext}$. Report
       each one's RMS residual about a local fit: degree 4, over 9 consecutive samples, in $u$.
       Every arm shares $\phi$. If production's $\sigma_\phi$ is near $F$, it is a smooth accuracy
       floor that X does not remove. Whether the rebuild removes it says whether the noise is the
       consumer's or is in the stored samples. That bears on Q3;
     - **the modulus.** For each factor, the largest $|\theta_i|/\texttt{TWO\_PI} \times (2\pi - \texttt{TWO\_PI})$
       over its sub-intervals, against $F$. Reducing against `TWO_PI` rather than $2\pi$ is a smooth
       phase error of about $0.18\,\varepsilon|\theta|$. P, A and X all share it.
6. **The predictions,** before the launch. Copy §5 unchanged into `log.md`'s "Predictions and
   verdicts". Save the same text as `state/predictions.md`, together with the list of groups that
   A1 covers: the scored groups with $F^A_g \le 0.1 \times 10^{-7}$, from step 5.
7. **The run.** Registered: campaign `levin-phase-floor`, prompt `04`, slug `remove`. Launch it
   detached from the worktree, so that the registry records `98e68d3`; it writes into the main
   checkout's `var/runs/`. The units are §3's. A row that fails M0 stops the run. Check once that it
   has started, and **end the turn**.
8. **The analysis,** in `scripts/analyse.py`, from the checkpoints alone, into `out/remove_04.txt`:
   - per row, arm and rung: `total`, `total_abserr`, `total_converged`, `total_phase_limited`,
     regions, maximum depth, evaluations and elapsed time;
   - per group: regions, the depth histogram by type, the count at depth 18 or more, `value` and
     `abserr`, beside $F^{\rm arm}_g$;
   - §5's scores, each with its numbers.
9. **Hash the store again** into `state/store_files_after.txt`; it must equal step 2. Run
   `store fingerprint` without `--write`.
10. **Write the records.**
    - `log.md`, in README §5.1's form;
    - `verification.md`: the text of `docs/levin-phase-floor/VERIFICATION.md` **§4 "The floor
      removed"**, with its links relative to `docs/levin-phase-floor/` in the form `04/out/…`;
    - `issues.md`.

## 5. Predictions, fixed before the run (README §5 rule 3)

Copy these into the log unchanged, and score them there. The gates come first. A gate that fails is
a stop, not a verdict.

**Gates:**
- **E1 — the arms compute the same phase.** Every bound of §4 step 5 holds at every abscissa.
- **E2 — the floor model is right.** In every scored group, under each of P, A and X, the median
  $\sigma_\theta$ over 03's sampled regions is at most $\varepsilon \sum_i \Phi^{\rm arm}_i$, taken
  as a median at the same regions' midpoints. Under X it is also at most $10^{-2} F$.
- **M0 — P is production.** On all seven rows, the P unit equals the stored row **bit for bit**, by
  `==` and by IEEE-754 bit pattern. The fields are 03's: `total`, `total_abserr`,
  `total_converged`, `WKB_Levin`, `numeric_quad`, `WKB_Levin_data.max_depth` and `num_regions`, and
  every field of every group and sub-interval record of `metadata["WKB_Levin"]["subintervals"]`
  except `elapsed`. `analytic_rad` is compared and reported, but not gated.

**Scored:**
- **R1 — remove the floor and the cap goes** (findings §2.6 item 3).
  - **The rule:** under X at `rtol = 1e-7`, every one of the row's scored groups has fewer than 10
    regions at depth 18 or more, and the row's region count is at most a tenth of its P unit's.
  - **Holds** if true on at least six of the seven rows. An over-budget X unit counts as false on
    its row.
  - If R1 fails, findings §2.6's own clause fires: *"item 3 leaves the depth at 20"*, so the
    hypothesis is wrong.
- **A1 — layer 1(a) is enough where $G$'s anchor is the floor** (H-anchor's prediction).
  - **The rule:** under A at `rtol = 1e-7`, every scored group with $F^A_g \le 0.1 \times 10^{-7}$
    has fewer than 10 regions at depth 18 or more.
  - **Holds** if true in every such group. **Undecided** if there is no such group. An over-budget
    A unit counts as false for each such group on its row.
  - Report the other scored groups too, unscored: their depth beside X's, and their $F^A_g$.
- **N4 — the stored error is honest about the jitter.**
  - **The rule:** $|{\rm total}_P - {\rm total}_{\rm ref}| \le$ `total_abserr`$_P$, with
    ${\rm total}_{\rm ref}$ from X at `rtol = 1e-8`. If that unit was over budget, use X at `1e-7`
    and say so. If both were, the row is undecided.
  - **Holds** if true on all seven rows. Otherwise list each failing row with the ratio of the two
    sides. Each such row is one whose stored error understates the jitter's effect.
- **R2 — with the floor gone, the declared error is honest.**
  - **The rule:** $|{\rm total}_X(10^{-7}) - {\rm total}_X(10^{-8})| \le$ `total_abserr`$_X(10^{-7})$.
  - **Holds** if true on all seven rows. A row with either unit over budget is undecided.
  - **Q5's number,** reported on every row whichever way R2 goes:
    $|{\rm total}_X(10^{-7}) - {\rm total}_X(10^{-8})| / |{\rm total}_X(10^{-8})|$. Set it beside
    the 2026-09 sweep's 1.9e-6 (mid band), 2.84e-6 (low) and 8.3e-7 (seam).

**Not scored, and why.** H-cancel is not re-scored. 03 showed that its condition holds in the
non-cancelling groups as well, so it fails as the whole account. Both hypotheses predict R1, and
neither A nor X can tell them apart. Nothing here predicts which groups still reach depth 20 under A
when $F^A_g$ is large. In P, 16 of 45 groups were left unscored although their rows' $F$ was the
same, so a large floor is not enough to send a group to the cap. Those groups are descriptive.

## 6. Acceptance

1. E1, E2 and M0 hold.
2. The store is byte-identical and its fingerprint is `0e32f24a…`. No lock was taken and Ray was
   never initialised.
3. R1, A1, N4 and R2 are each scored holds, fails or undecided, against §5 as written, with their
   numbers. Q5's number is reported for every row.
4. The records are under `var/levin-phase-floor-04/`: `out/`, `scripts/`, `state/`, `log.md`,
   `verification.md` and `issues.md`. The checkpoints stay in the run directory.

The orchestrator copies the records to `docs/levin-phase-floor/04/`, without the checkpoints or any
CSV. It appends §4 to `VERIFICATION.md`, writes `logs/04-remove-the-floor.md`, records the verdicts
on the board and the narrowings on the boards that hold them, and commits. Gate **G1** then opens.

## 7. Stop conditions — stop and ask

- Any Gate item not as stated.
- Route (a) does not work at `98e68d3`.
- An arm cannot be built by patching `PrimitivePhase.theta_mod_2pi` alone.
- E1, E2 or M0 fails.
- Any store file changes.
- The run would need Ray, a lock, or a change to production code, a test file or `AdaptiveLevin/`.
