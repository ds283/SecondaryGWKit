# Prompt 02 — phase-noise injection on the realistic fixture

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25, with the plan.
**Where it runs:** on the laptop. No Ray, no store, no lock. The sweep is a registered run,
launched detached; the prompt ends its first turn after one check that it has started, and is
resumed for the report (§2 step 6).
**Lands:** one commit, after the report. It holds:
- the script `docs/levin-phase-floor/noise_injection.py`;
- its records under `docs/levin-phase-floor/02/`;
- `docs/levin-phase-floor/VERIFICATION.md` §2;
- the log;
- the board and `docs/OPEN_ISSUES.md`.

**Closes:** nothing. **Narrows:** `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` and
`[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` (both on the `handover`
board).
**Recommended model:** **Opus**.

**Read first:**
1. [`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §2, and §2.6 item 2
   especially.
2. This campaign's [`README.md`](README.md) §0.2 and §5. If 01 has landed by the time you run,
   read its log as well. If it has not, do not wait for it.
3. [`docs/handover/REALISTIC-LARGE-X.md`](../../docs/handover/REALISTIC-LARGE-X.md) Tables 7–8,
   and [`docs/handover/realistic_large_x.py`](../../docs/handover/realistic_large_x.py): how a cell
   is built (`seed_fixtures`, `scaled_shape`, `Case`, `run_cell`).
   - **Do not edit that file.** Its checkpoint is gated on its own SHA-256, so one changed
     character discards sixty cells (`prompts/run-registry` board,
     `[02-realistic-large-x-is-outside-the-registry]`).
   - Import from it, or copy the few lines you need and say so.
4. `ComputeTargets/tests/test_quadsource_integral.py`: `Case`, `SHAPES` (`:321-327`) and
   `Case.run` (`:511`).
5. `ComputeTargets/phase_groups.py`:
   - the module docstring's "Phase composition";
   - `PhaseGroup`, a frozen dataclass, so use `dataclasses.replace`;
   - `build_phase_groups`.
6. `ComputeTargets/QuadSourceIntegral.py:15` and `:1045`. `phase_group_Levin_integral` calls
   `build_phase_groups` through the module attribute it imported, which is where the noise goes in.
7. [`RunDrivers/quadsource_atol_sweep.py`](../../RunDrivers/quadsource_atol_sweep.py), for how a
   script registers a run, heartbeats and checkpoints. Read `CLAUDE.md`, "Long-running jobs".

**Gate:**
- `git status --short` is empty;
- `runregistrykit list` shows nothing `running` or stale.

The rehearsal lock is not needed: this prompt starts no Ray process.

---

## 0. The question

Is a non-smooth error in a group's phase **enough** to drive the Levin driver to its depth cap at
production's tolerance? Does it do so at an amplitude near `rtol`? And does it do so in any group,
sum, difference or alone, or only in some?

Production's phases are composed from separately reduced remainders. They carry a deterministic,
non-smooth error of size $\varepsilon\times$(the phase's magnitude): findings §2.1, and
`[00-consumer-anchoring-floor]`. On A2's fixtures that error is small, about $\varepsilon x$, which
is roughly 2e-12 at $x_{\rm resp}=10^4$, far below `rtol = 1e-7`. So the fixtures give a clean
baseline, and the noise can be added on top at a controlled amplitude.

## 1. The construction

**Cases.** Each case is built as `realistic_large_x.run_cell` builds it, with $b = 0$ and the seam
**closed**, so that no clamp enters. There are eight:
- **flavour:** `exact` or `realistic`;
- **shape:** `together` ($k=1.1\times10^4$, $q=10^4$, $r=1.2\times10^4$), or `q-smooth`
  ($k=10^4$, $q=10^2$, $r=1.1\times10^4$). `q-smooth` is the fixture nearest production's squeezed
  `r ≈ k ≫ q`;
- **$x_{\rm resp}$:** $980$ or $10^4$.

`Case.run(atol=1e-45, rtol=DEFAULT_QUADRATURE_RTOL)`:
- **`rtol`** is production's `1e-7`. Import the constant; do not write it as a literal.
- **`atol = 1e-45`** is A2's `REF_ATOL`. It keeps the length-share `local_atol` branch dead, as it
  is in production. That death is half of the mechanism, so the injection must not reopen it.

**The noise.** Wrap `ComputeTargets.QuadSourceIntegral.build_phase_groups`. For each returned group
whose label is in the arm's set, replace `theta_mod_2pi` with $\theta(x) + \delta\,\xi(x)$. Leave
`theta` and `theta_deriv` alone: in production they are the smooth inputs, and the remainder is
where the floor lives. Two models of $\xi$:
- **white:** $\xi(x) = 2u - 1$, with $u$ the first eight bytes of
  `blake2b(label.encode() + struct.pack("<d", x))` divided by $2^{64}$. It is deterministic per
  abscissa, as rounding is, and independent across abscissae and groups.
- **staircase:** $\xi(x) = \big(\Delta\,\mathrm{round}(\theta_{\rm raw}(x)/\Delta) -
  \theta_{\rm raw}(x)\big)/\delta$, with $\Delta = 2\delta$ and $\theta_{\rm raw}$ the group's own
  `theta`. This is quantisation of the phase to a grid: a sawtooth of amplitude $\delta$, closer to
  what rounding a large number does.

**The ladder.** $\delta \in \{0, 10^{-10}, 10^{-9}, 10^{-8}, 3\times10^{-8}, 10^{-7},
3\times10^{-7}, 10^{-6}, 10^{-5}, 10^{-4}\}$.

**The arms.**
- **A:** white noise on every group, in all eight cases, over the whole ladder. 80 cells.
- **B:** one group label at a time, in the case `realistic` / `q-smooth` / $10^4$.
  - Every label that case produces, each labelled slow or fast from its `theta_deriv`, checked
    numerically at a few abscissae.
  - At $\delta \in \{10^{-9}, 10^{-7}, 10^{-5}\}$, with both noise models.
- **C:** staircase noise on every group, in all eight cases, at $\delta \in \{10^{-9}, 10^{-7},
  10^{-5}\}$. 24 cells.

**Per cell, record:**
- `total`, `total_abserr`, `total_converged` and `total_phase_limited`;
- `WKB_Levin_data`'s `max_depth` and `num_regions`;
- every group record of `metadata["WKB_Levin"]["subintervals"]`;
- the wall time;
- $|{\rm total}(\delta) - {\rm total}(0)|/|{\rm total}(0)|$ against the same case's $\delta = 0$
  cell.

Checkpoint each cell as one JSON line in the run directory, keyed on the script's SHA-256. A record
under another hash is recomputed, never re-stamped.

## 2. Steps

1. **Write the script** `docs/levin-phase-floor/noise_injection.py`. It has three entry points:
   - `--probe` runs §3's timing probe;
   - `--run` runs the sweep under the run registry;
   - `--report` writes the tables from the checkpoint alone.

   It writes a provenance header: its own SHA-256, `git HEAD` and dirty state, Python, numpy and
   scipy versions. `black` it.
2. **Check that the wrapper works:**
   - an injected $\delta = 10^{-4}$ cell changes `total` from the $\delta = 0$ cell;
   - a $\delta = 0$ cell through the wrapper is **bit-identical** to the same case run without it.

   Record both.
3. **The timing probe.** Time these four cells:
   - `exact` / `together` / 980 at $\delta = 0$;
   - the same at $10^{-5}$;
   - `realistic` / `q-smooth` / $10^4$ at $\delta = 0$;
   - the same at $10^{-5}$.

   Project the sweep's total wall time from them. **If the projection exceeds 24 hours, stop and
   ask**, with a proposal for which cells to drop.
4. **Launch.** Register the sweep (`campaign levin-phase-floor`, `prompt 02`, slug `noise`) and
   launch it detached with `nohup`, its stdout and stderr in the run directory. Check **once** that
   it started and has written its first cell, and **end the turn** (`CLAUDE.md` rule 4).
5. **When it is done,** confirm `runregistrykit list` shows it `done`.
6. **Report.** Run `--report` into `docs/levin-phase-floor/02/out/noise_02.txt`, and copy the
   checkpoint and the run's logs to `docs/levin-phase-floor/02/state/`. Score §4 in the log. Write
   `VERIFICATION.md` §2 "Phase-noise injection". `docs/levin-phase-floor/VERIFICATION.md` is created
   by 01; if 01 has not landed, create it with a §1 placeholder that says so.
7. **Commit**: one commit (README §4.1's contents, on the laptop).

## 3. Acceptance

1. Step 2's two checks hold.
2. Every cell of arms A–C ran, or the log names each one missing and says why.
3. The four predictions of §4 are scored holds, fails or undecided, against the rules as written,
   with the numbers.
4. `git status` after the commit is clean. No file outside `docs/levin-phase-floor/`,
   `prompts/levin-phase-floor/`, the `handover` board and `docs/OPEN_ISSUES.md` changed.

## 4. Predictions, fixed before the sweep runs (README §5 rule 3)

Write these into the log's "Predictions and verdicts" section unchanged, and score them there.
**$\delta^*$** for a case is the smallest rung at which `max_depth` reaches
`DEFAULT_LEVIN_MAX_DEPTH`.

- **N1 — noise alone reaches the cap near `rtol`** (findings §2.6 item 2).
  - In arm A, for every case, the $\delta = 0$ cell stays below the cap.
  - $\delta^*$ exists and is at most $10\,\texttt{rtol} = 10^{-6}$.
  - From $\delta^*$ up, `num_regions` is non-decreasing in $\delta$, within 10 %.
  - **Holds** if all three are true in all eight cases. **Fails** if any is false in any
    `realistic` case.
- **N2 — the exact flavour's flatness is a property of its phases.** Two parts, each scored on its
  own:
  - at $\delta = 0$, each shape's `exact` `num_regions` at $x_{\rm resp} = 10^4$ is within a factor
    of 2 of its value at $980$, which is A2's flatness;
  - at $\delta = 10^{-6}$, the same ratio is at least 3.

  If both hold, noise in the phase is sufficient to produce the cost growth
  `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` records, and that issue is narrowed to
  say so. It is not closed: sufficiency is not identity.
- **N3 — any group is vulnerable.** In arm B, with white noise at $\delta = 10^{-5}$, **every** label
  reaches the cap when it alone is noised: slow and fast, sum and difference.
  - H-anchor needs this to hold for the fast groups. **It fails H-anchor**, as the account of A3
    v2's fast-group cost, if a fast group stays below the cap at $10^{-5}$.
  - The injection tests whether a group can be driven to the cap, not where A3 v2's noise comes
    from. So N3 cannot confirm either hypothesis. It can only fail H-anchor.
- **N4 — the declared error stays honest under noise.** In every cell of arms A and C with
  $\delta > 0$, $|{\rm total}(\delta) - {\rm total}(0)| \le$ `total_abserr`$(\delta)$.
  - **Holds** if true in every cell.
  - Otherwise report the cells where it fails and their ratio. Each such cell is a case where the
    stored error would understate the true one, which is the honesty layer 2 exists to restore.

Also report, as descriptive and unscored:
- white against staircase at each rung;
- the share of each cell's `elapsed` by group;
- `total_converged` and `total_phase_limited` against $\delta$.

## 5. Stop conditions — stop and ask

- Step 2's bit-identity check fails: the wrapper changes something at $\delta = 0$.
- The probe projects more than 24 hours.
- A `δ = 0` cell reaches the cap. The fixture would then already be past its floor, and the
  baseline is not clean.
- Any edit is needed outside `docs/levin-phase-floor/`, or to `realistic_large_x.py`, a test file
  or production code.
