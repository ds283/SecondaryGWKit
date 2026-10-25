# Prompt 01 — the phase-group census of A3 v2

**Campaign:** [`README.md`](README.md) · **Board:** [`IMPLEMENTATION_STATE.md`](IMPLEMENTATION_STATE.md)
**Written:** 2026-10-25, with the plan.
**Where it runs:** on the Mac Studio, in a Claude Code session there, as `handover` 10 did. **It
commits nothing there.** Everything it writes is under `var/levin-phase-floor-01/`, and the laptop
orchestrator commits it (README §4.1).
**Uses:** no Ray, no lock, no store write. SQLite `mode=ro` on A3 v2's primary and four shards.
**Closes:** nothing. **Narrows:**
- `[a3-baseline-quadrature-tolerance-is-unreachable-on-squeezed-triangles]` (`handover` board);
- `[a3-baseline-unconverged-rows-are-stored-and-cannot-be-removed]` (`handover`);
- `[00-consumer-anchoring-floor]` (`GkTk-remedial`);
- `[04-roundoff-floor-can-be-infinite]` (`levin-refactor`), if the +inf rows are confirmed.

**Recommended model:** **Opus**. The census itself is mechanical. Scoring two hypotheses against
it honestly is not.

**Read first:**
1. [`docs/handover/A3-V2-FINDINGS.md`](../../docs/handover/A3-V2-FINDINGS.md) §1–§2, in full.
2. This campaign's [`README.md`](README.md) §0.2 (the reconnaissance, and the two hypotheses) and
   §5.
3. `ComputeTargets/QuadSourceIntegral.py`: `evaluate_QuadSource_integral` (`:734-962`) and
   `phase_group_Levin_integral` (`:1010-1096`). This is what a row's `metadata` holds, and how
   `total_abserr` and `total_converged` are formed from the groups.
4. `ComputeTargets/phase_groups.py`: `build_phase_groups` and `_composed_theta_mod_2pi`
   (`:444-550`). This is how a group's phase is the signed sum of its factors' remainders.
5. [`docs/handover/a3-v2/after/scripts/depth_census.py`](../../docs/handover/a3-v2/after/scripts/depth_census.py),
   the read-only pattern to follow: shard 0's maps, `mode=ro`.
   [`docs/handover/a3-v2/findings/depth20_floor.py`](../../docs/handover/a3-v2/findings/depth20_floor.py)
   has the $\eta(z)$ and $\varepsilon$ to reuse. Import them or copy them, and say which.
6. [`measure/recon_groups.py`](measure/recon_groups.py), the orchestrator's first read, and its
   output [`measure/out/recon_groups.txt`](measure/out/recon_groups.txt). This prompt supersedes it.

**Gate** (check and save to `out/gate_01.txt`):
- the dispatch commit is pushed, and is this prompt's commit or later;
- on the Mac Studio, `runregistrykit list` shows nothing `running` or stale;
- the checkout's `git status --short` is empty;
- `store show` gives A3 v2 live, fingerprint `0e32f24a…`.

---

## 0. The question

Which phase group carries the depth-20 cost of A3 v2? And does each group's declared error track
the floor $\varepsilon k\eta_R$ the same way whatever its signs, or only where a cancelling pair
exists?

Every row stores one record per group per Levin sub-interval: `label`, `value`, `abserr`,
`converged`, `phase_limited`, `regions`, `simple_regions`, `evaluations` and `elapsed`. The values
are scaled by $(1+z_{\rm response})$, as `total` is. No row stores a depth per group, so this prompt
cannot say which group reached depth 20. `regions` and `elapsed` are the per-group cost, and they
are what it scores.

## 1. Predictions, fixed before the census runs (README §5 rule 3)

**Definitions.**
- A **depth-20 row** has `WKB_Levin_max_depth = 20`.
- A **group record** is one entry of a sub-interval's `groups`.
- A record is **cancelling** if its label gives $G$ and a transfer factor the **same sign**, and
  that factor's wavenumber **equals** $k$. On `r == k` these are `G+r`, `G+q+r` and `G-q+r`. On
  `q == k` they are `G+q`, `G+q+r` and `G+q-r`. Every other record containing $G$ is
  **non-cancelling**: the lone `G`, groups whose $G$ and factor carry opposite signs, and groups
  whose same-signed factor's wavenumber is not $k$.
- **Why the same sign.** $\theta_G$ decreases with $z'$ while $\theta_q$ and $\theta_r$ increase
  (`prompts/source-remediation/logs/07-phase-group-algebra.md`, note 4). So the stationary group
  of $G\,T_r$ on `r == k` is `G+r`. That note warns to check `theta_deriv` rather than assume
  this. **Confirm it from the code before the census runs:** read how `GkSourceFunctions.phase`
  and `TkSourceFunctions.phase` are built, and record the reading, with file and line, in the log.
  If the code says otherwise, **stop and ask**, since every prediction below depends on it.
- **The floor** is $F = \varepsilon\,k\,\eta(z_{\rm response})$, the quantity of findings §2.1.
- A group record's **declared relative error** is $e_g = \texttt{abserr}/|\texttt{value}|$.

**P1 — where the cost sits.** $S$ is the share of the summed per-group `elapsed`, over all
depth-20 rows, that falls on non-cancelling records.
- **H-cancel** predicts $S < 0.25$: the cost is in cancelling groups.
- **H-anchor** predicts nothing about $S$ by itself. It holds only if P2 holds as well.
- **The rule:** $S \ge 0.25$ **fails H-cancel**.

**P2 — the error tracks the floor whatever the signs.** Take the depth-20 group records in the mid
and low bands with $e_g$ finite and non-zero. Split them into two classes, cancelling and
non-cancelling, and in each class fit $\log e_g$ on $\log F$ by least squares. For each class
report the slope, the median of $e_g/F$, and its 10th and 90th percentiles.
- **H-anchor** predicts that both classes, if each has at least 100 records:
  - have a slope in $[0.6, 1.4]$;
  - have a median $e_g/F$ in $[0.1, 10]$;
  - have medians within a factor of 10 of each other.
- **H-cancel** predicts that the cancelling class satisfies those slope and median conditions, and
  the non-cancelling class does not.
- **The rule:** score both hypotheses on these conditions as written. A class with fewer than 100
  records makes the verdict **undecided** for the hypothesis that needs it.

**P3 — the cost follows the floor, not the group.** Among non-cancelling depth-20 records, take
the share of `elapsed` on records with $F > \texttt{rtol} = 10^{-7}$, and the same share among
cancelling records.
- **H-anchor** predicts both shares are at least 0.8.
- This is descriptive for H-cancel.
- **The rule:** report both shares, and score H-anchor on them.

Write these three rules, unchanged, into the log's "Predictions and verdicts" section, and score
them there. Do not re-pose a rule after seeing its number. If one proves badly posed, say so and
leave it scored as written.

## 2. Steps

Make `var/levin-phase-floor-01/{out,scripts,state}`. Save every command and its output. Run
commands with variable arguments through `bash`.

1. **Check out the dispatch commit.**
   - `git fetch origin && git merge --ff-only origin/handover-remedial`;
   - `HEAD` is the dispatch SHA, and `git status --short` is empty;
   - `./venv/bin/pip freeze` is unchanged by the move (save it before and after; no pin moved
     since `c9026af`).
2. **Hash the store.** The SHA-256, size and mtime of the primary, the four shards and the sidecar:
   `state/store_files_before.txt`.
3. **Write `scripts/group_census.py`**, read-only and Ray-free, following `depth_census.py`. It
   writes a header with:
   - its own SHA-256;
   - each store file's SHA-256 from step 2;
   - the store's fingerprint, from `store show`;
   - `git HEAD`.

   Then:
   - **(a) The population.** QuadSourceIntegral rows at `(1e-32, 1e-7)`. **9280 expected; any other
     count is a stop.** Every row's `metadata` must parse.
   - **(b) One flat record per group** for every row with a Levin sub-interval, in
     `out/groups_01.csv`:
     - shard, serial, $k$, $q$, $r$, $z_{\rm response}$, band (low $<9$, mid $9$–$10^5$, seam
       $>10^5$, as `depth20_floor.py`), and geometry (`q == r == k`, `r == k` with $q<r$, `q == k`,
       other);
     - the row's `WKB_Levin_max_depth`, `compute_time`, `total`, `total_abserr` and
       `total_converged`;
     - the sub-interval's `z_min`, `z_max` and regime string;
     - the group's label and all nine fields;
     - the cancelling flag of §1, $F$, and $e_g$.
   - **(c) The tables.** Write them to `out/census_01.txt`:
     - **T1.** `elapsed` (CPU-hours), `regions` and record count by (regime, label), over all rows
       and over depth-20 rows. Each with its share of the column total.
     - **T2.** For each depth-20 row, the group with the largest `elapsed`. Counts by
       (geometry, regime, label). Also the median share of the row's Levin `elapsed` that this top
       group takes.
     - **T3.** T1's depth-20 column split by band and by $k$.
     - **T4.** The P1–P3 statistics, exactly as §1 defines them.
     - **T5.** Where in source time the cost sits. For the top group of each depth-20 row, the
       sub-interval's $\log(1+z_{\rm max})$ and $\log(1+z_{\rm min})$, summarised by band and
       label as quartiles. This is descriptive and scores nothing.
   - **(d) Findings §2.5, confirmed on the store.**
     - The rows whose `total_abserr` is not finite: their count, keys, and which sub-interval and
       group carry a non-finite `abserr`. Report how SQLite returns them (`inf`, `NULL`, …) as
       read.
     - Recompute the 1086 / 257 / 132 / 5.0e-2 counts of §2.5 item 1 from the store.
     - At group level: of the group records flagged `converged`, how many have $e_g > 10^{-6}$ and
       how many have $e_g > 10^{-4}$, on depth-20 rows and on all rows.
   - **(e) The candidates for 03** (README §4.3). At least two rows per class: top group `G+r`
     (slow) on `r == k`; `G-r` (fast) on `r == k`; the lone `G`; a `Gqr` group. Prefer rows of 100–800 s in the mid
     band, and fall back to the low band where the mid band has none. Write `out/candidates_01.txt`
     with each row's `(shard, serial)`, $(k, q, r, z_{\rm response})$, `compute_time` and per-group
     records.
4. **Run it** with `./venv/bin/python -I scripts/group_census.py <primary>`. It should take
   minutes. If it may exceed ten minutes, run it in the background and wait for the notification.
   Do not poll.
5. **Check the reconnaissance.** Re-derive README §0.2's table from `groups_01.csv`: the depth-20
   rows over 100 s, the same population. Any difference from `measure/out/recon_groups.txt` is
   explained in the log.
6. **Hash the store again:** `state/store_files_after.txt`, identical to step 2. Run
   `store fingerprint` (without `--write`); it must equal the sidecar's.
7. **Write the records.**
   - `log.md`, in README §5.1's form, with **Predictions and verdicts** first;
   - `verification.md`: the text of `docs/levin-phase-floor/VERIFICATION.md` §1 "The phase-group
     census", giving T1–T5, P1–P3 with their verdicts, and §2.5's confirmations;
   - `issues.md`, holding anything found and not acted on.

## 3. Acceptance

1. 9280 rows read. Every store file is byte-identical before and after, and the fingerprint is
   `0e32f24a…`.
2. `groups_01.csv` and `census_01.txt` exist, and T1–T5 are complete.
3. P1–P3 are each scored holds, fails or undecided, against §1's rules as written, with the number
   each was scored on.
4. §2.5's counts are recomputed, and the non-finite rows are identified to group level.
5. `candidates_01.txt` holds at least two rows per class, or says which class has none.
6. Records under `var/levin-phase-floor-01/`: `out/`, `scripts/`, `state/`, `log.md`,
   `verification.md` and `issues.md`.

The orchestrator then:
- copies everything to `docs/levin-phase-floor/01/`, with `groups_01.csv` gzipped if it is over
  10 MB;
- creates `docs/levin-phase-floor/VERIFICATION.md` with §1;
- writes `logs/01-phase-group-census.md`;
- records the verdicts and the narrowings on the boards;
- commits.

## 4. Stop conditions — stop and ask

- Any Gate item not as stated.
- A population other than 9280, or a row whose `metadata` does not parse or has no `WKB_Levin`
  key.
- Any store file changes.
- A depth-20 row with no group records.
- The code's sign convention is not the one §1 states.
