# levin-phase-floor 01 — the phase-group census of A3 v2

> **Written on the Mac Studio, committed on the laptop** by the orchestrator (README §4.1). A path
> `scripts/…`, `out/…` or `state/…` below is under `var/levin-phase-floor-01/` there, and is
> committed at `docs/levin-phase-floor/01/`, except the CSVs, which `.git/info/exclude` keeps out of
> git and which stay on the Mac Studio. `verification.md` is
> `docs/levin-phase-floor/VERIFICATION.md` §1, corrected on committing (review check 9).

**Subject:** prompt 01, `01-phase-group-census.md`, run on the Mac Studio at dispatch SHA `757ef01`.
**Commit:** none here; records committed by the orchestrator (README §4.1).
**Result:** pass. No stop condition met. The census read 9280 rows, and the store is byte-identical before and after, at fingerprint `0e32f24a…`. **P1 fails H-cancel ($S$ = 0.4385). P2 holds for H-anchor and fails H-cancel. P3 holds for H-anchor.**

## Predictions and verdicts

The rules below were written into this file at the time recorded in
`state/predictions_written.txt`, before `scripts/group_census.py` was written or run. They are
prompt §1 lines 54–101, copied unchanged (`sed -n 56,101p`); the verdicts follow in the subsection
after them.

### The rules, as the prompt wrote them

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

### The verdicts

The rules above were fixed before the census ran. Each is scored here against its words, with the
number it was scored on, from `out/census_01.txt` T4.

**Before the census: the sign convention, from the code.** Prompt §1 makes this a stop if it
disagrees. It agrees: $\theta_G$ decreases with $z'$, and $\theta_q$ and $\theta_r$ increase, so
the same-signed group is the slower one.
- **$G$.** `GK_PHASE_SIGN = -1` (`ComputeTargets/GkSourcePolicyData.py:136-142`) is passed to
  `PrimitivePhase` in `_build_phase` (`:184`, `:193`). That is how `GkSourceFunctions.phase` is
  built (`:742-751`, `phase=WKB_theta_phase`).
  - `PrimitivePhase.raw_theta` returns $\text{sign}\cdot k\cdot\tau.\delta(z, z_R) + \phi$
    (`ComputeTargets/primitive_phase.py:276-283`).
  - `theta_deriv` is $\text{sign}\cdot k\cdot \text{rate}(z)\,(1+z) + \phi'$, with rate $= 1/H > 0$
    by default (`:291-310`).
  - So $d\theta_G/d\log(1+z') = -k(1+z')/H + \phi' < 0$.
- **$T$.** `TK_PHASE_SIGN = +1` (`ComputeTargets/TkSourceFunctions.py:184`) is passed to
  `PrimitivePhase` at `:393` and `:403`, with rate $c_s/H > 0$ (`_sound_horizon_rate`,
  `:205-236`). The module docstring (`:72-74`) says $\theta$ is an increasing function of $z$.
- **What the composed groups use.** `phase_groups._OscillatoryT.theta_deriv` (`phase_groups.py:395-397`)
  returns $\omega\,(1+z) > 0$, with $\omega = \sqrt{\cdot}$. `_OscillatoryG.theta_deriv` (`:436-437`)
  returns the $G$ phase's `theta_deriv`, which is negative. `_composed_theta_deriv` (`:553-557`) is
  the signed sum.
- **Agreement with the prompt.** `prompts/source-remediation/logs/07-phase-group-algebra.md` note 4
  says the same. This reading does not rest on it.
- **A caveat that is not a stop.** The leading rates differ: $1/H$ for $G$ and $c_s/H$ for $T$. So
  on `r == k`, `G+r` is slower than `G−r` but not stationary (`issues.md` I1).

**P1 — where the cost sits.** $S$ = **0.4385**: 125.270 of 285.650 depth-20 CPU-hours are on
non-cancelling records, and none is on a record without $G$.
- **H-cancel: fails.** The rule is *"$S \ge 0.25$ fails H-cancel"*, and $0.4385 \ge 0.25$.
- **H-anchor: holds, by P2 and not by $S$.** The rule says H-anchor *"predicts nothing about $S$ by
  itself. It holds only if P2 holds as well."* P2 holds for H-anchor (below), so it holds here.
  $S$ was not tested against H-anchor.

**P2 — the error tracks the floor whatever the signs.** These are the depth-20 records in the mid
and low bands with $e_g$ finite and non-zero, fitted as $\log_{10} e_g$ on $\log_{10} F$:
- **cancelling:** n = 2946, slope **0.968**, median $e_g/F$ **2.117** (10th 0.257, 90th 13.52);
- **non-cancelling:** n = 6943, slope **0.822**, median **1.489** (10th 0.148, 90th 20.42);
- the ratio of the medians is **1.42**.

The verdicts:
- **H-anchor: holds.** Both classes have at least 100 records. Both slopes are in $[0.6, 1.4]$, both
  medians are in $[0.1, 10]$, and the medians are within a factor of 10 of each other (1.42).
- **H-cancel: fails.** The cancelling class meets the slope and median conditions, as H-cancel
  predicts. But H-cancel also predicts that the non-cancelling class does not, and it does (0.822,
  1.489).
- **Neither class falls under 100 records**, so neither verdict is undecided.

**P3 — the cost follows the floor, not the group.** The share of `elapsed` on depth-20 records with
$F > 10^{-7}$:
- non-cancelling: **0.9991** (125.160 of 125.270 CPU-hr);
- cancelling: **0.9997** (160.339 of 160.380 CPU-hr).

The verdicts:
- **H-anchor: holds.** Both shares are at least 0.8.
- **H-cancel:** descriptive, as the rule says.

**How well posed the rules were.** None is re-posed, and each stays scored as written.
- **P3 discriminates weakly.** $F$ is a property of the row, shared by all its groups, and depth-20
  rows are almost all rows with $F > {\rm rtol}$. Both classes would pass under any hypothesis in
  which the floor selects the rows (`issues.md` I5).
- **P2's windows are wide.** By band the slopes differ more: low 1.29 and 1.24, mid 0.89 and 0.76.
  So do some labels: the lone `G` has slope 0.65 and median 0.34, `G−q+r` slope 0.45. These numbers
  are descriptive and change no verdict.
- **The verdict pattern.** H-anchor holds on every rule, and H-cancel fails P1 and P2. That does
  not exclude README §0.2's *"both can hold"*, a floor everywhere made worse where a slow group
  forms. T3 shows `G+r`'s share of the cost rising with $k$, from 24.5 % at $9.85\times10^5$ to
  80.8 % at $9.56\times10^7$, while `G−r`'s does not. No rule scores that, and it is for 03.

## What shipped

Everything is under `var/levin-phase-floor-01/`. Nothing was committed, and nothing outside this
directory was written.
- **`scripts/`:**
  - `gate_01.sh`, the Gate, read-only;
  - `hash_store.sh`, the store files' SHA-256, size and mtime;
  - `peek_schema.py`, a read-only look at the schema and one row's metadata before writing the
    census;
  - `group_census.py`, the census, steps 3–4;
  - `recon_check.py`, step 5;
  - `eta_check.py`, a follow-up to an $\eta$ anomaly seen in the census header.
- **`out/`:**
  - `gate_01.txt` and `peek_schema.txt`;
  - `census_01.txt`, with T1–T5, (d) and the summary of (e);
  - `groups_01.csv`, 37,325 records, 12.9 MB;
  - `candidates_01.txt`;
  - `recon_check_01.txt` and `recon_diff_01.txt`, the latter empty but for its exit line;
  - `eta_check_01.txt`;
  - `fingerprint_01.txt`, from `store fingerprint` without `--write`.
- **`state/`:**
  - `pip_freeze.txt` and `predictions_written.txt`;
  - `store_files_before.txt`, `store_files_after.txt` and `store_files_diff.txt`;
  - `store_files_before_FAILED_gnu_stat.txt`, the first attempt, made with GNU `stat`. Its SHA-256
    values equal the good run's;
  - `group_census_run.{stdout,stderr,exit}`, for the final run;
  - `git_status_after.txt`;
  - `superseded_run1/` and `superseded_run2/`, the two earlier census runs. Neither is a result
    (deviation 3).
- **The records:** `log.md`, `verification.md` (the text of `VERIFICATION.md` §1) and `issues.md`.

## Deviations from the prompt

1. **Step 1 became a check — `STRUCTURALLY REQUIRED` (parallel dispatch).** No `git fetch` or
   `git merge` was run, because `handover` 11 ran in the same checkout and no git command that
   writes could be run (orchestrator note §0). Instead:
   - `HEAD` was checked to equal the dispatch SHA `757ef016…`;
   - `git status --short` was checked to be empty;
   - `pip freeze` was recorded once (`state/pip_freeze.txt`), not before and after a move, since
     there was no move.

   The Gate's "pushed" item was checked read-only, without a fetch: `origin/handover-remedial`, as
   last fetched, equals `757ef01`. `git merge-base --is-ancestor` also confirms that the prompt's
   commit `6af798f` is an ancestor of the dispatch SHA.
2. **$\eta$ was imported, not copied — `IMPLEMENTATION CHOICE`.** The prompt offered either.
   `group_census.py` imports `eta` and `EPS` from `docs/handover/a3-v2/findings/depth20_floor.py`
   by file path, so $F$ is exactly findings §2.1's quantity. The output header records that file's
   SHA-256.
3. **The census ran three times — `IMPLEMENTATION CHOICE`.** Each run took about 3 s. Each re-run
   was a full recompute by a script with a new hash, and nothing was re-stamped (README §5 rule 5).
   - **Run 2** added a summary of the non-finite rows. Run 1 had found 558, not 5, and listed them
     one by one with no summary.
   - **Run 3** corrected `groups_01.csv`'s `F` column, which had been written as
     `np.float64(…)`. Run 3's `census_01.txt` body is identical to run 2's.

   Runs 1 and 2 are kept under `state/superseded_run{1,2}/`. `superseded_run2/` also holds two
   other files. One is the step-5 check made on run 2's CSV, identical to the final one. The other
   is the first `eta_check.py` attempt, which failed on the `np.float64(…)` strings; that failure
   is how the CSV defect was found.
4. **Two additions to (b) — `IMPLEMENTATION CHOICE`.** The CSV has a `sub_index` column, and the
   provenance header is repeated as `#` lines at its top. `recon_check.py` skips those lines.
5. **Three descriptive additions to T4 — `IMPLEMENTATION CHOICE`:**
   - $S$ within each band;
   - P2's fit within each band and within each label;
   - P2's excluded-record counts.

   None is scored.
6. **Two choices in (e) — `IMPLEMENTATION CHOICE`:**
   - A class with more than 8 candidate rows was thinned to 8, spread over $k$ (round-robin by $k$,
     the middle row by time first).
   - The `G+r` and `G−r` classes include both `r == k` geometries. Every row picked happens to be
     `r==k,q<r`.

No `UNINTENDED DRIFT`.

## Verification performed

- **The Gate** (`out/gate_01.txt`):
  - `HEAD` = `origin/handover-remedial` = `757ef01`, and the prompt's commit is an ancestor of it;
  - `runregistrykit list` shows 10 runs, none running or stale;
  - `git status --short` is empty;
  - `store show` gives kind `registry`, problems none, digest `0e32f24a2dc2…d56`.
- **Population.** 9280 rows at `(1e-32, 1e-7)`, 2320 per shard and 0 at any other pair. All the
  metadata parses and has `WKB_Levin`. Every one of the 2265 depth-20 rows has group records.
- **The store is untouched:**
  - six files hashed before and after, and the `diff` is empty (`state/store_files_diff.txt`);
  - the census header records the same six SHA-256 values, each marked `(= step 2)`;
  - `store fingerprint`, without `--write`, gives `0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56`
    and reports *"recorded: matches"*.
- **The checkout after the census.** `git status --short` is empty and `HEAD` is unchanged
  (`state/git_status_after.txt`).
- **The reconnaissance.** `recon_check.py` reproduces `measure/out/recon_groups.txt` from
  `groups_01.csv` line for line, apart from the header, so `recon_diff_01.txt` is empty: 629 rows,
  282.93 and 282.88 CPU-hr, and every table entry.
- **Findings §2.5 item 1.** 1086, 257, 132 and 5.03e-2 are all reproduced from the store.
- **$\eta$.** `depth20_floor.py`'s $\eta$ against the stored `eta_response` gives 1.0009–1.0015 in
  the low and mid bands, findings §2.1's 0.09–0.13 %.

## Observations not acted on

All of these are in `issues.md`.
- **I1.** The slow group's leading terms do not cancel. $G$'s leading phase rate is $k/H$ and $T$'s
  is $k c_s/H$, so on `r == k` the group `G+r` oscillates at about $k(1-c_s)/H$, not zero. That is
  about 0.42 $k/H$ in radiation domination, where T5 puts the costly sub-intervals. Findings §2.2
  and README §0.2 say the terms "cancel identically", and that is not what the code does. This is
  for 03 to check numerically, and for G1 Q2(b).
- **I2.** 558 rows have `total_abserr` = +inf: 5 at depth 20 and 553 at depth 1–19, mostly in the
  seam band. 803 group records carry it, mainly `q−r` and `G±q∓r` on $q = r$. This narrows
  `[04-roundoff-floor-can-be-infinite]`: confirmed on the store, and much wider than findings
  §2.5's five.
- **I3.** README §0.2's "229.7 of the 282.9 CPU-hours in the low band" should be 247.88, by the
  reconnaissance's own table.
- **I4.** `depth20_floor.py`'s $\eta$ is wrong above $z \sim 10^{12}$ and negative above
  $10^{15}$. That has no effect here.
- **I5.** P3 discriminates weakly.

## State handed on

- **The verdicts.** H-cancel fails P1 and P2, and H-anchor holds on P1–P3. I1 and T3's
  rise of `G+r` with $k$ are the open threads for 03.
- **03's rows.** `out/candidates_01.txt` holds the candidates: 8 `G+r`, 8 `G−r`, 3 lone `G` and 8
  `Gqr` rows, all in the mid band at 100–800 s, each with its per-group records.
- **What the orchestrator does next:**
  - copy to `docs/levin-phase-floor/01/`, gzipping `groups_01.csv` (12.9 MB, over 10 MB);
  - create `VERIFICATION.md` §1 from `verification.md`;
  - place I1–I5 on the boards;
  - narrow the four issues the prompt names.
- **The checkout and the store.** The checkout is unchanged at `757ef01`, and `var/handover-11/`
  was neither read nor written. A3 v2 is untouched.
