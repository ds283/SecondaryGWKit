### 5a.x The wider re-measurement for $T$ (prompt 05a §6, with §9's T1)

**Prompt 05a §6, with T1 as amended by §9, run 2026-10-26 on the Mac Studio (mps026298).** There
was no Ray, no lock and no store write (route (a), as in §3–§5). Two trees took part:
- $c$, the code commit's parent `c4eaefd`, whose code is 05's $c$: $G$ node-anchored, $T$ not. It
  ran from the git worktree `SecondaryGWKit-lpf05a-c4eaefd`.
- $t$, the code commit `1fc349c`: $T$'s remainder node-anchored and reduced against $2\pi$. It ran
  from the main checkout at `149c393`, whose code is `1fc349c`'s up to `bd38e50`'s comments.

The rows are 05's 39, from 05's `sample.json` copied unchanged
([`state/sample.json`](05a/wider/state/sample.json), SHA-256 `73aa7876…`, equal to the
`sample_sha256` 05's run recorded). The harness is 05's
([`scripts/`](05a/wider/scripts/); its diff against 05's copy is
[`out/scripts_diff_against_05_final.txt`](05a/wider/out/scripts_diff_against_05_final.txt)).
Its trees are renamed $c$/$t$, and it adds the phase probe ([`scripts/probe.py`](05a/wider/scripts/probe.py)).

Each worker loaded its packages from its own tree only, as recorded per unit. Under $c$, every $T$
phase had `exact_remainder` False and every $G$ phase True. Under $t$, both were True.

**The run.** `levin-phase-floor-05a-wider-20261026T123047` ended `done`, 109/109 units, exit 0,
in under two minutes. Its checkpoints and pickles stay in that run directory on the Mac Studio.
- The units ran in order: $c$@1e-7 on all 39 rows, $t$@1e-7 on all 39, and $t$@1e-8 on the 31
  depth-20 rows. The budget of the $t$ units was 3× the row's $c$ wall time.
- Every record carries `run_wider.py`'s SHA-256 (`af656ad4…`) and its tree's `HEAD` and empty
  `git status --short`.
- No unit went over budget (the largest wall/budget is 0.861), none raised, and the probe raised
  nowhere.

**The store** was byte-identical before, before the launch, after the run and after the
fingerprint ([`before`](05a/wider/state/store_files_before.txt),
[`after`](05a/wider/state/store_files_after_fingerprint.txt)). No journal, `-wal` or `-shm` file
appeared. `store fingerprint`, run without `--write`, gives
`0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56` and `problems: none`
([`out/fingerprint_05a_wider.txt`](05a/wider/out/fingerprint_05a_wider.txt)).

**The phase probe** reads phase objects only and runs no quadrature.
- **Where.** It covers every Levin sub-interval of every row with an oscillatory $T$ factor, and
  every such factor: 91 windows on 37 rows. Each window starts at the sub-interval's low-$z$ end
  $z_{\min}$ and is $[z_{\min}, z_{\min}(1+w)]$ with $w = 10^{-6}/2^n$.
- **What each tree computes at every level** until its 65 samples stop being distinct doubles:
  - $\sigma_\theta$ of `theta_mod_2pi` sampled in $z$, as §3 test 3 computes it;
  - the same sampled in $u$;
  - the largest unwrapped step;
  - $\max|p_z|$, from `cs_tau.end_partial`.
- **The level used** is the first at which both trees' largest unwrapped step is below $\pi/2$.
  No window with $\varepsilon|\theta_T| \ge 10^{-11}$ was unresolved.
- **Where it is kept.** Every level of both ladders, and the samples as `float.hex` at the level
  used, are in [`state/probe_windows.json`](05a/wider/state/probe_windows.json).
  [`scripts/rescore_t1.py`](05a/wider/scripts/rescore_t1.py) re-scores T1 from that file alone,
  recomputing every $\sigma$ from the samples, and agrees in every window
  ([`out/cmd/24_rescore_t1.txt`](05a/wider/out/cmd/24_rescore_t1.txt)).

**Outputs:**
- [`out/T0.txt`](05a/wider/out/T0.txt);
- [`out/wider_05a.txt`](05a/wider/out/wider_05a.txt): every unit, every group, every probe window
  under both readings, and the scores;
- [`out/convergence_kinds_t.txt`](05a/wider/out/convergence_kinds_t.txt);
- [`out/commands.log`](05a/wider/out/commands.log), which has every command with its output.

#### Gate T0

| | Number | Verdict |
|---|---|---|
| **(a)** `git diff --stat 120fdfe c4eaefd -- . ':!docs' ':!prompts'` | `CLAUDE.md` alone | holds |
| **(b)** $c$@1e-7 against 05's $c$ unit, on `low-G+r-1` and `ctrl-low-1` | `total`, `total_abserr`, and all 7 calls' value and abserr: 16 doubles per row, identical by `==` and `float.hex`. So are the whole result summary, the call summaries and every region array of 05's pickles | holds |
| **(c)** two in-process runs of the same $t$ unit, on both rows | result, calls, regions, probe summaries and every probe sample bit-identical | holds |

[`state/predictions.md`](05a/wider/state/predictions.md) was written at 12:30:03Z, after T0 held
(12:28:33Z) and before the 12:30:47Z launch. It quotes §6's predictions with §9's T1 in place of
§6's, and fixes how each rule is read.

#### Verdicts, scored as written

*(T1's scored verdict was later taken by the true step, by the user's decision: it holds. See
below the table. The T1 row here is the literal reading, kept beside it.)*

| | Rule | Number | Verdict |
|---|---|---|---|
| **T1** (§9) | in every resolved window with $\varepsilon\lvert\theta_T\rvert \ge 10^{-11}$ rad, sampled in $z$: $\sigma_c \ge 0.1\,\varepsilon\lvert\theta_T\rvert$ and $\sigma_t \le \max(10^{-3}\varepsilon\lvert\theta_T\rvert,\ \varepsilon(k\max\lvert p_z\rvert+2\pi),\ 10^{-13})$ | 70 scored windows, 0 unresolved. The $c$ clause holds in all 70: $\sigma_c$ = 0.196–0.374 $\varepsilon\lvert\theta_T\rvert$. The $t$ clause holds in 34 ($\sigma_t \le 1.3\times10^{-5}\,\varepsilon\lvert\theta_T\rvert$) and fails in 36 ($\sigma_t$ 0.016–0.23 $\varepsilon\lvert\theta_T\rvert$). Those 36 are exactly the windows the literal step rule accepts while the phase steps 4.9–352 rad per sample (below) | **fails** |
| **T2** | every row (39), $t$@1e-7: no group has a region at depth ≥ 18 | 39 of 39; largest depth 11 | **holds** |
| **T3** | every row (39): $\lvert{\rm total}_t - {\rm total}_c\rvert \le$ `total_abserr`$_c$ at 1e-7 | 39 of 39; largest ratio $4.66\times10^{-6}$ (`low-G-1`); largest relative change $3.6\times10^{-12}$ (R4); 13 totals bit-identical | **holds** |
| **T4** | every depth-20 row (31) with both $t$ units in budget: $\lvert t(10^{-7}) - t(10^{-8})\rvert \le$ `total_abserr`$_t(10^{-7})$ | 29 of 31; fails on `mid-G-r-3` (1.29) and `mid-G-3` (1.70), the same rows and ratios as 05's L3 under $c$; holds on all 11 rows capped under B1 (largest 0.166) | **fails** |

**T1's failure is the probe's step rule, not the phase.**
- §9 halves a window while "the largest step of the unwrapped samples" is $\pi/2$ or more. Read
  as written, that is the folded step, which can be small while the samples advance by many cycles.
- On those windows, `numpy.unwrap` removes whole cycles per index. The rounding of the abscissae
  then reads as jitter of $0.13$–$0.30\,\varepsilon z|d\theta_T/dz|$, whatever the remainder.
- **By the true step** (unscored; both readings were fixed before the run, and both are kept), the
  same rule holds in **70 of 70** windows:
  - $\sigma_t$ is $4.4\times10^{-10}$–$1.31\times10^{-5}\,\varepsilon|\theta_T|$, at most 0.013 of
    its bound;
  - $\sigma_c$ is 0.198–0.374.
  - The reading by the true step means the first level at which $|d\theta_T/dz|(b-a)/64 < \pi/2$
    also holds in both trees.
- The question of which reading §9 means is the user's (05a's `issues.md` X1).

**T1's scored verdict, by the user's decision after the run: holds.**
- The coordinator relayed the decision to the Mac Studio session at about 13:23Z. The user chose
  "True step (Recommended)": *"T1 is scored on windows whose true step |dθ/dz|(b−a)/64 is below
  π/2, narrowing further where it isn't. This is what §9 meant by 'resolved'. The literal
  unwrapped-step reading is reported beside it, unscored."*

| | Rule | Number | Verdict |
|---|---|---|---|
| **T1** (§9, scored by the true step) | as above, on windows resolved by $\lvert d\theta_T/dz\rvert(b-a)/64 < \pi/2$ under both $c$ and $t$ | 91 windows, 70 with $\varepsilon\lvert\theta_T\rvert \ge 10^{-11}$, 0 unresolved, true in 70. $\sigma_c$ 0.198–0.374 $\varepsilon\lvert\theta_T\rvert$; $\sigma_t$ $4.4\times10^{-10}$–$1.31\times10^{-5}\,\varepsilon\lvert\theta_T\rvert$, at most 0.0131 of its bound ([`out/cmd/26_T1_true_step_scored.txt`](05a/wider/out/cmd/26_T1_true_step_scored.txt)) | **holds** |
| T1 (§9, literal step; now unscored) | the table above | 36 of 70 false, all aliased | (fails) |

- **No re-probe was needed.** The kept levels resolve every window by the true step, and the level
  chosen by the true step alone is, in all 91 windows, the level whose samples were saved.
- **The timing, as the files show it,** is in
  [`state/predictions_note_T1_reading.md`](05a/wider/state/predictions_note_T1_reading.md):
  - `predictions.md` fixed both readings at 12:30:03Z;
  - both were first scored at 12:35:49Z;
  - the decision reached the Mac Studio session at about 13:23Z.

**§9's unscored items.**
- The term $\varepsilon(k\max|p_z| + 2\pi)$ is the largest of the three in **0** windows: the
  windows sit at or near table nodes, with $k\max|p_z| \le 10^{-6}|\theta_T|$.
- **Sampled in $u$**, as production calls the phase, $\sigma_t$ is 0.14–0.29 of the abscissa term
  $\varepsilon\tfrac{z}{1+z}|d\theta_T/du|$ everywhere.
  - It exceeds $0.1\,\varepsilon|\theta_T|$ in 14 windows at $z \gg 1$, at 0.10–0.22.
  - Under $c$ it is 0.21–0.40.
  - At $z < 1$ it falls to 0.0025–0.0096 under $t$.
  - This is §0.1's prediction: for $T$ the abscissa term is the size of the removed floor wherever
    $z \gg 1$. An issue is proposed (X2).

#### Descriptive

- **$t$ against $c$:**
  - all 219 Levin group calls are the same sequence, with the same region count, maximum depth and
    convergence flag. That includes the 140 `Gqr` groups and the regions at each sub-interval's
    low-$z$ end;
  - `total_converged` is unchanged on every row;
  - the totals move by at most $4.7\times10^{-6}$ of `total_abserr`$_c$.

  On these rows, $T$'s two errors were never what the integrator saw.
- **Cost:** the regions are unchanged (low band 288, mid band 321, R1–R7 250, the 11 capped rows
  426 at 1e-7). The wall times are 0.06–1.15 s per unit and carry the host's load.
- **The kinds of non-convergence** under $t$ ([`out/convergence_kinds_t.txt`](05a/wider/out/convergence_kinds_t.txt);
  `convergence_kinds.py` unchanged, SHA-256 `49c83a72…`):

  | | resolution | bound only | phase-limited |
  |---|---|---|---|
  | 1e-7 | 18 | 32 | 17 |
  | 1e-8 | 21 | 24 | 27 |

  These equal 05's $c$ (`prompts/levin-phase-floor/measure/out/convergence_kinds.txt`).
