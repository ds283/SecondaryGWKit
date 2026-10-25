# Log 09 — `analytic_G_w` outside its domain: store `NULL`, and plot only what is sensible

**Prompt:** [`prompts/handover/09-analytic-g-w-outside-its-domain.md`](../09-analytic-g-w-outside-its-domain.md),
including the dated addendum after §2's item 5
**Commit:** `1c19998` — "Return None from the constant-w Green's function outside its domain"
**Model:** Claude Opus 5.5, in a fresh-context subagent on the laptop, dispatched at `1739e0d`
**Date:** 2026-10-25 (09:14–10:30 GMT)
**Result:** COMPLETE WITH DEVIATIONS. Every acceptance item of prompt §6 holds, and no §7 stop condition
fired. The rehearsal store fingerprints `39809dca…`.

The working notes are under `var/handover-09/`, which is ignored:
- `prelaunch.txt`, `lock.txt`, `launch.txt`, `check.txt`;
- `state.sh`, with its outputs `state-before-sidecar.txt` and `state-after-compare.txt`;
- `sidecar.txt`, `compare.txt`, `null-census.txt`, `readers.txt`;
- `extract/`.

The suite logs and the breakage outputs went to the session scratchpad.

## What shipped

**`ComputeTargets/analytic_Gk.py`**
- **New module docstring.** It cites spec 02 R2–R9, the domain $1+3w>0$ and why it holds, and the
  A3 v2 value.
- **New `constant_w_form_applies(w: float) -> bool`** (`:20`). It returns `1.0 + 3.0 * w > 0.0`, the
  same expression as the denominator of $b$, with no tolerance band.
- **`compute_analytic_G(k, w, tau_source, tau, H_source) -> Optional[float]`** (`:32`):
  - returns `None` when `not constant_w_form_applies(w)`, before `b` or any Bessel function is
    evaluated;
  - then evaluates the expression exactly as before, same operations in the same order, into `G`;
  - returns `None` if `not isfinite(G)`, otherwise `G`.
- **`compute_analytic_Gprime(k, w, tau_source, tau, H_source, H) -> Optional[float]`** (`:72`):
  likewise, into `Gprime`.
- **The docstrings** cite spec 02 R2/R9 and §0.1 item (2). They say when `None` is returned and
  why, and that an exact 0 is returned.
- **The callers are unchanged** (`GkNumericIntegration.py:465-498`, `GkWKBIntegration.py:535-585`).
  Read and confirmed:
  - `GkNumericValue` and `GkWKBValue` take `analytic_G_w`/`analytic_Gprime_w` as `Optional[float]`
    (`GkNumericIntegration.py:521-524`, `GkWKBIntegration.py:612-613`);
  - the four stored columns are `nullable=True` (`ObjectFactories/GkWKBIntegration.py:1017-1020`,
    `ObjectFactories/GkNumericIntegration.py:683-686`);
  - `GkSource.py:498-506` chooses numeric or WKB by `is not None` on the object, so a `None` value
    passes through.

**`extract_common.py`**
- `from math import fabs` → `from math import fabs, isfinite`.
- **New `ANALYTIC_PLOT_DECADES = 3`** (`:411`), with a comment saying it is a plotting choice.
- **New `sensible_log_points(points, reference_values, decades=ANALYTIC_PLOT_DECADES, signed=False)
  -> List[Tuple[float, float]]`** (`:414`), beside `safe_fabs`. It keeps a point `(x, y)` when:
  - `y` is not `None`, is finite and is non-zero;
  - and `|y| ≤ 10**decades ×` the largest finite non-zero `|value|` in `reference_values`. With an
    empty or all-`None` reference there is no ceiling.

  It returns `(x, |y|)`, or `(x, y)` when `signed=True`, in input order. There is one rule; the
  boundary is applied to `|y|` in both forms.

**The four readers.** Each imports `sensible_log_points` from `extract_common`.
- **`extract_GkWKB_data.py`** (`plot_Gk`), and in addition `safe_fabs`:
  - `analytic_w_points` became `analytic_w_raw_points`, filled with the signed stored values from
    both branches (`:177-179`, `:218-220`);
  - after both branches,
    `analytic_w_points = sensible_log_points(analytic_w_raw_points, abs_G_column)` (`:267-269`).
    The reference is the plot's $|G|$, numeric and WKB together;
  - every `fabs(value.analytic_…)` became `safe_fabs(…)`: the two radiation plot-point lists and the
    four CSV `abs_` columns (`:175`, `:196-203`, `:216`, `:239-246`);
  - `fabs` remains only on `G`, `G_WKB` and `theta`.
- **`extract_Gk_data.py`** (`plot_Gk`):
  - `abs_analytic_G_w_points = sensible_log_points([(value.z, value.analytic_G_w) …], [value.G …])`
    (`:147-152`);
  - the signed `analytic_G_w_points` for the `plots/Gk-linear/` series goes through the same call
    with `signed=True` (`:156-161`), by the addendum;
  - both unpackings are guarded against an empty list with `else ([], [])` (`:165-167`,
    `:171-173`), the readers' existing idiom;
  - `:299-300` fills `analytic_G_w_column`/`analytic_Gprime_w_column` from `value.analytic_G_w` and
    `value.analytic_Gprime_w`, where it used the `_rad` attributes.
- **`extract_GkSource_data.py`** (`:261-277`) and **`extract_QuadSourceIntegral_data.py`**
  (`:709-725`):
  - the existing `abs_analytic_G_w_points` comprehension is wrapped, unchanged, in
    `sensible_log_points(…, [y for _, y in abs_G_points] + [y for _, y in abs_G_WKB_points])`. The
    reference is that plot's normalised $|G|$, numeric and WKB;
  - the unpacking is guarded (`:350-352`, `:730-732`).
- **Nothing else in the readers changed:** no axis, label, file name or other column.

**New test modules** (32 tests, no Ray, no datastore; each docstring names the defect it guards):
- **`ComputeTargets/tests/test_analytic_gk_domain.py`** (8 tests):
  - **The domain.** `1 + 3w` is exactly `0.0` at `w = -1.0/3.0`, because `3.0 * (-1.0/3.0)` rounds to
    exactly `-1.0`. The predicate is strict. Both functions return `None` at $w = -1/3$,
    $-1/3 - 10^{-12}$, $-1/2$ and $-1$, over five argument sets.
  - **Bit-identity.** It is checked with `struct.pack("<d", …)` against `old_G`/`old_Gprime`, the
    pre-09 expressions written inline from `scipy.special`. The cases are $w = 1/3$, $0$, $0.9$ over
    five argument sets, and $w = -1/3 + 10^{-3}$ (order 667.17) over three sets with $k\tau \ge 1000$,
    where it is finite: 18 cases.
  - **An exact zero is returned**, at $\tau = \tau_s$.
  - **Overflow.** At $w = 0$, $k = 1$, $\tau_s = 10^{-250}$, $\tau = 1$, `yv(1.5, 1e-250)` is `-inf`,
    because $Y_{3/2}(x) \approx -\Gamma(3/2)/\pi\,(2/x)^{3/2} \approx -8\times10^{374}$, beyond the
    largest double. The old expressions give `-inf` and `+inf`; the new ones give `None`.
  - **The radiation form** at $w = 1/3$ equals spec 02's closed form,
    $G = -H_s\sin(k(\tau-\tau_s))/k$ and $dG/dz = (H_s/H)\cos(k(\tau-\tau_s))$. These come from R9
    with DLMF 10.16.1 and spec 02 §0.1 item (2), and are checked over 80 cases. The tolerance is
    $64\,\varepsilon\max(1,k\tau,k\tau_s)$ of the envelope. **Its source:** the phase is rebuilt from
    $k\tau$ and $k\tau_s$ rounded separately. Over this grid the measured worst was $4.75\,\varepsilon$
    (G) and $12.6\,\varepsilon$ (G′) in those units, so 64 is five times the worst measured.
- **`ComputeTargets/tests/test_sensible_log_points.py`** (16 tests):
  - `None`, `nan`, `±inf`, `0` and `-0` are dropped, with no reference and with one;
  - $10^{3.0001}\times$ the reference is dropped and $10^{2.9999}\times$ kept;
  - the reference ignores `None`, `nan`, `inf` and `0`, and `decades` is honoured;
  - an empty and an all-`None` reference keep every finite non-zero point;
  - the A3 v2 values: $|G|$ 9.6e18–4.0e21 keeps 3e21 and drops 1.4e34, 2.1e57 and 7.57e243;
  - negative values come back as their modulus, input order is kept, and `x` is passed through;
  - **the signed form:** a negative value keeps its sign, the boundary is applied to the modulus,
    and it keeps exactly the points the log form keeps.
- **`ComputeTargets/tests/test_analytic_g_w_readers.py`** (8 tests, `ast`):
  - **The classifier.** Each `analytic_G_w`/`analytic_Gprime_w` attribute in the four readers must
    be one of three kinds:
    - *wrapped*: inside a `sensible_log_points` call;
    - *column*: in a statement assigning or extending a `*_column` name;
    - *routed*: in a statement building a list whose every later load is the helper's first
      argument or its own `extend`/`append`.

    Anything else is `BYPASS`.
  - **What it finds now:**
    - GkWKB: 2 routed and 4 column;
    - Gk: 2 wrapped, one of them with `signed=True` and assigned to `analytic_G_w_points`, and 2 column;
    - GkSource: 1 wrapped;
    - QSI: 1 wrapped.
  - **On the readers at `1739e0d`** it flags 6 bypasses: GkWKB `:175`, `:212`; Gk `:147`, `:152`;
    GkSource `:264`; QSI `:712`.
  - **The other tests:**
    - each reader imports the helper;
    - the GkWKB reader's two branches route to one series;
    - no bare `fabs` applies to an `analytic_…` attribute in the GkWKB reader, and exactly six
      `safe_fabs(value.analytic_…)` calls are there;
    - `extract_Gk_data.py`'s `from_dict` maps `"analytic_G_w"` and `"analytic_Gprime_w"` to
      `[value.analytic_G_w for value in values]` and `[value.analytic_Gprime_w for value in values]`.

## Deviations from the prompt

1. **The radiation-form series were not routed through the helper** (§2.2: "may … say which you
   did"). They are plotted as before. In the GkWKB reader they now go through `safe_fabs`, by §2.3.
   - **Considered:** routing them too, for uniformity, which would also keep a `None` from reaching
     matplotlib.
   - **Not chosen**, for three reasons:
     - the addendum says item 5 "still holds for everything else on that plot";
     - nothing has shown the radiation form outside three decades of $|G|$ (in the A3 v2 failing
       pair, `analytic_G_rad` agrees with `G_WKB`);
     - matplotlib draws a `None` in a $y$ sequence as a gap, which I checked on a log axis before
       relying on it.

   A later reader may disagree: §1.2's overflow rule now reaches `analytic_G_rad` too, so a
   radiation `None` would leave a gap, not a crash. **IMPLEMENTATION CHOICE.**
2. **How "§1.4 untouched" was read for `analytic_G_rad`.** The call site is untouched: $w = 1/3$
   literally, in both callers. Its value can now be `None` on overflow, because the rule is in
   `compute_analytic_G`, which it calls. Inside the domain with a finite result it is
   bit-identical (tested). The rehearsal's `39809dca…` and the NULL census below show that no
   stored rehearsal value moved either way. **IMPLEMENTATION CHOICE** (the orchestrator's note).
3. **Line drift.** The prompt cites `extract_Gk_data.py:282-283` for the mislabel; at `1739e0d` it
   was `:286-287`, and is `:299-300` after the edit. The CSV dict was at `:303-304`. Nothing
   followed from it. **STRUCTURALLY REQUIRED.**
4. **The `ast` test is a flow rule, not a lexical one.** The GkWKB plot's $|G|$ combines both
   branches, so its `analytic_w` points are collected raw from both and filtered once, after both,
   against the whole plot's $|G|$.
   - **The alternative**, filtering each branch against its own branch's $|G|$, would make the helper
     call lexically enclose each attribute. It applies a different reference from the one §2.2 names.
   - **Done instead:** the test accepts a *routed* list, whose only later uses are the helper and
     its own `extend`, so a raw list reaching `zip` or `ax.plot` still fails.

   **IMPLEMENTATION CHOICE.**
5. **The `ast` test sits in its own module**, `test_analytic_g_w_readers.py`, not inside
   `test_sensible_log_points.py`. The prompt allowed either ("in the same module or beside
   `test_reader_read_only`"). **IMPLEMENTATION CHOICE.**
6. **Two extra breakage checks.** One is a sixth breakage for the signed form, as the brief asks.
   The other: the classifier was also run on the four readers as they stood at `1739e0d`, and it
   flags all four. **IMPLEMENTATION CHOICE.**
7. **The rehearsal's manifest says `git_dirty: true`** at `git_head` `1739e0d`. The rehearsal had to
   run on 09's edits, before the one commit, as `rpk-v030-adoption` 01's deviation 4. The
   `--purpose` says so. **STRUCTURALLY REQUIRED.**
8. **An extra `bash` between the shell and the launch.** My launch wrapped
   `date … >> launch.txt && nohup sh -c "…" "<purpose>" > … &` in one `bash -c`, so the `&` applied
   to the whole `&&` list.
   - Bash forked a subshell, pid **38829** (reparented to pid 1), to run it. `$!` recorded that
     subshell as "sh pid".
   - The `nohup sh -c` was **38838**, its child, and the Python driver **38841**, the child of 38838.
   - It was harmless: 38829 waited on 38838, which reaped 38841 and printed `** driver exit status
     0`. All three were dead after the run, and the lock and the registry held 38841 throughout.
   - The `date` line was written after the `sh pid` line, by that subshell.

   **UNINTENDED DRIFT**, kept: it changed nothing the run recorded.
9. **The orchestrator's baseline saw a second wall-clock failure**,
   `test_background_tau…test_build_cost`, which passed alone. My before and after runs did not
   fail it. Recorded, not acted on. **IMPLEMENTATION CHOICE** (to record rather than re-run).

## Verification performed

**§3's tests.** The three modules ran 32 tests, OK, before and after each breakage was restored.

**The breakage record.** Each breakage was applied by a script (`run.py` in the scratchpad's
`p09/break/`). The three modules were then run, and the file was restored from a byte copy, checked
equal with `filecmp.cmp(shallow=False)`. After all six, the modules ran 32, OK, and `git status`
showed only 09's edits.

| | Breakage | File | Result | Named tests that failed |
|---|---|---|---|---|
| (a) | the domain test removed from `compute_analytic_G` | `analytic_Gk.py` | 10 failures, 5 errors | `TestTheDomain.test_both_functions_return_none_outside_the_domain`. At $w = -1/2$ and $-1$, five argument sets each, it fails `… is not None` (e.g. `np.float64(-3.777768341413112)`). At $w = -1/3$ it errors with `ZeroDivisionError: float division by zero` in `b`. |
| (b) | widened to `1 + 3w >= 0` | `analytic_Gk.py` | 1 failure, 5 errors | `test_the_predicate_is_the_strict_inequality` (`True is not false`), and `test_both_functions_return_none_outside_the_domain` at $w = -1/3$ (`ZeroDivisionError`). |
| (c) | the helper keeps `inf` (`isnan` for `not isfinite`) | `extract_common.py` | 2 failures | `TestDropped.test_dropped_with_no_reference`, `TestDropped.test_dropped_in_the_signed_form` |
| (d) | one bare `fabs(value.analytic_G_w)` restored, in a CSV column | `extract_GkWKB_data.py` | 2 failures | `test_no_bare_fabs_of_an_analytic_attribute`, `test_every_analytic_attribute_under_a_modulus_uses_safe_fabs` |
| (e) | the CSV mislabel restored (`analytic_G_w_column` from `analytic_G_rad`) | `extract_Gk_data.py` | 1 failure | `test_the_csv_maps_each_w_column_to_its_own_attribute` (subtest `column='analytic_G_w'`) |
| (f) | the signed form's boundary applied to `y`, not `|y|` | `extract_common.py` | 2 failures | `TestTheSignedForm.test_the_boundary_is_applied_to_the_modulus`, `test_the_signed_form_keeps_the_points_the_log_form_keeps` |

- **Under (a), $w = -1/3 - 10^{-12}$ is not caught by its own subtest.** There
  $b \approx -6.7\times10^{11}$, the Bessel products give `nan`, and §1.2's finiteness rule returns
  `None`.
- **(c) is caught only with no reference.** With a reference, the ceiling drops `inf` anyway. That
  is why `TestDropped` tests both.

**Callers that could now receive `None`** (prompt §7's last stop). None can. Each was read:

| Caller | $w$ | Why it stays in the domain |
|---|---|---|
| `convergence_reference.py:490-510` | `wBackground` of a `RadiationModel`, 1/3 | constant $w$ |
| `test_phase_groups.py:266`, `:347`, `:564` | `W_VALUES = (1/3, 0.2)` | in the domain |
| `test_quadsource_integral.py:481` | `w_of_b(b)` with $b \ge 0$ | $1+3w = 2/(1+b) > 0$ |
| `test_background_tau.py:657-658` | 1/3 | in the domain |

Every one passes after the change, so none hits overflow either.

**§4.1, the seven roots**, serially, from the repository root, by `CLAUDE.md`'s commands:

| Root | Before (`1739e0d`, 09:15–09:28 GMT) | After (09's edits, 09:31–09:41) | Change |
|---|---|---|---|
| AdaptiveLevin | 32 OK | 32 OK | 0 |
| ComputeTargets | 700, FAILED (failures=1) | 732, FAILED (failures=1) | **+32**, the three new modules |
| CosmologyModels | 39 OK | 39 OK | 0 |
| LiouvilleGreen | 148 OK (skipped=1) | 148 OK (skipped=1) | 0 |
| ObjectFactories | 280 OK | 280 OK | 0 |
| RunDrivers | 120 OK | 120 OK | 0 |
| tools | 49 OK | 49 OK | 0 |

- **The one ComputeTargets failure, both times,** is the known wall-clock test,
  `test_tk_wkb_phase.TestCost.test_wall_time_per_object`.
  - In the suite: 0.078 s before and 0.072 s after, against limits of 0.05 and 0.06.
  - **Re-run alone:** before, at load 89.7, it failed at 0.084 s against 0.06. After, at load 28.6,
    it failed twice: 0.057 against 0.05, and 0.116 against 0.06.
  - The load averages ran from 11 to 121 during the roots.
  - Not a stop.

**§4.2, rehearsal R5**, under the rehearsal lock. Every lock command was a shell command of its
own, `>> var/handover-09/lock.txt 2>&1`, then `echo $?`, with no pipe.

| When (GMT) | Command | Exit | What it said |
|---|---|---|---|
| 09:41:52 | `acquire --project SecondaryGWKit --purpose "handover 09 rehearsal (analytic_G_w outside its domain), dispatch 1739e0d with 09's edits uncommitted; must fingerprint 39809dca" --ray-temp-dir /tmp/ray-sgk --ray-address in-process` | 0 | acquired; owner file `pid: -` |
| 09:41 | `ray-processes --ray-temp-dir /tmp/ray-sgk` | 0 | none |
| 09:43 | `set-pid … 38841` | 0 | recorded |
| 09:43 | `ray-processes …` | 1 | 19 processes, children of 38841 or of its raylet 38917: `gcs_server` 38881, `raylet` 38917, `ray::SerialPoolBroker`, `ray::Datastore` ×4, `ray::numeric_with_phase_cut` ×6, and Python helpers |
| *(after the readers)* | `ray-processes`, `release`, `status` | see "The close" | |

- **Pre-launch** (`prelaunch.txt`, 09:31:48):
  - `HEAD` was `1739e0d`, with 09's edits only, and one worktree;
  - `RAY_ADDRESS` was unset; 10 CPUs;
  - `runregistrykit list` showed 52 runs (50 finished, 2 unknown), none `running` or stale;
  - `store show var/datastores/handover-09-reference.sqlite` gave kind `absent` and no run naming
    it, and nothing under `var/datastores` or `var/runs` held `handover-09`;
  - neither lock location existed, and there was no journal;
  - the three packages were at their pins, not editable;
  - R3's `var/runs/rrk-v020-adoption-03-reference-20261018T232838` was present.
- **The launch**, at 09:42:11, load 27.70 38.40 56.10. It was R3 §1.1's form:
  ```
  nohup sh -c "PYTHONPATH=. ./venv/bin/python -u RunDrivers/quadsource_atol_sweep.py --build --rehearsal
    --database var/datastores/handover-09-reference.sqlite --job-name handover-09 --atol 1e-32 --rtol 1e-7
    --cpus 6 --register r5-reference --purpose \"\$0\" --campaign handover --prompt 09;
    echo \"** driver exit status \$?\"" "<purpose>" > var/runs/handover-09-reference.launch.log 2>&1 &
  ```
  - It had the extra wrapping of deviation 8.
  - The purpose names this prompt, `1739e0d` with 09's edits uncommitted, the three package
    commits, `39809dca` and R3's `reference`.
  - The driver echoed main.py's argv:
    `--database …handover-09-reference.sqlite --job-name handover-09 --shards 4 --zend 250
    --source-samples-log10z 100`, with "rehearsal scope, NOT A3 v2: k = [1e5, 3.16e5, 1e6], zend
    250" and four `replaced 1 occurrence` lines.
- **The one check**, once launch-log line 12 said "Started a local Ray instance." (09:42:27):
  - **The manifest:**
    - run id `handover-09-r5-reference-20261025T094215`;
    - `git_head` `1739e0d…`, `git_dirty` true;
    - `script` `RunDrivers/quadsource_atol_sweep.py`, `script_sha256` `5b984c75…`, the same as
      `rpk-v030-adoption` 01's;
    - `packages` at `datastorekit` `f47b9a25…`, `raypoolkit` `67a75b97…`, `runregistrykit`
      `0bae3a3c…`, none editable;
    - `host` this machine.
  - **`status.json`:** `running`, pid 38841, which answered `kill -0`; host `id` and `boot` equal to
    `this_host()`.
  - **The registry:** `list` showed it `running`.

  I then ended my turn. The orchestrator waited on pid 38841 and resumed me. Neither of us polled.
- **The final state:**
  - `status.json` says `done`, `exit_code` 0, stage "CALCULATE QUADRATIC SOURCE INTEGRALS, all work
    items complete", with the last heartbeat at 10:01:10;
  - the launch log ends `** driver exit status 0`;
  - 38841, 38838 and 38829 are dead;
  - no `Traceback`, `DeprecationWarning` or `ConfigError` is in the launch log, `stdout.log` or
    `stderr.log`.

  Wall time was **18 m 55 s** (09:42:15–10:01:10), against `rpk-v030-adoption` 01's 14 m 36 s.
- **The fingerprint** (`sidecar.txt`, `compare.txt`):
  - The run's own fingerprint, taken 10:01:06 at `1739e0d` with `git_dirty` true, was
    **`39809dcac5c7b231c6b4468f9ea6cc48a4ad9c3a52d469f18f43291f781b55d2`**: format 6, 21 classes,
    `problems {}`. The counts were GkWKBIntegration 3477, GkNumericIntegration 1368, GkSource 291 and
    QuadSourceIntegral 864.
  - `store create` wrote sidecar `store_id` `a30dd6ae91c64f228f4e24672b923c44`.
  - `store fingerprint --write` gave the same digest, "problems: none", "it replaced none".
  - `compare_with_recorded.py <store> --against var/runs/rrk-v020-adoption-03-reference-20261018T232838
    var/runs/handover-09-r5-reference-20261025T094215` exited 0, verdict **`equal`**. There were 0
    differences against the sidecar, against R3's `reference` (`aab4075`, `git_dirty` false) and
    against this run's.
  - **So §1 changed no stored value at `--zend 250`.**
- **The NULL census** (`null-census.txt`; `sqlite3` with `mode=ro` on each shard). It found no `NULL`
  in `analytic_G_w` or `analytic_Gprime_w` in any row of `GkWKBValue` (147,642 rows) or
  `GkNumericValue` (35,874). The largest `|analytic_G_w|` was 4.87e19 (WKB) and 1.02e17 (numeric).
  Shard 2 holds neither table's rows.
- **The store's five files** had the same SHA-256, mtime and size before `store create`, after the
  comparison and after the census, with no journal (`state-before-sidecar.txt`,
  `state-after-compare.txt`).

**§4.3, the readers.** `bash docs/dk-and-rpk-migration/rehearsal/run_readers.sh
var/datastores/handover-09-reference.sqlite var/handover-09/extract` was run, with the lock held.

The script exited 0 (`readers.txt`):

| Reader | Started (GMT) | Exit | Wall | Files written |
|---|---|---|---|---|
| `extract_Gk_data` | 10:02:46 | 0 | 72 s | 468 |
| `extract_GkWKB_data` | 10:04:02 | 0 | 241 s | 2436 |
| `extract_GkSource_data` | 10:08:06 | 0 | 74 s | 190 |
| `extract_TkWKB_data` | 10:09:23 | 0 | 33 s | 27 |
| `extract_tensor_source_data` | 10:09:58 | 0 | 52 s | 27 |
| `extract_QuadSourceIntegral_data` | 10:10:52 | 0 | 46 s | 70 |

- **The store was unchanged by every reader**: the same SHA-256, mtime and size before and after
  each, with no journal. `state.sh` afterwards still equals `state-before-sidecar.txt`.
- No reader log holds `Traceback` or `Error`, and no Ray process of this venv was left after any
  reader.
- **Each reader used `/tmp/ray-sgk`**, which holds seven sessions: the build's (`…_38841`) and one per
  reader.
- **The CSV columns** (`gk-csv-check.txt`): in all 52 `extract_Gk_data` CSVs, `analytic_G_w` differs
  from `analytic_G_rad` somewhere. Before 09 the two were identical by construction. For example,
  at one $z$: `G` 7.967763e+08, `analytic_G_rad` 7.967653e+08, `analytic_G_w` 7.968229e+08.
- The one-minute load reached 132 during the readers.

**The close**, each command its own shell call:
- `ray-processes` exited 0 and listed none.
- `release` exited 0: "released the rehearsal lock".
- `status` exited 0: "lock: free (… does not exist)", "Ray processes of /tmp/ray-sgk: none".
- The lock was free at 10:12:22 GMT.

**§4.4.** `black --check` is clean on all nine touched `.py` files: the six edited and the three new.

## Observations not acted on

- **`extract_GkSource_data.py` and `extract_QuadSourceIntegral_data.py` plot the radiation form twice.**
  - Each draws "Analytic $G_k$ [$w=w(z)$]" from `abs_analytic_G_rad_x`/`_y`
    (`extract_GkSource_data.py` `plot_Gk`; `extract_QuadSourceIntegral_data.py` `plot_Gk`, written as
    `GkSource.pdf`).
  - Their `abs_analytic_G_w_points` are used only in the plot's `if`.
  - In any case `GkSource`'s table stores no `analytic_G_w` column (`ObjectFactories/GkSource.py`
    has only the two `_rad` columns), so a `GkSource` read back from a store has
    `analytic_G_w = None`.
  - This is out of scope by `CLAUDE.md` rule 4 and §2.5 ("no … label"). §2.2's acceptance holds
    without it, since those points do go through the helper.
  - Opened as `[09-two-readers-plot-the-radiation-form-under-the-w-label]`.
- **`numpy` raises `RuntimeWarning: invalid value encountered`** when the Bessel products give `nan`
  inside the domain, for example $w = -1/3 + 10^{-3}$ at small $k\tau$. The value is then `None`. The
  warning is numpy's, and left as it is.
- **`extract_Gk_data.py` writes `redshift` objects into its CSV's `z_response` column.** The column
  is `[value.z for value in values]` (`:294`), and `value.z` is a `CosmologyConcepts.redshift`
  object, so every cell reads `<CosmologyConcepts.redshift.redshift object at 0x…>`. The CSV then
  cannot be read back against $z$, and its `sort_values` orders by the objects' own comparison. The
  plots in the same function pass the same objects to matplotlib as $x$. This was seen in passing,
  while checking the `analytic_G_w` columns, and is out of scope. Opened as
  `[09-extract-gk-data-writes-redshift-objects-into-its-z-response-column]`.
- **The reader file counts are not comparable between runs.** `extract_Gk_data.py:416-432` and
  `extract_GkWKB_data.py:459-472` take `random.sample` subsets of $k$ and $z$.

## State handed to the next step

- **For prompt 10** (the fixed readers on A3 v2, on the Mac Studio): the tree is `1c19998`.
  - On A3 v2, `extract_GkWKB_data.py` will drop the `analytic_G_w` points more than three decades
    above $|G|$, including all 293 above 1e200, from its full-range plots.
  - `extract_Gk_data.py`'s CSVs now carry the stored $w(z)$ values, astronomically large ones
    included. CSVs are not filtered.
- **New stores** hold `NULL` in `GkWKBValue`/`GkNumericValue` `analytic_G_w` and
  `analytic_Gprime_w` wherever $1+3w \le 0$ or the form is not finite. Their fingerprints differ
  from a pre-09 store's exactly there. `--zend 250` stores are unaffected (`39809dca…`).
- **Left in place, for the user to retire:** `var/datastores/handover-09-reference.sqlite` (four
  shards, sidecar `a30dd6ae…`), `var/runs/handover-09-r5-reference-20261025T094215`,
  `var/runs/handover-09-reference.launch.log` and `var/handover-09/`.
