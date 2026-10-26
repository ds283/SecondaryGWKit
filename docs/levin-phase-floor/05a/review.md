# levin-phase-floor 05a, Part B — the Mac Studio orchestrator's review (§B.3)

**Host:** mps026298 (the Mac Studio). **Date:** 2026-10-26, 12:17–12:55 UTC.
**Reviewer:** the Mac Studio orchestrator (Claude Opus 5.5), on the orchestrator note's §B and its
addendum §B.5. Nothing was committed and no git command that writes was run, in either tree.

**Verdict: PASS.** All ten checks hold. No condition of prompt §8 fired.
- **T1, as §9 is written** (the halving rule read literally, as `state/predictions.md` fixed
  before the run), **fails**: 36 of 70 scored windows, all on the $t$ clause.
- **Read by the true step, T1 holds in 70 of 70.** That reading was reported apart and is unscored.
- **Which reading is §9's rule** is a question the re-measurement agent raised after the launch. It
  was relayed to the user verbatim, and is **unanswered**.
- **T2 and T3 hold. T4 fails**, on the same two rows and ratios as 05's L3 under $c$.
- By §B.4's closing rule, T1 or T4 failing means **the user is asked** before either header issue
  is closed.

The scratch readers used below are reproduced in full at the end of this file. They ran from the
session scratchpad (`…/scratchpad/review/`), and wrote their outputs there.

---

## Gate (before dispatch, 12:17 UTC)

Re-read the state §B.5 records; did not fetch. Every item matched.

```
main HEAD 149c3938bb9e074df3682f8e511c7b57cd9c55c5, git status --short: empty
worktree list: SecondaryGWKit 149c3938 [handover-remedial]; SecondaryGWKit-lpf05a-c4eaefd c4eaefda (detached HEAD)
worktree HEAD c4eaefda26ccb615fa5864cac877454ba364c304, git status --short: empty; no venv in it
runregistrykit list: 16 run(s) … 16 finished; 0 with no host recorded
rehearsal_lock status --ray-temp-dir /tmp/ray-sgk: lock: free (… does not exist); Ray processes of /tmp/ray-sgk: none; exit 0
~/.raypoolkit/ empty; ~/ScienceData/.rehearsal-lock: No such file or directory
ls var/levin-phase-floor-05a: No such file or directory; ls var/datastores | grep 05a: nothing
05 sample.json: 73aa7876469a59121782de31c46b5fa9224857eec2799d7ee6611ba918d833f7, 26284 bytes; 05 wider run dir: 122 files
store show A3 v2: digest 0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56, problems {}
~/.ray/auth_token -rw-------; 118G free; load 4.54 5.58 5.08
```

The marker was touched at 2026-10-26 12:18:26.787 UTC, immediately before the two dispatches.

## Dispatch and the waits

- **Two fresh-context Opus agents.** The rehearsal agent took prompt §5 and the re-measurement
  agent prompt §6 with §9. Each was given:
  - the prompt, the dispatch SHA `149c393` and the worktree path;
  - §B.1 and §B.5, copied verbatim into a scratchpad file (not `orchestrator/`);
  - everything §B.2 says to tell it.
- **Each pid was waited on once, in the background** (`while kill -0 <pid>; do sleep …; done`),
  and its agent was resumed with the final `status.json`:
  - **R11:** driver 22684, ended 12:23:42; `done`, exit 0, `1b332ffa…`.
  - **R12:** driver 24897, ended 12:27:30; `done`, exit 0, `1b332ffa…`.
  - **`wider`:** driver 28746, ended 12:32:32; `done`, 109/109, exit 0.
- **The agent question, relayed to the user verbatim and not answered by the orchestrator.** The
  re-measurement agent's launch report raised it: does §9's halving rule ("largest step of the
  unwrapped samples < π/2") mean `max|diff(unwrap(θ))|`, which admits aliased windows, or the true
  step? It is unanswered at this review.

---

## Check 1 — Scope: holds

```
$ git rev-parse HEAD; git status --short                     (main)
149c3938bb9e074df3682f8e511c7b57cd9c55c5
(status lines: 0)
$ git -C <worktree> rev-parse HEAD; git -C <worktree> status --short --ignored
c4eaefda26ccb615fa5864cac877454ba364c304
(status --ignored lines: 0)
$ find var -newer <marker>          (grouped to three path levels)
var/datastores/levin-phase-floor-05a-r11{,-shard0000..0003}.sqlite   (5)
var/datastores/levin-phase-floor-05a-r12{,-shard0000..0003}.sqlite   (5)
var/levin-phase-floor-05a/ ; …/rehearsal (3251 entries) ; …/wider (91 entries)
var/runs/ ; var/runs/levin-phase-floor-05a-r11-parent-20261026T122041 (5) ; var/runs/levin-phase-floor-05a-r11.launch.log
var/runs/levin-phase-floor-05a-r12-reanchor-t-20261026T122434 (5) ; var/runs/levin-phase-floor-05a-r12.launch.log
var/runs/levin-phase-floor-05a-wider-20261026T123047 (123)
(plus the directory entries var/ and var/datastores/ themselves)
$ find <worktree> -newer <marker> -not -path '*/__pycache__/*'
(nothing)        ; find <worktree> -name __pycache__: nothing
```

Every path is one of §B.2's: the records, the two stores, their run directories and launch logs,
and the `wider` run directory. The worktree is unwritten.

**A3 v2's six files**, hashed now, equal the re-measurement agent's `state/store_files_before.txt`
(taken 12:24:04Z) file by file, sidecar included:
- `1542e05c…` `.sqlite`;
- `b311698f…`, `c5e341ce…`, `175a6731…`, `76b83fb3…` (the shards);
- `3201f6a5…` `.manifest.json`.

There is no `-wal`, `-shm` or `-journal` file in `var/datastores/`. `find var/levin-phase-floor-05
var/runs/levin-phase-floor-05-wider-20261026T020005 -newer <marker>` lists nothing, and 05's run
directory still holds 122 files.

## Check 2 — The runs: holds

```
$ ./venv/bin/python -m runregistrykit list
   levin-phase-floor-05a-wider-20261026T123047           done        109/109
   levin-phase-floor-05a-r12-reanchor-t-20261026T122434  done              -
   levin-phase-floor-05a-r11-parent-20261026T122041      done              -
19 run(s) … 19 finished; 0 with no host recorded.
```

| run | campaign | prompt | slug (in the run id) | git_head | git_dirty | cwd / script |
|---|---|---|---|---|---|---|
| R11 | levin-phase-floor | 05a | `r11-parent` | `c4eaefda…` (the parent) | false | worktree; worktree's launcher `5b984c75…` |
| R12 | levin-phase-floor | 05a | `r12-reanchor-t` | `149c3938…` (dispatch SHA) | false | main checkout; `5b984c75…` |
| wider | levin-phase-floor | 05a | `wider` | `149c3938…` | false | `var/levin-phase-floor-05a/wider`; `run_wider.py` `af656ad4…` |

R11's and R12's argv carry `--build --rehearsal --database <absolute> --job-name
levin-phase-floor-05a --atol 1e-32 --rtol 1e-7 --cpus 6 --register …`.

## Check 3 — The fingerprints: holds

Both `status.json` record `1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7`,
`problems {}`.

```
$ ./venv/bin/python -m runregistrykit store fingerprint var/datastores/levin-phase-floor-05a-r11.sqlite   # exit 0
digest:   1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7
  QuadSourceIntegral        864  8913eda5ce7b
problems: none
$ ./venv/bin/python -m runregistrykit store fingerprint var/datastores/levin-phase-floor-05a-r12.sqlite   # exit 0
digest:   1b332ffa948a3dbdfd51cf81af2d281ac9c3ba5440f994209f1dbc775dce8df7
problems: none
$ diff fp_r11.json fp_r12.json      # the two outputs differ only in the line naming the store
```

Neither was run with `--write`.

## Check 4 — The values: holds

```
$ PYTHONPATH=. ./venv/bin/python -m tools.store_value_diff var/datastores/levin-phase-floor-05a-r11.sqlite \
      var/datastores/levin-phase-floor-05a-r12.sqlite --json > <scratch>/svd.json
store_value_diff exit 1
c9d68587e0c5ed90d7056416b15eea5fdc9e469009e9b1dcaa0d16fb184ac07c  <scratch>/svd.json
c9d68587e0c5ed90d7056416b15eea5fdc9e469009e9b1dcaa0d16fb184ac07c  rehearsal/out/store_value_diff.json
BYTE-IDENTICAL to saved JSON
$ ./venv/bin/python -I rehearsal/scripts/classify_non_qsi.py <scratch>/svd.json > <scratch>/classify.txt
classify exit 0 ; identical to rehearsal/out/classify_non_qsi.txt
UNCLASSIFIED differences or structural problems outside QuadSourceIntegral: 0
```

- **Every class** (20 outside `QuadSourceIntegral`) reports only-A / only-B / unmatched /
  unresolvable `(0, 0, 0, 0)`, and "value tables not identical none".
- **The differing columns** are `timing`, `label`, `parents_digest` and `rho_reuse` only. They
  fall in wavenumber_exit_time, BackgroundModel, TkNumericIntegration, TkWKBIntegration,
  GkNumericIntegration, GkWKBIntegration, GkSource and QuadSource.
- **The copied scripts** have the same SHA-256 as 05's: `classify_non_qsi.py` `26690e61…` and
  `tabulate_qsi.py` `b7508351…` (`rehearsal/out/script_sha256.txt`).

**The tabulation** (`rehearsal/out/qsi_tabulation.txt`):
- 864 = 864 rows, with 0 `total` mismatches against the tool;
- `total` differs in 424 rows;
- the largest $|\Delta|/|{\rm total}|$ is 2.924e-11, and the largest $|\Delta|/$abserr 8.810e-5;
- no transition in converged, phase_limited, max_depth or region count;
- `numeric_quad` and `analytic_rad` are identical in 864/864.

**The spot-check,** by a scratch reader opening each shard with `sqlite3` `mode=ro` (`spot_qsi.py`,
below), on the tabulation's row of largest relative change:

```
tabulation row: k 1e6, q 316227.7660168379, r 1e6, z_response 1329131612.3012946,
                total_A -0x1.9a14da7f27de2p-61, total_B -0x1.9a14da7f5b5e2p-61, abserr_A 1.2091690228282147e-21, abserr_B 1.2091690296636564e-21
r11 shard1: total -0x1.9a14da7f27de2p-61 (-6.947050136539057e-19)  abserr 0x1.6d72ebf643244p-70   (1 row)
r12 shard1: total -0x1.9a14da7f5b5e2p-61 (-6.9470501367421885e-19)  abserr 0x1.6d72ec18ec0d6p-70   (1 row)
float.hex of the tabulation's abserrs: 0x1.6d72ebf643244p-70 0x1.6d72ec18ec0d6p-70
```

R11 and R12 are not value-identical in `QuadSourceIntegral`, so each tree ran its own code.

## Check 5 — The readers and the lock: holds

**The readers.** `rehearsal/out/readers.txt` records all six readers with `exit 0` (28, 57, 20,
15, 14 and 17 s), and `run_readers.sh exit 0`.

**R12's five files** have the same SHA-256 before the readers (12:29:42Z), after them (12:32:37Z)
and now:
- `b42aa513…`, `a5d54bbb…`, `af01699e…`, `5b249c85…` (the shards);
- `94dae1cc…` (the primary).

The before/after files differ only in their timestamp line, and now differs from them only in the
path prefix. There is no journal file.

**The lock transcript** (`rehearsal/out/lock.log`):

| command | exit |
|---|---|
| `acquire` | 0 |
| `set-pid 22684` (R11's driver) | 0 |
| `ray-processes` | 0, none |
| `set-pid 24897` (R12's driver) | 0 |
| `ray-processes` | 0, none |
| `release` ("released the rehearsal lock") | 0 |
| `status` (free) | 0 |
| `ray-processes` | 0, none |

Run by me, each as a command of its own:

```
$ ./venv/bin/python -m raypoolkit.rehearsal_lock ray-processes --ray-temp-dir /tmp/ray-sgk
exit 0                                   (nothing listed)
$ ./venv/bin/python -m raypoolkit.rehearsal_lock status --ray-temp-dir /tmp/ray-sgk
lock: free (/Users/ds283/.raypoolkit/rehearsal-lock does not exist)
Ray processes of /tmp/ray-sgk: none
exit 0
```

## Check 6 — T0 and the store: holds

**T0 in the saved output** (`wider/out/T0.txt`, written 12:28:33.95Z; the `wider` manifest is
12:30:47.58Z):
- **(a)** `wider/out/cmd/05_T0_clause1_git_diff.txt` gives `CLAUDE.md | 8 +++++---`, 1 file
  changed.
- **(b)** on `low-G+r-1` and `ctrl-low-1`, compared with `==` and `float.hex`: `total`,
  `total_abserr` and "the clause's own fields (16 doubles): all bit-identical". "T0 (b): HOLDS".
- **(c)** two in-process $t$ runs on both rows: result, calls, regions and 31,200 and 31,070
  probe samples identical. "T0 (c): HOLDS".

**Independently,** by a scratch reader of the two `checkpoint.jsonl` (`c_vs_05c.py`, below). It
decodes each recorded `{repr, hex}`, checks that the two agree, and compares bit patterns. On the
depth-20 row R4:

```
== c-1e-7-R4: 05a role d20, tree_head c4eaefda…; 05's tree_head 53084dcb…
   total              05a -0x1.5c2dd6c0d2a10p-85  05 -0x1.5c2dd6c0d2a10p-85  ==
   total_abserr       05a 0x1.b3fc38b16dfe8p-97  05 0x1.b3fc38b16dfe8p-97  ==
   call[0].value      05a -0x1.4efaafd89c8f5p-83  05 -0x1.4efaafd89c8f5p-83  ==
   call[0].abserr     05a 0x1.a3c267b920000p-105  05 0x1.a3c267b920000p-105  ==
   call[1].value      05a 0x1.c9bb3f1b3cca8p-83  05 0x1.c9bb3f1b3cca8p-83  ==
   call[1].abserr     05a 0x1.093d8cba00000p-105  05 0x1.093d8cba00000p-105  ==
   call[2].value      05a -0x1.035251e548504p-84  05 -0x1.035251e548504p-84  ==
   call[2].abserr     05a 0x1.0e5d3f5178000p-106  05 0x1.0e5d3f5178000p-106  ==
   call[3].value      05a -0x1.58bf4f37c4aecp-86  05 -0x1.58bf4f37c4aecp-86  ==
   call[3].abserr     05a 0x1.4322d94250b33p-102  05 0x1.4322d94250b33p-102  ==
   call[4].value      05a 0x1.cc3e6e0c81bd4p-88  05 0x1.cc3e6e0c81bd4p-88  ==
   call[4].abserr     05a 0x1.6a4534202e821p-104  05 0x1.6a4534202e821p-104  ==
   call[5].value      05a 0x1.606531e7d8fd4p-86  05 0x1.606531e7d8fd4p-86  ==
   call[5].abserr     05a 0x1.28c29a235d729p-102  05 0x1.28c29a235d729p-102  ==
   call[6].value      05a -0x1.c1c5a5d9ef723p-88  05 -0x1.c1c5a5d9ef723p-88  ==
   call[6].abserr     05a 0x1.6293de6010717p-104  05 0x1.6293de6010717p-104  ==
c-1e-7 units compared: 39; not bit-identical in total/abserr/every call value+abserr: 0
```

So all 39 of 05a's $c$ units at `1e-7` equal 05's $c$ units bit for bit, not only T0's two.

**The store:**

```
$ ./venv/bin/python -m runregistrykit store fingerprint var/datastores/handover-A3-v2-lambdacdm.sqlite   # exit 0, no --write
digest:   0e32f24a2dc2f4c9ac34f7522f9c228da790096c068b8c3109d58416f8b21d56
problems: none
```

- **The six files** are hashed identical before and after my fingerprint, and identical to the
  agent's before-hashes (check 1). The agent's `store_files_{before,prelaunch,after,after_fingerprint}.txt`
  agree in SHA-256, size and mtime.
- **No journal, `-wal` or `-shm` file** is beside any store file.

## Check 7 — The trees, read: holds

**The launch.** It is `env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 …/venv/bin/python -B
scripts/launch.py` (`wider/state/launch.txt`). `run_wider.py` sets each worker's `PYTHONPATH` to
its own tree (`env["PYTHONPATH"] = str(TREES[tree_name][0])`).

**The $c$ worker** (`worker_c_hello.json`, pid 28750):
- `PYTHONPATH` is `/Users/ds283/Documents/Code/SecondaryGWKit-lpf05a-c4eaefd`;
- `head` is `c4eaefda…`, `status_short` empty, `all_in_tree` true;
- `ComputeTargets.TkSourceFunctions` is `…/SecondaryGWKit-lpf05a-c4eaefd/ComputeTargets/TkSourceFunctions.py`;
- `TkSourceFunctions_exact_remainder_default` is `absent`.

**The $t$ worker** (`worker_t_hello.json`, pid 29357):
- `PYTHONPATH` is `/Users/ds283/Documents/Code/SecondaryGWKit`;
- `head` is `149c3938…`, `status_short` empty, `all_in_tree` true;
- `ComputeTargets.TkSourceFunctions` is `/Users/ds283/Documents/Code/SecondaryGWKit/ComputeTargets/TkSourceFunctions.py`;
- the default is `True`.

**Every one of the 109 checkpoint records**, grouped by tree:

```
37 ('c', 'c4eaefd', status empty, "{'Tq': False, 'Tr': False}", ['G exact_remainder=True', 'T exact_remainder=False'])
 2 ('c', 'c4eaefd', status empty, "{'Tq': False, 'Tr': False}", ['T exact_remainder=False'])
68 ('t', '149c393', status empty, "{'Tq': True, 'Tr': True}",   ['G exact_remainder=True', 'T exact_remainder=True'])
 2 ('t', '149c393', status empty, "{'Tq': True, 'Tr': True}",   ['T exact_remainder=True'])
```

Under $t$, $T$'s phase objects report `exact_remainder` true; under $c$, false.

## Check 8 — The sample and the predictions: holds

```
$ shasum -a 256 wider/state/sample.json
73aa7876469a59121782de31c46b5fa9224857eec2799d7ee6611ba918d833f7
$ cmp wider/state/sample.json var/levin-phase-floor-05/wider/state/sample.json     # identical
$ ls -l --time-style=full-iso
2026-10-26 12:28:33.951 wider/out/T0.txt
2026-10-26 12:29:30.091 wider/state/predictions_attempt1_wrong_window_lines.md
2026-10-26 12:30:03.204 wider/state/predictions.md
2026-10-26 12:30:47.576 var/runs/levin-phase-floor-05a-wider-20261026T123047/manifest.json
```

So `predictions.md` is later than T0 and earlier than the manifest. Its first write, the attempt
file, was also between them; it is the agent's deviation 9, classified `UNINTENDED DRIFT`.

**Word for word, with §9's T1** (§B.5's note on check 8). The prompt is unchanged since `149c393`
(`git diff --quiet 149c393 -- <prompt>`). Each block of the prompt was searched for verbatim in
`predictions.md`:

```
heading 297: verbatim
§9 T1 401-410: verbatim
§6 u-clause 303-305: verbatim
§6 T2..descriptive 306-328: verbatim
§9 window 388-399: verbatim
raw §6 = prompt 297-328 ; raw §9 T1 = prompt 401-410
```

§6's T1 (lines 298–302) is replaced by §9's T1, and `predictions.md` says so. Its "Readings fixed
before the run" add readings and change no rule; the halving rule's literal reading is one of
them.

**The harness diff against 05's copy.** 05's `var/` and `docs/` copies are `diff`-identical. The
saved `wider/out/scripts_diff_against_05_final.txt` reproduces against the scripts as they stand
(timestamps aside). By file:

| status | files | deviation |
|---|---|---|
| unchanged | `dispatch.py`, `harness.py`, `rebuild.py`, `hash_store.py`, `draw_sample.py`, `l0.py` | 15 (`draw_sample.py`, `l0.py`) |
| changed | `prov.py`, `run_wider.py`, `launch.py` | 2 |
| changed | `worker.py` | 2, 4, 7 |
| changed | `rec.sh` | 2, 8 |
| changed | `analyse.py` | 11 |
| new | `probe.py` | 4 |
| new | `t0.py` | 6 |
| new | `rescore_t1.py` | 12 |
| new | `convergence_kinds.py` | 13 (`cmp`-identical to `prompts/levin-phase-floor/measure/`, `49c83a72…`) |

The budget is raised from 05's unchanged `dispatch.py`. Every checkpoint record carries one set of
run-time script hashes, and those hashes equal the scripts now. `analyse.py` and `rescore_t1.py`
came after the run and are not among them.

## Check 9 — The verdicts, re-scored: holds (the scoring agrees with mine everywhere)

**T1, from `wider/state/probe_windows.json`, by `rescore_t1_orch.py` (below).**
- **The levels.** The level is chosen from each window's saved halving ladder (both trees), under
  each reading in turn.
- **σ_θ is recomputed** from the saved hex samples at that level, by test 3's definition:
  `numpy.unwrap`, a degree-4 fit in $t\in[-1,1]$, and the RMS residual. The samples were checked
  to be `linspace(a, b, 65)`.
- **$\max|p_z|$ is taken** from the saved `p_z_t`.
- **The bound** is $\max(10^{-3}\varepsilon|\theta_T|,\ \varepsilon(k\max|p_z|+2\pi),\ 10^{-13})$.

```
== T1 under the literal reading: windows with eps|theta| >= 1e-11: 70; resolved 70; unresolved 0
   c clause fails in 0; sigma_c/(eps|theta|) 0.196..0.374
   t clause holds in 34, fails in 36
     holding: sigma_t/(eps|theta|) max 1.31e-05; sigma_t/bound max 0.0131
     failing: sigma_t/(eps|theta|) 0.0159..0.232; sigma_t/bound up to 232
     failing windows' true step per sample (rad): 4.94..352
   pz term largest in 0 windows
   level/sigma mismatches with the agent's record: 0 []
   VERDICT (literal): fails
== T1 under the true_step reading: windows with eps|theta| >= 1e-11: 70; resolved 70; unresolved 0
   c clause fails in 0; sigma_c/(eps|theta|) 0.198..0.374
   t clause holds in 70, fails in 0
     holding: sigma_t/(eps|theta|) max 1.31e-05; sigma_t/bound max 0.0131
   pz term largest in 0 windows
   level/sigma mismatches with the agent's record: 0 []
   VERDICT (true_step): holds
windows total 91, rows 37
```

**T1 as written: fails** (36 of 70 windows, $t$ clause). The 36 are the windows the literal rule
accepts while the phase steps 4.94–352 rad per sample. **By the true step: holds** (70 of 70).
**Unresolved: 0**, so neither reading is undecided. **The $p_z$ term is the largest in 0
windows.** The user's ruling on the reading is pending.

**The $u$-sampled σ under $t$** (descriptive), by `u_sampled.py`:

```
literal: sigma_u_t > 0.1 eps|theta| in 27 of 70; there sigma_u/(eps|theta|) 0.102..2.2, abscissa/(eps|theta|) 0.161..0.978; sigma_u_t/abscissa 0.136..2.84
true_step: sigma_u_t > 0.1 eps|theta| in 14 of 70; there sigma_u/(eps|theta|) 0.103..0.216, abscissa/(eps|theta|) 0.687..0.978; sigma_u_t/abscissa 0.136..0.288
```

The agent's log reports the true-step level (14 windows) and names it so, with the literal level's
27 beside it. The final report it sent the orchestrator gave the 14 without naming the level. The
issue the $u$-clause asks for is proposed (X2).

**T2, T3 and T4, from the run's `checkpoint.jsonl`, by `rescore_t234.py` (below).** Exact
inequalities use `fractions.Fraction`; an infinite `total_abserr` makes an inequality true. The 11
rows capped under B1 are read from 05's run (its $p$ at `1e-7` reaching depth 20).

```
statuses: Counter({'done': 109}) roles at c-1e-7: Counter({'d20': 31, 'control': 8})
rows 39, depth-20 rows 31, capped under B1 (05's p at depth 20) 11: ['R1','R3','R4','R5','R6','R7','low-G+r-3','low-G-1','low-G-3','low-G-r-2','low-G-r-3']
units not done: []
T2: false on 0 of 39 rows []; largest depth under t@1e-7 11 -> holds
T3: false on 0 of 39 []; largest |t-c|/abserr_c 4.66e-06 (low-G-1); largest relative 3.64e-12 (R4); bit-identical totals 13 -> holds
T4: scored 31 of 31; true on 29; false on [('mid-G-3', 1.698), ('mid-G-r-3', 1.294)]; capped rows largest ratio 0.166 -> fails
05's L3 under c false on: [('mid-G-3', 1.698), ('mid-G-r-3', 1.294)]
budgeted units 70; largest wall/budget ('t-1e-8-R3', 0.8610272803241554)
t units whose budget != 3 x c wall: [] ; c units with a budget: []
```

- **The over-budget units:** none. Every $t$ unit's budget is exactly 3× its row's $c$ wall.
- **The descriptive items are present:**
  - $|t-c|$ per row, and regions and depth per group with `Gqr` and the low-$z$ end apart, in
    `wider/out/wider_05a.txt` and the log;
  - `total_converged` under both trees;
  - the $u$-sampled σ;
  - §9's unscored counts;
  - `convergence_kinds.py`'s output at both tolerances (`wider/out/convergence_kinds_t.txt`,
    "tree t at rtol 1e-7" and "1e-8"; capped run 05's).

## Check 10 — The records: holds

**`rehearsal/log.md`** has README §5.1's sections: subject, commit ("none") and result; What
shipped; Predictions and verdicts; Deviations; Verification performed; Observations not acted on;
State handed on. Its 10 deviations are each classified. Deviation 1 is the worktree (`STRUCTURALLY
REQUIRED`); deviation 9 (`gate.txt` cannot show the records directory absent) is `UNINTENDED
DRIFT`. It records no failed attempt.

**`wider/log.md`** has the same sections. Its 15 deviations are each classified:
- deviation 1 is the worktree (`STRUCTURALLY REQUIRED`);
- deviation 9, the first `predictions.md`, is the only `UNINTENDED DRIFT`;
- deviation 5, the literal reading, is an `IMPLEMENTATION CHOICE` and says the question went to
  the user unanswered.

Its Predictions and verdicts keep §6's T1 marked "Replaced before the run by §9", with §9's T1
after it.

**Both `verification.md`** are written as §5a subsections ("5a.x"). Every link is relative to
`docs/levin-phase-floor/`, in the `05a/rehearsal/…` and `05a/wider/…` form, and every link target
exists under `var/levin-phase-floor-05a/`. **Both `issues.md` exist.** The rehearsal agent's opens
nothing new and records four recurrences. The re-measurement agent's proposes two tags:
`[05a-t1-halving-rule-admits-aliased-windows]` (X1) and `[05a-t-phase-abscissa-rounding-remains]`
(X2).

---

## Notes for the laptop (§B.4), not checks

- **T1's reading is the user's.** As written, T1 fails; by the true step it holds. Under §B.4's
  closing rule a T1 or T4 failure means asking the user before closing
  `[05-t-remainder-keeps-the-anchoring-floor]` or `[04-g-phase-is-reduced-against-the-double-two-pi]`.
  T4 fails as written either way, on the two rows on which 05's L3 failed (05's W5, untouched by
  $T$).
- **Links to files §B.4's copy leaves out.** `rehearsal/verification.md` links to
  `out/qsi_rows.csv` and `out/store_value_diff.json` (41 MB), and `rehearsal/issues.md` cites
  `out/qsi_rows.csv`. §B.4's `rsync` takes no CSV, so those links break unless the laptop decides
  otherwise.
- **Readers' output.** `rehearsal/extract/` (177 MB) is the readers' output, and its agent marks it
  "not for copying back".
- **Pickles.** `wider/state/T0/*/*.pkl` and `wider/state/smoke/*.pkl` may be left out.
  `wider/state/probe_windows.json` (4 MB) is what T1's re-score, and this review's, read.
- **Kept on this host, the user's to keep or retire:** R11's store (primary `85538005…`) and R12's
  (`94dae1cc…`), with no sidecar, and the worktree `SecondaryGWKit-lpf05a-c4eaefd`.

---

## The scratch readers

### `spot_qsi.py` (SHA-256 2222eed39f3d9a9f…)

```python
# Scratch reader (orchestrator review, check 4): one differing QuadSourceIntegral row, read-only.
import csv, sqlite3, sys
csvp, base = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(csvp)) if r["total_bit_identical"] in ("False", "0", "false")]
row = max(rows, key=lambda r: float(r["rel_dtotal"]))
print("tabulation row (max rel_dtotal):", {k: row[k] for k in ("k","q","r","z_response","total_A","total_B","rel_dtotal","abserr_A","abserr_B")})
for tag in ("r11", "r12"):
    hits = []
    for s in range(4):
        con = sqlite3.connect(f"file:{base}-{tag}-shard{s:04d}.sqlite?mode=ro", uri=True)
        q = ("SELECT (SELECT w.k_inv_Mpc FROM wavenumber_exit_time e JOIN wavenumber w ON w.serial=e.wavenumber_serial WHERE e.serial=q.k_wavenumber_exit_serial),"
             " (SELECT w.k_inv_Mpc FROM wavenumber_exit_time e JOIN wavenumber w ON w.serial=e.wavenumber_serial WHERE e.serial=q.q_wavenumber_exit_serial),"
             " (SELECT w.k_inv_Mpc FROM wavenumber_exit_time e JOIN wavenumber w ON w.serial=e.wavenumber_serial WHERE e.serial=q.r_wavenumber_exit_serial),"
             " (SELECT z FROM redshift WHERE serial=q.z_response_serial), q.total, q.total_abserr FROM QuadSourceIntegral q")
        for k, qq, rr, z, tot, ab in con.execute(q):
            if (float(k).hex(), float(qq).hex(), float(rr).hex(), float(z).hex()) == tuple(float(row[c]).hex() for c in ("k","q","r","z_response")):
                hits.append((s, tot, ab))
        con.close()
    for s, tot, ab in hits:
        print(f"{tag} shard{s}: total {float(tot).hex()} ({tot!r})  abserr {float(ab).hex()}")
    print(tag, "matches", len(hits), "row(s)")
```

### `c_vs_05c.py` (SHA-256 356e56454ad59321…)

```python
# Scratch reader (orchestrator review, check 6): 05a's c units at 1e-7 against 05's c units, from the two checkpoint.jsonl.
import json, sys
new, old, detail = sys.argv[1], sys.argv[2], sys.argv[3]
def load(p):
    return {json.loads(l)["unit"]: json.loads(l)["data"] for l in open(p)}
N, O = load(new), load(old)
def h(x):
    # values are recorded as {"repr", "hex"}; check the two agree, then compare the bit pattern
    if x is None: return None
    if isinstance(x, dict):
        assert float(x["repr"]).hex() == x["hex"] or x["repr"] in ("nan", "inf", "-inf"), x
        return x["hex"]
    return float(x).hex()
def fields(d):
    out = [("total", d["result"]["total"]), ("total_abserr", d["result"]["total_abserr"])]
    for c in d["calls"]:
        out += [(f"call[{c['index']}].value", c["value"]), (f"call[{c['index']}].abserr", c["abserr"])]
    return out
rows = sorted(k for k in N if k.startswith("c-1e-7-"))
nbad = 0
for u in rows:
    a, b = fields(N[u]), fields(O[u])
    same = len(a) == len(b) and all(x[0] == y[0] and h(x[1]) == h(y[1]) for x, y in zip(a, b))
    nbad += not same
    if u == detail:
        print(f"== {u}: 05a role {N[u]['role']}, tree_head {N[u]['tree_head']}; 05's tree_head {O[u].get('tree_head')}")
        for x, y in zip(a, b):
            print(f"   {x[0]:18s} 05a {h(x[1])}  05 {h(y[1])}  {'==' if h(x[1]) == h(y[1]) else 'DIFFER'}")
print(f"c-1e-7 units compared: {len(rows)}; not bit-identical in total/abserr/every call value+abserr: {nbad}")
```

### `rescore_t1_orch.py` (SHA-256 e9563a331a6e1786…)

```python
# Orchestrator's scratch re-score of 05a T1 (prompt §9) from state/probe_windows.json, check 9.
# sigma_theta as test 3's _sigma_theta: numpy.unwrap, degree-4 least squares in t in [-1,1], RMS residual.
import json, math, sys
import numpy as np

def F(v):
    # samples are saved as float.hex strings (or floats); decode exactly
    return np.array([float.fromhex(x) if isinstance(x, str) else float(x) for x in v], dtype=float)
P = json.load(open(sys.argv[1]))
EPS = 2.0 ** -52
assert P["eps"] == EPS
HALF_PI = math.pi / 2

def sig(x, th):
    x = np.asarray(x, float); th = np.unwrap(np.asarray(th, float))
    a, b = x[0], x[-1]
    t = (2.0 * x - (a + b)) / (b - a)
    coef = np.polynomial.polynomial.polyfit(t, th, 4)
    res = th - np.polynomial.polynomial.polyval(t, coef)
    return float(np.sqrt(np.mean(res * res))), float(np.max(np.abs(np.diff(th))))

def level(win, rule):
    L = {tr: {e["n"]: e for e in win["ladder"][tr]} for tr in ("c", "t")}
    for n in sorted(L["c"]):
        ec, et = L["c"][n], L["t"].get(n)
        if et is None or not (ec["distinct"] and et["distinct"]):
            return None                    # distinctness lost before resolution: unresolved
        if rule == "literal" and ec["step_z"] < HALF_PI and et["step_z"] < HALF_PI:
            return n
        if rule == "true_step" and ec["true_step_z_est"] < HALF_PI and et["true_step_z_est"] < HALF_PI:
            return n
    return None

out = {}
for rule in ("literal", "true_step"):
    scored = unresolved = c_fail = t_fail = 0; pz_largest = []; mism = []; t_ratios_fail = []; t_ratios_ok = []; c_ratios = []
    worst = None
    for w in P["windows"]:
        e = EPS * w["abs_theta"]
        if e < 1e-11:
            continue
        n = level(w, rule)
        rec = w.get(rule)
        if n is None:
            unresolved += 1
            if rec and rec.get("n") is not None: mism.append((w["row"], w["subinterval"], w["factor"], "agent level", rec.get("n"), "mine None"))
            continue
        scored += 1
        if rec is None or rec.get("n") != n:
            mism.append((w["row"], w["subinterval"], w["factor"], "agent level", None if rec is None else rec.get("n"), "mine", n)); continue
        s = rec["samples"]
        z = F(s["z"])
        a, b = float.fromhex(next(x for x in w["ladder"]["c"] if x["n"] == n)["a"]), float.fromhex(next(x for x in w["ladder"]["c"] if x["n"] == n)["b"])
        assert np.array_equal(z, np.linspace(a, b, 65)), "samples are not linspace(a,b,65)"
        sc, stc = sig(z, F(s["theta_z_c"])); st, stt = sig(z, F(s["theta_z_t"]))
        if not (math.isclose(sc, rec["sigma_c"], rel_tol=1e-9, abs_tol=0) and math.isclose(st, rec["sigma_t"], rel_tol=1e-9, abs_tol=0)):
            mism.append((w["row"], w["subinterval"], w["factor"], "sigma", sc, rec["sigma_c"], st, rec["sigma_t"]))
        pz = float(np.max(np.abs(F(s["p_z_t"]))))
        terms = {"1e-3": 1e-3 * e, "pz": EPS * (w["k"] * pz + 2 * math.pi), "1e-13": 1e-13}
        bound = max(terms.values())
        if max(terms, key=terms.get) == "pz": pz_largest.append(st / e)
        c_ok = sc >= 0.1 * e; t_ok = st <= bound
        c_fail += not c_ok; t_fail += not t_ok
        c_ratios.append(sc / e)
        (t_ratios_ok if t_ok else t_ratios_fail).append((st / e, st / bound, w["row"], w["subinterval"], w["factor"], w["literal"]["true_step"] if rule == "literal" else None))
    verdict = "fails" if (c_fail or t_fail) else ("undecided" if unresolved else "holds")
    print(f"== T1 under the {rule} reading: windows with eps|theta| >= 1e-11: {scored + unresolved}; resolved {scored}; unresolved {unresolved}")
    print(f"   c clause fails in {c_fail}; sigma_c/(eps|theta|) {min(c_ratios):.3g}..{max(c_ratios):.3g}")
    print(f"   t clause holds in {len(t_ratios_ok)}, fails in {t_fail}")
    if t_ratios_ok: print(f"     holding: sigma_t/(eps|theta|) max {max(r[0] for r in t_ratios_ok):.3g}; sigma_t/bound max {max(r[1] for r in t_ratios_ok):.3g}")
    if t_ratios_fail:
        print(f"     failing: sigma_t/(eps|theta|) {min(r[0] for r in t_ratios_fail):.3g}..{max(r[0] for r in t_ratios_fail):.3g}; sigma_t/bound up to {max(r[1] for r in t_ratios_fail):.3g}")
        ts = [r[5] for r in t_ratios_fail]
        print(f"     failing windows' true step per sample (rad): {min(ts):.3g}..{max(ts):.3g}")
    print(f"   pz term largest in {len(pz_largest)} windows" + (f"; sigma_t/(eps|theta|) there {min(pz_largest):.3g}..{max(pz_largest):.3g}" if pz_largest else ""))
    print(f"   level/sigma mismatches with the agent's record: {len(mism)} {mism[:5]}")
    print(f"   VERDICT ({rule}): {verdict}")
# u-sampled, descriptive, at the literal level
over = []; ratio_abs = []
for w in P["windows"]:
    e = EPS * w["abs_theta"]
    if e < 1e-11 or w["literal"].get("n") is None: continue
    su = w["literal"]["sigma_u_t"]; ratio_abs.append(su / w["abscissa_term"])
    if su > 0.1 * e: over.append((su / e, su / w["abscissa_term"], w["abscissa_term"] / e))
print(f"== u-sampled sigma under t (literal level): above 0.1 eps|theta| in {len(over)} windows" + (f"; there sigma_u/(eps|theta|) {min(o[0] for o in over):.3g}..{max(o[0] for o in over):.3g}, abscissa term/(eps|theta|) {min(o[2] for o in over):.3g}..{max(o[2] for o in over):.3g}" if over else ""))
print(f"   sigma_u_t / abscissa term over scored windows: {min(ratio_abs):.3g}..{max(ratio_abs):.3g}")
print(f"windows total {len(P['windows'])}, rows {len(set(w['row'] for w in P['windows']))}")
```

### `u_sampled.py` (SHA-256 5614a026dd201532…)

```python
# Orchestrator scratch: the u-sampled sigma under t (descriptive), at each reading's level, from the saved records.
import json, sys
P = json.load(open(sys.argv[1])); EPS = 2.0 ** -52
for rule in ("literal", "true_step"):
    over = []; ratio = []
    for w in P["windows"]:
        e = EPS * w["abs_theta"]
        if e < 1e-11 or w[rule].get("n") is None: continue
        su = w[rule]["sigma_u_t"]; ratio.append(su / w["abscissa_term"])
        if su > 0.1 * e: over.append((su / e, w["abscissa_term"] / e))
    print(f"{rule}: sigma_u_t > 0.1 eps|theta| in {len(over)} of {len(ratio)}; there sigma_u/(eps|theta|) {min(o[0] for o in over):.3g}..{max(o[0] for o in over):.3g}, abscissa/(eps|theta|) {min(o[1] for o in over):.3g}..{max(o[1] for o in over):.3g}; sigma_u_t/abscissa {min(ratio):.3g}..{max(ratio):.3g}")
```

### `rescore_t234.py` (SHA-256 aa41a0e4e52c3d88…)

```python
# Orchestrator's scratch re-score of 05a T2, T3, T4 from the run's checkpoint.jsonl (check 9).
import json, math, sys
from collections import Counter
from fractions import Fraction
new, old = sys.argv[1], sys.argv[2]
def load(p): return {json.loads(l)["unit"]: json.loads(l)["data"] for l in open(p)}
N, O = load(new), load(old)
def v(x): return float.fromhex(x["hex"]) if isinstance(x, dict) else float(x)
print("statuses:", Counter(d["status"] for d in N.values()), "roles at c-1e-7:", Counter(d["role"] for u, d in N.items() if u.startswith("c-1e-7-")))
rows = sorted(d["row"] for u, d in N.items() if u.startswith("c-1e-7-"))
d20 = sorted(d["row"] for u, d in N.items() if u.startswith("c-1e-7-") and d["role"] == "d20")
capped = sorted(r for r in d20 if any((c.get("max_depth") or 0) >= 20 for c in O[f"p-1e-7-{r}"]["calls"]))
over = [u for u, d in N.items() if d["status"] != "done"]
print(f"rows {len(rows)}, depth-20 rows {len(d20)}, capped under B1 (05's p at depth 20) {len(capped)}: {capped}")
print("units not done:", over)
# T2
bad2 = []; maxd = 0
for r in rows:
    d = N[f"t-1e-7-{r}"]
    if d["status"] != "done": bad2.append((r, d["status"])); continue
    for c in d["calls"]:
        md = c.get("max_depth") or 0; maxd = max(maxd, md)
        if md >= 18 or (c.get("n_depth_ge_18") or 0) > 0: bad2.append((r, c["index"], md))
print(f"T2: false on {len(bad2)} of {len(rows)} rows {bad2}; largest depth under t@1e-7 {maxd} -> {'holds' if not bad2 else 'fails'}")
# T3
bad3 = []; worst = (0, None); worst_rel = (0, None); ident = 0
for r in rows:
    c, t = N[f"c-1e-7-{r}"]["result"], N[f"t-1e-7-{r}"]["result"]
    tc, tt, ac = v(c["total"]), v(t["total"]), v(c["total_abserr"])
    ident += float(tc).hex() == float(tt).hex()
    d = abs(Fraction(tt) - Fraction(tc))
    ok = math.isinf(ac) or d <= Fraction(ac)
    if not ok: bad3.append(r)
    if not math.isinf(ac) and ac > 0: worst = max(worst, (float(d / Fraction(ac)), r))
    if tc != 0: worst_rel = max(worst_rel, (float(d / abs(Fraction(tc))), r))
print(f"T3: false on {len(bad3)} of {len(rows)} {bad3}; largest |t-c|/abserr_c {worst[0]:.3g} ({worst[1]}); largest relative {worst_rel[0]:.3g} ({worst_rel[1]}); bit-identical totals {ident} -> {'holds' if not bad3 else 'fails'}")
# T4
res4 = []
for r in d20:
    a, b = N[f"t-1e-7-{r}"], N[f"t-1e-8-{r}"]
    if a["status"] != "done" or b["status"] != "done": res4.append((r, None)); continue
    t7, t8, ab = v(a["result"]["total"]), v(b["result"]["total"]), v(a["result"]["total_abserr"])
    d = abs(Fraction(t7) - Fraction(t8))
    ok = math.isinf(ab) or d <= Fraction(ab)
    res4.append((r, ok, (float(d / Fraction(ab)) if not math.isinf(ab) and ab > 0 else 0.0)))
scored = [x for x in res4 if x[1] is not None]
false4 = [(x[0], round(x[2], 3)) for x in scored if not x[1]]
print(f"T4: scored {len(scored)} of {len(d20)}; true on {sum(x[1] for x in scored)}; false on {false4}; capped rows largest ratio {max(x[2] for x in scored if x[0] in capped):.3g} -> {'holds' if not false4 else 'fails'}")
# 05's L3 under c on the same rows, for comparison
l3 = []
for r in d20:
    a, b = O[f"c-1e-7-{r}"], O[f"c-1e-8-{r}"]
    t7, t8, ab = v(a["result"]["total"]), v(b["result"]["total"]), v(a["result"]["total_abserr"])
    if not math.isinf(ab) and abs(Fraction(t7) - Fraction(t8)) > Fraction(ab): l3.append((r, round(float(abs(Fraction(t7) - Fraction(t8)) / Fraction(ab)), 3)))
print("05's L3 under c false on:", l3)
# budget
fr = [(u, d["wall_s"] / d["budget_s"]) for u, d in N.items() if d.get("budget_s")]
print(f"budgeted units {len(fr)}; largest wall/budget {max(fr, key=lambda x: x[1])}")
```


---

## Addendum, 2026-10-26 13:30 UTC — T1 scored by the true step, on the user's decision

Added after the sections above, which stand as they were written at 12:47Z.

**The decision.** It was relayed to this session at about 13:23Z by the laptop orchestration
session (`local_3e43e245-…`).
- The question put to the user: "Which reading of §9's resolution rule should be T1's scored
  verdict?"
- The user chose "True step (Recommended)".
- **T1's scored verdict** is taken on windows resolved by the true step: $|d\theta/dz|(b-a)/64 <
  \pi/2$ under both $c$ and $t$, halving further where needed.
- **The literal reading** is reported beside it, unscored.
- **The laptop amends §9's wording** in 05a's records commit.

**The timing, from the files only.** The laptop session asked for the note to say that no T1
number existed when the decision was made. These files do not bear that out, so the note was told
to record the times alone:
- `predictions.md` fixed both readings at 12:30:03Z, before the run;
- `analyse.py` first scored both at 12:35:49Z;
- this review carried both from 12:47Z;
- the decision reached this session at about 13:23Z.

When the user decided is the laptop session's to record. This was sent to that session at 13:24Z.

**The agent's records**, read by me:
- **Deviation 16**, `STRUCTURALLY REQUIRED`, quotes the user's choice.
- **`state/predictions_note_T1_reading.md`** (13:24:48Z) quotes the decision and tabulates the
  times as above. It says the session "cannot say whether any T1 number had reached the user by
  then".
- **`log.md`, `verification.md` and `issues.md` X1** were added to, not rewritten.
- **The new script** `scripts/t1_true_step_only.py` (`316fa3bd…`) reads only
  `state/probe_windows.json` (`9919f483…`). Its output is `out/cmd/26_T1_true_step_scored.txt`.

**Check 8, re-read: holds.** `predictions.md` is unchanged: SHA-256 `6a3cc8ce…`, mtime 12:30:03Z,
and every quoted block still verbatim against the prompt. Its first copy is unchanged too (mtime
12:29:30Z). The note sits beside them and changes neither.

**Check 9, T1 re-scored by the true-step reading: holds.** My `rescore_t1_orch.py` (above) already
scored this reading, choosing the level from each window's saved ladder as the first at which both
trees' true step is below π/2:

```
== T1 under the true_step reading: windows with eps|theta| >= 1e-11: 70; resolved 70; unresolved 0
   c clause fails in 0; sigma_c/(eps|theta|) 0.198..0.374
   t clause holds in 70, fails in 0
     holding: sigma_t/(eps|theta|) max 1.31e-05; sigma_t/bound max 0.0131
   pz term largest in 0 windows
   level/sigma mismatches with the agent's record: 0 []
   VERDICT (true_step): holds
```

The agent's `out/cmd/26` agrees:
- 70 true, 21 below 1e-11 and not scored, 0 unresolved, so HOLDS;
- σ_c/ε|θ| 0.1979–0.3737; σ_t/ε|θ| 4.379e-10–1.313e-05; the largest σ_t/bound 0.01313;
- no window needed a re-probe, so none was run.

The literal reading, unscored, fails in 36 of 70.

**Scope, re-checked.** Since 12:47Z nothing changed in `var/` outside
`var/levin-phase-floor-05a/wider/`, and nothing in the worktree. Both trees' `git status` is empty
(the worktree's with `--ignored`).

**Verdict, as it now stands: PASS.** The verdicts are:
- **T0 holds.**
- **T1, scored by the true step, holds** (70 of 70). As written, the literal reading fails (36 of
  70).
- **T2 holds** (39 of 39).
- **T3 holds** (39 of 39).
- **T4 fails** (29 of 31: `mid-G-3` 1.70, `mid-G-r-3` 1.29, as 05's L3).

No unit went over budget. Under §B.4's closing rule T4's failure still means asking the user
before either header issue is closed.
