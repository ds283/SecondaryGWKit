# Log 04 — does A2's Levin cost growth survive production's phases?

> **Committed on the laptop** in A4's records commit. The paths below under `var/handover-04/` are committed under [`docs/handover/a4/`](../../../docs/handover/a4/), the memo as [`docs/handover/GK-PHASE-COST-GROWTH.md`](../../../docs/handover/GK-PHASE-COST-GROWTH.md).

**Prompt:** prompts/handover/04-gk-phase-decision-test.md
**Commit:** none (the prompt commits nothing on the Mac Studio; the laptop makes the records commit)
**Dispatch SHA:** `28ebaf0743e0555fb2c4512ce714884005a60eff`, branch `handover-remedial`, main checkout
`/Users/ds283/Documents/Code/SecondaryGWKit` on the Mac Studio (mps026298)
**Model:** Claude Opus 5.5 (`claude-opus-5-5`)
**Date:** 2026-10-26
**Result:** COMPLETE WITH DEVIATIONS.
- **Run 2, the result.** `handover-04-cost-growth-20261026T173720` ended `done` (exit 0) at about
  21:20 UTC with all 60 cells `ok`. None was `not reached`.
- **The memo,** `GK-PHASE-COST-GROWTH.md`, answers prompt §4 with **neither reading cleanly**:
  - arm P's growth lies between E's and S's on every shape: 6.26 / 3.29 / 2.16, against E's
    1.29 / 1.86 / 0.94 and S's 6.8 / 46.9 / 40.2;
  - P₀ grows 8,930 / 2,360 / 619.
- **Run 1, the record of the failed run.** `handover-04-cost-growth-20261026T152704` ended `failed`
  (exit 2): 34 `ok`, 6 `raised`, 20 `skipped` (deviation 12). Its records are kept:
  - `GK-PHASE-COST-GROWTH-run1.md`, `scripts-run1/`, `out/run1/` and `state/readings-run1.md`;
  - its run directory, untouched;
  - the sections below, as first written.

## What shipped (under `var/handover-04/` only)

- `scripts/a4_cells.py`: the sibling of `realistic_large_x.py`. It imports `RUNS`, `scaled_shape`,
  `fixture_x_max`, `seed_fixtures`, `reference_of`, `REF_ATOL` and `REF_RTOL`, and adds:
  - `build_case(base, x_resp, arm)` for arms E, S, P0 and P;
  - `phase_objects` and `construction_problems`, the §2.1 check;
  - `GroupRecorder`, the §2.2 wrapper at `qsi_module.phase_group_Levin_integral`;
  - `run_cell(shape, x_resp, arm, instrument=True)`, whose `handover` and partition reads replace
    the clamp reads;
  - a `--cell SHAPE X ARM --out PATH` entry point, one cell per process.
- `scripts/a4_checks.py`: the `construction`, `transparency` and `determinism` checks.
- `scripts/a4_driver.py`: the registered run's driver. It runs 60 cells, up to 4 at once, with a
  3 h budget per cell, and skips the rungs above a raise.
- `scripts/a4_launch.py`: the launcher, after `docs/levin-phase-floor/05/wider/scripts/launch.py`.
- `out/`: the Gate, probes, the smoke test and the construction check, each with its output.
- `probes/`: three scratch probes. `smoke1/`: the driver's smoke test, with no registry.

SHA-256 at the stop (`out/state_at_stop.txt`):

| file | SHA-256 |
|---|---|
| `a4_cells.py` | `201419cb264b…` |
| `a4_checks.py` | `0c3920e4f16d…` |
| `a4_driver.py` | `70dd520be4dd…` |
| `a4_launch.py` | `6a85c1904d16…` |

## Deviations from the prompt

1. **UNINTENDED DRIFT — `var/handover-04/` was created by me before the Gate's "no
   `var/handover-04/`" item was checked.** I ran `mkdir -p var/handover-04/{out,scripts,state}` in
   the same command as the Gate, so that the Gate's output could be saved there. So the Gate's
   own output cannot show the directory absent. The orchestrator's Gate at 14:43 UTC found no
   `var/handover-04/`. Its birth time is 14:44:06 UTC, one second before the Gate's `date`, and
   at that moment it held only the three empty directories I made
   (`out/gate-varhandover04-note.txt`). Kept: the directory is this prompt's own.
2. **STRUCTURALLY REQUIRED (blocking) — the realistic $T$ does not build at rungs $10^4$–$10^6$.**
   The prompt assumes that `TkSourceFunctions(fixture.exact_envelope_model(), k, Tn, Tw)` builds
   on a `_TableCsTauFixture(w, k, x_max, drop_first_WKB_sample=True)` at every rung. It does not.
   `_TableCsTauFixture.exact_envelope_model` builds its `cs_tau` `CumulativeTable` on the
   fixture's own `z_WKB`. With the first sample dropped, that grid starts at $z_{\rm switch}$, one
   grid step **below** $z_{\rm init}$. The phase is anchored at $z_{\rm init}$, so
   `build_phi_samples` evaluates `cs_tau.delta(z, z_init)`. `CumulativeTable._nearest_node`
   refuses a $u = \log(1+z)$ more than one grid interval above the top node. $z_{\rm init}$ is
   exactly one interval above, so the decision is made by the rounding of `log1p`
   (`out/probe_table_margin.txt`):
   - at the rungs that build, $(u_{\rm init} - u_{\rm top}) - (u_{\rm top} - u_{\rm next})$ is
     $-1$ or $0$ ulp;
   - at the rungs that fail it is $+1$ ulp ($+1.776\times10^{-15}$), and the table raises
     `RuntimeError: CumulativeTable[cs_tau]: z=… lies more than one grid interval above the table`.

   Affected (`out/check_construction.txt`): every realistic arm (S, P0 and P) at $10^4$, $10^5$ and
   $10^6$ on all three shapes, which is 27 of 45 realistic cells. On `together` and `T-first` both
   $T_q$ and $T_r$ fail; on `q-smooth` only $T_r$. Arm E, and the realistic arms at 980, $10^7$
   and $10^8$, build and pass the §2.1 check. Not worked around: prompt §8 forbids substituting an
   approximation under the arm's name, and fixing the fixture would mean editing
   `test_g_phase_reanchor.py`. Stopped.
3. **IMPLEMENTATION CHOICE — how the realistic $T$ objects enter the cell.** They are built once
   per cell during set-up, with the `k` that `evaluate_QuadSource_integral` passes (`case.q.k`,
   `case.r.k`). `Case.Tk_builder` returns them after checking it was handed exactly those input
   objects, as `Case`'s exact branch returns prebuilt objects. So the construction check reads
   the objects the integral runs. The alternative, building inside the hook as `Case`'s realistic
   branch does, would build objects the check never sees. `case.Tq_inputs`/`Tr_inputs` are the
   dropped table fixture's captured samples. `Case._fixtures` is seeded by A2's `seed_fixtures`
   (plain `Fixture`s), so the `Case` itself, its source grid and its $G$ grid are A2's, as in A2's
   open seam.
4. **IMPLEMENTATION CHOICE — only `.phase` differs.** For P0 and P, `case.GkPolicy.functions` (a
   namedtuple) is `_replace`d in `phase` alone, so the amplitude, `numeric_Gk` and every other
   field are `OffsetBesselPhaseGk`'s own bound methods. `_ProductionPhaseGk` is built on a grid
   recomputed as `Case` builds it, and checked equal to `case.gk.WKB_region`. Its inherited
   `phase_spline` built at every rung that got as far as $G$ (980, $10^7$, $10^8$), so the §2.1
   fallback was never taken. The fallback, if needed, patches
   `test_phase_groups.phase_spline` to return `None` for the constructor's duration, and the path
   taken is recorded per cell.
5. **IMPLEMENTATION CHOICE — schedule.** The prompt says "arm E, then rung by rung upward". Within a
   rung, the shapes run in A2's order and the arms in the order P, P0, S, because S is the
   expensive one. A cell starts only after the cell below it on the same shape and arm has ended.
   That honours §8's "stop before the next rung of that shape and arm" with four cells in flight.
   The rungs above a `not reached` cell still run: the prompt calls over-budget a result, not a
   failure, and A2's curves are not monotone (`together` $10^8$ was cheaper than $10^7$).
6. **IMPLEMENTATION CHOICE — environment.** Every Python process here runs with `-B` and
   `PYTHONDONTWRITEBYTECODE=1`, so that nothing is written into the tree. The cells, probes and
   checks also run with `OMP_NUM_THREADS=1`, as `levin-phase-floor` 03–05 did. The checks use the
   same environment as the run's workers. The driver reads `git --no-optional-locks status --short`
   so that `.git/index` is not refreshed.
7. **IMPLEMENTATION CHOICE — probes and a smoke test before the checks.** These ran before the §2.1
   check and are saved:
   - `probes/probe_build.py` (`out/probe_build.txt`): the four arms built at `together` 980;
   - `probes/probe_run.py` (`out/probe_run_{E,S,P0,P}.txt`): one cell per arm at `together` 980;
   - the driver's smoke test (`out/smoke1_driver.txt`, `smoke1/`), with no registry and an 8 s
     budget. This test found the failure: `P|together|10000` raised.

   The probes ran an earlier text of `a4_cells.py`, which differs only in one `bool(...)` and one
   docstring sentence. They are development measurements, not the prompt's checks.

## Verification performed

- **Gate** (`out/gate.txt`, 14:44:07 UTC): `HEAD` is `28ebaf0743e0555fb2c4512ce714884005a60eff`;
  `git status --short` is empty; `realistic_large_x.py` has SHA-256 `c1cd3598c23a52051a0eb269…`;
  `runregistrykit list` shows 19 runs, all finished, none running or stale. `var/handover-04/`:
  see deviation 1. Load average 4.99.
- **§2.1 construction check** (`out/check_construction.txt`, `out/check_construction.json`):
  **FAILS.**
  - 33 of 60 cells build and pass. In those, S's $G$ is a `phase_spline`. P0's and P's $G$ are
    `PrimitivePhase`s on a `TablePrimitive`, with `exact_remainder` false and true. Every
    realistic $T$ is a `PrimitivePhase` on a `TablePrimitive` with `exact_remainder` true, from a
    `_TableCsTauFixture` with its first sample dropped.
  - 27 raise as deviation 2 says.
- **§2.2 transparency and §3 determinism:** not run, because the construction check failed first.
  For information only, the probes at `together` 980 (instrumented, one run each):

  | arm | regions | max depth | `integral_time` |
  |---|---|---|---|
  | E | 7 | 0 | 0.53 s |
  | S | 1,254 | 12 | 17.3 s |
  | P0 | 53 | 6 | 0.98 s |
  | P | 46 | 5 | 0.89 s |

  A2 recorded 1,304 regions at this rung, before B1. These are not a result of this prompt.
- **Nothing launched.** No run directory and no `runregistrykit.begin`: `var/runs/` holds no
  `handover-04` entry. No lock taken. No Ray. `git status --short` is still empty.
  `realistic_large_x.py` still hashes `c1cd3598…`.

## Observations not acted on

1. `_TableCsTauFixture` (`ComputeTargets/tests/test_g_phase_reanchor.py:448`) builds its table on
   the fixture's own `z_WKB`. With `drop_first_WKB_sample=True` that grid does not cover the
   phase anchor $z_{\rm init}$, which then sits exactly one interval above the table. Whether it
   builds is decided by one ulp of `log1p`. Production's `functions.cs_tau` covers the whole
   background grid, so production does not have this. `issues.md` item 1.

## State handed to the next step

- Awaiting the user's answer to the question below. Then: re-run the construction check, run
  transparency and determinism, write `state/readings.md` (prompt §4, word for word), and launch.
- If any script changes, every check is re-run on the new text, because the checks record
  `a4_cells.py`'s SHA-256.

## The question for the user

**The realistic $T$ will not build as §2.1 specifies on 27 of 45 realistic cells.** A
`_TableCsTauFixture` with `drop_first_WKB_sample=True` builds its `cs_tau` table on the dropped
grid. The phase anchor $z_{\rm init}$ then sits exactly one grid step above the table's top node,
and `CumulativeTable` refuses it at rungs $10^4$–$10^6$ by one ulp of `log1p`. How should A4's
realistic $T$ be built?

- **(a)** Build the table on the fixture's **undropped** WKB grid, so that $z_{\rm init}$ is its top
  node, as in production, where `cs_tau` covers $z_{\rm init}$. The `TkSourceFunctions` samples are
  still the dropped fixture's, so the seam stays production's. The change would live in A4's own
  script, as a subclass of `_TableCsTauFixture`, with no test file edited. One difference from
  production: $z_{\rm init}$ becomes a table node, so $p_R = 0$, where production's
  $z_{\rm init}$ is off-node.
- **(b)** Build $T$ from an undropped `_TableCsTauFixture`. That uses A2's closed seam, not
  production's, in S, P0 and P alike.
- **(c)** Something else: for example, fix `_TableCsTauFixture` in the test module first (B2's, or
  a prompt of its own), and run A4 after that.

---

## Resumed 2026-10-26: the user's answer, and a second stop

**The user's answer**, as the orchestrator relayed it: *"Extend the table (Recommended)": "In A4's
own script, build cs_tau on the fixture's numeric and WKB nodes together, so the table covers z_init
as production's does and z_init is off a node. The seam stays production's, and no test file is
edited. Record the fixture defect as an issue."* The orchestrator added: "z_init must be off a node:
record p_R for each fixture."

8. **STRUCTURALLY REQUIRED (blocking) — on the union of the numeric and WKB nodes, $z_{\rm init}$
   is a node, not off one.**
   - **The conflict.** The answer assumes that a table built on the union of the fixture's numeric
     and WKB nodes has $z_{\rm init}$ off a node. It does not. `Fixture` builds
     `z_numeric = log_grid(..., one_plus_z_at(exp(efolds_subh)))`, and `np.linspace` includes its
     end point. So `z_numeric[-1]` **is** `crossover_z` = $z_{\rm init}$, bit for bit, at all 30
     fixtures of the ladder (`out/probe_union_nodes.txt`). Because the dropped `z_WKB` and
     `z_numeric_below` start one step below it, the union contains $z_{\rm init}$ as a node.
   - **Union exactly as stated** ("`Union`", `probes/probe_union_table.py`,
     `out/probe_union_table.txt`): $T$ builds at every rung (0 of 30 raise). It is a
     `PrimitivePhase` on a `TablePrimitive` with `exact_remainder` true. The anchor is node 369 and
     **$p_R = 0$ at every fixture**, so "off a node" fails.
   - **Union without the one numeric node equal to $z_{\rm init}$** ("`UnionOff`", same probe):
     $T$ builds at every rung (0 of 30 raise), the anchor is off a node, and $p_R \ne 0$. $p_R$
     runs from $-4.4\times10^{-3}$ (`q-smooth` 980, $T_q$) to $-3.6\times10^{-10}$ (`together`
     $10^8$, $T_r$), with signs as the nearest node falls, and roughly $\propto 1/x_{\rm resp}$.
     The interval around $z_{\rm init}$ in this table is then two grid steps wide.

   The two stated properties, "on the union of numeric and WKB nodes" and "z_init off a node",
   cannot both hold for this fixture. Choosing between them is the user's call, so I stopped.
   Nothing launched; no script in `scripts/` changed since the first stop.

**The question for the user:** with the table on the union of numeric and WKB nodes, $z_{\rm init}$
**is** a table node, because the fixture's numeric grid ends exactly at $z_{\rm init}$. Which do
you want?
- **(i)** the union as stated, with $z_{\rm init}$ a node ($p_R = 0$, as $G$'s anchor is); or
- **(ii)** the union minus the one numeric node equal to $z_{\rm init}$, so that $z_{\rm init}$ is
  off a node ($p_R \ne 0$, as in production), at the cost of a two-step interval around it.

Both build at all 60 cells.

---

## Resumed again 2026-10-26: the user chose (ii), and the run launched

**The user's answer**, as the orchestrator relayed it: *"(ii) Off a node (Recommended)": "The union
of numeric and WKB nodes minus the one numeric node equal to z_init: z_init is off a node with p_R
!= 0, as in production; the table interval around it is two grid steps wide."*

**Why there was a second stop.** The first answer's instruction contradicted itself for this
fixture: "the union of numeric and WKB nodes" and "z_init off a node" cannot both hold, because
`Fixture`'s numeric grid ends exactly on z_init. Deviation 8 records the measurement.

9. **STRUCTURALLY REQUIRED (the user's choice, answering deviations 2 and 8) — the realistic $T$'s
   table.** `a4_cells._UnionTableCsTauFixture(_TableCsTauFixture)` overrides only
   `exact_envelope_model`. It builds the `cs_tau` `CumulativeTable` on the union of `z_numeric`,
   `z_numeric_below` and `z_WKB`, without the node equal to `crossover_z`, descending. It checks
   that the table's top node is above $z_{\rm init}$ and its bottom node below, and shifts to the
   closed form at the top node, as `_TableCsTauFixture` does. Everything else is as §2.1 says:
   - `drop_first_WKB_sample=True`, so production's seam;
   - samples captured with `exact_Tk_inputs`;
   - `TkSourceFunctions(fixture.exact_envelope_model(), k, Tn, Tw)` with no keyword.

   Every cell records, for $T_q$ and $T_r$, `z_anchor` (equal to `z_init`), `anchor_node_index`
   (`None`, so off a node), the nearest node, and `p_R`. The construction check now also requires
   that the anchor is $z_{\rm init}$, off a node, with $p_R \ne 0$. No test file was edited, and
   the fixture defect stays in `issues.md` item 1.
10. **IMPLEMENTATION CHOICE — the first stop's construction output is kept, not overwritten.**
   `out/check_construction_stop1.{txt,json}` are the failing check on the first text of
   `a4_cells.py`; `out/check_construction.{txt,json}` are the check on the text that ran.
11. **IMPLEMENTATION CHOICE — `black`** was run on `scripts/` before the checks, which the
   repository's formatting rule asks for at landing. `a4_driver.py` and `a4_launch.py` came out
   unchanged.

### Verification performed (the run's scripts)

The scripts' SHA-256 when the checks ran (`out/scripts_sha256_before_checks.txt`) and at launch
(`out/prelaunch.txt`) are the same:

| file | SHA-256 |
|---|---|
| `a4_cells.py` | `8d7dc6ecd618ea9263438f4dff2d5ac09a6b324fde6b00b64798bf0f9b78711d` |
| `a4_checks.py` | `a023b7d23f6cf18b1517f936e7c3eff020cf3e2a477670565b7cda91cd89dd4a` |
| `a4_driver.py` | `70dd520be4ddef238745c13dbdb4ade272d4b55c35d855e054a9b37ec99a95f5` |
| `a4_launch.py` | `6a85c1904d16e09c49756e12b3e0e3b6f2d74db326d51dc3b47a9ee8e259e33e` |

The `diff` line in `out/prelaunch.txt` reports a difference only because the two listings use
different whitespace; the hashes are identical.

- **§2.1 construction** (`out/check_construction.txt`): **HOLDS, 60 of 60.**
  - S's $G$ is a `phase_spline` (15 cells). P0's $G$ is a `PrimitivePhase` on a `TablePrimitive`
    with `exact_remainder` False (15), and P's the same with True (15). `_ProductionPhaseGk`'s own
    spline built everywhere, so the fallback was never taken.
  - Every realistic $T$ is a `PrimitivePhase` on a `TablePrimitive` with `exact_remainder` True,
    from `_UnionTableCsTauFixture` with its first sample dropped, anchored at $z_{\rm init}$ off
    a node.
  - $p_R$ runs from $-4.45\times10^{-3}$ (`q-smooth` 980, $T_q$) to $-3.64\times10^{-10}$
    (`together` $10^8$, $T_r$). It is positive for $T_r$ at 980 on `together`, `T-first`
    ($+3.63\times10^{-5}$) and `q-smooth` ($+3.96\times10^{-5}$).
- **§2.2 transparency** (`out/check_transparency.txt`): **HOLDS.** Arm S, `together` 980, wrapped
  against unwrapped:
  - `total` is `0x1.03eeb8b66f1f6p-40` both ways, and `total_abserr` is
    `0x1.ab7246e48e2b9p-66` both ways;
  - `numeric_quad` and `WKB_Levin` are bit-identical too;
  - the region count is 1,254 both ways, and the wrapper's own count is also 1,254.
- **§3 determinism** (`out/check_determinism.txt`): **HOLDS for every arm.** Two in-process runs
  of `together` 980 are bit-identical in `total`, `total_abserr` and every group's regions, depths,
  phase-limited count, value and abserr:

  | arm | regions | max depth |
  |---|---|---|
  | E | 7 | 0 |
  | S | 1,254 | 12 |
  | P0 | 53 | 6 |
  | P | 46 | 5 |

- **`state/readings.md`** is prompt §4, from its heading to the `---` before §5. It was taken
  from `git show HEAD:prompts/handover/04-gk-phase-decision-test.md`, checked to be a substring
  of it, and written at **15:26:53 UTC** (sha256 `d6f74ae64201…`, `out/readings_written.txt`).
  That is after the three checks and before the launch.
- **The launch** (`out/launch.txt`, `out/started_check.txt`):
  - the manifest was written at **15:27:04 UTC**, so `readings.md` is 11 s older;
  - run id `handover-04-cost-growth-20261026T152704`;
  - `sh` pid 7022 (parent 1), and the driver, a Python process, pid 7030, child of 7022;
  - the one check, at 15:27:10, showed `running` with 2 of 60 units, and the first checkpoint
    record is `E|together|980` `ok`, stamped with script `70dd520b…`, `git_head` `28ebaf07…` and
    `git_dirty` false.

  Load average at launch: 6.32 / 5.35 / 5.05 (`out/prelaunch.txt`).

---

## The run, and the third stop

Final `status.json`, read by the orchestrator at 16:08:04 UTC: `state` `failed`, `exit_code` 2,
60 of 60 units, last stage `cell S|together|1e+08 ok`, pid 7030. `summary.json` counts 34 `ok`,
6 `raised` and 20 `skipped`. No cell reached its 3 h budget, so **no cell is `not reached`**, and
§9's list of `not reached` cells is empty. The longest cell was `S|q-smooth|1e+05`, with
1,370.7 s of integral.

12. **STRUCTURALLY REQUIRED (blocking) — P0 and P raise in their $G$ phase's `tau` table.** The
   prompt assumes `_ProductionPhaseGk` built on `Case`'s $G$ grid gives a usable phase at every
   rung. The construction check (deviation 9's verification) built it at all 60 cells. But its
   table has a zero-width bottom interval at 13 of the 15 rungs:
   - the grid's last node `exp(log(1+z_resp)) - 1` is $z_{\rm resp}$ minus one ulp;
   - `_ProductionPhaseGk` adds $z_{\rm resp}$ itself to the node set;
   - the two nodes have the same $u = \log(1+z)$, so `CumulativeTable` refuses any abscissa one
     ulp below them.

   Six cells raised when the Levin driver formed such an abscissa: P and P0 at `q-smooth` 980,
   `together` $10^4$ and `T-first` $10^4$. Each gave `RuntimeError: CumulativeTable[tau]: z=…
   lies more than one grid interval below the table`. The driver skipped the 20 rungs above them,
   as §8 asks. The construction check could not see this, because it builds objects and evaluates
   nothing below $z_{\rm resp}$. Diagnosed by object-only probes (`probes/probe_tau_table_bottom.py`
   and `probes/probe_tau_table_margin.py`, outputs in `out/`). Nothing was repaired, no script
   changed and nothing was relaunched. `issues.md` item 2.
13. **IMPLEMENTATION CHOICE — how the memo is made.** `scripts/a4_tables.py`, written after the
   run, reads only the run's checkpoint and prints every table (`out/tables.md`). It labels each
   missing cell by its recorded status, `raised` or `skipped`. `memo-src/memo.template.md` holds
   the prose. `GK-PHASE-COST-GROWTH.md` is the template with the generated tables and
   `state/readings.md` (quoted) substituted in. `a4_tables.py` was formatted with `black`.

### Verification of the run's records

- Every one of the 60 checkpoint records carries:
  - driver SHA-256 `70dd520b…`;
  - `a4_cells` SHA-256 `8d7dc6ec…`, the text the checks ran;
  - launch `git_head` `28ebaf07…` with `git_dirty` false;
  - per-cell `tree_head` `28ebaf07…` and an empty `git status --short`.
- **Every `ok` cell's phase objects are what §2.1 says** (`construction_ok` true):

  | arm | cells | G phase | T phases |
  |---|---|---|---|
  | E | 15 | `_ExactPhase` | `_ExactPhase` |
  | S | 15 | `phase_spline` | `PrimitivePhase`, True, on `TablePrimitive` |
  | P0 | 2 | `PrimitivePhase`, False | `PrimitivePhase`, True, on `TablePrimitive` |
  | P | 2 | `PrimitivePhase`, True | `PrimitivePhase`, True, on `TablePrimitive` |

  $p_R$ is recorded for each fixture (memo Table I).
- **The checks' cells match the run's cells.** The run's `ok` cells at `together` 980 reproduce
  the determinism check's region counts and depths: E 7/0, S 1,254/12, P0 53/6, P 46/5.

### Acceptance (prompt §7)

1. Not met: ended `failed`, not `done`. Every cell is recorded, as `ok`, `raised` or `skipped`.
2. Met: the checks ran and were saved before the launch, and `readings.md` (15:26:53) is older
   than the manifest (15:27:04).
3. Met for every `ok` cell.
4. Partly met: §2.2's fields for every `ok` cell; growth for E and S; the level at 980 for all arms
   where they ran.
5. Not met for P: neither reading can be answered, and the memo §3 says so.
6. Recorded where P0 and P ran, and for S everywhere.
7. Met: nothing written outside `var/handover-04/` and the run directory; `realistic_large_x.py`
   still hashes `c1cd3598…`; no test file was touched.

### Observations not acted on (added)

2. `CumulativeTable.__init__` checks that nodes are strictly descending in $z$, not distinct in
   $u = \log(1+z)$. It accepts a zero-width interval and then refuses everything just beyond it.
   Whether production's background grid can produce such a pair was not checked
   (`issues.md` item 2).
3. Context only, not a reading: S's growth here (6.8 / 46.9 / 40.2) differs from A2's pre-B1
   realistic curve (12 / 105 / 37 over the rungs A2 reached). The two differ in $T$ and in the
   seam as well as in the tree. Memo Table F sets them side by side.

### State handed on

The run directory is `var/runs/handover-04-cost-growth-20261026T152704`. It holds the checkpoint,
`summary.json`, and `cells/` with each cell's JSON, stdout and stderr. The run is ended `failed`;
nothing needs `runregistrykit end`.

## The question for the user (third stop)

**Six P0/P cells raised in the test fixture's $G$ `tau` table, which has a zero-width bottom
interval, and twenty above them were skipped. So A4 has no growth for P, and its readings cannot
be answered. How should A4 proceed?** The options are in the report to the orchestrator.

---

## Run 2: the user chose (a), and the run relaunched

**The user's answer**, as the orchestrator relayed it: *"(a) Fix grid, rerun (Recommended)": "In A4's
own script, pass _ProductionPhaseGk the Case's G grid with its last node set exactly to z_resp,
removing the near-duplicate node; redo the construction, transparency and determinism checks and
readings.md; then a fresh full 60-cell registered run. This run stays as a record. The fixture
defect is a proposed issue."* The orchestrator added:
- extend the construction check to evaluate each P and P0 G phase a few ulp below z_resp and at
  the lowest abscissa the Levin driver can reach;
- keep the first run's `readings.md`, check outputs and run directory as records.

14. **STRUCTURALLY REQUIRED (the user's choice, answering deviation 12) — the G grid's last node is
   $z_{\rm resp}$.**
   - **The grid.** In `build_case`, P0 and P now pass `_ProductionPhaseGk` the `Case`'s G grid
     (still checked equal to `case.gk.WKB_region` first) with its last node replaced by
     `case.z_resp`, after checking that it lies within 4 ulp of it. That node moved by 1 ulp at 13
     of the 15 rungs and by 0 at `together` $10^8$ and `q-smooth` $10^6$ (recorded per cell as
     `G_grid.last_node_moved_by_ulp`). The `tau` table therefore holds $z_{\rm resp}$ once.
   - **What is unchanged.** S's `OffsetBesselPhaseGk`, its grid and the amplitude are untouched,
     and only `.phase` differs between S, P0 and P.
   - **The new check.** `a4_cells.below_anchor_check(case)` now runs in every cell before its
     integral, and in the construction check. In the bottom sub-interval, it evaluates
     `theta_mod_2pi`, `raw_theta` and `theta_deriv` (log-$z$) of every factor oscillatory there
     at these points:
     - $a = \log(1+z_{\min})$, formed as `phase_group_Levin_integral` forms it;
     - $a$ minus 1 to 8 ulp;
     - the lowest abscissa `adaptive_levin_sincos` can form there. `lowest_levin_abscissa` finds
       it through `chebyshev_matrices` itself, over the left bisection chain to depth 21 and every
       order the driver can use (24 down to 8 in steps of 2, and $2N-1$ for the fallback).

     A P0 or P cell whose G phase fails there is a construction failure.
   - **The check bites.** `probes/probe_below_anchor_bites.py` (`out/probe_below_anchor_bites.txt`)
     shows run 1's construction failing the new check at `together` $10^4$, `q-smooth` 980 and
     `together` 980 (8 or 9 failing points each, for P and P0), and passing at `together` $10^8$.
     The run-2 construction passes at all four.
15. **IMPLEMENTATION CHOICE — the check evaluates only the factors that are oscillatory in the bottom
   sub-interval's regime.** The first version evaluated every phase. At `q-smooth` 980 it flagged
   $T_q$, whose WKB phase ends at $z = 1.95$ (`out/probe_qsmooth_T_below.txt`): $T_q$ is smooth at
   $z_{\rm resp}$ on that shape (regime `Gr`), and its WKB phase is never evaluated there. That
   first output is kept as `out/probe_below_anchor_bites_v1.txt`.
16. **IMPLEMENTATION CHOICE — preserving run 1's records.**
   - Moved with `mv`, which keeps mtimes: `out/check_*`, `prelaunch`, `launch`, `started_check`,
     `scripts_sha256_before_checks`, `readings_written` and `tables` went to `out/run1/`, and the
     memo was renamed `GK-PHASE-COST-GROWTH-run1.md`.
   - Copied with `cp -p`: `state/readings.md` to `state/readings-run1.md`, and `scripts/` to
     `scripts-run1/` (`out/preserve_run1.txt`).
   - Run 1's directory under `var/runs/` was not touched.
17. **IMPLEMENTATION CHOICE — the launcher's purpose text names this as run 2** and says why, so a
   reader of `runregistrykit list` can tell the two runs apart. `a4_launch.py` changed only there.
   `a4_driver.py` is byte-identical to run 1's.

### Verification before run 2's launch

Script SHA-256 at the checks and at launch, identical (`out/scripts_sha256_before_checks.txt`,
`out/prelaunch.txt`):

| file | SHA-256 |
|---|---|
| `a4_cells.py` | `67badeab27e49e0f97a709ca67f8911d9663e0d8763e789c5946415cc9e64798` |
| `a4_checks.py` | `a8a71a6d52f6c6b1f744ee14f0ba0e97ddf25bf938a638891d4a1a7278d432fa` |
| `a4_driver.py` | `70dd520be4ddef238745c13dbdb4ade272d4b55c35d855e054a9b37ec99a95f5` |
| `a4_launch.py` | `cfa774dcf918d293f8f1c2033e71cf9117a454f7c6a82dfa29aee4754267d564` |

- **§2.1 construction, extended** (`out/check_construction.txt`): **HOLDS, 60 of 60.**
  - Phase classes and `exact_remainder` are as in run 1, and the fallback was never taken.
  - Every cell, all four arms, passes `below_anchor_check` with `G_ok` and `T_ok` true.
  - The lowest Levin abscissa sits 0, 1, 2 or 4 ulp below $a$ (8, 18, 11 and 23 cells
    respectively). Every phase evaluates at it and at $a - 1$ to $8$ ulp.
- **§2.2 transparency** (`out/check_transparency.txt`): **HOLDS.** On S, `together` 980, `total` is
  `0x1.03eeb8b66f1f6p-40` and `total_abserr` is `0x1.ab7246e48e2b9p-66`, wrapped and unwrapped;
  the region count is 1,254 both ways.
- **§3 determinism** (`out/check_determinism.txt`): **HOLDS for every arm** at `together` 980:

  | arm | regions | max depth |
  |---|---|---|
  | E | 7 | 0 |
  | S | 1,254 | 12 |
  | P0 | 53 | 6 |
  | P | 46 | 5 |

  P's `total` is bit-identical to run 1's at this rung. Its `total_abserr` differs in the last
  bits (`…227f3fdap-66`, against run 1's `…219d05bap-66`), because its $\phi$ sample at
  $z_{\rm resp}$ and its table changed.
- **`state/readings.md`**, prompt §4 copied from `HEAD`, was rewritten at **17:37:09 UTC**. It is
  identical in content to `readings-run1.md` (sha256 `d6f74ae64201…`). That is after the three
  checks and before the launch.
- **The launch** (`out/launch.txt`, `out/started_check.txt`):
  - manifest at **17:37:21 UTC**;
  - run id `handover-04-cost-growth-20261026T173720`;
  - `sh` pid 69642 (parent 1), and the driver, Python, pid 69649;
  - the one check, at 17:37:27, showed `running` with 2 of 60 units. The first checkpoint record
    is `E|T-first|980` `ok`, with script `70dd520b…`, `git_head` `28ebaf07…`, `git_dirty` false
    and `a4_cells_sha256` `67badeab…`.

  Load average at launch: **13.5 / 14.5 / 15.3**. Another job shares the host, which A4 permits, so
  times are indicative only.

---

## Run 2 ended: the result, and the records

Final `status.json`, read by the orchestrator at 21:23:37 UTC: `state` `done`, `exit_code` 0, 60 of
60 units, last stage `cell P0|together|1e+08 ok`, pid 69649. `summary.json` counts 60 `ok` with no
problems.
- **Cells not reached: none.** The longest cell was `P0|together|1e+08`, at 7,198 s of
  integral, under its 10,800 s budget. Next were `P0|together|1e+07` (4,950 s) and
  `P0|q-smooth|1e+06` (4,422 s).
- **Run directory:** `var/runs/handover-04-cost-growth-20261026T173720/` (checkpoint,
  `summary.json`, and `cells/` with each cell's JSON, stdout and stderr). It was only read here.

18. **IMPLEMENTATION CHOICE — `a4_tables.py` gained Table K, the attribution as ratios of growths.**
   It gives S/P₀ and P₀/P per shape, P/E, and the group carrying the maximum depth at each arm's
   top rung. Prompt §4 asks for exactly that, and run 1 never reached it. The script still reads
   only the checkpoint.
   - Its SHA-256 moved from `aa792008…` (run 1's, kept in `scripts-run1/`) to `4ae8f56a…`.
   - `out/tables.md` is now run 2's; run 1's is `out/run1/tables.md`.
19. **IMPLEMENTATION CHOICE — the memo is rebuilt from a new template.** It is
   `memo-src/memo-run2.template.md` with run 2's tables and `state/readings.md` substituted in.
   `memo-src/memo.template.md`, run 1's, is kept.
   - One claim in the template was corrected before assembly, after checking it against
     `out/run1_vs_run2_ok_cells.txt`: run 1's 30 E and S cells are identical to run 2's in
     `total`, `total_abserr`, regions and depth.
   - Run 1's four P and P₀ cells at 980 match run 2's in `total` and regions, but
     `total_abserr` differs in the last digits at three of them.
   - Five statements about depth were corrected against Table G.

### Verification of run 2's records

- **Provenance.** Every one of the 60 checkpoint records carries:
  - driver SHA-256 `70dd520b…` and `a4_cells` SHA-256 `67badeab…`, the text the checks ran;
  - launch `git_head` `28ebaf07…` with `git_dirty` false;
  - per-cell `tree_head` `28ebaf07…` and an empty `git status --short`.
- **Construction in every cell** (`out/run2_record_checks.txt`):
  - `construction_ok` is true in all 60, and `below_anchor` has `G_ok` and `T_ok` true in all 60;
  - P and P₀'s $G$ grid last node moved by 1 ulp in 26 cells and by 0 in 4.
- **Phase objects, by class and `exact_remainder`, as §2.1 says in every cell:**

  | arm | cells | G phase | T phases |
  |---|---|---|---|
  | E | 15 | `_ExactPhase` | `_ExactPhase` |
  | S | 15 | `phase_spline` | `PrimitivePhase`, True, on `TablePrimitive` |
  | P0 | 15 | `PrimitivePhase`, False | `PrimitivePhase`, True, on `TablePrimitive` |
  | P | 15 | `PrimitivePhase`, True | `PrimitivePhase`, True, on `TablePrimitive` |

- **The result** (memo §2–§3):

  | shape | growth E | growth S | growth P₀ | growth P | S/P₀ | P₀/P |
  |---|---|---|---|---|---|---|
  | `together` | 1.29 | 6.8 | 8.93e3 | 6.26 | 7.6e-4 | 1.43e3 |
  | `T-first` | 1.86 | 46.9 | 2.36e3 | 3.29 | 0.020 | 715 |
  | `q-smooth` | 0.94 | 40.2 | 619 | 2.16 | 0.065 | 287 |

  - **Level at 980, regions over E's:** S 179 / 102 / 49.4; P₀ 7.57 / 4.86 / 40.4; P 6.57 /
    4.86 / 40.4.
  - **Maximum depth:** P's is ≤14 everywhere. S and P₀ reach the cap of 20 on every shape.
- **The readings:** neither reading cleanly. P's growth lies between E's and S's on every shape,
  which is §4's own example. No reading was chosen, and no threshold was used.

### Acceptance (prompt §7), run 2

1. Met: one command (`a4_launch.py`) launched it from the main checkout's root; it is registered
   and ended `done`, with every cell recorded `ok`.
2. Met: the checks were saved before the launch. `readings.md` (17:37:09) is older than the
   manifest (17:37:21).
3. Met: every cell's record holds its phase objects, as the table above shows.
4. Met: §2.2's fields for every cell; growth per shape and arm (memo Tables B, K); the level at 980
   (memo Table C).
5. Met: "neither reading cleanly", with the attribution (memo §3).
6. Met: the representation term for S, P₀ and P in every cell, with both declared errors, and no
   verdict (memo §5).
7. Met: nothing written outside `var/handover-04/`, the two run directories and their launch logs;
   `realistic_large_x.py` still hashes `c1cd3598…`; no test file or production code was touched;
   `git status --short` is empty.

### Observations not acted on (added)

4. **The anchoring floor dominates the growth.** P₀, a `PrimitivePhase` $G$ without the node
   anchor, grows 287–1,430× more than P on these fixtures, and 15–1,300× more than A2's
   `phase_spline` $G$ (S). This is a measurement within A4's scope, not a new defect: production
   has had the node anchor since `levin-phase-floor` 05. It bears on how
   `[02-levin-cost-growth-may-be-a-stale-Gk-phase-artefact]` is narrowed, which is the user's call
   in the records commit.
5. **P still grows on `together` about as much as S** (6.26 against 6.8; P's regions peak at 313 at
   $10^7$), with its deepest regions in `2:Gqr:G+q+r` at depth 13–14 near $z' \sim 10^4$–$10^5$.
   Whatever drives that is not $G$'s phase construction alone, and is not examined here.
6. **No new issue from run 2.** The two fixture defects stay proposed in `issues.md` items 1 and 2.

### State handed on

For the laptop's records commit:
- **The scripts.** Copy `var/handover-04/` back, without checkpoints. The scripts that ran are
  `scripts/` (run 2) and `scripts-run1/` (run 1). Each is byte-identical to the SHA-256 its
  records carry.
- **The memo.** It goes to `docs/handover/GK-PHASE-COST-GROWTH.md`. Run 1's memo,
  `GK-PHASE-COST-GROWTH-run1.md`, is a record.
- **The readings.** Prompt §4's two readings and the user's decision on
  `[02-levin-cost-growth-…]` are the input to that commit.
