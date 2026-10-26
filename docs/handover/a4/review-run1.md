# Review — handover prompt 04 (A4), on the Mac Studio

**Verdict: STOP.** The run ended `failed`, not `done` (check 2): six P/P₀ cells raised in their
$G$ phase's `tau` table and the 20 rungs above them were skipped, so prompt §4's readings cannot be
answered for arm P. The agent has stopped at prompt §8 with a question for the user (its `log.md`,
"The question for the user (third stop)"). Every other check passes.

- Orchestrator: the Mac Studio session, reviewing per `prompts/handover/orchestrator/prompt-04.md` §3.
- Dispatch SHA: `28ebaf0743e0555fb2c4512ce714884005a60eff`. Marker touched 2026-10-26 14:43:37 UTC.
- Run: `handover-04-cost-growth-20261026T152704`, driver pid 7030, started 15:27:04, ended 16:07:47 UTC.
- Review taken 2026-10-26, about 16:15–16:30 UTC. Nothing repaired, nothing re-launched, nothing committed.

## Check 1 — Scope: PASS

```
$ git rev-parse HEAD; git status --short
28ebaf0743e0555fb2c4512ce714884005a60eff
(empty)
$ find var -newer <marker> -maxdepth 2 | grep -v -e '^var/handover-04' -e '^var/runs/handover-04-cost-growth-20261026T152704'
var
var/runs
$ find . -path ./.git -prune -o -path ./var -prune -o -newer <marker> -print
(nothing)
$ shasum -a 256 docs/handover/realistic_large_x.py
c1cd3598c23a52051a0eb2695bf67977265383b5fa02977fb61d8251c2cd98a3
```
`var` and `var/runs` appear only because their own mtimes changed when `var/handover-04/` and the
run directory were created. No other session's directory is newer than the marker.

## Check 2 — The run: **FAIL**

```
$ ./venv/bin/python -m runregistrykit list
handover-04-cost-growth-20261026T152704   failed   60/60
status.json: {"exit_code": 2, "state": "failed", "units_done": 60, "units_total": 60, "pid": 7030, "stage": "cell S|together|1e+08 ok"}
summary.json counts: ok 34, raised 6, skipped 20
```
- Manifest: `campaign` handover, `prompt` 04, run id `handover-04-cost-growth-…` (there is no
  separate `slug` key; the slug is in the run id), `git_dirty` false, `git_head` the dispatch SHA.
  **Holds.**
- 60 checkpoint records, 60 unique units, one driver hash `70dd520b…`, one `git_head`, all
  `git_dirty` false; every executed cell carries `a4_cells` hash `8d7dc6ec…` (the 20 skipped carry
  none, as they never ran). **Holds.**
- "Each `ok` or `not reached`": **fails.** Six are `raised`, twenty `skipped`; none `not reached`;
  no cell hit its 3-hour budget. The six:
  - `P|q-smooth|980`, `P0|q-smooth|980`: `RuntimeError: CumulativeTable[tau]: z=5.48046 lies more than one grid interval below the table (min node z=5.48046)`
  - `P|together|10000`, `P0|together|10000`, `P|T-first|10000`, `P0|T-first|10000`: the same at `z=6.0696`.

## Check 3 — The arms: PASS

Read in `var/handover-04/scripts/a4_cells.py` (`build_case`, `:210-277`):
- **S** is `Case`'s own `OffsetBesselPhaseGk`, unchanged (only $T$ is replaced via `Tk_builder`).
- **P₀ / P**: `_ProductionPhaseGk(shape.k, w, case.z_resp, case.model, grid)` on the grid `Case`
  hands `OffsetBesselPhaseGk` (asserted equal to `case.gk.WKB_region`); only
  `case.GkPolicy.functions.phase` is replaced, by `.old_phase` (P₀) or `.phase` (P).
- **$T$ in S, P₀, P**: `_UnionTableCsTauFixture(w, k, x_max, drop_first_WKB_sample=True)`, inputs by
  `exact_Tk_inputs`, built `TkSourceFunctions(fixture.exact_envelope_model(), k, Tn, Tw)` with no
  keyword. The subclass is the user's choice (ii) (log deviations 2, 8, 9): `cs_tau` on the union
  of numeric and WKB nodes minus the numeric node equal to $z_{\rm init}$.
- **E** is `Case(exact=True)`: `ExactGk` and the closed-form $T$.
- Tolerances are A2's imported `REF_ATOL`, `REF_RTOL`; the clamp reads are replaced by
  `partition["handover"]` (record field `handover`).

Independently, by a scratch reader (`review_cells.py`, below) that builds `together` 980 in each
arm and inspects the objects itself, against the checkpoint record:

| arm | $G$ phase (built) | record | $T_q$, $T_r$ (built) | record |
|---|---|---|---|---|
| E | none on `ExactGk` | `_ExactPhase` | closed form (`Case` lambda) | `_ExactPhase` |
| S | `LiouvilleGreen.phase_spline.phase_spline` | `phase_spline` | `PrimitivePhase`, `exact_remainder` True, on `TablePrimitive`, `_UnionTableCsTauFixture`, first sample dropped | same |
| P₀ | `PrimitivePhase`, `exact_remainder` **False**, `TablePrimitive` | same | as S | same |
| P | `PrimitivePhase`, `exact_remainder` **True**, `TablePrimitive` | same | as S | same |

## Check 4 — The wrapper: PASS

`GroupRecorder` (`a4_cells.py:382-420`) replaces `qsi_module.phase_group_Levin_integral`, calls
the original with the arguments unchanged, appends `groups` and `Levin_results` to a list, and
returns the payload unchanged; it restores the original on exit. `out/check_transparency.txt`
(S, `together` 980): `total` `0x1.03eeb8b66f1f6p-40` and `total_abserr` `0x1.ab7246e48e2b9p-66`
bit-identical wrapped and unwrapped; 1,254 regions both ways.

## Check 5 — Determinism and the readings: PASS

- `out/check_determinism.txt`: E, S, P₀, P at `together` 980, two runs each, `total`,
  `total_abserr`, every group's regions and depth bit-identical (regions 7, 1254, 53, 46).
- Times (full-iso mtimes): `check_construction.txt` 15:25:07, `check_transparency.txt` 15:25:52,
  `check_determinism.txt` 15:26:38, **`state/readings.md` 15:26:53**, run `manifest.json`
  15:27:04. Later than the checks, earlier than the manifest.
- Prompt §4's body (between its heading and the next `---`, 1,120 characters) is a substring of
  `state/readings.md`: `True`.

## Check 6 — Cells re-run by me: PASS

The note's example, `P|T-first|1e4`, raised in the run and has no values, so three cells were
re-run through the script's own `run_cell` in a scratch process:

```
RERUN P|T-first|980   ok  total 7.469443338319054e-11 ==, total_abserr 2.579184425459271e-19 ==, Levin_regions 34 ==, max_depth 7 ==, max_depth_group ==   (all True)
RERUN S|T-first|10000 ok  total 7.758909092836051e-13 ==, total_abserr 2.2569324126592278e-21 ==, Levin_regions 3050 ==, max_depth 17 ==, max_depth_group ==   (all True)
RERUN P|together|10000 raised RuntimeError CumulativeTable[tau]: z=6.0696 lies more than one grid interval below the table (min node z=6.0696)   (record: the same)
```
Bit-identical where it ran; the failure reproduces with the recorded message.

## Check 7 — The numbers, re-derived from the checkpoint: PASS

Regions / maximum depth per rung (980 upward):

| shape | E | S | P₀ | P |
|---|---|---|---|---|
| together | 7/0 8/1 8/1 9/2 9/2 9/2 | 1254/12 4943/19 34606/20 50678/20 58532/20 8526/20 | 53/6, raised, 4 skipped | 46/5, raised, 4 skipped |
| T-first | 7/0 10/2 9/1 12/3 13/4 | 716/11 3050/17 11410/20 12148/20 33570/20 | 34/7, raised, 3 skipped | 34/7, raised, 3 skipped |
| q-smooth | 17/8 21/6 21/6 16/3 | 839/11 8528/18 88593/20 33752/20 | raised, 3 skipped | raised, 3 skipped |

- Growth (highest rung over 980): E 1.29 / 1.86 / 0.941; S 6.8 / 46.9 / 40.2; P₀ and P undefined.
- Level at 980 over E: S 179 / 102 / 49.4; P₀ 7.57 / 4.86 / raised; P 6.57 / 4.86 / raised.
- `not reached`: none.

These match the memo's Tables B and C and §3. The memo's §3 says neither reading can be answered
for P from these cells, chooses none, forms no attribution, and invents no threshold. That is the
honest reading of prompt §4 as written. (Table B lists P/P₀ "growth 1" at "highest rung ok 980";
its "rungs ok 1 of 6" column and §3 make plain that this is not a growth.)

## Check 8 — The representation term: PASS

Memo §5, Table H: $N_{\rm arm}-N_{\rm E}$ for S, P₀, P per cell with both declared errors; missing
cells marked `raised`/`skipped`; no verdict.

## Check 9 — The records: PASS

- `log.md`: campaign README §5.1 sections; 13 numbered deviations, each classified
  (`UNINTENDED DRIFT` 1; `STRUCTURALLY REQUIRED` 2, 8, 9, 12; `IMPLEMENTATION CHOICE` 3–7, 10, 11,
  13); status STOPPED; the user's two choices quoted (2, 8, 9); the third question at its end.
- The memo has prompt §5's sections §0–§6.
- `issues.md` exists: item 1, `_TableCsTauFixture`'s table does not reach $z_{\rm init}$ with
  `drop_first_WKB_sample=True` (and the numeric grid ends on $z_{\rm init}$); item 2, the
  zero-width bottom interval in `_ProductionPhaseGk`'s `tau` table.

## The scratch reader used for checks 3 and 6

Run as `PYTHONPATH=. ./venv/bin/python -B review_cells.py` from the repository root; it writes
nothing.

```python
import sys, json
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/handover-04/scripts")
import a4_cells as C
from ComputeTargets.primitive_phase import PrimitivePhase
CK = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/handover-04-cost-growth-20261026T152704/checkpoint.jsonl"
recs = {json.loads(l)["unit"]: json.loads(l)["data"] for l in open(CK)}
print("a4_cells sha", C.SCRIPT_SHA256)
# check 3: independent inspection of built objects, one cell per arm
base = {s.name: s for s in C.SHAPES}
for arm in ("E", "S", "P0", "P"):
    case, built = C.build_case(base["together"], 980.0, arm)
    G = case.GkPolicy.functions.phase if not case.exact else None
    def desc(ph):
        if ph is None: return None
        lead = getattr(ph, "_leading", None)
        return (type(ph).__module__ + "." + type(ph).__name__, getattr(ph, "exact_remainder", None),
                None if lead is None else type(lead).__name__)
    gk = type(case.gk).__name__ if hasattr(case, "gk") else None
    out = {"G_phase": desc(G), "gk_obj": gk}
    if built["T"]:
        for e in built["T"].values():
            out[e["label"]] = desc(e["T"].phase) + (type(e["fixture"]).__name__, e["fixture"].z_switch != e["fixture"].crossover_z)
    else:
        out["Tk_builder"] = getattr(case.Tk_builder, "__qualname__", str(case.Tk_builder))
    rec = recs[f"{arm}|together|980"]["phase_objects"]
    print(arm, json.dumps(out, default=str))
    print("   record:", json.dumps({k: (v.get("class"), v.get("exact_remainder"), v.get("leading_class")) if isinstance(v, dict) else v for k, v in rec.items() if k in ("G","Tq","Tr")}))
# check 6: re-run cells, compare
for shape, x, arm in [("T-first", 980.0, "P"), ("T-first", 1.0e4, "S"), ("together", 1.0e4, "P")]:
    unit = f"{arm}|{shape}|{x:g}" if x != 980.0 else f"{arm}|{shape}|980"
    r0 = recs.get(unit) or recs.get(f"{arm}|{shape}|{int(x)}")
    try:
        r = C.run_cell(shape, x, arm)
        cmp = {k: (r.get(k), r0.get(k), r.get(k) == r0.get(k)) for k in ("total", "total_abserr", "Levin_regions", "max_depth", "max_depth_group")}
        print("RERUN", unit, r.get("status"), json.dumps(cmp, default=str))
    except Exception as exc:
        print("RERUN", unit, "raised", type(exc).__name__, str(exc)[:200], "| record:", r0.get("status"), str(r0.get("error", r0.get("exception","")))[:200])
```
