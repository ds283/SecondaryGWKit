# Review — handover prompt 04 (A4), run 2, on the Mac Studio

**Verdict: PASS.** Run 2 ended `done` with all 60 cells `ok`; every check of
`prompts/handover/orchestrator/prompt-04.md` §3 holds. The memo answers prompt §4 with **"Neither
reading cleanly"**, chooses no reading, and leaves it to the user.

- Orchestrator: the Mac Studio session. Dispatch SHA `28ebaf0743e0555fb2c4512ce714884005a60eff`.
- Run 1, `handover-04-cost-growth-20261026T152704`, ended `failed` (six P/P₀ cells raised in
  `_ProductionPhaseGk`'s `tau` table); its review, verdict STOP, is `review-run1.md`. The user chose
  (a) at the third stop: fix the G grid in A4's own script and re-run in full.
- **Second marker** touched 2026-10-26 17:31:21 UTC, before the agent was resumed for run 2. The
  review below starts again from check 1 against it.
- Run 2: `handover-04-cost-growth-20261026T173720`, driver pid 69649, manifest 17:37:21, ended
  `done` at 21:20:08 UTC. Review taken about 21:30–21:50 UTC. Nothing committed.

## Check 1 — Scope: PASS

```
$ git rev-parse HEAD; git status --short
28ebaf0743e0555fb2c4512ce714884005a60eff
(empty)
$ find var -newer <marker-2> -maxdepth 2 | grep -v -e '^var/handover-04' -e '^var/runs/handover-04-cost-growth-20261026T173720'
var/runs
$ find . -path ./.git -prune -o -path ./var -prune -o -newer <marker-2> -print
(nothing)
$ find var/runs/handover-04-cost-growth-20261026T152704 -newer <marker-2>
(nothing — run 1's directory untouched)
$ shasum -a 256 docs/handover/realistic_large_x.py
c1cd3598c23a52051a0eb2695bf67977265383b5fa02977fb61d8251c2cd98a3
```
`var/runs` appears only because run 2's directory was created in it.

## Check 2 — The run: PASS

```
$ ./venv/bin/python -m runregistrykit list
handover-04-cost-growth-20261026T173720   done   60/60
status.json: {"exit_code":0,"state":"done","units_done":60,"units_total":60,"pid":69649,"heartbeat":"2026-10-26T21:20:08+00:00"}
summary.json: {'ok': 60}, problems []
```
- Manifest: campaign `handover`, prompt `04`, run id `handover-04-cost-growth-…`, `git_dirty`
  false, `git_head` the dispatch SHA, driver hash `70dd520b…` (byte-identical to run 1's driver).
- 60 checkpoint records, 60 unique units, all `ok`; one driver hash, one `git_head`, every
  `git_dirty` false and `tree_status_short` empty; every record carries `a4_cells` `67badeab…`.
- No cell over its 3-hour budget; the longest, `P0|together|1e8`, 7,198 s integral (7,203 s wall).

## Check 3 — The arms: PASS

Read in `scripts/a4_cells.py`, and its diff against run 1's copy in `scripts-run1/`:
- **S**, **E** and the **$T$ construction** (`_UnionTableCsTauFixture`, `drop_first_WKB_sample=True`,
  `TkSourceFunctions(…)` with no keyword) are unchanged from run 1 and as prompt §2.1 and the user's
  choices (ii) say.
- **P₀ / P**: the grid `Case` hands `OffsetBesselPhaseGk` (still asserted equal to
  `case.gk.WKB_region`), then — the user's choice (a) — its last node set exactly to `z_resp`
  (guarded to be within 4 ulp, and the grid still strictly descending), passed to
  `_ProductionPhaseGk`; only `case.GkPolicy.functions.phase` is replaced (`.old_phase` / `.phase`).
- New: `below_anchor_check` evaluates each phase oscillatory in the bottom sub-interval at
  $\log(1+z_{\rm resp})$, 1–8 ulp below, and at the lowest abscissa `chebyshev_matrices` gives over
  the bisection chain to depth 21; a P/P₀ $G$ failure is a construction failure. (Deviation 15: $T$
  is tested only where oscillatory; a $T$ failure is recorded but not a construction failure. The
  user's instruction named $G$.) All 60 records: `construction_ok` True, `below_anchor.G_ok` True.
- Tolerances `REF_ATOL`, `REF_RTOL` imported from A2; clamp reads replaced by `partition["handover"]`.

Independently, by a scratch reader (`review_cells_run2.py`, below) at `together` $10^4$, the rung
where run 1 raised:

| arm | $G$ phase (built) | record | $T_q$, $T_r$ (built) | record |
|---|---|---|---|---|
| E | none on `ExactGk` | `_ExactPhase` | closed form | `_ExactPhase` |
| S | `phase_spline` | `phase_spline` | `PrimitivePhase`, `exact_remainder` True, `TablePrimitive`, `_UnionTableCsTauFixture`, first sample dropped | same |
| P₀ | `PrimitivePhase`, `exact_remainder` False, `TablePrimitive` | same | as S | same |
| P | `PrimitivePhase`, `exact_remainder` True, `TablePrimitive` | same | as S | same |

For P₀ and P it also found the `tau` table's bottom node equal to `z_resp`, its nodes distinct in
$\log(1+z)$, and the phase evaluating 2 ulp below $\log(1+z_{\rm resp})$.

## Check 4 — The wrapper: PASS

`GroupRecorder` is unchanged from run 1 (calls the original with unchanged arguments, returns its
payload unchanged, restores it on exit). `out/check_transparency.txt` (S, `together` 980): `total`
`0x1.03eeb8b66f1f6p-40`, `total_abserr` `0x1.ab7246e48e2b9p-66`, bit-identical wrapped and
unwrapped; 1,254 regions both ways.

## Check 5 — Determinism and the readings: PASS

- `out/check_determinism.txt`: E, S, P₀, P at `together` 980, two runs each, bit-identical in
  `total`, `total_abserr`, every group's regions and depth (7, 1254, 53, 46).
- mtimes (UTC): `check_construction.txt` 17:35:29, `check_transparency.txt` 17:36:12,
  `check_determinism.txt` 17:37:00, **`state/readings.md` 17:37:09**, run 2 `manifest.json`
  17:37:21. Later than the checks, earlier than the manifest.
- Prompt §4's body is a substring of `state/readings.md`: `True`. Identical to run 1's
  `state/readings-run1.md` (15:26:53, kept).

## Check 6 — Cells re-run by me: PASS

Through the script's own `run_cell`, in a scratch process, against run 2's checkpoint:
```
RERUN P|T-first|10000   ok  total 7.758908970249765e-13, total_abserr 2.2564546320219925e-21, Levin_regions 44, max_depth 7, max_depth_group — all equal
RERUN P|q-smooth|10000  ok  total 3.90633988609251e-15,  total_abserr 6.756172589294017e-25, Levin_regions 1191, max_depth 14, max_depth_group — all equal
RERUN P0|together|10000 ok  total 1.732808095777418e-14, total_abserr 4.561104988065042e-23, Levin_regions 78, max_depth 8, max_depth_group — all equal
```
`P|T-first|10000` is the note's example; the other two are rungs where run 1 raised.

## Check 7 — The numbers, re-derived from the checkpoint: PASS

Regions / maximum depth, 980 upward:

| shape | E | S | P₀ | P |
|---|---|---|---|---|
| together | 7/0 8/1 8/1 9/2 9/2 9/2 | 1254/12 4943/19 34606/20 50678/20 58532/20 8526/20 | 53/6 78/8 1551/19 56451/20 308667/20 473552/20 | 46/5 66/6 80/8 102/9 313/14 288/13 |
| T-first | 7/0 10/2 9/1 12/3 13/4 | 716/11 3050/17 11410/20 12148/20 33570/20 | 34/7 44/7 62/7 607/17 80117/20 | 34/7 44/7 51/7 45/7 112/9 |
| q-smooth | 17/8 21/6 21/6 16/3 | 839/11 8528/18 88593/20 33752/20 | 687/11 1540/14 31814/20 425137/20 | 686/11 1191/14 1204/14 1480/14 |

Growth (top rung over 980), and the attribution as ratios of growths:

| shape | E | S | P₀ | P | S/P₀ | P₀/P | P/E |
|---|---|---|---|---|---|---|---|
| together | 1.286 | 6.799 | 8935 | 6.261 | 7.61e-4 | 1427 | 4.87 |
| T-first | 1.857 | 46.89 | 2356 | 3.294 | 0.0199 | 715.3 | 1.77 |
| q-smooth | 0.941 | 40.23 | 618.8 | 2.157 | 0.065 | 286.8 | 2.29 |

Level at 980 over E: S 179 / 102 / 49.4; P₀ 7.57 / 4.86 / 40.4; P 6.57 / 4.86 / 40.4.
Deepest group at the top rung: P₀ and P share it on each shape (`G+q+r`, `G+q-r`, `G-r`).
`not reached`: none.

These match the memo's Tables B, C, K and §3. **The memo's §3 against prompt §4 as written:** it
says "Neither reading cleanly": P is not flat like E (4.9×, 1.8×, 2.3× E's growth), its growth lies
between E's and S's on every shape — §4's own example of "neither" — and the shapes disagree
(`together` P ≈ S; `T-first` and `q-smooth` P 14× and 19× below S). It gives the attribution per
shape with the carrying group, uses no threshold, and leaves the reading to the user. That is what
§4 asks.

## Check 8 — The representation term: PASS

Memo §5, Table H: $N_{\rm arm}-N_{\rm E}$ for S, P₀ and P in every cell, with both cells' declared
errors; the only "verdict" in it is the heading's "no verdict".

## Check 9 — The records: PASS

- `log.md`: campaign README §5.1 sections, status **COMPLETE WITH DEVIATIONS**; 19 numbered
  deviations, each classified (`UNINTENDED DRIFT` 1; `STRUCTURALLY REQUIRED` 2, 8, 9, 12, 14;
  `IMPLEMENTATION CHOICE` 3–7, 10, 11, 13, 15–19); the user's three choices quoted (2/8/9, 14);
  run 1's numbers kept as the failed run's record; acceptance for run 2, all seven met.
- The memo has prompt §5's sections §0–§6.
- `issues.md` exists: two proposed issues (`[04-table-cs-tau-fixture-does-not-cover-z-init-…]`,
  `[04-production-phase-gk-tau-table-has-a-zero-width-bottom-interval]`), no new one from run 2, and
  notes for the narrowings of the two `02-` issues and `[12-…]`.
- Run 1's records are kept apart: `GK-PHASE-COST-GROWTH-run1.md`, `scripts-run1/`, `out/run1/`,
  `state/readings-run1.md`, `review-run1.md`, and its run directory.

## Notes for the laptop (not failures)

- The load average ran 5.4–21.1 during run 2; another job shared the host. Times are indicative
  only; region counts and depths are the measurement (prompt §3).
- `docs/handover/a4/scripts/` should carry run 2's scripts; run 1's are in `scripts-run1/`, and
  `scripts/a4_tables.py` changed after the run (deviation 18, reads only the checkpoint).

## The scratch reader used for checks 3 and 6

Run as `PYTHONPATH=. ./venv/bin/python -B review_cells_run2.py` from the repository root; it
writes nothing.

```python
import sys, json
from math import log1p
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/handover-04/scripts")
import a4_cells as C
CK = "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/handover-04-cost-growth-20261026T173720/checkpoint.jsonl"
recs = {json.loads(l)["unit"]: json.loads(l)["data"] for l in open(CK)}
print("a4_cells sha", C.SCRIPT_SHA256)
base = {s.name: s for s in C.SHAPES}
def desc(ph):
    if ph is None: return None
    lead = getattr(ph, "_leading", None)
    return (type(ph).__module__ + "." + type(ph).__name__, getattr(ph, "exact_remainder", None),
            None if lead is None else type(lead).__name__)
# check 3: one cell per arm, at together 1e4 (where run 1 raised)
for arm in ("E", "S", "P0", "P"):
    case, built = C.build_case(base["together"], 1.0e4, arm)
    out = {"gk_obj": type(case.gk).__name__, "G_phase": None if case.exact else desc(case.GkPolicy.functions.phase)}
    if built["T"]:
        for e in built["T"].values():
            out[e["label"]] = desc(e["T"].phase) + (type(e["fixture"]).__name__, bool(e["fixture"].z_switch != e["fixture"].crossover_z))
    if arm in ("P0", "P"):
        ph = case.GkPolicy.functions.phase
        z = list(ph._leading._table._z)
        L = [log1p(v) for v in z]
        out["tau_bottom_node==z_resp"] = z[-1] == case.z_resp
        out["tau_nodes_distinct_in_log1p"] = len(set(L)) == len(L)
        # evaluate below z_resp
        import numpy as np
        a = log1p(case.z_resp); x = float(np.nextafter(np.nextafter(a, -np.inf), -np.inf))
        ph.theta_mod_2pi(x, x_is_log=True); out["evaluates_2ulp_below"] = True
    rec = recs[f"{arm}|together|10000"]["phase_objects"]
    print(arm, json.dumps(out, default=str))
    print("   record:", json.dumps({k: (v.get("class"), v.get("exact_remainder"), v.get("leading_class")) for k, v in rec.items() if k in ("G","Tq","Tr")}))
# check 6
for shape, x, arm, unit in [("T-first", 1.0e4, "P", "P|T-first|10000"), ("q-smooth", 1.0e4, "P", "P|q-smooth|10000"), ("together", 1.0e4, "P0", "P0|together|10000")]:
    r0 = recs[unit]; r = C.run_cell(shape, x, arm)
    cmp = {k: (r.get(k), r0.get(k), r.get(k) == r0.get(k)) for k in ("total", "total_abserr", "Levin_regions", "max_depth")}
    print("RERUN", unit, r.get("status"), "max_depth_group equal:", r.get("max_depth_group") == r0.get("max_depth_group"), json.dumps(cmp))
```
