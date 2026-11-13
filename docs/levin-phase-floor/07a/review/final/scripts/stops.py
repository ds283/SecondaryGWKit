"""Orchestrator's own tally of window stops, 07 against h, over calls located at either (07a Part C §C.2). Read only
but for OUT_JSON. usage: python -I -B stops.py OUT_JSON W_RESCORE_JSON
Per rtol: windows and windows stopped at each tree; calls with a stopped window; returned window leaves flagged
phase-limited; for W7's violating calls (from my w_rescore.json), the stop transition and the move."""
import hashlib, json, sys
from collections import Counter
from pathlib import Path
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
R7A = MAIN / "var/runs/levin-phase-floor-07a-census-20261112T132744"; R07 = MAIN / "var/runs/levin-phase-floor-07-census-20261108T171137"
IX = MAIN / "var/levin-phase-floor-07a/review/stop3"
def rl(f, e):
    f.seek(e[0]); raw = f.read(e[1]); assert hashlib.sha256(raw).hexdigest() == e[2]; return json.loads(raw)["data"]
i7a = json.load(open(IX / "index07a.json")); i07 = json.load(open(IX / "index07.json"))
assert i7a["sha256"].startswith("461df2ec") and i07["sha256"].startswith("03dfe786")
W = json.load(open(sys.argv[2]))
viol = {(v[0], v[1], v[2]): v[3] for v in W["W7"]["violations"]}
T = Counter(); trans = Counter(); vt = []
f7a = open(R7A / "checkpoint.jsonl", "rb"); f07 = open(R07 / "checkpoint.jsonl", "rb")
for key in sorted(i7a["units"]):
    dh, d7 = rl(f7a, i7a["units"][key]), rl(f07, i07["units"][key]); rt = dh["rtol"]
    c7s = {(k, c["index"]): c for k in ("calls", "other_calls") for c in d7[k]}
    for kind in ("calls", "other_calls"):
        for ch in dh[kind]:
            c7 = c7s[(kind, ch["index"])]
            for nm, c in (("h", ch), ("07", c7)):
                g = c["guard_07"]
                T[f"{nm} {rt} windows"] += g["n_windows"]; T[f"{nm} {rt} windows_stopped"] += g["n_windows_stopped"]
                T[f"{nm} {rt} calls_with_stopped"] += g["n_windows_stopped"] > 0
                T[f"{nm} {rt} window_leaves_phase_limited"] += g["returned"]["n_window_phase_limited"]
                T[f"{nm} {rt} calls_with_stop_unconverged"] += (g["n_windows_stopped"] > 0 and not c["converged"])
            s7, sh = c7["guard_07"]["n_windows_stopped"] > 0, ch["guard_07"]["n_windows_stopped"] > 0
            if c7["guard_07"]["n_windows"] or ch["guard_07"]["n_windows"]:
                trans[f"{rt} stopped07={s7} stoppedh={sh}"] += 1
            if (key, kind, ch["index"]) in viol:
                vt.append((key, kind, ch["index"], viol[(key, kind, ch["index"])], s7, sh, ch["group"], c7["num_regions"], ch["num_regions"]))
res = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), totals=dict(sorted(T.items())),
           transitions=dict(sorted(trans.items())), w7_violations=vt,
           w7_viol_transitions=dict(Counter(f"stopped07={x[4]} stoppedh={x[5]}" for x in vt)))
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
