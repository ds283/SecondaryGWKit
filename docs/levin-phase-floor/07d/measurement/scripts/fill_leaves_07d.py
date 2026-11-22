"""levin-phase-floor prompt 07d Part A, sections 2.2-2.4: fill each leaf stratum in its seeded order
(state/selection_leaves_07d.json) until it holds FILL scored leaves or is exhausted. New in 07d.

Rounds: round 1 measures the first FIRST leaves of every stratum; each later round, for each stratum still short of FILL
scored leaves with leaves left, measures the next ceil(1.3 (FILL - scored) measured / max(scored, 1)) leaves (at least
MIN_NEXT), in order, up to CAP measured a stratum (reported if reached). A leaf is **scored** if the verdict of the
tree's declaration on it is not undecided (state/refs_rule_leaf_07d.txt): for a Clenshaw-Curtis stratum, max(pair,
phase_err) on CC_{2N-1}; for a Levin band (above 6 pi, where the tree's route is the Levin solve), the Levin solve's own
declaration on V_P. The tables (d_e_table_07d.py, d_t_table_07d.py) use each stratum's leaves in order up to and
including its FILL-th scored leaf; leaves measured past that point in its last round are reported as measured past
the fill, and no table counts them.
Each round's leaves are grouped by call into leafcc_07d.py specs (out/leaves/specs/round<r>/) and run six at once
(outputs out/leaves/round<r>/). Writes out/leaves/fill_state.json after every round.
Usage: PYTHONPATH=<tree>:<scripts> python -B fill_leaves_07d.py [--max-rounds R]
"""

import json
import math
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07d-49ce964"
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
FILL, FIRST, MIN_NEXT, CAP, WORKERS = 40, 60, 20, 600, 6
fx = float.fromhex


def fname(sid):
    return sid.replace(":", "_").replace("|", "_")


def verdict(D, v, ref, eps):
    te = abs(v - ref)
    if D >= te + eps:
        return "bounded"
    if D < te - eps:
        return "understates"
    return "undecided"


def tree_verdict(st, l):
    if "error" in l:
        return "error"
    ref, eps = fx(l["ref"]), fx(l["eps_ref"])
    if st.startswith("band_"):
        return verdict(fx(l["levin"]["own_decl"]), fx(l["levin"]["vP"]), ref, eps)
    return verdict(max(fx(l["pair"]), fx(l["floor"])), fx(l["cc47"]), ref, eps)


def main():
    max_rounds = int(sys.argv[sys.argv.index("--max-rounds") + 1]) if "--max-rounds" in sys.argv else 8
    sel = json.loads((ROOT / "state" / "selection_leaves_07d.json").read_text())["selection"]
    man = {}
    for p in sorted((ROOT / "state").glob("calls_07d*.json")):
        man.update(json.loads(p.read_text())["calls"])
    cen = {}
    for p in sorted((ROOT / "state").glob("census_refs_07d*.pkl")):
        cen.update(pickle.loads(p.read_bytes())["calls"])
    fs_p = ROOT / "out" / "leaves" / "fill_state.json"
    state = json.loads(fs_p.read_text()) if fs_p.exists() else {"rounds": [], "strata": {st: {"measured": 0, "scored": 0, "results": []} for st in sel}}
    env = dict(os.environ, PYTHONPATH=f"{TREE}:{HERE}", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    if "--repair" in sys.argv:
        # Repair (added after round 2): every leaf recorded "missing" or "error" (a spec that exited nonzero, or a leaf
        # whose measurement raised) is measured again, in place in its stratum's order, with leafcc_07d.py as it now is;
        # its result replaces the old one and the scored counts are recounted. Rounds then continue as before.
        rep_n = sum(1 for k in state.get("repairs", [])) + 1
        todo_r = {}
        for st, S in state["strata"].items():
            for res in S["results"]:
                if res["verdict"] in ("missing", "error"):
                    todo_r.setdefault(st, []).append(res["rank"])
        spec_dir = ROOT / "out" / "leaves" / "specs" / f"repair{rep_n}"
        out_dir = ROOT / "out" / "leaves" / f"repair{rep_n}"
        spec_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "logs").mkdir(parents=True, exist_ok=False)
        by_sid = {}
        for st, ranks in todo_r.items():
            for rk in ranks:
                l = sel[st]["order"][rk]
                by_sid.setdefault(l["sid"], []).append({"id": f"{st}#{rk}", "start": l["start"], "end": l["end"],
                                                        "type": l["type"], "value_own": l["value_own"]})
        for sid, leaves in by_sid.items():
            c = man[sid]
            cr = cen[c["census"]]["call"]
            (spec_dir / f"{fname(sid)}.json").write_text(json.dumps({
                "sid": sid, "capture": {k: c[k] for k in ("path", "offset", "length", "sha256")}, "rtol": cr["rtol"],
                "atol": cr["atol"], "call_value": cr["value"]["hex"], "x_span": [cr["x_span"][0]["hex"], cr["x_span"][1]["hex"]],
                "leaves": leaves}))
        pending = sorted(spec_dir.glob("*.json"))
        running, rc = {}, {}
        t0 = time.time()
        while pending or running:
            while pending and len(running) < WORKERS:
                sp = pending.pop(0)
                log = open(out_dir / "logs" / f"{sp.stem}.log", "w")
                running[sp.stem] = subprocess.Popen([PY, "-B", str(HERE / "leafcc_07d.py"), str(sp), str(out_dir / sp.name)],
                                                    cwd=TREE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                log.close()
            for k_, p in list(running.items()):
                if p.poll() is not None:
                    rc[k_] = p.returncode
                    del running[k_]
            time.sleep(0.2)
        got = {}
        for op in out_dir.glob("*.json"):
            for l in json.loads(op.read_text())["leaves"]:
                got[l["id"]] = (op.name, l)
        n_fixed = 0
        for st, ranks in todo_r.items():
            S = state["strata"][st]
            for res in S["results"]:
                if res["rank"] in ranks:
                    key = f"{st}#{res['rank']}"
                    if key in got:
                        res["verdict"] = tree_verdict(st, got[key][1])
                        res["file"] = f"repair{rep_n}/{got[key][0]}"
                        res["repaired"] = rep_n
                        n_fixed += res["verdict"] not in ("missing", "error")
            S["scored"] = sum(1 for r in S["results"] if r["verdict"] in ("bounded", "understates"))
        state.setdefault("repairs", []).append({"repair": rep_n, "n_leaves": sum(len(v) for v in todo_r.values()),
                                                "n_fixed": n_fixed, "nonzero_exits": {k: v for k, v in rc.items() if v},
                                                "wall_s": time.time() - t0})
        fs_p.write_text(json.dumps(state, indent=1))
        print(f"repair {rep_n}: {sum(len(v) for v in todo_r.values())} leaves in {len(rc)} calls, {n_fixed} now measured, "
              f"nonzero exits { {k: v for k, v in rc.items() if v} }", flush=True)
    for rnd in range(len(state["rounds"]) + 1, max_rounds + 1):
        todo = {}
        for st, s in sel.items():
            S = state["strata"][st]
            n_left = len(s["order"]) - S["measured"]
            if S["scored"] >= FILL or n_left <= 0 or S["measured"] >= CAP:
                continue
            if S["measured"] == 0:
                k = FIRST
            else:
                k = max(MIN_NEXT, math.ceil(1.3 * (FILL - S["scored"]) * S["measured"] / max(S["scored"], 1)))
            k = min(k, n_left, CAP - S["measured"])
            todo[st] = list(range(S["measured"], S["measured"] + k))
        if not todo:
            break
        spec_dir = ROOT / "out" / "leaves" / "specs" / f"round{rnd}"
        out_dir = ROOT / "out" / "leaves" / f"round{rnd}"
        spec_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "logs").mkdir(parents=True, exist_ok=False)
        by_sid = {}
        for st, ranks in todo.items():
            for rk in ranks:
                l = sel[st]["order"][rk]
                by_sid.setdefault(l["sid"], []).append({"id": f"{st}#{rk}", "start": l["start"], "end": l["end"],
                                                        "type": l["type"], "value_own": l["value_own"]})
        for sid, leaves in by_sid.items():
            c = man[sid]
            cr = cen[c["census"]]["call"]
            spec = {"sid": sid, "capture": {k: c[k] for k in ("path", "offset", "length", "sha256")},
                    "rtol": cr["rtol"], "atol": cr["atol"], "call_value": cr["value"]["hex"],
                    "x_span": [cr["x_span"][0]["hex"], cr["x_span"][1]["hex"]], "leaves": leaves}
            (spec_dir / f"{fname(sid)}.json").write_text(json.dumps(spec))
        pending = sorted(spec_dir.glob("*.json"), key=lambda p: -len(json.loads(p.read_text())["leaves"]))
        running, rc = {}, {}
        t0 = time.time()
        while pending or running:
            while pending and len(running) < WORKERS:
                sp = pending.pop(0)
                log = open(out_dir / "logs" / f"{sp.stem}.log", "w")
                running[sp.stem] = subprocess.Popen([PY, "-B", str(HERE / "leafcc_07d.py"), str(sp), str(out_dir / sp.name)],
                                                    cwd=TREE, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                log.close()
            for k_, p in list(running.items()):
                if p.poll() is not None:
                    rc[k_] = p.returncode
                    del running[k_]
            time.sleep(0.2)
        bad = {k_: v for k_, v in rc.items() if v != 0}
        got = {}
        for op in out_dir.glob("*.json"):
            for l in json.loads(op.read_text())["leaves"]:
                got[l["id"]] = (op.name, l)
        for st, ranks in todo.items():
            S = state["strata"][st]
            for rk in ranks:
                key = f"{st}#{rk}"
                if key not in got:
                    v = "missing"
                    S["results"].append({"rank": rk, "verdict": v, "file": None})
                else:
                    v = tree_verdict(st, got[key][1])
                    S["results"].append({"rank": rk, "verdict": v, "file": f"round{rnd}/{got[key][0]}"})
                S["measured"] += 1
                S["scored"] += v in ("bounded", "understates")
        state["rounds"].append({"round": rnd, "n_leaves": sum(len(v) for v in todo.values()), "n_specs": len(rc),
                                "nonzero_exits": bad, "wall_s": time.time() - t0})
        fs_p.write_text(json.dumps(state, indent=1))
        print(f"round {rnd}: {sum(len(v) for v in todo.values())} leaves in {len(rc)} calls, {time.time() - t0:.0f} s, "
              f"nonzero exits {bad}", flush=True)
        for st in sorted(todo):
            S = state["strata"][st]
            print(f"   {st}: measured {S['measured']} of {len(sel[st]['order'])}, scored {S['scored']}", flush=True)
    print("final:")
    for st in sorted(sel):
        S = state["strata"][st]
        flag = "filled" if S["scored"] >= FILL else ("exhausted" if S["measured"] >= len(sel[st]["order"]) else ("CAP" if S["measured"] >= CAP else "short"))
        print(f"   {st}: population {len(sel[st]['order'])}, measured {S['measured']}, scored {S['scored']}: {flag}")


if __name__ == "__main__":
    sys.exit(main())
