"""depth20-tiling step 2 (levin-phase-floor 06c follow-up): capture the inputs of the sampled
depth-20 calls by the census's own route. Copied from the census's scripts/x1_capture.py
(var/levin-phase-floor-06c/census/scripts/x1_capture.py, sha256 dafe695a..., a record, read
only); changed here only where marked "depth20-tiling": PYTHONPATH is <tree g>:<this scripts
dir> (the brief's rule for every Python command), given to the workers too; WIDER (prov.py) is
this task's directory, so state/rows.json is this task's copy of the census's (equal SHA-256).
The census's text follows, unchanged.

levin-phase-floor prompt 06c section 5, X1: capture the inputs of the selected calls, after the
run. New in 06c.

X1's 24 calls are known only once g has run ("moved" compares g with 06b), and capturing every
candidate's inputs during the run, as 06b's W2 did for its 975, would take about 1.35 MB per call
over tens of thousands of candidates. So the run captures nothing, and this script re-runs, at g,
each selected call's unit in a fresh census worker (worker.py, the producer cache, the same 300 s
budget, the scan and W3 as in the run), with the unit's ``"capture"`` list naming the selected
calls. The worker pickles each listed call's span, integrand ``f``, phase dict ``theta`` and
keyword arguments as the call saw them, with its value, abserr, num_regions, max_depth and
converged. The capture **counts only if** that record equals the census run's call bit for bit
(value and abserr by hex; num_regions, max_depth, converged), which Gate X0's third clause (the new
code is deterministic) leads one to expect; a call whose capture does not match is reported and
its reference is not computed from it.

Input SELECTION_JSON: a list of entries {"unit": "g-<rtol>-<shard>-<serial>", "index", "notify_label",
"census_value", "census_abserr" (hex), "census_num_regions", "census_max_depth",
"census_converged", ...}; any other keys are carried. One worker per unit, at most six at once.
Writes OUT_DIR/<worker>/ (spec, emit, log, inputs pickle) and OUT_JSON: the entries, each with
its capture's {"dir", "pickle", "offset", "length", "sha256"} and ``capture_matches_census``.

Usage: PYTHONPATH=<tree g> python -B x1_capture.py SELECTION_JSON OUT_DIR OUT_JSON [--workers N]
"""

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import PYTHON, TREES, WIDER, header, sha256  # noqa: E402

BUDGET_S = 300.0
ROWS_JSON = WIDER / "state" / "rows.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selection")
    ap.add_argument("out_dir")
    ap.add_argument("out_json")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.workers > 6:
        raise SystemExit("at most six workers")
    # the captured records hold functions pickled by reference to g's modules, and are unpickled
    # here to be matched to the census (out/cmd/11, the first development capture, stopped there)
    # depth20-tiling: PYTHONPATH is <tree g>:<this scripts dir>
    if os.environ.get("PYTHONPATH") != str(TREES["g"][0]) + os.pathsep + str(HERE):
        raise SystemExit("PYTHONPATH must be <tree g>:<this scripts dir>")
    for l in header(Path(__file__).resolve(), "depth20-tiling: the sampled calls' inputs, captured at g", [Path(args.selection), ROWS_JSON]):
        print(l)
    sel = json.loads(Path(args.selection).read_text())
    rows = json.loads(ROWS_JSON.read_text())["rows"]
    by_key = {f"{r['shard']}-{r['serial']}": r for r in rows}
    by_unit = {}
    for e in sel:
        by_unit.setdefault(e["unit"], []).append(e)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    pending = []
    for i, (uid, ents) in enumerate(sorted(by_unit.items())):
        parts = uid.split("-")  # "g-1e-7-<shard>-<serial>": the rtol holds a "-" (06b's deviation 20)
        tree, shard, serial, rtol = parts[0], parts[-2], parts[-1], "-".join(parts[1:-2])
        if tree != "g" or rtol not in ("1e-7", "1e-8"):
            raise SystemExit(f"{uid}: not a g unit")
        R = by_key[f"{shard}-{serial}"]
        u = {
            "uid": uid,
            "tree": "g",
            "rtol": rtol,
            "row": R["name"],
            "names05": R["names05"],
            "shard": R["shard"],
            "serial": R["serial"],
            "k_key": R["k_key"],
            "budget": BUDGET_S,
            "scan": rtol == "1e-7",
            "w3": rtol == "1e-7",
            "capture": [[e["index"], e["notify_label"]] for e in sorted(ents, key=lambda e: e["index"])],
        }
        name = f"C{i:02d}"
        wdir = out_dir / name
        wdir.mkdir()
        spec = {
            "tree": "g",
            "name": name,
            "units": [u],
            "out_dir": str(wdir),
            "emit_fd": None,
            "emit_file": str(wdir / "emit.jsonl"),
            "rows_sha256": sha256(ROWS_JSON),
            "w2_capture_sha256": None,
        }
        (wdir / "spec.json").write_text(json.dumps(spec, indent=1))
        pending.append((name, wdir))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TREES["g"][0]) + os.pathsep + str(HERE)  # depth20-tiling
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    running = {}
    rc = {}
    while pending or running:
        while pending and len(running) < args.workers:
            name, wdir = pending.pop(0)
            log = open(wdir / "worker.log", "ab")
            running[name] = subprocess.Popen(
                [str(PYTHON), "-B", str(HERE / "worker.py"), str(wdir / "spec.json")],
                cwd=str(WIDER), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            )
            log.close()
            print(f"{name}: pid {running[name].pid}", flush=True)
        for name, p in list(running.items()):
            if p.poll() is not None:
                rc[name] = p.returncode
                del running[name]
        time.sleep(0.5)
    print(f"exit codes {rc}")
    # read back
    import hashlib

    out = []
    for name in sorted(rc):
        wdir = out_dir / name
        units, bye = {}, None
        for line in open(wdir / "emit.jsonl"):
            m = json.loads(line)
            if m["kind"] == "unit":
                units[m["data"]["unit"]] = m["data"]
            elif m["kind"] == "bye":
                bye = m
        for uid, d in units.items():
            inp = {e["index"]: e for e in (d.get("x1_inputs") or [])}
            for e in by_unit[uid]:
                r = dict(e)
                r["capture_worker"] = name
                r["capture_unit_status"] = d["status"]
                r["capture_ray_initialized"] = None if bye is None else bye["ray_initialized"]
                ent = inp.get(e["index"])
                if ent is None or ent.get("error"):
                    r["capture_error"] = "not captured" if ent is None else ent["error"]
                    r["capture_matches_census"] = False
                    out.append(r)
                    continue
                with open(wdir / ent["pickle"], "rb") as f:
                    f.seek(ent["offset"])
                    blob = f.read(ent["length"])
                if hashlib.sha256(blob).hexdigest() != ent["sha256"]:
                    raise RuntimeError(f"{uid} {e['index']}: capture blob sha256 differs")
                rec = pickle.loads(blob)
                r.update({"dir": str(wdir), "pickle": ent["pickle"], "offset": ent["offset"], "length": ent["length"], "sha256": ent["sha256"]})
                got = (rec["value"], rec["abserr"], rec["num_regions"], rec["max_depth"], rec["converged"], rec["notify_label"])
                want = (e["census_value"], e["census_abserr"], e["census_num_regions"], e["census_max_depth"], e["census_converged"], e["notify_label"])
                r["capture_record"] = {"value": rec["value"], "abserr": rec["abserr"], "num_regions": rec["num_regions"],
                                       "max_depth": rec["max_depth"], "converged": rec["converged"]}
                r["capture_matches_census"] = got == want
                r["capture_error"] = None
                out.append(r)
    Path(args.out_json).write_text(json.dumps(out, indent=1))
    n_ok = sum(1 for r in out if r["capture_matches_census"])
    print(f"captured and matching the census bit for bit: {n_ok} of {len(sel)}")
    for r in out:
        print(f"   {r['unit']} {r['index']}: matches {r['capture_matches_census']}; error {r.get('capture_error')}; ray_initialized {r['capture_ray_initialized']}")
    print(f"wrote {args.out_json} sha256 {sha256(args.out_json)}")
    return 0 if n_ok == len(sel) and len(out) == len(sel) else 1


if __name__ == "__main__":
    sys.exit(main())
