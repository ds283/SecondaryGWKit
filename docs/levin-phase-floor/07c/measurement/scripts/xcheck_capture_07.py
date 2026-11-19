"""levin-phase-floor prompt 07, beyond section 6 (the cross-check the user asked for at stop 4): capture, at g, one call
per row of the disagreement list's classes (c) and (d) (state/xcheck_capture_selection_pre.json, from
out/xcheck/xcheck.json: the call at 1e-7 where it is listed there, else at 1e-8; the integrand is the same at both
rtols), for the reference-cost projection only. After x1_capture.py (06c's, the census harness's), which runs one fresh
worker per unit; here the units are grouped one worker per (rtol, k), as the census groups them (so each worker rebuilds
its k's producers once), and the workers run one at a time, each under ``nice -n 19``, so that one computing process
runs. Each unit's spec is x1_capture.py's (the census's budget, scan and W3 as in the census, and ``capture``), and the
worker is the census's worker.py, unchanged. The capture counts only if its record equals 07's census call bit for bit
(value and abserr by hex, num_regions, max_depth, converged, label); check_captures.py then checks the regions, the
call record and the harness. New in 07. No Ray, no store write, nothing written in the tree.

Writes OUT_DIR/<worker>/ and OUT_JSON (x1_capture.py's form). Usage: PYTHONPATH=<tree g> python -B xcheck_capture_07.py
SELECTION_PRE_JSON SELECTION_JSON OUT_DIR OUT_JSON
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import PYTHON, TREES, WIDER, header, sha256  # noqa: E402


def main():
    pre_p, sel_p, out_dir, out_json = (Path(a) for a in sys.argv[1:5])
    if os.environ.get("PYTHONPATH") != str(TREES["g"][0]):
        raise SystemExit("PYTHONPATH must be the tree g")
    ck = RC.CENSUS07 / "checkpoint.jsonl"
    for l in header(Path(__file__).resolve(), "levin-phase-floor 07: the cross-check's class (c)/(d) captures at g", [pre_p, RC.ROWS]):
        print(l)
    if sha256(RC.ROWS) != RC.ROWS_SHA256 or sha256(ck) != RC.CENSUS07_CHECKPOINT_SHA256:
        raise SystemExit("rows.json or 07's census checkpoint has another SHA-256")
    pre = json.loads(pre_p.read_text())
    want = {e["unit"] for e in pre}
    recs = {}
    with open(ck) as f:
        for line in f:
            uid = line[10:60].split('"', 1)[0]
            if uid in want:
                recs[uid] = json.loads(line)["data"]
    sel = []
    for e in pre:
        d = recs[e["unit"]]
        key = "calls" if e["kind"] == "group" else "other_calls"
        c = [c for c in d[key] if c["index"] == e["index"]]
        if len(c) != 1 or d["status"] != "done":
            raise SystemExit(f"{e['unit']} #{e['index']}: not one done call")
        c = c[0]
        sel.append(dict(e, notify_label=c["notify_label"], census_value=c["value"]["hex"], census_abserr=c["abserr"]["hex"],
                        census_num_regions=c["num_regions"], census_max_depth=c["max_depth"],
                        census_converged=c["converged"], census_unit_budget_s=d["budget_s"]))
    if sel_p.exists():
        raise SystemExit(f"{sel_p} exists")
    sel_p.write_text(json.dumps(sel, indent=1))
    rows = {f"{r['shard']}-{r['serial']}": r for r in json.loads(RC.ROWS.read_text())["rows"]}
    groups = {}
    for e in sel:
        p = e["unit"].split("-")
        rtol, key = f"{p[1]}-{p[2]}", f"{p[3]}-{p[4]}"
        R = rows[key]
        u = {"uid": e["unit"], "tree": "g", "rtol": rtol, "row": R["name"], "names05": R["names05"], "shard": R["shard"],
             "serial": R["serial"], "k_key": R["k_key"], "budget": e["census_unit_budget_s"], "scan": rtol == "1e-7",
             "w3": rtol == "1e-7", "capture": [[e["index"], e["notify_label"]]]}
        groups.setdefault((rtol, R["k_key"]), []).append(u)
    order = {r["name"]: i for i, r in enumerate(json.loads(RC.ROWS.read_text())["rows"])}
    out_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TREES["g"][0])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    names = []
    for i, ((rtol, kk), units) in enumerate(sorted(groups.items())):
        units.sort(key=lambda u: order[u["row"]])  # rows.json's order, as the census runs them
        name = f"X{i:02d}"
        wdir = out_dir / name
        wdir.mkdir()
        spec = {"tree": "g", "name": name, "units": units, "out_dir": str(wdir), "emit_fd": None,
                "emit_file": str(wdir / "emit.jsonl"), "rows_sha256": sha256(RC.ROWS), "w2_capture_sha256": None}
        (wdir / "spec.json").write_text(json.dumps(spec, indent=1))
        with open(wdir / "worker.log", "ab") as log:
            rc = subprocess.run(["nice", "-n", "19", str(PYTHON), "-B", str(HERE / "worker.py"), str(wdir / "spec.json")],
                                cwd=str(WIDER), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).returncode
        print(f"{name} ({rtol}, k {kk}): {len(units)} units, exit {rc}", flush=True)
        names.append((name, rc))
    import hashlib

    by_unit = {e["unit"]: e for e in sel}
    out = []
    for name, rc in names:
        wdir = out_dir / name
        units, bye = {}, None
        for line in open(wdir / "emit.jsonl"):
            m = json.loads(line)
            if m["kind"] == "unit":
                units[m["data"]["unit"]] = m["data"]
            elif m["kind"] == "bye":
                bye = m
        for uid, d in units.items():
            e = by_unit[uid]
            r = dict(e, capture_worker=name, capture_unit_status=d["status"],
                     capture_ray_initialized=None if bye is None else bye["ray_initialized"])
            ent = {x["index"]: x for x in (d.get("x1_inputs") or [])}.get(e["index"])
            if ent is None or ent.get("error"):
                r.update(capture_error="not captured" if ent is None else ent["error"], capture_matches_census=False)
                out.append(r)
                continue
            with open(wdir / ent["pickle"], "rb") as f:
                f.seek(ent["offset"])
                blob = f.read(ent["length"])
            if hashlib.sha256(blob).hexdigest() != ent["sha256"]:
                raise RuntimeError(f"{uid}: capture blob sha256 differs")
            rec = pickle.loads(blob)
            r.update({"dir": str(wdir), "pickle": ent["pickle"], "offset": ent["offset"], "length": ent["length"], "sha256": ent["sha256"]})
            got = (rec["value"], rec["abserr"], rec["num_regions"], rec["max_depth"], rec["converged"], rec["notify_label"])
            want_ = (e["census_value"], e["census_abserr"], e["census_num_regions"], e["census_max_depth"], e["census_converged"], e["notify_label"])
            r["capture_matches_census"] = got == want_
            r["capture_error"] = None
            out.append(r)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1))
    n_ok = sum(1 for r in out if r["capture_matches_census"])
    print(f"captured and matching 07's census bit for bit: {n_ok} of {len(sel)}; ray initialised in any worker: "
          f"{any(r.get('capture_ray_initialized') for r in out)}")
    print(f"wrote {out_json} sha256 {sha256(out_json)}")
    return 0 if n_ok == len(sel) and len(out) == len(sel) else 1


if __name__ == "__main__":
    sys.exit(main())
