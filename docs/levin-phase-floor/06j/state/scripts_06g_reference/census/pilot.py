"""levin-phase-floor prompt 06a section 3.1, the pilot and its projection. New in 06a.

``run OUT_DIR``: one fresh census worker (worker.py, tree h, the producer cache), timed end to end
with the producer re-run of its k, on one (k, z_response) block in each band (three blocks of 8
rows), at rtol 1e-7 (with the scan, as in the run) and then at 1e-8 (as in the run, no scan), in
that one worker. The choice, fixed before the pilot ran:
- k is ``3:201`` (k = 9.70e6/Mpc), the k with the largest stored A3 v2 compute time (the costliest
  under the code A3 v2 was built with);
- in each band, the block whose summed stored compute time is the band's median for that k
  (low z_response serial 1680, mid 1344, seam 511).
The runner reads the worker's pipe and timestamps every line as it arrives.

``project OUT_DIR``: the projection, from the pilot alone:
- per unit and band, the time between consecutive unit lines (the first unit of each rtol less the
  producer re-run and assembly the cache did in it, which are counted once per worker);
- a worker of the run = the process's start to its context + the producer re-run of its k +
  sum over bands of (units in the band for one k) x (mean unit time in the band), per rtol;
- the run: eight such workers per rtol on six slots, two rounds, so the wall time is
  2 x worker(1e-7) + 2 x worker(1e-8); CPU time 8 x the same;
- the run directory: per unit, the checkpoint line (the unit's JSON plus the registry's record
  wrapper, measured on the registry's own record format below), the pickle record, the event lines
  and the worker log, by band, times the units in each band, plus each worker's fixed files.
It records the worker's peak RSS (``ru_maxrss`` of the child).
Usage: python -B pilot.py run|project OUT_DIR
"""

import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import PYTHON, TREES, WIDER, header, sha256  # noqa: E402

ROWS_JSON = WIDER / "state" / "rows.json"
K_KEY = "3:201"
BLOCKS = {"low": 1680, "mid": 1344, "seam": 511}
BUDGET_S = 300.0


def cmd_run(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    doc = json.loads(ROWS_JSON.read_text())
    rows = [
        r for r in doc["rows"]
        if r["k_key"] == K_KEY and r["z_response_serial"] in BLOCKS.values()
    ]
    assert len(rows) == 24, len(rows)
    units = []
    for rtol in ("1e-7", "1e-8"):
        for r in rows:
            units.append(
                {
                    "uid": f"h-{rtol}-{r['shard']}-{r['serial']}",
                    "tree": "h",
                    "rtol": rtol,
                    "row": r["name"],
                    "names05": r["names05"],
                    "shard": r["shard"],
                    "serial": r["serial"],
                    "k_key": r["k_key"],
                    "budget": BUDGET_S,
                    "scan": rtol == "1e-7",
                }
            )
    before = sorted(os.listdir("/tmp/ray-sgk")) if os.path.isdir("/tmp/ray-sgk") else []
    r_fd, w_fd = os.pipe()
    spec = {
        "tree": "h",
        "name": "pilot",
        "units": units,
        "out_dir": str(out_dir),
        "emit_fd": w_fd,
        "emit_file": None,
        "rows_sha256": sha256(ROWS_JSON),
    }
    sp = out_dir / "spec.json"
    sp.write_text(json.dumps(spec, indent=1))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TREES["h"][0])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    log = open(out_dir / "worker.log", "ab")
    t_start = time.time()
    child = subprocess.Popen(
        [str(PYTHON), "-B", str(HERE / "worker.py"), str(sp)],
        cwd=str(WIDER), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        pass_fds=(w_fd,),
    )
    log.close()
    os.close(w_fd)
    print(f"pilot worker pid {child.pid}; {len(units)} units", flush=True)
    with os.fdopen(r_fd, "r") as rd, open(out_dir / "emit_timed.jsonl", "w") as out:
        for line in rd:
            out.write(json.dumps({"t": time.time(), "line": line.rstrip("\n")}) + "\n")
            out.flush()
    rc = child.wait()
    t_end = time.time()
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    after = sorted(os.listdir("/tmp/ray-sgk")) if os.path.isdir("/tmp/ray-sgk") else []
    meta = {
        "t_start": t_start,
        "t_end": t_end,
        "wall_s": t_end - t_start,
        "exit": rc,
        "child_ru_maxrss_bytes": ru.ru_maxrss,
        "child_ru_utime_s": ru.ru_utime,
        "child_ru_stime_s": ru.ru_stime,
        "tmp_ray_sgk_new": sorted(set(after) - set(before)),
    }
    (out_dir / "pilot_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    return 0 if rc == 0 else 1


def cmd_project(out_dir):
    out_dir = Path(out_dir)
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06a: the pilot's projection",
                    [ROWS_JSON, out_dir / "emit_timed.jsonl", out_dir / "pilot_meta.json"]):
        print(l)
    doc = json.loads(ROWS_JSON.read_text())
    band_of = {r["name"]: r["band"] for r in doc["rows"]}
    n_band_per_k = {}
    for r in doc["rows"]:
        if r["k_key"] == K_KEY:
            n_band_per_k[r["band"]] = n_band_per_k.get(r["band"], 0) + 1
    n_k = len({r["k_key"] for r in doc["rows"]})
    meta = json.loads((out_dir / "pilot_meta.json").read_text())
    msgs = [json.loads(l) for l in open(out_dir / "emit_timed.jsonl")]
    t_context = None
    units = []
    events = {}
    prev_t = None
    for m in msgs:
        x = json.loads(m["line"])
        if x["kind"] == "context":
            t_context = m["t"]
            prev_t = m["t"]
        elif x["kind"] == "event":
            events.setdefault(x["unit"], []).append(x["event"])
        elif x["kind"] == "unit":
            d = x["data"]
            units.append((d, m["t"] - prev_t, len(m["line"]) + 1))
            prev_t = m["t"]
    startup = t_context - meta["t_start"]
    print(f"\n== the pilot: {len(units)} units in {meta['wall_s']:.1f} s wall; exit {meta['exit']}; "
          f"peak RSS {meta['child_ru_maxrss_bytes'] / 2**20:.0f} MiB; "
          f"/tmp/ray-sgk new entries {meta['tmp_ray_sgk_new']}")
    print(f"   process start to context: {startup:.1f} s")
    prodG = [e for evs in events.values() for e in evs if e["kind"] == "producers_G"]
    t_G = sum(e["summary"]["timing"]["total"] for e in prodG)
    print(f"   producer re-run of k (rerun_k_G): {t_G:.1f} s "
          f"({prodG[0]['summary']['G_objects_rerun']} GkWKBIntegration, "
          f"{prodG[0]['summary']['G_samples_rerun']} samples)")
    gk = [e for evs in events.values() for e in evs if e["kind"] == "Gk"]
    t_gk = [e["summary"]["timing"]["total"] for e in gk]
    print(f"   GkSource + GkSourcePolicyData assemblies: {len(gk)}, "
          f"{', '.join(f'{t:.2f}' for t in t_gk)} s")
    other = [e for evs in events.values() for e in evs if e["kind"] in ("producers_T", "QuadSource")]
    print(f"   T re-runs and QuadSources built: {len(other)}")
    per = {}
    size = {}
    for d, dt, line_len in units:
        # the producer re-run is counted once per worker, not in the unit it ran in
        evs = events.get(d["unit"], [])
        dt_adj = dt - sum(e["summary"]["timing"]["total"] for e in evs if e["kind"] == "producers_G")
        key = (d["rtol"], band_of[d["row"]])
        per.setdefault(key, []).append(dt_adj)
        ev_bytes = sum(len(json.dumps({"kind": "event", "worker": "x", "unit": d["unit"], "event": e})) + 1 for e in evs)
        # the registry's record: {"unit", "recorded", "script_sha256", "git_head", "git_dirty", "data"}
        rec_len = line_len + 200
        size.setdefault(key, []).append(rec_len + (d.get("pickle_length") or 0) + ev_bytes)
    log_bytes = (out_dir / "worker.log").stat().st_size
    log_per_unit = log_bytes / len(units)
    print("\n   per unit, by rtol and band (seconds between unit lines, the k's producer re-run taken out):")
    worker_s = {}
    for rtol in ("1e-7", "1e-8"):
        tot = startup + t_G
        for b in ("low", "mid", "seam"):
            v = per[(rtol, b)]
            mean = sum(v) / len(v)
            print(f"     {rtol} {b:4s}: n {len(v)}; mean {mean:.3f}; min {min(v):.3f}; max {max(v):.3f}; "
                  f"wall_s {[round(d['wall_s'], 3) for d, _, _ in units if d['rtol'] == rtol and band_of[d['row']] == b]}")
            tot += n_band_per_k[b] * mean
        worker_s[rtol] = tot
        print(f"     a worker at {rtol} (one k, {sum(n_band_per_k.values())} units "
              f"{n_band_per_k}): {tot:.0f} s = {tot / 3600:.2f} h")
    rounds = -(-n_k // 6)
    wall = rounds * (worker_s["1e-7"] + worker_s["1e-8"])
    cpu = n_k * (worker_s["1e-7"] + worker_s["1e-8"])
    print(f"\n   PROJECTION, wall time at six workers: {rounds} rounds x (worker 1e-7 + worker 1e-8) "
          f"= {wall:.0f} s = {wall / 3600:.2f} h; CPU {cpu / 3600:.2f} h ({n_k} k)")
    total_bytes = 0
    for rtol in ("1e-7", "1e-8"):
        for b in ("low", "mid", "seam"):
            v = size[(rtol, b)]
            mean = sum(v) / len(v)
            total_bytes += n_k * n_band_per_k[b] * (mean + log_per_unit)
            print(f"     bytes per unit {rtol} {b:4s}: mean {mean:.0f} (+ log {log_per_unit:.0f})")
    fixed = 2 * n_k * 20000
    total_bytes += fixed
    print(f"   PROJECTION, run directory size: {total_bytes / 1e6:.0f} MB = {total_bytes / 1e9:.2f} GB "
          f"(fixed per-worker files about {fixed / 1e6:.1f} MB)")
    stop = wall > 24 * 3600 or total_bytes > 20e9
    print(f"   section 3.1 stop rule (more than 24 h or 20 GB): {'STOP' if stop else 'within'}")
    statuses = {}
    for d, _, _ in units:
        statuses[d["status"]] = statuses.get(d["status"], 0) + 1
    print(f"   unit statuses: {statuses}; max depth by unit: "
          f"{[(d['rtol'], band_of[d['row']], (d.get('result') or {}).get('WKB_Levin_data', {}) and d['result']['WKB_Levin_data']['max_depth']) for d, _, _ in units]}")
    print(f"   scan seconds per 1e-7 unit: {[round(d['scan_s'], 3) for d, _, _ in units if d['scan_s'] is not None]}")
    return 2 if stop else 0


if __name__ == "__main__":
    cmd, arg = sys.argv[1], sys.argv[2]
    sys.exit({"run": cmd_run, "project": cmd_project}[cmd](arg))
