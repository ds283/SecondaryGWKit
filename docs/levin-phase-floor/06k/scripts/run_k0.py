"""levin-phase-floor prompt 06k, Gate K0: run a list of census units at the tree with one variant of
``AdaptiveLevin/levin_quadrature.py`` loaded (or with the tree's own module), in 06g's census worker.
New in 06k (after 06c's x1_capture.py and 06g's run_census.py).

Each unit's spec is ``run_census.plan``'s (budget 300 s; ``scan`` and ``w3`` at 1e-7; no capture),
taken from 06g's rows.json. The units are grouped one worker per (rtol, k_key), each worker's units
in rows.json's order, as the census groups them; at most six workers run at once. Each worker's
spec carries ``"variant"`` ({name, path, sha256}) unless VARIANT is ``tree`` (no variant: the tree's
module, as 06g's census ran). Writes OUT_DIR/<worker>/ (spec, emit, log, region pickle) and
OUT_DIR/units.jsonl (one line per unit: {"unit", "worker", "data"}, the worker's emitted record).
Exits 1 unless every unit reports ``done``.

Usage: PYTHONPATH=<tree> python -B run_k0.py VARIANT UNITS_JSON OUT_DIR [--workers N]
       VARIANT is tree, identity, route1, route2 or route3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import PYTHON, TREES, WIDER, header, sha256  # noqa: E402
import refs_common as RC  # noqa: E402
import run_census as RCN  # noqa: E402

VARIANTS = ("tree", "identity", "route1", "route2", "route3", "route2lit")  # prompt 06k: route2lit after stop 1


def variant_spec(name):
    if name == "tree":
        return None
    p = HERE / "variants" / f"levin_quadrature_{name}.py"
    return {"name": name, "path": str(p.resolve()), "sha256": sha256(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=VARIANTS)
    ap.add_argument("units_json")
    ap.add_argument("out_dir")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.workers > 6:
        raise SystemExit("at most six workers")
    if os.environ.get("PYTHONPATH") != str(TREES["g"][0]):
        raise SystemExit("PYTHONPATH must be the tree g")
    vspec = variant_spec(args.variant)
    inputs = [Path(args.units_json), RC.ROWS] + ([Path(vspec["path"])] if vspec else [])
    for l in header(Path(__file__).resolve(), f"levin-phase-floor 06k: census units under the variant {args.variant}", inputs):
        print(l)
    if sha256(RC.ROWS) != RC.ROWS_SHA256:
        raise SystemExit("rows.json sha256 differs")
    print(f"# variant {json.dumps(vspec)}")
    want = [u["uid"] for u in json.loads(Path(args.units_json).read_text())["units"]]
    rows_doc = json.loads(RC.ROWS.read_text())
    plans = RCN.plan(rows_doc, tree="g")  # every unit of the census, as run_census builds them
    by_uid = {}
    order = []
    for rt in RCN.RTOLS:
        for wname, units in plans[rt]:
            for u in units:
                by_uid[u["uid"]] = (rt, wname, u)
                order.append(u["uid"])
    missing = [u for u in want if u not in by_uid]
    if missing:
        raise SystemExit(f"units not in the census plan: {missing[:5]}")
    wantset = set(want)
    groups = {}
    for uid in order:  # the census's order: per rtol, worker by worker, rows.json's order within
        if uid in wantset:
            rt, wname, u = by_uid[uid]
            groups.setdefault(wname, []).append(u)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    pending = []
    for wname, units in groups.items():
        name = f"{args.variant}-{wname}"
        wdir = out_dir / name
        wdir.mkdir()
        spec = {"tree": "g", "name": name, "units": units, "out_dir": str(wdir), "emit_fd": None,
                "emit_file": str(wdir / "emit.jsonl"), "rows_sha256": sha256(RC.ROWS), "w2_capture_sha256": None,
                "variant": vspec}
        (wdir / "spec.json").write_text(json.dumps(spec, indent=1))
        pending.append((name, wdir, [u["uid"] for u in units]))
    print(f"# {len(want)} units in {len(pending)} workers: " + ", ".join(f"{n} ({len(us)})" for n, _, us in pending), flush=True)
    env = RCN.worker_env("g")
    running, rc, t_start = {}, {}, time.time()
    while pending or running:
        while pending and len(running) < args.workers:
            name, wdir, uids = pending.pop(0)
            log = open(wdir / "worker.log", "ab")
            running[name] = (subprocess.Popen([str(PYTHON), "-B", str(HERE / "worker.py"), str(wdir / "spec.json")],
                                              cwd=str(WIDER), env=env, stdin=subprocess.DEVNULL, stdout=log,
                                              stderr=subprocess.STDOUT), wdir, uids, time.time())
            log.close()
            print(f"{name}: pid {running[name][0].pid} ({len(uids)} units)", flush=True)
        for name, (p, wdir, uids, t0) in list(running.items()):
            if p.poll() is not None:
                rc[name] = (p.returncode, wdir, uids, time.time() - t0)
                print(f"{name}: exit {p.returncode} after {time.time() - t0:.1f} s", flush=True)
                del running[name]
        time.sleep(0.5)
    got = {}
    bye = {}
    refused = []
    with open(out_dir / "units.jsonl", "w") as fo:
        for name, (code, wdir, uids, wall) in sorted(rc.items()):
            for line in open(wdir / "emit.jsonl"):
                m = json.loads(line)
                if m["kind"] == "unit":
                    d = m["data"]
                    got[d["unit"]] = d["status"]
                    fo.write(json.dumps({"unit": d["unit"], "worker": name, "data": d}) + "\n")
                elif m["kind"] == "bye":
                    bye[name] = {"ray_initialized": m["ray_initialized"], "modules_foreign": m["modules_foreign"],
                                 "modules_by_root": m["modules_by_root"]}
                elif m["kind"] == "refused":
                    refused.append((name, m.get("why")))
    n_done = sum(1 for u in want if got.get(u) == "done")
    print(f"exit codes {{{', '.join(f'{k}: {v[0]}' for k, v in sorted(rc.items()))}}}")
    print(f"refused: {refused}")
    for name, b in sorted(bye.items()):
        print(f"bye {name}: ray_initialized {b['ray_initialized']}; foreign modules {b['modules_foreign']}; by root {b['modules_by_root']}")
    print(f"units done {n_done} of {len(want)}; not done {[u for u in want if got.get(u) != 'done']}; wall {time.time() - t_start:.0f} s")
    print(f"wrote {out_dir / 'units.jsonl'} sha256 {sha256(out_dir / 'units.jsonl')}")
    ok = n_done == len(want) and all(v[0] == 0 for v in rc.values()) and not refused and all(
        not b["ray_initialized"] and not b["modules_foreign"] for b in bye.values()) and len(bye) == len(rc)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
