"""levin-phase-floor prompt 06g section 5, addendum 1: capture V1's 12 group calls, at the parent
against R23 and at g against R24. New in 06g, after 06c's x1_capture.py.

For each row of ../out/v1_rows.json and each tree (p: the parent worktree 4f4f9ed, against R23;
g: the code worktree 1c5336e, against R24), the census harness of this directory (the census
agent's, with addendum 1's marked changes: the rehearsal scope's k sample and zend, the rehearsal
stores through ../stores/r23 and ../stores/r24, the 1e-8 tolerance optional) re-runs the row's unit
(``<tree>-1e-7-<shard>-<serial>``, the store's own shard and serial; no budget; no scan, no W3, no
capture) in a fresh census worker, at most six at once:
- **pass 1:** the unit's group calls are read from the worker's emitted record, and the one call
  whose value and abserr, multiplied by (1 + z_response) as ``phase_group_Levin_integral`` stores a
  group (one double product each), equal the store's sub-interval-1 ``G+q+r`` group (v23/e23 at p,
  v24/e24 at g) **bit for bit**, with a label naming ``group G+q+r``, is identified. Exactly one must; otherwise
  the row is a stop;
- **pass 2:** the unit is re-run with that call listed in ``capture``; the worker pickles its span,
  f, theta and arguments, and the same run's record of the call must again equal the store's
  value and abserr bit for bit (the pickle is copied to ../state/inputs/<wid>-<tree>.pkl, not
  unpickled here).
Writes ../state/capture/<pass>/<worker>/ and ../out/v1_capture.json; prints a line per row and tree.
Exits 1 if any row does not reproduce. Ray is never initialised (each worker reports it).

Usage: python -B v1_capture.py   (from var/levin-phase-floor-06g/references/v1/)
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import PYTHON, TREES, header, sha256  # noqa: E402

STORE_TAG = {"p": "R23", "g": "R24"}
MAX_WORKERS = 6


def run_workers(jobs):
    """jobs: [(name, tree, wdir, unit)]; at most six at once. Returns {name: (rc, unit data, bye)}."""
    pending = list(jobs)
    running = {}
    out = {}
    while pending or running:
        while pending and len(running) < MAX_WORKERS:
            name, tree, wdir, u = pending.pop(0)
            wdir.mkdir(parents=True, exist_ok=False)
            spec = {"tree": tree, "name": name, "units": [u], "out_dir": str(wdir), "emit_fd": None,
                    "emit_file": str(wdir / "emit.jsonl"), "rows_sha256": None, "w2_capture_sha256": None}
            (wdir / "spec.json").write_text(json.dumps(spec, indent=1))
            env = dict(os.environ, PYTHONPATH=str(TREES[tree][0]), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1", GIT_OPTIONAL_LOCKS="0")
            log = open(wdir / "worker.log", "ab")
            p = subprocess.Popen([str(PYTHON), "-B", str(HERE / "worker.py"), str(wdir / "spec.json")], cwd=str(ROOT),
                                 env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            running[name] = (p, wdir)
            print(f"   {name}: pid {p.pid}", flush=True)
        for name, (p, wdir) in list(running.items()):
            if p.poll() is not None:
                unit, bye = None, None
                for line in open(wdir / "emit.jsonl"):
                    m = json.loads(line)
                    if m["kind"] == "unit":
                        unit = m["data"]
                    elif m["kind"] == "bye":
                        bye = m
                out[name] = (p.returncode, unit, bye)
                del running[name]
        time.sleep(0.5)
    return out


def stored_form(c, z_response):
    """A Levin call's (value, abserr) as the store holds its group: ``phase_group_Levin_integral``
    stores ``scale * data["value"]`` and ``scale * data["abserr"]`` with ``scale = 1.0 + z_resp``
    (ComputeTargets/QuadSourceIntegral.py), one double product each, reproduced here exactly."""
    scale = 1.0 + float(z_response)
    return ((scale * float.fromhex(c["value"]["hex"])).hex(), (scale * float.fromhex(c["abserr"]["hex"])).hex())


def unit_spec(tree, rec, capture=None):
    uid = f"{tree}-1e-7-{rec['shard']}-{rec['serial']}"
    return {"uid": uid, "tree": tree, "rtol": "1e-7", "row": f"{rec['shard']}-{rec['serial']}", "names05": None,
            "shard": rec["shard"], "serial": rec["serial"], "k_key": f"{rec['shard']}:{rec['k_exit_serial']}",
            "budget": None, "scan": False, "w3": False, "capture": capture}


def main():
    rows_p = ROOT / "out" / "v1_rows.json"
    out_p = ROOT / "out" / "v1_capture.json"
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    for l in header(Path(__file__).resolve(), "levin-phase-floor 06g addendum 1: V1's group calls captured at p (R23) and g (R24)", [rows_p]):
        print(l)
    rows = json.loads(rows_p.read_text())
    res = {r["wid"]: {"wid": r["wid"], "k": r["k"], "q": r["q"], "r": r["r"], "z_response": r["z_response"]} for r in rows}
    # pass 1
    jobs = []
    for r in rows:
        for tree in ("p", "g"):
            rec = r[STORE_TAG[tree]]
            jobs.append((f"P1-{r['wid']}-{tree}", tree, ROOT / "state" / "capture" / "pass1" / f"{r['wid']}-{tree}", unit_spec(tree, rec)))
    print(f"pass 1: {len(jobs)} units")
    p1 = run_workers(jobs)
    ok = True
    jobs2 = []
    for r in rows:
        for tree in ("p", "g"):
            tag = STORE_TAG[tree]
            want = r[tag]["group"]
            rc, unit, bye = p1[f"P1-{r['wid']}-{tree}"]
            e = res[r["wid"]].setdefault(tree, {})
            e.update(store=tag, store_value=want["value"], store_abserr=want["abserr"], pass1_rc=rc,
                     pass1_status=None if unit is None else unit["status"],
                     ray_initialized=None if bye is None else bye["ray_initialized"])
            hits = []
            if unit is not None and unit["status"] == "done":
                e["unit_total"] = unit["result"]["total"]["hex"] if unit.get("result") else None
                e["unit_total_matches_store"] = e["unit_total"] == r[tag]["total"]
                for c in unit["calls"]:
                    if stored_form(c, r["z_response"]) == (want["value"], want["abserr"]):
                        hits.append(c)
            e["n_matching_calls"] = len(hits)
            good = len(hits) == 1 and "group G+q+r" in hits[0]["notify_label"]
            if good:
                c = hits[0]
                e.update(index=c["index"], notify_label=c["notify_label"], converged=c["converged"],
                         call_value=c["value"]["hex"], call_abserr=c["abserr"]["hex"],
                         num_regions=c["num_regions"], max_depth=c["max_depth"], x_span=[x["hex"] for x in c["x_span"]])
                jobs2.append((f"P2-{r['wid']}-{tree}", tree, ROOT / "state" / "capture" / "pass2" / f"{r['wid']}-{tree}",
                              unit_spec(tree, r[tag], capture=[[c["index"], c["notify_label"]]])))
            ok &= good
            print(f"{r['wid']} {tree} ({tag}): pass 1 rc {rc} status {e['pass1_status']}; row total equals the store's "
                  f"{e.get('unit_total_matches_store')}; calls equal to the stored group bit for bit: {len(hits)}"
                  + (f" (#{e['index']} {e['notify_label'][-46:]})" if good else " -- DOES NOT REPRODUCE"))
    if not ok:
        out_p.write_text(json.dumps(list(res.values()), indent=1))
        print("STOP: a row does not reproduce its stored group in pass 1")
        return 1
    print(f"pass 2: {len(jobs2)} units")
    p2 = run_workers(jobs2)
    for r in rows:
        for tree in ("p", "g"):
            e = res[r["wid"]][tree]
            name = f"P2-{r['wid']}-{tree}"
            rc, unit, bye = p2[name]
            wdir = ROOT / "state" / "capture" / "pass2" / f"{r['wid']}-{tree}"
            ent = None
            if unit is not None:
                ent = {x["index"]: x for x in (unit.get("x1_inputs") or [])}.get(e["index"])
            good = ent is not None and not ent.get("error")
            if good:
                with open(wdir / ent["pickle"], "rb") as f:
                    f.seek(ent["offset"])
                    blob = f.read(ent["length"])
                if hashlib.sha256(blob).hexdigest() != ent["sha256"]:
                    raise RuntimeError(f"{name}: blob sha256 differs")
                # the captured call's record in the same worker run (the pickle names the tree's
                # modules, and this process has neither tree on its path, so it is not unpickled here)
                cc = [c for c in unit["calls"] if c["index"] == e["index"]]
                dst = ROOT / "state" / "inputs" / f"{r['wid']}-{tree}.pkl"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    raise SystemExit(f"{dst} exists")
                dst.write_bytes(blob)
                good = (len(cc) == 1 and cc[0]["notify_label"] == e["notify_label"]
                        and cc[0]["value"]["hex"] == e["call_value"] and cc[0]["abserr"]["hex"] == e["call_abserr"]
                        and stored_form(cc[0], r["z_response"]) == (e["store_value"], e["store_abserr"]))
                e.update(pickle=str(dst), sha256=ent["sha256"], bytes=len(blob),
                         capture_value=cc[0]["value"]["hex"] if cc else None, capture_abserr=cc[0]["abserr"]["hex"] if cc else None)
            e["pass2_rc"] = rc
            e["pass2_ray_initialized"] = None if bye is None else bye["ray_initialized"]
            e["reproduces_store"] = bool(good)
            ok &= good
            print(f"{r['wid']} {tree}: capture {'REPRODUCES' if good else 'DOES NOT REPRODUCE'} the store's value and abserr bit for bit"
                  f" ({e.get('capture_value')} {e.get('capture_abserr')}); ray_initialized {e['pass2_ray_initialized']}")
    out_p.write_text(json.dumps(list(res.values()), indent=1))
    print(f"wrote {out_p} sha256 {sha256(out_p)}; all reproduce: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
