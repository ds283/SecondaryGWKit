"""S2's references (prompt 07 section 5.6): P1's two rows' G+q+r calls, captured at R27's tree, referenced by
tools/filon_reference.py with four readings (QAWO and FCC at two resolutions; FCC at resolution 2 the
reference). levin-phase-floor 07, rehearsal agent. New.

Two parts:
- ``launch``: registers the run with runregistrykit (campaign levin-phase-floor, prompt 07, slug
  p1-references; manifest first), then starts ``drive`` detached (nohup sh -c, a new session, stdout and
  stderr into the run directory) and records the sh pid; it does not wait.
- ``drive --run-dir D``: heartbeats with its own pid, then for each call runs the tool's command line,
  ``python -B -m tools.filon_reference PICKLE --tolerance T --unpickler ray.cloudpickle --workers 6 --out
  JSON``, in the code worktree (cwd), PYTHONPATH the code worktree then rehearsal/scripts/capture (the
  pickles name the capture harness's reducer), one call at a time (at most six worker processes), and
  records each call's exit and wall time; finishes the run done (exit 0) if both exit 0, failed otherwise.

The tolerance (the count rule's, 06g's form; prompt 07 section 6 V4 states it for the census as
min(abserr_06m, abserr_g)/100): min(abserr_R27, abserr_R28)/100, the two stores' own call abserrs (unscaled):
R27's from out/captures.json (the capture's record, checked against R27's stored group), R28's from the
replay at g (the call that reproduces R28's stored group bit for bit).

Ray-free; the stores are not opened (only the capture pickles and the replay's records are read).
"""

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

R = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal")
C = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07-7460035")
PY = "/Users/ds283/Documents/Code/SecondaryGWKit/venv/bin/python"
OUT = R / "out" / "p1_references"
sys.path.insert(0, str(R / "scripts" / "capture"))


def calls():
    import replay07 as RP

    caps = json.loads((R / "out" / "captures.json").read_text())["captures"]
    units = RP.load_units(R / "state" / "capture" / "replay")
    rows = json.loads((R / "state" / "replay_rows.json").read_text())["rows"]
    out = []
    for c in [c for c in caps if c["label"] == "G+q+r" and c["tree"] == "p"]:
        e = [e for e in rows if e["key"] == c["key"]][0]
        rec = [r for r in e["records"] if r["kind"] == "group" and r["label"] == "G+q+r"][0]
        u, _, _ = units[f"g-1e-7-{e['R28']['shard']}-{e['R28']['serial']}"]
        hits = [x for x in u["calls"] if x["notify_label"].endswith("group G+q+r")
                and RP.stored_form(x, e["key"][3]) == (rec["R28"]["value"], rec["R28"]["abserr"])]
        assert len(hits) == 1
        e27, e28 = float.fromhex(c["call_abserr"]), float.fromhex(hits[0]["abserr"]["hex"])
        out.append({"p1": e["p1"], "key": e["key"], "pickle": c["pickle"], "pickle_sha256": c["sha256"],
                    "R27_value": c["call_value"], "R27_abserr": c["call_abserr"],
                    "R28_value": hits[0]["value"]["hex"], "R28_abserr": hits[0]["abserr"]["hex"],
                    "R28_guard_07": hits[0]["guard_07"], "tolerance": min(e27, e28) / 100.0,
                    "out": str(OUT / f"p1_row{e['p1']}.json")})
    return sorted(out, key=lambda x: x["p1"])


def launch():
    import runregistrykit

    cs = calls()
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "calls.json").write_text(json.dumps(cs, indent=1))
    script = str(Path(__file__).resolve())
    run = runregistrykit.begin(
        campaign="levin-phase-floor", prompt="07", slug="p1-references",
        purpose=("levin-phase-floor prompt 07 section 5.6 (S2) on the Mac Studio (mps026298; Python 3.13.16): "
                 "tools/filon_reference.py's four readings (QAWO and FCC at two resolutions, FCC at resolution 2 the "
                 "reference) of P1's two rows' G+q+r calls, row 1 (3.162e5)^3 and row 2 (1e5, 3.162e5, 3.162e5) at "
                 "z_response 250, captured at R27's tree 91472dbe (06m's production code) by 06g's v1 capture harness "
                 "and checked bit for bit against R27's stored groups; run from the code worktree 74600350 (07 "
                 "ad732887); count tolerance min(abserr_R27, abserr_R28)/100; Ray-free, no store opened, at most six "
                 "worker processes, one call at a time"),
        argv=[PY, "-B", script, "drive", "--run-dir", "<run dir>"], cwd=str(C), script=script,
        unit="calls", expected_units=len(cs), checkpoint=None, results=None,
        scope="2 captured calls x 4 readings (S2's references); outputs in var/levin-phase-floor-07/rehearsal/out/p1_references/",
        heartbeat_means="Refreshed by the launcher with the sh pid, by the driver at start with its own pid, and at each call's start and end.",
    )
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = " ".join(shlex.quote(a) for a in [PY, "-B", script, "drive", "--run-dir", str(run.path)]) + "; exit $?"
    so, se = open(run.stdout_path, "ab"), open(run.stderr_path, "ab")
    child = subprocess.Popen(["nohup", "sh", "-c", command], cwd=str(C), env=env, stdin=subprocess.DEVNULL,
                             stdout=so, stderr=se, start_new_session=True, close_fds=True)
    so.close()
    se.close()
    run.heartbeat(pid=child.pid, stage="launched (detached sh started)")
    print(f"run id {run.id}\nrun dir {run.path}\nsh pid {child.pid}\ncommand {command}")
    for c in cs:
        print(f"row {c['p1']} {c['key']}: pickle {c['pickle']} tolerance {c['tolerance']!r}")


def drive(run_dir):
    import runregistrykit

    run = runregistrykit.Run(run_dir)
    run.heartbeat(pid=os.getpid(), stage="driver started")
    cs = json.loads((OUT / "calls.json").read_text())
    env = dict(os.environ, PYTHONPATH=f"{C}:{R / 'scripts' / 'capture'}", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    results = []
    for i, c in enumerate(cs):
        run.heartbeat(units_done=i, units_total=len(cs), stage=f"row {c['p1']} running")
        t0 = time.time()
        log = open(OUT / f"p1_row{c['p1']}.log", "ab")
        rc = subprocess.call([PY, "-B", "-m", "tools.filon_reference", c["pickle"], "--tolerance", repr(c["tolerance"]),
                              "--unpickler", "ray.cloudpickle", "--workers", "6", "--out", c["out"]],
                             cwd=str(C), env=env, stdout=log, stderr=subprocess.STDOUT)
        log.close()
        results.append({"p1": c["p1"], "rc": rc, "wall_s": time.time() - t0})
        print(json.dumps(results[-1]), flush=True)
        run.heartbeat(units_done=i + 1, units_total=len(cs), stage=f"row {c['p1']} exit {rc} ({time.time() - t0:.0f} s)")
    (OUT / "drive.json").write_text(json.dumps(results, indent=1))
    ok = all(r["rc"] == 0 for r in results)
    run.finish("done" if ok else "failed", exit_code=0 if ok else 1)
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "launch":
        launch()
    elif sys.argv[1] == "drive":
        sys.exit(drive(sys.argv[3]))
