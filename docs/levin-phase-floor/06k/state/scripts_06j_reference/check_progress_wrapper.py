"""levin-phase-floor 06g addendum 3: one_reading.py with --progress reproduces part 1's readings bit
for bit (value_30, imag_30, value_hex) on z25 (Route A, four readings) and z13 (Route B, QAWO r1 and
FCC r1). Runs each in its own process as run_refs.py does; writes out/check_progress_wrapper/."""
import json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); import refs_common as RC
env = dict(os.environ, PYTHONPATH=f"{RC.G_TREE}:{HERE}", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MKL_NUM_THREADS="1", GIT_OPTIONAL_LOCKS="0")
sel = {e["sid"]: e for e in json.loads((ROOT / "out/sample/selection.json").read_text())}
out = ROOT / "out" / "check_progress_wrapper"; out.mkdir(exist_ok=False)
ok = True
for sid, keys in (("z25", [("qawo", 1), ("qawo", 2), ("fcc", 1), ("fcc", 2)]), ("z13", [("qawo", 1), ("fcc", 1)])):
    e = sel[sid]
    for impl, res in keys:
        dst = out / f"{sid}-{impl}-r{res}.json"; prog = out / f"{sid}-{impl}-r{res}.progress.json"
        rc = subprocess.run([str(RC.PYTHON), "-B", str(HERE / "one_reading.py"), str(ROOT / "state/inputs" / f"{sid}.pkl"), impl, str(res), str(dst),
                             "--sid", sid, "--unit", e["unit"], "--index", str(e["index"]), "--rtol", e["rtol"], "--workers", "6", "--progress", str(prog)],
                            cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode
        a = json.loads(dst.read_text()); b = json.loads((ROOT / "out/readings_06i13" / f"{sid}-{impl}-r{res}.json").read_text())
        same = all(a[k] == b[k] for k in ("value_30", "imag_30", "value_hex"))
        ok &= same and rc == 0
        print(f"{sid} {impl} r{res}: rc {rc}; progress {json.loads(prog.read_text())}; identical to part 1: {same}")
print("ALL IDENTICAL" if ok else "NOT IDENTICAL"); sys.exit(0 if ok else 1)
