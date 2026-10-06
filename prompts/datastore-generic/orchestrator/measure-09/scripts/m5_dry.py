"""M5: what docs/handover/quadsource_atol_sweep.py --build --rehearsal would hand main.py, without
Ray and without writing anything: the scope, main.py's argv, the substitutions (each must match
once), and that the scoped driver loads. Writes the substituted main.py source to <out>.
Usage (cwd = tree, PYTHONPATH=tree): python m5_dry.py <tree> <database path> <out-prefix>"""
import argparse, contextlib, hashlib, importlib.util, io, json, sys
from pathlib import Path
tree, database, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(tree))
spec = importlib.util.spec_from_file_location("_sweep", str(tree / "docs/handover/quadsource_atol_sweep.py"))
sweep = importlib.util.module_from_spec(spec); spec.loader.exec_module(sweep)
args = argparse.Namespace(rehearsal=True, build=True, database=str(database), resume=False,
                          job_name="datastore-generic-09", atol=1e-32, rtol=1e-7, cpus=6,
                          register="rehearsal", purpose="x", campaign="datastore-generic", prompt="09")
k, zend, text = sweep.build_scope(args)
argv = sweep.build_main_argv(args, database)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    source = sweep.substitute_main_py(filter_cases=False)
Path(out + "_main_substituted.py").write_text(source)
scoped = sweep._load_scoped_driver()
import config.model_list as ml
res = {"k": k, "zend": zend, "scope": text.format(atol=args.atol, rtol=args.rtol, cpus=args.cpus, database=database.name),
       "main_argv": argv[1:], "substitutions": buf.getvalue().splitlines(),
       "substituted_main_sha256": hashlib.sha256(source.encode()).hexdigest(),
       "script_sha256": hashlib.sha256((tree / "docs/handover/quadsource_atol_sweep.py").read_bytes()).hexdigest(),
       "scoped_driver_has": [n for n in ("RegisteredStream", "StageTracker", "HEARTBEAT_MEANS", "BEAT_MIN_INTERVAL", "SignalLatch", "install_killed_handler") if hasattr(scoped, n)],
       "build_model_list": hasattr(ml, "build_model_list"),
       "ray_initialized": "ray" in sys.modules and sys.modules["ray"].is_initialized()}
Path(out + ".json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
