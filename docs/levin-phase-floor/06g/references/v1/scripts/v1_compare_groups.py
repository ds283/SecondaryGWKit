"""levin-phase-floor prompt 06g section 5, addendum 1: the two captures of each V1 call, at the parent
(R23) and at g (R24), must hold bit-identical groups. New in 06g. Both pickles are loaded here, at g
(PYTHONPATH the code worktree, then this directory for the harness's functions), and compared by
compare13.py's ``describe`` (references/scripts/compare13.py, imported, unchanged): the route, the
span, the active factors' signs and each one's phase spline (knots, coefficients, degree) and
leading table (nodes, breaks), by bytes (the criterion); and the integrand as the Filon tool reads
it (f_sin, f_cos, theta, theta_mod_2pi, theta_deriv at 65 points; production_edges). Writes
../out/v1_compare_groups.json; exits 1 on any difference.
Usage: PYTHONPATH=<tree g>:<this directory> python -B v1_compare_groups.py"""
import json, sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.append(str(ROOT.parent / "scripts"))  # compare13.py and refs_common.py, after this directory
import compare13 as C  # noqa: E402
import tools.filon_reference as F  # noqa: E402
cap = json.loads((ROOT / "out" / "v1_capture.json").read_text())
print(f"# v1_compare_groups.py sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; compare13.py sha256 {hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest()}")
out, bad = [], 0
for e in cap:
    d = {}
    for t in ("p", "g"):
        print(f"# input {e[t]['pickle']} sha256 {e[t]['sha256']}")
        d[t] = C.describe(C.load(e[t]["pickle"]), F)
    dc = C.first_diff(d["p"][0], d["g"][0]); di = C.first_diff(d["p"][1], d["g"][1]); dd = C.first_diff(d["p"][2], d["g"][2])
    bad += dc is not None or di is not None
    out.append({"wid": e["wid"], "route": d["g"][0]["route"], "criterion_first_difference": dc, "integrand_first_difference": di,
                "descriptive_first_difference": dd, "n_factors": len(d["g"][0].get("factors", [])),
                "n_edges": len(d["g"][1].get("production_edges", []))})
    print(f"{e['wid']} route {d['g'][0]['route']}: criterion {'EQUAL' if dc is None else 'DIFFERS ' + dc}; integrand {'EQUAL' if di is None else 'DIFFERS ' + di}; descriptive {'equal' if dd is None else 'differs ' + dd}")
dst = ROOT / "out" / "v1_compare_groups.json"
dst.write_text(json.dumps(out, indent=1))
print(f"{len(out) - bad} of {len(out)} groups equal between p and g; wrote {dst}")
sys.exit(0 if bad == 0 else 1)
