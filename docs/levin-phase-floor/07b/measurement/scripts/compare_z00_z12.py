"""levin-phase-floor 06g, addendum 6's open check: are z00's and z12's captured integrands equal? Both
captures (state/inputs/z00.pkl, z12.pkl) are loaded at g and compared by compare13.py's ``describe``
(unchanged), the criterion used for 06i's 13: the route and span; the active factors' signs and each
factor's phase spline (knots, coefficients, degree) and leading table (nodes, breaks), by bytes; the
integrand the tool reads (f_sin, f_cos, theta, theta_mod_2pi, theta_deriv at 65 points; Route B's
production_edges); and, descriptive, the keyword arguments and the phase dict's keys. Every field
that differs is listed (not only the first). Also: the two pickles' top-level keys and every scalar
field of the captured records. Writes out/compare_z00_z12.json.
Usage: PYTHONPATH=<tree g>:<this directory> python -B compare_z00_z12.py"""
import json, sys, hashlib
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import compare13 as C
import tools.filon_reference as F
def all_diffs(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present in one only")
            else:
                out += all_diffs(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: lengths {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += all_diffs(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(f"{path}: {str(a)[:100]} vs {str(b)[:100]}")
    return out
res = {}
P = {s: ROOT / "state" / "inputs" / f"{s}.pkl" for s in ("z00", "z12")}
for s, p in P.items():
    print(f"# input {p} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
print(f"# compare_z00_z12.py sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; compare13.py {hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest()}; tool {F.__file__}")
calls = {s: C.load(p) for s, p in P.items()}
D = {s: C.describe(c, F) for s, c in calls.items()}
names = ("criterion", "integrand", "descriptive")
for i, n in enumerate(names):
    d = all_diffs(D["z00"][i], D["z12"][i])
    res[n] = {"equal": not d, "differences": d, "fields": sorted(D["z00"][i])}
    print(f"{n}: {'EQUAL' if not d else 'DIFFERS'} ({len(d)} differing entries) over fields {sorted(D['z00'][i])}")
    for x in d[:20]:
        print("   ", x)
scal = {}
for s, c in calls.items():
    scal[s] = {k: (v if isinstance(v, (str, int, float, bool, type(None))) else (list(v) if isinstance(v, tuple) else type(v).__name__)) for k, v in c.items() if k not in ("f", "theta", "kw")}
d = all_diffs(scal["z00"], scal["z12"])
res["record_fields"] = {"keys": sorted(calls["z00"]), "differences": d}
print("captured records' other fields: " + ("equal" if not d else "differ: " + "; ".join(d)))
(ROOT / "out" / "compare_z00_z12.json").write_text(json.dumps(res, indent=1, default=str))
