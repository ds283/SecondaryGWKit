"""Scratch (not a record): a Filon reference for R|together|980's four-factor group calls at HEAD,
against the Levin values at HEAD and in B2's checkpoint."""

import json
import sys
from pathlib import Path

REPO = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
sys.path.insert(0, str(REPO / "docs/handover/b2/scripts"))
sys.path.insert(0, str(REPO))
import b2_cells  # noqa: E402
import importlib  # noqa: E402

Q = importlib.import_module("ComputeTargets.QuadSourceIntegral")  # noqa: E402
from tools import filon_reference as FR  # noqa: E402
import inspect  # noqa: E402

_src = inspect.getsource(FR.production_edges)
_src = _src.replace(
    "tables.append(ph._leading._table)",
    "tables.append(ph._leading._table) if hasattr(ph._leading, '_table') else None",
)
_src = _src.replace(
    "xs.update(float(t) for t in ph._spline.t)",
    "xs.update(float(t) for t in ph._spline.t) if hasattr(ph, '_spline') else None",
)
_src = _src.replace(
    "tables.append(f._f._friction_F._table)",
    "tables.append(f._f._friction_F._table) if hasattr(f._f._friction_F, '_table') else None",
)
_src = _src.replace(
    "xs.update(float(t) for t in G._Gk.sin_amplitude._spline.t)",
    "xs.update(float(t) for t in G._Gk.sin_amplitude._spline.t) if hasattr(G._Gk.sin_amplitude, '_spline') else None",
)
_ns = {}
exec(_src.replace("def production_edges", "def tolerant_edges"), FR.__dict__, _ns)
tolerant_edges = _ns["tolerant_edges"]

shape, x_resp = sys.argv[1], float(sys.argv[2])
labels = sys.argv[3].split(",")
captured = []
real = Q.adaptive_levin_sincos


def wrap(x_span, f, theta=None, **kw):
    out = real(x_span, f, theta=theta, **kw)
    lab = kw.get("notify_label", "")
    for g in labels:
        if lab.endswith(f"group {g}"):
            captured.append(
                (g, {"x_span": x_span, "f": f, "theta": theta, "kw": kw}, out)
            )
    return out


Q.adaptive_levin_sincos = wrap
ARM = sys.argv[4] if len(sys.argv) > 4 else "base"
if ARM == "phase":
    import ComputeTargets.phase_groups as PG

    PG._OscillatoryT.theta_deriv = lambda self, log_z: self.phase.theta_deriv(
        log_z, x_is_log=True, log_derivative=True
    )
elif ARM == "noderiv":
    Q.LEVIN_USE_THETA_DERIV = False
print("arm", ARM)
rec = b2_cells.run_cell(shape, x_resp, "R", instrument=False)
old = {
    json.loads(l)["unit"]: json.loads(l)["data"]
    for l in open(
        REPO / "var/runs/handover-13-residue-20261027T110832/checkpoint.jsonl"
    )
}[f"R|{shape}|{x_resp:g}"]
from math import exp

for g, call, out in captured:
    a, b = call["x_span"]
    zmin, zmax = exp(a) - 1, exp(b) - 1
    og = [
        x
        for x in old["groups"]
        if x["label"] == g
        and abs(x["z_min"] - zmin) <= 1e-9 * zmin
        and abs(x["z_max"] - zmax) <= 1e-9 * zmax
    ]
    try:
        edges, ntab = tolerant_edges(call)
        en = f"tolerant, {len(edges)} edges, {ntab} tables"
    except Exception as exc:  # noqa: BLE001
        edges, en = None, f"none ({type(exc).__name__}: {exc})"
    try:
        ref = FR.reference(
            call, tolerance=abs(out["abserr"]) * 1e-3, route="B", edges=edges, workers=1
        )
    except Exception as exc:  # noqa: BLE001
        print(g, "reference failed:", type(exc).__name__, exc, "edges:", en)
        continue
    R = float.fromhex(ref["value_hex"])
    print(f"{g} z in ({zmin:.6g}, {zmax:.6g}) edges {en}")
    print(f"  ref {R:.17e} scatter {ref['scatter']:.3e}")
    print(
        f"  HEAD {out['value']:.17e} abserr {out['abserr']:.3e} conv {out['converged']} regions {out['num_regions']}"
        f"  |HEAD-ref|/abserr {abs(out['value'] - R) / out['abserr']:.3g}"
    )
    for x in og:
        rv = x.get("raw_value", x["value"])
        ra = x.get("raw_abserr", x["abserr"])
        print(
            f"  B2   {rv:.17e} abserr {ra:.3e} conv {x['converged']} regions {x['num_regions']}  |B2-ref|/abserr {abs(rv - R) / ra:.3g}"
        )
