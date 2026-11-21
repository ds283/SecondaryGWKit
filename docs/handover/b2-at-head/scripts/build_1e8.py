"""Rebuild one of R|together|X's four-factor group calls, for tools/filon_reference.

``call(x, label)`` runs B2's cell ``R|together|x`` through ``b2_cells.run_cell`` with every
``G±q±r`` group call cut to ``depth_max = 1``, captures each such call's arguments (``x_span``,
the integrands, the phase and the keywords as the cell passed them), and returns the one named.
The integrand and the phase do not depend on ``depth_max``, so the captured call is the call the
full cell makes; only the Levin run inside the capture is cut short. A spawned Filon worker
rebuilds its call this way (``call_source = ("build_1e8", "call", (x, label))``), because the
captured integrands close over local functions and cannot be pickled. Each process caches all
four calls of a cell after the first.

``tolerant_edges`` is ``ref_gqr.py``'s copy of ``tools.filon_reference.production_edges``, which
skips the tables the fixture's objects do not carry.
"""

import contextlib
import importlib
import inspect
import io
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
for _p in (REPO / "docs/handover/b2/scripts", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tools import filon_reference as FR  # noqa: E402

LABELS = ("G+q+r", "G+q-r", "G-q+r", "G-q-r")

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

_cache = {}


def call(x, label):
    key = (str(x), label)
    if key in _cache:
        return _cache[key]
    warnings.simplefilter("ignore")
    import b2_cells

    Q = importlib.import_module("ComputeTargets.QuadSourceIntegral")
    real = Q.adaptive_levin_sincos
    caps = {}

    def wrap(x_span, f, theta=None, **kw):
        g = kw.get("notify_label", "").rsplit("group ", 1)[-1]
        if g in LABELS:
            caps[g] = {"x_span": x_span, "f": f, "theta": theta, "kw": dict(kw)}
            kw = dict(kw)
            kw["depth_max"] = 1
        return real(x_span, f, theta=theta, **kw)

    Q.adaptive_levin_sincos = wrap
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            b2_cells.run_cell("together", float(x), "R", instrument=False)
    finally:
        Q.adaptive_levin_sincos = real
    for g, c in caps.items():
        _cache[(str(x), g)] = c
    return _cache[key]
