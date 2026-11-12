"""levin-phase-floor prompt 06j, S-a: the representation seams of a captured cosmological group call, with
their kinds. New in 06j.

``tools.filon_reference.production_edges`` (the tool at the tree, b5626ecb's) walks the captured
integrand's closures and returns the merged x edges at which some ingredient changes representation. This
module walks the same closures in the same way and keeps every point **before** merging, with its kind and
source:
- ``phase_knot``: each active factor's phase spline knots (``fac.phase._spline.t``), source the factor's
  phase label and its sign;
- ``amp_knot``: the Green's function amplitude spline's knots (``G._Gk.sin_amplitude._spline.t``);
- ``table_node``: each distinct cumulative table's nodes (``t._u_asc``), source the table's label and who
  uses it (a factor's leading phase table, or a q/r factor's friction table);
- ``switch_mid``: its nearest-node switch midpoints (``asc[:-1] + (asc[1:] - asc[:-1]) / 2``, the tool's
  formula);
- ``break``: its declared breaks (``t._breaks``).
``check_against_tool`` merges the points by the tool's own rule (1e-9) and compares with
``production_edges(call)[0]`` exactly.

Reads only; changes nothing.
"""

import math

import numpy as np


def _closure_dict(fn):
    return dict(zip(fn.__code__.co_freevars, [c.cell_contents for c in fn.__closure__]))


def seams(call):
    """Every seam strictly inside the call's span, as dicts {x, kind, source}, sorted by x (duplicates kept,
    one entry per (x, kind, source))."""
    a, b = (float(v) for v in call["x_span"])
    ta = call["f"][0].__closure__[2].cell_contents
    cl = _closure_dict(ta.__wrapped__)
    active = call["theta"]["theta"].__closure__[0].cell_contents
    pts = []
    tables = []  # (table, user)
    for sign, fac in active:
        ph = fac.phase
        lab = f"phase {getattr(ph, '_label', '?')} (sign {sign}, factor {type(fac).__name__})"
        for t in ph._spline.t:
            pts.append((float(t), "phase_knot", lab))
        tables.append((ph._leading._table, f"leading table of {lab}"))
    for name in ("q_factor", "r_factor"):
        f = cl[name]
        if hasattr(f, "_f") and hasattr(f._f, "_friction_F"):
            tables.append((f._f._friction_F._table, f"friction table of {name} ({type(f).__name__})"))
    G = cl["G_factor"]
    if hasattr(G, "_Gk"):
        for t in G._Gk.sin_amplitude._spline.t:
            pts.append((float(t), "amp_knot", f"G amplitude spline ({type(G).__name__})"))
    seen = {}
    for t, user in tables:
        if id(t) in seen:
            seen[id(t)][1].append(user)
            continue
        seen[id(t)] = (t, [user])
    for t, users in seen.values():
        lab = f"table {getattr(t, '_label', '?')} [{'; '.join(users)}]"
        asc = np.asarray(t._u_asc, dtype=float)
        for v in asc.tolist():
            pts.append((float(v), "table_node", lab))
        for v in (asc[:-1] + (asc[1:] - asc[:-1]) / 2).tolist():
            pts.append((float(v), "switch_mid", lab))
        if t._breaks is not None:
            for v in np.asarray(t._breaks).ravel():
                pts.append((float(v), "break", lab))
    out = sorted({(x, k, s) for x, k, s in pts if a < x < b})
    return [{"x": x, "kind": k, "source": s} for x, k, s in out], len(seen)


def merged(call, sm):
    """The tool's merge rule applied to the seams' x values (production_edges, verbatim in effect)."""
    a, b = (float(v) for v in call["x_span"])
    inner = sorted({s["x"] for s in sm})
    edges = [a]
    for v in inner:
        if v - edges[-1] > 1e-9:
            edges.append(v)
    if b - edges[-1] <= 1e-9:
        edges.pop()
    edges.append(b)
    return edges


def check_against_tool(call, sm, n_tables):
    import tools.filon_reference as F

    edges, nt = F.production_edges(call)
    mine = merged(call, sm)
    return {"equal": [float(v).hex() for v in edges] == [float(v).hex() for v in mine] and nt == n_tables,
            "n_edges_tool": len(edges), "n_edges_mine": len(mine), "n_tables_tool": nt, "n_tables_mine": n_tables}


def held(sm_x, a, b):
    """Indices of seams strictly inside (a, b); sm_x is the sorted array of seam x values."""
    lo = int(np.searchsorted(sm_x, a, side="right"))
    hi = int(np.searchsorted(sm_x, b, side="left"))
    return list(range(lo, hi))


def distance(sm_x, a, b):
    """For a span holding no seam: (distance from the nearer end to the nearest seam, from the centre), in units
    of the width; (inf, inf) if there are no seams."""
    if len(sm_x) == 0:
        return math.inf, math.inf
    w = b - a
    c = (a + b) / 2
    i = int(np.searchsorted(sm_x, c))
    cand = [sm_x[j] for j in (i - 1, i) if 0 <= j < len(sm_x)]
    d_end = min(min(abs(x - a), abs(x - b)) for x in cand) / w
    d_cen = min(abs(x - c) for x in cand) / w
    return d_end, d_cen


def null_fraction(sm_x, a0, b0, depth):
    """Fraction of the 2**depth cells of [a0, b0) (uniform, width (b0 - a0) 2**-depth) holding at least one seam."""
    n = 2**depth
    w = (b0 - a0) / n
    cells = {int(math.floor((x - a0) / w)) for x in sm_x if a0 < x < b0}
    return len(cells) / n
