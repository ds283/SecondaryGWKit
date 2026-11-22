"""levin-phase-floor prompt 07d Part A, section 2.6: Route B's projection features, 07c Part A's model inputs. New in 07d.

``features`` is 07's (07c Part A's copy scripts/from_07c_measurement/refs_projection_07.py, sha256
8d9a5b856108a47825618f8bb4bb230fec4fe16eba74f8803cb9ca641ec6d65b) and ``feats`` is 07c's (its refs_projection_07c.py, sha256
3706a4b457f8037f37f347c7ba35a5cca0287914efc4be7d686adfe962c06d29), both copied verbatim: those modules cannot be imported here,
since refs_projection_07 imports refs_common, whose location check fails one level deeper. ``split_steep`` is leafref_07d's
(07c's, unchanged), at 07c's SPLIT = 1.8.
"""

import numpy as np

from leafref_07d import split_steep

SPLIT = 1.8


def features(call):
    from tools.filon_reference import production_edges

    edges, _ = production_edges(call)
    th, thd = call["theta"]["theta"], call["theta"]["theta_deriv"]
    y = np.cos(np.pi * np.arange(33) / 32)
    out = {"n_seg": len(edges) - 1, "span_q": 0.0, "span_f": 0.0, "D_q": 0.0, "D_f": 0.0, "n_nonmono": 0, "n_steep": 0}
    for a, b in zip(edges[:-1], edges[1:]):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        d = np.array([float(thd(mid + half * v)) for v in y])
        mono = bool(np.all(d > 0) or np.all(d < 0))
        s = half * float(np.max(np.abs(d)))
        span = abs(float(th(b)) - float(th(a)))
        if not mono:
            out["D_q"] += s
            out["D_f"] += s
            out["n_nonmono"] += 1
            continue
        out["span_q"] += span
        if float(np.max(np.abs(d)) / np.min(np.abs(d))) > 2.0:
            out["D_f"] += s
            out["n_steep"] += 1
        else:
            out["span_f"] += span
    return out


def feats(call):
    from tools.filon_reference import production_edges

    f = features(call)
    edges, _ = production_edges(call)
    lo, hi = min(edges), max(edges)
    f["n_split"] = len(edges) - 1 + len(split_steep(call, list(edges), lo, hi, SPLIT))
    th, thd = call["theta"]["theta"], call["theta"]["theta_deriv"]
    y = np.cos(np.pi * np.arange(33) / 32)
    dn = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        d = np.array([float(thd(mid + half * v)) for v in y])
        if not (np.all(d > 0) or np.all(d < 0)):
            dn += half * float(np.max(np.abs(d)))
    f["D_nonmono"] = dn
    return f
