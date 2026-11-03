"""Orchestrator check 4 (06j review), written from AdaptiveLevin.levin_quadrature._check_leaf at the tree.
(a) S-a for three sampled leaves (s07, s08, s19): production_edges on the capture, whether [a, b] holds an
    edge, else the distance to the nearest in widths; and each leaf's r against 06g's census region pickle,
    located from the census checkpoint's own unit record.
(b) s07: the trust check on the leaf, S-b's first two halvings along the branch holding the seam, and S-d's
    split at the seam.
PYTHONPATH=<worktree>:<06j scripts> python -B c4_partS.py
"""

import hashlib
import json
import pickle

import numpy as np
import ray
import ray.cloudpickle as cp

import AdaptiveLevin.levin_quadrature as LQ
from tools.filon_reference import production_edges

V = "/Users/ds283/Documents/Code/SecondaryGWKit/var"
RUN = f"{V}/runs/levin-phase-floor-06g-census-20261101T114315"
DIAG = f"{V}/levin-phase-floor-06j/review/checks/levin_diagnostics"
LEAVES = {  # sid: (unit, index, start, end) as partS_leaves.csv gives them
    "s07": ("g-1e-7-1-35492", 3, 18.88686723504442, 18.88688274229765),
    "s08": ("g-1e-7-3-1188", 5, 20.285344640613594, 20.28534923130716),
    "s19": ("g-1e-8-0-38451", 3, 17.232905840347527, 17.23293268228719),
}


def census_regions(unit, index):
    with open(f"{RUN}/checkpoint.jsonl") as f:
        for line in f:
            if f'"unit": "{unit}"' in line:
                d = json.loads(line)
                if d["unit"] == unit:
                    rec = d["data"]
    with open(f"{RUN}/{rec['pickle']}", "rb") as f:
        f.seek(rec["pickle_offset"])
        blob = f.read(rec["pickle_length"])
    assert hashlib.sha256(blob).hexdigest() == rec["pickle_record_sha256"]
    (reg,) = [x for x in pickle.loads(blob)["regions"] if x["index"] == index]
    return reg


class Rule:
    def __init__(self, cap):
        self.cap = cap
        self.A = LQ._Basis_SinCos(cap["theta"])
        self.cache = {}
        kw = cap["kw"]
        self.order = kw.get("chebyshev_order", LQ.DEFAULT_LEVIN_CHEBSHEV_ORDER)
        self.rtol = kw.get("rtol", LQ.DEFAULT_LEVIN_RELTOL)

    def solve(self, a, b):
        return LQ._adaptive_levin_subregion(
            (a, b), self.cap["f"], self.A, id_label="06j-review", chebyshev_order=self.order,
            rtol=self.rtol, diagnostics_path=DIAG, vectorize_cache=self.cache)

    def check(self, a, b):
        P = self.solve(a, b)
        c = (b + a) / 2.0
        L, R = self.solve(a, c), self.solve(c, b)
        cL, cR = (c + a) / 2.0, (b + c) / 2.0
        q = (self.solve(a, cL)["value"] + self.solve(cL, c)["value"]) + (
            self.solve(c, cR)["value"] + self.solve(cR, b)["value"])
        h = L["value"] + R["value"]
        d1, d2 = float(np.fabs(P["value"] - h)), float(np.fabs(h - q))
        r = (0.0 if d2 == 0 else np.inf) if d1 == 0 else d2 / d1
        phi = P.get("phase_err", 0.0) or 0.0
        trusted = bool(r < 0.25 or (r < 1.0 and d1 <= phi))
        return dict(a=a, b=b, c=c, d1=d1, d2=d2, r=r, phi=phi, d1_over_phi=d1 / phi, trusted_b=trusted,
                    is_direct=bool(P.get("is_direct", False)))


for sid, (unit, index, a, b) in LEAVES.items():
    cap = cp.loads(open(f"{V}/levin-phase-floor-06j/state/inputs/{sid}.pkl", "rb").read())
    assert cap["unit"] == unit and cap["index"] == index
    edges, ntab = production_edges(cap)
    inner = np.array(edges[1:-1])
    w = b - a
    held = inner[(inner >= a) & (inner <= b)]
    if held.size:
        sa = f"holds {held.size} edge(s) at {[repr(float(x)) for x in held]}"
    else:
        dist = np.min(np.minimum(np.abs(inner - a), np.abs(inner - b))) / w
        sa = f"holds no edge; nearest edge {dist:.6g} widths from its nearer end"
    reg = census_regions(unit, index)
    (i,) = np.nonzero((reg["start"] == a) & (reg["end"] == b))[0]
    print(f"{sid} {unit} #{index} [{a!r}, {b!r}] width {w:.6g}: {len(edges)} edges ({ntab} tables); {sa}; "
          f"census leaf depth {reg['depth'][i]} trusted {reg['trusted'][i]} r {reg['convergence_ratio'][i]!r}")
    if sid != "s07":
        continue
    rule = Rule(cap)
    s = float(held[0])
    top = rule.check(a, b)
    print(f"  s07 rule on the leaf: delta1 {top['d1']!r} delta2 {top['d2']!r} r {top['r']!r} phi {top['phi']!r} "
          f"is_direct {top['is_direct']}")
    lo, hi = a, b
    for depth in (21, 22):
        c = (hi + lo) / 2.0
        lo, hi = (lo, c) if s <= c else (c, hi)
        assert lo <= s <= hi
        x = rule.check(lo, hi)
        print(f"  S-b seam branch depth {depth} [{lo!r}, {hi!r}]: r {x['r']!r} d1/phi {x['d1_over_phi']:.6g} "
              f"trusted_b {x['trusted_b']}")
    for side, (l, r) in (("left", (a, s)), ("right", (s, b))):
        x = rule.check(l, r)
        print(f"  S-d {side} of seam [{l!r}, {r!r}] width {r - l:.6g}: r {x['r']!r} d1/phi {x['d1_over_phi']:.6g} "
              f"trusted_b {x['trusted_b']}")
print(f"ray initialised: {ray.is_initialized()}")
