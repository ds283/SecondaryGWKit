"""Orchestrator check 5 (06j review): for two W calls, |v - ref|/e for 06g and 06h from the four readings and the
two census checkpoints (read here, by unit and index), and each reading's usable clause.
python3 c5_partW.py  (standard library only)
"""

import json
from decimal import Decimal, getcontext

getcontext().prec = 40
V = "/Users/ds283/Documents/Code/SecondaryGWKit/var"
CK = {"06g": f"{V}/runs/levin-phase-floor-06g-census-20261101T114315/checkpoint.jsonl",
      "06h": f"{V}/runs/levin-phase-floor-06h-census-20261101T005037/checkpoint.jsonl"}
CALLS = {"f01": ("g-1e-7-0-12959", 1), "f06": ("g-1e-8-0-12959", 1), "c04": ("g-1e-7-1-35052", 6)}
REF_SID = {"f01": "f01", "f06": "f01", "c04": "c04"}  # a W-5 row's 1e-8 call shares its 1e-7 reference


def census(path, unit, index):
    with open(path) as f:
        for line in f:
            if f'"unit": "{unit}"' in line:
                d = json.loads(line)
                if d["unit"] == unit:
                    (c,) = [x for x in d["data"]["calls"] if x["index"] == index]
                    return c


for sid, (unit, index) in CALLS.items():
    rd = {}
    for impl in ("qawo", "fcc"):
        for res in (1, 2):
            r = json.load(open(f"{V}/levin-phase-floor-06j/out/readings/{REF_SID[sid]}-{impl}-r{res}.json"))
            tm = r["tmap"]
            segs_nonmono = [s for s in r["segments"] if not s.get("monotone", True)]
            usable = bool(tm["all_monotone"]) or all(s.get("direct", False) for s in segs_nonmono)
            rd[f"{impl}-r{res}"] = Decimal(r["value_30"])
            print(f"{sid} reading {REF_SID[sid]}-{impl}-r{res}: unit {r['unit']} #{r['index']} route {r['route']} "
                  f"all_monotone {tm['all_monotone']} direct_segments {tm.get('direct_segments')} "
                  f"non-monotone segments {len(segs_nonmono)} -> usable {usable}; value {r['value_30']}")
    g, h = census(CK["06g"], unit, index), census(CK["06h"], unit, index)
    vals = list(rd.values())
    spread = max(vals) - min(vals)
    eg, eh = Decimal(g["abserr"]["repr"]), Decimal(h["abserr"]["repr"])
    tol = min(eg, eh) / 100
    print(f"{sid} {unit} #{index}: spread of four readings {spread:.4e}, tolerance min(e)/100 {tol:.4e}, "
          f"spread/tol {spread / tol:.3e}")
    for name, ref in rd.items():
        rg = abs(Decimal(g["value"]["repr"]) - ref) / eg
        rh = abs(Decimal(h["value"]["repr"]) - ref) / eh
        print(f"  ref {name}: |v_g-ref|/e_g {rg:.4f} (06g regions {g['num_regions']} depth {g['max_depth']} "
              f"conv {g['converged']}); |v_06h-ref|/e_06h {rh:.4f} (06h regions {h['num_regions']} depth "
              f"{h['max_depth']} conv {h['converged']})")
