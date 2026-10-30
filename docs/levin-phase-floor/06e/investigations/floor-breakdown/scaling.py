"""Measure 3 and 4: [0, 1] partitioned uniformly into 2^d leaves, d = 6, 8, 10, 12, 14, each leaf
solved by 06d's own subregion solve (which picks Levin or the CC fallback by its phase gate),
with every floor term and the same-scheme mp round-off per leaf. No exact integrals (not needed:
round-off is double minus same-scheme mp). Writes scaling.json (summary) and
scaling_leaves_<case>_d<d>.json (per leaf).

Usage: scaling.py [depth ...]
"""

import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402

lq = C.lq
DEPTHS = (6, 8, 10, 12, 14)
WORKERS = 4
CHUNK = 256


def work(args):
    ci, spans = args
    case = C.CASES[ci]
    rows = []
    for span in spans:
        r = C.analyse_leaf(case, span, exact=False)
        if r["kind"] == "CC":
            # the fine nodes' displacement from the exact nodes, in units of 2^-53
            grid, _ = lq.chebyshev_matrices(span, 2 * C.ORDER - 1)
            with mp.workdps(C.MP_DPS):
                a, b = mp.mpf(span[0]), mp.mpf(span[1])
                mid, half = (a + b) / 2, (b - a) / 2
                n = 2 * C.ORDER - 2
                disp = [
                    float((mp.mpf(grid[k]) - (mid + half * mp.cos(mp.pi * k / n))) * 2**53)
                    for k in range(len(grid))
                ]
            r["node_disp_u"] = [round(d, 6) for d in disp]
        r.pop("mp_value", None)
        rows.append(r)
    return ci, rows


def med(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.median(xs)) if xs else None


def summarise(case, d, rows):
    xr = C.x_root(case)
    lev = [r for r in rows if r["kind"] == "Levin"]
    cc = [r for r in rows if r["kind"] == "CC"]
    terms_L = ["L1_eps_f_h2", "L2_G1_over_G0", "L3_maxG1k2_over_G0"]
    terms_C = ["C1_n4_uS", "C2_argument", "C3_node"]
    s = {
        "depth": d,
        "h": 2.0**-d,
        "leaves": len(rows),
        "levin": len(lev),
        "cc": len(cc),
        "inf_floor": sum(1 for r in rows if not np.isfinite(r["floor_code"])),
        "sum_floor": math.fsum(r["floor_code"] for r in rows if np.isfinite(r["floor_code"])),
        "terms": {},
    }
    for t, group in [(t, lev) for t in terms_L] + [(t, cc) for t in terms_C]:
        vals = [r["terms"][t] for r in group]
        s["terms"][t] = {
            "sum": math.fsum(v for v in vals if np.isfinite(v)),
            "median": med(vals),
            "max": max(vals) if vals else None,
        }
    s["levin_k2_dominates"] = sum(1 for r in lev if r["k2_dominates"])
    s["levin_L3_with_k2_sum"] = math.fsum(
        r["terms"]["L3_maxG1k2_over_G0"] for r in lev if r["k2_dominates"] and np.isfinite(r["floor_code"])
    )
    s["cc_node_theta_part_sum"] = math.fsum(r["C3_node_theta_part"] for r in cc)
    s["cc_node_fprime_part_sum"] = math.fsum(r["C3_node_fprime_part"] for r in cc)
    s["roundoff"] = {
        "median_abs_levin": med([abs(r["roundoff"]) for r in lev]),
        "median_abs_levin_rest": med([abs(r["roundoff_rest_part"]) for r in lev]),
        "median_abs_levin_endpoint": med([abs(r["roundoff_endpoint_phase_part"]) for r in lev]),
        "median_abs_cc": med([abs(r["roundoff"]) for r in cc]),
        "median_abs_cc_node": med([abs(r["roundoff_node_part"]) for r in cc]),
        "median_abs_cc_rest": med([abs(r["roundoff_rest_part"]) for r in cc]),
        "sum_abs_all": math.fsum(abs(r["roundoff"]) for r in rows),
        "abs_sum_all": abs(math.fsum(r["roundoff"] for r in rows)),
        "sum_abs_levin": math.fsum(abs(r["roundoff"]) for r in lev),
        "abs_sum_levin": abs(math.fsum(r["roundoff"] for r in lev)),
        "sum_abs_levin_rest": math.fsum(abs(r["roundoff_rest_part"]) for r in lev),
        "abs_sum_levin_rest": abs(math.fsum(r["roundoff_rest_part"] for r in lev)),
        "abs_sum_levin_endpoint": abs(math.fsum(r["roundoff_endpoint_phase_part"] for r in lev)),
        "sum_abs_cc": math.fsum(abs(r["roundoff"]) for r in cc),
        "abs_sum_cc": abs(math.fsum(r["roundoff"] for r in cc)),
        "sum_abs_cc_node": math.fsum(abs(r["roundoff_node_part"]) for r in cc),
        "abs_sum_cc_node": abs(math.fsum(r["roundoff_node_part"] for r in cc)),
    }
    s["median_ratio_floor_over_roundoff_levin"] = med(
        [r["floor_code"] / abs(r["roundoff"]) for r in lev if r["roundoff"] != 0]
    )
    s["median_ratio_floor_over_roundoff_cc"] = med(
        [r["floor_code"] / abs(r["roundoff"]) for r in cc if r["roundoff"] != 0]
    )
    s["median_ratio_C3_over_node_part"] = med(
        [r["terms"]["C3_node"] / abs(r["roundoff_node_part"]) for r in cc if r["roundoff_node_part"] != 0]
    )
    above = [r for r in rows if abs(r["roundoff"]) > r["floor_code"]]
    above_rest = [r for r in lev if abs(r["roundoff_rest_part"]) > r["floor_code"]]
    s["leaves_roundoff_above_floor"] = len(above)
    s["levin_leaves_rest_above_floor"] = len(above_rest)
    s["worst_rest_over_floor_levin"] = max(
        [abs(r["roundoff_rest_part"]) / r["floor_code"] for r in lev if np.isfinite(r["floor_code"])],
        default=None,
    )
    s["worst_rest_over_floor_levin_leaf"] = None
    if lev:
        w = max(lev, key=lambda r: abs(r["roundoff_rest_part"]) / r["floor_code"] if np.isfinite(r["floor_code"]) else 0)
        s["worst_rest_over_floor_levin_leaf"] = {
            k: w[k] for k in ("a", "b", "phase_span", "G0", "G1", "G1_over_G0", "k2_dominates", "floor_code", "roundoff", "roundoff_rest_part", "roundoff_endpoint_phase_part")
        }
    s["worst_roundoff_over_floor_cc"] = max(
        [abs(r["roundoff"]) / r["floor_code"] for r in cc], default=None
    )
    # where x_r is
    hold = [r for r in rows if r["a"] <= xr <= r["b"]]
    s["x_r_leaf"] = [
        {k: r.get(k) for k in ("a", "b", "kind", "floor_code", "roundoff", "terms")} for r in hold
    ]
    # the leaves with the largest floor and the largest |roundoff|
    top_floor = sorted([r for r in rows if np.isfinite(r["floor_code"])], key=lambda r: -r["floor_code"])[:3]
    top_ro = sorted(rows, key=lambda r: -abs(r["roundoff"]))[:3]
    s["top_floor_leaves"] = [{"a": r["a"], "kind": r["kind"], "floor": r["floor_code"], "dist_to_x_r": r["a"] + r["h"] / 2 - xr} for r in top_floor]
    s["top_roundoff_leaves"] = [{"a": r["a"], "kind": r["kind"], "roundoff": r["roundoff"], "dist_to_x_r": r["a"] + r["h"] / 2 - xr} for r in top_ro]
    # node displacement pattern: identical in every CC leaf of this depth within one binade?
    pats = {}
    for r in cc:
        key = (math.floor(math.log2(r["a"])) if r["a"] > 0 else None, tuple(r["node_disp_u"]))
        pats[key] = pats.get(key, 0) + 1
    s["cc_node_disp_patterns"] = len(pats)
    s["cc_node_disp_max_u"] = max((max(abs(x) for x in r["node_disp_u"]) for r in cc), default=None)
    s["cc_node_bound_delta_u"] = max((r["node_delta"] * 2**53 for r in cc), default=None)
    return s


def main():
    depths = tuple(int(a) for a in sys.argv[1:]) or DEPTHS
    out_path = HERE / "scaling.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {"runs": {}}
    out["header_" + "_".join(map(str, depths))] = C.header(__file__)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for d in depths:
            N = 2**d
            spans = [(i / N, (i + 1) / N) for i in range(N)]
            for ci, case in enumerate(C.CASES):
                t0 = time.perf_counter()
                tasks = [(ci, spans[i : i + CHUNK]) for i in range(0, N, CHUNK)]
                rows = []
                for _, part in ex.map(work, tasks):
                    rows.extend(part)
                s = summarise(case, d, rows)
                s["wall_s"] = time.perf_counter() - t0
                out["runs"][f"{case['name']} d={d}"] = s
                (HERE / f"scaling_leaves_{case['name'].replace(' ', '_').replace('=', '')}_d{d}.json").write_text(
                    json.dumps(rows, default=str)
                )
                out_path.write_text(json.dumps(out, indent=1, default=str))
                print(
                    f"{case['name']} d={d}: levin {s['levin']} cc {s['cc']} inf {s['inf_floor']} "
                    f"sum_floor {s['sum_floor']:.3e} | "
                    + " ".join(f"{t}={v['sum']:.2e}" for t, v in s["terms"].items())
                    + f" | sum|ro| {s['roundoff']['sum_abs_all']:.2e} |sum ro| {s['roundoff']['abs_sum_all']:.2e}"
                    f" | ro>floor {s['leaves_roundoff_above_floor']} rest>floor {s['levin_leaves_rest_above_floor']}"
                    f" | {s['wall_s']:.0f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
