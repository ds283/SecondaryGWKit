"""levin-phase-floor prompt 07d Part A, section 2.1: one row per call of 07c's census, for the draws. New in 07d.

Reads 07c's census checkpoint (RUN07C, SHA-256 checked before and after) and its spec files (which unit is in which
k band: the spec name ``<rtol>-k<band>_<n>``). Per call (group calls ``calls``, analytic ``other_calls``), from the
checkpoint's call record only:
  sid ``<unit>|<g|a><index>``; unit, rtol, shard, serial, kind, index, notify_label, k (from the label), k band; value,
  abserr (hex), converged, num_regions, max_depth; leaves (levin, cc, trusted, untrusted, floor_limited); windows and
  window pieces, guarded regions and guards at depth_max (``guard_07``); the chase stop's three counts and its carrier
  (route, trusted before the stop, share of the call's abserr: ``chase_07c``); elapsed.
Writes state/census_index_07d.pkl ({"header", "rows": [dict]}) and out/census_index_07d_summary.json.
Usage: PYTHONPATH=<tree>:<scripts> python -B census_index_07d.py
"""

import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07C_SHA256, RUN07C, WIDER, header, sha256  # noqa: E402


def num(x):
    if isinstance(x, dict) and "repr" in x:
        return float(x["repr"])
    return x


def main():
    out_p = WIDER / "state" / "census_index_07d.pkl"
    sum_p = WIDER / "out" / "census_index_07d_summary.json"
    for p in (out_p, sum_p):
        if p.exists():
            raise SystemExit(f"{p} exists")
    ck = RUN07C / "checkpoint.jsonl"
    if sha256(ck) != CHECKPOINT07C_SHA256:
        raise SystemExit("07c's checkpoint has another SHA-256")
    band = {}
    for sp in sorted(RUN07C.glob("spec_*.json")):
        name = json.loads(sp.read_text())["name"]
        m = re.match(r"(1e-\d)-k(\d)_\d+$", name)
        for u in json.loads(sp.read_text())["units"]:
            band[u["uid"]] = int(m.group(2))
    rows = []
    n_units = 0
    status = Counter()
    with open(ck) as f:
        for line in f:
            r = json.loads(line)
            d = r["data"]
            status[d.get("status")] += 1
            if d.get("status") != "done":
                continue
            n_units += 1
            for key, kind in (("calls", "group"), ("other_calls", "analytic")):
                for c in d.get(key) or []:
                    lab = c["notify_label"]
                    mk = re.search(r"k=([0-9.e+\-]+)/Mpc", lab)
                    g7 = c.get("guard_07") or {}
                    ch = c.get("chase_07c") or {}
                    car = ch.get("carrier") or None
                    lv = c.get("leaves") or {}
                    rows.append({
                        "sid": f"{d['unit']}|{kind[0]}{c['index']}", "unit": d["unit"], "rtol": d["rtol"],
                        "shard": d["shard"], "serial": d["serial"], "kind": kind, "index": c["index"],
                        "notify_label": lab, "k": float(mk.group(1)) if mk else None, "k_band": band.get(d["unit"]),
                        "regime": c.get("regime"), "group": c.get("group"),
                        "value": c["value"]["hex"], "abserr": c["abserr"]["hex"], "converged": bool(c["converged"]),
                        "num_regions": c["num_regions"], "max_depth": c["max_depth"],
                        "n_levin": lv.get("levin"), "n_cc": lv.get("cc"), "n_trusted": lv.get("trusted"),
                        "n_untrusted": lv.get("untrusted"), "n_floor_limited": lv.get("floor_limited"),
                        "n_windows": g7.get("n_windows"), "n_window_pieces": g7.get("num_window_pieces"),
                        "n_guarded": g7.get("num_guarded_regions"), "n_guard_at_depth_max": g7.get("num_guard_at_depth_max"),
                        "num_chase_stops": c.get("num_chase_stops"), "num_chase_stopped_leaves": c.get("num_chase_stopped_leaves"),
                        "num_chase_noise_bisections": c.get("num_chase_noise_bisections"),
                        "chase_by_route": ch.get("by_route"),
                        "carrier": None if car is None else {"type": car.get("type"), "trusted_before": car.get("trusted_before"),
                                                             "share": num(car.get("share")), "depth": car.get("depth"),
                                                             "start": (car.get("start") or {}).get("hex"),
                                                             "end": (car.get("end") or {}).get("hex"),
                                                             "forced_direct": car.get("forced_direct")},
                        "elapsed": c.get("elapsed"), "unit_wall_s": d.get("wall_s"), "budget_s": d.get("budget_s"),
                    })
    if sha256(ck) != CHECKPOINT07C_SHA256:
        raise SystemExit("07c's checkpoint changed")
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: 07c's census, one row per call", [ck])
    out_p.write_bytes(pickle.dumps({"header": hdr, "rows": rows}, protocol=5))
    cnt = Counter((r["kind"], r["rtol"], r["k_band"]) for r in rows)
    summ = {"header": hdr, "units_done": n_units, "status": dict(status), "n_calls": len(rows),
            "by_kind_rtol_band": {f"{k[0]}|{k[1]}|k{k[2]}": v for k, v in sorted(cnt.items(), key=str)},
            "n_chase_acted": sum(1 for r in rows if r["num_chase_stops"]),
            "n_chase_acted_analytic": sum(1 for r in rows if r["num_chase_stops"] and r["kind"] == "analytic"),
            "n_with_cc": sum(1 for r in rows if r["n_cc"]), "n_with_levin": sum(1 for r in rows if r["n_levin"]),
            "n_with_windows": sum(1 for r in rows if r["n_windows"]),
            "n_guard_at_depth_max": sum(1 for r in rows if r["n_guard_at_depth_max"]),
            "pickle_sha256": sha256(out_p)}
    sum_p.write_text(json.dumps(summ, indent=1))
    for l in hdr:
        print(l)
    print(json.dumps({k: v for k, v in summ.items() if k != "header"}, indent=1))


if __name__ == "__main__":
    main()
