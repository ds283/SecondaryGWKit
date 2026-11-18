"""levin-phase-floor prompt 06m section 5, the descriptive tables (not scored). New in 06m. Read only.

One pass over each of three census checkpoints: the census m (this prompt's run), route B's (06l,
levin-phase-floor-06l-census-routeB-20261105T100126, 5cb3d3c0...) and 06g's (levin-phase-floor-06g-census-20261101T114315,
03adcbba...), each checkpoint's SHA-256 checked first (m's is recorded). The 720 are 06a's stationary-point rows by
06a's rule, as 06l's records identify them (refs_common.scan06a_sp: the rows whose 1e-7 unit's scan in 06a's census
shows a sign change of theta' in any call); "outside" means outside them.

1. **The flag's cost** (m and 06g; route B's too, for the record, since Gate M makes it m's): every unconverged call
   (``converged`` false), group calls (``data.calls``) and analytic calls (``data.other_calls``) apart, outside and in
   the 720: its ``abserr`` against its request max(atol, rtol |value|) (the call's own atol and rtol as recorded), its
   ``node_limited``, and its share abserr / total_abserr of its unit's (row's) ``result.total_abserr``. Counted by
   calls, units and rows (shard-serial), with the largest ratio to the request and the largest share, and the calls
   outside the 720 listed (CSV).
2. **Cost:** each census's wall (manifest ``created`` to status.json's last ``heartbeat``, as 06l's results.md), its
   unit walls (``wall_s``) and its group-call regions (``num_regions`` over ``data.calls``), at each rtol and in all.
3. **Route B's firings** (route B's census only, from each call's ``route_06l.n_firings``): calls that fire and
   firings, group and analytic, outside and in the 720.
4. **Checked Levin leaves with delta1 / phi_P > 10** (the census m only, from each unit's region-pickle record,
   ``checked_leaves_06g``: every checked leaf, returned or bisected, with type 0 = Levin, delta1 and phase_err = phi_P):
   the count with phase_err > 0 and delta1 / phase_err > 10, by returned / bisected, by the leaf's final trust and
   by its r (convergence_ratio) in 06g's rule (b)'s bands r < 1/4, 1/4 <= r < 1, r >= 1,
   outside and in the 720, group and analytic apart; the largest ratio; leaves with phase_err <= 0 or NaN counted apart.

Usage: PYTHONPATH=<tree>:<scripts> python -B descriptive_06m.py M_RUN_DIR OUT_DIR [--processes N]
Writes OUT_DIR/descriptive_06m.json and OUT_DIR/unconverged_outside_720_{m,06g}.csv.
"""

import argparse
import csv
import datetime
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from compare_k0 import load_pickle  # noqa: E402
from prov import CHECKPOINT06G_SHA256, CHECKPOINT06L_B_SHA256, RUN06G, RUN06L_B, header, sha256  # noqa: E402

RATIO = 10.0
SP = None


def H(x):
    return None if x is None else float.fromhex(x["hex"] if isinstance(x, dict) else x)


def scan(args):
    """One unit's numbers. args: (census label, checkpoint line, pickle dir or None)."""
    label, line, pdir = args
    r = json.loads(line)
    d = r["data"]
    out = {"unit": r["unit"], "rtol": d["rtol"], "row": f"{d['shard']}-{d['serial']}", "status": d["status"],
           "wall_s": d.get("wall_s"), "group_regions": 0, "n_group": 0, "n_analytic": 0, "unconv": [],
           "fired": {"group": [0, 0], "analytic": [0, 0]}, "levin": None}
    res = d.get("result") or {}
    tae = H(res.get("total_abserr"))
    for kind, key in (("group", "calls"), ("analytic", "other_calls")):
        for c in d.get(key) or []:
            out["n_" + kind] += 1
            if kind == "group":
                out["group_regions"] += int(c["num_regions"])
            rb = c.get("route_06l")
            if rb is not None and int(rb.get("n_firings") or 0) > 0:
                out["fired"][kind][0] += 1
                out["fired"][kind][1] += int(rb["n_firings"])
            if not c["converged"]:
                ab, v = H(c["abserr"]), H(c["value"])
                req = max(float(c["atol"]), float(c["rtol"]) * abs(v))
                out["unconv"].append({
                    "kind": kind, "index": c["index"], "notify_label": c["notify_label"], "abserr": ab, "value": v,
                    "atol": float(c["atol"]), "rtol": float(c["rtol"]), "request": req,
                    "ratio": ab / req if req > 0 else math.inf, "within": ab <= req,
                    "node_limited": c.get("node_limited"), "total_abserr": tae,
                    "share": ab / tae if tae and math.isfinite(tae) and tae > 0 else None,
                    "max_depth": c.get("max_depth"), "n_at_depth_max": c.get("n_at_depth_max"),
                    "num_untrusted_regions": c.get("num_untrusted_regions"),
                })
    if pdir is not None and d["status"] == "done":
        rec = load_pickle(Path(pdir) / d["pickle"], d["pickle_offset"], d["pickle_length"], d["pickle_record_sha256"])
        kinds = {c["index"]: "group" for c in d.get("calls") or []} | {c["index"]: "analytic" for c in d.get("other_calls") or []}
        lev = {}
        for cl in rec.get("checked_leaves_06g") or []:
            if cl is None:
                continue
            kind = kinds.get(cl["index"], "?")
            t = cl["type"] == 0
            if not t.any():
                continue
            d1, phi = cl["delta1"][t], cl["phase_err"][t]
            ret, tf = cl["returned"][t], cl["trusted_final"][t]
            a = lev.setdefault(kind, {"n_levin_checked": 0, "n_phi_nonpositive": 0, "n_gt": 0, "n_gt_returned": 0,
                                      "n_gt_bisected": 0, "n_gt_trusted": 0, "n_gt_untrusted": 0, "n_gt_trust_reset": 0,
                                      "max_ratio": 0.0, "n_gt_returned_untrusted": 0, "n_gt_r_lt_quarter": 0,
                                      "n_gt_r_quarter_to_1": 0, "n_gt_r_ge_1": 0, "n_gt_r_quarter_to_1_returned": 0,
                                      "max_ratio_r_quarter_to_1": 0.0, "n_levin_checked_r_quarter_to_1": 0})
            a["n_levin_checked"] += int(t.sum())
            ok = phi > 0
            a["n_phi_nonpositive"] += int((~ok).sum())
            ratio = d1[ok] / phi[ok]
            gt = ratio > RATIO
            a["n_gt"] += int(gt.sum())
            a["n_gt_returned"] += int((ret[ok] & gt).sum())
            a["n_gt_bisected"] += int((~ret[ok] & gt).sum())
            a["n_gt_trusted"] += int(((tf[ok] == 1) & gt).sum())
            a["n_gt_untrusted"] += int(((tf[ok] == 0) & gt).sum())
            a["n_gt_trust_reset"] += int(((tf[ok] == -1) & gt).sum())
            a["n_gt_returned_untrusted"] += int((ret[ok] & (tf[ok] == 0) & gt).sum())
            if ratio.size:
                a["max_ratio"] = max(a["max_ratio"], float(ratio.max()))
            rr = cl["convergence_ratio"][t][ok]  # r, by the bands of 06g's rule (b)
            band = (rr >= 0.25) & (rr < 1.0)
            a["n_levin_checked_r_quarter_to_1"] += int(band.sum())
            a["n_gt_r_lt_quarter"] += int((gt & (rr < 0.25)).sum())
            a["n_gt_r_quarter_to_1"] += int((gt & band).sum())
            a["n_gt_r_ge_1"] += int((gt & ~(rr < 1.0)).sum())
            a["n_gt_r_quarter_to_1_returned"] += int((gt & band & ret[ok]).sum())
            if (gt & band).any():
                a["max_ratio_r_quarter_to_1"] = max(a["max_ratio_r_quarter_to_1"], float(ratio[gt & band].max()))
        out["levin"] = lev
    return out


def walk(label, run, pdir, procs, expected_sha=None):
    ck = run / "checkpoint.jsonl"
    got = sha256(ck)
    if expected_sha is not None and got != expected_sha:
        raise SystemExit(f"{ck}: sha256 {got}, not {expected_sha}")

    def lines():
        with open(ck, "rb") as f:
            for line in f:
                yield (label, line, None if pdir is None else str(pdir))

    with mp.get_context("spawn").Pool(procs) as pool:
        rows = list(pool.imap(scan, lines(), chunksize=32))
    man = json.loads((run / "manifest.json").read_text())
    st = json.loads((run / "status.json").read_text())
    t0 = datetime.datetime.fromisoformat(man["created"])
    t1 = datetime.datetime.fromisoformat(st["heartbeat"])
    return rows, {"checkpoint": str(ck), "checkpoint_sha256": got, "created": man["created"], "last_heartbeat": st["heartbeat"],
                  "state": st.get("state"), "census_wall_s": (t1 - t0).total_seconds()}


def summarise(rows, meta, sp):
    s = {"meta": meta, "n_units": len(rows), "statuses": {}}
    for r in rows:
        s["statuses"][r["status"]] = s["statuses"].get(r["status"], 0) + 1
    cost = {}
    for rt in ("1e-7", "1e-8", "all"):
        rr = [r for r in rows if rt == "all" or r["rtol"] == rt]
        cost[rt] = {"unit_wall_s": sum(r["wall_s"] or 0.0 for r in rr), "group_call_regions": sum(r["group_regions"] for r in rr),
                    "n_group_calls": sum(r["n_group"] for r in rr), "n_analytic_calls": sum(r["n_analytic"] for r in rr)}
    s["cost"] = cost
    fl = {}
    for where in ("outside", "in720"):
        for kind in ("group", "analytic"):
            for rt in ("1e-7", "1e-8", "all"):
                calls = [(r, c) for r in rows for c in r["unconv"]
                         if c["kind"] == kind and ((r["row"] in sp) == (where == "in720")) and (rt == "all" or r["rtol"] == rt)]
                rat = [c["ratio"] for _, c in calls]
                sh = [c["share"] for _, c in calls if c["share"] is not None]
                over = [c for _, c in calls if not c["within"]]
                fl[f"{where}|{kind}|{rt}"] = {
                    "n_calls": len(calls), "n_units": len({r["unit"] for r, _ in calls}), "n_rows": len({r["row"] for r, _ in calls}),
                    "n_within_request": sum(1 for _, c in calls if c["within"]),
                    "n_over_request": len(over),
                    "n_node_limited": sum(1 for _, c in calls if c["node_limited"]),
                    "n_over_request_node_limited": sum(1 for c in over if c["node_limited"]),
                    "n_over_request_not_node_limited": sum(1 for c in over if not c["node_limited"]),
                    "max_ratio_to_request": max(rat) if rat else None,
                    "max_ratio_over_request_only": max((c["ratio"] for c in over), default=None),
                    "ratio_bands": {"<=0.1": sum(1 for x in rat if x <= 0.1), "(0.1,1]": sum(1 for x in rat if 0.1 < x <= 1),
                                    "(1,1.05]": sum(1 for x in rat if 1 < x <= 1.05), "(1.05,2]": sum(1 for x in rat if 1.05 < x <= 2),
                                    ">2": sum(1 for x in rat if x > 2)},
                    "max_share": max(sh) if sh else None,
                    "share_bands": {"<=1%": sum(1 for x in sh if x <= 0.01), "(1%,5%]": sum(1 for x in sh if 0.01 < x <= 0.05),
                                    "(5%,50%]": sum(1 for x in sh if 0.05 < x <= 0.5), ">50%": sum(1 for x in sh if x > 0.5)},
                    "n_share_none": sum(1 for _, c in calls if c["share"] is None),
                    "n_with_untrusted_leaf": sum(1 for _, c in calls if (c["num_untrusted_regions"] or 0) > 0),
                    "n_with_leaf_at_depth_max": sum(1 for _, c in calls if (c["n_at_depth_max"] or 0) > 0),
                    "n_within_with_untrusted_leaf": sum(1 for _, c in calls if c["within"] and (c["num_untrusted_regions"] or 0) > 0),
                    "n_within_not_node_limited": sum(1 for _, c in calls if c["within"] and not c["node_limited"]),
                    "n_node_limited_with_untrusted_leaf": sum(1 for _, c in calls if c["node_limited"] and (c["num_untrusted_regions"] or 0) > 0),
                    "n_neither_node_limited_nor_untrusted_leaf": sum(1 for _, c in calls if not c["node_limited"] and not (c["num_untrusted_regions"] or 0) > 0),
                }
    s["flag_cost"] = fl
    fi = {}
    for where in ("outside", "in720"):
        for kind in ("group", "analytic"):
            rr = [r for r in rows if (r["row"] in sp) == (where == "in720")]
            fi[f"{where}|{kind}"] = {"calls_firing": sum(r["fired"][kind][0] for r in rr), "firings": sum(r["fired"][kind][1] for r in rr)}
    s["firings_route_06l"] = fi
    if any(r["levin"] is not None for r in rows):
        lv = {}
        for where in ("outside", "in720"):
            for kind in ("group", "analytic"):
                a = {}
                for r in rows:
                    if (r["row"] in sp) != (where == "in720") or not r["levin"] or kind not in r["levin"]:
                        continue
                    for k, v in r["levin"][kind].items():
                        a[k] = max(a.get(k, 0.0), v) if k.startswith("max_ratio") else a.get(k, 0) + v
                lv[f"{where}|{kind}"] = a
        s["levin_delta1_over_phiP_gt_10"] = lv
    return s


def write_csv(path, rows, sp, hdr):
    with open(path, "w", newline="") as f:
        for l in hdr:
            f.write(l + "\n")
        w = csv.writer(f)
        w.writerow(["unit", "row", "rtol", "kind", "index", "notify_label", "abserr", "value", "atol", "rtol_call", "request",
                    "abserr_over_request", "within_request", "node_limited", "total_abserr", "share", "max_depth",
                    "n_at_depth_max", "num_untrusted_regions"])
        for r in rows:
            if r["row"] in sp:
                continue
            for c in r["unconv"]:
                w.writerow([r["unit"], r["row"], r["rtol"], c["kind"], c["index"], c["notify_label"], repr(c["abserr"]),
                            repr(c["value"]), repr(c["atol"]), repr(c["rtol"]), repr(c["request"]), repr(c["ratio"]),
                            c["within"], c["node_limited"], repr(c["total_abserr"]), repr(c["share"]), c["max_depth"],
                            c["n_at_depth_max"], c["num_untrusted_regions"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("m_run")
    ap.add_argument("out_dir")
    ap.add_argument("--processes", type=int, default=1)
    a = ap.parse_args()
    if a.processes > 6:
        raise SystemExit("at most six processes")
    m_run, out = Path(a.m_run).resolve(), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06m: the descriptive tables",
                 [m_run / "checkpoint.jsonl", RUN06L_B / "checkpoint.jsonl", RUN06G / "checkpoint.jsonl",
                  RC.RUN06A / "checkpoint.jsonl", HERE / "compare_k0.py"])
    for l in hdr:
        print(l, flush=True)
    sp = RC.scan06a_sp()
    print(f"# the 720: {len(sp)} rows by 06a's rule", flush=True)
    if len(sp) != 720:
        raise SystemExit(f"06a's rule gives {len(sp)} rows, not 720")
    doc = {"header": hdr, "n_720": len(sp), "ratio_threshold": RATIO, "censuses": {}}
    for label, run, pdir, exp in (("m", m_run, m_run, None), ("routeB", RUN06L_B, None, CHECKPOINT06L_B_SHA256),
                                  ("06g", RUN06G, None, CHECKPOINT06G_SHA256)):
        rows, meta = walk(label, run, pdir, a.processes, exp)
        doc["censuses"][label] = summarise(rows, meta, sp)
        if label in ("m", "06g"):
            write_csv(out / f"unconverged_outside_720_{label}.csv", rows, sp, hdr)
        print(f"# {label}: {len(rows)} units; wall {meta['census_wall_s']:.0f} s", flush=True)
        del rows
    p = out / "descriptive_06m.json"
    p.write_text(json.dumps(doc, indent=1, default=str))
    for label, s in doc["censuses"].items():
        print(f"== {label}: statuses {s['statuses']}; wall {s['meta']['census_wall_s']:.0f} s; cost {json.dumps(s['cost'])}")
        for k in ("outside|group|all", "outside|group|1e-7", "outside|group|1e-8", "outside|analytic|all", "in720|group|all",
                  "in720|analytic|all"):
            print(f"   flag {k}: {json.dumps(s['flag_cost'][k])}")
        print(f"   firings {json.dumps(s['firings_route_06l'])}")
        if "levin_delta1_over_phiP_gt_10" in s:
            print(f"   levin d1/phiP>10 {json.dumps(s['levin_delta1_over_phiP_gt_10'])}")
    print(f"wrote {p} sha256 {sha256(p)}")
    for label in ("m", "06g"):
        q = out / f"unconverged_outside_720_{label}.csv"
        print(f"wrote {q} sha256 {sha256(q)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
