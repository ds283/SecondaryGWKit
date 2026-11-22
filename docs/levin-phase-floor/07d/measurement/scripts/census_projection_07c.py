"""levin-phase-floor prompt 07c section 6 (and section 9.4): the census's wall-clock projection, from 07b's census and from
Gate Y0's K0 units at h. Copied from 07b's census_projection_07b.py (a record, not edited), changed only where marked
"prompt 07c": the "before" is 07b's census (prov.RUN07B, checkpoint 396784bc..., wall 45,016 s, 07b's census log), every
"07a" below reads "07b", and one figure is added: 07b's own projection ran short by the ratio of its measured wall to its
central projection (45,016 / 37,615.6 = 1.197; 07c section 9.4 and the dispatch: "state the projection with that
margin"), so ``projection_central_x_07b_shortfall_s`` is the central projection times that ratio, and
``projection_highest_x_07b_shortfall_s`` the highest margin's. The limit is 48 hours (07c section 6 and section 9.4).
07b's text follows.

levin-phase-floor prompt 07b section 6 (and section 9.4): the census's wall-clock projection, from 07a's census and from
Gate U0's K0 units at h. New in 07b, after 07a's census_projection_07a.py (a record, not edited), whose unit time and
class ratio it keeps, with a schedule simulation and stated margins added. Read only.

- **A unit's time** t (07a's definition): ``wall_s`` plus every post-timed record's seconds (``scan_s``, ``scan_07_s``,
  ``w3.seconds``, ``deep_lines.seconds``, ``cc_floor_s``, ``declared_06h_s``, ``checked_06g_s``,
  ``analytic_06h.seconds``), each missing one counted 0. ``rebuild_s`` (the producers, unchanged by 07b) is kept apart.
- **The ratio** rho(class) = sum over the class's K0 units of t at h / the same sum in 07a's census. Three classings:
  (i) 07a's: (rtol, in the 720 or not); (ii) (rtol, k_key), the census's workers; (iii) one ratio for all.
- **Model 1 (07a's form):** P1 = 07a's census wall x (sum over 07a's 18,560 units of rho x t, plus rebuild) / (sum of t,
  plus rebuild), with classing (i).
- **Model 2 (the schedule):** the census runs one worker per (rtol, k), 8 per rtol, 1,160 units each, at most six at
  once, every 1e-7 worker before any 1e-8 worker, started in run_census.plan's order (all have 1,160 units, so by k_key)
  as a slot frees. Each worker's duration is simulated as the sum of its units' rho x t plus their rebuild_s plus its
  overhead in 07a (the worker's span in 07a, from its first unit's start to its last record, less its units' t and
  rebuild_s). The simulation is checked first on 07a's own times (rho = 1) against 07a's measured wall (23,284 s); the
  ratio of measured to simulated is applied as a calibration. Run under classings (i), (ii) and (iii).
- **Margins:** (a) the overhead scaled by rho as well (pessimistic: pickling and JSON grow with regions); (b) rho at the
  upper end of a bootstrap over the K0 units (2,000 resamples within each class, its 97.5th percentile, seed 7); (c) the
  over-budget pass: every census unit whose rho x t exceeds 300 s re-runs at up to 3,600 s, added serially at the end
  (pessimistic), counted from the largest per-unit K0 ratio applied to 07a's unit walls.
Writes OUT_JSON and prints a summary. Usage: python -B census_projection_07b.py K0_H_DIR K0_PARENT_DIR OUT_JSON
"""

import datetime
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07A_SHA256, H_TREE, RUN07A, header, sha256  # noqa: E402
from prov import CHECKPOINT07B_SHA256, RUN07B  # noqa: E402  prompt 07c
RUN07A, CHECKPOINT07A_SHA256 = RUN07B, CHECKPOINT07B_SHA256  # prompt 07c: the names kept, 07b's census read

WALL_07A = 23284.0  # 07a's census wall (docs/levin-phase-floor/07a/census/log.md)
WALL_07A = 45016.0  # prompt 07c: 07b's census wall (docs/levin-phase-floor/07b/census/log.md), the name kept
SHORTFALL_07B = 45016.0 / 37615.55663062413  # prompt 07c: 07b's measured wall over its central projection
K0_UNITS = H_TREE / "docs" / "levin-phase-floor" / "06l" / "out" / "k0" / "units.json"
LIMIT_S = 48 * 3600.0  # 07b section 9.4; prompt 07c: 07c section 9.4, the same


def t_unit(d):
    t = d.get("wall_s") or 0.0
    for k in ("scan_s", "scan_07_s", "cc_floor_s", "declared_06h_s", "checked_06g_s"):
        t += d.get(k) or 0.0
    for k in ("w3", "deep_lines", "analytic_06h"):
        t += (d.get(k) or {}).get("seconds") or 0.0
    return t


def iso(s):
    return datetime.datetime.fromisoformat(s).timestamp()


def simulate(workers_by_rtol, dur, slots=6):
    """The census's schedule: per rtol, workers in order, each started as a slot frees; the next rtol after all end."""
    t0 = 0.0
    for rt in ("1e-7", "1e-8"):
        free = [t0] * slots
        ends = []
        for w in workers_by_rtol[rt]:
            i = min(range(slots), key=lambda j: free[j])
            free[i] = free[i] + dur[w]
            ends.append(free[i])
        t0 = max(ends)
    return t0


def main():
    h_dir, p_dir, out_p = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ck = RUN07A / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07c: the census's wall-clock projection",  # prompt 07c
                 [ck, K0_UNITS, h_dir / "units.jsonl", p_dir / "units.jsonl"])
    if sha256(ck) != CHECKPOINT07A_SHA256:
        raise SystemExit("07b's checkpoint has another SHA-256")  # prompt 07c
    k0 = {"h-" + u["uid"].split("-", 1)[1]: u for u in json.loads(K0_UNITS.read_text())["units"]}
    rows720 = {u["row"] for u in k0.values() if u["in_720"]}
    U = {}  # 07a's units: uid -> dict
    with open(ck) as f:
        for line in f:
            r = json.loads(line)
            d = r["data"]
            U[r["unit"]] = {"t": t_unit(d), "rb": d.get("rebuild_s") or 0.0, "rtol": d["rtol"], "row": d["row"],
                            "k": d["k_key"], "worker": d["worker"], "wall": d["wall_s"], "rec": iso(r["recorded"])}
    # 07a's workers: order, span and overhead
    workers = {}
    for uid, u in U.items():
        workers.setdefault(u["worker"], []).append(uid)
    order = {rt: sorted([w for w in workers if w.startswith(rt + "-")]) for rt in ("1e-7", "1e-8")}
    over = {}
    span = {}
    for w, uids in workers.items():
        uids.sort(key=lambda x: U[x]["rec"])
        first = U[uids[0]]
        start = first["rec"] - first["t"] - first["rb"]
        span[w] = U[uids[-1]]["rec"] - start
        over[w] = span[w] - sum(U[x]["t"] + U[x]["rb"] for x in uids)
    sim07a = simulate(order, {w: span[w] for w in workers})
    calib = WALL_07A / sim07a

    def load(dirp):
        return {json.loads(l)["unit"]: json.loads(l)["data"] for l in open(dirp / "units.jsonl")}

    H, P = load(h_dir), load(p_dir)
    if set(H) != set(k0) or set(P) != set(k0):
        raise SystemExit("the K0 runs do not hold K0's 100 units")
    classings = {
        "i_rtol_720": lambda uid: (U[uid]["rtol"], U[uid]["row"] in rows720),
        "ii_rtol_k": lambda uid: (U[uid]["rtol"], U[uid]["k"]),
        "iii_all": lambda uid: "all",
    }
    out = {"header": hdr, "wall_07a_s": WALL_07A, "limit_s": LIMIT_S, "n_units_07a": len(U),
           "sum_t_07a_s": sum(u["t"] for u in U.values()), "sum_rebuild_07a_s": sum(u["rb"] for u in U.values()),
           "workers_07a": {w: {"span_s": span[w], "overhead_s": over[w], "n": len(workers[w])} for w in sorted(workers)},
           "schedule_check": {"simulated_07a_s": sim07a, "measured_07a_s": WALL_07A, "calibration": calib}}
    per_unit = {uid: {"t_07a": U[uid]["t"], "t_h": t_unit(H[uid]), "t_parent": t_unit(P[uid]),
                      "ratio_h": t_unit(H[uid]) / U[uid]["t"] if U[uid]["t"] > 0 else None,
                      "wall_h": H[uid]["wall_s"], "wall_07a": U[uid]["wall"], "status_h": H[uid]["status"]}
                for uid in sorted(k0)}
    out["k0_units"] = per_unit
    out["k0_sum_t"] = {"07a": sum(v["t_07a"] for v in per_unit.values()), "h": sum(v["t_h"] for v in per_unit.values()),
                       "parent": sum(v["t_parent"] for v in per_unit.values())}
    out["k0_max_wall_h_s"] = max(v["wall_h"] for v in per_unit.values())
    rs = sorted(v["ratio_h"] for v in per_unit.values() if v["ratio_h"] is not None)
    out["k0_unit_ratio_h"] = {"min": rs[0], "median": rs[len(rs) // 2], "max": rs[-1]}
    rng = random.Random(7)
    res = {}
    for cname, cf in classings.items():
        groups = {}
        for uid in k0:
            groups.setdefault(cf(uid), []).append(uid)
        rho = {c: sum(per_unit[u]["t_h"] for u in us) / sum(per_unit[u]["t_07a"] for u in us) for c, us in groups.items()}
        rho_p = {c: sum(per_unit[u]["t_parent"] for u in us) / sum(per_unit[u]["t_07a"] for u in us) for c, us in groups.items()}
        boot = {}
        for c, us in groups.items():
            vals = []
            for _ in range(2000):
                smp = [rng.choice(us) for _ in us]
                vals.append(sum(per_unit[u]["t_h"] for u in smp) / sum(per_unit[u]["t_07a"] for u in smp))
            vals.sort()
            boot[c] = vals[int(0.975 * len(vals))]

        def rho_of(uid, table):
            c = cf(uid)
            if c in table:
                return table[c]
            return sum(per_unit[u]["t_h"] for u in k0) / sum(per_unit[u]["t_07a"] for u in k0)  # a class K0 lacks

        def project(table, scale_overhead):
            dur = {}
            for w, uids in workers.items():
                body = sum(U[x]["t"] * rho_of(x, table) + U[x]["rb"] for x in uids)
                rw = (sum(U[x]["t"] * rho_of(x, table) for x in uids) / sum(U[x]["t"] for x in uids)) if scale_overhead else 1.0
                dur[w] = body + over[w] * rw
            return simulate(order, dur) * calib, dur

        tot_t = sum(u["t"] for u in U.values())
        rb = sum(u["rb"] for u in U.values())
        p1 = WALL_07A * (sum(u["t"] * rho_of(x, rho) for x, u in U.items()) + rb) / (tot_t + rb)
        p2, dur2 = project(rho, False)
        p2a, _ = project(rho, True)
        p2b, _ = project(boot, True)
        res[cname] = {"rho_h": {str(c): v for c, v in rho.items()}, "rho_parent": {str(c): v for c, v in rho_p.items()},
                      "rho_h_boot975": {str(c): v for c, v in boot.items()},
                      "n_k0_by_class": {str(c): len(us) for c, us in groups.items()},
                      "model1_07a_form_s": p1, "model2_schedule_s": p2, "model2_overhead_scaled_s": p2a,
                      "model2_overhead_scaled_boot975_s": p2b,
                      "model2_worker_durations_s": dur2}
    out["projections"] = res
    # (c) the over-budget pass, pessimistic: the largest K0 unit ratio applied to every 07a unit's wall
    rmax = out["k0_unit_ratio_h"]["max"]
    ob = [x for x, u in U.items() if u["wall"] * rmax > 300.0]
    out["overbudget_pessimistic"] = {"ratio_applied": rmax, "n_units_over_300s": len(ob), "units": ob[:50],
                                     "added_serial_s_at_3600_each": 3600.0 * len(ob)}
    allp = [v for r in res.values() for k, v in r.items() if k.startswith("model") and isinstance(v, float)]
    out["projection_central_s"] = res["i_rtol_720"]["model2_schedule_s"]
    out["projection_highest_s"] = max(allp)
    out["projection_highest_with_overbudget_s"] = max(allp) + out["overbudget_pessimistic"]["added_serial_s_at_3600_each"]
    out["within_48h"] = out["projection_highest_with_overbudget_s"] < LIMIT_S
    # prompt 07c: 07b's shortfall applied
    out["shortfall_07b"] = SHORTFALL_07B
    out["projection_central_x_07b_shortfall_s"] = out["projection_central_s"] * SHORTFALL_07B
    out["projection_highest_x_07b_shortfall_s"] = out["projection_highest_with_overbudget_s"] * SHORTFALL_07B
    out["within_48h_x_07b_shortfall"] = out["projection_highest_x_07b_shortfall_s"] < LIMIT_S
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=1, default=str))
    summ = {k: v for k, v in out.items() if k not in ("header", "k0_units", "workers_07a", "projections")}
    print(json.dumps(summ, indent=1, default=str))
    for c, r in res.items():
        print(c, json.dumps({k: v for k, v in r.items() if k != "model2_worker_durations_s"}, default=str))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
