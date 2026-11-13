"""levin-phase-floor prompt 07a section 6: the census's cost projection, from 07's census and from the K0 units' timings
at h and at the parent. New in 07a, after 07's census_projection_07.py (a record, not edited), whose model it keeps with
07's census in 06m's place. Read only.

- **A unit's time** is what its worker spends on it outside the producer rebuild: ``wall_s`` plus every post-timed
  record's seconds (``scan_s``, ``scan_07_s``, ``w3.seconds``, ``deep_lines.seconds``, ``cc_floor_s``,
  ``declared_06h_s``, ``checked_06g_s``, ``analytic_06h.seconds``), each missing one counted 0. ``rebuild_s`` is left out
  of the ratio (a K0 worker runs a few of its k's units, a census worker 1,160) and 07's census rebuild time is added
  back unchanged.
- **The ratio** rho(class) = sum over the class's K0 units of the time at h / the same sum in 07's census, per class
  (rtol, in the 720 or not; the 720 from 06k's K0 list's ``in_720``, checked against 07's census scan at 1e-7).
- **The projection:** P = 07's census wall (22,708 s, six workers) x (sum over 07's 18,560 units of rho(class) x its
  time, plus 07's rebuild time) / (sum of 07's times, plus 07's rebuild time); the same with the parent's K0 timings (the
  harness's added diagnostics at 07's code); and 07's own K0 stage wall at its tree g (359 s) against this run's K0 stage
  walls.

Writes OUT_JSON and prints it. Usage: python -B census_projection_07a.py K0_H_DIR K0_PARENT_DIR OUT_JSON
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT07_SHA256, H_TREE, RUN07, header, sha256  # noqa: E402

WALL_07 = 22708.0  # 07's census wall (docs/levin-phase-floor/07/census/log.md)
K0_STAGE_07_G = 359.0820007324219  # 07's 61_k0_run_g (docs/levin-phase-floor/07/census/out/k0_results.json)
K0_UNITS = H_TREE / "docs" / "levin-phase-floor" / "06l" / "out" / "k0" / "units.json"


def t_unit(d):
    t = d.get("wall_s") or 0.0
    for k in ("scan_s", "scan_07_s", "cc_floor_s", "declared_06h_s", "checked_06g_s"):
        t += d.get(k) or 0.0
    for k in ("w3", "deep_lines", "analytic_06h"):
        t += (d.get(k) or {}).get("seconds") or 0.0
    return t


def main():
    h_dir, p_dir, out_p = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ck = RUN07 / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07a: the census's cost projection",
                 [ck, K0_UNITS, h_dir / "units.jsonl", p_dir / "units.jsonl"])
    if sha256(ck) != CHECKPOINT07_SHA256:
        raise SystemExit("07's checkpoint has another SHA-256")
    k0 = {u["uid"]: u for u in json.loads(K0_UNITS.read_text())["units"]}
    rows720 = {u["row"] for u in k0.values() if u["in_720"]}
    t07 = {}
    sp_rows = set()
    with open(ck) as f:
        for line in f:
            r = json.loads(line)
            d = r["data"]
            t07[r["unit"]] = (t_unit(d), d["rtol"], d["row"], d["wall_s"], d.get("rebuild_s") or 0.0)
            if d["rtol"] == "1e-7" and any(s and s["n_sign_changes"] > 0 for s in d["scan"] or []):
                sp_rows.add(d["row"])
    if not rows720 <= sp_rows:
        raise SystemExit("a K0 unit marked in_720 is not a row of the 720")

    def load(dirp):
        return {"g-" + json.loads(l)["unit"].split("-", 1)[1]: json.loads(l)["data"] for l in open(dirp / "units.jsonl")}

    H, P = load(h_dir), load(p_dir)

    def cls(uid):
        _t, rt, row, _w, _rb = t07[uid]
        return (rt, row in sp_rows)

    out = {"header": hdr, "wall_07_s": WALL_07, "n_rows_720": len(sp_rows), "classes": {}}
    for name, X in (("h", H), ("parent", P)):
        rho, num = {}, {}
        for uid, d in X.items():
            c = cls(uid)
            a = num.setdefault(c, [0.0, 0.0, 0, 0.0, 0.0])
            a[0] += t_unit(d)
            a[1] += t07[uid][0]
            a[2] += 1
            a[3] += d["wall_s"]
            a[4] += t07[uid][3]
        for c, (tx, t7, n, wx, w7) in num.items():
            rho[c] = tx / t7
            out["classes"].setdefault(f"{c[0]}|720={c[1]}", {})[name] = {"n": n, "time_k0": tx, "time_07": t7,
                                                                       "rho_time": tx / t7, "rho_wall_s": wx / w7}
        rb7 = sum(v[4] for v in t07.values())
        tot7 = sum(v[0] for v in t07.values()) + rb7
        totx = sum(v[0] * rho[cls(u)] for u, v in t07.items()) + rb7
        out[f"projection_{name}_s"] = WALL_07 * totx / tot7
        out[f"ratio_{name}"] = totx / tot7
    res = json.loads((HERE.parent / "out" / "k0_results.json").read_text())["results"]
    walls = {r["name"]: r["wall_s"] for r in res}
    out["k0_stage_walls_s"] = {"07_k0_g": K0_STAGE_07_G, "parent": walls.get("41_k0_run_parent"), "h": walls.get("61_k0_run_h")}
    out["projection_from_k0_stage_wall_h_s"] = WALL_07 * walls["61_k0_run_h"] / K0_STAGE_07_G
    out["sum_time_07_s"] = sum(v[0] for v in t07.values())
    out["sum_rebuild_07_s"] = sum(v[4] for v in t07.values())
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "header"}, indent=1))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
