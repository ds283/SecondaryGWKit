"""levin-phase-floor prompt 07 section 6: the census's cost projection, from 06m's census and from the K0 units' timings at
g. New in 07, after 06m's census_projection_06m.py (a record, not edited). Read only.

- **A unit's time** is what its worker spends on it: ``rebuild_s + wall_s`` plus every post-timed record's seconds
  (``scan_s``, ``scan_07_s``, ``w3.seconds``, ``deep_lines.seconds``, ``cc_floor_s``, ``declared_06h_s``,
  ``checked_06g_s``, ``analytic_06h.seconds``), each missing one counted 0.
- **The ratio** rho(class) = sum over the class's K0 units of the time at g / the same sum in 06m's census, per class
  (rtol, in the 720 or not; the 720 from 06k's K0 list's ``in_720``).
- **The projection:** P = 06m's census wall (18,465 s) x (sum over 06m's 18,560 units of rho(class) x its time) /
  (sum of 06m's times); and, for comparison, the same with p's K0 timings (the harness's added records at 06m's code)
  and 06m's own K0 run wall (330 s) against this run's two K0 stage walls.

Writes OUT_JSON and prints it. Usage: python -B census_projection_07.py K0_G_DIR K0_P_DIR OUT_JSON
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import CHECKPOINT06M_SHA256, G_TREE, RUN06M, header, sha256  # noqa: E402

WALL_06M = 18465.0
K0_UNITS = G_TREE / "docs" / "levin-phase-floor" / "06l" / "out" / "k0" / "units.json"


def t_unit(d):
    t = (d.get("rebuild_s") or 0.0) + (d.get("wall_s") or 0.0)
    for k in ("scan_s", "scan_07_s", "cc_floor_s", "declared_06h_s", "checked_06g_s"):
        t += d.get(k) or 0.0
    for k in ("w3", "deep_lines", "analytic_06h"):
        t += (d.get(k) or {}).get("seconds") or 0.0
    return t


def main():
    g_dir, p_dir, out_p = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    ck = RUN06M / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07: the census's cost projection",
                 [ck, K0_UNITS, g_dir / "units.jsonl", p_dir / "units.jsonl"])
    if sha256(ck) != CHECKPOINT06M_SHA256:
        raise SystemExit("06m's checkpoint has another SHA-256")
    k0 = {u["uid"]: u for u in json.loads(K0_UNITS.read_text())["units"]}
    rows720 = {u["row"] for u in k0.values() if u["in_720"]}
    t06 = {}
    with open(ck) as f:
        for line in f:
            r = json.loads(line)
            t06[r["unit"]] = (t_unit(r["data"]), r["data"]["rtol"], r["data"]["row"], r["data"]["wall_s"])

    def load(dirp):
        return {"g-" + json.loads(l)["unit"].split("-", 1)[1]: json.loads(l)["data"] for l in open(dirp / "units.jsonl")}

    G, P = load(g_dir), load(p_dir)
    # every row of the 720: those with a G+q+r call holding a sign change in 06m's 1e-7 scan
    sp_rows = set()
    with open(ck) as f:
        for line in f:
            r = json.loads(line)
            d = r["data"]
            if d["rtol"] == "1e-7" and any(s and s["n_sign_changes"] > 0 for s in d["scan"] or []):
                sp_rows.add(d["row"])
    if not rows720 <= sp_rows:
        raise SystemExit("a K0 unit marked in_720 is not a row of the 720")

    def cls(uid):
        _t, rt, row, _w = t06[uid]
        return (rt, row in sp_rows)

    out = {"header": hdr, "wall_06m_s": WALL_06M, "n_rows_720": len(sp_rows), "classes": {}}
    for name, X in (("g", G), ("p", P)):
        rho, num = {}, {}
        for uid, d in X.items():
            c = cls(uid)
            a = num.setdefault(c, [0.0, 0.0, 0, 0.0, 0.0])
            a[0] += t_unit(d)
            a[1] += t06[uid][0]
            a[2] += 1
            a[3] += d["wall_s"]
            a[4] += t06[uid][3]
        for c, (tx, t6, n, wx, w6) in num.items():
            rho[c] = tx / t6
            out["classes"].setdefault(f"{c[0]}|720={c[1]}", {})[name] = {"n": n, "time_k0": tx, "time_06m": t6,
                                                                       "rho_time": tx / t6, "rho_wall_s": wx / w6}
        tot6 = sum(v[0] for v in t06.values())
        totx = sum(v[0] * rho[cls(u)] for u, v in t06.items())
        out[f"projection_{name}_s"] = WALL_06M * totx / tot6
        out[f"ratio_{name}"] = totx / tot6
    res = json.loads((HERE.parent / "out" / "k0_results.json").read_text())["results"]
    walls = {r["name"]: r["wall_s"] for r in res}
    out["k0_stage_walls_s"] = {"06m_k0": 330.0, "p": walls.get("41_k0_run_p"), "g": walls.get("61_k0_run_g")}
    out["projection_from_k0_stage_wall_g_s"] = WALL_06M * walls["61_k0_run_g"] / 330.0
    out["sum_time_06m_s"] = sum(v[0] for v in t06.values())
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "header"}, indent=1))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
