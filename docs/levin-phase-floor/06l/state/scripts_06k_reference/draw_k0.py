"""levin-phase-floor prompt 06k, Gate K0: the 100 units, by the rule written before this script ran
(state/predictions_and_rules_20261104T0915Z.md section 2). New in 06k. Read only.

U46: the g_unit of every entry of 06j's out/sample/selection.json (the committed copy in the tree).
U54: from 06g's census units (state/index_06g.json, made from its checkpoint) less U46, 9 per (rtol,
stratum): V8out (units holding a V8 lost call, whose row is outside the 720), SP720 (rows in the
720), rest; candidates sorted ascending, numpy PCG64(seed).choice(n, 9, replace=False), seed =
20261104 * 100 + 10 * s + t (s = 1, 2, 3; t = 1 for 1e-7, 2 for 1e-8).
Writes OUT_JSON: the 100 units with their source and stratum, and every count behind the draw.
Usage: python -B draw_k0.py OUT_JSON
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import header  # noqa: E402

SEL = RC.G_TREE / "docs" / "levin-phase-floor" / "06j" / "out" / "sample" / "selection.json"
V8 = RC.MAIN / "var" / "levin-phase-floor-06g" / "census" / "out" / "v8_calls.json"
V8_SHA256 = "1e9a62b0f8d95deade4f265d1fa1a8b5bbd63f587f7c9761d78882db5367c42e"
INDEX = HERE.parent / "state" / "index_06g.json"
STRATA = ("V8out", "SP720", "rest")
RTOLS = ("1e-7", "1e-8")


def rowkey(uid):
    return "-".join(uid.rsplit("-", 2)[-2:])


def main():
    out_p = Path(sys.argv[1])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06k: Gate K0's 100 units", [SEL, V8, INDEX])
    for l in hdr:
        print(l)
    if RC.sha256(V8) != V8_SHA256:
        raise SystemExit("v8_calls.json sha256 differs")
    sel = json.loads(SEL.read_text())
    u46 = [e["g_unit"] for e in sel]
    if len(set(u46)) != 46:
        raise SystemExit("U46 is not 46 distinct units")
    idx = json.loads(INDEX.read_text())
    units = idx["units"]
    if len(units) != 18560:
        raise SystemExit("the index does not hold 18,560 units")
    sp = RC.scan06a_sp()
    if len(sp) != 720:
        raise SystemExit(f"the 720 are {len(sp)}")
    v8 = json.loads(V8.read_text())
    v8_units = {c["unit"] for c in v8["lost"] if not c["stationary_point_row"]}
    # a lost call's row flag must agree with the 720
    bad_flag = [c["unit"] for c in v8["lost"] if c["stationary_point_row"] != (rowkey(c["unit"]) in sp)]
    if bad_flag:
        raise SystemExit(f"v8 rows' stationary-point flags disagree with the 720: {bad_flag[:5]}")
    pool = sorted(set(units) - set(u46))
    out = [{"uid": u, "source": "U46", "stratum": None, "sid": e["sid"]} for u, e in zip(u46, sel)]
    counts = {}
    for t, rt in enumerate(RTOLS, 1):
        cand = {s: [] for s in STRATA}
        for u in pool:
            if units[u]["rtol"] != rt:
                continue
            if rowkey(u) in sp:
                cand["SP720"].append(u)
            elif u in v8_units:
                cand["V8out"].append(u)
            else:
                cand["rest"].append(u)
        for s_i, s in enumerate(STRATA, 1):
            c = sorted(cand[s])
            counts[f"{rt}/{s}"] = len(c)
            if len(c) < 9:
                raise SystemExit(f"stratum {rt}/{s} has {len(c)} candidates: a stop")
            seed = 20261104 * 100 + 10 * s_i + t
            pick = np.random.Generator(np.random.PCG64(seed)).choice(len(c), size=9, replace=False)
            for i in pick:
                out.append({"uid": c[int(i)], "source": "U54", "stratum": s, "seed": seed, "index_in_stratum": int(i)})
    uids = [o["uid"] for o in out]
    if len(set(uids)) != 100:
        raise SystemExit("not 100 distinct units")
    for o in out:
        u = units[o["uid"]]
        o.update({"rtol": u["rtol"], "row": u["row"], "k_key": u["k_key"], "status_06g": u["status"],
                  "wall_s_06g": u["wall_s"], "rebuild_s_06g": u["rebuild_s"], "n_group_calls": u["n_group_calls"],
                  "n_group_unconverged_06g": u["n_group_unconverged"], "in_720": rowkey(o["uid"]) in sp,
                  "holds_v8_lost_outside_720": o["uid"] in v8_units})
    doc = {"header": hdr, "rule": "state/predictions_and_rules_20261104T0915Z.md section 2", "counts": counts,
           "n_720": len(sp), "n_v8_units_outside_720": len(v8_units), "units": out}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc, indent=1))
    print(f"strata candidates {counts}")
    for o in out:
        print(f"{o['uid']:>18} {o['source']} {o['stratum'] or o.get('sid')} k {o['k_key']} wall_06g {o['wall_s_06g']:.2f} "
              f"group calls {o['n_group_calls']} unconverged {o['n_group_unconverged_06g']} 720 {o['in_720']}")
    print(f"wrote {out_p} sha256 {RC.sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
