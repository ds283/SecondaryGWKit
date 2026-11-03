"""levin-phase-floor prompt 05 section 6: draw the rows by the rule fixed in state/sample.md.

Reads 01's census (groups_01.csv) and, for the NULL-depth controls only, the store read-only
(sqlite3 mode=ro, as 01's census read it). Writes state/sample.json and prints out/sample.txt's
text to stdout. Imports nothing from either tree.

Usage: python -I -B draw_sample.py OUT_JSON
"""

import collections
import csv
import json
import math
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prov import CENSUS, PRIMARY, STORE_DIR, header, sha256  # noqa: E402

PAIR = (1e-32, 1e-7)
R03 = [
    ("R1", 2, 37527),
    ("R2", 0, 1797),
    ("R3", 1, 37670),
    ("R4", 2, 37720),
    ("R5", 0, 37966),
    ("R6", 2, 34728),
    ("R7", 2, 34820),
]
BANDS_D20 = ["low", "mid"]
CLASSES = ["G+r", "G-r", "G", "Gqr"]
BANDS_CTRL = ["low", "mid", "seam"]


def band(z):
    # 01's group_census.py: low z < 9, mid 9 <= z <= 1e5, seam z > 1e5
    return "low" if z < 9.0 else ("mid" if z <= 1.0e5 else "seam")


def ro(path):
    return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)


def read_census():
    with open(CENSUS) as f:
        lines = [line for line in f if not line.startswith("#")]
    rows = collections.OrderedDict()
    for d in csv.DictReader(lines):
        rows.setdefault((int(d["shard"]), int(d["serial"])), []).append(d)
    out = []
    for key, gs in rows.items():
        g0 = gs[0]
        top = gs[0]
        for g in gs:
            if float(g["elapsed"]) > float(top["elapsed"]):
                top = g
        if top["label"] in ("G+r", "G-r", "G"):
            cls = top["label"]
        elif top["regime"] == "Gqr":
            cls = "Gqr"
        else:
            cls = None
        depth = g0["WKB_Levin_max_depth"]
        out.append(
            dict(
                shard=key[0],
                serial=key[1],
                k=g0["k"],
                q=g0["q"],
                r=g0["r"],
                z_response=float(g0["z_response"]),
                band=g0["band"],
                geometry=g0["geometry"],
                depth=None if depth == "" else int(depth),
                compute_time=float(g0["compute_time"]),
                total=g0["total"],
                total_abserr=g0["total_abserr"],
                total_converged=g0["total_converged"],
                top_label=top["label"],
                top_regime=top["regime"],
                top_elapsed=float(top["elapsed"]),
                cls=cls,
                has_G_group=any("G" in g["label"] for g in gs),
                n_groups=len(gs),
                F=float(g0["F"]),
            )
        )
    return out


def walk(cands, n):
    cands = sorted(cands, key=lambda r: (r["compute_time"], r["shard"], r["serial"]))
    taken, ks = [], set()
    for r in cands:
        if r["k"] in ks:
            continue
        taken.append(r)
        ks.add(r["k"])
        if len(taken) == n:
            break
    return taken


def null_candidates():
    p = ro(PRIMARY)
    shards = [
        os.path.join(str(STORE_DIR), os.path.basename(f))
        for (f,) in p.execute("SELECT filename FROM shards ORDER BY serial")
    ]
    p.close()
    c0 = ro(shards[0])
    zmap = dict(c0.execute("select serial, z from redshift"))
    kmap = dict(
        c0.execute(
            "select e.serial, w.k_inv_Mpc from wavenumber_exit_time e "
            "join wavenumber w on w.serial = e.wavenumber_serial"
        )
    )
    tolmap = {
        s: 10.0**l for s, l in c0.execute("select serial, log10_tol from tolerance")
    }
    c0.close()
    out = []
    n_pop = 0
    n_null = 0
    for i, sh in enumerate(shards):
        c = ro(sh)
        for t in c.execute(
            "select serial, k_wavenumber_exit_serial, q_wavenumber_exit_serial, "
            'r_wavenumber_exit_serial, z_response_serial, atol_serial, rtol_serial, "WKB_Levin_max_depth", '
            "compute_time, total, total_abserr, total_converged from QuadSourceIntegral"
        ):
            serial, ks, qs, rs, zs, at, rt, depth, ct, tot, abserr, conv = t
            if not (
                math.isclose(tolmap[at], PAIR[0], rel_tol=1e-9)
                and math.isclose(tolmap[rt], PAIR[1], rel_tol=1e-9)
            ):
                continue
            n_pop += 1
            if depth is not None:
                continue
            n_null += 1
            if ct is None or ct > 60.0:
                continue
            z = zmap[zs]
            out.append(
                dict(
                    shard=i,
                    serial=serial,
                    # the census writes k with repr(); do the same, so distinct-k compares alike
                    k=repr(kmap[ks]),
                    q=repr(kmap[qs]),
                    r=repr(kmap[rs]),
                    z_response=z,
                    band=band(z),
                    depth=None,
                    compute_time=ct,
                    total=repr(tot),
                    total_abserr=repr(abserr),
                    total_converged=conv,
                    top_label=None,
                    top_regime=None,
                    cls=None,
                    has_G_group=False,
                    n_groups=0,
                )
            )
        c.close()
    return out, n_pop, n_null, shards


def main():
    out_json = sys.argv[1]
    lines = header(
        os.path.abspath(__file__), "levin-phase-floor 05 section 6: the rows", [CENSUS]
    )
    rows = read_census()
    lines.append(f"# census rows (distinct (shard, serial)): {len(rows)}")

    sample = []

    def add(r, role, cell, names):
        key = (r["shard"], r["serial"])
        for s in sample:
            if (s["shard"], s["serial"]) == key:
                s["names"] += names
                s["cells"].append(cell)
                return
        d = dict(r)
        d.update(role=role, cells=[cell], names=list(names))
        sample.append(d)

    lines.append("")
    lines.append("== depth-20 cells: band x class, up to 3 rows each")
    for b in BANDS_D20:
        for cls in CLASSES:
            cands = [
                r
                for r in rows
                if r["band"] == b
                and r["cls"] == cls
                and r["depth"] == 20
                and r["compute_time"] <= 900.0
            ]
            taken = walk(cands, 3)
            lines.append(
                f"   {b:4s} {cls:4s}: {len(cands)} candidates, "
                f"{len({r['k'] for r in cands})} distinct k; taken {len(taken)}"
            )
            for j, r in enumerate(taken):
                add(r, "d20", f"{b}/{cls}", [f"{b}-{cls}-{j + 1}"])
    by_key = {(r["shard"], r["serial"]): r for r in rows}
    for name, shard, serial in R03:
        r = by_key[(shard, serial)]
        if r["depth"] != 20:
            raise SystemExit(f"{name} is not a depth-20 row in the census")
        add(r, "d20", "03", [name])

    lines.append("")
    lines.append(
        "== controls by band: depth < 20, compute_time <= 60 s, a group with G; 2 each"
    )
    for b in BANDS_CTRL:
        cands = [
            r
            for r in rows
            if r["band"] == b
            and r["depth"] is not None
            and r["depth"] < 20
            and r["compute_time"] <= 60.0
            and r["has_G_group"]
        ]
        taken = walk(cands, 2)
        lines.append(f"   {b:4s}: {len(cands)} candidates; taken {len(taken)}")
        for j, r in enumerate(taken):
            add(r, "control", f"ctrl/{b}", [f"ctrl-{b}-{j + 1}"])

    nulls, n_pop, n_null, shards = null_candidates()
    lines.append("")
    lines.append(
        f"== controls with NULL depth, from the store (mode=ro): population {n_pop} rows at "
        f"{PAIR}; NULL depth {n_null}; with compute_time <= 60 s {len(nulls)}"
    )
    for sh in shards:
        lines.append(f"   read {sh}")
    taken = walk(nulls, 2)
    for j, r in enumerate(taken):
        add(r, "control", "ctrl/NULL", [f"ctrl-NULL-{j + 1}"])

    for i, s in enumerate(sample):
        s["index"] = i
        s["name"] = "+".join(s["names"])

    lines.append("")
    lines.append(
        f"== the sample: {sum(1 for s in sample if s['role'] == 'd20')} depth-20 rows, "
        f"{sum(1 for s in sample if s['role'] == 'control')} controls"
    )
    lines.append(
        "   #  name                      (shard, serial)  band  class top-group  depth  "
        "compute_time  k  q  r  z_response"
    )
    for s in sample:
        lines.append(
            f"  {s['index']:2d}  {s['name']:<24s}  ({s['shard']}, {s['serial']:>5d})  "
            f"{s['band']:4s}  {str(s['cls']):4s} {str(s['top_label']):6s}  {str(s['depth']):4s}  "
            f"{s['compute_time']:10.4f}  {float(s['k']):.4g}  {float(s['q']):.4g}  "
            f"{float(s['r']):.4g}  {s['z_response']:.6g}"
        )
    d20 = [s for s in sample if s["role"] == "d20"]
    ctrl = [s for s in sample if s["role"] == "control"]
    lines.append("")
    lines.append(f"L0 rows: depth-20 {d20[0]['name']}; control {ctrl[0]['name']}")
    with open(out_json, "w") as f:
        json.dump(
            {
                "census": str(CENSUS),
                "census_sha256": sha256(CENSUS),
                "script_sha256": sha256(os.path.abspath(__file__)),
                "rows": sample,
                "L0": {"d20": d20[0]["name"], "control": ctrl[0]["name"]},
            },
            f,
            indent=1,
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
