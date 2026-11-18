"""levin-phase-floor prompt 06a section 2 item 2: the row list. New in 06a.

Every QuadSourceIntegral row of A3 v2, read by (shard, serial) with sqlite3 ``mode=ro`` from A3 v2
itself (var/datastores), with its (k, q, r, z_response), its band (01's bands: low z < 9, mid
9 <= z <= 1e5, seam z > 1e5; 05's draw_sample.py ``band``), the class of its top group from 01's
census (var/levin-phase-floor-01/out/groups_01.csv: the group with the largest stored elapsed;
its label if G+r, G-r or G, else "Gqr" if its regime is Gqr, else none, as 05's draw_sample.py
``read_census`` classes it), its stored depth, regions, simple regions, total, total_abserr,
total_converged and compute time. 05's 39 rows (05's sample.json, as 06 kept it, sha256
73aa7876...) are marked by their names.

The rows are grouped by the response wavenumber k (one worker per k): ``k_key`` is
"<shard>:<k_wavenumber_exit_serial>" (the exit-time serial is the shard's own). Within a k the rows
are ordered by z_response (descending z, the block), then q, then r, by their exit-time serials.

Usage: python -I -B make_rows.py OUT_JSON   (prints a summary; writes OUT_JSON)
Imports nothing from the tree. Read-only.
"""

import collections
import csv
import json
import math
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prov import CENSUS, PRIMARY, SAMPLE05, SAMPLE_SHA256, STORE_DIR, header, sha256  # noqa: E402

PAIR = (1e-32, 1e-7)
EXPECT = 9280


def band(z):
    # 01's group_census.py / 05's draw_sample.py: low z < 9, mid 9 <= z <= 1e5, seam z > 1e5
    return "low" if z < 9.0 else ("mid" if z <= 1.0e5 else "seam")


def ro(path):
    return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)


def fx(x):
    if x is None:
        return None
    x = float(x)
    return {"repr": repr(x), "hex": x.hex()}


def read_census_classes():
    """05's draw_sample.read_census, reduced to the class of each row's top group."""
    with open(CENSUS) as f:
        lines = [line for line in f if not line.startswith("#")]
    rows = collections.OrderedDict()
    for d in csv.DictReader(lines):
        rows.setdefault((int(d["shard"]), int(d["serial"])), []).append(d)
    out = {}
    for key, gs in rows.items():
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
        out[key] = dict(
            cls=cls,
            top_label=top["label"],
            top_regime=top["regime"],
            n_groups_stored=len(gs),
            regimes_stored=sorted({g["regime"] for g in gs}),
        )
    return out


def main():
    out_json = sys.argv[1]
    lines = header(
        os.path.abspath(__file__), "levin-phase-floor 06a: the row list", [CENSUS, SAMPLE05]
    )
    if sha256(SAMPLE05) != SAMPLE_SHA256:
        raise SystemExit(f"05's sample.json sha256 {sha256(SAMPLE05)} != {SAMPLE_SHA256}")
    classes = read_census_classes()
    sample = json.loads(open(SAMPLE05).read())
    names05 = {(r["shard"], r["serial"]): (r["names"], r["role"]) for r in sample["rows"]}

    p = ro(PRIMARY)
    shards = [
        os.path.join(str(STORE_DIR), os.path.basename(f))
        for (f,) in p.execute("SELECT filename FROM shards ORDER BY serial")
    ]
    p.close()
    rows = []
    offpair = 0
    for i, sh in enumerate(shards):
        c = ro(sh)
        zmap = dict(c.execute("select serial, z from redshift"))
        kmap = dict(
            c.execute(
                "select e.serial, w.k_inv_Mpc from wavenumber_exit_time e "
                "join wavenumber w on w.serial = e.wavenumber_serial"
            )
        )
        tolmap = {s: 10.0**l for s, l in c.execute("select serial, log10_tol from tolerance")}
        for t in c.execute(
            "select serial, k_wavenumber_exit_serial, q_wavenumber_exit_serial, "
            "r_wavenumber_exit_serial, z_response_serial, atol_serial, rtol_serial, "
            '"WKB_Levin_max_depth", "WKB_Levin_num_regions", "WKB_Levin_simple_regions", '
            '"WKB_Levin_elapsed", compute_time, total, total_abserr, total_converged, '
            "total_phase_limited from QuadSourceIntegral order by serial"
        ):
            (serial, ks, qs, rs, zs, at, rt, depth, nreg, nsimp, lev_el, ct, tot, abserr, conv,
             plim) = t
            if not (
                math.isclose(tolmap[at], PAIR[0], rel_tol=1e-9)
                and math.isclose(tolmap[rt], PAIR[1], rel_tol=1e-9)
            ):
                offpair += 1
                continue
            z = zmap[zs]
            cl = classes.get((i, serial), {})
            nm = names05.get((i, serial))
            rows.append(
                dict(
                    name=f"{i}-{serial}",
                    shard=i,
                    serial=serial,
                    k_key=f"{i}:{ks}",
                    k_exit_serial=ks,
                    q_exit_serial=qs,
                    r_exit_serial=rs,
                    z_response_serial=zs,
                    k=repr(kmap[ks]),
                    q=repr(kmap[qs]),
                    r=repr(kmap[rs]),
                    z_response=repr(z),
                    band=band(z),
                    cls=cl.get("cls"),
                    top_label=cl.get("top_label"),
                    top_regime=cl.get("top_regime"),
                    n_groups_stored=cl.get("n_groups_stored", 0),
                    regimes_stored=cl.get("regimes_stored", []),
                    stored_depth=depth,
                    stored_num_regions=nreg,
                    stored_simple_regions=nsimp,
                    stored_WKB_Levin_elapsed=lev_el,
                    stored_compute_time=ct,
                    stored_total=fx(tot),
                    stored_total_abserr=fx(abserr),
                    stored_total_converged=None if conv is None else bool(conv),
                    stored_total_phase_limited=None if plim is None else bool(plim),
                    names05=None if nm is None else nm[0],
                    role05=None if nm is None else nm[1],
                )
            )
        c.close()
    # order: k_key, then z_response descending, then q, r exit serials
    rows.sort(key=lambda r: (r["shard"], r["k_exit_serial"], -float(r["z_response"]),
                             r["q_exit_serial"], r["r_exit_serial"]))
    lines.append(f"# QuadSourceIntegral rows at (atol, rtol) = {PAIR}: {len(rows)} "
                 f"(expected {EXPECT}); at any other pair: {offpair}")
    by_k = collections.Counter(r["k_key"] for r in rows)
    lines.append(f"# rows by k_key: {dict(sorted(by_k.items()))}")
    for kk in sorted(by_k):
        ks = {r["k"] for r in rows if r["k_key"] == kk}
        zs = {r["z_response_serial"] for r in rows if r["k_key"] == kk}
        lines.append(f"#   {kk}: k {sorted(ks)}; {len(zs)} z_response blocks")
    lines.append(f"# by band: {dict(collections.Counter(r['band'] for r in rows))}")
    lines.append(f"# by class: {dict(collections.Counter(str(r['cls']) for r in rows))}")
    lines.append(
        "# by stored depth bin: "
        + str(dict(collections.Counter(
            "NULL" if r["stored_depth"] is None else ("<=10" if r["stored_depth"] <= 10 else (
                "11-19" if r["stored_depth"] < 20 else "20")) for r in rows)))
    )
    marked = [r for r in rows if r["names05"]]
    lines.append(f"# 05's rows marked: {len(marked)} (sample has {len(sample['rows'])}); "
                 f"depth-20 role: {sum(1 for r in marked if r['role05'] == 'd20')}")
    missing = [n for (key, (n, _)) in names05.items() if not any((r['shard'], r['serial']) == key for r in rows)]
    lines.append(f"# 05's rows not found: {missing}")
    if len(rows) != EXPECT or len(marked) != len(sample["rows"]):
        lines.append("# !! STOP: population or sample mismatch")
    doc = {
        "header": lines,
        "census": str(CENSUS),
        "census_sha256": sha256(CENSUS),
        "sample05": str(SAMPLE05),
        "sample05_sha256": sha256(SAMPLE05),
        "store_read": str(STORE_DIR),
        "rows": rows,
    }
    with open(out_json, "x") as f:
        json.dump(doc, f, indent=0)
    for l in lines:
        print(l)
    print(f"# wrote {out_json} sha256 {sha256(out_json)}")
    return 0 if len(rows) == EXPECT and len(marked) == len(sample["rows"]) else 2


if __name__ == "__main__":
    sys.exit(main())
