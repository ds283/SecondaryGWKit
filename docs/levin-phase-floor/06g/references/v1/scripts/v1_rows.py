"""levin-phase-floor prompt 06g section 5, addendum 1: V1's 12 rows in R23 and R24. New in 06g.

The rows are those of the rehearsal agent's ``rehearsal/out/qsi_rows.csv`` with ``V1 == VIOLATES``
(read only), identified by their physical key (k, q, r, z_response, as that CSV prints them). Each
store (R23 for the tree p, R24 for g) is read with sqlite3 ``mode=ro`` on the real files in
var/datastores (its primary's ``shards`` table names the shard files): for each row its shard,
serial, k exit-time serial, total and total_abserr, and from its metadata's
``WKB_Levin.subintervals[1].groups`` the group labelled ``G+q+r`` (the rehearsal agent's
v1_violations_detail.txt: "sub-interval 1, group G+q+r" carries each row's move): its value,
abserr, converged, phase_limited and regions. These value and abserr are v23/e23 (R23) and v24/e24
(R24), and each is checked against v1_violations_detail.txt's text, where it prints them.

Writes ../out/v1_rows.json: ids w00..w11 ordered by (k, z_response). Reads only.

Usage: python -I -B v1_rows.py
"""

import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
REH = MAIN / "var" / "levin-phase-floor-06g" / "rehearsal" / "out"
STORES = {"p": MAIN / "var" / "datastores" / "levin-phase-floor-06g-r23.sqlite",
          "g": MAIN / "var" / "datastores" / "levin-phase-floor-06g-r24.sqlite"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ro(p):
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def read(primary):
    con = ro(primary)
    shards = [(r[0], r[1]) for r in con.execute("SELECT serial, filename FROM shards ORDER BY serial")]
    con.close()
    out = {}
    kq = ("(SELECT w.k_inv_Mpc FROM wavenumber_exit_time e JOIN wavenumber w "
          "ON w.serial = e.wavenumber_serial WHERE e.serial = q.{})")
    for shard, name in shards:
        con = ro(primary.parent / Path(name).name)
        sql = (f"SELECT {kq.format('k_wavenumber_exit_serial')}, {kq.format('q_wavenumber_exit_serial')}, "
               f"{kq.format('r_wavenumber_exit_serial')}, (SELECT z FROM redshift WHERE serial = q.z_response_serial), "
               "q.serial, q.k_wavenumber_exit_serial, q.total, q.total_abserr, q.metadata, "
               "(SELECT log10_tol FROM tolerance WHERE serial = q.rtol_serial) FROM QuadSourceIntegral q")
        for row in con.execute(sql):
            key = tuple(float(x) for x in row[:4])
            assert key not in out
            out[key] = {"shard": shard, "serial": row[4], "k_exit_serial": row[5], "total": row[6],
                        "total_abserr": row[7], "metadata": row[8], "log10_rtol": row[9]}
        con.close()
    return out


def main():
    csv_p = REH / "qsi_rows.csv"
    det_p = REH / "v1_violations_detail.txt"
    print(f"# v1_rows.py sha256 {sha(__file__)}")
    for p in (csv_p, det_p, *STORES.values()):
        print(f"# input {p} sha256 {sha(p)}")
    keys = []
    with open(csv_p) as f:
        for r in csv.DictReader(f):
            if r["V1"] == "VIOLATES":
                keys.append((float(r["k"]), float(r["q"]), float(r["r"]), float(r["z_response"])))
    if len(keys) != 12:
        raise SystemExit(f"expected 12 violating rows, found {len(keys)}")
    det = det_p.read_text()
    data = {t: read(p) for t, p in STORES.items()}
    rows = []
    ok = True
    for i, key in enumerate(sorted(keys, key=lambda k: (k[0], k[3]))):
        e = {"wid": f"w{i:02d}", "k": key[0], "q": key[1], "r": key[2], "z_response": key[3]}
        for t, tag in (("p", "23"), ("g", "24")):
            rec = data[t][key]
            md = json.loads(rec["metadata"])
            sub = md["WKB_Levin"]["subintervals"][1]
            gs = [g for g in sub["groups"] if g["label"] == "G+q+r"]
            if len(gs) != 1:
                raise SystemExit(f"{e['wid']} {t}: {len(gs)} G+q+r groups in sub-interval 1")
            g = gs[0]
            v, a = float(g["value"]).hex(), float(g["abserr"]).hex()
            in_det = v in det and a in det
            ok &= in_det
            e[f"R{tag}"] = {"shard": rec["shard"], "serial": rec["serial"], "k_exit_serial": rec["k_exit_serial"],
                            "log10_rtol": rec["log10_rtol"], "total": float(rec["total"]).hex(),
                            "total_abserr": float(rec["total_abserr"]).hex(),
                            "sub1": {k_: sub.get(k_) for k_ in ("regime", "z_max", "z_min")},
                            "group": {"label": g["label"], "value": v, "abserr": a, "converged": g.get("converged"),
                                      "phase_limited": g.get("phase_limited"), "regions": g.get("regions")},
                            "in_v1_violations_detail": in_det}
        rows.append(e)
        print(f"{e['wid']} k={key[0]:.6g} q={key[1]:.6g} r={key[2]:.6g} z={key[3]!r}: "
              f"R23 {e['R23']['shard']}-{e['R23']['serial']} v23 {e['R23']['group']['value']} e23 {e['R23']['group']['abserr']} | "
              f"R24 {e['R24']['shard']}-{e['R24']['serial']} v24 {e['R24']['group']['value']} e24 {e['R24']['group']['abserr']} | "
              f"in detail.txt {e['R23']['in_v1_violations_detail']}/{e['R24']['in_v1_violations_detail']}")
    out = HERE.parent / "out" / "v1_rows.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out} sha256 {sha(out)}; every v/e found in v1_violations_detail.txt: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
