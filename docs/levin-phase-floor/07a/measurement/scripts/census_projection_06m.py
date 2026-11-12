"""levin-phase-floor prompt 06m section 5: the census projection, from Gate M0's unit walls against route B's census
walls on the same 100 units. New in 06m (after 06l's census_projection_06l.py, a record). The ratio rho is
sum(wall_s at m on K0) / sum(wall_s of the same units in route B's census); the projection is route B's census wall
(18,340 s, manifest created to last heartbeat, 06l's results.md) times rho. 06g's census wall (20,361 s) is printed
beside it. Read only. Usage: PYTHONPATH=<tree> python -B census_projection_06m.py K0_UNITS_JSONL OUT_JSON"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import header, sha256  # noqa: E402

INDEX = HERE.parent / "state" / "index_routeB.json"
WALL_B, WALL_G = 18340.0, 20361.0


def main():
    src, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06m: the census projection", [src, INDEX])
    idx = json.loads(INDEX.read_text())["units"]
    m = {json.loads(l)["unit"]: json.loads(l)["data"]["wall_s"] for l in open(src)}
    by_rt = {}
    for u, w in m.items():
        rt = u.split("-")[1] + "-" + u.split("-")[2]
        a = by_rt.setdefault(rt, [0.0, 0.0])
        a[0] += w
        a[1] += idx[u]["wall_s"]
    sm, sb = sum(v[0] for v in by_rt.values()), sum(v[1] for v in by_rt.values())
    rho = sm / sb
    doc = {"header": hdr, "n_units": len(m), "sum_wall_m": sm, "sum_wall_routeB_census": sb, "rho": rho,
           "by_rtol": {k: {"m": v[0], "routeB": v[1], "rho": v[0] / v[1]} for k, v in by_rt.items()},
           "routeB_census_wall_s": WALL_B, "g06_census_wall_s": WALL_G, "projection_s": WALL_B * rho,
           "projection_from_06g_s": WALL_G * rho * (WALL_B / WALL_G)}
    out_p.write_text(json.dumps(doc, indent=1))
    print("\n".join(hdr))
    print(json.dumps({k: v for k, v in doc.items() if k != "header"}, indent=1))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")


if __name__ == "__main__":
    main()
