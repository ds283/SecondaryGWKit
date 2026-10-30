"""Print driver_run.json's per-leaf table (read only)."""
import json, math, sys
d = json.load(open(sys.argv[1]))
for name, cd in d["cases"].items():
    L = cd["leaves"]
    print(f"\n## {name}  x_r={cd['summary']['x_r']!r}")
    print(" a          b          h        dep kind  floor     | terms                         | |roundoff| (node/endpt part) trunc     actual    declared  xr")
    for r in L:
        t = r["terms"]
        tt = " ".join(f"{v:.1e}" for v in t.values())
        part = r.get("roundoff_node_part", r.get("roundoff_endpoint_phase_part"))
        flag = "XR" if r["holds_x_r"] else ""
        if abs(r["roundoff"]) > r["floor_code"]: flag += " RO>FLOOR"
        if abs(r["actual_error"]) > r["declared_error"]: flag += " ACT>DECL"
        print(f"{r['a']:.8f} {r['b']:.8f} {r['h']:.2e} {r['depth']:3d} {r['kind']:5s} {r['floor_code']:.2e} | {tt} | {r['roundoff']:+.2e} ({part:+.1e}) {r['truncation']:+.1e} {r['actual_error']:+.1e} {r['declared_error']:.1e} {flag}")
    cc = [r for r in L if r["kind"] == "CC"]
    print("sum node part", math.fsum(r["roundoff_node_part"] for r in cc), "sum|node|", math.fsum(abs(r["roundoff_node_part"]) for r in cc))
    print("CC: sum roundoff", math.fsum(r["roundoff"] for r in cc), "sum|.|", math.fsum(abs(r["roundoff"]) for r in cc), "floor", math.fsum(r["floor_code"] for r in cc))
    lv = [r for r in L if r["kind"] == "Levin"]
    print("Levin: sum roundoff", math.fsum(r["roundoff"] for r in lv), "sum|.|", math.fsum(abs(r["roundoff"]) for r in lv), "floor", math.fsum(r["floor_code"] for r in lv))
