import csv, sqlite3, sys
csv.field_size_limit(10**9)
D = "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/"
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["table"] == "GkWKBValue" and r["mod_A"] != r["mod_B"]]
r = max(rows, key=lambda r: float(r["theta_abs"]))
print("chosen: k", r["k"], "z", r["z"], "|theta|", r["theta_abs"], "div", r["div_A"], r["div_B"])
print("tabulated mod_A", r["mod_A"], "mod_B", r["mod_B"], "residual_B", r["residual_B"], "G_A", r["G_WKB_A"], "G_B", r["G_WKB_B"])
for tag, mod, div in (("r13", r["mod_A"], r["div_A"]), ("r14", r["mod_B"], r["div_B"])):
    m = float.fromhex(mod)
    hits = []
    for s in range(4):
        c = sqlite3.connect(f"file:{D}levin-phase-floor-06-{tag}-shard{s:04d}.sqlite?mode=ro", uri=True)
        cols = "serial, theta_div_2pi, theta_mod_2pi, G_WKB" + (", theta_residual" if tag == "r14" else "")
        for h in c.execute(f"select {cols} from GkWKBValue where theta_div_2pi = ? and theta_mod_2pi = ?", (int(div), m)):
            hits.append((s,) + tuple(h))
        c.close()
    print(tag, "matches", len(hits))
    for h in hits:
        out = [h[0], h[1], h[2], float.hex(h[3])] + [float(h[4]).__repr__()] + ([float.hex(h[5]), repr(h[5])] if tag == "r14" else [])
        print("  shard serial div mod G_WKB [residual]:", out)
mA, mB = float.fromhex(r["mod_A"]), float.fromhex(r["mod_B"])
eps = 2.220446049250313e-16
print("|dmod| =", abs(mA - mB), " eps|theta| =", eps * float(r["theta_abs"]), " ratio", abs(mA - mB) / (eps * float(r["theta_abs"])), " tabulated", r["dmod_over_eps_theta"])
