import sqlite3, sys, glob
M = sys.argv[1]
want = {"r13": float.fromhex("-0x1.1f665c7aeff31p-59"), "r14": float.fromhex("-0x1.1f665c9de3ef8p-59")}
for tag, v in want.items():
    for p in sorted(glob.glob(f"{M}/var/datastores/levin-phase-floor-06-{tag}-shard*.sqlite")):
        c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        for r in c.execute("select serial, total, total_abserr, WKB_Levin, numeric_quad, analytic_rad, WKB_Levin_num_regions from QuadSourceIntegral where total = ?", (v,)):
            print(tag, p.rsplit('/',1)[1], 'serial', r[0], 'total', float.hex(r[1]), 'abserr', float.hex(r[2]), r[2], 'WKB_Levin', float.hex(r[3]), 'numeric_quad', float.hex(r[4]), 'analytic_rad', r[5] if r[5] is None else float.hex(r[5]), 'regions', r[6])
        c.close()
