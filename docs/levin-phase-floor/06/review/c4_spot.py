import sqlite3, sys, glob, math
M = sys.argv[1]
def con(p): return sqlite3.connect(f"file:{p}?mode=ro", uri=True)
def shards(tag): return sorted(glob.glob(f"{M}/var/datastores/levin-phase-floor-06-{tag}-shard*.sqlite"))
modA = float.fromhex("-0x1.5b3fd23b4506ep+1"); modB = float.fromhex("-0x1.5b3fd23835f08p+1")
for tag, mod in (("r13", modA), ("r14", modB)):
    for p in shards(tag):
        c = con(p)
        cols = "serial, wkb_serial, z_serial, theta_div_2pi, theta_mod_2pi, G_WKB" + (", theta_residual" if tag == "r14" else "")
        for row in c.execute(f"select {cols} from GkWKBValue where theta_div_2pi = -7459548"):
            if row[4] == mod:
                print(tag, p.rsplit('/',1)[1], 'serial', row[0], 'wkb', row[1], 'z', row[2], 'div', row[3], 'mod', float.hex(row[4]), 'G_WKB', float.hex(row[5]), *(('residual', float.hex(row[6]), row[6]) if tag=='r14' else ()))
        c.close()
# residual range over all of R14's three tables
lo, hi = -math.pi - 2e-3, math.pi + 2e-3
for t in ("GkWKBValue", "TkWKBValue", "GkSourceValue"):
    n = nnull = bad = 0; mn = math.inf; mx = -math.inf
    for p in shards("r14"):
        c = con(p)
        for (r,) in c.execute(f"select theta_residual from {t}"):
            n += 1
            if r is None: nnull += 1; continue
            mn = min(mn, r); mx = max(mx, r)
            if not (lo < r < hi): bad += 1
        c.close()
    print(t, 'rows', n, 'null', nnull, 'min', mn, 'max', mx, 'outside', bad)
