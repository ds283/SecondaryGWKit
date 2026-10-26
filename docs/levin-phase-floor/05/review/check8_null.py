import sqlite3
base = "var/datastores/handover-A3-v2-lambdacdm"
cand = []
for s in range(4):
    con = sqlite3.connect(f"file:{base}-shard{s:04d}.sqlite?mode=ro", uri=True)
    q = """select qsi.serial, qsi.compute_time, wk.k_inv_Mpc from QuadSourceIntegral qsi
    join tolerance ta on ta.serial=qsi.atol_serial join tolerance tr on tr.serial=qsi.rtol_serial
    join wavenumber_exit_time ek on ek.serial=qsi.k_wavenumber_exit_serial join wavenumber wk on wk.serial=ek.wavenumber_serial
    where ta.log10_tol=-32 and tr.log10_tol=-7 and qsi.WKB_Levin_max_depth is null and qsi.compute_time<=60"""
    for serial, ct, k in con.execute(q): cand.append((ct, (s, serial), k))
    con.close()
cand.sort()
ks, taken = set(), []
for ct, key, k in cand:
    if k in ks: continue
    ks.add(k); taken.append(key)
    if len(taken) == 2: break
print(len(cand), "NULL candidates; taken", taken)
