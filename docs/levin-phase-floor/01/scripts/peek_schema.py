"""Read-only peek at A3 v2's QuadSourceIntegral schema and one row's metadata keys (prompt 01 prep)."""
import json, sqlite3, sys
d = "var/datastores/"
p = sqlite3.connect(f"file:{d}handover-A3-v2-lambdacdm.sqlite?mode=ro", uri=True)
print("primary tables:", [t for (t,) in p.execute("select name from sqlite_master where type='table'")])
print("shards:", list(p.execute("select * from shards")))
c = sqlite3.connect(f"file:{d}handover-A3-v2-lambdacdm-shard0000.sqlite?mode=ro", uri=True)
print("QuadSourceIntegral columns:", [r[1] for r in c.execute("pragma table_info(QuadSourceIntegral)")])
print("tolerance:", list(c.execute("select * from tolerance")))
s, md = c.execute('select serial, metadata from QuadSourceIntegral where "WKB_Levin_max_depth"=20 limit 1').fetchone()
m = json.loads(md)
print("serial", s, "metadata keys", list(m))
w = m["WKB_Levin"]
print("WKB_Levin keys", list(w))
for sub in w["subintervals"]:
    print(" sub", {k: v for k, v in sub.items() if k != "groups"})
    for g in sub["groups"]:
        print("   ", g)
print("total_abserr typeof:", list(c.execute("select typeof(total_abserr), count(*) from QuadSourceIntegral group by 1")))
