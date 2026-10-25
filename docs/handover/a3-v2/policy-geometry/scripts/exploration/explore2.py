import sqlite3, os, collections

d = "var/datastores/"


def ro(p):
    return sqlite3.connect(f"file:{os.path.abspath(p)}?mode=ro", uri=True)


c = ro(d + "handover-A3-v2-lambdacdm-shard0002.sqlite")
z = dict(c.execute("select serial,z from redshift"))
print(
    c.execute("select source, response, count(*) from redshift group by 1,2").fetchall()
)
ser = sorted(z, key=lambda s: z[s])
print("serial order == z order?", ser == sorted(z))
print("z range", z[ser[0]], z[ser[-1]])
r = c.execute(
    "select d.serial,d.type,d.quality,d.crossover_z,s.serial,s.z_response_serial,s.z_max_serial,s.z_samples,s.numeric_smallest_z_serial,s.primary_WKB_largest_z_serial from GkSourcePolicyData d join GkSource s on s.serial=d.source_serial where d.type=2 limit 5"
).fetchall()
for row in r:
    print(row)
    ps = row[4]
    vals = c.execute(
        "select z_source_serial,G is not null, theta_mod_2pi is not null from GkSourceValue where parent_serial=? ",
        (ps,),
    ).fetchall()
    vals.sort(key=lambda v: z[v[0]])
    zs = [z[v[0]] for v in vals]
    print(
        " zresp",
        z[row[5]],
        " nodes",
        len(vals),
        " zmin",
        zs[0],
        " zmax",
        zs[-1],
        " zmax_serial z",
        z[row[6]],
    )
    print(" numeric_smallest", z[row[8]], " WKB_largest", z[row[9]])
    # pattern
    pat = "".join(("N" if v[1] else ".") + ("W" if v[2] else ".") + " " for v in vals)
    import itertools

    runs = [
        (k, len(list(g)))
        for k, g in itertools.groupby([(bool(v[1]), bool(v[2])) for v in vals])
    ]
    print(" runs low->high (hasNum,hasWKB,len):", runs)
    xs = [x for x in zs if x == row[3]]
    print(" crossover in nodes:", bool(xs), "index", zs.index(row[3]) if xs else None)
