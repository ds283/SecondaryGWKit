import sqlite3, os

d = "var/datastores/"


def ro(p):
    return sqlite3.connect(f"file:{os.path.abspath(p)}?mode=ro", uri=True)


p = ro(d + "handover-A3-v2-lambdacdm.sqlite")
print(p.execute("select * from replicated_tables").fetchall())
print(p.execute("select * from sharded_tables").fetchall()[:30])
for i in range(4):
    c = ro(d + f"handover-A3-v2-lambdacdm-shard000{i}.sqlite")
    print(
        i,
        [
            (t, c.execute(f'select count(*) from "{t}"').fetchone()[0])
            for t in (
                "redshift",
                "wavenumber_exit_time",
                "tolerance",
                "GkSource",
                "GkSourceValue",
                "GkSourcePolicyData",
                "GkSourcePolicy",
                "BackgroundModel",
            )
        ],
    )
    print(
        c.execute(
            "select z_samples, count(*) from GkSource group by z_samples"
        ).fetchall()
    )
    print(
        c.execute(
            "select wavenumber_exit_serial,count(*) from GkSource group by 1"
        ).fetchall()
    )
    print(
        c.execute(
            "select count(*), count(distinct parent_serial) from GkSourceValue"
        ).fetchall()
    )
    print(c.execute("select * from GkSourcePolicy").fetchall())
