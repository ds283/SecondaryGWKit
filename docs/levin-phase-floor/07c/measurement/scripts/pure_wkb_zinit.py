"""levin-phase-floor prompt 06 section 6: how A3 v2 records a pure-WKB GkWKBIntegration's z_init
(read-only, sqlite3 mode=ro, every shard). Prompt 06 section 6 tags a G window pure-WKB "if every
one of its 9 source samples comes from a pure-WKB object (GkSource's z_init null)"; this counts,
per shard, the GkWKBIntegration rows with no numeric parent and how their z_init relates to their
z_source, the seeded rows likewise, and the GkSourceValue rows with WKB data whose z_init is null."""
import sqlite3
from pathlib import Path

D = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores")
for i in range(4):
    c = sqlite3.connect(f"{(D / f'handover-A3-v2-lambdacdm-shard{i:04d}.sqlite').as_uri()}?mode=ro", uri=True)
    q = 'select count(*), sum(w.z_init is null), sum(w.z_init = r.z), sum(w.z_init < r.z) from "GkWKBIntegration" w join redshift r on r.serial = w.z_source_serial where w.numeric_serial is {} null'
    pure = c.execute(q.format("")).fetchone()
    seeded = c.execute(q.format("not")).fetchone()
    gsv = c.execute('select count(*), sum(z_init is null) from "GkSourceValue" where theta_mod_2pi is not null').fetchone()
    print(f"shard {i}: pure-WKB rows (numeric_serial NULL): n={pure[0]}, z_init NULL {pure[1]}, z_init == z_source {pure[2]}, z_init < z_source {pure[3]}")
    print(f"shard {i}: seeded rows: n={seeded[0]}, z_init NULL {seeded[1]}, z_init == z_source {seeded[2]}, z_init < z_source {seeded[3]}")
    print(f"shard {i}: GkSourceValue rows with WKB data: n={gsv[0]}, z_init NULL {gsv[1]}")
    c.close()
