"""Review check 6 (scratch): recompute R1's QuadSource at p with the agent's own rebuild_row, and
compare its node values in float.hex with A3 v2's stored QuadSourceValue rows, read by sqlite3
mode=ro independently of the harness."""
import sqlite3, sys, json
from pathlib import Path
SCRIPTS = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-05/wider/scripts")
sys.path.insert(0, str(SCRIPTS))
import ComputeTargets
print("ComputeTargets from", ComputeTargets.__file__)
import rebuild
from harness import close_pool, open_readonly_pool, standin_ray
captured = {}
orig = rebuild.recompute_QuadSource
def wrap(ctx, model_proxy, q, r, Tq_num, Tr_num, Tk_tags, stored_qs):
    rb, chk = orig(ctx, model_proxy, q, r, Tq_num, Tr_num, Tk_tags, stored_qs)
    captured["values"] = [(v.z.store_id, v.z.z, {c: getattr(v, c) for c in rebuild.QS_VALUE_COLUMNS}) for v in rb.values]
    captured["chk"] = chk
    return rb, chk
rebuild.recompute_QuadSource = wrap
SHARD, SERIAL = 2, 37527
with standin_ray() as cluster:
    pool, _ = open_readonly_pool(rebuild.PRIMARY, cluster)
    try:
        ctx = rebuild.build_context(pool, log=lambda s: None)
        rebuild.rebuild_row(ctx, pool, SHARD, SERIAL, log=lambda s: None, rtol=None)
    finally:
        close_pool(pool, cluster)
chk = captured["chk"]
qs_serial = chk["stored_serial"]
print("harness qs_check:", json.dumps({k: chk[k] for k in ("stored_serial","n_stored","n_recomputed","n_common_differing_columns","added","L0a_holds")}))
# independent read of the stored QuadSource row's values
base = "/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores/handover-A3-v2-lambdacdm"
stored = {}
src_shard = None
for s in range(4):
    con = sqlite3.connect(f"file:{base}-shard{s:04d}.sqlite?mode=ro", uri=True)
    cols = [r[1] for r in con.execute('PRAGMA table_info("QuadSourceValue")')]
    head = con.execute('SELECT "Tq_serial", "Tr_serial" FROM "QuadSource" WHERE serial = ?', (qs_serial,)).fetchone()
    print("shard", s, "QuadSource serial", qs_serial, "Tq/Tr", head, "want", chk["Tq_Tr_serials"][2:4])
    if head is None or list(head) != list(chk["Tq_Tr_serials"][2:4]):
        con.close(); continue
    rows = con.execute('SELECT * FROM "QuadSourceValue" WHERE parent_serial = ?', (qs_serial,)).fetchall()
    if rows:
        assert src_shard is None, "two shards match"

        src_shard = s
        for row in rows:
            d = dict(zip(cols, row))
            stored[d["z_serial"]] = d
    con.close()
print("stored QuadSourceValue rows for QuadSource serial", qs_serial, "on shard", src_shard, ":", len(stored), "columns", cols)
rec = {zid: vals for zid, z, vals in captured["values"]}
common = sorted(set(stored) & set(rec)); only_rec = sorted(set(rec) - set(stored)); only_st = sorted(set(stored) - set(rec))
ndiff = 0; ncmp = 0
for zid in common:
    for c in rebuild.QS_VALUE_COLUMNS:
        a, b = rec[zid][c], stored[zid][c]
        ha = None if a is None else float(a).hex(); hb = None if b is None else float(b).hex()
        ncmp += 1
        if ha != hb: ndiff += 1; print("DIFF", zid, c, ha, hb)
print(f"common nodes {len(common)}; (node,column) pairs compared {ncmp}; differing in float.hex {ndiff}; recomputed-only z serials {only_rec}; stored-only {only_st}")
z0 = common[0]
print("example node", z0, {c: float(rec[z0][c]).hex() for c in ("source","undiff","diff")}, {c: float(stored[z0][c]).hex() for c in ("source","undiff","diff")})
