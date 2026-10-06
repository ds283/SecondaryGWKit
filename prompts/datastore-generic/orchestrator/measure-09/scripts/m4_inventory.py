"""M4(b): read the inventory of existing stores with <tree>'s code, and write its fingerprint
(`taken` removed) and the whole inventory (parents normalised for the polymorphic cosmology field,
as measure-08's m1_fingerprint.py does) to <outdir>/{full,real}-{fingerprint,inventory}-<label>.json.
At dbd8a51 read_inventory takes no registry; after 06 it takes config.datastore.factories.
Usage: python m4_inventory.py <tree> <stores-dir containing full/ and real/> <label> <outdir>"""
import inspect, json, sys
from pathlib import Path
tree, stores, label, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, tree)
out.mkdir(parents=True, exist_ok=True)
import Datastore.store_inventory as si
from RunRegistry.stores import fingerprint_of
print(label, "store_inventory:", si.__file__)
takes_registry = len(inspect.signature(si.read_inventory).parameters) > 1
if takes_registry:
    from config.datastore import factories

def polymorphic(c):
    if hasattr(c, "parent_types"):
        return {f: (tc, {str(k): v for k, v in t.items()}) for f, (tc, t) in c.parent_types.items()}
    res = {}
    for f, of in c.parents.items():
        if of == getattr(si, "COSMOLOGY", object()):
            res[f] = ("cosmology_type", {str(k): v for k, v in si.cosmology_classes().items()})
    return res

for kind, stem in (("full", "full-store"), ("real", "fixture-store")):
    primary = stores / kind / f"{stem}.sqlite"
    inv = si.read_inventory(primary, factories) if takes_registry else si.read_inventory(primary)
    fp = fingerprint_of(inv, {"when": None, "git_head": None, "git_dirty": None, "run_id": None})
    fp.pop("taken")
    (out / f"{kind}-fingerprint-{label}.json").write_text(json.dumps(fp, sort_keys=True, indent=1))
    flat = {"class_order": list(inv.classes), "shards": [str(s) for s in inv.shards], "classes": {}}
    for name, c in inv.classes.items():
        poly = polymorphic(c)
        flat["classes"][name] = {
            "replicated": c.replicated, "tagged": c.tagged, "count": c.count,
            "problems": list(c.problems), "records": [r.as_json() for r in c.records],
            "earliest": str(c.earliest_timestamp), "latest": str(c.latest_timestamp),
            "parents_normalised": {f: (f"<polymorphic: {poly[f][0]} -> {poly[f][1]}>" if f in poly else of)
                                   for f, of in c.parents.items()},
            "parent_sets": {f: dict(m) for f, m in c.parent_sets.items()},
            "members": {f: {d: [list(m) for m in ms] for d, ms in sets.items()} for f, sets in c.members.items()},
        }
    (out / f"{kind}-inventory-{label}.json").write_text(json.dumps(flat, sort_keys=True, indent=1, default=str))
    print(label, kind, "digest", fp["digest"], "classes", len(inv.classes),
          "problems", sum(len(c.problems) for c in inv.classes.values()),
          "records", sum(len(c.records) for c in inv.classes.values()))
