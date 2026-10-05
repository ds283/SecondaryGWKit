"""M1: the fingerprint of build_full_store, and build_real_store's inventory, in a given tree.

usage: python m1_fingerprint.py <tree> <label> <scratch-dir>
Writes <scratch>/m1/full-fingerprint-<label>.json and real-inventory-<label>.json.
"""
import dataclasses, inspect, json, shutil, sys
from pathlib import Path

tree, label, scratch = sys.argv[1], sys.argv[2], Path(sys.argv[3]) / "m1"
sys.path.insert(0, tree)
scratch.mkdir(parents=True, exist_ok=True)

from Datastore.tests.real_store_fixtures import build_full_store, build_real_store
from Datastore.store_inventory import read_inventory
from RunRegistry.stores import fingerprint_of

takes_registry = len(inspect.signature(read_inventory).parameters) == 2
if takes_registry:
    from config.datastore import factories

    def take_inventory(primary):
        return read_inventory(primary, factories)
else:
    def take_inventory(primary):
        return read_inventory(primary)

print(label, "read_inventory takes the registry:", takes_registry)

for kind, build in (("full", build_full_store), ("real", build_real_store)):
    d = scratch / f"{kind}-{label}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    store = build(d)
    inv = take_inventory(store.primary)
    fp = fingerprint_of(inv, {"when": None, "git_head": None, "git_dirty": None, "run_id": None})
    fp.pop("taken")
    (scratch / f"{kind}-fingerprint-{label}.json").write_text(json.dumps(fp, sort_keys=True, indent=1))
    # the inventory itself: problems and records of every class, paths made relative
    flat = {
        name: {
            "count": c.count,
            "problems": list(c.problems),
            "records": [json.loads(json.dumps(dataclasses.asdict(r), default=str)) for r in c.records],
            "replicated": c.replicated,
        }
        for name, c in inv.classes.items()
    }
    (scratch / f"{kind}-inventory-{label}.json").write_text(json.dumps(flat, sort_keys=True, indent=1, default=str))
    problems = {n: v["problems"] for n, v in flat.items() if v["problems"]}
    print(label, kind, "digest:", fp["digest"], "problems:", problems or "{}",
          "replicated classes:", sorted(n for n, v in flat.items() if v["replicated"]))
