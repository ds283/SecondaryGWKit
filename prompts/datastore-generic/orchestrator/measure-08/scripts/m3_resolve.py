"""M3: StoreInventory.resolve(name, record) for every record of every class of build_full_store and
build_real_store, in a tree. usage: python m3_resolve.py <tree> <label> <scratch-dir>
Writes <scratch>/m3/{full,real}-resolve-<label>.json.
"""
import json, shutil, sys
from pathlib import Path

tree, label, scratch = sys.argv[1], sys.argv[2], Path(sys.argv[3]) / "m3"
sys.path.insert(0, tree)
scratch.mkdir(parents=True, exist_ok=True)

from Datastore.tests.real_store_fixtures import build_full_store, build_real_store
from Datastore.store_inventory import read_inventory
from config.datastore import factories

for kind, build in (("full", build_full_store), ("real", build_real_store)):
    d = scratch / f"{kind}-{label}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    inv = read_inventory(build(d).primary, factories)
    out = {
        name: [inv.resolve(name, r) for r in c.records] for name, c in inv.classes.items()
    }
    n = sum(len(v) for v in out.values())
    unresolved = json.dumps(out).count('"unresolved"')
    (scratch / f"{kind}-resolve-{label}.json").write_text(json.dumps(out, sort_keys=True, indent=1))
    print(label, kind, "records resolved:", n, "unresolved entries:", unresolved)
