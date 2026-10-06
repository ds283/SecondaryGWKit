import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from Datastore.tests.real_store_fixtures import build_full_store, build_real_store
from Datastore.store_inventory import read_inventory
from config.datastore import factories

out = {}
for kind, b in (("full", build_full_store), ("real", build_real_store)):
    with tempfile.TemporaryDirectory() as t:
        inv = read_inventory(b(Path(t)).primary, factories)
        out[kind] = {
            n: [inv.resolve(n, r) for r in c.records] for n, c in inv.classes.items()
        }
Path(sys.argv[2]).write_text(json.dumps(out, sort_keys=True, default=str))
