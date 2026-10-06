"""M4: build build_full_store and build_real_store with <tree>'s own code and fixtures, into
<outdir>/full and <outdir>/real (removed first). Usage: python m4_build.py <tree> <outdir>"""
import shutil, sys
from pathlib import Path
tree, out = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, tree)
from Datastore.tests.real_store_fixtures import build_full_store, build_real_store
import Datastore
assert Datastore.__file__.startswith(str(Path(tree).resolve())) or Datastore.__file__.startswith(tree), Datastore.__file__
for kind, build in (("full", build_full_store), ("real", build_real_store)):
    d = out / kind
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    store = build(d)
    print(kind, store.primary, sorted(p.name for p in store.shard_files.values()))
