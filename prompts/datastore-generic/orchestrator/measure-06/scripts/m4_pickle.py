"""M4: cloudpickle the registries without ray.init; the factories by reference."""
import pickle, sys
sys.path.insert(0, sys.argv[1])
import ray
from ray import cloudpickle
import config.datastore as cd

for name in ("factories", "serial_batch_sizes", "drop_groups"):
    obj = getattr(cd, name)
    blob = cloudpickle.dumps(obj)
    back = pickle.loads(blob)
    same = back == obj
    if name == "factories":
        ident = all(back[k] is obj[k] for k in obj)
        by_ref = all(obj[k].__name__.encode() in blob for k in obj)
        print(f"{name}: {len(blob)} bytes, entries {len(obj)}, equal {same}, every value the same class object {ident}, every class named in the blob (by reference) {by_ref}")
    else:
        print(f"{name}: {len(blob)} bytes, entries {len(obj)}, equal {same}")
# a module-level `factories` pickled through a ShardedPool kwarg: the dict copy the pool holds
blob = cloudpickle.dumps(dict(cd.factories))
print("dict(factories):", len(blob), "bytes")
print("ray initialised:", ray.is_initialized())
