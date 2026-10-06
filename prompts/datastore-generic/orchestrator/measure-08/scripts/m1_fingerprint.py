"""M1: the fingerprint of build_full_store and build_real_store, and the whole inventory, in a tree.

usage: python m1_fingerprint.py <tree> <label> <scratch-dir>
Writes <scratch>/m1/{full,real}-fingerprint-<label>.json and {full,real}-inventory-<label>.json.
The inventory dump holds, per class in build order: replicated, tagged, count, problems, records,
parents, parent_sets, members, and (where the tree has it) parent_types. For comparison across the
two trees, `parents` is also given normalised: a polymorphic field's value is written as
"<polymorphic: type_column -> {type: class}>" in both trees (the old tree's "cosmology" marker
and cosmology_classes(), the new tree's None and parent_types).
"""
import dataclasses, json, shutil, sys
from pathlib import Path

tree, label, scratch = sys.argv[1], sys.argv[2], Path(sys.argv[3]) / "m1"
sys.path.insert(0, tree)
scratch.mkdir(parents=True, exist_ok=True)

from Datastore.tests.real_store_fixtures import build_full_store, build_real_store
import Datastore.store_inventory as si
from Datastore.store_inventory import read_inventory
from RunRegistry.stores import fingerprint_of
from config.datastore import factories

print(label, "store_inventory file:", si.__file__)


def polymorphic(c):
    """field -> (type column, {type: class}) for each polymorphic field of class c, in either tree."""
    if hasattr(c, "parent_types"):
        return {f: (tc, {str(k): v for k, v in t.items()}) for f, (tc, t) in c.parent_types.items()}
    out = {}
    for f, of in c.parents.items():
        if of == getattr(si, "COSMOLOGY", object()):
            out[f] = ("cosmology_type", {str(k): v for k, v in si.cosmology_classes().items()})
    return out


for kind, build in (("full", build_full_store), ("real", build_real_store)):
    d = scratch / f"{kind}-{label}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir()
    store = build(d)
    inv = read_inventory(store.primary, factories)
    fp = fingerprint_of(inv, {"when": None, "git_head": None, "git_dirty": None, "run_id": None})
    fp.pop("taken")
    (scratch / f"{kind}-fingerprint-{label}.json").write_text(json.dumps(fp, sort_keys=True, indent=1))
    flat = {"class_order": list(inv.classes), "shards": list(inv.shards), "classes": {}}
    for name, c in inv.classes.items():
        poly = polymorphic(c)
        flat["classes"][name] = {
            "replicated": c.replicated,
            "tagged": c.tagged,
            "count": c.count,
            "problems": list(c.problems),
            "records": [r.as_json() for r in c.records],
            "earliest": str(c.earliest_timestamp),
            "latest": str(c.latest_timestamp),
            "parents_normalised": {
                f: (f"<polymorphic: {poly[f][0]} -> {poly[f][1]}>" if f in poly else of)
                for f, of in c.parents.items()
            },
            "parent_sets": {f: dict(m) for f, m in c.parent_sets.items()},
            "members": {f: {d: [list(m) for m in ms] for d, ms in sets.items()} for f, sets in c.members.items()},
        }
    (scratch / f"{kind}-inventory-{label}.json").write_text(json.dumps(flat, sort_keys=True, indent=1, default=str))
    raw_parents = {n: dict(c.parents) for n, c in inv.classes.items() if any(f in polymorphic(c) for f in c.parents)}
    print(label, kind, "digest:", fp["digest"], "problems:", sum(len(c.problems) for c in inv.classes.values()),
          "classes:", len(inv.classes), "raw parents of polymorphic classes:", json.dumps(raw_parents, default=str))
