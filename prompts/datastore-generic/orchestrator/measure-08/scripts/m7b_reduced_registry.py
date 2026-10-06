"""M7b: read_inventory given a registry that lacks one class (and its association/value tables),
on the M2 full store, in a tree: which error, with what text, comes first.

usage: python m7b_reduced_registry.py <tree> <primary>
"""
import json, sys

sys.path.insert(0, sys.argv[1])
from config.datastore import factories
from Datastore.store_inventory import read_inventory

CASES = {
    "without OneLoopIntegral (a class nothing references)": ["OneLoopIntegral", "OneLoopIntegral_tags"],
    "without BackgroundModel (a class others reference)": ["BackgroundModel", "BackgroundModel_tags", "BackgroundModelValue"],
    "without store_tag (the tag table)": ["store_tag"],
}
for label, drop in CASES.items():
    reduced = {n: f for n, f in factories.items() if n not in drop}
    try:
        inv = read_inventory(sys.argv[2], reduced)
        print(json.dumps({"case": label, "type": None, "classes": len(inv.classes)}))
    except Exception as e:
        print(json.dumps({"case": label, "type": type(e).__name__, "message": str(e)[:600]}))
