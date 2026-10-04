"""Measured facts 3-4: fingerprint of build_full_store, and of a store missing a table, in a given tree.

usage: python m4_fingerprint.py <tree> <label> <store-dir>
"""

import json
import sys
import traceback
from pathlib import Path

tree, label, store_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, tree)

from Datastore.tests.real_store_fixtures import build_full_store, build_real_store
from Datastore.store_inventory import read_inventory
from RunRegistry.stores import fingerprint_of

out = {}
full_dir = store_dir / f"full-{label}"
full_dir.mkdir(parents=True)
full = build_full_store(full_dir)
inv = read_inventory(full.primary)
fp = fingerprint_of(
    inv, {"when": None, "git_head": None, "git_dirty": None, "run_id": None}
)
fp.pop("taken")
out["full_store_fingerprint"] = fp
(store_dir / f"full-fingerprint-{label}.json").write_text(
    json.dumps(fp, sort_keys=True, indent=1)
)
print(label, "full store digest:", fp["digest"], "problems:", fp["problems"])

# a store missing a table, for the CLI (built here; the CLI is run separately)
miss_dir = store_dir / f"missing-{label}"
miss_dir.mkdir(parents=True)
miss = build_real_store(miss_dir, missing_tables={1: ["OneLoopIntegral_tags"]})
print(label, "missing-table store:", miss.primary)
try:
    inv = read_inventory(miss.primary)
    fp = fingerprint_of(
        inv, {"when": None, "git_head": None, "git_dirty": None, "run_id": None}
    )
    print(
        label,
        "missing-table store: library fingerprint digest",
        fp["digest"],
        "problems",
        fp["problems"],
    )
except Exception as e:
    print(label, "missing-table store: library raised", type(e).__name__, ":", e)
