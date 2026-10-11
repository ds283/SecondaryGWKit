"""
Count the tests `unittest` loads from Datastore/tests, by module, and split them between the 26
modules DatastoreKit's SGK checklist (item 7) says the package now carries and the 18 that stay.

Run from the repository root, at the commit to be measured:

    PYTHONPATH=. ./venv/bin/python prompts/dk-and-rpk-migration/measure/scripts/count_datastore_tests.py

It loads the tests and runs none of them. Its output at b510bc9 is
measure/out/b510bc9-datastore-tests-by-module.txt.
"""

import unittest

# DatastoreKit docs/adoption/secondarygwkit.md item 7: the 8 moved verbatim, then the 18 ported
DELETED = """
test_shard_paths test_shard_file_name test_shardedpool_shard_paths test_copy_move_store
test_delete_store test_sharded_store_script test_shard_key_audit_copy test_shard_key_audit_refusals
test_replicated_write test_reconcile_at_open test_prune_at_open test_version_row_at_open
test_read_only_pool test_one_timestamp_per_write test_absolute_shard_record_refused
test_closed_store_refusals test_store_inventory test_store_schema test_store_reader
test_foreign_key_check test_schema_builder test_inventory_declarations test_declared_facts
test_layer_registry test_drop_refuses_dangling_references test_layer_is_generic
""".split()
assert len(DELETED) == 26


def walk(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from walk(test)
        else:
            yield test


def main():
    suite = unittest.TestLoader().discover("Datastore/tests", top_level_dir=".")
    counts = {}
    for test in walk(suite):
        module = type(test).__module__.split(".")[-1]
        counts[module] = counts.get(module, 0) + 1
    total = sum(counts.values())
    deleted = sum(counts.get(m, 0) for m in DELETED)
    missing = [m for m in DELETED if m not in counts]
    print(f"Datastore/tests: {total} tests loaded in {len(counts)} modules")
    print(f"in the 26 modules the package carries: {deleted}; not found: {missing}")
    print(
        f"in the {len(counts) - len(DELETED) + len(missing)} staying modules: {total - deleted}"
    )
    for module in sorted(counts):
        label = "DELETE" if module in DELETED else "stay  "
        print(f"  {label} {module}: {counts[module]}")


if __name__ == "__main__":
    main()
