"""Phase B: the minimal edits to existing tests (and two comments) for the names prompt 08 removes.

usage (from the tree): python phaseB_edit_tests.py
"""
from pathlib import Path


def edit(path, pairs):
    p = Path(path)
    s = p.read_text()
    for old, new in pairs:
        if new in s:
            continue  # already applied: the script is re-runnable
        assert s.count(old) == 1, (path, old, s.count(old))
        s = s.replace(old, new)
    p.write_text(s)
    print("edited", path, len(pairs))


# 1. test_inventory_report: the class list is derived from the registry
edit(
    "Datastore/tests/test_inventory_report.py",
    [
        (
            "from Datastore.store_inventory import INVENTORY_CLASSES, read_inventory\n",
            "from Datastore.store_inventory import inventory_classes, read_inventory\n",
        ),
        (
            "from config.datastore import factories as _factories\n",
            "from config.datastore import factories as _factories\n\n"
            "# the classes of the inventory, as read_inventory derives them from the registry\n"
            "INVENTORY_CLASSES = tuple(inventory_classes(_factories))\n",
        ),
    ],
)

# 2. test_store_inventory: the same, and the ast scan looks for inventory_spec, outside base.py
edit(
    "Datastore/tests/test_store_inventory.py",
    [
        (
            "from Datastore.store_inventory import (\n    INVENTORY_CLASSES,\n    canonical,\n",
            "from Datastore.store_inventory import (\n    canonical,\n",
        ),
        (
            "    read_inventory,\n)\n",
            "    inventory_classes,\n    read_inventory,\n)\n",
        ),
        (
            "\nfrom config.datastore import factories as _factories\n\nREPO_ROOT",
            "\nfrom config.datastore import factories as _factories\n\n"
            "# the classes of the inventory, as read_inventory derives them from the registry\n"
            "INVENTORY_CLASSES = tuple(inventory_classes(_factories))\n\nREPO_ROOT",
        ),
        (
            "``format`` or ``repr``, and no factory's ``inventory_records`` formats anything.",
            "``format`` or ``repr``, and no factory's ``inventory_spec`` formats anything.",
        ),
        (
            "        for path in sorted(FACTORIES.glob(\"*.py\")):\n"
            "            for func in ast.walk(ast.parse(path.read_text())):\n"
            "                if (\n"
            "                    isinstance(func, ast.FunctionDef)\n"
            "                    and func.name == \"inventory_records\"\n"
            "                ):",
            "        for path in sorted(FACTORIES.glob(\"*.py\")):\n"
            "            # base.py's hook declares no class: it returns None\n"
            "            if path.name == \"base.py\":\n"
            "                continue\n"
            "            for func in ast.walk(ast.parse(path.read_text())):\n"
            "                if (\n"
            "                    isinstance(func, ast.FunctionDef)\n"
            "                    and func.name == \"inventory_spec\"\n"
            "                ):",
        ),
    ],
)

# 3. test_wkb_numeric_parent_key: read the declaration instead of capturing a read_records call
edit(
    "Datastore/tests/test_wkb_numeric_parent_key.py",
    [
        (
            """            with self.subTest(factory=factory.__name__):
                captured = {}

                def fake_read_records(conn, table, tables, context, **kwargs):
                    captured.update(kwargs)

                with mock.patch(
                    "Datastore.store_inventory.read_records", fake_read_records
                ):
                    factory.inventory_records(None, None, None, None)
                self.assertNotIn("z_init", captured["leaves"])
                self.assertEqual(
                    captured["parents"]["numeric"].column, "numeric_serial"
                )""",
            """            with self.subTest(factory=factory.__name__):
                captured = factory.inventory_spec().keywords()
                self.assertNotIn("z_init", captured["leaves"])
                self.assertEqual(
                    captured["parents"]["numeric"].column, "numeric_serial"
                )""",
        ),
    ],
)

# 4. test_store_fingerprint: the compute targets whose label it changes, named in the test
edit(
    "RunRegistry/tests/test_store_fingerprint.py",
    [
        (
            "from Datastore.store_inventory import COMPUTE_TARGETS, read_inventory\n",
            "from Datastore.store_inventory import read_inventory\n",
        ),
        (
            "from config.datastore import factories as _factories\n",
            "from config.datastore import factories as _factories\n\n"
            "# the compute targets, whose labels are not identity\n"
            "COMPUTE_TARGETS = (\n"
            '    "TkNumericIntegration",\n'
            '    "TkWKBIntegration",\n'
            '    "GkNumericIntegration",\n'
            '    "GkWKBIntegration",\n'
            '    "GkSource",\n'
            '    "GkSourcePolicyData",\n'
            '    "QuadSource",\n'
            '    "QuadSourceIntegral",\n'
            '    "OneLoopIntegral",\n'
            ")\n",
        ),
    ],
)

# 5. comments and docstrings that name the removed names
edit(
    "Datastore/tests/real_store_fixtures.py",
    [
        (
            "# inventory (Datastore/store_inventory.py INVENTORY_CLASSES), with tags",
            "# inventory (Datastore/store_inventory.py inventory_classes), with tags",
        )
    ],
)
edit(
    "ComputeTargets/tests/test_qcd_cosmology_inventory_record.py",
    [
        (
            "as its factory's ``inventory_records`` declares it",
            "as its factory's ``inventory_spec`` declares it",
        )
    ],
)
