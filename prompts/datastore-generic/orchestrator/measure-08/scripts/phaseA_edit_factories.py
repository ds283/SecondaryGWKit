"""Phase A: turn every factory's inventory_records(conn, table, tables, context) -- a single
read_records(conn, table, tables, context, **keywords) call -- into a static inventory_spec()
returning InventorySpec(**keywords). The polymorphic cosmology Parent is declared through
Datastore.SQL.ObjectFactories.cosmology_types.COSMOLOGY_TYPES.

usage (from the tree): python phaseA_edit_factories.py
"""
import re
from pathlib import Path

changed = 0
for path in sorted(Path("Datastore/SQL/ObjectFactories").glob("*.py")):
    s = path.read_text()
    if "def inventory_records(" not in s:
        continue
    n = s.count("def inventory_records(conn, table, tables, context):")
    s = s.replace(
        "def inventory_records(conn, table, tables, context):", "def inventory_spec():"
    )
    # the import inside the method
    s = re.sub(
        r"from Datastore\.store_inventory import ([^\n]*)read_records",
        lambda m: "from Datastore.store_inventory import "
        + m.group(1)
        + "InventorySpec",
        s,
    )
    s = s.replace(
        "from Datastore.store_inventory import COSMOLOGY, Parent, InventorySpec",
        "from Datastore.SQL.ObjectFactories.cosmology_types import COSMOLOGY_TYPES\n"
        "        from Datastore.store_inventory import InventorySpec, Parent",
    )
    s = s.replace(
        "from Datastore.store_inventory import Parent, InventorySpec",
        "from Datastore.store_inventory import InventorySpec, Parent",
    )
    s = s.replace(
        "from Datastore.store_inventory import Parent, ParentSet, InventorySpec",
        "from Datastore.store_inventory import InventorySpec, Parent, ParentSet",
    )
    # the call
    s = re.sub(
        r"return read_records\(\s*conn,\s*table,\s*tables,\s*context,\s*",
        "return InventorySpec(",
        s,
    )
    s = s.replace(
        'Parent("cosmology_serial", COSMOLOGY, "cosmology_type")',
        'Parent(\n                    "cosmology_serial",\n                    type_column="cosmology_type",\n'
        "                    types=COSMOLOGY_TYPES,\n                )",
    )
    assert "read_records" not in s and "COSMOLOGY," not in s, path
    path.write_text(s)
    changed += n
    print(path, n)
print("methods changed:", changed)
