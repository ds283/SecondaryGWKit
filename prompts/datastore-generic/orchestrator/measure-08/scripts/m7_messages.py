"""M7: every message whose text prompt 08 touches or adds, produced in a tree.

usage: python m7_messages.py <tree>    (prints one JSON object per line: {case, type, message})
Cases present in both trees: the read_records dependency-order RuntimeError, for a plain parent, a
polymorphic parent (the base tree's COSMOLOGY marker; the new tree's declared type map, the same
map) and a parent-set member. Cases only in the new tree: the new ValueErrors.
"""
import json, sys

sys.path.insert(0, sys.argv[1])
import sqlalchemy as sqla
import Datastore.store_inventory as si
from Datastore.store_inventory import Parent, ParentSet, ShardContext, read_records

NEW = hasattr(si, "inventory_classes")
md = sqla.MetaData()
child = sqla.Table("Child", md, sqla.Column("serial", sqla.Integer, primary_key=True),
                   sqla.Column("p_serial", sqla.Integer), sqla.Column("kind", sqla.Integer))
members = sqla.Table("Child_members", md, sqla.Column("owner", sqla.Integer), sqla.Column("m_serial", sqla.Integer))
tables = {"Child": child, "Child_members": members}
TYPES = {0: "LambdaCDM", 1: "QCD_Cosmology"}


def emit(case, fn):
    try:
        fn()
        print(json.dumps({"case": case, "type": None, "message": None}))
    except Exception as e:
        print(json.dumps({"case": case, "type": type(e).__name__, "message": str(e)}))


ctx = ShardContext(shard=3, keys={}, digests={"LambdaCDM": {}}, all_digests={})
emit("read_records: plain parent not built",
     lambda: read_records(None, child, tables, ctx, parents={"p": Parent("p_serial", "Parent")}))
emit("read_records: cross-shard parent not built",
     lambda: read_records(None, child, tables, ctx, parents={"p": Parent("p_serial", "Parent", cross_shard=True)}))
if NEW:
    poly = Parent("p_serial", type_column="kind", types=TYPES)
else:
    poly = Parent("p_serial", si.COSMOLOGY, "kind")
emit("read_records: polymorphic parent, second class not built",
     lambda: read_records(None, child, tables, ctx, leaves=("kind",), parents={"cosmology": poly}))
emit("read_records: parent-set member not built",
     lambda: read_records(None, child, tables, ctx, parent_sets={
         "s": ParentSet(table="Child_members", owner="owner", members={"m": Parent("m_serial", "Member")})}))

if NEW:
    from Datastore.store_inventory import InventorySpec, inventory_classes

    emit("Parent: neither of nor types", lambda: Parent("c"))
    emit("Parent: both of and types", lambda: Parent("c", "A", type_column="t", types={0: "A"}))
    emit("Parent: types without type_column", lambda: Parent("c", types={0: "A"}))
    emit("Parent: type_column without types", lambda: Parent("c", "A", type_column="t"))
    emit("Parent: empty types", lambda: Parent("c", type_column="t", types={}))
    emit("ParentSet: polymorphic member", lambda: ParentSet(
        table="T", owner="o", members={"a": Parent("a", "A"), "b": Parent("b", type_column="t", types={0: "A"})}))

    def registry(**specs):
        return {n: type(n, (), {"inventory_spec": staticmethod(lambda s=s: s)}) for n, s in specs.items()}

    emit("inventory_classes: a parent with no spec",
         lambda: inventory_classes(registry(B=InventorySpec(parents={"a": Parent("a_serial", "A")}))))
    emit("inventory_classes: two parents with no spec",
         lambda: inventory_classes(registry(B=InventorySpec(parents={"a": Parent("a_serial", "A"), "c": Parent("c_serial", "C")}))))
    emit("inventory_classes: a tagged class, no tag-table spec",
         lambda: inventory_classes(registry(B=InventorySpec(tags=("B_tags", "b_serial")))))
    emit("inventory_classes: polymorphic type column not a leaf",
         lambda: inventory_classes(registry(A=InventorySpec(), B=InventorySpec(parents={"a": Parent("a_serial", type_column="t", types={0: "A"})}))))
    emit("inventory_classes: self reference",
         lambda: inventory_classes(registry(A=InventorySpec(parents={"a": Parent("a_serial", "A")}))))
    emit("inventory_classes: cycle",
         lambda: inventory_classes(registry(R=InventorySpec(), A=InventorySpec(parents={"b": Parent("b_serial", "B")}),
                                            B=InventorySpec(parents={"a": Parent("a_serial", "A")}),
                                            C=InventorySpec(parents={"a": Parent("a_serial", "A")}))))
