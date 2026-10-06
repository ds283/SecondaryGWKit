"""M4: the derived class order against the base tree's INVENTORY_CLASSES, and what a plain
dependency sort (no roots-first step) would give. Run in the new tree (PYTHONPATH=<new tree>).

usage: python m4_order.py <base tree>
"""
import ast, json, sys
from pathlib import Path

base = Path(sys.argv[1])
src = (base / "Datastore" / "store_inventory.py").read_text()
env = {}
for node in ast.parse(src).body:
    if isinstance(node, ast.AnnAssign) and node.target.id in ("COMPUTE_TARGETS", "INVENTORY_CLASSES"):
        env[node.target.id] = eval(compile(ast.Expression(node.value), "x", "eval"), {}, env)
old = list(env["INVENTORY_CLASSES"])

from config.datastore import factories
from Datastore.store_inventory import inventory_classes, inventory_specs

specs = inventory_specs(factories)
derived = inventory_classes(factories)
deps = {n: [d for d in s.dependencies() if d != n] for n, s in specs.items()}


def plain_sort():
    order, placed = [], set()
    while len(order) < len(specs):
        ready = [n for n in specs if n not in placed and all(d in placed for d in deps[n])]
        order.append(ready[0])
        placed.add(ready[0])
    return order


def plain_sort_without_tag_table():
    d2 = {n: [d for d in v if d != "store_tag"] for n, v in deps.items()}
    order, placed = [], set()
    while len(order) < len(specs):
        ready = [n for n in specs if n not in placed and all(d in placed for d in d2[n])]
        order.append(ready[0])
        placed.add(ready[0])
    return order


plain = plain_sort()
print("registry order of the spec'd classes:", list(specs))
print("dependencies:", json.dumps(deps))
print("old INVENTORY_CLASSES:          ", old)
print("derived (roots first):          ", derived)
print("derived == old:", derived == old)
print("plain dependency sort:          ", plain)
print("plain == old:", plain == old)
print("plain sort, no tag-table dependency == plain:", plain_sort_without_tag_table() == plain)
print("first difference plain vs old:", next(((i, a, b) for i, (a, b) in enumerate(zip(plain, old)) if a != b), None))
roots = [n for n in specs if not deps[n]]
print("roots:", roots)
# roots-first without the tag-table dependency: a tagged class with no declared parent would be a root
roots_no_tag = [n for n in specs if not [d for d in deps[n] if d != "store_tag"]]
print("roots if the tag table were not a dependency:", roots_no_tag, "same:", roots_no_tag == roots)
# every class after its dependencies, in both orders
for label, order in (("derived", derived), ("plain", plain), ("old", old)):
    pos = {n: i for i, n in enumerate(order)}
    print(label, "every class after its dependencies:", all(pos[d] < pos[n] for n in order for d in deps[n]))
