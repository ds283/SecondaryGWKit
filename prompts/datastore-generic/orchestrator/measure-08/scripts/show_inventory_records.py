"""Print every factory's inventory_records method source (with the decorator line) and its file's
imports from Datastore.store_inventory. A measurement record of datastore-generic prompt 08."""
import ast, sys, pathlib
root = pathlib.Path(sys.argv[1])
for path in sorted((root / "Datastore/SQL/ObjectFactories").glob("*.py")):
    src = path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and "store_inventory" in ast.unparse(node):
            print(f"{path.name}:{node.lineno}: IMPORT {ast.unparse(node)}")
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for f in cls.body:
            if isinstance(f, ast.FunctionDef) and f.name == "inventory_records":
                start = min([d.lineno for d in f.decorator_list] + [f.lineno])
                print(f"=== {path.name} class {cls.name} lines {start}-{f.end_lineno}")
                print("\n".join(lines[start - 1 : f.end_lineno]))
