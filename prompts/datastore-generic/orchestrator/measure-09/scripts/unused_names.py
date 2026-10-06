"""List module-level imports and assignments in a file that nothing else in it uses.
Usage: python unused_names.py <file>..."""
import ast, sys
for path in sys.argv[1:]:
    tree = ast.parse(open(path).read())
    bound = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound[(a.asname or a.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    bound[t.id] = node.lineno
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bound[node.name] = node.lineno
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    for name, line in sorted(bound.items(), key=lambda x: x[1]):
        if name not in used and not name.startswith("Test") and name != "__name__":
            print(f"{path}:{line} {name}")
