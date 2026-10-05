"""Phase B: add the registry as the second argument of every read_inventory/open_read_only call
with one argument, and import it as _factories where the module does not bind that name."""
import ast, sys

for path in sys.argv[1:]:
    src = open(path).read()
    tree = ast.parse(src)
    lines = src.split("\n")
    inserts = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("read_inventory", "open_read_only")
            and len(node.args) == 1
            and not node.keywords
        ):
            a = node.args[0]
            inserts.append((a.end_lineno, a.end_col_offset))
    for ln, col in sorted(inserts, reverse=True):
        l = lines[ln - 1]
        lines[ln - 1] = l[:col] + ", _factories" + l[col:]
    out = "\n".join(lines)
    bound = any(
        isinstance(n, ast.ImportFrom)
        and any((al.asname or al.name) == "_factories" for al in n.names)
        and n.col_offset == 0
        for n in tree.body
    )
    if inserts and not bound:
        # after the last module-level from-import
        last = max(n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
        ls = out.split("\n")
        ls.insert(last, "from config.datastore import factories as _factories")
        out = "\n".join(ls)
    open(path, "w").write(out)
    print(path, len(inserts), "calls", "import added" if inserts and not bound else "")
