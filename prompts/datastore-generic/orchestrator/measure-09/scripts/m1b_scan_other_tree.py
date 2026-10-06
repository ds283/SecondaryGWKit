"""M1b: the guard (from <guard-tree>, with its registry vocabulary) run over another tree's layer
files: counts per check and per file. Usage: PYTHONPATH=<guard-tree> python m1b_scan_other_tree.py <guard-tree> <target-tree>"""
import ast, collections, sys
from pathlib import Path
gtree, target = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(gtree))
import Datastore.tests.test_layer_is_generic as g
forbidden = g.forbidden_words()
g.REPO_ROOT = target
files = g.layer_files()
print("files", len(files))
for name in ("CODE", "STRING", "COMMENT", "IMPORT"):
    hits = []
    for rel in files:
        src = (target / rel).read_text(); tree = ast.parse(src)
        hits += {"CODE": lambda: g.scan_code_names(rel, tree, forbidden), "STRING": lambda: g.scan_strings(rel, tree, forbidden),
                 "COMMENT": lambda: g.scan_comments(rel, src, forbidden), "IMPORT": lambda: g.scan_imports(rel, tree)}[name]()
    per = collections.Counter(h.split(":")[0] for h in hits)
    print(name, len(hits), dict(per))
    if name == "IMPORT":
        for h in hits: print("  ", h)
