"""M1: run the guard's scans on a tree and print every hit, by check.
Usage: PYTHONPATH=<tree> python m1_scan.py <tree>"""
import sys, time
from pathlib import Path
tree = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tree))
import Datastore.tests.test_layer_is_generic as g
assert g.REPO_ROOT == tree, g.REPO_ROOT
t0 = time.time()
files = g.layer_files()
forbidden = g.forbidden_words()
tables, columns = g.registry_words()
print("FILES", len(files))
for f in files: print("  ", f)
print("MISSING_FROM_AUDIT_LIST", sorted(g.AUDIT_LIST - set(files)))
print("EXTRA_BEYOND_AUDIT_LIST", sorted(set(files) - g.AUDIT_LIST))
print("VOCAB tables+classes", len(tables), "columns", len(columns), "packages", sorted(g.project_packages()), "total", len(forbidden))
import ast
for name, fn in (("CODE", lambda r, s, t: g.scan_code_names(r, t, forbidden)),
                 ("STRING", lambda r, s, t: g.scan_strings(r, t, forbidden)),
                 ("COMMENT", lambda r, s, t: g.scan_comments(r, s, forbidden)),
                 ("IMPORT", lambda r, s, t: g.scan_imports(r, t))):
    hits = []
    for rel in files:
        src = (tree / rel).read_text()
        hits += fn(rel, src, ast.parse(src))
    print(f"{name} {len(hits)}")
    for h in hits: print("  ", h)
print("elapsed %.2fs" % (time.time() - t0))
