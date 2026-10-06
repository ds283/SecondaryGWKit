"""List test ids (module.Class.method) in Datastore/tests at a git rev and in the working tree,
via the unittest loader on the working tree and the rev's files exported to a temp dir.
Usage: python m2_test_ids.py <worktree> <rev>"""
import subprocess, sys, tempfile, unittest, os, json
tree, rev = sys.argv[1], sys.argv[2]
def ids(root):
    code = ("import unittest,sys,json\n"
            "s=unittest.defaultTestLoader.discover('Datastore/tests',top_level_dir='.')\n"
            "def walk(x):\n"
            "  for t in x:\n"
            "    yield from (walk(t) if isinstance(t,unittest.TestSuite) else [t.id()])\n"
            "print(json.dumps(sorted(walk(s))))\n")
    out = subprocess.run([os.path.join(tree, "venv/bin/python"), "-c", code], cwd=root,
                         env=dict(os.environ, PYTHONPATH=root), capture_output=True, text=True)
    return set(json.loads(out.stdout.strip().splitlines()[-1]))
with tempfile.TemporaryDirectory(dir=os.path.dirname(tree.rstrip("/"))) as d:
    subprocess.run(f"git -C {tree} archive {rev} | tar -x -C {d}", shell=True, check=True)
    before = ids(d)
after = ids(tree)
print("at", rev, len(before), "rough", len(after))
for t in sorted(before - after): print("-", t)
for t in sorted(after - before): print("+", t)
