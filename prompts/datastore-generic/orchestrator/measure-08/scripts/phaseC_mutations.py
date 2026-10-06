"""Phase C: deliberate breakage of the rough change, each mutation applied to Datastore/store_inventory.py
alone, the new test module run, and the file restored. usage (from the tree): python phaseC_mutations.py
"""
import subprocess, sys
from pathlib import Path

P = Path("Datastore/store_inventory.py")
original = P.read_text()
MUTATIONS = {
    "plain sort (no roots-first)": (
        "    order: List[str] = [name for name in specs if len(depends[name]) == 0]\n",
        "    order: List[str] = []\n",
    ),
    "resolve ignores the type map": (
        "        if field in self.parent_types:\n",
        "        if False:\n",
    ),
    "no cycle refusal (loop would spin: guarded by IndexError)": (
        "        if len(ready) == 0:\n",
        "        if False:\n",
    ),
    "polymorphic classes not dependencies": (
        "            named.extend(parent.classes())\n        for spec in self.parent_sets.values():",
        "            named.extend(parent.classes() if parent.types is None else ())\n        for spec in self.parent_sets.values():",
    ),
    "a project name in a comment": (
        "# up to this many examples are named in a problem\n",
        "# up to this many examples are named in a problem (e.g. GkSource)\n",
    ),
}
try:
    for label, (old, new) in MUTATIONS.items():
        assert original.count(old) == 1, label
        P.write_text(original.replace(old, new))
        r = subprocess.run([sys.executable, "-m", "unittest", "Datastore.tests.test_inventory_declarations"],
                           capture_output=True, text=True, timeout=300)
        lines = [l for l in r.stderr.splitlines() if l.startswith(("FAIL:", "ERROR:", "Ran", "OK", "FAILED"))]
        print(f"=== {label}")
        for l in lines:
            print("   ", l)
finally:
    P.write_text(original)
