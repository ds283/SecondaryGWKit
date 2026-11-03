"""levin-phase-floor 06g (census agent, stage 1): a static check, by symtable, that no function in the named
scripts reads a global name that is neither defined at module level nor a builtin. New in 06g. Imports nothing
from any tree. Usage: python3 static_names.py FILE...
"""

import builtins
import symtable
import sys

ok = True
for fn in sys.argv[1:]:
    src = open(fn).read()
    st = symtable.symtable(src, fn, "exec")
    mod_names = set(st.get_identifiers())
    bad = set()

    def walk(t):
        for ch in t.get_children():
            for sym in ch.get_symbols():
                n = sym.get_name()
                if sym.is_global() and n not in mod_names and not hasattr(builtins, n):
                    bad.add((ch.get_name(), n))
            walk(ch)

    walk(st)
    ok &= not bad
    print(fn, sorted(bad) or "no undefined globals")
print("ALL CLEAN" if ok else "UNDEFINED GLOBALS FOUND")
sys.exit(0 if ok else 1)
