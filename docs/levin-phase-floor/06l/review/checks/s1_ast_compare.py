# Orchestrator's own AST comparison of 06l's variants against the tree's module (review stop 1).
import ast, hashlib, difflib
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06l-af4c4dc/AdaptiveLevin/levin_quadrature.py"
VD = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06l/scripts/variants/"
def load(p):
    src = open(p, "rb").read()
    return ast.parse(src), hashlib.sha256(src).hexdigest()
def topmap(mod):
    out = {}
    for n in mod.body:
        name = getattr(n, "name", None) or (type(n).__name__ + ":" + ast.dump(n)[:120])
        out.setdefault(name, []).append(ast.dump(n))
    return out
def fn(mod, name):
    return next(n for n in mod.body if getattr(n, "name", "") == name)
def inner(f):
    d = {}
    for i, s in enumerate(f.body):
        nm = getattr(s, "name", None) or f"#{i}:{type(s).__name__}"
        d[nm] = s
    return d
t, th = load(TREE)
print("tree", TREE, th)
for v in ["identity", "routeB"]:
    m, h = load(VD + f"levin_quadrature_{v}.py")
    tm, mm = topmap(t), topmap(m)
    diff = sorted(k for k in set(tm) | set(mm) if tm.get(k) != mm.get(k))
    print(f"\n{v}: sha256 {h}; top-level differing: {diff or 'none'}")
    if "_adaptive_levin" not in diff:
        continue
    a, b = fn(t, "_adaptive_levin"), fn(m, "_adaptive_levin")
    print("  signature equal:", ast.dump(a.args) == ast.dump(b.args),
          "decorators equal:", [ast.dump(x) for x in a.decorator_list] == [ast.dump(x) for x in b.decorator_list])
    # statement-level diff of _adaptive_levin's body by unparse
    ua = [ast.unparse(s) for s in a.body]; ub = [ast.unparse(s) for s in b.body]
    sm = difflib.SequenceMatcher(a=ua, b=ub, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal": continue
        print(f"  body {op}: tree[{i1}:{i2}] -> variant[{j1}:{j2}]")
        for s in ua[i1:i2]: print("    - " + (getattr(a.body[ua.index(s)], 'name', None) or s.splitlines()[0][:150]))
        for s in ub[j1:j2]: print("    + " + (getattr(b.body[ub.index(s)], 'name', None) or s.splitlines()[0][:150]))
    ia, ib = inner(a), inner(b)
    ca, cb = ia.get("_check_leaf"), ib.get("_check_leaf")
    if ca is not None and cb is not None:
        print("  _check_leaf line diff (unparsed, docstrings kept):")
        for line in difflib.unified_diff(ast.unparse(ca).splitlines(), ast.unparse(cb).splitlines(), lineterm="", n=1):
            print("    " + line)
