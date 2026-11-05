# Orchestrator's own AST comparison of each 06k variant against the tree's module (review stop 1).
import ast, sys, hashlib
TREE = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf06k-8937060/AdaptiveLevin/levin_quadrature.py"
VD = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06k/scripts/variants/"
def top(path):
    src = open(path, "rb").read()
    mod = ast.parse(src)
    out = {}
    for i, n in enumerate(mod.body):
        name = getattr(n, "name", None)
        if name is None:
            if isinstance(n, ast.Assign):
                name = "assign:" + ",".join(ast.dump(t) for t in n.targets)
            elif isinstance(n, ast.AnnAssign):
                name = "annassign:" + ast.dump(n.target)
            else:
                name = f"stmt#{type(n).__name__}:{ast.dump(n)[:200]}"
        out.setdefault(name, []).append(ast.dump(n))
    return out, hashlib.sha256(src).hexdigest()
t, th = top(TREE)
print("tree", TREE, th)
for v in ["identity", "route1", "route2", "route3"]:
    p = VD + f"levin_quadrature_{v}.py"
    m, h = top(p)
    diff = sorted(k for k in set(t) | set(m) if t.get(k) != m.get(k))
    print(f"{v}: sha256 {h}; top-level statements differing: {diff if diff else 'none'}")
    if "_adaptive_levin" in diff:
        a = next(n for n in ast.parse(open(TREE).read()).body if getattr(n, "name", "") == "_adaptive_levin")
        b = next(n for n in ast.parse(open(p).read()).body if getattr(n, "name", "") == "_adaptive_levin")
        # nested defs inside _adaptive_levin
        def inner(f):
            d = {}
            for i, s in enumerate(f.body):
                nm = getattr(s, "name", None) or f"#{i}:{type(s).__name__}"
                d[nm] = ast.dump(s)
            return d
        ia, ib = inner(a), inner(b)
        named_a = {k for k in ia if not k.startswith("#")}
        named_b = {k for k in ib if not k.startswith("#")}
        print("   nested defs added:", sorted(named_b - named_a), "removed:", sorted(named_a - named_b),
              "changed:", sorted(k for k in named_a & named_b if ia[k] != ib[k]))
        print("   signature equal:", ast.dump(a.args) == ast.dump(b.args), "decorators equal:",
              [ast.dump(x) for x in a.decorator_list] == [ast.dump(x) for x in b.decorator_list])
