"""levin-phase-floor prompt 06l section 1: copied from 06k's scripts/ast_check_variants.py (a record; its copy is in
state/scripts_06k_reference/), changed only where marked "prompt 06l": the names a variant binds must end in
``_06l`` (06k: ``_06k``), and the title names 06l. 06k's text follows.

levin-phase-floor prompt 06k section 1 ("The diff is confined"): an AST comparison of each variant
with the tree's ``AdaptiveLevin/levin_quadrature.py``. New in 06k.

For each variant it shows:
1. that every top-level statement of the module except ``def _adaptive_levin`` is equal to the
   tree's (``ast.dump``, no attributes), so nothing outside ``_adaptive_levin`` differs (comments are
   not in the AST, and the variants change none outside it either: ``out/variant_*_diff.txt``);
2. inside ``_adaptive_levin``, every differing statement, located by a sequence match of the
   statements' dumps, recursing into compound statements (def, while, if, for, try, with) whose
   heads are equal, with the old and new code (``ast.unparse``);
3. the names each variant binds that the tree's function does not: each must end in ``_06k``.

The identity variant must show no difference at all. Writes its report to stdout.
Usage: python -B ast_check_variants.py TREE_FILE VARIANT_FILE [VARIANT_FILE ...]
"""

import ast
import difflib
import hashlib
import sys
from pathlib import Path


def dump(node):
    return ast.dump(node, include_attributes=False)


def head_dump(node):
    """A compound statement's dump with its bodies emptied, to match heads."""
    clone = ast.parse(ast.unparse(node)).body[0] if not isinstance(node, ast.expr) else node
    for field in ("body", "orelse", "finalbody", "handlers"):
        if hasattr(clone, field):
            setattr(clone, field, [])
    return ast.dump(clone, include_attributes=False)


COMPOUND = (ast.FunctionDef, ast.While, ast.If, ast.For, ast.Try, ast.With)


def diff_bodies(old, new, path, out):
    a = [dump(s) for s in old]
    b = [dump(s) for s in new]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if (
            tag == "replace"
            and i2 - i1 == 1
            and j2 - j1 == 1
            and isinstance(old[i1], COMPOUND)
            and type(old[i1]) is type(new[j1])
            and head_dump(old[i1]) == head_dump(new[j1])
        ):
            o, n = old[i1], new[j1]
            label = f"{path} > {type(o).__name__}" + (f" {o.name}" if isinstance(o, ast.FunctionDef) else f" (line {n.lineno})")
            for field in ("body", "orelse", "finalbody"):
                if hasattr(o, field):
                    diff_bodies(getattr(o, field), getattr(n, field), f"{label}.{field}", out)
            continue
        out.append(
            {
                "path": path,
                "tag": tag,
                "old_lines": [getattr(s, "lineno", None) for s in old[i1:i2]],
                "new_lines": [getattr(s, "lineno", None) for s in new[j1:j2]],
                "old": [ast.unparse(s) for s in old[i1:i2]],
                "new": [ast.unparse(s) for s in new[j1:j2]],
            }
        )


def bound_names(fn):
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef,)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def find_fn(mod, name):
    for s in mod.body:
        if isinstance(s, ast.FunctionDef) and s.name == name:
            return s
    raise SystemExit(f"no {name}")


def main():
    tree_file = Path(sys.argv[1])
    t_src = tree_file.read_text()
    t_mod = ast.parse(t_src)
    print("# levin-phase-floor 06l: the variants' AST comparison with the tree's module")  # prompt 06l
    print(f"# script ast_check_variants.py sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}")
    print(f"# input {tree_file} sha256 {hashlib.sha256(tree_file.read_bytes()).hexdigest()}")
    ok_all = True
    for vf in map(Path, sys.argv[2:]):
        v_src = vf.read_text()
        v_mod = ast.parse(v_src)
        print(f"\n== {vf.name} (sha256 {hashlib.sha256(vf.read_bytes()).hexdigest()})")
        # 1. top level
        top_ok = len(t_mod.body) == len(v_mod.body)
        differing_top = []
        for s_t, s_v in zip(t_mod.body, v_mod.body):
            if dump(s_t) != dump(s_v):
                name = getattr(s_t, "name", type(s_t).__name__)
                differing_top.append(name)
        print(f"top-level statements: tree {len(t_mod.body)}, variant {len(v_mod.body)}; differing: {differing_top}")
        only_adaptive = top_ok and set(differing_top) <= {"_adaptive_levin"}
        print(f"only _adaptive_levin differs at top level: {only_adaptive}")
        # 2. inside _adaptive_levin
        f_t, f_v = find_fn(t_mod, "_adaptive_levin"), find_fn(v_mod, "_adaptive_levin")
        same_sig = dump(f_t.args) == dump(f_v.args) and ast.get_docstring(f_t) == ast.get_docstring(f_v)
        print(f"_adaptive_levin's signature and docstring unchanged: {same_sig}")
        out = []
        diff_bodies(f_t.body, f_v.body, "_adaptive_levin", out)
        print(f"differing statement groups inside _adaptive_levin: {len(out)}")
        for k, d in enumerate(out, 1):
            print(f"-- [{k}] at {d['path']}: {d['tag']}; tree lines {d['old_lines']}, variant lines {d['new_lines']}")
            for s in d["old"]:
                print("   - " + s.replace("\n", "\n     "))
            for s in d["new"]:
                print("   + " + s.replace("\n", "\n     "))
        # 3. new names
        new_names = sorted(bound_names(f_v) - bound_names(f_t))
        bad = [n for n in new_names if not n.endswith("_06l")]  # prompt 06l (06k: _06k)
        print(f"names bound in the variant's _adaptive_levin and not the tree's: {new_names}")
        print(f"of them, not ending in _06l: {bad}")  # prompt 06l
        ok = only_adaptive and same_sig and not bad
        if vf.name.endswith("_identity.py"):
            ok = ok and not out and not differing_top and v_src == t_src
            print(f"identity: no difference at all: {ok}")
        print(f"CONFINED: {ok}")
        ok_all &= ok
    print(f"\nALL CONFINED: {ok_all}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
