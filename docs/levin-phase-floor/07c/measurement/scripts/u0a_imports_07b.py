"""levin-phase-floor prompt 07b section 6, Gate U0's first clause, the files loaded: copied from 07a's w0a_imports_07a.py
(a record, not edited), changed where marked "prompt 07b": the K0 run at the parent (in h, 0a103b5a, with 7b82a0cd's
module installed) executes 07a's code if no file h loaded differs from 07a's census tree 4bf50d34 except the two
LiouvilleGreen modules whose AST without docstrings is equal (out/u0/u0a_ast.txt), h's own levin_quadrature.py is not
loaded, the variant loaded is the parent's file (4ea1948b..., 07a's), and nothing loaded differs between the parent
7b82a0cd and h but levin_quadrature.py (replaced). 07a's text follows.

levin-phase-floor prompt 07a section 6, Gate W0's first clause and the parent's install: the files of tree h that the
K0 run's workers loaded (each worker's ``bye``: ``modules_own_tree``, and the module counted under the root ``variant``),
against ``git diff --name-only dadd63a0 4bf50d34 -- . ':!docs' ':!prompts'`` (the parent against h) and against
``git diff --name-only e182bc7a dadd63a0 -- . ':!docs' ':!prompts'`` (S0', 07's census tree against the parent). New in
07a, after 07's v0a_imports_07.py (a record, not edited). Read only.

The K0 run executes the parent's code if: no file h loaded differs from the parent's except the two LiouvilleGreen modules
whose AST without docstrings is equal (out/w0/w0a_ast_lg.txt); h's own AdaptiveLevin/levin_quadrature.py is not among the
files loaded; the variant loaded is the parent's file (its SHA-256 4e5da371...); and nothing the K0 workers loaded differs
between 07's census tree e182bc7a and the parent (S0').

Usage: python -B w0a_imports_07a.py RUN_K0_DIR OUT_TXT
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import H_SHA, H_TREE, LQ_SHA256_07APARENT, PARENT_SHA_07A, PARENT_VARIANT_07A, header, sha256  # noqa: E402
from prov import H07A_SHA, LQ_SHA256_07BPARENT, PARENT_SHA_07B, PARENT_VARIANT_07B  # noqa: E402  prompt 07b

AST_EQUAL = {"LiouvilleGreen/bessel_phase.py", "LiouvilleGreen/three_bessel_integrals.py"}  # out/w0/w0a_ast_lg.txt; prompt 07b: out/u0/u0a_ast.txt


def git_names(a, b):
    return subprocess.run(["git", "--no-optional-locks", "-C", str(H_TREE), "diff", "--name-only", a, b, "--", ".",
                           ":!docs", ":!prompts"], capture_output=True, text=True, check=True).stdout.split()


def main():
    run_dir, out_p = Path(sys.argv[1]), Path(sys.argv[2])
    files = set()
    variants = set()
    n_workers = 0
    for wdir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        for line in open(wdir / "emit.jsonl"):
            m = json.loads(line)
            if m["kind"] == "bye":
                n_workers += 1
                for _name, f in m["modules_own_tree"]:
                    files.add(str(Path(f).resolve().relative_to(H_TREE)))
            if m["kind"] == "hello":
                v = m["evidence"].get("variant_06k") or {}
                variants.add((v.get("path"), v.get("sha256_now")))
    diff_ph = git_names(PARENT_SHA_07B, H_SHA)  # prompt 07b: the parent 7b82a0cd against h
    diff_s0 = git_names(H07A_SHA, H_SHA)  # prompt 07b: 07a's census tree 4bf50d34 against h (S0' and the code commit)
    loaded = files | {"main.py"}  # main.py is read by the harness (test_main_plumbing reads it with ast), not imported
    inter_ph = sorted(set(diff_ph) & loaded)
    inter_s0 = sorted(set(diff_s0) & loaded)
    lq_loaded = "AdaptiveLevin/levin_quadrature.py" in files
    variant_ok = variants == {(str(PARENT_VARIANT_07B.resolve()), LQ_SHA256_07BPARENT)}  # prompt 07b
    ok = not inter_ph and not lq_loaded and variant_ok and set(inter_s0) <= AST_EQUAL  # prompt 07b
    lines = header(Path(__file__).resolve(), "levin-phase-floor 07b: Gate U0 (1), the files of h the K0 workers loaded",
                   [PARENT_VARIANT_07B])  # prompt 07b
    lines.append(f"# workers with a bye: {n_workers}; distinct files of h loaded: {len(files)}")
    lines.append(f"# variants installed (path, sha256 at hello): {sorted(variants)}; the parent's: {variant_ok}")
    lines.append(f"# h's own AdaptiveLevin/levin_quadrature.py among the files loaded: {lq_loaded}")
    lines.append(f"# git diff --name-only {PARENT_SHA_07B[:8]} {H_SHA[:8]} -- . ':!docs' ':!prompts': {diff_ph}")  # prompt 07b
    lines.append(f"# files both changed (parent to h) and loaded: {inter_ph} (none expected: levin_quadrature.py is replaced)")
    lines.append(f"# git diff --name-only {H07A_SHA[:8]} {H_SHA[:8]} -- . ':!docs' ':!prompts' (07a's census tree to h): {diff_s0}; "
                 f"loaded among them: {inter_s0}; of these, AST-equal without docstrings (out/u0/u0a_ast.txt): "
                 f"{sorted(set(inter_s0) & AST_EQUAL)}; other: {sorted(set(inter_s0) - AST_EQUAL)}")
    lines.append(f"# verdict: {'HOLDS (the code the K0 workers executed is the parent code, and the parent is 07as)' if ok else 'FAILS'}")
    lines += sorted(files)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith("#")))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
