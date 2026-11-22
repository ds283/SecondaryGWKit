"""levin-phase-floor prompt 07c section 6, Gate Y0's first clause, the files loaded: new in 07c, after 07b's
u0a_imports_07b.py (a record, not edited), whose test it keeps. The K0 run at the parent (in h, cc3368ce, with cff4237f's
module installed) executes 07b's code if: no file of h the K0 workers loaded differs between the parent cff4237f and h
(levin_quadrature.py is replaced, and so is not loaded from h); h's own AdaptiveLevin/levin_quadrature.py is not among the
files loaded; the variant loaded is the parent's file (8956c7b6..., 07b's census module); and no file the K0 workers
loaded differs between 07b's census tree 0a103b5a and h, but levin_quadrature.py (replaced) (out/y0/y0a_ast.txt: S0'
lists CLAUDE.md alone). Read only.

Usage: python -B y0a_imports_07c.py RUN_K0_DIR OUT_TXT
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import H07B_SHA, H_SHA, H_TREE, LQ_SHA256_07CPARENT, PARENT_SHA_07C, PARENT_VARIANT_07C, header, sha256  # noqa: E402


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
    diff_ph = git_names(PARENT_SHA_07C, H_SHA)
    diff_b = git_names(H07B_SHA, H_SHA)
    loaded = files | {"main.py"}
    inter_ph = sorted(set(diff_ph) & loaded)
    inter_b = sorted(set(diff_b) & loaded)
    lq_loaded = "AdaptiveLevin/levin_quadrature.py" in files
    variant_ok = variants == {(str(PARENT_VARIANT_07C.resolve()), LQ_SHA256_07CPARENT)}
    ok = not inter_ph and not inter_b and not lq_loaded and variant_ok and n_workers > 0
    lines = header(Path(__file__).resolve(), "levin-phase-floor 07c: Gate Y0 (1), the files of h the K0 workers loaded",
                   [PARENT_VARIANT_07C])
    lines.append(f"# workers with a bye: {n_workers}; distinct files of h loaded: {len(files)}")
    lines.append(f"# variants installed (path, sha256 at hello): {sorted(variants)}; the parent's: {variant_ok}")
    lines.append(f"# h's own AdaptiveLevin/levin_quadrature.py among the files loaded: {lq_loaded}")
    lines.append(f"# git diff --name-only {PARENT_SHA_07C[:8]} {H_SHA[:8]} -- . ':!docs' ':!prompts': {diff_ph}")
    lines.append(f"# files both changed (parent to h) and loaded: {inter_ph} (none expected: levin_quadrature.py is replaced)")
    lines.append(f"# git diff --name-only {H07B_SHA[:8]} {H_SHA[:8]} -- . ':!docs' ':!prompts' (07b's census tree to h): "
                 f"{diff_b}; loaded among them: {inter_b} (none expected)")
    lines.append(f"# verdict: {'HOLDS (the code the K0 workers executed is the parent code, and the parent is 07bs)' if ok else 'FAILS'}")
    lines += sorted(files)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if l.startswith("#")))
    print(f"wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
