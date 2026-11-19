"""levin-phase-floor prompt 07c Part A, section 2.4: register a variant (or the instrument) for replay_07c.py. New in 07c.

Records scripts/variants/levin_quadrature_07c_<name>.py in out/variants/variants.json with its SHA-256, and writes its
unified diff against the tree's module (8956c7b6...) to out/variants/<name>.diff. An entry once written is never changed:
a changed variant is registered under a new name. The variant's ``_VARIANT_07C`` must equal its name.
Usage: python -B register_variant_07c.py NAME [NOTE]
"""
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE_LQ = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-2fd979f/AdaptiveLevin/levin_quadrature.py")
TREE_SHA = "8956c7b6a8ec285f8898353eed65f4039a02b42edc6927565be3896cda52def1"


def main():
    name = sys.argv[1]
    note = sys.argv[2] if len(sys.argv) > 2 else ""
    vp = HERE / "variants" / f"levin_quadrature_07c_{name}.py"
    src = vp.read_text()
    m = re.search(r'^_VARIANT_07C = ["\']([^"\']+)["\']', src, re.M)  # v2: either quote (v1 in scripts/superseded/)
    if not m or m.group(1) != name:
        raise SystemExit("the variant's _VARIANT_07C is not its name")
    tree = TREE_LQ.read_text()
    if hashlib.sha256(tree.encode()).hexdigest() != TREE_SHA:
        raise SystemExit("the tree's module has another SHA-256")
    vj = ROOT / "out" / "variants" / "variants.json"
    vj.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(vj.read_text()) if vj.exists() else {"tree_module": str(TREE_LQ), "tree_sha256": TREE_SHA, "variants": {}}
    if name in doc["variants"]:
        raise SystemExit(f"{name} is registered already")
    sha = hashlib.sha256(src.encode()).hexdigest()
    diff = "".join(difflib.unified_diff(tree.splitlines(True), src.splitlines(True), "tree/AdaptiveLevin/levin_quadrature.py",
                                        f"scripts/variants/{vp.name}", n=3))
    dp = ROOT / "out" / "variants" / f"{name}.diff"
    if dp.exists():
        raise SystemExit(f"{dp} exists")
    dp.write_text(diff)
    doc["variants"][name] = {"path": str(vp), "sha256": sha, "diff": str(dp), "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                             "diff_lines_added": sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")),
                             "diff_lines_removed": sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")),
                             "note": note}
    vj.write_text(json.dumps(doc, indent=1))
    print(f"{name}: {vp} sha256 {sha}; diff {dp} (+{doc['variants'][name]['diff_lines_added']} -{doc['variants'][name]['diff_lines_removed']})")


if __name__ == "__main__":
    main()
