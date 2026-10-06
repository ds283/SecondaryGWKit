"""M6: every registry table name, project package name, project column and naming pattern left in
the given files, by line, split into code and text (comments, docstrings, string literals).

usage (from a tree, PYTHONPATH=.): python m6_names_left.py <file> [<file> ...]
"""
import io
import re
import sys
import tokenize
from collections import Counter

from config.datastore import factories

PACKAGES = ["MetadataConcepts", "ComputeTargets", "CosmologyConcepts", "CosmologyModels",
            "LiouvilleGreen", "AdaptiveLevin", "Units", "config", "model_ids"]
COLUMNS = ["source", "response", "validated", "label", "tag_serial", "model_serial", "cosmology",
           "cosmology_type", "cosmology_serial", "z_source", "z_init", "Tr", "Tq", "parents_digest",
           "COSMOLOGY", "COMPUTE_TARGETS", "INVENTORY_CLASSES", "cosmology_classes",
           "LAMBDACDM_IDENTIFIER", "QCD_EOS_IDENTIFIER", "Gk", "WKB"]
PATTERNS = [r"\*Value", r"\*_tags", r"Run_<"]
NAMES = list(factories) + PACKAGES + COLUMNS

for path in sys.argv[1:]:
    text = open(path).read()
    lines = text.splitlines()
    hits = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        kind = {tokenize.NAME: "code", tokenize.STRING: "string",
                tokenize.COMMENT: "comment"}.get(tok.type)
        if kind is None:
            continue
        if kind == "string" and tok.string.lstrip("rbfuRBFU")[:3] in ('"""', "'''"):
            kind = "docstring"
        # a multi-line token: report each hit on its own line
        for offset, part in enumerate(tok.string.split("\n")):
            lineno = tok.start[0] + offset
            for name in NAMES:
                if kind == "code":
                    found = part == name
                else:
                    found = re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", part) is not None
                if found:
                    hits.append((lineno, kind, name, lines[lineno - 1].strip()[:130]))
            if kind != "code":
                for pat in PATTERNS:
                    if re.search(pat, part):
                        hits.append((lineno, kind, pat, lines[lineno - 1].strip()[:130]))
    print(f"=== {path}: {len(hits)} hits; by kind: {dict(Counter(h[1] for h in hits))}; by name: {dict(Counter(h[2] for h in hits))}")
    for h in hits:
        print(f"  {h[0]:5d} {h[1]:9s} {h[2]:20s} {h[3]}")
