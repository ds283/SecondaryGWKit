"""M6: every registry table name, project package name and the named columns left in the pool
and the actor, by line, split into code and text (comments, docstrings, string literals).

usage (from the tree, PYTHONPATH=.): python m6_names_left.py
"""
import io
import re
import tokenize
from collections import Counter

from config.datastore import factories

FILES = ["Datastore/SQL/ShardedPool.py", "Datastore/SQL/Datastore.py"]
PACKAGES = ["MetadataConcepts", "ComputeTargets", "CosmologyConcepts", "CosmologyModels",
            "LiouvilleGreen", "AdaptiveLevin", "Units", "config"]
COLUMNS = ["source", "response", "validated", "label", "tag_serial", "model_serial"]
NAMES = list(factories) + PACKAGES + COLUMNS

for path in FILES:
    text = open(path).read()
    hits = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        kind = {tokenize.NAME: "code", tokenize.STRING: "string",
                tokenize.COMMENT: "comment"}.get(tok.type)
        if kind is None:
            continue
        if kind == "string" and tok.string.lstrip("rbfuRBFU")[:3] in ('"""', "'''"):
            kind = "docstring"
        for name in NAMES:
            if kind == "code":
                found = tok.string == name
            else:
                found = re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", tok.string) is not None
            if found:
                line = text.splitlines()[tok.start[0] - 1].strip()
                hits.append((tok.start[0], kind, name, line[:140]))
    print(f"=== {path}: {len(hits)} hits; by name: {dict(Counter(h[2] for h in hits))}")
    for h in hits:
        print(f"  {h[0]:5d} {h[1]:9s} {h[2]:14s} {h[3]}")
