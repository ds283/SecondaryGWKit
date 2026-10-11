"""
Rewrite SGK's imports of its datastore layer to ``datastorekit``, by module path only.

dk-and-rpk-migration prompt 04, A2. The record of how the rewrite was made.

The rewrite is driven by DatastoreKit's measuring script, not by a search of this tree. Its JSON
lists every import statement, string literal and ``import_module`` call that names one of the 17
layer modules, each with its file and line, at one commit:

    (from DatastoreKit's root, with its venv)
    ./venv/bin/python docs/extraction/measure_client_imports.py sgk --commit <SHA> --json > <json>

This script reads that JSON and rewrites each listed site **at its file and line**, outside
``prompts/``, by DatastoreKit's adoption README section 2 module map. It changes the module path
and nothing else on the line: names, aliases, nesting and formatting stay.

It refuses a site, and changes no file at all, when:
- the line in the working tree is not the line at the JSON's commit (a stale line number);
- the line does not name the module the JSON says, as a whole dotted name: after ``from`` and
  before ``import`` for a statement, inside quotes for a literal or an ``import_module`` argument,
  exactly once;
- the statement is not a ``from`` import of exactly the one module it names (no such statement
  stands outside ``prompts/`` at 04's ``HEAD``; one would need a hand rewrite of its uses).

A site in a file that is absent from the working tree (deleted by A1 or A3) is skipped and
counted. Before writing anything it checks that every name each rewritten ``from`` statement
imports exists on the package module it will name, by ``getattr``. A missing name is a refusal,
except ``_factories`` from ``Datastore.SQL.Datastore``: it left that module at ``19f07ee``, and the
six ``docs/`` files that import it fail before and after (A5).

Usage, from SGK's root:

    ./venv/bin/python prompts/dk-and-rpk-migration/scripts/rewrite_layer_imports.py <json> [--dry-run]

It prints ``file:line old -> new`` for each change and the counts, and exits 0, or 1 on a refusal.
"""

import argparse
import importlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# DatastoreKit docs/adoption/README.md section 2, for SGK's 17 layer files
MODULE_MAP = {
    "Datastore": "datastorekit",
    "Datastore.object": "datastorekit.object",
    "Datastore.contract": "datastorekit.contract",
    "Datastore.replication": "datastorekit.replication",
    "Datastore.shard_paths": "datastorekit.shard_paths",
    "Datastore.store_reader": "datastorekit.store_reader",
    "Datastore.store_inventory": "datastorekit.store_inventory",
    "Datastore.SQL": "datastorekit.SQL",
    "Datastore.SQL.schema": "datastorekit.SQL.schema",
    "Datastore.SQL.ShardedPool": "datastorekit.SQL.ShardedPool",
    "Datastore.SQL.Datastore": "datastorekit.SQL.Datastore",
    "Datastore.SQL.ClientPool": "datastorekit.SQL.ClientPool",
    "Datastore.SQL.SerialPoolBroker": "datastorekit.SQL.SerialPoolBroker",
    "Datastore.SQL.ProfileAgent": "datastorekit.SQL.ProfileAgent",
    "Datastore.SQL.ObjectFactories.base": "datastorekit.SQL.factory_base",
    "tools.sharded_store": "datastorekit.tools.sharded_store",
    "tools.shard_key_audit": "datastorekit.tools.shard_key_audit",
}

# A5: the one name a staying statement imports that the package lacks, and the module it is
# imported from. It left SGK's layer at 19f07ee; the files that import it fail before and after.
KNOWN_MISSING = {("Datastore.SQL.Datastore", "_factories")}

# a dotted name is whole when no identifier character or dot touches either end
_EDGE = r"[\w.]"


def whole(name):
    return rf"(?<!{_EDGE}){re.escape(name)}(?!{_EDGE})"


class Refusal(Exception):
    pass


def line_at(commit, path, line):
    text = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return text.splitlines(keepends=True)[line - 1]


def substitute(text, pattern, old, new, where):
    hits = list(re.finditer(pattern, text))
    if len(hits) != 1:
        raise Refusal(f"{where}: {len(hits)} matches of {old!r} on the line, not 1")
    m = hits[0]
    return text[: m.start("mod")] + new + text[m.end("mod") :]


def plan_statement(site, text, where):
    if site["statement"] != "from":
        raise Refusal(f"{where}: an `import` statement; rewrite it by hand")
    old = site["written"]
    if site["modules"] != [old] or old not in MODULE_MAP:
        raise Refusal(
            f"{where}: written {old!r} is not the one module {site['modules']}"
        )
    new = MODULE_MAP[old]
    pattern = rf"\bfrom\s+(?P<mod>{whole(old)})\s+import\b"
    return old, new, substitute(text, pattern, old, new, where)


def plan_quoted(module, literal, text, where):
    if module not in MODULE_MAP:
        raise Refusal(f"{where}: {module!r} is not a layer module")
    new_literal = MODULE_MAP[module] + literal[len(module) :]
    pattern = rf"(?P<q>[\"'])(?P<mod>{re.escape(literal)})(?P=q)"
    return literal, new_literal, substitute(text, pattern, literal, new_literal, where)


def check_names(sites):
    """Every name a rewritten ``from`` statement imports, by getattr on the package module."""
    missing, checked = [], 0
    for site, where in sites:
        module = importlib.import_module(MODULE_MAP[site["written"]])
        for name in site["names"]:
            name = name.split(" as ")[0]
            checked += 1
            if not hasattr(module, name):
                missing.append((site["written"], name, where))
    return checked, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    data = json.loads(args.json.read_text())
    commit = data["commit"]
    kinds = (
        ("statement", data["imports"]),
        ("literal", data["string literals"]),
        ("import_module", data["import_module calls"]),
    )

    counts = Counter()
    edits = {}  # path -> {line: (kind, old, new, new_text)}
    statements = []
    try:
        for kind, sites in kinds:
            for site in sites:
                path, line = site["file"], site["line"]
                where = f"{path}:{line}"
                if path.startswith("prompts/"):
                    counts[f"{kind}, prompts/ (left)"] += 1
                    continue
                if not Path(path).exists():
                    counts[f"{kind}, file deleted (skipped)"] += 1
                    continue
                lines = edits.setdefault(path, {})
                current = Path(path).read_text().splitlines(keepends=True)
                text = lines[line][3] if line in lines else current[line - 1]
                if line not in lines and text != line_at(commit, path, line):
                    raise Refusal(f"{where}: the line is not the line at {commit[:7]}")
                if kind == "statement":
                    old, new, text = plan_statement(site, text, where)
                    statements.append((site, where))
                    counts["statement, nested" if site["nested"] else "statement"] += 1
                elif kind == "literal":
                    old, new, text = plan_quoted(
                        site["module"], site["literal"], text, where
                    )
                    counts[kind] += 1
                else:
                    old, new, text = plan_quoted(
                        site["module"], site["argument"], text, where
                    )
                    counts[kind] += 1
                lines[line] = (kind, old, new, text)

        checked, missing = check_names(statements)
        unknown = [m for m in missing if (m[0], m[1]) not in KNOWN_MISSING]
        for module, name, where in missing:
            known = "REFUSED" if (module, name, where) in unknown else "known (A5)"
            print(f"missing name: {where} {module}.{name}: {known}")
        if unknown:
            raise Refusal(f"{len(unknown)} imported names are not in the package")
    except Refusal as refusal:
        print(f"REFUSED, nothing written: {refusal}", file=sys.stderr)
        return 1

    for path in sorted(edits):
        current = Path(path).read_text().splitlines(keepends=True)
        for line, (kind, old, new, text) in sorted(edits[path].items()):
            print(f"{path}:{line} {old} -> {new}")
            current[line - 1] = text
        if not args.dry_run:
            Path(path).write_text("".join(current))

    print()
    print(f"JSON: {args.json} (commit {commit[:7]})")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    changed = sum(len(v) for v in edits.values())
    print(f"lines changed: {changed}, in {len(edits)} files")
    print(f"imported names checked: {checked}; missing: {len(missing)}")
    if args.dry_run:
        print("dry run: nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
