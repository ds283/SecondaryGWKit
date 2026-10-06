"""
The generic layer names nothing of the project, and imports nothing of it but two generic
modules (prompts/datastore-generic, prompt 09; README §5 rule 11; audit §C).

**What the layer is.** Every module under ``Datastore/`` but its tests and the client's factories,
with ``Datastore/SQL/ObjectFactories/base.py`` (the factory contract the layer imports), and every
module of ``RayTools/``. ``layer_files()`` derives the set from the filesystem, not from
``git ls-files`` and not from a list, so that a module added to the layer is checked before anyone
commits it or adds it here. ``test_the_derived_files_hold_the_audits_list`` pins that the derived
set still holds audit §C's list, so that a derivation that came out empty or narrow cannot pass.

**What it forbids.** The vocabulary is the registry's tables and registered class names, its
identifier columns (an underscore or a capital), the project's packages, and the names prompt 08
removed from the inventory, less ``LAYER_WORDS`` (the layer's own words, each with its reason). In
strings and comments it also forbids the two declared monotone flags' phrases and the project's
naming conventions. For each file, by ``ast`` and ``tokenize``:

1. ``test_no_code_name``: names, attributes, definitions, arguments, keywords and imported names;
2. ``test_no_string_or_docstring``: every string constant, docstrings and f-string text included;
3. ``test_no_comment``: every comment.

**What it allows.** ``test_every_import_is_allowed``: the standard library, ``ray`` and
``sqlalchemy``; the layer's own modules (of the client's factories only the contract, ``base``);
and, each by the user's choice of 2026-10-06, ``config.defaults`` and ``utilities``.
``test_the_allow_list`` pins that list without a mutation. No other entry exists: an English word
that collides with a name is reworded in the layer, not allowed here.

This module replaces three tests of one file each: prompt 07's check of ``ShardedPool.py`` in
``test_declared_facts`` and prompt 08's two of ``store_inventory.py`` in
``test_inventory_declarations``. What the layer may not name is defined once, here. What a module
*loads* when it runs is not checked here; the two fresh-interpreter tests of
``test_inventory_declarations`` do that.

Each failure names ``file:line word``, for every file at once. Nothing under ``var/`` is read, and
no Ray starts.
"""

import ast
import io
import re
import sys
import tokenize
import unittest
from pathlib import Path

import sqlalchemy as sqla

from config.datastore import factories
from Datastore import contract
from Datastore.SQL.schema import build_schema

REPO_ROOT = Path(__file__).resolve().parents[2]

# the client's factories live inside the layer's package; ``base.py`` is the contract the layer
# imports and is the layer's
_FACTORIES_DIR = ("Datastore", "SQL", "ObjectFactories")
_FACTORY_CONTRACT = "base.py"


def layer_files() -> list:
    """The layer's modules, relative to the repository root, sorted."""
    found = set()
    for path in (REPO_ROOT / "Datastore").rglob("*.py"):
        rel = path.relative_to(REPO_ROOT)
        if rel.parts[:2] == ("Datastore", "tests"):
            continue
        if rel.parts[:3] == _FACTORIES_DIR and rel.name != _FACTORY_CONTRACT:
            continue
        found.add(rel.as_posix())
    for path in (REPO_ROOT / "RayTools").glob("*.py"):
        found.add(path.relative_to(REPO_ROOT).as_posix())
    return sorted(found)


# audit §C's list, and the module prompt 07 added for the layer's contract
AUDIT_LIST = {
    "Datastore/SQL/ShardedPool.py",
    "Datastore/SQL/Datastore.py",
    "Datastore/SQL/ClientPool.py",
    "Datastore/store_inventory.py",
    "Datastore/store_reader.py",
    "Datastore/SQL/schema.py",
    "Datastore/SQL/SerialPoolBroker.py",
    "Datastore/SQL/ProfileAgent.py",
    "Datastore/object.py",
    "Datastore/replication.py",
    "Datastore/shard_paths.py",
    "RayTools/RayWorkPool.py",
    "Datastore/contract.py",
}

# ------------------------------------------------------------------------------------------------
# the vocabulary the layer may not use
# ------------------------------------------------------------------------------------------------

# the layer's own words, each of which is also a table or a column of the registry, and so would
# be flagged; each is allowed for a stated reason
LAYER_WORDS = {
    # the version and tag tables and their columns: the layer's contract (prompt 07, U4), named
    # once in Datastore.contract
    contract.VERSION_TABLE,
    contract.VERSION_LABEL,
    contract.TAG_TABLE,
    contract.TAG_LABEL,
    contract.TAG_SERIAL,
    # the column every table has, and the version column that schema.py prepends to every table
    "serial",
    "version",
    # the column the check at open compares, which every replicated write stamps (prompt 03)
    "timestamp",
    # the replication action record's own key
    "stepping",
    # the record's own field and ``read_records``' keyword, and the flag the check at open
    # recomputes: the layer defines it, it does not assume a column of that name
    "validated",
}

# packages that are part of the layer, and ``config``, which the import check handles: the layer
# imports ``config.defaults`` and nothing else of it, and "config" is an English word
_NOT_PROJECT_PACKAGES = {"Datastore", "RayTools", "config"}

# names prompt 08 removed from the inventory, which are not tables or identifier columns
EXTRA_NAMES = {
    "cosmology",
    "cosmology_type",
    "cosmology_serial",
    "COSMOLOGY",
    "model_ids",
}

# the project's naming conventions: a tag table, a value table, a run label
FRAGMENTS = ("_tags", "*Value", "Run_<")

# the two declared monotone flags, which are plain words and so not in the column list
FLAG_PHRASES = re.compile(
    r"\b(source|response)\s+(or|and)\s+(source|response)\b|\b(source|response)\s+flags?\b"
)

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def project_packages() -> set:
    return {
        path.name
        for path in REPO_ROOT.iterdir()
        if (path / "__init__.py").is_file() and path.name not in _NOT_PROJECT_PACKAGES
    }


def registry_words() -> tuple:
    """(tables and registered class names, identifier columns) of the real registry, less the
    layer's own. A column is an identifier if it has an underscore or a capital: ``model_serial``,
    ``k_inv_Mpc``. A plain-word column, such as ``source`` or ``value``, cannot be told from
    English; ``FLAG_PHRASES`` checks the two declared flags instead."""
    built = build_schema(sqla.MetaData(), factories)
    tables = (set(built.tables) | set(factories)) - LAYER_WORDS
    columns = {
        c.name
        for t in built.tables.values()
        for c in t.columns
        if "_" in c.name or c.name.lower() != c.name
    } - LAYER_WORDS
    return tables, columns


def forbidden_words() -> set:
    tables, columns = registry_words()
    return tables | columns | project_packages() | EXTRA_NAMES


# ------------------------------------------------------------------------------------------------
# the imports the layer may make
# ------------------------------------------------------------------------------------------------

# the standard library, and the two third-party packages the layer is built on
_ALLOWED_ROOTS = set(sys.stdlib_module_names) | {"ray", "sqlalchemy"}

_ALLOWED_MODULES = {
    # DEFAULT_STRING_LENGTH, the width of the layer's own text columns (ShardedPool, ProfileAgent):
    # the user's choice of 2026-10-06, over a constant of the layer's own in Datastore.contract
    "config.defaults",
    # WallclockTimer and format_time (Datastore, ProfileAgent, RayWorkPool): a shared helper that
    # imports only the standard library and knows no table; the user's choice of 2026-10-06
    "utilities",
}


def import_allowed(module: str) -> bool:
    root = module.split(".")[0]
    if root in _ALLOWED_ROOTS or module in _ALLOWED_MODULES:
        return True
    if root == "RayTools":
        return True
    if root == "Datastore":
        # the layer's own modules, and of the client's factories only the contract
        factories_pkg = ".".join(_FACTORIES_DIR)
        return not (
            module.startswith(factories_pkg)
            and module != factories_pkg + "." + _FACTORY_CONTRACT[: -len(".py")]
        )
    return False


# ------------------------------------------------------------------------------------------------
# the scans
# ------------------------------------------------------------------------------------------------


def _text_hits(text: str, forbidden: set) -> list:
    """(line offset in ``text``, word) of every forbidden word, flag phrase and fragment."""
    hits = [
        (m.start(), m.group(0)) for m in _WORD.finditer(text) if m.group(0) in forbidden
    ]
    hits += [(m.start(), repr(m.group(0))) for m in FLAG_PHRASES.finditer(text)]
    hits += [
        (m.start(), repr(f)) for f in FRAGMENTS for m in re.finditer(re.escape(f), text)
    ]
    return [(text.count("\n", 0, at), w) for at, w in hits]


def _code_names(node) -> list:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.keyword) and node.arg is not None:
        return [node.arg]
    if isinstance(node, ast.alias):
        return [n for n in (node.name.split(".")[-1], node.asname) if n]
    return []


def scan_code_names(rel: str, tree, forbidden: set) -> list:
    found = set()
    for node in ast.walk(tree):
        for name in _code_names(node):
            if {name, *name.split("_")} & forbidden:
                found.add((node.lineno, name))
    return [f"{rel}:{line} {name}" for line, name in sorted(found)]


def scan_strings(rel: str, tree, forbidden: set) -> list:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # the line within a multi-line string is exact for a docstring, approximate for a
            # string with escaped or joined lines
            found += [
                (node.lineno + off, w) for off, w in _text_hits(node.value, forbidden)
            ]
    return [f"{rel}:{line} {w}" for line, w in sorted(set(found))]


def scan_comments(rel: str, source: str, forbidden: set) -> list:
    found = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            found += [
                (token.start[0], w) for _, w in _text_hits(token.string, forbidden)
            ]
    return [f"{rel}:{line} {w}" for line, w in sorted(set(found))]


def _absolute(rel: str, node: ast.ImportFrom) -> str:
    """The module a ``from`` import names, with a relative import resolved against ``rel``."""
    if not node.level:
        return node.module
    package = Path(rel).parent.parts
    base = package[: len(package) - (node.level - 1)]
    return ".".join(base + ((node.module,) if node.module else ()))


def scan_imports(rel: str, tree) -> list:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [_absolute(rel, node)]
        else:
            continue
        found += [f"{rel}:{node.lineno} {m}" for m in modules if not import_allowed(m)]
    return found


class _LayerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = layer_files()
        cls.forbidden = forbidden_words()
        cls.sources = {rel: (REPO_ROOT / rel).read_text() for rel in cls.files}
        cls.trees = {rel: ast.parse(src) for rel, src in cls.sources.items()}

    def assertNoHits(self, found):
        self.assertEqual([], found, "\n" + "\n".join(found))


class TestTheLayer(_LayerTestCase):
    def test_the_derived_files_hold_the_audits_list(self):
        self.assertLessEqual(AUDIT_LIST, set(self.files))

    def test_no_test_and_no_client_factory_is_in_it(self):
        self.assertIn("Datastore/SQL/ObjectFactories/base.py", self.files)
        for rel in self.files:
            self.assertFalse(rel.startswith("Datastore/tests/"), rel)
            if rel.startswith("Datastore/SQL/ObjectFactories/"):
                self.assertEqual("Datastore/SQL/ObjectFactories/base.py", rel)

    def test_the_vocabulary_holds_the_registry_and_the_packages(self):
        # a vocabulary that came out empty would pass every scan below
        tables, columns = registry_words()
        self.assertLessEqual({"BackgroundModel", "redshift", "tolerance"}, tables)
        self.assertLessEqual({"model_serial", "parent_serial"}, columns)
        self.assertLessEqual(
            {"CosmologyModels", "MetadataConcepts", "tools"}, project_packages()
        )
        self.assertFalse(LAYER_WORDS & self.forbidden)


class TestTheLayerNamesNoProjectWord(_LayerTestCase):
    def test_no_code_name(self):
        found = []
        for rel in self.files:
            found += scan_code_names(rel, self.trees[rel], self.forbidden)
        self.assertNoHits(found)

    def test_no_string_or_docstring(self):
        found = []
        for rel in self.files:
            found += scan_strings(rel, self.trees[rel], self.forbidden)
        self.assertNoHits(found)

    def test_no_comment(self):
        found = []
        for rel in self.files:
            found += scan_comments(rel, self.sources[rel], self.forbidden)
        self.assertNoHits(found)


class TestTheLayerImportsNoProjectPackage(_LayerTestCase):
    def test_every_import_is_allowed(self):
        found = []
        for rel in self.files:
            found += scan_imports(rel, self.trees[rel])
        self.assertNoHits(found)

    def test_the_allow_list(self):
        # an allow-list that grew silently would pass every scan above
        for module in (
            "config.defaults",
            "utilities",
            "Datastore.SQL.ObjectFactories.base",
            "RayTools.RayWorkPool",
            "sqlalchemy.exc",
        ):
            with self.subTest(allowed=module):
                self.assertTrue(import_allowed(module))
        for module in (
            "config.sharding",
            "config.datastore",
            "Datastore.SQL.ObjectFactories.GkSource",
            "CosmologyModels.model_ids",
            "MetadataConcepts",
        ):
            with self.subTest(refused=module):
                self.assertFalse(import_allowed(module))


if __name__ == "__main__":
    unittest.main()
