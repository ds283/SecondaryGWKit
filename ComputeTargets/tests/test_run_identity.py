"""
Naming a run at write time and verifying it at read time: prompt 14 of
``prompts/qcd-background-audit``.

Three things had gone wrong together, and they share a root -- nothing recorded *which run* an
object belonged to:

1. ``sqla_BackgroundModel_factory.build()`` filtered on ``(cosmology_type, cosmology_serial,
   atol_serial, rtol_serial)`` -- since prompt 05 of ``prompts/tolerance-convergence`` the
   tolerance half of that key is the three Gauss orders ``(tau_gauss_order, cs_tau_gauss_order,
   friction_F_gauss_order)``, which is what the payloads below carry -- plus ``LargestSourceZTag`` / ``SmallestSourceZTag`` / (until
   prompt 16 retired it) ``SourceSamplesPerLog10ZTag``, every one of which is **unchanged** when
   the grid's shape changes; and it never filtered on ``z_sample`` at all. So when prompt 11 gave the QCD grid 41
   extra samples, a pre-prompt-11 datastore went on serving its 1,732-node background for the new
   1,773-node grid. It was the one surviving row in a store whose every compute target the grid
   tag had already invalidated, which makes it worse than a miss: the next run finds it, uses it,
   and is wrong.
2. All seven ``extract_*.py`` scripts passed **zero** tags, against ``main.py``'s eight-plus
   tagged call sites -- safe while a store held one run, unsafe the moment it holds two.
3. Nothing named a run at all, so *"read me the run I did in March"* could not be expressed.

What is asserted here, in the order prompt 14 §3 states it:

* the construction version is declared **once**, in ``CosmologyConcepts/wavenumber.py``, and the
  factory reads it rather than inlining a literal (by ``ast``, following
  ``test_cosmology_representation_key.test_the_filter_is_not_a_literal``: a literal stops
  tracking the constant the moment someone bumps it, which is the exact failure the key exists to
  prevent);
* two generations of the grid do not collide, and the same generation reuses one row -- against a
  real in-memory SQLite database built from the factories' own ``register()``, because the point
  of the prompt is what SQL does with a candidate row;
* a run round-trips by name; a label spanning two configurations is **refused, naming both**;
* a store that records no run label at all still **reads**, labelled unknown, on the read path,
  while the compute path refuses it -- two different answers to the same old datastore, which is
  the whole of prompt 14's archival requirement;
* ``main.py`` tags what it builds and parses the new argument, and every ``extract_*.py`` selects
  a run -- both by ``ast``, since ``main.py`` cannot be imported (``CLAUDE.md``) and the extract
  scripts open a Ray connection and a ``ShardedPool`` at module scope for the same reason.

Nothing here needs Ray or the project's ``Datastore`` machinery: the tables are built from the
factories' own ``register()`` output in the shape ``Datastore.SQL.Datastore._build_schema`` builds
them, and the inserter is the shape of ``Datastore._insert`` reduced to what these tables use.
"""

import ast
from importlib import import_module
import unittest
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import sqlalchemy as sqla
from sqlalchemy.exc import OperationalError

import Datastore.SQL.ObjectFactories.BackgroundModel as background_model_factory_module
from ComputeTargets.BackgroundModel import BackgroundModel, BackgroundModelValue
from ComputeTargets.tests.test_main_plumbing import MAIN_PY
from CosmologyConcepts import redshift, redshift_array
from CosmologyConcepts.wavenumber import SOURCE_GRID_CONSTRUCTION_VERSION
from Datastore.SQL.ObjectFactories.BackgroundModel import (
    sqla_BackgroundModelFactory,
    sqla_BackgroundModelTagAssociation_factory,
    sqla_BackgroundModelValue_factory,
)
from Datastore.SQL.ObjectFactories.integration_metadata import (
    sqla_IntegrationSolver_factory,
)
from Datastore.SQL.ObjectFactories.redshift import sqla_redshift_factory
from Datastore.SQL.ObjectFactories.store_tag import sqla_store_tag_factory
from MetadataConcepts import store_tag
from Quadrature.integration_metadata import IntegrationData, IntegrationSolver
from Units import Mpc_units
from extract_common import (
    RUN_LABEL_TAG_PREFIX,
    choose_run_label,
    describe_background_generation,
    run_label_tag,
    source_grid_construction_tag,
)

# the module, not the class: ComputeTargets/__init__.py rebinds the package attribute to the
# class, so the three Gauss orders have to be reached through import_module, exactly as the
# factory reaches them
BackgroundModelModule = import_module("ComputeTargets.BackgroundModel")

REPO_ROOT = Path(__file__).parents[2]

WAVENUMBER_SOURCE = REPO_ROOT / "CosmologyConcepts" / "wavenumber.py"
FACTORY_SOURCE = (
    REPO_ROOT / "Datastore" / "SQL" / "ObjectFactories" / "BackgroundModel.py"
)

CONSTANT_NAME = "SOURCE_GRID_CONSTRUCTION_VERSION"
DIGEST_COLUMN = "source_grid_digest"
CONSTRUCTION_COLUMN = "source_grid_construction"

EXTRACT_SCRIPTS = [
    "extract_Gk_data.py",
    "extract_GkSource_data.py",
    "extract_GkWKB_data.py",
    "extract_QuadSourceIntegral_data.py",
    "extract_TkWKB_data.py",
    "extract_tensor_source_data.py",
]


# ---------------------------------------------------------------------------------------------
# schema and connection helpers (the shape of test_cosmology_representation_key's, widened to the
# tables BackgroundModel.build() and store() reach into)
# ---------------------------------------------------------------------------------------------


def _append_table(
    name: str, factory, metadata: sqla.MetaData, omit: Optional[List[str]] = None
) -> sqla.Table:
    """
    Reproduce the table Datastore._build_schema builds for this factory, from the factory's own
    register() output -- so the schema under test is the production schema, not a copy of it.

    ``omit`` drops columns, which is how the pre-prompt-14 schema is reproduced: there is no other
    record of it, and hand-writing the old column list would not stay in step with the factory.
    """
    registration = factory.register()
    omit = omit if omit is not None else []

    tab = sqla.Table(name, metadata)
    if registration.get("serial", True):
        tab.append_column(sqla.Column("serial", sqla.Integer, primary_key=True))
    if registration.get("version", False):
        tab.append_column(sqla.Column("version", sqla.Integer))
    if registration.get("timestamp", False):
        tab.append_column(sqla.Column("timestamp", sqla.DateTime()))
    if registration.get("stepping", False):
        tab.append_column(sqla.Column("stepping", sqla.Integer))

    for column in registration.get("columns", []):
        if column.name in omit:
            continue
        tab.append_column(column)

    return tab


def _inserter_for(table: sqla.Table):
    """
    What Datastore._insert does for these tables: the serial the engine assigns, and a timestamp
    for the tables that registered one (BackgroundModelValue registers "timestamp": False, being
    a high-volume child table).
    """
    has_timestamp = "timestamp" in table.c

    def inserter(conn, data):
        payload = dict(data)
        if has_timestamp:
            payload["timestamp"] = datetime.now()
        return conn.execute(sqla.insert(table), payload).lastrowid

    return inserter


class _StandInCosmology:
    """
    The duck type BackgroundModel's factory needs: an identity and a unit system. Nothing here
    evaluates a background, so no real cosmology is built (and none needs to be: the subject is
    the lookup key, not the physics).
    """

    type_id = 4242

    def __init__(self, store_id: int = 1):
        self.store_id = store_id
        self.units = Mpc_units()


class _Schema:
    """One in-memory SQLite database carrying the tables build() and store() touch."""

    def __init__(self, omit_grid_identity: bool = False):
        omit = [DIGEST_COLUMN, CONSTRUCTION_COLUMN] if omit_grid_identity else []

        self.metadata = sqla.MetaData()
        self.tables = {
            "store_tag": _append_table(
                "store_tag", sqla_store_tag_factory, self.metadata
            ),
            "IntegrationSolver": _append_table(
                "IntegrationSolver", sqla_IntegrationSolver_factory, self.metadata
            ),
            "redshift": _append_table("redshift", sqla_redshift_factory, self.metadata),
            "BackgroundModel": _append_table(
                "BackgroundModel", sqla_BackgroundModelFactory, self.metadata, omit=omit
            ),
        }
        self.tables["BackgroundModel_tags"] = _append_table(
            "BackgroundModel_tags",
            sqla_BackgroundModelTagAssociation_factory,
            self.metadata,
        )
        self.tables["BackgroundModelValue"] = _append_table(
            "BackgroundModelValue", sqla_BackgroundModelValue_factory, self.metadata
        )

        self.engine = sqla.create_engine("sqlite://", future=True)
        self.metadata.create_all(self.engine)
        self.conn = self.engine.connect()

        self.inserters = {
            name: _inserter_for(table) for name, table in self.tables.items()
        }

        self.table = self.tables["BackgroundModel"]
        self.inserter = self.inserters["BackgroundModel"]

        # the ingredients every background model in this database shares
        self.cosmology = _StandInCosmology()
        self.solver = IntegrationSolver(store_id=1, label="test-solver", stepping=0)

        self.conn.execute(
            sqla.insert(self.tables["IntegrationSolver"]),
            [{"serial": 1, "label": "test-solver", "stepping": 0}],
        )

        self._next_z_serial = 1
        self._next_tag_serial = 1

    def close(self):
        self.conn.close()
        self.engine.dispose()

    # -- fixtures ------------------------------------------------------------------------------

    def z_sample(self, z_values) -> redshift_array:
        """A redshift grid, with its rows in the redshift table so that build() can read it back."""
        points = []
        for z in z_values:
            serial = self._next_z_serial
            self._next_z_serial += 1
            self.conn.execute(
                sqla.insert(self.tables["redshift"]),
                [{"serial": serial, "z": float(z), "source": True, "response": False}],
            )
            points.append(redshift(store_id=serial, z=float(z), is_source=True))

        return redshift_array(points)

    def tag(self, label: str) -> store_tag:
        serial = self._next_tag_serial
        self._next_tag_serial += 1
        self.conn.execute(
            sqla.insert(self.tables["store_tag"]), [{"serial": serial, "label": label}]
        )
        return store_tag(store_id=serial, label=label)

    def write_model(self, z_sample: redshift_array, tags=None, label="test-model"):
        """
        Store a background model through the production ``store()``, then validate it through the
        production ``validate()``. Only the sample values are synthetic; every column that
        matters here -- the digest, the construction version, the tag rows -- is written by the
        factory from the same sources ``build()`` reads.
        """
        obj = BackgroundModel(
            payload=None,
            solver_labels={},
            cosmology=self.cosmology,
            z_sample=z_sample,
            label=label,
            tags=list(tags) if tags is not None else [],
        )
        obj._solver = self.solver
        # the three orders store() records off the compute_background payload: this object is
        # standing in for one whose tables were built at the orders this run is configured at
        # (prompts/tolerance-convergence, prompt 05b)
        obj._tau_gauss_order = BackgroundModelModule.TAU_GAUSS_ORDER
        obj._cs_tau_gauss_order = BackgroundModelModule.CS_TAU_GAUSS_ORDER
        obj._friction_F_gauss_order = BackgroundModelModule.FRICTION_F_GAUSS_ORDER
        obj._data = IntegrationData(
            compute_time=1.0,
            compute_steps=1,
            RHS_evaluations=1,
            mean_RHS_time=1.0,
            max_RHS_time=1.0,
            min_RHS_time=1.0,
        )
        obj._values = [
            BackgroundModelValue(
                store_id=None,
                z=z,
                Hubble=1.0,
                wBackground=1.0 / 3.0,
                wPerturbations=1.0 / 3.0,
                rho=1.0,
                tau=1.0,
                T_photon=1.0,
                d_lnH_dz=1.0,
                d2_lnH_dz2=1.0,
                d3_lnH_dz3=1.0,
                d_wPerturbations_dz=1.0,
                d2_wPerturbations_dz2=1.0,
                tau_lo=0.0,
                cs_tau=1.0,
                cs_tau_lo=0.0,
                friction_F=1.0,
            )
            for z in z_sample
        ]

        sqla_BackgroundModelFactory.store(
            obj, self.conn, self.table, self.inserter, self.tables, self.inserters
        )
        sqla_BackgroundModelFactory.validate(obj, self.conn, self.table, self.tables)

        return obj

    def read_model(self, z_sample: Optional[redshift_array], tags=None):
        """The production build(), on whichever path ``z_sample`` selects."""
        payload = {
            "solver_labels": {},
            "cosmology": self.cosmology,
            "z_sample": z_sample,
            "tags": list(tags) if tags is not None else [],
        }
        return sqla_BackgroundModelFactory.build(
            payload, self.conn, self.table, self.inserter, self.tables, self.inserters
        )

    def rows(self):
        return list(
            self.conn.execute(
                sqla.select(
                    self.table.c.serial,
                    self.table.c.source_grid_digest,
                    self.table.c.source_grid_construction,
                )
            )
        )


# ---------------------------------------------------------------------------------------------
# ast helpers
# ---------------------------------------------------------------------------------------------


def _factory_function(name: str) -> ast.FunctionDef:
    """
    One method of sqla_BackgroundModelFactory. The module also carries the tag-association and
    per-value factories, each with a build() of its own, so the search is scoped to the class.
    """
    tree = ast.parse(FACTORY_SOURCE.read_text(), filename=str(FACTORY_SOURCE))
    classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "sqla_BackgroundModelFactory"
    ]
    assert len(classes) == 1, "expected exactly one sqla_BackgroundModelFactory"

    functions = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(functions) == 1, f"expected exactly one {name}() in the factory"
    return functions[0]


def _is_column(node: ast.AST, column: str) -> bool:
    """``table.c.<column>``"""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == column
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "c"
    )


def _comparisons_on(function: ast.FunctionDef, column: str) -> List[ast.Compare]:
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Compare) and _is_column(node.left, column)
    ]


def _dict_values_for(function: ast.FunctionDef, key: str) -> List[ast.AST]:
    return [
        value
        for node in ast.walk(function)
        if isinstance(node, ast.Dict)
        for k, value in zip(node.keys, node.values)
        if isinstance(k, ast.Constant) and k.value == key
    ]


def _main_py_tree() -> ast.Module:
    return ast.parse(MAIN_PY.read_text(), filename=str(MAIN_PY))


def _tag_lists(tree: ast.Module) -> List[ast.List]:
    """
    Every literal list in main.py that names TkProductionTag or GkProductionTag: that is the
    campaign's existing marker for "a list of store_tags identifying a stored object", so it is
    the right set to require the run's identity of.
    """
    lists = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        names = {e.id for e in node.elts if isinstance(e, ast.Name)}
        if "TkProductionTag" in names or "GkProductionTag" in names:
            lists.append(node)
    return lists


# ---------------------------------------------------------------------------------------------


class TestTheConstructionVersionDeclaration(unittest.TestCase):
    """Prompt 14 §2 item 2: one declaration, naming the algorithm rather than the grid."""

    def test_it_is_an_integer_identifier(self):
        self.assertIsInstance(SOURCE_GRID_CONSTRUCTION_VERSION, int)
        self.assertNotIsInstance(SOURCE_GRID_CONSTRUCTION_VERSION, bool)
        self.assertGreaterEqual(SOURCE_GRID_CONSTRUCTION_VERSION, 1)

    def test_it_is_declared_exactly_once(self):
        tree = ast.parse(WAVENUMBER_SOURCE.read_text(), filename=str(WAVENUMBER_SOURCE))
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == CONSTANT_NAME
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        self.assertIsInstance(assignments[0].value, ast.Constant)

    def test_the_factory_imports_it_rather_than_declaring_its_own(self):
        tree = ast.parse(FACTORY_SOURCE.read_text(), filename=str(FACTORY_SOURCE))

        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and any(alias.name == CONSTANT_NAME for alias in node.names)
        ]
        self.assertEqual(len(imports), 1)
        self.assertEqual(imports[0].module, "CosmologyConcepts.wavenumber")

        # ... and nowhere in the factory is it re-declared
        self.assertEqual(
            [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == CONSTANT_NAME
                    for target in node.targets
                )
            ],
            [],
        )

    def test_the_column_types_are_identifiers_not_measurements(self):
        columns = {c.name: c for c in sqla_BackgroundModelFactory.register()["columns"]}

        self.assertIn(DIGEST_COLUMN, columns)
        self.assertIn(CONSTRUCTION_COLUMN, columns)

        self.assertIsInstance(columns[DIGEST_COLUMN].type, sqla.String)
        self.assertIsInstance(columns[CONSTRUCTION_COLUMN].type, sqla.Integer)

        # neither is a measured quantity, so neither may be a float: DEFAULT_FLOAT_PRECISION has
        # no business anywhere near a key that is compared for equality
        self.assertNotIsInstance(columns[DIGEST_COLUMN].type, sqla.Float)
        self.assertNotIsInstance(columns[CONSTRUCTION_COLUMN].type, sqla.Float)

        self.assertFalse(columns[DIGEST_COLUMN].nullable)
        self.assertFalse(columns[CONSTRUCTION_COLUMN].nullable)


class TestTheFilterReadsTheConstants(unittest.TestCase):
    """
    Prompt 14 §3 item 3, following prompt 03's test of the same name. A literal here would
    silently stop tracking the constant the moment someone bumps it, which is the exact failure
    the key exists to prevent.
    """

    def test_the_construction_filter_is_not_a_literal(self):
        build = _factory_function("build")
        comparisons = _comparisons_on(build, CONSTRUCTION_COLUMN)
        self.assertEqual(len(comparisons), 1)

        operand = comparisons[0].comparators[0]
        self.assertNotIsInstance(operand, ast.Constant)
        self.assertIsInstance(operand, ast.Name)

        bindings = [
            node
            for node in ast.walk(build)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == operand.id
                for target in node.targets
            )
        ]
        self.assertEqual(len(bindings), 1)
        self.assertIsInstance(bindings[0].value, ast.Name)
        self.assertEqual(bindings[0].value.id, CONSTANT_NAME)

    def test_the_digest_filter_is_not_a_literal(self):
        build = _factory_function("build")
        comparisons = _comparisons_on(build, DIGEST_COLUMN)
        self.assertEqual(len(comparisons), 1)

        operand = comparisons[0].comparators[0]
        self.assertNotIsInstance(operand, ast.Constant)
        self.assertIsInstance(operand, ast.Name)

        bindings = [
            node
            for node in ast.walk(build)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == operand.id
                for target in node.targets
            )
        ]
        self.assertEqual(len(bindings), 1)
        # the digest is computed from the grid itself, never handed in and never written out
        self.assertIn("digest", ast.dump(bindings[0].value))

    def test_the_insert_writes_what_the_filter_reads(self):
        store = _factory_function("store")

        construction = _dict_values_for(store, CONSTRUCTION_COLUMN)
        self.assertEqual(len(construction), 1)
        self.assertIsInstance(construction[0], ast.Name)
        self.assertEqual(construction[0].id, CONSTANT_NAME)

        digest = _dict_values_for(store, DIGEST_COLUMN)
        self.assertEqual(len(digest), 1)
        self.assertIsInstance(digest[0], ast.Call)
        self.assertEqual(digest[0].func.attr, "digest")


class TestTwoGenerationsDoNotCollide(unittest.TestCase):
    """
    Prompt 14 §3 item 2 and §4 row 2, against a real database. The whole point of the prompt is
    what SQL does with a candidate row, so this runs the production store() and build() against
    SQLite rather than reasoning about the compiled query.
    """

    def setUp(self):
        self.db = _Schema()
        self.addCleanup(self.db.close)

        self.grid = self.db.z_sample([100.0, 10.0, 1.0])
        self.other_grid = self.db.z_sample([100.0, 50.0, 10.0, 1.0])

    def test_the_same_grid_and_generation_reuse_one_row(self):
        written = self.db.write_model(self.grid)
        read = self.db.read_model(self.grid)

        self.assertTrue(read.available)
        self.assertEqual(read.store_id, written.store_id)
        self.assertEqual(len(self.db.rows()), 1)
        self.assertEqual(
            read._source_grid_identity,
            (SOURCE_GRID_CONSTRUCTION_VERSION, self.grid.digest()),
        )

    def test_a_different_construction_version_misses(self):
        self.db.write_model(self.grid)

        original = background_model_factory_module.SOURCE_GRID_CONSTRUCTION_VERSION
        background_model_factory_module.SOURCE_GRID_CONSTRUCTION_VERSION = original + 1
        try:
            # the same parameters and the same grid, built by a different algorithm: a different
            # background, which must not be served from the old row
            missed = self.db.read_model(self.grid)
            self.assertFalse(missed.available)

            second = self.db.write_model(self.grid)
        finally:
            background_model_factory_module.SOURCE_GRID_CONSTRUCTION_VERSION = original

        rows = self.db.rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row.source_grid_construction for row in rows}, {original, original + 1}
        )

        # ... and, back at the original version, the original row is the one that is served
        served = self.db.read_model(self.grid)
        self.assertNotEqual(served.store_id, second.store_id)

    def test_a_different_grid_misses(self):
        """
        The defect itself: before prompt 14 the endpoints and samples-per-decade tags were
        unchanged when the grid's shape changed, so this lookup hit.
        """
        self.db.write_model(self.grid)

        missed = self.db.read_model(self.other_grid)
        self.assertFalse(missed.available)

        self.db.write_model(self.other_grid)
        digests = {row.source_grid_digest for row in self.db.rows()}
        self.assertEqual(digests, {self.grid.digest(), self.other_grid.digest()})


class TestARunRoundTripsByName(unittest.TestCase):
    """Prompt 14 §3 item 4 and §4 row 3."""

    def setUp(self):
        self.db = _Schema()
        self.addCleanup(self.db.close)

        self.march = self.db.tag(run_label_tag("march"))
        self.september = self.db.tag(run_label_tag("september"))
        self.march_grid = self.db.z_sample([100.0, 10.0, 1.0])
        self.september_grid = self.db.z_sample([100.0, 50.0, 10.0, 1.0])

    def test_a_named_run_reads_back_its_own_objects(self):
        march = self.db.write_model(
            self.march_grid, tags=[self.march], label="march-model"
        )
        september = self.db.write_model(
            self.september_grid, tags=[self.september], label="september-model"
        )
        self.assertNotEqual(march.store_id, september.store_id)

        # the read path: no z_sample, because the reader is asking which grid was used rather
        # than asserting one
        recovered = self.db.read_model(None, tags=[self.march])
        self.assertTrue(recovered.available)
        self.assertEqual(recovered.store_id, march.store_id)
        self.assertEqual(recovered.label, "march-model")
        self.assertEqual(
            recovered._source_grid_identity,
            (SOURCE_GRID_CONSTRUCTION_VERSION, self.march_grid.digest()),
        )
        self.assertEqual(len(recovered.z_sample), len(self.march_grid))

        other = self.db.read_model(None, tags=[self.september])
        self.assertEqual(other.store_id, september.store_id)
        self.assertEqual(
            other._source_grid_identity,
            (SOURCE_GRID_CONSTRUCTION_VERSION, self.september_grid.digest()),
        )

    def test_an_untagged_read_of_a_two_run_store_is_refused(self):
        """
        ... and this is why the scripts had to stop passing zero tags: with two runs in the store
        the tagless query matches both.
        """
        self.db.write_model(self.march_grid, tags=[self.march])
        self.db.write_model(self.september_grid, tags=[self.september])

        with self.assertRaises(RuntimeError):
            self.db.read_model(None)


class TestALabelSpanningTwoConfigurationsIsRefused(unittest.TestCase):
    """
    Prompt 14 §3 item 5 -- "the check that makes naming safe and the single most important test in
    the prompt". A label is chosen by a human and nothing stops it being reused across a changed
    configuration; what must not happen is a silent mixture.
    """

    def setUp(self):
        self.db = _Schema()
        self.addCleanup(self.db.close)

        self.label = self.db.tag(run_label_tag("production"))
        self.first = self.db.z_sample([100.0, 10.0, 1.0])
        self.second = self.db.z_sample([100.0, 50.0, 10.0, 1.0])

    def test_it_refuses_and_names_both_generations(self):
        self.db.write_model(self.first, tags=[self.label], label="first")
        self.db.write_model(self.second, tags=[self.label], label="second")

        with self.assertRaises(RuntimeError) as caught:
            self.db.read_model(None, tags=[self.label])

        message = str(caught.exception)
        self.assertIn(self.first.digest(), message)
        self.assertIn(self.second.digest(), message)
        self.assertIn(str(SOURCE_GRID_CONSTRUCTION_VERSION), message)
        self.assertIn("--run-label", message)

    def test_two_constructions_under_one_label_are_also_refused(self):
        self.db.write_model(self.first, tags=[self.label], label="first")

        original = background_model_factory_module.SOURCE_GRID_CONSTRUCTION_VERSION
        background_model_factory_module.SOURCE_GRID_CONSTRUCTION_VERSION = original + 1
        try:
            self.db.write_model(self.second, tags=[self.label], label="second")
        finally:
            background_model_factory_module.SOURCE_GRID_CONSTRUCTION_VERSION = original

        with self.assertRaises(RuntimeError) as caught:
            self.db.read_model(None, tags=[self.label])

        message = str(caught.exception)
        self.assertIn(f"construction version {original}", message)
        self.assertIn(f"construction version {original + 1}", message)


class TestAStoreWithNoGridIdentity(unittest.TestCase):
    """
    Prompt 14 §2 item 4 and §3 item 6: the two paths give **different** answers to the same old
    datastore, and that is deliberate. The compute path cannot write into it, because there is no
    defensible grid identity to assume for a row that records none; the read path must keep
    reading it, because a superseded datastore keeps its archival value long after it can no
    longer serve as a numerical base, and extract_*.py is how that value is realised.
    """

    def setUp(self):
        # the pre-prompt-14 schema: the production registration with the two columns dropped
        self.db = _Schema(omit_grid_identity=True)
        self.addCleanup(self.db.close)

        self.grid = self.db.z_sample([100.0, 10.0, 1.0])

        # a row written the way a pre-prompt-14 run wrote one. store() now writes the two columns,
        # so the row is inserted directly here -- there is no other way to produce the old shape.
        self.serial = self.db.conn.execute(
            sqla.insert(self.db.table),
            [
                {
                    "serial": 1,
                    "label": "archived",
                    "cosmology_type": self.db.cosmology.type_id,
                    "cosmology_serial": self.db.cosmology.store_id,
                    # the accuracy half of the key as the current schema spells it: prompt 05 of
                    # prompts/tolerance-convergence replaced the tolerance pair with the three
                    # Gauss orders, and _Schema builds this table from the factory's own
                    # register(), so the archival row has to carry them
                    "tau_gauss_order": BackgroundModelModule.TAU_GAUSS_ORDER,
                    "cs_tau_gauss_order": BackgroundModelModule.CS_TAU_GAUSS_ORDER,
                    "friction_F_gauss_order": BackgroundModelModule.FRICTION_F_GAUSS_ORDER,
                    "solver_serial": 1,
                    "z_init_serial": self.grid.min.store_id,
                    "z_samples": len(self.grid),
                    "compute_time": 1.0,
                    "compute_steps": 1,
                    "RHS_evaluations": 1,
                    "mean_RHS_time": 1.0,
                    "max_RHS_time": 1.0,
                    "min_RHS_time": 1.0,
                    "validated": True,
                    "timestamp": datetime.now(),
                }
            ],
        )
        for z in self.grid:
            self.db.conn.execute(
                sqla.insert(self.db.tables["BackgroundModelValue"]),
                [
                    {
                        "model_serial": 1,
                        "z_serial": z.store_id,
                        "Hubble_GeV": 1.0,
                        "wBackground": 1.0 / 3.0,
                        "wPerturbations": 1.0 / 3.0,
                        "rho_GeV": 1.0,
                        "tau_Mpc": 1.0,
                        "tau_lo_Mpc": 0.0,
                        "cs_tau_Mpc": 1.0,
                        "cs_tau_lo_Mpc": 0.0,
                        "friction_F": 1.0,
                        "T_photon_GeV": 1.0,
                        "T_photon_Kelvin": 1.0,
                        "d_lnH_dz": 1.0,
                        "d2_lnH_dz2": 1.0,
                        "d3_lnH_dz3": 1.0,
                        "d_wPerturbations_dz": 1.0,
                        "d2_wPerturbations_dz2": 1.0,
                    }
                ],
            )

        # what Datastore._build_schema builds from the *code*: _ensure_tables() creates absent
        # tables but never alters an existing one, which is why the two can differ
        self.current = _append_table(
            "BackgroundModel", sqla_BackgroundModelFactory, sqla.MetaData()
        )

    def _read(self, z_sample):
        payload = {
            "solver_labels": {},
            "cosmology": self.db.cosmology,
            "z_sample": z_sample,
            "tags": [],
        }
        return sqla_BackgroundModelFactory.build(
            payload,
            self.db.conn,
            self.current,
            self.db.inserter,
            self.db.tables,
            self.db.inserters,
        )

    def test_the_read_path_still_reads_it_and_labels_it_unknown(self):
        model = self._read(None)

        self.assertTrue(model.available)
        self.assertEqual(model.label, "archived")
        self.assertEqual(len(model.z_sample), len(self.grid))
        self.assertIsNone(model._source_grid_identity)
        self.assertIn("unknown", describe_background_generation(model))
        self.assertIn("prompt 14", describe_background_generation(model))

    def test_the_compute_path_demands_regeneration(self):
        with self.assertRaises(RuntimeError) as caught:
            self._read(self.grid)

        message = str(caught.exception)
        self.assertIn(DIGEST_COLUMN, message)
        self.assertIn("prompt 14", message)
        self.assertIn("regenerated", message)
        self.assertIn("no migration", message)
        self.assertIsInstance(caught.exception.__cause__, OperationalError)

    def test_an_unrelated_database_error_is_not_disguised(self):
        class _RaisingConnection:
            def execute(self, query):
                raise OperationalError(
                    "SELECT ...", {}, Exception("database is locked")
                )

        payload = {
            "solver_labels": {},
            "cosmology": self.db.cosmology,
            "z_sample": self.grid,
            "tags": [],
        }
        with self.assertRaises(OperationalError):
            sqla_BackgroundModelFactory.build(
                payload,
                _RaisingConnection(),
                self.current,
                self.db.inserter,
                self.db.tables,
                self.db.inserters,
            )


class TestChoosingARun(unittest.TestCase):
    """
    Prompt 14 §2 item 4, third and fourth bullets, and §3 items 6 and 7. The property being
    preserved is the one that made the tagless design pleasant: a store with one run needs no
    argument.
    """

    def test_a_store_with_exactly_one_run_needs_no_label_and_says_which(self):
        selection = choose_run_label(["march"], None)

        self.assertEqual(selection.label, "march")
        self.assertIn("march", selection.description)
        self.assertIn("only run", selection.description)

    def test_a_store_with_no_runs_reads_as_unknown(self):
        selection = choose_run_label([], None)

        self.assertIsNone(selection.label)
        self.assertEqual(selection.tags, [])
        self.assertIn("unknown", selection.description)

    def test_an_ambiguous_store_is_refused_naming_the_candidates(self):
        with self.assertRaises(RuntimeError) as caught:
            choose_run_label(["march", "september"], None)

        message = str(caught.exception)
        self.assertIn("march", message)
        self.assertIn("september", message)
        self.assertIn("--run-label", message)

    def test_a_label_that_is_not_there_is_refused_naming_what_is(self):
        with self.assertRaises(RuntimeError) as caught:
            choose_run_label(["march"], "september")

        message = str(caught.exception)
        self.assertIn("september", message)
        self.assertIn("march", message)

    def test_a_requested_label_is_used_and_said_to_be_the_users_choice(self):
        selection = choose_run_label(["march", "september"], "september")

        self.assertEqual(selection.label, "september")
        self.assertIn("--run-label", selection.description)

    def test_the_tag_spelling_round_trips(self):
        self.assertTrue(run_label_tag("march").startswith(RUN_LABEL_TAG_PREFIX))
        self.assertEqual(run_label_tag("march")[len(RUN_LABEL_TAG_PREFIX) :], "march")
        self.assertIn(
            str(SOURCE_GRID_CONSTRUCTION_VERSION),
            source_grid_construction_tag(SOURCE_GRID_CONSTRUCTION_VERSION),
        )


class TestMainPyNamesItsRun(unittest.TestCase):
    """
    Prompt 14 §3 item 8. main.py cannot be imported (CLAUDE.md), so this is read with ast -- which
    is also the only way to assert something about *every* tag list rather than about the handful
    a run would exercise.
    """

    def setUp(self):
        self.tree = _main_py_tree()
        self.source = MAIN_PY.read_text()

    def test_it_parses_the_new_argument(self):
        arguments = [
            node.args[0].value
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and len(node.args) > 0
            and isinstance(node.args[0], ast.Constant)
        ]
        self.assertIn("--run-label", arguments)

    def test_every_tag_list_carries_the_run_and_the_construction(self):
        tag_lists = _tag_lists(self.tree)

        # 26 lists name a production tag at this commit; a change that adds a stored object
        # without tagging it with the run fails here rather than silently producing an object no
        # reader can find. The bound is loose on purpose -- the assertion below is the substance,
        # and this only guards against the search matching nothing
        self.assertGreaterEqual(len(tag_lists), 20)

        for node in tag_lists:
            names = {e.id for e in node.elts if isinstance(e, ast.Name)}
            self.assertIn("RunLabelTag", names, f"line {node.lineno}")
            self.assertIn("SourceGridConstructionTag", names, f"line {node.lineno}")

    def test_the_background_model_lookup_carries_the_run(self):
        # main.py names the class by string, so the lookup is found by its first argument
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and len(node.args) > 0
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "BackgroundModel"
        ]
        self.assertEqual(len(calls), 1)

        tags = [kw.value for kw in calls[0].keywords if kw.arg == "tags"]
        self.assertEqual(len(tags), 1)
        names = {e.id for e in tags[0].elts if isinstance(e, ast.Name)}
        self.assertIn("RunLabelTag", names)
        self.assertIn("SourceGridConstructionTag", names)

    def test_the_tag_labels_are_built_from_the_shared_helpers(self):
        # not spelled out as f-strings here and again in extract_common: the writer and the seven
        # readers agree on the spelling by construction
        self.assertIn("run_label_tag(run_label)", self.source)
        self.assertIn(
            "source_grid_construction_tag(SOURCE_GRID_CONSTRUCTION_VERSION)",
            self.source,
        )
        self.assertNotIn(f'"{RUN_LABEL_TAG_PREFIX}', self.source)


class TestEveryExtractScriptSelectsARun(unittest.TestCase):
    """
    Prompt 14 §2 item 4 and §3 item 9. These scripts open a Ray connection and a ShardedPool at
    module scope exactly as main.py does, so they cannot be imported either, and none of them can
    be *run* without a populated datastore -- see log 14 for what was and was not executed.
    """

    def _source(self, name: str) -> str:
        return (REPO_ROOT / name).read_text()

    def test_each_script_adds_the_argument_and_resolves_a_run(self):
        for name in EXTRACT_SCRIPTS:
            with self.subTest(script=name):
                source = self._source(name)
                self.assertIn("add_run_selection_argument(parser)", source)
                self.assertIn(
                    "resolve_run_selection(pool, args.run_label)",
                    source,
                )
                self.assertIn("run_selection.tags", source)

    def test_each_script_selects_the_background_model_on_the_run(self):
        for name in EXTRACT_SCRIPTS:
            with self.subTest(script=name):
                tree = ast.parse(self._source(name), filename=name)
                calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and len(node.args) > 0
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == "BackgroundModel"
                ]
                self.assertEqual(len(calls), 1)

                tags = [kw.value for kw in calls[0].keywords if kw.arg == "tags"]
                self.assertEqual(len(tags), 1)
                self.assertEqual(ast.unparse(tags[0]), "run_selection.tags")

    def test_no_script_reproduces_the_writers_tag_configuration(self):
        """
        Prompt 14 §2 item 4, last bullet: the scripts are *readers*. Their requirement is
        selection and disambiguation, not a transcription of main.py's tag lists, which would
        have to be kept in step with it forever.
        """
        for name in EXTRACT_SCRIPTS:
            with self.subTest(script=name):
                source = self._source(name)
                for writer_tag in (
                    "SourceZGridSizeTag",
                    "ResponseZGridSizeTag",
                    "SourceSamplesPerLog10ZTag",
                    "OutsideHorizonEfoldsTag",
                    "TkOneLoopDensity",
                    "GkOneLoopDensity",
                ):
                    self.assertNotIn(writer_tag, source)

                # ... and no script spells the run tag prefix itself
                self.assertNotIn(f'"{RUN_LABEL_TAG_PREFIX}', source)


if __name__ == "__main__":
    unittest.main()
