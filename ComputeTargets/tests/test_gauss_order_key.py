"""
Tests for prompt 05 of ``prompts/tolerance-convergence``: the Gauss orders are the datastore
lookup key of the four object types whose ``atol``/``rtol`` pair reached no solver.

Four object types carried ``atol_serial`` and ``rtol_serial`` as part of their lookup key while
the pair reached nothing: ``BackgroundModel``'s three primitives are Gauss-Legendre cumulative
tables, and both WKB sectors' phases come from those tables plus a Gauss-Legendre residual table,
so what actually sets their accuracy is an *integer order*. Those orders -- ``TAU_GAUSS_ORDER``,
``CS_TAU_GAUSS_ORDER``, ``FRICTION_F_GAUSS_ORDER`` and ``RHO_GAUSS_ORDER`` -- were module
constants in no column at all. Raise one and re-run, and the key did not move: the pipeline found
the row it had and served it, with the truth recorded only in a ``solver_label`` that is joined
and never compared (``docs/tolerance-convergence/TOLERANCE-INVENTORY.md`` §2.4).
``GkSource`` is the fourth case and a different one: it assembles and integrates nothing, so the
pair is dropped and nothing replaces it.

What is asserted here, in the order prompt 05 §7 states it:

1. **The schema is the one §2's table specifies** -- the pair gone from all four, three integer
   orders on ``BackgroundModel``, one on each WKB factory, nothing new on ``GkSource``.
2. **Each order is filtered on, not merely selected.** The ``build()`` query is compiled with no
   database behind it and its ``WHERE`` clause read back -- the technique
   ``test_numeric_break_point_key.py`` established for ``break_point_kind``, and for the same
   reason: a column a query *selects* and never *compares* is exactly the defect being repaired.
3. **When the constant moves, the key moves.** The single declaration is repointed and the
   compiled ``whereclause``, the value ``store()`` writes and the compute class's own accessor all
   follow it together. A test that only checked the column exists would not show this.
4. **A row at another order misses, and its own hits** -- what SQL does with a candidate row is
   the equality this test performs directly.
5. **No tolerance survives in ``GkSource``'s criteria.**

The mechanism behind 3, which is why there is no separate "the stored order is the computed order"
test to write: the order is reached through *one* name, resolved at call time, on every path.
``compute_background`` and ``WKB_phase_function`` read the module constant; the compute class's
accessor reads the same module constant; the factory's ``store()`` writes that accessor and its
``build()`` filters on the same module attribute. There is no keyword, no payload key and no
default anywhere on the path by which a caller could supply a different order, so the value
written into the key cannot differ from the order the computation used. The two tests below that
repoint the constant are the demonstration.

Nothing here needs Ray, a datastore or SQLite. The tables are built from the factories' own
``register()`` output in the shape ``Datastore.SQL.Datastore._build_schema`` builds them, and the
connection is a stand-in that captures the query instead of executing it.
"""

import ast
import unittest
from importlib import import_module
from pathlib import Path
from unittest import mock

import sqlalchemy as sqla

import ComputeTargets.phase_residual as phase_residual
from Datastore.SQL.ObjectFactories.BackgroundModel import sqla_BackgroundModelFactory
from Datastore.SQL.ObjectFactories.GkSource import sqla_GkSource_factory
from Datastore.SQL.ObjectFactories.GkWKBIntegration import (
    sqla_GkWKBIntegration_factory,
)
from Datastore.SQL.ObjectFactories.TkWKBIntegration import (
    sqla_TkWKBIntegration_factory,
)
from Units import Mpc_units

# the module, not the class: ComputeTargets/__init__.py rebinds the package attribute to the
# class, so the three background orders have to be reached the way the factory reaches them
background_model = import_module("ComputeTargets.BackgroundModel")

REPO_ROOT = Path(__file__).parents[2]

# the retired columns, which must appear nowhere in any of the four schemas or criteria
RETIRED_COLUMNS = ("atol_serial", "rtol_serial")

_BACKGROUND_ORDER_NAMES = (
    "TAU_GAUSS_ORDER",
    "CS_TAU_GAUSS_ORDER",
    "FRICTION_F_GAUSS_ORDER",
)


# ---------------------------------------------------------------------------------------------
# stand-ins
# ---------------------------------------------------------------------------------------------


class _Serial:
    """anything the factory reads only a store_id from"""

    def __init__(self, store_id: int):
        self.store_id = store_id


class _Cosmology:
    type_id = 4242
    store_id = 7


class _ZSample:
    """a redshift grid the factory reads only a digest and a minimum from"""

    min = _Serial(21)
    max = _Serial(22)

    def __len__(self):
        return 3

    def digest(self) -> str:
        return "deadbeef"


class _Data:
    compute_time = 1.0
    compute_steps = 2
    RHS_evaluations = 3
    mean_RHS_time = 4.0
    max_RHS_time = 5.0
    min_RHS_time = 6.0


class _QueryCaptured(Exception):
    """
    raised by the stand-in connection in place of executing. Deliberately *not* a
    SQLAlchemyError, so that it passes through each factory's missing-column handler untouched.
    """

    def __init__(self, query):
        super().__init__("query captured")
        self.query = query


class _CapturingConnection:
    def execute(self, query):
        raise _QueryCaptured(query)


def _build_table(name: str, factory, metadata: sqla.MetaData) -> sqla.Table:
    """
    Reproduce the table Datastore._build_schema builds for this factory, from the factory's own
    register() output -- so that the schema under test is the production schema, not a copy of it.
    """
    registration = factory.register()

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
        tab.append_column(column)

    return tab


def _support_tables(tag_table: str, metadata: sqla.MetaData) -> dict:
    """the other tables the four build() methods join against or name"""
    return {
        "IntegrationSolver": sqla.Table(
            "IntegrationSolver",
            metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True),
            sqla.Column("label", sqla.String(256)),
            sqla.Column("stepping", sqla.Integer),
        ),
        "redshift": sqla.Table(
            "redshift",
            metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True),
            sqla.Column("z", sqla.Float(64)),
            sqla.Column("source", sqla.Boolean),
            sqla.Column("response", sqla.Boolean),
        ),
        "tolerance": sqla.Table(
            "tolerance",
            metadata,
            sqla.Column("serial", sqla.Integer, primary_key=True),
            sqla.Column("log10_tol", sqla.Float(64)),
        ),
        tag_table: sqla.Table(
            tag_table,
            metadata,
            sqla.Column("model_serial", sqla.Integer),
            sqla.Column("wkb_serial", sqla.Integer),
            sqla.Column("parent_serial", sqla.Integer),
            sqla.Column("tag_serial", sqla.Integer),
        ),
    }


# (factory, table name, tag table, the build() payload) for each of the four targets
TARGETS = {
    "BackgroundModel": (
        sqla_BackgroundModelFactory,
        "BackgroundModel",
        "BackgroundModel_tags",
        {
            "solver_labels": {},
            "cosmology": _Cosmology(),
            "z_sample": _ZSample(),
            "tags": [],
        },
    ),
    "GkWKBIntegration": (
        sqla_GkWKBIntegration_factory,
        "GkWKBIntegration",
        "GkWKB_tags",
        {
            "solver_labels": {},
            "k": _Serial(13),
            "model": _Serial(14),
            "z_sample": None,
            "z_source": None,
            "tags": [],
        },
    ),
    "TkWKBIntegration": (
        sqla_TkWKBIntegration_factory,
        "TkWKBIntegration",
        "TkWKB_tags",
        {
            "solver_labels": {},
            "k": _Serial(13),
            "model": _Serial(14),
            "z_sample": None,
            "tags": [],
        },
    ),
    "GkSource": (
        sqla_GkSource_factory,
        "GkSource",
        "GkSource_tags",
        {
            "k": _Serial(13),
            "model": _Serial(14),
            "z_sample": None,
            "z_response": None,
            "tags": [],
        },
    ),
}

# target -> {column: the accessor on the compute object that supplies it}
ORDER_COLUMNS = {
    "BackgroundModel": {
        "tau_gauss_order": "tau_gauss_order",
        "cs_tau_gauss_order": "cs_tau_gauss_order",
        "friction_F_gauss_order": "friction_F_gauss_order",
    },
    "GkWKBIntegration": {"rho_gauss_order": "rho_gauss_order"},
    "TkWKBIntegration": {"rho_gauss_order": "rho_gauss_order"},
    "GkSource": {},
}

# column -> (the module that declares the constant, the constant's name)
DECLARATIONS = {
    "tau_gauss_order": (background_model, "TAU_GAUSS_ORDER"),
    "cs_tau_gauss_order": (background_model, "CS_TAU_GAUSS_ORDER"),
    "friction_F_gauss_order": (background_model, "FRICTION_F_GAUSS_ORDER"),
    "rho_gauss_order": (phase_residual, "RHO_GAUSS_ORDER"),
}


def _capture_build_query(target: str):
    """
    Run the production build() far enough to form its query, and hand the query back.

    The connection raises instead of executing, so nothing needs a database; the payload carries
    only what build() reads before the query is issued.
    """
    factory, table_name, tag_table, payload = TARGETS[target]

    metadata = sqla.MetaData()
    table = _build_table(table_name, factory, metadata)
    tables = _support_tables(tag_table, metadata)

    try:
        factory.build(payload, _CapturingConnection(), table, None, tables, None)
    except _QueryCaptured as e:
        return e.query

    raise AssertionError(f"{target}.build() did not issue a query")


def _equality_criteria(query) -> dict:
    """
    column name -> the value the WHERE clause requires it to equal, for the top-level equality
    criteria of a compiled select. This is the whole semantics of the lookup key: SQL will return
    a row when, and only when, every one of these holds of it.
    """
    criteria = {}
    for clause in query.whereclause.clauses:
        left = getattr(clause, "left", None)
        right = getattr(clause, "right", None)
        if left is None or not hasattr(left, "name"):
            continue
        if not hasattr(right, "value"):
            continue
        criteria[left.name] = right.value

    return criteria


# ---------------------------------------------------------------------------------------------
# store()-side stand-ins
# ---------------------------------------------------------------------------------------------


class _BackgroundStandIn:
    """what sqla_BackgroundModelFactory.store() reads off a BackgroundModel"""

    label = "stand-in"
    tags = []
    values = []

    def __init__(self):
        self.cosmology = _Cosmology()
        self.solver = _Serial(15)
        self.z_sample = _ZSample()
        self.data = _Data()
        self._units = Mpc_units()

    # the production accessors, reproduced by delegation rather than copied: each reads the
    # single module-level declaration at call time, which is the property under test
    @property
    def tau_gauss_order(self) -> int:
        return background_model.BackgroundModel.tau_gauss_order.fget(self)

    @property
    def cs_tau_gauss_order(self) -> int:
        return background_model.BackgroundModel.cs_tau_gauss_order.fget(self)

    @property
    def friction_F_gauss_order(self) -> int:
        return background_model.BackgroundModel.friction_F_gauss_order.fget(self)


class _WKBStandIn:
    """what the two WKB factories' store() read off an integration object"""

    label = "stand-in"
    tags = []
    values = []

    sin_coeff = 1.0
    cos_coeff = 1.0
    z_init = 1.0
    G_init = 0.0
    Gprime_init = 1.0
    T_init = 1.0
    Tprime_init = 0.0
    has_WKB_violation = False
    WKB_violation_z = None
    WKB_violation_efolds_subh = None
    metadata = None

    def __init__(self, cls):
        self._cls = cls
        self._k_exit = _Serial(13)
        self.model_proxy = _Serial(14)
        self.solver = _Serial(15)
        self.phase_solver = _Serial(15)
        self.friction_solver = _Serial(16)
        self.z_source = _Serial(17)
        self.z_sample = _ZSample()
        self.stage_1_data = _Data()
        self.stage_2_data = _Data()
        self.friction_data = _Data()
        self._init_efolds_subh = 1.0
        self._init_efolds_suph = 1.0
        self._metadata = None

    @property
    def rho_gauss_order(self) -> int:
        return self._cls.rho_gauss_order.fget(self)


class _GkSourceStandIn:
    """what sqla_GkSource_factory.store() reads off a GkSource -- no order and no tolerance"""

    label = "stand-in"
    tags = []
    values = []

    numeric_smallest_z = None
    primary_WKB_largest_z = None

    def __init__(self):
        self._k_exit = _Serial(13)
        self.model_proxy = _Serial(14)
        self.z_response = _Serial(17)
        self.z_sample = _ZSample()
        self._metadata = None


def _class_accessor(target: str, accessor: str) -> int:
    """The compute class's own order accessor, called on the class with no instance: the getter
    reads the module constant and nothing else."""
    module = import_module(f"ComputeTargets.{target}")
    cls = getattr(module, target)
    return getattr(cls, accessor).fget(cls)


def _store_payload(target: str) -> dict:
    """run the production store() with a capturing inserter, and hand back what it inserted"""
    factory, table_name, tag_table, _ = TARGETS[target]

    if target == "BackgroundModel":
        obj = _BackgroundStandIn()
        value_table = "BackgroundModelValue"
    elif target == "GkSource":
        obj = _GkSourceStandIn()
        value_table = "GkSourceValue"
    else:
        module = import_module(f"ComputeTargets.{target}")
        obj = _WKBStandIn(getattr(module, target))
        value_table = f"{target[:2]}WKBValue"

    captured = {}

    def inserter(conn, data):
        captured.update(data)
        return 99

    factory.store(
        obj,
        None,
        None,
        inserter,
        None,
        {tag_table: None, value_table: None},
    )

    return captured


# ---------------------------------------------------------------------------------------------


class TestTheSchemaIsTheOneD3Settled(unittest.TestCase):
    """Prompt 05 §2 and §7 item 1."""

    def test_the_vestigial_pair_is_gone_from_all_four(self):
        for target, (factory, _, _, _) in TARGETS.items():
            with self.subTest(target=target):
                names = {c.name for c in factory.register()["columns"]}
                for column in RETIRED_COLUMNS:
                    self.assertNotIn(column, names)

    def test_each_target_gains_exactly_the_orders_it_should(self):
        for target, expected in ORDER_COLUMNS.items():
            with self.subTest(target=target):
                factory = TARGETS[target][0]
                columns = {c.name: c for c in factory.register()["columns"]}

                for name in expected:
                    self.assertIn(name, columns)
                    column = columns[name]
                    self.assertFalse(column.nullable)
                    self.assertIsInstance(column.type, sqla.Integer)

                # ... and nothing else that looks like an order
                found = {name for name in columns if name.endswith("gauss_order")}
                self.assertEqual(found, set(expected))

    def test_gk_source_gains_nothing(self):
        """It assembles and integrates nothing, so there is no order to record."""
        self.assertEqual(ORDER_COLUMNS["GkSource"], {})

        columns = {c.name for c in sqla_GkSource_factory.register()["columns"]}
        self.assertEqual({name for name in columns if "gauss" in name}, set())
        self.assertEqual({name for name in columns if "tol" in name}, set())


class TestTheOrderIsFilteredOn(unittest.TestCase):
    """Prompt 05 §3 and §7 item 2: filtered on, not merely selected."""

    def test_every_order_appears_as_an_equality_criterion(self):
        for target, expected in ORDER_COLUMNS.items():
            criteria = _equality_criteria(_capture_build_query(target))
            for column in expected:
                with self.subTest(target=target, column=column):
                    self.assertIn(column, criteria)

                    module, constant = DECLARATIONS[column]
                    self.assertEqual(criteria[column], getattr(module, constant))

    def test_no_tolerance_survives_in_any_of_the_four_criteria(self):
        for target in TARGETS:
            criteria = _equality_criteria(_capture_build_query(target))
            with self.subTest(target=target):
                for column in RETIRED_COLUMNS:
                    self.assertNotIn(column, criteria)

    def test_the_orders_reach_the_compiled_sql(self):
        for target, expected in ORDER_COLUMNS.items():
            if len(expected) == 0:
                continue
            compiled = str(
                _capture_build_query(target).whereclause.compile(
                    compile_kwargs={"literal_binds": True}
                )
            )
            for column in expected:
                with self.subTest(target=target, column=column):
                    self.assertIn(column, compiled)

    def test_the_key_that_was_already_there_is_not_disturbed(self):
        """The orders stand *alongside* the rest of each key, not instead of it."""
        expected = {
            "BackgroundModel": ("cosmology_type", "cosmology_serial"),
            "GkWKBIntegration": ("wavenumber_exit_serial", "model_serial"),
            "TkWKBIntegration": ("wavenumber_exit_serial", "model_serial"),
            "GkSource": ("wavenumber_exit_serial", "model_serial"),
        }
        for target, columns in expected.items():
            criteria = _equality_criteria(_capture_build_query(target))
            for column in columns:
                with self.subTest(target=target, column=column):
                    self.assertIn(column, criteria)


class TestMovingTheConstantMovesTheKey(unittest.TestCase):
    """
    Prompt 05 §3 and §7 items 3 and 4 -- the property the whole prompt exists to deliver, and the
    one a "does the column exist?" test would miss entirely.
    """

    def test_repointing_the_declaration_moves_the_query_and_the_stored_value_together(
        self,
    ):
        for target, expected in ORDER_COLUMNS.items():
            for column, accessor in expected.items():
                module, constant = DECLARATIONS[column]
                sentinel = 97
                self.assertNotEqual(getattr(module, constant), sentinel)

                with self.subTest(target=target, column=column):
                    with mock.patch.object(module, constant, sentinel):
                        # the lookup asks for the new order ...
                        self.assertEqual(
                            _equality_criteria(_capture_build_query(target))[column],
                            sentinel,
                        )
                        # ... store() writes the new order ...
                        self.assertEqual(_store_payload(target)[column], sentinel)
                        # ... and so does the compute class's own accessor, which is what
                        # store() reads and what the computation is performed at
                        self.assertEqual(_class_accessor(target, accessor), sentinel)

                    # and the tree is back where it started
                    self.assertEqual(
                        _equality_criteria(_capture_build_query(target))[column],
                        getattr(module, constant),
                    )

    def test_store_writes_no_tolerance_for_any_of_the_four(self):
        for target in TARGETS:
            written = _store_payload(target)
            with self.subTest(target=target):
                for column in RETIRED_COLUMNS:
                    self.assertNotIn(column, written)

    def test_store_writes_the_order_the_query_asks_for(self):
        for target, expected in ORDER_COLUMNS.items():
            written = _store_payload(target)
            criteria = _equality_criteria(_capture_build_query(target))
            for column in expected:
                with self.subTest(target=target, column=column):
                    self.assertEqual(written[column], criteria[column])

    def test_a_row_at_another_order_misses_and_its_own_hits(self):
        """
        What SQL does with a candidate row is exactly this equality: a row carrying the order the
        query asks for is returned, a row carrying any other order -- including the order the
        store was built at before someone moved the constant -- is not.
        """
        for target, expected in ORDER_COLUMNS.items():
            criteria = _equality_criteria(_capture_build_query(target))
            for column in expected:
                required = criteria[column]
                with self.subTest(target=target, column=column):
                    self.assertTrue(required == required)
                    for other in (2, 6, 8, 12, 16):
                        if other == required:
                            continue
                        self.assertFalse(other == required)

    def test_no_order_is_written_as_a_literal(self):
        """
        Every order written or queried is read from its declaring module at call time. If any
        site had inlined the number instead, the patch above would have moved the others and left
        that one at 4, which is what these four assertions would catch.
        """
        for target, expected in ORDER_COLUMNS.items():
            for column in expected:
                module, constant = DECLARATIONS[column]
                with self.subTest(target=target, column=column):
                    with mock.patch.object(module, constant, 97):
                        self.assertNotEqual(_store_payload(target)[column], 4)
                        self.assertNotEqual(
                            _equality_criteria(_capture_build_query(target))[column], 4
                        )


class TestTheComputationReadsTheSameDeclaration(unittest.TestCase):
    """
    The other half of prompt 05 §3's second invariant, read with ``ast`` because neither module
    can be imported and driven without Ray and the point is the argument, not the call.

    ``store()`` and ``build()`` are shown above to follow the declaration. These assert that the
    *computation* does too, so that the three cannot drift: the tables are built at the order the
    key records because every site reads one name.
    """

    def test_compute_background_builds_each_table_at_its_own_declared_order(self):
        source = (REPO_ROOT / "ComputeTargets" / "BackgroundModel.py").read_text()
        tree = ast.parse(source, filename="BackgroundModel.py")

        orders = [
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in _BACKGROUND_ORDER_NAMES
        ]
        for name in _BACKGROUND_ORDER_NAMES:
            with self.subTest(order=name):
                # the declaration, the class accessor, and at least one CumulativeTable site
                self.assertGreaterEqual(orders.count(name), 3, msg=f"{name}: {orders}")

    def test_no_production_caller_overrides_the_residual_order(self):
        """
        ``build_phase_residual`` and ``cached_phase_residual`` take ``order`` with
        ``RHO_GAUSS_ORDER`` as its default, and a default argument is bound once at ``def`` time.
        That is safe only while no caller supplies the keyword: the moment one does, the table is
        built at an order the key does not record. No production caller does, and this is what
        says so.
        """
        offenders = []
        for path in REPO_ROOT.rglob("*.py"):
            relative = path.relative_to(REPO_ROOT)
            if relative.parts[0] in ("docs", "venv", "prompts"):
                continue
            if relative.parts[-1] == "phase_residual.py":
                continue
            if "tests" in relative.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if name not in ("build_phase_residual", "cached_phase_residual"):
                    continue
                if any(kw.arg == "order" for kw in node.keywords) or len(node.args) > 4:
                    offenders.append(f"{relative}:{node.lineno}")

        self.assertEqual(offenders, [])

    def test_the_phase_integrator_records_the_declared_order(self):
        source = (
            REPO_ROOT / "Quadrature" / "integrators" / "WKB_phase_function.py"
        ).read_text()
        tree = ast.parse(source, filename="WKB_phase_function.py")

        recorded = [
            value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "N_rho"
        ]
        self.assertEqual(len(recorded), 1)
        self.assertIsInstance(recorded[0], ast.Name)
        self.assertEqual(recorded[0].id, "RHO_GAUSS_ORDER")


class TestTheOrdersAreStillFour(unittest.TestCase):
    """
    Prompt 05 §4 and §7 item 5: this prompt records the orders in a key, it does not revisit them.
    Prompt 04 measured every one of them ``unchanged`` at 4
    (``docs/tolerance-convergence/ORDER-AUDIT.md``) and prompt 04b landed that evidence.
    """

    def test_all_four_orders_are_four(self):
        for column, (module, constant) in DECLARATIONS.items():
            with self.subTest(column=column):
                self.assertEqual(getattr(module, constant), 4)


if __name__ == "__main__":
    unittest.main()
