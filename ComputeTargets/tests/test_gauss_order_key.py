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
3. **When the constant moves, the key moves.** The single declaration is repointed and both the
   compiled ``whereclause`` and the value ``store()`` writes for an object built while it says so
   follow it together. A test that only checked the column exists would not show this.
4. **A row at another order misses, and its own hits** -- what SQL does with a candidate row is
   the equality this test performs directly.
5. **No tolerance survives in ``GkSource``'s criteria.**

and, added by prompt **05b**, the invariant that is about the *object* rather than the key:

6. **An object reports the order it was built at, on every path by which it can come into
   existence.** Computed fresh, it reports the order its tables were actually constructed with;
   rehydrated from a row, it reports the row's stored order and rebuilds its tables at that
   order.
7. **``build()`` still filters on the current module constant.** That is the lookup semantics --
   *give me a row computed at the order this run is configured for* -- and 05b does not weaken
   it. A row computed at another order is a different row, not a miss to repair.

Prompt 05 delivered 1-5 through *one name resolved at call time*: the compute path read the
module constant, the object's property re-read it, ``store()`` wrote that property and ``build()``
filtered on the same module attribute. That is right about the key and wrong about the object. A
property that re-reads a constant reports what the module currently says, not what the object is,
and the two parted company in two places: ``phase_residual``'s ``order=`` keyword let a residual
table be built at one order and persisted at another (a default argument is bound once, at
``def`` time, so the table could not even follow a re-pointed constant while the key column did),
and a rehydrated ``BackgroundModel`` reassembled its three cumulative tables at the *current*
constant while its row's three order columns were selected and never passed to the constructor.
Both were masked by ``build()``'s filter, which is what makes correctness rest on an argument
about the filter rather than on construction.

Since 05b the order travels as data: ``WKB_phase_function`` carries the residual table's own
``CumulativeTable.order`` out in its payload, ``compute_background``'s payload echoes the three
orders its tables were built at, each ``store()`` records the payload's value, and each
``build()`` hands the row's column to the constructor. ``TestAnObjectReportsTheOrderItWasBuiltAt``
below is the demonstration, on both paths.

Nothing here needs Ray, a datastore or SQLite. The tables are built from the factories' own
``register()`` output in the shape ``Datastore.SQL.Datastore._build_schema`` builds them, and the
connection is a stand-in that captures the query instead of executing it.
"""

import ast
import inspect
import types
import unittest
from importlib import import_module
from pathlib import Path
from unittest import mock

import numpy as np
import sqlalchemy as sqla

import ComputeTargets.phase_residual as phase_residual
from ComputeTargets.WKB_Gk import Gk_d_ln_omegaEff_dz, Gk_omegaEff_sq
from ComputeTargets.phase_residual import (
    cached_phase_residual,
    clear_phase_residual_cache,
)

# prompt 06's stand-ins and fixtures, as test_residual_table_reuse.py imports them: an exact
# radiation model with the three cumulative tables attached, and the production phase integrator
from ComputeTargets.tests.test_gk_wkb_phase import (
    _KExit as _PhaseKExit,
    _Proxy as _PhaseProxy,
    _radiation_z_e3,
    radiation_model_with_tables,
)
from ComputeTargets.tests.wkb_reference import to_redshift_array
from Quadrature.integrators.WKB_phase_function import (
    PHASE_SOLVER_LABEL,
    WKB_phase_function,
)
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

        # a freshly computed model: compute_background built its three tables at the orders the
        # run is configured at and echoed them into its payload, and BackgroundModel.store()
        # recorded those echoes. Read here, at construction, for the same reason -- an object
        # that is built now carries the orders that hold now
        self._tau_gauss_order = background_model.TAU_GAUSS_ORDER
        self._cs_tau_gauss_order = background_model.CS_TAU_GAUSS_ORDER
        self._friction_F_gauss_order = background_model.FRICTION_F_GAUSS_ORDER

    # the production accessors, reproduced by delegation rather than copied: each reports the
    # order recorded on this object, which is the property under test
    _order = background_model.BackgroundModel._order

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
        # a freshly computed object: WKB_phase_function built the residual table at the order the
        # run is configured at and handed it back in its payload, and store() recorded it
        self._rho_gauss_order = phase_residual.RHO_GAUSS_ORDER

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
                        # ... and store() writes the new order, read off an object built
                        # while the declaration says so. (Before prompt 05b there was a third
                        # assertion here, that the compute class's accessor re-read the module
                        # constant; that is the mechanism 05b removed -- the accessor now
                        # reports the object, and the computation's own reading of the
                        # declaration is asserted by TestTheComputationReadsTheSameDeclaration
                        # and by the behavioural test of the WKB path below.)
                        self.assertEqual(_store_payload(target)[column], sentinel)

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
                # the declaration, the CumulativeTable compute_background builds, and the
                # payload key that echoes what it was built at. Since prompt 05b the rebuild
                # sites and the accessors read the *object's* orders instead, which is why this
                # counts three and not more.
                self.assertGreaterEqual(orders.count(name), 3, msg=f"{name}: {orders}")

    def test_the_residual_entry_points_resolve_their_order_at_call_time(self):
        """
        Prompt 05b. ``build_phase_residual``, ``cached_phase_residual`` and
        ``phase_residual_cache_key`` took ``order: int = RHO_GAUSS_ORDER`` until this prompt, and
        a default argument is evaluated once, at ``def`` time: the table could not follow a
        re-pointed declaration, while the row's key column did. The sentinel default resolves the
        same name when the call is made, so a table built without an explicit order is built at
        the order the run is configured at -- and the parameter keeps working for
        ``docs/tolerance-convergence/order_audit.py``, which sweeps it (prompt §4).
        """
        for function in (
            phase_residual.build_phase_residual,
            phase_residual.cached_phase_residual,
            phase_residual.phase_residual_cache_key,
        ):
            with self.subTest(function=function.__name__):
                default = inspect.signature(function).parameters["order"].default
                self.assertIsNone(default)

    def test_no_production_caller_overrides_the_residual_order(self):
        """
        No production caller supplies ``order``; the production path takes the order the run is
        configured at. Since prompt 05b a caller that *did* supply one would be recorded at what
        it asked for rather than silently persisting another number, so this is a statement about
        the production path rather than the guard it was when prompt 05 wrote it.
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

    def test_the_phase_integrator_reads_the_order_through_its_module(self):
        """
        Prompt 05 asserted that ``WKB_phase_function`` records ``RHO_GAUSS_ORDER`` by that name;
        it reached it through a ``from ... import``, which is a snapshot taken when this module
        was imported and therefore cannot follow the declaration either. Since prompt 05b the
        integrator reads it through the module -- so what it reports is what the run is
        configured at -- and the value it finally records is the residual table's own
        ``order``, which the behavioural test below drives rather than reads.
        """
        source = (
            REPO_ROOT / "Quadrature" / "integrators" / "WKB_phase_function.py"
        ).read_text()
        tree = ast.parse(source, filename="WKB_phase_function.py")

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == (
                "ComputeTargets.phase_residual"
            ):
                self.assertNotIn(
                    "RHO_GAUSS_ORDER", [alias.name for alias in node.names]
                )

        reads = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "RHO_GAUSS_ORDER"
        ]
        self.assertGreaterEqual(len(reads), 1)
        for node in reads:
            self.assertEqual(getattr(node.value, "id", None), "phase_residual")


# ---------------------------------------------------------------------------------------------
# prompt 05b: the object reports the order it was built at, on both paths
# ---------------------------------------------------------------------------------------------

# orders no production run is configured at, so that a value that came from the module constant
# and a value that came from the object can never be confused
ROW_ORDERS = {"tau": 6, "cs_tau": 8, "friction_F": 12, "rho": 6}

# a tiny background grid: enough nodes for a CumulativeTable, few enough to cost nothing
ROW_Z_NODES = (1000.0, 800.0, 600.0, 400.0)


class _RowCosmology:
    """what BackgroundModel.build() and _build_*_primitive read off a cosmology"""

    type_id = 4242
    store_id = 7

    def __init__(self):
        self.units = Mpc_units()

    def Hubble(self, z: float) -> float:
        return (1.0 + z) ** 2

    def wPerturbations(self, z: float) -> float:
        return 1.0 / 3.0


def _namespace(**columns):
    return types.SimpleNamespace(**columns)


class _BackgroundRowConnection:
    """
    A stand-in for the datastore: the first query is the model row, the second its values. No
    SQLite and no engine -- the queries are formed by the production build() and thrown away,
    and what comes back is the row a store would hold.
    """

    def __init__(self, orders: dict):
        self._orders = orders
        self._calls = 0

    def execute(self, query):
        self._calls += 1
        if self._calls == 1:
            return [
                _namespace(
                    serial=11,
                    compute_time=1.0,
                    compute_steps=len(ROW_Z_NODES),
                    RHS_evaluations=16,
                    mean_RHS_time=1.0,
                    max_RHS_time=1.0,
                    min_RHS_time=1.0,
                    solver_serial=3,
                    label="stand-in row",
                    z_samples=len(ROW_Z_NODES),
                    solver_label="cumulative-GL-stepping4",
                    solver_stepping=4,
                    source_grid_digest="deadbeef",
                    source_grid_construction=2,
                    tau_gauss_order=self._orders["tau"],
                    cs_tau_gauss_order=self._orders["cs_tau"],
                    friction_F_gauss_order=self._orders["friction_F"],
                )
            ]

        return [
            _namespace(
                serial=100 + i,
                z_serial=200 + i,
                z=z,
                z_is_source=True,
                z_is_response=True,
                Hubble_GeV=1.0,
                wBackground=1.0 / 3.0,
                wPerturbations=1.0 / 3.0,
                rho_GeV=1.0,
                tau_Mpc=1.0 / (1.0 + z),
                tau_lo_Mpc=0.0,
                cs_tau_Mpc=1.0 / (1.0 + z),
                cs_tau_lo_Mpc=0.0,
                friction_F=0.0,
                T_photon_GeV=1.0,
                d_lnH_dz=2.0 / (1.0 + z),
                d2_lnH_dz2=0.0,
                d3_lnH_dz3=0.0,
                d_wPerturbations_dz=0.0,
                d2_wPerturbations_dz2=0.0,
            )
            for i, z in enumerate(ROW_Z_NODES)
        ]


class _WKBRowConnection:
    """the same, for either WKB factory: one row, and no values (_do_not_populate)"""

    def __init__(self, order: int):
        self._order = order

    def execute(self, query):
        return self

    def one_or_none(self):
        return _namespace(
            serial=11,
            sin_coeff=1.0,
            cos_coeff=0.0,
            stage_1_compute_time=1.0,
            stage_1_compute_steps=1,
            stage_1_RHS_evaluations=1,
            stage_1_mean_RHS_time=None,
            stage_1_max_RHS_time=None,
            stage_1_min_RHS_time=None,
            stage_2_compute_time=None,
            stage_2_compute_steps=None,
            stage_2_RHS_evaluations=None,
            stage_2_mean_RHS_time=None,
            stage_2_max_RHS_time=None,
            stage_2_min_RHS_time=None,
            friction_compute_time=None,
            friction_compute_steps=None,
            friction_RHS_evaluations=None,
            friction_mean_RHS_time=None,
            friction_max_RHS_time=None,
            friction_min_RHS_time=None,
            has_WKB_violation=False,
            WKB_violation_z=None,
            WKB_violation_efolds_subh=None,
            init_efolds_subh=1.0,
            init_efolds_suph=1.0,
            metadata=None,
            rho_gauss_order=self._order,
            solver_serial=3,
            solver_label=PHASE_SOLVER_LABEL,
            solver_stepping=4,
            phase_solver_serial=3,
            phase_solver_label=PHASE_SOLVER_LABEL,
            phase_solver_stepping=4,
            friction_solver_serial=4,
            friction_solver_label=PHASE_SOLVER_LABEL,
            friction_solver_stepping=4,
            label="stand-in row",
            z_source_serial=17,
            z_source=1.0e5,
            z_source_is_source=True,
            z_source_is_response=True,
            z_samples=3,
            z_init=1.0e5,
            G_init=0.0,
            Gprime_init=1.0,
            T_init=1.0,
            Tprime_init=0.0,
        )


class _RowKExit:
    """a wavenumber_exit_time stand-in carrying the units check_units performs"""

    def __init__(self, units):
        self.store_id = 13
        self.units = units
        self.k = _RowWavenumber(units)
        self.z_exit_subh_e3 = 1.0e6


class _RowWavenumber:
    def __init__(self, units):
        self.store_id = 13
        self.units = units
        self.k = 1.0e5
        self.k_inv_Mpc = 1.0e5


class _RowProxy:
    def __init__(self, units):
        self.store_id = 14
        self.units = units


def _value_tables(metadata: sqla.MetaData) -> dict:
    """the BackgroundModelValue table build() reads its samples from"""
    columns = [
        "model_serial",
        "z_serial",
        "Hubble_GeV",
        "wBackground",
        "wPerturbations",
        "rho_GeV",
        "tau_Mpc",
        "tau_lo_Mpc",
        "cs_tau_Mpc",
        "cs_tau_lo_Mpc",
        "friction_F",
        "T_photon_GeV",
        "d_lnH_dz",
        "d2_lnH_dz2",
        "d3_lnH_dz3",
        "d_wPerturbations_dz",
        "d2_wPerturbations_dz2",
    ]
    table = sqla.Table(
        "BackgroundModelValue",
        metadata,
        sqla.Column("serial", sqla.Integer, primary_key=True),
        *[sqla.Column(name, sqla.Float(64)) for name in columns],
    )
    return {"BackgroundModelValue": table}


def _rehydrate(target: str, connection):
    """the production build(), against a row a store would hold rather than a database"""
    factory, table_name, tag_table, _ = TARGETS[target]

    metadata = sqla.MetaData()
    table = _build_table(table_name, factory, metadata)
    tables = _support_tables(tag_table, metadata)

    if target == "BackgroundModel":
        tables.update(_value_tables(metadata))
        payload = {
            "solver_labels": {},
            "cosmology": _RowCosmology(),
            # the read path: no grid is asserted, so build() does not filter on the digest
            "z_sample": None,
            "tags": [],
        }
    else:
        units = Mpc_units()
        payload = {
            "solver_labels": {},
            "k": _RowKExit(units),
            "model": _RowProxy(units),
            "z_sample": None,
            "z_source": None,
            "tags": [],
            # the object's own values are not what is under test here, and reading them would
            # need a second canned query
            "_do_not_populate": True,
        }

    return factory.build(payload, connection, table, None, tables, None)


class TestAnObjectReportsTheOrderItWasBuiltAt(unittest.TestCase):
    """
    Prompt 05b §2, the invariant on the **rehydration** path: an object built from a row reports
    that row's order, and ``BackgroundModel`` reassembles its three cumulative tables at it.

    Each row below carries an order no production run is configured at, so a value that came from
    the module constant is distinguishable from one that came from the row. Both of these fail
    against ``90d0114``, where the accessors re-read the constant and ``build()`` selected the
    three ``BackgroundModel`` columns without passing any of them to the constructor.
    """

    def test_a_rehydrated_background_model_reports_its_row_and_not_the_module(self):
        obj = _rehydrate("BackgroundModel", _BackgroundRowConnection(ROW_ORDERS))

        self.assertEqual(obj.tau_gauss_order, ROW_ORDERS["tau"])
        self.assertEqual(obj.cs_tau_gauss_order, ROW_ORDERS["cs_tau"])
        self.assertEqual(obj.friction_F_gauss_order, ROW_ORDERS["friction_F"])

        # and none of them is the constant, which is what makes the assertion above meaningful
        self.assertNotEqual(background_model.TAU_GAUSS_ORDER, ROW_ORDERS["tau"])
        self.assertNotEqual(background_model.CS_TAU_GAUSS_ORDER, ROW_ORDERS["cs_tau"])
        self.assertNotEqual(
            background_model.FRICTION_F_GAUSS_ORDER, ROW_ORDERS["friction_F"]
        )

    def test_a_rehydrated_background_model_rebuilds_its_tables_at_its_row_order(self):
        """
        The half of the defect the issue did not name. The persisted (hi, lo) limbs were
        integrated at the row's order; every off-grid partial ``delta()`` later evaluated against
        the reassembled table uses the table's own order, so a table reassembled at the module's
        order applies one rule over nodes produced by another.
        """
        obj = _rehydrate("BackgroundModel", _BackgroundRowConnection(ROW_ORDERS))

        self.assertEqual(obj._build_tau_primitive().table.order, ROW_ORDERS["tau"])
        self.assertEqual(
            obj._build_cs_tau_primitive().table.order, ROW_ORDERS["cs_tau"]
        )
        self.assertEqual(
            obj._build_friction_F_primitive().table.order, ROW_ORDERS["friction_F"]
        )

    def test_a_rehydrated_wkb_object_reports_its_row_and_not_the_module(self):
        for target in ("GkWKBIntegration", "TkWKBIntegration"):
            with self.subTest(target=target):
                obj = _rehydrate(target, _WKBRowConnection(ROW_ORDERS["rho"]))

                self.assertEqual(obj.rho_gauss_order, ROW_ORDERS["rho"])
                self.assertNotEqual(phase_residual.RHO_GAUSS_ORDER, ROW_ORDERS["rho"])

    def test_build_still_filters_on_the_module_constant(self):
        """
        Prompt 05b §2's second half, which is **not** weakened: the lookup asks for a row
        computed at the order this run is configured for, so the row rehydrated above -- which a
        real database would never have returned -- is a different row rather than a miss to
        repair.
        """
        for target, expected in ORDER_COLUMNS.items():
            criteria = _equality_criteria(_capture_build_query(target))
            for column in expected:
                module, constant = DECLARATIONS[column]
                with self.subTest(target=target, column=column):
                    self.assertEqual(criteria[column], getattr(module, constant))

    def test_an_object_with_no_order_refuses_rather_than_reporting_the_module(self):
        """
        A query-shaped object has computed nothing and came from no row, so it has no order to
        report. Before 05b it answered with the module constant, which is the whole defect in
        miniature.
        """
        obj = _rehydrate("GkWKBIntegration", _EmptyWKBRowConnection())

        with self.assertRaises(RuntimeError):
            obj.rho_gauss_order


class _EmptyWKBRowConnection:
    """a lookup that misses: build() returns an unpopulated object"""

    def execute(self, query):
        return self

    def one_or_none(self):
        return None


class TestTheRecordedOrderIsTheOrderTheTableWasBuiltAt(unittest.TestCase):
    """
    Prompt 05b §6 test 2, on the **compute** path: drive the production phase integrator at an
    order that is not ``RHO_GAUSS_ORDER`` and check that what ``store()`` would write is the
    order the residual table was actually built at.

    This fails against ``90d0114`` twice over: the payload carried no order at all, and
    ``build_phase_residual``'s ``order`` default was bound at ``def`` time, so the table went on
    being built at 4 while the key column recorded the re-pointed constant.
    """

    @classmethod
    def setUpClass(cls):
        cls.units = Mpc_units()
        cls.k = 1.0e7
        z_e3 = _radiation_z_e3(cls.k)
        cls.z_nodes = np.geomspace(z_e3, 1.0e3, 60)
        cls.model = radiation_model_with_tables(cls.z_nodes)
        cls.samples = to_redshift_array([float(z) for z in cls.z_nodes[5:40]])
        cls.z_init = float(cls.z_nodes[3])

    def setUp(self):
        clear_phase_residual_cache()

    def tearDown(self):
        clear_phase_residual_cache()

    def _drive(self) -> dict:
        return WKB_phase_function._function(
            _PhaseProxy(self.model, self.units),
            _PhaseKExit(self.k, self.units),
            self.z_init,
            self.samples,
            sector="Gk",
            omega_sq=Gk_omegaEff_sq,
            d_ln_omega_dz=Gk_d_ln_omegaEff_dz,
            task_label="test_gauss_order_key",
            object_label="test",
        )

    def _table(self):
        """the table the run above used: a cache hit, asserted to be one"""
        table, reused = cached_phase_residual(
            self.model, self.k, self.model.functions.tau.table.z_nodes, "Gk"
        )
        self.assertTrue(reused)
        return table

    def test_the_payload_carries_the_table_s_own_order(self):
        for order in (phase_residual.RHO_GAUSS_ORDER, 6):
            with self.subTest(order=order):
                clear_phase_residual_cache()
                with mock.patch.object(phase_residual, "RHO_GAUSS_ORDER", order):
                    payload = self._drive()
                    table = self._table()

                    self.assertEqual(table.order, order)
                    self.assertEqual(payload["rho_gauss_order"], table.order)
                    self.assertEqual(payload["metadata"]["N_rho"], table.order)

    def test_store_writes_the_order_the_table_was_built_at(self):
        clear_phase_residual_cache()
        with mock.patch.object(phase_residual, "RHO_GAUSS_ORDER", 6):
            payload = self._drive()
            table = self._table()

        # what store() records, through the production accessor: an object whose phase came from
        # that table reports that table's order
        obj = _WKBStandIn(
            import_module("ComputeTargets.GkWKBIntegration").GkWKBIntegration
        )
        obj._rho_gauss_order = int(payload["rho_gauss_order"])

        captured = {}

        def inserter(conn, data):
            captured.update(data)
            return 99

        sqla_GkWKBIntegration_factory.store(
            obj, None, None, inserter, None, {"GkWKB_tags": None, "GkWKBValue": None}
        )

        self.assertEqual(captured["rho_gauss_order"], table.order)
        self.assertEqual(captured["rho_gauss_order"], 6)
        self.assertNotEqual(captured["rho_gauss_order"], phase_residual.RHO_GAUSS_ORDER)


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
