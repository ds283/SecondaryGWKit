"""
Tests for prompt 20 of ``prompts/GkTk-remedial``: the numeric break-point policy is part of the
two numeric datastore lookup keys.

Since prompt 19 ``numeric_with_phase_cut`` takes a ``break_point_kind``, and the two sectors pass
different values -- ``TkNumericIntegration`` asks for every declared break point,
``GkNumericIntegration`` for the jumps alone. That argument changes the integration: on
``QCD_Cosmology`` it moves the stored transfer function by up to 1.61e-04 of the envelope
(``docs/gktk-remedial/TK-NUMERIC-ATOL-SWEEP.md`` §10.5) on top of the 2.82e-04 the split itself
moved it (§9.4). Before this commit it was in neither lookup key, so a row computed under one
policy was indistinguishable by key from a row computed under the other and a pipeline run that
found the older row took it.

Four things are checked, in the order prompt 20 §3 states them:

1. **The two uses cannot drift.** Each compute target carries exactly one declaration of its
   policy, ``BREAK_POINT_KIND``; ``compute()`` passes *that* to the integrator (read with ``ast``),
   the factory's ``store()`` writes *that*, and the factory's ``build()`` filters on *that*.
   Re-pointing the single constant moves all three together.
2. **A lookup under the wrong policy misses, and under the right policy hits.** The ``build()``
   query is compiled with no database behind it and its ``WHERE`` clause read back.
3. **An old-schema datastore raises**, with a message naming this prompt.
4. **No computed value moves.** That is not tested here -- it is a property of the diff, which
   does not touch ``Quadrature/`` at all, and is measured in ``logs/20-key-the-break-point-policy.md``
   by running a production QCD object in both sectors on this tree and on ``HEAD~1``.

Nothing here needs Ray, a datastore or SQLite. The tables are built from the factories' own
``register()`` output in the shape ``Datastore.SQL.Datastore._build_schema`` builds them, and the
connection is a stand-in that captures the query instead of executing it -- the "compile the query
without a connection" route prompt 20 §4 names, alongside the ``ast`` route that
``test_numeric_break_points.test_each_production_call_site_passes_the_kind_its_sector_decided_on``
already uses for the call sites.
"""

import ast
import unittest
from pathlib import Path

import sqlalchemy as sqla
from sqlalchemy.exc import OperationalError

from ComputeTargets.BackgroundModel import (
    BREAK_POINT_ALL,
    BREAK_POINT_DISCONTINUITY,
)
from ComputeTargets.GkNumericIntegration import GkNumericIntegration
from ComputeTargets.TkNumericIntegration import TkNumericIntegration
from CosmologyModels.GenericEOS.GenericEOS import BREAK_POINT_KINDS
from Datastore.SQL.ObjectFactories.GkNumericIntegration import (
    sqla_GkNumericIntegration_factory,
)
from Datastore.SQL.ObjectFactories.TkNumericIntegration import (
    sqla_TkNumericIntegration_factory,
)

REPO_ROOT = Path(__file__).parents[2]

# the value used when the test needs a policy that is *not* the sector's own, to show that the
# filter separates them. It is never written to anything.
FOREIGN_KIND = "not-the-policy-this-sector-asked-for"


# ---------------------------------------------------------------------------------------------
# stand-ins
# ---------------------------------------------------------------------------------------------


class _Serial:
    """anything the factory reads only a store_id from"""

    def __init__(self, store_id: int):
        self.store_id = store_id


class _QueryCaptured(Exception):
    """
    raised by the stand-in connection in place of executing. Deliberately *not* a
    SQLAlchemyError, so that it passes through the factory's new missing-column handler
    untouched.
    """

    def __init__(self, query):
        super().__init__("query captured")
        self.query = query


class _CapturingConnection:
    def execute(self, query):
        raise _QueryCaptured(query)


class _RaisingConnection:
    def __init__(self, exc):
        self._exc = exc

    def execute(self, query):
        raise self._exc


def _missing_column_error(table: str) -> OperationalError:
    """
    what SQLite raises when a query names a column an existing table does not have. The message is
    reproduced verbatim from sqlite3; the factory matches on the column name inside it, exactly as
    the BackgroundModel factory does for prompts 03 and 04.
    """
    return OperationalError(
        f"SELECT {table}.serial ... WHERE {table}.break_point_kind = ?",
        {},
        Exception(f"no such column: {table}.break_point_kind"),
    )


def _build_table(name: str, factory, metadata: sqla.MetaData) -> sqla.Table:
    """
    Reproduce the table Datastore._build_schema builds for this factory, from the factory's own
    register() output -- so that the schema under test is the production schema, not a copy of it.
    Foreign keys are left pointing at tables this MetaData does not hold; nothing here resolves
    them, because every join below supplies an explicit ON clause.
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


def _support_tables(sector: str, metadata: sqla.MetaData) -> dict:
    """the other tables the two build() methods join against or name"""
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
        f"{sector}Numeric_tags": sqla.Table(
            f"{sector}Numeric_tags",
            metadata,
            sqla.Column("integration_serial", sqla.Integer),
            sqla.Column("tag_serial", sqla.Integer),
        ),
    }


def _capture_build_query(sector: str):
    """
    Run the production build() far enough to form its query, and hand the query back.

    The connection raises instead of executing, so nothing needs a database; the payload carries
    only the store_ids build() reads before the query is issued.
    """
    if sector == "Gk":
        factory = sqla_GkNumericIntegration_factory
        redshift_key = "z_source"
    else:
        factory = sqla_TkNumericIntegration_factory
        redshift_key = "z_init"

    metadata = sqla.MetaData()
    table = _build_table(f"{sector}NumericIntegration", factory, metadata)
    tables = _support_tables(sector, metadata)

    payload = {
        "solver_labels": {},
        "atol": _Serial(11),
        "rtol": _Serial(12),
        "k": _Serial(13),
        "model": _Serial(14),
        "z_sample": None,
        redshift_key: None,
        "tags": [],
    }

    try:
        factory.build(payload, _CapturingConnection(), table, None, tables, None)
    except _QueryCaptured as e:
        return e.query

    raise AssertionError(f"{sector}NumericIntegration.build() did not issue a query")


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


def _integrator_call(module: str) -> ast.Call:
    """the single numeric_with_phase_cut.remote(...) call in a compute-target module"""
    path = REPO_ROOT / module
    tree = ast.parse(path.read_text(), filename=str(path))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "remote"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "numeric_with_phase_cut"
    ]
    assert len(calls) == 1, module
    return calls[0]


# ---------------------------------------------------------------------------------------------


SECTORS = {
    "Gk": (GkNumericIntegration, BREAK_POINT_DISCONTINUITY),
    "Tk": (TkNumericIntegration, BREAK_POINT_ALL),
}


class TestOneDeclarationPerSector(unittest.TestCase):
    """
    Prompt 20 §2.1 and §3 item 3: the value passed to the integrator and the value used in the key
    cannot differ.
    """

    def test_each_sector_declares_the_kind_its_measurement_chose(self):
        for sector, (cls, expected) in SECTORS.items():
            with self.subTest(sector=sector):
                self.assertEqual(cls.BREAK_POINT_KIND, expected)
                self.assertIn(cls.BREAK_POINT_KIND, BREAK_POINT_KINDS)

        # the whole hazard depends on the two being different; if they were not, nothing would
        # need keying
        self.assertNotEqual(
            GkNumericIntegration.BREAK_POINT_KIND,
            TkNumericIntegration.BREAK_POINT_KIND,
        )

    def test_the_accessor_returns_the_class_constant(self):
        for sector, (cls, expected) in SECTORS.items():
            with self.subTest(sector=sector):
                # the property's getter, called on the class rather than an instance: an instance
                # needs units and a model proxy, and the getter reads nothing else
                self.assertEqual(cls.break_point_kind.fget(cls), expected)

    def test_the_integrator_call_site_reads_the_class_constant(self):
        """
        Read with ``ast``, as prompt 19's call-site test is: neither integration module can be
        imported and called without Ray, and the point is the argument, not the call.

        Prompt 19 wrote the literal name here. Prompt 20 replaces it with the class constant,
        because the same value is now also stored and queried -- a literal at the call site would
        be one of three independent copies.
        """
        for sector, module in (
            ("Gk", "ComputeTargets/GkNumericIntegration.py"),
            ("Tk", "ComputeTargets/TkNumericIntegration.py"),
        ):
            with self.subTest(sector=sector):
                keywords = {
                    kw.arg: kw.value for kw in _integrator_call(module).keywords
                }
                self.assertIn("break_point_kind", keywords)

                argument = keywords["break_point_kind"]
                self.assertIsInstance(argument, ast.Attribute)
                self.assertEqual(argument.attr, "BREAK_POINT_KIND")
                self.assertIsInstance(argument.value, ast.Name)
                self.assertEqual(argument.value.id, "self")

    def test_repointing_the_constant_moves_the_query_and_the_stored_value_together(
        self,
    ):
        """
        The direct demonstration that the uses cannot drift: change the one declaration, and both
        the key the factory queries on and the value the factory stores follow it. Nothing else in
        either module mentions the policy.
        """
        for sector, (cls, original) in SECTORS.items():
            with self.subTest(sector=sector):
                sentinel = f"sentinel-{sector}"
                setattr(cls, "BREAK_POINT_KIND", sentinel)
                try:
                    self.assertEqual(
                        _equality_criteria(_capture_build_query(sector))[
                            "break_point_kind"
                        ],
                        sentinel,
                    )
                    self.assertEqual(
                        _store_payload(sector)["break_point_kind"], sentinel
                    )
                    self.assertEqual(cls.break_point_kind.fget(cls), sentinel)
                finally:
                    setattr(cls, "BREAK_POINT_KIND", original)

                # and the sector is back where it started
                self.assertEqual(cls.BREAK_POINT_KIND, original)


class TestTheKeyField(unittest.TestCase):
    """Prompt 20 §2.2 and §3 item 2."""

    def test_the_column_is_registered_and_not_nullable(self):
        for sector, factory in (
            ("Gk", sqla_GkNumericIntegration_factory),
            ("Tk", sqla_TkNumericIntegration_factory),
        ):
            with self.subTest(sector=sector):
                columns = {c.name: c for c in factory.register()["columns"]}
                self.assertIn("break_point_kind", columns)

                column = columns["break_point_kind"]
                self.assertFalse(column.nullable)
                self.assertIsInstance(column.type, sqla.String)

    def test_build_filters_on_the_policy_beside_the_tolerances(self):
        for sector, (cls, expected) in SECTORS.items():
            with self.subTest(sector=sector):
                criteria = _equality_criteria(_capture_build_query(sector))

                # filtered on, not merely selected
                self.assertIn("break_point_kind", criteria)
                self.assertEqual(criteria["break_point_kind"], expected)

                # ... alongside, not instead of, the key that was already there
                for column in (
                    "wavenumber_exit_serial",
                    "model_serial",
                    "atol_serial",
                    "rtol_serial",
                ):
                    self.assertIn(column, criteria)

    def test_the_two_sectors_compile_to_different_criteria(self):
        gk = _capture_build_query("Gk").whereclause.compile(
            compile_kwargs={"literal_binds": True}
        )
        tk = _capture_build_query("Tk").whereclause.compile(
            compile_kwargs={"literal_binds": True}
        )

        self.assertIn("break_point_kind", str(gk))
        self.assertIn("break_point_kind", str(tk))
        self.assertIn(f"'{BREAK_POINT_DISCONTINUITY}'", str(gk))
        self.assertIn(f"'{BREAK_POINT_ALL}'", str(tk))

    def test_a_row_stored_under_the_other_policy_misses_and_its_own_hits(self):
        """
        The point of the prompt. The criterion is an equality, so what SQL does with a candidate
        row is exactly this comparison: a row carrying the policy the query asks for is returned,
        a row carrying any other policy -- including the *other sector's* policy, which is what a
        QCD datastore spanning prompt 19 would hold -- is not.
        """
        for sector, (cls, expected) in SECTORS.items():
            with self.subTest(sector=sector):
                required = _equality_criteria(_capture_build_query(sector))[
                    "break_point_kind"
                ]

                other = SECTORS["Tk" if sector == "Gk" else "Gk"][1]

                self.assertTrue(_stored_row_matches(expected, required))
                self.assertFalse(_stored_row_matches(other, required))
                self.assertFalse(_stored_row_matches(FOREIGN_KIND, required))

    def test_store_writes_the_policy_the_query_asks_for(self):
        for sector, (cls, expected) in SECTORS.items():
            with self.subTest(sector=sector):
                written = _store_payload(sector)
                self.assertEqual(written["break_point_kind"], expected)
                self.assertEqual(
                    written["break_point_kind"],
                    _equality_criteria(_capture_build_query(sector))[
                        "break_point_kind"
                    ],
                )


class TestOldSchemaFailsLoudly(unittest.TestCase):
    """
    Prompt 20 §2.3 and §3 item 4. There is no defensible default for a row computed under a policy
    nobody recorded, so there is no migration and no silent miss.
    """

    def test_a_missing_column_names_the_prompt_and_demands_regeneration(self):
        for sector, table_name in (
            ("Gk", "GkNumericIntegration"),
            ("Tk", "TkNumericIntegration"),
        ):
            with self.subTest(sector=sector):
                with self.assertRaises(RuntimeError) as caught:
                    _run_build_against(sector, _missing_column_error(table_name))

                message = str(caught.exception)
                self.assertIn("break_point_kind", message)
                self.assertIn("prompt 20", message)
                self.assertIn("regenerated", message)
                self.assertIn("no migration", message)
                self.assertIn(table_name, message)

    def test_an_unrelated_database_error_is_not_disguised(self):
        unrelated = OperationalError("SELECT ...", {}, Exception("database is locked"))

        for sector in ("Gk", "Tk"):
            with self.subTest(sector=sector):
                with self.assertRaises(OperationalError):
                    _run_build_against(sector, unrelated)


# ---------------------------------------------------------------------------------------------
# store()-side stand-ins, defined after the sector table so that they can read from it
# ---------------------------------------------------------------------------------------------


class _Data:
    compute_time = 1.0
    compute_steps = 2
    RHS_evaluations = 3
    mean_RHS_time = 4.0
    max_RHS_time = 5.0
    min_RHS_time = 6.0


class _ZSample:
    min = _Serial(21)


class _StoreStandIn:
    """
    What the factory's store() reads off a numeric integration object. Everything except
    ``break_point_kind`` is filler; ``break_point_kind`` reads the production class constant
    through the production accessor, which is the field under test.
    """

    label = "stand-in"
    tags = []
    values = []

    def __init__(self, cls):
        self._cls = cls
        self._k_exit = _Serial(13)
        self.model_proxy = _Serial(14)
        self._atol = _Serial(11)
        self._rtol = _Serial(12)
        self._solver = _Serial(15)
        self._z_source = _Serial(16)
        self._z_init = _Serial(16)
        self._z_sample = _ZSample()
        self._values = []
        self._data = _Data()
        self._has_unresolved_osc = False
        self._unresolved_z = None
        self._unresolved_efolds_subh = None
        self._init_efolds_suph = 1.0
        self._stop_deltaz_subh = 1.0
        self._stop_G = 1.0
        self._stop_Gprime = 1.0
        self._stop_T = 1.0
        self._stop_Tprime = 1.0

    @property
    def break_point_kind(self) -> str:
        return self._cls.break_point_kind.fget(self._cls)


def _store_payload(sector: str) -> dict:
    """run the production store() with a capturing inserter, and hand back what it inserted"""
    cls, _ = SECTORS[sector]
    factory = (
        sqla_GkNumericIntegration_factory
        if sector == "Gk"
        else sqla_TkNumericIntegration_factory
    )

    captured = {}

    def inserter(conn, data):
        captured.update(data)
        return 99

    factory.store(
        _StoreStandIn(cls),
        None,
        None,
        inserter,
        None,
        {f"{sector}Numeric_tags": None, f"{sector}NumericValue": None},
    )

    return captured


def _stored_row_matches(stored: str, required: str) -> bool:
    """what SQL's equality does with a candidate row's column value"""
    return stored == required


def _run_build_against(sector: str, exc: Exception):
    """issue the production build() against a connection that raises ``exc``"""
    if sector == "Gk":
        factory = sqla_GkNumericIntegration_factory
        redshift_key = "z_source"
    else:
        factory = sqla_TkNumericIntegration_factory
        redshift_key = "z_init"

    metadata = sqla.MetaData()
    table = _build_table(f"{sector}NumericIntegration", factory, metadata)
    tables = _support_tables(sector, metadata)

    payload = {
        "solver_labels": {},
        "atol": _Serial(11),
        "rtol": _Serial(12),
        "k": _Serial(13),
        "model": _Serial(14),
        "z_sample": None,
        redshift_key: None,
        "tags": [],
    }

    return factory.build(payload, _RaisingConnection(exc), table, None, tables, None)


if __name__ == "__main__":
    unittest.main()
