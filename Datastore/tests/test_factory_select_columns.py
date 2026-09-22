"""
Structural guard: every attribute a factory's ``build()`` reads off a query result must be a
column that query actually asked for.

The defect this exists for (prompts/datastore-readback, prompt 01): the ``SELECT`` in
``Datastore/SQL/ObjectFactories/QuadSourceIntegral.py`` omitted ``table.c.numeric_quad`` while
``build()`` read ``row_data.numeric_quad``. SQLAlchemy raises ``NoSuchColumnError`` only when the
attribute is touched, so the defect is invisible until a *stored* row is read back. A fresh
pipeline run computes these rows and never reads them; a resume does, and lost a 10 h 33 m run.
Nothing in the tree exercised the read-back path, so nothing caught it.

The check is static. The module sources under ``Datastore/SQL/ObjectFactories/`` are parsed with
``ast`` -- following the precedent of ``ComputeTargets/tests/test_main_plumbing.py``, which reads
``main.py`` statically rather than executing it -- and for each ``build()`` we collect

  * the columns its ``sqla.select(...)`` (plus any ``.add_columns(...)``) requests, and
  * the attribute names it reads off the variable the executed query was assigned to,

then assert the second is a subset of the first. No Ray, no datastore, no SQLAlchemy engine: the
test never runs a query, and so cannot be defeated by the absence of one.

WHAT THIS TEST CANNOT SEE
=========================

It is a static approximation and the following are outside it. Each is a real hole, not a
theoretical one; three of them exist in this tree today.

1. **Dynamic attribute access.** Reads written as ``getattr(row, name)`` or
   ``row._mapping[name]`` are not ``ast.Attribute`` nodes and are not collected.
   ``sqla_wavenumber_exit_time_factory.build`` reads all of its ``z_exit_suph_e*`` /
   ``z_exit_subh_e*`` columns through ``row_data._mapping[f"..."]``; this test says nothing
   about them.

2. **Dynamically named columns.** ``table.c[f"z_exit_suph_e{z_offset}"]`` inside a loop cannot be
   resolved to a name. Such columns can only *add* to the requested set, so a query carrying them
   can still be cleared when nothing is missing -- but it can never be *convicted*, and this test
   reports it as undecidable rather than as a defect. ``sqla_wavenumber_exit_time_factory`` is
   again the case in point.

3. **Only ``build()``.** ``read_batch()`` reads rows too, through a nested ``make_object(row)``
   whose ``row`` is a function parameter the analyser cannot link back to a query, and its
   ``SELECT`` is extended by ``add_columns`` in a loop. ``read_batch`` is therefore not checked
   at all. See the issue opened against this in the campaign board.

4. **Rows whose query cannot be resolved.** A row taken from a query built in a nested helper,
   or assembled conditionally, is not decidable. ``sqla_BackgroundModelFactory.build`` is the
   case: its lookup row comes from ``rows[0]`` where ``rows`` came from
   ``_build_query(with_grid_identity)``, whose column list has the two grid-identity columns on
   one branch and not the other, with the reads of them guarded by ``hasattr``. Such a binding is
   declared in ``KNOWN_BLIND`` and neither cleared nor convicted.

5. **Control flow.** Reads are charged to the last binding of that name *above* them in the
   source text, not to the binding that control flow would actually reach. Every ``build()`` here
   has the straight-line shape ``look up, then unpack``, for which the two agree.

   Points 4 and 5 are why the analyser's own reach is pinned:
   :meth:`TestFactorySelectColumns.test_analyser_coverage_is_undiminished` and
   :meth:`TestFactorySelectColumns.test_blind_spots_are_the_declared_ones` fail when a refactor
   moves a read out of sight, rather than letting the guard pass vacuously.

6. **Column existence.** A ``SELECT`` naming a column the table does not have would be caught by
   SQLAlchemy at query-build time, not here; this test compares reads against requests, not
   requests against the schema.
"""

import ast
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Set

FACTORY_DIR = Path(__file__).parents[2] / "Datastore" / "SQL" / "ObjectFactories"

# query-builder methods that carry the selected column list through unchanged
_PASSTHROUGH = frozenset(
    {
        "filter",
        "where",
        "join",
        "outerjoin",
        "select_from",
        "order_by",
        "group_by",
        "having",
        "limit",
        "offset",
        "distinct",
    }
)

# result methods that yield a single row object
_FETCH_ONE = frozenset({"one_or_none", "one", "first", "fetchone"})

# result methods that yield an iterable of row objects
_FETCH_MANY = frozenset({"all", "fetchall"})


def _column_name(node: ast.AST) -> Optional[str]:
    """
    The name a selected-column expression will carry in the result row, or None when that cannot
    be decided statically.
    """
    # table.c.NAME
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "c"
    ):
        return node.attr

    # table.c["NAME"] -- and table.c[f"..."], which is not decidable
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "c"
    ):
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, str):
            return index.value
        return None

    # <anything>.label("NAME")
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "label"
    ):
        if (
            len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            return node.args[0].value
        return None

    return None


class _QuerySpec:
    """The column names a query requests, and the expressions that defeated resolution."""

    def __init__(self):
        self.columns: Set[str] = set()
        self.unresolved: List[str] = []

    def copy(self) -> "_QuerySpec":
        other = _QuerySpec()
        other.columns = set(self.columns)
        other.unresolved = list(self.unresolved)
        return other

    def add(self, nodes) -> None:
        for node in nodes:
            name = _column_name(node)
            if name is None:
                self.unresolved.append(ast.unparse(node))
            else:
                self.columns.add(name)


class Finding:
    """
    One binding of a row variable inside a ``build()``: the query it came from, and the attributes
    read off it while that binding was live.
    """

    def __init__(
        self,
        module: str,
        cls: str,
        row_var: str,
        lineno: int,
        spec: Optional[_QuerySpec],
    ):
        self.module = module
        self.cls = cls
        self.row_var = row_var
        self.lineno = lineno
        self.spec = spec
        self.reads: Set[str] = set()

    @property
    def key(self) -> str:
        return f"{self.module}:{self.cls}"

    @property
    def site(self) -> str:
        return f"{self.module}:{self.cls}.{self.row_var}@{self.lineno}"

    @property
    def decidable(self) -> bool:
        if self.spec is None:
            return False
        # dynamically named columns can only enlarge the requested set, so an empty shortfall is
        # still sound; a non-empty one is not, because a loop may supply exactly what is missing
        if len(self.spec.unresolved) > 0 and len(self.missing) > 0:
            return False
        return True

    @property
    def missing(self) -> Set[str]:
        if self.spec is None:
            return set()
        return self.reads - self.spec.columns


def _source_order(node: ast.AST):
    """Depth-first pre-order, which is source order."""
    yield node
    for child in ast.iter_child_nodes(node):
        yield from _source_order(child)


def _executed_query(node: ast.AST) -> Optional[ast.AST]:
    """If `node` is ``conn.execute(<query>)``, return the query expression."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "execute"
        and len(node.args) >= 1
    ):
        return node.args[0]
    return None


def analyse_build(fn: ast.FunctionDef, module: str, cls: str) -> List[Finding]:
    """
    Walk `fn` in source order, tracking the queries it builds, the variables executed results land
    in, and the attributes read off them.

    A row variable rebound part-way through a ``build()`` -- ``row`` used first for a lookup row
    and then for a sample row, as in ``sqla_BackgroundModelFactory.build`` -- yields one Finding
    per binding, and a read is charged to whichever binding is live where it appears. That is
    still an approximation: it follows source order, not control flow, so a read inside a loop
    body is charged to the last binding *above* it in the text. It is sound for the straight-line
    ``lookup, then unpack`` shape every factory here uses.
    """
    queries: Dict[str, _QuerySpec] = {}
    row_lists: Dict[str, Optional[_QuerySpec]] = {}
    live: Dict[str, Finding] = {}
    findings: List[Finding] = []

    # attribute nodes that are the callee of a call are method calls, not column reads
    callees = {
        id(n.func)
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }

    def resolve(expr: ast.AST) -> Optional[_QuerySpec]:
        if isinstance(expr, ast.Name):
            return queries.get(expr.id)
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
            func = expr.func
            if func.attr == "select" and isinstance(func.value, ast.Name):
                spec = _QuerySpec()
                spec.add(expr.args)
                return spec
            base = resolve(func.value)
            if base is None:
                return None
            if func.attr == "add_columns":
                spec = base.copy()
                spec.add(expr.args)
                return spec
            if func.attr in _PASSTHROUGH:
                return base.copy()
            return None
        return None

    def result_of(expr: ast.AST, methods) -> Optional[ast.AST]:
        """``conn.execute(q).<method>()`` -> the query expression, else None."""
        if (
            isinstance(expr, ast.Call)
            and isinstance(expr.func, ast.Attribute)
            and expr.func.attr in methods
        ):
            return _executed_query(expr.func.value)
        return None

    def bind_row(name: str, lineno: int, spec: Optional[_QuerySpec]) -> None:
        finding = Finding(module, cls, name, lineno, spec)
        live[name] = finding
        findings.append(finding)

    def row_list_of(expr: ast.AST):
        """
        Whether `expr` denotes a sequence of result rows, and the query behind it. Returns
        (True, spec-or-None) or (False, None).
        """
        inner = expr
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id in ("list", "tuple")
            and len(inner.args) == 1
        ):
            inner = inner.args[0]
        if isinstance(inner, ast.Name) and inner.id in row_lists:
            return True, row_lists[inner.id]
        many = result_of(inner, _FETCH_MANY)
        if many is not None:
            return True, resolve(many)
        query_expr = _executed_query(inner)
        if query_expr is not None:
            return True, resolve(query_expr)
        return False, None

    def single_row_of(expr: ast.AST):
        """
        Whether `expr` denotes one result row, and the query behind it. Covers
        ``conn.execute(q).one_or_none()`` and an index into a tracked row list, including the
        ``rows[0] if len(rows) == 1 else None`` idiom.
        """
        if isinstance(expr, ast.IfExp):
            for branch in (expr.body, expr.orelse):
                if isinstance(branch, ast.Constant) and branch.value is None:
                    continue
                return single_row_of(branch)
            return False, None
        single = result_of(expr, _FETCH_ONE)
        if single is not None:
            return True, resolve(single)
        if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
            if expr.value.id in row_lists:
                return True, row_lists[expr.value.id]
        return False, None

    for node in _source_order(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                value = node.value

                spec = resolve(value)
                if spec is not None:
                    queries[name] = spec
                    continue

                is_row, spec = single_row_of(value)
                if is_row:
                    bind_row(name, node.lineno, spec)
                    continue

                is_list, spec = row_list_of(value)
                if is_list:
                    row_lists[name] = spec
                    continue

        if isinstance(node, (ast.For, ast.comprehension)):
            if isinstance(node.target, ast.Name):
                is_list, spec = row_list_of(node.iter)
                if is_list:
                    lineno = getattr(node, "lineno", node.iter.lineno)
                    bind_row(node.target.id, lineno, spec)
                    continue

        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            finding = live.get(node.value.id)
            if finding is None:
                continue
            if id(node) in callees or node.attr.startswith("_"):
                continue
            finding.reads.add(node.attr)

    return [finding for finding in findings if len(finding.reads) > 0]


def analyse_factories() -> List[Finding]:
    """Every row-variable binding under ObjectFactories/ that a ``build()`` reads attributes off."""
    findings = []
    for path in sorted(FACTORY_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "build"
            ]:
                findings.extend(analyse_build(fn, path.name, cls.name))
    return findings


# Defects the audit of prompt 01 (prompts/datastore-readback) found and was forbidden by that
# prompt to fix, each with the campaign issue that owns it. An entry here is a known bug, not an
# exemption: delete it in the commit that fixes the factory, and the test below will hold the fix.
KNOWN_UNFIXED = {
    # [01-backgroundmodelvalue-hubble]: the column is "Hubble_GeV", and the equality check on the
    # row-exists branch reads row_data.Hubble, which no SELECT requests and the table does not
    # have. Not reachable from the pipeline today -- BackgroundModel.build() reads its sample rows
    # with its own SELECT rather than through this factory, and nothing calls
    # object_get("BackgroundModelValue") -- so only a future caller of that path would fire it.
    "BackgroundModel.py:sqla_BackgroundModelValue_factory": ["Hubble"],
}


# The row-variable bindings the analyser can see and decide, as "module:class.variable". Pinned
# so that a refactor which moves a row read out of the analyser's reach fails here rather than
# quietly reducing coverage to nothing -- a static guard that has gone blind still passes, which
# is the failure mode that costs everything to detect. Add to it when a factory gains a checkable
# build(); removing an entry needs a reason, recorded in the campaign board.
EXPECTED_COVERAGE = frozenset(
    {
        "BackgroundModel.py:sqla_BackgroundModelFactory.row",
        "BackgroundModel.py:sqla_BackgroundModelValue_factory.row_data",
        "GkNumericIntegration.py:sqla_GkNumericIntegration_factory.row_data",
        "GkNumericIntegration.py:sqla_GkNumericIntegration_factory.row",
        "GkSource.py:sqla_GkSource_factory.row_data",
        "GkSource.py:sqla_GkSource_factory.row",
        "GkSourcePolicyData.py:sqla_GkSourcePolicyData_factory.row_data",
        "GkWKBIntegration.py:sqla_GkWKBIntegration_factory.row_data",
        "GkWKBIntegration.py:sqla_GkWKBIntegration_factory.row",
        "OneLoopIntegral.py:sqla_OneLoopIntegral_factory.row_data",
        "QuadSource.py:sqla_QuadSource_factory.row_data",
        "QuadSource.py:sqla_QuadSource_factory.row",
        "QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory.row_data",
        "TkNumericIntegration.py:sqla_TkNumericIntegration_factory.row_data",
        "TkNumericIntegration.py:sqla_TkNumericIntegration_factory.row",
        "TkWKBIntegration.py:sqla_TkWKBIntegration_factory.row_data",
        "TkWKBIntegration.py:sqla_TkWKBIntegration_factory.row",
        "redshift.py:sqla_redshift_factory.row_data",
        "wavenumber.py:sqla_wavenumber_factory.row_data",
        "wavenumber.py:sqla_wavenumber_exit_time_factory.row_data",
    }
)

# Row variables the analyser sees reads on but cannot decide, with the reason. These are holes,
# listed so that they are visible rather than silently clean, and so that a new one has to be
# looked at by a person instead of appearing as coverage.
KNOWN_BLIND = {
    # its 17 reads are off rows[0], where rows came from a nested _build_query(with_grid_identity)
    # helper whose column list is assembled conditionally. The two prompt-14 grid-identity columns
    # are present on one branch and absent on the other, and the reads of them are guarded by
    # hasattr() -- so the shortfall is genuinely branch-dependent and no column-set comparison can
    # decide it. A path-insensitive analyser must decline rather than guess.
    "BackgroundModel.py:sqla_BackgroundModelFactory.row_data": (
        "row comes from rows[0] of a conditionally-built query in a nested helper"
    ),
}


class TestFactorySelectColumns(unittest.TestCase):
    def setUp(self):
        self.findings = analyse_factories()

    def _decidable_shortfalls(self) -> Dict[str, List[str]]:
        shortfalls: Dict[str, Set[str]] = {}
        for finding in self.findings:
            if not finding.decidable:
                continue
            if len(finding.missing) == 0:
                continue
            shortfalls.setdefault(finding.key, set()).update(finding.missing)
        return {key: sorted(attrs) for key, attrs in shortfalls.items()}

    def test_build_reads_only_columns_its_select_requests(self):
        """
        The guard proper. This is the check that would have caught the QuadSourceIntegral defect
        before it cost a resume; see the module docstring for what it cannot see.
        """
        self.assertEqual(
            self._decidable_shortfalls(),
            KNOWN_UNFIXED,
            "a build() reads an attribute its SELECT does not request; SQLAlchemy raises "
            "NoSuchColumnError the first time a stored row is read back, and only on the "
            "read-back path, so nothing else in this tree will tell you. Add the column to the "
            "SELECT -- or, if this is a known defect being carried deliberately, move it into "
            "KNOWN_UNFIXED naming the issue that owns it.",
        )

    def test_known_unfixed_entries_are_still_present(self):
        """A fixed defect must be struck from KNOWN_UNFIXED, not left to rot there."""
        shortfalls = self._decidable_shortfalls()
        for key, attrs in KNOWN_UNFIXED.items():
            self.assertIn(
                key,
                shortfalls,
                f"{key} no longer falls short of its SELECT: if it has been fixed, remove its "
                "KNOWN_UNFIXED entry; if it has become undecidable, say so in the board",
            )
            self.assertEqual(attrs, shortfalls[key])

    def test_analyser_coverage_is_undiminished(self):
        """
        Guard the guard. A static approximation that stops seeing a factory goes on passing, so
        what it sees is pinned rather than assumed.
        """
        covered = {
            f"{finding.key}.{finding.row_var}"
            for finding in self.findings
            if finding.decidable
        }
        self.assertEqual(
            covered,
            set(EXPECTED_COVERAGE),
            "the set of row reads this guard can decide has changed. If a build() has been "
            "rewritten so its reads are no longer visible, teach the analyser the new shape "
            "rather than leaving it blind; if a factory has been added, add its entry.",
        )

    def test_blind_spots_are_the_declared_ones(self):
        """
        A row variable whose query cannot be resolved is neither cleared nor convicted. The ones
        that exist are declared above with their reason; a new one is a decision for a person.
        """
        blind = {
            f"{finding.key}.{finding.row_var}"
            for finding in self.findings
            if not finding.decidable
        }
        self.assertEqual(
            blind,
            set(KNOWN_BLIND),
            "the analyser cannot decide a row read it is not declared blind to. Extend it, or "
            "declare the hole in KNOWN_BLIND with the reason it cannot be closed.",
        )
