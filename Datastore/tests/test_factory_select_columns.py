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

3. **Only structural shape, not semantics.** The guard (extended by
   ``prompts/datastore-integrity`` prompt 04 to ``read_batch`` and every ``_build_impl_*`` helper,
   by inlining a nested ``make_obj``/``make_object`` helper as a for-loop over the row variable the
   comprehension that calls it iterates over; and by prompt 04b, G4, to every function of every
   factory class and every module-level function -- ``validate_on_startup``'s seven bindings
   included -- rather than a hard-coded list of names) still compares column *names*, never what a
   column is *used for*. It cannot see a ``.label(...)`` attached to the wrong source column
   (``GkNumericIntegration.py``'s two ``…_is_response`` labels, both taken from ``.source``, audit
   §3.11), a column correctly selected but never passed on to the object it builds
   (``TkWKBValue.read_batch``'s missing ``friction``, audit §3.11), or a correctly selected column
   used with the wrong meaning (``QuadSourceIntegral.read_batch`` labels a wavenumber serial and
   then uses it as an exit-time ``store_id``, audit §3.2). Those need a round-trip test that
   actually executes a query, not this guard -- prompt 05's P9.

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

6. **Column existence, only where the table is resolvable and the name is a literal.**
   ``TestColumnExistence`` (prompt 04, G2; walking module-level functions too, and declaring
   everything it does not check, as prompt 04b, G6, adds) resolves every ``<var>.c.<name>`` /
   ``<var>.c["<name>"]`` whose ``<var>`` is the factory's own ``table``, a ``tables["X"]``, or an
   ``.alias(...)`` of either, against the columns ``build_schema`` actually declares, and does the
   same for every key of a dict passed to an ``inserter``/``value_inserter`` call -- catching a
   wrong name before any query or insert ever runs. A ``tables["X"]`` or ``inserters["X"]`` naming
   no schema table at all is reported the same way, wherever the literal appears. It still cannot
   decide ``subquery.c.<name>`` (derived from columns already checked when the subquery itself was
   built), ``query.c.<name>`` (read off a select statement mid-build, before it becomes a table),
   or a bare parameter named ``table`` in a module-level function (which has no entry in
   ``cls_to_table``) -- each such (function, base) pair is declared, with the reason, in
   ``KNOWN_UNRESOLVED_TABLES``, rather than silently skipped. Nor can it decide the keys of an
   insert whose payload is not a dict literal -- a function parameter, or a value built by another
   function -- each such call is declared, with the reason, in ``KNOWN_UNCHECKED_INSERTS``. A
   column named at runtime by an f-string, such as ``wavenumber.py`` and
   ``QuadSourceIntegral.py``'s ``z_exit_..._e{z_offset}`` loops, is declared, with the reason, in
   ``KNOWN_BLIND_COLUMNS``.
"""

import ast
import copy
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
        fn: str,
        row_var: str,
        lineno: int,
        spec: Optional[_QuerySpec],
    ):
        self.module = module
        self.cls = cls
        self.fn = fn
        self.row_var = row_var
        self.lineno = lineno
        self.spec = spec
        self.reads: Set[str] = set()

    @property
    def key(self) -> str:
        """
        The identity ``KNOWN_UNFIXED`` / ``KNOWN_UNFIXED_COLUMNS`` are keyed by: "module:class",
        with no function in it. Prompt 04b (G5) leaves this exactly as it was, because those two
        lists are prompt 05's interface (README §4) and must keep their keying and their contents;
        the function-qualified identity a coverage pin needs is ``coverage_key`` below instead.
        """
        return f"{self.module}:{self.cls}"

    @property
    def coverage_key(self) -> str:
        """
        The identity ``EXPECTED_COVERAGE`` / ``KNOWN_BLIND`` are keyed by (prompt 04b, G5):
        "module:class.function.variable". Unlike ``key``, this carries the function, so a class's
        ``_build_impl_serial`` and ``_build_impl_model`` -- both of which bind a variable named
        ``row_data`` -- pin as two entries, not one that collapses.
        """
        return f"{self.module}:{self.cls}.{self.fn}.{self.row_var}"

    @property
    def site(self) -> str:
        return f"{self.module}:{self.cls}.{self.fn}.{self.row_var}@{self.lineno}"

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
        finding = Finding(module, cls, fn.name, name, lineno, spec)
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


def iter_functions(tree: ast.Module):
    """
    Every function of every module under ``ObjectFactories/``: ``(class name, FunctionDef)`` for
    a class method, and ``("<module-level>", FunctionDef)`` for a function defined at module
    scope, such as ``OneLoopIntegral.py``'s ``read_batch`` (audit §3.12), never wired to a class.

    Prompt 04b (G4): this replaces the hard-coded name list (``build``, ``read_batch``,
    ``_build_impl_serial``, ``_build_impl_model``) with "every function", so a helper the guard
    has never heard of -- ``validate_on_startup``, which holds a decidable row binding in seven
    factories and runs on every pool open -- is visited without being named here.
    """
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef):
                yield cls.name, fn
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef):
            yield "<module-level>", fn


def inline_nested_row_helper(
    fn: ast.FunctionDef,
) -> Tuple[ast.FunctionDef, Optional[Tuple[str, str, str]]]:
    """
    Return a copy of ``fn`` with its nested one-argument row helper (``make_obj``/``make_object``)
    inlined as a for-loop over the row variable the comprehension that calls it iterates over --
    the one link ``analyse_build`` cannot make on its own, a helper's row parameter to the query it
    is called with.

    Returns ``(fn, None)`` when there is no such nested helper, or when the comprehension that
    calls it cannot be found (so there is nothing to link it to) -- the caller then knows inlining
    linked nothing, as for ``OneLoopIntegral.py``'s module-level ``read_batch``, whose
    ``make_object`` is defined but never called.
    """
    fn = copy.deepcopy(fn)
    nested = [n for n in fn.body if isinstance(n, ast.FunctionDef)]
    if not nested:
        return fn, None
    helper = nested[0]
    if not helper.args.args:
        return fn, None
    param = helper.args.args[0].arg

    # the row variable: the name the comprehension that calls the helper iterates over
    row_var = None
    for node in ast.walk(fn):
        if isinstance(node, ast.ListComp) and isinstance(node.elt, ast.Call):
            if isinstance(node.elt.func, ast.Name) and node.elt.func.id == helper.name:
                row_var = node.generators[0].iter.id
    if row_var is None:
        return fn, None

    fn.body = [n for n in fn.body if n is not helper]
    loop = ast.For(
        target=ast.Name(id=param, ctx=ast.Store()),
        iter=ast.Name(id=row_var, ctx=ast.Load()),
        body=helper.body,
        orelse=[],
        lineno=helper.lineno,
    )
    fn.body.insert(len(fn.body) - 1, loop)
    ast.fix_missing_locations(fn)
    return fn, (helper.name, param, row_var)


def analyse_read_batch(fn: ast.FunctionDef, module: str, cls: str) -> List[Finding]:
    """
    ``read_batch``'s nested row helper is inlined (see ``inline_nested_row_helper``) and the
    result walked exactly as a ``build()`` is.

    When inlining links nothing -- the module-level ``OneLoopIntegral.read_batch`` is the one case
    in this tree -- ``analyse_build`` returns no findings at all, and a hole that produces no
    findings is invisible to ``test_analyser_coverage_is_undiminished`` and
    ``test_blind_spots_are_the_declared_ones`` alike, which is exactly the silently-blind failure
    mode this guard exists to refuse. So a single undecidable Finding is manufactured for the whole
    function, which puts it in ``KNOWN_BLIND`` rather than nowhere.
    """
    inlined, _link = inline_nested_row_helper(fn)
    findings = analyse_build(inlined, module, cls)
    if findings:
        return findings
    blind = Finding(module, cls, "read_batch", "read_batch", fn.lineno, None)
    blind.reads.add("<no row binding could be resolved>")
    return [blind]


def analyse_factories() -> List[Finding]:
    """
    Every row-variable binding under ``ObjectFactories/`` that any function -- of any class, or at
    module level -- reads attributes off.

    Prompt 04b (G4): every function of ``iter_functions`` is visited, not a hard-coded list of
    names. ``read_batch`` goes through ``analyse_read_batch`` (its nested row helper needs
    inlining first); every other function -- ``build``, ``_build_impl_serial``,
    ``_build_impl_model``, ``validate_on_startup``, and anything not yet imagined -- goes through
    ``inline_nested_row_helper`` and then ``analyse_build`` directly. Inlining is a no-op for a
    function with no nested one-argument helper matching the ``make_obj``/``make_object`` shape,
    so nothing changes for a function that never had one. A function that reads no rows off a
    query result -- ``register``, ``add_tag``, and most of a factory's other methods -- yields no
    Finding at all: that is not a hole, only a function this guard has nothing to say about.
    """
    findings = []
    for path in sorted(FACTORY_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for cls, fn in iter_functions(tree):
            if fn.name == "read_batch":
                findings.extend(analyse_read_batch(fn, path.name, cls))
            else:
                inlined, _link = inline_nested_row_helper(fn)
                findings.extend(analyse_build(inlined, path.name, cls))
    return findings


# Defects found by an audit and deliberately carried rather than fixed, each with the campaign
# issue that owns it, as {"module.py:class": [attribute, ...]}. An entry here is a known bug, not
# an exemption: it is deleted in the commit that fixes the factory, and
# test_known_unfixed_entries_are_still_present holds the fix by failing while a stale entry
# remains.
#
# It is empty, and that is the intended steady state. Prompt 01 of prompts/datastore-readback
# opened it with [01-backgroundmodelvalue-hubble] -- sqla_BackgroundModelValue_factory.build()
# reading row_data.Hubble where the column is Hubble_GeV -- which its own §4 forbade it to fix.
# That defect was fixed on 2026-09-22 and the entry struck; see
# Datastore/tests/test_backgroundmodelvalue_roundtrip.py, which now exercises that build().
#
# prompts/datastore-integrity prompt 04 extended the analyser to _build_impl_serial /
# _build_impl_model and read_batch (docs/datastore-integrity-audit.md §3.10, probe_02), and found
# one further defect class, in both WKB value factories: `_build_impl_serial` reads
# `row_data.WKB_criterio` (missing the trailing "n") where the column selected and the table
# column are both `WKB_criterion`. Real -- every hit raises AttributeError -- and was carried
# deliberately rather than fixed there. Prompt 05a fixed both sites (GkWKBIntegration.py:947,
# TkWKBIntegration.py:1001) and struck them here; nothing else in this tree is known-unfixed by
# this guard today.
KNOWN_UNFIXED = {}


# The row-variable bindings the analyser can see and decide, as "module:class.function.variable"
# (prompt 04b, G5: the function joined the identity, so a class's _build_impl_serial and
# _build_impl_model -- both of which bind a variable named row_data -- pin as two entries, not one
# that collapses). Pinned so that a refactor which moves a row read out of the analyser's reach,
# or out from under one particular helper, fails here rather than quietly reducing coverage --
# a static guard that has gone blind still passes, which is the failure mode that costs everything
# to detect. Add to it when a factory gains a checkable function; removing an entry needs a
# reason, recorded in the campaign board.
#
# This is prompt 04's 30 entries (then "module:class.variable", with no function), each carrying
# its function back in, plus the 13 prompt 04b adds: the 7 validate_on_startup bindings (G4;
# previously unvisited), and 6 more of the same "16 sites, one per function" that prompt 04's log
# explained collapsing to 10 under the old identity -- restoring the function distinguishes them
# again, so 04's 10 _build_impl_*/read_batch keys become the 16
# docs/datastore-integrity-audit/lookup_keys/probe_02_readback_guard_by_hand.py lists. 30 + 7 + 6
# = 43. prompts/datastore-integrity/orchestrator/review_04_measurements.py §D computes the same
# 43, and that every one of 04's 30 entries is the projection (drop the function) of at least one
# entry here -- verified again in this prompt's log by the same computation, not by inspection.
EXPECTED_COVERAGE = frozenset(
    {
        "BackgroundModel.py:sqla_BackgroundModelFactory.build.row",
        "BackgroundModel.py:sqla_BackgroundModelFactory.validate_on_startup.model",
        "BackgroundModel.py:sqla_BackgroundModelValue_factory.build.row_data",
        "GkNumericIntegration.py:sqla_GkNumericIntegration_factory.build.row",
        "GkNumericIntegration.py:sqla_GkNumericIntegration_factory.build.row_data",
        "GkNumericIntegration.py:sqla_GkNumericIntegration_factory.validate_on_startup.integration",
        "GkNumericIntegration.py:sqla_GkNumericValue_factory._build_impl_model.row_data",
        "GkNumericIntegration.py:sqla_GkNumericValue_factory._build_impl_serial.row_data",
        "GkNumericIntegration.py:sqla_GkNumericValue_factory.read_batch.row",
        "GkSource.py:sqla_GkSourceValue_factory._build_impl_model.row_data",
        "GkSource.py:sqla_GkSourceValue_factory._build_impl_serial.row_data",
        "GkSource.py:sqla_GkSource_factory.build.row",
        "GkSource.py:sqla_GkSource_factory.build.row_data",
        "GkSource.py:sqla_GkSource_factory.validate_on_startup.obj",
        "GkSourcePolicyData.py:sqla_GkSourcePolicyData_factory.build.row_data",
        "GkWKBIntegration.py:sqla_GkWKBIntegration_factory.build.row",
        "GkWKBIntegration.py:sqla_GkWKBIntegration_factory.build.row_data",
        # prompts/datastore-integrity prompt 07: store()'s consistency check reads the numeric
        # parent's row, or for a pure-WKB row the z_source row
        "GkWKBIntegration.py:sqla_GkWKBIntegration_factory._check_numeric_parent.parent",
        "GkWKBIntegration.py:sqla_GkWKBIntegration_factory._check_numeric_parent.z_source_row",
        "GkWKBIntegration.py:sqla_GkWKBIntegration_factory.validate_on_startup.integration",
        "GkWKBIntegration.py:sqla_GkWKBValue_factory._build_impl_model.row_data",
        "GkWKBIntegration.py:sqla_GkWKBValue_factory._build_impl_serial.row_data",
        "GkWKBIntegration.py:sqla_GkWKBValue_factory.read_batch.row",
        "OneLoopIntegral.py:sqla_OneLoopIntegral_factory.build.row_data",
        "OneLoopIntegral.py:sqla_OneLoopIntegral_factory.read_batch.row_data",
        "QuadSource.py:sqla_QuadSourceValue_factory._build_impl_model.row_data",
        "QuadSource.py:sqla_QuadSourceValue_factory._build_impl_serial.row_data",
        "QuadSource.py:sqla_QuadSource_factory.build.row",
        "QuadSource.py:sqla_QuadSource_factory.build.row_data",
        "QuadSource.py:sqla_QuadSource_factory.validate_on_startup.calc",
        "QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory.build.row_data",
        "QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory.read_batch.row",
        "TkNumericIntegration.py:sqla_TkNumericIntegration_factory.build.row",
        "TkNumericIntegration.py:sqla_TkNumericIntegration_factory.build.row_data",
        "TkNumericIntegration.py:sqla_TkNumericIntegration_factory.validate_on_startup.integration",
        "TkNumericIntegration.py:sqla_TkNumericValue_factory._build_impl_model.row_data",
        "TkNumericIntegration.py:sqla_TkNumericValue_factory._build_impl_serial.row_data",
        "TkWKBIntegration.py:sqla_TkWKBIntegration_factory.build.row",
        "TkWKBIntegration.py:sqla_TkWKBIntegration_factory.build.row_data",
        # prompts/datastore-integrity prompt 07: store()'s consistency check reads the numeric
        # parent's row
        "TkWKBIntegration.py:sqla_TkWKBIntegration_factory._check_numeric_parent.parent",
        "TkWKBIntegration.py:sqla_TkWKBIntegration_factory.validate_on_startup.integration",
        "TkWKBIntegration.py:sqla_TkWKBValue_factory._build_impl_model.row_data",
        "TkWKBIntegration.py:sqla_TkWKBValue_factory._build_impl_serial.row_data",
        "TkWKBIntegration.py:sqla_TkWKBValue_factory.read_batch.row",
        "redshift.py:sqla_redshift_factory.build.row_data",
        "wavenumber.py:sqla_wavenumber_exit_time_factory.build.row_data",
        "wavenumber.py:sqla_wavenumber_factory.build.row_data",
    }
)

# Row variables the analyser sees reads on but cannot decide, with the reason. These are holes,
# listed so that they are visible rather than silently clean, and so that a new one has to be
# looked at by a person instead of appearing as coverage. Keyed the same way as EXPECTED_COVERAGE
# (prompt 04b, G5): "module:class.function.variable".
KNOWN_BLIND = {
    # its 17 reads are off rows[0], where rows came from a nested _build_query(with_grid_identity)
    # helper whose column list is assembled conditionally. The two prompt-14 grid-identity columns
    # are present on one branch and absent on the other, and the reads of them are guarded by
    # hasattr() -- so the shortfall is genuinely branch-dependent and no column-set comparison can
    # decide it. A path-insensitive analyser must decline rather than guess.
    "BackgroundModel.py:sqla_BackgroundModelFactory.build.row_data": (
        "row comes from rows[0] of a conditionally-built query in a nested helper"
    ),
    # Prompt 04 (G1)'s manufactured entry for OneLoopIntegral.py's module-level read_batch --
    # a function with no return statement, whose make_object was defined but never called -- is
    # gone: prompt 05b (P8) made read_batch a method of sqla_OneLoopIntegral_factory with a
    # return, so it is now a decidable coverage entry (EXPECTED_COVERAGE, above) rather than a
    # declared blind spot.
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
            finding.coverage_key for finding in self.findings if finding.decidable
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
            finding.coverage_key for finding in self.findings if not finding.decidable
        }
        self.assertEqual(
            blind,
            set(KNOWN_BLIND),
            "the analyser cannot decide a row read it is not declared blind to. Extend it, or "
            "declare the hole in KNOWN_BLIND with the reason it cannot be closed.",
        )


# ---------------------------------------------------------------------------------------------
# G2 -- column existence (prompts/datastore-integrity prompt 04). A SELECT or an insert naming a
# column the table does not have is not caught by anything above: the read-back guard compares
# reads against requests, never requests against the schema (module docstring, point 6). SQLAlchemy
# only raises on this at query-build time, i.e. only when the code actually runs; on a path nothing
# runs today it is a silent, permanent AttributeError-in-waiting. This resolves every
# ``<var>.c.<name>`` / ``<var>.c["<name>"]`` in the factory sources, and every key of every dict
# passed to an ``inserter``/``value_inserter`` call, against the schema
# ``build_schema(MetaData(), _factories)`` actually builds. It needs no Ray, no engine and no store
# -- ``build_schema`` only declares SQLAlchemy Table objects in memory.
# ---------------------------------------------------------------------------------------------

import sqlalchemy as sqla  # noqa: E402

from Datastore.SQL.Datastore import _factories  # noqa: E402
from Datastore.SQL.schema import build_schema  # noqa: E402

# Column-existence defects found today, each (module, class, function, table, name) exactly as
# ColumnProblem below reports it. Each names the board item that will fix it (prompt 05, board §2);
# a struck entry that has not been fixed is caught by
# test_known_unfixed_columns_are_still_present, mirroring KNOWN_UNFIXED / P6 above.
#
# Prompt 05a fixed and struck the two P6 entries this held (sqla_GkSourceValue_factory reading
# table.c.z_serial where the column is z_source_serial, GkSource.py:766; and
# sqla_QuadSourceValue_factory reading quadsource_table.c.model_id where the column is
# model_serial, QuadSource.py:672). Prompt 05b (P8) struck the last one: sqla_OneLoopIntegral_
# factory.store no longer writes "validated" to OneLoopIntegral.py, whose table has no such
# column -- D3 bounded the fix to this mechanical defect, and the rest (source_serial) is logged
# on the board as [00-oneloop-factory-cannot-round-trip-a-row] for when the target is actually
# built. Empty since; a future entry needs a reason recorded on the campaign board.
ColumnProblem = Tuple[str, str, str, str, str]

KNOWN_UNFIXED_COLUMNS: Set[ColumnProblem] = set()

# Sites where a table variable resolves (it is the factory's own `table`, a `tables["X"]`, or an
# alias of either) but the column name is built at runtime -- either by an f-string, or by a plain
# loop variable ranging over a tuple of literal names declared a few lines above it -- and so
# cannot be resolved to a literal string statically. Structurally undecidable, not a defect. Keyed
# as "module.py:class.function", each naming every dynamic pattern at that site.
KNOWN_BLIND_COLUMNS: Dict[str, str] = {
    "wavenumber.py:sqla_wavenumber_exit_time_factory.build": (
        'table.c[f"z_exit_suph_e{z_offset}"] / table.c[f"z_exit_subh_e{z_offset}"], looped over '
        "WAVENUMBER_EXIT_TIMES_SUPERHORIZON_EFOLDS / _SUBHORIZON_EFOLDS"
    ),
    "QuadSourceIntegral.py:sqla_QuadSourceIntegral_factory.read_batch": (
        'q_exit_tab.c[f"z_exit_suph_e{z_offset}"] / r_exit_tab.c[f"z_exit_suph_e{z_offset}"] / '
        'q_exit_tab.c[f"z_exit_subh_e{z_offset}"] / r_exit_tab.c[f"z_exit_subh_e{z_offset}"], same '
        "loops"
    ),
    # `table.c[name] == payload[name] for name in key_columns`: `name` is a loop variable, not a
    # constant, even though `key_columns` a few lines above it is a tuple of string literals --
    # resolving it would mean tracking a loop variable back to the tuple it ranges over, which
    # this resolver does not do. Both are prompt 01's replica insert-or-verify check (W3).
    "wavenumber.py:sqla_wavenumber_exit_time_factory.store": (
        "table.c[name] for name in key_columns, a tuple of literal column names"
    ),
    "BackgroundModel.py:sqla_BackgroundModelFactory.store": (
        "table.c[name] for name in key_columns, a tuple of literal column names"
    ),
}


# Sites where a `<base>.c.<name>` reference's `base` does not resolve to a schema table at all
# (prompt 04b, G6, review finding 3): `_table_of` returns None, and 04's resolver silently skipped
# these rather than declaring them. Keyed by (function site, the unparsed `base` expression), each
# naming why that base cannot be resolved. `orchestrator/review_04_measurements.py` §C computed 12
# pairs (nine `subquery`, two `query`, one `table`) at prompt 04b; prompt 05b (P1) rewrote both
# `read_batch`es' tag joins onto `table.c.serial`, so the two `query` pairs and the module-level
# `OneLoopIntegral.read_batch`'s `table` pair (which existed only because that function was not a
# method, so `cls_to_table` had no entry for its own `table` parameter -- moot once it became one)
# are gone, leaving the nine `subquery` pairs.
UnresolvedTableSite = Tuple[str, str]

KNOWN_UNRESOLVED_TABLES: Dict[UnresolvedTableSite, str] = {
    # `subquery = <query>.subquery()`: `subquery` is a plain local bound to a `.subquery()` call,
    # which `_table_of` does not follow (it is not a `tables["X"]`, a name already in `env`, or an
    # `.alias(...)`). Every reference is `subquery.c.<name>`, so it can only enlarge what a
    # "missing" check would find, never convict -- structurally the same shape as a dynamically
    # named column (module docstring point 2), just for the table half instead of the name half.
    (
        "GkNumericIntegration.py:sqla_GkNumericValue_factory._build_impl_model",
        "subquery",
    ): "subquery = integration_query.subquery(); _table_of does not follow .subquery()",
    (
        "GkNumericIntegration.py:sqla_GkNumericValue_factory.read_batch",
        "subquery",
    ): "subquery = integration_query.subquery(); _table_of does not follow .subquery()",
    (
        "GkSource.py:sqla_GkSourceValue_factory._build_impl_model",
        "subquery",
    ): "subquery = integration_query.subquery(); _table_of does not follow .subquery()",
    (
        "GkWKBIntegration.py:sqla_GkWKBValue_factory._build_impl_model",
        "subquery",
    ): "subquery = integration_query.subquery(); _table_of does not follow .subquery()",
    (
        "GkWKBIntegration.py:sqla_GkWKBValue_factory.read_batch",
        "subquery",
    ): "subquery = integration_query.subquery(); _table_of does not follow .subquery()",
    (
        "QuadSource.py:sqla_QuadSourceValue_factory._build_impl_model",
        "subquery",
    ): "subquery = integration_query.subquery(); _table_of does not follow .subquery()",
    (
        "TkNumericIntegration.py:sqla_TkNumericValue_factory._build_impl_model",
        "subquery",
    ): "subquery = integration_query.subquery(); _table_of does not follow .subquery()",
    (
        "TkWKBIntegration.py:sqla_TkWKBValue_factory._build_impl_model",
        "subquery",
    ): "subquery = integration_query.subquery(); _table_of does not follow .subquery()",
    (
        "TkWKBIntegration.py:sqla_TkWKBValue_factory.read_batch",
        "subquery",
    ): "subquery = integration_query.subquery(); _table_of does not follow .subquery()",
    # prompts/datastore-integrity prompt 07: the labels-only reader mode selects the one column
    # `numeric_serial` from the lookup's own query, to name the parents when more than one row
    # matches. The same shape as the entries above, written inline
    (
        "GkWKBIntegration.py:sqla_GkWKBIntegration_factory.build",
        "query.subquery()",
    ): "query.subquery().c.numeric_serial; _table_of does not follow .subquery()",
    (
        "TkWKBIntegration.py:sqla_TkWKBIntegration_factory.build",
        "query.subquery()",
    ): "query.subquery().c.numeric_serial; _table_of does not follow .subquery()",
}

# Insert calls G2 does not check because the payload is not a dict literal (prompt 04b, G6, review
# finding 3): `inserter(conn, x)` / `value_inserter(conn, x)` where `x` is a bare name, not an
# `ast.Dict`. Keyed by (module, class-or-"<module-level>", function, the inserter parameter's own
# name -- "inserter" or "value_inserter"), each naming what `x` is bound to and why its keys are
# not checked here. `orchestrator/review_04_measurements.py`'s insert-call walk (this prompt's
# log) finds the same 13 sites.
InsertSite = Tuple[str, str, str, str]

KNOWN_UNCHECKED_INSERTS: Dict[InsertSite, str] = {
    # x is the function's own `payload` parameter, whose keys are the caller's to choose --
    # nothing here says what they are, so there is no literal set to check.
    (
        "BackgroundModel.py",
        "sqla_BackgroundModelFactory",
        "store",
        "inserter",
    ): "inserter(conn, payload); payload is store's own parameter, keys unknown here",
    (
        "wavenumber.py",
        "sqla_wavenumber_exit_time_factory",
        "store",
        "inserter",
    ): "inserter(conn, payload); payload is store's own parameter, keys unknown here",
    # x is built by a call to another function (value_payload(...)), not a literal here.
    (
        "BackgroundModel.py",
        "sqla_BackgroundModelFactory",
        "store",
        "value_inserter",
    ): "value_inserter(conn, value_data); value_data = value_payload(value, store_id), a call",
    # x is a local first bound to a dict literal, then conditionally extended by
    # x["serial"] = payload["serial"]; the literal keys could be checked by tracking that
    # augmentation, but this prompt declares the site rather than adding that tracking.
    (
        "GkSourcePolicy.py",
        "sqla_GkSourcePolicy_factory",
        "build",
        "inserter",
    ): 'insert_data = {"label": ..., ...}, then insert_data["serial"] = ... if present; declared, not resolved',
    (
        "LambdaCDM.py",
        "sqla_LambdaCDM_factory",
        "build",
        "inserter",
    ): 'insert_data = {"name": ..., ...}, then insert_data["serial"] = ... if present; declared, not resolved',
    (
        "QCD_Cosmology.py",
        "sqla_QCDCosmology_factory",
        "build",
        "inserter",
    ): 'insert_data = {"name": ..., ...}, then insert_data["serial"] = ... if present; declared, not resolved',
    (
        "QuadSourcePolicy.py",
        "sqla_QuadSourcePolicy_factory",
        "build",
        "inserter",
    ): 'insert_data = {"label": ..., ...}, then insert_data["serial"] = ... if present; declared, not resolved',
    (
        "integration_metadata.py",
        "sqla_IntegrationSolver_factory",
        "build",
        "inserter",
    ): 'insert_data = {"label": ..., ...}, then insert_data["serial"] = ... if present; declared, not resolved',
    (
        "redshift.py",
        "sqla_redshift_factory",
        "build",
        "inserter",
    ): 'insert_data = {"z": ..., ...}, then insert_data["serial"] = ... if present; declared, not resolved',
    (
        "store_tag.py",
        "sqla_store_tag_factory",
        "build",
        "inserter",
    ): 'insert_data = {"label": ...}, then insert_data["serial"] = ... if present; declared, not resolved',
    (
        "tolerance.py",
        "sqla_tolerance_factory",
        "build",
        "inserter",
    ): 'insert_data = {"log10_tol": ...}, then insert_data["serial"] = ... if present; declared, not resolved',
    (
        "version.py",
        "sqla_version_factory",
        "build",
        "inserter",
    ): 'insert_data = {"label": ...}, then insert_data["serial"] = ... if present; declared, not resolved',
    (
        "wavenumber.py",
        "sqla_wavenumber_factory",
        "build",
        "inserter",
    ): 'insert_data = {"k_inv_Mpc": ..., ...}, then insert_data["serial"] = ... if present; declared, not resolved',
}


def _table_of(expr: ast.AST, env: Dict[str, str]) -> Optional[str]:
    """The schema table name `expr` denotes, resolved the way probe_03 does: the factory's own
    `table`, `tables["X"]`, a plain local bound to one of those, or an `.alias(...)` of either.
    """
    if (
        isinstance(expr, ast.Subscript)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "tables"
        and isinstance(expr.slice, ast.Constant)
    ):
        return expr.slice.value
    if isinstance(expr, ast.Name):
        return env.get(expr.id)
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "alias"
    ):
        return _table_of(expr.func.value, env)
    return None


def _column_ref(node: ast.AST) -> Optional[Tuple[ast.AST, str, bool]]:
    """`(base, name, dynamic)` for a `<base>.c.<name>` / `<base>.c["<name>"]` node, else None.
    `dynamic` is True for `<base>.c[f"..."]`, whose name cannot be read off statically.
    """
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "c"
    ):
        return node.value.value, node.attr, False
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "c"
    ):
        base = node.value.value
        index = node.slice
        if isinstance(index, ast.Constant) and isinstance(index.value, str):
            return base, index.value, False
        return base, "<dynamic>", True
    return None


def column_existence_problems() -> Tuple[
    Set[ColumnProblem],
    Dict[str, Set[str]],
    Dict[UnresolvedTableSite, int],
    Set[InsertSite],
]:
    """
    Four results, over every function of every class and every module-level function (prompt 04b,
    G6: walking module-level functions as well as class methods):

    - every `<var>.c.<name>` / insert-payload key that resolves to a table but not to one of its
      columns, as `(module, class, function, table, name)` -- including a `tables["X"]` or
      `inserters["X"]` naming no schema table at all, reported the same way (review finding 3: 04
      skipped this case rather than reporting it; there are none in this tree today, but the
      mutation-breakage record shows it is caught when there is one);
    - every site where the table resolves but the column name does not (a dynamic f-string), as
      `{"module.py:class.function": {pattern, ...}}`;
    - every `<base>.c.<name>` whose `base` does not resolve to a table at all, as
      `{(function site, base): count}` -- declared in `KNOWN_UNRESOLVED_TABLES` rather than
      silently skipped (review finding 3);
    - every `inserter`/`value_inserter` call whose payload is not a dict literal, as a set of
      `(module, class, function, inserter name)` -- declared in `KNOWN_UNCHECKED_INSERTS` rather
      than silently skipped (review finding 3).
    """
    built = build_schema(sqla.MetaData(), _factories)
    columns = {name: set(t.c.keys()) for name, t in built.tables.items()}
    cls_to_table = {f.__name__: name for name, f in _factories.items()}

    problems: Set[ColumnProblem] = set()
    blind: Dict[str, Set[str]] = {}
    unresolved_tables: Dict[UnresolvedTableSite, int] = {}
    unchecked_inserts: Set[InsertSite] = set()

    for path in sorted(FACTORY_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for cls, fn in iter_functions(tree):
            own = cls_to_table.get(cls)
            env: Dict[str, str] = {}
            if own is not None:
                env["table"] = own

            site = f"{path.name}:{cls}.{fn.name}"

            for node in sorted(
                (n for n in ast.walk(fn) if hasattr(n, "lineno")),
                key=lambda n: (n.lineno, n.col_offset),
            ):
                # `tables["X"]` / `inserters["X"]` naming no schema table is a problem in its own
                # right, reported like a missing column, wherever the literal appears -- not only
                # where it is later used with `.c.` (review finding 3).
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ("tables", "inserters")
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                    and node.slice.value not in columns
                ):
                    problems.add(
                        (
                            path.name,
                            cls,
                            fn.name,
                            f"{node.value.id}[X]",
                            node.slice.value,
                        )
                    )

                if isinstance(node, ast.Assign) and len(node.targets) == 1:
                    tgt = node.targets[0]
                    if isinstance(tgt, ast.Name):
                        t = _table_of(node.value, env)
                        if t is not None:
                            env[tgt.id] = t

                ref = _column_ref(node)
                if ref is None:
                    continue
                base, name, dynamic = ref
                t = _table_of(base, env)
                if t is None:
                    key = (site, ast.unparse(base))
                    unresolved_tables[key] = unresolved_tables.get(key, 0) + 1
                    continue
                if t not in columns:
                    continue
                if dynamic:
                    blind.setdefault(site, set()).add(f"{ast.unparse(node)} ({t})")
                elif name not in columns[t]:
                    problems.add((path.name, cls, fn.name, t, name))

            # insert payload keys: inserter(conn, {...}) / value_inserter(conn, {...}), where
            # value_inserter = inserters["X"] for some literal X
            for node in ast.walk(fn):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id.endswith("inserter")
                    and len(node.args) == 2
                ):
                    continue
                if not isinstance(node.args[1], ast.Dict):
                    unchecked_inserts.add((path.name, cls, fn.name, node.func.id))
                    continue
                target = own
                if node.func.id != "inserter":
                    for a in ast.walk(fn):
                        if (
                            isinstance(a, ast.Assign)
                            and isinstance(a.targets[0], ast.Name)
                            and a.targets[0].id == node.func.id
                            and isinstance(a.value, ast.Subscript)
                            and isinstance(a.value.slice, ast.Constant)
                        ):
                            target = a.value.slice.value
                if target not in columns:
                    continue
                for k in node.args[1].keys:
                    if isinstance(k, ast.Constant) and k.value not in columns[target]:
                        problems.add(
                            (
                                path.name,
                                cls,
                                fn.name,
                                f"{target} [insert key]",
                                k.value,
                            )
                        )

    return problems, blind, unresolved_tables, unchecked_inserts


class TestColumnExistence(unittest.TestCase):
    """G2: every `<table>.c.<name>` and every insert key resolves against the schema
    `build_schema` builds. Shape (b) of the module docstring's point 6."""

    def setUp(self):
        (
            self.problems,
            self.blind,
            self.unresolved_tables,
            self.unchecked_inserts,
        ) = column_existence_problems()

    def test_column_references_resolve_against_schema(self):
        """
        The guard proper. A `<table>.c.<name>` or an insert key naming a column the schema does
        not have raises AttributeError / is silently dropped only when the code actually runs;
        this catches it statically instead.
        """
        self.assertEqual(
            self.problems,
            KNOWN_UNFIXED_COLUMNS,
            "a factory names a column its table does not have. If this is newly broken, correct "
            "the name -- or, if it is a known defect being carried deliberately, move it into "
            "KNOWN_UNFIXED_COLUMNS naming the issue that owns it.",
        )

    def test_known_unfixed_columns_are_still_present(self):
        """A fixed defect must be struck from KNOWN_UNFIXED_COLUMNS, not left to rot there."""
        for entry in KNOWN_UNFIXED_COLUMNS:
            self.assertIn(
                entry,
                self.problems,
                f"{entry} no longer names a column absent from its table: if it has been fixed, "
                "remove its KNOWN_UNFIXED_COLUMNS entry",
            )

    def test_dynamic_column_sites_are_the_declared_ones(self):
        """
        A resolvable table with a runtime-built column name is neither cleared nor convicted. The
        sites that exist are declared above with every dynamic pattern found there; a new one is a
        decision for a person.
        """
        self.assertEqual(
            set(self.blind),
            set(KNOWN_BLIND_COLUMNS),
            "a new site resolves its table but not a literal column name. Declare it in "
            "KNOWN_BLIND_COLUMNS with the patterns found there, or teach the resolver to decide it.",
        )

    def test_unresolved_table_sites_are_the_declared_ones(self):
        """
        G6 (prompt 04b, review finding 3). A `<base>.c.<name>` whose `base` cannot be resolved to
        a table at all is neither cleared nor convicted -- previously it was silently skipped. The
        pairs that exist are declared above with the reason; a new one is a decision for a person.
        """
        self.assertEqual(
            set(self.unresolved_tables),
            set(KNOWN_UNRESOLVED_TABLES),
            "a new (function, base) pair cannot be resolved to a schema table. Declare it in "
            "KNOWN_UNRESOLVED_TABLES with the reason, or teach _table_of to resolve it.",
        )

    def test_unchecked_insert_sites_are_the_declared_ones(self):
        """
        G6 (prompt 04b, review finding 3). An `inserter`/`value_inserter` call whose payload is
        not a dict literal is neither cleared nor convicted -- previously it was silently skipped.
        The sites that exist are declared above with the reason; a new one is a decision for a
        person.
        """
        self.assertEqual(
            self.unchecked_inserts,
            set(KNOWN_UNCHECKED_INSERTS),
            "a new insert call's payload is not a dict literal, so its keys are not checked. "
            "Declare it in KNOWN_UNCHECKED_INSERTS with the reason, or resolve its keys.",
        )
