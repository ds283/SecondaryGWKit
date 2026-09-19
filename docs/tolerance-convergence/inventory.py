"""
Regenerate the table sections of ``docs/tolerance-convergence/TOLERANCE-INVENTORY.md``.

Written for prompt 02 of ``prompts/tolerance-convergence``.

**What this script derives, and what it is told.**

Derived from the tree, so that it cannot go stale silently:

* the **keyed object types** and the columns each one's lookup filters on --- read out of
  ``Datastore/SQL/ObjectFactories/`` by importing the factory classes, calling their ``register()``
  to get the column list, and walking the ``ast`` of their ``build()`` to collect the comparison
  predicates. A hand-typed key list is how this campaign's subject came to be miscounted in the
  first place (``prompts/tolerance-convergence/RECONCILIATION.md`` section 2.1);
* the **values** of the accuracy constants --- by importing them, never by parsing;
* the **hard-coded tolerance literals** at production call sites --- by walking the ``ast`` of a
  declared list of production modules for keyword arguments named ``atol`` / ``rtol`` / ``xtol`` /
  ``epsabs`` / ``epsrel`` / ``phase_atol`` / ``amplitude_rtol`` / ``deriv_rtol`` whose value is a
  numeric literal.

Told, and printed as inputs with their source rather than pretended to be measured:

* the **object counts** of each sector. 50 and ~65,000 come from ``main.py`` and from
  ``prompts/GkTk-remedial/README.md`` section 6; a script that recomputed them would be measuring
  the pipeline rather than inventorying it (prompt 02 section 5).

The judgement columns --- *Reaches*, *The real knob*, *Provenance*, *Owned by* --- are declared in
``PARAMETERS`` below, each with the file and line the judgement was read off. They are findings,
not derivations: the script carries them so that the table regenerates as one piece, and the
document's prose carries the argument for them.

``main.py`` is **never imported** (it parses ``sys.argv``, opens a Ray connection and a
``ShardedPool`` at module scope); it is read with ``ast``. Nothing here needs Ray or a datastore.

Usage, from the repository root::

    PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/inventory.py           # to stdout
    PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/inventory.py --check   # verify
    PYTHONPATH=. ./venv/bin/python docs/tolerance-convergence/inventory.py --write   # rewrite

``--check`` exits non-zero if the generated block in ``TOLERANCE-INVENTORY.md`` differs from what
this script renders, so a later reader can tell at a glance whether the document still describes
the tree.
"""

import argparse
import ast
import inspect
import re
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOCUMENT = Path(__file__).resolve().parent / "TOLERANCE-INVENTORY.md"

BEGIN_MARKER = "<!-- BEGIN GENERATED: docs/tolerance-convergence/inventory.py -->"
END_MARKER = "<!-- END GENERATED -->"


# ---------------------------------------------------------------------------------------------
# 1. the keyed object types, derived from the factory classes
# ---------------------------------------------------------------------------------------------

#: Columns that identify an accuracy parameter when they appear in a lookup predicate.
#:
#: **Widened 2026-09-18 by prompt 06.** The set was written when an accuracy parameter in a lookup
#: key was always a tolerance. Prompt 05 replaced the vestigial ``(atol_serial, rtol_serial)`` pair
#: on the four order-governed targets with the Gauss orders that really set their accuracy, so on
#: the tree this set was first written for, ``BackgroundModel``, ``GkSource``, ``GkWKBIntegration``
#: and ``TkWKBIntegration`` would now drop out of §5.1 altogether -- not because they stopped being
#: keyed on an accuracy parameter but because the parameter stopped being a tolerance, which is the
#: campaign's own result. ``GkSource`` keeps no accuracy column at all after prompt 05 and leaves
#: the table for the right reason; the other three stay, on their orders.
ACCURACY_COLUMNS = frozenset(
    {
        "atol_serial",
        "rtol_serial",
        "log10_tol",
        "Levin_threshold",
        "tau_gauss_order",
        "cs_tau_gauss_order",
        "friction_F_gauss_order",
        "rho_gauss_order",
    }
)

#: Predicate columns that are bookkeeping rather than identity, and are reported separately.
BOOKKEEPING_COLUMNS = frozenset({"serial", "validated"})

_COMPARISON_OPS = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


class Predicate(NamedTuple):
    alias: str
    column: str
    op: str

    def render(self) -> str:
        if self.alias == "table":
            return f"{self.column} {self.op}"
        return f"{self.alias}.{self.column} {self.op}"


def _column_attributes(node: ast.AST) -> List[ast.Attribute]:
    """Every ``<name>.c.<column>`` attribute access inside ``node``."""
    found = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Attribute):
            continue
        parent = child.value
        if not isinstance(parent, ast.Attribute) or parent.attr != "c":
            continue
        if not isinstance(parent.value, ast.Name):
            continue
        found.append(child)
    return found


def _alias_tables(tree: ast.AST) -> Dict[str, str]:
    """Local names bound to ``tables["<name>"].alias(...)``, mapped to the table they alias."""
    aliases: Dict[str, str] = {"table": "<this table>"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "alias"
        ):
            base = value.func.value
            if (
                isinstance(base, ast.Subscript)
                and isinstance(base.value, ast.Name)
                and base.value.id == "tables"
                and isinstance(base.slice, ast.Constant)
            ):
                aliases[target.id] = str(base.slice.value)
    return aliases


def lookup_predicates(factory) -> List[Predicate]:
    """
    The comparison predicates a factory's ``build()`` filters its own table on.

    The heuristic, stated so that it can be checked: every ``ast.Compare`` inside ``build()``
    whose operands mention exactly one ``<alias>.c.<column>`` is a filter on that column; a
    comparison mentioning two is a join condition and is dropped. Only aliases of this object's
    own table and of tables bound through ``tables["..."].alias(...)`` in the same function are
    kept, which drops the secondary queries ``build()`` makes against the tag and value tables.
    """
    try:
        source = textwrap.dedent(inspect.getsource(factory.build))
    except (OSError, TypeError):
        return []
    tree = ast.parse(source)
    aliases = _alias_tables(tree)

    predicates: List[Predicate] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left] + list(node.comparators)
        if sum(1 for operand in operands if _column_attributes(operand)) > 1:
            continue  # a join condition, not a filter
        op = _COMPARISON_OPS.get(type(node.ops[0]), "?")
        for attribute in _column_attributes(node):
            alias = attribute.value.value.id
            if alias not in aliases:
                continue
            predicate = Predicate(alias, attribute.attr, op)
            if predicate.column in BOOKKEEPING_COLUMNS and alias == "table":
                continue
            if predicate not in predicates:
                predicates.append(predicate)
    return predicates


def keyed_object_types() -> List[Tuple[str, List[str], List[Predicate]]]:
    """Every registered table whose lookup predicate set mentions an accuracy column."""
    from Datastore.SQL.Datastore import _factories

    rows = []
    for name, factory in sorted(_factories.items()):
        predicates = lookup_predicates(factory)
        if not any(p.column in ACCURACY_COLUMNS for p in predicates):
            continue
        registration = factory.register()
        columns = [c.name for c in registration.get("columns") or []]
        rows.append((name, columns, predicates))
    return rows


# ---------------------------------------------------------------------------------------------
# 2. the constant values, imported
# ---------------------------------------------------------------------------------------------


def imported_constants() -> Dict[str, object]:
    import importlib

    # importlib, not "from X import Y": several of these packages re-export a *class* under the
    # same name as its module, and ``from ComputeTargets import BackgroundModel`` binds the class.
    defaults = importlib.import_module("config.defaults")
    levin_quadrature = importlib.import_module("AdaptiveLevin.levin_quadrature")
    BackgroundModel = importlib.import_module("ComputeTargets.BackgroundModel")
    QuadSourceIntegral = importlib.import_module("ComputeTargets.QuadSourceIntegral")
    phase_residual = importlib.import_module("ComputeTargets.phase_residual")
    wavenumber = importlib.import_module("CosmologyConcepts.wavenumber")
    LambdaCDM_GenericEOS = importlib.import_module(
        "CosmologyModels.GenericEOS.LambdaCDM_GenericEOS"
    )
    bessel_phase = importlib.import_module("LiouvilleGreen.bessel_phase")
    bessel_near_region = importlib.import_module("LiouvilleGreen.bessel_near_region")
    bessel_tail = importlib.import_module("LiouvilleGreen.bessel_tail")

    wanted = {
        "config.defaults": (
            defaults,
            [
                "DEFAULT_ABS_TOLERANCE",
                "DEFAULT_REL_TOLERANCE",
                # the six prompt 05a shipped; DEFAULT_TK_NUMERIC_ABS_TOLERANCE is one of them
                # and predates it
                "DEFAULT_HEXIT_ABS_TOLERANCE",
                "DEFAULT_HEXIT_REL_TOLERANCE",
                "DEFAULT_GK_NUMERIC_ABS_TOLERANCE",
                "DEFAULT_GK_NUMERIC_REL_TOLERANCE",
                "DEFAULT_TK_NUMERIC_REL_TOLERANCE",
                "DEFAULT_TK_NUMERIC_ABS_TOLERANCE",
                "DEFAULT_QUADRATURE_ATOL",
                "DEFAULT_QUADRATURE_RTOL",
                "DEFAULT_LEVIN_THRESHOLD",
                "DEFAULT_FLOAT_PRECISION",
                "DEFAULT_REDSHIFT_RELATIVE_PRECISION",
            ],
        ),
        "ComputeTargets.BackgroundModel": (
            BackgroundModel,
            [
                "TAU_GAUSS_ORDER",
                "CS_TAU_GAUSS_ORDER",
                "FRICTION_F_GAUSS_ORDER",
                "DERIVATIVE_SPLINE_ORDER",
                "STORED_SAMPLE_SPLINE_ORDER",
                "DERIVATIVE_FIT_PAD_POINTS",
                "DERIVATIVE_FIT_REFINE",
                "DERIVATIVE_FIT_PAD_FLOOR",
                "DERIVATIVE_FIT_PAD_FRACTION",
            ],
        ),
        "ComputeTargets.phase_residual": (
            phase_residual,
            ["RHO_GAUSS_ORDER", "RESIDUAL_WKB_REGION_MARGIN"],
        ),
        "ComputeTargets.QuadSourceIntegral": (
            QuadSourceIntegral,
            ["CHEBYSHEV_ORDER", "BESSEL_ORDER_CHECK_TOL"],
        ),
        "AdaptiveLevin.levin_quadrature": (
            levin_quadrature,
            [
                "DEFAULT_LEVIN_ABSTOL",
                "DEFAULT_LEVIN_RELTOL",
                "DEFAULT_LEVIN_CHEBSHEV_ORDER",
                "DEFAULT_LEVIN_MAX_DEPTH",
            ],
        ),
        "LiouvilleGreen.bessel_phase": (
            bessel_phase,
            [
                "DEFAULT_PHASE_ATOL",
                "DEFAULT_AMPLITUDE_RTOL",
                "SAMPLED_PHASE_FLOOR",
                "SAMPLED_AMPLITUDE_FLOOR",
            ],
        ),
        "LiouvilleGreen.bessel_near_region": (
            bessel_near_region,
            [
                "DEFAULT_PANEL_DEGREE",
                "MIN_PANEL_DEGREE",
                "DEFAULT_MAX_REFINEMENT_PASSES",
            ],
        ),
        "LiouvilleGreen.bessel_tail": (
            bessel_tail,
            ["DEFAULT_CROSSOVER_SAFETY", "TAIL_SERIES_MAX_TERMS"],
        ),
        "CosmologyConcepts.wavenumber": (
            wavenumber,
            [
                "DEFAULT_HEXIT_TOLERANCE",
                "SOURCE_GRID_CONSTRUCTION_VERSION",
                "SOURCE_GRID_MAX_SPACING_FACTOR",
                "SOURCE_GRID_MAX_REFINEMENT",
                "SOURCE_GRID_CUBIC_ERROR_CONST",
                "SOURCE_GRID_CURVATURE_STEP_U",
                "SOURCE_GRID_CURVATURE_FD_STEP_U",
                "SOURCE_GRID_CROSSING_MASK_U",
                "SOURCE_GRID_CONSUMER_TARGET_RAD",
                "SOURCE_GRID_SPLINE_EDGE_INTERVALS",
                "SOURCE_GRID_SPLINE_EDGE_FACTOR",
                "SOURCE_GRID_BREAK_STANDOFF",
                "SOURCE_GRID_BREAK_HALF_WIDTH",
                "SOURCE_GRID_BREAK_REFINEMENT",
                "SOURCE_GRID_MESH_GUARD",
                "SOURCE_GRID_MIN_SEPARATION",
            ],
        ),
        "CosmologyModels.GenericEOS.LambdaCDM_GenericEOS": (
            LambdaCDM_GenericEOS,
            ["DEFAULT_T_Z_SPLINE_SAMPLES", "DEFAULT_T_Z_SPLINE_ORDER"],
        ),
        # a class attribute, not a module constant
        "CosmologyModels.GenericEOS.LambdaCDM_GenericEOS.LambdaCDM_GenericEOS": (
            LambdaCDM_GenericEOS.LambdaCDM_GenericEOS,
            ["T_Z_REPRESENTATION_VERSION"],
        ),
    }

    values: Dict[str, object] = {}
    for module_name, (module, names) in wanted.items():
        for name in names:
            values[f"{module_name}.{name}"] = getattr(module, name)
    return values


def format_value(value) -> str:
    if isinstance(value, bool) or isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = repr(value)
        return text
    return str(value)


# ---------------------------------------------------------------------------------------------
# 3. hard-coded tolerance literals at production call sites, by ast
# ---------------------------------------------------------------------------------------------

#: Keyword names that carry an accuracy request into a numerical method.
TOLERANCE_KEYWORDS = frozenset(
    {
        "atol",
        "rtol",
        "xtol",
        "epsabs",
        "epsrel",
        "phase_atol",
        "amplitude_rtol",
        "deriv_rtol",
    }
)

#: Production trees swept for hard-coded literals. Test trees, ``docs/`` scripts and
#: ``prompts/`` scripts are deliberately out: they measure the pipeline, they are not it.
PRODUCTION_TREES = (
    "AdaptiveLevin",
    "ComputeTargets",
    "CosmologyConcepts",
    "CosmologyModels",
    "Datastore",
    "LiouvilleGreen",
    "MetadataConcepts",
    "Quadrature",
    "RayTools",
    "Units",
    "config",
    "utilities",
)

#: Top-level production modules swept alongside the trees above.
PRODUCTION_FILES = ("main.py",)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return "<call>"


def _literal(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            return None
        return repr(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _literal(node.operand)
        return None if inner is None else f"-{inner}"
    return None


def production_files() -> List[Path]:
    files: List[Path] = []
    for name in PRODUCTION_FILES:
        path = REPO_ROOT / name
        if path.exists():
            files.append(path)
    for tree in PRODUCTION_TREES:
        root = REPO_ROOT / tree
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "tests" in path.parts or "__pycache__" in path.parts:
                continue
            files.append(path)
    for path in sorted(REPO_ROOT.glob("extract_*.py")):
        files.append(path)
    return files


def hard_coded_literals() -> List[Tuple[str, int, int, str, str, str]]:
    """
    ``(path, call line, argument line, called function, keyword, literal)`` per literal request.

    Both line numbers are reported because this campaign's documents cite both conventions:
    ``prompts/background-solver-robustness/PROVENANCE.md`` cites the ``root_scalar(`` line and
    ``prompts/tolerance-convergence`` README section 3.2 cites the ``phase_atol=`` line.
    """
    found = []
    for path in production_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover - the tree does not hold one
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg not in TOLERANCE_KEYWORDS:
                    continue
                literal = _literal(keyword.value)
                if literal is None:
                    continue
                found.append(
                    (
                        relative,
                        node.lineno,
                        keyword.value.lineno,
                        _call_name(node),
                        keyword.arg,
                        literal,
                    )
                )
    found.sort()
    return found


# ---------------------------------------------------------------------------------------------
# 4. object counts, which are inputs and are printed as such
# ---------------------------------------------------------------------------------------------

OBJECT_COUNTS = [
    (
        "`wavenumber_exit_time`",
        "50 per model",
        "`main.py:3584`, `:3596` -- `NUMBER_SOURCE_K_VALUES = 50` and "
        "`NUMBER_RESPONSE_K_VALUES = NUMBER_SOURCE_K_VALUES`, over the same "
        "`logspace(1e5, 3e8)`, so the two samples are the same 50 wavenumbers",
    ),
    (
        "`BackgroundModel`",
        "1 per (cosmology, source grid)",
        "`main.py:1062` -- one `object_get` per model, outside every loop",
    ),
    (
        "`TkNumericIntegration`",
        "50 per model",
        "`main.py:1215`, inside the $k$ loop alone (README section 2 (c))",
    ),
    (
        "`TkWKBIntegration`",
        "50 per model",
        "one object per $k$ (`prompts/GkTk-remedial/README.md` section 6, review section 12.1)",
    ),
    (
        "`GkNumericIntegration`",
        "~65,000 per model",
        "one per $(k, z_{\\rm source})$, `main.py:1770-1791`; the count is "
        "`prompts/GkTk-remedial/README.md` section 6's, **taken on the version-0 source grid** "
        "and not re-taken here",
    ),
    (
        "`GkWKBIntegration`",
        "~65,000 per model",
        "same shape as `GkNumericIntegration` (`prompts/GkTk-remedial/README.md` section 6, "
        "version-0 grid)",
    ),
    (
        "`GkSource`",
        "50 x (response redshifts) per model",
        "one per $(k, z_{\\rm response})$, `main.py:2365`; no absolute count is recorded in the "
        "tree and this prompt measures nothing, so the form is given rather than a number",
    ),
    (
        "`QuadSourceIntegral`",
        "1,275 x 50 x (response redshifts) per model",
        "one per $(k, q, r, z_{\\rm response})$; the $(q, r)$ multiplicity is "
        "`itertools.combinations_with_replacement` over the 50 source wavenumbers "
        "(`main.py:1698`) = 1,275 pairs",
    ),
    (
        "`GkSourcePolicy`, `QuadSourcePolicy`",
        "2 each per run",
        "`main.py:3542`, `:3548`, `:3560`, `:3566` -- the 1.5 and 5.0 threshold variants",
    ),
    (
        "`OneLoopIntegral`",
        "0",
        "the table is registered (`Datastore/SQL/Datastore.py:120`) and sharded "
        "(`config/sharding.py:34`), and `main.py` never builds one",
    ),
]


# ---------------------------------------------------------------------------------------------
# 5. the inventory itself
# ---------------------------------------------------------------------------------------------


class Parameter(NamedTuple):
    name: str  #: parameter, with its defining file:line
    value: (
        str  #: "@<key>" resolves against imported_constants(); anything else is literal
    )
    keys: str  #: the datastore types whose lookup key contains it, or "none"
    reaches: str  #: solver / lookup key / both / neither
    method: str  #: the numerical method it feeds
    knob: str  #: the real knob for this quantity
    count: str  #: object count of the sector
    provenance: str  #: campaign, prompt and log -- or "never chosen"
    owner: str  #: this campaign's prompt, another campaign, or nobody


PARAMETERS: Dict[str, List[Parameter]] = {
    "A. The `config/defaults.py` accuracy constants": [
        Parameter(
            name="`DEFAULT_ABS_TOLERANCE` (`config/defaults.py:23`)",
            value="@config.defaults.DEFAULT_ABS_TOLERANCE",
            keys="**none, since prompt 05a (2026-09-18).** It keyed seven object types until "
            "prompt 05 removed the pair from the four that never used it and prompt 05a gave "
            "each of the three that did a constant of its own",
            reaches="**neither.** No lookup key and no solver. What is left of it is **seven bare "
            "float comparisons** and a set of signature defaults production always overrides",
            method="`fabs(a - b) < DEFAULT_ABS_TOLERANCE` at `ComputeTargets/GkSource.py:96`, "
            "`:104`, `:275`; `Quadrature/integrators/numeric_with_phase_cut.py:618`, `:737`, "
            "`:790`; `LiouvilleGreen/WKBtools.py:83`. Plus the `atol` default of "
            "`numeric_with_phase_cut.integrate_numeric_with_phase_cut` (`:536`), which every "
            "production call overrides",
            knob="not a solver tolerance at all. Comparing two redshifts is a different quantity "
            "wearing a tolerance's name -- `config/defaults.py:14-22` says so at the point of use",
            count="7 comparison sites; 0 objects keyed",
            provenance="**never chosen**, and its provenance cannot be established from the "
            "record: no campaign document, log or code comment records a measurement behind "
            "`1e-10`. It is now the epsilon of "
            "`[02-shared-atol-doubles-as-a-float-comparison-epsilon]`, which prompt 05a left open "
            "deliberately: renaming it would reach seven sites in three modules and would relabel "
            "the defect rather than repair it",
            owner="`[02-shared-atol-doubles-as-a-float-comparison-epsilon]`, open and unassigned. "
            "**Not to be retuned as a solver tolerance by anyone**",
        ),
        Parameter(
            name="`DEFAULT_REL_TOLERANCE` (`config/defaults.py:24`)",
            value="@config.defaults.DEFAULT_REL_TOLERANCE",
            keys="**none, since prompt 05a.** It keyed eight object types",
            reaches="**neither**",
            method="none. It survives as the `rtol` signature default of "
            "`numeric_with_phase_cut.integrate_numeric_with_phase_cut`, which production always "
            "overrides, and has no float-comparison use",
            knob="none",
            count="0 objects keyed",
            provenance="**never chosen.** `1e-8` is the value the pipeline was written with. Its "
            "three former consumers are now `DEFAULT_HEXIT_REL_TOLERANCE` (changed to 1e-9), "
            "`DEFAULT_GK_NUMERIC_REL_TOLERANCE` (kept at 1e-8) and "
            "`DEFAULT_TK_NUMERIC_REL_TOLERANCE` (changed to 3e-11), each with its own measurement",
            owner="nobody. Recorded so that `docs/TOLERANCE-PROVENANCE.md` (prompt 06a) can say "
            "in those words that the value was never chosen and now sets nothing",
        ),
        Parameter(
            name="`DEFAULT_HEXIT_REL_TOLERANCE` (`config/defaults.py:53`)",
            value="@config.defaults.DEFAULT_HEXIT_REL_TOLERANCE",
            keys="`wavenumber_exit_time` -- **by inequality, not equality** (section 2.3)",
            reaches="**both**",
            method="`scipy.optimize.root_scalar`, Brent, bracketed in $u = \\log(1+z)$, as `rtol` "
            "(`CosmologyConcepts/wavenumber.py:1017-1022`)",
            knob="the pair. Brent stops at `xtol + rtol|u|`, so at the largest production "
            "$|u| = 38.04$ this guarantees a displacement of 3.81e-08",
            count="50 per model, each solved at 1 + 3 superhorizon + 5 subhorizon offsets",
            provenance="**chosen**, and **changed** from the shared `1e-8`. "
            "`prompts/tolerance-convergence` prompt 03a, "
            "`docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` section 7.6, **version-2 "
            "grid at each cosmology's own anchor**; accepted by the user 2026-09-17 (D1, closed) "
            "**on the guarantee reading** -- on the achieved-displacement reading the same "
            "measurement gives `unchanged`. The measurement is at `config/defaults.py:26-52`",
            owner="settled (README section 7 D1); shipped by prompt 05a",
        ),
        Parameter(
            name="`DEFAULT_HEXIT_ABS_TOLERANCE` (`config/defaults.py:63`)",
            value="@config.defaults.DEFAULT_HEXIT_ABS_TOLERANCE",
            keys="`wavenumber_exit_time`",
            reaches="**both**",
            method="the same `root_scalar`, as `xtol`",
            knob="the pair, and it is **coupled**: `xtol = 1e-10` floors the pair at ~1e-10 "
            "relative however far `rtol` is tightened, taking over below `rtol ~ 2.6e-12`",
            count="as `DEFAULT_HEXIT_REL_TOLERANCE`",
            provenance="**inert and unchosen**, and README section 1.2's closing rule applies: "
            "the provenance of `1e-10` cannot be established from the record. It is "
            "`DEFAULT_ABS_TOLERANCE`'s value, inherited. Prompt 03a measured it binding at **0 of "
            "150** (k, offset) pairs on each of three models",
            owner="settled at its inherited value (D1); shipped by prompt 05a",
        ),
        Parameter(
            name="`DEFAULT_GK_NUMERIC_REL_TOLERANCE` (`config/defaults.py:86`)",
            value="@config.defaults.DEFAULT_GK_NUMERIC_REL_TOLERANCE",
            keys="`GkNumericIntegration`",
            reaches="**both**",
            method="DOP853, `ComputeTargets/GkNumericIntegration.py:380`",
            knob="the pair, and `rtol` is the whole lever: four decades move the maximum "
            "envelope-relative error by x13,300",
            count="29,290 / 38,105 / 58,350 per model on the version-2 grid",
            provenance="**chosen**, value **unchanged** at `1e-8`. "
            "`prompts/tolerance-convergence` prompt 03, "
            "`docs/tolerance-convergence/GK-NUMERIC-SWEEP.md` sections 5.2 and 7, 50 $k$ x 3 "
            "models x 15 cells, **version-2 grid at each cosmology's own anchor**; accepted by "
            "the user 2026-09-17 under README section 6.1 rule 4, the consumer's own spline "
            "carrying **x631 to x37,700** the solver's error",
            owner="settled (D1); shipped by prompt 05a",
        ),
        Parameter(
            name="`DEFAULT_GK_NUMERIC_ABS_TOLERANCE` (`config/defaults.py:93`)",
            value="@config.defaults.DEFAULT_GK_NUMERIC_ABS_TOLERANCE",
            keys="`GkNumericIntegration`",
            reaches="**both**",
            method="DOP853, `ComputeTargets/GkNumericIntegration.py:379`",
            knob="nothing: $|G|$ is 2.1e+12 to 2.9e+18 in `Mpc_units`, so an absolute floor of "
            "`1e-10` cannot bind",
            count="as `DEFAULT_GK_NUMERIC_REL_TOLERANCE`",
            provenance="**inert and unchosen** -- README section 1.2's closing rule again. Four "
            "decades of it move the maximum by at most **1.2 %** (prompt 03), and the record does "
            "not say who chose `1e-10` or for what",
            owner="settled at its inherited value (D1); shipped by prompt 05a",
        ),
        Parameter(
            name="`DEFAULT_TK_NUMERIC_REL_TOLERANCE` (`config/defaults.py:123`)",
            value="@config.defaults.DEFAULT_TK_NUMERIC_REL_TOLERANCE",
            keys="`TkNumericIntegration`",
            reaches="**both**",
            method="DOP853, `ComputeTargets/TkNumericIntegration.py:413`",
            knob="the pair, and `rtol` sets the level while `atol` selects which wavenumber "
            "excurses",
            count="50 per model",
            provenance="**chosen**, and **changed** from the shared `1e-8`. "
            "`prompts/tolerance-convergence` prompt 03a, "
            "`docs/tolerance-convergence/TK-NUMERIC-AND-EXIT-TIME.md` sections 4.1 and 5, three "
            "models, all fifty wavenumbers, **version-2 grid at each cosmology's own anchor** "
            "under `BREAK_POINT_ALL`; the loosest of nine settings whose maximum (3.88e-08) "
            "clears the re-measured 2.39e-06 initial-condition floor, at **+39.4 %** of the "
            "sector's evaluations. Accepted 2026-09-17. **A measured setting and not a bound** -- "
            "the maximum is not monotone in `rtol` "
            "(`[03a-tk-numeric-excursion-is-sporadic-in-rtol]`, open)",
            owner="settled (D1); shipped by prompt 05a",
        ),
        Parameter(
            name="`DEFAULT_TK_NUMERIC_ABS_TOLERANCE` (`config/defaults.py:157`)",
            value="@config.defaults.DEFAULT_TK_NUMERIC_ABS_TOLERANCE",
            keys="`TkNumericIntegration` alone",
            reaches="**both**",
            method="DOP853, via `TkNumericIntegration.py:412` and "
            "`Quadrature/integrators/numeric_with_phase_cut.py`",
            knob="**a step-selection knob, not the level.** Across `1e-12` to `1e-14` it moves "
            "the median of the per-$k$ maxima by at most 2.1x and the maximum by up to **205x**, "
            "by changing which wavenumber draws a bad step sequence",
            count="50 per model",
            provenance="`prompts/GkTk-remedial` prompt 12, confirmed against the production grid "
            "by prompt 17 and **settled by the user 2026-09-12** (**version-0 grid**); "
            "**re-characterised, value unchanged**, by `prompts/tolerance-convergence` prompt 03a "
            "on the version-2 grid. The measurement is in `config/defaults.py:125-156`",
            owner="settled; not reopened (README section 7 D1)",
        ),
        Parameter(
            name="`DEFAULT_QUADRATURE_ATOL` (`config/defaults.py:166`)",
            value="@config.defaults.DEFAULT_QUADRATURE_ATOL",
            keys="`QuadSourceIntegral`",
            reaches="**both**",
            method="`adaptive_levin_sincos` (Levin / Clenshaw-Curtis, "
            "`ComputeTargets/QuadSourceIntegral.py:1059`) and `scipy.quad` / DOP853 via "
            "`Quadrature/simple_quadrature.py`; distributed per sub-interval by log-width at "
            "`QuadSourceIntegral.py:819` and per phase group at `:1054`. It also reaches "
            "`analytic_integral`, so the stored `analytic_rad` column depends on it",
            knob="**inert at this value.** `prompts/tolerance-convergence` prompt 06 measured it "
            "binding only at `1e-16` and looser; twenty-eight decades below that it changes "
            "nothing, $|total|$ being 9.3e-13 to 2.3e-08 in the fixture",
            count="1,275 x 50 x (response z) per model",
            provenance="`prompts/source-remediation` prompt 12, against the analytic oracle, on a "
            "**live run**: raised from `1e-25`, where 58 % of work items met their tolerance "
            "before doing any work. The measurement is in `config/defaults.py:160-165`. "
            "**Re-measured read-only by prompt 06**, `QUADSOURCE-READONLY.md` sections 4 and 8: "
            "`unchanged`, the representation floor dominating by **x3.46e+04 to x2.15e+11**",
            owner="`prompts/levin-refactor` / `prompts/qsi-phase-groups`; prompt 06 measured "
            "read-only (README section 0.4) and handed over",
        ),
        Parameter(
            name="`DEFAULT_QUADRATURE_RTOL` (`config/defaults.py:159`)",
            value="@config.defaults.DEFAULT_QUADRATURE_RTOL",
            keys="`QuadSourceIntegral`",
            reaches="**both**",
            method="as above",
            count="1,275 x 50 x (response z) per model",
            knob="**the binding half of the pair**, and the only parameter of this sector that "
            "moves the answer: about a decade of quadrature error per decade of tolerance "
            "(`QUADSOURCE-READONLY.md` section 5)",
            provenance="`prompts/source-remediation` prompt 12 found it **did not bind** -- 1e-8 "
            "to 1e-11 bit-identical on 159 live items -- but that was measured at "
            "`atol = 1e-25`, where `atol` bound. **That finding does not transfer to this tree**: "
            "prompt 06 measured the same three decades moving `total` by a factor of 274 at "
            "`atol = 1e-32`. The value is nevertheless `unchanged`, because the representation "
            "floor is orders above both ends (`QUADSOURCE-READONLY.md` sections 5, 7 and 8)",
            owner="as above",
        ),
        Parameter(
            name="`DEFAULT_LEVIN_THRESHOLD` (`config/defaults.py:45`)",
            value="@config.defaults.DEFAULT_LEVIN_THRESHOLD",
            keys="`GkSourcePolicy` and `QuadSourcePolicy` -- **only as a default that production "
            "never takes**",
            reaches="**neither.** Nothing reads `Levin_threshold` off a policy object, and "
            "`main.py:3542`, `:3548`, `:3560`, `:3566` pass 1.5 and 5.0 explicitly, so the "
            "default value 1.0 never reaches the datastore either",
            method="none. The Levin/direct decision is made per region by "
            "`adaptive_levin_sincos`'s own total-variation gate "
            "(`MetadataConcepts/QuadSourcePolicy.py:25-36`)",
            knob="none for `QuadSourcePolicy` (the driver's gate); for `GkSourcePolicy` the two "
            "shipped values 1.5 and 5.0 are the two production policies, and they are not this "
            "constant",
            count="2 policy objects each per run; `GkSourcePolicyData` is keyed through them",
            provenance="**never chosen**, and never used. README section 1.2 lists it among the "
            "parameters nobody has chosen; this inventory adds that nothing takes the default",
            owner="nobody. Recorded for `docs/TOLERANCE-PROVENANCE.md` (prompt 06)",
        ),
    ],
    "B. The integer orders and the region margin": [
        Parameter(
            name="`TAU_GAUSS_ORDER` (`ComputeTargets/BackgroundModel.py:34`)",
            value="@ComputeTargets.BackgroundModel.TAU_GAUSS_ORDER",
            keys="**`BackgroundModel`, since prompt 05 (2026-09-18)** -- `tau_gauss_order`, an "
            "equality predicate of `build()` (section 5.1). It also reaches the "
            "`IntegrationSolver` label and `stepping`, which is stored and not filtered on",
            reaches="**both** -- it is the knob *and* it is now in the key",
            method="Gauss-Legendre cumulative table, order 4 per interval, split at the "
            "cosmology's break points (`ComputeTargets.cumulative_table.CumulativeTable`)",
            knob="itself",
            count="1 `BackgroundModel` per (cosmology, grid); every target reads its output",
            provenance="`prompts/GkTk-remedial` prompt 02, **superseded**: re-measured by "
            "`prompts/tolerance-convergence` prompt 04, "
            "`docs/tolerance-convergence/ORDER-AUDIT.md`, on the corrected background and the "
            "3-point break set over the **version-2 grid at each cosmology's own anchor**, and "
            "written into `ComputeTargets/tests/wkb_reference_data.json`'s `convergence` block by "
            "prompt 04b on 2026-09-18. `unchanged` at 4, the double-precision accumulation floor "
            "(2.16e-16) dominating order 2 by **x3.03e5**",
            owner="settled by prompts 04, 04b and 05; no longer open",
        ),
        Parameter(
            name="`CS_TAU_GAUSS_ORDER` (`ComputeTargets/BackgroundModel.py:42`)",
            value="@ComputeTargets.BackgroundModel.CS_TAU_GAUSS_ORDER",
            keys="**`BackgroundModel`** -- `cs_tau_gauss_order`, an equality predicate",
            reaches="**both**",
            method="Gauss-Legendre cumulative table for $c_s\\tau$",
            knob="itself",
            count="as `TAU_GAUSS_ORDER`",
            provenance="as `TAU_GAUSS_ORDER`; `unchanged` at 4 against an accumulation floor of "
            "3.30e-16, dominating order 2 by **x1.99e5**",
            owner="settled by prompts 04, 04b and 05",
        ),
        Parameter(
            name="`FRICTION_F_GAUSS_ORDER` (`ComputeTargets/BackgroundModel.py:43`)",
            value="@ComputeTargets.BackgroundModel.FRICTION_F_GAUSS_ORDER",
            keys="**`BackgroundModel`** -- `friction_F_gauss_order`, an equality predicate",
            reaches="**both**",
            method="Gauss-Legendre cumulative table for $F$",
            knob="itself",
            count="as `TAU_GAUSS_ORDER`",
            provenance="as `TAU_GAUSS_ORDER`; `unchanged` at 4 against an accumulation floor of "
            "8.01e-14, dominating order 2 by **x48.4**",
            owner="settled by prompts 04, 04b and 05",
        ),
        Parameter(
            name="`RHO_GAUSS_ORDER` (`ComputeTargets/phase_residual.py:90`)",
            value="@ComputeTargets.phase_residual.RHO_GAUSS_ORDER",
            keys="**`GkWKBIntegration` and `TkWKBIntegration`, since prompt 05** -- "
            "`rho_gauss_order`, an equality predicate of both `build()`s",
            reaches="**both**",
            method="Gauss-Legendre panels for the Liouville-Green phase residual $\\rho$, per "
            "$(model, k, sector)$",
            knob="itself",
            count="~65,000 `GkWKBIntegration` + 50 `TkWKBIntegration` per model",
            provenance="`prompts/GkTk-remedial` prompts 02 and 06 at three wavenumbers, "
            "**superseded** by prompt 04's measurement at **all fifty** in both sectors "
            "(`ORDER-AUDIT.md`), which found the three right and lucky by only x1.5-x1.7. "
            "`unchanged` at 4, the $\\rho$ quadrature floor (6.51e-17 rad) dominating order 2 by "
            "**x2.35e6**. Prompt 05b then made the *recorded* order the order an object was built "
            "at, on both the computed and the rehydrated path",
            owner="settled by prompts 04, 04b, 05 and 05b",
        ),
        Parameter(
            name="`RESIDUAL_WKB_REGION_MARGIN` (`ComputeTargets/phase_residual.py:251`)",
            value="@ComputeTargets.phase_residual.RESIDUAL_WKB_REGION_MARGIN",
            keys="**none, anywhere** -- not in a key, a label or a tag, and prompt 05 "
            "deliberately did not make it a column",
            reaches="**neither**",
            method="sets the band `residual_node_range` covers, by requiring the "
            "Liouville-Green frequency to stay this fraction of its leading term. Re-used by "
            "`main.source_grid_spacing_profile` as the band the version-2 density criterion runs "
            "over",
            knob="itself",
            count="both WKB sectors, and (through the re-use) the source grid every target "
            "shares",
            provenance="**never chosen against an accuracy criterion, and it has none** -- "
            "prompt 04, `ORDER-AUDIT.md` section 7.2, applied README section 6.1 rule 6 and found "
            "no accuracy floor to target, because the residual a producer reads is a `delta` "
            "between two fixed redshifts. What it has is a measured *reachability* bound, cleared "
            "at every margin from 0.05 to **0.9** on three models and both sectors with the "
            "residual **bit-identical** throughout. That bit-identity is what let prompt 05 leave "
            "it out of the key",
            owner="settled as `unchanged` by prompt 04; the re-use is "
            "`[01-density-criterion-imposed-outside-the-wkb-region]`, measured by prompt 04b "
            "section 8 and still open",
        ),
    ],
    "C. Root solves in production code": [
        Parameter(
            name="`_solve_horizon_exit` `xtol`/`rtol` "
            "(`CosmologyConcepts/wavenumber.py:1017-1022`)",
            value="its own pair since prompt 05a: `DEFAULT_HEXIT_ABS_TOLERANCE` / "
            "`DEFAULT_HEXIT_REL_TOLERANCE` (section A)",
            keys="`wavenumber_exit_time` -- **by inequality, not by equality** (section 2)",
            reaches="**both**",
            method="`scipy.optimize.root_scalar`, Brent, bracketed in $u = \\log(1+z)$ by a "
            "geometric widening search, with an acceptance guard "
            "`|q_root| <= DEFAULT_HEXIT_TOLERANCE`",
            knob="the pair. `atol` is absolute in $u$ and `rtol` relative to a root of size ~38, "
            "so neither is comparable with either sector's (README section 2 (e))",
            count="50 per model, each solved at 1 + 3 superhorizon + 5 subhorizon offsets",
            provenance="**measured, at last**, by `prompts/tolerance-convergence` prompt 03a "
            "(T6): `xtol` binds at **0 of 150** (k, offset) pairs on every model and `rtol` is "
            "what fixes the anchor. Decoupled and settled -- see section A for each half's own "
            "provenance, which differs",
            owner="settled (D1, closed 2026-09-17); shipped by prompt 05a",
        ),
        Parameter(
            name="`DEFAULT_HEXIT_TOLERANCE` (`CosmologyConcepts/wavenumber.py:884`)",
            value="@CosmologyConcepts.wavenumber.DEFAULT_HEXIT_TOLERANCE",
            keys="none",
            reaches="**neither** -- it is an acceptance guard on the root, and the bracket "
            "search's own step criterion",
            method="raises if $|q(z_{\\rm root})| > 10^{-2}$ after `root_scalar` reports "
            "convergence (`wavenumber.py:993`)",
            knob="not an accuracy knob: it is eight orders looser than the `xtol` it guards, so "
            "it can only catch a gross failure",
            count="50 per model",
            provenance="**never chosen**; no comment or log records it",
            owner="nobody. Recorded so that prompt 03 does not mistake it for the solve's error "
            "bound",
        ),
        Parameter(
            name="`find_phase_extremum` `xtol=1e-6, rtol=1e-4` "
            "(`LiouvilleGreen/integration_tools.py:95-96`)",
            value="`xtol = 1e-6`, `rtol = 1e-4`",
            keys="none directly -- it fixes the numeric stop point $z_{\\rm init}$, which **is** "
            "a key column (`TkNumericIntegration.z_init_serial`, `GkWKBIntegration.z_init`)",
            reaches="**solver**, and a key through the redshift it returns",
            method="`root_scalar`, bracketed, on the derivative of the numeric solution, after a "
            "sampled sign-change search at `PHASE_STEPS_PER_CYCLE = 16` steps per cycle",
            knob="the pair, and the sampling step that brackets it",
            count="once per `TkNumericIntegration` (50 per model) and once per "
            "`GkNumericIntegration` run in stop mode",
            provenance="**never chosen.** `[11-stop-point-root-tolerance]`, "
            "`docs/OPEN_ISSUES.md` section 1.1, owned by the hand-over campaign. "
            "`LiouvilleGreen/bessel_phase.py:48` records a case where exactly this pair produced "
            "an artefact elsewhere",
            owner="the hand-over campaign. **Recorded here and not retuned** "
            "(README section 0.5)",
        ),
        Parameter(
            name="`_solve_T_z` `xtol=1e-300, rtol=1e-14` "
            "(`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:636`)",
            value="`xtol = 1e-300`, `rtol = 1e-14`",
            keys="not itself; the *representation version* it belongs to is "
            "(`QCD_Cosmology.T_z_representation`, currently "
            "@CosmologyModels.GenericEOS.LambdaCDM_GenericEOS.LambdaCDM_GenericEOS."
            "T_Z_REPRESENTATION_VERSION)",
            reaches="**solver**, and a key through the version",
            method="`root_scalar`, Brent, bracketed between two analytic bounds with a straddle "
            "check at `:621`",
            count="~3,176 calls per `QCD_Cosmology` construction, once per run",
            knob="the pair",
            provenance="`prompts/qcd-background-audit` prompt 04 (`71b842a`), "
            "`logs/04-tighten-node-solve.md`. **Lifted from "
            "`prompts/background-solver-robustness/PROVENANCE.md` section 1, not re-derived**",
            owner="`prompts/qcd-background-audit`; settled",
        ),
        Parameter(
            name="`_find_rho_equality` `xtol=1e-300, rtol=8.9e-16` "
            "(`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:1137`)",
            value="`xtol = 1e-300`, `rtol = 8.9e-16`",
            keys="**yes, indirectly**: the two equality redshifts are production source-grid "
            "sample locations, so they reach the `BackgroundModel` lookup key through the grid "
            "digest (`PROVENANCE.md` section 3.1)",
            reaches="**both**",
            method="`root_scalar`, Brent, bracketed by a $\\sqrt2$ expansion in $1+z$ clamped to "
            "the $T(z)$ representation's bounds",
            count="2 per model construction, once per run",
            knob="the pair; `rtol` is Brent's own $4\\varepsilon$ floor",
            provenance="`prompts/background-solver-robustness` prompt 02, user decision "
            "2026-09-16. **Lifted from that campaign's `PROVENANCE.md` section 3**",
            owner="`prompts/background-solver-robustness`; settled",
        ),
        Parameter(
            name="`temperature_crossing_log1pz` `xtol=1e-15, rtol=1e-15` "
            "(`CosmologyModels/tests/T_z_reference.py:285`)",
            value="`xtol = 1e-15`, `rtol = 1e-15`",
            keys="none",
            reaches="**neither.** Zero production calls -- prompt 07 of "
            "`prompts/qcd-background-audit` took it off the production path and prompt 04 of "
            "`prompts/background-solver-robustness` moved it out of the production class",
            method="`root_scalar`, Brent, bracketed -- on a residual that **need not have a "
            "root**, because the $T(z)$ representation is segmented at exactly these "
            "temperatures",
            knob="none; there is no root for a tolerance to converge to",
            count="0 in the pipeline; 3 per run of one `ComputeTargets` test",
            provenance="**never chosen.** Introduced by `prompts/GkTk-remedial` prompt 03 "
            "without a measurement. **Lifted from "
            "`prompts/background-solver-robustness/PROVENANCE.md` section 2**",
            owner="nobody -- it is test machinery. Listed because README section 3.2's original "
            "anchor `LambdaCDM_GenericEOS.py:864` pointed here",
        ),
    ],
    "D. The Bessel phase representation": [
        Parameter(
            name="`phase_atol=1e-12` (`main.py:1119`, `:1127`)",
            value="`1e-12`",
            keys="none",
            reaches="**solver** -- the construction budget of the two Bessel phase objects",
            method="two-region construction: a sampled, branch-tracked near region of "
            "Chebyshev-like panels refined until the budget is met "
            "(`LiouvilleGreen/bessel_near_region.py:1004`) and a closed-form asymptotic tail "
            "whose crossover is placed from the same budget (`bessel_tail.py:481`)",
            knob="the budget pair; the panel degree and initial width are subordinate to it",
            count="2 objects per run (`nu = 1/2 + b` and `5/2 + b`), consumed by every "
            "`QuadSourceIntegral`",
            provenance="chosen **by argument, not by sweep**, in the comment at "
            "`main.py:1105-1114`: one order tighter than the 1e-11 "
            "`prompts/transfer-remedial` accepts, with the declared errors quoted (5.0e-13 rad "
            "at $\\nu = 5/2$, 3.0e-13 relative amplitude) and the observation that another order "
            "buys almost nothing",
            owner="`prompts/transfer-remedial` owns the representation; prompt 06 records it in "
            "`docs/TOLERANCE-PROVENANCE.md`",
        ),
        Parameter(
            name="`amplitude_rtol=1e-12` (`main.py:1120`, `:1128`)",
            value="`1e-12`",
            keys="none",
            reaches="**solver**",
            method="as above",
            knob="the budget pair",
            count="as above",
            provenance="as above; the comment records that the realised 3.0e-13 is set by the "
            "scaled-Hankel sampling floor rather than by the request",
            owner="as above",
        ),
        Parameter(
            name="`DEFAULT_PHASE_ATOL` (`LiouvilleGreen/bessel_phase.py:134`)",
            value="@LiouvilleGreen.bessel_phase.DEFAULT_PHASE_ATOL",
            keys="none",
            reaches="**neither in production** -- `main.py` overrides it at both call sites",
            method="as above, when a caller supplies nothing",
            knob="n/a in production",
            count="0 in the pipeline",
            provenance="`prompts/transfer-remedial`'s low-order acceptance target, recorded at "
            "`bessel_phase.py:128-133`",
            owner="`prompts/transfer-remedial`",
        ),
        Parameter(
            name="`DEFAULT_AMPLITUDE_RTOL` (`LiouvilleGreen/bessel_phase.py:137`)",
            value="@LiouvilleGreen.bessel_phase.DEFAULT_AMPLITUDE_RTOL",
            keys="none",
            reaches="**neither in production**",
            method="as above",
            knob="n/a in production",
            count="0 in the pipeline",
            provenance="as `DEFAULT_PHASE_ATOL`",
            owner="`prompts/transfer-remedial`",
        ),
        Parameter(
            name="`bessel_phase(atol=, rtol=)` (`LiouvilleGreen/bessel_phase.py:874-875`)",
            value="`None`, and **deprecated and ignored**",
            keys="none",
            reaches="**neither.** They describe an ODE that no longer exists; supplying them "
            "raises a `DeprecationWarning` and maps to nothing (`:932-945`)",
            method="none",
            knob="`phase_atol` / `amplitude_rtol`",
            count="0",
            provenance="n/a -- the pair is a compatibility shim. Listed because a keyword sweep "
            "for `atol=` finds it and misses the live budgets (prompt 02 section 2)",
            owner="nobody",
        ),
    ],
    "E. The Levin driver": [
        Parameter(
            name="`CHEBYSHEV_ORDER` (`ComputeTargets/QuadSourceIntegral.py:71`)",
            value="@ComputeTargets.QuadSourceIntegral.CHEBYSHEV_ORDER",
            keys="**none** -- `QuadSourceIntegral` stores the *achieved* "
            "`WKB_Levin_chebyshev_min_order` as a diagnostic column but does not filter on it",
            reaches="**solver**",
            method="the collocation order of every `adaptive_levin_sincos` call the source "
            "integral makes",
            knob="itself -- this is the same integer-order case as the Gauss orders",
            count="1,275 x 50 x (response z) per model",
            provenance="`prompts/source-remediation`, recorded in the comment at "
            "`QuadSourceIntegral.py:60-70`: a self-consistency sweep, with the caveat in the "
            "code that it 'cannot rule out an order-independent bias shared by every order "
            "tested'",
            owner="`prompts/levin-refactor` / `prompts/qsi-phase-groups`; prompt 06 read-only",
        ),
        Parameter(
            name="`DEFAULT_LEVIN_MAX_DEPTH` (`AdaptiveLevin/levin_quadrature.py:154`)",
            value="@AdaptiveLevin.levin_quadrature.DEFAULT_LEVIN_MAX_DEPTH",
            keys="none",
            reaches="**solver** -- production takes this default; no caller overrides it",
            method="maximum bisection depth of the adaptive Levin driver",
            knob="itself, together with the caller's `atol`/`rtol`",
            count="as `CHEBYSHEV_ORDER`",
            provenance="**never chosen.** The comment at `:153` gives the geometric reading "
            "('1/2^20 is roughly 1E-6') and no measurement",
            owner="`prompts/levin-refactor`; prompt 06 records it",
        ),
        Parameter(
            name="`limit=100` (`Quadrature/simple_quadrature.py:92`)",
            value="`100`",
            keys="none",
            reaches="**solver**",
            method="the subdivision cap of `scipy.integrate.quad`, on the "
            '`method="quad"` path `QuadSourceIntegral` takes at `:1498` '
            "(`_three_bessel_quad`) and `:1721` (`numeric_quad_integral`)",
            knob="itself, together with `DEFAULT_QUADRATURE_ATOL`/`_RTOL`: a quadrature that "
            "exhausts the cap returns without meeting the tolerance",
            count="the non-oscillatory sub-intervals of every `QuadSourceIntegral`",
            provenance="**never chosen**; the literal carries no comment",
            owner="`prompts/levin-refactor` / `prompts/qsi-phase-groups`; prompt 06 records it",
        ),
        Parameter(
            name="`DEFAULT_LEVIN_ABSTOL` (`AdaptiveLevin/levin_quadrature.py:157`)",
            value="@AdaptiveLevin.levin_quadrature.DEFAULT_LEVIN_ABSTOL",
            keys="none",
            reaches="**neither in production** -- every production call passes its own `atol`",
            method="n/a",
            knob="n/a",
            count="0 in the pipeline",
            provenance="**never chosen**; the comment reads 'default abs tolerance'",
            owner="`prompts/levin-refactor`",
        ),
        Parameter(
            name="`DEFAULT_LEVIN_RELTOL` (`AdaptiveLevin/levin_quadrature.py:160`)",
            value="@AdaptiveLevin.levin_quadrature.DEFAULT_LEVIN_RELTOL",
            keys="none",
            reaches="**neither in production**",
            method="n/a",
            knob="n/a",
            count="0 in the pipeline",
            provenance="**never chosen**",
            owner="`prompts/levin-refactor`",
        ),
        Parameter(
            name="`DEFAULT_LEVIN_CHEBSHEV_ORDER` "
            "(`AdaptiveLevin/levin_quadrature.py:150`)",
            value="@AdaptiveLevin.levin_quadrature.DEFAULT_LEVIN_CHEBSHEV_ORDER",
            keys="none",
            reaches="**neither in production** -- `QuadSourceIntegral` passes 24",
            method="n/a",
            knob="n/a",
            count="0 in the pipeline",
            provenance="`prompts/adaptive-levin-benchmark`'s order sweep, recorded at "
            "`:139-149`",
            owner="`prompts/levin-refactor`",
        ),
    ],
    "F. Representation orders that are not in any key": [
        Parameter(
            name="`DEFAULT_T_Z_SPLINE_SAMPLES` "
            "(`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:72`)",
            value="@CosmologyModels.GenericEOS.LambdaCDM_GenericEOS.DEFAULT_T_Z_SPLINE_SAMPLES",
            keys="through `T_Z_REPRESENTATION_VERSION` only",
            reaches="**lookup key** (via the version) **and** the representation's accuracy",
            method="node count of the tabulated $T(z)$ representation",
            knob="itself, with the spline order",
            count="1 per `QCD_Cosmology`, once per run",
            provenance="`prompts/qcd-background-audit` prompt 06 (`a1d667a`) raised it to 3,000",
            owner="`prompts/qcd-background-audit`",
        ),
        Parameter(
            name="`DEFAULT_T_Z_SPLINE_ORDER` "
            "(`CosmologyModels/GenericEOS/LambdaCDM_GenericEOS.py:73`)",
            value="@CosmologyModels.GenericEOS.LambdaCDM_GenericEOS.DEFAULT_T_Z_SPLINE_ORDER",
            keys="through `T_Z_REPRESENTATION_VERSION` only",
            reaches="**lookup key** (via the version) **and** the representation's accuracy",
            method="interpolation order of the $T(z)$ spline",
            knob="itself",
            count="as above",
            provenance="`prompts/qcd-background-audit`",
            owner="`prompts/qcd-background-audit`",
        ),
        Parameter(
            name="`DERIVATIVE_SPLINE_ORDER` (`ComputeTargets/BackgroundModel.py:64`)",
            value="@ComputeTargets.BackgroundModel.DERIVATIVE_SPLINE_ORDER",
            keys="none",
            reaches="**neither**",
            method="order of the segmented fit `BackgroundModel` differentiates to get the "
            "background derivatives",
            knob="itself, with `DERIVATIVE_FIT_PAD_POINTS` and `DERIVATIVE_FIT_REFINE`",
            count="1 `BackgroundModel` per (cosmology, grid); read by every target",
            provenance="`prompts/qcd-background-audit` prompt 13",
            owner="`prompts/qcd-background-audit`. **Not in this campaign's scope**, and listed "
            "so that prompt 04's answer to 'which accuracy parameters are in no key' is complete",
        ),
        Parameter(
            name="`STORED_SAMPLE_SPLINE_ORDER` (`ComputeTargets/BackgroundModel.py:69`)",
            value="@ComputeTargets.BackgroundModel.STORED_SAMPLE_SPLINE_ORDER",
            keys="none",
            reaches="**neither**",
            method="order of the spline through the stored background samples",
            knob="itself",
            count="as above",
            provenance="`prompts/qcd-background-audit` prompt 13",
            owner="`prompts/qcd-background-audit`",
        ),
        Parameter(
            name="`DERIVATIVE_FIT_PAD_POINTS`, `DERIVATIVE_FIT_REFINE`, "
            "`DERIVATIVE_FIT_PAD_FLOOR`, `DERIVATIVE_FIT_PAD_FRACTION` "
            "(`ComputeTargets/BackgroundModel.py:54-62`)",
            value="@ComputeTargets.BackgroundModel.DERIVATIVE_FIT_PAD_POINTS / "
            "@ComputeTargets.BackgroundModel.DERIVATIVE_FIT_REFINE / "
            "@ComputeTargets.BackgroundModel.DERIVATIVE_FIT_PAD_FLOOR / "
            "@ComputeTargets.BackgroundModel.DERIVATIVE_FIT_PAD_FRACTION",
            keys="none",
            reaches="**neither**",
            method="the padding and refinement of the derivative fit grid",
            knob="itself",
            count="as above",
            provenance="`prompts/qcd-background-audit` prompt 13; "
            "`[03-derivative-pad-clamp-on-coarse-grids]` is open against the pad clamp",
            owner="`prompts/qcd-background-audit`",
        ),
        Parameter(
            name="`BESSEL_ORDER_CHECK_TOL` (`ComputeTargets/QuadSourceIntegral.py:93`)",
            value="@ComputeTargets.QuadSourceIntegral.BESSEL_ORDER_CHECK_TOL",
            keys="none",
            reaches="**neither** -- it is a guard against a wrong Bessel order, not an accuracy "
            "request",
            method="compares a reconstructed $J_\\nu$ against the phase object's own envelope",
            knob="n/a",
            count="as `CHEBYSHEV_ORDER`",
            provenance="argued in the comment at `QuadSourceIntegral.py:79-92`: nine orders "
            "above the reconstruction floor and two below the smallest defect it must catch",
            owner="`prompts/levin-refactor` / `prompts/qsi-phase-groups`",
        ),
    ],
    "G. The source-grid construction constants": [
        Parameter(
            name="`SOURCE_GRID_CONSTRUCTION_VERSION` "
            "(`CosmologyConcepts/wavenumber.py:61`)",
            value="@CosmologyConcepts.wavenumber.SOURCE_GRID_CONSTRUCTION_VERSION",
            keys="`BackgroundModel.source_grid_construction`, an equality predicate "
            "(section 1)",
            reaches="**lookup key**",
            method="none -- it is the identity of the construction the constants below define",
            knob="the constants below",
            count="1 per (cosmology, grid)",
            provenance="`prompts/qcd-background-audit` prompt 15",
            owner="`prompts/qcd-background-audit`; **held fixed here** (README section 0.5)",
        ),
        Parameter(
            name="`SOURCE_GRID_MAX_SPACING_FACTOR` "
            "(`CosmologyConcepts/wavenumber.py:181`)",
            value="@CosmologyConcepts.wavenumber.SOURCE_GRID_MAX_SPACING_FACTOR",
            keys="through the grid digest, on every target that carries one",
            reaches="**lookup key** (via the digest) **and** the grid's density",
            method="caps the criterion so that it may only refine, never coarsen",
            knob="itself",
            count="the whole pipeline -- every target is sampled on this grid",
            provenance="`prompts/qcd-background-audit` prompt 15, on the user's instruction "
            "quoted verbatim at `wavenumber.py:160-170`; the declined saving is costed in "
            "`docs/qcd-background-verification.md` section 10.5",
            owner="`prompts/qcd-background-audit`; held fixed",
        ),
        Parameter(
            name="`SOURCE_GRID_CONSUMER_TARGET_RAD` "
            "(`CosmologyConcepts/wavenumber.py:223`)",
            value="@CosmologyConcepts.wavenumber.SOURCE_GRID_CONSUMER_TARGET_RAD",
            keys="through the grid digest",
            reaches="**lookup key** (via the digest) **and** the grid's density",
            method="the ceiling on the per-case phase error target the density criterion "
            "equidistributes",
            knob="itself -- **this is the accuracy target of the source grid**",
            count="the whole pipeline",
            provenance="`prompts/qcd-background-audit` prompt 10 section 5",
            owner="`prompts/qcd-background-audit`; held fixed",
        ),
        Parameter(
            name="`SOURCE_GRID_CUBIC_ERROR_CONST`, `SOURCE_GRID_MAX_REFINEMENT`, "
            "`SOURCE_GRID_CURVATURE_STEP_U`, `SOURCE_GRID_CURVATURE_FD_STEP_U`, "
            "`SOURCE_GRID_CROSSING_MASK_U`, `SOURCE_GRID_SPLINE_EDGE_INTERVALS`, "
            "`SOURCE_GRID_SPLINE_EDGE_FACTOR` (`CosmologyConcepts/wavenumber.py:188-244`)",
            value="@CosmologyConcepts.wavenumber.SOURCE_GRID_CUBIC_ERROR_CONST / "
            "@CosmologyConcepts.wavenumber.SOURCE_GRID_MAX_REFINEMENT / "
            "@CosmologyConcepts.wavenumber.SOURCE_GRID_CURVATURE_STEP_U / "
            "@CosmologyConcepts.wavenumber.SOURCE_GRID_CURVATURE_FD_STEP_U / "
            "@CosmologyConcepts.wavenumber.SOURCE_GRID_CROSSING_MASK_U / "
            "@CosmologyConcepts.wavenumber.SOURCE_GRID_SPLINE_EDGE_INTERVALS / "
            "@CosmologyConcepts.wavenumber.SOURCE_GRID_SPLINE_EDGE_FACTOR",
            keys="through the grid digest",
            reaches="**lookup key** (via the digest) **and** the grid's density",
            method="the version-2 curvature criterion "
            "$h^4|\\varphi''''|/384 \\le \\varepsilon$ and its stencil",
            knob="themselves",
            count="the whole pipeline",
            provenance="`prompts/qcd-background-audit` prompts 12 and 15; each carries its own "
            "measurement in the comment above it, and "
            "`docs/qcd-background-verification.md` section 10 holds the tables",
            owner="`prompts/qcd-background-audit`; held fixed",
        ),
        Parameter(
            name="`SOURCE_GRID_BREAK_STANDOFF`, `SOURCE_GRID_BREAK_HALF_WIDTH`, "
            "`SOURCE_GRID_BREAK_REFINEMENT`, `SOURCE_GRID_MESH_GUARD`, "
            "`SOURCE_GRID_MIN_SEPARATION` (`CosmologyConcepts/wavenumber.py:113-145`)",
            value="@CosmologyConcepts.wavenumber.SOURCE_GRID_BREAK_STANDOFF / "
            "@CosmologyConcepts.wavenumber.SOURCE_GRID_BREAK_HALF_WIDTH / "
            "@CosmologyConcepts.wavenumber.SOURCE_GRID_BREAK_REFINEMENT / "
            "@CosmologyConcepts.wavenumber.SOURCE_GRID_MESH_GUARD / "
            "@CosmologyConcepts.wavenumber.SOURCE_GRID_MIN_SEPARATION",
            keys="through the grid digest",
            reaches="**lookup key** (via the digest) **and** the grid's density near a crossing",
            method="the version-1 straddling pair and refined neighbourhood at each declared "
            "crossing",
            knob="themselves",
            count="the whole pipeline",
            provenance="`prompts/qcd-background-audit` prompts 10 and 11, scored in "
            "`docs/qcd-background-verification.md` section 8 and log 11 section 3",
            owner="`prompts/qcd-background-audit`; held fixed",
        ),
    ],
}


# ---------------------------------------------------------------------------------------------
# 6. rendering
# ---------------------------------------------------------------------------------------------

COLUMN_HEADINGS = (
    "Parameter",
    "Value",
    "Keys which object types",
    "Reaches",
    "Method it feeds",
    "The real knob",
    "Object count of the sector",
    "Provenance",
    "Owned by",
)


_REFERENCE = re.compile(r"@([A-Za-z_][A-Za-z0-9_.]*[A-Za-z0-9_])")


def _resolve(text: str, constants: Dict[str, object]) -> str:
    """Replace every ``@dotted.name`` in ``text`` with the imported constant's value."""

    def substitute(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key not in constants:
            raise KeyError(f"inventory.py: no imported constant named {key!r}")
        return f"`{format_value(constants[key])}`"

    return _REFERENCE.sub(substitute, text)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render() -> str:
    constants = imported_constants()
    lines: List[str] = []

    lines.append("### 5.1 The keyed object types, and what each lookup filters on")
    lines.append("")
    lines.append(
        "Derived by importing every factory in `Datastore/SQL/Datastore.py`'s `_factories` map, "
        "calling `register()` for the column list and walking the `ast` of `build()` for the "
        "comparison predicates; the rule is stated in `inventory.py.lookup_predicates`. "
        "`serial` and `validated` are dropped as bookkeeping. **A table appears here if and only "
        "if one of its predicates mentions one of `"
        + "`, `".join(sorted(ACCURACY_COLUMNS))
        + "`** — the four Gauss orders having joined the set on 2026-09-18, when prompt 05 made "
        "them the key column of the targets whose tolerance pair reached no solver."
    )
    lines.append("")
    lines.append("| Table | Columns | Lookup predicates |")
    lines.append("|---|---|---|")
    types = keyed_object_types()
    for name, columns, predicates in types:
        rendered = ", ".join(f"`{p.render()}`" for p in predicates)
        lines.append(f"| `{name}` | {len(columns)} | {rendered} |")
    lines.append("")
    lines.append(f"**{len(types)} keyed object types.**")
    lines.append("")

    lines.append("### 5.2 Constant values, imported")
    lines.append("")
    lines.append("| Constant | Value |")
    lines.append("|---|---|")
    for key in sorted(constants):
        module, _, name = key.rpartition(".")
        lines.append(f"| `{module}.{name}` | `{format_value(constants[key])}` |")
    lines.append("")

    lines.append("### 5.3 Hard-coded tolerance literals at production call sites")
    lines.append("")
    lines.append(
        "Every keyword argument named "
        + ", ".join(f"`{k}`" for k in sorted(TOLERANCE_KEYWORDS))
        + " whose value is a numeric literal, in the production trees "
        + ", ".join(f"`{t}`" for t in PRODUCTION_TREES)
        + ", in `main.py` and in the `extract_*.py` readers. Test trees, `docs/` and `prompts/` "
        "are excluded: they measure the pipeline, they are not it."
    )
    lines.append("")
    lines.append("| Call site | Argument line | Called | Keyword | Literal |")
    lines.append("|---|---|---|---|---|")
    literals = hard_coded_literals()
    for path, call_line, arg_line, call, keyword, literal in literals:
        lines.append(
            f"| `{path}:{call_line}` | `:{arg_line}` | `{call}` | `{keyword}` | `{literal}` |"
        )
    lines.append("")
    lines.append(f"**{len(literals)} literal requests.**")
    lines.append("")

    lines.append("### 5.4 The inventory")
    lines.append("")
    for heading, rows in PARAMETERS.items():
        lines.append(f"#### {heading}")
        lines.append("")
        lines.append("| " + " | ".join(COLUMN_HEADINGS) + " |")
        lines.append("|" + "---|" * len(COLUMN_HEADINGS))
        for row in rows:
            cells = (
                row.name,
                row.value,
                row.keys,
                row.reaches,
                row.method,
                row.knob,
                row.count,
                row.provenance,
                row.owner,
            )
            lines.append(
                "| " + " | ".join(_cell(_resolve(c, constants)) for c in cells) + " |"
            )
        lines.append("")
    total = sum(len(rows) for rows in PARAMETERS.values())
    lines.append(f"**{total} rows.**")
    lines.append("")

    lines.append("### 5.5 Object counts, which are inputs and not measurements")
    lines.append("")
    lines.append(
        "This script is told these. A script that recomputed them would be measuring the "
        "pipeline rather than inventorying it, and prompt 02 measures nothing."
    )
    lines.append("")
    lines.append("| Sector | Count | Source |")
    lines.append("|---|---|---|")
    for sector, count, source in OBJECT_COUNTS:
        lines.append(f"| {sector} | {count} | {source} |")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def splice(document: str, generated: str) -> str:
    before, marker, rest = document.partition(BEGIN_MARKER)
    if not marker:
        raise RuntimeError(f"inventory.py: {BEGIN_MARKER} not found in {DOCUMENT}")
    _, end_marker, after = rest.partition(END_MARKER)
    if not end_marker:
        raise RuntimeError(f"inventory.py: {END_MARKER} not found in {DOCUMENT}")
    return f"{before}{BEGIN_MARKER}\n\n{generated}\n{END_MARKER}{after}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="rewrite the generated block in the document",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the document's generated block is out of date",
    )
    args = parser.parse_args()

    generated = render()

    if not (args.write or args.check):
        sys.stdout.write(generated)
        return 0

    document = DOCUMENT.read_text()
    updated = splice(document, generated)

    if args.check:
        if updated == document:
            print(f"inventory.py: {DOCUMENT.name} is up to date")
            return 0
        print(f"inventory.py: {DOCUMENT.name} is OUT OF DATE; re-run with --write")
        return 1

    DOCUMENT.write_text(updated)
    print(f"inventory.py: rewrote the generated block of {DOCUMENT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
