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
ACCURACY_COLUMNS = frozenset(
    {"atol_serial", "rtol_serial", "log10_tol", "Levin_threshold"}
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


SHARED = "`DEFAULT_ABS_TOLERANCE`/`DEFAULT_REL_TOLERANCE`"

PARAMETERS: Dict[str, List[Parameter]] = {
    "A. The `config/defaults.py` accuracy constants": [
        Parameter(
            name="`DEFAULT_ABS_TOLERANCE` (`config/defaults.py:5`)",
            value="@config.defaults.DEFAULT_ABS_TOLERANCE",
            keys="`wavenumber_exit_time` (inequality, see section 2), `BackgroundModel`, "
            "`GkNumericIntegration`, `GkWKBIntegration`, `TkWKBIntegration`, `GkSource`, "
            "`OneLoopIntegral`",
            reaches="**both** -- but a solver in only one of the seven",
            method="`root_scalar` (Brent, bracketed) in $\\log(1+z)$ via "
            "`CosmologyConcepts/wavenumber.py:982`; DOP853 via `GkNumericIntegration.py:379`. "
            "Also used as a bare float-comparison epsilon at seven further sites "
            "(`ComputeTargets/GkSource.py:96`, `:104`, `:275`; "
            "`Quadrature/integrators/numeric_with_phase_cut.py:618`, `:737`, `:790`; "
            "`LiouvilleGreen/WKBtools.py:83`)",
            knob="the pair, for the two live consumers; nothing at all for the other five",
            count="50 + 1 + ~65,000 + ~65,000 + 50 + (50 x response z) + 0 per model",
            provenance="**never chosen.** No campaign document, log or code comment records a "
            "measurement behind `1e-10`; `config/defaults.py:12-15` argues only that it does not "
            "bind for $G$",
            owner="prompts 03 and 05 (`wavenumber_exit_time`, `GkNumericIntegration`); "
            "prompt 04 recommends what replaces it on the order-governed three (D3)",
        ),
        Parameter(
            name="`DEFAULT_REL_TOLERANCE` (`config/defaults.py:6`)",
            value="@config.defaults.DEFAULT_REL_TOLERANCE",
            keys="the seven above, plus `TkNumericIntegration`",
            reaches="**both** -- a solver in three of the eight",
            method="`root_scalar` (`wavenumber.py:983`); DOP853 "
            "(`GkNumericIntegration.py:380`, `TkNumericIntegration.py:413`)",
            knob="the pair, for the three live consumers",
            count="as above, plus 50 per model for `TkNumericIntegration`",
            provenance="**never chosen.** `1e-8` is the value the pipeline was written with; "
            "`prompts/GkTk-remedial` prompt 17 section 7 measured one decade of it in the $T_k$ "
            "sector and recommended tightening, and the recommendation is D1, still open",
            owner="prompt 03 measures, prompt 05 ships (D1)",
        ),
        Parameter(
            name="`DEFAULT_TK_NUMERIC_ABS_TOLERANCE` (`config/defaults.py:33`)",
            value="@config.defaults.DEFAULT_TK_NUMERIC_ABS_TOLERANCE",
            keys="`TkNumericIntegration` alone",
            reaches="**both**",
            method="DOP853, via `TkNumericIntegration.py:412` and "
            "`Quadrature/integrators/numeric_with_phase_cut.py:675`",
            knob="the pair",
            count="50 per model",
            provenance="`prompts/GkTk-remedial` prompt 12, confirmed against the production grid "
            "by prompt 17 and **settled by the user 2026-09-12**; the measurement is in "
            "`config/defaults.py:8-32` at the point of use (**version-0 grid**)",
            owner="settled; not reopened (README section 7 D1)",
        ),
        Parameter(
            name="`DEFAULT_QUADRATURE_ATOL` (`config/defaults.py:42`)",
            value="@config.defaults.DEFAULT_QUADRATURE_ATOL",
            keys="`QuadSourceIntegral`",
            reaches="**both**",
            method="`adaptive_levin_sincos` (Levin / Clenshaw-Curtis, "
            "`ComputeTargets/QuadSourceIntegral.py:1059`) and `scipy.quad` / DOP853 via "
            "`Quadrature/simple_quadrature.py`; distributed per sub-interval by log-width at "
            "`QuadSourceIntegral.py:819` and per phase group at `:1054`",
            knob="the pair",
            count="1,275 x 50 x (response z) per model",
            provenance="`prompts/source-remediation` prompt 12, against the analytic oracle; the "
            "measurement is in `config/defaults.py:36-41`",
            owner="`prompts/levin-refactor` / `prompts/qsi-phase-groups`; prompt 06 measures "
            "read-only (README section 0.4)",
        ),
        Parameter(
            name="`DEFAULT_QUADRATURE_RTOL` (`config/defaults.py:35`)",
            value="@config.defaults.DEFAULT_QUADRATURE_RTOL",
            keys="`QuadSourceIntegral`",
            reaches="**both**",
            method="as above",
            count="1,275 x 50 x (response z) per model",
            knob="the pair",
            provenance="`prompts/source-remediation` prompt 12 confirmed it **does not bind** -- "
            "1e-8 to 1e-11 bit-identical on 159 items (`config/defaults.py:38-39`)",
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
            name="`TAU_GAUSS_ORDER` (`ComputeTargets/BackgroundModel.py:35`)",
            value="@ComputeTargets.BackgroundModel.TAU_GAUSS_ORDER",
            keys="**none.** It reaches the `IntegrationSolver` label and `stepping` "
            "(`main.py:3513`, `BackgroundModel.py:49`), which `BackgroundModel` stores in "
            "`solver_serial` but **does not filter on** (section 1's predicate list)",
            reaches="**neither** -- it is the knob, and it is in no key",
            method="Gauss-Legendre cumulative table, order 4 per interval, split at the "
            "cosmology's break points (`BackgroundModel.py:427`)",
            knob="itself",
            count="1 `BackgroundModel` per (cosmology, grid); every target reads its output",
            provenance="`prompts/GkTk-remedial` prompt 02, recorded in "
            "`ComputeTargets/tests/wkb_reference_data.json`'s `convergence` block, "
            "**generated 2026-09-10 against a background and a break-point set that no longer "
            "exist** (`RECONCILIATION.md` section 5)",
            owner="prompt 04 re-measures; prompt 05 puts it in the key if the user settles D3",
        ),
        Parameter(
            name="`CS_TAU_GAUSS_ORDER` (`ComputeTargets/BackgroundModel.py:43`)",
            value="@ComputeTargets.BackgroundModel.CS_TAU_GAUSS_ORDER",
            keys="none -- and unlike $\\tau$ it does not even reach a solver label",
            reaches="**neither**",
            method="Gauss-Legendre cumulative table for $c_s\\tau$ (`BackgroundModel.py:449`)",
            knob="itself",
            count="as `TAU_GAUSS_ORDER`",
            provenance="as `TAU_GAUSS_ORDER` -- the same stale `convergence` block",
            owner="prompt 04, prompt 05 (D3)",
        ),
        Parameter(
            name="`FRICTION_F_GAUSS_ORDER` (`ComputeTargets/BackgroundModel.py:44`)",
            value="@ComputeTargets.BackgroundModel.FRICTION_F_GAUSS_ORDER",
            keys="none",
            reaches="**neither**",
            method="Gauss-Legendre cumulative table for $F$ (`BackgroundModel.py:466`)",
            knob="itself",
            count="as `TAU_GAUSS_ORDER`",
            provenance="as `TAU_GAUSS_ORDER`",
            owner="prompt 04, prompt 05 (D3)",
        ),
        Parameter(
            name="`RHO_GAUSS_ORDER` (`ComputeTargets/phase_residual.py:90`)",
            value="@ComputeTargets.phase_residual.RHO_GAUSS_ORDER",
            keys="**none.** It reaches `PHASE_SOLVER_STEPPING` "
            "(`Quadrature/integrators/WKB_phase_function.py:86`) and so the phase solver's "
            "label, which both WKB factories store and neither filters on",
            reaches="**neither**",
            method="Gauss-Legendre panels for the Liouville-Green phase residual $\\rho$, per "
            "$(model, k, sector)$ (`phase_residual.py:164`)",
            knob="itself",
            count="~65,000 `GkWKBIntegration` + 50 `TkWKBIntegration` per model",
            provenance="`prompts/GkTk-remedial` prompts 02 and 06, via the same 2026-09-10 "
            "`convergence` block",
            owner="prompt 04, prompt 05 (D3)",
        ),
        Parameter(
            name="`RESIDUAL_WKB_REGION_MARGIN` (`ComputeTargets/phase_residual.py:238`)",
            value="@ComputeTargets.phase_residual.RESIDUAL_WKB_REGION_MARGIN",
            keys="**none, anywhere** -- not in a key, a label or a tag",
            reaches="**neither**",
            method="sets the band `residual_node_range` covers, by requiring the "
            "Liouville-Green frequency to stay this fraction of its leading term "
            "(`phase_residual.py:246-267`). Re-used by `main.source_grid_spacing_profile` as the "
            "band the version-2 density criterion runs over",
            knob="itself",
            count="both WKB sectors, and (through the re-use) the source grid every target "
            "shares",
            provenance="**never chosen against a criterion.** Its own comment argues it is a "
            "*permissive* bound designed never to exclude a producer's anchor, not a resolution "
            "target; `[01-density-criterion-imposed-outside-the-wkb-region]` is what the re-use "
            "costs",
            owner="prompt 04 measures what it buys (README section 0.5 holds its value fixed)",
        ),
    ],
    "C. Root solves in production code": [
        Parameter(
            name="`_solve_horizon_exit` `xtol`/`rtol` "
            "(`CosmologyConcepts/wavenumber.py:982-983`)",
            value=f"the shared pair: {SHARED}",
            keys="`wavenumber_exit_time` -- **by inequality, not by equality** (section 2)",
            reaches="**both**",
            method="`scipy.optimize.root_scalar`, Brent, bracketed in $u = \\log(1+z)$ by a "
            "geometric widening search (`wavenumber.py:930-977`), with an acceptance guard "
            "`|q_root| <= DEFAULT_HEXIT_TOLERANCE` at `:993`",
            knob="the pair. `atol` is absolute in $u$ and `rtol` relative to a root of size ~30, "
            "so neither is comparable with either sector's (README section 2 (e))",
            count="50 per model, each solved at 1 + 3 superhorizon + 5 subhorizon offsets "
            "(`wavenumber.py:1021-1033`)",
            provenance="**never chosen.** T6 on the board records it as never measured; "
            "README section 3.1 adds that the radiation case has a one-line oracle nobody used",
            owner="prompt 03 (T6), prompt 05 (D1)",
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
        "if one of its predicates mentions `atol_serial`, `rtol_serial`, `log10_tol` or "
        "`Levin_threshold`.**"
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
