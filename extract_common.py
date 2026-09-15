"""
Shared services for the extract_*.py data-product scripts -- plot furniture, and (prompt 14 of
prompts/qcd-background-audit) the naming and selection of *runs*.

The run-naming convention lives here rather than in main.py because main.py cannot be imported
(it parses sys.argv, opens a Ray connection and a ShardedPool at module scope: CLAUDE.md), so a
constant declared there could only reach these scripts as a second copy. main.py imports
``run_label_tag`` and ``source_grid_construction_tag`` from this module, which makes the writer
and the seven readers agree on the tag spelling by construction rather than by a test.

**Why the scripts need any of this.** Before prompt 14 all seven passed *zero* tags, against
main.py's eight-plus tagged call sites. That was a reasonable development choice: while a store
held exactly one complete run, a tagless query needed no adjustment whenever the compute
configuration changed. It stops being safe the moment a store holds two generations -- what a
partial regeneration, or an archived store read beside a current one, produces -- because the
scripts cannot tell them apart and will mix silently or pick arbitrarily.

The mechanism here preserves the ergonomic property that made the tagless design pleasant:

  * **select** on the run label, which is the only thing a user should have to type;
  * if no label is given and the store holds exactly one run, use it and *say which*;
  * only an ambiguous store requires the user to choose, and then the refusal names the
    candidates;
  * a store written before prompt 14 carries no run labels at all. That is **not** an error: its
    rows are an unknown generation, and they stay queryable and plottable, because a superseded
    datastore keeps its archival value long after it can no longer serve as a numerical base.

**Verification** -- that everything pulled under one label really is one generation -- is not done
here. It is done by ``Datastore/SQL/ObjectFactories/BackgroundModel.py``'s ``build()``, which is
the first lookup every one of these scripts makes and the only object that carries the grid's
identity as columns rather than as a tag; it refuses, naming every generation it found, rather
than returning a mixture. ``describe_background_generation`` below renders what it found.
"""

from math import fabs
from typing import List, NamedTuple, Optional, Sequence

import ray

from ComputeTargets import GkSource, GkSourcePolicyData
from ComputeTargets.GkSourcePolicyData import GkSourceFunctions
from CosmologyConcepts import wavenumber_exit_time, redshift

# ---------------------------------------------------------------------------------------------
# run identity (prompt 14 of prompts/qcd-background-audit)
# ---------------------------------------------------------------------------------------------

# A run's name reaches the datastore as a store_tag, in the manner of main.py's TkProductionTag /
# GkProductionTag ("TkOneLoopDensity", "GkOneLoopDensity"): a hand-chosen label string, written
# with everything the run produces and filtered on when looking those objects up again. The
# prefix is what lets a reader recover the set of runs a store holds from the store_tag table
# alone, which is the only table a reader can enumerate without a lookup key.
RUN_LABEL_TAG_PREFIX = "Run_"

# ... and the generation of the grid-construction algorithm that produced the run's sample grid,
# CosmologyConcepts.wavenumber.SOURCE_GRID_CONSTRUCTION_VERSION. It is carried as a tag beside the
# grid's content digest because the two answer different questions: the digest says *which exact
# grid*, and cannot be inverted or range-queried; this says *which generation*, and is the only
# thing comparable across grids whose values legitimately differ (two cosmologies in one run have
# different grids and the same construction).
SOURCE_GRID_CONSTRUCTION_TAG_PREFIX = "SourceGridConstruction_"


def run_label_tag(run_label: str) -> str:
    """The store_tag label under which a run of name ``run_label`` records itself."""
    return f"{RUN_LABEL_TAG_PREFIX}{run_label}"


def source_grid_construction_tag(version: int) -> str:
    """The store_tag label recording which grid-construction algorithm built a run's grid."""
    return f"{SOURCE_GRID_CONSTRUCTION_TAG_PREFIX}{version}"


class RunSelection(NamedTuple):
    """
    Which run an extract_*.py script is reading.

    ``label`` is the run's name, or None when the datastore records no runs at all -- a store
    written before prompt 14, whose rows are an unknown generation and are read exactly as they
    were before. ``tags`` is what to hand every ``pool.object_get`` as ``tags=``; it is empty in
    the unknown case, which reproduces the pre-prompt-14 query verbatim. ``description`` is one
    line for the script to print, so that a reader of the output knows what was selected and
    whether the script chose it or the user did.
    """

    label: Optional[str]
    tags: list
    description: str


def add_run_selection_argument(parser) -> None:
    """Add the one argument a reader should ever have to type."""
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help=(
            "read the run of this name (main.py --run-label). If omitted, and the datastore "
            "holds exactly one run, that run is used and named in the output; a datastore "
            "holding more than one run must be disambiguated"
        ),
    )


def available_run_labels(pool) -> List[str]:
    """
    The names of the runs this datastore holds, in sorted order.

    store_tag is a replicated table and its inventory() reports every label, so this needs no
    lookup key and no knowledge of what the store contains -- which is the property that makes
    the "exactly one run" case answerable without the user saying anything.
    """
    inventory = pool.inventory("store_tag")
    labels = inventory.get("values", []) if isinstance(inventory, dict) else []

    return sorted(
        {
            label[len(RUN_LABEL_TAG_PREFIX) :]
            for label in labels
            if isinstance(label, str) and label.startswith(RUN_LABEL_TAG_PREFIX)
        }
    )


def choose_run_label(
    available: Sequence[str], requested: Optional[str]
) -> RunSelection:
    """
    Decide which run to read, from the names a datastore holds and what the user asked for.

    Pure: it takes the label set rather than the pool, so the four cases below can be tested
    without a datastore. It returns a RunSelection whose ``tags`` is empty -- materialising the
    store_tag objects needs the pool and is :func:`resolve_run_selection`'s job.
    """
    available = list(available)

    if requested is not None:
        if requested not in available:
            known = (
                ", ".join(available)
                if len(available) > 0
                else "none (this datastore records no runs)"
            )
            raise RuntimeError(
                f'extract: this datastore holds no run named "{requested}". Runs present: '
                f"{known}."
            )
        return RunSelection(
            label=requested,
            tags=[],
            description=f'reading run "{requested}" (selected with --run-label)',
        )

    if len(available) == 0:
        # a datastore written before prompt 14. Not an error: it is read exactly as it was read
        # before, and what it holds is reported as unknown rather than guessed at.
        return RunSelection(
            label=None,
            tags=[],
            description=(
                "this datastore records no run labels: reading it as a single unnamed run of "
                "unknown generation (it predates prompts/qcd-background-audit prompt 14)"
            ),
        )

    if len(available) == 1:
        return RunSelection(
            label=available[0],
            tags=[],
            description=(
                f'reading run "{available[0]}", the only run in this datastore (no --run-label '
                f"given)"
            ),
        )

    raise RuntimeError(
        "extract: this datastore holds more than one run and none was selected. Runs present: "
        f"{', '.join(available)}. Choose one with --run-label."
    )


def resolve_run_selection(pool, requested: Optional[str]) -> RunSelection:
    """
    :func:`choose_run_label` against the runs this datastore actually holds, with the chosen
    run's store_tag materialised so that it can be passed straight to ``pool.object_get``.
    """
    selection = choose_run_label(available_run_labels(pool), requested)

    if selection.label is None:
        return selection

    tag = ray.get(pool.object_get("store_tag", label=run_label_tag(selection.label)))

    return selection._replace(tags=[tag])


def describe_background_generation(model) -> str:
    """
    Which generation of the source grid a BackgroundModel row belongs to, as a printable string.

    ``_source_grid_identity`` is set by Datastore/SQL/ObjectFactories/BackgroundModel.py's
    build(). It is None for a datastore whose BackgroundModel table predates prompt 14 and
    therefore records nothing about the grid -- which is reported as unknown rather than
    defaulted, because there is no defensible value to assume.
    """
    identity = getattr(model, "_source_grid_identity", None)
    if identity is None:
        return "grid generation unknown (this datastore predates prompt 14)"

    construction, digest = identity
    return f"source grid construction version {construction}, digest {digest}"


TEXT_DISPLACEMENT_MULTIPLIER = 0.85

TOP_ROW = 1.12
MIDDLE_ROW = 1.07
BOTTOM_ROW = 1.02

LEFT_COLUMN = 0.0
MIDDLE_COLUMN = 0.4
RIGHT_COLUMN = 0.85

BOTTOM_MARGIN = 0.0


def safe_fabs(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None

    return fabs(x)


def safe_div(x: Optional[float], y: float) -> Optional[float]:
    if x is None or y is None:
        return None

    try:
        return x / y
    except ZeroDivisionError:
        pass

    return None


def set_loglinear_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.grid(True)
    ax.xaxis.set_inverted(True)


def set_loglog_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="best")
    ax.grid(True)
    ax.xaxis.set_inverted(True)


def set_linear_axes(ax):
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.legend(loc="best")
    ax.grid(True)
    ax.xaxis.set_inverted(True)


# Matplotlib line style from https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
#      ('loosely dotted',        (0, (1, 10))),
#      ('dotted',                (0, (1, 1))),
#      ('densely dotted',        (0, (1, 1))),
#      ('long dash with offset', (5, (10, 3))),
#      ('loosely dashed',        (0, (5, 10))),
#      ('dashed',                (0, (5, 5))),
#      ('densely dashed',        (0, (5, 1))),
#
#      ('loosely dashdotted',    (0, (3, 10, 1, 10))),
#      ('dashdotted',            (0, (3, 5, 1, 5))),
#      ('densely dashdotted',    (0, (3, 1, 1, 1))),
#
#      ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
#      ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
#      ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

LOOSE_DASHED = (0, (4, 4))
LOOSE_DOTTED = (0, (1, 2))


def add_zexit_lines(ax, k_exit: wavenumber_exit_time, col1: str = "b", col2: str = "r"):
    ax.axvline(k_exit.z_exit_subh_e3, linestyle=(0, (1, 1)), color=col1)  # dotted
    ax.axvline(k_exit.z_exit_subh_e5, linestyle=(0, (1, 1)), color=col1)  # dotted
    ax.axvline(k_exit.z_exit_suph_e3, linestyle=(0, (1, 1)), color=col1)  # dotted
    ax.axvline(
        k_exit.z_exit, linestyle=(0, (3, 1, 1, 1)), color=col2
    )  # densely dashdotted
    trans = ax.get_xaxis_transform()
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_suph_e3,
        0.75,
        "$-3$ e-folds",
        transform=trans,
        fontsize="x-small",
        color="b",
    )
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_subh_e3,
        0.85,
        "$+3$ e-folds",
        transform=trans,
        fontsize="x-small",
        color="b",
    )
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit_subh_e5,
        0.75,
        "$+5$ e-folds",
        transform=trans,
        fontsize="x-small",
        color="b",
    )
    ax.text(
        TEXT_DISPLACEMENT_MULTIPLIER * k_exit.z_exit,
        0.92,
        "re-entry",
        transform=trans,
        fontsize="x-small",
        color="r",
    )


def add_GkSourcePolicyData_lines(ax, GkPolicy):
    trans = ax.get_xaxis_transform()
    if GkPolicy.type == "mixed" and GkPolicy.crossover_z is not None:
        ax.axvline(
            GkPolicy.crossover_z, linestyle=(5, (10, 3)), color="m"
        )  # long dash with offset
        ax.text(
            TEXT_DISPLACEMENT_MULTIPLIER * GkPolicy.crossover_z,
            0.15,
            "crossover_z",
            transform=trans,
            fontsize="x-small",
            color="m",
        )
    if (
        GkPolicy.type == "mixed" or GkPolicy.type == "WKB"
    ) and GkPolicy.Levin_z is not None:
        ax.axvline(
            GkPolicy.Levin_z, linestyle=(0, (5, 10)), color="m"
        )  # loosely dashed
        ax.text(
            TEXT_DISPLACEMENT_MULTIPLIER * GkPolicy.Levin_z,
            0.05,
            "Levin boundary",
            transform=trans,
            fontsize="x-small",
            color="m",
        )


def add_k_labels(
    ax,
    k_exit: Optional[wavenumber_exit_time],
    q_exit: Optional[wavenumber_exit_time],
    r_exit: Optional[wavenumber_exit_time],
):
    if k_exit is not None:
        ax.text(
            LEFT_COLUMN,
            BOTTOM_ROW,
            f"$k$ = {k_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if q_exit is not None:
        ax.text(
            MIDDLE_COLUMN,
            BOTTOM_ROW,
            f"$q$ =  {q_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if r_exit is not None:
        ax.text(
            RIGHT_COLUMN,
            BOTTOM_ROW,
            f"$r$ = {r_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
            transform=ax.transAxes,
            fontsize="x-small",
        )


def add_simple_plot_labels(
    ax,
    z_source: Optional[redshift] = None,
    z_response: Optional[redshift] = None,
    k_exit: Optional[wavenumber_exit_time] = None,
    q_exit: Optional[wavenumber_exit_time] = None,
    r_exit: Optional[wavenumber_exit_time] = None,
    model_label: str = "LambdaCDM",
):
    add_k_labels(ax, k_exit, q_exit, r_exit)

    if z_source is not None:
        ax.text(
            LEFT_COLUMN,
            MIDDLE_ROW,
            f"z-source: {z_source.z:.5g}",
            transform=ax.transAxes,
            fontsize="x-small",
        )
    elif z_response is not None:
        ax.text(
            LEFT_COLUMN,
            MIDDLE_ROW,
            f"z-response: {z_response.z:.5g}",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if model_label is not None:
        ax.text(
            RIGHT_COLUMN,
            MIDDLE_ROW,
            f"Model: {model_label}",
            transform=ax.transAxes,
            fontsize="x-small",
        )


def add_GkSource_plot_labels(
    ax, Gk: GkSource, GkPolicy: GkSourcePolicyData, model_label: str = "LambdaCDM"
):
    k_exit: wavenumber_exit_time = Gk._k_exit
    fns: GkSourceFunctions = GkPolicy.functions

    # TOP ROW
    if fns.WKB_region is not None:
        ax.text(
            LEFT_COLUMN,
            TOP_ROW,
            f"WKB region: ({fns.WKB_region[0]:.3g}, {fns.WKB_region[1]:.3g})",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if fns.numeric_region is not None:
        ax.text(
            MIDDLE_COLUMN,
            TOP_ROW,
            f"Numeric region: ({fns.numeric_region[0]:.3g}, {fns.numeric_region[1]:.3g})",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    ax.text(
        RIGHT_COLUMN,
        TOP_ROW,
        f"Model: {model_label}",
        transform=ax.transAxes,
        fontsize="x-small",
    )

    # MIDDLE ROW
    if fns.type is not None:
        ax.text(
            LEFT_COLUMN,
            MIDDLE_ROW,
            f"Type: {fns.type}",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if fns.quality is not None:
        ax.text(
            MIDDLE_COLUMN,
            MIDDLE_ROW,
            f"Quality: {fns.quality}",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    if fns.phase is not None:
        ax.text(
            RIGHT_COLUMN,
            MIDDLE_ROW,
            f"Chunks: {fns.phase.num_chunks}",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    # BOTTOM ROW
    ax.text(
        LEFT_COLUMN,
        BOTTOM_ROW,
        f"z-response: {Gk.z_response.z:.5g}",
        transform=ax.transAxes,
        fontsize="x-small",
    )
    ax.text(
        MIDDLE_COLUMN,
        BOTTOM_ROW,
        f"z-exit: {k_exit.z_exit:.5g}",
        transform=ax.transAxes,
        fontsize="x-small",
    )
    ax.text(
        RIGHT_COLUMN,
        BOTTOM_ROW,
        f"$k$ = {k_exit.k.k_inv_Mpc:.5g} Mpc$^{{-1}}$",
        transform=ax.transAxes,
        fontsize="x-small",
    )

    # BOTTOM MARGIN
    ax.text(
        LEFT_COLUMN,
        BOTTOM_MARGIN,
        f"policy: {GkPolicy.policy.label}",
        transform=ax.transAxes,
        fontsize="x-small",
    )


def add_region_labels(
    ax,
    z_min_quad,
    z_max_quad,
    z_min_Levin,
    z_max_Levin,
    model_label: str = "LambdaCDM",
):
    if z_min_quad is not None and z_max_quad is not None:
        ax.text(
            LEFT_COLUMN,
            MIDDLE_ROW,
            f"numeric: [{z_min_quad.z:.5g}, {z_max_quad.z:.5g}]",
            transform=ax.transAxes,
            fontsize="x-small",
        )
    if z_min_Levin is not None and z_max_Levin is not None:
        ax.text(
            RIGHT_COLUMN,
            MIDDLE_ROW,
            f"WKB Levin: [{z_min_Levin.z:.5g}, {z_max_Levin.z:.5g}]",
            transform=ax.transAxes,
            fontsize="x-small",
        )

    ax.text(
        RIGHT_COLUMN,
        TOP_ROW,
        f"Model: {model_label}",
        transform=ax.transAxes,
        fontsize="x-small",
    )
