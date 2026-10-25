"""
The numeric -> Liouville-Green hand-over of T_k: where the consumers switch representation.

Three redshifts are held apart (prompts/handover prompt 12, README section 7 D1 as decided):

  * **z_init** -- the Liouville-Green initialisation point, the phase extremum that
    `find_phase_extremum` returns. `TkWKBIntegration` is keyed on it and starts there; it is
    `z_exit - stop_deltaz_subh`. It is a root, not a grid node.
  * **the switch node, z_sw** -- the largest source-grid node strictly below z_init. It is the
    Liouville-Green row's first stored sample (main.py truncates the WKB grid at z_init), and the
    point at which `TkSourceFunctions`, `QuadSource` and `QuadSourceIntegral` change from the
    numeric representation to the Liouville-Green one.
  * **the overlap** -- `HANDOVER_OVERLAP_NODES` numeric nodes kept below z_sw, so that z_sw sits
    at least that many intervals inside the numeric spline rather than in its end interval.

Both representations are *stored* at z_sw and at the overlap nodes: in mode="stop" the numeric
solve runs to the end of its search window and samples every grid node on the way
(`Quadrature/integrators/numeric_with_phase_cut.py`), 118-126 nodes below z_init on A3 v2. Nothing
here changes what is computed or stored; only which stored samples the consumers read.

The one helper both consumers use is `handover_switch_node`. `QuadSource` has no
`TkWKBIntegration`, so it finds the node on its own source grid; `TkSourceFunctions` checks that its
Liouville-Green row's first sample is the same node of the numeric grid.
"""

from typing import Optional, Sequence, TypeVar

# README section 7 D1, decided 2026-10-25: "the numeric representation extends at least two
# nodes beyond that node". Two keeps the switch node out of the numeric spline's end interval
# ([05-numeric-region-is-now-the-accuracy-floor]); A3 v2 stores 118-126 numeric nodes below
# z_init, so two is always available on the grid that ships.
HANDOVER_OVERLAP_NODES = 2

Node = TypeVar("Node")


def node_z(node) -> float:
    """
    The redshift of a grid node: a `redshift` (or anything exposing `.z` as a float), or a float.
    """
    z = getattr(node, "z", node)
    return float(z)


def same_redshift(a, b) -> bool:
    """
    Redshift identity of two stored samples' redshifts: the `redshift` objects' `store_id` where
    both carry one, otherwise exact float equality of the stored z.

    Never a log(1+z) round trip, and never a tolerance: two samples of the same grid node carry
    the same stored value (CLAUDE.md, "Redshift arithmetic").
    """
    id_a = getattr(a, "store_id", None)
    id_b = getattr(b, "store_id", None)
    if id_a is not None and id_b is not None:
        return id_a == id_b

    return node_z(a) == node_z(b)


def handover_switch_node(z_init: float, grid: Sequence[Node]) -> Optional[Node]:
    """
    The switch node: the largest node of `grid` strictly below `z_init`, or None if every node is
    at or above it. `grid` holds redshift-like nodes (`.z`) or floats, in any order; the node is
    returned as given, so that a caller can match it by identity.

    The comparison is direct, in z, against a z_init that is a root rather than a grid node: no
    tolerance, because none is needed to decide which side of a root a stored node lies on.
    """
    z_init = float(z_init)
    best = None
    best_z = None
    for node in grid:
        z = node_z(node)
        if z < z_init and (best_z is None or z > best_z):
            best = node
            best_z = z

    return best
