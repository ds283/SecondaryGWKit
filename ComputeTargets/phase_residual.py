"""
The residual part of the Liouville-Green phase, as a cumulative Gauss-Legendre table.

**The identity this module exists to serve** (review §6; ``prompts/GkTk-remedial/README.md``
§2 (a), (c)). Write the Liouville-Green frequency as a leading part and a correction,

    omega^2 = omega_0^2 + C,      omega_0 = k/H      (Green's function, ``sector="Gk"``)
    omega_T^2 = omega_{T,0}^2 + C_T, omega_{T,0} = k c_s/H  (transfer function, ``sector="Tk"``)

with ``C`` the **non-leading terms of** ``Gk_omegaEff_sq``/``Tk_omegaEff_sq`` evaluated directly
(``Gk_omegaEff_sq_correction``, ``Tk_omegaEff_sq_correction``), never as ``omega^2 - omega_0^2``:
the leading term is 1e12-1e24 times the rest and the subtraction would lose all of it
(``RECONCILIATION.md`` §2 item 3). Then, *exactly*,

    theta(z; z_i) = -[ k tau.delta(z_i, z) + rho.delta(z_i, z) ]

with ``tau`` the conformal-time primitive (``cs_tau``, the sound horizon, in the ``Tk`` sector)
and

    rho.delta(z_i, z) = int_z^{z_i} C/(omega + omega_0) dz.

The rationalised denominator ``omega + omega_0`` is what avoids the cancelling difference
``omega - omega_0``; ``C`` sits in the numerator unmodified, so the integrand is as accurate as
``C`` itself.

Both terms carry the campaign's interval sign convention, ``X.delta(z_a, z_b) = X(z_b) - X(z_a)``,
positive when ``z_b < z_a`` for a positive integrand (README §2 (c)). So ``theta`` is negative and
decreasing towards lower redshift, which is the author's convention.

**Exact-radiation check.** In exact radiation (``eps = 2``, ``eps' = 0``) the Green's-function
correction vanishes identically -- ``C = (3*2/2 - 2*2/4 - 2)/s^2 = 0`` with no rounding -- so
``rho_G`` is bit-exactly zero and ``theta_G(z; z_i) = -k tau.delta(z_i, z) = k(1/s_i - 1/s)``
with ``tau = 1/(H_0 s)``, ``s = 1+z``: the closed form the campaign's ``RadiationModel`` control
scores against. The transfer function is different: ``C_T = -2/s^2`` there, and
``rho_T(z; z_i) -> 1/x - 1/x_i`` with ``x = k c_s tau``, about -0.09 rad over the production range
on both models and at every ``k`` (review §12.2, §12.4). ``rho_T`` is *not* negligible and is
carried for that reason. (Review §12.4 quotes the same quantity as ``1/x_i - 1/x``; that is the
opposite sign convention -- its ``rho`` is ``theta - (x_i - x)`` -- and ``ComputeTargets/tests/
wkb_reference.py`` records the difference. Everything here is in the README §2 convention.)

**Size and cost.** Over the whole WKB range, from the 3-e-fold sub-horizon anchor down to
``z = 0.1``: ``|rho_G|`` is 2.6e-7 rad (LambdaCDM, ``k = 1e5``) falling to 1.4e-10 at
``k = 3e8``, and 3.0e-5 to 1.2e-3 rad on ``QCD_Cosmology``; ``|rho_T|`` is 0.086-0.093 rad on
both models at every ``k``. One table per ``(model, k, sector)`` costs ``RHO_GAUSS_ORDER`` times
the number of Gauss panels -- 5.5k integrand evaluations and ~10 ms on LambdaCDM, 6.9k and
0.1-0.3 s on ``QCD_Cosmology`` -- against the 2.5e6 right-hand-side evaluations and 64 s per
object of the two-stage phase ODE this replaces (review §4).

**Gauss order and the break-point scheme.** ``RHO_GAUSS_ORDER = 4``, fixed by measurement in
``prompts/GkTk-remedial/logs/02-qcd-residual-convergence.md`` (``N_rho = 4``;
``rho_adaptive_fallback_required = False``, so no adaptive rule is implemented here). As for the
other primitives, order 4 is only at the floor on ``QCD_Cosmology`` when every panel is split at
the cosmology's break points -- the ``T(z)`` spline knots and the equation-of-state branch
temperatures -- so the table is built with ``_cosmology_break_points`` exactly as
``compute_background`` builds ``tau``, ``cs_tau`` and ``friction_F``.

**One table per wavenumber, not per object** (prompt 14). ``rho`` depends only on
``(model, k, sector)``, so ``residual_node_range`` fixes a node range that serves *every* object
of that wavenumber -- the background grid, cut at the top where the Liouville-Green frequency
stops keeping ``RESIDUAL_WKB_REGION_MARGIN`` of its leading term, which is above every anchor a
producer can accept -- and ``cached_phase_residual`` memoises the table on that key inside the
worker process. Prompt 06 built it per object, anchored on the object's own ``z_init``, which
cost 5,536 of 6,000 integrand evaluations and ~92 % of the 0.031 s per object at ``k = 3e8`` on
LambdaCDM and 0.08-0.28 s per object on ``QCD_Cosmology``, repeated identically for each of the
~1,700 source redshifts of that ``k``. The object's anchor is now off the table's grid and is
reached through ``CumulativeTable.delta``'s off-grid partial -- which is the term that accessor
exists for -- at ``order`` integrand evaluations once per object.

No Ray and no datastore: this module holds the integrand, the table builder, the node-range rule
and the cache only. The producers (prompts 06 and 07) own where to anchor.
"""

import weakref
from collections import OrderedDict
from math import sqrt
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from ComputeTargets.BackgroundModel import _cosmology_break_points
from ComputeTargets.WKB_Gk import Gk_omegaEff_sq_correction, Gk_omegaEff_sq_leading
from ComputeTargets.WKB_Tk import Tk_omegaEff_sq_correction, Tk_omegaEff_sq_leading
from ComputeTargets.cumulative_table import CumulativeTable

# Gauss-Legendre order per panel for the residual, fixed by measurement in
# prompts/GkTk-remedial/logs/02-qcd-residual-convergence.md (N_rho = 4). The cumulative residual
# at order 4 is accurate to 5.6e-18 rad (LambdaCDM) and 3.8e-16 rad (QCD_Cosmology), a decade
# and more below README §6's 1e-6 rad target; order 2 is 7.3e-11 relative on the exact-radiation
# control and order 6 buys nothing.
RHO_GAUSS_ORDER = 4

# Prompt 02 measured whether a fixed-order rule converges for rho across the QCD_Cosmology spline
# knots (review §11, "the first test of any implementation"). It does -- rho is small enough that
# even an unconverged rule delivers it to 1e-9 rad, and what needed the break-point subdivision
# was the *leading* term, not the residual -- so the adaptive fallback the review holds in
# reserve for rho alone is not implemented. Recorded here so that a later reader knows it was
# decided by measurement and not overlooked.
RHO_ADAPTIVE_FALLBACK_REQUIRED = False

SECTORS = ("Gk", "Tk")

_SECTOR_FUNCTIONS = {
    "Gk": (Gk_omegaEff_sq_leading, Gk_omegaEff_sq_correction),
    "Tk": (Tk_omegaEff_sq_leading, Tk_omegaEff_sq_correction),
}


def _check_sector(sector: str) -> str:
    if sector not in _SECTOR_FUNCTIONS:
        raise ValueError(
            f"phase_residual: unknown sector '{sector}' (expected one of {SECTORS})"
        )
    return sector


def phase_residual_integrand(model, k: float, sector: str) -> Callable[[float], float]:
    """
    The residual integrand ``C/(omega + omega_0)`` in ``z``, for one background model, one
    wavenumber and one sector.

    This is the function ``CumulativeTable`` accumulates, and it is handed over *unsigned*: the
    table holds ``R(z) = int_z^{z_top} C/(omega + omega_0) dz'``, so
    ``R.delta(z_i, z) = int_z^{z_i} C/(omega + omega_0) dz'`` is the ``rho`` of the identity in
    the module docstring with no sign flip anywhere.

    :param model: anything exposing ``.functions`` (a ``BackgroundModel``, or one of the
        campaign's stand-ins)
    :param k: the comoving wavenumber, in the model's units
    :param sector: ``"Gk"`` or ``"Tk"``
    :raises ValueError: if ``omega^2 <= 0`` at the requested redshift (the Liouville-Green region
        has ``omega^2 > 0`` by construction; outside it the phase is not defined and
        ``WKB_phase_function`` raises in the same way), or if the leading term is negative (a
        ``c_s^2 < 0`` stand-in in the ``Tk`` sector)
    """
    _check_sector(sector)
    leading_fn, correction_fn = _SECTOR_FUNCTIONS[sector]
    k = float(k)

    def f(z: float) -> float:
        leading = leading_fn(model, k, z)
        correction = correction_fn(model, k, z)
        omega_sq = leading + correction

        if leading < 0.0 or omega_sq <= 0.0:
            raise ValueError(
                f"phase_residual[{sector}]: the Liouville-Green frequency is not positive at "
                f"z = {z:.8g} for k = {k:.8g} (leading = {leading:.8g}, "
                f"correction = {correction:.8g}, omega^2 = {omega_sq:.8g}) on cosmology "
                f"{type(getattr(model, 'cosmology', None)).__name__} "
                f"(store_id={getattr(getattr(model, 'cosmology', None), 'store_id', None)}); "
                "the residual is only defined inside the WKB region"
            )

        return correction / (sqrt(omega_sq) + sqrt(leading))

    return f


def build_phase_residual(
    model,
    k: float,
    z_nodes: Sequence[float],
    sector: str,
    order: Optional[int] = None,
) -> CumulativeTable:
    """
    Tabulate ``R(z) = int_z^{z_top} C/(omega + omega_0) dz'`` on ``z_nodes``.

    The caller chooses the nodes: they must lie inside the WKB region for this ``k`` (the
    integrand refuses ``omega^2 <= 0``), and an anchor slightly off the top node -- the numeric
    hand-over redshift, which is a ``root_scalar`` root and never a grid point
    (``RECONCILIATION.md`` §2 item 5) -- is reached by ``delta``'s off-grid partial, at most one
    grid interval above the top node.

    The table is **single-limb in effect**: ``rho`` is at most 0.1 rad, so a plain double carries
    it to 1e-17 absolute and the double-double machinery of ``CumulativeTable`` costs nothing but
    also buys nothing here (the low limb is computed and is ~1e-18). It is the *leading* term,
    ``k tau``, that reaches 1e12 rad and needs the two limbs.

    :param model: anything exposing ``.functions``, and optionally ``.cosmology`` (used only to
        ask the cosmology for its integration break points, duck-typed exactly as
        ``compute_background`` does; a model with no cosmology is treated as smooth)
    :param k: the comoving wavenumber, in the model's units
    :param z_nodes: the grid, strictly descending in ``z``; a ``redshift_array`` is accepted
    :param sector: ``"Gk"`` or ``"Tk"``
    :param order: Gauss-Legendre order per panel; ``None`` (the default) means the order this run
        is configured at, ``RHO_GAUSS_ORDER``, read **here** rather than bound as a default
        argument -- see the note below
    :return: the ``CumulativeTable``, whose ``delta(z_i, z)`` is ``rho(z; z_i)``

    ``order`` was written ``order: int = RHO_GAUSS_ORDER`` until prompt 05b of
    ``prompts/tolerance-convergence``. A default argument is evaluated once, at ``def`` time, so
    that spelling snapshotted the constant at import: the table could not follow a later
    re-pointing of the declaration, while the row's key column did. The sentinel resolves the
    same name at *call* time, so the table this function returns is built at the order the run is
    configured at, and ``CumulativeTable.order`` -- which is what the object records -- is the
    order it was actually built at whether the caller supplied one or not.
    """
    _check_sector(sector)

    if order is None:
        order = RHO_GAUSS_ORDER

    if hasattr(z_nodes, "as_float_list"):
        z_nodes = z_nodes.as_float_list()
    z = np.array([float(v) for v in z_nodes], dtype=float)
    if z.size < 2:
        raise ValueError(
            f"build_phase_residual[{sector}]: at least two nodes are required "
            f"(got {z.size})"
        )

    cosmology = getattr(model, "cosmology", None)
    if cosmology is None:
        break_points = np.empty(0, dtype=float)
    else:
        break_points = _cosmology_break_points(
            cosmology, float(np.min(z)), float(np.max(z))
        )

    return CumulativeTable(
        z,
        phase_residual_integrand(model, k, sector),
        order,
        break_points=break_points,
        label=f"rho_{sector}@k={float(k):.6e}",
    )


# ---------------------------------------------------------------------------------------------
# the node range one table serves, and the per-(model, k, sector) cache (prompt 14)
# ---------------------------------------------------------------------------------------------


# How far above its turning point the node range is allowed to reach: a node is retained only
# where the frequency keeps at least this fraction of its leading term, ``omega^2 >= margin *
# omega_0^2``. The bare condition ``omega^2 > 0`` is not usable. On ``QCD_Cosmology`` the sign of
# the frequency near the turning point is not resolved by the background: at k = 3e8 the ratio
# ``omega^2/omega_0^2`` scatters by +-0.1 between neighbouring nodes around z ~ 4e15
# ([02-qcd-T-z-spline-node-tolerance]), the sign is not monotone in z -- the Green's-function
# frequency is positive again at the top of the production grid, z >= 1.9e16 -- and a panel whose
# two nodes are both positive can still hold an abscissa where omega^2 < 0, measured at
# z = 3.61e15, where the build raises.
#
# One half is far above that scatter and far below any anchor a producer can hand in. The cut it
# makes lies *above*, in z, the highest node at which the WKB validity criterion
# |d ln omega/dz|/omega <= 1 holds -- measured on both production models, both sectors, at
# k = 1e5 and 3e8 -- and ``WKB_phase_function`` refuses an anchor where that criterion is
# violated. So the range covers every anchor the producer accepts, with margin: production
# anchors are three e-folds inside the horizon, where the ratio exceeds 0.99.
RESIDUAL_WKB_REGION_MARGIN = 0.5


def residual_node_range(
    model,
    k: float,
    z_grid: Sequence[float],
    sector: str,
    margin: float = RESIDUAL_WKB_REGION_MARGIN,
) -> np.ndarray:
    """
    The nodes the residual table for one ``(model, k, sector)`` is built on: the background
    model's own grid, restricted at the top to the **highest node at which the Liouville-Green
    frequency is positive**, by the margin ``RESIDUAL_WKB_REGION_MARGIN`` of its leading term,
    and at the bottom to the lowest grid node.

    The range depends only on ``(model, k, sector)``, never on an object's anchor or sample set,
    so one table serves every object of that wavenumber. The top must be *found*, not assumed:
    ``phase_residual_integrand`` refuses to evaluate where ``omega^2 <= 0``, and how far up the
    grid the frequency stays positive is ``k``-dependent and model-dependent (on ``LambdaCDM``
    the Green's-function correction vanishes in the radiation era and nothing at all is cut; in
    the transfer-function sector the cut removes 250-620 of the production grid's 1,732 nodes,
    landing about 1.25 e-folds inside the horizon).

    It is found by walking the grid **upwards from its lowest node** -- where the mode is deep
    inside the horizon and the frequency is positive by construction -- and stopping at the first
    node where ``leading < 0`` or ``leading + correction < margin * leading``, tested through the
    same split functions the integrand uses so that the test and the integrand cannot disagree.
    The range is therefore the largest run of nodes reaching the bottom of the grid, not merely
    everything below the highest positive node: see ``RESIDUAL_WKB_REGION_MARGIN`` for why the
    difference matters.

    :param model: anything exposing ``.functions``
    :param k: the comoving wavenumber, in the model's units
    :param z_grid: the background model's grid, strictly descending in ``z``
    :param sector: ``"Gk"`` or ``"Tk"``
    :param margin: the fraction of the leading term the frequency must keep at a retained node
    :return: the nodes, strictly descending in ``z``
    :raises ValueError: if the lowest node of the grid already fails the test, or if fewer than
        two nodes remain
    """
    _check_sector(sector)
    leading_fn, correction_fn = _SECTOR_FUNCTIONS[sector]
    k = float(k)
    margin = float(margin)

    z = np.asarray(z_grid, dtype=float)
    if z.ndim != 1 or z.size < 2:
        raise ValueError(
            f"residual_node_range[{sector}]: at least two grid nodes are required "
            f"(got {z.size})"
        )
    if not np.all(np.diff(z) < 0.0):
        raise ValueError(
            f"residual_node_range[{sector}]: the grid must be strictly descending in z"
        )

    z_list = z.tolist()
    top = len(z_list)
    for j in range(len(z_list) - 1, -1, -1):
        z_node = z_list[j]
        leading = leading_fn(model, k, z_node)
        if leading < 0.0:
            break
        if leading + correction_fn(model, k, z_node) < margin * leading:
            break
        top = j

    if top >= len(z_list):
        raise ValueError(
            f"residual_node_range[{sector}]: the Liouville-Green frequency does not keep the "
            f"fraction {margin:.3g} of its leading term even at the lowest node of the grid, "
            f"z = {z[-1]:.8g}, for k = {k:.8g}; no part of the grid lies inside the WKB region "
            "for this wavenumber"
        )

    nodes = z[top:]
    if nodes.size < 2:
        raise ValueError(
            f"residual_node_range[{sector}]: only {nodes.size} node(s) of the grid lie inside "
            f"the WKB region for k = {k:.8g} (from z = {z[top]:.8g}); at least two are required"
        )
    return nodes


# The cache is keyed on (model, k, sector, order) and holds one CumulativeTable per key. It lives
# for the life of the worker process: Ray runs one task at a time in a worker, so no locking is
# needed. Measured footprint of a 1,732-node table -- the numpy limbs, their Python-list copies,
# the exact-node dict and the two u arrays -- is 0.395 MB (tracemalloc, ten tables on the
# production grid). A production run builds one background model (main.py:471) over ~50
# wavenumbers in two sectors, so 100 keys and ~40 MB per worker; the worst case of both
# production models in one worker would be 200 keys and ~79 MB. The cap is set above that, so it
# never binds in production and the reuse can never silently degrade into thrashing, while a
# process that walks many more wavenumbers is still bounded at ~101 MB.
RESIDUAL_CACHE_MAX_ENTRIES = 256


class _ResidualCacheEntry:
    """One cached table, with the means to confirm it belongs to the model asking for it.

    A model with a datastore id is identified by that id. A model without one --
    ``ModelProxy.store_id`` is ``None`` for an unavailable model, and every offline stand-in has
    no id at all -- is identified by the identity of the model object itself, held as a *weak*
    reference so that the cache neither keeps a dead model alive nor hands its table to a
    different object that happens to have inherited its ``id()``.
    """

    __slots__ = ("table", "_ref", "_strong", "_by_identity")

    def __init__(self, table: CumulativeTable, model, by_identity: bool):
        self.table = table
        self._by_identity = by_identity
        self._ref = None
        self._strong = None
        if by_identity:
            try:
                self._ref = weakref.ref(model)
            except TypeError:
                # a model that cannot be weak-referenced is held strongly; the cap bounds it
                self._strong = model

    def serves(self, model) -> bool:
        if not self._by_identity:
            return True
        if self._strong is not None:
            return self._strong is model
        return self._ref() is model


_RESIDUAL_CACHE: "OrderedDict[tuple, _ResidualCacheEntry]" = OrderedDict()


def phase_residual_cache_key(
    model,
    k: float,
    sector: str,
    store_id: Optional[int] = None,
    order: Optional[int] = None,
) -> tuple:
    """
    The cache key for one residual table.

    ``order`` is resolved at call time exactly as ``build_phase_residual`` resolves it, so that a
    caller who supplies nothing and a caller who supplies the current ``RHO_GAUSS_ORDER`` land on
    the same key -- and a table built at any other order lands on a different one.

    ``store_id`` is the model's datastore id (``ModelProxy.store_id``), which is ``None`` for an
    unavailable model and for every offline stand-in. A ``None`` id is **not** a key: two
    different stand-ins in one process would then share a table. Identity takes its place, and
    ``_ResidualCacheEntry.serves`` confirms it on every hit.
    """
    _check_sector(sector)
    if order is None:
        order = RHO_GAUSS_ORDER
    if store_id is not None:
        return ("store", int(store_id), float(k), sector, int(order))
    return ("obj", id(model), float(k), sector, int(order))


def cached_phase_residual(
    model,
    k: float,
    z_grid: Sequence[float],
    sector: str,
    order: Optional[int] = None,
    store_id: Optional[int] = None,
) -> Tuple[CumulativeTable, bool]:
    """
    The residual table for one ``(model, k, sector)``, built on ``residual_node_range`` the first
    time it is asked for in this process and returned unchanged thereafter.

    Transparent by construction: the nodes depend on nothing but ``(model, k, sector)``, so two
    calls that differ only in the object's anchor or sample set get the *same table object*, and
    a different wavenumber, sector or model does not. ``z_grid`` is a property of the model --
    the background tables' own nodes -- and is therefore read only when the table is built, not
    compared on a hit.

    A cached table holds the integrand, which closes over ``model``, so an entry pins its model
    in memory for as long as it lives; the weak reference in ``_ResidualCacheEntry`` is a
    correctness guard on ``id()`` reuse, not a memory one. In production that pins one
    ``BackgroundModel`` per worker, which the worker holds anyway.

    :param model: anything exposing ``.functions`` (and optionally ``.cosmology``)
    :param k: the comoving wavenumber, in the model's units
    :param z_grid: the background model's grid, strictly descending in ``z``
    :param sector: ``"Gk"`` or ``"Tk"``
    :param order: Gauss-Legendre order per panel; ``None`` means the order this run is configured
        at, resolved here (``build_phase_residual``'s note says why it is not a default argument)
    :param store_id: the model's datastore id, or ``None``
    :return: ``(table, reused)`` -- ``reused`` is ``True`` when the table came from the cache and
        this call therefore spent no integrand evaluation building it
    """
    if order is None:
        order = RHO_GAUSS_ORDER

    key = phase_residual_cache_key(model, k, sector, store_id=store_id, order=order)

    entry = _RESIDUAL_CACHE.get(key)
    if entry is not None and entry.serves(model):
        _RESIDUAL_CACHE.move_to_end(key)
        return entry.table, True

    nodes = residual_node_range(model, k, z_grid, sector)
    table = build_phase_residual(model, k, nodes, sector, order=order)

    _RESIDUAL_CACHE[key] = _ResidualCacheEntry(
        table, model, by_identity=store_id is None
    )
    _RESIDUAL_CACHE.move_to_end(key)
    while len(_RESIDUAL_CACHE) > RESIDUAL_CACHE_MAX_ENTRIES:
        _RESIDUAL_CACHE.popitem(last=False)

    return table, False


def clear_phase_residual_cache() -> None:
    """Drop every cached table. For tests; production never needs it."""
    _RESIDUAL_CACHE.clear()


def phase_residual_cache_size() -> int:
    """The number of tables currently cached."""
    return len(_RESIDUAL_CACHE)
