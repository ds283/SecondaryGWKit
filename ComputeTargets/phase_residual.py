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

No Ray and no datastore: this module holds the integrand and the table builder only. The
producers (prompts 06 and 07) own the decision of which nodes to build over and where to anchor.
"""

from math import sqrt
from typing import Callable, Sequence

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
    order: int = RHO_GAUSS_ORDER,
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
    :param order: Gauss-Legendre order per panel; defaults to the measured ``RHO_GAUSS_ORDER``
    :return: the ``CumulativeTable``, whose ``delta(z_i, z)`` is ``rho(z; z_i)``
    """
    _check_sector(sector)

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
