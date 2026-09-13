from typing import Tuple

from ComputeTargets.BackgroundModel import BackgroundModel


def _Gk_correction_terms(
    model: BackgroundModel, k: float, z: float
) -> Tuple[float, float]:
    """
    The two non-leading terms ``B`` and ``C`` of ``Gk_omegaEff_sq``, returned separately so that
    ``Gk_omegaEff_sq`` can still form ``A + B + C`` in its original order (the returned value is a
    stored column, ``GkWKBValue.omega_WKB_sq``, and must not move by a bit) while
    ``Gk_omegaEff_sq_correction`` can hand ``B + C`` to the phase residual.

    Neither term involves ``k``.
    """
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z

    eps = model.functions.epsilon(z)
    epsPrime = model.functions.d_epsilon_dz(z)

    B = -epsPrime / 2.0 / one_plus_z
    C = (3.0 * eps / 2.0 - eps * eps / 4.0 - 2.0) / one_plus_z_2

    return B, C


def Gk_omegaEff_sq_leading(model: BackgroundModel, k: float, z: float) -> float:
    """
    The leading part of the Green's-function Liouville-Green frequency, ``(k/H)^2``.

    This is the term whose integral is the *shared* primitive: ``int (k/H) dz = k tau``, so the
    leading part of the phase is ``k`` times a ``k``-independent function of redshift, tabulated
    once per background model as ``BackgroundModel.functions.tau`` (review §6 (d), §7;
    ``prompts/GkTk-remedial/README.md`` §2 (a)).
    """
    H = model.functions.Hubble(z)

    k_over_H = k / H

    return k_over_H * k_over_H


def Gk_omegaEff_sq_correction(model: BackgroundModel, k: float, z: float) -> float:
    """
    The non-leading part of the Green's-function Liouville-Green frequency,
    ``C(z) = -eps'/(2 s) + (3 eps/2 - eps^2/4 - 2)/s^2`` with ``s = 1+z``.

    **It does not depend on ``k``** -- the argument is accepted only so that the three
    ``Gk_omegaEff_sq*`` functions share one signature. That ``k``-independence is why the
    residual of the phase is cheap: the only ``k`` dependence of
    ``rho = int C/(omega + k/H) dz`` is through the denominator, and ``C`` itself vanishes
    identically in exact radiation (``eps = 2``, ``eps' = 0``), which is why ``rho`` is
    2.5e-7 rad over the whole WKB range on LambdaCDM and 1.5e-3 rad on ``QCD_Cosmology``
    (review §6).

    **Never form this by subtraction.** ``Gk_omegaEff_sq`` returns ``A + B + C`` with
    ``A = (k/H)^2`` between 1e12 and 1e24 times the rest, so ``omega_sq - (k/H)^2`` would lose
    every digit of it (``RECONCILIATION.md`` §2 item 3).
    """
    B, C = _Gk_correction_terms(model, k, z)

    return B + C


def Gk_omegaEff_sq(model: BackgroundModel, k: float, z: float) -> float:
    """
    The Green's-function Liouville-Green frequency squared, ``A + B + C``: unchanged in value
    from before the leading/correction split, down to the last bit (it is the stored column
    ``GkWKBValue.omega_WKB_sq``).

    Note that this is ``Gk_omegaEff_sq_leading + B + C`` and *not*
    ``Gk_omegaEff_sq_leading + Gk_omegaEff_sq_correction``: floating-point addition is not
    associative, and the two differ in the last ulp on about a fifth of the production range.
    The summation order here is the one that was measured and stored.
    """
    B, C = _Gk_correction_terms(model, k, z)

    return Gk_omegaEff_sq_leading(model, k, z) + B + C


def Gk_d_ln_omegaEff_dz(model: BackgroundModel, k: float, z: float) -> float:
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z
    one_plus_z_3 = one_plus_z_2 * one_plus_z

    H = model.functions.Hubble(z)
    eps = model.functions.epsilon(z)
    epsPrime = model.functions.d_epsilon_dz(z)
    epsPrimePrime = model.functions.d2_epsilon_dz2(z)

    k_over_H = k / H
    k_over_H_2 = k_over_H * k_over_H

    omega_eff_sq = Gk_omegaEff_sq(model, k, z)

    A = (-epsPrimePrime / 2.0 - 2.0 * eps * k_over_H_2) / one_plus_z
    B = (2.0 * epsPrime - eps * epsPrime / 2.0) / one_plus_z_2
    C = -(3.0 * eps - eps * eps / 2.0 - 4.0) / one_plus_z_3

    numerator = A + B + C
    denominator = 2.0 * omega_eff_sq

    return numerator / denominator
