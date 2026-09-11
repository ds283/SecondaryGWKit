from typing import Tuple

from ComputeTargets.BackgroundModel import BackgroundModel


def _Tk_correction_terms(
    model: BackgroundModel, k: float, z: float
) -> Tuple[float, float]:
    """
    The two non-leading terms ``B`` and ``C`` of ``Tk_omegaEff_sq``, returned separately so that
    ``Tk_omegaEff_sq`` can still form ``A + B + C`` in its original order (the returned value is a
    stored column, ``TkWKBValue.omega_WKB_sq``, and must not move by a bit) while
    ``Tk_omegaEff_sq_correction`` can hand ``B + C`` to the phase residual.

    Neither term involves ``k``.
    """
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z

    w = model.functions.wPerturbations(z)
    wPrime = model.functions.d_wPerturbations_dz(z)
    eps = model.functions.epsilon(z)
    epsPrime = model.functions.d_epsilon_dz(z)

    B = (3.0 / 2.0 * wPrime - epsPrime / 2.0) / one_plus_z
    C = (
        3.0 / 2.0 * (1.0 + eps) * (1.0 + w)
        - eps * (3.0 + eps / 2.0) / 2.0
        - 9.0 / 4.0 * (1.0 + w) * (1.0 + w)
    ) / one_plus_z_2

    return B, C


def Tk_omegaEff_sq_leading(model: BackgroundModel, k: float, z: float) -> float:
    """
    The leading part of the transfer-function Liouville-Green frequency, ``w (k/H)^2`` with
    ``w = c_s^2 = wPerturbations`` (the author's convention in this sector).

    Its square root is ``k c_s/H``, so the leading part of the transfer-function phase is ``k``
    times the **sound horizon** ``tau_s = int c_s dz/H``: again a ``k``-independent function of
    redshift, tabulated once per background model as ``BackgroundModel.functions.cs_tau``
    (review §12.2, §12.7; ``prompts/GkTk-remedial/README.md`` §2 (a)).
    """
    H = model.functions.Hubble(z)
    w = model.functions.wPerturbations(z)

    k_over_H = k / H
    k_over_H_2 = k_over_H * k_over_H

    return w * k_over_H_2


def Tk_omegaEff_sq_correction(model: BackgroundModel, k: float, z: float) -> float:
    """
    The non-leading part of the transfer-function Liouville-Green frequency, ``C_T = B + C`` in
    the notation of ``Tk_omegaEff_sq``.

    **It does not depend on ``k``** -- the argument is accepted only so that the three
    ``Tk_omegaEff_sq*`` functions share one signature. Unlike the Green's function's ``C``, this
    does *not* vanish in exact radiation: with ``eps = 2``, ``w = 1/3`` and ``eps' = w' = 0`` it
    is ``-2/s^2``, and the residual it generates,
    ``rho_T = int C_T/(omega_T + k c_s/H) dz = 1/x - 1/x_i`` asymptotically with ``x = k c_s
    tau``, is about -0.09 rad on both production models at every ``k``. It is not negligible and
    must be carried (review §12.2, §12.4).

    **Never form this by subtraction** of ``Tk_omegaEff_sq`` and ``w (k/H)^2``: the leading term
    is 1e12-1e24 times the rest (``RECONCILIATION.md`` §2 item 3).
    """
    B, C = _Tk_correction_terms(model, k, z)

    return B + C


def Tk_omegaEff_sq(model: BackgroundModel, k: float, z: float) -> float:
    """
    The transfer-function Liouville-Green frequency squared, ``A + B + C``: unchanged in value
    from before the leading/correction split, down to the last bit (it is the stored column
    ``TkWKBValue.omega_WKB_sq``).

    As in ``Gk_omegaEff_sq``, this is ``Tk_omegaEff_sq_leading + B + C`` and *not*
    ``Tk_omegaEff_sq_leading + Tk_omegaEff_sq_correction``: floating-point addition is not
    associative and the two differ in the last ulp over part of the production range. The
    summation order here is the one that was measured and stored.
    """
    B, C = _Tk_correction_terms(model, k, z)

    return Tk_omegaEff_sq_leading(model, k, z) + B + C


def Tk_d_ln_omegaEff_dz(model: BackgroundModel, k: float, z: float) -> float:
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z
    one_plus_z_3 = one_plus_z_2 * one_plus_z

    H = model.functions.Hubble(z)
    w = model.functions.wPerturbations(z)
    wPrime = model.functions.d_wPerturbations_dz(z)
    wPrimePrime = model.functions.d2_wPerturbations_dz2(z)
    eps = model.functions.epsilon(z)
    epsPrime = model.functions.d_epsilon_dz(z)
    epsPrimePrime = model.functions.d2_epsilon_dz2(z)

    k_over_H = k / H
    k_over_H_2 = k_over_H * k_over_H

    omega_eff_sq = Tk_omegaEff_sq(model, k, z)

    A = wPrime * k_over_H_2
    B = (
        3.0 / 2.0 * wPrimePrime - epsPrimePrime / 2.0 - 2.0 * eps * w * k_over_H_2
    ) / one_plus_z
    C = (
        epsPrime / 2.0 * (3.0 * w - eps + 1.0)
        # the coefficient here is 3(1+w), not (3/2)(1+w): it collects (3/2)(1+eps) w' - (9/2)(1+w) w' - (3/2) w'
        # from differentiating omega_eff^2. The handwritten notes (NUM 09 p.6) carry a (3/2) slip that
        # propagated here; corrected 2026-09-07 (docs/spec/01-transfer-function.md R30).
        + 3.0 / 2.0 * wPrime * (eps - 3.0 * (1.0 + w))
    ) / one_plus_z_2
    D = (
        -(
            3.0 * (1.0 + eps) * (1.0 + w)
            - eps * (3.0 + eps / 2.0)
            - 9.0 / 2.0 * (1.0 + w) * (1.0 + w)
        )
        / one_plus_z_3
    )

    numerator = A + B + C + D
    denominator = 2.0 * omega_eff_sq

    return numerator / denominator
