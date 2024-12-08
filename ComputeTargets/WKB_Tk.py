from ComputeTargets.BackgroundModel import BackgroundModel


def Tk_omegaEff_sq(model: BackgroundModel, k: float, z: float) -> float:
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z

    H = model.functions.Hubble(z)
    w = model.functions.wPerturbations(z)
    wPrime = model.functions.d_wPerturbations_dz(z)
    eps = model.functions.epsilon(z)
    epsPrime = model.functions.d_epsilon_dz(z)

    k_over_H = k / H
    k_over_H_2 = k_over_H * k_over_H

    A = w * k_over_H_2
    B = (3.0 / 2.0 * wPrime - epsPrime / 2.0) / one_plus_z
    C = (
        3.0 / 2.0 * (1.0 + eps) * (1.0 + w)
        - eps * (3.0 + eps / 2.0) / 2.0
        - 9.0 / 4.0 * (1.0 + w) * (1.0 + w)
    ) / one_plus_z_2

    return A + B + C


def Tk_d_ln_omegaEffPrime_dz(model: BackgroundModel, k: float, z: float) -> float:
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
        + 3.0 / 2.0 * wPrime * (eps - 3.0 / 2.0 * (1.0 + w))
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
