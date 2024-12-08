from ComputeTargets.BackgroundModel import BackgroundModel


def Gk_omegaEff_sq(model: BackgroundModel, k: float, z: float) -> float:
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z

    H = model.functions.Hubble(z)
    eps = model.functions.epsilon(z)
    epsPrime = model.functions.d_epsilon_dz(z)

    k_over_H = k / H
    k_over_H_2 = k_over_H * k_over_H

    A = k_over_H_2
    B = -epsPrime / 2.0 / one_plus_z
    C = (3.0 * eps / 2.0 - eps * eps / 4.0 - 2.0) / one_plus_z_2

    return A + B + C


def Gk_d_ln_omegaEffPrime_dz(model: BackgroundModel, k: float, z: float) -> float:
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
