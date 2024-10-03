from CosmologyModels import BaseCosmology


def WKB_omegaEff_sq(cosmology: BaseCosmology, k: float, z: float):
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z

    H = cosmology.Hubble(z)
    eps = cosmology.epsilon(z)
    epsPrime = cosmology.d_epsilon_dz(z)

    k_over_H = k / H
    k_over_H_2 = k_over_H * k_over_H

    A = k_over_H_2
    B = -epsPrime / 2.0 / one_plus_z
    C = (3.0 * eps / 2.0 - eps * eps / 4.0 - 2.0) / one_plus_z_2

    return A + B + C


def WKB_d_ln_omegaEffPrime_dz(cosmology: BaseCosmology, k: float, z: float):
    one_plus_z = 1.0 + z
    one_plus_z_2 = one_plus_z * one_plus_z
    one_plus_z_3 = one_plus_z_2 * one_plus_z

    eps = cosmology.epsilon(z)
    epsPrime = cosmology.d_epsilon_dz(z)
    epsPrimePrime = cosmology.d2_epsilon_dz2(z)

    omega_eff_sq = WKB_omegaEff_sq(cosmology, k, z)

    A = -epsPrimePrime / 2.0 / one_plus_z
    B = (2.0 * epsPrimePrime - eps * epsPrime / 2.0) / one_plus_z_2
    C = (3.0 * eps - eps * eps / 2.0 - 4.0) / one_plus_z_3

    numerator = A + B + C
    denominator = 2.0 * omega_eff_sq

    return numerator / denominator
