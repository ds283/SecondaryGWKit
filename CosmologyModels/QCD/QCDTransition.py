import numpy as np

# Fitting functions from Appendix C of Saikawa & Shirai https://arxiv.org/pdf/1803.01038

# Energy density coefficients
a_coeffs = [
    1,
    1.11724,
    3.12672e-1,
    -4.680e-2,
    -2.65004e-2,
    -1.19760e-3,
    1.82812e-4,
    1.36436e-4,
    8.55051e-5,
    1.2284e-5,
    3.82259e-7,
    -6.87035e-9,
]

b_coeffs = [
    1.43382e-2,
    1.37559e-2,
    2.92108e-3,
    -5.38533e-4,
    -1.62496e-4,
    -2.87906e-5,
    -3.84278e-6,
    2.78776e-6,
    7.40342e-7,
    1.1721e-7,
    3.72499e-9,
    -6.74107e-11,
]

# Entropy density coefficients
c_coeffs = [
    1,
    6.07869e-1,
    -1.54485e-1,
    -2.24034e-1,
    -2.82147e-2,
    2.9062e-2,
    6.86778e-3,
    -1.00005e-3,
    -1.69104e-4,
    1.06301e-5,
    1.69528e-6,
    -9.33311e-8,
]

d_coeffs = [
    7.07388e1,
    9.18011e1,
    3.31892e1,
    -1.39779,
    -1.52558,
    -1.97857e-2,
    -1.60146e-1,
    8.22615e-5,
    2.02651e-2,
    -1.82134e-5,
    7.83943e-5,
    7.13518e-5,
]

# Particle masses in GeV
m_e = 511e-6
m_mu = 0.1056
m_pi0 = 0.135
m_piplus = m_piminus = 0.140
m_1 = 0.5
m_2 = 0.77
m_3 = 1.2
m_4 = 2.0


def polynomial_sum(coeffs: list, x: float) -> float:
    """Compute sum of polynomial terms"""
    return sum(c * x**i for i, c in enumerate(coeffs))


# Fitting functions for energy density
def f_rho(x: float) -> float:
    """Low temperature fitting function for fermion energy density"""
    return np.exp(-1.04855 * x) * (1 + 1.03757 * x + 0.508630 * x**2 + 0.0893988 * x**3)


def b_rho(x: float) -> float:
    """Low temperature fitting function for boson energy density"""
    return np.exp(-1.03149 * x) * (1 + 1.03317 * x + 0.398264 * x**2 + 0.0648056 * x**3)


# Fitting functions for entropy density
def f_s(x: float) -> float:
    """Low temperature fitting function for fermion entropy"""
    return np.exp(-1.04190 * x) * (1 + 1.03400 * x + 0.456426 * x**2 + 0.0595249 * x**3)


def b_s(x: float) -> float:
    """Low temperature fitting function for boson entropy"""
    return np.exp(-1.03365 * x) * (1 + 1.03397 * x + 0.342548 * x**2 + 0.0506182 * x**3)


def S_fit(x: float) -> float:
    """Combined entropy fitting function"""
    return 1 + 7 / 4 * f_s(x)


# Complete effective degrees of freedom functions
def G(T: float) -> float:
    """Complete effective d.o.f for energy density across all temperatures"""
    if T > 1e16:
        return 106.75  # Asymptotic high temperature limit
    elif 0.12 <= T <= 1e16:
        lnT = np.log(T)
        return polynomial_sum(a_coeffs, lnT) / polynomial_sum(b_coeffs, lnT)
    elif 1e-5 <= T < 0.12:
        return (
            2.030
            + 1.353 * S_fit(m_e / T) ** (4 / 3)
            + 3.495 * f_rho(m_e / T)
            + 3.446 * f_rho(m_mu / T)
            + 1.05 * b_rho(m_pi0 / T)
            + 2.08 * b_rho(m_piplus / T)
            + 4.165 * b_rho(m_1 / T)
            + 30.55 * b_rho(m_2 / T)
            + 89.4 * b_rho(m_3 / T)
            + 8209 * b_rho(m_4 / T)
        )
    else:
        return 3.36  # Low temperature limit



def Gs(T: float) -> float:
    """Complete effective d.o.f for entropy density across all temperatures"""
    if T > 1e16:
        return 106.75  # Asymptotic high temperature limit
    elif 0.12 <= T <= 1e16:
        lnT = np.log(T)
        return G(T) * polynomial_sum(d_coeffs, lnT) / (polynomial_sum(d_coeffs, lnT) + polynomial_sum(c_coeffs, lnT))
    elif 1e-5 <= T < 0.12:
        return (
            2.008
            + 1.923 * S_fit(m_e / T) ** (4 / 3)
            + 3.442 * f_s(m_e / T)
            + 3.468 * f_s(m_mu / T)
            + 1.034 * b_s(m_pi0 / T)
            + 2.068 * b_s(m_piplus / T)
            + 4.16 * b_s(m_1 / T)
            + 30.55 * b_s(m_2 / T)
            + 90 * b_s(m_3 / T)
            + 6209 * b_s(m_4 / T)
        )
    else:
        return 3.91  # Low temperature limit



def w_of_T(T: float) -> float:
    """Equation of state parameter as function of temperature"""
    G_val = G(T)  # Use complete G function
    Gs_val = Gs(T)  # Use complete Gs function
    return 4 * Gs_val / (3 * G_val) - 1
