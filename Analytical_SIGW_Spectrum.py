"""
Analytical Scalar-Induced Gravitational Wave (SIGW) Calculator
================================================================
This code computes the tensor power spectrum and energy density of 
scalar-induced gravitational waves during radiation domination (w=1/3).

Based on the analytical formulation with:
- Dimensionless power spectrum: 𝒫_h(k) = (k³/2π²) P_h(k)
- All tilde variables are dimensionless: k̃ = k/k_eq
- Source integral I with analytical Bessel function solution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import jn, yn
from scipy.constants import c
from dataclasses import dataclass
from typing import Tuple, Callable, Optional
import warnings
warnings.filterwarnings('ignore')

# LaTeX settings for publication-quality plots
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{xcolor}\usepackage{amssymb}\boldmath'
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.4*plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = 1.4*plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = 1.4*plt.rcParams['font.size']
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.minor.size'] = 3


@dataclass
class CosmologicalParameters:
    """
    Standard Planck 2018 cosmological parameters.
    All parameters are dimensionless or in standard units.
    """
    # Density parameters
    Omega_r: float = 9.1e-5       # Radiation density parameter today
    Omega_m: float = 0.315        # Matter density parameter today
    Omega_Lambda: float = 0.6847  # Dark energy density parameter today
    
    # Hubble and redshift parameters
    h: float = 0.674              # Reduced Hubble parameter (H₀ = 100h km/s/Mpc)
    z_eq: float = 3387            # Redshift of matter-radiation equality
    
    # Primordial power spectrum parameters
    A_zeta: float = 2.1e-9        # Primordial scalar amplitude at pivot scale
    k_star: float = 0.05          # Pivot scale in Mpc⁻¹
    n_s: float = 1.0              # Spectral index (1.0 = scale invariant)
    
    # Equation of state parameters
    w_star: float = 1/3           # EoS during radiation domination
    cs: float = 1/np.sqrt(3)      # Sound speed in radiation (c_s = 1/√3)
    b: float = 0.0                # b=0 for radiation domination
    
    def __post_init__(self):
        """Validate parameters after initialization"""
        total_omega = self.Omega_r + self.Omega_m + self.Omega_Lambda
        if not np.isclose(total_omega, 1.0, rtol=1e-3):
            print(f"Warning: Ω_total = {total_omega:.4f} ≠ 1")


class AnalyticalSIGWCalculator:
    """
    Calculator for analytical tensor power spectrum from scalar-induced 
    gravitational waves. Uses consistent dimensionless variables throughout.
    
    The dimensionless power spectrum is:
    𝒫_h(k) = (k³/2π²) P_h(k)
    
    where P_h(k) is the dimensional power spectrum.
    """
    
    def __init__(self, cosmo_params: Optional[CosmologicalParameters] = None):
        """
        Initialize calculator with cosmological parameters.
        
        Parameters:
        -----------
        cosmo_params : CosmologicalParameters, optional
            Cosmological parameters to use. If None, uses Planck 2018 values.
        """
        self.params = cosmo_params or CosmologicalParameters()
        
        # Derived physical quantities
        self.H0 = 100 * self.params.h                    # Hubble constant in km/s/Mpc
        self.H0_Mpc = self.H0 / 2.998e5                  # H₀ in Mpc⁻¹ (using c = 2.998e5 km/s)
        
        # Matter-radiation equality scale
        # k_eq = a_eq × H_eq = a₀H₀ × √(2Ω_m) × √(1+z_eq)
        self.k_eq_Mpc = self.H0_Mpc * np.sqrt(2 * self.params.Omega_m) * np.sqrt(1 + self.params.z_eq)
        
        # Dimensionless factor: k_eq/(a₀H₀) ≈ 46.71
        self.k_eq_over_H0 = np.sqrt(2 * self.params.Omega_m) * np.sqrt(1 + self.params.z_eq)
        
        # Key prefactor for the dimensionless power spectrum
        # 324 × (k_eq/(a₀H₀))⁴ ≈ 1.5423 × 10⁹
        self.power_prefactor = 324 * self.k_eq_over_H0**4
        
        # Print initialization info
        self._print_initialization_info()
    
    def _print_initialization_info(self):
        """Print key parameters for verification"""
        print("="*60)
        print("SIGW Calculator Initialized")
        print("="*60)
        print(f"Cosmological Parameters:")
        print(f"  Ω_r = {self.params.Omega_r:.2e}")
        print(f"  Ω_m = {self.params.Omega_m:.3f}")
        print(f"  Ω_Λ = {self.params.Omega_Lambda:.3f}")
        print(f"  h = {self.params.h:.3f}")
        print(f"  z_eq = {self.params.z_eq:.0f}")
        print(f"\nDerived Quantities:")
        print(f"  H₀ = {self.H0:.1f} km/s/Mpc = {self.H0_Mpc:.3e} Mpc⁻¹")
        print(f"  k_eq = {self.k_eq_Mpc:.3e} Mpc⁻¹")
        print(f"  k_eq/(a₀H₀) = {self.k_eq_over_H0:.2f}")
        print(f"  324 × (k_eq/(a₀H₀))⁴ = {self.power_prefactor:.3e}")
        print("="*60 + "\n")
    
    def f(self, z: float) -> float:
        """
        Dimensionless Hubble function: f(z) = H(z)/H₀
        
        f(z) = √[Ω_r(1+z)⁴ + Ω_m(1+z)³ + Ω_Λ]
        
        Parameters:
        -----------
        z : float
            Redshift
            
        Returns:
        --------
        float
            H(z)/H₀
        """
        return np.sqrt(
            self.params.Omega_r * (1 + z)**4 + 
            self.params.Omega_m * (1 + z)**3 + 
            self.params.Omega_Lambda
        )
    
    def conformal_time_integral(self, z: float) -> float:
        """
        Compute the conformal time integral: ∫_{∞}^{z} -dz'/f(z')
        
        This appears in the source integral I(q,r,z).
        
        Parameters:
        -----------
        z : float
            Redshift
            
        Returns:
        --------
        float
            Conformal time integral value
        """
        def integrand(zp):
            return -1.0 / self.f(zp)
        
        # Integrate from large z (effectively infinity) to z
        result, error = quad(integrand, 1e6, z, limit=200, epsrel=1e-10)
        
        if abs(error/result) > 1e-3:
            print(f"Warning: Large integration error in conformal time: {error/result:.2e}")
        
        return result
    
    def scalar_power_spectrum(self, k: float) -> float:
        """
        Primordial scalar power spectrum 𝒫_ζ(k).
        
        For scale-invariant spectrum: 𝒫_ζ(k) = A_ζ × (k/k_*)^(n_s-1)
        
        Parameters:
        -----------
        k : float
            Wavenumber in Mpc⁻¹
            
        Returns:
        --------
        float
            Scalar power spectrum value
        """
        return self.params.A_zeta * (k / self.params.k_star)**(self.params.n_s - 1)
    
    def source_integral_I(self, k_tilde: float, q_tilde: float, 
                         r_tilde: float, z: float) -> float:
        """
        Compute the dimensionless source integral Ĩ(k̃,q̃,r̃,z) × (a₀H₀)².
        
        From the analytical formula:
        I(q,r,τ) = (1/8)√(3π³/2) × [(-k²+q²+r²)²/(q³r³√k)] × (1/√τ) × 
                   [Y_{1/2}(kτ) + (J_{1/2}(kτ)/π) × ln|...|]
        
        where we need to apply sound speed corrections: q → c_s×q, r → c_s×r
        
        Parameters:
        -----------
        k_tilde, q_tilde, r_tilde : float
            Dimensionless wavenumbers (k/k_eq, q/k_eq, r/k_eq)
        z : float
            Redshift
            
        Returns:
        --------
        float
            Dimensionless source integral value × (a₀H₀)²
        """
        # Apply sound speed to scalar modes
        q_cs = q_tilde * self.params.cs
        r_cs = r_tilde * self.params.cs
        
        # Triangle condition with sound speed
        if abs(q_cs - r_cs) > k_tilde or k_tilde > q_cs + r_cs:
            return 0.0
        
        # Compute conformal time integral
        tau_integral = self.conformal_time_integral(z)
        if abs(tau_integral) < 1e-15:
            return 0.0
        
        # Bessel function argument
        x = k_tilde * self.f(self.params.z_eq) / (1 + self.params.z_eq) * abs(tau_integral)
        
        if x <= 0 or x > 1e4:  # Numerical stability check
            return 0.0
        
        # Momentum factor (using sound speed corrected values)
        momentum_factor = (-k_tilde**2 + q_cs**2 + r_cs**2)**2 / \
                         (q_cs**3 * r_cs**3 * np.sqrt(k_tilde))
        
        # Logarithmic term
        numerator = (k_tilde + q_cs + r_cs) * (-k_tilde + q_cs + r_cs)
        denominator = (k_tilde + q_cs - r_cs) * (k_tilde - q_cs + r_cs)
        
        if abs(denominator) < 1e-15:
            return 0.0
        
        log_term = np.log(abs(numerator / denominator))
        
        # Bessel functions J_{1/2} and Y_{1/2}
        # J_{1/2}(x) = √(2/(πx)) × sin(x)
        # Y_{1/2}(x) = -√(2/(πx)) × cos(x)
        J_half = np.sqrt(2/(np.pi * x)) * np.sin(x)
        Y_half = -np.sqrt(2/(np.pi * x)) * np.cos(x)
        
        # Combine all factors for the dimensional source integral
        base_prefactor = (1/8) * np.sqrt(3 * np.pi**3 / 2)
        
        # Dimensional conversion factors
        dimension_factor = np.sqrt(self.k_eq_over_H0) * \
                          (1 + self.params.z_eq) / self.f(self.params.z_eq)
        
        # Time factor
        time_factor = 1 / np.sqrt(abs(tau_integral))
        
        # Complete source integral
        result = base_prefactor * dimension_factor * momentum_factor * time_factor * \
                 (Y_half + (J_half / np.pi) * log_term)
        
        return result
    
    def theta_integrand(self, theta: float, k_tilde: float, q_tilde: float, 
                       z: float) -> float:
        """
        Integrand for the angular (θ) integration.
        
        Includes the factor: sin⁵(θ) × 𝒫_ζ(r̃×k_eq)/r̃³ × Ĩ²
        where r̃ = √(k̃² + q̃² - 2k̃q̃cos(θ))
        
        Parameters:
        -----------
        theta : float
            Angle between k and q vectors
        k_tilde, q_tilde : float
            Dimensionless wavenumbers
        z : float
            Redshift
            
        Returns:
        --------
        float
            Value of angular integrand
        """
        # Compute r_tilde using law of cosines
        r_tilde_sq = k_tilde**2 + q_tilde**2 - 2 * k_tilde * q_tilde * np.cos(theta)
        
        if r_tilde_sq <= 0:
            return 0.0
        
        r_tilde = np.sqrt(r_tilde_sq)
        
        if r_tilde < 1e-10:  # Avoid division by zero
            return 0.0
        
        # Compute source integral (returns Ĩ × (a₀H₀)²)
        I_tilde = self.source_integral_I(k_tilde, q_tilde, r_tilde, z)
        
        # Get scalar power spectrum at r
        P_zeta_r = self.scalar_power_spectrum(r_tilde * self.k_eq_Mpc)
        
        # Complete integrand with sin⁵(θ) from angular measure
        integrand = (P_zeta_r / r_tilde**3) * np.sin(theta)**5 * I_tilde**2
        
        return integrand
    
    def compute_dimensionless_power_spectrum(self, k_tilde: float, 
                                            z: float = 1000,
                                            q_limits: Tuple[float, float] = (0.01, 100),
                                            nq: int = 80,
                                            verbose: bool = False) -> float:
        """
        Compute the dimensionless tensor power spectrum 𝒫_h(k̃).
        
        From the analytical formula:
        𝒫_h(k̃) = 1.5423×10⁹ × k̃³ × [(1+3w*)/(5+3w*)]⁴ × 
                  ∫ q̃³ 𝒫_ζ(q̃k_eq) × [∫ sin⁵(θ) × ...] dq̃
        
        Parameters:
        -----------
        k_tilde : float
            Dimensionless wavenumber k/k_eq
        z : float
            Redshift (default: 1000, deep in radiation era)
        q_limits : tuple
            Integration limits for q̃ as (q_min/k, q_max/k)
        nq : int
            Number of q points for integration
        verbose : bool
            Print progress information
            
        Returns:
        --------
        float
            Dimensionless tensor power spectrum 𝒫_h(k̃)
        """
        # Mode factor for radiation domination
        w = self.params.w_star
        mode_factor = ((1 + 3*w) / (5 + 3*w))**4  # = (4/3)⁴ = 256/81 for w=1/3
        
        # Complete prefactor: 1.5423×10⁹ for our parameters
        prefactor = self.power_prefactor * mode_factor
        
        if verbose:
            print(f"Computing 𝒫_h for k̃ = {k_tilde:.2e}")
            print(f"  Prefactor = {prefactor:.3e}")
        
        # Set up q integration grid (logarithmic for better sampling)
        q_min = k_tilde * q_limits[0]
        q_max = k_tilde * q_limits[1]
        q_grid = np.logspace(np.log10(max(q_min, 1e-6)), np.log10(q_max), nq)
        
        integral_sum = 0.0
        
        for i in range(len(q_grid) - 1):
            q_tilde = (q_grid[i] + q_grid[i+1]) / 2  # Midpoint rule
            dq = q_grid[i+1] - q_grid[i]
            
            # Compute angular integral
            theta_integral, theta_error = quad(
                lambda theta: self.theta_integrand(theta, k_tilde, q_tilde, z),
                0, np.pi, 
                limit=100,
                epsrel=1e-8
            )
            
            # Check integration error
            if abs(theta_integral) > 0 and theta_error > 1e-6 * abs(theta_integral):
                if verbose:
                    print(f"  Warning: Large θ integration error at q̃={q_tilde:.2e}")
            
            # Get scalar power spectrum at q
            P_zeta_q = self.scalar_power_spectrum(q_tilde * self.k_eq_Mpc)
            
            # Add contribution to sum
            integral_sum += q_tilde**3 * P_zeta_q * theta_integral * dq
        
        # Final dimensionless power spectrum
        P_h_dimensionless = prefactor * k_tilde**3 * integral_sum
        
        if verbose:
            print(f"  𝒫_h(k̃={k_tilde:.2e}) = {P_h_dimensionless:.2e}")
        
        return P_h_dimensionless
    
    def compute_omega_gw(self, k_tilde: float, z: float = 0,
                        **kwargs) -> float:
        """
        Compute the gravitational wave energy density parameter Ω_GW(k).
        
        The correct relation with dimensionless power spectrum:
        Ω_GW = (1/12) × (k/(aH))² × 𝒫_h(k)
        
        where 𝒫_h(k) is the dimensionless power spectrum.
        Factor 1/12 for both polarizations.
        
        Parameters:
        -----------
        k_tilde : float
            Dimensionless wavenumber k/k_eq
        z : float
            Redshift for evaluation (default: 0, present day)
        **kwargs : dict
            Additional arguments passed to compute_dimensionless_power_spectrum
            
        Returns:
        --------
        float
            Ω_GW(k) at the given redshift
        """
        # Get dimensionless power spectrum (computed in radiation era)
        P_h_dimensionless = self.compute_dimensionless_power_spectrum(
            k_tilde, z=1000, **kwargs
        )
        
        # Correct formula with (1+z) factor
        omega_gw = (1/12) * (k_tilde * self.k_eq_over_H0 * (1+z) / self.f(z))**2 * P_h_dimensionless
        
        return omega_gw
    
    def compute_h2_omega_gw(self, f_Hz: float, **kwargs) -> float:
        """
        Compute h²Ω_GW(f) for a given frequency in Hz.
        
        Uses the relation: k = 2πf/c with c in appropriate units.
        
        Parameters:
        -----------
        f_Hz : float
            Frequency in Hz
        **kwargs : dict
            Additional arguments for computation
            
        Returns:
        --------
        float
            h²Ω_GW at the given frequency
        """
        # Convert frequency to wavenumber
        # k [Mpc⁻¹] ≈ 2πf [Hz] / (9.72 × 10⁻¹⁵) 
        k_Mpc = 2 * np.pi * f_Hz / 9.72e-15
        k_tilde = k_Mpc / self.k_eq_Mpc
        
        omega_gw = self.compute_omega_gw(k_tilde, **kwargs)
        
        return self.params.h**2 * omega_gw


def verify_source_integral():
    """
    Verify the source integral computation with specific test cases.
    """
    print("\n" + "="*60)
    print("Verifying Source Integral Implementation")
    print("="*60)
    
    calc = AnalyticalSIGWCalculator()
    
    # Test cases
    test_cases = [
        (1.0, 0.5, 0.8, 100),   # k̃=1, q̃=0.5, r̃=0.8, z=100
        (0.1, 0.05, 0.08, 1000), # Small k values
        (10, 5, 8, 100),         # Large k values
    ]
    
    print("\nTest results for source integral Ĩ × (a₀H₀)²:")
    print("-" * 60)
    
    for k_test, q_test, r_test, z_test in test_cases:
        # Check triangle condition with sound speed
        q_cs = q_test * calc.params.cs
        r_cs = r_test * calc.params.cs
        triangle_ok = (abs(q_cs - r_cs) <= k_test <= q_cs + r_cs)
        
        if triangle_ok:
            I_value = calc.source_integral_I(k_test, q_test, r_test, z_test)
            print(f"k̃={k_test:4.1f}, q̃={q_test:4.2f}, r̃={r_test:4.2f}, z={z_test:4.0f} "
                  f"→ Ĩ = {I_value:+.4e}")
        else:
            print(f"k̃={k_test:4.1f}, q̃={q_test:4.2f}, r̃={r_test:4.2f}, z={z_test:4.0f} "
                  f"→ Triangle condition violated")
    
    print("-" * 60)
    
    # Check conformal time integral
    print("\nConformal time integral checks:")
    z_values = [10, 100, 1000, 3387]
    for z in z_values:
        tau_int = calc.conformal_time_integral(z)
        print(f"  ∫_∞^{z:4.0f} -dz'/f(z') = {tau_int:.4e}")
    
    print("\n" + "="*60)


# ============= ANALYTICAL SENSITIVITY CURVES (EMBEDDED) =============
# Based on PyCBC analytical space PSDs
# Source: https://pycbc.org/pycbc/latest/html/_modules/pycbc/psd/analytical_space.html

def omega_length(f, len_arm):
    """The function to calculate 2*pi*f*arm_length/c."""
    return 2*np.pi*f * len_arm/c

def averaged_fplus_sq_approximated(f, len_arm):
    """Simplified fit for TDI-based space-borne GW detectors' squared antenna response."""
    fp_sq_approx = (3./20.)*(1./(1.+0.6*omega_length(f, len_arm)**2))
    return fp_sq_approx

def averaged_tianqin_fplus_sq_numerical(f, len_arm=np.sqrt(3)*1e8):
    """Numerical fit for TianQin's squared antenna response function."""
    base = averaged_fplus_sq_approximated(f, len_arm)
    a = [1, 1e-4, 2639e-4, 231/5*1e-4, -2093/1.25*1e-4, 2173e-5,
         2101e-6, 3027/2*1e-5, -42373/5*1e-6, 176087e-8,
         -8023/5*1e-7, 5169e-9]
    omega_len = omega_length(f, len_arm)
    
    # For omega_len < 4.1
    omega_len_low_f_indices = np.where(omega_len < 4.1)
    omega_len_low_f = omega_len[omega_len_low_f_indices]
    base_low_f = base[omega_len_low_f_indices]
    low_f_modulation = np.polyval(a[::-1], omega_len_low_f)
    low_f_result = np.multiply(base_low_f, low_f_modulation)
    
    # For omega_len >= 4.1
    omega_len_high_f_indices = np.where(omega_len >= 4.1)
    omega_len_high_f = omega_len[omega_len_high_f_indices]
    base_high_f = base[omega_len_high_f_indices]
    high_f_modulation = np.exp(-0.322 * np.sin(2*omega_len_high_f-4.712) + 0.078)
    high_f_result = np.multiply(base_high_f, high_f_modulation)
    
    fp_sq_numerical = np.concatenate((low_f_result, high_f_result))
    return fp_sq_numerical

def tianqin_sensitivity_curve(flow=1e-4, fhigh=1, Npts=1000,
                            len_arm=np.sqrt(3)*1e8,
                            acc_noise_level=1e-15,
                            oms_noise_level=1e-12):
    """TianQin's analytical sensitivity curve."""
    fr = np.logspace(np.log10(flow), np.log10(fhigh), Npts)
    
    # Acceleration noise PSD
    s_acc_d = acc_noise_level**2 * (2*np.pi*fr)**(-4) * (1+1e-4/fr)
    s_acc_nu = (2*np.pi*fr/c)**2 * s_acc_d
    
    # OMS noise PSD
    s_oms_d = oms_noise_level**2
    s_oms_nu = s_oms_d * (2*np.pi*fr/c)**2
    
    # Antenna response
    fp_sq = averaged_tianqin_fplus_sq_numerical(fr, len_arm)
    omega_len = omega_length(fr, len_arm)
    
    # Total sensitivity
    sense_curve = ((s_oms_nu + s_acc_nu*(3+np.cos(2*omega_len))) /
                   (omega_len**2*fp_sq))
    
    # Convert to Omega_GW
    h2_omega = fr * sense_curve/2
    
    return fr, h2_omega

def taiji_sensitivity_curve(flow=1e-5, fhigh=1, Npts=1000,
                          len_arm=3e9,
                          acc_noise_level=3e-15,
                          oms_noise_level=8e-12):
    """Taiji's analytical sensitivity curve."""
    fr = np.logspace(np.log10(flow), np.log10(fhigh), Npts)
    
    # Acceleration noise PSD
    s_acc = acc_noise_level**2 * (1+(4e-4/fr)**2)*(1+(fr/8e-3)**4)
    s_acc_d = s_acc * (2*np.pi*fr)**(-4)
    s_acc_nu = (2*np.pi*fr/c)**2 * s_acc_d
    
    # OMS noise PSD  
    s_oms_d = oms_noise_level**2 * (1+(2e-3/fr)**4)
    s_oms_nu = s_oms_d * (2*np.pi*fr/c)**2
    
    # Antenna response
    fp_sq = averaged_fplus_sq_approximated(fr, len_arm)
    omega_len = omega_length(fr, len_arm)
    
    # Total sensitivity
    sense_curve = ((s_oms_nu + s_acc_nu*(3+np.cos(2*omega_len))) /
                   (omega_len**2*fp_sq))
    
    # Convert to Omega_GW
    h2_omega = fr * sense_curve/2
    
    return fr, h2_omega

def lisa_sensitivity_curve(flow=1e-5, fhigh=1, Npts=1000):
    """LISA's approximate analytical sensitivity curve."""
    fr = np.logspace(np.log10(flow), np.log10(fhigh), Npts)
    L_lisa = 2.5e9  # meters
    f_star = c / (2 * np.pi * L_lisa)
    
    # Simplified LISA sensitivity
    h2_omega = 2e-13 * (fr/1e-3)**2 * (1 + (fr/(5*f_star))**2)
    
    # Add low-frequency behavior
    low_f_mask = fr < 1e-4
    h2_omega[low_f_mask] *= (1e-4/fr[low_f_mask])**2
    
    return fr, h2_omega


# ============= CONVERSION FUNCTIONS =============

def k_to_frequency(k_Mpc):
    """
    Convert wavenumber k [Mpc^-1] to frequency f [Hz].
    k = 2πf/c, where c needs proper unit conversion.
    """
    # Conversion factor: c [Mpc/s] = 2.998e5 km/s / 3.086e19 km/Mpc = 9.72e-15 Mpc/s
    c_Mpc_per_s = 9.72e-15
    f_Hz = k_Mpc * c_Mpc_per_s / (2 * np.pi)
    return f_Hz

def frequency_to_k(f_Hz):
    """
    Convert frequency f [Hz] to wavenumber k [Mpc^-1].
    """
    c_Mpc_per_s = 9.72e-15
    k_Mpc = 2 * np.pi * f_Hz / c_Mpc_per_s
    return k_Mpc


# ============= NEW PLOTTING FUNCTIONS =============

def plot_tensor_power_spectrum_only(calc, k_tilde_values=None, P_h_values=None):
    """
    Plot only the dimensionless tensor power spectrum 𝒫_h(k̃).
    """
    if k_tilde_values is None or P_h_values is None:
        print("\nComputing Tensor Power Spectrum...")
        k_tilde_values = np.logspace(-2, 2, 30)
        P_h_values = []
        
        for i, k_tilde in enumerate(k_tilde_values):
            print(f"Progress: {i+1}/{len(k_tilde_values)} - k̃ = {k_tilde:.2e}")
            P_h = calc.compute_dimensionless_power_spectrum(k_tilde, z=1000, nq=50)
            P_h_values.append(P_h)
            print(f"  𝒫_h = {P_h:.2e}")
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Primary axis: k̃ = k/k_eq
    ax1.loglog(k_tilde_values, P_h_values, 'b-', linewidth=2.5, 
               label='Analytical (Radiation Era)')
    ax1.set_xlabel(r'$\tilde{k} = k/k_{\mathrm{eq}}$', fontsize=16)
    ax1.set_ylabel(r'$\mathcal{P}_h(\tilde{k})$', fontsize=16)
    ax1.set_title(r'\textbf{Dimensionless Tensor Power Spectrum}', fontsize=18)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5, label=r'$k = k_{\mathrm{eq}}$')
    
    # Secondary x-axis: frequency in Hz
    ax2 = ax1.secondary_xaxis('top')
    ax2.set_xlabel(r'\textbf{Frequency [Hz]}', fontsize=16)
    
    # Convert k̃ to frequency for the secondary axis
    k_physical = k_tilde_values * calc.k_eq_Mpc  # Convert to physical k
    f_values = k_to_frequency(k_physical)
    
    # Set up the secondary axis with proper scaling
    def k_tilde_to_f(k_tilde):
        return k_to_frequency(k_tilde * calc.k_eq_Mpc)
    
    def f_to_k_tilde(f):
        return frequency_to_k(f) / calc.k_eq_Mpc
    
    ax2.set_xscale('log')
    ax2.set_xlim(k_tilde_to_f(k_tilde_values[0]), k_tilde_to_f(k_tilde_values[-1]))
    
    # Add frequency band labels
    # ax1.text(0.1, 1e-10, 'PTA band\n(nHz)', fontsize=10, alpha=0.7, ha='center')
    # ax1.text(10, 1e-10, 'LISA band\n(mHz)', fontsize=10, alpha=0.7, ha='center')
    # ax1.text(1000, 1e-10, 'LIGO band\n(Hz-kHz)', fontsize=10, alpha=0.7, ha='center')
    
    ax1.legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    
    # Save
    plt.savefig('SIGW_tensor_power_spectrum.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('SIGW_tensor_power_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return k_tilde_values, P_h_values

def plot_omega_gw_with_sensitivities(calc, k_tilde_values=None, omega_gw_values=None):
    """
    Plot Ω_GW with detector sensitivity curves.
    """
    if k_tilde_values is None or omega_gw_values is None:
        print("\nComputing Omega_GW...")
        k_tilde_values = np.logspace(-2, 3, 100)  # Extended range for full frequency coverage
        omega_gw_values = []
        
        for i, k_tilde in enumerate(k_tilde_values):
            if i % 10 == 0:
                print(f"Progress: {i+1}/{len(k_tilde_values)} - k̃ = {k_tilde:.2e}")
            omega_gw = calc.compute_omega_gw(k_tilde, z=0, nq=30)
            omega_gw_values.append(omega_gw)
    
    # Convert to arrays
    omega_gw_values = np.array(omega_gw_values)
    h2_omega_gw = calc.params.h**2 * omega_gw_values
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(14, 9))
    
    # Convert k̃ to frequency for plotting
    k_physical = k_tilde_values * calc.k_eq_Mpc
    f_values = k_to_frequency(k_physical)
    
    # Plot SIGW spectrum
    ax1.loglog(f_values, h2_omega_gw, 'b-', linewidth=3, 
               label=r'SIGW: Scale-invariant $A_\zeta = 2.1 \times 10^{-9}$')
    
    # Add analytical sensitivity curves using embedded functions
    
    # TianQin
    f_tq, h2_omega_tq = tianqin_sensitivity_curve()
    ax1.loglog(f_tq, h2_omega_tq, 'r--', linewidth=2, alpha=0.7, label='TianQin')
    
    # Taiji
    f_tj, h2_omega_tj = taiji_sensitivity_curve()
    ax1.loglog(f_tj, h2_omega_tj, 'g--', linewidth=2, alpha=0.7, label='Taiji')
    
    # LISA
    f_lisa, h2_omega_lisa = lisa_sensitivity_curve()
    ax1.loglog(f_lisa, h2_omega_lisa, 'm--', linewidth=2, alpha=0.7, label='LISA')
    
    # PTA sensitivity band (NANOGrav-like)
    f_pta = np.logspace(-9, -7, 100)
    h2_omega_pta = 2e-15 * (f_pta/1e-8)**(2/3)
    ax1.fill_between(f_pta, h2_omega_pta, 1e-5, alpha=0.2, color='orange', label='PTA (NANOGrav)')
    
    # LIGO/Virgo band
    f_ligo = np.logspace(0.5, 3, 200)
    h2_omega_ligo = 5e-9 * (10/f_ligo)**4 * (f_ligo < 100) + 5e-9 * (f_ligo >= 100)
    ax1.fill_between(f_ligo, h2_omega_ligo, 1e-5, alpha=0.2, color='cyan', label='LIGO/Virgo')
    
    # Primary axis: frequency in Hz
    ax1.set_xlabel(r'\textbf{Frequency [Hz]}', fontsize=16)
    ax1.set_ylabel(r'$h^2 \Omega_{\mathrm{GW}}$', fontsize=16)
    ax1.set_xlim(1e-9, 1e3)
    ax1.set_ylim(1e-20, 1e-5)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_title(r'\textbf{Gravitational Wave Energy Density Spectrum}', fontsize=18)
    
    # Secondary x-axis: k in Mpc^-1
    ax2 = ax1.secondary_xaxis('top')
    ax2.set_xlabel(r'\textbf{Wavenumber $k$ [Mpc$^{-1}$]}', fontsize=16)
    ax2.set_xscale('log')
    
    # Set up the transform functions
    ax2.set_xlim(frequency_to_k(1e-9), frequency_to_k(1e3))
    
    # Add frequency labels
    # ax1.text(1e-8, 1e-6, 'nHz', fontsize=12, alpha=0.7)
    # ax1.text(1e-6, 1e-6, r'$\mu$Hz', fontsize=12, alpha=0.7)
    # ax1.text(1e-3, 1e-6, 'mHz', fontsize=12, alpha=0.7)
    # ax1.text(1, 1e-6, 'Hz', fontsize=12, alpha=0.7)
    # ax1.text(1e3, 1e-6, 'kHz', fontsize=12, alpha=0.7)
    
    # Add k_eq marker
    f_eq = k_to_frequency(calc.k_eq_Mpc)
    ax1.axvline(x=f_eq, color='gray', linestyle='--', alpha=0.5)
    ax1.text(f_eq*1.5, 1e-18, r'$k_{\mathrm{eq}}$', fontsize=12, alpha=0.7)
    
    ax1.legend(fontsize=11, loc='lower left', ncol=2)
    plt.tight_layout()
    
    # Save
    plt.savefig('SIGW_omega_gw_with_sensitivities.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('SIGW_omega_gw_with_sensitivities.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return f_values, h2_omega_gw

def plot_both_spectra(calc):
    """
    Compute and plot both tensor power spectrum and Omega_GW in separate figures.
    """
    print("\n" + "="*60)
    print("Computing SIGW Spectra")
    print("="*60)
    
    # Compute spectra
    k_tilde_values = np.logspace(-2, 3, 50)
    P_h_values = []
    omega_gw_values = []
    
    for i, k_tilde in enumerate(k_tilde_values):
        if i % 5 == 0:
            print(f"Progress: {i+1}/{len(k_tilde_values)} - k̃ = {k_tilde:.2e}")
        
        P_h = calc.compute_dimensionless_power_spectrum(k_tilde, z=0.1, nq=40)
        omega_gw = calc.compute_omega_gw(k_tilde, z=0, nq=40)
        
        P_h_values.append(P_h)
        omega_gw_values.append(omega_gw)
    
    # Plot tensor power spectrum
    print("\nPlotting tensor power spectrum...")
    plot_tensor_power_spectrum_only(calc, k_tilde_values, P_h_values)
    
    # Plot Omega_GW with sensitivities
    print("\nPlotting Omega_GW with sensitivity curves...")
    plot_omega_gw_with_sensitivities(calc, k_tilde_values, omega_gw_values)
    
    return k_tilde_values, P_h_values, omega_gw_values


# ============= MAIN EXECUTION BLOCK =============

if __name__ == "__main__":
    # Run verification
    verify_source_integral()
    
    # Initialize calculator
    calc = AnalyticalSIGWCalculator()
    
    # Compute ONLY the tensor power spectrum
    print("\n" + "="*60)
    print("Computing Tensor Power Spectrum Only")
    print("="*60)
    
    # Compute spectrum
    k_tilde_values = np.logspace(-2, 2, 30)  # k/k_eq from 0.01 to 100
    P_h_values = []
    
    for i, k_tilde in enumerate(k_tilde_values):
        print(f"Progress: {i+1}/{len(k_tilde_values)} - k̃ = {k_tilde:.2e}")
        P_h = calc.compute_dimensionless_power_spectrum(k_tilde, z=1000, nq=50)
        P_h_values.append(P_h)
        print(f"  𝒫_h = {P_h:.2e}")
    
    # Plot ONLY tensor power spectrum
    print("\nPlotting tensor power spectrum...")
    plot_tensor_power_spectrum_only(calc, k_tilde_values, P_h_values)
    
    # Print summary for power spectrum only
    print("\n" + "="*60)
    print("Summary Statistics - Power Spectrum")
    print("="*60)
    print(f"Max 𝒫_h: {max(P_h_values):.2e}")
    print(f"Min 𝒫_h: {min(P_h_values):.2e}")
    print(f"Peak location: k̃ ≈ {k_tilde_values[np.argmax(P_h_values)]:.2f}")
    
    # Convert peak to frequency
    k_peak_physical = k_tilde_values[np.argmax(P_h_values)] * calc.k_eq_Mpc
    f_peak = k_to_frequency(k_peak_physical)
    print(f"Peak frequency: f ≈ {f_peak:.2e} Hz")
    print("="*60)

