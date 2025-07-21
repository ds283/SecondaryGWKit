import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn, yn
from scipy.integrate import quad, dblquad
from dataclasses import dataclass
from typing import Tuple, Callable
from matplotlib import rcParams
from matplotlib import rc
from matplotlib.patches import FancyArrowPatch

# activate latex text rendering
plt.rc('text', usetex=True)

#LaTex setting
plt.rcParams['text.latex.preamble']=r"\usepackage{amsmath}"
plt.rcParams['text.latex.preamble'] = r'\boldmath'


#Plot setting:
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['font.size'] = 11
plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'
plt.rcParams['text.latex.preamble'] = (
    r'\usepackage{amsmath}\usepackage{amssymb}'
    r'\usepackage{xcolor}\boldmath'
)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{xcolor}\usepackage{amssymb}\boldmath'

#plt.rcParams['font.family'] = 'Times New Roman'

plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.4*plt.rcParams['font.size']
#plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = 1.4*plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = 1.4*plt.rcParams['font.size']
# dots per inch: dpi
#plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']

plt.rcParams['xtick.major.size'] = 3
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['xtick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['ytick.minor.width'] = 1

#legends
#plt.rcParams['legend.frameon'] = False
#plt.rcParams['legend.loc'] = 'center left'
#plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']

plt.rcParams['axes.linewidth'] = 1

#border setting
#plt.gca().spines['right'].set_color('none')
#plt.gca().spines['top'].set_color('none')

#ticks position setting
# plt.gca().xaxis.set_ticks_position('bottom')
# plt.gca().yaxis.set_ticks_position('left')
# fig, ax = plt.subplots()
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
#f = plt.figure()
#ax = f.add_subplot(111)
#ax.tick_params(labeltop=False, labelright=True)
#If we don't want to use x-axis and y-axis values.
#plt.gca().axes.xaxis.set_ticks([])
#plt.gca().axes.yaxis.set_ticks([]) 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


@dataclass
class CosmologicalParameters:
    """Standard Planck 2018 cosmological parameters"""
    Omega_r: float = 9.1e-5      # Radiation density parameter
    Omega_m: float = 0.315       # Matter density parameter  
    Omega_Lambda: float = 0.6847 # Dark energy density parameter (1 - Omega_r - Omega_m)
    h: float = 0.674             # Reduced Hubble parameter
    z_eq: float = 3387           # Matter-radiation equality redshift
    A_zeta: float = 2.1e-9       # Primordial scalar amplitude at CMB scales
    k_star: float = 0.05         # Pivot scale in Mpc^-1
    n_s: float = 1.0             # Spectral index (1.0 for scale invariant)
    w_star: float = 1/3          # Equation of state during radiation domination
    cs: float = 1/np.sqrt(3)     # Speed of sound in radiation

class AnalyticalGWCalculator:
    """
    Calculator for analytical tensor power spectrum from scalar-induced gravitational waves.
    Based on the formulation provided, with conventions from Domènech and Kohri-Terada.
    """
    
    def __init__(self, cosmo_params: CosmologicalParameters = None):
        self.params = cosmo_params or CosmologicalParameters()
        
        # Derived quantities
        self.H0 = 100 * self.params.h  # Hubble constant in km/s/Mpc
        self.a0 = 1.0  # Present scale factor
        self.k_eq = self.a0 * self.H0 / 46.71  # Equality scale in Mpc^-1
        
    def f(self, z: float) -> float:
        """Dimensionless Hubble function H(z)/H0"""
        return np.sqrt(
            self.params.Omega_r * (1 + z)**4 + 
            self.params.Omega_m * (1 + z)**3 + 
            self.params.Omega_Lambda
        )
    
    def tau_integral(self, z: float) -> float:
        """Conformal time integral ∫_{∞}^{z} -dz'/f(z')"""
        def integrand(zp):
            return -1.0 / self.f(zp)
        
        # For numerical stability, integrate from large z (e.g., 1e6) instead of infinity
        result, _ = quad(integrand, 1e6, z)
        return result
    
    def scalar_power_spectrum(self, k: float) -> float:
        """
        Scale-invariant primordial scalar power spectrum.
        P_zeta(k) = A_zeta * (k/k_*)^(n_s-1)
        """
        return self.params.A_zeta * (k / self.params.k_star)**(self.params.n_s - 1)
    
    def source_time_integral(self, k: float, q: float, r: float, z: float) -> float:
        """
        Compute the source time integral I(k,q,r,z).
        Uses the analytical expression with Bessel functions J_{1/2} and Y_{1/2}.
        """
        q = q * self.params.cs
        r = r * self.params.cs
        # Compute conformal time
        tau = self.tau_integral(z)
        
        # Triangle condition check
        if abs(q - r) > k or k > q + r:
            return 0.0
        
        # Compute the momentum factor
        momentum_factor = (-k**2 + q**2 + r**2)**2 / (q**3 * r**3 * np.sqrt(k))
        
        # Compute the logarithmic term
        numerator = (k + q + r) * (-k + q + r)
        denominator = (k + q - r) * (k - q + r)
        
        # Handle potential division by zero
        if abs(denominator) < 1e-15:
            return 0.0
        
        log_term = np.log(abs(numerator / denominator))
        
        # Time factor
        time_factor = 1.0 / np.sqrt(tau)
        
        # Bessel function arguments
        x = k * tau
        
        # Bessel functions of order 1/2
        J_half = np.sqrt(2/(np.pi * x)) * np.sin(x)  # J_{1/2}(x) = sqrt(2/(πx)) * sin(x)
        Y_half = -np.sqrt(2/(np.pi * x)) * np.cos(x)  # Y_{1/2}(x) = -sqrt(2/(πx)) * cos(x)
        
        # Complete expression
        prefactor = (1/8) * np.sqrt(3 * np.pi**3 / 2)
        bracket_term = Y_half + (J_half / np.pi) * log_term
        
        result = prefactor * momentum_factor * time_factor * bracket_term
        
        return result
    
    def dimensionless_source_integral(self, k_tilde: float, q_tilde: float, 
                                     r_tilde: float, z: float) -> float:
        """
        Dimensionless version of the source time integral.
        I_tilde(q*k_eq, r*k_eq, z) * (a0*H0)^2
        """
        # Convert dimensionless variables to physical
        k = k_tilde * self.k_eq
        q = q_tilde * self.k_eq
        r = r_tilde * self.k_eq
        q = q * self.params.cs
        r = r * self.params.cs
        
        # Factors for dimensionless conversion
        prefactor = np.sqrt(self.k_eq / (self.a0 * self.H0))
        z_factor = (1 + self.params.z_eq) / self.f(self.params.z_eq)
        
        # Modified conformal time for dimensionless variables
        tau_z = self.tau_integral(z)
        tau_factor = 1.0 / np.sqrt(abs(tau_z))
        
        # Triangle condition
        if abs(q_tilde - r_tilde) > k_tilde or k_tilde > q_tilde + r_tilde:
            return 0.0
        
        # Momentum factor
        momentum_factor = (-k_tilde**2 + q_tilde**2 + r_tilde**2)**2 / \
                         (q_tilde**3 * r_tilde**3 * np.sqrt(k_tilde))
        
        # Log term
        numerator = (k_tilde + q_tilde + r_tilde) * (-k_tilde + q_tilde + r_tilde)
        denominator = (k_tilde + q_tilde - r_tilde) * (k_tilde - q_tilde + r_tilde)
        
        if abs(denominator) < 1e-15:
            return 0.0
        
        log_term = np.log(abs(numerator / denominator))
        
        # Bessel function argument
        x = k_tilde * self.f(self.params.z_eq) / (1 + self.params.z_eq) * abs(tau_z)
        
        # Bessel functions
        if x > 0:
            J_half = np.sqrt(2/(np.pi * x)) * np.sin(x)
            Y_half = -np.sqrt(2/(np.pi * x)) * np.cos(x)
        else:
            return 0.0
        
        # Complete expression
        base_prefactor = (1/8) * np.sqrt(3 * np.pi**3 / 2)
        full_prefactor = base_prefactor * prefactor * z_factor
        bracket_term = Y_half + (J_half / np.pi) * log_term
        
        result = full_prefactor * momentum_factor * tau_factor * bracket_term
        
        return result
    
    def theta_integrand(self, theta: float, k_tilde: float, q_tilde: float, 
                       z: float = 0.1) -> float:
        """
        Integrand for the angular (theta) integration.
        Includes sin^5(theta) factor and handles the transformation r = |k - q|.
        """
        # Compute r_tilde using law of cosines
        r_tilde_sq = k_tilde**2 + q_tilde**2 - 2 * k_tilde * q_tilde * np.cos(theta)
        
        if r_tilde_sq < 0:
            return 0.0
        
        r_tilde = np.sqrt(r_tilde_sq)
        
        if r_tilde < 1e-10:  # Avoid division by zero
            return 0.0
        
        # Compute source integral
        I_val = self.dimensionless_source_integral(k_tilde, q_tilde, r_tilde, z)
        
        # Power spectrum at r
        P_r = self.scalar_power_spectrum(r_tilde * self.k_eq)
        
        # Full integrand
        integrand = P_r / r_tilde**3 * np.sin(theta)**5 * I_val**2
        
        return integrand
    
    def compute_tensor_power_spectrum(self, k_tilde: float, z: float = 0.1,
                                     q_min: float = 0.1, q_max: float = 100.0,
                                     nq: int = 50) -> float:
        """
        Compute the dimensionless tensor power spectrum P_h(k_tilde).
        
        Parameters:
        -----------
        k_tilde : float
            Dimensionless wavenumber k/k_eq
        z : float
            Redshift at which to evaluate (default: 0.1)
        q_min, q_max : float
            Integration limits for q_tilde
        nq : int
            Number of q points for integration
        
        Returns:
        --------
        P_h : float
            Dimensionless tensor power spectrum
        """
        # Prefactor
        w = self.params.w_star
        prefactor = 1.5423e9 * k_tilde**3 * ((1 + 3*w) / (5 + 3*w))**4
        
        # Logarithmic q grid for better sampling
        q_grid = np.logspace(np.log10(q_min), np.log10(q_max), nq)
        
        integral_sum = 0.0
        
        for i in range(len(q_grid) - 1):
            q = (q_grid[i] + q_grid[i+1]) / 2  # Midpoint
            dq = q_grid[i+1] - q_grid[i]
            
            # Compute theta integral
            theta_integral, _ = quad(
                lambda theta: self.theta_integrand(theta, k_tilde, q, z),
                0, np.pi, limit=100
            )
            
            # Add to sum
            P_q = self.scalar_power_spectrum(q * self.k_eq)
            integral_sum += q**3 * P_q * theta_integral * dq
        
        return prefactor * integral_sum
    
    def compute_omega_gw(self, k_tilde: float, z: float = 0.1) -> float:
        """
        Compute the gravitational wave energy density parameter Omega_GW.
        For both polarizations: Omega_GW = (1/12) * (k/H)^2 * P_h(k)
        """
        P_h = self.compute_tensor_power_spectrum(k_tilde, z)
        
        # Convert to physical frequency
        k_phys = k_tilde * self.k_eq
        H = self.H0 * self.f(z)
        
        omega_gw = (1/12) * (k_phys / H)**2 * P_h
        
        return omega_gw


# Plotting
if __name__ == "__main__":
    # Initialize calculator
    calc = AnalyticalGWCalculator()
    
    # Compute tensor power spectrum for a range of k values
    k_tilde_values = np.logspace(-2, 2, 50)  # k/k_eq from 0.01 to 100
    P_h_values = []
    omega_gw_values = []
    
    print("Computing tensor power spectrum...")
    print("k_tilde\t\tP_h(k)\t\tOmega_GW(k)")
    print("-" * 50)
    
    for k_tilde in k_tilde_values:
        P_h = calc.compute_tensor_power_spectrum(k_tilde, z=0.1, nq=30)
        omega_gw = calc.compute_omega_gw(k_tilde, z=0.1)
        
        P_h_values.append(P_h)
        omega_gw_values.append(omega_gw)
        
        print(f"{k_tilde:.4f}\t\t{P_h:.4e}\t{omega_gw:.4e}")
    
    # Plotting
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Tensor power spectrum
    plt.loglog(k_tilde_values, P_h_values, 'b-', linewidth=2)
    plt.xlabel(r'$\tilde{k} = k/k_{eq}$')
    plt.ylabel(r'$\mathcal{P}_h(\tilde{k})$')
    plt.title('Dimensionless Tensor Power Spectrum')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Analytical_Tensor_Power.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('Analytical_Tensor_Power.png', bbox_inches='tight', dpi=300)
    plt.show()
    

    
    # Omega_GW
    plt.loglog(k_tilde_values, omega_gw_values, 'r-', linewidth=2)
    plt.xlabel(r'$\tilde{k} = k/k_{eq}$')
    plt.ylabel(r'$\Omega_{GW}(k)$')
    plt.title('Gravitational Wave Energy Density')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Analytical_Omega_GW.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('Analytical_Omega_GW.png', bbox_inches='tight', dpi=300)
    plt.show()
    
    # Additional diagnostic: Check source integral at specific points
    print("\n\nSource integral diagnostic:")
    k_test, q_test, r_test = 1.0, 0.5, 0.8
    z_test = 0.1
    I_test = calc.dimensionless_source_integral(k_test, q_test, r_test, z_test)
    print(f"I({k_test}, {q_test}, {r_test}, z={z_test}) = {I_test}")