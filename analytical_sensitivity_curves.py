import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
import matplotlib as mpl  # type: ignore
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

# -----------------------------------------------------------------------------
# Shared constants & helpers
# -----------------------------------------------------------------------------
H0_KMS_MPC = 67.4              # Hubble constant (Planck 2018 best fit) [km/s/Mpc]
KM_PER_MPC = 3.085677581e19    # exact definition
H0 = H0_KMS_MPC / KM_PER_MPC   # H0 in 1/s

_CONST = 2 * np.pi ** 2 / (3 * H0 ** 2)  # factor for Sn -> Omega_GW


def _omega_from_Sn(f: np.ndarray, Sn: np.ndarray) -> np.ndarray:
    """Convert single-sided strain PSD Sn(f) → Ω_GW(f)."""
    return _CONST * f ** 3 * Sn

# -----------------------------------------------------------------------------
#  Generic PSD building blocks already present for Taiji / TianQin – import
# -----------------------------------------------------------------------------
from gw_sensitivity_curves import (
    _psd_acc_noise as _psd_acc_tj,  # shared helper for LISA/Taiji like missions
    _psd_oms_noise as _psd_oms_tj,
    psd_tianqin_acc_noise,
    psd_tianqin_oms_noise,
    psd_taiji_acc_noise,
    psd_taiji_oms_noise,
    omega_length,
    averaged_fplus_sq_approximated,
    averaged_tianqin_fplus_sq_numerical,
)

# -----------------------------------------------------------------------------
#  Space-borne interferometers
# -----------------------------------------------------------------------------

def sensitivity_curve_lisa_analytical(flow: float,
                                      fhigh: float,
                                      n_pts: int,
                                      len_arm: float = 2.5e9,  # 2.5 Gm
                                      acc_noise_lvl: float = 3e-15,
                                      oms_noise_lvl: float = 15e-12) -> tuple[np.ndarray, np.ndarray]:
    """Analytical LISA sensitivity (6-link Michelson),
    averaged over sky & polarization, following the formulation used in
    PyCBCʼs *analytical_space* module (similar to Robson, Cornish & Liu 2019).
    This is the same functional form we use for Taiji, just with LISA-specific
    arm length and noise levels.
    """
    len_arm   = np.float64(len_arm)
    acc_noise_lvl = np.float64(acc_noise_lvl)
    oms_noise_lvl = np.float64(oms_noise_lvl)

    f = np.logspace(np.log10(flow), np.log10(fhigh), n_pts)

    # Single-link noise PSDs (converted to frequency noise units)
    s_acc_nu = _psd_acc_tj(f, acc_noise_lvl)
    s_oms_nu = _psd_oms_tj(f, oms_noise_lvl)

    # Detector response
    fp_sq     = averaged_fplus_sq_approximated(f, len_arm)
    omega_len = omega_length(f, len_arm)

    Sn = (s_oms_nu + s_acc_nu * (3 + np.cos(2 * omega_len))) / (omega_len ** 2 * fp_sq)
    Sn = Sn / 2  # convert to single-sided strain PSD
    return f, _omega_from_Sn(f, Sn)


def sensitivity_curve_decigo_analytical(flow: float,
                                        fhigh: float,
                                        n_pts: int,
                                        len_arm: float = 1.0e6,  # 1000 km
                                        acc_noise_lvl: float = 2.0e-15,
                                        shot_noise_lvl: float = 8.0e-20) -> tuple[np.ndarray, np.ndarray]:
    """Rough analytical DECIGO sensitivity (after Yagi & Seto 2011 fit)."""
    f = np.logspace(np.log10(flow), np.log10(fhigh), n_pts)
    Sn = ( (5e-52)*( (f/7.36)**-4.14 * (1 + (f/7.36)**2 ) )
          + (2.3e-54) )  # fit to the official design curve (strain^2/Hz)

    Omega = _omega_from_Sn(f, Sn)
    return f, Omega


def sensitivity_curve_et_analytical(flow: float,
                                    fhigh: float,
                                    n_pts: int) -> tuple[np.ndarray, np.ndarray]:
    """Very rough ET-D fit following Hild et al. 2011 (strain) converted to Ω."""
    f = np.logspace(np.log10(flow), np.log10(fhigh), n_pts)
    # dimensionless frequency x = f / f0 with f0 = 100 Hz
    x = f / 100.
    Sn = 1e-50 * (2.39e-27 * x**-15.64 + 0.349 * x**-2.145 + 1.77 * x**-0.69 + 0.409 * x**2.145)
    Omega = _omega_from_Sn(f, Sn)
    return f, Omega

# -----------------------------------------------------------------------------
#  Wrapper
# -----------------------------------------------------------------------------
_detector_map = {
    'lisa': sensitivity_curve_lisa_analytical,
    'taiji': None,      # will be patched below
    'tianqin': None,    # will be patched below
    'decigo': sensitivity_curve_decigo_analytical,
    'et': sensitivity_curve_et_analytical,
}

# we import the existing analytical helpers for Taiji/TianQin from the original module
from gw_sensitivity_curves import (
    sensitivity_curve_taiji_analytical as _taiji_raw,
    sensitivity_curve_tianqin_analytical as _tianqin_raw,
)

# Convert raw strain-PSD helpers to Ω_GW on the fly
_taiji = lambda fl, fh, n, **kw: (lambda fr, Sn: (fr, _omega_from_Sn(fr, Sn)))(*_taiji_raw(fl, fh, n, **kw))
_tianqin = lambda fl, fh, n, **kw: (lambda fr, Sn: (fr, _omega_from_Sn(fr, Sn)))(*_tianqin_raw(fl, fh, n, **kw))

_detector_map['taiji'] = _taiji
_detector_map['tianqin'] = _tianqin


def get_OmegaGW_curve(detector: str,
                       f_low: float = 1e-5,
                       f_high: float = 1e1,
                       n_pts: int = 1000,
                       **kwargs):
    """Unified public API. Returns (f, Ω_GW) arrays for *detector*.

    Parameters
    ----------
    detector : str
        Key in {'lisa','taiji','tianqin','decigo','et'}  (case-insensitive).
    f_low, f_high : float
        Frequency range in Hz.
    n_pts : int
        Number of log-spaced samples.
    **kwargs :
        Forwarded to the underlying detector helper for easy parameter scans.
    """
    func = _detector_map.get(detector.lower())
    if func is None:
        raise ValueError(f"Unknown detector '{detector}'. Available: {list(_detector_map)}")
    return func(f_low, f_high, n_pts, **kwargs)

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Publication-style plot: only LISA / Taiji / TianQin – same look as
    # GW_sensitivity.py 
    # ------------------------------------------------------------------
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\boldmath
"""

    mpl.rcParams.update({
        'figure.figsize': (10, 7),
        'font.size': 11,
        'axes.labelsize': 11,
        'axes.titlesize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'axes.linewidth': 1,
        'xtick.major.size': 3,
        'xtick.minor.size': 3,
        'xtick.major.width': 1,
        'xtick.minor.width': 1,
        'ytick.major.size': 3,
        'ytick.minor.size': 3,
        'ytick.major.width': 1,
        'ytick.minor.width': 1,
    })

    # --- Data preparation -------------------------------------------------
    h = H0_KMS_MPC / 100.0  # dimensionless little-h

    cfg = [
    ("lisa",   '#5bb7c5',  r"\textbf{LISA}"),
    ("taiji",  '#bb33d7',  r"\textbf{Taiji}"),
    ("tianqin", '#f28c28', r"\textbf{TianQin}"),
]

    # --- Plot -------------------------------------------------------------
    fig, ax = plt.subplots()

    ymax = 1e-4
    ymin = 1e-15
    for det, col, label in cfg:
        f, omega = get_OmegaGW_curve(det)
        y = h**2 * omega
        ax.plot(f, y, color=col, linewidth=2)
        ax.fill_between(f, y, ymax, color=col, alpha=0.26)
        # add centred text label near minimum
        # idx = np.argmin(y)
        # ax.text(f[idx], y[idx]*1.4, label, color=col, fontsize=16,
        #         ha='center', va='center', bbox={'facecolor':'white', 'alpha':0.12, 'pad':4})


## Annotations for LISA, Taiji, TianQin
    ax.text(6.88e-5, 1.047e-10, r"\textbf{LISA}", color='#5bb7c5', fontsize=16,
            ha='center', va='center', bbox={'facecolor':'white', 'alpha':0.12, 'pad':4})
    ax.text(0.0034, 5.13e-13, r"\textbf{Taiji}", color='#bb33d7', fontsize=16,
            ha='center', va='center', bbox={'facecolor':'white', 'alpha':0.12, 'pad':4})
    ax.text(0.1164, 2.69e-9, r"\textbf{TianQin}", color='#f28c28', fontsize=16,
            ha='center', va='center', bbox={'facecolor':'white', 'alpha':0.12, 'pad':4})
    ax.text(0.277, 1.54e-14, r"\textbf{Analytical fitting}", color='brown', fontsize=16,
            ha='center', va='center', bbox={'facecolor':'white', 'alpha':0.12, 'pad':4})

    # Axis formatting
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(8e-6, 1)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(r'\textbf{Frequency (Hz)}', fontsize=20)
    ax.set_ylabel(r'$h^{2}\,\Omega_{\mathrm{GW}}$', fontsize=20)
    # ax.grid(True, which='both', ls=':', alpha=0.3)
    plt.grid()

    # --- secondary k-axis --------------------------------------------------
    def freq_to_k(f):
        return 2 * np.pi * f / (9.72e-15)  # 1 Hz -> k [Mpc^{-1}]
    def k_to_freq(k):
        return k * 9.72e-15 / (2 * np.pi)
    secax = ax.secondary_xaxis('top', functions=(freq_to_k, k_to_freq))
    secax.set_xscale('log')
    secax.set_xlabel(r"\textbf{Wavenumber $k$ (Mpc$^{-1}$)}", fontsize=18)

    fig.tight_layout()
    fig.savefig('Analytical_OmegaGW_LISA_Taiji_TianQin.pdf', bbox_inches='tight', dpi=300)
    fig.savefig('Analytical_OmegaGW_LISA_Taiji_TianQin.png', bbox_inches='tight', dpi=300)
    plt.show()  
