import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c

# activate latex text rendering
plt.rc("text", usetex=True)

# LaTex setting
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["text.latex.preamble"] = r"\boldmath"


# Plot setting:
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.size"] = 11
plt.rcParams["text.latex.preamble"] = r"\usepackage{amssymb}"
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}\usepackage{color}\usepackage{amssymb}\boldmath"
)
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}\usepackage{xcolor}\usepackage{amssymb}\boldmath"
)

# plt.rcParams['font.family'] = 'Times New Roman'

plt.rcParams["axes.labelsize"] = plt.rcParams["font.size"]
plt.rcParams["axes.titlesize"] = 1.4 * plt.rcParams["font.size"]
# plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams["xtick.labelsize"] = 1.4 * plt.rcParams["font.size"]
plt.rcParams["ytick.labelsize"] = 1.4 * plt.rcParams["font.size"]
# dots per inch: dpi
# plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']

plt.rcParams["xtick.major.size"] = 3
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.major.size"] = 3
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["ytick.minor.width"] = 1

# legends
# plt.rcParams['legend.frameon'] = False
# plt.rcParams['legend.loc'] = 'center left'
# plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']

plt.rcParams["axes.linewidth"] = 1

# border setting
# plt.gca().spines['right'].set_color('none')
# plt.gca().spines['top'].set_color('none')

# ticks position setting
# plt.gca().xaxis.set_ticks_position('bottom')
# plt.gca().yaxis.set_ticks_position('left')
# f = plt.figure()
# ax = f.add_subplot(111)
# ax.tick_params(labeltop=False, labelright=True)
# If we don't want to use x-axis and y-axis values.
# plt.gca().axes.xaxis.set_ticks([])
# plt.gca().axes.yaxis.set_ticks([])
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# ==============================================================================
# SENSITIVITY CURVE GENERATION FUNCTIONS
# (Adapted from pycbc.psd.analytical_space)
# ==============================================================================


def _psd_acc_noise(f: np.ndarray, acc_noise_level: float) -> np.ndarray:
    s_acc = acc_noise_level**2 * (1 + (4e-4 / f) ** 2) * (1 + (f / 8e-3) ** 4)
    s_acc_d = s_acc * (2 * np.pi * f) ** (-4)
    s_acc_nu = (2 * np.pi * f / c) ** 2 * s_acc_d
    return s_acc_nu


def psd_tianqin_acc_noise(f: np.ndarray, acc_noise_level: float = 1e-15) -> np.ndarray:
    s_acc_d = acc_noise_level**2 * (2 * np.pi * f) ** (-4) * (1 + 1e-4 / f)
    s_acc_nu = (2 * np.pi * f / c) ** 2 * s_acc_d
    return s_acc_nu


def psd_taiji_acc_noise(f: np.ndarray, acc_noise_level: float = 3e-15) -> np.ndarray:
    s_acc_nu = _psd_acc_noise(f, acc_noise_level)
    return s_acc_nu


def _psd_oms_noise(f: np.ndarray, oms_noise_level: float) -> np.ndarray:
    s_oms_d = oms_noise_level**2 * (1 + (2e-3 / f) ** 4)
    s_oms_nu = s_oms_d * (2 * np.pi * f / c) ** 2
    return s_oms_nu


def psd_tianqin_oms_noise(f: np.ndarray, oms_noise_level: float = 1e-12) -> np.ndarray:
    s_oms_d = oms_noise_level**2
    s_oms_nu = s_oms_d * (2 * np.pi * f / c) ** 2
    return s_oms_nu


def psd_taiji_oms_noise(f: np.ndarray, oms_noise_level: float = 8e-12) -> np.ndarray:
    s_oms_nu = _psd_oms_noise(f, oms_noise_level)
    return s_oms_nu


def tianqin_psd_components(
    f: np.ndarray, acc_noise_level: float = 1e-15, oms_noise_level: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    low_freq_component = psd_tianqin_acc_noise(f, acc_noise_level)
    high_freq_component = psd_tianqin_oms_noise(f, oms_noise_level)
    return low_freq_component, high_freq_component


def taiji_psd_components(
    f: np.ndarray, acc_noise_level: float = 3e-15, oms_noise_level: float = 8e-12
) -> tuple[np.ndarray, np.ndarray]:
    low_freq_component = psd_taiji_acc_noise(f, acc_noise_level)
    high_freq_component = psd_taiji_oms_noise(f, oms_noise_level)
    return low_freq_component, high_freq_component


def omega_length(f: np.ndarray, len_arm: float) -> np.ndarray:
    omega_len = 2 * np.pi * f * len_arm / c
    return omega_len


def averaged_fplus_sq_approximated(f: np.ndarray, len_arm: float) -> np.ndarray:
    fp_sq_approx = (3.0 / 20.0) * (1.0 / (1.0 + 0.6 * omega_length(f, len_arm) ** 2))
    return fp_sq_approx


def averaged_tianqin_fplus_sq_numerical(
    f: np.ndarray, len_arm: float = np.sqrt(3) * 1e8
) -> np.ndarray:
    base = averaged_fplus_sq_approximated(f, len_arm)
    a = [
        1,
        1e-4,
        2639e-4,
        231 / 5 * 1e-4,
        -2093 / 1.25 * 1e-4,
        2173e-5,
        2101e-6,
        3027 / 2 * 1e-5,
        -42373 / 5 * 1e-6,
        176087e-8,
        -8023 / 5 * 1e-7,
        5169e-9,
    ]
    omega_len = omega_length(f, len_arm)
    omega_len_low_f_indices = np.where(omega_len < 4.1)
    omega_len_low_f = omega_len[omega_len_low_f_indices]
    base_low_f = base[omega_len_low_f_indices]
    low_f_modulation = np.polyval(a[::-1], omega_len_low_f)
    low_f_result = np.multiply(base_low_f, low_f_modulation)
    omega_len_high_f_indices = np.where(omega_len >= 4.1)
    omega_len_high_f = omega_len[omega_len_high_f_indices]
    base_high_f = base[omega_len_high_f_indices]
    high_f_modulation = np.exp(-0.322 * np.sin(2 * omega_len_high_f - 4.712) + 0.078)
    high_f_result = np.multiply(base_high_f, high_f_modulation)
    fp_sq_numerical = np.concatenate((low_f_result, high_f_result))
    return fp_sq_numerical


def sensitivity_curve_tianqin_analytical(
    flow: float,
    fhigh: float,
    Npts: int,
    len_arm: float = np.sqrt(3) * 1e8,
    acc_noise_level: float = 1e-15,
    oms_noise_level: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    fr = np.logspace(np.log10(flow), np.log10(fhigh), Npts)
    fp_sq = averaged_tianqin_fplus_sq_numerical(fr, len_arm)
    s_acc_nu, s_oms_nu = tianqin_psd_components(fr, acc_noise_level, oms_noise_level)
    omega_len = omega_length(fr, len_arm)
    sense_curve = (s_oms_nu + s_acc_nu * (3 + np.cos(2 * omega_len))) / (
        omega_len**2 * fp_sq
    )
    return fr, sense_curve / 2


def sensitivity_curve_taiji_analytical(
    flow: float,
    fhigh: float,
    Npts: int,
    len_arm: float = 3e9,
    acc_noise_level: float = 3e-15,
    oms_noise_level: float = 8e-12,
) -> tuple[np.ndarray, np.ndarray]:
    fr = np.logspace(np.log10(flow), np.log10(fhigh), Npts)
    fp_sq = averaged_fplus_sq_approximated(fr, len_arm)
    s_acc_nu, s_oms_nu = taiji_psd_components(fr, acc_noise_level, oms_noise_level)
    omega_len = omega_length(fr, len_arm)
    sense_curve = (s_oms_nu + s_acc_nu * (3 + np.cos(2 * omega_len))) / (
        omega_len**2 * fp_sq
    )
    return fr, sense_curve / 2


# ==============================================================================

# --- NEW: Generate data for Taiji & TianQin and define conversion functions ---

# Constants for conversion
H0_kms_Mpc = 67.4  # Hubble constant in km/s/Mpc
km_per_Mpc = 3.0857e19
H0_per_s = H0_kms_Mpc / km_per_Mpc  # H0 in 1/s


def Sn_to_OmegaGW(f, Sn):
    """Converts strain power spectral density Sn(f) to Omega_GW(f)."""
    return (10 * np.pi**2 / (3 * H0_per_s**2)) * f**3 * Sn


def OmegaGW_to_hc(f, omega_gw):
    """Converts Omega_GW to characteristic strain hc."""
    # Sn = (3 * H0_per_s**2) / (10 * np.pi**2 * f**3) * omega_gw
    # hc = sqrt(f * Sn)
    hc = np.sqrt(omega_gw * (3 * H0_per_s**2) / (10 * np.pi**2 * f**2))
    return hc


# Generate sensitivity curves up to 10 Hz
flow = 1e-5
fhigh = 10.0  # Extended frequency range
Npts = 2000

# Taiji
f_taiji, Sn_taiji = sensitivity_curve_taiji_analytical(flow, fhigh, Npts)
OmegaGW_taiji = Sn_to_OmegaGW(f_taiji, Sn_taiji)
hc_taiji = np.sqrt(f_taiji * Sn_taiji)

# TianQin
f_tianqin, Sn_tianqin = sensitivity_curve_tianqin_analytical(flow, fhigh, Npts)
OmegaGW_tianqin = Sn_to_OmegaGW(f_tianqin, Sn_tianqin)
hc_tianqin = np.sqrt(f_tianqin * Sn_tianqin)

# --- End of New Data Generation Section ---

# Load data files
data_lisa = np.loadtxt("GW sensitivity data/Data/plis_LISA.dat", comments="#")
data_decigo = np.loadtxt("GW sensitivity data/Data/plis_DECIGO.dat", comments="#")
data_et = np.loadtxt("GW sensitivity data/Data/plis_ET.dat", comments="#")
data_ipta = np.loadtxt("GW sensitivity data/Data/plis_IPTA.dat", comments="#")
data_epta = np.loadtxt("GW sensitivity data/Data/plis_EPTA.dat", comments="#")
data_nanograv = np.loadtxt("GW sensitivity data/Data/plis_NANOGrav.dat", comments="#")
data_SKA = np.loadtxt("GW sensitivity data/Data/plis_SKA.dat", comments="#")

# ==============================================================================
# FIGURE 1: OMEGA_GW PLOT (Your original plot, slightly modified for new data)
# ==============================================================================

# Create plot
fig1, ax1 = plt.subplots()

# Define detectors with their properties
# detectors = [
#     (data_lisa,    'LISA',              'CadetBlue',   0.26),
#     (data_decigo,  'DECIGO',            'MidnightBlue',    0.26),
#     (data_et,      'Einstein Telescope','green',  0.26),
#     (data_nanograv, 'PTA',          'brown',  0.26)
# ]
detectors = [(data_lisa, "LISA", "CadetBlue", 0.26)]

Or = 9.1 * 10 ** (-5)
Om = 0.315
Ol = 1 - Or - Om

for data, name, color, alpha in detectors:
    x = 10 ** data[:, 0]  # Frequency [Hz]
    y = 10 ** data[:, 1]  # h²Ω_GW
    ax1.plot(x, y, color=color, linewidth=2)
    ax1.fill_between(x, y, 10, color=color, alpha=alpha)

# Plot Taiji and TianQin (Omega_GW)
ax1.plot(f_taiji, OmegaGW_taiji, color="purple", linewidth=2)
ax1.fill_between(f_taiji, OmegaGW_taiji, 10, color="purple", alpha=0.26)

ax1.plot(f_tianqin, OmegaGW_tianqin, color="orange", linewidth=2)
ax1.fill_between(f_tianqin, OmegaGW_tianqin, 10, color="orange", alpha=0.26)


# Define conversion functions for secondary x-axis (k in Mpc⁻¹)
def freq_to_k(f):
    return 2 * np.pi * f / (9.72e-15)  # Convert Hz to k (Mpc⁻¹)


def k_to_freq(k):
    return k * 9.72e-15 / (2 * np.pi)  # Inverse conversion


# Create secondary x-axis for wavenumber k
secax1 = ax1.secondary_xaxis("top", functions=(freq_to_k, k_to_freq))
secax1.set_xlabel(r"\textbf{Wavenumber $k$ (Mpc$^{-1}$)}", fontsize=18)
secax1.set_xscale("log")

# Configure axes
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylim(1e-17, 1e-6)
ax1.set_xlim(5e-6, 1)  # Use the new extended frequency range
# Labels and Text
ax1.set_xlabel(r"\textbf{Frequency (Hz)}", fontsize=20)
ax1.set_ylabel(r"\textbf{$h^2 \Omega_{\text{GW}}$}", fontsize=20)
ax1.grid(True, which="both", ls="-", alpha=0.2)

# plt.text(1.8e-7, 4.46e-9 , r"\textbf{PTA}", style='normal', fontsize=18, color='brown',
#          verticalalignment='center', horizontalalignment='center', rotation = 'horizontal',
#           bbox={'facecolor': 'white', 'alpha': 0.12, 'pad': 5},zorder=3)
plt.text(
    5.33e-5,
    1.106e-11,
    r"\textbf{LISA}",
    style="normal",
    fontsize=18,
    color="CadetBlue",
    verticalalignment="center",
    horizontalalignment="center",
    rotation="horizontal",
    bbox={"facecolor": "white", "alpha": 0.12, "pad": 5},
    zorder=3,
)
# plt.text(1.04e-3, 2.17e-16 , r"\textbf{DECIGO}", style='normal', fontsize=18, color='MidnightBlue',
#          verticalalignment='center', horizontalalignment='center', rotation = 'horizontal',
#           bbox={'facecolor': 'white', 'alpha': 0.12, 'pad': 5},zorder=3)
# plt.text(485.409, 1.45e-12 , r"\textbf{ET}", style='normal', fontsize=18, color='g',
#          verticalalignment='center', horizontalalignment='center', rotation = 'horizontal',
#           bbox={'facecolor': 'white', 'alpha': 0.12, 'pad': 5},zorder=3)
plt.text(
    0.00099,
    5.72e-10,
    r"\textbf{Taiji}",
    style="normal",
    fontsize=18,
    color="purple",
    verticalalignment="center",
    horizontalalignment="center",
    rotation="horizontal",
    bbox={"facecolor": "white", "alpha": 0.12, "pad": 5},
    zorder=3,
)
plt.text(
    0.1,
    5.4e-8,
    r"\textbf{TianQin}",
    style="normal",
    fontsize=18,
    color="orange",
    verticalalignment="center",
    horizontalalignment="center",
    rotation="horizontal",
    bbox={"facecolor": "white", "alpha": 0.12, "pad": 5},
    zorder=3,
)

# Save and display
fig1.tight_layout()
fig1.savefig("Omega_GW_sensitivity.pdf", bbox_inches="tight", dpi=300)
fig1.savefig("Omega_GW_sensitivity.png", bbox_inches="tight", dpi=300)

# ==============================================================================
# # FIGURE 2: CHARACTERISTIC STRAIN (hc) PLOT
# # ==============================================================================
# fig2, ax2 = plt.subplots()

# # Plot detectors from files
# for data, name, color, alpha in detectors:
#     f = 10**data[:, 0]
#     omega_gw = 10**data[:, 1]
#     hc = OmegaGW_to_hc(f, omega_gw)
#     ax2.loglog(f, hc, color=color, linewidth=2, label=name)

# # Plot our generated curves
# ax2.loglog(f_taiji, hc_taiji, color='purple', linewidth=2, label='Taiji')
# ax2.loglog(f_tianqin, hc_tianqin, color='orange', linewidth=2, label='TianQin')

# # Configure axes
# ax2.set_xscale('log')
# ax2.set_yscale('log')
# ax2.set_xlim(flow, fhigh) # Use the new extended frequency range
# ax2.set_ylim(1e-24, 1e-12)

# # Labels and Legend
# ax2.set_xlabel(r"\textbf{Frequency (Hz)}", fontsize=20)
# ax2.set_ylabel(r"\textbf{Characteristic Strain ($h_c$)}", fontsize=20)
# ax2.grid(True, which="both", ls="-", alpha=0.2)
# ax2.legend(loc='lower left', fontsize=14)

# fig2.tight_layout()
# fig2.savefig('Characteristic_Strain_sensitivity.pdf', bbox_inches='tight', dpi=300)
# fig2.savefig('Characteristic_Strain_sensitivity.png', bbox_inches='tight', dpi=300)

# # ==============================================================================
# # FIGURE 3: CHARACTERISTIC STRAIN (hc) PLOT for LISA, Taiji, and TianQin only
# # ==============================================================================
# # fig3, ax3 = plt.subplots()

# # # Extract and convert LISA data
# # f_lisa = 10**data_lisa[:, 0]
# # omega_gw_lisa = 10**data_lisa[:, 1]
# # hc_lisa = OmegaGW_to_hc(f_lisa, omega_gw_lisa)

# # # Plot the three curves of interest
# # ax3.loglog(f_lisa, hc_lisa, color='CadetBlue', linewidth=2, label='LISA')
# # ax3.loglog(f_taiji, hc_taiji, color='purple', linewidth=2, label='Taiji')
# # ax3.loglog(f_tianqin, hc_tianqin, color='orange', linewidth=2, label='TianQin')

# # # Configure axes
# # ax3.set_xscale('log')
# # ax3.set_yscale('log')
# # ax3.set_xlim(flow, fhigh)
# # ax3.set_ylim(1e-24, 1e-16)

# # # Labels and Legend
# # ax3.set_xlabel(r"\textbf{Frequency (Hz)}", fontsize=20)
# # ax3.set_ylabel(r"\textbf{Characteristic Strain ($h_c$)}", fontsize=20)
# # plt.grid()
# # ax3.legend(loc='lower left', fontsize=14)

# # fig3.tight_layout()
# # fig3.savefig('Characteristic_Strain_LISA-TJ-TQ.pdf', bbox_inches='tight', dpi=300)
# # fig3.savefig('Characteristic_Strain_LISA-TJ-TQ.png', bbox_inches='tight', dpi=300)

plt.show()
