import matplotlib.pyplot as plt
import numpy as np

# --- Import cosmology and QCD transition modules ---
from CosmologyModels.GenericEOS.QCD_Cosmology import QCD_Cosmology
from CosmologyModels.GenericEOS.QCD_EOS import QCD_EOS
from CosmologyModels.LambdaCDM.LambdaCDM import LambdaCDM
from CosmologyModels.LambdaCDM.Planck import Planck2018
from Units import Mpc_units

# --- Matplotlib configuration for LaTeX rendering ---
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}\usepackage{xcolor}\usepackage{amssymb}\boldmath"
)
plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = plt.rcParams["font.size"]
plt.rcParams["axes.titlesize"] = 1.4 * plt.rcParams["font.size"]
plt.rcParams["xtick.labelsize"] = 1.4 * plt.rcParams["font.size"]
plt.rcParams["ytick.labelsize"] = 1.4 * plt.rcParams["font.size"]
plt.rcParams["xtick.major.size"] = 3
plt.rcParams["xtick.minor.size"] = 3
plt.rcParams["xtick.major.width"] = 1
plt.rcParams["xtick.minor.width"] = 1
plt.rcParams["ytick.major.size"] = 3
plt.rcParams["ytick.minor.size"] = 3
plt.rcParams["ytick.major.width"] = 1
plt.rcParams["ytick.minor.width"] = 1
plt.rcParams["axes.linewidth"] = 1

# # --- Append the project root directory to sys.path ---
# root_dir = Path(__file__).parent.parent.parent
# sys.path.append(str(root_dir))


def test_T_z_conversion():
    """
    Test and plot the T(z) conversion along with effective degrees of freedom (G and Gs)
    as functions of redshift. This helps verify that your QCDCosmology class is converting
    redshift to temperature accurately.
    """

    # Choose a redshift range wide enough to cover the temperature domain from ~T0 up to the QCD transition
    # Note: With T_CMB ~ 2.7255 K and conversion factor ~8.6173e-14, reaching T ~ 0.12 GeV requires z ~ 5e11.
    # We use a logarithmic spacing for z.
    z_values = np.logspace(0.1, 17, 5000)  # z from 1 to 1e13

    # Initialize cosmology models
    units = Mpc_units()
    eos = QCD_EOS(units)
    params = Planck2018()
    lcdm = LambdaCDM(store_id=1, units=units, params=params)
    qcd = QCD_Cosmology(store_id=2, units=units, params=params, max_z=z_values.max())

    T_vals = []  # Temperature in GeV computed from T_of_z(z)
    g_star_vals = []  # Effective d.o.f for energy density G(T)
    g_s_vals = []  # Effective d.o.f for entropy density Gs(T)
    w_lcdm_vals = []  # LCDM equation of state parameter
    w_qcd_vals = []  # QCD equation of state parameter
    # w_lcdm = lcdm.wBackground(z) # LCDM equation of state parameter
    # Iterate over z-values to compute T, G, Gs, and w
    for z in z_values:
        T = qcd.T_z(z)

        T_vals.append(T)
        w_lcdm = lcdm.wBackground(z)
        w_lcdm_vals.append(w_lcdm)
        # Use the QCDCosmology method which applies the switch
        w_qcd_vals.append(eos.w(z))
        g_star_vals.append(eos.G(T))
        g_s_vals.append(eos.Gs(T))

    # --- Plot Temperature vs. Redshift ---
    # fig1, ax1 = plt.subplots()
    # ax1.semilogx(z_values, T_vals, 'b-', label=r'$T(z)$ [GeV]')
    # ax1.set_xlabel(r'Redshift, $z$')
    # ax1.set_ylabel(r'Temperature, $T$ [GeV]')
    # ax1.set_title(r'Conversion: Temperature vs. Redshift')
    # ax1.grid(True)
    # plt.xscale('log')
    # plt.yscale('log')
    # ax1.axhline(0.12, color='g', linestyle=':', label=r'$T_{QCD}$')
    # ax1.axvline(5e11, color='r', linestyle=':', label=r'$z_{QCD}$')
    # ax1.legend()
    # plt.tight_layout()
    # plt.show()

    # --- Plot Effective Degrees of Freedom (G and Gs) vs. Redshift ---
    fig2, ax2 = plt.subplots()
    # ax2.semilogx(z_values, g_star_vals, 'r-', label=r'$g_*(T)$')
    # ax2.semilogx(z_values, g_s_vals, 'g--', label=r'$g_s(T)$')
    # ax2.semilogx(z_values, w_lcdm_vals, 'b-', label=r'${\Lambda\mathrm{CDM}}$')
    # ax2.semilogx(z_values, w_qcd_vals, 'r-', label=r'$QCD$')
    plt.plot(z_values, w_qcd_vals, "r-", lw=3, label=r"$QCD$")
    plt.plot(z_values, w_lcdm_vals, "b--", lw=3, label=r"${\Lambda\mathrm{CDM}}$")
    # plt.plot(T_vals, g_star_vals, 'b-', label=r'$g_*(T)$')
    plt.xscale("log")
    # plt.yscale('log')
    # plt.axvline(5e-4, color='g', lw=2.5, linestyle=':', label=r'$e_{+} \,e_{-}$ annihilation')
    # plt.axvline(0.12, color='purple', lw=2.5, linestyle=':', label=r'$T_{QCD}$')
    plt.axvline(3400, color="orange", lw=2.5, linestyle=":", label=r"M-R equality")
    # ax2.semilogx(z_values, w_lcdm_vals, 'g--', label=r'$\Lambda CDM$')
    # ax2.set_xlabel(r'Redshift, $z$')
    # ax2.set_ylabel(r'Effective Degrees of Freedom')
    # ax2.set_title(r'Effective Degrees of Freedom vs. Redshift')
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_T_z_conversion()


# def test_single_T():
#     """Test temperature at a specific redshift"""

#     # Initialize QCD cosmology
#     units = Mpc_units()
#     params = Planck2018()
#     qcd = QCDCosmology(store_id=1, units=units, params=params)

# Get temperature at z=100
#     z = 1e12
#     T = qcd.T_of_z(z)

#     print(f"Temperature at z = {z}: T = {T:.3e} GeV")

# if __name__ == "__main__":
#     test_single_T()
