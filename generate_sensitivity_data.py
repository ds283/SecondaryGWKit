import sys
sys.path.append('/tmp')

import numpy as np
from gw_sensitivity_curves import sensitivity_curve_taiji_analytical, sensitivity_curve_tianqin_analytical

# Constants
H0_kms_Mpc = 67.4  # Hubble constant in km/s/Mpc
km_per_Mpc = 3.0857e19
H0_per_s = H0_kms_Mpc / km_per_Mpc # H0 in 1/s

def Sn_to_OmegaGW(f, Sn):
    """Converts strain power spectral density Sn(f) to Omega_GW(f)."""
    return (2 * np.pi**2 * f**3 * Sn) / (3 * H0_per_s**2)

# Generate sensitivity curves
flow = 1e-5
fhigh = 1.0
Npts = 1000

# Taiji
f_taiji, Sn_taiji = sensitivity_curve_taiji_analytical(flow, fhigh, Npts)
OmegaGW_taiji = Sn_to_OmegaGW(f_taiji, Sn_taiji)

# TianQin
f_tianqin, Sn_tianqin = sensitivity_curve_tianqin_analytical(flow, fhigh, Npts)
OmegaGW_tianqin = Sn_to_OmegaGW(f_tianqin, Sn_tianqin)

# Save to file
# Assuming f_taiji and f_tianqin are the same
output_data = np.vstack((f_taiji, OmegaGW_taiji, OmegaGW_tianqin)).T
np.savetxt('/tmp/sensitivity_data.txt', output_data,
           header='Frequency(Hz) Omega_GW(Taiji) Omega_GW(TianQin)')

print("Sensitivity data saved to /tmp/sensitivity_data.txt") 