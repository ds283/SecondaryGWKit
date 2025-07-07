import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import c

# Helper functions from pycbc.psd.analytical_space
# Source: https://pycbc.org/pycbc/latest/html/_modules/pycbc/psd/analytical_space.html

def _psd_acc_noise(f, acc_noise_level=None):
    """ The PSD of TDI-based space-borne GW
    detectors' acceleration noise. Note that
    this is suitable for LISA and Taiji, TianQin
    has a different form.
    """
    s_acc = acc_noise_level**2 * (1+(4e-4/f)**2)*(1+(f/8e-3)**4)
    s_acc_d = s_acc * (2*np.pi*f)**(-4)
    s_acc_nu = (2*np.pi*f/c)**2 * s_acc_d
    return s_acc_nu

def psd_tianqin_acc_noise(f, acc_noise_level=1e-15):
    """ The PSD of TianQin's acceleration noise.
    """
    s_acc_d = acc_noise_level**2 * (2*np.pi*f)**(-4) * (1+1e-4/f)
    s_acc_nu = (2*np.pi*f/c)**2 * s_acc_d
    return s_acc_nu

def psd_taiji_acc_noise(f, acc_noise_level=3e-15):
    """ The PSD of Taiji's acceleration noise.
    """
    s_acc_nu = _psd_acc_noise(f, acc_noise_level)
    return s_acc_nu

def _psd_oms_noise(f, oms_noise_level=None):
    """ The PSD of TDI-based space-borne GW detectors' OMS noise.
    Note that this is suitable for LISA and Taiji, TianQin
    has a different form.
    """
    s_oms_d = oms_noise_level**2 * (1+(2e-3/f)**4)
    s_oms_nu = s_oms_d * (2*np.pi*f/c)**2
    return s_oms_nu

def psd_tianqin_oms_noise(f, oms_noise_level=1e-12):
    """ The PSD of TianQin's OMS noise.
    """
    s_oms_d = oms_noise_level**2
    s_oms_nu = s_oms_d * (2*np.pi*f/c)**2
    return s_oms_nu

def psd_taiji_oms_noise(f, oms_noise_level=8e-12):
    """ The PSD of Taiji's OMS noise.
    """
    s_oms_nu = _psd_oms_noise(f, oms_noise_level)
    return s_oms_nu

def tianqin_psd_components(f, acc_noise_level=1e-15, oms_noise_level=1e-12):
    """ The PSD of TianQin's acceleration and OMS noise.
    """
    acc_noise_level = np.float64(acc_noise_level)
    oms_noise_level = np.float64(oms_noise_level)
    low_freq_component = psd_tianqin_acc_noise(f, acc_noise_level)
    high_freq_component = psd_tianqin_oms_noise(f, oms_noise_level)
    return low_freq_component, high_freq_component

def taiji_psd_components(f, acc_noise_level=3e-15, oms_noise_level=8e-12):
    """ The PSD of Taiji's acceleration and OMS noise.
    """
    acc_noise_level = np.float64(acc_noise_level)
    oms_noise_level = np.float64(oms_noise_level)
    low_freq_component = psd_taiji_acc_noise(f, acc_noise_level)
    high_freq_component = psd_taiji_oms_noise(f, oms_noise_level)
    return low_freq_component, high_freq_component

def omega_length(f, len_arm=None):
    """ The function to calculate 2*pi*f*arm_length.
    """
    omega_len = 2*np.pi*f * len_arm/c
    return omega_len

def averaged_fplus_sq_approximated(f, len_arm=None):
    r""" A simplified fit for TDI-based space-borne GW detectors'
    squared antenna response function, averaged over sky and
    polarization angle.
    """
    fp_sq_approx = (3./20.)*(1./(1.+0.6*omega_length(f, len_arm)**2))
    return fp_sq_approx

def averaged_tianqin_fplus_sq_numerical(f, len_arm=np.sqrt(3)*1e8):
    """ A numerical fit for TianQin's squared antenna response function,
    averaged over sky and polarization angle.
    """
    base = averaged_fplus_sq_approximated(f, len_arm)
    a = [1, 1e-4, 2639e-4, 231/5*1e-4, -2093/1.25*1e-4, 2173e-5,
         2101e-6, 3027/2*1e-5, -42373/5*1e-6, 176087e-8,
         -8023/5*1e-7, 5169e-9]
    omega_len = omega_length(f, len_arm)
    # The polyval can be unstable at the edges of the interval. This can be fixed
    # by taking only the values where omega_len is less than 4.1
    omega_len_low_f_indices = np.where(omega_len < 4.1)
    omega_len_low_f = omega_len[omega_len_low_f_indices]

    base_low_f = base[omega_len_low_f_indices]

    low_f_modulation = np.polyval(a[::-1], omega_len_low_f)
    low_f_result = np.multiply(base_low_f, low_f_modulation)

    # For the rest of the values, use the other formula
    omega_len_high_f_indices = np.where(omega_len >= 4.1)
    omega_len_high_f = omega_len[omega_len_high_f_indices]
    base_high_f = base[omega_len_high_f_indices]
    high_f_modulation = np.exp(
        -0.322 * np.sin(2*omega_len_high_f-4.712) + 0.078
    )
    high_f_result = np.multiply(base_high_f, high_f_modulation)
    fp_sq_numerical = np.concatenate((low_f_result, high_f_result))

    return fp_sq_numerical


def sensitivity_curve_tianqin_analytical(flow, fhigh, Npts,
                                         len_arm=np.sqrt(3)*1e8,
                                         acc_noise_level=1e-15,
                                         oms_noise_level=1e-12):
    """ The analytical TianQin's sensitivity curve (6-links),
    averaged over sky and polarization angle.
    """
    len_arm = np.float64(len_arm)
    acc_noise_level = np.float64(acc_noise_level)
    oms_noise_level = np.float64(oms_noise_level)
    fr = np.logspace(np.log10(flow), np.log10(fhigh), Npts)
    fp_sq = averaged_tianqin_fplus_sq_numerical(fr, len_arm)
    s_acc_nu, s_oms_nu = tianqin_psd_components(
                            fr, acc_noise_level, oms_noise_level)
    omega_len = omega_length(fr, len_arm)
    sense_curve = ((s_oms_nu + s_acc_nu*(3+np.cos(2*omega_len))) /
                   (omega_len**2*fp_sq))
    return fr, sense_curve/2

def sensitivity_curve_taiji_analytical(flow, fhigh, Npts,
                                       len_arm=3e9, acc_noise_level=3e-15,
                                       oms_noise_level=8e-12):
    """ The analytical Taiji's sensitivity curve (6-links),
    averaged over sky and polarization angle.
    """
    len_arm = np.float64(len_arm)
    acc_noise_level = np.float64(acc_noise_level)
    oms_noise_level = np.float64(oms_noise_level)
    fr = np.logspace(np.log10(flow), np.log10(fhigh), Npts)
    fp_sq = averaged_fplus_sq_approximated(fr, len_arm)
    s_acc_nu, s_oms_nu = taiji_psd_components(
                            fr, acc_noise_level, oms_noise_level)
    omega_len = omega_length(fr, len_arm)
    sense_curve = ((s_oms_nu + s_acc_nu*(3+np.cos(2*omega_len))) /
                   (omega_len**2*fp_sq))
    return fr, sense_curve/2 