"""phase_spline chunking on the consumer geometry: theta(z_source) at fixed z_response, exact radiation."""
import sys, numpy as np, mpmath as mp
sys.path.insert(0, __file__.rsplit('/',1)[0])
from common import *
from LiouvilleGreen.phase_spline import phase_spline
from LiouvilleGreen.WKBtools import WKB_mod_2pi
from LiouvilleGreen.constants import TWO_PI
def build(k, s_lo, s_hi, per_decade, chunklog):
    s = np.geomspace(s_lo, s_hi, int(np.log10(s_hi/s_lo)*per_decade)+1); z = s - 1
    # theta(z_r=0 ; z_s) = k(1/s_s - 1) : negative, magnitude increasing with z_s
    th = [mp.mpf(k)*(1/mp.mpf(float(v)) - 1) for v in s]
    pairs = [WKB_mod_2pi(float(t)) for t in th]
    d, m = map(list, zip(*pairs)); base = d[0]; d = [v - base for v in d]
    spl = phase_spline(z, d, m, x_is_redshift=True, increasing=False, chunk_step=None, chunk_logstep=chunklog)
    return s, z, th, d, m, spl, base
def probe(k, per_decade, chunklog):
    s, z, th, d, m, spl, base = build(k, 10.0, 1e4, per_decade, chunklog)
    # fine evaluation grid: 10 per interval, avoid last 3 intervals each end for "interior"
    logs = np.log(s); fine = np.concatenate([np.linspace(logs[i], logs[i+1], 11)[:-1] for i in range(len(s)-1)])
    zf = np.exp(fine) - 1
    exact = np.array([float(mp.mpf(k)*(mp.exp(-mp.mpf(float(u))) - 1) - base*2*mp.pi) for u in fine])
    raw = np.array([spl.raw_theta(float(u), x_is_log=True) for u in fine])
    err = raw - exact
    nint = len(s)-1; interior = slice(3*10, (nint-3)*10)
    # chunk switching: which chunk is used at each fine point
    which = np.array([spl._chunk_list.index(next(c for c, sp in spl._splines.items() if sp is spl._match_chunk(float(u))[0])) for u in fine])
    switches = np.nonzero(np.diff(which))[0]
    jumps = []
    for i in switches:
        u = 0.5*(fine[i]+fine[i+1])
        a = spl._splines[spl._chunk_list[which[i]]]; b = spl._splines[spl._chunk_list[which[i+1]]]
        ta = a.raw_theta(raw_x=np.exp(u)-1, log_x=u, warn_unsafe=False); tb = b.raw_theta(raw_x=np.exp(u)-1, log_x=u, warn_unsafe=False)
        da = a.theta_deriv(raw_x=np.exp(u)-1, log_x=u, log_derivative=True, warn_unsafe=False); db = b.theta_deriv(raw_x=np.exp(u)-1, log_x=u, log_derivative=True, warn_unsafe=False)
        jumps.append((float(np.exp(u)), ta-tb, (da-db)/da))
    ymax = max(np.max(np.abs(sp._y_points)) for sp in spl._splines.values())
    knot_err = max(abs(float(spl.raw_theta(float(zz), x_is_log=False) - (float(t) - base*2*np.pi))) for zz, t in zip(z, th))
    print(f"k={k:.0e} {per_decade:5.0f}/dec chunklog={str(chunklog):>4}: chunks={spl.num_chunks} | interior max|err|={np.max(np.abs(err[interior])):.2e} rad, incl. ends={np.max(np.abs(err)):.2e} | predicted h^4 x_max/384={ (np.log(10)/per_decade)**4*k/10/384:.1e} | max spline ordinate={ymax:.3g} rad (global span {abs(float(th[-1]-th[0])):.3g}) | knot residual={knot_err:.2e} | switches={len(switches)} " + "; ".join(f"at s={sz:.4g}: dtheta={dj:.2e} rad, dtheta'/theta'={dd:.1e}" for sz, dj, dd in jumps))
for k in [1e6, 1e8]:
    for per_decade in [100, 300]:
        for chunklog in [None, 125]:
            probe(k, per_decade, chunklog)
