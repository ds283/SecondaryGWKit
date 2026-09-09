import sys, numpy as np, importlib.util
from math import atan2, sin, sqrt, pi, exp, floor, fabs
sys.path.insert(0, __file__.rsplit('/',1)[0])
from LiouvilleGreen.WKBtools import WKB_mod_2pi, shift_theta_sample, wrap_theta
from LiouvilleGreen.constants import TWO_PI
def sim(k, x_r, mode, rebase):
    s_e3, s_e4 = k/exp(3), k/exp(4); s_lim = sqrt(s_e3*s_e4)
    src = np.geomspace(s_lim, s_e3, int(np.log10(s_e3/s_lim)*100)+1)
    full = np.geomspace(1.0, s_e3, int(np.log10(s_e3)*100)+1); resp = full[::-1][::12][::-1]
    theta = lambda a, b: k*(1/a - 1/b); s_r = resp[np.argmin(abs(resp - k/x_r))]
    recs = []
    for s_s in src:
        x_s = k/s_s; p4 = -(exp(4)-x_s); off = pi/2 if mode=="extremum" else 0.0
        n = floor((-p4 - off)/pi); th_i = -(off + n*pi)
        while th_i > p4: th_i -= pi
        s_i = 1/(1/s_s - th_i/k)
        G_i = (s_s**2/k)*sin(th_i); Gp_i = np.cos(th_i)*s_s**2/s_i**2; om_i = k/s_i**2
        delta = atan2(sqrt(om_i)*G_i, Gp_i/sqrt(om_i))
        rs = resp[(resp <= min(s_e3, s_i)*(1+1e-7))][::-1]
        d, m = map(list, zip(*[WKB_mod_2pi(theta(s_i, s)) for s in rs]))
        if rebase: d2, m2 = shift_theta_sample(d, m, delta)
        else:
            sh = [wrap_theta(mm + delta) for mm in m]; d2 = [dd + s for dd, (s, _) in zip(d, sh)]; m2 = [mm for (_, mm) in sh]
        j = int(np.argmin(abs(rs - s_r))); recs.append((int(d2[j]), float(m2[j]), theta(s_s, s_r), wrap_theta(m[0]+delta)[0]))
    last_mod = last_div = None; sub = blk = None; rect = []
    for div, mod, ex, b in recs:
        if last_mod is None: sub = div; blk = 0; rd = 0
        else:
            if TWO_PI*div + mod > TWO_PI*last_div + last_mod:
                dm = mod - last_mod; jump = min([(fabs(dm),0),(fabs(dm+TWO_PI),1),(fabs(dm-TWO_PI),-1)], key=lambda x:x[0])[1]
                rd = blk + jump; sub = div - rd
            else: rd = div - sub
            blk = rd
        last_mod, last_div = mod, div; rect.append(TWO_PI*rd + mod)
    dev = np.diff(np.array(rect)) - np.diff(np.array([r[2] for r in recs]))
    bases = [r[3] for r in recs]
    return len(recs), int(np.sum(np.abs(dev) > pi)), bases.count(1), bases.count(-1)
for mode in ["extremum", "zero"]:
    for x_r in [1e2, 1e3, 1e4]:
        tot = dict(True_=[0,0,0,0], False_=[0,0,0,0])
        for k in np.geomspace(1e6, 1e8, 15):
            for rebase in [True, False]:
                n, kinks, bp, bm = sim(k, x_r, mode, rebase)
                key = "True_" if rebase else "False_"
                tot[key][0]+=n; tot[key][1]+=kinks; tot[key][2]+=bp; tot[key][3]+=bm
        for key in tot:
            n, kinks, bp, bm = tot[key]
            print(f"stop@{mode:8s} x_r={x_r:.0e} rebase={key[:-1]:5s}: {n} objects over 15 k values, base=+1: {bp}, base=-1: {bm}, 2pi-kinks after rectifier: {kinks}")
