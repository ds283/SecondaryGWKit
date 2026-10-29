import numpy as np, mpmath as mp
mp.mp.dps = 40
from AdaptiveLevin import levin_quadrature as LQ
u = 2.0**-53
N = 16; n = 2*N-1
def cell(c, x0, h, reduced, offset=0.0):
    a, b = x0, x0 + h
    th = lambda x: offset + c * x
    thm = lambda x: float(mp.fmod(mp.mpf(offset) + mp.mpf(c) * mp.mpf(x), 2 * mp.pi))
    d = {"theta": th}
    if reduced: d["theta_mod_2pi"] = thm
    B = LQ._Basis_SinCos(d)
    f = [lambda x: 1.0, lambda x: 0.0]
    grid, Dm = LQ.chebyshev_matrices((a, b), N)
    _, tp, _, _, ps = B.build_Levin_data(grid, Dm, need_AmatT=False)
    r = LQ._adaptive_levin_subregion_cc((a, b), f, B, N, ps, tp, None, None)
    A, Bm, C, O = mp.mpf(a), mp.mpf(b), mp.mpf(c), mp.mpf(offset)
    ref = (mp.cos(O + C*A) - mp.cos(O + C*Bm)) / C
    err = mp.mpf(r["value"]) - ref
    # rule with exact nodes and exact weights (float weights) evaluated in mp: isolates node rounding
    xb = [mp.cos(k*mp.pi/(n-1)) for k in range(n)]
    xe = [(A+Bm)/2 + (Bm-A)/2*t for t in xb]
    w = LQ._cc_weights((a, b), n)
    fg, _ = LQ.chebyshev_matrices((a, b), n)
    v_exact_nodes = mp.fsum(mp.mpf(w[k]) * mp.sin(O + C*xe[k]) for k in range(n))
    v_float_nodes = mp.fsum(mp.mpf(w[k]) * mp.sin(O + C*mp.mpf(fg[k])) for k in range(n))
    args = np.array([thm(x) if reduced else th(x) for x in fg])
    S = np.sum(np.abs(w) * np.abs(np.sin(args)))
    phi = (n + 4) * u * S + u * np.sum(np.abs(w) * np.abs(args))
    return float(err), float(v_float_nodes - v_exact_nodes), float(v_exact_nodes - ref), phi
print("theta = c x on [1, 1+10/c]")
for reduced in (False, True):
    for c in [1e3, 1e6, 1e9, 1e12]:
        e, dn, dr, phi = cell(c, 1.0, 10.0/c, reduced)
        print(f" red={reduced} c={c:.0e} err={e:+.3e} node_shift={dn:+.3e} rule_exact_nodes={dr:+.3e} phi={phi:.3e} |err|/phi={abs(e)/phi:.3g}")
print("theta = Theta + 10 x on [0, 1]  (large |theta|, small theta' x)")
for reduced in (False, True):
    for Th in [1e3, 1e6, 1e9, 1e12]:
        e, dn, dr, phi = cell(10.0, 0.0, 1.0, reduced, offset=Th)
        print(f" red={reduced} Theta={Th:.0e} err={e:+.3e} node_shift={dn:+.3e} rule={dr:+.3e} phi={phi:.3e} |err|/phi={abs(e)/phi:.3g}")

print("candidate node term: 2u * max|theta'_coarse| * sum_k |w_k| (max(|a|,|b|)+h) sum_i|f_ik|")
def node_term(c, a, b):
    n2 = n
    w = LQ._cc_weights((a, b), n2)
    return 2*u*abs(c)*np.sum(np.abs(w))*(max(abs(a),abs(b)) + abs(b-a))
for reduced in (False, True):
    for c in [1.0, 1e3, 1e6, 1e9, 1e12]:
        e, dn, dr, phi = cell(c, 1.0, 10.0/c, reduced)
        nt = node_term(c, 1.0, 1.0+10.0/c)
        print(f" red={reduced} c={c:.0e} |err|={abs(e):.3e} phi={phi:.3e} node={nt:.3e} |err|/(phi+node)={abs(e)/(phi+nt):.3g}")
