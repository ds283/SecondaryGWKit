"""Probe one Levin leaf whose non-endpoint round-off exceeds eq. (151): where does it come from?
Compares the double solve's intermediate quantities with the same scheme in mpmath."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, mpmath as mp
import common as C
lq = C.lq
import json
print(json.dumps(C.header(__file__)))
ci = int(sys.argv[1]); a = float(sys.argv[2]); b = float(sys.argv[3])
case = C.CASES[ci]; span = (a, b); N = C.ORDER
ph = C.phase(case); f = C.amplitudes()
grid, Dmat = lq.chebyshev_matrices(span, N)
thp = ph["theta_deriv"](grid); th = ph["theta"](grid)
F = np.concatenate([fn(grid) for fn in f])
L = Dmat.astype(complex); L[np.diag_indices(N)] += 1j * thp
rhs = F[:N] + 1j * F[N:]
span_phase = float(np.mean(np.abs(thp)) * abs(grid[0] - grid[-1]))
q_solve = np.linalg.solve(L, rhs)
q_lstsq = np.linalg.lstsq(L, rhs, rcond=None)[0]
print("phase_span", span_phase, "path", "solve" if span_phase > lq._LEVIN_DIRECT_SOLVE_PHASE_SPAN else "lstsq", "cond", np.linalg.cond(L))
data = C.solve_leaf(case, span)
def val(q, wb, wa):
    P = np.vstack([q.real, q.imag])
    return sum(P[i, 0] * wb[i] for i in range(2)) - sum(P[i, -1] * wa[i] for i in range(2))
wb = [np.sin(th[0]), np.cos(th[0])]; wa = [np.sin(th[-1]), np.cos(th[-1])]
print("driver value", data["value"], "solve", val(q_solve, wb, wa), "lstsq", val(q_lstsq, wb, wa))
with mp.workdps(C.MP_DPS):
    th_m, thp_m, f1, f2 = C._mp_funcs(case)
    t, Db = C.mp_base(N)
    A, B = mp.mpf(a), mp.mpf(b); mid, half = (A + B) / 2, (B - A) / 2
    x = [mid + half * ti for ti in t]
    Lm = mp.matrix(N, N)
    for i in range(N):
        for j in range(N):
            Lm[i, j] = Db[i, j] / half
        Lm[i, i] += mp.mpc(0, 1) * thp_m(x[i])
    qm = mp.lu_solve(Lm, mp.matrix([f1(xi) + mp.mpc(0, 1) * f2(xi) for xi in x]))
    print("q end values: double", q_solve[0], q_solve[-1])
    print("              mp    ", complex(qm[0]), complex(qm[N - 1]))
    q_used = q_solve if span_phase > lq._LEVIN_DIRECT_SOLVE_PHASE_SPAN else q_lstsq
    print("q_used vs mp, rel err at ends:", abs(complex(qm[0]) - q_used[0]) / abs(q_used[0]), abs(complex(qm[N - 1]) - q_used[-1]) / abs(q_used[-1]), "sv", np.linalg.svd(L, compute_uv=False)[[0, -1]])
    print("rel err q[0]", abs(complex(qm[0]) - q_solve[0]) / abs(q_solve[0]), "q[-1]", abs(complex(qm[N - 1]) - q_solve[-1]) / abs(q_solve[-1]))
    print("theta(b): double", th[0], "exact", mp.nstr(th_m(B), 25), "diff", float(mp.mpf(th[0]) - th_m(B)))
    print("theta(a): double", th[-1], "exact", mp.nstr(th_m(A), 25), "diff", float(mp.mpf(th[-1]) - th_m(A)))
    print("sin(fl theta b): np", wb[0], "mp", float(mp.sin(mp.mpf(th[0]))), "cos", wb[1], float(mp.cos(mp.mpf(th[0]))))
    print("sin err ulps", float((mp.mpf(wb[0]) - mp.sin(mp.mpf(th[0]))) / mp.mpf(np.spacing(abs(wb[0])))), float((mp.mpf(wb[1]) - mp.cos(mp.mpf(th[0]))) / mp.mpf(np.spacing(abs(wb[1])))))
    print("sin err ulps a", float((mp.mpf(wa[0]) - mp.sin(mp.mpf(th[-1]))) / mp.mpf(np.spacing(abs(wa[0])))), float((mp.mpf(wa[1]) - mp.cos(mp.mpf(th[-1]))) / mp.mpf(np.spacing(abs(wa[1])))))
    # per-node displacement
    disp = [float((mp.mpf(grid[k]) - x[k]) * 2**53) for k in range(N)]
    print("node displacement (u):", [round(d, 3) for d in disp])
    print("theta' rel err at nodes:", max(abs(float((mp.mpf(thp[k]) - thp_m(x[k])) / thp_m(x[k]))) for k in range(N)))
    # value with mp P and double w; double P and mp w
    Pm = [[mp.re(qm[i]) for i in range(N)], [mp.im(qm[i]) for i in range(N)]]
    def vm(P, wb_, wa_):
        return sum(P[i][0] * wb_[i] for i in range(2)) - sum(P[i][-1] * wa_[i] for i in range(2))
    wb_e = [mp.sin(th_m(B)), mp.cos(th_m(B))]; wa_e = [mp.sin(th_m(A)), mp.cos(th_m(A))]
    wb_d = [mp.mpf(v) for v in wb]; wa_d = [mp.mpf(v) for v in wa]
    Pd = [[mp.mpf(q_used[i].real) for i in range(N)], [mp.mpf(q_used[i].imag) for i in range(N)]]
    exact_scheme = vm(Pm, wb_e, wa_e)
    print("value parts: (Pd,wd)-(Pm,we)", float(vm(Pd, wb_d, wa_d) - exact_scheme),
          " (Pm,wd)-(Pm,we) [endpoint basis]", float(vm(Pm, wb_d, wa_d) - exact_scheme),
          " (Pd,we)-(Pm,we) [P error]", float(vm(Pd, wb_e, wa_e) - exact_scheme))
    print("floor", data["phase_err"])
    # sensitivity: perturb theta' and f by 1 ulp at random, 200 trials
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(200):
        thp2 = thp * (1 + rng.choice([-1, 1], N) * 2**-53)
        F2 = F * (1 + rng.choice([-1, 1], 2 * N) * 2**-53)
        L2 = Dmat.astype(complex); L2[np.diag_indices(N)] += 1j * thp2
        r2 = F2[:N] + 1j * F2[N:]
        q2 = np.linalg.solve(L2, r2) if span_phase > lq._LEVIN_DIRECT_SOLVE_PHASE_SPAN else np.linalg.lstsq(L2, r2, rcond=None)[0]
        vals.append(val(q2, wb, wa))
    print("ulp-perturbation spread of value (std, max-min):", np.std(vals), np.ptp(vals))
