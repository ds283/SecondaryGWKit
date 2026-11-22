from pathlib import Path

p = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07d/measurement/scripts/leafcc_07d.py")
s = p.read_text()
old = '''            ttp = np.cos(np.pi * (np.arange(Mp) + 0.5) / Mp)
            xsp = 0.5 * (pa + pb) + 0.5 * lp_ * ttp
            gsp, fsp = g_on(xsp, with_parts=True)
            _, Ilo_p, rlo_p, _ = ls_fit(pa, pb, gsp, ttp, dl)
            _, Ihi_p, rhi_p, qk_p = ls_fit(pa, pb, gsp, ttp, dh)'''
new = '''            ttp = np.cos(np.pi * (np.arange(Mp) + 0.5) / Mp)
            xsp = 0.5 * (pa + pb) + 0.5 * lp_ * ttp
            gsp, fsp = g_on(xsp, with_parts=True)
            # v3: raise the degrees until the fit has converged: while the d_lo fit's residual RMS exceeds CONV_RATIO
            # times the d_hi fit's, both step up by the step (8 for one piece, 4 for several), up to M_p // 4 (one piece)
            # or M_p // 3 (several)
            step = 8 if len(brk) == 2 else 4
            cap = Mp // 4 if len(brk) == 2 else Mp // 3
            while True:
                _, Ilo_p, rlo_p, _ = ls_fit(pa, pb, gsp, ttp, dl)
                _, Ihi_p, rhi_p, qk_p = ls_fit(pa, pb, gsp, ttp, dh)
                s_lo_, s_hi_ = float(np.sqrt(np.mean(rlo_p ** 2))), float(np.sqrt(np.mean(rhi_p ** 2)))
                if s_lo_ <= CONV_RATIO * s_hi_ or dh + step > cap:
                    break
                dl, dh = dl + step, dh + step'''
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace("PIECE_MIN_M = 96  # v2: least squares on each knot interval, at least this many points a piece\n",
              "PIECE_MIN_M = 96  # v2: least squares on each knot interval, at least this many points a piece\n"
              "CONV_RATIO = 1.5  # v3: a fit has converged when sigma(d_lo) <= CONV_RATIO sigma(d_hi)\n")
s = s.replace('''"deg": [d_lo, d_hi],''', '''"deg": [d_lo, d_hi], "deg_pieces": [list(p["d"]) for p in pieces],''')
p.write_text(s)
print("patched v3")
