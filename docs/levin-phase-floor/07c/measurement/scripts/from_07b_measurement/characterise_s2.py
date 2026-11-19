"""levin-phase-floor prompt 06 section 6: a read-only characterisation of S2's windows from
state/probe_windows.json (written by analyse06.py), descriptive. For the T windows: sigma_s/sigma_t5,
sigma in units of eps|theta| and of F, by verdict, z range, factor and row; and the same for the
pure-WKB and seeded G windows. Reads nothing else."""
import json
import sys
import numpy as np

EPS = 2.0**-52
W = json.load(open(sys.argv[1]))
def q(a):
    a = np.asarray(a, dtype=float)
    return f"min {a.min():.3g} median {np.median(a):.3g} p90 {np.percentile(a, 90):.3g} max {a.max():.3g}" if a.size else "none"
T = [w for w in W if w["factor"] in ("Tq", "Tr") and w["matched"]]
for label, sel in (("S2 true", lambda w: w["true"]), ("S2 false", lambda w: not w["true"])):
    ws = [w for w in T if sel(w)]
    print(f"== T windows, {label}: {len(ws)}")
    if not ws:
        continue
    print("   sigma_s/sigma_t5:", q([w["sigma_s"] / w["sigma_t5"] for w in ws]))
    print("   sigma_t5/(eps|theta|):", q([w["sigma_t5"] / (EPS * w["abs_theta_centre"]) for w in ws]))
    print("   sigma_s/(eps|theta|):", q([w["sigma_s"] / (EPS * w["abs_theta_centre"]) for w in ws]))
    print("   sigma_s/F:", q([w["sigma_s"] / w["F"] for w in ws]))
    print("   sigma_s (rad):", q([w["sigma_s"] for w in ws]))
    print("   z_centre:", q([w["z_centre"] for w in ws]))
    print("   |theta| at centre:", q([w["abs_theta_centre"] for w in ws]))
    by = {}
    for w in ws:
        by.setdefault((w["row"], w["factor"]), []).append(w["z_centre"])
    print("   by (row, factor): " + "; ".join(f"{k[0]} {k[1]} {len(v)} (z {min(v):.3g}-{max(v):.3g})" for k, v in sorted(by.items())))
# the S2 false windows as a fraction, binned in z
zs_all = np.array([w["z_centre"] for w in T]); f_all = np.array([not w["true"] for w in T])
edges = [0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e20]
print("== T windows by z_centre bin: n, false")
for a, b in zip(edges[:-1], edges[1:]):
    m = (zs_all >= a) & (zs_all < b)
    print(f"   [{a:.0e}, {b:.0e}): {m.sum()}, {f_all[m].sum()}")
G = [w for w in W if w["factor"] == "G" and w["matched"]]
for label, sel in (("pure-WKB (I)", lambda w: w.get("pure_I")), ("seeded", lambda w: not w.get("pure_I"))):
    ws = [w for w in G if sel(w)]
    print(f"== G windows, {label}: {len(ws)}")
    print("   sigma_t5/(eps|theta|):", q([w["sigma_t5"] / (EPS * w["abs_theta_centre"]) for w in ws]))
    print("   sigma_s/(eps|theta|):", q([w["sigma_s"] / (EPS * w["abs_theta_centre"]) for w in ws]))
    print("   sigma_t5/F:", q([w["sigma_t5"] / w["F"] for w in ws]))
    print("   sigma_s/F:", q([w["sigma_s"] / w["F"] for w in ws]))
    print("   z_centre:", q([w["z_centre"] for w in ws]))
    if label == "seeded":
        du = np.array([w["dtheta_rms_unwrapped"] for w in ws]); dr = np.array([w["dtheta_rms_raw"] for w in ws])
        print("   dtheta RMS unwrapped/F:", q(du / np.array([w["F"] for w in ws])))
        print("   dtheta RMS raw/F:", q(dr / np.array([w["F"] for w in ws])))
        print("   sigma_s / dtheta RMS unwrapped:", q(np.array([w["sigma_s"] for w in ws]) / np.where(du > 0, du, np.nan)))
        print("   windows with dtheta RMS unwrapped > 1e-3 rad:", int((du > 1e-3).sum()))
