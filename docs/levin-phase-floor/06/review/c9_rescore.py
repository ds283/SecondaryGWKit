import json, sys, collections, math
import numpy as np
W, D = sys.argv[1], sys.argv[2]
PHI_WINDOW, PHI_DEG = 9, 4
def wrms(u, y):
    t = (2.0 * u - (u[0] + u[-1])) / (u[-1] - u[0])
    c = np.polynomial.polynomial.polyfit(t, y, PHI_DEG)
    r = y - np.polynomial.polynomial.polyval(t, c)
    return float(np.sqrt(np.mean(r * r)))
H = lambda v: np.array([float.fromhex(x) for x in v])
wins = json.load(open(f"{W}/state/probe_windows.json"))
print("windows", len(wins), collections.Counter(w['factor'] for w in wins))
sig_mismatch = 0; tagI_mismatch = 0; tagL = 0; zinit_tree_mismatch = 0
S1 = collections.Counter(); S2 = collections.Counter(); seeded = []
s1_ratio = []
for w in wins:
    z = H(w['z']); u = np.log1p(z)
    st5 = wrms(u, H(w['phi_t5'])); ss = wrms(u, H(w['phi_s']))
    if st5 != w['sigma_t5'] or ss != w['sigma_s']: sig_mismatch += 1
    ok = ss <= max(1e-3 * st5, 1e-6 * w['F'])
    if w['factor'] == 'G':
        zi = w['z_init_s']
        if zi != w['z_init_t5']: zinit_tree_mismatch += 1
        pureI = all(a is not None and a == b for a, b in zip(zi, z.tolist())) and len(zi) == 9
        if any(a is None for a in zi) and all(a is None for a in zi): tagL += 1
        if pureI != w['pure_I']: tagI_mismatch += 1
        if pureI:
            S1['true' if ok else 'false'] += 1; s1_ratio.append(ss / w['F'])
        else:
            seeded.append((ss / w['F'], st5 / w['F'], w['dtheta_rms_unwrapped'] / w['F'] if w['dtheta_rms_unwrapped'] is not None else None))
    else:
        S2['true' if ok else 'false'] += 1
print("sigma recomputed != saved:", sig_mismatch)
print("pure-WKB(I) tag re-derived != saved:", tagI_mismatch, "; G windows whose z_init differs between trees:", zinit_tree_mismatch, "; windows tagged by (L), all null:", tagL)
print("S1 (I): pure-WKB G windows", sum(S1.values()), dict(S1), "-> holds" if S1 and S1['false'] == 0 else "-> NOT holds", "; sigma_s/F max", max(s1_ratio) if s1_ratio else None, "median", float(np.median(s1_ratio)) if s1_ratio else None)
sd = np.array([x[0] for x in seeded]); st = np.array([x[1] for x in seeded]); dt = np.array([x[2] for x in seeded if x[2] is not None])
print("S1 seeded windows", len(seeded), "sigma_s/F median", float(np.median(sd)), "max", float(sd.max()), "; sigma_t5/F median", float(np.median(st)), "max", float(st.max()), "; dtheta RMS/F median", float(np.median(dt)), "n", dt.size)
print("S2: T windows", sum(S2.values()), dict(S2), "-> holds" if S2['false'] == 0 else "-> fails")
# checkpoints
U = {}
for l in open(f"{D}/checkpoint.jsonl"):
    d = json.loads(l)['data']; U[d['unit']] = d
print("units", len(U), collections.Counter(d['status'] for d in U.values()))
rows = sorted({d['row'] for d in U.values()}); d20 = sorted({d['row'] for d in U.values() if d['role'] == 'd20'})
print("rows", len(rows), "d20", len(d20))
over = [k for k, d in U.items() if d['status'] != 'done']
print("not done (over budget / error):", over)
f = lambda d, k: float.fromhex(d['result'][k]['hex'])
# S3
s3f = []; maxd = 0
for r in rows:
    d = U.get(f"s-1e-7-{r}")
    if d is None or d['status'] != 'done': s3f.append((r, 'over budget/missing')); continue
    md = max((c['max_depth'] for c in d['calls']), default=0); maxd = max(maxd, md)
    if md >= 18 or any(c.get('n_depth_ge_18') for c in d['calls']): s3f.append((r, md))
print("rows with no Levin calls under s@1e-7:", [r for r in rows if not U[f"s-1e-7-{r}"]["calls"]]); print("S3: rows", len(rows), "false", s3f, "largest depth", maxd, "-> holds" if not s3f else "-> fails")
# S4
s4 = []
for r in rows:
    a, b = U[f"t5-1e-7-{r}"], U[f"s-1e-7-{r}"]
    dlt = abs(f(b, 'total') - f(a, 'total')); e = f(a, 'total_abserr')
    if not dlt <= e: s4.append((r, dlt / e if e else math.inf))
print("S4: true", len(rows) - len(s4), "of", len(rows), "false", [(r, round(x, 3)) for r, x in s4])
# S5
s5 = []; n5 = 0
for r in d20:
    a, b = U.get(f"s-1e-7-{r}"), U.get(f"s-1e-8-{r}")
    if not a or not b or a['status'] != 'done' or b['status'] != 'done': continue
    n5 += 1
    dlt = abs(f(a, 'total') - f(b, 'total')); e = f(a, 'total_abserr')
    if not dlt <= e: s5.append((r, dlt / e if e else math.inf))
print("S5: rows with both s in budget", n5, "true", n5 - len(s5), "false", [(r, round(x, 3)) for r, x in s5])
