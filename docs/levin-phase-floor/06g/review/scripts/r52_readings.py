"""Orchestrator's reader: r52's readings against g's and 06h's census values (read-only)."""
import json, os, itertools

R = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-06g/references/out/"
c = [x for x in json.load(open(R + "capture.json")) if x["sid"] == "r52"][0]
vg, eg = float.fromhex(c["g_record"]["value"]["hex"]), float.fromhex(c["g_record"]["abserr"]["hex"])
vh, eh = float.fromhex(c["06h_record"]["value"]["hex"]), float.fromhex(c["06h_record"]["abserr"]["hex"])
tol = min(eg, eh) / 100
print(f"r52 {c['g_unit']} #{c['index']} {c['notify_label']}")
print(f"  g   value {vg!r} ({vg.hex()}) abserr {eg:.4e} converged {c['g_record']['converged']}")
print(f"  06h value {vh!r} ({vh.hex()}) abserr {eh:.4e} converged {c['06h_record']['converged']}")
print(f"  agreement tolerance 1/100 min(abserr) = {tol:.4e}")
rd = {}
for impl, res in itertools.product(("qawo", "fcc"), (1, 2)):
    f = f"{R}readings/r52-{impl}-r{res}.json"
    if os.path.exists(f):
        d = json.load(open(f))
        rd[f"{impl} r{res}"] = float.fromhex(d["value_hex"])
        print(f"  {impl} r{res}: {rd[f'{impl} r{res}']!r} ({d['value_hex']}) wall {d['wall_s']:.0f} s"
              f"  (v-g)/e_g {(rd[f'{impl} r{res}'] - vg) / eg:+.4f}  (v-06h)/e_06h {(rd[f'{impl} r{res}'] - vh) / eh:+.4f}")
    else:
        print(f"  {impl} r{res}: no value (stopped or not yet run)")
ks = list(rd)
for a, b in itertools.combinations(ks, 2):
    d = abs(rd[a] - rd[b])
    print(f"  |{a} - {b}| = {d:.4e} = {d / tol:.3e} x tolerance = {d / eg:.3e} e_g")
if rd:
    sp = max(rd.values()) - min(rd.values())
    print(f"  spread over {len(rd)} readings: {sp:.4e} = {sp / tol:.3e} x tolerance")
    ref = sum(rd.values()) / len(rd)
    print(f"  mean of readings {ref!r}: |g - ref|/e_g {abs(vg - ref) / eg:.4f}; |06h - ref|/e_06h {abs(vh - ref) / eh:.4f}")
