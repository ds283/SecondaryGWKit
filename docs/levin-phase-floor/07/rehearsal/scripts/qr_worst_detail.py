"""The worst q-r call (1e5, 1e6, 1e6, z 6.985e9): R28's value against each of the reference's readings (GL at
m = 6..14 and QAGS), in multiples of R28's declared abserr. levin-phase-floor 07, rehearsal agent; written at
the orchestrator's request to state on what R28's verdict there rests. Reads out/qr_reference/qmr-g-3-182.json.
"""
import hashlib, json
from pathlib import Path
import mpmath as mp
mp.mp.dps = 40
R = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07/rehearsal")
f = R / "out" / "qr_reference" / "qmr-g-3-182.json"
print(f"# script {Path(__file__).resolve()} sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}")
print(f"# input {f} sha256 {hashlib.sha256(f.read_bytes()).hexdigest()}")
d = json.loads(f.read_text())
v = mp.mpf(float.fromhex(d["capture_record"]["value"])); e = mp.mpf(float.fromhex(d["capture_record"]["abserr"]))
print(f"R28 value {d['capture_record']['value']} abserr {d['capture_record']['abserr']} = {mp.nstr(e, 6)}")
ref = d["reference"]
for m, g in sorted(ref["GL"].items(), key=lambda t: int(t[0])):
    x = mp.mpf(float.fromhex(g["value"]))
    print(f"GL m={m} ({g['evaluations']} evaluations): {g['value']}; v28 - GL = {mp.nstr(v - x, 6)} = {mp.nstr(abs(v - x) / e, 4)} x abserr28")
q = mp.mpf(float.fromhex(ref["QAGS"]["value"]))
print(f"QAGS {ref['QAGS']['value']} (own abserr {ref['QAGS']['abserr']}, neval {ref['QAGS']['neval']}, intervals {ref['QAGS']['last']}); "
      f"v28 - QAGS = {mp.nstr(v - q, 6)} = {mp.nstr(abs(v - q) / e, 4)} x abserr28; QAGS - GL14 = {mp.nstr(q - mp.mpf(float.fromhex(ref['reference'])), 6)} "
      f"= {mp.nstr(abs(q - mp.mpf(float.fromhex(ref['reference']))) / mp.mpf(ref['QAGS']['abserr']), 4)} x QAGS's own abserr")
print(f"stated error {ref['stated_error']} = {mp.nstr(mp.mpf(ref['stated_error']) / e, 4)} x abserr28; rounding scale {ref['rounding_scale_eps_abs_sum_m14']}")
