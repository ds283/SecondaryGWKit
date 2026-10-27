"""levin-phase-floor 06 §5.4: further counts from tabulate_qsi.py's CSV (reads only the CSV).
Rows with total_abserr inf; the rows with |dtotal|/abserr_A > 1 in full; rows whose depth changed.

    python3 -I qsi_extra_counts.py <qsi_rows.csv>
"""
import csv, hashlib, sys
from pathlib import Path
me = Path(__file__).resolve()
print(f"# {me.name} sha256 {hashlib.sha256(me.read_bytes()).hexdigest()}")
print(f"# input {sys.argv[1]} sha256 {hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()}")
rows = list(csv.DictReader(open(sys.argv[1])))
inf = [r for r in rows if r["abserr_A"] == "inf" or r["abserr_B"] == "inf"]
print(f"rows with total_abserr inf (A or B): {len(inf)}; A inf {sum(r['abserr_A']=='inf' for r in rows)}, "
      f"B inf {sum(r['abserr_B']=='inf' for r in rows)}; of them total differs: "
      f"{sum(r['total_bit_identical']=='False' for r in inf)}")
print("rows with |dtotal|/abserr_A > 1 or /abserr_B > 1:")
for r in rows:
    a = float(r["dtotal_over_abserr_A"] or 0); b = float(r["dtotal_over_abserr_B"] or 0)
    if a > 1 or b > 1:
        print(f"  k={float(r['k']):.6g} q={float(r['q']):.6g} r={float(r['r']):.6g} z_resp={float(r['z_response']):.6g}: "
              f"total A {float.fromhex(r['total_A']):.10e} B {float.fromhex(r['total_B']):.10e}, rel {float(r['rel_dtotal']):.3e}, "
              f"abserr A {float(r['abserr_A']):.3e} B {float(r['abserr_B']):.3e}, /abserr_A {a:.3e} /abserr_B {b:.3e}, "
              f"depth {r['max_depth_A']}->{r['max_depth_B']}, regions {r['regions_A']}->{r['regions_B']}, "
              f"converged {r['converged_A']}->{r['converged_B']}, phase_limited {r['phase_limited_A']}->{r['phase_limited_B']}")
print("rows whose max depth changed:")
for r in rows:
    if r["max_depth_A"] != r["max_depth_B"]:
        print(f"  k={float(r['k']):.6g} q={float(r['q']):.6g} r={float(r['r']):.6g} z_resp={float(r['z_response']):.6g}: "
              f"depth {r['max_depth_A']}->{r['max_depth_B']}, regions {r['regions_A']}->{r['regions_B']}, "
              f"/abserr_A {float(r['dtotal_over_abserr_A'] or 0):.3e}, phase_limited {r['phase_limited_A']}->{r['phase_limited_B']}")
pl = [r for r in rows if r["phase_limited_A"] != r["phase_limited_B"]]
print(f"rows whose phase_limited changed: {len(pl)}")
for r in pl:
    print(f"  k={float(r['k']):.6g} q={float(r['q']):.6g} r={float(r['r']):.6g} z_resp={float(r['z_response']):.6g}: "
          f"{r['phase_limited_A']}->{r['phase_limited_B']}, depth {r['max_depth_A']}->{r['max_depth_B']}, /abserr_A {float(r['dtotal_over_abserr_A'] or 0):.3e}")
