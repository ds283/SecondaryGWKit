"""levin-phase-floor 06g, descriptive ("the values"): every matched call outside the 720 whose value moved by more
than abserr_06h + abserr_g (the call-level form of V2's rule, not scored), per rtol, by kind and group, with both
abserrs, regions and depth; the 40 largest listed. Read only. New in 06g.
Usage: python -B values_beyond06g.py RUN_DIR"""
import collections, json, sys
from fractions import Fraction
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse06g import load, match, scan06a_sp, H  # noqa: E402
from prov import RUN06H  # noqa: E402
run = Path(sys.argv[1])
sp = scan06a_sp()
G, Hh = load(run), load(RUN06H)
out = []
for (rt, rk), ug in G.items():
    if rk in sp:
        continue
    for cg, ch in match(ug, Hh[(rt, rk)])[0]:
        eg, eh = H(cg["abserr"]), H(ch["abserr"])
        if eg is None or eh is None or eg != eg or eh != eh or eg == float("inf") or eh == float("inf"):
            continue
        d = abs(Fraction(H(cg["value"])) - Fraction(H(ch["value"])))
        if d > Fraction(eg) + Fraction(eh):
            out.append((float(d / (Fraction(eg) + Fraction(eh))), rt, rk, cg, ch, float(d)))
out.sort(key=lambda t: -t[0])
for rt in ("1e-7", "1e-8"):
    L = [x for x in out if x[1] == rt]
    print(f"== {rt}: calls outside the 720 with |v_g - v_06h| > abserr_06h + abserr_g: {len(L)} in {len({x[2] for x in L})} rows; "
          f"by kind {dict(collections.Counter(x[3]['kind'] for x in L))}; by group {dict(collections.Counter(x[3]['group'] or x[3]['notify_label'] for x in L).most_common(12))}")
    print(f"   same region count under both: {sum(1 for x in L if x[3]['num_regions'] == x[4]['num_regions'])}; g converged {sum(1 for x in L if x[3]['converged'])}, 06h converged {sum(1 for x in L if x[4]['converged'])}")
    for r_, _, rk, cg, ch, d in L[:40]:
        print(f"   {rk} {cg['kind']} {cg['index']} {cg['group'] or cg['notify_label']}: ratio {r_:.3g}; |dv| {d:.3e}; abserr 06h {H(ch['abserr']):.3e} g {H(cg['abserr']):.3e}; "
              f"regions {ch['num_regions']} -> {cg['num_regions']}; depth {ch['max_depth']} -> {cg['max_depth']}; conv {ch['converged']} -> {cg['converged']}")
