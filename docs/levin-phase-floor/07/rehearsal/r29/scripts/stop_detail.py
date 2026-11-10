"""The four records that break the R29-vs-R28 rule: their rows' analytic oracle (analytic_rad, analytic.abserr,
analytic.converged) and the two blocks' records in R27, R28 and R29, read with sqlite mode=ro. levin-phase-floor 07,
rehearsal agent, R29. New. Uses tabulate_qsi_guard.py's reader (copied unchanged, a1eaba31)."""
import sys
from pathlib import Path
import mpmath as mp
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tabulate_qsi_guard as T
mp.mp.dps = 30
print(f"# script {Path(__file__).resolve()} sha256 {T.sha256(__file__)}")
D = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores")
S = {t: T.read(D / f"levin-phase-floor-07-{t}.sqlite")[0] for t in ("r27", "r28", "r29")}
keys = [k for k in S["r29"] if (abs(k[0] - 10**5.5) < 1e-6 * k[0] and abs(k[3] - 1329131612.3012946) < 1) or (k[0] == 1e6 and k[1] == 1e6 and abs(k[3] - 4017630440.8276954) < 1)]
keys = [k for k in keys if k[0] == k[1] == k[2]]
for k in keys:
    print(f"\nrow {k}")
    for t in ("r27", "r28", "r29"):
        rec = S[t][k]; _, m, _ = T.fields(rec)
        an = m["analytic"]
        print(f"  {t}: analytic_rad {T.hx(rec['analytic_rad'])} analytic.abserr {T.hx(an['abserr'])} converged {an['converged']}; total {T.hx(rec['total'])}")
        for blk in ("J4_data", "Y4_data"):
            b = an["2pt5"]["Levin"][blk]
            print(f"      2pt5 {blk}: abserr {T.hx(b['abserr'])} regions {b['regions']} converged {b['converged']} phase_limited {b['phase_limited']} counts {[b.get(g) for g in T.GUARD_LEAVES]}")
    a27, a29 = S["r27"][k], S["r29"][k]
    e27 = T.fields(a27)[1]["analytic"]["abserr"]; e29 = T.fields(a29)[1]["analytic"]["abserr"]
    d = abs(mp.mpf(a29["analytic_rad"]) - mp.mpf(a27["analytic_rad"]))
    print(f"  |rad29 - rad27| = {mp.nstr(d, 6)} = {mp.nstr(d / mp.mpf(e27), 4)} x abserr27 = {mp.nstr(d / mp.mpf(e29), 4)} x abserr29")
