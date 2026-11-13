"""Orchestrator's own reading of S4 (07a Part C §C.2): R30's four O9 blocks (analytic 2pt5 J4 and Y4 at the two rows
R29 guarded, identified by k = q = r and z_response as the rehearsal agent's block_scores.txt names them, and
asserted guarded in R29), read from R30's store (sqlite mode=ro, by my stop-1 reader's row keying), against R29's block references
(docs/levin-phase-floor/07/rehearsal/r29/out/block_reference/{J4,Y4}-g-{0-157,1-10283}.json, field `reference`).
Also the same blocks in R29's store. The store holds each block's abserr, regions and counts, not its value: the value is taken from the rehearsal agent's
R30 replay (rehearsal/out/block_reference/{J4,Y4}-h-*.json), accepted only if that replay's abserr, regions, located and
guarded counts and converged equal the store's block bit for bit and its x_span equals the reference's.
Holds iff each R30 block locates a root (stationary_points >= 1), guards no leaf
(guarded_regions 0, guard_at_depth_max 0), and |v - ref| <= abserr. Exact Fractions. Read only; prints only.
usage: python -I -B s4_check.py"""
import hashlib, json, sys
from fractions import Fraction
from pathlib import Path
sys.path.insert(0, "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/review/stop1/scripts")
from qsi_compare import read, sha  # my stop-1 reader
REF = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3/docs/levin-phase-floor/07/rehearsal/r29/out/block_reference")
HREP = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/rehearsal/out/block_reference")
HTAG = {"0-157": "0-10129", "1-10283": "1-20022"}
DS = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/datastores")
print("script", sha(__file__))
stores = {"R29": DS / "levin-phase-floor-07-r29.sqlite", "R30": DS / "levin-phase-floor-07a-r30.sqlite"}
rows = {n: read(str(p)) for n, p in stores.items()}
ok_all = True
for tag in ("0-157", "1-10283"):
    for blk in ("J4", "Y4"):
        f = REF / f"{blk}-g-{tag}.json"
        r = json.load(open(f)); print("\ninput", sha(f), f.name, r["unit"], r["notify_label"])
        ref = Fraction(float.fromhex(r["reference"]["reference"]))
        cap = r["capture_record"]
        KZ = {"0-157": (316227.7660168379, 1329131612.3012946), "1-10283": (1000000.0, 4017630440.8276954)}[tag]
        for n in ("R29", "R30"):
            hits = [(key, md["analytic"]["2pt5"]["Levin"][blk + "_data"]) for key, (fl, md, *_r) in rows[n].items()
                    if float.fromhex(key[0]) == float.fromhex(key[1]) == float.fromhex(key[2]) == KZ[0]
                    and float.fromhex(key[3]) == KZ[1] and blk + "_data" in md["analytic"].get("2pt5", {}).get("Levin", {})]
            assert len(hits) == 1, (n, blk, tag, len(hits))
            key, b = hits[0]
            if n == "R29":
                assert b.get("guarded_regions", 0) >= 1, ("R29 block not guarded", blk, tag)
            e = Fraction(b["abserr"])
            loc, gr, gd = b.get("stationary_points", 0), b.get("guarded_regions", 0), b.get("guard_at_depth_max", 0)
            line = (f"  {n}: k={float.fromhex(key[0]):.10g} z={float.fromhex(key[3]):.10g} abserr {b['abserr'].hex()} "
                    f"regions {b.get('regions')} converged {b.get('converged')} located {loc} window_pieces {b.get('window_pieces', 0)} "
                    f"guarded {gr} at_depth_max {gd}")
            if n == "R30":
                hp = HREP / f"{blk}-h-{HTAG[tag]}.json"; h = json.load(open(hp)); rp = h["replay"]
                same = (rp["abserr"] == b["abserr"].hex() and rp["num_regions"] == b.get("regions")
                        and rp["stationary_points"] == loc and rp["num_guarded_regions"] == gr
                        and rp["converged"] == b.get("converged") and h["x_span"] == r["x_span"])
                roots_ok = sorted(pt["x"] for pt in rp["points"] if pt["kind"] == "root") == sorted(r["roots"])
                ratio = abs(Fraction(float.fromhex(rp["value"])) - ref) / e
                ok = same and roots_ok and loc >= 1 and gr == 0 and gd == 0 and ratio <= 1
                ok_all &= ok
                line += (f"\n        replay {hp.name} sha {sha(hp)[:12]}: matches the store's block {same}; root(s) equal R29 reference's "
                         f"{roots_ok}; value {rp['value']} |v-ref|/abserr {float(ratio):.6g} -> {'OK' if ok else 'FAILS'}")
            print(line)

print("\nS4", "HOLDS" if ok_all else "FAILS")
