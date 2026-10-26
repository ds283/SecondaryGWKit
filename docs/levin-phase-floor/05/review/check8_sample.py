"""Review check 8 (scratch): redraw prompt 05 section 6's rows from groups_01.csv per state/sample.md."""
import csv, json, sys
CSV = "var/levin-phase-floor-01/out/groups_01.csv"
lines = [l for l in open(CSV) if not l.startswith("#")]
recs = list(csv.DictReader(lines))
rows = {}
for r in recs:
    key = (int(r["shard"]), int(r["serial"]))
    row = rows.setdefault(key, {"k": r["k"], "band": r["band"], "depth": r["WKB_Levin_max_depth"], "ct": float(r["compute_time"]), "groups": []})
    row["groups"].append(r)
def top(row):
    best = None
    for g in row["groups"]:  # first in file order wins a tie
        if best is None or float(g["elapsed"]) > float(best["elapsed"]):
            best = g
    return best
def cls(row):
    t = top(row)
    if t["label"] == "G+r": return "G+r"
    if t["label"] == "G-r": return "G-r"
    if t["label"] == "G": return "G"
    if t["regime"] == "Gqr": return "Gqr"
    return None
drawn = []
for band in ("low", "mid"):
    for c in ("G+r", "G-r", "G", "Gqr"):
        cand = sorted([(row["ct"], key) for key, row in rows.items() if row["band"] == band and cls(row) == c and row["depth"] == "20" and row["ct"] <= 900])
        ks, taken = set(), []
        for ct, key in cand:
            if rows[key]["k"] in ks: continue
            ks.add(rows[key]["k"]); taken.append(key)
            if len(taken) == 3: break
        print(band, c, len(cand), "candidates; taken", taken)
        drawn += taken
for band in ("low", "mid", "seam"):
    cand = sorted([(row["ct"], key) for key, row in rows.items() if row["band"] == band and row["depth"] not in ("", "20", "None") and int(row["depth"]) < 20 and row["ct"] <= 60 and any("G" in g["label"] for g in row["groups"])])
    ks, taken = set(), []
    for ct, key in cand:
        if rows[key]["k"] in ks: continue
        ks.add(rows[key]["k"]); taken.append(key)
        if len(taken) == 2: break
    print("ctrl", band, len(cand), "candidates; taken", taken)
    drawn += taken
s = json.load(open("var/levin-phase-floor-05/wider/state/sample.json"))
agent = [(r["shard"], r["serial"]) for r in s["rows"] if r.get("role") == "d20" and not r["name"].startswith("R")] + \
        [(r["shard"], r["serial"]) for r in s["rows"] if r.get("role") == "control" and "NULL" not in r["name"]]
print("agent's drawn (cells + band controls):", agent)
print("MATCH" if agent == drawn else "MISMATCH")
