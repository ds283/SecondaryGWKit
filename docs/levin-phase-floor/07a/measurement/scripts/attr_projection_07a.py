"""levin-phase-floor prompt 07a Part A, section 2.2's cost rule: the attribution's cost projected from the pilot on one
call (v22) before the rest runs. New in 07a. Read only.

The pilot's wall at its worker count is scaled to six workers (wall x workers / 6) and divided by 07's own FCC
r1 + r2 walls for the same call (six workers; 07's out/v4/v4_scores.json ``wall_s``), giving the ratio of this
method's cost to 07's readings'. Each call to attribute (V4's calls with a stopped window: 07a's replays at a) is
projected as that ratio times 07's FCC r1 + r2 wall (z12: 06g's FCC readings, scores_stage3/scores.json). The rule:
a projection over 12 h is reported before the rest runs, with the calls it would drop to fit.
Usage: python -B attr_projection_07a.py OUT_JSON"""
import ast, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-f93aafd")
V4 = TREE / "docs/levin-phase-floor/07/census/out/v4/v4_scores.json"
S06G = TREE / "docs/levin-phase-floor/06g/references/out/scores_stage3/scores.json"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out = Path(sys.argv[1])
    pilot = json.loads((ROOT / "out/attr/pilot_v22.json").read_text())
    w = pilot["header"]["workers"]
    pilot_wall = sum(r["wall_s"] for r in pilot["readings"].values())
    pilot6 = pilot_wall * w / 6
    sc = {c["sid"]: c for c in json.loads(V4.read_text())["calls"]}
    w07 = {s: c["wall_s"]["fcc_r1"] + c["wall_s"]["fcc_r2"] for s, c in sc.items()}
    z = [r for r in json.loads(S06G.read_text())["rows"] if r["sid"] == "z12"][0]
    rd = ast.literal_eval(z["readings"]) if isinstance(z["readings"], str) else z["readings"]
    w07["cal_z12"] = rd["fcc_r1"]["wall_s"] + rd["fcc_r2"]["wall_s"]
    ratio = pilot6 / w07["v22"]
    stopped = {}
    for p in sorted((ROOT / "out/replays/a").glob("*.json")):
        if p.name.endswith(".instrument.json"): continue
        d = json.loads(p.read_text())
        if d["set"] == "v4": stopped[d["sid"]] = d["n_windows_stopped"]
    todo = sorted(s for s, n in stopped.items() if n > 0 and s != "v22")
    proj = {s: ratio * w07[s] for s in todo}
    total = sum(proj.values())
    doc = {"script": str(Path(__file__).resolve()), "script_sha256": sha(__file__), "pilot": "v22",
           "pilot_json_sha256": sha(ROOT / "out/attr/pilot_v22.json"), "pilot_workers": w, "pilot_wall_s": pilot_wall,
           "pilot_wall_at_6_s": pilot6, "w07_v22_fcc_r1_r2_s": w07["v22"], "ratio": ratio,
           "calls_without_stopped_window": sorted(s for s, n in stopped.items() if n == 0),
           "projected_s": proj, "projected_total_s": total, "projected_total_h": total / 3600,
           "over_12h": total > 12 * 3600, "sources": {str(V4): sha(V4), str(S06G): sha(S06G)}}
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "projected_s"}, indent=1))
    for s in sorted(proj, key=proj.get): print(f"{s} {proj[s]:.0f} s")
main()
