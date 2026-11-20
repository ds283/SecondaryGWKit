"""Orchestrator's stop-3 supplement: the Y8 ids whose recorded 07b (value, abserr) matches more than one call
in its 07b unit. For each, the candidate keys' 07b five-tuples and 07c five-tuples are compared, and the replay is
compared with the 07c call at every candidate. usage: stop3_y8_ambiguous.py <stop3 dir> <(S) replay dir>"""
import collections, hashlib, json, os, sys
print("script", hashlib.sha256(open(__file__, "rb").read()).hexdigest())
D, S = sys.argv[1:3]
rep = json.load(open(os.path.join(D, "stop3_y1_y8.json")))
amb = [f[0] for f in rep["y8_failures"] if f[1] == "07b match count"]
units = set()
reps = {}
for f in amb:
    r = json.load(open(os.path.join(S, f))); reps[f] = r; units.add(r["census_unit"])
def load(p):
    out = collections.defaultdict(dict)
    for line in open(p):
        c = json.loads(line)
        if c["unit"] in units:
            out[c["unit"]][(c["kind"], c["index"])] = (c["value"], c["abserr"], c["converged"], c["num_regions"], c["max_depth"], c.get("num_chase_stops"))
    return out
A, B = load(os.path.join(D, "calls_07b.jsonl")), load(os.path.join(D, "calls_07c.jsonl"))
tally = collections.Counter(); calls = set()
for f, r in sorted(reps.items()):
    u = r["census_unit"]; cen = r["census"]; rp = r["replay"]
    cand = sorted(k for k, t in A[u].items() if t[0] == cen["value"] and t[1] == cen["abserr"])
    a5 = {tuple(A[u][k][:5]) for k in cand}
    b5 = {tuple(B[u][k][:5]) for k in cand}
    rp5 = (rp["value"], rp["abserr"], rp["converged"], rp["num_regions"], rp["max_depth"])
    cen5 = (cen["value"], cen["abserr"], cen["converged"], cen["num_regions"], cen["max_depth"])
    eq = [k for k in cand if tuple(B[u][k][:5]) == rp5]
    status = ("07b candidates identical" if len(a5) == 1 else "07b candidates differ",
              "07c candidates identical" if len(b5) == 1 else "07c candidates differ",
              "07b equals census field" if a5 == {cen5} else "07b != census field",
              f"replay equals 07c at {len(eq)} of {len(cand)}")
    tally[status] += 1
    for k in cand: calls.add((u, k))
    print(f, u, [list(k) for k in cand], status, "stops", [B[u][k][5] for k in cand])
print("tally", dict(tally)); print("distinct candidate calls", len(calls))
