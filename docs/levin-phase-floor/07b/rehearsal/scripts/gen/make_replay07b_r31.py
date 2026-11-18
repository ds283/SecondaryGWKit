"""Generate scripts/capture/replay07b_r31.py from R30's replay07a_r30.py (c7ff4d85): the store tags shifted
(R29 -> R30, R30 -> R31 in every name, key and label, outside the comment tags "prompt 07a r30", "route_07a" and
"changed_vs_r28"), then asserted single replacements, each marked "prompt 07b r31" in the output, and two new
subcommands. levin-phase-floor 07b, rehearsal agent. Run once: python -I -B make_replay07b_r31.py"""
import re

D = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/rehearsal/scripts/capture/"
s = open(D + "replay07a_r30.py").read()
doc_end = s.index('"""', 3) + 3
doc, body = s[:doc_end], s[doc_end:]
prot = {"@@P1@@": "prompt 07a r30", "@@P2@@": "route_07a", "@@P3@@": "changed_vs_r28"}
for k, v in prot.items():
    body = body.replace(v, k)
n = [0]


def f(m):
    n[0] += 1
    return {"29": "30", "30": "31"}[m.group(1)]


body = re.sub(r'(?<=[A-Za-z_])(29|30)(?![0-9])', f, body)
print("tag shifts", n[0])
for k, v in prot.items():
    body = body.replace(k, v)
reps = [
    ('REH = MAIN / "var" / "levin-phase-floor-07a" / "rehearsal"  # prompt 07a r30\n',
     'REH = MAIN / "var" / "levin-phase-floor-07b" / "rehearsal"  # prompt 07b r31\n'),
    ('STORES = {"h": MAIN / "var" / "datastores" / "levin-phase-floor-07a-r31.sqlite"}  # prompt 07a r30\n',
     'STORES = {"h": MAIN / "var" / "datastores" / "levin-phase-floor-07b-r31.sqlite"}  # prompt 07b r31\n'),
    ('TAG = {"h": "R31"}  # prompt 07a r30\n', 'TAG = {"h": "R31"}  # prompt 07b r31\n'),
    ('R30_STORE = MAIN / "var" / "datastores" / "levin-phase-floor-07-r30.sqlite"  # prompt 07a r30: for the record only\n',
     'R30_STORE = MAIN / "var" / "datastores" / "levin-phase-floor-07a-r30.sqlite"  # prompt 07b r31: R30, the parent rehearsal, read for the record\n'),
    ('R30_REH = MAIN / "var" / "levin-phase-floor-07" / "rehearsal" / "r30"  # prompt 07a r30: R30\'s block captures (read only)\n',
     'R30_REH = MAIN / "var" / "levin-phase-floor-07a" / "rehearsal"  # prompt 07b r31: R30\'s block captures (read only)\n'),
    ('W = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-4bf50d3")  # prompt 07a r30\n',
     'W = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5")  # prompt 07b r31\n'),
    ('    ins = [v1_rows, rec_csv] + sorted(STORES["h"].parent.glob("levin-phase-floor-07a-r31*.sqlite")) + sorted(R30_STORE.parent.glob("levin-phase-floor-07-r30*.sqlite"))\n',
     '    ins = [v1_rows, rec_csv] + sorted(STORES["h"].parent.glob("levin-phase-floor-07b-r31*.sqlite")) + sorted(R30_STORE.parent.glob("levin-phase-floor-07a-r30*.sqlite"))  # prompt 07b r31\n'),
    ('"prompt 07a r30: the rows to replay and the four-block rows"',
     '"prompt 07b r31: the rows to replay, the four-block rows and the records of interest"'),
    ('''    # prompt 07a r30: the 49 group records whose abserr moved against R30 with value, regions and location unchanged
    changed = set()
    with open(rec_csv) as f:
        for r_ in csv.DictReader(f):
            fs = set(r_["fields_differing"].split())
            if (r_["kind"] == "group" and "value" not in fs and "abserr" in fs
                    and r_["regions_30"] == r_["regions_31"] and r_["stationary_points_30"] == "0" and r_["stationary_points_31"] == "0"):
                changed.add(((float(r_["k"]), float(r_["q"]), float(r_["r"]), float(r_["z_response"])), r_["record"]))
''', '''    # prompt 07b r31: the records of interest (changed_vs_r28 read as "a record of interest"): every Levin record whose
    # |value_31 - value_30| exceeds abserr_30 + abserr_31 (exactly, in Fractions of the stored doubles), and every Levin
    # record whose converged changed (out/qsi_records_r30_r31.csv)
    from fractions import Fraction as _F
    changed = set()
    with open(rec_csv) as f:
        for r_ in csv.DictReader(f):
            k4 = (float(r_["k"]), float(r_["q"]), float(r_["r"]), float(r_["z_response"]))
            if r_["converged_30"] != r_["converged_31"]:
                changed.add((k4, r_["record"]))
            if r_["value_30"] not in ("", "None") and r_["value_31"] not in ("", "None"):
                v0, v1 = _F(float.fromhex(r_["value_30"])), _F(float.fromhex(r_["value_31"]))
                if abs(v1 - v0) > _F(float.fromhex(r_["abserr_30"])) + _F(float.fromhex(r_["abserr_31"])):
                    changed.add((k4, r_["record"]))
'''),
    ('# prompt 07a r30: changed_vs_r28 read as "holds one of the 49 abserr-only records"',
     '# prompt 07b r31: changed_vs_r28 read as "holds a record of interest"'),
    ('print(f"abserr-only records (R31 vs R30) {len(changed)}', 'print(f"records of interest (R31 vs R30) {len(changed)}'),
    ('f"prompt 07a r30: run {spec[\'name\']}"', 'f"prompt 07b r31: run {spec[\'name\']}"'),
    ('"prompt 07a r30: the replay checked against R31, record by record"',
     '"prompt 07b r31: the replay checked against R31, record by record"'),
    ('''                           "abserr_only_vs_r30": bool(r.get("changed_vs_r28")),  # prompt 07a r30
                           "route_07a": (c or {}).get("route_07a")})  # prompt 07a r30
''', '''                           "interest_07b": bool(r.get("changed_vs_r28")),  # prompt 07b r31
                           "route_07a": (c or {}).get("route_07a"),  # prompt 07a r30
                           "index": (c or {}).get("index"), "notify_label": (c or {}).get("notify_label"),  # prompt 07b r31
                           "refusals_07b": (c or {}).get("refusals_07b")})  # prompt 07b r31
'''),
    ('# prompt 07a r30: the route diagnostic, over every replayed call, and on the 49 abserr-only records',
     '# prompt 07a r30: the route diagnostic, over every replayed call (prompt 07b r31: kept, descriptive only)'),
    ('''    ao = [c for c in checks if c["abserr_only_vs_r30"]]
    print(f"the abserr-only records: {len(ao)}; reproduced {sum(c['reproduces'] for c in ao)}")
    for c in ao:
        rt = c["route_07a"] or {}
        print(f"  {c['key']} {c['path']} {c['label']}: reproduces {c['reproduces']}; solves {rt.get('n_solves')}, route differs {rt.get('n_route_differs')}, "
              f"LU differs {rt.get('n_lu_differs')}; classes {rt.get('counts')}")
''', '''    # prompt 07b r31: the refusals (prompt 07b section 1.4) over every replayed call, by kind, and in the records of interest
    ref = Counter()
    ncalls = Counter()
    for e in rows:
        if not (e["touched"] or e["p1"] or e["wid"]):
            continue
        u, _, _ = units[f"h-1e-7-{e['R31']['shard']}-{e['R31']['serial']}"]
        for kind_, cl in (("group", u.get("calls") or []), ("other", u.get("other_calls") or [])):
            for c in cl:
                rf = c.get("refusals_07b") or {}
                ncalls[kind_] += 1
                if rf.get("num_edge_rule_refusals") is None:
                    ref[(kind_, "<no count>")] += 1
                    continue
                ref[(kind_, "edge")] += rf["num_edge_rule_refusals"]
                ref[(kind_, "range")] += rf["num_range_rule_refusals"]
                ref[(kind_, "calls acted")] += (rf["num_edge_rule_refusals"] + rf["num_range_rule_refusals"]) > 0
                ref[(kind_, "calls with an edge refusal")] += rf["num_edge_rule_refusals"] > 0
    print(f"refusals over the replayed calls ({dict(ncalls)}): {dict(sorted(ref.items()))}")
    ao = [c for c in checks if c["interest_07b"]]
    print(f"the records of interest: {len(ao)}; reproduced {sum(c['reproduces'] for c in ao)}")
    for c in ao:
        print(f"  {c['key']} {c['path']} {c['label']}: reproduces {c['reproduces']}; call index {c['index']}; refusals {c['refusals_07b']}")
'''),
    ('"prompt 07a r30: the four blocks at h, checked against R31, and each capture against R30\'s"',
     '"prompt 07b r31: the four blocks at h, checked against R31, and each capture against R30\'s"'),
    ('''                    # prompt 07a r30: R30's capture of the same block (its g pickle), compared byte for byte
                    p30 = R30_REH / "state" / "inputs" / f"{blk}-g-{e['R30']['shard']}-{e['R30']['serial']}.pkl"
''', '''                    # prompt 07b r31: R30's capture of the same block (its h pickle), compared byte for byte
                    p30 = R30_REH / "state" / "inputs" / f"{blk}-h-{e['R30']['shard']}-{e['R30']['serial']}.pkl"
'''),
]
for a, b in reps:
    assert body.count(a) == 1, (a[:100], body.count(a))
    body = body.replace(a, b)
extra = open(D + "../gen/replay07b_r31_extra.py.txt").read()
main_old = '''    elif cmd == "blockcheck":
        sys.exit(cmd_blockcheck(sys.argv[2], sys.argv[3]))
'''
main_new = main_old + '''    elif cmd == "refcapspec":  # prompt 07b r31
        cmd_spec_refcap(sys.argv[2], sys.argv[3])
    elif cmd == "refcapcheck":  # prompt 07b r31
        sys.exit(cmd_refcapcheck(sys.argv[2], sys.argv[3]))
'''
assert body.count(main_old) == 1
body = body.replace(main_old, main_new)
i = body.index('\n\nif __name__ == "__main__":')
body = body[:i] + "\n\n" + extra.rstrip("\n") + body[i:]
newdoc = '''"""[prompt 07b section 5 rehearsal agent, R31: R30's replay07a_r30.py (c7ff4d85), changed where marked "prompt 07b r31"
(and, throughout, the store tags shifted: R29 -> R30, R30 -> R31, r29 -> r30, r30 -> r31, in names, keys and labels; by
scripts/gen/make_replay07b_r31.py): one tree h (0a103b5a, code 5e19c4b6) reading R31 (R30 read for the record); the
rows: every R31 row with a nonzero count, P1's two rows, the w rows, the two block rows, and the rows holding a record
of interest (out/qsi_records_r30_r31.csv: a Levin record whose |value_31 - value_30| exceeds abserr_30 + abserr_31,
exactly, or whose converged changed); the check also reports each call's refusals (worker.py's refusals_07b) and sums
them over the replayed calls; the blockcheck compares each capture's blob with R30's (state/inputs/ of R30's records);
new subcommands ``refcapspec ROWS OUT`` and ``refcapcheck ROWS OUT`` capture the group calls of the records of interest
and check each capture against R31.]

'''
open(D + "replay07b_r31.py", "w").write(newdoc + doc.replace('"""', '', 1) + body)
print("written")
