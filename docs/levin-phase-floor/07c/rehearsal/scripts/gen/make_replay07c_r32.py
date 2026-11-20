"""Generate scripts/capture/replay07c_r32.py from 07b's rehearsal replay07b_r31.py (ac7319ca): the store tags shifted
(R30 -> R31, R31 -> R32 in every name, key and label, outside the protected strings "prompt 07b r31", "prompt 07a r30",
"route_07a", "changed_vs_r28", "refusals_07b", "interest_07b", "replay07b_r31"), then asserted single replacements, each
marked "prompt 07c r32" in the output: the paths (this rehearsal, R32's store, R31 as the parent store and R31's block
captures, the 07c worktree), and each call's chase_07c (worker.py) carried into the checks and summed over the replayed
calls. levin-phase-floor 07c, rehearsal agent. Run once: python -I -B make_replay07c_r32.py"""
import hashlib
import re

D = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07c/rehearsal/scripts/capture/"
raw = open(D + "replay07b_r31.py", "rb").read()
assert hashlib.sha256(raw).hexdigest().startswith("ac7319ca")
s = raw.decode()
doc_end = s.index('"""', 3) + 3
doc, body = s[:doc_end], s[doc_end:]
prot = {"@@P1@@": "prompt 07b r31", "@@P2@@": "prompt 07a r30", "@@P3@@": "route_07a", "@@P4@@": "changed_vs_r28",
        "@@P5@@": "refusals_07b", "@@P6@@": "interest_07b", "@@P7@@": "replay07b_r31"}
for k, v in prot.items():
    body = body.replace(v, k)
n = [0]


def f(m):
    n[0] += 1
    return {"30": "31", "31": "32"}[m.group(1)]


body = re.sub(r'(?<=[A-Za-z_])(30|31)(?![0-9])', f, body)
print("tag shifts", n[0])
for k, v in prot.items():
    body = body.replace(k, v)
reps = [
    ('REH = MAIN / "var" / "levin-phase-floor-07b" / "rehearsal"  # prompt 07b r31\n',
     'REH = MAIN / "var" / "levin-phase-floor-07c" / "rehearsal"  # prompt 07c r32\n'),
    ('STORES = {"h": MAIN / "var" / "datastores" / "levin-phase-floor-07b-r32.sqlite"}  # prompt 07b r31\n',
     'STORES = {"h": MAIN / "var" / "datastores" / "levin-phase-floor-07c-r32.sqlite"}  # prompt 07c r32\n'),
    ('TAG = {"h": "R32"}  # prompt 07b r31\n', 'TAG = {"h": "R32"}  # prompt 07c r32\n'),
    ('R31_STORE = MAIN / "var" / "datastores" / "levin-phase-floor-07a-r31.sqlite"  # prompt 07b r31: R31, the parent rehearsal, read for the record\n',
     'R31_STORE = MAIN / "var" / "datastores" / "levin-phase-floor-07b-r31.sqlite"  # prompt 07c r32: R31, the parent rehearsal, read for the record\n'),
    ('R31_REH = MAIN / "var" / "levin-phase-floor-07a" / "rehearsal"  # prompt 07b r31: R31\'s block captures (read only)\n',
     'R31_REH = MAIN / "var" / "levin-phase-floor-07b" / "rehearsal"  # prompt 07c r32: R31\'s block captures (read only)\n'),
    ('W = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5")  # prompt 07b r31\n',
     'W = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c")  # prompt 07c r32\n'),
    ('sorted(STORES["h"].parent.glob("levin-phase-floor-07b-r32*.sqlite")) + sorted(R31_STORE.parent.glob("levin-phase-floor-07a-r31*.sqlite"))  # prompt 07b r31\n',
     'sorted(STORES["h"].parent.glob("levin-phase-floor-07c-r32*.sqlite")) + sorted(R31_STORE.parent.glob("levin-phase-floor-07b-r31*.sqlite"))  # prompt 07c r32\n'),
    ('"prompt 07b r31: the rows to replay, the four-block rows and the records of interest"',
     '"prompt 07c r32: the rows to replay, the four-block rows and the records of interest"'),
    ('f"prompt 07b r31: run {spec[\'name\']}"', 'f"prompt 07c r32: run {spec[\'name\']}"'),
    ('"prompt 07b r31: the replay checked against R32, record by record"',
     '"prompt 07c r32: the replay checked against R32, record by record"'),
    ('                           "refusals_07b": (c or {}).get("refusals_07b")})  # prompt 07b r31\n',
     '                           "refusals_07b": (c or {}).get("refusals_07b"),  # prompt 07b r31\n'
     '                           "chase_07c": (c or {}).get("chase_07c")})  # prompt 07c r32\n'),
    ('    print(f"refusals over the replayed calls ({dict(ncalls)}): {dict(sorted(ref.items()))}")\n',
     '    print(f"refusals over the replayed calls ({dict(ncalls)}): {dict(sorted(ref.items()))}")\n'
     '    # prompt 07c r32: the chase stop\'s counts (prompt 07c section 1.4) over every replayed call, by kind\n'
     '    ch = Counter()\n'
     '    for e in rows:\n'
     '        if not (e["touched"] or e["p1"] or e["wid"]):\n'
     '            continue\n'
     '        u, _, _ = units[f"h-1e-7-{e[\'R32\'][\'shard\']}-{e[\'R32\'][\'serial\']}"]\n'
     '        for kind_, cl in (("group", u.get("calls") or []), ("other", u.get("other_calls") or [])):\n'
     '            for c in cl:\n'
     '                cs = c.get("chase_07c") or {}\n'
     '                if cs.get("num_chase_stops") is None:\n'
     '                    ch[(kind_, "<no count>")] += 1\n'
     '                    continue\n'
     '                for k_ in ("num_chase_stops", "num_chase_stopped_leaves", "num_chase_noise_bisections"):\n'
     '                    ch[(kind_, k_)] += cs[k_]\n'
     '                ch[(kind_, "calls with a chase stop")] += cs["num_chase_stops"] > 0\n'
     '                rf = c.get("refusals_07b") or {}\n'
     '                ch[(kind_, "calls with a rate-rule refusal")] += ((rf.get("num_edge_rule_refusals") or 0) + (rf.get("num_range_rule_refusals") or 0)) > 0\n'
     '    print(f"chase stops over the replayed calls: {dict(sorted(ch.items()))}")  # prompt 07c r32\n'),
    ('        print(f"  {c[\'key\']} {c[\'path\']} {c[\'label\']}: reproduces {c[\'reproduces\']}; call index {c[\'index\']}; refusals {c[\'refusals_07b\']}")\n',
     '        print(f"  {c[\'key\']} {c[\'path\']} {c[\'label\']}: reproduces {c[\'reproduces\']}; call index {c[\'index\']}; refusals {c[\'refusals_07b\']}; chase {c[\'chase_07c\']}")  # prompt 07c r32\n'),
    ('"prompt 07b r31: the four blocks at h, checked against R32, and each capture against R31\'s"',
     '"prompt 07c r32: the four blocks at h, checked against R32, and each capture against R31\'s"'),
    ('"prompt 07b r31: the captures of the group records of interest, checked against R32"',
     '"prompt 07c r32: the captures of the group records of interest, checked against R32"'),
    ('                           converged=c["converged"], regions=c["num_regions"], refusals_07b=c.get("refusals_07b"),\n',
     '                           converged=c["converged"], regions=c["num_regions"], refusals_07b=c.get("refusals_07b"),\n'
     '                           chase_07c=c.get("chase_07c"), replay_chase_07c=rc[0].get("chase_07c"),  # prompt 07c r32\n'),
    ('and c["index"] == rc[0]["index"] and c.get("refusals_07b") == rc[0].get("refusals_07b"))\n',
     'and c["index"] == rc[0]["index"] and c.get("refusals_07b") == rc[0].get("refusals_07b")\n'
     '                      and c.get("chase_07c") == rc[0].get("chase_07c"))  # prompt 07c r32\n'),
]
for a, b in reps:
    assert body.count(a) == 1, (a[:120], body.count(a))
    body = body.replace(a, b)
newdoc = '''"""[prompt 07c section 5 rehearsal agent, R32: 07b's rehearsal replay07b_r31.py (ac7319ca), changed where marked
"prompt 07c r32" (and, throughout, the store tags shifted: R30 -> R31, R31 -> R32, r30 -> r31, r31 -> r32, in names, keys
and labels; by scripts/gen/make_replay07c_r32.py): one tree h (cc3368ce, code 67b5af74) reading R32 (R31 read for the
record); the rows: every R32 row with a nonzero count, P1's two rows, the w rows, the two block rows, and the rows holding
a record of interest (out/qsi_records_r31_r32.csv: a Levin record whose |value_32 - value_31| exceeds abserr_31 +
abserr_32, exactly, or whose converged changed); the check also reports each call's chase stops (worker.py's chase_07c)
beside its refusals and sums them over the replayed calls; the blockcheck compares each capture's blob with R31's
(state/inputs/ of 07b's rehearsal); refcapspec and refcapcheck capture the group calls of the records of interest and
check each capture against R32 (and its chase counts against the replay's).]

'''
open(D + "replay07c_r32.py", "w").write(newdoc + doc.replace('"""', '', 1) + body)
print("written")
