"""Generate the 07c rehearsal's s1_score_r32.py, s2_score_r32.py and block_score_r32.py from 07b's rehearsal's
s1_score_r31.py (305d44d0), s2_score_r31.py (5c499eb2) and block_score_r31.py (fcb8b436): after each docstring the store
tags shifted (R30 -> R31, R31 -> R32 in names, keys and labels, by (?<=[A-Za-z_])(30|31)(?![0-9])), outside the protected
strings "prompt 07b r31", "prompt 07a r30", "refusals_07b", "value_30", "ref_30", "replay07b_r31", "block_reference_r30";
then asserted replacements, each marked "prompt 07c r32": the paths, the replay module, R31's own score files (07b's
rehearsal) as the parent's context, the keys those files use, R32's block-row serials, and each call's chase_07c. A new
bracketed line opens each docstring. Run once: python -I -B make_scores_07c.py"""
import hashlib
import re

V = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/rehearsal/scripts/"
D = "/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07c/rehearsal/scripts/"
PROT = ["prompt 07b r31", "prompt 07a r30", "refusals_07b", "value_30", "ref_30", "replay07b_r31", "block_reference_r30"]


def shift(body):
    for i, p in enumerate(PROT):
        body = body.replace(p, f"@@P{i}@@")
    body, n = re.subn(r'(?<=[A-Za-z_])(30|31)(?![0-9])', lambda m: {"30": "31", "31": "32"}[m.group(1)], body)
    for i, p in enumerate(PROT):
        body = body.replace(f"@@P{i}@@", p)
    return body, n


def make(src, sha, dst, newline, reps):
    raw = open(V + src, "rb").read()
    assert hashlib.sha256(raw).hexdigest().startswith(sha), src
    s = raw.decode()
    end = s.index('"""', 3) + 3
    doc, body = s[:end], s[end:]
    body, n = shift(body)
    for a, b, cnt in reps:
        assert body.count(a) == cnt, (src, a[:100], body.count(a))
        body = body.replace(a, b)
    open(D + dst, "w").write('"""' + newline + "\n\n" + doc[3:] + body)
    print(dst, "tag shifts", n)


Z07B = '"/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/rehearsal"'
Z07C = '"/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07c/rehearsal"'
make("s1_score_r31.py", "305d44d0", "s1_score_r32.py",
     "[prompt 07c section 5 item 5, rehearsal agent, R32: 07b's rehearsal s1_score_r31.py (305d44d0), changed where marked "
     "\"prompt 07c r32\" (and the store tags shifted R30 -> R31, R31 -> R32 by scripts/gen/make_scores_07c.py): R32's store, the "
     "replay at h (replay07c_r32.py), 06g's scores.json read from the 07c worktree's docs; the keys v31/abserr31/... renamed "
     "v32/abserr32/... (they hold R32's numbers); each call's chase stops recorded; each row compared with R31's s1_scores.json "
     "(07b's rehearsal, keys v31/abserr31/regions31/ratio_v31).]",
     [(f'R = Path({Z07B})  # prompt 07b r31\n', f'R = Path({Z07C})  # prompt 07c r32\n', 1),
      ('R31_S1 = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/rehearsal/out/s1_scores.json")  # prompt 07b r31: read only\n',
       'R31_S1 = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/rehearsal/out/s1_scores.json")  # prompt 07c r32: R31\'s, read only\n', 1),
      ('import replay07b_r31 as RP  # noqa: E402  prompt 07b r31\n', 'import replay07c_r32 as RP  # noqa: E402  prompt 07c r32\n', 1),
      ('SCORES = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5/docs/', 'SCORES = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c/docs/', 1),
      ('glob("levin-phase-floor-07b-r32*.sqlite")):  # prompt 07b r31\n', 'glob("levin-phase-floor-07c-r32*.sqlite")):  # prompt 07c r32\n', 1),
      ('for k_ in ("v29", "abserr29", "regions29", "ratio_v29")},  # prompt 07b r31\n',
       'for k_ in ("v31", "abserr31", "regions31", "ratio_v31")},  # prompt 07c r32: R31\'s keys in 07b\'s s1_scores.json\n', 1),
      ('(r31[w["wid"]]["v29"], r31[w["wid"]]["abserr29"])', '(r31[w["wid"]]["v31"], r31[w["wid"]]["abserr31"])', 1),
      ("{row['R31']['ratio_v29']} (regions {row['R31']['regions29']})", "{row['R31']['ratio_v31']} (regions {row['R31']['regions31']})", 1),
      ('               "refusals_07b": c.get("refusals_07b"),  # prompt 07b r31\n',
       '               "refusals_07b": c.get("refusals_07b"),  # prompt 07b r31\n               "chase_07c": c.get("chase_07c"),  # prompt 07c r32\n', 1),
      ("refusals {row['refusals_07b']}; ", "refusals {row['refusals_07b']}; chase {row['chase_07c']}; ", 1),
      ])
make("s2_score_r31.py", "5c499eb2", "s2_score_r32.py",
     "[prompt 07c section 5 item 5, rehearsal agent, R32: 07b's rehearsal s2_score_r31.py (5c499eb2), changed where marked "
     "\"prompt 07c r32\" (and the store tags shifted R30 -> R31, R31 -> R32 by scripts/gen/make_scores_07c.py): the replay at h "
     "(replay07c_r32.py) and R32's records; R32's call compared with R28's (stop 1) and R31's (07b's rehearsal s2_scores.json, "
     "key R31); each call's chase stops recorded; outputs under the 07c rehearsal.]",
     [(f'Z = Path({Z07B})  # prompt 07b r31\n', f'Z = Path({Z07C})  # prompt 07c r32\n', 1),
      ('Z31 = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/rehearsal")  # prompt 07b r31: R31\'s s2_scores.json, for R31\'s call\n',
       'Z31 = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/rehearsal")  # prompt 07c r32: R31\'s s2_scores.json, for R31\'s call\n', 1),
      ('import replay07b_r31 as RP  # noqa: E402  prompt 07b r31\n', 'import replay07c_r32 as RP  # noqa: E402  prompt 07c r32\n', 1),
      ('"refusals_07b": call.get("refusals_07b")}  # prompt 07b r31\n',
       '"refusals_07b": call.get("refusals_07b"), "chase_07c": call.get("chase_07c")}  # prompt 07c r32\n', 1),
      ('              f"refusals {row[\'refusals_07b\']}")  # prompt 07b r31\n',
       '              f"refusals {row[\'refusals_07b\']}; chase {row[\'chase_07c\']}")  # prompt 07c r32\n', 1),
      ])
make("block_score_r31.py", "fcb8b436", "block_score_r32.py",
     "[prompt 07c section 5 item 5 (S4), rehearsal agent, R32: 07b's rehearsal block_score_r31.py (fcb8b436), changed where "
     "marked \"prompt 07c r32\" (and the store tags shifted R30 -> R31, R31 -> R32 by scripts/gen/make_scores_07c.py): R32's "
     "block replays and blocks.json; R32's shard-serials; R31's block_scores.json (07b's rehearsal, read only, key R31) as "
     "context with R32_equals_R31. The integrand is still compared with R29's capture, on which the references were taken.]",
     [(f'Z = Path({Z07B})  # prompt 07b r31\n', f'Z = Path({Z07C})  # prompt 07c r32\n', 1),
      ('Z31 = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07a/rehearsal")  # prompt 07b r31: R31\'s scores, read only\n',
       'Z31 = Path("/Users/ds283/Documents/Code/SecondaryGWKit/var/levin-phase-floor-07b/rehearsal")  # prompt 07c r32: R31\'s scores, read only\n', 1),
      ('BLOCKS = [("J4", "0-10232", "0-157"), ("Y4", "0-10232", "0-157"), ("J4", "1-20127", "1-10283"), ("Y4", "1-20127", "1-10283")]\n',
       'BLOCKS = [("J4", "0-10172", "0-157"), ("Y4", "0-10172", "0-157"), ("J4", "1-20068", "1-10283"), ("Y4", "1-20068", "1-10283")]  # prompt 07c r32: R32\'s serials\n', 1),
      ])
print("done")
