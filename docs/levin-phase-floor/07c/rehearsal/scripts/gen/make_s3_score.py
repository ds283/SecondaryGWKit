"""Generate the 07c rehearsal's s3_score.py from 07b's (f6396428): docstring rewritten; the store tags 31 -> 32 then
30 -> 31 shifted after the docstring by (?<![0-9.])3([01])(?![0-9]); the reader module tabulate_qsi_r31_r32; and
section 9.5's named row (07c) reported, each replacement asserted. Usage: python -I -B make_s3_score.py SRC DST"""
import hashlib, re, sys
src, dst = sys.argv[1:3]
raw = open(src, "rb").read(); assert hashlib.sha256(raw).hexdigest().startswith("f6396428")
s = raw.decode(); end = s.index('"""', 3) + 3
DOC = '''"""S3 -- consistency (prompt 07c section 5 item 4), R32 against R31, over every QuadSourceIntegral row.

levin-phase-floor 07c, rehearsal agent, R32. 07b's rehearsal agent's s3_score.py (f6396428...), with the store tags
30 -> 31 and 31 -> 32 and the reader's module (tabulate_qsi_r31_r32.py) changed, section 9.5's named row added, and
this docstring (by gen/make_s3_score.py). Reads both stores with sqlite mode=ro through tabulate_qsi_r31_r32.py's
reader (06g's, verbatim). The rule: every row with finite total_abserr in both has
|total_32 - total_31| <= total_abserr_31 + total_abserr_32, or is one section 9 predicts to move further. Section 9.5
names one row, R31's record G+q+r at (k, q, r) = (1e5, 1e6, 1e6), z = 144627.7 (predicted to converge at 86
regions, with total_converged possibly false -> true); it is listed whether or not it is beyond the bound. Holds if no
row other than a named one violates it. Evaluated exactly, in fractions.Fraction on the doubles (no rounding); the
ratio |d|/(e31 + e32) is printed as a float.

    python -B s3_score.py <R31 primary> <R32 primary>
"""'''
body = s[end:]
body, n = re.subn(r"(?<![0-9.])3([01])(?![0-9])", lambda m: "3" + str(int(m.group(1)) + 1), body)
assert body.count("from tabulate_qsi_r31_r32 import read") == 1 and body.count('"tabulate_qsi_r31_r32.py"') == 1  # the module name shifts with the tags
reps = [
 ("def main():\n", "NAMED = {(100000.0, 1000000.0, 1000000.0, 144627.70674226896)}  # prompt 07c section 9.5 (S3's named row)\n\n\ndef main():\n"),
 ("        ok = d <= s\n        viol += not ok\n",
  "        ok = d <= s\n        if key in NAMED:  # prompt 07c: a named row is listed, and does not count as a violation\n"
  "            named.append({\"key\": list(key), \"ratio\": float(d / s) if s else 0.0, \"within_bound\": ok, \"t31\": t31.hex(), \"t32\": t32.hex(), \"e31\": e31.hex(), \"e32\": e32.hex()})\n"
  "        viol += (not ok) and key not in NAMED\n"),
 ("    rows = []\n", "    rows = []\n    named = []  # prompt 07c\n"),
 ('    print(f"S3 {', '    for x in named:  # prompt 07c\n        print(f"  section 9.5\'s named row {x[\'key\']}: |d|/(e31 + e32) {x[\'ratio\']!r}, within the bound {x[\'within_bound\']}; total {x[\'t31\']} -> {x[\'t32\']}, abserr {x[\'e31\']} / {x[\'e32\']}")\n    print(f"S3 {'),
 ('"moved": rows}', '"moved": rows, "named_9_5": named}'),
]
for a, b in reps:
    assert body.count(a) == 1, a; body = body.replace(a, b)
open(dst, "w").write(DOC + body); print("tag shifts", n)
