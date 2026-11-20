"""Generate qsi_summary_r31_r32.py from 07b's rehearsal's qsi_summary_r30_r31.py (50d33417).

Changes, each asserted: the docstring (to the first closing triple quote) rewritten; the tree path
changed; the store tags 31 -> 32 then 30 -> 31 shifted on every line after the docstring by the
regex (?<![0-9.])3([01])(?![0-9]) (the local names c30 becomes c31 with them; c29 is untouched).
Usage: python -I -B make_qsi_summary_r31_r32.py SRC DST
"""
import hashlib, re, sys

src, dst = sys.argv[1:3]
raw = open(src, "rb").read()
assert hashlib.sha256(raw).hexdigest().startswith("50d33417")
s = raw.decode()
end = s.index('"""', 3) + 3
DOC = '''"""Summaries of the R32-against-R31 tabulation (tabulate_qsi_r31_r32.py's two CSVs), for stop 1.

levin-phase-floor 07c, rehearsal agent, prompt 07c section 5 item 3. 07b's rehearsal agent's
qsi_summary_r30_r31.py (50d33417...), with the store tags 30 -> 31 and 31 -> 32, the tree path and
this docstring changed by gen/make_qsi_summary_r31_r32.py; nothing else. Its function
`additions_07b` (called at the end of main) is 07b's, its name and heading kept: the |dtotal|/abserr
bands on rows whose total moved; regions per moved row and by (k, q, r); max-depth changes and the
rows at depth 20; the analytic and group records' regions; the group records whose value moved,
against both abserrs; the Levin records whose converged changed; and the group records' abserr
ratios. Reads only the two CSVs.

    python -B qsi_summary_r31_r32.py <rows csv> <records csv>
"""'''
body = s[end:]
old_t = 't = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5"'
assert body.count(old_t) == 1
body = body.replace(old_t, 't = "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c"')
body, n = re.subn(r"(?<![0-9.])3([01])(?![0-9])", lambda m: "3" + str(int(m.group(1)) + 1), body)
open(dst, "w").write(DOC + body)
print("tag shifts", n)
