"""Compare two m3_messages.py outputs. For each message, the expected after-message is the
before-message with the repeated prefix removed: the second `Cannot <verb> sharded datastore
"<path>"[ to "<dst>"]: ` where there are two; the leading one for _read_closed_store's own; and, for the
reader, the reader's suffix added. Prints each pair, how many times each says "sharded
datastore", and whether the after-message is the expected one.
Usage: python m3_compare.py before.json after.json"""

import json, re, sys

before, after = (json.load(open(f)) for f in sys.argv[1:3])
PREFIX = r'Cannot \w+ sharded datastore "[^"]*"(?: to "[^"]*")?: '
SUFFIX = ". Nothing was read or repaired"
bad = 0
for case in sorted(before):
    for op in sorted(before[case]):
        b, a = before[case][op], after[case][op]
        if op == "_read_closed_store":
            expected = re.sub(r"^(RuntimeError: )" + PREFIX, r"\1", b)
        elif op == "open_read_only":
            expected = b + SUFFIX
        else:
            first = re.search(PREFIX, b)
            expected = b[: first.end()] + re.sub(PREFIX, "", b[first.end() :], count=1)
        verdict = "as expected" if a == expected else "UNEXPECTED"
        bad += a != expected
        print(f"== {case} / {op}: {verdict}")
        print(f"   before ({b.count('sharded datastore')}x): {b}")
        print(f"   after  ({a.count('sharded datastore')}x): {a}")
print(f"{bad} unexpected")
