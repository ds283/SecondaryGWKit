"""Final review (levin-phase-floor 07c Part C, §C.2): my own byte index of one census checkpoint. Read only but for OUT.

usage: python -I -B build_index.py CHECKPOINT OUT_JSON

One pass: the file's SHA-256, and for each line its byte offset, length (newline excluded) and SHA-256, keyed by the
line's unit with the tree letter stripped (row@rtol, e.g. "2-32862@1e-7"). The unit is taken from the line's leading
'{"unit": "<x>-<rtol>-<shard>-<serial>"' and asserted equal to the parsed line's unit for every line. A unit appearing
twice is recorded under "duplicates" and not in "units".
"""
import hashlib, json, re, sys
from pathlib import Path

PFX = re.compile(rb'^\{"unit": "([a-z])-(1e-[78])-(\d+-\d+)"')


def main():
    ck, out = Path(sys.argv[1]), Path(sys.argv[2])
    fh = hashlib.sha256(); units = {}; dup = []; trees = set(); off = 0; n = 0
    with open(ck, "rb") as f:
        for raw in f:
            fh.update(raw)
            body = raw[:-1] if raw.endswith(b"\n") else raw
            m = PFX.match(body)
            assert m, (off, body[:80])
            tree, rt, row = (x.decode() for x in m.groups())
            assert json.loads(body)["unit"] == f"{tree}-{rt}-{row}"
            trees.add(tree)
            key = f"{row}@{rt}"
            e = [off, len(body), hashlib.sha256(body).hexdigest()]
            if key in units:
                dup.append([key, units[key], e])
            else:
                units[key] = e
            off += len(raw); n += 1
    for k, a, b in dup:
        units.pop(k, None)
    res = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checkpoint=str(ck),
               sha256=fh.hexdigest(), size=off, n_lines=n, trees=sorted(trees), n_units=len(units), duplicates=dup,
               units=units)
    out.write_text(json.dumps(res))
    print(ck, "sha256", res["sha256"], "lines", n, "units", len(units), "duplicates", len(dup), "trees", sorted(trees))


if __name__ == "__main__":
    main()
