"""Count, in an r2_resolve.py output, the absent-parent forms prompt 02 moves, kept apart from
the parent-set members that already resolve to None.
Usage: r2_count.py out.json"""

import json, sys

d = json.load(open(sys.argv[1]))
key_none = [0]  # a GkWKBIntegration key whose `numeric` key parent is None
member_none = [0]  # a GkSource parent-set member whose `numeric` field is None
unresolved = [0, 0]  # mappings holding "unresolved": all, and those with digest None


def walk(x, cls, in_set):
    if isinstance(x, dict):
        if "unresolved" in x:
            unresolved[0] += 1
            unresolved[1] += x["digest"] is None
        if x.get("numeric", 0) is None:
            if cls == "GkWKBIntegration":
                key_none[0] += 1
            elif in_set:
                member_none[0] += 1
        for k, v in x.items():
            if k == "key" and "class" in x:
                walk(v, x["class"], False)
            elif k == "set":
                for member in v:
                    walk(member, None, True)
            else:
                walk(v, None, False)
    elif isinstance(x, list):
        for v in x:
            walk(v, None, in_set)


for kind in d:
    for name, records in d[kind].items():
        for r in records:
            walk(r, name, False)
print(f"GkWKBIntegration key `numeric` None: {key_none[0]}")
print(f"parent-set member `numeric` None:   {member_none[0]}")
print(f"unresolved mappings: {unresolved[0]}, of which digest None: {unresolved[1]}")
