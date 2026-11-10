"""Hash every regular file under each path given (a file or a directory), read only.

Usage: python -I hash_tree.py OUT PATH [PATH ...]
Writes one line per file to OUT: sha256  size_bytes  mtime_ns  path (absolute, sorted).
Orchestrator's check for levin-phase-floor 06m Part B (Mac Studio); opens nothing for writing
except OUT.
"""

import hashlib
import os
import sys


def files_under(p):
    if os.path.isfile(p):
        yield p
        return
    for root, dirs, names in os.walk(p):
        dirs.sort()
        for n in sorted(names):
            yield os.path.join(root, n)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    rows = []
    for p in paths:
        for f in files_under(os.path.abspath(p)):
            st = os.lstat(f)
            rows.append((f, sha256(f), st.st_size, st.st_mtime_ns))
    rows.sort()
    with open(out, "w") as fh:
        fh.write("# sha256  size_bytes  mtime_ns  path\n")
        for f, h, s, m in rows:
            fh.write(f"{h}  {s}  {m}  {f}\n")
    print(f"{len(rows)} files, {sum(r[2] for r in rows)} bytes -> {out}")


if __name__ == "__main__":
    main()
