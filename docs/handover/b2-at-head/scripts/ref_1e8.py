"""Filon references (tools/filon_reference, route B) for R|together|1e8's four G±q±r group calls.

Usage: ``python ref_1e8.py OUT.jsonl [--workers N] [--x 1e8] [--labels G+q+r,...]``.

Each reading (QAWO and FCC, at resolutions 1 and 2) is appended to OUT.jsonl as soon as it is
taken, with the label, the call's z range, the edges used, and this script's and
``build_1e8.py``'s SHA-256 and the git SHA at launch. A reading already in OUT.jsonl with the same
script hashes is not taken again, so a stopped run resumes; one with other hashes is ignored, never
re-stamped. The reference itself (scatter, the count rule's verdict, the comparison with the
Levin values) is formed from these lines afterwards, by ``score_1e8.py``.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from math import exp
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (HERE, REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ORDER = [("qawo", 1), ("fcc", 1), ("qawo", 2), ("fcc", 2)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--x", default="1e8")
    ap.add_argument("--labels", default="G+q+r,G+q-r,G-q+r,G-q-r")
    args = ap.parse_args()

    import build_1e8
    from tools import filon_reference as FR

    prov = {
        "ref_1e8_sha256": sha256(__file__),
        "build_1e8_sha256": sha256(build_1e8.__file__),
        "b2_cells_sha256": sha256(REPO / "docs/handover/b2/scripts/b2_cells.py"),
        "filon_reference_sha256": sha256(FR.__file__),
        "git_sha": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            same = all(
                r.get(k) == prov[k]
                for k in ("ref_1e8_sha256", "build_1e8_sha256", "b2_cells_sha256")
            )
            if same and r.get("x") == args.x:
                done.add((r["label"], r["impl"], r["res"]))

    for label in args.labels.split(","):
        c = build_1e8.call(args.x, label)
        a, b = c["x_span"]
        edges, ntab = build_1e8.tolerant_edges(c)
        for impl, res in ORDER:
            if (label, impl, res) in done:
                print(label, impl, res, "already taken", flush=True)
                continue
            t0 = time.time()
            r = FR.reading(
                c,
                impl,
                res,
                route="B",
                edges=edges,
                workers=args.workers,
                call_source=("build_1e8", "call", (args.x, label)),
            )
            r.pop("value_mp", None)
            rec = {
                "x": args.x,
                "label": label,
                "impl": impl,
                "res": res,
                "z_min": exp(a) - 1,
                "z_max": exp(b) - 1,
                "n_edges": len(edges),
                "n_tables": ntab,
                "workers": args.workers,
                "wall": time.time() - t0,
                "loadavg": os.getloadavg(),
                "reading": r,
                **prov,
            }
            with open(out, "a") as fh:
                fh.write(json.dumps(rec, default=repr) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(label, impl, res, f"{rec['wall']:.0f} s", r["value_30"], flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
