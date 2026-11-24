"""levin-phase-floor 07d Part A' addendum, item A.2: ref48, the second rule taken further, on the selected leaves
(state/rule_ref48_add.md, its section A.2; state/selection_add.json).

Composite 48-point Gauss-Legendre on the leaf split at v4's knots, n = 512 and 1,024 uniform panels a piece (math.fsum),
and 2,048 where |I_1024 - I_512| exceeds the noise term 2 sigma_hat sqrt(sum w^2). Writes OUT_DIR/<leaf>.json, one per
leaf; refuses to overwrite. Leaves are dealt to N workers greedily by piece count (largest first), worker I taking its
share.
Usage: PYTHONPATH=<tree>:<a2 scripts>:<addendum scripts> python -B ref48_add.py OUT_DIR I N
"""

import datetime
import hashlib
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

ADD = Path(__file__).resolve().parent.parent
GX, GW = np.polynomial.legendre.leggauss(48)
fx = float.fromhex


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def gl(G, brk, n):
    terms, w2 = [], 0.0
    for pa, pb in zip(brk[:-1], brk[1:]):
        h = (pb - pa) / n
        x0 = pa + h * np.arange(n)
        x1 = np.append(x0[1:], pb)
        half = 0.5 * (x1 - x0)
        xs = (0.5 * (x0 + x1))[:, None] + half[:, None] * GX[None, :]
        ww = half[:, None] * GW[None, :]
        terms.append((ww * G.g_on(xs.ravel()).reshape(xs.shape)).ravel())
        w2 += float(np.sum(ww * ww))
    return math.fsum(np.concatenate(terms)), w2


def deal(leaves, n):
    order = sorted(leaves, key=lambda i: (-leaves[i]["n_pieces"], i))
    load, share = [0] * n, [[] for _ in range(n)]
    for i in order:
        k = load.index(min(load))
        share[k].append(i)
        load[k] += leaves[i]["n_pieces"]
    return share


def main():
    out_dir, part, nparts = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    sel = json.loads((ADD / "state" / "selection_add.json").read_text())
    mine = deal(sel["leaves"], nparts)[part]
    import AdaptiveLevin.levin_quadrature as LQ
    import leafref_v4 as R4
    from prov import TREE_LQ_SHA

    if sha(LQ.__file__) != TREE_LQ_SHA:
        raise SystemExit("the module is not the tree's")
    out_dir.mkdir(parents=True, exist_ok=True)
    caps = {}
    for i in mine:
        p = out_dir / (i.replace("|", "_").replace("#", "_") + ".json")
        if p.exists():
            raise SystemExit(f"{p} exists")
        lv = sel["leaves"][i]
        doc = json.loads(Path(lv["file"]).read_text())
        spec = json.loads(Path(doc["header"]["spec"]).read_text())
        key = spec["capture"]["sha256"]
        if key not in caps:
            caps = {key: R4.load_capture(spec["capture"])}
        rec = caps[key]
        xs0, xs1 = (fx(v) for v in spec["x_span"])
        edges, _, _ = R4.call_knots(rec, xs0, xs1)
        a, b = fx(lv["start"]), fx(lv["end"])
        kn = R4.interior(edges, a, b)
        brk = [a] + (kn or []) + [b]
        G = R4.Integrand(LQ, rec)
        sig = lv["sigma_hat"]
        t0 = time.perf_counter()
        n0 = G.n_eval
        levels = {}
        for n in (512, 1024):
            levels[n] = gl(G, brk, n)
        c = abs(levels[1024][0] - levels[512][0])
        noise = 2 * sig * math.sqrt(levels[1024][1])
        finest = 1024
        moving_2048 = None
        if c > noise:
            levels[2048] = gl(G, brk, 2048)
            c = abs(levels[2048][0] - levels[1024][0])
            noise = 2 * sig * math.sqrt(levels[2048][1])
            finest = 2048
            moving_2048 = c > noise
        ref = levels[finest][0]
        r = {"header": {"title": f"levin-phase-floor 07d Part A' addendum: ref48 on {i}",
                        "written": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "host": socket.gethostname(), "script": str(Path(__file__).resolve()), "script_sha256": sha(__file__),
                        "leafref_v4_sha256": sha(R4.__file__), "selection_sha256": sha(ADD / "state" / "selection_add.json"),
                        "leaf_record": lv["file"], "leaf_record_sha256": sha(lv["file"])},
             "id": i, "n_pieces": len(brk) - 1, "sigma_hat": sig,
             "levels": {str(n): {"I": v[0].hex(), "sum_w2": v[1]} for n, v in levels.items()},
             "finest": finest, "ref48": ref.hex(), "change": c, "noise": noise, "eps48": c + noise,
             "dominant": "change" if c >= noise else "noise", "moving_at_2048": moving_2048,
             "evals": G.n_eval - n0, "wall_s": time.perf_counter() - t0}
        p.write_text(json.dumps(r, indent=1))
        print(i, finest, r["dominant"], f"{r['wall_s']:.1f} s", flush=True)


if __name__ == "__main__":
    main()
