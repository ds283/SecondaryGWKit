"""levin-phase-floor prompt 07d Part A, section 2.6: the reference readings' inputs and their cost features, for a drawn set
((Ra) or (Rt)). New in 07d.

For each drawn integrand (its call's capture, the blob's SHA-256 checked) it writes the call's own pickle
(state/refs/inputs/<key>.pkl, as one_reading.py and leafref read it), the tool's route (``route_of``), and the features the
projection reads: Route A, the panel counts of FCC and QAWO at resolutions 1 and 2 (the tool's ``RouteA.panels``); Route B,
07c Part A's features (refs_projection_07c.feats: production segments, n_split at split ratio 1.8, D_nonmono).
Writes out/refs/inputs_<SET>.json. Usage: PYTHONPATH=<tree>:<scripts> python -B refs_inputs_07d.py SET SELECTION_JSON
(SET ra or rt; the selection's ``selection`` entries carry ``sid``).
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE / "from_07c_measurement"))
from prov import header, sha256  # noqa: E402


def qawo_panel_count(A, res):
    """The number of panels RouteA.panels("qawo", res) returns, counted without listing them (at a large K the list
    has millions of entries): per u-segment, the tool's dyadic_tiling count of full panels plus its remainder loop,
    the same arithmetic in the same order."""
    import math

    import mpmath as mp
    from tools.filon_reference import ROUTE_A_QAWO, SLIVER_BITS

    cfg = ROUTE_A_QAWO[res]
    ee = [mp.exp(u) for u in A.u_segments(cfg["du"])]
    Kabs = abs(float(A.K_exact))
    n = 0
    for L, R in zip(ee[:-1], ee[1:]):
        h_max = min(cfg["Pmax"] / (2 * Kabs), float(R - L) / 2)
        h = math.ldexp(1.0, math.floor(math.log2(h_max)))
        count = int(mp.floor((R - L) / (2 * h)))
        n += count
        pos = L + count * (2 * h)
        rem = R - pos
        hh = h / 2
        min_width = 2 * h * 2.0 ** (-SLIVER_BITS)
        while rem >= min_width and hh > 0:
            if rem >= 2 * hh:
                n += 1
                pos = pos + 2 * hh
                rem = R - pos
            hh = hh / 2
        if rem > 0:
            n += 1
    return n


def main():
    set_, sel_p = sys.argv[1], Path(sys.argv[2])
    out_p = ROOT / "out" / "refs" / f"inputs_{set_}.json"
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    (ROOT / "state" / "refs" / "inputs").mkdir(parents=True, exist_ok=True)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    from replay_07d import manifest
    from tools.filon_reference import _make_state, route_of

    man = manifest()
    by_capture = {}
    for c in man.values():
        by_capture.setdefault(c["sha256"], c)
    sel = json.loads(sel_p.read_text())["selection"]
    rows = {}
    for e in sel:
        sid = e["sid"]
        c = man.get(f"d:{sid}")
        if c is None:
            raise SystemExit(f"{sid}: not in the replay manifest")
        key = f"{set_}-{sid.replace('|', '-')}"
        with open(c["path"], "rb") as f:
            f.seek(c["offset"])
            blob = f.read(c["length"])
        if hashlib.sha256(blob).hexdigest() != c["sha256"]:
            raise SystemExit(f"{sid}: capture blob has another SHA-256")
        pk = ROOT / "state" / "refs" / "inputs" / f"{key}.pkl"
        if not pk.exists():
            pk.write_bytes(blob)
        import ray.cloudpickle as cp

        call = cp.loads(blob)
        route = route_of(call)
        feat = {}
        if route == "A":
            st = _make_state(call, "A", None)
            A = st["route"]
            for res in (1, 2):
                feat[f"panels_fcc_r{res}"] = len(A.panels("fcc", res))
                feat[f"panels_qawo_r{res}"] = qawo_panel_count(A, res)
            feat["K_abs"] = abs(float(A.K_exact))
        else:
            from refs_feats_07d import feats  # 07c's, copied (refs_projection_07c cannot be imported here)

            feat = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in feats(call).items()}
        rows[key] = {"sid": sid, "unit": call["unit"], "index": call["index"], "rtol": call["rtol"], "route": route,
                     "kind": "analytic" if route == "A" else "group", "pickle": str(pk), "pickle_sha256": c["sha256"],
                     "notify_label": call["notify_label"], "features": feat}
        print(key, route, feat, flush=True)
    hdr = header(Path(__file__).resolve(), f"levin-phase-floor 07d Part A: reference inputs ({set_})", [sel_p])
    out_p.write_text(json.dumps({"header": hdr, "inputs": rows}, indent=1))
    print("wrote", out_p, sha256(out_p))


if __name__ == "__main__":
    main()
