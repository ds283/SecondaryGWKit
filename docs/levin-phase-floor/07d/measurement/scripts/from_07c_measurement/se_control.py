"""levin-phase-floor prompt 06j, S-e's control stencil, redone with a per-h window. New in 06j; supersedes nothing in
run 1's output (out/partS/*.json.gz keep their control None), and is recorded as a deviation. probe_s.py's rule took
the control at the midpoint between the seam and the leaf's farther end only if no seam lay within 3 h_max of it,
with h_max = w/4: that window (0.75 w) always held the seam itself, so no control was ever taken. Here the window is
3 h for each step h separately; a step whose window holds a seam has no control. Everything else is probe_s.py's
(its ``stencil`` function, imported unchanged).

For every seam held by a sampled leaf (out/partS/*.json.gz), writes the seam's stencils (recomputed, and checked equal
to run 1's) and the control's, per step, to out/partS_se_control.json, and prints the jumps against the controls.

Usage: PYTHONPATH=<tree>:<this directory> python -B se_control.py
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
import seams_06j as SM  # noqa: E402
from probe_s import Probe, stencil, STEPS  # noqa: E402
from prov import header  # noqa: E402


def main():
    import ray.cloudpickle as cp

    import AdaptiveLevin.levin_quadrature as LQ

    files = sorted((RC.ROOT / "out" / "partS").glob("s*.json.gz"))
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06j: S-e's control stencil with a per-h window", files)
    for l in hdr:
        print(l)
    out = {"header": hdr, "seams": []}
    for p in files:
        d = json.loads(gzip.open(p).read())
        for L in d["leaves"]:
            if not L.get("S_e"):
                continue
            rec = cp.loads((RC.ROOT / "state" / "inputs" / f"{d['sid']}.pkl").read_bytes())
            pr = Probe(LQ, rec)
            sm, _ = SM.seams(rec)
            sm_x = np.array(sorted({s["x"] for s in sm}))
            a, b = L["start"], L["end"]
            w = b - a
            for se in L["S_e"]:
                xs = se["x"]
                far = b if (b - xs) >= (xs - a) else a
                xc = (xs + far) / 2.0
                per = {}
                for k in STEPS:
                    h = w * 2.0 ** -k
                    st = stencil(pr, xs, h)
                    same = st == se["stencils"][f"h=w*2^-{k}"] or json.loads(json.dumps(st)) == se["stencils"][f"h=w*2^-{k}"]
                    ok = len(SM.held(sm_x, xc - 3 * h, xc + 3 * h)) == 0
                    ctrl = stencil(pr, xc, h) if ok else None
                    per[f"h=w*2^-{k}"] = {"seam": st, "seam_equals_run1": same, "control_x": xc if ok else None, "control": ctrl}
                    print(f"{d['sid']} seam {xs!r} h=w*2^-{k}: F jump {st['F']['jump'][0]:.3e} (d1 {st['F']['jump'][1]:.3e}, d2 {st['F']['jump'][2]:.3e})"
                          + (f"; control {ctrl['F']['jump'][0]:.3e} (d1 {ctrl['F']['jump'][1]:.3e}, d2 {ctrl['F']['jump'][2]:.3e})" if ctrl else "; no control")
                          + f"; theta_mod_2pi jump {st['theta_mod_2pi']['jump'][0]:.3e}"
                          + (f" control {ctrl['theta_mod_2pi']['jump'][0]:.3e}" if ctrl else "")
                          + f"; |F| at seam {abs(se['at_seam']['F']):.3e}; equals run 1 {same}")
                out["seams"].append({"sid": d["sid"], "leaf": [a, b], "x": xs, "z": se["z"], "kinds": se["kinds"], "steps": per})
    (RC.ROOT / "out" / "partS_se_control.json").write_text(json.dumps(out, indent=1))
    print(f"wrote out/partS_se_control.json sha256 {RC.sha256(RC.ROOT / 'out' / 'partS_se_control.json')}")


if __name__ == "__main__":
    main()
