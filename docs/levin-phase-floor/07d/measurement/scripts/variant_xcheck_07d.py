"""levin-phase-floor prompt 07d Part A, section 2.5: the cross-check of a variant's declaration against D-E's scored form,
recomputed from the files the agent's first, unrecorded check wrote at 10:36Z (deviation 19). New in 07d.

Inputs (out/replays_test/, written at 10:36Z): xcheck_variant_leaves.json, the first four Clenshaw-Curtis leaves of the
Ea smoke replay of R:c-1e-7-0-13757-5 (out/replays_test/Ea/R_c-1e-7-0-13757-5.json); xcheck_spec.json, those leaves as a
leafcc spec; xcheck_out.json, leafcc_07d.py's measurement of them. Per leaf: the variant's abserr against nest3 + 4 s_own and
its phase_err against floor + 4 s_own (d_e_table_07d.s_own, TAIL_K 8), and the relative differences.
Usage: PYTHONPATH=<tree>:<scripts> python -B variant_xcheck_07d.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prov import header  # noqa: E402
from d_e_table_07d import s_own  # noqa: E402

fx = float.fromhex


def main():
    d = ROOT / "out" / "replays_test"
    srcs = [d / "xcheck_variant_leaves.json", d / "xcheck_spec.json", d / "xcheck_out.json", d / "Ea" / "R_c-1e-7-0-13757-5.json"]
    for l in header(Path(__file__).resolve(), "levin-phase-floor 07d Part A: Ea's declaration against nest3 + 4 s_own", srcs):
        print(l)
    v = json.loads(srcs[0].read_text())
    o = json.loads(srcs[2].read_text())["leaves"]
    worst = 0.0
    for a, b in zip(v, o):
        assert (a["start"], a["end"]) == (b["start"], b["end"]) or (a["start"], a["end"]) == (b["end"], b["start"]), "leaf mismatch"
        s = s_own(b, 8)
        n3 = fx(b["ingredients"]["nest3"])
        r1 = fx(a["abserr"]) / (n3 + 4 * s) - 1
        r2 = fx(a["phase_err"]) / (fx(b["floor"]) + 4 * s) - 1
        worst = max(worst, abs(r1), abs(r2))
        print(f"[{a['start']}, {a['end']}] variant abserr {fx(a['abserr']):.6e} nest3+4s {n3 + 4 * s:.6e} | "
              f"phase_err {fx(a['phase_err']):.6e} floor+4s {fx(b['floor']) + 4 * s:.6e} | rel {r1:.2e} {r2:.2e}")
    print(f"{len(v)} leaves; largest |relative difference| {worst:.2e}")


if __name__ == "__main__":
    main()
