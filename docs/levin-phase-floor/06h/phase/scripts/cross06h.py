"""levin-phase-floor prompt 06h section 5, the phase re-measurement: one tree's code scored at the
other tree's points. New in 06h.

H1 is scored on h's points: every node of every returned leaf of h's replay, and the dense sample.
"The parent's declaration is scored on the same points beside it" (prompt 06h section 5): where h's
and the parent's returned leaves differ, the node sets differ, so each tree's replay scores its own
nodes, and this script scores the tree on PYTHONPATH (normally ``p``) at the **other** tree's points
(normally ``h``'s), with its own captured group: production's ``theta_mod_2pi(u)`` and
``theta_abserr(u)`` of this tree, against the other tree's stored reference at the exact e^u
(replay06f.py's ``ref_exact_mod_2pi``, 40 digits, reduced into (-pi, pi]). The reference is a
function of the group's orders, coefficients and signs alone, which both trees' captures share
(checked here: the two groups' coefficients, signs and orders are equal, and the two calls' spans
are equal bit for bit). No mpmath Bessel function is evaluated here; the difference is reduced at
30 digits, as in the replay.

The dense sample is the same in both trees when the spans are equal (one seed, one span): this
script also checks that the other tree's dense points and references equal this tree's own, bit
for bit and digit for digit.

Writes OUT_DIR/<this unit>_<index>/cross.json (summary) and rows.json.gz (one row per point).

Usage: PYTHONPATH=<tree T>:<scripts> python -B cross06h.py CAPTURE_JSON UNIT INDEX OWN_REPLAY_DIR
       OTHER_REPLAY_DIR OUT_DIR
(OWN_REPLAY_DIR and OTHER_REPLAY_DIR are the two trees' out/replay_<tree>/<unit>_<index> dirs.)
"""

import gzip
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import TREES, sha256, tree_state  # noqa: E402
from replay06f import MP_DPS, jdump, load_capture, wrap  # noqa: E402


def main():
    cap_json, uid, index, own_dir, other_dir, out_dir = sys.argv[1:7]
    index = int(index)
    own_dir, other_dir = Path(own_dir), Path(other_dir)
    import mpmath as mp

    import AdaptiveLevin.levin_quadrature as LQ

    tree = LQ.__file__.split("/AdaptiveLevin/")[0]
    head, status = tree_state(tree)
    names = [n for n, (t, _s) in TREES.items() if str(t) == tree]
    tree_name = names[0] if len(names) == 1 else None
    hdr = {
        "title": "levin-phase-floor 06h: one tree's code scored at the other tree's points",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "script": "cross06h.py",
        "script_sha256": sha256(Path(__file__).resolve()),
        "replay06f_sha256": sha256(HERE / "replay06f.py"),
        "prov_sha256": sha256(HERE / "prov.py"),
        "tree": tree,
        "tree_name": tree_name,
        "tree_head": head,
        "tree_status_short": status,
        "three_bessel_integrals_sha256": sha256(
            Path(tree) / "LiouvilleGreen" / "three_bessel_integrals.py"
        ),
        "capture_json": cap_json,
        "capture_json_sha256": sha256(cap_json),
        "unit": uid,
        "index": index,
        "own_replay_dir": str(own_dir),
        "other_replay_dir": str(other_dir),
        "inputs_sha256": {
            str(p): sha256(p)
            for p in (
                own_dir / "result.json",
                own_dir / "dense.json.gz",
                other_dir / "result.json",
                other_dir / "nodes.json.gz",
                other_dir / "dense.json.gz",
            )
        },
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "dont_write_bytecode": sys.dont_write_bytecode,
    }
    if (
        tree_name is None
        or head != TREES[tree_name][1]
        or status.strip()
        or not uid.startswith(tree_name + "-")
    ):
        raise SystemExit(f"tree evidence does not hold: {hdr}")
    rec, cap = load_capture(cap_json, uid, index)
    hdr["capture_blob_sha256"] = cap["sha256"]
    group = rec["theta"]["theta_mod_2pi"].__self__
    own = json.loads((own_dir / "result.json").read_text())
    other = json.loads((other_dir / "result.json").read_text())
    og = other["M2"]["group"]
    checks = {
        "spans_equal": [x["hex"] for x in own["x_span"]]
        == [x["hex"] for x in other["x_span"]],
        "coefficients_equal": [float(c).hex() for c in group.coefficients]
        == [float(c).hex() for c in og["coefficients"]],
        "signs_equal": list(map(float, group.signs)) == list(map(float, og["signs"])),
        "orders_equal": [float(p.nu) for p in group.phases]
        == [float(n) for n in og["nu"]],
        "K_equal": float(group.K).hex() == float(og["K"]).hex(),
    }
    own_dense = json.load(gzip.open(own_dir / "dense.json.gz", "rt"))
    other_dense = json.load(gzip.open(other_dir / "dense.json.gz", "rt"))
    checks["dense_points_equal"] = [r["u"] for r in own_dense] == [
        r["u"] for r in other_dense
    ]
    checks["dense_references_equal"] = [r["ref_exact_mod_2pi"] for r in own_dense] == [
        r["ref_exact_mod_2pi"] for r in other_dense
    ]
    if not all(
        checks[k]
        for k in ("coefficients_equal", "signs_equal", "orders_equal", "K_equal")
    ):
        raise SystemExit(f"the two trees' groups differ: {checks}")
    mp.mp.dps = MP_DPS
    rows = []
    for name, kind in (("nodes.json.gz", "node"), ("dense.json.gz", "dense")):
        for r in json.load(gzip.open(other_dir / name, "rt")):
            u = float.fromhex(r["u"])
            th = float(group.theta_mod_2pi(u))
            ta = float(group.theta_abserr(u))
            d = float(wrap(mp, mp.mpf(th) - mp.mpf(r["ref_exact_mod_2pi"])))
            parts = None
            if hasattr(group, "_declared_error_parts"):
                parts = [float(x) for x in group._declared_error_parts(u)]
            rows.append(
                {
                    "set": kind,
                    "leaf_other": r.get("leaf"),
                    "k": r.get("k", r.get("j")),
                    "u": r["u"],
                    "theta_mod_2pi": th.hex(),
                    "delta_theta": d,
                    "theta_abserr": ta,
                    "ratio": abs(d) / ta if ta > 0 else None,
                    "theta_abserr_parts": parts,
                    "other_theta_mod_2pi": r["theta_mod_2pi"],
                    "other_delta_theta": r["delta_theta"],
                    "other_theta_abserr": r["theta_abserr"],
                    "other_ratio": r["ratio"],
                }
            )
    odir = Path(out_dir) / f"{uid}_{index}"
    odir.mkdir(parents=True, exist_ok=False)
    rows_sha = jdump(odir / "rows.json.gz", rows)
    summ = {}
    for kind in ("node", "dense", "all"):
        sel = [r for r in rows if kind == "all" or r["set"] == kind]
        rat = [r["ratio"] for r in sel if r["ratio"] is not None]
        summ[kind] = {
            "n": len(sel),
            "n_violating": sum(1 for x in rat if x > 1.0),
            "max_ratio": max(rat) if rat else None,
            "n_other_violating": sum(
                1
                for r in sel
                if r["other_ratio"] is not None and r["other_ratio"] > 1.0
            ),
            "max_other_ratio": max(
                (r["other_ratio"] for r in sel if r["other_ratio"] is not None),
                default=None,
            ),
        }
    out = {
        "header": hdr,
        "checks": checks,
        "summary": summ,
        "rows_sha256": rows_sha,
    }
    (odir / "cross.json").write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {
                "unit": uid,
                "index": index,
                "checks": checks,
                "all": summ["all"],
                "digest": hashlib.sha256(rows_sha.encode()).hexdigest(),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
