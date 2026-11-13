"""levin-phase-floor prompt 07 section 6: an index of 06m's census checkpoint (07's "before"), read only. Copied from
06m's census scripts/index_routeB_06m.py (a record, not edited), changed only where marked "prompt 07": the run is
06m's census (prov.RUN06M), its checkpoint SHA-256 ea5bc3ab... (prov.CHECKPOINT06M_SHA256), and the title. 06m's text
follows.

levin-phase-floor prompt 06m section 5: an index of route B's census checkpoint (06l's, the "before"), read only.
Copied from 06l's scripts/index_06g.py (a record, not edited), changed only where marked "prompt 06m": the run is
route B's census (prov.RUN06L_B), its checkpoint SHA-256 5cb3d3c0... (prov.CHECKPOINT06L_B_SHA256), and the title.
06l's state/index_06g.json was deleted on 2026-11-05 (06l's KEPT_ON_MAC_STUDIO.md), and no index of route B's checkpoint
is kept; this one is built from the kept checkpoint. 06l's text follows.

levin-phase-floor prompt 06k: an index of 06g's census checkpoint, read only. New in 06k.

One pass over var/runs/levin-phase-floor-06g-census-20261101T114315/checkpoint.jsonl (its SHA-256
checked against 03adcbba... first). Per unit: its byte offset and length in the file, the line's
SHA-256, the record's status, worker, wall_s, rebuild_s, recorded time, the number of group and
other calls, the number of unconverged group calls, and the region-pickle pointer. Writes OUT_JSON.
Usage: python -B index_06g.py OUT_JSON
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import refs_common as RC  # noqa: E402
from prov import CHECKPOINT06L_B_SHA256, RUN06L_B, header  # noqa: E402  prompt 06m
from prov import CHECKPOINT06M_SHA256, RUN06M  # noqa: E402  prompt 07

CHECKPOINT06G_SHA256 = "03adcbba8a78d82d9d6b899b8eee8b9ae1659763f52deb066d2840af3d82c0fa"


def main():
    out_p = Path(sys.argv[1])
    run = RUN06M  # prompt 07 (06m: RUN06L_B; 06l: RC.census_run_06g())
    ck = run / "checkpoint.jsonl"
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 07: an index of 06m's census checkpoint", [ck])  # prompt 06l: the title (06k: 06k); prompt 06m: the title
    for l in hdr:
        print(l)
    if RC.sha256(ck) != CHECKPOINT06M_SHA256:  # prompt 07 (06m: CHECKPOINT06L_B_SHA256)
        raise SystemExit("06m's checkpoint sha256 differs")
    units = {}
    off = 0
    with open(ck, "rb") as f:
        for line in f:
            r = json.loads(line)
            d = r["data"]
            uid = r["unit"]
            if uid in units:
                raise SystemExit(f"{uid} recorded twice")
            units[uid] = {
                "offset": off,
                "length": len(line),
                "line_sha256": hashlib.sha256(line).hexdigest(),
                "recorded": r.get("recorded"),
                "status": d.get("status"),
                "worker": d.get("worker"),
                "rtol": d.get("rtol"),
                "row": d.get("row"),
                "k_key": d.get("k_key"),
                "wall_s": d.get("wall_s"),
                "rebuild_s": d.get("rebuild_s"),
                "n_group_calls": len(d.get("calls") or []),
                "n_other_calls": len(d.get("other_calls") or []),
                "n_group_unconverged": sum(1 for c in (d.get("calls") or []) if not c.get("converged")),
                "n_cache_built": len(d.get("cache_built") or []),
                "pickle": [d.get("pickle"), d.get("pickle_offset"), d.get("pickle_length"), d.get("pickle_record_sha256")],
            }
            off += len(line)
    doc = {"header": hdr, "checkpoint": str(ck), "checkpoint_sha256": CHECKPOINT06M_SHA256,  # prompt 07
           "n_units": len(units), "units": units}
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(doc))
    walls = sorted(u["wall_s"] for u in units.values() if u["wall_s"] is not None)
    st = {}
    for u in units.values():
        st[u["status"]] = st.get(u["status"], 0) + 1
    print(f"units {len(units)}; statuses {st}; wall_s max {walls[-1]:.1f} p99 {walls[int(0.99 * len(walls))]:.1f} "
          f"median {walls[len(walls) // 2]:.2f} sum {sum(walls):.0f}")
    rb = sorted((u["rebuild_s"] or 0.0) for u in units.values())
    print(f"rebuild_s max {rb[-1]:.1f} p99 {rb[int(0.99 * len(rb))]:.2f} median {rb[len(rb) // 2]:.3f} sum {sum(rb):.0f}")
    print(f"wrote {out_p} sha256 {RC.sha256(out_p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
