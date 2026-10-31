"""levin-phase-floor prompt 06b section 5, W2: the calls whose inputs the census captures. New in 06b.

W2's sample is drawn, after the run, from the calls at ``rtol = 1e-7`` that were phase-limited in
06a, whose result moved under b, and with no sign change of theta' in 06a's scan (spscan.py). The
first and third conditions are known before the run, from 06a's checkpoint; the second only after.
So every call meeting the first and third is captured (its span, f, theta and arguments pickled by
the worker after the timed section), and the sample is drawn from them after the run by the rule
fixed in state/predictions.md.

A call is phase-limited in 06a if its checkpoint summary's ``phase_limited`` is true (06a recorded
group calls only). "No sign change" is 06a's scan record for the call (matched by ``index``) with
``n_sign_changes == 0``.

Writes state/w2_capture.json: {"header": ..., "capture": {"1e-7-<shard>-<serial>": [[index,
notify_label], ...]}, "n_calls": N}. Read-only outside it.

Usage: python -B make_capture.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import RUN06A, WIDER, header, sha256  # noqa: E402


def main():
    ck = RUN06A / "checkpoint.jsonl"
    capture = {}
    n = 0
    n_pl = 0
    n_scan_missing = 0
    with open(ck) as f:
        for line in f:
            d = json.loads(line)["data"]
            if d["rtol"] != "1e-7":
                continue
            scan = {s["index"]: s for s in (d.get("scan") or []) if s}
            for c in d["calls"]:
                if not c["phase_limited"]:
                    continue
                n_pl += 1
                s = scan.get(c["index"])
                if s is None:
                    n_scan_missing += 1
                    continue
                if s["n_sign_changes"] != 0:
                    continue
                key = f"1e-7-{d['shard']}-{d['serial']}"
                capture.setdefault(key, []).append([c["index"], c["notify_label"]])
                n += 1
    hdr = header(Path(__file__).resolve(), "levin-phase-floor 06b: W2's capture list", [ck])
    doc = {
        "header": hdr,
        "source": str(ck),
        "source_sha256": sha256(ck),
        "n_phase_limited_1e-7_calls_06a": n_pl,
        "n_scan_missing": n_scan_missing,
        "n_calls": n,
        "n_units": len(capture),
        "capture": capture,
    }
    out = WIDER / "state" / "w2_capture.json"
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.write_text(json.dumps(doc, indent=1))
    for l in hdr:
        print(l)
    print(f"06a phase-limited calls at 1e-7: {n_pl}; scan missing {n_scan_missing}")
    print(f"captured: {n} calls on {len(capture)} units (no sign change of theta' in 06a's scan)")
    print(f"wrote {out} sha256 {sha256(out)}")


if __name__ == "__main__":
    main()
