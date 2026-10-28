"""levin-phase-floor 06b: the run's scripts now, against those Gate W0's workers and the driver smoke
test recorded (each worker's hello evidence and each smoke unit's scripts_sha256). New in 06b.
Usage: python -B scripts_equal.py"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import WIDER, sha256  # noqa: E402

now = {p.name: sha256(p) for p in sorted(HERE.glob("*.py"))}
for name, h in now.items():
    print(f"{h}  {name}")
ok = True
for src in sorted((WIDER / "state" / "W0").glob("*/emit.jsonl")) + [WIDER / "state" / "smoke" / "ledger.jsonl"]:
    for line in open(src):
        m = json.loads(line)
        rec = m.get("evidence", {}).get("scripts_sha256") if m.get("kind") == "hello" else (m.get("data") or {}).get("scripts_sha256")
        if not rec:
            continue
        diff = {k: (v[:8], now.get(k, "absent")[:8]) for k, v in rec.items() if now.get(k) != v}
        ok &= not diff
        print(f"{src.relative_to(WIDER)} ({m.get('kind') or m.get('unit')}): {len(rec)} scripts; differing now: {diff or 'none'}")
print("ALL EQUAL" if ok else "SOME DIFFER")
