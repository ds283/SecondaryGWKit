"""levin-phase-floor 06b: state/predictions.md's Part A against prompt 06b's lines 296-319, byte for
byte; and its SHA-256 and mtime. New in 06b. Usage: python -B check_predictions.py"""
import datetime
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import B_TREE, WIDER, sha256  # noqa: E402

prompt = B_TREE / "prompts" / "levin-phase-floor" / "06b-roundoff-exit-premise.md"
lines = prompt.read_bytes().split(b"\n")
want = b"\n".join(lines[295:319]) + b"\n"
pred = WIDER / "state" / "predictions.md"
data = pred.read_bytes()
a = data.index(b"## Part A")
start = data.index(b"\n\n", a) + 2
end = data.index(b"\n## Part B")
got = data[start:end]
got = got.rstrip(b"\n") + b"\n"
print(f"prompt {prompt} sha256 {sha256(prompt)}; lines 296-319: {len(want)} bytes")
print(f"predictions {pred}: Part A {len(got)} bytes; equal to the prompt's bytes: {got == want}")
st = os.stat(pred)
print(f"predictions sha256 {sha256(pred)} size {st.st_size} mtime "
      f"{datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat()}")
sys.exit(0 if got == want else 1)
