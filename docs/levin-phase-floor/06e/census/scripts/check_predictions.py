"""levin-phase-floor 06e (prompt 06e: copied from 06c's check_predictions.py; changed only in the prompt, 06e-exact-end-nodes.md, and its lines, 345-375: section 5's "Predictions, fixed before the run" and Z2-Z5). 06c's text: levin-phase-floor 06c: state/predictions.md's Part A against prompt 06c's lines 363-392 (section
5's "Predictions, fixed before the run" and X1-X5), byte for byte; and its SHA-256 and mtime.
Copied from 06b's check_predictions.py (kept in state/scripts_06b_reference/); changed (prompt
06c) only in the prompt, its lines and the tree it is read from.
Usage: python -B check_predictions.py"""
import datetime
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prov import G_TREE, WIDER, sha256  # noqa: E402  prompt 06c

prompt = G_TREE / "prompts" / "levin-phase-floor" / "06e-exact-end-nodes.md"  # prompt 06c; prompt 06e
lines = prompt.read_bytes().split(b"\n")
want = b"\n".join(lines[344:375]) + b"\n"  # prompt 06c: lines 363-392; prompt 06e: lines 345-375
pred = WIDER / "state" / "predictions.md"
data = pred.read_bytes()
a = data.index(b"## Part A")
start = data.index(b"\n\n", a) + 2
end = data.index(b"\n## Part B")
got = data[start:end]
got = got.rstrip(b"\n") + b"\n"
print(f"prompt {prompt} sha256 {sha256(prompt)}; lines 345-375: {len(want)} bytes")  # prompt 06e
print(f"predictions {pred}: Part A {len(got)} bytes; equal to the prompt's bytes: {got == want}")
st = os.stat(pred)
print(f"predictions sha256 {sha256(pred)} size {st.st_size} mtime "
      f"{datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat()}")
sys.exit(0 if got == want else 1)
