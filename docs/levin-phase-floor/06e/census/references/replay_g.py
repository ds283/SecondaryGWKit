"""levin-phase-floor prompt 06e section 5, Z3: does each sample's captured input stand for g's census
call? New in 06e. For each sample in references/inputs/inputs.json: the pickle is loaded (ray's
cloudpickle, as orch_ref.py loads it) and the driver is replayed on it at the tree on PYTHONPATH, as
orch_ref.py replays it (``adaptive_levin_sincos(span, f, theta=theta, **kw)``, scaled by the
record's ``scale`` if it has one). The sample's g call is the census run's unit with the sample's
rtol, shard and serial and the same call index (``calls`` or ``other_calls``), with the same
notify_label and the same x_span (hex). The input stands for the g call iff the replay's value and
abserr equal that call's bit for bit (and its num_regions, max_depth and converged agree). 06c's
value and abserr are the captured record's (depth20-tiling's census_value, census_abserr).

Usage: PYTHONPATH=<tree g> python -B replay_g.py CENSUS_RUN_DIR [SAMPLE ...]   (one process, no Ray)
Writes nothing; prints one JSON line per sample after a header."""
import hashlib
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(path):
    import ray.cloudpickle as cp

    blob = Path(path).read_bytes()
    return cp.loads(blob), hashlib.sha256(blob).hexdigest()


def census_call(run_dir, uid, index):
    with open(Path(run_dir) / "checkpoint.jsonl") as f:
        for line in f:
            if f'"unit": "{uid}"' not in line[:200]:
                continue
            r = json.loads(line)
            if r["unit"] != uid:
                continue
            d = r["data"]
            for c in (d.get("calls") or []) + (d.get("other_calls") or []):
                if c["index"] == index:
                    return d, c
            return d, None
    return None, None


def main():
    run_dir = Path(sys.argv[1])
    want = {int(x) for x in sys.argv[2:]}
    import AdaptiveLevin.levin_quadrature as LQ

    tree = LQ.__file__.split("/AdaptiveLevin/")[0]
    git = ["git", "--no-optional-locks", "-C", tree]
    print("# host", socket.gethostname(), "utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    print("# tree", tree, subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
          "status lines", len(subprocess.run(git + ["status", "--short"], capture_output=True, text=True).stdout.splitlines()))
    print("# script sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "levin_quadrature sha256",
          hashlib.sha256(Path(LQ.__file__).read_bytes()).hexdigest())
    print("# census run", run_dir, "checkpoint sha256", hashlib.sha256((run_dir / "checkpoint.jsonl").read_bytes()).hexdigest())
    inputs = json.loads((HERE / "inputs" / "inputs.json").read_text())
    all_ok = True
    for key, e in sorted(inputs.items(), key=lambda kv: int(kv[0])):
        n = int(key)
        if want and n not in want:
            continue
        rec, sha = load(HERE / "inputs" / e["pickle"])
        if sha != e["sha256"]:
            raise SystemExit(f"s{n:02d}: pickle sha256 differs from inputs.json")
        span = tuple(rec["x_span"])
        rr = LQ.adaptive_levin_sincos(span, rec["f"], theta=rec["theta"], **rec["kw"])
        scale = rec.get("scale")
        v = rr["value"] if not scale else float.fromhex(scale) * rr["value"]
        a = rr["abserr"] if not scale else abs(float.fromhex(scale)) * rr["abserr"]
        _tree, rtol, shard, serial = e["unit"].split("-", 1)[0], *e["unit"].split("-", 1)[1].rsplit("-", 2)
        uid = f"g-{rtol}-{shard}-{serial}"
        d, c = census_call(run_dir, uid, e["index"])
        out = {"sample": n, "unit": uid, "index": e["index"], "scale": scale,
               "replay": {"value": float(v).hex(), "abserr": float(a).hex(), "num_regions": rr["num_regions"],
                          "max_depth": rr["max_depth"], "converged": bool(rr.get("converged"))},
               "census06c": {"value": e["census06c_value"], "abserr": e["census06c_abserr"]}}
        if c is None:
            out["g_call"] = None
            out["stands_for_g_call"] = False
            out["why"] = "unit or call not found" if d is None else f"unit status {d['status']}, call not found"
        else:
            same_call = c["notify_label"] == e["notify_label"] and [x["hex"] for x in c["x_span"]] == [
                float.fromhex(h).hex() for h in e["x_span"]]
            same_val = (c["value"]["hex"], c["abserr"]["hex"]) == (out["replay"]["value"], out["replay"]["abserr"])
            same_rest = (c["num_regions"], c["max_depth"], bool(c["converged"])) == (
                rr["num_regions"], rr["max_depth"], bool(rr.get("converged")))
            out["g_call"] = {"value": c["value"]["hex"], "abserr": c["abserr"]["hex"], "num_regions": c["num_regions"],
                             "max_depth": c["max_depth"], "converged": c["converged"], "unit_status": d["status"]}
            out["same_label_and_span"] = same_call
            out["stands_for_g_call"] = bool(same_call and same_val and same_rest)
        all_ok &= out["stands_for_g_call"]
        print(json.dumps(out), flush=True)
    print("# every sample's input stands for its g call:", all_ok)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
