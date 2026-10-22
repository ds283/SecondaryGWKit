"""R4's four runs: manifest and final status, in run_records_03.txt's form. From the repo root."""

import glob, json, os, sys

sys.path.insert(0, "var/handover-06/scripts")
from after_launch import EXPECTED_PACKAGES, DISPATCH, LAUNCHER_SHA
import runregistrykit

h = runregistrykit.this_host()
print("== R4's four runs: manifest and final status")
for s in ("reference", "resumed-a", "resumed-b", "resumed-c"):
    d = sorted(glob.glob(f"var/runs/handover-06-{s}-*/"))[0].rstrip("/")
    m = json.load(open(f"{d}/manifest.json"))
    st = json.load(open(f"{d}/status.json"))
    argv = m["argv"]
    reg = argv[argv.index("--register") + 1]
    print(f"-- {os.path.basename(d)}")
    print(f"   purpose: {m['purpose']}")
    print(
        f"   created {m['created']}; argv[0] {argv[0]}; argv[1:5] {argv[1:5]}... --register {reg}; --resume {'--resume' in argv}"
    )
    print(f"   scope: {m['scope']}")
    print(
        f"   git_head {m['git_head']} (= dispatch: {m['git_head'] == DISPATCH}) git_dirty {m['git_dirty']} script {m['script']} script_sha256 {m['script_sha256'][:16]}... (= b2fa4ef1: {m['script_sha256'] == LAUNCHER_SHA}) results {m['results']} results_store_id {m['results_store_id']}"
    )
    print(f"   packages {json.dumps(m['packages'], sort_keys=True)}")
    print(
        f"   packages equal to the expected record: {m['packages'] == EXPECTED_PACKAGES}; keys {len(m)}"
    )
    print(
        f"   manifest host {json.dumps(m['host'], sort_keys=True)}; id/name = this_host(): {m['host']['id'] == h['id'] and m['host']['name'] == h['name']}"
    )
    print(
        f"   status: state {st['state']} exit_code {st['exit_code']} pid {st['pid']} heartbeat {st['heartbeat']} stage {st['stage']!r}"
    )
    print(
        f"   status host {json.dumps(st['host'], sort_keys=True)}; id/boot = this_host(): {st['host'] == {'id': h['id'], 'boot': h['boot']}}"
    )
    f = st.get("fingerprint") or {}
    print(
        f"   fingerprint format {f.get('fingerprint_format')} encoding_format {f.get('encoding_format')} digest {f.get('digest')} classes {len(f.get('classes', {}))} problems {f.get('problems')} taken {json.dumps(f.get('taken'), sort_keys=True)}"
    )
    print(f"   fingerprint_sidecar: {st.get('fingerprint_sidecar')}")
    print(f"   run dir files: {sorted(os.listdir(d))}")
