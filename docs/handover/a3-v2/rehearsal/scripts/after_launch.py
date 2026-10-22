"""One check after a launch, in R3's prelaunch_checks_03.txt "one check after" form.

Usage, from the repository root:
    ./venv/bin/python var/handover-06/scripts/after_launch.py <slug> [<interrupter pid>]
Finds the newest var/runs/handover-06-<slug>-* directory; prints the manifest's provenance against
the expected record, the status, kill -0, ps of the driver and its sh, and the run's list row.
"""

import glob
import json
import os
import subprocess
import sys

import runregistrykit

EXPECTED_PACKAGES = {
    "datastorekit": {
        "commit": "f47b9a255e9ad58e0d0eaf85644f53b575884d31",
        "editable": False,
        "requested_revision": "v0.5.0",
        "url": "https://github.com/ds283/DatastoreKit",
        "version": "0.5.0",
    },
    "raypoolkit": {
        "commit": "eaef0ce60e2fc354bb32f0ad576010e6a83c1976",
        "editable": False,
        "requested_revision": "v0.2.0",
        "url": "https://github.com/ds283/RayPoolKit",
        "version": "0.2.0",
    },
    "runregistrykit": {
        "commit": "0bae3a3c501a8264ee4be5bc24f06422af38720f",
        "editable": False,
        "requested_revision": "v0.2.0",
        "url": "https://github.com/ds283/RunRegistryKit",
        "version": "0.2.0",
    },
}
DISPATCH = "3152af943a4f8dd5cf5920f7ed6b209f1c7e217b"
LAUNCHER_SHA = "b2fa4ef1bbdbb09fea1780952710eeb53dee401ef6c0febcd385dd98aa0526ef"
R3_MANIFEST = "var/handover-06/recorded/rrk-v020-adoption-03-reference-20261018T232838/manifest.json"


def main():
    slug = sys.argv[1]
    interrupter = int(sys.argv[2]) if len(sys.argv) > 2 else None
    dirs = sorted(
        d for d in glob.glob(f"var/runs/handover-06-{slug}-*") if os.path.isdir(d)
    )
    print("run dirs:", dirs)
    d = dirs[-1]
    print("run dir:", d)
    for name in sorted(os.listdir(d)):
        print("  ", name)
    m = json.load(open(os.path.join(d, "manifest.json")))
    s = json.load(open(os.path.join(d, "status.json")))
    host = runregistrykit.this_host()
    r3 = json.load(open(R3_MANIFEST))
    print(
        "manifest run_id",
        m.get("run_id"),
        "git_head",
        m.get("git_head"),
        "git_dirty",
        m.get("git_dirty"),
        "script",
        m.get("script"),
        "script_sha256",
        m.get("script_sha256"),
        "results_store_id",
        m.get("results_store_id"),
    )
    print("manifest results", m.get("results"))
    print("manifest packages", json.dumps(m.get("packages"), sort_keys=True))
    print(
        "packages equal as JSON to the expected record:",
        m.get("packages") == EXPECTED_PACKAGES,
    )
    print(
        "git_head is the dispatch SHA:",
        m.get("git_head") == DISPATCH,
        "; git_dirty false:",
        m.get("git_dirty") is False,
    )
    print("script_sha256 is b2fa4ef1...:", m.get("script_sha256") == LAUNCHER_SHA)
    new = sorted(set(m) - set(r3))
    missing = sorted(set(r3) - set(m))
    print(
        f"manifest keys {len(m)}; equal to R3's reference manifest's keys: {not new and not missing}; new {new}; missing {missing}"
    )
    mh = m.get("host") or {}
    print("manifest host", json.dumps(mh, sort_keys=True))
    print(
        "manifest host id/name equal this_host's:",
        mh.get("id") == host["id"],
        mh.get("name") == host["name"],
        "; keys",
        sorted(mh),
    )
    argv = m.get("argv") or []
    reg = argv[argv.index("--register") + 1] if "--register" in argv else None
    print(
        "argv[0]",
        argv[0] if argv else None,
        "; argv has --resume:",
        "--resume" in argv,
        "; --register",
        reg,
    )
    print(
        "status state",
        s.get("state"),
        "pid",
        s.get("pid"),
        "stage",
        s.get("stage"),
        "heartbeat",
        s.get("heartbeat"),
    )
    print("status keys", sorted(s))
    sh = s.get("host") or {}
    print("status host", json.dumps(sh, sort_keys=True))
    print(
        "status host id/boot equal this_host's:",
        sh.get("id") == host["id"],
        sh.get("boot") == host["boot"],
    )
    pid = s.get("pid")
    print(f"kill -0 {pid}", "ok" if runregistrykit.pid_alive(pid) else "FAILS")
    pids = [str(pid)]
    if interrupter:
        print(
            f"kill -0 interrupter {interrupter}",
            "ok" if runregistrykit.pid_alive(interrupter) else "FAILS",
        )
        pids.append(str(interrupter))
    ppid = subprocess.run(
        ["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    if ppid:
        pids.insert(0, ppid)
    out = subprocess.run(
        ["ps", "-o", "pid,ppid,lstart,command", "-p", ",".join(pids)],
        capture_output=True,
        text=True,
    ).stdout
    for line in out.splitlines():
        print("  " + line[:400])
    lst = subprocess.run(
        ["./venv/bin/python", "-m", "runregistrykit", "list"],
        capture_output=True,
        text=True,
    )
    for line in lst.stdout.splitlines():
        if os.path.basename(d) in line or "run(s) under" in line:
            print(line[:200])
    print("list rc", lst.returncode, "stderr:", repr(lst.stderr))
    print("liveness(status, 900):", runregistrykit.liveness(s, 900))
    print("this_host()", json.dumps(host, sort_keys=True))


if __name__ == "__main__":
    main()
