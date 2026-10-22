"""Prompt 07 §3: the one check after the launch. Run from the repository root."""

import glob, json, os, re, subprocess
import runregistrykit

d = sorted(glob.glob("var/runs/handover-07-a3-v2-*/"))[0].rstrip("/")
m = json.load(open(f"{d}/manifest.json"))
s = json.load(open(f"{d}/status.json"))
h = runregistrykit.this_host()
exp_pk = json.load(
    open(sorted(glob.glob("var/runs/handover-06-reference-*/manifest.json"))[0])
)["packages"]
print(f"== {runregistrykit.now_iso()} one check after the A3 v2 launch")
print("run dir", d, "; files", sorted(os.listdir(d)))
print("run_id", m["run_id"])
print("scope:", m["scope"])
sc = m["scope"]
for frag in (
    "full build:",
    "atol=1e-32 rtol=1e-07",
    "6 cpus",
    "LambdaCDM",
    "8 log-spaced k over 100000-3e+08 /Mpc",
    "handover-A3-v2-lambdacdm.sqlite",
):
    ok = sc.startswith(frag) if frag == "full build:" else frag in sc
    print(f"   scope {'begins' if frag == 'full build:' else 'names'} {frag!r}: {ok}")
print(
    "git_dirty",
    m["git_dirty"],
    "; git_head",
    m["git_head"],
    "= dispatch:",
    m["git_head"] == "b90752dda46e57d588f8107cbd4d4557134fadb2",
)
print(
    "host",
    json.dumps(m["host"], sort_keys=True),
    "; id/name = this_host() and 06's d120d1cf03551e6f:",
    m["host"]["id"] == h["id"] == "d120d1cf03551e6f",
    m["host"]["name"] == h["name"],
)
print("packages", json.dumps(m["packages"], sort_keys=True))
print(
    "packages equal to R4's manifests' (the pins):",
    m["packages"] == exp_pk,
    "; any editable:",
    any(v["editable"] for v in m["packages"].values()),
)
print(
    "script",
    m["script"],
    "; script_sha256",
    m["script_sha256"],
    "= 06's b2fa4ef1...:",
    m["script_sha256"]
    == "b2fa4ef1bbdbb09fea1780952710eeb53dee401ef6c0febcd385dd98aa0526ef",
)
print(
    "argv[1:]",
    m["argv"][1:12],
    "...; --rehearsal:",
    "--rehearsal" in m["argv"],
    "; --resume:",
    "--resume" in m["argv"],
)
print("purpose:", m["purpose"])
print(
    "results",
    m["results"],
    "; results_store_id",
    m["results_store_id"],
    "; manifest keys",
    len(m),
)
print(
    "status: state",
    s["state"],
    "pid",
    s["pid"],
    "stage",
    repr(s["stage"]),
    "heartbeat",
    s["heartbeat"],
)
print(
    "status host",
    json.dumps(s["host"], sort_keys=True),
    "= this_host():",
    s["host"] == {"id": h["id"], "boot": h["boot"]},
)
print(f"kill -0 {s['pid']}:", "ok" if runregistrykit.pid_alive(s["pid"]) else "FAILS")
print("liveness(status, 900):", runregistrykit.liveness(s, 900))
out = [
    re.sub(r"\x1b\[[0-9;]*m", "", l)
    for l in open(f"{d}/stdout.log", errors="replace").read().splitlines()
]
rep = [l for l in out if "replaced 1 occurrence" in l]
print("stdout.log lines", len(out), "; 'replaced 1 occurrence' lines:", len(rep))
for l in rep:
    print("   ", l[:200])
argv = [l for l in out if "main.py argv" in l]
print("'main.py argv' lines:", len(argv))
for l in argv:
    print("   ", l[:400])
if argv:
    print(
        "   has --zend 0.1:",
        bool(re.search(r"--zend['\",]*\s*['\"]?0\.1\b", argv[0])),
        "; has --no-prune-unvalidated:",
        "--no-prune-unvalidated" in argv[0],
    )
boot = [
    l
    for l in out
    if re.search(r"bootstrapping a local Ray instance|Started a local Ray instance", l)
]
print("local Ray bootstrap lines:", len(boot))
for l in boot:
    print("   ", l[:200])
lg = open("var/runs/handover-07-a3-v2.launch.log", errors="replace").read()
print(
    "launch log: 'Started a local Ray instance':",
    "Started a local Ray instance" in lg,
    "; a '!!' line:",
    any(l.startswith("!!") for l in lg.splitlines()),
    "; Traceback:",
    "Traceback" in lg,
)
print("stderr.log:", repr(open(f"{d}/stderr.log", errors="replace").read()[:300]))
ps = subprocess.run(
    ["ps", "-o", "pid,ppid,lstart,command", "-p", f"{s['pid']},84431"],
    capture_output=True,
    text=True,
).stdout
for l in ps.splitlines():
    print("  ", l[:260])
lst = subprocess.run(
    ["./venv/bin/python", "-m", "runregistrykit", "list"],
    capture_output=True,
    text=True,
)
print(
    "$ ./venv/bin/python -m runregistrykit list   (rc %d; stderr %r)"
    % (lst.returncode, lst.stderr)
)
print(lst.stdout)
