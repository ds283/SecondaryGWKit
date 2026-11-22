"""levin-phase-floor 07c Part A: the gate's hash clauses (prompt 07c Gate (Part A); orchestrator note A.0), read only.
Checks, by SHA-256 and size, against their KEPT_ON_MAC_STUDIO.txt lines:
- 07b's census run directory var/runs/levin-phase-floor-07b-census-20261117T010200/ (every file listed; no extra file),
  its checkpoint 396784bc...;
- var/levin-phase-floor-07b/census/out/stop3/compare_census_calls.calls.jsonl;
- 07b Part A's kept files (var/levin-phase-floor-07b/measurement/KEPT_ON_MAC_STUDIO.txt, paths relative to var/);
- R31's record capture and R31's store files (07b rehearsal KEPT, paths relative to the main checkout);
- A3 v2's scratch copy against copy_files_certified.txt (sha, size, mtime_ns; no extra file).
Writes OUT (argv[2]) as JSON with its own SHA-256 in the header. Usage: python -B gate_hashes_07c.py LABEL OUT"""
import hashlib, json, sys, socket, datetime
from pathlib import Path
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
VAR = MAIN / "var"
CK = VAR / "levin-phase-floor-07b/census/KEPT_ON_MAC_STUDIO.txt"
MK = VAR / "levin-phase-floor-07b/measurement/KEPT_ON_MAC_STUDIO.txt"
RK = VAR / "levin-phase-floor-07b/rehearsal/KEPT_ON_MAC_STUDIO.txt"
CERT = VAR / "levin-phase-floor-06/wider/state/copy_files_certified.txt"
COPY = VAR / "levin-phase-floor-06/wider/state/a3v2_copy"
RUN = "levin-phase-floor-07b-census-20261117T010200"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
def check(rows, path, want, size):
    p = Path(path)
    if not p.exists():
        rows.append({"path": str(p), "ok": False, "missing": True}); return
    got, sz = sha(p), p.stat().st_size
    rows.append({"path": str(p), "want": want, "got": got, "size_want": size, "size": sz, "ok": got == want and sz == size})
def main():
    label, out = sys.argv[1], Path(sys.argv[2])
    res = {"header": {"script": __file__, "script_sha256": sha(__file__), "label": label, "host": socket.gethostname(),
                      "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "kept_sha256": {str(k): sha(k) for k in (CK, MK, RK, CERT)}}}
    # census
    rows, listed = [], set()
    sec = None
    for line in CK.read_text().splitlines():
        if line.startswith("## "):
            sec = line; continue
        p = line.split()
        if len(p) != 3 or len(p[0]) != 64:
            continue
        if sec and "under var/levin-phase-floor-07b/census/" in sec and p[2] == "out/stop3/compare_census_calls.calls.jsonl":
            check(rows, VAR / "levin-phase-floor-07b/census" / p[2], p[0], int(p[1]))
        elif p[2].startswith(RUN + "/"):
            listed.add(p[2].split("/", 1)[1]); check(rows, VAR / "runs" / p[2], p[0], int(p[1]))
    have = {x.name for x in (VAR / "runs" / RUN).iterdir()}
    res["census"] = rows
    res["census_run_files_listed"] = len(listed); res["census_run_files_present"] = len(have)
    res["census_run_extra"] = sorted(have - listed); res["census_run_missing"] = sorted(listed - have)
    ck = [r for r in rows if r["path"].endswith(RUN + "/checkpoint.jsonl")]
    res["census_checkpoint"] = ck[0]["got"] if ck else None
    # 07b Part A kept
    rows = []
    for line in MK.read_text().splitlines():
        p = line.split()
        if len(p) == 3 and len(p[0]) == 64:
            check(rows, VAR / p[2], p[0], int(p[1]))
    res["parta"] = rows
    # rehearsal: R31 capture and R31 store files
    rows = []
    for line in RK.read_text().splitlines():
        p = line.split()
        if len(p) == 3 and len(p[0]) == 64 and (p[2].endswith("ref-G+q+r-h-3-20363-1.pkl") or p[2].startswith("var/datastores/levin-phase-floor-07b-r31")):
            check(rows, MAIN / p[2], p[0], int(p[1]))
    res["r31"] = rows
    # scratch copy
    rows = []
    for line in CERT.read_text().splitlines():
        if line.startswith("#"):
            continue
        p = line.split(); f = COPY / p[4]; st = f.stat(); got = sha(f)
        rows.append({"path": p[4], "want": p[0], "got": got, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                     "ok": got == p[0] and st.st_size == int(p[1]) and st.st_mtime_ns == int(p[2])})
    res["copy"] = rows
    res["copy_extra"] = sorted(x.name for x in COPY.iterdir() if x.name not in {c["path"] for c in rows})
    s = {k: [sum(r["ok"] for r in res[k]), len(res[k])] for k in ("census", "parta", "r31", "copy")}
    s.update(census_checkpoint=res["census_checkpoint"], census_run_extra=res["census_run_extra"],
             census_run_missing=res["census_run_missing"], census_run_files=[res["census_run_files_listed"], res["census_run_files_present"]],
             copy_extra=res["copy_extra"])
    res["summary"] = s
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(s), "wrote", out, sha(out))
    ok = all(v[0] == v[1] for k, v in s.items() if k in ("census", "parta", "r31", "copy")) and not res["census_run_extra"] \
        and not res["census_run_missing"] and not res["copy_extra"] and (res["census_checkpoint"] or "").startswith("396784bc")
    return 0 if ok else 1
if __name__ == "__main__":
    sys.exit(main())
