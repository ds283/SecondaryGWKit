"""levin-phase-floor 07a Part A, the Gate's hash clauses: 07's V4 and cross-check captures against 07's
census KEPT_ON_MAC_STUDIO.txt, and A3 v2's scratch copy against copy_files_certified.txt. Reads only;
writes OUT (argv[2]) with its own SHA-256 in the header. Usage: python -B gate_hashes.py LABEL OUT"""
import hashlib, json, sys, socket, datetime
from pathlib import Path
MAIN = Path("/Users/ds283/Documents/Code/SecondaryGWKit")
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07a-f93aafd")
KEPT = TREE / "docs/levin-phase-floor/07/census/KEPT_ON_MAC_STUDIO.txt"
CERT = MAIN / "var/levin-phase-floor-06/wider/state/copy_files_certified.txt"
COPY = MAIN / "var/levin-phase-floor-06/wider/state/a3v2_copy"
C07 = MAIN / "var/levin-phase-floor-07/census"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
def main():
    label, out = sys.argv[1], Path(sys.argv[2])
    res = {"header": {"script": __file__, "script_sha256": sha(__file__), "label": label, "host": socket.gethostname(),
                      "utc": datetime.datetime.utcnow().isoformat() + "Z", "kept_sha256": sha(KEPT), "cert_sha256": sha(CERT)},
           "captures": [], "copy": []}
    for line in KEPT.read_text().splitlines():
        p = line.split()
        if len(p) == 3 and (p[2].startswith("state/v4/capture/") or p[2].startswith("state/xcheck/capture/")):
            f = C07 / p[2]
            got = sha(f); size = f.stat().st_size
            res["captures"].append({"path": p[2], "want": p[0], "got": got, "size_want": int(p[1]), "size": size, "ok": got == p[0] and size == int(p[1])})
    for line in CERT.read_text().splitlines():
        if line.startswith("#"):
            continue
        p = line.split()
        f = COPY / p[4]
        st = f.stat()
        got = sha(f)
        res["copy"].append({"path": p[4], "want": p[0], "got": got, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                            "ok": got == p[0] and st.st_size == int(p[1]) and st.st_mtime_ns == int(p[2])})
    extra = sorted(x.name for x in COPY.iterdir() if x.name not in {c["path"] for c in res["copy"]})
    res["copy_extra_files"] = extra
    nc, nk = sum(c["ok"] for c in res["captures"]), sum(c["ok"] for c in res["copy"])
    res["summary"] = {"captures_ok": nc, "captures": len(res["captures"]), "copy_ok": nk, "copy": len(res["copy"]), "copy_extra": extra}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"]), "wrote", out, sha(out))
    return 0 if nc == len(res["captures"]) == 78 and nk == len(res["copy"]) == 5 and not extra else 1
if __name__ == "__main__":
    sys.exit(main())
