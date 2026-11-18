"""levin-phase-floor 07b section 5 (Part C), the R31 rehearsal agent: the descriptive addition the user asked for at stop 1
("settle which store understates"). New. Each capture of a group record of interest (out/refcap.json: the 7 records
whose |v31 - v30| exceeds e30 + e31, and the one whose converged changed), replayed from its pickle in this process at
one code, and compared bit for bit (float.hex) with a store's record:

- CODE ``h``: the tree's own AdaptiveLevin/levin_quadrature.py (0a103b5a, code 5e19c4b6, 8956c7b6...), against R31's
  record: the call's value and abserr times (1 + z_response) equal R31's stored value and abserr, and its regions and
  converged equal R31's; also the capture's own value, abserr, num_regions, max_depth and converged (what the harness
  recorded when it captured), and the refusal counts.
- CODE ``a07``: 07a's module (git show 1ec734d7:AdaptiveLevin/levin_quadrature.py, state/variants/
  levin_quadrature_1ec734d7.py, sha256 4ea1948b..., R30's tree's) installed in place of the tree's by Part A's
  variant_loader.py (unchanged), against R30's record in the same way. This shows that the captured integrand is
  R30's as well as R31's.

The call is ``adaptive_levin_sincos(x_span, f, theta, **kw)`` with the capture's own arguments. The pickle is loaded with
ray.cloudpickle; Ray is never initialised (checked). Writes OUT_JSON; refuses to overwrite.

Usage: PYTHONPATH=<tree> python -B refcap_replay.py h|a07 OUT_JSON
"""

import datetime
import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
Z = HERE.parents[1]  # var/levin-phase-floor-07b/rehearsal
TREE = Path("/Users/ds283/Documents/Code/SecondaryGWKit-lpf07b-0a103b5")
VARIANT = Z / "state" / "variants" / "levin_quadrature_1ec734d7.py"
VARIANT_SHA = "4ea1948b0a45b5e3e7028c360d8b6a3165680f747391adfdb4783b6646ca2577"
TREE_LQ_SHA = "8956c7b6a8ec285f8898353eed65f4039a02b42edc6927565be3896cda52def1"
REFCAP = Z / "out" / "refcap.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    code, out_p = sys.argv[1], Path(sys.argv[2])
    if out_p.exists():
        raise SystemExit(f"{out_p} exists")
    if code == "a07":
        sys.path.insert(0, str(HERE))
        import variant_loader

        variant_loader.install(VARIANT, VARIANT_SHA, "07a (1ec734d7)")
    elif code != "h":
        raise SystemExit("code is h or a07")
    import AdaptiveLevin
    import ray
    import ray.cloudpickle as cp

    lq = sys.modules["AdaptiveLevin.levin_quadrature"]
    head = subprocess.run(["git", "--no-optional-locks", "-C", str(TREE), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    doc = {"header": {"script": __file__, "script_sha256": sha256(__file__), "host": socket.gethostname(),
                      "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                      "tree": str(TREE), "tree_head": head, "code": code, "module_file": lq.__file__,
                      "module_sha256": sha256(lq.__file__), "refcap": str(REFCAP), "refcap_sha256": sha256(REFCAP)},
           "calls": []}
    if code == "h":
        assert doc["header"]["module_sha256"] == TREE_LQ_SHA, doc["header"]
    else:
        assert doc["header"]["module_sha256"] == VARIANT_SHA and variant_loader.evidence()["package_function_is_variants"]
        doc["header"]["variant_evidence"] = variant_loader.evidence()
    caps = json.loads(REFCAP.read_text())["captures"]
    ok_all = True
    for c in caps:
        blob = Path(c["pickle"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == c["sha256"], c["pickle"]
        call = cp.loads(blob)
        z = float(c["key"][3])
        t0 = time.time()
        res = AdaptiveLevin.adaptive_levin_sincos(call["x_span"], call["f"], call["theta"], **call["kw"])
        wall = time.time() - t0
        v, e = float(res["value"]), float(res["abserr"])
        md = int(max(r.depth for r in res["regions"]))
        store = c["R31"] if code == "h" else c["R30"]
        got = {"value": v.hex(), "abserr": e.hex(), "regions": int(res["num_regions"]), "max_depth": md,
               "converged": bool(res["converged"]),
               "stored_value": ((1.0 + z) * v).hex(), "stored_abserr": ((1.0 + z) * e).hex(),
               "refusals_07b": {k: res.get(k) for k in ("num_edge_rule_refusals", "num_range_rule_refusals")}}
        checks = {"stored_value": got["stored_value"] == store["value"], "stored_abserr": got["stored_abserr"] == store["abserr"],
                  "regions": got["regions"] == store["regions"], "converged": got["converged"] == store["converged"]}
        if code == "h":
            checks.update(capture_value=got["value"] == call["value"], capture_abserr=got["abserr"] == call["abserr"],
                          capture_regions=got["regions"] == call["num_regions"], capture_max_depth=md == call["max_depth"],
                          capture_converged=got["converged"] == call["converged"],
                          refusals_as_replay=got["refusals_07b"] == c["refusals_07b"])
        ok = all(checks.values())
        ok_all &= ok
        doc["calls"].append({"key": c["key"], "record": c["record"], "label": c["label"], "pickle": c["pickle"],
                             "sha256": c["sha256"], "notify_label": call["notify_label"], "x_span": [float(x).hex() for x in call["x_span"]],
                             "rtol": call["rtol"], "atol": call["atol"], "store": "R31" if code == "h" else "R30",
                             "store_record": store, "got": got, "checks": checks, "reproduces": ok, "wall_s": wall})
        print(f"{c['key']} {c['label']}: {'REPRODUCES' if ok else 'DOES NOT REPRODUCE'} {'R31' if code == 'h' else 'R30'}; "
              f"value {got['value']} abserr {got['abserr']} regions {got['regions']} depth {md} converged {got['converged']}; "
              f"refusals {got['refusals_07b']}; {wall:.1f} s; checks {checks}")
    doc["ray_initialized"] = ray.is_initialized()
    doc["all_reproduce"] = ok_all and not doc["ray_initialized"]
    out_p.write_text(json.dumps(doc, indent=1))
    print(f"ray initialised {doc['ray_initialized']}; all reproduce {doc['all_reproduce']}; wrote {out_p} sha256 {sha256(out_p)}")
    return 0 if doc["all_reproduce"] else 1


if __name__ == "__main__":
    sys.exit(main())
