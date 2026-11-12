"""levin-phase-floor prompt 06k: every worker of the four censuses and the K0 runs, from its bye and hello records:
Ray never initialised, no module from a foreign root, the variant declared and in place. New in 06k. Read only.
Usage: python -B check_runs_after.py RUN_DIR [RUN_DIR ...]"""
import json, sys
from pathlib import Path
ok = True
for rd in map(Path, sys.argv[1:]):
    byes = sorted(rd.glob("worker_*_bye.json"))
    hellos = sorted(rd.glob("worker_*_hello.json"))
    n_ray = n_foreign = n_bad_variant = 0
    variants = set()
    for b in byes:
        m = json.loads(b.read_text())
        n_ray += bool(m["ray_initialized"])
        n_foreign += bool(m["modules_foreign"])
    for h in hellos:
        ev = json.loads(h.read_text())["evidence"]
        v = ev.get("variant_06k")
        variants.add(None if v is None else (v["name"], v["sha256"][:12]))
        n_bad_variant += not (ev["all_in_tree"] and ev["in_tree"]["AdaptiveLevin.levin_quadrature"])
    good = n_ray == 0 and n_foreign == 0 and n_bad_variant == 0 and len(byes) == len(hellos) and byes
    ok &= bool(good)
    print(f"{rd.name}: workers {len(hellos)} hello / {len(byes)} bye; ray initialised {n_ray}; with foreign modules {n_foreign}; "
          f"evidence failing {n_bad_variant}; variants {sorted(variants, key=str)}; {'OK' if good else 'NOT OK'}")
print("ALL OK" if ok else "NOT ALL OK")
sys.exit(0 if ok else 1)
