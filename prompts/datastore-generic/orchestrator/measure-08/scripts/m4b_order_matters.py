"""M4b: does the class order reach any output? Read build_full_store's inventory in the derived
order and in the plain dependency-sort order (inventory_classes patched), and compare the
fingerprint (taken removed), the report (plain and verbose) and the order of inventory.classes.
Run in the new tree. usage: python m4b_order_matters.py <scratch-dir>
"""
import json, shutil, sys
from pathlib import Path
from unittest import mock

scratch = Path(sys.argv[1]) / "m4b"
if scratch.exists():
    shutil.rmtree(scratch)
scratch.mkdir(parents=True)

import Datastore.store_inventory as si
from Datastore.tests.real_store_fixtures import build_full_store
from RunRegistry.stores import fingerprint_of
from config.datastore import factories
from tools.inventory_report import format_inventory_report

PLAIN = ['version', 'store_tag', 'redshift', 'wavenumber', 'tolerance', 'LambdaCDM', 'QCD_Cosmology',
         'wavenumber_exit_time', 'IntegrationSolver', 'BackgroundModel', 'TkNumericIntegration',
         'TkWKBIntegration', 'GkNumericIntegration', 'GkWKBIntegration', 'GkSourcePolicy', 'GkSource',
         'GkSourcePolicyData', 'QuadSourcePolicy', 'QuadSource', 'QuadSourceIntegral', 'OneLoopIntegral']
(scratch / "store").mkdir()
primary = build_full_store(scratch / "store").primary
meta = {"when": None, "git_head": None, "git_dirty": None, "run_id": None}
out = {}
for label, patch in (("derived", None), ("plain", PLAIN)):
    if patch is None:
        inv = si.read_inventory(primary, factories)
    else:
        with mock.patch.object(si, "inventory_classes", lambda f: list(PLAIN)):
            inv = si.read_inventory(primary, factories)
    fp = fingerprint_of(inv, meta)
    fp.pop("taken")
    out[label] = (json.dumps(fp, sort_keys=True), format_inventory_report(inv, "x"),
                  format_inventory_report(inv, "x", verbose=True), list(inv.classes))
a, b = out["derived"], out["plain"]
print("fingerprint identical:", a[0] == b[0])
print("report identical:", a[1] == b[1], " verbose identical:", a[2] == b[2])
print("inventory.classes order identical:", a[3] == b[3])
