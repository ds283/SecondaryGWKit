"""
Prompt 10 §1 step 6: is the GkWKB read-back complete? Reads only the reader's output under
var/handover-10/extract/extract_GkWKB_data/ and its log; touches no store, starts no Ray.

1. Counts the full-range PDFs, PNGs and CSVs, per (k, z_source) pair, from the file names.
2. Reads the reader's work list from its log (the RayWorkPool "GENERATING GkNumeric/GkWKB DATA
   PRODUCTS" queue's own count lines) -- printed verbatim, for the reader of this output to check.
3. The pair 08 died on: whether its full-range PDF, PNG and CSV exist, and their sizes.
4. The plotted analytic_G_w points sensible_log_points dropped, recomputed from each CSV by calling
   extract_common.sensible_log_points itself on the CSV's (redshift, analytic_G_w) against its
   abs_G column -- the same two lists plot_Gk passes it (extract_GkWKB_data.py:269). For the pair,
   and summed over every CSV this run wrote (that is, over this run's sampled pairs, not the store).

Run from the repository root:  PYTHONPATH=. ./venv/bin/python var/handover-10/scripts/gkwkb_readback.py
"""

import math
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from extract_common import ANALYTIC_PLOT_DECADES, sensible_log_points

ROOT = Path("var/handover-10/extract/extract_GkWKB_data")
LOG = Path("var/handover-10/extract/extract_GkWKB_data.log")
MODEL = ROOT / "LambdaCDM"
PAIR = ("k=1e+05-k-serial=1", "zsource=1.3633e+16-z-serial=19")


def pairs_of(directory: Path, suffix: str):
    return {
        (p.parent.name, p.stem) for p in directory.glob(f"*/*{suffix}") if p.is_file()
    }


print(f"output root: {ROOT}")
print(f"files under it, by top-level kind and suffix:")
counts = Counter()
for p in ROOT.rglob("*"):
    if p.is_file():
        rel = p.relative_to(MODEL)
        counts[
            (rel.parts[0], rel.parts[1] if rel.parts[0] == "plots" else "", p.suffix)
        ] += 1
for key in sorted(counts):
    print(f"  {'/'.join(x for x in key[:2] if x):<24} {key[2]:<5} {counts[key]}")
print(f"  total files: {sum(counts.values())}")

full = MODEL / "plots" / "full-range"
csvd = MODEL / "csv"
pdf = pairs_of(full, ".pdf")
png = pairs_of(full, ".png")
csv = pairs_of(csvd, ".csv")
print()
print(f"full-range PDFs: {len(pdf)}")
print(f"full-range PNGs: {len(png)}")
print(f"CSVs:            {len(csv)}")
print(f"PDF pairs == PNG pairs: {pdf == png}; PDF pairs == CSV pairs: {pdf == csv}")
for name, a, b in (
    ("PDF-not-PNG", pdf, png),
    ("PDF-not-CSV", pdf, csv),
    ("CSV-not-PDF", csv, pdf),
):
    d = sorted(a - b)
    if d:
        print(f"  {name}: {len(d)}: {d[:10]}")
ks = sorted({k for k, _ in pdf})
zs = sorted({z for _, z in pdf})
print(f"distinct k in the full-range PDFs: {len(ks)}: {ks}")
print(f"distinct z_source in the full-range PDFs: {len(zs)}")

print()
print(f"the reader's log, {LOG}: its work-list lines (ANSI stripped)")
ansi = re.compile(r"\x1b\[[0-9;]*m")
for n, line in enumerate(LOG.read_text(errors="replace").splitlines(), 1):
    s = ansi.sub("", line)
    if re.search(
        r"GENERATING|Queue summary|work items|items|FATAL|Traceback|Error|RUNNING PIPELINE|SKIPPING",
        s,
    ):
        print(f"  {n}: {s[:240]}")

print()
print(f"the pair 08 died on: {PAIR[0]}/{PAIR[1]}")
for label, path in (
    ("full-range PDF", full / PAIR[0] / f"{PAIR[1]}.pdf"),
    ("full-range PNG", full / PAIR[0] / f"{PAIR[1]}.png"),
    (
        "zoom-matching PDF",
        MODEL / "plots" / "zoom-matching" / PAIR[0] / f"{PAIR[1]}.pdf",
    ),
    (
        "zoom-matching PNG",
        MODEL / "plots" / "zoom-matching" / PAIR[0] / f"{PAIR[1]}.png",
    ),
    ("theta PDF", MODEL / "plots" / "theta" / PAIR[0] / f"{PAIR[1]}.pdf"),
    ("CSV", csvd / PAIR[0] / f"{PAIR[1]}.csv"),
):
    print(
        f"  {label:<18} {'present' if path.is_file() else 'MISSING'}"
        f"{f', {path.stat().st_size} bytes' if path.is_file() else ''}  {path}"
    )


def dropped(path: Path):
    df = pd.read_csv(path)
    points = list(
        zip(df["redshift"], df["analytic_G_w"].where(df["analytic_G_w"].notna(), None))
    )
    reference = [
        None if (v is None or (isinstance(v, float) and math.isnan(v))) else v
        for v in df["abs_G"]
    ]
    kept = sensible_log_points(points, reference)
    n_none = sum(1 for _, y in points if y is None)
    n_nonfinite = sum(1 for _, y in points if y is not None and not math.isfinite(y))
    n_zero = sum(
        1 for _, y in points if y is not None and math.isfinite(y) and y == 0.0
    )
    ref = [abs(v) for v in reference if v is not None and math.isfinite(v) and v != 0.0]
    ceiling = max(ref) * 10.0**ANALYTIC_PLOT_DECADES if ref else None
    n_above = sum(
        1
        for _, y in points
        if y is not None
        and math.isfinite(y)
        and y != 0.0
        and ceiling is not None
        and abs(y) > ceiling
    )
    finite = [abs(y) for _, y in points if y is not None and math.isfinite(y)]
    return dict(
        rows=len(points),
        kept=len(kept),
        dropped=len(points) - len(kept),
        none=n_none,
        nonfinite=n_nonfinite,
        zero=n_zero,
        above_ceiling=n_above,
        ceiling=ceiling,
        max_abs_G=max(ref) if ref else None,
        max_abs_analytic_G_w=max(finite) if finite else None,
    )


print()
print(
    f"analytic_G_w points sensible_log_points drops (decades = {ANALYTIC_PLOT_DECADES}), from the CSVs"
)
pair_csv = csvd / PAIR[0] / f"{PAIR[1]}.csv"
if pair_csv.is_file():
    print(f"  the pair: {dropped(pair_csv)}")
else:
    print("  the pair: no CSV")
tot = Counter()
n_with_drop = 0
for path in sorted(csvd.glob("*/*.csv")):
    d = dropped(path)
    for key in (
        "rows",
        "kept",
        "dropped",
        "none",
        "nonfinite",
        "zero",
        "above_ceiling",
    ):
        tot[key] += d[key]
    if d["dropped"]:
        n_with_drop += 1
print(
    f"  every CSV this run wrote ({len(list(csvd.glob('*/*.csv')))} pairs, this run's random sample, not the whole store): "
    f"{dict(tot)}; pairs with at least one point dropped: {n_with_drop}"
)
sys.exit(0)
