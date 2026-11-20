"""Generate tabulate_qsi_r31_r32.py from 07b's rehearsal's tabulate_qsi_r30_r31.py (a7dc4941).

Changes, each asserted: the docstring (lines 1-39) rewritten; TREES, RUNS and the DEPTH_MAX comment
(lines 52-60) replaced; the store tags 31 -> 32 then 30 -> 31 shifted on every other line from 40 on,
outside the verbatim reader (lines 86-178, `# ---- copied verbatim` .. `# ---- end of the verbatim
copy ----`), by the regex (?<![0-9.])3([01])(?![0-9]). The verbatim block is asserted unchanged.
Usage: python -I -B make_tabulate_qsi_r31_r32.py SRC DST
"""
import hashlib, re, sys

src, dst = sys.argv[1:3]
L = open(src).read().split("\n")
assert hashlib.sha256(open(src, "rb").read()).hexdigest().startswith("a7dc4941")
assert L[85].startswith("# ---- copied verbatim") and L[177] == "# ---- end of the verbatim copy ----"
assert L[0].startswith('"""R31 against R30') and L[38] == '"""'
assert L[51] == "TREES = (" and L[59].startswith("DEPTH_MAX = 20")
DOC = '''"""R32 against R31 inside QuadSourceIntegral: tabulate every row that moved.

levin-phase-floor 07c, rehearsal agent, prompt 07c section 5 item 3 ("inside it, every moved row
tabulated, with the rows whose total_converged changes in each direction"). 07b's rehearsal
agent's tabulate_qsi_r30_r31.py (a7dc4941...), with the store tags 30 -> 31 and 31 -> 32 in every
name, label and column, TREES, RUNS, the DEPTH_MAX comment and this docstring changed, by
gen/make_tabulate_qsi_r31_r32.py; nothing else. Its store reader (`read`), `flatten`, `bits`,
`col_kind`, `leaf_kind` and `fields` are the verbatim copy (from 07's R29 tabulate_qsi_guard_r29.py,
47e18cbe..., from 06m's qsi_identity_r24_r25.py, 06g's), unchanged here, so the field
classification is 06g's:
- every column of table QuadSourceIntegral, in every shard, read with sqlite `mode=ro`, and every
  leaf of its `metadata` JSON (json.loads, flattened to a dotted path with [i] list indices);
- rows matched by their physical key (k, q, r, z_response) through the store's own wavenumber and
  redshift tables, with the tolerances (log10 atol, log10 rtol) carried and compared;
- a field is `timing`, `label`, `identity`, `counter` (WKB_Levin_num_regions, _evaluations,
  _simple_regions, _SVD_errors, _order_changes, _chebyshev_min_order, _max_depth,
  numeric_quad_compute_steps, numeric_quad_RHS_evaluations, and the metadata leaves regions,
  simple_regions, evaluations) or `value` (everything else);
- one more kind split out of `value`: `guard`, the four counts of prompt 07 section 1.6
  (stationary_points, window_pieces, guarded_regions, guard_at_depth_max), present in every Levin
  record (group and analytic block) of BOTH stores (ComputeTargets is the same at 0a103b5a and
  cc3368ce). Their presence in exactly the Levin records is asserted in both. The store does not
  keep 07b's refusal counts or 07c's chase-stop count: QuadSourceIntegral does not write them, and
  the field sets of the two stores are asserted equal as before.

A row is MOVED iff some value, counter or guard field differs (bit for bit: float.hex, else type
and value). Rows whose only differences are timing and label are UNMOVED. Every moved row is
tabulated into <rows csv>: total, total_abserr, |dtotal| against R31's and R32's total_abserr,
regions (WKB_Levin_num_regions), maximum depth (WKB_Levin_max_depth), total_converged, and the
four counts summed over the group records and over the analytic blocks, in each store; and
analytic_rad's change against analytic.abserr in each. Every Levin record that differs in any
non-timing field is written to <records csv> with both stores' fields. The rows whose
total_converged changes are listed in each direction, and so are group records and analytic
blocks whose `converged` changes.

Imports no tree code; reads the stores with sqlite3 mode=ro and store_value_diff's JSON (whose
QuadSourceIntegral row counts are checked against this script's).

    python -B tabulate_qsi_r31_r32.py <R31 primary> <R32 primary> <store_value_diff json> <rows csv> <records csv>
"""'''.split("\n")
HEAD = '''TREES = (
    "/Users/ds283/Documents/Code/SecondaryGWKit-lpf07c-cc3368c",  # R32's tree, cc3368ce (code 67b5af74)
)
RUNS = (
    "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07b-r31-07b-20261116T191849",
    "/Users/ds283/Documents/Code/SecondaryGWKit/var/runs/levin-phase-floor-07c-r32-07c-20261119T140417",
)
GUARD_LEAVES = ("stationary_points", "window_pieces", "guarded_regions", "guard_at_depth_max")
DEPTH_MAX = 20  # DEFAULT_LEVIN_MAX_DEPTH (unchanged by 67b5af74; levin_quadrature.py at cc3368ce)'''.split("\n")
assert L[58].startswith("GUARD_LEAVES")
pat = re.compile(r"(?<![0-9.])3([01])(?![0-9])")
out, n = [], 0
for i, line in enumerate(L):
    if i <= 38 or 51 <= i <= 59 or 85 <= i <= 177:
        out.append(line)
        continue
    new, k = pat.subn(lambda m: "3" + str(int(m.group(1)) + 1), line)
    n += k
    out.append(new)
out = DOC + out[39:51] + HEAD + out[60:]
s = "\n".join(out)
vb_old = "\n".join(L[85:178])
assert s.count(vb_old) == 1, "verbatim block changed"
open(dst, "w").write(s)
print("tag shifts", n, "lines", len(out))
