"""The policy-geometry census of a store's GkSourcePolicyData rows (handover prompt 03).

For every stored GkSourcePolicyData row it reports the source-grid geometry the policy assumed:
how many grid intervals `crossover_z` has between itself and `numeric_smallest_z` (numeric side)
and between itself and `primary_WKB_largest_z` (WKB side), counted by NODE INDEX in the stored
source grid (matched on the redshift `serial`, never by converting a log(1+z) width back to z),
together with the stored clearances, the type/quality census, the overlap's position in e-folds
against the three landmarks of campaign README 2 (q), the non-contiguity count and the exit-time
check.

Read-only and Ray-free, following docs/handover/a3-v2/seam/scripts/seam_census.py: the primary and
every shard are opened `mode=ro`; sqlite3, json and numpy only; nothing from this repository's
packages is imported. The constants it needs (MIN_SPLINE_DATA_POINTS, CLEARANCE_*,
CLASSIFICATION_BANDS, the integer codes of `type` and `quality`, SOURCE_GRID_CONSTRUCTION_VERSION,
DEFAULT_FLOAT_PRECISION) are read from the source TEXT with `ast.literal_eval` and recorded.

    ./venv/bin/python -I var/handover-03/scripts/policy_geometry.py <primary.sqlite>

Everything goes to stdout, as markdown. Exit status 0; 1 if a hard check fails (the type/quality
count disagrees with prompt 11's table, a `mixed` row's crossover_z matches no source-grid node
exactly, a policy row and its GkSource parent name different exit-time rows, ...). A failed check
is printed as a line beginning `!! STOP-CHECK`, and the census still runs to its end so that the
evidence is complete.
"""

import ast
import collections
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

# prompt 11's table (A3-V2-VERIFICATION.md 5.3), the cell-by-cell check of prompt 03 section 0.1
EXPECT_A3V2 = {
    ("numeric", "complete"): 363,
    ("mixed", "complete"): 71,
    ("mixed", "minimal"): 1,
    ("WKB", "complete"): 725,
}
# verification 5.5 (audit 4.2), 462 rows, the 2026-09 store that no longer exists
OLD_55 = {
    ("numeric", "complete"): 270,
    ("WKB", "complete"): 123,
    ("mixed", "complete"): 61,
    ("mixed", "minimal"): 1,
    ("fail", "incomplete"): 7,
}
OLD_55_TOTAL = 462

FAILED = []


def ro(path):
    return sqlite3.connect(f"file:{os.path.abspath(path)}?mode=ro", uri=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_assign(path, name):
    """The literal assigned to a module-level `name` in the source text of `path`."""
    with open(os.path.join(REPO, path)) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def stop_check(ok, message):
    if not ok:
        FAILED.append(message)
        print(f"!! STOP-CHECK FAILED: {message}")
    return ok


def g(x, n=4):
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{n}g}"


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")


def dist(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return None
    return [
        float(np.min(a)),
        float(np.percentile(a, 5)),
        float(np.median(a)),
        float(np.percentile(a, 95)),
        float(np.max(a)),
    ]


def dist_cells(values, n=4):
    d = dist(values)
    if d is None:
        return ["-"] * 5
    return [g(v, n) for v in d]


def ranks(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=float)
    groups = collections.defaultdict(list)
    for i, v in enumerate(a):
        groups[v].append(i)
    for idxs in groups.values():
        if len(idxs) > 1:
            r[idxs] = float(np.mean(r[idxs]))
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def decade(x):
    return int(math.floor(math.log10(x)))


def efold_function(ex):
    """N(z), e-folds sub-horizon (negative: super-horizon), by linear interpolation in log(1+z)
    through the exit-time row's stored z_exit_suph_e5..e1, z_exit, z_exit_subh_e1..e6. Only the
    direction z -> log(1+z) is used. Returns (N, flag) with flag 'lo' (N < -5, above the table),
    'hi' (N > 6, below the table) or ''."""
    knots = []
    for n in range(5, 0, -1):
        knots.append((-float(n), ex[f"z_exit_suph_e{n}"]))
    knots.append((0.0, ex["z_exit"]))
    for n in range(1, 7):
        knots.append((float(n), ex[f"z_exit_subh_e{n}"]))
    # ascending in log(1+z) means descending in N
    xs = [math.log(1.0 + z) for _, z in reversed(knots)]
    ns = [n for n, _ in reversed(knots)]

    def f(z):
        x = math.log(1.0 + z)
        if x > xs[-1]:
            return float("-inf"), "lo"
        if x < xs[0]:
            return float("inf"), "hi"
        return float(np.interp(x, xs, ns)), ""

    return f


def efold_cell(n, flag):
    if flag == "lo":
        return "<-5"
    if flag == "hi":
        return ">6"
    return f"{n:.3f}"


def main():
    t_start = time.time()
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    primary = os.path.abspath(sys.argv[1])
    store_dir = os.path.dirname(primary)

    # ---------------------------------------------------------------- constants read from text
    GKP = "ComputeTargets/GkSourcePolicyData.py"
    MIN_PTS = read_assign(GKP, "MIN_SPLINE_DATA_POINTS")
    CL_GOOD = read_assign(GKP, "CLEARANCE_GOOD")
    CL_ACC = read_assign(GKP, "CLEARANCE_ACCEPTABLE")
    CL_MARG = read_assign(GKP, "CLEARANCE_MARGINAL")
    BANDS = read_assign(GKP, "CLASSIFICATION_BANDS")
    FPF = "ObjectFactories/GkSourcePolicyData.py"
    TYPE = read_assign(FPF, "_type_deserialize")
    QUALITY = read_assign(FPF, "_quality_deserialize")
    SGC = read_assign(
        "CosmologyConcepts/wavenumber.py", "SOURCE_GRID_CONSTRUCTION_VERSION"
    )
    EPS = read_assign("config/defaults.py", "DEFAULT_FLOAT_PRECISION")

    # ---------------------------------------------------------------- header
    print("# POLICY-GEOMETRY CENSUS (handover prompt 03)")
    print()
    print("```")
    print(f"script           {os.path.abspath(__file__)}")
    print(f"script sha256    {sha256(__file__)}")
    head = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"git HEAD         {head}")
    print(f"python           {sys.version.split()[0]}   numpy {np.__version__}")
    p = ro(primary)
    shards = [
        os.path.join(store_dir, os.path.basename(f))
        for (f,) in p.execute("SELECT filename FROM shards ORDER BY serial")
    ]
    p.close()
    print(f"store            {primary}")
    print(f"  {sha256(primary)}  {os.path.basename(primary)}")
    for s in shards:
        print(f"  {sha256(s)}  {os.path.basename(s)}")
    sidecar = primary[: -len(".sqlite")] + ".manifest.json"
    print(f"  {sha256(sidecar)}  {os.path.basename(sidecar)}")
    with open(sidecar) as f:
        side = json.load(f)
    fp = side["fingerprint"]["digest"]
    print(f"store fingerprint (sidecar's) {fp}")
    print("constants read from source text (ast.literal_eval, nothing imported):")
    print(f"  {GKP}: MIN_SPLINE_DATA_POINTS = {MIN_PTS}")
    print(
        f"  {GKP}: CLEARANCE_GOOD / ACCEPTABLE / MARGINAL = {CL_GOOD} / {CL_ACC} / {CL_MARG}"
    )
    print(f"  {GKP}: CLASSIFICATION_BANDS (10):")
    for i, (q, props) in enumerate(BANDS):
        print(f"    band {i}: {q:<10} {props}")
    print(f"  {FPF}: _type_deserialize = {TYPE}")
    print(f"  {FPF}: _quality_deserialize = {QUALITY}")
    print(
        f"  CosmologyConcepts/wavenumber.py: SOURCE_GRID_CONSTRUCTION_VERSION = {SGC}"
    )
    print(f"  config/defaults.py: DEFAULT_FLOAT_PRECISION = {EPS}")
    print("```")
    stop_check(
        fp.startswith("0e32f24a"), f"the sidecar's fingerprint is {fp}, not 0e32f24a..."
    )
    stop_check(len(BANDS) == 10, "CLASSIFICATION_BANDS does not have ten bands")
    stop_check(SGC == 2, f"SOURCE_GRID_CONSTRUCTION_VERSION is {SGC}, not 2")

    # band -> quality, and quality -> the band indices it implies
    band_quality = [q for q, _ in BANDS]
    quality_bands = collections.defaultdict(list)
    for i, q in enumerate(band_quality):
        quality_bands[q].append(i)

    # ---------------------------------------------------------------- the grid and the shared tables
    c0 = ro(shards[0])
    zmap = dict(c0.execute("select serial, z from redshift"))
    source_flag = dict(c0.execute("select serial, source from redshift"))
    tolmap = dict(c0.execute("select serial, log10_tol from tolerance"))
    ex_cols = (
        ["serial", "wavenumber_serial", "atol_serial", "rtol_serial", "z_exit"]
        + [f"z_exit_suph_e{n}" for n in range(1, 6)]
        + [f"z_exit_subh_e{n}" for n in range(1, 7)]
    )
    exits = {}
    for row in c0.execute(
        f"select {', '.join(ex_cols)}, "
        "(select k_inv_Mpc from wavenumber w where w.serial = e.wavenumber_serial) "
        "from wavenumber_exit_time e"
    ):
        d = dict(zip(ex_cols + ["k"], row))
        exits[d["serial"]] = d
    c0.close()
    order = sorted(zmap, key=lambda s: zmap[s])
    gidx = {s: i for i, s in enumerate(order)}
    n_grid = len(order)

    # the replicated tables are replicated: check
    same_grid = True
    for i, sh in enumerate(shards[1:], start=1):
        c = ro(sh)
        same_grid &= dict(c.execute("select serial, z from redshift")) == zmap
        same_exit = {
            r[0]: r[1:]
            for r in c.execute(
                "select serial, wavenumber_serial, atol_serial, rtol_serial, z_exit, "
                + ", ".join(ex_cols[5:])
                + " from wavenumber_exit_time"
            )
        } == {s: tuple(d[k] for k in ex_cols[1:]) for s, d in exits.items()}
        same_grid &= same_exit
        c.close()
    stop_check(same_grid, "redshift / wavenumber_exit_time differ between shards")

    print()
    print("## 0. The grid, the grid construction and the exit-time rows")
    print()
    print(
        f"redshift rows: {n_grid}; flagged as source nodes: {sum(1 for s in source_flag.values() if s)}; "
        f"ascending-z order differs from serial order: {order != sorted(zmap)}. z from "
        f"{zmap[order[0]]:.6g} to {zmap[order[-1]]:.6g}. The shards agree on the redshift and "
        f"wavenumber_exit_time tables: {same_grid}."
    )
    steps = np.diff([math.log(1.0 + zmap[s]) for s in order])
    print(
        f"Source grid steps in log(1+z), all {n_grid - 1} intervals: min {steps.min():.4e}, "
        f"median {np.median(steps):.4e}, mean {steps.mean():.4e}, max {steps.max():.4e}."
    )
    print()
    print(
        "Exit-time rows (the 8 wavenumbers; atol/rtol are the exit-time row's own pair):"
    )
    print()
    table(
        [
            "exit serial",
            "k (/Mpc)",
            "atol serial (log10)",
            "rtol serial (log10)",
            "z_e3",
            "sqrt(z_e3 z_e4)",
            "z_e4",
            "z_e6",
        ],
        [
            [
                s,
                f"{d['k']:.6e}",
                f"{d['atol_serial']} ({tolmap[d['atol_serial']]:g})",
                f"{d['rtol_serial']} ({tolmap[d['rtol_serial']]:g})",
                f"{d['z_exit_subh_e3']:.6e}",
                f"{math.sqrt(d['z_exit_subh_e3'] * d['z_exit_subh_e4']):.6e}",
                f"{d['z_exit_subh_e4']:.6e}",
                f"{d['z_exit_subh_e6']:.6e}",
            ]
            for s, d in sorted(exits.items(), key=lambda kv: kv[1]["k"])
        ],
    )
    print()
    print(
        "Is a step of log(1+z) an e-fold? The log((1+z_eN)/(1+z_e(N+1))) of each wavenumber, N = 3, 4, 5:"
    )
    print()
    table(
        ["k (/Mpc)", "e3 -> e4", "e4 -> e5", "e5 -> e6", "exit -> e1"],
        [
            [
                f"{d['k']:.6e}",
                f"{math.log((1 + d['z_exit_subh_e3']) / (1 + d['z_exit_subh_e4'])):.5f}",
                f"{math.log((1 + d['z_exit_subh_e4']) / (1 + d['z_exit_subh_e5'])):.5f}",
                f"{math.log((1 + d['z_exit_subh_e5']) / (1 + d['z_exit_subh_e6'])):.5f}",
                f"{math.log((1 + d['z_exit']) / (1 + d['z_exit_subh_e1'])):.5f}",
            ]
            for s, d in sorted(exits.items(), key=lambda kv: kv[1]["k"])
        ],
    )

    # ---------------------------------------------------------------- read the rows
    rows = []
    tagsets = collections.Counter()
    model_constructions = collections.Counter()
    parent_checks = collections.Counter()
    policy_serials = collections.Counter()
    node_set_bad = []
    for si, sh in enumerate(shards):
        c = ro(sh)
        # tag sets of the GkSource rows, and the construction integer on the BackgroundModel
        tags = collections.defaultdict(list)
        for ps, label in c.execute(
            "select t.parent_serial, s.label from GkSource_tags t join store_tag s on s.serial = t.tag_serial"
        ):
            tags[ps].append(label)
        models = dict(
            c.execute("select serial, source_grid_construction from BackgroundModel")
        )
        # exit-time agreement of the producers: every GkSource_parents integration row's
        # wavenumber_exit_serial against its GkSource's
        for kind, tbl, col in (
            ("numeric", "GkNumericIntegration", "numeric_serial"),
            ("WKB", "GkWKBIntegration", "wkb_serial"),
        ):
            bad, n = c.execute(
                f"select sum(case when i.wavenumber_exit_serial != s.wavenumber_exit_serial then 1 else 0 end), count(*) "
                f"from GkSource_parents p join GkSource s on s.serial = p.gk_source_serial "
                f'join "{tbl}" i on i.serial = p.{col}'
            ).fetchone()
            parent_checks[(kind, "rows")] += n
            parent_checks[(kind, "exit differs from the GkSource's")] += bad or 0
        query = (
            "select d.serial, d.source_serial, d.policy_serial, d.wavenumber_exit_serial, d.type, d.quality, "
            "d.crossover_z, d.metadata, s.wavenumber_exit_serial, s.z_response_serial, s.z_max_serial, "
            "s.z_samples, s.numeric_smallest_z_serial, s.primary_WKB_largest_z_serial, s.model_serial "
            "from GkSourcePolicyData d join GkSource s on s.serial = d.source_serial order by d.serial"
        )
        for r in c.execute(query).fetchall():
            (
                dser,
                sser,
                pol,
                dex,
                ty,
                ql,
                xz,
                md,
                sex,
                zr,
                zmaxs,
                zsamp,
                ns_ser,
                wl_ser,
                model_ser,
            ) = r
            policy_serials[pol] += 1
            tagsets[tuple(sorted(tags[sser]))] += 1
            model_constructions[models[model_ser]] += 1
            vals = c.execute(
                "select z_source_serial, (G is not null and Gprime is not null), (theta_mod_2pi is not null) "
                "from GkSourceValue where parent_serial = ?",
                (sser,),
            ).fetchall()
            vals.sort(key=lambda v: gidx[v[0]])
            nodes = [v[0] for v in vals]
            has_num = [bool(v[1]) for v in vals]
            has_wkb = [bool(v[2]) for v in vals]
            # the GkSource's source grid must be the slice of the global grid from z_response up
            expected_nodes = order[gidx[zr] :]
            if nodes != expected_nodes or zsamp != len(nodes):
                node_set_bad.append((si, sser))
            rows.append(
                dict(
                    shard=si,
                    serial=dser,
                    source_serial=sser,
                    policy_serial=pol,
                    exit_policy=dex,
                    exit_source=sex,
                    type=TYPE[ty],
                    quality=QUALITY[ql],
                    crossover_z=xz,
                    metadata=json.loads(md) if md else {},
                    z_resp_serial=zr,
                    z_max_serial=zmaxs,
                    ns_ser=ns_ser,
                    wl_ser=wl_ser,
                    nodes=nodes,
                    has_num=has_num,
                    has_wkb=has_wkb,
                )
            )
        c.close()

    N = len(rows)
    print()
    print("## 1. What was read, and which grid it was taken on")
    print()
    print(
        f"GkSourcePolicyData rows read from the shards (mode=ro): {N}; by shard {[sum(1 for r in rows if r['shard'] == i) for i in range(len(shards))]}"
    )
    print(f"policy_serial values: {dict(policy_serials)}")
    stop_check(N == 1160, f"{N} policy rows, not 1160")
    print()
    print("Tag sets carried by the GkSource parents (from GkSource_tags / store_tag):")
    print()
    table(
        ["rows", "tags"],
        [[n, ", ".join(t)] for t, n in tagsets.most_common()],
    )
    print()
    print(
        f"`BackgroundModel.source_grid_construction` of each row's GkSource.model_serial: {dict(model_constructions)}."
    )
    stop_check(
        all("SourceGridConstruction_2" in t for t in tagsets)
        and len(tagsets) == 1
        and list(model_constructions) == [2],
        "the rows do not all carry SourceGridConstruction_2 / construction 2",
    )
    # the sidecar's tag set for GkSource, cross-checked
    side_tags = side["fingerprint"]["classes"]["GkSource"]["tag_sets"]
    print()
    print(
        "Sidecar's fingerprint tag sets for the GkSource class: "
        + "; ".join(f"{t['count']} rows with {t['tags']}" for t in side_tags)
    )
    print()
    print(
        f"Each GkSource's source grid (its value rows' z_source serials, sorted by z) is the slice of the "
        f"{n_grid}-node grid from its z_response upward, and `z_samples` equals its length: "
        f"{'yes, all rows' if not node_set_bad else 'NO: ' + str(node_set_bad[:10])}."
    )
    stop_check(
        not node_set_bad,
        "a GkSource's grid is not the slice of the global grid from z_response up",
    )

    # ---------------------------------------------------------------- per-row geometry
    for r in rows:
        ex = exits[r["exit_policy"]]
        r["k"] = ex["k"]
        r["ex"] = ex
        r["efold"] = efold_function(ex)
        zr = r["z_resp_serial"]
        r["z_resp"] = zmap[zr]
        r["gi"] = [gidx[s] for s in r["nodes"]]  # global node indices, ascending
        base = gidx[zr]
        r["idx"] = {
            s: gidx[s] - base for s in r["nodes"]
        }  # index in the GkSource's own z_sample
        nn = len(r["nodes"])
        r["nn"] = nn
        # recompute the producer's summary redshifts from the stored value rows
        num_idx = [i for i in range(nn) if r["has_num"][i]]
        r["i_num_re"] = num_idx[0] if num_idx else None
        i_w = None
        if nn and r["has_wkb"][0]:
            i_w = 0
            while i_w + 1 < nn and r["has_wkb"][i_w + 1]:
                i_w += 1
        r["i_wkb_re"] = i_w
        r["i_num"] = r["idx"][r["ns_ser"]] if r["ns_ser"] is not None else None
        r["i_wkb"] = r["idx"][r["wl_ser"]] if r["wl_ser"] is not None else None
        r["z_num"] = zmap[r["ns_ser"]] if r["ns_ser"] is not None else None
        r["z_wkb"] = zmap[r["wl_ser"]] if r["wl_ser"] is not None else None
        # crossover: exact equality with a node's stored z
        r["i_x"] = None
        if r["crossover_z"] is not None:
            hits = [i for i, s in enumerate(r["nodes"]) if zmap[s] == r["crossover_z"]]
            r["i_x"] = hits[0] if len(hits) == 1 else (-1 if not hits else -2)
        # the WKB-hole census (the warning of assemble_GkSource_values, GkSource.py)
        holes = 0
        seen = False
        prev = None
        hole_start = None
        first_hole = None
        for i in range(nn):
            if r["has_wkb"][i]:
                if prev is not True and seen:
                    holes += 1
                    if first_hole is None:
                        first_hole = (hole_start, i)
                seen = True
                prev = True
            else:
                if prev is True:
                    hole_start = i
                prev = False
        r["wkb_hole_events"] = holes
        r["first_hole"] = (
            first_hole  # (lowest node without WKB, lowest node of the island)
        )
        r["no_primary"] = not (nn and r["has_wkb"][0])
        top_wkb = max([i for i in range(nn) if r["has_wkb"][i]], default=None)
        r["wkb_island"] = (
            i_w is not None and top_wkb is not None and top_wkb > i_w
        ) or (i_w is None and top_wkb is not None)
        r["z_lt_e4"] = r["z_resp"] < r["ex"]["z_exit_subh_e4"]

    # ---------------------------------------------------------------- 2. type / quality census
    print()
    print("## 2. The type/quality census, checked against prompt 11's table")
    print()
    cells = collections.Counter((r["type"], r["quality"]) for r in rows)
    ok_cells = True
    out_rows = []
    for key in sorted(set(cells) | set(EXPECT_A3V2)):
        a, b = EXPECT_A3V2.get(key, 0), cells.get(key, 0)
        ok_cells &= a == b
        out_rows.append(
            [key[0], key[1], a, b, f"{100.0 * b / N:.2f} %", "yes" if a == b else "NO"]
        )
    out_rows.append(
        [
            "",
            "**total**",
            sum(EXPECT_A3V2.values()),
            N,
            "",
            "yes" if sum(EXPECT_A3V2.values()) == N else "NO",
        ]
    )
    print("<!-- section:typequality -->")
    table(
        [
            "type",
            "quality",
            "prompt 11 (A3-V2-VERIFICATION 5.3)",
            "this census",
            "share",
            "agree",
        ],
        out_rows,
    )
    print("<!-- end:typequality -->")
    print()
    print(f"EVERY CELL AGREES: {ok_cells}")
    stop_check(
        ok_cells and sum(EXPECT_A3V2.values()) == N,
        "the type/quality count disagrees with prompt 11's table",
    )
    n_fail_inc = sum(
        1 for r in rows if r["type"] == "fail" or r["quality"] == "incomplete"
    )
    print(f"`fail` or `incomplete` rows: {n_fail_inc}.")

    # comparison of shape against verification 5.5
    print()
    print(
        "### 2.1 The shape, set beside verification 5.5 (462 rows, a store that no longer exists)"
    )
    print()
    print("<!-- section:shape -->")

    def shape_rows(subset):
        n = len(subset)
        c = collections.Counter((r["type"], r["quality"]) for r in subset)
        keys = sorted(set(c) | set(OLD_55), key=lambda kk: -OLD_55.get(kk, 0))
        out = []
        for kk in keys:
            old = OLD_55.get(kk, 0)
            new = c.get(kk, 0)
            out.append(
                [
                    kk[0],
                    kk[1],
                    old,
                    f"{100.0 * old / OLD_55_TOTAL:.2f} %",
                    new,
                    f"{100.0 * new / n:.2f} %" if n else "-",
                ]
            )
        out.append(["", "**total**", OLD_55_TOTAL, "100 %", n, "100 %"])
        return out

    hdr = ["type", "quality", "5.5 rows", "5.5 share", "A3 v2 rows", "A3 v2 share"]
    print("**(a) The full store: k from 1e5 to 3e8 /Mpc.**")
    print()
    table(hdr, shape_rows(rows))
    restricted = [r for r in rows if 1e5 <= r["k"] <= 1e7]
    print()
    print(
        "**(b) Restricted to 1e5 <= k <= 1e7 /Mpc** (removes difference 3, the wider k span): "
        f"k values {sorted({r['k'] for r in restricted})}."
    )
    print()
    table(hdr, shape_rows(restricted))
    print()
    print(
        "**(c) By k decade, across the full span** (counts of numeric / mixed / WKB rows, and mixed rows' share):"
    )
    print()
    dec_rows = []
    for dk in sorted({decade(r["k"]) for r in rows}):
        sub = [r for r in rows if decade(r["k"]) == dk]
        c = collections.Counter(r["type"] for r in sub)
        dec_rows.append(
            [
                f"1e{dk} <= k < 1e{dk + 1}",
                len(sub),
                c.get("numeric", 0),
                c.get("mixed", 0),
                c.get("WKB", 0),
                c.get("fail", 0),
                f"{100.0 * c.get('mixed', 0) / len(sub):.2f} %",
                sorted({round(r["k"], 1) for r in sub}),
            ]
        )
    table(
        [
            "k decade",
            "rows",
            "numeric",
            "mixed",
            "WKB",
            "fail",
            "mixed share",
            "k values",
        ],
        dec_rows,
    )
    print()
    print(
        "**(d) By individual k** (type shares; z_response runs over the same 145-node response grid for every k):"
    )
    print()
    k_rows = []
    for k in sorted({r["k"] for r in rows}):
        sub = [r for r in rows if r["k"] == k]
        c = collections.Counter(r["type"] for r in sub)
        k_rows.append(
            [
                f"{k:.6e}",
                len(sub),
                c.get("numeric", 0),
                c.get("mixed", 0),
                c.get("WKB", 0),
                f"{100.0 * c.get('numeric', 0) / len(sub):.1f} %",
                f"{100.0 * c.get('mixed', 0) / len(sub):.1f} %",
                f"{100.0 * c.get('WKB', 0) / len(sub):.1f} %",
                f"{min(r['z_resp'] for r in sub):.3g} .. {max(r['z_resp'] for r in sub):.3g}",
            ]
        )
    table(
        [
            "k (/Mpc)",
            "rows",
            "numeric",
            "mixed",
            "WKB",
            "numeric %",
            "mixed %",
            "WKB %",
            "z_response range",
        ],
        k_rows,
    )
    print()
    print(
        "**(e) The type mix over z_response decade, restricted to 1e5 <= k <= 1e7** (numeric / mixed / WKB):"
    )
    print()
    zdecs = sorted({decade(r["z_resp"]) for r in rows})
    ze_rows = []
    for dz in zdecs:
        sub = [r for r in restricted if decade(r["z_resp"]) == dz]
        if not sub:
            continue
        c = collections.Counter(r["type"] for r in sub)
        ze_rows.append(
            [
                f"1e{dz}",
                len(sub),
                c.get("numeric", 0),
                c.get("mixed", 0),
                c.get("WKB", 0),
            ]
        )
    table(["z_response decade", "rows", "numeric", "mixed", "WKB"], ze_rows)
    print()
    print(
        "**(f) Mixed, numeric and WKB rows per k, against what 5.5's 462 rows give if its seven k carried equal counts** "
        "(462 = 7 x 66; 5.5's TkWKB table quotes seven k, 1e5 to 1e7 /Mpc in steps of 2.154; that every k carried 66 response "
        "redshifts is an inference from the division, not a recorded fact):"
    )
    print()
    nk_old = 7
    per_k_old = {kk: v / nk_old for kk, v in OLD_55.items()}
    nk_new = len({r["k"] for r in restricted})
    c_restr = collections.Counter((r["type"], r["quality"]) for r in restricted)
    pk_rows = []
    for kk in (
        ("numeric", "complete"),
        ("mixed", "complete"),
        ("mixed", "minimal"),
        ("WKB", "complete"),
        ("fail", "incomplete"),
    ):
        pk_rows.append(
            [
                kk[0],
                kk[1],
                f"{per_k_old.get(kk, 0):.2f}",
                f"{c_restr.get(kk, 0) / nk_new:.2f}",
            ]
        )
    pk_rows.append(
        [
            "",
            "**all rows per k**",
            f"{OLD_55_TOTAL / nk_old:.2f}",
            f"{len(restricted) / nk_new:.2f}",
        ]
    )
    table(
        [
            "type",
            "quality",
            "5.5 rows per k (7 k)",
            f"A3 v2 rows per k ({nk_new} k, 1e5 <= k <= 1e7)",
        ],
        pk_rows,
    )
    print()
    print(
        "**(g) A probe, not a re-baseline: the same restriction and z_response >= 1e7.** 5.5's own TkWKB table quotes sample ranges "
        "that start at z = 1e7 for every k, which suggests (it does not establish) that the 2026-09 store's grid began there; its "
        "Gk response range is not recorded. The A3 v2 rows with 1e5 <= k <= 1e7 and z_response >= 1e7:"
    )
    print()
    probe = [r for r in restricted if r["z_resp"] >= 1e7]
    table(hdr, shape_rows(probe))
    print("<!-- end:shape -->")

    # ---------------------------------------------------------------- 3. geometry of every mixed row
    mixed = [r for r in rows if r["type"] == "mixed"]
    stop_check(len(mixed) == 72, f"{len(mixed)} mixed rows, not 72")
    nomatch = [r for r in mixed if r["i_x"] is None or r["i_x"] < 0]
    stop_check(
        not nomatch,
        f"{len(nomatch)} mixed rows have a crossover_z that matches no source-grid node exactly "
        f"(or more than one): {[(r['shard'], r['serial']) for r in nomatch][:10]}",
    )
    exit_disagree = [r for r in rows if r["exit_policy"] != r["exit_source"]]
    stop_check(
        not exit_disagree,
        f"{len(exit_disagree)} policy rows name a different exit-time row from their GkSource parent",
    )

    for r in rows:
        r["recomputed_ok"] = (r["i_num"] == r["i_num_re"]) and (
            r["i_wkb"] == r["i_wkb_re"]
        )

    for r in mixed:
        zn, zw, xz = r["z_num"], r["z_wkb"], r["crossover_z"]
        ln, lw, lx = math.log(1.0 + zn), math.log(1.0 + zw), math.log(1.0 + xz)
        r["numeric_intervals"] = r["i_x"] - r["i_num"]
        r["WKB_intervals"] = r["i_wkb"] - r["i_x"]
        r["overlap_intervals"] = r["i_wkb"] - r["i_num"]
        r["overlap_log_width"] = lw - ln
        r["c_num"] = (lx - ln) / ln
        r["c_WKB"] = (lw - lx) / lw
        r["L_num"], r["L_WKB"] = ln, lw
        r["step_num"] = (
            (lx - ln) / r["numeric_intervals"]
            if r["numeric_intervals"]
            else float("nan")
        )
        r["step_WKB"] = (
            (lw - lx) / r["WKB_intervals"] if r["WKB_intervals"] else float("nan")
        )
        # spline data points the policy counted
        r["n_num_pts"] = sum(r["has_num"][r["i_num"] :])
        r["n_wkb_pts"] = sum(r["has_wkb"][: r["i_wkb"] + 1])
        # holes in the numeric / WKB side between the nodes counted
        r["holes_num_side"] = sum(
            1 for i in range(r["i_num"], r["i_x"] + 1) if not r["has_num"][i]
        )
        r["holes_wkb_side"] = sum(
            1 for i in range(r["i_x"], r["i_wkb"] + 1) if not r["has_wkb"][i]
        )
        r["overlap_nodes_both"] = sum(
            1
            for i in range(r["i_num"], r["i_wkb"] + 1)
            if r["has_num"][i] and r["has_wkb"][i]
        )
        # replicate _classify_crossover's band ladder on the stored geometry
        pts = []
        for i in range(r["i_num"], r["i_wkb"] + 1):
            if not (r["has_num"][i] and r["has_wkb"][i]):
                continue
            z = zmap[r["nodes"][i]]
            lz = math.log(1.0 + z)
            cn = (lz - ln) / ln
            cw = (lw - lz) / lw
            pts.append(
                (
                    i,
                    cn,
                    cw,
                    {
                        "numeric_good": cn > CL_GOOD,
                        "WKB_good": cw > CL_GOOD,
                        "numeric_acceptable": cn > CL_ACC,
                        "WKB_acceptable": cw > CL_ACC,
                        "numeric_marginal": cn > CL_MARG,
                        "WKB_marginal": cw > CL_MARG,
                        "numeric_minimal": cn > 0.0,
                        "WKB_minimal": cw > 0.0,
                    },
                )
            )
        rep_band, rep_i, rep_q = None, None, None
        for bi, (bq, props) in enumerate(BANDS):
            cand = [p for p in pts if all(p[3][pp] for pp in props)]
            if cand:
                rep_band, rep_q = bi, bq
                rep_i = max(p[0] for p in cand)  # maximize-WKB: the largest z
                break
        if rep_band is None:
            rep_band, rep_q, rep_i = 9, "minimal", r["i_wkb"]  # the fallback
        r["rep_band"], r["rep_quality"], r["rep_i_x"] = rep_band, rep_q, rep_i
        r["replicates"] = rep_i == r["i_x"] and rep_q == r["quality"]
        # the first band the stored crossover point itself satisfies
        own = [p for p in pts if p[0] == r["i_x"]]
        r["own_band"] = None
        if own:
            for bi, (bq, props) in enumerate(BANDS):
                if all(own[0][3][pp] for pp in props):
                    r["own_band"] = bi
                    break
        r["implied_bands"] = quality_bands[r["quality"]]
        # e-fold positions
        r["N_num"] = r["efold"](zn)
        r["N_wkb"] = r["efold"](zw)
        r["N_x"] = r["efold"](xz)
        z3, z4 = r["ex"]["z_exit_subh_e3"], r["ex"]["z_exit_subh_e4"]
        zm = math.sqrt(z3 * z4)
        r["z_e3"], r["z_e4"], r["z_mid"] = z3, z4, zm
        band_nodes = [i for i in range(r["nn"]) if z4 <= zmap[r["nodes"][i]] < zm]
        r["band_nodes"] = len(band_nodes)
        r["band_both"] = sum(
            1 for i in band_nodes if r["has_num"][i] and r["has_wkb"][i]
        )
        r["band_first_node_ok"] = bool(band_nodes) and r["i_num"] <= band_nodes[0]
        r["band_last_node_ok"] = bool(band_nodes) and r["i_wkb"] >= band_nodes[-1]

    # ---------------------------------------------------------------- 3. the deliverable
    print()
    print(
        "## 3. The deliverable: grid intervals each side of `crossover_z`, on the 72 `mixed` rows"
    )
    print()
    print(
        "`numeric_intervals` = node-index difference between `crossover_z` and `numeric_smallest_z`; "
        "`WKB_intervals` = node-index difference between `primary_WKB_largest_z` and `crossover_z`; "
        "both on the stored source grid (matched on redshift serial)."
    )
    print()
    print("<!-- section:distributions -->")
    drows = []
    for name, key in (
        ("numeric_intervals", "numeric_intervals"),
        ("WKB_intervals", "WKB_intervals"),
        ("min(numeric_intervals, WKB_intervals)", None),
        ("overlap_intervals", "overlap_intervals"),
        ("overlap_log_width (log(1+z))", "overlap_log_width"),
    ):
        if key is None:
            vals = [min(r["numeric_intervals"], r["WKB_intervals"]) for r in mixed]
        else:
            vals = [r[key] for r in mixed]
        drows.append([name, len(vals)] + dist_cells(vals))
    table(
        [
            "quantity (over all mixed rows)",
            "rows",
            "min",
            "5th",
            "median",
            "95th",
            "max",
        ],
        drows,
    )
    print()
    mixed_complete = [r for r in mixed if r["quality"] == "complete"]
    drows = []
    for name, key in (
        ("numeric_intervals", "numeric_intervals"),
        ("WKB_intervals", "WKB_intervals"),
    ):
        drows.append(
            [name + " (mixed/complete only)", len(mixed_complete)]
            + dist_cells([r[key] for r in mixed_complete])
        )
    table(["quantity", "rows", "min", "5th", "median", "95th", "max"], drows)
    print("<!-- end:distributions -->")
    print()
    at_top = [r for r in mixed if r["wl_ser"] == r["z_max_serial"]]
    xs_complete = sorted({r["crossover_z"] for r in mixed_complete})
    print(
        f"`primary_WKB_largest_z` is the top node of the source grid (the GkSource's `z_max_serial`) on {len(at_top)} of "
        f"{len(mixed)} mixed rows; the rows where it is not: "
        f"{[(r['shard'], r['serial']) for r in mixed if r not in at_top]}. "
        f"Distinct `crossover_z` values over the {len(mixed_complete)} mixed/complete rows: {len(xs_complete)}"
        + (f" ({xs_complete[0]:.7e})" if len(xs_complete) == 1 else "")
        + f"; distinct over all mixed rows: {len({r['crossover_z'] for r in mixed})}. "
        f"The grid's top node is z = {zmap[order[-1]]:.7e}; `crossover_z` / that = {xs_complete[0] / zmap[order[-1]]:.4e}."
    )
    print()
    few = [r for r in mixed if r["numeric_intervals"] < 2 or r["WKB_intervals"] < 2]
    print(
        f"**Mixed rows with fewer than two grid intervals on either side: {len(few)} of {len(mixed)}.**"
    )
    print()
    print("<!-- section:fewer -->")
    if few:
        table(
            [
                "shard",
                "serial",
                "k",
                "z_response",
                "quality",
                "crossover_z",
                "numeric_intervals",
                "WKB_intervals",
            ],
            [
                [
                    r["shard"],
                    r["serial"],
                    f"{r['k']:.6e}",
                    f"{r['z_resp']:.6e}",
                    r["quality"],
                    f"{r['crossover_z']:.7e}",
                    r["numeric_intervals"],
                    r["WKB_intervals"],
                ]
                for r in few
            ],
        )
    else:
        print(
            "None: no `mixed` row has fewer than two grid intervals on either side of `crossover_z`."
        )
    print("<!-- end:fewer -->")
    print()
    print("**The `minimal` row(s), tabulated individually whatever the counts:**")
    print()
    print("<!-- section:minimal -->")
    minimal = [r for r in rows if r["quality"] == "minimal"]
    for r in minimal:
        table(
            ["field", "value"],
            [
                [
                    "shard / policy serial / source serial",
                    f"{r['shard']} / {r['serial']} / {r['source_serial']}",
                ],
                ["type / quality", f"{r['type']} / {r['quality']}"],
                ["k (/Mpc)", f"{r['k']:.7e}"],
                ["z_response", f"{r['z_resp']:.7e}"],
                ["crossover_z", f"{r['crossover_z']:.7e}"],
                (
                    ["numeric_smallest_z", f"{r['z_num']:.7e}"]
                    if r["z_num"] is not None
                    else ["numeric_smallest_z", "-"]
                ),
                ["primary_WKB_largest_z", f"{r['z_wkb']:.7e}"],
                [
                    "numeric_intervals / WKB_intervals / overlap_intervals",
                    f"{r.get('numeric_intervals')} / {r.get('WKB_intervals')} / {r.get('overlap_intervals')}",
                ],
                ["overlap_log_width", g(r.get("overlap_log_width"), 5)],
                [
                    "c_num / c_WKB at crossover_z",
                    f"{g(r.get('c_num'), 5)} / {g(r.get('c_WKB'), 5)}",
                ],
                [
                    "numeric / WKB spline data points",
                    f"{r.get('n_num_pts')} / {r.get('n_wkb_pts')}",
                ],
                ['metadata["comment"] (verbatim)', f"`{r['metadata'].get('comment')}`"],
                [
                    "metadata (all keys)",
                    f"`{json.dumps(r['metadata'], sort_keys=True)}`",
                ],
                [
                    "N_sub-horizon of numeric_smallest_z / crossover_z / primary_WKB_largest_z",
                    " / ".join(efold_cell(*r[k]) for k in ("N_num", "N_x", "N_wkb")),
                ],
                [
                    "N_sub-horizon of z_response; z_e4 and sqrt(z_e3 z_e4) of this k",
                    f"{efold_cell(*r['efold'](r['z_resp']))}; {r['ex']['z_exit_subh_e4']:.7e} and {math.sqrt(r['ex']['z_exit_subh_e3'] * r['ex']['z_exit_subh_e4']):.7e}",
                ],
                [
                    "WKB hole above the primary run (first hole: lowest node without WKB, lowest node of the island above it; N of each)",
                    (
                        "none"
                        if r["first_hole"] is None
                        else f"z = {zmap[r['nodes'][r['first_hole'][0]]]:.7e} (N {efold_cell(*r['efold'](zmap[r['nodes'][r['first_hole'][0]]]))}) and z = {zmap[r['nodes'][r['first_hole'][1]]]:.7e} (N {efold_cell(*r['efold'](zmap[r['nodes'][r['first_hole'][1]]]))}); hole events {r['wkb_hole_events']}"
                    ),
                ],
            ],
        )
    print("<!-- end:minimal -->")

    print()
    print("### 3.1 Every `mixed` row")
    print()
    print("<!-- section:mixedrows -->")
    table(
        [
            "shard",
            "serial",
            "k (/Mpc)",
            "z_response",
            "quality",
            "crossover_z",
            "overlap_log_width",
            "overlap_intervals",
            "numeric_intervals",
            "WKB_intervals",
            "c_num",
            "c_WKB",
            "band",
        ],
        [
            [
                r["shard"],
                r["serial"],
                f"{r['k']:.4e}",
                f"{r['z_resp']:.4e}",
                r["quality"],
                f"{r['crossover_z']:.6e}",
                f"{r['overlap_log_width']:.4f}",
                r["overlap_intervals"],
                r["numeric_intervals"],
                r["WKB_intervals"],
                f"{r['c_num']:.4f}",
                f"{r['c_WKB']:.4f}",
                f"{r['rep_band']}"
                + (
                    ""
                    if r["own_band"] == r["rep_band"]
                    else f" (point: {r['own_band']})"
                ),
            ]
            for r in sorted(mixed, key=lambda r: (r["k"], r["z_resp"]))
        ],
    )
    print("<!-- end:mixedrows -->")

    # ---------------------------------------------------------------- 4. tie to 3.2
    print()
    print(
        "## 4. The tie to mechanism 3.2: the band ladder replicated on the stored geometry"
    )
    print()
    n_rep = sum(1 for r in mixed if r["replicates"])
    print(
        f"The ladder of `_classify_crossover` (overlap nodes with both a numeric and a WKB value between "
        f"`numeric_smallest_z` and `primary_WKB_largest_z`, scored by c_num and c_WKB, ten bands in order, the largest z "
        f"of the first non-empty band under `maximize-WKB`, the fallback to `primary_WKB_largest_z` and `minimal`) was re-run "
        f"on every `mixed` row from the stored value rows. It reproduces the stored `crossover_z` node **and** `quality` on "
        f"{n_rep} of {len(mixed)} rows."
    )
    print()
    bad_rep = [r for r in mixed if not r["replicates"]]
    if bad_rep:
        table(
            [
                "shard",
                "serial",
                "stored quality",
                "stored node index",
                "replicated quality",
                "replicated node index",
            ],
            [
                [
                    r["shard"],
                    r["serial"],
                    r["quality"],
                    r["i_x"],
                    r["rep_quality"],
                    r["rep_i_x"],
                ]
                for r in bad_rep
            ],
        )
        print()
    print(
        "`band` column of 3.1: the first band the ladder reaches (0 = complete, 1-3 acceptable, 4-8 marginal, 9 minimal); "
        "'(point: n)' is shown only where the chosen crossover point's own first-satisfied band differs."
    )
    bandcount = collections.Counter(r["rep_band"] for r in mixed)
    print()
    table(
        ["band reached", "quality it implies", "mixed rows"],
        [[b, band_quality[b], bandcount.get(b, 0)] for b in range(10)],
    )
    print()
    consistent = sum(1 for r in mixed if r["rep_band"] in r["implied_bands"])
    print(
        f"Stored `quality` is one of the bands' qualities implied by the band reached on {consistent} of {len(mixed)} rows."
    )
    print()
    n_recomp = sum(1 for r in rows if r["recomputed_ok"])
    print(
        f"`numeric_smallest_z` and `primary_WKB_largest_z` recomputed from the stored value rows (the lowest node with a numeric "
        f"value; the top of the WKB run contiguous from the bottom of the GkSource's grid) equal the GkSource row's stored serials "
        f"on {n_recomp} of {N} rows."
    )
    stop_check(
        n_recomp == N,
        f"the recomputed numeric_smallest_z / primary_WKB_largest_z disagree with the GkSource row on {N - n_recomp} rows",
    )
    print()
    print(
        "Spline data points the policy counted, `MIN_SPLINE_DATA_POINTS` = %d:"
        % MIN_PTS
    )
    print()
    table(
        ["quantity (mixed rows)", "min", "5th", "median", "95th", "max"],
        [
            ["numeric spline points (has_numeric, numeric_smallest_z up)"]
            + dist_cells([r["n_num_pts"] for r in mixed]),
            [
                "WKB spline points (has_WKB, bottom of the grid up to primary_WKB_largest_z)"
            ]
            + dist_cells([r["n_wkb_pts"] for r in mixed]),
        ],
    )
    print()
    print(
        f"Rows whose numeric side (numeric_smallest_z..crossover_z) has a node with no numeric value: "
        f"{sum(1 for r in mixed if r['holes_num_side'])}; whose WKB side (crossover_z..primary_WKB_largest_z) has a node with "
        f"no WKB value: {sum(1 for r in mixed if r['holes_wkb_side'])}; whose overlap holds a node lacking either value: "
        f"{sum(1 for r in mixed if r['overlap_nodes_both'] != r['overlap_intervals'] + 1)} "
        f"(overlap_intervals + 1 = nodes from numeric_smallest_z to primary_WKB_largest_z inclusive)."
    )

    # ---------------------------------------------------------------- 5. relation between the clearances and the intervals
    print()
    print("## 5. The relation between the stored clearances and the interval counts")
    print()
    print(
        "Identity (no approximation): numeric_intervals = c_num x log(1+z_num) / (mean step on the numeric side), and "
        "WKB_intervals = c_WKB x log(1+z_WKB) / (mean step on the WKB side), the mean steps being the node-to-node mean in log(1+z). "
        "So what a clearance threshold means in grid intervals is set by log(1+z) at the row and the local step, and this section "
        "measures those two."
    )
    print()
    edges = [0, 1, 2, 5, 10, 20, 50, 100, 200, 10**9]
    labels = ["0", "1", "2-4", "5-9", "10-19", "20-49", "50-99", "100-199", ">=200"]

    def binned(key, ckey, lkey, skey):
        out = []
        for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
            sub = [r for r in mixed if lo <= r[key] < hi]
            if not sub:
                out.append([lab, 0, "-", "-", "-", "-", "-"])
                continue
            qs = collections.Counter(r["quality"] for r in sub)
            out.append(
                [
                    lab,
                    len(sub),
                    g(min(r[ckey] for r in sub)),
                    g(float(np.median([r[ckey] for r in sub]))),
                    g(max(r[ckey] for r in sub)),
                    g(
                        float(
                            np.median(
                                [r[lkey] / r[skey] for r in sub if r[skey] == r[skey]]
                            )
                        )
                        if any(r[skey] == r[skey] for r in sub)
                        else None
                    ),
                    ", ".join(f"{q} {n}" for q, n in qs.items()),
                ]
            )
        return out

    print("<!-- section:relation -->")
    print(
        "**Numeric side: bins of `numeric_intervals`, and the stored `c_num` in each.**"
    )
    print()
    table(
        [
            "numeric_intervals",
            "rows",
            "c_num min",
            "c_num median",
            "c_num max",
            "median log(1+z_num)/step (steps per unit clearance)",
            "stored quality",
        ],
        binned("numeric_intervals", "c_num", "L_num", "step_num"),
    )
    print()
    print("**WKB side: bins of `WKB_intervals`, and the stored `c_WKB` in each.**")
    print()
    table(
        [
            "WKB_intervals",
            "rows",
            "c_WKB min",
            "c_WKB median",
            "c_WKB max",
            "median log(1+z_WKB)/step (steps per unit clearance)",
            "stored quality",
        ],
        binned("WKB_intervals", "c_WKB", "L_WKB", "step_WKB"),
    )
    print()
    print(
        "**What each threshold is, in grid intervals, at the rows' own log(1+z) and step** (threshold x log(1+z) / mean step of the side; "
        "rows with 0 intervals on a side have no step and are left out of that side):"
    )
    print()
    trows = []
    for tname, thr in (
        ("CLEARANCE_GOOD", CL_GOOD),
        ("CLEARANCE_ACCEPTABLE", CL_ACC),
        ("CLEARANCE_MARGINAL", CL_MARG),
    ):
        for side, lkey, skey in (
            ("numeric", "L_num", "step_num"),
            ("WKB", "L_WKB", "step_WKB"),
        ):
            vals = [thr * r[lkey] / r[skey] for r in mixed if r[skey] == r[skey]]
            trows.append([f"{tname} = {thr}", side, len(vals)] + dist_cells(vals))
    table(["threshold", "side", "rows", "min", "5th", "median", "95th", "max"], trows)
    print()
    ratio_rows = []
    for name, rs in (("mixed / complete", mixed_complete),):
        vals = [
            r["WKB_intervals"] / (CL_GOOD * r["L_WKB"] / r["step_WKB"])
            for r in rs
            if r["step_WKB"] == r["step_WKB"]
        ]
        vals2 = [
            r["numeric_intervals"] / (CL_GOOD * r["L_num"] / r["step_num"])
            for r in rs
            if r["step_num"] == r["step_num"]
        ]
        ratio_rows.append(
            [name, "WKB_intervals / (CLEARANCE_GOOD log(1+z_WKB)/step)", len(vals)]
            + dist_cells(vals, 5)
        )
        ratio_rows.append(
            [name, "numeric_intervals / (CLEARANCE_GOOD log(1+z_num)/step)", len(vals2)]
            + dist_cells(vals2, 5)
        )
    table(["rows", "ratio", "n", "min", "5th", "median", "95th", "max"], ratio_rows)
    print("<!-- end:relation -->")
    print()
    print(
        "Spearman rank correlation over the 72 mixed rows: "
        f"c_num vs numeric_intervals {spearman([r['c_num'] for r in mixed], [r['numeric_intervals'] for r in mixed]):.3f}; "
        f"c_WKB vs WKB_intervals {spearman([r['c_WKB'] for r in mixed], [r['WKB_intervals'] for r in mixed]):.3f}; "
        f"log(1+z_WKB) vs WKB_intervals {spearman([r['L_WKB'] for r in mixed], [r['WKB_intervals'] for r in mixed]):.3f}; "
        f"log(1+z_num) vs numeric_intervals {spearman([r['L_num'] for r in mixed], [r['numeric_intervals'] for r in mixed]):.3f}."
    )

    # ---------------------------------------------------------------- 6. e-folds (section 2.1 item 1)
    print()
    print("## 6. Where the overlap sits, in e-folds (prompt 2.1 item 1)")
    print()
    print(
        "Landmarks from the exit-time row each policy row names (`wavenumber_exit_serial`, checked equal to its GkSource parent's), "
        "never looked up by tolerance. N is e-folds sub-horizon, interpolated in log(1+z) through the row's stored z_exit_suph_e5 .. "
        "z_exit_subh_e6; `<-5` and `>6` mean outside that table. Prediction (README 2 (q)): the band z_e4 < z_source < sqrt(z_e3 z_e4) "
        "(4 to 3.5 e-folds, half an e-fold, the same for every k) carries both representations on every mixed row; the upper "
        "`GkWKBIntegration` branch adds above it on some."
    )
    print()
    print("<!-- section:efolds -->")
    cat = lambda n, f: (
        "<-5"
        if f == "lo"
        else (
            ">6"
            if f == "hi"
            else (
                "[-5,0)"
                if n < 0
                else (
                    "[0,3)"
                    if n < 3
                    else (
                        "[3,3.5)"
                        if n < 3.5
                        else ("[3.5,4)" if n < 4 else ("[4,6]" if n <= 6 else ">6"))
                    )
                )
            )
        )
    )
    cats = ["<-5", "[-5,0)", "[0,3)", "[3,3.5)", "[3.5,4)", "[4,6]", ">6"]
    erows = []
    for label, key in (
        ("numeric_smallest_z", "N_num"),
        ("crossover_z", "N_x"),
        ("primary_WKB_largest_z", "N_wkb"),
    ):
        c = collections.Counter(cat(*r[key]) for r in mixed)
        erows.append([label] + [c.get(x, 0) for x in cats])
    table(["of the mixed rows (72), N sub-horizon of ..."] + cats, erows)
    print()
    print(
        "Per mixed row, in order of k then z_response (N as defined above; band nodes = source-grid nodes with z_e4 <= z < sqrt(z_e3 z_e4); "
        "both = of those, nodes carrying a numeric AND a WKB value):"
    )
    print()
    table(
        [
            "shard",
            "serial",
            "k (/Mpc)",
            "z_response",
            "N(numeric_smallest_z)",
            "N(crossover_z)",
            "N(primary_WKB_largest_z)",
            "numeric_smallest_z / z_e4",
            "primary_WKB_largest_z / sqrt(z_e3 z_e4)",
            "band nodes",
            "both",
            "numeric reaches band bottom",
            "WKB reaches band top",
        ],
        [
            [
                r["shard"],
                r["serial"],
                f"{r['k']:.4e}",
                f"{r['z_resp']:.4e}",
                efold_cell(*r["N_num"]),
                efold_cell(*r["N_x"]),
                efold_cell(*r["N_wkb"]),
                f"{r['z_num'] / r['z_e4']:.4f}",
                f"{r['z_wkb'] / r['z_mid']:.4g}",
                r["band_nodes"],
                r["band_both"],
                "yes" if r["band_first_node_ok"] else "no",
                "yes" if r["band_last_node_ok"] else "no",
            ]
            for r in sorted(mixed, key=lambda r: (r["k"], r["z_resp"]))
        ],
    )
    print("<!-- end:efolds -->")
    print()
    n_band = sum(1 for r in mixed if r["band_both"] >= 1)
    n_full = sum(
        1
        for r in mixed
        if r["band_first_node_ok"]
        and r["band_last_node_ok"]
        and r["band_both"] == r["band_nodes"]
    )
    print(
        f"**The half-e-fold band holds at least one node carrying both representations on {n_band} of {len(mixed)} mixed rows.** "
        f"On {n_full} of {len(mixed)} it holds all of its nodes with both, with `numeric_smallest_z` at or below the band's first node and "
        f"`primary_WKB_largest_z` at or above its last. Band nodes per row: "
        + ", ".join(
            f"{k}: {v}"
            for k, v in sorted(
                collections.Counter(r["band_nodes"] for r in mixed).items()
            )
        )
        + "."
    )
    absent = [r for r in mixed if r["band_both"] == 0]
    print()
    print("<!-- section:bandabsent -->")
    if absent:
        print(
            f"Mixed rows with NO node carrying both representations in the band ({len(absent)}):"
        )
        print()
        table(
            [
                "shard",
                "serial",
                "k (/Mpc)",
                "z_response",
                "z_response / z_e4",
                "N(numeric_smallest_z)",
                "N(primary_WKB_largest_z)",
                "band nodes",
            ],
            [
                [
                    r["shard"],
                    r["serial"],
                    f"{r['k']:.4e}",
                    f"{r['z_resp']:.4e}",
                    f"{r['z_resp'] / r['z_e4']:.4g}",
                    efold_cell(*r["N_num"]),
                    efold_cell(*r["N_wkb"]),
                    r["band_nodes"],
                ]
                for r in absent
            ],
        )
    else:
        print(
            "None: every mixed row has at least one node carrying both representations in the band."
        )
    print("<!-- end:bandabsent -->")
    print()
    print(
        "The same count over **all** rows that have both a numeric and a WKB value somewhere (not only `mixed`):"
    )
    print()
    both_any = [
        r for r in rows if any(a and b for a, b in zip(r["has_num"], r["has_wkb"]))
    ]
    print(
        f"rows with at least one node carrying both representations: {len(both_any)} of {N}; of those, `mixed`: {sum(1 for r in both_any if r['type'] == 'mixed')}."
    )
    print()
    print(
        "By type, rows that have a node with both representations at all: "
        + str(dict(collections.Counter(r["type"] for r in both_any)))
        + "."
    )

    # ---------------------------------------------------------------- 7. the proxy (2.1 item 2)
    print()
    print("## 7. The proxy for 'z_response or k' (prompt 2.1 item 2)")
    print()
    print(
        "A3 v2 has no `incomplete` row (section 2), so the clustering test of README 2 (q) is vacuous on this store. What it holds "
        "instead is tabulated below. This is a proxy."
    )
    kdecs = sorted({decade(r["k"]) for r in rows})
    zdecs_m = sorted({decade(r["z_resp"]) for r in mixed})
    print()
    print("<!-- section:proxy -->")
    print(
        "**(a) `overlap_log_width` in log(1+z) (e-folds at these redshifts, see section 0) over k decade x z_response decade, mixed rows** "
        "(cell: rows, median [min .. max]; the lever map predicts a floor of 0.5 in every cell):"
    )
    print()
    prow = []
    for dk in kdecs:
        line = [f"1e{dk}"]
        for dz in zdecs_m:
            sub = [
                r for r in mixed if decade(r["k"]) == dk and decade(r["z_resp"]) == dz
            ]
            line.append(
                "-"
                if not sub
                else f"{len(sub)}, {np.median([r['overlap_log_width'] for r in sub]):.2f} [{min(r['overlap_log_width'] for r in sub):.2f} .. {max(r['overlap_log_width'] for r in sub):.2f}]"
            )
        prow.append(line)
    table(["k decade \\ z_response decade"] + [f"1e{d}" for d in zdecs_m], prow)
    print()
    ovw = [r["overlap_log_width"] for r in mixed]
    print(
        f"Smallest overlap width over the 72 mixed rows: {min(ovw):.4f} in log(1+z); rows below 0.5: {sum(1 for w in ovw if w < 0.5)}."
    )
    print()
    mins = [min(r["numeric_intervals"], r["WKB_intervals"]) for r in mixed]
    thr10 = float(np.percentile(mins, 10))
    low = [r for r in mixed if min(r["numeric_intervals"], r["WKB_intervals"]) <= thr10]
    print(
        f"**(b) The lowest decile of min(numeric_intervals, WKB_intervals) over the mixed rows** (value <= the 10th percentile {thr10:.4g}; "
        f"{len(low)} rows, ties included), plus the `minimal` row, tabulated individually:"
    )
    print()
    table(
        [
            "shard",
            "serial",
            "k (/Mpc)",
            "z_response",
            "quality",
            "numeric_intervals",
            "WKB_intervals",
            "min",
        ],
        [
            [
                r["shard"],
                r["serial"],
                f"{r['k']:.4e}",
                f"{r['z_resp']:.4e}",
                r["quality"],
                r["numeric_intervals"],
                r["WKB_intervals"],
                min(r["numeric_intervals"], r["WKB_intervals"]),
            ]
            for r in sorted(
                low,
                key=lambda r: (
                    min(r["numeric_intervals"], r["WKB_intervals"]),
                    r["k"],
                    r["z_resp"],
                ),
            )
        ]
        + [
            [
                r["shard"],
                r["serial"],
                f"{r['k']:.4e}",
                f"{r['z_resp']:.4e}",
                r["quality"] + " (minimal row)",
                r["numeric_intervals"],
                r["WKB_intervals"],
                min(r["numeric_intervals"], r["WKB_intervals"]),
            ]
            for r in minimal
            if r not in low
        ],
    )
    print()
    sel = low + [r for r in minimal if r not in low]
    print("By k (rows in the selection / mixed rows at that k):")
    print()
    table(
        ["k (/Mpc)", "selected", "mixed rows at this k"],
        [
            [
                f"{k:.6e}",
                sum(1 for r in sel if r["k"] == k),
                sum(1 for r in mixed if r["k"] == k),
            ]
            for k in sorted({r["k"] for r in rows})
        ],
    )
    print()
    print("By z_response decade (selected / mixed rows in the decade):")
    print()
    table(
        ["z_response decade", "selected", "mixed rows in this decade"],
        [
            [
                f"1e{d}",
                sum(1 for r in sel if decade(r["z_resp"]) == d),
                sum(1 for r in mixed if decade(r["z_resp"]) == d),
            ]
            for d in zdecs_m
        ],
    )
    print()
    print(
        "**(c) The type mix (numeric / mixed / WKB rows) over k decade x z_response decade, all 1160 rows:**"
    )
    print()
    allz = sorted({decade(r["z_resp"]) for r in rows})
    trow = []
    for dk in kdecs:
        line = [f"1e{dk}"]
        for dz in allz:
            sub = [
                r for r in rows if decade(r["k"]) == dk and decade(r["z_resp"]) == dz
            ]
            c = collections.Counter(r["type"] for r in sub)
            line.append(
                "-"
                if not sub
                else f"{c.get('numeric', 0)}/{c.get('mixed', 0)}/{c.get('WKB', 0)}"
            )
        trow.append(line)
    table(["k decade \\ z_response decade"] + [f"1e{d}" for d in allz], trow)
    print()
    print("**(d) The same, for each of the eight k individually:**")
    print()
    trow = []
    for k in sorted({r["k"] for r in rows}):
        line = [f"{k:.4e}"]
        for dz in allz:
            sub = [r for r in rows if r["k"] == k and decade(r["z_resp"]) == dz]
            c = collections.Counter(r["type"] for r in sub)
            line.append(
                "-"
                if not sub
                else f"{c.get('numeric', 0)}/{c.get('mixed', 0)}/{c.get('WKB', 0)}"
            )
        trow.append(line)
    table(["k (/Mpc) \\ z_response decade"] + [f"1e{d}" for d in allz], trow)
    print()
    print(
        "**(e) Mixed rows per k, with the extent of their z_response range, and the z_response / z_e4 and z_response / z_e3 they span:**"
    )
    print()
    mrows = []
    for k in sorted({r["k"] for r in mixed}):
        sub = [r for r in mixed if r["k"] == k]
        ex = sub[0]["ex"]
        n_all = [r["efold"](r["z_resp"])[0] for r in sub]
        n_lo, n_hi = min(n_all), max(n_all)
        in_window = [
            r
            for r in rows
            if r["k"] == k and n_lo <= r["efold"](r["z_resp"])[0] <= n_hi
        ]
        wc = collections.Counter(r["type"] for r in in_window)
        mrows.append(
            [
                f"{k:.6e}",
                len(sub),
                f"{n_lo:.3f} .. {n_hi:.3f}",
                f"{len(in_window)}: {wc.get('numeric', 0)}/{wc.get('mixed', 0)}/{wc.get('WKB', 0)}",
                f"{min(r['z_resp'] for r in sub):.3e} .. {max(r['z_resp'] for r in sub):.3e}",
                f"{min(r['z_resp'] for r in sub) / ex['z_exit_subh_e4']:.3g} .. {max(r['z_resp'] for r in sub) / ex['z_exit_subh_e4']:.3g}",
                f"{ex['z_exit_subh_e3']:.3e}",
                f"{ex['z_exit_subh_e4']:.3e}",
                f"{ex['z_exit_subh_e6']:.3e}",
                g(min(r["numeric_intervals"] for r in sub)),
                g(min(r["WKB_intervals"] for r in sub)),
            ]
        )
    table(
        [
            "k (/Mpc)",
            "mixed rows",
            "N(z_response) of the mixed rows, min .. max",
            "all rows of this k with N(z_response) in that range: count (numeric/mixed/WKB)",
            "z_response range",
            "z_response / z_e4",
            "z_e3",
            "z_e4",
            "z_e6",
            "min numeric_intervals",
            "min WKB_intervals",
        ],
        mrows,
    )
    print("<!-- end:proxy -->")

    # ---------------------------------------------------------------- 8. non-contiguity
    print()
    print("## 8. The non-contiguity count (prompt 3 item 2)")
    print()
    print(
        "`primary_WKB_largest_z` is the top of the WKB run contiguous from the bottom of the GkSource's grid (its `z_response`); "
        "`assemble_GkSource_values` warns when a WKB value follows a hole, with z_response < z_e4. Counted from the stored value rows, "
        "all 1160 GkSource rows, by the warning's own test (a node with WKB whose next-lower node has none, after a WKB node was seen):"
    )
    print()
    print("<!-- section:contiguity -->")

    def contig_row(label, sub):
        holes = [r for r in sub if r["wkb_hole_events"] > 0]
        return [
            label,
            len(sub),
            sum(1 for r in holes if r["z_lt_e4"]),
            sum(1 for r in holes if not r["z_lt_e4"]),
            sum(1 for r in sub if r["wkb_island"]),
            sum(1 for r in sub if r["no_primary"]),
            sum(r["wkb_hole_events"] for r in sub),
        ]

    table(
        [
            "rows",
            "count",
            "with a WKB hole, z_response < z_e4 (the warning's condition)",
            "with a WKB hole, z_response >= z_e4",
            "with WKB nodes above the primary run",
            "with no WKB at the bottom node (no primary region)",
            "hole events in all",
        ],
        [
            contig_row("all", rows),
            contig_row("type mixed", mixed),
            contig_row("type numeric", [r for r in rows if r["type"] == "numeric"]),
            contig_row("type WKB", [r for r in rows if r["type"] == "WKB"]),
        ],
    )
    print("<!-- end:contiguity -->")
    print()
    holes_lt = [r for r in rows if r["wkb_hole_events"] > 0 and r["z_lt_e4"]]
    holes_ge = [r for r in rows if r["wkb_hole_events"] > 0 and not r["z_lt_e4"]]
    if holes_lt or holes_ge:
        print("Rows with a hole (first 40):")
        print()
        table(
            [
                "shard",
                "serial",
                "type",
                "quality",
                "k (/Mpc)",
                "z_response",
                "N(z_response)",
                "z_response < z_e4",
                "hole events",
                "first hole: lowest node without WKB, N",
                "first hole: lowest node of the island above it, N",
                "N(primary_WKB_largest_z)",
            ],
            [
                [
                    r["shard"],
                    r["serial"],
                    r["type"],
                    r["quality"],
                    f"{r['k']:.4e}",
                    f"{r['z_resp']:.4e}",
                    efold_cell(*r["efold"](r["z_resp"])),
                    r["z_lt_e4"],
                    r["wkb_hole_events"],
                    efold_cell(*r["efold"](zmap[r["nodes"][r["first_hole"][0]]])),
                    efold_cell(*r["efold"](zmap[r["nodes"][r["first_hole"][1]]])),
                    (
                        efold_cell(*r["efold"](r["z_wkb"]))
                        if r["z_wkb"] is not None
                        else "none"
                    ),
                ]
                for r in (holes_lt + holes_ge)[:40]
            ],
        )
    # was the hole in the mixed rows above the crossover? (is the WKB side affected?)
    print()
    print(
        f"Also: rows where the stored `primary_WKB_largest_z` is not the highest node carrying a WKB value (a WKB island above the primary run): "
        f"{sum(1 for r in rows if r['wkb_island'])}; of those `mixed`: {sum(1 for r in mixed if r['wkb_island'])}."
    )

    # ---------------------------------------------------------------- 9. exit-time check
    print()
    print("## 9. The exit-time check (prompt 3 item 4)")
    print()
    print(
        f"Policy row's `wavenumber_exit_serial` equal to its GkSource parent's, every row: "
        f"{'yes, all ' + str(N) + ' rows' if not exit_disagree else 'NO on ' + str(len(exit_disagree)) + ' rows'}. "
        "The exit-time row was never looked up by tolerance."
    )
    print()
    print(
        "Producers' agreement: every integration row a GkSource was assembled from (`GkSource_parents`) names the same exit-time row as the GkSource:"
    )
    print()
    table(
        [
            "integration rows read through GkSource_parents",
            "rows",
            "of which name a different exit-time row",
        ],
        [
            [
                kind,
                parent_checks[(kind, "rows")],
                parent_checks[(kind, "exit differs from the GkSource's")],
            ]
            for kind in ("numeric", "WKB")
        ],
    )
    print()
    pairs = collections.Counter(
        (exits[r["exit_policy"]]["atol_serial"], exits[r["exit_policy"]]["rtol_serial"])
        for r in rows
    )
    print(
        "The exit-time rows' (atol serial, rtol serial) pairs over the 1160 policy rows: "
        + ", ".join(
            f"({a}, {b}) = (log10 {tolmap[a]:g}, log10 {tolmap[b]:g}): {n} rows"
            for (a, b), n in pairs.items()
        )
        + "."
    )
    stop_check(
        all(v == 0 for (kind, what), v in parent_checks.items() if what != "rows"),
        "an integration row of a GkSource names a different exit-time row from the GkSource",
    )

    # ---------------------------------------------------------------- 10. overlap on the non-mixed rows (context)
    print()
    print("## 10. The stop conditions of prompt 7 that this census can evaluate")
    print()
    frac = len(few) / len(mixed) if mixed else float("nan")
    print(
        f"- mixed rows with fewer than two grid intervals on a side: {len(few)} of {len(mixed)} ({100 * frac:.2f} %)."
    )
    print(
        f"- a mixed row's crossover_z matching no source-grid node exactly: {len(nomatch)}."
    )
    print(
        f"- policy rows and GkSource parents naming different exit-time rows: {len(exit_disagree)}."
    )
    print(
        f"- rows with WKB holes (the warning's condition): {len(holes_lt)}; with holes at z_response >= z_e4: {len(holes_ge)}."
    )
    print(
        f"- mixed rows with no node carrying both representations in the half-e-fold band: {len(absent)}."
    )
    print(
        f"- type/quality count agrees with prompt 11's table in every cell: {ok_cells}."
    )

    # ---------------------------------------------------------------- 11. every row
    print()
    print("## 11. Every GkSourcePolicyData row")
    print()
    print(
        "(`overlap_*`, `numeric_intervals`, `WKB_intervals`, `c_num`, `c_WKB` and `band` are defined for `mixed` rows only; "
        "N = e-folds sub-horizon of the row's `z_response`, `<-5`/`>6` outside the exit-time table.)"
    )
    print()
    print("<!-- section:allrows -->")
    table(
        [
            "shard",
            "serial",
            "source_serial",
            "k (/Mpc)",
            "z_response",
            "N(z_response)",
            "type",
            "quality",
            "crossover_z",
            "overlap_log_width",
            "overlap_intervals",
            "numeric_intervals",
            "WKB_intervals",
            "c_num",
            "c_WKB",
            "band",
        ],
        [
            [
                r["shard"],
                r["serial"],
                r["source_serial"],
                f"{r['k']:.6e}",
                f"{r['z_resp']:.6e}",
                efold_cell(*r["efold"](r["z_resp"])),
                r["type"],
                r["quality"],
                "-" if r["crossover_z"] is None else f"{r['crossover_z']:.7e}",
            ]
            + (
                [
                    f"{r['overlap_log_width']:.4f}",
                    r["overlap_intervals"],
                    r["numeric_intervals"],
                    r["WKB_intervals"],
                    f"{r['c_num']:.4f}",
                    f"{r['c_WKB']:.4f}",
                    r["rep_band"],
                ]
                if r["type"] == "mixed"
                else ["-"] * 7
            )
            for r in sorted(rows, key=lambda r: (r["k"], r["z_resp"]))
        ],
    )
    print("<!-- end:allrows -->")

    print()
    print("## 12. Result")
    print()
    print(f"Hard checks failed: {len(FAILED)}")
    for m in FAILED:
        print(f"- {m}")
    print(f"\nruntime {time.time() - t_start:.1f} s")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
