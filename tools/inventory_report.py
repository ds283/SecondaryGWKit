"""
Formatting for the datastore-contents inventory report (audit F2, prompt 09).

Kept separate from main.py so the formatting logic can be exercised without
going through main.py's argparse-driven top level (main.py owns the
--inventory flag and the ShardedPool construction; this module only turns
pool.inventory(...)'s output into readable text).

pool.inventory(class_name) (Datastore.SQL.ShardedPool.ShardedPool.inventory,
Datastore.SQL.Datastore.Datastore.inventory) returns one of three top-level
shapes, established by prompts 06-08 (see prompts/backport-modules/logs/06-,
07-, 08-inventory-*.md):

  - labelled buckets: {"validated": {"labels": [...], "earliest_timestamp":
    ..., "latest_timestamp": ...}, "unvalidated": {...}} -- every compute
    target with a "validated" column, sharded or replicated (BackgroundModel).
  - flat value list: {"values": [...], "earliest_timestamp": ...,
    "latest_timestamp": ...} -- small replicated configuration/grid tables.
    A class registered "timestamp": False (only "version") omits the
    timestamp keys entirely rather than returning None for them.
  - flat count: {"count": n, ...optionally "earliest_timestamp",
    "latest_timestamp", or other numeric fields...} -- high-volume value
    tables and compute targets with no "validated" column.

Each class is queried independently; a factory call that raises (a table this
pool's Datastore does not recognise, or a buggy inventory()) is reported
inline as "(error: ...)" rather than aborting the whole report.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Categories are reported in this order, and list every class this campaign's
# object factories implement inventory() for (13 replicated + 15 sharded = 28,
# per IMPLEMENTATION_STATE.md). A class with no inventory() at all (the
# tag-association factories) is deliberately not listed here -- see prompt
# 09's log for why the pool cannot reach that failure mode through this
# report's class list.
INVENTORY_CATEGORIES: List[Tuple[str, List[str]]] = [
    (
        "Datastore metadata",
        ["version", "store_tag"],
    ),
    (
        "Cosmology models",
        ["LambdaCDM", "QCD_Cosmology", "BackgroundModel", "BackgroundModelValue"],
    ),
    (
        "Grid definitions (wavenumbers, redshifts, tolerances)",
        ["wavenumber", "redshift", "wavenumber_exit_time", "tolerance"],
    ),
    (
        "Solvers and policies",
        ["IntegrationSolver", "GkSourcePolicy", "QuadSourcePolicy"],
    ),
    (
        "Compute targets",
        [
            "TkNumericIntegration",
            "TkWKBIntegration",
            "QuadSource",
            "GkNumericIntegration",
            "GkWKBIntegration",
            "GkSource",
            "GkSourcePolicyData",
            "QuadSourceIntegral",
            "OneLoopIntegral",
        ],
    ),
    (
        "Value tables",
        [
            "TkNumericValue",
            "TkWKBValue",
            "QuadSourceValue",
            "GkNumericValue",
            "GkWKBValue",
            "GkSourceValue",
        ],
    ),
]


def _format_timestamp(value: Optional[datetime]) -> str:
    if value is None:
        return "?"
    return value.strftime("%Y-%m-%d %H:%M")


def _format_range(
    earliest: Optional[datetime], latest: Optional[datetime]
) -> Optional[str]:
    if earliest is None and latest is None:
        return None
    return f"{_format_timestamp(earliest)} – {_format_timestamp(latest)}"


def _sort_key(value: Any) -> str:
    # values/labels are floats, strings, or json-safe dicts -- str() gives a
    # stable (if not always numerically "natural") ordering across all three,
    # so two runs against the same datastore produce comparable output
    return str(value)


def _format_value_lines(
    values: List[Any], verbose: bool, max_shown: int = 5
) -> List[str]:
    if not values:
        return []

    ordered = sorted(values, key=_sort_key)
    shown = ordered if verbose else ordered[:max_shown]
    lines = [f"         {v}" for v in shown]

    if not verbose and len(ordered) > max_shown:
        lines.append(f"         ... and {len(ordered) - max_shown:,} more")

    return lines


def _format_bucketed(
    class_name: str, data: Dict[str, Dict], verbose: bool
) -> List[str]:
    """
    {"validated": {"labels": [...], "earliest_timestamp": ..., "latest_timestamp":
    ...}, "unvalidated": {...}} -- shared by every compute-target class,
    sharded or replicated (BackgroundModel).
    """
    parts = []
    earliest_all = []
    latest_all = []
    detail_lines = []
    any_labels = False

    # "validated" reads more usefully first (it is usually the bucket a
    # reader cares about); any other/unexpected bucket name is appended after,
    # sorted for determinism
    preferred_order = ["validated", "unvalidated"]
    bucket_names = [b for b in preferred_order if b in data] + sorted(
        b for b in data.keys() if b not in preferred_order
    )

    for bucket_name in bucket_names:
        bucket = data[bucket_name] or {}
        labels = bucket.get("labels") or []
        if labels:
            any_labels = True
        parts.append(f"{len(labels):,} {bucket_name}")

        if bucket.get("earliest_timestamp") is not None:
            earliest_all.append(bucket["earliest_timestamp"])
        if bucket.get("latest_timestamp") is not None:
            latest_all.append(bucket["latest_timestamp"])

        if labels:
            detail_lines.append(f"         -- {bucket_name}:")
            detail_lines.extend(_format_value_lines(labels, verbose))

    header = f"      @@ {class_name}: " + ", ".join(parts)
    rng = _format_range(
        min(earliest_all) if earliest_all else None,
        max(latest_all) if latest_all else None,
    )
    if rng is not None:
        header += f" | {rng}"
    elif not any_labels:
        header += " (empty)"

    return [header] + detail_lines


def _format_value_list(
    class_name: str, data: Dict[str, Any], verbose: bool
) -> List[str]:
    """
    {"values": [...], "earliest_timestamp": ..., "latest_timestamp": ...} --
    the small replicated configuration/grid tables.
    """
    values = data.get("values") or []
    plural = "s" if len(values) != 1 else ""
    header = f"      {class_name}: {len(values):,} value{plural}"

    rng = _format_range(data.get("earliest_timestamp"), data.get("latest_timestamp"))
    if rng is not None:
        header += f" | {rng}"

    extra = []
    unit = data.get("values_unit")
    if unit is not None:
        extra.append(f"unit={unit}")
    physical = data.get("values_physical")
    if physical is not None:
        extra.append(
            f"+{len(physical):,} physical values (unit={data.get('values_physical_unit', '?')})"
        )
    if extra:
        header += " [" + ", ".join(extra) + "]"

    if not values:
        header += " (empty)"

    return [header] + _format_value_lines(values, verbose)


def _format_count(class_name: str, data: Dict[str, Any]) -> List[str]:
    """
    {"count": n, ...optional timestamps and/or other numeric fields...} -- the
    high-volume value tables and the label-free compute targets.
    """
    count = data.get("count") or 0
    plural = "s" if count != 1 else ""
    header = f"      {class_name}: {count:,} row{plural}"

    rng = _format_range(data.get("earliest_timestamp"), data.get("latest_timestamp"))
    if rng is not None:
        header += f" | {rng}"
    if count == 0:
        header += " (empty)"

    for key in sorted(data.keys()):
        if key in ("count", "earliest_timestamp", "latest_timestamp"):
            continue
        value = data[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            header += f", {key}={value:,}"
        else:
            header += f", {key}={value}"

    return [header]


def _format_entry(class_name: str, data: Any, verbose: bool) -> List[str]:
    if not data:
        return [f"      {class_name}: (empty)"]

    if isinstance(data, dict) and all(
        isinstance(v, dict) and "labels" in v for v in data.values()
    ):
        return _format_bucketed(class_name, data, verbose)

    if isinstance(data, dict) and "values" in data:
        return _format_value_list(class_name, data, verbose)

    if isinstance(data, dict) and "count" in data:
        return _format_count(class_name, data)

    # a shape none of the above recognise -- report it rather than dropping it
    return [f"      {class_name}: {data!r}"]


def format_inventory_report(pool, db_name: Any, verbose: bool = False) -> str:
    """
    Query pool.inventory() for every class this campaign's object factories
    support, grouped by category, and return the formatted report as a single
    string. A class whose inventory() call raises is reported inline as
    "(error: ...)" rather than aborting the report -- this is the first
    end-to-end exercise of the service, and a datastore predating some
    tables, or a factory bug, must not make the whole report unusable.

    pool.inventory(...) already ray.get()s internally for both replicated and
    sharded classes (see prompts/backport-modules/logs/06-inventory-plumbing.md),
    so this function never wraps the call in ray.get() itself.
    """
    lines = [f"== Datastore inventory: {db_name} =="]

    for category_title, class_names in INVENTORY_CATEGORIES:
        lines.append("")
        lines.append(f"   -- {category_title}")
        for class_name in class_names:
            try:
                data = pool.inventory(class_name)
            except Exception as e:
                lines.append(f"      {class_name}: (error: {e})")
                continue
            lines.extend(_format_entry(class_name, data, verbose))

    return "\n".join(lines)
